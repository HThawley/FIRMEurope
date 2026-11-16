# type: ignore
import gc
import os
from re import sub
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from firm_ce.common.helpers import parse_comma_separated
from firm_ce.common.typing import npfloat, npintp
from firm_ce.io.binary_logs import recover_interrupted_run
from firm_ce.io.data_model import ModelData
from firm_ce.io.file_manager import DataFile
from firm_ce.io.traces import (
    load_datafiles_to_fuels,
    load_datafiles_to_generators,
    load_datafiles_to_network,
    load_datafiles_to_storages,
    unload_data_from_fuels,
    unload_data_from_generators,
    unload_data_from_network,
    unload_data_from_storages,
)
from firm_ce.optimisation.solver import get_solver
from firm_ce.scalar.entities.components import construct_Fleet_object
from firm_ce.scalar.entities.topology import construct_Network_object
from firm_ce.scalar.methods import static_m
from firm_ce.system.config import ModelConfig
from firm_ce.system.parameters import construct_ScenarioParameters_object
from firm_ce.system.space import (
    approximate_feasible_solution,
    assign_indices_and_get_abs_bounds,
    assign_unit_type_idx,
    build_capacity_space_scaler,
    construct_relative_space,
    get_tech_index_map,
    parse_x0,
)
from firm_ce.tensor.entities.static import StaticTensor


class Scenario:
    def __init__(
        self,
        model_data: ModelData,
        config: ModelConfig,
        scenario_id: int,
    ) -> None:
        self.data_status = False
        self.logger, self.results_dir = model_data.logger, model_data.results_dir

        self.model_data = model_data
        self.config = config
        self.scenario_data = self.model_data.scenarios[scenario_id]
        self.id = scenario_id
        self.name = self.scenario_data["scenario_name"].lower()

        safe_name = sub(r"[^a-zA-Z0-9_\-]", "_", f"{self.name}_{self.config.balancing_type}")
        self.solution_dir = os.path.join(self.results_dir, safe_name)

        self.network = construct_Network_object(
            self.get_scenario_dicts(model_data.nodes),
            self.get_scenario_dicts(model_data.lines),
            self.scenario_data["networksteps_max"],
        )
        self.static = construct_ScenarioParameters_object(
            self.scenario_data,
            len(self.network.nodes),
            self.config.limit_timesteps,
            self.config.interval_aggregation,
        )
        self.fleet = construct_Fleet_object(
            self.get_scenario_dicts(model_data.generators),
            self.get_scenario_dicts(model_data.storages),
            self.get_scenario_dicts(model_data.fuels),
            self.network.minor_lines,
            self.network.nodes,
        )

        self.lower_bounds_abs, self.upper_bounds_abs, self.asset_node_map = assign_indices_and_get_abs_bounds(
            self.fleet, self.network
        )

        self.x0_raw, self.x0_is_relative = parse_x0(self.name, model_data.x0s)
        self.x0_abs = np.array([], dtype=npfloat)
        self.x0_rel = np.array([], dtype=npfloat)

        self.statistics = None
        self.assign_unit_type_idx()
        self.relative_space_constructed = False

    def __repr__(self):
        return f"Scenario({self.id!r} {self.name!r})"

    def create_solution_directory(self) -> None:
        os.makedirs(self.solution_dir, exist_ok=True)

    def assign_indices_and_get_abs_bounds(self) -> tuple[NDArray[npfloat], NDArray[npfloat]]:
        lower, upper, asset_node_map = assign_indices_and_get_abs_bounds(self.fleet, self.network)
        self.asset_node_map = asset_node_map
        return lower, upper

    def construct_relative_space(self) -> None:
        if not self.data_status:
            raise RuntimeError("Load datafiles before constructing relative space")

        self.lower_bounds_rel, self.upper_bounds_rel, self.abs_rel_scaler = construct_relative_space(
            self.fleet,
            self.network,
            self.lower_bounds_abs,
            self.upper_bounds_abs,
        )
        self.relative_space_constructed = True

    def convert_x_to_rel(self, x_abs: NDArray[float]) -> NDArray[float]:
        if not self.relative_space_constructed:
            raise RuntimeError(
                "Must have called `scenario.construct_relative_space` before running arbitrary conversion"
            )
        return x_abs / self.abs_rel_scaler

    def convert_x_to_abs(self, x_rel: NDArray[float]) -> NDArray[float]:
        if not self.relative_space_constructed:
            raise RuntimeError(
                "Must have called `scenario.construct_relative_space` before running arbitrary conversion"
            )
        return x_rel * self.abs_rel_scaler

    def set_relative_scalers(self) -> None:
        if self.config.parameterisation == "relative":
            for node in self.network.nodes.values():
                node.relative_scaler = node.mean_demand

            for line in self.network.major_lines.values():
                idx = line.candidate_x_idx
                if idx != -1:
                    line.relative_scaler = self.abs_rel_scaler[idx]

        elif self.config.parameterisation == "absolute":
            for node in self.network.nodes.values():
                node.relative_scaler = 1.0

            for line in self.network.major_lines.values():
                idx = line.candidate_x_idx
                if idx != -1:
                    line.relative_scaler = 1.0

    def set_canonical_bounds_and_x0(self) -> None:
        if self.config.parameterisation == "relative":
            self.lower_bounds, self.upper_bounds, self.x0 = (
                self.lower_bounds_rel,
                self.upper_bounds_rel,
                self.x0_rel,
            )
        elif self.config.parameterisation == "absolute":
            self.lower_bounds, self.upper_bounds, self.x0 = (
                self.lower_bounds_abs,
                self.upper_bounds_abs,
                self.x0_abs,
            )
        else:
            raise ValueError(f"Unknown parameterisation type: {self.config.parameterisation}")

    def encode_nodes(self) -> None:
        self.Nodel = [node.name for node in self.network.nodes.values()]
        self.Nodel_int = np.arange(len(self.Nodel), dtype=npintp)

    def construct_tensors(self) -> None:
        if not self.data_status:
            raise RuntimeError("Load datafiles before constructing tensors")

        self.encode_nodes()
        self.staticTensor = StaticTensor(
            self.static,
            self.fleet,
            self.network,
            self.asset_node_map,
            self.abs_rel_scaler,
            self.config.parameterisation == "relative",
        )

    def deconstruct_tensors(self) -> None:
        if hasattr(self, "staticTensor"):
            del self.staticTensor

    def load_datafiles(self) -> None:
        datafiles = self._get_datafiles(self.model_data.datafiles, self.model_data.data_directory)

        yeartuple = None

        if self.config.limit_timesteps is not None:
            self.logger.info(f"Slicing data to first {self.config.limit_timesteps} timesteps per config file.")
        else:
            firstyear = self.scenario_data.get("firstyear", "auto")
            finalyear = self.scenario_data.get("finalyear", "auto")
            yeartuple = firstyear, finalyear

        load_datafiles_to_network(
            self.network,
            datafiles,
            self.config.limit_timesteps,
            yeartuple,
            self.config.demand_multiple,
            self.config.interval_aggregation,
        )
        load_datafiles_to_generators(
            self.fleet,
            datafiles,
            self.static.resolution,
            self.config.limit_timesteps,
            yeartuple,
            self.config.interval_aggregation,
        )
        load_datafiles_to_fuels(
            self.fleet,
            datafiles,
            yeartuple,
            self.config.interval_aggregation,
        )
        load_datafiles_to_storages(
            self.fleet,
            datafiles,
            self.config.limit_timesteps,
            yeartuple,
            self.config.interval_aggregation,
        )

        static_m.set_year_energy_demand(self.static, self.network.nodes)
        self.data_status = True

        self.construct_relative_space()
        self.projection_matrix = self.build_capacity_space_scaler()
        self.set_relative_scalers()

        if self.x0_raw.size > 0:
            if self.x0_is_relative:
                self.x0_rel = np.clip(self.x0_raw, self.lower_bounds_rel, self.upper_bounds_rel)
                self.x0_abs = self.convert_x_to_abs(self.x0_rel)
            else:
                self.x0_abs = np.clip(self.x0_raw, self.lower_bounds_abs, self.upper_bounds_abs)
                self.x0_rel = self.convert_x_to_rel(self.x0_abs)
        else:
            self.x0_rel, self.x0_abs = self._approximate_feasible_solution()

        self.set_canonical_bounds_and_x0()

        if self.config.backend == "tensor":
            self.construct_tensors()

        return None

    def unload_datafiles(self) -> None:
        unload_data_from_network(self.network)
        unload_data_from_generators(self.fleet)
        unload_data_from_fuels(self.fleet)
        unload_data_from_storages(self.fleet)

        static_m.unset_year_energy_demand(self.static)

        self.deconstruct_tensors()
        self.data_status = False

        gc.collect()

        return None

    def reset_static(self) -> None:
        self.static = construct_ScenarioParameters_object(
            self.scenario_data,
            len(self.network.nodes),
            self.config.limit_timesteps,
            self.config.interval_aggregation,
        )
        return None

    def get_scenario_dicts(self, imported_dict: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Extract scenario dict from model dict."""
        return {
            idx: imported_dict[idx]
            for idx in imported_dict
            if self.name in parse_comma_separated(imported_dict[idx]["scenarios"])
            or parse_comma_separated(imported_dict[idx]["scenarios"]) == ["all"]
        }

    def _get_datafiles(
        self, datafile_filenames_dict: Dict[str, Dict[str, str]], data_directory: str
    ) -> Dict[str, DataFile]:
        """Filter or prepare datafiles specific to this scenario."""
        return {
            idx: DataFile(datafile_filenames_dict[idx], data_directory)
            for idx in datafile_filenames_dict
            if self.name.lower() in parse_comma_separated(datafile_filenames_dict[idx]["scenarios"])
            or parse_comma_separated(datafile_filenames_dict[idx]["scenarios"]) == ["all"]
        }

    def _approximate_feasible_solution(self) -> tuple[NDArray[npfloat], NDArray[npfloat]]:
        if not self.data_status:
            raise RuntimeError("Load datafiles before constructing approximate feasible solution")
        if not self.relative_space_constructed:
            raise RuntimeError(
                "Must have called `scenario.construct_relative_space` before constructing approximate feasible solution"
            )
        return approximate_feasible_solution(
            self.fleet,
            self.network,
            self.lower_bounds_abs,
            self.lower_bounds_rel,
            self.upper_bounds_rel,
            self.abs_rel_scaler,
        )

    def assign_unit_type_idx(self) -> None:
        backend = getattr(self.config, "backend", "scalar")
        self.unit_type_idx, self.details_length = assign_unit_type_idx(self.fleet, self.network, backend)

    def get_nodal_demand_totals(self) -> np.ndarray:
        """
        Returns 1D array of total horizon energy demand (in model units, e.g. GWh)
        for each node in canonical node order.
        """
        if not self.data_status:
            raise RuntimeError("Load datafiles before querying nodal demand totals.")
        n_nodes = len(self.network.nodes)
        totals = np.zeros(n_nodes, dtype=npfloat)
        hours = self.static.intervals_count * self.static.resolution
        for node in self.network.nodes.values():
            totals[node.order] = npfloat(node.mean_demand * hours)
        return totals

    def build_capacity_space_scaler(
        self,
        space_norm: str | None = None,
        space_agg: str | None = None,
    ) -> np.ndarray:
        return build_capacity_space_scaler(self, space_norm, space_agg)

    def inspect_mhmga_recombination(
        self,
        starting_population: np.ndarray,
        objectives: np.ndarray = None,
        constraints: np.ndarray = None,
        evaluate_offspring: bool = True,
        **hyperparameters,
    ) -> dict:
        solver = get_solver(self)
        return solver.inspect_mhmga_recombination(
            starting_population,
            objectives,
            constraints,
            evaluate_offspring,
            **hyperparameters,
        )

    def build_and_evaluate_solution(self, x, retain=True):
        if not self.data_status:
            raise RuntimeError("Load data first.")
        x = x.astype(npfloat)

        if self.config.backend == "tensor":
            from firm_ce.tensor.solution import EvaluateTensor, SolutionTensor, prep_solution_for_postprocessing
            from firm_ce.tensor.to_scalar import map_tensor_to_scalar

            print(f"Building and evaluating tensor solution for scenario: '{self.name}'")
            solutionTensor = SolutionTensor(x, self.staticTensor)
            EvaluateTensor(solutionTensor)
            prep_solution_for_postprocessing(solutionTensor)
            solution = map_tensor_to_scalar(self, solutionTensor)

        elif self.config.backend == "scalar":
            from firm_ce.scalar.solution import Solution, evaluate

            print(f"Building and evaluating scalar solution for scenario: '{self.name}'")
            solution = Solution(
                x,
                self.static,
                self.fleet,
                self.network,
                self.config.balancing_type,
                self.config.fixed_costs_threshold,
            )
            evaluate(solution)
            solutionTensor = None
        else:
            raise ValueError(f"Unknown config.backend. Got: '{self.config.backend}'")
        if retain:
            self.solution = solution
            self.solutionTensor = solutionTensor
        return solution, solutionTensor

    def build_and_evaluate_noptima(self, xs, retain=True):
        if not self.data_status:
            raise RuntimeError("Load data first.")
        xs = xs.astype(npfloat)

        if self.config.backend == "tensor":
            from firm_ce.tensor.solution import build_eval_and_return_solutions, prep_solution_for_postprocessing
            from firm_ce.tensor.to_scalar import map_tensor_to_scalar

            print(f"Building and evaluating tensor noptima for scenario: '{self.name}'")
            solutionTensors = build_eval_and_return_solutions(xs, self.staticTensor)
            solutionTensors = [solutionTensors[j] for j in range(len(xs))]
            for sol in solutionTensors:
                prep_solution_for_postprocessing(sol)
            solutions = [map_tensor_to_scalar(self, sol) for sol in solutionTensors]

        elif self.config.backend == "scalar":
            from firm_ce.scalar.solution import build_eval_and_return_solutions

            print(f"Building and evaluating scalar noptima for scenario: '{self.name}'")
            solutions = build_eval_and_return_solutions(
                xs,
                self.static,
                self.fleet,
                self.network,
                self.config.balancing_type,
                self.config.fixed_costs_threshold,
            )
            solutions = [solutions[j] for j in range(len(xs))]
            solutionTensors = None
        else:
            raise ValueError(f"Unknown config.backend. Got: '{self.config.backend}'")
        if retain:
            self.noptima = solutions
            self.noptimaTensors = solutionTensors

            self.solution = solutions[0]
            self.solutionTensor = solutionTensors[0] if solutionTensors is not None else None
        return solutions, solutionTensors

    def get_tech_index_map(self) -> dict:
        return get_tech_index_map(self.fleet, self.network)

    def solve(self) -> OptimizeResult:
        self.create_solution_directory()

        solver = get_solver(self)
        solver.evaluate()
        return solver.result

    def recover_logs(self) -> NDArray[npfloat] | None:
        """Cleans up binary logs and converts them to CSV after an interrupted run."""
        return recover_interrupted_run(self.solution_dir)

    def polish(self, initial_population: NDArray[npfloat]) -> OptimizeResult:
        _polish_flag = self.config.polish_flag
        self.config.polish_flag = True
        solver = get_solver(self, initial_population)
        solver.evaluate()
        self.config.polish_flag = _polish_flag
        return solver.result
