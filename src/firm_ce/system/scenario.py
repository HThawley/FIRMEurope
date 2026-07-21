# type: ignore
import gc
from typing import Dict
from re import sub
import os
import numpy as np
from collections import defaultdict
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from firm_ce.common.helpers import parse_comma_separated
from firm_ce.common.typing import npfloat, npintp, nbintp, unicode_type, TypedDict
from firm_ce.constructors.component_cons import construct_Fleet_object
from firm_ce.constructors.parameter_cons import construct_ScenarioParameters_object
from firm_ce.constructors.topology_cons import construct_Network_object
from firm_ce.constructors.traces_cons import (
    load_datafiles_to_fuels,
    load_datafiles_to_generators,
    load_datafiles_to_storages,
    load_datafiles_to_network,
    unload_data_from_fuels,
    unload_data_from_generators,
    unload_data_from_storages,
    unload_data_from_network,
)
from firm_ce.system.tensor.static import StaticTensor
from firm_ce.fast_methods import static_m
from firm_ce.io.file_manager import DataFile
from firm_ce.io.data_model import ModelData
from firm_ce.optimisation.solver import get_solver
from firm_ce.system.scalar.parameters import ModelConfig


class Scenario:
    def __init__(
        self,
        model_data: ModelData,
        config: ModelConfig,
        scenario_id: int
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
            self.config.interval_aggregation
        )
        self.fleet = construct_Fleet_object(
            self.get_scenario_dicts(model_data.generators),
            self.get_scenario_dicts(model_data.storages),
            self.get_scenario_dicts(model_data.fuels),
            self.network.minor_lines,
            self.network.nodes,
        )

        self.lower_bounds_abs, self.upper_bounds_abs = self.assign_indices_and_get_abs_bounds()
        self.x0_abs = self._get_x0(model_data.x0s)

        if len(self.x0_abs) > 0:
            if (self.x0_abs - self.lower_bounds_abs).min() < 0 or (self.x0_abs - self.upper_bounds_abs).max() > 0:
                self.logger.info("Initial guess (x0) is out of bounds. Clipping to bounds.")
                self.x0_abs = np.clip(self.x0_abs, self.lower_bounds_abs, self.upper_bounds_abs)

        self.statistics = None
        self.assign_unit_type_idx()
        self.get_unit_type_projection_matrix()
        self.relative_space_constructed = False

    def __repr__(self):
        return f"Scenario({self.id!r} {self.name!r})"

    def create_solution_directory(self) -> None:
        os.makedirs(self.solution_dir, exist_ok=True)

    def assign_indices_and_get_abs_bounds(self) -> None:
        lower, upper = [], []
        x_index = 0

        # ensure unit_types are contiguous in x
        g_unit_types = defaultdict(list)
        for gen in self.fleet.generators.values():
            g_unit_types[gen.unit_type].append(gen.node.order)

        s_unit_types = defaultdict(list)
        for sto in self.fleet.storages.values():
            s_unit_types[sto.unit_type].append(sto.node.order)

        for unit_type in g_unit_types.keys():
            for gen in self.fleet.generators.values():
                if gen.unit_type != unit_type:
                    continue
                if gen.max_build > 0:
                    gen.candidate_x_idx = x_index
                    lower.append(gen.min_build)
                    upper.append(gen.max_build)
                    x_index += 1
                else:
                    gen.candidate_x_idx = -1

        for unit_type in s_unit_types.keys():
            for sto in self.fleet.storages.values():
                if sto.unit_type != unit_type:
                    continue
                if sto.max_build_p > 0:
                    sto.candidate_p_x_idx = x_index
                    lower.append(sto.min_build_p)
                    upper.append(sto.max_build_p)
                    x_index += 1
                else:
                    sto.candidate_p_x_idx = -1

        for unit_type in s_unit_types.keys():
            for sto in self.fleet.storages.values():
                if sto.unit_type != unit_type:
                    continue
                if sto.max_build_e > 0 and sto.duration == 0:
                    sto.candidate_e_x_idx = x_index
                    lower.append(sto.min_build_e)
                    upper.append(sto.max_build_e)
                    x_index += 1
                else:
                    sto.candidate_e_x_idx = -1

        for line in self.network.major_lines.values():
            if line.max_build > 0:
                line.candidate_x_idx = x_index
                lower.append(line.min_build)
                upper.append(line.max_build)
                x_index += 1
            else:
                line.candidate_x_idx = -1

        self.asset_node_map = TypedDict.empty(key_type=unicode_type, value_type=nbintp[:])
        for k, v in g_unit_types.items():
            self.asset_node_map[k] = np.array(v, dtype=npintp)
        for k, v in s_unit_types.items():
            self.asset_node_map[k] = np.array(v, dtype=npintp)

        return np.array(lower, npfloat), np.array(upper, npfloat)

    def scale_index(self, idx, denominator) -> None:
        self.lower_bounds_rel[idx] = self.lower_bounds_abs[idx] / denominator
        self.upper_bounds_rel[idx] = self.upper_bounds_abs[idx] / denominator
        self.abs_rel_scaler[idx] = denominator

    def construct_relative_space(self) -> None:
        if not self.data_status:
            raise RuntimeError("Load datafiles before constructing relative space")

        self.lower_bounds_rel = np.zeros_like(self.lower_bounds_abs)
        self.upper_bounds_rel = np.zeros_like(self.upper_bounds_abs)
        self.abs_rel_scaler = np.zeros_like(self.lower_bounds_abs)

        processed_idx = []
        for gen in self.fleet.generators.values():
            idx = gen.candidate_x_idx
            if idx != -1:
                nodal_demand = gen.node.mean_demand
                self.scale_index(idx, nodal_demand)
                processed_idx.append(idx)

        for sto in self.fleet.storages.values():
            idx_p = sto.candidate_p_x_idx
            if idx_p != -1:
                nodal_demand = sto.node.mean_demand
                self.scale_index(idx_p, nodal_demand)
                processed_idx.append(idx_p)

            idx_e = sto.candidate_e_x_idx
            if idx_e != -1:
                # new-build phes is the only unit type here
                self.scale_index(idx_e, nodal_demand)
                processed_idx.append(idx_e)

        for line in self.network.major_lines.values():
            idx = line.candidate_x_idx
            if idx != -1:
                nodal_demand = max(line.node_start.mean_demand, line.node_end.mean_demand)
                self.scale_index(idx, nodal_demand)
                processed_idx.append(idx)

        assert (set(range(len(self.lower_bounds_abs))) == set(processed_idx)
                ), "Not all indices were processed in relative space construction"

        self.relative_space_constructed = True

    def convert_x_to_rel(self, x_abs: NDArray[float]) -> NDArray[float]:
        if not self.relative_space_constructed:
            raise RuntimeError("Must have called `scenario.construct_relative_space` before running arbitrary conversion")

        x_rel = x_abs / self.abs_rel_scaler

        return x_rel

    def convert_x_to_abs(self, x_rel: NDArray[float]) -> NDArray[float]:
        if not self.relative_space_constructed:
            raise RuntimeError("Must have called `scenario.construct_relative_space` before running arbitrary conversion")

        x_abs = x_rel * self.abs_rel_scaler

        return x_abs

    def set_relative_scalers(self) -> None:
        if self.config.parameterisation == "relative":
            for gen in self.fleet.generators.values():
                idx = gen.candidate_x_idx
                if idx != -1:
                    gen.relative_scaler = self.abs_rel_scaler[idx]

            for sto in self.fleet.storages.values():
                idx_p = sto.candidate_p_x_idx
                if idx_p != -1:
                    sto.relative_scaler = self.abs_rel_scaler[idx_p]
                # energy uses the same scaler

            for line in self.network.major_lines.values():
                idx = line.candidate_x_idx
                if idx != -1:
                    line.relative_scaler = self.abs_rel_scaler[idx]

        elif self.config.parameterisation == "absolute":
            for gen in self.fleet.generators.values():
                idx = gen.candidate_x_idx
                if idx != -1:
                    gen.relative_scaler = 1.0

            for sto in self.fleet.storages.values():
                idx_p = sto.candidate_p_x_idx
                if idx_p != -1:
                    sto.relative_scaler_p = 1.0
                sto.relative_energy = False

            for line in self.network.major_lines.values():
                idx = line.candidate_x_idx
                if idx != -1:
                    line.relative_scaler = 1.0

    def set_canonical_bounds_and_x0(self) -> None:
        if self.config.parameterisation == "relative":
            self.lower_bounds, self.upper_bounds, self.x0 = self.lower_bounds_rel, self.upper_bounds_rel, self.x0_rel
        elif self.config.parameterisation == "absolute":
            self.lower_bounds, self.upper_bounds, self.x0 = self.lower_bounds_abs, self.upper_bounds_abs, self.x0_abs
        else:
            raise ValueError(f"Unknown parameterisation type: {self.config.parameterisation}")

    def encode_nodes(self) -> None:
        self.Nodel = [node.name for node in self.network.nodes.values()]
        self.Nodel_int = np.arange(len(self.Nodel), dtype=npintp)

    def construct_tensors(
        self,
    ) -> None:
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

    def deconstruct_tensors(
        self,
    ) -> None:
        del self.staticTensor

    def load_datafiles(
        self,
    ) -> None:
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
            self.config.interval_aggregation
        )

        static_m.set_year_energy_demand(self.static, self.network.nodes)
        self.data_status = True

        self.construct_relative_space()
        self.set_relative_scalers()

        if self.x0_abs.size == 0:
            self.x0_rel, self.x0_abs = self._approximate_feasible_solution()
        elif getattr(self, "x0_rel", np.array([])).size == 0:
            self.x0_rel = self.convert_x_to_rel(self.x0_abs)
        elif getattr(self, "x0_abs", np.array([])).size == 0:
            self.x0_abs = self.convert_x_to_abs(self.x0_rel)

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
        self.static = construct_ScenarioParameters_object(self.scenario_data, len(self.network.nodes))
        return None

    def get_scenario_dicts(self, imported_dict: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Extract scenario dict from model dict."""
        return {
            idx: imported_dict[idx]
            for idx in imported_dict
            if self.name in parse_comma_separated(imported_dict[idx]["scenarios"])
            or parse_comma_separated(imported_dict[idx]["scenarios"]) == ["all"]
        }

    def _get_datafiles(self, datafile_filenames_dict: Dict[str, Dict[str, str]], data_directory: str) -> Dict[str, DataFile]:
        """Filter or prepare datafiles specific to this scenario."""
        return {
            idx: DataFile(datafile_filenames_dict[idx], data_directory)
            for idx in datafile_filenames_dict
            if self.name.lower() in parse_comma_separated(datafile_filenames_dict[idx]["scenarios"])
            or parse_comma_separated(datafile_filenames_dict[idx]["scenarios"]) == ["all"]
        }

    def _get_x0(self, all_x0s: Dict[str, Dict[str, str]]) -> NDArray[npfloat]:
        """Get the initial guess corresponding to this scenario."""
        for entry in all_x0s.values():
            if entry["scenario"] == self.name:
                try:  # TODO: more elegant
                    x0_list = [float(x) for x in entry["x_0"].strip().split(",") if x.strip()]
                except AttributeError:
                    x0_list = []
                return np.array(x0_list, dtype=npfloat)
        return np.array([], npfloat)

    def _approximate_feasible_solution(self) -> NDArray[npfloat]:
        """ If no initial guess is provided, create an approximate feasible solution."""
        if not self.data_status:
            raise RuntimeError("Load datafiles before constructing approximate feasible solution")
        if not self.relative_space_constructed:
            raise RuntimeError(
                "Must have called `scenario.construct_relative_space` before constructing approximate feasible solution"
            )

        # Determine the size of the decision vector based on assigned indices
        heuristic_x_rel = np.zeros_like(self.lower_bounds_abs, dtype=npfloat)

        factors = {
            # 'name': <approx energy fraction> / <approx capacity factor>
            "ccgt": 0.05 / 0.05,
            "pv_fixed": 0.3 / 0.15,
            "pv_track": 0.2 / 0.15,
            "onsw": 0.3 / 0.35,
            "offw": 0.3 / 0.5,
            "biogas": 0.02 / 0.2,
            "biomass": 0.02 / 0.2,
            "nuclear": 0.1 / 0.9,
            "nuclear_lte": np.inf,
        }

        for gen in self.fleet.generators.values():
            idx = gen.candidate_x_idx
            unit_type = gen.unit_type
            # assignment pattern is avg / <capacity factor> * < net energy contrib.>
            if idx == -1:
                continue
            heuristic_x_rel[idx] = factors[unit_type]

        for sto in self.fleet.storages.values():
            idx_p = sto.candidate_p_x_idx
            if idx_p != -1:
                heuristic_x_rel[idx_p] = 0.33

            idx_e = sto.candidate_e_x_idx
            if idx_e != -1:
                # supply < 0.25 > of average load for < 64 > hours
                heuristic_x_rel[idx_e] = 0.25 * 64  # only applies to phes

        for line in self.network.major_lines.values():
            idx = line.candidate_x_idx

            heuristic_x_rel[idx] = 0.2

        heuristic_x_rel = np.clip(heuristic_x_rel, self.lower_bounds_rel, self.upper_bounds_rel)
        heuristic_x_abs = self.convert_x_to_abs(heuristic_x_rel)

        return heuristic_x_rel, heuristic_x_abs

    def assign_unit_type_idx(self) -> None:
        """
        Calculates exact memory requirements for operational data and assigns
        1D array offsets to each asset based on its unit type.
        """
        # Extract unique types deterministically
        gen_types = sorted(list(set(g.unit_type for g in self.fleet.generators.values())))
        sto_types = sorted(list(set(s.unit_type for s in self.fleet.storages.values())))

        line_types_set = set(line.unit_type for line in self.network.major_lines.values())
        line_types_set.update(line.unit_type for line in self.network.minor_lines.values())
        line_types = sorted(list(line_types_set))

        self.unit_type_idx = {}
        current_idx = 2  # reserve first two for lcoe and penalties

        # Generators need 2 slots (lt_generation, unit_lt_hours)
        for t in gen_types:
            self.unit_type_idx[t] = current_idx
            current_idx += 2

        # Storages need 1 slot (lt_generation)
        for t in sto_types:
            self.unit_type_idx[t] = current_idx
            current_idx += 1

        # Lines need 1 slot (lt_flows)
        for t in line_types:
            self.unit_type_idx[t] = current_idx
            current_idx += 1

        self.details_length = current_idx

        # Assign to instances
        for g in self.fleet.generators.values():
            g.unit_type_idx = self.unit_type_idx[g.unit_type]
        for s in self.fleet.storages.values():
            s.unit_type_idx = self.unit_type_idx[s.unit_type]
        for line in self.network.major_lines.values():
            line.unit_type_idx = self.unit_type_idx[line.unit_type]
        for line in self.network.minor_lines.values():
            line.unit_type_idx = self.unit_type_idx[line.unit_type]

    def get_unit_type_projection_matrix(self) -> np.ndarray:
        """
        Constructs an (N, K) projection matrix for space_scaler, mapping N decision
        variables to K aggregate unit types.
        """
        # allows safe append in a single line without checking if the key exists first
        groups = defaultdict(list)

        for gen in self.fleet.generators.values():
            if gen.candidate_x_idx != -1:
                groups[gen.unit_type].append(gen.candidate_x_idx)

        for sto in self.fleet.storages.values():
            if sto.candidate_p_x_idx != -1:
                groups[f"{sto.unit_type}_power"].append(sto.candidate_p_x_idx)
            if sto.candidate_e_x_idx != -1:
                groups[f"{sto.unit_type}_energy"].append(sto.candidate_e_x_idx)

        for line in self.network.major_lines.values():
            if line.candidate_x_idx != -1:
                groups[line.unit_type].append(line.candidate_x_idx)

        n_vars = len(self.lower_bounds_abs)
        k_dims = len(groups)

        # Build the (N, K) matrix
        projection_matrix = np.zeros((n_vars, k_dims), dtype=npfloat)

        for k, (_, indices) in enumerate(groups.items()):
            # space is the mean of each unit type
            # TODO: parameterise this choice
            projection_matrix[indices, k] = 1.0/len(indices)

        self.projection_groups = groups
        self.projection_matrix = projection_matrix

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
            from firm_ce.backend.tensor.solution import SolutionTensor, EvaluateTensor, prep_solution_for_postprocessing
            from firm_ce.constructors.tensor_to_scalar import map_tensor_to_scalar

            print(f"Building and evaluating tensor solution for scenario: '{self.name}'")
            solutionTensor = SolutionTensor(x, self.staticTensor)
            EvaluateTensor(solutionTensor)
            prep_solution_for_postprocessing(solutionTensor)
            solution = map_tensor_to_scalar(self, solutionTensor)
            solution = solution

        elif self.config.backend == "scalar":
            from firm_ce.backend.scalar.solution import Solution, evaluate

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
            solution = solution
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
            from firm_ce.backend.tensor.solution import build_eval_and_return_solutions, prep_solution_for_postprocessing
            from firm_ce.constructors.tensor_to_scalar import map_tensor_to_scalar

            print(f"Building and evaluating tensor noptima for scenario: '{self.name}'")
            solutionTensors = build_eval_and_return_solutions(xs, self.staticTensor)  # dict
            # dict -> list
            solutionTensors = [solutionTensors[j] for j in range(len(xs))]
            for sol in solutionTensors:
                prep_solution_for_postprocessing(sol)
            solutions = [map_tensor_to_scalar(self, sol) for sol in solutionTensors]

        elif self.config.backend == "scalar":
            from firm_ce.backend.scalar.solution import build_eval_and_return_solutions

            print(f"Building and evaluating scalar noptima for scenario: '{self.name}'")
            solutions = build_eval_and_return_solutions(
                xs,
                self.static,
                self.fleet,
                self.network,
                self.config.balancing_type,
                self.config.fixed_costs_threshold,
            )  # dict
            # dict -> list
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

    @staticmethod
    def identify_tech(name: str) -> str:
        """Maps asset attributes to simplified plotting categories."""
        name_lower = name.lower()
        if any(x in name_lower for x in ("solar", "pv", "fix", "sat")): return "Utility Solar"
        if "roof" in name_lower: return "Rooftop Solar"
        if any(x in name_lower for x in ("onshore", "onsw")): return "Onshore Wind"
        if any(x in name_lower for x in ("offshore", "offw")): return "Offshore Wind"
        if any(x in name_lower for x in ("hydro", "ror", "pond")): return "Hydro"
        if any(x in name_lower for x in ("nuke", "nuclear")): return "Nuclear"
        if any(x in name_lower for x in ("gas", "ccgt", "ocgt")): return "Fossil Gas"
        if "biomass" in name_lower: return "Biomass"
        if "biogas" in name_lower: return "Biogas"
        if "coal" in name_lower: return "Coal"
        if "bess" in name_lower: return "Battery"
        if "phes" in name_lower: return "PHES"
        if "geo" in name_lower: return "Geothermal"
        return "Other"

    def get_tech_index_map(self) -> dict:
        """Maps technology categories to their specific column indices in the x vector."""
        from collections import defaultdict
        tech_map = defaultdict(list)

        for gen in self.fleet.generators.values():
            tech = self.identify_tech(gen.name)
            tech_map[tech].append(gen.candidate_x_idx)

        for sto in self.fleet.storages.values():
            tech = self.identify_tech(sto.name)
            tech_map[f"{tech} Power (GW)"].append(sto.candidate_p_x_idx)
            tech_map[f"{tech} Energy (GWh)"].append(sto.candidate_e_x_idx)

        for line in self.network.major_lines.values():
            tech_map["Transmission (GW)"].append(line.candidate_x_idx)

        return dict(tech_map)

    def solve(self) -> OptimizeResult:
        self.create_solution_directory()

        solver = get_solver(self)
        solver.evaluate()
        return solver.result

    def polish(self, initial_population: NDArray[npfloat]) -> OptimizeResult:
        _polish_flag = self.config.polish_flag
        self.config.polish_flag = True
        solver = get_solver(self, initial_population)
        solver.evaluate()
        self.config.polish_flag = _polish_flag
        return solver.result
