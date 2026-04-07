# type: ignore
import gc
from typing import Dict, List, Union
from re import sub
import os

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from firm_ce.common.helpers import parse_comma_separated, chain
from firm_ce.common.typing import npfloat
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
from firm_ce.fast_methods import static_m
from firm_ce.io.file_manager import DataFile
from firm_ce.io.data_model import ModelData
from firm_ce.optimisation.solver import Solver
from firm_ce.system.parameters import ModelConfig
from firm_ce.system.components import Generator_InstanceType, Storage_InstanceType
from firm_ce.system.topology import Line_InstanceType


class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        self.data_status = False
        self.logger, self.results_dir = model_data.logger, model_data.results_dir

        self.model_data = model_data
        self.scenario_data = self.model_data.scenarios[scenario_id]
        self.id = scenario_id
        self.name = self.scenario_data["scenario_name"].lower()
        self.type = self.scenario_data["type"]

        self.limit_timesteps = None
        for item in self.model_data.config.values():
            if item["name"] == "limit_timesteps":
                self.limit_timesteps = int(item["value"])
            elif item["name"] == "balancing_type":
                balancing_type = str(item["value"])

        safe_name = sub(r"[^a-zA-Z0-9_\-]", "_", f"{self.name}_{balancing_type}")
        self.solution_dir = os.path.join(self.results_dir, safe_name)

        self.network = construct_Network_object(
            self.get_scenario_dicts(model_data.nodes),
            self.get_scenario_dicts(model_data.lines),
            self.scenario_data["networksteps_max"],
        )
        self.static = construct_ScenarioParameters_object(self.scenario_data, len(self.network.nodes), self.limit_timesteps)
        self.fleet = construct_Fleet_object(
            self.get_scenario_dicts(model_data.generators),
            self.get_scenario_dicts(model_data.storages),
            self.get_scenario_dicts(model_data.fuels),
            self.network.minor_lines,
            self.network.nodes,
        )

        self.lower_bounds, self.upper_bounds = self.get_bounds()
        self.x0 = self._get_x0(model_data.x0s)

        if len(self.x0) > 0:
            if (self.x0 - self.lower_bounds).min() < 0 or (self.x0 - self.upper_bounds).max() > 0:
                self.logger.info("Initial guess (x0) is out of bounds. Clipping to bounds.")
                self.x0 = np.clip(self.x0, self.lower_bounds, self.upper_bounds)

        self.statistics = None
        self.assign_x_indices()

    def __repr__(self):
        return f"Scenario({self.id!r} {self.name!r})"

    def create_solution_directory(self) -> None:
        os.makedirs(self.solution_dir, exist_ok=True)

    def get_bounds(self) -> NDArray[npfloat]:
        def power_capacity_bounds(
            asset_list: Union[List[Generator_InstanceType], List[Storage_InstanceType], List[Line_InstanceType]],
            build_cap_constraint: str,
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) for asset in asset_list]

        def energy_capacity_bounds(
                asset_list: List[Storage_InstanceType],
                build_cap_constraint: str
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) if asset.duration == 0 else 0.0 for asset in asset_list]

        generators = list(self.fleet.generators.values())
        storages = list(self.fleet.storages.values())
        lines = list(self.network.major_lines.values())

        lower_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "min_build"),
                    power_capacity_bounds(storages, "min_build_p"),
                    energy_capacity_bounds(storages, "min_build_e"),
                    power_capacity_bounds(lines, "min_build"),
                )
            )
        )

        upper_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "max_build"),
                    power_capacity_bounds(storages, "max_build_p"),
                    energy_capacity_bounds(storages, "max_build_e"),
                    power_capacity_bounds(lines, "max_build"),
                )
            )
        )

        return lower_bounds, upper_bounds

    def load_datafiles(
        self,
        datafile_filenames_dict: Dict[str, DataFile],
        data_directory: str,
    ) -> None:
        datafiles = self._get_datafiles(datafile_filenames_dict, data_directory)

        yeartuple = None

        if self.limit_timesteps is not None:
            self.logger.info(f"Slicing data to first {self.limit_timesteps} timesteps per config file.")
        else:
            firstyear = self.scenario_data.get("firstyear", "auto")
            finalyear = self.scenario_data.get("finalyear", "auto")
            yeartuple = firstyear, finalyear

        load_datafiles_to_network(self.network, datafiles, self.limit_timesteps, yeartuple)
        load_datafiles_to_generators(self.fleet, datafiles, self.static.resolution, self.limit_timesteps, yeartuple)
        load_datafiles_to_fuels(self.fleet, datafiles, yeartuple)
        load_datafiles_to_storages(self.fleet, datafiles, self.limit_timesteps, yeartuple)

        static_m.set_year_energy_demand(self.static, self.network.nodes)
        self.data_status = True

        if len(self.x0) == 0:
            self.x0 = self._approximate_feasible_solution()

        return None

    def unload_datafiles(self) -> None:
        unload_data_from_network(self.network)
        unload_data_from_generators(self.fleet)
        unload_data_from_fuels(self.fleet)
        unload_data_from_storages(self.fleet)

        static_m.unset_year_energy_demand(self.static)
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
            self.logger.warning("Datafiles not loaded. Node data is empty; heuristic may fail or return zeros.")

        # Determine the size of the decision vector based on assigned indices
        num_vars = len(self.fleet.generators) + (2 * len(self.fleet.storages)) + len(self.network.major_lines)
        heuristic_x = np.zeros(num_vars, dtype=npfloat)

        # Pre-calculate node metrics to avoid redundant array operations
        node_metrics = {}
        for node in self.network.nodes.values():
            node_metrics[node.id] = (
                np.max(node.data),  # peak
                np.mean(node.data),  # avg
            )

        factors = {
            # 'name': <approx energy fraction> / <approx capacity factor>
            "ccgt": 0.2 / 0.2,
            "pv_fixed": 0.5 / 0.15,
            "pv_track": 0.2 / 0.2,
            "onsw": 0.4 / 0.35,
            "offw": 0.4 / 0.45,
            "biogas": 0.02 / 0.4,
            "biomass": 0.02 / 0.4,
        }

        for gen in self.fleet.generators.values():
            peak, avg = node_metrics[gen.node.id]

            idx = gen.candidate_x_idx
            unit_type = gen.unit_type
            # assignment pattern is avg / <capacity factor> * < net energy contrib.>

            if unit_type in factors:
                heuristic_x[idx] = avg * factors[unit_type]

            if unit_type == "nuclear":
                if "LTE" in gen.name:
                    heuristic_x[idx] = gen.max_build
                else:
                    heuristic_x[idx] = avg / 0.9 * 0.1

        for sto in self.fleet.storages.values():
            peak, avg = node_metrics[sto.node.id]

            idx_p = sto.candidate_p_x_idx
            idx_e = sto.candidate_e_x_idx
            heuristic_x[idx_p] = peak * 0.4  # multiple types of storage -> at least 1x peak
            heuristic_x[idx_e] = avg / 0.25 * 48  # only applies to phes, will be clipped for fixed duration battery

        for line in self.network.major_lines.values():
            idx = line.candidate_x_idx
            peak_start, _ = node_metrics[line.node_start.id]
            peak_end, _ = node_metrics[line.node_end.id]
            max_connecting_peak = max(peak_start, peak_end)

            heuristic_x[idx] = max_connecting_peak * 0.25

        return np.clip(heuristic_x, self.lower_bounds, self.upper_bounds)

    def assign_x_indices(self) -> None:
        x_index = 0
        for generator in self.fleet.generators.values():
            generator.candidate_x_idx = x_index
            x_index += 1
        for storage in self.fleet.storages.values():
            storage.candidate_p_x_idx = x_index
            x_index += 1
        for storage in self.fleet.storages.values():
            storage.candidate_e_x_idx = x_index
            x_index += 1
        for line in self.network.major_lines.values():
            line.candidate_x_idx = x_index
            x_index += 1
        return None

    def solve(self, config: ModelConfig) -> OptimizeResult:
        self.create_solution_directory()

        solver = Solver(self, config)
        solver.evaluate()
        return solver.result

    def polish(self, config: ModelConfig, initial_population: NDArray[npfloat]) -> OptimizeResult:
        solver = Solver(
            self, config, True, initial_population
        )
        solver.evaluate()
        return solver.result
