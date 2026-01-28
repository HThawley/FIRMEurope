# type: ignore
import gc
from typing import Dict, List, Union
from re import sub
import os

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from firm_ce.common.helpers import parse_comma_separated, chain
from firm_ce.constructors.component_cons import construct_Fleet_object
from firm_ce.constructors.parameter_cons import construct_ScenarioParameters_object
from firm_ce.constructors.topology_cons import construct_Network_object
from firm_ce.constructors.traces_cons import (
    load_datafiles_to_generators,
    load_datafiles_to_reservoirs,
    load_datafiles_to_network,
    unload_data_from_generators,
    unload_data_from_reservoirs,
    unload_data_from_network,
)
from firm_ce.fast_methods import static_m
from firm_ce.io.file_manager import DataFile
from firm_ce.io.data_model import ModelData
from firm_ce.optimisation.solver import Solver
from firm_ce.system.parameters import ModelConfig
from firm_ce.system.components import Generator_InstanceType, Reservoir_InstanceType, Storage_InstanceType
from firm_ce.system.topology import Line_InstanceType


class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        self.logger, self.results_dir = model_data.logger, model_data.results_dir

        self.id = scenario_id
        self.name = self.scenario_data["scenario_name"].lower()
        self.type = self.scenario_data["type"]

        self.model_data = model_data
        self.limit_timesteps = None
        for item in self.model_data.config.values():
            if item["name"] == "limit_timesteps":
                self.limit_timesteps = int(item["value"])
            elif item["name"] == "balancing_type":
                balancing_type = str(item["value"])
        self.solution_dir = self.create_solution_directory(self.results_dir, self.name + "_" + balancing_type)
        self.scenario_data = self.model_data.scenarios[scenario_id]

        self.network = construct_Network_object(
            self.get_scenario_dicts(model_data.nodes),
            self.get_scenario_dicts(model_data.lines),
            self.scenario_data["networksteps_max"],
        )
        self.static = construct_ScenarioParameters_object(self.scenario_data, len(self.network.nodes), self.limit_timesteps)
        self.fleet = construct_Fleet_object(
            self.get_scenario_dicts(model_data.generators),
            self.get_scenario_dicts(model_data.reservoirs),
            self.get_scenario_dicts(model_data.storages),
            self.get_scenario_dicts(model_data.fuels),
            self.network.minor_lines,
            self.network.nodes,
        )

        self.x0 = self._get_x0(model_data.x0s)
        self.lower_bounds, self.upper_bounds = self.get_bounds()
        if len(self.x0) > 0:
            if (self.x0 - self.lower_bounds).min() < 0 or (self.x0 - self.upper_bounds).max() > 0:
                self.logger.info("Initial guess (x0) is out of bounds. Clipping to bounds.")
                self.x0 = np.clip(self.x0, self.lower_bounds, self.upper_bounds)

        self.statistics = None
        self.assign_x_indices()

    def __repr__(self):
        return f"Scenario({self.id!r} {self.name!r})"

    def create_solution_directory(self, result_directory: str, solution_name: str) -> str:
        safe_name = sub(r"[^a-zA-Z0-9_\-]", "_", solution_name)
        solution_dir = os.path.join(result_directory, safe_name)
        os.makedirs(solution_dir, exist_ok=True)
        return solution_dir

    def get_bounds(self) -> NDArray[np.float64]:
        def power_capacity_bounds(
            asset_list: Union[List[Generator_InstanceType], List[Reservoir_InstanceType],
                              List[Storage_InstanceType], List[Line_InstanceType]],
            build_cap_constraint: str,
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) for asset in asset_list]

        def energy_capacity_bounds(
                asset_list: Union[List[Storage_InstanceType], List[Reservoir_InstanceType]],
                build_cap_constraint: str
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) if asset.duration == 0 else 0.0 for asset in asset_list]

        generators = list(self.fleet.generators.values())
        reservoirs = list(self.fleet.reservoirs.values())
        storages = list(self.fleet.storages.values())
        lines = list(self.network.major_lines.values())

        lower_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "min_build"),
                    power_capacity_bounds(reservoirs, "min_build_p"),
                    energy_capacity_bounds(reservoirs, "min_build_e"),
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
                    power_capacity_bounds(reservoirs, "max_build_p"),
                    energy_capacity_bounds(reservoirs, "max_build_e"),
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
        load_datafiles_to_reservoirs(self.fleet, datafiles, self.limit_timesteps, yeartuple)

        static_m.set_year_energy_demand(self.static, self.network.nodes)

        return None

    def unload_datafiles(self) -> None:
        unload_data_from_network(self.network)

        unload_data_from_generators(self.fleet)
        unload_data_from_reservoirs(self.fleet)

        static_m.unset_year_energy_demand(self.static)

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

    def _get_x0(self, all_x0s: Dict[str, Dict[str, str]]) -> NDArray[np.float64]:
        """Get the initial guess corresponding to this scenario."""
        for entry in all_x0s.values():
            if entry["scenario"] == self.name:
                try:  # TODO: more elegant
                    x0_list = [float(x) for x in entry["x_0"].strip().split(",") if x.strip()]
                except AttributeError:
                    x0_list = []
                return np.array(x0_list, dtype=np.float64)
        return np.array([], dtype=np.float64)

    def assign_x_indices(self) -> None:
        x_index = 0
        for generator in self.fleet.generators.values():
            generator.candidate_x_idx = x_index
            x_index += 1
        for reservoir in self.fleet.reservoirs.values():
            reservoir.candidate_p_x_idx = x_index
            x_index += 1
        for reservoir in self.fleet.reservoirs.values():
            reservoir.candidate_e_x_idx = x_index
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
        solver = Solver(self, config)
        solver.evaluate()
        return solver.result

    def polish(self, config: ModelConfig, initial_population: NDArray[np.float64]) -> OptimizeResult:
        solver = Solver(
            self, config, True, initial_population
        )
        solver.evaluate()
        return solver.result
