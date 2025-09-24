import gc
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from firm_ce.common.helpers import parse_comma_separated
from firm_ce.constructors.component_cons import construct_Fleet_object
from firm_ce.constructors.parameter_cons import construct_ScenarioParameters_object
from firm_ce.constructors.topology_cons import construct_Network_object
from firm_ce.constructors.traces_cons import (
    load_datafiles_to_generators,
    load_datafiles_to_network,
    unload_data_from_generators,
    unload_data_from_network,
)
from firm_ce.fast_methods import static_m
from firm_ce.io.file_manager import DataFile
from firm_ce.io.validate import ModelData
from firm_ce.optimisation.solver import Solver
from firm_ce.system.parameters import ModelConfig


class Scenario:
    def __init__(self, model_data: ModelData, scenario_id: int) -> None:
        self.logger, self.results_dir = model_data.logger, model_data.results_dir

        self.scenario_data = model_data.scenarios.get(scenario_id)

        self.id = scenario_id
        self.name = self.scenario_data.get("scenario_name", "")
        self.type = self.scenario_data.get("type", "")
        self.x0 = self._get_x0(model_data.x0s)

        self.network = construct_Network_object(
            self.scenario_data.get("nodes", "").split(","),
            self._get_scenario_dicts(model_data.lines),
            self.scenario_data.get("networksteps_max", 0),
        )
        self.static = construct_ScenarioParameters_object(self.scenario_data, len(self.network.nodes))
        self.fleet = construct_Fleet_object(
            self._get_scenario_dicts(model_data.generators),
            self._get_scenario_dicts(model_data.storages),
            self._get_scenario_dicts(model_data.fuels),
            self.network.minor_lines,
            self.network.nodes,
        )
        self.statistics = None

        self.assign_x_indices()

    def __repr__(self):
        return f"Scenario({self.id!r} {self.name!r})"

    def load_datafiles(self, all_datafiles: Dict[str, DataFile], data_directory: str) -> None:
        datafiles = self._get_datafiles(all_datafiles, data_directory)

        load_datafiles_to_network(self.network, datafiles)

        load_datafiles_to_generators(self.fleet, datafiles, self.static.resolution)

        static_m.set_year_energy_demand(self.static, self.network.nodes)

        return None

    def unload_datafiles(self) -> None:
        unload_data_from_network(self.network)

        unload_data_from_generators(self.fleet)

        static_m.unset_year_energy_demand(self.static)

        gc.collect()

        return None

    def reset_static(self) -> None:
        self.static = construct_ScenarioParameters_object(self.scenario_data, len(self.network.nodes))
        return None

    def _get_scenario_dicts(self, imported_dict: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Extract scenario dict from model dict."""
        return {
            idx: imported_dict[idx]
            for idx in imported_dict
            if self.name in parse_comma_separated(imported_dict[idx]["scenarios"])
        }

    def _get_datafiles(self, all_datafiles: Dict[str, Dict[str, str]], data_directory: str) -> Dict[str, DataFile]:
        """Filter or prepare datafiles specific to this scenario."""
        return {
            idx: DataFile(all_datafiles[idx]["filename"], all_datafiles[idx]["datafile_type"], data_directory)
            for idx in all_datafiles
            if self.name in parse_comma_separated(all_datafiles[idx]["scenarios"])
        }

    def _get_x0(self, all_x0s: Dict[str, Dict[str, str]]) -> NDArray[np.float64]:
        """Get the initial guess corresponding to this scenario."""
        for entry in all_x0s.values():
            if entry.get("scenario") == self.name:
                if isinstance(entry.get("x_0", ""), float) and np.isnan(entry.get("x_0", "")):
                    x0_list = []
                else:
                    x0_str = entry.get("x_0", "").strip()
                    x0_list = [float(x) for x in x0_str.split(",") if x.strip()]
                return np.array(x0_list, dtype=np.float64)
        return None

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
        solver = Solver(config, self.x0, self.static, self.fleet, self.network, self.logger, self.name)
        solver.evaluate()
        return solver.result

    def polish(self, config: ModelConfig, initial_population: NDArray[np.float64]) -> OptimizeResult:
        solver = Solver(
            config, self.x0, self.static, self.fleet, self.network, self.logger, self.name, True, initial_population
        )
        solver.evaluate()
        return solver.result
