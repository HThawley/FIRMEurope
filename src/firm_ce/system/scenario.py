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
from firm_ce.system.tensors import StaticTensor, CostTensor
from firm_ce.fast_methods import static_m
from firm_ce.io.file_manager import DataFile
from firm_ce.io.data_model import ModelData
from firm_ce.optimisation.solver import get_solver
from firm_ce.system.parameters import ModelConfig


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

        self.lower_bounds, self.upper_bounds = self.assign_indices_and_get_bounds()
        self.x0 = self._get_x0(model_data.x0s)

        if len(self.x0) > 0:
            if (self.x0 - self.lower_bounds).min() < 0 or (self.x0 - self.upper_bounds).max() > 0:
                self.logger.info("Initial guess (x0) is out of bounds. Clipping to bounds.")
                self.x0 = np.clip(self.x0, self.lower_bounds, self.upper_bounds)

        self.statistics = None
        self.assign_unit_type_idx()
        self.get_unit_type_projection_matrix()

    def __repr__(self):
        return f"Scenario({self.id!r} {self.name!r})"

    def create_solution_directory(self) -> None:
        os.makedirs(self.solution_dir, exist_ok=True)

    def assign_indices_and_get_bounds(self) -> None:
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

    def encode_nodes(self) -> None:
        self.Nodel = [node.name for node in self.network.nodes.values()]
        self.Nodel_int = np.arange(len(self.Nodel), dtype=npintp)

    def construct_tensors(
        self,
    ):
        if not self.data_status:
            raise RuntimeError("Load datafiles before constructing tensors")

        self.encode_nodes()
        self.staticTensor = StaticTensor(self.static, self.fleet, self.network, self.asset_node_map)
        self.costTensor = CostTensor(self.staticTensor, self.fleet, self.network)

    def deconstruct_tensors(
        self,
    ):
        del self.staticTensor, self.costTensor

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

        if self.config.backend == "tensor":
            self.construct_tensors()

        if len(self.x0) == 0:
            self.x0 = self._approximate_feasible_solution()

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
            "ccgt": 0.2 / 0.3,
            "pv_fixed": 0.5 / 0.15,
            "pv_track": 0.2 / 0.15,
            "onsw": 0.4 / 0.4,
            "offw": 0.4 / 0.4,
            "biogas": 0.02 / 0.2,
            "biomass": 0.02 / 0.2,
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
            heuristic_x[idx_p] = peak * 0.2  # multiple types of storage -> at least 1x peak
            heuristic_x[idx_e] = avg / 0.25 * 24  # only applies to phes, will be clipped for fixed duration battery

        for line in self.network.major_lines.values():
            idx = line.candidate_x_idx
            peak_start, _ = node_metrics[line.node_start.id]
            peak_end, _ = node_metrics[line.node_end.id]
            max_connecting_peak = max(peak_start, peak_end)

            heuristic_x[idx] = max_connecting_peak * 0.25

        return np.clip(heuristic_x, self.lower_bounds, self.upper_bounds)

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

        n_vars = len(self.lower_bounds)
        k_dims = len(groups)

        # Build the (N, K) matrix
        projection_matrix = np.zeros((n_vars, k_dims), dtype=npfloat)

        for k, (_, indices) in enumerate(groups.items()):
            projection_matrix[indices, k] = 1.0

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
