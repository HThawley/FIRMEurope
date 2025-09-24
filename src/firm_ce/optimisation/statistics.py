import os
import re
import shutil
import time

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import PENALTY_MULTIPLIER, SAVE_POPULATION
from firm_ce.common.helpers import safe_divide
from firm_ce.fast_methods import fleet_m, generator_m, ltcosts_m, network_m, static_m
from firm_ce.io.file_manager import ResultFile
from firm_ce.optimisation.single_time import Solution
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType


class Statistics:
    def __init__(
        self,
        x_candidate: NDArray[np.float64],
        parameters_static: ScenarioParameters_InstanceType,
        fleet_static: Fleet_InstanceType,
        network_static: Network_InstanceType,
        solution_results_directory: str,
        scenario_name: str,
        balancing_type: str,
        fixed_costs_threshold: float,
        copy_callback: bool = True,
    ):
        self.solution = Solution(
            x_candidate, parameters_static, fleet_static, network_static, balancing_type, fixed_costs_threshold
        )
        start_time = time.time()
        self.solution.evaluate()
        end_time = time.time()
        print(f"Statistics solution evaluation time: {end_time - start_time:.4f} seconds")
        print(
            f"{scenario_name} LCOE: {self.solution.lcoe} [$/MWh], "
            f"Penalties: {self.solution.penalties}, "
            f"Deficit: {self.solution.penalties / PENALTY_MULTIPLIER}"
        )

        self.results_directory = self.create_solution_directory(
            solution_results_directory, scenario_name + "_" + balancing_type
        )
        self.copy_temp_files(copy_callback)
        self.result_files = None

        self.full_intervals_count = self.solution.static.block_lengths.sum()
        self.block_first_intervals, self.block_last_intervals = static_m.get_block_intervals(
            self.solution.static.block_lengths
        )

    def create_solution_directory(self, result_directory: str, solution_name: str) -> str:
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", solution_name)
        solution_dir = os.path.join(result_directory, safe_name)
        os.makedirs(solution_dir, exist_ok=True)
        return solution_dir

    def copy_temp_files(self, copy_callback: bool) -> None:
        if copy_callback:
            temp_dir = os.path.join("results", "temp")
            shutil.copy(os.path.join(temp_dir, "callback.csv"), os.path.join(self.results_directory, "callback.csv"))

            if SAVE_POPULATION:
                shutil.copy(
                    os.path.join(temp_dir, "population.csv"), os.path.join(self.results_directory, "population.csv")
                )
                shutil.copy(
                    os.path.join(temp_dir, "population_energies.csv"),
                    os.path.join(self.results_directory, "population_energies.csv"),
                )
        return None

    def expand_block_data(self, block_array: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.full_intervals_count == self.solution.static.intervals_count:
            return block_array

        expanded_array = np.zeros(self.full_intervals_count, dtype=np.float64)
        for block in range(self.solution.static.intervals_count):
            for idx in range(self.block_first_intervals[block], self.block_last_intervals[block]):
                expanded_array[idx] = block_array[block]
        return expanded_array

    def generate_result_files(self) -> None:
        self.result_files = {
            "capacities": self.generate_capacities_file(),
            "component_costs": self.generate_component_costs_file(),
            "energy_balance_ASSETS": self.generate_energy_balance_file("assets"),
            "energy_balance_NODES": self.generate_energy_balance_file("nodes"),
            "energy_balance_NETWORK": self.generate_energy_balance_file("network"),
            "levelised_costs": self.generate_levelised_costs_file(),
            "summary": self.generate_summary_file(),
            "x": self.generate_x_file(),
        }
        return None

    def write_results(self) -> None:
        if not self.solution.evaluated:
            print("WARNING: Solution must be evaluated before writing statistics.")
        for result_file in self.result_files.values():
            result_file.write()
        return None

    def get_asset_column_count(self, include_minor_lines: bool = True, include_energy_limits: bool = True) -> int:
        return (
            len(self.solution.fleet.generators)
            + 2 * len(self.solution.fleet.storages)
            + len(self.solution.network.major_lines)
            + include_minor_lines * len(self.solution.network.minor_lines)
            + include_energy_limits * fleet_m.count_generator_unit_type(self.solution.fleet, "flexible")
        )

    def generate_capacities_file(self) -> ResultFile:
        header = ["Asset Name"]
        row_labels = np.array(
            [
                "Total Capacity",
                "New Build Capacity",
                "Min Build",
                "Max Build",
            ],
            dtype=object,
        )
        data_array = np.empty(
            (4, self.get_asset_column_count(include_minor_lines=True, include_energy_limits=False)), dtype=np.float64
        )

        column_counter = 0
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + " [GW]")
            data_array[0, column_counter] = round(generator.capacity, 3)
            data_array[1, column_counter] = round(generator.new_build, 3)
            data_array[2, column_counter] = round(generator.min_build, 3)
            data_array[3, column_counter] = round(generator.max_build, 3)
            column_counter += 1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + " [GW]")
            data_array[0, column_counter] = round(storage.power_capacity, 3)
            data_array[1, column_counter] = round(storage.new_build_p, 3)
            data_array[2, column_counter] = round(storage.min_build_p, 3)
            data_array[3, column_counter] = round(storage.max_build_p, 3)
            column_counter += 1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + " [GWh]")
            data_array[0, column_counter] = round(storage.energy_capacity, 3)
            data_array[1, column_counter] = round(storage.new_build_e, 3)
            data_array[2, column_counter] = round(storage.min_build_e, 3)
            data_array[3, column_counter] = round(storage.max_build_e, 3)
            column_counter += 1

        for line in self.solution.network.major_lines.values():
            header.append(line.name + " [GW]")
            data_array[0, column_counter] = round(line.capacity, 3)
            data_array[1, column_counter] = round(line.new_build, 3)
            data_array[2, column_counter] = round(line.min_build, 3)
            data_array[3, column_counter] = round(line.max_build, 3)
            column_counter += 1

        for line in self.solution.network.minor_lines.values():
            header.append(line.name + " [GW]")
            data_array[0, column_counter] = round(line.capacity, 3)
            data_array[1, column_counter] = round(line.new_build, 3)
            data_array[2, column_counter] = round(line.min_build, 3)
            data_array[3, column_counter] = round(line.max_build, 3)
            column_counter += 1

        labelled_data_array = np.column_stack((row_labels, data_array))

        result_file = ResultFile("capacities", self.results_directory, header, labelled_data_array, decimals=None)
        return result_file

    def generate_component_costs_file(self) -> ResultFile:
        asset_count = (
            len(self.solution.fleet.generators)
            + len(self.solution.fleet.storages)
            + len(self.solution.network.major_lines)
            + len(self.solution.network.minor_lines)
        )
        header = ["Cost Type"]
        row_labels = np.array(
            ["Annualised Build Cost [$]", "Fixed O&M Cost [$]", "Variable O&M Cost [$]", "Fuel Cost [$]"], dtype=object
        )
        data_array = np.zeros((4, asset_count))

        col = 0
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name)
            data_array[0, col] = generator.lt_costs.annualised_build
            data_array[1, col] = generator.lt_costs.fom
            data_array[2, col] = generator.lt_costs.vom
            data_array[3, col] = generator.lt_costs.fuel
            col += 1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name)
            data_array[0, col] = storage.lt_costs.annualised_build
            data_array[1, col] = storage.lt_costs.fom
            data_array[2, col] = storage.lt_costs.vom
            data_array[3, col] = storage.lt_costs.fuel
            col += 1

        for line in self.solution.network.major_lines.values():
            header.append(line.name)
            data_array[0, col] = line.lt_costs.annualised_build
            data_array[1, col] = line.lt_costs.fom
            data_array[2, col] = line.lt_costs.vom
            data_array[3, col] = line.lt_costs.fuel
            col += 1

        for line in self.solution.network.minor_lines.values():
            header.append(line.name)
            data_array[0, col] = line.lt_costs.annualised_build
            data_array[1, col] = line.lt_costs.fom
            data_array[2, col] = line.lt_costs.vom
            data_array[3, col] = line.lt_costs.fuel
            col += 1

        labelled_data_array = np.column_stack((row_labels, data_array))

        result_file = ResultFile("component_costs", self.results_directory, header, labelled_data_array, decimals=None)
        return result_file

    def generate_energy_balance_file(self, aggregation_type: str) -> ResultFile:
        header = []
        match aggregation_type:
            case "assets":
                column_count = 3 * len(self.solution.network.nodes) + self.get_asset_column_count(
                    include_minor_lines=False, include_energy_limits=True
                )
            case "nodes":
                column_count = 10 * len(self.solution.network.nodes) + len(self.solution.network.major_lines)
            case "network":
                column_count = 10
        data_array = np.zeros((self.full_intervals_count, column_count), dtype=np.float64)

        column_counter = 0
        match aggregation_type:
            case "assets":
                for node in self.solution.network.nodes.values():
                    header.append(node.name + " Demand [MW]")
                    data_array[:, column_counter] = self.expand_block_data(node.data * 1000)
                    column_counter += 1

                for generator in self.solution.fleet.generators.values():
                    header.append(generator.name + " [MW]")
                    match generator.unit_type:
                        case "flexible":
                            data_array[:, column_counter] = self.expand_block_data(generator.dispatch_power * 1000)
                        case _:
                            data_array[:, column_counter] = self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                    column_counter += 1

                for storage in self.solution.fleet.storages.values():
                    header.append(storage.name + " [MW]")
                    data_array[:, column_counter] = self.expand_block_data(storage.dispatch_power * 1000)
                    column_counter += 1

                for generator in self.solution.fleet.generators.values():
                    if generator.unit_type == "flexible":
                        header.append(generator.name + " Remaining Energy [MWh]")
                        data_array[:, column_counter] = self.expand_block_data(generator.remaining_energy * 1000)
                        column_counter += 1

                for storage in self.solution.fleet.storages.values():
                    header.append(storage.name + " Stored Energy [MWh]")
                    data_array[:, column_counter] = self.expand_block_data(storage.stored_energy * 1000)
                    column_counter += 1

                for node in self.solution.network.nodes.values():
                    header.append(node.name + " Spillage [MW]")
                    data_array[:, column_counter] = self.expand_block_data(node.spillage * 1000)
                    column_counter += 1

                for node in self.solution.network.nodes.values():
                    header.append(node.name + " Deficit [MW]")
                    data_array[:, column_counter] = self.expand_block_data(node.deficits * 1000)
                    column_counter += 1

                for line in self.solution.network.major_lines.values():
                    header.append(line.name + " [MW]")
                    data_array[:, column_counter] = self.expand_block_data(line.flows * 1000)
                    column_counter += 1

            case "nodes":
                for node in self.solution.network.nodes.values():
                    header.append(node.name + " Demand [MW]")
                    data_array[:, column_counter] = self.expand_block_data(node.data * 1000)
                    column_counter += 1

                for header_item in [
                    "Solar [MW]",
                    "Wind [MW]",
                    "Baseload [MW]",
                    "Flexible Dispatch [MW]",
                    "Storage Dispatch [MW]",
                    "Flexible Remaining [MWh]",
                    "Stored Energy [MWh]",
                ]:
                    for node in self.solution.network.nodes.values():
                        header.append(node.name + f" {header_item}")
                        column_counter += 1

                for generator in self.solution.fleet.generators.values():
                    match generator.unit_type:
                        case "solar":
                            column_idx = len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                        case "wind":
                            column_idx = 2 * len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                        case "baseload":
                            column_idx = 3 * len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                        case "flexible":
                            column_idx = 4 * len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(generator.dispatch_power * 1000)

                for storage in self.solution.fleet.storages.values():
                    column_idx = 5 * len(self.solution.network.nodes) + storage.node.order
                    data_array[:, column_idx] += self.expand_block_data(storage.dispatch_power * 1000)

                for generator in self.solution.fleet.generators.values():
                    if generator.unit_type == "flexible":
                        column_idx = 6 * len(self.solution.network.nodes) + generator.node.order
                        data_array[:, column_idx] += self.expand_block_data(generator.remaining_energy * 1000)

                for storage in self.solution.fleet.storages.values():
                    column_idx = 7 * len(self.solution.network.nodes) + storage.node.order
                    data_array[:, column_idx] += self.expand_block_data(storage.stored_energy * 1000)
                for node in self.solution.network.nodes.values():
                    header.append(node.name + " Spillage [MW]")
                    data_array[:, column_counter] += self.expand_block_data(node.spillage * 1000)
                    column_counter += 1

                for node in self.solution.network.nodes.values():
                    header.append(node.name + " Deficit [MW]")
                    data_array[:, column_counter] += self.expand_block_data(node.deficits * 1000)
                    column_counter += 1

                for line in self.solution.network.major_lines.values():
                    header.append(line.name + " Flows [MW]")
                    data_array[:, column_counter] += self.expand_block_data(line.flows * 1000)
                    column_counter += 1

            case "network":
                for header_item in [
                    "Demand [MW]",
                    "Solar [MW]",
                    "Wind [MW]",
                    "Baseload [MW]",
                    "Flexible Dispatch [MW]",
                    "Storage Dispatch [MW]",
                    "Flexible Remaining [MWh]",
                    "Stored Energy [MWh]",
                    "Spillage [MW]",
                    "Deficit [MW]",
                ]:
                    header.append(header_item)
                for node in self.solution.network.nodes.values():
                    data_array[:, 0] += self.expand_block_data(node.data * 1000)
                    data_array[:, 8] += self.expand_block_data(node.spillage * 1000)
                    data_array[:, 9] += self.expand_block_data(node.deficits * 1000)

                for generator in self.solution.fleet.generators.values():
                    match generator.unit_type:
                        case "solar":
                            data_array[:, 1] += self.expand_block_data(generator.data * generator.capacity * 1000)
                        case "wind":
                            data_array[:, 2] += self.expand_block_data(generator.data * generator.capacity * 1000)
                        case "baseload":
                            data_array[:, 3] += self.expand_block_data(generator.data * generator.capacity * 1000)
                        case "flexible":
                            data_array[:, 4] += self.expand_block_data(generator.dispatch_power * 1000)
                            data_array[:, 6] += self.expand_block_data(generator.remaining_energy * 1000)

                for storage in self.solution.fleet.storages.values():
                    data_array[:, 5] += self.expand_block_data(storage.dispatch_power * 1000)
                    data_array[:, 7] += self.expand_block_data(storage.stored_energy * 1000)

        result_file = ResultFile(
            f"energy_balance_{aggregation_type.upper()}", self.results_directory, header, data_array, decimals=0
        )
        return result_file

    def generate_levelised_costs_file(self) -> ResultFile:
        header = [
            "LCOE [$/MWh]",
            "LCOG [$/MWh]",
            "LCOB [$/MWh]",
            "LCOB_storage [$/MWh]",
            "LCOB_transmission [$/MWh]",
            "LCOB_losses_spillage [$/MWh]",
        ]
        data_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        total_energy = (
            np.abs(
                sum(self.solution.static.year_energy_demand) - network_m.calculate_lt_line_losses(self.solution.network)
            )
            * 1000
        )
        total_generation = (
            sum(
                sum(generator.dispatch_power)
                for generator in self.solution.fleet.generators.values()
                if generator.unit_type == "flexible"
            )
            * self.solution.static.resolution
            * 1000
        )
        total_generation += (
            sum(
                sum(generator.data * generator.capacity)
                for generator in self.solution.fleet.generators.values()
                if generator.unit_type != "flexible"
            )
            * self.solution.static.resolution
            * 1000
        )

        # LCOE and Total Levelised Values
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + " LCOE [$/MWh]")
            if generator_m.check_unit_type(generator, "flexible"):
                data_array.append(ltcosts_m.get_total(generator.lt_costs) / total_energy)
            else:
                data_array.append(ltcosts_m.get_total(generator.lt_costs) / total_energy)
            data_array[0] += ltcosts_m.get_total(generator.lt_costs)
            data_array[1] += ltcosts_m.get_total(generator.lt_costs)

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + " LCOE [$/MWh]")
            data_array.append(ltcosts_m.get_total(storage.lt_costs) / total_energy)
            data_array[0] += ltcosts_m.get_total(storage.lt_costs)
            data_array[3] += ltcosts_m.get_total(storage.lt_costs)

        for line in self.solution.network.major_lines.values():
            header.append(line.name + " LCOE [$/MWh]")
            data_array.append(ltcosts_m.get_total(line.lt_costs) / total_energy)
            data_array[0] += ltcosts_m.get_total(line.lt_costs)
            data_array[4] += ltcosts_m.get_total(line.lt_costs)

        for line in self.solution.network.minor_lines.values():
            header.append(line.name + " LCOE [$/MWh]")
            data_array.append(ltcosts_m.get_total(line.lt_costs) / total_energy)
            data_array[0] += ltcosts_m.get_total(line.lt_costs)
            data_array[4] += ltcosts_m.get_total(line.lt_costs)

        data_array[0] /= total_energy  # LCOE
        data_array[1] /= total_generation  # LCOG
        data_array[2] = data_array[0] - data_array[1]  # LCOB
        data_array[3] /= total_energy  # LCOB_storage
        data_array[4] /= total_energy  # LCOB_transmission
        data_array[5] = data_array[2] - data_array[3] - data_array[4]  # LCOB_spillage_losses

        # LCOG, LCOS and LCOT
        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + " LCOG [$/MWh]")
            if generator_m.check_unit_type(generator, "flexible"):
                data_array.append(
                    safe_divide(
                        ltcosts_m.get_total(generator.lt_costs),
                        sum(generator.dispatch_power) * self.solution.static.resolution * 1000,
                    )
                )
            else:
                data_array.append(
                    safe_divide(
                        ltcosts_m.get_total(generator.lt_costs),
                        sum(generator.data * generator.capacity) * self.solution.static.resolution * 1000,
                    )
                )

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + " LCOS [$/MWh_discharged]")
            data_array.append(
                safe_divide(
                    ltcosts_m.get_total(storage.lt_costs),
                    sum(np.maximum(storage.dispatch_power, 0)) * self.solution.static.resolution * 1000,
                )
            )

        for line in self.solution.network.major_lines.values():
            header.append(line.name + " LCOT [$/MWh_flow]")
            data_array.append(
                safe_divide(
                    ltcosts_m.get_total(line.lt_costs), sum(np.abs(line.flows)) * self.solution.static.resolution * 1000
                )
            )

        for line in self.solution.network.minor_lines.values():
            header.append(line.name + " LCOT [$/MWh_flow]")
            data_array.append(
                safe_divide(
                    ltcosts_m.get_total(line.lt_costs), sum(np.abs(line.flows)) * self.solution.static.resolution * 1000
                )
            )

        result_file = ResultFile(
            "levelised_costs", self.results_directory, header, [np.array(data_array, dtype=np.float64)], decimals=2
        )
        return result_file

    def calculate_annual_energies(self, arr: NDArray[np.float64], decimals: int = 3) -> float:
        annual_energies_arr = np.zeros(self.solution.static.year_count + 1, dtype=np.float64)
        for year in range(self.solution.static.year_count):
            first_t, last_t = static_m.get_year_t_boundaries(self.solution.static, year)
            annual_energies_arr[year] = round(
                sum(arr[first_t : last_t + 1] * self.solution.static.interval_resolutions[first_t : last_t + 1]),
                decimals,
            )
        annual_energies_arr[-1] = round(sum(arr * self.solution.static.interval_resolutions), decimals)
        return annual_energies_arr

    def generate_summary_file(self) -> ResultFile:
        header = ["Year"]
        row_labels = np.array(
            list(range(self.solution.static.first_year, self.solution.static.final_year + 1)) + ["Total"], dtype=object
        )
        column_count = (
            3 * len(self.solution.network.nodes)
            + len(self.solution.fleet.generators)
            + len(self.solution.fleet.storages)
            + len(self.solution.network.major_lines)
        )
        data_array = np.zeros((len(row_labels), column_count), dtype=np.float64)
        col = 0

        for node in self.solution.network.nodes.values():
            header.append(node.name + " Annual Demand [GWh]")
            data_array[:, col] = self.calculate_annual_energies(node.data)
            col += 1

        for generator in self.solution.fleet.generators.values():
            header.append(generator.name + " Annual Gen [GWh]")
            if generator_m.check_unit_type(generator, "flexible"):
                data_array[:, col] = self.calculate_annual_energies(generator.dispatch_power)
            else:
                data_array[:, col] = self.calculate_annual_energies(generator.data * generator.capacity)
            col += 1

        for storage in self.solution.fleet.storages.values():
            header.append(storage.name + " Annual Discharge [GWh]")
            data_array[:, col] = self.calculate_annual_energies(np.maximum(storage.dispatch_power, 0))
            col += 1

        for node in self.solution.network.nodes.values():
            header.append(node.name + " Annual Spillage [GWh]")
            data_array[:, col] = self.calculate_annual_energies(node.spillage)
            col += 1

        for node in self.solution.network.nodes.values():
            header.append(node.name + " Annual Deficit [GWh]")
            data_array[:, col] = self.calculate_annual_energies(node.deficits)
            col += 1

        for line in self.solution.network.major_lines.values():
            header.append(line.name + " Annual Flows [GWh]")
            data_array[:, col] = self.calculate_annual_energies(np.abs(line.flows))
            col += 1

        labelled_data_array = np.column_stack((row_labels, data_array))

        result_file = ResultFile("summary", self.results_directory, header, labelled_data_array, decimals=None)
        return result_file

    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile("x", self.results_directory, [], [self.solution.x], decimals=None)
        return result_file

    def dump(self):
        residual_load_header = [node.name for node in self.solution.network.nodes.values()]
        residual_load_data = np.array(
            [node.residual_load for node in self.solution.network.nodes.values()], dtype=np.float64
        ).T
        ResultFile("residual_load", self.results_directory, residual_load_header, residual_load_data).write()
        ResultFile(
            "block_lengths",
            self.results_directory,
            ["Intervals per Block"],
            self.solution.static.block_lengths.reshape(-1, 1),
        ).write()
