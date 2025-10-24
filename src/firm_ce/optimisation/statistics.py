# type: ignore
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Callable, Any

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.common.helpers import safe_divide, safe_divide_array
from firm_ce.fast_methods import ltcosts_m, network_m, static_m
from firm_ce.io.file_manager import ResultFile
from firm_ce.optimisation.single_time import Solution
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType


def prod(args):
    retval = 1
    for arg in args:
        retval *= arg
    return retval


def is_any(asset: Any) -> bool:
    return True


def is_flexible(asset: Any) -> bool:
    return asset.unit_type == "flexible"


def is_solar(asset: Any) -> bool:
    return asset.unit_type == "solar"


def is_wind(asset: Any) -> bool:
    return asset.unit_type == "wind"


def is_baseload(asset: Any) -> bool:
    return asset.unit_type == "baseload"


def is_not_flexible(asset: Any) -> bool:
    return asset.unit_type != "flexible"


asset_containers = {
    "generators": "fleet",
    "reservoirs": "fleet",
    "storages": "fleet",
    "major_lines": "network",
    "minor_lines": "network",
    "nodes": "network",
}


asset_class_to_display = {
    "generators": "Generator",
    "reservoirs": "Reservoir",
    "storages": "Storage",
    "major_lines": "Major Line",
    "minor_lines": "Minor Line",
    "nodes": "Node",
}


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
        print(f"{scenario_name} LCOE: {self.solution.lcoe} [$/MWh], " f"Penalties: {self.solution.penalties}")

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
                    os.path.join(temp_dir, "latest_population.csv"),
                    os.path.join(self.results_directory, "latest_population.csv"),
                )
                shutil.copy(
                    os.path.join(temp_dir, "population.csv"), os.path.join(self.results_directory, "population.csv")
                )
                shutil.copy(
                    os.path.join(temp_dir, "population_energies.csv"),
                    os.path.join(self.results_directory, "population_energies.csv"),
                )
        return None

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

    def generate_capacities_file(self) -> ResultFile:
        """Generates the capacities CSV"""

        def append_asset(
            df: pd.DataFrame, asset_class: str, attribute: str, affix: bool
        ) -> pd.DataFrame:
            """Add all assets in an asset class (generators, reservoirs, ...) to the capacities DataFrame"""
            if attribute.lower() == "power":
                column_name, column_units, suffix_str = "Power Capacity", "[GW]", "_p"
            elif attribute.lower() == "energy":
                column_name, column_units, suffix_str = "Energy Capacity", "[GWh]", "_e"
            else:
                raise ValueError(f"'attribute should be 'energy' or 'power'. Got '{attribute}'.")

            if affix:
                capacity_attr = f"{attribute.lower()}_capacity"
                new_build_attr = f"new_build{suffix_str}"
                min_build_attr = f"min_build{suffix_str}"
                max_build_attr = f"max_build{suffix_str}"
            else:
                capacity_attr = "capacity"
                new_build_attr = "new_build"
                min_build_attr = "min_build"
                max_build_attr = "max_build"

            df = pd.concat((
                df,
                pd.concat((
                    pd.Series([
                        asset.name,
                        asset_class_to_display[asset_class],
                        asset.id,
                        column_name,
                        column_units,
                        round(getattr(asset, capacity_attr), 3),
                        round(getattr(asset, new_build_attr), 3),
                        round(getattr(asset, min_build_attr), 3),
                        round(getattr(asset, max_build_attr), 3),
                    ], index=df.index)
                    for asset in getattr(getattr(self.solution, asset_containers[asset_class]),
                                         asset_class).values()
                ), axis=1),
            ), axis=1)
            return df

        df = pd.DataFrame(
            index=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Column Units",
                   "Total Capacity", "New Build Capacity", "Min Build", "Max Build"]
        )
        df = append_asset(df, "generators", "power", False)
        df = append_asset(df, "reservoirs", "power", True)
        df = append_asset(df, "reservoirs", "energy", True)
        df = append_asset(df, "storages", "power", True)
        df = append_asset(df, "storages", "energy", True)
        df = append_asset(df, "major_lines", "power", False)
        df = append_asset(df, "minor_lines", "power", False)

        result_file = ResultFile("capacities", self.results_directory, df)
        return result_file

    def generate_component_costs_file(self) -> ResultFile:
        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
        ) -> pd.DataFrame:
            """Add all assets of an asset class to the DataFrame"""
            df = pd.concat((
                df,
                pd.concat((
                    pd.Series([
                        asset.name,
                        asset_class_to_display[asset_class],
                        asset.id,
                        "Total Cost",
                        "[$]",
                        round(asset.lt_costs.annualised_build, 3),
                        round(asset.lt_costs.fom, 3),
                        round(asset.lt_costs.vom, 3),
                        round(asset.lt_costs.fuel, 3),
                    ], index=df.index)
                    for asset in getattr(getattr(self.solution, asset_containers[asset_class]),
                                         asset_class).values()
                ), axis=1),
            ), axis=1)
            return df

        df = pd.DataFrame(
            index=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Column Units",
                   "Annualised Build", "Fixed O&M", "Variable O&M", "Fuel"]
        )
        df = append_asset(df, "generators")
        df = append_asset(df, "reservoirs")
        df = append_asset(df, "storages")
        df = append_asset(df, "major_lines")
        df = append_asset(df, "minor_lines")

        result_file = ResultFile("component_costs", self.results_directory, df)
        return result_file

    def generate_energy_balance_file(self, aggregation_type: str) -> List[ResultFile]:
        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
            column_name: str,
            column_units: str,
            time_series_getter: Callable,
            condition: Callable = is_any,
        ) -> pd.DataFrame:
            """Add a time series feature (power trace, etc.) of all assets of an asset class
            to the DataFrame"""
            assets = [
                asset
                for asset in getattr(getattr(self.solution, asset_containers[asset_class]), asset_class).values()
                if condition(asset)
            ]
            if len(assets) > 0:
                df_to_join = pd.concat((
                    pd.concat((
                        pd.Series([asset.name, asset_class_to_display[asset_class], asset.id, column_name, column_units]),
                        pd.Series(time_series_getter(asset)),
                    ), ignore_index=True)
                    for asset in assets
                ), axis=1)
                df_to_join.index = df.index
                df = pd.concat((df, df_to_join), axis=1)
            return df

        def append_node(
            df: pd.DataFrame,
            asset_class: str,
            column_name: str,
            column_units: str,
            time_series_getter: Callable,
            condition: Callable = is_any,
        ) -> pd.DataFrame:
            """Add a time series feature (power trace, etc.) of all aggregated assets of an asset class
            in a node to the DataFrame"""
            for node in self.solution.network.nodes.values():
                assets = [
                    asset
                    for asset in getattr(getattr(self.solution, asset_containers[asset_class]), asset_class).values()
                    if condition(asset) and asset.node.id == node.id
                ]
                if len(assets) > 0:
                    df_to_join = pd.concat(
                        (
                            pd.Series([node.name, "Node", node.id, column_name, column_units]),
                            pd.Series(sum((time_series_getter(asset) for asset in assets))),
                        ),
                        ignore_index=True,
                    )
                    df_to_join.index = df.index
                    df = pd.concat((df, df_to_join), axis=1)
            return df

        def append_network(
            df: pd.DataFrame,
            asset_class: str,
            column_name: str,
            column_units: str,
            time_series_getter: Callable,
            condition: Callable = is_any,
        ) -> pd.DataFrame:
            """Add a time series feature (power trace, etc.) of all aggregated assets of an asset class
            in a node to the DataFrame"""
            assets = [
                asset
                for asset in getattr(getattr(self.solution, asset_containers[asset_class]), asset_class).values()
                if condition(asset)
            ]
            if len(assets) > 0:
                df_to_join = pd.concat((
                    pd.Series(["Network", "Network", 0, column_name, column_units]),
                    pd.Series(sum((time_series_getter(asset) for asset in assets))),
                ), ignore_index=True)
                df_to_join.index = df.index
                df = pd.concat((df, df_to_join), axis=1)
            return df

        def get_data_power(asset) -> NDArray[np.float64]:
            return asset.data * 1000.0  # MW

        def get_spillage_power(asset) -> NDArray[np.float64]:
            return asset.spillage * 1000.0  # MW

        def get_deficit_power(asset) -> NDArray[np.float64]:
            return asset.deficits * 1000.0  # MW

        def get_dispatched_power(asset) -> NDArray[np.float64]:
            return asset.dispatch_power * 1000.0  # MW

        def get_inflexible_power(asset) -> NDArray[np.float64]:
            return asset.data * asset.capacity * 1000.0  # MW

        def get_flow_power(asset) -> NDArray[np.float64]:
            return asset.flows * 1000.0  # MW

        def get_stored_energy(asset) -> NDArray[np.float64]:
            return asset.stored_energy * 1000.0  # MWh

        def get_remaining_energy(asset) -> NDArray[np.float64]:
            return asset.remaining_energy * 1000.0  # MWh

        df = pd.concat((
            pd.DataFrame(index=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Column Units"]),
            pd.DataFrame(index=pd.RangeIndex(self.full_intervals_count))))

        match aggregation_type:
            case "assets":
                df = append_asset(df, "nodes", "Demand", "[MW]", get_data_power)
                df = append_asset(df, "generators", "Dispatch", "[MW]", get_inflexible_power, condition=is_not_flexible)
                df = append_asset(df, "generators", "Flexible Dispatch", "[MW]", get_dispatched_power, condition=is_flexible)
                df = append_asset(df, "reservoirs", "Reservoir Dispatch", "[MW]", get_dispatched_power)
                df = append_asset(df, "storages", "Storage Dispatch", "[MW]", get_dispatched_power)
                df = append_asset(df, "generators", "Flexible Remaining", "[MWh]", get_remaining_energy, condition=is_flexible)
                df = append_asset(df, "reservoirs", "Reservoir Energy", "[MWh]", get_stored_energy)
                df = append_asset(df, "storages", "Stored Energy", "[MWh]", get_stored_energy)
                df = append_asset(df, "reservoirs", "Inflow", "[MWh]", get_data_power)
                df = append_asset(df, "nodes", "Spillage", "[MW]", get_spillage_power)
                df = append_asset(df, "nodes", "Deficit", "[MW]", get_deficit_power)
                df = append_asset(df, "major_lines", "Flow", "[MW]", get_flow_power)

            case "nodes":
                df = append_asset(df, "nodes", "Demand", "[MW]", get_data_power)
                df = append_node(df, "generators", "Solar", "[MW]", get_inflexible_power, condition=is_solar)
                df = append_node(df, "generators", "Wind", "[MW]", get_inflexible_power, condition=is_wind)
                df = append_node(df, "generators", "Baseload", "[MW]", get_inflexible_power, condition=is_baseload)
                df = append_node(df, "generators", "Flexible Dispatch", "[MW]", get_dispatched_power, condition=is_flexible)
                df = append_node(df, "reservoirs", "Reservoir Dispatch", "[MW]", get_dispatched_power)
                df = append_node(df, "storages", "Storage Dispatch", "[MW]", get_dispatched_power)
                df = append_node(df, "generators", "Flexible Remaining", "[MWh]", get_remaining_energy, condition=is_flexible)
                df = append_node(df, "reservoirs", "Stored Energy", "[MWh]", get_stored_energy)
                df = append_node(df, "storages", "Stored Energy", "[MWh]", get_stored_energy)
                df = append_node(df, "reservoirs", "Reservoir Inflow", "[MWh]", get_data_power)
                df = append_asset(df, "nodes", "Spillage", "[MW]", get_spillage_power)
                df = append_asset(df, "nodes", "Deficit", "[MW]", get_deficit_power)
                df = append_asset(df, "major_lines", "Flow", "[MW]", get_flow_power)

            case "network":
                df = append_network(df, "nodes", "Demand", "[MW]", get_data_power)
                df = append_network(df, "generators", "Solar", "[MW]", get_inflexible_power, condition=is_solar)
                df = append_network(df, "generators", "Wind", "[MW]", get_inflexible_power, condition=is_wind)
                df = append_network(df, "generators", "Baseload", "[MW]", get_inflexible_power, condition=is_baseload)
                df = append_network(df, "generators", "Flexible Dispatch", "[MW]", get_dispatched_power, condition=is_flexible)
                df = append_network(df, "reservoirs", "Reservoir Dispatch", "[MW]", get_dispatched_power)
                df = append_network(df, "storages", "Storage Dispatch", "[MW]", get_dispatched_power)
                df = append_network(df, "generators", "Flexible Remaining", "[MWh]", get_remaining_energy, condition=is_flexible)
                df = append_network(df, "reservoirs", "Stored Energy", "[MWh]", get_stored_energy)
                df = append_network(df, "storages", "Stored Energy", "[MWh]", get_stored_energy)
                df = append_network(df, "reservoirs", "Reservoir Inflow", "[MWh]", get_data_power)
                df = append_network(df, "nodes", "Spillage", "[MW]", get_spillage_power)
                df = append_network(df, "nodes", "Deficit", "[MW]", get_deficit_power)

        result_file = ResultFile(f"energy_balance_{aggregation_type.upper()}", self.results_directory, df)

        return result_file

    def generate_levelised_costs_file(self) -> ResultFile:
        def get_ltcost(asset):
            return ltcosts_m.get_total(asset.lt_costs)  # $

        def get_dispatched_energy(asset):
            return sum(asset.dispatch_power) * self.solution.static.resolution * 1000.0  # MWh

        def get_inflexible_energy(asset):
            return sum(asset.data) * asset.capacity * self.solution.static.resolution * 1000.0  # MWh

        def get_minor_line_use(asset):
            # TODO: check this logic. Is there a single shared line instance? or more?
            return sum(asset.line.flows) * 1000.0

        def get_dispatched_storage(asset):
            return sum(np.maximum(0, asset.dispatch_power)) * self.solution.static.resolution * 1000.0  # MWh

        def get_storage_losses(asset):
            return (
                -(sum(np.minimum(0, asset.dispatch_power)) + sum(np.maximum(0, asset.dispatch_power)))
                * self.solution.static.resolution
                - (asset.stored_energy[-1] - asset.stored_energy[0])
            ) * 1000.0  # MWh

        def get_line_losses(asset):
            # TODO: line losses
            return 0

        def get_line_use(asset):
            return sum(np.abs(asset.flows)) * 1000.0

        def get_zero(asset):
            return 0

        def get_inflexible_power(asset):
            return asset.data * asset.capacity

        def get_flexible_power(asset):
            return asset.dispatch_power

        def get_nodal_generation(asset):
            """get time series of total generation at the node of an asset (including the asset's contribution)"""
            node_generation = sum((get_inflexible_power(_asset)
                                   for _asset in self.solution.fleet.generators.values()
                                   if (_asset.node.id == asset.node.id) and is_not_flexible(asset)))
            node_generation += sum((get_flexible_power(_asset)
                                    for _asset in self.solution.fleet.generators.values()
                                    if (_asset.node.id == asset.node.id) and is_flexible(asset)))
            node_generation += sum((get_flexible_power(_asset)
                                    for _asset in self.solution.fleet.reservoirs.values()
                                    if _asset.node.id == asset.node.id))
            # in principle, when spillage occurs this is zero - but calculated for robustness
            node_generation += sum((np.maximum(get_flexible_power(_asset), 0)
                                    for _asset in self.solution.fleet.storages.values()
                                    if _asset.node.id == asset.node.id))
            # in principle, when spillage occurs this is zero or negative but calculated for robustness
            node_generation += np.maximum(sum((line.flows
                                               for line in self.solution.network.major_lines.values()
                                               if line.node_start.id == asset.node.id))
                                          - sum((line.flows
                                                 for line in self.solution.network.major_lines.values()
                                                 if line.node_end.id == asset.node.id)),
                                          0)
            return node_generation

        def get_inflexible_curtailment(asset):
            """
            Apportion spillage over the active generators linearly.
            """
            nodal_generation = get_nodal_generation(asset)
            asset_generation = asset.data * asset.capacity
            curtailment = -np.minimum(0, safe_divide_array(asset_generation, nodal_generation) * asset.node.spillage)
            return sum(curtailment) * self.solution.static.resolution * 1000.0

        def get_flexible_curtailment(asset):
            """
            Apportion spillage over the active generators linearly.
            Should always be zero but maintained for robustness
            """
            nodal_generation = get_nodal_generation(asset)
            asset_generation = asset.dispatch_power
            curtailment = -np.minimum(0, safe_divide_array(asset_generation, nodal_generation) * asset.node.spillage)
            return sum(curtailment) * self.solution.static.resolution * 1000.0

        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
            cost_getter: Callable = get_ltcost,
            generation_getter: Callable = get_zero,
            storage_getter: Callable = get_zero,
            transmission_getter: Callable = get_zero,
            curtailment_getter: Callable = get_zero,
            loss_getter: Callable = get_zero,
            condition: Callable = is_any,
        ) -> pd.DataFrame:
            assets = [
                asset
                for asset in getattr(getattr(self.solution, asset_containers[asset_class]), asset_class).values()
                if condition(asset)
            ]
            if len(assets) > 0:
                series = []
                for asset in assets:
                    column = pd.Series(index=df.index, dtype=object)
                    column["Asset Name"] = asset.name
                    column["Asset Type"] = asset_class_to_display[asset_class]
                    column["Asset ID"] = asset.id
                    column["Discounted Cost [$]"] = cost_getter(asset)
                    column["Generation [MWh]"] = generation_getter(asset)
                    column["Storage [MWh]"] = storage_getter(asset)
                    column["Transmission [MWh]"] = transmission_getter(asset)
                    column["Curtailment [MWh]"] = curtailment_getter(asset)
                    column["Loss [MWh]"] = loss_getter(asset)
                    column["LCOE [$/MWh]"] = safe_divide(column["Discounted Cost [$]"], total_energy)
                    column["LCOG [$/MWh]"] = safe_divide(column["Discounted Cost [$]"], column["Generation [MWh]"])
                    column["LCOB storage"] = safe_divide(column["Discounted Cost [$]"], column["Storage [MWh]"])
                    column["LCOB transmission"] = safe_divide(column["Discounted Cost [$]"], column["Transmission [MWh]"])
                    column["LCOB spillage & loss"] = 0.0
                    column["LCOB [$/MWh]"] = (column["LCOB storage"]
                                              + column["LCOB transmission"]
                                              + column["LCOB spillage & loss"])
                    series.append(column)
                df = pd.concat((df, *series), axis=1)
            return df

        def append_system_placeholder(df: pd.DataFrame) -> pd.DataFrame:
            df_to_join = pd.DataFrame(["System", "", "", *(0,) * 12], index=df.index)
            df = pd.concat((df, df_to_join), axis=1)
            return df

        df = pd.DataFrame(
            index=[
                "Asset Name",
                "Asset Type",
                "Asset ID",
                "Discounted Cost [$]",
                "Generation [MWh]",
                "Storage [MWh]",
                "Transmission [MWh]",
                "Curtailment [MWh]",
                "Loss [MWh]",
                "LCOE [$/MWh]",
                "LCOG [$/MWh]",
                "LCOB [$/MWh]",
                "LCOB storage",
                "LCOB transmission",
                "LCOB spillage & loss",
            ],
            dtype=object,
        )

        total_energy = 1000 * abs(
            sum(self.solution.static.year_energy_demand) - network_m.calculate_lt_line_losses(self.solution.network)
        )  # MWh
        total_generation = (
            1000 * self.solution.static.resolution
            * sum(sum(generator.dispatch_power)
                  for generator in self.solution.fleet.generators.values()
                  if generator.unit_type == "flexible")
        )  # MWh
        total_generation += (
            1000 * self.solution.static.resolution
            * sum(sum(generator.data * generator.capacity)
                  for generator in self.solution.fleet.generators.values()
                  if generator.unit_type != "flexible")
        )  # MWh
        total_generation += (
            1000 * self.solution.static.resolution
            * sum(sum(reservoir.dispatch_power) for reservoir in self.solution.fleet.reservoirs.values())
        )  # MWh
        df = append_system_placeholder(df)
        df = append_asset(
            df,
            "generators",
            generation_getter=get_inflexible_energy,
            curtailment_getter=get_inflexible_curtailment,
            condition=is_not_flexible,
        )
        df = append_asset(
            df,
            "generators",
            generation_getter=get_dispatched_energy,
            curtailment_getter=get_flexible_curtailment,
            condition=is_flexible,
        )
        df = append_asset(
            df,
            "reservoirs",
            generation_getter=get_dispatched_energy,
            curtailment_getter=get_flexible_curtailment,
        )
        df = append_asset(
            df,
            "storages",
            storage_getter=get_dispatched_storage,
            curtailment_getter=get_flexible_curtailment,
            loss_getter=get_storage_losses,
        )
        df = append_asset(df, "major_lines", transmission_getter=get_line_use)
        df = append_asset(df, "major_lines", transmission_getter=get_line_use)

        df.columns = pd.RangeIndex(len(df.columns))
        for row in (
            "Discounted Cost [$]",
            "Generation [MWh]",
            "Storage [MWh]",
            "Transmission [MWh]",
            "Curtailment [MWh]",
            "Loss [MWh]",
        ):
            df.loc[row, 0] = sum(df.loc[row, :])

        first_mask = np.ones(len(df.columns), dtype=bool)
        first_mask[0] = False
        df.loc["LCOE [$/MWh]", 0] = safe_divide(df.loc["Discounted Cost [$]", 0], total_energy)
        df.loc["LCOG [$/MWh]", 0] = safe_divide(
            sum(df.loc["Discounted Cost [$]", (df.loc["Generation [MWh]"] > 0) & first_mask]), df.loc["Generation [MWh]", 0]
        )
        df.loc["LCOB [$/MWh]", 0] = df.loc["LCOE [$/MWh]", 0] - df.loc["LCOG [$/MWh]", 0]
        df.loc["LCOB storage", 0] = safe_divide(
            sum(df.loc["Discounted Cost [$]", (df.loc["Storage [MWh]"] > 0) & first_mask]),
            total_energy,
        )
        df.loc["LCOB transmission", 0] = safe_divide(
            sum(df.loc["Discounted Cost [$]", (df.loc["Transmission [MWh]"] > 0) & first_mask]),
            total_energy,
        )
        df.loc["LCOB spillage & loss", 0] = (df.loc["LCOB [$/MWh]", 0]
                                             - df.loc["LCOB storage", 0]
                                             - df.loc["LCOB transmission", 0])

        result_file = ResultFile("levelised_costs", self.results_directory, df)
        return result_file

    def calculate_annual_energies(self, arr: NDArray[np.float64], decimals: int = 3) -> float:
        annual_energies_arr = np.zeros(self.solution.static.year_count + 1, dtype=np.float64)
        for year in range(self.solution.static.year_count):
            first_t, last_t = static_m.get_year_t_boundaries(self.solution.static, year)
            annual_energies_arr[year] = round(
                sum(arr[first_t:last_t] * self.solution.static.interval_resolutions[first_t:last_t]),
                decimals,
            )
        annual_energies_arr[-1] = round(sum(arr * self.solution.static.interval_resolutions), decimals)
        return annual_energies_arr

    def generate_summary_file(self) -> ResultFile:
        def get_data_power(asset) -> NDArray[np.float64]:
            return asset.data * self.solution.static.resolution  # GWh / interval

        def get_spillage_power(asset) -> NDArray[np.float64]:
            return asset.spillage * self.solution.static.resolution  # GWh / interval

        def get_deficit_power(asset) -> NDArray[np.float64]:
            return asset.deficits * self.solution.static.resolution  # GWh / interval

        def get_dispatched_power(asset) -> NDArray[np.float64]:
            return asset.dispatch_power * self.solution.static.resolution  # GWh / interval

        def get_inflexible_power(asset) -> NDArray[np.float64]:
            return asset.data * asset.capacity * self.solution.static.resolution  # GWh / interval

        def get_discharge_power(asset) -> NDArray[np.float64]:
            return np.maximum(0, asset.dispatch_power) * self.solution.static.resolution  # GWh / interval

        def get_flow_power(asset) -> NDArray[np.float64]:
            return np.abs(asset.flows) * self.solution.static.resolution  # GWh / interval

        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
            asset_class_name: str,
            column_name: str,
            time_series_getter: Callable,
            condition: Callable = is_any,
        ) -> pd.DataFrame:
            """Add all assets of an asset class to the DataFrame"""
            assets = [
                asset
                for asset in getattr(getattr(self.solution, asset_containers[asset_class]), asset_class).values()
                if condition(asset)
            ]
            series = []
            if len(assets) > 0:
                for asset in assets:
                    full_trace = time_series_getter(asset)
                    series.append(
                        pd.Series(
                            [
                                asset.name,
                                asset_class_name,
                                asset.id,
                                column_name,
                                "[GWh]",
                                *tuple(sum(full_trace[slice(*idx)]) for idx in year_indices),
                                sum(full_trace),
                            ],
                            index=df.index,
                        )
                    )
                df = pd.concat((df, *series), axis=1)
            return df

        year_indices = [
            static_m.get_year_t_boundaries(self.solution.static, year)
            for year in range(self.solution.static.year_count)
        ]

        df = pd.DataFrame(
            index=[
                "Asset Name",
                "Asset Type",
                "Asset ID",
                "Column Name",
                "Column Units",
                *tuple(range(self.solution.static.first_year, self.solution.static.final_year + 1)),
                "Total",
            ]
        )

        df = append_asset(df, "nodes", "Node", "Annual Demand", get_data_power)
        df = append_asset(df, "generators", "Generator", "Annual Generation", get_inflexible_power, is_not_flexible)
        df = append_asset(df, "generators", "Generator", "Annual Generation", get_dispatched_power, is_flexible)
        df = append_asset(df, "reservoirs", "Reservoir", "Annual Generation", get_dispatched_power)
        df = append_asset(df, "storages", "Storage", "Annual Dispatch", get_discharge_power)
        df = append_asset(df, "reservoirs", "Reservoir", "Annual Inflow", get_data_power)
        df = append_asset(df, "nodes", "Node", "Spillage", get_spillage_power)
        df = append_asset(df, "nodes", "Node", "Deficit", get_deficit_power)
        df = append_asset(df, "major_lines", "Major Line", "Flow", get_flow_power)

        result_file = ResultFile("summary", self.results_directory, df)
        return result_file

    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile(
            "x", self.results_directory, pd.DataFrame(self.solution.x).T, write_kwargs={"index": False}
        )
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
