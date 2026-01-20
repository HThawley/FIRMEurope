# type: ignore
import os
import shutil
import time

from re import sub
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Callable, Any

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.common.helpers import safe_divide
from firm_ce.fast_methods import ltcosts_m, network_m, static_m
from firm_ce.analysis.accessor import Accessor
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
        self.statistics_generated = False

    def create_solution_directory(self, result_directory: str, solution_name: str) -> str:
        safe_name = sub(r"[^a-zA-Z0-9_\-]", "_", solution_name)
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
        if not self.solution.evaluated:
            raise RuntimeError("Solution must be evaluated before generating statistics.")

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

        self.statistics_generated = True

        return None

    def write_results(self) -> None:
        if not self.statistics_generated:
            raise RuntimeError("Statistics must be generated before writing results.")

        for result_file in self.result_files.values():
            result_file.write()

        return None

    def generate_capacities_file(self) -> ResultFile:
        """Generates the capacities CSV"""
        accessor = Accessor(self.solution, "GW")

        def _construct_column(asset, asset_class, column_name, column_units, index) -> pd.Series:
            capacity = accessor.get_capacity(asset)
            new_build, min_build, max_build = accessor.get_build_limits(asset)

            return pd.Series([
                asset.name,
                accessor.get_display_name(asset_class),
                asset.id,
                column_name,
                column_units,
                round(capacity, 3),
                round(new_build, 3),
                round(min_build, 3),
                round(max_build, 3),
            ], index=index)

        def append_asset(
            df: pd.DataFrame, asset_class: str, attribute: str
        ) -> pd.DataFrame:
            """Add all assets in an asset class (generators, reservoirs, ...) to the capacities DataFrame"""

            match attribute.lower():
                case "power":
                    column_name, column_units = "Power Capacity", "[GW]"
                case "energy":
                    column_name, column_units = "Energy Capacity", "[GWh]"
                case _:
                    raise ValueError(f"'attribute should be 'energy' or 'power'. Got '{attribute}'.")

            df = pd.concat((
                df,
                pd.concat((
                    _construct_column(asset, asset_class, column_name, column_units, df.index)
                    for asset in accessor.get_assets(asset_class).values()
                ), axis=1),
            ), axis=1)
            return df

        df = pd.DataFrame(
            index=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Column Units",
                   "Total Capacity", "New Build Capacity", "Min Build", "Max Build"]
        )
        df = append_asset(df, "generators", "power")
        df = append_asset(df, "reservoirs", "power")
        df = append_asset(df, "reservoirs", "energy")
        df = append_asset(df, "storages", "power")
        df = append_asset(df, "storages", "energy")
        df = append_asset(df, "major_lines", "power")
        df = append_asset(df, "minor_lines", "power")

        result_file = ResultFile("capacities", self.results_directory, df, decimals=3)
        return result_file

    def generate_component_costs_file(self) -> ResultFile:
        def _construct_column(asset: Any, asset_class: str, index: pd.Index) -> pd.Series:
            return pd.Series([
                asset.name,
                asset_class_to_display[asset_class],
                asset.id,
                "Total Cost",
                "[$]",
                round(asset.lt_costs.annualised_build, 3),
                round(asset.lt_costs.fom, 3),
                round(asset.lt_costs.vom, 3),
                round(asset.lt_costs.fuel, 3),
            ], index=index)

        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
        ) -> pd.DataFrame:
            """Add all assets of an asset class to the DataFrame"""
            df = pd.concat((
                df,
                pd.concat((
                    _construct_column(asset, asset_class, df.index)
                    for asset in Accessor.get_assets_from_solution(self.solution, asset_class).values()
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

        result_file = ResultFile("component_costs", self.results_directory, df, decimals=2)
        return result_file

    def generate_energy_balance_file(self, aggregation_type: str) -> List[ResultFile]:
        accessor = Accessor(self.solution, "MW")

        def _construct_column(
            asset: Any,
            asset_class: str,
            column_name: str,
            column_units: str,
            time_series_getter: Callable,
            index: pd.Index,
        ) -> pd.Series:
            series = pd.concat((
                pd.Series([asset.name, accessor.get_display_name(asset_class), asset.id, column_name, column_units]),
                pd.Series(time_series_getter(asset)),
            ), ignore_index=True)
            series.index = index
            return series

        def append_series(
            df: pd.DataFrame,
            aggregation: str,
            asset_class: str,
            column_name: str,
            column_units: str,
            time_series_getter: Callable,
            condition: Callable = Accessor.is_any,
        ) -> pd.DataFrame:
            """
            Add an (aggregated) time series feature (power trace, etc.) of all assets of an asset class
            to the DataFrame
            """
            match aggregation:
                case "asset":
                    for asset in accessor.get_assets(asset_class).values():
                        if not condition(asset):
                            continue
                        df = pd.concat(
                            (df, _construct_column(asset, asset_class, column_name, column_units, time_series_getter, df.index)),
                            axis=1
                        )
                    return df

                case "node":
                    for node in accessor.get_assets("nodes").values():
                        column = pd.Series([], index=df.index)
                        column.iloc[:5] = node.name, accessor.get_display_name(asset_class), node.id, column_name, column_units
                        for asset in accessor.get_assets(asset_class).values():
                            if not condition(asset) or asset.node.id != node.id:
                                continue
                            column.iloc[5:] += _construct_column(
                                node, asset_class, column_name, column_units, time_series_getter, df.index).iloc[5:]
                        df = pd.concat((df, column), axis=1)
                    return df

                case "network":
                    column = pd.Series([], index=df.index)
                    column.iloc[:5] = "Network", "Network", 0, column_name, column_units
                    for asset in accessor.get_assets(asset_class).values():
                        if not condition(asset):
                            continue
                        column.iloc[5:] += _construct_column(
                            asset, asset_class, column_name, column_units, time_series_getter, df.index).iloc[5:]
                    df = pd.concat((df, column), axis=1)
                    return df

        df = pd.concat((
            pd.DataFrame(index=["Asset Name", "Asset Type", "Asset ID", "Column Name", "Column Units"]),
            pd.DataFrame(index=pd.RangeIndex(self.full_intervals_count))))

        match aggregation_type:
            case "assets":
                df = append_series(df, "asset", "nodes", "Demand", "[MW]", accessor.get_power_trace)
                df = append_series(df, "asset", "generators", "Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "asset", "reservoirs", "Reservoir Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "asset", "storages", "Storage Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "asset", "generators", "Flexible Remaining", "[MWh]", accessor.get_remaining_energy_trace,
                                   condition=Accessor.is_flexible)
                df = append_series(df, "asset", "reservoirs", "Reservoir Energy", "[MWh]", accessor.get_storage_level_trace)
                df = append_series(df, "asset", "storages", "Stored Energy", "[MWh]", accessor.get_storage_level_trace)
                df = append_series(df, "asset", "reservoirs", "Inflow", "[MWh]", accessor.get_inflow_energy)
                df = append_series(df, "asset", "nodes", "Spillage", "[MW]", accessor.get_spillage_trace)
                df = append_series(df, "asset", "nodes", "Deficit", "[MW]", accessor.get_deficit_trace)
                df = append_series(df, "asset", "major_lines", "Flow", "[MW]", accessor.get_transmission_trace)

            case "nodes":
                df = append_series(df, "asset", "nodes", "Demand", "[MW]", accessor.get_power_trace)
                df = append_series(df, "node", "generators", "Solar", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_solar)
                df = append_series(df, "node", "generators", "Wind", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_wind)
                df = append_series(df, "node", "generators", "Run-of-river", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_ror)
                df = append_series(df, "node", "generators", "Baseload", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_baseload)
                df = append_series(df, "node", "generators", "Flexible Dispatch", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_flexible)
                df = append_series(df, "node", "reservoirs", "Reservoir Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "node", "storages", "Storage Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "node", "generators", "Flexible Remaining", "[MWh]", accessor.get_remaining_energy_trace,
                                   condition=Accessor.is_flexible)
                df = append_series(df, "node", "reservoirs", "Reservoir Energy", "[MWh]", accessor.get_storage_level_trace)
                df = append_series(df, "node", "storages", "Stored Energy", "[MWh]", accessor.get_storage_level_trace)
                df = append_series(df, "node", "reservoirs", "Reservoir Inflow", "[MWh]", accessor.get_inflow_energy)
                df = append_series(df, "asset", "nodes", "Spillage", "[MW]", accessor.get_spillage_trace)
                df = append_series(df, "asset", "nodes", "Deficit", "[MW]", accessor.get_deficit_trace)
                df = append_series(df, "asset", "major_lines", "Flow", "[MW]", accessor.get_transmission_trace)

            case "network":
                df = append_series(df, "network", "nodes", "Demand", "[MW]", accessor.get_power_trace)
                df = append_series(df, "network", "generators", "Solar", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_solar)
                df = append_series(df, "network", "generators", "Wind", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_wind)
                df = append_series(df, "network", "generators", "Ror", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_ror)
                df = append_series(df, "network", "generators", "Baseload", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_baseload)
                df = append_series(df, "network", "generators", "Flexible Dispatch", "[MW]", accessor.get_power_trace,
                                   condition=Accessor.is_flexible)
                df = append_series(df, "network", "reservoirs", "Reservoir Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "network", "storages", "Storage Dispatch", "[MW]", accessor.get_power_trace)
                df = append_series(df, "network", "generators", "Flexible Remaining", "[MWh]", accessor.get_remaining_energy_trace,
                                   condition=Accessor.is_flexible)
                df = append_series(df, "network", "reservoirs", "Reservoir Energy", "[MWh]", accessor.get_storage_level_trace)
                df = append_series(df, "network", "storages", "Stored Energy", "[MWh]", accessor.get_storage_level_trace)
                df = append_series(df, "network", "reservoirs", "Reservoir Inflow", "[MWh]", accessor.get_inflow_energy)
                df = append_series(df, "network", "nodes", "Spillage", "[MW]", accessor.get_spillage_trace)
                df = append_series(df, "network", "nodes", "Deficit", "[MW]", accessor.get_deficit_trace)

        result_file = ResultFile(f"energy_balance_{aggregation_type.upper()}", self.results_directory, df, decimals=3)

        return result_file

    def generate_levelised_costs_file(self) -> ResultFile:

        accessor = Accessor(self.solution)

        def get_ltcost(asset):
            return ltcosts_m.get_total(asset.lt_costs)  # $

        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
            cost_getter: Callable = get_ltcost,
            generation_getter: Callable = Accessor.get_zero,
            storage_getter: Callable = Accessor.get_zero,
            transmission_getter: Callable = Accessor.get_zero,
            curtailment_getter: Callable = Accessor.get_zero,
            loss_getter: Callable = Accessor.get_zero,
            condition: Callable = Accessor.is_any,
        ) -> pd.DataFrame:
            for asset in accessor.get_assets(asset_class).values():
                if not condition(asset):
                    continue
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
                df = pd.concat((df, column), axis=1)
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
            generation_getter=accessor.get_energy_net,
            curtailment_getter=accessor.get_nominal_curtailment_net,
        )
        df = append_asset(
            df,
            "reservoirs",
            generation_getter=accessor.get_energy_net,
            curtailment_getter=accessor.get_nominal_curtailment_net,
        )
        df = append_asset(
            df,
            "storages",
            storage_getter=accessor.get_discharge_net,
            curtailment_getter=accessor.get_nominal_curtailment_net,
            loss_getter=accessor.get_storage_losses,
        )
        df = append_asset(df, "major_lines", transmission_getter=accessor.get_line_use_net)
        df = append_asset(df, "major_lines", transmission_getter=accessor.get_line_use_net)

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

        result_file = ResultFile("levelised_costs", self.results_directory, df, decimals=2)
        return result_file

    def generate_summary_file(self) -> ResultFile:
        accessor = Accessor(self.solution, "GW")
        year_indices = [
            static_m.get_year_t_boundaries(self.solution.static, year)
            for year in range(self.solution.static.year_count)
        ]

        def _construct_column(
            asset: Any,
            asset_class: str,
            column_name: str,
            time_series_getter: Callable,
            index: pd.Index
        ) -> pd.Series:
            full_trace = time_series_getter(asset)
            return pd.Series([
                asset.name,
                asset_class_to_display[asset_class],
                asset.id,
                column_name,
                "[GWh]",
                *tuple(sum(full_trace[slice(*idx)]) for idx in year_indices),
                sum(full_trace),
            ], index=index)

        def append_asset(
            df: pd.DataFrame,
            asset_class: str,
            asset_class_name: str,
            column_name: str,
            time_series_getter: Callable,
            condition: Callable = Accessor.is_any,
        ) -> pd.DataFrame:
            """Add all assets of an asset class to the DataFrame"""
            for asset in accessor.get_assets(asset_class).values():
                if not condition(asset):
                    continue
                df = pd.concat(
                    (df, _construct_column(asset, asset_class, column_name, time_series_getter, df.index)),
                    axis=1
                )

            return df

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

        df = append_asset(df, "nodes", "Node", "Annual Demand", accessor.get_power_trace)
        df = append_asset(df, "generators", "Generator", "Annual Generation", accessor.get_power_trace)
        df = append_asset(df, "reservoirs", "Reservoir", "Annual Generation", accessor.get_power_trace)
        df = append_asset(df, "storages", "Storage", "Annual Dispatch", accessor.get_discharge_trace)
        df = append_asset(df, "reservoirs", "Reservoir", "Annual Inflow", accessor.get_inflow_trace)
        df = append_asset(df, "nodes", "Node", "Spillage", accessor.get_spillage_trace)
        df = append_asset(df, "nodes", "Node", "Deficit", accessor.get_deficit_trace)
        df = append_asset(df, "major_lines", "Major Line", "Flow", accessor.get_transmission_trace())

        result_file = ResultFile("summary", self.results_directory, df, decimals=3)
        return result_file

    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile(
            "x", self.results_directory, pd.DataFrame(self.solution.x).T, write_kwargs={"index": False}, decimals=3
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
