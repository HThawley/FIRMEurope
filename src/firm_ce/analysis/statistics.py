# type: ignore
import os
import shutil
import time

from re import sub
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.fast_methods import static_m
from firm_ce.analysis.accessor import Accessor
from firm_ce.io.file_manager import ResultFile
from firm_ce.optimisation.single_time import Solution, evaluate
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
        evaluate(self.solution)
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

        self.df_static, self.df_temporal = self._build_master_tables()
        self.statistics_generated = False

    def _build_master_tables(self):
        accessor = Accessor(self.solution, "GW")
        static_data = []
        temporal_data = {}
        asset_classes = ["nodes", "generators", "storages", "major_lines"]  # , "minor_lines"]
        meta_data_names = ("Asset ID", "Asset Name", "Asset Type", "Asset Class", "Unit Type", "Node")

        for asset_class in asset_classes:
            is_node = asset_class == "nodes"
            assets = accessor.get_assets(asset_class)
            for asset in assets.values():
                meta_data = (
                    asset.id,
                    asset.name,
                    accessor.get_display_name(asset_class),
                    asset_class,
                    getattr(asset, "unit_type", "node" if is_node else None),
                    asset.node.name if hasattr(asset, "node") else (asset.name if is_node else None),
                )
                row = dict(zip(meta_data_names, meta_data))

                if hasattr(asset, "lt_costs"):
                    row.update({"Annualised Build": getattr(asset.lt_costs, "annualised_build", 0.0)})
                    row.update({"Fixed O&M": getattr(asset.lt_costs, "fom", 0.0)})
                    row.update({"Variable O&M": getattr(asset.lt_costs, "vom", 0.0)})
                    row.update({"Fuel Cost": getattr(asset.lt_costs, "fuel", 0.0)})

                row["Power Capacity"] = accessor.get_power_capacity(asset, errors="coerce")
                row["Energy Capacity"] = accessor.get_energy_capacity(asset, errors="coerce")
                row.update(
                    dict(
                        zip(
                            ("Existing Power", "New Build Power", "Min Build Power", "Max Build Power"),
                            accessor.get_build_power(asset, errors="coerce"),
                        )
                    )
                )
                row.update(
                    dict(
                        zip(
                            ("Existing Energy", "New Build Energy", "Min Build Energy", "Max Build Energy"),
                            accessor.get_build_energy(asset, errors="coerce"),
                        )
                    )
                )
                static_data.append(row)

                if accessor.is_node(asset):
                    temporal_data[(*meta_data, "Demand")] = accessor.get_power_trace(asset)
                    temporal_data[(*meta_data, "Spillage")] = accessor.get_spillage_trace(asset)
                    temporal_data[(*meta_data, "Deficit")] = accessor.get_deficit_trace(asset)

                # elif accessor.is_major_line(asset):
                elif accessor.is_line(asset):
                    temporal_data[(*meta_data, "Flow")] = accessor.get_transmission_trace(asset)

                else:
                    # For Generators, and Storage
                    temporal_data[(*meta_data, "Dispatch")] = accessor.get_power_trace(asset)

                    # Batteries / Storage
                    if accessor.is_storage(asset):
                        temporal_data[(*meta_data, "Stored_Energy")] = accessor.get_storage_level_trace(asset)
                        temporal_data[(*meta_data, "Charge")] = accessor.get_charge_trace(asset)
                        temporal_data[(*meta_data, "Discharge")] = accessor.get_discharge_trace(asset)

                    if accessor.has_inflows(asset):
                        temporal_data[(*meta_data, "Inflows")] = accessor.get_inflow_trace(asset)

        for asset in accessor.get_assets('fuels').values():
            meta_data = (
                asset.id,
                asset.name,
                accessor.get_display_name(asset_class),
                asset_class,
                "fuel",
                "network",
            )
            row = dict(zip(meta_data_names, meta_data))
            temporal_data[(*meta_data, "Fuel_Remaining")] = accessor.get_remaining_energy_trace(asset)

        df_static = pd.DataFrame(static_data)
        if not df_static.empty:
            node_mask = df_static["Asset Class"] == "nodes"
            for column in ("Power Capacity", "Energy Capacity", "Existing Power", "Existing Energy",
                           "New Build Power", "Min Build Power", "Max Build Power", "New Build Energy",
                           "Min Build Energy", "Max Build Energy", "Annualised Build",
                           "Fixed O&M", "Variable O&M", "Fuel Cost"):
                nodal_values = df_static[
                    df_static["Asset Type"].isin(("Generator", "Storage"))
                ].fillna(0.0).groupby("Node")[column].sum()

                df_static.loc[node_mask, column] = df_static.loc[node_mask, "Asset Name"].map(nodal_values)

            df_static.set_index("Asset ID", inplace=True)
            # Fill NaNs for assets that missed certain optional fields (like costs for nodes)
            # df_static.fillna(0.0, inplace=True)

        # B. Temporal Master Table (MultiIndex Columns)
        if temporal_data:
            df_temporal = pd.DataFrame(temporal_data)

            index_names = (*meta_data_names, "Variable")

            df_temporal.columns = pd.MultiIndex.from_tuples(df_temporal.columns, names=index_names)
            df_temporal.index.name = "Time_Step"
        else:
            df_temporal = pd.DataFrame()

        return df_static, df_temporal

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
        """
        Generates all result files using the high-level master DataFrames.
        """
        if not self.statistics_generated:
            # Ensure master tables are built if not already done
            if not hasattr(self, "df_static"):
                self.df_static, self.df_temporal = self._build_master_tables()

        self.result_files = {
            "capacities": self._view_capacities(),
            "component_costs": self._view_component_costs(),
            "energy_balance_ASSETS": self._view_energy_balance("assets"),
            "energy_balance_NODES": self._view_energy_balance("nodes"),
            "energy_balance_NETWORK": self._view_energy_balance("network"),
            "levelised_costs": self._view_levelised_costs(),
            "summary": self._view_summary(),
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

    def _view_capacities(self) -> ResultFile:
        """
        View: Static Capacity Data.
        Units: MW -> GW (Output)
        """
        # Select relevant columns from master static table
        string_cols = [
            "Asset Name",
            "Asset Type",
            "Unit Type",
            "Node",
        ]

        numeric_cols = [
            "Power Capacity",
            "Energy Capacity",
            "Existing Power",
            "Existing Energy",
            "New Build Power",
            "New Build Energy",
            "Min Build Power",
            "Min Build Energy",
            "Max Build Power",
            "Max Build Energy",
        ]

        df = self.df_static.loc[:, string_cols + numeric_cols].copy()

        df[numeric_cols] = df[numeric_cols]
        df[numeric_cols] = df[numeric_cols].round(3)
        df = df.T

        return ResultFile("capacities", self.results_directory, df)

    def _view_component_costs(self) -> ResultFile:
        """
        View: Asset Costs.
        Units: $ (No conversion needed)
        """
        string_cols = [
            "Asset Name",
            "Asset Type",
            "Unit Type",
            "Node",
        ]

        numeric_columns = [
            "Annualised Build",
            "Fixed O&M",
            "Variable O&M",
            "Fuel Cost",
        ]

        # Filter for assets with costs (exclude Nodes)
        df = self.df_static.loc[:, string_cols + numeric_columns].copy()
        df["Total Cost"] = df[numeric_columns].sum(axis=1)

        all_numeric_columns = ["Total Cost"] + numeric_columns
        # Reorder to put Total Cost first
        df = df[string_cols + all_numeric_columns]
        df[all_numeric_columns] = (df[all_numeric_columns] / 1e6).round(3)

        for col in all_numeric_columns:
            df[f"{col} ($/kW/year)"] = (df[col] / self.df_static["Power Capacity"]).round(3)

        # Filter out zero-cost rows (like Nodes)
        df = df.loc[df["Total Cost"] > 1e-6, :]
        df = df.rename(columns=dict(zip(all_numeric_columns, [col + " (M$/year)" for col in all_numeric_columns])))
        df = df.T

        return ResultFile("component_costs", self.results_directory, df)

    def _view_energy_balance(self, aggregation: str) -> ResultFile:
        """
        View: Time Series Energy Balance.
        Units: MW (Maintained from Master)
        """
        string_cols = [
            "Asset Name",
            "Asset Type",
            "Unit Type",
            "Node",
            "Variable"
        ]

        df_t = self.df_temporal.copy().T
        time_cols = [c for c in df_t.columns if c not in self.df_temporal.columns.names]

        # Isolate Fuel_Remaining before aggregation
        is_fuel_rem = df_t.index.get_level_values("Variable") == "Fuel_Remaining"
        if aggregation in ("assets", "nodes"):
            df_fuel_rem = df_t[is_fuel_rem].reset_index()
            if not df_fuel_rem.empty:
                df_fuel_rem = df_fuel_rem.loc[:, string_cols + time_cols].copy()
            df_t = df_t[~is_fuel_rem]
        else:
            df_fuel_rem = pd.DataFrame()

        if aggregation.lower() == "assets":
            df_t = df_t.reset_index()
            df_t = df_t.loc[:, string_cols + time_cols].copy()

        elif aggregation == "nodes":
            is_flow = df_t.index.get_level_values("Variable") == "Flow"
            # Group single-node variables, ignoring lines (NaN nodes drop automatically)
            df_base = df_t[~is_flow].groupby(level=["Node", "Variable"]).sum()

            df_flows = df_t[is_flow]
            if not df_flows.empty:
                asset_names = df_flows.index.get_level_values("Asset Name")
                # Send nodes: Negate flow (- negative values become positive)
                df_send = pd.DataFrame(-df_flows.values, columns=df_flows.columns)
                df_send["Node"] = [name.split("-")[0] for name in asset_names]
                df_send["Variable"] = "Flow"

                # Receive nodes: Maintain original flow
                df_rec = pd.DataFrame(df_flows.values, columns=df_flows.columns)
                df_rec["Node"] = [name.split("-")[1] for name in asset_names]
                df_rec["Variable"] = "Flow"

                # Combine node base data with directional flows and sum by Node
                df_t = pd.concat([
                    df_base,
                    df_send.groupby(["Node", "Variable"]).sum(),
                    df_rec.groupby(["Node", "Variable"]).sum()
                ]).groupby(level=["Node", "Variable"]).sum()
            else:
                df_t = df_base

            df_t = df_t.reset_index()
            df_t["Asset Name"] = df_t["Node"]
            df_t["Asset Type"] = "Node"
            df_t["Unit Type"] = "Node"
            df_t = df_t.loc[:, string_cols + time_cols].copy()

        elif aggregation.lower() == "network":
            df_t = df_t.replace(np.inf, 0).groupby(level="Variable").sum()
            df_t = df_t.reset_index()
            df_t["Asset Name"] = "Network"
            df_t["Asset Type"] = "Network"
            df_t["Unit Type"] = "Network"
            df_t["Node"] = "System"
            df_t = df_t.loc[:, string_cols + time_cols].copy()
            df_t = df_t[df_t["Variable"] != "Flow"]

        var_order = [
            'Demand', 'Deficit', 'Spillage', 'Dispatch', 'Flow', 'Discharge',
            'Charge', 'Inflows', 'Stored_Energy', 'Fuel_Remaining'
        ]
        sort_map = {var: i for i, var in enumerate(var_order)}
        df_t['_var_sort'] = df_t['Variable'].map(lambda x: sort_map.get(x, 9999))
        asset_order = {"Node": 1, "Generator": 2, "Storage": 3}
        df_t['_asset_sort'] = df_t['Asset Type'].map(lambda x: asset_order.get(x, 999))
        df_t['_node_sort'] = df_t['Node'].fillna('zzzz_lines')

        df_t = df_t.sort_values(['_node_sort', '_asset_sort', 'Asset Name', '_var_sort']).drop(
            columns=['_node_sort', '_asset_sort', '_var_sort']
        )

        if not df_fuel_rem.empty:
            df_t = pd.concat([df_t, df_fuel_rem], ignore_index=True)

        df_out = df_t.rename(columns={"Variable": "Timestep"}).T

        return ResultFile(f"energy_balance_{aggregation.upper()}", self.results_directory, df_out, decimals=3)

    def _view_summary(self) -> ResultFile:
        resolution = self.solution.static.resolution

        string_cols = [
            "Asset Name",
            "Asset Type",
            "Unit Type",
            "Node",
            "Variable"
        ]

        df_t = self.df_temporal.copy().T
        df_t = df_t[~(df_t.index.get_level_values("Variable") == "Fuel_Remaining")]
        time_cols = [c for c in df_t.columns if c not in self.df_temporal.columns.names]
        df_t = df_t.reset_index().loc[:, string_cols + time_cols].copy()

        df_t["Total_GWh"] = (df_t[time_cols].abs().sum(axis=1) * resolution)
        index_cols = [col for col in string_cols if col != "Variable"]
        df_summary = df_t.pivot_table(
            index=index_cols,
            columns="Variable",
            values="Total_GWh",
            aggfunc="sum",
        ).fillna(0.0)

        cols_to_drop = [c for c in ['Fuel_Remaining', 'Stored_Energy'] if c in df_summary.columns]
        df_summary = df_summary.drop(columns=cols_to_drop)

        if 'Inflows' in df_summary.columns:
            # inflows is already an energy
            df_summary['Inflows'] /= resolution

        # Reset index to bring metadata into columns
        df_summary = df_summary.reset_index()

        asset_order = {"Node": 1, "Generator": 2, "Storage": 3}
        df_summary['_asset_sort'] = df_summary['Asset Type'].map(lambda x: asset_order.get(x, 999))
        df_summary['_node_sort'] = df_summary['Node'].fillna('zzzz_lines')
        df_summary = df_summary.sort_values(['_node_sort', '_asset_sort', 'Asset Name']).drop(
            columns=['_node_sort', '_asset_sort']
        )
        var_order = [
            'Demand', 'Deficit', 'Spillage', 'Dispatch', 'Flow', 'Discharge',
            'Charge', 'Inflows'
        ]
        ordered_vars = [v for v in var_order if v in df_summary.columns]
        ordered_vars += [v for v in df_summary.columns if v not in ordered_vars and v not in index_cols]
        df_summary = df_summary[index_cols + ordered_vars].T

        return ResultFile("summary", self.results_directory, df_summary, decimals=3)

    def _view_levelised_costs(self) -> ResultFile:
        resolution = self.solution.static.resolution
        year_count = self.solution.static.year_count

        # Calculate Total Demand for System LCOE
        if "Demand" in self.df_temporal.columns.get_level_values("Variable"):
            total_demand_mwh = self.df_temporal.xs("Demand", level="Variable", axis=1).sum().sum() * resolution * 1000
        else:
            total_demand_mwh = 0.0

        string_cols = ["Asset ID", "Asset Name", "Asset Type", "Asset Class", "Unit Type", "Node"]

        # 1. Temporal Aggregations (GWh)
        df_t = self.df_temporal.copy()
        # Take absolute value before summing (critical for capturing transmission flow correctly)
        df_totals = (df_t.abs().sum(axis=0) * resolution).reset_index(name="Total_GWh")

        for c in string_cols:
            df_totals[c] = df_totals[c].fillna("None")

        df_totals_pivot = df_totals.pivot_table(
            index=string_cols,
            columns="Variable",
            values="Total_GWh",
            aggfunc="sum"
        ).fillna(0.0).reset_index()

        df_costs = self.df_static.copy().reset_index()
        for c in string_cols:
            if c in df_costs.columns:
                df_costs[c] = df_costs[c].fillna("None")

        cost_cols = ["Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"]
        for c in cost_cols:
            if c not in df_costs.columns:
                df_costs[c] = 0.0
            df_costs[c] = df_costs[c] / 1e6  # Convert to M$

        # 3. Merge Static and Temporal
        df_merged = pd.merge(df_costs, df_totals_pivot, on=string_cols, how="outer").fillna(0.0)

        def get_col(name):
            return df_merged[name] if name in df_merged.columns else pd.Series(0.0, index=df_merged.index)

        dispatch = get_col("Dispatch")
        inflows = get_col("Inflows")
        spillage = get_col("Spillage")
        flow = get_col("Flow")

        # 4. Energy and Cost Metrics Mapping
        df_merged["Generation [GWh]"] = dispatch
        stor_mask = df_merged["Asset Type"].astype(str).str.lower() == "storage"
        df_merged.loc[stor_mask, "Generation [GWh]"] = inflows[stor_mask]

        df_merged["Storage [GWh]"] = 0.0
        df_merged.loc[stor_mask, "Storage [GWh]"] = dispatch[stor_mask]

        df_merged["Transmission [GWh]"] = flow
        df_merged["Curtailment [GWh]"] = spillage

        df_merged.rename(columns={
            "Annualised Build": "Annualised Build [M$/yr]",
            "Fixed O&M": "Fixed O&M [$M/yr]",
            "Variable O&M": "Variable O&M [M$/yr]",
            "Fuel Cost": "Fuel Cost [M$/yr]"
        }, inplace=True)

        mapped_costs = [
            "Annualised Build [M$/yr]", "Fixed O&M [$M/yr]",
            "Variable O&M [M$/yr]", "Fuel Cost [M$/yr]"
        ]
        node_mask = df_merged["Asset Class"].astype(str).str.lower() == "nodes"
        df_merged.loc[node_mask, mapped_costs] = 0.0

        df_merged["Total Cost [M$/yr]"] = df_merged[mapped_costs].sum(axis=1)

        # Helper to calculate Levelised Cost ($/MWh = M$ * 1000 / GWh)
        def calc_lco(cost_m, energy_gwh):
            return np.where(energy_gwh > 1e-6, (cost_m * 1000) / energy_gwh, 0.0)

        df_merged["LCOG [$/MWh]"] = calc_lco(df_merged["Total Cost [M$/yr]"] * year_count, df_merged["Generation [GWh]"])
        df_merged["LCOS [$/MWh]"] = calc_lco(df_merged["Total Cost [M$/yr]"] * year_count, df_merged["Storage [GWh]"])
        df_merged["LCOT [$/MWh]"] = calc_lco(df_merged["Total Cost [M$/yr]"] * year_count, df_merged["Transmission [GWh]"])
        df_merged["LCOE [$/MWh]"] = 0.0

        # 5. Build Aggregations (Nodal & System)
        def aggregate_rows(df_sub, name, asset_type, unit_type, node, is_system=False):
            agg = {
                "Asset Name": name, "Asset Type": asset_type,
                "Unit Type": unit_type, "Node": node
            }
            cols_to_sum = mapped_costs + [
                "Generation [GWh]", "Storage [GWh]", "Transmission [GWh]", "Curtailment [GWh]", "Total Cost [M$/yr]"
            ]
            for c in cols_to_sum:
                agg[c] = df_sub[c].sum()

            gen, stor, trans = agg["Generation [GWh]"], agg["Storage [GWh]"], agg["Transmission [GWh]"]

            # Weighted sums for Levelised Costs
            agg["LCOG [$/MWh]"] = (df_sub["LCOG [$/MWh]"] * df_sub["Generation [GWh]"]).sum() / gen if gen > 1e-6 else 0.0
            agg["LCOS [$/MWh]"] = (df_sub["LCOS [$/MWh]"] * df_sub["Storage [GWh]"]).sum() / stor if stor > 1e-6 else 0.0
            agg["LCOT [$/MWh]"] = (df_sub["LCOT [$/MWh]"] * df_sub["Transmission [GWh]"]).sum() / trans if trans > 1e-6 else 0.0

            if is_system:
                agg["LCOE [$/MWh]"] = (
                    (agg["Total Cost [M$/yr]"] * 1e6 * year_count) / total_demand_mwh
                    if total_demand_mwh > 1e-6 else 0.0
                )
            else:
                agg["LCOE [$/MWh]"] = 0.0

            return pd.Series(agg)

        nodes_list = []
        for node_val, group in df_merged.groupby("Node"):
            if pd.notna(node_val) and str(node_val).lower() not in ("system", "network"):
                nodes_list.append(aggregate_rows(group, node_val, "Node", "Node", node_val))

        df_nodes = pd.DataFrame(nodes_list) if nodes_list else pd.DataFrame()
        df_system = pd.DataFrame([aggregate_rows(df_merged, "System", "System", "System", "System", is_system=True)])

        # Remove the base node assets to prevent duplication (we use df_nodes instead)
        mask_non_base_node = df_merged["Asset Class"].astype(str).str.lower() != "nodes"
        df_assets = df_merged[mask_non_base_node].copy()

        df_final = pd.concat([df_system, df_nodes, df_assets], ignore_index=True)

        # Force Left-to-Right layout: System (0), Nodal aggregates (1), Assets (2), Lines (3)
        def get_sort_category(row):
            if row["Asset Name"] == "System":
                return 0
            elif str(row["Asset Class"]).lower() == "major_lines" or str(row["Asset Type"]).lower() in ("line", "major line"):
                return 3
            elif row["Asset Type"] == "Node":
                return 1
            else:
                return 2

        df_final['_sort_cat'] = df_final.apply(get_sort_category, axis=1)
        df_final['_node_val'] = df_final['Node'].replace("None", "zzzz")
        df_final['_asset_type'] = df_final['Asset Type']
        df_final['_asset_name'] = df_final['Asset Name']

        df_final = df_final.sort_values(['_sort_cat', '_node_val', '_asset_type', '_asset_name']).drop(
            columns=['_sort_cat', '_node_val', '_asset_type', '_asset_name']
        )

        keep_cols = [
            "Asset Name", "Asset Type", "Unit Type", "Node", "Total Cost [M$/yr]",
            "Annualised Build [M$/yr]", "Fixed O&M [$M/yr]", "Variable O&M [M$/yr]", "Fuel Cost [M$/yr]",
            "Generation [GWh]", "Storage [GWh]", "Transmission [GWh]", "Curtailment [GWh]",
            "LCOG [$/MWh]", "LCOS [$/MWh]", "LCOT [$/MWh]", "LCOE [$/MWh]"
        ]

        df_final = df_final[keep_cols].T

        return ResultFile("levelised_costs", self.results_directory, df_final, decimals=3)

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
