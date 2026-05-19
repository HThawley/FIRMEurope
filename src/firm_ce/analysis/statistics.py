# type: ignore
import os
import shutil
import time

from re import sub
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.common.typing import npfloat
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
        x_candidate: NDArray[npfloat],
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
            x_candidate.astype(npfloat), parameters_static, fleet_static, network_static, balancing_type, fixed_costs_threshold
        )

        start_time = time.time()
        evaluate(self.solution)
        end_time = time.time()
        print(f"Statistics solution evaluation time: {end_time - start_time:.4f} seconds")
        print(f"{scenario_name} LCOE: {self.solution.lcoe} [$/MWh], " f"Penalties: {self.solution.penalties}")

        self.results_directory = self.create_solution_directory(
            solution_results_directory, f"{scenario_name}_{balancing_type}"
        )
        self.copy_temp_files(copy_callback)
        self.result_files = None

        self.full_intervals_count = self.solution.static.block_lengths.sum()
        self.block_first_intervals, self.block_last_intervals = static_m.get_block_intervals(
            self.solution.static.block_lengths
        )

        self.result_files = {}
        self.master_tables_built = False
        self.statistics_generated = False

    def _build_master_tables(self):
        accessor = Accessor(self.solution, "GW")
        static_data = []

        self.temporal_file_path = os.path.join(self.results_directory, "temporal_data.parquet")
        if os.path.exists(self.temporal_file_path):
            os.remove(self.temporal_file_path)

        schema = pa.schema([
            ('Time_Step', pa.int32()),
            ('Asset Name', pa.string()),
            ('Asset Type', pa.string()),
            ('Unit Type', pa.string()),
            ('Node', pa.string()),
            ('Variable', pa.string()),
            ('Value', pa.float32())  # Downcast to float32
        ])
        writer = pq.ParquetWriter(self.temporal_file_path, schema)
        time_steps = np.arange(self.full_intervals_count, dtype=np.int32)
        n_steps = len(time_steps)

        def write_trace(meta, variable_name, trace_array):
            # Convert to pyarrow arrays to bypass Pandas overhead
            table = pa.Table.from_arrays([
                time_steps,
                pa.array([meta[1]] * n_steps),  # Name
                pa.array([meta[2]] * n_steps),  # Type
                pa.array([meta[4]] * n_steps),  # Unit Type
                pa.array([meta[5]] * n_steps),  # Node
                pa.array([variable_name] * n_steps),
                pa.array(trace_array.astype(np.float32))
            ], schema=schema)
            writer.write_table(table)

        asset_classes = ["nodes", "generators", "storages", "major_lines"]  # , "minor_lines"]
        meta_data_names = ("Asset ID", "Asset Name", "Asset Type", "Asset Class", "Unit Type", "Node")
        power_build_types = ("Existing Power", "New Build Power", "Min Build Power", "Max Build Power")
        energy_build_types = ("Existing Energy", "New Build Energy", "Min Build Energy", "Max Build Energy")

        for asset_class in asset_classes:
            is_node = asset_class == "nodes"
            assets = accessor.get_assets(asset_class)
            for asset in assets.values():
                meta_data = (
                    asset.id, asset.name, accessor.get_display_name(asset_class), asset_class,
                    getattr(asset, "unit_type", "node" if is_node else None),
                    asset.node.name if hasattr(asset, "node") else (asset.name if is_node else None),
                )
                row = dict(zip(meta_data_names, meta_data))

                row.update(accessor.get_all_costs(asset, errors="coerce"))

                row["Power Capacity"] = accessor.get_power_capacity(asset, errors="coerce")
                row["Energy Capacity"] = accessor.get_energy_capacity(asset, errors="coerce")
                row.update(dict(zip(power_build_types, accessor.get_build_power(asset, errors="coerce"))))
                row.update(dict(zip(energy_build_types, accessor.get_build_energy(asset, errors="coerce"))))
                static_data.append(row)

                if accessor.is_node(asset):
                    write_trace(meta_data, "Demand", accessor.get_power_trace(asset))
                    write_trace(meta_data, "Spillage", accessor.get_spillage_trace(asset))
                    write_trace(meta_data, "Deficit", accessor.get_deficit_trace(asset))

                # elif accessor.is_major_line(asset):
                elif accessor.is_line(asset):
                    write_trace(meta_data, "Flow", accessor.get_transmission_trace(asset))

                else:
                    # For Generators, and Storage
                    write_trace(meta_data, "Dispatch", accessor.get_power_trace(asset))

                    # Batteries / Storage
                    if accessor.is_storage(asset):
                        write_trace(meta_data, "Stored_Energy", accessor.get_storage_level_trace(asset))
                        write_trace(meta_data, "Charge", accessor.get_charge_trace(asset))
                        write_trace(meta_data, "Discharge", accessor.get_discharge_trace(asset))

                    if accessor.has_inflows(asset):
                        write_trace(meta_data, "Inflows", accessor.get_inflow_trace(asset))

        for asset in accessor.get_assets("fuels").values():
            meta_data = (asset.id, asset.name, accessor.get_display_name("fuels"), "fuels", "fuel", "network")
            row = dict(zip(meta_data_names, meta_data))
            write_trace(meta_data, "Fuel_Remaining", accessor.get_remaining_energy_trace(asset))

        writer.close()

        df_static = pd.DataFrame(static_data)
        if not df_static.empty:
            node_mask = df_static["Asset Class"] == "nodes"
            for column in (
                "Power Capacity", "Energy Capacity", "Existing Power", "Existing Energy", "New Build Power",
                "Min Build Power", "Max Build Power", "New Build Energy", "Min Build Energy", "Max Build Energy",
                "Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"
            ):
                nodal_values = df_static[
                    df_static["Asset Type"].isin(("Generator", "Storage"))
                ].fillna(0.0).groupby("Node")[column].sum()

                df_static.loc[node_mask, column] = df_static.loc[node_mask, "Asset Name"].map(nodal_values)

            df_static.set_index("Asset ID", inplace=True)
            # Fill NaNs for assets that missed certain optional fields (like costs for nodes)
            # df_static.fillna(0.0, inplace=True)

        self.master_tables_built = True

        return df_static

    def create_solution_directory(self, result_directory: str, solution_name: str) -> str:
        safe_name = sub(r"[^a-zA-Z0-9_\-]", "_", solution_name)
        solution_dir = os.path.join(result_directory, safe_name)
        os.makedirs(solution_dir, exist_ok=True)
        return solution_dir

    def copy_temp_files(self, copy_callback: bool) -> None:
        if copy_callback:
            temp_dir = os.path.join("results", "temp")
            if os.path.exists(os.path.join(temp_dir, "callback.csv")):
                shutil.copy(
                    os.path.join(temp_dir, "callback.csv"),
                    os.path.join(self.results_directory, "callback.csv")
                )

            if SAVE_POPULATION:
                for file in ("latest_population.csv", "population.csv", "population_energies.csv",
                             "details.csv", "latest_details.csv"):
                    temp_path = os.path.join(temp_dir, file)
                    if os.path.exists(temp_path):
                        shutil.copy(
                            temp_path,
                            os.path.join(self.results_directory, file),
                        )
        return None

    def generate_result_files(self, file='all', write=True, delete=True) -> None:
        """
        Generates all result files using the high-level master DataFrames.
        """
        if not self.master_tables_built:
            # Ensure master tables are built if not already done
            if not hasattr(self, "df_static"):
                self.df_static = self._build_master_tables()

        file_functions = {
            "x": self.generate_x_file,
            "summary": self._view_summary,
            "capacities": self._view_capacities,
            "component_costs": self._view_component_costs,
            "levelised_costs": self._view_levelised_costs,
            "energy_balance_NETWORK": self._view_energy_balance_network,
            "energy_balance_NODES": self._view_energy_balance_nodes,
            "energy_balance_ASSETS": self._view_energy_balance_assets,
        }

        for name, func in file_functions.items():
            if name in file or file == 'all':
                self.result_files[name] = func()
            if write:
                self.result_files[name].write()
            if delete:
                del self.result_files[name]

        self.statistics_generated = True
        return None

    def write_results(self) -> None:
        if not self.statistics_generated:
            raise RuntimeError("Statistics must be generated before writing results.")

        for result_file in self.result_files.values():
            result_file.write()

        return None

    def _apply_standard_sort(
        self,
        frame: pl.LazyFrame | pl.DataFrame,
        index_cols: list[str] = None,
        sort_variable_rows: bool = False,
        sort_variable_columns: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Sorts rows by standard hierarchy: Node -> Asset Type -> Asset Name.
        Optionally sorts Variable columns into a standardized horizontal order.
        Operates on the pivoted format where assets are rows.
        """
        is_lazy = isinstance(frame, pl.LazyFrame)
        lf = frame if is_lazy else frame.lazy()

        lf = lf.with_columns(
            pl.when(pl.col("Node").is_null()).then(pl.lit("zzzz_lines"))
              .when(pl.col("Node").str.to_lowercase() == "system").then(pl.lit("0000_system"))
              .otherwise(pl.col("Node")).alias("_node_sort"),

            pl.when(pl.col("Asset Type") == "Node").then(1)
              .when(pl.col("Asset Type") == "Generator").then(2)
              .when(pl.col("Asset Type") == "Storage").then(3)
              .when(pl.col("Asset Type").str.to_lowercase().str.contains("line")).then(4)
              .otherwise(999).alias("_asset_sort")
        )

        sort_by = ["_node_sort", "_asset_sort", "Asset Name"]
        drop_cols = ["_node_sort", "_asset_sort"]

        var_order = [
            'Demand', 'Deficit', 'Spillage', 'Dispatch', 'Flow', 'Line_Input_Power',
            'Line_Output_Power', 'Net_Imports', 'Net_Exports', 'Power_Into_Lines',
            'Power_Out_Of_Lines', 'Discharge', 'Charge', 'Inflows', 'Stored_Energy', 'Fuel_Remaining'
        ]

        if sort_variable_rows:
            mapping_lf = pl.LazyFrame({
                "Variable": var_order,
                "_var_sort": list(range(len(var_order)))
            }).with_columns(pl.col("_var_sort").cast(pl.UInt32))

            lf = lf.join(mapping_lf, on="Variable", how="left").with_columns(pl.col("_var_sort").fill_null(9999))

            sort_by.append("_var_sort")
            drop_cols.append("_var_sort")

        lf = lf.sort(sort_by).drop(drop_cols)

        if sort_variable_columns:
            if index_cols is None:
                index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]

            # Extract current schema names directly from the computation graph
            current_cols = lf.collect_schema().names()

            ordered_vars = [v for v in var_order if v in current_cols] + \
                           [v for v in current_cols if v not in var_order and v not in index_cols]

            lf = lf.select(index_cols + ordered_vars)

        return lf if is_lazy else lf.collect()

    def _view_capacities(self) -> ResultFile:
        """
        View: Static Capacity Data.
        Units: MW -> GW (Output)
        """
        # Select relevant columns from master static table
        string_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]

        numeric_cols = [
            "Power Capacity", "Energy Capacity", "Existing Power", "Existing Energy", "New Build Power",
            "New Build Energy", "Min Build Power", "Min Build Energy", "Max Build Power", "Max Build Energy",
        ]

        df = pl.from_pandas(self.df_static.reset_index()).select(string_cols + numeric_cols)
        df = self._apply_standard_sort(df, index_cols=string_cols)

        df = df.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in string_cols], separator="|").alias("_asset_string")
        ).drop(string_cols)

        df = df.transpose(include_header=True, header_name="Metric", column_names="_asset_string")

        return ResultFile("capacities", self.results_directory, df.lazy(), decimals=3, write_kwargs={"multiindex_delimiter": "|"})

    def _view_component_costs(self) -> ResultFile:
        """
        View: Asset Costs.
        Units: $ (No conversion needed)
        """
        string_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
        cost_cols = ["Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"]

        df = pl.from_pandas(self.df_static.reset_index())

        # Ensure all cost cols exist
        for c in cost_cols:
            if c not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(c))

        df = df.select(string_cols + cost_cols + ["Power Capacity"])
        df = df.with_columns(pl.sum_horizontal(cost_cols).alias("Total Cost"))

        # Reorder and Convert to M$
        all_numeric = ["Total Cost"] + cost_cols
        df = df.with_columns([(pl.col(c) / 1e6).alias(c) for c in all_numeric])

        # Calc $/kW/year
        df = df.with_columns([
            (pl.col(c) / pl.col("Power Capacity")).alias(f"{c} ($/kW/year)") for c in all_numeric
        ])

        # Filter out 0 total cost (Nodes) and rename
        df = (
            df.filter(pl.col("Total Cost") > 1e-6)
            .rename({c: f"{c} (M$/year)" for c in all_numeric})
            .drop("Power Capacity")
        )

        df = self._apply_standard_sort(df, index_cols=string_cols)
        df = df.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in string_cols], separator="|").alias("_asset_string")
        ).drop(string_cols)

        df = df.transpose(include_header=True, header_name="Metric", column_names="_asset_string")

        return ResultFile(
            "component_costs",
            self.results_directory,
            df.lazy(),
            decimals=3,
            write_kwargs={"multiindex_delimiter": "|"}
        )

    def _view_energy_balance_assets(self) -> ResultFile:
        return self._view_energy_balance("assets")

    def _view_energy_balance_nodes(self) -> ResultFile:
        return self._view_energy_balance("nodes")

    def _view_energy_balance_network(self) -> ResultFile:
        return self._view_energy_balance("network")

    def _view_energy_balance(self, aggregation: str) -> ResultFile:
        aggregation = aggregation.lower()
        index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]

        lf_all = pl.scan_parquet(self.temporal_file_path)
        lf_base = lf_all.filter(~pl.col("Variable").is_in(["Flow", "Fuel_Remaining"]))
        lf_fuel = lf_all.filter(pl.col("Variable") == "Fuel_Remaining")

        accessor = Accessor(self.solution, "GW")
        lines = accessor.get_assets('major_lines')
        line_data = []
        for a in lines.values():
            parts = str(a.name).split("-")
            line_data.append({
                "Asset Name": a.name,
                "Unit Type": a.unit_type,
                "Node_A": parts[0] if len(parts) > 0 else "Unknown",
                "Node_B": parts[1] if len(parts) > 1 else "Unknown",
                "Eff": getattr(a, 'efficiency', 1.0)
            })
        lf_lines = pl.LazyFrame(line_data, schema_overrides={"Eff": pl.Float32})
        lf_flow = lf_all.filter(pl.col("Variable") == "Flow").join(
            lf_lines, on=["Asset Name", "Unit Type"], how="left")

        if aggregation.lower() == "assets":
            # Positive f_val: A -> B. Node A exports (-f_val), Node B imports (+f_val * effs)
            # Negative f_val: B -> A. Node B exports (f_val), Node A imports (-f_val * effs)
            lf_A = lf_flow.with_columns(
                pl.col("Node_A").alias("Node"),
                pl.when(pl.col("Value") > 0).then(-pl.col("Value"))
                  .otherwise(-pl.col("Value") * pl.col("Eff")).alias("Value")
            ).drop(["Node_A", "Node_B", "Eff"])

            lf_B = lf_flow.with_columns(
                pl.col("Node_B").alias("Node"),
                pl.when(pl.col("Value") > 0).then(pl.col("Value") * pl.col("Eff"))
                  .otherwise(pl.col("Value")).alias("Value")
            ).drop(["Node_A", "Node_B", "Eff"])

            lf_main = pl.concat([lf_base, lf_A, lf_B, lf_fuel])

        elif aggregation == "nodes":
            node_meta = [pl.col("Node").alias("Asset Name"), pl.lit("Node").alias("Asset Type"), pl.lit("Node").alias("Unit Type")]

            lf_base_n = lf_base.group_by(["Time_Step", "Node", "Variable"]).agg(pl.col("Value").sum()).with_columns(node_meta)
            lf_fuel_n = lf_fuel.group_by(["Time_Step", "Node", "Variable"]).agg(pl.col("Value").sum()).with_columns(node_meta)

            # Define a strict 32-bit float zero
            zero_f32 = pl.lit(0.0, dtype=pl.Float32)

            # Positive f_val: A -> B. Node A exports (-f_val), Node B imports (+f_val * effs)
            # Negative f_val: B -> A. Node B exports (f_val), Node A imports (-f_val * effs)
            lf_exp_A = lf_flow.with_columns(
                pl.col("Node_A").alias("Node"),
                pl.lit("Net_Exports").alias("Variable"),
                pl.when(pl.col("Value") > 0).then(-pl.col("Value")).otherwise(zero_f32).alias("Value"))
            lf_imp_B = lf_flow.with_columns(
                pl.col("Node_B").alias("Node"),
                pl.lit("Net_Imports").alias("Variable"),
                pl.when(pl.col("Value") > 0).then(pl.col("Value") * pl.col("Eff")).otherwise(zero_f32).alias("Value")
            )
            lf_exp_B = lf_flow.with_columns(
                pl.col("Node_B").alias("Node"),
                pl.lit("Net_Exports").alias("Variable"),
                pl.when(pl.col("Value") < 0).then(pl.col("Value")).otherwise(zero_f32).alias("Value")
            )
            lf_imp_A = lf_flow.with_columns(
                pl.col("Node_A").alias("Node"),
                pl.lit("Net_Imports").alias("Variable"),
                pl.when(pl.col("Value") < 0).then(-pl.col("Value") * pl.col("Eff")).otherwise(zero_f32).alias("Value")
            )

            lf_flow_n = (
                pl.concat([lf_exp_A, lf_imp_B, lf_exp_B, lf_imp_A])
                  .group_by(["Time_Step", "Node", "Variable"])
                  .agg(pl.col("Value").sum()).with_columns(node_meta)
            )

            lf_main = pl.concat([lf_base_n, lf_flow_n, lf_fuel_n])

        elif aggregation.lower() == "network":
            net_meta = [pl.lit("Network").alias("Asset Name"),
                        pl.lit("Network").alias("Asset Type"),
                        pl.lit("Network").alias("Unit Type"),
                        pl.lit("System").alias("Node")
                        ]

            lf_base_net = lf_base.group_by(["Time_Step", "Variable"]).agg(pl.col("Value").sum()).with_columns(net_meta)

            lf_flow_net = (
                lf_flow
                .select([
                    pl.col("Time_Step"),
                    pl.col("Value").abs().alias("Power_Into_Lines"),
                    (pl.col("Value").abs() * pl.col("Eff")).cast(pl.Float32).alias("Power_Out_Of_Lines")
                ])
                .unpivot(index="Time_Step", variable_name="Variable", value_name="Value")
                .group_by(["Time_Step", "Variable"]).agg(pl.col("Value").sum())
                .with_columns(net_meta)
            )

            lf_main = pl.concat([lf_base_net, lf_flow_net])

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # 4. Extract and Sort Column Blueprints
        lf_meta = lf_main.select(index_cols + ["Variable"]).unique()
        df_meta = self._apply_standard_sort(
            lf_meta,
            index_cols=index_cols,
            sort_variable_rows=True,
            sort_variable_columns=False
        )
        df_meta = df_meta.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols]
                          + [pl.col("Variable")], separator="|").alias("_col")
        ).collect().get_column("_col").to_list()

        # 5. Pack metadata in the main dataframe and Pivot
        lf_main = lf_main.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols]
                          + [pl.col("Variable")], separator="|").alias("_col")
        ).select(["Time_Step", "_col", "Value"])

        df_main = lf_main.collect().pivot(
            values="Value",
            index="Time_Step",
            on="_col",
            aggregate_function="sum",  # aggregates identical assets
        ).fill_null(0.0)

        # 6. Apply strictly ordered columns and drop unneeded strings
        df_main = (
            df_main
            .select(["Time_Step"] + [c for c in df_meta if c in df_main.columns])
            .sort("Time_Step")
        )

        return ResultFile(
            f"energy_balance_{aggregation.upper()}",
            self.results_directory,
            df_main.lazy(),
            decimals=3,
            write_kwargs={"multiindex_delimiter": "|"}
        )

    def _view_summary(self) -> ResultFile:
        resolution = self.solution.static.resolution
        index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]

        # 1. Aggregation
        summary_df = (
            pl.scan_parquet(self.temporal_file_path)
              .filter(~pl.col("Variable").is_in(["Fuel_Remaining", "Stored_Energy"]))
              .group_by(index_cols + ["Variable"])
              .agg((pl.col("Value").abs().sum() * resolution).alias("Total_GWh"))
        )

        # 2. Materialize and Pivot (Variables become columns)
        summary_df = summary_df.collect().pivot(
            values="Total_GWh",
            index=index_cols,
            on="Variable",
            aggregate_function=None
        ).fill_null(0.0)

        # 3. Energy Adjustments
        if "Inflows" in summary_df.columns:
            summary_df = summary_df.with_columns(pl.col("Inflows") / resolution)
        # summary_df = summary_df.drop(['Fuel_Remaining', 'Stored_Energy'], strict=False)

        # 4. Sort Rows (Assets)
        summary_df = self._apply_standard_sort(
            summary_df,
            index_cols=index_cols,
            sort_variable_rows=False,
            sort_variable_columns=True,
        )

        # 5. Condense Metadata for Transpose
        summary_df = summary_df.with_columns(
            pl.concat_str(
                [pl.col(c).cast(pl.String).fill_null("None") for c in index_cols],
                separator="|"
            ).alias("_asset_string")
        ).drop(index_cols)

        summary_df = summary_df.transpose(
            include_header=True,
            header_name="Variable",
            column_names="_asset_string"
        )

        return ResultFile(
            "summary",
            self.results_directory,
            summary_df.lazy(),
            decimals=3,
            write_kwargs={"multiindex_delimiter": "|"}
        )

    def _view_levelised_costs(self) -> ResultFile:
        resolution = self.solution.static.resolution
        year_count = self.solution.static.year_count

        string_cols = ["Asset ID", "Asset Name", "Asset Type", "Asset Class", "Unit Type", "Node"]
        cost_cols = ["Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"]

        lf_all = pl.scan_parquet(self.temporal_file_path)

        df_nodal_demand = (
            lf_all.filter(pl.col("Variable") == "Demand")
            .group_by("Node")
            .agg((pl.col("Value").sum() * resolution * 1000).alias("Nodal_Demand_MWh"))
            .collect()
        )

        total_demand_mwh = df_nodal_demand["Nodal_Demand_MWh"].sum()
        total_demand_mwh = total_demand_mwh if total_demand_mwh is not None else 0.0

        df_totals = (
            lf_all.group_by(["Asset Name", "Unit Type", "Variable"])
            .agg((pl.col("Value").abs().sum() * resolution).alias("Total_GWh"))
            .collect()
            .pivot(values="Total_GWh", index=["Asset Name", "Unit Type"], on="Variable", aggregate_function=None)
            .fill_null(pl.lit(0.0))
        )
        # Inflows are natively energy, not power
        df_totals = df_totals.with_columns(
            (pl.col("Inflows") / resolution).alias("Inflows")
        )

        df_costs = pl.from_pandas(self.df_static.reset_index())
        for c in cost_cols:
            if c not in df_costs.columns:
                df_costs = df_costs.with_columns(pl.lit(0.0).alias(c))

        df_costs = df_costs.with_columns([
            (pl.col(c) / 1e6).alias(f"{c} [M$/yr]") for c in cost_cols
        ]).select(string_cols + [f"{c} [M$/yr]" for c in cost_cols])

        df_merged = df_costs.join(df_totals, on=["Asset Name", "Unit Type"], how="left").fill_null(0.0)

        # Ensure required temporal columns exist before mapping
        for v in ["Dispatch", "Inflows", "Spillage", "Flow"]:
            if v not in df_merged.columns:
                df_merged = df_merged.with_columns(pl.lit(0.0).alias(v))

        df_merged = df_merged.with_columns([
            pl.when(pl.col("Asset Type").str.to_lowercase() == "storage")
              .then(pl.col("Inflows")).otherwise(pl.col("Dispatch")).alias("Generation [GWh]"),
            pl.when(pl.col("Asset Type").str.to_lowercase() == "storage")
              .then(pl.col("Discharge")).otherwise(0.0).alias("Storage [GWh]"),
            pl.col("Flow").alias("Transmission [GWh]"),
            pl.col("Spillage").alias("Curtailment [GWh]")
        ])

        mapped_costs = [f"{c} [M$/yr]" for c in cost_cols]

        df_merged = df_merged.with_columns([
            pl.when(pl.col("Asset Class").str.to_lowercase() == "nodes")
              .then(0.0).otherwise(pl.col(c)).alias(c) for c in mapped_costs
        ])

        df_merged = df_merged.with_columns(pl.sum_horizontal(mapped_costs).alias("Total Cost [M$/yr]"))

        def calc_lco(cost_col, energy_col):
            # Helper to calculate Levelised Cost ($/MWh = M$ * 1000 / GWh)
            return (
                pl.when(pl.col(energy_col) > 1e-6)
                  .then((pl.col(cost_col) * year_count * 1000) / pl.col(energy_col))
                  .otherwise(0.0)
            )

        df_merged = df_merged.with_columns([
            calc_lco("Total Cost [M$/yr]", "Generation [GWh]").alias("LCOG [$/MWh]"),
            calc_lco("Total Cost [M$/yr]", "Storage [GWh]").alias("LCOS [$/MWh]"),
            calc_lco("Total Cost [M$/yr]", "Transmission [GWh]").alias("LCOT [$/MWh]"),
            pl.lit(0.0).alias("LCOE [$/MWh]")
        ])

        cols_to_sum = mapped_costs + ["Generation [GWh]", "Storage [GWh]", "Transmission [GWh]",
                                      "Curtailment [GWh]", "Total Cost [M$/yr]"]

        # Base math logic for weighted averages
        def weighted_lco(lco_col, energy_col):
            total_weighted_cost = (pl.col(lco_col) * pl.col(energy_col)).sum()
            total_energy = pl.col(energy_col).sum()
            return (total_weighted_cost / total_energy).fill_nan(0.0)

        df_nodes = (
            df_merged.filter(pl.col("Node").is_not_null()
                             & (pl.col("Node").str.to_lowercase() != "system")
                             & (pl.col("Node").str.to_lowercase() != "network"))
            .group_by("Node").agg([
                pl.sum(c) for c in cols_to_sum
            ] + [
                weighted_lco("LCOG [$/MWh]", "Generation [GWh]").alias("LCOG [$/MWh]"),
                weighted_lco("LCOS [$/MWh]", "Storage [GWh]").alias("LCOS [$/MWh]"),
                weighted_lco("LCOT [$/MWh]", "Transmission [GWh]").alias("LCOT [$/MWh]"),
            ]).with_columns([
                pl.col("Node").alias("Asset Name"), pl.lit("Node").alias("Asset Type"), pl.lit("Node").alias("Unit Type")
            ])
        )

        df_nodes = df_nodes.join(df_nodal_demand, on="Node", how="left").fill_null(0.0)
        df_nodes = df_nodes.with_columns([
            pl.when(pl.col("Nodal_Demand_MWh") > 1e-6)
              .then((pl.col("Total Cost [M$/yr]") * 1e6 * year_count) / pl.col("Nodal_Demand_MWh"))
              .otherwise(0.0).alias("LCOE [$/MWh]")
        ]).drop("Nodal_Demand_MWh")

        sys_lcoe = (pl.col("Total Cost [M$/yr]").sum() * 1e6 * year_count / total_demand_mwh)

        df_system = (
            df_merged.select([
                pl.sum(c) for c in cols_to_sum
            ] + [
                weighted_lco("LCOG [$/MWh]", "Generation [GWh]").alias("LCOG [$/MWh]"),
                weighted_lco("LCOS [$/MWh]", "Storage [GWh]").alias("LCOS [$/MWh]"),
                weighted_lco("LCOT [$/MWh]", "Transmission [GWh]").alias("LCOT [$/MWh]")
            ]).with_columns([
                pl.lit("System").alias("Asset Name"), pl.lit("System").alias("Asset Type"),
                pl.lit("System").alias("Unit Type"), pl.lit("System").alias("Node"),
                sys_lcoe.alias("LCOE [$/MWh]")
            ])
        )

        df_assets = df_merged.filter(pl.col("Asset Class").str.to_lowercase() != "nodes")

        keep_cols = ["Asset Name", "Asset Type", "Unit Type", "Node", "Total Cost [M$/yr]"] + mapped_costs + [
            "Generation [GWh]", "Storage [GWh]", "Transmission [GWh]", "Curtailment [GWh]",
            "LCOG [$/MWh]", "LCOS [$/MWh]", "LCOT [$/MWh]", "LCOE [$/MWh]"
        ]

        df_final = pl.concat([
            df_system.select(keep_cols),
            df_nodes.select(keep_cols),
            df_assets.select(keep_cols)
        ], how="vertical")

        index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
        df_final = self._apply_standard_sort(df_final, index_cols=index_cols)
        df_final = df_final.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols], separator="|").alias("_asset_string")
        ).drop(index_cols)

        df_final = df_final.transpose(include_header=True, header_name="Metric", column_names="_asset_string")

        return ResultFile(
            "levelised_costs",
            self.results_directory,
            df_final.lazy(),
            decimals=3,
            write_kwargs={"multiindex_delimiter": "|"}
        )

    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile(
            "x", self.results_directory, pd.DataFrame(self.solution.x).T, write_kwargs={"index": False}, decimals=3
        )
        return result_file
