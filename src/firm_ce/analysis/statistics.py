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
from firm_ce.analysis.accessor import Accessor
from firm_ce.io.file_manager import ResultFile
from firm_ce.optimisation.st_solution import Solution, evaluate
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType


class Statistics:
    def __init__(
        self,
        solution_results_directory: str,
        scenario_name: str,
        x_candidate: NDArray[npfloat] = None,
        parameters_static: ScenarioParameters_InstanceType = None,
        fleet_static: Fleet_InstanceType = None,
        network_static: Network_InstanceType = None,
        balancing_type: str = None,
        fixed_costs_threshold: float = None,
        copy_callback: bool = True,
        *,
        solution: Solution = None,
    ):
        self.scenario_name = scenario_name
        self.balancing_type = balancing_type

        if solution is not None:
            self.solution = solution
            self.balancing_type = solution.balancing_type
            if not getattr(self.solution, "evaluated", False):
                self._evaluate_solution()
        else:
            self.solution = Solution(
                x_candidate.astype(npfloat), parameters_static, fleet_static, network_static, balancing_type, fixed_costs_threshold
            )
            self._evaluate_solution()

        self.statistics_dir = self.create_solution_directory(
            solution_results_directory,
            f"statistics/{self.scenario_name}_{self.balancing_type}"
        )
        self.copy_temp_files(copy_callback)
        self.result_files = None

        self.intervals_count = self.solution.static.intervals_count

        self.result_files = {}
        self.master_tables_built = False
        self.statistics_generated = False

    def _evaluate_solution(self):
        start_time = time.time()
        evaluate(self.solution)
        end_time = time.time()
        print(f"Statistics solution evaluation time: {end_time - start_time:.4f} seconds")
        print(f"{self.scenario_name} LCOE: {self.solution.lcoe} [$/MWh], " f"Penalties: {self.solution.penalties}")

    def _build_master_tables(self):
        accessor = Accessor(self.solution, "GW")
        static_data = []

        self.temporal_file_path = os.path.join(self.statistics_dir, "temporal_data.parquet")
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
        time_steps = np.arange(self.intervals_count, dtype=np.int32)
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
                    os.path.join(self.statistics_dir, "callback.csv")
                )

            if SAVE_POPULATION:
                for file in ("latest_population.csv", "population.csv", "population_energies.csv",
                             "details.csv", "latest_details.csv"):
                    temp_path = os.path.join(temp_dir, file)
                    if os.path.exists(temp_path):
                        shutil.copy(
                            temp_path,
                            os.path.join(self.statistics_dir, file),
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
            "nodal_capacity_matrix": self._view_nodal_capacity_matrix,
            "summary_ASSETS": self._view_summary_assets,
            "summary_NODES": self._view_summary_nodes,
            "capacities_ASSETS": self._view_capacities_assets,
            "capacities_NODES": self._view_capacities_nodes,
            "capacities_UNIT_TYPES": self._view_capacities_unit_types,
            "components_ASSETS": self._view_component_costs_assets,
            "components_NODES": self._view_component_costs_nodes,
            "levelised_cost_ASSETS": self._view_levelised_cost_assets,
            "levelised_cost_NODES": self._view_levelised_cost_nodes,
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

    def _view_nodal_capacity_matrix(self) -> ResultFile:
        gw_cols = ["Power Capacity"]
        gwh_cols = ["Energy Capacity"]

        df_gen_stor = self._get_base_capacity_df(["Node", "Asset Type", "Unit Type"], gw_cols + gwh_cols)
        df_lines = self._get_lines_capacity_df("double_counted", gw_cols, gwh_cols)

        df_all = pl.concat([df_gen_stor, df_lines], how="vertical")

        df_agg = df_all.group_by("Node").agg([
            pl.when(pl.col("Asset Type") == "Generator"
                    ).then(pl.col("Power Capacity")).otherwise(0.0).sum().alias("Generation (GW)"),
            pl.when(pl.col("Asset Type") == "Storage"
                    ).then(pl.col("Power Capacity")).otherwise(0.0).sum().alias("Storage Power (GW)"),
            pl.when(pl.col("Asset Type") == "Storage"
                    ).then(pl.col("Energy Capacity")).otherwise(0.0).sum().alias("Storage Energy (GWh)"),
            pl.when(pl.col("Asset Type") == "Line"
                    ).then(pl.col("Power Capacity")).otherwise(0.0).sum().alias("Transmission (GW)")
        ])

        df_pivot = df_all.pivot(values="Power Capacity", index="Node", on="Unit Type", aggregate_function="sum").fill_null(0.0)
        df_matrix = df_agg.join(df_pivot, on="Node", how="left").fill_null(0.0)

        gen_units = sorted([u for u in df_all.filter(pl.col("Asset Type") == "Generator"
                                                     ).select("Unit Type").unique().to_series().to_list() if u])
        stor_units = sorted([u for u in df_all.filter(pl.col("Asset Type") == "Storage"
                                                      ).select("Unit Type").unique().to_series().to_list() if u])
        line_units = sorted([u for u in df_all.filter(pl.col("Asset Type") == "Line"
                                                      ).select("Unit Type").unique().to_series().to_list() if u])

        agg_cols = ["Generation (GW)", "Storage Power (GW)", "Storage Energy (GWh)", "Transmission (GW)"]
        ordered_cols = ["Node"] + agg_cols + gen_units + stor_units + line_units
        df_matrix = df_matrix.select(ordered_cols).sort("Node")

        network_row = df_matrix.select([
            pl.lit("Network").alias("Node"),
            pl.col("Generation (GW)").sum(),
            pl.col("Storage Power (GW)").sum(),
            pl.col("Storage Energy (GWh)").sum(),
            (pl.col("Transmission (GW)").sum() / 2).alias("Transmission (GW)")
        ] + [pl.col(u).sum() for u in gen_units + stor_units]
          + [(pl.col(u).sum() / 2).alias(u) for u in line_units])

        df_matrix = pl.concat([network_row, df_matrix], how="vertical")

        return ResultFile("nodal_capacity_matrix", self.statistics_dir, df_matrix.lazy(), decimals=3)

    def _view_capacities_assets(self):
        return self._view_capacities(aggregation="assets")

    def _view_capacities_nodes(self):
        return self._view_capacities(aggregation="nodes")

    def _view_capacities(self, aggregation="assets") -> ResultFile:
        gw_cols = ["Power Capacity", "Existing Power", "New Build Power", "Min Build Power", "Max Build Power"]
        gwh_cols = ["Energy Capacity", "Existing Energy", "New Build Energy", "Min Build Energy", "Max Build Energy"]
        all_numeric = gw_cols + gwh_cols

        if aggregation == "nodes":
            index_cols = ["Node", "Asset Type", "Unit Type"]
            df_gen_stor = self._get_base_capacity_df(index_cols, all_numeric)
            df_lines = self._get_lines_capacity_df("double_counted", gw_cols, gwh_cols)
            df_all = pl.concat([df_gen_stor, df_lines], how="vertical")

            agg_exprs = []
            for c in all_numeric:
                unit = "(GW)" if c in gw_cols else "(GWh)"
                agg_exprs.append(pl.col(c).sum().alias(f"Total {c} {unit}"))
            for c in gw_cols:
                agg_exprs.append(pl.when(pl.col("Asset Type") == "Generator"
                                         ).then(pl.col(c)).otherwise(0.0).sum().alias(f"Generation {c} (GW)"))
            for c in all_numeric:
                unit = "(GW)" if c in gw_cols else "(GWh)"
                agg_exprs.append(pl.when(pl.col("Asset Type") == "Storage"
                                         ).then(pl.col(c)).otherwise(0.0).sum().alias(f"Storage {c} {unit}"))
            for c in gw_cols:
                agg_exprs.append(pl.when(pl.col("Asset Type") == "Line"
                                         ).then(pl.col(c)).otherwise(0.0).sum().alias(f"Transmission {c} (GW)"))

            df = df_all.group_by("Node").agg(agg_exprs)

            network_exprs = [pl.lit("Network").alias("Node")]
            for c in all_numeric:
                unit = "(GW)" if c in gw_cols else "(GWh)"
                tot_col = f"Total {c} {unit}"
                if c in gw_cols:
                    network_exprs.append((pl.col(tot_col).sum() - (pl.col(f"Transmission {c} (GW)").sum() / 2)).alias(tot_col))
                else:
                    network_exprs.append(pl.col(tot_col).sum().alias(tot_col))

            for c in gw_cols: network_exprs.append(pl.col(f"Generation {c} (GW)").sum().alias(f"Generation {c} (GW)"))
            for c in all_numeric:
                unit = "(GW)" if c in gw_cols else "(GWh)"
                network_exprs.append(pl.col(f"Storage {c} {unit}").sum().alias(f"Storage {c} {unit}"))
            for c in gw_cols: network_exprs.append((pl.col(f"Transmission {c} (GW)").sum() / 2).alias(f"Transmission {c} (GW)"))

            network_row = df.select(network_exprs).select(df.columns)
            df = pl.concat([network_row, df], how="vertical")
        else:
            index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
            df = self._get_base_capacity_df(index_cols, all_numeric)

        rename_map = None
        if aggregation == "assets":
            rename_map = {c: f"{c} (GW)" for c in gw_cols}
            rename_map.update({c: f"{c} (GWh)" for c in gwh_cols})

        return self._format_and_transpose_view(
            df,
            aggregation,
            index_cols=["Node"] if aggregation == "nodes" else index_cols,
            header_name="Metric",
            file_name=f"capacities_{aggregation.upper()}",
            rename_mapping=rename_map,
        )

    def _view_capacities_unit_types(self) -> ResultFile:
        gw_cols = ["Power Capacity", "Existing Power", "New Build Power", "Min Build Power", "Max Build Power"]
        gwh_cols = ["Energy Capacity", "Existing Energy", "New Build Energy", "Min Build Energy", "Max Build Energy"]
        all_numeric = gw_cols + gwh_cols

        df_gen_stor = self._get_base_capacity_df(["Asset Type", "Unit Type"], all_numeric)
        df_lines = self._get_lines_capacity_df("single_counted", gw_cols, gwh_cols)

        df = pl.concat([df_gen_stor, df_lines], how="vertical")
        df = df.group_by("Unit Type").agg([pl.col(c).sum() for c in all_numeric])

        rename_map = {c: f"{c} (GW)" for c in gw_cols}
        rename_map.update({c: f"{c} (GWh)" for c in gwh_cols})

        return self._format_and_transpose_view(
            df, aggregation="unit_types", index_cols=["Unit Type"],
            header_name="Metric", file_name="capacities_UNIT_TYPES", rename_mapping=rename_map
        )

    def _view_component_costs_assets(self):
        return self._view_component_costs(aggregation="assets")

    def _view_component_costs_nodes(self):
        return self._view_component_costs(aggregation="nodes")

    def _view_component_costs(self, aggregation) -> ResultFile:
        cost_cols = ["Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"]
        df_assets = pl.from_pandas(self.df_static.reset_index())

        for c in cost_cols:
            if c not in df_assets.columns:
                df_assets = df_assets.with_columns(pl.lit(0.0).alias(c))

        if aggregation == "nodes":
            # 1. Base Generation and Storage
            df_gen_stor = df_assets.filter(
                pl.col("Asset Type").is_in(["Generator", "Storage"]) & pl.col("Node").is_not_null()
            ).select(["Node", "Asset Type", "Power Capacity"] + cost_cols).fill_null(0.0)

            # 2. Extract and apportion Lines 50/50
            accessor = Accessor(self.solution, "GW")
            line_rows = []
            for line in accessor.get_assets("major_lines").values():
                cap = accessor.get_power_capacity(line, errors="coerce")
                if pd.isna(cap): cap = 0.0

                costs = accessor.get_all_costs(line, errors="coerce")

                row_base = {"Asset Type": "Line", "Power Capacity": cap / 2.0}
                for c in cost_cols:
                    val = costs.get(c, 0.0)
                    row_base[c] = (0.0 if pd.isna(val) else val) / 2.0

                line_rows.append({**row_base, "Node": line.node_start.name})
                line_rows.append({**row_base, "Node": line.node_end.name})

            df_lines = pl.DataFrame(line_rows, schema=df_gen_stor.schema)
            df_all = pl.concat([df_gen_stor, df_lines], how="vertical")
            df_all = df_all.with_columns(pl.sum_horizontal(cost_cols).alias("Total Cost"))

            all_cost_cols = ["Total Cost"] + cost_cols

            # 3. Build Aggregation Expressions
            agg_exprs = []
            for c in all_cost_cols:
                agg_exprs.append(pl.col(c).sum().alias(f"Total {c}"))
            agg_exprs.append(pl.col("Power Capacity").sum().alias("Total Power Capacity"))

            for atype, prefix in [("Generator", "Generation"), ("Storage", "Storage Power"), ("Line", "Transmission")]:
                for c in all_cost_cols:
                    agg_exprs.append(pl.when(pl.col("Asset Type") == atype
                                             ).then(pl.col(c)).otherwise(0.0).sum().alias(f"{prefix} {c}"))
                agg_exprs.append(pl.when(pl.col("Asset Type") == atype
                                         ).then(pl.col("Power Capacity")).otherwise(0.0).sum().alias(f"{prefix} Power Capacity"))

            df = df_all.group_by("Node").agg(agg_exprs).sort("Node")

            # 4. Calculate Network Row (System Total)
            # Since lines were apportioned 50/50, summing the nodes directly yields the correct system total
            network_exprs = [pl.lit("Network").alias("Node")]
            for col in df.columns:
                if col != "Node":
                    network_exprs.append(pl.col(col).sum().alias(col))

            network_row = df.select(network_exprs)
            df = pl.concat([network_row, df], how="vertical")

            # 5. Format to M$ and $/kW/year across the duplicated blocks
            final_exprs = [pl.col("Node")]
            prefixes = ["Total", "Generation", "Storage Power", "Transmission"]

            for p in prefixes:
                for c in all_cost_cols:
                    base_col = f"{p} {c}"
                    cap_col = f"{p} Power Capacity"

                    # Clean up the naming so "Total Total Cost" becomes just "Total Cost"
                    out_col = base_col.replace("Total Total Cost", "Total Cost")

                    final_exprs.append((pl.col(base_col) / 1e6).alias(f"{out_col} (M$/year)"))
                    final_exprs.append(
                        pl.when(pl.col(cap_col) > 1e-6)
                          .then((pl.col(base_col) / 1e6) / pl.col(cap_col))
                          .otherwise(0.0)
                          .alias(f"{out_col} ($/kW/year)")
                    )

            df = df.select(final_exprs)

            return self._format_and_transpose_view(
                df, aggregation="nodes", index_cols=["Node"], header_name="Metric", file_name="components_NODES"
            )

        else:
            # Asset logic remains unchanged
            index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
            df = df_assets.select(index_cols + cost_cols + ["Power Capacity"]).fill_null(0.0)
            df = df.with_columns(pl.sum_horizontal(cost_cols).alias("Total Cost"))
            all_numeric = ["Total Cost"] + cost_cols

            df = df.with_columns([(pl.col(c) / 1e6).alias(f"{c} (M$/year)") for c in all_numeric])
            df = df.with_columns([
                pl.when(pl.col("Power Capacity") > 1e-6)
                  .then(pl.col(f"{c} (M$/year)") / pl.col("Power Capacity"))
                  .otherwise(0.0)
                  .alias(f"{c} ($/kW/year)") for c in all_numeric
            ])

            df = df.filter(pl.col("Total Cost (M$/year)") > 1e-6).drop("Power Capacity")

            return self._format_and_transpose_view(
                df, aggregation, index_cols, header_name="Metric", file_name="components_ASSETS"
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
            self.statistics_dir,
            df_main.lazy(),
            decimals=3,
            write_kwargs={"multiindex_delimiter": "|"}
        )

    def _view_summary_assets(self):
        return self._view_summary(aggregation="assets")

    def _view_summary_nodes(self):
        return self._view_summary(aggregation="nodes")

    def _view_summary(self, aggregation="assets") -> ResultFile:
        resolution = self.solution.static.resolution
        year_count = getattr(self.solution.static, "year_count", 1.0)
        index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"] if aggregation == "assets" else ["Node"]

        lf_all = pl.scan_parquet(self.temporal_file_path)
        lf_base = lf_all.filter(~pl.col("Variable").is_in(["Flow", "Fuel_Remaining", "Stored_Energy"]))

        lf_lines = self._get_lines_flow_lf()
        lf_flow = lf_all.filter(pl.col("Variable") == "Flow").join(lf_lines, on=["Asset Name", "Unit Type"], how="left")
        lf_flow_split = self._split_directional_flows(lf_flow)

        if aggregation == "nodes":
            node_meta = [pl.col("Node").alias("Asset Name"), pl.lit("Node").alias("Asset Type"), pl.lit("Node").alias("Unit Type")]
            lf_base_n = lf_base.group_by(["Node", "Variable"]).agg(pl.col("Value").abs().sum()).with_columns(node_meta)
            lf_flow_n = lf_flow_split.group_by(["Node", "Variable"]).agg(pl.col("Value").abs().sum()).with_columns(node_meta)
            summary_lf = pl.concat([lf_base_n, lf_flow_n])
        else:
            lf_flow_assets = lf_flow_split.drop(["Node_A", "Node_B", "Eff"])
            summary_lf = pl.concat([lf_base, lf_flow_assets]).group_by(index_cols + ["Variable"]).agg(pl.col("Value").abs().sum())

        summary_lf = summary_lf.with_columns([
            pl.when(pl.col("Variable") == "Inflows")
              .then(pl.col("Value") / (year_count * 1000))
              .otherwise((pl.col("Value") * resolution) / (year_count * 1000)).alias("Total_TWh_yr")
        ])

        summary_df = summary_lf.collect().pivot(
            values="Total_TWh_yr",
            index=index_cols,
            on="Variable",
            aggregate_function="sum"
        ).fill_null(0.0)
        rename_mapping = {col: f"{col} (TWh/yr)" for col in summary_df.columns if col not in index_cols}

        return self._format_and_transpose_view(
            summary_df, aggregation, index_cols, header_name="Variable", file_name=f"summary_{aggregation.upper()}",
            sort_variable_columns=(aggregation == "assets"), rename_mapping=rename_mapping
        )

    def _view_levelised_cost_nodes(self):
        return self._view_levelised_cost(aggregation="nodes")

    def _view_levelised_cost_assets(self):
        return self._view_levelised_cost(aggregation="assets")

    def _view_levelised_cost(self, aggregation: str = "assets") -> ResultFile:
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

        df_lines = df_merged.filter(pl.col("Asset Class").str.to_lowercase() == "major_lines")
        df_base = df_merged.filter(pl.col("Asset Class").str.to_lowercase() != "major_lines")

        df_lines_A = df_lines.with_columns([
            pl.col("Asset Name").str.split("-").list.get(0).alias("Node"),
            *[(pl.col(c) / 2.0).alias(c) for c in cols_to_sum]
        ])

        df_lines_B = df_lines.with_columns([
            pl.col("Asset Name").str.split("-").list.get(1).alias("Node"),
            *[(pl.col(c) / 2.0).alias(c) for c in cols_to_sum]
        ])

        df_nodal_pool = pl.concat([df_base, df_lines_A, df_lines_B])

        df_nodes = (
            df_nodal_pool.filter(pl.col("Node").is_not_null()
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

        if aggregation == "nodes":
            df_final = pl.concat([df_system.select(keep_cols), df_nodes.select(keep_cols)], how="vertical")
        else:
            df_final = df_assets.select(keep_cols)

        index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
        df_final = self._apply_standard_sort(df_final, index_cols=index_cols)
        df_final = df_final.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols], separator="|").alias("_asset_string")
        ).drop(index_cols)

        df_final = df_final.transpose(include_header=True, header_name="Metric", column_names="_asset_string")

        return ResultFile(
            f"levelised_cost_{aggregation.upper()}",
            self.statistics_dir,
            df_final.lazy(),
            decimals=3,
            write_kwargs={"multiindex_delimiter": "|"}
        )

    def _get_base_capacity_df(self, index_cols: list[str], numeric_cols: list[str]) -> pl.DataFrame:
        """Extracts Generator and Storage assets for capacity aggregations."""
        df_assets = pl.from_pandas(self.df_static.reset_index())
        df_base = df_assets.filter(pl.col("Asset Type").is_in(["Generator", "Storage"]))

        if "Node" in index_cols:
            df_base = df_base.filter(pl.col("Node").is_not_null())

        df_base = df_base.select(index_cols + numeric_cols).fill_null(0.0)
        # Cast to Float64 for safe concatenation later
        return df_base.with_columns([pl.col(c).cast(pl.Float64) for c in numeric_cols])

    def _get_lines_capacity_df(
        self,
        mode: str,
        gw_cols: list[str],
        gwh_cols: list[str]
    ) -> pl.DataFrame:
        """Extracts transmission lines and formats them for single or double-counted capacity aggregation."""
        accessor = Accessor(self.solution, "GW")
        line_rows = []
        for line in accessor.get_assets("major_lines").values():
            base_p = accessor.get_power_capacity(line, errors="coerce")
            b_limits = accessor.get_build_power(line, errors="coerce")
            vals = [0.0 if pd.isna(x) else x for x in [base_p] + list(b_limits)]

            row = {"Asset Type": "Line", "Unit Type": getattr(line, "unit_type", "transmission")}
            for col, val in zip(gw_cols, vals):
                row[col] = val
            for col in gwh_cols:
                row[col] = 0.0

            if mode == "double_counted":
                line_rows.append({**row, "Node": line.node_start.name})
                line_rows.append({**row, "Node": line.node_end.name})
            elif mode == "single_counted":
                line_rows.append(row)

        schema = {}
        if mode == "double_counted":
            schema["Node"] = pl.String

        schema["Asset Type"] = pl.String
        schema["Unit Type"] = pl.String

        for c in gw_cols + gwh_cols:
            schema[c] = pl.Float64

        return pl.DataFrame(line_rows, schema=schema)

    def _get_lines_flow_lf(self) -> pl.LazyFrame:
        """Extracts metadata for transmission lines to join with temporal flow traces."""
        accessor = Accessor(self.solution, "GW")
        line_data = []
        for a in accessor.get_assets('major_lines').values():
            parts = str(a.name).split("-")
            line_data.append({
                "Asset Name": a.name,
                "Unit Type": getattr(a, 'unit_type', 'transmission'),
                "Node_A": parts[0] if len(parts) > 0 else "Unknown",
                "Node_B": parts[1] if len(parts) > 1 else "Unknown",
                "Eff": getattr(a, 'efficiency', 1.0)
            })
        return pl.LazyFrame(line_data, schema_overrides={"Eff": pl.Float32})

    def _split_directional_flows(self, lf_flow: pl.LazyFrame) -> pl.LazyFrame:
        """Splits bidirectional line flows into stacked Net_Imports and Net_Exports."""
        zero_f32 = pl.lit(0.0, dtype=pl.Float32)
        lf_exp_A = lf_flow.with_columns(
            pl.col("Node_A").alias("Node"), pl.lit("Net_Exports").alias("Variable"),
            pl.when(pl.col("Value") > 0).then(-pl.col("Value")).otherwise(zero_f32).alias("Value")
        )
        lf_imp_B = lf_flow.with_columns(
            pl.col("Node_B").alias("Node"), pl.lit("Net_Imports").alias("Variable"),
            pl.when(pl.col("Value") > 0).then(pl.col("Value") * pl.col("Eff")).otherwise(zero_f32).alias("Value")
        )
        lf_exp_B = lf_flow.with_columns(
            pl.col("Node_B").alias("Node"), pl.lit("Net_Exports").alias("Variable"),
            pl.when(pl.col("Value") < 0).then(pl.col("Value")).otherwise(zero_f32).alias("Value")
        )
        lf_imp_A = lf_flow.with_columns(
            pl.col("Node_A").alias("Node"), pl.lit("Net_Imports").alias("Variable"),
            pl.when(pl.col("Value") < 0).then(-pl.col("Value") * pl.col("Eff")).otherwise(zero_f32).alias("Value")
        )
        return pl.concat([lf_exp_A, lf_imp_B, lf_exp_B, lf_imp_A])

    def _format_and_transpose_view(
        self,
        df: pl.DataFrame,
        aggregation: str,
        index_cols: list[str],
        header_name: str,
        file_name: str,
        sort_variable_columns: bool = False,
        rename_mapping: dict = None,
    ) -> ResultFile:
        """Centralized formatter for sorting, unit renaming, string-concatenation, and transposing final views."""
        if aggregation == "assets":
            df = self._apply_standard_sort(df, index_cols=index_cols, sort_variable_columns=sort_variable_columns)
            df = df.with_columns(
                pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols], separator="|").alias("_col_string")
            ).drop(index_cols)
            col_names = "_col_string"
        elif aggregation == "nodes":
            df = df.with_columns(pl.col("Node").cast(pl.String).fill_null("Network")).sort("Node")
            col_names = "Node"
        elif aggregation == "unit_types":
            df = df.sort("Unit Type")
            col_names = "Unit Type"
        else:
            col_names = index_cols[0] if index_cols else None

        if rename_mapping:
            df = df.rename(rename_mapping)

        df = df.transpose(include_header=True, header_name=header_name, column_names=col_names)

        return ResultFile(file_name, self.statistics_dir, df.lazy(), decimals=3, write_kwargs={"multiindex_delimiter": "|"})

    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile(
            "x", self.statistics_dir, pd.DataFrame(self.solution.x).T, write_kwargs={"index": False}, decimals=3
        )
        return result_file
