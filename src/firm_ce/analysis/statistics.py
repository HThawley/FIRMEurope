# type: ignore
import os
import time

from re import sub
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from firm_ce.common.typing import npfloat
from firm_ce.analysis.accessor import Accessor
from firm_ce.io.file_manager import ResultFile
from firm_ce.constructors.tensor_to_scalar import map_tensor_to_scalar
from firm_ce.backend.scalar.solution import Solution, evaluate
from firm_ce.backend.tensor.solution import SolutionTensor, EvaluateTensor


class Statistics:
    def __init__(
        self,
        scenario,
        *,
        # kwarg only arguments
        x: NDArray[npfloat] = None,
        solution: Solution = None,
        solutionTensor: SolutionTensor = None,
        solution_results_directory: str = None,
        destination_folder_name: str = None,
    ):
        self.scenario = scenario

        destination_folder_name = "statistics" if destination_folder_name is None else destination_folder_name

        x_is_None = x is None
        sol_is_None = solution is None
        solT_is_None = solutionTensor is None

        # check that inputs are consistent
        if (not x_is_None) and (not sol_is_None):
            if not np.isclose(x, solution.x).all():
                raise ValueError("'x' does not match 'solution.x'")
        if (not sol_is_None) and (not solT_is_None):
            if not np.isclose(solution.x, solutionTensor.x).all():
                raise ValueError("'solution.x' does not match 'solutionTensor.x'")
        if (not x_is_None) and (not solT_is_None):
            if not np.isclose(x, solutionTensor.x).all():
                raise ValueError("'x' does not match 'solutionTensor.x'")

        if all((x_is_None, sol_is_None, solT_is_None)):
            # If no x or solution provided, try grab from scenario
            if getattr(scenario, "solution", None) is None:
                raise ValueError("Initialise 'scenario.solution' or Provide 'x', 'solution', or 'solutionTensor'")
            self.solution = scenario.solution
            self.solutionTensor = scenario.solution
        elif (not sol_is_None):
            # first, if a 'solution' is provided, use it
            self.solution = solution
            self.solutionTensor = solutionTensor
            if not getattr(self.solution, "evaluated", False):
                self._evaluate_solution()
            # solutionTensor not directly used so does not need to be evaluated (indeed, may be None)
        elif (not solT_is_None):
            # next, if a 'solutionTensor' is provided, use it
            self.solutionTensor = solutionTensor
            if not getattr(self.solutionTensor, "evaluated", False):
                self._evaluate_solutionTensor()
            # solution will be "evaluated=True" because solutionTensor was evaluated
            self.solution = map_tensor_to_scalar(self.scenario, self.solutionTensor)
        else:
            # next, use x to construct a new solution/solutionTensor
            self.solution, self.solutionTensor = scenario.build_and_evaluate(x, False)

        self.accessor = Accessor(self.solution, "GW")
        if solution_results_directory is None:
            solution_results_directory = self.scenario.results_dir

        self.statistics_dir = self.create_solution_directory(
            solution_results_directory,
            f"{self.scenario.name}_{self.scenario.config.balancing_type}",
            destination_folder_name,
        )
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
        print(f"{self.scenario.name} LCOE: {self.solution.lcoe} [$/MWh], " f"Penalties: {self.solution.penalties}")

    def _evaluate_solutionTensor(self):
        start_time = time.time()
        EvaluateTensor(self.solutionTensor)
        end_time = time.time()
        print(f"Statistics solution tensor evaluation time: {end_time - start_time:.4f} seconds")
        print(f"{self.scenario.name} LCOE: {self.solutionTensor.lcoe} [$/MWh], " f"Penalties: {self.solutionTensor.penalties}")

    def _write_temporal_parquet(self) -> None:
        self.temporal_file_path = os.path.join(self.statistics_dir, "temporal_data.parquet")
        if os.path.exists(self.temporal_file_path):
            os.remove(self.temporal_file_path)

        schema = pa.schema([
            ("Time_Step", pa.int32()),
            ("Asset Name", pa.string()),
            ("Asset Type", pa.string()),
            ("Unit Type", pa.string()),
            ("Node", pa.string()),
            ("Variable", pa.string()),
            ("Value", pa.float32()),
        ])
        time_steps = pa.array(np.arange(self.intervals_count, dtype=np.int32))
        n_steps = self.intervals_count

        # Declarative mapping: asset_class -> [(Parquet Variable, Accessor Metric, Optional Predicate)]
        class_trace_specs = {
            "nodes": [
                ("Demand", "demand", None),
                ("Curtailment", "curtail", None),
                ("Deficit", "deficit", None),
            ],
            "generators": [
                ("Dispatch", "power", None),
            ],
            "storages": [
                ("Dispatch", "power", None),
                ("Stored_Energy", "storage_level", None),
                ("Charge", "charge", None),
                ("Discharge", "discharge", None),
                ("Inflows", "inflow", self.accessor.has_inflows),
                ("Spillage", "spillage", self.accessor.has_inflows),
            ],
            "major_lines": [
                ("Flow", "transmission", None),
            ],
            "fuels": [
                ("Fuel_Remaining", "remaining_energy", None),
            ],
        }

        with pq.ParquetWriter(self.temporal_file_path, schema) as writer:
            for asset_class, trace_specs in class_trace_specs.items():
                display_type = self.accessor.get_display_name(asset_class)

                for asset in self.accessor.get_assets(asset_class).values():
                    if asset_class == "nodes":
                        unit_type, node_name = "node", asset.name
                    elif asset_class == "fuels":
                        unit_type, node_name = "fuel", "network"
                    else:
                        unit_type = getattr(asset, "unit_type", None)
                        node_name = asset.node.name if hasattr(asset, "node") else None

                    base_cols = [
                        time_steps,
                        pa.repeat(asset.name, n_steps),
                        pa.repeat(display_type, n_steps),
                        pa.repeat(unit_type, n_steps),
                        pa.repeat(node_name, n_steps),
                    ]

                    for var_name, metric, condition in trace_specs:
                        if condition is None or condition(asset):
                            trace = self.accessor.get_trace(metric, asset)
                            table = pa.Table.from_arrays(
                                base_cols + [
                                    pa.repeat(var_name, n_steps),
                                    pa.array(trace, type=pa.float32()),
                                ],
                                schema=schema,
                            )
                            writer.write_table(table)

    def _build_static_df(self) -> pl.DataFrame:
        """Constructs static asset and nodal metadata directly as a Polars DataFrame."""
        static_data = []
        asset_classes = ["nodes", "generators", "storages", "major_lines"]
        meta_data_names = ("Asset ID", "Asset Name", "Asset Type", "Asset Class", "Unit Type", "Node", "Node_A", "Node_B")
        power_build_types = ("Existing Power", "New Build Power", "Min Build Power", "Max Build Power")
        energy_build_types = ("Existing Energy", "New Build Energy", "Min Build Energy", "Max Build Energy")

        for asset_class in asset_classes:
            is_node = asset_class == "nodes"
            assets = self.accessor.get_assets(asset_class)
            for asset in assets.values():
                meta_data = (
                    asset.id,
                    asset.name,
                    self.accessor.get_display_name(asset_class),
                    asset_class,
                    getattr(asset, "unit_type", "node" if is_node else None),
                    asset.node.name if hasattr(asset, "node") else (asset.name if is_node else None),
                    asset.node_start.name if hasattr(asset, "node_start") else None,
                    asset.node_end.name if hasattr(asset, "node_end") else None
                )
                row = dict(zip(meta_data_names, meta_data))
                row.update(self.accessor.get_all_costs(asset, errors="coerce"))
                row["Power Capacity"] = self.accessor.get_power_capacity(asset, errors="coerce")
                row["Energy Capacity"] = self.accessor.get_energy_capacity(asset, errors="coerce")
                row.update(dict(zip(power_build_types, self.accessor.get_build_power(asset, errors="coerce"))))
                row.update(dict(zip(energy_build_types, self.accessor.get_build_energy(asset, errors="coerce"))))
                static_data.append(row)

        df_static = pl.DataFrame(static_data, infer_schema_length=None)

        if not df_static.is_empty():
            sum_cols = [
                "Power Capacity", "Energy Capacity", "Existing Power", "Existing Energy",
                "New Build Power", "Min Build Power", "Max Build Power",
                "New Build Energy", "Min Build Energy", "Max Build Energy",
                "Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"
            ]

            # Compute nodal sums for generation and storage
            nodal_sums = (
                df_static.filter(pl.col("Asset Type").is_in(["Generator", "Storage"]))
                .group_by("Node")
                .agg([pl.col(c).fill_null(0.0).sum().alias(f"{c}_nodal") for c in sum_cols])
            )

            df_static = df_static.join(nodal_sums, left_on="Asset Name", right_on="Node", how="left")

            override_exprs = [
                pl.when(pl.col("Asset Class") == "nodes")
                  .then(pl.col(f"{c}_nodal").fill_null(0.0))
                  .otherwise(pl.col(c))
                  .alias(c)
                for c in sum_cols
            ]

            df_static = df_static.with_columns(override_exprs).drop([f"{c}_nodal" for c in sum_cols])

        return df_static

    def _split_lines_to_nodes(self, df_lines: pl.DataFrame, split_cols: list[str]) -> pl.DataFrame:
        """Splits line values 50/50 between start and end nodes based on static metadata."""
        df_A = df_lines.with_columns([
            pl.col("Node_A").alias("Node"),
            *[(pl.col(c) / 2.0).alias(c) for c in split_cols]
        ])
        df_B = df_lines.with_columns([
            pl.col("Node_B").alias("Node"),
            *[(pl.col(c) / 2.0).alias(c) for c in split_cols]
        ])
        return pl.concat([df_A, df_B]).drop(["Node_A", "Node_B"])

    def _ensure_master_tables(self) -> None:
        """Ensures temporal Parquet side effects are written and static data is built."""
        if not self.master_tables_built:
            self._write_temporal_parquet()
            self.df_static = self._build_static_df()
            self.master_tables_built = True

    def create_solution_directory(
        self,
        result_directory: str,
        solution_name: str,
        folder: str = "statistics"
    ) -> str:
        safe_name = sub(r"[^a-zA-Z0-9_\-]", "_", solution_name)
        solution_dir = os.path.join(result_directory, safe_name, folder)
        os.makedirs(solution_dir, exist_ok=True)
        return solution_dir

    def generate_result_files(self, file='all', write=True, delete=True) -> None:
        """
        Generates all result files using the high-level master DataFrames.
        """
        self._ensure_master_tables()

        file_functions = {
            "x_abs": self.generate_x_abs_file,
            "x_rel": self.generate_x_rel_file,
            "nodal_capacity_matrix": self._view_nodal_capacity_matrix,
            "summary_ASSETS": self._view_summary_assets,
            "summary_NODES": self._view_summary_nodes,
            "summary_UNIT_TYPES": self._view_summary_unit_types,
            "capacities_ASSETS": self._view_capacities_assets,
            "capacities_NODES": self._view_capacities_nodes,
            "capacities_UNIT_TYPES": self._view_capacities_unit_types,
            "components_ASSETS": self._view_component_costs_assets,
            "components_NODES": self._view_component_costs_nodes,
            "components_UNIT_TYPES": self._view_component_costs_unit_types,
            "levelised_cost_ASSETS": self._view_levelised_cost_assets,
            "levelised_cost_NODES": self._view_levelised_cost_nodes,
            "levelised_cost_UNIT_TYPES": self._view_levelised_cost_unit_types,
            "energy_balance_SYSTEM": self._view_energy_balance_system,
            "energy_balance_NODES": self._view_energy_balance_nodes,
            # "energy_balance_ASSETS": self._view_energy_balance_assets,
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
            'Demand', 'Deficit', 'Curtailment', 'Dispatch', 'Flow', 'Line_Input_Power',
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
        
        df_lines = self.df_static.filter(pl.col("Asset Type") == "Line").select(["Asset Type", "Unit Type", "Node_A", "Node_B"] + gw_cols + gwh_cols).fill_null(0.0)
        df_lines = self._split_lines_to_nodes(df_lines, gw_cols + gwh_cols)

        df_all = pl.concat([df_gen_stor, df_lines.select(df_gen_stor.columns)], how="vertical")

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

        gen_units = sorted([u for u in df_all.filter(pl.col("Asset Type") == "Generator").select("Unit Type").unique().to_series().to_list() if u])
        stor_units = sorted([u for u in df_all.filter(pl.col("Asset Type") == "Storage").select("Unit Type").unique().to_series().to_list() if u])
        line_units = sorted([u for u in df_all.filter(pl.col("Asset Type") == "Line").select("Unit Type").unique().to_series().to_list() if u])

        agg_cols = ["Generation (GW)", "Storage Power (GW)", "Storage Energy (GWh)", "Transmission (GW)"]
        ordered_cols = ["Node"] + agg_cols + gen_units + stor_units + line_units
        df_matrix = df_matrix.select(ordered_cols).sort("Node")

        # Because lines are apportioned 50/50, summing the nodes yields the exact system total
        system_row = df_matrix.select([
            pl.lit("System").alias("Node"),
            pl.col("Generation (GW)").sum(),
            pl.col("Storage Power (GW)").sum(),
            pl.col("Storage Energy (GWh)").sum(),
            pl.col("Transmission (GW)").sum()
        ] + [pl.col(u).sum() for u in gen_units + stor_units + line_units])

        df_matrix = pl.concat([system_row, df_matrix], how="vertical")

        return ResultFile("nodal_capacity_matrix", self.statistics_dir, df_matrix.lazy(), decimals=3)

    def _view_capacities_assets(self):
        return self._view_capacities(aggregation="assets")

    def _view_capacities_nodes(self):
        return self._view_capacities(aggregation="nodes")
        
    def _view_capacities_unit_types(self):
        return self._view_capacities(aggregation="unit_types")

    def _view_capacities(self, aggregation="assets") -> ResultFile:
        gw_cols = ["Power Capacity", "Existing Power", "New Build Power", "Min Build Power", "Max Build Power"]
        gwh_cols = ["Energy Capacity", "Existing Energy", "New Build Energy", "Min Build Energy", "Max Build Energy"]
        all_numeric = gw_cols + gwh_cols

        df_lines = self.df_static.filter(pl.col("Asset Type") == "Line").select(["Asset Name", "Asset Type", "Unit Type", "Node_A", "Node_B"] + all_numeric).fill_null(0.0)

        if aggregation == "nodes":
            index_cols = ["Node", "Asset Type", "Unit Type"]
            df_gen_stor = self._get_base_capacity_df(index_cols, all_numeric)
            df_lines = self._split_lines_to_nodes(df_lines, all_numeric)
            df_all = pl.concat([df_gen_stor, df_lines.select(df_gen_stor.columns)], how="vertical")

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

            # System Row using pure summation
            system_exprs = [pl.lit("System").alias("Node")]
            for col in df.columns:
                if col != "Node":
                    system_exprs.append(pl.col(col).sum().alias(col))

            system_row = df.select(system_exprs)
            df = pl.concat([system_row, df], how="vertical")
            
        elif aggregation == "unit_types":
            index_cols = ["Asset Type", "Unit Type"]
            df_gen_stor = self._get_base_capacity_df(index_cols, all_numeric)
            df_all = pl.concat([df_gen_stor, df_lines.select(index_cols + all_numeric)], how="vertical")
            df = df_all.group_by("Unit Type").agg([pl.col(c).sum() for c in all_numeric])
            
        else:
            index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
            df_gen_stor = self._get_base_capacity_df(index_cols, all_numeric)
            df_lines_clean = df_lines.drop(["Node_A", "Node_B"]).with_columns(pl.lit(None).alias("Node"))
            df = pl.concat([df_gen_stor, df_lines_clean.select(df_gen_stor.columns)], how="vertical")

        rename_map = None
        if aggregation in ("assets", "unit_types"):
            rename_map = {c: f"{c} (GW)" for c in gw_cols}
            rename_map.update({c: f"{c} (GWh)" for c in gwh_cols})

        return self._format_and_transpose_view(
            df,
            aggregation,
            index_cols=["Node"] if aggregation == "nodes" else (["Unit Type"] if aggregation == "unit_types" else index_cols),
            header_name="Metric",
            file_name=f"capacities_{aggregation.upper()}",
            rename_mapping=rename_map,
        )

    def _view_component_costs_assets(self):
        return self._view_component_costs(aggregation="assets")

    def _view_component_costs_nodes(self):
        return self._view_component_costs(aggregation="nodes")
        
    def _view_component_costs_unit_types(self):
        return self._view_component_costs(aggregation="unit_types")

    def _view_component_costs(self, aggregation) -> ResultFile:
        cost_cols = ["Annualised Build", "Fixed O&M", "Variable O&M", "Fuel Cost"]
        df_assets = self.df_static

        for c in cost_cols:
            if c not in df_assets.columns:
                df_assets = df_assets.with_columns(pl.lit(0.0).alias(c))

        if aggregation in ("assets", "unit_types"):
            intervals_count = self.intervals_count
            df_temporal = pl.scan_parquet(self.temporal_file_path).filter(
                pl.col("Variable").is_in(["Dispatch", "Flow"])
            ).group_by(["Asset Name", "Unit Type", "Asset Type"]).agg(
                (pl.col("Value").abs().sum() / intervals_count).alias("Mean_Power_GW")
            ).collect()
            
            df_assets = df_assets.join(df_temporal, on=["Asset Name", "Unit Type", "Asset Type"], how="left").fill_null(0.0)

        if aggregation == "nodes":
            df_gen_stor = df_assets.filter(
                pl.col("Asset Type").is_in(["Generator", "Storage"]) & pl.col("Node").is_not_null()
            ).select(["Node", "Asset Type", "Power Capacity"] + cost_cols).fill_null(0.0)

            df_lines = df_assets.filter(pl.col("Asset Type") == "Line").select(["Node_A", "Node_B", "Asset Type", "Power Capacity"] + cost_cols).fill_null(0.0)
            df_lines = self._split_lines_to_nodes(df_lines, cost_cols + ["Power Capacity"])

            df_all = pl.concat([df_gen_stor, df_lines.select(df_gen_stor.columns)], how="vertical")
            df_all = df_all.with_columns(pl.sum_horizontal(cost_cols).alias("Total Cost"))

            all_cost_cols = ["Total Cost"] + cost_cols

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

            network_exprs = [pl.lit("System").alias("Node")]
            for col in df.columns:
                if col != "Node":
                    network_exprs.append(pl.col(col).sum().alias(col))

            network_row = df.select(network_exprs)
            df = pl.concat([network_row, df], how="vertical")

            final_exprs = [pl.col("Node")]
            prefixes = ["Total", "Generation", "Storage Power", "Transmission"]

            for p in prefixes:
                for c in all_cost_cols:
                    base_col = f"{p} {c}"
                    cap_col = f"{p} Power Capacity"

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
            
        elif aggregation == "unit_types":
            index_cols = ["Unit Type"]
            df = df_assets.select(index_cols + cost_cols + ["Power Capacity", "Mean_Power_GW"]).fill_null(0.0)
            df = df.with_columns(pl.sum_horizontal(cost_cols).alias("Total Cost"))
            df = df.group_by(index_cols).sum()

            all_numeric = ["Total Cost"] + cost_cols
            df = df.with_columns([(pl.col(c) / 1e6).alias(f"{c} (M$/year)") for c in all_numeric])
            df = df.with_columns([
                pl.when(pl.col("Power Capacity") > 1e-6)
                  .then(pl.col(f"{c} (M$/year)") / pl.col("Power Capacity"))
                  .otherwise(0.0)
                  .alias(f"{c} ($/kW/year)") for c in all_numeric
            ])

            # Calculate CF
            df = df.with_columns(
                pl.when(pl.col("Power Capacity") > 1e-6)
                  .then((pl.col("Mean_Power_GW") / pl.col("Power Capacity")) * 100.0)
                  .otherwise(0.0).alias("Capacity Factor (%)")
            )

            # Filter out non-existent assets, but retain 0-cost assets if they generated power
            df = df.filter((pl.col("Total Cost (M$/year)") > 1e-6) | (pl.col("Capacity Factor (%)") > 1e-6)).drop(["Power Capacity", "Mean_Power_GW"])

            return self._format_and_transpose_view(
                df, aggregation, index_cols, header_name="Metric", file_name="components_UNIT_TYPES"
            )

        else:  # aggregation == "assets":
            index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]
            df = df_assets.select(index_cols + cost_cols + ["Power Capacity", "Mean_Power_GW"]).fill_null(0.0)
            df = df.with_columns(pl.sum_horizontal(cost_cols).alias("Total Cost"))
            all_numeric = ["Total Cost"] + cost_cols

            df = df.with_columns([(pl.col(c) / 1e6).alias(f"{c} (M$/year)") for c in all_numeric])
            df = df.with_columns([
                pl.when(pl.col("Power Capacity") > 1e-6)
                  .then(pl.col(f"{c} (M$/year)") / pl.col("Power Capacity"))
                  .otherwise(0.0)
                  .alias(f"{c} ($/kW/year)") for c in all_numeric
            ])

            # Calculate CF
            df = df.with_columns(
                pl.when(pl.col("Power Capacity") > 1e-6)
                  .then((pl.col("Mean_Power_GW") / pl.col("Power Capacity")) * 100.0)
                  .otherwise(0.0).alias("Capacity Factor (%)")
            )

            # Filter out non-existent assets, but retain 0-cost assets if they generated power
            df = df.filter((pl.col("Total Cost (M$/year)") > 1e-6) | (pl.col("Capacity Factor (%)") > 1e-6)).drop(["Power Capacity", "Mean_Power_GW"])

            return self._format_and_transpose_view(
                df, aggregation, index_cols, header_name="Metric", file_name="components_ASSETS"
            )

    def _view_energy_balance_assets(self) -> ResultFile:
        return self._view_energy_balance("assets")

    def _view_energy_balance_nodes(self) -> ResultFile:
        return self._view_energy_balance("nodes")

    def _view_energy_balance_system(self) -> ResultFile:
        return self._view_energy_balance("system")

    def _view_energy_balance(self, aggregation: str) -> ResultFile:
        aggregation = aggregation.lower()
        index_cols = ["Asset Name", "Asset Type", "Unit Type", "Node"]

        lf_all = pl.scan_parquet(self.temporal_file_path)
        lf_base = lf_all.filter(~pl.col("Variable").is_in(["Flow", "Fuel_Remaining"]))
        lf_fuel = lf_all.filter(pl.col("Variable") == "Fuel_Remaining")

        lines = self.accessor.get_assets('major_lines')
        line_data = []
        for a in lines.values():
            line_data.append({
                "Asset Name": a.name,
                "Unit Type": a.unit_type,
                "Node_A": a.node_start.name,
                "Node_B": a.node_end.name,
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

            lf_flow_n = (
                self._split_directional_flows(lf_flow)
                  .group_by(["Time_Step", "Node", "Variable"])
                  .agg(pl.col("Value").sum()).with_columns(node_meta)
            )

            lf_main = pl.concat([lf_base_n, lf_flow_n, lf_fuel_n])

        elif aggregation.lower() == "system":
            net_meta = [pl.lit("System").alias("Asset Name"),
                        pl.lit("System").alias("Asset Type"),
                        pl.lit("System").alias("Unit Type"),
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

        lf_main = lf_main.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols]
                          + [pl.col("Variable")], separator="|").alias("_col")
        ).select(["Time_Step", "_col", "Value"])

        df_main = lf_main.collect().pivot(
            values="Value",
            index="Time_Step",
            on="_col",
            aggregate_function="sum",
        ).fill_null(0.0)

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
        
    def _view_summary_unit_types(self):
        return self._view_summary(aggregation="unit_types")

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
            
            sys_meta = [
                pl.lit("System").alias("Node"), 
                pl.lit("System").alias("Asset Name"), 
                pl.lit("System").alias("Asset Type"), 
                pl.lit("System").alias("Unit Type")
            ]
            lf_sys = summary_lf.group_by(["Variable"]).agg(pl.col("Value").sum()).with_columns(sys_meta)
            summary_lf = pl.concat([summary_lf, lf_sys.select(summary_lf.collect_schema().names())])
            
        elif aggregation == "unit_types":
            index_cols = ["Unit Type"]
            lf_flow_assets = lf_flow_split.drop(["Node_A", "Node_B", "Eff"])
            summary_lf = pl.concat([lf_base, lf_flow_assets]).group_by(["Unit Type", "Variable"]).agg(pl.col("Value").abs().sum())
            
        else:
            lf_flow_assets = lf_flow_split.drop(["Node_A", "Node_B", "Eff"])
            summary_lf = pl.concat([lf_base, lf_flow_assets]).group_by(index_cols + ["Variable"]).agg(pl.col("Value").abs().sum())

        summary_lf = summary_lf.with_columns(
            ((pl.col("Value") * resolution) / (year_count * 1000)).alias("Total_TWh_yr")
        )

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
        
    def _view_levelised_cost_unit_types(self):
        return self._view_levelised_cost(aggregation="unit_types")

    def _view_levelised_cost(self, aggregation: str = "assets") -> ResultFile:
        resolution = self.solution.static.resolution
        year_count = self.solution.static.year_count

        string_cols = ["Asset ID", "Asset Name", "Asset Type", "Asset Class", "Unit Type", "Node", "Node_A", "Node_B"]
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

        df_costs = self.df_static
        for c in cost_cols:
            if c not in df_costs.columns:
                df_costs = df_costs.with_columns(pl.lit(0.0).alias(c))

        df_costs = df_costs.with_columns([
            (pl.col(c) / 1e6).alias(f"{c} [M$/yr]") for c in cost_cols
        ]).select(string_cols + [f"{c} [M$/yr]" for c in cost_cols])

        df_merged = df_costs.join(df_totals, on=["Asset Name", "Unit Type"], how="left").fill_null(0.0)

        for v in ["Dispatch", "Inflows", "Curtailment", "Flow"]:
            if v not in df_merged.columns:
                df_merged = df_merged.with_columns(pl.lit(0.0).alias(v))

        df_merged = df_merged.with_columns([
            pl.when(pl.col("Asset Type").str.to_lowercase() == "storage")
              .then(pl.col("Inflows")).otherwise(pl.col("Dispatch")).alias("Generation [GWh]"),
            pl.when(pl.col("Asset Type").str.to_lowercase() == "storage")
              .then(pl.col("Discharge")).otherwise(0.0).alias("Storage [GWh]"),
            pl.col("Flow").alias("Transmission [GWh]"),
            pl.col("Curtailment").alias("Curtailment [GWh]")
        ])

        mapped_costs = [f"{c} [M$/yr]" for c in cost_cols]

        df_merged = df_merged.with_columns([
            pl.when(pl.col("Asset Class").str.to_lowercase() == "nodes")
              .then(0.0).otherwise(pl.col(c)).alias(c) for c in mapped_costs
        ])

        df_merged = df_merged.with_columns(pl.sum_horizontal(mapped_costs).alias("Total Cost [M$/yr]"))

        def calc_lco(cost_col, energy_col):
            """Helper to calculate Levelised Cost ($/MWh = M$ * 1000 / GWh)"""
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

        def weighted_lco(lco_col, energy_col):
            """Base math logic for weighted averages"""
            total_weighted_cost = (pl.col(lco_col) * pl.col(energy_col)).sum()
            total_energy = pl.col(energy_col).sum()
            return (total_weighted_cost / total_energy).fill_nan(0.0)

        df_lines = df_merged.filter(pl.col("Asset Class").str.to_lowercase() == "major_lines")
        df_base = df_merged.filter(pl.col("Asset Class").str.to_lowercase() != "major_lines")

        keep_cols = ["Asset Name", "Asset Type", "Unit Type", "Node", "Total Cost [M$/yr]"] + mapped_costs + [
            "Generation [GWh]", "Storage [GWh]", "Transmission [GWh]", "Curtailment [GWh]",
            "LCOG [$/MWh]", "LCOS [$/MWh]", "LCOT [$/MWh]", "LCOE [$/MWh]"
        ]

        if aggregation == "nodes":
            df_lines_split = self._split_lines_to_nodes(df_lines, cols_to_sum)
            df_base_clean = df_base.drop(["Node_A", "Node_B"])
            df_nodal_pool = pl.concat([df_base_clean, df_lines_split.select(df_base_clean.columns)], how="vertical")

            df_nodes = (
                df_nodal_pool.filter(pl.col("Node").is_not_null()
                                     & (pl.col("Node").str.to_lowercase() != "system"))
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
            df_final = pl.concat([df_system.select(keep_cols), df_nodes.select(keep_cols)], how="vertical")
            
        elif aggregation == "unit_types":
            df_unit_types = (
                df_merged.filter(pl.col("Unit Type").is_not_null())
                .group_by("Unit Type").agg([
                    pl.sum(c) for c in cols_to_sum
                ] + [
                    weighted_lco("LCOG [$/MWh]", "Generation [GWh]").alias("LCOG [$/MWh]"),
                    weighted_lco("LCOS [$/MWh]", "Storage [GWh]").alias("LCOS [$/MWh]"),
                    weighted_lco("LCOT [$/MWh]", "Transmission [GWh]").alias("LCOT [$/MWh]"),
                ]).with_columns([
                    pl.col("Unit Type").alias("Asset Name"), pl.lit("Unit Type").alias("Asset Type"), pl.lit("System").alias("Node"), pl.lit(0.0).alias("LCOE [$/MWh]")
                ])
            )
            df_final = df_unit_types.select(keep_cols)
            
        else:
            df_assets = df_merged.filter(pl.col("Asset Class").str.to_lowercase() != "nodes")
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
        df_base = self.df_static.filter(pl.col("Asset Type").is_in(["Generator", "Storage"]))
        if "Node" in index_cols:
            df_base = df_base.filter(pl.col("Node").is_not_null())

        df_base = df_base.select(index_cols + numeric_cols).fill_null(0.0)
        return df_base

    def _get_lines_flow_lf(self) -> pl.LazyFrame:
        line_data = []
        for a in self.accessor.get_assets('major_lines').values():
            line_data.append({
                "Asset Name": a.name,
                "Unit Type": getattr(a, 'unit_type', 'transmission'),
                "Node_A": a.node_start.name,
                "Node_B": a.node_end.name,
                "Eff": getattr(a, 'efficiency', 1.0)
            })
        return pl.LazyFrame(line_data, schema_overrides={"Eff": pl.Float32})

    def _split_directional_flows(self, lf_flow: pl.LazyFrame) -> pl.LazyFrame:
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
        if aggregation == "assets":
            df = self._apply_standard_sort(df, index_cols=index_cols, sort_variable_columns=sort_variable_columns)
            df = df.with_columns(
                pl.concat_str([pl.col(c).cast(pl.String).fill_null("None") for c in index_cols], separator="|").alias("_col_string")
            ).drop(index_cols)
            col_names = "_col_string"
        elif aggregation == "nodes":
            df = df.with_columns(pl.col("Node").cast(pl.String).fill_null("System")).sort("Node")
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

    def generate_x_abs_file(self) -> ResultFile:
        if self.scenario.config.parameterisation == "relative":
            x_abs = self.solution.x * self.scenario.abs_rel_scaler
        else:
            x_abs = self.solution.x

        result_file = ResultFile(
            "x_abs", self.statistics_dir, pd.DataFrame(x_abs).T, write_kwargs={"index": False, "mode": "w"}, decimals=3
        )
        return result_file

    def generate_x_rel_file(self) -> ResultFile:
        if self.scenario.config.parameterisation == "relative":
            x_rel = self.solution.x
        else:
            x_rel = self.solution.x / self.scenario.abs_rel_scaler

        result_file = ResultFile(
            "x_rel", self.statistics_dir, pd.DataFrame(x_rel).T, write_kwargs={"index": False, "mode": "w"}, decimals=3
        )
        return result_file
