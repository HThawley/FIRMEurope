import math
import os
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

from firm_ce.analysis.accessor import Accessor
from firm_ce.backend.scalar.solution import Solution, evaluate
from firm_ce.common.typing import npfloat
from firm_ce.system.scalar.parameters import ModelConfig
from firm_ce.system.scenario import Scenario

from firm_ce.analysis.display_presets import (
    MAP_SPECS,
    SUMMARY_PRESETS,
    MetricSpec,
    PlotStyle,
    is_generation_or_hydro,
    is_rechargeable_storage,
)


# ==============================================================================
# Display Engine
# ==============================================================================


class Display:
    """
    Visualizes and extracts optimization results for the European energy system
    across single solutions and MGA ensembles.
    """

    presets: Dict[str, List[MetricSpec]] = dict(SUMMARY_PRESETS)
    map_specs: Dict[str, MetricSpec] = dict(MAP_SPECS)

    def __init__(
        self,
        scenario: Scenario,
        config: ModelConfig,
        solution: Solution = None,
        noptima: List[Solution] = None,
        map_path: str = "./inputs/map/europe.geojson",
    ):
        self.scenario = scenario
        self.config = config
        self.mhmga = self.config.type == "mhmga"
        self.style = PlotStyle()

        self.noptima: List[Solution] = []
        self.solution: Solution = None

        if self.mhmga:
            if noptima:
                for sol in noptima:
                    self.noptima.append(sol)
                    if not self.noptima[-1].evaluated:
                        evaluate(self.noptima[-1])
            else:
                self._read_and_evaluate_noptima()
            self.solution = self.noptima[0]
        else:
            if solution is not None:
                self.solution = solution
            elif hasattr(scenario, "statistics"):
                self.solution = scenario.statistics.solution
            else:
                self._read_and_evaluate_optimum()
            if not self.solution.evaluated:
                evaluate(self.solution)

        self._accessor_cache: Dict[int, Accessor] = {}
        self._load_map_data(map_path)
        self._init_colors()

    # --------------------------------------------------------------------------
    # Registration & Configuration
    # --------------------------------------------------------------------------

    @classmethod
    def register_preset(cls, name: str, specs: Sequence[MetricSpec]) -> None:
        """Registers a custom named list of MetricSpecs for `plot_summary` and `extract`."""
        cls.presets[name.lower()] = list(specs)

    @classmethod
    def register_map_spec(cls, name: str, spec: MetricSpec) -> None:
        """Registers a custom named MetricSpec for `plot_map`."""
        cls.map_specs[name.lower()] = spec

    def set_dpi(self, dpi: int):
        self.style.dpi = dpi

    def set_base_fontsize(self, size: int):
        self.style.base_fontsize = size

    def set_large_fontsize(self, size: int):
        self.style.large_fontsize = size

    def set_small_fontsize(self, size: int):
        self.style.small_fontsize = size

    # --------------------------------------------------------------------------
    # Data Extraction & Aggregation Layer
    # --------------------------------------------------------------------------

    def _get_accessor(self, solution: Solution) -> Accessor:
        sol_id = id(solution)
        if sol_id not in self._accessor_cache:
            self._accessor_cache[sol_id] = Accessor(solution, "GW")
        return self._accessor_cache[sol_id]

    def _resolve_solution(self, idx: Union[int, Solution]) -> Solution:
        if isinstance(idx, Solution):
            if not idx.evaluated:
                evaluate(idx)
            return idx
        if self.mhmga and self.noptima:
            return self.noptima[idx]
        if idx != 0:
            raise ValueError(f"Solution index {idx} requested, but mhmga is False.")
        return self.solution

    def _resolve_map_spec(self, spec: Union[MetricSpec, str, None]) -> MetricSpec | None:
        if spec is None or isinstance(spec, MetricSpec):
            return spec
        key = spec.lower()
        if key not in self.map_specs:
            raise ValueError(f"Unknown map spec '{spec}'. Available: {list(self.map_specs.keys())}")
        return self.map_specs[key]

    def _resolve_summary_specs(self, specs: Union[str, MetricSpec, Sequence[MetricSpec]]) -> List[MetricSpec]:
        if isinstance(specs, MetricSpec):
            return [specs]
        if isinstance(specs, str):
            key = specs.lower()
            if key not in self.presets:
                raise ValueError(f"Unknown preset '{specs}'. Available: {list(self.presets.keys())}")
            return self.presets[key]
        return list(specs)

    @staticmethod
    def _get_build_power_capacity(accessor: Accessor, asset, build: str = "all") -> float:
        match str(build).lower():
            case "none" | "all":
                return accessor.get_power_capacity(asset)
            case "new_build":
                return accessor.get_new_build_capacity(asset, "power")
            case "existing" | "initial":
                return accessor.get_existing_capacity(asset, "power")
            case _:
                raise ValueError(f"Unknown build filter: '{build}'")

    @staticmethod
    def _get_build_energy_capacity(accessor: Accessor, asset, build: str = "all") -> float:
        match str(build).lower():
            case "none" | "all":
                return accessor.get_energy_capacity(asset)
            case "new_build":
                return accessor.get_new_build_capacity(asset, "energy")
            case "existing" | "initial":
                return accessor.get_existing_capacity(asset, "energy")
            case _:
                raise ValueError(f"Unknown build filter: '{build}'")

    def _resolve_label(self, asset, group_by: Union[str, Callable]) -> str:
        if callable(group_by):
            return group_by(asset)
        match group_by:
            case "tech":
                return self.scenario.identify_tech(asset.name)
            case "subtech":
                return self._get_display_label(asset)
            case "node":
                return asset.node.name if hasattr(asset, "node") else f"{asset.node_start.name}-{asset.node_end.name}"
            case _:
                raise ValueError(f"Unknown group_by mode: '{group_by}'")

    def _eval_asset_metric(self, accessor: Accessor, asset, spec: MetricSpec, scale: float) -> Dict[str, float]:
        """Evaluates a MetricSpec on a single asset, returning {label: value}."""
        if spec.metric == "none":
            return {}

        label = self._resolve_label(asset, spec.group_by)

        if spec.metric == "storage_sources":
            out = {}
            raw_type = str(getattr(asset, "unit_type", "")).lower()
            if raw_type in ("clphes", "olphes", "nphes", "bess2h", "bess4h") or "bess" in raw_type:
                out[f"{label} (Electrical)"] = abs(accessor.get_charge_gross(asset)) * scale
                if accessor.has_inflows(asset):
                    out[f"{label} (Inflows)"] = accessor.get_inflow_gross(asset) * scale
            return out

        match spec.metric:
            case "power_capacity":
                val = self._get_build_power_capacity(accessor, asset, spec.build)
            case "energy_capacity":
                val = self._get_build_energy_capacity(accessor, asset, spec.build)
            case "dispatch":
                val = accessor.get_dispatch_gross(asset)
            case "post_curtailment_power":
                val = accessor.get_gross("post_curtailment_power", asset)
            case "discharge":
                val = accessor.get_discharge_gross(asset)
            case "charge":
                val = abs(accessor.get_charge_gross(asset))
            case "inflows":
                val = accessor.get_inflow_gross(asset) if accessor.has_inflows(asset) else 0.0
            case "line_flow":
                val = accessor.get_line_use_gross(asset)
            case "line_flow_net":
                val = accessor.get_line_use_gross(asset) - accessor.get_line_loss_gross(asset)
            case _:
                raise ValueError(f"Unknown metric '{spec.metric}'")

        return {label: val * scale}

    def _aggregate_spec(self, solution: Solution, spec: MetricSpec, by_node: bool = False) -> dict:
        """
        Core query engine.
        Returns `{label: float}` if `by_node=False`, or `{node_name: {label: float}}` if `by_node=True`.
        """
        accessor = self._get_accessor(solution)
        scale = spec.scale_factor
        if spec.annualize:
            scale /= self.scenario.static.year_count * 1000.0

        data: dict = {}

        for asset_class in spec.assets:
            for asset in accessor.get_assets(asset_class).values():
                base_tech = self.scenario.identify_tech(asset.name) if asset_class != "major_lines" else ""
                if not spec.asset_filter(asset, base_tech):
                    continue

                contributions = self._eval_asset_metric(accessor, asset, spec, scale)

                if by_node:
                    node_name = asset.node.name
                    node_dict = data.setdefault(node_name, {})
                    for k, v in contributions.items():
                        node_dict[k] = node_dict.get(k, 0.0) + v
                else:
                    for k, v in contributions.items():
                        data[k] = data.get(k, 0.0) + v

                # Asset-level balances
                if not by_node and spec.include_balances:
                    if "storage_losses" in spec.include_balances and asset_class == "storages":
                        data["Storage Losses"] = (
                            data.get("Storage Losses", 0.0) + accessor.get_storage_loss_gross(asset) * scale
                        )
                    if "spillage" in spec.include_balances and asset_class == "storages":
                        data["Spillage"] = data.get("Spillage", 0.0) + accessor.get_spillage_gross(asset) * scale
                    if "line_losses" in spec.include_balances and asset_class == "major_lines":
                        data["Transmission Losses"] = (
                            data.get("Transmission Losses", 0.0) + accessor.get_line_loss_gross(asset) * scale
                        )

        # System-level balances
        if not by_node and "curtailment" in spec.include_balances:
            data["Curtailment"] = data.get("Curtailment", 0.0) + accessor.get_curtail_gross("system") * scale

        return data

    def extract(
        self,
        specs: Union[str, MetricSpec, Sequence[MetricSpec]] = "system_overview",
        solutions: Union[int, Sequence[int]] = 0,
        by_node: bool = False,
    ) -> pd.DataFrame:
        """
        Extracts aggregated metrics into a tidy pandas DataFrame for inspection or export.
        """
        resolved_specs = self._resolve_summary_specs(specs)
        sol_indices = [solutions] if isinstance(solutions, int) else list(solutions)

        records = []
        for sol_idx in sol_indices:
            sol = self._resolve_solution(sol_idx)
            for spec in resolved_specs:
                agg = self._aggregate_spec(sol, spec, by_node=by_node)
                if by_node:
                    for node, mix in agg.items():
                        for label, val in mix.items():
                            records.append(
                                {
                                    "solution": sol_idx,
                                    "metric": spec.title,
                                    "unit": spec.unit,
                                    "node": node,
                                    "label": label,
                                    "value": val,
                                }
                            )
                else:
                    for label, val in agg.items():
                        records.append(
                            {
                                "solution": sol_idx,
                                "metric": spec.title,
                                "unit": spec.unit,
                                "label": label,
                                "value": val,
                            }
                        )
        return pd.DataFrame.from_records(records)

    # --------------------------------------------------------------------------
    # Primary Plotting Entry Points
    # --------------------------------------------------------------------------

    def plot_map(
        self,
        node_spec: Union[MetricSpec, str] = "capacity",
        line_spec: Union[MetricSpec, str, None] = "auto",
        *,
        solutions: Union[int, Sequence[int]] = 0,
        mode: Literal["absolute", "delta", "atlas_delta"] = "absolute",
        ref_solution: int = 0,
        node_chart: Literal["pie", "bar"] = "auto",
        grid: Tuple[int, int] | None = None,
        ax: Union[plt.Axes, Sequence[plt.Axes], None] = None,
        max_scale: float | None = None,
        chart_scale: float = 1.0,
        threshold: float = 1e-5,
        legend: bool = True,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Unified spatial network plotter supporting custom MetricSpecs, single/atlas layouts,
        and delta comparisons.
        """
        resolved_node_spec = self._resolve_map_spec(node_spec)

        if line_spec == "auto":
            is_energy = resolved_node_spec.metric in ("dispatch", "post_curtailment_power", "discharge")
            resolved_line_spec = self.map_specs["line_energy" if is_energy else "line_capacity"].with_options(
                build=resolved_node_spec.build
            )
        else:
            resolved_line_spec = self._resolve_map_spec(line_spec)

        sol_list = [solutions] if isinstance(solutions, int) else list(solutions)
        if mode in ("delta", "atlas_delta") and not self.mhmga and len(sol_list) > 1:
            raise ValueError("Cannot plot multiple solutions when mhmga is False.")

        if node_chart == "auto":
            node_chart = "bar" if mode in ("delta", "atlas_delta") else "pie"
        if mode in ("delta", "atlas_delta") and node_chart == "pie":
            raise ValueError("Delta map plotting requires node_chart='bar'.")

        n_plots = len(sol_list)
        if grid is None:
            ncols = min(n_plots, 3) if n_plots > 1 else 1
            nrows = math.ceil(n_plots / ncols)
            grid = (nrows, ncols)

        with plt.rc_context(self.style.to_rc()):
            if ax is None:
                fig, axes_flat = self._setup_map_axis(nrows=grid[0], ncols=grid[1])
            else:
                axes_flat = np.atleast_1d(ax).flatten()
                fig = axes_flat[0].figure
                for a in axes_flat:
                    self._format_single_map_ax(a)

            ref_sol = self._resolve_solution(ref_solution)
            ref_node_data = (
                self._aggregate_spec(ref_sol, resolved_node_spec, by_node=True)
                if mode in ("delta", "atlas_delta")
                else None
            )

            # Pre-calculate global scaling across all panels for visual consistency
            global_max_abs = max_scale or 0.0
            global_max_delta = max_scale or 0.0

            panel_payloads = []
            for idx, sol_idx in enumerate(sol_list):
                target_sol = self._resolve_solution(sol_idx)
                target_node_data = self._aggregate_spec(target_sol, resolved_node_spec, by_node=True)
                is_delta_panel = (mode == "delta") or (mode == "atlas_delta" and idx > 0)

                if is_delta_panel:
                    delta_data = self._calculate_delta_dict(ref_node_data, target_node_data)
                    if max_scale is None:
                        panel_max = max(
                            (max((abs(v) for v in d.values()), default=0.0) for d in delta_data.values()),
                            default=1.0,
                        )
                        global_max_delta = max(global_max_delta, panel_max)
                    panel_payloads.append(("delta", sol_idx, target_sol, delta_data))
                else:
                    if max_scale is None:
                        panel_max = max((sum(d.values()) for d in target_node_data.values()), default=1.0)
                        global_max_abs = max(global_max_abs, panel_max)
                    panel_payloads.append(("abs", sol_idx, target_sol, target_node_data))

            global_max_abs = global_max_abs or 1.0
            global_max_delta = global_max_delta or 1.0

            for i, axis in enumerate(axes_flat):
                if i >= len(panel_payloads):
                    axis.set_visible(False)
                    continue

                p_type, sol_idx, target_sol, node_data = panel_payloads[i]
                if p_type == "delta":
                    self._draw_delta_on_axis(
                        axis,
                        node_data,
                        global_max_delta,
                        chart_scale=chart_scale,
                        threshold=max(threshold, 1e-3),
                        legend=legend,
                    )
                    axis.set_title(f"Delta: Alt {sol_idx} - Alt {ref_solution}")
                else:
                    self._draw_nodes_on_axis(
                        axis,
                        node_data,
                        global_max_abs,
                        node_chart=node_chart,
                        chart_scale=chart_scale,
                        threshold=threshold,
                        legend=legend,
                    )
                    if resolved_line_spec is not None:
                        self._draw_transmission_on_axis(axis, target_sol, resolved_line_spec)
                    if n_plots > 1 or mode != "absolute":
                        axis.set_title(f"Absolute: Alt {sol_idx}")

            build_suffix = (
                f" ({resolved_node_spec.build})"
                if resolved_node_spec.build != "all"
                else ""
            )
            fig.suptitle(
                f"{resolved_node_spec.title}{build_suffix} [{resolved_node_spec.unit}]",
                fontsize=self.style.large_fontsize,
            )

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, axes_flat

    def plot_summary(
        self,
        specs: Union[str, MetricSpec, Sequence[MetricSpec]] = "system_overview",
        *,
        solutions: Union[int, Sequence[int]] = 0,
        mode: Literal["absolute", "delta"] = "absolute",
        ref_solution: int = 0,
        chart_type: Literal["stacked_bar", "treemap", "pie"] = "stacked_bar",
        normalize: bool | Literal["auto"] = "auto",
        sort: Literal["name", "size"] = "name",
        grid: Tuple[int, int] | None = None,
        figsize: Tuple[float, float] | None = None,
        ax: Union[plt.Axes, Sequence[plt.Axes], None] = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Unified aggregate fleet chart plotter.
        Supports stacked bars (across metrics or across multiple MGA solutions),
        treemaps, and area-scaled pie charts.
        """
        resolved_specs = self._resolve_summary_specs(specs)
        sol_list = [solutions] if isinstance(solutions, int) else list(solutions)

        units_set = {s.unit for s in resolved_specs}
        if normalize == "auto":
            normalize = len(units_set) > 1 and chart_type == "stacked_bar" and mode == "absolute"

        if mode == "delta" and chart_type in ("pie", "treemap"):
            raise ValueError("Delta mode in plot_summary only supports chart_type='stacked_bar'.")

        ref_sol = self._resolve_solution(ref_solution) if mode == "delta" else None

        # Build bar/panel items: List of (label, mix_dict, unit)
        items: List[Tuple[str, Dict[str, float], str, int | None]] = []
        for sol_idx in sol_list:
            base_offset = len(items)
            sol = self._resolve_solution(sol_idx)
            for spec in resolved_specs:
                mix = self._aggregate_spec(sol, spec, by_node=False)
                if mode == "delta":
                    ref_mix = self._aggregate_spec(ref_sol, spec, by_node=False)
                    all_k = set(ref_mix.keys()) | set(mix.keys())
                    mix = {k: mix.get(k, 0.0) - ref_mix.get(k, 0.0) for k in all_k}

                if len(sol_list) > 1 and len(resolved_specs) == 1:
                    item_label = f"Alt {sol_idx}" if mode == "absolute" else f"Alt {sol_idx} - Alt {ref_solution}"
                elif len(sol_list) > 1:
                    item_label = f"{spec.title}\n(Alt {sol_idx})"
                else:
                    item_label = spec.title

                resolved_ref = (base_offset + spec.norm_ref) if spec.norm_ref is not None else None
                items.append((item_label, mix, spec.unit, resolved_ref))

        with plt.rc_context(self.style.to_rc()):
            if chart_type == "stacked_bar":
                fig, axes_flat = self._render_stacked_bars(
                    items,
                    normalize=normalize,
                    is_delta=(mode == "delta"),
                    sort=sort,
                    figsize=figsize,
                    ax=ax,
                )
            elif chart_type in ("treemap", "pie"):
                fig, axes_flat = self._render_part_to_whole_grid(
                    items,
                    chart_type=chart_type,
                    sort="size" if sort == "name" else sort,
                    grid=grid,
                    figsize=figsize,
                    ax=ax,
                )
            else:
                raise ValueError(f"Unsupported chart_type '{chart_type}'.")

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, axes_flat

    # --------------------------------------------------------------------------
    # Backwards-Compatible Wrappers
    # --------------------------------------------------------------------------

    def plot(self, data_type: str = "energy", *, atlas: bool = False, delta: bool = False, **kwargs):
        indices = kwargs.pop("indices", [0, 1] if delta else [0])
        curtailment = kwargs.pop("curtailment", False)
        build = kwargs.pop("build", "all")
        chart_type = kwargs.pop("chart_type", "auto")

        spec_key = "net_energy" if (data_type == "energy" and curtailment) else data_type
        spec = self._resolve_map_spec(spec_key).with_options(build=build)

        if atlas and delta:
            mode, sols, ref = "atlas_delta", indices, indices[0]
        elif delta:
            mode, sols, ref = "delta", indices[1], indices[0]
        elif atlas:
            mode, sols, ref = "absolute", indices, 0
        else:
            mode, sols, ref = "absolute", indices[0], 0

        _, axes = self.plot_map(
            node_spec=spec,
            solutions=sols,
            mode=mode,
            ref_solution=ref,
            node_chart=chart_type,
            **kwargs,
        )
        return axes if atlas else axes[0]

    def plot_energy_mix(self, *, atlas: bool = False, delta: bool = False, **kwargs):
        return self.plot(data_type="energy", atlas=atlas, delta=delta, **kwargs)

    def plot_power_capacity(self, *, atlas: bool = False, delta: bool = False, **kwargs):
        return self.plot(data_type="capacity", atlas=atlas, delta=delta, **kwargs)

    def plot_fleet_bars(
        self,
        view: str = "generation",
        *,
        include_storage: bool = True,
        energy_type: str = "both",
        normalize: bool = True,
        alternative: int = 0,
        figsize: Tuple[float, float] = None,
        save_path: str = None,
    ):
        view_map = {
            "generation": "system_overview",
            "storage": "storage_profile",
            "power_overview": "power_capacity",
            "energy_overview": "energy_balance",
        }
        preset_key = view_map.get(view.lower(), view.lower())
        specs = self._resolve_summary_specs(preset_key)

        if view.lower() == "generation" and not include_storage:
            specs = [s.with_options(asset_filter=is_generation_or_hydro) for s in specs]
        if energy_type == "generation":
            specs = [s.with_options(include_balances=()) for s in specs]

        fig, _ = self.plot_summary(
            specs=specs,
            solutions=alternative,
            chart_type="stacked_bar",
            normalize=normalize,
            figsize=figsize,
            save_path=save_path,
        )
        return fig

    # --------------------------------------------------------------------------
    # Rendering Primitives
    # --------------------------------------------------------------------------

    def _render_stacked_bars(
        self,
        items: List[Tuple[str, Dict[str, float], str, int | None]],
        normalize: bool,
        is_delta: bool,
        sort: str,
        figsize: Tuple[float, float] | None,
        ax: plt.Axes | None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        units_set = {u for _, _, u, _ in items}
        if not normalize and len(units_set) > 1:
            raise ValueError(
                f"Cannot plot unnormalized stacked bars across mixed units {units_set}. Set normalize=True."
            )

        if ax is None:
            if figsize is None:
                figsize = (max(6.0, 2.2 * len(items) + 2.0), 7.0)
            fig, axis = plt.subplots(figsize=figsize, dpi=self.style.dpi)
        else:
            axis = np.atleast_1d(ax).flatten()[0]
            fig = axis.figure

        x_positions = np.arange(len(items))
        bar_width = 0.55
        legend_handles = {}

        # Pre-compute positive/negative totals and global axis extrema up front
        bar_pos_totals = [sum(v for v in mix.values() if v > 1e-6) for _, mix, _, _ in items]
        bar_neg_totals = [sum(v for v in mix.values() if v < -1e-6) for _, mix, _, _ in items]

        if normalize and not is_delta:
            bar_denoms = [
                bar_pos_totals[norm_ref] if (norm_ref is not None and 0 <= norm_ref < len(items)) else pos_tot
                for pos_tot, (_, _, _, norm_ref) in zip(bar_pos_totals, items)
            ]
            bar_heights = [
                (pos_tot / denom * 100.0) if denom > 0 else 0.0
                for pos_tot, denom in zip(bar_pos_totals, bar_denoms)
            ]
            max_pos_height = max(bar_heights, default=100.0)
            min_neg_height = 0.0
        else:
            bar_denoms = bar_pos_totals
            max_pos_height = max(bar_pos_totals, default=1.0)
            min_neg_height = min(bar_neg_totals, default=0.0)

        annot_offset = 1.5 if (normalize and not is_delta) else max(max_pos_height * 0.02, 0.1)

        for idx, (_label, mix, unit, _norm_ref) in enumerate(items):
            clean_mix = {k: v for k, v in mix.items() if abs(v) > 1e-6}
            denom = bar_denoms[idx]
            net_total = sum(clean_mix.values())

            sorted_items = (
                sorted(clean_mix.items(), key=lambda x: abs(x[1]), reverse=True)
                if sort == "size"
                else self._sort_mix_items(clean_mix)
            )

            pos_bottom = 0.0
            neg_bottom = 0.0

            for tech, val in sorted_items:
                if normalize and not is_delta:
                    if denom <= 0:
                        continue
                    height = (val / denom) * 100.0
                    bottom = pos_bottom
                    pos_bottom += height
                else:
                    height = val
                    if val >= 0:
                        bottom = pos_bottom
                        pos_bottom += height
                    else:
                        bottom = neg_bottom
                        neg_bottom += height

                bar_container = axis.bar(
                    x_positions[idx],
                    height,
                    width=bar_width,
                    bottom=bottom,
                    color=self._get_color(tech),
                    hatch=self._get_hatch(tech),
                    edgecolor="black",
                    linewidth=0.5,
                )
                if tech not in legend_handles:
                    legend_handles[tech] = bar_container[0]

            prefix = "+" if (is_delta and net_total > 0) else ""
            axis.text(
                x_positions[idx],
                pos_bottom + annot_offset,
                f"{prefix}{net_total:,.1f} {unit}",
                ha="center",
                va="bottom",
                fontsize=self.style.small_fontsize,
                fontweight="bold",
            )

        if is_delta:
            axis.axhline(0, color="black", linewidth=0.8)

        axis.set_xticks(x_positions)
        axis.set_xticklabels([b[0] for b in items], fontsize=self.style.base_fontsize)
        unit_label = next(iter(units_set)) if len(units_set) == 1 else ""
        axis.set_ylabel(
            "Share (%)" if (normalize and not is_delta) else f"{'Delta' if is_delta else 'Total'} ({unit_label})",
            fontsize=self.style.base_fontsize,
        )

        y_top = max(110.0, max_pos_height * 1.12) if (normalize and not is_delta) else max(max_pos_height * 1.15, 1.0)
        y_bot = min_neg_height * 1.15 if is_delta else 0.0
        axis.set_ylim(y_bot, y_top)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if legend_handles:
            axis.legend(
                list(legend_handles.values()),
                list(legend_handles.keys()),
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                title="Component",
            )

        return fig, np.array([axis])

    def _render_part_to_whole_grid(
        self,
        items: List[Tuple[str, Dict[str, float], str, int | None]],
        chart_type: str,
        sort: str,
        grid: Tuple[int, int] | None,
        figsize: Tuple[float, float] | None,
        ax: Union[plt.Axes, Sequence[plt.Axes], None],
    ) -> Tuple[plt.Figure, np.ndarray]:
        n_items = len(items)
        if grid is None:
            ncols = min(n_items, 3)
            nrows = math.ceil(n_items / ncols)
            grid = (nrows, ncols)

        if ax is None:
            if figsize is None:
                figsize = (6.0 * grid[1], 5.0 * grid[0])
            fig, axes = plt.subplots(grid[0], grid[1], figsize=figsize, dpi=self.style.dpi, squeeze=False)
            axes_flat = axes.flatten()
            fig.subplots_adjust(hspace=0.35, wspace=0.5)
        else:
            axes_flat = np.atleast_1d(ax).flatten()
            fig = axes_flat[0].figure

        # Normalize area per shared physical unit
        unit_maxes: Dict[str, float] = {}
        for _, mix, unit, _ in items:
            total = sum(v for v in mix.values() if v > 0)
            unit_maxes[unit] = max(unit_maxes.get(unit, 1.0), total)

        for i, axis in enumerate(axes_flat):
            if i >= n_items:
                axis.axis("off")
                continue

            title, mix, unit, _ = items[i]
            clean_mix = {k: v for k, v in mix.items() if v > 1e-6}
            total = sum(clean_mix.values())

            if total <= 0:
                axis.axis("off")
                continue

            sorted_items = (
                sorted(clean_mix.items(), key=lambda x: x[1], reverse=True)
                if sort == "size"
                else self._sort_mix_items(clean_mix)
            )
            labels = [k for k, _ in sorted_items]
            values = [v for _, v in sorted_items]
            colors = [self._get_color(k) for k in labels]
            hatches = [self._get_hatch(k) for k in labels]
            scale = np.sqrt(total / unit_maxes[unit])

            if chart_type == "pie":
                handles, _ = axis.pie(
                    values,
                    radius=scale,
                    colors=colors,
                    hatches=hatches,
                    wedgeprops={"linewidth": 0.5, "edgecolor": "black"},
                )
            else:
                handles = self._draw_treemap(axis, values, colors, hatches, scale)

            axis.set_title(f"{title}\nTotal: {total:,.1f} {unit}", pad=10)
            axis.set_aspect("equal")
            axis.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        return fig, axes_flat

    def _draw_nodes_on_axis(
        self,
        ax: plt.Axes,
        node_data: Dict[str, Dict[str, float]],
        max_total: float,
        node_chart: str = "pie",
        chart_scale: float = 1.0,
        threshold: float = 1e-5,
        legend: bool = True,
    ):
        for node_name, mix in node_data.items():
            if node_name not in self.centroids:
                raise RuntimeError(f"Node {node_name} not found in map centroids.")

            filtered_mix = {k: v for k, v in mix.items() if abs(v) > threshold}
            if not filtered_mix:
                continue

            x, y = self.centroids[node_name]
            if node_chart == "bar":
                self._draw_bars(ax, filtered_mix, x, y, max_total, chart_scale=chart_scale)
            else:
                radius = 100_000 * chart_scale * np.sqrt(sum(filtered_mix.values()) / max_total)
                keys = sorted(filtered_mix.keys())
                values = [filtered_mix[k] for k in keys]
                colors = [self._get_color(k) for k in keys]
                hatches = [self._get_hatch(k) for k in keys]
                self._draw_pie(ax, values, x, y, radius, colors, hatches)

        if legend:
            self._add_legend(ax, node_data)

    def _draw_delta_on_axis(
        self,
        ax: plt.Axes,
        delta_dict: Dict[str, Dict[str, float]],
        global_max_delta: float,
        chart_scale: float = 1.0,
        threshold: float = 1e-3,
        legend: bool = True,
    ):
        for node_name, mix in delta_dict.items():
            if node_name not in self.centroids:
                continue

            filtered_mix = {k: v for k, v in mix.items() if abs(v) > threshold}
            if not filtered_mix:
                continue

            x_coord, y_coord = self.centroids[node_name]
            self._draw_bars(
                ax,
                filtered_mix,
                x_coord,
                y_coord,
                global_max_delta,
                is_delta=True,
                chart_scale=chart_scale,
            )

        if legend:
            self._add_legend(ax, delta_dict)

    def _draw_transmission_on_axis(self, ax: plt.Axes, solution: Solution, line_spec: MetricSpec):
        max_line_width = 5.0
        min_line_width = 0.3

        lines_data = []
        max_val = 0.0
        accessor = self._get_accessor(solution)

        for line in accessor.get_assets("major_lines").values():
            n_start, n_end = line.node_start.name, line.node_end.name
            if n_start not in self.centroids or n_end not in self.centroids:
                raise RuntimeError(f"Line endpoints ({n_start}, {n_end}) not found in map centroids.")

            val = sum(self._eval_asset_metric(accessor, line, line_spec, line_spec.scale_factor).values())
            max_val = max(max_val, val)
            lines_data.append((self.centroids[n_start], self.centroids[n_end], val))

        if max_val <= 0:
            return

        is_cap = line_spec.metric == "power_capacity"
        color = "red" if is_cap else "blue"
        alpha = 0.7 if is_cap else 0.5

        for p1, p2, val in lines_data:
            scaled_width = (val / max_val) * max_line_width
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=max(scaled_width, min_line_width),
                zorder=50,
                alpha=alpha,
            )

    def _draw_treemap(self, ax, values, colors, hatches, scale):
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")

        side = 2.0 * scale
        rects = self._compute_treemap_rects(values, -scale, -scale, side, side)
        patches = []
        for (rx, ry, rw, rh), color, hatch in zip(rects, colors, hatches):
            rect = plt.Rectangle(
                (rx, ry),
                rw,
                rh,
                facecolor=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            patches.append(rect)
        return patches

    @classmethod
    def _compute_treemap_rects(cls, values, x, y, dx, dy):
        if len(values) == 0:
            return []
        if len(values) == 1:
            return [(x, y, dx, dy)]

        total = sum(values)
        if total <= 0:
            return [(x, y, 0.0, 0.0) for _ in values]

        cumsum = np.cumsum(values)
        split_idx = int(np.argmin(np.abs(cumsum - total / 2.0))) + 1
        split_idx = min(max(split_idx, 1), len(values) - 1)

        left_vals = values[:split_idx]
        right_vals = values[split_idx:]
        frac = sum(left_vals) / total

        if dx >= dy:
            w1 = dx * frac
            return cls._compute_treemap_rects(left_vals, x, y, w1, dy) + cls._compute_treemap_rects(
                right_vals, x + w1, y, dx - w1, dy
            )
        else:
            h1 = dy * frac
            return cls._compute_treemap_rects(left_vals, x, y, dx, h1) + cls._compute_treemap_rects(
                right_vals, x, y + h1, dx, dy - h1
            )

    def _draw_pie(self, ax, dist, xpos, ypos, radius, colors, hatches):
        if sum(dist) == 0:
            return
        data = np.array(dist) / sum(dist)
        start_angle = 90

        for i, val in enumerate(data):
            if val == 0:
                continue
            end_angle = start_angle + val * 360
            w = Wedge(
                (xpos, ypos),
                radius,
                start_angle,
                end_angle,
                facecolor=colors[i],
                hatch=hatches[i],
                zorder=100,
                edgecolor="none",
            )
            ax.add_patch(w)
            start_angle = end_angle

        outline = Wedge((xpos, ypos), radius, 0, 360, facecolor="none", edgecolor="black", linewidth=0.5, zorder=101)
        ax.add_patch(outline)

    def _draw_bars(self, ax, mix, xpos, ypos, y_limit, is_delta=False, chart_scale=1.0):
        width_m = 250_000 * chart_scale
        height_m = 250_000 * chart_scale

        keys = sorted(mix.keys())
        values = [mix[k] for k in keys]
        colors = [self._get_color(k) for k in keys]
        hatches = [self._get_hatch(k) for k in keys]

        bar_width = width_m / (len(keys) + 1)
        x_start = xpos - (width_m / 2)
        baseline_y = ypos if is_delta else ypos - (height_m / 2)

        for i, val in enumerate(values):
            h = (val / y_limit) * (height_m / (2 if is_delta else 1))
            bar_x = x_start + (i + 0.5) * bar_width
            rect = plt.Rectangle(
                (bar_x - bar_width / 2, baseline_y),
                bar_width * 0.8,
                h,
                facecolor=colors[i],
                hatch=hatches[i],
                edgecolor="black",
                linewidth=0.5,
                zorder=110,
            )
            ax.add_patch(rect)

        ax.plot([x_start, x_start + width_m], [baseline_y, baseline_y], color="black", lw=0.8, zorder=111)

    # --------------------------------------------------------------------------
    # Map, Solution I/O & Styling Helpers
    # --------------------------------------------------------------------------

    def _format_single_map_ax(self, ax: plt.Axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.set_aspect("equal")
        self.map_data.plot(ax=ax, edgecolor="black", facecolor="#eeeeee", zorder=1)

        minx, miny, maxx, maxy = self.map_data.total_bounds
        margin = 200_000
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

    def _setup_map_axis(self, nrows=1, ncols=1, figsize=None):
        if figsize is None:
            figsize = (8 * ncols, 8 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.style.dpi, squeeze=False)
        axes_flat = axes.flatten()
        for ax in axes_flat:
            self._format_single_map_ax(ax)
        return fig, axes_flat

    @staticmethod
    def _calculate_delta_dict(dict_a: dict, dict_b: dict) -> dict:
        delta = {}
        for n in set(dict_a.keys()) | set(dict_b.keys()):
            node_a, node_b = dict_a.get(n, {}), dict_b.get(n, {})
            delta[n] = {t: node_b.get(t, 0.0) - node_a.get(t, 0.0) for t in set(node_a.keys()) | set(node_b.keys())}
        return delta

    def _construct_solution(self, x):
        return Solution(
            x,
            self.scenario.static,
            self.scenario.fleet,
            self.scenario.network,
            self.config.balancing_type,
            self.config.fixed_costs_threshold,
        )

    def _read_and_evaluate_optimum(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(self.scenario.solution_dir, "x.csv")
        x = pd.read_csv(filepath, header=None).to_numpy().flatten()
        self.solution = self._construct_solution(x.astype(npfloat))
        evaluate(self.solution)

    def _read_and_evaluate_noptima(self, filepath: str = None):
        if filepath is None:
            mhmga_dir = os.path.join(self.scenario.solution_dir, "mga_logs")
            filepath = os.path.join(mhmga_dir, "mga_alternatives.csv")
        noptima_df = pd.read_csv(filepath)
        noptima_x = [row.to_numpy() for _, row in noptima_df.iloc[:, 3:].iterrows()]
        self.noptima = [self._construct_solution(x.astype(npfloat)) for x in noptima_x]
        for sol in self.noptima:
            evaluate(sol)
        self.solution = self.noptima[0]

    def _load_map_data(self, filepath: str):
        self.map_data = gpd.read_file(filepath).to_crs(epsg=3035)
        self.centroids = self._calculate_centroids()

    def _calculate_centroids(self) -> Dict[str, Tuple[float, float]]:
        centroids = {}
        map_cents = self.map_data.geometry.centroid
        for node in self.solution.network.nodes.values():
            match_indices = self.map_data.index[self.map_data["ISO3"].str.lower() == node.name.lower()]
            if not match_indices.empty:
                pt = map_cents[match_indices[0]]
                centroids[node.name] = (pt.x, pt.y)
            else:
                warnings.warn(f"No map geometry found for node {node.name}.", UserWarning, 3)
                centroids[node.name] = (0.0, 0.0)
        return centroids

    def _get_display_label(self, asset) -> str:
        base = self.scenario.identify_tech(asset.name) if getattr(asset, "object_class", "") != "line" else ""
        raw_type = str(getattr(asset, "unit_type", "")).lower()

        if base in ("Biomass", "Biogas"):
            return "Bioenergy"

        if is_rechargeable_storage(asset, base):
            if raw_type == "nphes":
                return "New PHES"
            if raw_type == "clphes":
                return "Closed-loop PHES"
            if raw_type == "olphes":
                return "Open-loop PHES"
            if "bess" in raw_type or raw_type == "battery":
                return "Battery"

        if getattr(asset, "object_class", "") == "line":
            tx_map = {
                "ac_ohl_transmission": "AC OHL",
                "ac_ohl_mountain_transmission": "AC OHL (Mountain)",
                "dc_subsea_transmission": "DC Subsea",
                "dc_underground_transmission": "DC Underground",
            }
            return tx_map.get(raw_type, raw_type)

        return base

    def _init_colors(self):
        self.tech_colors = {
            "Utility Solar": "#F59E0B",
            "Rooftop Solar": "#FEF08A",
            "Onshore Wind": "#60A5FA",
            "Offshore Wind": "#1D4ED8",
            "Hydro": "#0D9488",
            "Pondage": "#2DD4BF",
            "Run of River": "#99F6E4",
            "Bioenergy": "#16A34A",
            "Biomass": "#15803D",
            "Biogas": "#86EFAC",
            "Geothermal": "#B91C1C",
            "Nuclear": "#E11D48",
            "Fossil Gas": "#78716C",
            "Coal": "#27272A",
            "Battery": "#A855F7",
            "Closed-loop PHES": "#EC4899",
            "Open-loop PHES": "#6366F1",
            "New PHES": "#3730A3",
            "Legacy PHES": "#818CF8",
            "PHES": "#6366F1",
            "AC OHL": "#9A3412",
            "AC OHL (Mountain)": "#EA580C",
            "DC Subsea": "#0E7490",
            "DC Underground": "#1E293B",
            "Curtailment": "#E5E7EB",
            "Storage Losses": "#9CA3AF",
            "Transmission Losses": "#6B7280",
            "Spillage": "#BAE6FD",
        }
        self.tech_hatches = {
            "Curtailment": "xx",
            "Storage Losses": "--",
            "Transmission Losses": "\\\\",
            "Spillage": "oo",
        }

    def _get_color(self, tech: str) -> str:
        base_tech = tech.removesuffix(" (Electrical)").removesuffix(" (Inflows)")
        return self.tech_colors.get(tech, self.tech_colors.get(base_tech, "#6B7280"))

    def _get_hatch(self, tech: str) -> str | None:
        if tech.endswith(" (Electrical)"):
            return "//"
        if tech.endswith(" (Inflows)"):
            return ".."
        return self.tech_hatches.get(tech, None)

    def _add_legend(self, ax: plt.Axes, data_dict: dict):
        present_techs = {tech for mix in data_dict.values() for tech in mix.keys()}
        handles, labels = [], []
        for tech in sorted(present_techs):
            handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=self._get_color(tech),
                    hatch=self._get_hatch(tech),
                    edgecolor="black",
                    linewidth=0.5,
                )
            )
            labels.append(tech)

        ax.legend(handles, labels, loc="upper right", title="Technology", frameon=False)

    @staticmethod
    def _sort_mix_items(mix: dict) -> list[tuple[str, float]]:
        secondary = {"Curtailment", "Storage Losses", "Transmission Losses", "Spillage"}
        return sorted(mix.items(), key=lambda item: (item[0] in secondary, item[0]))
