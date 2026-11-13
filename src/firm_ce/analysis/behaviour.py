# type: ignore
import calendar
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from firm_ce.analysis.accessor import Accessor
from firm_ce.analysis.display_presets import (
    is_generation_or_hydro,
    is_hydro_asset,
    is_rechargeable_storage,
)
from firm_ce.analysis.taxonomy import COMPOSITE_GROUPS, PlotStyle, get_color, get_hatch
from firm_ce.backend.scalar.solution import Solution, evaluate
from firm_ce.common.constants import LEAPDAYS
from firm_ce.common.typing import npfloat, npintp
from firm_ce.system.scenario import Scenario

MONTH_MAP: Dict[str, Tuple[int, ...]] = {
    "jan": (1,), "january": (1,), "feb": (2,), "february": (2,),
    "mar": (3,), "march": (3,), "apr": (4,), "april": (4,),
    "may": (5,), "jun": (6,), "june": (6,),
    "jul": (7,), "july": (7,), "aug": (8,), "august": (8,),
    "sep": (9,), "sept": (9,), "september": (9,), "oct": (10,), "october": (10,),
    "nov": (11,), "november": (11,), "dec": (12,), "december": (12,),
}

SEASON_MAP: Dict[str, Tuple[int, ...]] = {
    "winter": (12, 1, 2), "djf": (12, 1, 2),
    "spring": (3, 4, 5), "mam": (3, 4, 5),
    "summer": (6, 7, 8), "jja": (6, 7, 8),
    "autumn": (9, 10, 11), "fall": (9, 10, 11), "son": (9, 10, 11),
}

DAYS_NON_LEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
DAYS_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _validate_solution_object(sol: object) -> Solution:
    if sol is None:
        raise ValueError("Passed solution object is None.")
    cls_name = type(sol).__name__
    if cls_name == "SolutionTensor" or hasattr(sol, "ops") or hasattr(sol, "staticTensor"):
        raise NotImplementedError(
            f"Behaviour received '{cls_name}', which is not yet supported. Pass a scalar 'Solution'."
        )
    if not isinstance(sol, Solution):
        raise TypeError(f"Expected a scalar 'Solution' object, got '{cls_name}'.")
    if not sol.evaluated:
        evaluate(sol)
    return sol


class Behaviour:
    """
    Calculates and plots diurnal (time-of-day) average behaviour profiles,
    raw time-series traces, and co-located storage-charging-storage diagnostics.
    """

    def __init__(self, scenario: Scenario, solution: object, units: str = "GW"):
        self.scenario = scenario
        self.solution = _validate_solution_object(solution)
        self.static = self.solution.static
        self.accessor = Accessor(self.solution, units=units)
        self.units = units
        self.style = PlotStyle()

        self.resolution = float(self.static.resolution)
        self.intervals_count = int(self.static.intervals_count)
        self.first_year = int(self.static.first_year)
        self.year_count = int(self.static.year_count)

        self._month_of_interval, self._tod_bin, self.hours_of_day = self._build_calendar_arrays()

    # ==========================================================================
    # 1. Calendar & Nodal Filtering Helpers
    # ==========================================================================

    def _build_calendar_arrays(self) -> Tuple[NDArray[npintp], NDArray[npintp], NDArray[npfloat]]:
        month_of_interval = np.empty(self.intervals_count, dtype=npintp)
        hour_of_day_per_t = np.empty(self.intervals_count, dtype=npfloat)
        year_first_t = np.asarray(self.static.year_first_t, dtype=npintp)

        for y_idx, start_t in enumerate(year_first_t):
            end_t = int(year_first_t[y_idx + 1]) if (y_idx + 1 < len(year_first_t)) else self.intervals_count
            start_t, end_t = int(start_t), min(end_t, self.intervals_count)
            if start_t >= end_t:
                continue

            is_leap = bool(LEAPDAYS and calendar.isleap(self.first_year + y_idx))
            cum_hours = np.cumsum(DAYS_LEAP if is_leap else DAYS_NON_LEAP) * 24.0
            hour_in_year = np.arange(end_t - start_t, dtype=npfloat) * self.resolution

            month_of_interval[start_t:end_t] = np.clip(
                np.searchsorted(cum_hours, hour_in_year, side="right") + 1, 1, 12
            )
            hour_of_day_per_t[start_t:end_t] = np.round(np.mod(hour_in_year, 24.0), decimals=6)

        unique_hours, tod_bin = np.unique(hour_of_day_per_t, return_inverse=True)
        return month_of_interval, tod_bin.astype(npintp), unique_hours.astype(npfloat)

    def _resolve_nodes(self, nodes: Union[str, Sequence[str], None]) -> List[object]:
        """Returns the list of Node objects matching `nodes` (or all nodes if None/'all')."""
        all_nodes = list(self.accessor.get_assets("nodes").values())
        if nodes is None or (isinstance(nodes, str) and nodes.strip().lower() in ("all", "system")):
            return all_nodes

        raw = [nodes] if isinstance(nodes, str) else list(nodes)
        lookup = {n.name.lower(): n for n in all_nodes}
        resolved = []
        for item in raw:
            key = item.strip().lower()
            if key not in lookup:
                raise ValueError(f"Unknown node '{item}'. Available: {sorted(n.name for n in all_nodes)}")
            resolved.append(lookup[key])
        return resolved

    def _resolve_period_mask(self, period) -> NDArray[np.bool_]:
        if isinstance(period, tuple) and len(period) == 2:
            a, b = period
            if isinstance(a, int) and isinstance(b, str):
                return self._resolve_year_mask(a) & self._resolve_period_mask(b)
            if isinstance(a, int) and isinstance(b, int):
                mask = np.zeros(self.intervals_count, dtype=bool)
                mask[a:b] = True
                return mask
        if isinstance(period, int):
            return self._resolve_year_mask(period)
        if isinstance(period, str):
            key = period.strip().lower()
            if key in ("all", "annual", "full"):
                return np.ones(self.intervals_count, dtype=bool)
            if key.isdigit():
                return self._resolve_year_mask(int(key))
            target_months = MONTH_MAP.get(key) or SEASON_MAP.get(key)
            if target_months is None:
                raise ValueError(f"Unknown period keyword '{period}'.")
            return np.isin(self._month_of_interval, target_months)
        if isinstance(period, Sequence):
            mask = np.zeros(self.intervals_count, dtype=bool)
            for p in period:
                mask |= self._resolve_period_mask(p)
            return mask
        raise ValueError(f"Unsupported period specification: {period}")

    def _resolve_year_mask(self, year: int) -> NDArray[np.bool_]:
        year_idx = (year - self.first_year) if year >= self.first_year else year
        if not (0 <= year_idx < self.year_count):
            raise ValueError(f"Year '{year}' out of range.")
        return np.asarray(self.static.year_of_interval) == year_idx

    def _resolve_label(
        self,
        asset,
        group_by: Union[str, Callable],
        aggregate: Union[bool, str] = False,
        technologies: Union[str, Sequence[str], Callable] = "all",
    ) -> str:
        if isinstance(aggregate, str) and aggregate:
            return aggregate
        if aggregate is True:
            if isinstance(technologies, str) and technologies.lower() not in ("all", "generators", "storages"):
                return technologies.replace("_", " ").title()
            if isinstance(technologies, Sequence) and not isinstance(technologies, str) and len(technologies) == 1:
                return str(technologies[0]).replace("_", " ").title()
            return "Aggregated Selection"
        if callable(group_by):
            return group_by(asset)
        match group_by:
            case "tech":
                return self.scenario.identify_tech(asset.name)
            case "subtech":
                return self._get_subtech_label(asset)
            case "unit_type":
                return str(getattr(asset, "unit_type", asset.name))
            case "node":
                return asset.node.name
            case "asset":
                return asset.name
            case _:
                raise ValueError(f"Unknown group_by mode: '{group_by}'")

    def _get_subtech_label(self, asset) -> str:
        base = self.scenario.identify_tech(asset.name) if getattr(asset, "object_class", "") != "line" else ""
        raw_type = str(getattr(asset, "unit_type", "")).lower()
        if base in ("Biomass", "Biogas"):
            return "Bioenergy"
        if is_rechargeable_storage(asset, base):
            return {
                "aa_phes": "AA PHES",
                "e_phes": "E PHES",
                "clphes": "Closed-loop PHES",
                "olphes": "Open-loop PHES",
            }.get(raw_type, "Battery" if ("bess" in raw_type or raw_type == "battery") else base)
        return base

    def _matches_tech_filter(self, asset, technologies, group_by) -> bool:
        if technologies is None or technologies == "all":
            return True
        base_tech = self.scenario.identify_tech(asset.name) if getattr(asset, "object_class", "") != "line" else ""
        if callable(technologies):
            return technologies(asset, base_tech)
        if isinstance(technologies, str):
            key = technologies.strip().lower()
            if key == "all":
                return True
            if key in ("storages", "storage"):
                return getattr(asset, "object_class", "") == "storage"
            if key in ("rechargeable_storage", "rechargeable"):
                return is_rechargeable_storage(asset, base_tech)
            if key in ("generators", "generation"):
                return getattr(asset, "object_class", "") == "generator"
            if key == "generation_or_hydro":
                return is_generation_or_hydro(asset, base_tech)
            if key == "hydro":
                return is_hydro_asset(asset, base_tech)
            tech_list = [technologies]
        else:
            tech_list = list(technologies)

        allowed = set()
        for t in tech_list:
            t_low = t.strip().lower()
            allowed.add(t_low)
            if t_low in COMPOSITE_GROUPS:
                allowed.update(u.lower() for u in COMPOSITE_GROUPS[t_low][0])

        candidates = {
            asset.name.lower(),
            str(getattr(asset, "unit_type", "")).lower(),
            base_tech.lower(),
            self._get_subtech_label(asset).lower(),
            str(self._resolve_label(asset, group_by)).lower(),
        }
        return bool(allowed & candidates)

    # ==========================================================================
    # 2. Core Trace Extraction & Diurnal Aggregation
    # ==========================================================================

    def get_traces(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        nodes: Union[str, Sequence[str], None] = None,
        metric: str = "dispatch_and_charge",
        assets: Sequence[str] = ("generators", "storages"),
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        include_demand: bool = False,
        include_curtailment: bool = False,
        aggregate: Union[bool, str] = False,
    ) -> Dict[str, NDArray[npfloat]]:
        target_nodes = self._resolve_nodes(nodes)
        node_ids = {n.id for n in target_nodes}
        traces: Dict[str, NDArray[npfloat]] = {}

        def _add(label: str, arr: NDArray[npfloat]):
            if label in traces:
                traces[label] += arr
            else:
                traces[label] = arr.copy()

        for asset_class in assets:
            for asset in self.accessor.get_assets(asset_class).values():
                if hasattr(asset, "node") and asset.node.id not in node_ids:
                    continue
                if not self._matches_tech_filter(asset, technologies, group_by):
                    continue

                label = self._resolve_label(asset, group_by, aggregate=aggregate)
                if metric == "dispatch_and_charge":
                    if getattr(asset, "object_class", "") == "generator":
                        _add(label, self.accessor.get_generation_trace(asset))
                    elif getattr(asset, "object_class", "") == "storage":
                        _add(label, self.accessor.get_discharge_trace(asset))
                        _add(f"{label} (Electrical)", self.accessor.get_charge_trace(asset))
                else:
                    _add(label, self.accessor.get_trace(metric, asset))

        if include_curtailment:
            for n in target_nodes:
                _add("Curtailment", -self.accessor.get_curtail_trace(n))

        if include_demand:
            for n in target_nodes:
                _add("Demand", self.accessor.get_demand_trace(n))

        return {k: v for k, v in traces.items() if np.max(np.abs(v)) > self.accessor.tolerance}

    def compute_daily_average(
        self,
        traces: Dict[str, NDArray[npfloat]],
        period="all",
        agg: Literal["mean", "median"] = "mean",
    ) -> pd.DataFrame:
        mask = self._resolve_period_mask(period)
        if not np.any(mask):
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        bins = self._tod_bin[mask]
        n_bins = len(self.hours_of_day)
        counts = np.bincount(bins, minlength=n_bins).astype(npfloat)
        valid = counts > 0

        out: Dict[str, NDArray[npfloat]] = {"hour": self.hours_of_day[valid]}
        for label, arr in traces.items():
            vals = arr[mask]
            if agg == "mean":
                sums = np.bincount(bins, weights=vals, minlength=n_bins)
                out[label] = (sums[valid] / counts[valid]).astype(npfloat)
            else:
                out[label] = np.array([np.median(vals[bins == b]) for b in np.flatnonzero(valid)], dtype=npfloat)

        return pd.DataFrame(out).set_index("hour")

    # ==========================================================================
    # 3. Standard Behaviour Plotting (Diurnal & Raw)
    # ==========================================================================

    def plot_daily_average(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        periods="all",
        nodes: Union[str, Sequence[str], None] = None,
        metric: str = "dispatch_and_charge",
        assets: Sequence[str] = ("generators", "storages"),
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        chart_type: Literal["stacked", "line"] = "stacked",
        include_demand: bool = True,
        include_curtailment: bool = False,
        agg: Literal["mean", "median"] = "mean",
        aggregate: Union[bool, str] = False,
        figsize: Tuple[float, float] | None = None,
        ax: Union[plt.Axes, Sequence[plt.Axes], None] = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        period_list = (
            [periods]
            if isinstance(periods, (str, int)) or (isinstance(periods, tuple) and isinstance(periods[0], int))
            else list(periods)
        )
        traces = self.get_traces(
            technologies=technologies,
            nodes=nodes,
            metric=metric,
            assets=assets,
            group_by=group_by,
            include_demand=include_demand,
            include_curtailment=include_curtailment,
            aggregate=aggregate,
        )

        n_panels = len(period_list)
        node_tag = self._format_nodes_title(nodes)
        with plt.rc_context(self.style.to_rc()):
            if ax is None:
                fig, axes = plt.subplots(
                    1, n_panels, figsize=figsize or (6.5 * n_panels, 5.0), dpi=self.style.dpi, sharey=True, squeeze=False
                )
                axes_flat = axes.flatten()
            else:
                axes_flat = np.atleast_1d(ax).flatten()
                fig = axes_flat[0].figure

            for i, p in enumerate(period_list):
                df = self.compute_daily_average(traces, period=p, agg=agg)
                self._draw_trace_panel(
                    axes_flat[i],
                    df.index.to_numpy(),
                    df,
                    chart_type=chart_type,
                    x_label="Hour of Day (h)",
                    title=f"{self._format_period_title(p)}{node_tag}",
                    show_legend=(i == n_panels - 1),
                )
                axes_flat[i].set_xticks(np.arange(0, 25, 4))

            if axes_flat.size > 0:
                axes_flat[0].set_ylabel(f"{metric.replace('_', ' ').title()} ({self.units})")
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            return fig, axes_flat

    def plot_raw_trace(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        period=(0, 168),
        nodes: Union[str, Sequence[str], None] = None,
        metric: str = "dispatch_and_charge",
        assets: Sequence[str] = ("generators", "storages"),
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        chart_type: Literal["stacked", "line"] = "stacked",
        include_demand: bool = True,
        include_curtailment: bool = False,
        aggregate: Union[bool, str] = False,
        figsize: Tuple[float, float] = (12.0, 5.0),
        ax: plt.Axes | None = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        traces = self.get_traces(
            technologies=technologies,
            nodes=nodes,
            metric=metric,
            assets=assets,
            group_by=group_by,
            include_demand=include_demand,
            include_curtailment=include_curtailment,
            aggregate=aggregate,
        )
        mask = self._resolve_period_mask(period)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        hours = indices * self.resolution
        df = pd.DataFrame({k: v[mask] for k, v in traces.items()}, index=hours)
        node_tag = self._format_nodes_title(nodes)

        with plt.rc_context(self.style.to_rc()):
            fig, axis = plt.subplots(figsize=figsize, dpi=self.style.dpi) if ax is None else (ax.figure, ax)
            self._draw_trace_panel(
                axis,
                hours,
                df,
                chart_type=chart_type,
                x_label="Simulation Time (h)",
                title=f"Raw Trace: {self._format_period_title(period)}{node_tag}",
                show_legend=True,
            )
            axis.set_ylabel(f"{metric.replace('_', ' ').title()} ({self.units})")
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            return fig, axis

    # ==========================================================================
    # 4. Co-Located Storage-Charging-Storage
    # ==========================================================================

# ==========================================================================
    # 4. Nodal Storage Charging Source Attribution & Cross-Charging
    # ==========================================================================

    def compute_storage_charging_sources(
        self,
        storages: Union[str, Sequence[str], Callable] = "all",
        *,
        sources: Union[Literal["all", "storages", "generators"], Sequence[str], Callable] = "all",
        nodes: Union[str, Sequence[str], None] = None,
        period="all",
        source_group_by: Literal["subtech", "unit_type", "tech"] = "subtech",
        storage_group_by: Literal["subtech", "unit_type", "tech"] = "subtech",
        aggregate_storages: Union[bool, str] = False,
        rechargeable_only: bool = True,
        include_imports: bool = True,
    ) -> Tuple[Dict[str, NDArray[npfloat]], Dict[str, NDArray[npfloat]], pd.DataFrame]:
        """
        Node-by-node breakdown of the energy sources supplying storage electrical charging.

        At each node n and interval t:
          - Total nodal supply pool S_n(t) = sum_g Gen_{n,g}(t) + sum_s Disch_{n,s}(t) + Import_n(t)
          - Each charging storage group j receives power from source k != j proportional to
            S_{n,k}(t) / sum_{m != j} S_{n,m}(t).

        Returns:
          - source_traces: Dict[source_label, 1D array] (summed across selected charging storages & nodes)
          - pair_traces: Dict['Source -> Charger', 1D array]
          - matrix_df: pd.DataFrame (rows = Charging Sources, cols = Charging Storages) in annualised GWh/yr
        """
        mask = self._resolve_period_mask(period)
        n_selected = int(np.sum(mask))
        if n_selected == 0:
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        years_equiv = (n_selected * self.resolution) / 8760.0
        to_gwh_yr = (1.0 / self.accessor.factor) * self.resolution / max(years_equiv, 1e-9)

        target_nodes = self._resolve_nodes(nodes)
        target_node_ids = {n.id for n in target_nodes}

        # Bucket generators and storages by node.id
        gens_by_node: Dict[int, List[object]] = {}
        for gen in self.accessor.get_assets("generators").values():
            if gen.node.id in target_node_ids:
                gens_by_node.setdefault(gen.node.id, []).append(gen)

        stos_by_node: Dict[int, List[object]] = {}
        for sto in self.accessor.get_assets("storages").values():
            if sto.node.id in target_node_ids:
                stos_by_node.setdefault(sto.node.id, []).append(sto)

        source_traces: Dict[str, NDArray[npfloat]] = {}
        pair_traces: Dict[str, NDArray[npfloat]] = {}
        matrix_accum: Dict[Tuple[str, str], float] = {}
        all_source_labels = set()
        all_charger_labels = set()

        for node_obj in target_nodes:
            nid = node_obj.id
            node_stos = stos_by_node.get(nid, [])
            if not node_stos:
                continue

            # 1. Collect charging demand C_{n,j}(t) for target storages at node n
            c_group: Dict[str, NDArray[npfloat]] = {}
            # Also track raw discharge & charge by source_group_by to net out same-subtech D & C
            sto_d_src: Dict[str, NDArray[npfloat]] = {}
            sto_c_src: Dict[str, NDArray[npfloat]] = {}

            for sto in node_stos:
                base_tech = self.scenario.identify_tech(sto.name)
                s_lbl = self._resolve_label(sto, source_group_by)
                d_val = self.accessor.get_discharge_trace(sto)
                c_val = np.abs(self.accessor.get_charge_trace(sto))

                sto_d_src[s_lbl] = sto_d_src[s_lbl] + d_val if s_lbl in sto_d_src else d_val.copy()
                sto_c_src[s_lbl] = sto_c_src[s_lbl] + c_val if s_lbl in sto_c_src else c_val.copy()

                if rechargeable_only and not is_rechargeable_storage(sto, base_tech):
                    continue
                if not self._matches_tech_filter(sto, storages, storage_group_by):
                    continue

                c_lbl = self._resolve_label(
                    sto, storage_group_by, aggregate=aggregate_storages, technologies=storages
                )
                c_group[c_lbl] = c_group[c_lbl] + c_val if c_lbl in c_group else c_val.copy()

            if not c_group:
                continue

            # Net out internal simultaneous D & C within each storage group
            net_sto_disch: Dict[str, NDArray[npfloat]] = {}
            for s_lbl in sto_d_src:
                net_d = np.maximum(0.0, sto_d_src[s_lbl] - sto_c_src[s_lbl])
                if np.max(net_d[mask]) > self.accessor.tolerance:
                    net_sto_disch[s_lbl] = net_d

            if not aggregate_storages and source_group_by == storage_group_by:
                for c_lbl in list(c_group.keys()):
                    c_group[c_lbl] = np.maximum(0.0, sto_c_src[c_lbl] - sto_d_src[c_lbl])

            # 2. Build the nodal supply pool S_{n,k}(t) across all candidate sources
            supply_pool: Dict[str, NDArray[npfloat]] = {}

            # 2a. Generators (deducting nodal curtailment from VRE safely without raising RuntimeError)
            if sources != "storages":
                raw_gen_pool: Dict[str, NDArray[npfloat]] = {}
                vre_labels = set()
                for gen in gens_by_node.get(nid, []):
                    if sources not in ("all", "generators") and not self._matches_tech_filter(
                        gen, sources, source_group_by
                    ):
                        continue
                    g_lbl = self._resolve_label(gen, source_group_by)
                    g_val = self.accessor.get_generation_trace(gen)
                    if np.max(g_val[mask]) > self.accessor.tolerance:
                        raw_gen_pool[g_lbl] = raw_gen_pool[g_lbl] + g_val if g_lbl in raw_gen_pool else g_val.copy()
                        if not self.accessor.is_flexible(gen):
                            vre_labels.add(g_lbl)

                # Safely deduct nodal curtailment from non-flexible (VRE) generation first, then total generation
                if raw_gen_pool:
                    curt = self.accessor.get_curtail_trace(node_obj)
                    vre_tot = sum((raw_gen_pool[k] for k in vre_labels), start=np.zeros_like(curt))
                    curt_vre = np.minimum(curt, vre_tot)
                    vre_scale = np.where(vre_tot > 1e-12, (vre_tot - curt_vre) / np.maximum(vre_tot, 1e-12), 0.0)

                    for g_lbl, g_arr in raw_gen_pool.items():
                        net_g = (g_arr * vre_scale) if g_lbl in vre_labels else g_arr
                        if np.max(net_g[mask]) > self.accessor.tolerance:
                            supply_pool[g_lbl] = net_g

            # 2b. Discharging Storages
            if sources != "generators":
                for s_lbl, d_arr in net_sto_disch.items():
                    supply_pool[s_lbl] = supply_pool[s_lbl] + d_arr if s_lbl in supply_pool else d_arr.copy()

            # 2c. Net Nodal Imports
            if include_imports and sources == "all":
                imp_val = self.accessor.get_import_trace(node_obj)
                if np.max(imp_val[mask]) > self.accessor.tolerance:
                    supply_pool["Net Import"] = imp_val

            if not supply_pool:
                continue

            # 3. Allocate charging C_{n,j}(t) across sources k != j at node n
            src_labels = sorted(supply_pool.keys())
            S_mat = np.vstack([supply_pool[k] for k in src_labels])
            S_tot = S_mat.sum(axis=0)

            for c_lbl, c_arr in c_group.items():
                c_masked = np.where(mask, c_arr, 0.0)
                if np.max(c_masked) <= self.accessor.tolerance:
                    continue
                all_charger_labels.add(c_lbl)

                # Exclude self-discharging if the charger label matches a storage source label
                excl_idx = src_labels.index(c_lbl) if c_lbl in src_labels else -1
                denom = (S_tot - S_mat[excl_idx]) if excl_idx >= 0 else S_tot

                active = (c_masked > self.accessor.tolerance) & (denom > self.accessor.tolerance)
                if not np.any(active):
                    continue

                # If filtering to a subset of sources (e.g. sources="storages"), cap by available pool
                effective_c = np.minimum(c_masked, denom) if sources != "all" else c_masked
                inv_denom = np.where(active, 1.0 / np.maximum(denom, 1e-12), 0.0)

                for i, s_lbl in enumerate(src_labels):
                    if i == excl_idx:
                        continue
                    contrib = effective_c * (S_mat[i] * inv_denom)
                    if np.max(contrib) <= self.accessor.tolerance:
                        continue

                    all_source_labels.add(s_lbl)
                    source_traces[s_lbl] = source_traces[s_lbl] + contrib if s_lbl in source_traces else contrib.copy()

                    pair_key = f"{s_lbl} \u2192 {c_lbl}"
                    pair_traces[pair_key] = pair_traces[pair_key] + contrib if pair_key in pair_traces else contrib.copy()

                    matrix_accum[(s_lbl, c_lbl)] = matrix_accum.get((s_lbl, c_lbl), 0.0) + float(
                        np.sum(contrib) * to_gwh_yr
                    )

        matrix_df = pd.DataFrame(0.0, index=sorted(all_source_labels), columns=sorted(all_charger_labels))
        for (s_lbl, c_lbl), val in matrix_accum.items():
            matrix_df.loc[s_lbl, c_lbl] = val

        return source_traces, pair_traces, matrix_df

    def plot_storage_charging_sources(
        self,
        storages: Union[str, Sequence[str], Callable] = "all",
        *,
        sources: Union[Literal["all", "storages", "generators"], Sequence[str], Callable] = "all",
        nodes: Union[str, Sequence[str], None] = None,
        period="all",
        view: Literal["daily_average", "raw_trace", "matrix", "stacked_bar"] = "daily_average",
        source_group_by: Literal["subtech", "unit_type", "tech"] = "subtech",
        storage_group_by: Literal["subtech", "unit_type", "tech"] = "subtech",
        aggregate_storages: Union[bool, str] = True,
        rechargeable_only: bool = True,
        include_imports: bool = True,
        normalize: bool = False,
        chart_type: Literal["stacked", "line"] = "stacked",
        agg: Literal["mean", "median"] = "mean",
        figsize: Tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the energy sources supplying storage charging across the specified `nodes` and `period`.
          - view='daily_average': Time-of-day profile of charging power stacked by supplying technology
          - view='raw_trace': Raw time-series of charging power stacked by supplying technology
          - view='matrix': 2D heatmap (Rows = Supplying Technology, Cols = Charging Storage) in GWh/yr
          - view='stacked_bar': Stacked bar chart of annualised charging sources (GWh/yr or %) per storage type
        """
        # For matrix and stacked_bar views, default to showing individual storage columns unless user forced a string
        agg_sto = False if (view in ("matrix", "stacked_bar") and aggregate_storages is True) else aggregate_storages

        source_traces, pair_traces, matrix_df = self.compute_storage_charging_sources(
            storages=storages,
            sources=sources,
            nodes=nodes,
            period=period,
            source_group_by=source_group_by,
            storage_group_by=storage_group_by,
            aggregate_storages=agg_sto,
            rechargeable_only=rechargeable_only,
            include_imports=include_imports,
        )
        traces_to_plot = source_traces if agg_sto else pair_traces
        title_suffix = f" ({self._format_period_title(period)}{self._format_nodes_title(nodes)})"

        with plt.rc_context(self.style.to_rc()):
            if not traces_to_plot:
                fig, axis = plt.subplots(figsize=figsize or (6.0, 4.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                axis.text(0.5, 0.5, f"No storage charging detected{title_suffix}", ha="center", va="center")
                axis.axis("off")
                return fig, axis

            if view == "daily_average":
                fig, axis = plt.subplots(figsize=figsize or (7.5, 5.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                df = self.compute_daily_average(traces_to_plot, period=period, agg=agg)
                self._draw_trace_panel(
                    axis, df.index.to_numpy(), df,
                    chart_type=chart_type, x_label="Hour of Day (h)",
                    title=f"Storage Charging by Source{title_suffix}", show_legend=True,
                )
                axis.set_xticks(np.arange(0, 25, 4))
                axis.set_ylabel(f"Charging Power ({self.units})")

            elif view == "raw_trace":
                fig, axis = plt.subplots(figsize=figsize or (12.0, 5.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                mask = self._resolve_period_mask(period)
                hours = np.flatnonzero(mask) * self.resolution
                df = pd.DataFrame({k: v[mask] for k, v in traces_to_plot.items()}, index=hours)
                self._draw_trace_panel(
                    axis, hours, df,
                    chart_type=chart_type, x_label="Simulation Time (h)",
                    title=f"Raw Storage Charging by Source{title_suffix}", show_legend=True,
                )
                axis.set_ylabel(f"Charging Power ({self.units})")

            elif view == "matrix":
                fig, axis = plt.subplots(figsize=figsize or (7.0, 6.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                self._draw_cross_charge_matrix(axis, matrix_df)
                axis.set_title(f"Storage Charging Sources [GWh/yr]{title_suffix}", pad=20)

            elif view == "stacked_bar":
                fig, axis = plt.subplots(figsize=figsize or (7.5, 5.5), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                self._draw_charging_sources_bar(axis, matrix_df, normalize=normalize)
                axis.set_title(f"Storage Charging Mix{title_suffix}")

            else:
                raise ValueError(f"Unknown view '{view}'.")

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            return fig, axis

    def compute_storage_cross_charging(
        self,
        *,
        nodes: Union[str, Sequence[str], None] = None,
        period="all",
        group_by: Literal["subtech", "unit_type", "tech", "asset"] = "subtech",
        rechargeable_only: bool = False,
    ) -> Tuple[Dict[str, NDArray[npfloat]], pd.DataFrame]:
        """
        Evaluates co-located simultaneous discharging and charging node-by-node:
            O_n(t) = min(sum_i D_{n,i}(t), sum_j C_{n,j}(t))
            O_{n, i->j}(t) = O_n(t) * (D_{n,i}(t) / D_n(t)) * (C_{n,j}(t) / C_n(t))  for i != j
        and sums both the pairwise time-series traces and the annualised GWh/yr
        (Discharger x Charger) matrix across the requested `nodes`.
        """
        mask = self._resolve_period_mask(period)
        n_selected = int(np.sum(mask))
        if n_selected == 0:
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        years_equiv = (n_selected * self.resolution) / 8760.0
        to_gwh_yr = (1.0 / self.accessor.factor) * self.resolution / max(years_equiv, 1e-9)

        target_node_ids = {n.id for n in self._resolve_nodes(nodes)}
        storages_by_node: Dict[str, List[object]] = {}
        for sto in self.accessor.get_assets("storages").values():
            if sto.node.id not in target_node_ids:
                continue
            if rechargeable_only and not is_rechargeable_storage(sto, self.scenario.identify_tech(sto.name)):
                continue
            storages_by_node.setdefault(sto.node.name, []).append(sto)

        pair_traces: Dict[str, NDArray[npfloat]] = {}
        matrix_accum: Dict[Tuple[str, str], float] = {}
        all_labels = set()

        for sto_list in storages_by_node.values():
            if len(sto_list) < 2:
                continue

            d_group: Dict[str, NDArray[npfloat]] = {}
            c_group: Dict[str, NDArray[npfloat]] = {}
            for sto in sto_list:
                lbl = self._resolve_label(sto, group_by)
                all_labels.add(lbl)
                d_val = self.accessor.get_discharge_trace(sto)
                c_val = np.abs(self.accessor.get_charge_trace(sto))
                d_group[lbl] = d_group[lbl] + d_val if lbl in d_group else d_val.copy()
                c_group[lbl] = c_group[lbl] + c_val if lbl in c_group else c_val.copy()

            labels = sorted(d_group.keys())
            if len(labels) < 2:
                continue

            # Net out any internal simultaneous D & C within the same group_by label at this node
            net_dc = np.vstack([d_group[k] - c_group[k] for k in labels])
            D_mat = np.maximum(0.0, net_dc)
            C_mat = np.maximum(0.0, -net_dc)

            D_tot = D_mat.sum(axis=0)
            C_tot = C_mat.sum(axis=0)
            overlap_n = np.minimum(D_tot, C_tot)

            active = (overlap_n > self.accessor.tolerance) & mask
            if not np.any(active):
                continue

            inv_D = np.where(active, 1.0 / np.maximum(D_tot, 1e-12), 0.0)
            inv_C = np.where(active, 1.0 / np.maximum(C_tot, 1e-12), 0.0)

            for i, lbl_d in enumerate(labels):
                d_share = D_mat[i] * inv_D
                if not np.any(d_share > 0):
                    continue
                for j, lbl_c in enumerate(labels):
                    if i == j:
                        continue
                    pair_ov = overlap_n * d_share * (C_mat[j] * inv_C)
                    if np.max(pair_ov) <= self.accessor.tolerance:
                        continue

                    pair_key = f"{lbl_d} \u2192 {lbl_c}"
                    if pair_key in pair_traces:
                        pair_traces[pair_key] += pair_ov
                    else:
                        pair_traces[pair_key] = pair_ov.copy()

                    matrix_accum[(lbl_d, lbl_c)] = matrix_accum.get((lbl_d, lbl_c), 0.0) + float(
                        np.sum(pair_ov) * to_gwh_yr
                    )

        ordered = sorted(all_labels)
        matrix_df = pd.DataFrame(0.0, index=ordered, columns=ordered)
        for (lbl_d, lbl_c), val in matrix_accum.items():
            matrix_df.loc[lbl_d, lbl_c] = val

        return pair_traces, matrix_df

    def plot_storage_cross_charging(
        self,
        *,
        nodes: Union[str, Sequence[str], None] = None,
        period="all",
        view: Literal["daily_average", "raw_trace", "pair_matrix"] = "daily_average",
        group_by: Literal["subtech", "unit_type", "tech"] = "subtech",
        rechargeable_only: bool = False,
        chart_type: Literal["stacked", "line"] = "stacked",
        agg: Literal["mean", "median"] = "mean",
        figsize: Tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Wrapper around `plot_storage_charging_sources` restricted to storage-to-storage transfers."""
        mapped_view = "matrix" if view == "pair_matrix" else view
        return self.plot_storage_charging_sources(
            storages="all",
            sources="storages",
            nodes=nodes,
            period=period,
            view=mapped_view,
            source_group_by=group_by,
            storage_group_by=group_by,
            aggregate_storages=False,
            rechargeable_only=rechargeable_only,
            include_imports=False,
            chart_type=chart_type,
            agg=agg,
            figsize=figsize,
            ax=ax,
            save_path=save_path,
        )

    # ==========================================================================
    # 5. Shared Rendering Primitives
    # ==========================================================================

    def _draw_trace_panel(
        self,
        ax: plt.Axes,
        x: NDArray[npfloat],
        df: pd.DataFrame,
        *,
        chart_type: str,
        x_label: str,
        title: str,
        show_legend: bool,
    ) -> None:
        demand_series = df["Demand"].to_numpy() if "Demand" in df.columns else None
        tech_cols = [c for c in df.columns if c != "Demand"]
        pair_hatches = [None, "//", "\\\\", "xx", "..", "++", "oo"]

        if chart_type == "line":
            for col in tech_cols:
                base_col = col.split(" \u2192 ")[0]
                ls = "--" if col.endswith(" (Electrical)") else "-"
                ax.plot(
                    x, df[col].to_numpy(),
                    label=col.replace(" (Electrical)", " (Charge)"),
                    color=get_color(base_col), linestyle=ls, lw=1.8,
                )
        elif chart_type == "stacked":
            pos_bottom = np.zeros_like(x, dtype=npfloat)
            neg_bottom = np.zeros_like(x, dtype=npfloat)

            for idx, col in enumerate(sorted(tech_cols)):
                y = df[col].to_numpy()
                y_pos, y_neg = np.maximum(0.0, y), np.minimum(0.0, y)
                is_pair = " \u2192 " in col
                base_col = col.split(" \u2192 ")[0]
                color = get_color(base_col)
                hatch = pair_hatches[idx % len(pair_hatches)] if is_pair else get_hatch(col)
                clean_label = col.replace(" (Electrical)", " (Charge)")

                if np.any(y_pos > 1e-6):
                    ax.fill_between(
                        x, pos_bottom, pos_bottom + y_pos,
                        facecolor=color, hatch=hatch,
                        edgecolor="black" if hatch else "none",
                        linewidth=0.3 if hatch else 0.0, alpha=0.9, label=clean_label,
                    )
                    pos_bottom += y_pos

                if np.any(y_neg < -1e-6):
                    ax.fill_between(
                        x, neg_bottom + y_neg, neg_bottom,
                        facecolor=color, hatch=hatch or "//",
                        edgecolor="black", linewidth=0.3, alpha=0.85,
                        label=clean_label if not np.any(y_pos > 1e-6) else f"{clean_label} (Neg)",
                    )
                    neg_bottom += y_neg
        else:
            raise ValueError(f"Unknown chart_type '{chart_type}'.")

        if demand_series is not None:
            ax.plot(x, demand_series, color="black", lw=2.0, label="Demand", zorder=10)

        ax.axhline(0.0, color="black", lw=0.8, zorder=5)
        if x.size > 1:
            ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    def _draw_cross_charge_matrix(self, ax: plt.Axes, matrix_df: pd.DataFrame) -> None:
        active_rows = [r for r in matrix_df.index if matrix_df.loc[r].sum() > 1e-3]
        active_cols = [c for c in matrix_df.columns if matrix_df[c].sum() > 1e-3]
        sub_df = matrix_df.loc[active_rows, active_cols] if (active_rows and active_cols) else matrix_df

        M = sub_df.to_numpy(dtype=float)
        nrows, ncols = M.shape
        vmax = max(float(np.nanmax(M)) if M.size else 1.0, 1e-3)

        ax.imshow(M, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        for i in range(nrows):
            for j in range(ncols):
                val = M[i, j]
                if sub_df.index[i] == sub_df.columns[j] and val == 0:
                    ax.text(j, i, "\u2013", ha="center", va="center", color="gray")
                elif val >= 0.1:  # Physical threshold (>= 0.1 GWh/yr) so large Hydro values don't hide smaller cells
                    ax.text(
                        j, i, self.style.format_sigfigs(val, 2),
                        ha="center", va="center",
                        color="white" if val > 0.55 * vmax else "black",
                        fontsize=self.style.small_fontsize,
                    )

        ax.set_xticks(range(ncols))
        ax.set_xticklabels(sub_df.columns, rotation=30, ha="left")
        ax.set_yticks(range(nrows))
        ax.set_yticklabels(sub_df.index)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Charging Storage")
        ax.set_ylabel("Supplying Technology")

    def _draw_charging_sources_bar(self, ax: plt.Axes, matrix_df: pd.DataFrame, normalize: bool = False) -> None:
        active_cols = [c for c in matrix_df.columns if matrix_df[c].sum() > 1e-3]
        sub_df = matrix_df[active_cols] if active_cols else matrix_df
        x = np.arange(len(sub_df.columns))
        col_totals = sub_df.sum(axis=0).to_numpy(dtype=npfloat)
        bottom = np.zeros(len(sub_df.columns), dtype=npfloat)

        for src in sub_df.index:
            vals = sub_df.loc[src].to_numpy(dtype=npfloat)
            if np.max(vals) <= 1e-3:
                continue
            heights = np.where(col_totals > 0, vals / col_totals * 100.0, 0.0) if normalize else vals
            ax.bar(
                x, heights, width=0.55, bottom=bottom,
                color=get_color(src), edgecolor="black", linewidth=0.5, label=src,
            )
            bottom += heights

        ax.set_xticks(x)
        ax.set_xticklabels(sub_df.columns)
        ax.set_ylabel("Share of Charging Energy (%)" if normalize else "Charging Energy (GWh/yr)")
        if normalize:
            ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Supplying Source")

    # ==========================================================================
    # 3b. Capacity Factor, Run-Time & Conditional Output Diagnostics
    # ==========================================================================

    def _collect_capacity_factor_groups(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        nodes: Union[str, Sequence[str], None] = None,
        metric: Literal["dispatch", "generation", "discharge", "charge", "post_curtailment_power"] = "dispatch",
        assets: Sequence[str] = ("generators", "storages"),
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        aggregate: Union[bool, str] = False,
        active_rel_tol: float = 1e-4,
    ) -> Dict[str, Dict[str, Union[float, NDArray[npfloat]]]]:
        """
        Aggregates installed power capacity K_G, dispatch trace P_G(t), and active
        online capacity K_active_G(t) = sum_{a in G} K_a * I(P_a(t) > active_rel_tol * K_a).
        """
        target_node_ids = {n.id for n in self._resolve_nodes(nodes)}
        groups: Dict[str, Dict[str, Union[float, NDArray[npfloat]]]] = {}

        for asset_class in assets:
            for asset in self.accessor.get_assets(asset_class).values():
                if hasattr(asset, "node") and asset.node.id not in target_node_ids:
                    continue
                if not self._matches_tech_filter(asset, technologies, group_by):
                    continue

                cap = float(self.accessor.get_power_capacity(asset, errors="coerce")) * self.accessor.factor
                if not np.isfinite(cap) or cap <= self.accessor.tolerance:
                    continue

                trace = np.abs(self.accessor.get_trace(metric, asset))
                thresh = max(active_rel_tol * cap, self.accessor.tolerance)
                is_on = (trace > thresh).astype(npfloat)

                label = self._resolve_label(asset, group_by, aggregate=aggregate, technologies=technologies)
                if label not in groups:
                    groups[label] = {
                        "capacity": cap,
                        "power": trace.copy(),
                        "active_cap": (cap * is_on),
                    }
                else:
                    groups[label]["capacity"] += cap
                    groups[label]["power"] += trace
                    groups[label]["active_cap"] += cap * is_on

        return groups

    def compute_capacity_factor_metrics(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        period="all",
        nodes: Union[str, Sequence[str], None] = None,
        metric: Literal["dispatch", "generation", "discharge", "charge", "post_curtailment_power"] = "dispatch",
        assets: Sequence[str] = ("generators", "storages"),
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        aggregate: Union[bool, str] = False,
        active_rel_tol: float = 1e-4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes:
          1. `summary_df`: Overall period metrics per group (`group_by`):
             - capacity_gw, total_hours, run_hours, run_hours_yr, duty_cycle_pct,
               overall_cf_pct, cf_when_on_pct, avg_output, avg_output_when_on, flh_yr
          2. `diurnal_df`: MultiIndex DataFrame (hour x group) with time-of-day trajectories:
             - 'duty_cycle_pct': Share of installed capacity online at hour h (%)
             - 'overall_cf_pct': Unconditional capacity factor at hour h (%)
             - 'cf_when_on_pct': Conditional capacity factor while online at hour h (%)
             - 'avg_output': Unconditional mean output at hour h (units)
             - 'avg_output_when_on': Mean output when online at hour h (units)
        """
        mask = self._resolve_period_mask(period)
        n_selected = int(np.sum(mask))
        if n_selected == 0:
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        total_hours = n_selected * self.resolution
        annual_scale = 8760.0 / max(total_hours, 1e-9)

        groups = self._collect_capacity_factor_groups(
            technologies=technologies,
            nodes=nodes,
            metric=metric,
            assets=assets,
            group_by=group_by,
            active_rel_tol=active_rel_tol,
            aggregate=aggregate,
        )

        bins = self._tod_bin[mask]
        n_bins = len(self.hours_of_day)
        bin_counts = np.bincount(bins, minlength=n_bins).astype(npfloat)
        valid_bins = bin_counts > 0
        hours_idx = self.hours_of_day[valid_bins]

        summary_rows = []
        diurnal_records: Dict[Tuple[str, str], NDArray[npfloat]] = {}

        for label in sorted(groups.keys()):
            g = groups[label]
            cap = float(g["capacity"])
            p_masked = g["power"][mask]
            act_masked = g["active_cap"][mask]

            sum_p = float(np.sum(p_masked))
            sum_act = float(np.sum(act_masked))
            denom_all = cap * n_selected

            duty_cycle = (sum_act / denom_all) if denom_all > 0 else 0.0
            overall_cf = (sum_p / denom_all) if denom_all > 0 else 0.0
            cf_when_on = (sum_p / sum_act) if sum_act > 0 else 0.0

            run_hours = duty_cycle * total_hours
            summary_rows.append(
                {
                    "group": label,
                    f"capacity_{self.units.lower()}": cap,
                    "total_hours": total_hours,
                    "run_hours": run_hours,
                    "run_hours_yr": run_hours * annual_scale,
                    "duty_cycle_pct": duty_cycle * 100.0,
                    "overall_cf_pct": overall_cf * 100.0,
                    "cf_when_on_pct": cf_when_on * 100.0,
                    f"avg_output_{self.units.lower()}": overall_cf * cap,
                    f"avg_output_when_on_{self.units.lower()}": cf_when_on * cap,
                    "flh_yr": overall_cf * 8760.0,
                }
            )

            # Time-of-day bin aggregations
            p_bin_sum = np.bincount(bins, weights=p_masked, minlength=n_bins)[valid_bins]
            act_bin_sum = np.bincount(bins, weights=act_masked, minlength=n_bins)[valid_bins]
            c_valid = bin_counts[valid_bins]

            tod_duty = act_bin_sum / (cap * c_valid)
            tod_cf = p_bin_sum / (cap * c_valid)
            tod_cf_on = np.where(act_bin_sum > 1e-12, p_bin_sum / np.maximum(act_bin_sum, 1e-12), 0.0)

            diurnal_records[("duty_cycle_pct", label)] = (tod_duty * 100.0).astype(npfloat)
            diurnal_records[("overall_cf_pct", label)] = (tod_cf * 100.0).astype(npfloat)
            diurnal_records[("cf_when_on_pct", label)] = (tod_cf_on * 100.0).astype(npfloat)
            diurnal_records[("avg_output", label)] = (tod_cf * cap).astype(npfloat)
            diurnal_records[("avg_output_when_on", label)] = (tod_cf_on * cap).astype(npfloat)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.set_index("group")

        if diurnal_records:
            diurnal_df = pd.DataFrame(diurnal_records, index=hours_idx)
            diurnal_df.index.name = "hour"
        else:
            diurnal_df = pd.DataFrame(index=hours_idx)

        return summary_df, diurnal_df

    def plot_capacity_factors(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        periods="all",
        nodes: Union[str, Sequence[str], None] = None,
        metric: Literal["dispatch", "generation", "discharge", "charge", "post_curtailment_power"] = "dispatch",
        assets: Sequence[str] = ("generators", "storages"),
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        view: Literal["dashboard", "diurnal", "summary"] = "dashboard",
        output_mode: Literal["cf_pct", "power"] = "cf_pct",
        active_rel_tol: float = 1e-4,
        aggregate: Union[bool, str] = False,
        figsize: Tuple[float, float] | None = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Displays capacity factor, run-time, and conditional 'while on' metrics.
          - view='dashboard': 3-panel plot (Summary Run-Time & CF Bars | Diurnal Duty Cycle | Diurnal Output While On)
          - view='diurnal': Multi-panel diurnal comparison across `periods` (Duty Cycle, Overall CF/Power, Conditional CF/Power)
          - view='summary': Grouped bar chart comparing Duty Cycle (%), Overall CF (%), and CF While On (%)
        """
        period_list = (
            [periods]
            if isinstance(periods, (str, int)) or (isinstance(periods, tuple) and isinstance(periods[0], int))
            else list(periods)
        )
        node_tag = self._format_nodes_title(nodes)

        with plt.rc_context(self.style.to_rc()):
            if view == "summary":
                p0 = period_list[0]
                summary_df, _ = self.compute_capacity_factor_metrics(
                    technologies, period=p0, nodes=nodes, metric=metric,
                    assets=assets, group_by=group_by, active_rel_tol=active_rel_tol,
                    aggregate=aggregate,
                )
                fig, ax = plt.subplots(figsize=figsize or (8.5, 5.0), dpi=self.style.dpi)
                self._draw_cf_summary_bars(
                    ax, summary_df,
                    title=f"Capacity Factor & Run-Time Summary ({self._format_period_title(p0)}{node_tag})",
                )
                out_axes = ax

            elif view == "diurnal":
                n_p = len(period_list)
                fig, axes = plt.subplots(
                    2, n_p,
                    figsize=figsize or (6.5 * n_p, 7.5),
                    dpi=self.style.dpi, sharex=True, sharey="row", squeeze=False,
                )
                for j, p in enumerate(period_list):
                    _, diurnal_df = self.compute_capacity_factor_metrics(
                        technologies, period=p, nodes=nodes, metric=metric,
                        assets=assets, group_by=group_by, active_rel_tol=active_rel_tol,
                        aggregate=aggregate,
                    )
                    self._draw_cf_diurnal_column(
                        axes[0, j], axes[1, j], diurnal_df,
                        output_mode=output_mode,
                        col_title=f"{self._format_period_title(p)}{node_tag}",
                        show_legend=(j == n_p - 1),
                    )
                out_axes = axes

            elif view == "dashboard":
                p0 = period_list[0]
                summary_df, diurnal_df = self.compute_capacity_factor_metrics(
                    technologies, period=p0, nodes=nodes, metric=metric,
                    assets=assets, group_by=group_by, active_rel_tol=active_rel_tol,
                    aggregate=aggregate,
                )
                fig, axes = plt.subplots(
                    1, 3,
                    figsize=figsize or (18.0, 5.2),
                    dpi=self.style.dpi,
                    gridspec_kw={"width_ratios": [1.15, 1.0, 1.0], "wspace": 0.3},
                )
                self._draw_cf_summary_bars(axes[0], summary_df, title="1. Run-Time & CF Breakdown")
                self._draw_cf_diurnal_column(
                    axes[1], axes[2], diurnal_df,
                    output_mode=output_mode,
                    col_title=None,
                    show_legend=True,
                )
                axes[1].set_title("2. Diurnal Duty Cycle (% Cap Online)")
                axes[2].set_title(
                    "3. Diurnal CF: Overall vs While On (%)"
                    if output_mode == "cf_pct"
                    else f"3. Diurnal Output: Overall vs While On ({self.units})"
                )
                fig.suptitle(
                    f"Operational Capacity Factor Diagnostics ({self._format_period_title(p0)}{node_tag})",
                    fontsize=self.style.large_fontsize, y=1.03,
                )
                out_axes = axes
            else:
                raise ValueError(f"Unknown view '{view}'. Choose 'dashboard', 'diurnal', or 'summary'.")

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            return fig, out_axes

    def _draw_cf_summary_bars(self, ax: plt.Axes, summary_df: pd.DataFrame, title: str) -> None:
        if summary_df.empty:
            ax.text(0.5, 0.5, "No active capacity found", ha="center", va="center")
            ax.axis("off")
            return

        labels = list(summary_df.index)
        x = np.arange(len(labels))
        w = 0.26

        duty = summary_df["duty_cycle_pct"].to_numpy()
        cf_on = summary_df["cf_when_on_pct"].to_numpy()
        cf_all = summary_df["overall_cf_pct"].to_numpy()
        run_hrs_yr = summary_df["run_hours_yr"].to_numpy()

        ax.bar(x - w, duty, width=w, color="#3B82F6", edgecolor="black", lw=0.5, label="Time Online / Duty Cycle (%)")
        ax.bar(x, cf_on, width=w, color="#F59E0B", edgecolor="black", lw=0.5, label="Avg CF While On (%)")
        ax.bar(x + w, cf_all, width=w, color="#10B981", edgecolor="black", lw=0.5, label="Overall CF (%)")

        # Annotate annualised run hours above the Duty Cycle bar
        for i, hrs in enumerate(run_hrs_yr):
            ax.text(
                x[i] - w, duty[i] + 1.5, f"{int(round(hrs)):,}h/yr",
                ha="center", va="bottom", fontsize=max(self.style.small_fontsize - 1, 8), rotation=90 if len(labels) > 4 else 0,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
        ax.set_ylim(0.0, 115.0)
        ax.set_ylabel("Percentage of Nameplate (%)")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", frameon=False)

    def _draw_cf_diurnal_column(
        self,
        ax_duty: plt.Axes,
        ax_out: plt.Axes,
        diurnal_df: pd.DataFrame,
        *,
        output_mode: str,
        col_title: str | None,
        show_legend: bool,
    ) -> None:
        if diurnal_df.empty:
            for a in (ax_duty, ax_out):
                a.axis("off")
            return

        hours = diurnal_df.index.to_numpy()
        duty_df = diurnal_df["duty_cycle_pct"]
        uncond_key = "overall_cf_pct" if output_mode == "cf_pct" else "avg_output"
        cond_key = "cf_when_on_pct" if output_mode == "cf_pct" else "avg_output_when_on"

        for label in duty_df.columns:
            color = get_color(label)
            ax_duty.plot(hours, duty_df[label].to_numpy(), color=color, lw=2.0, label=label)

            # Solid = Conditional output while online; Dashed = Unconditional overall average
            ax_out.plot(
                hours, diurnal_df[(cond_key, label)].to_numpy(),
                color=color, lw=2.0, ls="-", label=f"{label} (While On)",
            )
            ax_out.plot(
                hours, diurnal_df[(uncond_key, label)].to_numpy(),
                color=color, lw=1.6, ls="--", alpha=0.8, label=f"{label} (Overall)",
            )

        for a in (ax_duty, ax_out):
            if hours.size > 1:
                a.set_xlim(hours[0], hours[-1])
            a.set_xticks(np.arange(0, 25, 4))
            a.set_xlabel("Hour of Day (h)")
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.grid(True, linestyle="--", alpha=0.3)

        ax_duty.set_ylim(0.0, 105.0)
        ax_duty.set_ylabel("Capacity Online (%)")
        if col_title:
            ax_duty.set_title(f"Duty Cycle: {col_title}")

        if output_mode == "cf_pct":
            ax_out.set_ylim(0.0, 105.0)
            ax_out.set_ylabel("Capacity Factor (%)")
        else:
            ax_out.set_ylabel(f"Output ({self.units})")

        if col_title:
            ax_out.set_title(f"Output (Solid=While On, Dashed=Overall): {col_title}")

        if show_legend:
            ax_out.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    @staticmethod
    def _format_period_title(period) -> str:
        if isinstance(period, tuple) and len(period) == 2:
            return f"Intervals [{period[0]}, {period[1]})" if isinstance(period[1], int) else f"{period[1].title()} {period[0]}"
        if isinstance(period, Sequence) and not isinstance(period, str):
            return ", ".join(str(p).title() for p in period)
        return str(period).title()

    @staticmethod
    def _format_nodes_title(nodes) -> str:
        if nodes is None or (isinstance(nodes, str) and nodes.strip().lower() in ("all", "system")):
            return ""
        if isinstance(nodes, str):
            return f" | {nodes.upper()}"
        return f" | {', '.join(str(n).upper() for n in nodes)}"
