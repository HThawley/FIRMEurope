# type: ignore
import calendar
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from firm_ce.analysis.accessor import Accessor
from firm_ce.common.constants import LEAPDAYS
from firm_ce.common.typing import npfloat, npintp
from firm_ce.scalar.solution import Solution, evaluate
from firm_ce.system.scenario import Scenario
from firm_ce.system.taxonomy import (
    COMPOSITE_GROUPS,
    get_subtech_label,
    identify_tech,
    resolve_asset_label,
)

MONTH_MAP: Dict[str, Tuple[int, ...]] = {
    "jan": (1,),
    "january": (1,),
    "feb": (2,),
    "february": (2,),
    "mar": (3,),
    "march": (3,),
    "apr": (4,),
    "april": (4,),
    "may": (5,),
    "jun": (6,),
    "june": (6,),
    "jul": (7,),
    "july": (7,),
    "aug": (8,),
    "august": (8,),
    "sep": (9,),
    "sept": (9,),
    "september": (9,),
    "oct": (10,),
    "october": (10,),
    "nov": (11,),
    "november": (11,),
    "dec": (12,),
    "december": (12,),
}

SEASON_MAP: Dict[str, Tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "djf": (12, 1, 2),
    "spring": (3, 4, 5),
    "mam": (3, 4, 5),
    "summer": (6, 7, 8),
    "jja": (6, 7, 8),
    "autumn": (9, 10, 11),
    "fall": (9, 10, 11),
    "son": (9, 10, 11),
}

DAYS_NON_LEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
DAYS_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _is_hydro_asset(asset, base_tech: str = "") -> bool:
    raw_type = str(getattr(asset, "unit_type", "")).lower()
    return base_tech in ("Hydro", "Pondage", "Run of River") or raw_type in ("hydro", "pond", "ror")


def _is_rechargeable_storage(asset, base_tech: str = "") -> bool:
    return getattr(asset, "object_class", "") == "storage" and not _is_hydro_asset(asset, base_tech)


def _is_generation_or_hydro(asset, base_tech: str = "") -> bool:
    if getattr(asset, "object_class", "") == "storage":
        return _is_hydro_asset(asset, base_tech)
    return True


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


class BehaviourAnalyzer:
    """
    Headless diurnal (time-of-day) aggregation, raw time-series extraction,
    co-located storage-charging-source attribution, and capacity factor metrics.
    """

    def __init__(self, scenario: Scenario, solution: object, units: str = "GW"):
        self.scenario = scenario
        self.solution = _validate_solution_object(solution)
        self.static = self.solution.static
        self.accessor = Accessor(self.solution, units=units)
        self.units = units

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

    def _matches_tech_filter(self, asset, technologies, group_by) -> bool:
        if technologies is None or technologies == "all":
            return True
        base_tech = identify_tech(asset.name) if getattr(asset, "object_class", "") != "line" else ""
        if callable(technologies):
            return technologies(asset, base_tech)
        if isinstance(technologies, str):
            key = technologies.strip().lower()
            if key == "all":
                return True
            if key in ("storages", "storage"):
                return getattr(asset, "object_class", "") == "storage"
            if key in ("rechargeable_storage", "rechargeable"):
                return _is_rechargeable_storage(asset, base_tech)
            if key in ("generators", "generation"):
                return getattr(asset, "object_class", "") == "generator"
            if key == "generation_or_hydro":
                return _is_generation_or_hydro(asset, base_tech)
            if key == "hydro":
                return _is_hydro_asset(asset, base_tech)
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
            get_subtech_label(asset).lower(),
            str(resolve_asset_label(asset, group_by)).lower(),
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

                label = resolve_asset_label(asset, group_by, aggregate=aggregate)
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
    # 3. Storage Charging Source Attribution
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
        mask = self._resolve_period_mask(period)
        n_selected = int(np.sum(mask))
        if n_selected == 0:
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        years_equiv = (n_selected * self.resolution) / 8760.0
        to_gwh_yr = (1.0 / self.accessor.factor) * self.resolution / max(years_equiv, 1e-9)

        target_nodes = self._resolve_nodes(nodes)
        target_node_ids = {n.id for n in target_nodes}

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

            c_group: Dict[str, NDArray[npfloat]] = {}
            sto_d_src: Dict[str, NDArray[npfloat]] = {}
            sto_c_src: Dict[str, NDArray[npfloat]] = {}

            for sto in node_stos:
                base_tech = identify_tech(sto.name)
                s_lbl = resolve_asset_label(sto, source_group_by)
                d_val = self.accessor.get_discharge_trace(sto)
                c_val = np.abs(self.accessor.get_charge_trace(sto))

                sto_d_src[s_lbl] = sto_d_src[s_lbl] + d_val if s_lbl in sto_d_src else d_val.copy()
                sto_c_src[s_lbl] = sto_c_src[s_lbl] + c_val if s_lbl in sto_c_src else c_val.copy()

                if rechargeable_only and not _is_rechargeable_storage(sto, base_tech):
                    continue
                if not self._matches_tech_filter(sto, storages, storage_group_by):
                    continue

                c_lbl = resolve_asset_label(
                    sto, storage_group_by, aggregate=aggregate_storages, technologies=storages
                )
                c_group[c_lbl] = c_group[c_lbl] + c_val if c_lbl in c_group else c_val.copy()

            if not c_group:
                continue

            net_sto_disch: Dict[str, NDArray[npfloat]] = {}
            for s_lbl in sto_d_src:
                net_d = np.maximum(0.0, sto_d_src[s_lbl] - sto_c_src[s_lbl])
                if np.max(net_d[mask]) > self.accessor.tolerance:
                    net_sto_disch[s_lbl] = net_d

            if not aggregate_storages and source_group_by == storage_group_by:
                for c_lbl in list(c_group.keys()):
                    c_group[c_lbl] = np.maximum(0.0, sto_c_src[c_lbl] - sto_d_src[c_lbl])

            supply_pool: Dict[str, NDArray[npfloat]] = {}

            if sources != "storages":
                raw_gen_pool: Dict[str, NDArray[npfloat]] = {}
                vre_labels = set()
                for gen in gens_by_node.get(nid, []):
                    if sources not in ("all", "generators") and not self._matches_tech_filter(
                        gen, sources, source_group_by
                    ):
                        continue
                    g_lbl = resolve_asset_label(gen, source_group_by)
                    g_val = self.accessor.get_generation_trace(gen)
                    if np.max(g_val[mask]) > self.accessor.tolerance:
                        raw_gen_pool[g_lbl] = raw_gen_pool[g_lbl] + g_val if g_lbl in raw_gen_pool else g_val.copy()
                        if not self.accessor.is_flexible(gen):
                            vre_labels.add(g_lbl)

                if raw_gen_pool:
                    curt = self.accessor.get_curtail_trace(node_obj)
                    vre_tot = sum((raw_gen_pool[k] for k in vre_labels), start=np.zeros_like(curt))
                    curt_vre = np.minimum(curt, vre_tot)
                    vre_scale = np.where(vre_tot > 1e-12, (vre_tot - curt_vre) / np.maximum(vre_tot, 1e-12), 0.0)

                    for g_lbl, g_arr in raw_gen_pool.items():
                        net_g = (g_arr * vre_scale) if g_lbl in vre_labels else g_arr
                        if np.max(net_g[mask]) > self.accessor.tolerance:
                            supply_pool[g_lbl] = net_g

            if sources != "generators":
                for s_lbl, d_arr in net_sto_disch.items():
                    supply_pool[s_lbl] = supply_pool[s_lbl] + d_arr if s_lbl in supply_pool else d_arr.copy()

            if include_imports and sources == "all":
                imp_val = self.accessor.get_import_trace(node_obj)
                if np.max(imp_val[mask]) > self.accessor.tolerance:
                    supply_pool["Net Import"] = imp_val

            if not supply_pool:
                continue

            src_labels = sorted(supply_pool.keys())
            S_mat = np.vstack([supply_pool[k] for k in src_labels])
            S_tot = S_mat.sum(axis=0)

            for c_lbl, c_arr in c_group.items():
                c_masked = np.where(mask, c_arr, 0.0)
                if np.max(c_masked) <= self.accessor.tolerance:
                    continue
                all_charger_labels.add(c_lbl)

                excl_idx = src_labels.index(c_lbl) if c_lbl in src_labels else -1
                denom = (S_tot - S_mat[excl_idx]) if excl_idx >= 0 else S_tot

                active = (c_masked > self.accessor.tolerance) & (denom > self.accessor.tolerance)
                if not np.any(active):
                    continue

                effective_c = np.minimum(c_masked, denom) if sources != "all" else c_masked
                inv_denom = np.where(active, 1.0 / np.maximum(denom, 1e-12), 0.0)

                for i, s_lbl in enumerate(src_labels):
                    if i == excl_idx:
                        continue
                    contrib = effective_c * (S_mat[i] * inv_denom)
                    if np.max(contrib) <= self.accessor.tolerance:
                        continue

                    all_source_labels.add(s_lbl)
                    source_traces[s_lbl] = (
                        source_traces[s_lbl] + contrib if s_lbl in source_traces else contrib.copy()
                    )

                    pair_key = f"{s_lbl} \u2192 {c_lbl}"
                    pair_traces[pair_key] = (
                        pair_traces[pair_key] + contrib if pair_key in pair_traces else contrib.copy()
                    )

                    matrix_accum[(s_lbl, c_lbl)] = matrix_accum.get((s_lbl, c_lbl), 0.0) + float(
                        np.sum(contrib) * to_gwh_yr
                    )

        matrix_df = pd.DataFrame(0.0, index=sorted(all_source_labels), columns=sorted(all_charger_labels))
        for (s_lbl, c_lbl), val in matrix_accum.items():
            matrix_df.loc[s_lbl, c_lbl] = val

        return source_traces, pair_traces, matrix_df

    # ==========================================================================
    # 4. Capacity Factor, Run-Time & Conditional Output Diagnostics
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

                label = resolve_asset_label(asset, group_by, aggregate=aggregate, technologies=technologies)
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
