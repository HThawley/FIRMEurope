# type: ignore
import calendar
from typing import Callable, Dict, Literal, Sequence, Tuple, Union
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
    "jan": (1,), "january": (1,),
    "feb": (2,), "february": (2,),
    "mar": (3,), "march": (3,),
    "apr": (4,), "april": (4,),
    "may": (5,),
    "jun": (6,), "june": (6,),
    "jul": (7,), "july": (7,),
    "aug": (8,), "august": (8,),
    "sep": (9,), "sept": (9,), "september": (9,),
    "oct": (10,), "october": (10,),
    "nov": (11,), "november": (11,),
    "dec": (12,), "december": (12,),
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
    """Checks the passed solution object directly, raising NotImplementedError for tensors."""
    if sol is None:
        raise ValueError("Passed solution object is None.")
    cls_name = type(sol).__name__
    if cls_name == "SolutionTensor" or hasattr(sol, "ops") or hasattr(sol, "staticTensor"):
        raise NotImplementedError(
            f"Behaviour trace analysis received '{cls_name}', which is not yet supported. "
            "Pass a scalar 'Solution' instance (or use map_tensor_to_scalar first)."
        )
    if not isinstance(sol, Solution):
        raise TypeError(f"Expected a scalar 'Solution' object, got '{cls_name}'.")
    if not sol.evaluated:
        evaluate(sol)
    return sol


class Behaviour:
    """
    Calculates and plots diurnal (time-of-day) average behaviour profiles
    and raw time-series traces from evaluated scalar Solution objects.
    """

    def __init__(
        self,
        scenario: Scenario,
        solution: object,
        units: str = "GW",
    ):
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

    # --------------------------------------------------------------------------
    # Calendar & Filtering Helpers (Aligned with parameter_cons.py)
    # --------------------------------------------------------------------------

    def _build_calendar_arrays(self) -> Tuple[NDArray[npintp], NDArray[npintp], NDArray[npfloat]]:
        """
        Uses `ScenarioParameters` (`year_first_t`, `first_year`, `resolution`, and `LEAPDAYS`)
        to map every interval `t` to its calendar month (1..12) and time-of-day bin index.
        """
        month_of_interval = np.empty(self.intervals_count, dtype=npintp)
        hour_of_day_per_t = np.empty(self.intervals_count, dtype=npfloat)

        year_first_t = np.asarray(self.static.year_first_t, dtype=npintp)
        n_years_present = len(year_first_t)

        for y_idx in range(n_years_present):
            start_t = int(year_first_t[y_idx])
            end_t = (
                int(year_first_t[y_idx + 1])
                if (y_idx + 1 < n_years_present)
                else self.intervals_count
            )
            end_t = min(end_t, self.intervals_count)
            if start_t >= end_t:
                continue

            cal_year = self.first_year + y_idx
            is_leap = bool(LEAPDAYS and calendar.isleap(cal_year))
            month_days = DAYS_LEAP if is_leap else DAYS_NON_LEAP
            cum_hours = np.cumsum(month_days) * 24.0

            t_local = np.arange(end_t - start_t, dtype=npfloat)
            hour_in_year = t_local * self.resolution

            # Assign month 1..12 based on cumulative hours in the specific calendar year
            months = np.searchsorted(cum_hours, hour_in_year, side="right") + 1
            month_of_interval[start_t:end_t] = np.clip(months, 1, 12)

            # Time-of-day in [0, 24) hours, robust to interval_aggregation
            hour_of_day_per_t[start_t:end_t] = np.round(np.mod(hour_in_year, 24.0), decimals=6)

        unique_hours, tod_bin = np.unique(hour_of_day_per_t, return_inverse=True)
        return month_of_interval, tod_bin.astype(npintp), unique_hours.astype(npfloat)

    def _resolve_period_mask(
        self,
        period: Union[str, int, Sequence[Union[str, int]], Tuple[int, int], Tuple[int, str]],
    ) -> NDArray[np.bool_]:
        """
        Returns a boolean mask of length `intervals_count`.
        Supports:
          - 'all' / 'annual' / 'full'
          - month names ('jan', 'july') or season names ('winter', 'jja')
          - calendar year int/str (2025 or '2025')
          - (year, month_or_season) tuple, e.g. (2025, 'winter')
          - explicit interval slice tuple (start_idx, end_idx)
          - sequence of keywords, e.g. ['dec', 'jan']
        """
        # 1. Tuple: either (year, period_str) or (start_interval, end_interval)
        if isinstance(period, tuple) and len(period) == 2:
            a, b = period
            if isinstance(a, int) and isinstance(b, str):
                return self._resolve_year_mask(a) & self._resolve_period_mask(b)
            if isinstance(a, int) and isinstance(b, int):
                mask = np.zeros(self.intervals_count, dtype=bool)
                mask[a:b] = True
                return mask

        # 2. Integer year
        if isinstance(period, int):
            return self._resolve_year_mask(period)

        # 3. String keyword (month, season, 'all', or year string)
        if isinstance(period, str):
            key = period.strip().lower()
            if key in ("all", "annual", "full"):
                return np.ones(self.intervals_count, dtype=bool)
            if key.isdigit():
                return self._resolve_year_mask(int(key))
            if key in MONTH_MAP:
                target_months = MONTH_MAP[key]
            elif key in SEASON_MAP:
                target_months = SEASON_MAP[key]
            else:
                valid = list(MONTH_MAP.keys()) + list(SEASON_MAP.keys()) + ["all"]
                raise ValueError(f"Unknown period keyword '{period}'. Valid keywords: {valid}")
            return np.isin(self._month_of_interval, target_months)

        # 4. Sequence of periods (union)
        if isinstance(period, Sequence):
            mask = np.zeros(self.intervals_count, dtype=bool)
            for p in period:
                mask |= self._resolve_period_mask(p)
            return mask

        raise ValueError(f"Unsupported period specification: {period}")

    def _resolve_year_mask(self, year: int) -> NDArray[np.bool_]:
        """Filters intervals by calendar year (e.g. 2025) or 0-indexed simulation year."""
        year_idx = (year - self.first_year) if year >= self.first_year else year
        if not (0 <= year_idx < self.year_count):
            raise ValueError(
                f"Year '{year}' out of range for scenario years "
                f"[{self.first_year}, {self.first_year + self.year_count - 1}]."
            )
        return np.asarray(self.static.year_of_interval) == year_idx

    def _resolve_label(self, asset, group_by: Union[str, Callable]) -> str:
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
                return asset.node.name if hasattr(asset, "node") else f"{asset.node_start.name}-{asset.node_end.name}"
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

    def _matches_tech_filter(
        self,
        asset,
        technologies: Union[str, Sequence[str], Callable],
        group_by: Union[str, Callable],
    ) -> bool:
        if technologies == "all" or technologies is None:
            return True
        if callable(technologies):
            base_tech = self.scenario.identify_tech(asset.name) if getattr(asset, "object_class", "") != "line" else ""
            return technologies(asset, base_tech)

        if isinstance(technologies, str):
            key = technologies.strip().lower()
            if key == "all":
                return True
            if key in ("storages", "storage"):
                return getattr(asset, "object_class", "") == "storage"
            if key in ("rechargeable_storage", "rechargeable"):
                base_tech = self.scenario.identify_tech(asset.name)
                return is_rechargeable_storage(asset, base_tech)
            if key in ("generators", "generation"):
                return getattr(asset, "object_class", "") == "generator"
            if key == "generation_or_hydro":
                base_tech = self.scenario.identify_tech(asset.name)
                return is_generation_or_hydro(asset, base_tech)
            if key == "hydro":
                base_tech = self.scenario.identify_tech(asset.name)
                return is_hydro_asset(asset, base_tech)
            tech_list = [technologies]
        else:
            tech_list = list(technologies)

        allowed_tokens = set()
        for t in tech_list:
            t_low = t.strip().lower()
            allowed_tokens.add(t_low)
            if t_low in COMPOSITE_GROUPS:
                allowed_tokens.update(u.lower() for u in COMPOSITE_GROUPS[t_low][0])

        candidates = {
            asset.name.lower(),
            str(getattr(asset, "unit_type", "")).lower(),
            self.scenario.identify_tech(asset.name).lower() if getattr(asset, "object_class", "") != "line" else "",
            self._get_subtech_label(asset).lower(),
            str(self._resolve_label(asset, group_by)).lower(),
        }
        return bool(allowed_tokens & candidates)

    # --------------------------------------------------------------------------
    # Core Trace Calculation
    # --------------------------------------------------------------------------

    def get_traces(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        metric: str = "dispatch_and_charge",
        assets: Sequence[str] = ("generators", "storages"),
        node: str | None = None,
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        include_demand: bool = False,
        include_curtailment: bool = False,
    ) -> Dict[str, NDArray[npfloat]]:
        """
        Extracts full-horizon 1D behaviour traces grouped by `group_by`.
        If `metric='dispatch_and_charge'`, storage charging is returned as separate
        negative traces (`"<Label> (Electrical)"`) alongside positive discharge (`"<Label>"`).
        """
        traces: Dict[str, NDArray[npfloat]] = {}

        def _add(label: str, arr: NDArray[npfloat]):
            if label not in traces:
                traces[label] = arr.copy()
            else:
                traces[label] += arr

        for asset_class in assets:
            for asset in self.accessor.get_assets(asset_class).values():
                if node is not None:
                    asset_node = getattr(asset, "node", None)
                    if asset_node is None or asset_node.name.lower() != node.lower():
                        continue
                if not self._matches_tech_filter(asset, technologies, group_by):
                    continue

                label = self._resolve_label(asset, group_by)

                if metric == "dispatch_and_charge":
                    if getattr(asset, "object_class", "") == "generator":
                        _add(label, self.accessor.get_generation_trace(asset))
                    elif getattr(asset, "object_class", "") == "storage":
                        _add(label, self.accessor.get_discharge_trace(asset))
                        _add(f"{label} (Electrical)", self.accessor.get_charge_trace(asset))
                else:
                    _add(label, self.accessor.get_trace(metric, asset))

        if include_curtailment:
            scope = self._get_node_obj(node) if node else "system"
            _add("Curtailment", -self.accessor.get_curtail_trace(scope))

        if include_demand:
            scope = self._get_node_obj(node) if node else "system"
            _add("Demand", self.accessor.get_demand_trace(scope))

        return {k: v for k, v in traces.items() if np.max(np.abs(v)) > self.accessor.tolerance}

    def _get_node_obj(self, node_name: str):
        for n in self.accessor.get_assets("nodes").values():
            if n.name.lower() == node_name.lower():
                return n
        raise ValueError(f"Node '{node_name}' not found in solution network.")

    def compute_daily_average(
        self,
        traces: Dict[str, NDArray[npfloat]],
        period: Union[str, int, Sequence[Union[str, int]], Tuple[int, int], Tuple[int, str]] = "all",
        agg: Literal["mean", "median"] = "mean",
    ) -> pd.DataFrame:
        """
        Filters `traces` by `period` and aggregates across all matching intervals
        for each time-of-day bin (`self.hours_of_day`). Uses `np.bincount` for fast
        vectorised means without requiring `limit_timesteps` or `interval_aggregation`
        to be an exact multiple of 24 hours.
        """
        mask = self._resolve_period_mask(period)
        if not np.any(mask):
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        bins = self._tod_bin[mask]
        n_bins = len(self.hours_of_day)
        counts = np.bincount(bins, minlength=n_bins).astype(npfloat)
        valid_bins = counts > 0

        out: Dict[str, NDArray[npfloat]] = {"hour": self.hours_of_day[valid_bins]}
        for label, arr in traces.items():
            vals = arr[mask]
            if agg == "mean":
                sums = np.bincount(bins, weights=vals, minlength=n_bins)
                out[label] = (sums[valid_bins] / counts[valid_bins]).astype(npfloat)
            elif agg == "median":
                out[label] = np.array(
                    [np.median(vals[bins == b]) for b in np.flatnonzero(valid_bins)],
                    dtype=npfloat,
                )
            else:
                raise ValueError(f"Unknown aggregation '{agg}'. Expected 'mean' or 'median'.")

        return pd.DataFrame(out).set_index("hour")

    # --------------------------------------------------------------------------
    # Plotting Methods
    # --------------------------------------------------------------------------

    def plot_daily_average(
        self,
        technologies: Union[str, Sequence[str], Callable] = "all",
        *,
        periods: Union[str, int, Sequence[Union[str, int, Tuple[int, str]]]] = "all",
        metric: str = "dispatch_and_charge",
        assets: Sequence[str] = ("generators", "storages"),
        node: str | None = None,
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        chart_type: Literal["stacked", "line"] = "stacked",
        include_demand: bool = True,
        include_curtailment: bool = False,
        agg: Literal["mean", "median"] = "mean",
        figsize: Tuple[float, float] | None = None,
        ax: Union[plt.Axes, Sequence[plt.Axes], None] = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plots average time-of-day behaviour for `technologies` across one or more `periods`
        (e.g., `periods='all'`, `periods=['winter', 'summer']`, `periods=[(2024, 'jan'), (2025, 'jan')]`).
        """
        if isinstance(periods, (str, int)) or (
            isinstance(periods, tuple) and len(periods) == 2 and isinstance(periods[0], int)
        ):
            period_list = [periods]
        else:
            period_list = list(periods)

        traces = self.get_traces(
            technologies=technologies,
            metric=metric,
            assets=assets,
            node=node,
            group_by=group_by,
            include_demand=include_demand,
            include_curtailment=include_curtailment,
        )

        n_panels = len(period_list)
        with plt.rc_context(self.style.to_rc()):
            if ax is None:
                if figsize is None:
                    figsize = (6.5 * n_panels, 5.0)
                fig, axes = plt.subplots(
                    1, n_panels, figsize=figsize, dpi=self.style.dpi, sharey=True, squeeze=False
                )
                axes_flat = axes.flatten()
            else:
                axes_flat = np.atleast_1d(ax).flatten()
                fig = axes_flat[0].figure

            for i, p in enumerate(period_list):
                df = self.compute_daily_average(traces, period=p, agg=agg)
                x_vals = df.index.to_numpy()
                self._draw_trace_panel(
                    axes_flat[i],
                    x_vals,
                    df,
                    chart_type=chart_type,
                    x_label="Hour of Day (h)",
                    title=f"Period: {self._format_period_title(p)}",
                    show_legend=(i == n_panels - 1),
                )
                if x_vals.size > 1:
                    axes_flat[i].set_xlim(x_vals[0], x_vals[-1])
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
        period: Union[str, int, Tuple[int, int], Tuple[int, str]] = (0, 168),
        metric: str = "dispatch_and_charge",
        assets: Sequence[str] = ("generators", "storages"),
        node: str | None = None,
        group_by: Union[Literal["tech", "subtech", "unit_type", "node", "asset"], Callable] = "subtech",
        chart_type: Literal["stacked", "line"] = "stacked",
        include_demand: bool = True,
        include_curtailment: bool = False,
        figsize: Tuple[float, float] = (12.0, 5.0),
        ax: plt.Axes | None = None,
        save_path: str | None = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots raw, unaveraged behaviour time series over a specified interval window
        `(start_interval, end_interval)` or calendar keyword (`'jan'`, `'winter'`, `(2025, 'jan')`, `'all'`).
        """
        traces = self.get_traces(
            technologies=technologies,
            metric=metric,
            assets=assets,
            node=node,
            group_by=group_by,
            include_demand=include_demand,
            include_curtailment=include_curtailment,
        )
        mask = self._resolve_period_mask(period)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError(f"Period '{period}' matched 0 intervals.")

        hours = indices * self.resolution
        df = pd.DataFrame({k: v[mask] for k, v in traces.items()}, index=hours)

        with plt.rc_context(self.style.to_rc()):
            if ax is None:
                fig, axis = plt.subplots(figsize=figsize, dpi=self.style.dpi)
            else:
                axis = ax
                fig = axis.figure

            title_str = (
                f"Intervals [{period[0]}, {period[1]})"
                if (isinstance(period, tuple) and isinstance(period[1], int))
                else f"Raw Trace: {self._format_period_title(period)}"
            )
            self._draw_trace_panel(
                axis,
                hours,
                df,
                chart_type=chart_type,
                x_label="Simulation Time (h)",
                title=title_str,
                show_legend=True,
            )
            if hours.size > 1:
                axis.set_xlim(hours[0], hours[-1])
            axis.set_ylabel(f"{metric.replace('_', ' ').title()} ({self.units})")

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, axis

    @staticmethod
    def _format_period_title(period) -> str:
        if isinstance(period, tuple) and len(period) == 2 and isinstance(period[1], str):
            return f"{period[1].title()} {period[0]}"
        if isinstance(period, Sequence) and not isinstance(period, str):
            return ", ".join(str(p).title() for p in period)
        return str(period).title()

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

        if chart_type == "line":
            for col in tech_cols:
                y = df[col].to_numpy()
                ls = "--" if col.endswith(" (Electrical)") else "-"
                ax.plot(
                    x,
                    y,
                    label=col.replace(" (Electrical)", " (Charge)"),
                    color=get_color(col),
                    linestyle=ls,
                    lw=1.8,
                )
        elif chart_type == "stacked":
            pos_bottom = np.zeros_like(x, dtype=npfloat)
            neg_bottom = np.zeros_like(x, dtype=npfloat)

            for col in sorted(tech_cols):
                y = df[col].to_numpy()
                y_pos = np.maximum(0.0, y)
                y_neg = np.minimum(0.0, y)

                color = get_color(col)
                hatch = get_hatch(col)
                clean_label = col.replace(" (Electrical)", " (Charge)")

                if np.any(y_pos > 1e-6):
                    ax.fill_between(
                        x,
                        pos_bottom,
                        pos_bottom + y_pos,
                        facecolor=color,
                        hatch=hatch,
                        edgecolor="black" if hatch else "none",
                        linewidth=0.3 if hatch else 0.0,
                        alpha=0.9,
                        label=clean_label,
                    )
                    pos_bottom += y_pos

                if np.any(y_neg < -1e-6):
                    ax.fill_between(
                        x,
                        neg_bottom + y_neg,
                        neg_bottom,
                        facecolor=color,
                        hatch=hatch or "//",
                        edgecolor="black",
                        linewidth=0.3,
                        alpha=0.85,
                        label=clean_label if not np.any(y_pos > 1e-6) else f"{clean_label} (Neg)",
                    )
                    neg_bottom += y_neg
        else:
            raise ValueError(f"Unknown chart_type '{chart_type}'. Expected 'stacked' or 'line'.")

        if demand_series is not None:
            ax.plot(x, demand_series, color="black", lw=2.0, ls="-", label="Demand", zorder=10)

        ax.axhline(0.0, color="black", lw=0.8, zorder=5)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
