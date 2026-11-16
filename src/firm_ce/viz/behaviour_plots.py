# type: ignore
from typing import Callable, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from firm_ce.analysis.behaviour_calc import BehaviourAnalyzer
from firm_ce.common.typing import npfloat
from firm_ce.system.scenario import Scenario
from firm_ce.viz.style import PlotStyle, get_color, get_hatch


class Behaviour(BehaviourAnalyzer):
    """
    Calculates and plots diurnal (time-of-day) average behaviour profiles,
    raw time-series traces, and co-located storage-charging-storage diagnostics.
    """

    def __init__(self, scenario: Scenario, solution: object, units: str = "GW"):
        super().__init__(scenario, solution, units=units)
        self.style = PlotStyle()

    # ==========================================================================
    # 1. Standard Behaviour Plotting (Diurnal & Raw)
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
    # 2. Storage Charging Sources & Cross-Charging Plots
    # ==========================================================================

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
                fig, axis = (
                    plt.subplots(figsize=figsize or (6.0, 4.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                )
                axis.text(0.5, 0.5, f"No storage charging detected{title_suffix}", ha="center", va="center")
                axis.axis("off")
                return fig, axis

            if view == "daily_average":
                fig, axis = (
                    plt.subplots(figsize=figsize or (7.5, 5.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                )
                df = self.compute_daily_average(traces_to_plot, period=period, agg=agg)
                self._draw_trace_panel(
                    axis,
                    df.index.to_numpy(),
                    df,
                    chart_type=chart_type,
                    x_label="Hour of Day (h)",
                    title=f"Storage Charging by Source{title_suffix}",
                    show_legend=True,
                )
                axis.set_xticks(np.arange(0, 25, 4))
                axis.set_ylabel(f"Charging Power ({self.units})")

            elif view == "raw_trace":
                fig, axis = (
                    plt.subplots(figsize=figsize or (12.0, 5.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                )
                mask = self._resolve_period_mask(period)
                hours = np.flatnonzero(mask) * self.resolution
                df = pd.DataFrame({k: v[mask] for k, v in traces_to_plot.items()}, index=hours)
                self._draw_trace_panel(
                    axis,
                    hours,
                    df,
                    chart_type=chart_type,
                    x_label="Simulation Time (h)",
                    title=f"Raw Storage Charging by Source{title_suffix}",
                    show_legend=True,
                )
                axis.set_ylabel(f"Charging Power ({self.units})")

            elif view == "matrix":
                fig, axis = (
                    plt.subplots(figsize=figsize or (7.0, 6.0), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                )
                self._draw_cross_charge_matrix(axis, matrix_df)
                axis.set_title(f"Storage Charging Sources [GWh/yr]{title_suffix}", pad=20)

            elif view == "stacked_bar":
                fig, axis = (
                    plt.subplots(figsize=figsize or (7.5, 5.5), dpi=self.style.dpi) if ax is None else (ax.figure, ax)
                )
                self._draw_charging_sources_bar(axis, matrix_df, normalize=normalize)
                axis.set_title(f"Storage Charging Mix{title_suffix}")

            else:
                raise ValueError(f"Unknown view '{view}'.")

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            return fig, axis

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
    # 3. Capacity Factor Diagnostics Plots
    # ==========================================================================

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
                    technologies,
                    period=p0,
                    nodes=nodes,
                    metric=metric,
                    assets=assets,
                    group_by=group_by,
                    active_rel_tol=active_rel_tol,
                    aggregate=aggregate,
                )
                fig, ax = plt.subplots(figsize=figsize or (8.5, 5.0), dpi=self.style.dpi)
                self._draw_cf_summary_bars(
                    ax,
                    summary_df,
                    title=f"Capacity Factor & Run-Time Summary ({self._format_period_title(p0)}{node_tag})",
                )
                out_axes = ax

            elif view == "diurnal":
                n_p = len(period_list)
                fig, axes = plt.subplots(
                    2,
                    n_p,
                    figsize=figsize or (6.5 * n_p, 7.5),
                    dpi=self.style.dpi,
                    sharex=True,
                    sharey="row",
                    squeeze=False,
                )
                for j, p in enumerate(period_list):
                    _, diurnal_df = self.compute_capacity_factor_metrics(
                        technologies,
                        period=p,
                        nodes=nodes,
                        metric=metric,
                        assets=assets,
                        group_by=group_by,
                        active_rel_tol=active_rel_tol,
                        aggregate=aggregate,
                    )
                    self._draw_cf_diurnal_column(
                        axes[0, j],
                        axes[1, j],
                        diurnal_df,
                        output_mode=output_mode,
                        col_title=f"{self._format_period_title(p)}{node_tag}",
                        show_legend=(j == n_p - 1),
                    )
                out_axes = axes

            elif view == "dashboard":
                p0 = period_list[0]
                summary_df, diurnal_df = self.compute_capacity_factor_metrics(
                    technologies,
                    period=p0,
                    nodes=nodes,
                    metric=metric,
                    assets=assets,
                    group_by=group_by,
                    active_rel_tol=active_rel_tol,
                    aggregate=aggregate,
                )
                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=figsize or (18.0, 5.2),
                    dpi=self.style.dpi,
                    gridspec_kw={"width_ratios": [1.15, 1.0, 1.0], "wspace": 0.3},
                )
                self._draw_cf_summary_bars(axes[0], summary_df, title="1. Run-Time & CF Breakdown")
                self._draw_cf_diurnal_column(
                    axes[1],
                    axes[2],
                    diurnal_df,
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
                    fontsize=self.style.large_fontsize,
                    y=1.03,
                )
                out_axes = axes
            else:
                raise ValueError(f"Unknown view '{view}'. Choose 'dashboard', 'diurnal', or 'summary'.")

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")
            return fig, out_axes

    # ==========================================================================
    # 4. Shared Rendering Primitives
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
                    x,
                    df[col].to_numpy(),
                    label=col.replace(" (Electrical)", " (Charge)"),
                    color=get_color(base_col),
                    linestyle=ls,
                    lw=1.8,
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
                elif val >= 0.1:
                    ax.text(
                        j,
                        i,
                        self.style.format_sigfigs(val, 2),
                        ha="center",
                        va="center",
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
                x,
                heights,
                width=0.55,
                bottom=bottom,
                color=get_color(src),
                edgecolor="black",
                linewidth=0.5,
                label=src,
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

        for i, hrs in enumerate(run_hrs_yr):
            ax.text(
                x[i] - w,
                duty[i] + 1.5,
                f"{int(round(hrs)):,}h/yr",
                ha="center",
                va="bottom",
                fontsize=max(self.style.small_fontsize - 1, 8),
                rotation=90 if len(labels) > 4 else 0,
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

            ax_out.plot(
                hours,
                diurnal_df[(cond_key, label)].to_numpy(),
                color=color,
                lw=2.0,
                ls="-",
                label=f"{label} (While On)",
            )
            ax_out.plot(
                hours,
                diurnal_df[(uncond_key, label)].to_numpy(),
                color=color,
                lw=1.6,
                ls="--",
                alpha=0.8,
                label=f"{label} (Overall)",
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
            return (
                f"Intervals [{period[0]}, {period[1]})"
                if isinstance(period[1], int)
                else f"{period[1].title()} {period[0]}"
            )
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
