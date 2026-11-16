# type: ignore
from typing import Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binned_statistic_2d

from firm_ce.analysis.ensemble import EnsembleDataset, _pareto_frontier_2d
from firm_ce.system.scenario import Scenario
from firm_ce.viz.style import PlotStyle, get_color


class SpaceExplorer:
    """
    Unified high-dimensional decision-space and near-optimal ensemble visualiser.
    """

    def __init__(
        self,
        scenario: Scenario,
        dataset: Optional[EnsembleDataset] = None,
        max_relaxation: float | None = 1.25,
        precision: int | None = 1,
    ):
        self.scenario = scenario
        self.style = PlotStyle()
        self.dataset = dataset or EnsembleDataset(
            scenario, max_relaxation=max_relaxation, precision=precision
        )

    @property
    def opt_lcoe(self) -> float:
        return self.dataset.opt_lcoe

    @staticmethod
    def _unit_label(space: str) -> str:
        return "Relative to Nodal Demand" if space.startswith("rel") else "Absolute (GW / GWh)"

    def _compute_2d_grid(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        agg: str = "min",
        bins: Union[int, str] = "quantized",
        max_unique_for_pivot: int = 120,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a 2D gridded surface Z over (x, y) using either fast quantized groupby
        (when `bins='quantized'`) or `binned_statistic_2d` (when `bins` is an int).
        """
        if (
            bins == "quantized"
            and self.dataset.precision is not None
            and df[x].nunique() <= max_unique_for_pivot
            and df[y].nunique() <= max_unique_for_pivot
        ):
            piv = df.groupby([y, x], observed=False)[z].agg(agg).unstack(fill_value=np.nan)
            x_vals = piv.columns.to_numpy(dtype=float)
            y_vals = piv.index.to_numpy(dtype=float)
            X, Y = np.meshgrid(x_vals, y_vals)
            return X, Y, piv.to_numpy(dtype=float)

        n_bins = 40 if isinstance(bins, str) else int(bins)
        stat, x_edge, y_edge, _ = binned_statistic_2d(
            df[x].to_numpy(),
            df[y].to_numpy(),
            df[z].to_numpy(),
            statistic=agg,
            bins=n_bins,
        )
        x_cent = x_edge[:-1] + 0.5 * np.diff(x_edge)
        y_cent = y_edge[:-1] + 0.5 * np.diff(y_edge)
        X, Y = np.meshgrid(x_cent, y_cent)
        return X, Y, stat.T

    # --------------------------------------------------------------------------
    # 1. Bivariate Surfaces & Multi-Panel Heatmaps
    # --------------------------------------------------------------------------

    def plot_surface(
        self,
        x: str,
        y: str,
        z: str = "cost",
        *,
        aggs: Union[str, Sequence[str]] = "min",
        space: Literal["rel", "abs"] = "rel",
        threshold: float | None = 1.15,
        isolines: Optional[Sequence[float]] = None,
        bins: Union[int, str] = "quantized",
        cmap: str = "RdYlBu_r",
        share_cmap: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Union[plt.Axes, Sequence[plt.Axes], None] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        agg_list = [aggs] if isinstance(aggs, str) else list(aggs)
        df = self.dataset.get_frame(space=space, threshold=threshold)

        grids = [self._compute_2d_grid(df, x, y, z, agg=a, bins=bins) for a in agg_list]

        if share_cmap and len(grids) > 1:
            vmin = min(np.nanmin(g[2]) for g in grids)
            vmax = max(np.nanmax(g[2]) for g in grids)
            norm = Normalize(vmin, vmax)
        else:
            norm = None

        with plt.rc_context(self.style.to_rc()):
            if ax is None:
                figsize = figsize or (5.5 * len(agg_list), 4.8)
                fig, axes = plt.subplots(1, len(agg_list), figsize=figsize, dpi=self.style.dpi, squeeze=False)
                axes_flat = axes.flatten()
            else:
                axes_flat = np.atleast_1d(ax).flatten()
                fig = axes_flat[0].figure

            meshes = []
            for i, (agg_name, (X, Y, Z)) in enumerate(zip(agg_list, grids)):
                curr_ax = axes_flat[i]
                mesh = curr_ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=cmap, norm=norm, shading="auto")
                meshes.append(mesh)

                if not share_cmap or len(agg_list) == 1:
                    fig.colorbar(mesh, ax=curr_ax, label=f"{agg_name}({z})")

                if isolines:
                    _, _, Z_cost = self._compute_2d_grid(df, x, y, "cost", agg="min", bins=bins)
                    levels = [self.opt_lcoe * t for t in isolines]
                    cs = curr_ax.contour(X, Y, Z_cost, levels=levels, colors="black", linewidths=1.0, alpha=0.85)
                    fmt = {lvl: f"+{(t - 1.0) * 100:.0f}%" for lvl, t in zip(cs.levels, isolines)}
                    curr_ax.clabel(cs, inline=True, fontsize=self.style.small_fontsize, fmt=fmt)

                curr_ax.set_xlabel(f"{x} [{self._unit_label(space)}]")
                curr_ax.set_ylabel(f"{y} [{self._unit_label(space)}]")
                curr_ax.set_title(f"{agg_name.capitalize()} of {z}")

            if share_cmap and len(agg_list) > 1:
                fig.colorbar(meshes[-1], ax=axes_flat.tolist(), label=z)

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, axes_flat

    # --------------------------------------------------------------------------
    # 2. Pairwise Trade-off Matrix
    # --------------------------------------------------------------------------

    def plot_pairgrid(
        self,
        vars: Sequence[str],
        *,
        z: str = "cost",
        agg: str = "min",
        space: Literal["rel", "abs"] = "rel",
        threshold: float | None = 1.10,
        bins: Union[int, str] = 30,
        corner: bool = True,
        show_pareto: bool = False,
        show_diagonal: bool = True,
        cmap: str = "RdYlBu_r",
        log_color_scale: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        df = self.dataset.get_frame(space=space, threshold=threshold)
        n_vars = len(vars)
        n_grid = n_vars if show_diagonal else n_vars - 1

        if n_grid < 1:
            raise ValueError("Insufficient variables to plot. Need at least 2 variables if show_diagonal=False.")

        figsize = figsize or (3.2 * n_grid, 3.0 * n_grid)

        cell_grids = {}
        vmin, vmax = np.inf, -np.inf
        for r in range(n_vars):
            for c in range(n_vars):
                if r == c or (corner and r < c):
                    continue
                X, Y, Z = self._compute_2d_grid(df, vars[c], vars[r], z, agg=agg, bins=bins)
                cell_grids[(r, c)] = (X, Y, Z)
                vmin = min(vmin, float(np.nanmin(Z)))
                vmax = max(vmax, float(np.nanmax(Z)))

        if np.isfinite(vmin):
            if log_color_scale:
                norm = LogNorm(vmin=1, vmax=max(vmax, 10))
            else:
                norm = Normalize(vmin, vmax) if np.isfinite(vmin) else None
        else:
            norm = None

        with plt.rc_context(self.style.to_rc()):
            fig, axs = plt.subplots(n_grid, n_grid, figsize=figsize, dpi=self.style.dpi, squeeze=False)
            last_mesh = None

            for r in range(n_vars):
                for c in range(n_vars):
                    if r == c and not show_diagonal:
                        continue
                    if corner and r < c:
                        continue

                    ax_r = r if show_diagonal else r - 1
                    ax_c = c

                    if ax_r < 0 or ax_r >= n_grid or ax_c >= n_grid:
                        continue

                    ax = axs[ax_r, ax_c]

                    if r == c:
                        color = get_color(vars[c])
                        ax.hist(df[vars[c]], bins=25, color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
                        ax.set_yticks([])
                    else:
                        X, Y, Z = cell_grids[(r, c)]
                        last_mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=cmap, norm=norm, shading="auto")

                        if show_pareto:
                            valid_pts = np.column_stack([X[~np.isnan(Z)], Y[~np.isnan(Z)]])
                            if len(valid_pts) > 0:
                                p_mask = _pareto_frontier_2d(valid_pts, maximize_x=False, maximize_y=False)
                                ax.scatter(
                                    valid_pts[p_mask, 0],
                                    valid_pts[p_mask, 1],
                                    marker="x",
                                    color="red",
                                    s=25,
                                    linewidths=1.2,
                                    zorder=5,
                                )

                    if ax_r == n_grid - 1:
                        ax.set_xlabel(vars[c])
                    else:
                        ax.set_xticklabels([])

                    if ax_c == 0:
                        ax.set_ylabel(vars[r])
                    elif r != c:
                        ax.set_yticklabels([])

            if corner and not show_diagonal:
                for ax_r in range(n_grid):
                    for ax_c in range(n_grid):
                        if ax_r < ax_c:
                            axs[ax_r, ax_c].axis("off")

            if last_mesh is not None:
                fig.colorbar(last_mesh, ax=axs.ravel().tolist(), label=f"{agg}({z})", shrink=0.7)

            fig.suptitle(
                f"Pairwise Trade-offs ({self._unit_label(space)}, <= {threshold}x Opt LCOE)",
                fontsize=self.style.large_fontsize,
                y=1.01,
            )
            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, axs

    # --------------------------------------------------------------------------
    # 3. Capacity Slack vs. Cost Relaxation Ribbons
    # --------------------------------------------------------------------------

    def plot_slack_ribbons(
        self,
        vars: Sequence[str],
        *,
        max_threshold: float = 1.15,
        steps: int = 25,
        quantiles: Tuple[float, float] = (0.10, 0.90),
        space: Literal["rel", "abs"] = "rel",
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        df = self.dataset.get_frame(space=space, threshold=max_threshold)
        thresholds = np.linspace(1.0, max_threshold, steps)

        df_sorted = df.sort_values("cost")
        costs = df_sorted["cost"].to_numpy()

        mins = np.full((steps, len(vars)), np.nan)
        maxs = np.full((steps, len(vars)), np.nan)
        q_low = np.full((steps, len(vars)), np.nan)
        q_high = np.full((steps, len(vars)), np.nan)

        vals = df_sorted[list(vars)].to_numpy()
        cutoff_indices = np.searchsorted(costs, self.opt_lcoe * thresholds, side="right")

        for i, end_idx in enumerate(cutoff_indices):
            if end_idx == 0:
                continue
            sub = vals[:end_idx]
            mins[i] = sub.min(axis=0)
            maxs[i] = sub.max(axis=0)
            if quantiles is not None and end_idx > 2:
                q_low[i], q_high[i] = np.quantile(sub, quantiles, axis=0)

        with plt.rc_context(self.style.to_rc()):
            figsize = figsize or (8.5, 2.2 * len(vars))
            fig, axs = plt.subplots(len(vars), 1, sharex=True, figsize=figsize, dpi=self.style.dpi, squeeze=False)
            axes_flat = axs.flatten()

            pct_relaxation = (thresholds - 1.0) * 100.0
            for j, tech in enumerate(vars):
                ax = axes_flat[j]
                color = get_color(tech)
                ax.fill_between(pct_relaxation, mins[:, j], maxs[:, j], color=color, alpha=0.3, label="Min–Max")
                if quantiles is not None:
                    ax.fill_between(
                        pct_relaxation,
                        q_low[:, j],
                        q_high[:, j],
                        color=color,
                        alpha=0.65,
                        label=f"P{int(quantiles[0]*100)}–P{int(quantiles[1]*100)}",
                    )
                ax.set_ylabel(tech)
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            axes_flat[0].legend(loc="upper right", frameon=False)
            axes_flat[-1].set_xlabel("Cost Relaxation above Optimal LCOE (%)")
            axes_flat[-1].set_xlim(pct_relaxation[-1], 0.0)
            fig.suptitle(f"Capacity Slack vs. Cost Relaxation [{self._unit_label(space)}]")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, axes_flat

    # --------------------------------------------------------------------------
    # 4. 1D Marginal Distributions (Boxen / Violin)
    # --------------------------------------------------------------------------

    def plot_distributions(
        self,
        vars: Optional[Sequence[str]] = None,
        *,
        threshold: float = 1.05,
        space: Literal["rel", "abs"] = "rel",
        kind: Literal["boxen", "violin", "box"] = "boxen",
        max_sample: int = 100_000,
        figsize: Tuple[float, float] = (11, 5),
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        df = self.dataset.get_frame(space=space, threshold=threshold)
        if vars is None:
            vars = [c for c in self.dataset.taxonomy.columns if c in df.columns and df[c].max() > 0]

        if len(df) > max_sample:
            df = df.sample(n=max_sample, random_state=42)

        melted = df[list(vars)].melt(var_name="Technology", value_name="Capacity")
        palette = {tech: get_color(tech) for tech in vars}

        with plt.rc_context(self.style.to_rc()):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, dpi=self.style.dpi)
            else:
                fig = ax.figure

            if kind == "boxen":
                sns.boxenplot(
                    data=melted, x="Technology", y="Capacity", palette=palette, k_depth="trustworthy", ax=ax
                )
            elif kind == "violin":
                sns.violinplot(data=melted, x="Technology", y="Capacity", palette=palette, cut=0, ax=ax)
            else:
                sns.boxplot(data=melted, x="Technology", y="Capacity", palette=palette, ax=ax)

            ax.tick_params(axis="x", rotation=-40)
            ax.set_ylabel(self._unit_label(space))
            ax.set_title(f"Near-Optimal Capacity Distributions (<= {threshold}x Opt LCOE)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if save_path:
                fig.savefig(save_path, dpi=self.style.dpi, bbox_inches="tight")

            return fig, ax
