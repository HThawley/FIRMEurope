import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy.stats import binned_statistic_2d

@njit
def is_pareto_efficient(arr):
    is_efficient = np.arange(arr.shape[0])
    n_points = arr.shape[0]
    next_point_index = 0
    while next_point_index < len(arr):
        nondominated_point_mask = np.empty(len(arr), np.bool_)
        for i in range(len(arr)):
            nondominated_point_mask[i] = np.any(arr[i] < arr[next_point_index])
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        arr = arr[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(n_points, dtype=np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask

class SpacePlotter:
    def __init__(self, scenario, precision: int = 4):
        self.scenario = scenario
        self.config = scenario.config
        self.tech_map = self.scenario.get_tech_index_map()
        
        # Load and cleanly vectorise the generation of both parameterisations
        self.raw_data = self._load_and_deduplicate(precision)
        self.data_abs, self.data_rel = self._aggregate_technologies()
        
        self.opt_lcoe = self.data_abs['objective'].min()

    def _load_and_deduplicate(self, precision: int) -> pd.DataFrame:
        filepath = os.path.join(self.scenario.solution_dir, "population.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find population data at {filepath}")
            
        df = pd.read_csv(filepath, header=None)
        df.rename(columns={0: 'objective', 1: 'violations', 2: 'fitness'}, inplace=True)
        
        df = df[df['violations'] == 0].copy() 
        var_cols = [c for c in df.columns if isinstance(c, int) and c >= 3]
        
        if precision is None:
            df = df.drop_duplicates(subset=var_cols)
        else:
            df = df[~df[var_cols].round(precision).duplicated()]
            
        return df.reset_index(drop=True)

    def _aggregate_technologies(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_raw = self.raw_data.drop(columns=['objective', 'violations', 'fitness']).to_numpy()
        
        # Vectorized conversion across the entire 2D array simultaneously
        if self.config.parameterisation == "relative":
            x_rel = x_raw
            x_abs = self.scenario.convert_x_to_abs(x_rel)
        else:
            x_abs = x_raw
            x_rel = self.scenario.convert_x_to_rel(x_abs)

        df_abs = pd.DataFrame({'objective': self.raw_data['objective']})
        df_rel = pd.DataFrame({'objective': self.raw_data['objective']})

        # Pre-calculate grouped capacities for both parameterisations
        for tech_name, indices in self.tech_map.items():
            if indices:
                df_abs[tech_name] = x_abs[:, indices].sum(axis=1)
                df_rel[tech_name] = x_rel[:, indices].sum(axis=1)

        return df_abs, df_rel

    def _get_data(self, param: str) -> pd.DataFrame:
        if param == "absolute":
            return self.data_abs
        elif param == "relative":
            return self.data_rel
        raise ValueError("param must be 'absolute' or 'relative'")

    def _get_unit_label(self, param: str) -> str:
        return "Absolute" if param == "absolute" else "Relative Scale"

    def plot_bivariate_footprint(
        self, 
        x_var: str, 
        y_var: str, 
        threshold: float = 1.05, 
        isoline_thresholds: list[float] = None,
        bins: int = 40,
        param: str = "absolute",
        ax: plt.Axes = None
    ) -> plt.Axes:
        ax = ax or plt.gca()
        data = self._get_data(param)
        max_obj = self.opt_lcoe * threshold
        valid_df = data[data['objective'] <= max_obj]
        
        sns.kdeplot(
            data=valid_df, x=x_var, y=y_var, 
            fill=True, cmap='Blues', alpha=0.6, ax=ax, thresh=0.05
        )
        
        if isoline_thresholds:
            x_data = data[x_var].to_numpy()
            y_data = data[y_var].to_numpy()
            z_data = data['objective'].to_numpy()
            
            stat, x_edge, y_edge, _ = binned_statistic_2d(
                x_data, y_data, z_data, statistic='min', bins=bins
            )
            
            X, Y = np.meshgrid(x_edge[:-1] + np.diff(x_edge)/2, y_edge[:-1] + np.diff(y_edge)/2)
            Z = stat.T 
            
            levels = [self.opt_lcoe * t for t in isoline_thresholds]
            contour = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.0, alpha=0.8)
            
            fmt = {l: f"{(t - 1.0) * 100:.1f}%" for l, t in zip(contour.levels, isoline_thresholds)}
            ax.clabel(contour, inline=True, fontsize=9, fmt=fmt)

        ax.set_xlabel(f"{x_var} ({self._get_unit_label(param)})")
        ax.set_ylabel(f"{y_var} ({self._get_unit_label(param)})")
        ax.set_title(f"Near-Optimal Footprint (< {threshold}x LCOE)")
        return ax

    def plot_threshold_areas(self, techs: list[str], max_threshold: float = 1.05, steps: int = 15, param: str = "absolute"):
        data = self._get_data(param)
        thresholds = np.linspace(1.0, max_threshold, steps)[::-1]
        
        minima = pd.DataFrame(index=thresholds, columns=techs)
        maxima = pd.DataFrame(index=thresholds, columns=techs)
        
        for thresh in thresholds:
            subset = data[data['objective'] <= self.opt_lcoe * thresh]
            minima.loc[thresh] = subset[techs].min()
            maxima.loc[thresh] = subset[techs].max()
            
        fig, axs = plt.subplots(len(techs), 1, sharex=True, figsize=(8, 2.5 * len(techs)))
        if len(techs) == 1: axs = [axs]
        
        palette = sns.color_palette(n_colors=len(techs))
        
        for i, tech in enumerate(techs):
            axs[i].fill_between(
                thresholds, maxima[tech].astype(float), minima[tech].astype(float), 
                color=palette[i], alpha=0.5
            )
            axs[i].set_ylabel(f"{tech}\n({self._get_unit_label(param)})")
            
        axs[-1].set_xlabel('Near-Optimal Threshold Multiplier')
        axs[-1].set_xlim(max_threshold, 1.0)
        fig.suptitle("Capacity Slack vs. Cost Relaxation")
        fig.tight_layout()
        return fig

    def plot_boxen_distributions(self, threshold: float = 1.05, param: str = "absolute", ax=None):
        data = self._get_data(param)
        valid_df = data[data['objective'] <= self.opt_lcoe * threshold].copy()
        valid_df = valid_df.drop(columns=['objective'])
        
        melted = valid_df.melt(var_name='Technology', value_name='Capacity')
        
        ax = ax or plt.gca()
        sns.boxenplot(data=melted, x='Technology', y='Capacity', k_depth="full", ax=ax)
        ax.tick_params(axis='x', rotation=-45)
        ax.set_ylabel(f"Installed Capacity ({self._get_unit_label(param)})")
        ax.set_title(f"Structural Distributions (Threshold: {threshold})")
        return ax

    def plot_pairwise_structure(self, structural_vars: list[str], threshold: float = 1.05, param: str = "absolute"):
        data = self._get_data(param)
        valid_df = data[data['objective'] <= self.opt_lcoe * threshold][structural_vars].copy()
        
        mask = is_pareto_efficient(valid_df.to_numpy())
        pareto_data = valid_df[mask]
        
        g = sns.PairGrid(valid_df, corner=True, despine=False)
        g.map_lower(sns.kdeplot, fill=True, cmap='Greys', alpha=0.8, thresh=0.05)
        g.map_diag(sns.histplot, color='gray', element="step", stat="density")
        
        g.data = pareto_data
        g.map_lower(sns.scatterplot, marker='x', color='red', s=40, linewidths=1.5, label="Structural Frontier")
        
        g.figure.suptitle(f"Structural Trade-offs within {threshold}x LCOE Envelope ({self._get_unit_label(param)})", y=1.02)
        return g