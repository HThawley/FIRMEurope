import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import os
import pandas as pd
import warnings

from firm_ce.common.typing import npfloat
from firm_ce.system.scenario import Scenario
from firm_ce.backend.scalar.solution import Solution, evaluate
from firm_ce.system.scalar.parameters import ModelConfig
from firm_ce.analysis.accessor import Accessor


class Display:
    """
    Visualizes optimization results for the European energy system based on a
    Solution objects.
    """

    def __init__(
        self,
        scenario: Scenario,
        config: ModelConfig,
        solution: Solution = None,
        noptima: List[Solution] = None,
    ):
        """
        Initialize the plotter with a Solution object and map data.
        """
        self.scenario = scenario
        self.config = config
        self.mhmga = self.config.type == "mhmga"

        self.noptima = []
        self.solution = None

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
            elif hasattr(scenario, 'statistics'):
                self.solution = scenario.statistics.solution
            else:
                self._read_and_evaluate_optimum()
            if not self.solution.evaluated:
                evaluate(solution)

        self._load_map_data("./inputs/map/europe.geojson")
        self._init_colors()

    def plot(
        self,
        data_type: str = "energy",
        *,
        atlas: bool = False,
        delta: bool = False,
        **kwargs,
    ):
        return self._dispatch_plot(data_type=data_type, atlas=atlas, delta=delta, **kwargs)

    def plot_energy_mix(
        self,
        *,
        atlas: bool = False,
        delta: bool = False,
        **kwargs,
    ):
        """
        Visualizes the energy generation mix (GWh) across the network.

        Args:
            mode (str): Visualization layout.
            **kwargs:
                atlas: Grid of MGA alternatives (requires MHMGA config).
                delta: Comparative bar chart between two solutions (requires MHMGA config).
                alternative (int): Index of the solution to plot in 'single' mode. Default 0.
                alt_a, alt_b (int): Indices for comparison in 'delta' mode.
                curtailment (bool): If True, plots post-curtailment net energy. Default False.
                chart_type (str): 'pie' or 'bar'. Note: 'delta' mode only supports 'bar'.
                max_scale (float): Value used to normalize chart sizes across the map.
                threshold (float): Minimum value to display in charts. Default 1e-5.
                legend (bool): Toggle the technology legend. Default True.
                save_path (str): File path to save the resulting figure.
        """
        return self._dispatch_plot(data_type="energy", atlas=atlas, delta=delta, **kwargs)

    def plot_power_capacity(
        self,
        *,
        atlas: bool = False,
        delta: bool = False,
        **kwargs,
    ):
        """
        Visualizes installed power capacity (GW) across the network.

        Args:
            mode (str): Visualization layout ('single', 'atlas', or 'delta').
            **kwargs:
                build (str): Filters capacity by investment status.
                    'all' or None: Total installed capacity.
                    'new_build': Only capacity added by the optimizer.
                    'existing' or 'initial': Only starting/brownfield capacity.
                alternative (int): Solution index for 'single' mode.
                chart_type (str): 'pie' or 'bar'.
                max_scale (float): Normalization constant for pie/bar scaling.
                legend (bool): Toggle legend visibility.
                save_path (str): File path to save the figure.
        """
        return self._dispatch_plot(data_type="capacity", atlas=atlas, delta=delta, **kwargs)

    def plot_fleet_summary(self, energy_type="both", save_path=None):
        """
        Plots a 3x3 grid of pie charts summarizing the fleet:
        Row 1: Power Capacities (Generation, Transmission, Storage)
        Row 2: Energy Capacities (Hydro, Storage) & Storage Charging Sources
        Row 3: Energy Flow (Generation, Transmission, Storage)
        """
        import seaborn as sns
        import numpy as np
        
        if energy_type == "consumption":
            raise NotImplementedError("Methods for attributing curtailment are not yet defined.")
        elif energy_type not in ["generation", "both"]:
            raise ValueError("energy_type must be 'consumption', 'generation', or 'both'")

        accessor = Accessor(self.solution, "GW")
        year_count = self.solution.static.year_count
        
        # Organize data by layout rows
        row1 = {"Generation (Power)": {}, "Transmission (Power)": {}, "Rechargeable Storage (Power)": {}}
        row2 = {"Hydro (Energy Cap)": {}, "Rechargeable Storage (Energy Cap)": {}, "Rechargeable Storage (Sources)": {}}
        row3 = {"Generation (Energy Mix)": {}, "Transmission (Flows)": {}, "Rechargeable Storage (Discharge)": {}}

        # Set up aesthetics
        tx_palette = sns.color_palette("mako", 4)
        src_palette = sns.color_palette("Set2", 8)
        
        custom_colors = {
            "AC OHL": tx_palette[0],
            "AC OHL (Mountain)": tx_palette[1],
            "DC Subsea": tx_palette[2],
            "DC Underground": tx_palette[3],
            "Losses": (0.7, 0.7, 0.7),
            "Curtailment": (0.7, 0.7, 0.7),
            "Spillage": (0.6, 0.8, 0.9),
            "Bioenergy": (0.4, 0.7, 0.3),
            "New PHES": sns.color_palette("Paired")[1],
            "Legacy PHES": sns.color_palette("Paired")[0],
            "Battery": sns.color_palette("Paired")[10],
            "New PHES (Electrical)": src_palette[0],
            "Legacy PHES (Electrical)": src_palette[1],
            "Legacy PHES (Inflows)": src_palette[2],
            "Battery (Electrical)": src_palette[3],
        }

        def get_chart_color(label):
            if label in custom_colors:
                return custom_colors[label]
            if hasattr(self, '_get_color'):
                color = self._get_color(label)
                if color != (0.5, 0.5, 0.5) or label == "Fossil Gas":
                    return color
            return (0.5, 0.5, 0.5)

        def process_tech_label(asset):
            tech = self.scenario.identify_tech(asset.name)
            if tech in ["Biomass", "Biogas"]:
                return "Bioenergy"
            return tech

        def process_storage_label(asset):
            raw_type = str(getattr(asset, 'unit_type', '')).lower()
            if raw_type == 'nphes':
                return 'New PHES'
            elif raw_type in ['clphes', 'olphes']:
                return 'Legacy PHES'
            elif 'bess' in raw_type or raw_type == 'battery':
                return 'Battery'
            return process_tech_label(asset)

        # 1. Generators 
        for asset in accessor.get_assets("generators").values():
            label = process_tech_label(asset)
            
            row1["Generation (Power)"][label] = row1["Generation (Power)"].get(label, 0) + accessor.get_power_capacity(asset)
            
            energy_yr = (accessor.get_discharge_net(asset) / year_count) / 1000
            row3["Generation (Energy Mix)"][label] = row3["Generation (Energy Mix)"].get(label, 0) + energy_yr
            
            if energy_type == "both":
                curtail_yr = (accessor.get_nominal_curtailment_net(asset) / year_count) / 1000
                row3["Generation (Energy Mix)"]["Curtailment"] = row3["Generation (Energy Mix)"].get("Curtailment", 0) + curtail_yr

        # 2. Storages (Routing Hydro to Generation, extracting BESS/PHES to Storage)
        for asset in accessor.get_assets("storages").values():
            base_tech = process_tech_label(asset)
            raw_type = str(getattr(asset, 'unit_type', '')).lower()
            
            # Hydro leaks into storage charts if the raw_type doesn't perfectly match "hydro" (e.g., "pond" or "ror"). 
            # Checking base_tech isolates it cleanly.
            if base_tech in ["Hydro", "Run of River"] or raw_type in ["hydro", "pond", "ror"]:
                label = base_tech
                
                # Power and Discharge route to Generation
                row1["Generation (Power)"][label] = row1["Generation (Power)"].get(label, 0) + accessor.get_power_capacity(asset)
                energy_yr = (accessor.get_discharge_net(asset) / year_count) / 1000
                row3["Generation (Energy Mix)"][label] = row3["Generation (Energy Mix)"].get(label, 0) + energy_yr
                
                if energy_type == "both":
                    curtail_yr = (accessor.get_nominal_curtailment_net(asset) / year_count) / 1000
                    row3["Generation (Energy Mix)"]["Curtailment"] = row3["Generation (Energy Mix)"].get("Curtailment", 0) + curtail_yr
                
                # Hydro Energy Capacity populates the isolated pie in Row 2
                row2["Hydro (Energy Cap)"][label] = row2["Hydro (Energy Cap)"].get(label, 0) + accessor.get_energy_capacity(asset)

            else:
                # Rechargeable Storage Processing
                label = process_storage_label(asset)
                
                row1["Rechargeable Storage (Power)"][label] = row1["Rechargeable Storage (Power)"].get(label, 0) + accessor.get_power_capacity(asset)
                row2["Rechargeable Storage (Energy Cap)"][label] = row2["Rechargeable Storage (Energy Cap)"].get(label, 0) + accessor.get_energy_capacity(asset)
                
                energy_yr = (accessor.get_discharge_net(asset) / year_count) / 1000
                row3["Rechargeable Storage (Discharge)"][label] = row3["Rechargeable Storage (Discharge)"].get(label, 0) + energy_yr
                
                if energy_type == "both":
                    losses_yr = (accessor.get_storage_losses(asset) / year_count) / 1000
                    spillage_yr = (accessor.get_storage_spillage_net(asset) / year_count) / 1000
                    
                    row3["Rechargeable Storage (Discharge)"]["Losses"] = row3["Rechargeable Storage (Discharge)"].get("Losses", 0) + losses_yr
                    row3["Rechargeable Storage (Discharge)"]["Spillage"] = row3["Rechargeable Storage (Discharge)"].get("Spillage", 0) + spillage_yr

                # Rechargeable Storage Sources (Row 2, Col 3)
                if raw_type in ["clphes", "olphes", "nphes", "bess2h", "bess4h"]:
                    prefix = label # Inherit the clean 'Legacy PHES', 'New PHES', 'Battery' label mapped above
                    
                    elec_label = f"{prefix} (Electrical)"
                    elec_val = (abs(accessor.get_charge_net(asset)) / year_count) / 1000
                    row2["Rechargeable Storage (Sources)"][elec_label] = row2["Rechargeable Storage (Sources)"].get(elec_label, 0) + elec_val
                    
                    if accessor.has_inflows(asset):
                        inflow_label = f"{prefix} (Inflows)"
                        inflow_val = ((np.sum(accessor.get_inflow_trace(asset)) * accessor.resolution) / year_count) / 1000
                        row2["Rechargeable Storage (Sources)"][inflow_label] = row2["Rechargeable Storage (Sources)"].get(inflow_label, 0) + inflow_val

        # 3. Transmission (Major Lines only)
        transmission_labels = {
            "ac_ohl_transmission": "AC OHL",
            "ac_ohl_mountain_transmission": "AC OHL (Mountain)",
            "dc_subsea_transmission": "DC Subsea",
            "dc_underground_transmission": "DC Underground"
        }
        
        for asset in accessor.get_assets("major_lines").values():
            raw_u_type = getattr(asset, 'unit_type', 'transmission')
            label = transmission_labels.get(raw_u_type, raw_u_type)
            
            row1["Transmission (Power)"][label] = row1["Transmission (Power)"].get(label, 0) + accessor.get_power_capacity(asset)
            
            energy_yr = (accessor.get_line_use_net(asset) / year_count) / 1000
            row3["Transmission (Flows)"][label] = row3["Transmission (Flows)"].get(label, 0) + energy_yr
            
            if energy_type == "both":
                losses_yr = (accessor.get_line_losses(asset) / year_count) / 1000
                row3["Transmission (Flows)"]["Losses"] = row3["Transmission (Flows)"].get("Losses", 0) + losses_yr

        # Plotting Setup
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 18))
        fig.subplots_adjust(hspace=0.4, wspace=0.6)

        def draw_row(data_dict, row_axes, units):
            # Calculate max total dynamically per unit type so GWh doesn't distort TWh scales
            unit_maxes = {}
            for mix, unit in zip(data_dict.values(), units):
                total = sum(v for v in mix.values() if v > 0)
                unit_maxes[unit] = max(unit_maxes.get(unit, 1), total)

            items = list(data_dict.items())
            
            for i, ax in enumerate(row_axes):
                if i >= len(items):
                    ax.axis('off')
                    continue
                    
                category, mix = items[i]
                unit = units[i]
                clean_mix = {k: v for k, v in mix.items() if v > 1e-6}
                total = sum(clean_mix.values())
                
                if total <= 0:
                    ax.axis('off')
                    continue
                
                radius = np.sqrt(total / unit_maxes[unit])
                labels = list(clean_mix.keys())
                values = list(clean_mix.values())
                colors = [get_chart_color(label) for label in labels]

                wedges, _ = ax.pie(
                    values, radius=radius, colors=colors, 
                    wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'}
                )
                
                ax.set_title(f"{category}\nTotal: {total:,.1f} {unit}", pad=10)
                ax.set_aspect('equal')
                
                ax.legend(
                    wedges, labels, 
                    loc="center left", 
                    bbox_to_anchor=(1, 0.5), 
                    frameon=False, 
                    fontsize=9
                )

        # Execute Grid Renders
        draw_row(row1, axes[0], ["GW", "GW", "GW"])
        draw_row(row2, axes[1], ["GWh", "GWh", "TWh/yr"])
        draw_row(row3, axes[2], ["TWh/yr", "TWh/yr", "TWh/yr"])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _dispatch_plot(
            self,
            data_type: str,
            atlas: bool = False,
            delta: bool = False,
            **kwargs
    ):
        if (atlas or delta) and not self.mhmga:
            raise ValueError(f"Cannot plot multiple solutions when mhmga is False. ({atlas=}, {delta=})")

        indices = kwargs.get("indices", [0, 1] if delta else [0])
        if delta and len(indices) < 2:
            raise ValueError(f"Delta plotting requires at least two indices in 'indices' kwarg. (Got: {indices})")

        chart_type = kwargs.get("chart_type", "bar" if delta else "pie")
        if delta and chart_type == "pie":
            raise ValueError("Delta plotting does not support 'pie' charts.")
        kwargs["chart_type"] = chart_type

        if atlas:
            grid = kwargs.get("grid", (2, 2))
            if grid[0] * grid[1] < len(indices):
                warnings.warn("Not enough axes to plot all graphs. Some graphs will not be rendered"
                              "Supply via kwarg 'grid'=(nrows, ncols)", UserWarning, 4)
            if grid[0] * grid[1] > len(indices):
                warnings.warn("More axes than graphs to plot. There will be blank graphs."
                              "Supply via kwarg 'grid'=(nrows, ncols)", UserWarning, 4)
            fig, axes = self._setup_map_axis(nrows=grid[0], ncols=grid[1])
            plot_targets = indices
        else:
            fig, axes = self._setup_map_axis(nrows=1, ncols=1)
            plot_targets = [indices[1]] if delta else [indices[0]]

        ref_idx = indices[0]
        ref_sol = self.noptima[ref_idx] if self.mhmga else self.solution

        #  Pre-calculate reference data and global scaling if deltas are required
        ref_data = None
        global_max_delta = 1
        if delta:
            ref_data = (
                self._aggregate_solution_energy_by_node(
                    ref_sol,
                    kwargs.get("curtailment", False),
                    kwargs.get("energy_mode", "generation")
                )
                if data_type == "energy"
                else self._aggregate_solution_capacity_by_node(ref_sol, kwargs.get("build", "all"))
            )

            for idx in indices[1:]:
                comp_sol = self.noptima[idx] if self.mhmga else self.solution
                comp_data = (
                    self._aggregate_solution_energy_by_node(
                        comp_sol,
                        kwargs.get("curtailment", False),
                        kwargs.get("energy_mode", "generation")
                    )
                    if data_type == "energy"
                    else self._aggregate_solution_capacity_by_node(comp_sol, kwargs.get("build", "all"))
                )
                delta_dict = self._calculate_delta_dict(ref_data, comp_data)
                max_d = max((max([abs(v) for v in d.values()], default=0) for d in delta_dict.values()), default=1)
                global_max_delta = max(global_max_delta, max_d)

        for i, ax in enumerate(axes):
            if i >= len(plot_targets):
                ax.set_visible(False)
                continue

            target_idx = plot_targets[i]
            target_sol = self.noptima[target_idx] if self.mhmga else self.solution

            # i==0 in atlas mode is the reference absolute plot.
            # Otherwise, if delta is True, it's a delta plot.
            is_delta_axis = delta and ((atlas and i > 0) or not atlas)

            if is_delta_axis:
                comp_data = (
                    self._aggregate_solution_energy_by_node(
                        target_sol,
                        kwargs.get("curtailment", False),
                        kwargs.get("energy_mode", "generation")
                    )
                    if data_type == "energy"
                    else self._aggregate_solution_capacity_by_node(target_sol, kwargs.get("build", "all"))
                )
                delta_dict = self._calculate_delta_dict(ref_data, comp_data)

                self._draw_delta_on_axis(ax, delta_dict, global_max_delta, **kwargs)
                ax.set_title(f"Delta: Alt {target_idx} - Alt {ref_idx}", fontsize=10)
            else:
                self._draw_solution_on_axis(ax, target_sol, data_type, **kwargs)
                ax.set_title(f"Absolute: Alt {target_idx}", fontsize=10)

        build_info = f" - Capacity: {kwargs.get('build', 'all')}" if data_type == "capacity" else ""
        fig.suptitle(f"{data_type.capitalize()} Mix{build_info}", fontsize=14)

        if kwargs.get("save_path"):
            plt.savefig(kwargs.get("save_path"), dpi=300, bbox_inches='tight')

        return axes if atlas else axes[0]

    def _draw_delta_on_axis(self, ax, delta_dict, global_max_delta, **kwargs):
        """Worker method to draw delta bars and legend on a specific axis."""
        chart_scale = kwargs.get("chart_scale", 1.0)

        for node_name, mix in delta_dict.items():
            if node_name not in self.centroids: continue

            filtered_mix = {k: v for k, v in mix.items() if abs(v) > kwargs.get("threshold", 1e-3)}
            if not filtered_mix: continue

            x_coord, y_coord = self.centroids[node_name]
            self._draw_bars(
                ax, filtered_mix, x_coord, y_coord, global_max_delta,
                is_delta=True, chart_scale=chart_scale
            )

        if kwargs.get("legend", True):
            self._add_legend(ax, delta_dict)

    def _calculate_delta_dict(self, dict_a, dict_b):
        """Helper to subtract two node-tech dictionaries."""
        delta = {}
        all_nodes = set(dict_a.keys()) | set(dict_b.keys())
        for n in all_nodes:
            delta[n] = {}
            node_a = dict_a.get(n, {})
            node_b = dict_b.get(n, {})
            all_techs = set(node_a.keys()) | set(node_b.keys())
            for t in all_techs:
                delta[n][t] = node_b.get(t, 0.0) - node_a.get(t, 0.0)
        return delta

    def _get_delta_color(self, tech, value):
        """
        Placeholder - may Returns a modified version of the tech color:
        Slightly brighter/redder for positive, darker/bluer for negative?
        Currently no difference
        """
        base_color = self._get_color(tech)
        if value >= 0:
            return base_color
        else:
            return base_color

    def _draw_solution_on_axis(self, ax, solution, data_type, **kwargs):
        """
        Unified worker that switches between _draw_pie and _draw_bars.
        """
        chart_type = kwargs.get("chart_type", "pie")
        chart_scale = kwargs.get("chart_scale", 1.0)

        # Data aggregation
        if data_type == "energy":
            node_data = self._aggregate_solution_energy_by_node(
                solution,
                kwargs.get("curtailment", False),
                kwargs.get("energy_mode", "generation")
            )
            flow_type = "energy"
        else:
            node_data = self._aggregate_solution_capacity_by_node(solution, kwargs.get("build", "all"))
            flow_type = "capacity"

        # Scaling logic
        max_total = kwargs.get("max_scale", max((sum(d.values()) for d in node_data.values()), default=1))

        for node_name, mix in node_data.items():
            if node_name not in self.centroids:
                raise RuntimeError(f"Node {node_name} not found in map centroids.")

            # Filter zero-valued entries for bars to keep the x-axis dynamic
            filtered_mix = {k: v for k, v in mix.items() if abs(v) > kwargs.get("threshold", 1e-5)}
            if not filtered_mix:
                continue

            x, y = self.centroids[node_name]

            if chart_type == "bar":
                # For non-delta plots, y_limit is the max_total of the node
                # or a global max for consistency.
                self._draw_bars(ax, filtered_mix, x, y, max_total, chart_scale=chart_scale)
            else:
                radius = 100_000 * chart_scale * np.sqrt(sum(mix.values()) / max_total)
                keys = sorted(mix.keys())
                values = [mix[k] for k in keys]
                colors = [self._get_color(k) for k in keys]
                self._draw_pie(ax, values, x, y, radius, colors)

        self._draw_solution_transmission(solution, ax, flow_type=flow_type, build=kwargs.get("build"))

        if kwargs.get("legend", True):
            self._add_legend(ax, node_data)

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
        [evaluate(sol) for sol in self.noptima]
        # first noptimum is optimum
        self.solution = self.noptima[0]

    def _load_map_data(self, filepath: str) -> gpd.GeoDataFrame:
        self.map_data = gpd.read_file(filepath)
        self.map_data = self.map_data.to_crs(epsg=3035)
        # Calculate centroids for node placement
        # Assumes the 'name' column in GeoJSON matches node.name in the Solution
        self.centroids = self._calculate_centroids()

    def _calculate_centroids(self) -> Dict[str, Tuple[float, float]]:
        """
        Matches network nodes to map geometries and calculates centroids.
        """
        centroids = {}
        # Calculate centroids directly on the projected map
        map_cents = self.map_data.geometry.centroid
        for node in self.solution.network.nodes.values():
            # Match ISO3 or Name
            # Ensure your GeoJSON column name matches (e.g. 'ISO3', 'id', 'name')
            match_indices = self.map_data.index[
                self.map_data["ISO3"].str.lower() == node.name.lower()
            ]

            if not match_indices.empty:
                idx = match_indices[0]
                pt = map_cents[idx]
                centroids[node.name] = (pt.x, pt.y)
            else:
                print(f"Warning: No map geometry found for node {node.name}.")
                centroids[node.name] = (0, 0)
        return centroids

    def _draw_pie(self, ax, dist, xpos, ypos, radius, colors):
        """
        Draws a pie chart using Wedge patches which respect data coordinates.
        Replaces ax.scatter to fix the 'exploding wedge' distortion.
        """
        if sum(dist) == 0:
            return
        # Normalize distribution for slice angles
        data = np.array(dist)
        data = data / data.sum()
        start_angle = 90

        for i, val in enumerate(data):
            if val == 0:
                continue

            deg = val * 360
            end_angle = start_angle + deg

            w = Wedge(
                (xpos, ypos),
                radius,
                start_angle,
                end_angle,
                facecolor=colors[i],
                zorder=100,
                edgecolor='none'
            )
            ax.add_patch(w)
            start_angle = end_angle

        # Outline
        outline = Wedge(
            (xpos, ypos), radius, 0, 360,
            facecolor="none", edgecolor="black", linewidth=0.5, zorder=101
        )
        ax.add_patch(outline)

    def _draw_bars(self, ax, mix, xpos, ypos, y_limit, is_delta=False, chart_scale=1.0):
        # Dimensions in map units (EPSG:3035 meters)
        width_m = 250_000 * chart_scale
        height_m = 250_000 * chart_scale

        keys = sorted(mix.keys())
        values = [mix[k] for k in keys]
        colors = [self._get_color(k) for k in keys]

        num_bars = len(keys)
        bar_width = width_m / (num_bars + 1)
        x_start = xpos - (width_m / 2)

        # Baseline: For absolute plots, baseline is bottom. For delta, it's center.
        baseline_y = ypos if is_delta else ypos - (height_m / 2)

        # Optional: draw a faint background box for the chart area
        # ax.add_patch(plt.Rectangle((x_start, ypos - height_m/2), width_m, height_m, ...))

        for i, val in enumerate(values):
            # Scale height. If delta, height can be negative.
            # If absolute, val/y_limit scales 0 to 1.
            h_ratio = (val / y_limit)
            h = h_ratio * (height_m / (2 if is_delta else 1))

            bar_x = x_start + (i + 0.5) * bar_width

            rect = plt.Rectangle(
                (bar_x - bar_width/2, baseline_y),
                bar_width * 0.8,
                h,
                facecolor=colors[i],
                edgecolor='black',
                linewidth=0.5,
                zorder=110
            )
            ax.add_patch(rect)

        # Draw a horizontal line at the baseline
        ax.plot([x_start, x_start + width_m], [baseline_y, baseline_y], color='black', lw=0.8, zorder=111)

    def _setup_map_axis(self, nrows=1, ncols=1, figsize=None):
        """Handles single and multi-axis creation with background maps."""
        if figsize is None:
            figsize = (8 * ncols, 8 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, squeeze=False)
        axes_flat = axes.flatten()

        for ax in axes_flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            ax.set_aspect("equal")
            self.map_data.plot(ax=ax, edgecolor="black", facecolor="#eeeeee", zorder=1)

            # Set Limits
            minx, miny, maxx, maxy = self.map_data.total_bounds
            margin = 200_000  # 200 km margin
            ax.set_xlim(minx - margin, maxx + margin)
            ax.set_ylim(miny - margin, maxy + margin)

        return fig, axes_flat

    def _draw_solution_transmission(self, solution, ax, flow_type="capacity", build=None):
        """
        Draws transmission lines with dynamic width scaling.
        Uses a two-pass approach:
        1. Scan all lines to find the maximum value (capacity or flow).
        2. Draw lines scaled relative to that maximum.
        """
        # Configuration for visual scaling
        MAX_LINE_WIDTH = 5.0  # The thickest line will be this wide (in points)
        MIN_LINE_WIDTH = 0.3  # The thinnest visible line

        lines_data = []
        max_val = 0.0
        accessor = Accessor(solution, "GW")

        for line in solution.network.major_lines.values():
            n_start = line.node_start.name
            n_end = line.node_end.name
            if n_start not in self.centroids:
                raise RuntimeError(f"Line node {n_start} not found in map centroids.")
            if n_end not in self.centroids:
                raise RuntimeError(f"Line node {n_end} not found in map centroids.")

            if flow_type == "capacity":
                match str(build).lower():
                    case "none" | "all":
                        val = accessor.get_power_capacity(line)
                    case "new_build":
                        val = accessor.get_new_build_capacity(line, "power")
                    case "existing" | "initial":
                        val = accessor.get_existing_capacity(line, "power")
            elif flow_type == "energy":
                val = accessor.get_line_use_net(line)
            else:
                raise ValueError(f"Invalid 'flow_type'. Expected \"capacity\" or \"energy\". Got {flow_type}")
            max_val = max(max_val, val)

            p1 = self.centroids[n_start]
            p2 = self.centroids[n_end]
            lines_data.append((p1, p2, val))

        if max_val == 0:
            return

        for p1, p2, val in lines_data:
            # Calculate dynamic width: (Current / Max) * Target_Max_Width
            scaled_width = (val / max_val) * MAX_LINE_WIDTH
            # Enforce minimum visibility
            final_width = max(scaled_width, MIN_LINE_WIDTH)
            color = "red" if flow_type == "capacity" else "blue"
            alpha = 0.7 if flow_type == "capacity" else 0.5
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=final_width,
                zorder=50,
                alpha=alpha,
            )

    def _aggregate_solution_energy_by_node(self, solution, curtailment=False, energy_mode="generation"):
        """
        Scans fleet generators, storages to sum energy by node and tech.
        """
        data = {}
        accessor = Accessor(solution, "GW")

        for asset_class in ("generators", "storages"):
            for asset in accessor.get_assets(asset_class).values():
                n = asset.node.name
                tech = self.scenario.identify_tech(asset.name)

                if n not in data:
                    data[n] = {}
                if tech not in data[n]:
                    data[n][tech] = 0.0
                if curtailment:
                    data[n][tech] += accessor.get_post_curtailment_energy_net(asset)
                else:
                    data[n][tech] += accessor.get_discharge_net(asset)
        # --- Debug print ---
        # for n in data:
        #     for tech in data[n]:
        #         print(f"Node: {n}, Tech: {tech}, Energy: {data[n][tech]} GWh")
        return data

    def _aggregate_solution_capacity_by_node(self, solution, build=None):
        """
        Scans fleet to sum capacity (GW) by node and tech.
        """
        data = {}
        accessor = Accessor(solution, "GW")

        for asset_class in ("generators", "storages"):
            for asset in accessor.get_assets(asset_class).values():
                n = asset.node.name
                tech = self.scenario.identify_tech(asset.name)

                if n not in data:
                    data[n] = {}
                if tech not in data[n]:
                    data[n][tech] = 0.0
                match str(build).lower():
                    case "none" | "all":
                        data[n][tech] += accessor.get_power_capacity(asset)
                    case "new_build":
                        data[n][tech] += accessor.get_new_build_capacity(asset, "power")
                    case "existing" | "initial":
                        data[n][tech] += accessor.get_existing_capacity(asset, "power")
        # --- Debug print ---
        # for n in data:
        #     for tech in data[n]:
        #         print(f"Node: {n}, Tech: {tech}, Capacity: {data[n][tech]} GW")
        return data

    def _init_colors(self):
        self.colors = sns.color_palette("Paired")
        self.tech_colors = {
            "Utility Solar": self.colors[7],
            "Rooftop Solar": self.colors[6],
            "Onshore Wind": self.colors[8],
            "Offshore Wind": self.colors[9],
            "Hydro": self.colors[1],
            "Biomass": self.colors[3],
            "Biogas": self.colors[2],
            "Fossil Gas": (0.5, 0.5, 0.5),
            "Nuclear": self.colors[4],
            "Battery": self.colors[10],
            "PHES": self.colors[0],
            "Geothermal": self.colors[5],
            "Coal": (0.15, 0.15, 0.15),
        }

    def _get_color(self, tech):
        return self.tech_colors.get(tech, (0.5, 0.5, 0.5))

    def _add_legend(self, ax, data_dict):
        """Dynamically creates legend based on present technologies."""
        present_techs = set()
        for mix in data_dict.values():
            present_techs.update(mix.keys())

        handles = []
        labels = []
        for tech in sorted(present_techs):
            handles.append(plt.Rectangle((0, 0), 1, 1, color=self._get_color(tech)))
            labels.append(tech)

        ax.legend(
            handles,
            labels,
            loc="upper right",
            title="Technology",
            frameon=False,
        )
