import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import seaborn as sns
from typing import Dict, Tuple

# Import necessary types for type hinting if available in the environment
from firm_ce.optimisation.single_time import Solution
from firm_ce.analysis.accessor import Accessor


class Display:
    """
    Visualizes optimization results for the European energy system based on a
    Solution objects.
    """

    def __init__(
        self,
        solution: Solution,
    ):
        """
        Initialize the plotter with a Solution object and map data.
        """
        self.solution = solution
        self.accessor = Accessor(solution, "GW")
        self.map_data = gpd.read_file("./inputs/map/europe.geojson")

        # Filter for relevant bounds or specific countries if needed.
        # Here we assume the geojson contains the relevant EU states.
        self.map_data = self.map_data.to_crs(epsg=3035)

        # Calculate centroids for node placement
        # Assumes the 'name' column in GeoJSON matches node.name in the Solution
        self.centroids = self._calculate_centroids()

        self.colors = sns.color_palette("deep")
        self.tech_colors = {
            "Utility Solar": self.colors[1],
            "Rooftop Solar": self.colors[8],
            "Onshore Wind": self.colors[5],
            "Offshore Wind": self.colors[4],
            "Hydro": self.colors[0],
            "Biomass": self.colors[3],
            "Gas": self.colors[7],
            "Nuclear": self.colors[6],
            "Battery": self.colors[2],
            "PHES": self.colors[9],
            "Coal": (0.2, 0.2, 0.2),
        }

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

    def plot_energy_mix(self, curtailment=False, save_path=None):
        """
        Plots total energy generation mix (GWh) per node.
        """
        fig, ax = self._setup_map_axis()
        node_mix = self._aggregate_energy_by_node(curtailment)
        max_total = max((sum(mix.values()) for mix in node_mix.values()), default=0)

        MAX_RADIUS_METERS = 100_000

        for node_name, mix in node_mix.items():
            if node_name not in self.centroids:
                continue

            total = sum(mix.values())
            if total == 0:
                continue

            # Calculate Radius: Proportional to sqrt(Area) to preserve perception
            # R_node = R_max * sqrt(Val_node / Val_max)
            radius = MAX_RADIUS_METERS * np.sqrt(total / max_total)

            x, y = self.centroids[node_name]

            # Prepare data for pie chart
            keys = sorted(mix.keys())
            values = [mix[k] for k in keys]
            colors = [self._get_color(k) for k in keys]
            self._draw_pie(ax, values, x, y, radius, colors)

        self._draw_transmission(ax, flow_type="energy")
        self._add_legend(ax, node_mix)

        if curtailment:
            ax.set_title("Annual Energy Mix post-curtailment (GWh)")
        else:
            ax.set_title("Annual Energy Mix pre-curtailment (GWh)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_power_capacity(self, build=None, save_path=None):
        """
        Plots installed power capacity (GW) per node.
        """
        fig, ax = self._setup_map_axis()
        node_cap = self._aggregate_capacity_by_node(build)
        max_total = max((sum(cap.values()) for cap in node_cap.values()), default=0)

        MAX_RADIUS_METERS = 100_000

        for node_name, cap in node_cap.items():
            if node_name not in self.centroids:
                continue

            total = sum(cap.values())
            if total == 0:
                continue

            radius = MAX_RADIUS_METERS * np.sqrt(total / max_total)
            x, y = self.centroids[node_name]

            keys = sorted(cap.keys())
            values = [cap[k] for k in keys]
            colors = [self._get_color(k) for k in keys]

            self._draw_pie(ax, values, x, y, radius, colors)

        self._draw_transmission(ax, flow_type="capacity", build=build)
        self._add_legend(ax, node_cap)
        match str(build).lower():
            case "none" | "all":
                ax.set_title("Installed Power Capacity (GW)")
            case "new_build":
                ax.set_title("Installed Power Capacity (GW) (New build only)")
            case "existing" | "initial":
                ax.set_title("Installed Power Capacity (GW) (Existing only)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def _setup_map_axis(self):
        fig, ax = plt.subplots(1, figsize=(10, 10), dpi=150)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        ax.set_aspect("equal")

        # Plot background map
        self.map_data.plot(ax=ax, edgecolor="black", facecolor="lightgrey", zorder=1)

        # Set Limits dynamically based on the map bounds
        minx, miny, maxx, maxy = self.map_data.total_bounds
        margin = 200_000  # 200km margin
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)
        return fig, ax

    def _draw_transmission(self, ax, flow_type="capacity", build=None):
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

        for line in self.solution.network.major_lines.values():
            n_start = line.node_start.name
            n_end = line.node_end.name
            if n_start not in self.centroids or n_end not in self.centroids:
                continue

            if flow_type == "capacity":
                match str(build).lower():
                    case "none" | "all":
                        val = self.accessor.get_power_capacity(line)
                    case "new_build":
                        val = self.accessor.get_new_build_capacity(line, "power")
                    case "existing" | "initial":
                        val = self.accessor.get_existing_capacity(line, "power")
            elif flow_type == "energy":
                val = self.accessor.get_line_use_net(line)
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

    def _aggregate_energy_by_node(self, curtailment=False):
        """
        Scans fleet generators, reservoirs, storages to sum energy by node and tech.
        """
        data = {}

        for asset_class in ("generators", "reservoirs", "storages"):
            for asset in self.accessor.get_assets(asset_class).values():
                n = asset.node.name
                tech = self._identify_tech(asset.name)

                if n not in data:
                    data[n] = {}
                if tech not in data[n]:
                    data[n][tech] = 0.0
                if curtailment:
                    data[n][tech] += self.accessor.get_post_curtailment_energy_net(asset)
                else:
                    data[n][tech] += self.accessor.get_discharge_net(asset)

        return data

    def _aggregate_capacity_by_node(self, build=None):
        """
        Scans fleet to sum capacity (GW) by node and tech.
        """
        data = {}

        for asset_class in ("generators", "reservoirs", "storages"):
            for asset in self.accessor.get_assets(asset_class).values():
                n = asset.node.name
                tech = self._identify_tech(asset.name)

                if n not in data:
                    data[n] = {}
                if tech not in data[n]:
                    data[n][tech] = 0.0
                match str(build).lower():
                    case "none" | "all":
                        data[n][tech] += self.accessor.get_power_capacity(asset)
                    case "new_build":
                        data[n][tech] += self.accessor.get_new_build_capacity(asset, "power")
                    case "existing" | "initial":
                        data[n][tech] += self.accessor.get_existing_capacity(asset, "power")
        return data

    def _identify_tech(self, name):
        """Maps asset attributes to simplified plotting categories."""
        name_lower = name.lower()
        if "solar" in name_lower or "pv" in name_lower or "roof" in name_lower:
            return "Utility Solar"
        if "roof" in name_lower:
            return "Rooftop Solar"
        if "onshore" in name_lower or "onsw" in name_lower:
            return "Onshore Wind"
        if "offshore" in name_lower or "offw" in name_lower:
            return "Offshore Wind"
        if "hydro" in name_lower or "ror" in name_lower:
            return "Hydro"
        if "nuke" in name_lower or "nuclear" in name_lower:
            return "Nuclear"
        if "gas" in name_lower or "ccgt" in name_lower or "ocgt" in name_lower:
            return "Gas"
        if "flexible" in name_lower:
            return "Gas"  # TODO: update flexible classification
        if "bio" in name_lower:
            return "Biomass"
        if "coal" in name_lower:
            return "Coal"
        if "bess" in name_lower:
            return "Battery"
        if "phes" in name_lower:
            return "PHES"
        return "Other"

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
