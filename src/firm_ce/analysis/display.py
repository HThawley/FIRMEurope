import geopandas as gpd
import matplotlib.pyplot as plt
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
        self.map_data = self.map_data.to_crs(epsg=4326)  # Ensure WGS84

        # Calculate centroids for node placement
        # Assumes the 'name' column in GeoJSON matches node.name in the Solution
        self.centroids = self._calculate_centroids()

        self.colors = sns.color_palette("deep")
        self.tech_colors = {
            "Solar": self.colors[1],
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

        # Create a temporary projected copy (EPSG:3035) for accurate math
        map_projected = self.map_data.to_crs(epsg=3035)
        # 2. Calculate centroids in the projected CRS
        cents_projected = map_projected.geometry.centroid
        # 3. Convert centroids back to WGS84 (EPSG:4326) to match the map plot axes
        cents_wgs84 = cents_projected.to_crs(epsg=4326)

        # Iterate through nodes in the solution network
        for node in self.solution.network.nodes.values():
            # Find corresponding geometry in GeoJSON
            match_indices = self.map_data.index[self.map_data["ISO3"].str.lower() == node.name.lower()]

            if not match_indices.empty:
                idx = match_indices[0]
                pt = cents_wgs84[idx]
                centroids[node.name] = (pt.x, pt.y)
            else:
                print(f"Warning: No map geometry found for node {node.name}. Using (0,0).")
                centroids[node.name] = (0, 0)
        return centroids

    def _draw_pie(self, ax, dist, xpos, ypos, size, colors):
        """
        Helper to draw a pie chart at a specific location
        """
        if sum(dist) == 0:
            return

        # Normalize distance for pie slices
        cumsum = np.cumsum(dist)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()

        for i, r in enumerate(zip(pie[:-1], pie[1:])):
            r1, r2 = r
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            ax.scatter(
                [xpos],
                [ypos],
                marker=xy,
                s=size,
                zorder=100,
                facecolor=colors[i],
                edgecolor="none",
            )

        # Draw outline
        ax.scatter(
            [xpos],
            [ypos],
            marker="o",
            s=size,
            zorder=101,
            facecolor=[1, 1, 1, 0],
            edgecolor=[0, 0, 0, 1],
            linewidth=0.5,
        )

    def plot_energy_mix(self, save_path=None):
        """
        Plots total energy generation mix (GWh) per node.
        """
        fig, ax = self._setup_map_axis()

        # Aggregate Generation Data
        node_mix = self._aggregate_energy_by_node()

        # Scale factor for pie charts
        max_energy = max(sum(mix.values()) for mix in node_mix.values()) if node_mix else 1
        scale_factor = 2000  # Adjust visualization size

        for node_name, mix in node_mix.items():
            if node_name not in self.centroids:
                continue

            x, y = self.centroids[node_name]
            total = sum(mix.values())
            if total == 0:
                continue

            # Prepare data for pie chart
            keys = sorted(mix.keys())
            values = [mix[k] for k in keys]
            colors = [self._get_color(k) for k in keys]

            # Size relative to total energy, log scale or sqrt often better for maps
            size = (total / max_energy) * scale_factor + 50

            self._draw_pie(ax, values, x, y, size, colors)

        # Draw Transmission Flows (Net Flow)
        self._draw_transmission(ax, flow_type="energy")

        self._add_legend(ax, node_mix)
        ax.set_title("Annual Energy Mix (GWh)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_power_capacity(self, save_path=None):
        """
        Plots installed power capacity (GW) per node.
        """
        fig, ax = self._setup_map_axis()

        # Aggregate Capacity Data
        node_cap = self._aggregate_capacity_by_node()

        max_cap = max(sum(cap.values()) for cap in node_cap.values()) if node_cap else 1
        scale_factor = 2500

        for node_name, cap in node_cap.items():
            if node_name not in self.centroids:
                continue

            x, y = self.centroids[node_name]
            total = sum(cap.values())
            if total == 0:
                continue

            keys = sorted(cap.keys())
            values = [cap[k] for k in keys]
            colors = [self._get_color(k) for k in keys]

            size = (total / max_cap) * scale_factor + 50

            self._draw_pie(ax, values, x, y, size, colors)

        # Draw Transmission Capacities
        self._draw_transmission(ax, flow_type="capacity")

        self._add_legend(ax, node_cap)
        ax.set_title("Installed Power Capacity (GW)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def _setup_map_axis(self):
        fig, ax = plt.subplots(1, figsize=(10, 10), dpi=150)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Plot background map
        self.map_data.plot(ax=ax, edgecolor="black", facecolor="lightgrey", zorder=1)

        # Set European Limits (-25W to 45E, 34N to 72N)
        ax.set_xlim(-25, 45)
        ax.set_ylim(34, 72)

        return fig, ax

    def _draw_transmission(self, ax, flow_type="capacity"):
        """
        Draws lines between nodes.
        flow_type='capacity': Line width represents installed capacity.
        flow_type='energy': Arrows represent net energy flow.
        """
        # Iterate over major lines in the network
        # Note: Solution.network.major_lines is a typed dict
        for line in self.solution.network.major_lines.values():
            n_start = line.node_start.name
            n_end = line.node_end.name

            if n_start not in self.centroids or n_end not in self.centroids:
                continue

            p1 = np.array(self.centroids[n_start])
            p2 = np.array(self.centroids[n_end])

            if flow_type == "capacity":
                # Just draw a line with thickness proportional to capacity
                width = line.capacity / 2000.0  # Scaling factor
                if width > 0.5:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=width, zorder=50, alpha=0.7)

            elif flow_type == "energy":
                # Net flow calculation would require accessing flow time series
                # Assuming 'line.flow_sum' or similar exists, or calculating from time series
                # As a placeholder, we draw capacity lines but distinct style
                width = line.capacity / 2000.0
                if width > 0.5:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue", linewidth=width, zorder=50, alpha=0.5)

    def _aggregate_energy_by_node(self):
        """
        Scans fleet generators, reservoirs, storages to sum energy by node and tech.
        """
        data = {}

        for asset_class in ("generators", "reservoirs", "storages"):
            for asset in self.accessor.get_assets(asset_class).values():
                n = asset.node.name
                tech = self._identify_tech(asset.unit_type)

                if n not in data:
                    data[n] = {}
                if tech not in data[n]:
                    data[n][tech] = 0.0

                data[n][tech] += self.accessor.get_power_net(asset)

        return data

    def _aggregate_capacity_by_node(self):
        """
        Scans fleet to sum capacity (GW) by node and tech.
        """
        data = {}

        for asset_class in ("generators", "reservoirs", "storages"):
            for asset in self.accessor.get_assets(asset_class).values():
                n = asset.node.name
                tech = self._identify_tech(asset.unit_type)

                if n not in data:
                    data[n] = {}
                if tech not in data[n]:
                    data[n][tech] = 0.0

                data[n][tech] += self.accessor.get_power_capacity(asset)

        return data

    def _identify_tech(self, name):
        """Maps asset attributes to simplified plotting categories."""
        name_lower = name.lower()

        if "solar" in name_lower or "pv" in name_lower:
            return "Solar"
        if "onshore" in name_lower or "onsw" in name_lower:
            return "Onshore Wind"
        if "offshore" in name_lower or "offw" in name_lower:
            return "Offshore Wind"
        if "hydro" in name_lower:
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

        ax.legend(handles, labels, loc="lower left", title="Technology", frameon=False)
