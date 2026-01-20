import numpy as np
from typing import Any
from numpy.typing import NDArray

from firm_ce.common.helpers import safe_divide_array


asset_class_to_display = {
    "generators": "Generator",
    "reservoirs": "Reservoir",
    "storages": "Storage",
    "major_lines": "Major Line",
    "minor_lines": "Minor Line",
    "nodes": "Node",
}


# --- Data Retrievers ---
class Accessor:
    def __init__(self, solution, units=1.0):
        self.solution = solution
        self.resolution = solution.static.resolution
        match units.lower():
            case "mw" | "mwh":
                self.factor = 1000.0  # GW to MW
            case "gw" | "gwh":
                self.factor = 1.0
            case _:
                if isinstance(units, (int, float)):
                    self.factor = float(units)
                else:
                    raise ValueError(f"Unknown units for capacity retrieval: {units}")

    # --- Asset Type Checkers ---
    @staticmethod
    def is_any(asset: Any) -> bool:
        return True

    @staticmethod
    def is_flexible(asset: Any) -> bool:
        return asset.unit_type == "flexible"

    @staticmethod
    def is_not_flexible(asset: Any) -> bool:
        return asset.unit_type != "flexible"

    @staticmethod
    def is_solar(asset: Any) -> bool:
        return asset.unit_type == "solar"

    @staticmethod
    def is_ror(asset: Any) -> bool:
        return asset.unit_type == "ror"

    @staticmethod
    def is_wind(asset: Any) -> bool:
        return asset.unit_type == "wind"

    @staticmethod
    def is_baseload(asset: Any) -> bool:
        return asset.unit_type == "baseload"

    @staticmethod
    def is_generator(asset: Any) -> bool:
        return asset.object_class == "generator"

    @staticmethod
    def is_storage(asset: Any) -> bool:
        return asset.object_class == "storage"

    @staticmethod
    def is_reservoir(asset: Any) -> bool:
        return asset.object_class == "reservoir"

    @staticmethod
    def is_line(asset: Any) -> bool:
        return asset.object_class == "line"

    @staticmethod
    def is_node(asset: Any) -> bool:
        return asset.object_class == "node"

    @staticmethod
    def get_zero(*args) -> float:
        return 0.0

    # -- Objects --
    @staticmethod
    def get_assets_from_solution(solution, asset_class: str) -> dict[str, Any]:
        """Static method version of get_assets."""
        match asset_class:
            case "generators" | "reservoirs" | "storages":
                return getattr(solution.fleet, asset_class)
            case "major_lines" | "minor_lines" | "nodes":
                return getattr(solution.network, asset_class)
            case _:
                raise ValueError(f"Unknown asset class for asset retrieval: {asset_class}")

    def get_assets(self, asset_class: str) -> dict[str, Any]:
        """Returns the assets for a given asset class."""
        return self.get_assets_from_solution(self.solution, asset_class)

    @staticmethod
    def get_display_name(asset_class: str) -> str:
        return asset_class_to_display.get(asset_class, asset_class)

    # -- Capacity --
    @staticmethod
    def get_power_capacity(asset: Any) -> float:
        """Safe retrieval of installed capacity in GW."""
        match asset.object_class:
            case "generator" | "line":
                return asset.capacity
            case "storage" | "reservoir":
                return asset.power_capacity
            case _:
                raise ValueError(f"Unknown asset type for capacity retrieval: {asset.name} ({asset.object_class})")

    @staticmethod
    def get_energy_capacity(asset: Any) -> float:
        """Safe retrieval of installed capacity in GW."""
        match asset.object_class:
            case "generator" | "line":
                raise ValueError(f"Asset: {asset.name} ({asset.object_class}) does not have energy capacity.")
            case "storage" | "reservoir":
                return asset.energy_capacity
            case _:
                raise ValueError(f"Unknown asset type for capacity retrieval: {asset.name} ({asset.object_class})")

    def get_capacity(self, asset: Any, attribute: str) -> float:
        """Safe retrieval of installed capacity in GW."""
        match attribute.lower():
            case "power":
                return self.get_power_capacity(asset)
            case "energy":
                return self.get_energy_capacity(asset)
            case _:
                raise ValueError(f"Unknown attribute for capacity retrieval: {attribute}")

    @staticmethod
    def get_build_limits_power(asset: Any) -> tuple[float, float, float]:
        """Returns the build limits for power capacity (new_build, min_build, max_build)."""
        match asset.object_class:
            case "generator" | "line":
                return asset.new_build, asset.min_build, asset.max_build
            case "storage" | "reservoir":
                return asset.new_build_p, asset.min_build_p, asset.max_build_p
            case _:
                raise ValueError(f"Unknown asset type for build limits (power) retrieval: {asset.name} ({asset.object_class})")

    @staticmethod
    def get_build_limits_energy(asset: Any) -> tuple[float, float, float]:
        """Returns the build limits for energy capacity (new_build, min_build, max_build)."""
        match asset.object_class:
            case "storage" | "reservoir":
                return asset.new_build_e, asset.min_build_e, asset.max_build_e
            case _:
                raise ValueError(f"Unknown asset type for build limits (energy) retrieval: {asset.name} ({asset.object_class})")

    def get_build_limits(self, asset: Any, attribute: str) -> tuple[float, float, float]:
        """Returns the build limits for capacity (new_build, min_build, max_build)."""
        match attribute.lower():
            case "power":
                return self.get_build_limits_power(asset)
            case "energy":
                return self.get_build_limits_energy(asset)
            case _:
                raise ValueError(f"Unknown attribute for build limits retrieval: {attribute}")

    # -- Traces --
    def get_power_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns the power output time series (MW) for an object.
        """
        match asset.object_class:
            case "generator":
                if self.is_flexible(asset):
                    if not hasattr(asset, "dispatch_power"):
                        raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have 'dispatch_power' attribute.")
                    return asset.dispatch_power * self.factor
                elif self.is_inflexible(asset):
                    if not hasattr(asset, "data"):
                        raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have 'data' attribute.")
                    if not hasattr(asset, "capacity"):
                        raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have 'capacity' attribute.")
                    return asset.data * asset.capacity * self.factor
            case "storage" | "reservoir":
                # Positive = Generation, Negative = Load
                if not hasattr(asset, "dispatch_power"):
                    raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have 'dispatch_power' attribute.")
                return asset.dispatch_power * self.factor
            case "node":
                # returns demand
                return asset.data * self.factor
            case "line":
                return asset.flow * self.factor
            case _:
                raise ValueError(f"Unknown asset type for power retrieval: {asset.name} ({asset.unit_type})")

    def get_discharge_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns only the POSITIVE generation component (clipping pumping/charging).
        Useful for 'Energy Mix' charts where load is treated separately.
        Units: MW >= 0
        """
        trace = self.get_power(asset)
        return np.maximum(0, trace)

    def get_charge_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns only the NEGATIVE generation component (clipping pumping/charging).
        Useful for 'Energy Mix' charts where load is treated separately.
        Units: MW <= 0
        """
        trace = self.get_power(asset)
        return np.minimum(0, trace)

    def get_spillage_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns the spillage power time series (MW) for nodes.
        """
        if not self.is_node(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Node and therefore has no spillage.")
        return asset.spillage * self.factor

    def get_deficit_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns the deficit power time series (MW) for nodes.
        """
        if not self.is_node(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Node and therefore has no deficit.")
        return asset.deficits * self.factor

    def get_transmission_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns the transmission power time series (MW) for lines.
        Positive values indicate flow from initial_node to terminal_node.
        Negative values indicate flow from terminal_node to initial_node.
        """
        if not self.is_line(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Line and therefore has no transmission power.")
        return asset.flow * self.factor

    def get_inflow_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Return the inflow energy time series (MWh) for reservoirs.
        """
        if not hasattr(asset, "data"):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have 'data' attribute.")
        if not asset.data_status:
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) has data_status=False, data not loaded.")
        return asset.data * self.factor

    def get_storage_level_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns the storage level time series (MWh) for storage units and reservoirs.
        """
        if not (self.is_storage(asset) or self.is_reservoir(asset)):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not Storage/Reservoir and has no 'stored_energy' attr.")
        return asset.stored_energy * self.factor

    def get_remaining_energy_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        Returns the remaining energy (MWh) for flexible generators with fuel constraints.
        """
        if not self.is_generator(asset) or not self.is_flexible(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a flexible Generator "
                             "and has no 'remaining_energy' attr.")
        return asset.remaining_energy * self.factor

    def get_nodal_generation_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        get time series of total supply at the node of an asset (including the asset's contribution)
        gets generation only. charging is not included, only discharging.
        Used for calculating curtailment at a node.
        """
        node_generation = sum(
            (self.get_power(_asset) for _asset in self.solution.fleet.generators.values() if _asset.node.id == asset.node.id)
        )
        node_generation += sum(
            (self.get_power(_asset) for _asset in self.solution.fleet.reservoirs.values() if _asset.node.id == asset.node.id)
        )

        # in principle, when spillage occurs this is zero - but calculated for robustness
        node_generation += sum(
            (self.get_discharge_power(_asset) for _asset in self.solution.fleet.storages.values()
             if _asset.node.id == asset.node.id)
        )
        return node_generation * self.factor

    def get_nominal_curtailment_trace(self, asset: Any) -> NDArray[np.float64]:
        """
        get time series of curtailment for a given asset (calculated by apportioning spillage)
        nominal curtailment is all spillage apportioned according to the asset's share of generation at the node
        """
        nodal_generation = self.get_nodal_generation_power(asset)
        asset_generation = self.get_power(asset)
        spillage = self.get_spillage_power(asset.node)

        curtailment = spillage * safe_divide_array(asset_generation, nodal_generation)
        return curtailment

    # -- Aggregate Energy --
    def get_energy_net(self, asset: Any) -> float:
        """
        Returns the total dispatched energy (MWh) for an asset over the simulation period.
        """
        power_trace = self.get_power_trace(asset)
        return np.sum(power_trace) * self.resolution

    def get_discharge_net(self, asset: Any) -> float:
        """
        Returns the total dispatched energy (MWh) for storage assets over the simulation period.
        Only counts positive discharge energy.
        """
        power_trace = self.get_discharge_trace(asset)
        return np.sum(power_trace) * self.resolution

    def get_charge_net(self, asset: Any) -> float:
        """
        Returns the total dispatched energy (MWh) for storage assets over the simulation period.
        Only counts positive discharge energy.
        """
        power_trace = self.get_charge_trace(asset)
        return np.sum(power_trace) * self.resolution

    def get_line_use_net(self, asset: Any) -> float:
        """
        Returns the total line use (MWh) for line assets over the simulation period.
        """
        return np.sum(self.get_transmission_trace(asset)) * self.resolution

    def get_storage_losses(self, asset: Any) -> float:
        """
        Returns the total storage losses (MWh) for storage assets over the simulation period.
        """
        stored_energy = self.get_storage_level_trace(asset)
        return (-(np.sum(self.get_charge_trace(asset)) + np.sum(self.get_discharge_trace(asset)))
                * self.resolution
                - (stored_energy[-1] - stored_energy[0])
                )

    def get_line_losses(self, asset: Any) -> float:
        """
        Returns the total line losses (MWh) for line assets over the simulation period.
        """
        # TODO: line losses
        return self.get_zero()

    def get_nominal_curtailment_net(self, asset: Any) -> NDArray[np.float64]:
        """
        get time series of curtailment for a given asset (calculated by apportioning spillage)
        nominal curtailment is all spillage apportioned according to the asset's share of generation at the node
        """
        curtailment = self.get_nominal_curtailment_trace(asset)
        return np.sum(curtailment) * self.resolution
