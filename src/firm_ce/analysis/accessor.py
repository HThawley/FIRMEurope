import numpy as np
from typing import Any
from numpy.typing import NDArray

from firm_ce.common.typing import npfloat
from firm_ce.common.helpers import safe_divide_array
from firm_ce.common.constants import TOLERANCE


asset_class_to_display = {
    "generators": "Generator",
    "storages": "Storage",
    "major_lines": "Major Line",
    "minor_lines": "Minor Line",
    "nodes": "Node",
    "fuels": "Fuel",
}


# --- Data Retrievers ---
class Accessor:
    def __init__(self, solution, units=1.0):
        self.solution = solution
        self.resolution = solution.static.resolution
        self._curtailment_cache = {}
        self.update_units(units)

        self._trace_registry = {
            "power": (self._power_trace_single, ("generators", "storages")),
            "dispatch": (self._dispatch_trace_single, ("generators", "storages")),
            "generation": (self._generation_trace_single, ("generators",)),
            "discharge": (self._discharge_trace_single, ("storages",)),
            "charge": (self._charge_trace_single, ("storages",)),
            "charge_loss": (self._charge_loss_trace_single, ("storages",)),
            "discharge_loss": (self._discharge_loss_trace_single, ("storages",)),
            "storage_loss": (self._storage_loss_trace_single, ("storages",)),
            "inflow": (self._inflow_trace_single, ("storages",)),
            "spillage": (self._spillage_trace_single, ("storages",)),
            "storage_level": (self._storage_level_trace_single, ("storages",)),
            "remaining_energy": (self._remaining_energy_trace_single, ("fuels",)),
            "demand": (self._demand_trace_single, ("nodes",)),
            "deficit": (self._deficit_trace_single, ("nodes",)),
            "curtail": (self._curtail_trace_single, ("nodes",)),
            "net_flow": (self._net_flow_trace_single, ("nodes",)),
            "import": (self._import_trace_single, ("nodes",)),
            "export": (self._export_trace_single, ("nodes",)),
            "retention": (self._retention_trace_single, None),
            "nominal_curtailment": (self._nominal_curtailment_trace_single, ("generators", "storages")),
            "expected_curtailment": (self._expected_curtailment_trace_single, ("generators", "storages")),
            "post_curtailment_power": (self._post_curtailment_power_trace_single, ("generators", "storages")),
            "local_consumption": (self._local_consumption_trace_single, ("generators", "storages")),
            "transmission": (self._transmission_trace_single, ("major_lines", "minor_lines")),
            "line_loss": (self._line_loss_trace_single, ("major_lines", "minor_lines")),
            "line_use": (self._line_use_trace_single, ("major_lines", "minor_lines")),
        }

    def update_units(self, units: Any):
        if isinstance(units, str):
            match units.lower():
                case "mw" | "mwh":
                    self.factor = 1000.0
                case "gw" | "gwh":
                    self.factor = 1.0
                case _:
                    raise ValueError(f"Unknown units for capacity retrieval: {units}")
        elif isinstance(units, (int, float)):
            self.factor = float(units)
        else:
            raise ValueError(f"Unknown units for capacity retrieval: {units}")

        self.tolerance = TOLERANCE * self.factor

    # --- Asset Type Checkers ---
    @staticmethod
    def is_any(asset: Any) -> bool:
        return True

    @staticmethod
    def is_system(asset: Any) -> bool:
        return (isinstance(asset, str) and asset.lower() == "system")

    @staticmethod
    def is_flexible(asset: Any) -> bool:
        return getattr(asset, "is_flexible", False)

    @staticmethod
    def is_not_flexible(asset: Any) -> bool:
        return not getattr(asset, "is_flexible", False)

    @staticmethod
    def has_inflows(asset: Any) -> bool:
        return getattr(asset, "inflows", False)

    @staticmethod
    def is_fuel(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "fuel"

    @staticmethod
    def is_solar(asset: Any) -> bool:
        return getattr(asset, "unit_type", None) == "solar"

    @staticmethod
    def is_ror(asset: Any) -> bool:
        return getattr(asset, "unit_type", None) == "ror"

    @staticmethod
    def is_wind(asset: Any) -> bool:
        return getattr(asset, "unit_type", None) == "wind"

    @staticmethod
    def is_baseload(asset: Any) -> bool:
        return getattr(asset, "unit_type", None) == "baseload"

    @staticmethod
    def is_generator(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "generator"

    @staticmethod
    def is_storage(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "storage"

    @staticmethod
    def is_line(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "line"

    @staticmethod
    def is_major_line(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "line" and getattr(asset, "major", False)

    @staticmethod
    def is_minor_line(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "line" and not getattr(asset, "major", False)

    @staticmethod
    def is_node(asset: Any) -> bool:
        return getattr(asset, "object_class", None) == "node"

    @staticmethod
    def get_zero(*args) -> float:
        return 0.0

    # -- Objects --
    @staticmethod
    def get_assets_from_solution(solution, asset_class: str, errors: str = "raise") -> dict[str, Any]:
        """Static method version of get_assets."""
        match asset_class:
            case "generators" | "storages" | "fuels":
                return getattr(solution.fleet, asset_class)
            case "major_lines" | "minor_lines" | "nodes":
                return getattr(solution.network, asset_class)
            case _:
                return _handle_errors(errors, f"Unknown asset class for asset retrieval: {asset_class}")

    def get_assets(self, asset_class: str, errors: str = "raise") -> dict[str, Any]:
        """Returns the assets for a given asset class."""
        return self.get_assets_from_solution(self.solution, asset_class, errors=errors)

    @staticmethod
    def get_display_name(asset_class: str) -> str:
        return asset_class_to_display.get(asset_class, asset_class)

    # -- Capacity --
    @staticmethod
    def get_power_capacity(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of installed capacity in GW."""
        match asset.object_class:
            case "generator" | "line":
                return asset.capacity
            case "storage":
                return asset.power_capacity
            case _:
                return _handle_errors(
                    errors, f"Unknown asset type for capacity retrieval: {asset.name} ({asset.object_class})"
                )

    @staticmethod
    def get_energy_capacity(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of installed capacity in GWh."""
        match asset.object_class:
            case "generator" | "line":
                return _handle_errors(
                    errors, f"Asset: {asset.name} ({asset.object_class}) does not have energy capacity."
                )
            case "storage":
                return asset.energy_capacity
            case _:
                return _handle_errors(
                    errors, f"Unknown asset type for capacity retrieval: {asset.name} ({asset.object_class})"
                )

    def get_capacity(self, asset: Any, attribute: str, errors: str = "raise") -> float:
        """Safe retrieval of installed capacity in GW / GWh."""
        match attribute.lower():
            case "power":
                return self.get_power_capacity(asset, errors=errors)
            case "energy":
                return self.get_energy_capacity(asset, errors=errors)
            case _:
                return _handle_errors(errors, f"Unknown attribute for capacity retrieval: '{attribute}'")

    @staticmethod
    def get_new_build_power(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of new build capacity in GW."""
        match asset.object_class:
            case "generator" | "line":
                return asset.new_build
            case "storage":
                return asset.new_build_p
            case _:
                return _handle_errors(
                    errors, f"Unknown asset type for new_build (power) retrieval: {asset.name} ({asset.object_class})"
                )

    @staticmethod
    def get_new_build_energy(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of new build capacity in GWh."""
        match asset.object_class:
            case "generator" | "line":
                return _handle_errors(
                    errors, f"Asset: {asset.name} ({asset.object_class}) does not have energy capacity."
                )
            case "storage":
                return asset.new_build_e
            case _:
                return _handle_errors(
                    errors, f"Unknown asset type for new_build (energy) retrieval: {asset.name} ({asset.object_class})"
                )

    def get_new_build_capacity(self, asset: Any, attribute: str, errors: str = "raise") -> float:
        """Safe retrieval of new build capacity in GW / GWh."""
        match attribute.lower():
            case "power":
                return self.get_new_build_power(asset, errors=errors)
            case "energy":
                return self.get_new_build_energy(asset, errors=errors)
            case _:
                return _handle_errors(errors, f"Unknown attribute for capacity retrieval: {attribute}")

    @staticmethod
    def get_existing_power_capacity(asset: Any, errors: str = "raise") -> float:
        match asset.object_class:
            case "generator" | "line":
                return asset.initial_capacity
            case "storage":
                return asset.initial_power_capacity
            case _:
                return _handle_errors(
                    errors,
                    f"Unknown asset type for existing capacity (power) retrieval: {asset.name} ({asset.object_class})",
                )

    @staticmethod
    def get_existing_energy_capacity(asset: Any, errors: str = "raise") -> float:
        match asset.object_class:
            case "generator" | "line":
                return _handle_errors(
                    errors, f"Asset: {asset.name} ({asset.object_class}) does not have energy capacity."
                )
            case "storage":
                return asset.initial_energy_capacity
            case _:
                return _handle_errors(
                    errors,
                    f"Unknown asset type for existing capacity (energy) retrieval: {asset.name} ({asset.object_class})",
                )

    def get_existing_capacity(self, asset: Any, attribute: str, errors: str = "raise") -> float:
        match attribute.lower():
            case "power":
                return self.get_existing_power_capacity(asset, errors=errors)
            case "energy":
                return self.get_existing_energy_capacity(asset, errors=errors)
            case _:
                return _handle_errors(errors, f"Unknown attribute for capacity retrieval: '{attribute}'")

    @staticmethod
    def get_build_power(asset: Any, errors: str = "raise") -> tuple[float, float, float, float]:
        """Returns the build limits for power capacity (existing, new_build, min_build, max_build)."""
        match asset.object_class:
            case "generator" | "line":
                return asset.initial_capacity, asset.new_build, asset.min_build, asset.max_build
            case "storage":
                return asset.initial_power_capacity, asset.new_build_p, asset.min_build_p, asset.max_build_p
            case _:
                return _handle_errors(
                    errors,
                    f"Unknown asset type for build limits (power) retrieval: {asset.name} ({asset.object_class})",
                    (np.nan, np.nan, np.nan, np.nan),
                )

    @staticmethod
    def get_build_energy(asset: Any, errors: str = "raise") -> tuple[float, float, float, float]:
        """Returns the build limits for energy capacity (existing, new_build, min_build, max_build)."""
        match asset.object_class:
            case "storage":
                return asset.initial_energy_capacity, asset.new_build_e, asset.min_build_e, asset.max_build_e
            case "generator" | "line":
                return _handle_errors(
                    errors,
                    f"Asset: {asset.name} ({asset.object_class}) does not have energy capacity.",
                    (np.nan, np.nan, np.nan, np.nan),
                )
            case _:
                return _handle_errors(
                    errors,
                    f"Unknown asset type for build limits (energy) retrieval: {asset.name} ({asset.object_class})",
                    (np.nan, np.nan, np.nan, np.nan),
                )

    def get_build(self, asset: Any, attribute: str, errors: str = "raise") -> tuple[float, float, float, float]:
        """Returns the build limits for capacity (existing, new_build, min_build, max_build)."""
        match attribute.lower():
            case "power":
                return self.get_build_power(asset, errors=errors)
            case "energy":
                return self.get_build_energy(asset, errors=errors)
            case _:
                return _handle_errors(errors, f"Unknown attribute for build limits retrieval: '{attribute}'")

    @staticmethod
    def get_annualised_build_cost(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of total annualised capital costs (Power + Energy)."""
        if hasattr(asset, "lt_costs"):
            p_cost = getattr(asset.lt_costs, "annualised_build_p", 0.0)
            e_cost = getattr(asset.lt_costs, "annualised_build_e", 0.0)
            return p_cost + e_cost

        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    @staticmethod
    def get_fixed_om_cost(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of fixed operations & maintenance costs."""
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "fom", 0.0)

        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    @staticmethod
    def get_variable_om_cost(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of variable operations & maintenance costs."""
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "vom", 0.0)

        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    @staticmethod
    def get_fuel_cost(asset: Any, errors: str = "raise") -> float:
        """Safe retrieval of total fuel costs."""
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "fuel", 0.0)

        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    def get_all_costs(self, asset: Any, errors: str = "raise") -> dict[str, float]:
        """Returns a dictionary containing all standard long-term costs for an asset."""
        return {
            "Annualised Build": self.get_annualised_build_cost(asset, errors=errors),
            "Fixed O&M": self.get_fixed_om_cost(asset, errors=errors),
            "Variable O&M": self.get_variable_om_cost(asset, errors=errors),
            "Fuel Cost": self.get_fuel_cost(asset, errors=errors),
        }

    # -- Other static attributes --
    @staticmethod
    def get_charge_efficiency(asset: Any) -> float:
        _check_asset_has_attr(asset, "charge_efficiency")
        return asset.charge_efficiency

    @staticmethod
    def get_discharge_efficiency(asset: Any) -> float:
        _check_asset_has_attr(asset, "discharge_efficiency")
        return asset.discharge_efficiency

    @staticmethod
    def get_round_efficiency(asset: Any) -> float:
        _check_asset_has_attr(asset, "discharge_efficiency")
        _check_asset_has_attr(asset, "charge_efficiency")
        return asset.discharge_efficiency * asset.charge_efficiency

    @staticmethod
    def get_transm_efficiency(asset: Any) -> float:
        _check_asset_has_attr(asset, "efficiency")
        return asset.efficiency

    def get_efficiency(self, asset: Any, attribute: str = None) -> float:
        match asset.object_class:
            case "line" | "route":
                return self.get_transm_efficiency(asset)
            case "storage":
                match attribute:
                    case "charge":
                        return self.get_charge_efficiency(asset)
                    case "discharge":
                        return self.get_discharge_efficiency(asset)
                    case "round":
                        return self.get_round_efficiency(asset)
                    case _:
                        raise ValueError(
                            "Cannot retrieve efficiency of storage object. Supply attribute (charge, discharge, round)."
                        )
            case _:
                raise ValueError(f"Unknown asset type for efficiency retrieval: {asset.name} ({asset.object_class})")

    # =========================================================================
    # --- Core Trace & Gross Routing Architecture ---
    # =========================================================================

    def _zeros(self) -> NDArray[npfloat]:
        return np.zeros(self.solution.static.intervals_count, dtype=npfloat)

    def _trace_to_gross(self, trace: NDArray[npfloat]) -> npfloat:
        """Returns the total energy (xWh) or volume for a given time series."""
        return np.sum(trace) * self.resolution

    def get_trace(self, metric: str, asset: Any = "system") -> NDArray[npfloat]:
        """
        Universal time-series dispatcher.
        `asset` can be an individual asset (Generator, Storage, Line, Fuel),
        a `Node` object, or `"system"`.
        """
        if metric not in self._trace_registry:
            raise ValueError(f"Unknown trace metric: '{metric}'. Valid options: {list(self._trace_registry.keys())}")

        single_fn, agg_classes = self._trace_registry[metric]

        # System-wide ("system") scope
        if self.is_system(asset):
            if agg_classes is None:
                raise ValueError(f"Metric '{metric}' cannot be aggregated to the system level.")
            total = self._zeros()
            for cls in agg_classes:
                for a in self.get_assets(cls).values():
                    total += single_fn(a)
            return total

        # 2. Nodal scope
        if self.is_node(asset):
            if agg_classes is None or "nodes" in agg_classes:
                return single_fn(asset)
            if any(cls in ("major_lines", "minor_lines", "fuels") for cls in agg_classes):
                raise ValueError(f"Metric '{metric}' is defined on {agg_classes} and cannot be aggregated by Node.")

            total = self._zeros()
            for a in self._get_assets_at_node_cached(asset.id, agg_classes):
                total += single_fn(a)
            return total

        # 3. Single asset scope
        return single_fn(asset)

    def get_gross(self, metric: str, asset: Any = "system") -> npfloat:
        """
        Universal gross energy dispatcher (xWh).
        Integrates `get_trace(metric, asset)` over the simulation period.
        """
        if metric in ("storage_level", "remaining_energy", "retention"):
            raise ValueError(f"Metric '{metric}' is a state/ratio trace and cannot be integrated to gross energy.")
        return self._trace_to_gross(self.get_trace(metric, asset))

    # =========================================================================
    # --- Single-Entity Trace Implementations ---
    # =========================================================================

    def _power_trace_single(self, asset: Any) -> NDArray[npfloat]:
        match asset.object_class:
            case "generator":
                if self.is_flexible(asset):
                    _check_asset_has_attr(asset, "dispatch_power")
                    return asset.dispatch_power * self.factor
                else:
                    _check_asset_has_attr(asset, "data")
                    _check_asset_has_attr(asset, "capacity")
                    return asset.data * asset.capacity * self.factor
            case "storage":
                _check_asset_has_attr(asset, "dispatch_power")
                return asset.dispatch_power * self.factor
            case "line":
                _check_asset_has_attr(asset, "flows")
                return asset.flows * self.factor
            case _:
                raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not support power trace.")

    def _dispatch_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Positive power output from either a generator or a storage asset."""
        if not (self.is_generator(asset) or self.is_storage(asset)):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is neither a Generator nor a Storage.")
        return np.maximum(0.0, self._power_trace_single(asset))

    def _generation_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Positive power output strictly from a generator."""
        if not self.is_generator(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Generator.")
        return np.maximum(0.0, self._power_trace_single(asset))

    def _discharge_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Positive power output strictly from a storage asset."""
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        return np.maximum(0.0, self._power_trace_single(asset))

    def _charge_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Negative power component (charging load <= 0) strictly for a storage asset."""
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        return np.minimum(0.0, self._power_trace_single(asset))

    def _charge_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Thermodynamic power lost while charging (xW >= 0)."""
        eff = self.get_charge_efficiency(asset)
        return (1.0 - eff) * np.abs(self._charge_trace_single(asset))

    def _discharge_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Thermodynamic power lost while discharging (xW >= 0)."""
        eff = self.get_discharge_efficiency(asset)
        return (1.0 / eff - 1.0) * self._discharge_trace_single(asset)

    def _storage_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Total thermodynamic power loss (charging + discharging, xW >= 0)."""
        return self._charge_loss_trace_single(asset) + self._discharge_loss_trace_single(asset)

    def _inflow_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Inflow power equivalent (xW) for reservoir storages."""
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        if not self.has_inflows(asset):
            return self._zeros()
        _check_asset_has_data(asset)
        return asset.data * self.factor / self.resolution

    def _storage_level_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Stored energy time series (xWh) at the end of each interval."""
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        _check_asset_has_attr(asset, "stored_energy")
        return asset.stored_energy * self.factor

    def _spillage_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Spilled inflow power time series (xW) for a storage asset."""
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        if not self.has_inflows(asset):
            return self._zeros()

        inflows = self._inflow_trace_single(asset) * self.resolution  # xWh
        stored = self._storage_level_trace_single(asset)  # xWh (end of interval)

        delta_e = np.empty_like(stored)
        initial_soc = 0.5 * self.get_energy_capacity(asset) * self.factor
        delta_e[0] = stored[0] - initial_soc
        delta_e[1:] = np.diff(stored)

        eff_c = self.get_charge_efficiency(asset)
        eff_d = self.get_discharge_efficiency(asset)

        charge_internal = np.abs(self._charge_trace_single(asset)) * self.resolution * eff_c
        discharge_internal = (self._discharge_trace_single(asset) * self.resolution) / eff_d

        spillage_xwh = np.maximum(0.0, inflows + charge_internal - discharge_internal - delta_e)
        return spillage_xwh / self.resolution

    def _remaining_energy_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_fuel(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Fuel.")
        _check_asset_has_attr(asset, "remaining_energy")
        return asset.remaining_energy * self.factor

    def _demand_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_node(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Node.")
        _check_asset_has_attr(asset, "data")
        return asset.data * self.factor

    def _curtail_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_node(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Node.")
        _check_asset_has_attr(asset, "curtail")
        return np.abs(asset.curtail) * self.factor

    def _deficit_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_node(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Node.")
        _check_asset_has_attr(asset, "deficits")
        return asset.deficits * self.factor

    def _net_flow_trace_single(self, asset: Any) -> NDArray[npfloat]:
        """Positive = Net Import, Negative = Net Export."""
        if not self.is_node(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Node.")
        _check_asset_has_attr(asset, "imports_exports")
        return asset.imports_exports * self.factor

    def _import_trace_single(self, asset: Any) -> NDArray[npfloat]:
        return np.maximum(0.0, self._net_flow_trace_single(asset))

    def _export_trace_single(self, asset: Any) -> NDArray[npfloat]:
        return np.maximum(0.0, -self._net_flow_trace_single(asset))

    def _transmission_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_line(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Line.")
        _check_asset_has_attr(asset, "flows")
        return asset.flows * self.factor

    def _line_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        flows = self._transmission_trace_single(asset)
        efficiency = self.get_transm_efficiency(asset)
        return np.abs(flows) * (1.0 - efficiency)

    def _line_use_trace_single(self, asset: Any) -> NDArray[npfloat]:
        return np.abs(self._transmission_trace_single(asset))

    def _nominal_curtailment_trace_single(self, asset: Any) -> NDArray[npfloat]:
        nodal_dispatch = self.get_trace("dispatch", asset.node)
        nodal_curtailment = self._curtail_trace_single(asset.node)

        excess = nodal_curtailment - nodal_dispatch
        if np.any(excess > self.tolerance):
            bad_idx = int(np.argmax(excess))
            raise RuntimeError(
                f"Unallocated nominal curtailment at Node '{asset.node.name}' (id={asset.node.id}): "
                f"curtailment exceeds total nodal dispatch by {excess[bad_idx]:.6f} at interval {bad_idx}."
            )

        asset_dispatch = self._dispatch_trace_single(asset)
        share = np.empty_like(nodal_curtailment)
        share = safe_divide_array(asset_dispatch, nodal_dispatch, share)
        return np.minimum(asset_dispatch, nodal_curtailment * share)

    def _expected_curtailment_trace_single(self, asset: Any) -> NDArray[npfloat]:
        tier_gen_totals, tier_curt_totals = self._compute_nodal_tier_data(asset.node)
        tier = self._get_asset_tier(asset)

        asset_dispatch = self._dispatch_trace_single(asset)
        share_of_tier = np.empty_like(asset_dispatch)
        share_of_tier = safe_divide_array(asset_dispatch, tier_gen_totals[tier], share_of_tier)

        return tier_curt_totals[tier] * share_of_tier

    def _post_curtailment_power_trace_single(self, asset: Any) -> NDArray[npfloat]:
        curtailment = self._expected_curtailment_trace_single(asset)
        dispatch = self._dispatch_trace_single(asset)

        if self.is_storage(asset):
            if ((curtailment > self.tolerance) & ~(dispatch > self.tolerance)).any():
                raise RuntimeError(f"Storage {asset.name} is curtailed while not dispatching")

        return dispatch - curtailment

    def _retention_trace_single(self, node: Any) -> NDArray[npfloat]:
        if not self.is_node(node):
            raise ValueError(f"Asset {node.name} ({node.object_class}) is not a Node.")

        actual_nodal_gen = self.get_trace("post_curtailment_power", node)
        exports = self._export_trace_single(node)

        retention = np.empty_like(actual_nodal_gen)
        retention = safe_divide_array(actual_nodal_gen - exports, actual_nodal_gen, retention)
        return np.maximum(0.0, retention)

    def _local_consumption_trace_single(self, asset: Any) -> NDArray[npfloat]:
        gen_trace = self._post_curtailment_power_trace_single(asset)
        retention_trace = self._retention_trace_single(asset.node)
        return gen_trace * retention_trace

    # =========================================================================
    # --- Convenience Trace & Gross Wrappers ---
    # =========================================================================

    def get_power_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("power", asset)

    def get_dispatch_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("dispatch", asset)

    def get_generation_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("generation", asset)

    def get_discharge_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("discharge", asset)

    def get_charge_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("charge", asset)

    def get_charge_loss_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("charge_loss", asset)

    def get_discharge_loss_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("discharge_loss", asset)

    def get_storage_loss_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("storage_loss", asset)

    def get_inflow_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("inflow", asset)

    def get_spillage_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("spillage", asset)

    def get_storage_level_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("storage_level", asset)

    def get_remaining_energy_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("remaining_energy", asset)

    def get_demand_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("demand", asset)

    def get_deficit_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("deficit", asset)

    def get_curtail_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("curtail", asset)

    def get_net_flow_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("net_flow", asset)

    def get_import_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("import", asset)

    def get_export_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("export", asset)

    def get_retention_trace(self, node: Any) -> NDArray[npfloat]:
        return self.get_trace("retention", node)

    def get_nominal_curtailment_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("nominal_curtailment", asset)

    def get_expected_curtailment_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("expected_curtailment", asset)

    def get_post_curtailment_power_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("post_curtailment_power", asset)

    def get_local_consumption_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("local_consumption", asset)

    def get_transmission_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("transmission", asset)

    def get_line_loss_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("line_loss", asset)

    def get_line_use_trace(self, asset: Any = "system") -> NDArray[npfloat]:
        return self.get_trace("line_use", asset)

    # -- Gross Energy Wrappers --
    def get_power_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("power", asset)

    def get_dispatch_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("dispatch", asset)

    def get_generation_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("generation", asset)

    def get_discharge_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("discharge", asset)

    def get_charge_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("charge", asset)

    def get_charge_loss_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("charge_loss", asset)

    def get_discharge_loss_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("discharge_loss", asset)

    def get_storage_loss_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("storage_loss", asset)

    def get_inflow_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("inflow", asset)

    def get_spillage_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("spillage", asset)

    def get_demand_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("demand", asset)

    def get_deficit_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("deficit", asset)

    def get_curtail_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("curtail", asset)

    def get_import_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("import", asset)

    def get_export_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("export", asset)

    def get_nominal_curtailment_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("nominal_curtailment", asset)

    def get_expected_curtailment_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("expected_curtailment", asset)

    def get_post_curtailment_power_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("post_curtailment_power", asset)

    def get_local_consumption_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("local_consumption", asset)

    def get_transmission_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("transmission", asset)

    def get_line_loss_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("line_loss", asset)

    def get_line_use_gross(self, asset: Any = "system") -> npfloat:
        return self.get_gross("line_use", asset)

    # =========================================================================
    # --- Internal Curtailment & Caching Helpers ---
    # =========================================================================

    def _get_asset_tier(self, asset: Any) -> int:
        """
        Curtailment merit order:
            1. Storage and flexibles
            2. Solar, wind, ror
            3. Reserved for future dev
            4. Others
        """
        if self.is_storage(asset) or self.is_flexible(asset):
            return 1
        if self.is_solar(asset) or self.is_wind(asset) or self.is_ror(asset):
            return 2
        return 4

    def _get_assets_at_node_cached(
        self, node_id: int, asset_classes: tuple[str, ...] = ("generators", "storages")
    ) -> list[Any]:
        cache_key = (f"assets_at_{node_id}", asset_classes)
        if cache_key in self._curtailment_cache:
            return self._curtailment_cache[cache_key]

        assets = []
        for asset_class in asset_classes:
            for asset in self.get_assets(asset_class).values():
                if asset.node.id == node_id:
                    assets.append(asset)

        self._curtailment_cache[cache_key] = assets
        return assets

    def _compute_nodal_tier_data(self, node: Any) -> tuple[dict, dict]:
        """
        Calculates dispatch and allocated curtailment for each priority tier at a node.
        Returns:
            (tier_dispatch_traces, tier_curtailment_traces)
        """
        cache_key = f"tier_data_{node.id}"
        if cache_key in self._curtailment_cache:
            return self._curtailment_cache[cache_key]

        assets = self._get_assets_at_node_cached(node.id, ("generators", "storages"))
        curtail = self._curtail_trace_single(node)

        zeros = self._zeros()
        tier_gen = {1: zeros.copy(), 2: zeros.copy(), 3: zeros.copy(), 4: zeros.copy()}

        for asset in assets:
            tier = self._get_asset_tier(asset)
            tier_gen[tier] += self._dispatch_trace_single(asset)

        tier_curtailment = {}
        remaining_curtail = curtail.copy()

        for tier in range(1, 5):
            allocated_curtailment = np.minimum(remaining_curtail, tier_gen[tier])
            tier_curtailment[tier] = allocated_curtailment
            remaining_curtail -= allocated_curtailment

        if np.any(remaining_curtail > self.tolerance):
            bad_idx = int(np.argmax(remaining_curtail))
            raise RuntimeError(
                f"Unallocated curtailment at Node '{node.name}' (id={node.id}): "
                f"{remaining_curtail[bad_idx]:.6f} exceeds total nodal dispatch at interval {bad_idx}."
            )

        result = (tier_gen, tier_curtailment)
        self._curtailment_cache[cache_key] = result
        return result


# -- General Utility Helpers --
def _check_asset_has_attr(asset: Any, attr: str, raise_if_missing: bool = True):
    if not hasattr(asset, attr):
        if raise_if_missing:
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have '{attr}' attribute.")
        return False
    return True


def _check_asset_has_data(asset: Any, raise_if_missing: bool = True):
    _check_asset_has_attr(asset, "data", raise_if_missing=raise_if_missing)
    if not asset.data_status:
        if raise_if_missing:
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) has data_status=False, data not loaded.")
        return False
    return True


def _handle_errors(errors: str, message: str, coerce_value=np.nan):
    if errors == "raise":
        raise ValueError(message)
    elif errors == "coerce":
        return coerce_value
    raise ValueError(f"Unknown error handling method. Expected 'raise' or 'coerce'. Got: {errors}")
