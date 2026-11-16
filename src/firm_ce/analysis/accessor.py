from typing import Any
import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import TOLERANCE
from firm_ce.common.helpers import safe_divide_array
from firm_ce.common.typing import npfloat

asset_class_to_display = {
    "generators": "Generator",
    "storages": "Storage",
    "major_lines": "Major Line",
    "minor_lines": "Minor Line",
    "nodes": "Node",
    "fuels": "Fuel",
}

_CAP_ATTR_MAP = {
    ("generator", "power", "total"): "capacity",
    ("generator", "power", "new"): "new_build",
    ("generator", "power", "existing"): "initial_capacity",
    ("generator", "power", "min"): "min_build",
    ("generator", "power", "max"): "max_build",
    ("line", "power", "total"): "capacity",
    ("line", "power", "new"): "new_build",
    ("line", "power", "existing"): "initial_capacity",
    ("line", "power", "min"): "min_build",
    ("line", "power", "max"): "max_build",
    ("storage", "power", "total"): "power_capacity",
    ("storage", "power", "new"): "new_build_p",
    ("storage", "power", "existing"): "initial_power_capacity",
    ("storage", "power", "min"): "min_build_p",
    ("storage", "power", "max"): "max_build_p",
    ("storage", "energy", "total"): "energy_capacity",
    ("storage", "energy", "new"): "new_build_e",
    ("storage", "energy", "existing"): "initial_energy_capacity",
    ("storage", "energy", "min"): "min_build_e",
    ("storage", "energy", "max"): "max_build_e",
}


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
            "transmission": (self._transmission_trace_single, ("major_lines",)),
            "line_loss": (self._line_loss_trace_single, ("major_lines",)),
            "line_use": (self._line_use_trace_single, ("major_lines",)),
        }

    def __getattr__(self, name: str):
        """Dynamically resolves `get_<metric>_trace` and `get_<metric>_gross` calls."""
        if name.startswith("get_") and name.endswith("_trace"):
            metric = name[4:-6]
            if metric in self._trace_registry:
                return lambda asset="system": self.get_trace(metric, asset)
        if name.startswith("get_") and name.endswith("_gross"):
            metric = name[4:-6]
            if metric in self._trace_registry:
                return lambda asset="system": self.get_gross(metric, asset)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        return isinstance(asset, str) and asset.lower() == "system"

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
        return getattr(asset, "unit_type", None) in ("pv_track", "pv_fixed")

    @staticmethod
    def is_ror(asset: Any) -> bool:
        return getattr(asset, "unit_type", None) == "ror"

    @staticmethod
    def is_wind(asset: Any) -> bool:
        return getattr(asset, "unit_type", None) in ("onsw", "offw")

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

    # -- Objects --
    @staticmethod
    def get_assets_from_solution(solution, asset_class: str, errors: str = "raise") -> dict[str, Any]:
        match asset_class:
            case "generators" | "storages" | "fuels":
                return getattr(solution.fleet, asset_class)
            case "major_lines" | "minor_lines" | "nodes":
                return getattr(solution.network, asset_class)
            case _:
                return _handle_errors(errors, f"Unknown asset class for asset retrieval: {asset_class}")

    def get_assets(self, asset_class: str, errors: str = "raise") -> dict[str, Any]:
        return self.get_assets_from_solution(self.solution, asset_class, errors=errors)

    @staticmethod
    def get_display_name(asset_class: str) -> str:
        return asset_class_to_display.get(asset_class, asset_class)

    # -- Unified Capacity & Build Retrieval --
    @staticmethod
    def _lookup_cap(asset: Any, kind: str, build: str = "total", errors: str = "raise") -> float:
        key = (getattr(asset, "object_class", None), kind.lower(), build.lower())
        attr = _CAP_ATTR_MAP.get(key)
        if attr is not None:
            return getattr(asset, attr)
        return _handle_errors(
            errors,
            f"Asset '{getattr(asset, 'name', asset)}' ({getattr(asset, 'object_class', None)}) "
            f"does not support {build} {kind} capacity.",
        )

    @classmethod
    def get_power_capacity(cls, asset: Any, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, "power", "total", errors)

    @classmethod
    def get_energy_capacity(cls, asset: Any, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, "energy", "total", errors)

    @classmethod
    def get_capacity(cls, asset: Any, attribute: str, build: str = "total", errors: str = "raise") -> float:
        b_norm = {"all": "total", "none": "total", "new_build": "new", "initial": "existing"}.get(
            build.lower(), build.lower()
        )
        return cls._lookup_cap(asset, attribute, b_norm, errors)

    @classmethod
    def get_new_build_power(cls, asset: Any, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, "power", "new", errors)

    @classmethod
    def get_new_build_energy(cls, asset: Any, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, "energy", "new", errors)

    @classmethod
    def get_new_build_capacity(cls, asset: Any, attribute: str, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, attribute, "new", errors)

    @classmethod
    def get_existing_power_capacity(cls, asset: Any, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, "power", "existing", errors)

    @classmethod
    def get_existing_energy_capacity(cls, asset: Any, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, "energy", "existing", errors)

    @classmethod
    def get_existing_capacity(cls, asset: Any, attribute: str, errors: str = "raise") -> float:
        return cls._lookup_cap(asset, attribute, "existing", errors)

    @classmethod
    def get_build(cls, asset: Any, attribute: str, errors: str = "raise") -> tuple[float, float, float, float]:
        """Returns (existing, new_build, min_build, max_build) for 'power' or 'energy'."""
        kind = attribute.lower()
        if (getattr(asset, "object_class", None), kind, "total") not in _CAP_ATTR_MAP:
            return _handle_errors(
                errors,
                f"Asset: {getattr(asset, 'name', asset)} ({getattr(asset, 'object_class', None)}) "
                f"does not have {kind} build limits.",
                (np.nan, np.nan, np.nan, np.nan),
            )
        return (
            cls._lookup_cap(asset, kind, "existing", errors),
            cls._lookup_cap(asset, kind, "new", errors),
            cls._lookup_cap(asset, kind, "min", errors),
            cls._lookup_cap(asset, kind, "max", errors),
        )

    @classmethod
    def get_build_power(cls, asset: Any, errors: str = "raise") -> tuple[float, float, float, float]:
        return cls.get_build(asset, "power", errors=errors)

    @classmethod
    def get_build_energy(cls, asset: Any, errors: str = "raise") -> tuple[float, float, float, float]:
        return cls.get_build(asset, "energy", errors=errors)

    # -- Costs --
    @staticmethod
    def get_annualised_build_cost(asset: Any, errors: str = "raise") -> float:
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "annualised_build_p", 0.0) + getattr(
                asset.lt_costs, "annualised_build_e", 0.0
            )
        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    @staticmethod
    def get_fixed_om_cost(asset: Any, errors: str = "raise") -> float:
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "fom", 0.0)
        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    @staticmethod
    def get_variable_om_cost(asset: Any, errors: str = "raise") -> float:
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "vom", 0.0)
        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    @staticmethod
    def get_fuel_cost(asset: Any, errors: str = "raise") -> float:
        if hasattr(asset, "lt_costs"):
            return getattr(asset.lt_costs, "fuel", 0.0)
        return _handle_errors(errors, f"Asset {asset.name} ({asset.object_class}) does not have 'lt_costs'.")

    def get_all_costs(self, asset: Any, errors: str = "raise") -> dict[str, float]:
        return {
            "Annualised Build": self.get_annualised_build_cost(asset, errors=errors),
            "Fixed O&M": self.get_fixed_om_cost(asset, errors=errors),
            "Variable O&M": self.get_variable_om_cost(asset, errors=errors),
            "Fuel Cost": self.get_fuel_cost(asset, errors=errors),
        }

    # -- Efficiencies --
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
                        raise ValueError("Supply attribute ('charge', 'discharge', 'round') for storage efficiency.")
            case _:
                raise ValueError(f"Unknown asset type for efficiency retrieval: {asset.name} ({asset.object_class})")

    # =========================================================================
    # --- Core Trace & Gross Routing Architecture ---
    # =========================================================================

    def _zeros(self) -> NDArray[npfloat]:
        return np.zeros(self.solution.static.intervals_count, dtype=npfloat)

    def _trace_to_gross(self, trace: NDArray[npfloat]) -> npfloat:
        return np.sum(trace) * self.resolution

    def get_trace(self, metric: str, asset: Any = "system") -> NDArray[npfloat]:
        if metric not in self._trace_registry:
            raise ValueError(f"Unknown trace metric: '{metric}'. Valid options: {list(self._trace_registry.keys())}")

        single_fn, agg_classes = self._trace_registry[metric]

        if self.is_system(asset):
            if agg_classes is None:
                raise ValueError(f"Metric '{metric}' cannot be aggregated to the system level.")
            total = self._zeros()
            for cls in agg_classes:
                for a in self.get_assets(cls).values():
                    total += single_fn(a)
            return total

        if self.is_node(asset):
            if agg_classes is None or "nodes" in agg_classes:
                return single_fn(asset)
            if any(cls in ("major_lines", "minor_lines", "fuels") for cls in agg_classes):
                raise ValueError(f"Metric '{metric}' is defined on {agg_classes} and cannot be aggregated by Node.")
            total = self._zeros()
            for a in self._get_assets_at_node_cached(asset.id, agg_classes):
                total += single_fn(a)
            return total

        return single_fn(asset)

    def get_gross(self, metric: str, asset: Any = "system") -> npfloat:
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
        if not (self.is_generator(asset) or self.is_storage(asset)):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is neither a Generator nor a Storage.")
        return np.maximum(0.0, self._power_trace_single(asset))

    def _generation_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_generator(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Generator.")
        return np.maximum(0.0, self._power_trace_single(asset))

    def _discharge_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        return np.maximum(0.0, self._power_trace_single(asset))

    def _charge_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        return np.minimum(0.0, self._power_trace_single(asset))

    def _charge_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        eff = self.get_charge_efficiency(asset)
        return (1.0 - eff) * np.abs(self._charge_trace_single(asset))

    def _discharge_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        eff = self.get_discharge_efficiency(asset)
        return (1.0 / eff - 1.0) * self._discharge_trace_single(asset)

    def _storage_loss_trace_single(self, asset: Any) -> NDArray[npfloat]:
        return self._charge_loss_trace_single(asset) + self._discharge_loss_trace_single(asset)

    def _inflow_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        if not self.has_inflows(asset):
            return self._zeros()
        _check_asset_has_data(asset)
        return asset.data * self.factor / self.resolution

    def _storage_level_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        _check_asset_has_attr(asset, "stored_energy")
        return asset.stored_energy * self.factor

    def _spillage_trace_single(self, asset: Any) -> NDArray[npfloat]:
        if not self.is_storage(asset):
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) is not a Storage.")
        if not self.has_inflows(asset):
            return self._zeros()

        inflows = self._inflow_trace_single(asset) * self.resolution
        stored = self._storage_level_trace_single(asset)

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

        assets = [
            asset
            for asset_class in asset_classes
            for asset in self.get_assets(asset_class).values()
            if asset.node.id == node_id
        ]
        self._curtailment_cache[cache_key] = assets
        return assets

    def _compute_nodal_tier_data(self, node: Any) -> tuple[dict, dict]:
        cache_key = f"tier_data_{node.id}"
        if cache_key in self._curtailment_cache:
            return self._curtailment_cache[cache_key]

        assets = self._get_assets_at_node_cached(node.id, ("generators", "storages"))
        curtail = self._curtail_trace_single(node)

        zeros = self._zeros()
        tier_gen = {1: zeros.copy(), 2: zeros.copy(), 3: zeros.copy(), 4: zeros.copy()}

        for asset in assets:
            tier_gen[self._get_asset_tier(asset)] += self._dispatch_trace_single(asset)

        tier_curtailment = {}
        remaining_curtail = curtail.copy()

        for tier in range(1, 5):
            allocated = np.minimum(remaining_curtail, tier_gen[tier])
            tier_curtailment[tier] = allocated
            remaining_curtail -= allocated

        if np.any(remaining_curtail > self.tolerance):
            bad_idx = int(np.argmax(remaining_curtail))
            raise RuntimeError(
                f"Unallocated curtailment at Node '{node.name}' (id={node.id}): "
                f"{remaining_curtail[bad_idx]:.6f} exceeds total nodal dispatch at interval {bad_idx}."
            )

        result = (tier_gen, tier_curtailment)
        self._curtailment_cache[cache_key] = result
        return result


def _check_asset_has_attr(asset: Any, attr: str, raise_if_missing: bool = True) -> bool:
    if not hasattr(asset, attr):
        if raise_if_missing:
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) does not have '{attr}' attribute.")
        return False
    return True


def _check_asset_has_data(asset: Any, raise_if_missing: bool = True) -> bool:
    _check_asset_has_attr(asset, "data", raise_if_missing=raise_if_missing)
    if not asset.data_status:
        if raise_if_missing:
            raise ValueError(f"Asset {asset.name} ({asset.object_class}) has data_status=False, data not loaded.")
        return False
    return True


def _handle_errors(errors: str, message: str, coerce_value=np.nan):
    if errors == "raise":
        raise ValueError(message)
    if errors == "coerce":
        return coerce_value
    raise ValueError(f"Unknown error handling method. Expected 'raise' or 'coerce'. Got: {errors}")
