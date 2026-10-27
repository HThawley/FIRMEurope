from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Literal, Sequence, Union

AssetClass = Literal["generators", "storages", "major_lines"]
MetricType = Literal[
    "power_capacity",
    "energy_capacity",
    "dispatch",
    "dispatch_and_charge",
    "post_curtailment_power",
    "discharge",
    "charge",
    "inflows",
    "storage_sources",
    "line_flow",
    "line_flow_net",
]
BuildFilter = Literal["all", "new_build", "existing", "initial"]
BalanceType = Literal["curtailment", "storage_losses", "spillage", "line_losses"]


# ==============================================================================
# Domain Predicates & Spec Architecture
# ==============================================================================


def is_hydro_asset(asset, base_tech: str = "") -> bool:
    """Identifies hydro/pondage/RoR assets even when classified under storage objects."""
    raw_type = str(getattr(asset, "unit_type", "")).lower()
    return base_tech in ("Hydro", "Pondage", "Run of River") or raw_type in ("hydro", "pond", "ror")


def is_rechargeable_storage(asset, base_tech: str = "") -> bool:
    """Identifies non-hydro rechargeable storage (BESS, PHES)."""
    return getattr(asset, "object_class", "") == "storage" and not is_hydro_asset(asset, base_tech)


def is_generation_or_hydro(asset, base_tech: str = "") -> bool:
    """Matches generators plus hydro storages treated as generation fleet."""
    if getattr(asset, "object_class", "") == "storage":
        return is_hydro_asset(asset, base_tech)
    return True


MetricType = Literal[
    "none",
    "power_capacity",
    "energy_capacity",
    "dispatch",
    "post_curtailment_power",
    "discharge",
    "charge",
    "inflows",
    "storage_sources",
    "line_flow",
    "line_flow_net",
]


@dataclass(frozen=True)
class MetricSpec:
    title: str
    unit: str
    assets: Sequence[AssetClass]
    metric: MetricType = "none"  # type: ignore
    build: BuildFilter = "all"
    asset_filter: Callable[[object, str], bool] = lambda asset, base_tech: True
    group_by: Union[Literal["tech", "subtech", "node"], Callable[[object], str]] = "subtech"
    include_balances: Sequence[BalanceType] = field(default_factory=tuple)
    negate_balances: bool = False  # Plots balances (curtailment, losses, spillage) below 0
    annualize: bool = False
    scale_factor: float = 1.0
    norm_ref: int | None = None  # Index of spec in preset to normalize against when normalize=True
    norm_by: Literal["self", "demand"] = "self"  # "demand" sets 100% = system demand

    def with_options(self, **kwargs) -> "MetricSpec":
        """Returns a modified copy of this MetricSpec with updated fields."""
        return replace(self, **kwargs)

# ==============================================================================
# 2. Preset Registry (Single Node/Map Specs & Summary Spec Groupings)
# ==============================================================================


MAP_SPECS: Dict[str, MetricSpec] = {
    "capacity": MetricSpec(
        title="Power Capacity",
        unit="GW",
        assets=("generators", "storages"),
        metric="power_capacity",
        group_by="tech",
    ),
    "new_capacity": MetricSpec(
        title="New Build Power Capacity",
        unit="GW",
        assets=("generators", "storages"),
        metric="power_capacity",
        build="new_build",
        group_by="tech",
    ),
    "energy": MetricSpec(
        title="Energy Mix",
        unit="GWh",
        assets=("generators", "storages"),
        metric="dispatch",
        group_by="tech",
    ),
    "net_energy": MetricSpec(
        title="Post-Curtailment Energy Mix",
        unit="GWh",
        assets=("generators", "storages"),
        metric="post_curtailment_power",
        group_by="tech",
    ),
    "line_capacity": MetricSpec(
        title="Transmission Capacity",
        unit="GW",
        assets=("major_lines",),
        metric="power_capacity",
    ),
    "line_energy": MetricSpec(
        title="Transmission Flows",
        unit="GWh",
        assets=("major_lines",),
        metric="line_flow",
    ),
}

SUMMARY_PRESETS: Dict[str, List[MetricSpec]] = {
    "system_overview": [
        MetricSpec(
            title="Power Capacity",
            unit="GW",
            assets=("generators", "storages"),
            metric="power_capacity",
        ),
        MetricSpec(
            title="Annual Energy Mix",
            unit="TWh/yr",
            assets=("generators", "storages"),
            metric="dispatch",
            include_balances=("curtailment", "storage_losses", "spillage"),
            annualize=True,
        ),
    ],
    "power_capacity": [
        MetricSpec(
            title="Generation",
            unit="GW",
            assets=("generators", "storages"),
            metric="power_capacity",
            asset_filter=is_generation_or_hydro,
        ),
        MetricSpec(
            title="Transmission",
            unit="GW",
            assets=("major_lines",),
            metric="power_capacity",
        ),
        MetricSpec(
            title="Rechargeable Storage",
            unit="GW",
            assets=("storages",),
            metric="power_capacity",
            asset_filter=is_rechargeable_storage,
        ),
    ],
    "energy_balance": [
        MetricSpec(
            title="Generation Mix",
            unit="TWh/yr",
            assets=("generators", "storages"),
            metric="dispatch",
            asset_filter=is_generation_or_hydro,
            include_balances=("curtailment",),
            annualize=True,
        ),
        MetricSpec(
            title="Transmission Flows",
            unit="TWh/yr",
            assets=("major_lines",),
            metric="line_flow_net",
            include_balances=("line_losses",),
            annualize=True,
        ),
        MetricSpec(
            title="Storage Sources",
            unit="TWh/yr",
            assets=("storages",),
            metric="storage_sources",
            asset_filter=is_rechargeable_storage,
            annualize=True,
        ),
        MetricSpec(
            title="Storage Discharge",
            unit="TWh/yr",
            assets=("storages",),
            metric="discharge",
            asset_filter=is_rechargeable_storage,
            include_balances=("storage_losses", "spillage"),
            annualize=True,
        ),
    ],
    "storage_profile": [
        MetricSpec(
            title="Power Capacity",
            unit="GW",
            assets=("storages",),
            metric="power_capacity",
            asset_filter=is_rechargeable_storage,
        ),
        MetricSpec(
            title="Energy Capacity",
            unit="GWh",
            assets=("storages",),
            metric="energy_capacity",
            asset_filter=is_rechargeable_storage,
        ),
        MetricSpec(
            title="Hydro Energy Cap",
            unit="GWh",
            assets=("storages",),
            metric="energy_capacity",
            asset_filter=is_hydro_asset,
        ),
        MetricSpec(
            title="Discharge Energy",
            unit="TWh/yr",
            assets=("storages",),
            metric="discharge",
            asset_filter=is_rechargeable_storage,
            include_balances=("storage_losses", "spillage"),
            annualize=True,
        ),
    ],
    "system_overview_with_losses": [
        # Bar 0: Installed Power Capacity (stacks to 100% of total GW)
        MetricSpec(
            title="Power Capacity",
            unit="GW",
            assets=("generators", "storages"),
            metric="power_capacity",
        ),
        # Bar 1: Transmission capacity
        MetricSpec(
            title="Transmission Capacity",
            unit="GW",
            assets=("major_lines",),
            metric="power_capacity",
        ),
        # Bar 2: Gross Annual Energy Mix + negative Charging, Losses & Curtailment
        MetricSpec(
            title="Annual Energy Mix",
            unit="TWh/yr",
            assets=("generators", "storages", "major_lines"),
            metric="dispatch_and_charge",
            include_balances=("curtailment", "spillage", "line_losses"),
            negate_balances=True,
            annualize=True,
            norm_by="demand",
        ),
    ],
}
