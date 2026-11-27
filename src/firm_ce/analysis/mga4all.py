# type: ignore
import calendar
from dataclasses import dataclass, field
import os
from typing import Dict, List, Literal, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import polars as pl

from firm_ce.analysis.accessor import Accessor
from firm_ce.common.constants import LEAPDAYS
from firm_ce.common.typing import npfloat
from firm_ce.scalar.solution import Solution, evaluate
from firm_ce.system.scenario import Scenario
from firm_ce.tensor.solution import EvaluateTensor, SolutionTensor, prep_solution_for_postprocessing
from firm_ce.tensor.to_scalar import map_tensor_to_scalar
from firm_ce.viz.style import TECH_COLORS

DAYS_NON_LEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
DAYS_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class TechMeta:
    tech: str
    color: str
    exclude_from_capacity_plot: bool
    tech_type: Literal["generation", "storage", "transmission", "demand", "other"]
    display_name: str
    carrier_in: str = ""
    carrier_out: str = ""
    from_node: str = ""
    to_node: str = ""


DEFAULT_SUBTECH_META: Dict[str, TechMeta] = {
    "solar_pv": TechMeta(
        tech="solar_pv",
        color="#F9D956",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Solar PV",
        carrier_in="",
        carrier_out="power",
    ),
    "wind_onshore": TechMeta(
        tech="wind_onshore",
        color="#56F9C0",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Onshore wind",
        carrier_in="",
        carrier_out="power",
    ),
    "wind_offshore": TechMeta(
        tech="wind_offshore",
        color="#56CBF9",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Offshore wind",
        carrier_in="",
        carrier_out="power",
    ),
    "bioenergy": TechMeta(
        tech="bioenergy",
        color="#936944",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Bioenergy power plant",
        carrier_in="",
        carrier_out="power",
    ),
    "nuclear": TechMeta(
        tech="nuclear",
        color="#e83a57",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Nuclear power plant",
        carrier_in="",
        carrier_out="power",
    ),
    "ocgt": TechMeta(
        tech="ocgt",
        color="#E37A72",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Open cycle gas turbine",
        carrier_in="",
        carrier_out="power",
    ),
    "ccgt": TechMeta(
        tech="ccgt",
        color="#E37A72",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Closed cycle gas turbine",
        carrier_in="",
        carrier_out="power",
    ),
    "coal": TechMeta(
        tech="coal",
        color="#48423d",
        exclude_from_capacity_plot=True,
        tech_type="generation",
        display_name="Coal plant",
        carrier_in="",
        carrier_out="power",
    ),
    "ror": TechMeta(
        tech="ror",
        color="#99F6E4",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Run of river hydro",
        carrier_in="",
        carrier_out="power",
    ),
    "hydro": TechMeta(
        tech="hydro",
        color="#0D9488",
        exclude_from_capacity_plot=False,
        tech_type="generation",
        display_name="Reservoir & pondage hydro",
        carrier_in="",
        carrier_out="power",
    ),
    "battery": TechMeta(
        tech="battery",
        color="#33F6FF",
        exclude_from_capacity_plot=False,
        tech_type="storage",
        display_name="Battery energy storage",
        carrier_in="power",
        carrier_out="power",
    ),
    "phes": TechMeta(
        tech="phes",
        color="#9958D6",
        exclude_from_capacity_plot=False,
        tech_type="storage",
        display_name="Pumped hydro energy storage",
        carrier_in="power",
        carrier_out="power",
    ),
    "curtailment": TechMeta(
        tech="curtailment",
        color="#70002a",
        exclude_from_capacity_plot=True,
        tech_type="other",
        display_name="Curtailment",
        carrier_in="power",
        carrier_out="power",
    ),
    "demand_power": TechMeta(
        tech="demand_power",
        color="#072486",
        exclude_from_capacity_plot=True,
        tech_type="demand",
        display_name="Power demand",
        carrier_in="power",
        carrier_out="",
    ),
    "lost_load": TechMeta(
        tech="lost_load",
        color="#030303",
        exclude_from_capacity_plot=True,
        tech_type="other",
        display_name="Lost load",
        carrier_in="power",
        carrier_out="power",
    ),
}

UNIT_TYPE_TO_SUBTECH: Dict[str, str] = {
    "pv_fixed": "solar_pv",
    "pv_track": "solar_pv",
    "onsw": "wind_onshore",
    "offw": "wind_offshore",
    "biomass": "bioenergy",
    "biogas": "bioenergy",
    "nuclear": "nuclear",
    "nuclear_lte": "nuclear",
    "ocgt": "ocgt",
    "ccgt": "ccgt",
    "coal": "coal",
    "ror": "ror",
    "hydro": "hydro",
    "pond": "hydro",
    "bess2h": "battery",
    "bess4h": "battery",
    "battery": "battery",
    "aa_phes": "phes",
    "e_phes": "phes",
    "clphes": "phes",
    "olphes": "phes",
}

UNIT_TYPE_DISPLAY_NAMES: Dict[str, str] = {
    "pv_fixed": "Fixed-tilt solar PV",
    "pv_track": "Single-axis tracking solar PV",
    "onsw": "Onshore wind",
    "offw": "Offshore wind",
    "biomass": "Biomass power plant",
    "biogas": "Biogas power plant",
    "nuclear": "Nuclear power plant",
    "nuclear_lte": "Nuclear power plant (LTE)",
    "ocgt": "Open cycle gas turbine",
    "ccgt": "Closed cycle gas turbine",
    "coal": "Coal plant",
    "ror": "Run of river hydro",
    "hydro": "Reservoir hydro",
    "pond": "Pondage hydro",
    "bess2h": "2-hour battery storage",
    "bess4h": "4-hour battery storage",
    "aa_phes": " Closed-loop PHES (Class AA)",
    "e_phes": "Closed-loop PHES (Class E)",
    "clphes": "Existing closed-loop PHES",
    "olphes": "Existing open-loop PHES",
    "ac_ohl_transmission": "AC OHL",
    "ac_ohl_mountain_transmission": "AC OHL (Mountain)",
    "dc_subsea_transmission": "DC Subsea",
    "dc_underground_transmission": "DC Underground",
}

LINE_UNIT_TYPE_SUFFIX: Dict[str, str] = {
    "ac_ohl_transmission": "ac_ohl",
    "ac_ohl_mountain_transmission": "ac_ohl_mountain",
    "dc_subsea_transmission": "dc_subsea",
    "dc_underground_transmission": "dc_underground",
}


@dataclass
class MGA4AllConfig:
    """Configuration options for exporting FIRM solutions to the mga4all format."""

    group_by: Literal["subtech", "unit_type", "asset"] = "subtech"
    units: Literal["MW", "GW"] = "MW"
    year: int | Sequence[int] | Literal["first", "last", "all"] | None = "last"
    merge_parallel_lines: bool = False
    transmission_color: str = "#6783E3"
    map_path: str = "./inputs/map/europe.geojson"
    include_zero_capacity_techs: bool = True
    uppercase_nodes: bool = True
    decimals: int | None = 6
    custom_tech_meta: Dict[str, TechMeta] = field(default_factory=dict)


def _coerce_scalar_solution(scenario: Scenario, sol: Union[Solution, SolutionTensor, NDArray[npfloat]]) -> Solution:
    if isinstance(sol, Solution):
        if not getattr(sol, "evaluated", False):
            evaluate(sol)
        return sol
    if isinstance(sol, SolutionTensor) or type(sol).__name__ == "SolutionTensor":
        if not getattr(sol, "evaluated", False):
            EvaluateTensor(sol)
        prep_solution_for_postprocessing(sol)
        return map_tensor_to_scalar(scenario, sol)
    if isinstance(sol, np.ndarray):
        scalar_sol, _ = scenario.build_and_evaluate_solution(sol, retain=False)
        return scalar_sol
    raise TypeError(f"Unsupported solution object of type '{type(sol).__name__}'.")


class MGA4AllExporter:
    """
    Converts evaluated FIRM solution(s) into the 4 CSV table types required by `mga4all`:
      1. `node_coordinates.csv`
      2. `tech_parameters.csv`
      3. `capacity_values.csv`
      4. `flow_timeseries_alternative_{i}.csv`
    """

    def __init__(
        self,
        scenario: Scenario,
        solutions: Union[Solution, SolutionTensor, Sequence[Union[Solution, SolutionTensor]]] | None = None,
        config: MGA4AllConfig | None = None,
        **kwargs,
    ):
        self.scenario = scenario
        if not self.scenario.data_status:
            self.scenario.load_datafiles()

        if config is None:
            self.config = MGA4AllConfig(**kwargs)
        else:
            self.config = config

        self.solutions: List[Solution] = self._resolve_solutions(solutions)
        self.static = self.solutions[0].static
        self.resolution = float(self.static.resolution)

        self._line_tech_names: Dict[int, str] = self._build_line_tech_name_map()
        self._time_mask, self.timesteps = self._build_timesteps()

    def _resolve_solutions(
        self,
        solutions: Union[Solution, SolutionTensor, Sequence[Union[Solution, SolutionTensor]]] | None,
    ) -> List[Solution]:
        if solutions is not None:
            if isinstance(solutions, (list, tuple)):
                raw_list = list(solutions)
            else:
                raw_list = [solutions]
        elif getattr(self.scenario, "noptima", None):
            raw_list = list(self.scenario.noptima)
        elif getattr(self.scenario, "noptimaTensors", None):
            raw_list = list(self.scenario.noptimaTensors)
        elif getattr(self.scenario, "solution", None) is not None:
            raw_list = [self.scenario.solution]
        elif getattr(self.scenario, "solutionTensor", None) is not None:
            raw_list = [self.scenario.solutionTensor]
        else:
            raise ValueError(
                "No evaluated solutions found on `scenario`. Pass `solutions` explicitly or evaluate the scenario first."
            )

        if len(raw_list) == 0:
            raise ValueError("Empty solutions list provided to MGA4AllExporter.")

        return [_coerce_scalar_solution(self.scenario, s) for s in raw_list]

    # --------------------------------------------------------------------------
    # 1. Calendar & Temporal Filtering / Aggregation
    # --------------------------------------------------------------------------

    def _build_full_timestamps(self) -> List[str]:
        intervals_count = int(self.static.intervals_count)
        first_year = int(self.static.first_year)
        year_first_t = np.asarray(self.static.year_first_t, dtype=int)

        timestamps: List[str] = [""] * intervals_count

        for y_idx, start_t in enumerate(year_first_t):
            end_t = int(year_first_t[y_idx + 1]) if (y_idx + 1 < len(year_first_t)) else intervals_count
            start_t, end_t = int(start_t), min(end_t, intervals_count)
            if start_t >= end_t:
                continue

            yr = first_year + y_idx
            is_leap = bool(LEAPDAYS and calendar.isleap(yr))
            days_in_months = DAYS_LEAP if is_leap else DAYS_NON_LEAP
            cum_days = np.cumsum(days_in_months)

            n_steps = end_t - start_t
            hours_in_year = np.arange(n_steps, dtype=npfloat) * self.resolution
            day_of_year = (hours_in_year // 24.0).astype(int)
            hour_of_day = np.mod(hours_in_year, 24.0)

            month_idx = np.clip(np.searchsorted(cum_days, day_of_year, side="right"), 0, 11)
            prev_cum_days = np.where(month_idx > 0, cum_days[month_idx - 1], 0)
            day_of_month = day_of_year - prev_cum_days + 1
            months = month_idx + 1

            hrs = np.floor(hour_of_day + 1e-9).astype(int)
            mins = np.round((hour_of_day - hrs) * 60.0).astype(int)

            for i in range(n_steps):
                timestamps[start_t + i] = f"{day_of_month[i]:02d}/{months[i]:02d}/{yr:04d} {hrs[i]:02d}:{mins[i]:02d}"

        return timestamps

    def _resolve_time_mask(self) -> NDArray[np.bool_]:
        intervals_count = int(self.static.intervals_count)
        first_year = int(self.static.first_year)
        final_year = int(self.static.final_year)
        year_of_t = np.asarray(self.static.year_of_interval, dtype=int) + first_year

        yr_spec = self.config.year
        if yr_spec is None or (isinstance(yr_spec, str) and yr_spec.lower() in ("all", "full")):
            return np.ones(intervals_count, dtype=bool)

        if isinstance(yr_spec, str):
            key = yr_spec.lower().strip()
            if key == "first":
                target_years = [first_year]
            elif key == "last":
                target_years = [final_year]
            elif key.isdigit():
                target_years = [int(key)]
            else:
                raise ValueError(f"Unknown year specification '{yr_spec}'. Expected int, 'first', 'last', or 'all'.")
        elif isinstance(yr_spec, int):
            target_years = [yr_spec]
        elif isinstance(yr_spec, Sequence):
            target_years = [int(y) for y in yr_spec]
        else:
            raise ValueError(f"Unsupported year specification: {yr_spec}")

        resolved_years = []
        for y in target_years:
            y_full = (first_year + y) if (0 <= y < self.static.year_count) else y
            if not (first_year <= y_full <= final_year):
                raise ValueError(f"Requested year {y} is outside scenario horizon [{first_year}, {final_year}].")
            resolved_years.append(y_full)

        mask = np.isin(year_of_t, resolved_years)
        if not np.any(mask):
            raise ValueError(f"Year specification '{yr_spec}' matched 0 intervals.")
        return mask

    def _build_timesteps(self) -> Tuple[NDArray[np.bool_], List[str]]:
        mask = self._resolve_time_mask()
        all_ts = self._build_full_timestamps()
        selected_ts = [ts for ts, m in zip(all_ts, mask) if m]
        return mask, selected_ts

    def _filter_power_trace(
        self,
        trace: NDArray[npfloat],
        to_energy: bool = True,
    ) -> NDArray[npfloat]:
        """
        Slices a 1D power trace (in configured power units, e.g. MW) by `_time_mask`
        and optionally converts to energy per interval (`power * resolution`).
        """
        sliced = np.asarray(trace[self._time_mask], dtype=npfloat)
        if to_energy:
            return sliced * self.resolution
        return sliced

    # --------------------------------------------------------------------------
    # 2. Technology Classification & Metadata Resolution
    # --------------------------------------------------------------------------

    def _format_node_name(self, name: str) -> str:
        return name.upper() if self.config.uppercase_nodes else name

    def _canonical_line_endpoints(self, line) -> Tuple[str, str, bool]:
        """
        Returns `(from_node, to_node, is_reversed)` for a major transmission line,
        sorting endpoints alphabetically.
        """
        n_a = self._format_node_name(line.node_start.name)
        n_b = self._format_node_name(line.node_end.name)
        if n_a <= n_b:
            return n_a, n_b, False
        return n_b, n_a, True

    def _build_line_tech_name_map(self) -> Dict[int, str]:
        """
        Maps each major line's `id` to its exported `techs` identifier.
        If `merge_parallel_lines=True`, all lines between `A` and `B` share `{A}_to_{B}`.
        If `merge_parallel_lines=False`, single lines between `A` and `B` use `{A}_to_{B}`,
        while parallel corridors are disambiguated by unit_type suffix (e.g. `{A}_to_{B}_ac_ohl`,
        `{A}_to_{B}_dc_subsea`) and, if still duplicate, by an integer index.
        """
        sol = self.solutions[0]
        lines = list(sol.network.major_lines.values())
        pair_groups: Dict[Tuple[str, str], List[object]] = {}
        for line in lines:
            from_n, to_n, _ = self._canonical_line_endpoints(line)
            pair_groups.setdefault((from_n, to_n), []).append(line)

        name_map: Dict[int, str] = {}
        for (from_n, to_n), group in pair_groups.items():
            base_name = f"{from_n}_to_{to_n}"
            if self.config.merge_parallel_lines or len(group) == 1:
                for line in group:
                    name_map[line.id] = base_name
            else:
                ut_counts: Dict[str, int] = {}
                for line in group:
                    raw_ut = str(getattr(line, "unit_type", "")).lower()
                    ut_suffix = LINE_UNIT_TYPE_SUFFIX.get(
                        raw_ut, raw_ut.removesuffix("_transmission") or f"line_{line.id}"
                    )
                    ut_counts[ut_suffix] = ut_counts.get(ut_suffix, 0) + 1

                ut_seen: Dict[str, int] = {}
                for line in group:
                    raw_ut = str(getattr(line, "unit_type", "")).lower()
                    ut_suffix = LINE_UNIT_TYPE_SUFFIX.get(
                        raw_ut, raw_ut.removesuffix("_transmission") or f"line_{line.id}"
                    )
                    candidate = f"{base_name}_{ut_suffix}"
                    if ut_counts[ut_suffix] > 1:
                        idx = ut_seen.get(ut_suffix, 0) + 1
                        ut_seen[ut_suffix] = idx
                        candidate = f"{candidate}_{idx}"
                    name_map[line.id] = candidate

        return name_map

    def _canonical_line_pair(self, line) -> Tuple[str, str, str, bool]:
        """
        Returns `(from_node, to_node, tech_name, is_reversed)` for a major transmission line.
        """
        from_n, to_n, is_reversed = self._canonical_line_endpoints(line)
        tech_name = self._line_tech_names.get(line.id, f"{from_n}_to_{to_n}")
        return from_n, to_n, tech_name, is_reversed

    def _resolve_asset_tech_meta(self, asset) -> TechMeta:
        obj_cls = getattr(asset, "object_class", "")
        raw_ut = str(getattr(asset, "unit_type", "")).lower()

        if obj_cls == "line":
            from_n, to_n, tech_name, _ = self._canonical_line_pair(asset)
            if tech_name in self.config.custom_tech_meta:
                return self.config.custom_tech_meta[tech_name]
            return TechMeta(
                tech=tech_name,
                color=self.config.transmission_color,
                exclude_from_capacity_plot=False,
                tech_type="transmission",
                display_name="",
                carrier_in="power",
                carrier_out="power",
                from_node=from_n,
                to_node=to_n,
            )

        if self.config.group_by == "subtech":
            subtech = UNIT_TYPE_TO_SUBTECH.get(raw_ut, raw_ut or asset.name.lower())
            if subtech in self.config.custom_tech_meta:
                return self.config.custom_tech_meta[subtech]
            if subtech in DEFAULT_SUBTECH_META:
                return DEFAULT_SUBTECH_META[subtech]

        tech_key = raw_ut if self.config.group_by == "unit_type" else asset.name
        if tech_key in self.config.custom_tech_meta:
            return self.config.custom_tech_meta[tech_key]

        is_hydro_stor = obj_cls == "storage" and raw_ut in ("hydro", "pond", "ror")
        is_rechargeable = obj_cls == "storage" and not is_hydro_stor

        subtech_fallback = UNIT_TYPE_TO_SUBTECH.get(raw_ut, "")
        default_meta = DEFAULT_SUBTECH_META.get(subtech_fallback)
        color = TECH_COLORS.get(
            raw_ut,
            default_meta.color if default_meta else "#6B7280",
        )
        disp = UNIT_TYPE_DISPLAY_NAMES.get(
            raw_ut,
            default_meta.display_name if default_meta else tech_key.replace("_", " ").title(),
        )

        return TechMeta(
            tech=tech_key,
            color=color,
            exclude_from_capacity_plot=False,
            tech_type="storage" if is_rechargeable else "generation",
            display_name=disp,
            carrier_in="power" if is_rechargeable else "",
            carrier_out="power",
        )

    def _get_special_nodal_meta(self, tech_name: str) -> TechMeta:
        if tech_name in self.config.custom_tech_meta:
            return self.config.custom_tech_meta[tech_name]
        return DEFAULT_SUBTECH_META[tech_name]

    # --------------------------------------------------------------------------
    # 3. Table Builders
    # --------------------------------------------------------------------------

    def build_node_coordinates(self) -> pd.DataFrame:
        """
        Builds `node_coordinates.csv` with columns: `nodes,latitude,longitude`.
        Computes geographic centroids from `europe.geojson` (projected via EPSG:3035 for accuracy,
        then converted back to EPSG:4326 latitude/longitude).
        """
        sol = self.solutions[0]
        node_names = sorted(self._format_node_name(n.name) for n in sol.network.nodes.values())

        coords: Dict[str, Tuple[float, float]] = {}
        if self.config.map_path and os.path.exists(self.config.map_path):
            gdf = gpd.read_file(self.config.map_path)
            gdf_proj = gdf.to_crs(epsg=3035)
            cents_4326 = gpd.GeoSeries(gdf_proj.geometry.centroid, crs=3035).to_crs(epsg=4326)
            gdf["c_lat"] = cents_4326.y
            gdf["c_lon"] = cents_4326.x

            for n_name in node_names:
                matches = gdf[gdf["ISO3"].str.lower() == n_name.lower()]
                if not matches.empty:
                    row = matches.iloc[0]
                    coords[n_name] = (float(row["c_lat"]), float(row["c_lon"]))

        rows = []
        for n_name in node_names:
            lat, lon = coords.get(n_name, (0.0, 0.0))
            rows.append(
                {
                    "nodes": n_name,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        return pd.DataFrame(rows)

    def _collect_active_tech_metas(self) -> Dict[str, TechMeta]:
        sol = self.solutions[0]
        accessor = Accessor(sol, units=self.config.units)
        metas: Dict[str, TechMeta] = {}

        for line in accessor.get_assets("major_lines").values():
            meta = self._resolve_asset_tech_meta(line)
            metas[meta.tech] = meta

        for cls in ("generators", "storages"):
            for asset in accessor.get_assets(cls).values():
                meta = self._resolve_asset_tech_meta(asset)
                metas[meta.tech] = meta

        for special in ("curtailment", "demand_power", "lost_load"):
            metas[special] = self._get_special_nodal_meta(special)

        return metas

    def build_tech_parameters(self) -> pd.DataFrame:
        """
        Builds `tech_parameters.csv` with columns:
        `techs,color,exclude_from_capacity_plot,type,display_name,from_node,to_node`
        Transmission lines are listed first (sorted alphabetically), followed by non-transmission technologies.
        """
        metas = self._collect_active_tech_metas()

        trans_keys = sorted(k for k, m in metas.items() if m.tech_type == "transmission")
        other_keys = sorted(k for k, m in metas.items() if m.tech_type != "transmission")

        rows = []
        for k in trans_keys + other_keys:
            m = metas[k]
            rows.append(
                {
                    "techs": m.tech,
                    "color": m.color,
                    "exclude_from_capacity_plot": m.exclude_from_capacity_plot,
                    "type": m.tech_type,
                    "display_name": m.display_name,
                    "from_node": m.from_node,
                    "to_node": m.to_node,
                }
            )
        return pd.DataFrame(rows)

    def _extract_solution_node_tech_data(
        self,
        sol: Solution,
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Union[float, str]]],
        Dict[Tuple[str, str], Dict[str, Union[NDArray[npfloat], str]]],
    ]:
        """
        Extracts capacity records and full time-series flow records for every `(node, tech)` pair
        in a single evaluated `Solution`.
        """
        accessor = Accessor(sol, units=self.config.units)
        factor = accessor.factor
        n_steps = len(self.timesteps)
        zeros = np.zeros(n_steps, dtype=npfloat)

        all_metas = self._collect_active_tech_metas()
        non_trans_metas = {k: m for k, m in all_metas.items() if m.tech_type != "transmission"}
        nodes_sorted = sorted(accessor.get_assets("nodes").values(), key=lambda n: n.name)

        cap_records: Dict[Tuple[str, str], Dict[str, Union[float, str]]] = {}
        flow_records: Dict[Tuple[str, str], Dict[str, Union[NDArray[npfloat], str]]] = {}

        def _ensure_entry(node_name: str, meta: TechMeta):
            key = (node_name, meta.tech)
            if key not in cap_records:
                cap_records[key] = {
                    "carrier_in": meta.carrier_in,
                    "carrier_out": meta.carrier_out,
                    "capacity": 0.0,
                    "capacity_min": 0.0,
                }
                flow_records[key] = {
                    "carrier_out": "power",
                    "carrier_in": "power",
                    "flow_out": zeros.copy(),
                    "flow_in": zeros.copy(),
                }
            return key

        if self.config.include_zero_capacity_techs and self.config.group_by != "asset":
            for node in nodes_sorted:
                n_fmt = self._format_node_name(node.name)
                for tech_key in sorted(non_trans_metas.keys()):
                    _ensure_entry(n_fmt, non_trans_metas[tech_key])

        # 1. Major transmission lines (listed at both endpoint nodes)
        for line in accessor.get_assets("major_lines").values():
            meta = self._resolve_asset_tech_meta(line)
            from_n, to_n, _, is_reversed = self._canonical_line_pair(line)

            cap_total = float(accessor.get_power_capacity(line, errors="coerce")) * factor
            existing, _, min_b, _ = accessor.get_build_power(line, errors="coerce")
            cap_min = float(existing + min_b) * factor

            f_power = accessor.get_transmission_trace(line)
            if is_reversed:
                f_power = -f_power
            eff = float(accessor.get_efficiency(line))

            pos_flow = np.where(f_power > 0.0, f_power, 0.0)
            neg_flow = np.where(f_power < 0.0, -f_power, 0.0)

            # At from_node: sending to to_node is flow_in (leaving from_node); receiving from to_node is flow_out
            from_flow_in = self._filter_power_trace(pos_flow, to_energy=True)
            from_flow_out = self._filter_power_trace(neg_flow * eff, to_energy=True)

            # At to_node: receiving from from_node is flow_out (entering to_node); sending to from_node is flow_in
            to_flow_out = self._filter_power_trace(pos_flow * eff, to_energy=True)
            to_flow_in = self._filter_power_trace(neg_flow, to_energy=True)

            for node_name, f_out, f_in in (
                (from_n, from_flow_out, from_flow_in),
                (to_n, to_flow_out, to_flow_in),
            ):
                key = _ensure_entry(node_name, meta)
                cap_records[key]["capacity"] += cap_total
                cap_records[key]["capacity_min"] += cap_min
                flow_records[key]["flow_out"] += f_out
                flow_records[key]["flow_in"] += f_in

        # 2. Generators
        for gen in accessor.get_assets("generators").values():
            meta = self._resolve_asset_tech_meta(gen)
            node_name = self._format_node_name(gen.node.name)
            key = _ensure_entry(node_name, meta)

            cap_total = float(accessor.get_power_capacity(gen, errors="coerce")) * factor
            existing, _, min_b, _ = accessor.get_build_power(gen, errors="coerce")
            cap_min = float(existing + min_b) * factor

            gen_power = accessor.get_generation_trace(gen)
            gen_energy = self._filter_power_trace(gen_power, to_energy=True)

            cap_records[key]["capacity"] += cap_total
            cap_records[key]["capacity_min"] += cap_min
            flow_records[key]["flow_out"] += gen_energy

        # 3. Storages (Rechargeable storages vs Hydro/Pondage storages)
        for sto in accessor.get_assets("storages").values():
            meta = self._resolve_asset_tech_meta(sto)
            node_name = self._format_node_name(sto.node.name)
            key = _ensure_entry(node_name, meta)

            cap_total = float(accessor.get_power_capacity(sto, errors="coerce")) * factor
            existing, _, min_b, _ = accessor.get_build_power(sto, errors="coerce")
            cap_min = float(existing + min_b) * factor

            disch_power = accessor.get_discharge_trace(sto)
            chrg_power = np.abs(accessor.get_charge_trace(sto))

            disch_energy = self._filter_power_trace(disch_power, to_energy=True)
            chrg_energy = self._filter_power_trace(chrg_power, to_energy=True)

            cap_records[key]["capacity"] += cap_total
            cap_records[key]["capacity_min"] += cap_min
            flow_records[key]["flow_out"] += disch_energy
            if meta.tech_type == "storage":
                flow_records[key]["flow_in"] += chrg_energy

        # 4. Nodal Demand, Curtailment, and Lost Load (Deficit)
        demand_meta = self._get_special_nodal_meta("demand_power")
        curt_meta = self._get_special_nodal_meta("curtailment")
        lost_meta = self._get_special_nodal_meta("lost_load")

        for node in nodes_sorted:
            n_name = self._format_node_name(node.name)

            dem_energy = self._filter_power_trace(accessor.get_demand_trace(node), to_energy=True)
            curt_energy = self._filter_power_trace(accessor.get_curtail_trace(node), to_energy=True)
            lost_energy = self._filter_power_trace(accessor.get_deficit_trace(node), to_energy=True)

            k_dem = _ensure_entry(n_name, demand_meta)
            cap_records[k_dem]["capacity"] = float(np.max(dem_energy) / self.resolution) if len(dem_energy) else 0.0
            cap_records[k_dem]["capacity_min"] = 0.0
            flow_records[k_dem]["flow_in"] = dem_energy

            k_curt = _ensure_entry(n_name, curt_meta)
            cap_records[k_curt]["capacity"] = (
                float(np.max(curt_energy) / self.resolution) if len(curt_energy) else 0.0
            )
            cap_records[k_curt]["capacity_min"] = 0.0
            flow_records[k_curt]["flow_in"] = curt_energy

            k_lost = _ensure_entry(n_name, lost_meta)
            cap_records[k_lost]["capacity"] = (
                float(np.max(lost_energy) / self.resolution) if len(lost_energy) else 0.0
            )
            cap_records[k_lost]["capacity_min"] = 0.0
            flow_records[k_lost]["flow_out"] = lost_energy

        return cap_records, flow_records

    def build_capacity_values(self) -> pd.DataFrame:
        """
        Builds `capacity_values.csv` across all alternatives with columns:
        `alternative,nodes,techs,carrier_in,carrier_out,capacity,capacity_min`
        """
        rows = []
        for alt_idx, sol in enumerate(self.solutions):
            cap_records, _ = self._extract_solution_node_tech_data(sol)
            for (node_name, tech_name) in sorted(cap_records.keys()):
                rec = cap_records[(node_name, tech_name)]
                cap_val = float(rec["capacity"])
                min_val = float(rec["capacity_min"])
                if self.config.decimals is not None:
                    cap_val = round(cap_val, self.config.decimals)
                    min_val = round(min_val, self.config.decimals)
                rows.append(
                    {
                        "alternative": alt_idx,
                        "nodes": node_name,
                        "techs": tech_name,
                        "carrier_in": rec["carrier_in"],
                        "carrier_out": rec["carrier_out"],
                        "capacity": cap_val,
                        "capacity_min": min_val,
                    }
                )
        return pd.DataFrame(rows)

    def build_flow_timeseries(self, alternative: int = 0) -> pl.DataFrame:
        """
        Builds `flow_timeseries_alternative_{alternative}.csv` as a Polars DataFrame with columns:
        `alternative,nodes,techs,carrier_out,carrier_in,timesteps,flow_out,flow_in`
        """
        if not (0 <= alternative < len(self.solutions)):
            raise IndexError(f"Alternative index {alternative} out of range for {len(self.solutions)} solutions.")

        sol = self.solutions[alternative]
        _, flow_records = self._extract_solution_node_tech_data(sol)

        sorted_keys = sorted(flow_records.keys())
        n_groups = len(sorted_keys)
        n_steps = len(self.timesteps)
        total_rows = n_groups * n_steps

        alt_arr = np.full(total_rows, alternative, dtype=np.int32)
        nodes_arr = np.repeat([k[0] for k in sorted_keys], n_steps)
        techs_arr = np.repeat([k[1] for k in sorted_keys], n_steps)
        c_out_arr = np.repeat([flow_records[k]["carrier_out"] for k in sorted_keys], n_steps)
        c_in_arr = np.repeat([flow_records[k]["carrier_in"] for k in sorted_keys], n_steps)
        ts_arr = np.tile(self.timesteps, n_groups)

        flow_out_arr = np.concatenate([flow_records[k]["flow_out"] for k in sorted_keys])
        flow_in_arr = np.concatenate([flow_records[k]["flow_in"] for k in sorted_keys])

        if self.config.decimals is not None:
            flow_out_arr = np.round(flow_out_arr, self.config.decimals)
            flow_in_arr = np.round(flow_in_arr, self.config.decimals)

        return pl.DataFrame(
            {
                "alternative": alt_arr,
                "nodes": nodes_arr,
                "techs": techs_arr,
                "carrier_out": c_out_arr,
                "carrier_in": c_in_arr,
                "timesteps": ts_arr,
                "flow_out": flow_out_arr,
                "flow_in": flow_in_arr,
            }
        )

    # --------------------------------------------------------------------------
    # 4. File Export
    # --------------------------------------------------------------------------

    def export(self, output_dir: str | None = None) -> Dict[str, str]:
        """
        Writes all `mga4all` CSV files (`node_coordinates.csv`, `tech_parameters.csv`,
        `capacity_values.csv`, and `flow_timeseries_alternative_{i}.csv` for each alternative)
        to `output_dir` (defaults to `<scenario.solution_dir>/mga4all`).
        Returns a dictionary mapping file labels to their written filepaths.
        """
        if output_dir is None:
            output_dir = os.path.join(self.scenario.solution_dir, "mga4all")
        os.makedirs(output_dir, exist_ok=True)

        written_files: Dict[str, str] = {}

        coords_path = os.path.join(output_dir, "node_coordinates.csv")
        df_coords = self.build_node_coordinates()
        df_coords.to_csv(coords_path, index=False)
        written_files["node_coordinates"] = coords_path

        techs_path = os.path.join(output_dir, "tech_parameters.csv")
        df_techs = self.build_tech_parameters()
        df_techs.to_csv(techs_path, index=False)
        written_files["tech_parameters"] = techs_path

        cap_path = os.path.join(output_dir, "capacity_values.csv")
        df_cap = self.build_capacity_values()
        df_cap.to_csv(cap_path, index=False)
        written_files["capacity_values"] = cap_path

        for alt_idx in range(len(self.solutions)):
            flow_path = os.path.join(output_dir, f"flow_timeseries_alternative_{alt_idx}.csv")
            df_flow = self.build_flow_timeseries(alternative=alt_idx)
            df_flow.write_csv(flow_path)
            written_files[f"flow_timeseries_alternative_{alt_idx}"] = flow_path

        return written_files


def export_to_mga4all(
    scenario: Scenario,
    solutions: Union[Solution, SolutionTensor, Sequence[Union[Solution, SolutionTensor]]] | None = None,
    output_dir: str | None = None,
    config: MGA4AllConfig | None = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Convenience function to export evaluated FIRM solutions for a `Scenario` into the
    CSV format expected by the third-party `mga4all` tool.
    """
    exporter = MGA4AllExporter(scenario, solutions=solutions, config=config, **kwargs)
    return exporter.export(output_dir=output_dir)
