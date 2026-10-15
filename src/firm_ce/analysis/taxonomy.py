# type: ignore
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from firm_ce.common.constants import TOLERANCE
from firm_ce.common.typing import npfloat


# ==============================================================================
# 1. Shared Styling & Palettes
# ==============================================================================

TECH_COLORS: Dict[str, str] = {
    # Display / Human-readable names
    "Utility Solar": "#F59E0B",
    "Rooftop Solar": "#FEF08A",
    "Onshore Wind": "#60A5FA",
    "Offshore Wind": "#1D4ED8",
    "Hydro": "#0D9488",
    "Pondage": "#2DD4BF",
    "Run of River": "#99F6E4",
    "Bioenergy": "#16A34A",
    "Biomass": "#15803D",
    "Biogas": "#86EFAC",
    "Geothermal": "#B91C1C",
    "Nuclear": "#E11D48",
    "Fossil Gas": "#78716C",
    "Coal": "#27272A",
    "Battery": "#A855F7",
    "Closed-loop PHES": "#EC4899",
    "Open-loop PHES": "#6366F1",
    "New PHES": "#3730A3",
    "Legacy PHES": "#818CF8",
    "PHES": "#6366F1",
    "AC OHL": "#9A3412",
    "AC OHL (Mountain)": "#EA580C",
    "DC Subsea": "#0E7490",
    "DC Underground": "#1E293B",
    "Curtailment": "#E5E7EB",
    "Storage Losses": "#9CA3AF",
    "Transmission Losses": "#6B7280",
    "Spillage": "#BAE6FD",
    # Unit-type & Composite short keys (for SpaceExplorer)
    "solar": "#F59E0B",
    "pv_fixed": "#D97706",
    "pv_track": "#FBBF24",
    "wind": "#3B82F6",
    "onsw": "#60A5FA",
    "offw": "#1D4ED8",
    "hydro": "#0D9488",
    "ror": "#99F6E4",
    "bio": "#16A34A",
    "biomass": "#15803D",
    "biogas": "#86EFAC",
    "nuke_all": "#E11D48",
    "nuclear": "#E11D48",
    "nuclear_lte": "#FB7185",
    "ccgt": "#78716C",
    "bess": "#A855F7",
    "bess_e": "#9333EA",
    "bess2h": "#C084FC",
    "bess4h": "#7E22CE",
    "bess2h_e": "#C084FC",
    "bess4h_e": "#7E22CE",
    "phes": "#6366F1",
    "phes_e": "#4F46E5",
    "nphes": "#3730A3",
    "nphes_e": "#312E81",
    "legacy_phes": "#818CF8",
    "legacy_phes_e": "#6366F1",
    "rechargable": "#8B5CF6",
    "rechargable_e": "#7C3AED",
    "trans": "#EA580C",
    "hvac": "#9A3412",
    "hvdc": "#0E7490",
    "renewable": "#10B981",
}

TECH_HATCHES: Dict[str, str] = {
    "Curtailment": "xx",
    "Storage Losses": "--",
    "Transmission Losses": "\\\\",
    "Spillage": "oo",
}

# Composite definitions: {composite_name: (tuple_of_component_unit_types, category_kind)}
# category_kind is "node_power", "node_energy", or "line_power"
COMPOSITE_GROUPS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "solar": (("pv_fixed", "pv_track"), "node_power"),
    "wind": (("onsw", "offw"), "node_power"),
    "bio": (("biomass", "biogas"), "node_power"),
    "nuke_all": (("nuclear", "nuclear_lte"), "node_power"),
    "bess": (("bess2h", "bess4h"), "node_power"),
    "bess_e": (("bess2h_e", "bess4h_e"), "node_energy"),
    "legacy_phes": (("clphes", "olphes"), "node_power"),
    "legacy_phes_e": (("clphes_e", "olphes_e"), "node_energy"),
    "phes": (("nphes", "clphes", "olphes"), "node_power"),
    "phes_e": (("nphes_e", "clphes_e", "olphes_e"), "node_energy"),
    "rechargable": (("bess2h", "bess4h", "nphes", "clphes", "olphes"), "node_power"),
    "rechargable_e": (("bess2h_e", "bess4h_e", "nphes_e", "clphes_e", "olphes_e"), "node_energy"),
    "renewable": (("pv_fixed", "pv_track", "onsw", "offw", "hydro", "ror"), "node_power"),
    "hvdc": (("dc_subsea_transmission", "dc_underground_transmission"), "line_power"),
    "hvac": (("ac_ohl_transmission", "ac_ohl_mountain_transmission"), "line_power"),
    "trans": (
        (
            "dc_subsea_transmission",
            "dc_underground_transmission",
            "ac_ohl_transmission",
            "ac_ohl_mountain_transmission",
        ),
        "line_power",
    ),
}


@dataclass
class PlotStyle:
    """Scoped Matplotlib style configuration."""

    dpi: int = 300
    base_fontsize: int = 12
    large_fontsize: int = 14
    small_fontsize: int = 10
    annotation_format: Union[str, Callable[[float], str]] = "{:,.1f}"

    def to_rc(self) -> dict:
        return {
            "font.size": self.base_fontsize,
            "legend.fontsize": self.small_fontsize,
            "legend.title_fontsize": self.small_fontsize,
            "axes.titlesize": self.small_fontsize,
        }

    @staticmethod
    def format_sigfigs(val: float, sigfigs: int = 2) -> str:
        if val == 0 or not np.isfinite(val):
            return "0"
        decimals = sigfigs - int(math.floor(math.log10(abs(val)))) - 1
        rounded_val = round(val, decimals)
        if decimals <= 0:
            return f"{int(rounded_val):,}"
        return f"{rounded_val:,.{decimals}f}"


def get_color(tech: str) -> str:
    base = tech.removesuffix(" (Electrical)").removesuffix(" (Inflows)")
    return TECH_COLORS.get(tech, TECH_COLORS.get(base, "#6B7280"))


def get_hatch(tech: str) -> str | None:
    if tech.endswith(" (Electrical)"):
        return "//"
    if tech.endswith(" (Inflows)"):
        return ".."
    return TECH_HATCHES.get(tech, None)


# ==============================================================================
# 2. TechTaxonomy & Linear Projection Engine
# ==============================================================================


class TechTaxonomy:
    """
    Single source of truth for mapping the ~450 nodal decision variables and existing
    fleet capacities into unit-type and composite columns under both Absolute (GW/GWh)
    and Relative (dimensionless nodal-demand-normalised) parameterisations.
    """

    def __init__(self, scenario, rel_norm: Literal["eligible", "all_nodes"] = "eligible"):
        """
        Parameters
        ----------
        scenario : Scenario
            Loaded FIRM Scenario object.
        rel_norm : {"eligible", "all_nodes"}
            Denominator rule for relative averages:
            - "eligible": divides the sum of nodal relative capacities by the number of
              nodes (or lines) where that unit-type/composite group exists or can be built.
            - "all_nodes": divides all nodal technology sums by total system nodes (e.g. 33),
              guaranteeing strict additivity across all nodal sub-technologies.
        """
        self.scenario = scenario
        self.rel_norm = rel_norm
        self.n_nodes = len(scenario.network.nodes)
        self.n_vars = len(scenario.lower_bounds_abs)

        self._build_projection_matrices()

    def _build_projection_matrices(self) -> None:
        # Track per-unit-type:
        # - x_weights_abs: dict[ut, list[(x_idx, weight_abs)]]
        # - existing_abs: dict[ut, float]
        # - existing_rel: dict[ut, float]
        # - active_entities: dict[ut, set[entity_id]]  (nodes for gen/sto, lines for major_lines)
        # - is_line: dict[ut, bool]
        x_weights_abs = defaultdict(list)
        existing_abs = defaultdict(float)
        existing_rel = defaultdict(float)
        active_entities = defaultdict(set)
        is_line_type = defaultdict(bool)

        # 1. Generators
        for gen in self.scenario.fleet.generators.values():
            ut = gen.unit_type
            denom = gen.node.mean_demand
            if gen.candidate_x_idx != -1 or gen.initial_capacity > TOLERANCE:
                active_entities[ut].add(gen.node.id)
            if gen.candidate_x_idx != -1:
                x_weights_abs[ut].append((gen.candidate_x_idx, 1.0))
            if gen.initial_capacity > TOLERANCE:
                existing_abs[ut] += gen.initial_capacity
                existing_rel[ut] += gen.initial_capacity / denom

        # 2. Storages (Power & Energy)
        for sto in self.scenario.fleet.storages.values():
            ut = sto.unit_type
            ut_e = f"{ut}_e"
            denom = sto.node.mean_demand
            dur = float(sto.duration)

            if sto.candidate_p_x_idx != -1 or sto.initial_power_capacity > TOLERANCE:
                active_entities[ut].add(sto.node.id)
                active_entities[ut_e].add(sto.node.id)

            if sto.candidate_p_x_idx != -1:
                x_weights_abs[ut].append((sto.candidate_p_x_idx, 1.0))
                # Fixed-duration storages (e.g. bess2h, bess4h) derive energy from power x_idx
                if dur > 0:
                    x_weights_abs[ut_e].append((sto.candidate_p_x_idx, dur))

            if sto.candidate_e_x_idx != -1:
                active_entities[ut_e].add(sto.node.id)
                x_weights_abs[ut_e].append((sto.candidate_e_x_idx, 1.0))

            if sto.initial_power_capacity > TOLERANCE:
                existing_abs[ut] += sto.initial_power_capacity
                existing_rel[ut] += sto.initial_power_capacity / denom

            init_e = sto.initial_energy_capacity if sto.initial_energy_capacity > 0 else dur * sto.initial_power_capacity
            if init_e > TOLERANCE:
                existing_abs[ut_e] += init_e
                existing_rel[ut_e] += init_e / denom

        # 3. Major Lines
        for line in self.scenario.network.major_lines.values():
            ut = line.unit_type
            is_line_type[ut] = True
            denom = max(line.node_start.mean_demand, line.node_end.mean_demand)
            if line.candidate_x_idx != -1 or line.initial_capacity > TOLERANCE:
                active_entities[ut].add(line.id)
            if line.candidate_x_idx != -1:
                x_weights_abs[ut].append((line.candidate_x_idx, 1.0))
            if line.initial_capacity > TOLERANCE:
                existing_abs[ut] += line.initial_capacity
                existing_rel[ut] += line.initial_capacity / denom

        # Order unit types deterministically
        base_uts = list(x_weights_abs.keys() | existing_abs.keys())
        # Ensure all components referenced by COMPOSITE_GROUPS exist in base_uts
        for comp_uts, _ in COMPOSITE_GROUPS.values():
            for u in comp_uts:
                if u not in base_uts:
                    base_uts.append(u)

        all_columns = base_uts + [c for c in COMPOSITE_GROUPS.keys() if c not in base_uts]
        self.columns: List[str] = all_columns
        n_cols = len(all_columns)

        # Build projection matrices:
        # P_abs maps X_abs (N, n_vars) -> U_abs (N, n_cols)
        # P_rel maps X_rel (N, n_vars) -> U_rel (N, n_cols)
        self.P_abs = np.zeros((self.n_vars, n_cols), dtype=npfloat)
        self.P_rel = np.zeros((self.n_vars, n_cols), dtype=npfloat)
        self.offset_abs = np.zeros(n_cols, dtype=npfloat)
        self.offset_rel = np.zeros(n_cols, dtype=npfloat)

        def _get_denom_count(entities: set, is_line: bool) -> float:
            if is_line or self.rel_norm == "eligible":
                return float(max(len(entities), 1))
            return float(self.n_nodes)

        # Populate base unit-type columns
        for col_idx, ut in enumerate(base_uts):
            cnt = _get_denom_count(active_entities[ut], is_line_type[ut])
            for x_idx, w in x_weights_abs[ut]:
                self.P_abs[x_idx, col_idx] += w
                self.P_rel[x_idx, col_idx] += w / cnt
            self.offset_abs[col_idx] = existing_abs[ut]
            self.offset_rel[col_idx] = existing_rel[ut] / cnt

        # Populate composite columns directly in the projection matrices so a single
        # BLAS matrix multiply computes both base and composite columns simultaneously!
        for comp_name, (comp_uts, kind) in COMPOSITE_GROUPS.items():
            col_idx = all_columns.index(comp_name)
            is_line = kind == "line_power"
            union_entities = set()
            for u in comp_uts:
                union_entities |= active_entities[u]
            cnt = _get_denom_count(union_entities, is_line)

            for u in comp_uts:
                for x_idx, w in x_weights_abs[u]:
                    self.P_abs[x_idx, col_idx] += w
                    self.P_rel[x_idx, col_idx] += w / cnt
                self.offset_abs[col_idx] += existing_abs[u]
                self.offset_rel[col_idx] += existing_rel[u] / cnt

    def project(
        self,
        x_matrix: NDArray[npfloat],
        meta_cols: NDArray[npfloat] | None = None,
        *,
        input_space: Literal["abs", "rel"],
        output_space: Literal["abs", "rel"],
        include_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Projects an (N, n_vars) decision matrix into the ~35 unit-type & composite columns
        using a single BLAS matrix multiplication.
        """
        x_matrix = np.asarray(x_matrix, dtype=npfloat)
        if x_matrix.ndim == 1:
            x_matrix = x_matrix[np.newaxis, :]

        scaler = self.scenario.abs_rel_scaler.astype(npfloat)

        if output_space == "abs":
            x_in = (x_matrix * scaler) if input_space == "rel" else x_matrix
            projected = x_in @ self.P_abs
            if include_existing:
                projected += self.offset_abs
        elif output_space == "rel":
            x_in = (x_matrix / scaler) if input_space == "abs" else x_matrix
            projected = x_in @ self.P_rel
            if include_existing:
                projected += self.offset_rel
        else:
            raise ValueError(f"Unknown output_space '{output_space}'. Expected 'abs' or 'rel'.")

        if meta_cols is not None:
            meta_cols = np.asarray(meta_cols, dtype=npfloat)
            if meta_cols.ndim == 1:
                meta_cols = meta_cols[np.newaxis, :]
            data = np.hstack([meta_cols[:, :3], projected])
            cols = ["cost", "violations", "fitness"] + self.columns
        else:
            data = projected
            cols = self.columns

        return pd.DataFrame(data, columns=cols)
