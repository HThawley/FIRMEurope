from dataclasses import dataclass
import math
from typing import Callable, Dict, Union

import numpy as np


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
    "AA PHES": "#3730A3",
    "E PHES": "#4338CA",
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
    "ocgt": "#5A5652",
    "bess": "#A855F7",
    "bess_e": "#9333EA",
    "bess2h": "#C084FC",
    "bess4h": "#7E22CE",
    "bess2h_e": "#C084FC",
    "bess4h_e": "#7E22CE",
    "phes": "#6366F1",
    "phes_e": "#4F46E5",
    "aa_phes": "#3730A3",
    "aa_phes_e": "#312E81",
    "e_phes": "#4338CA",
    "e_phes_e": "#3730A3",
    "new_phes": "#3730A3",
    "new_phes_e": "#312E81",
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
