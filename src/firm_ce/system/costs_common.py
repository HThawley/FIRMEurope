# type: ignore
from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.helpers import get_annuity_factor
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import nbfloat, npfloat

__all__ = [
    "get_annuity_factor",
    "get_generator_costs",
    "get_storage_costs",
    "get_line_costs",
    "apportion_new_phes",
]


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_generator_costs(gen, res, years_float):
    annual_build = 1e6 * (gen.cost.capex_p / gen.cost.annuity_factor)
    fom = 1e6 * gen.cost.fom
    vom = res * 1e3 * gen.cost.vom / years_float
    # fuel_cost_h currently 0 for all generators
    fuel = res * 1e3 * gen.cost.fuel_cost_mwh / years_float
    return annual_build, fom, vom, fuel


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_storage_costs(sto, res, years_float):
    annual_build_p = 1e6 * (sto.cost.capex_p / sto.cost.annuity_factor)
    annual_build_e = 1e6 * (sto.cost.capex_e / sto.cost.annuity_factor)
    fom = 1e6 * sto.cost.fom
    vom = res * 1e3 * sto.cost.vom / years_float
    return annual_build_p, annual_build_e, fom, vom


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_line_costs(line, res, years_float):
    annual_build = 1e6 * (
        (line.length * line.cost.capex_p / line.cost.annuity_factor)
        + (line.cost.transformer_capex / line.cost.annuity_factor)
    )
    fom = 1e6 * (line.length * line.cost.fom)
    vom = res * 1e3 * line.cost.vom / years_float
    return annual_build, fom, vom


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def apportion_new_phes(
    tot_p: nbfloat,
    tot_e: nbfloat,
    e_aa_max: nbfloat,
) -> tuple[nbfloat, nbfloat, nbfloat, nbfloat]:
    """
    Allocates new-build PHES energy to Class AA first (up to e_aa_max), with residual
    to Class E. Apportions power strictly at equal storage duration.
    """
    e_aa = min(tot_e, e_aa_max)
    e_e = max(npfloat(0.0), tot_e - e_aa)

    if tot_e > npfloat(0.0):
        aa_frac = e_aa / tot_e
        p_aa = tot_p * aa_frac
        p_e = tot_p - p_aa
    else:
        if e_aa_max > npfloat(0.0):
            p_aa, p_e = tot_p, npfloat(0.0)
        else:
            p_aa, p_e = npfloat(0.0), tot_p

    return p_aa, e_aa, p_e, e_e
