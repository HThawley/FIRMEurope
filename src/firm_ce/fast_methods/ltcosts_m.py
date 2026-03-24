# type: ignore
from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import float64, unicode_type
from firm_ce.system.costs import LTCosts_InstanceType, UnitCost_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_total(ltcosts_instance: LTCosts_InstanceType) -> float64:
    return (
        ltcosts_instance.annualised_build_p
        + ltcosts_instance.annualised_build_e
        + ltcosts_instance.fom
        + ltcosts_instance.vom
        + ltcosts_instance.fuel
    )


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_variable(ltcosts_instance: LTCosts_InstanceType) -> float64:
    return ltcosts_instance.vom + ltcosts_instance.fuel


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_fixed(ltcosts_instance: LTCosts_InstanceType) -> float64:
    return ltcosts_instance.annualised_build_p + ltcosts_instance.annualised_build_e + ltcosts_instance.fom


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_annuity_factor(discount_rate: float64, lifetime: float64) -> float64:
    return (1 - (1 + discount_rate) ** (-1 * lifetime)) / discount_rate


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_annualised_build_power(
    ltcosts_instance: LTCosts_InstanceType,
    power_capacity: float64,
    line_length: float64,
    unit_costs: UnitCost_InstanceType,
    asset_type: unicode_type,
) -> None:
    annuity_factor = get_annuity_factor(unit_costs.discount_rate, unit_costs.lifetime)
    if asset_type == "generator":
        ltcosts_instance.annualised_build_p = (
            (power_capacity * 1e6 * unit_costs.capex_p)
            / annuity_factor
            if annuity_factor > 1e-6
            else 0
        )
    elif asset_type == "storage":
        ltcosts_instance.annualised_build_p = (
            (power_capacity * 1e6 * unit_costs.capex_p)
            / annuity_factor
            if annuity_factor > 1e-6
            else 0
        )
    elif asset_type == "line":
        ltcosts_instance.annualised_build_p = (
            (power_capacity * 1e3 * line_length * unit_costs.capex_p
             + power_capacity * 1e3 * unit_costs.transformer_capex)
            / annuity_factor
            if annuity_factor > 1e-6
            else 0
        )
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_annualised_build_energy(
    ltcosts_instance: LTCosts_InstanceType,
    energy_capacity: float64,
    unit_costs: UnitCost_InstanceType,
    asset_type: unicode_type,
) -> None:
    annuity_factor = get_annuity_factor(unit_costs.discount_rate, unit_costs.lifetime)
    if asset_type == "storage":
        ltcosts_instance.annualised_build_e = (
            (energy_capacity * 1e6 * unit_costs.capex_e)
            / annuity_factor
            if annuity_factor > 1e-6
            else 0
        )

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lcoe(
    ltcosts_instance: LTCosts_InstanceType,
    generation: float64,
    years_float: float64,
) -> float64:
    total_annual_cost = get_total(ltcosts_instance)

    if generation > 1e-6:
        return total_annual_cost * years_float / generation  # discounting factors cancel out
    return 0.0


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fom(
    ltcosts_instance: LTCosts_InstanceType,
    power_capacity: float64,
    line_length: float64,
    unit_costs: UnitCost_InstanceType,
    asset_type: unicode_type,
) -> None:
    if asset_type == "generator" or asset_type == "storage":
        ltcosts_instance.fom = power_capacity * 1e6 * unit_costs.fom
    elif asset_type == "line":
        ltcosts_instance.fom = power_capacity * 1e3 * line_length * unit_costs.fom
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_vom(
    ltcosts_instance: LTCosts_InstanceType,
    generation: float64,
    year_float: float64,
    unit_costs: UnitCost_InstanceType,
) -> None:
    ltcosts_instance.vom = generation * 1e3 * unit_costs.vom / year_float
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fuel(
    ltcosts_instance: LTCosts_InstanceType,
    generation: float64,
    year_float: float64,
    unit_hours: float64,
    unit_costs: UnitCost_InstanceType,
) -> None:
    ltcosts_instance.fuel = (generation * 1e3 * unit_costs.fuel_cost_mwh + unit_hours * unit_costs.fuel_cost_h) / year_float
    return None
