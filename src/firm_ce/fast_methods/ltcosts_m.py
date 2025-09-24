from firm_ce.common.constants import FASTMATH
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import float64, int64, unicode_type
from firm_ce.system.costs import LTCosts_InstanceType, UnitCost_InstanceType


@njit(fastmath=FASTMATH)
def get_total(ltcosts_instance: LTCosts_InstanceType) -> float64:
    return ltcosts_instance.annualised_build + ltcosts_instance.fom + ltcosts_instance.vom + ltcosts_instance.fuel


@njit(fastmath=FASTMATH)
def get_variable(ltcosts_instance: LTCosts_InstanceType) -> float64:
    return ltcosts_instance.vom + ltcosts_instance.fuel


@njit(fastmath=FASTMATH)
def get_fixed(ltcosts_instance: LTCosts_InstanceType) -> float64:
    return ltcosts_instance.annualised_build + ltcosts_instance.fom


@njit(fastmath=FASTMATH)
def get_present_value(discount_rate: float64, lifetime: float64) -> float64:
    return (1 - (1 + discount_rate) ** (-1 * lifetime)) / discount_rate


@njit(fastmath=FASTMATH)
def calculate_annualised_build(
    ltcosts_instance: LTCosts_InstanceType,
    energy_capacity: float64,
    power_capacity: float64,
    line_length: float64,
    unit_costs: UnitCost_InstanceType,
    year_count: int64,
    asset_type: unicode_type,
) -> None:
    present_value = get_present_value(unit_costs.discount_rate, unit_costs.lifetime)
    if asset_type == "generator" or asset_type == "storage":
        ltcosts_instance.annualised_build = (
            year_count
            * (energy_capacity * 1e6 * unit_costs.capex_e + power_capacity * 1e6 * unit_costs.capex_p)
            / present_value
            if present_value > 1e-6
            else 0
        )
    elif asset_type == "line":
        ltcosts_instance.annualised_build = (
            year_count
            * (
                power_capacity * 1e3 * line_length * unit_costs.capex_p
                + power_capacity * 1e3 * unit_costs.transformer_capex
            )
            / present_value
            if present_value > 1e-6
            else 0
        )
    return None


@njit(fastmath=FASTMATH)
def calculate_fom(
    ltcosts_instance: LTCosts_InstanceType,
    power_capacity: float64,
    years_float: float64,
    line_length: float64,
    unit_costs: UnitCost_InstanceType,
    asset_type: unicode_type,
) -> None:
    if asset_type == "generator" or asset_type == "storage":
        ltcosts_instance.fom = power_capacity * 1e6 * unit_costs.fom * years_float
    elif asset_type == "line":
        ltcosts_instance.fom = power_capacity * 1e3 * line_length * unit_costs.fom * years_float
    return None


@njit(fastmath=FASTMATH)
def calculate_vom(
    ltcosts_instance: LTCosts_InstanceType, generation: float64, unit_costs: UnitCost_InstanceType
) -> None:
    ltcosts_instance.vom = generation * 1e3 * unit_costs.vom
    return None


@njit(fastmath=FASTMATH)
def calculate_fuel(
    ltcosts_instance: LTCosts_InstanceType, generation: float64, unit_hours: float64, unit_costs: UnitCost_InstanceType
) -> None:
    ltcosts_instance.fuel = generation * 1e3 * unit_costs.fuel_cost_mwh + unit_hours * unit_costs.fuel_cost_h
    return None
