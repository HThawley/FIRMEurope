# type: ignore
from typing import Optional

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.helpers import get_annuity_factor
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, nbint

if JIT_ENABLED:
    unitcost_spec = [
        ("capex_p", nbfloat),
        ("fom", nbfloat),
        ("vom", nbfloat),
        ("lifetime", nbint),
        ("discount_rate", nbfloat),
        ("annuity_factor", nbfloat),
        ("heat_rate_base", nbfloat),
        ("heat_rate_incr", nbfloat),
        ("fuel_cost_mwh", nbfloat),
        ("fuel_cost_h", nbfloat),
        ("capex_e", nbfloat),
        ("transformer_capex", nbfloat),
    ]
else:
    unitcost_spec = []


@jitclass(unitcost_spec)
class UnitCost:
    """
    Represents cost parameters for a generator, storage, or line object.
    """

    def __init__(
        self,
        capex_p: nbfloat,
        fom: nbfloat,
        vom: nbfloat,
        lifetime: nbint,
        discount_rate: nbfloat,
        heat_rate_base: nbfloat,
        heat_rate_incr: nbfloat,
        fuel_cost: nbfloat,
        capex_e: nbfloat,
        transformer_capex: nbfloat,
    ) -> None:
        self.capex_p = capex_p  # $/kW
        self.capex_e = capex_e  # $/kWh, non-zero for energy storage/reservoir
        self.fom = fom  # $/kW/year
        self.vom = vom  # $/MWh
        self.lifetime = lifetime  # years
        self.discount_rate = discount_rate  # [0,1]
        self.annuity_factor = get_annuity_factor(self.discount_rate, self.lifetime)

        self.heat_rate_base = heat_rate_base
        self.heat_rate_incr = heat_rate_incr

        self.fuel_cost_mwh = fuel_cost * self.heat_rate_incr
        self.fuel_cost_h = fuel_cost * self.heat_rate_base

        self.transformer_capex = transformer_capex  # $/kW, non-zero for lines


if JIT_ENABLED:
    UnitCost_InstanceType = UnitCost.class_type.instance_type
else:
    UnitCost_InstanceType = UnitCost

if JIT_ENABLED:
    ltcosts_spec = [
        ("annualised_build_p", nbfloat),
        ("annualised_build_e", nbfloat),
        ("fom", nbfloat),
        ("vom", nbfloat),
        ("fuel", nbfloat),
    ]
else:
    ltcosts_spec = []


@jitclass(ltcosts_spec)
class LTCosts:
    def __init__(self):
        self.annualised_build_p = 0.0
        self.annualised_build_e = 0.0
        self.fom = 0.0
        self.vom = 0.0
        self.fuel = 0.0


if JIT_ENABLED:
    LTCosts_InstanceType = LTCosts.class_type.instance_type
else:
    LTCosts_InstanceType = LTCosts


def construct_UnitCost_object(
    capex_p: float,
    fom: float,
    vom: float,
    lifetime: int,
    discount_rate: float,
    heat_rate_base: float = 0.0,
    heat_rate_incr: float = 0.0,
    fuel: Optional[object] = None,
    capex_e: float = 0.0,
    transformer_capex: float = 0.0,
) -> UnitCost_InstanceType:
    """
    Initialises a UnitCost jitclass instance from asset cost parameters.
    """
    fuel_cost = fuel.cost if fuel is not None else 0.0
    return UnitCost(
        capex_p,
        fom,
        vom,
        lifetime,
        discount_rate,
        heat_rate_base,
        heat_rate_incr,
        fuel_cost,
        capex_e,
        transformer_capex,
    )
