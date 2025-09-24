from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import float64, int64

if JIT_ENABLED:
    unitcost_spec = [
        ("capex_p", float64),
        ("fom", float64),
        ("vom", float64),
        ("lifetime", int64),
        ("discount_rate", float64),
        ("fuel_cost_mwh", float64),
        ("fuel_cost_h", float64),
        ("capex_e", float64),
        ("transformer_capex", float64),
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
        capex_p: float64,
        fom: float64,
        vom: float64,
        lifetime: int64,
        discount_rate: float64,
        fuel_cost_mwh: float64,
        fuel_cost_h: float64,
        capex_e: float64,
        transformer_capex: float64,
    ) -> None:
        """
        Initialize cost attributes for a Generator, Storage or Line object.

        Parameters:
        -------
        capex_p (float): Power capacity capital cost ($/kW for generator/storage, $/MW-km for line)
        fom (float): Fixed O&M cost ($/kW/year for generator/storage, $/MW/km/year for line)
        vom (float): Variable O&M cost ($/MWh)
        lifetime (int): Asset lifetime in years
        discount_rate (float): Annual discount rate in range [0,1]
        heat_rate_base (float): Constant heat rate term (GJ/h)
        heat_rate_incr (float): First order marginal heat rate term (GJ/MWh)
        fuel (Fuel): Fuel object
        capex_e (float): Energy capacity capital cost ($/kWh for storage only)
        transformer_capex (float): Transformer-specific cost ($/MW)
        length (float): Line length (used for scaling costs and transmission losses)
        """

        self.capex_p = capex_p  # $/kW
        self.capex_e = capex_e  # $/kWh, non-zero for energy storage
        self.fom = fom  # $/kW/year
        self.vom = vom  # $/MWh
        self.lifetime = lifetime  # years
        self.discount_rate = discount_rate  # [0,1]

        self.fuel_cost_mwh = fuel_cost_mwh  # $/MWh = $/GJ * GJ/MWh
        self.fuel_cost_h = fuel_cost_h  # $/h/unit = $/GJ * GJ/h/unit

        self.transformer_capex = transformer_capex  # $/kW, non-zero for lines


if JIT_ENABLED:
    UnitCost_InstanceType = UnitCost.class_type.instance_type
else:
    UnitCost_InstanceType = UnitCost

if JIT_ENABLED:
    ltcosts_spec = [
        ("annualised_build", float64),
        ("fom", float64),
        ("vom", float64),
        ("fuel", float64),
    ]
else:
    ltcosts_spec = []


@jitclass(ltcosts_spec)
class LTCosts:
    def __init__(self):
        self.annualised_build = 0.0
        self.fom = 0.0
        self.vom = 0.0
        self.fuel = 0.0


if JIT_ENABLED:
    LTCosts_InstanceType = LTCosts.class_type.instance_type
else:
    LTCosts_InstanceType = LTCosts
