# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, LEAPDAYS
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, nbintp, npfloat

if JIT_ENABLED:
    scenario_parameters_spec = [
        ("resolution", nbfloat),
        ("allowance", nbfloat),
        ("first_year", nbintp),
        ("final_year", nbintp),
        ("year_count", nbintp),
        ("leap_year_count", nbintp),
        ("year_first_t", nbintp[:]),
        ("year_of_interval", nbintp[:]),
        ("intervals_count", nbintp),
        ("node_count", nbintp),
        ("fom_scalar", nbfloat),
        ("year_float", nbfloat),
        ("year_energy_demand", nbfloat[:]),
        ("mean_annual_demand_mwh", nbfloat),
        ("demand_sum_mwh", nbfloat),
    ]
else:
    scenario_parameters_spec = []


@jitclass(scenario_parameters_spec)
class ScenarioParameters:
    def __init__(
        self,
        resolution: nbfloat,
        allowance: nbfloat,
        first_year: nbintp,
        final_year: nbintp,
        year_count: nbintp,
        leap_year_count: nbintp,
        year_first_t: nbintp[:],
        year_of_interval: nbintp[:],
        intervals_count: nbintp,
        node_count: nbintp,
    ):

        self.resolution = resolution  # length of time interval in hours
        self.allowance = allowance  # % annual demand allowed as unserved energy
        self.first_year = first_year  # YYYY
        self.final_year = final_year  # YYYY
        self.year_count = year_count
        self.leap_year_count = leap_year_count if LEAPDAYS else 0
        self.year_first_t = year_first_t
        self.year_of_interval = year_of_interval
        self.intervals_count = intervals_count
        self.node_count = node_count
        if LEAPDAYS:
            self.fom_scalar = (
                year_count + leap_year_count / 365
            ) / year_count  # Scale average annual fom to account for leap days for PLEXOS consistency
        else:
            self.fom_scalar = 1.0
        self.year_float = self.year_count * self.fom_scalar
        self.year_energy_demand = np.zeros(self.year_count, dtype=npfloat)
        self.mean_annual_demand_mwh = 0.0
        self.demand_sum_mwh = 0.0


if JIT_ENABLED:
    ScenarioParameters_InstanceType = ScenarioParameters.class_type.instance_type
else:
    ScenarioParameters_InstanceType = ScenarioParameters
