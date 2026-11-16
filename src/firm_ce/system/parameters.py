# type: ignore
import calendar
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import JIT_ENABLED, LEAPDAYS
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, nbintp, npfloat, npint, npintp

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


def determine_interval_parameters(
    first_year: int,
    year_count: int,
    resolution: float,
) -> Tuple[int, NDArray, int]:
    """
    Calculate parameters associated with time intervals, accounting for leap years.
    """
    year_first_t = np.zeros(year_count, dtype=npintp)

    leap_days = 0
    for i in range(year_count):
        year = first_year + i
        first_t = i * (8760 // resolution)

        if LEAPDAYS:
            leap_days_so_far = calendar.leapdays(first_year, year)
            leap_adjust = leap_days_so_far * (24 // resolution)
            year_first_t[i] = first_t + leap_adjust
            leap_days += calendar.leapdays(year, year + 1)
        else:
            year_first_t[i] = first_t

    hours_total = year_count * 8760 + leap_days * 24
    intervals_count = int(hours_total // resolution)

    return leap_days, year_first_t, intervals_count


def determine_year_of_interval(
    year_first_t: NDArray,
    intervals_count: int,
) -> NDArray:
    """
    Maps each time interval to its zero-indexed simulation year, using the
    year-boundary markers computed in `determine_interval_parameters`.
    """
    t_indices = np.arange(intervals_count)
    year_of_interval = np.searchsorted(year_first_t, t_indices, side="right") - 1

    return year_of_interval.astype(npintp)


def construct_ScenarioParameters_object(
    scenario_data_dict: Dict[str, str],
    node_count: int,
    limit_timesteps: int = None,
    interval_aggregation: int = 1,
) -> ScenarioParameters_InstanceType:
    """
    Initialises a static ScenarioParameters jitclass instance from scenario config data.
    """
    resolution = float(scenario_data_dict.get("resolution", 0.0))
    allowance = float(scenario_data_dict.get("allowance", 0.0))

    first_year = int(scenario_data_dict.get("firstyear", 0))
    if limit_timesteps is not None:
        intervals_count = limit_timesteps
        year_count = int(intervals_count * resolution // 8759 + 1)
        final_year = first_year + year_count - 1
        leap_year_count, year_first_t, _ = determine_interval_parameters(
            first_year,
            year_count,
            resolution,
        )
        if year_first_t[-1] > limit_timesteps:
            year_first_t = year_first_t[:-1]

    else:
        final_year = int(scenario_data_dict.get("finalyear", 0))
        year_count = final_year - first_year + 1
        leap_year_count, year_first_t, intervals_count = determine_interval_parameters(
            first_year,
            year_count,
            resolution,
        )

    if interval_aggregation > 1:
        resolution = resolution * interval_aggregation
        intervals_count = int(np.ceil(intervals_count / interval_aggregation))
        year_first_t = year_first_t // interval_aggregation

    year_of_interval = determine_year_of_interval(year_first_t, intervals_count)

    return ScenarioParameters(
        npfloat(resolution),
        npfloat(allowance),
        npintp(first_year),
        npintp(final_year),
        npint(year_count),
        npint(leap_year_count),
        npintp(year_first_t),
        npintp(year_of_interval),
        npint(intervals_count),
        npint(node_count),
    )
