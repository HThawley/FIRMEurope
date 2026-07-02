# type: ignore
import calendar
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import LEAPDAYS
from firm_ce.common.typing import npint, npintp, npfloat
from firm_ce.system.scalar.parameters import ScenarioParameters, ScenarioParameters_InstanceType


def determine_interval_parameters(
    first_year: int,
    year_count: int,
    resolution: float,
) -> Tuple[int, NDArray, int]:
    """
    Calculate parameters associated with time intervals, accounting for leap years. The first_year
    and last_year in `config/scenarios.csv` determines whether or not an interval is considered
    a leap year

    Parameters:
    -------
    first_year (int): The first year of the scenario, specified in `config/scenarios.csv`.
    year_count (int): The total number of years in the scenario.
    resolution (float): The time resolution of each interval for the input data [hours/interval].

    Returns:
    -------
    Tuple[int, NDArray, int]: A tuple containing the number of leap days in the scenario,
        a numpy array specifying the first time interval of each year, and the total number
        of time intervals in the scenario.
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
    year-boundary markers already computed in `determine_interval_parameters`.
    Used to index annual resource budgets (e.g. biomass/biogas fuel
    allowances) that reset once per year but are tracked at interval
    resolution in `Solution.operations`.

    Parameters:
    -------
    year_first_t (NDArray): First time interval of each year, as returned by
        `determine_interval_parameters` (and stored on `ScenarioParameters`).
    intervals_count (int): Total number of time intervals in the scenario.

    Returns:
    -------
    NDArray: Shape (intervals_count,), dtype npintp. year_of_interval[t] gives
        the zero-indexed year that interval t belongs to.
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
    Takes data required to initialise the ScenarioParameters object, casts values into Numba-compatible
    types, and returns an instance of the ScenarioParameters jitclass. The ScenarioParameters are static
    data referenced by the unit committment model.

    Parameters:
    -------
    scenario_data_dict (Dict[str, str]): A dictionary containing data for a single scenario,
        imported from `config/scenarios.csv`.
    node_count (int): The number of nodes (buses) in the network for the scenario.

    Returns:
    -------
    ScenarioParameters_InstanceType: A static instance of the ScenarioParameters jitclass.
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
