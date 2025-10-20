# type: ignore
import calendar
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.system.parameters import ScenarioParameters, ScenarioParameters_InstanceType


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
    year_first_t = np.zeros(year_count, dtype=np.int64)

    leap_days = 0
    for i in range(year_count):
        year = first_year + i
        first_t = i * (8760 // resolution)

        leap_days_so_far = calendar.leapdays(first_year, year)

        leap_adjust = leap_days_so_far * (24 // resolution)
        year_first_t[i] = first_t + leap_adjust

        leap_days += calendar.leapdays(year, year + 1)

    hours_total = year_count * 8760 + leap_days * 24
    intervals_count = int(hours_total // resolution)

    return leap_days, year_first_t, intervals_count


def construct_ScenarioParameters_object(
    scenario_data_dict: Dict[str, str],
    node_count: int,
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
    final_year = int(scenario_data_dict.get("finalyear", 0))
    year_count = final_year - first_year + 1
    leap_year_count, year_first_t, intervals_count = determine_interval_parameters(
        first_year,
        year_count,
        resolution,
    )

    return ScenarioParameters(
        resolution,
        allowance,
        first_year,
        final_year,
        year_count,
        leap_year_count,
        year_first_t,
        intervals_count,
        node_count,
    )
