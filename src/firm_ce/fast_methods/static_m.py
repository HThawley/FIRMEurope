# type: ignore
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba.core.types import UniTuple

from firm_ce.common.constants import FASTMATH, LEAPDAYS, BOUNDSCHECK
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, nbfloat, npfloat, nbint, npint, nbintp, npintp
from firm_ce.fast_methods import node_m
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Node_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_year_t_boundaries(
    static_instance: ScenarioParameters_InstanceType,
    year: nbintp,
) -> UniTuple(nbintp, 2):
    """
    Get the first and last time interval for a year in the modelling horizon. The first time interval of
    each year is stored in the year_first_t array of the ScenarioParameters instance.

    Parameters:
    -------
    static_instance (ScenarioParameters_InstanceType): An instance of the ScenarioParameters jitclass. All of these
        parameters are static and should not be modified during unit committment.
    year (nbintp): Index for the year, with indexation starting at the first year in the modelling horizon.

    Returns:
    -------
    UniTuple(nbintp, 2): A tuple of two nbintp values that specify the index of the first (inclusive) and last (exclusive)
        time interval for the year.
    """
    if year < static_instance.year_count - 1:
        last_t = static_instance.year_first_t[year + 1]
    else:
        last_t = static_instance.intervals_count
    return static_instance.year_first_t[year], last_t


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_year_first_block(static_instance: ScenarioParameters_InstanceType, blocks_per_day: nbint) -> None:
    static_instance.year_first_t = np.zeros(static_instance.year_count, dtype=npintp)

    leap_days = 0
    for i in range(static_instance.year_count):
        static_instance.year_first_t[i] = blocks_per_day * (i * 365 + leap_days)

        year = static_instance.first_year + i
        if LEAPDAYS and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            leap_days += 1
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_year_energy_demand(
    static_instance: ScenarioParameters_InstanceType,
    nodes_typed_dict: DictType(nbintp, Node_InstanceType),
) -> None:
    for year in range(static_instance.year_count):
        first_t, last_t = get_year_t_boundaries(static_instance, year)
        for node in nodes_typed_dict.values():
            static_instance.year_energy_demand[year] += (
                sum(node_m.get_data(node, "trace")[first_t:last_t]) * static_instance.resolution
            )
    static_instance.mean_annual_demand_mwh = np.mean(static_instance.year_energy_demand) * 1000
    static_instance.demand_sum_mwh = np.sum(static_instance.year_energy_demand) * 1000
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def unset_year_energy_demand(
    static_instance: ScenarioParameters_InstanceType,
) -> None:
    """
    Resets the total annual energy demand values to zero.

    Parameters:
    -------
    static_instance (ScenarioParameters_InstanceType): An instance of the ScenarioParameters jitclass. All of these
        parameters are static and should not be modified during unit committment.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for ScenarioParameters instance: year_energy_demand, mean_annual_demand.
    """
    static_instance.year_energy_demand = np.zeros(static_instance.year_count, dtype=npfloat)
    static_instance.mean_annual_demand_mwh = 0.0
    static_instance.demand_sum_mwh = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_reliability_constraint(
    static_instance: ScenarioParameters_InstanceType,
    year: nbintp,
    year_unserved_energy: nbfloat,
) -> boolean:
    return (year_unserved_energy / static_instance.year_energy_demand[year]) <= static_instance.allowance
