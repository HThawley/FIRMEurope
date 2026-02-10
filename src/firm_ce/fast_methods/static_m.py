# type: ignore
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba.core.types import UniTuple

from firm_ce.common.constants import FASTMATH
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64
from firm_ce.fast_methods import node_m
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Node_InstanceType


@njit(fastmath=FASTMATH)
def get_year_t_boundaries(
    static_instance: ScenarioParameters_InstanceType,
    year: int64,
) -> UniTuple(int64, 2):
    """
    Get the first and last time interval for a year in the modelling horizon. The first time interval of
    each year is stored in the year_first_t array of the ScenarioParameters instance.

    Parameters:
    -------
    static_instance (ScenarioParameters_InstanceType): An instance of the ScenarioParameters jitclass. All of these
        parameters are static and should not be modified during unit committment.
    year (int64): Index for the year, with indexation starting at the first year in the modelling horizon.

    Returns:
    -------
    UniTuple(int64, 2): A tuple of two int64 values that specify the index of the first (inclusive) and last (exclusive)
        time interval for the year.
    """
    if year < static_instance.year_count - 1:
        last_t = static_instance.year_first_t[year + 1]
    else:
        last_t = static_instance.intervals_count
    return static_instance.year_first_t[year], last_t


@njit(fastmath=FASTMATH)
def set_year_first_block(static_instance: ScenarioParameters_InstanceType, blocks_per_day: int64) -> None:
    static_instance.year_first_t = np.zeros(static_instance.year_count, dtype=np.int64)

    leap_days = 0
    for i in range(static_instance.year_count):
        static_instance.year_first_t[i] = blocks_per_day * (i * 365 + leap_days)

        year = static_instance.first_year + i
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            leap_days += 1
    return None


@njit(fastmath=FASTMATH)
def set_year_energy_demand(
    static_instance: ScenarioParameters_InstanceType,
    nodes_typed_dict: DictType(int64, Node_InstanceType),
) -> None:
    for year in range(static_instance.year_count):
        first_t, last_t = get_year_t_boundaries(static_instance, year)
        for node in nodes_typed_dict.values():
            static_instance.year_energy_demand[year] += (
                sum(node_m.get_data(node, "trace")[first_t:last_t]) * static_instance.resolution
            )
    static_instance.mean_annual_demand = np.mean(static_instance.year_energy_demand)
    return None


@njit(fastmath=FASTMATH)
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
    Attributes modified for ScenarioParameters instance: year_energy_demand.
    """
    static_instance.year_energy_demand = np.zeros(static_instance.year_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def check_reliability_constraint(
    static_instance: ScenarioParameters_InstanceType,
    year: int64,
    year_unserved_energy: float64,
) -> boolean:
    return (year_unserved_energy / static_instance.year_energy_demand[year]) <= static_instance.allowance


@njit(fastmath=FASTMATH)
def set_block_resolutions(
    static_instance: ScenarioParameters_InstanceType,
    block_durations: int64[:],
) -> None:
    """
    Calculates the resolution of each time interval. Allows for variable time interval lengths for future
    simplified balancing methods. The block durations array specifies the number of original time intervals
    forming each block. Simplified balancing method could allow for rapid optimisation to the near-optimal
    space, followed by a polishing step that optimises using the full time series data.

    Parameters:
    -------
    static_instance (ScenarioParameters_InstanceType): An instance of the ScenarioParameters jitclass. All of these
        parameters are static and should not be modified during unit committment.
    block_durations (int64[:]): Array defining the length of each block (i.e., number of original time intervals
        per simplified block).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for ScenarioParameters instance: interval_resolutions [hours per block].
    """
    static_instance.interval_resolutions = block_durations * static_instance.resolution
    return None


def get_block_intervals(
    block_lengths: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Get the first and last time intervals contained within each block. Allows for variable time interval lengths (i.e., blocks)
    for future simplified balancing methods. Block data can then be apportioned into arrays with the same length as the
    original time-series data, allowing for the time-series to be easily plotted as a step function.

    Parameters:
    -------
    block_lengths (NDArray[np.int64]): A 1-dimensional Numpy array with a length equal to the number of blocks in the modelling
        horizon, and values equal to the number of time intervals per block.

    Returns:
    -------
    Tuple[NDArray[np.int64], NDArray[np.int64]]: A tuple of two 1-dimensional Numpy arrays. The first array contains the index for
        the first time interval of each block, and the second array contains the index for the final time interval of each block.
    """
    block_final_intervals = np.cumsum(block_lengths, dtype=np.int64)
    block_first_intervals = np.concatenate(([0], block_final_intervals[:-1]))
    return block_first_intervals, block_final_intervals
