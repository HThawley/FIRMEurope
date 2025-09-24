import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import FASTMATH
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64
from firm_ce.fast_methods import node_m
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Node_InstanceType


@njit(fastmath=FASTMATH)
def get_year_t_boundaries(static_instance: ScenarioParameters_InstanceType, year: int64) -> int64[:]:
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
    static_instance: ScenarioParameters_InstanceType, nodes_typed_dict: DictType(int64, Node_InstanceType)
) -> None:
    for year in range(static_instance.year_count):
        first_t, last_t = get_year_t_boundaries(static_instance, year)
        for node in nodes_typed_dict.values():
            static_instance.year_energy_demand[year] += (
                sum(node_m.get_data(node, "trace")[first_t:last_t]) * static_instance.resolution
            )
    return None


@njit(fastmath=FASTMATH)
def unset_year_energy_demand(static_instance: ScenarioParameters_InstanceType) -> None:
    static_instance.year_energy_demand = np.zeros(static_instance.year_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def check_reliability_constraint(
    static_instance: ScenarioParameters_InstanceType, year: int64, year_unserved_energy: float64
) -> boolean:
    return (year_unserved_energy / static_instance.year_energy_demand[year]) <= static_instance.allowance


@njit(fastmath=FASTMATH)
def set_block_resolutions(static_instance: ScenarioParameters_InstanceType, block_durations: int64[:]) -> None:
    static_instance.interval_resolutions = block_durations * static_instance.resolution
    return None


def get_block_intervals(block_lengths: NDArray[np.int64]):
    block_final_intervals = np.cumsum(block_lengths, dtype=np.int64)
    block_first_intervals = np.concatenate(([0], block_final_intervals[:-1]))
    return block_first_intervals, block_final_intervals
