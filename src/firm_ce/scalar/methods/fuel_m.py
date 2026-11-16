# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import raise_static_modification_error, raise_getting_unloaded_data_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import nbfloat, unicode_type, npfloat, nbintp
from firm_ce.scalar.entities.components import Fuel, Fuel_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    fuel_instance: Fuel_InstanceType,
) -> Fuel_InstanceType:
    fuel_copy = Fuel(
        False,
        fuel_instance.id,
        fuel_instance.name,
        fuel_instance.cost,
        fuel_instance.emissions,
    )
    fuel_copy.annual_limit = fuel_instance.annual_limit.copy()
    fuel_copy.remaining_energy = fuel_instance.remaining_energy.copy()
    fuel_copy.data_status = fuel_instance.data_status

    return fuel_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(fuel_instance: Fuel_InstanceType, intervals_count: int) -> None:
    if len(fuel_instance.annual_limit) > 0:
        fuel_instance.remaining_energy = np.zeros(intervals_count, dtype=npfloat)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def load_data(
    fuel_instance: Fuel_InstanceType,
    annual_limit: nbfloat[:],
) -> None:
    fuel_instance.annual_limit = annual_limit
    fuel_instance.data_status = True
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def unload_data(fuel_instance: Fuel_InstanceType) -> None:
    fuel_instance.annual_limit = np.empty((0,), dtype=npfloat)
    fuel_instance.data_status = False
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_data(
    fuel_instance: Fuel_InstanceType,
    data_type: unicode_type,
) -> nbfloat[:]:
    if not fuel_instance.data_status:
        raise_getting_unloaded_data_error()

    if data_type == "annual_limit":
        return fuel_instance.annual_limit
    else:
        raise RuntimeError("Invalid data_type argument for Fuel.get_data(data_type).")
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_annual_limit(
    fuel_instance: Fuel_InstanceType,
    year: nbintp,
) -> None:
    return get_data(fuel_instance, "annual_limit")[year]


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_annual_limits(fuel_instance: Fuel_InstanceType, year: int, first_t: int) -> None:
    if fuel_instance.static_instance:
        raise_static_modification_error()
    if len(fuel_instance.annual_limit) > 0:
        fuel_instance.remaining_energy[first_t - 1] = fuel_instance.annual_limit[year]
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_trickling_reserves(fuel_instance: Fuel_InstanceType) -> None:
    fuel_instance.trickling_reserves = (
        fuel_instance.deficit_block_max_energy - fuel_instance.deficit_block_min_energy
    )
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_deficit_block_bounds(
    fuel_instance: Fuel_InstanceType,
    remaining_energy: nbfloat,
) -> None:
    fuel_instance.deficit_block_min_energy = min(fuel_instance.deficit_block_min_energy, remaining_energy)
    fuel_instance.deficit_block_max_energy = max(fuel_instance.deficit_block_max_energy, remaining_energy)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_deficit_block(
    fuel_instance: Fuel_InstanceType,
    interval: nbintp,
) -> None:
    fuel_instance.remaining_energy_temp_reverse = fuel_instance.remaining_energy[interval - 1]
    fuel_instance.deficit_block_max_energy = fuel_instance.remaining_energy_temp_reverse
    fuel_instance.deficit_block_min_energy = fuel_instance.remaining_energy_temp_reverse


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_precharging_flags(
    fuel_instance: Fuel_InstanceType,
    interval: nbintp,
) -> None:
    fuel_instance.trickling_flag = (
        fuel_instance.remaining_energy[interval] - fuel_instance.trickling_reserves > TOLERANCE
    )
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_precharging_flags(
    fuel_instance: Fuel_InstanceType,
    interval: nbintp,
) -> None:
    fuel_instance.remaining_trickling_reserves = max(
        fuel_instance.remaining_energy[interval] - fuel_instance.trickling_reserves, 0.0
    )
    fuel_instance.trickling_flag = (
        fuel_instance.remaining_trickling_reserves > TOLERANCE
    ) and fuel_instance.trickling_flag

    return None
