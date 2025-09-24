from typing import Union

import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import ltcosts_m, node_m
from firm_ce.system.components import Generator, Generator_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType


@njit(fastmath=FASTMATH)
def create_dynamic_copy(
    generator_instance: Generator_InstanceType,
    nodes_typed_dict: DictType(int64, Node_InstanceType),
    lines_typed_dict: DictType(int64, Line_InstanceType),
) -> Generator_InstanceType:
    node_copy = nodes_typed_dict[generator_instance.node.order]
    line_copy = lines_typed_dict[generator_instance.line.order]

    generator_copy = Generator(
        False,
        generator_instance.id,
        generator_instance.order,
        generator_instance.name,
        generator_instance.unit_size,
        generator_instance.max_build,
        generator_instance.min_build,
        generator_instance.capacity,
        generator_instance.unit_type,
        generator_instance.near_optimum_check,
        node_copy,
        generator_instance.fuel,  # This remains static
        line_copy,
        generator_instance.group,
        generator_instance.cost,  # This remains static
    )
    generator_copy.data_status = generator_instance.data_status
    generator_copy.data = generator_instance.data  # This remains static
    generator_copy.annual_constraints_data = generator_instance.annual_constraints_data  # This remains static
    generator_copy.candidate_x_idx = generator_instance.candidate_x_idx
    generator_copy.lt_generation = generator_instance.lt_generation

    return generator_copy


@njit(fastmath=FASTMATH)
def build_capacity(
    generator_instance: Generator_InstanceType, new_build_power_capacity: float64, interval_resolutions: float64[:]
) -> None:
    if generator_instance.static_instance:
        raise_static_modification_error()
    generator_instance.capacity += new_build_power_capacity
    generator_instance.new_build += new_build_power_capacity
    generator_instance.line.capacity += new_build_power_capacity
    generator_instance.line.new_build += new_build_power_capacity

    update_residual_load(generator_instance, new_build_power_capacity, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def load_data(
    generator_instance: Generator_InstanceType,
    generation_trace: float64[:],
    annual_constraints: float64[:],
    interval_resolutions: float64[:],
) -> None:
    generator_instance.data_status = "loaded"
    generator_instance.data = generation_trace
    generator_instance.annual_constraints_data = annual_constraints

    update_residual_load(generator_instance, generator_instance.initial_capacity, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def unload_data(generator_instance: Generator_InstanceType) -> None:
    generator_instance.data_status = "unloaded"
    generator_instance.data = np.empty((0,), dtype=np.float64)
    generator_instance.annual_constraints_data = np.empty((0,), dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def get_data(generator_instance: Generator_InstanceType, data_type: unicode_type) -> Union[float64[:], None]:
    if generator_instance.data_status == "unloaded":
        raise_getting_unloaded_data_error()

    if data_type == "annual_constraints_data":
        return generator_instance.annual_constraints_data
    elif data_type == "trace":
        return generator_instance.data
    else:
        raise RuntimeError("Invalid data_type argument for Generator.get_data(data_type).")
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(generator_instance: Generator_InstanceType, intervals_count: int64) -> None:
    if generator_instance.static_instance:
        raise_static_modification_error()
    generator_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    if len(get_data(generator_instance, "annual_constraints_data")) > 0:
        generator_instance.remaining_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def update_residual_load(
    generator_instance: Generator_InstanceType, added_capacity: float64, interval_resolutions: float64[:]
) -> None:
    if get_data(generator_instance, "trace").shape[0] > 0 and added_capacity > 0.0:
        new_trace = get_data(generator_instance, "trace") * added_capacity
        node_m.get_data(generator_instance.node, "residual_load")[:] -= new_trace
        update_lt_generation(generator_instance, new_trace, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def update_lt_generation(
    generator_instance: Generator_InstanceType, generation_trace: float64[:], interval_resolutions: float64[:]
) -> None:
    generator_instance.lt_generation += sum(generation_trace * interval_resolutions)
    generator_instance.line.lt_flows += generator_instance.lt_generation
    return None


@njit(fastmath=FASTMATH)
def initialise_annual_limit(
    generator_instance: Generator_InstanceType,
    year: int64,
    first_t: int64,
) -> None:
    if len(get_data(generator_instance, "annual_constraints_data")) > 0:
        generator_instance.remaining_energy[first_t - 1] = get_data(generator_instance, "annual_constraints_data")[year]
    return None


@njit(fastmath=FASTMATH)
def get_annual_limit(
    generator_instance: Generator_InstanceType,
    year: int64,
) -> None:
    return get_data(generator_instance, "annual_constraints_data")[year]


@njit(fastmath=FASTMATH)
def check_unit_type(generator_instance: Generator_InstanceType, unit_type: unicode_type) -> boolean:
    return generator_instance.unit_type == unit_type


@njit(fastmath=FASTMATH)
def set_flexible_max_t(
    generator_instance: Generator_InstanceType,
    interval: int64,
    resolution: float64,
    merit_order_idx: int64,
    forward_time_flag: boolean,
) -> None:
    if forward_time_flag:
        generator_instance.flexible_max_t = min(
            generator_instance.capacity, generator_instance.remaining_energy[interval - 1] / resolution
        )
    else:
        generator_instance.flexible_max_t = min(
            generator_instance.capacity, generator_instance.remaining_energy_temp_reverse / resolution
        )

    if merit_order_idx == 0:
        generator_instance.node.flexible_max_t[0] = generator_instance.flexible_max_t
    else:
        generator_instance.node.flexible_max_t[merit_order_idx] = (
            generator_instance.node.flexible_max_t[merit_order_idx - 1] + generator_instance.flexible_max_t
        )
    return None


@njit(fastmath=FASTMATH)
def dispatch(generator_instance: Generator_InstanceType, interval: int64, merit_order_idx: int64) -> None:
    if merit_order_idx == 0:
        generator_instance.dispatch_power[interval] = min(
            max(generator_instance.node.netload_t - generator_instance.node.storage_power[interval], 0.0),
            generator_instance.flexible_max_t,
        )
    else:
        generator_instance.dispatch_power[interval] = min(
            max(
                generator_instance.node.netload_t
                - generator_instance.node.storage_power[interval]
                - generator_instance.node.flexible_max_t[merit_order_idx - 1],
                0.0,
            ),
            generator_instance.flexible_max_t,
        )
    generator_instance.node.flexible_power[interval] += generator_instance.dispatch_power[interval]
    return None


@njit(fastmath=FASTMATH)
def update_remaining_energy(
    generator_instance: Generator_InstanceType,
    interval: int64,
    resolution: float64,
    forward_time_flag: boolean,
    previous_year_flag: boolean,
) -> None:
    if forward_time_flag:
        generator_instance.remaining_energy[interval] = (
            generator_instance.remaining_energy[interval - 1] - generator_instance.dispatch_power[interval] * resolution
        )

    else:
        if previous_year_flag:
            generator_instance.remaining_energy_temp_reverse = (
                generator_instance.remaining_energy[interval - 1]
                - generator_instance.dispatch_power[interval] * resolution
            )
        else:
            generator_instance.remaining_energy_temp_reverse -= generator_instance.dispatch_power[interval] * resolution
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_generation(generator_instance: Generator_InstanceType, interval_resolutions: float64[:]) -> None:
    update_lt_generation(generator_instance, generator_instance.dispatch_power, interval_resolutions)
    generator_instance.unit_lt_hours = sum(
        np.ceil(generator_instance.dispatch_power / generator_instance.unit_size) * interval_resolutions
    )
    return None


@njit(fastmath=FASTMATH)
def calculate_variable_costs(generator_instance: Generator_InstanceType) -> float64:
    ltcosts_m.calculate_vom(generator_instance.lt_costs, generator_instance.lt_generation, generator_instance.cost)
    ltcosts_m.calculate_fuel(
        generator_instance.lt_costs,
        generator_instance.lt_generation,
        generator_instance.unit_lt_hours,
        generator_instance.cost,
    )
    return ltcosts_m.get_variable(generator_instance.lt_costs)


@njit(fastmath=FASTMATH)
def calculate_fixed_costs(
    generator_instance: Generator_InstanceType, years_float: float64, year_count: int64
) -> float64:
    ltcosts_m.calculate_annualised_build(
        generator_instance.lt_costs,
        0.0,
        generator_instance.new_build,
        0.0,
        generator_instance.cost,
        year_count,
        "generator",
    )
    ltcosts_m.calculate_fom(
        generator_instance.lt_costs, generator_instance.capacity, years_float, 0.0, generator_instance.cost, "generator"
    )
    return ltcosts_m.get_fixed(generator_instance.lt_costs)


@njit(fastmath=FASTMATH)
def initialise_deficit_block(generator_instance: Generator_InstanceType, interval: int64) -> None:
    generator_instance.remaining_energy_temp_reverse = generator_instance.remaining_energy[interval - 1]
    generator_instance.deficit_block_max_energy = generator_instance.remaining_energy_temp_reverse
    generator_instance.deficit_block_min_energy = generator_instance.remaining_energy_temp_reverse


@njit(fastmath=FASTMATH)
def update_deficit_block_bounds(generator_instance: Generator_InstanceType, remaining_energy: float64) -> None:
    generator_instance.deficit_block_min_energy = min(generator_instance.deficit_block_min_energy, remaining_energy)
    generator_instance.deficit_block_max_energy = max(generator_instance.deficit_block_max_energy, remaining_energy)
    return None


@njit(fastmath=FASTMATH)
def initialise_precharging_flags(generator_instance: Generator_InstanceType, interval: int64) -> None:
    generator_instance.trickling_flag = (
        generator_instance.remaining_energy[interval] - generator_instance.trickling_reserves > TOLERANCE
    )
    return None


@njit(fastmath=FASTMATH)
def update_precharging_flags(generator_instance: Generator_InstanceType, interval: int64) -> None:
    generator_instance.remaining_trickling_reserves = max(
        generator_instance.remaining_energy[interval] - generator_instance.trickling_reserves, 0.0
    )
    generator_instance.trickling_flag = (
        generator_instance.remaining_trickling_reserves > TOLERANCE
    ) and generator_instance.trickling_flag

    return None


@njit(fastmath=FASTMATH)
def set_precharging_max_t(
    generator_instance: Generator_InstanceType, interval: int64, resolution: float64, merit_order_idx: int64
) -> None:
    if generator_instance.trickling_flag:
        generator_instance.flexible_max_t = min(
            generator_instance.remaining_trickling_reserves / resolution,
            generator_instance.capacity - generator_instance.dispatch_power[interval],
        )
    else:
        generator_instance.flexible_max_t = 0.0

    # Update nodal flexible_max_t values
    if merit_order_idx == 0:
        generator_instance.node.flexible_max_t[0] = generator_instance.flexible_max_t
    else:
        generator_instance.node.flexible_max_t[merit_order_idx] = (
            generator_instance.node.flexible_max_t[merit_order_idx - 1] + generator_instance.flexible_max_t
        )
    return None


@njit(fastmath=FASTMATH)
def update_precharge_dispatch(
    generator_instance: Generator_InstanceType,
    interval: int64,
    resolution: float64,
    dispatch_power_update: float64,
    merit_order_idx: int64,
) -> None:
    dispatch_energy_update = -dispatch_power_update / resolution

    generator_instance.dispatch_power[interval] += dispatch_power_update
    generator_instance.node.flexible_power[interval] += dispatch_power_update

    generator_instance.flexible_max_t -= dispatch_power_update
    generator_instance.node.flexible_max_t[merit_order_idx:] -= dispatch_power_update
    generator_instance.node.precharge_surplus -= dispatch_power_update
    generator_instance.trickling_reserves += dispatch_energy_update
    return None


@njit(fastmath=FASTMATH)
def assign_trickling_reserves(generator_instance: Generator_InstanceType) -> None:
    generator_instance.trickling_reserves = (
        generator_instance.deficit_block_max_energy - generator_instance.deficit_block_min_energy
    )
    return None
