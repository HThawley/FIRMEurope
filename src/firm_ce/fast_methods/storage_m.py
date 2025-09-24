import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import ltcosts_m
from firm_ce.system.components import Storage, Storage_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType


@njit(fastmath=FASTMATH)
def create_dynamic_copy(
    storage_instance: Storage_InstanceType,
    nodes_typed_dict: DictType(int64, Node_InstanceType),
    lines_typed_dict: DictType(int64, Line_InstanceType),
) -> Storage_InstanceType:
    node_copy = nodes_typed_dict[storage_instance.node.order]
    line_copy = lines_typed_dict[storage_instance.line.order]

    storage_copy = Storage(
        False,
        storage_instance.id,
        storage_instance.order,
        storage_instance.name,
        storage_instance.power_capacity,
        storage_instance.energy_capacity,
        storage_instance.duration,
        storage_instance.charge_efficiency,
        storage_instance.discharge_efficiency,
        storage_instance.max_build_p,
        storage_instance.max_build_e,
        storage_instance.min_build_p,
        storage_instance.min_build_e,
        storage_instance.unit_type,
        storage_instance.near_optimum_check,
        node_copy,
        line_copy,
        storage_instance.group,
        storage_instance.cost,  # This remains static
    )

    storage_copy.candidate_p_x_idx = storage_instance.candidate_p_x_idx
    storage_copy.candidate_e_x_idx = storage_instance.candidate_e_x_idx

    return storage_copy


@njit(fastmath=FASTMATH)
def build_capacity(
    storage_instance: Storage_InstanceType, new_build_capacity: float64, capacity_type: unicode_type
) -> None:
    if storage_instance.static_instance:
        raise_static_modification_error()
    if capacity_type == "power":
        storage_instance.power_capacity += new_build_capacity
        storage_instance.new_build_p += new_build_capacity
        storage_instance.line.capacity += new_build_capacity
        storage_instance.line.new_build += new_build_capacity

        if storage_instance.duration > 0:
            storage_instance.energy_capacity += new_build_capacity * storage_instance.duration
            storage_instance.new_build_e += new_build_capacity * storage_instance.duration

    if capacity_type == "energy":
        if storage_instance.duration == 0:
            storage_instance.energy_capacity += new_build_capacity
            storage_instance.new_build_e += new_build_capacity
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(storage_instance: Storage_InstanceType, intervals_count: int64) -> None:
    if storage_instance.static_instance:
        raise_static_modification_error()
    storage_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    storage_instance.stored_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def initialise_stored_energy(storage_instance: Storage_InstanceType) -> None:
    if storage_instance.static_instance:
        raise_static_modification_error()
    # Make it possible for user to define custom value for initial stored energy in future
    # Stored in final time interval for simplicity. Overwritten upon performing balancing for final time interval.
    storage_instance.stored_energy[-1] = 0.5 * storage_instance.energy_capacity
    return None


@njit(fastmath=FASTMATH)
def set_dispatch_max_t(
    storage_instance: Storage_InstanceType,
    interval: int64,
    resolution: float64,
    merit_order_idx: int64,
    forward_time_flag: boolean,
) -> None:
    if forward_time_flag:
        storage_instance.discharge_max_t = min(
            storage_instance.power_capacity,
            storage_instance.stored_energy[interval - 1] * storage_instance.discharge_efficiency / resolution,
        )
        storage_instance.charge_max_t = min(
            storage_instance.power_capacity,
            (storage_instance.energy_capacity - storage_instance.stored_energy[interval - 1])
            / storage_instance.charge_efficiency
            / resolution,
        )
    else:
        storage_instance.discharge_max_t = min(
            storage_instance.power_capacity,
            (storage_instance.energy_capacity - storage_instance.stored_energy_temp_reverse)
            * storage_instance.discharge_efficiency
            / resolution,
        )
        storage_instance.charge_max_t = min(
            storage_instance.power_capacity,
            storage_instance.stored_energy_temp_reverse / storage_instance.charge_efficiency / resolution,
        )

    if merit_order_idx == 0:
        storage_instance.node.discharge_max_t[0] = storage_instance.discharge_max_t
        storage_instance.node.charge_max_t[0] = storage_instance.charge_max_t
    else:
        storage_instance.node.discharge_max_t[merit_order_idx] = (
            storage_instance.node.discharge_max_t[merit_order_idx - 1] + storage_instance.discharge_max_t
        )
        storage_instance.node.charge_max_t[merit_order_idx] = (
            storage_instance.node.charge_max_t[merit_order_idx - 1] + storage_instance.charge_max_t
        )
    return None


@njit(fastmath=FASTMATH)
def dispatch(storage_instance: Storage_InstanceType, interval: int64, merit_order_idx: int64) -> None:
    if merit_order_idx == 0:
        storage_instance.dispatch_power[interval] = max(
            min(
                storage_instance.node.netload_t - storage_instance.node.flexible_power[interval],
                storage_instance.discharge_max_t,
            ),
            0.0,
        ) + min(
            max(
                storage_instance.node.netload_t - storage_instance.node.flexible_power[interval],
                -storage_instance.charge_max_t,
            ),
            0.0,
        )
    else:
        storage_instance.dispatch_power[interval] = max(
            min(
                storage_instance.node.netload_t
                - storage_instance.node.flexible_power[interval]
                - storage_instance.node.discharge_max_t[merit_order_idx - 1],
                storage_instance.discharge_max_t,
            ),
            0.0,
        ) + min(
            max(
                storage_instance.node.netload_t
                - storage_instance.node.flexible_power[interval]
                + storage_instance.node.charge_max_t[merit_order_idx - 1],
                -storage_instance.charge_max_t,
            ),
            0.0,
        )
    storage_instance.node.storage_power[interval] += storage_instance.dispatch_power[interval]
    return None


@njit(fastmath=FASTMATH)
def update_stored_energy(
    storage_instance: Storage_InstanceType, interval: int64, resolution: float64, forward_time_flag: boolean
) -> None:
    if forward_time_flag:
        storage_instance.stored_energy[interval] = (
            storage_instance.stored_energy[interval - 1]
            - max(storage_instance.dispatch_power[interval], 0) / storage_instance.discharge_efficiency * resolution
            - min(storage_instance.dispatch_power[interval], 0) * storage_instance.charge_efficiency * resolution
        )
    else:
        storage_instance.stored_energy_temp_reverse += (
            max(storage_instance.dispatch_power[interval], 0) / storage_instance.discharge_efficiency * resolution
            + min(storage_instance.dispatch_power[interval], 0) * storage_instance.charge_efficiency * resolution
        )
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_discharge(storage_instance: Storage_InstanceType, interval_resolutions: float64[:]) -> None:
    storage_instance.lt_discharge = sum(np.maximum(storage_instance.dispatch_power, 0) * interval_resolutions)

    storage_instance.line.lt_flows += sum(np.abs(storage_instance.dispatch_power) * interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def calculate_variable_costs(storage_instance: Storage_InstanceType) -> float64:
    ltcosts_m.calculate_vom(storage_instance.lt_costs, storage_instance.lt_discharge, storage_instance.cost)
    ltcosts_m.calculate_fuel(storage_instance.lt_costs, storage_instance.lt_discharge, 0, storage_instance.cost)
    return ltcosts_m.get_variable(storage_instance.lt_costs)


@njit(fastmath=FASTMATH)
def calculate_fixed_costs(storage_instance: Storage_InstanceType, years_float: float64, year_count: int64) -> float64:
    ltcosts_m.calculate_annualised_build(
        storage_instance.lt_costs,
        storage_instance.new_build_e,
        storage_instance.new_build_p,
        0.0,
        storage_instance.cost,
        year_count,
        "storage",
    )
    ltcosts_m.calculate_fom(
        storage_instance.lt_costs, storage_instance.power_capacity, years_float, 0.0, storage_instance.cost, "storage"
    )
    return ltcosts_m.get_fixed(storage_instance.lt_costs)


@njit(fastmath=FASTMATH)
def initialise_deficit_block(storage_instance: Storage_InstanceType, interval: int64) -> None:
    storage_instance.stored_energy_temp_reverse = storage_instance.stored_energy[interval - 1]
    storage_instance.deficit_block_min_storage = storage_instance.stored_energy_temp_reverse
    storage_instance.deficit_block_max_storage = storage_instance.stored_energy_temp_reverse


@njit(fastmath=FASTMATH)
def update_deficit_block_bounds(storage_instance: Storage_InstanceType, stored_energy: float64) -> None:
    storage_instance.deficit_block_min_storage = min(storage_instance.deficit_block_min_storage, stored_energy)
    storage_instance.deficit_block_max_storage = max(storage_instance.deficit_block_max_storage, stored_energy)
    return None


@njit(fastmath=FASTMATH)
def assign_precharging_reserves(storage_instance: Storage_InstanceType) -> None:
    storage_instance.precharge_flag = (
        storage_instance.deficit_block_max_storage - storage_instance.deficit_block_min_storage
        > storage_instance.stored_energy_temp_forward
    )
    if storage_instance.precharge_flag:
        storage_instance.precharge_energy = max(
            storage_instance.stored_energy_temp_reverse - storage_instance.stored_energy_temp_forward, 0.0
        )
    else:
        storage_instance.precharge_energy = 0.0
    storage_instance.trickling_reserves = (
        storage_instance.deficit_block_max_storage - storage_instance.deficit_block_min_storage
    )
    return None


@njit(fastmath=FASTMATH)
def initialise_precharging_flags(storage_instance: Storage_InstanceType, interval: int64) -> None:
    storage_instance.trickling_flag = (
        storage_instance.stored_energy[interval] - storage_instance.trickling_reserves > TOLERANCE
    ) and (storage_instance.precharge_energy < TOLERANCE)
    storage_instance.precharge_flag = storage_instance.precharge_energy > TOLERANCE
    return None


@njit(fastmath=FASTMATH)
def update_precharging_flags(storage_instance: Storage_InstanceType, interval: int64) -> None:
    storage_instance.remaining_trickling_reserves = max(
        storage_instance.stored_energy[interval] - storage_instance.trickling_reserves, 0.0
    )
    storage_instance.trickling_flag = (
        storage_instance.remaining_trickling_reserves > TOLERANCE
    ) and storage_instance.trickling_flag
    storage_instance.precharge_flag = (
        (storage_instance.stored_energy[interval] + TOLERANCE < storage_instance.energy_capacity)
        and (storage_instance.precharge_energy > TOLERANCE)
        and storage_instance.precharge_flag
    )

    return None


@njit(fastmath=FASTMATH)
def set_precharging_max_t(
    storage_instance: Storage_InstanceType, interval: int64, resolution: float64, merit_order_idx: int64
) -> None:
    # Set discharge_max_t for trickle chargers
    if storage_instance.trickling_flag:
        charge_reduction_constraint_power = min(
            storage_instance.remaining_trickling_reserves / storage_instance.charge_efficiency / resolution,
            -min(storage_instance.dispatch_power[interval], 0.0),
        )
        charge_reduction_constraint_energy = (
            charge_reduction_constraint_power * storage_instance.charge_efficiency * resolution
        )
        discharge_increase_constraint_power = min(
            (storage_instance.remaining_trickling_reserves - charge_reduction_constraint_energy)
            * storage_instance.discharge_efficiency
            / resolution,
            storage_instance.power_capacity - max(storage_instance.dispatch_power[interval], 0.0),
        )
        storage_instance.discharge_max_t = charge_reduction_constraint_power + discharge_increase_constraint_power
    else:
        storage_instance.discharge_max_t = 0.0

    # Set charge_max_t for pre-chargers
    if storage_instance.precharge_flag:
        discharge_reduction_constraint_power = min(
            storage_instance.precharge_energy * storage_instance.discharge_efficiency / resolution,
            max(storage_instance.dispatch_power[interval], 0.0),
        )
        discharge_reduction_constraint_energy = (
            discharge_reduction_constraint_power / storage_instance.discharge_efficiency * resolution
        )
        charge_increase_constraint_power = min(
            (storage_instance.precharge_energy - discharge_reduction_constraint_energy)
            / storage_instance.charge_efficiency
            / resolution,
            storage_instance.power_capacity + min(storage_instance.dispatch_power[interval], 0.0),
        )
        storage_instance.charge_max_t = discharge_reduction_constraint_power + charge_increase_constraint_power
    else:
        storage_instance.charge_max_t = 0.0

    # Update nodal dispatch_max_t values
    if merit_order_idx == 0:
        storage_instance.node.discharge_max_t[0] = storage_instance.discharge_max_t
        storage_instance.node.charge_max_t[0] = storage_instance.charge_max_t
    else:
        storage_instance.node.discharge_max_t[merit_order_idx] = (
            storage_instance.node.discharge_max_t[merit_order_idx - 1] + storage_instance.discharge_max_t
        )
        storage_instance.node.charge_max_t[merit_order_idx] = (
            storage_instance.node.charge_max_t[merit_order_idx - 1] + storage_instance.charge_max_t
        )
    return None


@njit(fastmath=FASTMATH)
def calculate_dispatch_energy_update(
    storage_instance: Storage_InstanceType,
    dispatch_power_original: float64,
    dispatch_power_update: float64,
    resolution: float64,
) -> float64:
    dispatch_energy_update = 0.0

    if dispatch_power_original > 0.0:  # If originally discharging
        if dispatch_power_update > 0.0:  # Increase discharging power
            dispatch_energy_update = -dispatch_power_update / storage_instance.discharge_efficiency * resolution
        else:  # Reduce discharging power and increase charging power
            dispatch_energy_update = (
                min(dispatch_power_original, -dispatch_power_update)
                / storage_instance.discharge_efficiency
                * resolution
                - min(dispatch_power_original + dispatch_power_update, 0.0)
                * storage_instance.charge_efficiency
                * resolution
            )

    # If originally charging
    elif dispatch_power_original < 0.0:
        if dispatch_power_update > 0.0:  # Reduce charging power and increase discharging power
            dispatch_energy_update = (
                -min(dispatch_power_original, -dispatch_power_update) * storage_instance.charge_efficiency * resolution
                + min(dispatch_power_original + dispatch_power_update, 0.0)
                / storage_instance.discharge_efficiency
                * resolution
            )
        else:  # Increase charging power
            dispatch_energy_update = -dispatch_power_update * storage_instance.charge_efficiency * resolution

    else:
        if dispatch_power_update > 0:
            dispatch_energy_update = -dispatch_power_update / storage_instance.discharge_efficiency * resolution
        else:
            dispatch_energy_update = -dispatch_power_update * storage_instance.charge_efficiency * resolution

    return dispatch_energy_update


@njit(fastmath=FASTMATH)
def update_precharge_dispatch(
    storage_instance: Storage_InstanceType,
    interval: int64,
    resolution: float64,
    dispatch_power_update: float64,
    precharging_flag: boolean,
    merit_order_idx: int64,
) -> None:
    dispatch_energy_update = calculate_dispatch_energy_update(
        storage_instance, storage_instance.dispatch_power[interval], dispatch_power_update, resolution
    )

    storage_instance.dispatch_power[interval] += dispatch_power_update
    storage_instance.node.storage_power[interval] += dispatch_power_update

    if precharging_flag:
        storage_instance.charge_max_t += dispatch_power_update
        storage_instance.node.charge_max_t[merit_order_idx:] += dispatch_power_update
        storage_instance.node.precharge_fill += dispatch_power_update
        storage_instance.precharge_energy -= dispatch_energy_update
    else:
        storage_instance.discharge_max_t -= dispatch_power_update
        storage_instance.node.discharge_max_t[merit_order_idx:] -= dispatch_power_update
        storage_instance.node.precharge_surplus -= dispatch_power_update
        storage_instance.trickling_reserves += dispatch_energy_update
    return None


@njit(fastmath=FASTMATH)
def calculate_available_dispatch(storage_instance: Storage_InstanceType, interval: int64) -> None:
    storage_instance.remaining_discharge_max_t = (
        storage_instance.discharge_max_t - storage_instance.dispatch_power[interval]
    )
    storage_instance.remaining_charge_max_t = storage_instance.charge_max_t + storage_instance.dispatch_power[interval]
    return None
