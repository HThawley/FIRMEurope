# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, nbfloat, npfloat, nbint, nbintp, unicode_type
from firm_ce.scalar.entities.components import Storage, Storage_InstanceType
from firm_ce.scalar.entities.topology import Line_InstanceType, Node_InstanceType
from firm_ce.scalar.methods import ltcosts_m


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    storage_instance: Storage_InstanceType,
    nodes_typed_dict: DictType(nbintp, Node_InstanceType),
    lines_typed_dict: DictType(nbintp, Line_InstanceType),
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
        storage_instance.chargeable,
        storage_instance.inflows,
        storage_instance.charge_efficiency,
        storage_instance.discharge_efficiency,
        storage_instance.max_build_p,
        storage_instance.max_build_e,
        storage_instance.min_build_p,
        storage_instance.min_build_e,
        storage_instance.unit_type,
        node_copy,
        line_copy,
        storage_instance.cost,  # This remains static
    )
    storage_copy.unit_type_idx = storage_instance.unit_type_idx

    if storage_instance.inflows:
        storage_copy.data_status = storage_instance.data_status
        storage_copy.data = storage_instance.data
    storage_copy.lt_generation = storage_instance.lt_generation

    storage_copy.candidate_p_x_idx = storage_instance.candidate_p_x_idx
    storage_copy.candidate_e_x_idx = storage_instance.candidate_e_x_idx

    return storage_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacity_abs(
    storage_instance: Storage_InstanceType,
    new_build_capacity: nbfloat,
    capacity_type: unicode_type,
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

    if capacity_type == "energy" and storage_instance.duration == 0:
        storage_instance.energy_capacity += new_build_capacity
        storage_instance.new_build_e += new_build_capacity
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacity(
    storage_instance: Storage_InstanceType,
    x_val_capacity: nbfloat,
    capacity_type: unicode_type,
) -> None:
    new_build_capacity = x_val_capacity * storage_instance.node.relative_scaler
    build_capacity_abs(storage_instance, new_build_capacity, capacity_type)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def load_data(
    storage_instance: Storage_InstanceType,
    inflow_trace: nbfloat[:],
) -> None:
    """
    Load the inflow trace data to the Storage instance. This is done before solving a Scenario.
    """
    storage_instance.data = inflow_trace
    storage_instance.data_status = True

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def unload_data(storage_instance: Storage_InstanceType) -> None:
    """
    Unload the inflow data from the Storage instance. This is done after solving a Scenario to reduce memory usage.
    """
    storage_instance.data = np.empty((0,), dtype=npfloat)
    storage_instance.data_status = False
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_data(
    storage_instance: Storage_InstanceType,
    data_type: unicode_type,
) -> nbfloat[:]:
    """
    Gets the specified data_type from the Storage instance.
    """
    if not storage_instance.data_status:
        raise_getting_unloaded_data_error()

    if data_type == "inflow":
        return storage_instance.data
    else:
        raise RuntimeError("Invalid data_type argument for Storage.get_data(data_type).")


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(
    storage_instance: Storage_InstanceType,
    intervals_count: nbint,
) -> None:
    """
    Memory associated with endogenous time-series data for a Storage system is only allocated after a dynamic copy of
    the Storage instance is created. This is to minimise memory usage of the static instances.
    """
    if storage_instance.static_instance:
        raise_static_modification_error()
    storage_instance.dispatch_power = np.zeros(intervals_count, dtype=npfloat)
    storage_instance.stored_energy = np.zeros(intervals_count, dtype=npfloat)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_stored_energy(
    storage_instance: Storage_InstanceType,
) -> None:
    """
    Initialise the stored energy for a Storage system. Called at the start of the modelling period.
    """
    if storage_instance.static_instance:
        raise_static_modification_error()
    storage_instance.stored_energy[-1] = 0.5 * storage_instance.energy_capacity
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_dispatch_max_t(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    merit_order_idx: nbintp,
    forward_time_flag: boolean,
) -> None:
    inflow_energy = 0.0
    if storage_instance.inflows:
        inflow_energy = storage_instance.data[interval]

    if forward_time_flag:
        storage_instance.discharge_max_t = min(
            storage_instance.power_capacity,
            (storage_instance.stored_energy[interval - 1] + inflow_energy)
            * storage_instance.discharge_efficiency
            / resolution,
        )
        if storage_instance.chargeable:
            storage_instance.charge_max_t = min(
                storage_instance.power_capacity,
                max(
                    storage_instance.energy_capacity
                    - storage_instance.stored_energy[interval - 1]
                    - inflow_energy,
                    0.0,
                )
                / storage_instance.charge_efficiency
                / resolution,
            )
        else:
            storage_instance.charge_max_t = 0.0
    else:
        storage_instance.discharge_max_t = min(
            storage_instance.power_capacity,
            max(
                storage_instance.energy_capacity
                - storage_instance.stored_energy_temp_reverse
                - inflow_energy,
                0,
            )
            * storage_instance.discharge_efficiency
            / resolution,
        )
        if storage_instance.chargeable:
            storage_instance.charge_max_t = min(
                storage_instance.power_capacity,
                max(storage_instance.stored_energy_temp_reverse - inflow_energy, 0.0)
                / storage_instance.charge_efficiency
                / resolution,
            )
        else:
            storage_instance.charge_max_t = 0.0

    offset = 0.0 if merit_order_idx == 0 else storage_instance.node.discharge_max_t[merit_order_idx - 1]
    storage_instance.node.discharge_max_t[merit_order_idx] = offset + storage_instance.discharge_max_t

    offset = 0.0 if merit_order_idx == 0 else storage_instance.node.charge_max_t[merit_order_idx - 1]
    storage_instance.node.charge_max_t[merit_order_idx] = offset + storage_instance.charge_max_t
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def dispatch(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
    merit_order_idx: nbintp,
) -> None:
    """
    Dispatches the Storage system according to its place in the merit order for the Storage.node.
    The total storage power at that node is also updated according to the dispatch of the Storage
    system.
    """
    target_load = storage_instance.node.netload_t - storage_instance.node.flexible_power[interval]

    offset = 0.0 if merit_order_idx == 0 else storage_instance.node.discharge_max_t[merit_order_idx - 1]
    storage_instance.dispatch_power[interval] = max(min(target_load - offset, storage_instance.discharge_max_t), 0.0)

    offset = 0.0 if merit_order_idx == 0 else storage_instance.node.charge_max_t[merit_order_idx - 1]
    storage_instance.dispatch_power[interval] += min(max(target_load + offset, -storage_instance.charge_max_t), 0.0)

    storage_instance.node.storage_power[interval] += storage_instance.dispatch_power[interval]
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_stored_energy(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    forward_time_flag: boolean,
) -> None:
    dispatched_energy = (
        max(storage_instance.dispatch_power[interval], 0) / storage_instance.discharge_efficiency * resolution
        + min(storage_instance.dispatch_power[interval], 0) * storage_instance.charge_efficiency * resolution
    )

    if forward_time_flag:
        soc = storage_instance.stored_energy[interval - 1] - dispatched_energy
        if storage_instance.inflows:
            soc += max(min(storage_instance.data[interval], storage_instance.energy_capacity - soc), 0.0)

        storage_instance.stored_energy[interval] = soc
    else:
        storage_instance.stored_energy_temp_reverse += dispatched_energy

        if storage_instance.inflows:
            actual_inflow = max(
                min(
                    storage_instance.data[interval],
                    storage_instance.energy_capacity - storage_instance.stored_energy_temp_reverse,
                ),
                0.0,
            )

            storage_instance.stored_energy_temp_reverse = max(
                storage_instance.stored_energy_temp_reverse - actual_inflow,
                0.0,
            )

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lt_generation(
    storage_instance: Storage_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Calculate the total energy discharged over the long-term modelling horizon for a Storage system.
    """
    lt_gen = 0.0
    lt_flows = 0.0
    for i in range(len(storage_instance.dispatch_power)):
        power = storage_instance.dispatch_power[i]
        if power > TOLERANCE:
            lt_gen += power * resolution
        lt_flows += abs(power) * resolution

    storage_instance.lt_generation = lt_gen
    storage_instance.line.lt_flows += lt_flows
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_variable_costs(
    storage_instance: Storage_InstanceType,
    year_float: nbfloat,
) -> nbfloat:
    """
    Calculate the total variable costs for a Storage system at the end of unit committment.
    """
    ltcosts_m.calculate_vom(
        storage_instance.lt_costs,
        storage_instance.lt_generation,
        year_float,
        storage_instance.cost,
    )
    ltcosts_m.calculate_fuel(
        storage_instance.lt_costs,
        storage_instance.lt_generation,
        year_float,
        0.0,
        storage_instance.cost,
    )
    return ltcosts_m.get_variable(storage_instance.lt_costs)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fixed_costs(
    storage_instance: Storage_InstanceType,
    include_existing: bool,
) -> nbfloat:
    """
    Calculate the total fixed costs for a Storage system.
    """
    if include_existing:
        ltcosts_m.calculate_annualised_build_power(
            storage_instance.lt_costs,
            storage_instance.power_capacity,
            0.0,
            storage_instance.cost,
            "storage",
        )
        ltcosts_m.calculate_annualised_build_energy(
            storage_instance.lt_costs,
            storage_instance.energy_capacity,
            storage_instance.cost,
            "storage",
        )
    else:
        ltcosts_m.calculate_annualised_build_power(
            storage_instance.lt_costs,
            storage_instance.new_build_p,
            0.0,
            storage_instance.cost,
            "storage",
        )
        ltcosts_m.calculate_annualised_build_energy(
            storage_instance.lt_costs,
            storage_instance.new_build_e,
            storage_instance.cost,
            "storage",
        )

    ltcosts_m.calculate_fom(
        storage_instance.lt_costs,
        storage_instance.power_capacity,
        0.0,
        storage_instance.cost,
        "storage",
    )
    return ltcosts_m.get_fixed(storage_instance.lt_costs)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_partial_cost_power(
    storage_instance: Storage_InstanceType,
    year_float: nbfloat,
):
    return ltcosts_m.get_partial_cost_power(
        storage_instance.new_build_p,
        storage_instance.power_capacity,
        0.0,
        storage_instance.lt_generation,
        year_float,
        0.0,
        storage_instance.cost,
        "storage",
    )


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_partial_cost_energy(
    storage_instance: Storage_InstanceType,
):
    return ltcosts_m.get_partial_cost_energy(
        storage_instance.new_build_e,
        storage_instance.cost,
        "storage",
    )


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_deficit_block(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
) -> None:
    """
    Upon resolving a deficit block, initialise the temporary stored energy,
    max stored energy, and min stored energy values for a Storage system.
    """
    storage_instance.stored_energy_temp_reverse = storage_instance.stored_energy[interval - 1]
    storage_instance.deficit_block_min_storage = storage_instance.stored_energy_temp_reverse
    storage_instance.deficit_block_max_storage = storage_instance.stored_energy_temp_reverse


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_deficit_block_bounds(
    storage_instance: Storage_InstanceType,
    stored_energy: nbfloat,
) -> None:
    """
    Update the temporary minimum and maximum stored energy values for the Storage system in the
    deficit block.
    """
    storage_instance.deficit_block_min_storage = min(storage_instance.deficit_block_min_storage, stored_energy)
    storage_instance.deficit_block_max_storage = max(storage_instance.deficit_block_max_storage, stored_energy)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_precharging_reserves(
    storage_instance: Storage_InstanceType,
) -> None:
    """
    Calculates the precharge energy that the Storage system must charge during the precharging period in order
    to dispatch during the deficit block (for prechargers).
    """
    storage_instance.precharge_flag = (
        storage_instance.deficit_block_max_storage - storage_instance.deficit_block_min_storage
        > storage_instance.stored_energy_temp_forward
    ) and (storage_instance.chargeable or storage_instance.inflows)
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_precharging_flags(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
) -> None:
    """
    Initialises the trickling flag and precharge flag for a Storage system once precharging in the lead-up to the deficit
    block begins.
    """
    storage_instance.trickling_flag = (
        storage_instance.stored_energy[interval] - storage_instance.trickling_reserves > TOLERANCE
    ) and (storage_instance.precharge_energy < TOLERANCE)
    storage_instance.precharge_flag = (storage_instance.precharge_energy > TOLERANCE) and (
        storage_instance.chargeable or storage_instance.inflows
    )
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_precharging_flags(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
) -> None:
    """
    At the start of a time interval within the precharging period, the remaining trickling reserves,
    trickling flag, and precharge flag for the Storage system is updated.
    """
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_precharging_max_t(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    merit_order_idx: nbintp,
) -> None:
    # Set discharge_max_t for trickle chargers
    if storage_instance.trickling_flag:
        if storage_instance.chargeable:
            charge_reduction_constraint_power = min(
                storage_instance.remaining_trickling_reserves / storage_instance.charge_efficiency / resolution,
                -min(storage_instance.dispatch_power[interval], 0.0),
            )
        else:
            charge_reduction_constraint_power = 0.0

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

        if storage_instance.chargeable:
            discharge_reduction_constraint_energy = (
                discharge_reduction_constraint_power / storage_instance.discharge_efficiency * resolution
            )
            charge_increase_constraint_power = min(
                (storage_instance.precharge_energy - discharge_reduction_constraint_energy)
                / storage_instance.charge_efficiency
                / resolution,
                storage_instance.power_capacity + min(storage_instance.dispatch_power[interval], 0.0),
            )
        else:
            charge_increase_constraint_power = 0.0

        storage_instance.charge_max_t = discharge_reduction_constraint_power + charge_increase_constraint_power
    else:
        storage_instance.charge_max_t = 0.0

    # Update nodal dispatch_max_t values
    offset = 0.0 if merit_order_idx == 0 else storage_instance.node.discharge_max_t[merit_order_idx - 1]
    storage_instance.node.discharge_max_t[merit_order_idx] = offset + storage_instance.discharge_max_t

    offset = 0.0 if merit_order_idx == 0 else storage_instance.node.charge_max_t[merit_order_idx - 1]
    storage_instance.node.charge_max_t[merit_order_idx] = offset + storage_instance.charge_max_t
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_dispatch_energy_update(
    storage_instance: Storage_InstanceType,
    dispatch_power_original: nbfloat,
    dispatch_power_update: nbfloat,
    resolution: nbfloat,
) -> nbfloat:
    dispatch_energy_update = 0.0

    if dispatch_power_original > 0.0:  # If originally discharging
        if dispatch_power_update > 0.0:  # Increase discharging power
            dispatch_energy_update = -dispatch_power_update / storage_instance.discharge_efficiency * resolution
        else:  # Reduce discharging power and increase charging power
            reduced_discharge = min(dispatch_power_original, -dispatch_power_update)
            dispatch_energy_update = reduced_discharge / storage_instance.discharge_efficiency * resolution
            if storage_instance.chargeable:
                new_charge = -min(dispatch_power_original + dispatch_power_update, 0.0)
                dispatch_energy_update += new_charge * storage_instance.charge_efficiency * resolution

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
        elif storage_instance.chargeable:
            dispatch_energy_update = -dispatch_power_update * storage_instance.charge_efficiency * resolution

    return dispatch_energy_update


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_precharge_dispatch(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    dispatch_power_update: nbfloat,
    precharging_flag: boolean,
    merit_order_idx: nbintp,
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_available_dispatch(
    storage_instance: Storage_InstanceType,
    interval: nbintp,
) -> None:
    """
    Calculate the maximum adjustment to dispatch power that is possible for a Storage system.
    """
    storage_instance.remaining_discharge_max_t = (
        storage_instance.discharge_max_t - storage_instance.dispatch_power[interval]
    )
    storage_instance.remaining_charge_max_t = storage_instance.charge_max_t + storage_instance.dispatch_power[interval]
    return None
