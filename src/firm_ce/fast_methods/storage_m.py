# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
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
        storage_instance.chargeable,
        storage_instance.inflows,
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

    if storage_instance.inflows:
        storage_copy.data_status = storage_instance.data_status
        storage_copy.data = storage_instance.data
    storage_copy.lt_generation = storage_instance.lt_generation

    storage_copy.candidate_p_x_idx = storage_instance.candidate_p_x_idx
    storage_copy.candidate_e_x_idx = storage_instance.candidate_e_x_idx

    return storage_copy


@njit(fastmath=FASTMATH)
def build_capacity(
    storage_instance: Storage_InstanceType,
    new_build_capacity: float64,
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

    if capacity_type == "energy":
        if storage_instance.duration == 0:
            storage_instance.energy_capacity += new_build_capacity
            storage_instance.new_build_e += new_build_capacity
    return None


@njit(fastmath=FASTMATH)
def load_data(
    storage_instance: Storage_InstanceType,
    inflow_trace: float64[:],
) -> None:
    """
    Load the inflow trace data to the Storage instance. This is done before solving a Scenario.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    inflow_trace (float64[:]): Array containing the time-series inflow trace for the Storage. Each element
        provides the capacity factor for a time interval.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: data_status, data.
    """
    storage_instance.data = inflow_trace
    storage_instance.data_status = True

    return None


@njit(fastmath=FASTMATH)
def unload_data(storage_instance: Storage_InstanceType) -> None:
    """
    Unload the inflow data from the Storage instance. This is done after solving a Scenario to reduce memory usage.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: data_status, data.
    """
    storage_instance.data = np.empty((0,), dtype=np.float64)
    storage_instance.data_status = False
    return None


@njit(fastmath=FASTMATH)
def get_data(
    storage_instance: Storage_InstanceType,
    data_type: unicode_type,
) -> float64[:]:
    """
    Gets the specified data_type from the Storage instance.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    data_type (unicode_type): String associated with the data array.

    Returns:
    -------
    float64[:]: The data array associated with data_type.

    Raises:
    -------
    RuntimeError: Raised if data_status is False or if data_type does not correspond
        to any data arrays for the Reservoir jitclass.
    """
    if not storage_instance.data_status:
        raise_getting_unloaded_data_error()

    if data_type == "inflow":
        return storage_instance.data
    else:
        raise RuntimeError("Invalid data_type argument for Storage.get_data(data_type).")


@njit(fastmath=FASTMATH)
def allocate_memory(
    storage_instance: Storage_InstanceType,
    intervals_count: int64
) -> None:
    """
    Memory associated with endogenous time-series data for a Storage system is only allocated after a dynamic copy of
    the Storage instance is created. This is to minimise memory usage of the static instances.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): A dynamic instance of the Storage jitclass.
    intervals_count (int64): Total number of time intervals over the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: dispatch_power, stored_energy.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if storage_instance.static_instance:
        raise_static_modification_error()
    storage_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    storage_instance.stored_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def initialise_stored_energy(
    storage_instance: Storage_InstanceType,
) -> None:
    """
    Initialise the stored energy for a Storage system. Called at the start of the modelling period.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: stored_energy. The initial stored energy is stored
    in the final time interval for simplicity. It is overwritten upon performing balancing for final time interval.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if storage_instance.static_instance:
        raise_static_modification_error()
    # TODO:  Make it possible for user to define custom value for initial stored energy in future
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
    inflow_energy = 0.0
    if storage_instance.inflows:
        inflow_energy = storage_instance.data[interval]

    if forward_time_flag:
        storage_instance.discharge_max_t = min(
            storage_instance.power_capacity,
            (storage_instance.stored_energy[interval - 1] + inflow_energy) * storage_instance.discharge_efficiency / resolution,
        )
        if storage_instance.chargeable:
            storage_instance.charge_max_t = min(
                storage_instance.power_capacity,
                max(storage_instance.energy_capacity - storage_instance.stored_energy[interval - 1] - inflow_energy, 0.0)
                / storage_instance.charge_efficiency / resolution,
            )
        else:
            storage_instance.charge_max_t = 0.0
    else:
        storage_instance.discharge_max_t = min(
            storage_instance.power_capacity,
            max(storage_instance.energy_capacity - storage_instance.stored_energy_temp_reverse - inflow_energy, 0)
            * storage_instance.discharge_efficiency / resolution,
        )
        if storage_instance.chargeable:
            storage_instance.charge_max_t = min(
                storage_instance.power_capacity,
                max(storage_instance.stored_energy_temp_reverse - inflow_energy, 0.0)
                / storage_instance.charge_efficiency / resolution,
            )
        else:
            storage_instance.charge_max_t = 0.0

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
def dispatch(
    storage_instance: Storage_InstanceType,
    interval: int64,
    merit_order_idx: int64,
) -> None:
    """
    Dispatches the Storage system according to its place in the merit order for the Storage.node.
    The total storage power at that node is also updated according to the dispatch of the Storage
    system.

    Negative values of dispatch power indicate charging, positive values indicate discharging.
    Dispatch power is constrained by power capacity, energy capacity (charging), and stored energy
    (discharging).

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    interval (int64): Index for the time interval during unit committment.
    merit_order_idx (int64): Location of the Storage system in the merit order at the Storage.node.
        Lower merit_order_idx indicates shorter duration and higher priority in the merit order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: dispatch_power, node.
    Attributes modified for referenced Storage.node: storage_power.
    """
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
    storage_instance: Storage_InstanceType,
    interval: int64,
    resolution: float64,
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
                    storage_instance.energy_capacity - storage_instance.stored_energy_temp_reverse
                ),
                0.0
            )

            storage_instance.stored_energy_temp_reverse = max(
                storage_instance.stored_energy_temp_reverse - actual_inflow,
                0.0
            )

    return None


@njit(fastmath=FASTMATH)
def calculate_lt_generation(
    storage_instance: Storage_InstanceType,
    interval_resolutions: float64[:],
) -> None:
    """
    Calculate the total energy discharged over the long-term modelling horizon for a Storage system.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: lt_generation, line.
    Attributes modified for the referenced Storage.line: lt_flows.
    """
    storage_instance.lt_generation = sum(np.maximum(storage_instance.dispatch_power, 0) * interval_resolutions)

    storage_instance.line.lt_flows += sum(np.abs(storage_instance.dispatch_power) * interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def calculate_variable_costs(
    storage_instance: Storage_InstanceType,
    year_float: float64,
) -> float64:
    """
    Calculate the total variable costs for a Storage system at the end of unit committment.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    year_float (float64): Number of years. Leap days provide additional fractional value.

    Returns:
    -------
    float64: Total variable costs ($), equal to sum of fuel and VO&M costs.

    Side-effects:
    -------
    Attributes modified for the Storage instance: lt_costs.
    Attributes modified for the referenced Storage.lt_costs: vom, fuel.
    """
    ltcosts_m.calculate_vom(
        storage_instance.lt_costs,
        storage_instance.lt_generation,
        year_float,
        storage_instance.cost
    )
    ltcosts_m.calculate_fuel(
        storage_instance.lt_costs,
        storage_instance.lt_generation,
        year_float,
        0.0,
        storage_instance.cost
    )
    return ltcosts_m.get_variable(storage_instance.lt_costs)


@njit(fastmath=FASTMATH)
def calculate_fixed_costs(
    storage_instance: Storage_InstanceType,
    include_existing: bool,
) -> float64:
    """
    Calculate the total fixed costs for a Storage system.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.

    Returns:
    -------
    float64: Total fixed costs ($), equal to sum of annualised build and FO&M costs.

    Side-effects:
    -------
    Attributes modified for the Storage instance: lt_costs.
    Attributes modified for the referenced Storage.lt_costs: annualised_build, fom.
    """
    if include_existing:
        ltcosts_m.calculate_annualised_build(
            storage_instance.lt_costs,
            storage_instance.energy_capacity,
            storage_instance.power_capacity,
            0.0,
            storage_instance.cost,
            "storage",
        )
    else:
        ltcosts_m.calculate_annualised_build(
            storage_instance.lt_costs,
            storage_instance.new_build_e,
            storage_instance.new_build_p,
            0.0,
            storage_instance.cost,
            "storage",
        )
    ltcosts_m.calculate_fom(
        storage_instance.lt_costs,
        storage_instance.power_capacity,
        0.0,
        storage_instance.cost,
        "storage"
    )
    return ltcosts_m.get_fixed(storage_instance.lt_costs)


@njit(fastmath=FASTMATH)
def initialise_deficit_block(
    storage_instance: Storage_InstanceType,
    interval: int64,
) -> None:
    """
    Upon resolving a deficit block, initialise the temporary stored energy,
    max stored energy, and min stored energy values for a Storage system. These temporary
    variables are updated while performing unit committment in the reverse time direction for each time interval
    in the deficit block.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    interval (int64): Index for the first time interval immediately following the deficit block.
        During unit committment for the deficit block, time intervals will decrease in value (reverse
        time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: stored_energy_temp_reverse, deficit_block_max_storage,
        deficit_block_min_storage.
    """
    storage_instance.stored_energy_temp_reverse = storage_instance.stored_energy[interval - 1]
    storage_instance.deficit_block_min_storage = storage_instance.stored_energy_temp_reverse
    storage_instance.deficit_block_max_storage = storage_instance.stored_energy_temp_reverse


@njit(fastmath=FASTMATH)
def update_deficit_block_bounds(
    storage_instance: Storage_InstanceType,
    stored_energy: float64,
) -> None:
    """
    Update the temporary minimum and maximum stored energy values for the Storage system in the
    deficit block. These values are updated in each time interval for the deficit block. The minimum
    and maximum stored energies are used to define the trickling reserves that must be retained in
    the precharging period leading up to the deficit block such that the Storage system is capable of dispatching
    during the deficit block.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    stored_energy (float64): The stored energy in a time interval for the Storage system.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: deficit_block_max_storage, deficit_block_min_storage.
    """
    storage_instance.deficit_block_min_storage = min(storage_instance.deficit_block_min_storage, stored_energy)
    storage_instance.deficit_block_max_storage = max(storage_instance.deficit_block_max_storage, stored_energy)
    return None


@njit(fastmath=FASTMATH)
def assign_precharging_reserves(
    storage_instance: Storage_InstanceType,
) -> None:
    """
    Calculates the precharge energy that the Storage system must charge during the precharging period in order
    to dispatch during the deficit block (for prechargers). Based upon the discontinuity between the forwards-time
    stored energy and reverse-time stored energy in the first time interval of the deficit block.

    Alternatively, calculates the trickling reserves that must be retained during precharging such that the
    Storage system can dispatch within the following deficit block (for trickle chargers). Based upon the difference
    between maximum and minimum remaining energy values within the deficit block.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: precharge_flag, precharge_energy, trickling_reserves.
    """
    storage_instance.precharge_flag = storage_instance.chargeable and (
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
def initialise_precharging_flags(
    storage_instance: Storage_InstanceType,
    interval: int64,
) -> None:
    """
    Initialises the trickling flag and precharge flag for a Storage system once precharging in the lead-up to the deficit
    block begins.

    The trickling flag is True if the Storage system has sufficient stored energy
    such that it still retains the trickling reserves required to dispatch in the subsequent
    deficit block. When the trickling flag is True, a Storage system is assumed to be available for
    trickle charging a Storage precharger.

    The precharge flag is True if the Storage system is required to charge prior to the deficit block such that it can
    dispatch during the deficit block.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    interval (int64): Index for the first time interval of the deficit block (immediately following the
        precharging period). Time intervals during the precharging period will decrease in value (reverse
        time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: trickling_flag, precharge_flag.
    """
    storage_instance.trickling_flag = (
        storage_instance.stored_energy[interval] - storage_instance.trickling_reserves > TOLERANCE
    ) and (storage_instance.precharge_energy < TOLERANCE)
    storage_instance.precharge_flag = storage_instance.precharge_energy > TOLERANCE
    return None


@njit(fastmath=FASTMATH)
def update_precharging_flags(
    storage_instance: Storage_InstanceType,
    interval: int64,
) -> None:
    """
    At the start of a time interval within the precharging period, the remaining trickling reserves,
    trickling flag, and precharge flag for the Storage system is updated. The remaining trickling reserves define the
    amount of energy available for trickle charging, ensuring that the Storage system retains sufficient
    reserves to dispatch during the deficit block immediately after the precharging period.

    The trickling flag is True if the Storage system has sufficient stored energy
    such that it still retains the trickling reserves required to dispatch in the subsequent
    deficit block. When the trickling flag is True, a Storage system is assumed to be available for
    trickle charging a Storage precharger.

    The precharge flag is True if the Storage system still requires more energy to be charged prior to the deficit block
    such that it can dispatch during the deficit block.

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    interval (int64): Index for the current time interval in the precharging period. Time intervals during
        the precharging period will decrease in value (reverse time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: remaining_trickling_reserves, trickling_flag, precharge_flag.
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


@njit(fastmath=FASTMATH)
def set_precharging_max_t(
    storage_instance: Storage_InstanceType,
    interval: int64,
    resolution: float64,
    merit_order_idx: int64,
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
    if storage_instance.precharge_flag and storage_instance.chargeable:
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
def calculate_available_dispatch(
    storage_instance: Storage_InstanceType,
    interval: int64,
) -> None:
    """
    Calculate the maximum adjustment to dispatch power that is possible for a Storage system. When
    a precharging action is found to be infeasible (while attempting to dispatch according to the
    precharging period powers), adjustments must be made to find a feasible method of dispatch.
    Infeasible dispatch indicates that the deficit block cannot be resolved by precharging.

    Remaining discharge max accounts for a reduction in charging power (and vice versa for remaining
    charge max).

    Parameters:
    -------
    storage_instance (Storage_InstanceType): An instance of the Storage jitclass.
    interval (int64): Index for the current time interval when re-calculating stored energy by dispatching
        according to the precharging dispatch powers. Note that this is done in forward time.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Storage instance: remaining_discharge_max_t, remaining_charge_max_t.
    """
    storage_instance.remaining_discharge_max_t = (
        storage_instance.discharge_max_t - storage_instance.dispatch_power[interval]
    )
    storage_instance.remaining_charge_max_t = storage_instance.charge_max_t + storage_instance.dispatch_power[interval]
    return None
