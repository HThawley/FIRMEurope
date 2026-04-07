# type: ignore
from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, TypedDict, boolean, nbfloat, nbint, nbintp, unicode_type
from firm_ce.fast_methods import generator_m, storage_m, fuel_m
from firm_ce.system.components import Fleet, Fleet_InstanceType, Generator_InstanceType, Storage_InstanceType, Fuel_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    fleet_instance: Fleet_InstanceType,
    nodes_typed_dict: DictType(nbintp, Node_InstanceType),
    lines_typed_dict: DictType(nbintp, Line_InstanceType),
) -> Fleet_InstanceType:
    """
    A 'static' instance of the Fleet jitclass (Fleet.static_instance=True) is copied
    and marked as a 'dynamic' instance (Fleet.static_instance=False).

    Static instances are created during Model initialisation and supplied as arguments
    to the differential evolution. These arguments are references to the original jitclass instances (not copies).
    Candidate solutions within the differential evolution are tested in embarrasingly parrallel,
    making it unsafe for multiple workers to similtaneously modify the same memory referenced
    across each process.

    Instead, each worker must create a deep copy of the referenced instance that is safe to modify
    within that worker process. Not all attributes within a dynamic instance are safe to modify.
    Only attributes that are required to be modified when testing the candidate solution are
    copied in order to save memory. If an attribute is unsafe to modify after copying, it will
    be marked with a comment that says "This remains static" in the create_dynamic_copy fast_method for
    that jitclass.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): A static instance of the Fleet jitclass.
    nodes_typed_dict (DictType(nbintp, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_typed_dict (DictType(nbintp, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.

    Returns:
    -------
    Fleet_InstanceType: A dynamic instance of the Fleet jitclass.
    """
    generators_copy = TypedDict.empty(key_type=nbintp, value_type=Generator_InstanceType)
    storages_copy = TypedDict.empty(key_type=nbintp, value_type=Storage_InstanceType)
    fuels_copy = TypedDict.empty(key_type=nbintp, value_type=Fuel_InstanceType)

    for order, fuel in fleet_instance.fuels.items():
        fuels_copy[order] = fuel_m.create_dynamic_copy(fuel)

    for order, generator in fleet_instance.generators.items():
        generators_copy[order] = generator_m.create_dynamic_copy(
            generator,
            nodes_typed_dict,
            lines_typed_dict,
            fuels_copy[generator.fuel.id],
        )

    for order, storage in fleet_instance.storages.items():
        storages_copy[order] = storage_m.create_dynamic_copy(storage, nodes_typed_dict, lines_typed_dict)

    fleet_copy = Fleet(
        False,
        generators_copy,
        storages_copy,
        fuels_copy,
    )

    return fleet_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacities(
    fleet_instance: Fleet_InstanceType,
    decision_x: nbfloat[:],
    interval_resolutions: nbfloat[:],
) -> None:
    """
    The candidate solution defines new build capacity for each Generator, Storage, and Line (major_lines) object. This
    function modifies each Generator and Storage object in the Fleet to build new capacity and updates the
    residual_load at corresponding nodes.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): A dynamic instance of the Fleet jitclass.
    decision_x (nbfloat[:]): A 1-dimensional array containing the candidate solution for the differential
        evolution. The candidate solution defines new build capacity for each decision variable (either power
        or energy capacity).
    interval_resolutions (nbfloat[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Generator instance in Fleet.generators: new_build, capacity, line, node, lt_generation.
    Attributes modified for each Line instance referenced in Generator.line: new_build, capacity, lt_flows.
    Attributes modified for each Node instance referenced in Generator.node: residual_load.
    Attributes modified for each Line instance referenced in Generator.line: new_build, capacity, lt_flows.
    Attributes modified for each Node instance referenced in Generator.node: residual_load.
    Attributes modified for each Storage instance in Fleet.storages: power_capacity, new_build_p, energy_capacity, new_build_e,
        line.
    Attributes modified for each Line instance referenced in Storage.line: new_build, capacity.
    """
    if fleet_instance.static_instance:
        raise_static_modification_error()

    for generator in fleet_instance.generators.values():
        generator_m.build_capacity(generator, decision_x[generator.candidate_x_idx], interval_resolutions)

    for storage in fleet_instance.storages.values():
        storage_m.build_capacity(storage, decision_x[storage.candidate_p_x_idx], "power")
        storage_m.build_capacity(storage, decision_x[storage.candidate_e_x_idx], "energy")
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(
    fleet_instance: Fleet_InstanceType,
    intervals_count: nbint,
) -> None:
    """
    Memory associated with time-series data for flexible generators and storage systems is only
    allocated after a dynamic copy of the Fleet instance is created. This is to minimise memory
    usage of the static instances.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): A dynamic instance of the Fleet jitclass.
    intervals_count (nbint): Total number of time intervals in the unit committment formulation.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each 'flexible' Generator instance in Fleet.generators: dispatch_power, remaining_energy.
    Attributes modified for each Storage instance in Fleet.storages: dispatch_power, stored_energy.
    """
    if fleet_instance.static_instance:
        raise_static_modification_error()

    for fuel in fleet_instance.fuels.values():
        fuel_m.allocate_memory(fuel, intervals_count)

    for generator in fleet_instance.generators.values():
        if generator.is_flexible:
            generator_m.allocate_memory(generator, intervals_count)

    for storage in fleet_instance.storages.values():
        storage_m.allocate_memory(storage, intervals_count)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_stored_energies(
    fleet_instance: Fleet_InstanceType,
) -> None:
    """
    An initial value for state-of-charge is defined for each storage system in the Fleet. This is done once
    per optimisation.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): A dynamic instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Storage instance in Fleet.storages: stored_energy.
    """
    if fleet_instance.static_instance:
        raise_static_modification_error()
    for storage in fleet_instance.storages.values():
        storage_m.initialise_stored_energy(storage)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_annual_limits(
    fleet_instance: Fleet_InstanceType,
    year: nbintp,
    first_t: nbintp,
) -> None:
    """
    The energy generation constraint for each flexible Generator is initialised. This is done once
    per year.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): A dynamic instance of the Fleet jitclass.
    year (nbintp): Defines the number of years that have completed balancing since the start of the
        optimisation. Used as the index for the Generator.annual_constraints_data array.
    first_t (nbintp): Index for the first time interval in the year.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each flexible Generator instance in Fleet.generators: remaining_energy.
    """
    if fleet_instance.static_instance:
        raise_static_modification_error()
    for fuel in fleet_instance.fuels.values():
        fuel_m.initialise_annual_limits(fuel, year, first_t)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def count_generator_unit_type(
    fleet_instance: Fleet_InstanceType,
    unit_type: unicode_type,
) -> nbint:
    """
    Returns a count of the number of generators of the specified unit_type within the Fleet.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    unit_type (unicode_type): The Generator.unit_type to be counted.

    Returns:
    -------
    nbint: The count of the number of generators of the specified unit_type.
    """
    count = 0
    for generator in fleet_instance.generators.values():
        if generator.unit_type == unit_type:
            count += 1
    return count


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_stored_energies(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    forward_time_flag: boolean,
) -> None:
    """
    Once the dispatch_power for the Storage objects have been determined for a time interval, the stored_energy
    for each Storage system is updated. During precharging actions, a temporary value is updated to track stored_energy
    constraints for dispatching.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.
    resolution (nbfloat): Resolution of the time interval (hours per time interval).
    forward_time_flag (boolean): True indicates the unit committment is iterating forwards through time. False
        indicates that it is moving backwards through time during precharging.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each Storage instance in Fleet.storages: stored_energy (forwards_time_flag = True) or
        stored_energy_temp_reverse (forwards_time_flag = False).
    """
    for storage in fleet_instance.storages.values():
        storage_m.update_stored_energy(storage, interval, resolution, forward_time_flag)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_remaining_flexible_energies(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    forward_time_flag: boolean,
) -> None:
    """
    Once the dispatch_power for the flexible Generator objects have been determined for a time interval, the remaining_energy
    for each flexible Generator system is updated. During precharging actions, a temporary value is updated to track
    remaining_energy constraints for dispatching.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.
    resolution (nbfloat): Resolution of the time interval (hours per time interval).
    forward_time_flag (boolean): True indicates the unit committment is iterating forwards through time. False
        indicates that it is moving backwards through time during precharging.
    previous_year_flag (boolean): True indicates that the interval for the precharging process has crossed into the previous
        year, indicating that the remaining_energy_temp_reverse must be based upon the previous year's remaining_energy constraint.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each flexible Generator instance in Fleet.generators: remaining_energy (forwards_time_flag = True) or
        remaining_energy_temp_reverse (forwards_time_flag = False).
    """
    if forward_time_flag:
        for fuel in fleet_instance.fuels.values():
            fuel.remaining_energy[interval] = fuel.remaining_energy[interval - 1]
        for generator in fleet_instance.generators.values():
            generator.fuel.remaining_energy[interval] -= generator.dispatch_power[interval] / resolution

    else:
        for fuel in fleet_instance.fuels.values():
            fuel.remaining_energy_temp_reverse = fuel.remaining_energy[interval - 1]
        for generator in fleet_instance.generators.values():
            generator.fuel.remaining_energy_temp_reverse -= generator.dispatch_power[interval] / resolution

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lt_generations(
    fleet_instance: Fleet_InstanceType,
    interval_resolutions: nbfloat[:],
) -> None:
    """
    The total energy generated by each flexible Generator and discharged from each Storage system during
    unit committment is calculated from the interval values.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval_resolutions (nbfloat[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each flexible Generator instance in Fleet.generators: unit_lt_hours, lt_generation, line.
    Attributes modified for each Line instance referenced in Generator.line: lt_flows.
    Attributes modified for each Storage instance in Fleet.storages: lt_generation, line.
    Attributes modified for each Line instance referenced in Storage.line: lt_flows.
    """
    for generator in fleet_instance.generators.values():
        if generator.is_flexible:
            generator_m.calculate_lt_generation(generator, interval_resolutions)

    for storage in fleet_instance.storages.values():
        storage_m.calculate_lt_generation(storage, interval_resolutions)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_deficit_block(
    fleet_instance: Fleet_InstanceType,
    interval_after_deficit_block: nbintp,
) -> None:
    """
    Initialise temporary energy constraint parameters and deficit block min/max energies for flexible Generator and
    Storage objects upon beginning precharging. The min/max energies for the deficit block are used to ensure
    Generator and Storage objects maintain sufficient reserves during precharging to complete dispatch during the
    deficit block.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval_after_deficit_block (nbintp): Index for the time interval immediatly following the deficit block.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each Storage instance in Fleet.storages: stored_energy_temp_reverse, deficit_block_min_storage,
        deficit_block_max_storage.
    Attributes modified for each flexible Generator instance in Fleet.generators: remaining_energy_temp_reverse,
        deficit_block_min_energy, deficit_block_max_energy.
    """
    for storage in fleet_instance.storages.values():
        storage_m.initialise_deficit_block(storage, interval_after_deficit_block)

    for fuel in fleet_instance.fuels.values():
        fuel_m.initialise_deficit_block(fuel, interval_after_deficit_block)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_flexible(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
) -> None:
    """
    Reset dispatch for all flexible Generator objects in a given time interval.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each flexible Generator instance in Fleet.generators: dispatch_power.
    """
    for generator in fleet_instance.generators.values():
        if generator.is_flexible:
            generator.dispatch_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_dispatch(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
) -> None:
    """
    Reset dispatch for all Storage systems and flexible Generator objects in a given time interval.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each Storage instance in Fleet.storages: dispatch_power.
    Attributes modified for each flexible Generator instance in Fleet.generators: dispatch_power.
    """
    for storage in fleet_instance.storages.values():
        storage.dispatch_power[interval] = 0.0
    reset_flexible(fleet_instance, interval)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_deficit_block(
    fleet_instance: Fleet_InstanceType,
) -> None:
    """
    Updates the min/max energies for Storage and flexible Generator objects within the deficit block
    during precharging. The min/max energies for the deficit block are used to ensure
    Generator and Storage objects maintain sufficient reserves during precharging to complete dispatch during the
    deficit block.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each Storage instance in Fleet.storages: deficit_block_min_storage, deficit_block_max_storage.
    Attributes modified for each flexible Generator instance in Fleet.generators: deficit_block_min_energy,
        deficit_block_max_energy.
    """
    for storage in fleet_instance.storages.values():
        storage_m.update_deficit_block_bounds(storage, storage.stored_energy_temp_reverse)

    for fuel in fleet_instance.fuels.values():
        fuel_m.update_deficit_block_bounds(fuel, fuel.remaining_energy_temp_reverse)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_precharging_values(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    year: nbintp,
) -> None:
    """
    Once the first time interval in a deficit block is located (during reverse-time precharging),
    the precharging energy for Storage prechargers and trickling reserves for Storage tricklers and
    flexible Generators are defined. These parameters are used to constrain discharging from trickle
    chargers (ensuring they maintain enough energy to dispatch during deficit block) and charging from
    prechargers (ensuring they stop precharging once sufficient energy has been stored to dispatch during the
    deficit block) in the precharging period leading up to the deficit block.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the current time interval.
    resolution (nbfloat): Resolution of the interval (hours per time interval).
    year (nbintp): Defines the number of years that have completed balancing since the start of the
        optimisation. Used as the index for the Generator.annual_constraints_data array.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each flexible Generator instance in Fleet.generators: remaining_energy_temp_forward,
        deficit_block_min_energy, deficit_block_max_energy, trickling_reserves.
    Attributes modified for each Storage instance in Fleet.storages: stored_energy_temp_forward,
        deficit_block_min_storage, deficit_block_max_storage, precharge_flag, precharge_energy, trickling_reserves.
    """
    for fuel in fleet_instance.fuels.values():
        fuel.remaining_energy[interval] = fuel.remaining_energy[interval - 1]
        fuel.remaining_energy_temp_forward = min(max(fuel.remaining_energy_temp_forward, 0.0), fuel_m.get_annual_limit(fuel, year))

        fuel_m.update_deficit_block_bounds(fuel, fuel.remaining_energy_temp_forward)
        fuel_m.assign_trickling_reserves(fuel)

    for generator in fleet_instance.generators.values():
        if generator.is_flexible:
            generator.fuel.remaining_energy[interval] -= generator.dispatch_power[interval] / resolution

    for storage in fleet_instance.storages.values():
        # After reverse charging, the stored energy is discontinuous in the forward and reverse directions
        dispatched_energy = (
            max(storage.dispatch_power[interval], 0.0) / storage.discharge_efficiency * resolution
            + min(storage.dispatch_power[interval], 0.0) * storage.charge_efficiency * resolution
        )
        soc_forward = storage.stored_energy[interval - 1] - dispatched_energy
        if storage.inflows:
            soc_forward += max(min(storage.data[interval], storage.energy_capacity - soc_forward), 0.0)

        storage.stored_energy_temp_forward = min(max(soc_forward, 0.0), storage.energy_capacity)
        storage_m.update_deficit_block_bounds(storage, storage.stored_energy_temp_forward)
        storage_m.assign_precharging_reserves(storage)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_precharging_flags(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
) -> None:
    """
    Initialise boolean flags that control precharging and trickling
    behaviour for Storage systems and flexible Generators upon beginning precharging
    in the lead-up to the deficit block. Precharge flag is True when there is remaining
    precharging energy for the storage system. Trickling flag is True when there are sufficient
    reserves for the Storage tricklers and flexible Generators to dispatch within the deficit block.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the first interval in the deficit block.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each Storage instance in Fleet.storages: precharge_flag, trickling_flag.
    Attributes modified for each flexible Generator instance in Fleet.generators: trickling_flag.
    """
    for storage in fleet_instance.storages.values():
        storage_m.initialise_precharging_flags(storage, interval)

    for fuel in fleet_instance.fuels.values():
        fuel_m.initialise_precharging_flags(fuel, interval)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_precharging_flags(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
) -> None:
    """
    Update boolean flags and remaining_trickling_reserves that control precharging and trickling
    behaviour for Storage systems and flexible Generators at the start of a time interval
    during the precharging process. Precharge flag is True when there is remaining
    precharging energy for the storage system. Trickling flag is True when there are sufficient
    reserves for the Storage tricklers and flexible Generators to dispatch within the deficit block.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects
    -------
    Attributes modified for each Storage instance in Fleet.storages: precharge_flag, trickling_flag,
        remaining_trickling_reserves.
    Attributes modified for each flexible Generator instance in Fleet.generators: trickling_flag,
        remaining_trickling_reserves.
    """
    for storage in fleet_instance.storages.values():
        storage_m.update_precharging_flags(storage, interval)

    for fuel in fleet_instance.fuels.values():
        fuel_m.update_precharging_flags(fuel, interval)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_precharge_remaining(
    fleet_instance: Fleet_InstanceType,
) -> boolean:
    """
    Check whether any Storage objects are still attempting to precharge.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    boolean: True if any Storage system in the Fleet is still attempting to precharge, otherwise False.
    """
    for storage in fleet_instance.storages.values():
        if storage.precharge_flag:
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_trickling_remaining(
    fleet_instance: Fleet_InstanceType,
) -> boolean:
    """
    Check whether any Storage objects flexible Generators are still available for trickling.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    boolean: True if any Storage system or flexible Generator in the Fleet is able to trickle charge, otherwise False.
    """
    for storage in fleet_instance.storages.values():
        if storage.trickling_flag:
            return True

    for fuel in fleet_instance.fuels.values():
        for generator in fleet_instance.generators.values():
            if generator.fuel.id == fuel.id and generator.is_flexible and fuel.trickling_flag:
                return True

    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def determine_feasible_storage_dispatch(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
) -> boolean:
    """
    Determine whether the Storage dispatch_powers for a time interval calculated during reverse time precharging are
    still feasible when resolving the discontinuity created at the beginning of the precharging period.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.

    Returns:
    -------
    boolean: True if any original Storage.dispatch_power[interval] was found to be infeasible and adjusted.

    Side-effects:
    -------
    If an original Storage.dispatch_power[interval] would exceed the energy capacity constraints for the system,
    the power is adjusted for that time interval. The Storage.node.storage_power[interval] is also modified when
    these adjustments are made.
    """
    infeasible_flag = False
    for storage in fleet_instance.storages.values():
        original_dispatch_power = storage.dispatch_power[interval]
        storage.dispatch_power[interval] = max(min(original_dispatch_power, storage.discharge_max_t), 0.0) + min(
            max(original_dispatch_power, -storage.charge_max_t), 0.0
        )
        dispatch_power_adjustment = original_dispatch_power - storage.dispatch_power[interval]
        if abs(dispatch_power_adjustment) > TOLERANCE:
            storage.node.storage_power[interval] -= dispatch_power_adjustment
            infeasible_flag = True
    return infeasible_flag


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def determine_feasible_flexible_dispatch(
    fleet_instance: Fleet_InstanceType,
    interval: nbintp,
) -> boolean:
    """
    Determine whether the flexible Generator dispatch_powers for a time interval calculated during reverse time precharging are
    still feasible when resolving the discontinuity created at the beginning of the precharging period.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.

    Returns:
    -------
    boolean: True if any original flexible Generator.dispatch_power[interval] was found to be infeasible and adjusted.

    Side-effects:
    -------
    If an original flexible Generator.dispatch_power[interval] would exceed the remaining energy constraints for the system,
    the power is adjusted for that time interval. The Generator.node.flexible_power[interval] is also modified when
    these adjustments are made.
    """
    infeasible_flag = False
    for generator in fleet_instance.generators.values():
        if not generator.is_flexible:
            continue
        original_dispatch_power = generator.dispatch_power[interval]
        generator.dispatch_power[interval] = min(original_dispatch_power, generator.flexible_max_t)
        dispatch_power_adjustment = original_dispatch_power - generator.dispatch_power[interval]
        if abs(dispatch_power_adjustment) > TOLERANCE:
            generator.node.flexible_power[interval] -= dispatch_power_adjustment
            infeasible_flag = True
    return infeasible_flag


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_available_storage_dispatch(fleet_instance: Fleet_InstanceType, interval: nbintp) -> None:
    """
    Calculates the maximum amount that dispatch_power for each Storage system in a particular time interval can be adjusted.
    The remaining_discharge_max_t accounts for charging power reduction and discharging power increases. Vice versa for
    remaining_charge_max_t.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.
    interval (nbintp): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Storage instance in Fleet.storages: remaining_discharge_max_t, remaining_charge_max_t.
    """
    for storage in fleet_instance.storages.values():
        storage_m.calculate_available_dispatch(storage, interval)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_flexible_reserves(fleet_instance: Fleet_InstanceType) -> None:
    """
    Resets the trickling reserves for all flexible Generators to 0. Required when
    the precharging period crosses into the previous calendar year.

    Parameters:
    -------
    fleet_instance (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each flexible Generator instance in Fleet.generators: trickling_reserves.
    """
    for fuel in fleet_instance.fuels.values():
        fuel.trickling_reserves = 0
    return None
