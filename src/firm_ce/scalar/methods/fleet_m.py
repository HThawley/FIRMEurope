# type: ignore
from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, TypedDict, boolean, nbfloat, nbint, nbintp, unicode_type
from firm_ce.scalar.entities.components import (
    Fleet,
    Fleet_InstanceType,
    Fuel_InstanceType,
    Generator_InstanceType,
    Storage_InstanceType,
)
from firm_ce.scalar.entities.topology import Line_InstanceType, Node_InstanceType
from firm_ce.scalar.methods import fuel_m, generator_m, storage_m
from firm_ce.system.costs_common import apportion_new_phes


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    fleet_instance: Fleet_InstanceType,
    nodes_typed_dict: DictType(nbintp, Node_InstanceType),
    lines_typed_dict: DictType(nbintp, Line_InstanceType),
) -> Fleet_InstanceType:
    """
    A 'static' instance of the Fleet jitclass (Fleet.static_instance=True) is copied
    and marked as a 'dynamic' instance (Fleet.static_instance=False).
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
    resolution: nbfloat,
) -> None:
    if fleet_instance.static_instance:
        raise_static_modification_error()

    for generator in fleet_instance.generators.values():
        if generator.candidate_x_idx != -1:
            generator_m.build_capacity(generator, decision_x[generator.candidate_x_idx], resolution)

    for storage in fleet_instance.storages.values():
        ut = storage.unit_type
        if ut == "aa_phes" or ut == "e_phes":
            n_order = storage.node.order
            scaler = storage.node.relative_scaler
            tot_p = decision_x[storage.candidate_p_x_idx] * scaler if storage.candidate_p_x_idx != -1 else 0.0
            tot_e = decision_x[storage.candidate_e_x_idx] * scaler if storage.candidate_e_x_idx != -1 else 0.0

            e_aa_max = 0.0
            for other in fleet_instance.storages.values():
                if other.node.order == n_order and other.unit_type == "aa_phes":
                    e_aa_max = other.max_build_e
                    break

            p_aa, e_aa, p_e, e_e = apportion_new_phes(tot_p, tot_e, e_aa_max)
            if ut == "aa_phes":
                storage_m.build_capacity_abs(storage, p_aa, "power")
                storage_m.build_capacity_abs(storage, e_aa, "energy")
            else:
                storage_m.build_capacity_abs(storage, p_e, "power")
                storage_m.build_capacity_abs(storage, e_e, "energy")

            # Set endogenous duration for scalar merit-order sorting, nudging e_phes so aa_phes dispatches first
            if storage.power_capacity > TOLERANCE:
                dur = storage.energy_capacity / storage.power_capacity
                storage.duration = (dur - 1e-6) if ut == "e_phes" else dur
        else:
            if storage.candidate_p_x_idx != -1:
                storage_m.build_capacity(storage, decision_x[storage.candidate_p_x_idx], "power")
            if storage.candidate_e_x_idx != -1:
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
    for each Storage system is updated.
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
    for each flexible Generator system is updated.
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
    resolution: nbfloat,
) -> None:
    """
    The total energy generated by each flexible Generator and discharged from each Storage system during
    unit committment is calculated from the interval values.
    """
    for generator in fleet_instance.generators.values():
        if generator.is_flexible:
            generator_m.calculate_lt_generation(generator, resolution)

    for storage in fleet_instance.storages.values():
        storage_m.calculate_lt_generation(storage, resolution)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_deficit_block(
    fleet_instance: Fleet_InstanceType,
    interval_after_deficit_block: nbintp,
) -> None:
    """
    Initialise temporary energy constraint parameters and deficit block min/max energies for flexible Generator and
    Storage objects upon beginning precharging.
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
    during precharging.
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
    flexible Generators are defined.
    """
    for fuel in fleet_instance.fuels.values():
        fuel.remaining_energy[interval] = fuel.remaining_energy[interval - 1]
        fuel.remaining_energy_temp_forward = min(
            max(fuel.remaining_energy_temp_forward, 0.0), fuel_m.get_annual_limit(fuel, year)
        )

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
    in the lead-up to the deficit block.
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
    during the precharging process.
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
    Check whether any Storage objects or flexible Generators are still available for trickling.
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
    """
    for storage in fleet_instance.storages.values():
        storage_m.calculate_available_dispatch(storage, interval)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_flexible_reserves(fleet_instance: Fleet_InstanceType) -> None:
    """
    Resets the trickling reserves for all flexible Generators to 0. Required when
    the precharging period crosses into the previous calendar year.
    """
    for fuel in fleet_instance.fuels.values():
        fuel.trickling_reserves = 0
    return None
