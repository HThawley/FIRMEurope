# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, nbfloat, npfloat, nbint, nbintp, unicode_type
from firm_ce.fast_methods import ltcosts_m, node_m
from firm_ce.system.scalar.components import Generator, Generator_InstanceType, Fuel_InstanceType
from firm_ce.system.scalar.topology import Line_InstanceType, Node_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    generator_instance: Generator_InstanceType,
    nodes_typed_dict: DictType(nbintp, Node_InstanceType),
    lines_typed_dict: DictType(nbintp, Line_InstanceType),
    fuel_dynamic_copy: Fuel_InstanceType,
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
        generator_instance.is_flexible,
        generator_instance.near_optimum_check,
        node_copy,
        fuel_dynamic_copy,  # This does not always remain static but must be updated at the same time as everything else
        line_copy,
        generator_instance.group,
        generator_instance.cost,  # This remains static
    )
    generator_copy.unit_type_idx = generator_instance.unit_type_idx
    generator_copy.data_status = generator_instance.data_status
    generator_copy.data = generator_instance.data  # This remains static
    generator_copy.candidate_x_idx = generator_instance.candidate_x_idx
    generator_copy.lt_generation = generator_instance.lt_generation
    generator_copy.heat_base_consumption = generator_instance.heat_base_consumption

    return generator_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacity(
    generator_instance: Generator_InstanceType,
    new_build_power_capacity: nbfloat,
    resolution: nbfloat,
) -> None:
    if generator_instance.static_instance:
        raise_static_modification_error()
    generator_instance.capacity += new_build_power_capacity
    generator_instance.new_build += new_build_power_capacity
    generator_instance.heat_base_consumption = generator_instance.capacity * generator_instance.cost.heat_rate_base  # GWh/h
    generator_instance.line.capacity += new_build_power_capacity
    generator_instance.line.new_build += new_build_power_capacity

    update_residual_load(generator_instance, new_build_power_capacity, resolution)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def load_data(
    generator_instance: Generator_InstanceType,
    generation_trace: nbfloat[:],
    resolution: nbfloat,
) -> None:
    """
    Load the capacity factor trace and flexible annual constraint data to the Generator instance. This is done
    before solving a Scenario.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    generation_trace (nbfloat[:]): Array containing the time-series capacity factor trace for the Generator. Each element
        provides the capacity factor for a time interval.
    resolution (nbfloat): A scalar containing the resolution for every time interval

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: data_status, data, lt_generation.
    Attributes modified for the referenced Generator.line: lt_flows.
    Attributes modified for the referenced Generator.node: residual_load.
    """
    generator_instance.data = generation_trace
    generator_instance.data_status = True

    update_residual_load(generator_instance, generator_instance.initial_capacity, resolution)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def unload_data(generator_instance: Generator_InstanceType) -> None:
    """
    Unload the capacity factor trace and flexible annual constraint data from the Generator instance. This is done
    after solving a Scenario to reduce memory usage.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: data_status, data, annual_constraints_data.
    """
    generator_instance.data = np.empty((0,), dtype=npfloat)
    generator_instance.data_status = False
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_data(
    generator_instance: Generator_InstanceType,
    data_type: unicode_type,
) -> nbfloat[:]:
    """
    Gets the specified data_type from the Generator instance.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    data_type (unicode_type): String associated with the data array.

    Returns:
    -------
    nbfloat[:]: The data array associated with data_type.

    Raises:
    -------
    RuntimeError: Raised if data_status is False or if data_type does not correspond
        to any data arrays for the Generator jitclass.
    """
    if not generator_instance.data_status:
        raise_getting_unloaded_data_error()

    if data_type == "trace":
        return generator_instance.data
    else:
        raise RuntimeError("Invalid data_type argument for Generator.get_data(data_type).")
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(
    generator_instance: Generator_InstanceType,
    intervals_count: nbint,
) -> None:
    """
    Memory associated with endogenous time-series data for a flexible Generator is only allocated after a dynamic copy of
    the Generator instance is created. This is to minimise memory usage of the static instances.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): A dynamic instance of the Generator jitclass.
    intervals_count (nbint): Total number of time intervals over the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the flexible Generator instance: dispatch_power, remaining_energy.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if generator_instance.static_instance:
        raise_static_modification_error()
    generator_instance.dispatch_power = np.zeros(intervals_count, dtype=npfloat)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_residual_load(
    generator_instance: Generator_InstanceType,
    added_capacity: nbfloat,
    resolution: nbfloat,
) -> None:
    if get_data(generator_instance, "trace").shape[0] > 0 and added_capacity > 0.0:
        new_trace = get_data(generator_instance, "trace") * added_capacity
        node_m.get_data(generator_instance.node, "residual_load")[:] -= new_trace
        update_lt_generation(generator_instance, new_trace, resolution)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_lt_generation(
    generator_instance: Generator_InstanceType,
    generation_trace: nbfloat[:],
    resolution: nbfloat,
) -> None:
    generator_instance.lt_generation += sum(generation_trace * resolution)
    generator_instance.line.lt_flows += generator_instance.lt_generation
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_unit_type(
    generator_instance: Generator_InstanceType,
    unit_type: unicode_type,
) -> boolean:
    """
    Check whether a Generator.unit_type has a specified value. Commonly used to check if
    a Generator has the 'flexible' unit_type.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    unit_type (unicode_type): String specifiying a unit_type to compare with the Generator.unit_type.
        Expected to have a value of 'solar', 'wind', 'baseload', or 'flexible'.

    Returns:
    -------
    boolean: If the specified unit_type matches the Generator.unit_type, returns True. Otherwise, False.
    """
    return generator_instance.unit_type == unit_type


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_flexible_max_t(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    merit_order_idx: nbintp,
    forward_time_flag: boolean,
) -> None:
    advertised_limit = min(generator_instance.capacity, generator_instance.fuel.allocated_energy / resolution)
    generator_instance.flexible_max_t = advertised_limit
    generator_instance.fuel.allocated_energy -= advertised_limit * resolution

    update_node_flexible_max_t(generator_instance, merit_order_idx)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_precharging_max_t(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    merit_order_idx: nbintp,
) -> None:
    if generator_instance.fuel.trickling_flag:
        advertised_limit = min(
            generator_instance.fuel.allocated_trickling / resolution,
            generator_instance.capacity - generator_instance.dispatch_power[interval],
        )
        generator_instance.flexible_max_t = advertised_limit
        generator_instance.fuel.allocated_trickling -= advertised_limit * resolution
    else:
        generator_instance.flexible_max_t = 0.0

    update_node_flexible_max_t(generator_instance, merit_order_idx)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_live_flexible_max_t(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    merit_order_idx: nbintp,
    forward_time_flag: boolean,
) -> None:
    if forward_time_flag:
        live_pool = generator_instance.fuel.remaining_energy[interval]
    else:
        live_pool = generator_instance.fuel.remaining_energy_temp_reverse

    generator_instance.flexible_max_t = min(
        generator_instance.capacity,
        generator_instance.dispatch_power[interval] + (live_pool / resolution)
    )

    update_node_flexible_max_t(generator_instance, merit_order_idx)

    return None


@njit(fastmath=FASTMATH)
def update_node_flexible_max_t(
    generator_instance: Generator_InstanceType,
    merit_order_idx: nbintp,
):
    offset = 0.0 if merit_order_idx == 0 else generator_instance.node.flexible_max_t[merit_order_idx - 1]
    generator_instance.node.flexible_max_t[merit_order_idx] = offset + generator_instance.flexible_max_t
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_live_trickling_max_t(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
) -> None:
    if generator_instance.fuel.trickling_flag:
        live_remaining_trickling = max(
            generator_instance.fuel.remaining_energy[interval] - generator_instance.fuel.trickling_reserves,
            0.0
        )
        # Note: Delta limit for dispatch_power_update
        generator_instance.flexible_max_t = min(
            generator_instance.capacity - generator_instance.dispatch_power[interval],
            live_remaining_trickling / resolution
        )
    else:
        generator_instance.flexible_max_t = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_fuel_reserve(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    delta_power: nbfloat,
    forward_time_flag: boolean,
) -> None:
    if forward_time_flag:
        generator_instance.fuel.remaining_energy[interval] -= delta_power * resolution
    else:
        generator_instance.fuel.remaining_energy_temp_reverse -= delta_power * resolution

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def dispatch(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    merit_order_idx: nbintp,
    resolution: nbfloat,
    forward_time_flag: boolean,
) -> None:
    """
    Dispatches the flexible Generator according to its place in the merit order for the Generator.node.
    The total flexible power at that node is also updated according to the dispatch of the Generator.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    interval (nbintp): Index for the time interval during unit committment.
    merit_order_idx (nbintp): Location of the flexible Generator in the merit order at the Generator.node.
        Lower merit_order_idx indicates lower variable costs and higher priority in the merit order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the flexible Generator instance: dispatch_power, node.
    Attributes modified for referenced Generator.node: flexible_power.
    """
    prev_power = generator_instance.dispatch_power[interval]
    set_live_flexible_max_t(generator_instance, interval, resolution, merit_order_idx, forward_time_flag)

    offset = 0.0 if merit_order_idx == 0 else generator_instance.node.flexible_max_t[merit_order_idx - 1]
    new_power = min(
        max(
            generator_instance.node.netload_t
            - generator_instance.node.storage_power[interval]
            - offset,
            0.0,
        ),
        generator_instance.flexible_max_t,
    )

    delta_power = new_power - prev_power
    generator_instance.dispatch_power[interval] = new_power
    generator_instance.node.flexible_power[interval] += new_power

    if abs(delta_power) <= TOLERANCE:
        return None

    update_fuel_reserve(generator_instance, interval, resolution, delta_power, forward_time_flag)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lt_generation(
    generator_instance: Generator_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Calculate the total generation over the long-term modelling horizon for a flexible Generator. Also
    calculate the hours of operation for each unit of the Generator over the modelling horizon.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    esolution (nbfloat): A scalar containing the resolution for every time interval

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the flexible Generator instance: lt_generation, line, unit_lt_hours.
    Attributes modified for the referenced Generator.line: lt_flows.
    """
    update_lt_generation(generator_instance, generator_instance.dispatch_power, resolution)
    total_hours = 0.0
    for i in range(len(generator_instance.dispatch_power)):
        total_hours += np.ceil(generator_instance.dispatch_power[i] / generator_instance.unit_size) * resolution
    generator_instance.unit_lt_hours = total_hours
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_variable_costs(
    generator_instance: Generator_InstanceType,
    year_float: nbfloat,
) -> nbfloat:
    """
    Calculate the total variable costs for a Generator system at the end of unit committment.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    year_float (nbfloat): Number of years. Leap days provide additional fractional value.

    Returns:
    -------
    nbfloat: Total variable costs ($), equal to sum of fuel and VO&M costs.

    Side-effects:
    -------
    Attributes modified for the Generator instance: lt_costs.
    Attributes modified for the referenced Generator.lt_costs: vom, fuel.
    """
    ltcosts_m.calculate_vom(
        generator_instance.lt_costs,
        generator_instance.lt_generation,
        year_float,
        generator_instance.cost
    )
    ltcosts_m.calculate_fuel(
        generator_instance.lt_costs,
        generator_instance.lt_generation,
        year_float,
        generator_instance.unit_lt_hours,
        generator_instance.cost,
    )
    return ltcosts_m.get_variable(generator_instance.lt_costs)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fixed_costs(
    generator_instance: Generator_InstanceType,
    include_existing: bool,
) -> nbfloat:
    """
    Calculate the total fixed costs for a Generator system.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.

    Returns:
    -------
    nbfloat: Total fixed costs ($), equal to sum of annualised build and FO&M costs.

    Side-effects:
    -------
    Attributes modified for the Generator instance: lt_costs.
    Attributes modified for the referenced Generator.lt_costs: annualised_build, fom.
    """
    if include_existing:
        ltcosts_m.calculate_annualised_build_power(
            generator_instance.lt_costs,
            generator_instance.capacity,
            0.0,
            generator_instance.cost,
            "generator",
        )
    else:
        ltcosts_m.calculate_annualised_build_power(
            generator_instance.lt_costs,
            generator_instance.new_build,
            0.0,
            generator_instance.cost,
            "generator",
        )
    ltcosts_m.calculate_fom(
        generator_instance.lt_costs,
        generator_instance.capacity,
        0.0,
        generator_instance.cost,
        "generator"
    )
    return ltcosts_m.get_fixed(generator_instance.lt_costs)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_partial_cost(
    generator_instance: Generator_InstanceType,
    year_float: nbfloat,
):
    return ltcosts_m.get_partial_cost_power(
        generator_instance.new_build,
        generator_instance.capacity,
        0.0,
        generator_instance.lt_generation,
        year_float,
        generator_instance.unit_lt_hours,
        generator_instance.cost,
        "generator",
    )


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_precharge_dispatch(
    generator_instance: Generator_InstanceType,
    interval: nbintp,
    resolution: nbfloat,
    dispatch_power_update: nbfloat,
    merit_order_idx: nbintp,
) -> None:
    generator_instance.dispatch_power[interval] += dispatch_power_update
    generator_instance.node.flexible_power[interval] += dispatch_power_update

    generator_instance.node.flexible_max_t[merit_order_idx:] -= dispatch_power_update
    generator_instance.node.precharge_surplus -= dispatch_power_update

    return None
