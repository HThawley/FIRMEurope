# type: ignore
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
    generator_instance: Generator_InstanceType,
    new_build_power_capacity: float64,
    interval_resolutions: float64[:],
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
    """
    Load the capacity factor trace and flexible annual constraint data to the Generator instance. This is done
    before solving a Scenario.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    generation_trace (float64[:]): Array containing the time-series capacity factor trace for the Generator. Each element
        provides the capacity factor for a time interval.
    annual_constraints (float64[:]): Array containing the annual generation constraints for a flexible Generator.
        Each element provides the maximum annual generation (GWh) for a given year for the Generator.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: data_status, data, annual_constraints_data, lt_generation.
    Attributes modified for the referenced Generator.line: lt_flows.
    Attributes modified for the referenced Generator.node: residual_load.
    """
    generator_instance.data = generation_trace
    generator_instance.annual_constraints_data = annual_constraints
    generator_instance.data_status = True

    update_residual_load(generator_instance, generator_instance.initial_capacity, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
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
    generator_instance.data = np.empty((0,), dtype=np.float64)
    generator_instance.annual_constraints_data = np.empty((0,), dtype=np.float64)
    generator_instance.data_status = False
    return None


@njit(fastmath=FASTMATH)
def get_data(
    generator_instance: Generator_InstanceType,
    data_type: unicode_type,
) -> float64[:]:
    """
    Gets the specified data_type from the Generator instance.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    data_type (unicode_type): String associated with the data array.

    Returns:
    -------
    float64[:]: The data array associated with data_type.

    Raises:
    -------
    RuntimeError: Raised if data_status is False or if data_type does not correspond
        to any data arrays for the Generator jitclass.
    """
    if not generator_instance.data_status:
        raise_getting_unloaded_data_error()

    if data_type == "annual_constraints_data":
        return generator_instance.annual_constraints_data
    elif data_type == "trace":
        return generator_instance.data
    else:
        raise RuntimeError("Invalid data_type argument for Generator.get_data(data_type).")
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(
    generator_instance: Generator_InstanceType,
    intervals_count: int64,
) -> None:
    """
    Memory associated with endogenous time-series data for a flexible Generator is only allocated after a dynamic copy of
    the Generator instance is created. This is to minimise memory usage of the static instances.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): A dynamic instance of the Generator jitclass.
    intervals_count (int64): Total number of time intervals over the modelling horizon.

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
    generator_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    if len(get_data(generator_instance, "annual_constraints_data")) > 0:
        generator_instance.remaining_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def update_residual_load(
    generator_instance: Generator_InstanceType,
    added_capacity: float64,
    interval_resolutions: float64[:],
) -> None:
    if get_data(generator_instance, "trace").shape[0] > 0 and added_capacity > 0.0:
        new_trace = get_data(generator_instance, "trace") * added_capacity
        node_m.get_data(generator_instance.node, "residual_load")[:] -= new_trace
        update_lt_generation(generator_instance, new_trace, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def update_lt_generation(
    generator_instance: Generator_InstanceType,
    generation_trace: float64[:],
    interval_resolutions: float64[:],
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
def dispatch(
    generator_instance: Generator_InstanceType,
    interval: int64,
    merit_order_idx: int64,
) -> None:
    """
    Dispatches the flexible Generator according to its place in the merit order for the Generator.node.
    The total flexible power at that node is also updated according to the dispatch of the Generator.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    interval (int64): Index for the time interval during unit committment.
    merit_order_idx (int64): Location of the flexible Generator in the merit order at the Generator.node.
        Lower merit_order_idx indicates lower variable costs and higher priority in the merit order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the flexible Generator instance: dispatch_power, node.
    Attributes modified for referenced Generator.node: flexible_power.
    """
    if merit_order_idx == 0:
        generator_instance.dispatch_power[interval] = min(
            max(
                generator_instance.node.netload_t
                - generator_instance.node.storage_power[interval],
                0.0
            ),
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
def calculate_lt_generation(
    generator_instance: Generator_InstanceType,
    interval_resolutions: float64[:],
) -> None:
    """
    Calculate the total generation over the long-term modelling horizon for a flexible Generator. Also
    calculate the hours of operation for each unit of the Generator over the modelling horizon.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the flexible Generator instance: lt_generation, line, unit_lt_hours.
    Attributes modified for the referenced Generator.line: lt_flows.
    """
    update_lt_generation(generator_instance, generator_instance.dispatch_power, interval_resolutions)
    generator_instance.unit_lt_hours = sum(
        np.ceil(generator_instance.dispatch_power / generator_instance.unit_size) * interval_resolutions
    )
    return None


@njit(fastmath=FASTMATH)
def calculate_variable_costs(
    generator_instance: Generator_InstanceType,
    year_float: float64,
) -> float64:
    """
    Calculate the total variable costs for a Generator system at the end of unit committment.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    year_float (float64): Number of years. Leap days provide additional fractional value.

    Returns:
    -------
    float64: Total variable costs ($), equal to sum of fuel and VO&M costs.

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


@njit(fastmath=FASTMATH)
def calculate_fixed_costs(
    generator_instance: Generator_InstanceType,
) -> float64:
    """
    Calculate the total fixed costs for a Generator system.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.

    Returns:
    -------
    float64: Total fixed costs ($), equal to sum of annualised build and FO&M costs.

    Side-effects:
    -------
    Attributes modified for the Generator instance: lt_costs.
    Attributes modified for the referenced Generator.lt_costs: annualised_build, fom.
    """
    ltcosts_m.calculate_annualised_build(
        generator_instance.lt_costs,
        0.0,
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


@njit(fastmath=FASTMATH)
def initialise_deficit_block(
    generator_instance: Generator_InstanceType,
    interval: int64,
) -> None:
    """
    Upon resolving a deficit block, initialise the temporary remaining energy,
    max remaining energy, and min remaining energy values for a flexible Generator. These temporary
    variables are updated while performing unit committment in the reverse time direction for each time interval
    in the deficit block.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    interval (int64): Index for the first time interval immediately following the deficit block.
        During unit committment for the deficit block, time intervals will decrease in value (reverse
        time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: remaining_energy_temp_reverse, deficit_block_max_energy,
        deficit_block_min_energy.
    """
    generator_instance.remaining_energy_temp_reverse = generator_instance.remaining_energy[interval - 1]
    generator_instance.deficit_block_max_energy = generator_instance.remaining_energy_temp_reverse
    generator_instance.deficit_block_min_energy = generator_instance.remaining_energy_temp_reverse


@njit(fastmath=FASTMATH)
def update_deficit_block_bounds(
    generator_instance: Generator_InstanceType,
    remaining_energy: float64,
) -> None:
    """
    Update the temporary minimum and maximum remaining energy values for the flexible Generator in the
    deficit block. These values are updated in each time interval for the deficit block. The minimum
    and maximum remaining energies are used to define the trickling reserves that must be retained in
    the precharging period leading up to the deficit block such that the Generator is capable of dispatching
    during the deficit block.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    remaining_energy (float64): The remaining energy in a time interval that a flexible Generator has
        available for the calendar year such that it complies with its annual generation constraint.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: deficit_block_max_energy, deficit_block_min_energy.
    """
    generator_instance.deficit_block_min_energy = min(generator_instance.deficit_block_min_energy, remaining_energy)
    generator_instance.deficit_block_max_energy = max(generator_instance.deficit_block_max_energy, remaining_energy)
    return None


@njit(fastmath=FASTMATH)
def initialise_precharging_flags(
    generator_instance: Generator_InstanceType,
    interval: int64,
) -> None:
    """
    Initialises the trickling flag for a flexible Generator once precharging in the lead-up to the deficit
    block begins. The trickling flag is True if the flexible Generator has sufficient energy remaining for
    the calendar year such that it still retains the trickling reserves required to dispatch in the subsequent
    deficit block. When the trickling flag is True, a flexible Generator is assumed to be available for
    trickle charging a Storage precharger.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    interval (int64): Index for the first time interval of the deficit block (immediately following the
        precharging period). Time intervals during the precharging period will decrease in value (reverse
        time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: trickling_flag.
    """
    generator_instance.trickling_flag = (
        generator_instance.remaining_energy[interval] - generator_instance.trickling_reserves > TOLERANCE
    )
    return None


@njit(fastmath=FASTMATH)
def update_precharging_flags(
    generator_instance: Generator_InstanceType,
    interval: int64,
) -> None:
    """
    At the start of a time interval within the precharging period, the remaining trickling reserves and
    trickling flag for the flexible Generator is updated. The remaining trickling reserves define the
    amount of energy available for trickle charging, ensuring that the Generator retains sufficient
    reserves to dispatch during the deficit block immediately after the precharging period.

    The trickling flag is True if the flexible Generator has sufficient energy remaining for
    the calendar year such that it still retains the trickling reserves required to dispatch in the subsequent
    deficit block. When the trickling flag is True, a flexible Generator is assumed to be available for
    trickle charging a Storage precharger.

    Parameters:
    -------
    generator_instance (Generator_InstanceType): An instance of the Generator jitclass.
    interval (int64): Index for the current time interval in the precharging period. Time intervals during
        the precharging period will decrease in value (reverse time).

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Generator instance: remaining_trickling_reserves, trickling_flag.
    """
    generator_instance.remaining_trickling_reserves = max(
        generator_instance.remaining_energy[interval] - generator_instance.trickling_reserves, 0.0
    )
    generator_instance.trickling_flag = (
        generator_instance.remaining_trickling_reserves > TOLERANCE
    ) and generator_instance.trickling_flag

    return None


@njit(fastmath=FASTMATH)
def set_precharging_max_t(
    generator_instance: Generator_InstanceType,
    interval: int64,
    resolution: float64,
    merit_order_idx: int64,
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
