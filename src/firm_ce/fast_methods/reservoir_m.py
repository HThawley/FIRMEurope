# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import ltcosts_m
from firm_ce.system.components import Reservoir, Reservoir_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType


@njit(fastmath=FASTMATH)
def create_dynamic_copy(
    reservoir_instance: Reservoir_InstanceType,
    nodes_typed_dict: DictType(int64, Node_InstanceType),
    lines_typed_dict: DictType(int64, Line_InstanceType),
) -> Reservoir_InstanceType:
    """
    A 'static' instance of the Reservoir jitclass (Reservoir.static_instance=True) is copied
    and marked as a 'dynamic' instance (Reservoir.static_instance=False).

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
    reservoir_instance (Reservoir_InstanceType): A static instance of the Reservoir jitclass.
    nodes_typed_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_typed_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.

    Returns:
    -------
    Reservoir_InstanceType: A dynamic instance of the Reservoir jitclass.
    """
    node_copy = nodes_typed_dict[reservoir_instance.node.order]
    line_copy = lines_typed_dict[reservoir_instance.line.order]

    reservoir_copy = Reservoir(
        False,
        reservoir_instance.id,
        reservoir_instance.order,
        reservoir_instance.name,
        reservoir_instance.unit_size,
        reservoir_instance.power_capacity,
        reservoir_instance.energy_capacity,
        reservoir_instance.duration,
        reservoir_instance.discharge_efficiency,
        reservoir_instance.max_build_p,
        reservoir_instance.max_build_e,
        reservoir_instance.min_build_p,
        reservoir_instance.min_build_e,
        reservoir_instance.unit_type,
        reservoir_instance.near_optimum_check,
        node_copy,
        reservoir_instance.fuel,  # This remains static
        line_copy,
        reservoir_instance.group,
        reservoir_instance.cost,  # This remains static
    )

    reservoir_copy.candidate_p_x_idx = reservoir_instance.candidate_p_x_idx
    reservoir_copy.candidate_e_x_idx = reservoir_instance.candidate_e_x_idx

    reservoir_copy.data_status = reservoir_instance.data_status
    reservoir_copy.data = reservoir_instance.data  # This remains static
    reservoir_copy.lt_generation = reservoir_instance.lt_generation

    return reservoir_copy


@njit(fastmath=FASTMATH)
def build_capacity(
    reservoir_instance: Reservoir_InstanceType,
    new_build_capacity: float64,
    capacity_type: unicode_type,
) -> None:
    """
    Takes a new_build_capacity (either power capacity or energy capacity) and adds it to the corresponding existing capacity
    and new build attributes. New build power capacity also adds to the capacity and new_build of the minor line that connects
    the Reservoir to the transmission network.

    If the Reservoir.duration is equal to zero, then the energy capacity is assumed to be independent of power capacity and is
    built separately. For non-zero Reservoir.duration values, energy capacity is assumed to be dependent upon power capacity
    and is built at the same time as new build power capacity.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): A dynamic instance of the Reservoir jitclass.
    new_build_capacity (float64): Additional capacity [GW or GWh as determined by capacity_type value] to be built for the
        Reservoir.
    capacity_type (unicode_type): String that specifies new build capacity is either "power" or "energy" capacity.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: power_capacity, new_build_p, line, energy_capacity, new_build_e.
    Attributes modified for the referenced Reservoir.line: capacity, new_build.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if reservoir_instance.static_instance:
        raise_static_modification_error()
    if capacity_type == "power":
        reservoir_instance.power_capacity += new_build_capacity
        reservoir_instance.new_build_p += new_build_capacity
        reservoir_instance.line.capacity += new_build_capacity
        reservoir_instance.line.new_build += new_build_capacity

        if reservoir_instance.duration > 0:
            reservoir_instance.energy_capacity += new_build_capacity * reservoir_instance.duration
            reservoir_instance.new_build_e += new_build_capacity * reservoir_instance.duration

    if capacity_type == "energy":
        if reservoir_instance.duration == 0:
            reservoir_instance.energy_capacity += new_build_capacity
            reservoir_instance.new_build_e += new_build_capacity
    return None


@njit(fastmath=FASTMATH)
def load_data(
    reservoir_instance: Reservoir_InstanceType,
    inflow_trace: float64[:],
) -> None:
    """
    Load the inflow trace data to the Reservoir instance. This is done before solving a Scenario.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    inflow_trace (float64[:]): Array containing the time-series inflow trace for the Reservoir. Each element
        provides the capacity factor for a time interval.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: data_status, data, lt_generation.
    Attributes modified for the referenced Reservoir.line: lt_flows.
    Attributes modified for the referenced Reservoir.node: residual_load.
    """
    reservoir_instance.data = inflow_trace
    reservoir_instance.data_status = "loaded"

    return None


@njit(fastmath=FASTMATH)
def unload_data(reservoir_instance: Reservoir_InstanceType) -> None:
    """
    Unload the inflow data from the Reservoir instance. This is done after solving a Scenario to reduce memory usage.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: data_status, data, annual_constraints_data.
    """
    reservoir_instance.data = np.empty((0,), dtype=np.float64)
    reservoir_instance.data_status = "unloaded"
    return None


@njit(fastmath=FASTMATH)
def get_data(
    reservoir_instance: Reservoir_InstanceType,
    data_type: unicode_type,
) -> float64[:]:
    """
    Gets the specified data_type from the Reservoir instance.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    data_type (unicode_type): String associated with the data array.

    Returns:
    -------
    float64[:]: The data array associated with data_type.

    Raises:
    -------
    RuntimeError: Raised if data_status is "unloaded" or if data_type does not correspond
        to any data arrays for the Reservoir jitclass.
    """
    if reservoir_instance.data_status == "unloaded":
        raise_getting_unloaded_data_error()

    if data_type == "inflow":
        return reservoir_instance.data
    else:
        raise RuntimeError("Invalid data_type argument for Reservoir.get_data(data_type).")


@njit(fastmath=FASTMATH)
def allocate_memory(reservoir_instance: Reservoir_InstanceType, intervals_count: int64) -> None:
    """
    Memory associated with endogenous time-series data for a Reservoir system is only allocated after a dynamic copy of
    the Reservoir instance is created. This is to minimise memory usage of the static instances.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): A dynamic instance of the Reservoir jitclass.
    intervals_count (int64): Total number of time intervals over the modelling horizon.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: dispatch_power, stored_energy.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if reservoir_instance.static_instance:
        raise_static_modification_error()
    reservoir_instance.dispatch_power = np.zeros(intervals_count, dtype=np.float64)
    reservoir_instance.stored_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def initialise_stored_energy(
    reservoir_instance: Reservoir_InstanceType,
) -> None:
    """
    Initialise the stored energy for a Storage system. Called at the start of the modelling period.

    Parameters:
    -------
    reservoir_instance (Storage_InstanceType): An instance of the Reservoir jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: stored_energy. The initial stored energy is stored
    in the final time interval for simplicity. It is overwritten upon performing balancing for final time interval.

    Raises:
    -------
    RuntimeError: Raised if static_instance is True. Only dynamic instances can be modified by this pseudo-method.
    """
    if reservoir_instance.static_instance:
        raise_static_modification_error()
    # TODO:  Make it possible for user to define custom value for initial stored energy in future
    reservoir_instance.stored_energy[-1] = 0.5 * reservoir_instance.energy_capacity
    return None


@njit(fastmath=FASTMATH)
def set_reservoir_max_t(
    reservoir_instance: Reservoir_InstanceType,
    interval: int64,
    resolution: float64,
    merit_order_idx: int64,
    forward_time_flag: boolean,
) -> None:
    """
    Set the maximum possible dispatch power for a Reservoir system at the start of a time interval.
    The maximum dispatch power is constrained by the power capacity and the stored energy (discharging)
    or energy capacity (charging). Within a deficit block, the temporary stored_energy_temp_reverse
    is used instead of the stored_energy since permanent reservoir of this value is not required during
    reverse time processes.

    Reservoir systems are dispatched according to a merit order at each node. The cumulative maximum
    discharge/charge power for each step in the merit order is stored in an array in the Reservoir.node.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    interval (int64): Index for the time interval during unit committment.
    resolution (float64): Temporal resolution for the time interval (hours).
    merit_order_idx (int64): Location of the Reservoir system in the merit order at the Reservoir.node.
        Lower merit_order_idx indicates shorter duration and higher priority in the merit order.
    forward_time_flag (boolean): True value indicates unit committment is operating in the forwards
        time direction. False indicates reverse time during deficit block or precharging processes.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: reservoir_max_t, node.
    Attributes modified for referenced Reservoir.node: reservoir_max_t.
    """
    if forward_time_flag:
        reservoir_instance.discharge_max_t = min(
            reservoir_instance.power_capacity,
            reservoir_instance.stored_energy[interval - 1] * reservoir_instance.discharge_efficiency / resolution,
        )

    else:
        reservoir_instance.discharge_max_t = min(
            reservoir_instance.power_capacity,
            (reservoir_instance.energy_capacity - reservoir_instance.stored_energy_temp_reverse)
            * reservoir_instance.discharge_efficiency
            / resolution,
        )

    if merit_order_idx == 0:
        # node does not know or care about reservoir charging
        reservoir_instance.node.reservoir_max_t[0] = reservoir_instance.discharge_max_t
    else:
        reservoir_instance.node.reservoir_max_t[merit_order_idx] = (
            reservoir_instance.node.reservoir_max_t[merit_order_idx - 1] + reservoir_instance.discharge_max_t
        )
    return None


@njit(fastmath=FASTMATH)
def dispatch(
    reservoir_instance: Reservoir_InstanceType,
    interval: int64,
    merit_order_idx: int64,
) -> None:
    """
    Dispatches the flexible Reservoir according to its place in the merit order for the Reservoir.node.
    The total flexible power at that node is also updated according to the dispatch of the Reservoir.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    interval (int64): Index for the time interval during unit committment.
    merit_order_idx (int64): Location of the flexible Reservoir in the merit order at the Reservoir.node.
        Lower merit_order_idx indicates lower variable costs and higher priority in the merit order.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the flexible Reservoir instance: dispatch_power, node.
    Attributes modified for referenced Reservoir.node: flexible_power.
    """
    if merit_order_idx == 0:
        reservoir_instance.dispatch_power[interval] = min(
            max(reservoir_instance.node.netload_t - reservoir_instance.node.storage_power[interval], 0.0),
            reservoir_instance.discharge_max_t,
        )
    else:
        reservoir_instance.dispatch_power[interval] = min(
            max(
                reservoir_instance.node.netload_t
                - reservoir_instance.node.storage_power[interval]
                - reservoir_instance.node.discharge_max_t[merit_order_idx - 1],
                0.0,
            ),
            reservoir_instance.discharge_max_t,
        )
    reservoir_instance.node.reservoir_power[interval] += reservoir_instance.dispatch_power[interval]
    return None


@njit(fastmath=FASTMATH)
def update_stored_energy(
    reservoir_instance: Reservoir_InstanceType,
    interval: int64,
    resolution: float64,
    forward_time_flag: boolean,
) -> None:
    """
    Once the dispatch power for a Reservoir system has been established for a time interval, the stored energy
    is updated. A temporary value is stored when unit committment is being performed in reverse time within a
    deficit block.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    interval (int64): Index for the time interval during unit committment.
    resolution (float64): Temporal resolution for the time interval (hours).
    forward_time_flag (boolean): True value indicates unit committment is operating in the forwards
        time direction. False indicates reverse time during deficit block processes.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: stored_energy, stored_energy_temp_reverse.
    """
    if forward_time_flag:
        reservoir_instance.stored_energy[interval] = (
            # starting point
            reservoir_instance.stored_energy[interval - 1]
            # less discharge energy and discharge efficiency losses
            - max(reservoir_instance.dispatch_power[interval], 0) / reservoir_instance.discharge_efficiency * resolution
            # plus inflows
            + min(
                # inflow energy less efficiency losses
                reservoir_instance.data[interval] * reservoir_instance.charge_efficiency,
                # clipped by energy capacity of reservoir
                (
                    # simple remaining energy capacity
                    reservoir_instance.energy_capacity
                    - reservoir_instance.stored_energy[interval - 1]
                    # plus the energy capacity freed up by power dipstached this time interval
                    + (
                        reservoir_instance.dispatch_power[interval]
                        / reservoir_instance.discharge_efficiency
                        * resolution
                    )
                )
                / reservoir_instance.charge_efficiency,
            )
            / resolution
        )
    else:
        # + dispatch
        reservoir_instance.stored_energy_temp_reverse += (
            max(reservoir_instance.dispatch_power[interval], 0) / reservoir_instance.discharge_efficiency * resolution
        )
        # - charge
        reservoir_instance.stored_energy_temp_reverse -= (
            # charge is inflows clipped by energy capacity
            min(
                # inflows (energy)
                reservoir_instance.data[interval] / reservoir_instance.charge_efficiency,
                # energy capacity + energy dipsatched in same time interval
                (
                    reservoir_instance.stored_energy_temp_reverse
                    + (
                        reservoir_instance.dispatch_power[interval]
                        / reservoir_instance.discharge_efficiency
                        * resolution
                    )
                )
                / reservoir_instance.charge_efficiency,
            )
            # energy to power
            / resolution
        )
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_generation(
    reservoir_instance: Reservoir_InstanceType,
    interval_resolutions: float64[:],
) -> None:
    """
    Calculate the total energy discharged over the long-term modelling horizon for a Reservoir system.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    interval_resolutions (float64[:]): A 1-dimensional array containing the resolution for every time interval
        in the unit committment formulation (hours per time interval). An array is used instead of a single
        scalar value to allow for variable time step simplified balancing methods to be developed in future.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: lt_discharge, line.
    Attributes modified for the referenced Reservoir.line: lt_flows.
    """
    reservoir_instance.lt_generation = sum(np.maximum(reservoir_instance.dispatch_power, 0) * interval_resolutions)

    reservoir_instance.line.lt_flows += sum(np.abs(reservoir_instance.dispatch_power) * interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def calculate_variable_costs(reservoir_instance: Reservoir_InstanceType) -> float64:
    """
    Calculate the total variable costs for a Reservoir system at the end of unit committment.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.

    Returns:
    -------
    float64: Total variable costs ($), equal to sum of fuel and VO&M costs.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: lt_costs.
    Attributes modified for the referenced Reservoir.lt_costs: vom, fuel.
    """
    ltcosts_m.calculate_vom(reservoir_instance.lt_costs, reservoir_instance.lt_generation, reservoir_instance.cost)
    ltcosts_m.calculate_fuel(
        reservoir_instance.lt_costs,
        reservoir_instance.lt_generation,
        reservoir_instance.unit_lt_hours,
        reservoir_instance.cost,
    )
    return ltcosts_m.get_variable(reservoir_instance.lt_costs)


@njit(fastmath=FASTMATH)
def calculate_fixed_costs(
    reservoir_instance: Reservoir_InstanceType,
    years_float: float64,
    year_count: int64,
) -> float64:
    """
    Calculate the total fixed costs for a Reservoir system.

    Parameters:
    -------
    reservoir_instance (Reservoir_InstanceType): An instance of the Reservoir jitclass.
    years_float (float64): Number of non-leap years. Leap days provide additional fractional value.
    year_count (int64): Total number of years across modelling horizon.

    Returns:
    -------
    float64: Total fixed costs ($), equal to sum of annualised build and FO&M costs.

    Side-effects:
    -------
    Attributes modified for the Reservoir instance: lt_costs.
    Attributes modified for the referenced Reservoir.lt_costs: annualised_build, fom.
    """
    ltcosts_m.calculate_annualised_build(
        reservoir_instance.lt_costs,
        reservoir_instance.new_build_e,
        reservoir_instance.new_build_p,
        0.0,
        reservoir_instance.cost,
        year_count,
        "reservoir",
    )
    ltcosts_m.calculate_fom(
        reservoir_instance.lt_costs,
        reservoir_instance.power_capacity,
        years_float,
        0.0,
        reservoir_instance.cost,
        "reservoir",
    )
    return ltcosts_m.get_fixed(reservoir_instance.lt_costs)


# TODO: precharging rules
