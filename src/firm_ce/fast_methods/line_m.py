# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.exceptions import (
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, nbfloat, npfloat, nbint, nbintp, unicode_type
from firm_ce.fast_methods import ltcosts_m
from firm_ce.system.topology import Line, Line_InstanceType, Node, Node_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    line_instance: Line_InstanceType, nodes_typed_dict: DictType(nbintp, Node_InstanceType), line_type: unicode_type
) -> Line_InstanceType:
    """
    A 'static' instance of the Line jitclass (Line.static_instance=True) is copied
    and marked as a 'dynamic' instance (Line.static_instance=False).

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

    Dynamic copies of Node instances are required for major_lines. For minor_lines, a generic Node instance
    is created, named MINOR_NODE.

    Parameters:
    -------
    line_instance (Line_InstanceType): A static instance of the Line jitclass.
    nodes_typed_dict (DictType(nbintp, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_type (unicode_type): Text that specifies if the Line is a 'major_line' or 'minor_line'.

    Returns:
    -------
    Line_InstanceType: A dynamic instance of the Line jitclass.
    """
    if line_type == "major":
        node_start_copy = nodes_typed_dict[line_instance.node_start.order]
        node_end_copy = nodes_typed_dict[line_instance.node_end.order]
    elif line_type == "minor":
        node_start_copy = Node(False, -1, -1, "MINOR_NODE")
        node_end_copy = Node(False, -1, -1, "MINOR_NODE")

    line_copy = Line(
        False,
        line_instance.id,
        line_instance.order,
        line_instance.name,
        line_instance.length,
        node_start_copy,
        node_end_copy,
        line_instance.loss_factor,
        line_instance.max_build,
        line_instance.min_build,
        line_instance.capacity,
        line_instance.unit_type,
        line_type == "major",
        line_instance.near_optimum_check,
        line_instance.group,
        line_instance.cost,  # This remains static
    )
    line_copy.candidate_x_idx = line_instance.candidate_x_idx
    return line_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacity(line_instance: Line_InstanceType, new_build_power_capacity: nbfloat) -> None:
    """
    Takes a new_build_power_capacity and adds it to the existing capacity and new_build attributes.

    Parameters:
    -------
    line_instance (Line_InstanceType): A dynamic instance of the Line jitclass.
    new_build_power_capacity (nbfloat): Additional capacity [GW] to be built for the line.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Line instance: capacity, new_build.
    """
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.capacity += new_build_power_capacity
    line_instance.new_build += new_build_power_capacity
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(line_instance: Line_InstanceType, intervals_count: nbint) -> None:
    """
    Memory associated with time-series data for a Line is only allocated after a dynamic copy of the Line instance
    is created. This is to minimise memory usage of the static instances.

    Parameters:
    -------
    line_instance (Line_InstanceType): A dynamic instance of the Line jitclass.
    intervals_count (nbint): Total number of time intervals in the unit committment formulation.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Line instance: flows
    """
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.flows = np.zeros(intervals_count, dtype=npfloat)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lt_flow(line_instance: Line_InstanceType, interval_resolutions: nbfloat[:]) -> None:
    line_instance.lt_flows = sum(np.abs(line_instance.flows) * interval_resolutions)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_variable_costs(
    line_instance: Line_InstanceType,
    year_float: nbfloat,
) -> nbfloat:
    """
    Calculate the total variable costs for a Line system at the end of unit committment.

    Parameters:
    -------
    line_instance (Line_InstanceType): An instance of the Line jitclass.
    year_float (nbfloat): Number of years. Leap days provide additional fractional value.

    Returns:
    -------
    nbfloat: Total variable costs ($), equal to sum of fuel and VO&M costs.

    Side-effects:
    -------
    Attributes modified for the Line instance: lt_costs.
    Attributes modified for the referenced Line.lt_costs: vom, fuel.
    """
    ltcosts_m.calculate_vom(
        line_instance.lt_costs,
        line_instance.lt_flows,
        year_float,
        line_instance.cost
    )
    ltcosts_m.calculate_fuel(
        line_instance.lt_costs,
        line_instance.lt_flows,
        year_float,
        0,
        line_instance.cost
    )
    return ltcosts_m.get_variable(line_instance.lt_costs)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fixed_costs(
    line_instance: Line_InstanceType,
    include_existing: bool,
) -> nbfloat:
    """
    Calculate the total fixed costs for a Line system.

    Parameters:
    -------
    line_instance (Line_InstanceType): An instance of the Line jitclass.

    Returns:
    -------
    nbfloat: Total fixed costs ($), equal to sum of annualised build and FO&M costs.

    Side-effects:
    -------
    Attributes modified for the Line instance: lt_costs.
    Attributes modified for the referenced Line.lt_costs: annualised_build, fom.
    """
    if include_existing:
        ltcosts_m.calculate_annualised_build_power(
            line_instance.lt_costs,
            line_instance.capacity,
            line_instance.length,
            line_instance.cost,
            "line",
        )
    else:
        ltcosts_m.calculate_annualised_build_power(
            line_instance.lt_costs,
            line_instance.new_build,
            line_instance.length,
            line_instance.cost,
            "line",
        )

    ltcosts_m.calculate_fom(
        line_instance.lt_costs,
        line_instance.capacity,
        line_instance.length,
        line_instance.cost,
        "line"
    )
    return ltcosts_m.get_fixed(line_instance.lt_costs)


@njit(fastmath=FASTMATH)
def get_partial_cost(
    line_instance: Line_InstanceType,
    year_float: nbfloat,
):
    return ltcosts_m.get_partial_cost_power(
        line_instance.new_build,
        line_instance.capacity,
        line_instance.length,
        line_instance.lt_flows,
        year_float,
        0.0,
        line_instance.cost,
        "line",
    )
