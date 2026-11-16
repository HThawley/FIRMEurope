# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, nbfloat, npfloat, nbint, nbintp, unicode_type
from firm_ce.scalar.entities.topology import Line, Line_InstanceType, Node, Node_InstanceType
from firm_ce.scalar.methods import ltcosts_m


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    line_instance: Line_InstanceType, nodes_typed_dict: DictType(nbintp, Node_InstanceType), line_type: unicode_type
) -> Line_InstanceType:
    if line_type == "major":
        node_start_copy = nodes_typed_dict[line_instance.node_start.order]
        node_end_copy = nodes_typed_dict[line_instance.node_end.order]
    elif line_type == "minor":
        node_start_copy = Node(False, -1, -1, "MINOR_NODE", 0.0)
        node_end_copy = Node(False, -1, -1, "MINOR_NODE", 0.0)

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
        line_instance.cost,  # This remains static
    )
    line_copy.unit_type_idx = line_instance.unit_type_idx
    line_copy.candidate_x_idx = line_instance.candidate_x_idx
    line_copy.relative_scaler = line_instance.relative_scaler
    return line_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacity(
    line_instance: Line_InstanceType,
    x_val_capacity: nbfloat,
) -> None:
    if line_instance.static_instance:
        raise_static_modification_error()

    new_build_power_capacity = x_val_capacity * line_instance.relative_scaler

    line_instance.capacity += new_build_power_capacity
    line_instance.new_build += new_build_power_capacity
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(line_instance: Line_InstanceType, intervals_count: nbint) -> None:
    if line_instance.static_instance:
        raise_static_modification_error()
    line_instance.flows = np.zeros(intervals_count, dtype=npfloat)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lt_flow(line_instance: Line_InstanceType, resolution: nbfloat) -> None:
    line_instance.lt_flows = np.abs(line_instance.flows).sum() * resolution
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_variable_costs(
    line_instance: Line_InstanceType,
    year_float: nbfloat,
) -> nbfloat:
    ltcosts_m.calculate_vom(
        line_instance.lt_costs,
        line_instance.lt_flows,
        year_float,
        line_instance.cost,
    )
    ltcosts_m.calculate_fuel(
        line_instance.lt_costs,
        line_instance.lt_flows,
        year_float,
        0,
        line_instance.cost,
    )
    return ltcosts_m.get_variable(line_instance.lt_costs)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fixed_costs(
    line_instance: Line_InstanceType,
    include_existing: bool,
) -> nbfloat:
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
        "line",
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
