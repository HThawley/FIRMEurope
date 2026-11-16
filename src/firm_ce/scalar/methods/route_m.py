# type: ignore
from typing import Tuple

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, TypedList, boolean, nbfloat, nbintp
from firm_ce.scalar.entities.topology import Line_InstanceType, Node_InstanceType, Route, Route_InstanceType


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    route_instance: Route_InstanceType,
    nodes_typed_dict: DictType(nbintp, Node_InstanceType),
    lines_typed_dict: DictType(nbintp, Line_InstanceType),
) -> Route_InstanceType:
    nodes_list_copy = TypedList.empty_list(Node_InstanceType)
    lines_list_copy = TypedList.empty_list(Line_InstanceType)

    for node in route_instance.nodes:
        nodes_list_copy.append(nodes_typed_dict[node.order])

    for line in route_instance.lines:
        lines_list_copy.append(lines_typed_dict[line.order])

    return Route(
        False,
        nodes_typed_dict[route_instance.initial_node.order],
        nodes_list_copy,
        lines_list_copy,
        route_instance.line_directions.copy(),
        route_instance.legs,
    )


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_contains_line(
    route_instance: Route_InstanceType,
    new_line: Line_InstanceType,
) -> boolean:
    for line in route_instance.lines:
        if new_line.order == line.order:
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_contains_node(
    route_instance: Route_InstanceType,
    new_node: Node_InstanceType,
) -> boolean:
    for node in route_instance.nodes:
        if new_node.order == node.order:
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_forward_outflow(current_flow: nbfloat, exist_flow: nbfloat, capacity: nbfloat, l_eff: nbfloat) -> nbfloat:
    """Helper to determine the constrained outflow during the forward pass."""
    max_cancel_inflow = max(0.0, -exist_flow) * l_eff

    avaliable_capacity = max_cancel_inflow + capacity - max(0.0, exist_flow)
    constrained_inflow = min(current_flow, avaliable_capacity)

    inflow_cancel = min(constrained_inflow, max_cancel_inflow)
    inflow_new = constrained_inflow - inflow_cancel

    return (inflow_cancel / l_eff) + (inflow_new * l_eff)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_backward_inflow_and_delta(required_flow: nbfloat, exist_flow: nbfloat, l_eff: nbfloat) -> Tuple[nbfloat, nbfloat]:
    """Helper to determine the required inflow and line flow delta during the backward pass."""
    outflow_cancel = min(required_flow, max(0.0, -exist_flow))
    outflow_new = required_flow - outflow_cancel

    required_inflow = (outflow_cancel * l_eff) + (outflow_new / l_eff)
    flow_delta = outflow_cancel + (outflow_new / l_eff)

    return required_inflow, flow_delta


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_flow_update(
    route_instance: Route_InstanceType,
    interval: nbintp,
) -> None:
    current_flow = route_instance.nodes[-1].temp_surplus

    for leg in range(route_instance.legs, -1, -1):
        line = route_instance.lines[leg]

        exist_flow = (line.flows[interval] + line.temp_leg_flows) * route_instance.line_directions[leg]
        current_flow = get_forward_outflow(current_flow, exist_flow, line.capacity, line.efficiency)

    route_instance.flow_update = current_flow
    route_instance.initial_node.available_imports += current_flow

    required_flow = current_flow

    for leg in range(route_instance.legs + 1):
        line = route_instance.lines[leg]

        exist_flow = (line.flows[interval] + line.temp_leg_flows) * route_instance.line_directions[leg]
        required_inflow, flow_delta = get_backward_inflow_and_delta(required_flow, exist_flow, line.efficiency)

        line.temp_leg_flows += route_instance.line_directions[leg] * flow_delta
        required_flow = required_inflow

    route_instance.nodes[-1].temp_surplus -= required_flow
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_exports(
    route_instance: Route_InstanceType,
    interval: nbintp,
) -> None:
    required_flow = route_instance.flow_update

    for leg in range(route_instance.legs + 1):
        line = route_instance.lines[leg]

        exist_flow = line.flows[interval] * route_instance.line_directions[leg]
        required_inflow, flow_delta = get_backward_inflow_and_delta(required_flow, exist_flow, line.efficiency)

        line.flows[interval] += route_instance.line_directions[leg] * flow_delta
        required_flow = required_inflow

    route_instance.nodes[-1].imports_exports[interval] -= required_flow
    route_instance.nodes[-1].surplus -= required_flow

    return None
