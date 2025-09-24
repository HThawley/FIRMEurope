from firm_ce.common.constants import FASTMATH, NP_FLOAT_MAX
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, TypedList, boolean, float64, int64
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType, Route, Route_InstanceType


@njit(fastmath=FASTMATH)
def create_dynamic_copy(
    route_instance: Route_InstanceType,
    nodes_typed_dict: DictType(int64, Node_InstanceType),
    lines_typed_dict: DictType(int64, Line_InstanceType),
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


@njit(fastmath=FASTMATH)
def check_contains_line(route_instance: Route_InstanceType, new_line: Line_InstanceType) -> boolean:
    for line in route_instance.lines:
        if new_line.order == line.order:
            return True
    return False


@njit(fastmath=FASTMATH)
def check_contains_node(route_instance: Route_InstanceType, new_node: Node_InstanceType) -> boolean:
    for node in route_instance.nodes:
        if new_node.order == node.order:
            return True
    return False


@njit(fastmath=FASTMATH)
def get_max_flow_update(route_instance: Route_InstanceType, interval: int64) -> float64:
    max_flow = NP_FLOAT_MAX
    for leg in range(route_instance.legs + 1):
        max_flow = min(
            max_flow,
            route_instance.lines[leg].capacity
            - route_instance.line_directions[leg]
            * (route_instance.lines[leg].flows[interval] + route_instance.lines[leg].temp_leg_flows),
        )
    return max_flow


@njit(fastmath=FASTMATH)
def calculate_flow_update(route_instance: Route_InstanceType, interval: int64) -> None:
    route_instance.flow_update = min(
        route_instance.nodes[-1].temp_surplus, get_max_flow_update(route_instance, interval)
    )

    route_instance.initial_node.available_imports += route_instance.flow_update

    # If multiple routes on the same leg end with the same node, they must be constrained by surplus committed for that leg
    route_instance.nodes[-1].temp_surplus -= route_instance.flow_update

    # If multiple routes on the same leg use the same lines, they must be constrained by capacity committed for that leg
    for leg in range(route_instance.legs + 1):
        route_instance.lines[leg].temp_leg_flows += route_instance.line_directions[leg] * route_instance.flow_update
    return None


@njit(fastmath=FASTMATH)
def update_exports(route_instance: Route_InstanceType, interval: int64) -> None:
    route_instance.nodes[-1].imports_exports[interval] -= route_instance.flow_update
    route_instance.nodes[-1].surplus -= route_instance.flow_update

    for leg in range(route_instance.legs + 1):
        route_instance.lines[leg].flows[interval] += route_instance.line_directions[leg] * route_instance.flow_update
    return None
