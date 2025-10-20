# type: ignore
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
def check_contains_line(
    route_instance: Route_InstanceType,
    new_line: Line_InstanceType,
) -> boolean:
    """
    Checks if a Route already contains a particular Line instance, identified by the unique 'order' of
    that Line. Routes can only contain one of each Line instance in order to prevent the formation of
    loops along the path.

    Parameters:
    -------
    route_instance (Route_InstanceType): A static instance of the Route jitclass which is being extended
        during initialisation of the Network.
    new_line (Line_InstanceType): A candidate Line that could be used to extend the Route.

    Returns:
    -------
    boolean: If the candidate Line is already part of the Route, returns True. Otherwise, False.
    """
    for line in route_instance.lines:
        if new_line.order == line.order:
            return True
    return False


@njit(fastmath=FASTMATH)
def check_contains_node(
    route_instance: Route_InstanceType,
    new_node: Node_InstanceType,
) -> boolean:
    """
    Checks if a Route already contains a particular Node instance, identified by the unique 'order' of
    that Node. Routes can only contain one of each Node instance in order to prevent the formation of
    loops along the path. Loops formed by double-usage of a Node (even if all Lines are unique)
    could unnecessarily clog up the transmission lines that form the loop component.

    Parameters:
    -------
    route_instance (Route_InstanceType): A static instance of the Route jitclass which is being extended
        during initialisation of the Network.
    new_node (Node_InstanceType): A candidate Node that could be used to extend the Route.

    Returns:
    -------
    boolean: If the candidate Node is already part of the Route, returns True. Otherwise, False.
    """
    for node in route_instance.nodes:
        if new_node.order == node.order:
            return True
    return False


@njit(fastmath=FASTMATH)
def get_max_flow_update(
    route_instance: Route_InstanceType,
    interval: int64,
) -> float64:
    """
    During a transmission action, the maximum flow update for a Route represents the maximum
    amount of electricity that could be imported to the fill node along that Route. It is constrained
    by the capacity of the Lines forming the Route, accounting for flows that have already been committed
    for transmission along the line. The temporary leg flows are flows that could be committed for Routes
    of the same length to the same fill node that have already been checked for this transmission action.

    Parameters:
    -------
    route_instance (Route_InstanceType): A static instance of the Route jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    float64: The maximum flow for the Route for this transmission action [GW], accounting for the capacity
        of Lines along the Route that has already been committed for transmission.
    """
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
def calculate_flow_update(
    route_instance: Route_InstanceType,
    interval: int64,
) -> None:
    """
    During a transmission action, the flow update of each Route for a given fill node and route length
    is initially based upon the surplus energy at the final Node in the route. It is constrained by
    the maximum flow update possible for that Route. A temporary surplus value, rather than the total
    surplus, is used to account for surplus energy that has already been committed to Routes for the
    same fill node and route length. The temporary surplus value prevents double-counting of surplus
    energy.

    Note that each Line in a Route has a direction that is defined by the order of the start and end Nodes of
    that Line. Increasing flow in one direction is equivalent to decreasing flow in the opposite direction.

    Parameters:
    -------
    route_instance (Route_InstanceType): A static instance of the Route jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for Route instance: flow_update, initial_node, nodes, lines.
    Attributes modified for Node referenced by Route.initial_node (i.e., the fill node): available_imports.
    Attributes modified for Node referenced by Route.nodes[-1]: temp_surplus.
    Attributes modified for all Line instances referenced in Route.lines: temp_leg_flows.
    """
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
def update_exports(
    route_instance: Route_InstanceType,
    interval: int64,
) -> None:
    """
    Once the flow updates for the Route instances corresponding to a particular start node and route length
    have been determined, the transmission action is executed.

    The Node instances at the end of each Route supply the surplus generation for balancing, so their exports
    are increased and surplus energy reduced by the flow updates. Updates to line flows have equal magnitude
    to the flow updates of the corresponding Route, with the direction defined by the exogenous definition
    of the start and end Node of the Line.

    Parameters:
    -------
    route_instance (Route_InstanceType): An instance of the Route jitclass.
    interval (int64): Index for the time interval.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for Node referenced by Route.nodes[-1]: imports_exports (positive is imports, negative is exports),
        surplus.
    Attributes modified for all Line instances referenced in Route.lines: flows.
    """
    route_instance.nodes[-1].imports_exports[interval] -= route_instance.flow_update
    route_instance.nodes[-1].surplus -= route_instance.flow_update

    for leg in range(route_instance.legs + 1):
        route_instance.lines[leg].flows[interval] += route_instance.line_directions[leg] * route_instance.flow_update
    return None
