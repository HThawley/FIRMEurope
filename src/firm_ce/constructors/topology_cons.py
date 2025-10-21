# type: ignore
from typing import Dict, Any

import numpy as np

from firm_ce.common.typing import DictType, ListType, TypedDict, TypedList, UniTuple, int64
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.fast_methods import route_m
from firm_ce.io.validate import is_nan
from firm_ce.system.topology import (
    Line,
    Line_InstanceType,
    Network,
    Network_InstanceType,
    Node,
    Node_InstanceType,
    Route,
    Route_InstanceType,
)


def construct_Node_object(order: int, node_dict: dict) -> Node_InstanceType:
    """
    Takes data required to initialise a single node object and returns an instance of the Node jitclass.
    The nodes (also called buses) represent a spatial location that is treated as a copper-plate. All
    Generator and Storage assets are located at a particular node in the network.
    Although constructor does not do much at this stage, it has been implemented for future-proofing
    in case nodes become more complex later on.

    Parameters:
    -------
    order (int): The scenario-level id associated with the node.
    node_name (str): A name associated with the node.

    Returns:
    -------
    Node_InstanceType: A static instance of the Node jitclass.
    """
    return Node(True, node_dict.get("id"), order, node_dict.get("name"))


def construct_Line_object(
    order: int,
    line_dict: Dict[str, Any],
    nodes_object_dict: DictType(int64, Node_InstanceType),
) -> Line_InstanceType:
    """
    Takes data required to initialise a single line object, casts values into Numba-compatible
    types, and returns an instance of the Line jitclass. The Lines define interconnections
    between nodes (buses) in the network.

    Parameters:
    -------
    order (int): The scenario-specific id for the Storage instance.
    line_dict (Dict[str,Any]): A dictionary containing the attributes of
        a single line object in `config/lines.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.

    Returns:
    -------
    Line_InstanceType: A static instance of the Line jitclass.
    """
    idx = int(line_dict["id"])
    name = str(line_dict["name"])
    length = float(line_dict["length"])
    loss_factor = float(line_dict["loss_factor"])
    max_build = float(line_dict["max_build"])
    min_build = float(line_dict["min_build"])
    capacity = float(line_dict["initial_capacity"])
    unit_type = str(line_dict["unit_type"])
    near_opt = str(line_dict.get("near_optimum", "")).lower() in ("true", "1", "yes")
    minor_node = construct_Node_object(-1, {"name": "MINOR_NODE", "id": -1})

    raw_group = line_dict.get("range_group", "")
    group = (
        name
        if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == ""
        else str(raw_group).strip()
    )

    node_start = next(
        (
            node
            for node in nodes_object_dict.values()
            if not is_nan(line_dict["node_start"]) and node.name == str(line_dict["node_start"])
        ),
        minor_node,
    )

    node_end = next(
        (
            node
            for node in nodes_object_dict.values()
            if not is_nan(line_dict["node_end"]) and node.name == str(line_dict["node_end"])
        ),
        minor_node,
    )

    cost = construct_UnitCost_object(
        float(line_dict["capex"]),
        float(line_dict["fom"]),
        float(line_dict["vom"]),
        int(line_dict["lifetime"]),
        float(line_dict["discount_rate"]),
        transformer_capex=int(line_dict["transformer_capex"]),
    )

    return Line(
        True,
        idx,
        order,
        name,
        length,
        node_start,
        node_end,
        loss_factor,
        max_build,
        min_build,
        capacity,
        unit_type,
        near_opt,
        group,
        cost,
    )


def construct_new_Route_object(
    initial_node: Node_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int,
) -> Route_InstanceType:
    """
    Initialises a single route object with the first line in the route. Routes are
    later extended after the Route instance has been created. A Route defines a
    possible sequence of steps through the network.

    Parameters:
    -------
    initial_node (Node_InstanceType): The node where the route begins.
    new_node (Node_InstanceType): The node at the other end of the first line in the route.
    new_line (Line_InstanceType): The first line in the route, connecting the initial_node
        and new_node.
    line_direction (int): Defines the direction of initial_node -> new_node relative to
        line.start_node -> line.end_node. Same direction is +1, different direction is -1.
    leg (int): The total number of steps (i.e., lines) in the route object.

    Returns:
    -------
    Route_InstanceType: A static instance of the Route jitclass.
    """
    route_nodes = TypedList.empty_list(Node_InstanceType)
    route_lines = TypedList.empty_list(Line_InstanceType)

    route_nodes.append(new_node)
    route_lines.append(new_line)

    return Route(True, initial_node, route_nodes, route_lines, np.array([line_direction], dtype=np.int64), leg)


def extend_route(
    route: Route_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int
) -> Route_InstanceType:
    """
    Takes an existing route, extends it by a single leg, and returns the extended
    Route instance.

    Parameters:
    -------
    route (Route_InstanceType): An existing Route instance to be extended.
    new_node (Node_InstanceType): The node at the other end of the new_line in the route.
    new_line (Line_InstanceType): The new line in the route, connecting the final node
        in the existing route to the new_node.
    line_direction (int): Defines the direction of route.nodes[-1] -> new_node relative to
        line.start_node -> line.end_node. Same direction is +1, different direction is -1.
    leg (int): The total number of steps (i.e., lines) along the extended route object.

    Returns:
    -------
    Route_InstanceType: A static instance of the Route jitclass.
    """
    route_nodes = route.nodes.copy()
    route_nodes.append(new_node)

    route_lines = route.lines.copy()
    route_lines.append(new_line)

    route_line_directions = list(route.line_directions)
    route_line_directions.append(line_direction)

    return Route(
        True, route.initial_node, route_nodes, route_lines, np.array(route_line_directions, dtype=np.int64), leg
    )


def get_routes_for_node(
    initial_node: Node_InstanceType,
    routes_typed_dict: DictType(UniTuple(int64, 2), ListType(Route_InstanceType)),
    lines_object_dict: DictType(int64, Line_InstanceType),
    leg: int,
) -> ListType(Route_InstanceType):
    """
    Builds a list of routes with length 'leg' to an initial_node. Routes can only use
    each node and line in the network once.
    If no routes have yet been added to the routes_typed_dict for the initial_node,
    the first routes are initialised. Otherwise, the previous leg of each route is
    extended to form the new list.

    Parameters:
    -------
    initial_node (Node_InstanceType): The destination of the route.
    routes_typed_dict (DictType(UniTuple(int64,2), ListType(Route_InstanceType))):
        A typed dictionary where values are lists of routes to the initial_node. The key
        is a tuple (intial_node.order, leg) so that routes of a specified length to a node
        can quickly be accessed.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.
    leg (int): The total number of steps (i.e., lines) along the route objects for the
        list returned by this function.

    Returns:
    -------
    ListType(Route_InstanceType): A typed list of Route instances. Typed list required for
        JIT-compatibility.
    """
    routes_to_node_curr_leg = TypedList.empty_list(Route_InstanceType)
    key = (initial_node.order, leg - 1)
    if key in routes_typed_dict:
        routes_to_node_prev_leg = routes_typed_dict[key].copy()
        for route in routes_to_node_prev_leg:
            for line in lines_object_dict.values():
                if route_m.check_contains_line(route, line):  # Remove loops
                    continue
                if line.node_start.order == route.nodes[-1].order:
                    if route_m.check_contains_node(route, line.node_end):  # Remove loops
                        continue
                    new_route = extend_route(route, line.node_end, line, -1, leg)
                    routes_to_node_curr_leg.append(new_route)
                elif line.node_end.order == route.nodes[-1].order:
                    if route_m.check_contains_node(route, line.node_start):  # Remove loops
                        continue
                    new_route = extend_route(route, line.node_start, line, 1, leg)
                    routes_to_node_curr_leg.append(new_route)
    else:
        routes_to_node_prev_leg = TypedList.empty_list(Route_InstanceType)
        for line in lines_object_dict.values():
            if line.node_start.order == initial_node.order:
                new_route = construct_new_Route_object(initial_node, line.node_end, line, -1, leg)
                routes_to_node_curr_leg.append(new_route)
            elif line.node_end.order == initial_node.order:
                new_route = construct_new_Route_object(initial_node, line.node_start, line, 1, leg)
                routes_to_node_curr_leg.append(new_route)
    return routes_to_node_curr_leg


def build_routes_typed_dict(
    networksteps_max: int,
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
) -> DictType(UniTuple(int64, 2), ListType(Route_InstanceType)):
    """
    Builds a typed dictionary where values are lists of routes to the initial_node. The key
    is a tuple (intial_node.order, leg) so that routes of a specified length to a node
    can quickly be accessed.

    Parameters:
    -------
    networksteps_max (int): The maximum number of legs allowed in a route for a given scenario.
        Can be adjusted in `config/scenarios.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.

    Returns:
    -------
    DictType(UniTuple(int64,2), ListType(Route_InstanceType)): A typed dictionary where values are
        lists of routes to the initial_node. The key is a tuple (intial_node.order, leg) so that
        routes of a specified length to a node can quickly be accessed.
    """
    routes_typed_dict = TypedDict.empty(key_type=UniTuple(int64, 2), value_type=ListType(Route_InstanceType))

    for leg in range(networksteps_max):
        for node in nodes_object_dict.values():
            routes_typed_list = get_routes_for_node(
                node,
                routes_typed_dict,
                lines_object_dict,
                leg,
            )

            routes_typed_dict[(node.order, leg)] = routes_typed_list
    return routes_typed_dict


def construct_Network_object(
    nodes_imported_dict: Dict[int, Dict[str, Any]],
    lines_imported_dict: Dict[int, Dict[str, Any]],
    networksteps_max: int,
) -> Network_InstanceType:
    """
    Takes data required to initialise a single network object, casts values into Numba-compatible
    types, builds Node and Line jitclasses, builds transmission routes, and returns an instance of
    the Network jitclass.

    The Network consists of all objects related to network topology and transmission.
    The major_lines are those associated with transmission topology, while minor_lines are those
    required to link Generator and Storage objects to the main transmission network.

    Parameters:
    -------
    nodes_imported_dict (Dict[int, Dict[str, Any]]): A dictionary containing data for all nodes
        imported from `config/nodes.csv`.
    lines_imported_dict (Dict[int, Dict[str, Any]]): A dictionary containing data for all
        lines imported from `config/lines.csv`.
    networksteps_max (int): The maximum number of legs allowed in a route for a given scenario.
        Can be adjusted in `config/scenarios.csv`.

    Returns:
    -------
    Network_InstanceType: A static instance of the Network jitclass.
    """

    nodes = TypedDict.empty(key_type=int64, value_type=Node_InstanceType)
    for order in nodes_imported_dict:
        nodes[order] = construct_Node_object(order, nodes_imported_dict[order])

    major_lines = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    minor_lines = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    order_major = 0
    order_minor = 0
    for idx in lines_imported_dict:
        if not is_nan(lines_imported_dict[idx]["node_start"]) and not is_nan(lines_imported_dict[idx]["node_end"]):
            major_lines[order_major] = construct_Line_object(order_major, lines_imported_dict[idx], nodes)
            order_major += 1
        else:
            minor_lines[order_minor] = construct_Line_object(order_minor, lines_imported_dict[idx], nodes)
            order_minor += 1

    routes = build_routes_typed_dict(networksteps_max, nodes, major_lines)

    return Network(True, nodes, major_lines, minor_lines, routes, networksteps_max)
