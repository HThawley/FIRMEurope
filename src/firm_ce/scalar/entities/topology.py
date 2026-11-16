# type: ignore
from typing import Any, Dict, List, Tuple

import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import (
    DictType,
    ListType,
    TypedDict,
    TypedList,
    UniTuple,
    boolean,
    nbfloat,
    nbint,
    nbintp,
    npfloat,
    npint,
    npintp,
    unicode_type,
)
from firm_ce.io.config_validation import is_nan
from firm_ce.scalar.entities.costs import (
    LTCosts,
    LTCosts_InstanceType,
    UnitCost_InstanceType,
    construct_UnitCost_object,
)

if JIT_ENABLED:
    node_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("id", nbintp),
        ("order", nbintp),
        ("name", unicode_type),
        ("data_status", boolean),
        ("mean_demand", nbfloat),
        ("relative_scaler", nbfloat),
        ("internal_loss", nbfloat),
        ("data", nbfloat[:]),
        ("residual_load", nbfloat[:]),
        # Dynamic
        ("storage_merit_order", nbintp[:]),
        ("flexible_merit_order", nbintp[:]),
        ("netload_t", nbfloat),
        ("discharge_max_t", nbfloat[:]),
        ("charge_max_t", nbfloat[:]),
        ("flexible_max_t", nbfloat[:]),
        ("fill", nbfloat),
        ("surplus", nbfloat),
        ("temp_surplus", nbfloat),
        ("available_imports", nbfloat),
        ("imports_exports", nbfloat[:]),
        ("deficits", nbfloat[:]),
        ("curtail", nbfloat[:]),
        ("flexible_power", nbfloat[:]),
        ("storage_power", nbfloat[:]),
        # Precharging
        ("imports_exports_temp", nbfloat),
        ("imports_exports_update", nbfloat),
        ("existing_surplus", nbfloat),
        ("precharge_fill", nbfloat),
        ("precharge_surplus", nbfloat),
    ]
else:
    node_spec = []


@jitclass(node_spec)
class Node:
    """
    Represents a node (bus) in the network.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: nbintp,
        order: nbintp,
        name: unicode_type,
        internal_loss: nbfloat = 0.0,
    ) -> None:
        self.object_class = "node"
        self.static_instance = static_instance
        self.id = idx
        self.order = order
        self.name = name
        self.data_status = False
        self.data = np.empty((0,), dtype=npfloat)
        self.mean_demand = 0.0
        self.relative_scaler = 1.0
        self.internal_loss = internal_loss

        self.residual_load = np.empty((0,), dtype=npfloat)

        # Dynamic
        self.flexible_merit_order = np.empty((0,), dtype=npintp)
        self.storage_merit_order = np.empty((0,), dtype=npintp)
        self.netload_t = 0.0  # GW
        self.discharge_max_t = np.empty((0,), dtype=npfloat)  # GW
        self.charge_max_t = np.empty((0,), dtype=npfloat)  # GW
        self.flexible_max_t = np.empty((0,), dtype=npfloat)  # GW

        self.fill = 0.0  # GW
        self.surplus = 0.0  # GW
        self.temp_surplus = 0.0  # GW
        self.available_imports = 0.0  # GW

        self.imports_exports = np.empty((0,), dtype=npfloat)
        self.deficits = np.empty((0,), dtype=npfloat)
        self.curtail = np.empty((0,), dtype=npfloat)

        self.flexible_power = np.empty((0,), dtype=npfloat)
        self.storage_power = np.empty((0,), dtype=npfloat)

        # Precharging
        self.imports_exports_temp = 0.0
        self.imports_exports_update = 0.0
        self.existing_surplus = 0.0
        self.precharge_fill = 0.0
        self.precharge_surplus = 0.0


if JIT_ENABLED:
    Node_InstanceType = Node.class_type.instance_type
else:
    Node_InstanceType = Node

if JIT_ENABLED:
    line_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("id", nbintp),
        ("order", nbintp),
        ("name", unicode_type),
        ("length", nbfloat),
        ("node_start", Node_InstanceType),
        ("node_end", Node_InstanceType),
        ("loss_factor", nbfloat),
        ("efficiency", nbfloat),
        ("max_build", nbfloat),
        ("min_build", nbfloat),
        ("initial_capacity", nbfloat),
        ("unit_type", unicode_type),
        ("unit_type_idx", nbintp),
        ("major", boolean),
        ("cost", UnitCost_InstanceType),
        ("candidate_x_idx", nbintp),
        ("relative_scaler", nbfloat),
        # Dynamic
        ("new_build", nbfloat),
        ("capacity", nbfloat),
        ("flows", nbfloat[:]),
        ("temp_leg_flows", nbfloat),
        ("lt_flows", nbfloat),
        ("lt_costs", LTCosts_InstanceType),
    ]
else:
    line_spec = []


@jitclass(line_spec)
class Line:
    """
    Transmission line connecting two Nodes or a minor line connecting an asset to the network.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: nbintp,
        order: nbintp,
        name: unicode_type,
        length: nbfloat,
        node_start: Node_InstanceType,
        node_end: Node_InstanceType,
        loss_factor: nbfloat,
        max_build: nbfloat,
        min_build: nbfloat,
        capacity: nbfloat,
        unit_type: unicode_type,
        major: boolean,
        cost: UnitCost_InstanceType,
    ) -> None:
        self.object_class = "line"
        self.static_instance = static_instance
        self.id = idx
        self.order = order
        self.name = name
        self.length = length  # km
        self.node_start = node_start
        self.node_end = node_end
        self.loss_factor = loss_factor  # Transmission losses % per 1000 km
        self.efficiency = 1 - (self.loss_factor * self.length / 1000)
        self.max_build = max_build  # GW/year
        self.min_build = min_build  # GW/year
        self.initial_capacity = capacity  # GW
        self.unit_type = unit_type
        self.major = major
        self.cost = cost

        self.candidate_x_idx = -1
        self.relative_scaler = 1.0

        # Dynamic
        self.new_build = 0.0  # GW
        self.capacity = capacity  # GW
        self.flows = np.empty(0, dtype=npfloat)  # GW
        self.temp_leg_flows = 0.0  # GW
        self.lt_flows = 0.0  # GWh

        self.lt_costs = LTCosts()


if JIT_ENABLED:
    Line_InstanceType = Line.class_type.instance_type
else:
    Line_InstanceType = Line

if JIT_ENABLED:
    route_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("initial_node", Node_InstanceType),
        ("nodes", ListType(Node_InstanceType)),
        ("lines", ListType(Line_InstanceType)),
        ("line_directions", nbint[:]),
        ("legs", nbint),
        ("flow_update", nbfloat),
        ("efficiency", nbfloat),
    ]
else:
    route_spec = []


@jitclass(route_spec)
class Route:
    """
    An ordered path of a specified length (legs) that steps through the transmission network.
    """

    def __init__(
        self,
        static_instance: boolean,
        initial_node: Node_InstanceType,
        nodes_typed_list: ListType(Node_InstanceType),
        lines_typed_list: ListType(Line_InstanceType),
        line_directions: nbint[:],
        legs: nbint,
    ):
        self.object_class = "route"
        self.static_instance = static_instance
        self.initial_node = initial_node
        self.nodes = nodes_typed_list
        self.lines = lines_typed_list
        self.line_directions = line_directions
        self.legs = legs
        self.efficiency = 1.0
        for line in self.lines:
            self.efficiency *= line.efficiency

        # Dynamic
        self.flow_update = 0.0


if JIT_ENABLED:
    Route_InstanceType = Route.class_type.instance_type
    routes_key_type = UniTuple(nbintp, 2)
    routes_list_type = ListType(Route_InstanceType)
else:
    Route_InstanceType = Route
    routes_key_type = Tuple[int, int]
    routes_list_type = List[Route_InstanceType]

if JIT_ENABLED:
    network_spec = [
        ("object_type", unicode_type),
        ("static_instance", boolean),
        ("nodes", DictType(nbintp, Node_InstanceType)),
        ("major_lines", DictType(nbintp, Line_InstanceType)),
        ("minor_lines", DictType(nbintp, Line_InstanceType)),
        ("networksteps_max", nbint),
        ("routes", DictType(routes_key_type, routes_list_type)),
        ("major_line_count", nbint),
    ]
else:
    network_spec = []


@jitclass(network_spec)
class Network:
    """
    Contains nodes, lines, and routes required to define the network topology and perform transmission.
    """

    def __init__(
        self,
        static_instance: boolean,
        nodes: DictType(nbintp, Node_InstanceType),
        major_lines: DictType(nbintp, Line_InstanceType),
        minor_lines: DictType(nbintp, Line_InstanceType),
        routes: DictType(routes_key_type, routes_list_type),
        networksteps_max: nbint,
    ) -> None:
        self.object_type = "Network"
        self.static_instance = static_instance
        self.nodes = nodes
        self.major_lines = major_lines
        self.minor_lines = minor_lines
        self.routes = routes
        self.networksteps_max = networksteps_max
        self.major_line_count = len(major_lines)


if JIT_ENABLED:
    Network_InstanceType = Network.class_type.instance_type
else:
    Network_InstanceType = Network


def construct_Node_object(order: int, node_dict: dict) -> Node_InstanceType:
    return Node(True, node_dict["id"], order, node_dict["name"], node_dict.get("internal_loss", 0.0))


def construct_Line_object(
    order: int,
    line_dict: Dict[str, Any],
    nodes_object_dict: DictType(nbintp, Node_InstanceType),
    major: bool,
) -> Line_InstanceType:
    idx = int(line_dict["id"])
    name = str(line_dict["name"])
    length = float(line_dict["length"])
    loss_factor = float(line_dict["loss_factor"])
    max_build = float(line_dict["max_build"])
    min_build = float(line_dict["min_build"])
    capacity = float(line_dict["initial_capacity"])
    unit_type = str(line_dict["unit_type"])
    minor_node = construct_Node_object(-1, {"name": "MINOR_NODE", "id": -1, "internal_loss": 0.0})

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
        major,
        cost,
    )


def construct_new_Route_object(
    initial_node: Node_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int,
) -> Route_InstanceType:
    route_nodes = TypedList.empty_list(Node_InstanceType)
    route_lines = TypedList.empty_list(Line_InstanceType)

    route_nodes.append(new_node)
    route_lines.append(new_line)

    return Route(True, initial_node, route_nodes, route_lines, np.array([line_direction], dtype=npint), leg)


def extend_route(
    route: Route_InstanceType,
    new_node: Node_InstanceType,
    new_line: Line_InstanceType,
    line_direction: int,
    leg: int,
) -> Route_InstanceType:
    route_nodes = route.nodes.copy()
    route_nodes.append(new_node)

    route_lines = route.lines.copy()
    route_lines.append(new_line)

    route_line_directions = list(route.line_directions)
    route_line_directions.append(line_direction)

    return Route(
        True, route.initial_node, route_nodes, route_lines, np.array(route_line_directions, dtype=npint), leg
    )


def _route_contains_line(route: Route_InstanceType, candidate_line: Line_InstanceType) -> bool:
    for line in route.lines:
        if candidate_line.order == line.order:
            return True
    return False


def _route_contains_node(route: Route_InstanceType, candidate_node: Node_InstanceType) -> bool:
    for node in route.nodes:
        if candidate_node.order == node.order:
            return True
    return False


def get_routes_for_node(
    initial_node: Node_InstanceType,
    routes_typed_dict: DictType(UniTuple(nbintp, 2), ListType(Route_InstanceType)),
    lines_object_dict: DictType(nbintp, Line_InstanceType),
    leg: int,
) -> ListType(Route_InstanceType):
    routes_to_node_curr_leg = TypedList.empty_list(Route_InstanceType)
    key = (initial_node.order, leg - 1)
    if key in routes_typed_dict:
        routes_to_node_prev_leg = routes_typed_dict[key].copy()
        for route in routes_to_node_prev_leg:
            for line in lines_object_dict.values():
                if _route_contains_line(route, line):
                    continue
                if line.node_start.order == route.nodes[-1].order:
                    if _route_contains_node(route, line.node_end):
                        continue
                    new_route = extend_route(route, line.node_end, line, -1, leg)
                    routes_to_node_curr_leg.append(new_route)
                elif line.node_end.order == route.nodes[-1].order:
                    if _route_contains_node(route, line.node_start):
                        continue
                    new_route = extend_route(route, line.node_start, line, 1, leg)
                    routes_to_node_curr_leg.append(new_route)
    else:
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
    nodes_object_dict: DictType(nbintp, Node_InstanceType),
    lines_object_dict: DictType(nbintp, Line_InstanceType),
) -> DictType(UniTuple(nbintp, 2), ListType(Route_InstanceType)):
    routes_typed_dict = TypedDict.empty(key_type=UniTuple(nbintp, 2), value_type=ListType(Route_InstanceType))

    for leg in range(networksteps_max):
        for node in nodes_object_dict.values():
            routes_typed_list = get_routes_for_node(
                node,
                routes_typed_dict,
                lines_object_dict,
                leg,
            )
            routes_typed_dict[(node.order, nbintp(leg))] = routes_typed_list
    return routes_typed_dict


def construct_Network_object(
    nodes_imported_dict: Dict[int, Dict[str, Any]],
    lines_imported_dict: Dict[int, Dict[str, Any]],
    networksteps_max: int,
) -> Network_InstanceType:
    nodes = TypedDict.empty(key_type=nbintp, value_type=Node_InstanceType)
    for order, idx in enumerate(nodes_imported_dict):
        nodes[order] = construct_Node_object(order, nodes_imported_dict[idx])

    major_lines = TypedDict.empty(key_type=nbintp, value_type=Line_InstanceType)
    minor_lines = TypedDict.empty(key_type=nbintp, value_type=Line_InstanceType)
    order_major = 0
    order_minor = 0
    for idx in lines_imported_dict:
        if not is_nan(lines_imported_dict[idx]["node_start"]) and not is_nan(lines_imported_dict[idx]["node_end"]):
            major_lines[order_major] = construct_Line_object(order_major, lines_imported_dict[idx], nodes, True)
            order_major += 1
        else:
            minor_lines[order_minor] = construct_Line_object(order_minor, lines_imported_dict[idx], nodes, False)
            order_minor += 1

    routes = build_routes_typed_dict(networksteps_max, nodes, major_lines)

    return Network(True, nodes, major_lines, minor_lines, routes, networksteps_max)
