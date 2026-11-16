# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, TypedDict, TypedList, boolean, nbfloat, npfloat, nbint, nbintp, unicode_type
from firm_ce.scalar.entities.components import Generator_InstanceType, Storage_InstanceType
from firm_ce.scalar.entities.topology import (
    Line_InstanceType,
    Network,
    Network_InstanceType,
    Node_InstanceType,
    Route_InstanceType,
    routes_key_type,
    routes_list_type,
)
from firm_ce.scalar.methods import line_m, node_m, route_m


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def create_dynamic_copy(
    network_instance: Network_InstanceType,
) -> Network_InstanceType:
    """
    A 'static' instance of the Network jitclass (Network.static_instance=True) is copied
    and marked as a 'dynamic' instance (Network.static_instance=False).
    """
    nodes_copy = TypedDict.empty(key_type=nbintp, value_type=Node_InstanceType)
    major_lines_copy = TypedDict.empty(key_type=nbintp, value_type=Line_InstanceType)
    minor_lines_copy = TypedDict.empty(key_type=nbintp, value_type=Line_InstanceType)
    routes_copy = TypedDict.empty(key_type=routes_key_type, value_type=routes_list_type)

    for order, node in network_instance.nodes.items():
        nodes_copy[order] = node_m.create_dynamic_copy(node)

    for order, line in network_instance.major_lines.items():
        major_lines_copy[order] = line_m.create_dynamic_copy(line, nodes_copy, "major")

    for order, line in network_instance.minor_lines.items():
        minor_lines_copy[order] = line_m.create_dynamic_copy(line, nodes_copy, "minor")

    for tuple_key, routes_list in network_instance.routes.items():
        routes_copy[tuple_key] = TypedList.empty_list(Route_InstanceType)
        for route in routes_list:
            routes_copy[tuple_key].append(route_m.create_dynamic_copy(route, nodes_copy, major_lines_copy))

    network_copy = Network(
        False,
        nodes_copy,
        major_lines_copy,
        minor_lines_copy,
        routes_copy,
        network_instance.networksteps_max,
    )
    network_copy.major_line_count = network_instance.major_line_count
    return network_copy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def build_capacity(
    network_instance: Network_InstanceType,
    decision_x: nbfloat[:],
) -> None:
    """
    The candidate solution defines new build capacity for each Generator, Storage, and Line (major_lines) object. This
    function modifies each major_line Line object in the Network to build new capacity.
    """
    if network_instance.static_instance:
        raise_static_modification_error()
    for line in network_instance.major_lines.values():
        if line.candidate_x_idx != -1:
            line_m.build_capacity(line, decision_x[line.candidate_x_idx])
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def unload_data(
    network_instance: Network_InstanceType,
) -> None:
    """
    Unload the operational demand trace for each Node instance in Network.nodes.
    """
    for node in network_instance.nodes.values():
        node_m.unload_data(node)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def allocate_memory(
    network_instance: Network_InstanceType,
    intervals_count: nbint,
) -> None:
    """
    Memory associated with endogenous time-series data for Node and Line instances is only
    allocated after a dynamic copy of the Network instance is created.
    """
    if network_instance.static_instance:
        raise_static_modification_error()
    for node in network_instance.nodes.values():
        node_m.allocate_memory(node, intervals_count)
    for line in network_instance.major_lines.values():
        line_m.allocate_memory(line, intervals_count)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_remaining_netloads(
    network_instance: Network_InstanceType,
    interval: nbintp,
    check_case: unicode_type,
) -> boolean:
    for node in network_instance.nodes.values():
        if node_m.check_remaining_netload(node, interval, check_case):
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_period_unserved_energy(
    network_instance: Network_InstanceType,
    first_t: nbintp,
    last_t: nbintp,
    resolution: nbfloat,
) -> nbfloat:
    unserved_energy = 0.0
    for node in network_instance.nodes.values():
        _acc = 0.0
        for t in range(first_t, last_t):
            _acc += node.deficits[t] * resolution
        unserved_energy += _acc

    return unserved_energy


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_transmission(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> None:
    """
    Transmission flows and nodal import/export values are iteratively updated when solving the unit committment
    problem for a given time interval. This function resets all transmission variables to zero for a time interval.
    """
    for line in network_instance.major_lines.values():
        line.flows[interval] = 0.0
    for node in network_instance.nodes.values():
        node.imports_exports[interval] = 0.0
        node_m.update_netload_t(node, interval, False)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_flow_updates(
    network_instance: Network_InstanceType,
) -> None:
    """
    Resets the temporary flow update value for all Route instances in the Network.
    """
    for route_list in network_instance.routes.values():
        for route in route_list:
            route.flow_update = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_route_surpluses(
    network_instance: Network_InstanceType,
    fill_node: Node_InstanceType,
    leg: nbintp,
) -> boolean:
    """
    Checks each Route in a route list (corresponding to a particular start node and route length) and determines if
    the final Node in any Route has surplus energy available.
    """
    for route in network_instance.routes[fill_node.order, leg]:
        if node_m.surplus_available(route.nodes[-1]):
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_network_surplus(
    network_instance: Network_InstanceType,
) -> boolean:
    """
    Checks whether any Node in the Network has surplus energy available.
    """
    for node in network_instance.nodes.values():
        if node_m.surplus_available(node):
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_network_fill(
    network_instance: Network_InstanceType,
) -> boolean:
    """
    Checks whether any Node in the Network is attempting to balance fill energy.
    """
    for node in network_instance.nodes.values():
        if node_m.fill_required(node):
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_node_flow_updates(
    network_instance: Network_InstanceType,
    fill_node: Node_InstanceType,
    leg: nbintp,
    interval: nbintp,
) -> None:
    fill_node.available_imports = 0.0
    for route in network_instance.routes[fill_node.order, leg]:
        route_m.calculate_flow_update(route, interval)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def scale_flow_updates_to_fill(
    network_instance: Network_InstanceType,
    fill_node: Node_InstanceType,
    leg: nbintp,
) -> nbfloat:
    if fill_node.available_imports > fill_node.fill:
        scale_factor = fill_node.fill / fill_node.available_imports
        for route in network_instance.routes[fill_node.order, leg]:
            route.flow_update *= scale_factor
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_transmission_flows(
    network_instance: Network_InstanceType,
    fill_node: Node_InstanceType,
    leg: nbintp,
    interval: nbintp,
) -> None:
    for route in network_instance.routes[fill_node.order, leg]:
        fill_node.imports_exports[interval] += route.flow_update
        fill_node.fill -= route.flow_update
        route_m.update_exports(route, interval)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_netloads(
    network_instance: Network_InstanceType,
    interval: nbintp,
    precharging_flag: boolean,
) -> None:
    """
    Recalculate the netload for each Node in the network.
    """
    for node in network_instance.nodes.values():
        node_m.update_netload_t(node, interval, precharging_flag)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_line_temp_flows(
    network_instance: Network_InstanceType,
) -> None:
    """
    Reset the temporary value for committed flows for each major Line in the Network to zero.
    """
    for line in network_instance.major_lines.values():
        line.temp_leg_flows = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_node_temp_surpluses(
    network_instance: Network_InstanceType,
) -> None:
    """
    Reset the temporary value for committed surplus energy for each Node in the Network to the total
    surplus for that Node.
    """
    for node in network_instance.nodes.values():
        node.temp_surplus = node.surplus
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def fill_with_transmitted_surpluses(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> None:
    """
    Overarching function that manages transmission actions. Surplus energy from Nodes is transmitted to
    balance fill energy at other Nodes.
    """
    reset_flow_updates(network_instance)
    if not (check_network_surplus(network_instance) and check_network_fill(network_instance)):
        return None

    for leg in range(network_instance.networksteps_max):
        for node in network_instance.nodes.values():
            if not node_m.fill_required(node):
                continue
            if len(network_instance.routes[node.order, leg]) == 0:
                continue
            if not check_route_surpluses(network_instance, node, leg):
                continue
            reset_line_temp_flows(network_instance)
            reset_node_temp_surpluses(network_instance)
            calculate_node_flow_updates(network_instance, node, leg, interval)
            scale_flow_updates_to_fill(network_instance, node, leg)
            update_transmission_flows(network_instance, node, leg, interval)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_node_fills_and_surpluses(
    network_instance: Network_InstanceType,
    transmission_case: unicode_type,
    interval: nbintp,
) -> None:
    if transmission_case == "surplus":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t, 0.0)
            node.surplus = -min(node.netload_t, 0.0)

    elif transmission_case == "storage_discharge":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval], 0.0)
            node.surplus = max(node.discharge_max_t[-1] - node.storage_power[interval], 0.0)

    elif transmission_case == "flexible":
        for node in network_instance.nodes.values():
            node.fill = max(
                node.netload_t - node.storage_power[interval] - node.flexible_power[interval],
                0.0,
            )
            node.surplus = max(node.flexible_max_t[-1] - node.flexible_power[interval], 0.0)

    elif transmission_case == "storage_charge":
        for node in network_instance.nodes.values():
            node.fill = max(node.charge_max_t[-1] + node.storage_power[interval], 0.0)
            node.surplus = -min(
                node.netload_t - node.storage_power[interval] - node.flexible_power[interval],
                0.0,
            )

    elif transmission_case == "precharging_surplus":
        for node in network_instance.nodes.values():
            node.fill = node.precharge_fill
            node.surplus = node.existing_surplus

    elif transmission_case == "precharging_transfers":
        for node in network_instance.nodes.values():
            node.fill = node.precharge_fill
            node.surplus = node.precharge_surplus

    elif transmission_case == "precharging_adjust_storage":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - max(node.storage_power[interval], 0.0), 0.0)
            node.surplus = -min(node.netload_t - max(node.storage_power[interval], 0), 0.0)

    elif transmission_case == "precharging_adjust_surplus":
        for node in network_instance.nodes.values():
            node.fill = max(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval],
                0.0,
            )
            node.surplus = -min(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval],
                0.0,
            )
    elif transmission_case == "precharging_adjust_flexible":
        for node in network_instance.nodes.values():
            node.fill = -min(node.storage_power[interval], 0.0)
            node.surplus = -min(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval],
                0.0,
            )
    else:
        raise RuntimeError("set_node_fills_and_surpluses contains incorrect transmission_case argument.")
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_curtail_and_deficit(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> None:
    """
    Calculate the power spilled/curtailed and the unserved power (deficit) for each Node in the network
    upon completing the balancing actions for a time interval.
    """
    for node in network_instance.nodes.values():
        _imbalance = node.netload_t - node.storage_power[interval] - node.flexible_power[interval]
        node.deficits[interval] = max(_imbalance, 0)
        node.curtail[interval] = min(_imbalance, 0)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_storage_merit_orders(
    network_instance: Network_InstanceType,
    storages_typed_dict: DictType(nbintp, Storage_InstanceType),
) -> None:
    """
    For each Node, finds all Storage instances at that Node. Sorts the Storage systems at that Node from
    longest to shortest duration.
    """
    for node in network_instance.nodes.values():
        node_m.assign_storage_merit_order(node, storages_typed_dict)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_flexible_merit_orders(
    network_instance: Network_InstanceType,
    generators_typed_dict: DictType(nbintp, Generator_InstanceType),
) -> None:
    for node in network_instance.nodes.values():
        node_m.assign_flexible_merit_order(node, generators_typed_dict)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def assign_route_merit_orders(
    network_instance: Network_InstanceType,
) -> None:
    """
    Sorts the routes within each route list in the network based on cumulative efficiency.
    Higher efficiency routes are prioritized (placed at the start of the list).
    """
    for tuple_key, routes_list in network_instance.routes.items():
        count = len(routes_list)
        if count <= 1:
            continue

        effs = np.zeros(count, dtype=npfloat)
        for i in range(count):
            effs[i] = routes_list[i].efficiency

        # Sort descending (highest efficiency first)
        sort_order = np.argsort(-effs)

        # Create a new typed list and replace the old one
        sorted_routes = TypedList.empty_list(Route_InstanceType)
        for i in range(count):
            sorted_routes.append(routes_list[sort_order[i]])

        network_instance.routes[tuple_key] = sorted_routes

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_lt_flows(
    network_instance: Network_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    After completing unit committment, calculates the total long-term major Line flows across the entire
    modelling horizon.
    """
    for line in network_instance.major_lines.values():
        line_m.calculate_lt_flow(line, resolution)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_flexible(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> None:
    """
    Resets the nodal flexible power to zero at each Node for a given time interval.
    """
    for node in network_instance.nodes.values():
        node.flexible_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def reset_dispatch(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> None:
    """
    Resets the nodal storage power and flexible power to zero at each Node for a given time interval.
    """
    for node in network_instance.nodes.values():
        node.storage_power[interval] = 0.0
        node.flexible_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_precharging_end(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> boolean:
    """
    Check whether the start of the deficit block has been reached.
    """
    if interval == 0:
        return True
    for node in network_instance.nodes.values():
        if (
            node.residual_load[interval - 1]
            - node.imports_exports[interval - 1]
            - node.storage_power[interval - 1]
            - node.flexible_power[interval - 1]
            > TOLERANCE
        ):
            return False
    return True


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_existing_surplus(
    network_instance: Network_InstanceType,
) -> boolean:
    """
    During the precharging period, check if any Nodes have an existing surplus that can be used to
    precharge Storage systems.
    """
    for node in network_instance.nodes.values():
        if node.existing_surplus > TOLERANCE:
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_storage_precharge_fills_and_surpluses(
    network_instance: Network_InstanceType,
) -> None:
    """
    Updates the temporary precharging period fill and surplus values that are used to evaluate
    inter-storage transfers.
    """
    for node in network_instance.nodes.values():
        node.precharge_fill = node.charge_max_t[-1]
        node.precharge_surplus = node.discharge_max_t[-1]
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def set_flexible_precharge_fills_and_surpluses(
    network_instance: Network_InstanceType,
) -> None:
    """
    Updates the temporary precharging period fill and surplus values that are used to evaluate
    flexible precharging of Storage systems.
    """
    for node in network_instance.nodes.values():
        node.precharge_fill = node.charge_max_t[-1]
        node.precharge_surplus = node.flexible_max_t[-1]
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def update_imports_exports_temp(
    network_instance: Network_InstanceType,
    interval: nbintp,
) -> None:
    """
    Within the precharging period, imports/exports at each Node are adjusted to allow for additional
    transmission of electricity to precharge Storage systems.
    """
    for node in network_instance.nodes.values():
        node.imports_exports_update = node.imports_exports_temp - node.imports_exports[interval]
        node_m.set_imports_exports_temp(node, interval)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_precharge_fill(
    network_instance: Network_InstanceType,
) -> boolean:
    """
    Within the precharging period, checks whether any Node in the Network has fill energy.
    """
    for node in network_instance.nodes.values():
        if node.precharge_fill > TOLERANCE:
            return True
    return False


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_precharge_surplus(
    network_instance: Network_InstanceType,
) -> boolean:
    """
    Within the precharging period, checks whether any Node in the Network has surplus energy.
    """
    for node in network_instance.nodes.values():
        if node.precharge_surplus > TOLERANCE:
            return True
    return False
