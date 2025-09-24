import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import raise_static_modification_error
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import TypedDict, TypedList, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import line_m, node_m, route_m
from firm_ce.system.topology import (
    Line_InstanceType,
    Network,
    Network_InstanceType,
    Node_InstanceType,
    Route_InstanceType,
    routes_key_type,
    routes_list_type,
)


@njit(fastmath=FASTMATH)
def create_dynamic_copy(network_instance: Network_InstanceType) -> Network_InstanceType:
    nodes_copy = TypedDict.empty(key_type=int64, value_type=Node_InstanceType)
    major_lines_copy = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
    minor_lines_copy = TypedDict.empty(key_type=int64, value_type=Line_InstanceType)
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


@njit(fastmath=FASTMATH)
def build_capacity(network_instance: Network_InstanceType, decision_x: float64[:]) -> None:
    if network_instance.static_instance:
        raise_static_modification_error()
    for line in network_instance.major_lines.values():
        line_m.build_capacity(line, decision_x[line.candidate_x_idx])
    return None


@njit(fastmath=FASTMATH)
def unload_data(network_instance: Network_InstanceType) -> None:
    for node in network_instance.nodes.values():
        node_m.unload_data(node)
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(network_instance: Network_InstanceType, intervals_count: int64) -> None:
    if network_instance.static_instance:
        raise_static_modification_error()
    for node in network_instance.nodes.values():
        node_m.allocate_memory(node, intervals_count)
    for line in network_instance.major_lines.values():
        line_m.allocate_memory(line, intervals_count)
    return None


@njit(fastmath=FASTMATH)
def check_remaining_netloads(
    network_instance: Network_InstanceType, interval: int64, check_case: unicode_type
) -> boolean:
    for node in network_instance.nodes.values():
        if node_m.check_remaining_netload(node, interval, check_case):
            return True
    return False


@njit(fastmath=FASTMATH)
def calculate_period_unserved_energy(
    network_instance: Network_InstanceType, first_t: int64, last_t: int64, interval_resolutions: float64[:]
) -> float64:
    unserved_energy = 0
    for node in network_instance.nodes.values():
        unserved_energy += sum(node.deficits[first_t : last_t + 1] * interval_resolutions[first_t : last_t + 1])
    return unserved_energy


@njit(fastmath=FASTMATH)
def reset_transmission(network_instance: Network_InstanceType, interval: int64) -> None:
    for line in network_instance.major_lines.values():
        line.flows[interval] = 0.0
    for node in network_instance.nodes.values():
        node.imports_exports[interval] = 0.0
        node_m.update_netload_t(node, interval, False)
    return None


@njit(fastmath=FASTMATH)
def reset_flow_updates(network_instance: Network_InstanceType) -> None:
    for route_list in network_instance.routes.values():
        for route in route_list:
            route.flow_update = 0.0
    return None


@njit(fastmath=FASTMATH)
def check_route_surpluses(network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64) -> boolean:
    # Check if final node in the route has a surplus available
    for route in network_instance.routes[fill_node.order, leg]:
        if node_m.surplus_available(route.nodes[-1]):
            return True
    return False


@njit(fastmath=FASTMATH)
def check_network_surplus(network_instance: Network_InstanceType) -> boolean:
    for node in network_instance.nodes.values():
        if node_m.surplus_available(node):
            return True
    return False


@njit(fastmath=FASTMATH)
def check_network_fill(network_instance: Network_InstanceType) -> boolean:
    for node in network_instance.nodes.values():
        if node_m.fill_required(node):
            return True
    return False


@njit(fastmath=FASTMATH)
def calculate_node_flow_updates(
    network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64, interval: int64
) -> None:
    fill_node.available_imports = 0.0
    for route in network_instance.routes[fill_node.order, leg]:
        route_m.calculate_flow_update(route, interval)
    return None


@njit(fastmath=FASTMATH)
def scale_flow_updates_to_fill(
    network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64
) -> float64:
    if fill_node.available_imports > fill_node.fill:
        scale_factor = fill_node.fill / fill_node.available_imports
        for route in network_instance.routes[fill_node.order, leg]:
            route.flow_update *= scale_factor
    return None


@njit(fastmath=FASTMATH)
def update_transmission_flows(
    network_instance: Network_InstanceType, fill_node: Node_InstanceType, leg: int64, interval: int64
) -> None:
    for route in network_instance.routes[fill_node.order, leg]:
        fill_node.imports_exports[interval] += route.flow_update
        fill_node.fill -= route.flow_update
        route_m.update_exports(route, interval)
    return None


@njit(fastmath=FASTMATH)
def update_netloads(network_instance: Network_InstanceType, interval: int64, precharging_flag: boolean) -> None:
    for node in network_instance.nodes.values():
        node_m.update_netload_t(node, interval, precharging_flag)
    return None


@njit(fastmath=FASTMATH)
def reset_line_temp_flows(network_instance: Network_InstanceType) -> None:
    for line in network_instance.major_lines.values():
        line.temp_leg_flows = 0.0
    return None


@njit(fastmath=FASTMATH)
def reset_node_temp_surpluses(network_instance: Network_InstanceType) -> None:
    for node in network_instance.nodes.values():
        node.temp_surplus = node.surplus
    return None


@njit(fastmath=FASTMATH)
def fill_with_transmitted_surpluses(network_instance: Network_InstanceType, interval: int64) -> None:
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


@njit(fastmath=FASTMATH)
def set_node_fills_and_surpluses(
    network_instance: Network_InstanceType, transmission_case: unicode_type, interval: int64
) -> None:
    if transmission_case == "surplus":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t, 0)
            node.surplus = -min(node.netload_t, 0)
    elif transmission_case == "storage_discharge":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval], 0)
            node.surplus = max(node.discharge_max_t[-1] - node.storage_power[interval], 0)

    elif transmission_case == "flexible":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
            node.surplus = max(node.flexible_max_t[-1] - node.flexible_power[interval], 0)

    elif transmission_case == "storage_charge":
        for node in network_instance.nodes.values():
            node.fill = max(node.charge_max_t[-1] + node.storage_power[interval], 0)
            node.surplus = -min(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0.0)

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
            node.fill = max(node.netload_t - max(node.storage_power[interval], 0), 0)
            node.surplus = -min(node.netload_t - max(node.storage_power[interval], 0), 0)

    elif transmission_case == "precharging_adjust_surplus":
        for node in network_instance.nodes.values():
            node.fill = max(node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval], 0)
            node.surplus = -min(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval], 0
            )
    elif transmission_case == "precharging_adjust_flexible":
        for node in network_instance.nodes.values():
            node.fill = -min(node.storage_power[interval], 0.0)
            node.surplus = -min(
                node.netload_t - max(node.storage_power[interval], 0.0) - node.flexible_power[interval], 0
            )
    else:
        raise RuntimeError("set_node_fills_and_surpluses contains incorrect transmission_case argument.")
    return None


@njit(fastmath=FASTMATH)
def calculate_spillage_and_deficit(network_instance: Network_InstanceType, interval: int64) -> None:
    for node in network_instance.nodes.values():
        node.deficits[interval] = max(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
        node.spillage[interval] = min(node.netload_t - node.storage_power[interval] - node.flexible_power[interval], 0)
    return None


@njit(fastmath=FASTMATH)
def assign_storage_merit_orders(
    network_instance: Network_InstanceType, storages_typed_dict  # DictType(int64, Storage_InstanceType)
) -> None:
    for node in network_instance.nodes.values():
        node_m.assign_storage_merit_order(node, storages_typed_dict)
    return None


@njit(fastmath=FASTMATH)
def assign_flexible_merit_orders(
    network_instance: Network_InstanceType, generators_typed_dict  # DictType(int64, Generators_InstanceType)
) -> None:
    for node in network_instance.nodes.values():
        node_m.assign_flexible_merit_order(node, generators_typed_dict)
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_flows(network_instance: Network_InstanceType, interval_resolutions: float64[:]) -> None:
    for line in network_instance.major_lines.values():
        line_m.calculate_lt_flow(line, interval_resolutions)
    return None


@njit(fastmath=FASTMATH)
def calculate_lt_line_losses(network_instance: Network_InstanceType) -> float64:
    total_line_losses = 0.0
    for line in network_instance.major_lines.values():
        total_line_losses += line_m.get_lt_losses(line)
    for line in network_instance.minor_lines.values():
        total_line_losses += line_m.get_lt_losses(line)
    return total_line_losses


@njit(fastmath=FASTMATH)
def calculate_net_residual_load(network_instance: Network_InstanceType) -> float64[:]:
    net_residual_load = np.zeros_like(network_instance.nodes[0].residual_load, dtype=np.float64)
    for node in network_instance.nodes.values():
        net_residual_load += node.residual_load
    return net_residual_load


@njit(fastmath=FASTMATH)
def reset_flexible(network_instance: Network_InstanceType, interval: int64) -> None:
    for node in network_instance.nodes.values():
        node.flexible_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH)
def reset_dispatch(network_instance: Network_InstanceType, interval: int64) -> None:
    for node in network_instance.nodes.values():
        node.storage_power[interval] = 0.0
        node.flexible_power[interval] = 0.0
    return None


@njit(fastmath=FASTMATH)
def check_precharging_end(network_instance: Network_InstanceType, interval: int64) -> boolean:
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


@njit(fastmath=FASTMATH)
def check_existing_surplus(network_instance: Network_InstanceType) -> boolean:
    for node in network_instance.nodes.values():
        if node.existing_surplus > TOLERANCE:
            return True
    return False


@njit(fastmath=FASTMATH)
def set_storage_precharge_fills_and_surpluses(network_instance: Network_InstanceType) -> None:
    for node in network_instance.nodes.values():
        node.precharge_fill = node.charge_max_t[-1]
        node.precharge_surplus = node.discharge_max_t[-1]
    return None


@njit(fastmath=FASTMATH)
def set_flexible_precharge_fills_and_surpluses(network_instance: Network_InstanceType) -> None:
    for node in network_instance.nodes.values():
        node.precharge_fill = node.charge_max_t[-1]
        node.precharge_surplus = node.flexible_max_t[-1]
    return None


@njit(fastmath=FASTMATH)
def update_imports_exports_temp(network_instance: Network_InstanceType, interval: int64) -> None:
    for node in network_instance.nodes.values():
        node.imports_exports_update = node.imports_exports_temp - node.imports_exports[interval]
        node_m.set_imports_exports_temp(node, interval)
    return None


@njit(fastmath=FASTMATH)
def check_precharge_fill(network_instance: Network_InstanceType) -> boolean:
    for node in network_instance.nodes.values():
        if node.precharge_fill > TOLERANCE:
            return True
    return False


@njit(fastmath=FASTMATH)
def check_precharge_surplus(network_instance: Network_InstanceType) -> boolean:
    for node in network_instance.nodes.values():
        if node.precharge_surplus > TOLERANCE:
            return True
    return False
