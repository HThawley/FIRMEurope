from typing import Union

import numpy as np

from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.exceptions import (
    raise_getting_unloaded_data_error,
    raise_static_modification_error,
)
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.fast_methods import generator_m
from firm_ce.system.components import Generator_InstanceType, Storage_InstanceType
from firm_ce.system.topology import Node, Node_InstanceType


@njit(fastmath=FASTMATH)
def create_dynamic_copy(node_instance: Node_InstanceType) -> Node_InstanceType:
    node_copy = Node(False, node_instance.id, node_instance.order, node_instance.name)
    node_copy.data_status = node_instance.data_status
    node_copy.data = node_instance.data  # This remains static
    node_copy.residual_load = node_instance.residual_load.copy()
    return node_copy


@njit(fastmath=FASTMATH)
def load_data(node_instance: Node_InstanceType, trace: float64[:]) -> None:
    node_instance.data_status = "loaded"
    node_instance.data = trace
    node_instance.residual_load = trace.copy()
    return None


@njit(fastmath=FASTMATH)
def unload_data(node_instance: Node_InstanceType) -> None:
    node_instance.data_status = "unloaded"
    node_instance.data = np.empty((0,), dtype=np.float64)
    node_instance.residual_load = np.empty((0,), dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def get_data(node_instance: Node_InstanceType, data_type: unicode_type) -> Union[float64, None]:
    if node_instance.data_status == "unloaded":
        raise_getting_unloaded_data_error()

    if data_type == "trace":
        return node_instance.data
    elif data_type == "residual_load":
        return node_instance.residual_load
    else:
        raise RuntimeError("Invalid data_type argument for Node.get_data(data_type).")
    return None


@njit(fastmath=FASTMATH)
def allocate_memory(node_instance: Node_InstanceType, intervals_count: int64) -> None:
    if node_instance.static_instance:
        raise_static_modification_error()
    node_instance.imports_exports = np.zeros(intervals_count, dtype=np.float64)
    node_instance.deficits = np.zeros(intervals_count, dtype=np.float64)
    node_instance.spillage = np.zeros(intervals_count, dtype=np.float64)

    node_instance.flexible_power = np.zeros(intervals_count, dtype=np.float64)
    node_instance.storage_power = np.zeros(intervals_count, dtype=np.float64)
    node_instance.flexible_energy = np.zeros(intervals_count, dtype=np.float64)
    node_instance.storage_energy = np.zeros(intervals_count, dtype=np.float64)
    return None


@njit(fastmath=FASTMATH)
def initialise_netload_t(node_instance: Node_InstanceType, interval: int64) -> None:
    node_instance.netload_t = get_data(node_instance, "residual_load")[interval]
    return None


@njit(fastmath=FASTMATH)
def update_netload_t(node_instance: Node_InstanceType, interval: int64, precharging_flag: boolean) -> None:
    # Note: exports are negative, so they add to load
    node_instance.netload_t = (
        get_data(node_instance, "residual_load")[interval] - node_instance.imports_exports[interval]
    )

    if precharging_flag:
        node_instance.netload_t -= node_instance.storage_power[interval] + node_instance.flexible_power[interval]
    return None


@njit(fastmath=FASTMATH)
def fill_required(node_instance: Node_InstanceType) -> boolean:
    return node_instance.fill > TOLERANCE


@njit(fastmath=FASTMATH)
def surplus_available(node_instance) -> bool:
    return node_instance.surplus > TOLERANCE


@njit(fastmath=FASTMATH)
def assign_storage_merit_order(
    node_instance: Node_InstanceType, storages_typed_dict: DictType(int64, Storage_InstanceType)
) -> None:
    storages_count = len(storages_typed_dict)
    temp_orders = np.full(storages_count, -1, dtype=np.int64)
    temp_durations = np.full(storages_count, -1, dtype=np.float64)

    idx = 0
    for storage_order, storage in storages_typed_dict.items():
        if storage.node.order == node_instance.order:
            temp_orders[idx] = storage_order
            temp_durations[idx] = storage.duration
            idx += 1

    if idx == 0:
        return

    temp_orders = temp_orders[:idx]
    temp_durations = temp_durations[:idx]

    sort_order = np.argsort(temp_durations)
    node_instance.storage_merit_order = temp_orders[sort_order]
    return None


@njit(fastmath=FASTMATH)
def assign_flexible_merit_order(
    node_instance: Node_InstanceType, generators_typed_dict: DictType(int64, Generator_InstanceType)
) -> None:
    generators_count = len(generators_typed_dict)
    temp_orders = np.full(generators_count, -1, dtype=np.int64)
    temp_marginal_costs = np.full(generators_count, -1, dtype=np.float64)

    idx = 0
    for generator_order, generator in generators_typed_dict.items():
        if not generator_m.check_unit_type(generator, "flexible"):
            continue

        if generator.node.order == node_instance.order:
            temp_orders[idx] = generator_order
            temp_marginal_costs[idx] = (
                generator.cost.vom
                + generator.cost.fuel_cost_mwh
                + generator.cost.fuel_cost_h * 1000 * generator.unit_size
            )
            idx += 1

    if idx == 0:
        return

    temp_orders = temp_orders[:idx]
    temp_marginal_costs = temp_marginal_costs[:idx]

    sort_order = np.argsort(temp_marginal_costs)
    node_instance.flexible_merit_order = temp_orders[sort_order]
    return None


@njit(fastmath=FASTMATH)
def check_remaining_netload(node_instance: Node_InstanceType, interval: int64, check_case: unicode_type) -> boolean:
    if check_case == "deficit":
        return (
            node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]
            > TOLERANCE
        )
    elif check_case == "spillage":
        return (
            node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]
            < -TOLERANCE
        )
    elif check_case == "both":
        return (
            abs(
                node_instance.netload_t - node_instance.storage_power[interval] - node_instance.flexible_power[interval]
            )
            > TOLERANCE
        )
    return False


@njit(fastmath=FASTMATH)
def set_imports_exports_temp(node_instance: Node_InstanceType, interval: int64) -> None:
    node_instance.imports_exports_temp = node_instance.imports_exports[interval]
    return None


@njit(fastmath=FASTMATH)
def reset_dispatch_max_t(node_instance: Node_InstanceType) -> None:
    if len(node_instance.storage_merit_order) > 0:
        node_instance.discharge_max_t = np.zeros(len(node_instance.storage_merit_order), dtype=np.float64)
        node_instance.charge_max_t = np.zeros(len(node_instance.storage_merit_order), dtype=np.float64)
    else:
        node_instance.discharge_max_t = np.zeros(1, dtype=np.float64)
        node_instance.charge_max_t = np.zeros(1, dtype=np.float64)

    if len(node_instance.flexible_merit_order) > 0:
        node_instance.flexible_max_t = np.zeros(len(node_instance.flexible_merit_order), dtype=np.float64)
    else:
        node_instance.flexible_max_t = np.zeros(1, dtype=np.float64)
    return None
