# type: ignore
from firm_ce.common.constants import FASTMATH, TOLERANCE
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import boolean, float64, int64, unicode_type
from firm_ce.fast_methods import (
    fleet_m,
    generator_m,
    reservoir_m,
    network_m,
    node_m,
    static_m,
    storage_m,
)
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


@njit(fastmath=FASTMATH)
def initialise_interval(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
    forward_time_flag: boolean,
) -> None:
    """
    Initialise state of all Node, Storage, and flexible Generator instances for a time interval. This
    function is used for forward-time balancing, deficit block balancing, and when resolving the
    discontinuity created after the precharging period. The initialise_precharging_interval should
    instead be used for the precharging period.

    Parameters:
    -------
    interval (int64): Index of the time interval to initialise.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.
    forward_time_flag (boolean): True for forward-time balancing and when resolving the discontinuity
        created after the precharging period, False for reverse-time deficit block balancing.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: netload_t, discharge_max_t,
        charge_max_t, reservoir_max_t, flexible_max_t.
    Attributes modified for the flexible Generators referenced in Fleet.generators: flexible_max_t, node.
    Attributes modified for the Reservoirs referenced in Fleet.reservoirs: discharge_max_t, node.
    Attributes modified for the Storage systems referenced in Fleet.storages: discharge_max_t, charge_max_t,
        node.
    """
    for node in network.nodes.values():
        node_m.initialise_netload_t(node, interval)
        node_m.reset_dispatch_max_t(node)

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_flexible_max_t(
                fleet.generators[flexible_order], interval, resolution, idx, forward_time_flag
            )
        for idx, reservoir_order in enumerate(node.reservoir_merit_order):
            reservoir_m.set_reservoir_max_t(
                fleet.reservoirs[reservoir_order], interval, resolution, idx, forward_time_flag
            )
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_dispatch_max_t(fleet.storages[storage_order], interval, resolution, idx, forward_time_flag)
    return None


@njit(fastmath=FASTMATH)
def balance_with_transmission(
    interval: int64,
    network: Network_InstanceType,
    transmission_case: unicode_type,
    precharging_flag: boolean,
) -> None:
    """
    Perform a transmission balancing step for the specified case and update netloads for each Node.

    Notes:
    -------
    - If the precharging_flag flag is True (deficit block or precharging period), then the change in imports/exports
    resulting from the transmission step is calculated and stored.

    Parameters:
    -------
    interval (int64): Index of the time interval to initialise.
    network (Network_InstanceType): An instance of the Network jitclass.
    transmission_case (unicode_type): String that sets the transmission case defining the fill and surplus available
        at each Node. Refer to network_m.set_node_fills_and_surpluses pseudo-method for specifics on each case.
    precharging_flag (boolean): True if currently balancing a deficit block or precharging period. Otherwise, False.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for each Node in Network.nodes: fill, surplus, available_imports, imports_exports, temp_surplus,
        netload_t, imports_exports_update, imports_exports_temp.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Network.lines: temp_leg_flows, flows.
    """
    network_m.set_node_fills_and_surpluses(network, transmission_case, interval)
    network_m.fill_with_transmitted_surpluses(network, interval)
    network_m.update_netloads(network, interval, precharging_flag)

    if precharging_flag:
        network_m.update_imports_exports_temp(network, interval)
    return None


@njit(fastmath=FASTMATH)
def balance_with_storage(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    """
    Dispatch Storage systems according to the merit order at each Node to balance remaining netload.
    Positive netload requires Storage systems to discharge, negative netload provides electricity
    for charging.

    Notes:
    -----
    - Nodes that have no remaining netload for balancing are skipped.
    - Nodal storage power is reset before dispatching the Storage systems.

    Parameters:
    -------
    interval (int64): Index of the time interval to balance.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for Nodes in Network.nodes: storage_power.
    Attributes modified for the Storage systems in Fleet.storages: dispatch_power, node.
    """
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "both"):
            continue
        node.storage_power[interval] = 0
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.dispatch(fleet.storages[storage_order], interval, idx)
    return None


@njit(fastmath=FASTMATH)
def balance_with_reservoir(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    """
    Dispatch Reservoir systems according to the merit order at each Node to balance remaining netload.
    Positive netload requires Reservoir systems to discharge.

    Notes:
    -----
    - Nodes that have no remaining netload for balancing are skipped.
    - Nodal reservoir power is reset before dispatching the Reservoir systems.

    Parameters:
    -------
    interval (int64): Index of the time interval to balance.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for Nodes in Network.nodes: reservoir_power.
    Attributes modified for the Reservoir systems in Fleet.reservoirs: dispatch_power, node.
    """
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "deficit"):
            continue
        node.reservoir_power[interval] = 0
        for idx, reservoir_order in enumerate(node.reservoir_merit_order):
            reservoir_m.dispatch(fleet.reservoirs[reservoir_order], interval, idx)
    return None


@njit(fastmath=FASTMATH)
def balance_with_flexible(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    """
    Dispatch flexible Generators according to the merit order at each Node to balance remaining netload.
    Positive netload can be balanced by flexible Generators.

    Notes:
    -----
    - Nodes that have no remaining power deficits for balancing are skipped.
    - Nodal flexible power is reset before dispatching the flexible Generators.

    Parameters:
    -------
    interval (int64): Index of the time interval to balance.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for Nodes in Network.nodes: flexible_power.
    Attributes modified for the Generators in Fleet.generators: dispatch_power, node.
    """
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "deficit"):
            continue
        node.flexible_power[interval] = 0
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.dispatch(fleet.generators[flexible_order], interval, idx)
    return None


@njit(fastmath=FASTMATH)
def energy_balance_for_interval(
    solution,
    interval: int64,
    forward_time_flag: boolean,
) -> None:
    """
    The core sequence of unit committment business rules for balancing the residual load in a
    time interval. The high-level process is:

        1. Transmit surplus generation to balance load at other Nodes.
        2. Balance residual load at each Node with Storage systems local to that Node.
        3. Balance residual load at each Node with transmission to Storage systems at
        other Nodes in the network.
        4. Balance residual load at each Node with flexible Generators local to that Node.
        5. Balance residual load at each Node with transmission to flexible Generators at
        other Nodes in the network.
        6. Transmit remaining surplus generation to charge Storage systems at other Nodes
        in the network.

    Parameters:
    -------
    solution (Solution_InstanceType): An instance of the Solution jitclass providing a complete description
        of the system for this candidate solution.
    interval (int64): Index of the time interval to balance.
    forward_time_flag (boolean): True for forward-time balancing and when resolving the discontinuity
        created after the precharging period, False for reverse-time deficit block balancing.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Solution.network.nodes: netload_t, discharge_max_t,
        charge_max_t, reservoir_max_t, flexible_max_t, fill, surplus, available_imports, imports_exports,
        temp_surplus, imports_exports_update, imports_exports_temp, storage_power, reservoir_power, flexible_power.
    Attributes modified for the flexible Generators referenced in Solution.fleet.generators: flexible_max_t, node,
        dispatch_power.
    Attributes modified for the Reservoirs referenced in Solution.fleet.reservoirs: discharge_max_t, node,
        dispatch_power.
    Attributes modified for the Storage systems referenced in Solution.fleet.storages: discharge_max_t, charge_max_t,
        node, dispatch_power.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Solution.network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Solution.network.lines: temp_leg_flows, flows.
    """
    initialise_interval(
        interval,
        solution.network,
        solution.fleet,
        solution.static.interval_resolutions[interval],
        forward_time_flag,
    )
    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "surplus", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Local storage

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "storage_discharge", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Neighbouring and local storage
        balance_with_reservoir(interval, solution.network, solution.fleet)  # Local reservoir

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "reservoir", False)
        balance_with_reservoir(interval, solution.network, solution.fleet)  # Local reservoir
        balance_with_flexible(interval, solution.network, solution.fleet)  # Local flexible

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "flexible", False)
        balance_with_flexible(interval, solution.network, solution.fleet)  # Neighbouring and local flexible

    if network_m.check_remaining_netloads(solution.network, interval, "spillage"):
        balance_with_transmission(interval, solution.network, "storage_charge", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Charge neighbouring storage

    return None


@njit(fastmath=FASTMATH)
def balance_for_period(
    start_t: int64,
    end_t: int64,
    precharging_allowed: boolean,
    solution,
    year: int64,
) -> None:
    """
    Iterates through the time intervals within a specified period and performs the unit committment.
    When a deficit block is encountered, precharging actions are also initiated.

    Parameters:
    -------
    start_t (int64): First time interval index for the period (inclusive).
    end_t (int64): Final time interval index for the period (exclusive).
    precharging_allowed (boolean): True if the precharging business rules can be execute. During balancing of a
        deficit block, this argument is False.
    solution (Solution_InstanceType): An instance of the Solution jitclass providing a complete description
        of the system for this candidate solution.
    year (int64): Year index used manage flexible Generator remaining energy constraints during precharging.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Solution.network.nodes: netload_t, discharge_max_t,
        charge_max_t, flexible_max_t, fill, surplus, available_imports, imports_exports, temp_surplus,
        deficits, spillage, imports_exports_update, imports_exports_temp, storage_power, flexible_power.
    Attributes modified for the flexible Generators referenced in Solution.fleet.generators: flexible_max_t, node,
        dispatch_power, remaining_energy, remaining_energy_temp_reverse.
    Attributes modified for the Storage systems referenced in Solution.fleet.storages: discharge_max_t, charge_max_t,
        node, dispatch_power, stored_energy, stored_energy_temp_reverse.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Solution.network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Solution.network.lines: temp_leg_flows, flows.

    ATTRIBUTES MODIFIED DURING PRECHARGING
    """
    perform_precharge = False
    for t in range(start_t, end_t):
        energy_balance_for_interval(solution, t, True)
        network_m.calculate_spillage_and_deficit(solution.network, t)
        fleet_m.update_stored_energies(solution.fleet, t, solution.static.interval_resolutions[t], True)
        fleet_m.update_remaining_flexible_energies(
            solution.fleet, t, solution.static.interval_resolutions[t], True, False
        )

        if not precharging_allowed:
            continue

        if not perform_precharge and network_m.check_remaining_netloads(solution.network, t, "deficit"):
            perform_precharge = True

        if perform_precharge and (
            not network_m.check_remaining_netloads(solution.network, t, "deficit") or t == end_t - 1
        ):
            precharge_storage(solution, t, year)
            perform_precharge = False
    return None


@njit(fastmath=FASTMATH)
def determine_precharge_energies_for_deficit_block(
    interval: int64,
    solution,
    year: int64,
) -> int64:
    """
    Iterate backwards through the time intervals in a deficit block, dispatching Storage systems
    and flexible Generators according to reverse-time rules. Determines the amount of energy each
    Storage precharger requires to balance residual loads in the deficit block, as well as the
    energy available for trickle-charging from Storage systems and flexible Generators.

    Parameters:
    -------
    interval (int64): Interval immediately after the deficit block.
    solution (Solution_InstanceType): An instance of the Solution jitclass providing a complete description
        of the system for this candidate solution.
    year (int64): Year index for the time interval immediately after the deficit block.

    Returns:
    -------
    int64: Index of the first interval of the deficit block. Returns 0 if deficit block intersects with the
        start of the modelling period.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Solution.network.nodes: netload_t, discharge_max_t,
        charge_max_t, flexible_max_t, fill, surplus, available_imports, imports_exports, temp_surplus,
        imports_exports_update, imports_exports_temp, storage_power, flexible_power.
    Attributes modified for the flexible Generators referenced in Solution.fleet.generators: flexible_max_t, node,
        dispatch_power, remaining_energy, remaining_energy_temp_reverse, deficit_block_min_energy,
        deficit_block_max_energy, remaining_energy_temp_forward, trickling_reserves.
    Attributes modified for the Storage systems referenced in Solution.fleet.storages: discharge_max_t, charge_max_t,
        node, dispatch_power, stored_energy, stored_energy_temp_reverse, deficit_block_min_storage,
        deficit_block_max_storage, stored_energy_temp_forward, precharge_flag, precharge_energy, trickling_reserves.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Solution.network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Solution.network.lines: temp_leg_flows, flows.
    """
    fleet_m.initialise_deficit_block(solution.fleet, interval)
    previous_year_flag = False
    precharging_year = year
    first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)

    while True:
        if interval == first_t:
            previous_year_flag = True
            precharging_year -= 1
            first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)

        interval -= 1

        if interval < 0:
            return 0

        network_m.reset_transmission(solution.network, interval)
        network_m.reset_dispatch(solution.network, interval)
        fleet_m.reset_dispatch(solution.fleet, interval)

        energy_balance_for_interval(solution, interval, False)

        fleet_m.update_stored_energies(solution.fleet, interval, solution.static.interval_resolutions[interval], False)
        fleet_m.update_remaining_flexible_energies(
            solution.fleet, interval, solution.static.interval_resolutions[interval], False, previous_year_flag
        )
        fleet_m.update_deficit_block(solution.fleet)

        if network_m.check_precharging_end(solution.network, interval):
            fleet_m.assign_precharging_values(
                solution.fleet, interval, solution.static.interval_resolutions[interval], year
            )
            return interval

        previous_year_flag = False


@njit(fastmath=FASTMATH)
def initialise_precharging_interval(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Initialise state of all Node, Storage, and flexible Generator instances for a time interval in the
    precharging period.

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period to initialise.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: imports_exports_temp, netload_t, discharge_max_t,
        charge_max_t, flexible_max_t.
    Attributes modified for the flexible Generators referenced in Fleet.generators: flexible_max_t, node.
    Attributes modified for the Storage systems referenced in Fleet.storages: discharge_max_t, charge_max_t,
        node.
    """
    fleet_m.update_precharging_flags(fleet, interval)

    for node in network.nodes.values():
        node_m.set_imports_exports_temp(node, interval)
        node_m.update_netload_t(node, interval, True)
        node_m.reset_dispatch_max_t(node)

        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_precharging_max_t(fleet.storages[storage_order], interval, resolution, idx)
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_precharging_max_t(fleet.generators[flexible_order], interval, resolution, idx)
    return None


@njit(fastmath=FASTMATH)
def perform_local_surplus_transfers(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Use any existing local (intranode) surplus generation to charge Storage prechargers.

    Notes:
    -----
    - The reverse of the storage merit order is used within the precharging period, due to iterating
    backwards through reverse time (short-duration storage should still discharge earlier than
    long-duration storage).

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: existing_surplus, storage_power, charge_max_t,
        precharge_fill.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy.
    """
    for node in network.nodes.values():
        node.existing_surplus = min(0, node.netload_t)
        if node.existing_surplus > -TOLERANCE:
            continue

        for idx_reverse, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag:
                continue

            idx = len(node.storage_merit_order) - idx_reverse - 1

            dispatch_power_update = min(max(node.existing_surplus, -fleet.storages[storage_order].charge_max_t), 0.0)
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            node.existing_surplus -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_transmitted_surplus_transfers(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Transmit any existing surplus generation between Nodes to charge Storage prechargers.

    Notes:
    -----
    - The reverse of the storage merit order is used within the precharging period, due to iterating
    backwards through reverse time (short-duration storage should still discharge earlier than
    long-duration storage).

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: imports_exports_update, storage_power, charge_max_t,
        precharge_fill, fill, surplus, available_imports, imports_exports, temp_surplus, netload_t,
        imports_exports_temp.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Network.lines: temp_leg_flows, flows.
    """
    if not network_m.check_existing_surplus(network):
        return None

    balance_with_transmission(interval, network, "precharging_surplus", True)

    for node in network.nodes.values():
        for idx_reverse, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag:
                continue

            idx = len(node.storage_merit_order) - idx_reverse - 1

            dispatch_power_update = min(
                max(node.imports_exports_update, -fleet.storages[storage_order].charge_max_t), 0.0
            )
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_intranode_interstorage_transfers(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Transfer energy within a Node from Storage trickle-chargers to Storage prechargers.

    Notes:
    -----
    - The reverse of the storage merit order is used within the precharging period, due to iterating
    backwards through reverse time (short-duration storage should still discharge earlier than
    long-duration storage).

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: storage_power, charge_max_t,
        precharge_fill, discharge_max_t, precharge_surplus.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy, discharge_max_t, trickling_reserves.
    """
    for node in network.nodes.values():
        intranode_transfer_power = min(node.precharge_surplus, node.precharge_fill)
        intranode_trickle = intranode_transfer_power
        intranode_precharge = -intranode_transfer_power

        for idx_reverse, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag and not fleet.storages[storage_order].trickling_flag:
                continue

            idx = len(node.storage_merit_order) - idx_reverse - 1

            if fleet.storages[storage_order].trickling_flag:
                dispatch_power_update = max(min(intranode_trickle, fleet.storages[storage_order].discharge_max_t), 0.0)
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, False, idx
                )
                intranode_trickle -= dispatch_power_update

            if fleet.storages[storage_order].precharge_flag:
                dispatch_power_update = min(max(intranode_precharge, -fleet.storages[storage_order].charge_max_t), 0.0)
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
                )
                intranode_precharge -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_internode_interstorage_transfers(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Transfer energy between Nodes from Storage trickle-chargers to Storage prechargers.

    Notes:
    -----
    - The reverse of the storage merit order is used within the precharging period, due to iterating
    backwards through reverse time (short-duration storage should still discharge earlier than
    long-duration storage).

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: imports_exports_update, storage_power, charge_max_t,
        precharge_fill, fill, surplus, available_imports, imports_exports, temp_surplus, netload_t,
        imports_exports_temp, discharge_max_t, precharge_surplus.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy, discharge_max_t, trickling_reserves.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Network.lines: temp_leg_flows, flows.
    """
    if not (network_m.check_precharge_fill(network) and network_m.check_precharge_surplus(network)):
        return None

    balance_with_transmission(interval, network, "precharging_transfers", True)

    for node in network.nodes.values():

        for idx_reverse, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag and not fleet.storages[storage_order].trickling_flag:
                continue

            idx = len(node.storage_merit_order) - idx_reverse - 1

            if fleet.storages[storage_order].trickling_flag:
                dispatch_power_update = max(
                    min(node.imports_exports_update, fleet.storages[storage_order].discharge_max_t), 0.0
                )
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, False, idx
                )
                node.imports_exports_update -= dispatch_power_update

            if fleet.storages[storage_order].precharge_flag:
                dispatch_power_update = min(
                    max(node.imports_exports_update, -fleet.storages[storage_order].charge_max_t), 0.0
                )
                storage_m.update_precharge_dispatch(
                    fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
                )
                node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_intranode_flexible_transfers(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Transfer energy within a Node from flexible Generator trickle-chargers to Storage prechargers.

    Notes:
    -----
    - The reverse of the storage merit order is used within the precharging period, due to iterating
    backwards through reverse time (short-duration storage should still discharge earlier than
    long-duration storage).

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: storage_power, charge_max_t,
        precharge_fill, flexible_max_t, precharge_surplus, flexible_power.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy.
    Attributes modified for the flexible Generators referenced in Fleet.generators: dispatch_power, node,
        flexible_max_t, trickling_reserves.
    """
    for node in network.nodes.values():
        intranode_transfer_power = min(node.precharge_surplus, node.precharge_fill)
        intranode_trickle = intranode_transfer_power
        intranode_precharge = -intranode_transfer_power

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            if not fleet.generators[flexible_order].trickling_flag:
                continue

            dispatch_power_update = max(min(intranode_trickle, fleet.generators[flexible_order].flexible_max_t), 0.0)
            generator_m.update_precharge_dispatch(
                fleet.generators[flexible_order], interval, resolution, dispatch_power_update, idx
            )
            intranode_trickle -= dispatch_power_update

        for idx, storage_order in enumerate(node.storage_merit_order):
            if not fleet.storages[storage_order].precharge_flag:
                continue

            dispatch_power_update = min(max(intranode_precharge, -fleet.storages[storage_order].charge_max_t), 0.0)
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            intranode_precharge -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_internode_flexible_transfers(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: float64,
) -> None:
    """
    Transfer energy between Nodes from flexible Generator trickle-chargers to Storage prechargers.

    Notes:
    -----
    - The reverse of the storage merit order is used within the precharging period, due to iterating
    backwards through reverse time (short-duration storage should still discharge earlier than
    long-duration storage).

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: imports_exports_update, storage_power, charge_max_t,
        precharge_fill, fill, surplus, available_imports, imports_exports, temp_surplus, netload_t,
        imports_exports_temp, flexible_max_t, precharge_surplus, flexible_power.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy.
    Attributes modified for the flexible Generators referenced in Fleet.generators: dispatch_power, node,
        flexible_max_t, trickling_reserves.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Network.lines: temp_leg_flows, flows.
    """
    if not (network_m.check_precharge_fill(network) and network_m.check_precharge_surplus(network)):
        return None

    balance_with_transmission(interval, network, "precharging_transfers", True)

    for node in network.nodes.values():
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            if not fleet.generators[flexible_order].trickling_flag:
                continue
            dispatch_power_update = max(
                min(node.imports_exports_update, fleet.generators[flexible_order].flexible_max_t), 0.0
            )
            generator_m.update_precharge_dispatch(
                fleet.generators[flexible_order], interval, resolution, dispatch_power_update, idx
            )
            node.imports_exports_update -= dispatch_power_update

        for idx_reverse, storage_order in enumerate(node.storage_merit_order[::-1]):
            if not fleet.storages[storage_order].precharge_flag:
                continue
            idx = len(node.storage_merit_order) - idx_reverse - 1

            dispatch_power_update = min(
                max(node.imports_exports_update, -fleet.storages[storage_order].charge_max_t), 0.0
            )
            storage_m.update_precharge_dispatch(
                fleet.storages[storage_order], interval, resolution, dispatch_power_update, True, idx
            )
            node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def perform_flexible_precharging(
    solution,
    interval: int64,
) -> None:
    """
    Use flexible Generators to trickle-charge Storage prechargers for a given time interval within the
    precharging period.

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.
    resolution (float64): Length of the time interval, units hours.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: imports_exports_update, storage_power, charge_max_t,
        precharge_fill, fill, surplus, available_imports, imports_exports, temp_surplus, netload_t,
        imports_exports_temp, flexible_max_t, precharge_surplus, flexible_power.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy.
    Attributes modified for the flexible Generators referenced in Fleet.generators: dispatch_power, node,
        flexible_max_t, trickling_reserves.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Network.lines: temp_leg_flows, flows.
    """
    for node in solution.network.nodes.values():
        node_m.set_imports_exports_temp(node, interval)
    network_m.set_flexible_precharge_fills_and_surpluses(solution.network)

    perform_intranode_flexible_transfers(
        interval,
        solution.network,
        solution.fleet,
        solution.static.interval_resolutions[interval],
    )

    perform_internode_flexible_transfers(
        interval,
        solution.network,
        solution.fleet,
        solution.static.interval_resolutions[interval],
    )

    return None


@njit(fastmath=FASTMATH)
def determine_power_adjustments_for_precharging_period(
    interval: int64,
    solution,
    year: int64,
) -> int64:
    """
    Iterate backwards through the time intervals in the precharging period, adjusting the dispatch power of
    Storage systems and flexible Generators to transfer enough energy to Storage precharges to balance the
    deficit block.

    Notes:
    -------
    - Storage systems and flexible Generators that transfer energy into other Storage systems are called
    trickle-chargers.
    - Storage systems that receive transfered energy are called prechargers.
    - Dispatch powers that were originally determined during forward-time balancing are adjusted in the
    precharging period, rather than recalculated from scratch. This is different to the behaviour during
    the deficit block balancing, even though both periods iterate backwards through time.
    - Stored energy (Storage) and remaining energy (Generator) are not updated in the precharging period.
    This is because a discontinuity will exist between the original forward-time balancing and the adjustment
    resulting from the reverse-time balancing actions. The energy value discontinuity is instead resolved in
    a subsequent step.

    Parameters:
    -------
    interval (int64): Index of the first interval of the deficit block.
    solution (Solution_InstanceType): An instance of the Solution jitclass providing a complete description
        of the system for this candidate solution.
    year (int64): Year index for the time interval immediately after the deficit block.

    Returns:
    -------
    int64: Index of the first interval of the precharging period. Returns 0 if precharging period intersects with the
        start of the modelling period.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: imports_exports_update, storage_power, charge_max_t,
        precharge_fill, fill, surplus, available_imports, imports_exports, temp_surplus, netload_t,
        imports_exports_temp, flexible_max_t, precharge_surplus, flexible_power, discharge_max_t, existing_surplus.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power, node, charge_max_t,
        precharge_energy, discharge_max_t, trickling_reserves, trickling_flag, precharge_flag, remaining_trickling_reserves.
    Attributes modified for the flexible Generators referenced in Fleet.generators: dispatch_power, node,
        flexible_max_t, trickling_reserves, trickling_flag, remaining_trickling_reserves.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    Attributes modified for all Line instances Network.lines: temp_leg_flows, flows.
    """
    first_interval_precharge = 0
    fleet_m.initialise_precharging_flags(solution.fleet, interval)
    precharging_year = year
    first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)

    while True:
        if interval == first_t:
            precharging_year -= 1
            first_t, _ = static_m.get_year_t_boundaries(solution.static, precharging_year)
            fleet_m.reset_flexible_reserves(solution.fleet)

        interval -= 1

        if interval < 0:
            first_interval_precharge = 0
            break

        initialise_precharging_interval(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )
        network_m.set_storage_precharge_fills_and_surpluses(solution.network)

        perform_local_surplus_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        perform_transmitted_surplus_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        perform_intranode_interstorage_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        perform_internode_interstorage_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
        )

        if fleet_m.check_precharge_remaining(solution.fleet):
            perform_flexible_precharging(solution, interval)

        if (not fleet_m.check_precharge_remaining(solution.fleet)) or (
            not fleet_m.check_trickling_remaining(solution.fleet)
        ):
            first_interval_precharge = interval
            break

    return first_interval_precharge


@njit(fastmath=FASTMATH)
def perform_fill_adjustment(
    interval: int64,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    """
    Increases Storage charging power (reduces discharge power) if there is still fill energy required by the
    node after a transmission step. This function is called when resolving the energy discontinuity at the
    start of the precharging period if the flexible trickle-charging dispatch is found to be infeasible. That
    is, the trickle-charging from a flexible Generator would breach the remaining energy constraint for that
    Generator once the discontinuity is resolved.

    Parameters:
    -------
    interval (int64): Index of the time interval within the precharging period.
    network (Network_InstanceType): An instance of the Network jitclass.
    fleet (Fleet_InstanceType): An instance of the Fleet jitclass.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Network.nodes: storage_power, fill.
    Attributes modified for the Storage systems referenced in Fleet.storages: dispatch_power.
    """
    for node in network.nodes.values():
        for storage_order in node.storage_merit_order:
            dispatch_power_update = min(node.fill, fleet.storages[storage_order].remaining_charge_max_t)
            fleet.storages[storage_order].dispatch_power[interval] += dispatch_power_update
            node.storage_power[interval] += dispatch_power_update
            node.fill -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH)
def resolve_energy_discontinuities(
    first_interval_precharge: int64,
    interval_after_deficit_block: int64,
    solution,
) -> None:
    """
    Iterate forwards through the precharging period and deficit block, checking whether the dispatch
    powers of Storage systems and flexible Generators determined for this period are feasible (i.e.,
    the stored energy and remaining energy constraints contiguous with the interval immediately prior
    to the precharging period would not be breached). When infeasible dispatch is found, adjust
    the dispatch to resolve the violation.

    Notes:
    -------
    - An infeasible dispatch power likely indicates that the deficit block cannot be resolved through
    precharging. The operational load is simply to substantial for the system to balance according to
    the unit committment business rules.

    Parameters:
    -------
    first_interval_precharge (int64): First interval of the precharging period.
    interval_after_deficit_block (int64): First interval after the deficit block.
    solution (Solution_InstanceType): An instance of the Solution jitclass providing a complete description
        of the system for this candidate solution.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the Nodes referenced in Solution.network.nodes: storage_power, flexible_power, fill, surplus, netload_t,
        deficits, spillage, discharge_max_t, charge_max_t, flexible_max_t, imports_exports, available_imports, temp_surplus,
        imports_exports_update, imports_exports_temp.
    Attributes modified for the Storage systems referenced in Solution.fleet.storages: dispatch_power, remaining_discharge_max_t,
        remaining_charge_max_t, stored_energy, discharge_max_t, charge_max_t, node.
    Attributes modified for the flexible Generators referenced in Solution.fleet.generators: dispatch_power, flexible_max_t, node,
        remaining_energy.
    Attributes modified for each Line instance in Solution.network.lines: flows, temp_leg_flows.
    Attributes modified for each Route instance in the route lists (the lists corresponds to a particular start node and
        route length) contained in Network.routes: flow_update, initial_node, nodes, lines.
    """
    for interval in range(first_interval_precharge, interval_after_deficit_block):
        initialise_interval(
            interval,
            solution.network,
            solution.fleet,
            solution.static.interval_resolutions[interval],
            True,
        )

        infeasible_flag = fleet_m.determine_feasible_storage_dispatch(solution.fleet, interval)
        if infeasible_flag:
            network_m.reset_transmission(solution.network, interval)
            network_m.reset_flexible(solution.network, interval)
            fleet_m.reset_flexible(solution.fleet, interval)

            balance_with_transmission(interval, solution.network, "precharging_adjust_storage", False)
            balance_with_flexible(interval, solution.network, solution.fleet)  # Local flexible

            if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
                balance_with_transmission(interval, solution.network, "flexible", False)
                balance_with_flexible(interval, solution.network, solution.fleet)  # Neighbouring and local flexible
        else:
            infeasible_flag = fleet_m.determine_feasible_flexible_dispatch(solution.fleet, interval)
            if infeasible_flag:
                network_m.reset_transmission(solution.network, interval)
                fleet_m.calculate_available_storage_dispatch(solution.fleet, interval)

                balance_with_transmission(interval, solution.network, "precharging_adjust_surplus", False)
                balance_with_transmission(interval, solution.network, "precharging_adjust_flexible", False)

                perform_fill_adjustment(
                    interval,
                    solution.network,
                    solution.fleet,
                )
            else:
                network_m.update_netloads(solution.network, interval, False)

        network_m.calculate_spillage_and_deficit(solution.network, interval)

        fleet_m.update_stored_energies(solution.fleet, interval, solution.static.interval_resolutions[interval], True)
        fleet_m.update_remaining_flexible_energies(
            solution.fleet, interval, solution.static.interval_resolutions[interval], True, False
        )
    return None


@njit(fastmath=FASTMATH)
def precharge_storage(
    solution,
    interval_after_deficit_block: int64,
    year: int64,
) -> None:
    """
    Core unit committment business rules for precharging Storage systems. The high-level process is:

        1. Iterate backwards through a period of intervals that have power deficits (i.e., deficit block)
        and dispatch assets using reverse-time symmetry rules. That is, discharging adds energy to the previous
        time interval and charging removes energy. Determine the amount of energy required to be transfered into
        Storage prechargers such that they can complete the deficit block balancing.
        2. Upon reaching the start of the deficit block, iterate backwards through the precharging period adjusting
        the dispatch power of Storage systems and flexible Generators to transfer sufficient energy from
        trickle-chargers to prechargers.
        3. Once precharging is completed, iterate back forwards through time from the start of the precharging period
        checking that the stored energy and remaining energy constraints contiguous with the period immediately prior
        to precharging will not be violated. Adjust dispatch decisions to maintain feasibility.

    Parameters:
    -------
    solution (Solution_InstanceType): An instance of the Solution jitclass providing a complete description
        of the system for this candidate solution.
    interval_after_deficit_block (int64): First interval after the deficit block.
    year (int64): Year index for the time interval immediately after the deficit block.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Substantial modification of dynamic and precharging attributes in the Fleet and Network jitclasses. Refer to the
    docstrings of the functions called below for details.
    """
    first_interval_deficit_block = determine_precharge_energies_for_deficit_block(
        interval_after_deficit_block, solution, year
    )

    first_interval_precharge = determine_power_adjustments_for_precharging_period(
        first_interval_deficit_block, solution, year
    )

    resolve_energy_discontinuities(first_interval_precharge, interval_after_deficit_block + 1, solution)
    return None
