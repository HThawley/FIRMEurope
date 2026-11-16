# type: ignore
from firm_ce.common.constants import FASTMATH, TOLERANCE, BOUNDSCHECK
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import boolean, nbfloat, nbintp, unicode_type
from firm_ce.scalar.entities.components import Fleet_InstanceType
from firm_ce.scalar.entities.topology import Network_InstanceType
from firm_ce.scalar.methods import (
    fleet_m,
    generator_m,
    network_m,
    node_m,
    static_m,
    storage_m,
)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_interval(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
    forward_time_flag: boolean,
) -> None:
    """
    Initialise state of all Node, Storage, and flexible Generator instances for a time interval. This
    function is used for forward-time balancing, deficit block balancing, and when resolving the
    discontinuity created after the precharging period. The initialise_precharging_interval should
    instead be used for the precharging period.
    """
    for fuel in fleet.fuels.values():
        if forward_time_flag and len(fuel.annual_limit) > 0:
            fuel.remaining_energy[interval] = fuel.remaining_energy[interval - 1]
        if forward_time_flag:
            fuel.allocated_energy = fuel.remaining_energy[interval]
        else:
            fuel.allocated_energy = fuel.remaining_energy_temp_reverse

    for node in network.nodes.values():
        node_m.initialise_netload_t(node, interval)
        node_m.reset_dispatch_max_t(node)

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_flexible_max_t(
                fleet.generators[flexible_order], interval, resolution, idx, forward_time_flag
            )
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_dispatch_max_t(fleet.storages[storage_order], interval, resolution, idx, forward_time_flag)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def balance_with_transmission(
    interval: nbintp,
    network: Network_InstanceType,
    transmission_case: unicode_type,
    precharging_flag: boolean,
) -> None:
    """
    Perform a transmission balancing step for the specified case and update netloads for each Node.
    """
    network_m.set_node_fills_and_surpluses(network, transmission_case, interval)
    network_m.fill_with_transmitted_surpluses(network, interval)
    network_m.update_netloads(network, interval, precharging_flag)

    if precharging_flag:
        network_m.update_imports_exports_temp(network, interval)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def balance_with_storage(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    """
    Dispatch Storage systems according to the merit order at each Node to balance remaining netload.
    """
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "both"):
            continue
        node.storage_power[interval] = 0
        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.dispatch(fleet.storages[storage_order], interval, idx)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def balance_with_flexible(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
    forward_time_flag: boolean,
) -> None:
    """
    Dispatch flexible Generators according to the merit order at each Node to balance remaining netload.
    """
    for node in network.nodes.values():
        if not node_m.check_remaining_netload(node, interval, "deficit"):
            continue
        node.flexible_power[interval] = 0
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.dispatch(fleet.generators[flexible_order], interval, idx, resolution, forward_time_flag)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def energy_balance_for_interval(
    solution,
    interval: nbintp,
    forward_time_flag: boolean,
) -> None:
    """
    The core sequence of unit committment business rules for balancing the residual load in a
    time interval.
    """
    initialise_interval(
        interval,
        solution.network,
        solution.fleet,
        solution.static.resolution,
        forward_time_flag,
    )
    # Check deficits
    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        # Transmit energy from nodes in surplus to nodes in deficit
        balance_with_transmission(interval, solution.network, "surplus", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Local storage

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        # Discharge local storage to balance remaining deficits
        balance_with_transmission(interval, solution.network, "storage_discharge", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Neighbouring and local storage
        # Local flexible
        balance_with_flexible(interval, solution.network, solution.fleet, solution.static.resolution, forward_time_flag)

    if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
        balance_with_transmission(interval, solution.network, "flexible", False)
        # Neighbouring and local flexible
        balance_with_flexible(interval, solution.network, solution.fleet, solution.static.resolution, forward_time_flag)

    # check surpluses
    if network_m.check_remaining_netloads(solution.network, interval, "curtail"):
        # export surpluses to neighbours who can use it
        balance_with_transmission(interval, solution.network, "storage_charge", False)
        balance_with_storage(interval, solution.network, solution.fleet)  # Charge neighbouring storage

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def balance_for_period(
    start_t: nbintp,
    end_t: nbintp,
    precharging_allowed: boolean,
    solution,
    year: nbintp,
) -> None:
    """
    Iterates through the time intervals within a specified period and performs the unit committment.
    When a deficit block is encountered, precharging actions are also initiated.
    """
    perform_precharge = False
    for t in range(start_t, end_t):
        energy_balance_for_interval(solution, t, True)
        network_m.calculate_curtail_and_deficit(solution.network, t)
        fleet_m.update_stored_energies(solution.fleet, t, solution.static.resolution, True)

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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def determine_precharge_energies_for_deficit_block(
    interval: nbintp,
    solution,
    year: nbintp,
) -> nbintp:
    """
    Iterate backwards through the time intervals in a deficit block, dispatching Storage systems
    and flexible Generators according to reverse-time rules.
    """
    fleet_m.initialise_deficit_block(solution.fleet, interval)

    while True:
        interval -= 1

        if interval < 0:
            return 0

        network_m.reset_transmission(solution.network, interval)
        network_m.reset_dispatch(solution.network, interval)
        fleet_m.reset_dispatch(solution.fleet, interval)

        energy_balance_for_interval(solution, interval, False)

        fleet_m.update_stored_energies(solution.fleet, interval, solution.static.resolution, False)
        fleet_m.update_deficit_block(solution.fleet)

        if network_m.check_precharging_end(solution.network, interval):
            fleet_m.assign_precharging_values(solution.fleet, interval, solution.static.resolution, year)
            return interval


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def initialise_precharging_interval(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Initialise state of all Node, Storage, and flexible Generator instances for a time interval in the
    precharging period.
    """
    fleet_m.update_precharging_flags(fleet, interval)

    for fuel in fleet.fuels.values():
        fuel.allocated_trickling = fuel.remaining_trickling_reserves

    for node in network.nodes.values():
        node_m.set_imports_exports_temp(node, interval)
        node_m.update_netload_t(node, interval, True)
        node_m.reset_dispatch_max_t(node)

        for idx, storage_order in enumerate(node.storage_merit_order):
            storage_m.set_precharging_max_t(fleet.storages[storage_order], interval, resolution, idx)
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator_m.set_precharging_max_t(fleet.generators[flexible_order], interval, resolution, idx)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_local_surplus_transfers(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Use any existing local (intranode) surplus generation to charge Storage prechargers.
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_transmitted_surplus_transfers(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Transmit any existing surplus generation between Nodes to charge Storage prechargers.
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_intranode_interstorage_transfers(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Transfer energy within a Node from Storage trickle-chargers to Storage prechargers.
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_internode_interstorage_transfers(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Transfer energy between Nodes from Storage trickle-chargers to Storage prechargers.
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_intranode_flexible_transfers(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Transfer energy within a Node from flexible Generator trickle-chargers to Storage prechargers.
    """
    for node in network.nodes.values():
        intranode_transfer_power = min(node.precharge_surplus, node.precharge_fill)
        intranode_trickle = intranode_transfer_power
        intranode_precharge = -intranode_transfer_power

        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator = fleet.generators[flexible_order]
            if not generator.fuel.trickling_flag:
                continue
            generator_m.set_live_trickling_max_t(generator, interval, resolution)
            dispatch_power_update = max(min(intranode_trickle, generator.flexible_max_t), 0.0)
            generator_m.update_fuel_reserve(generator, interval, resolution, dispatch_power_update, True)
            generator_m.update_precharge_dispatch(generator, interval, resolution, dispatch_power_update, idx)
            intranode_trickle -= dispatch_power_update

        for idx, storage_order in enumerate(node.storage_merit_order):
            storage = fleet.storages[storage_order]
            if not storage.precharge_flag:
                continue

            dispatch_power_update = min(max(intranode_precharge, -storage.charge_max_t), 0.0)
            storage_m.update_precharge_dispatch(storage, interval, resolution, dispatch_power_update, True, idx)
            intranode_precharge -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_internode_flexible_transfers(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
    resolution: nbfloat,
) -> None:
    """
    Transfer energy between Nodes from flexible Generator trickle-chargers to Storage prechargers.
    """
    if not (network_m.check_precharge_fill(network) and network_m.check_precharge_surplus(network)):
        return None

    balance_with_transmission(interval, network, "precharging_transfers", True)

    for node in network.nodes.values():
        for idx, flexible_order in enumerate(node.flexible_merit_order):
            generator = fleet.generators[flexible_order]
            if not generator.fuel.trickling_flag:
                continue
            generator_m.set_live_trickling_max_t(generator, interval, resolution)
            dispatch_power_update = max(min(node.imports_exports_update, generator.flexible_max_t), 0.0)
            generator_m.update_fuel_reserve(generator, interval, resolution, dispatch_power_update, True)
            generator_m.update_precharge_dispatch(generator, interval, resolution, dispatch_power_update, idx)
            node.imports_exports_update -= dispatch_power_update

        for idx_reverse, storage_order in enumerate(node.storage_merit_order[::-1]):
            storage = fleet.storages[storage_order]
            if not storage.precharge_flag:
                continue
            idx = len(node.storage_merit_order) - idx_reverse - 1

            dispatch_power_update = min(max(node.imports_exports_update, -storage.charge_max_t), 0.0)
            storage_m.update_precharge_dispatch(storage, interval, resolution, dispatch_power_update, True, idx)
            node.imports_exports_update -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_flexible_precharging(
    solution,
    interval: nbintp,
) -> None:
    """
    Use flexible Generators to trickle-charge Storage prechargers for a given time interval within the
    precharging period.
    """
    for node in solution.network.nodes.values():
        node_m.set_imports_exports_temp(node, interval)
    network_m.set_flexible_precharge_fills_and_surpluses(solution.network)

    perform_intranode_flexible_transfers(
        interval,
        solution.network,
        solution.fleet,
        solution.static.resolution,
    )

    perform_internode_flexible_transfers(
        interval,
        solution.network,
        solution.fleet,
        solution.static.resolution,
    )

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def determine_power_adjustments_for_precharging_period(
    interval: nbintp,
    solution,
    year: nbintp,
) -> nbintp:
    """
    Iterate backwards through the time intervals in the precharging period, adjusting the dispatch power of
    Storage systems and flexible Generators to transfer enough energy to Storage precharges to balance the
    deficit block.
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
            solution.static.resolution,
        )
        network_m.set_storage_precharge_fills_and_surpluses(solution.network)

        perform_local_surplus_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.resolution,
        )

        perform_transmitted_surplus_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.resolution,
        )

        perform_intranode_interstorage_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.resolution,
        )

        perform_internode_interstorage_transfers(
            interval,
            solution.network,
            solution.fleet,
            solution.static.resolution,
        )

        if fleet_m.check_precharge_remaining(solution.fleet):
            perform_flexible_precharging(solution, interval)

        if (not fleet_m.check_precharge_remaining(solution.fleet)) or (
            not fleet_m.check_trickling_remaining(solution.fleet)
        ):
            first_interval_precharge = interval
            break

    return first_interval_precharge


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def perform_fill_adjustment(
    interval: nbintp,
    network: Network_InstanceType,
    fleet: Fleet_InstanceType,
) -> None:
    """
    Increases Storage charging power (reduces discharge power) if there is still fill energy required by the
    node after a transmission step.
    """
    for node in network.nodes.values():
        for storage_order in node.storage_merit_order:
            dispatch_power_update = min(node.fill, fleet.storages[storage_order].remaining_charge_max_t)
            fleet.storages[storage_order].dispatch_power[interval] += dispatch_power_update
            node.storage_power[interval] += dispatch_power_update
            node.fill -= dispatch_power_update
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def resolve_energy_discontinuities(
    first_interval_precharge: nbintp,
    interval_after_deficit_block: nbintp,
    solution,
) -> None:
    """
    Iterate forwards through the precharging period and deficit block, checking whether the dispatch
    powers of Storage systems and flexible Generators determined for this period are feasible.
    """
    for interval in range(first_interval_precharge, interval_after_deficit_block):
        initialise_interval(
            interval,
            solution.network,
            solution.fleet,
            solution.static.resolution,
            True,
        )

        infeasible_flag = fleet_m.determine_feasible_storage_dispatch(solution.fleet, interval)
        if infeasible_flag:
            network_m.reset_transmission(solution.network, interval)
            network_m.reset_flexible(solution.network, interval)
            fleet_m.reset_flexible(solution.fleet, interval)

            balance_with_transmission(interval, solution.network, "precharging_adjust_storage", False)
            balance_with_flexible(interval, solution.network, solution.fleet, solution.static.resolution, True)

            if network_m.check_remaining_netloads(solution.network, interval, "deficit"):
                balance_with_transmission(interval, solution.network, "flexible", False)
                balance_with_flexible(interval, solution.network, solution.fleet, solution.static.resolution, True)
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
                for generator in solution.fleet.generators.values():
                    if generator.is_flexible:
                        generator.fuel.remaining_energy[interval] -= (
                            generator.dispatch_power[interval] * solution.static.resolution
                        )

        network_m.calculate_curtail_and_deficit(solution.network, interval)

        fleet_m.update_stored_energies(solution.fleet, interval, solution.static.resolution, True)
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def precharge_storage(
    solution,
    interval_after_deficit_block: nbintp,
    year: nbintp,
) -> None:
    """
    Core unit committment business rules for precharging Storage systems.
    """
    first_interval_deficit_block = determine_precharge_energies_for_deficit_block(
        interval_after_deficit_block, solution, year
    )

    first_interval_precharge = determine_power_adjustments_for_precharging_period(
        first_interval_deficit_block, solution, year
    )

    resolve_energy_discontinuities(first_interval_precharge, interval_after_deficit_block + 1, solution)
    return None
