from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
from firm.SimulationDynamics import UpdateUnbalancedt, CommitTrickle, UpdateSOCt


STALL_WINDOW = 24  # daily
STALL_MIN_DELTA = 0.1  # 100 MW
STALL_PATIENCE = 14  # consecutive stalled windows to count as 'stalled'
STALL_NODES = 3  # how many stalled nodes to trigger abort
STALL_CHECKPOINT_SENTINEL = 1e18  # approx inf


@njit()
def ReversePassHydro(solution):
    nodes = solution.static.nodes
    rolling_deficits = solution.operations.rolling_deficits
    node_precharge_fill = solution.operations.fill_buffer

    rolling_deficits.fill(0.0)

    InitStallTracking(solution)
    intervals_since_stall_check = 0

    t_1 = solution.static.intervals - 1
    InitHydroMinFuture(solution, t_1)
    InitStorageMinMaxFuture(solution, t_1)

    for t in range(solution.static.intervals - 1, -1, -1):
        UpdateHydroMinFuture(solution, t)
        UpdateStorageMinMaxFuture(solution, t)

        for n in range(nodes):
            if solution.operations.Mdeficit[t, n] > 1e-6:
                rolling_deficits[n] += solution.operations.Mdeficit[t, n]

        if (rolling_deficits > 1e-6).any():
            SetupPrechargePools(solution, t)
            SetupStorageDonors(solution, t)

            if (node_precharge_fill > 1e-6).any():
                TrickleStorageHydro(solution, t)

        intervals_since_stall_check += 1
        if intervals_since_stall_check >= STALL_WINDOW:
            intervals_since_stall_check = 0
            WriteOffStalledHydroNodes(solution)


@njit()
def ReversePassGas(solution):
    nodes = solution.static.nodes
    rolling_deficits = solution.operations.rolling_deficits
    node_precharge_fill = solution.operations.fill_buffer

    rolling_deficits.fill(0.0)

    InitStallTracking(solution)
    intervals_since_stall_check = 0

    t_1 = solution.static.intervals - 1
    InitStorageMinMaxFuture(solution, t_1)

    for t in range(solution.static.intervals - 1, -1, -1):
        UpdateStorageMinMaxFuture(solution, t)

        for n in range(nodes):
            if solution.operations.Mdeficit[t, n] > 1e-6:
                rolling_deficits[n] += solution.operations.Mdeficit[t, n]

        if (rolling_deficits > 1e-6).any():
            SetupPrechargePools(solution, t)

            if (node_precharge_fill > 1e-6).any():
                TrickleGas(solution, t)

        intervals_since_stall_check += 1
        if intervals_since_stall_check >= STALL_WINDOW:
            intervals_since_stall_check = 0
            if CheckNodeStalls(solution):
                print("stalled")
                # return False  # disabled for now

    _feasible = (rolling_deficits > 1e-6).any()
    return _feasible


@njit(inline="always")
def InitHydroMinFuture(solution, t):
    for n in range(solution.static.nodes):
        solution.operations.hydro_min_future[n, 0] = solution.operations.Mreservoir[t, n, 0]
        solution.operations.hydro_min_future[n, 1] = solution.operations.Mreservoir[t, n, 1]


@njit(inline="always")
def InitStorageMinMaxFuture(solution, t):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.storage_min_future[n, s] = solution.operations.Mstorage[t, n, s]
            solution.operations.storage_max_future[n, s] = solution.operations.Mstorage[t, n, s]


@njit(inline="always")
def UpdateHydroMinFuture(solution, t):
    for n in range(solution.static.nodes):
        solution.operations.hydro_min_future[n, 0] = min(
            solution.operations.hydro_min_future[n, 0], solution.operations.Mreservoir[t, n, 0]
        )
        solution.operations.hydro_min_future[n, 1] = min(
            solution.operations.hydro_min_future[n, 1], solution.operations.Mreservoir[t, n, 1]
        )


@njit(inline="always")
def UpdateStorageMinMaxFuture(solution, t):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.storage_min_future[n, s] = min(
                solution.operations.storage_min_future[n, s], solution.operations.Mstorage[t, n, s]
            )
            solution.operations.storage_max_future[n, s] = max(
                solution.operations.storage_max_future[n, s], solution.operations.Mstorage[t, n, s]
            )


@njit(inline="always")
def SetupPrechargePools(solution, t):
    nodes = solution.static.nodes

    rolling_deficits = solution.operations.rolling_deficits
    node_precharge_fill = solution.operations.fill_buffer

    node_precharge_fill.fill(0.0)

    for n in range(nodes):
        for s in range(4):
            solution.operations.precharge_flag[n, s] = False
            solution.operations.charge_max_t[t, n, s] = 0.0

        if rolling_deficits[n] > 1e-6:
            remaining_fill = rolling_deficits[n]
            for s in (3, 2, 1, 0):  # Shortest duration first
                headroom = GetReverseStorageHeadroom(solution, t, n, s)

                if headroom > 1e-6:
                    solution.operations.precharge_flag[n, s] = True

                    # headroom in terms of demand not supply
                    rt_eff = solution.static.storage_charge_eff[s] * solution.static.storage_discha_eff[s]
                    headroom *= rt_eff

                    allocated_deficit = min(remaining_fill, headroom)

                    required_generation = allocated_deficit / rt_eff

                    solution.operations.charge_max_t[t, n, s] = required_generation
                    node_precharge_fill[n] += required_generation
                    remaining_fill -= allocated_deficit


@njit()
def SetupStorageDonors(solution, t):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.trickling_flag[n, s] = False
            solution.operations.discharge_max_t[t, n, s] = 0.0

        if solution.operations.Mcurtail[t, n] > 1e-6:
            for s in (0, 1, 2, 3):  # Longest duration first
                available_e_power = (
                    solution.operations.storage_min_future[n, s]
                    * solution.static.storage_discha_eff[s]
                    / solution.static.resolution
                )
                available_p_power = solution.assets.CstorageP[n, s] - solution.operations.Mdischarge[t, n, s]

                max_d = max(0.0, min(available_p_power, available_e_power))
                if max_d > 1e-6:
                    solution.operations.trickling_flag[n, s] = True
                    solution.operations.discharge_max_t[t, n, s] = max_d


@njit(inline="always")
def FillPrechargers(solution, t, n, transfer_amount):
    res = solution.static.resolution
    rolling_deficits = solution.operations.rolling_deficits

    for s in (3, 2, 1, 0):
        if solution.operations.precharge_flag[n, s] and transfer_amount > 1e-6:
            allocated = min(transfer_amount, solution.operations.charge_max_t[t, n, s])
            solution.operations.Mcharge[t, n, s] += allocated
            solution.operations.charge_max_t[t, n, s] -= allocated
            transfer_amount -= allocated

            energy_added = allocated * solution.static.storage_charge_eff[s] * res
            solution.operations.storage_max_future[n, s] += energy_added
            solution.operations.storage_min_future[n, s] += energy_added

            rolling_deficits[n] = max(0.0, rolling_deficits[n] - allocated)
    return transfer_amount


@njit(inline="always")
def DrainHydroDonors(solution, t, n, transfer_amount):
    res = solution.static.resolution
    hydro_headroom = solution.operations.hydro_headroom

    for h in (0, 1):
        allocated = min(transfer_amount, hydro_headroom[n, h])
        solution.operations.Mhydro[t, n, h] += allocated
        hydro_headroom[n, h] -= allocated
        transfer_amount -= allocated
        solution.operations.hydro_min_future[n, h] -= allocated * res
    return transfer_amount


@njit(inline="always")
def DrainStorageDonors(solution, t, n, transfer_amount):
    res = solution.static.resolution
    for s in (0, 1, 2, 3):
        if solution.operations.trickling_flag[n, s] and transfer_amount > 1e-6:
            allocated = min(transfer_amount, solution.operations.discharge_max_t[t, n, s])
            solution.operations.Mdischarge[t, n, s] += allocated
            solution.operations.discharge_max_t[t, n, s] -= allocated
            transfer_amount -= allocated

            energy_removed = allocated / solution.static.storage_discha_eff[s] * res
            solution.operations.storage_max_future[n, s] -= energy_removed
            solution.operations.storage_min_future[n, s] -= energy_removed
    return transfer_amount


@njit(inline="always")
def TrickleStorageHydro(solution, t):  # noqa: C901
    nodes = solution.static.nodes
    res = solution.static.resolution

    surplus = solution.operations.surplus_buffer
    surplus_orig = solution.operations.surplus_orig
    hydro_headroom = solution.operations.hydro_headroom
    node_precharge_fill = solution.operations.fill_buffer
    precharge_fill_orig = solution.operations.fill_orig

    surplus.fill(0.0)
    hydro_headroom.fill(0.0)

    for n in range(nodes):
        # Hydro Headroom
        for h in (0, 1):
            energy_headroom = solution.operations.hydro_min_future[n, h] / res
            power_headroom = solution.assets.ChydP[n, h] - solution.operations.Mhydro[t, n, h]
            available = max(0.0, min(energy_headroom, power_headroom))

            hydro_headroom[n, h] = available
            surplus[n] += available

        # Storage Headroom
        for s in range(4):
            if solution.operations.trickling_flag[n, s]:
                surplus[n] += solution.operations.discharge_max_t[t, n, s]

    # Local Transfers
    for n in range(nodes):
        transfer = min(surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            surplus[n] -= transfer
            node_precharge_fill[n] -= transfer

            # Cascade the transfer balance through both asset classes
            rem_transfer = DrainStorageDonors(solution, t, n, transfer)
            DrainHydroDonors(solution, t, n, rem_transfer)

            FillPrechargers(solution, t, n, transfer)

    # Network Transfers
    if (surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        precharge_fill_orig[:] = node_precharge_fill
        surplus_orig[:] = surplus

        Interconnection(
            solution,
            node_precharge_fill,
            surplus,
            solution.operations.Tnetflow[t],
            solution.operations.Mimport[t],
            solution.operations.Mexport[t]
        )

        for n in range(nodes):
            exports = surplus_orig[n] - surplus[n]
            if exports > 1e-6:
                rem_exports = DrainStorageDonors(solution, t, n, exports)
                DrainHydroDonors(solution, t, n, rem_exports)

            imports = precharge_fill_orig[n] - node_precharge_fill[n]
            if imports > 1e-6:
                FillPrechargers(solution, t, n, imports)


@njit(inline="always")
def TrickleGas(solution, t):
    nodes = solution.static.nodes
    surplus = solution.operations.surplus_buffer
    surplus_orig = solution.operations.surplus_orig
    node_precharge_fill = solution.operations.fill_buffer
    precharge_fill_orig = solution.operations.fill_orig

    surplus.fill(0.0)

    for n in range(nodes):
        surplus[n] = max(0.0, solution.assets.Cgas[n] - solution.operations.Mgas[t, n])

    for n in range(nodes):
        transfer = min(surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            surplus[n] -= transfer
            node_precharge_fill[n] -= transfer
            solution.operations.Mgas[t, n] += transfer
            FillPrechargers(solution, t, n, transfer)

    if (surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        surplus_orig[:] = surplus
        precharge_fill_orig[:] = node_precharge_fill

        Interconnection(
            solution,
            node_precharge_fill,
            surplus,
            solution.operations.Tnetflow[t],
            solution.operations.Mimport[t],
            solution.operations.Mexport[t]
        )

        for n in range(nodes):
            exports = surplus_orig[n] - surplus[n]
            if exports > 1e-6:
                solution.operations.Mgas[t, n] += exports

            imports = precharge_fill_orig[n] - node_precharge_fill[n]
            if imports > 1e-6:
                FillPrechargers(solution, t, n, imports)


@njit(fastmath=True, inline="always")
def GetReverseStorageHeadroom(solution, t, n, s):
    """Calculates headroom during Backward Sweeps based on tracked future bounds"""
    max_e_power = (solution.assets.CstorageE[n, s] - solution.operations.storage_max_future[n, s]
                   ) / solution.static.storage_charge_eff[s] / solution.static.resolution

    available_power = solution.assets.CstorageP[n, s] - solution.operations.Mcharge[t, n, s]
    return max(0.0, min(available_power, max_e_power))


@njit(inline="always")
def InitStallTracking(solution):
    solution.operations.stall_checkpoint.fill(STALL_CHECKPOINT_SENTINEL)
    solution.operations.stall_counter.fill(0)


@njit(inline="always")
def WriteOffStalledHydroNodes(solution):
    """
    Per-node stall detection for the hydro/storage trickling pass. A node
    whose rolling_deficits hasn't shrunk by at least STALL_MIN_DELTA over
    STALL_PATIENCE consecutive checks is written off: further hydro/storage
    precharge attempts are abandoned for the backlog it has currently
    accumulated. This does NOT touch solution.Feasible as feasibility is not yet
    determinable at this point in Simulate. The unresolved deficit falls through
    unchanged to ForwardPassGas / ReversePassGas.

    Resetting checkpoint/counter on write-off (rather than leaving the node
    permanently excluded) lets it re-enter consideration if fresh deficit
    appears at an earlier interval.
    """
    rolling_deficits = solution.operations.rolling_deficits
    checkpoint = solution.operations.stall_checkpoint
    counter = solution.operations.stall_counter

    for n in range(solution.static.nodes):
        if rolling_deficits[n] <= 1e-6:
            counter[n] = 0
            checkpoint[n] = STALL_CHECKPOINT_SENTINEL
            continue

        if rolling_deficits[n] <= checkpoint[n] - STALL_MIN_DELTA:
            counter[n] = 0
        else:
            counter[n] += 1

        checkpoint[n] = rolling_deficits[n]

        if counter[n] >= STALL_PATIENCE:
            rolling_deficits[n] = 0.0
            counter[n] = 0
            checkpoint[n] = STALL_CHECKPOINT_SENTINEL


@njit(inline="always")
def CheckNodeStalls(solution):
    """
    Compares each node's current rolling_deficits against its value at the
    last checkpoint (STALL_WINDOW intervals ago). A node that hasn't shrunk
    its backlog by at least STALL_MIN_DELTA increments its stall counter;
    any real shrinkage resets it. Returns True once at least
    STALL_NODES of currently-deficient nodes have been stalled for
    STALL_PATIENCE consecutive checks.
    """
    rolling_deficits = solution.operations.rolling_deficits
    checkpoint = solution.operations.stall_checkpoint
    counter = solution.operations.stall_counter

    deficient_nodes = 0
    stalled_nodes = 0

    for n in range(solution.static.nodes):
        if rolling_deficits[n] > 1e-6:
            deficient_nodes += 1

            if rolling_deficits[n] <= checkpoint[n] - STALL_MIN_DELTA:
                counter[n] = 0
            else:
                counter[n] += 1

            if counter[n] >= STALL_PATIENCE:
                stalled_nodes += 1
        else:
            counter[n] = 0

        checkpoint[n] = rolling_deficits[n]

    if deficient_nodes == 0:
        return False

    return stalled_nodes >= STALL_NODES


@njit(inline="always")
def UpdateDynamics(solution):
    for t in range(solution.static.intervals):
        UpdateUnbalancedt(solution, t)
        CommitTrickle(solution, t)
        UpdateSOCt(solution, t)
