# import numpy as np
from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
# from firm.Utils import array_sum_2d_axis1  # type: ignore


@njit
def Simulate(solution):
    # Base Forward Pass (Includes Local/Network Gas Dispatch)
    ForwardPass(solution)

    # Sweep A: Hydro & Storage Trickling
    SweepHydro(solution)

    # Intermediate Forward Pass (Lock in Hydro/Storage actions)
    UpdateDynamics(solution)

    # Sweep B: Flexible Gas Trickling
    # Only run if deficits STILL exist after Sweep A exhausted free/stored energy
    if (solution.operations.Mdeficit > 1e-6).any():
        SweepGas(solution)
        # Final Forward Pass (Lock in Gas actions)
        UpdateDynamics(solution)


@njit()
def ForwardPass(solution):  # noqa: C901
    working_buffer = solution.operations.surplus_buffer
    working_buffer_orig = solution.operations.surplus_orig
    working_buffer.fill(0.0)

    nodes = solution.static.nodes
    for t in range(solution.static.intervals):
        UpdateBalancingt(solution, t)

        # Fill deficits from network curtailment
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            if (solution.operations.Mcurtail[t] > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    solution.operations.Mcurtail[t],
                    solution.operations.Tnetflow[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t]
                )
                UpdateBalancingt(solution, t)

        # Fill deficits from network storage reserves
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            GetSurplust(solution, t, working_buffer)
            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t]
                )
                UpdateBalancingt(solution, t)

        # Local Flexible Gas
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            for n in range(nodes):
                if solution.operations.Mdeficit[t, n] > 1e-6:
                    avail = max(0.0, solution.assets.Cgas[n] - solution.operations.Mgas[t, n])
                    solution.operations.Mgas[t, n] += min(solution.operations.Mdeficit[t, n], avail)
            UpdateSpillDeft(solution, t)

        # Network Flexible Gas
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            for n in range(nodes):
                working_buffer[n] = max(0.0, solution.assets.Cgas[n] - solution.operations.Mgas[t, n])

            if (working_buffer > 1e-6).any():
                working_buffer_orig[:] = working_buffer

                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t]
                )

                for n in range(nodes):
                    dispatched = working_buffer_orig[n] - working_buffer[n]
                    solution.operations.Mgas[t, n] += dispatched
                UpdateSpillDeft(solution, t)

        # Push remaining curtailment to storage headroom
        if (solution.operations.Mcurtail[t] > 1e-6).any():
            for n in range(nodes):
                headroom = 0.0
                for s in range(4):
                    headroom += GetForwardStorageHeadroom(solution, s, n, t)

                working_buffer[n] = headroom

            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    working_buffer,
                    solution.operations.Mcurtail[t],
                    solution.operations.Tnetflow[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t]
                )
                UpdateBalancingt(solution, t)

        UpdateSOCt(solution, t)


@njit()
def SweepHydro(solution):
    nodes = solution.static.nodes
    rolling_deficits = solution.operations.rolling_deficits
    node_precharge_fill = solution.operations.fill_buffer

    rolling_deficits.fill(0.0)

    t_1 = solution.static.intervals - 1
    InitHydroMinFuture(solution, t_1)
    InitStorageMinMaxFuture(solution, t_1)

    active_deficits = False

    for t in range(solution.static.intervals - 1, -1, -1):
        if not active_deficits:
            if (solution.operations.Mdeficit[t] > 1e-6).any():
                active_deficits = True
                InitHydroMinFuture(solution, t)
                InitStorageMinMaxFuture(solution, t)
            else:
                continue  # early exit while deficits don't exist

        UpdateHydroMinFuture(solution, t)
        UpdateStorageMinMaxFuture(solution, t)

        for n in range(nodes):
            if solution.operations.Mdeficit[t, n] > 1e-6:
                rolling_deficits[n] += solution.operations.Mdeficit[t, n]

        if (rolling_deficits > 1e-6).any():
            SetupPrechargePools(solution, t)
            SetupStorageDonors(solution, t)

            if (node_precharge_fill > 1e-6).any():
                TrickleHydro(solution, t)
            if (node_precharge_fill > 1e-6).any():
                TrickleStorage(solution, t)

            if not (rolling_deficits > 1e-6).any():
                active_deficits = False


@njit()
def SweepGas(solution):
    nodes = solution.static.nodes
    rolling_deficits = solution.operations.rolling_deficits
    node_precharge_fill = solution.operations.fill_buffer

    rolling_deficits.fill(0.0)

    t_1 = solution.static.intervals - 1
    InitStorageMinMaxFuture(solution, t_1)

    active_deficits = False

    for t in range(solution.static.intervals - 1, -1, -1):
        if not active_deficits:
            if (solution.operations.Mdeficit[t] > 1e-6).any():
                active_deficits = True
                InitStorageMinMaxFuture(solution, t)
            else:
                continue

        UpdateStorageMinMaxFuture(solution, t)

        for n in range(nodes):
            if solution.operations.Mdeficit[t, n] > 1e-6:
                rolling_deficits[n] += solution.operations.Mdeficit[t, n]

        if (rolling_deficits > 1e-6).any():
            SetupPrechargePools(solution, t, rolling_deficits)

            if (node_precharge_fill > 1e-6).any():
                TrickleGas(solution, t)

            if not (rolling_deficits > 1e-6).any():
                active_deficits = False


@njit(inline="always")
def InitHydroMinFuture(solution, t):
    for n in range(solution.static.nodes):
        solution.operations.hydro_min_future[0, n] = solution.operations.Mreservoir[0, t, n]
        solution.operations.hydro_min_future[1, n] = solution.operations.Mreservoir[1, t, n]


@njit(inline="always")
def InitStorageMinMaxFuture(solution, t):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.storage_min_future[s, n] = solution.operations.Mstorage[s, t, n]
            solution.operations.storage_max_future[s, n] = solution.operations.Mstorage[s, t, n]


@njit(inline="always")
def UpdateHydroMinFuture(solution, t):
    for n in range(solution.static.nodes):
        solution.operations.hydro_min_future[0, n] = min(
            solution.operations.hydro_min_future[0, n], solution.operations.Mreservoir[0, t, n]
        )
        solution.operations.hydro_min_future[1, n] = min(
            solution.operations.hydro_min_future[1, n], solution.operations.Mreservoir[1, t, n]
        )


@njit(inline="always")
def UpdateStorageMinMaxFuture(solution, t):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.storage_min_future[s, n] = min(
                solution.operations.storage_min_future[s, n], solution.operations.Mstorage[s, t, n]
            )
            solution.operations.storage_max_future[s, n] = max(
                solution.operations.storage_max_future[s, n], solution.operations.Mstorage[s, t, n]
            )


@njit(inline="always")
def UpdateDynamics(solution):
    for t in range(solution.static.intervals):
        UpdateUnbalancedt(solution, t)
        # UpdateStoraget(solution, t)
        UpdateSOCt(solution, t)
        UpdateSpillDeft(solution, t)


@njit(inline="always")
def SetupPrechargePools(solution, t):
    nodes = solution.static.nodes

    rolling_deficits = solution.operations.rolling_deficits

    node_precharge_fill = solution.operations.fill_buffer
    node_precharge_fill.fill(0.0)

    for n in range(nodes):
        for s in range(4):
            solution.operations.precharge_flag[s, n] = False
            solution.operations.charge_max_t[s, t, n] = 0.0

        if rolling_deficits[n] > 1e-6:
            remaining_fill = rolling_deficits[n]
            for s in (3, 2, 1, 0):  # Shortest duration first
                headroom = GetSweepStorageHeadroom(solution, s, n, t)

                if headroom > 1e-6:
                    solution.operations.precharge_flag[s, n] = True

                    # headroom in terms of demand not supply
                    rt_eff = solution.static.storage_charge_eff[s] * solution.static.storage_discha_eff[s]
                    headroom *= rt_eff

                    allocated_deficit = min(remaining_fill, headroom)

                    required_generation = allocated_deficit / rt_eff

                    solution.operations.charge_max_t[s, t, n] = required_generation
                    node_precharge_fill[n] += required_generation
                    remaining_fill -= allocated_deficit


@njit()
def SetupStorageDonors(solution, t):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.trickling_flag[s, n] = False
            solution.operations.discharge_max_t[s, t, n] = 0.0

        if solution.operations.Mcurtail[t, n] > 1e-6:
            for s in (0, 1, 2, 3):  # Longest duration first
                available_e_power = (
                    solution.operations.storage_min_future[s, n]
                    * solution.static.storage_discha_eff[s]
                    / solution.static.resolution
                )
                available_p_power = solution.assets.CstorageP[s, n] - solution.operations.Mdischarge[s, t, n]

                max_d = max(0.0, min(available_p_power, available_e_power))
                if max_d > 1e-6:
                    solution.operations.trickling_flag[s, n] = True
                    solution.operations.discharge_max_t[s, t, n] = max_d


@njit(inline="always")
def FillPrechargers(solution, n, t, transfer_amount):
    res = solution.static.resolution
    rolling_deficits = solution.operations.rolling_deficits

    for s in (3, 2, 1, 0):
        if solution.operations.precharge_flag[s, n] and transfer_amount > 1e-6:
            allocated = min(transfer_amount, solution.operations.charge_max_t[s, t, n])
            solution.operations.Mcharge[s, t, n] += allocated
            solution.operations.charge_max_t[s, t, n] -= allocated
            transfer_amount -= allocated

            energy_added = allocated * solution.static.storage_charge_eff[s] * res
            solution.operations.storage_max_future[s, n] += energy_added
            solution.operations.storage_min_future[s, n] += energy_added

            rolling_deficits[n] = max(0.0, rolling_deficits[n] - allocated)


@njit(inline="always")
def DrainHydroDonors(solution, n, t, transfer_amount):
    res = solution.static.resolution
    hydro_headroom = solution.operations.hydro_headroom

    for h in (0, 1):
        allocated = min(transfer_amount, hydro_headroom[h, n])
        solution.operations.Mhydro[h, t, n] += allocated
        hydro_headroom[h, n] -= allocated
        transfer_amount -= allocated
        solution.operations.hydro_min_future -= allocated * res


@njit(inline="always")
def DrainStorageDonors(solution, n, t, transfer_amount):
    res = solution.static.resolution
    for s in (0, 1, 2, 3):
        if solution.operations.trickling_flag[s, n] and transfer_amount > 1e-6:
            allocated = min(transfer_amount, solution.operations.discharge_max_t[s, t, n])
            solution.operations.Mdischarge[s, t, n] += allocated
            solution.operations.discharge_max_t[s, t, n] -= allocated
            transfer_amount -= allocated

            energy_removed = allocated / solution.static.storage_discha_eff[s] * res
            solution.operations.storage_max_future[s, n] -= energy_removed
            solution.operations.storage_min_future[s, n] -= energy_removed


@njit(inline="always")
def TrickleHydro(solution, t):
    nodes = solution.static.nodes

    surplus = solution.operations.surplus_buffer
    surplus_orig = solution.operations.surplus_orig
    hydro_headroom = solution.operations.hydro_headroom
    node_precharge_fill = solution.operations.fill_buffer
    precharge_fill_orig = solution.operations.fill_orig

    surplus.fill(0.0)
    hydro_headroom.fill(0.0)

    for n in range(nodes):
        for h in (0, 1):
            prev_res = (
                solution.operations.Mreservoir_init[h, n] if t == 0
                else solution.operations.Mreservoir[h, t - 1, n]
            )
            available = min(solution.assets.CpondP[n] if h == 0 else solution.assets.ChydP[n],
                            (prev_res - solution.operations.hydro_min_future[h, n]) / solution.static.resolution)
            available = max(0.0, available - solution.operations.Mhydro[h, t, n])
            hydro_headroom[h, n] = available
            surplus[n] += available

    for n in range(nodes):
        transfer = min(surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            surplus[n] -= transfer
            node_precharge_fill[n] -= transfer
            DrainHydroDonors(solution, n, t, transfer)
            FillPrechargers(solution, n, t, transfer)

    if (surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        # Track initial state to calculate precise deltas
        precharge_fill_orig[:] = node_precharge_fill
        surplus_orig[:] = surplus

        Interconnection(
            solution,
            node_precharge_fill,
            surplus,
            solution.operations.Tnetflow[t],
            solution.operations.Timport[t],
            solution.operations.Texport[t]
        )

        for n in range(nodes):
            exports = surplus_orig[n] - surplus[n]
            if exports > 1e-6:
                DrainHydroDonors(solution, n, t, exports)

            imports = precharge_fill_orig[n] - node_precharge_fill[n]
            if imports > 1e-6:
                FillPrechargers(solution, n, t, imports)


@njit(inline="always")
def TrickleStorage(solution, t):
    nodes = solution.static.nodes
    surplus = solution.operations.surplus_buffer
    surplus_orig = solution.operations.surplus_orig
    node_precharge_fill = solution.operations.fill_buffer
    precharge_fill_orig = solution.operations.fill_orig

    surplus.fill(0.0)

    for n in range(nodes):
        for s in range(4):
            if solution.operations.trickling_flag[s, n]:
                surplus[n] += solution.operations.discharge_max_t[s, t, n]

    for n in range(nodes):
        transfer = min(surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            surplus[n] -= transfer
            node_precharge_fill[n] -= transfer
            DrainStorageDonors(solution, n, t, transfer)
            FillPrechargers(solution, n, t, transfer)

    if (surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        # Track initial state to calculate precise deltas
        precharge_fill_orig[:] = node_precharge_fill
        surplus_orig[:] = surplus

        Interconnection(
            solution,
            node_precharge_fill,
            surplus,
            solution.operations.Tnetflow[t],
            solution.operations.Timport[t],
            solution.operations.Texport[t]
        )

        for n in range(nodes):
            exports = surplus_orig[n] - surplus[n]
            if exports > 1e-6:
                DrainStorageDonors(solution, n, t, exports)

            imports = precharge_fill_orig[n] - node_precharge_fill[n]
            if imports > 1e-6:
                FillPrechargers(solution, n, t, imports)


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
            FillPrechargers(solution, n, t, transfer)

    if (surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        surplus_orig[:] = surplus
        precharge_fill_orig[:] = node_precharge_fill

        Interconnection(
            solution,
            node_precharge_fill,
            surplus,
            solution.operations.Tnetflow[t],
            solution.operations.Timport[t],
            solution.operations.Texport[t]
        )

        for n in range(nodes):
            exports = surplus_orig[n] - surplus[n]
            if exports > 1e-6:
                solution.operations.Mgas[t, n] += exports

            imports = precharge_fill_orig[n] - node_precharge_fill[n]
            if imports > 1e-6:
                FillPrechargers(solution, n, t, imports)


@njit(fastmath=True, inline="always")
def GetForwardStorageHeadroom(solution, s, n, t):
    """Calculates headroom during the Forward Pass based on current SOC"""
    res = solution.static.resolution
    prev_soc = solution.operations.Mstorage_init[s, n] if t == 0 else solution.operations.Mstorage[s, t - 1, n]

    current_energy_change = res * (
        solution.operations.Mcharge[s, t, n] * solution.static.storage_charge_eff[s]
        - solution.operations.Mdischarge[s, t, n] / solution.static.storage_discha_eff[s]
    )

    max_e_power = (solution.assets.CstorageE[s, n] - (prev_soc + current_energy_change)
                   ) / solution.static.storage_charge_eff[s] / res
    available_power = solution.assets.CstorageP[s, n] - solution.operations.Mcharge[s, t, n]

    return max(0.0, min(available_power, max_e_power))


@njit(fastmath=True, inline="always")
def GetSweepStorageHeadroom(solution, s, n, t):
    """Calculates headroom during Backward Sweeps based on tracked future bounds"""
    max_e_power = (solution.assets.CstorageE[s, n] - solution.operations.storage_max_future[s, n]
                   ) / solution.static.storage_charge_eff[s] / solution.static.resolution

    available_power = solution.assets.CstorageP[s, n] - solution.operations.Mcharge[s, t, n]
    return max(0.0, min(available_power, max_e_power))


@njit(inline="always")
def GetSurplust(solution, t, Msurplust):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        surplus = solution.operations.Mcurtail[t, n]
        for s in range(4):
            prev_soc = solution.operations.Mstorage_init[s, n] if t == 0 else solution.operations.Mstorage[s, t - 1, n]
            surplus += solution.operations.Mcharge[s, t, n]
            surplus += min(solution.assets.CstorageP[s, n], (prev_soc * solution.static.storage_discha_eff[s]) / res)
            surplus -= solution.operations.Mdischarge[s, t, n]

        prev_pond = solution.operations.Mreservoir_init[0, n] if t == 0 else solution.operations.Mreservoir[0, t - 1, n]
        surplus += min(solution.assets.CpondP[n], prev_pond / res) - solution.operations.Mhydro[0, t, n]

        prev_hyd = solution.operations.Mreservoir_init[1, n] if t == 0 else solution.operations.Mreservoir[1, t - 1, n]
        surplus += min(solution.assets.ChydP[n], prev_hyd / res) - solution.operations.Mhydro[1, t, n]
        Msurplust[n] = max(0.0, surplus)


@njit(inline="always")
def UpdateUnbalancedt(solution, t):
    for n in range(solution.static.nodes):
        _Timport = 0.0
        for m in range(solution.static.nhvi):
            _Timport += solution.operations.Timport[t, n, m]
            _Timport += solution.operations.Texport[t, n, m]
        solution.operations.Munbalanced[t, n] = solution.operations.Mnetload[t, n] - _Timport


@njit(inline="always")
def UpdateStoraget(solution, t):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        unbal = solution.operations.Munbalanced[t, n]

        prev_pond = solution.operations.Mreservoir_init[0, n] if t == 0 else solution.operations.Mreservoir[0, t - 1, n]
        inflow_e_pond = solution.static.TSpond_inflow[t, n] * solution.assets.CpondP[n] * res
        discharge_cap = (prev_pond + inflow_e_pond) / res
        solution.operations.Mhydro[0, t, n] = min(max(0, unbal), solution.assets.CpondP[n], discharge_cap)
        unbal -= solution.operations.Mhydro[0, t, n]

        for s in range(4):
            prev_soc = solution.operations.Mstorage_init[s, n] if t == 0 else solution.operations.Mstorage[s, t - 1, n]
            charge_cap = (solution.assets.CstorageE[s, n] - prev_soc) / solution.static.storage_charge_eff[s] / res
            solution.operations.Mcharge[s, t, n] = min(-min(0, unbal), solution.assets.CstorageP[s, n], charge_cap)
            discharge_cap = prev_soc * solution.static.storage_discha_eff[s] / res
            solution.operations.Mdischarge[s, t, n] = min(max(0, unbal), solution.assets.CstorageP[s, n], discharge_cap)
            unbal += solution.operations.Mcharge[s, t, n] - solution.operations.Mdischarge[s, t, n]

        prev_hyd = solution.operations.Mreservoir_init[1, n] if t == 0 else solution.operations.Mreservoir[1, t - 1, n]
        inflow_e_hyd = solution.static.TShyd_inflow[t, n] * solution.assets.ChydP[n] * res
        discharge_cap = (prev_hyd + inflow_e_hyd) / res
        solution.operations.Mhydro[1, t, n] = min(max(0, unbal), solution.assets.ChydP[n], discharge_cap)


@njit(fastmath=True, inline="always")
def UpdateSOCt(solution, t):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        prev_soc = solution.operations.Mreservoir_init[0, n] if t == 0 else solution.operations.Mreservoir[0, t - 1, n]

        inflow_e_pond = solution.static.TSpond_inflow[t, n] * solution.assets.CpondP[n] * res
        solution.operations.Mreservoir[0, t, n] = min(
            solution.assets.CpondE[n],
            prev_soc + inflow_e_pond - solution.operations.Mhydro[0, t, n] * res
        )

        prev_soc = solution.operations.Mreservoir_init[1, n] if t == 0 else solution.operations.Mreservoir[1, t - 1, n]

        inflow_e_hyd = solution.static.TShyd_inflow[t, n] * solution.assets.ChydP[n] * res
        solution.operations.Mreservoir[1, t, n] = min(
            solution.assets.ChydE[n],
            prev_soc + inflow_e_hyd - solution.operations.Mhydro[1, t, n] * res
        )

        for s in range(4):
            prev_soc = solution.operations.Mstorage_init[s, n] if t == 0 else solution.operations.Mstorage[s, t - 1, n]

            solution.operations.Mstorage[s, t, n] = (
                prev_soc + res * (
                    solution.operations.Mcharge[s, t, n] * solution.static.storage_charge_eff[s]
                    - solution.operations.Mdischarge[s, t, n] / solution.static.storage_discha_eff[s]
                )
            )


@njit(inline="always")
def UpdateSpillDeft(solution, t):
    for n in range(solution.static.nodes):
        total_charge = 0.0
        total_discharge = 0.0
        for s in range(4):
            total_charge += solution.operations.Mcharge[s, t, n]
            total_discharge += solution.operations.Mdischarge[s, t, n]

        total_discharge += solution.operations.Mhydro[0, t, n] + solution.operations.Mhydro[1, t, n]
        total_discharge += solution.operations.Mgas[t, n]  # <-- IMPORTANT: Mgas included

        _inter = solution.operations.Munbalanced[t, n] + total_charge - total_discharge
        solution.operations.Mdeficit[t, n] = max(0.0, _inter)
        solution.operations.Mcurtail[t, n] = -min(0.0, _inter)


@njit(inline="always")
def UpdateBalancingt(solution, t):
    UpdateUnbalancedt(solution, t)
    UpdateStoraget(solution, t)
    UpdateSpillDeft(solution, t)
