import numpy as np
from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
from firm.Utils import array_sum_2d_axis1  # type: ignore


@njit
def Simulate(solution):  # noqa: C901
    working_buffer = np.zeros(solution.static.nodes, np.float64)

    # Base Forward Pass (Includes Local/Network Gas Dispatch)
    ForwardPass(solution, working_buffer)

    # Sweep A: Hydro & Storage Trickling
    hydro_min_future = np.zeros((2, solution.static.nodes), np.float64)
    SweepHydro(solution, hydro_min_future)

    # Intermediate Forward Pass (Lock in Hydro/Storage actions)
    UpdateDynamics(solution)

    # Sweep B: Flexible Gas Trickling
    # Only run if deficits STILL exist after Sweep A exhausted free/stored energy
    if (solution.operations.Mdeficit > 1e-6).any():
        SweepGas(solution)
        # 5. Final Forward Pass (Lock in Gas actions)
        UpdateDynamics(solution)


@njit()
def ForwardPass(solution, working_buffer):  # noqa: C901
    nodes = solution.static.nodes
    for t in range(solution.static.intervals):
        UpdateBalancingt(solution, t)

        # Fill deficits from network curtailment
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            if (solution.operations.Mspillage[t] > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    solution.operations.Mspillage[t],
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
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Timport[t],
                    solution.operations.Texport[t]
                )
                for n in range(nodes):
                    exports = array_sum_2d_axis1(solution.operations.Texport[t])[n]
                    solution.operations.Mgas[t, n] += exports
                UpdateSpillDeft(solution, t)

        # Push remaining curtailment to storage headroom
        if (solution.operations.Mspillage[t] > 1e-6).any():
            for n in range(nodes):
                headroom = 0.0
                for s in range(4):
                    current_net_power = solution.operations.Mcharge[s, t, n] - solution.operations.Mdischarge[s, t, n]
                    headroom += GetStorageHeadroom(solution, s, n, t, current_net_power)

                working_buffer[n] = headroom

            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    working_buffer,
                    solution.operations.Mspillage[t],
                    solution.operations.Timport[t],
                    solution.operations.Texport[t]
                )
                UpdateBalancingt(solution, t)

        UpdateSOCt(solution, t)


@njit()
def SweepHydro(solution, hydro_min_future):
    nodes = solution.static.nodes
    rolling_deficits = np.zeros(nodes, np.float64)
    running_charge = np.zeros((4, nodes), np.float64)
    running_discharge = np.zeros((4, nodes), np.float64)

    for n in range(nodes):
        hydro_min_future[0, n] = solution.operations.Mreservoir[0, solution.static.intervals - 1, n]
        hydro_min_future[1, n] = solution.operations.Mreservoir[1, solution.static.intervals - 1, n]

    for t in range(solution.static.intervals - 1, -1, -1):
        for n in range(nodes):
            hydro_min_future[0, n] = min(hydro_min_future[0, n], solution.operations.Mreservoir[0, t, n])
            hydro_min_future[1, n] = min(hydro_min_future[1, n], solution.operations.Mreservoir[1, t, n])

            if solution.operations.Mdeficit[t, n] > 1e-6:
                rolling_deficits[n] += solution.operations.Mdeficit[t, n] / solution.static.storage_charge_eff[0]

        if (rolling_deficits > 1e-6).any():
            node_precharge_fill = SetupPrechargePools(solution, t, rolling_deficits, running_charge)
            SetupStorageDonors(solution, t, running_discharge)

            if (node_precharge_fill > 1e-6).any():
                TrickleHydro(
                    solution,
                    t,
                    node_precharge_fill,
                    hydro_min_future,
                    rolling_deficits,
                    running_charge
                )
            if (node_precharge_fill > 1e-6).any():
                TrickleStorage(
                    solution,
                    t,
                    node_precharge_fill,
                    rolling_deficits,
                    running_charge,
                    running_discharge
                )


@njit()
def SweepGas(solution):
    nodes = solution.static.nodes
    rolling_deficits = np.zeros(nodes, np.float64)
    running_charge = np.zeros((4, nodes), np.float64)

    for t in range(solution.static.intervals - 1, -1, -1):
        for n in range(nodes):
            if solution.operations.Mdeficit[t, n] > 1e-6:
                rolling_deficits[n] += solution.operations.Mdeficit[t, n] / solution.static.storage_charge_eff[0]

        if (rolling_deficits > 1e-6).any():
            node_precharge_fill = SetupPrechargePools(solution, t, rolling_deficits, running_charge)

            if (node_precharge_fill > 1e-6).any():
                TrickleGas(solution, t, node_precharge_fill, rolling_deficits, running_charge)


@njit(inline="always")
def UpdateDynamics(solution):
    for t in range(solution.static.intervals):
        UpdateUnbalancedt(solution, t)
        UpdateStoraget(solution, t)
        UpdateSOCt(solution, t)
        UpdateSpillDeft(solution, t)


@njit(inline="always")
def SetupPrechargePools(solution, t, rolling_deficits, running_charge):
    nodes = solution.static.nodes
    node_precharge_fill = np.zeros(nodes, dtype=np.float64)

    for n in range(nodes):
        for s in range(4):
            solution.operations.precharge_flag[s, n] = False
            solution.operations.charge_max_t[s, t, n] = 0.0

        if rolling_deficits[n] > 1e-6:
            remaining_fill = rolling_deficits[n]
            for s in (3, 2, 1, 0):  # Shortest duration first
                headroom = GetStorageHeadroom(solution, s, n, t, running_charge[s, n])

                if headroom > 1e-6:
                    solution.operations.precharge_flag[s, n] = True
                    max_p = max(0.0, solution.assets.CstorageP[s, n] - solution.operations.Mcharge[s, t, n])
                    solution.operations.charge_max_t[s, t, n] = min(max_p, headroom)

                    allocated = min(remaining_fill, solution.operations.charge_max_t[s, t, n])
                    node_precharge_fill[n] += allocated
                    remaining_fill -= allocated
    return node_precharge_fill


@njit()
def SetupStorageDonors(solution, t, running_discharge):
    for n in range(solution.static.nodes):
        for s in range(4):
            solution.operations.trickling_flag[s, n] = False
            solution.operations.discharge_max_t[s, t, n] = 0.0

        if solution.operations.Mspillage[t, n] > 1e-6:
            for s in (0, 1, 2, 3):  # Longest duration first
                available = (solution.operations.Mstorage[s, t - 1, n] / solution.static.resolution
                             ) - running_discharge[s, n]
                if available > 1e-6:
                    solution.operations.trickling_flag[s, n] = True
                    max_p = max(0.0, solution.assets.CstorageP[s, n] - solution.operations.Mdischarge[s, t, n])
                    solution.operations.discharge_max_t[s, t, n] = min(max_p, available)


@njit(inline="always")
def FillPrechargers(solution, n, t, transfer_amount, rolling_deficits, running_charge):
    rem_transfer = transfer_amount
    for s in (3, 2, 1, 0):
        if solution.operations.precharge_flag[s, n] and rem_transfer > 1e-6:
            allocated = min(rem_transfer, solution.operations.charge_max_t[s, t, n])
            solution.operations.Mcharge[s, t, n] += allocated
            solution.operations.charge_max_t[s, t, n] -= allocated
            rem_transfer -= allocated
            running_charge[s, n] += allocated
            rolling_deficits[n] = max(0.0, rolling_deficits[n] - allocated)  # Resolve tracking in real-time


@njit(inline="always")
def DrainHydroDonors(solution, n, t, transfer_amount, hydro_headroom):
    rem_transfer = transfer_amount
    for h in (0, 1):
        allocated = min(rem_transfer, hydro_headroom[h, n])
        solution.operations.Mhydro[h, t, n] += allocated
        hydro_headroom[h, n] -= allocated
        rem_transfer -= allocated


@njit(inline="always")
def DrainStorageDonors(solution, n, t, transfer_amount, running_discharge):
    rem_transfer = transfer_amount
    for s in (0, 1, 2, 3):
        if solution.operations.trickling_flag[s, n] and rem_transfer > 1e-6:
            allocated = min(rem_transfer, solution.operations.discharge_max_t[s, t, n])
            solution.operations.Mdischarge[s, t, n] += allocated
            solution.operations.discharge_max_t[s, t, n] -= allocated
            rem_transfer -= allocated
            running_discharge[s, n] += allocated


@njit(inline="always")
def TrickleHydro(solution, t, node_precharge_fill, hydro_min_future, rolling_deficits, running_charge):
    nodes = solution.static.nodes
    hydro_surplus = np.zeros(nodes, dtype=np.float64)
    hydro_headroom = np.zeros((2, nodes), dtype=np.float64)

    for n in range(nodes):
        for h in (0, 1):
            available = min(solution.assets.CpondP[n] if h == 0 else solution.assets.ChydP[n],
                            (solution.operations.Mreservoir[h, t - 1, n] - hydro_min_future[h, n]
                             ) / solution.static.resolution)
            available = max(0.0, available - solution.operations.Mhydro[h, t, n])
            hydro_headroom[h, n] = available
            hydro_surplus[n] += available

    for n in range(nodes):
        transfer = min(hydro_surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            hydro_surplus[n] -= transfer
            node_precharge_fill[n] -= transfer
            DrainHydroDonors(solution, n, t, transfer, hydro_headroom)
            FillPrechargers(solution, n, t, transfer, rolling_deficits, running_charge)

    if (hydro_surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        solution.operations.Timport[t][:] = 0.0
        solution.operations.Texport[t][:] = 0.0
        Interconnection(
            solution,
            node_precharge_fill,
            hydro_surplus,
            solution.operations.Timport[t],
            solution.operations.Texport[t]
        )
        for n in range(nodes):
            exports = array_sum_2d_axis1(solution.operations.Texport[t])[n]
            if exports > 1e-6:
                DrainHydroDonors(solution, n, t, exports, hydro_headroom)

            imports = array_sum_2d_axis1(solution.operations.Timport[t])[n]
            if imports > 1e-6:
                node_precharge_fill[n] -= imports
                FillPrechargers(solution, n, t, imports, rolling_deficits, running_charge)


@njit(inline="always")
def TrickleStorage(solution, t, node_precharge_fill, rolling_deficits, running_charge, running_discharge):
    nodes = solution.static.nodes
    storage_surplus = np.zeros(nodes, dtype=np.float64)
    for n in range(nodes):
        for s in range(4):
            if solution.operations.trickling_flag[s, n]:
                storage_surplus[n] += solution.operations.discharge_max_t[s, t, n]

    for n in range(nodes):
        transfer = min(storage_surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            storage_surplus[n] -= transfer
            node_precharge_fill[n] -= transfer
            DrainStorageDonors(solution, n, t, transfer, running_discharge)
            FillPrechargers(solution, n, t, transfer, rolling_deficits, running_charge)

    if (storage_surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        solution.operations.Timport[t][:] = 0.0
        solution.operations.Texport[t][:] = 0.0
        Interconnection(
            solution,
            node_precharge_fill,
            storage_surplus,
            solution.operations.Timport[t],
            solution.operations.Texport[t]
        )
        for n in range(nodes):
            exports = array_sum_2d_axis1(solution.operations.Texport[t])[n]
            if exports > 1e-6:
                DrainStorageDonors(solution, n, t, exports, running_discharge)

            imports = array_sum_2d_axis1(solution.operations.Timport[t])[n]
            if imports > 1e-6:
                node_precharge_fill[n] -= imports
                FillPrechargers(solution, n, t, imports, rolling_deficits, running_charge)


@njit(inline="always")
def TrickleGas(solution, t, node_precharge_fill, rolling_deficits, running_charge):
    nodes = solution.static.nodes
    gas_surplus = np.zeros(nodes, dtype=np.float64)
    for n in range(nodes):
        gas_surplus[n] = max(0.0, solution.assets.Cgas[n] - solution.operations.Mgas[t, n])

    for n in range(nodes):
        transfer = min(gas_surplus[n], node_precharge_fill[n])
        if transfer > 1e-6:
            gas_surplus[n] -= transfer
            node_precharge_fill[n] -= transfer
            solution.operations.Mgas[t, n] += transfer
            FillPrechargers(solution, n, t, transfer, rolling_deficits, running_charge)

    if (gas_surplus > 1e-6).any() and (node_precharge_fill > 1e-6).any():
        solution.operations.Timport[t][:] = 0.0
        solution.operations.Texport[t][:] = 0.0
        Interconnection(
            solution,
            node_precharge_fill,
            gas_surplus,
            solution.operations.Timport[t],
            solution.operations.Texport[t]
        )
        for n in range(nodes):
            exports = array_sum_2d_axis1(solution.operations.Texport[t])[n]
            if exports > 1e-6:
                solution.operations.Mgas[t, n] += exports

            imports = array_sum_2d_axis1(solution.operations.Timport[t])[n]
            if imports > 1e-6:
                node_precharge_fill[n] -= imports
                FillPrechargers(solution, n, t, imports, rolling_deficits, running_charge)


@njit(fastmath=True, inline="always")
def GetStorageHeadroom(
    solution,
    s: int,
    n: int,
    t: int,
    current_net_power: np.float64
) -> np.float64:
    """
    Calculates the maximum additional power a storage system can absorb
    in the current timestep without violating power or energy limits.
    """
    # parameters
    power_cap = solution.assets.CstorageP[s, n]
    energy_cap = solution.assets.CstorageE[s, n]
    soc_prev = solution.operations.Mstorage[s, t - 1, n]
    charge_eff = solution.static.storage_charge_eff[s]
    res = solution.static.resolution

    # energy limit in power terms
    max_e_power = (energy_cap - soc_prev) / charge_eff / res
    # power limit
    max_power = min(power_cap, max_e_power)

    return max(0.0, max_power - current_net_power)


@njit(inline="always")
def GetSurplust(solution, t, Msurplust):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        surplus = solution.operations.Mspillage[t, n]
        for s in range(4):
            surplus += solution.operations.Mcharge[s, t, n]
            surplus += min(
                solution.assets.CstorageP[s, n],
                (solution.operations.Mstorage[s, t - 1, n] * solution.static.storage_discha_eff[s]) / res
                )
            surplus -= solution.operations.Mdischarge[s, t, n]

        surplus += min(solution.assets.CpondP[n], solution.operations.Mreservoir[0, t - 1, n] / res
                       ) - solution.operations.Mhydro[0, t, n]
        surplus += min(solution.assets.ChydP[n], solution.operations.Mreservoir[1, t - 1, n] / res
                       ) - solution.operations.Mhydro[1, t, n]
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

        inflow_e = solution.static.TSpond_inflow[t, n] * res
        discharge_cap = (solution.operations.Mreservoir[0, t - 1, n] + inflow_e) / res
        solution.operations.Mhydro[0, t, n] = min(max(0, unbal), solution.assets.CpondP[n], discharge_cap)
        unbal -= solution.operations.Mhydro[0, t, n]

        for s in range(4):
            charge_cap = (solution.assets.CstorageE[s, n] - solution.operations.Mstorage[s, t - 1, n]
                          ) / solution.static.storage_charge_eff[s] / res
            solution.operations.Mcharge[s, t, n] = min(-min(0, unbal), solution.assets.CstorageP[s, n], charge_cap)
            discharge_cap = solution.operations.Mstorage[s, t - 1, n] * solution.static.storage_discha_eff[s] / res
            solution.operations.Mdischarge[s, t, n] = min(max(0, unbal), solution.assets.CstorageP[s, n], discharge_cap)
            unbal += solution.operations.Mcharge[s, t, n] - solution.operations.Mdischarge[s, t, n]

        inflow_e = solution.static.TShyd_inflow[t, n] * res
        discharge_cap = (solution.operations.Mreservoir[1, t - 1, n] + inflow_e) / res
        solution.operations.Mhydro[1, t, n] = min(max(0, unbal), solution.assets.ChydP[n], discharge_cap)


@njit(fastmath=True, inline="always")
def UpdateSOCt(solution, t):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        inflow_e_pond = solution.static.TSpond_inflow[t, n] * solution.assets.CpondP[n] * res
        solution.operations.Mreservoir[0, t, n] = min(
            solution.assets.CpondE[n],
            solution.operations.Mreservoir[0, t - 1, n] + inflow_e_pond - solution.operations.Mhydro[0, t, n] * res
        )

        inflow_e_hyd = solution.static.TShyd_inflow[t, n] * solution.assets.ChydP[n] * res
        solution.operations.Mreservoir[1, t, n] = min(
            solution.assets.ChydE[n],
            solution.operations.Mreservoir[1, t - 1, n] + inflow_e_hyd - solution.operations.Mhydro[1, t, n] * res
        )

        for s in range(4):
            solution.operations.Mstorage[s, t, n] = (
                solution.operations.Mstorage[s, t - 1, n] + res * (
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
        solution.operations.Mspillage[t, n] = -min(0.0, _inter)


@njit(inline="always")
def UpdateBalancingt(solution, t):
    UpdateUnbalancedt(solution, t)
    UpdateStoraget(solution, t)
    UpdateSpillDeft(solution, t)
