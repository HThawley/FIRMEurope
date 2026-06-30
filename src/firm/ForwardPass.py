from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
from firm.SimulationDynamics import (
    UpdateUnbalancedt,
    UpdateLocalCharge,
    UpdateLocalDischarge,
    GetLongDurSurplust,
    GetNaiveCurtailDeficit,
    GetShortDurSurplust,
    UpdateSOCt,
)


@njit()
def ForwardPassRenewables(solution):  # noqa: C901
    working_buffer = solution.operations.surplus_buffer
    working_buffer.fill(0.0)

    nodes = solution.static.nodes
    for t in range(solution.static.intervals):
        UpdateUnbalancedt(solution, t)
        GetNaiveCurtailDeficit(solution, t)

        # Fill deficits from network curtailment
        if solution.operations.has_deficit_t and solution.operations.has_curtail_t:
            Interconnection(
                solution,
                solution.operations.Mdeficit[t],
                solution.operations.Mcurtail[t],
                solution.operations.Tnetflow[t],
                solution.operations.Mimport[t],
                solution.operations.Mexport[t]
            )
            UpdateUnbalancedt(solution, t)
        UpdateLocalCharge(solution, t)
        UpdateLocalDischarge(solution, t)

        # Fill deficits from network (long duration) storage reserves
        if solution.operations.has_deficit_t:
            GetLongDurSurplust(solution, t, working_buffer)
            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                    solution.operations.Mexport[t]
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalDischarge(solution, t)

        # Fill deficits from network (short duration) storage reserves
        if solution.operations.has_deficit_t:
            GetShortDurSurplust(solution, t, working_buffer)
            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                    solution.operations.Mexport[t]
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalDischarge(solution, t)

        # Push remaining curtailment to storage headroom
        if solution.operations.has_curtail_t:
            for n in range(nodes):
                working_buffer[n] = GetForwardStorageHeadroom(solution, t, n)

            if (working_buffer > 1e-6).any():
                Interconnection(
                    solution,
                    working_buffer,
                    solution.operations.Mcurtail[t],
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                    solution.operations.Mexport[t]
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalCharge(solution, t)

        UpdateSOCt(solution, t)


@njit()
def ForwardPassGas(solution):  # noqa: C901
    nodes = solution.static.nodes
    working_buffer = solution.operations.surplus_buffer
    working_buffer_orig = solution.operations.surplus_orig

    for t in range(solution.static.intervals):
        # Local Flexible Gas
        if solution.operations.has_deficit_t:
            for n in range(nodes):
                if solution.operations.Mdeficit[t, n] > 1e-6:
                    avail = max(0.0, solution.assets.Cgas[n] - solution.operations.Mgas[t, n])
                    dispatched = min(solution.operations.Mdeficit[t, n], avail)

                    if dispatched > 1e-6:
                        solution.operations.Mgas[t, n] += dispatched
                        solution.operations.Mdeficit[t, n] -= dispatched

        # Network Flexible Gas
        if (solution.operations.Mdeficit[t] > 1e-6).any():
            working_buffer.fill(0.0)
            for n in range(nodes):
                working_buffer[n] = max(0.0, solution.assets.Cgas[n] - solution.operations.Mgas[t, n])

            if (working_buffer > 1e-6).any():
                working_buffer_orig[:] = working_buffer

                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],  # mutated in place
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                    solution.operations.Mexport[t]
                )

                for n in range(nodes):
                    dispatched = working_buffer_orig[n] - working_buffer[n]
                    if dispatched > 1e-6:
                        solution.operations.Mgas[t, n] += dispatched
        UpdateUnbalancedt(solution, t)
    UpdateSOCt(solution, t)


@njit(fastmath=True, inline="always")
def GetForwardStorageHeadroom(solution, t, n):
    """Calculates headroom during the Forward Pass based on current SOC"""
    res = solution.static.resolution

    headroom = 0.0
    for s in range(4):
        prev_soc = solution.operations.Mstorage_init[n, s] if t == 0 else solution.operations.Mstorage[t - 1, n, s]
        if s == 0:
            prev_soc += solution.static.TSphes_inflow[t, n]
            prev_soc = min(solution.assets.CstorageE[n, s], prev_soc)

        current_energy_change = res * (
            solution.operations.Mcharge[t, n, s] * solution.static.storage_charge_eff[s]
            - solution.operations.Mdischarge[t, n, s] / solution.static.storage_discha_eff[s]
        )

        max_e_power = (solution.assets.CstorageE[n, s] - (prev_soc + current_energy_change)
                       ) / solution.static.storage_charge_eff[s] / res
        available_power = solution.assets.CstorageP[n, s] - solution.operations.Mcharge[t, n, s]

        headroom += max(0.0, min(available_power, max_e_power))
    return headroom
