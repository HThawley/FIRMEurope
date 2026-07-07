# type: ignore
from firm_ce.common.jit_overload import njit

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, TOLERANCE
from firm_ce.backend.tensor.interconnection import Interconnection
from firm_ce.backend.tensor.dynamics import (
    DispatchPeak,
    GetPeakHeadroom,
    UpdateUnbalancedt,
    UpdateLocalCharge,
    UpdateLocalDischarge,
    GetLongDurSurplust,
    GetNaiveCurtailDeficit,
    GetShortDurSurplust,
    UpdateSOCt,
)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
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
            )
            UpdateUnbalancedt(solution, t)
        UpdateLocalCharge(solution, t)
        UpdateLocalDischarge(solution, t)

        # Fill deficits from network (long duration) storage reserves
        if solution.operations.has_deficit_t:
            GetLongDurSurplust(solution, t, working_buffer)
            if (working_buffer > TOLERANCE).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalDischarge(solution, t)

        # Fill deficits from network (short duration) storage reserves
        if solution.operations.has_deficit_t:
            GetShortDurSurplust(solution, t, working_buffer)
            if (working_buffer > TOLERANCE).any():
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalDischarge(solution, t)

        # Push remaining curtailment to storage headroom
        if solution.operations.has_curtail_t:
            for n in range(nodes):
                working_buffer[n] = GetForwardStorageHeadroom(solution, t, n)

            if (working_buffer > TOLERANCE).any():
                Interconnection(
                    solution,
                    working_buffer,
                    solution.operations.Mcurtail[t],
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalCharge(solution, t)
                # UpdateLocalDischarge(solution, t)

        UpdateSOCt(solution, t)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def ForwardPassPeak(solution):  # noqa: C901
    for t in range(solution.static.intervals):
        for k in range(solution.static.npeak):
            if solution.operations.has_deficit_t:
                LocalDispatchPeakTier(solution, t, k)
            NetworkDispatchPeakTier(solution, t, k)
        UpdateUnbalancedt(solution, t)
    UpdateSOCt(solution, t)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetForwardStorageHeadroom(solution, t, n):
    """Calculates headroom during the Forward Pass based on current SOC"""
    res = solution.static.resolution

    headroom = 0.0
    for s in range(solution.static.nstor):
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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def LocalDispatchPeakTier(solution, t, k):
    nodes = solution.static.nodes
    for n in range(nodes):
        if solution.operations.Mdeficit[t, n] > TOLERANCE:
            avail = GetPeakHeadroom(solution, t, n, k)
            dispatched = min(solution.operations.Mdeficit[t, n], avail)
            if dispatched > TOLERANCE:
                DispatchPeak(solution, t, n, k, dispatched)
                solution.operations.Mdeficit[t, n] -= dispatched


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def NetworkDispatchPeakTier(solution, t, k):
    nodes = solution.static.nodes
    working_buffer = solution.operations.surplus_buffer
    working_buffer_orig = solution.operations.surplus_orig

    if not (solution.operations.Mdeficit[t] > TOLERANCE).any():
        return

    working_buffer.fill(0.0)
    for n in range(nodes):
        working_buffer[n] = GetPeakHeadroom(solution, t, n, k)

    if (working_buffer > TOLERANCE).any():
        working_buffer_orig[:] = working_buffer
        Interconnection(
            solution,
            solution.operations.Mdeficit[t],  # mutated in place
            working_buffer,
            solution.operations.Tnetflow[t],
            solution.operations.Mimport[t],
        )
        for n in range(nodes):
            dispatched = working_buffer_orig[n] - working_buffer[n]
            if dispatched > TOLERANCE:
                DispatchPeak(solution, t, n, k, dispatched)
