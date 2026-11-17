# type: ignore
from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, TOLERANCE
from firm_ce.common.jit_overload import njit
from firm_ce.tensor.dynamics import (
    DispatchPeak,
    GetLongDurSurplust,
    GetNaiveCurtailDeficit,
    GetPeakHeadroom,
    GetShortDurSurplust,
    UpdateLocalCharge,
    UpdateLocalChargeDischarge,
    UpdateLocalDischarge,
    UpdateSOCt,
    UpdateUnbalancedt,
    any_gt_1d,
)
from firm_ce.tensor.interconnection import Interconnection


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
        UpdateLocalChargeDischarge(solution, t)

        # Fill deficits from network (short duration: pondage, B2, B4) reserves first
        if solution.operations.has_deficit_t:
            if GetShortDurSurplust(solution, t, working_buffer):
                Interconnection(
                    solution,
                    solution.operations.Mdeficit[t],
                    working_buffer,
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalDischarge(solution, t)

        # Fill deficits from network (long duration: clphes/nphes, olphes, reservoir) reserves
        if solution.operations.has_deficit_t:
            if GetLongDurSurplust(solution, t, working_buffer):
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
            has_headroom = False
            for n in range(nodes):
                hr = GetForwardStorageHeadroom(solution, t, n)
                working_buffer[n] = hr
                if hr > TOLERANCE:
                    has_headroom = True

            if has_headroom:
                Interconnection(
                    solution,
                    working_buffer,
                    solution.operations.Mcurtail[t],
                    solution.operations.Tnetflow[t],
                    solution.operations.Mimport[t],
                )
                UpdateUnbalancedt(solution, t)
                UpdateLocalCharge(solution, t)

        UpdateSOCt(solution, t)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def ForwardPassPeak(solution):
    for t in range(solution.static.intervals):
        if not any_gt_1d(solution.operations.Mdeficit[t], TOLERANCE):
            continue
        for k in range(solution.static.npeak):
            if not LocalDispatchPeakTier(solution, t, k):
                break
            NetworkDispatchPeakTier(solution, t, k)
            if not any_gt_1d(solution.operations.Mdeficit[t], TOLERANCE):
                break
        UpdateUnbalancedt(solution, t)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetForwardStorageHeadroom(solution, t, n):
    """Calculates headroom during the Forward Pass based on current SOC"""
    res = solution.static.resolution

    headroom = 0.0
    for s in range(solution.static.nstor):
        prev_soc = solution.operations.Mstorage_init[n, s] if t == 0 else solution.operations.Mstorage[t - 1, n, s]
        if s == 0:
            prev_soc += solution.static.TSphes_inflow[t, n]

        current_energy_change = res * (
            solution.operations.Mcharge[t, n, s] * solution.static.storage_charge_eff[s]
            - solution.operations.Mdischarge[t, n, s] / solution.static.storage_discha_eff[s]
        )

        max_e_power = (
            (solution.assets.CstorageE[n, s] - (prev_soc + current_energy_change))
            / solution.static.storage_charge_eff[s]
            / res
        )
        available_power = solution.assets.CstorageP[n, s] - solution.operations.Mcharge[t, n, s]

        headroom += max(0.0, min(available_power, max_e_power))
    return headroom


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def LocalDispatchPeakTier(solution, t, k):
    nodes = solution.static.nodes
    still_has_deficit = False
    for n in range(nodes):
        def_n = solution.operations.Mdeficit[t, n]
        if def_n > TOLERANCE:
            avail = GetPeakHeadroom(solution, t, n, k)
            dispatched = min(def_n, avail)
            if dispatched > TOLERANCE:
                DispatchPeak(solution, t, n, k, dispatched)
                def_n -= dispatched
                solution.operations.Mdeficit[t, n] = def_n
            if def_n > TOLERANCE:
                still_has_deficit = True
    return still_has_deficit


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def NetworkDispatchPeakTier(solution, t, k):
    nodes = solution.static.nodes
    working_buffer = solution.operations.surplus_buffer
    working_buffer_orig = solution.operations.surplus_orig

    has_headroom = False
    for n in range(nodes):
        hr = GetPeakHeadroom(solution, t, n, k)
        working_buffer[n] = hr
        if hr > TOLERANCE:
            has_headroom = True

    if has_headroom:
        working_buffer_orig[:] = working_buffer
        Interconnection(
            solution,
            solution.operations.Mdeficit[t],
            working_buffer,
            solution.operations.Tnetflow[t],
            solution.operations.Mimport[t],
        )
        for n in range(nodes):
            dispatched = working_buffer_orig[n] - working_buffer[n]
            if dispatched > TOLERANCE:
                DispatchPeak(solution, t, n, k, dispatched)
