from numba import njit  # type: ignore

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, TOLERANCE


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetSurplust(solution, t, Msurplust):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        surplus = solution.operations.Mcurtail[t, n]
        for s in range(solution.static.nstor):
            prev_soc = solution.operations.Mstorage_init[n, s] if t == 0 else solution.operations.Mstorage[t - 1, n, s]
            surplus += solution.operations.Mcharge[t, n, s]
            surplus += min(solution.assets.CstorageP[n, s], (prev_soc * solution.static.storage_discha_eff[s]) / res)
            surplus -= solution.operations.Mdischarge[t, n, s]

        for h in range(solution.static.nhyd):
            prev_soc = (
                solution.operations.Mreservoir_init[n, h] if t == 0
                else solution.operations.Mreservoir[t - 1, n, h]
            )
            surplus += min(solution.assets.ChydP[n, h], prev_soc / res) - solution.operations.Mhydro[t, n, h]

        Msurplust[n] = max(0.0, surplus)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def UpdateUnbalancedt(solution, t):
    for n in range(solution.static.nodes):
        solution.operations.Munbalanced[t, n] = solution.operations.Mnetload[t, n] - solution.operations.Mimport[t, n]


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetNaiveCurtailDeficit(solution, t):
    has_deficit = False
    has_curtail = False
    for n in range(solution.static.nodes):
        unbal = solution.operations.Munbalanced[t, n]
        if unbal > TOLERANCE:
            solution.operations.Mdeficit[t, n] = unbal
            solution.operations.Mcurtail[t, n] = 0.0
            has_deficit = True
        elif unbal < -TOLERANCE:
            solution.operations.Mdeficit[t, n] = 0.0
            solution.operations.Mcurtail[t, n] = -unbal
            has_curtail = True
        else:
            solution.operations.Mdeficit[t, n] = 0.0
            solution.operations.Mcurtail[t, n] = 0.0

    solution.operations.has_deficit_t = has_deficit
    solution.operations.has_curtail_t = has_curtail


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def UpdateLocalCharge(solution, t):
    res = solution.static.resolution
    has_deficit = False
    has_curtail = False

    for n in range(solution.static.nodes):
        unbal = solution.operations.Munbalanced[t, n]

        # Wipe local arrays to allow safe recalculation from scratch
        for s in range(solution.static.nstor):
            solution.operations.Mcharge[t, n, s] = 0.0
        solution.operations.Mdeficit[t, n] = 0.0
        solution.operations.Mcurtail[t, n] = 0.0

        if unbal < -TOLERANCE:
            for s in range(solution.static.nstor):
                prev_soc = (
                    solution.operations.Mstorage_init[n, s] if t == 0
                    else solution.operations.Mstorage[t - 1, n, s]
                )
                if s == 0:
                    prev_soc += solution.static.TSphes_inflow[t, n]

                charge_cap = (solution.assets.CstorageE[n, s] - prev_soc) / solution.static.storage_charge_eff[s] / res
                charge_amt = max(0.0, min(-unbal, solution.assets.CstorageP[n, s], charge_cap))
                solution.operations.Mcharge[t, n, s] = charge_amt
                unbal += charge_amt

            if unbal < -TOLERANCE:
                solution.operations.Mcurtail[t, n] = -unbal
                has_curtail = True
        elif unbal > TOLERANCE:
            solution.operations.Mdeficit[t, n] = unbal
            has_deficit = True

    # Lock state for the interval
    solution.operations.has_curtail_t = has_curtail
    solution.operations.has_deficit_t = has_deficit


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def UpdateLocalDischarge(solution, t):
    res = solution.static.resolution
    has_deficit = False

    for n in range(solution.static.nodes):
        unbal = solution.operations.Munbalanced[t, n]

        # Unbal must account for any charging that has happened
        for s in range(solution.static.nstor):
            unbal += solution.operations.Mcharge[t, n, s]

        # Wipe discharge arrays
        for s in range(solution.static.nstor):
            solution.operations.Mdischarge[t, n, s] = 0.0
        for h in range(solution.static.nhyd):
            solution.operations.Mhydro[t, n, h] = 0.0

        if unbal > TOLERANCE:
            # Pondage (h=0)
            h = 0
            prev_soc = (
                solution.operations.Mreservoir_init[n, h] if t == 0
                else solution.operations.Mreservoir[t - 1, n, h]
            )
            inflow_e = solution.static.TShyd_inflow[t, n, h]
            discharge_cap = (prev_soc + inflow_e) / res
            discharge_amt = max(0.0, min(unbal, solution.assets.ChydP[n, h], discharge_cap))
            solution.operations.Mhydro[t, n, h] = discharge_amt
            unbal -= discharge_amt

            # Storage (s=0 to 3)
            for s in range(solution.static.nstor):
                prev_soc = (
                    solution.operations.Mstorage_init[n, s] if t == 0
                    else solution.operations.Mstorage[t - 1, n, s]
                )
                if s == 0:
                    prev_soc += solution.static.TSphes_inflow[t, n]

                discharge_cap = prev_soc * solution.static.storage_discha_eff[s] / res
                discharge_amt = max(0.0, min(unbal, solution.assets.CstorageP[n, s], discharge_cap))
                solution.operations.Mdischarge[t, n, s] = discharge_amt
                unbal -= discharge_amt

            # Reservoir (h=1)
            h = 1
            prev_soc = (
                solution.operations.Mreservoir_init[n, h] if t == 0
                else solution.operations.Mreservoir[t - 1, n, h]
            )
            inflow_e = solution.static.TShyd_inflow[t, n, h]
            discharge_cap = (prev_soc + inflow_e) / res
            discharge_amt = max(0.0, min(unbal, solution.assets.ChydP[n, h], discharge_cap))
            solution.operations.Mhydro[t, n, h] = discharge_amt
            unbal -= discharge_amt

            # peak deduction
            for k in range(solution.static.npeak):
                unbal -= solution.operations.Mpeak[t, n, k]

            if unbal > TOLERANCE:
                solution.operations.Mdeficit[t, n] = unbal
                has_deficit = True
            else:
                solution.operations.Mdeficit[t, n] = 0.0
        else:
            solution.operations.Mdeficit[t, n] = 0.0

    solution.operations.has_deficit_t = has_deficit


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetLongDurSurplust(solution, t, Msurplust):
    res = solution.static.resolution
    Msurplust.fill(0.0)
    for n in range(solution.static.nodes):
        surplus = 0.0
        # Pondage & Reservoir
        for h in range(solution.static.nhyd):
            prev_soc = (solution.operations.Mreservoir_init[n, h] if t == 0
                        else solution.operations.Mreservoir[t - 1, n, h])
            inflow_e = solution.static.TShyd_inflow[t, n, h]
            discharge_cap = (prev_soc + inflow_e) / res
            surplus += max(
                0.0, min(solution.assets.ChydP[n, h], discharge_cap) - solution.operations.Mhydro[t, n, h])

        # PHES (s=0)
        s = 0
        prev_soc = solution.operations.Mstorage_init[n, s] if t == 0 else solution.operations.Mstorage[t - 1, n, s]
        inflow_e = solution.static.TSphes_inflow[t, n]
        discharge_cap = (prev_soc + inflow_e) * solution.static.storage_discha_eff[s] / res
        surplus += max(
            0.0, min(solution.assets.CstorageP[n, s], discharge_cap) - solution.operations.Mdischarge[t, n, s]
        )

        Msurplust[n] = surplus


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetShortDurSurplust(solution, t, Msurplust):
    res = solution.static.resolution
    Msurplust.fill(0.0)
    for n in range(solution.static.nodes):
        surplus = 0.0
        # B4, B2, B1 (s=1, 2, 3)
        for s in range(1, solution.static.nstor):
            prev_soc = solution.operations.Mstorage_init[n, s] if t == 0 else solution.operations.Mstorage[t - 1, n, s]
            discharge_cap = prev_soc * solution.static.storage_discha_eff[s] / res
            surplus += max(
                0.0, min(solution.assets.CstorageP[n, s], discharge_cap) - solution.operations.Mdischarge[t, n, s]
            )

        Msurplust[n] = surplus


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def CommitTrickle(solution, t):
    """Safely updates ledger without erasing trickle allocations"""
    has_deficit = False
    has_curtail = False
    for n in range(solution.static.nodes):
        unbal = solution.operations.Munbalanced[t, n]

        for s in range(solution.static.nstor):
            unbal += solution.operations.Mcharge[t, n, s] - solution.operations.Mdischarge[t, n, s]
        for h in range(solution.static.nhyd):
            unbal -= solution.operations.Mhydro[t, n, h]
        for k in range(solution.static.npeak):
            unbal -= solution.operations.Mpeak[t, n, k]

        if unbal > TOLERANCE:
            solution.operations.Mdeficit[t, n] = unbal
            solution.operations.Mcurtail[t, n] = 0.0
            has_deficit = True
        elif unbal < -TOLERANCE:
            solution.operations.Mdeficit[t, n] = 0.0
            solution.operations.Mcurtail[t, n] = -unbal
            has_curtail = True
        else:
            solution.operations.Mdeficit[t, n] = 0.0
            solution.operations.Mcurtail[t, n] = 0.0

    solution.operations.has_deficit_t = has_deficit
    solution.operations.has_curtail_t = has_curtail


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def UpdateSOCt(solution, t):
    res = solution.static.resolution
    for n in range(solution.static.nodes):
        for h in range(solution.static.nhyd):
            prev_soc = (
                solution.operations.Mreservoir_init[n, h] if t == 0
                else solution.operations.Mreservoir[t - 1, n, h]
            )

            inflow_e = solution.static.TShyd_inflow[t, n, h]

            theoretical_soc = prev_soc + inflow_e - (solution.operations.Mhydro[t, n, h] * res)
            solution.operations.Mhyd_spill[t, n, h] = max(0.0, theoretical_soc - solution.assets.ChydE[n, h])
            solution.operations.Mreservoir[t, n, h] = min(solution.assets.ChydE[n, h], theoretical_soc)

        for s in range(solution.static.nstor):
            prev_soc = solution.operations.Mstorage_init[n, s] if t == 0 else solution.operations.Mstorage[t - 1, n, s]
            net_charge = res * (
                solution.operations.Mcharge[t, n, s] * solution.static.storage_charge_eff[s]
                - solution.operations.Mdischarge[t, n, s] / solution.static.storage_discha_eff[s]
            )
            theoretical_soc = prev_soc + net_charge
            if s == 0:
                theoretical_soc += solution.static.TSphes_inflow[t, n]
                solution.operations.Mphes_spill[t, n] = max(0.0, theoretical_soc - solution.assets.CstorageE[n, 0])

            solution.operations.Mstorage[t, n, s] = min(solution.assets.CstorageE[n, s], theoretical_soc)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def ResetAnnualBudgets(solution):
    solution.operations.remaining_peak_budget[:, :] = solution.static.Bpeak


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetPeakHeadroom(solution, t, n, k):
    """Available headroom for tech k at (t, n): power-capped always,
    additionally budget-capped for biomass/biogas (gas has no annual limit)."""
    power_headroom = solution.assets.Cpeak[n, k] - solution.operations.Mpeak[t, n, k]
    if k == 2:
        return max(0.0, power_headroom)

    year = solution.static.year_of_interval[t]
    budget_headroom = solution.operations.remaining_peak_budget[year, k]
    return max(0.0, min(power_headroom, budget_headroom))


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def DispatchPeak(solution, t, n, k, amount):
    """Commits a dispatch amount to tech k, depleting the shared annual
    budget if applicable. amount must already respect GetPeakHeadroom."""
    solution.operations.Mpeak[t, n, k] += amount
    if k != 2:
        year = solution.static.year_of_interval[t]
        solution.operations.remaining_peak_budget[year, k] -= amount


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, inline="always")
def GetTotalPeakHeadroomBound(solution):
    """Sound upper bound on total deliverable peak energy across the whole
    horizon, used as the cheap feasibility pre-check before ReversePassPeak.
    Gas: power-only. Biomass/biogas: power-rate AND remaining annual budget
    both bound it -- using the smaller of the two keeps the bound sound."""
    intervals = solution.static.intervals

    gas_headroom = (
        solution.assets.Cpeak[:, 2].sum() * intervals
        - solution.operations.Mpeak[:, :, 2].sum()
    )

    biomass_power = (
        solution.assets.Cpeak[:, 0].sum() * intervals
        - solution.operations.Mpeak[:, :, 0].sum()
    )
    biomass_budget = solution.operations.remaining_peak_budget[:, 0].sum()
    biomass_headroom = min(biomass_power, biomass_budget)

    biogas_power = (
        solution.assets.Cpeak[:, 1].sum() * intervals
        - solution.operations.Mpeak[:, :, 1].sum()
    )
    biogas_budget = solution.operations.remaining_peak_budget[:, 1].sum()
    biogas_headroom = min(biogas_power, biogas_budget)

    return max(0.0, gas_headroom) + max(0.0, biomass_headroom) + max(0.0, biogas_headroom)
