import pytest

from firm_ce.tensor.dynamics import (
    CommitTrickle,
    DispatchPeak,
    GetHydroMinGen,
    GetLongDurSurplust,
    GetNaiveCurtailDeficit,
    GetPeakHeadroom,
    GetPhesMinGen,
    GetShortDurSurplust,
    GetTotalPeakHeadroomBound,
    ResetAnnualBudgets,
    UpdateLocalCharge,
    UpdateLocalChargeDischarge,
    UpdateLocalDischarge,
    UpdateSOCt,
    UpdateUnbalancedt,
)
from firm_ce.tensor.entities.operations import init_mnetload


def test_init_mnetload(make_solution):
    """Verifies vectorised netload calculation with VRE, nuclear, RoR, and internal loss."""
    sol = make_solution(intervals=2, nodes=1, lines=[])
    s, a, o = sol.static, sol.assets, sol.operations

    # Demand = 9.0 GW, internal_loss = 0.10 -> gross demand = 10.0 GW
    # RoR = 1.0 GW -> Mnetload_mror = 9.0 GW
    s.Mnetload_mror[0, 0] = 9.0
    s.Mnetload_mror[1, 0] = 5.0

    a.Cpfix[0], s.TSpfix[0, 0], s.TSpfix[1, 0] = 2.0, 0.5, 0.0      # 1.0, 0.0
    a.Cpsat[0], s.TSpsat[0, 0], s.TSpsat[1, 0] = 4.0, 0.25, 0.5     # 1.0, 2.0
    a.Coffw[0], s.TSoffw[0, 0], s.TSoffw[1, 0] = 3.0, 0.5, 1.0      # 1.5, 3.0
    a.Consw[0], s.TSonsw[0, 0], s.TSonsw[1, 0] = 2.0, 0.25, 0.5     # 0.5, 1.0
    a.Cnuke[0], s.TSnuke[0, 0], s.TSnuke[1, 0] = 2.0, 0.9, 0.9      # 1.8, 1.8

    init_mnetload(o.Mnetload, s, a)

    # t=0: 9.0 - (1.0 + 1.0 + 1.5 + 0.5 + 1.8) = 3.2
    # t=1: 5.0 - (0.0 + 2.0 + 3.0 + 1.0 + 1.8) = -2.8
    assert o.Mnetload[0, 0] == pytest.approx(3.2, abs=1e-5)
    assert o.Mnetload[1, 0] == pytest.approx(-2.8, abs=1e-5)


def test_get_hydro_and_phes_min_gen(make_solution):
    """Verifies must-run overflow floor calculation for hydro and open-loop PHES."""
    sol = make_solution(intervals=1, nodes=1, lines=[], resolution=2.0)
    s, a, o = sol.static, sol.assets, sol.operations

    # Hydro h=0: capacity E = 10 GWh, P = 3 GW, prev_soc = 8 GWh
    a.ChydE[0, 0] = 10.0
    a.ChydP[0, 0] = 3.0
    o.Mreservoir_init[0, 0] = 8.0

    # Case 1: Inflow = 1.5 GWh -> total 9.5 <= 10.0 -> min gen = 0
    s.TShyd_inflow[0, 0, 0] = 1.5
    assert GetHydroMinGen(sol, 0, 0, 0) == pytest.approx(0.0, abs=1e-6)

    # Case 2: Inflow = 6.0 GWh -> excess = 4.0 GWh over 2.0 h = 2.0 GW (<= 3.0 GW cap)
    s.TShyd_inflow[0, 0, 0] = 6.0
    assert GetHydroMinGen(sol, 0, 0, 0) == pytest.approx(2.0, abs=1e-5)

    # Case 3: Inflow = 12.0 GWh -> excess = 10.0 GWh over 2.0 h = 5.0 GW -> capped at ChydP = 3.0 GW
    s.TShyd_inflow[0, 0, 0] = 12.0
    assert GetHydroMinGen(sol, 0, 0, 0) == pytest.approx(3.0, abs=1e-5)

    # Open-loop PHES (s=0): capacity E = 20 GWh, P = 5 GW, prev_soc = 18 GWh, dis_eff = 0.8
    a.CstorageE[0, 0] = 20.0
    a.CstorageP[0, 0] = 5.0
    o.Mstorage_init[0, 0] = 18.0
    s.storage_discha_eff[0] = 0.8
    s.TSphes_inflow[0, 0] = 6.0  # excess = 4.0 GWh -> power = 4.0 * 0.8 / 2.0 = 1.6 GW
    assert GetPhesMinGen(sol, 0, 0) == pytest.approx(1.6, abs=1e-5)


def test_get_naive_curtail_deficit(make_solution):
    """Verifies must-run deduction and partitioning into Mdeficit and Mcurtail."""
    sol = make_solution(intervals=1, nodes=2, lines=[])
    s, a, o = sol.static, sol.assets, sol.operations

    # Node 0: netload = 5.0, pondage must-run = 1.5 -> deficit = 3.5
    o.Mnetload[0, 0] = 5.0
    a.ChydE[0, 0], a.ChydP[0, 0], o.Mreservoir_init[0, 0], s.TShyd_inflow[0, 0, 0] = 10.0, 5.0, 9.0, 2.5

    # Node 1: netload = -2.0, no must-run -> curtail = 2.0
    o.Mnetload[0, 1] = -2.0

    UpdateUnbalancedt(sol, 0)
    GetNaiveCurtailDeficit(sol, 0)

    assert o.Mhydro[0, 0, 0] == pytest.approx(1.5, abs=1e-5)
    assert o.Mdeficit[0, 0] == pytest.approx(3.5, abs=1e-5)
    assert o.Mcurtail[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert o.Mdeficit[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert o.Mcurtail[0, 1] == pytest.approx(2.0, abs=1e-5)
    assert o.has_deficit_t is True
    assert o.has_curtail_t is True


def test_update_local_charge_merit_order(make_solution):
    """Verifies local surplus charges shortest-duration storage first (s=3 -> 2 -> 1 -> 0)."""
    sol = make_solution(intervals=1, nodes=1, lines=[])
    s, a, o = sol.static, sol.assets, sol.operations

    # Set charge efficiency to 0.8 for all tiers
    s.storage_charge_eff[:] = 0.8
    # Give each tier P = 2.0 GW, E = 10.0 GWh, prev_soc = 9.2 GWh (so energy cap = (10 - 9.2) / 0.8 = 1.0 GW for s=3)
    for tier in range(4):
        a.CstorageP[0, tier] = 2.0
        a.CstorageE[0, tier] = 10.0
        o.Mstorage_init[0, tier] = 0.0
    # Constrain s=3 by energy to 1.0 GW charging; s=2 constrained by power to 2.0 GW; s=1 gets remaining 0.5 GW
    o.Mstorage_init[0, 3] = 9.2

    o.Munbalanced[0, 0] = -3.5
    UpdateLocalCharge(sol, 0)

    assert o.Mcharge[0, 0, 3] == pytest.approx(1.0, abs=1e-5)  # bess2h energy-limited
    assert o.Mcharge[0, 0, 2] == pytest.approx(2.0, abs=1e-5)  # bess4h power-limited
    assert o.Mcharge[0, 0, 1] == pytest.approx(0.5, abs=1e-5)  # clphes absorbs remainder
    assert o.Mcharge[0, 0, 0] == pytest.approx(0.0, abs=1e-5)  # olphes untouched
    assert o.Mcurtail[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_update_local_discharge_merit_order(make_solution):
    """
    Verifies local deficit discharges in exact merit order:
    pondage (h=0) -> bess2h (s=3) -> bess4h (s=2) -> clphes (s=1) -> olphes (s=0) -> reservoir (h=1).
    """
    sol = make_solution(intervals=1, nodes=1, lines=[])
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_discha_eff[:] = 1.0

    # Give each of the 6 tiers 1.0 GW of available discharge capability
    a.ChydP[0, 0], a.ChydE[0, 0], o.Mreservoir_init[0, 0] = 1.0, 10.0, 5.0
    for tier in range(4):
        a.CstorageP[0, tier] = 1.0
        a.CstorageE[0, tier] = 10.0
        o.Mstorage_init[0, tier] = 5.0
    a.ChydP[0, 1], a.ChydE[0, 1], o.Mreservoir_init[0, 1] = 1.0, 10.0, 5.0

    # Deficit of 5.4 GW should exhaust h=0, s=3, s=2, s=1, s=0 (1.0 each = 5.0) and draw 0.4 from h=1
    o.Munbalanced[0, 0] = 5.4
    UpdateLocalChargeDischarge(sol, 0)

    assert o.Mhydro[0, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdischarge[0, 0, 3] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdischarge[0, 0, 2] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdischarge[0, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdischarge[0, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert o.Mhydro[0, 0, 1] == pytest.approx(0.4, abs=1e-5)
    assert o.Mdeficit[0, 0] == pytest.approx(0.0, abs=1e-5)

    # Also verify UpdateLocalDischarge matches
    o.Munbalanced[0, 0] = 2.5
    UpdateLocalDischarge(sol, 0)
    assert o.Mhydro[0, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdischarge[0, 0, 3] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdischarge[0, 0, 2] == pytest.approx(0.5, abs=1e-5)
    assert o.Mdischarge[0, 0, 1] == pytest.approx(0.0, abs=1e-5)
    assert o.Mhydro[0, 0, 1] == pytest.approx(0.0, abs=1e-5)


def test_get_short_and_long_dur_surplus(make_solution):
    """Verifies short-duration (pondage, B2, B4) vs long-duration (reservoir, clphes, olphes) surplus grouping."""
    sol = make_solution(intervals=1, nodes=1, lines=[])
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_discha_eff[:] = 1.0

    # Short duration: pondage (h=0) = 1.0, B2 (s=3) = 2.0, B4 (s=2) = 3.0 -> sum = 6.0
    a.ChydP[0, 0], o.Mreservoir_init[0, 0] = 1.0, 10.0
    a.CstorageP[0, 3], o.Mstorage_init[0, 3] = 2.0, 10.0
    a.CstorageP[0, 2], o.Mstorage_init[0, 2] = 3.0, 10.0

    # Long duration: reservoir (h=1) = 4.0, clphes (s=1) = 5.0, olphes (s=0) = 6.0 -> sum = 15.0
    a.ChydP[0, 1], o.Mreservoir_init[0, 1] = 4.0, 10.0
    a.CstorageP[0, 1], o.Mstorage_init[0, 1] = 5.0, 10.0
    a.CstorageP[0, 0], o.Mstorage_init[0, 0] = 6.0, 10.0

    buf = o.surplus_buffer
    assert GetShortDurSurplust(sol, 0, buf) is True
    assert buf[0] == pytest.approx(6.0, abs=1e-5)

    assert GetLongDurSurplust(sol, 0, buf) is True
    assert buf[0] == pytest.approx(15.0, abs=1e-5)


def test_update_soc_t_and_spillage(make_solution):
    """Verifies SOC accrual and overflow spillage recording for hydro and open-loop PHES."""
    sol = make_solution(intervals=1, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 0.8
    s.storage_discha_eff[:] = 0.8

    # Hydro h=0: init = 9.0, inflow = 3.0, discharge = 1.0 -> theoretical = 11.0, cap = 10.0 -> spill = 1.0
    a.ChydE[0, 0] = 10.0
    o.Mreservoir_init[0, 0] = 9.0
    s.TShyd_inflow[0, 0, 0] = 3.0
    o.Mhydro[0, 0, 0] = 1.0

    # Open-loop PHES s=0: init = 8.0, inflow = 5.0, charge = 0, dis = 0 -> theoretical = 13.0, cap = 10.0 -> spill = 3.0
    a.CstorageE[0, 0] = 10.0
    o.Mstorage_init[0, 0] = 8.0
    s.TSphes_inflow[0, 0] = 5.0

    # B2 s=3: init = 2.0, charge = 2.0 (adds 1.6), dis = 0.8 (removes 1.0) -> theoretical = 2.6
    a.CstorageE[0, 3] = 5.0
    o.Mstorage_init[0, 3] = 2.0
    o.Mcharge[0, 0, 3] = 2.0
    o.Mdischarge[0, 0, 3] = 0.8

    UpdateSOCt(sol, 0)

    assert o.Mreservoir[0, 0, 0] == pytest.approx(10.0, abs=1e-5)
    assert o.Mhyd_spill[0, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert o.Mstorage[0, 0, 0] == pytest.approx(10.0, abs=1e-5)
    assert o.Mphes_spill[0, 0] == pytest.approx(3.0, abs=1e-5)
    assert o.Mstorage[0, 0, 3] == pytest.approx(2.6, abs=1e-5)


def test_commit_trickle_discharges_precharged_storage(make_solution):
    """Verifies CommitTrickle discharges newly pre-charged storage energy into remaining deficits."""
    sol = make_solution(intervals=2, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 1.0
    s.storage_discha_eff[:] = 0.8

    a.CstorageP[0, 3] = 3.0
    a.CstorageE[0, 3] = 6.0
    # After t=0 precharging, Mstorage[0, 0, 3] has 2.5 GWh stored (deliverable = 2.5 * 0.8 = 2.0 GW)
    o.Mstorage[0, 0, 3] = 2.5
    # At t=1, node has an unbalanced load (deficit) of 3.5 GW and 0 existing discharge
    o.Munbalanced[1, 0] = 3.5

    CommitTrickle(sol, 1)

    assert o.Mdischarge[1, 0, 3] == pytest.approx(2.0, abs=1e-5)
    assert o.Mdeficit[1, 0] == pytest.approx(1.5, abs=1e-5)


def test_peak_headroom_and_fuel_budget_scaling(make_solution):
    """Verifies GetPeakHeadroom, DispatchPeak, and GetTotalPeakHeadroomBound scale with resolution."""
    sol = make_solution(intervals=2, nodes=1, lines=[], resolution=2.0, years=1)
    s, a, o = sol.static, sol.assets, sol.operations

    # Biomass (k=0): P = 5.0 GW, annual budget = 6.0 GWh (with res=2.0h, max power from budget = 3.0 GW)
    # Biogas (k=1): P = 2.0 GW, annual budget = 10.0 GWh (max power = 2.0 GW, power-limited)
    # OCGT (k=2): P = 4.0 GW, no budget limit
    a.Cpeak[0, 0] = 5.0
    a.Cpeak[0, 1] = 2.0
    a.Cpeak[0, 2] = 4.0
    s.Bpeak[0, 0] = 6.0
    s.Bpeak[0, 1] = 10.0
    ResetAnnualBudgets(sol)

    assert GetPeakHeadroom(sol, 0, 0, 0) == pytest.approx(3.0, abs=1e-5)
    assert GetPeakHeadroom(sol, 0, 0, 1) == pytest.approx(2.0, abs=1e-5)
    assert GetPeakHeadroom(sol, 0, 0, 2) == pytest.approx(4.0, abs=1e-5)

    # Dispatch 2.0 GW of biomass at t=0 for 2.0 hours -> consumes 4.0 GWh, leaving 2.0 GWh (1.0 GW headroom)
    DispatchPeak(sol, 0, 0, 0, 2.0)
    assert o.remaining_peak_budget[0, 0] == pytest.approx(2.0, abs=1e-5)
    assert GetPeakHeadroom(sol, 1, 0, 0) == pytest.approx(1.0, abs=1e-5)

    # Total peak headroom bound across 2 intervals (res=2.0h):
    # Gas: 4.0 GW * 2 intervals * 2.0h = 16.0 GWh
    # Biomass: min((5.0*2 - 2.0)*2.0 = 16.0 GWh, remaining_budget = 2.0 GWh) = 2.0 GWh
    # Biogas: min(2.0*2*2.0 = 8.0 GWh, remaining_budget = 10.0 GWh) = 8.0 GWh
    # Total = 26.0 GWh
    assert GetTotalPeakHeadroomBound(sol) == pytest.approx(26.0, abs=1e-5)
