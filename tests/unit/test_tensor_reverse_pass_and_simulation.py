import pytest

from firm_ce.tensor.forward_pass import ForwardPassRenewables
from firm_ce.tensor.reverse_pass import (
    FillPrechargers,
    GetReverseStorageHeadroom,
    InitHydroMinFuture,
    InitStallTracking,
    InitStorageMinMaxFuture,
    ReversePassHydro,
    SetupPrechargePools,
    SetupStorageDonors,
    STALL_PATIENCE,
    UpdateDynamics,
    UpdateHydroMinFuture,
    UpdateStorageMinMaxFuture,
    WriteOffStalledHydroNodes,
)
from firm_ce.tensor.simulation import Simulate


def test_future_soc_envelope_tracking(make_solution):
    """Verifies backward minimum/maximum SOC envelope tracking and GetReverseStorageHeadroom."""
    sol = make_solution(intervals=3, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 0.8

    a.CstorageE[0, 3] = 10.0
    a.CstorageP[0, 3] = 5.0
    o.Mstorage[2, 0, 3] = 4.0
    o.Mstorage[1, 0, 3] = 8.4
    o.Mstorage[0, 0, 3] = 2.0

    InitStorageMinMaxFuture(sol, 2)
    assert o.storage_min_future[0, 3] == pytest.approx(4.0, abs=1e-5)
    assert o.storage_max_future[0, 3] == pytest.approx(4.0, abs=1e-5)

    UpdateStorageMinMaxFuture(sol, 1)
    assert o.storage_min_future[0, 3] == pytest.approx(4.0, abs=1e-5)
    assert o.storage_max_future[0, 3] == pytest.approx(8.4, abs=1e-5)

    UpdateStorageMinMaxFuture(sol, 0)
    assert o.storage_min_future[0, 3] == pytest.approx(2.0, abs=1e-5)
    assert o.storage_max_future[0, 3] == pytest.approx(8.4, abs=1e-5)

    # At t=0, future max SOC is 8.4, so max energy headroom is (10.0 - 8.4) / 0.8 = 2.0 GW
    assert GetReverseStorageHeadroom(sol, 0, 0, 3) == pytest.approx(2.0, abs=1e-5)

    # Also check hydro min future envelope
    o.Mreservoir[2, 0, 1] = 7.0
    o.Mreservoir[1, 0, 1] = 3.5
    InitHydroMinFuture(sol, 2)
    UpdateHydroMinFuture(sol, 1)
    assert o.hydro_min_future[0, 1] == pytest.approx(3.5, abs=1e-5)


def test_setup_precharge_pools_and_fill_prechargers(make_solution):
    """
    Verifies SetupPrechargePools grosses up charge_max_t by 1 / rt_eff and
    FillPrechargers decrements rolling_deficits by post-loss deliverable power (allocated * rt_eff).
    """
    sol = make_solution(intervals=1, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[3] = 0.8
    s.storage_discha_eff[3] = 0.8  # rt_eff = 0.64

    a.CstorageP[0, 3] = 5.0
    a.CstorageE[0, 3] = 10.0
    o.storage_max_future[0, 3] = 0.0
    o.rolling_deficits[0] = 1.6  # needs 1.6 / 0.64 = 2.5 GW of charging power

    has_fill = SetupPrechargePools(sol, 0)
    assert bool(has_fill) is True
    assert bool(o.precharge_flag[0, 3]) is True
    assert o.charge_max_t[0, 3] == pytest.approx(2.5, abs=1e-5)
    assert o.fill_buffer[0] == pytest.approx(2.5, abs=1e-5)

    # Supplying 2.5 GW of pre-charge power should reduce rolling_deficits by 2.5 * 0.64 = 1.6 to 0.0
    rem = FillPrechargers(sol, 0, 0, 2.5)
    assert rem == pytest.approx(0.0, abs=1e-5)
    assert o.Mcharge[0, 0, 3] == pytest.approx(2.5, abs=1e-5)
    assert o.rolling_deficits[0] == pytest.approx(0.0, abs=1e-5)


def test_setup_storage_donors_intranode_and_internode(make_solution):
    """
    Verifies SetupStorageDonors allows strictly longer-duration storage (s < s_pre)
    to act as an intra-node donor at a deficient node, while allowing all storage
    tiers with future energy headroom to act as inter-node donors at non-deficient nodes.
    """
    sol = make_solution(intervals=1, nodes=2, lines=[(0, 1, 1.0, 10.0)])
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_discha_eff[:] = 1.0

    for n in (0, 1):
        for tier in range(4):
            a.CstorageP[n, tier] = 2.0
            a.CstorageE[n, tier] = 10.0
            o.storage_min_future[n, tier] = 5.0

    # Node 0 has a rolling deficit and is pre-charging s=2 (bess4h)
    o.rolling_deficits[0] = 1.0
    o.precharge_flag[0, 2] = True

    # Node 1 has no rolling deficit
    o.rolling_deficits[1] = 0.0

    SetupStorageDonors(sol, 0)

    # At deficient Node 0: s=0 (olphes) and s=1 (clphes) can donate (s < 2), but s=2 and s=3 cannot
    assert list(o.trickling_flag[0]) == [True, True, False, False]
    # At non-deficient Node 1: all tiers s=0..3 can donate
    assert list(o.trickling_flag[1]) == [True, True, True, True]


def test_reverse_pass_hydro_resolves_power_constrained_deficit(make_solution):
    """
    2-step single-node test of ReversePassHydro + UpdateDynamics:
    Node 0 has a seasonal hydro reservoir (P = 2.0 GW, E = 20.0 GWh, init_soc = 10.0 GWh)
    and a 2h battery (s=3, P = 2.0 GW, E = 4.0 GWh, init_soc = 0.0 GWh, eff = 1.0).
    - At t=0: netload = 0.0 GW (both hydro and battery idle after ForwardPassRenewables).
    - At t=1: netload = 3.5 GW (hydro runs at max 2.0 GW, leaving a 1.5 GW deficit because battery is empty).
    ReversePassHydro should discharge 1.5 GW of spare hydro at t=0 to pre-charge the battery at t=0,
    and UpdateDynamics should discharge that 1.5 GW from the battery at t=1, eliminating the deficit.
    """
    sol = make_solution(intervals=2, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 1.0
    s.storage_discha_eff[:] = 1.0

    a.ChydP[0, 1], a.ChydE[0, 1], o.Mreservoir_init[0, 1] = 2.0, 20.0, 10.0
    a.CstorageP[0, 3], a.CstorageE[0, 3], o.Mstorage_init[0, 3] = 2.0, 4.0, 0.0

    o.Mnetload[0, 0] = 0.0
    o.Mnetload[1, 0] = 3.5

    ForwardPassRenewables(sol)
    assert o.Mdeficit[1, 0] == pytest.approx(1.5, abs=1e-5)

    ReversePassHydro(sol)
    UpdateDynamics(sol)

    assert o.Mhydro[0, 0, 1] == pytest.approx(1.5, abs=1e-5)
    assert o.Mcharge[0, 0, 3] == pytest.approx(1.5, abs=1e-5)
    assert o.Mdischarge[1, 0, 3] == pytest.approx(1.5, abs=1e-5)
    assert o.Mdeficit[1, 0] == pytest.approx(0.0, abs=1e-5)


def test_stall_detection_write_off(make_solution):
    """Verifies WriteOffStalledHydroNodes resets a stalled node's rolling deficit after STALL_PATIENCE checks."""
    sol = make_solution(intervals=1, nodes=1, lines=[])
    o = sol.operations

    InitStallTracking(sol)
    o.rolling_deficits[0] = 5.0

    # First check establishes the baseline checkpoint from STALL_CHECKPOINT_SENTINEL (counter remains 0)
    WriteOffStalledHydroNodes(sol)
    assert o.stall_checkpoint[0] == pytest.approx(5.0, abs=1e-5)
    assert o.stall_counter[0] == 0

    for step in range(STALL_PATIENCE - 1):
        WriteOffStalledHydroNodes(sol)
        assert o.rolling_deficits[0] == pytest.approx(5.0, abs=1e-5)
        assert o.stall_counter[0] == step + 1

    # On the STALL_PATIENCE-th consecutive check without shrinkage >= STALL_MIN_DELTA, backlog is written off
    WriteOffStalledHydroNodes(sol)
    assert o.rolling_deficits[0] == pytest.approx(0.0, abs=1e-5)
    assert o.stall_counter[0] == 0


def test_reverse_pass_peak_trickle_charging(make_solution):
    """
    Verifies ReversePassPeak uses spare peak generator headroom at t=0 to pre-charge
    a battery ahead of a t=1 demand spike that exceeds contemporaneous peak capacity.
    """
    sol = make_solution(intervals=2, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 1.0
    s.storage_discha_eff[:] = 1.0

    # Node 0 has 2.0 GW OCGT (k=2) and a 2h battery (s=3, P=2.0 GW, E=4.0 GWh, init=0.0)
    a.Cpeak[0, 2] = 2.0
    a.CstorageP[0, 3], a.CstorageE[0, 3], o.Mstorage_init[0, 3] = 2.0, 4.0, 0.0

    # t=0: netload = 0.5 GW (OCGT has 1.5 GW spare headroom)
    # t=1: netload = 3.2 GW (OCGT runs at 2.0 GW in ForwardPassPeak, leaving 1.2 GW deficit)
    o.Mnetload[0, 0] = 0.5
    o.Mnetload[1, 0] = 3.2

    Simulate(sol)

    # OCGT at t=0 should run at 0.5 (contemporaneous) + 1.2 (pre-charge) = 1.7 GW
    assert o.Mpeak[0, 0, 2] == pytest.approx(1.7, abs=1e-5)
    assert o.Mcharge[0, 0, 3] == pytest.approx(1.2, abs=1e-5)
    assert o.Mdischarge[1, 0, 3] == pytest.approx(1.2, abs=1e-5)
    assert o.Mdeficit[1, 0] == pytest.approx(0.0, abs=1e-5)
    assert sol.feasible is True


def test_simulate_early_exits_and_infeasibility_bound(make_solution):
    """Verifies Simulate early-exit feasibility and the 90% peak headroom infeasibility pre-check."""
    # Case 1: Deficit exceeds 0.9 * total_peak_headroom -> rejected immediately without ReversePassPeak
    sol = make_solution(intervals=2, nodes=1, lines=[], resolution=1.0)
    a, o = sol.assets, sol.operations
    a.Cpeak[0, 2] = 1.0  # Max 1.0 GW per interval
    a.CstorageP[0, 3], a.CstorageE[0, 3] = 2.0, 4.0
    o.Mnetload[0, 0] = 0.0   # 1.0 GWh spare peak headroom at t=0
    o.Mnetload[1, 0] = 3.0   # After 1.0 GW peak at t=1, 2.0 GWh deficit > 0.9 * 1.0 GWh remaining headroom

    Simulate(sol)
    assert sol.feasible is False
    assert sol.simulated is True
    # ReversePassPeak was skipped, so no pre-charging occurred at t=0
    assert o.Mcharge[0, 0, 3] == pytest.approx(0.0, abs=1e-5)
