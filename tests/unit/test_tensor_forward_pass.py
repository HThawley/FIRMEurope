import pytest

from firm_ce.tensor.dynamics import ResetAnnualBudgets
from firm_ce.tensor.forward_pass import (
    ForwardPassPeak,
    ForwardPassRenewables,
    GetForwardStorageHeadroom,
    LocalDispatchPeakTier,
    NetworkDispatchPeakTier,
)


def test_get_forward_storage_headroom(make_solution):
    """Verifies forward storage charging headroom accounts for prev_soc, inflows, and active charge."""
    sol = make_solution(intervals=1, nodes=1, lines=[], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 0.8
    s.storage_discha_eff[:] = 0.8

    # Open-loop PHES (s=0): E = 10.0, P = 5.0, init_soc = 6.0, inflow = 2.4 -> free E = 1.6 -> max_e_p = 2.0 GW
    # Already charging 0.5 GW -> remaining power = 4.5, remaining energy power = (10 - (8.4 + 0.4))/0.8 = 1.5 GW
    a.CstorageE[0, 0] = 10.0
    a.CstorageP[0, 0] = 5.0
    o.Mstorage_init[0, 0] = 6.0
    s.TSphes_inflow[0, 0] = 2.4
    o.Mcharge[0, 0, 0] = 0.5

    assert GetForwardStorageHeadroom(sol, 0, 0) == pytest.approx(1.5, abs=1e-5)


def test_local_and_network_dispatch_peak_tier(make_solution):
    """Verifies LocalDispatchPeakTier dispatches local capacity and NetworkDispatchPeakTier transmits neighbour capacity."""
    # Node 0 -> Node 1 line with eff = 0.8, cap = 10.0
    sol = make_solution(intervals=1, nodes=2, lines=[(0, 1, 0.8, 10.0)])
    s, a, o = sol.static, sol.assets, sol.operations

    # Biomass (k=0): Node 0 has 5.0 GW spare, Node 1 has 2.0 GW local capacity and 4.4 GW deficit
    a.Cpeak[0, 0] = 5.0
    a.Cpeak[1, 0] = 2.0
    s.Bpeak[0, 0] = 100.0
    ResetAnnualBudgets(sol)

    o.Mdeficit[0, 1] = 4.4

    still_has_def = LocalDispatchPeakTier(sol, 0, 0)
    assert still_has_def is True
    assert o.Mpeak[0, 1, 0] == pytest.approx(2.0, abs=1e-5)
    assert o.Mdeficit[0, 1] == pytest.approx(2.4, abs=1e-5)

    # Network dispatch of tier 0: Node 1 needs 2.4 GW -> Node 0 must send 2.4 / 0.8 = 3.0 GW
    NetworkDispatchPeakTier(sol, 0, 0)
    assert o.Mpeak[0, 0, 0] == pytest.approx(3.0, abs=1e-5)
    assert o.Mdeficit[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_forward_pass_renewables_step_sequence(make_solution):
    """
    Verifies the chronological step hierarchy in ForwardPassRenewables:
    1. t=0: Node 0 has surplus (-4.0) and local battery; Node 1 has deficit (+1.8).
       Line 0->1 (eff=0.9) transmits 2.0 from N0 to N1 (meeting N1's 1.8 deficit) BEFORE N0 charges its own battery.
       N0's remaining 2.0 surplus charges N0's 2h battery (1.5 GW cap) and exports the remaining 0.5 GW to charge N1's battery.
    2. t=1: Node 1 has a deficit (+2.7) and no local reserves; it draws first from N0's short-duration B2 reserve
       before touching N0's long-duration PHES reserve.
    """
    sol = make_solution(intervals=2, nodes=2, lines=[(0, 1, 0.9, 10.0)], resolution=1.0)
    s, a, o = sol.static, sol.assets, sol.operations
    s.storage_charge_eff[:] = 1.0
    s.storage_discha_eff[:] = 1.0

    # Node 0: B2 (s=3) P=1.5, E=4.0, init=0.0; CLPHES (s=1) P=5.0, E=20.0, init=20.0 (full so it doesn't charge at t=0)
    a.CstorageP[0, 3], a.CstorageE[0, 3], o.Mstorage_init[0, 3] = 1.5, 4.0, 0.0
    a.CstorageP[0, 1], a.CstorageE[0, 1], o.Mstorage_init[0, 1] = 5.0, 20.0, 20.0
    # Node 1: B2 (s=3) P=2.0, E=4.0, init=0.0
    a.CstorageP[1, 3], a.CstorageE[1, 3], o.Mstorage_init[1, 3] = 2.0, 4.0, 0.0

    # Interval 0: N0 netload = -4.0 (surplus), N1 netload = +1.8 (deficit)
    o.Mnetload[0, 0] = -4.0
    o.Mnetload[0, 1] = 1.8

    # Interval 1: N0 netload = 0.0, N1 netload = +2.7 (deficit)
    o.Mnetload[1, 0] = 0.0
    o.Mnetload[1, 1] = 2.7

    ForwardPassRenewables(sol)

    # Check t=0:
    # 1. N0 sends 2.0 to N1 to cover 1.8 deficit.
    # 2. N0 charges local B2 at max power 1.5 GW (Mstorage[0, 0, 3] = 1.5).
    # 3. N0 sends remaining 0.5 surplus to N1, which receives 0.45 and charges N1's B2 (Mstorage[0, 1, 3] = 0.45).
    assert o.Mcharge[0, 0, 3] == pytest.approx(1.5, abs=1e-5)
    assert o.Mcharge[0, 1, 3] == pytest.approx(0.45, abs=1e-5)
    assert o.Mdeficit[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert o.Mcurtail[0, 0] == pytest.approx(0.0, abs=1e-5)

    # Check t=1:
    # N1 has 2.7 deficit. First it discharges its own local B2 (0.45 GWh -> leaves 2.25 deficit).
    # Next it imports short-duration reserve from N0's B2 (1.5 GW sent -> 1.35 received at N1 -> leaves 0.90 deficit).
    # Next it imports long-duration reserve from N0's CLPHES (1.0 GW sent -> 0.90 received at N1 -> 0 deficit).
    assert o.Mdischarge[1, 1, 3] == pytest.approx(0.45, abs=1e-5)
    assert o.Mdischarge[1, 0, 3] == pytest.approx(1.5, abs=1e-5)
    assert o.Mdischarge[1, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert o.Mdeficit[1, 1] == pytest.approx(0.0, abs=1e-5)


def test_forward_pass_peak_merit_order(make_solution):
    """
    Verifies ForwardPassPeak dispatches in strict 6-step merit order:
    local biomass -> network biomass -> local biogas -> network biogas -> local OCGT -> network OCGT.
    """
    sol = make_solution(intervals=1, nodes=2, lines=[(0, 1, 1.0, 20.0)])  # eff = 1.0 for clean arithmetic
    s, a, o = sol.static, sol.assets, sol.operations

    # Node 1 has a deficit of 5.5 GW; each of the 6 tiers across N1 (local) and N0 (neighbour) has 1.0 GW capacity
    for k in range(3):
        a.Cpeak[1, k] = 1.0  # local at Node 1
        a.Cpeak[0, k] = 1.0  # neighbour at Node 0
    s.Bpeak[0, 0] = 100.0
    s.Bpeak[0, 1] = 100.0
    ResetAnnualBudgets(sol)

    o.Mnetload[0, 1] = 5.5
    o.Mdeficit[0, 1] = 5.5

    ForwardPassPeak(sol)

    # Tiers 1..5 (local biom, net biom, local biog, net biog, local gas) dispatch full 1.0 GW each;
    # Tier 6 (net gas at Node 0) dispatches the remaining 0.5 GW.
    assert o.Mpeak[0, 1, 0] == pytest.approx(1.0, abs=1e-5)
    assert o.Mpeak[0, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert o.Mpeak[0, 1, 1] == pytest.approx(1.0, abs=1e-5)
    assert o.Mpeak[0, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert o.Mpeak[0, 1, 2] == pytest.approx(1.0, abs=1e-5)
    assert o.Mpeak[0, 0, 2] == pytest.approx(0.5, abs=1e-5)
    assert o.Mdeficit[0, 1] == pytest.approx(0.0, abs=1e-5)
