import numpy as np
import pytest

from firm_ce.common.typing import npfloat, npintp
from firm_ce.tensor.entities.network import GenerateTensorNetwork
from firm_ce.tensor.interconnection import Interconnection


def test_generate_tensor_network_csr():
    """Verifies CSR adjacency list construction and filtering of invalid node endpoints."""
    # Valid nodes: 0, 1, 2. Line 0: 0->1, Line 1: 1->2, Line 2: 2->99 (invalid endpoint 99)
    raw_network = np.array([[0, 1], [1, 2], [2, 99]], dtype=npintp)
    nodel_int = np.array([0, 1, 2], dtype=npintp)

    valid_net, mask, neighbors, lines, offsets = GenerateTensorNetwork(raw_network, nodel_int)

    assert list(mask) == [True, True, False]
    assert valid_net.shape == (2, 2)
    np.testing.assert_array_equal(valid_net[0], [0, 1])
    np.testing.assert_array_equal(valid_net[1], [1, 2])

    # Node 0 has 1 neighbour (1 via line 0)
    # Node 1 has 2 neighbours (0 via line 0, 2 via line 1)
    # Node 2 has 1 neighbour (1 via line 1)
    np.testing.assert_array_equal(offsets, [0, 1, 3, 4])
    np.testing.assert_array_equal(neighbors[offsets[0] : offsets[1]], [1])
    np.testing.assert_array_equal(lines[offsets[0] : offsets[1]], [0])
    np.testing.assert_array_equal(neighbors[offsets[1] : offsets[2]], [0, 2])
    np.testing.assert_array_equal(lines[offsets[1] : offsets[2]], [0, 1])
    np.testing.assert_array_equal(neighbors[offsets[2] : offsets[3]], [1])
    np.testing.assert_array_equal(lines[offsets[2] : offsets[3]], [1])


def test_interconnection_single_line_loss_and_bottleneck(make_solution):
    """Verifies flow over a single line respects efficiency loss and line capacity bottleneck."""
    # Line 0: Node 0 -> Node 1, eff = 0.8, capacity = 5.0 GW
    sol = make_solution(intervals=1, nodes=2, lines=[(0, 1, 0.8, 5.0)])

    fill = np.array([0.0, 10.0], dtype=npfloat)
    surplus = np.array([8.0, 0.0], dtype=npfloat)
    netflow = sol.operations.Tnetflow[0]
    imp = sol.operations.Mimport[0]

    Interconnection(sol, fill, surplus, netflow, imp)

    # Line capacity limits send from Node 0 to 5.0 GW; Node 1 receives 5.0 * 0.8 = 4.0 GW
    assert surplus[0] == pytest.approx(3.0, abs=1e-5)
    assert fill[1] == pytest.approx(6.0, abs=1e-5)
    assert netflow[0] == pytest.approx(5.0, abs=1e-5)
    assert imp[0] == pytest.approx(-5.0, abs=1e-5)
    assert imp[1] == pytest.approx(4.0, abs=1e-5)


def test_interconnection_multi_hop_max_efficiency_path(make_solution):
    """
    On a 3-node triangle:
      - Direct line L0 (0 -> 2) has eff = 0.70, cap = 10.0
      - 2-hop path L1 (0 -> 1, eff = 0.90, cap = 4.0) + L2 (1 -> 2, eff = 0.90, cap = 10.0)
        has cumulative eff = 0.81 > 0.70.
    Verifies Dijkstra uses the 2-hop path first until L1 saturates (4.0 sent -> 3.24 received),
    then routes remaining deficit over the direct line L0.
    """
    sol = make_solution(
        intervals=1,
        nodes=3,
        lines=[
            (0, 2, 0.70, 10.0),  # L0
            (0, 1, 0.90, 4.0),   # L1
            (1, 2, 0.90, 10.0),  # L2
        ],
    )

    # Node 2 needs 6.04 GW: 3.24 GW from 2-hop path (4.0 sent) + 2.80 GW from direct L0 (4.0 sent)
    fill = np.array([0.0, 0.0, 6.04], dtype=npfloat)
    surplus = np.array([20.0, 0.0, 0.0], dtype=npfloat)
    netflow = sol.operations.Tnetflow[0]
    imp = sol.operations.Mimport[0]

    Interconnection(sol, fill, surplus, netflow, imp)

    assert fill[2] == pytest.approx(0.0, abs=1e-5)
    assert surplus[0] == pytest.approx(12.0, abs=1e-5)  # 4.0 + 4.0 sent
    assert netflow[1] == pytest.approx(4.0, abs=1e-5)   # L1 saturated at 4.0
    assert netflow[2] == pytest.approx(3.6, abs=1e-5)   # L2 receives 4.0 * 0.9 = 3.6 from N1
    assert netflow[0] == pytest.approx(4.0, abs=1e-5)   # L0 sends 4.0
    assert imp[0] == pytest.approx(-8.0, abs=1e-5)
    assert imp[1] == pytest.approx(0.0, abs=1e-5)       # transit node N1 nets to 0
    assert imp[2] == pytest.approx(6.04, abs=1e-5)


def test_interconnection_counterflow_loss_recovery(make_solution):
    """
    When a line already carries reverse flow (netflow = -5.0 on 0 -> 1 with eff = 0.8,
    meaning Node 1 sent 5.0 and Node 0 received 4.0), sending 4.0 from Node 0 -> Node 1
    cancels the counter-flow with effective efficiency 1 / 0.8 = 1.25, delivering 5.0
    at Node 1 and bringing netflow to 0.0.
    """
    sol = make_solution(intervals=1, nodes=2, lines=[(0, 1, 0.8, 10.0)])

    netflow = sol.operations.Tnetflow[0]
    imp = sol.operations.Mimport[0]
    # Existing reverse flow: Node 1 exported 5.0, Node 0 imported 4.0
    netflow[0] = -5.0
    imp[0] = 4.0
    imp[1] = -5.0

    fill = np.array([0.0, 5.0], dtype=npfloat)
    surplus = np.array([4.0, 0.0], dtype=npfloat)

    Interconnection(sol, fill, surplus, netflow, imp)

    assert surplus[0] == pytest.approx(0.0, abs=1e-5)
    assert fill[1] == pytest.approx(0.0, abs=1e-5)
    assert netflow[0] == pytest.approx(0.0, abs=1e-5)
    assert imp[0] == pytest.approx(0.0, abs=1e-5)
    assert imp[1] == pytest.approx(0.0, abs=1e-5)


def test_interconnection_largest_deficit_first_priority(make_solution):
    """Verifies that the node with the largest fill requirement is served first."""
    sol = make_solution(
        intervals=1,
        nodes=3,
        lines=[
            (0, 1, 0.9, 10.0),
            (0, 2, 0.9, 10.0),
        ],
    )

    # Node 2 has larger deficit (8.0) than Node 1 (3.0); Node 0 only has 5.0 surplus
    fill = np.array([0.0, 3.0, 8.0], dtype=npfloat)
    surplus = np.array([5.0, 0.0, 0.0], dtype=npfloat)
    netflow = sol.operations.Tnetflow[0]
    imp = sol.operations.Mimport[0]

    Interconnection(sol, fill, surplus, netflow, imp)

    # All 5.0 surplus from Node 0 should go to Node 2 (delivering 4.5), leaving Node 1 untouched
    assert surplus[0] == pytest.approx(0.0, abs=1e-5)
    assert fill[2] == pytest.approx(3.5, abs=1e-5)
    assert fill[1] == pytest.approx(3.0, abs=1e-5)
