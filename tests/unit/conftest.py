import numpy as np
import pytest

from firm_ce.common.typing import (
    TypedDict,
    nbintp,
    npfloat,
    npintp,
    unicode_type,
)
from firm_ce.scalar.entities.components import (
    Fleet,
    Fuel_InstanceType,
    Generator_InstanceType,
    Storage_InstanceType,
)
from firm_ce.scalar.entities.costs import construct_UnitCost_object
from firm_ce.scalar.entities.topology import (
    Line,
    Line_InstanceType,
    Network,
    Node,
    Node_InstanceType,
    routes_key_type,
    routes_list_type,
)
from firm_ce.system.parameters import ScenarioParameters
from firm_ce.tensor.entities.static import StaticTensor
from firm_ce.tensor.solution import SolutionTensor


def build_test_solution(
    intervals: int = 4,
    nodes: int = 3,
    lines: list[tuple[int, int, float, float]] | None = None,
    resolution: float = 1.0,
    years: int = 1,
    allowance_frac: float = 0.0,
) -> SolutionTensor:
    """
    Constructs a lightweight, real Numba `@jitclass` `SolutionTensor` in memory
    without any CSV/file I/O. All static, asset, and operational arrays can be
    mutated directly in individual unit tests.

    Parameters
    ----------
    intervals : int
        Number of simulation timesteps T.
    nodes : int
        Number of network nodes N.
    lines : list of (node_start, node_end, efficiency, capacity)
        Transmission lines to include. Defaults to [(0, 1, 0.9, 10.0), (1, 2, 0.9, 10.0)].
    resolution : float
        Interval duration in hours (Delta t).
    years : int
        Number of simulation years.
    allowance_frac : float
        Unserved energy allowance fraction.
    """
    if lines is None:
        if nodes >= 3:
            lines = [(0, 1, 0.9, 10.0), (1, 2, 0.9, 10.0)]
        elif nodes == 2:
            lines = [(0, 1, 0.9, 10.0)]
        else:
            lines = []

    year_first_t = np.linspace(0, intervals, years, endpoint=False, dtype=npintp)
    t_indices = np.arange(intervals, dtype=npintp)
    year_of_interval = (np.searchsorted(year_first_t, t_indices, side="right") - 1).astype(npintp)

    scen_params = ScenarioParameters(
        resolution=npfloat(resolution),
        allowance=npfloat(allowance_frac),
        first_year=npintp(2025),
        final_year=npintp(2025 + years - 1),
        year_count=npintp(years),
        leap_year_count=npintp(0),
        year_first_t=year_first_t,
        year_of_interval=year_of_interval,
        intervals_count=npintp(intervals),
        node_count=npintp(nodes),
    )

    nodes_dict = TypedDict.empty(key_type=nbintp, value_type=Node_InstanceType)
    for n in range(nodes):
        node_obj = Node(True, npintp(n), npintp(n), f"N{n}", npfloat(0.0))
        node_obj.data_status = True
        node_obj.data = np.zeros(intervals, dtype=npfloat)
        nodes_dict[npintp(n)] = node_obj

    dummy_cost = construct_UnitCost_object(
        capex_p=1.0,
        fom=0.0,
        vom=0.0,
        lifetime=30,
        discount_rate=0.05,
    )

    major_lines_dict = TypedDict.empty(key_type=nbintp, value_type=Line_InstanceType)
    for idx, (u, v, eff, cap) in enumerate(lines):
        loss_factor = 1.0 - eff
        line_obj = Line(
            True,
            npintp(idx),
            npintp(idx),
            f"L{idx}",
            npfloat(1000.0),
            nodes_dict[npintp(u)],
            nodes_dict[npintp(v)],
            npfloat(loss_factor),
            npfloat(100.0),
            npfloat(0.0),
            npfloat(cap),
            "ac",
            True,
            dummy_cost,
        )
        major_lines_dict[npintp(idx)] = line_obj

    minor_lines_dict = TypedDict.empty(key_type=nbintp, value_type=Line_InstanceType)
    routes_dict = TypedDict.empty(key_type=routes_key_type, value_type=routes_list_type)

    network = Network(
        True,
        nodes_dict,
        major_lines_dict,
        minor_lines_dict,
        routes_dict,
        0,
    )

    generators_dict = TypedDict.empty(key_type=nbintp, value_type=Generator_InstanceType)
    storages_dict = TypedDict.empty(key_type=nbintp, value_type=Storage_InstanceType)
    fuels_dict = TypedDict.empty(key_type=nbintp, value_type=Fuel_InstanceType)

    fleet = Fleet(True, generators_dict, storages_dict, fuels_dict)

    asset_node_map = TypedDict.empty(key_type=unicode_type, value_type=nbintp[:])
    n_lines = len(lines)
    abs_rel_scaler = np.ones(max(1, n_lines), dtype=npfloat)
    x = np.zeros(max(1, n_lines), dtype=npfloat)

    static = StaticTensor(
        scen_params,
        fleet,
        network,
        asset_node_map,
        abs_rel_scaler,
        False,
    )

    sol = SolutionTensor(x, static)
    # Default all operational arrays to clean zeros for deterministic unit tests
    sol.operations.Mnetload.fill(0.0)
    sol.operations.Munbalanced.fill(0.0)
    sol.operations.Mdeficit.fill(0.0)
    sol.operations.Mcurtail.fill(0.0)
    sol.operations.Mimport.fill(0.0)
    sol.operations.Tnetflow.fill(0.0)
    sol.operations.Mcharge.fill(0.0)
    sol.operations.Mdischarge.fill(0.0)
    sol.operations.Mstorage.fill(0.0)
    sol.operations.Mstorage_init.fill(0.0)
    sol.operations.Mphes_spill.fill(0.0)
    sol.operations.Mhydro.fill(0.0)
    sol.operations.Mreservoir.fill(0.0)
    sol.operations.Mreservoir_init.fill(0.0)
    sol.operations.Mhyd_spill.fill(0.0)
    sol.operations.Mpeak.fill(0.0)
    sol.operations.remaining_peak_budget.fill(0.0)
    return sol


@pytest.fixture
def make_solution():
    return build_test_solution


def pytest_sessionfinish(session, exitstatus):
    """Store exit status on config so pytest_unconfigure can exit cleanly if Numba threads linger."""
    session.config._exitstatus = exitstatus


def pytest_unconfigure(config):
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(getattr(config, "_exitstatus", 0))
