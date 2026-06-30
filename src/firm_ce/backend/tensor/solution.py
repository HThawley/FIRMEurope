# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit
from firm_ce.common.typing import boolean, nbfloat, npfloat
from firm_ce.system.tensors import (
    AssetTensor,
    AssetTensorType,
    CostTensorType,
    OperationTensor,
    OperationTensorType,
    StaticTensorType,
)
from firm_ce.common.helpers import safe_divide_array
from firm_ce.backend.tensor.simulation import Simulate


if JIT_ENABLED:
    from numba import set_num_threads
    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        # -- Solution Specification --
        ("x", nbfloat[:]),

        # -- Objects --
        ("static", StaticTensorType),
        ("costs", CostTensorType),
        ("assets", AssetTensorType),
        ("operations", OperationTensorType),

        # -- State variables --
        ("simulated", boolean),
        ("evaluated", boolean),

        # -- Computed Stats --
        ("estimated_deficit", nbfloat),
        ("penalties", nbfloat),
        ("feasible", boolean),
        ("total_annual_cost", nbfloat),
        ("lcoe", nbfloat)
    ]
else:
    solution_spec = []


@jitclass(solution_spec)
class SolutionTensor:
    def __init__(
        self,
        x: np.ndarray[npfloat],
        static: StaticTensorType,
        costs: CostTensorType,
    ):
        self.x = x
        self.static = static
        self.costs = costs
        self.assets = AssetTensor(x, self.static)
        self.operations = OperationTensor(self.static, self.assets)
        self.simulated = False
        self.evaluated = False

        self.estimated_deficit = -1.0
        self.penalties = -1.0
        self.feasible = True
        self.total_annual_cost = -1.0
        self.lcoe = -1.0


if JIT_ENABLED:
    SolutionTensorType = SolutionTensor.class_type.instance_type
else:
    SolutionTensorType = SolutionTensor


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def CalculateCost(
    solution: SolutionTensorType,
):
    a = solution.assets
    o = solution.operations
    c = solution.costs

    Mdischarge_sum = o.Mdischarge.sum(axis=0)
    Mpeak_sum = o.Mpeak.sum(axis=0)
    Mnuke_sum = o.Mnuke.sum(axis=0)

    # Apportionment not entirely accurate but this is a miniscule cost component
    _Mnuke_ratio = safe_divide_array(Mnuke_sum, a.Cnuke)
    nphes_discharge = safe_divide_array(Mdischarge_sum[:, 0], a.CstorageP[:, 0]) * a.CnphP

    cost = solution.static.legacy_costs
    cost += (
        # Fixed Costs
        a.Cpfix * c.Fpfix
        + a.Cpsat * c.Fpsat
        + a.Coffw * c.Foffw
        + a.Consw * c.Fonsw
        + a.Cpeak[:, 1] * c.Fbiog
        + a.Cpeak[:, 0] * c.Fbiom
        + a.Cpeak[:, 2] * c.Fgas
        + (a.Cnuke - a.Cnlte) * c.Fnuke
        + a.Cnlte * c.Fnlte
        + a.CnphP * c.FphP
        + a.CnphE * c.FphE
        + a.CstorageP[:, 1] * c.Fb4P
        + a.CstorageP[:, 2] * c.Fb2P

        # Variable Costs
        + o.Mpfix.sum(axis=0) * c.Vpfix
        + o.Mpsat.sum(axis=0) * c.Vpsat
        + o.Moffw.sum(axis=0) * c.Voffw
        + o.Monsw.sum(axis=0) * c.Vonsw
        + Mpeak_sum[:, 1] * c.Vbiog
        + Mpeak_sum[:, 0] * c.Vbiom
        + Mpeak_sum[:, 2] * c.Vgas
        + (_Mnuke_ratio * (a.Cnuke - a.Cnlte)) * c.Vnuke
        + (_Mnuke_ratio * a.Cnlte) * c.Vnlte
        + nphes_discharge * c.Vph
        + Mdischarge_sum[:, 1] * c.Vb4
        + Mdischarge_sum[:, 2] * c.Vb2
    ).sum()

    cost += (
        a.Clines * c.Flines
        + np.abs(o.Tnetflow).sum(axis=0) * c.Vlines
    ).sum()

    solution.total_annual_cost = cost
    solution.lcoe = cost / (solution.static.energy / solution.static.years_float)


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def EvaluateTensor(
    solution: SolutionTensorType,
):
    # TODO: import Simulate
    Simulate(solution)
    if solution.feasible:
        solution.penalties = 0.0
        solution.estimated_deficit = 0.0
        CalculateCost(solution)
    else:
        solution.penalties = ...
