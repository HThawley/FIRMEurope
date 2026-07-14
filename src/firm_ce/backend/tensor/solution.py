# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit
from firm_ce.common.typing import boolean, nbfloat, npfloat
from firm_ce.common.helpers import safe_divide_array
from firm_ce.backend.tensor.simulation import Simulate

from firm_ce.system.tensor.assets import AssetTensor, AssetTensorType
from firm_ce.system.tensor.operations import OperationTensor, OperationTensorType
from firm_ce.system.tensor.static import StaticTensorType

if JIT_ENABLED:
    from numba import set_num_threads
    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        # -- Solution Specification --
        ("x", nbfloat[:]),

        # -- Objects --
        ("static", StaticTensorType),
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
    ):
        self.x = x
        self.static = static
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
    c = solution.static.costs

    # Apportionment not entirely accurate but this is a miniscule cost component
    _Mnuke_ratio = o.Mnuke.sum(axis=0)
    safe_divide_array(_Mnuke_ratio, a.Cnuke, out=_Mnuke_ratio)

    # Apportionment not entirely accurate but this is a miniscule cost component
    _Mnphes_ratio = o.Mdischarge[:, :, 0].sum(axis=0)
    safe_divide_array(_Mnphes_ratio, a.CstorageP[:, 0], out=_Mnphes_ratio)

    cost = solution.static.legacy_costs

    for n in range(solution.static.nodes):
        cost += a.Cpfix[n] * c.Fpfix[n]
        cost += a.Cpsat[n] * c.Fpsat[n]
        cost += a.Coffw[n] * c.Foffw[n]
        cost += a.Consw[n] * c.Fonsw[n]
        cost += a.Cpeak[n, 1] * c.Fbiog[n]
        cost += a.Cpeak[n, 0] * c.Fbiom[n]
        cost += a.Cpeak[n, 2] * c.Fgas[n]
        cost += (a.Cnuke[n] - a.Cnlte[n]) * c.Fnuke[n]
        cost += a.Cnlte[n] * c.Fnlte[n]
        cost += a.CnphP[n] * c.FphP[n]
        cost += a.CnphE[n] * c.FphE[n]
        cost += a.CstorageP[n, 1] * c.Fb4P[n]
        cost += a.CstorageP[n, 2] * c.Fb2P[n]

        # Variable Costs
        cost += (_Mnuke_ratio[n] * (a.Cnuke[n] - a.Cnlte[n])) * c.Vnuke[n]
        cost += (_Mnuke_ratio[n] * a.Cnlte[n]) * c.Vnlte[n]
        cost += _Mnphes_ratio[n] * a.CnphP[n] * c.Vph[n]

    mpeak_by_node = o.Mpeak.sum(axis=0)   # stride-1 along T, good for SIMD

    cost += np.dot(mpeak_by_node[:, 0], c.Vbiom)
    cost += np.dot(mpeak_by_node[:, 1], c.Vbiog)
    cost += np.dot(mpeak_by_node[:, 2], c.Vgas)

    # # -- These all have 0.0 vom in current model --
    # for t in range(solution.static.intervals):
    #     for n in range(solution.static.nodes):
    #         cost += o.Mpfix[t, n] * c.Vpfix[n]
    #         cost += o.Mpsat[t, n] * c.Vpsat[n]
    #         cost += o.Moffw[t, n] * c.Voffw[n]
    #         cost += o.Monsw[t, n] * c.Vonsw[n]
    #         cost += np.abs(o.Tnetflow[t, n]) * c.Vlines[n]
    #         cost += o.Mdischarge[t, n, 1] * c.Vb4[n]
    #         cost += o.Mdischarge[t, n, 2] * c.Vb2[n]

    for n in range(solution.static.nhvi):
        cost += a.Clines[n] * c.Flines[n]

    solution.total_annual_cost = cost
    solution.lcoe = cost / solution.static.mean_annual_demand_mwh


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def EvaluateTensor(
    solution: SolutionTensorType,
):
    Simulate(solution)
    if solution.feasible:
        solution.penalties = 0.0
        solution.estimated_deficit = 0.0
        CalculateCost(solution)
    else:
        solution.penalties = solution.estimated_deficit
        CalculateCost(solution)
    solution.evaluated = True
