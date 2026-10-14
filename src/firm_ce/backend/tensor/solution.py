# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit, prange
from firm_ce.common.typing import boolean, nbfloat, npfloat, TypedDict, nbintp
from firm_ce.common.helpers import safe_divide_array
from firm_ce.backend.tensor.simulation import Simulate

from firm_ce.system.tensor.assets import AssetTensor, AssetTensorType
from firm_ce.system.tensor.operations import OperationTensor, OperationTensorType, reset_operations
from firm_ce.system.tensor.static import StaticTensorType

# Dynamic definition of tensor details columns.
TENSOR_DETAILS_COLUMNS = (
    "lcoe",
    "penalties",
    "nuke",
    "nlte",
    "nph",
    "biom",
    "biog",
    "gas",
    # -- Zero-VOM technologies in current model --
    # "pfix",
    # "psat",
    # "offw",
    # "onsw",
    # "b4",
    # "b2",
    # "lines",
)
TENSOR_DETAILS_LENGTH = len(TENSOR_DETAILS_COLUMNS)

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


@njit
def ResetSolution(
    solution: SolutionTensor,
    x: np.ndarray[npfloat]
) -> None:
    solution.x = x
    solution.assets = AssetTensor(x, solution.static)  # cheap enough to rebuild rather than reset
    reset_operations(solution.operations, solution.static, solution.assets)
    solution.simulated = False
    solution.evaluated = False
    solution.feasible = True
    solution.estimated_deficit = npfloat(-1.0)
    solution.penalties = npfloat(-1.0)
    solution.total_annual_cost = npfloat(-1.0)
    solution.lcoe = npfloat(-1.0)


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

    for t in range(solution.static.intervals):
        for n in range(solution.static.nodes):
            cost += o.Mpeak[t, n, 0] * c.Vbiom[n]
            cost += o.Mpeak[t, n, 1] * c.Vbiog[n]
            cost += o.Mpeak[t, n, 2] * c.Vgas[n]

            # -- These all have 0.0 vom in current model --
            # cost += o.Mpfix[t, n] * c.Vpfix[n]
            # cost += o.Mpsat[t, n] * c.Vpsat[n]
            # cost += o.Moffw[t, n] * c.Voffw[n]
            # cost += o.Monsw[t, n] * c.Vonsw[n]
            # cost += np.abs(o.Tnetflow[t, n]) * c.Vlines[n]
            # cost += o.Mdischarge[t, n, 1] * c.Vb4[n]
            # cost += o.Mdischarge[t, n, 2] * c.Vb2[n]

    for n in range(solution.static.nhvi):
        cost += a.Clines[n] * c.Flines[n]

    solution.total_annual_cost = cost
    solution.lcoe = cost / solution.static.mean_annual_demand_mwh


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def ExtractDetailsTensor(
    solution: SolutionTensorType,
    details_slice: nbfloat[:],
) -> None:
    """
    Aggregates operational dispatch by unit_type across the system into a 1D slice.
    Matches the column order defined in TENSOR_DETAILS_COLUMNS.
    """
    a = solution.assets
    o = solution.operations
    s = solution.static

    details_slice[:] = npfloat(0.0)
    details_slice[0] = solution.lcoe
    details_slice[1] = solution.penalties

    _Mnuke_ratio = o.Mnuke.sum(axis=0)
    safe_divide_array(_Mnuke_ratio, a.Cnuke, out=_Mnuke_ratio)

    _Mnphes_ratio = o.Mdischarge[:, :, 0].sum(axis=0)
    safe_divide_array(_Mnphes_ratio, a.CstorageP[:, 0], out=_Mnphes_ratio)

    nuke_gen = npfloat(0.0)
    nlte_gen = npfloat(0.0)
    nph_gen = npfloat(0.0)

    for n in range(s.nodes):
        nuke_gen += _Mnuke_ratio[n] * (a.Cnuke[n] - a.Cnlte[n])
        nlte_gen += _Mnuke_ratio[n] * a.Cnlte[n]
        nph_gen += _Mnphes_ratio[n] * a.CnphP[n]

    biom_gen = npfloat(0.0)
    biog_gen = npfloat(0.0)
    gas_gen = npfloat(0.0)

    # -- Zero-VOM accumulators --
    # pfix_gen = npfloat(0.0)
    # psat_gen = npfloat(0.0)
    # offw_gen = npfloat(0.0)
    # onsw_gen = npfloat(0.0)
    # b4_gen = npfloat(0.0)
    # b2_gen = npfloat(0.0)
    # lines_flow = npfloat(0.0)

    for t in range(s.intervals):
        for n in range(s.nodes):
            biom_gen += o.Mpeak[t, n, 0]
            biog_gen += o.Mpeak[t, n, 1]
            gas_gen += o.Mpeak[t, n, 2]

            # -- Zero-VOM technologies in current model --
            # pfix_gen += s.TSpfix[t, n] * a.Cpfix[n]
            # psat_gen += s.TSpsat[t, n] * a.Cpsat[n]
            # offw_gen += s.TSoffw[t, n] * a.Coffw[n]
            # onsw_gen += s.TSonsw[t, n] * a.Consw[n]
            # b4_gen += o.Mdischarge[t, n, 1]
            # b2_gen += o.Mdischarge[t, n, 2]
            # lines_flow += np.abs(o.Tnetflow[t, n])

    details_slice[2] = nuke_gen
    details_slice[3] = nlte_gen
    details_slice[4] = nph_gen
    details_slice[5] = biom_gen
    details_slice[6] = biog_gen
    details_slice[7] = gas_gen

    # -- Zero-VOM assignments --
    # details_slice[8] = pfix_gen
    # details_slice[9] = psat_gen
    # details_slice[10] = offw_gen
    # details_slice[11] = onsw_gen
    # details_slice[12] = b4_gen
    # details_slice[13] = b2_gen
    # details_slice[14] = lines_flow


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def EvaluateTensorFromDetails(
    x: nbfloat[:],
    static: StaticTensorType,
    details_slice: nbfloat[:],
) -> tuple[nbfloat, nbfloat]:
    """
    Reconstructs total annual cost and LCOE from decision vector `x`, `static` (potentially
    with updated cost assumptions), and `details_slice` WITHOUT allocating SolutionTensor
    or OperationTensor.

    Assumes variable costs are spatially uniform across nodes per unit type (indexing node 0).
    Returns (lcoe, penalties).
    """
    a = AssetTensor(x, static)
    c = static.costs
    cost = static.legacy_costs

    # Fixed Costs
    for n in range(static.nodes):
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

    for n in range(static.nhvi):
        cost += a.Clines[n] * c.Flines[n]

    # Variable Costs (aggregated by unit type, using uniform cost at index 0)
    cost += details_slice[2] * c.Vnuke[0]
    cost += details_slice[3] * c.Vnlte[0]
    cost += details_slice[4] * c.Vph[0]
    cost += details_slice[5] * c.Vbiom[0]
    cost += details_slice[6] * c.Vbiog[0]
    cost += details_slice[7] * c.Vgas[0]

    # -- Zero-VOM technologies in current model --
    # cost += details_slice[8] * c.Vpfix[0]
    # cost += details_slice[9] * c.Vpsat[0]
    # cost += details_slice[10] * c.Voffw[0]
    # cost += details_slice[11] * c.Vonsw[0]
    # cost += details_slice[12] * c.Vb4[0]
    # cost += details_slice[13] * c.Vb2[0]
    # cost += details_slice[14] * c.Vlines[0]

    lcoe = cost / static.mean_annual_demand_mwh
    penalties = details_slice[1]
    return lcoe, penalties


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


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def prep_solution_for_postprocessing(
    solution: SolutionTensorType,
) -> None:
    """
    Calculates some arrays which are required for downstream post-processing
    but which are not calculated in EvaluateTensor for code performance
    """
    if not solution.evaluated:
        raise Exception

    o = solution.operations
    s = solution.static
    a = solution.assets

    o.Mpfix = s.TSpfix * a.Cpfix
    o.Mpsat = s.TSpsat * a.Cpsat
    o.Moffw = s.TSoffw * a.Coffw
    o.Monsw = s.TSonsw * a.Consw

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, parallel=True)
def build_eval_and_return_solutions(
    xs: nbfloat[:, :],
    static: StaticTensorType,
) -> TypedDict[nbintp, SolutionTensorType]:
    """
    Convenience for efficient post-processing - not to be used in optimisation loops
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    solutions = TypedDict()

    for j in prange(n_points):
        solutions[j] = SolutionTensor(xs[j], static)
        EvaluateTensor(solutions[j])

    return solutions
