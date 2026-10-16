# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit, prange
from firm_ce.common.typing import boolean, nbfloat, npfloat, TypedDict, nbintp, npintp
from firm_ce.backend.tensor.simulation import Simulate

from firm_ce.system.tensor.assets import AssetTensor, AssetTensorType
from firm_ce.system.tensor.operations import OperationTensor, OperationTensorType, reset_operations
from firm_ce.system.tensor.static import StaticTensorType

# Column indices in SolutionTensor.nodal_energy (shape: nodes x 13)
# 0: pfix, 1: psat, 2: offw, 3: onsw, 4: nuke, 5: nlte,
# 6: biom, 7: biog, 8: gas, 9: phes_dis, 10: b4_dis, 11: b2_dis, 12: curtail
N_ENERGY_COMPONENTS = 13

# Integer codes for Numba energy-space extraction
NORM_RELATIVE = npintp(0)
NORM_ABSOLUTE = npintp(1)

AGG_NONE = npintp(0)
AGG_UNIT_TYPE = npintp(1)
AGG_NODE = npintp(2)

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

        # -- Pre-allocated Nodal Operational Summary --
        ("nodal_energy", nbfloat[:, :]),
        ("nodal_nph_gen", nbfloat[:]),

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
        self.nodal_energy = np.zeros((self.static.nodes, N_ENERGY_COMPONENTS), dtype=npfloat)
        self.nodal_nph_gen = np.zeros(self.static.nodes, dtype=npfloat)

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
    solution.nodal_energy[:, :] = npfloat(0.0)
    solution.nodal_nph_gen[:] = npfloat(0.0)
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
    s = solution.static
    c = s.costs
    E = solution.nodal_energy

    E[:, :] = npfloat(0.0)

    # Single-pass time accumulation over intervals into nodal_energy (shape: nodes x 13)
    for t in range(s.intervals):
        for n in range(s.nodes):
            E[n, 0] += s.TSpfix[t, n]
            E[n, 1] += s.TSpsat[t, n]
            E[n, 2] += s.TSoffw[t, n]
            E[n, 3] += s.TSonsw[t, n]
            E[n, 4] += o.Mnuke[t, n]
            E[n, 6] += o.Mpeak[t, n, 0]
            E[n, 7] += o.Mpeak[t, n, 1]
            E[n, 8] += o.Mpeak[t, n, 2]
            E[n, 9] += o.Mdischarge[t, n, 0]
            E[n, 10] += o.Mdischarge[t, n, 1]
            E[n, 11] += o.Mdischarge[t, n, 2]
            E[n, 12] += np.abs(o.Mcurtail[t, n])

    cost = s.legacy_costs

    for n in range(s.nodes):
        # Convert VRE capacity factor time-sums into gross energy
        E[n, 0] *= a.Cpfix[n]
        E[n, 1] *= a.Cpsat[n]
        E[n, 2] *= a.Coffw[n]
        E[n, 3] *= a.Consw[n]

        # Apportion total nuclear generation (currently in E[n, 4]) between new nuke and LTE nuke
        tot_nuke_gen = E[n, 4]
        nuke_ratio = tot_nuke_gen / a.Cnuke[n] if a.Cnuke[n] > 0.0 else npfloat(0.0)
        E[n, 4] = nuke_ratio * (a.Cnuke[n] - a.Cnlte[n])  # new nuclear generation
        E[n, 5] = nuke_ratio * a.Cnlte[n]                 # LTE nuclear generation

        # Apportion total PHES discharge (E[n, 9]) to candidate new-build PHES for variable cost
        phes_ratio = E[n, 9] / a.CstorageP[n, 0] if a.CstorageP[n, 0] > 0.0 else npfloat(0.0)
        solution.nodal_nph_gen[n] = phes_ratio * a.CnphP[n]

        # Fixed Costs
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

        # Variable Costs (using pre-summed nodal totals!)
        cost += E[n, 4] * c.Vnuke[n]
        cost += E[n, 5] * c.Vnlte[n]
        cost += solution.nodal_nph_gen[n] * c.Vph[n]
        cost += E[n, 6] * c.Vbiom[n]
        cost += E[n, 7] * c.Vbiog[n]
        cost += E[n, 8] * c.Vgas[n]

        # -- Zero-VOM technologies in current model --
        # cost += E[n, 0] * c.Vpfix[n]
        # cost += E[n, 1] * c.Vpsat[n]
        # cost += E[n, 2] * c.Voffw[n]
        # cost += E[n, 3] * c.Vonsw[n]
        # cost += E[n, 10] * c.Vb4[n]
        # cost += E[n, 11] * c.Vb2[n]

    for n in range(s.nhvi):
        cost += a.Clines[n] * c.Flines[n]

    solution.total_annual_cost = cost
    solution.lcoe = cost / s.mean_annual_demand_mwh


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def ExtractDetailsTensor(
    solution: SolutionTensorType,
    details_slice: nbfloat[:],
) -> None:
    """
    O(nodes) extraction of system-wide unit-type operational totals from `solution.nodal_energy`.
    Zero time-series iteration required.
    """
    E = solution.nodal_energy
    nph_gen = solution.nodal_nph_gen
    n_nodes = solution.static.nodes

    details_slice[:] = npfloat(0.0)
    details_slice[0] = solution.lcoe
    details_slice[1] = solution.penalties

    for n in range(n_nodes):
        details_slice[2] += E[n, 4]      # nuclear
        details_slice[3] += E[n, 5]      # nuclear_lte
        details_slice[4] += nph_gen[n]   # nphes
        details_slice[5] += E[n, 6]      # biomass
        details_slice[6] += E[n, 7]      # biogas
        details_slice[7] += E[n, 8]      # ccgt (gas)

        # -- Zero-VOM technologies in current model --
        # details_slice[8] += E[n, 0]    # pv_fixed
        # details_slice[9] += E[n, 1]    # pv_track
        # details_slice[10] += E[n, 2]   # offw
        # details_slice[11] += E[n, 3]   # onsw
        # details_slice[12] += E[n, 10]  # bess4h
        # details_slice[13] += E[n, 11]  # bess2h


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def ExtractEnergySpaceTensor(
    solution: SolutionTensorType,
    nodal_demand_totals: nbfloat[:],
    norm_mode: nbintp,
    agg_mode: nbintp,
    scaled_slice: nbfloat[:],
) -> None:
    """
    O(nodes) projection of `solution.nodal_energy` (gross generation + storage discharge
    + curtailment) into the target MGA fitness space (`scaled_slice`).
    """
    E = solution.nodal_energy
    n_nodes = solution.static.nodes
    scaled_slice[:] = npfloat(0.0)

    # Total system demand over the horizon = sum of nodal_demand_totals
    sys_demand_total = npfloat(0.0)
    for n in range(n_nodes):
        sys_demand_total += nodal_demand_totals[n]

    if agg_mode == AGG_UNIT_TYPE:
        # Output shape: (N_ENERGY_COMPONENTS,) = (13,)
        inv_nodes = npfloat(1.0) / npfloat(n_nodes)
        inv_sys_demand = npfloat(1.0) / sys_demand_total if sys_demand_total > 0.0 else npfloat(1.0)
        for n in range(n_nodes):
            weight = (inv_nodes / nodal_demand_totals[n]) if norm_mode == NORM_RELATIVE else inv_sys_demand
            for k in range(N_ENERGY_COMPONENTS):
                scaled_slice[k] += E[n, k] * weight

    elif agg_mode == AGG_NODE:
        # Output shape: (n_nodes,)
        inv_sys_demand = npfloat(1.0) / sys_demand_total if sys_demand_total > 0.0 else npfloat(1.0)
        for n in range(n_nodes):
            weight = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == NORM_RELATIVE else inv_sys_demand
            node_sum = npfloat(0.0)
            for k in range(N_ENERGY_COMPONENTS):
                node_sum += E[n, k]
            scaled_slice[n] = node_sum * weight

    else:  # AGG_NONE
        # Output shape: (n_nodes * N_ENERGY_COMPONENTS,)
        inv_sys_demand = npfloat(1.0) / sys_demand_total if sys_demand_total > 0.0 else npfloat(1.0)
        idx = 0
        for n in range(n_nodes):
            weight = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == NORM_RELATIVE else inv_sys_demand
            for k in range(N_ENERGY_COMPONENTS):
                scaled_slice[idx] = E[n, k] * weight
                idx += 1


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def EvaluateTensorFromDetails(
    x: nbfloat[:],
    static: StaticTensorType,
    details_slice: nbfloat[:],
) -> tuple[nbfloat, nbfloat]:
    """
    Reconstructs total annual cost and LCOE from decision vector `x`, `static`, and
    `details_slice` WITHOUT allocating SolutionTensor or OperationTensor.
    """
    a = AssetTensor(x, static)
    c = static.costs
    cost = static.legacy_costs

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

    cost += details_slice[2] * c.Vnuke[0]
    cost += details_slice[3] * c.Vnlte[0]
    cost += details_slice[4] * c.Vph[0]
    cost += details_slice[5] * c.Vbiom[0]
    cost += details_slice[6] * c.Vbiog[0]
    cost += details_slice[7] * c.Vgas[0]

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
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    solutions = TypedDict()

    for j in range(n_points):
        j_ = npintp(j)
        solutions[j_] = SolutionTensor(xs[j_], static)

    for j in prange(n_points):
        EvaluateTensor(solutions[npintp(j)])

    return solutions
