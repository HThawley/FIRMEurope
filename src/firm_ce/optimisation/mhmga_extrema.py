# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit, prange
from firm_ce.common.typing import nbintp, nbfloat, npfloat, unicode_type
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.backend.scalar.solution import Solution, evaluate, extract_details


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def apply_new_weighting(xs, details, weighting):
    xs = xs.astype(nbfloat)
    details = details.astype(nbfloat)

    n_points = xs.shape[0]
    ndim = xs.shape[1]

    weighted_sum = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    for j in prange(n_points):
        xj = xs[j]

        for k in range(ndim):
            weighted_sum[j] += xj[k] * weighting[k]

        penalties[j] += details[j, 1]

    return weighted_sum, penalties


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_wrapper_with_details(
    xs: nbfloat[:, :],
    weighting: nbfloat[:],
    lcoe_constr: nbfloat,
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
    details: nbfloat[:, :, :],
    niche_tracker: nbintp[:],
) -> tuple[nbfloat[:], nbfloat[:], nbfloat[:, :]]:
    """
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    ndim = xs.shape[1]

    weighted_sum = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    niche_idx = niche_tracker[0]

    for j in prange(n_points):
        xj = xs[j]
        solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        evaluate(solution)

        for k in range(ndim):
            weighted_sum[j] += xj[k] * weighting[k]

        if solution.lcoe > lcoe_constr:
            penalties[j] = (solution.lcoe - lcoe_constr)

        penalties[j] += solution.penalties
        extract_details(solution, details[niche_idx, j])

    niche_tracker[0] += 1

    return weighted_sum, penalties


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_wrapper(
    xs: nbfloat[:, :],
    weighting: nbfloat[:],
    lcoe_constr: nbfloat,
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
) -> tuple[nbfloat[:], nbfloat[:], nbfloat[:, :]]:
    """
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    ndim = xs.shape[1]

    weighted_sum = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    for j in prange(n_points):
        xj = xs[j]
        solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        evaluate(solution)

        for k in range(ndim):
            weighted_sum[j] += xj[k] * weighting[k]

        if solution.lcoe > lcoe_constr:
            penalties[j] = (solution.lcoe - lcoe_constr)

        penalties[j] += solution.penalties

    return weighted_sum, penalties
