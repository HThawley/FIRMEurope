# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit, prange
from firm_ce.common.typing import nbfloat, npfloat, unicode_type
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.backend.scalar.solution import Solution, evaluate, extract_details


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_wrapper_with_details(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
    details: nbfloat[:, :, :],
) -> tuple[nbfloat[:], nbfloat[:]]:
    """
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    lcoe = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)
    # scaled_points = np.zeros(xs.shape, dtype=npfloat)

    pop_size = details.shape[1]

    for j in prange(n_points):
        xj = xs[j]
        solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        evaluate(solution)

        lcoe[j] = solution.lcoe
        penalties[j] = solution.penalties
        # scaled_points[j] = get_scaled_points(solution)

        niche_idx = j // pop_size
        indiv_idx = j % pop_size
        extract_details(solution, details[niche_idx, indiv_idx])

    return lcoe, penalties


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_wrapper(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
) -> tuple[nbfloat[:], nbfloat[:]]:
    """
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    lcoe = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)
    # scaled_points = np.zeros(xs.shape, dtype=npfloat)

    for j in prange(n_points):
        xj = xs[j]
        solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        evaluate(solution)

        lcoe[j] = solution.lcoe
        penalties[j] = solution.penalties
        # scaled_points[j] = get_scaled_points(solution)

    return lcoe, penalties
