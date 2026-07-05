# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit, prange
from firm_ce.common.typing import nbfloat, npfloat
from firm_ce.system.tensor.static import StaticTensorType
from firm_ce.backend.tensor.solution import SolutionTensor, EvaluateTensor

# @njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
# def mga_wrapper_with_details(
#     xs: nbfloat[:, :],
#     static: StaticTensorType,
#     details: nbfloat[:, :, :],
# ) -> tuple[nbfloat[:], nbfloat[:]]:
#     """
#     """
#     # TODO: Rewrite this legacy scalar code to use tensor
#     xs = xs.astype(nbfloat)
#     n_points = xs.shape[0]
#     lcoe = np.zeros(n_points, dtype=npfloat)
#     penalties = np.zeros(n_points, dtype=npfloat)
#     # scaled_points = np.zeros(xs.shape, dtype=npfloat)

#     pop_size = details.shape[1]

#     for j in prange(n_points):
#         xj = xs[j]
#         solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
#         evaluate(solution)

#         lcoe[j] = solution.lcoe
#         penalties[j] = solution.penalties
#         # scaled_points[j] = get_scaled_points(solution)

#         niche_idx = j // pop_size
#         indiv_idx = j % pop_size
#         extract_details(solution, details[niche_idx, indiv_idx])

#     return lcoe, penalties


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_tensor_wrapper(
    xs: nbfloat[:, :],
    static: StaticTensorType,
) -> tuple[nbfloat[:], nbfloat[:]]:
    """
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    lcoe = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    for j in prange(n_points):
        xj = xs[j]
        solution = SolutionTensor(xj, static)
        EvaluateTensor(solution)

        lcoe[j] = solution.lcoe
        penalties[j] = solution.penalties

    return lcoe, penalties
