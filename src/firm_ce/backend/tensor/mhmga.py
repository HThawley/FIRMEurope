# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, NUM_THREADS
from firm_ce.common.jit_overload import njit, prange, get_thread_id, TypedList
from firm_ce.common.typing import nbfloat, npfloat
from firm_ce.system.tensor.static import StaticTensorType
from firm_ce.backend.tensor.solution import SolutionTensor, EvaluateTensor, ResetSolution

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

    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    lcoe = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    # This could be lifted out of loop, but I think not worth it
    pool = TypedList()
    for _ in range(NUM_THREADS):
        pool.append(SolutionTensor(xs[0], static))

    for j in prange(n_points):
        sol = pool[get_thread_id()]   # each thread owns its slot
        ResetSolution(sol, xs[j])
        EvaluateTensor(sol)
        lcoe[j] = sol.lcoe
        penalties[j] = sol.penalties

    return lcoe, penalties
