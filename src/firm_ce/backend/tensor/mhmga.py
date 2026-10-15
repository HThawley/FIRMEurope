# type: ignore
import numpy as np

from firm_ce.common.constants import FASTMATH, BOUNDSCHECK, NUM_THREADS
from firm_ce.common.jit_overload import njit, prange, get_thread_id
from firm_ce.common.typing import boolean, nbfloat, npfloat, nbintp, TypedList
from firm_ce.system.tensor.static import StaticTensorType
from firm_ce.backend.tensor.solution import (
    SolutionTensor,
    EvaluateTensor,
    ResetSolution,
    ExtractDetailsTensor,
    ExtractEnergySpaceTensor,
)


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def _eval_mga_population_core(
    xs: nbfloat[:, :],
    static: StaticTensorType,
    save_details: boolean,
    details: nbfloat[:, :, :],
    return_scaled: boolean,
    nodal_demand_totals: nbfloat[:],
    norm_mode: nbintp,
    agg_mode: nbintp,
    scaled_dim: nbintp,
) -> tuple[nbfloat[:], nbfloat[:], nbfloat[:, :]]:
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    lcoe = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    scaled_rows = n_points if return_scaled else 1
    scaled = np.zeros((scaled_rows, scaled_dim), dtype=npfloat)
    pop_size = details.shape[1]

    n_workers = min(NUM_THREADS, n_points)
    pool = TypedList()
    for _ in range(n_workers):
        pool.append(SolutionTensor(xs[0], static))

    for j in prange(n_points):
        sol = pool[get_thread_id()]
        ResetSolution(sol, xs[j])
        EvaluateTensor(sol)
        lcoe[j] = sol.lcoe
        penalties[j] = sol.penalties

        if save_details:
            niche_idx = j // pop_size
            indiv_idx = j % pop_size
            ExtractDetailsTensor(sol, details[niche_idx, indiv_idx])

        if return_scaled:
            ExtractEnergySpaceTensor(
                sol,
                nodal_demand_totals,
                norm_mode,
                agg_mode,
                scaled[j],
            )

    return lcoe, penalties, scaled


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_tensor_wrapper(
    xs: nbfloat[:, :],
    static: StaticTensorType,
) -> tuple[nbfloat[:], nbfloat[:]]:
    dummy_details = np.empty((1, 1, 1), dtype=npfloat)
    dummy_demand = np.empty(1, dtype=npfloat)
    lcoe, penalties, _ = _eval_mga_population_core(
        xs, static, False, dummy_details, False, dummy_demand, 0, 0, 1
    )
    return lcoe, penalties


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_tensor_wrapper_with_details(
    xs: nbfloat[:, :],
    static: StaticTensorType,
    details: nbfloat[:, :, :],
) -> tuple[nbfloat[:], nbfloat[:]]:
    dummy_demand = np.empty(1, dtype=npfloat)
    lcoe, penalties, _ = _eval_mga_population_core(
        xs, static, True, details, False, dummy_demand, 0, 0, 1
    )
    return lcoe, penalties


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_tensor_wrapper_scaled(
    xs: nbfloat[:, :],
    static: StaticTensorType,
    nodal_demand_totals: nbfloat[:],
    norm_mode: nbintp,
    agg_mode: nbintp,
    scaled_dim: nbintp,
    save_details: boolean,
    details: nbfloat[:, :, :],
) -> tuple[nbfloat[:], nbfloat[:], nbfloat[:, :]]:
    return _eval_mga_population_core(
        xs,
        static,
        save_details,
        details,
        True,
        nodal_demand_totals,
        norm_mode,
        agg_mode,
        scaled_dim,
    )
