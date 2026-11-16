# type: ignore
import time
import numpy as np

from firm_ce.common.constants import NUM_THREADS, PENALTY_MULTIPLIER, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit, prange
from firm_ce.common.typing import boolean, nbfloat, npfloat, nbintp, unicode_type
from firm_ce.scalar.entities.components import Fleet_InstanceType
from firm_ce.scalar.entities.topology import Network_InstanceType
from firm_ce.scalar.solution import (
    Solution,
    evaluate,
    extract_details,
    extract_energy_space_scalar,
)
from firm_ce.system.parameters import ScenarioParameters_InstanceType


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def parallel_wrapper(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
) -> nbfloat[:, :]:
    """
    A wrapper receives the vectorised differential evolution population and evaluates it over a parallel range.
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=npfloat)
    for j in prange(n_points):
        xj = xs[:, j]
        solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        evaluate(solution)
        result[0, j] = solution.lcoe + solution.penalties * PENALTY_MULTIPLIER
        result[1, j] = solution.lcoe
        result[2, j] = solution.penalties
    return result


def evaluate_vectorised_xs(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
):
    """
    A wrapper receives the vectorised differential evolution population and passes it to the parallel wrapper.
    """
    start_time = time.time()
    result = parallel_wrapper(xs, static, fleet, network, balancing_type, fixed_costs_threshold)
    end_time = time.time()
    print(f"Average objective time: {NUM_THREADS*(end_time-start_time)/xs.shape[1]:.4f} seconds.")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds for {NUM_THREADS} workers.")
    return result[0]


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def _eval_mga_scalar_core(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
    save_details: boolean,
    details: nbfloat[:, :, :],
    return_scaled: boolean,
    nodal_demand_totals: nbfloat[:],
    norm_mode: nbintp,
    agg_mode: nbintp,
    n_gen_types: nbintp,
    n_energy_types: nbintp,
    scaled_dim: nbintp,
) -> tuple[nbfloat[:], nbfloat[:], nbfloat[:, :]]:
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    lcoe = np.zeros(n_points, dtype=npfloat)
    penalties = np.zeros(n_points, dtype=npfloat)

    scaled_rows = n_points if return_scaled else 1
    scaled = np.zeros((scaled_rows, scaled_dim), dtype=npfloat)
    pop_size = details.shape[1]

    for j in prange(n_points):
        xj = xs[j]
        solution = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        evaluate(solution)

        lcoe[j] = solution.lcoe
        penalties[j] = solution.penalties

        if save_details:
            niche_idx = j // pop_size
            indiv_idx = j % pop_size
            extract_details(solution, details[niche_idx, indiv_idx])

        if return_scaled:
            extract_energy_space_scalar(
                solution,
                nodal_demand_totals,
                norm_mode,
                agg_mode,
                n_gen_types,
                n_energy_types,
                scaled[j],
            )

    return lcoe, penalties, scaled


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_wrapper(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
) -> tuple[nbfloat[:], nbfloat[:]]:
    dummy_details = np.empty((1, 1, 1), dtype=npfloat)
    dummy_demand = np.empty(1, dtype=npfloat)
    lcoe, penalties, _ = _eval_mga_scalar_core(
        xs,
        static,
        fleet,
        network,
        balancing_type,
        fixed_costs_threshold,
        False,
        dummy_details,
        False,
        dummy_demand,
        0,
        0,
        1,
        1,
        1,
    )
    return lcoe, penalties


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_wrapper_with_details(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
    details: nbfloat[:, :, :],
) -> tuple[nbfloat[:], nbfloat[:]]:
    dummy_demand = np.empty(1, dtype=npfloat)
    lcoe, penalties, _ = _eval_mga_scalar_core(
        xs,
        static,
        fleet,
        network,
        balancing_type,
        fixed_costs_threshold,
        True,
        details,
        False,
        dummy_demand,
        0,
        0,
        1,
        1,
        1,
    )
    return lcoe, penalties


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def mga_scalar_wrapper_scaled(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
    nodal_demand_totals: nbfloat[:],
    norm_mode: nbintp,
    agg_mode: nbintp,
    n_gen_types: nbintp,
    n_energy_types: nbintp,
    scaled_dim: nbintp,
    save_details: boolean,
    details: nbfloat[:, :, :],
) -> tuple[nbfloat[:], nbfloat[:], nbfloat[:, :]]:
    return _eval_mga_scalar_core(
        xs,
        static,
        fleet,
        network,
        balancing_type,
        fixed_costs_threshold,
        save_details,
        details,
        True,
        nodal_demand_totals,
        norm_mode,
        agg_mode,
        n_gen_types,
        n_energy_types,
        scaled_dim,
    )
