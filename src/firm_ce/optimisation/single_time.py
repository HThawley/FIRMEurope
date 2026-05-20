# type: ignore
import time
import numpy as np

from firm_ce.common.constants import NUM_THREADS, PENALTY_MULTIPLIER, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit, prange
from firm_ce.common.typing import nbfloat, npfloat, unicode_type
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType
from firm_ce.optimisation.st_solution import Solution, evaluate


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
    A Solution instance is created for each candidate solution and evaluated. The parallel range splits the candidate
    solutions within the population across a number of workers (defined by the NUM_THREADS environment variable).
    This is an embarassingly parallel process.

    Parameters:
    -------
    xs (nbfloat[:, :]): 2-dimensional array containing population for an iteration of the differential
        evolution. Each row is a separate candidate solution, each column is a decision variable.
    static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
    network (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
    balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over
        the entire time horizon at the specified resolution).
    fixed_costs_threshold (nbfloat): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit
        committment problem.

    Returns:
    -------
    nbfloat[:, :]: A 2-dimensional array with 3 rows and a separate column for each candidate solution in the
        population. The first row is the total energy (cost) of the objective function, second row is the LCOE, and
        third row is the penalties for each candidate solution.
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
    This function is not JITed which allows the `time` package to be used to evaluate optimisation times for
    the iteration.

    Parameters:
    -------
    xs (nbfloat[:, :]): 2-dimensional array containing population for an iteration of the differential
        evolution. Each row is a separate candidate solution, each column is a decision variable.
    static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
    network (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
    balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over
        the entire time horizon at the specified resolution).
    fixed_costs_threshold (nbfloat): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit
        committment problem.

    Returns:
    -------
    nbfloat[:]: Total energies (costs) of the evaluated objective functions for each candidate solution in the
        population. Each column is the energy of a different candidate solution. The energy is the sum of LCOE
        and the penalties. This is the value minimised by the differential evolution optimisation.
    """
    start_time = time.time()
    result = parallel_wrapper(xs, static, fleet, network, balancing_type, fixed_costs_threshold)
    end_time = time.time()
    print(f"Average objective time: {NUM_THREADS*(end_time-start_time)/xs.shape[1]:.4f} seconds.")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds for {NUM_THREADS} workers.")
    return result[0]
