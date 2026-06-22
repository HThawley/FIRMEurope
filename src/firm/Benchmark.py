import numpy as np
from numba import njit, prange  # type: ignore
from time import perf_counter

from firm.Utils import zero_safe_division, cclock
from firm.Input import Evaluate, Solution


def Benchmark(n, static, cost_model):
    _benchmark(n, static, cost_model)


@njit(parallel=True)
def _benchmark(n, static, cost_model):
    result = np.empty(n)
    for j in prange(n):
        result[j] = test(static.x0, static, cost_model)


@njit
def test(x, static, cost_model):
    solution = Solution(x, static)
    Evaluate(solution, cost_model)
    return solution.Lcoe + solution.Penalties


@njit
def get_profile_overhead(solution, n_test):
    profile_overhead = 0.0
    for _ in range(n_test):
        start = cclock()
        solution.profile.calls.overhead += 1
        solution.profile.times.overhead += cclock() - start
    solution.profile.calls.overhead += 1
    profile_overhead /= n_test
    return profile_overhead


def profile(
    x,
    static,
    cost_model,
):
    solution = Solution(x, static)
    start = perf_counter()
    Evaluate(solution, cost_model)
    time = perf_counter() - start

    overhead = get_profile_overhead(solution, 2)  # compile getter
    overhead = get_profile_overhead(solution, 1000)  # actual overhead

    cputime = solution.profile.times.get_total()
    profiletime = solution.profile.calls.get_total() * overhead
    ctwt = zero_safe_division(time, (cputime + profiletime))

    return solution, time, ctwt, overhead
