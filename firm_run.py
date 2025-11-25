from psutil import cpu_count

from firm.Input import (
    cost_model,
    x0,
)

from firm.Benchmark import (
    Benchmark,
    test,
    profile,
)

if __name__ == "__main__":

    import timeit

    # print("Before JIT")
    profile(x0, cost_model)
    test(x0, cost_model)

    n_attempts = 3
    n_parallel = 3
    n_eval = n_parallel*cpu_count(logical=True)
    print(f"Running timeit test now {n_attempts} attempts of {n_eval} in parallel")

    results = timeit.timeit(lambda: Benchmark(n_eval), number=n_attempts)/n_attempts
    print(f"Timeit calculated: {results} per batch of n_eval")
    print(f"\t{results/n_parallel} per parallel batch")
    print(f"\t{results/n_eval} per single eval")
