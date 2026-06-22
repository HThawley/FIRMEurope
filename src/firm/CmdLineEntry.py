import click
from firm.Parameters import Parameters, DE_Hyperparameters
from rich.console import Console  # type: ignore
from rich.text import Text  # type: ignore
from rich.table import Table  # type: ignore
from psutil import cpu_count
from time import perf_counter


@click.command
@click.option(
    "-s",
    "--scenario",
    default=1,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
def statistics(scenario: int):
    import numpy as np

    try:
        x = np.genfromtxt(f"Results/Optimisation_resultx{scenario}.csv", delimiter=",")
    except FileNotFoundError as e:
        print("No solution found. Run optimisation first.")
        raise e
    from firm.Statistics import Information

    Information(x)


@click.command
@click.option(
    "-s",
    "--scenario",
    default=1,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y",
    "--years",
    default=-1,
    type=click.IntRange(-1),
    required=False,
    show_default=True,
    help="no. of years to model",
)
@click.option(
    "-m",
    "--networksteps",
    default=-1,
    type=click.IntRange(-1),
    required=False,
    show_default=True,
    help="Maximum number of transmission steps for balancing",
)
@click.option(
    "-n",
    "--number",
    default=3,
    type=click.IntRange(1),
    show_default=True,
    required=False,
    help="How many batches",
)
@click.option(
    "-e",
    "--evals",
    default=cpu_count(True) * 3,
    type=click.IntRange(1),
    show_default=True,
    required=False,
    help="How many evaluations per batch",
)
def benchmark(
    scenario: int,
    years: int,
    networksteps: int,
    number: int,
    evals: int,
):

    print("Running Benchmarking...", end="")
    from firm.Benchmark import Benchmark
    from firm.Input import StaticData
    from firm.Costs import RawCosts

    parameters = Parameters(scenario, years, 0, networksteps)
    static = StaticData(*parameters)
    cost_model = RawCosts(static).CostFactors()
    # compile
    Benchmark(2, static, cost_model)
    start = perf_counter()
    for i in range(number):
        Benchmark(evals, static, cost_model)
    time = perf_counter() - start
    print(f"\rBenchmarking took {time/number} per parallel batch of {evals} ({time/number/evals} per eval).")


@click.command
@click.option(
    "-s",
    "--scenario",
    default=1,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y",
    "--years",
    default=-1,
    type=click.IntRange(-1),
    required=False,
    show_default=True,
    help="no. of years to model",
)
@click.option(
    "-p",
    "--profile-level",
    default=-1,
    type=click.IntRange(-1, 3),
    required=False,
    show_default=True,
    help="Profiling level",
)
@click.option(
    "-m",
    "--networksteps",
    default=-1,
    type=click.IntRange(-1),
    required=False,
    show_default=True,
    help="Maximum number of transmission steps for balancing",
)
def profile(
    scenario: int,
    years: int,
    profile_level: int,
    networksteps: int,
):
    # %%
    print("Running Profiling...", end="")
    from firm.Benchmark import profile
    from firm.Input import StaticData
    from firm.Costs import RawCosts
    from firm.Utils import zero_safe_division

    parameters = Parameters(scenario, years, profile_level, networksteps)
    static = StaticData(*parameters)
    cost_model = RawCosts(static).CostFactors()

    # ctwt - cpu time to wall time
    solution, time, ctwt, overhead = profile(static.x0, static, cost_model)  # compile
    solution, time, ctwt, overhead = profile(static.x0, static, cost_model)
    ctwt = ctwt * 1000  # seconds to microseconds
    print("\r", " " * 25, "\r")
    print(
        """Warning:
profile measures cpu-time.
Table times are wall-time apportioned over cpu-cycles
Not all of the profiling overhead is accounted for.
    empirically, unprofiled time is as much as ~30%
    faster (also depends on level).
Profiling overhead is not evenly split between components
    higher level functions (e.g. Transmission) which call lower level funcs
    have higher proportions of overhead attached
          """
    )
    table = Table(title="Profile Results")

    table.add_column("Function")
    table.add_column("Calls")
    table.add_column("Cpu-cycles")
    table.add_column("Cpu-cycles per call")
    table.add_column("Apportioned Time (ms)")
    table.add_column("Time per call")

    calls_profile = solution.profile.calls.get_total()
    time_profile = calls_profile * overhead

    table.add_row(
        "Profiler overhead",
        str(calls_profile),
        str(time_profile),
        str(overhead),
        str(ctwt * time_profile),
        str(ctwt * overhead),
    )
    for name, attr in [
        ("Simulation", "simulation"),
        ("Basic Sim", "basic"),
        ("interc0", "interc0"),
        ("interc1", "interc1"),
        ("interc2", "interc2"),
        ("interc3", "interc3"),
        ("storage behav", "storage_behavior"),
        ("storage behav t", "storage_behaviort"),
        ("spill/def", "spilldef"),
        ("spill/def t", "spilldeft"),
        ("soc", "update_soc"),
        ("soc t", "update_soct"),
        ("unbalanced", "unbalanced"),
        ("unbalanced t", "unbalancedt"),
        ("clip fill", "clip_fill"),
        ("get surplus", "get_surplus"),
    ]:
        _calls = getattr(solution.profile.calls, attr)
        _times = getattr(solution.profile.times, attr)
        _cycles_per_call = zero_safe_division(_times, _calls)
        table.add_row(
            name,
            str(_calls),
            str(_times),
            str(_cycles_per_call),
            str(ctwt * _times),
            str(ctwt * _cycles_per_call),
        )

    print("\r", " " * 30, sep="", end="")
    results = Table(title="Solution Result")
    results.add_column("")
    results.add_column("Value")
    results.add_row(
        "Lcoe",
        str(solution.Lcoe),
    )
    results.add_row(
        "Penalties",
        str(solution.Penalties),
    )
    results.add_row(
        "Wall-time",
        str(time),
    )

    console = Console()
    console.print(table)
    console.print(results)


# %%


@click.command
@click.option(
    "-s",
    "--scenario",
    default=1,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y",
    "--years",
    default=1,
    type=click.IntRange(-1, 10, clamp=True),
    required=False,
    show_default=True,
    help="No. of years to simulate. -1 indicates max",
)
@click.option(
    "-m",
    "--networksteps",
    default=-1,
    type=click.IntRange(-1),
    required=False,
    show_default=True,
    help="Maximum number of transmission steps for balancing",
)
@click.option(
    "-i",
    "--iterations",
    default=1000,
    show_default=True,
    type=click.IntRange(1, 4000),
    required=False,
    help="Maximum iterations",
)
@click.option(
    "-p",
    "--popsize",
    default=50,
    show_default=True,
    type=click.IntRange(2, 1000),
    required=False,
    help="Population size",
)
@click.option(
    "-m",
    "--mutation",
    default=None,
    show_default=True,
    type=click.FloatRange(0.0, 2.0),
    required=False,
    help="Mutation factor",
)
@click.option(
    "-d",
    "--dither",
    nargs=2,
    default=(None, None),
    type=click.Tuple([click.FloatRange(0.0, 2.0), click.FloatRange(0.0, 2.0)]),
    required=False,
    help="Mutation factor dither range (overrides --mutation)",
)
@click.option(
    "-r",
    "--recombination",
    default=0.4,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    required=False,
    help="Recombination factor",
)
@click.option(
    "-v",
    "--progress",
    default=1,
    type=click.IntRange(0),
    show_default=True,
    required=False,
    help="Print progress to console",
)
@click.option(
    "-e",
    "--stagnation",
    default=(5, 0.1),
    type=click.Tuple([click.IntRange(0), click.FloatRange(0)]),
    show_default=True,
    required=False,
    help="Stagnation condition: (no. of its, minimum improvement)",
)
@click.option(
    "-f",
    "--fileprint",
    default=1,
    type=click.IntRange(0),
    show_default=True,
    required=False,
    help="Frequency to print to file",
)
def optimise(
    scenario: int,
    years: int,
    networksteps: int,
    iterations: int,
    popsize: int,
    mutation: float,
    dither: tuple[float, float],
    recombination: float,
    progress: int,
    stagnation: tuple[int, float],
    fileprint: int,
):

    param = Parameters(
        s=scenario,
        y=years,
        p=0,
        n=networksteps,
    )
    hyperparam = DE_Hyperparameters(
        i=iterations,
        p=popsize,
        m=(
            mutation
            if mutation is not None
            else dither if dither[0] is not None and dither[1] is not None else (0.5, 1.0)
        ),
        r=recombination,
        v=progress,
        s=stagnation,
        f=fileprint,
    )

    from firm.Input import StaticData
    from firm.Optimisation import Optimise

    static = StaticData(*param)
    result, time = Optimise(static, hyperparam)
    # print(result.x)


@click.command
@click.option(
    "-s",
    "--scenario",
    default=1,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y",
    "--years",
    default=1,
    type=click.IntRange(-1, 10, clamp=True),
    required=False,
    show_default=True,
    help="No. of years to simulate. -1 indicates max",
)
@click.option(
    "-m",
    "--networksteps",
    default=-1,
    type=click.IntRange(-1),
    required=False,
    show_default=True,
    help="Maximum number of transmission steps for balancing",
)
@click.option(
    "-i",
    "--iterations",
    default=1000,
    show_default=True,
    type=click.IntRange(1, 4000),
    required=False,
    help="Maximum iterations",
)
@click.option(
    "-p",
    "--popsize",
    default=50,
    show_default=True,
    type=click.IntRange(2, 1000),
    required=False,
    help="Population size",
)
@click.option(
    "-m",
    "--mutation",
    default=None,
    show_default=True,
    type=click.FloatRange(0.0, 2.0),
    required=False,
    help="Mutation factor",
)
@click.option(
    "-d",
    "--dither",
    nargs=2,
    default=(None, None),
    type=click.Tuple([click.FloatRange(0.0, 2.0), click.FloatRange(0.0, 2.0)]),
    required=False,
    help="Mutation factor dither range (overrides --mutation)",
)
@click.option(
    "-r",
    "--recombination",
    default=0.4,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    required=False,
    help="Recombination factor",
)
@click.option(
    "-v",
    "--progress",
    default=1,
    type=click.IntRange(0),
    show_default=True,
    required=False,
    help="Print progress to console",
)
@click.option(
    "-e",
    "--stagnation",
    default=(5, 0.1),
    type=click.Tuple([click.IntRange(0), click.FloatRange(0)]),
    show_default=True,
    required=False,
    help="Stagnation condition: (no. of its, minimum improvement)",
)
@click.option(
    "-f",
    "--fileprint",
    default=1,
    type=click.IntRange(0),
    show_default=True,
    required=False,
    help="Frequency to print to file",
)
def polish(
    scenario: int,
    years: int,
    networksteps: int,
    iterations: int,
    popsize: int,
    mutation: float,
    dither: tuple[float, float],
    recombination: float,
    progress: int,
    stagnation: tuple[int, float],
    fileprint: int,
):
    import numpy as np

    try:
        x0 = np.genfromtxt(f"Results/Optimisation_resultx{scenario}.csv", delimiter=",")
    except FileNotFoundError as e:
        print("No solution found. Run optimisation first.")
        raise e
    param = Parameters(
        s=scenario,
        y=years,
        p=0,
        n=networksteps
    )
    hyperparam = DE_Hyperparameters(
        i=iterations,
        p=popsize,
        m=(
            mutation
            if mutation is not None
            else dither if dither[0] is not None and dither[1] is not None else (0.5, 1.0)
        ),
        r=recombination,
        v=progress,
        s=stagnation,
        f=fileprint,
    )

    from firm.Input import StaticData
    from firm.Optimisation import Polish

    static = StaticData(*param)

    result, time = Polish(x0, static, hyperparam)
    print(result.x)


@click.command
def info():
    console = Console()
    text = Text(
        r"""
 _____ ___ ____  __  __ ____  _
|  ___|_ _|  _ \|  \/  |  _ \| |_   _ ___
| |_   | || |_) | |\/| | |_) | | | | / __|
|  _|  | ||  _ <| |  | |  __/| | |_| \__ \
|_|   |___|_| \_\_|  |_|_|   |_|\__,_|___/
"""
    )
    console.print(text, style="cornflower_blue")
    version_string = "Version 0.0.1"
    console.print(version_string, style="cornflower_blue")


@click.group
def Entry():
    pass


Entry.add_command(benchmark)
Entry.add_command(profile)
Entry.add_command(optimise)
Entry.add_command(polish)
Entry.add_command(statistics)
Entry.add_command(info)
