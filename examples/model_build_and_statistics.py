# type: ignore
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mga.postprocessing import selection

from firm_ce.model import Model
from firm_ce.analysis.statistics import Statistics
from firm_ce.analysis.validate import Validation, ValidationTensor
from firm_ce.analysis.display import Display, MetricSpec, is_rechargeable_storage
from firm_ce.analysis.space_explorer import SpaceExplorer
from firm_ce.common.typing import npfloat
from firm_ce.analysis.cost_uncertainty import MCConfig, run_monte_carlo


def resolve_single_solution_x(scenario, run_mode: str, x_loc: str | None = None, x0_fallback: bool = True) -> np.ndarray:
    """Resolves the decision vector `x` for a single_time scenario using a clean fallback chain."""
    sol_dir = scenario.solution_dir

    if run_mode == "new":
        if scenario.x0.size == 0:
            raise ValueError(f"No initial guess (x0) provided for scenario '{scenario.name}'.")
        scenario.create_solution_directory()
        return scenario.x0.astype(npfloat)

    if x_loc == "latest_population":
        npy_path = os.path.join(sol_dir, "latest_population.npy")
        csv_path = os.path.join(sol_dir, "latest_population.csv")
        if os.path.exists(npy_path):
            return np.load(npy_path)[0, 3:].astype(npfloat)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path, header=None, nrows=1).to_numpy()[0, 3:].astype(npfloat)

    curr_opt = os.path.join(sol_dir, "current_optimum.csv")
    if os.path.exists(curr_opt):
        return pd.read_csv(curr_opt, header=None).to_numpy().flatten()[2:].astype(npfloat)

    for stat_name, is_rel in (("x_rel", True), ("x_abs", False)):
        stat_path = os.path.join(sol_dir, "statistics", f"{stat_name}.csv")
        if os.path.exists(stat_path):
            x = pd.read_csv(stat_path, header=None).to_numpy().flatten().astype(npfloat)
            if scenario.config.parameterisation == "relative" and not is_rel:
                return scenario.convert_x_to_rel(x)
            if scenario.config.parameterisation == "absolute" and is_rel:
                return scenario.convert_x_to_abs(x)
            return x

    if x0_fallback and scenario.x0.size > 0:
        return scenario.x0.astype(npfloat)

    raise FileNotFoundError(f"Could not locate any solved decision vector in '{sol_dir}'.")


def resolve_mhmga_noptima_xs(scenario) -> np.ndarray:
    """Resolves the near-optimal alternative vectors `xs` from mga_logs."""
    mga_dir = os.path.join(scenario.solution_dir, "mga_logs")
    ndim = len(scenario.lower_bounds)

    for fname in ("mga_alternatives.csv", "mga-noptima.csv"):
        fpath = os.path.join(mga_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if len(df) > 0:
                xs = df.iloc[:, -ndim:].to_numpy(dtype=npfloat)
                valid_mask = np.all(np.isfinite(xs), axis=1)
                if np.any(valid_mask):
                    return xs[valid_mask]

    evo_path = os.path.join(mga_dir, "mga-evolution.csv")
    if os.path.exists(evo_path):
        df = pd.read_csv(evo_path)
        df = df[df["iter"] == df["iter"].max()]
        xs = df.iloc[:, -ndim:].to_numpy(dtype=npfloat)
        valid_mask = np.all(np.isfinite(xs), axis=1)
        if np.any(valid_mask):
            return xs[valid_mask]

    raise FileNotFoundError(f"Could not find valid MHMGA alternatives in '{mga_dir}'.")


def get_x(
    scenario,
    run_mode: str,
    x_loc: str | None = None,
    x0_fallback: bool = True,
):
    if not scenario.data_status:
        scenario.load_datafiles()

    if scenario.config.type == "single_time":
        xs = resolve_single_solution_x(scenario, run_mode, x_loc=x_loc, x0_fallback=x0_fallback)
    elif scenario.config.type.startswith("mhmga"):
        xs = resolve_mhmga_noptima_xs(scenario)

    return xs


def build_and_evaluate_solutions(scenario, xs, mhmga_indices=None):
    if not scenario.data_status:
        scenario.load_datafiles()

    if scenario.config.type == "single_time":
        scenario.build_and_evaluate_solution(x)
    elif scenario.config.type.startswith("mhmga"):
        scenario.build_and_evaluate_noptima(xs[mhmga_indices])


def load_scenario(
    scenario,
    run_mode: str,
    x_loc: str | None = None,
    x0_fallback: bool = True,
    mhmga_indices: list | None = None,
    xs: pd.DataFrame | None = None,
) -> None:
    if not scenario.data_status:
        scenario.load_datafiles()

    if xs is None:
        xs = get_x(scenario, run_mode, x_loc, x0_fallback)

    build_and_evaluate_solutions(scenario, xs, mhmga_indices)

    return xs


def get_diverse_indices(scenario, xs, n, strategy="max-min", fitness_method="auto"):
    if fitness_method == "auto":
        fitness_method = scenario.config.mga_fitness[0].lower()

    scaled_xs = selection.project_space(xs, scenario.projection_matrix)
    diverse_idx = selection.get_diverse_indices(
        scaled_xs,
        n=n,
        strategy=strategy,
        fitness_method=fitness_method
    )

    return list(diverse_idx)


def validate_scenario(scenario, write: bool = True) -> None:
    print(f"Validating solution {scenario.name}")
    if scenario.config.backend == "tensor":
        scenario.validation = ValidationTensor(scenario.solutionTensor, scenario.solution_dir, scenario)
    else:
        scenario.validation = Validation(scenario.solution, scenario.solution_dir)

    scenario.validation.validate(verbose=True)
    if write:
        scenario.validation.dump_logs()


def run_statistics(scenario, write: bool = True) -> None:
    scenario.statistics = Statistics(scenario)
    print(f"Generating statistics for scenario {scenario.name}")
    scenario.statistics.generate_result_files(write=write, delete=True)


def run_cost_monte_carlo(
    scenario,
    n_samples: int = 10_000,
    rel_halfwidth: float = 0.10,
    perturb_mode: str = "group",  # "group" = one multiplier per cost parameter; "element" = per node/line
    mhmga_indices: np.ndarray = None,
):
    """Monte-Carlo over cost inputs for the pre-evaluated near-optimal alternatives."""
    if not scenario.config.type.startswith("mhmga"):
        raise ValueError(f"Cost Monte-Carlo needs an mhmga scenario, got '{scenario.config.type}'.")
    if scenario.config.backend != "tensor":
        raise NotImplementedError("Cost Monte-Carlo is only implemented for backend='tensor'.")

    cfg = MCConfig(n_samples=n_samples, rel_halfwidth=rel_halfwidth, perturb_mode=perturb_mode)

    # Push the already-evaluated noptima to the analysis engine
    summary = run_monte_carlo(scenario, mhmga_indices, cfg=cfg)

    print(summary.head(10).to_string(index=False))
    return summary


def display_graphs(scenario, mhmga_indices: list = None) -> None:
    # --- Tier 1: Simulated Solution Display ---
    if scenario.config.type == "single_time":
        print(f"Generating single_time plots for '{scenario.name}'")
        scenario.display = Display(scenario, scenario.config, solution=scenario.solution)
        scenario.display.set_annotation_format(lambda v: scenario.display.format_sigfigs(v, 2))

        scenario.display.plot_summary("system_overview_with_losses", chart_type="stacked_bar", normalize=True)
        scenario.display.plot_summary("energy_balance", chart_type="stacked_bar", normalize=False)
        scenario.display.plot_summary("power_capacity", chart_type="treemap")
        scenario.display.plot_map("energy")

    elif scenario.config.type.startswith("mhmga"):
        if mhmga_indices is None:
            mhmga_indices = list(range(len(scenario.noptima)))
        print(f"Generating MHMGA plots for '{scenario.name}'")
        scenario.display = Display(scenario, scenario.config, noptima=scenario.noptima)
        scenario.display.set_annotation_format(lambda v: scenario.display.format_sigfigs(v, 2))

        # 1. Tier 1: Compare specific simulated MGA niche alternatives
        gen_mix_spec = scenario.display.presets["energy_balance"][0]
        scenario.display.plot_summary(gen_mix_spec, solutions=mhmga_indices, normalize=False)
        scenario.display.plot_summary("power_capacity", solutions=2, ref_solution=0, mode="delta", normalize=False)

        custom_map_spec = MetricSpec(
            title="New Generation Capacity",
            unit="GWh",
            assets=("generators",),
            metric="power_capacity",
            build="new_build",
            group_by="subtech",
        )
        scenario.display.plot_map(node_spec=custom_map_spec, line_spec=None, solutions=mhmga_indices)

        # 2. Tier 2: Explore full 10M+ historical MGA ensemble in Relative or Absolute space
        scenario.explorer = SpaceExplorer(scenario, max_relaxation=1.15, precision=1, rel_norm="eligible")

        # Multi-panel surface (min, median, max rechargeable storage energy across solar vs wind)
        scenario.explorer.plot_surface(
            x="solar",
            y="wind",
            z="rechargable_e",
            aggs=("min", "median", "max"),
            space="abs",
            threshold=1.15,
            isolines=[1.02, 1.05, 1.10],
        )

        # Pairwise trade-off grid with Pareto frontier
        scenario.explorer.plot_pairgrid(
            vars=["solar", "wind", "bess_e", "phes_e", "nuke_all"],
            z="cost",
            agg="min",
            space="abs",
            threshold=1.1,
            show_pareto=True,
        )

        # Capacity slack vs cost relaxation ribbons
        scenario.explorer.plot_slack_ribbons(
            vars=["solar", "wind", "bess", "phes", "trans"],
            max_threshold=1.1,
            space="abs",
        )

        # Near-optimal capacity distributions
        scenario.explorer.plot_distributions(
            vars=["solar", "wind", "nuke_all", "bio", "ccgt", "bess", "phes", "trans"],
            threshold=1.5,
            space="abs",
        )


if __name__ == "__main__":
    RUN_MODE = "results/firmeur_7pc_explor_20261016_121741"

    start_time = time.time()
    model = Model(model_location=RUN_MODE)
    print(f"Model build time: {time.time() - start_time:.4f} seconds")

    for name in ("7percent",):
        scenario = model.scenarios[name]
        scenario.load_datafiles()

        xs = get_x(scenario, RUN_MODE)

        indices = None
        if scenario.config.type.startswith("mhmga"):
            indices = get_diverse_indices(scenario, xs, 4)

        build_and_evaluate_solutions(scenario, xs=xs, mhmga_indices=indices)

        run_cost_monte_carlo(scenario, n_samples=10_000)
        display_graphs(scenario)
        plt.show()
