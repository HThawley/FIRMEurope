# type: ignore
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firm_ce.model import Model
from firm_ce.analysis.statistics import Statistics
from firm_ce.analysis.validate import Validation, ValidationTensor
from firm_ce.analysis.display import Display, MetricSpec, is_rechargeable_storage
from firm_ce.analysis.space_explorer import SpaceExplorer
from firm_ce.common.typing import npfloat


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

    for fname, start_col in (("mga_alternatives.csv", 3), ("mga-noptima.csv", 5)):
        fpath = os.path.join(mga_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if len(df) > 0:
                return df.iloc[:, start_col:].to_numpy().astype(npfloat)

    evo_path = os.path.join(mga_dir, "mga-evolution.csv")
    if os.path.exists(evo_path):
        df = pd.read_csv(evo_path)
        df = df[df["iter"] == df["iter"].max()]
        return df.iloc[:, 6:].to_numpy().astype(npfloat)

    raise FileNotFoundError(f"Could not find MHMGA alternatives in '{mga_dir}'.")


def load_scenario(scenario, run_mode: str, x_loc: str | None = None, x0_fallback: bool = True) -> None:
    scenario.load_datafiles()

    if scenario.config.type == "single_time":
        x = resolve_single_solution_x(scenario, run_mode, x_loc=x_loc, x0_fallback=x0_fallback)
        scenario.build_and_evaluate_solution(x)
    elif scenario.config.type.startswith("mhmga"):
        xs = resolve_mhmga_noptima_xs(scenario)
        scenario.build_and_evaluate_noptima(xs)


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


def display_graphs(scenario) -> None:
    # --- Tier 1: Simulated Solution Display ---
    if scenario.config.type == "single_time":
        print(f"Generating single_time plots for '{scenario.name}'")
        disp = Display(scenario, scenario.config, solution=scenario.solution)
        disp.set_annotation_format(lambda v: disp.format_sigfigs(v, 2))

        disp.plot_summary("system_overview_with_losses", chart_type="stacked_bar", normalize=True)
        disp.plot_summary("energy_balance", chart_type="stacked_bar", normalize=False)
        disp.plot_summary("power_capacity", chart_type="treemap")
        disp.plot_map("energy")
        scenario.display = disp

    elif scenario.config.type.startswith("mhmga"):
        print(f"Generating MHMGA plots for '{scenario.name}'")
        disp = Display(scenario, scenario.config, noptima=scenario.noptima)
        disp.set_annotation_format(lambda v: disp.format_sigfigs(v, 2))

        # 1. Tier 1: Compare specific simulated MGA niche alternatives
        gen_mix_spec = disp.presets["energy_balance"][0]
        disp.plot_summary(gen_mix_spec, solutions=[0, 1, 2, 3, 4], normalize=False)
        disp.plot_summary("power_capacity", solutions=2, ref_solution=0, mode="delta", normalize=False)

        custom_map_spec = MetricSpec(
            title="New Storage Energy Capacity",
            unit="GWh",
            assets=("storages",),
            metric="energy_capacity",
            build="new_build",
            asset_filter=is_rechargeable_storage,
            group_by="subtech",
        )
        disp.plot_map(node_spec=custom_map_spec, line_spec=None, solutions=[0, 1, 2, 3])
        scenario.display = disp

        # 2. Tier 2: Explore full 10M+ historical MGA ensemble in Relative or Absolute space
        explorer = SpaceExplorer(scenario, max_relaxation=1.15, precision=1, rel_norm="eligible")
        scenario.explorer = explorer

        # Multi-panel surface (min, median, max rechargeable storage energy across solar vs wind)
        explorer.plot_surface(
            x="solar",
            y="wind",
            z="rechargable_e",
            aggs=("min", "median", "max"),
            space="rel",
            threshold=1.15,
            isolines=[1.02, 1.05, 1.10],
        )

        # Pairwise trade-off grid with Pareto frontier
        explorer.plot_pairgrid(
            vars=["solar", "wind", "bess_e", "phes_e", "nuke_all"],
            z="cost",
            agg="min",
            space="rel",
            threshold=1.05,
            show_pareto=True,
        )

        # Capacity slack vs cost relaxation ribbons
        explorer.plot_slack_ribbons(
            vars=["solar", "wind", "bess", "phes", "trans"],
            max_threshold=1.15,
            space="rel",
        )

        # Near-optimal capacity distributions
        explorer.plot_distributions(
            vars=["solar", "wind", "nuke_all", "bio", "ccgt", "bess", "phes", "trans"],
            threshold=1.05,
            space="rel",
        )


if __name__ == "__main__":
    RUN_MODE = "results/firmeur_10yr_20260922_093328"

    start_time = time.time()
    model = Model(model_location=RUN_MODE)
    print(f"Model build time: {time.time() - start_time:.4f} seconds")

    for name in ("7percent",):
        scenario = model.scenarios[name]
        load_scenario(scenario, RUN_MODE)
        display_graphs(scenario)
        plt.show()
