"""
An example is given for building a FIRM Model instance, generating the Statistics associated with the
initial guess for each scenario, and saving those results. All of the result files are saved to the
`results` folder.

The Model object is built using the default `inputs/config` and `inputs/data` files. Statistics are
only generated for scenarios with an initial guess provided in `initial_guess.csv`.

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from firm_ce.model import Model
from firm_ce.analysis.statistics import Statistics
from firm_ce.analysis.validate import Validation, ValidationTensor
from firm_ce.analysis.display import Display, MetricSpec, is_rechargeable_storage
from firm_ce.analysis.space_display import SpacePlotter
from firm_ce.common.typing import npfloat


def try_read_x(scenario):
    try:
        full_path = os.path.join(scenario.solution_dir, "current_optimum.csv")
        x = pd.read_csv(full_path, header=None).to_numpy().flatten().astype(npfloat)[2:]
    except FileNotFoundError as e:
        return e
    return x


def try_read_x_from_stat(scenario, filename):
    try:
        full_path = os.path.join(scenario.solution_dir, "statistics", f"{filename}.csv")
        x = pd.read_csv(full_path, header=None).to_numpy().flatten().astype(npfloat)
    except FileNotFoundError as e:
        return e
    return x


def try_read_noptima(scenario):
    try:
        full_path = os.path.join(scenario.solution_dir, "mga_logs", "mga-noptima.csv")
        xs = pd.read_csv(full_path).iloc[:, 5:].to_numpy().astype(npfloat)
        assert xs.size > 0
    except FileNotFoundError as e:
        return e
    except AssertionError as e:
        return e
    return xs


def try_read_evolution(scenario):
    """mga-noptima.csv generated at end of optimisation. This extracts intermediate noptima"""
    try:
        full_path = os.path.join(scenario.solution_dir, "mga_logs", "mga-evolution.csv")
        xs = pd.read_csv(full_path)
        xs = xs[xs["iter"] == xs["iter"].max()]
        xs = xs.iloc[:, 6:].to_numpy().astype(npfloat)
    except FileNotFoundError as e:
        return e
    return xs


def load_scenario(scenario, run_mode, x_loc=None, x0_fallback=True):
    scenario.load_datafiles()

    if scenario.config.type == "single_time":
        if run_mode == "new":
            if scenario.x0.size == 0:
                print(f"skipping {scenario.name} as no initial guess provided")
                scenario.unload_datafiles()
                return
            x = scenario.x0
            scenario.create_solution_directory()

        elif x_loc == "latest_population.csv":
            # TODO: more elegant
            population = pd.read_csv(os.path.join(scenario.solution_dir, "latest_population.csv"), header=None, nrows=1).to_numpy()
            x = population[0, 3:].astype(npfloat)

        else:
            x = try_read_x(scenario)

            if isinstance(x, FileNotFoundError):
                x_rel = try_read_x_from_stat(scenario, "x_rel")
                x_abs = try_read_x_from_stat(scenario, "x_abs")
                x_rel_failure = isinstance(x_rel, FileNotFoundError)
                x_abs_failure = isinstance(x_abs, FileNotFoundError)

                if x_rel_failure and x_abs_failure:
                    if x0_fallback:
                        x = scenario.x0
                    else:
                        scenario.unload_datafiles()
                        raise FileNotFoundError("Could not find x csv. Has the solution been run?")
                elif scenario.config.parameterisation == "relative":
                    x = scenario.convert_x_to_rel(x_abs) if x_rel_failure else x_rel
                elif scenario.config.parameterisation == "absolute":
                    x = scenario.convert_x_to_abs(x_rel) if x_abs_failure else x_abs
                else:
                    scenario.unload_datafiles()
                    raise ValueError(f"Unknown parameterisation type: {scenario.config.parameterisation}")

        x = x.astype(npfloat)
        scenario.build_and_evaluate_solution(x)

    elif scenario.config.type == "mhmga":
        xs = try_read_noptima(scenario)
        if isinstance(xs, FileNotFoundError):
            xs = try_read_evolution(scenario)
        elif isinstance(xs, AssertionError):
            print("WARNING: noptima file found but empty. Trying evolution file")
            xs = try_read_evolution(scenario)
        if isinstance(xs, FileNotFoundError):
            raise xs
        # TODO: verify absolute/relative parameterisation -> fail noisily
        scenario.build_and_evaluate_noptima(xs)


def unload_scenario(scenario):
    scenario.unload_datafiles()


def validate_scenario(scenario, write=True):
    print(f"Validating solution {scenario.name}")
    if scenario.config.backend == "tensor":
        scenario.validation = ValidationTensor(scenario.solutionTensor, scenario.solution_dir, scenario)
    else:
        scenario.validation = Validation(scenario.solution, scenario.solution_dir)

    scenario.validation.validate(verbose=True)

    if write:
        scenario.validation.dump_logs()


def run_statistics(scenario, write=True):
    scenario.statistics = Statistics(scenario)

    print(f"Generating statistics for scenario {scenario.name}")
    scenario.statistics.generate_result_files(write=True, delete=True)


def display_graphs(scenario):
    if model.config.type == "single_time":
        print(f"Generating single_time plots {scenario.name}")
        scenario.display = Display(scenario, model.config, solution=scenario.solution)
        scenario.display.set_annotation_format(lambda v: scenario.display.format_sigfigs(v, 2))

        scenario.display.plot_summary("system_overview_with_losses", chart_type="stacked_bar", normalize=True)

        scenario.display.plot_summary("energy_balance", chart_type="stacked_bar", normalize=False)
        scenario.display.plot_summary("energy_balance", chart_type="treemap", normalize=False)
        scenario.display.plot_summary("power_capacity", chart_type="treemap")
        scenario.display.plot_summary("power_capacity", chart_type="stacked_bar", normalize=False)

        # Custom Ad-Hoc Map Spec (e.g., plot ONLY new-build rechargeable storage power capacity by sub-tech)
        # custom_map_spec = MetricSpec(
        #     title="New Storage Energy Capacity",
        #     unit="GWh",
        #     assets=("storages",),
        #     metric="power_capacity",
        #     build="new_build",
        #     asset_filter=is_rechargeable_storage,
        #     group_by="subtech",
        # )
        # scenario.display.plot_map(node_spec=custom_map_spec, line_spec=None)
        scenario.display.plot_map("energy")

    if model.config.type == "mhmga":
        print(f"Generating mhmga plots {scenario.name}")
        scenario.display = Display(scenario, model.config, noptima=scenario.noptima)
        scenario.display.set_annotation_format(lambda v: scenario.display.format_sigfigs(v, 2))

        # 1. Initialization
        # Reads the full historical population (population.csv) and aggregates variables via the scenario's tech_map
        plotter = SpacePlotter(scenario, precision=1)

        # 2. Bivariate Footprint with Isolines
        # Shows the flat density envelope of solutions under a 1.05 threshold, overlaid with 
        # contour isolines mapping the physical expansion limits at 1.01, 1.03, and 1.05 LCOE thresholds.
        fig, ax = plt.subplots(figsize=(8, 6))
        plotter.plot_bivariate_footprint(
            x_var="Utility Solar",
            y_var="Onshore Wind",
            threshold=1.05,
            isoline_thresholds=[1.01, 1.03, 1.05],
            ax=ax
        )

        # 3. Threshold Area Expansion
        # Replicates compare_near_optimal_area.py to show how available capacity slack widens 
        # as the optimality threshold relaxes from 1.0 to 1.10.
        fig = plotter.plot_threshold_areas(
            techs=["Utility Solar", "Onshore Wind", "Battery Power (GW)"],
            max_threshold=1.10,
            steps=20
        )

        # 4. Boxen Distributions
        # Displays the statistical shape of technology deployment purely for configurations 
        # that successfully sit within the 1.02 near-optimal valley.
        fig, ax = plt.subplots(figsize=(10, 5))
        plotter.plot_boxen_distributions(threshold=1.02, ax=ax)

        # 5. Structural Pairwise Matrix
        # Compares purely structural trade-offs (excluding LCOE) inside the 1.03 threshold,
        # identifying the spatial frontiers where technologies directly cannibalize each other.
        fig = plotter.plot_pairwise_structure(
            structural_vars=["Utility Solar", "Onshore Wind", "Battery Power (GW)"],
            threshold=1.03
        )

        scenario.display.plot_summary("energy_balance", solutions=0, chart_type="stacked_bar", normalize=False)
        scenario.display.plot_summary("power_capacity", solutions=0, chart_type="treemap")

        # Multi-Solution MGA Comparison on a Single Axis
        # Passing a single MetricSpec and multiple solutions compares alternatives side-by-side:
        gen_mix_spec = scenario.display.presets["energy_balance"][0]
        scenario.display.plot_summary(gen_mix_spec, solutions=[0, 1, 2, 3, 4], normalize=False)

        # 3. Delta Summary Bars (Alt 2 minus Alt 0 across all power capacity categories)
        scenario.display.plot_summary("power_capacity", solutions=2, ref_solution=0, mode="delta", normalize=False)

        # 4. Custom Ad-Hoc Map Spec (e.g., plot ONLY new-build rechargeable storage energy capacity by sub-tech)
        custom_map_spec = MetricSpec(
            title="New Storage Energy Capacity",
            unit="GWh",
            assets=("storages",),
            metric="energy_capacity",
            build="new_build",
            asset_filter=is_rechargeable_storage,
            group_by="subtech",
        )
        scenario.display.plot_map(node_spec=custom_map_spec, line_spec=None, solutions=[0, 1, 2, 3])

        data = []
        for s in scenario.display.noptima:
            col = dict(zip(scenario.projection_groups.keys(), s.x @ scenario.projection_matrix))
            data.append(col)
        df = pd.DataFrame(data)
        scenario.noptima_summary = df
        print(df)

    return None


if __name__ == "__main__":

    # RUN_MODE = "latest"
    RUN_MODE = "results/firmeur_10yr_20260922_093328 (copy)"

    start_time = time.time()
    model = Model(model_location=RUN_MODE)
    model_build_time = time.time()
    print(f"Model build time: {model_build_time - start_time:.4f} seconds")

    for name in ("7percent",):
        scenario = model.scenarios[name]
        # model.config.type = 'single_time'
        load_scenario(scenario, RUN_MODE)
        # validate_scenario(scenario)
        # run_statistics(scenario)
        display_graphs(scenario)

        # unload_scenario(scenario)
