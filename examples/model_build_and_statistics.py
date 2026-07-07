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

from firm_ce.model import Model
from firm_ce.analysis.statistics import Statistics
from firm_ce.analysis.validate import Validation, ValidationTensor
from firm_ce.analysis.display import Display
from firm_ce.common.typing import npfloat


def try_read(filename):
    try:
        full_path = os.path.join(scenario.solution_dir, "statistics", f"{filename}.csv")
        x = pd.read_csv(full_path, header=None).to_numpy().flatten()
    except FileNotFoundError as e:
        return e
    return x


def run_statistics(scenario, run_mode, x0_fallback=True):
    scenario.load_datafiles()

    if run_mode == "new":
        if scenario.x0.size == 0:
            print(f"skipping {scenario.name} as no initial guess provided")
            scenario.unload_datafiles()
            return
        x = scenario.x0
        scenario.create_solution_directory()

    else:
        x_rel = try_read("x_rel")
        x_abs = try_read("x_abs")
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

    print(f"Validating solution {scenario.name}")
    if scenario.solutionTensor is not None:
        scenario.validation = ValidationTensor(scenario.solutionTensor, scenario.solution_dir, scenario)
        scenario.validation.validate(verbose=True)
        scenario.validation.dump_logs()
    else:
        scenario.validation = Validation(scenario.solution, scenario.solution_dir)
        scenario.validation.validate(verbose=True)
        scenario.validation.dump_logs()

    scenario.statistics = Statistics(scenario)

    print(f"Generating statistics for scenario {scenario.name}")
    scenario.statistics.generate_result_files(write=True, delete=True)
    print(f"Writing statistics results for scenario {scenario.name}")
    # scenario.statistics.write_results()

    # raise KeyboardInterrupt

    print(f"Generating plots {scenario.name}")
    scenario.display = Display(scenario, model.config, solution=scenario.statistics.solution)

    if model.config.type == "mhmga":
        scenario.display.plot_energy_mix(atlas=True, chart_type="pie", indices=[0, 1, 2, 3])
        scenario.display.plot_power_capacity(atlas=True, chart_type="pie", indices=[0, 1, 2, 3])
        scenario.display.plot_energy_mix(atlas=True, delta=True, chart_type="bar", indices=[0, 1, 2, 3])
        # scenario.display.plot_energy_mix(curtailment=False, alternative=2)
        data = []
        for s in scenario.display.noptima:
            col = dict(zip(scenario.projection_groups.keys(), s.x @ scenario.projection_matrix))
            data.append(col)
        df = pd.DataFrame(data)
        scenario.noptima_summary = df
        print(df)

    # TODO: Energy mix based on consumption / generation
    scenario.display.plot_energy_mix()
    scenario.display.plot_power_capacity()
    # scenario.display.plot_power_capacity(build="existing")
    # scenario.display.plot_power_capacity(build="new_build")

    data = []

    # raise KeyboardInterrupt

    # scenario.unload_datafiles()
    return None


if __name__ == "__main__":

    # RUN_MODE = "latest"
    RUN_MODE = "results/firmeur_tensor_20260703_211927"

    start_time = time.time()
    model = Model(model_location=RUN_MODE)
    model_build_time = time.time()
    print(f"Model build time: {model_build_time - start_time:.4f} seconds")

    for name in ("7percent",):
        scenario = model.scenarios[name]
        run_statistics(scenario, RUN_MODE)
