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
from firm_ce.analysis.validate import Validation
from firm_ce.analysis.display import Display

RUN_MODE = "new"

start_time = time.time()
model = Model(results_mode=RUN_MODE)
model_build_time = time.time()
print(f"Model build time: {model_build_time - start_time:.4f} seconds")

def run_statistics(scenario):
    scenario.solution_dir = scenario.solution_dir.replace("full", "simple")
    scenario.load_datafiles(model.datafile_filenames_dict, model.data_directory)

    if RUN_MODE == "new":
        if scenario.x0.size == 0:
            print(f"skipping {scenario.name} as no initial guess provided")
            scenario.unload_datafiles()
            return
        x = scenario.x0
        scenario.create_solution_directory()

    else:
        x_csv = os.path.join(scenario.solution_dir, "x.csv")
        if os.path.exists(x_csv):
            x = pd.read_csv(x_csv, header=None).to_numpy().flatten()
        else:
            scenario.unload_datafiles()
            raise FileNotFoundError("Could not find x.csv. Has the solution been run?")

    print(f"Instantiating statistics for scenario: {scenario.name}")
    scenario.statistics = Statistics(
        x,
        scenario.static,
        scenario.fleet,
        scenario.network,
        scenario.results_dir,
        scenario.name,
        model.config.balancing_type,
        model.config.fixed_costs_threshold,
        False,
    )
    print(f"Generating statistics for scenario {scenario.name}")
    scenario.statistics.generate_result_files()
    print(f"Writing statistics results for scenario {scenario.name}")
    scenario.statistics.write_results()

    print(f"Validating solution {scenario.name}")
    scenario.validation = Validation(scenario)
    scenario.validation.validate(verbose=True)
    scenario.validation.dump_logs()

    print(f"Generating plots {scenario.name}")
    display = Display(scenario, model.config)
    display.plot_energy_mix(mode="atlas", chart_type="bar", indices=[0, 1, 2])
    display.plot_energy_mix(curtailment=False, alternative=2)
    display.plot_energy_mix(curtailment=True)
    display.plot_power_capacity()
    display.plot_power_capacity(build="existing")
    display.plot_power_capacity(build="new_build")

    scenario.unload_datafiles()
    return None


for scenario in model.scenarios.values():
    run_statistics(scenario)

    
