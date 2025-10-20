"""
An example is given for building a FIRM Model instance, generating the Statistics associated with the initial guess for each scenario, and
saving those results. All of the result files are saved to the `results` folder.

The Model object is built using the default `inputs/config` and `inputs/data` files. Statistics are only generated for scenarios with an
initial guess provided in `initial_guess.csv`.

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""

import time

from firm_ce.model import Model
from firm_ce.optimisation.statistics import Statistics

start_time = time.time()
model = Model()
model_build_time = time.time()
print(f"Model build time: {model_build_time - start_time:.4f} seconds")

for scenario in model.scenarios.values():
    if scenario.x0.size == 0:
        continue

    scenario.load_datafiles(model.datafile_filenames_dict, model.data_directory)
    scenario.statistics = Statistics(
        scenario.x0,
        scenario.static,
        scenario.fleet,
        scenario.network,
        scenario.results_dir,
        scenario.name,
        model.config.balancing_type,
        model.config.fixed_costs_threshold,
        False,
    )
    scenario.statistics.generate_result_files()
    scenario.statistics.write_results()
    scenario.unload_datafiles()
