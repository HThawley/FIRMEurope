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

from firm_ce.model import Model
from firm_ce.common.constants import FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import njit, prange
from firm_ce.common.typing import npfloat
from firm_ce.backend.scalar.single_time import Solution, evaluate_from_details


def _try_path(folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        print(f"Using '{filename}' in {folder}.")
        return path
    print(f"No '{filename}' found in {folder}.")
    return False


@njit(parallel=True, fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def evaluate_points(points, details, static, fleet, network, balancing_type, fixed_costs_threshold):
    costs = np.zeros(points.shape[0], dtype=npfloat)
    for i in prange(points.shape[0]):
        solution = Solution(
            points[i],
            static,
            fleet,
            network,
            balancing_type,
            fixed_costs_threshold,
        )
        evaluate_from_details(solution, details[i])
        costs[i] = solution.lcoe
    return costs


def approx_costs(scenario, config):
    scenario.load_datafiles(model.datafile_filenames_dict, model.data_directory)

    pop_temp, ops_temp = False, False
    pop_path = _try_path(scenario.solution_dir, "latest_population.csv")
    if not pop_path:
        pop_path = _try_path("results/temp", "latest_population.csv")
        pop_temp = True
    ops_path = _try_path(scenario.solution_dir, "latest_details.csv")
    if not ops_path:
        ops_path = _try_path("results/temp", "latest_details.csv")
        ops_temp = True
    if not pop_path or not ops_path:
        scenario.unload_datafiles()
        raise FileNotFoundError(f"Did not find all required files. Found 'latest_population.csv'={bool(pop_path)}."
                                f"Found 'latest_details.csv'={bool(ops_path)}.")
    if pop_temp != ops_temp:
        print("Warning: found only one of 'latest_population.csv' and 'latest_details.csv'"
              f" in {scenario.solution_dir} and the other in temp.")

    points = pd.read_csv(pop_path, header=None).to_numpy().astype(npfloat)
    og_costs, penalties, fitness = points[:, 0], points[:, 1], points[:, 2]
    points = points[:, 3:]
    details = pd.read_csv(ops_path, header=None).to_numpy().astype(npfloat)

    ca_costs = evaluate_points(
        points,
        details,
        scenario.static,
        scenario.fleet,
        scenario.network,
        config.balancing_type,
        config.fixed_costs_threshold,
    )

    # scenario.unload_datafiles()
    return og_costs, penalties, fitness, ca_costs


if __name__ == "__main__":

    RUN_MODE = "latest"

    start_time = time.time()
    model = Model(model_location=RUN_MODE)
    model_build_time = time.time()
    print(f"Model build time: {model_build_time - start_time:.4f} seconds")

    for name in ("test",):
        scenario = model.scenarios[name]
        og_costs, penalties, fitness, ca_costs = approx_costs(scenario, model.config)
