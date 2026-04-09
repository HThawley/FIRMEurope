"""
An example is given for building a FIRM Model instance and executing a single pass
of the MHMGA recombination operators for inspection and hyperparameter tuning.

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""
import time
import os
import json
import numpy as np
import pandas as pd

from firm_ce.model import Model


def run_recombination_tuning(scenario, model, run_mode):
    print(f"Instantiating recombination inspection for scenario: {scenario.name}")

    if run_mode == "new":
        print(f"Generating new random starting population for scenario {scenario.name}")
        pop_size = model.config.mga_pop_size[0]
        ndim = len(scenario.lower_bounds)
        starting_population = np.random.uniform(
            scenario.lower_bounds,
            scenario.upper_bounds,
            size=(pop_size, ndim)
        )
        objectives = None
        constraints = None
    else:
        config_path = os.path.join(scenario.solution_dir, "mhmga_config.json")
        with open(config_path, "r") as f:
            mhmga_config = json.load(f)
            pop_size = mhmga_config["mga_pop_size"][-1]

        pop_path = os.path.join(scenario.solution_dir, 'latest_population.csv')
        if not os.path.exists(pop_path) and run_mode == "latest":
            pop_path = 'results/temp/latest_population.csv'
        if os.path.exists(pop_path):
            print(f"Loading starting population from {pop_path}")
            starting_population = pd.read_csv(pop_path, header=None)
            objectives = starting_population.iloc[-pop_size:, 0].to_numpy()
            constraints = starting_population.iloc[-pop_size:, 1].to_numpy()
            # fitnesses = starting_population.iloc[-pop_size:, 2]  # we don't need fitnesses
            starting_population = starting_population.iloc[-pop_size:, 3:].to_numpy()
        else:
            raise FileNotFoundError(f"Could not find {pop_path}. Has the solution been run with population saving enabled?")

    print(f"Running recombination operators for scenario {scenario.name}")
    results = model.inspect_recombination(
        scenario_name=scenario.name,
        starting_population=starting_population,
        objectives=objectives,
        constraints=constraints,
        evaluate_offspring=True,
        # Override specific hyperparameters for tuning tests:
        pop_size=pop_size,
        # mutation_prob=1.0,
        # crossover_prob=0.0
    )

    print(f"\n--- Results for {scenario.name} ---")
    print(f"Offspring shape: {results['offspring_points'].shape}")

    if results.get("offspring_objectives") is not None:
        print(f"Mean Objective: {np.mean(results['offspring_objectives']):.4f}")
        if results.get("offspring_violations") is not None:
            print(f"Mean Violations: {np.mean(results['offspring_violations']):.4f}")
    else:
        print("Offspring evaluation was bypassed.")

    return None


if __name__ == "__main__":

    RUN_MODE = "latest"

    start_time = time.time()
    model = Model(results_mode=RUN_MODE)
    model_build_time = time.time()
    print(f"Model build time: {model_build_time - start_time:.4f} seconds")

    for name in ("base",):
        scenario = model.scenarios[name]
        run_recombination_tuning(scenario, model, RUN_MODE)
