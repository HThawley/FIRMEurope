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
import matplotlib.pyplot as plt

from firm_ce.model import Model

def analyze_and_plot_results(results: dict, output_dir: str, scenario_name: str, save_plot: bool = False):
    """Generates a text summary and a 2x2 histogram plot for parent vs offspring."""
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"recombination_summary_{scenario_name}.txt")
    
    def write_stats(f, prefix: str, obj: np.ndarray, viol: np.ndarray):
        if obj is None:
            f.write(f"--- {prefix} ---\nEvaluation bypassed.\n\n")
            return
            
        is_feas = viol == 0 if viol is not None else np.ones(len(obj), dtype=bool)
        n_feas = np.sum(is_feas)
        n_infeas = len(obj) - n_feas
        
        f.write(f"--- {prefix} ---\n")
        f.write(f"Total: {len(obj)} | Feasible: {n_feas} | Infeasible: {n_infeas}\n")
        
        if n_feas > 0:
            feas_obj = obj[is_feas]
            f.write(f"Feasible Objective   -> Mean: {np.mean(feas_obj):.4f} | Min: {np.min(feas_obj):.4f} | Max: {np.max(feas_obj):.4f}\n")
        else:
            f.write("Feasible Objective   -> N/A (None)\n")
            
        if n_infeas > 0:
            infeas_viol = viol[~is_feas]
            f.write(f"Infeasible Violation -> Mean: {np.mean(infeas_viol):.4f} | Min: {np.min(infeas_viol):.4f} | Max: {np.max(infeas_viol):.4f}\n")
        else:
            f.write("Infeasible Violation -> N/A (None)\n")
        f.write("\n")

    # Write Text Summary
    with open(txt_path, 'w') as f:
        write_stats(f, "PARENTS", results.get("parent_objectives"), results.get("parent_violations"))
        write_stats(f, "OFFSPRING", results.get("offspring_objectives"), results.get("offspring_violations"))
    
    print(f"Summary written to: {txt_path}")

    # Plotting
    if results.get("offspring_objectives") is None:
        return # Skip plotting if evaluation was bypassed
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Recombination Metrics: {scenario_name}", fontsize=14)

    # Helper for histograms
    def plot_hist(ax, data, title, color):
        if data is not None and len(data) > 0:
            ax.hist(data, bins=15, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plot_hist(axs[0, 0], results.get("parent_objectives"), "Parent Objectives", "skyblue")
    plot_hist(axs[0, 1], results.get("parent_violations"), "Parent Violations", "lightcoral")
    plot_hist(axs[1, 0], results.get("offspring_objectives"), "Offspring Objectives", "steelblue")
    plot_hist(axs[1, 1], results.get("offspring_violations"), "Offspring Violations", "firebrick")

    plt.tight_layout()
    
    if save_plot:
        plot_path = os.path.join(output_dir, f"recombination_hist_{scenario_name}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
    
    plt.show()


def run_recombination_and_inspect(scenario, model, run_mode):
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

    return results


if __name__ == "__main__":

    RUN_MODE = "latest"

    start_time = time.time()
    model = Model(results_mode=RUN_MODE)
    model_build_time = time.time()
    print(f"Model build time: {model_build_time - start_time:.4f} seconds")

    for name in ("base",):
        scenario = model.scenarios[name]
        results = run_recombination_and_inspect(scenario, model, RUN_MODE)
        analyze_and_plot_results(
            results=results, 
            output_dir=scenario.solution_dir, 
            scenario_name=scenario.name, 
            save_plot=False # Toggle as needed
        )