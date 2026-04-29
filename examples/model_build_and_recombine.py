"""
An example is given for building a FIRM Model instance and executing a single pass
of the MHMGA recombination operators for inspection and hyperparameter tuning.

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""
from datetime import datetime as dt
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from firm_ce.model import Model


def write_recombination_summary(results: dict, output_dir: str, scenario_name: str, suffix: str = ''):
    """Generates a text summary of the recombination performance metrics."""
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"recomb_summary_{scenario_name}{suffix}.txt")

    def write_stats(f, prefix: str, obj: np.ndarray, viol: np.ndarray):
        if obj is None:
            f.write(f"--- {prefix} ---\nEvaluation bypassed.\n\n")
            return

        is_feas = viol == 0 if viol is not None else np.ones(len(obj), dtype=bool)
        n_feas, n_infeas = np.sum(is_feas), np.sum(~is_feas)

        f.write(f"--- {prefix} ---\nTotal: {len(obj)} | Feasible: {n_feas} | Infeasible: {n_infeas}\n")

        if n_feas > 0:
            f.write(f"Feasible Obj   -> Mean: {np.mean(obj[is_feas]):.4f} "
                    f"| Min: {np.min(obj[is_feas]):.4f} "
                    f"| Max: {np.max(obj[is_feas]):.4f}\n")
        else:
            f.write("Feasible Obj   -> N/A\n")

        if n_infeas > 0:
            f.write(f"Infeasible Viol-> Mean: {np.mean(viol[~is_feas]):.4f} "
                    f"| Min: {np.min(viol[~is_feas]):.4f} "
                    f"| Max: {np.max(viol[~is_feas]):.4f}\n")
        else:
            f.write("Infeasible Viol-> N/A\n")
        f.write("\n")

    with open(txt_path, 'w') as f:
        write_stats(f, "PARENTS", results["parents_objectives"], results["parents_violations"])
        write_stats(f, "OFFSPRING", results["offspring_objectives"], results["offspring_violations"])

    print(f"Summary written to: {txt_path}")


def plot_performance_metrics(
        results: dict,
        output_dir: str,
        scenario_name: str,
        save_plot: bool = False,
        suffix: str = ''
):
    """Generates a 2x3 Seaborn histogram plot for parent vs offspring performance."""
    if results.get("offspring_objectives") is None:
        return

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex='col', gridspec_kw={'width_ratios': [1, 4, 4]})
    fig.suptitle(f"Recombination Metrics: {scenario_name}", fontsize=14)

    axs[0, 2].sharey(axs[0, 1])
    axs[1, 1].sharey(axs[0, 1])
    axs[1, 2].sharey(axs[0, 1])

    def plot_stacked_bar(ax, obj, viol, title):
        if obj is None or len(obj) == 0:
            return
        pop_size = len(obj)
        is_feas = viol == 0 if viol is not None else np.ones(len(obj), dtype=bool)
        p_feas = np.sum(is_feas)

        ax.bar(["Status"], [p_feas], color='mediumseagreen', label='Feasible')
        ax.bar(["Status"], [pop_size - p_feas], bottom=[p_feas], color='lightcoral', label='Infeasible')
        ax.set(ylim=(0, pop_size), ylabel="Count", title=title)

    def plot_hist(ax, data, bins, title, color):
        valid = data[(~np.isnan(data)) & (data != 0)] if data is not None else []
        if len(valid) > 0:
            sns.histplot(valid, bins=bins, color=color, edgecolor='black', alpha=0.7, ax=ax)
            ax.set_ylabel("Count")
        ax.set(title=title)
        ax.grid(True, alpha=0.3)

    def get_shared_bins(d1, d2, num_bins=15):
        v1 = d1[(~np.isnan(d1)) & (d1 != 0)] if d1 is not None else []
        v2 = d2[(~np.isnan(d2)) & (d2 != 0)] if d2 is not None else []
        combined = np.concatenate([v1, v2])
        return np.histogram_bin_edges(combined, bins=num_bins) if len(combined) > 0 else num_bins

    o_bins = get_shared_bins(results.get("parents_objectives"), results.get("offspring_objectives"))
    v_bins = get_shared_bins(results.get("parents_violations"), results.get("offspring_violations"))

    plot_stacked_bar(axs[0, 0], results.get("parents_objectives"), results.get("parents_violations"), "Parent Feasibility")
    axs[0, 0].legend(loc='upper right', bbox_to_anchor=(1.4, 1.05), fontsize='small')
    plot_hist(axs[0, 1], results.get("parents_objectives"), o_bins, "Parent Objectives", "skyblue")
    plot_hist(axs[0, 2], results.get("parents_violations"), v_bins, "Parent Violations", "lightcoral")

    plot_stacked_bar(axs[1, 0], results.get("offspring_objectives"), results.get("offspring_violations"), "Offspring Feasibility")
    plot_hist(axs[1, 1], results.get("offspring_objectives"), o_bins, "Offspring Objectives", "steelblue")
    plot_hist(axs[1, 2], results.get("offspring_violations"), v_bins, "Offspring Violations", "firebrick")

    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(output_dir, f"recomb_performance_{scenario_name}{suffix}.png"), dpi=150)

    plt.show()


def plot_capacity_distributions(results: dict, scenario, save_plot: bool = False, suffix: str = ''):
    """Generates grouped Seaborn boxplots of capacity allocations by technology."""

    if results.get("offspring_points") is None:
        return

    tech_map = scenario.get_tech_index_map()
    records = []

    def aggregate_and_record(points, violations, pop_label):
        if points is None or len(points) == 0:
            return
        is_feasible = (violations == 0) if violations is not None else np.ones(len(points), dtype=bool)

        for tech, indices in tech_map.items():
            if not indices:
                continue
            tech_sums = points[:, indices].sum(axis=1)

            for i in range(len(points)):
                status = "Feasible" if is_feasible[i] else "Infeasible"
                records.append({
                    "Population": pop_label,
                    "Status": status,
                    "Group": f"{pop_label} ({status})",
                    "Technology": tech,
                    "Capacity": tech_sums[i]
                })

    aggregate_and_record(results["parents_points"], results["parents_violations"], "Parents")
    aggregate_and_record(results["offspring_points"], results["offspring_violations"], "Offspring")

    if not records:
        return

    df = pd.DataFrame(records)

    # Split and order technologies: Power first, then Energy
    unique_techs = df["Technology"].unique()
    energy_techs = [t for t in unique_techs if "Energy" in t]
    power_techs = [t for t in unique_techs if "Energy" not in t]
    ordered_techs = power_techs + energy_techs

    fig, axes = plt.subplots(5, 3, figsize=(15, 18))
    axes = axes.flatten()

    palette = {
        "Parents (Feasible)": "skyblue",
        "Parents (Infeasible)": "lightcoral",
        "Offspring (Feasible)": "steelblue",
        "Offspring (Infeasible)": "firebrick"
    }

    for i, ax in enumerate(axes):
        if i < len(ordered_techs):
            tech = ordered_techs[i]
            tech_df = df[(df["Technology"] == tech)]

            sns.boxplot(
                data=tech_df,
                x="Technology",
                y="Capacity",
                hue="Group",
                palette=palette,
                ax=ax
            )

            ax.set_title(tech, fontsize=12)
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_ylabel("")

            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            # Hide unused subplots if the grid is larger than the number of technologies
            ax.axis('off')

    # Add a single global legend and title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=11)
    fig.suptitle(f"Capacity Distribution by Technology: {scenario.name.capitalize()}", fontsize=16, y=1.01)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_plot:
        plot_path = os.path.join(scenario.solution_dir, f"recomb_capacities_{scenario.name}{suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

    plt.show()


def generate_recombination_reports(results: dict, scenario, save_plot: bool = False, suffix: str = ''):
    """
    Super function to generate text summaries, performance metrics, and capacity distribution plots.
    """
    print(f"\nGenerating recombination reports for {scenario.name}...")

    write_recombination_summary(
        results=results,
        output_dir=scenario.solution_dir,
        scenario_name=scenario.name,
        suffix=suffix,
    )

    plot_performance_metrics(
        results=results,
        output_dir=scenario.solution_dir,
        scenario_name=scenario.name,
        save_plot=save_plot,
        suffix=suffix,
    )

    plot_capacity_distributions(
        results=results,
        scenario=scenario,
        save_plot=save_plot,
        suffix=suffix,
    )


def _try_path(folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        print(f"Using '{filename}' in {folder}.")
        return path
    print(f"No '{filename}' found in {folder}.")
    return False


def load_population(scenario, run_mode, pop_size, population):
    if population == "x0":
        return scenario.x0, None, None

    if run_mode == "new":
        if population == "random":
            ndim = len(scenario.lower_bounds)
            return np.random.uniform(scenario.lower_bounds, scenario.upper_bounds, (pop_size, ndim)), None, None

    if population == "latest":
        trim = -pop_size
        pop_path = _try_path(scenario.solution_dir, "latest_population.csv")
        if not pop_path:
            pop_path = _try_path("results/temp", "latest_population.csv")
        if not pop_path:
            raise FileNotFoundError("No 'latest_population.csv' found.")

    elif population == "optimum":
        trim = -1
        pop_path = _try_path(scenario.solution_dir, "callback.csv")
        if not pop_path:
            pop_path = _try_path("results/temp", "callback.csv")
        if not pop_path:
            raise FileNotFoundError("No 'callback.csv' found.")

    df = pd.read_csv(pop_path, header=None).iloc[trim:, :]

    return df.iloc[:, 3:].to_numpy(), df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


def run_recombination_and_inspect(scenario, model, run_mode, population, **hyperparameters):
    print(f"\n--- Inspecting Recombination: {scenario.name} ---")

    if run_mode == "new":
        pop_size = model.config.mga_pop_size[0]
    else:
        with open(os.path.join(scenario.solution_dir, "mhmga_config.json")) as f:
            pop_size = json.load(f)["mga_pop_size"][-1]

    hyperparameters["pop_size"] = pop_size
    points, objectives, constraints = load_population(scenario, run_mode, pop_size, population)

    t0 = dt.now()
    results = model.inspect_recombination(
        scenario_name=scenario.name,
        starting_population=points,
        objectives=objectives,
        constraints=constraints,
        evaluate_offspring=True,
        # custom hyperparameters overriding config
        **hyperparameters,
    )

    print(f"Inspection finished in {(dt.now() - t0).total_seconds():.4f}s")
    return results


if __name__ == "__main__":

    RUN_MODE = "latest"

    t0 = dt.now()
    model = Model(results_mode=RUN_MODE)
    print(f"Model build time: {(dt.now() - t0).total_seconds():.4f}s")

    hyperparameters = dict(
        # elite_count=self.config.mga_elite_count[0],
        # tourn_count=self.config.mga_tourn_count[0],
        # tourn_size=self.config.mga_tourn_size[0],
        # mutation_prob=0.2,
        # mutation_sigma=0.1,
        # crossover_prob=self.config.mga_crossover_prob[0],
        # niche_elitism=self.config.mga_niche_elitism[0],
        # noptimal_rel=self.config.mga_noptimal_rel[0],
        # noptimal_abs=self.config.mga_noptimal_abs[0],
        # objective_scaler=1.0,
    )

    for name in ("base",):
        scenario = model.scenarios[name]
        results = run_recombination_and_inspect(scenario, model, RUN_MODE, 'optimum', **hyperparameters)
        generate_recombination_reports(results, scenario, save_plot=True, suffix="_o4")
