# type: ignore
from abc import abstractmethod
import csv
from datetime import datetime
import os
from typing import Dict

import numpy as np
import pandas as pd
from mga.mhmga import MGAProblem
from mga.population import Population
from mga.problem_definition import OptimizationProblem

from firm_ce.common.constants import PENALTY_MULTIPLIER, SAVE_POPULATION
from firm_ce.common.typing import npfloat
from firm_ce.io.binary_logs import (
    convert_binary_logs_to_csv,
    log_population_and_details,
    write_mhmga_config_summary,
)
from firm_ce.optimisation.base import BaseSolver, graceful_sigint_handler


class MhmgaSolverBase(BaseSolver):
    """Handles generation of near-optimal alternatives using MHMGA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def callback(self) -> None:
        """Callback function implemented by subclass."""
        pass

    @abstractmethod
    def instantiate_algorithm(self) -> None:
        """Algorithm instantiation implemented by subclass."""
        pass

    def _write_mhmga_config_summary(self) -> None:
        write_mhmga_config_summary(
            self.scenario.solution_dir,
            self.config,
            self.scenario,
            self.details,
        )

    def _log_population_and_details(self, population: Population) -> None:
        log_population_and_details(
            self.scenario.solution_dir,
            population,
            getattr(self.config, "save_details", False),
            self.details,
        )

    def _finalise_csv_logs(self) -> None:
        if SAVE_POPULATION:
            pop_cols = len(self.scenario.lower_bounds) + 3
            details_cols = int(self.details.shape[-1]) if getattr(self.config, "save_details", False) else None
            convert_binary_logs_to_csv(
                self.scenario.solution_dir,
                pop_cols=pop_cols,
                details_cols=details_cols,
                dtype=npfloat,
            )

    def inspect_mhmga_recombination(
        self,
        starting_population: np.ndarray,
        objectives: np.ndarray = None,
        constraints: np.ndarray = None,
        evaluate_offspring: bool = True,
        **hyperparameters,
    ) -> None:
        """
        Runs the recombination process with provided population as parents and returns resulting offspring.
        Used for hyperparameter tuning.
        """
        self.scenario.logger.info("[MHMGA] Initialising MGA algorithm for recombination inspection.")

        algorithm = self.instantiate_algorithm(log_path=None, init_callback=False)
        algorithm.add_niches(diversity=1)

        config_hyperparameters = dict(
            max_iter=1,
            pop_size=self.config.mga_pop_size[0],
            champ_count=self.config.mga_champ_count[0],
            elite_count=self.config.mga_elite_count[0],
            tourn_count=self.config.mga_tourn_count[0],
            tourn_size=self.config.mga_tourn_size[0],
            mutation_prob=self.config.mga_mutation_prob[0],
            mutation_sigma=self.config.mga_mutation_sigma[0],
            mutation_alpha=self.config.mga_mutation_alpha[0],
            crossover_prob=self.config.mga_crossover_prob[0],
            niche_elitism=self.config.mga_niche_elitism[0],
            noptimal_rel=self.config.mga_noptimal_rel[0],
            noptimal_abs=self.config.mga_noptimal_abs[0],
            violation_factor=PENALTY_MULTIPLIER,
            mutation_scaler="bounds",
            objective_scaler=1.0,
            extrema_epsilon_rel=self.config.mga_extrema_epsilon_rel[0],
            extrema_epsilon_abs=self.config.mga_extrema_epsilon_abs[0],
        )

        config_hyperparameters.update(hyperparameters)
        algorithm.update_hyperparameters(**config_hyperparameters)

        self.scenario.logger.info("[MHMGA] Beginning recombination inspection.")

        if starting_population.shape[0] == 1:
            starting_population = starting_population.reshape(starting_population.shape[1:])
        if objectives is not None and objectives.shape[0] == 1:
            objectives = objectives.reshape(objectives.shape[1:])
        if constraints is not None and constraints.shape[0] == 1:
            constraints = constraints.reshape(constraints.shape[1:])

        result = algorithm.inspect_recombination(
            starting_points=starting_population,
            point_objectives=objectives,
            point_constraints=constraints,
            evaluate_offspring=evaluate_offspring,
        )

        self.scenario.logger.info("[MHMGA] Finished recombination inspection. Returning results.")

        return result


class MhmgaSolver(MhmgaSolverBase):
    """Handles generation of near-optimal alternatives using MHMGA"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mga_log_dir = os.path.join(self.scenario.solution_dir, "mga_logs")
        os.makedirs(self.mga_log_dir, exist_ok=True)

    def instantiate_algorithm(self, log_path: str, init_callback: bool) -> MGAProblem:
        if self.config.restart_optimisation:
            if getattr(self.config, "model_location", None) == "new":
                pop_dir = "results/temp"
            else:
                pop_dir = self.scenario.solution_dir

            ndim = len(self.scenario.lower_bounds)
            bin_path = os.path.join(pop_dir, "population.bin")
            npy_path = os.path.join(pop_dir, "latest_population.npy")
            csv_path = os.path.join(pop_dir, "latest_population.csv")

            scaled_ndim = self.obj_bundle.scaled_ndim
            extrema_count = 0
            mode = self.config.mga_extrema_dims.lower()
            if mode == "all":
                extrema_count = 2 * scaled_ndim
            elif mode in ("all-min", "all-max"):
                extrema_count = scaled_ndim

            num_niches = 1 + extrema_count + self.config.mga_start_div_niches + self.config.mga_new_div_niches[0]
            pop_size = self.config.mga_pop_size[0]
            expected_rows = num_niches * pop_size

            if os.path.exists(bin_path):
                pop_cols = ndim + 3
                raw_data = np.fromfile(bin_path, dtype=npfloat).reshape(-1, pop_cols)
                raw_pop = raw_data[-expected_rows:]
            elif os.path.exists(npy_path):
                raw_pop = np.load(npy_path)
            elif os.path.exists(csv_path):
                raw_pop = pd.read_csv(csv_path, header=None, dtype=npfloat).to_numpy()
            else:
                raise FileNotFoundError(f"No population file found for restarting. Expected in '{pop_dir}'.")

            if raw_pop.shape[0] != expected_rows:
                raise ValueError(
                    f"Restart population row count ({raw_pop.shape[0]}) does not match "
                    f"expected (num_niches={num_niches} * pop_size={pop_size} = {expected_rows})."
                )

            starting_objectives = raw_pop[:, 0].reshape(num_niches, pop_size).astype(npfloat)
            starting_constraints = raw_pop[:, 1].reshape(num_niches, pop_size).astype(npfloat)
            previous = raw_pop[:, 3 : ndim + 3].reshape(num_niches, pop_size, ndim).astype(npfloat)

            feasible_mask = raw_pop[:, 1] == 0
            if np.any(feasible_mask):
                best_idx = np.where(feasible_mask)[0][np.argmin(raw_pop[feasible_mask, 0])]
            else:
                best_idx = np.argmin(raw_pop[:, 0] + raw_pop[:, 1] * PENALTY_MULTIPLIER)
            x0 = raw_pop[best_idx, 3 : ndim + 3].astype(npfloat)
        else:
            previous = None
            starting_objectives = None
            starting_constraints = None
            x0 = self.scenario.x0

        problem = OptimizationProblem(
            objective=self.objective_function,
            fargs=self.fargs,
            bounds=(self.scenario.lower_bounds, self.scenario.upper_bounds),
            maximize=False,
            vectorized=True,
            constraints=True,
            return_scaled=self.obj_bundle.return_scaled,
            scaled_ndim=self.obj_bundle.scaled_ndim,
        )

        algorithm = MGAProblem(
            problem=problem,
            x0=x0,
            starting_points=previous,
            starting_objectives=starting_objectives,
            starting_constraints=starting_constraints,
            log_dir=log_path,
            log_freq=self.config.mga_log_freq,
            random_seed=None,
            parallelize=False,
            callback=self.callback,
            include_obj_in_fitness=True,
            space_scaler=self.obj_bundle.space_scaler,
        )

        if init_callback:
            self.initialise_callback()

        return algorithm

    def save_results(self, results: Dict) -> None:
        filepath = os.path.join(self.mga_log_dir, "mga_alternatives.csv")

        optima = results["optima"]
        fitness = results["fitness"]
        objective = results["objective"]
        noptimality = results["noptimality"]

        header = ["fitness", "objective", "is_noptimal"] + [f"x{i}" for i in range(optima.shape[1])]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(optima.shape[0]):
                row = [fitness[i], objective[i], noptimality[i]] + list(optima[i])
                writer.writerow(row)

    def callback(self, population: Population) -> None:
        out_dir = self.scenario.solution_dir

        with open(os.path.join(out_dir, "callback.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            best_row = [
                population.optima_raw_objectives[0],
                population.optima_violations[0],
                population.optima_fitnesses[0],
                *population.optima_points[0],
            ]
            writer.writerow(best_row)

        with open(os.path.join(out_dir, "current_optimum.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    population.optima_raw_objectives[0],
                    population.optima_violations[0],
                    *population.optima_points[0],
                ]
            )

        if SAVE_POPULATION:
            self._log_population_and_details(population)

    def evaluate(self) -> None:
        self.scenario.logger.info("[MHMGA] Initialising MGA algorithm.")
        path_name = os.path.join(self.mga_log_dir, "mga")
        self._write_mhmga_config_summary()

        algorithm = self.instantiate_algorithm(path_name, True)

        extrema_kwargs = {}
        if self.config.mga_extrema_dims.lower() != "none":
            extrema_kwargs["extrema_dims"] = self.config.mga_extrema_dims
            if self.config.mga_extrema_is_min.lower() not in ("auto", "none"):
                extrema_kwargs["extrema_is_min"] = self.config.mga_extrema_is_min

        if self.config.restart_optimisation and algorithm.starting_points.ndim == 3:
            loaded_niches = algorithm.starting_points.shape[0]
            algorithm.add_niches(
                diversity=self.config.mga_start_div_niches + self.config.mga_new_div_niches[0],
                **extrema_kwargs,
            )
            self.scenario.logger.info(f"[MHMGA] Restarted MGA algorithm with {loaded_niches} niches from disk.")
        else:
            algorithm.add_niches(
                diversity=self.config.mga_start_div_niches,
                **extrema_kwargs,
            )
            n_opt = algorithm.niche_types.count(0)
            n_ext = algorithm.niche_types.count(1)
            n_div = algorithm.niche_types.count(2)

            self.scenario.logger.info(
                f"[MHMGA] MGA algorithm initialised with {algorithm.num_niches} niches "
                f"({n_opt} optimisation, {n_ext} extrema, {n_div} diversity)."
            )

        try:
            with graceful_sigint_handler(algorithm, self.scenario.logger):
                for step in range(self.config.mga_steps):
                    if getattr(algorithm, "_stop_requested", False):
                        break

                    algorithm.set_verbosity(
                        self.config.mga_disp_rate[step],
                        self.config.mga_verbose_level[step],
                    )

                    start_time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    self.scenario.logger.info(
                        f"[MHMGA] Starting step {step+1}/{self.config.mga_steps} at {start_time_str}"
                    )

                    if self.config.mga_new_div_niches[step] > 0:
                        self.scenario.logger.info(
                            f"[MHMGA] Adding {self.config.mga_new_div_niches[step]} diversity niches."
                        )
                        algorithm.add_niches(diversity=self.config.mga_new_div_niches[step])

                    algorithm.update_hyperparameters(
                        max_iter=self.config.mga_iter[step],
                        pop_size=self.config.mga_pop_size[step],
                        champ_count=self.config.mga_champ_count[step],
                        elite_count=self.config.mga_elite_count[step],
                        tourn_count=self.config.mga_tourn_count[step],
                        tourn_size=self.config.mga_tourn_size[step],
                        mutation_prob=self.config.mga_mutation_prob[step],
                        mutation_sigma=self.config.mga_mutation_sigma[step],
                        crossover_prob=self.config.mga_crossover_prob[step],
                        niche_elitism=self.config.mga_niche_elitism[step],
                        noptimal_rel=self.config.mga_noptimal_rel[step],
                        noptimal_abs=self.config.mga_noptimal_abs[step],
                        violation_factor=PENALTY_MULTIPLIER,
                        mutation_scaler=np.ones_like(self.scenario.lower_bounds),
                        objective_scaler=1.0,
                        fitness_method=self.config.mga_fitness[step],
                        extrema_epsilon_rel=self.config.mga_extrema_epsilon_rel[step],
                        extrema_epsilon_abs=self.config.mga_extrema_epsilon_abs[step],
                    )

                    algorithm.step()

                    if algorithm.population is not None:
                        results = algorithm.get_results()
                        self.save_results(results)

                    if getattr(algorithm, "_stop_requested", False):
                        self.scenario.logger.warning(f"[MHMGA] Halted early at iteration {algorithm.current_iter}.")
                        break

                    self.scenario.logger.info("[MHMGA] MGA step complete. Results saved.")

        finally:
            if algorithm.population is not None:
                results = algorithm.get_results()
                self.save_results(results)
                self.result = algorithm.population.optima_points[0].copy()
            self._finalise_csv_logs()
            self.scenario.logger.info("[MHMGA] Binary logs converted to CSV.")
