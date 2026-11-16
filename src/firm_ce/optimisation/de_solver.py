# type: ignore
import csv
import os

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.common.typing import npfloat
from firm_ce.io.binary_logs import convert_binary_logs_to_csv
from firm_ce.optimisation.base import BaseSolver


class DeSolverBase(BaseSolver):
    """Handles standard Differential Evolution Optimisations"""

    def callback(self, intermediate_result: OptimizeResult) -> None:
        out_dir = self.scenario.solution_dir

        with open(os.path.join(out_dir, "callback.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([intermediate_result.fun, *intermediate_result.x])

        with open(os.path.join(out_dir, "current_optimum.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([intermediate_result.fun, "", *intermediate_result.x])

        if SAVE_POPULATION:
            combined_block = np.column_stack(
                (
                    intermediate_result.population,
                    intermediate_result.population_energies,
                )
            ).astype(npfloat, copy=False)

            with open(os.path.join(out_dir, "population.bin"), "ab") as f_all:
                combined_block.tofile(f_all)
            np.save(os.path.join(out_dir, "latest_population.npy"), combined_block)

    def run_differential_evolution(self, objective_function, fargs) -> OptimizeResult:
        result = differential_evolution(
            x0=self.x0,
            func=objective_function,
            bounds=list(zip(self.scenario.lower_bounds, self.scenario.upper_bounds)),
            args=fargs,
            tol=0,
            maxiter=self.iterations,
            popsize=self.config.population,
            mutation=(0.2, self.config.mutation),
            recombination=self.config.recombination,
            disp=True,
            polish=False,
            updating="deferred",
            callback=self.callback,
            workers=1,
            vectorized=True,
        )
        return result


class SingleTimeSolver(DeSolverBase):
    """Simple Differential Evolution single-time optimisation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        polish_flag = getattr(self.config, "polish_flag", False)
        self.iterations = int(self.config.iterations // 2) if polish_flag else self.config.iterations

    def evaluate(self) -> None:
        self.scenario.logger.info("[Single Time] Starting evaluation")
        self.initialise_callback()
        self.result = self.run_differential_evolution(self.objective_function, self.fargs)
        if SAVE_POPULATION:
            convert_binary_logs_to_csv(
                self.scenario.solution_dir,
                pop_cols=len(self.scenario.lower_bounds) + 1,
            )
        self.scenario.logger.info("[Single Time] Optimisation complete")
        return None
