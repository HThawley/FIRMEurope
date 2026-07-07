# type: ignore
import csv
import os
from typing import Dict, Tuple, Union, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution
from abc import ABC, abstractmethod

from mga.problem_definition import OptimizationProblem
from mga.mhmga import MGAProblem
from mga.population import Population
import mga.utils.termination as mgat

from firm_ce.common.constants import SAVE_POPULATION, PENALTY_MULTIPLIER
from firm_ce.optimisation.objective_factory import build_objective
from firm_ce.common.typing import npfloat
from firm_ce.backend.scalar.broad_optimum import (
    append_to_midpoint_csv,
    broad_optimum_objective,
    build_broad_optimum_var_info,
    create_groups_dict,
    create_midpoint_csv,
    read_broad_optimum_bands,
    write_broad_optimum_bands,
    write_broad_optimum_records,
)


class BaseSolver(ABC):
    """Abstract base class for all optimisation strategies"""

    def __init__(
        self,
        scenario,
        initial_population: Union[NDArray[np.float64], None] = None,
    ):
        self.scenario = scenario
        self.config = scenario.config
        self.x0 = self.scenario.x0 if len(self.scenario.x0) > 0 else None
        self.initial_population = initial_population
        self.result = None

        self.objective_function, self.fargs = build_objective(self.scenario, self.config)

    @abstractmethod
    def evaluate(self) -> None:
        """Executes the specific optimization strategy. Must be implemented by subclasses."""
        pass

    def initialise_callback(self) -> None:
        out_dir = self.scenario.solution_dir
        os.makedirs(out_dir, exist_ok=True)

        if not self.config.restart_optimisation:
            files = ["callback"]
            if SAVE_POPULATION:
                files.extend(["latest_population", "population"])
                if self.config.save_details:
                    files.extend(["latest_details", "details"])

            for file in files:
                with open(os.path.join(out_dir, f"{file}.csv"), "w", newline="") as csvfile:
                    csv.writer(csvfile)


class DeSolverBase(BaseSolver):
    """ Handles standard Differential Evolution Optimisations """

    def callback(self, intermediate_result: OptimizeResult) -> None:
        out_dir = self.scenario.solution_dir

        # Save best solution from last iteration
        with open(os.path.join(out_dir, "callback.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([intermediate_result.fun, *intermediate_result.x])

        if SAVE_POPULATION:
            combined_block = np.column_stack((intermediate_result.population,
                                              intermediate_result.population_energies))

            with open(os.path.join(out_dir, "population.csv"), "a", newline="") as f_all, \
                 open(os.path.join(out_dir, "latest_population.csv"), "w", newline="") as f_latest:

                writer_all = csv.writer(f_all)
                writer_latest = csv.writer(f_latest)

                writer_all.writerows(combined_block)
                writer_latest.writerows(combined_block)

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
    """ simple optimisation """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        polish_flag = getattr(self.config, "polish_flag", False)
        self.iterations = int(self.config.iterations // 2) if polish_flag else self.config.iterations

    def evaluate(self) -> None:
        self.scenario.logger.info("[Single Time] Starting evaluation")
        self.initialise_callback()
        self.run_differential_evolution(self.objective_function, self.fargs)
        self.scenario.logger.info("[Single Time] Optimisation complete")
        return None


class BroadOptimumSolver(DeSolverBase):
    """Handles broad optimum banding and midpoint exploration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.broad_optimum_var_info = build_broad_optimum_var_info(self.scenario.fleet, self.scenario.network)

    def get_band_lcoe_max(self) -> float:
        from firm_ce.backend.scalar.solution import Solution

        solution = Solution(self.x0, *self.fargs)
        # TODO: evaluation??
        if solution.penalties > 1:
            self.scenario.logger.warning(
                f"Initial guess (assumed optimal solution) has a penalty of {solution.penalties}."
                "It is recommended to double-check initial_guess.csv contains the correct optimal solution."
            )

        self.optimal_lcoe = solution.lcoe
        band_lcoe_max = self.optimal_lcoe * (1 + self.config.near_optimal_tol)

        return band_lcoe_max

    def find_near_optimal_band(self) -> Dict[str, Tuple[float]]:
        band_lcoe_max = self.get_band_lcoe_max()
        evaluation_records = []
        bands = {}
        groups = create_groups_dict(self.broad_optimum_var_info)

        for group_key, idx_list in groups.items():
            self.scenario.logger.info(f"[near_optimum] exploring group '{group_key}'")

            bands_record = []
            for band_type in ("min", "max"):
                match band_type:
                    case "min":
                        self.scenario.logger.info(f"[near_optimum] finding MIN for group '{group_key}'")
                    case "max":
                        self.scenario.logger.info(f"[near_optimum] finding MAX for group '{group_key}'")

                args = (
                    self.fargs,
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    band_type,
                )

                result = self.run_differential_evolution(broad_optimum_objective, args)

                bands_record.append(result.x.copy())

            bands[group_key] = tuple(bands_record)

        write_broad_optimum_records(self.scenario.name, evaluation_records, self.broad_optimum_var_info)
        write_broad_optimum_bands(
            self.scenario.name,
            self.broad_optimum_var_info,
            bands,
            self.fargs,
            band_lcoe_max,
            groups,
        )
        return bands

    def explore_midpoints(self) -> None:
        self.scenario.logger.info(f"[midpoint_explore] beginning midpoint exploration: {self.config.midpoint_count} per group")
        band_lcoe_max = self.get_band_lcoe_max()
        group_bands = read_broad_optimum_bands(self.scenario.name, self.broad_optimum_var_info)
        csv_path = create_midpoint_csv(self.scenario.name, self.broad_optimum_var_info)

        for group_key, bands in group_bands.items():
            band_max, band_min = float(bands["max"]), float(bands["min"])
            step_size = (band_max - band_min) / (self.config.midpoint_count + 1)
            idx_list = [
                variable["idx"] for variable in self.broad_optimum_var_info if (variable[0] or variable[3]) == group_key
            ]

            self.scenario.logger.info(
                f"[midpoint_explore] group '{group_key}'  min={band_min:.3f}  max={band_max:.3f}  step={step_size:.3f}"
            )

            for midpoint in range(1, self.config.midpoint_count + 1):
                evaluation_records = []
                group_target = band_min + midpoint * step_size
                self.scenario.logger.info(
                    f"[midpoint_explore] midpoint {midpoint}/{self.config.midpoint_count}: "
                    f"target sum ≈ {group_target:.3f}"
                )

                args = (
                    self.fargs,
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    "midpoint",
                    group_target,
                    midpoint,
                )

                self.run_differential_evolution(broad_optimum_objective, args)

                append_to_midpoint_csv(self.scenario.name, evaluation_records)

        self.scenario.logger.info(f"[midpoint_explore] finished; wrote {len(evaluation_records)} feasible points to {csv_path}")

        return None

    def evaluate(self) -> None:
        if self.config.type == "near_optimum":
            self.scenario.logger.info("[near_optimum] Starting band finding.")
            self.find_near_optimal_band()

        elif self.config.type == "midpoint_explore":
            self.scenario.logger.info("[midpoint_explore] Starting midpoint exploration.")
            self.explore_midpoints()

        return None


class MhmgaSolverBase(BaseSolver):
    """ Handles generation of near-optimal alternatives using MHMGA """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.config, "save_details", False):
            self.details = self.fargs[-1]

    @abstractmethod
    def callback(self) -> None:
        """Callback function implemented by subclass"""
        pass

    @abstractmethod
    def instantiate_algorithm(self) -> None:
        """Algorithm instantiation implemented by subclass"""
        pass

    def _write_mhmga_config_summary(self) -> None:
        """Writes a JSON summary of MHMGA hyperparameters to the solution directory."""
        summary_path = os.path.join(self.scenario.solution_dir, "mhmga_config.json")

        mhmga_config = {}
        for key, value in self.config.__dict__.items():
            if key.startswith("mga_"):
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(value, np.ndarray):
                    mhmga_config[key] = value.tolist()
                else:
                    mhmga_config[key] = value

        # Inject structural data possibly relevant for loading populations later
        mhmga_config["ndim"] = len(self.scenario.lower_bounds)
        mhmga_config["lower_bounds"] = self.scenario.lower_bounds.tolist()
        mhmga_config["upper_bounds"] = self.scenario.upper_bounds.tolist()
        mhmga_config["unit_type_idxs"] = self.scenario.unit_type_idx

        with open(summary_path, "w") as f:
            json.dump(mhmga_config, f, indent=4)

    def get_approximate_jacobian(self) -> NDArray[np.float64]:
        """Calculates approximate dC/dx for all assets in the x vector."""

        flexible_costs = []

        for gen in self.scenario.fleet.generators.values():
            if gen.is_flexible:
                # Total marginal cost = VOM + Fuel Cost ($/MWh)
                marginal_cost_mwh = gen.cost.vom + gen.cost.fuel_cost_mwh
                # Convert $/MWh to $/GWh to match the energy variables
                flexible_costs.append(marginal_cost_mwh * 1e3)
        if not flexible_costs:
            self.scenario.logger.warning("No flexible generators found. Assuming 0 system marginal cost.")
            flexible_costs = [0.0]

        assumed_system_marginal_cost = np.mean(flexible_costs)

        def get_annuity_factor(dr: float, life: float) -> float:
            return (1 - (1 + dr) ** (-1 * life)) / dr

        jacobian = []

        for gen in self.scenario.fleet.generators.values():
            af = get_annuity_factor(gen.cost.discount_rate, gen.cost.lifetime)
            capex = (1e6 * gen.cost.capex_p) / af if af > 1e-6 else 0.0
            fom = 1e6 * gen.cost.fom
            dc_fixed = capex + fom
            dc_var = 0.0
            if not gen.is_flexible and len(gen.data) > 0:
                cf = np.mean(gen.data)
                annual_gen_gwh = cf * 8760
                # unit_costs.vom is $/MWh -> * 1e3 for $/GWh
                dc_var = annual_gen_gwh * ((gen.cost.vom * 1e3) - assumed_system_marginal_cost)
            jacobian.append(dc_fixed + dc_var)

        # Storages Power
        for sto in self.scenario.fleet.storages.values():
            af = get_annuity_factor(sto.cost.discount_rate, sto.cost.lifetime)
            capex = (1e6 * sto.cost.capex_p) / af if af > 1e-6 else 0.0
            fom = 1e6 * sto.cost.fom
            jacobian.append(capex + fom)  # dc_var assumed 0

        # Storages Energy
        for sto in self.scenario.fleet.storages.values():
            if sto.duration == 0:
                af = get_annuity_factor(sto.cost.discount_rate, sto.cost.lifetime)
                capex = (1e6 * sto.cost.capex_e) / af if af > 1e-6 else 0.0
                jacobian.append(capex)  # dc_var assumed 0
            else:
                jacobian.append(0.0)

        # Lines
        for line in self.scenario.network.major_lines.values():
            af = get_annuity_factor(line.cost.discount_rate, line.cost.lifetime)
            capex = (1e3 * line.length * line.cost.capex_p + 1e3 * line.cost.transformer_capex) / af if af > 1e-6 else 0.0
            fom = 1e3 * line.length * line.cost.fom
            jacobian.append(capex + fom)  # dc_var assumed 0

        jacobian = np.array(jacobian, dtype=np.float64)
        jacobian /= (self.scenario.static.mean_annual_demand_mwh)  # $/MWh
        return jacobian

    def inspect_mhmga_recombination(
        self,
        starting_population: np.ndarray,
        objectives: np.ndarray = None,
        constraints: np.ndarray = None,
        evaluate_offspring: bool = True,
        **hyperparameters,
    ) -> None:
        """
        Runs the recombination process with provided population as parents and displays resulting offspring.
        Used for hyperparameter tuning.
        """
        self.scenario.logger.info("[MHMGA] Initialising MGA algorithm for recombination inspection.")
        # jacobian = self.get_approximate_jacobian()

        # jacobian = abs(jacobian)
        # with open("results/temp/jacobian.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(jacobian)

        algorithm = self.instantiate_algorithm(log_path=None, init_callback=False)
        algorithm.add_niches(num_niches=1)

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
            # mutation_scaler=np.abs(jacobian),
            mutation_scaler='bounds',
            objective_scaler=1.0,
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
            if self.config.model_location == "new":
                pop_dir = "results/temp"
            else:
                pop_dir = self.scenario.solution_dir

            import pandas as pd
            ndim = len(self.scenario.lower_bounds)
            if not os.path.exists(os.path.join(pop_dir, "latest_population.csv")):
                raise FileNotFoundError(
                    f"No population file found for restarting. Expected at {os.path.join(pop_dir, 'latest_population.csv')}"
                )

            previous = pd.read_csv(
                os.path.join(pop_dir, "latest_population.csv"),
                header=None,
                dtype=float,
                usecols=range(3, ndim+3)
            ).to_numpy()
            num_niches = self.config.mga_start_niches + self.config.mga_new_niches[0]
            previous = previous.reshape(num_niches, self.config.mga_pop_size[0], ndim)
            x0 = previous[0]
        else:
            previous = None
            x0 = self.x0_rel

        problem = OptimizationProblem(
            objective=self.objective_function,
            fargs=self.fargs,
            bounds=(self.scenario.lower_bounds_rel, self.scenario.upper_bounds_rel),
            maximize=False,
            vectorized=True,
            constraints=True,
            return_scaled=False,
        )

        algorithm = MGAProblem(
            problem=problem,
            x0=x0,
            log_dir=log_path,
            log_freq=self.config.mga_log_freq,
            random_seed=None,
            parallelize=False,  # we implement parallelisation independently
            callback=self.callback,
            include_obj_in_fitness=True,
        )

        algorithm.starting_points = previous

        if init_callback:
            # callback inited after "results/temp/latest_population.csv" is loaded (if restarting from temp)
            self.initialise_callback()

        return algorithm

    def save_results(self, results: Dict) -> None:
        filepath = os.path.join(self.mga_log_dir, "mga_alternatives.csv")

        optima = results['optima']
        fitness = results['fitness']
        objective = results['objective']
        noptimality = results['noptimality']

        # Create header: meta-data columns first, then decision variables
        header = ["fitness", "objective", "is_noptimal"] + [f"x{i}" for i in range(optima.shape[1])]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(optima.shape[0]):
                row = [fitness[i], objective[i], noptimality[i]] + list(optima[i])
                writer.writerow(row)

    def callback(self, population: Population) -> None:
        out_dir = self.scenario.solution_dir

        # Save best solution from last iteration
        with open(os.path.join(out_dir, "callback.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            best_row = [
                population.optima_raw_objectives[0],
                population.optima_violations[0],
                population.optima_fitnesses[0],
                *population.optima_points[0]
            ]
            writer.writerow(best_row)

        if SAVE_POPULATION:
            # Vectorized flattening: reshape(-1) turns (num_niches, pop_size) into (total_pop,)
            # reshape(-1, ndim) turns (num_niches, pop_size, ndim) into (total_pop, ndim)
            obj_flat = population.raw_objectives.reshape(-1)
            viol_flat = population.violations.reshape(-1)
            fit_flat = population.fitnesses.reshape(-1)
            pts_flat = population.points.reshape(-1, population.points.shape[-1])

            # Combine everything into one matrix: [Obj, Viol, Fit, X0, X1, ...]
            # Shape: (total_pop, 3 + ndim)
            combined_block = np.column_stack((obj_flat, viol_flat, fit_flat, pts_flat))

            with open(os.path.join(out_dir, "population.csv"), "a", newline="") as f_all, \
                 open(os.path.join(out_dir, "latest_population.csv"), "w", newline="") as f_latest:

                writer_all = csv.writer(f_all)
                writer_latest = csv.writer(f_latest)

                writer_all.writerows(combined_block)
                writer_latest.writerows(combined_block)

            num_niches, pop_size, ndim = population.points.shape

            if self.config.save_details:
                details_flat = self.details[:num_niches, :pop_size].reshape(num_niches * pop_size, -1)
                with open(os.path.join(out_dir, "details.csv"), "a", newline="") as f_all, \
                     open(os.path.join(out_dir, "latest_details.csv"), "w", newline="") as f_latest:
                    writer_all = csv.writer(f_all)
                    writer_latest = csv.writer(f_latest)

                    writer_all.writerows(details_flat)
                    writer_latest.writerows(details_flat)

    def evaluate(self) -> None:
        self.scenario.logger.info("[MHMGA] Initialising MGA algorithm.")

        # jacobian = self.get_approximate_jacobian()
        path_name = os.path.join(self.mga_log_dir, "mga")

        self._write_mhmga_config_summary()

        algorithm = self.instantiate_algorithm(path_name, True)

        algorithm.add_niches(num_niches=self.config.mga_start_niches)
        self.scenario.logger.info(f"[MHMGA] MGA algorithm initialised with {self.config.mga_start_niches} niches.")

        for step in range(self.config.mga_steps):
            algorithm.set_verbosity(
                self.config.mga_disp_rate[step],
                self.config.mga_verbose_level[step],
            )

            start_time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self.scenario.logger.info(f"[MHMGA] Starting step {step+1}/{self.config.mga_steps} at {start_time_str}")

            if self.config.mga_new_niches[step] > 0:
                self.scenario.logger.info(f"[MHMGA] Adding {self.config.mga_new_niches[step]} niches.")
                algorithm.add_niches(num_niches=self.config.mga_new_niches[step])

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
                space_scaler=self.scenario.projection_matrix,
                objective_scaler=1.0,
                fitness_method=self.config.mga_fitness[step]
            )

            algorithm.step()

            # 4. Terminate and get results
            results = algorithm.get_results()
            self.save_results(results)

            self.scenario.logger.info("[MHMGA] MGA complete. Results saved.")

        self.result = algorithm.population.optima_points[0]


class MhmgaExtremaSolver(MhmgaSolverBase):
    """Handles searching for geometric extrema in the near-optimal space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mga_log_dir = os.path.join(self.scenario.solution_dir, "mhmga_extrema_logs")
        os.makedirs(self.mga_log_dir, exist_ok=True)

    def instantiate_algorithm(self, log_path, maximize, weighting, lcoe_constr, starting_points):
        problem = OptimizationProblem(
            objective=self.objective_function,
            fargs=self.fargs,
            bounds=(self.scenario.lower_bounds, self.scenario.upper_bounds),
            maximize=maximize,
            vectorized=True,
            constraints=True,
            return_scaled=False,
        )

        algorithm = MGAProblem(
            problem=problem,
            x0=starting_points[0],
            log_dir=log_path,
            log_freq=self.config.mga_log_freq,
            random_seed=None,
            parallelize=False,  # we implement parallelisation independently
            callback=self.mga_extrema_callback,
            include_obj_in_fitness=True,
        )

        algorithm.starting_points = starting_points

        algorithm.add_niches(num_niches=1)  # pure optimisation

        return algorithm

    def find_starting_points_in_population(self, weighting):
        # from firm_ce.optimisation.mhmga_extrema import apply_new_weighting

        pop_path = os.path.join(self.scenario.solution_dir, "population.csv")
        det_path = os.path.join(self.scenario.solution_dir, "details.csv")
        pop_exists = os.path.exists(pop_path)
        det_exists = os.path.exists(det_path)
        if not pop_exists or not det_exists:
            raise FileNotFoundError(
                f"[MHMGA-E] Could not find '{'population.csv' if not pop_exists else ''}'"
                f"{' and ' if not pop_exists and not det_exists else ''}"
                f"{'details.csv' if not det_exists else ''}'. "
                f"Expected in '{self.scenario.solution_dir}'"
            )

        pop = pd.read_csv(pop_path, header=None).to_numpy()
        # det = pd.read_csv(det_path, header=None).to_numpy()

        optimal_lcoe = pop[pop[:, 1] == 0, 0].min()  # best objective among feasible solutions
        noptimal_rel = self.config.mga_noptimal_rel[0]
        noptimal_abs = self.config.mga_noptimal_abs[0]
        noptimal_lcoe = optimal_lcoe + abs(optimal_lcoe) * (noptimal_rel) + noptimal_abs

        new_objective = pop[:, 3:] @ weighting  # weighted sum of decision variables
        feasible_mask = (pop[:, 0] <= noptimal_lcoe) & (pop[:, 1] == 0)  # near-optimal, feasible solutions
        feasible_objectives = new_objective[feasible_mask]
        feasible_points = pop[feasible_mask]

        pop_size = self.config.mga_pop_size[0]
        max_indices = np.argpartition(feasible_objectives, -pop_size)[-pop_size:]  # indices that maximize new objective
        min_indices = np.argpartition(feasible_objectives, pop_size-1)[:pop_size]  # indices that minimize new objective

        max_points = feasible_points[max_indices]
        min_points = feasible_points[min_indices]
        return max_points, min_points, noptimal_lcoe

    def callback(self, population: Population) -> None:
        out_dir = self.scenario.solution_dir

        if SAVE_POPULATION:
            # Vectorized flattening: reshape(-1) turns (num_niches, pop_size) into (total_pop,)
            # reshape(-1, ndim) turns (num_niches, pop_size, ndim) into (total_pop, ndim)
            obj_flat = population.raw_objectives.reshape(-1)
            viol_flat = population.violations.reshape(-1)
            fit_flat = population.fitnesses.reshape(-1)
            pts_flat = population.points.reshape(-1, population.points.shape[-1])

            combined_block = np.column_stack((obj_flat, viol_flat, fit_flat, pts_flat))

            with open(os.path.join(out_dir, "population.csv"), "a", newline="") as f_all, \
                 open(os.path.join(out_dir, "latest_population.csv"), "w", newline="") as f_latest:

                writer_all = csv.writer(f_all)
                writer_latest = csv.writer(f_latest)

                writer_all.writerows(combined_block)
                writer_latest.writerows(combined_block)

            num_niches, pop_size, ndim = population.points.shape
            details_flat = self.details[:num_niches, :pop_size].reshape(num_niches * pop_size, ndim)

            if self.config.save_details:
                with open(os.path.join(out_dir, "details.csv"), "a", newline="") as f_all, \
                     open(os.path.join(out_dir, "latest_details.csv"), "w", newline="") as f_latest:
                    writer_all = csv.writer(f_all)
                    writer_latest = csv.writer(f_latest)

                    writer_all.writerows(details_flat)
                    writer_latest.writerows(details_flat)

    def evaluate(self):
        self.scenario.logger.info("[MHMGA-E] Starting extrema generation.")

        path_name = os.path.join(self.mga_log_dir, "mga")

        weightings = []
        weightings.extend(construct_variable_weightings(self.scenario))
        weightings.extend(construct_unit_type_weightings(self.scenario))
        weightings.extend(construct_node_weightings(self.scenario))

        nweight = len(weightings)
        self.scenario.logger.info(f"[MHMGA-E] Constructed {nweight} weightings for extrema generation.")

        for w, weighting in enumerate(weightings):
            print(f"[MHMGA-E] Starting extremum generation for extremum {w+1}/{nweight}.")
            max_points, min_points, noptimal_lcoe, = self.find_starting_points_in_population(weighting)

            for maximize, points in zip((True, False), (max_points, min_points)):
                algorithm = self.instantiate_algorithm(
                    path_name, maximize, weighting, noptimal_lcoe, points
                )

                for step in range(self.config.mga_steps):
                    algorithm.set_verbosity(
                        self.config.mga_disp_rate[step],
                        self.config.mga_verbose_level[step],
                    )

                    start_time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    self.scenario.logger.info(f"[MHMGA] Starting step {step+1}/{self.config.mga_steps} at {start_time_str}")

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
                        niche_elitism=0,
                        noptimal_rel=0,
                        noptimal_abs=np.inf,
                        violation_factor=PENALTY_MULTIPLIER,
                        mutation_scaler='bounds',  # TODO: enlargen weightings?
                    )

                    abs_min = self.scenario.lower_bounds[weighting > 0].sum()
                    abs_max = self.scenario.upper_bounds[weighting > 0].sum()

                    algorithm.step(
                        termination_criteria=[
                            mgat.FixedValue(
                                value=abs_max if maximize else abs_min,
                                maximize=maximize,
                                attribute="current_optimum",
                                attribute_in_population=True,
                            ),  # terminate on reaching theoretical max/min of the weighted sum objective
                            mgat.GradientStagnation(
                                window=10,
                                improvement=0.01*abs_max,
                                maximize=maximize,
                                attribute="current_optimum",
                                attribute_in_population=True,
                            )  # terminate if less than 1% of theoretic max improvement in 10 iterations
                        ]
                    )

        #       log population - dedicated callback func?
        #   log extremum

        # second pass?


def get_solver(
    scenario,
    initial_population: Optional[np.ndarray] = None,
) -> BaseSolver:
    """Factory function to instantiate the correct solver strategy."""

    strategies = {
        "single_time": SingleTimeSolver,
        "mhmga": MhmgaSolver,
        "mhmga_extrema": MhmgaExtremaSolver,
        "near_optimum": BroadOptimumSolver,
        "midpoint_explore": BroadOptimumSolver,
    }

    solver_class = strategies.get(scenario.config.type)

    if not solver_class:
        raise ValueError(
            f"Unsupported config.type: '{scenario.config.type}'. "
            f"Must be one of {list(strategies.keys())}."
        )

    return solver_class(scenario, initial_population)


# TODO: dev
def construct_variable_weightings(scenario):
    weightings = []
    for i in range(len(scenario.lower_bounds)):
        # min/max each decision variable
        weighting = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
        weighting[i] = 1.0
        weightings.append(weighting)

    return weightings


def construct_unit_type_weightings(scenario):
    # min/max each collection of unit types
    unit_type_w = {}
    unit_type_group_map = {
        'pv_fixed': 'solar',
        'pv_track': 'solar',
        'offw': 'wind',
        'onsw': 'wind',
        'nuclear': 'nuclear_all',
        'nuclear_lte': 'nuclear_all',
        'ac_ohl_mountain_transmission': 'transm',
        'ac_ohl_transmission': 'transm',
        'dc_subsea_transmission': 'transm',
        'dc_underground_transmission': 'transm',
        'biogas': 'bioenergy',
        'biomass': 'bioenergy',
    }
    # custom groupings
    unit_type_w['solar'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
    unit_type_w['wind'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
    unit_type_w['nuclear_all'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
    # unit_type_w['hydro'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)  # only newphes expansion
    unit_type_w['transm'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
    unit_type_w['bioe'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)

    for gen in scenario.fleet.generators.values():
        ut = gen.unit_type
        if ut not in unit_type_w:
            unit_type_w[ut] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)

        if gen.candidate_x_idx != -1:
            unit_type_w[ut][gen.candidate_x_idx] = 1.0
            if ut in unit_type_group_map:
                unit_type_w[unit_type_group_map[ut]][gen.candidate_x_idx] = 1.0

    for sto in scenario.fleet.storages.values():
        ut_p = sto.unit_type + "_p"
        ut_e = sto.unit_type + "_e"
        if ut_p not in unit_type_w:
            unit_type_w[ut_p] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
        if ut_e not in unit_type_w:
            unit_type_w[ut_e] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)

        if sto.candidate_p_x_idx != -1:
            unit_type_w[ut_p][sto.candidate_p_x_idx] = 1.0
        if sto.candidate_e_x_idx != -1:
            unit_type_w[ut_e][sto.candidate_e_x_idx] = 1.0

    for line in scenario.network.major_lines.values():
        ut = line.unit_type
        if ut not in unit_type_w:
            unit_type_w[ut] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)

        if line.candidate_x_idx != -1:
            unit_type_w[ut][line.candidate_x_idx] = 1.0
            if ut in unit_type_group_map:
                unit_type_w[unit_type_group_map[ut]][line.candidate_x_idx] = 1.0

    return [w for w in unit_type_w.values() if w.sum() > 1]  # only return group weightings that have more than 1 member


def construct_node_weightings(scenario):
    node_w = {}

    for gen in scenario.fleet.generators.values():
        node = gen.node
        if node not in node_w:
            node_w[node] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)

        if gen.candidate_x_idx != -1:
            node_w[node][gen.candidate_x_idx] = 1.0

    for sto in scenario.fleet.storages.values():
        node = sto.node
        if node not in node_w:
            node_w[node] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)

        if sto.candidate_p_x_idx != -1:
            node_w[node][sto.candidate_p_x_idx] = 1.0
        if sto.candidate_e_x_idx != -1:
            node_w[node][sto.candidate_e_x_idx] = 1.0

    return [w for w in node_w.values() if w.sum() > 1]  # only return group weightings that have more than 1 member
