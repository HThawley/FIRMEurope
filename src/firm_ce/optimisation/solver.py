# type: ignore
import csv
import os
from typing import Dict, Tuple, Union, Callable, TYPE_CHECKING
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution

from mga.problem_definition import OptimizationProblem
from mga.mhmga import MGAProblem
from mga.population import Population

# Avoid circular import issues
if TYPE_CHECKING:
    from firm_ce.system import Scenario

from firm_ce.common.constants import SAVE_POPULATION, PENALTY_MULTIPLIER
from firm_ce.optimisation.broad_optimum import (
    append_to_midpoint_csv,
    broad_optimum_objective,
    build_broad_optimum_var_info,
    create_groups_dict,
    create_midpoint_csv,
    read_broad_optimum_bands,
    write_broad_optimum_bands,
    write_broad_optimum_records,
)
from firm_ce.optimisation.single_time import Solution, evaluate_vectorised_xs, mga_parallel_wrapper

from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType


class Solver:
    def __init__(
        self,
        scenario: "Scenario",
        config: ModelConfig,
        polish_flag: bool = False,
        initial_population: Union[NDArray[np.float64], None] = None,
    ) -> None:
        self.config = config
        self.decision_x0 = scenario.x0 if len(scenario.x0) > 0 else None
        self.lower_bounds, self.upper_bounds = scenario.lower_bounds, scenario.upper_bounds
        self.parameters_static = scenario.static
        self.fleet_static = scenario.fleet
        self.network_static = scenario.network
        self.logger = scenario.logger
        self.log_dir = scenario.results_dir
        self.solution_dir = scenario.solution_dir
        self.broad_optimum_var_info = build_broad_optimum_var_info(self.fleet_static, self.network_static)
        self.scenario_name = scenario.name
        self.result = None
        self.optimal_lcoe = None
        self.initial_population = initial_population

        if config.type != "mhmga":
            if polish_flag:
                self.iterations = int(config.iterations // 2)
            else:
                self.iterations = config.iterations
        else:
            self.mga_log_dir = os.path.join(self.solution_dir, "mga_logs")
            os.makedirs(self.mga_log_dir, exist_ok=True)

    def initialise_callback(self) -> None:
        temp_dir = os.path.join("results", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "callback.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "latest_population.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population_energies.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)

    def get_differential_evolution_args(
        self,
    ) -> Tuple[ScenarioParameters_InstanceType, Fleet_InstanceType, Network_InstanceType, str, float]:
        args = (
            self.parameters_static,
            self.fleet_static,
            self.network_static,
            self.config.balancing_type,
            self.config.fixed_costs_threshold,
        )
        return args

    def get_mhmga_kwargs(self) -> dict:
        kwargs = {
            "static": self.parameters_static,
            "fleet": self.fleet_static,
            "network": self.network_static,
            "balancing_type": self.config.balancing_type,
            "fixed_costs_threshold": self.config.fixed_costs_threshold,
        }
        return kwargs

    def run_differential_evolution(self, objective_function: Callable, args: Tuple) -> OptimizeResult:
        result = differential_evolution(
            x0=self.decision_x0,
            func=objective_function,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            args=args,
            tol=0,
            maxiter=self.iterations,
            popsize=self.config.population,
            mutation=(0.2, self.config.mutation),
            recombination=self.config.recombination,
            disp=True,
            polish=False,
            updating="deferred",
            callback=callback,
            workers=1,
            vectorized=True,
        )
        return result

    def single_time(self) -> None:
        self.initialise_callback()
        self.result = self.run_differential_evolution(
            evaluate_vectorised_xs, self.get_differential_evolution_args()
        )[0, :]  # just cost + penalties * penalty_multiplier

    def generate_alternatives(self) -> None:
        self.logger.info("[MHMGA] Initialising MGA algorithm. (this may take a while)")

        # fkwargs = self.get_mhmga_kwargs()
        fargs = self.get_differential_evolution_args()
        jacobian = self.get_approximate_jacobian()

        problem = OptimizationProblem(
            objective=mga_parallel_wrapper,
            fargs=fargs,
            bounds=(self.lower_bounds, self.upper_bounds),
            maximize=False,
            vectorized=True,
            constraints=True,
            return_scaled=True,
            known_optimum=self.decision_x0,
        )

        path_name = os.path.join(self.mga_log_dir, "mga_log")

        # create callback folder
        results_dir = os.path.join("results", "temp")
        os.makedirs(results_dir, exist_ok=True)

        algorithm = MGAProblem(
            problem=problem,
            log_dir=path_name,
            log_freq=self.config.mga_log_freq,
            random_seed=None,
            parallelize=False,  # we will implement parallelisation independently
            callback=mga_callback,
            include_obj_in_fitness=True,
        )
        algorithm.add_niches(num_niches=self.config.mga_start_niches)
        self.logger.info(f"[MHMGA] MGA algorithm initialised with {self.config.mga_start_niches} niches.")

        for step in range(self.config.mga_steps):
            start_time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self.logger.info(f"[MHMGA] Starting step {step+1}/{self.config.mga_steps} at {start_time_str}")

            if self.config.mga_new_niches[step] > 0:
                self.logger.info(f"[MHMGA] Adding {self.config.mga_new_niches[step]} niches.")
                algorithm.add_niches(num_niches=self.config.mga_new_niches[step])

            algorithm.update_hyperparameters(
                max_iter=self.config.mga_iter[step],
                pop_size=self.config.mga_pop_size[step],
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
                mutation_scaler=np.abs(jacobian),
                objective_scaler=1.0,
            )
            
            algorithm.step(disp_rate=self.config.mga_disp_rate)

            # 4. Terminate and get results
            results = algorithm.get_results()
            self.save_mga_results(results)

            self.logger.info("[MHMGA] MGA complete. Results saved.")

        self.result = algorithm.population.optima_points[0]

    def save_mga_results(self, results: Dict) -> None:
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

    def get_band_lcoe_max(self) -> float:
        solution = Solution(self.decision_x0, *self.get_differential_evolution_args())

        if solution.penalties > 1:
            self.logger.warning(
                f"Initial guess (assumed optimal solution) has a penalty of {solution.penalties}."
                f"It is recommended to double-check initial_guess.csv contains the correct optimal solution."
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
            self.logger.info(f"[near_optimum] exploring group '{group_key}'")

            bands_record = []
            for band_type in ("min", "max"):
                match band_type:
                    case "min":
                        self.logger.info(f"[near_optimum] finding MIN for group '{group_key}'")
                    case "max":
                        self.logger.info(f"[near_optimum] finding MAX for group '{group_key}'")

                args = (
                    self.get_differential_evolution_args(),
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    band_type,
                )

                result = self.run_differential_evolution(broad_optimum_objective, args)

                bands_record.append(result.x.copy())

            bands[group_key] = tuple(bands_record)

        write_broad_optimum_records(self.scenario_name, evaluation_records, self.broad_optimum_var_info)
        write_broad_optimum_bands(
            self.scenario_name,
            self.broad_optimum_var_info,
            bands,
            self.get_differential_evolution_args(),
            band_lcoe_max,
            groups,
        )
        return bands

    def explore_midpoints(self) -> None:
        self.logger.info(f"[midpoint_explore] beginning midpoint exploration: {self.config.midpoint_count} per group")
        band_lcoe_max = self.get_band_lcoe_max()
        group_bands = read_broad_optimum_bands(self.scenario_name, self.broad_optimum_var_info)
        csv_path = create_midpoint_csv(self.scenario_name, self.broad_optimum_var_info)

        for group_key, bands in group_bands.items():
            band_max, band_min = float(bands["max"]), float(bands["min"])
            step_size = (band_max - band_min) / (self.config.midpoint_count + 1)
            idx_list = [
                variable["idx"] for variable in self.broad_optimum_var_info if (variable[0] or variable[3]) == group_key
            ]

            self.logger.info(
                f"[midpoint_explore] group '{group_key}'  min={band_min:.3f}  max={band_max:.3f}  step={step_size:.3f}"
            )

            for midpoint in range(1, self.config.midpoint_count + 1):
                evaluation_records = []
                group_target = band_min + midpoint * step_size
                self.logger.info(
                    f"[midpoint_explore] midpoint {midpoint}/{self.config.midpoint_count}: "
                    f"target sum ≈ {group_target:.3f}"
                )

                args = (
                    self.get_differential_evolution_args(),
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    "midpoint",
                    group_target,
                    midpoint,
                )

                self.run_differential_evolution(broad_optimum_objective, args)

                append_to_midpoint_csv(self.scenario_name, evaluation_records)

        self.logger.info(f"[midpoint_explore] finished; wrote {len(evaluation_records)} feasible points to {csv_path}")

        return None

    def get_approximate_jacobian(self) -> NDArray[np.float64]:
        """Calculates approximate dC/dx for all assets in the x vector."""

        flexible_costs = []

        for gen in self.fleet_static.generators.values():
            if gen.is_flexible:
                # Total marginal cost = VOM + Fuel Cost ($/MWh)
                marginal_cost_mwh = gen.cost.vom + gen.cost.fuel_cost_mwh
                # Convert $/MWh to $/GWh to match the energy variables
                flexible_costs.append(marginal_cost_mwh * 1e3)
        if not flexible_costs:
            self.logger.warning("No flexible generators found. Assuming 0 system marginal cost.")
            flexible_costs = [0.0]

        assumed_system_marginal_cost = np.mean(flexible_costs)

        def get_annuity_factor(dr: float, life: float) -> float:
            return (1 - (1 + dr) ** (-1 * life)) / dr

        jacobian = []

        for gen in self.fleet_static.generators.values():
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
        for sto in self.fleet_static.storages.values():
            af = get_annuity_factor(sto.cost.discount_rate, sto.cost.lifetime)
            capex = (1e6 * sto.cost.capex_p) / af if af > 1e-6 else 0.0
            fom = 1e6 * sto.cost.fom
            jacobian.append(capex + fom)  # dc_var assumed 0

        # Storages Energy
        for sto in self.fleet_static.storages.values():
            if sto.duration == 0:
                af = get_annuity_factor(sto.cost.discount_rate, sto.cost.lifetime)
                capex = (1e6 * sto.cost.capex_e) / af if af > 1e-6 else 0.0
                jacobian.append(capex)  # dc_var assumed 0
            else:
                jacobian.append(0.0)

        # Lines
        for line in self.network_static.major_lines.values():
            af = get_annuity_factor(line.cost.discount_rate, line.cost.lifetime)
            capex = (1e3 * line.length * line.cost.capex_p + 1e3 * line.cost.transformer_capex) / af if af > 1e-6 else 0.0
            fom = 1e3 * line.length * line.cost.fom
            jacobian.append(capex + fom)  # dc_var assumed 0

        jacobian = np.array(jacobian, dtype=np.float64)
        jacobian /= (self.parameters_static.mean_annual_demand_mwh)  # $/MWh
        return jacobian

    def capacity_expansion(self):
        pass

    def evaluate(self) -> None:
        if self.config.type == "single_time":
            self.single_time()
        elif self.config.type == "near_optimum":
            self.find_near_optimal_band()
        elif self.config.type == "midpoint_explore":
            self.explore_midpoints()
        elif self.config.type == "capacity_expansion":
            self.capacity_expansion()
        elif self.config.type == "mhmga":
            self.generate_alternatives()
        else:
            raise Exception(
                "Model type in config must be 'single_time', 'capacity_expansion', 'near_optimum',"
                "'midpoint_explore', or 'mhmga'."
            )


def callback(intermediate_result: OptimizeResult) -> None:
    results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    # Save best solution from last iteration
    with open(os.path.join(results_dir, "callback.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([intermediate_result.fun, *intermediate_result.x])

    if SAVE_POPULATION:
        combined_block = np.column_stack((intermediate_result.population,
                                          intermediate_result.population_energies))

        with open(os.path.join(results_dir, "population.csv"), "a", newline="") as f_all, \
             open(os.path.join(results_dir, "latest_population.csv"), "w", newline="") as f_latest:

            writer_all = csv.writer(f_all)
            writer_latest = csv.writer(f_latest)

            writer_all.writerows(combined_block)
            writer_latest.writerows(combined_block)


def mga_callback(population: Population) -> None:

    # Save best solution from last iteration
    with open("results/temp/callback.csv", "a", newline="") as f:
        writer = csv.writer(f)
        best_row = [
            population.optima_raw_objectives[0],
            population.optima_penalized_objectives[0],
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

        with open("results/temp/population.csv", "a", newline="") as f_all, \
             open("results/temp/latest_population.csv", "w", newline="") as f_latest:

            writer_all = csv.writer(f_all)
            writer_latest = csv.writer(f_latest)

            writer_all.writerows(combined_block)
            writer_latest.writerows(combined_block)
