import csv
import os
from itertools import chain
from logging import Logger
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution

from firm_ce.common.constants import SAVE_POPULATION
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
from firm_ce.optimisation.single_time import Solution, evaluate_vectorised_xs
from firm_ce.system.components import Fleet_InstanceType, Generator_InstanceType, Storage_InstanceType
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Line_InstanceType, Network_InstanceType


class Solver:
    def __init__(
        self,
        config: ModelConfig,
        initial_x_candidate: NDArray[np.float64],
        parameters_static: ScenarioParameters_InstanceType,
        fleet_static: Fleet_InstanceType,
        network_static: Network_InstanceType,
        scenario_logger: Logger,
        scenario_name: str,
        polish_flag: bool = False,
        initial_population: Union[NDArray[np.float64], None] = None,
    ) -> None:
        self.config = config
        self.decision_x0 = initial_x_candidate if len(initial_x_candidate) > 0 else None
        self.parameters_static = parameters_static
        self.fleet_static = fleet_static
        self.network_static = network_static
        self.logger = scenario_logger
        self.lower_bounds, self.upper_bounds = self.get_bounds()
        self.broad_optimum_var_info = build_broad_optimum_var_info(fleet_static, network_static)
        self.scenario_name = scenario_name
        self.result = None
        self.optimal_lcoe = None
        self.initial_population = initial_population

        if polish_flag:
            self.iterations = int(config.iterations // 2)
        else:
            self.iterations = config.iterations

    def get_bounds(self) -> NDArray[np.float64]:
        def power_capacity_bounds(
            asset_list: Union[List[Generator_InstanceType], List[Storage_InstanceType], List[Line_InstanceType]],
            build_cap_constraint: str,
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) for asset in asset_list]

        def energy_capacity_bounds(storage_list: List[Storage_InstanceType], build_cap_constraint: str) -> List[float]:
            return [getattr(s, build_cap_constraint) if s.duration == 0 else 0.0 for s in storage_list]

        generators = list(self.fleet_static.generators.values())
        storages = list(self.fleet_static.storages.values())
        lines = list(self.network_static.major_lines.values())

        lower_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "min_build"),
                    power_capacity_bounds(storages, "min_build_p"),
                    energy_capacity_bounds(storages, "min_build_e"),
                    power_capacity_bounds(lines, "min_build"),
                )
            )
        )

        upper_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "max_build"),
                    power_capacity_bounds(storages, "max_build_p"),
                    energy_capacity_bounds(storages, "max_build_e"),
                    power_capacity_bounds(lines, "max_build"),
                )
            )
        )

        return lower_bounds, upper_bounds

    def initialise_callback(self) -> None:
        temp_dir = os.path.join("results", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "callback.csv"), "w", newline="") as csvfile:
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

    def single_time(self) -> None:
        self.initialise_callback()

        self.result = differential_evolution(
            x0=self.decision_x0,
            func=evaluate_vectorised_xs,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            args=self.get_differential_evolution_args(),
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

                result = differential_evolution(
                    broad_optimum_objective,
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

                differential_evolution(
                    broad_optimum_objective,
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

                append_to_midpoint_csv(self.scenario_name, evaluation_records)

        self.logger.info(f"[midpoint_explore] finished; wrote {len(evaluation_records)} feasible points to {csv_path}")

        return None

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
        else:
            raise Exception(
                "Model type in config must be 'single_time' or 'capacity_expansion' or 'near_optimum' or"
                "'midpoint_explore'"
            )


def callback(intermediate_result: OptimizeResult) -> None:
    results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    # Save best solution from last iteration
    with open(os.path.join(results_dir, "callback.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(intermediate_result.x))

    if SAVE_POPULATION:
        # Save population from last iteration
        if hasattr(intermediate_result, "population"):
            with open(os.path.join(results_dir, "population.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for individual in intermediate_result.population:
                    writer.writerow(list(individual))

        # Save population energies from last iteration
        if hasattr(intermediate_result, "population_energies"):
            with open(os.path.join(results_dir, "population_energies.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for energy in intermediate_result.population_energies:
                    writer.writerow([energy])
