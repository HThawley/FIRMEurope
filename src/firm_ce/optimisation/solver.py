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
import signal
from contextlib import contextmanager

from mga.problem_definition import OptimizationProblem
from mga.mhmga import MGAProblem
from mga.population import Population
import mga.utils.termination as mgat

from firm_ce.common.constants import SAVE_POPULATION, PENALTY_MULTIPLIER
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
from firm_ce.optimisation.objective_factory import build_objective_bundle


@contextmanager
def graceful_sigint_handler(algorithm: MGAProblem, logger=None):
    """
    Intercepts SIGINT (Ctrl+C) during Numba execution, deferring termination
    until the current iteration finishes. Prevents raising KeyboardInterrupt
    inside Numba's `_numba_unpickle` C-callbacks (which causes SystemError).
    """
    original_handler = signal.getsignal(signal.SIGINT)
    sigint_count = 0

    def _handler(signum, frame):
        nonlocal sigint_count
        sigint_count += 1
        algorithm._stop_requested = True

        if sigint_count == 1:
            msg = "[FIRM_CE] Interrupt received. Finishing current iteration and finalising logs (press Ctrl+C 3x to force-kill)..."
            if logger:
                logger.warning(msg)
            else:
                print(f"\n{msg}", flush=True)
        elif sigint_count >= 3:
            print("\n[FIRM_CE] Forced termination requested. Exiting immediately.", flush=True)
            os._exit(130)

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)


def convert_binary_logs_to_csv(
    out_dir: str,
    pop_cols: Optional[int] = None,
    details_cols: Optional[int] = None,
    chunk_rows: int = 50_000,
    dtype=npfloat,
) -> None:
    """
    Converts binary optimisation logs (`population.bin`, `latest_population.npy`,
    `details.bin`, `latest_details.npy`) in `out_dir` to `.csv` files for downstream
    processes.

    Can be called automatically at the end of `evaluate()` or manually from a script/REPL
    if an optimisation was terminated early. If `pop_cols` or `details_cols` are not
    provided, they are inferred automatically from `latest_population.npy` /
    `latest_details.npy` or `mhmga_config.json`.
    """
    dtype = np.dtype(dtype)

    # 1. Convert latest_*.npy snapshots first & infer column counts if needed
    latest_pop_npy = os.path.join(out_dir, "latest_population.npy")
    if os.path.exists(latest_pop_npy):
        arr = np.load(latest_pop_npy)
        if pop_cols is None:
            pop_cols = arr.shape[1]
        np.savetxt(os.path.join(out_dir, "latest_population.csv"), arr, delimiter=",", fmt="%.7g")

    latest_det_npy = os.path.join(out_dir, "latest_details.npy")
    if os.path.exists(latest_det_npy):
        arr = np.load(latest_det_npy)
        if details_cols is None:
            details_cols = arr.shape[1]
        np.savetxt(os.path.join(out_dir, "latest_details.csv"), arr, delimiter=",", fmt="%.7g")

    # Fallback column inference from mhmga_config.json if latest_*.npy wasn't written yet
    config_json = os.path.join(out_dir, "mhmga_config.json")
    if (pop_cols is None or details_cols is None) and os.path.exists(config_json):
        with open(config_json, "r") as f:
            meta = json.load(f)
        if pop_cols is None and "pop_cols" in meta:
            pop_cols = meta["pop_cols"]
        elif pop_cols is None and "ndim" in meta:
            pop_cols = meta["ndim"] + 3
        if details_cols is None and "details_cols" in meta:
            details_cols = meta["details_cols"]

    # 2. Stream-convert append-only .bin files using memmap in chunks
    def _stream_bin_to_csv(bin_path: str, csv_path: str, n_cols: Optional[int]) -> None:
        if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
            return
        if n_cols is None:
            raise ValueError(
                f"Cannot infer column count for '{bin_path}'. Please pass `pop_cols` / `details_cols` explicitly."
            )
        total_floats = os.path.getsize(bin_path) // dtype.itemsize
        n_rows = total_floats // n_cols
        if n_rows == 0:
            return

        mmap_arr = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_rows, n_cols))
        with open(csv_path, "w", newline="") as f_csv:
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                np.savetxt(f_csv, mmap_arr[start:end], delimiter=",", fmt="%.7g")

    _stream_bin_to_csv(
        os.path.join(out_dir, "population.bin"),
        os.path.join(out_dir, "population.csv"),
        pop_cols,
    )
    _stream_bin_to_csv(
        os.path.join(out_dir, "details.bin"),
        os.path.join(out_dir, "details.csv"),
        details_cols,
    )


def recover_interrupted_run(
    out_dir: str,
    dtype=npfloat,
) -> Optional[NDArray[npfloat]]:
    """
    Cleans up and finalises logging files in `out_dir` after an aborted/interrupted
    optimisation run:
      1. Truncates any torn trailing bytes in `population.bin` and `details.bin`.
      2. Converts `.bin` and `latest_*.npy` files to `.csv`.
      3. Reconstructs `current_optimum.csv` and returns the best feasible decision vector `x`.
    """
    dtype = np.dtype(dtype)
    config_json = os.path.join(out_dir, "mhmga_config.json")

    pop_cols = None
    details_cols = None
    if os.path.exists(config_json):
        with open(config_json, "r") as f:
            meta = json.load(f)
        pop_cols = meta.get("pop_cols", meta.get("ndim", 0) + 3 or None)
        details_cols = meta.get("details_cols", None)

    latest_pop_npy = os.path.join(out_dir, "latest_population.npy")
    if pop_cols is None and os.path.exists(latest_pop_npy):
        pop_cols = np.load(latest_pop_npy, mmap_mode="r").shape[1]

    # 1. Repair torn binary files (truncate to exact integer multiple of row_bytes)
    for fname, n_cols in (("population.bin", pop_cols), ("details.bin", details_cols)):
        bin_path = os.path.join(out_dir, fname)
        if os.path.exists(bin_path) and n_cols is not None:
            row_bytes = n_cols * dtype.itemsize
            file_size = os.path.getsize(bin_path)
            remainder = file_size % row_bytes
            if remainder != 0:
                valid_size = file_size - remainder
                with open(bin_path, "r+b") as f_bin:
                    f_bin.truncate(valid_size)
                print(f"[Recover] Truncated {remainder} corrupted trailing bytes from '{fname}'.")

    # 2. Convert binary logs to CSV
    convert_binary_logs_to_csv(
        out_dir=out_dir,
        pop_cols=pop_cols,
        details_cols=details_cols,
        dtype=dtype,
    )

    # 3. Extract and return the best feasible solution found so far
    best_x = None
    pop_bin_path = os.path.join(out_dir, "population.bin")
    if os.path.exists(pop_bin_path) and os.path.getsize(pop_bin_path) > 0 and pop_cols is not None:
        total_rows = os.path.getsize(pop_bin_path) // (pop_cols * dtype.itemsize)
        mmap_pop = np.memmap(pop_bin_path, dtype=dtype, mode="r", shape=(total_rows, pop_cols))

        # For MHMGA: col 0 = obj, col 1 = violation, col 2 = fitness, cols 3: = x
        # For SingleTime DE: cols 0:ndim = x, col -1 = energy
        if os.path.exists(config_json):
            feasible_mask = mmap_pop[:, 1] == 0
            if np.any(feasible_mask):
                feasible_indices = np.where(feasible_mask)[0]
                best_idx = feasible_indices[np.argmin(mmap_pop[feasible_mask, 0])]
            else:
                best_idx = int(np.argmin(mmap_pop[:, 0] + mmap_pop[:, 1] * PENALTY_MULTIPLIER))
            best_row = np.array(mmap_pop[best_idx], dtype=dtype)
            best_x = best_row[3:]

            with open(os.path.join(out_dir, "current_optimum.csv"), "w", newline="") as f:
                csv.writer(f).writerow([best_row[0], best_row[1], *best_x])
        else:
            best_idx = int(np.argmin(mmap_pop[:, -1]))
            best_row = np.array(mmap_pop[best_idx], dtype=dtype)
            best_x = best_row[:-1]

    print(f"[Recover] Log recovery and CSV conversion complete for '{out_dir}'.")
    return best_x


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

        self.obj_bundle = build_objective_bundle(self.scenario, self.config)
        self.objective_function = self.obj_bundle.func
        self.fargs = self.obj_bundle.fargs
        self.details = self.obj_bundle.details

    @abstractmethod
    def evaluate(self) -> None:
        """Executes the specific optimization strategy. Must be implemented by subclasses."""
        pass

    def initialise_callback(self) -> None:
        out_dir = self.scenario.solution_dir
        os.makedirs(out_dir, exist_ok=True)

        if not self.config.restart_optimisation:
            with open(os.path.join(out_dir, "callback.csv"), "w", newline="") as csvfile:
                csv.writer(csvfile)

            if SAVE_POPULATION:
                open(os.path.join(out_dir, "population.bin"), "wb").close()
                if getattr(self.config, "save_details", False):
                    open(os.path.join(out_dir, "details.bin"), "wb").close()


class DeSolverBase(BaseSolver):
    """ Handles standard Differential Evolution Optimisations """

    def callback(self, intermediate_result: OptimizeResult) -> None:
        out_dir = self.scenario.solution_dir

        # Save best solution from last iteration
        with open(os.path.join(out_dir, "callback.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([intermediate_result.fun, *intermediate_result.x])

        with open(os.path.join(out_dir, "current_optimum.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([intermediate_result.fun, "", *intermediate_result.x])

        if SAVE_POPULATION:
            combined_block = np.column_stack((
                intermediate_result.population,
                intermediate_result.population_energies,
            )).astype(npfloat, copy=False)

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
    """ simple optimisation """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        polish_flag = getattr(self.config, "polish_flag", False)
        self.iterations = int(self.config.iterations // 2) if polish_flag else self.config.iterations

    def evaluate(self) -> None:
        self.scenario.logger.info("[Single Time] Starting evaluation")
        self.initialise_callback()
        self.run_differential_evolution(self.objective_function, self.fargs)
        if SAVE_POPULATION:
            convert_binary_logs_to_csv(
                self.scenario.solution_dir,
                pop_cols=len(self.scenario.lower_bounds) + 1,
            )
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
                if isinstance(value, np.ndarray):
                    mhmga_config[key] = value.tolist()
                else:
                    mhmga_config[key] = value

        mhmga_config["parameterisation"] = self.config.parameterisation

        ndim = len(self.scenario.lower_bounds)
        mhmga_config["ndim"] = ndim
        mhmga_config["pop_cols"] = ndim + 3
        if getattr(self.config, "save_details", False):
            mhmga_config["details_cols"] = int(self.details.shape[-1])
        mhmga_config["lower_bounds"] = self.scenario.lower_bounds.tolist()
        mhmga_config["upper_bounds"] = self.scenario.upper_bounds.tolist()
        mhmga_config["unit_type_idxs"] = self.scenario.unit_type_idx
        mhmga_config["parameterisation"] = self.config.parameterisation
        mhmga_config["mga_space_basis"] = getattr(self.config, "mga_space_basis", "capacity")
        mhmga_config["mga_space_norm"] = getattr(self.config, "mga_space_norm", self.config.parameterisation)
        mhmga_config["mga_space_agg"] = getattr(self.config, "mga_space_agg", "unit_type")

        with open(summary_path, "w") as f:
            json.dump(mhmga_config, f, indent=4)

    def _log_population_and_details(self, population: Population) -> None:
        """Fast binary logging of population and details arrays."""
        out_dir = self.scenario.solution_dir

        obj_flat = population.raw_objectives.reshape(-1)
        viol_flat = population.violations.reshape(-1)
        fit_flat = population.fitnesses.reshape(-1)
        pts_flat = population.points.reshape(-1, population.points.shape[-1])

        # Shape: (total_pop, 3 + ndim), cast to npfloat (float32)
        combined_block = np.column_stack((obj_flat, viol_flat, fit_flat, pts_flat)).astype(npfloat, copy=False)

        with open(os.path.join(out_dir, "population.bin"), "ab") as f_all:
            combined_block.tofile(f_all)
        np.save(os.path.join(out_dir, "latest_population.npy"), combined_block)

        if getattr(self.config, "save_details", False):
            num_niches, pop_size, _ = population.points.shape
            details_flat = np.ascontiguousarray(
                self.details[:num_niches, :pop_size].reshape(num_niches * pop_size, -1),
                dtype=npfloat,
            )
            with open(os.path.join(out_dir, "details.bin"), "ab") as f_all:
                details_flat.tofile(f_all)
            np.save(os.path.join(out_dir, "latest_details.npy"), details_flat)

    def _finalise_csv_logs(self) -> None:
        """Converts binary logs to .csv at the end of an MHMGA run."""
        if SAVE_POPULATION:
            pop_cols = len(self.scenario.lower_bounds) + 3
            details_cols = int(self.details.shape[-1]) if getattr(self.config, "save_details", False) else None
            convert_binary_logs_to_csv(
                self.scenario.solution_dir,
                pop_cols=pop_cols,
                details_cols=details_cols,
                dtype=npfloat,
            )

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
            mutation_scaler='bounds',
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

            # Determine expected num_niches
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
                raise FileNotFoundError(
                    f"No population file found for restarting. Expected in '{pop_dir}'."
                )

            if raw_pop.shape[0] != expected_rows:
                raise ValueError(
                    f"Restart population row count ({raw_pop.shape[0]}) does not match "
                    f"expected (num_niches={num_niches} * pop_size={pop_size} = {expected_rows})."
                )

            # Extract objectives (col 0), violations (col 1), and points (cols 3:ndim+3)
            starting_objectives = raw_pop[:, 0].reshape(num_niches, pop_size).astype(npfloat)
            starting_constraints = raw_pop[:, 1].reshape(num_niches, pop_size).astype(npfloat)
            previous = raw_pop[:, 3:ndim + 3].reshape(num_niches, pop_size, ndim).astype(npfloat)

            # Best feasible point as 1D x0
            feasible_mask = raw_pop[:, 1] == 0
            if np.any(feasible_mask):
                best_idx = np.where(feasible_mask)[0][np.argmin(raw_pop[feasible_mask, 0])]
            else:
                best_idx = np.argmin(raw_pop[:, 0] + raw_pop[:, 1] * PENALTY_MULTIPLIER)
            x0 = raw_pop[best_idx, 3:ndim + 3].astype(npfloat)
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
            # callback inited after latest_population is loaded (if restarting from temp)
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

        with open(os.path.join(out_dir, "current_optimum.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                population.optima_raw_objectives[0],
                population.optima_violations[0],
                *population.optima_points[0]
            ])

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
                **extrema_kwargs
            )
            self.scenario.logger.info(f"[MHMGA] Restarted MGA algorithm with {loaded_niches} niches from disk.")
        else:
            algorithm.add_niches(
                diversity=self.config.mga_start_div_niches,
                **extrema_kwargs
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
                    self.scenario.logger.info(f"[MHMGA] Starting step {step+1}/{self.config.mga_steps} at {start_time_str}")

                    if self.config.mga_new_div_niches[step] > 0:
                        self.scenario.logger.info(f"[MHMGA] Adding {self.config.mga_new_div_niches[step]} diversity niches.")
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
            return_scaled=self.obj_bundle.return_scaled,
            scaled_ndim=self.obj_bundle.scaled_ndim,
        )

        algorithm = MGAProblem(
            problem=problem,
            x0=starting_points[0],
            log_dir=log_path,
            log_freq=self.config.mga_log_freq,
            random_seed=None,
            parallelize=False,  # we implement parallelisation independently
            callback=self.callback,
            include_obj_in_fitness=True,
            space_scaler=self.obj_bundle.space_scaler,
        )

        algorithm.starting_points = starting_points

        algorithm.add_niches(num_niches=1)  # pure optimisation

        return algorithm

    def find_starting_points_in_population(self, weighting):
        pop_bin_path = os.path.join(self.scenario.solution_dir, "population.bin")
        pop_csv_path = os.path.join(self.scenario.solution_dir, "population.csv")
        det_bin_path = os.path.join(self.scenario.solution_dir, "details.bin")
        det_csv_path = os.path.join(self.scenario.solution_dir, "details.csv")

        pop_exists = os.path.exists(pop_bin_path) or os.path.exists(pop_csv_path)
        det_exists = os.path.exists(det_bin_path) or os.path.exists(det_csv_path)
        if not pop_exists or not det_exists:
            raise FileNotFoundError(
                f"[MHMGA-E] Could not find population or details files in '{self.scenario.solution_dir}'"
            )

        if os.path.exists(pop_bin_path):
            pop_cols = len(self.scenario.lower_bounds) + 3
            pop = np.fromfile(pop_bin_path, dtype=npfloat).reshape(-1, pop_cols)
        else:
            pop = pd.read_csv(pop_csv_path, header=None).to_numpy()

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
        if SAVE_POPULATION:
            self._log_population_and_details(population)

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
            max_points, min_points, noptimal_lcoe = self.find_starting_points_in_population(weighting)

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
                        extrema_epsilon_rel=self.config.mga_extrema_epsilon_rel[step],
                        extrema_epsilon_abs=self.config.mga_extrema_epsilon_abs[step],
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
                            ),
                            mgat.GradientStagnation(
                                window=10,
                                improvement=0.01*abs_max,
                                maximize=maximize,
                                attribute="current_optimum",
                                attribute_in_population=True,
                            )
                        ]
                    )

        self._finalise_csv_logs()
        self.scenario.logger.info("[MHMGA-E] Binary logs converted to CSV.")


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
        weighting = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
        weighting[i] = 1.0
        weightings.append(weighting)

    return weightings


def construct_unit_type_weightings(scenario):
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
    unit_type_w['solar'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
    unit_type_w['wind'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
    unit_type_w['nuclear_all'] = np.zeros(scenario.lower_bounds.shape, dtype=npfloat)
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

    return [w for w in unit_type_w.values() if w.sum() > 1]


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

    return [w for w in node_w.values() if w.sum() > 1]
