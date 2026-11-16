# `src/firm_ce/optimisation/` — Layer 4: Search Algorithms & Objective Routing

## Summary
The `firm_ce.optimisation` subpackage bridges `Scenario` configuration and the `scalar/` and `tensor/` evaluation backends. It routes objective function bundles according to solver type, backend, and MGA fitness space configuration, and provides solver classes for SciPy Differential Evolution (`single_time`) and Multi-Niche Hierarchical MGA (`mhmga`).

---

## Module Breakdown

### 1. [`objective_factory.py`](./src/firm_ce/optimisation/objective_factory.py)
- **`class ObjectiveBundle`**: Frozen dataclass packaging `(func, fargs, return_scaled, space_scaler, scaled_ndim, details)`.
- **`build_objective_bundle(scenario, config)`**: Routes to the appropriate scalar or tensor population wrapper based on `config.type` (`single_time` vs `mhmga`), `config.backend` (`scalar` vs `tensor`), `config.save_details`, and the 3-axis MGA space configuration (`mga_space_basis`, `mga_space_norm`, `mga_space_agg`).

### 2. [`base.py`](./src/firm_ce/optimisation/base.py)
- **`graceful_sigint_handler(algorithm, logger)`**: Context manager that intercepts `SIGINT` (`Ctrl+C`) during Numba execution, sets `algorithm._stop_requested = True`, and waits for the current iteration to finish cleanly before converting binary logs to CSV (with 3x `Ctrl+C` force-kill).
- **`class BaseSolver(ABC)`**: Abstract base class that builds the `ObjectiveBundle` and initialises callback/binary log files in `scenario.solution_dir`.
- **`get_solver(scenario, initial_population=None)`**: Factory function returning `SingleTimeSolver` or `MhmgaSolver`.

### 3. [`de_solver.py`](./src/firm_ce/optimisation/de_solver.py)
- **`class DeSolverBase` & `class SingleTimeSolver`**: Wraps `scipy.optimize.differential_evolution` with deferred vectorized population evaluation, per-iteration `callback.csv` / `current_optimum.csv` / `population.bin` logging, and final CSV conversion.

### 4. [`mhmga_solver.py`](./src/firm_ce/optimisation/mhmga_solver.py)
- **`class MhmgaSolverBase` & `class MhmgaSolver`**: Orchestrates multi-step `MGAProblem` runs (optimisation niche + optional extrema niches + stepped diversity niches), checkpoint restarting from `population.bin` / `latest_population.npy`, recombination inspection (`inspect_mhmga_recombination`), `mga_alternatives.csv` export, and binary log finalisation.

### 5. [`solver.py`](./src/firm_ce/optimisation/solver.py)
- Convenience facade re-exporting `BaseSolver`, `DeSolverBase`, `SingleTimeSolver`, `MhmgaSolverBase`, `MhmgaSolver`, `get_solver`, `graceful_sigint_handler`, `convert_binary_logs_to_csv`, and `recover_interrupted_run`.
