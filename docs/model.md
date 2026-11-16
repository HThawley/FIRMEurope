# `src/firm_ce/model.py` — Top-Level Model Orchestrator

## Summary
[`src/firm_ce/model.py`](./src/firm_ce/model.py) defines the `Model` class, the primary user-facing entry point for building, validating, solving, and post-processing FIRM_CE capacity expansion scenarios.

---

## Contents & Key Functionality

### `class Model`
- **`__init__(config_directory="inputs/config", data_directory="inputs/data", logging_flag=True, **kwargs)`**:
  1. Instantiates [`ModelData`](./src/firm_ce/io/data_model.py) to load all CSV configuration files (`config.csv`, `scenarios.csv`, `generators.csv`, `storages.csv`, `fuels.csv`, `lines.csv`, `datafiles.csv`, `initial_guess.csv`).
  2. Applies any runtime keyword overrides (`**kwargs`) to `model_data.config_data`.
  3. Initialises the model results directory (`results/<model_name>_<timestamp>` or `results/temp`), archives `config.csv`, and creates the model logger.
  4. Validates all configuration schemas via `model_data.validate()`, raising `ValidationError` on failure.
  5. Constructs [`ModelConfig`](./src/firm_ce/system/config.py) and a dictionary of [`Scenario`](./src/firm_ce/system/scenario.py) objects keyed by `scenario_name`.
- **`_run_scenario(scenario, label)`**:
  - Shared execution pipeline used by both `single_time(scenario)` and `mhmga(scenario)`:
    1. Calls `scenario.load_datafiles()` to load time-series traces into static objects (and build `StaticTensor` if `backend == "tensor"`).
    2. Calls `scenario.solve()` to run the configured optimiser (`SingleTimeSolver` or `MhmgaSolver`).
    3. Evaluates the optimal decision vector via `scenario.build_and_evaluate_solution(best_x)`.
    4. Runs physical and operational verification (`Validation` for scalar or `ValidationTensor` for tensor) and writes `validation_report.txt`.
    5. Generates and writes CSV/Parquet summary tables via `Statistics`.
    6. Calls `scenario.unload_datafiles()` to release time-series memory before the next scenario.
- **`single_time(scenario)` / `mhmga(scenario)`**:
  - Convenience wrappers delegating to `_run_scenario`.
- **`solve(scenarios="all")`**:
  - Iterates over the requested scenario(s) and dispatches to `single_time` or `mhmga` according to `self.config.type`.
- **`inspect_recombination(scenario_name, starting_population, ...)`**:
  - Loads datafiles for a target scenario, executes a single pass of MHMGA recombination/mutation via `scenario.inspect_mhmga_recombination(...)` for hyperparameter tuning, and unloads datafiles.
