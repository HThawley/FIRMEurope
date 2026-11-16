# Module Summary: `firm_ce.model` (`model.py`)

**Source File:** [`src/firm_ce/model.py`](./src/firm_ce/model.py)

## Overview

`firm_ce.model` defines the [`Model`](./src/firm_ce/model.py#L16-L290) class, which serves as the primary top-level user interface for configuring, validating, solving, and post-processing long-term electricity system planning optimisations in the FIRM framework.

---

## Contents & Key Functionality

### `Model` Class ([`src/firm_ce/model.py`](./src/firm_ce/model.py#L16-L290))

#### 1. Initialisation (`Model.__init__`)
- **Inputs:**
  - `config_directory` (`str`, default `"inputs/config"`): Path to the folder containing model configuration CSVs (`config.csv`, `scenarios.csv`, `nodes.csv`, `generators.csv`, `storages.csv`, `fuels.csv`, `lines.csv`, `initial_guess.csv`, `datafiles.csv`).
  - `data_directory` (`str`, default `"inputs/data"`): Path to the folder containing exogenous time-series CSV files.
  - `logging_flag` (`bool`, default `True`): Controls whether a timestamped results directory (`results/<model_name>_<timestamp>`) is created or whether logs are written to `results/temp`.
  - `**kwargs`: Optional overrides applied to `model_data.config_data`.
- **Workflow:**
  1. Instantiates [`ModelData`](./src/firm_ce/io/data_model.py#L10-L47) to load all configuration CSVs.
  2. Initialises the results directory via [`init_results_directory`](./src/firm_ce/common/logging.py#L9-L30), archives a copy of `config.csv` as `config_archive.csv`, and configures the model logger via [`init_model_logger`](./src/firm_ce/common/logging.py#L33-L69).
  3. Runs `model_data.validate()` to check schema, numeric bounds, cross-references, and MHMGA hyperparameter broadcasting. Raises [`ValidationError`](./src/firm_ce/common/exceptions.py#L4-L5) if validation fails.
  4. Wraps validated configuration settings in [`ModelConfig`](./src/firm_ce/system/config.py#L8-L72).
  5. Constructs a dictionary of [`Scenario`](./src/firm_ce/system/scenario.py#L35-L772) instances (`self.scenarios`) keyed by scenario name. Exogenous time-series data is **not** loaded during `__init__` to conserve memory when multiple scenarios are defined.

#### 2. Solving Scenarios (`Model.solve`, `Model.single_time`, `Model.mhmga`)
- **`solve(*, scenarios="all")`** ([lines 214–255](./src/firm_ce/model.py#L214-L255)):
  - Iterates over the requested scenario(s) (accepts `"all"`, a single scenario name string, or a list of scenario names).
  - Dispatches each scenario to either `self.single_time(scenario)` (when `config.type == "single_time"`) or `self.mhmga(scenario)` (when `config.type == "mhmga"`).
- **`single_time(scenario)`** ([lines 103–156](./src/firm_ce/model.py#L103-L156)) & **`mhmga(scenario)`** ([lines 158–212](./src/firm_ce/model.py#L158-L212)):
  - Manage the full per-scenario execution lifecycle:
    1. **Load Datafiles:** Calls `scenario.load_datafiles()` to load time-series traces into static objects (and construct `StaticTensor` if `backend == "tensor"`).
    2. **Run Optimisation:** Calls `scenario.solve()` which delegates to the appropriate solver in [`firm_ce.optimisation.solver`](./src/firm_ce/optimisation/solver.py).
    3. **Build & Evaluate Optimal Solution:** Calls `scenario.build_and_evaluate_solution(...)` on the optimal decision vector.
    4. **Validate Solution & Generate Statistics:**
       - For `backend == "tensor"`, runs [`ValidationTensor`](./src/firm_ce/analysis/validate.py#L298-L496) and [`Statistics`](./src/firm_ce/analysis/statistics.py#L20-L1198) to write summary CSVs and `temporal_data.parquet`.
       - For `backend == "scalar"`, runs [`Validation`](./src/firm_ce/analysis/validate.py#L114-L296) and [`Statistics`](./src/firm_ce/analysis/statistics.py#L20-L1198).
    5. **Unload Datafiles:** Calls `scenario.unload_datafiles()` to free memory before the next scenario begins.

#### 3. MHMGA Recombination Inspection (`Model.inspect_recombination`)
- **`inspect_recombination(...)`** ([lines 257–290](./src/firm_ce/model.py#L257-L290)):
  - Loads datafiles for a named scenario, runs a single-step MHMGA recombination/mutation diagnostic on a supplied `starting_population` via `scenario.inspect_mhmga_recombination(...)` for hyperparameter tuning, unloads datafiles, and returns the offspring inspection dictionary.

---

## How It Fits into the Project

`Model` is the top-level orchestrator used by user scripts (such as those in `examples/`) and integration tests (`tests/integration/`). It bridges:
- **Input & Configuration Validation** ([`firm_ce.io.data_model`](./src/firm_ce/io/data_model.py) and [`firm_ce.system.config`](./src/firm_ce/system/config.py)),
- **Scenario Construction & Memory Management** ([`firm_ce.system.scenario`](./src/firm_ce/system/scenario.py)), and
- **Post-Optimisation Verification & Reporting** ([`firm_ce.analysis.validate`](./src/firm_ce/analysis/validate.py) and [`firm_ce.analysis.statistics`](./src/firm_ce/analysis/statistics.py)).
