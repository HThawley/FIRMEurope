# Module Summary: `firm_ce.io`

**Source Folder:** [`src/firm_ce/io/`](./src/firm_ce/io)

## Overview

The `firm_ce.io` subpackage handles all disk input/output, configuration schema validation, time-series trace loading and temporal aggregation, and result file serialisation (supporting CSV, Excel `.xlsx`, and Apache Parquet formats via Pandas and Polars).

---

## Contents & Key Functionality

### 1. [`file_manager.py`](./src/firm_ce/io/file_manager.py)
Provides classes and functions for reading input CSVs and writing formatted result tables:
- **[`ImportCSV`](./src/firm_ce/io/file_manager.py#L13-L76) & [`import_config_csvs`](./src/firm_ce/io/file_manager.py#L397-L409)**:
  - Reads the nine standard configuration CSVs (`scenarios.csv`, `nodes.csv`, `generators.csv`, `fuels.csv`, `lines.csv`, `storages.csv`, `config.csv`, `initial_guess.csv`, `datafiles.csv`) from `config_directory`, indexes rows by `id`, normalises string columns to lowercase (preserving case for `filename` and `units`), and returns nested dictionaries.
- **[`ImportDatafile`](./src/firm_ce/io/file_manager.py#L78-L116) & [`DataFile`](./src/firm_ce/io/file_manager.py#L119-L140)**:
  - Loads a time-series or annual-constraint CSV from `data_directory` and converts each column into a 1D NumPy `NDArray` keyed by lowercase column name, alongside `datafile_type` and `units` metadata.
- **[`ResultFile`](./src/firm_ce/io/file_manager.py#L143-L395)**:
  - Unified output wrapper accepting `pd.DataFrame`, `pl.DataFrame`, or `pl.LazyFrame` objects.
  - Supports configurable decimal rounding (`_apply_rounding`), multi-row delimited headers (`multiindex_delimiter`), and exporting via `.write_csv()`, `.write_xlsx()`, and `.write_parquet()` (using streaming `.sink_csv()` / `.sink_parquet()` when backed by a Polars `LazyFrame`).

### 2. [`data_model.py`](./src/firm_ce/io/data_model.py)
Defines [`ModelData`](./src/firm_ce/io/data_model.py#L10-L47) and comprehensive validation routines for all configuration files:
- **`ModelData` Class**:
  - Loads raw configuration dictionaries via `import_config_csvs`, extracts `model_name`, exposes attributes (`scenarios`, `nodes`, `generators`, `fuels`, `lines`, `storages`, `config`, `x0s`, `datafiles`), and provides `.validate()`.
- **Component & Topology Validators** ([lines 84–605](./src/firm_ce/io/data_model.py#L84-L605)):
  - `validate_config(model_data)` orchestrates:
    - `validate_model_config`: Validates solver parameters, backend selection (`"scalar"` or `"tensor"`), parameterisation (`"relative"` or `"absolute"`), balancing type (`"simple"` or `"full"`), and 3-axis MGA space settings (`mga_space_basis`, `mga_space_norm`, `mga_space_agg`).
    - `validate_scenarios`: Checks uniqueness of scenario names, positive `resolution`, `allowance` in $[0, 1]$, and integer or `"auto"` `firstyear`/`finalyear`.
    - `validate_nodes`, `validate_fuels`, `validate_lines`, `validate_generators`, `validate_storages`: Verify non-negative costs/capacities, `min_build <= max_build`, valid efficiencies/discount rates in $[0, 1]$, and cross-reference integrity (every generator/storage/line references a valid node, fuel, and minor/major line defined in that scenario).
    - `validate_initial_guess` & `validate_data`: Ensure `initial_guess.csv` and `datafiles.csv` entries match defined scenarios and files on disk.
- **MHMGA Hyperparameter Schema & Broadcasting** ([lines 679–774](./src/firm_ce/io/data_model.py#L679-L774)):
  - `expected_mga_hyperparameters` defines defaults, ditherability, broadcastability, and type/range constraints for 23 `mga_*` settings.
  - `parse_and_validate_mga` parses scalar, comma-separated stepped lists, and dithered `(lower, upper)` pairs, broadcasting them across `mga_steps` and writing the cast Python/NumPy objects back into the config dictionary.

### 3. [`aggregation.py`](./src/firm_ce/io/aggregation.py)
Provides temporal downsampling for time-series traces:
- **[`aggregate_trace(trace, factor, method)`](./src/firm_ce/io/aggregation.py#L4-L38)**:
  - Downsamples a 1D NumPy time-series array by an integer `factor` (`interval_aggregation`) using either `"mean"` (for capacity factors and demand power traces) or `"sum"` (for hydro energy inflows), handling non-divisible remainder intervals gracefully.

---

## How It Fits into the Project

`firm_ce.io` forms the boundary between external files on disk and `firm_ce`'s internal data structures:
1. During `Model.__init__`, [`ModelData`](./src/firm_ce/io/data_model.py#L10-L47) and [`ImportCSV`](./src/firm_ce/io/file_manager.py#L13-L76) ingest and validate all configuration CSVs before any `Scenario` or `@jitclass` object is constructed.
2. During `Scenario.load_datafiles()`, [`DataFile`](./src/firm_ce/io/file_manager.py#L119-L140) and [`aggregate_trace`](./src/firm_ce/io/aggregation.py#L4-L38) supply trimmed, unit-scaled, and temporally aggregated NumPy arrays to [`firm_ce.constructors.traces_cons`](./src/firm_ce/constructors/traces_cons.py).
3. During post-processing, [`Statistics`](./src/firm_ce/analysis/statistics.py) uses [`ResultFile`](./src/firm_ce/io/file_manager.py#L143-L395) to write all summary tables and Parquet datasets to the scenario's results directory.
