# `src/firm_ce/io/` — Layer 1: Disk I/O, Validation, Traces & Binary Logs

## Summary
The `firm_ce.io` subpackage manages all disk input/output operations: importing CSV config tables and time-series datafiles, validating input schemas prior to optimisation, loading/aggregating time-series traces into fleet and network entities, writing Polars/Pandas result files, and streaming/recovering binary optimisation logs (`population.bin`, `details.bin`).

---

## Module Breakdown

### 1. [`file_manager.py`](./src/firm_ce/io/file_manager.py)
- **`ImportCSV` / `ImportDatafile`**: Reads configuration CSVs into nested dictionaries and time-series CSVs into column-indexed NumPy arrays.
- **`DataFile`**: Wraps an imported time-series CSV along with its `type` (`demand`, `generation`, `inflow`, `fuel_constraint`) and physical `units`.
- **`ResultFile`**: Writes Polars `LazyFrame`/`DataFrame` or Pandas `DataFrame` outputs to CSV (supporting multi-index column headers delimited by `|`).

### 2. [`data_model.py`](./src/firm_ce/io/data_model.py)
- **`class ModelData`**: Lightweight container holding the raw imported config dictionaries (`scenarios`, `generators`, `fuels`, `lines`, `storages`, `config`, `x0s`, `datafiles`) and delegating pre-solve validation to `config_validation.validate_model_data(self)`.

### 3. [`config_validation.py`](./src/firm_ce/io/config_validation.py)
- Pre-solve schema and cross-reference validators (`validate_config`, `validate_scenarios`, `validate_fuels`, `validate_lines`, `validate_generators`, `validate_storages`, `validate_initial_guess`, `validate_datafiles`) ensuring all node, line, fuel, and scenario references are consistent before any JIT compilation or data loading occurs.

### 4. [`traces.py`](./src/firm_ce/io/traces.py)
- **`aggregate_trace(trace, factor, method)`**: Downsamples 1D time-series arrays by an integer `interval_aggregation` factor using `"mean"` or `"sum"`.
- **`select_datafile(...)`**: Matches an entity name and datafile type to its imported `DataFile`, slices by `limit_timesteps` or `(firstyear, finalyear)`, applies unit scaling (`G`, `M`, `k`, `T`, `C`), and aggregates intervals.
- **`load_datafiles_to_network` / `load_datafiles_to_generators` / `load_datafiles_to_fuels` / `load_datafiles_to_storages`** and corresponding **`unload_data_from_*`** functions: Populate and clear time-series arrays on static `@jitclass` instances.

### 5. [`binary_logs.py`](./src/firm_ce/io/binary_logs.py)
- **`write_mhmga_config_summary(...)`**: Writes `mhmga_config.json` containing MHMGA hyperparameters, bounds, and column counts.
- **`log_population_and_details(...)`**: Appends each generation's population and operational details as contiguous `float32` binary blocks to `population.bin` / `details.bin` and updates `latest_population.npy` / `latest_details.npy`.
- **`convert_binary_logs_to_csv(...)`**: Stream-converts `.bin` and `.npy` logs to `.csv` in memory-mapped chunks.
- **`recover_interrupted_run(out_dir)`**: Repairs torn trailing bytes in `.bin` files after an interrupted run, converts them to `.csv`, and reconstructs `current_optimum.csv`.
