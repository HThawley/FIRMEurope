# `src/firm_ce/analysis/` — Layer 5: Headless Post-Processing, Verification & Uncertainty

## Summary
The `firm_ce.analysis` subpackage contains all **headless** (non-plotting) post-processing, physical verification, statistical reporting, diurnal/charging-source analysis, historical ensemble loading, and Monte-Carlo cost uncertainty simulation modules. Plotting and visualisation live in [`src/firm_ce/viz/`](./docs/viz/README.md).

---

## Module Breakdown

### 1. [`accessor.py`](./src/firm_ce/analysis/accessor.py)
- **`class Accessor`**: Unified query interface over an evaluated scalar `Solution` with configurable unit scaling (`"GW"` / `"GWh"` vs `"MW"` / `"MWh"`).
- Provides asset-type predicates, capacity/build getters (`get_capacity`, `get_build_power`, `get_build_energy`), cost getters (`get_all_costs`), efficiency getters, and dynamic `get_<metric>_trace(asset)` / `get_<metric>_gross(asset)` methods for 26 operational metrics across individual assets, nodes, or `"system"`.

### 2. [`validate.py`](./src/firm_ce/analysis/validate.py)
- **`class Validation` & `class ValidationTensor`**: Post-solve physical and operational verifiers for the scalar and tensor backends respectively.
- Verifies 7 categories of invariants (`Decision Variables Bounds`, `Generator Limits`, `Transmission Limits`, `Energy Balance & Flows`, `Storage Constraints`, `Storage Accrual`, `Fuel Limits`) and writes `validation_report.txt` / `tensor_validation_report.txt`.

### 3. [`statistics.py`](./src/firm_ce/analysis/statistics.py)
- **`class Statistics`**: High-performance Polars + PyArrow reporting engine.
- Writes full-resolution temporal traces to `temporal_data.parquet` and generates 16 multi-dimensional CSV summary tables (`nodal_capacity_matrix.csv`, `summary_*.csv`, `capacities_*.csv`, `components_*.csv`, `levelised_cost_*.csv`, `energy_balance_*.csv`, `x_abs.csv`, `x_rel.csv`).

### 4. [`behaviour_calc.py`](./src/firm_ce/analysis/behaviour_calc.py)
- **`class BehaviourAnalyzer`**: Headless operational analysis engine:
  - Calendar/month/season/year interval masking and time-of-day binning.
  - `get_traces(...)` and `compute_daily_average(...)` for diurnal profile extraction.
  - `compute_storage_charging_sources(...)`: Nodal proportional supply-pool attribution of storage charging energy across VRE, flexible generators, other discharging storages, and nodal imports.
  - `compute_capacity_factor_metrics(...)`: Computes duty cycle (% time online), unconditional capacity factor, conditional capacity factor while online, and full-load hours per year.

### 5. [`ensemble.py`](./src/firm_ce/analysis/ensemble.py)
- **`_pareto_frontier_2d(points, maximize_x, maximize_y)`**: Fast $O(M \log M)$ 2D Pareto frontier filter.
- **`class EnsembleDataset`**: Chunked memory-mapped loader for `population.bin` / `.csv` and `details.bin` / `.csv` that filters feasible near-optimal solutions and projects them into unit-type and composite capacity DataFrames (`df_abs`, `df_rel`) via `TechTaxonomy`.

### 6. [`cost_uncertainty.py`](./src/firm_ce/analysis/cost_uncertainty.py)
- **`class MCConfig`, `build_component_matrix`, `monte_carlo`, `run_monte_carlo`**: Decomposes evaluated `SolutionTensor` alternatives into linear cost components (`$/yr`), verifies exact reproduction of `CalculateCost`, and evaluates thousands of common-random-number cost multiplier draws via a single matrix multiplication (`M @ comp.T`) without re-simulating dispatch.
