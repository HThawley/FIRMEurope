# `src/firm_ce/system/` — Layer 2: Domain Configuration, Scenario, Space & Taxonomy

## Summary
The `firm_ce.system` subpackage represents the backend-agnostic domain layer of FIRM_CE. It parses model and MHMGA hyperparameters, manages the `Scenario` lifecycle, defines the canonical decision-variable (`x`-vector) ordering and relative/absolute coordinate transformations, provides shared `@njit` cost and PHES apportionment math, and implements the `TechTaxonomy` projection engine.

---

## Module Breakdown

### 1. [`config.py`](./src/firm_ce/system/config.py)
- **`expected_mga_hyperparameters()` & `parse_and_validate_mga(config_dict, logger)`**: Defines and validates all multi-step MHMGA hyperparameters (`mga_iter`, `mga_pop_size`, `mga_space_basis`, `mga_space_norm`, `mga_space_agg`, etc.).
- **`class ModelConfig`**: Typed configuration object storing solver settings (`type`, `backend`, `balancing_type`, `parameterisation`, `fixed_costs_threshold`, `limit_timesteps`, `interval_aggregation`, `demand_multiple`, DE settings, and MHMGA step arrays).

### 2. [`parameters.py`](./src/firm_ce/system/parameters.py)
- **`ScenarioParameters` (`@jitclass`)**: Holds static horizon and calendar parameters (`resolution`, `first_year`, `final_year`, `year_count`, `year_float`, `intervals_count`, `year_first_t`, `year_of_interval`, `allowance`, `annual_generation_check`, `mean_annual_demand_mwh`, `demand_sum_mwh`).
- **`construct_ScenarioParameters_object(...)`**: Computes leap-year-aware interval counts and calendar boundaries from `scenarios.csv`.

### 3. [`costs_common.py`](./src/firm_ce/system/costs_common.py)
- **`get_annuity_factor(discount_rate, lifetime)`**: Present-value annuity factor calculation.
- **`get_generator_costs` / `get_storage_costs` / `get_line_costs`**: `@njit` unit-cost helpers shared by `CostTensor` and `map_tensor_to_scalar`.
- **`apportion_new_phes(tot_p, tot_e, e_aa_max)`**: `@njit` piecewise apportionment of shared nodal new-build PHES power and energy between Class AA (`aa_phes`, up to `e_aa_max`) and Class E (`e_phes`) at equal storage duration. Used identically by both the scalar and tensor backends.

### 4. [`taxonomy.py`](./src/firm_ce/system/taxonomy.py)
- **Pure domain taxonomy (no Matplotlib dependencies)**:
  - `SUBTECH_MAP`, `COMPOSITE_GROUPS`, `identify_tech(name)`, `get_subtech_label(asset)`, `resolve_asset_label(asset, group_by, ...)`.
  - **`class TechTaxonomy`**: Builds linear projection matrices (`P_abs`, `offset_abs`) and vectorised piecewise PHES apportionment to project `(N, n_vars)` decision matrices into unit-type and composite capacity columns in either `"abs"` (GW/GWh) or `"rel"` (nodal demand multiple) space.

### 5. [`space.py`](./src/firm_ce/system/space.py)
- **`assign_indices_and_get_abs_bounds(fleet, network)`**: Assigns canonical decision-variable indices (`candidate_x_idx`, `candidate_p_x_idx`, `candidate_e_x_idx`) and builds `asset_node_map`.
- **`construct_relative_space(...)`**: Computes `lower_bounds_rel`, `upper_bounds_rel`, and `abs_rel_scaler` by normalising each asset's bounds by its nodal mean demand.
- **`approximate_feasible_solution(...)`**: Heuristic initial guess (`x0`) generator when no entry is provided in `initial_guess.csv`.
- **`assign_unit_type_idx(fleet, network, backend)`**: Assigns 1D slice offsets for `details.bin` operational logging.
- **`build_capacity_space_scaler(scenario, space_norm, space_agg)`**: Builds the `(n_vars, k_dims)` projection matrix used by MHMGA for niche distance calculations in capacity space (`"none"`, `"unit_type"`, or `"node"`).

### 6. [`scenario.py`](./src/firm_ce/system/scenario.py)
- **`class Scenario`**: Represents a single capacity expansion scenario. Owns `network`, `static`, `fleet`, `staticTensor` (when `backend == "tensor"`), bounds, `x0`, and provides lifecycle methods (`load_datafiles`, `unload_datafiles`, `solve`, `build_and_evaluate_solution`, `build_and_evaluate_noptima`, `recover_logs`, `polish`).
