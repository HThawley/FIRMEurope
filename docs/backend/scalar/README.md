# Module Summary: `firm_ce.backend.scalar`

**Source Folder:** [`src/firm_ce/backend/scalar/`](./src/firm_ce/backend/scalar)

## Overview

`firm_ce.backend.scalar` implements the scalar Numba `@jitclass` simulation engine, including the `Solution` container, the chronological unit commitment and reverse-time deficit-block precharging rules (`balancing.py`), and the `@njit(parallel=True)` population evaluation wrappers for Differential Evolution (`single_time.py`, `broad_optimum.py`) and MHMGA (`mhmga.py`, `mhmga_extrema.py`).

---

## Contents & Key Functionality

### 1. [`solution.py`](./src/firm_ce/backend/scalar/solution.py) — `Solution` & Objective Evaluation
- **[`Solution` `@jitclass`](./src/firm_ce/backend/scalar/solution.py#L38-L139)**:
  - Initialised with a candidate decision vector `x`, static `ScenarioParameters`, static `Fleet`, static `Network`, `balancing_type`, and `fixed_costs_threshold`.
  - Creates worker-safe dynamic copies of `network` and `fleet` (`network_m.create_dynamic_copy`, `fleet_m.create_dynamic_copy`), builds candidate capacities (`fleet_m.build_capacities`, `network_m.build_capacity`), allocates endogenous time-series memory, and assigns storage, flexible generator, and route merit orders.
- **[`objective(solution)`](./src/firm_ce/backend/scalar/solution.py#L343-L390) & [`evaluate(solution)`](./src/firm_ce/backend/scalar/solution.py#L511-L528)**:
  1. Computes fixed costs (`calculate_fixed_costs`) and checks `check_fixed_costs(solution)` against `fixed_costs_threshold`. If exceeded, penalises `solution.penalties` and exits early without running unit commitment.
  2. Runs chronological balancing across all simulation years via `balance_residual_load(solution)`. At the end of each year, checks the annual unserved energy constraint (`static_m.check_reliability_constraint`); if breached, adds a scaled unserved energy penalty and exits early.
  3. If feasible, computes total variable costs (`calculate_variable_costs`) and sets `solution.lcoe = solution.annual_cost / solution.static.mean_annual_demand_mwh`.
- **Operational Details & MGA Energy Space Extraction**:
  - `extract_details(solution, details_slice)`: Aggregates long-term generation, operating hours, and flows by `unit_type_idx` into a 1D slice for binary logging.
  - `evaluate_from_details(solution, details_slice)`: Re-evaluates LCOE under modified cost assumptions without re-running time-series balancing.
  - `extract_energy_space_scalar(...)`: Projects gross generation, storage discharge, and nodal curtailment into the target MGA energy fitness space (`AGG_UNIT_TYPE`, `AGG_NODE`, or `AGG_NONE` with relative or absolute normalisation).
  - `build_eval_and_return_solutions(...)`: Parallel batch evaluator returning a `TypedDict` of evaluated `Solution` instances for post-processing.

### 2. [`balancing.py`](./src/firm_ce/backend/scalar/balancing.py) — Chronological Balancing & Deficit-Block Precharging
Implements the rule-based unit commitment and transmission dispatch algorithm:
- **[`energy_balance_for_interval(solution, interval, forward_time_flag)`](./src/firm_ce/backend/scalar/balancing.py#L198-L275)**:
  Executes the 6-step merit-order dispatch sequence for a single time step $t$:
  1. Transmit surplus VRE/baseload generation between nodes to fill deficits (`balance_with_transmission(..., "surplus")`).
  2. Dispatch local nodal storage systems (`balance_with_storage`).
  3. Transmit power from neighbouring storage systems (`balance_with_transmission(..., "storage_discharge")`) and re-balance storage.
  4. Dispatch local nodal flexible generators (`balance_with_flexible`).
  5. Transmit power from neighbouring flexible generators (`balance_with_transmission(..., "flexible")`) and re-balance flexible generators.
  6. Transmit any remaining nodal surplus to charge storage systems at neighbouring nodes (`balance_with_transmission(..., "storage_charge")`).
- **[`balance_for_period(start_t, end_t, precharging_allowed, solution, year)`](./src/firm_ce/backend/scalar/balancing.py#L277-L340) & [`precharge_storage`](./src/firm_ce/backend/scalar/balancing.py#L1120-L1188)**:
  - Steps forward through $t \in [\text{start\_t}, \text{end\_t})$.
  - When `precharging_allowed` is `True` (`balancing_type == "full"`) and a contiguous block of unserved energy (a **deficit block**) ends, calls `precharge_storage(solution, t, year)`:
    1. **Reverse-Time Deficit Block Assessment (`determine_precharge_energies_for_deficit_block`)**: Steps backwards through the deficit block to determine the energy required by each storage precharger and the reserves needed by trickle chargers.
    2. **Reverse-Time Precharging Period (`precharge_in_precharging_period`)**: Steps backwards prior to the deficit block, transferring local surpluses (`perform_local_surplus_transfers`), transmitted surpluses (`perform_transmitted_surplus_transfers`), intranode/internode storage trickle transfers (`perform_intranode_interstorage_transfers`, `perform_internode_interstorage_transfers`), and intranode/internode flexible generator trickle transfers (`perform_intranode_flexible_transfers`, `perform_internode_flexible_transfers`) into storage prechargers.
    3. **Forward Continuity Restoration (`resolve_precharging_discontinuity`)**: Steps forward from the start of the precharging period back to interval $t$ to ensure all storage state-of-charge and fuel reserve trajectories remain continuous and feasible.

### 3. [`single_time.py`](./src/firm_ce/backend/scalar/single_time.py) — Differential Evolution Population Wrapper
- **[`parallel_wrapper`](./src/firm_ce/backend/scalar/single_time.py#L14-L57) & [`evaluate_vectorised_xs`](./src/firm_ce/backend/scalar/single_time.py#L60-L97)**:
  - Evaluates a 2D SciPy Differential Evolution population `xs` of shape `(ndim, n_points)` in parallel via `prange(n_points)`, returning `lcoe + penalties * PENALTY_MULTIPLIER` for each candidate and printing per-iteration timing diagnostics.

### 4. [`broad_optimum.py`](./src/firm_ce/backend/scalar/broad_optimum.py) — Near-Optimal Banding & Midpoint Exploration
- Supports the `"near_optimum"` and `"midpoint_explore"` solver modes (`BroadOptimumSolver`):
  - `build_broad_optimum_var_info`, `create_groups_dict`: Group decision variables by `range_group` for Assets with `near_optimum_check == True`.
  - `broad_optimum_objective`: Evaluates candidate populations via `parallel_wrapper`, applies a soft LCOE band penalty `max(0, lcoe - band_lcoe_max) * PENALTY_MULTIPLIER`, records feasible near-optimal solutions, and returns objectives for minimising (`"min"`), maximising (`"max"`), or targeting a specific midpoint sum (`"midpoint"`) of grouped capacities.
  - `write_broad_optimum_records`, `write_broad_optimum_bands`, `read_broad_optimum_bands`, `create_midpoint_csv`, `append_to_midpoint_csv`: Read and write CSV outputs under `results/near_optimum/` and `results/midpoint_explore/`.

### 5. [`mhmga.py`](./src/firm_ce/backend/scalar/mhmga.py) & [`mhmga_extrema.py`](./src/firm_ce/backend/scalar/mhmga_extrema.py) — MHMGA Population Wrappers
- **[`mhmga.py`](./src/firm_ce/backend/scalar/mhmga.py)**:
  - `_eval_mga_scalar_core`: Parallel `@njit(parallel=True)` loop over `n_points` candidates `xs` `(n_points, ndim)`, computing `lcoe`, `penalties`, optional `details` (`extract_details`), and optional energy-space coordinates (`extract_energy_space_scalar`).
  - Exposes three wrapper signatures matching the `mga` package interface: `mga_wrapper`, `mga_wrapper_with_details`, and `mga_scalar_wrapper_scaled`.
- **[`mhmga_extrema.py`](./src/firm_ce/backend/scalar/mhmga_extrema.py)**:
  - Provides `apply_new_weighting`, `mga_wrapper_with_details`, and `mga_wrapper` for `MhmgaExtremaSolver`, evaluating directional linear projections `xs @ weighting` subject to an LCOE constraint `lcoe <= lcoe_constr`.

---

## How It Fits into the Project

`firm_ce.backend.scalar` is the reference operational simulation engine of `firm_ce`. It is called by `firm_ce.optimisation` during scalar-backend optimisations and by `Scenario.build_and_evaluate_solution` / `Statistics` when evaluating individual solutions on the scalar backend.
