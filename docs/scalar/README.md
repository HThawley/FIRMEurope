# `src/firm_ce/scalar/` — Layer 3A: Scalar Backend (Entities, Methods, Balancing & Solution)

## Summary
The `firm_ce.scalar` subpackage contains the complete **Scalar Backend** of FIRM_CE: object-oriented Numba `@jitclass` domain entities (`entities/`), `@njit` pseudo-methods (`methods/`), chronological interval balancing and reverse-time deficit-block precharging (`balancing.py`), the scalar `Solution` evaluator (`solution.py`), and parallel population wrappers (`wrappers.py`).

Furthermore, all post-processing, validation (`Validation`), statistics (`Statistics`), and visualisation (`Display`, `Behaviour`) tools operate on the scalar `Solution` representation (with tensor solutions mapped via `tensor/to_scalar.py`).

---

## Submodules

### 1. `entities/` — `@jitclass` Domain Entities & Constructors
- **[`entities/costs.py`](./src/firm_ce/scalar/entities/costs.py)**:
  - `UnitCost`: Capital, fixed O&M, variable O&M, fuel cost, discount rate, lifetime, and annuity factor per asset.
  - `LTCosts`: Long-term annualised build, FO&M, VO&M, and fuel cost accumulators.
  - `construct_UnitCost_object`: Parses asset cost parameters from CSV dictionaries.
- **[`entities/components.py`](./src/firm_ce/scalar/entities/components.py)**:
  - `Fuel`, `Generator`, `Storage`, and `Fleet` `@jitclass` definitions plus `construct_Fleet_object`.
- **[`entities/topology.py`](./src/firm_ce/scalar/entities/topology.py)**:
  - `Node`, `Line`, `Route`, and `Network` `@jitclass` definitions plus multi-leg route enumeration (`construct_Network_object`).

### 2. `methods/` — `@njit` Fast Pseudo-Methods
Because Numba `@jitclass` methods incur boxing/dispatch overhead when called across nested containers, all hot-loop operations on scalar entities are implemented as standalone `@njit` functions taking the `@jitclass` instance as their first argument:
- [`ltcosts_m.py`](./src/firm_ce/scalar/methods/ltcosts_m.py): Fixed and variable cost calculations (`calculate_annualised_build_power`, `calculate_fom`, `calculate_vom`, `calculate_fuel`).
- [`fuel_m.py`](./src/firm_ce/scalar/methods/fuel_m.py): Annual fuel budget initialisation, depletion, and deficit-block trickling reserve tracking.
- [`node_m.py`](./src/firm_ce/scalar/methods/node_m.py): Nodal demand/residual load management, merit-order sorting (`assign_storage_merit_order`, `assign_flexible_merit_order`), and netload updates.
- [`line_m.py`](./src/firm_ce/scalar/methods/line_m.py) & [`route_m.py`](./src/firm_ce/scalar/methods/route_m.py): Transmission line capacity expansion, route flow updates, and multi-leg efficiency scaling.
- [`generator_m.py`](./src/firm_ce/scalar/methods/generator_m.py) & [`storage_m.py`](./src/firm_ce/scalar/methods/storage_m.py): Dynamic copying, capacity expansion, interval dispatch, state-of-charge updates, and precharge/trickle constraints.
- [`fleet_m.py`](./src/firm_ce/scalar/methods/fleet_m.py) & [`network_m.py`](./src/firm_ce/scalar/methods/network_m.py): Fleet-wide and network-wide orchestration of dynamic copies, capacity builds (including `apportion_new_phes`), transmission fills/surpluses, and deficit-block state resets.
- [`static_m.py`](./src/firm_ce/scalar/methods/static_m.py): Annual unserved energy allowance checks and calendar year boundary lookups.

### 3. [`balancing.py`](./src/firm_ce/scalar/balancing.py)
- **`energy_balance_for_interval(solution, interval, forward_time_flag)`**: 6-step merit-order unit commitment rule sequence per interval (surplus transmission -> local storage -> neighbouring storage -> local flexible -> neighbouring flexible -> surplus storage charging).
- **`precharge_storage(solution, interval_after_deficit_block, year)`**: 3-phase reverse-time deficit-block resolution (backward deficit block sweep -> backward precharging period trickle transfers -> forward energy discontinuity resolution).

### 4. [`solution.py`](./src/firm_ce/scalar/solution.py)
- **`Solution` (`@jitclass`)**: Encapsulates a candidate `x` vector alongside dynamic copies of `Fleet` and `Network`.
- **`evaluate(solution)` / `objective(solution)`**: Evaluates fixed costs, checks `fixed_costs_threshold`, runs `balance_residual_load`, and computes variable costs and system LCOE (`$/MWh`).
- **`extract_details` / `evaluate_from_details` / `extract_energy_space_scalar` / `build_eval_and_return_solutions`**: Post-processing and MGA fitness space helpers.

### 5. [`wrappers.py`](./src/firm_ce/scalar/wrappers.py)
- **`evaluate_vectorised_xs` & `parallel_wrapper`**: Embarrassingly parallel `prange` population evaluator for SciPy Differential Evolution.
- **`mga_wrapper`, `mga_wrapper_with_details`, `mga_scalar_wrapper_scaled`**: Parallel population evaluators for MHMGA.
