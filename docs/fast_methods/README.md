# Module Summary: `firm_ce.fast_methods`

**Source Folder:** [`src/firm_ce/fast_methods/`](./src/firm_ce/fast_methods)

## Overview

The `firm_ce.fast_methods` subpackage implements `@njit`-compiled procedural "pseudo-methods" that operate on the scalar `@jitclass` data structures defined in [`firm_ce.system.scalar`](./src/firm_ce/system/scalar).

Because Numba `@jitclass` methods incur compilation overhead and cannot easily call across circularly dependent class types, `firm_ce` keeps `@jitclass` definitions purely as data containers (`system/scalar/`) and places all compiled operations in standalone `@njit` modules (`*_m.py`) in `fast_methods/`.

---

## Contents & Key Functionality

### 1. [`fleet_m.py`](./src/firm_ce/fast_methods/fleet_m.py) — Fleet Operations
Coordinates operations across all generators, storages, and fuels in a `Fleet` instance:
- **`create_dynamic_copy`**: Deep-copies a static `Fleet` into a worker-local dynamic `Fleet` (`static_instance = False`), linking dynamic `Node`, `Line`, and `Fuel` instances.
- **`build_capacities`**: Applies candidate decision vector `decision_x` to all generators and storages, including piecewise apportionment of nodal new-build PHES into `aa_phes` and `e_phes` via `apportion_new_phes` and setting endogenous storage duration for merit-order sorting.
- **`allocate_memory`, `initialise_stored_energies`, `initialise_annual_limits`**: Prepare endogenous time-series arrays at the start of a simulation or simulation year.
- **Interval & Precharging State Updates**: `update_stored_energies`, `update_remaining_flexible_energies`, `calculate_lt_generations`, `initialise_deficit_block`, `update_deficit_block`, `assign_precharging_values`, `initialise_precharging_flags`, `update_precharging_flags`, `check_precharge_remaining`, `check_trickling_remaining`, `determine_feasible_storage_dispatch`, `determine_feasible_flexible_dispatch`, `calculate_available_storage_dispatch`, `reset_flexible`, `reset_dispatch`, `reset_flexible_reserves`.

### 2. [`generator_m.py`](./src/firm_ce/fast_methods/generator_m.py) — Generator Operations
- **Lifecycle & Capacity**: `create_dynamic_copy`, `build_capacity` (scales `new_build_x_val` by `node.relative_scaler`, updates `capacity`, minor line capacity, and subtracts non-flexible VRE/baseload generation from `node.residual_load` via `update_residual_load`), `load_data`, `unload_data`, `get_data`, `allocate_memory`.
- **Flexible Dispatch & Merit Order**:
  - `set_flexible_max_t`, `set_precharging_max_t`, `set_live_flexible_max_t`, `set_live_trickling_max_t`, `update_node_flexible_max_t`: Compute interval power limits bounded by installed capacity and remaining fuel energy/trickling reserves, updating cumulative nodal `flexible_max_t`.
  - `dispatch`: Dispatches a flexible generator according to its position in `node.flexible_merit_order` to meet remaining `node.netload_t - node.storage_power[interval]`, and depletes `fuel.remaining_energy` via `update_fuel_reserve`.
  - `update_precharge_dispatch`: Adjusts flexible generator output during reverse-time trickle charging.
- **Long-Term Generation & Cost Accounting**: `calculate_lt_generation`, `calculate_variable_costs`, `calculate_fixed_costs`, `get_partial_cost`.

### 3. [`storage_m.py`](./src/firm_ce/fast_methods/storage_m.py) — Storage Operations
- **Lifecycle & Capacity**: `create_dynamic_copy`, `build_capacity`, `build_capacity_abs` (updates `power_capacity`, `energy_capacity`, and minor line capacity), `load_data`, `unload_data`, `get_data`, `allocate_memory`, `initialise_stored_energy` (initialises state of charge at $50\%$ of `energy_capacity`).
- **Forward & Reverse Dispatch**:
  - `set_dispatch_max_t`: Computes interval `discharge_max_t` and `charge_max_t` in forward or reverse time (accounting for inflows, charge/discharge efficiencies, and energy capacity limits) and updates cumulative nodal `discharge_max_t` / `charge_max_t`.
  - `dispatch`: Dispatches storage charging/discharging according to `node.storage_merit_order`.
  - `update_stored_energy`: Updates `stored_energy[interval]` (forward time) or `stored_energy_temp_reverse` (reverse-time deficit block), accounting for inflows and spillage.
- **Precharging & Trickling Reserves**: `initialise_deficit_block`, `update_deficit_block_bounds`, `assign_precharging_reserves`, `initialise_precharging_flags`, `update_precharging_flags`, `set_precharging_max_t`, `calculate_dispatch_energy_update`, `update_precharge_dispatch`, `calculate_available_dispatch`.
- **Long-Term Generation & Cost Accounting**: `calculate_lt_generation`, `calculate_variable_costs`, `calculate_fixed_costs`, `get_partial_cost_power`, `get_partial_cost_energy`.

### 4. [`fuel_m.py`](./src/firm_ce/fast_methods/fuel_m.py) — Fuel Budget & Reserve Operations
- Manages annual fuel constraints (`annual_limit`) and interval `remaining_energy` tracking shared across flexible generators consuming that fuel: `create_dynamic_copy`, `allocate_memory`, `load_data`, `unload_data`, `get_data`, `get_annual_limit`, `initialise_annual_limits`, `initialise_deficit_block`, `update_deficit_block_bounds`, `assign_trickling_reserves`, `initialise_precharging_flags`, `update_precharging_flags`.

### 5. [`network_m.py`](./src/firm_ce/fast_methods/network_m.py) — Network & Transmission Orchestration
- **Lifecycle & Setup**: `create_dynamic_copy`, `build_capacity`, `unload_data`, `allocate_memory`.
- **Merit-Order Assignment**:
  - `assign_storage_merit_orders`: Sorts storages at each node from longest to shortest duration.
  - `assign_flexible_merit_orders`: Sorts flexible generators at each node from lowest to highest marginal cost.
  - `assign_route_merit_orders`: Sorts `Route` lists for each `(node, leg)` pair in descending order of cumulative transmission efficiency.
- **Multi-Leg Route Transmission (`fill_with_transmitted_surpluses`, `set_node_fills_and_surpluses`)**:
  - Configures nodal `fill` and `surplus` quantities for 9 distinct transmission cases (`"surplus"`, `"storage_discharge"`, `"flexible"`, `"storage_charge"`, `"precharging_surplus"`, `"precharging_transfers"`, `"precharging_adjust_storage"`, `"precharging_adjust_surplus"`, `"precharging_adjust_flexible"`).
  - Iterates over route lengths `leg = 0 .. networksteps_max - 1`, computing deliverable flow updates along each `Route` (`calculate_node_flow_updates`), scaling proportionally when available imports exceed nodal `fill` (`scale_flow_updates_to_fill`), and committing line flows and nodal `imports_exports` (`update_transmission_flows`).
- **Precharging & Deficit Helpers**: `calculate_period_unserved_energy`, `calculate_curtail_and_deficit`, `calculate_lt_flows`, `check_precharging_end`, `set_storage_precharge_fills_and_surpluses`, `set_flexible_precharge_fills_and_surpluses`, `adjust_Transmission_for_infeasible_dispatch`.

### 6. [`node_m.py`](./src/firm_ce/fast_methods/node_m.py) — Node Operations
- `create_dynamic_copy`, `load_data` (loads demand trace, applies `demand_multiple`, initialises `residual_load` and `mean_demand`), `unload_data`, `get_data`, `allocate_memory`.
- `initialise_netload_t`, `update_netload_t`, `fill_required`, `surplus_available`, `assign_storage_merit_order`, `assign_flexible_merit_order`, `check_remaining_netload`, `set_imports_exports_temp`, `reset_dispatch_max_t`.

### 7. [`line_m.py`](./src/firm_ce/fast_methods/line_m.py) — Line Operations
- `create_dynamic_copy`, `build_capacity`, `allocate_memory`, `calculate_lt_flow`, `calculate_variable_costs`, `calculate_fixed_costs`, `get_partial_cost`.

### 8. [`route_m.py`](./src/firm_ce/fast_methods/route_m.py) — Route Flow & Loss Mechanics
- `create_dynamic_copy`, `check_contains_line`, `check_contains_node`.
- **`calculate_flow_update` & `update_exports`**:
  - Implements a two-pass (forward source-to-destination pass via `get_forward_outflow` and backward destination-to-source reservation pass via `get_backward_inflow_and_delta`) flow calculation along a multi-leg `Route`.
  - Explicitly models counter-flow cancellation (reversing existing flow on a line recovers transmission losses, whereas pushing new flow incurs `line.efficiency` losses).

### 9. [`ltcosts_m.py`](./src/firm_ce/fast_methods/ltcosts_m.py) — Long-Term Cost Calculations
- `@njit` functions for computing annualised capital build costs (`calculate_annualised_build_power`, `calculate_annualised_build_energy`), fixed O&M (`calculate_fom`), variable O&M (`calculate_vom`, `calculate_vom_generic`), fuel costs (`calculate_fuel`, `calculate_fuel_generic`), partial decision-variable costs (`get_partial_cost_power`, `get_partial_cost_energy`), and asset LCOE (`calculate_lcoe`, `get_total`, `get_fixed`, `get_variable`).

### 10. [`static_m.py`](./src/firm_ce/fast_methods/static_m.py) — ScenarioParameters Operations
- `get_year_t_boundaries(static_instance, year)`: Returns `(first_t, last_t)` interval indices for a given simulation year.
- `set_year_energy_demand`, `unset_year_energy_demand`: Computes annual and total horizon operational demand (`year_energy_demand`, `mean_annual_demand_mwh`, `demand_sum_mwh`).
- `check_reliability_constraint(static_instance, year, year_unserved_energy)`: Checks whether `year_unserved_energy / year_energy_demand[year] <= allowance`.

---

## How It Fits into the Project

`firm_ce.fast_methods` is the low-level computational workhorse of the scalar backend ([`firm_ce.backend.scalar`](./src/firm_ce/backend/scalar)), and is also called during scenario initialisation/data loading ([`firm_ce.constructors`](./src/firm_ce/constructors), [`firm_ce.system.scenario`](./src/firm_ce/system/scenario.py)) and tensor-to-scalar post-processing ([`map_tensor_to_scalar`](./src/firm_ce/constructors/tensor_to_scalar.py)).
