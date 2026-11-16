# Module Summary: `firm_ce.system.scalar`

**Source Folder:** [`src/firm_ce/system/scalar/`](./src/firm_ce/system/scalar)

## Overview

`firm_ce.system.scalar` defines the Numba `@jitclass` data structures representing the physical, topological, and economic entities of an electricity system for the scalar backend (and serving as the static input data containers from which `StaticTensor` is built, as well as the post-processing output containers for `Analysis`).

Because Numba `@jitclass` objects cannot be safely mutated concurrently across parallel worker threads, these classes use a **static vs. dynamic instance pattern** (`static_instance: bool`):
- **Static instances (`static_instance = True`)** are constructed once per scenario and hold read-only configuration attributes and exogenous time-series arrays (`data`).
- **Dynamic instances (`static_instance = False`)** are created per candidate evaluation inside worker threads via [`firm_ce.fast_methods`](./src/firm_ce/fast_methods), sharing pointers to read-only static arrays while allocating fresh memory for endogenous dispatch, state-of-charge, flow, and cost variables.

---

## Contents & Key Functionality

### 1. [`components.py`](./src/firm_ce/system/scalar/components.py)
Defines the generation and storage asset `@jitclass` structures and their container:
- **[`Fuel`](./src/firm_ce/system/scalar/components.py#L36-L107)**:
  - Stores fuel `id`, `name`, `cost` ($\$/\text{GJ}$ or $\$/\text{MWh}$), `emissions`, `annual_limit` (annual energy budget in GWh/year), `remaining_energy` time-series, and temporary reverse/forward deficit-block and trickle-charging reserve state variables.
- **[`Generator`](./src/firm_ce/system/scalar/components.py#L146-L271)**:
  - Represents a solar, wind, baseload, run-of-river, or flexible dispatchable generator.
  - Holds static metadata (`id`, `order`, `name`, `node`, `fuel`, `line`, `unit_size`, `max_build`, `min_build`, `initial_capacity`, `unit_type`, `unit_type_idx`, `is_flexible`, `near_optimum_check`, `group`, `cost`, `candidate_x_idx`), exogenous capacity factor trace (`data`), and dynamic evaluation state (`new_build`, `capacity`, `dispatch_power`, `flexible_max_t`, `lt_generation`, `unit_lt_hours`, `lt_costs`).
- **[`Storage`](./src/firm_ce/system/scalar/components.py#L331-L418)**:
  - Represents battery energy storage (BESS), pumped hydro energy storage (open-loop, closed-loop, class AA/E), or hydro reservoirs/pondage.
  - Holds static attributes (`initial_power_capacity`, `initial_energy_capacity`, `duration`, `chargeable`, `inflows`, `charge_efficiency`, `discharge_efficiency`, `max_build_p`, `max_build_e`, `min_build_p`, `min_build_e`, `candidate_p_x_idx`, `candidate_e_x_idx`, inflow `data`), dynamic state (`new_build_p`, `new_build_e`, `power_capacity`, `energy_capacity`, `dispatch_power`, `stored_energy`, `discharge_max_t`, `charge_max_t`, `lt_generation`, `lt_costs`), and reverse-time deficit-block/precharging state (`deficit_block_min_storage`, `deficit_block_max_storage`, `precharge_energy`, `trickling_reserves`, `precharge_flag`, `trickling_flag`).
- **[`Fleet`](./src/firm_ce/system/scalar/components.py#L433-L477)**:
  - Top-level container holding Numba `TypedDict`s of `generators`, `storages`, and `fuels` keyed by scenario order/id.

### 2. [`topology.py`](./src/firm_ce/system/scalar/topology.py)
Defines the transmission network topology `@jitclass` structures:
- **[`Node`](./src/firm_ce/system/scalar/topology.py#L62-L186)**:
  - Represents a copper-plate bus/region in the network.
  - Holds operational demand trace (`data`), `mean_demand`, `relative_scaler`, `internal_loss`, `residual_load` (demand minus non-flexible VRE/baseload generation), merit-order index arrays (`storage_merit_order`, `flexible_merit_order`), cumulative dispatch limit arrays (`discharge_max_t`, `charge_max_t`, `flexible_max_t`), transmission state (`fill`, `surplus`, `temp_surplus`, `available_imports`, `imports_exports`), endogenous nodal time-series (`deficits`, `curtail`, `flexible_power`, `storage_power`), and precharging variables.
- **[`Line`](./src/firm_ce/system/scalar/topology.py#L223-L347)**:
  - Represents either a **major line** interconnecting two `Node`s (`node_start` $\leftrightarrow$ `node_end`) or a **minor line** connecting a generator/storage asset to its node.
  - Computes linear transmission efficiency `1 - (loss_factor * length / 1000)` and stores `initial_capacity`, `min_build`, `max_build`, `candidate_x_idx`, `relative_scaler`, `new_build`, `capacity`, interval `flows`, `temp_leg_flows`, `lt_flows`, and `lt_costs`.
- **[`Route`](./src/firm_ce/system/scalar/topology.py#L365-L440)**:
  - Represents a pre-enumerated loop-free path of `legs` steps originating at `initial_node` (fill node) and traversing an ordered `TypedList` of `nodes` and `lines` with `line_directions` ($\pm 1$) and cumulative `efficiency` to a final surplus node.
- **[`Network`](./src/firm_ce/system/scalar/topology.py#L457-L527)**:
  - Container holding `nodes`, `major_lines`, `minor_lines`, `networksteps_max`, and `routes` (a `TypedDict` mapping `(initial_node.order, leg)` to a `TypedList` of `Route`s).

### 3. [`costs.py`](./src/firm_ce/system/scalar/costs.py)
Defines cost parameter and long-term cost accumulator `@jitclass` structures:
- **[`UnitCost`](./src/firm_ce/system/scalar/costs.py#L27-L86)**:
  - Stores exogenous unit cost assumptions (`capex_p`, `capex_e`, `fom`, `vom`, `lifetime`, `discount_rate`, `heat_rate_base`, `heat_rate_incr`, `transformer_capex`) and pre-calculates `annuity_factor`, `fuel_cost_mwh`, and `fuel_cost_h`.
- **[`LTCosts`](./src/firm_ce/system/scalar/costs.py#L100-L112)**:
  - Stores endogenously evaluated annualised cost components (`annualised_build_p`, `annualised_build_e`, `fom`, `vom`, `fuel`) for an asset.

### 4. [`parameters.py`](./src/firm_ce/system/scalar/parameters.py)
Defines scenario-wide static time-horizon and reliability parameters:
- **[`ScenarioParameters`](./src/firm_ce/system/scalar/parameters.py#L30-L71)**:
  - Stores `resolution` (hours/interval), `allowance` (annual unserved energy fraction limit), `first_year`, `final_year`, `year_count`, `leap_year_count`, `year_first_t` (starting interval index of each year), `year_of_interval` (mapping each interval $t$ to its year index), `intervals_count`, `node_count`, `fom_scalar`, `year_float`, `year_energy_demand`, `mean_annual_demand_mwh`, and `demand_sum_mwh`.

---

## How It Fits into the Project

`firm_ce.system.scalar` is the core object model of the repository:
- Instantiated by [`firm_ce.constructors`](./src/firm_ce/constructors) during `Scenario.__init__`.
- Operated on directly by `@njit` functions in [`firm_ce.fast_methods`](./src/firm_ce/fast_methods) and [`firm_ce.backend.scalar`](./src/firm_ce/backend/scalar).
- Read by [`StaticTensor`](./src/firm_ce/system/tensor/static.py) to initialise tensor arrays when `backend == "tensor"`.
- Populated by [`map_tensor_to_scalar`](./src/firm_ce/constructors/tensor_to_scalar.py) after tensor simulation so that [`Accessor`](./src/firm_ce/analysis/accessor.py), [`Statistics`](./src/firm_ce/analysis/statistics.py), [`Display`](./src/firm_ce/analysis/display.py), and [`Behaviour`](./src/firm_ce/analysis/behaviour.py) have a single consistent object interface regardless of which backend ran the simulation.
