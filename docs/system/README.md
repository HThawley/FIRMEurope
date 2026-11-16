# Module Summary: `firm_ce.system`

**Source Folder:** [`src/firm_ce/system/`](./src/firm_ce/system)

## Overview

The `firm_ce.system` subpackage defines the core representation of an energy system planning problem. At its root, it contains [`config.py`](./src/firm_ce/system/config.py) (`ModelConfig`) and [`scenario.py`](./src/firm_ce/system/scenario.py) (`Scenario`), and branches into two backend-specific data-structure subpackages:
- [`firm_ce.system.scalar`](./docs/system/scalar/README.md): Object-oriented Numba `@jitclass` representations of individual assets, nodes, lines, routes, costs, and parameters.
- [`firm_ce.system.tensor`](./docs/system/tensor/README.md): Vectorised, contiguous multi-dimensional array `@jitclass` representations (`StaticTensor`, `AssetTensor`, `OperationTensor`, `CostTensor`) for high-throughput simulation.

---

## Contents & Key Functionality

### 1. [`config.py`](./src/firm_ce/system/config.py) — `ModelConfig`
Transforms the validated `config.csv` dictionary from `ModelData` into a structured `ModelConfig` object:
- **Core Settings**:
  - `type`: Optimisation mode (`"single_time"`, `"near_optimum"`, `"midpoint_explore"`, `"mhmga"`, `"mhmga_extrema"`).
  - `backend`: Simulation engine (`"scalar"` or `"tensor"`).
  - `parameterisation`: Decision variable coordinate space (`"absolute"` in GW/GWh or `"relative"` scaled by nodal mean demand).
  - `balancing_type`: `"full"` (with reverse-time precharging) or `"simple"`.
  - `fixed_costs_threshold`: Maximum allowable fixed cost intensity ($\$/\text{MWh}$ of demand) before early rejection.
  - `limit_timesteps`, `demand_multiple`, `interval_aggregation`, `restart_optimisation`, `save_details`, `model_location`.
- **Mode-Specific Hyperparameters**:
  - **Single-Time DE**: `iterations`, `population`, `mutation`, `recombination`.
  - **Broad Optimum**: `near_optimal_tol`, `midpoint_count`.
  - **MHMGA**: `mga_steps`, the 3-axis MGA fitness space configuration (`mga_space_basis` $\in \{\text{"capacity"}, \text{"energy"}\}$, `mga_space_norm` $\in \{\text{"relative"}, \text{"absolute"}\}$, `mga_space_agg` $\in \{\text{"unit\_type"}, \text{"node"}, \text{"none"}\}$), and all 23 broadcasted `mga_*` hyperparameters from `expected_mga_hyperparameters`.

### 2. [`scenario.py`](./src/firm_ce/system/scenario.py) — `Scenario`
Encapsulates a single planning scenario, its static `@jitclass` system objects, decision-variable indexing, coordinate transformations, data loading/unloading, and evaluation/solving entry points:
- **System Object Construction (`__init__`)**:
  - Filters `nodes`, `lines`, `generators`, `storages`, and `fuels` for the scenario and constructs static `@jitclass` instances via [`firm_ce.constructors`](./src/firm_ce/constructors): `self.network` (`Network`), `self.static` (`ScenarioParameters`), and `self.fleet` (`Fleet`).
- **Canonical Decision Vector Indexing (`assign_indices_and_get_abs_bounds`, [lines 89–178](./src/firm_ce/system/scenario.py#L89-L178))**:
  - Assigns `candidate_x_idx` (and `candidate_p_x_idx` / `candidate_e_x_idx` for storages) in a strict canonical order that aligns with `StaticTensor` offset chains:
    1. Candidate generators grouped by `unit_type` (`pv_fixed`, `pv_track`, `offw`, `onsw`, `biogas`, `biomass`, `ocgt`, `nuclear`, `nuclear_lte`).
    2. Candidate new-build PHES power capacity (`aa_phes` + `e_phes` combined per node as `new_phes`).
    3. Candidate battery power capacity (`bess2h`, then `bess4h`).
    4. Candidate new-build PHES energy capacity (`new_phes` per node).
    5. Candidate major transmission lines.
  - Builds `self.asset_node_map`, `self.lower_bounds_abs`, and `self.upper_bounds_abs`.
- **Relative vs. Absolute Space & Initial Guess (`construct_relative_space`, `_approximate_feasible_solution`, [lines 180–267, 416–472](./src/firm_ce/system/scenario.py#L180-L267))**:
  - Computes `abs_rel_scaler` (scaling nodal assets by `node.mean_demand` and major lines by `max(node_start.mean_demand, node_end.mean_demand)`).
  - Provides `convert_x_to_rel(x_abs)` and `convert_x_to_abs(x_rel)`.
  - Parses `initial_guess.csv` (`_get_x0`) or synthesises a heuristic initial solution (`_approximate_feasible_solution`) when `x_0` is empty.
- **MGA Capacity Space Projection (`build_capacity_space_scaler`, [lines 551–635](./src/firm_ce/system/scenario.py#L551-L635))**:
  - Builds a linear projection matrix (`space_scaler`) mapping the raw decision vector $x$ into the target MGA niche diversity space according to `mga_space_norm` (`"relative"` vs. `"absolute"`) and `mga_space_agg` (`"unit_type"`, `"node"`, or `"none"`).
- **Data Lifecycle (`load_datafiles`, `unload_datafiles`, `construct_tensors`, [lines 272–375](./src/firm_ce/system/scenario.py#L272-L375))**:
  - Loads time-series traces into `network` and `fleet`, computes annual energy demand totals, builds the relative space and `StaticTensor` (when `backend == "tensor"`), and unloads all arrays (`gc.collect()`) when finished.
- **Solving & Solution Evaluation (`solve`, `build_and_evaluate_solution`, `build_and_evaluate_noptima`, `recover_logs`, `polish`, [lines 654–772](./src/firm_ce/system/scenario.py#L654-L772))**:
  - Instantiates the appropriate solver via `get_solver(self)` and evaluates single solutions or MGA ensembles (`noptima`) on either the scalar or tensor backend (mapping `SolutionTensor` back to scalar `Solution` via `map_tensor_to_scalar` for unified downstream analysis).

### 3. Subpackages
- **[`firm_ce.system.scalar`](./docs/system/scalar/README.md)**: Scalar `@jitclass` definitions (`components.py`, `costs.py`, `parameters.py`, `topology.py`).
- **[`firm_ce.system.tensor`](./docs/system/tensor/README.md)**: Tensor `@jitclass` definitions (`assets.py`, `costs.py`, `network.py`, `operations.py`, `static.py`).

---

## How It Fits into the Project

`firm_ce.system` is the central domain model of the codebase. `Scenario` owns the static representation of the grid and fleet, coordinates data loading, defines the canonical ordering and scaling of the optimisation decision vector $x$, and acts as the primary object passed into solvers (`firm_ce.optimisation`) and analysis tools (`firm_ce.analysis`).
