# `src/firm_ce/tensor/` — Layer 3B: Tensor Backend

## Summary
The `firm_ce.tensor` subpackage implements the high-throughput **Tensor Backend** of FIRM_CE. Rather than iterating over dictionaries of individual asset `@jitclass` objects, it aggregates nodal capacities, time-series traces, and operational states into contiguous 2D `(T, N)` and 3D `(T, N, K)` NumPy/Numba arrays. This enables SIMD vectorisation, zero-allocation thread-local `SolutionTensor` pooling across `prange` workers, and fast forward/reverse sweep dispatch with Dijkstra pathfinding.

---

## Submodules

### 1. `entities/` — Contiguous Tensor `@jitclass` Containers
- **[`entities/network.py`](./src/firm_ce/tensor/entities/network.py)**:
  - `GenerateTensorNetwork`: Compiles major transmission lines into a compressed CSR-style adjacency list (`neigh_neighbors`, `neigh_lines`, `neigh_offsets`).
- **[`entities/costs.py`](./src/firm_ce/tensor/entities/costs.py)**:
  - `CostTensor`: Pre-calculates nodal fixed (`F*`) and variable (`V*`) cost coefficients per technology and nodal `E_aa_max` bounds for PHES apportionment.
- **[`entities/static.py`](./src/firm_ce/tensor/entities/static.py)**:
  - `StaticTensor`: Immutable scenario-wide arrays (time-series capacity factor matrices `TS*`, nodal loads `Mload`, run-of-river `Mror`, existing capacities `E*`, x-vector slice offsets, CSR network topology, and `CostTensor`).
- **[`entities/assets.py`](./src/firm_ce/tensor/entities/assets.py)**:
  - `AssetTensor`: Maps a candidate decision vector `x` into nodal total capacities (`Cpfix`, `Cpsat`, `Coffw`, `Consw`, `Cpeak`, `Cnuke`, `Cnlte`, `CnphP`, `CnphE`, `CstorageP`, `CstorageE`, `ChydP`, `ChydE`, `Clines`).
- **[`entities/operations.py`](./src/firm_ce/tensor/entities/operations.py)**:
  - `OperationTensor` & `reset_operations`: Pre-allocates and resets all `(T, N)` and `(T, N, K)` operational ledger arrays and Dijkstra/reverse-pass scratch buffers without heap allocation.

### 2. Simulation Engine (`dynamics.py`, `interconnection.py`, `forward_pass.py`, `reverse_pass.py`, `simulation.py`)
- **[`dynamics.py`](./src/firm_ce/tensor/dynamics.py)**: Nodal netload, must-run hydro/open-loop PHES floors, local storage charging/discharging, SOC updates (`UpdateSOCt`), and annual flexible fuel budget management.
- **[`interconnection.py`](./src/firm_ce/tensor/interconnection.py)**: `Interconnection(...)` — greedy maximum-efficiency Dijkstra pathfinding over the CSR transmission graph, accounting for counter-flow loss savings (`1 / eff`) vs forward losses (`eff`).
- **[`forward_pass.py`](./src/firm_ce/tensor/forward_pass.py)**: `ForwardPassRenewables` (chronological VRE, transmission, storage, and hydro dispatch) and `ForwardPassPeak` (chronological biomass, biogas, and OCGT dispatch).
- **[`reverse_pass.py`](./src/firm_ce/tensor/reverse_pass.py)**: `ReversePassHydro` and `ReversePassPeak` — backward sweeps through time to precharge storage before deficit blocks using spare storage/hydro/peaker headroom, with stall detection (`WriteOffStalledHydroNodes`).
- **[`simulation.py`](./src/firm_ce/tensor/simulation.py)**: `Simulate(solution)` — orchestrates the multi-pass sequence with early-exit feasibility checks.

### 3. [`solution.py`](./src/firm_ce/tensor/solution.py)
- **`SolutionTensor` (`@jitclass`)**: Combines `StaticTensor`, `AssetTensor`, `OperationTensor`, and `nodal_energy`.
- **`EvaluateTensor` & `CalculateCost`**: Runs `Simulate(solution)`, accumulates nodal energy components in a single pass, apportions new PHES via `apportion_new_phes`, and computes `total_annual_cost` and `lcoe`.
- **`ExtractDetailsTensor` / `ExtractEnergySpaceTensor` / `EvaluateTensorFromDetails` / `prep_solution_for_postprocessing`**.

### 4. [`wrappers.py`](./src/firm_ce/tensor/wrappers.py)
- **`_eval_mga_population_core`**: Allocates a per-thread pool of `SolutionTensor` instances (`NUM_THREADS`) and evaluates populations in parallel via `ResetSolution` + `EvaluateTensor` with zero per-candidate heap allocation.
- **`mga_tensor_wrapper` / `mga_tensor_wrapper_with_details` / `mga_tensor_wrapper_scaled`**.

### 5. [`to_scalar.py`](./src/firm_ce/tensor/to_scalar.py)
- **`map_tensor_to_scalar(scenario, solutionTensor)`**: Converts an evaluated `SolutionTensor` into a fully populated scalar `Solution` instance so all downstream `Statistics`, `Accessor`, `Display`, and `Behaviour` tools work seamlessly across both backends.
