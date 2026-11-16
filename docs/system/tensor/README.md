# Module Summary: `firm_ce.system.tensor`

**Source Folder:** [`src/firm_ce/system/tensor/`](./src/firm_ce/system/tensor)

## Overview

`firm_ce.system.tensor` defines the vectorised, contiguous-array `@jitclass` data structures that power the high-performance `"tensor"` backend. Instead of iterating over dictionaries of individual asset objects, the tensor representation aggregates capacities, time-series traces, operational states, and unit costs into dense 1D, 2D `(intervals, nodes)`, and 3D `(intervals, nodes, tier)` NumPy arrays with stride-1 inner loops along the node dimension for SIMD auto-vectorisation.

---

## Contents & Key Functionality

### 1. [`static.py`](./src/firm_ce/system/tensor/static.py) — `StaticTensor`
Constructed once per scenario after datafiles are loaded (`Scenario.construct_tensors()`), `StaticTensor` packs all read-only scenario parameters, existing capacities, time-series traces, topology, and cost coefficients into contiguous arrays:
- **Horizon & Demand Metadata**: `resolution`, `allowance_frac`, `allowance` (in GWh), `years`, `years_float`, `intervals`, `year_of_interval`, `nodes`, `nhvi` (major line count), `energy`, `mean_annual_demand_mwh`, `legacy_costs` (annualised build + FOM of legacy hydro/PHES/RoR assets).
- **Tier Dimensions**:
  - `nstor = 4`: Storage tiers (`0: olphes`, `1: clphes/aa_phes/e_phes`, `2: bess4h`, `3: bess2h`).
  - `nhyd = 2`: Hydro tiers (`0: pond`, `1: hydro` reservoir).
  - `npeak = 3`: Dispatchable peak generator tiers (`0: biomass`, `1: biogas`, `2: ocgt`).
- **Time-Series & Nodal Power Traces**:
  - `Mload`, `Mror`, and `Mnetload_mror = (Mload / (1 - internal_loss)) - Mror` of shape `(intervals, nodes)`.
  - Nodal capacity factor traces `TSpfix`, `TSpsat`, `TSoffw`, `TSonsw`, `TSnuke`, `TSnlte`, `TSphes_inflow` of shape `(intervals, nodes)`, and `TShyd_inflow` of shape `(intervals, nodes, nhyd)`.
  - Annual fuel budgets `Bpeak` of shape `(years, npeak - 1)` for biomass and biogas.
- **Existing Capacities**: `Eror`, `Enuke`, `Elines`, `EstorageP`, `EstorageE`, `EhydP`, `EhydE`.
- **Decision Vector ($x$) Offset Chain**:
  - Pre-calculates `(offset, nodes_array, len)` triples for each technology (`pfix`, `psat`, `offw`, `onsw`, `biog`, `biom`, `ocgt`, `nuke`, `nlte`, `php`, `b2p`, `b4p`, `phe`, `lines_offset`) so `AssetTensor` can map any decision vector $x$ into nodal capacities in $O(|x|)$ without dictionary lookups.

### 2. [`network.py`](./src/firm_ce/system/tensor/network.py) — `GenerateTensorNetwork`
- **[`GenerateTensorNetwork(network, Nodel_int)`](./src/firm_ce/system/tensor/network.py#L8-L67)**:
  - `@njit` function that validates line endpoints against `Nodel_int` and builds a **Compressed Sparse Row (CSR)** adjacency representation of the transmission grid (`neigh_neighbors`, `neigh_lines_arr`, `neigh_offsets`).
  - Used by the Dijkstra pathfinder in [`firm_ce.backend.tensor.interconnection`](./src/firm_ce/backend/tensor/interconnection.py) for $O(V^2)$ maximum-efficiency power routing.

### 3. [`costs.py`](./src/firm_ce/system/tensor/costs.py) — `CostTensor` & `apportion_new_phes`
- **[`CostTensor`](./src/firm_ce/system/tensor/costs.py#L116-L247)**:
  - Pre-computes per-node annualised fixed cost coefficients (`Fpfix`, `Fpsat`, `Foffw`, `Fonsw`, `Fbiog`, `Fbiom`, `Fgas`, `Fnuke`, `Fnlte`, `Faa_phP`, `Faa_phE`, `Fe_phP`, `Fe_phE`, `Fb4P`, `Fb2P`, and per-line `Flines` in $\$/\text{GW-yr}$ or $\$/\text{GWh-yr}$) and variable cost coefficients (`Vpfix`, ..., `Vgas`, `Vph`, `Vb4`, `Vb2`, `Vlines`), plus nodal class AA PHES build limits (`E_aa_max`, `P_aa_max`, `P_e_max`).
- **[`apportion_new_phes(tot_p, tot_e, e_aa_max)`](./src/firm_ce/system/tensor/costs.py#L41-L66)**:
  - `@njit(inline="always")` helper that splits combined nodal new-build PHES power (`tot_p`) and energy (`tot_e`) between cheaper **Class AA** (`aa_phes`, up to `e_aa_max`) and **Class E** (`e_phes`, residual energy) at equal storage duration. Used across both tensor and scalar backends and in taxonomy projections.

### 4. [`assets.py`](./src/firm_ce/system/tensor/assets.py) — `AssetTensor`
- **[`AssetTensor(x, static)`](./src/firm_ce/system/tensor/assets.py#L37-L155)**:
  - Converts a candidate decision vector `x` (applying `abs_rel_scaler` if `static.relative_param` is `True`) plus existing capacities from `StaticTensor` into nodal and line capacity arrays: `Cpfix`, `Cpsat`, `Coffw`, `Consw`, `Cpeak` `(nodes, 3)`, `Cnuke`, `Cnlte`, `CnphP`, `CnphE`, `CstorageP` `(nodes, 4)`, `CstorageE` `(nodes, 4)`, `ChydP` `(nodes, 2)`, `ChydE` `(nodes, 2)`, `Clines` `(nhvi,)`, `Clongdur`, and `Cshortdur`.

### 5. [`operations.py`](./src/firm_ce/system/tensor/operations.py) — `OperationTensor` & `reset_operations`
- **[`OperationTensor(static, assets)`](./src/firm_ce/system/tensor/operations.py#L110-L185)**:
  - Holds all mutable time-series operational matrices and scratch buffers for a tensor simulation:
    - Nodal power matrices `(intervals, nodes)`: `Mnetload`, `Munbalanced`, `Mdeficit`, `Mcurtail`, `Mimport`, `Mphes_spill`, `Mnuke` (plus post-processing `Mpfix`, `Mpsat`, `Moffw`, `Monsw`).
    - Line flow matrix `(intervals, nhvi)`: `Tnetflow`.
    - Storage & hydro 3D tensors: `Mdischarge`, `Mcharge`, `Mstorage` `(intervals, nodes, nstor)` and `Mhydro`, `Mreservoir`, `Mhyd_spill` `(intervals, nodes, nhyd)`, initialised at 50% state of charge (`Mstorage_init`, `Mreservoir_init`).
    - Peak dispatch 3D tensor: `Mpeak` `(intervals, nodes, npeak)` and annual budget tracker `remaining_peak_budget` `(years, npeak - 1)`.
    - Pre-allocated scratch buffers for Dijkstra routing (`cap_fwd`, `cap_rev`, `eff_fwd`, `eff_rev`, `eff`, `visited`, `parent_node`, `parent_line`, `path_nodes`, `path_lines`) and reverse-pass precharging/stall tracking (`rolling_deficits`, `surplus_buffer`, `fill_buffer`, `stall_checkpoint`, `stall_counter`, `storage_min_future`, `storage_max_future`, `hydro_min_future`).
- **[`init_mnetload`](./src/firm_ce/system/tensor/operations.py#L12-L35) & [`reset_operations`](./src/firm_ce/system/tensor/operations.py#L187-L231)**:
  - Zero-allocation `@njit` routines that re-initialise an existing `OperationTensor` in-place for a new candidate vector $x$ during parallel MHMGA population evaluations.

---

## How It Fits into the Project

`firm_ce.system.tensor` provides the memory layout and data structures consumed by [`firm_ce.backend.tensor`](./src/firm_ce/backend/tensor). `StaticTensor` is built once from the scalar `Scenario` objects during `Scenario.load_datafiles()`, and `AssetTensor` / `OperationTensor` are wrapped inside `SolutionTensor` for ultra-fast parallel evaluation and post-optimisation validation (`ValidationTensor`).
