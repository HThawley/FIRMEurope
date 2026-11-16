# Module Summary: `firm_ce.backend.tensor`

**Source Folder:** [`src/firm_ce/backend/tensor/`](./src/firm_ce/backend/tensor)

## Overview

`firm_ce.backend.tensor` implements the high-throughput vectorised simulation engine for `firm_ce`. Operating on the contiguous array structures defined in [`firm_ce.system.tensor`](./docs/system/tensor/README.md), it executes chronological energy balancing across the entire modelling horizon via a sequence of forward and reverse sweeps (`Simulate`), routes nodal surpluses and deficits over the transmission grid using an inline Dijkstra maximum-efficiency pathfinder (`Interconnection`), and evaluates MHMGA populations in parallel using a zero-allocation thread-local pool of `SolutionTensor` buffers.

---

## Contents & Key Functionality

### 1. [`solution.py`](./src/firm_ce/backend/tensor/solution.py) — `SolutionTensor` & Cost/Space Evaluation
- **[`SolutionTensor` `@jitclass`](./src/firm_ce/backend/tensor/solution.py#L60-L85) & [`ResetSolution`](./src/firm_ce/backend/tensor/solution.py#L92-L110)**:
  - Wraps decision vector `x`, `static` (`StaticTensor`), `assets` (`AssetTensor`), `operations` (`OperationTensor`), pre-allocated nodal summary array `nodal_energy` `(nodes, 14)`, PHES split buffers (`nodal_aaph_gen`, `nodal_eph_gen`), and scalar outputs (`estimated_deficit`, `penalties`, `feasible`, `total_annual_cost`, `lcoe`).
  - `ResetSolution(solution, x)` rebuilds `AssetTensor` and zeroes `OperationTensor` in-place via `reset_operations` without allocating new time-series arrays on the heap.
- **[`EvaluateTensor(solution)`](./src/firm_ce/backend/tensor/solution.py#L345-L358) & [`CalculateCost(solution)`](./src/firm_ce/backend/tensor/solution.py#L112-L208)**:
  - Runs `Simulate(solution)` to determine operational feasibility, dispatch, and `estimated_deficit`.
  - Performs a single $T \times N$ pass to accumulate nodal generation, storage discharge, curtailment, and imports into `solution.nodal_energy`, apportions new-build PHES (`apportion_new_phes`) and nuclear/nuclear-LTE generation, and sums fixed + variable + legacy costs to compute `total_annual_cost` and `lcoe`.
- **Fast Post-Evaluation Extractors**:
  - `ExtractDetailsTensor(solution, details_slice)`: $O(\text{nodes})$ extraction of system-wide generation totals by unit type from `solution.nodal_energy` (zero time-series iteration).
  - `ExtractEnergySpaceTensor(solution, nodal_demand_totals, norm_mode, agg_mode, scaled_slice)`: $O(\text{nodes})$ projection of `solution.nodal_energy` into the 3-axis MGA energy fitness space (`AGG_UNIT_TYPE`, `AGG_NODE`, or `AGG_NONE`, relative or absolute).
  - `EvaluateTensorFromDetails(x, static, details_slice)`: Reconstructs LCOE directly from `x` and `details_slice` without allocating `SolutionTensor` or `OperationTensor`.
  - `prep_solution_for_postprocessing` & `build_eval_and_return_solutions`: Populate explicit VRE dispatch traces (`Mpfix`, `Mpsat`, `Moffw`, `Monsw`) and batch-evaluate solutions in parallel for post-processing.

### 2. [`simulation.py`](./src/firm_ce/backend/tensor/simulation.py) — Horizon Simulation Pipeline (`Simulate`)
Orchestrates the 4-stage operational simulation for a `SolutionTensor`:
1. **Stage 1 — Renewables, Hydro & Storage Forward Pass**: Resets annual peak fuel budgets (`ResetAnnualBudgets`) and runs [`ForwardPassRenewables(solution)`](./src/firm_ce/backend/tensor/forward_pass.py#L19-L88).
2. **Stage 2 — Hydro & Storage Reverse Trickle Pass**: Runs [`ReversePassHydro(solution)`](./src/firm_ce/backend/tensor/reverse_pass.py#L14-L48) to precharge short-duration storages using surplus energy from longer-duration storages and hydro reservoirs, then updates state-of-charge trajectories via `UpdateDynamics(solution)`. Exits early (`feasible = True`) if no deficits remain.
3. **Stage 3 — Peak Generators Forward Pass**: Runs [`ForwardPassPeak(solution)`](./src/firm_ce/backend/tensor/forward_pass.py#L90-L99) to dispatch biomass, biogas, and OCGT gas generators directly against remaining nodal deficits. Exits early (`feasible = True`) if `total_deficit <= allowance`.
4. **Stage 4 — Sound Upper-Bound Pre-Check & Peak Reverse Trickle Pass**:
   - Computes `total_peak_headroom = GetTotalPeakHeadroomBound(solution)`. If `total_deficit > total_peak_headroom * 0.9`, marks the solution infeasible (`feasible = False`, `estimated_deficit = total_deficit - total_peak_headroom * 0.8`) and exits early without running the expensive reverse peak pass.
   - Otherwise, runs [`ReversePassPeak(solution)`](./src/firm_ce/backend/tensor/reverse_pass.py#L50-L82) and `UpdateDynamics(solution)` to use spare peak generation in earlier intervals to precharge storages ahead of deficits, then checks final feasibility `final_deficit <= allowance`.

### 3. [`forward_pass.py`](./src/firm_ce/backend/tensor/forward_pass.py) — Chronological Forward Sweeps
- **[`ForwardPassRenewables(solution)`](./src/firm_ce/backend/tensor/forward_pass.py#L19-L88)**:
  For each interval $t = 0 \dots T-1$:
  1. Computes initial unbalanced load and locks in must-run hydro/open-loop PHES spill-prevention floors (`UpdateUnbalancedt`, `GetNaiveCurtailDeficit`).
  2. Transmits network curtailment to fill deficits (`Interconnection`), then charges/discharges local storage (`UpdateLocalCharge`, `UpdateLocalDischarge`).
  3. Transmits available short-duration storage surplus (`GetShortDurSurplust`: pondage, `bess2h`, `bess4h`) across the grid to fill deficits.
  4. Transmits available long-duration storage surplus (`GetLongDurSurplust`: closed-loop/new PHES, open-loop PHES, seasonal hydro reservoir) across the grid to fill deficits.
  5. Transmits any remaining curtailment to charge available storage headroom (`GetForwardStorageHeadroom`) at other nodes, and commits end-of-interval SOC (`UpdateSOCt`).
- **[`ForwardPassPeak(solution)`](./src/firm_ce/backend/tensor/forward_pass.py#L90-L99)**:
  For each interval $t$ with remaining deficits, dispatches peak tiers $k \in \{0:\text{biomass}, 1:\text{biogas}, 2:\text{ocgt}\}$ locally (`LocalDispatchPeakTier`) and then across the transmission network (`NetworkDispatchPeakTier`).

### 4. [`reverse_pass.py`](./src/firm_ce/backend/tensor/reverse_pass.py) — Reverse-Time Precharging & Stall Detection
- **[`ReversePassHydro(solution)`](./src/firm_ce/backend/tensor/reverse_pass.py#L14-L48) & [`ReversePassPeak(solution)`](./src/firm_ce/backend/tensor/reverse_pass.py#L50-L82)**:
  - Sweep backwards from $t = T - 1$ down to $0$, accumulating unresolved deficits into `rolling_deficits[n]` and tracking future minimum/maximum SOC bounds (`storage_min_future`, `storage_max_future`, `hydro_min_future`) so precharging never causes future over-charge or under-discharge violations.
  - At each step $t$, `SetupPrechargePools` identifies storage charging headroom (shortest duration first) needed to cover `rolling_deficits`, and `TrickleStorageHydro` / `TricklePeak` supply that precharge demand locally and via `Interconnection` from longer-duration storage/hydro donors (`SetupStorageDonors`) or spare peak generator headroom (`LocalTrickleTier`, `NetworkTrickleTier`).
  - Includes windowed stall detection (`WriteOffStalledHydroNodes`, `CheckNodeStalls` over `STALL_WINDOW = 24` intervals and `STALL_PATIENCE = 14` windows) to abandon fruitless precharge backlogs on stranded nodes.

### 5. [`dynamics.py`](./src/firm_ce/backend/tensor/dynamics.py) — Nodal Dispatch & State-of-Charge Primitives
Contains `@njit(inline="always")` kernels used by `forward_pass.py`, `reverse_pass.py`, and `simulation.py`:
- `UpdateUnbalancedt`, `GetHydroMinGen`, `GetPhesMinGen`, `GetNaiveCurtailDeficit`.
- `UpdateLocalCharge` (charges shortest-duration storage first: `bess2h` $\rightarrow$ `bess4h` $\rightarrow$ `clphes` $\rightarrow$ `olphes`).
- `UpdateLocalDischarge` (discharges pondage $\rightarrow$ `bess2h` $\rightarrow$ `bess4h` $\rightarrow$ `clphes` $\rightarrow$ `olphes` $\rightarrow$ seasonal hydro reservoir).
- `GetShortDurSurplust`, `GetLongDurSurplust`, `CommitTrickle`, `UpdateSOCt`.
- `ResetAnnualBudgets`, `GetPeakHeadroom`, `DispatchPeak`, `GetTotalPeakHeadroomBound`.

### 6. [`interconnection.py`](./src/firm_ce/backend/tensor/interconnection.py) — Dijkstra Maximum-Efficiency Transmission Router
- **[`Interconnection(solution, Fillt, Surplust, netflowt, Importt)`](./src/firm_ce/backend/tensor/interconnection.py#L6-L204)**:
  - Resolves nodal deficits (`Fillt`) from nodal surpluses (`Surplust`) across the transmission network at time step $t$.
  - Models bidirectional line capacities (`cap_fwd`, `cap_rev`) and effective marginal efficiencies (`eff_fwd`, `eff_rev`): when a transfer opposes existing flow on a line (`netflowt`), it cancels counter-flow with effective efficiency $1 / \eta_{\text{line}} > 1$ up to the zero-crossing point before pushing new flow at efficiency $\eta_{\text{line}} < 1$.
  - Iteratively selects the node with the largest remaining `Fillt[n]` and runs Dijkstra's algorithm over the CSR graph (`neigh_offsets`, `neigh_neighbors`, `neigh_lines`) to find the highest cumulative-efficiency path to a surplus node, pushing the bottleneck-constrained flow and updating `netflowt`, `Importt`, `Fillt`, and `Surplust` in-place.

### 7. [`mhmga.py`](./src/firm_ce/backend/tensor/mhmga.py) — Thread-Pooled MHMGA Population Wrappers
- **[`_eval_mga_population_core`](./src/firm_ce/backend/tensor/mhmga.py#L17-L64)**:
  - Pre-allocates a `TypedList` pool of `n_workers = min(NUM_THREADS, n_points)` `SolutionTensor` instances before entering the `prange(n_points)` loop.
  - Inside `prange`, each worker thread retrieves its private `SolutionTensor` via `pool[get_thread_id()]`, calls `ResetSolution(sol, xs[j])`, `EvaluateTensor(sol)`, and optionally `ExtractDetailsTensor` and `ExtractEnergySpaceTensor`.
- Exposes `mga_tensor_wrapper`, `mga_tensor_wrapper_with_details`, and `mga_tensor_wrapper_scaled` for [`firm_ce.optimisation.objective_factory`](./src/firm_ce/optimisation/objective_factory.py).

---

## How It Fits into the Project

`firm_ce.backend.tensor` is the primary high-speed simulation engine for large multi-year European grid optimisations and MHMGA ensemble generation. It consumes `StaticTensor` from `firm_ce.system.tensor`, is invoked by `firm_ce.optimisation` during solving, is verified by `ValidationTensor` in `firm_ce.analysis.validate`, and maps its evaluated `SolutionTensor` results into scalar `Solution` objects via `map_tensor_to_scalar` for downstream analysis and plotting.
