# Module Summary: `firm_ce.backend`

**Source Folder:** [`src/firm_ce/backend/`](./src/firm_ce/backend)

## Overview

The `firm_ce.backend` subpackage implements the operational simulation, chronological unit commitment, transmission balancing, LCOE evaluation, and parallel population evaluation wrappers for `firm_ce`. It is divided into two self-contained backend subpackages selectable via `backend` in `config.csv`:

1. **[`firm_ce.backend.scalar`](./docs/backend/scalar/README.md)**: Object-oriented Numba `@jitclass` simulation engine (`Solution`, `balancing.py`, `single_time.py`, `broad_optimum.py`, `mhmga.py`, `mhmga_extrema.py`).
2. **[`firm_ce.backend.tensor`](./docs/backend/tensor/README.md)**: Vectorised multi-dimensional array simulation engine (`SolutionTensor`, `simulation.py`, `forward_pass.py`, `reverse_pass.py`, `dynamics.py`, `interconnection.py`, `mhmga.py`).

---

## Comparison of `scalar` and `tensor` Backends

| Feature | Scalar Backend (`firm_ce.backend.scalar`) | Tensor Backend (`firm_ce.backend.tensor`) |
| :--- | :--- | :--- |
| **Solution Container** | [`Solution`](./src/firm_ce/backend/scalar/solution.py#L38-L139) | [`SolutionTensor`](./src/firm_ce/backend/tensor/solution.py#L60-L85) |
| **Memory Allocation per Candidate** | Deep-copies dynamic `Fleet` and `Network` `@jitclass` instances per evaluation (`create_dynamic_copy`). | Pre-allocates a thread-local pool of `NUM_THREADS` `SolutionTensor` instances and resets arrays in-place via `ResetSolution` (zero heap allocation in `prange`). |
| **Time Stepping & Precharging** | Year-by-year forward interval loop; triggers reverse-time deficit-block balancing and precharging (`precharge_storage`) whenever a deficit block ends. | Full-horizon passes: `ForwardPassRenewables` $\rightarrow$ `ReversePassHydro` $\rightarrow$ `ForwardPassPeak` $\rightarrow$ `ReversePassPeak` with early feasibility exits and stall detection. |
| **Transmission Routing** | Iterates over pre-enumerated loop-free `Route` lists sorted by leg length and efficiency (`fill_with_transmitted_surpluses`). | Inline Dijkstra maximum-efficiency pathfinding (`Interconnection`) over a CSR adjacency graph, accounting for counter-flow loss cancellation. |
| **Storage Merit Order (Forward)** | Longest-to-shortest duration (`node_m.assign_storage_merit_order`), reversed during precharging. | Shortest-to-longest duration (`bess2h` $\rightarrow$ `bess4h` $\rightarrow$ `clphes` $\rightarrow$ `olphes`, pondage before reservoir) in forward pass; longest-to-shortest donor draining in reverse pass. |
| **Post-Processing Compatibility** | Native `Solution` object consumed directly by `Accessor`, `Statistics`, `Display`, `Behaviour`. | Converted to a scalar `Solution` object via [`map_tensor_to_scalar`](./src/firm_ce/constructors/tensor_to_scalar.py#L19-L186). |

---

## How It Fits into the Project

`firm_ce.backend` is invoked by:
- **[`firm_ce.optimisation.objective_factory`](./src/firm_ce/optimisation/objective_factory.py)**, which binds the appropriate parallel population wrapper (`single_time.evaluate_vectorised_xs`, `scalar.mhmga.*`, or `tensor.mhmga.*`) into an `ObjectiveBundle` for the active solver.
- **[`Scenario.build_and_evaluate_solution`](./src/firm_ce/system/scenario.py#L654-L689)** and **[`Scenario.build_and_evaluate_noptima`](./src/firm_ce/system/scenario.py#L690-L730)**, which evaluate optimal or near-optimal decision vectors for validation, statistics generation, and interactive visualisation.
