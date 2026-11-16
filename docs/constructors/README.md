# Module Summary: `firm_ce.constructors`

**Source Folder:** [`src/firm_ce/constructors/`](./src/firm_ce/constructors)

## Overview

The `firm_ce.constructors` subpackage contains factory and adapter functions that bridge Python configuration dictionaries, time-series `DataFile` objects, and Numba `@jitclass` instances. Because Numba `@jitclass` `__init__` methods cannot accept arbitrary Python `dict` objects or perform complex graph/calendar preprocessing, `firm_ce.constructors` handles type casting, object linking, route enumeration, time-series loading/unloading, and post-evaluation tensor-to-scalar conversion.

---

## Contents & Key Functionality

### 1. [`component_cons.py`](./src/firm_ce/constructors/component_cons.py)
Constructs static `@jitclass` instances for generation and storage assets:
- **[`construct_Fuel_object(fuel_dict)`](./src/firm_ce/constructors/component_cons.py#L22-L42)**: Casts `id`, `name`, `cost`, and `emissions` and returns a static `Fuel` instance.
- **[`construct_Generator_object(...)`](./src/firm_ce/constructors/component_cons.py#L45-L126)**: Resolves the generator's `Node`, `Fuel`, and minor `Line` references, builds its `UnitCost` object via `construct_UnitCost_object`, and returns a static `Generator` instance.
- **[`construct_Storage_object(...)`](./src/firm_ce/constructors/component_cons.py#L129-L218)**: Resolves initial power/energy capacity and storage `duration` (computing energy from `power * duration` when `duration > 0`, or deriving duration from `energy / power` when `duration == 0`), links the `Node`, minor `Line`, and `UnitCost`, and returns a static `Storage` instance.
- **[`construct_Fleet_object(...)`](./src/firm_ce/constructors/component_cons.py#L221-L279)**: Populates Numba `TypedDict`s of `generators`, `storages`, and `fuels` and returns a static `Fleet` instance.

### 2. [`cost_cons.py`](./src/firm_ce/constructors/cost_cons.py)
- **[`construct_UnitCost_object(...)`](./src/firm_ce/constructors/cost_cons.py#L8-L62)**: Extracts `fuel.cost` (if a `Fuel` is attached) and instantiates a [`UnitCost`](./src/firm_ce/system/scalar/costs.py#L27-L86) `@jitclass` instance for a `Generator`, `Storage`, or `Line`.

### 3. [`parameter_cons.py`](./src/firm_ce/constructors/parameter_cons.py)
Calculates calendar and time-horizon indices and constructs `ScenarioParameters`:
- **[`determine_interval_parameters(first_year, year_count, resolution)`](./src/firm_ce/constructors/parameter_cons.py#L13-L53)**: Computes the number of leap days (respecting `LEAPDAYS`), the starting interval index of each simulated year (`year_first_t`), and the total `intervals_count`.
- **[`determine_year_of_interval(year_first_t, intervals_count)`](./src/firm_ce/constructors/parameter_cons.py#L56-L81)**: Maps every interval $t \in [0, \text{intervals\_count})$ to its zero-indexed simulation year via `np.searchsorted`.
- **[`construct_ScenarioParameters_object(...)`](./src/firm_ce/constructors/parameter_cons.py#L84-L148)**: Applies optional `limit_timesteps` and `interval_aggregation` scaling to `resolution`, `intervals_count`, and `year_first_t`, and returns a static [`ScenarioParameters`](./src/firm_ce/system/scalar/parameters.py#L30-L71) instance.

### 4. [`topology_cons.py`](./src/firm_ce/constructors/topology_cons.py)
Constructs the transmission network topology and pre-enumerates loop-free multi-leg transmission routes:
- **[`construct_Node_object`](./src/firm_ce/constructors/topology_cons.py#L22-L39) & [`construct_Line_object`](./src/firm_ce/constructors/topology_cons.py#L42-L127)**: Build static `Node` and `Line` `@jitclass` instances (assigning a placeholder `MINOR_NODE` with `order = -1` to minor lines that lack `node_start` / `node_end`).
- **[`construct_new_Route_object`](./src/firm_ce/constructors/topology_cons.py#L130-L162), [`extend_route`](./src/firm_ce/constructors/topology_cons.py#L165-L201), [`get_routes_for_node`](./src/firm_ce/constructors/topology_cons.py#L204-L261), [`build_routes_typed_dict`](./src/firm_ce/constructors/topology_cons.py#L264-L301)**:
  - Breadth-first path enumeration up to `networksteps_max` legs for every `initial_node`, filtering out node and line loops via `route_m.check_contains_node` and `route_m.check_contains_line`, and storing the resulting `TypedList[Route]` in a `TypedDict` keyed by `(node.order, leg)`.
- **[`construct_Network_object`](./src/firm_ce/constructors/topology_cons.py#L304-L349)**: Separates lines into `major_lines` and `minor_lines`, builds the `routes` dictionary, and returns a static `Network` instance.

### 5. [`traces_cons.py`](./src/firm_ce/constructors/traces_cons.py)
Manages loading, slicing, unit conversion, temporal aggregation, and unloading of exogenous time-series arrays:
- **[`select_datafile(...)`](./src/firm_ce/constructors/traces_cons.py#L27-L77)**:
  - Locates the matching column (`object_name`) in `datafiles_imported_dict` for a given `datafile_type` (`"demand"`, `"generation"`, `"inflow"`, `"fuel_constraint"`).
  - Slices the array by `limit_timesteps` (`trim_with_timesteps`) or `(firstyear, finalyear)` (`trim_with_years`).
  - Scales values to internal GW/GWh/CF units via `unit_multiples` (`"MW" -> 0.001`, `"GW" -> 1.0`, `"TW" -> 1000.0`, `"CF" -> 1.0`).
  - Downsamples via [`aggregate_trace`](./src/firm_ce/io/aggregation.py#L4-L38) when `interval_aggregation > 1`.
- **`load_datafiles_to_network`, `load_datafiles_to_generators`, `load_datafiles_to_fuels`, `load_datafiles_to_storages`**: Populate static `@jitclass` instances via `fast_methods` (`node_m.load_data`, `generator_m.load_data`, `fuel_m.load_data`, `storage_m.load_data`).
- **`unload_data_from_network`, `unload_data_from_generators`, `unload_data_from_fuels`, `unload_data_from_storages`**: Replace loaded time-series arrays with empty arrays to reclaim memory after a scenario finishes.

### 6. [`tensor_to_scalar.py`](./src/firm_ce/constructors/tensor_to_scalar.py)
- **[`map_tensor_to_scalar(scenario, solutionTensor)`](./src/firm_ce/constructors/tensor_to_scalar.py#L19-L186)**:
  - Converts an evaluated [`SolutionTensor`](./src/firm_ce/backend/tensor/solution.py#L60-L85) into a fully populated scalar [`Solution`](./src/firm_ce/backend/scalar/solution.py#L38-L139) instance.
  - Disaggregates nodal tier capacities, dispatch traces, storage state-of-charge traces (apportioning new-build PHES via `apportion_new_phes` and sharing tier dispatch proportionally by asset capacity), major line flows, nodal deficits/curtailment/imports, and long-term asset costs (`lt_costs`).

---

## How It Fits into the Project

`firm_ce.constructors` is the glue layer between Python I/O (`firm_ce.io`), the scenario manager (`firm_ce.system.scenario`), the Numba `@jitclass` representations (`firm_ce.system.scalar` and `firm_ce.system.tensor`), and post-processing (`firm_ce.analysis`). In particular, `map_tensor_to_scalar` enables the entire `firm_ce.analysis` suite (`Accessor`, `Statistics`, `Display`, `Behaviour`) to operate on solutions produced by the tensor backend without duplicating post-processing code.
