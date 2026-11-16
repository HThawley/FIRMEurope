# Module Summary: `firm_ce.common`

**Source Folder:** [`src/firm_ce/common/`](./src/firm_ce/common)

## Overview

The `firm_ce.common` subpackage provides foundational infrastructure shared across the entire `firm_ce` codebase: compile-time constants, precision-aware Numba/NumPy type definitions, seamless Numba JIT toggling for debugging, custom exceptions (including JIT-compatible error raisers), mathematical and string-parsing helpers, and logging configuration.

---

## Contents & Key Functionality

### 1. [`constants.py`](./src/firm_ce/common/constants.py)
Defines global runtime and compile-time constants controlling Numba compilation, floating-point precision, and numerical tolerances:
- **`JIT_ENABLED`** (`bool`, default `True`): Master switch for Numba JIT compilation across `@jitclass` and `@njit` decorators. Setting `JIT_ENABLED = False` allows stepping through all simulation code with the standard Python debugger.
- **`BITDEPTH`** (`int`, default `32`): Controls whether simulation arrays and scalars use 32-bit (`float32`/`int32`) or 64-bit (`float64`/`int64`) precision. When set to `32`, also sets `os.environ["MGA_USE_32BIT"] = "1"` so the external `mga` package matches precision.
- **`EPSILON_FLOAT`, `NP_FLOAT_MAX`, `NP_FLOAT_MIN`, `NP_INT_MAX`**: Machine limits derived from `BITDEPTH`.
- **`PENALTY_MULTIPLIER`** (`1e6`), **`TOLERANCE`** (`1e-6`), **`VALIDATION_TOL`** (`1e-2`): Numerical tolerances and soft-constraint penalty weights.
- **`NUM_THREADS`**: Number of parallel Numba worker threads (read from the `NUM_THREADS` environment variable or `os.cpu_count()`), with `NUMBA_THREADING_LAYER` defaulting to `"workqueue"`.
- **`FASTMATH`** (`True`), **`BOUNDSCHECK`** (`False`), **`LEAPDAYS`** (`False`), **`SAVE_POPULATION`** (`True`).

### 2. [`jit_overload.py`](./src/firm_ce/common/jit_overload.py)
Provides drop-in wrappers around Numba's decorators and parallel primitives based on `JIT_ENABLED`:
- When `JIT_ENABLED` is `True`, re-exports `njit`, `prange`, `get_thread_id`, and `jitclass` directly from `numba`.
- When `JIT_ENABLED` is `False`, provides no-op Python decorators (`jitclass`, `njit`), maps `prange` to `range`, and returns `0` for `get_thread_id()`.

### 3. [`typing.py`](./src/firm_ce/common/typing.py)
Centralises type annotations, Numba container types, and precision-dependent scalar/array types:
- **Precision Aliases** ([lines 89–105](./src/firm_ce/common/typing.py#L89-L105)):
  - `nbfloat`, `nbint`, `nbintp` (Numba types: `float32`/`float64`, `int32`/`int64`, and native 64-bit `int64` indexing).
  - `npfloat`, `npint`, `npintp` (NumPy dtypes: `np.float32`/`np.float64`, `np.int32`/`np.int64`, `np.int64`).
- **Numba Typed Containers & Fallbacks** ([lines 17–88](./src/firm_ce/common/typing.py#L17-L88)):
  - Exports `DictType`, `ListType`, `UniTuple`, `boolean`, `float64`, `float32`, `int64`, `int32`, `unicode_type`, `TypedDict`, and `TypedList` from `numba` when `JIT_ENABLED` is `True`, or lightweight pure-Python mock equivalents when `JIT_ENABLED` is `False`.
- **Composite Type Aliases**: `EvaluationRecord_Type`, `BroadOptimumVars_Type`, and `BandCandidates_Type` for broad-optimum exploration.

### 4. [`exceptions.py`](./src/firm_ce/common/exceptions.py)
Defines custom exceptions and `@njit`-compatible error-raising functions:
- **`ValidationError`**: Raised when model configuration or datafile validation fails.
- **`raise_static_modification_error()`**: `@njit` helper raising `ValueError` if a worker attempts to mutate a static `@jitclass` instance (`static_instance == True`).
- **`raise_getting_unloaded_data_error()`**: `@njit` helper raising `RuntimeError` if an asset's time-series array is accessed while `data_status == False`.
- **`raise_unknown_balancing_type_error()`**: `@njit` helper raising `RuntimeError` for invalid balancing modes.

### 5. [`helpers.py`](./src/firm_ce/common/helpers.py)
Provides general-purpose parsing and `@njit`-compiled numerical utilities:
- **String & Config Parsers**:
  - `parse_comma_separated(value, lower=True)`: Splits comma-separated strings into trimmed lists.
  - `parse_ditherable_hyperparameter(value)`: Parses stepped/dithered hyperparameter strings (e.g., `"(0.1,1.0),0.2"`) into 2D `[lower, upper]` lists for MHMGA steps.
  - `parse_boolean(value)`: Parses boolean strings (`"true"`, `"1"`, `"yes"`, etc.).
  - `flatten_list(values)` & `chain(*iterables)`: Iterable utilities.
- **Zero-Safe Division & Financial Math**:
  - `safe_divide`, `safe_divide2`, `@njit` `njit_safe_divide`, `@njit` `safe_divide_array` (1D/ND ravelled), and `@njit` `safe_divide_2d_1d` (2D array divided by 1D array along axis 1).
  - `@njit` `get_annuity_factor(discount_rate, lifetime)`: Computes the present-value annuity factor $\frac{1 - (1 + r)^{-L}}{r}$ for capital cost annualisation.

### 6. [`logging.py`](./src/firm_ce/common/logging.py)
Manages output directory resolution and Python `logging` setup:
- **`init_results_directory(model_name, logging_flag, results_mode)`**: Resolves the output path under `results/` (`"new"` creates `results/<model_name>_<timestamp>` or `results/temp`; `"latest"` locates the most recent non-temp run directory; or uses a custom path).
- **`init_model_logger(results_dir)`**: Configures a dual-handler `logging.Logger` writing `DEBUG`+ messages to `<results_dir>/log.txt` and `INFO`+ messages to stdout, while suppressing verbose Numba debug logs.

---

## How It Fits into the Project

`firm_ce.common` is the lowest-level dependency layer in the repository. Almost every module imports from `firm_ce.common` to obtain Numba decorators (`@njit`, `@jitclass`), consistent 32/64-bit types (`nbfloat`, `npfloat`, `nbintp`), compile flags (`FASTMATH`, `BOUNDSCHECK`), and parsing/math utilities. Its `JIT_ENABLED` and `BITDEPTH` abstractions allow the entire simulation engine to switch between fast 32-bit/64-bit native execution and pure-Python debugging without changing a single line of business logic.
