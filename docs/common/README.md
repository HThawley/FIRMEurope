# `src/firm_ce/common/` — Layer 0: Shared Utilities, Types & JIT Overloads

## Summary
The `firm_ce.common` subpackage is the foundational layer of the codebase with zero internal dependencies on other `firm_ce` subpackages. It centralises runtime flags, floating-point precision types, Numba `@njit`/`@jitclass` decorators (with pure-Python fallbacks when `JIT_ENABLED = False`), custom exceptions, array division helpers, and logging setup.

---

## Module Breakdown

### 1. [`constants.py`](./src/firm_ce/common/constants.py)
- **`JIT_ENABLED`**: Boolean controlling whether Numba JIT compilation is active (`True` by default).
- **`FASTMATH`, `BOUNDSCHECK`, `PARALLEL`, `INLINE`**: Compiler flags passed to `@njit`.
- **`NUM_THREADS`**: Thread count read from `NUMBA_NUM_THREADS` environment variable or CPU core count.
- **`PRECISION`**: Floating-point precision (`"float32"` or `"float64"`).
- **`TOLERANCE`, `VALIDATION_TOL`, `PENALTY_MULTIPLIER`, `LEAPDAYS`, `SAVE_POPULATION`**: Numerical tolerances and simulation constants.

### 2. [`typing.py`](./src/firm_ce/common/typing.py)
- Defines precision-aware Numba and NumPy scalar/array types (`nbfloat`, `npfloat`, `nbint`, `npint`, `nbintp`, `npintp`, `boolean`, `unicode_type`) and Numba container types (`DictType`, `ListType`, `UniTuple`, `TypedDict`, `TypedList`) with fallback stubs when `JIT_ENABLED = False`.

### 3. [`jit_overload.py`](./src/firm_ce/common/jit_overload.py)
- Re-exports `njit`, `jitclass`, `prange`, and `get_thread_id` from Numba when `JIT_ENABLED = True`, or provides no-op decorators and pure-Python equivalents when `JIT_ENABLED = False`.

### 4. [`helpers.py`](./src/firm_ce/common/helpers.py)
- **`parse_comma_separated(val)`**: Splits comma-delimited config strings into stripped lowercase lists.
- **`safe_divide(num, denom)` / `safe_divide_array(num, denom, out)` / `safe_divide_2d_1d(num_2d, denom_1d, out_2d)`**: `@njit`-compiled zero-safe division utilities used across both backends and post-processing.

### 5. [`exceptions.py`](./src/firm_ce/common/exceptions.py)
- Custom exception classes (`ValidationError`, `StaticModificationError`, `UnloadedDataError`) and `@njit`-compatible helper functions (`raise_static_modification_error()`, `raise_getting_unloaded_data_error()`).

### 6. [`logging.py`](./src/firm_ce/common/logging.py)
- **`init_results_directory(model_name, logging_flag, model_location)`**: Creates timestamped results folders (`results/<model_name>_<timestamp>`) or resolves `"latest"` / explicit directory paths.
- **`init_model_logger(results_dir)`**: Configures dual console + `log.txt` logging.
