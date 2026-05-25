import os
import numpy as np

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

JIT_ENABLED = True
SAVE_POPULATION = True
DEBUG = False
BITDEPTH = 32
if BITDEPTH == 64:
    EPSILON_FLOAT = np.finfo(np.float64).eps
    NP_FLOAT_MAX = np.finfo(np.float64).max
    NP_FLOAT_MIN = np.finfo(np.float64).min
    NP_INT_MAX = np.iinfo(np.int64).max
elif BITDEPTH == 32:
    EPSILON_FLOAT = np.finfo(np.float32).eps
    NP_FLOAT_MAX = np.finfo(np.float32).max
    NP_FLOAT_MIN = np.finfo(np.float32).min
    NP_INT_MAX = np.iinfo(np.int32).max
    os.environ["MGA_USE_32BIT"] = "1"
else:
    raise ValueError(f"BITDEPTH must be 32 or 64. Got {BITDEPTH}")
PENALTY_MULTIPLIER = 1e6
TOLERANCE = 1e-6
VALIDATION_TOL = 1e-2
NUM_THREADS = int(os.getenv("NUM_THREADS", os.cpu_count()))
FASTMATH = True
LEAPDAYS = False
# numba disables bounds checking for improved speed
# means that IndexErrors lead to ungraceful crashes - set to True for graceful handling
BOUNDSCHECK = True
