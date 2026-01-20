import os

import numpy as np

JIT_ENABLED = True
SAVE_POPULATION = True
DEBUG = False
EPSILON_FLOAT64 = np.finfo(np.float64).eps
NP_FLOAT_MAX = np.finfo(np.float64).max
NP_FLOAT_MIN = np.finfo(np.float64).min
NP_INT64_MAX = np.iinfo(np.int64).max
PENALTY_MULTIPLIER = 1e6
TOLERANCE = 1e-6
NUM_THREADS = int(os.getenv("NUM_THREADS", os.cpu_count()))
FASTMATH = True
