"""
An example is provided for building a FIRM Model instance. The Model object is built
using the default `inputs/config` and `inputs/data` files. Each scenario in `inputs/config/scenarios.csv`

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""

import time

from firm_ce.model import Model

start_time = time.time()
model = Model()
model_build_time = time.time()

print(model.scenarios)
print(f"Model build time: {model_build_time - start_time:.4f} seconds")
