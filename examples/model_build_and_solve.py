"""
An example is provided for building a FIRM Model instance and then solving it. The Model object is built using the default `inputs/config` and
`inputs/data` files. Each scenario in `inputs/config/scenarios.csv` is optimised sequentially using the SciPy differential evolution algorithm.
Results are saved in the `results` folder.

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""

import time

from firm_ce.model import Model

start_time = time.time()
model = Model()
model_build_time = time.time()

print(model.scenarios)
print(f"Model build time: {model_build_time - start_time:.4f} seconds")

model.solve(scenarios="base")
end_time = time.time()
print(f"Model solve time: {end_time - model_build_time:.4f} seconds")
