# type: ignore
from abc import ABC, abstractmethod
from contextlib import contextmanager
import csv
import os
import signal
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from mga.mhmga import MGAProblem

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.optimisation.objective_factory import build_objective_bundle


@contextmanager
def graceful_sigint_handler(algorithm: MGAProblem, logger=None):
    """
    Intercepts SIGINT (Ctrl+C) during Numba execution, deferring termination
    until the current iteration finishes. Prevents raising KeyboardInterrupt
    inside Numba's `_numba_unpickle` C-callbacks (which causes SystemError).
    """
    original_handler = signal.getsignal(signal.SIGINT)
    sigint_count = 0

    def _handler(signum, frame):
        nonlocal sigint_count
        sigint_count += 1
        algorithm._stop_requested = True

        if sigint_count == 1:
            msg = (
                "[FIRM_CE] Interrupt received. Finishing current iteration and finalising logs "
                "(press Ctrl+C 3x to force-kill)..."
            )
            if logger:
                logger.warning(msg)
            else:
                print(f"\n{msg}", flush=True)
        elif sigint_count >= 3:
            print("\n[FIRM_CE] Forced termination requested. Exiting immediately.", flush=True)
            os._exit(130)

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)


class BaseSolver(ABC):
    """Abstract base class for all optimisation strategies"""

    def __init__(
        self,
        scenario,
        initial_population: Union[NDArray[np.float64], None] = None,
    ):
        self.scenario = scenario
        self.config = scenario.config
        self.x0 = self.scenario.x0 if len(self.scenario.x0) > 0 else None
        self.initial_population = initial_population
        self.result = None

        self.obj_bundle = build_objective_bundle(self.scenario, self.config)
        self.objective_function = self.obj_bundle.func
        self.fargs = self.obj_bundle.fargs
        self.details = self.obj_bundle.details

    @abstractmethod
    def evaluate(self) -> None:
        """Executes the specific optimization strategy. Must be implemented by subclasses."""
        pass

    def initialise_callback(self) -> None:
        out_dir = self.scenario.solution_dir
        os.makedirs(out_dir, exist_ok=True)

        if self.config.restart_optimisation is not True or getattr(self.config, "model_location", None) == "new":
            with open(os.path.join(out_dir, "callback.csv"), "w", newline="") as csvfile:
                csv.writer(csvfile)

            if SAVE_POPULATION:
                open(os.path.join(out_dir, "population.bin"), "wb").close()
                if getattr(self.config, "save_details", False):
                    open(os.path.join(out_dir, "details.bin"), "wb").close()


def get_solver(
    scenario,
    initial_population: Optional[np.ndarray] = None,
) -> BaseSolver:
    """Factory function to instantiate the correct solver strategy."""
    from firm_ce.optimisation.de_solver import SingleTimeSolver
    from firm_ce.optimisation.mhmga_solver import MhmgaSolver

    strategies = {
        "single_time": SingleTimeSolver,
        "mhmga": MhmgaSolver,
    }

    solver_class = strategies.get(scenario.config.type)

    if not solver_class:
        raise ValueError(
            f"Unsupported config.type: '{scenario.config.type}'. Must be one of {list(strategies.keys())}."
        )

    return solver_class(scenario, initial_population)
