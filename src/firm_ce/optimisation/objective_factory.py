from typing import Callable, Tuple, Any

from firm_ce.backend.scalar.single_time import evaluate_vectorised_xs as default_st_obj
from firm_ce.optimisation.mhmga import mga_wrapper as default_mga_obj
from firm_ce.optimisation.mhmga import mga_wrapper_with_details

# TODO: Import alternative backend implementations here
# from firm_ce.backends.fast_c import fast_single_time as c_st_obj


def _get_default_args(scenario, config) -> Tuple:
    """Standard argument packing for the default backend."""
    return (
        scenario.static,
        scenario.fleet,
        scenario.network,
        config.balancing_type,
        config.fixed_costs_threshold,
    )


def _get_mga_details_args(scenario, config) -> Tuple:
    """Appends the pre-allocated details array for MHMGA."""
    import numpy as np

    # Logic extracted from the old get_mhmga_args
    details = np.empty(
        (
            config.mga_start_niches + sum(config.mga_new_niches),    # max niches
            max(config.mga_pop_size),
            scenario.details_length,
        ),
        dtype=np.float64
    )
    return (*_get_default_args(scenario, config), details)


def build_objective(scenario, config) -> Tuple[Callable, Tuple[Any, ...]]:
    """
    Routes to the correct objective function and argument tuple
    based on config type and backend.
    """
    backend = getattr(config, 'backend', 'scalar').lower()
    opt_type = config.type.lower()
    save_details = getattr(config, "save_details", False)

    if opt_type == "single_time" or opt_type == "near_optimum" or opt_type == "midpoint_explore":
        if backend == "scalar":
            return default_st_obj, _get_default_args(scenario, config)

        elif backend == "tensor":
            # TODO: new objective
            return default_st_obj, _get_default_args(scenario, config)

        else:
            raise ValueError(f"Unknown backend '{backend}' for {opt_type}.")

    elif opt_type.startswith("mhmga"):
        if backend == "scalar":
            if save_details:
                return mga_wrapper_with_details, _get_mga_details_args(scenario, config)
            return default_mga_obj, _get_default_args(scenario, config)

        elif backend == "tensor":
            # TODO: new objective
            if save_details:
                return mga_wrapper_with_details, _get_mga_details_args(scenario, config)
            return default_mga_obj, _get_default_args(scenario, config)

        else:
            raise ValueError(f"Unknown backend '{backend}' for {opt_type}.")

    else:
        raise ValueError(f"Unsupported optimisation type: {opt_type}")
