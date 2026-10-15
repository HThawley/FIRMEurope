# type: ignore
from dataclasses import dataclass
from typing import Callable, Tuple, Any, Optional
import numpy as np

from firm_ce.common.typing import npfloat, npintp
from firm_ce.backend.scalar.single_time import evaluate_vectorised_xs as default_st_obj
from firm_ce.backend.scalar.mhmga import (
    mga_wrapper as default_mga_obj,
    mga_wrapper_with_details,
    mga_scalar_wrapper_scaled,
)
from firm_ce.backend.tensor.mhmga import (
    mga_tensor_wrapper,
    mga_tensor_wrapper_with_details,
    mga_tensor_wrapper_scaled,
)
from firm_ce.backend.tensor.solution import (
    N_ENERGY_COMPONENTS,
    NORM_RELATIVE,
    NORM_ABSOLUTE,
    AGG_NONE,
    AGG_UNIT_TYPE,
    AGG_NODE,
)


@dataclass(frozen=True)
class ObjectiveBundle:
    func: Callable
    fargs: Tuple[Any, ...]
    return_scaled: bool = False
    space_scaler: Optional[np.ndarray] = None
    details: Optional[np.ndarray] = None


def _get_default_args(scenario, config) -> Tuple:
    return (
        scenario.static,
        scenario.fleet,
        scenario.network,
        config.balancing_type,
        config.fixed_costs_threshold,
    )


def _allocate_details_array(scenario, config) -> np.ndarray:
    return np.empty(
        (
            config.mga_start_niches + sum(config.mga_new_niches),
            max(config.mga_pop_size),
            scenario.details_length,
        ),
        dtype=npfloat,
    )


def _resolve_energy_space_dims(scenario, config, n_components: int) -> Tuple[npintp, npintp, npintp]:
    norm_str = getattr(config, "mga_space_norm", config.parameterisation).lower()
    agg_str = getattr(config, "mga_space_agg", "unit_type").lower()

    norm_code = NORM_RELATIVE if norm_str == "relative" else NORM_ABSOLUTE
    n_nodes = len(scenario.network.nodes)

    if agg_str == "unit_type":
        agg_code = AGG_UNIT_TYPE
        scaled_dim = n_components
    elif agg_str == "node":
        agg_code = AGG_NODE
        scaled_dim = n_nodes
    elif agg_str == "none":
        agg_code = AGG_NONE
        scaled_dim = n_nodes * n_components
    else:
        raise ValueError(f"Unknown mga_space_agg '{agg_str}'.")

    return npintp(norm_code), npintp(agg_code), npintp(scaled_dim)


def build_objective_bundle(scenario, config) -> ObjectiveBundle:
    """
    Routes to the correct objective function, argument tuple, `return_scaled` flag,
    `space_scaler` matrix, and `details` buffer based on config type, backend,
    and the 3-axis MGA fitness space configuration.
    """
    backend = getattr(config, "backend", "scalar").lower()
    opt_type = config.type.lower()
    save_details = getattr(config, "save_details", False)

    if opt_type in ("single_time", "near_optimum", "midpoint_explore"):
        if backend in ("scalar", "tensor"):
            return ObjectiveBundle(func=default_st_obj, fargs=_get_default_args(scenario, config))
        raise ValueError(f"Unknown backend '{backend}' for {opt_type}.")

    elif opt_type.startswith("mhmga"):
        space_basis = getattr(config, "mga_space_basis", "capacity").lower()
        details_buf = _allocate_details_array(scenario, config) if save_details else None

        if space_basis == "capacity":
            space_scaler = scenario.build_capacity_space_scaler()

            if backend == "scalar":
                if save_details:
                    return ObjectiveBundle(
                        func=mga_wrapper_with_details,
                        fargs=(*_get_default_args(scenario, config), details_buf),
                        return_scaled=False,
                        space_scaler=space_scaler,
                        details=details_buf,
                    )
                return ObjectiveBundle(
                    func=default_mga_obj,
                    fargs=_get_default_args(scenario, config),
                    return_scaled=False,
                    space_scaler=space_scaler,
                )

            elif backend == "tensor":
                if save_details:
                    return ObjectiveBundle(
                        func=mga_tensor_wrapper_with_details,
                        fargs=(scenario.staticTensor, details_buf),
                        return_scaled=False,
                        space_scaler=space_scaler,
                        details=details_buf,
                    )
                return ObjectiveBundle(
                    func=mga_tensor_wrapper,
                    fargs=(scenario.staticTensor,),
                    return_scaled=False,
                    space_scaler=space_scaler,
                )

        elif space_basis == "energy":
            nodal_demand_totals = scenario.get_nodal_demand_totals()
            det_arg = details_buf if save_details else np.empty((1, 1, 1), dtype=npfloat)
            n_vars = len(scenario.lower_bounds)

            if backend == "tensor":
                norm_code, agg_code, scaled_dim = _resolve_energy_space_dims(
                    scenario, config, N_ENERGY_COMPONENTS
                )
                # Pass a 2D dummy space_scaler so mga sets population.scaled_points to scaled_dim columns
                dummy_space_scaler = np.empty((n_vars, int(scaled_dim)), dtype=npfloat)

                fargs = (
                    scenario.staticTensor,
                    nodal_demand_totals,
                    norm_code,
                    agg_code,
                    scaled_dim,
                    bool(save_details),
                    det_arg,
                )
                return ObjectiveBundle(
                    func=mga_tensor_wrapper_scaled,
                    fargs=fargs,
                    return_scaled=True,
                    space_scaler=dummy_space_scaler,
                    details=details_buf,
                )

            elif backend == "scalar":
                n_gen_types = len({g.unit_type for g in scenario.fleet.generators.values()})
                n_sto_types = len({s.unit_type for s in scenario.fleet.storages.values()})
                n_energy_types = n_gen_types + n_sto_types + 1  # +1 for curtailment

                norm_code, agg_code, scaled_dim = _resolve_energy_space_dims(
                    scenario, config, n_energy_types
                )
                dummy_space_scaler = np.empty((n_vars, int(scaled_dim)), dtype=npfloat)

                fargs = (
                    *_get_default_args(scenario, config),
                    nodal_demand_totals,
                    norm_code,
                    agg_code,
                    npintp(n_gen_types),
                    npintp(n_energy_types),
                    scaled_dim,
                    bool(save_details),
                    det_arg,
                )
                return ObjectiveBundle(
                    func=mga_scalar_wrapper_scaled,
                    fargs=fargs,
                    return_scaled=True,
                    space_scaler=dummy_space_scaler,  # <-- Corrected
                    details=details_buf,
                )

        raise ValueError(f"Unsupported backend='{backend}' or mga_space_basis='{space_basis}'.")

    else:
        raise ValueError(f"Unsupported optimisation type: {opt_type}")


def build_objective(scenario, config) -> Tuple[Callable, Tuple[Any, ...]]:
    """Backwards-compatible 2-tuple wrapper around build_objective_bundle."""
    bundle = build_objective_bundle(scenario, config)
    return bundle.func, bundle.fargs
