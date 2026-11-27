# type: ignore
from typing import Dict
import numpy as np

from firm_ce.common.helpers import (
    parse_boolean,
    parse_comma_separated,
    parse_ditherable_hyperparameter,
    parse_restart_optimisation,
)

expected_mga_hyperparameters = {
    "mga_iter": {"default": 100, "ditherable": False, "broadcastable": True, "types": ((int, 1, np.inf),)},
    "mga_pop_size": {"default": 100, "ditherable": False, "broadcastable": True, "types": ((int, 2, np.inf),)},
    "mga_noptimal_rel": {"default": 0.0, "ditherable": False, "broadcastable": True, "types": ((float, 0, np.inf),)},
    "mga_noptimal_abs": {"default": 0.0, "ditherable": False, "broadcastable": True, "types": ((float, 0, np.inf),)},
    "mga_mutation_prob": {"default": 0.2, "ditherable": True, "broadcastable": True, "types": ((float, 0, 1),)},
    "mga_mutation_sigma": {"default": 0.1, "ditherable": True, "broadcastable": True, "types": ((float, 0, np.inf),)},
    "mga_mutation_alpha": {"default": 0.0, "ditherable": False, "broadcastable": True, "types": ((float, -np.inf, np.inf),)},
    "mga_crossover_prob": {"default": 0.2, "ditherable": True, "broadcastable": True, "types": ((float, 0, 1),)},
    "mga_tourn_size": {"default": 2, "ditherable": False, "broadcastable": True, "types": ((int, 2, np.inf),)},
    "mga_tourn_count": {"default": 0.8, "ditherable": False, "broadcastable": True, "types": ((int, -1, np.inf), (float, 0, 1),)},
    "mga_elite_count": {"default": 0.2, "ditherable": False, "broadcastable": True, "types": ((int, -1, np.inf), (float, 0, 1),)},
    "mga_champ_count": {"default": 0, "ditherable": False, "broadcastable": True, "types": ((int, -1, np.inf), (float, 0, 1),)},
    "mga_start_div_niches": {"default": 10, "ditherable": False, "broadcastable": False, "types": ((int, 0, np.inf),)},
    "mga_new_div_niches": {"default": 0, "ditherable": False, "broadcastable": True, "types": ((int, 0, np.inf),)},
    "mga_extrema_dims": {"default": "none", "ditherable": False, "broadcastable": False, "types":
                         ((str, ("none", "all", "all-max", "all-min", "custom")),)},
    "mga_extrema_is_min": {"default": "auto", "ditherable": False, "broadcastable": False, "types":
                           ((str, ("auto", "none", "custom")),)},
    "mga_niche_elitism": {"default": True, "ditherable": False, "broadcastable": True, "types":
                          ((bool, ("false", "none", "true")),)},
    "mga_log_freq": {"default": 1, "ditherable": False, "broadcastable": False, "types": ((int, -1, np.inf),)},
    "mga_disp_rate": {"default": 1, "ditherable": False, "broadcastable": True, "types": ((int, -1, np.inf),)},
    "mga_verbose_level": {"default": 3, "ditherable": False, "broadcastable": True, "types": ((int, 0, np.inf),)},
    "mga_fitness": {"default": "angular", "ditherable": False, "broadcastable": True, "types": ((str, ("angular", "l2", "l1")),)},
    "mga_extrema_epsilon_rel": {"default": 1e-3, "ditherable": False, "broadcastable": True, "types": ((float, 0, np.inf),)},
    "mga_extrema_epsilon_abs": {"default": 1e-6, "ditherable": False, "broadcastable": True, "types": ((float, 0, np.inf),)},
}


def coercive_type_cast(item, target):
    try:
        return target(item)
    except ValueError:
        return None


def check_type(param_name, param_dict, item):
    typepass = False
    for typer in param_dict["types"]:
        if typer[0] == str:
            if not isinstance(item, typer[0]):
                continue
            if item not in typer[1]:
                continue
        elif typer[0] == bool:
            if str(item).lower() not in ("false", "none", "true"):
                continue
        else:  # numeric
            cast_item = coercive_type_cast(item, typer[0])
            if not isinstance(cast_item, typer[0]):
                continue
            if cast_item < typer[1]:
                continue
            if cast_item > typer[2]:
                continue
        typepass = typer[0]
        break
    if not typepass:
        raise TypeError(
            f"dtype of {param_name} was not of acceptable type or out of bounds (got: {item} of type: {type(item)})"
        )
    return typepass


def parse_and_validate_mga(config_dict, model_logger):
    """Parses, broadcasts, validates, and explicitly casts MHMGA parameters."""
    flag = True
    config_by_name = {item["name"]: item for item in config_dict.values()}

    mga_steps = 1
    if "mga_steps" in config_by_name:
        try:
            mga_steps = int(config_by_name["mga_steps"]["value"])
            config_by_name["mga_steps"]["value"] = mga_steps
        except ValueError:
            model_logger.error("'mga_steps' must be an integer")
            flag = False

    def cast_value(val, target_type):
        if target_type is bool:
            return str(val).lower() in ("true", "t", "1", "yes", "y")
        return target_type(val)

    for param_name, param_dict in expected_mga_hyperparameters.items():
        string = config_by_name.get(param_name, {}).get("value", param_dict["default"])

        try:
            if param_dict["ditherable"]:
                value = np.array(parse_ditherable_hyperparameter(str(string)), dtype=object)
                if value.shape[0] == 1:
                    value = np.stack((value[0],) * mga_steps)
                elif value.shape[0] != mga_steps:
                    raise ValueError(f"{param_name} not broadcastable to mga_steps")

                cast_arr = np.empty_like(value)
                for idx, item in np.ndenumerate(value):
                    valid_type = check_type(param_name, param_dict, item)
                    cast_arr[idx] = cast_value(item, valid_type)
                final_value = cast_arr

            elif param_dict["broadcastable"]:
                value = parse_comma_separated(str(string))
                if len(value) == 1:
                    value = value * mga_steps
                elif len(value) != mga_steps:
                    raise ValueError(f"{param_name} not broadcastable to mga_steps")

                for i, item in enumerate(value):
                    valid_type = check_type(param_name, param_dict, item)
                    value[i] = cast_value(item, valid_type)
                final_value = value

            else:
                valid_type = check_type(param_name, param_dict, string)
                final_value = cast_value(string, valid_type)

            if param_name in config_by_name:
                config_by_name[param_name]["value"] = final_value
            else:
                config_dict[f"auto_{param_name}"] = {"name": param_name, "value": final_value}

        except (ValueError, TypeError) as e:
            model_logger.error(f"Validation failed for '{param_name}': {str(e)}")
            flag = False

    return flag


class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        config = {item["name"]: item["value"] for item in config_dict.values()}

        self.type = config["type"]
        self.backend = config.get("backend", "scalar")
        self.parameterisation = config.get("parameterisation", "absolute")
        self.model_name = config["model_name"]

        self.restart_optimisation = parse_restart_optimisation(config.get("restart_optimisation", False))
        if self.restart_optimisation and self.type != "mhmga":
            raise NotImplementedError("Restart only implemented for mhmga")

        self.save_details = parse_boolean(config.get("save_details", False))
        if self.save_details and self.type != "mhmga":
            raise NotImplementedError("Save details only implemented for mhmga")

        self.model_location = str(config.get("model_location", "new"))
        self.balancing_type = str(config["balancing_type"])
        self.fixed_costs_threshold = float(config.get("fixed_costs_threshold", 500.0))
        self.limit_timesteps = int(config.get("limit_timesteps")) if config.get("limit_timesteps") is not None else None
        self.demand_multiple = float(config.get("demand_multiple", 1.0))
        self.interval_aggregation = int(config.get("interval_aggregation", 1))

        if self.type == "single_time":
            self.iterations = int(config["iterations"])
            self.population = int(config["population"])
            self.mutation = float(config["mutation"])
            self.recombination = float(config["recombination"])

        if self.type == "mhmga":
            self.mga_steps = int(config.get("mga_steps", 1))

            # 3-Axis MGA Fitness Space Configuration
            self.mga_space_basis = str(config.get("mga_space_basis", "capacity")).lower()
            self.mga_space_norm = str(config.get("mga_space_norm", self.parameterisation)).lower()
            self.mga_space_agg = str(config.get("mga_space_agg", "unit_type")).lower()

            if self.mga_space_basis not in ("capacity", "energy"):
                raise ValueError(f"Invalid mga_space_basis '{self.mga_space_basis}'. Expected 'capacity' or 'energy'.")
            if self.mga_space_norm not in ("relative", "absolute"):
                raise ValueError(f"Invalid mga_space_norm '{self.mga_space_norm}'. Expected 'relative' or 'absolute'.")
            if self.mga_space_agg not in ("unit_type", "node", "none"):
                raise ValueError(f"Invalid mga_space_agg '{self.mga_space_agg}'. Expected 'unit_type', 'node', or 'none'.")

            for param_name, spec in expected_mga_hyperparameters.items():
                default = spec["default"]
                setattr(self, param_name, config.get(param_name, default))

            if self.mga_extrema_dims == "custom":
                raise NotImplementedError("Custom 'mga_extrema_dims' not implemented yet")
            if self.mga_extrema_is_min == "custom":
                raise NotImplementedError("Custom 'mga_extrema_is_min' not implemented yet")

    def update(self, new_params: dict) -> None:
        for key, value in new_params.items():
            setattr(self, key, value)
