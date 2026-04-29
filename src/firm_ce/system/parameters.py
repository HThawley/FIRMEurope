# type: ignore
from typing import Dict

import numpy as np

from firm_ce.common.constants import JIT_ENABLED, LEAPDAYS
from firm_ce.common.helpers import parse_boolean
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, npfloat, nbint, npint, nbintp
from firm_ce.common.helpers import parse_comma_separated, parse_ditherable_hyperparameter


if JIT_ENABLED:
    scenario_parameters_spec = [
        ("resolution", nbfloat),
        ("interval_resolutions", nbfloat[:]),
        ("allowance", nbfloat),
        ("first_year", nbintp),
        ("final_year", nbintp),
        ("year_count", nbint),
        ("leap_year_count", nbint),
        ("year_first_t", nbintp[:]),
        ("intervals_count", nbint),
        ("block_lengths", nbint[:]),
        ("node_count", nbint),
        ("fom_scalar", nbfloat),
        ("year_float", nbfloat),
        ("year_energy_demand", nbfloat[:]),
        ("mean_annual_demand_mwh", nbfloat),
        ("demand_sum_mwh", nbfloat),
    ]
else:
    scenario_parameters_spec = []


@jitclass(scenario_parameters_spec)
class ScenarioParameters:
    def __init__(
        self,
        resolution: nbfloat,
        allowance: nbfloat,
        first_year: nbintp,
        final_year: nbintp,
        year_count: nbint,
        leap_year_count: nbint,
        year_first_t: nbintp[:],
        intervals_count: nbint,
        node_count: nbint,
    ):

        self.resolution = resolution  # length of time interval in hours
        self.interval_resolutions = resolution * np.ones(
            intervals_count, dtype=npfloat
        )  # length of blocks in hours, for future 'simple' balancing_method
        self.allowance = allowance  # % annual demand allowed as unserved energy
        self.first_year = first_year  # YYYY
        self.final_year = final_year  # YYYY
        self.year_count = year_count
        self.leap_year_count = leap_year_count if LEAPDAYS else 0
        self.year_first_t = year_first_t
        self.intervals_count = intervals_count
        self.block_lengths = np.ones(intervals_count, dtype=npint)
        self.node_count = node_count
        if LEAPDAYS:
            self.fom_scalar = (
                year_count + leap_year_count / 365
            ) / year_count  # Scale average annual fom to account for leap days for PLEXOS consistency
        else:
            self.fom_scalar = 1.0
        self.year_float = self.year_count * self.fom_scalar
        self.year_energy_demand = np.zeros(self.year_count, dtype=npfloat)
        self.mean_annual_demand_mwh = 0.0
        self.demand_sum_mwh = 0.0


if JIT_ENABLED:
    ScenarioParameters_InstanceType = ScenarioParameters.class_type.instance_type
else:
    ScenarioParameters_InstanceType = ScenarioParameters


class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        config_dict = {item["name"]: item["value"] for item in config_dict.values()}
        self.type = config_dict["type"]
        self.model_name = config_dict["model_name"]
        self.restart_optimisation = parse_boolean(config_dict.get("restart_from_temp", False))
        if self.restart_optimisation and self.type != "mhmga":
            raise NotImplementedError("Restart from temp only implemented for mhmga")
        self.save_operations = parse_boolean(config_dict.get("save_operations", False))
        if self.save_operations and self.type != "mhmga":
            raise NotImplementedError("Save operations only implemented for mhmga")
        self.model_location = str(config_dict.get("model_location", "new"))
        self.balancing_type = str(config_dict["balancing_type"])
        self.fixed_costs_threshold = float(config_dict.get("fixed_costs_threshold", 500.0))
        self.limit_timesteps = config_dict.get("limit_timesteps")

        if self.type == "single_time":
            self.iterations = int(config_dict["iterations"])
            self.population = int(config_dict["population"])
            self.mutation = float(config_dict["mutation"])
            self.recombination = float(config_dict["recombination"])

        if self.type in ("near_optimum", "midpoint_explore"):
            self.near_optimal_tol = float(config_dict["near_optimal_tol"])
            self.midpoint_count = int(config_dict["midpoint_count"])

        if self.type == "mhmga":
            self.mga_steps = int(config_dict.get("mga_steps", 1))  # default: 1

            for param_name, param_dict in expected_mga_hyperparameters.items():
                string = config_dict.get(param_name, param_dict["default"])

                if param_dict["ditherable"]:
                    broadcastable = param_dict.get("broadcastable", True)
                    if not broadcastable:
                        raise ValueError(f"Parameters cannot be both ditherable and non-broadcastable (param: {param_name})")
                    value = np.array(parse_ditherable_hyperparameter(string))
                    if value.shape[0] == 1:
                        value = np.stack((value[0],) * self.mga_steps)
                    elif value.shape[0] == self.mga_steps:
                        pass
                    else:
                        raise ValueError(f"{param_name} not broadcastable to mga_steps")
                    for item in value.flatten():
                        check_type(param_name, param_dict, item)
                    setattr(self, param_name, value)

                elif param_dict["broadcastable"]:
                    value = parse_comma_separated(string)
                    if len(value) == 1:
                        value = value * self.mga_steps
                    elif len(value) == self.mga_steps:
                        pass
                    else:
                        raise ValueError(f"{param_name} not broadcastable to mga_steps")
                    for i, item in enumerate(value):
                        valid_type = check_type(param_name, param_dict, item)
                        value[i] = valid_type(item)
                    setattr(self, param_name, value)

                else:
                    assert param_name in ("mga_log_freq", "mga_disp_rate", "mga_start_niches")
                    string = config_dict.get(param_name, param_dict["default"])
                    valid_type = check_type(param_name, param_dict, string)
                    value = valid_type(string)
                    setattr(self, param_name, value)

    def update(self, new_params: dict) -> None:
        for key, value in new_params.items():
            setattr(self, key, value)


def check_type(param_name, param_dict, item):
    typepass = False
    for typer in param_dict["types"]:
        if typer[0] == str:
            if not isinstance(item, typer[0]):
                continue
            if item not in typer[1]:
                continue
        else:  # numeric
            item = coercive_type_cast(item, typer[0])
            if not isinstance(item, typer[0]):
                continue
            if item < typer[1]:
                continue
            if item > typer[2]:
                continue
        typepass = typer[0]
        break
    if not typepass:
        raise TypeError(f"dtype of {param_name} was not of acceptable type or out of bounds (got: {item} of type: {type(item)})")
    return typepass


def coercive_type_cast(item, target):
    try:
        return target(item)
    except ValueError:
        return None


expected_mga_hyperparameters = {
    "mga_iter": {
        "default": 100,
        "ditherable": False,
        "broadcastable": True,
        "types": ((int, 1, np.inf),),
    },
    "mga_pop_size": {
        "default": 100,
        "ditherable": False,
        "broadcastable": True,
        "types": ((int, 2, np.inf),),
    },
    "mga_noptimal_rel": {
        "default": 0.0,
        "ditherable": False,
        "broadcastable": True,
        "types": ((float, 0, np.inf),),
    },
    "mga_noptimal_abs": {
        "default": 0.0,
        "ditherable": False,
        "broadcastable": True,
        "types": ((float, 0, np.inf),),
    },
    "mga_mutation_prob": {
        "default": 0.2,
        "ditherable": True,
        "broadcastable": True,
        "types": ((float, 0, 1),),  # (type, lower, upper)
    },
    "mga_mutation_sigma": {
        "default": 0.1,
        "ditherable": True,
        "broadcastable": True,
        "types": ((float, 0, np.inf),),
    },
    "mga_mutation_alpha": {
        "default": 0.0,
        "ditherable": False,
        "broadcastable": True,
        "types": ((float, -np.inf, np.inf),),
    },
    "mga_crossover_prob": {
        "default": 0.2,
        "ditherable": True,
        "broadcastable": True,
        "types": ((float, 0, 1),),
    },
    "mga_tourn_size": {
        "default": 2,
        "ditherable": False,
        "broadcastable": True,
        "types": ((int, 2, np.inf),)
    },
    "mga_tourn_count": {
        "default": 0.8,
        "ditherable": False,
        "broadcastable": True,
        "types": (
            (float, 0, 1),
            (int, -1, np.inf),
        ),
    },
    "mga_elite_count": {
        "default": 0.2,
        "ditherable": False,
        "broadcastable": True,
        "types": (
            (float, 0, 1),
            (int, -1, np.inf),
        ),
    },
    "mga_champ_count": {
        "default": 0,
        "ditherable": False,
        "broadcastable": True,
        "types": (
            (float, 0, 1),
            (int, -1, np.inf),
        ),
    },
    "mga_start_niches": {
        "default": 10,
        "ditherable": False,
        "broadcastable": False,
        "types": ((int, 1, np.inf),),
    },
    "mga_new_niches": {
        "default": 0,
        "ditherable": False,
        "broadcastable": True,
        "types": ((int, 0, np.inf),),
    },
    "mga_niche_elitism": {
        "default": "selfish",
        "ditherable": False,
        "broadcastable": True,
        "types": ((str, ("none", "selfish", "unselfish")),),
    },
    "mga_log_freq": {
        "default": 1,
        "ditherable": False,
        "broadcastable": False,
        "types": ((int, -1, np.inf),),
    },
    "mga_disp_rate": {
        "default": 1,
        "ditherable": False,
        "broadcastable": True,
        "types": ((int, -1, np.inf),),
    },
    "mga_verbose_level": {
        "default": 3,
        "ditherable": False,
        "broadcastable": True,
        "types": ((int, 0, np.inf),),
    },
}
