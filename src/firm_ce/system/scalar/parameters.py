# type: ignore
from typing import Dict

import numpy as np

from firm_ce.common.constants import JIT_ENABLED, LEAPDAYS
from firm_ce.common.helpers import parse_boolean
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import nbfloat, nbintp, npfloat
from firm_ce.io.data_model import expected_mga_hyperparameters

if JIT_ENABLED:
    scenario_parameters_spec = [
        ("resolution", nbfloat),
        ("allowance", nbfloat),
        ("first_year", nbintp),
        ("final_year", nbintp),
        ("year_count", nbintp),
        ("leap_year_count", nbintp),
        ("year_first_t", nbintp[:]),
        ("year_of_interval", nbintp[:]),
        ("intervals_count", nbintp),
        ("node_count", nbintp),
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
        year_count: nbintp,
        leap_year_count: nbintp,
        year_first_t: nbintp[:],
        year_of_interval: nbintp[:],
        intervals_count: nbintp,
        node_count: nbintp,
    ):

        self.resolution = resolution  # length of time interval in hours
        self.allowance = allowance  # % annual demand allowed as unserved energy
        self.first_year = first_year  # YYYY
        self.final_year = final_year  # YYYY
        self.year_count = year_count
        self.leap_year_count = leap_year_count if LEAPDAYS else 0
        self.year_first_t = year_first_t
        self.year_of_interval = year_of_interval
        self.intervals_count = intervals_count
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


# TODO: move into it's own module
class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        # Values have already been parsed and validated in firm_ce.io.data_model
        config = {item["name"]: item["value"] for item in config_dict.values()}

        self.type = config["type"]
        self.backend = config.get("backend", "scalar")
        self.parameterisation = config.get("parameterisation", "absolute")
        self.model_name = config["model_name"]

        self.restart_optimisation = parse_boolean(config.get("restart_from_temp", False))
        if self.restart_optimisation and self.type != "mhmga":
            raise NotImplementedError("Restart from temp only implemented for mhmga")

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

        if self.type in ("near_optimum", "midpoint_explore"):
            self.near_optimal_tol = float(config["near_optimal_tol"])
            self.midpoint_count = int(config["midpoint_count"])

        if self.type == "mhmga":
            self.mga_steps = int(config.get("mga_steps", 1))  # default: 1

            for param_name in expected_mga_hyperparameters.keys():

                setattr(self, param_name, config[param_name])

    def update(self, new_params: dict) -> None:
        for key, value in new_params.items():
            setattr(self, key, value)
