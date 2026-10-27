# type: ignore
from typing import Dict

from firm_ce.common.helpers import parse_boolean
from firm_ce.io.data_model import expected_mga_hyperparameters


class ModelConfig:
    def __init__(self, config_dict: Dict[str, str]) -> None:
        # Values have already been parsed and validated in firm_ce.io.data_model
        config = {item["name"]: item["value"] for item in config_dict.values()}

        self.type = config["type"]
        self.backend = config.get("backend", "scalar")
        self.parameterisation = config.get("parameterisation", "absolute")
        self.model_name = config["model_name"]

        self.restart_optimisation = parse_boolean(config.get("restart_optimisation", False))
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

            # 3-Axis MGA Fitness Space Configuration
            # basis: "capacity" (linear x projection via space_scaler) or "energy" (non-linear via return_scaled=True)
            self.mga_space_basis = str(config.get("mga_space_basis", "capacity")).lower()
            # norm: "relative" (normalised by nodal demand) or "absolute" (physical / system-wide)
            self.mga_space_norm = str(config.get("mga_space_norm", self.parameterisation)).lower()
            # agg: "unit_type" (grouped by tech), "node" (grouped by node), or "none" (unaggregated)
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
