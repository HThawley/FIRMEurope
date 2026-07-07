# flake8: noqa: B023
import os

import numpy as np

from firm_ce.common.helpers import parse_comma_separated, parse_ditherable_hyperparameter
from firm_ce.io.file_manager import import_config_csvs


class ModelData:
    def __init__(self, config_directory: str, data_directory: str) -> None:
        self.config_directory = config_directory
        self.data_directory = data_directory

        # Get the config settings for the csvs
        self.config_data = import_config_csvs(config_directory=config_directory)

        # Get the model name
        self.model_name = self.get_model_name()

        # Set all the relevant parameters
        self.scenarios = self.config_data["scenarios"]
        self.nodes = self.config_data["nodes"]
        self.generators = self.config_data["generators"]
        self.fuels = self.config_data["fuels"]
        self.lines = self.config_data["lines"]
        self.storages = self.config_data["storages"]
        self.config = self.config_data["config"]
        self.x0s = self.config_data["initial_guess"]
        self.datafiles = self.config_data["datafiles"]

    def validate(self):
        return validate_config(self)

    def get_model_name(self) -> str:
        model_name = None

        if "config" in self.config_data:
            for record in self.config_data["config"].values():
                if record["name"] == "model_name":
                    model_name = record["value"]
                    break

        if model_name is None:
            model_name = "Model"

        return model_name


def validate_bool(val):
    return str(val).lower() in ("true", "false", "t", "f", "1", "0", "1.0", "0.0", "yes", "no", "y", "n")


def validate_range(val, min_val, max_val=None, inclusive=True):
    try:
        val = float(val)
        if inclusive:
            return min_val <= val <= max_val if max_val is not None else min_val <= val
        else:
            return min_val < val < max_val if max_val is not None else min_val < val
    except (TypeError, ValueError):
        return False


def validate_positive_int(val):
    try:
        return int(val) > 0
    except (TypeError, ValueError):
        return False


def validate_enum(val, options):
    return val in options


def parse_list(val, lower=True):
    return parse_comma_separated(val, lower) if not is_nan(val) else []


def is_nan(val):
    return isinstance(val, float) and np.isnan(val)


def validate_model_config(config_dict, model_logger):
    flag = True
    validators = {
        "backend": lambda v: str(v).lower() in ("tensor", "scalar"),
        "parameterisation": lambda v: str(v).lower() in ("relative", "absolute"),
        "mutation": lambda v: validate_range(v, 0, 2, inclusive=False),
        "iterations": validate_positive_int,
        "population": validate_positive_int,
        "restart_optimisation": validate_bool,
        "save_details": validate_bool,
        "model_location": lambda v: isinstance(v,str),
        "recombination": lambda v: validate_range(v, 0, 1),
        "type": lambda v: validate_enum(
            v,
            ["single_time", "capacity_expansion", "near_optimum", "midpoint_explore", "mhmga"],
        ),
        "model_name": None,
        "near_optimal_tol": lambda v: validate_range(v, 0, 1),
        "midpoint_count": validate_positive_int,
        "balancing_type": lambda v: validate_enum(v, ["simple", "full"]),
        "simple_blocks_per_day": validate_positive_int,
        "fixed_costs_threshold": lambda v: validate_range(v, 0),
        "limit_timesteps": validate_positive_int,
        "demand_multiple": lambda v: validate_range(v, 0.0, inclusive = False),
        "interval_aggregation": lambda v: validate_range(v, 1.0),
    }

    for item in config_dict.values():
        name = item["name"]
        value = item["value"]

        if name not in validators:
            if name.startswith("mga_"):
                # Skip here, handled by parse_and_validate_mga bel
                continue
            model_logger.warning(f"Unknown configuration name {name}")
            continue

        if not validators[name]:
            continue

        try:
            if not validators[name](value):
                model_logger.error("Invalid value for '%s': %s", name, value)
                flag = False
        except Exception as e:
            model_logger.exception("Exception during validation of '%s': %s", name, e)
            flag = False

    mga_flag = parse_and_validate_mga(config_dict, model_logger)

    return flag and mga_flag


def validate_scenarios(scenarios_dict, model_logger):
    flag = True
    scenarios_list = []
    firstyear = finalyear = None

    for item in scenarios_dict.values():
        name = item["scenario_name"]
        if name in scenarios_list:
            model_logger.error("Duplicate scenario name '%s'", name)
            flag = False
        scenarios_list.append(name.lower())

        if not validate_range(item["resolution"], 0):
            model_logger.error("'resolution' must be float greater than 0")
            flag = False

        if not validate_range(item["allowance"], 0, 1):
            model_logger.error("'allowance' must be float in range [0,1]")
            flag = False

        for year_field in ["firstyear", "finalyear"]:
            val = item.get(year_field, "auto")
            if str(val).lower().strip() in ("auto", "", "none"):
                continue
        try:
            int(val)
        except ValueError:
            model_logger.error(f"'{year_field}' must be an integer or 'auto'")
            flag = False

    return scenarios_list, flag


def validate_nodes(nodes_dict, scenarios_list, model_logger):
    flag = True
    scenario_nodes = {s: [] for s in scenarios_list}

    node_names = []
    for item in nodes_dict.values():
        name = item["name"]
        if name in node_names:
            model_logger.error("Duplicate node name '%s'", name)
            flag = False
        node_names.append(name)

        scenarios = parse_comma_separated(item["scenarios"])
        if scenarios == ["all"]:
            for scenario in scenario_nodes.keys():
                scenario_nodes[scenario].append(name)
        else:
            for scenario in scenarios:
                if scenario not in scenarios_list:
                    model_logger.error("Scenario '%s' of node '%s' not in scenarios.csv", scenario, name)
                    flag = False

                scenario_nodes[scenario].append(name)

    return scenario_nodes, flag


def validate_fuels(fuels_dict, scenarios_list, model_logger):
    flag = True
    scenario_fuels = {scenario: [] for scenario in scenarios_list}

    for idx, item in fuels_dict.items():
        if not validate_range(item["emissions"], 0):
            model_logger.error("'emissions' must be float greater than or equal to 0")
            flag = False

        if not validate_range(item["cost"], 0):
            model_logger.error("'cost' must be float greater than or equal to 0")
            flag = False

        scenarios = parse_list(item["scenarios"])
        if scenarios == ["all"]:
            for scenario in scenarios_list:
                scenario_fuels[scenario].append(item["name"])
        else:
            for scenario in scenarios:
                if scenario in scenarios_list:
                    scenario_fuels[scenario].append(item["name"])
                else:
                    model_logger.warning("scenario '%s' for fuel.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_fuels, flag


def validate_lines(lines_dict, scenarios_list, scenario_nodes, model_logger):
    flag = True
    scenario_lines = {s: [] for s in scenarios_list}
    scenario_minor_lines = {s: [] for s in scenarios_list}

    for idx, item in lines_dict.items():
        numeric_fields = {
            "length": int,
            "capex": float,
            "transformer_capex": float,
            "fom": float,
            "vom": float,
            "lifetime": int,
            "discount_rate": float,
            "loss_factor": float,
            "initial_capacity": float,
            "max_build": float,
            "min_build": float,
        }

        for field, cast in numeric_fields.items():
            try:
                val = cast(item[field])
                if "discount_rate" == field:
                    if not (0 <= val <= 1):
                        raise ValueError
                # elif "loss_factor" == field:
                #     if not (0 <= val < 1):
                #         raise ValueError
                else:
                    if val < 0:
                        raise ValueError
            except ValueError:
                model_logger.error("'%s' must be a valid %s in appropriate range", field, cast.__name__)
                flag = False

        if float(item["min_build"]) > float(item["max_build"]):
            model_logger.error("'min_build' must be less than or equal to 'max_build'")
            flag = False

        def _validate_line(flag):
            scenario_lines[scenario].append(item["name"])

            if any(is_nan(item[n]) for n in ["node_start", "node_end"]):
                scenario_minor_lines[scenario].append(item["name"])

            for endpoint in ["node_start", "node_end"]:
                node_val = item[endpoint]
                if (node_val not in scenario_nodes[scenario]) and not is_nan(node_val):
                    model_logger.error(
                        "'%s' %s for line %s is not defined in scenario %s",
                        endpoint,
                        node_val,
                        item["name"],
                        scenario,
                    )
                    return False
            return flag

        scenarios = parse_list(item["scenarios"])
        if scenarios == ["all"]:
            for scenario in scenarios_list:
                flag = _validate_line(flag)

        else:
            for scenario in scenarios:
                if scenario in scenarios_list:
                    flag = _validate_line(flag)
                else:
                    model_logger.warning("scenario '%s' for line.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_lines, scenario_minor_lines, flag


def validate_generators(generators_dict, scenarios_list, scenario_fuels, scenario_lines, scenario_nodes, model_logger):
    flag = True
    scenario_generators = {s: [] for s in scenarios_list}
    scenario_baseload = {s: [] for s in scenarios_list}

    for idx, item in generators_dict.items():
        for field in [
            "capex",
            "fom",
            "vom",
            "heat_rate_base",
            "heat_rate_incr",
            "initial_capacity",
            "max_build",
            "min_build",
        ]:
            if not validate_range(item[field], 0):
                model_logger.error("'%s' must be float greater than or equal to 0", field)
                flag = False

        if not validate_range(item["discount_rate"], 0, 1):
            model_logger.error("'discount_rate' must be float in range [0,1]")
            flag = False

        if float(item["min_build"]) > float(item["max_build"]):
            model_logger.error("'min_build' must be less than or equal to 'max_build'")
            flag = False

        def _validate_generator(flag):
            if item["name"] in scenario_generators[scenario]:
                model_logger.error("Duplicate generator name '%s' in scenario %s", item["name"], scenario)
                flag = False
            else:
                scenario_generators[scenario].append(item["name"])

            if item["unit_type"] == "baseload":
                scenario_baseload[scenario].append(item["name"])

            if item["node"] not in scenario_nodes[scenario]:
                model_logger.error(
                    "'node' %s for generator %s is not defined in scenario %s", item["node"], item["name"], scenario
                )
                flag = False

            if item["fuel"] not in scenario_fuels[scenario]:
                model_logger.error(
                    "'fuel' %s for generator %s is not defined in scenario %s", item["fuel"], item["name"], scenario
                )
                flag = False

            if item["line"] not in scenario_lines[scenario]:
                model_logger.error(
                    "'line' %s for generator %s is not defined in scenario %s", item["line"], item["name"], scenario
                )
                flag = False
            return flag

        scenarios = parse_list(item["scenarios"])
        if scenarios == ["all"]:
            for scenario in scenarios_list:
                flag = _validate_generator(flag)
        else:
            for scenario in scenarios:
                if scenario in scenarios_list:
                    flag = _validate_generator(flag)
                else:
                    model_logger.warning(
                        "scenario '%s' for generator.id %s not defined in scenarios.csv", scenario, idx
                    )

    return scenario_generators, scenario_baseload, flag


def validate_storages(storages_dict, scenarios_list, scenario_nodes, scenario_lines, model_logger):
    flag = True
    scenario_storages = {s: [] for s in scenarios_list}

    for idx, item in storages_dict.items():
        for field in [
            "capex_p",
            "capex_e",
            "fom",
            "vom",
            "initial_cha_capacity",
            "initial_discha_capacity",
            "initial_energy_capacity",
            "max_build_p",
            "min_build_p",
            "max_build_e",
            "min_build_e",
        ]:
            if not validate_range(item[field], 0):
                model_logger.error("'%s' must be float >= 0", field)
                flag = False

        for bounded in [("min_build_p", "max_build_p"), ("min_build_e", "max_build_e")]:
            if float(item[bounded[0]]) > float(item[bounded[1]]):
                model_logger.error("'%s' must be <= '%s'", bounded[0], bounded[1])
                flag = False

        # If lifetime or duration have a value less than 0, log this, set flag to false and continue
        for field in ["lifetime", "duration"]:
            if int(item[field]) < 0:
                model_logger.error(f"'{field}' must be int >= 0")
                flag = False

        for efficiency in ["charge_efficiency", "discharge_efficiency"]:
            if not validate_range(item[efficiency], 0, 1):
                model_logger.error("'%s' must be float in [0,1]", efficiency)
                flag = False

        if not validate_range(item["discount_rate"], 0, 1):
            model_logger.error("'discount_rate' must be float in [0,1]")
            flag = False

        def _validate_storage(flag):
            if item["name"] in scenario_storages[scenario]:
                model_logger.error("Duplicate storage name '%s' in scenario %s", item["name"], scenario)
                flag = False
            else:
                scenario_storages[scenario].append(item["name"])

            if item["node"] not in scenario_nodes[scenario]:
                model_logger.error(
                    "'node' %s for storage %s is not defined in scenario %s", item["node"], item["name"], scenario
                )
                flag = False

            if item["line"] not in scenario_lines[scenario]:
                model_logger.error(
                    "'line' %s for storage %s is not defined in scenario %s", item["line"], item["name"], scenario
                )
                flag = False
            return flag

        scenarios = parse_list(item["scenarios"])
        if scenarios == ["all"]:
            for scenario in scenarios_list:
                flag = _validate_storage(flag)
        else:
            for scenario in scenarios:
                if scenario in scenarios_list:
                    flag = _validate_storage(flag)
                else:
                    model_logger.warning("scenario '%s' for storage.id %s not defined in scenarios.csv", scenario, idx)

    return scenario_storages, flag


def validate_initial_guess(
    x0s_dict,
    scenarios_list,
    scenario_generators,
    scenario_storages,
    scenario_lines,
    scenario_baseload,
    scenario_minor_lines,
    model_logger,
):
    flag = True
    initial_guess_scenarios = []

    for item in x0s_dict.values():
        scenario = item["scenario"].lower()

        if scenario not in scenarios_list:
            model_logger.warning("scenario '%s'in initial_guess.csv not defined in scenarios.csv", scenario)

        initial_guess_scenarios.append(scenario)

        x0 = parse_list(item["x_0"])

        # TODO: check that works, accounting for 0 bounds

        # bound_length = len(
        #     scenario_generators[scenario]
        #     + scenario_storages[scenario]
        #     + scenario_storages[scenario]
        #     + scenario_lines[scenario]
        # ) - len(scenario_minor_lines[scenario])
        
        # if x0 and not (len(x0) == bound_length):
        #     print(x0)
        #     model_logger.error(
        #         "Initial guess 'x_0' for scenario %s contains %d elements, expected %d", scenario, len(x0), bound_length
        #     )
        #     flag = False

    for scenario in scenarios_list:
        if scenario not in initial_guess_scenarios:
            model_logger.error("scenario '%s'is defined in scenarios.csv but missing from initial_guess.csv", scenario)
            flag = False

    return flag


def validate_datafiles_config(scenario_filenames, scenario_datafile_types, model_logger, datafiles_directory: str):
    valid_types = {"demand", "generation", "reservoir_inflow", "flexible_annual_limit"}
    all_filenames = set(os.listdir(datafiles_directory))
    flag = True

    for fn in scenario_filenames:
        if fn not in all_filenames:
            model_logger.error(f"Missing file data/{fn}")
            flag = False

    for dtype in scenario_datafile_types:
        if not dtype or dtype not in valid_types:
            model_logger.error(f"Invalid or missing datafile_type '{dtype}'")
            flag = False

    return flag


def validate_electricity(node_list, model_logger):
    flag = True
    return flag


def validate_generation(solar_list, wind_list, baseload_list, model_logger):
    flag = True
    return flag


def validate_flexible_limits(flexible_list, model_logger):
    flag = True
    return flag


def validate_config(model_data: ModelData) -> bool:
    config_flag = True
    model_logger = model_data.logger

    if not validate_model_config(model_data.config, model_logger):
        model_logger.error("config.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("config.csv validated!")

    scenarios_list, flag = validate_scenarios(model_data.scenarios, model_logger)
    if not flag:
        model_logger.error("scenarios.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("scenarios.csv validated!")

    scenario_nodes, flag = validate_nodes(model_data.nodes, scenarios_list, model_logger)
    if not flag:
        model_logger.error("nodes.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("nodes.csv validated!")

    scenario_fuels, flag = validate_fuels(model_data.fuels, scenarios_list, model_logger)
    if not flag:
        model_logger.error("fuels.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("fuels.csv validated!")

    scenario_lines, scenario_minor_lines, flag = validate_lines(
        model_data.lines, scenarios_list, scenario_nodes, model_logger
    )
    if not flag:
        model_logger.error("lines.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("lines.csv validated!")

    scenario_generators, scenario_baseload, flag = validate_generators(
        model_data.generators, scenarios_list, scenario_fuels, scenario_lines, scenario_nodes, model_logger
    )
    if not flag:
        model_logger.error("generators.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("generators.csv validated!")

    scenario_storages, flag = validate_storages(
        model_data.storages, scenarios_list, scenario_nodes, scenario_lines, model_logger
    )
    if not flag:
        model_logger.error("storages.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("storages.csv validated!")

    if not validate_initial_guess(
        model_data.x0s,
        scenarios_list,
        scenario_generators,
        scenario_storages,
        scenario_lines,
        scenario_baseload,
        scenario_minor_lines,
        model_logger,
    ):
        model_logger.error("initial_guess.csv contains errors.")
        config_flag = False
    else:
        model_logger.info("initial_guess.csv validated!")

    return config_flag


def validate_data(all_datafiles, scenario_name, model_logger, datafiles_directory: str):
    flag = True
    scenario_filenames = []
    scenario_datafile_types = []

    for item in all_datafiles.values():
        scenario_list = parse_comma_separated(item["scenarios"])
        if scenario_name in scenario_list:
            scenario_filenames.append(item["filename"])
            scenario_datafile_types.append(item["datafile_type"])

    if not validate_datafiles_config(scenario_filenames, scenario_datafile_types, model_logger, datafiles_directory):
        model_logger.error(f"datafiles.csv contains errors for scenario {scenario_name}.")
        flag = False
    else:
        model_logger.info(f"datafiles.csv validated for scenario {scenario_name}!")

    """ if not validate_electricity(model_logger):
        model_logger.error(f'Demand profiles contain errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'Demand profiles validated for scenario {scenario_name}!')

    if not validate_generation(model_logger):
        model_logger.error(f'Generation traces contain errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'Generation traces validated for scenario {scenario_name}!')

    if not validate_flexible_limits(model_logger):
        model_logger.error(f'Flexible limits contains errors for scenario {scenario_name}.')
        flag = False
    else:
        model_logger.info(f'Flexible limits validated for scenario {scenario_name}!') """

    return flag


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
            if not str(item).lower() in ('false', 'none', 'true'):
                continue
        else:  # numeric
            # Assign to cast_item to prevent overwriting the original string
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
        raise TypeError(f"dtype of {param_name} was not of acceptable type or out of bounds (got: {item} of type: {type(item)})")
    return typepass


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
    "mga_tourn_count": {"default": 0.8, "ditherable": False, "broadcastable": True, "types": ((float, 0, 1), (int, -1, np.inf),)},
    "mga_elite_count": {"default": 0.2, "ditherable": False, "broadcastable": True, "types": ((float, 0, 1), (int, -1, np.inf),)},
    "mga_champ_count": {"default": 0, "ditherable": False, "broadcastable": True, "types": ((float, 0, 1), (int, -1, np.inf),)},
    "mga_start_niches": {"default": 10, "ditherable": False, "broadcastable": False, "types": ((int, 1, np.inf),)},
    "mga_new_niches": {"default": 0, "ditherable": False, "broadcastable": True, "types": ((int, 0, np.inf),)},
    "mga_niche_elitism": {"default": True, "ditherable": False, "broadcastable": True, "types": ((bool, ("none", "selfish", "unselfish")),)},
    "mga_log_freq": {"default": 1, "ditherable": False, "broadcastable": False, "types": ((int, -1, np.inf),)},
    "mga_disp_rate": {"default": 1, "ditherable": False, "broadcastable": True, "types": ((int, -1, np.inf),)},
    "mga_verbose_level": {"default": 3, "ditherable": False, "broadcastable": True, "types": ((int, 0, np.inf),)},
    "mga_fitness": {"default": "angular", "ditherable": False, "broadcastable": True, "types": ((str, ("angular", "l2", "l1")),)},
}


def parse_and_validate_mga(config_dict, model_logger):
    """Parses, broadcasts, validates, and explicitly casts MHMGA parameters."""
    flag = True
    config_by_name = {item["name"]: item for item in config_dict.values()}
    
    # 1. Establish mga_steps
    mga_steps = 1
    if "mga_steps" in config_by_name:
        try:
            mga_steps = int(config_by_name["mga_steps"]["value"])
            config_by_name["mga_steps"]["value"] = mga_steps
        except ValueError:
            model_logger.error("'mga_steps' must be an integer")
            flag = False

    # Helper function to correctly cast resolved types
    def cast_value(val, target_type):
        if target_type is bool:
            return str(val).lower() in ("true", "t", "1", "yes", "y")
        return target_type(val)

    # 2. Parse and Broadcast all schema properties
    for param_name, param_dict in expected_mga_hyperparameters.items():
        string = config_by_name.get(param_name, {}).get("value", param_dict["default"])
        
        try:
            if param_dict["ditherable"]:
                value = np.array(parse_ditherable_hyperparameter(str(string)), dtype=object)
                if value.shape[0] == 1:
                    value = np.stack((value[0],) * mga_steps)
                elif value.shape[0] != mga_steps:
                    raise ValueError(f"{param_name} not broadcastable to mga_steps")
                
                # Apply validation and casting across the dithered array
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
                    
                # Apply validation and casting across the broadcasted list
                for i, item in enumerate(value):
                    valid_type = check_type(param_name, param_dict, item)
                    value[i] = cast_value(item, valid_type)
                final_value = value

            else:
                # Apply validation and casting for standard scalars
                valid_type = check_type(param_name, param_dict, string)
                final_value = cast_value(string, valid_type)

            # Write the cast Python object back into the raw config dictionary
            if param_name in config_by_name:
                config_by_name[param_name]["value"] = final_value
            else:
                config_dict[f"auto_{param_name}"] = {"name": param_name, "value": final_value}

        except (ValueError, TypeError) as e:
            model_logger.error(f"Validation failed for '{param_name}': {str(e)}")
            flag = False

    return flag
