# type: ignore
from typing import Dict

import numpy as np

from firm_ce.common.typing import DictType, TypedDict, int64
from firm_ce.constructors.cost_cons import construct_UnitCost_object
from firm_ce.system.components import (
    Fleet,
    Fleet_InstanceType,
    Fuel,
    Fuel_InstanceType,
    Generator,
    Generator_InstanceType,
    Reservoir,
    Reservoir_InstanceType,
    Storage,
    Storage_InstanceType,
)
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType


def construct_Fuel_object(
        fuel_dict: Dict[str, str]
) -> Fuel_InstanceType:
    """
    Takes a fuel_dict, casts values into Numba-compatible types, and
    returns an instance of the Fuel jitclass.

    Parameters:
    -------
    fuel_dict (Dict[str,str]): A dictionary containing the attributes of
        a single fuel object in `config/fules.csv`.

    Returns:
    -------
    Fuel_InstanceType: A static instance of the Fuel jitclass.
    """
    idx = int(fuel_dict["id"])
    name = str(fuel_dict["name"])
    cost = float(fuel_dict["cost"])
    emissions = float(fuel_dict["emissions"])
    return Fuel(True, idx, name, cost, emissions)


def construct_Generator_object(
    generator_dict: Dict[str, str],
    fuels_imported_dict: Dict[str, Dict[str, str]],
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
    order: int,
) -> Generator_InstanceType:
    """
    Takes data required to initialise a single generator object,
    casts values into Numba-compatible types, and returns an
    instance of the Generator jitclass.

    Parameters:
    -------
    generator_dict (Dict[str,str]): A dictionary containing the attributes of
        a single generator object in `config/generators.csv`.
    fuels_imported_dict (Dict[str, Dict[str, str]]): A dictionary containing data for all fuels
        imported from `config/fules.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.
    order (int): The scenario-specific id for the Generator instance.

    Returns:
    -------
    Generator_InstanceType: A static instance of the Generator jitclass.
    """
    idx = int(generator_dict["id"])
    name = str(generator_dict["name"])
    unit_size = float(generator_dict["unit_size"])
    max_build = float(generator_dict["max_build"])
    min_build = float(generator_dict["min_build"])
    capacity = float(generator_dict["initial_capacity"])
    unit_type = str(generator_dict["unit_type"])
    near_optimum_check = str(generator_dict.get("near_optimum", "")).lower() in ("true", "1", "yes")

    node = next(node for node in nodes_object_dict.values() if node.name == str(generator_dict["node"]))

    fuel_dict = next(
        fuel_dict for fuel_dict in fuels_imported_dict.values() if fuel_dict["name"] == str(generator_dict["fuel"])
    )
    fuel = construct_Fuel_object(fuel_dict)

    line = next(line for line in lines_object_dict.values() if line.name == str(generator_dict["line"]))

    raw_group = generator_dict.get("range_group", "")
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == "":
        group = name
    else:
        group = str(raw_group).strip()

    cost = construct_UnitCost_object(
        capex_p=float(generator_dict["capex"]),
        fom=float(generator_dict["fom"]),
        vom=float(generator_dict["vom"]),
        lifetime=int(generator_dict["lifetime"]),
        discount_rate=float(generator_dict["discount_rate"]),
        heat_rate_base=float(generator_dict["heat_rate_base"]),
        heat_rate_incr=float(generator_dict["heat_rate_incr"]),
        fuel=fuel,
    )

    return Generator(
        True,
        idx,
        order,
        name,
        unit_size,
        max_build,
        min_build,
        capacity,
        unit_type,
        near_optimum_check,
        node,
        fuel,
        line,
        group,
        cost,
    )


def construct_Reservoir_object(
    reservoir_dict: Dict[str, str],
    fuels_imported_dict: Dict[str, Dict[str, str]],
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
    order: int,
) -> Reservoir_InstanceType:
    """
    Takes data required to initialise a single storage object,
    casts values into Numba-compatible types, and returns an
    instance of the Storage jitclass.

    Parameters:
    -------
    reservoir_dict (Dict[str,str]): A dictionary containing the attributes of
        a single reservoir object in `config/reservoirs.csv`.
    fuels_imported_dict (Dict[str, Dict[str, str]]): A dictionary containing data for all fuels
        imported from `config/fules.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.
    order (int): The scenario-specific id for the Reservoir instance.

    Returns:
    -------
    Reservoir_InstanceType: A static instance of the Reservoir jitclass.
    """
    idx = int(reservoir_dict["id"])
    name = str(reservoir_dict["name"])
    unit_size = float(reservoir_dict["unit_size"])
    power_capacity = float(reservoir_dict["initial_power_capacity"])

    duration = int(reservoir_dict["duration"]) if int(reservoir_dict["duration"]) > 0 else 0
    if duration == 0:
        energy_capacity = float(reservoir_dict["initial_energy_capacity"])
        duration = int(energy_capacity / power_capacity) if power_capacity > 0 else 0
    else:
        energy_capacity = float(power_capacity * duration)

    discharge_efficiency = float(reservoir_dict["discharge_efficiency"])
    max_build_p = float(reservoir_dict["max_build_p"])
    max_build_e = float(reservoir_dict["max_build_e"])
    min_build_p = float(reservoir_dict["min_build_p"])
    min_build_e = float(reservoir_dict["min_build_e"])
    unit_type = str(reservoir_dict["unit_type"])
    near_optimum_check = str(reservoir_dict.get("near_optimum", "")).lower() in ("true", "1", "yes")

    node = next(node for node in nodes_object_dict.values() if node.name == str(reservoir_dict["node"]))

    fuel_dict = next(
        fuel_dict for fuel_dict in fuels_imported_dict.values() if fuel_dict["name"] == str(reservoir_dict["fuel"])
    )
    fuel = construct_Fuel_object(fuel_dict)

    line = next(line for line in lines_object_dict.values() if line.name == str(reservoir_dict["line"]))

    raw_group = reservoir_dict.get("range_group", "")
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == "":
        group = name
    else:
        group = str(raw_group).strip()

    cost = construct_UnitCost_object(
        capex_p=float(reservoir_dict["capex_p"]),
        fom=float(reservoir_dict["fom"]),
        vom=float(reservoir_dict["vom"]),
        lifetime=int(reservoir_dict["lifetime"]),
        discount_rate=float(reservoir_dict["discount_rate"]),
        capex_e=float(reservoir_dict["capex_e"]),
        fuel=fuel,
    )
    return Reservoir(
        True,
        idx,
        order,
        name,
        unit_size,
        power_capacity,
        energy_capacity,
        duration,
        discharge_efficiency,
        max_build_p,
        max_build_e,
        min_build_p,
        min_build_e,
        unit_type,
        near_optimum_check,
        node,
        fuel,
        line,
        group,
        cost,
    )


def construct_Storage_object(
    storage_dict: Dict[str, str],
    nodes_object_dict: DictType(int64, Node_InstanceType),
    lines_object_dict: DictType(int64, Line_InstanceType),
    order: int,
) -> Storage_InstanceType:
    """
    Takes data required to initialise a single storage object,
    casts values into Numba-compatible types, and returns an
    instance of the Storage jitclass.

    Parameters:
    -------
    storage_dict (Dict[str,str]): A dictionary containing the attributes of
        a single storage object in `config/storages.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.
    order (int): The scenario-specific id for the Storage instance.

    Returns:
    -------
    Storage_InstanceType: A static instance of the Storage jitclass.
    """
    idx = int(storage_dict["id"])
    name = str(storage_dict["name"])
    power_capacity = float(storage_dict["initial_power_capacity"])

    duration = int(storage_dict["duration"]) if int(storage_dict["duration"]) > 0 else 0
    if duration == 0:
        energy_capacity = float(storage_dict["initial_energy_capacity"])
        duration = int(energy_capacity / power_capacity) if power_capacity > 0 else 0
    else:
        energy_capacity = float(power_capacity * duration)

    charge_efficiency = float(storage_dict["charge_efficiency"])
    discharge_efficiency = float(storage_dict["discharge_efficiency"])
    max_build_p = float(storage_dict["max_build_p"])
    max_build_e = float(storage_dict["max_build_e"])
    min_build_p = float(storage_dict["min_build_p"])
    min_build_e = float(storage_dict["min_build_e"])
    unit_type = str(storage_dict["unit_type"])
    near_optimum_check = str(storage_dict.get("near_optimum", "")).lower() in ("true", "1", "yes")

    node = next(node for node in nodes_object_dict.values() if node.name == str(storage_dict["node"]))

    line = next(line for line in lines_object_dict.values() if line.name == str(storage_dict["line"]))

    raw_group = storage_dict.get("range_group", "")
    if raw_group is None or (isinstance(raw_group, float) and np.isnan(raw_group)) or str(raw_group).strip() == "":
        group = name
    else:
        group = str(raw_group).strip()

    cost = construct_UnitCost_object(
        capex_p=float(storage_dict["capex_p"]),
        fom=float(storage_dict["fom"]),
        vom=float(storage_dict["vom"]),
        lifetime=int(storage_dict["lifetime"]),
        discount_rate=float(storage_dict["discount_rate"]),
        capex_e=float(storage_dict["capex_e"]),
    )
    return Storage(
        True,
        idx,
        order,
        name,
        power_capacity,
        energy_capacity,
        duration,
        charge_efficiency,
        discharge_efficiency,
        max_build_p,
        max_build_e,
        min_build_p,
        min_build_e,
        unit_type,
        near_optimum_check,
        node,
        line,
        group,
        cost,
    )


def construct_Fleet_object(
    generators_imported_dict: Dict[str, Dict[str, str]],
    reservoirs_imported_dict: Dict[str, Dict[str, str]],
    storages_imported_dict: Dict[str, Dict[str, str]],
    fuels_imported_dict: Dict[str, Dict[str, str]],
    lines_object_dict: DictType(int64, Line_InstanceType),
    nodes_object_dict: DictType(int64, Node_InstanceType),
) -> Fleet_InstanceType:
    """
    Takes data required to initialise a single fleet object, casts values into Numba-compatible
    types, and returns an instance of the Fleet jitclass. The Fleet consists of all assets that
    can be dispatched in the energy balance.

    Parameters:
    -------
    generators_imported_dict (Dict[str, Dict[str, str]]): A dictionary containing data for all
        generators imported from `config/generators.csv`.
    reservoirs_imported_dict (Dict[str, Dict[str, str]]): A dictionary containing data for all
        reservoirs imported from `config/reservoirs.csv`.
    storages_imported_dict (Dict[str, Dict[str, str]]): A dictionary containing data for all
        storages imported from `config/storages.csv`.
    fuels_imported_dict (Dict[str, Dict[str, str]]): A dictionary containing data for all fuels
        imported from `config/fules.csv`.
    nodes_object_dict (DictType(int64, Node_InstanceType)): A typed dictionary of
        all Node jitclass instances for the scenario. Key defined as Node.order.
    lines_object_dict (DictType(int64, Line_InstanceType)): A typed dictionary of
        all Line jitclass instances for the scenario. Key defined as Line.order.

    Returns:
    -------
    Fleet_InstanceType: A static instance of the Fleet jitclass.
    """

    generators = TypedDict.empty(key_type=int64, value_type=Generator_InstanceType)
    for order, idx in enumerate(generators_imported_dict):
        generators[order] = construct_Generator_object(
            generators_imported_dict[idx],
            fuels_imported_dict,
            nodes_object_dict,
            lines_object_dict,
            order,
        )

    reservoirs = TypedDict.empty(key_type=int64, value_type=Reservoir_InstanceType)
    for order, idx in enumerate(reservoirs_imported_dict):
        reservoirs[order] = construct_Reservoir_object(
            reservoirs_imported_dict[idx],
            fuels_imported_dict,
            nodes_object_dict,
            lines_object_dict,
            order,
        )

    storages = TypedDict.empty(key_type=int64, value_type=Storage_InstanceType)
    for order, idx in enumerate(storages_imported_dict):
        storages[order] = construct_Storage_object(
            storages_imported_dict[idx],
            nodes_object_dict,
            lines_object_dict,
            order,
        )

    return Fleet(
        True,
        generators,
        reservoirs,
        storages,
    )
