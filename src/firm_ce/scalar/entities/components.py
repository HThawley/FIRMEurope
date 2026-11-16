# type: ignore
from typing import Dict

import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.helpers import parse_boolean
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import (
    DictType,
    TypedDict,
    boolean,
    nbfloat,
    nbintp,
    npfloat,
    unicode_type,
)
from firm_ce.scalar.entities.costs import (
    LTCosts,
    LTCosts_InstanceType,
    UnitCost_InstanceType,
    construct_UnitCost_object,
)
from firm_ce.scalar.entities.topology import Line_InstanceType, Node_InstanceType

if JIT_ENABLED:
    fuel_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("id", nbintp),
        ("name", unicode_type),
        ("cost", nbfloat),
        ("emissions", nbfloat),
        ("annual_limit", nbfloat[:]),
        ("remaining_energy", nbfloat[:]),
        ("data_status", boolean),
        # Precharging
        ("remaining_energy_temp_reverse", nbfloat),
        ("remaining_energy_temp_forward", nbfloat),
        ("deficit_block_max_energy", nbfloat),
        ("deficit_block_min_energy", nbfloat),
        ("trickling_flag", boolean),
        ("trickling_reserves", nbfloat),
        ("remaining_trickling_reserves", nbfloat),
        ("allocated_energy", nbfloat),
        ("allocated_trickling", nbfloat),
    ]
else:
    fuel_spec = []


@jitclass(fuel_spec)
class Fuel:
    """
    Represents a fuel type with associated cost, emissions, and annual generation limits.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: nbintp,
        name: unicode_type,
        cost: nbfloat,
        emissions: nbfloat,
    ) -> None:
        self.object_class = "fuel"
        self.static_instance = static_instance
        self.id = idx
        self.name = name
        self.cost = cost  # $/GJ
        self.emissions = emissions  # kg/GJ
        self.annual_limit = np.zeros((0,), dtype=npfloat)  # GWh/year
        self.remaining_energy = np.zeros((0,), dtype=npfloat)  # GWh

        self.data_status = False
        # Precharging
        self.remaining_energy_temp_reverse = 0.0  # GWh
        self.remaining_energy_temp_forward = 0.0  # GWh
        self.deficit_block_max_energy = 0.0  # GWh
        self.deficit_block_min_energy = 0.0  # GWh
        self.trickling_flag = False
        self.trickling_reserves = 0.0  # GWh
        self.remaining_trickling_reserves = 0.0  # GWh
        self.allocated_energy = 0.0  # GWh
        self.allocated_trickling = 0.0  # GWh


if JIT_ENABLED:
    Fuel_InstanceType = Fuel.class_type.instance_type
else:
    Fuel_InstanceType = Fuel

if JIT_ENABLED:
    generator_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("id", nbintp),
        ("order", nbintp),
        ("name", unicode_type),
        ("node", Node_InstanceType),
        ("fuel", Fuel_InstanceType),
        ("unit_size", nbfloat),
        ("max_build", nbfloat),
        ("min_build", nbfloat),
        ("initial_capacity", nbfloat),
        ("line", Line_InstanceType),
        ("unit_type", unicode_type),
        ("unit_type_idx", nbintp),
        ("is_flexible", boolean),
        ("cost", UnitCost_InstanceType),
        ("data_status", boolean),
        ("data", nbfloat[:]),
        ("candidate_x_idx", nbintp),
        # Dynamic
        ("new_build", nbfloat),
        ("capacity", nbfloat),
        ("dispatch_power", nbfloat[:]),
        ("flexible_max_t", nbfloat),
        ("lt_generation", nbfloat),
        ("unit_lt_hours", nbfloat),
        ("lt_costs", LTCosts_InstanceType),
        ("heat_base_consumption", nbfloat),
    ]
else:
    generator_spec = []


@jitclass(generator_spec)
class Generator:
    """
    Represents a generator unit within the system.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: nbintp,
        order: nbintp,
        name: unicode_type,
        unit_size: nbfloat,
        max_build: nbfloat,
        min_build: nbfloat,
        capacity: nbfloat,
        unit_type: unicode_type,
        is_flexible: boolean,
        node: Node_InstanceType,
        fuel: Fuel_InstanceType,
        line: Line_InstanceType,
        cost: UnitCost_InstanceType,
    ) -> None:
        self.object_class = "generator"
        self.static_instance = static_instance
        self.id = idx
        self.order = order
        self.name = name
        self.unit_size = unit_size  # GW/unit
        self.max_build = max_build  # GW/year
        self.min_build = min_build  # GW/year
        self.initial_capacity = capacity  # GW
        self.unit_type = unit_type
        self.is_flexible = is_flexible
        self.node = node
        self.fuel = fuel
        self.line = line
        self.cost = cost

        self.data_status = False
        self.data = np.empty((0,), dtype=npfloat)

        self.candidate_x_idx = -1

        # Dynamic
        self.new_build = 0.0  # GW
        self.capacity = capacity  # GW
        self.dispatch_power = np.empty((0,), dtype=npfloat)  # GW

        self.flexible_max_t = 0.0  # GW
        self.lt_generation = 0.0  # GWh
        self.unit_lt_hours = 0.0  # hours/unit

        self.heat_base_consumption = 0.0  # MWh/unit/h

        self.lt_costs = LTCosts()


if JIT_ENABLED:
    Generator_InstanceType = Generator.class_type.instance_type
else:
    Generator_InstanceType = Generator

if JIT_ENABLED:
    storage_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("id", nbintp),
        ("order", nbintp),
        ("name", unicode_type),
        ("node", Node_InstanceType),
        ("initial_power_capacity", nbfloat),
        ("initial_energy_capacity", nbfloat),
        ("duration", nbfloat),
        ("chargeable", boolean),
        ("inflows", boolean),
        ("charge_efficiency", nbfloat),
        ("discharge_efficiency", nbfloat),
        ("max_build_p", nbfloat),
        ("max_build_e", nbfloat),
        ("min_build_p", nbfloat),
        ("min_build_e", nbfloat),
        ("line", Line_InstanceType),
        ("unit_type", unicode_type),
        ("unit_type_idx", nbintp),
        ("cost", UnitCost_InstanceType),
        ("candidate_p_x_idx", nbintp),
        ("candidate_e_x_idx", nbintp),
        ("data_status", boolean),
        ("data", nbfloat[:]),
        # Dynamic
        ("new_build_p", nbfloat),
        ("new_build_e", nbfloat),
        ("power_capacity", nbfloat),
        ("energy_capacity", nbfloat),
        ("dispatch_power", nbfloat[:]),
        ("stored_energy", nbfloat[:]),
        ("discharge_max_t", nbfloat),
        ("charge_max_t", nbfloat),
        ("lt_generation", nbfloat),
        ("unit_lt_hours", nbfloat),
        ("lt_costs", LTCosts_InstanceType),
        # Precharging & Reserves
        ("deficit_block_min_storage", nbfloat),
        ("deficit_block_max_storage", nbfloat),
        ("stored_energy_temp_reverse", nbfloat),
        ("stored_energy_temp_forward", nbfloat),
        ("precharge_energy", nbfloat),
        ("trickling_reserves", nbfloat),
        ("remaining_trickling_reserves", nbfloat),
        ("precharge_flag", boolean),
        ("trickling_flag", boolean),
        ("remaining_discharge_max_t", nbfloat),
        ("remaining_charge_max_t", nbfloat),
    ]
else:
    storage_spec = []


@jitclass(storage_spec)
class Storage:
    def __init__(
        self,
        static_instance: boolean,
        idx: nbintp,
        order: nbintp,
        name: unicode_type,
        power_capacity: nbfloat,
        energy_capacity: nbfloat,
        duration: nbfloat,
        chargeable: boolean,
        inflows: boolean,
        charge_efficiency: nbfloat,
        discharge_efficiency: nbfloat,
        max_build_p: nbfloat,
        max_build_e: nbfloat,
        min_build_p: nbfloat,
        min_build_e: nbfloat,
        unit_type: unicode_type,
        node: Node_InstanceType,
        line: Line_InstanceType,
        cost: UnitCost_InstanceType,
    ) -> None:
        self.object_class = "storage"
        self.static_instance = static_instance
        self.id = idx
        self.order = order
        self.name = name
        self.initial_power_capacity = power_capacity  # GW
        self.duration = duration  # hours
        self.initial_energy_capacity = energy_capacity if duration == 0 else duration * power_capacity  # GWh
        self.chargeable = chargeable
        self.inflows = inflows
        self.charge_efficiency = charge_efficiency  # unitless
        self.discharge_efficiency = discharge_efficiency  # unitless
        self.max_build_p = max_build_p  # GW/year
        self.max_build_e = max_build_e  # GWh/year
        self.min_build_p = min_build_p  # GW/year
        self.min_build_e = min_build_e  # GWh/year
        self.unit_type = unit_type
        self.node = node
        self.line = line
        self.cost = cost

        self.candidate_p_x_idx = -1
        self.candidate_e_x_idx = -1
        if self.inflows:
            self.data_status = False
            self.data = np.empty((0,), dtype=npfloat)

        # Dynamic
        self.new_build_p = 0.0  # GW
        self.new_build_e = 0.0  # GWh
        self.power_capacity = power_capacity  # GW
        self.energy_capacity = self.initial_energy_capacity  # GWh
        self.dispatch_power = np.empty((0,), dtype=npfloat)  # GW
        self.stored_energy = np.empty((0,), dtype=npfloat)  # GWh

        self.discharge_max_t = 0.0  # GW
        self.charge_max_t = 0.0  # GW
        self.lt_generation = 0.0  # GWh
        self.unit_lt_hours = 0.0  # hours/unit
        self.lt_costs = LTCosts()

        # Precharging & Reserves
        self.stored_energy_temp_reverse = 0.0  # GWh
        self.stored_energy_temp_forward = 0.0  # GWh
        self.deficit_block_min_storage = 0.0  # GWh
        self.deficit_block_max_storage = 0.0  # GWh
        self.precharge_energy = 0.0  # GWh
        self.trickling_reserves = 0.0  # GWh
        self.remaining_trickling_reserves = 0.0  # GWh
        self.precharge_flag = False
        self.trickling_flag = False

        self.remaining_discharge_max_t = 0.0  # GW
        self.remaining_charge_max_t = 0.0  # GW


if JIT_ENABLED:
    Storage_InstanceType = Storage.class_type.instance_type
else:
    Storage_InstanceType = Storage


if JIT_ENABLED:
    fleet_spec = [
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("generators", DictType(nbintp, Generator_InstanceType)),
        ("storages", DictType(nbintp, Storage_InstanceType)),
        ("fuels", DictType(nbintp, Fuel_InstanceType)),
    ]
else:
    fleet_spec = []


@jitclass(fleet_spec)
class Fleet:
    """
    Represents a collection of Generators, Storage systems, and Fuels in the scenario.
    """

    def __init__(
        self,
        static_instance: boolean,
        generators: DictType(nbintp, Generator_InstanceType),
        storages: DictType(nbintp, Storage_InstanceType),
        fuels: DictType(nbintp, Fuel_InstanceType),
    ):
        self.object_class = "fleet"
        self.static_instance = static_instance
        self.generators = generators
        self.storages = storages
        self.fuels = fuels


if JIT_ENABLED:
    Fleet_InstanceType = Fleet.class_type.instance_type
else:
    Fleet_InstanceType = Fleet


def construct_Fuel_object(fuel_dict: Dict[str, str]) -> Fuel_InstanceType:
    idx = int(fuel_dict["id"])
    name = str(fuel_dict["name"])
    cost = float(fuel_dict["cost"])
    emissions = float(fuel_dict["emissions"])
    return Fuel(True, idx, name, cost, emissions)


def construct_Generator_object(
    generator_dict: Dict[str, str],
    fuels_imported_dict: Dict[str, Dict[str, str]],
    nodes_object_dict: DictType(nbintp, Node_InstanceType),
    lines_object_dict: DictType(nbintp, Line_InstanceType),
    order: int,
) -> Generator_InstanceType:
    idx = int(generator_dict["id"])
    name = str(generator_dict["name"])
    unit_size = float(generator_dict["unit_size"])
    max_build = float(generator_dict["max_build"])
    min_build = float(generator_dict["min_build"])
    capacity = float(generator_dict["initial_capacity"])
    unit_type = str(generator_dict["unit_type"])
    is_flexible = parse_boolean(generator_dict["is_flexible"])

    node = next(node for node in nodes_object_dict.values() if node.name == str(generator_dict["node"]))

    fuel_dict = next(
        fuel_dict for fuel_dict in fuels_imported_dict.values() if fuel_dict["name"] == str(generator_dict["fuel"])
    )
    fuel = construct_Fuel_object(fuel_dict)

    line = next(line for line in lines_object_dict.values() if line.name == str(generator_dict["line"]))

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
        is_flexible,
        node,
        fuel,
        line,
        cost,
    )


def construct_Storage_object(
    storage_dict: Dict[str, str],
    nodes_object_dict: DictType(nbintp, Node_InstanceType),
    lines_object_dict: DictType(nbintp, Line_InstanceType),
    order: int,
) -> Storage_InstanceType:
    idx = int(storage_dict["id"])
    name = str(storage_dict["name"])

    chargeable = parse_boolean(storage_dict["chargeable"])
    inflows = parse_boolean(storage_dict["inflows"])

    power_capacity = float(storage_dict["initial_discha_capacity"])

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

    node = next(node for node in nodes_object_dict.values() if node.name == str(storage_dict["node"]))
    line = next(line for line in lines_object_dict.values() if line.name == str(storage_dict["line"]))

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
        chargeable,
        inflows,
        charge_efficiency,
        discharge_efficiency,
        max_build_p,
        max_build_e,
        min_build_p,
        min_build_e,
        unit_type,
        node,
        line,
        cost,
    )


def construct_Fleet_object(
    generators_imported_dict: Dict[str, Dict[str, str]],
    storages_imported_dict: Dict[str, Dict[str, str]],
    fuels_imported_dict: Dict[str, Dict[str, str]],
    lines_object_dict: DictType(nbintp, Line_InstanceType),
    nodes_object_dict: DictType(nbintp, Node_InstanceType),
) -> Fleet_InstanceType:
    fuels = TypedDict.empty(key_type=nbintp, value_type=Fuel_InstanceType)
    generators = TypedDict.empty(key_type=nbintp, value_type=Generator_InstanceType)
    for order, idx in enumerate(generators_imported_dict):
        generators[order] = construct_Generator_object(
            generators_imported_dict[idx],
            fuels_imported_dict,
            nodes_object_dict,
            lines_object_dict,
            order,
        )
        if generators[order].fuel.id not in fuels:
            fuels[generators[order].fuel.id] = generators[order].fuel

    storages = TypedDict.empty(key_type=nbintp, value_type=Storage_InstanceType)
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
        storages,
        fuels,
    )
