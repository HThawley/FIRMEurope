import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.system.costs import LTCosts, LTCosts_InstanceType, UnitCost_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType

if JIT_ENABLED:
    fuel_spec = [
        ("static_instance", boolean),
        ("id", int64),
        ("name", unicode_type),
        ("cost", float64),
        ("emissions", float64),
    ]
else:
    fuel_spec = []


@jitclass(fuel_spec)
class Fuel:
    """
    Represents a fuel type with associated cost and emissions.
    """

    def __init__(
        self, static_instance: boolean, idx: int64, name: unicode_type, cost: float64, emissions: float64
    ) -> None:
        """
        Initialize a Fuel object.

        Parameters:
        -------
        id (int): Unique identifier for the fuel.
        fuel_dict (Dict[str, str]): Dictionary containing 'name', 'cost', and 'emissions' keys.
        """

        self.static_instance = static_instance
        self.id = idx
        self.name = name
        self.cost = cost  # $/GJ
        self.emissions = emissions  # kg/GJ


if JIT_ENABLED:
    Fuel_InstanceType = Fuel.class_type.instance_type
else:
    Fuel_InstanceType = Fuel

if JIT_ENABLED:
    generator_spec = [
        ("static_instance", boolean),
        ("id", int64),
        ("order", int64),
        ("name", unicode_type),
        ("node", Node_InstanceType),
        ("fuel", Fuel_InstanceType),
        ("unit_size", float64),
        ("max_build", float64),
        ("min_build", float64),
        ("initial_capacity", float64),
        ("line", Line_InstanceType),
        ("unit_type", unicode_type),
        ("near_optimum_check", boolean),
        ("group", unicode_type),
        ("cost", UnitCost_InstanceType),
        ("data_status", unicode_type),
        ("data", float64[:]),
        ("annual_constraints_data", float64[:]),
        ("candidate_x_idx", int64),
        # Dynamic
        ("new_build", float64),
        ("capacity", float64),
        ("dispatch_power", float64[:]),
        ("remaining_energy", float64[:]),
        ("flexible_max_t", float64),
        ("lt_generation", float64),
        ("unit_lt_hours", float64),
        ("lt_costs", LTCosts_InstanceType),
        # Precharging
        ("remaining_energy_temp_reverse", float64),
        ("remaining_energy_temp_forward", float64),
        ("deficit_block_max_energy", float64),
        ("deficit_block_min_energy", float64),
        ("trickling_flag", boolean),
        ("trickling_reserves", float64),
        ("remaining_trickling_reserves", float64),
    ]
else:
    generator_spec = []


@jitclass(generator_spec)
class Generator:
    """
    Represents a generator unit within the system.

    Solar, wind and baseload generators require generation trace datafiles. Flexible
    generators require datafiles for annual generation limits. Datafiles must be stored in
    the 'data' folder and referenced in 'config/datafiles.csv'.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: int64,
        order: int64,
        name: unicode_type,
        unit_size: float64,
        max_build: float64,
        min_build: float64,
        capacity: float64,
        unit_type: unicode_type,
        near_optimum_check: boolean,
        node: Node_InstanceType,
        fuel: Fuel_InstanceType,
        line: Line_InstanceType,
        group: unicode_type,
        cost: UnitCost_InstanceType,
    ) -> None:
        """
        Initialize a Generator object.

        Parameters:
        -------
        id (int): Unique identifier for the generator.
        generator_dict (Dict[str, str]): Dictionary containing generator attributes.
        fuel (Fuel): The associated fuel object.
        line (Line): The generic minor line defined to connect the generator to the transmission network.
                        Minor lines should have empty node_start and node_end values. They do not form part
                        of the network topology, but are used to estimate connection costs.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order  # id specific to scenario
        self.name = name
        self.unit_size = unit_size  # GW/unit
        self.max_build = max_build  # GW/year
        self.min_build = min_build  # GW/year
        self.initial_capacity = capacity  # GW
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.node = node
        self.fuel = fuel
        self.line = line
        self.group = group
        self.cost = cost

        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)
        self.annual_constraints_data = np.empty((0,), dtype=np.float64)

        self.candidate_x_idx = -1

        # Dynamic
        self.new_build = 0.0  # GW
        self.capacity = capacity  # GW
        self.dispatch_power = np.empty((0,), dtype=np.float64)  # GW
        self.remaining_energy = np.empty((0,), dtype=np.float64)  # GWh

        self.flexible_max_t = 0.0  # GW
        self.lt_generation = 0.0  # GWh
        self.unit_lt_hours = 0.0  # hours/unit

        self.lt_costs = LTCosts()

        # Precharging
        self.remaining_energy_temp_reverse = 0.0  # GWh
        self.remaining_energy_temp_forward = 0.0  # GWh
        self.deficit_block_max_energy = 0.0  # GWh
        self.deficit_block_min_energy = 0.0  # GWh
        self.trickling_flag = False  # Determines whether flexible generator can precharge storage systems
        self.trickling_reserves = 0.0  # GWh
        self.remaining_trickling_reserves = 0.0  # GWh


if JIT_ENABLED:
    Generator_InstanceType = Generator.class_type.instance_type
else:
    Generator_InstanceType = Generator

if JIT_ENABLED:
    storage_spec = [
        ("static_instance", boolean),
        ("id", int64),
        ("order", int64),
        ("name", unicode_type),
        ("node", Node_InstanceType),
        ("initial_power_capacity", float64),
        ("initial_energy_capacity", float64),
        ("duration", int64),
        ("charge_efficiency", float64),
        ("discharge_efficiency", float64),
        ("max_build_p", float64),
        ("max_build_e", float64),
        ("min_build_p", float64),
        ("min_build_e", float64),
        ("line", Line_InstanceType),
        ("unit_type", unicode_type),
        ("near_optimum_check", boolean),
        ("group", unicode_type),
        ("cost", UnitCost_InstanceType),
        ("candidate_p_x_idx", int64),
        ("candidate_e_x_idx", int64),
        # Dynamic
        ("new_build_p", float64),
        ("new_build_e", float64),
        ("power_capacity", float64),
        ("energy_capacity", float64),
        ("dispatch_power", float64[:]),
        ("stored_energy", float64[:]),
        ("discharge_max_t", float64),
        ("charge_max_t", float64),
        ("lt_discharge", float64),
        ("lt_costs", LTCosts_InstanceType),
        # Precharging
        ("deficit_block_min_storage", float64),
        ("deficit_block_max_storage", float64),
        ("stored_energy_temp_reverse", float64),
        ("stored_energy_temp_forward", float64),
        ("precharge_energy", float64),
        ("trickling_reserves", float64),
        ("remaining_trickling_reserves", float64),
        ("precharge_flag", boolean),
        ("trickling_flag", boolean),
        ("remaining_discharge_max_t", float64),
        ("remaining_charge_max_t", float64),
    ]
else:
    storage_spec = []


@jitclass(storage_spec)
class Storage:
    """
    Represents an energy storage system unit in the system.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: int64,
        order: int64,
        name: unicode_type,
        power_capacity: float64,
        energy_capacity: float64,
        duration: float64,
        charge_efficiency: float64,
        discharge_efficiency: float64,
        max_build_p: float64,
        max_build_e: float64,
        min_build_p: float64,
        min_build_e: float64,
        unit_type: unicode_type,
        near_optimum_check: boolean,
        node: Node_InstanceType,
        line: Line_InstanceType,
        group: unicode_type,
        cost: UnitCost_InstanceType,
    ) -> None:
        """
        Initialize a Storage object.

        Parameters:
        -------
        id (int): Unique identifier for the storage unit.
        storage_dict (Dict[str, str]): Dictionary containing storage attributes.
        line (Line): The generic minor line defined to connect the generator to the transmission network.
                        Minor lines should have empty node_start and node_end values. They do not form part
                        of the network topology, but are used to estimate connection costs.
        """

        self.static_instance = static_instance
        self.id = idx
        self.order = order  # id specific to scenario
        self.name = name
        self.initial_power_capacity = power_capacity  # GW
        self.duration = duration  # hours
        self.initial_energy_capacity = energy_capacity if duration == 0 else duration * power_capacity  # GWh
        self.charge_efficiency = charge_efficiency  # %
        self.discharge_efficiency = discharge_efficiency  # %
        self.max_build_p = max_build_p  # GW/year
        self.max_build_e = max_build_e  # GWh/year
        self.min_build_p = min_build_p  # GW/year
        self.min_build_e = min_build_e  # GWh/year
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.node = node
        self.line = line
        self.group = group
        self.cost = cost

        self.candidate_p_x_idx = -1
        self.candidate_e_x_idx = -1

        # Dynamic
        self.new_build_p = 0.0  # GW
        self.new_build_e = 0.0  # GWh
        self.power_capacity = power_capacity  # GW
        self.energy_capacity = energy_capacity if duration == 0 else duration * power_capacity  # GWh
        self.dispatch_power = np.empty(0, dtype=np.float64)  # GW
        self.stored_energy = np.empty(0, dtype=np.float64)  # GWh

        self.discharge_max_t = 0.0  # GW
        self.charge_max_t = 0.0  # GW
        self.lt_discharge = 0.0  # GWh/year

        self.lt_costs = LTCosts()

        # Precharging
        self.stored_energy_temp_reverse = 0.0  # GWh
        self.stored_energy_temp_forward = 0.0  # GWh
        self.deficit_block_min_storage = 0.0  # GWh
        self.deficit_block_max_storage = 0.0  # GW  h
        self.precharge_energy = 0.0  # GWh
        self.trickling_reserves = 0.0  # GWh
        self.remaining_trickling_reserves = 0.0  # GWh
        self.precharge_flag = False  # Determines whether storage system can precharge
        self.trickling_flag = False  # Determines whether storage system can trickle-charge other storages

        self.remaining_discharge_max_t = 0.0  # GW
        self.remaining_charge_max_t = 0.0  # GW


if JIT_ENABLED:
    Storage_InstanceType = Storage.class_type.instance_type
else:
    Storage_InstanceType = Storage

if JIT_ENABLED:
    fleet_spec = [
        ("static_instance", boolean),
        ("generators", DictType(int64, Generator_InstanceType)),
        ("storages", DictType(int64, Storage_InstanceType)),
    ]
else:
    fleet_spec = []


@jitclass(fleet_spec)
class Fleet:
    def __init__(
        self,
        static_instance: boolean,
        generators: DictType(int64, Generator_InstanceType),
        storages: DictType(int64, Storage_InstanceType),
    ):
        self.static_instance = static_instance
        self.generators = generators
        self.storages = storages


if JIT_ENABLED:
    Fleet_InstanceType = Fleet.class_type.instance_type
else:
    Fleet_InstanceType = Fleet
