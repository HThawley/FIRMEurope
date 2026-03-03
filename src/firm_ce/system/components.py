# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import DictType, boolean, float64, int64, unicode_type
from firm_ce.system.costs import LTCosts, LTCosts_InstanceType, UnitCost_InstanceType
from firm_ce.system.topology import Line_InstanceType, Node_InstanceType

if JIT_ENABLED:
    fuel_spec = [
        ("object_class", unicode_type),
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
        self,
        static_instance: boolean,
        idx: int64,
        name: unicode_type,
        cost: float64,
        emissions: float64,
    ) -> None:
        """
        Initialize a Fuel object.

        Parameters:
        -------
        id (int): Unique identifier for the fuel.
        fuel_dict (Dict[str, str]): Dictionary containing 'name', 'cost', and 'emissions' keys.
        """
        self.object_class = "fuel"
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
        ("object_class", unicode_type),
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
        ("data_status", boolean),
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

    Solar, wind and baseload generators require generation trace data files. Flexible
    generators require data files for annual generation limits. Datafiles must be stored in
    the `inputs/data` folder and referenced in `inputs/config/datafiles.csv`.

    Notes:
    -----
    - Instances can be flagged as *static* or *dynamic* via static_instance. Static instances must not be
    modified inside worker processes used for the stochastic optimisation, whereas dynamic instances are
    safe to modify.
    - Memory for endogenous time-series dispatch and remaining energy arrays (flexible Generators) is allocated
    within worker processes for the optimisation.
    - Exogenous time-series data traces and annual constraint data is loaded prior to starting an optimisation.
    - Precharging fields are used in storage precharging period/deficit block steps.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit committment process.
    id (int64): A model-level identifier for the Generator instance.
    order (int64): A scenario-level identifier for the Generator instance.
    name (unicode_type): A string providing the oridinary name of the Generator.
    unit_size (float64): Nameplate unit size in GW. A Generator could be formed from multiple units.
    max_build (float64): Maximum build limit in GW.
    min_build (float64): Minimum build limit in GW.
    initial_capacity (float64): Installed capacity at model start in GW.
    unit_type (unicode_type): Type of Generator (e.g., 'solar', 'wind', 'baseload', 'flexible').
    near_optimum_check (boolean): Flag to perform near-optimum optimisation.
    node (Node_InstanceType): The Network Node where the Generator is located.
    fuel (Fuel_InstanceType): The Fuel consumed by the Generator.
    line (Line_InstanceType): Minor line connecting Generator to the transmission network.
    group (unicode_type): Group label used by broad optimum optimisation. Grouped assets are considered in aggregate
        when minimising/maximising installed capacity within the broad optimum space.
    cost (UnitCost_InstanceType): Exogenously defined cost assumptions.
    data_status (boolean): Status of data loading.
    data (float64[:]): Interval capacity factor trace data. Each value represents the capacity factor of the solar, wind
        or baseload Generator in each time interval of the modelling horizon.
    annual_constraints_data (float64[:]): Annual generation constraints for flexible Generators, units GWh/year.
    candidate_x_idx (int64): Index of the Generator's decision variable (new build capacity) in the candidate solution vector.
    new_build (float64): Capacity built for the candidate solution, units GW.
    capacity (float64): Current installed capacity, units GW.
    dispatch_power (float64[:]): Interval dispatch power of a flexible Generator, units GW.
    remaining_energy (float64[:]): Remaining annual energy for flexible Generators, units GWh.
    flexible_max_t (float64): Maximum dispatchable power in the current interval for a flexible Generator, units GW.
    lt_generation (float64): Long-term total generation over the entire modelling horizon, units GWh.
    unit_lt_hours (float64): Total hours of operation per unit, units hours.
    lt_costs (LTCosts_InstanceType): Endogenously calculated long-term costs of the Generator over the modelling horizon.
    remaining_energy_temp_reverse (float64): Temporary value for remaining energy when balancing deficit block in reverse time,
        units GWh.
    remaining_energy_temp_forward (float64): Temporary value for remaining energy when balancing deficit block in forward time,
        units GWh.
    deficit_block_max_energy (float64): Maximum value of remaining energy within a deficit block, units GWh.
    deficit_block_min_energy (float64): Minimum value of remaining energy within a deficit block, units GWh.
    trickling_flag (boolean): Flag indicating if flexible Generator is a trickle-charger and can precharge Storage systems.
    trickling_reserves (float64): Energy that must be retained during precharging so that flexible Generator can dispatch
        during deficit block, units GWh.
    remaining_trickling_reserves (float64): Energy remaining for trickle charging in the precharging period, units GWh.
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
        self.object_class = "generator"
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

        self.data_status = False
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
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("id", int64),
        ("order", int64),
        ("name", unicode_type),
        ("node", Node_InstanceType),
        ("initial_power_capacity", float64),
        ("initial_energy_capacity", float64),
        ("duration", float64),
        ("chargeable", boolean),
        ("inflows", boolean),
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
        ("data_status", boolean),
        ("data", float64[:]),
        # Dynamic
        ("new_build_p", float64),
        ("new_build_e", float64),
        ("power_capacity", float64),
        ("energy_capacity", float64),
        ("dispatch_power", float64[:]),
        ("stored_energy", float64[:]),
        ("discharge_max_t", float64),
        ("charge_max_t", float64),
        ("lt_generation", float64),
        ("unit_lt_hours", float64),
        ("lt_costs", LTCosts_InstanceType),
        # Precharging & Reserves
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
    def __init__(
        self,
        static_instance: boolean,
        idx: int64,
        order: int64,
        name: unicode_type,
        power_capacity: float64,
        energy_capacity: float64,
        duration: float64,
        chargeable: boolean,
        inflows: boolean,
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
        self.object_class = "storage"
        self.static_instance = static_instance
        self.id = idx
        self.order = order  # id specific to scenario
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
        self.near_optimum_check = near_optimum_check
        self.node = node
        self.line = line
        self.group = group
        self.cost = cost

        self.candidate_p_x_idx = -1
        self.candidate_e_x_idx = -1
        if self.inflows:
            self.data_status = False
            self.data = np.empty((0,), dtype=np.float64)

        # Dynamic
        self.new_build_p = 0.0  # GW
        self.new_build_e = 0.0  # GWh
        self.power_capacity = power_capacity  # GW
        self.energy_capacity = self.initial_energy_capacity  # GWh
        self.dispatch_power = np.empty((0,), dtype=np.float64)  # GW
        self.stored_energy = np.empty((0,), dtype=np.float64)  # GWh

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
        ("object_class", unicode_type),
        ("static_instance", boolean),
        ("generators", DictType(int64, Generator_InstanceType)),
        ("storages", DictType(int64, Storage_InstanceType)),
    ]
else:
    fleet_spec = []


@jitclass(fleet_spec)
class Fleet:
    """
    Represents a collection of Generators and Storage systems in the scenario.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit commitment process.
    generators (DictType(int64, Generator_InstanceType)): Typed dictionary of Generator instances keyed by their
        scenario-level orders.
    storages (DictType(int64, Storage_InstanceType)): Typed dictionary of Storage instances keyed by their scenario-level orders.
    """

    def __init__(
        self,
        static_instance: boolean,
        generators: DictType(int64, Generator_InstanceType),
        storages: DictType(int64, Storage_InstanceType),
    ):
        """
        Parameters:
        -------
        static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
            A static instance is unsafe to modify within a worker process for the unit commitment process.
        generators (DictType(int64, Generator_InstanceType)): Typed dictionary of Generator instances keyed by their
            scenario-level orders.
        storages (DictType(int64, Storage_InstanceType)): Typed dictionary of Storage instances keyed by their
            scenario-level orders.
        """
        self.object_class = "fleet"
        self.static_instance = static_instance
        self.generators = generators
        self.storages = storages


if JIT_ENABLED:
    Fleet_InstanceType = Fleet.class_type.instance_type
else:
    Fleet_InstanceType = Fleet
