# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import DictType, boolean, nbfloat, npfloat, nbintp, unicode_type
from firm_ce.system.scalar.costs import LTCosts, LTCosts_InstanceType, UnitCost_InstanceType
from firm_ce.system.scalar.topology import Line_InstanceType, Node_InstanceType

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
    Represents a fuel type with associated cost and emissions.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit committment process.
    id (nbintp): A model-level identifier for the Fuel instance.
    name (unicode_type): A string providing the oridinary name of the Fuel.
    cost (nbfloat): Cost in currency/MWh (thermal)
    emissions (nbfloat): Carbon emissions in tCO2-e / MWh (thermal)

    annual_limit (nbfloat[:]): Annual generation constraints in GWh/year.
    remaining_energy (nbfloat[:]): Amount of energy left in the annual limit

    remaining_energy_temp_reverse (nbfloat): Temporary value for remaining energy when balancing deficit block in reverse time,
        units GWh.
    remaining_energy_temp_forward (nbfloat): Temporary value for remaining energy when balancing deficit block in forward time,
        units GWh.
    deficit_block_max_energy (nbfloat): Maximum value of remaining energy within a deficit block, units GWh.
    deficit_block_min_energy (nbfloat): Minimum value of remaining energy within a deficit block, units GWh.
    trickling_flag (boolean): Flag indicating if flexible Generator is a trickle-charger and can precharge Storage systems.
    trickling_reserves (nbfloat): Energy that must be retained during precharging so that flexible Generator can dispatch
        during deficit block, units GWh.
    remaining_trickling_reserves (nbfloat): Energy remaining for trickle charging in the precharging period, units GWh.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: nbintp,
        name: unicode_type,
        cost: nbfloat,
        emissions: nbfloat,
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
        self.annual_limit = np.zeros((0,), dtype=npfloat)  # GWh/year
        self.remaining_energy = np.zeros((0,), dtype=npfloat)  # GWh

        self.data_status = False
        # Precharging
        self.remaining_energy_temp_reverse = 0.0  # GWh
        self.remaining_energy_temp_forward = 0.0  # GWh
        self.deficit_block_max_energy = 0.0  # GWh
        self.deficit_block_min_energy = 0.0  # GWh
        self.trickling_flag = False  # Determines whether flexible generator can precharge storage systems
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
        ("near_optimum_check", boolean),
        ("group", unicode_type),
        ("cost", UnitCost_InstanceType),
        ("data_status", boolean),
        ("data", nbfloat[:]),
        ("candidate_x_idx", nbintp),
        ("relative_scaler", nbfloat),
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
    id (nbintp): A model-level identifier for the Generator instance.
    order (nbintp): A scenario-level identifier for the Generator instance.
    name (unicode_type): A string providing the oridinary name of the Generator.
    unit_size (nbfloat): Nameplate unit size in GW. A Generator could be formed from multiple units.
    max_build (nbfloat): Maximum build limit in GW.
    min_build (nbfloat): Minimum build limit in GW.
    initial_capacity (nbfloat): Installed capacity at model start in GW.
    unit_type (unicode_type): Type of Generator (e.g., 'solar', 'wind', 'baseload', 'flexible').
    is_flexible (unicode_type): Whether Generator is flexible (e.g. ccgt).
    near_optimum_check (boolean): Flag to perform near-optimum optimisation.
    node (Node_InstanceType): The Network Node where the Generator is located.
    fuel (Fuel_InstanceType): The Fuel consumed by the Generator.
    line (Line_InstanceType): Minor line connecting Generator to the transmission network.
    group (unicode_type): Group label used by broad optimum optimisation. Grouped assets are considered in aggregate
        when minimising/maximising installed capacity within the broad optimum space.
    cost (UnitCost_InstanceType): Exogenously defined cost assumptions.
    data_status (boolean): Status of data loading.
    data (nbfloat[:]): Interval capacity factor trace data. Each value represents the capacity factor of the solar, wind
        or baseload Generator in each time interval of the modelling horizon.
    candidate_x_idx (nbintp): Index of the Generator's decision variable (new build capacity) in the candidate solution vector.
    new_build (nbfloat): Capacity built for the candidate solution, units GW.
    capacity (nbfloat): Current installed capacity, units GW.
    dispatch_power (nbfloat[:]): Interval dispatch power of a flexible Generator, units GW.
    remaining_energy (nbfloat[:]): Remaining annual energy for flexible Generators, units GWh.
    flexible_max_t (nbfloat): Maximum dispatchable power in the current interval for a flexible Generator, units GW.
    lt_generation (nbfloat): Long-term total generation over the entire modelling horizon, units GWh.
    unit_lt_hours (nbfloat): Total hours of operation per unit, units hours.
    lt_costs (LTCosts_InstanceType): Endogenously calculated long-term costs of the Generator over the modelling horizon.

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
        self.is_flexible = is_flexible
        self.near_optimum_check = near_optimum_check
        self.node = node
        self.fuel = fuel
        self.line = line
        self.group = group
        self.cost = cost

        self.data_status = False
        self.data = np.empty((0,), dtype=npfloat)

        self.candidate_x_idx = -1
        self.relative_scaler = 1.0

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
        ("near_optimum_check", boolean),
        ("group", unicode_type),
        ("cost", UnitCost_InstanceType),
        ("candidate_p_x_idx", nbintp),
        ("candidate_e_x_idx", nbintp),
        ("data_status", boolean),
        ("data", nbfloat[:]),
        ("relative_scaler_p", nbfloat),
        ("relative_energy", boolean),
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
        self.relative_scaler_p = 1.0
        self.relative_energy = False
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
        ("generators", DictType(nbintp, Generator_InstanceType)),
        ("storages", DictType(nbintp, Storage_InstanceType)),
        ("fuels", DictType(nbintp, Fuel_InstanceType)),
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
    generators (DictType(nbintp, Generator_InstanceType)): Typed dictionary of Generator instances keyed by their
        scenario-level orders.
    storages (DictType(nbintp, Storage_InstanceType)): Typed dictionary of Storage instances keyed by their scenario-level orders.
    fuels (DictType(nbintp, Fuel_InstanceType)): Typed dictionary of Fuel instances keyed by their scenario-level orders.
    """

    def __init__(
        self,
        static_instance: boolean,
        generators: DictType(nbintp, Generator_InstanceType),
        storages: DictType(nbintp, Storage_InstanceType),
        fuels: DictType(nbintp, Fuel_InstanceType),
    ):
        """
        Parameters:
        -------
        static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
            A static instance is unsafe to modify within a worker process for the unit commitment process.
        generators (DictType(nbintp, Generator_InstanceType)): Typed dictionary of Generator instances keyed by their
            scenario-level orders.
        storages (DictType(nbintp, Storage_InstanceType)): Typed dictionary of Storage instances keyed by their
            scenario-level orders.
        fuels (DictType(nbintp, Fuels_InstanceType)): Typed dictionary of Fuel instances keyed by their
            scenario-level orders.
        """
        self.object_class = "fleet"
        self.static_instance = static_instance
        self.generators = generators
        self.storages = storages
        self.fuels = fuels


if JIT_ENABLED:
    Fleet_InstanceType = Fleet.class_type.instance_type
else:
    Fleet_InstanceType = Fleet
