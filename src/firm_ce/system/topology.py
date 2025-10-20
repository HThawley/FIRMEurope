# type: ignore
from typing import List, Tuple

import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import DictType, ListType, UniTuple, boolean, float64, int64, unicode_type
from firm_ce.system.costs import LTCosts, LTCosts_InstanceType, UnitCost_InstanceType

if JIT_ENABLED:
    node_spec = [
        ("static_instance", boolean),
        ("id", int64),
        ("order", int64),
        ("name", unicode_type),
        ("data_status", unicode_type),
        ("data", float64[:]),
        ("residual_load", float64[:]),
        # Dynamic
        ("storage_merit_order", int64[:]),
        ("reservoir_merit_order", int64[:]),
        ("flexible_merit_order", int64[:]),
        ("netload_t", float64),
        ("discharge_max_t", float64[:]),
        ("charge_max_t", float64[:]),
        ("reservoir_max_t", float64[:]),
        ("flexible_max_t", float64[:]),
        ("fill", float64),
        ("surplus", float64),
        ("temp_surplus", float64),
        ("available_imports", float64),
        ("imports_exports", float64[:]),
        ("deficits", float64[:]),
        ("spillage", float64[:]),
        ("flexible_power", float64[:]),
        ("reservoir_power", float64[:]),
        ("storage_power", float64[:]),
        # Precharging
        ("imports_exports_temp", float64),
        ("imports_exports_update", float64),
        ("existing_surplus", float64),
        ("precharge_fill", float64),
        ("precharge_surplus", float64),
    ]
else:
    node_spec = []


@jitclass(node_spec)
class Node:
    """
    Represents a node (bus) in the network.

    Notes:
    -----
    - Instances can be flagged as *static* or *dynamic* via static_instance. Static instances must not be
    modified inside worker processes used for the stochastic optimisation, whereas dynamic instances are
    safe to modify.
    - Exogenous time-series operational demand data trace is loaded prior to starting an optimisation.
    - Memory for endogenous dispatch/state is allocated within worker processes for the optimisation.
    - Precharging fields are used in storage precharging period/deficit block steps.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit commitment process.
    id (int64): A model-level identifier for the Node instance.
    order (int64): A scenario-level identifier for the Node instance.
    name (unicode_type): A string providing the ordinary name of the Node.
    data_status (unicode_type): Status of data loading (e.g., 'unloaded', 'loaded').
    data (float64[:]): Operational demand time-series for the node, units GW.
    residual_load (float64[:]): Equal to the operational demand minus generation from solar, wind, and baseload generation in a
        given time interval, units GW.
    storage_merit_order (int64[:]): Merit-order of Storage assets connected to this node (array of Storage.order values). Ordered
        from shortest to longest storage duration.
    storage_merit_order (int64[:]): Merit-order of Reservoir assets connected to this node (array of Storage.order values). Ordered
        from shortest to longest storage duration.
    flexible_merit_order (int64[:]): Merit-order of flexible Generators connected to this node (array of Generator.order values).
        Order from lowest to highest marginal variable cost.
    netload_t (float64): Current net load for the interval based upon completed balancing actions, units GW. Equal to the sum of
        residual load and nodal imports/exports. When balancing a deficit block or precharging period, the nodal storage power
        and flexible power is also included in the sum to calculate net load.
    discharge_max_t (float64[:]): A 1-dimensional array defining the cumulative maximum discharge limits for the time interval
        across the storage merit order, units GW.
    charge_max_t (float64[:]): A 1-dimensional array defining the cumulative maximum charge limits for the time interval
        across the storage merit order, units GW.
    reservoir_max_t (float64[:]): A 1-dimensional array defining the cumulative maximum discharge limits for the time interval
        across the reservoir merit order, units GW.
    flexible_max_t (float64[:]): A 1-dimensional array defining the cumulative maximum generation limits for the time interval
        across the flexible merit order, units GW.
    fill (float64): Current energy that the node is attempting to fill through transmission actions, units GW.
    surplus (float64): Current surplus energy that the node has available for transmission actions, units GW.
    temp_surplus (float64): Uncommitted surplus energy remaining for the current transmission leg, units GW.
    available_imports (float64): Upper bound on imports available from surplus nodes on the current transmission leg, units GW.
    imports_exports (float64[:]): Endogenous time-series defining interval imports (+) and exports (-) at the node, units GW.
    deficits (float64[:]): Endogenous time-series defining interval power deficits after all balancing at the node, units GW.
    spillage (float64[:]): Endogenous time-series defining interval spillage/curtailment at the node, units GW.
    flexible_power (float64[:]): Endogenous time-series defining interval net dispatch of flexible Generators connected to
        this node, units GW.
    reservoir_power (float64[:]): Endogenous time-series defining interval net dispatch of Reservoirs connected to
        this node, units GW.
    storage_power (float64[:]): Endogenous time-series defining interval net storage power (discharge +, charge -) of Storages
        connected to this node, units GW.
    imports_exports_temp (float64): Imports/exports temporarily saved prior to a transmission action during precharging. Used
        to determine dispatch adjustments after the transmission action, units GW.
    imports_exports_update (float64): The difference between the original imports/exports (stored as a temporary value) and the
        adjusted imports/exports during a precharging action, units GW. Used to adjust dispatch power for precharging.
    existing_surplus (float64): Surplus that could be available for precharging Storage systems at the start of a time interval
        in the precharging period, units GW.
    precharge_fill (float64): Net energy that Storage prechargers at the node are attempting to fill for a precharging action,
        units GW.
    precharge_surplus (float64): Existing surplus or net energy available from Storage and flexible Generator trickle-chargers
        at the node for a precharging action, units GW.
    """

    def __init__(self, static_instance: boolean, idx: int64, order: int64, name: unicode_type) -> None:
        """
        Initialise a Node instance.

        Parameters:
        -------
        static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
            A static instance is unsafe to modify within a worker process for the unit commitment process.
        idx (int64): A model-level identifier for the Node instance.
        order (int64): A scenario-level identifier for the Node instance.
        name (unicode_type): A string providing the ordinary name of the Node.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order  # id specific to scenario
        self.name = name
        self.data_status = "unloaded"
        self.data = np.empty((0,), dtype=np.float64)

        self.residual_load = np.empty((0,), dtype=np.float64)

        # Dynamic
        self.flexible_merit_order = np.empty((0,), dtype=np.int64)
        self.reservoir_merit_order = np.empty((0,), dtype=np.int64)
        self.storage_merit_order = np.empty((0,), dtype=np.int64)
        self.netload_t = 0.0  # GW
        self.discharge_max_t = np.empty((0,), dtype=np.float64)  # GW
        self.charge_max_t = np.empty((0,), dtype=np.float64)  # GW
        self.reservoir_max_t = np.empty((0,), dtype=np.float64)  # GW
        self.flexible_max_t = np.empty((0,), dtype=np.float64)  # GW

        self.fill = 0.0  # GW, power attempting to import
        self.surplus = 0.0  # GW, power available for exports
        self.temp_surplus = 0.0  # GW, power remaining for exports in the current transmission leg
        self.available_imports = 0.0  # GW, maximum power that could be imported from other node surpluses

        self.imports_exports = np.empty((0,), dtype=np.float64)
        self.deficits = np.empty((0,), dtype=np.float64)
        self.spillage = np.empty((0,), dtype=np.float64)

        self.flexible_power = np.empty((0,), dtype=np.float64)
        self.reservoir_power = np.empty((0,), dtype=np.float64)
        self.storage_power = np.empty((0,), dtype=np.float64)

        # Precharging
        self.imports_exports_temp = 0.0  # GW, Existing imports/exports at start of precharging action
        self.imports_exports_update = 0.0  # GW, Update to imports/exports during precharging
        self.existing_surplus = 0.0  # GW
        self.precharge_fill = 0.0  # GW
        self.precharge_surplus = 0.0  # GW


if JIT_ENABLED:
    Node_InstanceType = Node.class_type.instance_type
else:
    Node_InstanceType = Node

if JIT_ENABLED:
    line_spec = [
        ("static_instance", boolean),
        ("id", int64),
        ("order", int64),
        ("name", unicode_type),
        ("length", float64),
        ("node_start", Node_InstanceType),
        ("node_end", Node_InstanceType),
        ("loss_factor", float64),
        ("max_build", float64),
        ("min_build", float64),
        ("initial_capacity", float64),
        ("unit_type", unicode_type),
        ("near_optimum_check", boolean),
        ("group", unicode_type),
        ("cost", UnitCost_InstanceType),
        ("candidate_x_idx", int64),
        # Dynamic
        ("new_build", float64),
        ("capacity", float64),
        ("flows", float64[:]),
        ("temp_leg_flows", float64),
        ("lt_flows", float64),
        ("lt_costs", LTCosts_InstanceType),
    ]
else:
    line_spec = []


@jitclass(line_spec)
class Line:
    """
    Transmission line connecting two Nodes or a minor line connecting an asset to the network.

    Notes:
    -----
    - Instances can be flagged as *static* or *dynamic* via static_instance. Static instances must not be
    modified inside worker processes used for the stochastic optimisation, whereas dynamic instances are
    safe to modify.
    - Major lines connect two Nodes (node_start to node_end) and form Routes along the transmission network.
    - Minor lines connect a Generator or Storage asset to its Node. They do not form part of the transmission
    topology.
    - Line losses are modelled via a simple linear function according to line length and a loss factor.
    - Memory for endogenous flow time-series is allocated within worker processes for the optimisation.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit commitment process.
    id (int64): A model-level identifier for the Line instance.
    order (int64): A scenario-level identifier for the Line instance.
    name (unicode_type): A string providing the oridinary name of the transmission line.
    length (float64): Line length, units km.
    node_start (Node_InstanceType): Starting Node for a major line (generic minor Node for minor Line).
    node_end (Node_InstanceType): Ending Node for a major line (generic minor Node for minor Line).
    loss_factor (float64): Energy loss percentage per 1000 km.
    max_build (float64): Maximum build limit in GW.
    min_build (float64): Minimum build limit in GW.
    initial_capacity (float64): Installed capacity at model start in GW.
    unit_type (unicode_type): Technology/type label (e.g., 'AC', 'HVDC', 'Minor'). Currently, only relevent for grouping
        similar Lines together in some figures within `tools/result_viewer.ipynb`.
    near_optimum_check (boolean): Flag to perform near-optimum optimisation.
    group (unicode_type): Group label used by broad optimum optimisation. Grouped assets are considered in aggregate
        when minimising/maximising installed capacity within the broad optimum space.
    cost (UnitCost_InstanceType): Exogenously defined cost assumptions.
    candidate_x_idx (int64): Index of the Line's decision variable (new build capacity) in the candidate solution vector.
    new_build (float64): Capacity built for the candidate solution, units GW.
    capacity (float64): Current installed capacity, units GW.
    flows (float64[:]): Interval line flows, units GW. Signed so that (+) flows are from node_start to node_end, (-) flows
        are from node_end to node_start.
    temp_leg_flows (float64): Flow reserved for the current transmission leg, units GW.
    lt_flows (float64): Long-term total energy transferred (in either direction) over the modelling horizon, units GWh.
    lt_costs (LTCosts_InstanceType): Endogenously calculated long-term costs for the line over the modelling horizon.
    """

    def __init__(
        self,
        static_instance: boolean,
        idx: int64,
        order: int64,
        name: unicode_type,
        length: float64,
        node_start: Node_InstanceType,
        node_end: Node_InstanceType,
        loss_factor: float64,
        max_build: float64,
        min_build: float64,
        capacity: float64,
        unit_type: unicode_type,
        near_optimum_check: boolean,
        group: unicode_type,
        cost: UnitCost_InstanceType,
    ) -> None:
        """
        Initialise a Line instance.

        Parameters:
        -------
        static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
            A static instance is unsafe to modify within a worker process for the unit commitment process.
        idx (int64): A model-level identifier for the Line instance.
        order (int64): A scenario-level identifier for the Line instance.
        name (unicode_type): A string providing the oridinary name of the transmission line.
        length (float64): Line length, units km.
        node_start (Node_InstanceType): Starting Node for a major line (generic minor Node for minor Line).
        node_end (Node_InstanceType): Ending Node for a major line (generic minor Node for minor Line).
        loss_factor (float64): Energy loss percentage per 1000 km.
        max_build (float64): Maximum build limit in GW.
        min_build (float64): Minimum build limit in GW.
        capacity (float64): Installed capacity at model start in GW.
        unit_type (unicode_type): Technology/type label (e.g., 'AC', 'HVDC', 'Minor'). Currently, only relevent for grouping
        similar Lines together in some figures within `tools/result_viewer.ipynb`.
        near_optimum_check (boolean): Flag to perform near-optimum optimisation.
        group (unicode_type): Group label used by broad optimum optimisation. Grouped assets are considered in aggregate
        when minimising/maximising installed capacity within the broad optimum space.
        cost (UnitCost_InstanceType): Exogenously defined cost assumptions.
        """
        self.static_instance = static_instance
        self.id = idx
        self.order = order  # id specific to scenario
        self.name = name
        self.length = length  # km
        self.node_start = node_start  # Starting node
        self.node_end = node_end  # Ending node
        self.loss_factor = loss_factor  # Transmission losses % per 1000 km
        self.max_build = max_build  # GW/year
        self.min_build = min_build  # GW/year
        self.initial_capacity = capacity  # GW
        self.unit_type = unit_type
        self.near_optimum_check = near_optimum_check
        self.group = group
        self.cost = cost

        self.candidate_x_idx = -1

        # Dynamic
        self.new_build = 0.0  # GW
        self.capacity = capacity  # GW
        self.flows = np.empty(0, dtype=np.float64)  # GW, total line flows
        self.temp_leg_flows = 0.0  # GW, line flows reserved for a route on the current leg
        self.lt_flows = 0.0  # GWh

        self.lt_costs = LTCosts()


if JIT_ENABLED:
    Line_InstanceType = Line.class_type.instance_type
else:
    Line_InstanceType = Line

if JIT_ENABLED:
    route_spec = [
        ("static_instance", boolean),
        ("initial_node", Node_InstanceType),
        ("nodes", ListType(Node_InstanceType)),
        ("lines", ListType(Line_InstanceType)),
        ("line_directions", int64[:]),
        ("legs", int64),
        ("flow_update", float64),
    ]
else:
    route_spec = []


@jitclass(route_spec)
class Route:
    """
    An ordered path of a specified length (legs) that steps through the transmission network.

    Notes:
    -------
    - Instances can be flagged as *static* or *dynamic* via static_instance. Static instances must not be
    modified inside worker processes used for the stochastic optimisation, whereas dynamic instances are
    safe to modify.
    - A Route is defined by an initial node, an ordered list of nodes and lines, and the direction on each line.
    The initial node is not included in the list of nodes and is generally treated as the 'fill' node during
    transmission. The final node in the nodes list is treated as the 'surplus' node during transmission.
    - The number of nodes/lines forming the route is equal to the legs.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit commitment process.
    initial_node (Node_InstanceType): Node where the route originates. Acts as the 'fill' Node for transmission.
    nodes (ListType(Node_InstanceType)): Ordered list of Nodes traversed by the Route. Final Node in the list is the
        'surplus' Node for transmission.
    lines (ListType(Line_InstanceType)): Ordered list of Lines corresponding to each leg of the Route.
    line_directions (int64[:]): Line direction per leg (+1 for node_start to node_end, -1 for reverse).
    legs (int64): Number of legs (Line/Node traversals) in the Route. Maximum value defined by networksteps_max in the
        `scenarios.csv` config file.
    flow_update (float64): Current flow allocated along this Route during a transmission action, units GW.
    """

    def __init__(
        self,
        static_instance: boolean,
        initial_node: Node_InstanceType,
        nodes_typed_list: ListType(Node_InstanceType),
        lines_typed_list: ListType(Line_InstanceType),
        line_directions: int64[:],
        legs: int64,
    ):
        """
        Initialise a Route instance.

        Parameters:
        -------
        static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
            A static instance is unsafe to modify within a worker process for the unit commitment process.
        initial_node (Node_InstanceType): Node where the route originates. Acts as the 'fill' Node for transmission.
        nodes_typed_list (ListType(Node_InstanceType)): Ordered list of Nodes traversed by the Route. Final Node in the list is the
            'surplus' Node for transmission.
        lines_typed_list (ListType(Line_InstanceType)): Ordered list of Lines corresponding to each leg of the Route.
        line_directions (int64[:]): Line direction per leg (+1 for node_start to node_end, -1 for reverse).
        legs (int64): Number of legs (Line/Node traversals) in the Route. Maximum value defined by networksteps_max in the
            `scenarios.csv` config file.
        """
        self.static_instance = static_instance
        self.initial_node = initial_node
        self.nodes = nodes_typed_list
        self.lines = lines_typed_list
        self.line_directions = line_directions
        self.legs = legs

        # Dynamic
        self.flow_update = 0.0


if JIT_ENABLED:
    Route_InstanceType = Route.class_type.instance_type
    routes_key_type = UniTuple(int64, 2)
    routes_list_type = ListType(Route_InstanceType)
else:
    Route_InstanceType = Route
    routes_key_type = Tuple[int, int]
    routes_list_type = List[Route_InstanceType]

if JIT_ENABLED:
    network_spec = [
        ("static_instance", boolean),
        ("nodes", DictType(int64, Node_InstanceType)),
        ("major_lines", DictType(int64, Line_InstanceType)),
        ("minor_lines", DictType(int64, Line_InstanceType)),
        ("networksteps_max", int64),
        ("routes", DictType(routes_key_type, routes_list_type)),  # Key is Tuple(initial_node.order, legs)
        ("major_line_count", int64),
    ]
else:
    network_spec = []


@jitclass(network_spec)
class Network:
    """
    Contains nodes, lines, and routes required to define the network topology and perform transmission.

    Notes:
    -------
    - Instances can be flagged as *static* or *dynamic* via static_instance. Static instances must not be
    modified inside worker processes used for the stochastic optimisation, whereas dynamic instances are
    safe to modify.

    Attributes:
    -------
    static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
        A static instance is unsafe to modify within a worker process for the unit commitment process.
    nodes (DictType(int64, Node_InstanceType)): Typed dictionary of Node instances keyed by their
        scenario-level orders.
    major_lines (DictType(int64, Line_InstanceType)): Typed dictionary of major Line instances keyed by their
        scenario-level orders. Major Lines are those that form connections between Nodes in the transmission
        network.
    minor_lines (DictType(int64, Line_InstanceType)): Typed dictionary of minor Line instances keyed by their
        scenario-level orders. Minor Lines are those that form connections between Generator or Storage systems
        and the transmission network.
    networksteps_max (int64): Maximum length of Routes used for transmission actions.
    routes (DictType(routes_key_type, routes_list_type)): Typed dictionary of Route lists keyed by (initial_node.order,
        legs). Each typed list value contains all Routes with the same initial node and the same length (legs).
    major_line_count (int64): Count of major lines in the Network.
    """

    def __init__(
        self,
        static_instance: boolean,
        nodes: DictType(int64, Node_InstanceType),
        major_lines: DictType(int64, Line_InstanceType),
        minor_lines: DictType(int64, Line_InstanceType),
        routes: DictType(routes_key_type, routes_list_type),
        networksteps_max: int64,
    ) -> None:
        """
        Initialise a Network instance.

        Parameters:
        -------
        static_instance (boolean): True value indicates 'static' instance, False indicates 'dynamic' instance.
            A static instance is unsafe to modify within a worker process for the unit commitment process.
        nodes (DictType(int64, Node_InstanceType)): Typed dictionary of Node instances keyed by their
            scenario-level orders.
        major_lines (DictType(int64, Line_InstanceType)): Typed dictionary of major Line instances keyed by their
            scenario-level orders. Major Lines are those that form connections between Nodes in the transmission
            network.
        minor_lines (DictType(int64, Line_InstanceType)): Typed dictionary of minor Line instances keyed by their
            scenario-level orders. Minor Lines are those that form connections between Generator or Storage systems
            and the transmission network.
        routes (DictType(routes_key_type, routes_list_type)): Typed dictionary of Route lists keyed by (initial_node.order,
            legs). Each typed list value contains all Routes with the same initial node and the same length (legs).
        networksteps_max (int64): Maximum length of Routes used for transmission actions.
        """
        self.static_instance = static_instance
        self.nodes = nodes
        self.major_lines = major_lines
        self.minor_lines = minor_lines
        self.routes = routes
        self.networksteps_max = networksteps_max
        self.major_line_count = len(major_lines)


if JIT_ENABLED:
    Network_InstanceType = Network.class_type.instance_type
else:
    Network_InstanceType = Network
