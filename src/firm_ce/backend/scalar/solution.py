# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit
from firm_ce.common.typing import boolean, nbfloat, unicode_type
from firm_ce.fast_methods import fleet_m, generator_m, line_m, network_m, static_m, storage_m, ltcosts_m
from firm_ce.backend.scalar.balancing import balance_for_period
from firm_ce.system.scalar.components import Fleet_InstanceType
from firm_ce.system.scalar.parameters import ScenarioParameters_InstanceType
from firm_ce.system.scalar.topology import Network_InstanceType


if JIT_ENABLED:
    from numba import set_num_threads

    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        ("x", nbfloat[:]),
        ("evaluated", boolean),
        ("annual_cost", nbfloat),
        ("lcoe", nbfloat),
        ("x_lcoe", nbfloat[:]),
        ("penalties", nbfloat),
        ("balancing_type", unicode_type),
        ("fixed_costs_threshold", nbfloat),
        # Static jitclass instances
        ("static", ScenarioParameters_InstanceType),
        # Dynamic jitclass instances
        ("fleet", Fleet_InstanceType),
        ("network", Network_InstanceType),
    ]
else:
    solution_spec = []


@jitclass(solution_spec)
class Solution:
    """
    Provides a complete description of the system associated with a candidate solution vector. The system can be
    evaluated according to the unit committment business rules.

    Notes:
    -------
    - The candidate solution vector 'x' defines the new-build capacities for Generator, Storage, and major Line
    objects in the system.
    - The Solution.static instance is unsafe to modify within the Solution class.
    - The Solution.fleet and Solution.network attributes are dynamic copies of the instances used to initialise
    an instance of this class. Most attributes of dynamic jitclass instances are safe to modify within an
    optimisation. Refer to the class definitions for specific jitclasses for information on which attributes
    remain unsafe to modify.
    - Reliability and fixed-cost constraints may terminate evaluation early, accumulating penalties. If the fixed cost
    threshold is set too low, it is very likely that the optimisation will
    get stuck in a local minimum (fixed costs just below threshold, reliability constraint still breached). This
    issue can be mitigated by increasing the mutation factor, raising the fixed cost threshold, or increasing the build
    limit of flexible Generator capacity.
    - The energy (cost) returned by the objective function is a system-level levelised cost of electricity (LCOE). This
    is calculated to be the sum of variable and fixed costs of all assets in the system divided by total operational
    demand over the modelling horizon, units $/MWh.

    Attributes:
    -------
    x (nbfloat[:]): Candidate solution decision variable vector.
    evaluated (boolean): Flag indicating whether `objective()` has been evaluated.
    lcoe (nbfloat): Levelised cost of electricity for the candidate, units $/MWh.
    penalties (nbfloat): Accumulated penalties for soft-constraint violations (fixed-costs and reliability), units $ or GW.
    balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over the entire
        time horizon at the specified resolution).
    fixed_costs_threshold (nbfloat): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit committment
        problem.
    static (ScenarioParameters_InstanceType): Static scenario parameters (unsafe to modify).
    fleet (Fleet_InstanceType): Dynamic copy of Fleet instance for this evaluation (safe to modify some attributes).
    network (Network_InstanceType): Dynamic copy of Network instance for this evaluation (safe to modify some attributes).
    """

    def __init__(
        self,
        x: nbfloat[:],
        static: ScenarioParameters_InstanceType,
        fleet: Fleet_InstanceType,
        network: Network_InstanceType,
        balancing_type: unicode_type,
        fixed_costs_threshold: nbfloat,
    ) -> None:
        """
        Initialise a Solution instance and construct dynamic copies of Fleet and Network.

        Parameters:
        -------
        x (nbfloat[:]): Candidate solution decision variable vector.
        static (ScenarioParameters_InstanceType): Static scenario parameters (unsafe to modify).
        fleet (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
        network (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
        balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over
            the entire time horizon at the specified resolution).
        fixed_costs_threshold (nbfloat): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
            low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit
            committment problem.

        Side-effects
        -------
        After creating the dynamic jitclass copies, they are modified to build new capacity, allocate memory for
        endogenously derived time-series data, and assign merit orders. A substantial number of attributes are
        modified in the dynamic instances. Refer to docstrings for the fast pseudo-methods called within this special
        method for details on these modifications.
        """
        self.x = x
        self.evaluated = False
        self.annual_cost = 0.0
        self.lcoe = 0.0
        self.x_lcoe = np.zeros(len(x), nbfloat)
        self.penalties = 0.0

        # These are static jitclass instances. It is UNSAFE to modify these
        # within a worker process of the optimiser
        self.static = static
        self.balancing_type = balancing_type
        self.fixed_costs_threshold = fixed_costs_threshold

        # These are dynamic jitclass instances. It is SAFE to modify
        # some attributes within a worker process of the optimiser
        self.network = network_m.create_dynamic_copy(network)  # Includes static reference to data
        self.fleet = fleet_m.create_dynamic_copy(
            fleet, self.network.nodes, self.network.minor_lines
        )  # Includes static reference to data

        fleet_m.build_capacities(self.fleet, x, self.static.resolution)
        network_m.build_capacity(self.network, x)

        fleet_m.allocate_memory(self.fleet, self.static.intervals_count)
        network_m.allocate_memory(self.network, self.static.intervals_count)

        network_m.assign_storage_merit_orders(self.network, self.fleet.storages)
        network_m.assign_flexible_merit_orders(self.network, self.fleet.generators)
        network_m.assign_route_merit_orders(self.network)


if JIT_ENABLED:
    Solution_InstanceType = Solution.class_type.instance_type
else:
    Solution_InstanceType = Solution


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def balance_residual_load(solution: Solution_InstanceType) -> boolean:
    """
    Evaluate the unit committment business rules over the entire modelling horizon.

    Notes:
    -------
    - At the end of each calendar year, the reliability constraint is evaluated. The method returns early
    if the reliability constraint is breached for any year.
    - Stored energy in Storage systems is initialised at the start of the modelling period.
    Annual generation limits for flexible Generators are initialised at the start of each calendar year.

    Parameters:
    -------
    None.

    Returns:
    -------
    boolean: Returns True if reliability constraint is satisfied for all years in the modelling horizon.
        Otherwise, False.

    Side-effects:
    -------
    Dynamic jitlass instances are substantially modified within this method. The stored energy of Storage systems
    and remaining energy for flexible Generators are initialised using Fleet pseudo-methods. The
    endogenous time-series data and temporary values are modified throughout the balance_for_period function.
    Attributes that are modified are marked using *Dynamic* or *Precharging* comments in the relevant jitclass
    definitions.
    """
    fleet_m.initialise_stored_energies(solution.fleet)
    for year in range(solution.static.year_count):
        first_t, last_t = static_m.get_year_t_boundaries(solution.static, year)
        fleet_m.initialise_annual_limits(solution.fleet, year, first_t)
        balance_for_period(first_t, last_t, solution.balancing_type == "full", solution, year)
        annual_unserved_energy = network_m.calculate_period_unserved_energy(
            solution.network, first_t, last_t, solution.static.resolution
        )

        # End early if reliability constraint breached for any year
        if not static_m.check_reliability_constraint(solution.static, year, annual_unserved_energy):
            solution.penalties += (solution.static.year_count - year) * annual_unserved_energy
            return False
    return True


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_fixed_costs(
    solution: Solution_InstanceType,
    include_existing: bool,
) -> None:
    """
    Calculate total fixed costs for all assets. Based upon the annualised build costs and fixed O&M costs
    incurred over the modelling horizon.

    Notes:
    -------
    - A years_float value is used to ensure leap days incur fixed O&M costs. This is consistent with the
    PLEXOS formulation.
    - Can be calculated before the unit committment evaluation, since these costs are independent of dispatch.

    Parameters:
    -------
    None.

    Returns:
    -------
    nbfloat: Total fixed costs over the modelling horizon, units $.

    Side-effects:
    -------
    Attributes modified for values in Solution.fleet.generators, Solution.fleet.storages,
        Solution.network.major_lines, Solution.network.minor_lines: lt_costs.
    Attributes modified for LTCosts instances referenced in the lt_costs attributes: fom, annualised_build.
    """
    # use a small accumulator to help mitigate float32 precision loss when using float32
    acc = 0.0
    for generator in solution.fleet.generators.values():
        acc += generator_m.calculate_fixed_costs(generator, include_existing)
    solution.annual_cost += acc

    acc = 0.0
    for storage in solution.fleet.storages.values():
        acc += storage_m.calculate_fixed_costs(storage, include_existing)
    solution.annual_cost += acc

    acc = 0.0
    for line in solution.network.major_lines.values():
        acc += line_m.calculate_fixed_costs(line, include_existing)
    solution.annual_cost += acc

    acc = 0.0
    for line in solution.network.minor_lines.values():
        acc += line_m.calculate_fixed_costs(line, include_existing)
    solution.annual_cost += acc

    return None


@njit(fastmath=FASTMATH)
def calculate_variable_costs(solution: Solution_InstanceType) -> None:
    """
    Calculate total variable costs based on dispatch and flows derived through unit committment
    business rules.

    Notes:
    -----
    - This method should not be called before complete evaluation of the unit committment business
    rules over the modelling horizon.

    Returns:
    -------
    nbfloat: Total variable costs over the modelling horizon, units $.

    Side-effects:
    -------
    Attributes modified for values in Solution.fleet.generators, Solution.fleet.storages,
        Solution.network.major_lines, Solution.network.minor_lines: lt_costs.
    Attributes modified for LTCosts instances referenced in the lt_costs attributes: vom, fuel.
    """
    fleet_m.calculate_lt_generations(
        solution.fleet,
        solution.static.resolution,
    )
    network_m.calculate_lt_flows(
        solution.network,
        solution.static.resolution,
    )

    # use a smaller accumulator to help mitigate float32 precision loss when using float32
    acc = 0.0
    for generator in solution.fleet.generators.values():
        acc += generator_m.calculate_variable_costs(generator, solution.static.year_float)
    solution.annual_cost += acc

    acc = 0.0
    for storage in solution.fleet.storages.values():
        acc += storage_m.calculate_variable_costs(storage, solution.static.year_float)
    solution.annual_cost += acc

    acc = 0.0
    for line in solution.network.major_lines.values():
        acc += line_m.calculate_variable_costs(line, solution.static.year_float)
    solution.annual_cost += acc

    acc = 0.0
    for line in solution.network.minor_lines.values():
        acc += line_m.calculate_variable_costs(line, solution.static.year_float)
    solution.annual_cost += acc

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def calculate_partial_costs(solution: Solution_InstanceType) -> None:
    for generator in solution.fleet.generators.values():
        solution.x_lcoe[generator.candidate_x_idx] = generator_m.get_partial_cost(generator, solution.static.year_float)

    for storage in solution.fleet.storages.values():
        solution.x_lcoe[storage.candidate_p_x_idx] = storage_m.get_partial_cost_power(storage, solution.static.year_float)
        solution.x_lcoe[storage.candidate_e_x_idx] = storage_m.get_partial_cost_energy(storage)

    for line in solution.network.major_lines.values():
        solution.x_lcoe[line.candidate_x_idx] = line_m.get_partial_cost(line, solution.static.year_float)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_fixed_costs(solution: Solution_InstanceType) -> boolean:
    """
    Check the fixed cost constraint against the configured threshold.

    Notes:
    -----
    - Fixed costs are evaluated relative to total operational demand. This provides consistency with the
    system-level LCOE, making it easier for users to set the fixed cost threshold.

    Parameters:
    -------
    fixed_costs (nbfloat): Total fixed costs over the modelling horizon, units $.

    Returns:
    -------
    boolean: True if fixed cost constraint is satisfied. Otherwise, False.
    """
    return (solution.annual_cost / solution.static.mean_annual_demand_mwh) < solution.fixed_costs_threshold  # $/MWh_demand


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def objective(solution: Solution_InstanceType) -> tuple[float]:
    """
    Evaluates the long-term energy planning system, through the calculation of investment and unit committment
    costs. Penalty functions are used to soft-constrain fixed costs and reliability.

    Notes:
    -------
    - Fixed costs are calculated first, allowing the fixed cost constraint to be evaluated before unit committment.
    This allows low-quality solutions to be rapidly discarded and penalised.
    - Variable costs require complete evaluation of the unit committment business rules.
    - If the fixed cost or reliability constraint is breached, then self.lcoe will return as $0/MWh. If the soft
    constraints are satisfied, then self.penalties will return as 0. The self.lcoe and self.penalties are summed
    together to provide the differential evolution energy (cost) of the candidate solution.

    Parameters:
    -------
    None.

    Returns:
    -------
    UniTuple(nbfloat, 2): A UniTuple containing two nbfloat values. The first value is the LCOE and the second value
        is the penalties for penalty function violations.

    Side-effects:
    -------
    Attributes modified for Solution instance: lcoe, penalties.
    Attributes modified for values in Solution.fleet.generators, Solution.fleet.storages, Solution.network.major_lines,
        Solution.network.minor_lines: lt_costs.
    Attributes modified for LTCosts instances referenced in the lt_costs attributes: fom, annualised_build, vom, fuel.

    Dynamic jitlass instances are substantially modified within this method. The endogenous time-series data and temporary
    values are modified throughout the balance_residual_load method. Attributes that are modified are marked using
    *Dynamic* or *Precharging* comments in the relevant jitclass definitions.
    """

    calculate_fixed_costs(solution, True)
    if not check_fixed_costs(solution):
        solution.penalties += solution.annual_cost
        return solution.lcoe, solution.penalties  # End early if fixed cost constraint breached
    reliability_check = balance_residual_load(solution)
    if not reliability_check:
        return solution.lcoe, solution.penalties  # End early if reliability constraint breached
    calculate_variable_costs(solution)

    solution.lcoe = solution.annual_cost / solution.static.mean_annual_demand_mwh  # $/MWh
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def extract_details(
    solution: Solution_InstanceType,
    details_slice: nbfloat[:]
) -> None:
    """
    Aggregates operational data directly into a pre-allocated 1D array slice
    using hard-coded idx logic.
    """
    details_slice[:] = 0.0
    details_slice[0] = solution.lcoe
    details_slice[1] = solution.penalties

    for gen in solution.fleet.generators.values():
        idx = gen.unit_type_idx
        details_slice[idx] += gen.lt_generation
        details_slice[idx + 1] += gen.unit_lt_hours

    for sto in solution.fleet.storages.values():
        idx = sto.unit_type_idx
        details_slice[idx] += sto.lt_generation

    for line in solution.network.major_lines.values():
        idx = line.unit_type_idx
        details_slice[idx] += line.lt_flows

    for line in solution.network.minor_lines.values():
        idx = line.unit_type_idx
        details_slice[idx] += line.lt_flows

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def evaluate_from_details(
    solution: Solution_InstanceType,
    details_slice: nbfloat[:],
) -> None:
    """
    Estimate cost from operational data by populating the lt_generation, unit_lt_hours, and lt_flows from previous evaluation.
    CAPEX fixed costs are calculated from the candidate solution vector
    OPEX costs are calculate from the operational data.
    NOTE: The operational costs are summed and saved on a unit-type by unit-type basis.
        This assumes that each unit type has the same variable costs, which is consistent with the current model
         but may not be true in the future
    """

    calculate_fixed_costs(solution, True)

    # Track which indices have been evaluated to ensure we only calculate variable costs ONCE per unit type.
    processed = np.zeros(details_slice.shape[0], dtype=np.bool_)

    acc = 0.0
    # Generators
    for gen in solution.fleet.generators.values():
        idx = gen.unit_type_idx
        if processed[idx]:
            continue
        agg_gen = details_slice[idx]
        agg_hrs = details_slice[idx + 1]

        acc += ltcosts_m.calculate_vom_generic(agg_gen, gen.cost.vom, solution.static.year_float)
        acc += ltcosts_m.calculate_fuel_generic(
            agg_gen, solution.static.year_float, agg_hrs, gen.cost.fuel_cost_mwh, gen.cost.fuel_cost_h
        )
        processed[idx] = True
        processed[idx + 1] = True

    solution.annual_cost += acc
    acc = 0.0

    # Storages
    for sto in solution.fleet.storages.values():
        idx = sto.unit_type_idx
        if processed[idx]:
            continue
        acc += ltcosts_m.calculate_vom_generic(details_slice[idx], sto.cost.vom, solution.static.year_float)
        processed[idx] = True

    solution.annual_cost += acc
    acc = 0.0

    # Major Lines
    for line in solution.network.major_lines.values():
        idx = line.unit_type_idx
        if processed[idx]:
            continue
        acc += ltcosts_m.calculate_vom_generic(details_slice[idx], line.cost.vom, solution.static.year_float)
        processed[idx] = True

    solution.annual_cost += acc
    acc = 0.0

    # Minor Lines
    for line in solution.network.minor_lines.values():
        idx = line.unit_type_idx
        if processed[idx]:
            continue
        acc += ltcosts_m.calculate_vom_generic(details_slice[idx], line.cost.vom, solution.static.year_float)
        processed[idx] = True

    solution.annual_cost += acc

    # note: solution.lcoe is stored in details_slice[0] but this function is to facilitate
    #       uncertainty analysis under cost assumptions, hence recalculation
    solution.lcoe = solution.annual_cost / solution.static.mean_annual_demand_mwh
    solution.penalties = details_slice[1]
    solution.evaluated = True

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def get_scaled_points(solution: Solution_InstanceType):
    calculate_partial_costs(solution)
    solution.x_lcoe /= solution.static.mean_annual_demand_mwh  # $/MWh
    return solution.x_lcoe


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def evaluate(solution: Solution_InstanceType) -> Solution_InstanceType:
    """
    Wrapper that evaluates the objective function and updates the evaluation state.

    Returns:
    -------
    Solution: The evaluated Solution instance with calculated LCOE, penalties, and endogenous time-series and cost
        data.

    Side-effects:
    -------
    Attributes modified for Solution instance: lcoe, penalties, evaluated.
    """
    objective(solution)
    solution.evaluated = True
    return None
