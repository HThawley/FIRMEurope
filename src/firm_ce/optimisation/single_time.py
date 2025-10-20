# type: ignore
import time

import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, PENALTY_MULTIPLIER
from firm_ce.common.jit_overload import jitclass, njit, prange
from firm_ce.common.typing import boolean, float64, unicode_type
from firm_ce.fast_methods import fleet_m, generator_m, line_m, network_m, reservoir_m, static_m, storage_m
from firm_ce.optimisation.balancing import balance_for_period
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType

if JIT_ENABLED:
    from numba import set_num_threads

    set_num_threads(int(NUM_THREADS))

    solution_spec = [
        ("x", float64[:]),
        ("evaluated", boolean),
        ("lcoe", float64),
        ("penalties", float64),
        ("balancing_type", unicode_type),
        ("fixed_costs_threshold", float64),
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
    - Reliability and fixed-cost constraints may terminate evaluation early, accumulating penalties scaled by
    PENALTY_MULTIPLIER. If the fixed cost threshold is set too low, it is very likely that the optimisation will
    get stuck in a local minimum (fixed costs just below threshold, reliability constraint still breached). This
    issue can be mitigated by increasing the mutation factor, raising the fixed cost threshold, or increasing the build
    limit of flexible Generator capacity.
    - The energy (cost) returned by the objective function is a system-level levelised cost of electricity (LCOE). This
    is calculated to be the sum of variable and fixed costs of all assets in the system divided by total operational
    demand over the modelling horizon, units $/MWh.

    Attributes:
    -------
    x (float64[:]): Candidate solution decision variable vector.
    evaluated (boolean): Flag indicating whether `objective()` has been evaluated.
    lcoe (float64): Levelised cost of electricity for the candidate, units $/MWh.
    penalties (float64): Accumulated penalties for soft-constraint violations (fixed-costs and reliability), units $ or GW.
    balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over the entire
        time horizon at the specified resolution).
    fixed_costs_threshold (float64): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit committment
        problem.
    static (ScenarioParameters_InstanceType): Static scenario parameters (unsafe to modify).
    fleet (Fleet_InstanceType): Dynamic copy of Fleet instance for this evaluation (safe to modify some attributes).
    network (Network_InstanceType): Dynamic copy of Network instance for this evaluation (safe to modify some attributes).
    """

    def __init__(
        self,
        x: float64[:],
        static: ScenarioParameters_InstanceType,
        fleet: Fleet_InstanceType,
        network: Network_InstanceType,
        balancing_type: unicode_type,
        fixed_costs_threshold: float64,
    ) -> None:
        """
        Initialise a Solution instance and construct dynamic copies of Fleet and Network.

        Parameters:
        -------
        x (float64[:]): Candidate solution decision variable vector.
        static (ScenarioParameters_InstanceType): Static scenario parameters (unsafe to modify).
        fleet (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
        network (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
        balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over
            the entire time horizon at the specified resolution).
        fixed_costs_threshold (float64): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
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
        self.lcoe = 0.0
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

        fleet_m.build_capacities(self.fleet, x, self.static.interval_resolutions)
        network_m.build_capacity(self.network, x)

        fleet_m.allocate_memory(self.fleet, self.static.intervals_count)
        network_m.allocate_memory(self.network, self.static.intervals_count)

        network_m.assign_storage_merit_orders(self.network, self.fleet.storages)
        network_m.assign_reservoir_merit_orders(self.network, self.fleet.reservoirs)
        network_m.assign_flexible_merit_orders(self.network, self.fleet.generators)

    def balance_residual_load(self) -> boolean:
        """
        Evaluate the unit committment business rules over the entire modelling horizon.

        Notes:
        -------
        - At the end of each calendar year, the reliability constraint is evaluated. The method returns early
        if the reliability constraint is breached for any year.
        - Stored energy in Reservoir and Storage systems is initialised at the start of the modelling period.
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
        Dynamic jitlass instances are substantially modified within this method. The stored energy of Reservoirs and
        Storage systems and remaining energy for flexible Generators are initialised using Fleet pseudo-methods. The
        endogenous time-series data and temporary values are modified throughout the balance_for_period function.
        Attributes that are modified are marked using *Dynamic* or *Precharging* comments in the relevant jitclass
        definitions.
        """
        fleet_m.initialise_stored_energies(self.fleet)
        for year in range(self.static.year_count):
            first_t, last_t = static_m.get_year_t_boundaries(self.static, year)
            fleet_m.initialise_annual_limits(self.fleet, year, first_t)
            balance_for_period(first_t, last_t, self.balancing_type == "full", self, year)
            annual_unserved_energy = network_m.calculate_period_unserved_energy(
                self.network, first_t, last_t, self.static.interval_resolutions
            )

            # End early if reliability constraint breached for any year
            if not static_m.check_reliability_constraint(self.static, year, annual_unserved_energy):
                self.penalties += (self.static.year_count - year) * annual_unserved_energy * PENALTY_MULTIPLIER
                return False
        return True

    def calculate_fixed_costs(self) -> float64:
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
        float64: Total fixed costs over the modelling horizon, units $.

        Side-effects:
        -------
        Attributes modified for values in Solution.fleet.generators, Solution.fleet.reservoirs, Solution.fleet.storages,
            Solution.network.major_lines, Solution.network.minor_lines: lt_costs.
        Attributes modified for LTCosts instances referenced in the lt_costs attributes: fom, annualised_build.
        """
        total_costs = 0.0
        years_float = self.static.year_count * self.static.fom_scalar

        for generator in self.fleet.generators.values():
            total_costs += generator_m.calculate_fixed_costs(generator, years_float, self.static.year_count)

        for reservoir in self.fleet.reservoirs.values():
            total_costs += reservoir_m.calculate_fixed_costs(reservoir, years_float, self.static.year_count)

        for storage in self.fleet.storages.values():
            total_costs += storage_m.calculate_fixed_costs(storage, years_float, self.static.year_count)

        for line in self.network.major_lines.values():
            total_costs += line_m.calculate_fixed_costs(line, years_float, self.static.year_count)

        for line in self.network.minor_lines.values():
            total_costs += line_m.calculate_fixed_costs(line, years_float, self.static.year_count)

        return total_costs

    def calculate_variable_costs(self) -> float64:
        """
        Calculate total variable costs based on dispatch and flows derived through unit committment
        business rules.

        Notes:
        -----
        - This method should not be called before complete evaluation of the unit committment business
        rules over the modelling horizon.

        Returns:
        -------
        float64: Total variable costs over the modelling horizon, units $.

        Side-effects:
        -------
        Attributes modified for values in Solution.fleet.generators, Solution.fleet.reservoirs,
            Solution.fleet.storages, Solution.network.major_lines, Solution.network.minor_lines: lt_costs.
        Attributes modified for LTCosts instances referenced in the lt_costs attributes: vom, fuel.
        """
        total_costs = 0.0

        fleet_m.calculate_lt_generations(
            self.fleet,
            self.static.interval_resolutions,
        )
        network_m.calculate_lt_flows(
            self.network,
            self.static.interval_resolutions,
        )

        for generator in self.fleet.generators.values():
            total_costs += generator_m.calculate_variable_costs(generator)

        for reservoir in self.fleet.reservoirs.values():
            total_costs += reservoir_m.calculate_variable_costs(reservoir)

        for storage in self.fleet.storages.values():
            total_costs += storage_m.calculate_variable_costs(storage)

        for line in self.network.major_lines.values():
            total_costs += line_m.calculate_variable_costs(line)

        for line in self.network.minor_lines.values():
            total_costs += line_m.calculate_variable_costs(line)

        return total_costs

    def check_fixed_costs(self, fixed_costs: float64) -> boolean:
        """
        Check the fixed cost constraint against the configured threshold.

        Notes:
        -----
        - Fixed costs are evaluated relative to total operational demand. This provides consistency with the
        system-level LCOE, making it easier for users to set the fixed cost threshold.

        Parameters:
        -------
        fixed_costs (float64): Total fixed costs over the modelling horizon, units $.

        Returns:
        -------
        boolean: True if fixed cost constraint is satisfied. Otherwise, False.
        """
        return (fixed_costs / sum(self.static.year_energy_demand) / 1000) < self.fixed_costs_threshold  # $/MWh_demand

    def objective(self):
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
        UniTuple(float64, 2): A UniTuple containing two float64 values. The first value is the LCOE and the second value
            is the penalties for penalty function violations.

        Side-effects:
        -------
        Attributes modified for Solution instance: lcoe, penalties.
        Attributes modified for values in Solution.fleet.generators, Solution.fleet.reservoirs, Solution.fleet.storages,
            Solution.network.major_lines, Solution.network.minor_lines: lt_costs.
        Attributes modified for LTCosts instances referenced in the lt_costs attributes: fom, annualised_build, vom, fuel.

        Dynamic jitlass instances are substantially modified within this method. The endogenous time-series data and temporary
        values are modified throughout the balance_residual_load method. Attributes that are modified are marked using
        *Dynamic* or *Precharging* comments in the relevant jitclass definitions.
        """

        total_costs = self.calculate_fixed_costs()
        if not self.check_fixed_costs(total_costs):
            return self.lcoe, total_costs * PENALTY_MULTIPLIER  # End early if fixed cost constraint breached
        reliability_check = self.balance_residual_load()
        if not reliability_check:
            return self.lcoe, self.penalties  # End early if reliability constraint breached
        total_costs += self.calculate_variable_costs()
        total_line_losses = network_m.calculate_lt_line_losses(self.network)

        lcoe = total_costs / np.abs(sum(self.static.year_energy_demand) - total_line_losses) / 1000  # $/MWh
        return lcoe, self.penalties

    def evaluate(self):
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
        self.lcoe, self.penalties = self.objective()
        self.evaluated = True
        return self


if JIT_ENABLED:
    Solution_InstanceType = Solution.class_type.instance_type
else:
    Solution_InstanceType = Solution


@njit(parallel=True)
def parallel_wrapper(
    xs: float64[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: float64,
) -> float64[:, :]:
    """
    A wrapper receives the vectorised differential evolution population and evaluates it over a parallel range.
    A Solution instance is created for each candidate solution and evaluated. The parallel range splits the candidate
    solutions within the population across a number of workers (defined by the NUM_THREADS environment variable).
    This is an embarassingly parallel process.

    Parameters:
    -------
    xs (float64[:, :]): 2-dimensional array containing population for an iteration of the differential
        evolution. Each row is a separate candidate solution, each column is a decision variable.
    static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
    network (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
    balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over
        the entire time horizon at the specified resolution).
    fixed_costs_threshold (float64): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit
        committment problem.

    Returns:
    -------
    float64[:, :]: A 2-dimensional array with 3 rows and a separate column for each candidate solution in the
        population. The first row is the total energy (cost) of the objective function, second row is the LCOE, and
        third row is the penalties for each candidate solution.
    """
    n_points = xs.shape[1]
    result = np.zeros((3, n_points), dtype=np.float64)
    for j in prange(n_points):
        xj = xs[:, j]
        sol = Solution(xj, static, fleet, network, balancing_type, fixed_costs_threshold)
        sol.evaluate()
        result[0, j] = sol.lcoe + sol.penalties
        result[1, j] = sol.lcoe
        result[2, j] = sol.penalties
    return result


def evaluate_vectorised_xs(
    xs: float64[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: float64,
):
    """
    A wrapper receives the vectorised differential evolution population and passes it to the parallel wrapper.
    This function is not JITed which allows the `time` package to be used to evaluate optimisation times for
    the iteration.

    Parameters:
    -------
    xs (float64[:, :]): 2-dimensional array containing population for an iteration of the differential
        evolution. Each row is a separate candidate solution, each column is a decision variable.
    static (ScenarioParameters_InstanceType): Static scenario parameters.
    fleet (Fleet_InstanceType): Static Fleet jitclass instance used to derive a dynamic copy for evaluation.
    network (Network_InstanceType): Static Network jitclass instance used to derive a dynamic copy for evaluation.
    balancing_type (unicode_type): Balancing mode (e.g., 'full' for balancing with the complete time-series over
        the entire time horizon at the specified resolution).
    fixed_costs_threshold (float64): Upper bound on fixed costs intensity, units $/MWh of operational demand. Allows
        low-quality solutions to be rapidly discarded and penalised without evaluating the time-consuming unit
        committment problem.

    Returns:
    -------
    float64[:]: Total energies (costs) of the evaluated objective functions for each candidate solution in the
        population. Each column is the energy of a different candidate solution. The energy is the sum of LCOE
        and the penalties. This is the value minimised by the differential evolution optimisation.
    """
    start_time = time.time()
    result = parallel_wrapper(xs, static, fleet, network, balancing_type, fixed_costs_threshold)
    end_time = time.time()
    print(f"Average objective time: {NUM_THREADS*(end_time-start_time)/xs.shape[1]:.4f} seconds.")
    print(f"Iteration time: {(end_time-start_time):.4f} seconds for {NUM_THREADS} workers.")
    return result[0, :]
