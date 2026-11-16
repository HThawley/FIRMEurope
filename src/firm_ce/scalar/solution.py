# type: ignore
import numpy as np

from firm_ce.common.constants import JIT_ENABLED, NUM_THREADS, FASTMATH, BOUNDSCHECK
from firm_ce.common.jit_overload import jitclass, njit, prange
from firm_ce.common.typing import boolean, nbfloat, unicode_type, TypedDict, nbintp, npfloat
from firm_ce.scalar.balancing import balance_for_period
from firm_ce.scalar.entities.components import Fleet_InstanceType
from firm_ce.scalar.entities.topology import Network_InstanceType
from firm_ce.scalar.methods import fleet_m, generator_m, line_m, ltcosts_m, network_m, static_m, storage_m
from firm_ce.system.parameters import ScenarioParameters_InstanceType


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
        self.x = x
        self.evaluated = False
        self.annual_cost = 0.0
        self.lcoe = 0.0
        self.x_lcoe = np.zeros(len(x), nbfloat)
        self.penalties = 0.0

        self.static = static
        self.balancing_type = balancing_type
        self.fixed_costs_threshold = fixed_costs_threshold

        self.network = network_m.create_dynamic_copy(network)
        self.fleet = fleet_m.create_dynamic_copy(fleet, self.network.nodes, self.network.minor_lines)

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
    Calculate total fixed costs for all assets.
    """
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
    """
    fleet_m.calculate_lt_generations(
        solution.fleet,
        solution.static.resolution,
    )
    network_m.calculate_lt_flows(
        solution.network,
        solution.static.resolution,
    )

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
    solution.x_lcoe[:] = 0.0
    for generator in solution.fleet.generators.values():
        if generator.candidate_x_idx != -1:
            solution.x_lcoe[generator.candidate_x_idx] += generator_m.get_partial_cost(
                generator, solution.static.year_float
            )

    for storage in solution.fleet.storages.values():
        if storage.candidate_p_x_idx != -1:
            solution.x_lcoe[storage.candidate_p_x_idx] += storage_m.get_partial_cost_power(
                storage, solution.static.year_float
            )
        if storage.candidate_e_x_idx != -1:
            solution.x_lcoe[storage.candidate_e_x_idx] += storage_m.get_partial_cost_energy(storage)

    for line in solution.network.major_lines.values():
        if line.candidate_x_idx != -1:
            solution.x_lcoe[line.candidate_x_idx] += line_m.get_partial_cost(line, solution.static.year_float)

    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def check_fixed_costs(solution: Solution_InstanceType) -> boolean:
    """
    Check the fixed cost constraint against the configured threshold.
    """
    return (solution.annual_cost / solution.static.mean_annual_demand_mwh) < solution.fixed_costs_threshold


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def objective(solution: Solution_InstanceType) -> tuple[float]:
    """
    Evaluates the long-term energy planning system, through the calculation of investment and unit committment
    costs. Penalty functions are used to soft-constrain fixed costs and reliability.
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
def extract_details(solution: Solution_InstanceType, details_slice: nbfloat[:]) -> None:
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
    """
    calculate_fixed_costs(solution, True)

    processed = np.zeros(details_slice.shape[0], dtype=np.bool_)

    acc = 0.0
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

    for sto in solution.fleet.storages.values():
        idx = sto.unit_type_idx
        if processed[idx]:
            continue
        acc += ltcosts_m.calculate_vom_generic(details_slice[idx], sto.cost.vom, solution.static.year_float)
        processed[idx] = True

    solution.annual_cost += acc
    acc = 0.0

    for line in solution.network.major_lines.values():
        idx = line.unit_type_idx
        if processed[idx]:
            continue
        acc += ltcosts_m.calculate_vom_generic(details_slice[idx], line.cost.vom, solution.static.year_float)
        processed[idx] = True

    solution.annual_cost += acc
    acc = 0.0

    for line in solution.network.minor_lines.values():
        idx = line.unit_type_idx
        if processed[idx]:
            continue
        acc += ltcosts_m.calculate_vom_generic(details_slice[idx], line.cost.vom, solution.static.year_float)
        processed[idx] = True

    solution.annual_cost += acc

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
    """
    objective(solution)
    solution.evaluated = True
    return None


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK, parallel=True)
def build_eval_and_return_solutions(
    xs: nbfloat[:, :],
    static: ScenarioParameters_InstanceType,
    fleet: Fleet_InstanceType,
    network: Network_InstanceType,
    balancing_type: unicode_type,
    fixed_costs_threshold: nbfloat,
) -> TypedDict[nbintp, Solution_InstanceType]:
    """
    Convenience for efficient post-processing - not to be used in optimisation loops
    """
    xs = xs.astype(nbfloat)
    n_points = xs.shape[0]
    solutions = TypedDict()

    for j in prange(n_points):
        solutions[j] = Solution(
            xs[j],
            static,
            fleet,
            network,
            balancing_type,
            fixed_costs_threshold,
        )
        evaluate(solutions[j])

    return solutions


@njit(fastmath=FASTMATH, boundscheck=BOUNDSCHECK)
def extract_energy_space_scalar(
    solution: Solution_InstanceType,
    nodal_demand_totals: nbfloat[:],
    norm_mode: nbintp,
    agg_mode: nbintp,
    n_gen_types: nbintp,
    n_energy_types: nbintp,
    scaled_slice: nbfloat[:],
) -> None:
    """
    Extracts gross generation + storage discharge + curtailment into `scaled_slice`
    according to (norm_mode, agg_mode) for the scalar backend.
    """
    scaled_slice[:] = npfloat(0.0)
    n_nodes = len(nodal_demand_totals)
    sys_demand = np.sum(nodal_demand_totals)
    inv_sys = npfloat(1.0) / sys_demand if sys_demand > 0.0 else npfloat(1.0)
    inv_nodes = npfloat(1.0) / npfloat(n_nodes)
    res = solution.static.resolution

    # 1. Generators (gross generation)
    for gen in solution.fleet.generators.values():
        n = gen.node.order
        slot = (gen.unit_type_idx - 2) // 2
        if gen.is_flexible:
            e_gwh = gen.lt_generation
        else:
            e_gwh = np.sum(gen.data) * gen.capacity * res

        if agg_mode == 1:  # AGG_UNIT_TYPE
            w = (inv_nodes / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[slot] += e_gwh * w
        elif agg_mode == 2:  # AGG_NODE
            w = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[n] += e_gwh * w
        else:  # AGG_NONE
            w = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[n * n_energy_types + slot] += e_gwh * w

    # 2. Storages (gross discharge)
    for sto in solution.fleet.storages.values():
        n = sto.node.order
        slot = sto.unit_type_idx - 2 - n_gen_types
        e_gwh = sto.lt_generation

        if agg_mode == 1:  # AGG_UNIT_TYPE
            w = (inv_nodes / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[slot] += e_gwh * w
        elif agg_mode == 2:  # AGG_NODE
            w = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[n] += e_gwh * w
        else:  # AGG_NONE
            w = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[n * n_energy_types + slot] += e_gwh * w

    # 3. Nodal Curtailment
    curt_slot = n_energy_types - 1
    for node in solution.network.nodes.values():
        n = node.order
        curt_gwh = np.sum(np.abs(node.curtail)) * res

        if agg_mode == 1:  # AGG_UNIT_TYPE
            w = (inv_nodes / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[curt_slot] += curt_gwh * w
        elif agg_mode == 2:  # AGG_NODE
            w = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[n] += curt_gwh * w
        else:  # AGG_NONE
            w = (npfloat(1.0) / nodal_demand_totals[n]) if norm_mode == 0 else inv_sys
            scaled_slice[n * n_energy_types + curt_slot] += curt_gwh * w
