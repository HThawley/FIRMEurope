# type: ignore
import numpy as np

from firm_ce.common.constants import TOLERANCE
from firm_ce.common.typing import npfloat
from firm_ce.system.scenario import Scenario
from firm_ce.backend.scalar.solution import Solution, Solution_InstanceType
from firm_ce.backend.tensor.solution import SolutionTensorType
from firm_ce.system.tensor.costs import (
    get_generator_costs,
    get_line_costs,
    get_storage_costs,
)

from firm_ce.fast_methods import generator_m, storage_m


def map_tensor_to_scalar(
    scenario: Scenario,
    solutionTensor: SolutionTensorType,
) -> Solution_InstanceType:

    assets = solutionTensor.assets
    ops = solutionTensor.operations
    res = solutionTensor.static.resolution
    years_float = solutionTensor.static.years_float

    if scenario.config.parameterisation == "relative":
        x_abs = scenario.convert_x_to_abs(solutionTensor.x)
    else:
        x_abs = solutionTensor.x

    solution = Solution(
        solutionTensor.x,
        scenario.static,
        scenario.fleet,
        scenario.network,
        scenario.config.balancing_type,
        scenario.config.fixed_costs_threshold,
    )

    for gen in solution.fleet.generators.values():
        n = gen.node.order
        x_idx = gen.candidate_x_idx

        gen.new_build = x_abs[x_idx] if x_idx != -1 else 0.0
        gen.capacity = npfloat(gen.initial_capacity) + npfloat(gen.new_build)

        if gen.unit_type == "pv_fixed":
            agg_cap, agg_dispatch = assets.Cpfix[n], ops.Mpfix[:, n]
        elif gen.unit_type == "pv_track":
            agg_cap, agg_dispatch = assets.Cpsat[n], ops.Mpsat[:, n]
        elif gen.unit_type == "offw":
            agg_cap, agg_dispatch = assets.Coffw[n], ops.Moffw[:, n]
        elif gen.unit_type == "onsw":
            agg_cap, agg_dispatch = assets.Consw[n], ops.Monsw[:, n]
        elif gen.unit_type == "biomass":
            agg_cap, agg_dispatch = assets.Cpeak[n, 0], ops.Mpeak[:, n, 0]
        elif gen.unit_type == "biogas":
            agg_cap, agg_dispatch = assets.Cpeak[n, 1], ops.Mpeak[:, n, 1]
        elif gen.unit_type == "ccgt":
            agg_cap, agg_dispatch = assets.Cpeak[n, 2], ops.Mpeak[:, n, 2]
        elif gen.unit_type == "nuclear":
            agg_cap, agg_dispatch = assets.Cnuke[n], ops.Mnuke[:, n]
        elif gen.unit_type == "nuclear_lte":
            agg_cap, agg_dispatch = assets.Cnuke[n], ops.Mnuke[:, n]
        elif gen.unit_type == "ror":
            agg_cap, agg_dispatch = solutionTensor.static.Eror[n], solutionTensor.static.Mror[:, n]
        else:
            continue

        ratio = npfloat(gen.capacity) / agg_cap if agg_cap > TOLERANCE else 0.0

        if gen.is_flexible:
            generator_m.allocate_memory(gen, solution.static.intervals_count)

        # apportion dispatch by power in the case of multiple generators at a single node
        gen.dispatch_power = agg_dispatch * ratio

        if gen.is_flexible:
            generator_m.calculate_lt_generation(gen, res)
            power_trace = gen.dispatch_power
        else:
            generator_m.update_lt_generation(gen, gen.dispatch_power, res)
            power_trace = gen.data * gen.capacity

        ann_build, fom, vom, fuel = get_generator_costs(gen, res, years_float)
        gen.lt_costs.annualised_build_p = ann_build * gen.capacity
        gen.lt_costs.fom = fom * gen.capacity
        gen.lt_costs.vom = vom * power_trace.sum()  # res and years already in `vom`
        gen.lt_costs.fuel = fuel * power_trace.sum()  # res and years already in `fuel`

    # Update Storages
    for sto in solution.fleet.storages.values():
        n = sto.node.order
        p_idx = sto.candidate_p_x_idx
        e_idx = sto.candidate_e_x_idx

        sto.new_build_p = x_abs[p_idx] if p_idx != -1 else 0.0
        sto.power_capacity = sto.initial_power_capacity + sto.new_build_p

        sto.new_build_e = x_abs[e_idx] if e_idx != -1 else 0.0
        sto.energy_capacity = sto.initial_energy_capacity + sto.new_build_e

        # Apportionment mapping
        if "phes" in sto.unit_type:
            agg_cap_p, agg_cap_e = assets.CstorageP[n, 0], assets.CstorageE[n, 0]
            agg_dispatch = ops.Mdischarge[:, n, 0] - ops.Mcharge[:, n, 0]
            agg_soc = ops.Mstorage[:, n, 0]
        elif sto.unit_type == "bess4h":
            agg_cap_p, agg_cap_e = assets.CstorageP[n, 1], assets.CstorageE[n, 1]
            agg_dispatch = ops.Mdischarge[:, n, 1] - ops.Mcharge[:, n, 1]
            agg_soc = ops.Mstorage[:, n, 1]
        elif sto.unit_type == "bess2h":
            agg_cap_p, agg_cap_e = assets.CstorageP[n, 2], assets.CstorageE[n, 2]
            agg_dispatch = ops.Mdischarge[:, n, 2] - ops.Mcharge[:, n, 2]
            agg_soc = ops.Mstorage[:, n, 2]
        elif sto.unit_type == "pond":
            agg_cap_p, agg_cap_e = assets.ChydP[n, 0], assets.ChydE[n, 0]
            agg_dispatch = ops.Mhydro[:, n, 0]
            agg_soc = ops.Mreservoir[:, n, 0]
        elif sto.unit_type == "hydro":
            agg_cap_p, agg_cap_e = assets.ChydP[n, 1], assets.ChydE[n, 1]
            agg_dispatch = ops.Mhydro[:, n, 1]
            agg_soc = ops.Mreservoir[:, n, 1]
        else:
            continue

        ratio_p = sto.power_capacity / agg_cap_p if agg_cap_p > 1e-9 else 0.0
        ratio_e = sto.energy_capacity / agg_cap_e if agg_cap_e > 1e-9 else 0.0

        storage_m.allocate_memory(sto, solution.static.intervals_count)

        sto.dispatch_power = agg_dispatch * ratio_p
        sto.stored_energy = agg_soc * ratio_e

        storage_m.calculate_lt_generation(sto, res)

        ann_build_p, ann_build_e, fom, vom = get_storage_costs(sto, res, years_float)
        sto.lt_costs.annualised_build_p = ann_build_p * sto.power_capacity
        sto.lt_costs.annualised_build_e = ann_build_e * sto.energy_capacity
        sto.lt_costs.fom = fom * sto.power_capacity
        sto.lt_costs.vom = vom * sto.dispatch_power.sum()  # res and years already in `vom`

    # Update Major Lines
    for line in solution.network.major_lines.values():
        idx = line.order
        x_idx = line.candidate_x_idx

        line.new_build = x_abs[x_idx] if x_idx != -1 else 0.0
        line.capacity = line.initial_capacity + line.new_build
        line.flows = ops.Tnetflow[:, idx].copy()
        line.lt_flows = np.sum(np.abs(line.flows)) * res

        ann_build, fom, vom = get_line_costs(line, res, years_float)
        line.lt_costs.annualised_build_p = ann_build * line.capacity
        line.lt_costs.fom = fom * line.capacity
        line.lt_costs.vom = vom * np.abs(line.flows).sum()

    # Update Nodes
    for node in solution.network.nodes.values():
        n = node.order
        node.deficits = ops.Mdeficit[:, n].copy()
        node.spillage = -ops.Mcurtail[:, n].copy()
        node.imports_exports = ops.Mimport[:, n]  # Mexport is negative

    solution.evaluated = solutionTensor.evaluated
    solution.penalties = solutionTensor.penalties
    return solution
