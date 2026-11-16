# type: ignore
from collections import defaultdict
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.typing import TypedDict, nbintp, npfloat, npintp, unicode_type
from firm_ce.system.taxonomy import identify_tech


def assign_indices_and_get_abs_bounds(
    fleet,
    network,
) -> tuple[NDArray[npfloat], NDArray[npfloat], TypedDict]:
    lower, upper = [], []
    x_index = 0

    # Canonical generator order matching StaticTensor offset chain
    gen_order = (
        "pv_fixed",
        "pv_track",
        "offw",
        "onsw",
        "biogas",
        "biomass",
        "ocgt",
        "nuclear",
        "nuclear_lte",
    )
    g_by_type = defaultdict(list)
    for gen in fleet.generators.values():
        gen.candidate_x_idx = -1
        if gen.max_build > 0:
            g_by_type[gen.unit_type].append(gen)

    g_unit_types = defaultdict(list)
    for ut in list(gen_order) + [k for k in g_by_type if k not in gen_order]:
        for gen in g_by_type[ut]:
            gen.candidate_x_idx = x_index
            lower.append(gen.min_build)
            upper.append(gen.max_build)
            g_unit_types[ut].append(gen.node.order)
            x_index += 1

    # Group storages in canonical order matching StaticTensor: new_phes -> bess2h -> bess4h
    phes_by_node = defaultdict(list)
    other_s_types = defaultdict(list)
    for sto in fleet.storages.values():
        sto.candidate_p_x_idx = -1
        sto.candidate_e_x_idx = -1
        if sto.unit_type in ("aa_phes", "e_phes"):
            if sto.max_build_p > 0 or sto.max_build_e > 0:
                phes_by_node[sto.node.order].append(sto)
        elif sto.max_build_p > 0:
            other_s_types[sto.unit_type].append(sto)

    s_unit_types = defaultdict(list)

    # 1. Storage Power: new_phes (in node order)
    for n_order in sorted(phes_by_node.keys()):
        stos = phes_by_node[n_order]
        min_p = sum(s.min_build_p for s in stos)
        max_p = sum(s.max_build_p for s in stos)
        if max_p > 0:
            for s in stos:
                s.candidate_p_x_idx = x_index
            lower.append(min_p)
            upper.append(max_p)
            s_unit_types["new_phes"].append(n_order)
            x_index += 1

    # 2. Storage Power: bess2h then bess4h (matching StaticTensor: b2p_offset then b4p_offset)
    sto_order = ("bess2h", "bess4h")
    for ut in list(sto_order) + [k for k in other_s_types if k not in sto_order]:
        for sto in other_s_types[ut]:
            sto.candidate_p_x_idx = x_index
            lower.append(sto.min_build_p)
            upper.append(sto.max_build_p)
            s_unit_types[ut].append(sto.node.order)
            x_index += 1

    # 3. Storage Energy: new_phes (in same node order as power)
    for n_order in sorted(phes_by_node.keys()):
        stos = phes_by_node[n_order]
        min_e = sum(s.min_build_e for s in stos)
        max_e = sum(s.max_build_e for s in stos)
        if max_e > 0:
            for s in stos:
                s.candidate_e_x_idx = x_index
            lower.append(min_e)
            upper.append(max_e)
            x_index += 1

    # 4. Major Lines
    for line in network.major_lines.values():
        if line.max_build > 0:
            line.candidate_x_idx = x_index
            lower.append(line.min_build)
            upper.append(line.max_build)
            x_index += 1
        else:
            line.candidate_x_idx = -1

    asset_node_map = TypedDict.empty(key_type=unicode_type, value_type=nbintp[:])
    for k, v in g_unit_types.items():
        asset_node_map[k] = np.array(v, dtype=npintp)
    for k, v in s_unit_types.items():
        asset_node_map[k] = np.array(v, dtype=npintp)

    return np.array(lower, npfloat), np.array(upper, npfloat), asset_node_map


def construct_relative_space(
    fleet,
    network,
    lower_bounds_abs: NDArray[npfloat],
    upper_bounds_abs: NDArray[npfloat],
) -> tuple[NDArray[npfloat], NDArray[npfloat], NDArray[npfloat]]:
    lower_bounds_rel = np.zeros_like(lower_bounds_abs)
    upper_bounds_rel = np.zeros_like(upper_bounds_abs)
    abs_rel_scaler = np.zeros_like(lower_bounds_abs)

    def _scale(idx: int, denom: float) -> None:
        lower_bounds_rel[idx] = lower_bounds_abs[idx] / denom
        upper_bounds_rel[idx] = upper_bounds_abs[idx] / denom
        abs_rel_scaler[idx] = denom

    processed_idx = set()
    for gen in fleet.generators.values():
        idx = gen.candidate_x_idx
        if idx != -1 and idx not in processed_idx:
            _scale(idx, gen.node.mean_demand)
            processed_idx.add(idx)

    for sto in fleet.storages.values():
        nodal_demand = sto.node.mean_demand
        idx_p = sto.candidate_p_x_idx
        if idx_p != -1 and idx_p not in processed_idx:
            _scale(idx_p, nodal_demand)
            processed_idx.add(idx_p)

        idx_e = sto.candidate_e_x_idx
        if idx_e != -1 and idx_e not in processed_idx:
            _scale(idx_e, nodal_demand)
            processed_idx.add(idx_e)

    for line in network.major_lines.values():
        idx = line.candidate_x_idx
        if idx != -1 and idx not in processed_idx:
            nodal_demand = max(line.node_start.mean_demand, line.node_end.mean_demand)
            _scale(idx, nodal_demand)
            processed_idx.add(idx)

    assert set(range(len(lower_bounds_abs))) == processed_idx, (
        "Not all indices were processed in relative space construction"
    )
    return lower_bounds_rel, upper_bounds_rel, abs_rel_scaler


def parse_x0(scenario_name: str, all_x0s: Dict[str, Dict[str, str]]) -> tuple[NDArray[npfloat], bool]:
    for entry in all_x0s.values():
        if entry["scenario"] == scenario_name:
            try:
                x0_list = [float(x) for x in entry["x_0"].strip().split(",") if x.strip()]
            except AttributeError:
                x0_list = []

            is_relative = str(entry.get("is_relative", "false")).lower() in ("true", "1", "t", "yes", "y")
            return np.array(x0_list, dtype=npfloat), is_relative
    return np.array([], npfloat), False


def approximate_feasible_solution(
    fleet,
    network,
    lower_bounds_abs: NDArray[npfloat],
    lower_bounds_rel: NDArray[npfloat],
    upper_bounds_rel: NDArray[npfloat],
    abs_rel_scaler: NDArray[npfloat],
) -> tuple[NDArray[npfloat], NDArray[npfloat]]:
    """If no initial guess is provided, create an approximate feasible solution."""
    heuristic_x_rel = np.zeros_like(lower_bounds_abs, dtype=npfloat)

    factors = {
        # 'name': <approx energy fraction> / <approx capacity factor>
        "ocgt": 0.05 / 0.05,
        "pv_fixed": 0.3 / 0.15,
        "pv_track": 0.2 / 0.15,
        "onsw": 0.3 / 0.35,
        "offw": 0.3 / 0.5,
        "biogas": 0.02 / 0.2,
        "biomass": 0.02 / 0.2,
        "nuclear": 0.1 / 0.9,
        "nuclear_lte": np.inf,
        # storage
        "bess2h": 0.33,
        "bess4h": 0.33,
        "aa_phes": 0.33,
        "e_phes": 0.33,
        # supply average load for 16 hours
        "aa_phes_e": 16,
        "e_phes_e": 16,
    }

    for gen in fleet.generators.values():
        idx = gen.candidate_x_idx
        if idx != -1:
            heuristic_x_rel[idx] = factors[gen.unit_type]

    for sto in fleet.storages.values():
        idx_p = sto.candidate_p_x_idx
        if idx_p != -1:
            heuristic_x_rel[idx_p] = factors[sto.unit_type]

        idx_e = sto.candidate_e_x_idx
        if idx_e != -1:
            heuristic_x_rel[idx_e] = factors[sto.unit_type + "_e"]

    for line in network.major_lines.values():
        idx = line.candidate_x_idx
        if idx != -1:
            heuristic_x_rel[idx] = 0.2

    heuristic_x_rel = np.clip(heuristic_x_rel, lower_bounds_rel, upper_bounds_rel)
    heuristic_x_abs = heuristic_x_rel * abs_rel_scaler

    return heuristic_x_rel, heuristic_x_abs


def assign_unit_type_idx(fleet, network, backend: str) -> tuple[Dict[str, int], int]:
    """
    Calculates exact memory requirements for operational data and assigns
    1D array offsets to each asset/unit_type based on the active backend.
    """
    backend = backend.lower()

    if backend == "tensor":
        tensor_unit_types = [
            "nuclear",
            "nuclear_lte",
            "aa_phes",
            "e_phes",
            "biomass",
            "biogas",
            "ocgt",
        ]
        unit_type_idx = {ut: idx for idx, ut in enumerate(tensor_unit_types, start=2)}
        details_length = 2 + len(tensor_unit_types)
        return unit_type_idx, details_length

    # --- Scalar Backend ---
    gen_types = sorted(list(set(g.unit_type for g in fleet.generators.values())))
    sto_types = sorted(list(set(s.unit_type for s in fleet.storages.values())))

    line_types_set = set(line.unit_type for line in network.major_lines.values())
    line_types_set.update(line.unit_type for line in network.minor_lines.values())
    line_types = sorted(list(line_types_set))

    unit_type_idx = {}
    current_idx = 2

    for t in gen_types:
        unit_type_idx[t] = current_idx
        current_idx += 2

    for t in sto_types:
        unit_type_idx[t] = current_idx
        current_idx += 1

    for t in line_types:
        unit_type_idx[t] = current_idx
        current_idx += 1

    details_length = current_idx

    for g in fleet.generators.values():
        g.unit_type_idx = unit_type_idx[g.unit_type]
    for s in fleet.storages.values():
        s.unit_type_idx = unit_type_idx[s.unit_type]
    for line in network.major_lines.values():
        line.unit_type_idx = unit_type_idx[line.unit_type]
    for line in network.minor_lines.values():
        line.unit_type_idx = unit_type_idx[line.unit_type]

    return unit_type_idx, details_length


def build_capacity_space_scaler(
    scenario,
    space_norm: str | None = None,
    space_agg: str | None = None,
) -> np.ndarray:
    space_norm = (space_norm or getattr(scenario.config, "mga_space_norm", scenario.config.parameterisation)).lower()
    space_agg = (space_agg or getattr(scenario.config, "mga_space_agg", "unit_type")).lower()
    param_mode = scenario.config.parameterisation.lower()

    n_vars = len(scenario.lower_bounds_abs)

    if param_mode == space_norm:
        row_scale = np.ones(n_vars, dtype=npfloat)
    elif param_mode == "relative" and space_norm == "absolute":
        row_scale = scenario.abs_rel_scaler.astype(npfloat)
    elif param_mode == "absolute" and space_norm == "relative":
        row_scale = (1.0 / scenario.abs_rel_scaler).astype(npfloat)
    else:
        raise ValueError(f"Unknown norm/parameterisation combination: {param_mode} -> {space_norm}")

    if space_agg == "none":
        return row_scale if not np.allclose(row_scale, 1.0) else np.ones(n_vars, dtype=npfloat)

    if space_agg == "unit_type":
        groups = defaultdict(list)
        for gen in scenario.fleet.generators.values():
            if gen.candidate_x_idx != -1:
                groups[gen.unit_type].append(gen.candidate_x_idx)
        for sto in scenario.fleet.storages.values():
            ut_key = "phes" if sto.unit_type in ("aa_phes", "e_phes") else sto.unit_type
            if sto.candidate_p_x_idx != -1 and sto.candidate_p_x_idx not in groups[f"{ut_key}_power"]:
                groups[f"{ut_key}_power"].append(sto.candidate_p_x_idx)
            if sto.candidate_e_x_idx != -1 and sto.candidate_e_x_idx not in groups[f"{ut_key}_energy"]:
                groups[f"{ut_key}_energy"].append(sto.candidate_e_x_idx)
        for line in scenario.network.major_lines.values():
            if line.candidate_x_idx != -1:
                groups[line.unit_type].append(line.candidate_x_idx)

        k_dims = len(groups)
        proj = np.zeros((n_vars, k_dims), dtype=npfloat)
        for k, (_, indices) in enumerate(groups.items()):
            weight = (1.0 / len(indices)) if space_norm == "relative" else 1.0
            proj[indices, k] = row_scale[indices] * weight

        scenario.projection_groups = groups
        return proj

    if space_agg == "node":
        n_nodes = len(scenario.network.nodes)
        proj = np.zeros((n_vars, n_nodes), dtype=npfloat)
        node_var_counts = np.zeros(n_nodes, dtype=npfloat)
        seen_sto_idx = set()

        for gen in scenario.fleet.generators.values():
            if gen.candidate_x_idx != -1:
                n = gen.node.order
                proj[gen.candidate_x_idx, n] = row_scale[gen.candidate_x_idx]
                node_var_counts[n] += 1.0

        for sto in scenario.fleet.storages.values():
            n = sto.node.order
            if sto.candidate_p_x_idx != -1 and sto.candidate_p_x_idx not in seen_sto_idx:
                proj[sto.candidate_p_x_idx, n] = row_scale[sto.candidate_p_x_idx]
                node_var_counts[n] += 1.0
                seen_sto_idx.add(sto.candidate_p_x_idx)
            if sto.candidate_e_x_idx != -1 and sto.candidate_e_x_idx not in seen_sto_idx:
                proj[sto.candidate_e_x_idx, n] = row_scale[sto.candidate_e_x_idx]
                node_var_counts[n] += 1.0
                seen_sto_idx.add(sto.candidate_e_x_idx)

        for line in scenario.network.major_lines.values():
            if line.candidate_x_idx != -1:
                u, v = line.node_start.order, line.node_end.order
                proj[line.candidate_x_idx, u] = 0.5 * row_scale[line.candidate_x_idx]
                proj[line.candidate_x_idx, v] = 0.5 * row_scale[line.candidate_x_idx]
                node_var_counts[u] += 0.5
                node_var_counts[v] += 0.5

        if space_norm == "relative":
            for n in range(n_nodes):
                if node_var_counts[n] > 0:
                    proj[:, n] /= node_var_counts[n]
        return proj

    raise ValueError(f"Unknown mga_space_agg: '{space_agg}'")


def get_tech_index_map(fleet, network) -> dict:
    """Maps technology categories to their specific column indices in the x vector."""
    tech_map = defaultdict(list)

    for gen in fleet.generators.values():
        if gen.candidate_x_idx != -1:
            tech = identify_tech(gen.name)
            tech_map[tech].append(gen.candidate_x_idx)

    for sto in fleet.storages.values():
        tech = identify_tech(sto.name)
        if sto.candidate_p_x_idx != -1 and sto.candidate_p_x_idx not in tech_map[f"{tech} Power (GW)"]:
            tech_map[f"{tech} Power (GW)"].append(sto.candidate_p_x_idx)
        if sto.candidate_e_x_idx != -1 and sto.candidate_e_x_idx not in tech_map[f"{tech} Energy (GWh)"]:
            tech_map[f"{tech} Energy (GWh)"].append(sto.candidate_e_x_idx)

    for line in network.major_lines.values():
        if line.candidate_x_idx != -1:
            tech_map["Transmission (GW)"].append(line.candidate_x_idx)

    return dict(tech_map)
