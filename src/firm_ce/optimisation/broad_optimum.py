# type: ignore
import csv
import os
from typing import Dict, List

import numpy as np

from firm_ce.common.constants import PENALTY_MULTIPLIER
from firm_ce.common.typing import (
    BandCandidates_Type,
    BroadOptimumVars_Type,
    EvaluationRecord_Type,
)
from firm_ce.optimisation.single_time import Solution, parallel_wrapper
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


def near_optimum_path(root: str, scenario_name: str):
    base = os.path.join("results", root, scenario_name)
    os.makedirs(base, exist_ok=True)
    return base


def create_broad_optimum_vars_record(
    candidate_x_idx: int, asset_name: str, near_optimum_check: bool, near_optimum_group: str
) -> BroadOptimumVars_Type:
    return (
        candidate_x_idx,
        asset_name,
        near_optimum_check,
        near_optimum_group,
    )


def build_broad_optimum_var_info(
    fleet: Fleet_InstanceType, network: Network_InstanceType
) -> List[BroadOptimumVars_Type]:
    """create a list of records mapping each decision variable index to:
    - its name
    - near_optimum on or off
    - its group key (to aggregate)"""

    broad_optimum_var_info = []

    for generator in fleet.generators.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                generator.candidate_x_idx, generator.name, generator.near_optimum_check, generator.group
            )
        )

    for storage in fleet.storages.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_p_x_idx, storage.name, storage.near_optimum_check, storage.group
            )
        )

    for storage in fleet.storages.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_e_x_idx, storage.name, storage.near_optimum_check, storage.group
            )
        )

    for line in network.major_lines.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(line.candidate_x_idx, line.name, line.near_optimum_check, line.group)
        )
    return broad_optimum_var_info


def create_evaluation_record(
    group_key: str,
    band_type: str,
    population_lcoes: List[float],
    de_population_penalties: List[float],
    band_population_penalties: List[float],
    band_population_candidates: List[List[float]],
    solution_index: int,
    target_group_var_sum: float | None = None,
) -> EvaluationRecord_Type:
    return (
        group_key,
        band_type,
        target_group_var_sum if target_group_var_sum else "N/A",
        float(population_lcoes[solution_index]),
        float(de_population_penalties[solution_index]),
        float(band_population_penalties[solution_index]),
        band_population_candidates[:, solution_index].copy(),
    )


def broad_optimum_objective(
    band_population_candidates: List[List[float]],  # 2-D array to allow vectorized DE
    differential_evolution_args,
    group_key: str,
    band_lcoe_max: float,
    bo_group_orders: List[int],
    evaluation_records: List[EvaluationRecord_Type],
    band_type: str,
    target_group_var_sum: float | None = None,
    midpoint_number: int | None = None,
) -> float:

    _, population_lcoes, de_population_penalties = parallel_wrapper(
        band_population_candidates, *differential_evolution_args
    )
    group_var_sums = band_population_candidates[bo_group_orders, :].sum(axis=0)
    band_population_penalties = np.maximum(0, population_lcoes - band_lcoe_max) * PENALTY_MULTIPLIER

    match band_type:
        case "min" | "max":
            for candidate_x in range(band_population_candidates.shape[1]):
                if not (
                    de_population_penalties[candidate_x] <= 0.001 and band_population_penalties[candidate_x] <= 0.001
                ):
                    continue
                evaluation_records.append(
                    create_evaluation_record(
                        group_key,
                        band_type,
                        population_lcoes,
                        de_population_penalties,
                        band_population_penalties,
                        band_population_candidates,
                        candidate_x,
                    )
                )
        case "midpoint":
            target_penalties = np.abs(group_var_sums - target_group_var_sum)
            for candidate_x in range(band_population_candidates.shape[1]):
                if not (
                    de_population_penalties[candidate_x] <= 0.001 and band_population_penalties[candidate_x] <= 0.001
                ):
                    continue
                evaluation_records.append(
                    create_evaluation_record(
                        group_key,
                        f"Midpoint {midpoint_number}",
                        population_lcoes,
                        de_population_penalties,
                        band_population_penalties,
                        band_population_candidates,
                        candidate_x,
                        target_group_var_sum,
                    )
                )

    match band_type:
        case "min":
            return band_population_penalties + de_population_penalties + group_var_sums
        case "max":
            return band_population_penalties + de_population_penalties - group_var_sums
        case "midpoint":
            return band_population_penalties + de_population_penalties + target_penalties
        case _:
            return None


def write_broad_optimum_records(
    scenario_name: str,
    evaluation_records: List[EvaluationRecord_Type],
    broad_optimum_var_info: List[BroadOptimumVars_Type],
) -> None:
    space_dir = near_optimum_path("near_optimum", scenario_name)

    space_path = os.path.join(space_dir, "near_optimal_space.csv")
    with open(space_path, "w", newline="") as f_space:
        writer_space = csv.writer(f_space)
        writer_space.writerow(
            [
                "Group",
                "Band_Type",
                "LCOE [$/MWh]",
                "Operational_Penalty",
                "Band_Penalty",
                *[f"{asset_name}" for _, asset_name, _, _ in broad_optimum_var_info],
            ]
        )
        for group, band_type, _, lcoe, de_penalty, band_penalty, candidate_x in evaluation_records:
            writer_space.writerow([group, band_type, lcoe, de_penalty, band_penalty, *candidate_x])
    return None


def get_broad_optimum_bands_path(scenario_name: str) -> str:
    space_dir = near_optimum_path("near_optimum", scenario_name)
    return os.path.join(space_dir, "near_optimal_bands.csv")


def write_broad_optimum_bands(
    scenario_name: str,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
    bands: BandCandidates_Type,
    de_args,
    band_lcoe_max: float,
    groups: Dict[str, List[int]],
) -> None:
    bands_path = get_broad_optimum_bands_path(scenario_name)
    near_optimal_asset_names = [
        asset_name for _, asset_name, near_optimum_check, _ in broad_optimum_var_info if near_optimum_check
    ]
    names_to_columns = {name: col for col, name in enumerate(near_optimal_asset_names)}

    with open(bands_path, "w", newline="") as f_bands:
        writer_bands = csv.writer(f_bands)
        header = [
            "Group",
            "Band_Type",
            "LCOE [$/MWh]",
            "Operational_Penalty",
            "Band_Penalty",
        ] + near_optimal_asset_names
        writer_bands.writerow(header)

        for group, (candidate_x_min, candidate_x_max) in bands.items():
            for band_type, candidate_x in (("min", candidate_x_min), ("max", candidate_x_max)):
                solution = Solution(candidate_x, *de_args)
                solution.evaluate()
                band_penalty = max(0, solution.lcoe - band_lcoe_max) * PENALTY_MULTIPLIER
                row = [group, band_type, solution.lcoe, solution.penalties, band_penalty]
                vals = [""] * len(near_optimal_asset_names)
                for candidate_x_idx in groups[group]:
                    _, asset_name, _, _ = broad_optimum_var_info[candidate_x_idx]
                    col = names_to_columns[asset_name]
                    vals[col] = candidate_x[candidate_x_idx]
                writer_bands.writerow(row + vals)
    return None


def read_broad_optimum_bands(
    scenario_name: str,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
) -> None:
    bands_path = get_broad_optimum_bands_path(scenario_name)

    group_names = {}
    group_bands = {}
    for _, name, near_optimum_check, group in broad_optimum_var_info:
        if not near_optimum_check:
            continue
        group_names.setdefault(group, []).append(name)

    with open(bands_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["Group"]
            band_type = row["Band_Type"]
            values = [float(row[variable_name]) for variable_name in group_names[group]]
            s = sum(values)
            group_bands.setdefault(group, {"min": None, "max": None})[band_type] = s
    return group_bands


def get_midpoint_csv_path(scenario_name: str) -> str:
    midpoint_dir = near_optimum_path("midpoint_explore", scenario_name)
    return os.path.join(midpoint_dir, "midpoint_space.csv")


def create_midpoint_csv(scenario_name: str, broad_optimum_var_info: List[BroadOptimumVars_Type]) -> str:
    csv_path = get_midpoint_csv_path(scenario_name)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Group",
                "Midpoint ID",
                "Target Value [GW or GWh]",
                "LCOE [$/MWh]",
                "Operational_Penalty",
                "Band_Penalty",
                *[f"{asset_name}" for _, asset_name, _, _ in broad_optimum_var_info],
            ]
        )
    return csv_path


def create_groups_dict(broad_optimum_var_info):
    groups = {}
    for record in broad_optimum_var_info:
        candidate_x_idx, _, near_optimum_check, group = record

        if not near_optimum_check:
            continue
        key = group or candidate_x_idx
        groups.setdefault(key, []).append(candidate_x_idx)
    return groups


def append_to_midpoint_csv(
    scenario_name: str,
    evaluation_records: List[EvaluationRecord_Type],
):
    csv_path = get_midpoint_csv_path(scenario_name)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        for group, band_type, target_value, lcoe, de_penalty, band_penalty, candidate_x in evaluation_records:
            writer.writerow([group, band_type, target_value, lcoe, de_penalty, band_penalty, *candidate_x])
    return None
