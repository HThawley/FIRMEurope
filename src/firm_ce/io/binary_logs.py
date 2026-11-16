# type: ignore
import csv
import json
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import PENALTY_MULTIPLIER, SAVE_POPULATION
from firm_ce.common.typing import npfloat

__all__ = [
    "convert_binary_logs_to_csv",
    "recover_interrupted_run",
    "write_mhmga_config_summary",
    "log_population_and_details",
]


def convert_binary_logs_to_csv(
    out_dir: str,
    pop_cols: Optional[int] = None,
    details_cols: Optional[int] = None,
    chunk_rows: int = 50_000,
    dtype=npfloat,
) -> None:
    """
    Converts binary optimisation logs (`population.bin`, `latest_population.npy`,
    `details.bin`, `latest_details.npy`) in `out_dir` to `.csv` files for downstream
    processes.
    """
    dtype = np.dtype(dtype)

    latest_pop_npy = os.path.join(out_dir, "latest_population.npy")
    if os.path.exists(latest_pop_npy):
        arr = np.load(latest_pop_npy)
        if pop_cols is None:
            pop_cols = arr.shape[1]
        np.savetxt(os.path.join(out_dir, "latest_population.csv"), arr, delimiter=",", fmt="%.7g")

    latest_det_npy = os.path.join(out_dir, "latest_details.npy")
    if os.path.exists(latest_det_npy):
        arr = np.load(latest_det_npy)
        if details_cols is None:
            details_cols = arr.shape[1]
        np.savetxt(os.path.join(out_dir, "latest_details.csv"), arr, delimiter=",", fmt="%.7g")

    config_json = os.path.join(out_dir, "mhmga_config.json")
    if (pop_cols is None or details_cols is None) and os.path.exists(config_json):
        with open(config_json, "r") as f:
            meta = json.load(f)
        if pop_cols is None and "pop_cols" in meta:
            pop_cols = meta["pop_cols"]
        elif pop_cols is None and "ndim" in meta:
            pop_cols = meta["ndim"] + 3
        if details_cols is None and "details_cols" in meta:
            details_cols = meta["details_cols"]

    def _stream_bin_to_csv(bin_path: str, csv_path: str, n_cols: Optional[int]) -> None:
        if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
            return
        if n_cols is None:
            raise ValueError(
                f"Cannot infer column count for '{bin_path}'. Please pass `pop_cols` / `details_cols` explicitly."
            )
        total_floats = os.path.getsize(bin_path) // dtype.itemsize
        n_rows = total_floats // n_cols
        if n_rows == 0:
            return

        mmap_arr = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_rows, n_cols))
        with open(csv_path, "w", newline="") as f_csv:
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                np.savetxt(f_csv, mmap_arr[start:end], delimiter=",", fmt="%.7g")

    _stream_bin_to_csv(
        os.path.join(out_dir, "population.bin"),
        os.path.join(out_dir, "population.csv"),
        pop_cols,
    )
    _stream_bin_to_csv(
        os.path.join(out_dir, "details.bin"),
        os.path.join(out_dir, "details.csv"),
        details_cols,
    )


def recover_interrupted_run(
    out_dir: str,
    dtype=npfloat,
) -> Optional[NDArray[npfloat]]:
    """
    Cleans up and finalises logging files in `out_dir` after an aborted/interrupted
    optimisation run:
      1. Truncates any torn trailing bytes in `population.bin` and `details.bin`.
      2. Converts `.bin` and `latest_*.npy` files to `.csv`.
      3. Reconstructs `current_optimum.csv` and returns the best feasible decision vector `x`.
    """
    dtype = np.dtype(dtype)
    config_json = os.path.join(out_dir, "mhmga_config.json")

    pop_cols = None
    details_cols = None
    if os.path.exists(config_json):
        with open(config_json, "r") as f:
            meta = json.load(f)
        pop_cols = meta.get("pop_cols", meta.get("ndim", 0) + 3 or None)
        details_cols = meta.get("details_cols", None)

    latest_pop_npy = os.path.join(out_dir, "latest_population.npy")
    if pop_cols is None and os.path.exists(latest_pop_npy):
        pop_cols = np.load(latest_pop_npy, mmap_mode="r").shape[1]

    for fname, n_cols in (("population.bin", pop_cols), ("details.bin", details_cols)):
        bin_path = os.path.join(out_dir, fname)
        if os.path.exists(bin_path) and n_cols is not None:
            row_bytes = n_cols * dtype.itemsize
            file_size = os.path.getsize(bin_path)
            remainder = file_size % row_bytes
            if remainder != 0:
                valid_size = file_size - remainder
                with open(bin_path, "r+b") as f_bin:
                    f_bin.truncate(valid_size)
                print(f"[Recover] Truncated {remainder} corrupted trailing bytes from '{fname}'.")

    convert_binary_logs_to_csv(
        out_dir=out_dir,
        pop_cols=pop_cols,
        details_cols=details_cols,
        dtype=dtype,
    )

    best_x = None
    pop_bin_path = os.path.join(out_dir, "population.bin")
    if os.path.exists(pop_bin_path) and os.path.getsize(pop_bin_path) > 0 and pop_cols is not None:
        total_rows = os.path.getsize(pop_bin_path) // (pop_cols * dtype.itemsize)
        mmap_pop = np.memmap(pop_bin_path, dtype=dtype, mode="r", shape=(total_rows, pop_cols))

        if os.path.exists(config_json):
            feasible_mask = mmap_pop[:, 1] == 0
            if np.any(feasible_mask):
                feasible_indices = np.where(feasible_mask)[0]
                best_idx = feasible_indices[np.argmin(mmap_pop[feasible_mask, 0])]
            else:
                best_idx = int(np.argmin(mmap_pop[:, 0] + mmap_pop[:, 1] * PENALTY_MULTIPLIER))
            best_row = np.array(mmap_pop[best_idx], dtype=dtype)
            best_x = best_row[3:]

            with open(os.path.join(out_dir, "current_optimum.csv"), "w", newline="") as f:
                csv.writer(f).writerow([best_row[0], best_row[1], *best_x])
        else:
            best_idx = int(np.argmin(mmap_pop[:, -1]))
            best_row = np.array(mmap_pop[best_idx], dtype=dtype)
            best_x = best_row[:-1]

    print(f"[Recover] Log recovery and CSV conversion complete for '{out_dir}'.")
    return best_x


def write_mhmga_config_summary(scenario, config, details: Optional[np.ndarray] = None) -> None:
    """Writes a JSON summary of MHMGA hyperparameters to the solution directory."""
    summary_path = os.path.join(scenario.solution_dir, "mhmga_config.json")

    mhmga_config = {}
    for key, value in config.__dict__.items():
        if key.startswith("mga_"):
            if isinstance(value, np.ndarray):
                mhmga_config[key] = value.tolist()
            else:
                mhmga_config[key] = value

    ndim = len(scenario.lower_bounds)
    mhmga_config["parameterisation"] = config.parameterisation
    mhmga_config["ndim"] = ndim
    mhmga_config["pop_cols"] = ndim + 3
    if getattr(config, "save_details", False) and details is not None:
        mhmga_config["details_cols"] = int(details.shape[-1])
    mhmga_config["lower_bounds"] = scenario.lower_bounds.tolist()
    mhmga_config["upper_bounds"] = scenario.upper_bounds.tolist()
    mhmga_config["unit_type_idxs"] = scenario.unit_type_idx
    mhmga_config["mga_space_basis"] = getattr(config, "mga_space_basis", "capacity")
    mhmga_config["mga_space_norm"] = getattr(config, "mga_space_norm", config.parameterisation)
    mhmga_config["mga_space_agg"] = getattr(config, "mga_space_agg", "unit_type")

    with open(summary_path, "w") as f:
        json.dump(mhmga_config, f, indent=4)


def log_population_and_details(out_dir: str, population, save_details: bool = False, details: Optional[np.ndarray] = None) -> None:
    """Fast binary logging of MHMGA population and details arrays."""
    if not SAVE_POPULATION:
        return

    obj_flat = population.raw_objectives.reshape(-1)
    viol_flat = population.violations.reshape(-1)
    fit_flat = population.fitnesses.reshape(-1)
    pts_flat = population.points.reshape(-1, population.points.shape[-1])

    combined_block = np.column_stack((obj_flat, viol_flat, fit_flat, pts_flat)).astype(npfloat, copy=False)

    with open(os.path.join(out_dir, "population.bin"), "ab") as f_all:
        combined_block.tofile(f_all)
    np.save(os.path.join(out_dir, "latest_population.npy"), combined_block)

    if save_details and details is not None:
        num_niches, pop_size, _ = population.points.shape
        details_flat = np.ascontiguousarray(
            details[:num_niches, :pop_size].reshape(num_niches * pop_size, -1),
            dtype=npfloat,
        )
        with open(os.path.join(out_dir, "details.bin"), "ab") as f_all:
            details_flat.tofile(f_all)
        np.save(os.path.join(out_dir, "latest_details.npy"), details_flat)
