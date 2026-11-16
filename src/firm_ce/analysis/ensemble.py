# type: ignore
import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from firm_ce.common.constants import TOLERANCE
from firm_ce.common.typing import npfloat
from firm_ce.system.scenario import Scenario
from firm_ce.system.taxonomy import TechTaxonomy


def _pareto_frontier_2d(points: np.ndarray, maximize_x: bool = True, maximize_y: bool = True) -> np.ndarray:
    """
    Fast O(M log M) 2D Pareto frontier mask for M unique 2D grid cells.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    pts = points.copy()
    if maximize_x:
        pts[:, 0] = -pts[:, 0]
    if maximize_y:
        pts[:, 1] = -pts[:, 1]

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    mask = np.zeros(len(points), dtype=bool)
    min_y = np.inf
    for idx in order:
        if pts[idx, 1] < min_y:
            mask[idx] = True
            min_y = pts[idx, 1]
    return mask


class EnsembleDataset:
    """
    Chunked, memory-safe loader and projector for MHMGA historical populations
    (`population.bin` / `.csv`) and operational details (`details.bin` / `.csv`).
    """

    def __init__(
        self,
        scenario: Scenario,
        max_relaxation: float | None = 1.25,
        precision: int | None = 1,
        include_details: bool = True,
        chunk_size: int = 250_000,
    ):
        self.scenario = scenario
        if not scenario.data_status:
            scenario.load_datafiles()

        self.taxonomy = TechTaxonomy(scenario)
        self.precision = precision

        self.df_abs, self.df_rel, self.opt_lcoe = self._build_projected_frames(
            max_relaxation=max_relaxation,
            include_details=include_details,
            chunk_size=chunk_size,
        )

        if precision is not None:
            self.df_abs = self.df_abs.round(precision)
            self.df_rel = self.df_rel.round(precision)

    def _build_projected_frames(
        self, max_relaxation: float | None, include_details: bool, chunk_size: int = 250_000
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        sol_dir = self.scenario.solution_dir
        n_vars = len(self.scenario.lower_bounds_abs)
        pop_cols = n_vars + 3

        bin_path = os.path.join(sol_dir, "population.bin")
        csv_path = os.path.join(sol_dir, "population.csv")
        param_mode = "rel" if self.scenario.config.parameterisation == "relative" else "abs"

        if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
            n_rows = os.path.getsize(bin_path) // (np.dtype(npfloat).itemsize * pop_cols)
            pop_mmap = np.memmap(bin_path, dtype=npfloat, mode="r", shape=(n_rows, pop_cols))
            feas_costs = pop_mmap[:, 0][pop_mmap[:, 1] <= TOLERANCE]
            if len(feas_costs) == 0:
                raise RuntimeError("No feasible solutions found in population.bin.")
            opt_lcoe = float(feas_costs.min())
            max_cost = opt_lcoe * max_relaxation if max_relaxation is not None else np.inf

            det_mmap = None
            det_bin = os.path.join(sol_dir, "details.bin")
            if include_details and os.path.exists(det_bin) and os.path.getsize(det_bin) > 0:
                det_cols = self.scenario.details_length
                if os.path.getsize(det_bin) // (np.dtype(npfloat).itemsize * det_cols) == n_rows:
                    det_mmap = np.memmap(det_bin, dtype=npfloat, mode="r", shape=(n_rows, det_cols))

            abs_chunks, rel_chunks = [], []
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                chunk = pop_mmap[start:end]
                mask = (chunk[:, 1] <= TOLERANCE) & (chunk[:, 0] <= max_cost)
                if not np.any(mask):
                    continue
                valid = np.array(chunk[mask], dtype=npfloat)
                df_a = self.taxonomy.project(valid[:, 3:], valid[:, :3], input_space=param_mode, output_space="abs")
                df_r = self.taxonomy.project(valid[:, 3:], valid[:, :3], input_space=param_mode, output_space="rel")

                if det_mmap is not None:
                    d_valid = np.array(det_mmap[start:end][mask], dtype=npfloat)
                    for ut_name, col_idx in self.scenario.unit_type_idx.items():
                        if col_idx < d_valid.shape[1]:
                            df_a[f"gen_{ut_name}"] = d_valid[:, col_idx]
                            df_r[f"gen_{ut_name}"] = d_valid[:, col_idx] / self.scenario.static.mean_annual_demand_mwh

                abs_chunks.append(df_a)
                rel_chunks.append(df_r)

            return pd.concat(abs_chunks, ignore_index=True), pd.concat(rel_chunks, ignore_index=True), opt_lcoe

        if os.path.exists(csv_path):
            opt_lcoe = np.inf
            for chunk_df in pd.read_csv(csv_path, header=None, usecols=[0, 1], dtype=npfloat, chunksize=chunk_size):
                c = chunk_df.to_numpy()
                feas = c[c[:, 1] <= TOLERANCE, 0]
                if len(feas) > 0:
                    opt_lcoe = min(opt_lcoe, float(feas.min()))
            if not np.isfinite(opt_lcoe):
                raise RuntimeError("No feasible solutions found in population.csv.")
            max_cost = opt_lcoe * max_relaxation if max_relaxation is not None else np.inf

            abs_chunks, rel_chunks = [], []
            for chunk_df in pd.read_csv(csv_path, header=None, dtype=npfloat, chunksize=chunk_size):
                chunk = chunk_df.to_numpy()
                mask = (chunk[:, 1] <= TOLERANCE) & (chunk[:, 0] <= max_cost)
                if not np.any(mask):
                    continue
                valid = chunk[mask]
                abs_chunks.append(
                    self.taxonomy.project(valid[:, 3:], valid[:, :3], input_space=param_mode, output_space="abs")
                )
                rel_chunks.append(
                    self.taxonomy.project(valid[:, 3:], valid[:, :3], input_space=param_mode, output_space="rel")
                )
            return pd.concat(abs_chunks, ignore_index=True), pd.concat(rel_chunks, ignore_index=True), opt_lcoe

        raise FileNotFoundError(f"Neither population.bin nor population.csv found in '{sol_dir}'.")

    def get_frame(self, space: Literal["abs", "rel"] = "rel", threshold: float | None = None) -> pd.DataFrame:
        df = self.df_rel if space.startswith("rel") else self.df_abs
        if threshold is not None:
            return df[df["cost"] <= self.opt_lcoe * threshold]
        return df
