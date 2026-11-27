"""
An example script for loading and evaluating FIRM solutions (single-solution or MHMGA near-optimal
alternatives) and exporting them into the CSV table format required by the third-party `mga4all` tool.

The exported folder contains:
  - `node_coordinates.csv`
  - `tech_parameters.csv`
  - `capacity_values.csv`
  - `flow_timeseries_alternative_{i}.csv` (one per alternative)
"""

import os
import time
import numpy as np
import pandas as pd

from firm_ce.model import Model
from firm_ce.common.typing import npfloat
from firm_ce.analysis.mga4all import MGA4AllConfig, MGA4AllExporter, export_to_mga4all


def resolve_single_solution_x(
    scenario,
    run_mode: str,
    x_loc: str | None = None,
    x0_fallback: bool = True,
) -> np.ndarray:
    """Resolves the decision vector `x` for a single_time scenario using a fallback chain."""
    sol_dir = scenario.solution_dir

    if run_mode == "new":
        if scenario.x0.size == 0:
            raise ValueError(f"No initial guess (x0) provided for scenario '{scenario.name}'.")
        scenario.create_solution_directory()
        return scenario.x0.astype(npfloat)

    if x_loc == "latest_population":
        npy_path = os.path.join(sol_dir, "latest_population.npy")
        csv_path = os.path.join(sol_dir, "latest_population.csv")
        if os.path.exists(npy_path):
            return np.load(npy_path)[0, 3:].astype(npfloat)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path, header=None, nrows=1).to_numpy()[0, 3:].astype(npfloat)

    curr_opt = os.path.join(sol_dir, "current_optimum.csv")
    if os.path.exists(curr_opt):
        return pd.read_csv(curr_opt, header=None).to_numpy().flatten()[2:].astype(npfloat)

    for stat_name, is_rel in (("x_rel", True), ("x_abs", False)):
        stat_path = os.path.join(sol_dir, "statistics", f"{stat_name}.csv")
        if os.path.exists(stat_path):
            x = pd.read_csv(stat_path, header=None).to_numpy().flatten().astype(npfloat)
            if scenario.config.parameterisation == "relative" and not is_rel:
                return scenario.convert_x_to_rel(x)
            if scenario.config.parameterisation == "absolute" and is_rel:
                return scenario.convert_x_to_abs(x)
            return x

    if x0_fallback and scenario.x0.size > 0:
        return scenario.x0.astype(npfloat)

    raise FileNotFoundError(f"Could not locate any solved decision vector in '{sol_dir}'.")


def resolve_mhmga_noptima_xs(scenario) -> np.ndarray:
    """Resolves the near-optimal alternative vectors `xs` from mga_logs."""
    mga_dir = os.path.join(scenario.solution_dir, "mga_logs")
    ndim = len(scenario.lower_bounds)

    for fname in ("mga_alternatives.csv", "mga-noptima.csv"):
        fpath = os.path.join(mga_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if len(df) > 0:
                xs = df.iloc[:, -ndim:].to_numpy(dtype=npfloat)
                valid_mask = np.all(np.isfinite(xs), axis=1)
                if np.any(valid_mask):
                    return xs[valid_mask]

    evo_path = os.path.join(mga_dir, "mga-evolution.csv")
    if os.path.exists(evo_path):
        df = pd.read_csv(evo_path)
        df = df[df["iter"] == df["iter"].max()]
        xs = df.iloc[:, -ndim:].to_numpy(dtype=npfloat)
        valid_mask = np.all(np.isfinite(xs), axis=1)
        if np.any(valid_mask):
            return xs[valid_mask]

    raise FileNotFoundError(f"Could not find valid MHMGA alternatives in '{mga_dir}'.")


def load_and_evaluate_solutions(
    scenario,
    run_mode: str,
    mhmga_indices: list[int] | None = None,
    deduplicate_optimum: bool = True,
):
    """Loads decision vectors and evaluates the scenario solution(s)."""
    if not scenario.data_status:
        scenario.load_datafiles()

    if scenario.config.type == "single_time":
        x = resolve_single_solution_x(scenario, run_mode)
        scenario.build_and_evaluate_solution(x)
    elif scenario.config.type.startswith("mhmga"):
        xs = resolve_mhmga_noptima_xs(scenario)
        if mhmga_indices is not None:
            xs = xs[mhmga_indices]
        scenario.build_and_evaluate_noptima(xs)

        if deduplicate_optimum and len(scenario.noptima) > 1:
            lcoes = [
                n.lcoe for n in (scenario.noptimaTensors if scenario.noptimaTensors is not None else scenario.noptima)
            ]
            opt_lcoe = lcoes[0]
            dup_mask = np.isclose(lcoes, opt_lcoe, rtol=1e-6)
            # Keep the first optimum (index 0) and filter out subsequent duplicate optimum entries
            dup_mask[0] = False
            scenario.noptima = [n for n, dup in zip(scenario.noptima, dup_mask) if not dup]
            if scenario.noptimaTensors is not None:
                scenario.noptimaTensors = [n for n, dup in zip(scenario.noptimaTensors, dup_mask) if not dup]


if __name__ == "__main__":
    RUN_MODE = r"results\firmeur_5yr_explor_20261023_155451"
    SCENARIOS = ("7percent",)

    # Configure mga4all export parameters:
    # - group_by: "subtech" (solar_pv, wind_onshore, wind_offshore, bioenergy, battery, phes, ...),
    #             "unit_type" (pv_fixed, pv_track, onsw, offw, bess2h, bess4h, ...), or "asset"
    # - units: "MW" (capacities in MW, interval flows in MWh) or "GW" (capacities in GW, interval flows in GWh)
    # - year: "last" (final year of horizon), "first", specific integer year (e.g. 2016), or "all"
    # - merge_parallel_lines: False (keep parallel transmission lines separate) or True (merge per node pair)
    export_config = MGA4AllConfig(
        group_by="asset",
        units="MW",
        year="all",
        merge_parallel_lines=False,
    )

    start_time = time.time()
    model = Model(model_location=RUN_MODE)
    print(f"Model build time: {time.time() - start_time:.4f} seconds")

    for name in SCENARIOS:
        scenario = model.scenarios[name]
        scenario.load_datafiles()

        # Select alternatives to evaluate and export (e.g. first 7 niche alternatives)
        indices = list(range(7)) if scenario.config.type.startswith("mhmga") else None
        load_and_evaluate_solutions(scenario, RUN_MODE, mhmga_indices=indices)

        t0 = time.time()
        output_dir = os.path.join(scenario.solution_dir, "mga4all")
        written_files = export_to_mga4all(
            scenario,
            output_dir=output_dir,
            config=export_config,
        )
        print(f"Exported {len(written_files)} mga4all files to '{output_dir}' in {time.time() - t0:.2f}s:")
        for label, fpath in written_files.items():
            print(f"  - {label}: {fpath}")
