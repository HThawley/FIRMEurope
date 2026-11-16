"""
Method
------
Dispatch (Simulate) does not depend on any cost parameter, and CalculateCost is
LINEAR in the cost coefficients:

    total_cost = legacy + sum_k  cap_k[n] * F_k[n]   +   sum_j  energy_j[n] * V_j[n]

so each alternative is simulated ONCE to obtain its capacities and nodal energies.
That gives a per-alternative vector of baseline cost components (in $/yr), and every
Monte-Carlo draw is then just a multiplier vector m applied to those components:

    LCOE_s(draw) = (m . components_s) / mean_annual_demand_mwh
"""
from dataclasses import dataclass
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firm_ce.system.costs_common import apportion_new_phes
from firm_ce.viz.uncertainty_plots import (
    plot_lcoe_distributions,
    plot_pairwise_win_rates,
    plot_rank_probability,
)


@dataclass
class MCConfig:
    n_samples: int = 10_000
    rel_halfwidth: float = 0.10  # +/-10 % uniform on every cost parameter
    perturb_mode: str = "group"
    seed: int = 12345
    chunk_solutions: int = 32
    chunk_samples: int = 2_000
    max_top_k_ranks: int = 10
    max_heatmap_rows: int = 10
    max_pdf_curves: int = 40
    max_pairwise_alts: int = 10
    kde_subsample: int = 2_000
    drop_infeasible: bool = True
    dedupe: bool = True
    check_rtol_cost: float = 1e-4
    check_rtol_objective: float = 1e-3


def _summarise(sol) -> dict:
    a, E, s = sol.assets, sol.nodal_energy, sol.static
    c = s.costs
    nodes = s.nodes

    p_aa = np.zeros(nodes, dtype=np.float64)
    e_aa = np.zeros(nodes, dtype=np.float64)
    p_e = np.zeros(nodes, dtype=np.float64)
    e_e = np.zeros(nodes, dtype=np.float64)
    clphes_gen = np.zeros(nodes, dtype=np.float64)

    for n in range(nodes):
        pa, ea, pe, ee = apportion_new_phes(a.CnphP[n], a.CnphE[n], c.E_aa_max[n])
        p_aa[n], e_aa[n], p_e[n], e_e[n] = pa, ea, pe, ee
        if p_aa[n] > 0.0:
            clphes_gen[n] = (sol.nodal_aaph_gen[n] / p_aa[n]) * s.EstorageP[n, 1]
        elif p_e[n] > 0.0:
            clphes_gen[n] = (sol.nodal_eph_gen[n] / p_e[n]) * s.EstorageP[n, 1]

    return dict(
        Cpfix=a.Cpfix.copy(),
        Cpsat=a.Cpsat.copy(),
        Coffw=a.Coffw.copy(),
        Consw=a.Consw.copy(),
        Cpeak=a.Cpeak.copy(),
        Cnuke=a.Cnuke.copy(),
        Cnlte=a.Cnlte.copy(),
        Caa_phP=p_aa,
        Caa_phE=e_aa,
        Ce_phP=p_e,
        Ce_phE=e_e,
        CstorageP=a.CstorageP.copy(),
        Clines=a.Clines.copy(),
        E=E.copy(),
        aaph_gen=sol.nodal_aaph_gen.copy(),
        eph_gen=sol.nodal_eph_gen.copy(),
        clphes_gen=clphes_gen,
        feasible=bool(sol.feasible),
        penalties=float(sol.penalties),
        lcoe=float(sol.lcoe),
        total=float(sol.total_annual_cost),
    )


def cost_terms(q: dict, static) -> list[tuple[str, np.ndarray]]:
    """(parameter_name, per-element $/yr) for every cost parameter in CalculateCost."""
    c, E = static.costs, q["E"]

    def f(x):
        return np.asarray(x, dtype=np.float64)

    return [
        # ---- fixed (capacity x $/unit) ----
        ("F_pfix", f(q["Cpfix"]) * f(c.Fpfix)),
        ("F_psat", f(q["Cpsat"]) * f(c.Fpsat)),
        ("F_offw", f(q["Coffw"]) * f(c.Foffw)),
        ("F_onsw", f(q["Consw"]) * f(c.Fonsw)),
        ("F_biog", f(q["Cpeak"][:, 1]) * f(c.Fbiog)),
        ("F_biom", f(q["Cpeak"][:, 0]) * f(c.Fbiom)),
        ("F_gas", f(q["Cpeak"][:, 2]) * f(c.Fgas)),
        ("F_nuke", f(q["Cnuke"] - q["Cnlte"]) * f(c.Fnuke)),
        ("F_nlte", f(q["Cnlte"]) * f(c.Fnlte)),
        ("F_aa_phP", f(q["Caa_phP"]) * f(c.Faa_phP)),
        ("F_aa_phE", f(q["Caa_phE"]) * f(c.Faa_phE)),
        ("F_e_phP", f(q["Ce_phP"]) * f(c.Fe_phP)),
        ("F_e_phE", f(q["Ce_phE"]) * f(c.Fe_phE)),
        ("F_b4P", f(q["CstorageP"][:, 2]) * f(c.Fb4P)),
        ("F_b2P", f(q["CstorageP"][:, 3]) * f(c.Fb2P)),
        ("F_lines", f(q["Clines"]) * f(c.Flines)),
        # ---- variable (energy x $/unit) ----
        ("V_nuke", f(E[:, 4]) * f(c.Vnuke)),
        ("V_nlte", f(E[:, 5]) * f(c.Vnlte)),
        ("V_ph", f(q["aaph_gen"] + q["eph_gen"]) * f(c.Vph)),
        ("V_biom", f(E[:, 6]) * f(c.Vbiom)),
        ("V_biog", f(E[:, 7]) * f(c.Vbiog)),
        ("V_gas", f(E[:, 8]) * f(c.Vgas)),
        # ---- legacy fleet (lumped scalar in StaticTensor) ----
        ("legacy", np.array([float(static.legacy_costs)])),
    ]


def build_component_matrix(summaries: list[dict], static, cfg: MCConfig):
    """Returns comp (n_alt, P) in $/yr, group_idx (P,), group_names (G,), and validates against total_annual_cost."""
    rows, group_idx, group_names = [], None, None
    for q in summaries:
        terms = cost_terms(q, static)
        rows.append(np.concatenate([t[1] for t in terms]))
        if group_idx is None:
            group_names = [t[0] for t in terms]
            group_idx = np.concatenate([np.full(len(t[1]), g) for g, t in enumerate(terms)])
    comp = np.vstack(rows)

    totals = np.array([q["total"] for q in summaries])
    if not np.allclose(comp.sum(axis=1), totals, rtol=cfg.check_rtol_cost):
        worst = np.max(np.abs(comp.sum(axis=1) / totals - 1))
        raise RuntimeError(
            f"Cost decomposition does not reproduce CalculateCost (max rel err {worst:.2e}). "
            "CalculateCost has probably changed -- update cost_terms()."
        )
    return comp, group_idx, group_names


def monte_carlo(comp: np.ndarray, group_idx: np.ndarray, demand_mwh: float, cfg: MCConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    S, P = comp.shape
    G = int(group_idx.max()) + 1
    n, k, h = cfg.n_samples, cfg.max_top_k_ranks, cfg.rel_halfwidth
    k = min(k, S)

    L = np.empty((n, S), dtype=np.float32)  # $/MWh
    rank_counts = np.zeros((S, k), dtype=np.int64)
    ar = np.arange(S)

    for lo in range(0, n, cfg.chunk_samples):
        m = min(cfg.chunk_samples, n - lo)
        if cfg.perturb_mode == "group":
            M = rng.uniform(1 - h, 1 + h, size=(m, G))[:, group_idx]
        elif cfg.perturb_mode == "element":
            M = rng.uniform(1 - h, 1 + h, size=(m, P))
        else:
            raise ValueError(cfg.perturb_mode)

        Lc = (M @ comp.T) / demand_mwh  # (m, S)
        L[lo : lo + m] = Lc

        order = np.argsort(Lc, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, (m, S)), axis=1)
        capped = np.minimum(ranks, k)
        for r in range(k):
            rank_counts[:, r] += (capped == r).sum(axis=0)

    return dict(L=L, rank_counts=rank_counts)


def alt_labels(n: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ... (A = best baseline LCOE)."""
    out = []
    for i in range(n):
        s, j = "", i
        while True:
            s = chr(65 + j % 26) + s
            j = j // 26 - 1
            if j < 0:
                break
        out.append(s)
    return out


def run_monte_carlo(
    scenario,
    mhmga_indices: list = None,
    df: pd.DataFrame = None,
    cfg: MCConfig | None = None,
    plot: bool = True,
):
    """
    Runs the Monte-Carlo cost-uncertainty analysis on a list of pre-evaluated solutions.
    """
    cfg = cfg or MCConfig()
    static = scenario.staticTensor
    if mhmga_indices is None:
        mhmga_indices = list(range(len(scenario.noptimaTensors)))

    out_dir = os.path.join(scenario.solution_dir, "monte_carlo")
    os.makedirs(out_dir, exist_ok=True)

    summaries = [_summarise(scenario.noptimaTensors[m]) for m in mhmga_indices]

    if df is not None and "objective" in df.columns:
        sim_lcoe = np.array([q["lcoe"] for q in summaries])
        bad = ~np.isclose(sim_lcoe, df["objective"].to_numpy(float), rtol=cfg.check_rtol_objective)
        if bad.any():
            print(f"[check] WARNING: {bad.sum()} alternatives have a different LCOE than the csv objective.")

    keep = np.array([q["feasible"] for q in summaries])
    if cfg.drop_infeasible and not keep.all():
        print(f"[simulate] dropping {np.sum(~keep)} infeasible alternatives")
        if df is not None:
            df = df[keep].reset_index(drop=True)
        summaries = [q for q, k in zip(summaries, keep) if k]

    comp, group_idx, _ = build_component_matrix(summaries, static, cfg)
    demand = float(static.mean_annual_demand_mwh)
    base = comp.sum(axis=1) / demand

    if df is not None and "niche_id" in df.columns and (df["niche_id"].astype(str) == "optimum").any():
        opt_idx = int(np.flatnonzero(df["niche_id"].astype(str) == "optimum")[0])
    else:
        opt_idx = int(np.argmin(base))

    mc = monte_carlo(comp, group_idx, demand, cfg)
    L, rank_counts = mc["L"], mc["rank_counts"]
    np.save(os.path.join(out_dir, "mc_lcoe.npy"), L)

    order = np.argsort(base)
    labels = [""] * len(base)
    for lab, s in zip(alt_labels(len(base)), order):
        labels[s] = lab

    beats_opt = (L < L[:, [opt_idx]]).mean(axis=0)
    thr = None

    P = rank_counts / cfg.n_samples
    expected_rank = (np.arange(1, rank_counts.shape[1] + 1) * P).sum(axis=1)

    summary_dict = {
        "label": labels,
        "baseline_lcoe": base,
        "mc_mean": L.mean(axis=0),
        "mc_std": L.std(axis=0),
        "mc_p05": np.percentile(L, 5, axis=0),
        "mc_p95": np.percentile(L, 95, axis=0),
        "p_rank1": P[:, 0],
        "p_beats_optimum": beats_opt,
        "expected_rank_lb": expected_rank,
    }

    if df is not None and "niche_id" in df.columns:
        summary_dict["niche_id"] = df["niche_id"].astype(str)

    summary = pd.DataFrame(summary_dict).sort_values("baseline_lcoe")
    summary.to_csv(os.path.join(out_dir, "alternatives_summary.csv"), index=False)

    if plot:
        plot_lcoe_distributions(L, base, labels, opt_idx, thr, cfg, os.path.join(out_dir, "lcoe_distributions.png"))
        plot_rank_probability(
            rank_counts,
            cfg.n_samples,
            base,
            labels,
            opt_idx,
            beats_opt,
            cfg,
            os.path.join(out_dir, "rank_probability.png"),
        )
        plot_pairwise_win_rates(
            L,
            base,
            labels,
            cfg,
            os.path.join(out_dir, "pairwise_win_rates.png"),
        )
        plt.show()
    return summary
