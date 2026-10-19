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

i.e. one matrix product for the whole study, no re-simulation.

Common random numbers: the SAME multiplier draw is applied to every alternative.
This is what makes rank comparisons meaningful (differences between alternatives are
driven by their different technology mixes, not by independent noise).
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


@dataclass
class MCConfig:
    n_samples: int = 10_000
    rel_halfwidth: float = 0.10  # +/-10 % uniform on every cost parameter
    # "group":   one multiplier per cost parameter (e.g. Fpfix), shared across nodes.
    #            Physically sensible: "solar costs are 8 % high" applies everywhere.
    # "element": independent multiplier per node/line per parameter (much more
    #            averaging -> narrower LCOE spread, far fewer rank swaps).
    perturb_mode: str = "group"
    seed: int = 12345
    chunk_solutions: int = 32  # solutions simulated per parallel batch (memory)
    chunk_samples: int = 2_000  # MC draws per matrix product (memory)
    top_k_ranks: int = 10  # rank columns shown: 1st..kth, then "k+"
    max_heatmap_rows: int = 10  # alternatives shown in rank matrix
    max_pdf_curves: int = 40  # alternatives drawn as PDFs in fig 1
    max_pairwise_alts: int = 10  # number of top alternatives to include in the pairwise win matrix
    kde_subsample: int = 2_000
    drop_infeasible: bool = True
    dedupe: bool = True
    check_rtol_cost: float = 1e-4  # decomposition vs solution.total_annual_cost
    check_rtol_objective: float = 1e-3  # re-simulated LCOE vs csv "objective"


def _summarise(sol) -> dict:
    a, E = sol.assets, sol.nodal_energy
    return dict(
        Cpfix=a.Cpfix.copy(), Cpsat=a.Cpsat.copy(), Coffw=a.Coffw.copy(), Consw=a.Consw.copy(),
        Cpeak=a.Cpeak.copy(), Cnuke=a.Cnuke.copy(), Cnlte=a.Cnlte.copy(),
        CnphP=a.CnphP.copy(), CnphE=a.CnphE.copy(), CstorageP=a.CstorageP.copy(),
        Clines=a.Clines.copy(),
        E=E.copy(), nph_gen=sol.nodal_nph_gen.copy(),
        feasible=bool(sol.feasible), penalties=float(sol.penalties),
        lcoe=float(sol.lcoe), total=float(sol.total_annual_cost),
    )


def cost_terms(q: dict, static) -> list[tuple[str, np.ndarray]]:
    """(parameter_name, per-element $/yr) for every cost parameter in CalculateCost.
    Each parameter name is one 'group' for perturb_mode='group'."""
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
        ("F_phP", f(q["CnphP"]) * f(c.FphP)),
        ("F_phE", f(q["CnphE"]) * f(c.FphE)),
        ("F_b4P", f(q["CstorageP"][:, 1]) * f(c.Fb4P)),
        ("F_b2P", f(q["CstorageP"][:, 2]) * f(c.Fb2P)),
        ("F_lines", f(q["Clines"]) * f(c.Flines)),
        # ---- variable (energy x $/unit); VRE & battery VOM are zero in current model ----
        ("V_nuke", f(E[:, 4]) * f(c.Vnuke)),
        ("V_nlte", f(E[:, 5]) * f(c.Vnlte)),
        ("V_ph", f(q["nph_gen"]) * f(c.Vph)),
        ("V_biom", f(E[:, 6]) * f(c.Vbiom)),
        ("V_biog", f(E[:, 7]) * f(c.Vbiog)),
        ("V_gas", f(E[:, 8]) * f(c.Vgas)),
        # ---- legacy fleet (lumped scalar in StaticTensor) ----
        ("legacy", np.array([float(static.legacy_costs)])),
    ]


def build_component_matrix(summaries: list[dict], static, cfg: MCConfig):
    """Returns comp (n_alt, P) in $/yr, group_idx (P,), group_names (G,), and also
    validates the decomposition against solution.total_annual_cost."""
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
    n, k, h = cfg.n_samples, cfg.top_k_ranks, cfg.rel_halfwidth

    L = np.empty((n, S), dtype=np.float32)  # $/MWh
    rank_counts = np.zeros((S, k + 1), dtype=np.int64)  # last column = "k+"
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
        L[lo:lo + m] = Lc

        order = np.argsort(Lc, axis=1)  # rank 0 = cheapest
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, (m, S)), axis=1)
        capped = np.minimum(ranks, k)
        for r in range(k + 1):
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


def _ordinal(r: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(r, f"{r}th")


def plot_lcoe_distributions(L, base, labels, opt_idx, nopt_threshold, cfg, path):
    """Fig 1: number line of baseline LCOEs (bottom) + per-alternative MC PDFs (top)."""
    S = L.shape[1]
    order = np.argsort(base)
    pick = list(order[: cfg.max_pdf_curves])
    if opt_idx not in pick:
        pick.append(opt_idx)

    lo, hi = float(L.min()), float(L.max())
    pad = 0.03 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 600)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1], sharex=ax)
    cmap = plt.cm.viridis

    for i, s in enumerate(pick):
        d = L[:, s]
        if len(d) > cfg.kde_subsample:
            d = rng.choice(d, cfg.kde_subsample, replace=False)
        pdf = gaussian_kde(d)(grid)
        if s == opt_idx:
            ax.plot(grid, pdf, color="crimson", lw=2.2, zorder=5, label=f"{labels[s]} (base-case optimum)")
            ax.fill_between(grid, pdf, color="crimson", alpha=0.12, zorder=4)
        else:
            ax.plot(grid, pdf, color=cmap(i / max(len(pick) - 1, 1)), lw=0.9, alpha=0.55)

    if nopt_threshold is not None:
        for a in (ax, axn):
            a.axvline(nopt_threshold, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.text(nopt_threshold, ax.get_ylim()[1] * 0.5, " near-optimal limit", ha="left", va="center", fontsize=8, rotation=90)

    ax.set_ylabel("Probability density")
    ax.set_title(f"LCOE of near-optimal alternatives under ±{cfg.rel_halfwidth:.0%} cost uncertainty "
                 f"({cfg.n_samples:,} draws, {cfg.perturb_mode}-level)")
    ax.legend(loc="upper left", frameon=False)
    plt.setp(ax.get_xticklabels(), visible=False)

    # number line: baseline (deterministic) LCOE of every alternative
    axn.scatter(base, np.zeros(S), s=28, marker="|", color="0.25", zorder=3)
    axn.scatter(base[opt_idx], 0, s=70, marker="*", color="crimson", zorder=4)
    for s in order[:2]:   # label only the best two; the rest would overprint
        axn.annotate(labels[s], (base[s], 0), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=8)
    axn.axhline(0, color="0.5", lw=0.8)
    axn.set_ylim(-1, 1.5)
    axn.set_yticks([])
    for sp in ("left", "right", "top"):
        axn.spines[sp].set_visible(False)
    axn.set_xlabel("LCOE ($/MWh)   |   markers = baseline cost of each alternative")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_rank_probability(rank_counts, n_samples, base, labels, opt_idx, beats_opt, cfg, path):
    """Fig 2: rank-probability heatmap (rows = alternatives by baseline LCOE, cols = rank)."""
    S, K1 = rank_counts.shape
    P = rank_counts / n_samples
    order = np.argsort(base)
    rows = order[: cfg.max_heatmap_rows]
    M = P[rows]

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 2.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.5 else "black")

    ax.set_xticks(range(K1))
    ax.set_xticklabels([_ordinal(r + 1) for r in range(K1 - 1)] + [f"{K1}th+"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[s] for s in rows])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Resulting rank")
    ax.set_ylabel("Alternative solutions")
    ax.set_title(f"Rank probability matrix (top {len(rows)} of {S} by baseline LCOE)", pad=34)

    # headline: how robust is the base-case optimum, and who displaces it?
    beats = beats_opt.copy()
    beats[opt_idx] = 0
    rival = int(np.argmax(beats))
    msg = (f"Base-case optimum ({labels[opt_idx]}) is cheapest in {P[opt_idx, 0]:.0%} of draws; "
           f"most often displaced by {labels[rival]} ({beats[rival]:.0%} of draws).")
    fig.text(0.5, 0.01, msg, ha="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Probability", fraction=0.04, pad=0.03)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_pairwise_win_rates(L, base, labels, cfg, path):
    """Fig 3: Square heatmap showing the probability that Row Alternative is cheaper than Column Alternative."""
    order = np.argsort(base)
    pick = order[: cfg.max_pairwise_alts]
    N = len(pick)

    # Extract the simulated LCOEs just for the top N alternatives
    L_pick = L[:, pick]  # shape: (n_samples, N)

    # Broadcast comparison: M[i, j] = P(L_i < L_j)
    # L_pick[:, :, None] is (samples, rows, 1) and L_pick[:, None, :] is (samples, 1, cols)
    M = (L_pick[:, :, None] < L_pick[:, None, :]).mean(axis=0)

    # Mask the diagonal (comparing an alternative to itself)
    np.fill_diagonal(M, np.nan)

    fig, ax = plt.subplots(figsize=(0.7 * N + 2, 0.7 * N + 1))

    # Diverging colormap centered at 50% (Blue = Row wins often, Red = Row loses often)
    cmap = plt.cm.RdBu
    cmap.set_bad(color='white')
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(N):
        for j in range(N):
            if i != j:
                v = M[i, j]
                # High contrast text color based on cell darkness
                text_color = "white" if (v > 0.75 or v < 0.25) else "black"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color=text_color)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="gray")

    tick_labels = [labels[s] for s in pick]
    ax.set_xticks(range(N))
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(N))
    ax.set_yticklabels(tick_labels)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Opponent (Column)")
    ax.set_ylabel("Subject (Row)")
    ax.set_title(f"Pairwise Win Rates: P(Row cheaper than Column)\nTop {N} alternatives by baseline LCOE", pad=25)

    fig.colorbar(im, ax=ax, label="Row Win Probability", fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def run_monte_carlo(
    scenario,
    mhmga_indices: list = None,
    df: pd.DataFrame = None,
    cfg: MCConfig | None = None
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

    # --- extract summaries directly from pre-evaluated solutions ---
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

    # --- baseline cost components ------------------------------------------------
    comp, group_idx, group_names = build_component_matrix(summaries, static, cfg)
    demand = float(static.mean_annual_demand_mwh)
    base = comp.sum(axis=1) / demand  # $/MWh, baseline

    # reference "optimum" row
    if df is not None and "niche_id" in df.columns and (df["niche_id"].astype(str) == "optimum").any():
        opt_idx = int(np.flatnonzero(df["niche_id"].astype(str) == "optimum")[0])
    else:
        opt_idx = int(np.argmin(base))

    # --- monte-carlo -------------------------------------------------------------
    mc = monte_carlo(comp, group_idx, demand, cfg)
    L, rank_counts = mc["L"], mc["rank_counts"]
    np.save(os.path.join(out_dir, "mc_lcoe.npy"), L)

    # labels: A = lowest baseline LCOE
    order = np.argsort(base)
    labels = [""] * len(base)
    for lab, s in zip(alt_labels(len(base)), order):
        labels[s] = lab

    beats_opt = (L < L[:, [opt_idx]]).mean(axis=0)           # P(alt cheaper than optimum)

    # near-optimal limit for plotting (mirrors NoptimalTolerance in solver.py)
    thr = None
    try:
        # We don't have scenario passed directly, but could infer from base[opt_idx]
        pass
    except Exception:
        pass

    # --- summary table -----------------------------------------------------------
    P = rank_counts / cfg.n_samples
    expected_rank = (np.arange(1, rank_counts.shape[1] + 1) * P).sum(axis=1)  # lower bound (k+ counted as k+1)

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

    # --- figures -----------------------------------------------------------------
    plot_lcoe_distributions(L, base, labels, opt_idx, thr, cfg, os.path.join(out_dir, "lcoe_distributions.png"))
    plot_rank_probability(rank_counts, cfg.n_samples, base, labels, opt_idx, beats_opt, cfg,
                          os.path.join(out_dir, "rank_probability.png"))
    plot_pairwise_win_rates(L, base, labels, cfg,
                            os.path.join(out_dir, "pairwise_win_rates.png"))
    plt.show()
    return summary


# ==================================================================================
# TODO(details): use the dedicated details file instead of re-simulating
# ==================================================================================
# The simulation step only exists to recover the variable-cost drivers (nodal
# generation by dispatchable tech). Fixed-cost drivers come straight from x via
# AssetTensor and never need a simulation. Once details are piped through:
#
#   details_path = os.path.join(mga_dir, "mga-noptima-details.csv")   # name TBC
#
#   details = pd.read_csv(details_path, header=None)
#   # Columns follow ExtractDetailsTensor:
#   #   0 lcoe | 1 penalties | 2 nuclear | 3 nuclear_lte | 4 nphes
#   #   5 biomass | 6 biogas | 7 ccgt | (8+ currently unused: zero-VOM techs)
#
#   # ALIGNMENT: details.bin rows correspond to population.bin rows, not to
#   # mga-noptima.csv rows. Match on x (hash of the rounded x vector), e.g.
#   #   pop = pd.read_csv(population_csv, header=None)    # cols: obj, viol, fit, x...
#   #   key = lambda M: [hash(np.round(r, 6).tobytes()) for r in M]
#   #   lookup = dict(zip(key(pop.iloc[:, 3:].to_numpy()), range(len(pop))))
#   #   idx = [lookup[k] for k in key(X)]          # raises if a row can't be matched
#   #   details = details.iloc[idx].reset_index(drop=True)
#   #   assert np.allclose(details[0], df["objective"], rtol=1e-3)   # sanity check
#
#   def terms_from_details(x, drow, static):
#       a = AssetTensor(x, static)                 # cheap; no simulation
#       fixed = <same F_* terms as cost_terms(), using `a` for capacities>
#       c = static.costs
#       # System-wide totals only -> must use a single V per tech, like
#       # EvaluateTensorFromDetails does (c.V*[0]):
#       variable = [("V_nuke", drow[2] * c.Vnuke[0]), ("V_nlte", drow[3] * c.Vnlte[0]),
#                   ("V_ph",   drow[4] * c.Vph[0]),   ("V_biom", drow[5] * c.Vbiom[0]),
#                   ("V_biog", drow[6] * c.Vbiog[0]), ("V_gas",  drow[7] * c.Vgas[0])]
#       return fixed + variable + [("legacy", [static.legacy_costs])]
#
#   # CAVEATS for the details route:
#   #  * perturb_mode must be "group" (nodal energy split is not in details, so
#   #    per-node variable-cost multipliers can't be applied).
#   #  * if Vnuke/Vbiom/... ever differ between nodes, c.V*[0] is wrong (this is
#   #    also true of EvaluateTensorFromDetails today).
