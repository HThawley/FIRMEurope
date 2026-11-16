import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import gaussian_kde


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
        ax.text(
            nopt_threshold,
            ax.get_ylim()[1] * 0.5,
            " near-optimal limit",
            ha="left",
            va="center",
            fontsize=8,
            rotation=90,
        )

    ax.set_ylabel("Probability density")
    ax.set_title(
        f"LCOE of near-optimal alternatives under ±{cfg.rel_halfwidth:.0%} cost uncertainty "
        f"({cfg.n_samples:,} draws, {cfg.perturb_mode}-level)"
    )
    ax.legend(loc="upper left", frameon=False)
    plt.setp(ax.get_xticklabels(), visible=False)

    axn.scatter(base, np.zeros(S), s=28, marker="|", color="0.25", zorder=3)
    axn.scatter(base[opt_idx], 0, s=70, marker="*", color="crimson", zorder=4)
    for s in order[:2]:
        axn.annotate(
            labels[s],
            (base[s], 0),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
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
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if v > 0.5 else "black",
                )

    ax.set_xticks(range(K1))
    ax.set_xticklabels([_ordinal(r + 1) for r in range(K1 - 1)] + [f"{K1}th+"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[s] for s in rows])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Resulting rank")
    ax.set_ylabel("Alternative solutions")
    ax.set_title(f"Rank probability matrix (top {len(rows)} of {S} by baseline LCOE)", pad=34)

    beats = beats_opt.copy()
    beats[opt_idx] = 0
    rival = int(np.argmax(beats))
    msg = (
        f"Base-case optimum ({labels[opt_idx]}) is cheapest in {P[opt_idx, 0]:.0%} of draws; "
        f"most often displaced by {labels[rival]} ({beats[rival]:.0%} of draws)."
    )
    fig.text(0.5, 0.01, msg, ha="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Probability", fraction=0.04, pad=0.03)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_pairwise_win_rates(L, base, labels, cfg, path):
    """Fig 3: Square heatmap showing the probability that Row Alternative is cheaper than Column Alternative."""
    order = np.argsort(base)
    pick = order[: cfg.max_pairwise_alts]
    N = len(pick)

    L_pick = L[:, pick]
    M = (L_pick[:, :, None] < L_pick[:, None, :]).mean(axis=0)
    np.fill_diagonal(M, np.nan)

    fig, ax = plt.subplots(figsize=(0.7 * N + 2, 0.7 * N + 1))

    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(N):
        for j in range(N):
            if i != j:
                v = M[i, j]
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
    ax.set_title(
        f"Pairwise Win Rates: P(Row cheaper than Column)\nTop {N} alternatives by baseline LCOE",
        pad=25,
    )

    fig.colorbar(im, ax=ax, label="Row Win Probability", fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
