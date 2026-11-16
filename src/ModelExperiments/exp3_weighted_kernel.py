"""Experiment 3 (claim 1, and what degree a polynomial needs): the forest's permutation
importances as per-bit weights in a polynomial kernel, against the same kernel with equal weights.
Run after exp1_degree.py (the degree figure also shows its results).

    python exp3_weighted_kernel.py"""

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

from models import cv_config, fold_weights, top_bits
from utils import (
    BLUE,
    GRAY,
    INK,
    NS,
    ORANGE,
    VIOLET,
    G,
    _mean_sd,
    best_of,
    grid,
    savefig,
    show_params,
)


def run():
    for n in NS:  # the bit weights of every fold first, so their energy is recorded on its own
        fold_weights(n)
    for n in NS:
        for d in G["WK_DEGS"]:
            cv_config(n, "UK", "ukpoly", "all", f"deg{d}", dict(deg=d, C=G["WK_UNIFORM_C"]))
            for C in G["WK_C"]:
                cv_config(n, "WK", "wkpoly", "all", f"deg{d}", dict(deg=d, C=C))


def table_weights():
    """How clearly the forest's weights single out a few bits."""
    W = {n: fold_weights(n) for n in NS}

    def mean(f):
        return [f"{np.mean([f(np.sort(w)[::-1]) for w in W[n]]):.3f}" for n in NS]

    rows = [
        ["10 largest weights on the same bits in every fold"]
        + ["yes" if len({tuple(top_bits(w)) for w in W[n]}) == 1 else "no" for n in NS],
        ["Share of the total weight on those 10 bits"] + mean(lambda w: w[:10].sum() / w.sum()),
        ["Bits with weight 0 (left out of the kernel)"]
        + [f"{np.mean([(w == 0).sum() for w in W[n]]):.1f}" for n in NS],
    ]
    grid(
        "Bit weights from one random forest per training fold (permutation importance)",
        rows,
        "Negative importances are set to 0, then the largest weight is scaled to 1; mean over"
        " folds.",
    )
    for n in NS:
        print(f"n = {n}: the 10 bits with the largest weights in fold 1: {top_bits(W[n][0])}")


def table_weighted_kernel():
    """The same polynomial kernel with equal and with forest weights, by degree."""
    rows = [["Equal weights"] + [""] * len(NS)]
    rows += [
        [f"    degree {d}"] + [_mean_sd(n, "all", ("ukpoly",), f"deg{d}") for n in NS]
        for d in G["WK_DEGS"]
    ]
    rows += [[""] * (len(NS) + 1), ["Forest weights (best C per degree)"] + [""] * len(NS)]
    rows += [
        [f"    degree {d}"] + [_mean_sd(n, "all", ("wkpoly",), f"deg{d}") for n in NS]
        for d in G["WK_DEGS"]
    ]
    best = {n: best_of(n, "all", ("wkpoly",))["best"] for n in NS}
    rows.append(
        ["        training accuracy, best setting"] + [f"{best[n]['train']:.3f}" for n in NS]
    )
    grid("Polynomial kernel LR on all bits: CV accuracy by degree (mean ± sd over folds)", rows)
    for n in NS:
        print(f"n = {n}: best weighted setting {show_params(best[n]['params'])}")


def fig_degree():
    """CV accuracy against the polynomial degree on all bits: explicit monomials, the paper's
    kernel, and the polynomial kernel with equal and with forest weights."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.8), sharey=True, layout="constrained")
    series = (  # drawn in this order, so LR on all bits stays visible where the two LR lines meet
        ("Kernel, forest weights", VIOLET, "s", "all", ("wkpoly",), lambda n: G["WK_DEGS"]),
        ("Kernel, equal weights", GRAY, "^", "all", ("ukpoly",), lambda n: G["WK_DEGS"]),
        ("Paper's kernel", ORANGE, "D", "all", ("kpoly",), lambda n: G["KP_DEGS"][n]),
        (
            "LR on all bits",
            BLUE,
            "o",
            "all",
            ("lr_l2",),
            lambda n: range(1, G["MAX_DEG"][n] + 1),
        ),
    )
    for ax, n in zip(axes, NS):
        for label, col, mk, cond, fams, degs in series:
            bs = [best_of(n, cond, fams, f"deg{d}")["best"] for d in degs(n)]
            m, s = np.array([b["mean"] for b in bs]), np.array([b["sd"] for b in bs])
            ax.plot(list(degs(n)), m, marker=mk, ms=4 if mk != "o" else 5, color=col, label=label)
            ax.fill_between(list(degs(n)), m - s, m + s, color=col, alpha=0.15, lw=0)
        base = best_of(n, "all", ("paper",))["best"]["mean"]
        ax.axhline(base, color=GRAY, ls="--", lw=1.3, label="Paper baseline")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 10, 20, 40, 80])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"n = {n}")
        ax.set_xlabel("polynomial degree")
        ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("CV accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    savefig(fig, "fig_degree")
    print("Mean over the folds (shaded: one sd), best setting at each degree.")


def fig_weights():
    """The forest's weight of every bit, mean over folds (bars) and the spread over folds."""
    fig, axes = plt.subplots(3, 1, figsize=(6.75, 4.2), layout="constrained")
    for ax, n in zip(axes, NS):
        W = np.array(fold_weights(n))
        m, lo, hi = W.mean(0), W.min(0), W.max(0)
        top = set(top_bits(m))
        cols = [VIOLET if i in top else "#cfcec9" for i in range(n)]
        ax.bar(range(n), m, color=cols, width=0.8)
        ax.errorbar(range(n), m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.8)
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_xticks(range(0, n, 5))
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"n = {n}")
        ax.grid(axis="x", alpha=0)
    axes[-1].set_xlabel("bit index")
    savefig(fig, "fig_weights")
    print(
        "Bars: mean weight over folds (violet: the 10 largest); whiskers: smallest to largest fold."
    )


if __name__ == "__main__":
    run()
    table_weights()
    fig_weights()
    table_weighted_kernel()
    fig_degree()
