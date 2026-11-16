"""Experiment 4 (claim 2, and how many examples are needed): learning curves of the best model
of each kind. Run after exp1 to exp3.

    python exp4_data_size.py"""

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

from models import make_estimator
from utils import AQUA, BLUE, CV, FOLD_TRAIN, G, GRAY, LC, MAGENTA, NS, VIOLET
from utils import best_of, cv_eval, run, savefig

LOOK = dict(  # label, colour and marker of each learning curve
    wkpoly=("Kernel, forest weights", VIOLET, "s"),
    lr=("LR on all bits", BLUE, "o"),
    rf=("Random forest on all bits", AQUA, "^"),
    mlp=("MLP 4×512 on all bits", MAGENTA, "v"),
)


def fig_learning_curves(cands):
    """Held-out accuracy against the number of training rows."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.0), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        for g in cands[n]:
            fam, cond, sub = g["key"]
            rs = sorted(LC.where(n=n, family=fam, cond=cond, sub=sub), key=lambda r: r["size"])
            S = [r["size"] for r in rs]
            m, s = np.array([r["mean"] for r in rs]), np.array([r["sd"] for r in rs])
            label, col, mk = LOOK["lr" if fam.startswith("lr") else fam]
            ax.plot(S, m, marker=mk, ms=4, color=col, label=label)
            ax.fill_between(S, m - s, m + s, color=col, alpha=0.15, lw=0)
        ax.axhline(0.5, color=GRAY, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([500, 2000, 10000])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"n = {n}")
        ax.set_xlabel("training rows")
        ax.set_ylim(0.40, 1.0)
    axes[0].set_ylabel("held-out accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    savefig(fig, "fig_learning_curves")
    print(
        "Mean over the first folds (shaded: one sd), each model at its CV-best setting. Dotted: 0.5"
    )


def run_learning_curves():
    """Train the best model of each kind on nested random subsets of the training folds and score
    it on the held-out fold; the weighted kernel computes its weights on the subset. For LR and
    the kernels, C multiplies the summed loss, so the same C penalises more strongly with fewer
    rows; C is scaled by (full rows / rows used) to keep the penalty per row fixed. Returns the
    models per n."""
    cands = {}
    for n in NS:
        cands[n] = [
            best_of(n, "all", ("wkpoly",)),
            best_of(n, "all"),
            best_of(n, "all", ("rf",)),
        ]
        if CV.where(n=n, family="mlp", exp="MLP"):
            cands[n].append(best_of(n, "all", ("mlp",)))
        for g in cands[n]:
            fam, cond, sub = g["key"]
            p = g["best"]["params"]
            meta = dict(exp="LC", n=n, family=fam, cond=cond, sub=sub)
            for size in [s for s in G["LC_SIZES"] if s < FOLD_TRAIN[n]]:
                ps = dict(p, C=p["C"] * FOLD_TRAIN[n] / size) if "C" in p else p
                run(
                    LC,
                    f"lc|{n}|{fam}|{cond}|{sub}|{size}",
                    lambda: cv_eval(
                        lambda: make_estimator(fam, ps, cond), n, size=size, folds=G["LC_FOLDS"]
                    ),
                    size=size,
                    **meta,
                )
            fr = [g["best"]["folds"][f] for f in G["LC_FOLDS"]]
            run(
                LC,
                f"lc|{n}|{fam}|{cond}|{sub}|{FOLD_TRAIN[n]}",
                lambda: dict(
                    folds=fr, mean=float(np.mean(fr)), sd=float(np.std(fr, ddof=1)), secs=0.0
                ),
                quiet=True,
                size=FOLD_TRAIN[n],
                **meta,
            )
    return cands


if __name__ == "__main__":
    fig_learning_curves(run_learning_curves())
