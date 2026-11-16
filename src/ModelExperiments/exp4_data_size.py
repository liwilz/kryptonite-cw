"""Experiment 4 (H3): learning curves. MANBAT, the paper's baseline, the random forest and the MLP,
each at its CV-best setting, trained on nested random subsets of the training rows of the first
three folds and scored on the fold's validation rows. Run after exp1 to exp3.

    python exp4_data_size.py"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from models import make_estimator
from utils import AQUA, BLUE, CV, FOLD_TRAIN, LC, MAGENTA, NS, VIOLET, G, GRAY
from utils import best_of, cv_eval, grid, pct, run, savefig

LOOK = dict(  # label, colour and marker of each learning curve
    wkpoly=("MANBAT", VIOLET, "s"),
    paper=("Baseline (degree 2, raw inputs)", BLUE, "o"),
    rf=("Random forest", AQUA, "^"),
    mlp=("MLP", MAGENTA, "v"),
)


def main():
    cands = run_learning_curves()
    fig_learning_curves(cands)
    table_learning_curves(cands)


def run_learning_curves():
    """Train each model on nested random subsets of the training folds (LC_FOLDS) and score it on
    the held-out fold. MANBAT computes its weights on the subset. C multiplies the summed loss, so
    for MANBAT it is scaled by (all training rows / rows used) to keep the penalty per row, lambda,
    fixed; the baseline keeps the paper's C = 0.85, as in the paper's own experiment. The point at
    all training rows is the ordinary CV score. Returns the models per n."""
    cands = {}
    for n in NS:
        cands[n] = [best_of(n, "all", (fam,)) for fam in ("wkpoly", "paper", "rf")]
        if CV.where(n=n, family="mlp", exp="MLP"):
            cands[n].append(best_of(n, "all", ("mlp",)))
        for g in cands[n]:
            fam, cond, sub = g["key"]
            p = g["best"]["params"]
            meta = dict(exp="LC", n=n, family=fam, cond=cond, sub=sub)
            for size in [s for s in G["LC_SIZES"] if s < FOLD_TRAIN[n]]:
                ps = dict(p, C=p["C"] * FOLD_TRAIN[n] / size) if fam == "wkpoly" else p
                run(
                    LC,
                    f"lc|{n}|{fam}|{cond}|{sub}|{size}",
                    lambda: cv_eval(
                        lambda: make_estimator(fam, ps, cond),
                        n,
                        fam=fam,  # the baseline uses the raw inputs, as in CV
                        size=size,
                        folds=G["LC_FOLDS"],
                    ),
                    quiet=True,
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


def curve(n, g):
    """The stored learning curve of one model: [(rows, mean, sd)], fewest rows first."""
    fam, cond, sub = g["key"]
    rs = sorted(LC.where(n=n, family=fam, cond=cond, sub=sub), key=lambda r: r["size"])
    return [(r["size"], r["mean"], r["sd"]) for r in rs]


def fig_learning_curves(cands):
    """Held-out accuracy against the number of training rows."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.0), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        for g in cands[n]:
            S, m, s = (np.array(v) for v in zip(*curve(n, g)))
            label, col, mk = LOOK[g["key"][0]]
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


def table_learning_curves(cands):
    """The numbers behind the figure that the report quotes."""
    rows = []
    for size in (250, 4000, "all"):
        rows.append([f"{size} training rows" if size != "all" else "all training rows"] + [""] * 3)
        for g in cands[NS[0]]:
            fam = g["key"][0]
            cells = []
            for n in NS:
                c = curve(n, next(h for h in cands[n] if h["key"][0] == fam))
                m = c[-1][1] if size == "all" else dict((s, a) for s, a, _ in c).get(size)
                cells.append("-" if m is None else pct(m))
            rows.append([f"    {LOOK[fam][0]}"] + cells)
    grid(
        "Learning curves: held-out accuracy (%), mean over folds "
        f"{', '.join(str(f + 1) for f in G['LC_FOLDS'])}",
        rows,
        "Figure: mean over these folds, shaded one sd; dotted line at 50%.",
    )


if __name__ == "__main__":
    main()
