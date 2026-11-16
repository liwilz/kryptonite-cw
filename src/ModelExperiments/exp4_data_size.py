"""Experiment 4 (H3): learning curves on the test set. MANBAT, the paper's baseline, the random
forest and the MLP, each at its CV-best setting, are trained on random subsets of the non-test rows
and scored on the test set. Nothing is chosen with these scores. Run after exp1 to exp3.

    python exp4_data_size.py"""

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from final_pipeline import test_score
from models import make_estimator
from utils import AQUA, BLUE, CV, DEV, LC, LOCKED, MAGENTA, NS, SEED, VIOLET, G, GRAY
from utils import _capped, acc, best_of, grid, load, pct, run, savefig

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


def test_eval(make, n, fam, size):
    """Fit make() on LC_REPEATS random subsets of `size` non-test rows and score each on the test
    set. The subsets are nested across sizes: the first `size` rows of one random order per
    repeat."""
    X, B, y = load(n)
    M = X if fam == "paper" else B  # the baseline uses the raw inputs, as everywhere
    t0, accs, capped = time.time(), [], 0
    for r in range(G["LC_REPEATS"]):
        tr = DEV[n][np.random.default_rng(SEED + r).permutation(len(DEV[n]))[:size]]
        est = make().fit(M[tr], y[tr])
        accs.append(acc(est, M[LOCKED[n]], y[LOCKED[n]]))
        capped += bool(_capped(est))
    return dict(
        capped=capped,
        folds=[round(a, 5) for a in accs],
        mean=float(np.mean(accs)),
        sd=float(np.std(accs, ddof=1)),
        secs=round(time.time() - t0, 1),
    )


def run_learning_curves():
    """Train each model on random subsets of LC_SIZES non-test rows and score it on the test set.
    MANBAT computes its weights on the subset. C multiplies the summed loss, so for MANBAT it is
    scaled by (all non-test rows / rows used) to keep the penalty per row, lambda, at its value in
    the final refit; the baseline keeps the paper's C = 0.85, as in the paper's own experiment. The
    point at all non-test rows is the model's test score in Table 1. Returns the models per n."""
    cands = {}
    for n in NS:
        cands[n] = [best_of(n, "all", (fam,)) for fam in ("wkpoly", "paper", "rf")]
        if CV.where(n=n, family="mlp", exp="MLP"):
            cands[n].append(best_of(n, "all", ("mlp",)))
        for g in cands[n]:
            fam, cond, sub = g["key"]
            p = g["best"]["params"]
            meta = dict(exp="LCT", n=n, family=fam, cond=cond, sub=sub)
            for size in [s for s in G["LC_SIZES"] if s < len(DEV[n])]:
                ps = dict(p, C=p["C"] * len(DEV[n]) / size) if fam == "wkpoly" else p
                run(
                    LC,
                    f"lct|{n}|{fam}|{cond}|{sub}|{size}",
                    lambda: test_eval(lambda: make_estimator(fam, ps, cond), n, fam, size),
                    quiet=True,
                    size=size,
                    **meta,
                )
            t = test_score(n, g)["mean"]  # the refit on all non-test rows (Table 1)
            run(
                LC,
                f"lct|{n}|{fam}|{cond}|{sub}|{len(DEV[n])}",
                lambda: dict(folds=[t], mean=t, sd=0.0, secs=0.0),
                quiet=True,
                size=len(DEV[n]),
                **meta,
            )
    return cands


def curve(n, g):
    """The stored learning curve of one model: [(rows, mean, sd)], fewest rows first."""
    fam, cond, sub = g["key"]
    rs = LC.where(exp="LCT", n=n, family=fam, cond=cond, sub=sub)
    rs = sorted(rs, key=lambda r: r["size"])
    return [(r["size"], r["mean"], r["sd"]) for r in rs]


def fig_learning_curves(cands):
    """Test accuracy against the number of training rows."""
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
    axes[0].set_ylabel("test accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    savefig(fig, "fig_learning_curves")


def table_learning_curves(cands):
    """The numbers behind the figure that the report quotes."""
    rows = []
    for size in (250, 4000, "all"):
        rows.append([f"{size} training rows" if size != "all" else "all non-test rows"] + [""] * 3)
        for g in cands[NS[0]]:
            fam = g["key"][0]
            cells = []
            for n in NS:
                c = curve(n, next(h for h in cands[n] if h["key"][0] == fam))
                m = c[-1][1] if size == "all" else dict((s, a) for s, a, _ in c).get(size)
                cells.append("-" if m is None else pct(m))
            rows.append([f"    {LOOK[fam][0]}"] + cells)
    grid(
        f"Learning curves: test accuracy (%), mean over {G['LC_REPEATS']} random subsets",
        rows,
        "Figure: the same means, shaded one sd over the subsets; the last point of each curve is"
        " the model's\ntest score in Table 1 (one refit on all non-test rows); dotted line at 50%.",
    )


if __name__ == "__main__":
    main()
