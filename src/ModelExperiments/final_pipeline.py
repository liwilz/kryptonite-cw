"""The final pipeline per n: chosen by CV, scored once on the locked rows, its controls, the
hidden-set predictions, and the summary table. Run after exp1 to exp4.

    python final_pipeline.py"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from models import make_estimator, top_bits
from utils import (
    CV,
    DATA,
    DEV,
    FIN,
    HIDDEN,
    INK,
    GRAY,
    LOCKED,
    NAME,
    NS,
    ROOT,
    VIOLET,
    WRITE_PREDICTIONS,
    G,
    K,
    acc,
    best_of,
    binomial_half,
    check_caps,
    cv_eval,
    difference,
    grid,
    groups,
    kernel_label,
    kname,
    load,
    ms,
    run,
    savefig,
    show_params,
)


def choose_final():
    """The final pipeline for each n: the candidate with the highest CV mean, the paper baseline
    excluded."""
    return {n: [g for g in groups(n) if g["key"][0] != "paper"][0] for n in NS}


def locked_score(final, n):
    """Refit the final pipeline on all dev rows and score it once on the locked rows (stored; later
    calls read it back)."""
    fam, cond, sub = final[n]["key"]
    p = final[n]["best"]["params"]
    X, B, y = load(n)
    tr, te = DEV[n], LOCKED[n]

    def job():
        t0 = time.time()
        est = make_estimator(fam, p, cond).fit(B[tr], y[tr])
        a = acc(est, B[te], y[te])
        return dict(
            mean=a,
            sd=0.0,
            secs=round(time.time() - t0, 1),
            n_test=int(len(te)),
            train=acc(est, B[tr], y[tr]),
            half=binomial_half(a, len(te)),
            top_bits=top_bits(est.w_) if hasattr(est, "w_") else None,
            params=p,
        )

    key = f"locked|{n}|{fam}|{cond}|{sub}|{json.dumps(p, sort_keys=True)}"
    return run(FIN, key, job, exp="FINAL", n=n, family=fam, cond=cond, sub=sub)


def run_controls(final):
    """Two controls on the first folds, both expected at about 0.5: the final pipeline with
    shuffled training labels (otherwise something leaks), and the best weighted kernel with its
    weights moved to the wrong bits (otherwise the weights are not what makes it work)."""
    for n in NS:
        fam, cond, sub = final[n]["key"]
        p = final[n]["best"]["params"]
        meta = dict(exp="CONTROL", n=n, family=fam, cond=cond, sub=sub)
        r = run(
            FIN,
            f"shuffle|{n}",
            lambda: cv_eval(
                lambda: make_estimator(fam, p, cond), n, folds=G["LC_FOLDS"], shuffle_labels=True
            ),
            quiet=True,
            **meta,
        )
        assert (
            abs(r["mean"] - 0.5) < 0.03
        ), f"n={n}: shuffled-label accuracy {r['mean']:.3f}, possible leakage"
        # the best weighted kernel, whether or not it is the final pipeline (if it is, this is the
        # same model)
        wk = best_of(n, "all", ("wkpoly",))
        pw = wk["best"]["params"]
        run(
            FIN,
            f"reversed|{n}",
            lambda: cv_eval(
                lambda: make_estimator("wkpoly", pw, "reversed"), n, folds=G["LC_FOLDS"]
            ),
            quiet=True,
            **dict(meta, family="wkpoly", sub=wk["key"][2]),
        )


def predict_hidden(final):
    """Refit the final pipeline on all labelled rows, check its predictions, write
    hiddenlabels/y_predicted_{n}.npy and run the course's check_submission.py."""
    if WRITE_PREDICTIONS:
        os.makedirs(HIDDEN, exist_ok=True)
    for n in NS:
        fam, cond, sub = final[n]["key"]
        X, B, y = load(n)
        Xh = np.load(f"{DATA}/hidden-kryptonite-{n}-X.npy")
        Bh = np.where(Xh > 0, 1.0, -1.0).astype(np.float32)
        p = final[n]["best"]["params"]
        good = [g for g in groups(n) if g["best"]["mean"] > 0.9 and g["key"][0] != fam]
        alt = good[0] if good else None

        def job():
            t0 = time.time()
            est = make_estimator(fam, p, cond).fit(B, y)
            out = dict(pred=est.predict(Bh).tolist())
            if hasattr(est, "w_"):
                out["top_bits"] = top_bits(est.w_)
            if alt:
                m = make_estimator(alt["key"][0], alt["best"]["params"], alt["key"][1]).fit(B, y)
                out["alt"] = m.predict(Bh).tolist()
            return dict(out, mean=0.0, sd=0.0, secs=round(time.time() - t0, 1))

        key = f"hidden|{n}|{fam}|{cond}|{sub}|{json.dumps(p, sort_keys=True)}"
        h = run(FIN, key, job, quiet=True, exp="HIDDEN", n=n, family=fam, cond=cond, sub=sub)
        pred = np.array(h["pred"], dtype=np.int64)
        assert pred.shape == (10_000,) and set(np.unique(pred)) <= {0, 1}, "wrong shape or values"
        assert 0.45 <= pred.mean() <= 0.55, f"suspicious class balance {pred.mean():.3f}"
        assert (
            np.abs(Bh.mean(0) - B.mean(0)).max() < 0.05
        ), "hidden bits do not look like the labelled bits"
        assert (
            np.mean(np.abs(Xh - Bh) < 0.5) > 0.999
        ), "hidden inputs are not clean +-1 bits plus small noise"
        msg = f"n={n}: {kname(final[n]['key'])}, predicted positive rate {pred.mean():.3f}"
        # The best model of another kind must mostly agree. If each errs on a share e
        # of rows, they can disagree on at most e1 + e2 of them; 0.02 allows for sampling noise.
        if alt:
            agree = float((np.array(h["alt"]) == pred).mean())
            low = 1 - (1 - final[n]["best"]["mean"]) - (1 - alt["best"]["mean"]) - 0.02
            msg += f"; agrees with {kname(alt['key'])} on {agree:.1%} (must be ≥ {low:.1%})"
            assert agree >= low, msg + ": one of the two models is worse on the hidden rows"
        print(msg)
        if WRITE_PREDICTIONS:
            np.save(f"{HIDDEN}/y_predicted_{n}.npy", pred)
    if WRITE_PREDICTIONS:
        try:  # the course's own format checker, at the repository root
            sys.path.insert(0, ROOT)
            import check_submission as CS

            for name in CS.FILES:
                pr = CS.problems_with(os.path.join(HIDDEN, name))
                print(
                    f"{'OK  ' if not pr else 'FAIL'}  {name}" + (f": {'; '.join(pr)}" if pr else "")
                )
        except ImportError:
            print("check_submission.py not found at the repository root; run it from there instead")


def table_final(final):
    """The final pipeline: CV, locked test, the two controls and the paired comparisons."""
    for n in NS:
        print(f"n = {n}: {kname(final[n]['key'])} ({show_params(final[n]['best']['params'])})")
    lk = {n: locked_score(final, n) for n in NS}
    rows = [
        ["CV accuracy on dev (mean ± sd)"] + [ms(final[n]["best"]) for n in NS],
        ["Locked test accuracy"] + [f"{lk[n]['mean']:.3f}" for n in NS],
        ["    95% interval"]
        + [
            f"[{lk[n]['mean'] - lk[n]['half']:.3f}, {lk[n]['mean'] + lk[n]['half']:.3f}]"
            for n in NS
        ],
        ["Control: shuffled training labels"]
        + [f"{FIN.rows[f'shuffle|{n}']['mean']:.3f}" for n in NS],
        ["Control: best weighted kernel, weights on the wrong bits"]
        + [f"{FIN.rows[f'reversed|{n}']['mean']:.3f}" for n in NS],
    ]
    others = [
        ("Final minus best explicit LR on all bits", lambda n: best_of(n, "all")),
        ("Final minus random forest on all bits", lambda n: best_of(n, "all", ("rf",))),
    ]
    if all(CV.where(n=n, family="mlp", exp="MLP") for n in NS):
        others.append(("Final minus MLP 4×512", lambda n: best_of(n, "all", ("mlp",))))
    for label, other in others:
        rows.append(
            [label] + [difference(final[n]["best"]["folds"], other(n)["best"]["folds"]) for n in NS]
        )
    print()
    grid(
        "Final pipeline",
        rows,
        "Both controls should be about 0.5. Differences: mean ± sd over the folds (folds where the"
        "\nfinal pipeline is better). The folds share training rows, so the sd understates the"
        "\nuncertainty; the locked test is the independent check.",
    )


def summary_rows(final):
    """The rows of the summary table: ('group', title), ('row', label, accuracy per n) and
    ('deg', label, degree per n), plus the footnote."""

    def best(cond, fams, sub=None):
        return [best_of(n, cond, fams, sub)["best"] for n in NS]

    def degree(cond, fams):
        return [int(best_of(n, cond, fams)["key"][2][3:]) for n in NS]

    rows = [
        ("group", "Polynomial LR on all bits (the paper's approach)"),
        ("row", "Paper baseline (quadratic)", [b["mean"] for b in best("all", ("paper",))]),
        (
            "row",
            "Explicit monomials, best degree",
            [b["mean"] for b in best("all", ("lr_l2",))],
        ),
        ("deg", "degree", degree("all", ("lr_l2",))),
        ("row", "Paper's kernel, best degree", [b["mean"] for b in best("all", ("kpoly",))]),
        ("deg", "degree", degree("all", ("kpoly",))),
        ("group", "Trained from scratch on all bits"),
        ("row", "Random forest", [b["mean"] for b in best("all", ("rf",))]),
    ]
    if all(CV.where(n=n, family="mlp", exp="MLP") for n in NS):
        rows.append(("row", NAME["mlp"], [b["mean"] for b in best("all", ("mlp",))]))
    rows.append(("group", "Polynomial kernel on all bits, forest-weighted"))
    rows.append(
        ("row", "Equal weights, best degree", [b["mean"] for b in best("all", ("ukpoly",))])
    )
    rows.append(("deg", "degree", degree("all", ("ukpoly",))))
    for d in G["WK_DEGS"]:
        rows.append(
            (
                "row",
                f"Forest weights, degree {d}",
                [b["mean"] for b in best("all", ("wkpoly",), f"deg{d}")],
            )
        )
    rows += [
        ("group", "Final pipeline on the locked test rows"),
        ("row", "Locked test accuracy", [locked_score(final, n)["mean"] for n in NS]),
    ]
    kp = ", ".join(kernel_label(i) for i in range(len(G["KP_DEGS"][NS[0]])))
    md = ", ".join(str(G["MAX_DEG"][n]) for n in NS)
    notes = [
        f"{K}-fold CV on the dev rows, except the last row. A coin flip scores 0.50.",
        f"Degrees tried on all bits: explicit monomials up to {md} (the largest that fits",
        f"in memory); kernel {kp}.",
    ]
    return rows, notes


def fig_summary_table(final):
    """The summary as a typeset table (rules, groups, best CV mean per column in bold)."""
    rows, notes = summary_rows(final)
    width, step, right = 6.2, 0.27, (0.72, 0.86, 1.0)
    bottom = len(rows) + 1.0 + 0.75 * (len(notes) - 1) + 0.6
    fig, ax = plt.subplots(figsize=(width, (bottom + 0.7) * step))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom, -0.7)
    ax.grid(False)
    group_of = []
    for i, r in enumerate(rows):
        group_of.append(i if r[0] == "group" else group_of[-1])
    counted = [i for i, r in enumerate(rows) if r[0] == "row" and group_of[i] != max(group_of)]
    top = [max(round(rows[i][2][c], 3) for i in counted) for c in range(len(NS))]

    def rule(y, lw):
        ax.plot([0, 1], [y, y], color=INK, lw=lw, solid_capstyle="butt", clip_on=False)

    for i, g in enumerate(group_of, start=1):  # tint the forest-weighted kernel rows
        if "forest-weighted" in rows[g][1]:
            ax.axhspan(i - 0.5, i + 0.5, color=VIOLET, alpha=0.1, lw=0)
    rule(-0.5, 1.2)
    for c, n in enumerate(NS):
        ax.text(right[c], 0, f"n = {n}", ha="right", va="center", fontweight="bold", color=INK)
    rule(0.5, 0.6)
    for i, r in enumerate(rows, start=1):
        if r[0] == "group":
            if i > 1:
                rule(i - 0.5, 0.3)
            ax.text(0, i, r[1], va="center", fontweight="bold", color=INK)
            continue
        indent, col = (0.06, GRAY) if r[0] == "deg" else (0.03, INK)
        ax.text(indent, i, r[1], va="center", color=col)
        for c, v in enumerate(r[2]):
            weight = "bold" if i - 1 in counted and round(v, 3) == top[c] else "normal"
            text, col = (str(v), GRAY) if r[0] == "deg" else (f"{v:.3f}", INK)
            ax.text(right[c], i, text, ha="right", va="center", color=col, fontweight=weight)
    rule(len(rows) + 0.5, 1.2)
    for j, line in enumerate(notes):
        ax.text(0, len(rows) + 1.0 + 0.75 * j, line, va="center", fontsize=8, color=GRAY)
    savefig(fig, "table_summary")


if __name__ == "__main__":
    FINAL = choose_final()
    run_controls(FINAL)
    table_final(FINAL)
    check_caps()
    print()
    predict_hidden(FINAL)
    fig_summary_table(FINAL)
