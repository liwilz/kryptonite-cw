"""Experiment 2 (claim 3): a random forest and an MLP with KryptoNet-40's shape (4 x 512),
trained from scratch on all bits; then ablations of the MLP, one change at a time. Run after
exp1_degree.py (its table shows the best polynomial for reference).

    python exp2_from_scratch.py"""

import numpy as np

from models import cv_by_fold, cv_config, make_estimator
from utils import (
    CV,
    G,
    K,
    NS,
    RUN_MLP,
    _mean_sd,
    best_of,
    cv_eval,
    difference,
    grid,
    run,
    show_params,
)


def run_models():
    for n in NS:
        for f in G["RF_MAXF"]:
            for leaf in G["RF_LEAF"]:
                cv_config(n, "RF", "rf", "all", "rf", dict(max_features=f, msl=leaf))
        if RUN_MLP:
            cv_by_fold(n, "mlp", dict(h=G["MLP_H"]), "4x512")


def table_families():
    """The best setting of every other model family on all bits."""
    rows = [
        ["Random forest"] + [_mean_sd(n, "all", ("rf",)) for n in NS],
        ["MLP 4×512"] + [_mean_sd(n, "all", ("mlp",)) for n in NS],
        ["Best polynomial LR, for reference"]
        + [_mean_sd(n, "all", ("lr_l2", "kpoly")) for n in NS],
    ]
    grid("Trained from scratch on all bits: CV accuracy (mean ± sd over folds)", rows)
    for n in NS:
        g = best_of(n, "all", ("rf",))
        print(f"n = {n}: best forest has {show_params(g['best']['params'])}")


def run_ablations():
    """The MLP with one thing changed, on the folds LC_FOLDS; every fold is stored when it
    finishes. These rows are not candidates for the final pipeline (their condition is not
    'all')."""
    if not RUN_MLP:
        return
    for n in NS:
        for key, (label, h, raw) in G["MLP_ABLATIONS"].items():
            p = dict(h=h)
            for f in G["LC_FOLDS"]:
                run(
                    CV,
                    f"abl|{n}|{key}|fold{f}",
                    lambda: cv_eval(lambda: make_estimator("mlp", p), n, folds=(f,), raw=raw),
                    exp="ABL",
                    n=n,
                    family="mlp",
                    cond="raw" if raw else "bits",
                    sub=f"{label}, fold {f + 1}",
                    params=p,
                )


def table_ablations():
    """Each ablation against the 4 x 512 MLP on the bits, on the same folds."""
    folds = G["LC_FOLDS"]
    ref = {n: [best_of(n, "all", ("mlp",))["best"]["folds"][f] for f in folds] for n in NS}

    def ms(a):
        return f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}"

    rows = [["4×512 on the bits (above)"] + [ms(ref[n]) for n in NS]]
    for key, (label, h, raw) in G["MLP_ABLATIONS"].items():
        a = {n: [CV.rows[f"abl|{n}|{key}|fold{f}"]["folds"][0] for f in folds] for n in NS}
        rows.append([label] + [ms(a[n]) for n in NS])
        rows.append(["    minus 4×512 on the bits"] + [difference(a[n], ref[n]) for n in NS])
    grid(
        f"MLP ablations: accuracy on folds {', '.join(str(f + 1) for f in folds)} of {K}"
        " (mean ± sd)",
        rows,
        "Differences: mean ± sd over the same folds (folds where the ablation is better).",
    )


if __name__ == "__main__":
    run_models()
    table_families()
    run_ablations()
    table_ablations()
