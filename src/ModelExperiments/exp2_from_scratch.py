"""Experiment 2 (H2): a random forest and an MLP with KryptoNet-40's shape (4 x 512), trained
from scratch on all bits; then three ablations of the MLP, one change at a time.

    python exp2_from_scratch.py"""

import numpy as np

from models import cv_by_fold, cv_config, make_estimator
from utils import (
    CV,
    G,
    K,
    NS,
    RUN_MLP,
    best_of,
    cv_eval,
    grid,
    mean_sd,
    run,
)


def main():
    run_models()
    table_models()
    run_ablations()
    table_ablations()


def run_models():
    for n in NS:
        for f in G["RF_MAXF"]:
            for leaf in G["RF_LEAF"]:
                cv_config(n, "RF", "rf", "all", "rf", dict(max_features=f, msl=leaf))
        if RUN_MLP:
            cv_by_fold(n, "mlp", dict(h=G["MLP_H"]), "4x512")


def table_models():
    """CV accuracy of the forest (its best setting) and of the MLP."""
    rf = {n: best_of(n, "all", ("rf",))["best"]["params"] for n in NS}
    rows = [
        ["Random forest"] + [mean_sd(n, "all", ("rf",)) for n in NS],
        ["    chosen: inputs per split, min leaf"]
        + [f"{rf[n]['max_features']}, {rf[n]['msl']}" for n in NS],
        ["MLP"] + [mean_sd(n, "all", ("mlp",)) for n in NS],
    ]
    grid("Trained from scratch: CV accuracy (%, mean ± sd over folds)", rows)


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
    """Each ablation against the MLP, on the same folds: accuracy, then the difference."""
    folds = G["LC_FOLDS"]
    ref = {n: [best_of(n, "all", ("mlp",))["best"]["folds"][f] for f in folds] for n in NS}

    def ms(a):
        return f"{100 * np.mean(a):.1f} ± {100 * np.std(a, ddof=1):.1f}"

    abl = {
        label: {n: [CV.rows[f"abl|{n}|{key}|fold{f}"]["folds"][0] for f in folds] for n in NS}
        for key, (label, h, raw) in G["MLP_ABLATIONS"].items()
    }

    def points(a, b):
        d = 100 * (np.asarray(a) - np.asarray(b))
        return f"{d.mean():+.1f} ± {d.std(ddof=1):.1f}"

    rows = [["Accuracy (%)"] + [""] * len(NS), ["    MLP (four hidden layers of 512)"]]
    rows[-1] += [ms(ref[n]) for n in NS]
    rows += [[f"    {label}"] + [ms(a[n]) for n in NS] for label, a in abl.items()]
    rows += [["Difference from the MLP (points)"] + [""] * len(NS)]
    rows += [[f"    {label}"] + [points(a[n], ref[n]) for n in NS] for label, a in abl.items()]
    grid(f"MLP ablations, on folds {', '.join(str(f + 1) for f in folds)} of {K} (mean ± sd)", rows)


if __name__ == "__main__":
    main()
