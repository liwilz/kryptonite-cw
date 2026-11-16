"""Experiment 3 (H2): MANBAT, the polynomial kernel with each bit weighted by a random forest's
permutation importance, against the same kernel with every weight 1.

    python exp3_weighted_kernel.py"""

import numpy as np

from models import cv_config, fold_weights, top_bits
from utils import NS, G, best_of, grid, mean_sd


def main():
    run()
    table_weights()
    table_kernels()


def run():
    for n in NS:  # the bit weights of every fold first, so their energy is recorded on its own
        fold_weights(n)
    for n in NS:
        for d in G["WK_DEGS"]:
            cv_config(n, "UK", "ukpoly", "all", f"deg{d}", dict(deg=d, C=G["WK_UNIFORM_C"]))
            for C in G["WK_C"]:
                cv_config(n, "WK", "wkpoly", "all", f"deg{d}", dict(deg=d, C=C))


def table_weights():
    """How clearly the forest's weights single out a few inputs."""
    W = {n: fold_weights(n) for n in NS}

    def mean(f):
        return [f"{100 * np.mean([f(np.sort(w)[::-1]) for w in W[n]]):.1f}%" for n in NS]

    rows = [
        ["10 largest weights on the same inputs in every fold"]
        + ["yes" if len({tuple(top_bits(w)) for w in W[n]}) == 1 else "no" for n in NS],
        ["Share of the total weight on those 10 inputs"] + mean(lambda w: w[:10].sum() / w.sum()),
        ["Inputs with weight 0 (left out of the kernel)"]
        + [f"{np.mean([(w == 0).sum() for w in W[n]]):.1f}" for n in NS],
    ]
    grid(
        "MANBAT's input weights: one random forest per training fold",
        rows,
        "Negative importances are set to 0, then the largest weight is scaled to 1; mean over"
        " folds.",
    )
    for n in NS:
        print(f"n = {n}: the 10 inputs: {', '.join(map(str, top_bits(W[n][0])))}")


def table_kernels():
    """The same polynomial kernel with equal and with forest weights, by degree."""
    rows = [[f"Equal weights (C = {G['WK_UNIFORM_C']})"] + [""] * len(NS)]
    rows += [
        [f"    degree {d}"] + [mean_sd(n, "all", ("ukpoly",), f"deg{d}") for n in NS]
        for d in G["WK_DEGS"]
    ]
    rows += [["MANBAT, forest weights (best C per degree)"] + [""] * len(NS)]
    rows += [
        [f"    degree {d}"] + [mean_sd(n, "all", ("wkpoly",), f"deg{d}") for n in NS]
        for d in G["WK_DEGS"]
    ]
    best = {n: best_of(n, "all", ("wkpoly",))["best"]["params"] for n in NS}
    rows.append(["    chosen: degree, C"] + [f"{best[n]['deg']}, {best[n]['C']}" for n in NS])
    grid(
        "MANBAT and the equal-weight kernel: CV accuracy by degree (%, mean ± sd over folds)", rows
    )


if __name__ == "__main__":
    main()
