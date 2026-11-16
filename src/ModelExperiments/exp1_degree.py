"""Experiment 1 (H1): the paper's baseline, and the paper's polynomial kernel at degree 10 (the
highest it reports) and at degree 2n (where it can represent every function of the bits), on
training and validation rows.

    python exp1_degree.py"""

from models import cv_config, test_kernel
from utils import NS, G, best_of, grid, mean_sd, pct


def main():
    run()
    table()


def run():
    test_kernel()  # our kernel equals the paper's, and our solver agrees with scikit-learn
    for n in NS:
        cv_config(n, "BASE", "paper", "all", "deg2", dict(deg=2, C=0.85))
        for d in G["KP_DEGS"][n]:
            cv_config(n, "KPOLY", "kpoly", "all", f"deg{d}", dict(deg=d, C=G["KP_C"]))


def table():
    """CV accuracy of the baseline and of the kernel, and the kernel's training accuracy."""
    top = {n: f"deg{G['KP_DEGS'][n][-1]}" for n in NS}
    rows = [
        ["Baseline (degree 2, raw inputs)"] + [mean_sd(n, "all", ("paper",)) for n in NS],
        ["Their kernel, degree 10"] + [mean_sd(n, "all", ("kpoly",), "deg10") for n in NS],
        ["Their kernel, degree 2n"] + [mean_sd(n, "all", ("kpoly",), top[n]) for n in NS],
        ["    training accuracy at degree 2n"]
        + [pct(best_of(n, "all", ("kpoly",), top[n])["best"]["train"]) for n in NS],
    ]
    grid(f"CV accuracy (%, mean ± sd over folds); the kernel uses C = {G['KP_C']}", rows)


if __name__ == "__main__":
    main()
