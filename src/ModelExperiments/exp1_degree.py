"""Experiment 1 (claim 1): polynomial logistic regression on all bits, by degree. Explicit
monomials up to the largest degree that fits in memory, then the paper's kernel up to degree 2n.

    python exp1_degree.py"""

from models import cv_config, test_kernel
from utils import G, NS, _mean_sd, best_of, grid, kernel_label


def run():
    test_kernel()
    for n in NS:
        cv_config(n, "BASE", "paper", "all", "deg2", dict(deg=2, C=0.85))
        for d in range(1, G["MAX_DEG"][n] + 1):
            for C in G["C_L2"]:
                cv_config(n, "LR", "lr_l2", "all", f"deg{d}", dict(deg=d, C=C))
        for d in G["KP_DEGS"][n]:
            cv_config(n, "KPOLY", "kpoly", "all", f"deg{d}", dict(deg=d, C=G["KP_C"]))


def table_degree_all():
    """Polynomial LR on all bits by degree: explicit monomials, then the paper's kernel."""
    blank = [""] * (len(NS) + 1)
    rows = [["Explicit monomials (best C per degree)"] + [""] * len(NS)]
    for d in range(1, max(G["MAX_DEG"].values()) + 1):
        rows.append([f"    degree {d}"] + [_mean_sd(n, "all", ("lr_l2",), f"deg{d}") for n in NS])
    rows += [blank, [f"The paper's kernel (C = {G['KP_C']})"] + [""] * len(NS)]
    last = len(G["KP_DEGS"][NS[0]]) - 1
    for i in range(last + 1):
        subs = {n: f"deg{G['KP_DEGS'][n][i]}" for n in NS}
        rows.append(
            [f"    degree {kernel_label(i)}"]
            + [_mean_sd(n, "all", ("kpoly",), subs[n]) for n in NS]
        )
    train = [
        best_of(n, "all", ("kpoly",), f"deg{G['KP_DEGS'][n][last]}")["best"]["train"] for n in NS
    ]
    rows.append(
        [f"        training accuracy at degree {kernel_label(last)}"] + [f"{t:.3f}" for t in train]
    )
    grid(
        "Polynomial LR on all bits: CV accuracy by degree (mean ± sd over folds)",
        rows,
        "- : not run, too many monomials to fit in memory.",
    )


if __name__ == "__main__":
    run()
    table_degree_all()
