"""The final model per n, chosen by CV; its controls; the test score of every model's chosen
setting (the report's Table 1); and the hidden-set predictions. Run after exp1 to exp4.

    python final_pipeline.py"""

import json
import os
import sys
import time

import numpy as np

from models import make_estimator, top_bits
from utils import (
    CV,
    DATA,
    DEV,
    FIN,
    HIDDEN,
    LOCKED,
    NAME,
    NS,
    ROOT,
    WRITE_PREDICTIONS,
    G,
    acc,
    best_of,
    check_caps,
    cv_eval,
    grid,
    groups,
    hoeffding_eps,
    kname,
    load,
    ms,
    pct,
    run,
    show_params,
)


def main():
    final = choose_final()
    run_controls(final)
    table_final(final)
    check_caps()
    table1(final)
    predict_hidden(final)


def choose_final():
    """The final model for each n: the candidate with the highest CV mean (the paper's baseline is
    not a candidate)."""
    return {n: [g for g in groups(n) if g["key"][0] != "paper"][0] for n in NS}


def test_score(n, g):
    """Refit one CV-chosen setting g (a group from groups/best_of) on all dev rows and score it
    once on the locked rows (stored; later calls read it back). Nothing is chosen with these
    scores: the final pipeline is picked by CV alone."""
    fam, cond, sub = g["key"]
    p = g["best"]["params"]
    X, B, y = load(n)
    M = X if fam == "paper" else B  # the paper baseline uses the raw inputs, as in CV
    tr, te = DEV[n], LOCKED[n]

    def job():
        t0 = time.time()
        est = make_estimator(fam, p, cond).fit(M[tr], y[tr])
        a = acc(est, M[te], y[te])
        return dict(
            mean=a,
            sd=0.0,
            secs=round(time.time() - t0, 1),
            n_test=int(len(te)),
            train=acc(est, M[tr], y[tr]),
            top_bits=top_bits(est.w_) if hasattr(est, "w_") else None,
            params=p,
        )

    key = f"locked|{n}|{fam}|{cond}|{sub}|{json.dumps(p, sort_keys=True)}"
    return run(FIN, key, job, quiet=True, exp="TEST", n=n, family=fam, cond=cond, sub=sub)


def run_controls(final):
    """Two controls on the folds LC_FOLDS, both expected at about 0.5: the final model with
    permuted training labels (otherwise label information leaks), and MANBAT at its best setting
    with its weights moved to the wrong bits (otherwise the weights are not what makes it work)."""
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
        # MANBAT's best setting, whether or not it is the final model (if it is, the same model)
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
    """Refit the final model on all labelled rows and predict the hidden rows. As their labels are
    unknown, the predictions must agree with those of the best model of another kind that scores
    above 0.9 in CV: two models with error rates e1 and e2 disagree on at most e1 + e2 of the rows,
    so a lower agreement (with 2 points of slack for sampling noise) stops the notebook. Then
    hiddenlabels/y_predicted_{n}.npy are written and their format is checked with the course's
    check_submission.py."""
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
        msg = f"n = {n}: {kname(final[n]['key'])}, predicted positive rate {pct(pred.mean())}%"
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

            for name in [f"y_predicted_{n}.npy" for n in NS]:
                pr = CS.problems_with(os.path.join(HIDDEN, name))
                print(
                    f"{'OK  ' if not pr else 'FAIL'}  {name}" + (f": {'; '.join(pr)}" if pr else "")
                )
        except ImportError:
            print("check_submission.py not found at the repository root; run it from there instead")


def table_final(final):
    """The final model: CV, its test score with the interval of Eq. 7, and the two controls."""
    lk = {n: test_score(n, final[n]) for n in NS}
    eps = {n: hoeffding_eps(lk[n]["n_test"]) for n in NS}
    rows = [
        ["Final model (highest CV mean)"] + [NAME[final[n]["key"][0]] for n in NS],
        ["    setting"] + [show_params(final[n]["best"]["params"]) for n in NS],
        ["CV accuracy (mean ± sd over folds)"] + [ms(final[n]["best"]) for n in NS],
        ["Test accuracy"] + [pct(lk[n]["mean"]) for n in NS],
        ["    test rows m"] + [str(lk[n]["n_test"]) for n in NS],
        ["    epsilon (Eq. 7, delta = 0.05), points"] + [pct(eps[n]) for n in NS],
        ["    true accuracy at least, with prob. 95%"] + [pct(lk[n]["mean"] - eps[n]) for n in NS],
        ["Control: permuted training labels"] + [pct(FIN.rows[f"shuffle|{n}"]["mean"]) for n in NS],
        ["Control: MANBAT, weights on the wrong inputs"]
        + [pct(FIN.rows[f"reversed|{n}"]["mean"]) for n in NS],
    ]
    print()
    grid(
        "Final model: accuracy (%)",
        rows,
        f"Controls: mean over folds {', '.join(str(f + 1) for f in G['LC_FOLDS'])}, expected"
        " about 50%.",
    )


def table1_groups(n):
    """The rows of the report's Table 1: (label, the CV-best setting of that model)."""
    top = f"deg{G['KP_DEGS'][n][-1]}"
    rows = [
        ("Baseline (degree 2, raw inputs)", best_of(n, "all", ("paper",))),
        ("Their kernel, degree $2n$", best_of(n, "all", ("kpoly",), top)),
        ("Random forest", best_of(n, "all", ("rf",))),
    ]
    if CV.where(n=n, family="mlp", exp="MLP"):
        rows.append(("MLP", best_of(n, "all", ("mlp",))))
    rows += [
        ("Equal-weight kernel", best_of(n, "all", ("ukpoly",))),
        ("MANBAT", best_of(n, "all", ("wkpoly",))),
    ]
    return rows


def table1(final):
    """The report's Table 1, in percent: the CV mean ± sd of every model's CV-best setting; their
    kernel at degree 2n on its own training rows; and every model on the test set."""
    rows = {n: table1_groups(n) for n in NS}
    labels = [label.replace("$2n$", "2n") for label, _ in rows[NS[0]]]
    top = {n: f"deg{G['KP_DEGS'][n][-1]}" for n in NS}

    def block(title, cell):
        return [[title] + [""] * len(NS)] + [
            [f"    {label}"] + [cell(n, rows[n][i][1]) for n in NS]
            for i, label in enumerate(labels)
        ]

    out = block("Cross-validation (mean ± sd over folds)", lambda n, g: ms(g["best"]))
    out += [["Their kernel, degree 2n, on its training rows"]]
    out[-1] += [pct(best_of(n, "all", ("kpoly",), top[n])["best"]["train"]) for n in NS]
    out += block("Test set", lambda n, g: pct(test_score(n, g)["mean"]))
    grid("Table 1: accuracy (%)", out)


if __name__ == "__main__":
    main()
