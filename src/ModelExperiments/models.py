"""The models: the bit weights, the polynomial kernels, the paper's baseline, and make_estimator,
which builds any of them by name; plus cross-validating one setting of a model."""

import hashlib
import itertools
import json
import time
from fractions import Fraction
from functools import lru_cache
from math import comb

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils import CV, FOLDS, G, K, SEED, WTS, cv_eval, load, run


# ---- Bit weights: one random forest per set of training rows. A bit's weight is its permutation
# importance (Breiman, 2001): the drop in the forest's accuracy on a held-out quarter of those rows
# when that bit's values are shuffled; clipped at 0 and scaled so the largest weight is 1.
def bit_weights(B, y, seed=SEED):
    fit, val = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
    rf = RandomForestClassifier(
        G["RF_TREES"], max_features=1.0, min_samples_leaf=5, n_jobs=-1, random_state=seed
    ).fit(B[fit], y[fit])
    # one process per shuffled bit; a forest that also split each small prediction across all
    # cores spends most of its time on that overhead. Same result either way.
    rf.set_params(n_jobs=1)
    imp = permutation_importance(
        rf, B[val], y[val], n_repeats=5, random_state=seed, n_jobs=-1
    ).importances_mean
    w = np.clip(imp, 0, None)
    return (w / w.max() if w.max() > 0 else np.ones_like(w)).tolist()


def get_weights(B, y, seed=SEED):
    """bit_weights, stored under a hash of the rows so the same rows are processed once."""
    key = hashlib.md5(np.ascontiguousarray(B).tobytes() + y.astype(np.int8).tobytes()).hexdigest()
    key += "" if seed == SEED else f"|seed{seed}"

    def job():
        t0 = time.time()
        w = bit_weights(B, y, seed)
        return dict(w=w, mean=0.0, sd=0.0, secs=round(time.time() - t0, 1))

    row = run(WTS, key, job, quiet=True, exp="WTS", n=B.shape[1], rows=len(y), seed=seed)
    return np.array(row["w"])


def fold_weights(n, seed=SEED):
    """The bit weights on the training rows of every fold."""
    X, B, y = load(n)
    return [get_weights(B[tr], y[tr], seed) for tr, _ in FOLDS[n]]


def top_bits(w, k=10):
    return sorted(np.argsort(-np.asarray(w))[:k].tolist())


# ---- Models


@lru_cache(None)
def paper_kernel_table(n, degree):
    """phi(x).phi(x') for PolynomialFeatures(degree) on +-1 bits, by the number m of agreeing bits,
    scaled so that k(x, x) = 1. This is the mock paper's kernel; exact integer arithmetic keeps
    degrees above 10 accurate."""
    tab = [0] * (n + 1)
    for s in range(min(degree, n) + 1):
        w = comb(n + (degree - s) // 2, n)
        for m in range(n + 1):
            tab[m] += w * sum(
                comb(m, a) * comb(n - m, s - a) * (-1) ** (s - a) for a in range(s + 1)
            )
    return np.array([float(Fraction(t, tab[n])) for t in tab])


class KernelPolyLR:
    """Logistic regression with the paper's polynomial kernel (that of PolynomialFeatures(d) on the
    bits), trained in the dual so the up to 2^n monomials are never built, only the N x N Gram
    matrix. The penalty a'Ka / (2C) is sklearn's L2 penalty on the implicit weights, so C means
    what it means in LogisticRegression. L-BFGS instead of the paper's Newton step allows
    degrees above 10."""

    def __init__(self, deg, C, maxiter=500):
        self.deg, self.C, self.maxiter = deg, C, maxiter

    def _gram(self, A, B):
        n = A.shape[1]
        tab = paper_kernel_table(n, self.deg)
        A = np.asarray(A, np.float32)
        B = np.asarray(B, np.float32)
        K = np.empty((len(A), len(B)))
        for i in range(0, len(A), 1000):  # row blocks: no large temporaries
            K[i : i + 1000] = tab[np.rint((n + A[i : i + 1000] @ B.T) / 2).astype(np.int64)]
        return K

    def fit(self, B, y):
        from scipy.optimize import minimize
        from scipy.special import expit

        N = len(y)
        K = self._gram(B, B)
        s = 2.0 * y - 1
        C = self.C

        def fg(p):
            a, b = p[:N], p[N]
            Ka = K @ a
            m = -s * (Ka + b)
            g = -s * expit(m)  # derivative of the logistic loss in f
            return np.logaddexp(0, m).sum() + 0.5 / C * (a @ Ka), np.append(K @ g + Ka / C, g.sum())

        r = minimize(
            fg,
            np.zeros(N + 1),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.maxiter, maxcor=20),
        )
        self.a_, self.b_, self.B_, self.iters_ = (
            r.x[:N],
            r.x[N],
            np.asarray(B, np.float32),
            int(r.nit),
        )
        self.f_train_ = K @ self.a_ + self.b_
        return self

    def decision_function(self, B):
        B = np.asarray(B, np.float32)
        # the training rows: do not rebuild the Gram matrix
        if B.shape == self.B_.shape and np.array_equal(B, self.B_):
            return self.f_train_
        return self._gram(B, self.B_) @ self.a_ + self.b_

    def predict(self, B):
        return (self.decision_function(B) > 0).astype(np.int64)


class WeightedKernelLR:
    """Logistic regression with the polynomial kernel k(x, x') = ((1 + sum_i w_i x_i x'_i) /
    (1 + sum_i w_i))^d, trained in the dual like KernelPolyLR. weights='forest': w is computed
    inside fit() from the rows it is fitted on (bit_weights), so it never sees validation or
    test rows; 'equal': w = 1 for every bit; 'reversed': the forest's weights moved to the
    wrong bits (the largest weight on the least important bit), a control. The Gram matrix is
    kept in float32 so that 32,000 rows fit in memory (4 GB). L-BFGS stops when the loss improves
    by less than a relative 1e-6 per step (ftol)."""

    def __init__(self, deg, C, weights="forest", maxiter=3000):
        self.deg, self.C, self.weights, self.maxiter = deg, C, weights, maxiter

    def _gram(self, A, B):
        A = np.asarray(A, np.float32) * self.w_.astype(np.float32)
        B = np.asarray(B, np.float32)
        K = np.empty((len(A), len(B)), np.float32)
        z = np.float32(1 + self.w_.sum())
        for i in range(0, len(A), 1000):  # row blocks: no large temporaries
            K[i : i + 1000] = ((1 + A[i : i + 1000] @ B.T) / z) ** self.deg
        return K

    def fit(self, B, y):
        from scipy.optimize import minimize
        from scipy.special import expit

        if self.weights == "equal":
            self.w_ = np.ones(B.shape[1])
        else:
            w = get_weights(B, y)
            if self.weights == "reversed":
                order = np.argsort(w)
                w2 = np.empty_like(w)
                w2[order] = np.sort(w)[::-1]
                w = w2
            self.w_ = np.asarray(w, np.float64)
        N = len(y)
        K = self._gram(B, B)
        s = 2.0 * y - 1
        C = self.C

        def fg(p):
            a, b = p[:N], p[N]
            Ka = (K @ a.astype(np.float32)).astype(np.float64)
            m = -s * (Ka + b)
            g = -s * expit(m)
            grad = (K @ g.astype(np.float32)).astype(np.float64) + Ka / C
            return np.logaddexp(0, m).sum() + 0.5 / C * (a @ Ka), np.append(grad, g.sum())

        r = minimize(
            fg,
            np.zeros(N + 1),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.maxiter, ftol=1e-6),
        )
        self.a_, self.b_, self.iters_ = r.x[:N].astype(np.float32), float(r.x[N]), int(r.nit)
        self.B_ = np.asarray(B, np.float32)
        self.f_train_ = (K @ self.a_).astype(np.float64) + self.b_
        return self

    def decision_function(self, B):
        B = np.asarray(B, np.float32)
        if B.shape == self.B_.shape and np.array_equal(B, self.B_):
            return self.f_train_
        out = np.empty(len(B))
        for i in range(0, len(B), 2000):
            out[i : i + 2000] = self._gram(B[i : i + 2000], self.B_) @ self.a_ + self.b_
        return out

    def predict(self, B):
        return (self.decision_function(B) > 0).astype(np.int64)


class PaperBaseline:
    """The mock paper's model: PolynomialFeatures(2) on the raw inputs, logistic regression, C =
    0.85."""

    def fit(self, X, y):
        self.pf_ = PolynomialFeatures(2)
        self.m_ = LogisticRegression(max_iter=100, solver="sag", C=0.85, random_state=SEED).fit(
            self.pf_.fit_transform(X), y
        )
        return self

    def predict(self, X):
        return self.m_.predict(self.pf_.transform(X))


def make_estimator(family, p, cond="all"):
    """Build any model. p: JSON-friendly hyper-parameters. cond is 'all' (every model uses all
    bits) or, for the weighted kernel, 'reversed' (the control with weights on the wrong bits)."""
    if family == "lr_l2":  # LR (L2 penalty) on all products of up to deg distinct bits
        est = Pipeline(
            [
                ("poly", PolynomialFeatures(p["deg"], interaction_only=True, include_bias=False)),
                ("lr", LogisticRegression(C=p["C"], max_iter=300)),
            ]
        )
    elif family == "kpoly":
        est = KernelPolyLR(p["deg"], p["C"])
    elif family == "wkpoly":
        est = WeightedKernelLR(p["deg"], p["C"], "reversed" if cond == "reversed" else "forest")
    elif family == "ukpoly":
        est = WeightedKernelLR(p["deg"], p["C"], "equal")
    elif family == "rf":
        est = RandomForestClassifier(
            G["RF_TREES"],
            max_features=p["max_features"],
            min_samples_leaf=p["msl"],
            n_jobs=-1,
            random_state=SEED,
        )
    elif family == "mlp":
        est = MLPClassifier(
            tuple(p["h"]),
            alpha=G["MLP_ALPHA"],
            early_stopping=True,
            n_iter_no_change=G["MLP_PATIENCE"],
            max_iter=G["MLP_MAXITER"],
            random_state=SEED,
        )
    elif family == "paper":
        est = PaperBaseline()
    else:
        raise ValueError(family)
    return est


def test_kernel():
    """(a) The Gram matrix of all 2^6 points has full rank exactly when d >= n. (b) The kernel is
    the inner product of PolynomialFeatures(d). (c) The dual solver agrees with sklearn on
    the explicit feature map of the same kernel."""
    P = np.array(list(itertools.product([-1.0, 1.0], repeat=6)))
    for d in (3, 5, 6, 7, 12):
        assert np.linalg.matrix_rank(KernelPolyLR(d, 1)._gram(P, P)) == min(
            64, sum(comb(6, s) for s in range(d + 1))
        ), f"rank at d={d}"
    Q = P[:, :5][::2]
    F = PolynomialFeatures(3).fit_transform(Q)
    E = F @ F.T
    assert np.allclose(
        KernelPolyLR(3, 1)._gram(Q, Q), E / np.diag(E)[0], atol=1e-9
    ), "kernel differs from PolynomialFeatures"
    r = np.random.default_rng(0)
    B = r.choice([-1.0, 1.0], (80, 8))
    y = ((B[:, 0] * B[:, 1] > 0) ^ (r.random(80) < 0.1)).astype(int)
    for d in (3, 8):
        m = KernelPolyLR(d, 0.85, maxiter=5000).fit(B, y)
        w, U = np.linalg.eigh(m._gram(B, B))
        k = w > 1e-12 * w.max()
        F = U[:, k] * np.sqrt(w[k])
        sk = LogisticRegression(C=0.85, tol=1e-10, max_iter=20000).fit(F, y)
        assert (
            np.abs(m.f_train_ - (F @ sk.coef_[0] + sk.intercept_[0])).max() < 1e-2
        ), f"dual solver differs from sklearn at d={d}"
    print("kernel tests passed")


# ---- Cross-validating one setting of a model
def cv_config(n, exp, family, cond, sub, p, **kw):
    """Cross-validate one model setting, unless it is already stored."""
    key = f"{exp}|{n}|{family}|{cond}|{sub}|{json.dumps(p, sort_keys=True)}"
    return run(
        CV,
        key,
        lambda: cv_eval(lambda: make_estimator(family, p, cond), n, fam=family, **kw),
        exp=exp,
        n=n,
        family=family,
        cond=cond,
        sub=sub,
        params=p,
    )


def cv_by_fold(n, family, p, sub):
    """Like cv_config for a slow model: every fold is stored when it finishes, then combined."""
    key = f"{family}|{n}|{sub}|{json.dumps(p, sort_keys=True)}"
    meta = dict(n=n, family=family, cond="all", params=p)
    parts = [
        run(
            CV,
            f"{key}|fold{f}",
            lambda: cv_eval(lambda: make_estimator(family, p), n, folds=(f,)),
            exp="FOLD",
            sub=f"{sub}, fold {f + 1}",
            **meta,
        )
        for f in range(K)
    ]

    def combine():
        a = [r["folds"][0] for r in parts]
        return dict(
            folds=a,
            mean=float(np.mean(a)),
            sd=float(np.std(a, ddof=1)),
            train=float(np.mean([r["train"] for r in parts])),
            capped=sum(r.get("capped", 0) for r in parts),
            secs=0.0,
        )

    return run(CV, key, combine, quiet=True, exp="MLP", sub=sub, **meta)
