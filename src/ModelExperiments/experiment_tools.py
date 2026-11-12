"""Helpers for Experiments.ipynb: settings, data splits, the bit selector, the models, result
storage, tables and figures. The notebook calls these; the experiments themselves are in it."""

import hashlib
import itertools
import json
import os
import re
import sys
import time
import warnings
from fractions import Fraction
from functools import lru_cache
from math import comb

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sklearn
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

# ---- Settings. KRYP_QUICK=1 runs a tiny dry run (see the end of this block); its numbers mean
# nothing.
QUICK = os.environ.get("KRYP_QUICK", "0") == "1"
ROOT = next((p for p in (".", "..", "../..", "../../..") if os.path.isdir(f"{p}/Datasets")), ".")
DATA, HIDDEN = f"{ROOT}/Datasets", f"{ROOT}/hiddenlabels"
NS = (20, 30, 40)
SEED = 0  # fixes the locked split, the folds and every model
K = 5  # CV folds
ROWS = None  # None = all labelled rows
OUT, FIG = "exp_results", "exp_figures"
WRITE_PREDICTIONS = True
RUN_KRYPTONET_SIZED = True  # the 4 x 512 MLP, on fold 0 only because it is slow

# Search grids. Changing one only adds new results to OUT/; stored results of other settings stay
# valid.
G = dict(
    C_L2=[1e-3, 1e-2, 1e-1, 1],
    C_L1=[3e-3, 1e-2, 3e-2, 1e-1],
    MAX_DEG={20: 4, 30: 3, 40: 2},  # largest explicit polynomial degree that fits in memory
    DT_DEPTH=[2, 4, 6, 8, 12, None],
    DT_LEAF=[1, 10],
    RF_TREES=200,
    RF_MAXF=["sqrt", 0.5, 1.0],
    RF_LEAF=[1, 5],
    HGB_DEPTH=[3, 6, None],
    KM_K=[8, 32, 128],
    MLP_H=[[128, 128]],
    KP_DEGS={20: [10, 20, 40], 30: [10, 30, 60], 40: [10, 40, 80]},  # degree 10, n and 2n
    KP_C=0.85,
    MLP_ALPHA=[1e-2],
    MLP_MAXITER=400,
    KN_MAXITER=60,  # epoch cap of the 4 x 512 MLP
    KN_PATIENCE=15,
    MLP_PATIENCE=50,
    SEL_RF_MAXF=["sqrt", 1.0],
    SEL_RF_LEAF=[1, 5],
    LC_SIZES=[250, 500, 1000, 2000, 4000],  # learning curves; the full fold-train set is added
    LC_FOLDS=(0, 1, 2),
    LC_TOP=3,
    NULL_REPS=20,  # shuffled-label runs of the selector per n
)
if QUICK:  # dry run: 3 folds, 3,000 rows, small grids, separate folders, no predictions
    K, ROWS, OUT, FIG, WRITE_PREDICTIONS = 3, 3000, "exp_results_quick", "exp_figures_quick", False
    G.update(
        C_L2=[1e-2, 1e-1],
        C_L1=[1e-2, 1e-1],
        MAX_DEG={20: 2, 30: 2, 40: 2},
        DT_DEPTH=[4, None],
        RF_TREES=30,
        RF_MAXF=["sqrt", 1.0],
        HGB_DEPTH=[3, None],
        KM_K=[4, 16],
        MLP_H=[[64, 64]],
        KP_DEGS={20: [10, 20], 30: [10, 30], 40: [10, 40]},
        MLP_MAXITER=40,
        KN_MAXITER=40,
        KN_PATIENCE=10,
        MLP_PATIENCE=10,
        LC_SIZES=[100, 250, 500],
        LC_FOLDS=(0, 1),
        LC_TOP=2,
        NULL_REPS=3,
    )
if not os.path.isfile(f"{DATA}/kryptonite-20-X.npy"):
    raise FileNotFoundError(
        f"{DATA}/kryptonite-20-X.npy not found: run from a folder inside the repository"
    )
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/.gitignore", "w") as f:  # stored results only let an interrupted run resume
    f.write("*\n")
os.makedirs(FIG, exist_ok=True)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul")


# ---- Data and splits: a locked 20% test set and K stratified folds on the other 80% (dev)


# ---- Names used in tables and figures
NAME = dict(
    lr_l2="LR, L2",
    lr_l1="LR, L1",
    kpoly="Kernel LR",
    lrfull="LR on all monomials",
    dtree="Decision tree",
    rf="Random forest",
    hgb="Gradient boosting",
    mlp="MLP 2×128",
    kmeans_major="K-means (cluster majority)",
    kmeans_onehot="K-means (one-hot + LR)",
    paper="Paper baseline",
)
SHORT = dict(
    NAME,
    rf="Forest",
    hgb="Boosting",
    mlp="MLP",
    kmeans_major="K-means",
    kmeans_onehot="K-means + LR",
)


def kname(key):
    """Full name of a (family, condition, sub-model) key."""
    fam, cond, sub = key
    return (
        NAME[fam]
        + (f", degree {sub[3:]}" if sub.startswith("deg") else "")
        + (" on selected bits" if cond != "all" else "")
    )


def sname(key):
    """Short name for figure axes."""
    fam, cond, sub = key
    s = SHORT[fam] + (f", degree {sub[3:]}" if sub.startswith("deg") and fam != "paper" else "")
    return s + (
        ", selected bits"
        if cond != "all"
        else (", all bits" if fam in ("rf", "dtree", "hgb") else "")
    )


def show_params(p):
    """{"max_features": 1.0, "msl": 5} -> "max features 1.0, min leaf 5"."""
    names = dict(
        msl="min leaf",
        max_features="max features",
        max_depth="max depth",
        deg="degree",
        pen="penalty",
        h="layers",
    )
    out = []
    for k, v in (p or {}).items():
        v = f"{len(v)}×{v[0]}" if k == "h" and len(set(v)) == 1 else ("none" if v is None else v)
        out.append(f"{names.get(k, k)} {v}")
    return ", ".join(out)


_DATA = {}


def load(n):
    """X (noisy), B (the bits sign(X), +-1) and y. The noise (sd 0.1) is far too small to flip a
    sign."""
    if n not in _DATA:
        X = np.load(f"{DATA}/kryptonite-{n}-X.npy")
        y = np.load(f"{DATA}/kryptonite-{n}-y.npy").astype(np.int64)
        if ROWS:
            keep, _ = train_test_split(
                np.arange(len(y)), train_size=ROWS, stratify=y, random_state=0
            )
            X, y = X[keep], y[keep]
        _DATA[n] = (X, np.where(X > 0, 1.0, -1.0).astype(np.float32), y)
    return _DATA[n]


DEV, FOLDS, LOCKED = {}, {}, {}
for n in NS:
    X, B, y = load(n)
    dev, locked = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    DEV[n], LOCKED[n] = dev, locked  # LOCKED is read only by the final test
    FOLDS[n] = [
        (dev[a], dev[b])
        for a, b in StratifiedKFold(K, shuffle=True, random_state=SEED).split(dev, y[dev])
    ]
FOLD_TRAIN = {n: len(FOLDS[n][0][0]) for n in NS}


def check_protocol():
    """Assert that the splits are valid and print their sizes."""
    for n in NS:
        X, B, y = load(n)
        assert set(np.unique(y)) <= {0, 1}, "labels must be 0/1"
        assert np.mean(np.abs(X - B) < 0.5) > 0.9999, "rounding to bits would lose information"
        assert not set(DEV[n]) & set(LOCKED[n]) and len(DEV[n]) + len(LOCKED[n]) == len(y)
        assert sorted(np.concatenate([va for _, va in FOLDS[n]])) == sorted(DEV[n])
        for tr, va in FOLDS[n]:
            assert (
                not set(tr) & set(va)
                and not set(tr) & set(LOCKED[n])
                and not set(va) & set(LOCKED[n])
            )
            assert abs(y[tr].mean() - y.mean()) < 0.02 and abs(y[va].mean() - y.mean()) < 0.03
        print(
            f"n={n}: {len(y)} labelled rows = {len(DEV[n])} dev ({K} folds of {FOLD_TRAIN[n]} train"
            f" / {len(FOLDS[n][0][1])} validation rows) + {len(LOCKED[n])} locked; {y.mean():.3f}"
            " positive"
        )


Z_FLOOR = 4.0  # only coefficients above this many standard errors are stored


@lru_cache(None)
def _triu(m):
    return np.triu_indices(m, 1)


def scan_core(R, z, maxdeg):
    """mean(z * prod(R[:, T])) for every subset T of the columns of R with |T| = 1..maxdeg (at most
    4), via matrix products. Returns {k: (subsets as an array of column indices, coefficients)}.
    """
    N, m = R.shape
    R = np.ascontiguousarray(R, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    assert maxdeg <= 4
    out = {1: (np.arange(m)[:, None], (R.T @ z) / N)}
    if maxdeg >= 2:
        M = (R * z[:, None]).T @ R / N
        iu = _triu(m)
        out[2] = (np.stack(iu, 1), M[iu])
    if maxdeg >= 3:
        cs, vs = [], []
        for i in range(m - 2):
            Rs = R[:, i + 1 :]
            M = (Rs * (z * R[:, i])[:, None]).T @ Rs / N
            a, b = _triu(m - i - 1)
            cs.append(np.stack([np.full(len(a), i), a + i + 1, b + i + 1], 1))
            vs.append(M[a, b])
        out[3] = (np.concatenate(cs), np.concatenate(vs))
    if maxdeg >= 4:
        cs, vs = [], []
        for i in range(m - 3):
            for j in range(i + 1, m - 2):
                Rs = R[:, j + 1 :]
                M = (Rs * (z * R[:, i] * R[:, j])[:, None]).T @ Rs / N
                a, b = _triu(m - j - 1)
                cs.append(
                    np.stack([np.full(len(a), i), np.full(len(a), j), a + j + 1, b + j + 1], 1)
                )
                vs.append(M[a, b])
        out[4] = (np.concatenate(cs), np.concatenate(vs))
    return out


def scan_bits(B, y, maxdeg=4):
    """Steps 1 and 2: the gate, its coefficient, and every scanned coefficient above Z_FLOOR
    standard errors."""
    N, n = B.shape
    s = (2.0 * y - 1).astype(np.float32)
    C = (B * s[:, None]).T @ B / N
    np.fill_diagonal(C, 0.0)
    a, b = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    pair = (int(min(a, b)), int(max(a, b)))
    rest = np.array([i for i in range(n) if i not in pair])
    res = scan_core(B[:, rest], s * B[:, pair[0]] * B[:, pair[1]], maxdeg)
    thr = Z_FLOOR / np.sqrt(N)
    hits = []
    for k, (cm, cf) in res.items():
        for i in np.where(np.abs(cf) > thr)[0]:
            hits.append([[int(rest[q]) for q in cm[i]], float(cf[i])])
    return dict(pair=list(pair), gate_coef=float(C[a, b]), hits=hits, N=int(N), n=int(n))


def choose_bits(res, max_bits=14, mass_thr=0.02):
    """Step 3: the gate pair plus the bits with the most mass."""
    N, n, pair = res["N"], res["n"], res["pair"]
    sq = np.sqrt(N)
    mass = np.zeros(n)
    for T, c in res["hits"]:
        if abs(c) > 4.0 / sq:
            for t in T:
                mass[t] += c * c
    ranked = [int(i) for i in np.argsort(-mass) if mass[i] > mass_thr and i not in pair]
    return dict(
        bits=sorted(pair + ranked[: max_bits - 2]),
        pair=pair,
        mass=[round(float(v), 5) for v in mass],
        gate_z=abs(res["gate_coef"]) * sq,
    )


_SCAN_PATH = f"{OUT}/scan_cache.json"
_SCAN = json.load(open(_SCAN_PATH)) if os.path.exists(_SCAN_PATH) else {}


def get_scan(B, y):
    """scan_bits with an on-disk cache keyed by the content of (B, y), so the same training rows
    are never scanned twice."""
    key = (
        hashlib.md5(np.ascontiguousarray(B).tobytes() + y.astype(np.int8).tobytes()).hexdigest()[
            :16
        ]
        + f"|{B.shape[0]}x{B.shape[1]}|z{Z_FLOOR}"
    )
    if key not in _SCAN:
        _SCAN[key] = scan_bits(B, y)
        json.dump(_SCAN, open(_SCAN_PATH, "w"))
    return _SCAN[key]


class BitModel:
    """Wraps any estimator: scan and select on the rows it is fitted on, then fit the estimator on
    those columns only. If nothing is selected it predicts the training majority class."""

    def __init__(self, base):
        self.base = base

    def fit(self, B, y):
        self.scan_ = get_scan(B, y)
        sel = choose_bits(self.scan_)
        self.cols_, self.pair_, self.mass_, self.gate_z_ = (
            sel["bits"],
            sel["pair"],
            sel["mass"],
            sel["gate_z"],
        )
        self.const_ = None if self.cols_ else int(y.mean() >= 0.5)
        if self.cols_:
            self.base.fit(B[:, self.cols_], y)
        return self

    def predict(self, B):
        return (
            np.full(len(B), self.const_)
            if self.const_ is not None
            else self.base.predict(B[:, self.cols_])
        )


def _brute_bits(B, y, maxdeg=4, mass_thr=0.02, max_bits=14):
    """A plain loop over all subsets: a slow reference implementation to test scan_bits against."""
    N, n = B.shape
    s = 2.0 * y - 1
    C = (B * s[:, None]).T @ B / N
    np.fill_diagonal(C, 0.0)
    a, b = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    pair = (int(min(a, b)), int(max(a, b)))
    z = (s * B[:, pair[0]] * B[:, pair[1]]).astype(np.float32)
    rest = [i for i in range(n) if i not in pair]
    mass, thr = np.zeros(n), 4.0 / np.sqrt(N)
    for k in range(1, maxdeg + 1):
        for c in itertools.combinations(rest, k):
            coef = float((z * np.prod(B[:, list(c)], axis=1)).mean())
            if abs(coef) > thr:
                for t in c:
                    mass[t] += coef**2
    ranked = [int(i) for i in np.argsort(-mass) if mass[i] > mass_thr and i not in pair]
    return sorted(list(pair) + ranked[: max_bits - 2]), mass


def self_test():
    """The fast scan agrees with the plain loop on a planted example, and a label that is one pair
    gives exactly that pair."""
    rng = np.random.default_rng(1)
    n, N = 12, 700
    B = rng.choice([-1.0, 1.0], (N, n)).astype(np.float32)
    f = (
        B[:, 0] * B[:, 1] * B[:, 2]
        + 0.8 * B[:, 0] * B[:, 1] * B[:, 5] * B[:, 6]
        + 0.6 * B[:, 0] * B[:, 1]
    )
    y = ((f + rng.normal(0, 0.3, N)) > 0).astype(int)
    bits_ref, mass_ref = _brute_bits(B, y)
    res = scan_bits(B, y)
    assert choose_bits(res)["bits"] == bits_ref, "fast scan disagrees with the plain loop"
    assert (
        np.abs(np.array(choose_bits(res)["mass"]) - mass_ref).max() < 1e-4
    ), "mass differs from the plain loop"
    assert set(res["pair"]) == {0, 1} and {0, 1, 2, 5, 6} <= set(
        bits_ref
    ), "planted gate terms not found"
    B2 = np.random.default_rng(2).choice([-1.0, 1.0], (7680, 20)).astype(np.float32)
    y2 = ((B2[:, 3] * B2[:, 5] > 0) ^ (np.random.default_rng(3).random(7680) < 0.03)).astype(int)
    r2 = scan_bits(B2, y2)
    assert r2["pair"] == [3, 5] and choose_bits(r2)["bits"] == [3, 5]
    print("selector tests passed")


# penalty= is deprecated since scikit-learn 1.8 in favour of l1_ratio=
NEW_SKLEARN = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)


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


class FullPolyLR:
    """Logistic regression on all 2^k - 1 products of distinct bits of its k input columns (a
    polynomial of degree k, so any function of those k bits). Meant for small k: it runs behind
    the bit selector, which keeps k <= 14."""

    def __init__(self, pen, C):
        self.pen, self.C = pen, C

    def fit(self, B, y):
        self.pf_ = PolynomialFeatures(B.shape[1], interaction_only=True, include_bias=False).fit(B)
        self.lr_ = lr_clf(self.pen, self.C).fit(self.pf_.transform(B), y)
        return self

    def predict(self, B):
        return self.lr_.predict(self.pf_.transform(B))


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


class KMeansMajority:
    """K-means on the bits (fitted on the training rows only); each cluster predicts its training
    majority label."""

    def __init__(self, k):
        self.k = k

    def fit(self, B, y):
        self.km_ = KMeans(self.k, n_init=3, max_iter=100, random_state=SEED).fit(B)
        lab = self.km_.labels_
        cnt = np.bincount(lab, minlength=self.k)
        pos = np.bincount(lab, weights=y, minlength=self.k)
        self.vote_ = (np.where(cnt > 0, pos / np.maximum(cnt, 1), y.mean()) > 0.5).astype(np.int64)
        return self

    def predict(self, B):
        return self.vote_[self.km_.predict(B)]


class KMeansOneHot:
    """The one-hot cluster id is the only input of a logistic regression."""

    def __init__(self, k):
        self.k = k

    def fit(self, B, y):
        self.km_ = KMeans(self.k, n_init=3, max_iter=100, random_state=SEED).fit(B)
        self.lr_ = LogisticRegression(C=1.0, max_iter=300).fit(
            np.eye(self.k, dtype=np.float32)[self.km_.labels_], y
        )
        return self

    def predict(self, B):
        return self.lr_.predict(np.eye(self.k, dtype=np.float32)[self.km_.predict(B)])


def lr_clf(pen, C):
    if pen == "l2":
        return LogisticRegression(C=C, max_iter=300)
    if NEW_SKLEARN:
        return LogisticRegression(
            C=C, l1_ratio=1, solver="liblinear", max_iter=100, random_state=SEED
        )
    return LogisticRegression(
        C=C, penalty="l1", solver="liblinear", max_iter=100, random_state=SEED
    )


def make_estimator(family, p, cond="all"):
    """Build any model. p: JSON-friendly hyper-parameters. cond: 'all' bits, or 'sel' = the bit
    selector runs inside fit()."""
    if family in ("lr_l1", "lr_l2"):
        est = Pipeline(
            [
                ("poly", PolynomialFeatures(p["deg"], interaction_only=True, include_bias=False)),
                ("lr", lr_clf(family[-2:], p["C"])),
            ]
        )
    elif family == "dtree":
        est = DecisionTreeClassifier(
            max_depth=p["max_depth"], min_samples_leaf=p["msl"], random_state=SEED
        )
    elif family == "rf":
        est = RandomForestClassifier(
            G["RF_TREES"],
            max_features=p["max_features"],
            min_samples_leaf=p["msl"],
            n_jobs=-1,
            random_state=SEED,
        )
    elif family == "hgb":
        est = HistGradientBoostingClassifier(
            max_depth=p["max_depth"],
            max_iter=300,
            learning_rate=0.1,
            early_stopping=True,
            random_state=SEED,
        )
    elif family == "mlp":
        est = MLPClassifier(
            tuple(p["h"]),
            alpha=p["alpha"],
            early_stopping=True,
            n_iter_no_change=G["MLP_PATIENCE"],
            max_iter=G["MLP_MAXITER"],
            random_state=SEED,
        )
    elif family == "kmeans_major":
        est = KMeansMajority(p["k"])
    elif family == "kmeans_onehot":
        est = KMeansOneHot(p["k"])
    elif family == "lrfull":
        est = FullPolyLR(p["pen"], p["C"])
    elif family == "kpoly":
        est = KernelPolyLR(p["deg"], p["C"])
    elif family == "paper":
        est = PaperBaseline()
    else:
        raise ValueError(family)
    return BitModel(est) if cond == "sel" else est


def kryptonet_sized():
    """The 4 x 512 MLP (KryptoNet's shape), with a capped training budget."""
    return MLPClassifier(
        (512, 512, 512, 512),
        alpha=G["MLP_ALPHA"][0],
        early_stopping=True,
        n_iter_no_change=G["KN_PATIENCE"],
        max_iter=G["KN_MAXITER"],
        random_state=SEED,
    )


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


class Store:
    def __init__(self, name):
        self.path, self.rows = f"{OUT}/{name}.jsonl", {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line)
                self.rows[r["key"]] = r

    def where(self, **kw):
        return [r for r in self.rows.values() if all(r.get(k) == v for k, v in kw.items())]


CV, LC, NUL, FIN = (Store(nm) for nm in ("cv_results", "learning_curves", "selector_null", "final"))


def _label(key, meta):
    """Short description of a result for the progress line."""
    if "family" not in meta:
        return key
    parts = [meta["exp"], f"n={meta['n']}", meta["sub"], show_params(meta.get("params"))]
    return " ".join(str(p) for p in parts if p)


def run(store, key, fn, quiet=False, **meta):
    """Run fn() unless this key is already stored. fn returns a dict (folds, mean, sd, secs, ...);
    meta labels the row. Fits that take 5 s or more print a progress line."""
    if key in store.rows:
        return store.rows[key]
    row = fn()
    row.update(key=key, **meta)
    store.rows[key] = row
    with open(store.path, "a") as f:
        f.write(json.dumps(row) + "\n")
    if not quiet and row.get("secs", 0) >= 5:  # only slow fits are announced
        print(
            f"{_label(key, meta):<64} {row['mean']:.3f} ± {row['sd']:.3f}  ({row.get('secs', 0)}s)"
        )
    return row


def acc(m, X, y):
    return float((m.predict(X) == y).mean())


def cv_eval(make, n, fam="", size=None, folds=None, post=None, shuffle_labels=False):
    """K-fold CV on dev: fit make() on the training folds, score on the held-out fold, so models
    can be compared fold by fold. size: use only that many training rows (a fixed random subset
    of each fold-train set)."""
    X, B, y = load(n)
    M = X if fam == "paper" else B
    t0, accs, trs, posts = time.time(), [], [], []
    for f, (tr, va) in enumerate(FOLDS[n]):
        if folds is not None and f not in folds:
            continue
        if size is not None and size < len(tr):
            tr = tr[np.random.default_rng(SEED + f).permutation(len(tr))[:size]]
        ytr = np.random.default_rng(1000 + f).permutation(y[tr]) if shuffle_labels else y[tr]
        est = make().fit(M[tr], ytr)
        accs.append(acc(est, M[va], y[va]))
        trs.append(acc(est, M[tr], ytr))
        if post:
            posts.append(post(est))
    return dict(
        folds=[round(a, 5) for a in accs],
        mean=float(np.mean(accs)),
        sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        train=float(np.mean(trs)),
        secs=round(time.time() - t0, 1),
        post=posts,
    )


def cv_config(n, exp, family, cond, sub, p, post=None, **kw):
    """Cross-validate one model setting, unless it is already stored."""
    key = f"{exp}|{n}|{family}|{cond}|{sub}|{json.dumps(p, sort_keys=True)}"
    return run(
        CV,
        key,
        lambda: cv_eval(lambda: make_estimator(family, p, cond), n, fam=family, post=post, **kw),
        exp=exp,
        n=n,
        family=family,
        cond=cond,
        sub=sub,
        params=p,
    )


def groups(n):
    """One entry per (family, condition, sub-model) with its CV-best setting, best first."""
    d = {}
    for r in CV.rows.values():
        if r["n"] == n and len(r["folds"]) == K:
            d.setdefault((r["family"], r["cond"], r["sub"]), []).append(r)
    pref = {"all": 0, "sel": 1}  # exact ties: all-bits first
    return sorted(
        [dict(key=k, best=max(rs, key=lambda r: r["mean"]), configs=rs) for k, rs in d.items()],
        key=lambda g: (-round(g["best"]["mean"], 9), pref[g["key"][1]]),
    )


def paired(a, b):
    """Paired fold differences a - b: mean, sd, Nadeau-Bengio standard error and 95% t interval.
    Folds share training rows, so the naive se sd/sqrt(k) is too small; the correction replaces
    1/k by 1/k + n_val/n_train."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    sd = d.std(ddof=1)
    se = sd * np.sqrt(1 / k + 1 / (K - 1))
    t = stats.t.ppf(0.975, k - 1)
    return dict(
        mean=float(d.mean()),
        sd=float(sd),
        se=float(se),
        lo=float(d.mean() - t * se),
        hi=float(d.mean() + t * se),
        wins=int((d > 0).sum()),
        k=k,
    )


def wilson_half(p, N, z=1.96):
    return float(z * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / (1 + z * z / N))


def interval(p):
    """'+0.107 [±0.041]': the mean and the half-width of the 95% interval from paired()."""
    return f"{p['mean']:+.3f} [±{(p['hi'] - p['lo']) / 2:.3f}]"


def lr_post(m):
    coef = m.named_steps["lr"].coef_
    return dict(nnz=int((coef != 0).sum()), n_feats=int(coef.shape[1]))


def sel_post(m):
    return dict(
        bits=[int(b) for b in m.cols_], pair=m.pair_, gate_z=float(m.gate_z_), n_sel=len(m.cols_)
    )


def jaccard_mean(sets):
    sets = [set(s) for s in sets]
    v = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(v)) if v else float("nan")


GENERIC = ("paper", "lr_l2", "lr_l1", "dtree", "rf", "hgb", "kmeans_major", "kmeans_onehot", "mlp")


def best_generic(n):
    """The best setting of each family on all bits."""
    out = []
    for fam in GENERIC:
        gs = [g for g in groups(n) if g["key"][0] == fam and g["key"][1] == "all"]
        if gs:
            out.append(max(gs, key=lambda g: g["best"]["mean"]))
    return out


def top_generic(n):
    """The best generic model; the paper baseline is a reference, not a candidate."""
    return max(
        (g for g in best_generic(n) if g["key"][0] != "paper"), key=lambda g: g["best"]["mean"]
    )


def sel_best(n, family="rf"):
    """The best setting of a model on the selected bits."""
    return [g for g in groups(n) if g["key"] == (family, "sel", family)][0]["best"]


# ---- Tables: plain text, one column per n
_NUMBER = re.compile(r"^[+\-−]?\d")


def text_table(header, rows, title=None, align=None, gap=2):
    """Print a table with padded columns. align: one 'l' or 'r' per column (default: columns whose
    cells all start with a digit or sign are right-aligned)."""
    rows = [[str(c) for c in r] for r in rows]
    cols = list(zip(header, *rows))
    width = [max(len(c) for c in col) for col in cols]
    align = align or "".join(
        "r" if all(_NUMBER.match(c) for c in col[1:] if c) else "l" for col in cols
    )

    def line(cells):
        return (
            (" " * gap)
            .join(c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(cells, width, align))
            .rstrip()
        )

    if title:
        print(title)
    print("\n".join([line(header), line(["-" * w for w in width])] + [line(r) for r in rows]))


def grid(title, rows, note=None):
    """A table with one row per quantity and one column per n."""
    text_table([""] + [f"n = {n}" for n in NS], rows, title, align="l" + "r" * len(NS))
    if note:
        print(note)


def ms(r):
    return f"{r['mean']:.3f} ± {r['sd']:.3f}"


def table_degree():
    """Paper baseline, the best explicit polynomial LR, and the kernel LR at degree 10, n and 2n."""
    rows = [["Paper baseline (quadratic LR)"] + [ms(CV.where(n=n, family="paper")[0]) for n in NS]]
    best = {
        n: max(
            (g for g in groups(n) if g["key"][0] in ("lr_l1", "lr_l2")),
            key=lambda g: g["best"]["mean"],
        )
        for n in NS
    }
    rows += [["Best explicit LR (degree up to 4 / 3 / 2)"] + [ms(best[n]["best"]) for n in NS]]
    rows += [
        ["    which"] + [f"{NAME[best[n]['key'][0]]}, degree {best[n]['key'][2][3:]}" for n in NS]
    ]
    for i, label in enumerate(("10", "n", "2n")[: len(G["KP_DEGS"][NS[0]])]):
        rs = [CV.where(n=n, family="kpoly", sub=f"deg{G['KP_DEGS'][n][i]}")[0] for n in NS]
        rows += [
            [f"Kernel LR, degree {label}"] + [ms(r) for r in rs],
            ["    train accuracy"] + [f"{r['train']:.3f}" for r in rs],
        ]
    grid("Cross-validated accuracy (mean ± sd over folds)", rows)


def table_kernel_c():
    """Held-out accuracy of the kernel LR at degree n on fold 0, for several C."""
    rows = []
    for C in (0.1, G["KP_C"], 10, 1000):
        cells = []
        for n in NS:
            r = [
                r
                for r in CV.where(n=n, family="kpoly")
                if r["params"]["C"] == C
                and r["params"]["deg"] == n
                and (r["exp"] == "KPOLYC" or C == G["KP_C"])
            ]
            cells.append(f"{r[0]['folds'][0]:.3f}" if r else "")
        rows.append([f"C = {C:g}" + (" (the paper's)" if C == G["KP_C"] else "")] + cells)
    grid("Kernel LR at degree n, fold 0 only: held-out accuracy by C", rows)


def table_families():
    """The best setting of every model family on all bits, by CV."""
    best = {n: {g["key"][0]: g for g in best_generic(n)} for n in NS}
    rows = [
        [NAME[f]] + [ms(best[n][f]["best"]) for n in NS]
        for f in ("dtree", "rf", "hgb", "kmeans_major", "kmeans_onehot", "mlp")
    ]
    kn = {r["n"]: r for r in CV.rows.values() if r["sub"].startswith("4x512")}
    if kn:
        rows += [["MLP 4×512, fold 0 only"] + [f"{kn[n]['folds'][0]:.3f}" for n in NS]]
        same = [best[n]["mlp"]["best"]["folds"][0] for n in NS]
        rows += [["    MLP 2×128, same fold"] + [f"{v:.3f}" for v in same]]
    grid(
        "Cross-validated accuracy on all bits (mean ± sd over folds, best setting of each family)",
        rows,
    )


def table_selector():
    """What the selector picks on the real labels and on shuffled labels."""
    post = {n: sel_best(n)["post"] for n in NS}
    rows = [
        ["Bits selected (mean over folds)"]
        + [f"{np.mean([p['n_sel'] for p in post[n]]):.1f}" for n in NS],
        ["Overlap of the bits between folds (Jaccard)"]
        + [f"{jaccard_mean([p['bits'] for p in post[n]]):.2f}" for n in NS],
        ["Gate z-score (mean)"] + [f"{np.mean([p['gate_z'] for p in post[n]]):.1f}" for n in NS],
        ["Bits selected on shuffled labels"]
        + [f"{np.mean([r['n_sel'] for r in NUL.where(n=n)]):.1f}" for n in NS],
    ]
    grid(
        "Bit selector",
        rows,
        "2 bits on shuffled labels = the gate pair alone, so no false bit is added.",
    )


def table_selected():
    """Models on the selected bits against the best model on all bits."""
    f, p, top = {}, {}, {}
    for n in NS:
        f[n], p[n], top[n] = sel_best(n), sel_best(n, "lrfull"), top_generic(n)
    rows = [
        ["Forest on the selected bits"] + [ms(f[n]) for n in NS],
        ["LR on all monomials of the selected bits"] + [ms(p[n]) for n in NS],
        ["Best model on all bits"]
        + [f"{top[n]['best']['mean']:.3f} ({SHORT[top[n]['key'][0]]})" for n in NS],
        ["Forest minus best on all bits"]
        + [interval(paired(f[n]["folds"], top[n]["best"]["folds"])) for n in NS],
        ["LR minus forest (same bits)"]
        + [interval(paired(p[n]["folds"], f[n]["folds"])) for n in NS],
    ]
    grid(
        "Cross-validated accuracy (mean ± sd over folds)",
        rows,
        "[±x] is the half-width of the 95% interval of the paired fold differences; an interval"
        " around 0 means\nthe two models cannot be told apart.",
    )


def table_summary(final):
    """One row per approach with its best setting's accuracy, grouped by what the approach uses."""
    best = {n: {g["key"][0]: g["best"] for g in best_generic(n)} for n in NS}
    kn = {r["n"]: r for r in CV.rows.values() if r["sub"].startswith("4x512")}
    sel = {n: (sel_best(n), sel_best(n, "lrfull")) for n in NS}
    k_sel = sorted({round(np.mean([p["n_sel"] for p in sel[n][0]["post"]])) for n in NS})

    def row(label, vals):
        return ["    " + label] + [f"{v:.3f}" for v in vals]

    def top(*fams):
        return [max(best[n][f]["mean"] for f in fams) for n in NS]

    degs = " / ".join(str(G["MAX_DEG"][n]) for n in NS)
    blank = [""] * (len(NS) + 1)
    rows = [
        ["The paper's approach: polynomials on all bits"] + [""] * len(NS),
        row("Quadratic LR (the paper's baseline)", [best[n]["paper"]["mean"] for n in NS]),
        row(f"LR on all monomials up to degree {degs}", top("lr_l1", "lr_l2")),
        row(
            "Kernel LR at degree n",
            [CV.where(n=n, family="kpoly", sub=f"deg{n}")[0]["mean"] for n in NS],
        ),
        blank,
        ["Other models on all bits"] + [""] * len(NS),
        row("K-means", top("kmeans_major", "kmeans_onehot")),
        row("Trees (best of tree, forest, boosting)", top("dtree", "rf", "hgb")),
        row("MLP 2×128", top("mlp")),
    ]
    if kn:
        rows += [row("MLP 4×512, fold 0 only †", [kn[n]["folds"][0] for n in NS])]
    bits = "/".join(map(str, k_sel))
    rows += [
        blank,
        [f"After selecting {bits} bits inside each fold"] + [""] * len(NS),
        row("Random forest on the selected bits", [sel[n][0]["mean"] for n in NS]),
        row("LR on all monomials of the selected bits", [sel[n][1]["mean"] for n in NS]),
        blank,
        ["Final pipeline, scored once on the locked test rows"] + [""] * len(NS),
        row("Locked test accuracy", [locked_score(final, n)["mean"] for n in NS]),
    ]
    note = (
        f"{K}-fold CV on the dev rows, mean of each approach's best setting; a coin flip is 0.50."
    )
    if kn:

        def fold0(vals):
            return " / ".join(f"{v:.3f}" for v in vals)

        same_fold = fold0([best[n]["mlp"]["folds"][0] for n in NS])
        forest_fold = fold0([sel[n][0]["folds"][0] for n in NS])
        note += (
            f"\n† Fold 0 only. On that fold MLP 2×128 scores {same_fold},"
            f"\n  and the forest on the selected bits {forest_fold}."
        )
    grid("Summary: accuracy of each approach", rows, note)


def table_final(final):
    """Cross-validation and locked-test accuracy of the final pipeline."""
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
        ["Shuffled-label control (should be 0.5)"]
        + [f"{FIN.rows[f'shuffle|{n}']['mean']:.3f}" for n in NS],
    ]
    print()
    grid("Final pipeline", rows)


# ---- Figures: drawn at their size in the report (one column 3.25 in, page width 6.75 in)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    }
)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, INK, GRAY = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#52514e",
    "#9a9892",
)
# Colour follows the model family and every family has its own marker, so nothing depends on colour
# alone.
COL = dict(
    lr_l2=BLUE,
    lr_l1=ORANGE,
    kpoly=VIOLET,
    lrfull=VIOLET,
    dtree=YELLOW,
    rf=AQUA,
    hgb=GREEN,
    mlp=MAGENTA,
    kmeans_major=GRAY,
    kmeans_onehot=INK,
    paper=GRAY,
)
MK = dict(lr_l2="o", lr_l1="s", kpoly="D")


def savefig(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.show()


def fig_degree():
    """Accuracy against polynomial degree: explicit LR (degree 1-4), the paper's kernel (degree 10,
    n, 2n), paper baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.7), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        for fam in ("lr_l2", "lr_l1"):
            degs = list(range(1, G["MAX_DEG"][n] + 1))
            bs = [
                max(CV.where(n=n, family=fam, sub=f"deg{d}"), key=lambda r: r["mean"]) for d in degs
            ]
            ax.plot(degs, [b["mean"] for b in bs], marker=MK[fam], color=COL[fam], label=NAME[fam])
        kd = G["KP_DEGS"][n]
        rs = [CV.where(n=n, family="kpoly", sub=f"deg{d}")[0] for d in kd]
        ax.plot(kd, [r["mean"] for r in rs], marker="D", color=VIOLET, label="Kernel LR, CV")
        ax.plot(
            kd, [r["train"] for r in rs], ls=":", marker="D", color=VIOLET, label="Kernel LR, train"
        )
        ax.axhline(
            CV.where(n=n, family="paper")[0]["mean"],
            color=GRAY,
            ls="--",
            lw=1.3,
            label="Paper baseline",
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 10, 20, 40, 80])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"n = {n}")
        ax.set_xlabel("polynomial degree")
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("accuracy")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    savefig(fig, "fig_degree")


def fig_models(final, top_n=8):
    """Cross-validation accuracy of the best candidates, one dot per fold and a diamond for the
    mean."""
    union = {}
    for n in NS:
        for g in [g for g in groups(n) if g["key"][0] != "paper"][:top_n] + [
            g for g in groups(n) if g["key"][0] == "paper"
        ]:
            union.setdefault(g["key"], []).append(g["best"]["mean"])
    # the same rows, in the same order, in every panel
    order = sorted(union, key=lambda k: -np.mean(union[k]))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.2), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        byk = {g["key"]: g for g in groups(n)}
        for i, key in enumerate(order):
            if key not in byk:
                continue
            g, sel = byk[key], key[1] != "all"
            col, chosen = (VIOLET if sel else COL[key[0]]), key == final[n]["key"]
            ax.scatter(
                g["best"]["folds"],
                [i] * len(g["best"]["folds"]),
                s=12,
                color=col,
                alpha=0.55,
                zorder=3,
            )
            ax.scatter(
                [g["best"]["mean"]],
                [i],
                s=55,
                color=col,
                marker="D",
                zorder=4,
                edgecolor=(INK if chosen else "white"),
                linewidth=1.5 if chosen else 0.8,
            )
        ax.axvline(0.9, color=GRAY, ls="--", lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([sname(k) for k in order])
        ax.grid(axis="y", alpha=0)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("CV accuracy")
        ax.set_xlim(0.45, 1.0)
        ax.set_xticks([0.5, 0.75, 1.0])
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    savefig(fig, "fig_models")
    print(
        "Dots: the folds. Diamond: the mean. Dashed: the 0.9 target. Dark outline: the pipeline"
        " chosen for that n. Violet: models on the selected bits."
    )


def fig_selected_bits():
    """The bits the selector chose in every fold."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.9), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        post = sel_best(n)["post"]
        for i, p in enumerate(post):
            ax.scatter(np.arange(n), [i] * n, s=5, color="#cfcec9", zorder=2)
            ax.vlines(p["bits"], i - 0.38, i + 0.38, color=VIOLET, lw=2.4, zorder=3)
        ax.set_yticks(range(len(post)))
        ax.set_yticklabels([str(i + 1) for i in range(len(post))])
        ax.grid(axis="y", alpha=0)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("bit index")
    axes[0].set_ylabel("fold")
    axes[0].set_ylim(len(sel_best(NS[0])["post"]) - 0.5, -0.5)
    savefig(fig, "fig_selected_bits")


def fig_learning_curves(cands):
    """Held-out accuracy against the number of training rows for the top candidates."""
    pal, mks, style = [VIOLET, ORANGE, BLUE, AQUA, MAGENTA], ["o", "s", "D", "^", "v"], {}
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.0), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        for g in cands[n]:
            fam, cond, sub = g["key"]
            rs = sorted(LC.where(n=n, family=fam, cond=cond, sub=sub), key=lambda r: r["size"])
            S, m, s = (
                [r["size"] for r in rs],
                np.array([r["mean"] for r in rs]),
                np.array([r["sd"] for r in rs]),
            )
            i = style.setdefault(sname(g["key"]), len(style))
            ax.plot(S, m, marker=mks[i % 5], ms=4, color=pal[i % 5], label=sname(g["key"]))
            ax.fill_between(S, m - s, m + s, color=pal[i % 5], alpha=0.12, lw=0)
        ax.axhline(0.9, color=GRAY, ls="--", lw=1)
        ax.axhline(0.5, color=GRAY, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([500, 2000, 10000])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"n = {n}")
        ax.set_xlabel("training rows")
        ax.set_ylim(0.40, 1.02)
    axes[0].set_ylabel("held-out accuracy")
    hl = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=3)
    savefig(fig, "fig_learning_curves")
    print("Mean over the first folds (shaded: one sd). Dashed: the 0.9 target. Dotted: chance.")


# ---- Experiments longer than a loop over a grid
def choose_final():
    """The final pipeline for each n: the candidate with the highest CV mean, the paper baseline
    excluded."""
    return {n: [g for g in groups(n) if g["key"][0] != "paper"][0] for n in NS}


def run_selector_null():
    """The selector on the training rows of fold 0 with permuted labels: any selected bit beyond
    the gate is false."""
    for n in NS:
        X, B, y = load(n)
        tr0 = FOLDS[n][0][0]
        for rep in range(G["NULL_REPS"]):

            def job():
                t0 = time.time()
                res = scan_bits(B[tr0], np.random.default_rng(rep).permutation(y[tr0]))
                sel = choose_bits(res)
                return dict(
                    mean=0.0,
                    sd=0.0,
                    secs=round(time.time() - t0, 1),
                    hits4=len(res["hits"]),
                    n_sel=len(sel["bits"]),
                    gate_z=sel["gate_z"],
                )

            run(NUL, f"null|{n}|{rep}", job, quiet=True, exp="NULL", n=n, rep=rep)


def run_learning_curves():
    """Train the top candidates on nested random subsets of the training folds and score them on
    the held-out fold. For logistic regression C multiplies the summed loss, so the same C
    penalises more strongly with fewer rows; C is scaled by (full rows / rows used) to keep the
    penalty per row fixed. Returns the candidates per n."""
    for k in [k for k, r in LC.rows.items() if r["family"] == "lrfull" and not k.endswith("|cs")]:
        del LC.rows[k]  # curves from an earlier version with an unscaled C: not used
    cands = {}
    for n in NS:
        seen, pick = set(), []
        for g in groups(n):
            k2 = g["key"][:2]
            if g["key"][0] == "paper" or k2 in seen:
                continue
            seen.add(k2)
            pick.append(g)
            if len(pick) == G["LC_TOP"]:
                break
        cands[n] = pick
        for g in pick:
            fam, cond, sub = g["key"]
            p = g["best"]["params"]
            suf = "|cs" if fam == "lrfull" else ""
            for size in G["LC_SIZES"]:
                if size >= FOLD_TRAIN[n]:
                    continue
                ps = dict(p, C=p["C"] * FOLD_TRAIN[n] / size) if fam == "lrfull" else p
                run(
                    LC,
                    f"lc|{n}|{fam}|{cond}|{sub}|{size}{suf}",
                    lambda: cv_eval(
                        lambda: make_estimator(fam, ps, cond),
                        n,
                        fam=fam,
                        size=size,
                        folds=G["LC_FOLDS"],
                    ),
                    exp="LC",
                    n=n,
                    family=fam,
                    cond=cond,
                    sub=sub,
                    size=size,
                )
            fr = [g["best"]["folds"][f] for f in G["LC_FOLDS"]]
            run(
                LC,
                f"lc|{n}|{fam}|{cond}|{sub}|{FOLD_TRAIN[n]}{suf}",
                lambda: dict(
                    folds=fr, mean=float(np.mean(fr)), sd=float(np.std(fr, ddof=1)), secs=0.0
                ),
                quiet=True,
                exp="LC",
                n=n,
                family=fam,
                cond=cond,
                sub=sub,
                size=FOLD_TRAIN[n],
            )
    return cands


def locked_score(final, n):
    """Refit the final pipeline on all dev rows and score it once on the locked rows (stored; later
    calls read it back)."""
    key = f"locked|{n}"
    if key in FIN.rows:
        return FIN.rows[key]
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
            half=wilson_half(a, len(te)),
            cv_mean=final[n]["best"]["mean"],
            cv_sd=final[n]["best"]["sd"],
            bits=[int(b) for b in est.cols_] if cond != "all" else None,
            params=p,
        )

    return run(FIN, key, job, exp="FINAL", n=n, family=fam, cond=cond, sub=sub)


def shuffled_label_control(final):
    """The final pipeline on shuffled training labels (first folds) must score about 0.5, otherwise
    something leaks."""
    for n in NS:
        fam, cond, sub = final[n]["key"]
        p = final[n]["best"]["params"]
        r = run(
            FIN,
            f"shuffle|{n}",
            lambda: cv_eval(
                lambda: make_estimator(fam, p, cond),
                n,
                fam=fam,
                folds=G["LC_FOLDS"],
                shuffle_labels=True,
            ),
            quiet=True,
            exp="CONTROL",
            n=n,
            family=fam,
            cond=cond,
            sub=sub,
        )
        assert (
            abs(r["mean"] - 0.5) < 0.03
        ), f"n={n}: shuffled-label accuracy {r['mean']:.3f}, possible leakage"


def predict_hidden(final):
    """Refit the final pipeline on all labelled rows, check its predictions, write
    hiddenlabels/y_predicted_{n}.npy and run the course's check_submission.py."""
    if WRITE_PREDICTIONS:
        os.makedirs(HIDDEN, exist_ok=True)
    for n in NS:
        fam, cond, sub = final[n]["key"]
        p = final[n]["best"]["params"]
        X, B, y = load(n)
        Xh = np.load(f"{DATA}/hidden-kryptonite-{n}-X.npy")
        Bh = np.where(Xh > 0, 1.0, -1.0).astype(np.float32)
        est = make_estimator(fam, p, cond).fit(B, y)
        pred = est.predict(Bh).astype(np.int64)
        assert pred.shape == (10_000,) and set(np.unique(pred)) <= {0, 1}, "wrong shape or values"
        assert 0.45 <= pred.mean() <= 0.55, f"suspicious class balance {pred.mean():.3f}"
        assert (
            np.abs(Bh.mean(0) - B.mean(0)).max() < 0.05
        ), "hidden bits do not look like the labelled bits"
        assert (
            np.mean(np.abs(Xh - Bh) < 0.5) > 0.999
        ), "hidden inputs are not clean +-1 bits plus small noise"
        msg = f"n={n}: {kname(final[n]['key'])}, predicted positive rate {pred.mean():.3f}"
        alt = next(
            (
                g
                for g in groups(n)
                if g["key"][:2] != (fam, cond)
                and g["key"][0] != "paper"
                and g["best"]["mean"] > 0.9
            ),
            None,
        )
        if alt:  # a second good model should mostly agree
            agree = float(
                (
                    make_estimator(alt["key"][0], alt["best"]["params"], alt["key"][1])
                    .fit(B, y)
                    .predict(Bh)
                    == pred
                ).mean()
            )
            msg += f"; agrees with {kname(alt['key'])} on {agree:.1%}" + (
                "  <-- LOW, look at this" if agree < 0.85 else ""
            )
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
