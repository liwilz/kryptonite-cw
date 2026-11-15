"""Helpers for Experiments.ipynb: settings, data splits, the models, the bit elimination, result
storage, tables and figures. The notebook calls these; the experiments themselves are in it."""

import hashlib
import importlib.util
import itertools
import json
import os
import re
import sys
import time
import warnings
from fractions import Fraction
from functools import lru_cache
from math import ceil, comb

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ---- Settings. KRYP_QUICK=1 runs a tiny dry run (see the end of this block); its numbers mean
# nothing.
QUICK = os.environ.get("KRYP_QUICK", "0") == "1"
ROOT = next((p for p in (".", "..", "../..", "../../..") if os.path.isdir(f"{p}/Datasets")), ".")
DATA, HIDDEN = f"{ROOT}/Datasets", f"{ROOT}/hiddenlabels"
NS = (20, 30, 40)
SEED = 0  # fixes the locked split, the folds and every model
K = 5  # CV folds
ROWS = None  # None = all labelled rows
OUT, FIG = "exp_cache", "exp_figures"  # OUT only lets an interrupted run resume
WRITE_PREDICTIONS = True
RUN_MLP = True  # the 4 x 512 MLP, the slowest experiment
MAX_BITS = 14  # most bits the elimination may keep: a degree-10 polynomial on more is too large

G = dict(
    C_L2=[1e-3, 1e-2, 1e-1, 1],
    C_L1=[3e-3, 1e-2, 3e-2, 1e-1],
    MAX_DEG={20: 4, 30: 3, 40: 2},  # largest degree whose monomials on all bits fit in memory
    KP_DEGS={n: [6, 10, 15, n, 2 * n] for n in NS},  # degrees of the paper's kernel
    KP_C=0.85,
    RF_TREES=200,
    RF_MAXF=["sqrt", 0.5, 1.0],
    RF_LEAF=[1, 5],
    MLP_H=[512, 512, 512, 512],  # KryptoNet's shape
    MLP_ALPHA=1e-2,
    MLP_MAXITER=200,  # epoch cap; check_caps() reports any fit that reached it
    MLP_PATIENCE=15,
    SEL_DEGS=list(range(1, 11)),  # degrees tried on the kept bits
    LC_SIZES=[250, 500, 1000, 2000, 4000],  # learning curves; the full fold-train set is added
    LC_FOLDS=(0, 1, 2),
    ELIM_SEEDS=(1, 2),  # extra forest seeds for the stability check of the elimination
)
if QUICK:  # dry run: 3 folds, 3,000 rows, small grids, separate folders, no predictions
    K, ROWS, OUT, FIG, WRITE_PREDICTIONS = 3, 3000, "exp_cache_quick", "exp_figures_quick", False
    G.update(
        C_L2=[1e-2, 1e-1],
        C_L1=[1e-2, 1e-1],
        MAX_DEG={20: 2, 30: 2, 40: 2},
        KP_DEGS={n: [10, n] for n in NS},
        RF_TREES=30,
        RF_MAXF=["sqrt", 1.0],
        RF_LEAF=[5],
        MLP_MAXITER=40,
        MLP_PATIENCE=10,
        SEL_DEGS=[2, 6, 10],
        LC_SIZES=[100, 250, 500],
        LC_FOLDS=(0, 1),
        ELIM_SEEDS=(1,),
    )
if not os.path.isfile(f"{DATA}/kryptonite-20-X.npy"):
    raise FileNotFoundError(
        f"{DATA}/kryptonite-20-X.npy not found: run from a folder inside the repository"
    )
if importlib.util.find_spec("codecarbon") is None:  # energy measurement, see run()
    raise ImportError("pip install codecarbon (tested with 3.3.2)")
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/.gitignore", "w") as f:
    f.write("*\n")
os.makedirs(FIG, exist_ok=True)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul")


# ---- Names used in tables and figures
NAME = dict(
    lr_l2="LR, L2",
    lr_l1="LR, L1",
    kpoly="Kernel LR",
    rf="Random forest",
    mlp="MLP 4×512",
    paper="Paper baseline",
)


def kname(key):
    """Full name of a (family, condition, sub-model) key."""
    fam, cond, sub = key
    return (
        NAME[fam]
        + (f", degree {sub[3:]}" if sub.startswith("deg") else "")
        + (" on the kept bits" if cond == "sel" else "")
    )


def show_params(p):
    """{"deg": 4, "C": 0.01} -> "degree 4, C 0.01"."""
    names = dict(deg="degree", h="layers", max_features="max features", msl="min leaf")
    out = []
    for k, v in (p or {}).items():
        v = f"{len(v)}×{v[0]}" if k == "h" else v
        out.append(f"{names.get(k, k)} {v}")
    return ", ".join(out)


# ---- Data and splits: a locked 20% test set and K stratified folds on the other 80% (dev)
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
            f" positive; noise sd {np.std(X - B):.3f}"
        )


# ---- Result store: one JSON line per finished result, so an interrupted run resumes
class Store:
    def __init__(self, name):
        self.path, self.rows = f"{OUT}/{name}.jsonl", {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line)
                self.rows[r["key"]] = r

    def where(self, **kw):
        return [r for r in self.rows.values() if all(r.get(k) == v for k, v in kw.items())]


CV, LC, SEL, FIN = (Store(nm) for nm in ("cv_results", "learning_curves", "elimination", "final"))


def _label(key, meta):
    """Short description of a result for the progress line."""
    if "family" not in meta:
        return key
    parts = [meta["exp"], f"n={meta['n']}", meta["sub"], show_params(meta.get("params"))]
    return " ".join(str(p) for p in parts if p)


# ---- Energy: CodeCarbon reads the CPU's own energy counters (RAPL) through the Windows Energy
# Meter Interface (Windows 11) or /sys/class/powercap (Linux). Each stored result records the
# energy used while it was computed, so the totals survive a resumed run. CodeCarbon also reads an
# NVIDIA GPU if present; that is stored but not reported, as no model uses the GPU. Where no
# counter can be read CodeCarbon falls back to an estimate; carbon_report() then
# says so and reports no energy.
MEASURED = ("intel_rapl", "windows_emi", "intel_power_gadget")
_TRACKER = []  # [tracker, task active?], made on first use


def _tracker():
    if not _TRACKER:
        from codecarbon import OfflineEmissionsTracker

        _TRACKER[:] = [
            OfflineEmissionsTracker(
                country_iso_code="GBR",
                tracking_mode="machine",
                save_to_file=False,
                log_level="error",
                allow_multiple_runs=True,
            ),
            False,
        ]
    return _TRACKER


def _measured(fn):
    """fn() plus the energy CodeCarbon recorded while it ran. A call inside another measured call
    (the bit elimination inside a model fit) is counted in the outer one."""
    tr = _tracker()
    if tr[1]:
        return dict(fn(), energy=None)  # counted in the enclosing result
    tr[1] = True
    tr[0].start_task()
    try:
        row = fn()
    finally:
        d = tr[0].stop_task()
        tr[1] = False
    modes = [getattr(h, "_mode", None) for h in tr[0]._hardware if type(h).__name__ == "CPU"]
    row["energy"] = dict(
        cpu_kwh=float(d.cpu_energy),
        gpu_kwh=float(d.gpu_energy),
        cpu_mode=modes[0] if modes else None,
        gpu=d.gpu_model or None,
        g_per_kwh=float(d.emissions / d.energy_consumed * 1000) if d.energy_consumed else None,
    )
    return row


def run(store, key, fn, quiet=False, **meta):
    """Run fn() unless this key is already stored. fn returns a dict (folds, mean, sd, secs, ...);
    meta labels the row. Fits that take 5 s or more print a progress line."""
    if key in store.rows:
        return store.rows[key]
    row = _measured(fn)
    row.update(key=key, **meta)
    store.rows[key] = row
    with open(store.path, "a") as f:
        f.write(json.dumps(row) + "\n")
    if not quiet and row.get("secs", 0) >= 5:  # only slow fits are announced
        print(
            f"{_label(key, meta):<64} {row['mean']:.3f} ± {row['sd']:.3f}  ({row.get('secs', 0)}s)"
        )
    return row


# ---- Bit selection: recursive feature elimination (Guyon et al., 2002) with a random forest and
# permutation importance (Breiman, 2001)
ELIM_VAL = 0.25  # share of the training rows held out inside the elimination


def eliminate_bits(B, y, seed=SEED):
    """Hold out a quarter of the rows. Fit a forest on the other three quarters using the bits
    still in, record its accuracy on the held-out quarter, and drop the bit with the lowest
    permutation importance (the drop in held-out accuracy when that bit's values are shuffled).
    Repeat until one bit is left. Returns [[bits, held-out accuracy], ...]."""
    fit, val = train_test_split(
        np.arange(len(y)), test_size=ELIM_VAL, stratify=y, random_state=seed
    )
    cols, path = list(range(B.shape[1])), []
    while cols:
        rf = RandomForestClassifier(
            G["RF_TREES"], max_features=1.0, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ).fit(B[fit][:, cols], y[fit])
        Bv = B[val][:, cols]
        path.append([list(cols), float((rf.predict(Bv) == y[val]).mean())])
        # one process per shuffled bit; a forest that also split each small prediction across
        # all cores spent most of its time on that overhead. Same result either way.
        rf.set_params(n_jobs=1)
        imp = permutation_importance(rf, Bv, y[val], n_repeats=5, random_state=seed, n_jobs=-1)
        cols.pop(int(np.argmin(imp.importances_mean)))
    return path


def choose_bits(path, N):
    """The smallest set of at most MAX_BITS bits whose held-out accuracy is within one standard
    error of the best such set (the one-standard-error rule). N: rows the elimination ran on."""
    path = [(bits, a) for bits, a in path if len(bits) <= MAX_BITS]
    best = max(a for _, a in path)
    se = np.sqrt(best * (1 - best) / ceil(ELIM_VAL * N))
    return sorted(min((bits for bits, a in path if a >= best - se), key=len))


def get_path(B, y, seed=SEED):
    """eliminate_bits, stored under a hash of the rows so the same rows are processed once."""
    key = hashlib.md5(np.ascontiguousarray(B).tobytes() + y.astype(np.int8).tobytes()).hexdigest()
    key += "" if seed == SEED else f"|seed{seed}"

    def job():
        t0 = time.time()
        path = eliminate_bits(B, y, seed)
        return dict(path=path, mean=0.0, sd=0.0, secs=round(time.time() - t0, 1))

    return run(SEL, key, job, quiet=True, exp="ELIM", n=B.shape[1], rows=len(y), seed=seed)["path"]


def fold_paths(n, seed=SEED):
    """The elimination path on the training rows of every fold."""
    X, B, y = load(n)
    return [get_path(B[tr], y[tr], seed) for tr, _ in FOLDS[n]]


class BitModel:
    """Wraps any estimator: eliminate bits on the rows it is fitted on, then fit the estimator on
    the bits kept. other=True is a control: it uses as many bits drawn at random from the
    eliminated ones."""

    def __init__(self, base, other=False):
        self.base, self.other = base, other

    def fit(self, B, y):
        self.cols_ = choose_bits(get_path(B, y), len(y))
        rest = [i for i in range(B.shape[1]) if i not in self.cols_]
        if self.other and rest:
            k = min(len(self.cols_), len(rest))
            self.cols_ = sorted(np.random.default_rng(SEED).choice(rest, k, replace=False).tolist())
        self.base.fit(B[:, self.cols_], y)
        return self

    def predict(self, B):
        return self.base.predict(B[:, self.cols_])


# ---- Models
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
    """Build any model. p: JSON-friendly hyper-parameters. cond: 'all' bits; 'sel' = the bit
    elimination runs inside fit(); 'other' = the control on eliminated bits."""
    if family in ("lr_l1", "lr_l2"):  # LR on all products of up to deg distinct bits
        est = Pipeline(
            [
                ("poly", PolynomialFeatures(p["deg"], interaction_only=True, include_bias=False)),
                ("lr", lr_clf(family[-2:], p["C"])),
            ]
        )
    elif family == "kpoly":
        est = KernelPolyLR(p["deg"], p["C"])
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
    return est if cond == "all" else BitModel(est, other=cond == "other")


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


# ---- Cross-validation and comparisons
def acc(m, X, y):
    return float((m.predict(X) == y).mean())


def _capped(est):
    """True if the fit stopped at its iteration cap instead of converging."""
    m = est.base if isinstance(est, BitModel) else est
    m = m.steps[-1][1] if isinstance(m, Pipeline) else m
    if isinstance(m, KernelPolyLR):
        return m.iters_ >= m.maxiter
    if isinstance(m, (LogisticRegression, MLPClassifier)):
        return int(np.max(m.n_iter_)) >= m.max_iter
    return False


def cv_eval(make, n, fam="", size=None, folds=None, shuffle_labels=False):
    """K-fold CV on dev: fit make() on the training folds, score on the held-out fold, so models
    can be compared fold by fold. size: use only that many training rows (a fixed random subset
    of each fold-train set)."""
    X, B, y = load(n)
    M = X if fam == "paper" else B
    t0, accs, trs, capped = time.time(), [], [], 0
    for f, (tr, va) in enumerate(FOLDS[n]):
        if folds is not None and f not in folds:
            continue
        if size is not None and size < len(tr):
            tr = tr[np.random.default_rng(SEED + f).permutation(len(tr))[:size]]
        ytr = np.random.default_rng(1000 + f).permutation(y[tr]) if shuffle_labels else y[tr]
        est = make().fit(M[tr], ytr)
        accs.append(acc(est, M[va], y[va]))
        trs.append(acc(est, M[tr], ytr))
        capped += bool(_capped(est))
    return dict(
        capped=capped,
        folds=[round(a, 5) for a in accs],
        mean=float(np.mean(accs)),
        sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        train=float(np.mean(trs)),
        secs=round(time.time() - t0, 1),
    )


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


def groups(n):
    """One entry per (family, condition, sub-model) with its CV-best setting, best first."""
    d = {}
    for r in CV.rows.values():
        if r["n"] == n and len(r["folds"]) == K and r["family"] in NAME:  # skip cut models
            d.setdefault((r["family"], r["cond"], r["sub"]), []).append(r)
    pref = {"all": 0, "sel": 1}  # exact ties: all-bits first
    return sorted(
        [dict(key=k, best=max(rs, key=lambda r: r["mean"]), configs=rs) for k, rs in d.items()],
        key=lambda g: (-round(g["best"]["mean"], 9), pref[g["key"][1]]),
    )


def difference(a, b):
    """Fold-by-fold accuracy difference a - b on the same folds: '+0.283 ± 0.012 (5 of 5)', the
    mean ± sd over folds and the number of folds where a is better."""
    d = np.asarray(a) - np.asarray(b)
    return f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({int((d > 0).sum())} of {len(d)})"


def binomial_half(p, N, z=1.96):
    """Half-width of the 95% interval of an accuracy p measured on N independent test rows: each
    row is right or wrong, so the count of correct rows is binomial with sd sqrt(p (1 - p) / N)."""
    return float(z * np.sqrt(p * (1 - p) / N))


def jaccard_mean(sets):
    sets = [set(s) for s in sets]
    v = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(v)) if v else float("nan")


def best_of(n, cond, fams=("lr_l1", "lr_l2"), sub=None):
    """The CV-best group among these families on these bits (optionally one sub-model)."""
    return next(
        g
        for g in groups(n)
        if g["key"][0] in fams and g["key"][1] == cond and sub in (None, g["key"][2])
    )


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


def _mean_sd(n, cond, fams, sub=None):
    """'0.675 ± 0.017' for the best such model, or '-' if it was not run."""
    try:
        return ms(best_of(n, cond, fams, sub)["best"])
    except StopIteration:
        return "-"


def kernel_label(i):
    """Name of the i-th kernel degree: a number, or 'n' / '2n' when it differs between datasets."""
    degs = [G["KP_DEGS"][n][i] for n in NS]
    return str(degs[0]) if len(set(degs)) == 1 else ("n" if degs[0] == NS[0] else "2n")


def table_degree_all():
    """Polynomial LR on all bits by degree: explicit monomials, then the paper's kernel."""
    blank = [""] * (len(NS) + 1)
    rows = [["Explicit monomials (best penalty and C per degree)"] + [""] * len(NS)]
    for d in range(1, max(G["MAX_DEG"].values()) + 1):
        rows.append(
            [f"    degree {d}"] + [_mean_sd(n, "all", ("lr_l1", "lr_l2"), f"deg{d}") for n in NS]
        )
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


def table_families():
    """The best setting of every other model family on all bits."""
    rows = [
        ["Random forest"] + [_mean_sd(n, "all", ("rf",)) for n in NS],
        ["MLP 4×512"] + [_mean_sd(n, "all", ("mlp",)) for n in NS],
        ["Best polynomial LR, for reference"]
        + [_mean_sd(n, "all", ("lr_l1", "lr_l2", "kpoly")) for n in NS],
    ]
    grid("Trained from scratch on all bits: CV accuracy (mean ± sd over folds)", rows)
    for n in NS:
        g = best_of(n, "all", ("rf",))
        print(f"n = {n}: best forest has {show_params(g['best']['params'])}")


def table_elimination():
    """What the elimination keeps in every fold, and the held-out accuracy around the kept set."""
    paths = {n: fold_paths(n) for n in NS}
    kept = {n: [choose_bits(p, FOLD_TRAIN[n]) for p in paths[n]] for n in NS}

    def oob(n, size):
        """Mean held-out accuracy over folds of the set with size(kept) bits."""
        vals = []
        for p, k in zip(paths[n], kept[n]):
            vals += [a for bits, a in p if len(bits) == size(k)]
        return f"{np.mean(vals):.3f}" if vals else "-"

    rows = [
        ["Bits kept (mean over folds)"] + [f"{np.mean([len(k) for k in kept[n]]):.1f}" for n in NS],
        ["Overlap of the kept bits between folds (Jaccard)"]
        + [f"{jaccard_mean(kept[n]):.2f}" for n in NS],
        ["Forest accuracy with all bits"] + [oob(n, lambda k, n=n: n) for n in NS],
        ["Forest accuracy with the kept bits"] + [oob(n, len) for n in NS],
        ["Forest accuracy with one bit fewer"] + [oob(n, lambda k: len(k) - 1) for n in NS],
    ]
    for seed in G["ELIM_SEEDS"]:
        same = [
            sum(choose_bits(p, FOLD_TRAIN[n]) == k for p, k in zip(fold_paths(n, seed), kept[n]))
            for n in NS
        ]
        rows.append(
            [f"Forest seed {seed}: folds that keep the same bits"] + [f"{v} of {K}" for v in same]
        )
    grid(
        "Recursive elimination of bits with a random forest (training rows of each fold)",
        rows,
        "Forest accuracy: on the quarter of each fold's training rows held out inside the"
        "\nelimination, mean over folds.",
    )


def table_degree_sel():
    """Polynomial LR on the kept bits by degree."""
    rows = [
        [f"degree {d}"] + [_mean_sd(n, "sel", ("lr_l2",), f"deg{d}") for n in NS]
        for d in G["SEL_DEGS"]
    ]
    grid("Polynomial LR on the kept bits: CV accuracy by degree (mean ± sd over folds)", rows)


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
            [b["mean"] for b in best("all", ("lr_l1", "lr_l2"))],
        ),
        ("deg", "degree", degree("all", ("lr_l1", "lr_l2"))),
        ("row", "Paper's kernel, best degree", [b["mean"] for b in best("all", ("kpoly",))]),
        ("deg", "degree", degree("all", ("kpoly",))),
        ("group", "Trained from scratch on all bits"),
        ("row", "Random forest", [b["mean"] for b in best("all", ("rf",))]),
    ]
    if all(CV.where(n=n, family="mlp", exp="MLP") for n in NS):
        rows.append(("row", NAME["mlp"], [b["mean"] for b in best("all", ("mlp",))]))
    rows.append(("group", "Polynomial LR on the bits kept by the forest"))
    for d in [d for d in (2, 4, 6, 10) if d in G["SEL_DEGS"]]:
        rows.append(("row", f"Degree {d}", [b["mean"] for b in best("sel", ("lr_l2",), f"deg{d}")]))
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


def table_summary(final):
    """The summary as plain text."""
    rows, notes = summary_rows(final)
    out = []
    for r in rows:
        if r[0] == "group":
            if out:
                out.append([""] * (len(NS) + 1))
            out.append([r[1]] + [""] * len(NS))
        else:
            fmt = "d" if r[0] == "deg" else ".3f"
            out.append(
                [("        " if r[0] == "deg" else "    ") + r[1]] + [format(v, fmt) for v in r[2]]
            )
    grid("Summary: accuracy of each approach", out, "\n".join(notes))


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
        ["Control: same model on eliminated bits"]
        + [
            f"{FIN.rows[f'other|{n}']['mean']:.3f}" if f"other|{n}" in FIN.rows else "-" for n in NS
        ],
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
BLUE, ORANGE, AQUA, VIOLET, MAGENTA, INK, GRAY = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#6250d6",
    "#e87ba4",
    "#52514e",
    "#9a9892",
)


def savefig(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.show()


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

    for i, g in enumerate(group_of, start=1):  # tint the models that use the kept bits
        if "bits kept" in rows[g][1]:
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


def fig_degree():
    """CV accuracy against the polynomial degree: explicit monomials and the paper's kernel on all
    bits, explicit monomials on the kept bits."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.8), sharey=True, layout="constrained")
    series = (  # drawn in this order, so LR on all bits stays visible where the two LR lines meet
        ("LR on the kept bits", VIOLET, "s", "sel", ("lr_l2",), lambda n: G["SEL_DEGS"]),
        ("Kernel LR on all bits", ORANGE, "D", "all", ("kpoly",), lambda n: G["KP_DEGS"][n]),
        (
            "LR on all bits",
            BLUE,
            "o",
            "all",
            ("lr_l1", "lr_l2"),
            lambda n: range(1, G["MAX_DEG"][n] + 1),
        ),
    )
    for ax, n in zip(axes, NS):
        for label, col, mk, cond, fams, degs in series:
            bs = [best_of(n, cond, fams, f"deg{d}")["best"] for d in degs(n)]
            m, s = np.array([b["mean"] for b in bs]), np.array([b["sd"] for b in bs])
            ax.plot(list(degs(n)), m, marker=mk, ms=4 if mk != "o" else 5, color=col, label=label)
            ax.fill_between(list(degs(n)), m - s, m + s, color=col, alpha=0.15, lw=0)
        base = best_of(n, "all", ("paper",))["best"]["mean"]
        ax.axhline(base, color=GRAY, ls="--", lw=1.3, label="Paper baseline")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 10, 20, 40, 80])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"n = {n}")
        ax.set_xlabel("polynomial degree")
        ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("CV accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    savefig(fig, "fig_degree")
    print("Mean over the folds (shaded: one sd), best setting at each degree.")


def fig_elimination():
    """Held-out accuracy of the forest against the number of bits still in, one line per fold."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        paths = fold_paths(n)
        for p in paths:
            ax.plot([len(b) for b, _ in p], [a for _, a in p], color=VIOLET, lw=1.2, alpha=0.6)
        kept = [len(choose_bits(p, FOLD_TRAIN[n])) for p in paths]
        ax.axvline(np.mean(kept), color=GRAY, ls="--", lw=1.3)
        ax.axhline(0.5, color=GRAY, ls=":", lw=1)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("bits still in")
        ax.set_xlim(n + 1, 0)
        ax.set_ylim(0.40, 1.0)
    axes[0].set_ylabel("held-out accuracy")
    savefig(fig, "fig_elimination")
    print("One line per fold; bits are removed from left to right. Dashed: bits kept. Dotted: 0.5.")


LOOK = dict(  # label, colour and marker of each learning curve
    sel=("LR on the kept bits", VIOLET, "s"),
    lr=("LR on all bits", BLUE, "o"),
    rf=("Random forest on all bits", AQUA, "^"),
    mlp=("MLP 4×512 on all bits", MAGENTA, "v"),
)


def fig_learning_curves(cands):
    """Held-out accuracy against the number of training rows."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.0), sharey=True, layout="constrained")
    for ax, n in zip(axes, NS):
        for g in cands[n]:
            fam, cond, sub = g["key"]
            rs = sorted(LC.where(n=n, family=fam, cond=cond, sub=sub), key=lambda r: r["size"])
            S = [r["size"] for r in rs]
            m, s = np.array([r["mean"] for r in rs]), np.array([r["sd"] for r in rs])
            label, col, mk = LOOK["sel" if cond == "sel" else fam[:2] if fam[:2] == "lr" else fam]
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
    axes[0].set_ylabel("held-out accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    savefig(fig, "fig_learning_curves")
    print(
        "Mean over the first folds (shaded: one sd), each model at its CV-best setting. Dotted: 0.5"
    )


# ---- Experiments longer than a loop over a grid
def choose_final():
    """The final pipeline for each n: the candidate with the highest CV mean, the paper baseline
    excluded."""
    return {n: [g for g in groups(n) if g["key"][0] != "paper"][0] for n in NS}


def run_learning_curves():
    """Train the best model of each kind on nested random subsets of the training folds and score
    it on the held-out fold. For LR, C multiplies the summed loss, so the same C penalises more
    strongly with fewer rows; C is scaled by (full rows / rows used) to keep the penalty per row
    fixed. Returns the models per n."""
    cands = {}
    for n in NS:
        cands[n] = [best_of(n, "sel"), best_of(n, "all"), best_of(n, "all", ("rf",))]
        if CV.where(n=n, family="mlp", exp="MLP"):
            cands[n].append(best_of(n, "all", ("mlp",)))
        for g in cands[n]:
            fam, cond, sub = g["key"]
            p = g["best"]["params"]
            meta = dict(exp="LC", n=n, family=fam, cond=cond, sub=sub)
            for size in [s for s in G["LC_SIZES"] if s < FOLD_TRAIN[n]]:
                ps = dict(p, C=p["C"] * FOLD_TRAIN[n] / size) if "C" in p else p
                run(
                    LC,
                    f"lc|{n}|{fam}|{cond}|{sub}|{size}",
                    lambda: cv_eval(
                        lambda: make_estimator(fam, ps, cond), n, size=size, folds=G["LC_FOLDS"]
                    ),
                    size=size,
                    **meta,
                )
            fr = [g["best"]["folds"][f] for f in G["LC_FOLDS"]]
            run(
                LC,
                f"lc|{n}|{fam}|{cond}|{sub}|{FOLD_TRAIN[n]}",
                lambda: dict(
                    folds=fr, mean=float(np.mean(fr)), sd=float(np.std(fr, ddof=1)), secs=0.0
                ),
                quiet=True,
                size=FOLD_TRAIN[n],
                **meta,
            )
    return cands


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
            bits=est.cols_ if cond == "sel" else None,
            params=p,
        )

    key = f"locked|{n}|{fam}|{cond}|{sub}|{json.dumps(p, sort_keys=True)}"
    return run(FIN, key, job, exp="FINAL", n=n, family=fam, cond=cond, sub=sub)


def run_controls(final):
    """Two controls for the final pipeline on the first folds, both expected at about 0.5: shuffled
    training labels (otherwise something leaks), and the same model on as many eliminated bits
    (otherwise the kept bits are not what carries the label)."""
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
        if cond == "sel":
            run(
                FIN,
                f"other|{n}",
                lambda: cv_eval(lambda: make_estimator(fam, p, "other"), n, folds=G["LC_FOLDS"]),
                quiet=True,
                **meta,
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
        good = [g for g in groups(n) if g["best"]["mean"] > 0.9 and g["key"][1] != cond]
        alt = good[0] if good else None

        def job():
            t0 = time.time()
            est = make_estimator(fam, p, cond).fit(B, y)
            out = dict(pred=est.predict(Bh).tolist(), bits=getattr(est, "cols_", None))
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
        # The best model of another kind (other bits) must mostly agree. If each errs on a share e
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


def check_caps():
    """Every stored fit that stopped at its iteration cap instead of converging."""
    rows = [r for st in (CV, LC, FIN) for r in st.rows.values() if r.get("capped")]
    if not rows:
        print("No fit stopped at its iteration cap.")
    for r in rows:
        print(f"{r['key']}: {r['capped']} fold(s) stopped at the iteration cap")


GRID_G_PER_KWH = 125  # NESO, "Britain's Electricity Explained: 2024 Review": GB average for 2024


def carbon_report():
    """Energy measured by CodeCarbon for every stored result, by experiment, and the emissions at
    GRID_G_PER_KWH."""
    rows = [r for st in (CV, LC, SEL, FIN) for r in st.rows.values()]
    dropped = [r for r in rows if r.get("family") == "kmeans" and r.get("energy")]
    rows = [r for r in rows if r.get("family") != "kmeans"]  # cut from the notebook, see below
    missing = [r for r in rows if "energy" not in r]
    rows = [r for r in rows if r.get("energy")]
    modes = {r["energy"]["cpu_mode"] for r in rows}
    if missing or not modes or not modes <= set(MEASURED):
        print(
            f"Not reported: {len(missing)} results have no energy record and the CPU modes are"
            f" {sorted(map(str, modes))}; energy is reported only when every result was measured"
            f" by the CPU's counters ({', '.join(MEASURED)})."
        )
        return
    names = dict(
        BASE="Polynomial LR on all bits",
        LR="Polynomial LR on all bits",
        KPOLY="Kernel LR on all bits",
        RF="Random forest on all bits",
        FOLD="MLP 4×512",
        MLP="MLP 4×512",
        LC="Learning curves",
        CONTROL="Final pipeline: controls and locked test",
        FINAL="Final pipeline: controls and locked test",
        HIDDEN="Hidden-set refits",
    )

    def name(r):
        if r.get("exp") == "ELIM":
            return "Bit elimination" + (" (stability check)" if r.get("seed", SEED) != SEED else "")
        if r.get("exp") == "LR" and r.get("cond") == "sel":
            return "Polynomial LR on the kept bits"
        return names.get(r.get("exp"), str(r.get("exp")))

    tot = {}
    for r in rows:
        t = tot.setdefault(name(r), np.zeros(2))
        t += [r.get("secs", 0) / 3600, r["energy"]["cpu_kwh"] * 1000]
    total = np.sum(list(tot.values()), axis=0)

    def line(k, v):
        return [k, f"{v[0]:.2f}", f"{v[1]:.1f}", f"{v[1] * GRID_G_PER_KWH / 1000:.1f}"]

    out = [line(k, v) for k, v in tot.items()] + [line("Total", total)]
    text_table(["", "hours", "CPU Wh", "g CO2e"], out, "Measured CPU energy by experiment")
    print(
        f"Read from the CPU's energy counters ({', '.join(sorted(modes))}), whole package. Every"
        " model runs on the CPU only,\nso no GPU energy is counted; RAM and screen are not"
        f" measured. g CO2e at {GRID_G_PER_KWH} g per kWh (NESO, GB 2024 average)."
    )
    if dropped:
        wh = sum(r["energy"]["cpu_kwh"] * 1000 for r in dropped)
        print(f"Not included: {len(dropped)} K-means results from an earlier version, {wh:.1f} Wh.")

    def cv_cost(pick):
        return sum(r["energy"]["cpu_kwh"] * 1000 for r in rows if pick(r))

    for n in NS:
        sel = cv_cost(
            lambda r: r.get("n") == n
            and (
                r.get("exp") == "ELIM"
                and r.get("seed", SEED) == SEED
                and r.get("rows") == FOLD_TRAIN[n]
                or r.get("exp") == "LR"
                and r.get("cond") == "sel"
            )
        )
        mlp = cv_cost(lambda r: r.get("n") == n and r.get("exp") in ("FOLD", "MLP"))
        print(
            f"n = {n}: {K}-fold CV of the kept-bits pipeline (elimination + LR grid) {sel:.1f} Wh;"
            f" of the MLP {mlp:.1f} Wh"
        )
