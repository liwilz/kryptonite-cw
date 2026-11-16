"""Shared by every experiment script: settings, the data and its splits, the result cache, the
energy measurement, cross-validation, looking up stored results, tables and the figure style."""

import importlib.util
import json
import os
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

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

G = dict(
    C_L2=[1e-3, 1e-2, 1e-1, 1],
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
    WK_DEGS=[6, 10, 15],  # weighted polynomial kernel: degrees
    WK_C=[10, 100],  # and C, with forest weights
    WK_UNIFORM_C=10,  # C of the same kernel with equal weights, the comparison
    LC_SIZES=[250, 500, 1000, 2000, 4000],  # learning curves; the full fold-train set is added
    LC_FOLDS=(0, 1, 2),
    # MLP ablations (Experiment 2), on LC_FOLDS: key -> (label, hidden layers, raw inputs?)
    MLP_ABLATIONS=dict(
        raw=("4×512 on the raw inputs", [512, 512, 512, 512], True),
        h1x512=("1×512 on the bits", [512], False),
        h4x64=("4×64 on the bits", [64, 64, 64, 64], False),
    ),
)
if QUICK:  # dry run: 3 folds, 3,000 rows, small grids, separate folders, no predictions
    K, ROWS, OUT, FIG, WRITE_PREDICTIONS = 3, 3000, "exp_cache_quick", "exp_figures_quick", False
    G.update(
        C_L2=[1e-2, 1e-1],
        MAX_DEG={20: 2, 30: 2, 40: 2},
        KP_DEGS={n: [10, n] for n in NS},
        RF_TREES=30,
        RF_MAXF=["sqrt", 1.0],
        RF_LEAF=[5],
        MLP_MAXITER=40,
        MLP_PATIENCE=10,
        WK_DEGS=[6, 10],
        WK_C=[10],
        LC_SIZES=[100, 250, 500],
        LC_FOLDS=(0, 1),
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
    kpoly="Kernel LR",
    wkpoly="Weighted kernel LR",
    ukpoly="Equal-weight kernel LR",
    rf="Random forest",
    mlp="MLP 4×512",
    paper="Paper baseline",
)


def kname(key):
    """Full name of a (family, condition, sub-model) key."""
    fam, cond, sub = key
    return NAME[fam] + (f", degree {sub[3:]}" if sub.startswith("deg") else "")


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


CV, LC, WTS, FIN = (Store(nm) for nm in ("cv_results", "learning_curves", "weights", "final"))


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
    (the bit weights inside a model fit) is counted in the outer one."""
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
        cpu=d.cpu_model or None,
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


# ---- Cross-validation and stored results
def acc(m, X, y):
    return float((m.predict(X) == y).mean())


def _capped(est):
    """True if the fit stopped at its iteration cap instead of converging."""
    m = est.steps[-1][1] if isinstance(est, Pipeline) else est
    if hasattr(m, "iters_"):  # the kernel models
        return m.iters_ >= m.maxiter
    if hasattr(m, "n_iter_") and hasattr(m, "max_iter"):  # LogisticRegression, MLPClassifier
        return int(np.max(m.n_iter_)) >= m.max_iter
    return False


def cv_eval(make, n, fam="", size=None, folds=None, shuffle_labels=False, raw=False):
    """K-fold CV on dev: fit make() on the training folds, score on the held-out fold, so models
    can be compared fold by fold. size: use only that many training rows (a fixed random subset
    of each fold-train set). raw: the noisy inputs instead of their signs (the paper baseline
    always uses them)."""
    X, B, y = load(n)
    M = X if fam == "paper" or raw else B
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


def groups(n):
    """One entry per (family, condition, sub-model) with its CV-best setting, best first."""
    d = {}
    for r in CV.rows.values():
        if r["n"] == n and len(r["folds"]) == K and r["family"] in NAME and r["cond"] == "all":
            d.setdefault((r["family"], r["cond"], r["sub"]), []).append(r)
    return sorted(
        [dict(key=k, best=max(rs, key=lambda r: r["mean"]), configs=rs) for k, rs in d.items()],
        key=lambda g: -round(g["best"]["mean"], 9),
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


def best_of(n, cond, fams=("lr_l2",), sub=None):
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
    """Save to FIG; show it in a notebook, close it when run as a script."""
    fig.savefig(f"{FIG}/{name}.pdf")
    if "IPython" in sys.modules:
        plt.show()
    else:
        plt.close(fig)


def check_caps():
    """Every stored fit that stopped at its iteration cap instead of converging."""
    rows = [r for st in (CV, LC, FIN) for r in st.rows.values() if r.get("capped")]
    if not rows:
        print("No fit stopped at its iteration cap.")
    for r in rows:
        print(f"{r['key']}: {r['capped']} fold(s) stopped at the iteration cap")
