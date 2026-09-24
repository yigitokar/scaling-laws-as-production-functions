"""common.py -- shared paths and helpers for module m8_measurement.

Nothing here edits the shared library sl.py; helpers that wrap it live in this folder.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one BLAS thread per process (<= 6 processes in total)

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

import sl  # noqa: E402  (shared estimation library; read-only)
import aer_style  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed", "m8_measurement")
TAB = os.path.join(ROOT, "output", "tables")
FIG = os.path.join(ROOT, "output", "figures")
MEMO = os.path.join(ROOT, "output", "memos")
for _d in (PROC, TAB, FIG, MEMO):
    os.makedirs(_d, exist_ok=True)

N_JOBS = 6  # hard cap on CPU processes (a GPU job is running on this machine)


# ----------------------------------------------------------------------------- fitting wrappers

# 16-start safety grid used together with a warm start (the full-sample estimate) in bootstrap and Monte Carlo
# replications; run.py checks on the first replications that it reaches the same optimum as sl.FAST_GRID.
MINI_GRID = dict(a=np.array([5.0, 20.0]), b=np.array([5.0, 20.0]), e=np.array([0.5]),
                 alpha=np.array([0.3, 0.6]), beta=np.array([0.3, 0.6]))

def fit_efixed(N, D, L, E, delta=1e-3, grid=None, init=None):
    """Chinchilla fit with E held fixed -- local replacement for sl.fit_chinchilla(E_fixed=...).
    WORKAROUND: sl.fit_chinchilla's E_fixed branch calls L-BFGS-B with default tolerances (ftol ~ 2.2e-9 in
    absolute terms when |objective| < 1), which stops far from the optimum when the objective is O(1e-4)
    (small samples, Huber loss). Here we use the same objective/gradient (sl._obj, sl._grad) with the tight
    tolerances of sl's free-E branch. Starts: `init` (if given) plus every point of `grid` (default FAST_GRID)."""
    from scipy.optimize import minimize
    lN, lD, lL = np.log(np.asarray(N, float)), np.log(np.asarray(D, float)), np.log(np.asarray(L, float))
    w = np.ones_like(lL)
    e0 = np.log(E)
    grid = sl.FAST_GRID if grid is None else grid
    starts = [] if grid is False else [np.array(st, float) for st in
                                       itertools.product(grid["a"], grid["b"], grid["e"], grid["alpha"], grid["beta"])]
    if init is not None:
        starts = [np.asarray(init, float)] + starts
    # the e-coordinate of each start is dropped (E is fixed), so grid points that differ only in e are identical
    # 4-vectors: keep the first occurrence of each (review fix; results unchanged, 3-5x fewer optimizations)
    starts4, seen = [], set()
    for st in starts:
        s4 = np.delete(st, 2)
        key = tuple(np.round(s4, 12))
        if key not in seen:
            seen.add(key)
            starts4.append(s4)
    f = lambda t4: sl._obj(np.r_[t4[:2], e0, t4[2:]], lN, lD, lL, delta, w)
    g = lambda t4: np.delete(sl._grad(np.r_[t4[:2], e0, t4[2:]], lN, lD, lL, delta, w), 2)
    best = None
    for s4 in starts4:
        r = minimize(f, s4, jac=g, method="L-BFGS-B", options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best[1]:
            best = (np.r_[r.x[:2], e0, r.x[2:]], r.fun)
    return sl.Chinchilla.from_theta(best[0], objective=float(best[1]), n=len(lL), delta=delta, E_fixed=E)


def fit_warm(N, D, L, init=None, grid=None, delta=1e-3, E_fixed=None):
    """Hoffmann/Besiroglu Huber-LSE fit (sl.fit_chinchilla) from a warm start *and* a coarse grid;
    keeps the best objective. Used for bootstrap/Monte Carlo replications."""
    if E_fixed is not None:
        return fit_efixed(N, D, L, E_fixed, delta=delta, grid=grid, init=init)
    best = None
    if init is not None:
        best = sl.fit_chinchilla(N, D, L, delta=delta, init=np.asarray(init, float), E_fixed=E_fixed)
    if grid is not False:
        m = sl.fit_chinchilla(N, D, L, delta=delta, grid=grid or sl.FAST_GRID, E_fixed=E_fixed)
        if best is None or m.extra["objective"] < best.extra["objective"] - 1e-12:
            best = m
    return best


def model_row(m, **extra):
    """Flat dict of the technology objects the paper reports everywhere (model_spec.md Section 3)."""
    s = m.summary()
    out = dict(E=s["E"], A=s["A"], B=s["B"], alpha=s["alpha"], beta=s["beta"], a=s["a_N"], gamma=s["gamma"],
               sigma_star=s["sigma_star"], objective=m.extra.get("objective", np.nan), n=m.extra.get("n", np.nan))
    out.update(extra)
    return out


# ----------------------------------------------------------------------------- weighted power-law fit

def wls_loglog(x, y, w=None):
    """Weighted least squares of log y on log x. Returns (slope, intercept, unweighted R^2)."""
    X = np.log(np.asarray(x, float))
    Y = np.log(np.asarray(y, float))
    w = np.ones_like(X) if w is None else np.asarray(w, float)
    W = w / w.sum()
    xm, ym = np.sum(W * X), np.sum(W * Y)
    b = np.sum(W * (X - xm) * (Y - ym)) / np.sum(W * (X - xm) ** 2)
    a0 = ym - b * xm
    r = Y - (a0 + b * X)
    r2 = 1 - np.sum(r ** 2) / np.sum((Y - Y.mean()) ** 2)
    return b, a0, r2


# ----------------------------------------------------------------------------- LaTeX helpers

def fmt(x, d=3):
    """Number for a LaTeX table cell (text mode): proper minus sign, '--' for missing."""
    if x is None or not np.isfinite(float(x)):
        return "--"
    s = f"{float(x):.{d}f}"
    return "$-$" + s[1:] if s.startswith("-") and float(s) != 0 else s.lstrip("-")


def se_cell(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"({x:.{d}f})"


def write_tex(path, body_rows, colspec, header_rows, caption, label, notes, size=r"\small", tabcolsep=None):
    """AER-style booktabs table inside a threeparttable; body_rows/header_rows are lists of LaTeX lines
    (without trailing \\\\) or the literal string 'MIDRULE'. tabcolsep (e.g. '3.5pt') tightens column spacing."""
    lines = [r"\begin{table}[htbp]", r"\centering", size]
    if tabcolsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines += [r"\begin{threeparttable}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    for h in header_rows:
        if h == "MIDRULE":
            lines.append(r"\midrule")
        elif h.startswith("\\cmidrule") or h.startswith("\\addlinespace"):
            lines.append(h)
        else:
            lines.append(h + r" \\")
    if header_rows:
        lines.append(r"\midrule")
    for r in body_rows:
        if r == "MIDRULE":
            lines.append(r"\midrule")
        elif r.startswith("\\addlinespace") or r.startswith("\\cmidrule"):
            lines.append(r)
        else:
            lines.append(r + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[flushleft]", r"\footnotesize",
              rf"\item \textit{{Notes:}} {notes}", r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parallel_map(fn, items, n_jobs=N_JOBS):
    """Process-pool map capped at N_JOBS workers (fn must be picklable/top-level)."""
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, items, chunksize=max(1, len(items) // (4 * n_jobs))))
