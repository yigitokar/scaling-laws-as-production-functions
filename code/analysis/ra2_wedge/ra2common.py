"""common.py -- paths, constants and small helpers for module ra2_wedge (the re-specified inversion).

Notation (paper/notes/model_spec.md, theory_main_text.md): u = A N^-alpha, v = B D^-beta, w = eps_N/eps_D,
M = D/N, C = 6ND, S = alpha+beta, a = beta/S, sigma* = 2/(2+S), 1/sigma* - 1 = S/2.
Generalized wedge (revision_plan A): under any objective V(L, N) - training cost, w - 1 = (marginal value of
compactness at fixed quality) / (training cost); in the developer-serves case w - 1 = lambda p T/(3D) = planned
serving expenditure / training expenditure. Headline object: the expenditure share s = (w - 1)/w.
Within the Chinchilla (and kappa) family the wedge depends on the technology only through (S, M*(C)):
    ln w = (S/2) [ln M - ln M*(C)],   ln M*(C) = -2 ln G + (1 - 2a) ln(C/6).
Module m3_wedge is re-used read-only (imported from code/analysis/m3_wedge); sl.py is never edited.
"""
from __future__ import annotations

import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
M3_CODE = os.path.join(ANALYSIS, "m3_wedge")
for _p in (ANALYSIS, M3_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sl  # noqa: E402,F401  (read-only)
import aer_style  # noqa: E402,F401

RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed", "ra2_wedge")
M3_PROC = os.path.join(ROOT, "data", "processed", "m3_wedge")
M1_PROC = os.path.join(ROOT, "data", "processed", "m1_chinchilla")
M2_PROC = os.path.join(ROOT, "data", "processed", "m2_techpanel")
TABLES = os.path.join(ROOT, "output", "tables")
FIGS = os.path.join(ROOT, "output", "figures")
MEMOS = os.path.join(ROOT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

PREFIX = "ra2_wedge"
SEED = 20260924
N_PROC = 5                 # CPU processes (GPU is reserved for the MLX training queues)
SNAPSHOT = "2026-09-24"
RA1_SIGMA = os.path.join(TABLES, "ra1_modelfree_sigma.csv")   # legacy name (never produced by ra1)
# ra1_modelfree's reviewed output (2026-09-24): model-free sigma* per IsoFLOP design; primary = random-effects mean over
# valid budgets ('sigma_re', 'se_sigma_re'), quadratic h = 1 path-centred windows, log-cubic frontier (ra1 memo H2).
RA1_SUMMARY = os.path.join(TABLES, "ra1_modelfree_isoflop_summary.csv")
# ra1's Farseer local (model-free) wedge vs parametric forms by M bin (ra1 memo H4; non-embedding and total N)
RA1_FARSEER_DELTA = os.path.join(TABLES, "ra1_modelfree_farseer_delta.csv")
RA1_FARSEER_SENS = os.path.join(TABLES, "ra1_modelfree_farseer_sensitivity.csv")

# ----------------------------------------------------------------------------- closed forms


def log_w(lnN, lnD, alpha, beta, lnG):
    """ln w = (alpha+beta) ln G - alpha ln N + beta ln D  (Chinchilla/kappa family; broadcasts)."""
    return (alpha + beta) * lnG - alpha * lnN + beta * lnD


def lnG_of(A, B, alpha, beta):
    return (np.log(alpha * A) - np.log(beta * B)) / (alpha + beta)


def ln_mstar(C, alpha, beta, lnG):
    """ln M*(C) = -2 ln G + (1 - 2a) ln(C/6), a = beta/(alpha+beta)."""
    a = beta / (alpha + beta)
    return -2 * lnG + (1 - 2 * a) * np.log(np.asarray(C, float) / 6.0)


def ab_from_S_a(S, a):
    """(alpha, beta) with alpha+beta = S and beta/(alpha+beta) = a (hybrid technologies: path from one source,
    curvature from another)."""
    return S * (1 - a), S * a


def share(w):
    """Expenditure share of planned serving in lifetime cost, s = (w-1)/w (bounded above by 1; negative for w<1)."""
    if not hasattr(w, "values"):
        w = np.asarray(w, float)
    return (w - 1.0) / w


def fmt(x, p=2):
    try:
        if x is None or not np.isfinite(x):
            return ""
    except TypeError:
        return str(x)
    return f"{x:.{p}f}"


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(PROC, "run_log.txt"), "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- inference helpers

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_boot(y, X, cluster, test_idx, B=9999, seed=SEED, weights="webb"):
    """Restricted wild cluster bootstrap-t (WCR; Cameron, Gelbach and Miller 2008) with Webb (2023) six-point
    weights for H0: beta[test_idx] = 0 in y = X beta + e, clustered by `cluster`.
    Returns dict(beta, se_crv1, t, p_boot, B, G). p is the symmetric bootstrap p-value; its floor is 1/(B+1)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    cl = np.asarray(cluster)
    groups, gi = np.unique(cl, return_inverse=True)
    G = len(groups)
    n, k = X.shape

    def ols(yv):
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ X.T @ yv
        e = yv - X @ b
        meat = np.zeros((k, k))
        for g in range(G):
            idx = gi == g
            sg = X[idx].T @ e[idx]
            meat += np.outer(sg, sg)
        c = G / (G - 1) * (n - 1) / (n - k)
        V = c * XtX_inv @ meat @ XtX_inv
        return b, np.sqrt(max(V[test_idx, test_idx], 0.0))

    b, se = ols(y)
    t0 = b[test_idx] / se if se > 0 else np.nan
    # restricted fit (impose beta_test = 0)
    Xr = np.delete(X, test_idx, axis=1)
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fit_r = Xr @ br
    er = y - fit_r
    rng = np.random.default_rng(seed)
    tb = np.empty(B)
    XtX_inv = np.linalg.pinv(X.T @ X)
    for bb in range(B):
        v = WEBB[rng.integers(0, 6, G)] if weights == "webb" else rng.choice([-1.0, 1.0], G)
        ys = fit_r + er * v[gi]
        bs = XtX_inv @ X.T @ ys
        es = ys - X @ bs
        meat = np.zeros((k, k))
        for g in range(G):
            idx = gi == g
            sg = X[idx].T @ es[idx]
            meat += np.outer(sg, sg)
        c = G / (G - 1) * (n - 1) / (n - k)
        V = c * XtX_inv @ meat @ XtX_inv
        tb[bb] = bs[test_idx] / np.sqrt(max(V[test_idx, test_idx], 1e-300))
    p = (1 + np.sum(np.abs(tb) >= abs(t0))) / (B + 1)
    return dict(beta=float(b[test_idx]), se_crv1=float(se), t=float(t0), p_boot=float(p), B=B, G=G,
                p_floor=1.0 / (B + 1))
