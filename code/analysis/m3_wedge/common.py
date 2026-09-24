"""common.py -- paths, constants and small helpers for module m3_wedge (revealed inference demand).

Notation follows paper/notes/model_spec.md:
  u = A N^-alpha, v = B D^-beta, w = eps_N/eps_D = alpha u/(beta v), M = D/N, C = 6ND,
  a = beta/(alpha+beta), gamma = alpha beta/(alpha+beta), sigma* = 2/(2+alpha+beta), G = (alpha A/(beta B))^(1/(alpha+beta)).
Proposition 4: with lifetime cost 6ND + 2NT, the FOC gives w = 1 + T/(3D), so T = 3D(w - 1) and w is the ratio of
lifetime to training compute. Farrell cost efficiency CE = C_min(L(N,D))/(6ND).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

import sl  # noqa: E402  (shared library; read-only here)
import aer_style  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
HFRAW = os.path.join(RAW, "m3_hf")
_OUT = os.environ.get("M3_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "m3_wedge")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

# upstream module outputs (read-only)
M1_PROC = os.path.join(ROOT, "data", "processed", "m1_chinchilla")
M2_PROC = os.path.join(ROOT, "data", "processed", "m2_techpanel")
REG_M1 = os.path.join(ROOT, "output", "tables", "technology_registry_m1.csv")
REG_M2 = os.path.join(ROOT, "output", "tables", "technology_registry_m2.csv")

SEED = 20260923
N_PROC = 6
SNAPSHOT = "2026-09-23"   # date of the Epoch / OpenRouter / HF snapshots

# ----------------------------------------------------------------------------- closed forms (model_spec Prop. 4)


def log_w(lnN, lnD, alpha, beta, lnG):
    """ln w = (alpha+beta) ln G - alpha ln N + beta ln D  (w = eps_N/eps_D under the Chinchilla form).
    Broadcasts: technology arrays (draws) against model arrays."""
    return (alpha + beta) * lnG - alpha * lnN + beta * lnD


def lnG_of(A, B, alpha, beta):
    return (np.log(alpha * A) - np.log(beta * B)) / (alpha + beta)


def ce_of_w(w, alpha, beta):
    """Farrell input-oriented cost efficiency relative to the training-only frontier, as a function of the wedge
    only (m7 Lemma A5): C/C_min = ((alpha + beta w)/(alpha+beta))^(1/gamma) w^(-1/alpha); CE = C_min/C."""
    gam = alpha * beta / (alpha + beta)
    lnCC = np.log((alpha + beta * w) / (alpha + beta)) / gam - np.log(w) / alpha
    return np.exp(-lnCC)


def mstar(C, alpha, beta, lnG):
    """Training-only compute-optimal tokens per parameter at compute C: M* = D*/N* = G^-2 (C/6)^(1-2a)."""
    a = beta / (alpha + beta)
    return np.exp(-2 * lnG + (1 - 2 * a) * np.log(np.asarray(C, float) / 6.0))


def homothetic_w(M, Mstar, rho):
    """Homothetic CES (alpha = beta = rho): w = (M/M*)^rho."""
    return (np.asarray(M, float) / Mstar) ** rho


def homothetic_ce(M, Mstar, rho):
    """C/C_min = cosh(rho/2 ln(M/M*))^(2/rho) -> CE = its inverse."""
    x = np.log(np.asarray(M, float) / Mstar)
    return np.cosh(rho / 2 * x) ** (-2.0 / rho)


def fmt(x, p=2):
    if x is None:
        return ""
    try:
        if not np.isfinite(x):
            return ""
    except TypeError:
        return str(x)
    return f"{x:.{p}f}"


def log(msg):
    import datetime
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(PROC, "run_log.txt"), "a") as f:
        f.write(line + "\n")
