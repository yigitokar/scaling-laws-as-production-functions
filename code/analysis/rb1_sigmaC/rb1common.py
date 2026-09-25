"""rb1common.py -- paths, seeds, logging and shared helpers for module rb1_sigmaC.

Module rb1_sigmaC ("the elasticity of substitution as a function of compute, and more extrapolation checks") answers
round-2 referee requests R1 New 2, R3 N1, R4 R2-M4, R2 Major 1 and Major 5 (paper/notes/revision_plan_v3.md).

It reads the reviewed outputs of module ra1_modelfree (budget-level model-free sigma*_b, Farseer's local path, design
summaries), module ra2_wedge (clean sample, reference zero points, partial-identification bounds on ln M - ln M*(C))
and module m8_measurement (Step Law inefficiency gradients), and re-uses their code by import (never edited).
The shared library code/analysis/sl.py is not edited.

Notation (paper/notes/model_spec.md): n = ln N, d = ln D, c = ln C, C = 6ND, M = D/N, w = eps_N/eps_D,
S = 2(1/sigma* - 1); the wedge scale is k = 1/sigma* - 1 = S/2; s = (w - 1)/w.
Model-free identity (ra1 H1): 1/sigma* - 1 = L_nn|_C / (2|dL*/d ln C|) at any compute-optimal point.
"""
from __future__ import annotations

import os
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
RA1DIR = os.path.join(ANALYSIS, "ra1_modelfree")
RA2DIR = os.path.join(ANALYSIS, "ra2_wedge")
M8DIR = os.path.join(ANALYSIS, "m8_measurement")
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
# ra1 first (its ra1_common inserts m1/m2 directories); ra2 only for its pure helpers
for _p in (ANALYSIS, RA1DIR):
    if _p not in sys.path:
        sys.path.append(_p)

import ra1_common as rc  # noqa: E402  (ra1's loaders, weights, spawn pool; imported unchanged)
import aer_style  # noqa: E402

# [rb4 integration] Chinchilla convention switch (documented at RA1_BUDGETS below). The 'ra1' sensitivity writes to
# its own root unless RB1_OUTPUT_ROOT is given, so that it cannot overwrite the primary outputs.
CHIN_SOURCE = os.environ.get("RB1_CHINCHILLA", "rb4").strip().lower()
if CHIN_SOURCE not in ("rb4", "ra1"):
    raise ValueError(f"RB1_CHINCHILLA must be 'rb4' (default) or 'ra1', not {CHIN_SOURCE!r}")
CHIN_REBUILT = CHIN_SOURCE == "rb4"
_OUT = os.environ.get("RB1_OUTPUT_ROOT",
                      ROOT if CHIN_REBUILT else os.path.join(ROOT, "output", "sensitivity", "rb1_chinchilla_T"))
PROC = os.path.join(_OUT, "data", "processed", "rb1_sigmaC")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

# upstream (read-only) outputs
UP_TABLES = os.path.join(ROOT, "output", "tables")
RA1_BUDGETS = os.path.join(UP_TABLES, "ra1_modelfree_isoflop_budgets.csv")
RA1_SUMMARY = os.path.join(UP_TABLES, "ra1_modelfree_isoflop_summary.csv")
RA1_SENS = os.path.join(UP_TABLES, "ra1_modelfree_isoflop_sens.csv")
RA1_FAR_PATH = os.path.join(UP_TABLES, "ra1_modelfree_farseer_path.csv")
RA1_FAR_POOL = os.path.join(UP_TABLES, "ra1_modelfree_farseer_pool.csv")
RA1_FAR_SLOPE = os.path.join(UP_TABLES, "ra1_modelfree_farseer_sigma_slope.csv")
RA1_FAR_SENS = os.path.join(UP_TABLES, "ra1_modelfree_farseer_sensitivity.csv")
RA1_FAR_DELTA = os.path.join(UP_TABLES, "ra1_modelfree_farseer_delta.csv")
RA1_FAR_LIN = os.path.join(UP_TABLES, "ra1_modelfree_farseer_lin.csv")
RA1_FAR_LNW = os.path.join(UP_TABLES, "ra1_modelfree_farseer_lnw.csv")
RA1_FAR_CV = os.path.join(UP_TABLES, "ra1_modelfree_farseer_bandwidth_cv.csv")
RA1_FAR_GRID = os.path.join(UP_TABLES, "ra1_modelfree_farseer_grid.csv")
RA1_MARIN_ACC = os.path.join(UP_TABLES, "ra1_modelfree_isoflop_marin_flop_accounting.csv")

# [rb4 integration, 2026-09-24] Chinchilla's parameter convention (switch RB1_CHINCHILLA; default "rb4").
#   "rb4" (primary): Chinchilla in FLOP-effective parameters N_F = F_T4/6, with FLOPs per token rebuilt from Hoffmann
#         et al.'s architecture table (module rb4_chinflop; 0.660 (0.023)). ra1's two IsoFLOP inputs are replaced by
#         rb4's override copies, in which only Chinchilla's rows differ (rb4_prop.write_overrides, run = 'new'). The
#         study-level variants then carry the FLOP-accounting elasticity for Meta only (Chinchilla's accounting is
#         observed), plus rows for Chinchilla's other FLOP counts and conventions (rb4_chinflop_summary.csv) and for its
#         window/membership sensitivities (rb4_chinflop_review_checks.csv, R11-R13).
#   "ra1" (sensitivity; rb1 as first published): Chinchilla in total parameters T, as in ra1. Outputs go to
#         output/sensitivity/rb1_chinchilla_T/ unless RB1_OUTPUT_ROOT is set.
# Not switched (still Chinchilla in T): the drift Monte Carlo (driftmc.py) and the symmetric-window check (symwin.py),
# which re-estimate from ra1's loaders; rb4's review stage (R14) reports symwin in N_F.
RA1_BUDGETS_RA1, RA1_SUMMARY_RA1 = RA1_BUDGETS, RA1_SUMMARY
RB4_PROC = os.path.join(ROOT, "data", "processed", "rb4_chinflop")
RB4_SUMMARY = os.path.join(UP_TABLES, "rb4_chinflop_summary.csv")
RB4_CHECKS = os.path.join(UP_TABLES, "rb4_chinflop_review_checks.csv")
if CHIN_REBUILT:
    RA1_BUDGETS = os.path.join(RB4_PROC, "override_new_isoflop_budgets.csv")
    RA1_SUMMARY = os.path.join(RB4_PROC, "override_new_isoflop_summary.csv")
RA2_MODELS = os.path.join(UP_TABLES, "ra2_wedge_models.csv")
RA2_FAMW = os.path.join(UP_TABLES, "ra2_wedge_family_level_W.csv")
RA2_SIGSENS = os.path.join(UP_TABLES, "ra2_wedge_sigma_sensitivity.csv")
RA2_TECHS = os.path.join(UP_TABLES, "ra2_wedge_technologies.csv")
RA2_PI_BOUNDS = os.path.join(ROOT, "data", "processed", "ra2_wedge", "pi_bounds_models.csv")
RA2_PI_SUMMARY = os.path.join(UP_TABLES, "ra2_wedge_pi_summary.csv")
M8_POLICIES = os.path.join(UP_TABLES, "m8_measurement_steplaw_policies.csv")

PREFIX = "rb1_sigmaC"
SEED = 20260925
N_PROC = 4                      # at most 4 CPU processes (GPU training queues running; GPU not used)
QUICK = bool(os.environ.get("RB1_QUICK"))

# reference technology of ra2 (kappa-free Chinchilla, Huber n = 240): a1 = 0.4244, b1 = 0.4305
REF_K = None                    # set from the data (slope of ln w_ref on ln(M/M*_ref)) in pid.py
SIGMA_REF = 0.701               # sigma*_kappa of the reference (ra2)


def seed_of(key: str) -> int:
    return SEED + zlib.crc32(key.encode()) % 1_000_000


_T0 = time.time()
_LOGF = None


def log(*a):
    global _LOGF
    msg = time.strftime("%H:%M:%S") + f" [{time.time() - _T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOGF is None:
        _LOGF = open(os.path.join(PROC, "run_log.txt"), "a")
    _LOGF.write(msg + "\n")
    _LOGF.flush()


def tab(df: pd.DataFrame, name: str, **kw):
    path = os.path.join(TABLES, f"{PREFIX}_{name}.csv")
    df.to_csv(path, index=False, float_format=kw.get("float_format", "%.6g"))
    return path


def share(w):
    w = np.asarray(w, float)
    return (w - 1.0) / w


def sigma_to_k(sig):
    return 1.0 / np.asarray(sig, float) - 1.0


def k_to_sigma(k):
    return 1.0 / (1.0 + np.asarray(k, float))


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def write_tex(path, body):
    with open(path, "w") as f:
        f.write(body)
    return path
