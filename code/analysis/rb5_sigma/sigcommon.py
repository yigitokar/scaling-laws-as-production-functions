"""sigcommon.py -- paths, seeds, logging and import plumbing for module rb5_sigma.

Module rb5_sigma ("sigma* re-analysis bundle", round-3 fix list item T1, the analysis part of T14, and the new numbers
behind the figure items T10 and T16; paper/notes/round3_fixlist.md) answers round-3 referee requests R1 New 2(b), (d),
minors 3, 4, 7; R2 NM2(1), (3), NM3(b), (c), the smaller point of NM3, minor 17; R4 N3(b)-(f), N3 requests 2-3,
minors 5, 14; and audit numbers_technology #4 ("better fix").

It reads the reviewed outputs and code of modules ra1_modelfree (estimator, design-conditional bootstrap, Farseer local
surface and its bootstrap cache), rb1_sigmaC (meta-regression machinery, local-wedge machinery and its bootstrap cache),
rb4_chinflop (Chinchilla in FLOP-effective parameters N_F; override inputs and specification grid) and m2_techpanel
(DataDecide loader and panel fits). Their code is imported unchanged; nothing of theirs is written.

Import order matters. rb1common reads RB1_CHINCHILLA at its first import (primary 'rb4': Chinchilla in N_F) and
rb4common forces RB1_CHINCHILLA='ra1' in the environment for its own purposes; rb1common is therefore imported first,
and the environment is restored afterwards so that spawned workers see the primary setting. rb1's and rb4's own log
files and output folders are redirected to a throw-away directory, so that importing them writes nothing in the
repository.

Data licence: Epoch AI's digitization of Chinchilla (data/raw/epoch_chinchilla) and the Llama 3 digitization carry no
redistribution licence. This module writes derived statistics only (budget-level estimates and their bootstrap draws,
never a run's N, D or loss).

CPU only (the GPU is training): at most 2 worker processes; run under `nice -n 10`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
RA1DIR = os.path.join(ANALYSIS, "ra1_modelfree")
RB1DIR = os.path.join(ANALYSIS, "rb1_sigmaC")
RB4DIR = os.path.join(ANALYSIS, "rb4_chinflop")
M2DIR = os.path.join(ANALYSIS, "m2_techpanel")

# ============================================================================ upstream modules (imported unchanged)
_SHADOW = os.environ.get("RB5_SHADOW_ROOT") or tempfile.mkdtemp(prefix="rb5_shadow_")
os.environ["RB5_SHADOW_ROOT"] = _SHADOW               # inherited by spawned workers: one shadow root per run
os.environ["RB1_OUTPUT_ROOT"] = os.path.join(_SHADOW, "rb1")
os.environ["RB4_OUTPUT_ROOT"] = os.path.join(_SHADOW, "rb4")
os.environ["RB1_CHINCHILLA"] = "rb4"                  # primary: Chinchilla in N_F (rb4's override inputs)
if RB1DIR not in sys.path:
    sys.path.insert(0, RB1DIR)
import rb1common as cm1  # noqa: E402  (rb1; puts rb1 first and appends ra1)
import ra1_common as rc  # noqa: E402  (ra1; moves ra1 to the front and appends m1/m2)
if RB4DIR not in sys.path:
    sys.path.append(RB4DIR)
import rb4common as cm4  # noqa: E402  (rb4; sets RB1_CHINCHILLA='ra1' in the environment)
os.environ["RB1_CHINCHILLA"] = "rb4"                  # restore for spawned workers (rb1common already imported)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
assert cm1.CHIN_REBUILT, "rb1common must be in its primary mode (Chinchilla in N_F)"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aer_style  # noqa: E402

# ============================================================================ outputs
_OUT = os.environ.get("RB5_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "rb5_sigma")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

# ============================================================================ upstream inputs (read-only)
UP = os.path.join(ROOT, "output", "tables")
UP_PROC = os.path.join(ROOT, "data", "processed")
RB4_BUDGETS = os.path.join(UP_PROC, "rb4_chinflop", "override_new_isoflop_budgets.csv")   # primary budgets (N_F)
RB4_SUMMARY = os.path.join(UP_PROC, "rb4_chinflop", "override_new_isoflop_summary.csv")
RB4_SPECGRID = os.path.join(UP, "rb4_chinflop_specgrid.csv")
RB4_CHECKS = os.path.join(UP, "rb4_chinflop_review_checks.csv")
RB1_STUDY = os.path.join(UP, "rb1_sigmaC_study_level.csv")
RB1_SLOPES = os.path.join(UP, "rb1_sigmaC_metareg_slopes.csv")
RB1_PRED = os.path.join(UP, "rb1_sigmaC_metareg_predictions.csv")
RB1_TOP = os.path.join(UP, "rb1_sigmaC_top_budget_sigma.csv")
RB1_BUDGETS_IN = os.path.join(UP, "rb1_sigmaC_budgets_input.csv")
RB1_PI = os.path.join(UP, "rb1_sigmaC_pi_sigmaC.csv")
RB1_CONV = os.path.join(UP, "rb1_sigmaC_conventions.csv")
RB1_EXTRAP_CONVEXITY = os.path.join(UP, "rb1_sigmaC_extrap_convexity.csv")
RB1_EXTRAP_DELTA = os.path.join(UP, "rb1_sigmaC_extrap_delta.csv")
RB1_EXTRAP_POOLED = os.path.join(UP, "rb1_sigmaC_extrap_marin_pooled.csv")
RB1_EXTRAP_PKL = os.path.join(UP_PROC, "rb1_sigmaC", "stage_extrap.pkl")
RA1_FARSEER_PKL = os.path.join(UP_PROC, "ra1_modelfree", "stage_farseer.pkl")
RA1_MC = os.path.join(UP, "ra1_modelfree_isoflop_mc.csv")
RA1_ORDER = os.path.join(UP, "ra1_modelfree_isoflop_order.csv")
RA1_FAR_SENS = os.path.join(UP, "ra1_modelfree_farseer_sensitivity.csv")
RA1_FAR_SLOPE = os.path.join(UP, "ra1_modelfree_farseer_sigma_slope.csv")
RA1_FAR_POOL = os.path.join(UP, "ra1_modelfree_farseer_pool.csv")
RA1_FAR_PATH = os.path.join(UP, "ra1_modelfree_farseer_path.csv")
RA1_FAR_DELTA = os.path.join(UP, "ra1_modelfree_farseer_delta.csv")
M2_MAGS = os.path.join(UP, "m2_neutrality_magnitudes.csv")
M2_GROUPS_DD = os.path.join(UP, "m2_neutrality_groups_datadecide.csv")
CLEAN_MODELS = os.path.join(UP_PROC, "rb2_decisions", "clean_models.csv")

PREFIX = "rb5_sigma"
SEED = 20260927                 # this module's master seed (new draws); reproductions use the upstream seeds
N_PROC = 2                      # at most 2 CPU worker processes (GPU queues A-E training)
QUICK = bool(os.environ.get("RB5_QUICK"))

# reference technology (kappa-free Chinchilla, 240 runs, total parameters; ra2/m2): inner exponents
REF_ALPHA, REF_BETA = 0.424359, 0.430527
DESIGNS_ISO = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]
FARSEER = "Farseer (local path)"
STUDY = {"Chinchilla": "Hoffmann", "Llama 3": "Meta", "Marin, Comma": "Marin", "Marin, DCLM": "Marin",
         "Marin, Nemotron-CC": "Marin", FARSEER: "Farseer"}


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


def tab(df: pd.DataFrame, name: str, float_format="%.6g"):
    path = os.path.join(TABLES, f"{PREFIX}_{name}.csv")
    df.to_csv(path, index=False, float_format=float_format)
    return path


def sig_from_S(S):
    return 2.0 / (2.0 + np.asarray(S, float))


def dsig_dS(S):
    return -2.0 / (2.0 + np.asarray(S, float)) ** 2


def pmap(fn, items, payload, chunksize=1):
    """ra1's spawn pool with this module's process cap."""
    return rc.pmap(fn, items, payload, procs=N_PROC, chunksize=chunksize)
