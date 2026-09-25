"""rb4common.py -- paths, seeds, logging and import plumbing for module rb4_chinflop.

Module rb4_chinflop ("Chinchilla's FLOP accounting rebuilt") answers round-2 Referee 2, Major 1 request 4 (round-1
Major 9(b)): rebuild FLOPs per token for every model of Hoffmann et al. (2022, Table A9) with their Appendix F count,
and recompute the model-free sigma* of Chinchilla in the paper's primary convention, FLOP-effective parameters
N_F = C/(6D) with C the actual training FLOPs, and in the non-embedding (P) and total (T) conventions. The rebuilt
Chinchilla estimates are then propagated into the study-level interval and the sigma*(C) meta-regression of rb1.

Code of ra1_modelfree (estimator, bootstrap, loaders) and rb1_sigmaC (meta-analysis, meta-regression) is imported
unchanged. rb1's module-level input paths (rb1common.RA1_BUDGETS / RA1_SUMMARY) are redirected to override copies in
which only the Chinchilla rows are replaced (derived statistics only); rb1's own log/outputs go to a temporary
directory so that nothing of rb1 is touched.

Data licence: Epoch AI's digitization (data/raw/epoch_chinchilla) has no redistribution licence. Nothing in this
module writes digitized rows (N, C, loss of a run) to a committed file; outputs are derived statistics only.
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

_OUT = os.environ.get("RB4_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "rb4_chinflop")
TABLES = os.path.join(_OUT, "output", "tables")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, MEMOS):
    os.makedirs(_d, exist_ok=True)

# rb1's module writes a log and creates output folders at import: send them to a throw-away root
os.environ.setdefault("RB1_OUTPUT_ROOT", tempfile.mkdtemp(prefix="rb4_rb1shadow_"))
# [rb4 integration, 2026-09-24] rb1's inputs now default to this module's rebuilt Chinchilla (RB1_CHINCHILLA = 'rb4').
# This module builds its overrides from, and validates its hook against, ra1's original inputs (Chinchilla in T):
# pin rb1 to them.
os.environ["RB1_CHINCHILLA"] = "ra1"

# import order: rb1common (puts rb1 first, appends ra1), then ra1_common (moves ra1 to the front, appends m1/m2)
if RB1DIR not in sys.path:
    sys.path.insert(0, RB1DIR)
import rb1common as cm1  # noqa: E402  (rb1, unchanged)
import ra1_common as rc  # noqa: E402  (ra1, unchanged)
# this module's directory must win over both (file names here are prefixed, but be safe)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
RAW_RB4 = os.path.join(RAW, "rb4_chinflop")
ARXIV_URL = "https://arxiv.org/e-print/2203.15556"
ARXIV_TGZ = os.path.join(RAW_RB4, "arxiv_2203.15556_src.tar.gz")
ARXIV_SHA256 = "6571e76a9ec57ab9d52cef6ff62e01887306c1c549a57eb148f9a407c57d9900"
ARXIV_TEX = os.path.join(RAW_RB4, "src", "main.tex")

UP_TABLES = os.path.join(ROOT, "output", "tables")
RB1_EXTRAP_PKL = os.path.join(ROOT, "data", "processed", "rb1_sigmaC", "stage_extrap.pkl")

PREFIX = "rb4_chinflop"
SEED = 20260926
B_ISO = 999                      # = ra1's B_ISO for the primary design bootstrap
B_META = 9999                    # = rb1's wild-cluster bootstrap draws for the meta-regression
N_PROC = 4                       # at most 4 CPU processes (GPU training queues running; GPU never used)
QUICK = bool(os.environ.get("RB4_QUICK"))
if QUICK:
    B_ISO, B_META = 99, 999


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


def sig_from_S(S):
    return 2.0 / (2.0 + np.asarray(S, float))


def S_from_sig(s):
    return 2.0 / np.asarray(s, float) - 2.0


def eta_equivalent(sig_from, sig_to):
    """eta such that sigma_to = 2/(2 + S_from/(1+eta)^2) (rb1's shift_eta), i.e. (1+eta)^2 = S_from/S_to."""
    return float(np.sqrt(S_from_sig(sig_from) / S_from_sig(sig_to)) - 1.0)


def shift_eta(sig, eta):
    """rb1/ra1 FLOP-accounting shift: sigma*_eff = 2/(2 + S/(1+eta)^2)."""
    S = S_from_sig(sig)
    return float(2.0 / (2.0 + S / (1.0 + eta) ** 2))
