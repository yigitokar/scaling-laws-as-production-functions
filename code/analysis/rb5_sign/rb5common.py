"""rb5common.py -- paths, constants and small helpers for module rb5_sign (round 3, work package WP4a, items S1-S5 of
paper/notes/round3_fixlist.md; decision D-3).

The module rebuilds the sign-identification result of Proposition 3(v) / A6(vii)-(viii) without the post hoc anchor
rule and with the sampling error of the path slopes. It imports module rb2_decisions (and through it ra2_wedge)
read-only and never calls a function that writes into another module's folders (rb2's `log` is never called).

Notation (paper/notes/round3_fixlist.md section 0.8): M = D/N, c = ln C, x*(c) = ln M*(C), path slope e,
tilt allowance tau, curvature k = 1/sigma* - 1, compactness share s = (w - 1)/w.
Style rule of the author: no em dashes in any generated text (tables, figure text, logs, memos).
"""
from __future__ import annotations

import datetime
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
RB2 = os.path.join(ANALYSIS, "rb2_decisions")
RA2 = os.path.join(ANALYSIS, "ra2_wedge")
for _p in (ANALYSIS, RA2, RB2):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

PREFIX = "rb5_sign"
SEED = 20260925
PROC = os.path.join(ROOT, "data", "processed", "rb5_sign")
TABLES = os.path.join(ROOT, "output", "tables")
FIGS = os.path.join(ROOT, "output", "figures")
MEMOS = os.path.join(ROOT, "output", "memos")
RB2_PROC = os.path.join(ROOT, "data", "processed", "rb2_decisions")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------------------------- inputs (parameters)
DEFAULT_MODELS = os.path.join(RB2_PROC, "clean_models.csv")        # clean sample (77 models), rb2 output, read-only
# [review] the default is WP4b's final unit file (49 budget-level units, readings_final.csv codes); it is the same
# partition of the 77 models as rb2's units_subgroup.csv (checked), so every output is unchanged by the switch.
DEFAULT_UNITS = os.path.join(ROOT, "data", "processed", "rb5_units", "units_primary.csv")   # 49 units (primary, D-2)
UNITS_SUBGROUP = os.path.join(RB2_PROC, "units_subgroup.csv")       # version 3's file with the same partition
DEFAULT_UNITS56 = os.path.join(RB2_PROC, "units_primary.csv")       # 56 family-label units (robustness row)
TECHS_CACHE = os.path.join(RB2_PROC, "techs_cache.pkl")             # ra2 technology registry (DeepSeek law, PI-4)
RA1_BUDGETS = os.path.join(TABLES, "ra1_modelfree_isoflop_budgets.csv")   # bracketed per-budget IsoFLOP minima
RB4_BUDGETS = os.path.join(TABLES, "rb4_chinflop_budgets.csv")       # Chinchilla minima in N_F (variant NF_T4)
RAW_ISO = os.path.join(ROOT, "data", "raw", "isoflop_experiments", "isoflop_experiments.csv")   # Marin N_cfg
RB1_PI_SIGMA = os.path.join(TABLES, "rb1_sigmaC_pi_sigmaC.csv")      # identified set of sigma*(C) (rb4 rebuild)
RB1_TOP = os.path.join(TABLES, "rb1_sigmaC_top_budget_sigma.csv")    # top-budget sigma* (0.594 [0.565, 0.622])

# ---------------------------------------------------------------------------------------------- settings
B_BOOT = 9999            # wild bootstrap draws per study
LEVEL = 0.95             # simultaneous coverage per study
N_GRID = 401             # grid points in c for the sup-t critical value (model computes are added exactly)
TAUS = [1.0, 1.3, 1.84, 3.4, 4.4, 8.0, 10.0]      # tilt allowances of S1 (fix list S1 outputs)
TAU_WHY = {1.0: "none", 1.3: "tokenizer differences", 1.84: "DataDecide tilt of 0.26 under the reference exponents",
           3.4: "DataDecide's own M* factor", 4.4: "both (3.4 x 1.3)", 8.0: "breakdown range", 10.0: "breakdown range"}
TAU_MARKS = [1.3, 1.84, 3.4]                       # marked on the breakdown frontier (S3)
N_SMALL = 15e9                                     # "models below 15B" (total parameters)
K_REF = 1.0 / 0.70055 - 1.0                        # reference curvature (kappa-free Chinchilla, total parameters)
# Webb six-point weights (as in rb2_decisions/rb2common.py)
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])

# Llama 3: FLOP-implied N_F = C/(6D) from the digitization; Meta's FLOP accounting is not stated. Bound on the
# conversion to total parameters at the anchor (N*_F about 3.1e9 at 1e21 FLOP), for a Llama-3-shaped model of that
# size (d = 3072, 28 layers, vocabulary 128,256; Llama 3.2 3B's shape) under three accountings of the budget:
# 6 N_total D (ratio 1); non-embedding parameters only (Kaplan); non-embedding + output head + attention FLOPs at
# Llama 3's pretraining context of 8,192 tokens (Kaplan's 6 n_layer n_ctx d per token, no causal halving); input
# embeddings tied or untied. Computed in data.llama3_conversion().
LLAMA3_SHAPE = dict(d=3072, n_layer=28, vocab=128256, n_ctx=8192, N_nonemb=None)
# Released models' FLOP-effective count for the all-N_F variant (S2(f)): Hoffmann et al.'s accounting (embedding and
# head as matrix products, attention logits and reduction without causal halving) at a common context of 4,096 tokens.
N_CTX_MODELS = 4096


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(PROC, "run_log.txt"), "a") as f:
        f.write(line + "\n")


def tab(df: pd.DataFrame, name: str, index=False):
    df.to_csv(os.path.join(TABLES, f"{PREFIX}_{name}.csv"), index=index)
    return df


def share(w):
    w = np.asarray(w, float)
    return (w - 1.0) / w


def rel(path):
    return os.path.relpath(path, ROOT)


def assert_no_em_dash(text: str, where: str):
    bad = [ch for ch in (chr(0x2014),) if ch in text]
    if bad or "---" in text:
        raise ValueError(f"em dash in generated text: {where}")
