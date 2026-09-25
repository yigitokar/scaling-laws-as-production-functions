"""rb2common.py -- paths, constants and helpers for module rb2_decisions (the revealed value of compactness at the level
of allocation decisions; round-2 referee requests R1 New 1, 3, 4, 5; R2 Majors 2, 4, 6, 7; R3 N2-N4; R4 R2-M1..M3).

Builds on module ra2_wedge (imported read-only from code/analysis/ra2_wedge; never edited) and on the shared library
code/analysis/sl.py (never edited).

Notation (paper/notes/model_spec.md): w = eps_N/eps_D, M = D/N, C = 6ND, S = alpha + beta, a = beta/S,
1/sigma* - 1 = S/2, ln w = (S/2)[ln M - ln M*(C)], ln M*(C) = -2 ln G + (1 - 2a) ln(C/6); expenditure share
s = (w - 1)/w = m_N/(1 + m_N), where m_N is the value of compactness per unit of training cost (Prop. A8).
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
RA2 = os.path.join(ANALYSIS, "ra2_wedge")
M2C = os.path.join(ANALYSIS, "m2_techpanel")
for _p in (ANALYSIS, RA2):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import ra2common as R2C  # noqa: E402  (ra2_wedge helpers; read-only)

# the CPU budget of this session: at most 2 worker processes (round 3: GPU queues A-E are running)
N_PROC = 2
R2C.N_PROC = N_PROC

PREFIX = "rb2_decisions"
SEED = 20260925
RAW = os.path.join(ROOT, "data", "raw", "rb2_decisions")
PROC = os.path.join(ROOT, "data", "processed", "rb2_decisions")
TABLES = os.path.join(ROOT, "output", "tables")
FIGS = os.path.join(ROOT, "output", "figures")
MEMOS = os.path.join(ROOT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

REF = "chin_q"          # reference technology (Chinchilla, kappa free)
SIGMA_RA1 = 0.6947      # ra1 model-free sigma*, random-effects over six designs (Table 1); fallback common curvature
SE_SIGMA_RA1 = 0.0086
# quantized memory-tier windows (ra2 Table F8; total parameters)
TIER_LO = np.array([2.4, 6.5, 11.5, 26.0, 65.0]) * 1e9
TIER_HI = np.array([3.3, 9.5, 14.9, 32.9, 72.9]) * 1e9
# literal windows of the task statement (<=3B; 7-9B; 12-14B; 27-32B; ~70B), sensitivity
TIER_LO_ALT = np.array([0.0, 7.0, 12.0, 27.0, 65.0]) * 1e9
TIER_HI_ALT = np.array([3.3, 9.5, 14.9, 32.9, 72.9]) * 1e9

log_w = R2C.log_w
ln_mstar = R2C.ln_mstar
ab_from_S_a = R2C.ab_from_S_a
share = R2C.share
wild_cluster_boot = R2C.wild_cluster_boot


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(PROC, "run_log.txt"), "a") as f:
        f.write(line + "\n")


def tab(df: pd.DataFrame, name: str, index=False):
    df.to_csv(os.path.join(TABLES, f"{PREFIX}_{name}.csv"), index=index)
    return df


def pct(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else np.nan


def in_tier(N, lo=TIER_LO, hi=TIER_HI):
    N = np.asarray(N, float)
    m = np.zeros(len(N), bool)
    for a, b in zip(lo, hi):
        m |= (N > a) & (N <= b)
    return m


def common_sigma():
    """Common model-free curvature for the lab-own rule (R2 Major 6): rb1_sigmaC's study-level headline if that module
    has written it (output/tables/rb1_sigmaC_*.csv, row 'PRIMARY: one estimate per study'), else ra1's 0.695."""
    import glob
    for f in sorted(glob.glob(os.path.join(TABLES, "rb1_sigmaC_*.csv"))):
        try:
            t = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if "variant" in t.columns and "mean_re" in t.columns:
            r = t[t["variant"].astype(str).str.startswith("PRIMARY")]
            if len(r):
                r = r.iloc[0]
                se = float(r["se_hksj"]) if "se_hksj" in r and np.isfinite(r["se_hksj"]) else SE_SIGMA_RA1
                df = int(r["k"]) - 1 if "k" in r and np.isfinite(r["k"]) else None   # HKSJ: t with k - 1 df
                return dict(sigma=float(r["mean_re"]), se=se, df=df, source=os.path.basename(f) + " (study-level, HKSJ)")
    return dict(sigma=SIGMA_RA1, se=SE_SIGMA_RA1,
                source="ra1_modelfree_heterogeneity.csv: model-free sigma*, six designs, random effects (rb1 not found)")


WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
