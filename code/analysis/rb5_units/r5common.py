"""r5common.py -- paths and helpers for module rb5_units (round 3, WP4b analysis: budget-level decision units as the
primary unit, readings coded under paper/notes/rb5_reading_protocol.md, identification classes under the virtual value of
compactness, the trend with developer-cluster inference, randomization inference for the serving test).

Fix list: paper/notes/round3_fixlist.md, items W1, W2, W14, W18, W19 (analysis), W24 (data corrections); decisions D-1, D-2.
CPU only; at most 2 processes (run under nice -n 10). Imports rb2_decisions (and through it ra2_wedge) read-only.
"""
from __future__ import annotations

import datetime
import os
import re
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
RB2 = os.path.join(ANALYSIS, "rb2_decisions")
RAW = os.path.join(ROOT, "data", "raw", "rb2_decisions")
PROC = os.path.join(ROOT, "data", "processed", "rb5_units")
PROC_RB2 = os.path.join(ROOT, "data", "processed", "rb2_decisions")
TABLES = os.path.join(ROOT, "output", "tables")
FIGS = os.path.join(ROOT, "output", "figures")
PREFIX = "rb5_units"
PROTOCOL_VERSION = "rb5_reading_protocol v1 (2026-09-25)"
SEED = 20260925
os.makedirs(PROC, exist_ok=True)

TOL = 1.10   # max/min D within a common-budget unit (decision D-2)


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(PROC, "run_log.txt"), "a") as f:
        f.write(line + "\n")


def tab(df, name, index=False):
    df.to_csv(os.path.join(TABLES, f"{PREFIX}_{name}.csv"), index=index)
    return df


def share(w):
    w = np.asarray(w, float)
    return (w - 1.0) / w


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


_TXT = {}


def source_text(key):
    if key not in _TXT:
        p = os.path.join(RAW, f"{key}.txt")
        _TXT[key] = norm(open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
    return _TXT[key]


def verify(key, quote):
    t = source_text(key)
    return t is not None and norm(quote) in t


def manifest():
    import json
    return json.load(open(os.path.join(RAW, "manifest.json")))


def budget_groups(df, tol=TOL):
    """Within each family (gen), cluster members by D (ascending; a new group starts when D exceeds tol x the group's
    smallest D). Returns Series uid -> group label 'gen#k' (identical to rb2_decisions.decisions.subgroup_labels)."""
    out = {}
    for gen, g in df.groupby("gen"):
        g = g.sort_values("D")
        k, dmin = 0, None
        for _, r in g.iterrows():
            if dmin is None or r["D"] > tol * dmin:
                k += 1
                dmin = r["D"]
            out[r["uid"]] = f"{gen}#{k}"
    return pd.Series(out)
