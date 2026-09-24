"""Shared helpers for the ra5_theory verification suite (Theory v2 for the revision).

Reuses the m7_theory helpers (technology, path objects, the observationally equivalent kappa-family) by importing
code/analysis/m7_theory/common.py under the module name ``m7common`` (m7's file is not edited).

Notation follows paper/notes/model_spec.md and paper/sections/appendix_proofs.tex:
    n = ln N, d = ln D, c = ln C = ln 6 + n + d, x = ln M = d - n,
    y = -ln(L - E) (log output index), eps_N = -d ln(L-E)/dn, eps_D = -d ln(L-E)/dd, w = eps_N/eps_D,
    s = (w - 1)/w (expenditure share), a = beta/(alpha+beta), gamma = alpha beta/(alpha+beta), sigma* = 2/(2+alpha+beta).
Every check registers a ``Check``; run.py prints PASS/FAIL per claim, writes the register, and exits 1 on any FAIL.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

_spec = importlib.util.spec_from_file_location("m7common", os.path.join(ANALYSIS, "m7_theory", "common.py"))
m7common = importlib.util.module_from_spec(_spec)
sys.modules["m7common"] = m7common
_spec.loader.exec_module(m7common)

lnR = m7common.lnR
elasticities = m7common.elasticities
path_objects = m7common.path_objects
kappa_family_member = m7common.kappa_family_member
ce_ratio_from_w = m7common.ce_ratio_from_w
PARAM_SETS = m7common.PARAM_SETS
BES = PARAM_SETS["Besiroglu"]
HOF = PARAM_SETS["Hoffmann (rounded)"]
MUE = PARAM_SETS["Muennighoff (alpha=beta)"]

TABLES = os.path.join(ROOT, "output", "tables")
FIGS = os.path.join(ROOT, "output", "figures")
PROC = os.path.join(ROOT, "data", "processed", "ra5_theory")
for _d in (TABLES, FIGS, PROC):
    os.makedirs(_d, exist_ok=True)

LN6 = np.log(6.0)


@dataclass
class Check:
    cid: str          # claim id, e.g. "GW.a.num"
    result: str       # appendix result it supports, e.g. "Prop. A8(iii)(a)"
    claim: str        # statement tested
    verdict: str      # NEW | VERIFIED | CORRECTED | FALSE(counterexample)
    method: str       # sympy | numeric | MC | sympy+numeric
    passed: bool
    metric: str = ""
    note: str = ""


RESULTS: list[Check] = []


def record(cid, result, claim, verdict, method, passed, metric="", note=""):
    c = Check(cid, result, claim, verdict, method, bool(passed), str(metric), note)
    RESULTS.append(c)
    flag = "PASS" if c.passed else "FAIL"
    print(f"[{flag}] {cid:<14s} {verdict:<9s} {claim[:92]}" + (f"   ({metric})" if metric else ""), flush=True)
    return c


def as_records():
    return [asdict(c) for c in RESULTS]


# ----------------------------------------------------------------------------- technologies used in the checks
def chin_y(n, d, p=BES):
    """Log output index y = -ln(L-E) for the Chinchilla technology (kappa = 1)."""
    return -lnR(n, d, p["A"], p["B"], p["alpha"], p["beta"])


def chin_w(n, d, p=BES):
    eN, eD = elasticities(n, d, p["A"], p["B"], p["alpha"], p["beta"])
    return eN / eD


def nonsep_L(n, d, E=1.8):
    """A smooth technology outside the generalized family (non-separable, not quasi-homothetic): three terms with an
    N-D interaction, loosely Farseer-like. Used to test the model-free results. Returns loss L (nats)."""
    return E + 480.0 * np.exp(-0.34 * n) + 2100.0 * np.exp(-0.37 * d) + 900.0 * np.exp(-0.2 * n - 0.22 * d)
