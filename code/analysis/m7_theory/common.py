"""Shared helpers for the m7_theory verification suite.

Notation follows paper/notes/model_spec.md:
    n = ln N, d = ln D, c = ln C = ln 6 + n + d,
    u = A e^{-alpha n}, v = B e^{-beta d}, R = u + v (reducible loss, omega-free),
    y = -ln(L - E), a = beta/(alpha+beta), b = alpha/(alpha+beta), gamma = alpha beta/(alpha+beta),
    G = (alpha A/(beta B))^{1/(alpha+beta)}, K = ((alpha+beta)/beta) A G^{-alpha}, sigma* = 2/(2+alpha+beta),
    w = eps_N/eps_D, M = D/N.

Every check registers a `Check` in RESULTS; run.py prints PASS/FAIL per claim and writes the claim register.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

import sl  # noqa: E402  (shared library; never edited here)

TABLES = os.path.join(ROOT, "output", "tables")
FIGS = os.path.join(ROOT, "output", "figures")
PROC = os.path.join(ROOT, "data", "processed", "m7_theory")
for _d in (TABLES, FIGS, PROC):
    os.makedirs(_d, exist_ok=True)

LN6 = np.log(6.0)


@dataclass
class Check:
    cid: str            # claim id, e.g. "L1.a"
    source: str         # model_spec / SYNTHESIS / task / new
    claim: str          # short statement that was tested (the corrected one if verdict != VERIFIED)
    verdict: str        # VERIFIED | IMPRECISE | FALSE | NEW
    method: str         # sympy | numeric | MC | sympy+numeric
    passed: bool        # did the test of the stated (or corrected) claim pass?
    metric: str = ""    # e.g. max abs error, MC deviation, counterexample value
    note: str = ""      # correction / caveat


RESULTS: list[Check] = []


def record(cid, source, claim, verdict, method, passed, metric="", note=""):
    c = Check(cid, source, claim, verdict, method, bool(passed), str(metric), note)
    RESULTS.append(c)
    flag = "PASS" if c.passed else "FAIL"
    print(f"[{flag}] {cid:<10s} {verdict:<9s} {claim[:95]}" + (f"   ({metric})" if metric else ""))
    return c


def as_records():
    return [asdict(c) for c in RESULTS]


# ----------------------------------------------------------------------------- technology helpers
# Generalized ("kappa-free") family:  R = (A e^{-a1 n} + B e^{-b1 d})^kappa ; Chinchilla is kappa = 1.

def lnR(n, d, A, B, a1, b1, kappa=1.0):
    return kappa * np.logaddexp(np.log(A) - a1 * np.asarray(n, float), np.log(B) - b1 * np.asarray(d, float))


def elasticities(n, d, A, B, a1, b1, kappa=1.0):
    u = A * np.exp(-a1 * np.asarray(n, float))
    v = B * np.exp(-b1 * np.asarray(d, float))
    Q = u + v
    return kappa * a1 * u / Q, kappa * b1 * v / Q


def path_objects(A, B, al, be):
    """Expansion path and frontier for kappa = 1 (Chinchilla)."""
    S = al + be
    a, b = be / S, al / S
    gam = al * be / S
    G = (al * A / (be * B)) ** (1.0 / S)
    K = S / be * A * G ** (-al)
    return dict(a=a, b=b, gamma=gam, G=G, K=K, sigma_star=2.0 / (2.0 + S))


def M_star(C, A, B, al, be):
    p = path_objects(A, B, al, be)
    return p["G"] ** -2 * (np.asarray(C, float) / 6.0) ** (p["b"] - p["a"])


def kappa_family_member(S, a, gamma, G, K):
    """Member of the observationally equivalent on-path family (Prop. DMR / P2):
    inner exponents a1 = (1-a) S, b1 = a S, outer exponent kappa = gamma/(a(1-a)S), and (A, B) chosen so
    that the compute-optimal path n* = ln G + a (c - ln 6) and the frontier R*(C) = K (C/6)^{-gamma} are
    exactly the given ones.  Valid for S > 0 (sigma* = 2/(2+S) in (0,1))."""
    b = 1.0 - a
    a1, b1 = b * S, a * S
    kap = gamma / (a * b * S)
    # inner frontier: Q*(C) = K1 (C/6)^{-gamma1}, gamma1 = a1 b1 / S = a b S; R* = Q*^kappa => K1 = K^{1/kappa}
    K1 = K ** (1.0 / kap)
    A = (b1 / S) * K1 * G ** a1               # from K1 = (S/b1) A G^{-a1}
    B = a1 * A / (b1 * G ** S)                # from G^S = a1 A/(b1 B)
    return dict(A=A, B=B, a1=a1, b1=b1, kappa=kap, S=S, sigma_star=2.0 / (2.0 + S))


def ce_ratio_from_w(w, al, be):
    """C / C_min as a function of the wedge only (new closed form, Corollary W2)."""
    gam = al * be / (al + be)
    return ((al + be * w) / (al + be)) ** (1.0 / gam) * w ** (-1.0 / al)


def ce_ratio_direct(N, D, A, B, al, be):
    """C / C_min computed by brute force: C = 6ND, C_min = minimal compute on the kappa=1 frontier for R(N,D)."""
    p = path_objects(A, B, al, be)
    R = A * N ** -al + B * D ** -be
    Cmin = 6.0 * (p["K"] / R) ** (1.0 / p["gamma"])
    return 6.0 * N * D / Cmin


PARAM_SETS = {
    "Hoffmann (rounded)": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
    "Hoffmann (TeX)": dict(E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849),
    "Besiroglu": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),
    "Muennighoff (alpha=beta)": dict(E=1.87, A=521.0, B=1488.0, alpha=0.3527, beta=0.3527),
}
