"""Verification of Corollary A-general (fix list P4; Referee 3, round 3, comment D(a)).

Claim. Let y = F(n, d) be twice continuously differentiable with F_n, F_d > 0 (n = ln N, d = ln D), and let the cost be
any known twice-differentiable C(N, D), with log-cost elasticities g_n = dlnC/dn > 0 and g_d = dlnC/dd > 0 at the
cost-minimizing (output-maximizing) point on an isocost, r = g_n / g_d, and sigma_C the Hicks elasticity of substitution
of the isocost itself (the elasticity of substitution of C read as a production function). Then at that point

    1/sigma* - 1/sigma_C = |y_nn|_C| / [ r (1 + r) g_d dy*/dc ],

where y_nn|_C is the second derivative of output along the isocost with respect to n and dy*/dc the slope of the
output-cost frontier in c = ln C. Special cases: C = 6ND (sigma_C = 1, r = g_d = 1) gives equation (eq:ident-mf);
a linear cost (sigma_C = infinity, g = cost shares s) gives 1/sigma* = |y_nn|_C| s_D / (s_N dy*/dc). The tangency is a
strict local maximum of output on the isocost iff 1/sigma* > 1/sigma_C (for positive elasticities, sigma* < sigma_C),
which generalizes Lemma 1. The ratio is invariant to strictly monotone transformations of output with nonzero
derivative, because the first derivative of output along the isocost vanishes at the optimum.

Checks.
  1. sympy: the identity for generic derivatives of F and of ln C at a tangency (Lemma A2's formula for sigma).
  2. numeric, Referee 3's seven cases: CES technology y = ln[(a N^rho + b D^rho)^(1/rho)] (sigma = 1/(1-rho)) under the
     multiplicative cost C = ND (sigma = 0.67, 0.40, 0.83) and a linear cost q_N N + q_D D (sigma = 0.67, 0.40, 1.43,
     0.83); optimum located numerically on the isocost, y_nn|_C and dy*/dc by finite differences, sigma analytic.
  3. numeric, further cases: the paper's reference technology (Chinchilla, kappa free) under 6ND, under a linear cost,
     under a size-dependent FLOP price 6 N^(1+delta) D and under a CES cost (theta = 2, sigma_C = -1); a
     non-separable technology under 6ND + data cost.
  4. invariance: the same ratio computed from loss L = E + exp(-y), from ln L and from bits.
  5. second-order condition: the sign of y_nn|_C agrees with 1/sigma* > 1/sigma_C in every case, including CES with
     sigma = 1.43 under the multiplicative cost (a minimum of output on the isocost).
Deterministic; CPU only; a few seconds. Usage: .venv/bin/python code/analysis/rb5_theory/verify_general_isocost.py
Writes code/analysis/rb5_theory/results/general_isocost.csv; exit code 1 on any FAIL.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = []


def record(cid, claim, ok, metric):
    RESULTS.append(dict(check=cid, claim=claim, passed=bool(ok), metric=metric))
    print(("[PASS] " if ok else "[FAIL] ") + f"{cid:<26} {claim}  ({metric})")


# --------------------------------------------------------------------------------------------- 1. symbolic
def symbolic():
    Fd, Fnn, Fnd, Fdd, gd, kn, knd, kd2, r = sp.symbols("F_d F_nn F_nd F_dd g_d k_nn k_nd k_dd r", real=True)
    Fn = r * Fd                         # tangency: F_n / F_d = g_n / g_d = r
    gn = r * gd
    # Lemma A2: sigma = P/(P+Q) in log coordinates, for the technology and for ln C
    P = Fn * Fd * (Fn + Fd)
    Q = -(Fnn * Fd ** 2 - 2 * Fnd * Fn * Fd + Fdd * Fn ** 2)
    PC = gn * gd * (gn + gd)
    QC = -(kn * gd ** 2 - 2 * knd * gn * gd + kd2 * gn ** 2)
    inv_sig = (P + Q) / P
    inv_sigC = (PC + QC) / PC
    # isocost d = phi(n): phi' = -r, phi'' from the implicit function theorem
    phi1 = -r
    phi2 = -(kn + 2 * knd * phi1 + kd2 * phi1 ** 2) / gd
    ynn = Fnn + 2 * Fnd * phi1 + Fdd * phi1 ** 2 + Fd * phi2
    lam = Fd / gd                        # dy*/dc by the envelope theorem (Lagrange multiplier)
    lhs = inv_sig - inv_sigC
    rhs = -ynn / (r * (1 + r) * gd * lam)
    diff = sp.simplify(lhs - rhs)
    record("sym.identity", "1/sigma - 1/sigma_C = -y_nn|_C / [r(1+r) g_d dy*/dc] for generic F and C at a tangency",
           diff == 0, f"simplified difference = {diff}")
    # multiplicative cost: k_nn = k_nd = k_dd = 0, r = g_d = 1 gives 1/sigma - 1 = -y_nn / (2 lam)
    d2 = sp.simplify((lhs - rhs).subs({kn: 0, knd: 0, kd2: 0, r: 1, gd: 1}))
    special = sp.simplify(inv_sigC.subs({kn: 0, knd: 0, kd2: 0}))
    record("sym.multiplicative", "C = 6ND: sigma_C = 1 and the identity is eq. (ident-mf)", d2 == 0 and special == 1,
           f"1/sigma_C = {special}")
    # linear cost: ln C = ln(qN e^n + qD e^d); g = shares; k_nn = k_dd = sN sD, k_nd = -sN sD  => 1/sigma_C = 0
    sN = sp.symbols("s_N", positive=True)
    sD = 1 - sN
    lin = {gd: sD, r: sN / sD, kn: sN * sD, knd: -sN * sD, kd2: sN * sD}
    inv_sigC_lin = sp.simplify(inv_sigC.subs(lin))
    rhs_lin = sp.simplify(rhs.subs(lin))
    target = sp.simplify(-ynn.subs(lin) * sD / (sN * lam.subs(lin)))
    record("sym.linear", "linear cost: 1/sigma_C = 0 and 1/sigma* = |y_nn| s_D / (s_N dy*/dc)",
           inv_sigC_lin == 0 and sp.simplify(rhs_lin - target) == 0, f"1/sigma_C = {inv_sigC_lin}")


# --------------------------------------------------------------------------------------------- 2-5. numeric
class Cost:
    def __init__(self, name, lnC, grad, inv_sigC):
        self.name, self.lnC, self.grad, self.inv_sigC = name, lnC, grad, inv_sigC


def mult_cost(k=1.0, delta=0.0):
    # ln C = ln k + (1+delta) n + d
    return Cost(f"multiplicative{'' if delta == 0 else f' N^(1+{delta})D'}",
                lambda n, d: np.log(k) + (1 + delta) * n + d, lambda n, d: (1 + delta, 1.0), 1.0)


def lin_cost(qN, qD):
    def lnC(n, d):
        return np.log(qN * np.exp(n) + qD * np.exp(d))

    def grad(n, d):
        a, b = qN * np.exp(n), qD * np.exp(d)
        return a / (a + b), b / (a + b)
    return Cost("linear", lnC, grad, 0.0)


def ces_cost(qN, qD, theta):
    def lnC(n, d):
        return np.log(qN * np.exp(theta * n) + qD * np.exp(theta * d)) / theta

    def grad(n, d):
        a, b = qN * np.exp(theta * n), qD * np.exp(theta * d)
        return a / (a + b), b / (a + b)
    return Cost(f"CES cost theta={theta}", lnC, grad, 1 - theta)   # sigma_C = 1/(1-theta)


def data_cost(k, cD):
    # C = 6ND + cD D: ln C = ln(k e^(n+d) + cD e^d)
    def lnC(n, d):
        return np.log(k * np.exp(n + d) + cD * np.exp(d))

    def grad(n, d):
        a, b = k * np.exp(n + d), cD * np.exp(d)
        return a / (a + b), 1.0
    # sigma_C: Hicks elasticity of C(N,D) = D (kN + cD); treat as production function f = D(kN+cD): computed numerically
    return Cost("6ND + data cost", lnC, grad, None)


def hicks_inv_sigma_logs(f, n, d, h=1e-4):
    """1/sigma from Lemma A2's log-coordinate formula applied to f (finite differences)."""
    fn = (f(n + h, d) - f(n - h, d)) / (2 * h)
    fd_ = (f(n, d + h) - f(n, d - h)) / (2 * h)
    fnn = (f(n + h, d) - 2 * f(n, d) + f(n - h, d)) / h ** 2
    fdd = (f(n, d + h) - 2 * f(n, d) + f(n, d - h)) / h ** 2
    fnd = (f(n + h, d + h) - f(n + h, d - h) - f(n - h, d + h) + f(n - h, d - h)) / (4 * h ** 2)
    P = fn * fd_ * (fn + fd_)
    Q = -(fnn * fd_ ** 2 - 2 * fnd * fn * fd_ + fdd * fn ** 2)
    return (P + Q) / P


class Technology:
    def __init__(self, name, y, inv_sigma):
        self.name, self.y, self.inv_sigma = name, y, inv_sigma   # inv_sigma(n, d): analytic where available


def ces_tech(rho, a=1.0, b=1.3):
    y = lambda n, d: np.log(a * np.exp(rho * n) + b * np.exp(rho * d)) / rho
    return Technology(f"CES sigma={1 / (1 - rho):.2f}", y, lambda n, d: 1 - rho)


def ref_tech():
    p = json.load(open(os.path.join(ROOT, "data", "processed", "ra2_wedge", "headline.json")))["chin_q_point"]
    A, B, al, be, kap = np.exp(p["lnA"]), np.exp(p["lnB"]), p["alpha"], p["beta"], p["q"]
    y = lambda n, d: -kap * np.log(A * np.exp(-al * n) + B * np.exp(-be * d))

    def inv_sigma(n, d):      # Lemma A1(ii): 1/sigma = 1 + (alpha + beta w)/(1 + w)
        w = al * A * np.exp(-al * n) / (be * B * np.exp(-be * d))
        return 1 + (al + be * w) / (1 + w)
    return Technology("reference (kappa free)", y, inv_sigma)


def nonsep_tech():
    y = lambda n, d: 0.7 * n + 0.4 * d - 0.05 * n ** 2 + 0.02 * n * d - 0.03 * d ** 2 + 0.1 * np.exp(-0.2 * n + 0.1 * d)
    return Technology("non-separable", y, None)


def iso_d(cost, n, c):
    """d on the isocost lnC(n, d) = c."""
    return brentq(lambda d: cost.lnC(n, d) - c, -60, 80, xtol=1e-14)


def n_upper(cost, c):
    """largest n for which the isocost lnC = c has a solution in d (d -> -60)."""
    f = lambda n: cost.lnC(n, -60.0) - c
    return brentq(f, -60, 80, xtol=1e-12) - 1e-6 if f(-60) < 0 < f(80) else 80.0


def analyse(tech, cost, c, bounds, sign=+1):
    bounds = (bounds[0], min(bounds[1], n_upper(cost, c) - 0.05))
    Y = lambda n: tech.y(n, iso_d(cost, n, c))
    # the tangency: maximize (sign=+1) or minimize (sign=-1) output along the isocost
    r_ = minimize_scalar(lambda n: -sign * Y(n), bounds=bounds, method="bounded", options={"xatol": 1e-12})
    n0 = r_.x
    d0 = iso_d(cost, n0, c)
    h = 2e-3
    ynn = (Y(n0 + h) - 2 * Y(n0) + Y(n0 - h)) / h ** 2
    # frontier slope: value of the (local) optimum as c moves
    def ystar(cc):
        rr = minimize_scalar(lambda n: -sign * tech.y(n, iso_d(cost, n, cc)), bounds=(n0 - 1.0, n0 + 1.0), method="bounded",
                             options={"xatol": 1e-12})
        return tech.y(rr.x, iso_d(cost, rr.x, cc))
    hc = 1e-3
    dy = (ystar(c + hc) - ystar(c - hc)) / (2 * hc)
    gn, gd = cost.grad(n0, d0)
    r = gn / gd
    inv_sig = tech.inv_sigma(n0, d0) if tech.inv_sigma is not None else hicks_inv_sigma_logs(tech.y, n0, d0)
    if cost.inv_sigC is None:
        inv_sigC = hicks_inv_sigma_logs(lambda n, d: cost.lnC(n, d), n0, d0)
    else:
        inv_sigC = cost.inv_sigC
    lhs = inv_sig - inv_sigC
    rhs = -ynn / (r * (1 + r) * gd * dy)
    # invariance: loss L = E + exp(-y) (E = 1.7), ln L, bits = L/ln 2; curvature along isocost over frontier slope
    out = dict(technology=tech.name, cost=cost.name, n=n0, d=d0, r=r, g_d=gd, inv_sigma=inv_sig, inv_sigma_C=inv_sigC,
               y_nn=ynn, dystar_dc=dy, lhs=lhs, rhs=rhs, rel_err=abs(rhs / lhs - 1) if lhs != 0 else abs(rhs))
    ratios = []
    for g in (lambda yy: 1.7 + np.exp(-yy), lambda yy: np.log(1.7 + np.exp(-yy)), lambda yy: (1.7 + np.exp(-yy)) / np.log(2)):
        G = lambda n: g(Y(n))
        Gnn = (G(n0 + h) - 2 * G(n0) + G(n0 - h)) / h ** 2
        Gs = lambda cc: g(ystar(cc))
        dG = (Gs(c + hc) - Gs(c - hc)) / (2 * hc)
        ratios.append(-Gnn / dG)       # loss falls with quality: |L_nn| / |dL*/dc| with signs as in eq. (ident-mf)
    out["ratio_y"] = -ynn / dy
    out["ratio_inv_max_rel_dev"] = max(abs(x / out["ratio_y"] - 1) for x in ratios)
    return out


def numeric():
    rows = []
    # Referee 3's seven cases, c chosen so that the optimum is interior
    cases = []
    for rho in (-0.5, -1.5, -0.2):       # sigma = 0.67, 0.40, 0.83 under C = ND
        cases.append((ces_tech(rho), mult_cost(1.0), 3.0, (-10, 13), "R3"))
    for rho in (-0.5, -1.5, 0.3, -0.2):  # sigma = 0.67, 0.40, 1.43, 0.83 under a linear cost
        cases.append((ces_tech(rho), lin_cost(1.0, 2.0), 3.0, (-20, 5), "R3"))
    ref = ref_tech()
    cases += [
        (ref, mult_cost(6.0), np.log(1e21), (15, 30), "extra"),
        (ref, lin_cost(6.0 * 1e11, 6.0 * 2e9), np.log(1e21), (15, 30), "extra"),
        (ref, mult_cost(6.0, 0.1), np.log(1e21), (12, 30), "extra"),
        (ces_tech(0.3), ces_cost(1.0, 2.0, 2.0), 3.0, (-5, 10), "extra"),
        (nonsep_tech(), data_cost(6.0, 3.0), 3.0, (-8, 8), "extra"),
    ]
    for tech, cost, c, bounds, tag in cases:
        o = analyse(tech, cost, c, bounds)
        o["source"] = tag
        rows.append(o)
        ok = o["rel_err"] < 2e-5 and o["ratio_inv_max_rel_dev"] < 2e-5
        soc = (o["y_nn"] < 0) == (o["lhs"] > 0)
        record(f"num.{tag}.{tech.name[:18]}|{cost.name[:14]}",
               "1/sigma* - 1/sigma_C = |y_nn| / [r(1+r) g_d dy*/dc]; invariant to output transformation; SOC iff 1/sigma* > 1/sigma_C",
               ok and soc, f"lhs={o['lhs']:.7f} rhs={o['rhs']:.7f} rel.err={o['rel_err']:.1e} "
               f"inv.dev={o['ratio_inv_max_rel_dev']:.1e} y_nn={o['y_nn']:.4f}")
    # sigma > sigma_C: CES sigma = 1.43 under the multiplicative cost. The symmetric-looking tangency minimizes output.
    o = analyse(ces_tech(0.3), mult_cost(1.0), 3.0, (-4, 7), sign=-1)
    o["source"] = "SOC"
    rows.append(o)
    ok = abs(o["rhs"] / o["lhs"] - 1) < 2e-5 and o["y_nn"] > 0 and o["lhs"] < 0
    record("num.SOC.CES1.43|multiplicative", "sigma* > sigma_C: the tangency is a minimum of output on the isocost "
           "(y_nn > 0), and the identity still holds", ok, f"lhs={o['lhs']:.7f} rhs={o['rhs']:.7f} y_nn={o['y_nn']:.4f}")
    return rows


if __name__ == "__main__":
    symbolic()
    rows = numeric()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "general_isocost.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(RESULTS).to_csv(out.replace(".csv", "_checks.csv"), index=False)
    nfail = sum(not r["passed"] for r in RESULTS)
    print(f"{len(RESULTS) - nfail}/{len(RESULTS)} checks passed; wrote {out}")
    sys.exit(1 if nfail else 0)
