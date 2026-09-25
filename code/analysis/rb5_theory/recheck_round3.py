"""Independent re-check of the round-3 theory changes (package WP3-proofcheck, 2026-09-25).

Written without reusing the solvers of verify_virtual_value.py, verify_general_isocost.py or verify_prop_pi.py: every
optimum is found here by a box-constrained quasi-Newton search followed by a Newton polish on the free coordinates, every
shadow value is an ENVELOPE derivative (the change of the optimal value when a bound moves, from re-solved problems), and
every elasticity of substitution is computed from the Hicks formula in LEVELS (not from Lemma A2's log form).
Technologies come from code/analysis/sl.py (Hoffmann et al., Besiroglu et al.) and the reference technology
(data/processed/ra2_wedge/headline.json, chin_q_point).

Parts
  A  Prop. 1 / Prop. A-wedge: virtual value v_N = m_N + nu_N (size cap, floor, latency, menu, token cap, both caps),
     the Neary-Roberts reading, and the bounds on m_N (incl. the both-caps case).
  B  Prop. 1(iv) / Prop. A-family (iii), (iv), (vi): family identities with capped / floored members, D chosen or capped.
  C  Corollary A-general: 1/sigma* - 1/sigma_C = |y_nn|_C| / [r(1+r) g_d dy*/dc] for six technology-cost pairs, in
     levels; the loss-unit version of part (iv).
  D  Prop. 2(ii) / Prop. A1(iii)(b): profiled criterion quartic along opposite-rate directions; quadratic at a fixed split.
  E  Prop. 3(v) / Prop. A6 (vii)-(viii): tilt translation, sharpness of an end point, simultaneous-band coverage of the
     widened union (own Scheffe bands), and a curved path under a linear band.
  F  Corollary A2 numbers; G  Prop. A2(iii) information slopes; H  numbers in the text; I  symbolic identities.

Run: .venv/bin/python code/analysis/rb5_theory/recheck_round3.py   (CPU only, one process, about a minute)
Output: code/analysis/rb5_theory/results/recheck_round3.csv
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
import sl  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "recheck_round3.csv")
ROWS = []


def rec(part, name, ok, detail):
    ROWS.append(dict(part=part, check=name, ok=bool(ok), detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {part:3s} {name}: {detail}")


HOF = sl.HOFFMANN
BES = sl.BESIROGLU
_ref = json.load(open(os.path.join(ROOT, "data/processed/ra2_wedge/headline.json")))["chin_q_point"]
REF = dict(E=_ref["E"], A=math.exp(_ref["lnA"]), B=math.exp(_ref["lnB"]), alpha=_ref["alpha"], beta=_ref["beta"], kappa=_ref["q"])


# ======================================================================================================== A. one developer
class Dev:
    """Developer of Prop. A-wedge with kappa-family technology L = E + (A N^-a + B D^-b)^k, objective
    V = -Psi*L - s*N (serving cost s*N, so m_N = s N / X_T), X_T = q0 N^(1+delta) 6 D, data cost qD*D."""

    def __init__(self, tech, Psi=1.0, s=2e-10, q0=1e-21, delta=0.0, qD=0.0):
        self.t = tech
        self.Psi, self.s, self.q0, self.delta, self.qD = Psi, s, q0, delta, qD

    def parts(self, n, d):
        t = self.t
        u = t["A"] * math.exp(-t["alpha"] * n)
        v = t["B"] * math.exp(-t["beta"] * d)
        R = u + v
        k = t.get("kappa", 1.0)
        L = t["E"] + R ** k
        X = 6 * self.q0 * math.exp((1 + self.delta) * n + d)
        return u, v, R, k, L, X

    def value(self, z):
        n, d = z
        u, v, R, k, L, X = self.parts(n, d)
        return -self.Psi * L - self.s * math.exp(n) - X - self.qD * math.exp(d)

    def grad(self, z):
        n, d = z
        t = self.t
        u, v, R, k, L, X = self.parts(n, d)
        Lam = self.Psi * k * R ** k  # -(dV/dL)(L-E) with L-E = R^k
        epsN, epsD = t["alpha"] * u / R, t["beta"] * v / R
        gn = Lam * epsN - self.s * math.exp(n) - (1 + self.delta) * X
        gd = Lam * epsD - X - self.qD * math.exp(d)
        return np.array([gn, gd])

    def wedge(self, n, d):
        u, v, *_ = self.parts(n, d)
        return self.t["alpha"] * u / (self.t["beta"] * v)

    def solve(self, nb=(10.0, 35.0), db=(10.0, 40.0), z0=(20.0, 24.0)):
        f = lambda z: -self.value(z)
        g = lambda z: -self.grad(z)
        r = minimize(f, np.array(z0), jac=g, method="L-BFGS-B", bounds=[nb, db],
                     options=dict(ftol=1e-15, gtol=1e-13, maxiter=5000))
        z = r.x.copy()
        # Newton polish on the free coordinates (finite-difference Hessian of the analytic gradient)
        free = [i for i, (lo, hi) in enumerate([nb, db]) if lo + 1e-9 < z[i] < hi - 1e-9]
        for i, b in enumerate([nb, db]):
            if i not in free:
                z[i] = b[0] if abs(z[i] - b[0]) < abs(z[i] - b[1]) else b[1]
        for _ in range(30):
            if not free:
                break
            gr = self.grad(z)[free]
            H = np.zeros((len(free), len(free)))
            h = 1e-6
            for a, i in enumerate(free):
                zp, zm = z.copy(), z.copy()
                zp[i] += h
                zm[i] -= h
                H[:, a] = (self.grad(zp)[free] - self.grad(zm)[free]) / (2 * h)
            step = np.linalg.solve(H, -gr)
            z[free] += step
            if np.max(np.abs(step)) < 1e-13:
                break
        return z, self.value(z)


def envelope(dev, key, bounds, h=1e-4):
    """d(optimal value)/d(bound) by central differences of re-solved problems. key in {'nhi','nlo','dhi'}."""
    nb, db = bounds

    def val(delta_):
        nb2, db2 = list(nb), list(db)
        if key == "nhi":
            nb2[1] += delta_
        elif key == "nlo":
            nb2[0] += delta_
        elif key == "dhi":
            db2[1] += delta_
        return dev.solve(tuple(nb2), tuple(db2))[1]

    return (val(h) - val(-h)) / (2 * h)


def part_A():
    base = dict(Psi=1.0, s=2e-10, q0=1e-21)
    for label, tech, extra in [("Hoffmann", dict(E=HOF.E, A=HOF.A, B=HOF.B, alpha=HOF.alpha, beta=HOF.beta), {}),
                               ("reference kappa free", REF, {}),
                               ("Hoffmann, delta=0.1, qD>0", dict(E=HOF.E, A=HOF.A, B=HOF.B, alpha=HOF.alpha, beta=HOF.beta),
                                dict(delta=0.1, qD=4e-12))]:
        dev = Dev(tech, **{**base, **extra})
        z0, _ = dev.solve()
        n0, d0 = z0
        # interior optimum: w = (1+delta+m_N)/(1+m_D), v_N = m_N
        u, v, R, k, L, X = dev.parts(n0, d0)
        mN = dev.s * math.exp(n0) / X
        mD = dev.qD * math.exp(d0) / X
        w = dev.wedge(n0, d0)
        rec("A", f"{label}: interior", abs(w / ((1 + dev.delta + mN) / (1 + mD)) - 1) < 1e-8,
            f"w={w:.6f} (1+delta+m_N)/(1+m_D)={(1 + dev.delta + mN) / (1 + mD):.6f}")
        cases = {
            "size cap": ((10.0, n0 - 0.7), (10.0, 40.0), "nhi"),
            "size floor": ((n0 + 0.6, 35.0), (10.0, 40.0), "nlo"),
            "token cap": ((10.0, 35.0), (10.0, d0 - 0.8), "dhi"),
        }
        for cname, (nb, db, key) in cases.items():
            z, V = dev.solve(nb, db)
            n, d = z
            u, v, R, k, L, X = dev.parts(n, d)
            mN = dev.s * math.exp(n) / X
            mD = dev.qD * math.exp(d) / X
            env = envelope(dev, key, (nb, db))
            if key == "nhi":       # Lagrangian Pi - xi (n - nbar): dPi*/dnbar = xi, g_n = +1
                nuN, mu = env / X, 0.0
            elif key == "nlo":     # Pi - xi (nlo - n): dPi*/dnlo = -xi, g_n = -1, so xi g_n = dPi*/dnlo
                nuN, mu = env / X, 0.0
            else:                  # Pi - xi_D (d - dbar)
                nuN, mu = 0.0, env / X
            vN = mN + nuN
            w = dev.wedge(n, d)
            pred = (1 + dev.delta + vN) / (1 + mD + mu)
            ok = abs(w / pred - 1) < 2e-5
            sign_ok = {"nhi": nuN > 0, "nlo": nuN < 0, "dhi": mu > 0}[key]
            bound = ""
            if dev.delta == 0 and dev.qD == 0:
                if key == "nhi":
                    sign_ok &= (w - 1 >= mN) and abs((w - 1) - vN) < 1e-4 * vN
                    bound = f"; w-1={w - 1:.4f} = v_N >= m_N={mN:.4f}"
                elif key == "nlo":
                    sign_ok &= (w - 1 <= mN)
                    bound = f"; w-1={w - 1:.4f} < m_N={mN:.4f}"
                else:
                    sign_ok &= (w - 1 <= vN)
                    bound = f"; w-1={w - 1:.4f} <= v_N={vN:.4f}"
            rec("A", f"{label}: {cname} (envelope shadow value)", ok and sign_ok,
                f"w={w:.6f} pred={pred:.6f} relerr={abs(w / pred - 1):.1e} nu_N={nuN:.4f} mu={mu:.4f}{bound}")
            if key == "nhi" and dev.delta == 0 and dev.qD == 0:
                # Neary-Roberts: an UNCONSTRAINED developer whose own value of compactness at the chosen point is v_N
                # (serving cost s' with s' N / X_T = v_N) makes the same choice.
                dev2 = Dev(tech, Psi=dev.Psi, s=vN * X / math.exp(n), q0=dev.q0)
                z2, _ = dev2.solve()
                rec("A", f"{label}: virtual price (Neary-Roberts) reproduces the capped choice",
                    np.max(np.abs(z2 - z)) < 1e-5, f"capped (n,d)=({n:.6f},{d:.6f}); unconstrained with m_N'=v_N: "
                                                   f"({z2[0]:.6f},{z2[1]:.6f})")
        # both caps: the bound on m_N can go either way, so Prop. 1(iii) needs 'without a cap on tokens' (mu = 0)
        if dev.delta == 0 and dev.qD == 0:
            out = []
            for dn, dd in [(0.3, 2.0), (1.5, 0.2)]:
                nb, db = (10.0, n0 - dn), (10.0, d0 - dd)
                z, _ = dev.solve(nb, db)
                n, d = z
                u, v, R, k, L, X = dev.parts(n, d)
                mN = dev.s * math.exp(n) / X
                nuN = envelope(dev, "nhi", (nb, db)) / X
                mu = envelope(dev, "dhi", (nb, db)) / X
                w = dev.wedge(n, d)
                out.append((w - 1, mN, mN + nuN, mu, abs(w / ((1 + mN + nuN) / (1 + mu)) - 1)))
            below = out[0][0] < out[0][1]
            above = out[1][0] > out[1][1]
            rec("A", f"{label}: both caps bind -> w-1 <= v_N always, but w-1 vs m_N either way", below and above
                and all(o[0] <= o[2] and o[4] < 2e-5 for o in out),
                "; ".join(f"w-1={o[0]:.3f}, m_N={o[1]:.3f}, v_N={o[2]:.3f}, mu={o[3]:.3f}" for o in out))
    # latency objective Z(N) = N^ell <= Zbar: case (b) with nu_N = ell * xi_Z / X_T, xi_Z = dPi*/d ln Zbar
    tech = dict(E=HOF.E, A=HOF.A, B=HOF.B, alpha=HOF.alpha, beta=HOF.beta)
    dev = Dev(tech, **base)
    z0, _ = dev.solve()
    ell = 0.8
    lnZbar = ell * (z0[0] - 0.5)
    nb, db = (10.0, lnZbar / ell), (10.0, 40.0)
    z, _ = dev.solve(nb, db)
    X = dev.parts(*z)[5]
    h = 1e-4
    xiZ = (dev.solve((10.0, (lnZbar + h) / ell), db)[1] - dev.solve((10.0, (lnZbar - h) / ell), db)[1]) / (2 * h)
    nuN = ell * xiZ / X
    mN = dev.s * math.exp(z[0]) / X
    w = dev.wedge(*z)
    rec("A", "latency objective: nu_N = ell xi_Z / X_T", abs(w / (1 + mN + nuN) - 1) < 2e-5,
        f"w={w:.6f} 1+m_N+ell xi_Z/X_T={1 + mN + nuN:.6f}")
    # menu of sizes {N_1, N_2}: at the chosen size, nu_N = (dPi/dn)/X_T, of either sign
    for nm, grid in [("menu binds as a cap", [z0[0] - 0.9, z0[0] + 1.6]), ("menu binds as a floor", [z0[0] - 1.6, z0[0] + 0.5])]:
        best = None
        for nn in grid:
            zz, V = dev.solve((nn, nn), (10.0, 40.0))
            if best is None or V > best[1]:
                best = (zz, V)
        zz = best[0]
        X = dev.parts(*zz)[5]
        nuN = dev.grad(zz)[0] / X
        mN = dev.s * math.exp(zz[0]) / X
        w = dev.wedge(*zz)
        sgn = nuN > 0 if "cap" in nm else nuN < 0
        rec("A", f"{nm}: w-1 = v_N, nu_N of the stated sign", abs(w / (1 + mN + nuN) - 1) < 1e-8 and sgn,
            f"chosen n={zz[0]:.3f} (unconstrained {z0[0]:.3f}); nu_N={nuN:.4f}; w-1={w - 1:.4f} m_N={mN:.4f}")


# ======================================================================================================== B. family
class Fam:
    """I members on one budget D, kappa = 1 technology: max sum_i [-Psi_i L(N_i,D) - s_i N_i] - q0 6 sum_i N_i D - qD D."""

    def __init__(self, tech, Psi, s, q0=1e-21, qD=0.0):
        self.t, self.Psi, self.s, self.q0, self.qD = tech, np.array(Psi), np.array(s), q0, qD
        self.I = len(Psi)

    def value(self, z):
        n, d = z[:-1], z[-1]
        t = self.t
        u = t["A"] * np.exp(-t["alpha"] * n)
        v = t["B"] * math.exp(-t["beta"] * d)
        X = 6 * self.q0 * np.exp(n + d)
        return float(np.sum(-self.Psi * (t["E"] + u + v) - self.s * np.exp(n) - X) - self.qD * math.exp(d))

    def grad(self, z):
        n, d = z[:-1], z[-1]
        t = self.t
        u = t["A"] * np.exp(-t["alpha"] * n)
        v = t["B"] * math.exp(-t["beta"] * d)
        X = 6 * self.q0 * np.exp(n + d)
        gn = self.Psi * t["alpha"] * u - self.s * np.exp(n) - X
        gd = np.sum(self.Psi * t["beta"] * v - X) - self.qD * math.exp(d)
        return np.append(gn, gd)

    def solve(self, bounds, z0):
        f = lambda z: -self.value(z)
        g = lambda z: -self.grad(z)
        r = minimize(f, np.array(z0), jac=g, method="L-BFGS-B", bounds=bounds, options=dict(ftol=1e-16, gtol=1e-14, maxiter=20000))
        z = r.x.copy()
        free = [i for i, (lo, hi) in enumerate(bounds) if lo + 1e-8 < z[i] < hi - 1e-8]
        for i, (lo, hi) in enumerate(bounds):
            if i not in free:
                z[i] = lo if abs(z[i] - lo) < abs(z[i] - hi) else hi
        for _ in range(40):
            gr = self.grad(z)[free]
            H = np.zeros((len(free), len(free)))
            for a, i in enumerate(free):
                zp, zm = z.copy(), z.copy()
                zp[i] += 1e-6
                zm[i] -= 1e-6
                H[:, a] = (self.grad(zp)[free] - self.grad(zm)[free]) / 2e-6
            step = np.linalg.solve(H, -gr)
            z[free] += step
            if np.max(np.abs(step)) < 1e-13:
                break
        return z, self.value(z)

    def objects(self, z):
        n, d = z[:-1], z[-1]
        t = self.t
        u = t["A"] * np.exp(-t["alpha"] * n)
        v = t["B"] * math.exp(-t["beta"] * d)
        X = 6 * self.q0 * np.exp(n + d)
        w = t["alpha"] * u / (t["beta"] * v)
        h = X / X.sum()
        pi = (h / w) / np.sum(h / w)
        wH = 1 / np.sum(h / w)
        mN = self.s * np.exp(n) / X
        mD = self.qD * math.exp(d) / X.sum()
        rho = self.Psi * t["beta"] * v / X
        return dict(w=w, h=h, pi=pi, wH=wH, mN=mN, mD=mD, rho=rho, X=X)


def fam_env(F, bounds, z0, idx, side, hstep=1e-4):
    def val(dl):
        b = [list(x) for x in bounds]
        b[idx][side] += dl
        return F.solve([tuple(x) for x in b], z0)[1]
    return (val(hstep) - val(-hstep)) / (2 * hstep)


def part_B():
    tech = dict(E=HOF.E, A=HOF.A, B=HOF.B, alpha=HOF.alpha, beta=HOF.beta)
    for qD, lab in [(0.0, "m_D=0"), (3e-12, "m_D>0")]:
        F = Fam(tech, Psi=[0.4, 1.0, 2.5], s=[1e-9, 3e-10, 5e-11], qD=qD)
        B0 = [(10.0, 35.0)] * 3 + [(10.0, 40.0)]
        z0 = np.array([19.0, 20.5, 22.0, 24.0])
        zs, _ = F.solve(B0, z0)
        # member 1 capped below its choice, member 3 floored above its choice
        B1 = [(10.0, zs[0] - 0.5), (10.0, 35.0), (zs[2] + 0.4, 35.0), (10.0, 40.0)]
        z, _ = F.solve(B1, zs)
        o = F.objects(z)
        xi1 = fam_env(F, B1, zs, 0, 1)             # dPi*/d nbar_1 = xi_1 >= 0
        xi3 = fam_env(F, B1, zs, 2, 0)             # dPi*/d nlo_3 = xi_3 g_n (g_n=-1) <= 0
        nu = np.array([xi1, 0.0, xi3]) / o["X"]
        vN = o["mN"] + nu
        lhs, rhs = np.sum(o["pi"] * (1 + vN)), (1 + o["mD"]) * o["wH"]
        rec("B", f"A-family(iii) {lab}: sum pi(1+v_N) = (1+m_D) w_H, envelope multipliers", abs(lhs / rhs - 1) < 3e-5
            and xi1 > 0 and xi3 < 0, f"lhs={lhs:.6f} rhs={rhs:.6f}; xi_1={xi1:.3e} xi_3={xi3:.3e}; "
                                    f"sum pi v_N={np.sum(o['pi'] * vN):.4f} vs w_H-1={o['wH'] - 1:.4f} (sum pi m_N={np.sum(o['pi'] * o['mN']):.4f})")
        # identity 1 - 1/w_H = sum h_i s_i
        rec("B", f"A-family(iii) {lab}: 1-1/w_H = sum h_i s_i", abs((1 - 1 / o["wH"]) - np.sum(o["h"] * (1 - 1 / o["w"]))) < 1e-12,
            f"{1 - 1 / o['wH']:.6f}")
        # (iv) all sizes capped: sum pi (1+m_N) <= (1+m_D) w_H
        B2 = [(10.0, zs[0] - 0.3), (10.0, zs[1] - 0.3), (10.0, zs[2] - 0.3), (10.0, 40.0)]
        z2, _ = F.solve(B2, zs)
        o2 = F.objects(z2)
        rec("B", f"A-family(iv) {lab}: every size capped -> sum pi(1+m_N) <= (1+m_D) w_H",
            np.sum(o2["pi"] * (1 + o2["mN"])) <= (1 + o2["mD"]) * o2["wH"],
            f"{np.sum(o2['pi'] * (1 + o2['mN'])):.4f} <= {(1 + o2['mD']) * o2['wH']:.4f}")
        # (vi) token cap with the same size constraints
        for frac in (0.5, 0.8):
            B3 = [B1[0], B1[1], B1[2], (10.0, z[3] + math.log(frac))]   # cap relative to the D chosen under B1
            z3, _ = F.solve(B3, zs)
            o3 = F.objects(z3)
            x1 = fam_env(F, B3, zs, 0, 1)
            x3 = fam_env(F, B3, zs, 2, 0)
            xD = fam_env(F, B3, zs, 3, 1)
            vN3 = o3["mN"] + np.array([x1, 0.0, x3]) / o3["X"]
            mu = xD / o3["X"].sum()
            lhs, rhs = np.sum(o3["pi"] * (1 + vN3)), (1 + o3["mD"] + mu) * o3["wH"]
            member = all(((o3["w"][i] <= (1 + vN3[i]) / (1 + o3["mD"]) + 1e-9) == (o3["rho"][i] >= 1 + o3["mD"] - 1e-9))
                         for i in range(3))
            rec("B", f"A-family(vi) {lab}, D capped at {int(frac * 100)}%: identity and member bound iff rho_i >= 1+m_D",
                abs(lhs / rhs - 1) < 3e-5 and member and mu > 0,
                f"lhs={lhs:.5f} rhs={rhs:.5f} mu={mu:.3f}; rho={np.round(o3['rho'], 3).tolist()} 1+m_D={1 + o3['mD']:.3f}")
            if qD > 0:
                # the gloss 'would train on more tokens at its own size': with the data cost shared in proportion to
                # compute, member i's own marginal condition in d is rho_i vs 1 + h_i qD D / X_i = 1 + m_D (exact);
                # bearing the whole data cost it would be rho_i vs 1 + qD D / X_i (different).
                own_prop = 1 + o3["h"] * F.qD * math.exp(z3[-1]) / o3["X"]
                own_full = 1 + F.qD * math.exp(z3[-1]) / o3["X"]
                rec("B", f"A-family(vi) gloss {lab}, cap {int(frac * 100)}%: proportional data-cost share gives exactly 1+m_D",
                    np.allclose(own_prop, 1 + o3["mD"], rtol=1e-12), f"1+m_D={1 + o3['mD']:.4f}; whole cost: {np.round(own_full, 4).tolist()}")


# ======================================================================================================== C. Cor. A-general
def hicks_levels(f, N, D, h=1e-4):
    """Hicks elasticity from level derivatives of f(N,D) by central differences (relative steps)."""
    hN, hD = h * N, h * D
    fN = (f(N + hN, D) - f(N - hN, D)) / (2 * hN)
    fD = (f(N, D + hD) - f(N, D - hD)) / (2 * hD)
    fNN = (f(N + hN, D) - 2 * f(N, D) + f(N - hN, D)) / hN ** 2
    fDD = (f(N, D + hD) - 2 * f(N, D) + f(N, D - hD)) / hD ** 2
    fND = (f(N + hN, D + hD) - f(N + hN, D - hD) - f(N - hN, D + hD) + f(N - hN, D - hD)) / (4 * hN * hD)
    return -fN * fD * (N * fN + D * fD) / (N * D * (fNN * fD ** 2 - 2 * fND * fN * fD + fDD * fN ** 2))


def part_C():
    tH = dict(E=HOF.E, A=HOF.A, B=HOF.B, alpha=HOF.alpha, beta=HOF.beta, kappa=1.0)

    def loss_k(t):
        return lambda N, D: t["E"] + (t["A"] * N ** -t["alpha"] + t["B"] * D ** -t["beta"]) ** t.get("kappa", 1.0)

    techs = {
        "Hoffmann": (loss_k(tH), tH["E"]),
        "reference (kappa free)": (loss_k(REF), REF["E"]),
        # CES output Y = (0.4 N^-1 + 0.6 D^-1)^-1 (sigma = 0.5), reported as a loss L = 2 + Y^-0.3 (y = -ln(L-2) = 0.3 ln Y)
        "CES sigma=0.5 (as a loss)": (lambda N, D: 2.0 + (0.4 / N + 0.6 / D) ** 0.3, 2.0),
        "non-separable": (lambda N, D: 1.7 + 400 * N ** -0.34 + 410 * D ** -0.28 + 3e3 * (N * D) ** -0.2, 1.7),
    }
    costs = {
        "6ND": lambda N, D: 6 * N * D,
        "N^1.3 D^0.9": lambda N, D: N ** 1.3 * D ** 0.9,
        "linear": lambda N, D: 6e10 * N + 6e9 * D,
        "6ND + data cost": lambda N, D: 6 * N * D + 1e12 * D,
        "CES cost sigma_C=2": lambda N, D: (N ** 0.5 + (0.05 * D) ** 0.5) ** 2,
        "6 N^1.1 D": lambda N, D: 6 * N ** 1.1 * D,
    }
    pairs = [("Hoffmann", "6ND", 1e21), ("Hoffmann", "N^1.3 D^0.9", 1e22), ("Hoffmann", "linear", 1e21),
             ("reference (kappa free)", "6ND + data cost", 1e22), ("CES sigma=0.5 (as a loss)", "CES cost sigma_C=2", 1e11),
             ("non-separable", "6 N^1.1 D", 1e22), ("reference (kappa free)", "linear", 1e21)]
    for tname, cname, C0 in pairs:
        Lf, E = techs[tname]
        Cf = costs[cname]
        y = lambda N, D: -math.log(Lf(N, D) - E)
        c0 = math.log(C0)

        def d_on(n, c):  # solve ln C(e^n, e^d) = c for d
            return brentq(lambda d: math.log(Cf(math.exp(n), math.exp(d))) - c, -50, 120, xtol=1e-14)

        def n_hi(c):  # largest n on the isocost (d -> -50)
            f = lambda n: math.log(Cf(math.exp(n), math.exp(-50.0))) - c
            return brentq(f, -50, 120) - 1e-3 if f(45) > 0 else 45.0

        def ystar(c):
            r = minimize_scalar(lambda n: -y(math.exp(n), math.exp(d_on(n, c))), bounds=(5, n_hi(c)), method="bounded",
                                options=dict(xatol=1e-12))
            return r.x, -r.fun

        n_s, y_s = ystar(c0)
        d_s = d_on(n_s, c0)
        N, D = math.exp(n_s), math.exp(d_s)
        # elasticities of the technology and of the isocost, in levels
        sig = hicks_levels(lambda a, b: y(a, b), N, D)
        sigC = hicks_levels(lambda a, b: math.log(Cf(a, b)), N, D)
        h = 1e-5
        gn = (math.log(Cf(N * math.exp(h), D)) - math.log(Cf(N * math.exp(-h), D))) / (2 * h)
        gd = (math.log(Cf(N, D * math.exp(h))) - math.log(Cf(N, D * math.exp(-h)))) / (2 * h)
        r = gn / gd
        hh = 2e-3
        Y = lambda n: y(math.exp(n), math.exp(d_on(n, c0)))
        ynn = (Y(n_s + hh) - 2 * Y(n_s) + Y(n_s - hh)) / hh ** 2
        Lc = lambda n: Lf(math.exp(n), math.exp(d_on(n, c0)))
        Lnn = (Lc(n_s + hh) - 2 * Lc(n_s) + Lc(n_s - hh)) / hh ** 2
        dc = 1e-3
        dys = (ystar(c0 + dc)[1] - ystar(c0 - dc)[1]) / (2 * dc)
        Lstar = lambda c: Lf(math.exp(ystar(c)[0]), math.exp(d_on(ystar(c)[0], c)))
        dLs = (Lstar(c0 + dc) - Lstar(c0 - dc)) / (2 * dc)
        inv_sC = 0.0 if abs(sigC) > 1e6 else 1 / sigC
        lhs = 1 / sig - inv_sC
        rhs = abs(ynn) / (r * (1 + r) * gd * dys)
        rhs_loss = Lnn / (r * (1 + r) * gd * abs(dLs))
        naive_loss = Lnn / abs(dLs)
        ok = abs(lhs / rhs - 1) < 2e-3 and abs(rhs_loss / rhs - 1) < 2e-3 and ynn < 0
        rec("C", f"Cor. A-general: {tname} | {cname}", ok,
            f"sigma*={sig:.4f} sigma_C={'inf' if inv_sC == 0 else f'{sigC:.4f}'} r={r:.3f} g_d={gd:.3f}; lhs={lhs:.5f} "
            f"rhs(y)={rhs:.5f} rhs(L, weighted)={rhs_loss:.5f}; unweighted L_nn/|dL*/dc|={naive_loss:.5f} "
            f"(differs by r(1+r)g_d={r * (1 + r) * gd:.3f})")


# ======================================================================================================== D. Prop. 2(ii)
def part_D():
    for lab, t in [("Hoffmann (alpha != beta)", HOF), ("equal exponents", sl.Chinchilla(E=1.7, A=400.0, B=900.0, alpha=0.31, beta=0.31))]:
        a = t.beta / (t.alpha + t.beta)
        gam = t.alpha * t.beta / (t.alpha + t.beta)
        G = (t.alpha * t.A / (t.beta * t.B)) ** (1 / (t.alpha + t.beta))
        cbar = math.log(3e20 / 6)
        n0, d0 = math.log(G) + a * cbar, -math.log(G) + (1 - a) * cbar
        U0, V0 = t.A * math.exp(-t.alpha * n0), t.B * math.exp(-t.beta * d0)
        cp = np.linspace(-3.5, 3.5, 25)   # 25 on-path levels over about three decades (a different design from the paper's)
        y0 = t.E + (U0 + V0) * np.exp(-gam * cp)

        def crit_free(rN, rD):   # profile over E, U, V (split free)
            Xm = np.column_stack([np.ones_like(cp), np.exp(-rN * cp), np.exp(-rD * cp)])
            coef, *_ = np.linalg.lstsq(Xm, y0, rcond=None)
            res = Xm @ coef - y0
            return float(res @ res)

        def crit_split(rN, rD, frac):   # split held at frac = U/(U+V); profile over E and the level
            z = frac * np.exp(-rN * cp) + (1 - frac) * np.exp(-rD * cp)
            Xm = np.column_stack([np.ones_like(cp), z])
            coef, *_ = np.linalg.lstsq(Xm, y0, rcond=None)
            res = Xm @ coef - y0
            return float(res @ res)

        ts = gam * np.array([1e-3, 2e-3, 4e-3])
        sl_free = np.polyfit(np.log(ts), np.log([crit_free(gam + 2 * s, gam - s) for s in ts]), 1)[0]
        frac = U0 / (U0 + V0)
        sl_split = np.polyfit(np.log(ts), np.log([crit_split(gam + 2 * s, gam - s, frac) for s in ts]), 1)[0]
        uflat = np.array([V0, -U0]) / max(U0, V0)
        sl_flat = np.polyfit(np.log(ts), np.log([crit_split(gam + uflat[0] * s, gam + uflat[1] * s, frac) for s in ts]), 1)[0]
        # sigma* along (2,-1): alpha = r_N/a, beta = r_D/(1-a)
        sig = lambda rN, rD: 2 / (2 + rN / a + rD / (1 - a))
        dsig = [abs(sig(gam + 2 * s, gam - s) - sig(gam, gam)) for s in ts]
        sl_sig = np.polyfit(np.log(ts), np.log(dsig), 1)[0]
        rec("D", f"Prop. 2(ii) {lab}: profiled over the split -> quartic; fixed split -> quadratic except along (V0,-U0)",
            abs(sl_free - 4) < 0.05 and abs(sl_split - 2) < 0.05 and abs(sl_flat - 4) < 0.05 and abs(sl_sig - 1) < 0.05,
            f"log-log slopes: split free (2,-1) {sl_free:.3f}; split fixed (2,-1) {sl_split:.3f}; split fixed (V0,-U0) "
            f"{sl_flat:.3f}; |d sigma*| along (2,-1) {sl_sig:.3f}")


# ======================================================================================================== E. Prop. A6
def xstar(lossf, c, lo=-8, hi=14):
    """argmin over x = ln(D/N) of loss at compute C = e^c (C = 6ND)."""
    def Lx(x):
        n = (c - math.log(6) - x) / 2
        return lossf(n, n + x)
    r = minimize_scalar(Lx, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-12))
    return r.x


def part_E(R=3000, seed=11):
    t = HOF
    lossH = lambda n, d: t.E + t.A * math.exp(-t.alpha * n) + t.B * math.exp(-t.beta * d)
    S = t.alpha + t.beta
    a = t.beta / S
    # E1: constant tilt psi_N = -psi_D = tt/2 shifts every argmin by tt = -2 chi / S
    ok, det = True, []
    for tt in (-0.61, 0.45):
        chi = t.beta * (-tt / 2) - t.alpha * (tt / 2)
        lossT = lambda n, d, tt=tt: lossH(n + tt / 2, d - tt / 2)
        sh = [xstar(lossT, math.log(C)) - xstar(lossH, math.log(C)) for C in (1e19, 1e22, 1e25)]
        ok &= max(abs(np.array(sh) - tt)) < 1e-6 and abs(-2 * chi / S - tt) < 1e-12
        det.append(f"t={tt}: shifts {np.round(sh, 7).tolist()}")
    rec("E", "A6(vii): constant tilt translates every IsoFLOP argmin by t = -2chi/S", ok, "; ".join(det))

    # E2: sharpness of the UPPER end point of X*_tau(c): (ii) construction with a kink at c_J, then the tilt ln tau
    cJ, c = math.log(3e21), math.log(1e25)
    tau = 1.84
    eU = 0.30
    e0 = 1 - 2 * a
    psiNp = (eU - e0) / (2 * (1 - a))     # Delta' = eU - e0 > 0: raise psi_N only
    def lossK(n, d):
        cc = math.log(6) + n + d
        pN = psiNp * max(cc - cJ, 0.0)
        return lossH(n + pN, d)
    lossDev = lambda n, d: lossK(n + math.log(tau) / 2, d - math.log(tau) / 2)
    target = xstar(lossH, cJ) + eU * (c - cJ) + math.log(tau)
    got = xstar(lossDev, c)
    # marginal products and single-peakedness of the developer technology at c
    xs = np.linspace(target - 6, target + 6, 241)
    Ls = []
    for x in xs:
        n = (c - math.log(6) - x) / 2
        Ls.append(lossDev(n, n + x))
    dL = np.diff(Ls)
    peaks = int(np.sum(np.diff(np.sign(dL)) != 0))
    h = 1e-6
    n = (c - math.log(6) - got) / 2
    Fn = -(lossDev(n + h, n + got) - lossDev(n - h, n + got)) / (2 * h)
    Fd = -(lossDev(n, n + got + h) - lossDev(n, n + got - h)) / (2 * h)
    below = abs(xstar(lossDev, cJ - 1.0) - (xstar(lossH, cJ - 1.0) + math.log(tau)))
    rec("E", "A6(vii) sharpness: upper end of X*_tau(1e25) is x*_dev of an admissible developer technology",
        abs(got - target) < 1e-6 and peaks == 1 and Fn > 0 and Fd > 0 and below < 1e-6,
        f"target {target:.6f} got {got:.6f}; one sign change of L_x: {peaks == 1}; F_n={Fn:.2e} F_d={Fd:.2e}; "
        f"below c_J within ln tau of the anchored technology: {below:.1e}")

    # E3: widened union of simultaneous Scheffe bands (linear paths, OLS on noisy minima) covers x*_dev on [c_J, cbar]
    rng = np.random.default_rng(seed)
    studies = [  # (intercept at c=ln 1e20, slope, budgets (log10 C), noise sd of a fitted minimum)
        (2.8, 0.05, np.arange(18.0, 21.6, 0.5), 0.06),
        (3.2, 0.12, np.arange(19.0, 21.1, 0.5), 0.10),
        (3.6, 0.20, np.arange(18.5, 20.6, 0.5), 0.08),
    ]
    cref = math.log(1e20)
    cgrid = np.log(10.0 ** np.linspace(21.5, 25.5, 41))
    jstar, tilt = 1, math.log(tau)   # the developer lies within tau of study 1, with the worst-case tilt
    q = math.sqrt(chi2.ppf(0.95, 2))
    cover_path, cover_union = 0, 0
    for _ in range(R):
        los, his, covj = [], [], []
        for j, (b0, b1, lc, sd) in enumerate(studies):
            cc = np.log(10.0 ** lc)
            Xm = np.column_stack([np.ones_like(cc), cc - cref])
            yv = b0 + b1 * (cc - cref) + sd * rng.standard_normal(len(cc))
            XtXi = np.linalg.inv(Xm.T @ Xm)
            bh = XtXi @ Xm.T @ yv
            G = np.column_stack([np.ones_like(cgrid), cgrid - cref])
            fit = G @ bh
            se = sd * np.sqrt(np.einsum("ij,jk,ik->i", G, XtXi, G))
            lo, hi = fit - q * se, fit + q * se      # Scheffe: simultaneous over all c (known sd)
            truth = b0 + b1 * (cgrid - cref)
            covj.append(bool(np.all((truth >= lo) & (truth <= hi))))
            los.append(lo - math.log(tau))
            his.append(hi + math.log(tau))
        xdev = studies[jstar][0] + studies[jstar][1] * (cgrid - cref) + tilt
        inunion = np.any([(xdev >= los[j]) & (xdev <= his[j]) for j in range(3)], axis=0)
        cover_path += covj[jstar]
        cover_union += bool(np.all(inunion))
    cp_, cu_ = cover_path / R, cover_union / R
    se = math.sqrt(0.95 * 0.05 / R)
    rec("E", "A6(viii): widened union covers x*_dev(c) on [c_J, cbar] simultaneously w.p. >= 1 - max p_j",
        cp_ >= 0.95 - 2 * se and cu_ >= cp_ - 1e-12, f"R={R}: band of path j* covers {cp_:.4f}; widened union covers {cu_:.4f}")

    # E4: the band must cover the path; a linear band for a path whose slope varies inside [e_L, e_U] can fail
    b0, b1, lc, sd = studies[0]
    kq = 0.012   # slope rises from 0.05 at 1e20 by 2 kq per nat: still inside a slope range such as [-0.16, 0.31]
    miss = 0
    for _ in range(R):
        cc = np.log(10.0 ** lc)
        truth_d = b0 + b1 * (cc - cref) + kq * (cc - cref) ** 2
        Xm = np.column_stack([np.ones_like(cc), cc - cref])
        yv = truth_d + sd * rng.standard_normal(len(cc))
        XtXi = np.linalg.inv(Xm.T @ Xm)
        bh = XtXi @ Xm.T @ yv
        G = np.column_stack([np.ones_like(cgrid), cgrid - cref])
        fit = G @ bh
        se_ = sd * np.sqrt(np.einsum("ij,jk,ik->i", G, XtXi, G))
        truth = b0 + b1 * (cgrid - cref) + kq * (cgrid - cref) ** 2
        miss += not np.all((truth >= fit - q * se_) & (truth <= fit + q * se_))
    slope_hi = b1 + 2 * kq * (cgrid[-1] - cref)
    rec("E", "A6(vii) caveat: a band from a linear fit needs a linear path (curved path, slope within [-0.16,0.31])",
        miss / R > 0.10, f"non-coverage {miss / R:.3f} with the path's slope rising to {slope_hi:.3f}")


# ======================================================================================================== F. Cor. A2
def part_F():
    al, be = HOF.alpha, HOF.beta
    gam = al * be / (al + be)
    w = 13.5
    exact = ((al + be * w) / (al + be)) ** (1 / gam) * w ** (-1 / al)
    rho = (al + be) / 2
    X = math.log(w) / rho
    cosh_form = math.cosh(rho / 2 * X) ** (2 / rho)
    rec("F", "Cor. A2: Hoffmann exponents at w=13.5, exact C/C_min vs cosh form", abs(exact - 101.2) < 0.05 and abs(cosh_form - 78.0) < 0.05,
        f"exact {exact:.2f}, cosh {cosh_form:.2f} (round-2/WP3 text had 102.0 and 78.5, the values at w=13.536; corrected to 101.2 and 78.0)")
    # homothetic case is exact
    al2 = be2 = 0.3
    g2 = al2 * be2 / (al2 + be2)
    ex2 = ((al2 + be2 * w) / (al2 + be2)) ** (1 / g2) * w ** (-1 / al2)
    co2 = math.cosh(0.5 * math.log(w)) ** (2 / al2)
    rec("F", "Cor. A2: alpha = beta, cosh form exact", abs(ex2 / co2 - 1) < 1e-12, f"{ex2:.6f} vs {co2:.6f}")


# ======================================================================================================== G. Prop. A2(iii)
def part_G():
    t = BES
    phi0 = None

    def to_prim(phi):
        S, a, gam, lnG, lnK = phi
        b1, a1 = a * S, (1 - a) * S
        kap = gam / (a * (1 - a) * S)
        A = (b1 / S) * math.exp(lnK / kap) * math.exp(a1 * lnG)
        B = a1 * A / (b1 * math.exp(S * lnG))
        return a1, b1, kap, A, B

    S0 = t.alpha + t.beta
    a0 = t.beta / S0
    g0 = t.alpha * t.beta / S0
    lnG0 = math.log(t.G)
    lnK0 = math.log(t.K)
    phi0 = np.array([S0, a0, g0, lnG0, lnK0])

    def yfun(phi, n, d):
        a1, b1, kap, A, B = to_prim(phi)
        return -kap * np.log(A * np.exp(-a1 * n) + B * np.exp(-b1 * d))

    budgets = np.log(np.logspace(18.5, 21.5, 9))
    offs = np.array([-2, -1, 0, 1, 2], float)
    out = {}
    for vs in (0.05, 0.1, 0.2):
        n_list, d_list = [], []
        for c in budgets:
            ns = lnG0 + a0 * (c - math.log(6))
            ds = c - math.log(6) - ns
            for o in offs:     # transverse offset in x = d - n by vs * o, compute held fixed
                n_list.append(ns - vs * o / 2)
                d_list.append(ds + vs * o / 2)
        n_, d_ = np.array(n_list), np.array(d_list)
        J = np.zeros((len(n_), 5))
        for k in range(5):
            e = np.zeros(5)
            e[k] = 1e-6 * max(1.0, abs(phi0[k]))
            J[:, k] = (yfun(phi0 + e, n_, d_) - yfun(phi0 - e, n_, d_)) / (2 * e[k])
        Iinv = np.linalg.inv(J.T @ J)
        grad_M = np.array([0, -2 * math.log(1e21 / 6), 0, -2, 0])    # ln M*(C0) = -2 ln G + (1-2a) ln(C0/6)
        out[vs] = (1 / Iinv[0, 0], 1 / Iinv[3, 3], 1 / (grad_M @ Iinv @ grad_M))
    v = np.log([0.05, 0.1, 0.2])
    slopes = [np.polyfit(v, np.log([out[s][k] for s in (0.05, 0.1, 0.2)]), 1)[0] for k in range(3)]
    rec("G", "A2(iii): efficient-information slopes in varsigma (S: 4, ln G: 2, ln M*(1e21): 2)",
        abs(slopes[0] - 4) < 0.05 and abs(slopes[1] - 2) < 0.05 and abs(slopes[2] - 2) < 0.05,
        f"slopes S {slopes[0]:.3f}, ln G {slopes[1]:.3f}, ln M*(1e21) {slopes[2]:.3f}; I(ln M*)/[I(ln G)/4] "
        f"= {out[0.1][2] / (out[0.1][1] / 4):.1f} at varsigma 0.1")


# ======================================================================================================== H. numbers
def part_H():
    f1 = pd.read_csv(os.path.join(ROOT, "output/tables/fig1_geometry_numbers.csv"))
    get = lambda o: float(f1.loc[f1["object"] == o, "value"].iloc[0])
    M = get("SmolLM2-1.7B:M")
    MoM = get("SmolLM2-1.7B:M_over_Mstar")
    w = get("SmolLM2-1.7B:w_ref")
    sig = 2 / (2 + REF["alpha"] + REF["beta"])
    k = 1 / sig - 1
    rec("H", "Section I: sigma*=0.701, 1/sigma*-1=0.427, M about 6,400, about 290 times, w about 11.3, s about 0.91, "
             "about 10 percent", abs(sig - 0.701) < 5e-4 and abs(k - 0.427) < 5e-4 and abs(M - 6400) < 100
        and abs(MoM - 290) < 10 and abs(w - 11.3) < 0.05 and abs((w - 1) / w - 0.91) < 0.005 and abs((w - 1) - 10) < 0.5
        and abs(math.exp(k * math.log(MoM)) - w) < 1e-6,
        f"sigma*={sig:.4f} k={k:.5f} M={M:.0f} M/M*={MoM:.1f} w={w:.3f} (=exp(k ln M/M*) {math.exp(k * math.log(MoM)):.3f}) "
        f"s={(w - 1) / w:.3f} w-1={w - 1:.2f}")
    mb = pd.read_csv(os.path.join(ROOT, "data/processed/rb5_sign/model_bounds_S1.csv"))
    sm = mb[mb["model"] == "SmolLM2-1.7B"].iloc[0]
    ll = mb[mb["model"] == "Llama-3.1-405B"].iloc[0]
    rec("H", "Section II: SmolLM2 1.7B about 47 times the upper end; M about 6,400, 1.1e23 FLOP",
        abs(math.exp(sm["dlo"]) - 47) < 0.5 and abs(sm["Cmp"] / 1.1e23 - 1) < 0.05,
        f"exp(dlo)={math.exp(sm['dlo']):.2f} (M={sm['M']:.0f}, Mstar_hi={sm['Mstar_hi_eff']:.1f}, C={sm['Cmp']:.3e})")
    rec("H", "Section II: Llama 3.1 405B inside the set, 38 tokens per parameter, 3.8e25 FLOP",
        ll["dlo"] < 0 < ll["dhi"] and abs(ll["M"] - 38) < 0.5 and abs(ll["Cmp"] / 3.8e25 - 1) < 0.01,
        f"dlo={ll['dlo']:.3f} dhi={ll['dhi']:.3f} M={ll['M']:.2f} C={ll['Cmp']:.3e}")
    ms = pd.read_csv(os.path.join(ROOT, "output/tables/rb5_sign_mstar_sets.csv"))
    row = ms[(ms["set"] == "S1") & (ms["conv"] == "total") & (np.isclose(ms["C"], 1e24)) & (ms["paths"] == "Chinchilla; Llama 3; Marin")].iloc[0]
    width = math.log(row["Mstar_hi"] / row["Mstar_lo"])
    rec("H", "Remark A-pi-param: M*(1e24) set about 3.8 log points", abs(width - 3.8) < 0.05,
        f"[{row['Mstar_lo']:.2f}, {row['Mstar_hi']:.1f}] width {width:.3f}; 32-technology range ln(426/1.4)={math.log(426 / 1.4):.2f}")
    Sref = REF["alpha"] + REF["beta"]
    rec("H", "Section II: tilt 0.26 -> tau approx 1.8 (1.84) under the reference exponents; 3.4 under DataDecide's",
        abs(math.exp(2 * 0.26 / Sref) - 1.84) < 0.005 and abs(math.exp(2 * 0.2561 / 0.4154) - 3.4) < 0.05,
        f"exp(2*0.26/S_ref)={math.exp(2 * 0.26 / Sref):.4f}; exp(2*0.2561/0.4154)={math.exp(2 * 0.2561 / 0.4154):.3f}")
    # Remark 1 / A3 simulation facts
    s = pd.read_csv(os.path.join(ROOT, "output/tables/ra5_theory_singular_mc_slopes.csv"))
    m = pd.read_csv(os.path.join(ROOT, "output/tables/ra5_theory_singular_mc.csv"))
    q75 = (s["slope_q75_unres"] / 2).tolist()
    q90 = (s["slope_q90_unres"] / 2).tolist()
    rq = (s["slope_q75_restricted"] / 2).tolist()
    q25 = (s["slope_q25_unres"] / 2).tolist()
    coinc = 1 - m["share_split"]
    rec("H", "Remark 1/A3: n^-0.24, n^-0.27..-0.28, restricted n^-0.50, q25 n^-1/2, about 40 percent coinciding, R=100, "
             "five noise levels 1e-3..1e-7", all(abs(x - 0.24) < 0.005 for x in q75) and all(0.27 <= round(x, 2) <= 0.28 for x in q90)
        and all(abs(x - 0.50) < 0.005 for x in rq) and all(abs(x - 0.5) < 0.01 for x in q25) and 0.35 <= coinc.mean() <= 0.45
        and set(m["R"]) == {100} and sorted(set(m["tau"])) == [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        f"q75 {np.round(q75, 3).tolist()} q90 {np.round(q90, 3).tolist()} restricted {np.round(rq, 3).tolist()} "
        f"q25 {np.round(q25, 3).tolist()}; coinciding {coinc.min():.2f}-{coinc.max():.2f} (mean {coinc.mean():.2f})")


# ======================================================================================================== I. symbolic
def part_I():
    # Prop. A-family (iii)/(vi): with rho_i = (1+v_i)/w_i and sum_i h_i rho_i = 1 + m_D + mu,
    # sum_i pi_i (1+v_i) = (1+m_D+mu) w_H, pi_i = (h_i/w_i)/sum_j(h_j/w_j), w_H = 1/sum_j(h_j/w_j).
    h1, h2, w1, w2, w3, v1, v2, v3, mD, mu = sp.symbols("h1 h2 w1 w2 w3 v1 v2 v3 m_D mu", positive=True)
    h3 = 1 - h1 - h2
    hs, ws, vs = [h1, h2, h3], [w1, w2, w3], [v1, v2, v3]
    rho = [(1 + v) / w for v, w in zip(vs, ws)]
    wH = 1 / sum(h / w for h, w in zip(hs, ws))
    pi = [(h / w) * wH for h, w in zip(hs, ws)]
    foc = sum(h * r for h, r in zip(hs, rho)) - (1 + mD + mu)
    lhs = sum(p * (1 + v) for p, v in zip(pi, vs))
    diff = sp.simplify(lhs - (1 + mD + mu) * wH - wH * foc)
    rec("I", "A-family: sum pi(1+v) - (1+m_D+mu) w_H = w_H * (FOC residual) identically", diff == 0, f"simplified: {diff}")
    s_share = sp.simplify((1 - 1 / wH) - sum(h * (1 - 1 / w) for h, w in zip(hs, ws)))
    rec("I", "A-family: 1 - 1/w_H = sum h_i s_i identically", s_share == 0, f"{s_share}")
    rec("I", "A-family: sum pi_i w_i = w_H identically", sp.simplify(sum(p * w for p, w in zip(pi, ws)) - wH) == 0, "")
    # Prop. 1(ii): X_S = varrho q_T 2 N T, X_T = q_T 6 N D, m_N = lambda d ln X_S / d ln N * X_S / X_T with phi=1, eta=0
    lam, vr, qT, N, D, T = sp.symbols("lambda varrho q_T N D T", positive=True)
    XS, XT = vr * qT * 2 * N * T, qT * 6 * N * D
    mN = lam * sp.diff(XS, N) * N / XT
    w = 1 + mN
    ok = sp.simplify(w - (1 + lam * vr * T / (3 * D))) == 0 and sp.simplify((w - 1) / w - lam * XS / (XT + lam * XS)) == 0
    rec("I", "Prop. 1(ii): w = 1 + lambda varrho T/(3D), s = lambda X_S/(X_T + lambda X_S)", ok, "")
    # Prop. 1(iii): distillation m_D = 2 N_T D q_T / (6 N D q_T) = N_T/(3N)
    NT = sp.symbols("N_T", positive=True)
    rec("I", "Prop. 1(iii): m_D = N_T/(3N)", sp.simplify(2 * NT * D * qT / (6 * N * D * qT) - NT / (3 * N)) == 0, "")
    # factor-augmenting contamination: w_L'/w_L = exp(beta psi_D - alpha psi_N)
    al, be, pN, pD, n, d, A, B = sp.symbols("alpha beta psi_N psi_D n d A B", real=True)
    wL = al * A * sp.exp(-al * n) / (be * B * sp.exp(-be * d))
    wLp = al * A * sp.exp(-al * (n + pN)) / (be * B * sp.exp(-be * (d + pD)))
    rec("I", "Prop. 1(iii): w_L/w_L' = e^{-chi}, chi = beta psi_D - alpha psi_N",
        sp.simplify(wL / wLp - sp.exp(-(be * pD - al * pN))) == 0, "")
    # Remark 1: sigma* = 2/{2 + gamma/[a(1-a)]} = 2/(2+alpha+beta) under kappa = 1
    alp, bet = sp.symbols("alpha beta", positive=True)
    a_ = bet / (alp + bet)
    g_ = alp * bet / (alp + bet)
    rec("I", "Remark 1: 2/{2+gamma/[a(1-a)]} = 2/(2+alpha+beta)", sp.simplify(2 / (2 + g_ / (a_ * (1 - a_))) - 2 / (2 + alp + bet)) == 0, "")
    # Cor. A-general (ii): linear cost in logs: 1/sigma_C = 0 and r(1+r) g_d = s_N/s_D
    qN, qD2 = sp.symbols("q_N q_D", positive=True)
    Cc = sp.log(qN * sp.exp(n) + qD2 * sp.exp(d))
    gn, gd = sp.diff(Cc, n), sp.diff(Cc, d)
    Pc = gn * gd * (gn + gd)
    Qc = -(sp.diff(Cc, n, 2) * gd ** 2 - 2 * sp.diff(Cc, n, d) * gn * gd + sp.diff(Cc, d, 2) * gn ** 2)
    inv_sC = sp.simplify(1 + Qc / Pc)
    r = gn / gd
    rec("I", "Cor. A-general(ii): linear cost has 1/sigma_C = 0 and r(1+r)g_d = s_N/s_D",
        inv_sC == 0 and sp.simplify(r * (1 + r) * gd - gn / gd) == 0, f"1/sigma_C = {inv_sC}")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()
    part_F()
    part_G()
    part_H()
    part_I()
    df = pd.DataFrame(ROWS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{int(df['ok'].sum())}/{len(df)} checks passed; wrote {OUT}")
