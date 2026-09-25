"""Verification of Proposition A-pi parts (vii)-(viii) and of other changed appendix statements (fix list P5, P8, P11).

Proposition A-pi (vii): interval anchor, path-slope range and tilt allowance tau; the widened set
    X*_tau(c) = [x_lo + e_L (c - c_J) - ln tau,  x_hi + e_U (c - c_J) + ln tau];
sharp for a point anchor and an anchored technology in the generalized family; a simultaneous confidence band for the
anchored path, widened by ln tau, covers the developer's x*_dev(c) with at least the band's coverage.
Proposition A-pi (viii): union over several anchored paths; coverage at least 1 - max_j p_j when the developer's
technology lies within tau of at least one path.

Checks.
  pi.tilt    A constant factor-augmenting tilt with psi_N = -psi_D = t/2 leaves compute unchanged and translates every
             IsoFLOP profile by t in x: argmin shifted by exactly t at every compute, positive marginal products and one
             sign change of Y_x per profile, for (a) the reference technology, (b) the Proposition A-pi(ii) construction
             (a path that bends at c_J) and (c) a non-separable technology; in the generalized family the shift equals
             -2 chi / S (Lemma A3(iii)).
  pi.sharp   Every point of X*_tau(c) (point anchor, generalized family) is x*_dev(c) of a developer technology that
             satisfies (a)-(d): the (ii) construction reproduces the anchored technology up to c_J, then the tilt.
  pi.contain 20,000 random draws of anchor, slope paths, tilt paths and (c, x): x*_dev(c) in X*_tau(c); sign of w-1
             and the magnitude bounds of (iv) hold for the developer's technology (generalized family, constant tilt).
  pi.band    Monte Carlo coverage (4,000 replications): Working-Hotelling simultaneous 95 percent bands for three
             linear anchored paths (known noise); the widened band of the path the developer lies within tau of, and the
             widened union, cover x*_dev(c) at every c on a grid with frequency >= 0.95 (within Monte Carlo error).
  corA2      Corollary A2: with unequal exponents ln w = ((alpha+beta)/2) ln(M/M*(C)) exactly; C/C_min from Lemma A5
             differs from the cosh form beyond second order in ln w (Hoffmann exponents: 101.2 vs 78.0 at w = 13.5;
             proof re-check 2026-09-25: the earlier 102.0 vs 78.5 was computed at w = 13.536).
  defA2.K    Definition A2: the generalized family's frontier R*(C) = K (C/6)^(-gamma) with
             K = ((S/b1) A G^(-a1))^kappa, G = (a1 A / (b1 B))^(1/S), gamma = kappa a1 b1 / S (brute force).
  A2.iii     Proposition A2(iii): in a nine-budget, five-offset factorial design the efficient information for
             ln M*(C0) at a fixed compute C0 (combining the scores for ln G and a) is O(varsigma^2), like that for ln G,
             and that for S is O(varsigma^4).
Deterministic; CPU only; well under a minute. Usage: .venv/bin/python code/analysis/rb5_theory/verify_prop_pi.py
Writes code/analysis/rb5_theory/results/prop_pi_checks.csv; exit code 1 on any FAIL.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location("m7common", os.path.join(ROOT, "code", "analysis", "m7_theory", "common.py"))
m7 = importlib.util.module_from_spec(_spec)
sys.modules["m7common"] = m7
_spec.loader.exec_module(m7)          # read-only import of the m7 helpers
LN6 = np.log(6.0)
RESULTS = []


def record(cid, claim, ok, metric):
    RESULTS.append(dict(check=cid, claim=claim, passed=bool(ok), metric=metric))
    print(("[PASS] " if ok else "[FAIL] ") + f"{cid:<14} {claim}  ({metric})")


# ----------------------------------------------------------------------------------------------- technologies
def ref_params():
    p = json.load(open(os.path.join(ROOT, "data", "processed", "ra2_wedge", "headline.json")))["chin_q_point"]
    return dict(A=float(np.exp(p["lnA"])), B=float(np.exp(p["lnB"])), a1=float(p["alpha"]), b1=float(p["beta"]),
                kappa=float(p["q"]))


def y_family(pr, psiN=0.0, psiD=0.0):
    """log output index y = -ln R of the generalized family, with (possibly c-dependent) factor-augmenting psi."""
    def y(n, d):
        c = LN6 + n + d
        pN = psiN(c) if callable(psiN) else psiN
        pD = psiD(c) if callable(psiD) else psiD
        return -m7.lnR(n + pN, d + pD, pr["A"], pr["B"], pr["a1"], pr["b1"], pr["kappa"])
    return y


def y_nonsep(n, d):
    return 0.7 * n + 0.4 * d - 0.05 * n ** 2 + 0.02 * n * d - 0.03 * d ** 2 + 0.1 * np.exp(-0.2 * n + 0.1 * d)


def tilt(y, t):
    """pure-bias factor-augmenting tilt psi_N = t/2, psi_D = -t/2 (compute unchanged)."""
    return lambda n, d: y(n + t / 2, d - t / 2)


def Yxc(y, x, c):
    n, d = (c - LN6 - x) / 2, (c - LN6 + x) / 2
    return y(n, d)


def argmin_x(y, c, lo=-8.0, hi=14.0):
    r = minimize_scalar(lambda x: -Yxc(y, x, c), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
    return r.x


def mp_and_peaks(y, c, xs, h=1e-5):
    """min marginal products over the profile and the number of sign changes of Y_x."""
    n, d = (c - LN6 - xs) / 2, (c - LN6 + xs) / 2
    Fn = (y(n + h, d) - y(n - h, d)) / (2 * h)
    Fd = (y(n, d + h) - y(n, d - h)) / (2 * h)
    Yx = (Yxc(y, xs + h, c) - Yxc(y, xs - h, c)) / (2 * h)
    sg = np.sign(Yx)
    sg = sg[sg != 0]                               # a grid point exactly at the peak is not a second sign change
    return min(Fn.min(), Fd.min()), int(np.sum(np.diff(sg) != 0))


def wedge(y, n, d, h=1e-5):
    Fn = (y(n + h, d) - y(n - h, d)) / (2 * h)
    Fd = (y(n, d + h) - y(n, d - h)) / (2 * h)
    return Fn / Fd


# ----------------------------------------------------------------------------------------------- (vii): tilt
def check_tilt():
    pr = ref_params()
    S = pr["a1"] + pr["b1"]
    a = pr["b1"] / S
    cJ = np.log(3e21)
    # (b) the A-pi(ii) construction: psi_N' = Delta'_+/[2(1-a)] beyond c_J (path slope raised by e_target - e_true)
    e_true = 1 - 2 * a
    e_target = e_true + 0.30
    dN = (e_target - e_true) / (2 * (1 - a))
    psiN = lambda c: dN * np.maximum(c - cJ, 0.0)
    techs = {"reference": y_family(pr), "(ii) construction": y_family(pr, psiN=psiN), "non-separable": y_nonsep}
    rows = []
    for label, y in techs.items():
        for t in (-0.61, 0.35, 1.20):
            yt = tilt(y, t)
            shifts, minmp, peaks = [], [], []
            cs = [np.log(1e19), np.log(1e21), np.log(1e23), np.log(1e25)] if label != "non-separable" else [2.0, 3.0, 4.0]
            for c in cs:
                if label == "non-separable":
                    x0 = argmin_x(y, c, -6, 6)
                    x1 = argmin_x(yt, c, -6, 6)
                    xs = np.linspace(x1 - 1.5, x1 + 1.5, 301)
                else:
                    x0 = argmin_x(y, c)
                    x1 = argmin_x(yt, c)
                    xs = np.linspace(x1 - 6, x1 + 6, 601)
                shifts.append(x1 - x0)
                mp, pk = mp_and_peaks(yt, c, xs)
                minmp.append(mp)
                peaks.append(pk)
            ok = np.max(np.abs(np.array(shifts) - t)) < 1e-6 and min(minmp) > 0 and all(p == 1 for p in peaks)
            extra = ""
            if label == "reference":
                chi = pr["b1"] * (-t / 2) - pr["a1"] * (t / 2)
                ok = ok and abs(-2 * chi / S - t) < 1e-12
                extra = f"; -2chi/S={-2 * chi / S:.4f}"
            rows.append(dict(tech=label, t=t, max_shift_err=np.max(np.abs(np.array(shifts) - t)), min_mp=min(minmp)))
            record(f"pi.tilt.{label[:6]}", f"{label}: tilt psi_N=-psi_D=t/2 with t={t}: argmin moves by t at every compute; "
                   "marginal products > 0; single-peaked", ok,
                   f"max|shift-t|={np.max(np.abs(np.array(shifts) - t)):.1e} min MP={min(minmp):.2e} peaks={peaks}{extra}")
    # sharpness: point anchor, generalized family; a target point y_target in X*_tau(c) at c = ln(1e25)
    c = np.log(1e25)
    xJ = argmin_x(techs["reference"], cJ)
    eL, eU, tau = -0.2, 0.4, 1.84
    lo, hi = xJ + eL * (c - cJ) - np.log(tau), xJ + eU * (c - cJ) + np.log(tau)
    ok_all = True
    for frac in (0.0, 0.13, 0.5, 0.87, 1.0):
        target = lo + frac * (hi - lo)
        # split target = y0 + t with y0 in the tau=1 set and |t| <= ln tau
        y0 = np.clip(target, xJ + eL * (c - cJ), xJ + eU * (c - cJ))
        t = target - y0
        e_t = (y0 - xJ) / (c - cJ)
        # (ii) construction to the path slope e_t beyond c_J
        de = e_t - e_true
        if de >= 0:
            anch = y_family(pr, psiN=lambda cc, k=de / (2 * (1 - a)): k * np.maximum(cc - cJ, 0.0))
        else:
            anch = y_family(pr, psiD=lambda cc, k=-de / (2 * a): k * np.maximum(cc - cJ, 0.0))
        dev = tilt(anch, t)
        # anchored technology reproduces the reference up to c_J
        same = max(abs(argmin_x(anch, cc) - argmin_x(techs["reference"], cc)) for cc in (np.log(1e19), np.log(1e20), cJ))
        within = max(abs(argmin_x(dev, cc) - argmin_x(anch, cc)) for cc in (np.log(1e19), cJ, np.log(1e23), c))
        xdev = argmin_x(dev, c)
        xs = np.linspace(xdev - 6, xdev + 6, 601)
        mp, pk = mp_and_peaks(dev, c, xs)
        ok = abs(xdev - target) < 1e-6 and same < 1e-6 and within <= np.log(tau) + 1e-6 and mp > 0 and pk == 1
        ok_all &= ok
        record("pi.sharp", f"point {frac:.2f} of X*_tau(c) at 1e25 is x*_dev of an admissible developer technology",
               ok, f"target={target:.4f} x*_dev={xdev:.4f} anchor reproduced to {same:.1e}, |tilt|={within:.3f} <= ln tau={np.log(tau):.3f}")
    return rows


# ----------------------------------------------------------------------------------------------- (vii): containment
def check_containment(ndraw=20000, seed=7):
    rng = np.random.default_rng(seed)
    pr = ref_params()
    S = pr["a1"] + pr["b1"]
    a = pr["b1"] / S
    theta = S / 2                                   # = 1/sigma* - 1 in the generalized family
    fails_set = fails_sign = fails_mag = 0
    cJ = np.log(3e21)
    for _ in range(ndraw):
        # anchored path: random point in the anchor interval, piecewise-constant slopes within [eL, eU]
        x_lo = rng.uniform(2.0, 3.5)
        x_hi = x_lo + rng.uniform(0, 0.6)
        eL = rng.uniform(-0.3, 0.1)
        eU = eL + rng.uniform(0, 0.5)
        tau = np.exp(rng.uniform(0, 1.5))
        c = cJ + rng.uniform(0.1, 10.0)
        k = rng.integers(1, 5)
        knots = np.sort(rng.uniform(cJ, c, k - 1))
        segs = np.diff(np.r_[cJ, knots, c])
        xstar = rng.uniform(x_lo, x_hi) + np.sum(segs * rng.uniform(eL, eU, k))
        t = rng.uniform(-np.log(tau), np.log(tau))
        xdev = xstar + t
        lo = x_lo + eL * (c - cJ) - np.log(tau)
        hi = x_hi + eU * (c - cJ) + np.log(tau)
        if not (lo - 1e-12 <= xdev <= hi + 1e-12):
            fails_set += 1
        # the developer's technology: generalized family with its argmin at xdev at compute c (shift by constant tilt)
        x_ref = -2 * np.log((pr["a1"] * pr["A"] / (pr["b1"] * pr["B"])) ** (1 / S)) + (1 - 2 * a) * (c - LN6)
        ydev = tilt(y_family(pr), xdev - x_ref)
        x = rng.uniform(lo - 4, hi + 4)
        n, d = (c - LN6 - x) / 2, (c - LN6 + x) / 2
        w = wedge(ydev, n, d)
        if np.sign(w - 1) != np.sign(x - xdev) and abs(x - xdev) > 1e-6:
            fails_sign += 1
        # (iv): magnitude bounds with [theta_L, theta_U] containing theta
        tL, tU = theta * rng.uniform(0.7, 1.0), theta * rng.uniform(1.0, 1.3)
        lw = np.log(w)
        if x > hi:
            bnd = (tL * (x - hi), tU * (x - lo))
        elif x < lo:
            bnd = (tU * (x - hi), tL * (x - lo))
        else:
            bnd = (tU * (x - hi), tU * (x - lo))
        if not (bnd[0] - 1e-7 <= lw <= bnd[1] + 1e-7):
            fails_mag += 1
    record("pi.contain", f"{ndraw} random draws: x*_dev in X*_tau(c); sign(w-1) = sign(x - x*_dev); (iv) magnitude bounds",
           fails_set == 0 and fails_sign == 0 and fails_mag == 0,
           f"set violations {fails_set}, sign violations {fails_sign}, magnitude violations {fails_mag}")


# ----------------------------------------------------------------------------------------------- (vii)-(viii): bands
def check_bands(R=4000, seed=11):
    rng = np.random.default_rng(seed)
    p = 0.05
    crit = np.sqrt(chi2.ppf(1 - p, 2))            # Working-Hotelling: simultaneous over all c for a straight line
    # three anchored paths (level, slope, budgets, noise s.d. of the IsoFLOP minima)
    paths = [dict(a=2.9, e=0.05, cb=np.log([6e18, 1e19, 3e19, 1e20, 3e20, 1e21, 3e21]), sd=0.12),
             dict(a=3.3, e=0.10, cb=np.log([6e18, 1e19, 3e19, 1e20, 3e20, 1e21]), sd=0.08),
             dict(a=3.8, e=-0.02, cb=np.log([1e18, 3e18, 1e19, 3e19, 1e20, 3e20]), sd=0.10)]
    cgrid = np.log(np.logspace(21.5, 26, 40))
    tau = 1.84
    jstar = 1                                      # the developer lies within tau of path 2
    tdev = lambda c: np.log(tau) * np.sign(np.sin(3 * c) + 1e-9)   # the worst case: |t| = ln tau, sign switching
    cover_path = cover_j = cover_union = 0
    for _ in range(R):
        bands = []
        for P in paths:
            cJ = P["cb"].max()
            X = np.c_[np.ones_like(P["cb"]), P["cb"] - cJ]
            ytrue = P["a"] + P["e"] * (P["cb"] - cJ)
            yb = ytrue + P["sd"] * rng.standard_normal(len(ytrue))
            beta, *_ = np.linalg.lstsq(X, yb, rcond=None)
            V = P["sd"] ** 2 * np.linalg.inv(X.T @ X)
            Xg = np.c_[np.ones_like(cgrid), cgrid - cJ]
            fit = Xg @ beta
            se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, V, Xg))
            bands.append((fit - crit * se, fit + crit * se))
        P = paths[jstar]
        xpath = P["a"] + P["e"] * (cgrid - P["cb"].max())
        xdev = xpath + tdev(cgrid)
        lo, hi = bands[jstar]
        cover_path += np.all((lo <= xpath) & (xpath <= hi))
        cover_j += np.all((lo - np.log(tau) <= xdev) & (xdev <= hi + np.log(tau)))
        inunion = np.zeros_like(cgrid, dtype=bool)
        for lo_, hi_ in bands:
            inunion |= (lo_ - np.log(tau) <= xdev) & (xdev <= hi_ + np.log(tau))
        cover_union += np.all(inunion)
    cp, cj, cu = cover_path / R, cover_j / R, cover_union / R
    mcse = np.sqrt(0.95 * 0.05 / R)
    record("pi.band", "simultaneous 95% band (Working-Hotelling, known noise) covers its path; widened by ln tau it covers "
           "x*_dev(c) at every c for a worst-case tilt |t| = ln tau, and so does the widened union (>= 0.95 - 2 MC s.e.)",
           cp >= 0.95 - 2 * mcse and cj >= cp - 1e-12 and cu >= cj - 1e-12,
           f"coverage: path by its band {cp:.4f}; x*_dev by the widened band {cj:.4f}; by the widened union {cu:.4f} "
           f"(MC s.e. {mcse:.4f}, R={R})")


# ----------------------------------------------------------------------------------------------- Corollary A2
def check_corA2():
    al, be = 0.3392, 0.2849                        # Hoffmann et al. (TeX precision)
    rho = (al + be) / 2
    rows = []
    ok = True
    for M_ratio in (np.exp(np.log(2.198) / rho), np.exp(np.log(5.454) / rho), np.exp(np.log(13.5) / rho), 1.2, 0.5):
        lw = rho * np.log(M_ratio)
        w = np.exp(lw)
        ce_exact = m7.ce_ratio_from_w(w, al, be)
        ce_cosh = np.cosh(rho / 2 * np.log(M_ratio)) ** (2 / rho)
        rows.append((w, ce_exact, ce_cosh, (np.log(ce_exact) - np.log(ce_cosh)) / abs(lw) ** 3))
    # second-order agreement: ln C/Cmin = (ln w)^2 / (2S) for both, S = alpha + beta
    S = al + be
    for lw in (0.01, 0.003):
        w = np.exp(lw)
        e = np.log(m7.ce_ratio_from_w(w, al, be))
        cs = (2 / rho) * np.log(np.cosh(lw / 2))
        ok &= abs(e / (lw ** 2 / (2 * S)) - 1) < 5 * lw and abs(cs / (lw ** 2 / (2 * S)) - 1) < 5 * lw
    big = rows[2]
    ok &= abs(big[1] - 101.2) < 0.05 and abs(big[2] - 78.0) < 0.05
    record("corA2", "unequal exponents: cosh form agrees with Lemma A5 only to second order in ln w",
           ok, "; ".join(f"w={r[0]:.3f}: A5 {r[1]:.3f} vs cosh {r[2]:.3f}" for r in rows[:3]))


# ----------------------------------------------------------------------------------------------- Definition A2: K
def check_K():
    pr = ref_params()
    A, B, a1, b1, kap = pr["A"], pr["B"], pr["a1"], pr["b1"], pr["kappa"]
    S = a1 + b1
    G = (a1 * A / (b1 * B)) ** (1 / S)
    K = ((S / b1) * A * G ** (-a1)) ** kap
    gam = kap * a1 * b1 / S
    errs = []
    for C in (1e19, 1e21, 1e23, 1e25):
        c = np.log(C)
        r = minimize_scalar(lambda n: m7.lnR(n, c - LN6 - n, A, B, a1, b1, kap), bounds=(10, 40), method="bounded",
                            options={"xatol": 1e-12})
        lnRstar = r.fun
        errs.append((abs(lnRstar - (np.log(K) - gam * (c - LN6))), abs(r.x - (np.log(G) + (b1 / S) * (c - LN6)))))
    eR, en = max(e[0] for e in errs), max(e[1] for e in errs)
    record("defA2.K", "generalized family: R*(C) = K (C/6)^(-gamma), K = ((S/b1) A G^(-a1))^kappa; path n* = ln G + a(c - ln 6)",
           eR < 1e-10 and en < 1e-6, f"max abs error: ln R* {eR:.1e}, n* {en:.1e} (kappa = {kap:.4f})")


# ----------------------------------------------------------------------------------------------- A2(iii)
def check_info():
    B_ = m7.PARAM_SETS["Besiroglu"]
    po = m7.path_objects(B_["A"], B_["B"], B_["alpha"], B_["beta"])
    phi0 = np.array([B_["alpha"] + B_["beta"], po["a"], po["gamma"], np.log(po["G"]), np.log(po["K"])])
    lcs = np.log(np.logspace(18, 22, 9) / 6)
    grid_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    def design(v):
        rows = []
        for lc in lcs:
            ns = phi0[3] + phi0[1] * lc
            for x in v * grid_x:
                rows.append((ns - x / 2, lc - ns + x / 2))
        return np.array(rows)

    def y(phi, X):
        S, a, gam, lG, lK = phi
        m = m7.kappa_family_member(S, a, gam, np.exp(lG), np.exp(lK))
        return -m7.lnR(X[:, 0], X[:, 1], m["A"], m["B"], m["a1"], m["b1"], m["kappa"])

    def jac(X):
        J = np.zeros((len(X), 5))
        for k in range(5):
            h = 1e-6 * max(1.0, abs(phi0[k]))
            pp, pm = phi0.copy(), phi0.copy()
            pp[k] += h
            pm[k] -= h
            J[:, k] = (y(pp, X) - y(pm, X)) / (2 * h)
        return J

    def eff(I, k):
        o = [j for j in range(I.shape[0]) if j != k]
        return I[k, k] - I[k, o] @ np.linalg.solve(I[np.ix_(o, o)], I[o, k])

    vs = np.logspace(-1.6, -0.6, 5)
    IS, IG, IM = [], [], []
    c0 = np.log(1e21 / 6)
    g = np.zeros(5)
    g[1], g[3] = -2 * c0, -2.0                     # ln M*(C0) = -2 ln G + (1 - 2a) ln(C0/6)
    for v in vs:
        J = jac(design(v))
        I = J.T @ J
        IS.append(eff(I, 0))
        IG.append(eff(I, 3))
        IM.append(1.0 / float(g @ np.linalg.solve(I, g)))
    sS = np.polyfit(np.log(vs), np.log(IS), 1)[0]
    sG = np.polyfit(np.log(vs), np.log(IG), 1)[0]
    sM = np.polyfit(np.log(vs), np.log(IM), 1)[0]
    ratio = np.array(IM) / (np.array(IG) / 4)
    record("A2.iii", "information for ln M*(C0) is O(varsigma^2) like ln G's, and for S O(varsigma^4)",
           abs(sS - 4) < 0.05 and abs(sG - 2) < 0.05 and abs(sM - 2) < 0.05,
           f"slopes: S {sS:.3f}, ln G {sG:.3f}, ln M*(1e21) {sM:.3f}; I(lnM*)/[I(lnG)/4] = {ratio.min():.0f}-{ratio.max():.0f}")


if __name__ == "__main__":
    check_tilt()
    check_containment()
    check_bands()
    check_corA2()
    check_K()
    check_info()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "prop_pi_checks.csv")
    pd.DataFrame(RESULTS).to_csv(out, index=False)
    nfail = sum(not r["passed"] for r in RESULTS)
    print(f"{len(RESULTS) - nfail}/{len(RESULTS)} checks passed; wrote {out}")
    sys.exit(1 if nfail else 0)
