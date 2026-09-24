"""profile.py -- kappa-family fits in normalized coordinates with analytic gradients, for profile likelihoods over
sigma* (fixed S = a1 + b1 = 2/sigma* - 2) and over the asymptote E (R1 c9(c), R4 M5, R2 Major 8).

Model: ln Lhat = LSE( kappa * LSE(lA - a1 x~, lB - b1 z~), lE ), x~ = ln N - mean(ln N), z~ = ln D - mean(ln D)
(Klump-McAdam-Willman normalization keeps the level parameters O(1) even at extreme sigma*).
Objectives: 'huber' = sum Huber_{1e-3}(r) (the reference estimator; LAD in practice), 'gauss' = 0.5 sum r^2.
LR statistics relative to the UNCONSTRAINED optimum:
  Gaussian              n ln(SSR_r / SSR_u)
  Laplace quasi-LR      2 n ln(H_r / H_u)                 (concentrated Laplace likelihood; H = Huber objective)
  Koenker-Bassett LR    4 f(0) (S1_r - S1_u)              (S1 = sum |r|; f(0) kernel density of the unrestricted
                                                           residuals at zero; median-regression LR with estimated
                                                           sparsity, Koenker and Bassett 1982)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber as _huber

DELTA = 1e-3


def _parts(q, x, z, mode, S=None, E=None):
    """Unpack q into (lA, lB, lE, a1, b1, k) and d(full)/d(q) mapping."""
    if mode == "S":          # q = (lA, lB, lE, sh, lk)
        lA, lB, lE, sh, lk = q
        return lA, lB, lE, S * sh, S * (1 - sh), np.exp(lk)
    if mode == "E":          # q = (lA, lB, a1, b1, lk); E fixed
        lA, lB, a1, b1, lk = q
        return lA, lB, np.log(E), a1, b1, np.exp(lk)
    lA, lB, lE, a1, b1, lk = q   # free
    return lA, lB, lE, a1, b1, np.exp(lk)


def obj_grad(q, x, z, y, kind, mode, S=None, E=None):
    lA, lB, lE, a1, b1, k = _parts(q, x, z, mode, S, E)
    t1, t2 = lA - a1 * x, lB - b1 * z
    m12 = np.maximum(t1, t2)
    s = m12 + np.log(np.exp(t1 - m12) + np.exp(t2 - m12))
    p1 = np.exp(t1 - s)
    p2 = 1.0 - p1
    T = k * s
    mm = np.maximum(T, lE)
    pred = mm + np.log(np.exp(T - mm) + np.exp(lE - mm))
    wt = np.exp(T - pred)
    we = 1.0 - wt
    r = pred - y
    if kind == "gauss":
        f, psi = 0.5 * float(np.sum(r * r)), r
    else:
        f, psi = float(np.sum(_huber(DELTA, r))), np.clip(r, -DELTA, DELTA)
    g_lA = np.sum(psi * wt * k * p1)
    g_lB = np.sum(psi * wt * k * p2)
    g_lE = np.sum(psi * we)
    g_a1 = np.sum(psi * wt * k * p1 * -x)
    g_b1 = np.sum(psi * wt * k * p2 * -z)
    g_lk = np.sum(psi * wt * s) * k
    if mode == "S":
        g = np.array([g_lA, g_lB, g_lE, S * g_a1 - S * g_b1, g_lk])
    elif mode == "E":
        g = np.array([g_lA, g_lB, g_a1, g_b1, g_lk])
    else:
        g = np.array([g_lA, g_lB, g_lE, g_a1, g_b1, g_lk])
    return f, g


def predict(full, x, z):
    lA, lB, lE, a1, b1, lk = full
    return np.logaddexp(np.exp(lk) * np.logaddexp(lA - a1 * x, lB - b1 * z), lE)


class KappaData:
    def __init__(self, N, D, L):
        self.ln, self.ld = np.log(np.asarray(N, float)), np.log(np.asarray(D, float))
        self.xm, self.zm = self.ln.mean(), self.ld.mean()
        self.x, self.z = self.ln - self.xm, self.ld - self.zm
        self.y = np.log(np.asarray(L, float))
        self.n = len(self.y)
        self.lEmax = float(np.log(np.min(L))) - 1e-6

    def from_raw(self, p):
        """m2 fit_q parameters (lnA, lnB, lnE, a1, b1, q) in raw units -> normalized full vector."""
        lnA, lnB, lnE, a1, b1, q = p
        return np.array([lnA - a1 * self.xm, lnB - b1 * self.zm, lnE, a1, b1, np.log(q)])

    def to_raw(self, full):
        lA, lB, lE, a1, b1, lk = full
        return np.array([lA + a1 * self.xm, lB + b1 * self.zm, lE, a1, b1, np.exp(lk)])


def _min(fun, q0, bounds, args):
    r = minimize(fun, q0, args=args, jac=True, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=20000, ftol=1e-15, gtol=1e-11))
    return r.x, r.fun


def fit_free(kd, starts, kind="huber"):
    bnds = [(-60, 60), (-60, 60), (-12, kd.lEmax), (1e-4, 60), (1e-4, 60), (np.log(1e-3), np.log(300))]
    best = None
    for st in starts:
        st = np.array(st, float)
        st[2] = min(st[2], kd.lEmax)
        q, f = _min(obj_grad, st, bnds, (kd.x, kd.z, kd.y, kind, "free"))
        if best is None or f < best[1]:
            best = (q, f)
    return best


def start_at_S(full_ref, S, kd):
    """Map a kappa-family point to a start with a1 + b1 = S that preserves the path slope a, the on-path frontier
    exponent kappa a1 b1/(a1+b1) and the loss level at the design centre."""
    lA, lB, lE, a1, b1, lk = full_ref
    k = np.exp(lk)
    Su = a1 + b1
    sh = a1 / Su
    k0 = k * Su / S
    lg = np.logaddexp(lA, lB)
    p1 = np.exp(lA - lg)
    lg0 = k * lg / k0
    return np.array([lg0 + np.log(max(p1, 1e-12)), lg0 + np.log(max(1 - p1, 1e-12)), min(lE, kd.lEmax),
                     min(max(sh, 0.011), 0.989), np.log(min(max(k0, 1.1e-3), 290))])


def fit_S(kd, S, starts, kind="huber"):
    bnds = [(-60, 60), (-60, 60), (-12, kd.lEmax), (0.01, 0.99), (np.log(1e-3), np.log(300))]
    best = None
    for st in starts:
        q, f = _min(obj_grad, np.asarray(st, float), bnds, (kd.x, kd.z, kd.y, kind, "S", S))
        if best is None or f < best[1]:
            best = (q, f)
    q = best[0]
    full = np.array([q[0], q[1], q[2], S * q[3], S * (1 - q[3]), q[4]])
    return full, best[1], q


def fit_E(kd, E, starts, kind="huber"):
    bnds = [(-60, 60), (-60, 60), (1e-4, 60), (1e-4, 60), (np.log(1e-3), np.log(300))]
    best = None
    for st in starts:
        st = np.asarray(st, float)
        q0 = np.array([st[0], st[1], st[3], st[4], st[5]])
        q, f = _min(obj_grad, q0, bnds, (kd.x, kd.z, kd.y, kind, "E", None, E))
        if best is None or f < best[1]:
            best = (q, f)
    q = best[0]
    return np.array([q[0], q[1], np.log(E), q[2], q[3], q[4]]), best[1]


def l1(full, kd):
    return float(np.sum(np.abs(predict(full, kd.x, kd.z) - kd.y)))


def f0_kernel(r):
    n = len(r)
    s = min(np.std(r, ddof=1), (np.percentile(r, 75) - np.percentile(r, 25)) / 1.349)
    bw = 0.9 * s * n ** (-0.2)
    return float(np.mean(np.exp(-0.5 * (r / bw) ** 2)) / (bw * np.sqrt(2 * np.pi)))


def profile_sigma(kd, grid, full_u, kind="huber"):
    """Profile objective over sigma* (forward and backward warm-started sweeps). Returns list of dicts."""
    out = []
    prev = None
    for s_ in grid:
        S = 2.0 / s_ - 2.0
        starts = [start_at_S(full_u, S, kd)] + ([prev] if prev is not None else [])
        full, f, q = fit_S(kd, S, starts, kind)
        prev = q
        out.append(dict(sigma_star=float(s_), S=S, obj=f, full=full, q=q))
    for i in range(len(grid) - 2, -1, -1):
        full, f, q = fit_S(kd, out[i]["S"], [out[i + 1]["q"]], kind)
        if f < out[i]["obj"] - 1e-14:
            out[i].update(obj=f, full=full, q=q)
    return out


def conf_set(sig, LR, crit=3.841458820694124):
    sig, LR = np.asarray(sig), np.asarray(LR)
    inset = LR <= crit
    if not inset.any():
        return dict(lo=np.nan, hi=np.nan, pieces=0, hits_lo=False, hits_hi=False)
    idx = np.where(inset)[0]
    pieces = int(1 + np.sum(np.diff(idx) > 1))
    return dict(lo=float(sig[idx].min()), hi=float(sig[idx].max()), pieces=pieces, hits_lo=bool(inset[0]),
                hits_hi=bool(inset[-1]))
