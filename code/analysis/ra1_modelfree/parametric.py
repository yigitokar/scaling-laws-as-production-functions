"""parametric.py -- parametric technologies (Chinchilla form, kappa family, Farseer Eq. 3) fitted with the machinery of
modules m1/m2 (sl.fit_chinchilla, m2_est.fit_q, m2_est.fit_farseer), plus wild-bootstrap helpers for LAD-type
(Huber, delta = 1e-3) fits.

kappa family: L = E + (A N^-a1 + B D^-b1)^kappa (kappa = 1: Chinchilla). sigma*_kappa = 2/(2 + a1 + b1); the wedge
w = a1 u'/(b1 v') uses the inner aggregator (kappa cancels).
"""
from __future__ import annotations

import numpy as np

import ra1_common as rc

sl = rc.sl


def fit_chin(N, D, L, est="huber", starts=None, grid="fast"):
    """Chinchilla form on log loss (Huber delta=1e-3 or Gaussian). Returns theta (lnA, lnB, lnE, alpha, beta), obj."""
    import m2_est as me
    delta = 1e-3 if est == "huber" else None
    best = None
    if grid is not None:
        g = sl.FAST_GRID if grid == "fast" else sl.DEFAULT_GRID
        m = sl.fit_chinchilla(N, D, L, delta=delta, grid=g)
        best = (m.theta, m.extra["objective"])
    cands = list(starts or []) + (me.level_starts(N, D, L) if grid is not None else [])
    for st in cands:
        st = np.asarray(st, float).copy()
        st[2] = max(st[2], -30.0)
        m = sl.fit_chinchilla(N, D, L, delta=delta, init=st)
        if np.all(np.isfinite(m.theta)) and (best is None or m.extra["objective"] < best[1]):
            best = (m.theta, m.extra["objective"])
    return best


def fit_kappa(N, D, L, est="huber", th_chin=None, init=None):
    """kappa-free fit (m2_est.fit_q; analytic gradient, starts at q in {1, .5, .75, 1.5, 2, 3} from th_chin)."""
    import m2_est as me
    return me.fit_q(N, D, L, est=est, th_chin=th_chin, init=init)


def sigma_star_chin(th):
    return 2.0 / (2.0 + th[3] + th[4])


def sigma_star_kappa(p):
    return 2.0 / (2.0 + p[3] + p[4])


def wedge_chin(th, N, D):
    lnA, lnB, lnE, al, be = th
    return np.exp(np.log(al) + lnA - al * np.log(N) - (np.log(be) + lnB - be * np.log(D)))


def wedge_kappa(p, N, D):
    lnA, lnB, lnE, a1, b1, k = p
    return np.exp(np.log(a1) + lnA - a1 * np.log(N) - (np.log(b1) + lnB - b1 * np.log(D)))


def pred_chin(th, N, D):
    lnA, lnB, lnE, al, be = th
    return np.logaddexp(np.logaddexp(lnA - al * np.log(N), lnB - be * np.log(D)), lnE)


def pred_kappa(p, N, D):
    lnA, lnB, lnE, a1, b1, k = p
    return np.logaddexp(k * np.logaddexp(lnA - a1 * np.log(N), lnB - b1 * np.log(D)), lnE)


def jac_log_chin(th, N, D):
    """n x 5 Jacobian of ln Lhat w.r.t. theta (for leverages)."""
    lN, lD = np.log(N), np.log(D)
    a, b, e, al, be = th
    t = np.column_stack([a - al * lN, b - be * lD, np.full_like(lN, e)])
    p = np.exp(t - t.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.column_stack([p[:, 0], p[:, 1], p[:, 2], -p[:, 0] * lN, -p[:, 1] * lD])


def fhh_residuals(r, J):
    """Feng-He-Hu (2011) residual adjustment for tau = 0.5 (quantreg::boot.rq 'wild'):
    r~ = r + h_ii (0.5 - 1{r<0}) / f(0), h_ii from the Jacobian hat matrix, f(0) a Gaussian-kernel density of the
    residuals at zero (Silverman bandwidth)."""
    Q, _ = np.linalg.qr(J)
    h = np.sum(Q * Q, axis=1)
    n = len(r)
    s = min(np.std(r, ddof=1), (np.percentile(r, 75) - np.percentile(r, 25)) / 1.349)
    bw = 0.9 * s * n ** (-0.2)
    f0 = np.mean(np.exp(-0.5 * (r / bw) ** 2)) / (bw * np.sqrt(2 * np.pi))
    return r + h * (0.5 - (r < 0)) / f0, h, f0
