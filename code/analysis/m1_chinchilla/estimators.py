"""estimators.py -- the estimator horse race for L = E + A N^-alpha + B D^-beta.

All estimators return an sl.Chinchilla with .extra holding the objective value and diagnostics.
Parameter vector theta = (ln A, ln B, ln E, alpha, beta) (the log-sum-exp parameterisation of Hoffmann et al.).

  huber      Hoffmann/Besiroglu: sum Huber_delta(ln Lhat - ln L), delta = 1e-3 (sl.fit_chinchilla)
  gauss_log  Gaussian NLS on log loss (delta = None in sl.fit_chinchilla)
  lad_log    LAD on log loss: Huber with delta = 1e-6 as a smooth start, then Nelder-Mead on the exact L1 objective
  nls_lev    NLS in levels, sum (L - Lhat)^2, direct L-BFGS-B on theta from a grid of starts
  vpnls      variable projection in levels (Golub-Pereyra): given (alpha, beta) the model is linear in (E, A, B), which
             are concentrated out by non-negative least squares; the 2-D profile is searched on a grid and polished.
             Same objective as nls_lev -- a check on whether direct 5-D optimisation finds the global minimum.
  norm       Klump-McAdam-Willman normalisation: inputs divided by their sample geometric means, so the level
             parameters are a' = A Nbar^-alpha and b' = B Dbar^-beta (reducible-loss terms at the geometric-mean
             input bundle). Same Huber(1e-3) objective as `huber`; a reparameterisation that changes conditioning,
             not the estimand.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize, nnls

from common import sl, from_theta, theta_of

HUBER_DELTA = 1e-3


# ----------------------------------------------------------------------------- helpers
def lse_pred(th, lN, lD):
    return sl._lse_pred(np.asarray(th, float), lN, lD)


def _logs(N, D, L):
    return np.log(np.asarray(N, float)), np.log(np.asarray(D, float)), np.log(np.asarray(L, float))


def huber_obj(th, N, D, L, delta=HUBER_DELTA):
    lN, lD, lL = _logs(N, D, L)
    return float(sl._obj(np.asarray(th, float), lN, lD, lL, delta, np.ones_like(lL)))


def l1_obj(th, lN, lD, lL):
    return float(np.sum(np.abs(lse_pred(th, lN, lD) - lL)))


# ----------------------------------------------------------------------------- log-space M-estimators
def fit_huber(N, D, L, inits=None, grid=None, delta=HUBER_DELTA):
    """Huber-LSE. With `inits` (list of theta) only those starts are used (warm starts for the bootstrap)."""
    if inits is None:
        m = sl.fit_chinchilla(N, D, L, delta=delta, grid=grid)
    else:
        best = None
        for th in inits:
            mm = sl.fit_chinchilla(N, D, L, delta=delta, init=np.asarray(th, float))
            if best is None or mm.extra["objective"] < best.extra["objective"]:
                best = mm
        m = best
    return m


def fit_gauss_log(N, D, L, inits=None, grid=None):
    return fit_huber(N, D, L, inits=inits, grid=grid, delta=None)


def fit_lad(N, D, L, inits=None, grid=None):
    """LAD on log loss. The L1 objective is not differentiable, so: (i) Huber(1e-6) from each start (smooth
    approximation, L-BFGS-B); (ii) Nelder-Mead polish on the exact L1 objective."""
    lN, lD, lL = _logs(N, D, L)
    starts = inits if inits is not None else [theta_of(fit_huber(N, D, L, grid=grid))]
    best = None
    for th0 in starts:
        m0 = sl.fit_chinchilla(N, D, L, delta=1e-6, init=np.asarray(th0, float))
        x = m0.theta
        for _ in range(3):  # restarted Nelder-Mead (standard remedy for premature simplex collapse)
            r = minimize(l1_obj, x, args=(lN, lD, lL), method="Nelder-Mead",
                         options=dict(maxiter=20000, maxfev=40000, xatol=1e-9, fatol=1e-12, adaptive=True))
            x = r.x
        f = l1_obj(x, lN, lD, lL)
        if best is None or f < best[1]:
            best = (x, f)
    return from_theta(best[0], objective=best[1], n=len(lL), estimator="lad_log")


# ----------------------------------------------------------------------------- levels
def _levels_obj(th, lN, lD, L):
    r = np.exp(lse_pred(th, lN, lD)) - L
    return 0.5 * float(np.sum(r * r))


def _levels_grad(th, lN, lD, L):
    a, b, e, al, be = th
    t1, t2, t3 = a - al * lN, b - be * lD, e * np.ones_like(lN)
    m = np.maximum(np.maximum(t1, t2), t3)
    p1, p2, p3 = np.exp(t1 - m), np.exp(t2 - m), np.exp(t3 - m)
    s = p1 + p2 + p3
    Lhat = np.exp(m) * s
    p1, p2, p3 = p1 / s, p2 / s, p3 / s
    g = (Lhat - L) * Lhat
    return np.array([np.sum(g * p1), np.sum(g * p2), np.sum(g * p3), np.sum(g * p1 * -lN), np.sum(g * p2 * -lD)])


def fit_nls_levels(N, D, L, inits=None, grid=None):
    lN, lD, _ = _logs(N, D, L)
    L = np.asarray(L, float)
    if inits is None:
        grid = grid or sl.DEFAULT_GRID
        inits = list(itertools.product(grid["a"], grid["b"], grid["e"], grid["alpha"], grid["beta"]))
    best = None
    for st in inits:
        r = minimize(_levels_obj, np.asarray(st, float), args=(lN, lD, L), jac=_levels_grad, method="L-BFGS-B",
                     options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
        if np.all(np.isfinite(r.x)) and (best is None or r.fun < best[1]):
            best = (r.x, r.fun)
    return from_theta(best[0], objective=float(best[1]), n=len(L), estimator="nls_lev")


def _vp_inner(al, be, N, D, L):
    """Concentrate out (E, A, B) >= 0 given the exponents: NNLS of L on [1, N^-al, D^-be] (columns scaled)."""
    X = np.column_stack([np.ones_like(L), N ** -al, D ** -be])
    sc = np.linalg.norm(X, axis=0)
    c, rnorm = nnls(X / sc, L)
    return c / sc, 0.5 * rnorm ** 2


def fit_vpnls(N, D, L, inits=None, grid_step=0.01, lo=0.02, hi=1.5):
    N, D, L = (np.asarray(v, float) for v in (N, D, L))
    f = lambda x: _vp_inner(x[0], x[1], N, D, L)[1] if (lo <= x[0] <= hi and lo <= x[1] <= hi) else 1e10
    cands = []
    if inits is None:
        g = np.arange(lo, hi + 1e-12, grid_step)
        vals = np.array([[f((a, b)) for b in g] for a in g])
        i, j = np.unravel_index(np.argmin(vals), vals.shape)
        cands.append(np.array([g[i], g[j]]))
    else:
        cands = [np.array([th[3], th[4]], float) for th in inits]
    best = None
    for x0 in cands:
        r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=5000))
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
    al, be = best[0]
    c, obj = _vp_inner(al, be, N, D, L)
    E, A, B = (max(v, 1e-300) for v in c)
    return sl.Chinchilla(E=E, A=A, B=B, alpha=al, beta=be,
                         extra=dict(objective=obj, n=len(L), estimator="vpnls", boundary=bool(np.any(c <= 0))))


# ----------------------------------------------------------------------------- normalised (KMW)
NORM_GRID = dict(a=np.array([-3.0, -1.5, 0.0]), b=np.array([-3.0, -1.5, 0.0]), e=np.array([-1.0, 0.0, 0.5]),
                 alpha=np.array([0.2, 0.5, 1.0]), beta=np.array([0.2, 0.5, 1.0]))


def to_norm(th, Nbar, Dbar):
    """theta -> normalised theta (ln a', ln b', ln E, alpha, beta)."""
    th = np.asarray(th, float).copy()
    th[0] -= th[3] * np.log(Nbar)
    th[1] -= th[4] * np.log(Dbar)
    return th


def from_norm(thn, Nbar, Dbar):
    th = np.asarray(thn, float).copy()
    th[0] += th[3] * np.log(Nbar)
    th[1] += th[4] * np.log(Dbar)
    return th


def fit_normalized(N, D, L, inits=None, grid=None, delta=HUBER_DELTA, Nbar=None, Dbar=None):
    N, D = np.asarray(N, float), np.asarray(D, float)
    Nbar = Nbar or float(np.exp(np.mean(np.log(N))))
    Dbar = Dbar or float(np.exp(np.mean(np.log(D))))
    if inits is None:
        m = sl.fit_chinchilla(N / Nbar, D / Dbar, L, delta=delta, grid=grid or NORM_GRID)
    else:
        m = fit_huber(N / Nbar, D / Dbar, L, inits=[to_norm(t, Nbar, Dbar) for t in inits], delta=delta)
    thn = m.theta
    out = from_theta(from_norm(thn, Nbar, Dbar), objective=m.extra["objective"], n=len(N), estimator="norm",
                     Nbar=Nbar, Dbar=Dbar, a_norm=float(np.exp(thn[0])), b_norm=float(np.exp(thn[1])))
    return out


# ----------------------------------------------------------------------------- registry
ESTIMATORS = {
    "huber": fit_huber,
    "gauss_log": fit_gauss_log,
    "lad_log": fit_lad,
    "nls_lev": fit_nls_levels,
    "vpnls": fit_vpnls,
    "norm": fit_normalized,
}
LABELS = {
    "huber": r"Huber-LSE, $\delta=10^{-3}$",
    "gauss_log": "Gaussian NLS, log loss",
    "lad_log": "LAD, log loss",
    "nls_lev": "NLS, levels",
    "vpnls": "VPNLS, levels",
    "norm": "Normalized (KMW), Huber",
}


def fit(name, N, D, L, **kw):
    return ESTIMATORS[name](N, D, L, **kw)


# ----------------------------------------------------------------------------- diagnostics
def jac_log(th, N, D):
    """n x 5 Jacobian of ln Lhat w.r.t. theta = (lnA, lnB, lnE, alpha, beta)."""
    lN, lD = np.log(np.asarray(N, float)), np.log(np.asarray(D, float))
    a, b, e, al, be = th
    t = np.column_stack([a - al * lN, b - be * lD, np.full_like(lN, e)])
    p = np.exp(t - t.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.column_stack([p[:, 0], p[:, 1], p[:, 2], -p[:, 0] * lN, -p[:, 1] * lD])


def jac_log_norm(thn, N, D, Nbar, Dbar):
    """Jacobian w.r.t. the normalised parameters (ln a', ln b', ln E, alpha, beta)."""
    return jac_log(thn, np.asarray(N) / Nbar, np.asarray(D) / Dbar)


def cond_stats(J):
    """Singular values; raw condition number; Belsley scaled condition number (columns scaled to unit length)."""
    s = np.linalg.svd(J, compute_uv=False)
    Js = J / np.linalg.norm(J, axis=0)
    ss = np.linalg.svd(Js, compute_uv=False)
    return dict(sv=s, cond=float(s[0] / s[-1]), sv_scaled=ss, cond_scaled=float(ss[0] / ss[-1]))


def share_linear_region(m, N, D, L, delta=HUBER_DELTA):
    r = lse_pred(theta_of(m), *_logs(N, D, L)[:2]) - np.log(np.asarray(L, float))
    return float(np.mean(np.abs(r) > delta)), r
