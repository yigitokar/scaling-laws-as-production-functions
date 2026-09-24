"""m2_est.py -- estimators, specification tests and bootstrap machinery for m2_techpanel.

Conventions (paper/notes/model_spec.md): L = E + A N^-alpha + B D^-beta; u = A N^-alpha, v = B D^-beta;
a = beta/(alpha+beta); gamma = alpha beta/(alpha+beta); sigma* = 2/(2+alpha+beta); M = D/N; C = 6ND.
All estimators work on log loss (Hoffmann/Besiroglu): Huber(delta=1e-3) or Gaussian NLS (delta=None).
The reference Chinchilla estimator is sl.fit_chinchilla (not modified here).
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import huber as _huber

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import sl  # noqa: E402

N_JOBS = 6
DELTA = {"huber": 1e-3, "nls": None}
CGRID = [1e21, 1e23, 1e25]


# ============================================================================ Chinchilla fits and derived objects
def fit_chin(N, D, L, est="huber", grid="default", init=None, weights=None):
    """Hoffmann/Besiroglu estimator via sl.fit_chinchilla; returns theta=(ln A, ln B, ln E, alpha, beta), objective."""
    g = sl.DEFAULT_GRID if grid == "default" else sl.FAST_GRID
    if init is not None:
        init = np.asarray(init, float).copy()
        init[2] = max(init[2], -50.0)          # E -> 0 boundary: keep ln E finite for warm starts
    with np.errstate(divide="ignore"):
        m = sl.fit_chinchilla(N, D, L, delta=DELTA[est], weights=weights, grid=g, init=init)
        th = m.theta
    th[2] = max(th[2], -50.0) if np.isfinite(th[2]) or th[2] < 0 else th[2]
    return th, m.extra["objective"]


def level_starts(N, D, L, alphas=(0.12, 0.25, 0.4, 0.6), betas=(0.12, 0.25, 0.4, 0.6), efracs=(0.3, 0.7, 0.9)):
    """Starting values that reproduce the loss level at the design centre: for each (alpha, beta, E) pick
    ln A, ln B so that u = v = (L_centre - E)/2 at the geometric means of N and D."""
    N, D, L = (np.asarray(v, float) for v in (N, D, L))
    lNg, lDg, Lc = np.mean(np.log(N)), np.mean(np.log(D)), np.exp(np.mean(np.log(L)))
    out = []
    for ef in efracs:
        E = ef * L.min()
        half = max(Lc - E, 1e-3) / 2
        for al in alphas:
            for be in betas:
                out.append(np.array([np.log(half) + al * lNg, np.log(half) + be * lDg, np.log(E), al, be]))
    return out


def fit_chin_multi(N, D, L, est="huber", starts=(), weights=None):
    """sl.fit_chinchilla from an explicit list of starting values (best objective wins)."""
    best = None
    for st in starts:
        th, f = fit_chin(N, D, L, est, init=st, weights=weights)
        if np.all(np.isfinite(th)) and (best is None or f < best[1]):
            best = (th, f)
    return best


def chin_from_theta(th):
    return sl.Chinchilla.from_theta(th)


def derived(th, N, D, centers=None):
    """Economic objects from theta. A_norm, B_norm = u, v at the sample geometric means (KMW-type
    normalization: these are the reducible-loss contributions at the design centre and are far better
    identified than A, B themselves). sigma range = min/max of local sigma over the sample points."""
    m = chin_from_theta(th)
    N, D = np.asarray(N, float), np.asarray(D, float)
    Ng, Dg = centers if centers is not None else (np.exp(np.mean(np.log(N))), np.exp(np.mean(np.log(D))))
    s = m.sigma(N, D)
    out = dict(E=m.E, A=m.A, B=m.B, alpha=m.alpha, beta=m.beta, a=m.a_N, gamma=m.gamma, sigma_star=m.sigma_star,
               A_norm=m.A * Ng ** -m.alpha, B_norm=m.B * Dg ** -m.beta, sigma_min=float(np.min(s)),
               sigma_max=float(np.max(s)), alpha_minus_beta=m.alpha - m.beta)
    for C in CGRID:
        out[f"Mstar_{C:.0e}".replace("+", "")] = float(m.D_opt(C) / m.N_opt(C))
    return out


# ============================================================================ bootstrap index draws
def boot_indices(cluster, B, seed):
    """Cluster (pairs) bootstrap index draws; returns list of index arrays."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(cluster)
    _, inv = np.unique(cl, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(inv.max() + 1)]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


# ---------------------------------------------------------------------------- translog curvature (rank-one test)
def translog(N, D, L, E, center=None):
    """OLS of ln(L-E) on a full quadratic in centred (n, d) = (ln N, ln D). Returns the Hessian of ln R and the
    'cross-curvature correlation' r = h_nd / sqrt(h_nn h_dd). Chinchilla implies H = c(n,d) [[a^2,-ab],[-ab,b^2]]:
    rank one, r = -1, null vector (beta, alpha) = expansion-path direction (model_spec Lemma 2)."""
    n, d = np.log(N), np.log(D)
    cn, cd = center if center is not None else (n.mean(), d.mean())
    n, d = n - cn, d - cd
    R = np.asarray(L) - E
    if np.any(R <= 0):
        return dict(r=np.nan, h_nn=np.nan, h_dd=np.nan, h_nd=np.nan, det=np.nan, ok=False)
    X = np.c_[np.ones_like(n), n, d, n ** 2, d ** 2, n * d]
    c = np.linalg.lstsq(X, np.log(R), rcond=None)[0]
    h_nn, h_dd, h_nd = 2 * c[3], 2 * c[4], c[5]
    r = h_nd / np.sqrt(h_nn * h_dd) if (h_nn > 0 and h_dd > 0) else np.nan
    return dict(r=float(r), h_nn=float(h_nn), h_dd=float(h_dd), h_nd=float(h_nd), tau=tau_stat(h_nn, h_dd, h_nd),
                det=float(h_nn * h_dd - h_nd ** 2), eps_N=float(-c[1]), eps_D=float(-c[2]), ok=True)


def tau_stat(h_nn, h_dd, h_nd):
    """Scale-free rank-one statistic tau = det(H)/||H||_F^2 in [-1/2, 1/2]; tau = 0 iff H has rank <= 1.
    Always defined (unlike r, which needs h_nn, h_dd > 0). tau < 0: indefinite curvature (saddle) in ln R."""
    return float((h_nn * h_dd - h_nd ** 2) / (h_nn ** 2 + h_dd ** 2 + 2 * h_nd ** 2))


# ---------------------------------------------------------------------------- Kaplan-type outer exponent q (kappa)
# L = E + (A' N^-a' + B' D^-b')^q ; q = 1 is Chinchilla, Kaplan (2020) has E = 0, inner D exponent 1, q = alpha_D.
# Isoquants are those of the inner aggregator, so sigma*_q = 2/(2 + a' + b'); frontier exponent gamma_q = q a'b'/(a'+b').
def _q_obj(th, x, z, y, delta):
    a, b, e, al, be, q = th
    t1, t2 = a - al * x, b - be * z
    m12 = np.maximum(t1, t2)
    s = m12 + np.log(np.exp(t1 - m12) + np.exp(t2 - m12))          # log inner aggregator
    p1 = np.exp(t1 - s); p2 = 1.0 - p1
    t = q * s                                                       # log R
    mm = np.maximum(t, e)
    pred = mm + np.log(np.exp(t - mm) + np.exp(e - mm))
    wt = np.exp(t - pred); we = 1.0 - wt
    r = pred - y
    if delta is None:
        f, psi = 0.5 * np.sum(r ** 2), r
    else:
        f, psi = np.sum(_huber(delta, r)), np.clip(r, -delta, delta)
    g = np.array([np.sum(psi * wt * q * p1), np.sum(psi * wt * q * p2), np.sum(psi * we),
                  np.sum(psi * wt * q * p1 * -x), np.sum(psi * wt * q * p2 * -z), np.sum(psi * wt * s)])
    return f, g


def fit_q(N, D, L, est="huber", th_chin=None, init=None):
    x, z, y = np.log(N), np.log(D), np.log(L)
    delta = DELTA[est]
    bounds = [(None, None)] * 5 + [(0.05, 20.0)]
    bounds[3] = bounds[4] = (1e-4, None)
    starts = []
    if init is not None:
        starts.append(np.asarray(init, float))
    if th_chin is not None:
        a, b, e, al, be = th_chin
        for q0 in (1.0, 0.5, 0.75, 1.5, 2.0, 3.0):
            starts.append(np.array([a / q0, b / q0, e, al / q0, be / q0, q0]))
    best = None
    for st in starts:
        r = minimize(_q_obj, st, args=(x, z, y, delta), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
    return best


def q_derived(th):
    a, b, e, al, be, q = th
    return dict(q=q, alpha_in=al, beta_in=be, sigma_star_q=2.0 / (2.0 + al + be), a_q=be / (al + be),
                gamma_q=q * al * be / (al + be), E_q=float(np.exp(e)))


# ---------------------------------------------------------------------------- panel ("plants") Chinchilla models
def panel_design(model, g, R):
    """Return list of 5 sparse matrices Z_k (n x p) mapping theta to per-observation (ln A, ln B, ln E, alpha, beta),
    plus parameter names. g in 0..R-1 = recipe/corpus index. Normalisations: group 0 is the reference."""
    n = len(g)
    rows = np.arange(n)
    names = []

    def col(v):
        return sparse.csr_matrix((np.ones(n), (rows, np.full(n, v))), shape=(n, P))

    def gcols(start, skip0=True):
        # one column per group (skip group 0 if skip0): per-obs dummy
        mask = (g > 0) if skip0 else np.ones(n, bool)
        c = start + (g - (1 if skip0 else 0))
        return sparse.csr_matrix((np.ones(mask.sum()), (rows[mask], c[mask])), shape=(n, P))

    Rm1 = R - 1
    if model == "pooled":
        P = 5; names = ["lnA", "lnB", "lnE", "alpha", "beta"]
        Z = [col(0), col(1), col(2), col(3), col(4)]
    elif model == "ces":
        P = 4; names = ["lnA", "lnB", "lnE", "alpha=beta"]
        Z = [col(0), col(1), col(2), col(3), col(3)]
    elif model == "Eshift":        # (iv) only the asymptote differs across recipes
        P = 4 + R; names = ["lnA", "lnB", "alpha", "beta"] + [f"lnE_{r}" for r in range(R)]
        Z = [col(0), col(1), gcols(4, skip0=False), col(2), col(3)]
    elif model == "hicks":         # (i) Hicks-neutral in reducible loss: A_r = e^{-w_r} A, B_r = e^{-w_r} B, common E
        P = 5 + Rm1; names = ["lnA", "lnB", "lnE", "alpha", "beta"] + [f"lam_{r}" for r in range(1, R)]
        Z = [col(0) + gcols(5), col(1) + gcols(5), col(2), col(3), col(4)]
    elif model == "hicksE":        # Hicks-neutral in reducible loss, recipe-specific E (pure factor-bias test vs CE)
        P = 4 + Rm1 + R; names = ["lnA", "lnB", "alpha", "beta"] + [f"lam_{r}" for r in range(1, R)] + [f"lnE_{r}" for r in range(R)]
        Z = [col(0) + gcols(4), col(1) + gcols(4), gcols(4 + Rm1, skip0=False), col(2), col(3)]
    elif model == "daug":          # (ii) data-augmenting: only B_r differs
        P = 5 + Rm1; names = ["lnA", "lnB", "lnE", "alpha", "beta"] + [f"kB_{r}" for r in range(1, R)]
        Z = [col(0), col(1) + gcols(5), col(2), col(3), col(4)]
    elif model == "paug":          # (iii) parameter-augmenting: only A_r differs
        P = 5 + Rm1; names = ["lnA", "lnB", "lnE", "alpha", "beta"] + [f"kA_{r}" for r in range(1, R)]
        Z = [col(0) + gcols(5), col(1), col(2), col(3), col(4)]
    elif model == "daugE":         # data-augmenting + recipe-specific E
        P = 4 + Rm1 + R; names = ["lnA", "lnB", "alpha", "beta"] + [f"kB_{r}" for r in range(1, R)] + [f"lnE_{r}" for r in range(R)]
        Z = [col(0), col(1) + gcols(4), gcols(4 + Rm1, skip0=False), col(2), col(3)]
    elif model == "paugE":         # parameter-augmenting + recipe-specific E
        P = 4 + Rm1 + R; names = ["lnA", "lnB", "alpha", "beta"] + [f"kA_{r}" for r in range(1, R)] + [f"lnE_{r}" for r in range(R)]
        Z = [col(0) + gcols(4), col(1), gcols(4 + Rm1, skip0=False), col(2), col(3)]
    elif model == "CE":            # common exponents; A_r, B_r, E_r free (general factor-augmenting + E shift)
        P = 2 + 3 * R; names = ["alpha", "beta"] + [f"lnA_{r}" for r in range(R)] + [f"lnB_{r}" for r in range(R)] + [f"lnE_{r}" for r in range(R)]
        Z = [gcols(2, False), gcols(2 + R, False), gcols(2 + 2 * R, False), col(0), col(1)]
    else:
        raise ValueError(model)
    return [z.tocsr() for z in Z], names


def _panel_obj(th, Z, ZT, x, z, y, delta, w):
    a, b, e, al, be = (Zk @ th for Zk in Z)
    t1, t2 = a - al * x, b - be * z
    m = np.maximum(np.maximum(t1, t2), e)
    p1, p2, p3 = np.exp(t1 - m), np.exp(t2 - m), np.exp(e - m)
    s = p1 + p2 + p3
    pred = m + np.log(s)
    p1, p2, p3 = p1 / s, p2 / s, p3 / s
    r = pred - y
    if delta is None:
        f, psi = 0.5 * np.sum(w * r ** 2), w * r
    else:
        f, psi = np.sum(w * _huber(delta, r)), w * np.clip(r, -delta, delta)
    grad = ZT[0] @ (psi * p1) + ZT[1] @ (psi * p2) + ZT[2] @ (psi * p3) + ZT[3] @ (psi * p1 * -x) + ZT[4] @ (psi * p2 * -z)
    return f, grad


def panel_perobs(model, th, g, R):
    Z, _ = panel_design(model, g, R)
    return np.vstack([Zk @ th for Zk in Z]).T  # n x 5


def panel_pred(model, th, x, z, g, R):
    P = panel_perobs(model, th, g, R)
    return np.logaddexp(np.logaddexp(P[:, 0] - P[:, 3] * x, P[:, 1] - P[:, 4] * z), P[:, 2])


def embed(model, perobs, g, R):
    """Starting value for `model` from per-observation parameters of another (nested) fit."""
    grp = [np.where(g == r)[0][0] for r in range(R)]
    G = perobs[grp]                                   # R x 5 : lnA, lnB, lnE, alpha, beta per group
    al, be = G[:, 3].mean(), G[:, 4].mean()
    if model == "pooled":
        return np.array([G[:, 0].mean(), G[:, 1].mean(), G[:, 2].mean(), al, be])
    if model == "ces":
        return np.array([G[:, 0].mean(), G[:, 1].mean(), G[:, 2].mean(), 0.5 * (al + be)])
    if model == "Eshift":
        return np.r_[G[:, 0].mean(), G[:, 1].mean(), al, be, G[:, 2]]
    if model == "hicks":
        lam = 0.5 * ((G[:, 0] - G[0, 0]) + (G[:, 1] - G[0, 1]))
        return np.r_[G[0, 0], G[0, 1], G[:, 2].mean(), al, be, lam[1:]]
    if model == "hicksE":
        lam = 0.5 * ((G[:, 0] - G[0, 0]) + (G[:, 1] - G[0, 1]))
        return np.r_[G[0, 0], G[0, 1], al, be, lam[1:], G[:, 2]]
    if model == "daug":
        return np.r_[G[:, 0].mean(), G[0, 1], G[:, 2].mean(), al, be, (G[:, 1] - G[0, 1])[1:]]
    if model == "paug":
        return np.r_[G[0, 0], G[:, 1].mean(), G[:, 2].mean(), al, be, (G[:, 0] - G[0, 0])[1:]]
    if model == "daugE":
        return np.r_[G[0, 0], G[0, 1], al, be, (G[:, 1] - G[0, 1])[1:], G[:, 2]]
    if model == "paugE":
        return np.r_[G[0, 0], G[0, 1], al, be, (G[:, 0] - G[0, 0])[1:], G[:, 2]]
    if model == "CE":
        return np.r_[al, be, G[:, 0], G[:, 1], G[:, 2]]
    raise ValueError(model)


def fit_panel(model, x, z, y, g, R, est="nls", starts=(), w=None):
    """Minimise the Huber/NLS log-loss objective of a panel model by L-BFGS-B from a list of starting values.
    Reference implementation: used during development to validate fit_panel_ls (same optimum to 7 digits);
    run.py uses the faster fit_panel_ls."""
    Z, names = panel_design(model, g, R)
    ZT = [Zk.T.tocsr() for Zk in Z]
    w = np.ones_like(y) if w is None else w
    best = None
    for st in starts:
        r = minimize(_panel_obj, np.asarray(st, float), args=(Z, ZT, x, z, y, DELTA[est], w), jac=True,
                     method="L-BFGS-B", options=dict(maxiter=50000, ftol=1e-15, gtol=1e-11))
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
    return best[0], best[1], names


def _panel_resjac(th, Z, x, z, y, sw):
    a, b, e, al, be = (Zk @ th for Zk in Z)
    t1, t2 = a - al * x, b - be * z
    m = np.maximum(np.maximum(t1, t2), e)
    p1, p2, p3 = np.exp(t1 - m), np.exp(t2 - m), np.exp(e - m)
    s = p1 + p2 + p3
    pred = m + np.log(s)
    p1, p2, p3 = p1 / s, p2 / s, p3 / s
    r = sw * (pred - y)
    J = (Z[0].multiply((sw * p1)[:, None]) + Z[1].multiply((sw * p2)[:, None]) + Z[2].multiply((sw * p3)[:, None])
         + Z[3].multiply((sw * p1 * -x)[:, None]) + Z[4].multiply((sw * p2 * -z)[:, None]))
    return r, np.asarray(J.todense())


def fit_panel_ls(model, x, z, y, g, R, est="nls", starts=(), w=None, tight=True):
    """Same objective as fit_panel (Gaussian NLS, or Huber with delta=1e-3 via least_squares(loss='huber',
    f_scale=delta), which minimises 2 x the Huber objective), solved by a trust-region Gauss-Newton method with an
    analytic sparse Jacobian. Much faster than L-BFGS on the 22k-observation DataDecide panel."""
    from scipy.optimize import least_squares
    Z, names = panel_design(model, g, R)
    sw = np.ones_like(y) if w is None else np.sqrt(w)
    cache = {}

    def fun(th):
        r, J = _panel_resjac(th, Z, x, z, y, sw)
        cache["J"] = J
        return r

    def jac(th):
        return _panel_resjac(th, Z, x, z, y, sw)[1]

    best = None
    tol = 1e-12 if tight else 1e-9
    for st in starts:
        kw = dict(loss="linear") if est == "nls" else dict(loss="huber", f_scale=DELTA["huber"])
        r = least_squares(fun, np.asarray(st, float), jac=jac, method="trf", x_scale="jac", ftol=tol, xtol=tol,
                          gtol=tol, max_nfev=2000 if tight else 300, tr_solver="exact", **kw)
        f = r.cost   # = 0.5*sum r^2 (NLS) or sum Huber_delta(r) (huber): same scale as fit_panel / sl.fit_chinchilla
        if best is None or f < best[1]:
            best = (r.x, f)
    return best[0], best[1], names


def fit_full(x, z, y, g, R, est="nls", inits=None, grid="fast"):
    """Unrestricted model: separate Chinchilla fit per group. Returns list of theta and total objective."""
    ths, tot = [], 0.0
    for r in range(R):
        i = g == r
        th, f = fit_chin(np.exp(x[i]), np.exp(z[i]), np.exp(y[i]), est, grid=grid,
                         init=None if inits is None else inits[r])
        ths.append(th); tot += f
    return ths, tot


def full_perobs(ths, g):
    return np.vstack([ths[r] for r in g])[:, [0, 1, 2, 3, 4]]


# ============================================================================ worker functions (top level, picklable)
def w_chin(data, items, est="huber", init=None, centers=None, do_translog=True):
    """Pairs/cluster bootstrap of the Chinchilla fit; each item is an index array. Returns theta (5) and translog r."""
    N, D, L = data["N"], data["D"], data["L"]
    wts = data.get("w")
    out = []
    for idx in items:
        try:
            th, f = fit_chin(N[idx], D[idx], L[idx], est, init=init, weights=None if wts is None else wts[idx])
            tl = translog(N[idx], D[idx], L[idx], np.exp(th[2]), center=centers) if do_translog else {}
            out.append(np.r_[th, f, tl.get("r", np.nan), tl.get("h_nn", np.nan), tl.get("h_dd", np.nan), tl.get("h_nd", np.nan)])
        except Exception:
            out.append(np.full(10, np.nan))
    return out


def w_translog_null(data, items, est="huber", th0=None, centers=None):
    """Residual bootstrap under the fitted Chinchilla model (H0 for the rank-one test): y* = fitted + resampled
    residuals; re-estimate Chinchilla (warm) and the translog. Items are index arrays for residual resampling."""
    N, D, L = data["N"], data["D"], data["L"]
    x, z = np.log(N), np.log(D)
    a, b, e, al, be = th0
    fit = np.logaddexp(np.logaddexp(a - al * x, b - be * z), e)
    res = np.log(L) - fit
    out = []
    for idx in items:
        Ls = np.exp(fit + res[idx])
        th, _ = fit_chin(N, D, Ls, est, init=th0)
        tl = translog(N, D, Ls, np.exp(th[2]), center=centers)
        out.append((tl["r"], tl.get("tau", np.nan)))
    return out


def w_q(data, items, est="huber", init=None, centers=None):
    """Pairs/cluster bootstrap of the generalized (q) form. Returns theta (6), objective, and the rank-one statistic
    tau of a translog in ln(L - E_q) with E_q from the same draw (column 7; review fix 2026-09-23)."""
    N, D, L = data["N"], data["D"], data["L"]
    out = []
    for idx in items:
        try:
            th, f = fit_q(N[idx], D[idx], L[idx], est, init=init)
            tl = translog(N[idx], D[idx], L[idx], np.exp(th[2]), center=centers)
            out.append(np.r_[th, f, tl.get("tau", np.nan)])
        except Exception:
            out.append(np.full(8, np.nan))
    return out


def w_translog_null_q(data, items, est="huber", thq0=None, centers=None):
    """Rank-one test under the fitted *generalized* (q) model (review fix). The q family also implies a rank-one
    Hessian of ln(L - E), but with its own E. Using the Chinchilla E-hat when q != 1 adds a constant to reducible loss,
    which by itself produces tau < 0 (a noise-free q-family surface analysed with the Chinchilla E-hat gives
    tau = -0.27 on the Farseer design). Here y* = fitted_q + resampled residuals; E is re-estimated by the q family
    (warm start) and tau is computed with that E."""
    N, D, L = data["N"], data["D"], data["L"]
    x, z = np.log(N), np.log(D)
    a, b, e, al, be, q = thq0
    fit = np.logaddexp(q * np.logaddexp(a - al * x, b - be * z), e)
    res = np.log(L) - fit
    out = []
    for idx in items:
        Ls = np.exp(fit + res[idx])
        try:
            th, _ = fit_q(N, D, Ls, est, init=thq0)
            tl = translog(N, D, Ls, np.exp(th[2]), center=centers)
            out.append((tl["r"], tl.get("tau", np.nan)))
        except Exception:
            out.append((np.nan, np.nan))
    return out


def _fit_any(model, x, z, y, g, R, start):
    """NLS fit of model (panel or 'full'); returns SSR (=2*objective) and theta/start for re-use."""
    if model == "full":
        ths, f = fit_full(x, z, y, g, R, "nls", inits=start)
        return 2 * f, ths
    th, f, _ = fit_panel_ls(model, x, z, y, g, R, "nls", starts=start, tight=False)
    return 2 * f, th


def w_wild(data, items, m0=None, m1=None, th0=None, th1=None):
    """Wild cluster *restricted* bootstrap (Cameron, Gelbach & Miller 2008) of the quasi-LR statistic
    n ln(SSR0/SSR1) for H0: model m0 within model m1 (Gaussian NLS on log loss). Each item = vector of
    cluster weights (Rademacher). y* = yhat0 + w_c * e0; both models are re-fitted on y*; the unrestricted
    fit starts from the restricted solution embedded in m1 (so SSR1* <= SSR0*) and from th1."""
    x, z, y, g, R, cl = data["x"], data["z"], data["y"], data["g"], data["R"], data["cl"]
    yhat0 = data["yhat0"]
    e0 = y - yhat0
    n = len(y)
    out = []
    for wc in items:
        ys = yhat0 + wc[cl] * e0
        s0, t0 = _fit_any(m0, x, z, ys, g, R, [th0] if m0 != "full" else th0)
        # embed restricted solution into the unrestricted parameterisation
        if m0 == "full":
            raise ValueError
        P0 = panel_perobs(m0, t0, g, R)
        if m1 == "full":
            grp = [np.where(g == r)[0][0] for r in range(R)]
            cand = [P0[i] for i in grp]
            s1a, t1a = _fit_any("full", x, z, ys, g, R, cand)
            s1b, _ = _fit_any("full", x, z, ys, g, R, th1)
            s1 = min(s1a, s1b)
        else:
            s1, _ = _fit_any(m1, x, z, ys, g, R, [embed(m1, P0, g, R), th1])
        s1 = min(s1, s0)
        out.append(n * np.log(s0 / s1))
    return out


# ---------------------------------------------------------------------------- local nonparametric sigma
def local_quad(x, z, f, x0, z0, hx, hz, exclude=None):
    """Kernel-weighted local quadratic regression of f on (x, z) at (x0, z0); returns coefficients
    [f, f_x, f_z, f_xx/2, f_zz/2, f_xz] and effective n."""
    dx, dz = x - x0, z - z0
    w = np.exp(-0.5 * ((dx / hx) ** 2 + (dz / hz) ** 2))
    if exclude is not None:
        w[exclude] = 0.0
    X = np.c_[np.ones_like(dx), dx, dz, dx ** 2, dz ** 2, dx * dz]
    sw = np.sqrt(w)
    c = np.linalg.lstsq(X * sw[:, None], f * sw, rcond=None)[0]
    return c, (w.sum() ** 2) / np.sum(w ** 2)


def sigma_from_derivs(fn, fd, fnn, fdd, fnd):
    """Hicks elasticity of substitution from the gradient and Hessian of ANY monotone transform f of output in
    (n, d) = (ln N, ln D):  1/sigma = 1 - (f_nn f_d^2 - 2 f_nd f_n f_d + f_dd f_n^2) / (f_n f_d (f_n + f_d)).
    (Invariant to f -> g(f): numerator and denominator both scale with g'^3, so E is not needed.)"""
    Q = fnn * fd ** 2 - 2 * fnd * fn * fd + fdd * fn ** 2
    return 1.0 / (1.0 - Q / (fn * fd * (fn + fd)))


def local_sigma(N, D, L, pts, hx, hz):
    x, z, f = np.log(N), np.log(D), np.log(L)
    out = []
    for (x0, z0) in pts:
        c, neff = local_quad(x, z, f, x0, z0, hx, hz)
        s = sigma_from_derivs(c[1], c[2], 2 * c[3], 2 * c[4], c[5])
        out.append((s, -c[1], -c[2], neff))   # sigma, elasticities of L wrt N and D (positive = loss falls)
    return np.array(out)


def w_localsig(data, items, pts=None, hx=None, hz=None):
    N, D, L = data["N"], data["D"], data["L"]
    return [local_sigma(N[idx], D[idx], L[idx], pts, hx, hz)[:, 0] for idx in items]


def loo_cv_bandwidth(N, D, L, mults, sx, sz):
    """Leave-one-out CV of the local quadratic predictor of ln L over bandwidth multipliers (h = mult * sd)."""
    x, z, f = np.log(N), np.log(D), np.log(L)
    res = {}
    for mx, mz in itertools.product(mults, mults):
        hx, hz = mx * sx, mz * sz
        err = []
        for i in range(len(f)):
            c, _ = local_quad(x, z, f, x[i], z[i], hx, hz, exclude=i)
            err.append(f[i] - c[0])
        res[(mx, mz)] = float(np.sqrt(np.mean(np.square(err))))
    return res


# ---------------------------------------------------------------------------- Farseer functional form
def farseer_lnL(th, N, D):
    """Li et al. (2025b) Eq. 3: L = exp(a3 N^g3 + b3) + exp(a2 N^g2 + b2) * D^(-exp(a1 N^g1 + b1)); N in units of 1e9
    (N units are absorbed by a_k; D enters in raw tokens as in the authors' code)."""
    a1, g1, b1, a2, g2, b2, a3, g3, b3 = th
    n = N / 1e9
    lnE = a3 * n ** g3 + b3
    k = np.exp(a1 * n ** g1 + b1)
    lnB = a2 * n ** g2 + b2
    return np.logaddexp(lnE, lnB - k * np.log(D))


def fit_farseer(N, D, L, est="nls", starts=None):
    y = np.log(L)
    delta = DELTA[est]

    def obj(th):
        r = farseer_lnL(th, N, D) - y
        if not np.all(np.isfinite(r)):
            return 1e10
        return 0.5 * np.sum(r ** 2) if delta is None else np.sum(_huber(delta, r))

    best = None
    for st in starts:
        for meth in ("L-BFGS-B", "Nelder-Mead"):
            try:
                r = minimize(obj, st, method=meth, options=dict(maxiter=40000, maxfev=40000) if meth == "Nelder-Mead"
                             else dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
            except Exception:
                continue
            if best is None or r.fun < best[1]:
                best = (r.x, r.fun)
            st = best[0]
    return best


def farseer_starts(N, D, L):
    """Starting values following the authors' multi-round logic: per-N fits L = E_N + B_N D^-k_N, then fit
    ln E_N, ln B_N, ln k_N as a_k (N/1e9)^g_k + b_k over the N groups."""
    Ns = np.unique(N)
    EN, BN, kN = [], [], []
    for n0 in Ns:
        i = N == n0
        d, l = D[i], L[i]
        best = None
        for E0 in (0.3 * l.min(), 0.6 * l.min(), 0.85 * l.min()):
            for k0 in (0.2, 0.4, 0.8):
                def o(p):
                    pr = np.logaddexp(p[0], p[1] - np.exp(p[2]) * np.log(d))
                    return np.sum((pr - np.log(l)) ** 2)
                st = np.array([np.log(E0), np.log(max(l.max() - E0, 1e-3)) + k0 * np.log(d.min()), np.log(k0)])
                r = minimize(o, st, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-10, fatol=1e-14))
                if best is None or r.fun < best[1]:
                    best = (r.x, r.fun)
        EN.append(best[0][0]); BN.append(best[0][1]); kN.append(best[0][2])
    n = Ns / 1e9
    EN, BN, kN = map(np.array, (EN, BN, kN))

    def fitpow(yv):
        best = None
        for g0 in np.linspace(-1.0, 1.0, 41):
            if abs(g0) < 1e-3:
                continue
            X = np.c_[n ** g0, np.ones_like(n)]
            c, *_ = np.linalg.lstsq(X, yv, rcond=None)
            s = np.sum((X @ c - yv) ** 2)
            if best is None or s < best[1]:
                best = ((c[0], g0, c[1]), s)
        return best[0]

    a1, g1, b1 = fitpow(kN)
    a2, g2, b2 = fitpow(BN)
    a3, g3, b3 = fitpow(EN)
    return np.array([a1, g1, b1, a2, g2, b2, a3, g3, b3])


def numderiv_sigma(fun, n0, d0, h=1e-3):
    """sigma from numerical derivatives of f(n, d) (= ln L as a function of ln N, ln D)."""
    f = lambda a, b: fun(a, b)
    fn = (f(n0 + h, d0) - f(n0 - h, d0)) / (2 * h)
    fd = (f(n0, d0 + h) - f(n0, d0 - h)) / (2 * h)
    fnn = (f(n0 + h, d0) - 2 * f(n0, d0) + f(n0 - h, d0)) / h ** 2
    fdd = (f(n0, d0 + h) - 2 * f(n0, d0) + f(n0, d0 - h)) / h ** 2
    fnd = (f(n0 + h, d0 + h) - f(n0 + h, d0 - h) - f(n0 - h, d0 + h) + f(n0 - h, d0 - h)) / (4 * h ** 2)
    return sigma_from_derivs(fn, fd, fnn, fdd, fnd), -fn, -fd


def gauss_ic(resid, k):
    """Gaussian log-likelihood, AIC, BIC of log-loss residuals with k mean parameters (+1 variance)."""
    n = len(resid)
    s2 = np.mean(resid ** 2)
    ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
    return ll, 2 * (k + 1) - 2 * ll, np.log(n) * (k + 1) - 2 * ll


def w_point(_, jobs):
    """Point estimates. Each job: (key, est, N, D, L, weights, mode, starts); mode in {'default','fast','starts'}.
    Grid fits are complemented by level-preserving starts as a safeguard against grid failures."""
    out = []
    for key, est, N, D, L, w, mode, starts in jobs:
        if mode == "starts":
            th, f = fit_chin_multi(N, D, L, est, starts=starts, weights=w)
        else:
            th, f = fit_chin(N, D, L, est, grid=mode, weights=w)
            th2, f2 = fit_chin_multi(N, D, L, est, starts=level_starts(N, D, L), weights=w)
            if f2 < f - 1e-12:
                th, f = th2, f2
        out.append((key, est, th, f))
    return out


def w_panel_ls(data, items, model="CE", est="nls", init=None):
    """Cluster bootstrap of a panel model with the fast solver (warm start)."""
    x, z, y, g, R = data["x"], data["z"], data["y"], data["g"], data["R"]
    out = []
    for idx in items:
        th, f, _ = fit_panel_ls(model, x[idx], z[idx], y[idx], g[idx], R, est, starts=[init], tight=False)
        out.append(th)
    return out


def w_recipe_boot(data, items, est="huber", init=None):
    N, D, L = data["N"], data["D"], data["L"]
    return [fit_chin(N[idx], D[idx], L[idx], est, init=init)[0] for idx in items]
