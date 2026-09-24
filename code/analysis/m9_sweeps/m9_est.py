"""m9_est.py -- estimators for module m9_sweeps (plan Section 2 (Estimators) and Section 3 (Q1-Q6)).

Reused code (imported, never edited):
  sl.fit_chinchilla            Huber-LSE (delta = 1e-3) / Gaussian NLS Chinchilla form on log loss
  m2_est.fit_q                 kappa (outer-exponent) family, L = E + (A N^-a1 + B D^-b1)^kappa; sigma*_k = 2/(2+a1+b1)
  m2_est.fit_panel_ls("CE")    corpus-pair model: common (alpha, beta), corpus-specific (ln A, ln B, ln E)
  m2_est.local_quad            Gaussian-kernel local quadratic of ln L in (ln N, ln D) (ra1's model-free machinery)
  m2_est.sigma_from_derivs     Hicks elasticity from gradient and Hessian of any monotone transform of output
  ra1 kprofile                 kappa-family profile over sigma* (fixed S = a1 + b1), LR relative to the unrestricted fit

Model-free sigma* (plan Q1). On an exact isocost C = 6 N D the plan's identity is
    1/sigma* - 1 = L_nn|_C / (2 |dL*/d ln C|).
For convention P the cost is C = fpt(N) D (actual FLOPs), eta = d ln fpt/d ln N; the compute-optimal point satisfies
w = eta and the same argument gives the generalized identity
    1/sigma* - 1 = (L_nn|_C - eta' |dL*/d ln C|) / (eta (1 + eta) |dL*/d ln C|),
which reduces to the plan's formula at eta = 1 and equals the Hicks elasticity between N_nonemb and D at that point.
Both are evaluated with the local-quadratic derivatives (E-free, form-free; invariant to monotone transforms of L).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import m9_common as mc

import m2_est as me  # noqa: E402

sl = mc.sl
DELTA = {"huber": 1e-3, "nls": None}


# ============================================================================ Chinchilla form
def fit_chin(N, D, L, est="huber", starts=None, grid=None):
    """Chinchilla form on log loss. starts: list of theta (lnA, lnB, lnE, alpha, beta); grid: None, 'fast' or
    'default' (sl grids) -- the level-preserving starts of m2 are always added unless starts is given alone."""
    N, D, L = (np.asarray(v, float) for v in (N, D, L))
    cands = list(starts or [])
    if grid is not None or not cands:
        cands += me.level_starts(N, D, L)
    best = None
    if grid is not None:
        g = sl.FAST_GRID if grid == "fast" else sl.DEFAULT_GRID
        m = sl.fit_chinchilla(N, D, L, delta=DELTA[est], grid=g)
        best = (m.theta, m.extra["objective"])
    for st in cands:
        st = np.asarray(st, float).copy()
        st[2] = max(st[2], -30.0)
        with np.errstate(all="ignore"):
            m = sl.fit_chinchilla(N, D, L, delta=DELTA[est], init=st)
        if np.all(np.isfinite(m.theta)) and (best is None or m.extra["objective"] < best[1]):
            best = (m.theta, m.extra["objective"])
    return best


def pred_chin(th, N, D):
    lnA, lnB, lnE, al, be = th
    return np.logaddexp(np.logaddexp(lnA - al * np.log(N), lnB - be * np.log(D)), lnE)


def w_chin(th, N, D):
    lnA, lnB, lnE, al, be = th
    return np.exp(np.log(al) + lnA - al * np.log(N) - (np.log(be) + lnB - be * np.log(D)))


def eps_chin(th, N, D):
    """Output elasticities of LOSS (eps_N, eps_D) = -d ln L/d ln N, -d ln L/d ln D."""
    lnA, lnB, lnE, al, be = th
    u = np.exp(lnA - al * np.log(N))
    v = np.exp(lnB - be * np.log(D))
    Lh = u + v + np.exp(lnE)
    return al * u / Lh, be * v / Lh


# ============================================================================ kappa family
def fit_kappa(N, D, L, est="huber", th_chin=None, init=None):
    return me.fit_q(np.asarray(N, float), np.asarray(D, float), np.asarray(L, float), est=est, th_chin=th_chin, init=init)


def pred_kappa(p, N, D):
    lnA, lnB, lnE, a1, b1, k = p
    return np.logaddexp(k * np.logaddexp(lnA - a1 * np.log(N), lnB - b1 * np.log(D)), lnE)


def w_kappa(p, N, D):
    lnA, lnB, lnE, a1, b1, k = p
    return np.exp(np.log(a1) + lnA - a1 * np.log(N) - (np.log(b1) + lnB - b1 * np.log(D)))


def eps_kappa(p, N, D):
    lnA, lnB, lnE, a1, b1, k = p
    u = np.exp(lnA - a1 * np.log(N))
    v = np.exp(lnB - b1 * np.log(D))
    R = (u + v) ** k
    Lh = R + np.exp(lnE)
    return k * R / Lh * a1 * u / (u + v), k * R / Lh * b1 * v / (u + v)


def sigma_star_kappa(p):
    return 2.0 / (2.0 + p[3] + p[4])


def sigma_star_chin(th):
    return 2.0 / (2.0 + th[3] + th[4])


def sigma_at_w(a1, b1, w):
    """Hicks sigma of the Chinchilla/kappa isoquants at a point with wedge w (share s = w/(1+w))."""
    s = w / (1.0 + w)
    return 1.0 / (1.0 + s * b1 + (1.0 - s) * a1)


# ============================================================================ cost conventions and isocosts
def iso_z(conv, lnC, x):
    """ln D on the isocost of compute C at ln N = x. conv 'P': actual FLOPs C = fpt(N) D (non-embedding N on the
    ladder); 'P6'/'T': C = 6 N D."""
    if conv == "P":
        return lnC - mc.LADDER.G_of_x(x)
    return lnC - np.log(6.0) - x


def path_target(conv, x):
    """ln w on the compute-optimal path: ln eta(x) under P, 0 under P6/T."""
    return np.log(mc.LADDER.eta(x)) if conv == "P" else np.zeros_like(np.asarray(x, float))


# ============================================================================ local quadratic (model-free)
def local_coefs(x, z, f, pts, hx, hz):
    """Coefficients [f, f_x, f_z, f_xx/2, f_zz/2, f_xz] and n_eff at each (x0, z0) in pts (m2_est.local_quad)."""
    out = np.zeros((len(pts), 7))
    for i, (x0, z0) in enumerate(pts):
        c, ne = me.local_quad(x, z, f, x0, z0, hx, hz)
        out[i, :6], out[i, 6] = c, ne
    return out


def lnw_local(c):
    return np.log(c[..., 1] / c[..., 2])


def sigma_hicks(c):
    return me.sigma_from_derivs(c[..., 1], c[..., 2], 2 * c[..., 3], 2 * c[..., 4], c[..., 5])


def sigma_identity(c, conv, x0):
    """Plan identity (generalized for P): 1/sigma - 1 = (f_xx - 2 eta f_xz + eta^2 f_zz) / (eta (1+eta) |f_z|)."""
    eta = mc.LADDER.eta(x0) if conv == "P" else 1.0
    fxx, fzz, fxz, F = 2 * c[3], 2 * c[4], c[5], -c[2]
    inv = (fxx - 2 * eta * fxz + eta ** 2 * fzz) / (eta * (1 + eta) * F)
    return 1.0 / (1.0 + inv)


def cv_bandwidth(x, z, f, mults=(0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5)):
    """Leave-one-out CV of the local quadratic predictor of ln L over h = (mx sd(x), mz sd(z))."""
    import itertools
    sx, sz = np.std(x), np.std(z)
    best, table = None, []
    for mx_, mz_ in itertools.product(mults, mults):
        hx, hz = mx_ * sx, mz_ * sz
        err = []
        for i in range(len(f)):
            c, _ = me.local_quad(x, z, f, x[i], z[i], hx, hz, exclude=i)
            err.append(f[i] - c[0])
        rmse = float(np.sqrt(np.mean(np.square(err))))
        table.append(dict(mx=mx_, mz=mz_, hx=hx, hz=hz, loocv_rmse=rmse))
        if best is None or rmse < best[0]:
            best = (rmse, hx, hz, mx_, mz_)
    return best, table


def pilot_fit(x, z, f, hx, hz):
    """Local-quadratic fitted values at the design points (bootstrap population of the model-free statistics)."""
    return local_coefs(x, z, f, list(zip(x, z)), hx, hz)[:, 0]


def make_hull(conv, sizes_x, dmin, dmax, slack=0.05, min_sizes=3):
    """Isocost segment inside the design: ln N between the sizes whose observed D range contains the isocost's D."""
    def hull(lnC):
        ok = [xx for xx, a, b in zip(sizes_x, dmin, dmax)
              if np.log((1 - slack) * a) <= iso_z(conv, lnC, xx) <= np.log((1 + slack) * b)]
        if len(ok) < min_sizes:
            return None, None
        return min(ok), max(ok)
    return hull


def local_path(x, z, f, hx, hz, conv, C_levels, hull):
    """Model-free compute-optimal path: on each isocost, the root of ln w_local(x) - ln w_target(x) inside the hull.
    Returns rows (x*, sigma_hicks, sigma_identity, lnM*, n_eff)."""
    rows = []
    for C in C_levels:
        lnC = np.log(C)
        lo, hi = hull(lnC)
        if lo is None:
            rows.append([np.nan] * 5)
            continue

        def g(xx):
            c, _ = me.local_quad(x, z, f, xx, iso_z(conv, lnC, xx), hx, hz)
            if c[1] >= 0 or c[2] >= 0:
                return np.nan
            return np.log(c[1] / c[2]) - float(path_target(conv, np.array([xx]))[0])

        try:
            glo, ghi = g(lo), g(hi)
            if not (np.isfinite(glo) and np.isfinite(ghi)) or np.sign(glo) == np.sign(ghi):
                rows.append([np.nan] * 5)
                continue
            x0 = brentq(g, lo, hi, xtol=1e-7)
        except (ValueError, FloatingPointError):
            rows.append([np.nan] * 5)
            continue
        z0 = float(iso_z(conv, lnC, x0))
        c, ne = me.local_quad(x, z, f, x0, z0, hx, hz)
        rows.append([x0, float(sigma_hicks(c)), float(sigma_identity(c, conv, x0)), z0 - x0, ne])
    return np.array(rows, float)


def sigma_slope(x, z, f, hx, hz, conv, path, C_levels, grid_pts):
    """First-derivative model-free sigma* (ra1, Farseer): regress ln w_local on u = ln(M/M*_local(C)) and u^2 through
    the origin over grid points inside the path's compute range; sigma* = 1/(1 + b1). Only for w = 1 paths."""
    if conv == "P":
        return np.nan
    ok = np.isfinite(path[:, 0])
    if ok.sum() < 3:
        return np.nan
    lc = np.log(np.asarray(C_levels))[ok]
    lMs = path[ok, 3]
    gx, gz = grid_pts[:, 0], grid_pts[:, 1]
    lnCg = np.log(6.0) + gx + gz
    inside = (lnCg >= lc.min()) & (lnCg <= lc.max())
    if inside.sum() < 8:
        return np.nan
    c = local_coefs(x, z, f, list(zip(gx[inside], gz[inside])), hx, hz)
    good = (c[:, 1] < 0) & (c[:, 2] < 0)
    y = lnw_local(c[good])
    u = (gz[inside] - gx[inside])[good] - np.interp(lnCg[inside][good], lc, lMs)
    X = np.column_stack([u, u * u])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1.0 / (1.0 + b[0]))


def param_path(conv, wfun, a1, b1, C_levels, xlo, xhi):
    """Parametric compute-optimal path for the same compute levels: root of ln w_param - ln w_target on each isocost
    (searched over [xlo, xhi] widened by 3 log points). Returns (x*, sigma at the path, lnM*)."""
    rows = []
    for C in C_levels:
        lnC = np.log(C)

        def g(xx):
            return float(np.log(wfun(np.exp(xx), np.exp(iso_z(conv, lnC, xx)))) - path_target(conv, np.array([xx]))[0])

        try:
            x0 = brentq(g, xlo - 3, xhi + 3, xtol=1e-8)
        except ValueError:
            rows.append((np.nan, np.nan, np.nan))
            continue
        w = float(np.exp(path_target(conv, np.array([x0]))[0]))
        rows.append((x0, float(sigma_at_w(a1, b1, w)), float(iso_z(conv, lnC, x0) - x0)))
    return np.array(rows, float)


# ============================================================================ corpus-pair (CE) model
def fit_ce(N, D, L, g, est="huber", starts=None):
    """Common exponents, corpus-specific (ln A, ln B, ln E); g = 0 (web), 1 (edu). Returns theta, objective.
    theta = [alpha, beta, lnA_web, lnA_edu, lnB_web, lnB_edu, lnE_web, lnE_edu]."""
    x, z, y = np.log(N), np.log(D), np.log(L)
    th, f, _ = me.fit_panel_ls("CE", x, z, y, np.asarray(g, int), 2, est, starts=starts)
    return th, f


def ce_starts(N, D, L, g, est="huber"):
    """Starts for the CE model from separate Chinchilla fits per corpus (m2_est.embed)."""
    ths = []
    for r in (0, 1):
        s = np.asarray(g) == r
        ths.append(fit_chin(N[s], D[s], L[s], est)[0])
    P = np.vstack([ths[r] for r in np.asarray(g, int)])
    st = me.embed("CE", P, np.asarray(g, int), 2)
    return [st], ths


def ce_tilt(th):
    """chi = Delta ln(A/B), edu - web."""
    return float((th[3] - th[5]) - (th[2] - th[4]))


def pred_ce(th, N, D, g):
    g = np.asarray(g, int)
    lnA = np.where(g == 1, th[3], th[2])
    lnB = np.where(g == 1, th[5], th[4])
    lnE = np.where(g == 1, th[7], th[6])
    return np.logaddexp(np.logaddexp(lnA - th[0] * np.log(N), lnB - th[1] * np.log(D)), lnE)


def ce_corpus_theta(th, r):
    """Chinchilla theta (lnA, lnB, lnE, alpha, beta) of corpus r (0 web, 1 edu) under the CE fit."""
    return np.array([th[2 + r], th[4 + r], th[6 + r], th[0], th[1]])


def mstar_ratio(th_edu, th_web, conv, C_levels):
    """M*_edu / M*_web on each corpus's compute-optimal path at the same compute (numerical; P uses actual FLOPs)."""
    out = []
    for C in C_levels:
        vals = []
        for th in (th_edu, th_web):
            pp = param_path(conv, lambda N, D, th=th: w_chin(th, N, D), th[3], th[4], [C], 10.0, 20.0)
            vals.append(pp[0, 2])
        out.append(float(np.exp(vals[0] - vals[1])))
    return np.array(out)


# ============================================================================ Q3 statistic
def q3_stat(N, D, M, lnw_loc, th_r, lnM_min=np.log(100.0)):
    """Delta_i = ln w_restricted - ln w_local at endpoints with M > 100; OLS slope of Delta on ln M (with intercept)."""
    sel = np.log(M) > lnM_min
    dl = np.log(w_chin(th_r, N[sel], D[sel])) - lnw_loc[sel]
    lm = np.log(M[sel])
    ok = np.isfinite(dl)
    if ok.sum() < 3:
        return dict(slope=np.nan, intercept=np.nan, mean=np.nan, n=int(ok.sum()))
    X = np.column_stack([np.ones(ok.sum()), lm[ok] - np.log(100.0)])
    b = np.linalg.lstsq(X, dl[ok], rcond=None)[0]
    return dict(slope=float(b[1]), intercept=float(b[0]), mean=float(np.mean(dl[ok])), n=int(ok.sum()))


# ============================================================================ Jacobian conditioning (Q6)
def cond_jac(th, N, D):
    """Condition number of J'J for the Chinchilla LSE parameterization at theta, raw and KMW-normalized
    (m6 mc_lib.design_conditioning evaluated at the fitted theta instead of the Monte Carlo truth)."""
    import mc_lib as ml
    lN, lD = np.log(N), np.log(D)
    J = ml.jac_lse(np.asarray(th, float), lN, lD)
    raw = np.linalg.cond(J.T @ J)
    Jn = J.copy()
    Jn[:, 3] = -(lN - lN.mean()) * J[:, 0]
    Jn[:, 4] = -(lD - lD.mean()) * J[:, 1]
    return float(raw), float(np.linalg.cond(Jn.T @ Jn))


def fit_ce_null(N, D, L, g, th_ce, est="huber", tight=True):
    """Corpus-pair model under H0: chi = 0 (m2 'hicksE': common exponents, A_r = e^-lam_r A, B_r = e^-lam_r B,
    corpus-specific E), started from the CE solution. Returns (theta0, fitted ln L). Used by the wild cluster
    RESTRICTED bootstrap test of chi = 0 (null imposed on the bootstrap population)."""
    x, z, y = np.log(N), np.log(D), np.log(L)
    g = np.asarray(g, int)
    P = me.panel_perobs("CE", th_ce, g, 2)
    st0 = me.embed("hicksE", P, g, 2)
    th0, _, _ = me.fit_panel_ls("hicksE", x, z, y, g, 2, est, starts=[st0], tight=tight)
    return th0, me.panel_pred("hicksE", th0, x, z, g, 2)


# ============================================================================ leverage-adjusted (CR2) residuals
def smoother_matrix(x, z, hx, hz):
    """Rows of the local-quadratic smoother S (pilot = S f) at the design points."""
    n = len(x)
    S = np.zeros((n, n))
    for i in range(n):
        w = np.exp(-0.5 * (((x - x[i]) / hx) ** 2 + ((z - z[i]) / hz) ** 2))
        X = np.column_stack([np.ones(n), x - x[i], z - z[i], (x - x[i]) ** 2, (z - z[i]) ** 2, (x - x[i]) * (z - z[i])])
        WX = X * w[:, None]
        S[i] = np.linalg.solve(X.T @ WX + 1e-12 * np.eye(6), WX.T)[0]
    return S


def hat_from_jac(J):
    """Gauss-Newton hat matrix J (J'J)^+ J' of a nonlinear least-squares-type fit."""
    return J @ np.linalg.pinv(J.T @ J, rcond=1e-12) @ J.T


def num_jac(fun, th, eps=1e-6):
    th = np.asarray(th, float)
    f0 = fun(th)
    J = np.zeros((len(f0), len(th)))
    for j in range(len(th)):
        tp = th.copy()
        tp[j] += eps
        tm = th.copy()
        tm[j] -= eps
        J[:, j] = (fun(tp) - fun(tm)) / (2 * eps)
    return J


def jac_chin(th, N, D):
    import mc_lib as ml
    return ml.jac_lse(np.asarray(th, float), np.log(N), np.log(D))


def jac_kappa(p, N, D):
    return num_jac(lambda q: pred_kappa(q, N, D), p)


def jac_ce(th, N, D, g):
    return num_jac(lambda q: pred_ce(q, N, D, g), th)


def cr2_residuals(e, H, clus):
    """Bell-McCaffrey (CR2) cluster residuals: e_g <- (I - H_gg)^(-1/2) e_g, with the symmetrized block and its
    eigenvalues floored at 0.01 (so a near-interpolating fit cannot blow a cluster up by more than 10x)."""
    e = np.asarray(e, float).copy()
    out = e.copy()
    clus = np.asarray(clus)
    for c in np.unique(clus):
        idx = np.where(clus == c)[0]
        Mg = np.eye(len(idx)) - H[np.ix_(idx, idx)]
        Mg = 0.5 * (Mg + Mg.T)
        lam, V = np.linalg.eigh(Mg)
        lam = np.clip(lam, 0.01, None)
        out[idx] = V @ np.diag(lam ** -0.5) @ V.T @ e[idx]
    return out
