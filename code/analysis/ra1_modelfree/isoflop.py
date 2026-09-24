"""isoflop.py -- model-free sigma* on IsoFLOP designs (Task 1; R1 comment 7, R2 Major 8, R3 M6, R4 M3).

Identity (verified numerically in `verify_identity`): at a compute-optimal point of ANY smooth technology L(n, d),
    1/sigma* - 1 = L_nn|_C / (2 |dL*/d ln C|) =: S/2,
the curvature of the IsoFLOP loss profile in ln N at its minimum over twice the slope of the loss-compute frontier.
Both numerator and denominator scale by g'(L*) under a monotone transform g of the loss, so the ratio needs no E,
no outer exponent and no functional form (and no knowledge of the loss units).

Estimator, per IsoFLOP budget b (design-conditional; the experimenter's (N, C_b) grid is held fixed):
  1. local polynomial of L in x = ln N (order p = 2 or 3) on a symmetric window |x - x*_b| <= h around the
     profile argmin x*_b, iterated until the window is stable (limits finite-grid/asymmetry bias, Czech et al. 2026);
  2. curvature kappa_b = L''(x*_b), minimum L*_b = L(x*_b); budget valid iff kappa_b > 0, the argmin lies inside
     the window's range and at least two runs lie on each side of it within the window;
  3. frontier slope s_b = dL*/d ln C at c_b from a smooth fit of L*_b on c_b (primary: quadratic in c);
  4. S_b = kappa_b/|s_b|, sigma*_b = 2/(2 + S_b); design summary: precision-weighted mean of S_b (fixed weights
     1/rse(S*_b)^2), DerSimonian-Laird random-effects mean across budgets, and a drift test (non-homotheticity).
Inference: wild bootstrap with the design and the windows held fixed: L*_i = Lhat_i + v_i r~_i + eta_b e~F_b, with
r~ the HC2-adjusted within-window residuals (Rademacher v_i) and e~F_b the leverage-adjusted residual of budget b's
minimum around the frontier fit (Webb weights eta_b; budget-level shocks such as a budget's own schedule).
"""
from __future__ import annotations



import numpy as np
import pandas as pd
from scipy.stats import f as fdist

import ra1_common as rc

H_GRID = (0.6, 0.8, 1.0, 1.25, 1.5, np.inf)
H_PRIMARY = 1.0
B_ISO = 999
if rc.QUICK:
    B_ISO = 99


# ============================================================================ per-budget local polynomial
def _argmin_poly(c):
    """c = (c0, c1, c2[, c3]) in powers of t. Local minimum t* and curvature f''(t*); (nan, nan) if none."""
    if len(c) == 3 or abs(c[3]) < 1e-14:
        if c[2] <= 0:
            return np.nan, np.nan
        return -c[1] / (2 * c[2]), 2 * c[2]
    c0, c1, c2, c3 = c
    disc = 4 * c2 * c2 - 12 * c3 * c1
    if disc < 0:
        return np.nan, np.nan
    r = np.array([(-2 * c2 + np.sqrt(disc)) / (6 * c3), (-2 * c2 - np.sqrt(disc)) / (6 * c3)])
    f2 = 2 * c2 + 6 * c3 * r
    k = np.where(f2 > 0)[0]
    if len(k) == 0:
        return np.nan, np.nan
    j = k[np.argmin(np.abs(r[k]))]
    return r[j], f2[j]


def _polyval(c, t):
    return sum(ci * t ** i for i, ci in enumerate(c))


def _vander(t, p):
    return np.column_stack([t ** i for i in range(p + 1)])


def fit_budget(x, L, order=2, h=H_PRIMARY, maxit=30):
    """Iterated symmetric-window local polynomial for one IsoFLOP profile. Returns a dict (see module doc)."""
    x, L = np.asarray(x, float), np.asarray(L, float)
    out = dict(ok=False, reason="", order=order, h=h, n_budget=len(x))
    if len(np.unique(np.round(x, 8))) < order + 2:
        out["reason"] = "too few distinct sizes"
        return out
    # preliminary argmin: global quadratic if convex with interior argmin, else the best run
    c2 = np.polyfit(x - x.mean(), L, 2)[::-1]
    t0, k0 = _argmin_poly(c2)
    x0 = x.mean() + t0 if (np.isfinite(t0) and x.min() <= x.mean() + t0 <= x.max()) else x[np.argmin(L)]
    for _ in range(maxit):
        m = np.abs(x - x0) <= h + 1e-9
        if m.sum() < order + 2 or len(np.unique(np.round(x[m], 8))) < order + 2:
            out["reason"] = "too few runs in window"
            return out
        c = np.linalg.lstsq(_vander(x[m] - x0, order), L[m], rcond=None)[0]
        t, kap = _argmin_poly(c)
        if not np.isfinite(t):
            out["reason"] = "no interior minimum (non-convex)"
            return out
        x_new = x0 + t
        m_new = np.abs(x - x_new) <= h + 1e-9
        x0 = x_new
        if np.array_equal(m_new, m):
            break
    m = np.abs(x - x0) <= h + 1e-9
    if m.sum() < order + 2 or len(np.unique(np.round(x[m], 8))) < order + 2:
        out["reason"] = "too few runs in window"
        return out
    xc = x0
    V = _vander(x[m] - xc, order)
    P = np.linalg.pinv(V)
    c = P @ L[m]
    t, kap = _argmin_poly(c)
    if not np.isfinite(t):
        out["reason"] = "no interior minimum (non-convex)"
        return out
    xs = xc + t
    fitted = V @ c
    Hm = V @ P
    lev = np.clip(np.diag(Hm), 0, 1)
    nl, nr = int(np.sum(x[m] < xs - 1e-9)), int(np.sum(x[m] > xs + 1e-9))
    out.update(xstar=xs, Lstar=_polyval(c, t), curv=kap, mask=m, xc=xc, P=P, V=V, coef=c, fitted=fitted, lev=lev,
               resid=L[m] - fitted, nwin=int(m.sum()), nleft=nl, nright=nr, xlo=float(x[m].min()), xhi=float(x[m].max()),
               rss=float(np.sum((L[m] - fitted) ** 2)), dof=int(m.sum() - order - 1))
    if kap <= 0:
        out["reason"] = "non-positive curvature"
    elif not (x[m].min() <= xs <= x[m].max()):
        out["reason"] = "argmin outside window range"
    elif nl < 2 or nr < 2:
        out["reason"] = "argmin not bracketed (<2 runs on a side)"
    else:
        out["ok"] = True
    return out


def fit_budget_centered(x, L, order, h, x0):
    """Local polynomial on the window |x - x0| <= h around a FIXED centre x0 (no data-driven re-centring, hence no
    'winner's curse' from centring on the noisy minimum). 'defined' = enough distinct runs in the window; 'ok' =
    defined, positive curvature and an interior minimum with >= 2 runs on each side (point-estimate validity).
    curv: L'' at the interior minimum (quadratic: 2 c2 everywhere); if there is no interior minimum (possible in
    bootstrap draws) the curvature at the window centre. Lstar: minimum of the fitted polynomial over the window."""
    x, L = np.asarray(x, float), np.asarray(L, float)
    out = dict(ok=False, defined=False, reason="", order=order, h=h, n_budget=len(x), x0=x0)
    m = np.abs(x - x0) <= h + 1e-9
    if m.sum() < order + 2 or len(np.unique(np.round(x[m], 8))) < order + 2:
        out["reason"] = "too few runs in window"
        return out
    V = _vander(x[m] - x0, order)
    P = np.linalg.pinv(V)
    c = P @ L[m]
    t, kap = _argmin_poly(c)
    tlo, thi = float(x[m].min() - x0), float(x[m].max() - x0)
    interior = np.isfinite(t) and tlo <= t <= thi
    if not interior:
        tt = np.linspace(tlo, thi, 201)
        t = float(tt[np.argmin(_polyval(c, tt))])
        kap = float(2 * c[2] + (6 * c[3] * t if order == 3 else 0.0)) if order == 3 else float(2 * c[2])
    xs = x0 + t
    fitted = V @ c
    lev = np.clip(np.diag(V @ P), 0, 1)
    nl, nr = int(np.sum(x[m] < xs - 1e-9)), int(np.sum(x[m] > xs + 1e-9))
    out.update(defined=True, xstar=xs, Lstar=float(_polyval(c, t)), curv=float(kap), mask=m, xc=x0, coef=c,
               fitted=fitted, lev=lev, resid=L[m] - fitted, nwin=int(m.sum()), nleft=nl, nright=nr,
               xlo=float(x[m].min()), xhi=float(x[m].max()), rss=float(np.sum((L[m] - fitted) ** 2)),
               dof=int(m.sum() - order - 1), interior=bool(interior))
    if not interior:
        out["reason"] = "no interior minimum in the window"
    elif kap <= 0:
        out["reason"] = "non-positive curvature"
    elif nl < 2 or nr < 2:
        out["reason"] = "argmin not bracketed (<2 runs on a side)"
    else:
        out["ok"] = True
    return out


def prep(df):
    """Per-budget arrays (fast repeated estimation in the bootstrap)."""
    out = []
    for b, g in df.groupby("b"):
        out.append(dict(b=b, idx=g.index.values, x=g["x"].values.astype(float), c=float(g["c"].iloc[0]),
                        C=float(g["C_b"].iloc[0])))
    return out


def path_centres(P, L):
    """Window centres from a pooled Approach-2 path: preliminary argmins of a global quadratic in each budget
    (convex, interior), then a Theil-Sen line of x0_b on c_b; returns ({budget: centre}, slope)."""
    from scipy.stats import theilslopes
    cs, xs = [], []
    for B_ in P:
        x, Lb = B_["x"], L[B_["idx"]]
        if len(np.unique(np.round(x, 8))) < 3:
            continue
        xm = x.mean()
        c = np.polyfit(x - xm, Lb, 2)[::-1]
        t, k = _argmin_poly(c)
        if np.isfinite(t) and x.min() <= xm + t <= x.max():
            cs.append(B_["c"])
            xs.append(xm + t)
    cs, xs = np.array(cs), np.array(xs)
    if len(cs) >= 3:
        sl_, ic, _, _ = theilslopes(xs, cs)
    else:
        sl_, ic = 0.5, np.mean(xs) - 0.5 * np.mean(cs)
    return {B_["b"]: ic + sl_ * B_["c"] for B_ in P}, float(sl_)


def refit_fixed(fb, Lwin):
    """Re-estimate one budget on new losses at the SAME window/centre/order (bootstrap)."""
    c = fb["P"] @ Lwin
    t, kap = _argmin_poly(c)
    if not np.isfinite(t) or kap <= 0:
        return np.nan, np.nan, np.nan
    xs = fb["xc"] + t
    if not (fb["xlo"] <= xs <= fb["xhi"]):
        return np.nan, np.nan, np.nan
    return xs, _polyval(c, t), kap


# ============================================================================ frontier slope
def frontier(cb, Ls, kind="quad", hc=1.5):
    """Slope dL*/dc at each c_b, fitted values, leverage. kinds: quad, cubic (polynomials in c - mean),
    loclin (Gaussian-kernel local linear, bandwidth hc in ln C), power (E + K exp(-g c): NOT E-free; check only)."""
    cb, Ls = np.asarray(cb, float), np.asarray(Ls, float)
    k = len(cb)
    cc = cb - cb.mean()
    if kind in ("quad", "cubic"):
        p = 2 if kind == "quad" else 3
        if k < p + 2:
            return None
        V = _vander(cc, p)
        P = np.linalg.pinv(V)
        b = P @ Ls
        slope = sum(i * b[i] * cc ** (i - 1) for i in range(1, p + 1))
        fit = V @ b
        lev = np.diag(V @ P)
        return dict(slope=slope, fit=fit, lev=lev, resid=Ls - fit)
    if kind in ("logquad", "logcubic"):
        # polynomial in c of ln L*: E-free smoother of the frontier; slope in levels dL*/dc = L* d ln L*/dc
        p = 2 if kind == "logquad" else 3
        if k < p + 2:
            return None
        V = _vander(cc, p)
        P = np.linalg.pinv(V)
        b = P @ np.log(Ls)
        dz = sum(i * b[i] * cc ** (i - 1) for i in range(1, p + 1))
        fit = np.exp(V @ b)
        lev = np.diag(V @ P)
        return dict(slope=Ls * dz, fit=fit, lev=lev, resid=Ls - fit)
    if kind == "loclin":
        slope, fit, lev = np.zeros(k), np.zeros(k), np.zeros(k)
        for j in range(k):
            w = np.exp(-0.5 * ((cb - cb[j]) / hc) ** 2)
            X = np.column_stack([np.ones(k), cb - cb[j]])
            WX = X * w[:, None]
            A = np.linalg.solve(X.T @ WX, WX.T)
            bb = A @ Ls
            slope[j], fit[j], lev[j] = bb[1], bb[0], A[0, j]
        return dict(slope=slope, fit=fit, lev=lev, resid=Ls - fit)
    if kind == "power":
        from scipy.optimize import least_squares
        f = lambda q: q[0] + np.exp(q[1] - q[2] * cc) - Ls
        best = None
        for E0 in (0.0, 0.5 * Ls.min(), 0.9 * Ls.min()):
            try:
                r = least_squares(f, [E0, np.log(max(Ls.mean() - E0, 1e-3)), 0.1],
                                  bounds=([0, -50, 1e-4], [0.999 * Ls.min(), 50, 3]))
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        E, lk, g = best.x
        slope = -g * np.exp(lk - g * cc)
        fit = E + np.exp(lk - g * cc)
        return dict(slope=slope, fit=fit, lev=np.full(k, 3.0 / k), resid=Ls - fit, E=E, gamma=g)
    raise ValueError(kind)


# ============================================================================ design-level estimate
def fkind_auto(k):
    """Primary frontier smoother: cubic in c of ln L* with >= 6 valid budgets, quadratic otherwise."""
    return "logcubic" if k >= 6 else "logquad"


def design_estimate(df, order=2, h=H_PRIMARY, fkind="auto", center="path", use_budgets=None, P=None, L=None):
    """center = 'path' (primary: windows centred on a pooled Approach-2 path) or 'iter' (windows iterated around
    each budget's own argmin; sensitivity). use_budgets (bootstrap): evaluate exactly these budgets whenever the
    window polynomial is defined (validity is judged once, at the point estimate). P, L: prepared arrays and a loss
    vector (fast path for the bootstrap)."""
    if P is None:
        P = prep(df)
    if L is None:
        L = df["L"].values.astype(float)
    fits = {}
    centres, pslope = path_centres(P, L) if center == "path" else ({}, np.nan)
    for B_ in P:
        b = B_["b"]
        if use_budgets is not None and b not in use_budgets:
            continue
        Lb = L[B_["idx"]]
        if center == "path":
            fb = fit_budget_centered(B_["x"], Lb, order, h, centres[b])
        else:
            fb = fit_budget(B_["x"], Lb, order, h)
            fb["defined"] = "curv" in fb
        fb["c"], fb["C"], fb["idx"] = B_["c"], B_["C"], B_["idx"]
        fits[b] = fb
    if use_budgets is None:
        valid = [b for b in sorted(fits) if fits[b]["ok"]]
    else:
        valid = [b for b in use_budgets if b in fits and fits[b].get("defined")]
    res = dict(fits=fits, valid=valid, order=order, h=h, fkind=fkind, center=center, path_slope=pslope)
    if len(valid) < 4:
        res["ok"] = False
        return res
    cb = np.array([fits[b]["c"] for b in valid])
    Ls = np.array([fits[b]["Lstar"] for b in valid])
    if fkind == "auto":
        fkind = fkind_auto(len(valid))
        res["fkind"] = fkind
    fr = frontier(cb, Ls, fkind)
    if fr is None:
        res["ok"] = False
        return res
    curv = np.array([fits[b]["curv"] for b in valid])
    S = curv / np.abs(fr["slope"])
    res.update(ok=bool(np.all(fr["slope"] < 0)), cb=cb, Ls=Ls, curv=curv, slope=fr["slope"], S=S,
               sigma=2.0 / (2.0 + S), fr=fr)
    return res


def pooled(S, w):
    w = np.asarray(w, float)
    return float(np.sum(w * S) / np.sum(w))


def dersimonian_laird(y, v):
    """DerSimonian-Laird random-effects mean; returns mu, se, tau2, Q, df, I2, p_Q, HKSJ se."""
    from scipy.stats import chi2
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1 / v
    mu_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu_f) ** 2))
    df = k - 1
    tau2 = max(0.0, (Q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (v + tau2)
    mu = np.sum(ws * y) / np.sum(ws)
    se = np.sqrt(1 / np.sum(ws))
    q_hk = np.sum(ws * (y - mu) ** 2) / df if df > 0 else np.nan
    se_hk = np.sqrt(q_hk / np.sum(ws)) if df > 0 else np.nan
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return dict(mu_fixed=float(mu_f), se_fixed=float(np.sqrt(1 / np.sum(w))), mu=float(mu), se=float(se),
                tau2=float(tau2), tau=float(np.sqrt(tau2)), Q=Q, df=df, p_Q=float(chi2.sf(Q, df)) if df > 0 else np.nan,
                I2=float(I2), se_hksj=float(se_hk), k=k)


# ============================================================================ wild bootstrap (design-conditional)
H_SMOOTH = 0.5


def smooth_dgp(df, hs=H_SMOOTH):
    """Bootstrap population: within each budget, a Gaussian-kernel local quadratic of L in ln N (bandwidth hs)
    evaluated at every run (linear smoother H). Residuals are rescaled by the smoother's degrees-of-freedom factor
    sqrt(n / (n - 2 tr H + tr H'H)) so that their variance matches the noise variance (Hastie-Tibshirani).
    Returns fitted values and rescaled residuals aligned with df."""
    fit = np.zeros(len(df))
    rt = np.zeros(len(df))
    for b, g in df.groupby("b"):
        x, L = g["x"].values, g["L"].values
        idx = g.index.values
        n = len(x)
        H = np.zeros((n, n))
        for j in range(n):
            w = np.exp(-0.5 * ((x - x[j]) / hs) ** 2)
            X = _vander(x - x[j], 2 if n >= 4 else 1)
            WX = X * w[:, None]
            try:
                A = np.linalg.solve(X.T @ WX, WX.T)
            except np.linalg.LinAlgError:
                A = np.linalg.pinv(X.T @ WX) @ WX.T
            H[j] = A[0]
        f = H @ L
        dfres = n - 2 * np.trace(H) + np.trace(H.T @ H)
        scale = np.sqrt(n / dfres) if dfres > 0.5 else np.sqrt(n / 0.5)
        fit[idx] = f
        rt[idx] = (L - f) * min(scale, 3.0)
    return fit, rt


def boot_design(df, res, B, seed, budget_shock=True, run_col=None, run_sd=0.0, return_pop=False):
    """Design-conditional wild bootstrap with FULL re-estimation (window centres, windows, polynomials, frontier):
    L*_i = Ltilde_i + v_i r~_i + eta_b e~F_b [+ z_run(i)], v Rademacher, eta Webb. Ltilde = smooth_dgp (population).
    Returns draws of S_b* for the point-estimate's valid budgets and S0_b (estimator on the noise-free population),
    used to centre basic intervals."""
    rng = np.random.default_rng(seed)
    Lt, rt = smooth_dgp(df)
    valid = res["valid"]
    k = len(valid)
    fr = res["fr"]
    eF = fr["resid"] / np.sqrt(np.clip(1 - fr["lev"], 0.1, 1))
    bpos = {b: j for j, b in enumerate(valid)}
    brow = df["b"].values
    shock_idx = np.array([bpos.get(b, -1) for b in brow])
    runs = None
    if run_col is not None and run_sd > 0:
        ru, runs = np.unique(df[run_col].values, return_inverse=True)

    P = prep(df)

    def est(Lvec):
        r2 = design_estimate(None, res["order"], res["h"], res["fkind"], res["center"], use_budgets=valid, P=P, L=Lvec)
        S = np.full(k, np.nan)
        cv = np.full(k, np.nan)
        Ls = np.full(k, np.nan)
        slp = np.full(k, np.nan)
        if "S" in r2:
            for j2, b in enumerate(r2["valid"]):
                if b in bpos:
                    S[bpos[b]], cv[bpos[b]], Ls[bpos[b]], slp[bpos[b]] = r2["S"][j2], r2["curv"][j2], r2["Ls"][j2], r2["slope"][j2]
        return S, cv, Ls, slp

    S0, cv0, Ls0, sl0 = est(Lt)
    Sd = np.full((B, k), np.nan)
    Lsd, curvd, slopes = Sd.copy(), Sd.copy(), Sd.copy()
    for i in range(B):
        v = rc.rademacher(rng, len(df))
        Lb = Lt + v * rt
        if budget_shock:
            eta = rc.webb(rng, k)
            sh = np.where(shock_idx >= 0, eta[np.clip(shock_idx, 0, None)] * eF[np.clip(shock_idx, 0, None)], 0.0)
            Lb = Lb + sh
        if runs is not None:
            Lb = Lb + rng.normal(0.0, run_sd, len(ru))[runs]
        Sd[i], curvd[i], Lsd[i], slopes[i] = est(Lb)
    return dict(S=Sd, Ls=Lsd, curv=curvd, slope=slopes, S0=S0, curv0=cv0, slope0=sl0)


def _basic(est, draws, pop):
    """Basic bootstrap interval: est - (quantiles of draws - pop)."""
    d = np.asarray(draws, float) - pop
    d = d[np.isfinite(d)]
    if len(d) < 5 or not np.isfinite(pop):
        return np.nan, np.nan, np.nan
    return float(np.std(d, ddof=1)), est - np.percentile(d, 97.5), est - np.percentile(d, 2.5)


def _ratio_pool(curv, slope, om):
    ok = np.isfinite(curv) & np.isfinite(slope)
    if ok.sum() < 3:
        return np.nan
    return float(np.sum(om[ok] * curv[ok]) / np.sum(om[ok] * np.abs(slope[ok])))


def summarize_design(name, res, bt, B):
    """Per-budget rows and the design-level row.
    Pooled S = sum_b om_b kappa_b / sum_b om_b |s_b| (ratio of weighted sums; om_b = |s_b| / Var*(kappa_b), so that
    the effective weight on S_b = kappa_b/|s_b| is its inverse variance; the weights depend on the design and the
    noise, not on the estimated curvature), sigma* = 2/(2 + S). The ratio of sums is linear in the noisy curvatures,
    which keeps it nearly unbiased when a budget's curvature is imprecise (a mean of logs or an inverse-variance
    mean of ratios is not). Intervals: basic bootstrap, centred at the estimator applied to the noise-free bootstrap
    population. Also: DerSimonian-Laird random effects of S_b across budgets, a bootstrap homogeneity test and the
    drift of S_b in log10 C (non-homotheticity)."""
    valid = res["valid"]
    S, Sd, S0 = res["S"], bt["S"], bt["S0"]
    k = len(valid)
    curv, slope = res["curv"], res["slope"]
    cd, sd_ = bt["curv"], bt["slope"]
    c0 = bt.get("curv0")
    s0 = bt.get("slope0")
    vk = np.array([rc.rse(cd[:, j]) ** 2 for j in range(k)])
    om = np.abs(slope) / vk
    Sbar = _ratio_pool(curv, slope, om)
    Sbar0 = _ratio_pool(c0, s0, om)
    Sbar_d = np.array([_ratio_pool(cd[i], sd_[i], om) for i in range(len(cd))])
    good = np.isfinite(Sbar_d)
    sig = lambda x: 2.0 / (2.0 + x)
    dsig = lambda x: -2.0 / (2.0 + x) ** 2
    se_S, lo_S, hi_S = _basic(Sbar, Sbar_d, Sbar0)
    vS = np.array([rc.rse(Sd[:, j]) ** 2 for j in range(k)])
    wS = 1 / vS
    lc = res["cb"] / np.log(10)

    def wls_slope(y, ww, xx):
        ok = np.isfinite(y)
        if ok.sum() < 4:
            return np.nan
        X = np.column_stack([np.ones(ok.sum()), xx[ok] - xx[ok].mean()])
        W = ww[ok]
        return float(np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * y[ok]))[1])

    dr = wls_slope(S, wS, lc)
    dr0 = wls_slope(S0, wS, lc)
    dr_d = np.array([wls_slope(Sd[i], wS, lc) for i in range(len(Sd))])
    se_dr, lo_dr, hi_dr = _basic(dr, dr_d, dr0)
    dd = dr_d[np.isfinite(dr_d)] - dr0
    p_drift = float((1 + np.sum(np.abs(dd) >= abs(dr))) / (len(dd) + 1)) if len(dd) else np.nan
    Sw = float(np.sum(wS * S) / np.sum(wS))
    Sw0 = float(np.sum(wS * S0) / np.sum(wS))
    Q = float(np.sum(wS * (S - Sw) ** 2))
    Qd = []
    for i in range(len(Sd)):
        if not np.all(np.isfinite(Sd[i])) or not np.all(np.isfinite(S0)):
            continue
        dev = (Sd[i] - S0) - (np.sum(wS * Sd[i]) / np.sum(wS) - Sw0)
        Qd.append(np.sum(wS * dev ** 2))
    Qd = np.array(Qd)
    pQ = (1 + np.sum(Qd >= Q)) / (len(Qd) + 1) if len(Qd) else np.nan
    dl = dersimonian_laird(S, vS)
    rows = []
    for j, b in enumerate(valid):
        fb = res["fits"][b]
        se_j, lo_j, hi_j = _basic(S[j], Sd[:, j], S0[j])
        rows.append(dict(design=name, budget_C=fb["C"], c=fb["c"], n_budget=fb["n_budget"], nwin=fb["nwin"],
                         nleft=fb["nleft"], nright=fb["nright"], xstar=fb["xstar"], Nstar=np.exp(fb["xstar"]),
                         Mstar=fb["C"] / (6 * np.exp(fb["xstar"]) ** 2), Lstar=fb["Lstar"], curv=fb["curv"],
                         slope=slope[j], S=S[j], sigma=float(sig(S[j])), sigma_pop=float(sig(S0[j])),
                         se_sigma=float(abs(dsig(S[j])) * se_j), lo_sigma=float(sig(hi_j)), hi_sigma=float(sig(lo_j)),
                         se_curv=float(np.nanstd(cd[:, j], ddof=1)), se_slope=float(np.nanstd(sd_[:, j], ddof=1)),
                         weight=float(om[j] * abs(slope[j]) / np.sum(om * np.abs(slope))),
                         share_nonconvex_draws=float(np.mean(cd[:, j] <= 0))))
    invalid = [dict(design=name, budget_C=res["fits"][b]["C"], reason=res["fits"][b]["reason"],
                    n_budget=res["fits"][b]["n_budget"]) for b in sorted(res["fits"]) if b not in valid]
    summ = dict(design=name, order=res["order"], h=res["h"], fkind=res["fkind"], center=res["center"], k_valid=k,
                k_budgets=len(res["fits"]), Cmin=float(np.exp(res["cb"].min())), Cmax=float(np.exp(res["cb"].max())),
                S_pooled=Sbar, sigma_fe=float(sig(Sbar)), sigma_fe_pop=float(sig(Sbar0)),
                se_sigma_fe=float(abs(dsig(Sbar)) * se_S), lo_sigma_fe=float(sig(hi_S)), hi_sigma_fe=float(sig(lo_S)),
                sigma_median_budget=float(np.median(sig(S))), sigma_min_budget=float(sig(S).min()),
                sigma_max_budget=float(sig(S).max()),
                sigma_re=float(sig(dl["mu"])), se_sigma_re=float(abs(dsig(dl["mu"])) * dl["se"]),
                se_sigma_re_hksj=float(abs(dsig(dl["mu"])) * dl["se_hksj"]),
                tau_S=dl["tau"], tau_sigma=float(abs(dsig(dl["mu"])) * dl["tau"]), I2=dl["I2"],
                Q=Q, p_Q_boot=float(pQ), p_Q_chi2=dl["p_Q"],
                drift_S_per_decade=dr, se_drift_S=se_dr,
                drift_sigma_per_decade=float(dsig(Sbar) * dr), se_drift_sigma=float(abs(dsig(Sbar)) * se_dr),
                lo_drift_sigma=float(dsig(Sbar) * hi_dr), hi_drift_sigma=float(dsig(Sbar) * lo_dr),
                p_drift=p_drift, B=B, B_pooled=int(good.sum()),
                share_draws_nonconvex_budget=float(np.mean(np.any(cd <= 0, axis=1))))
    return summ, rows, invalid, sig(Sbar_d[good] - Sbar0 + Sbar)


# ============================================================================ order selection (pooled F test)
def order_test(df, h):
    """Pooled F test of the cubic term across budgets, on each budget's quadratic window (h)."""
    r2 = r3 = 0.0
    k = dof3 = 0
    centres, _ = path_centres(prep(df), df["L"].values.astype(float))
    for b, g in df.groupby("b"):
        fb = fit_budget_centered(g["x"].values, g["L"].values, 2, h, centres[b])
        if not fb["ok"] or fb["nwin"] < 6:
            continue
        x = g["x"].values[fb["mask"]] - fb["xc"]
        y = g["L"].values[fb["mask"]]
        c3 = np.linalg.lstsq(_vander(x, 3), y, rcond=None)[0]
        rss3 = float(np.sum((y - _vander(x, 3) @ c3) ** 2))
        r2 += fb["rss"]
        r3 += rss3
        k += 1
        dof3 += len(y) - 4
    if k == 0 or dof3 <= 0 or r3 <= 0:
        return dict(F=np.nan, p=np.nan, k=k, dof=dof3)
    F = ((r2 - r3) / k) / (r3 / dof3)
    return dict(F=float(F), p=float(fdist.sf(F, k, dof3)), k=k, dof=dof3)


# ============================================================================ identity check (numerical)
def verify_identity():
    """1/sigma* - 1 = L_nn|_C/(2|dL*/dlnC|) against the analytic sigma* (Chinchilla, kappa family) and against
    the Hicks elasticity computed from full derivatives of ln L (Farseer Eq. 3, non-homothetic)."""
    import m2_est as me
    from scipy.optimize import minimize_scalar

    def prof_min(fL, c):
        r = minimize_scalar(lambda n: fL(n, c - n), bounds=(c / 2 - 8, c / 2 + 8), method="bounded",
                            options=dict(xatol=1e-12))
        return r.x, r.fun

    def mf(fL, C, h=1e-3):
        c = np.log(C / 6)
        n0, _ = prof_min(fL, c)
        Lnn = (fL(n0 + h, c - n0 - h) - 2 * fL(n0, c - n0) + fL(n0 - h, c - n0 + h)) / h ** 2
        s = (prof_min(fL, c + h)[1] - prof_min(fL, c - h)[1]) / (2 * h)
        sig_mf = 1 / (1 + Lnn / (2 * abs(s)))
        sig_full = me.numderiv_sigma(lambda a, b: np.log(fL(a, b)), n0, c - n0, h=1e-3)[0]
        # monotone-transform invariance: the same ratio computed on g(L) = ln(L)
        gL = lambda a, b: np.log(fL(a, b))
        Lnn_g = (gL(n0 + h, c - n0 - h) - 2 * gL(n0, c - n0) + gL(n0 - h, c - n0 + h)) / h ** 2
        s_g = (np.log(prof_min(fL, c + h)[1]) - np.log(prof_min(fL, c - h)[1])) / (2 * h)
        sig_g = 1 / (1 + Lnn_g / (2 * abs(s_g)))
        # review check (R1 c7.3): the slope of ln w in ln M along the isocost AT the compute-optimal point equals
        # 1/sigma* - 1 for ANY smooth technology (an identity, not a restriction of the Chinchilla family); only the
        # linearity of ln w in ln(M/M*) away from the path is testable. ln M = c - 2n along the isocost.
        def lnw(a, b, e=1e-4):
            fn = (fL(a + e, b) - fL(a - e, b)) / (2 * e)
            fd = (fL(a, b + e) - fL(a, b - e)) / (2 * e)
            return np.log(fn / fd)
        dlnw = (lnw(n0 + h, c - n0 - h) - lnw(n0 - h, c - n0 + h)) / (-4 * h)
        return sig_mf, sig_full, sig_g, dlnw

    rows = []
    m = rc.sl.BESIROGLU
    fch = lambda n, d: m.loss(np.exp(n), np.exp(d))
    kf = dict(E=1.757136, A=2305.02, B=9957.185, a1=0.424359, b1=0.430527, k=0.774144)  # m1/m2 kappa-free Chinchilla
    fk = lambda n, d: kf["E"] + (kf["A"] * np.exp(-kf["a1"] * n) + kf["B"] * np.exp(-kf["b1"] * d)) ** kf["k"]
    for lab, fL, truth in (("Chinchilla form (Besiroglu et al.)", fch, m.sigma_star),
                           ("kappa family (m2 Chinchilla kappa-free)", fk, 2 / (2 + kf["a1"] + kf["b1"]))):
        for C in (1e19, 1e21, 1e24):
            a, b_, g, dl = mf(fL, C)
            rows.append(dict(technology=lab, C=C, sigma_modelfree=a, sigma_hicks=b_, sigma_modelfree_lnL=g,
                             sigma_analytic=truth, err=a - truth, dlnw_dlnM_at_path=dl,
                             one_over_sigma_minus_1=1 / a - 1))
    return pd.DataFrame(rows), mf


def verify_farseer(thF, mf):
    import m2_est as me
    ff = lambda n, d: float(np.exp(me.farseer_lnL(thF, np.exp(n), np.exp(d))))
    rows = []
    for C in (1e18, 1e19, 1e20, 1e21, 1e22):
        a, b_, g, dl = mf(ff, C)
        rows.append(dict(technology="Farseer Eq. 3 (fitted, non-homothetic)", C=C, sigma_modelfree=a, sigma_hicks=b_,
                         sigma_modelfree_lnL=g, sigma_analytic=np.nan, err=a - b_, dlnw_dlnM_at_path=dl,
                         one_over_sigma_minus_1=1 / a - 1))
    return pd.DataFrame(rows)



# ============================================================================ stage runner
PORIAN_SEED_SD = {"Porian, RefinedWeb": 0.002, "Porian, OpenWebText2": 0.01}   # Porian et al.'s seed-noise sd (m8)
B_SENS = 199
B_PAR = 399
MC_DESIGNS = ("Chinchilla", "Llama 3", "Marin, DCLM")
MC_R, MC_B = 100, 149
if rc.QUICK:
    B_SENS, B_PAR, MC_R, MC_B = 30, 20, 10, 20


def _par_one(i, N, D, yhat_c, rf_c, th, p, seed0):
    """Design-conditional wild (FHH) bootstrap of the parametric sigma* on an IsoFLOP design (kappa = 1 and free)."""
    import m2_est as me
    rng = np.random.default_rng(seed0 + i)
    w = rc.fhh(rng, len(yhat_c))
    Lb = np.exp(yhat_c + w * np.abs(rf_c))
    m = rc.sl.fit_chinchilla(N, D, Lb, delta=1e-3, init=th)
    best = None
    for st in (p, np.r_[m.theta, 1.0]):
        r = me.fit_q(N, D, Lb, "huber", init=st)
        if best is None or r[1] < best[1]:
            best = r
    import parametric as pm
    return dict(sigma_chin=m.sigma_star, sigma_kappa=pm.sigma_star_kappa(best[0]))


def _mc_one(r_i, df, Ltrue, noise_sd, order, h, seed0, center="path", B=None):
    """One Monte Carlo replication: noise-free truth + Gaussian noise (sd = robust residual sd of the design);
    estimate + bootstrap basic interval; returns the pooled sigma* and its interval."""
    rng = np.random.default_rng(seed0 + r_i)
    d = df.copy()
    d["L"] = Ltrue + rng.normal(0.0, noise_sd, len(Ltrue))
    res = design_estimate(d, order, h, "auto", center)
    if not res.get("ok"):
        return {"fail": 1}
    bt = boot_design(d, res, B or MC_B, seed0 + 10_000 + r_i)
    s, _, _, _ = summarize_design("mc", res, bt, B or MC_B)
    return dict(sigma_fe=s["sigma_fe"], lo=s["lo_sigma_fe"], hi=s["hi_sigma_fe"], sigma_re=s["sigma_re"],
                se_re=s["se_sigma_re"], k=s["k_valid"], sigma_med=s["sigma_median_budget"])


def _boot_job(job, D):
    """Worker: one (design, specification) estimate with its design-conditional bootstrap."""
    df = D[job["name"]]
    r = design_estimate(df, job["order"], job["h"], job["fk"], job["cen"])
    if not r.get("ok"):
        return dict(job=job, ok=False, k_valid=len(r["valid"]))
    rsd = PORIAN_SEED_SD.get(job["name"], 0.0)
    bt = boot_design(df, r, job["B"], job["seed"], budget_shock=job["shock"], run_col="width" if rsd > 0 else None,
                     run_sd=rsd)
    s, rows, inv, sd = summarize_design(job["name"], r, bt, job["B"])
    return dict(job=job, ok=True, s=s, rows=rows, inv=inv, sd=sd if job["primary"] else None)


def run(log=rc.log):
    import parametric as pm
    D = rc.isoflop_designs()
    out = dict(summary=[], budgets=[], invalid=[], sens=[], param=[], bias=[], mc=[], order=[], draws={})
    jobs, orders = [], {}
    for name, df in D.items():
        ot = order_test(df, H_PRIMARY)
        order = 3 if (np.isfinite(ot["p"]) and ot["p"] < 0.05) else 2
        orders[name] = order
        out["order"].append(dict(design=name, F=ot["F"], p=ot["p"], k=ot["k"], dof=ot["dof"], order_chosen=order))
        base = dict(name=name, shock=True, primary=False)
        jobs.append(dict(base, order=order, h=H_PRIMARY, fk="auto", cen="path", B=B_ISO, seed=rc.seed_of(f"iso|{name}"),
                         primary=True, kind="primary"))
        jobs.append(dict(base, order=order, h=H_PRIMARY, fk="auto", cen="path", B=B_SENS, seed=rc.seed_of(f"iso0|{name}"),
                         shock=False, kind="within_only"))
        specs = [(2, h, "auto", "path") for h in H_GRID] + [(3, h, "auto", "path") for h in (1.0, 1.5)]
        specs += [(order, H_PRIMARY, fk, "path") for fk in ("logquad", "cubic", "power")]
        specs += [(order, H_PRIMARY, "auto", "iter")]
        for o, h, fk, cen in specs:
            jobs.append(dict(base, order=o, h=h, fk=fk, cen=cen, B=B_SENS, seed=rc.seed_of(f"sens|{name}|{o}|{h}|{fk}|{cen}"),
                             kind="sens"))
    # convention check (R2 Major 9(b)): Marin profiles against the configuration parameter count (not exact isocosts)
    import conventions
    Dc = conventions.config_designs()
    for name in Dc:
        jobs.append(dict(name=name, shock=True, primary=False, order=2, h=H_PRIMARY, fk="auto", cen="path", B=B_SENS,
                         seed=rc.seed_of(f"sens|{name}"), kind="sens"))
    Dall = dict(D, **Dc)
    res = rc.pmap(_boot_job, jobs, dict(D=Dall), chunksize=1)
    out["marin_eta"] = conventions.marin_eta()
    within = {}
    for r in res:
        if "_error" in r:
            log(f"  isoflop job error: {r['_error']}")
            continue
        j = r["job"]
        if j["kind"] == "primary":
            s = r["s"]
            s.update(n_runs=len(D[j["name"]]), run_seed_sd=PORIAN_SEED_SD.get(j["name"], 0.0),
                     **{f"meta_{k}": v for k, v in rc.DESIGN_META[j["name"]].items()})
            out["summary"].append(s)
            out["budgets"] += r["rows"]
            out["invalid"] += r["inv"]
            out["draws"][j["name"]] = r["sd"]
        elif j["kind"] == "within_only":
            within[j["name"]] = r["s"]["se_sigma_fe"] if r["ok"] else np.nan
        else:
            if not r["ok"]:
                out["sens"].append(dict(design=j["name"], order=j["order"], h=j["h"], fkind=j["fk"], center=j["cen"],
                                        k_valid=r["k_valid"], ok=False))
                continue
            s2 = r["s"]
            out["sens"].append(dict(design=j["name"], order=j["order"], h=j["h"], fkind=s2["fkind"], center=j["cen"],
                                    k_valid=s2["k_valid"], ok=True, sigma_fe=s2["sigma_fe"], se_sigma_fe=s2["se_sigma_fe"],
                                    sigma_re=s2["sigma_re"], se_sigma_re=s2["se_sigma_re"],
                                    sigma_median_budget=s2["sigma_median_budget"],
                                    drift_sigma_per_decade=s2["drift_sigma_per_decade"], p_drift=s2["p_drift"],
                                    primary=(j["name"] in orders and j["order"] == orders[j["name"]] and j["h"] == H_PRIMARY
                                             and j["fk"] == "auto" and j["cen"] == "path")))
    for s in out["summary"]:
        s["se_sigma_fe_within_only"] = within.get(s["design"], np.nan)
        log(f"  model-free {s['design']:22s} order {s['order']} ({s['fkind']}): k={s['k_valid']}/{s['k_budgets']} "
            f"RE {s['sigma_re']:.3f} ({s['se_sigma_re']:.3f}) tau {s['tau_sigma']:.3f}; FE {s['sigma_fe']:.3f} "
            f"[{s['lo_sigma_fe']:.3f}, {s['hi_sigma_fe']:.3f}]; drift/decade {s['drift_sigma_per_decade']:+.3f} (p {s['p_drift']:.3f})")
    for name, df in D.items():
        order = orders[name]
        # parametric sigma* on the same runs (C = the budget, D = C/(6N) in the design's convention)
        N = np.exp(df["x"].values)
        Dd = df["C_b"].values / (6 * N)
        L = df["L"].values
        th, fth = pm.fit_chin(N, Dd, L, starts=[rc.sl.BESIROGLU.theta])
        p, fp = pm.fit_kappa(N, Dd, L, th_chin=th)
        yc = pm.pred_chin(th, N, Dd)
        rf, _, _ = pm.fhh_residuals(np.log(L) - yc, pm.jac_log_chin(th, N, Dd))
        pr = [x for x in rc.pmap(_par_one, list(range(B_PAR)), dict(N=N, D=Dd, yhat_c=yc, rf_c=rf, th=th, p=p,
                                                                     seed0=rc.seed_of(f"par|{name}")), chunksize=8)
              if "_error" not in x]
        pr = pd.DataFrame(pr)
        out["param"].append(dict(design=name, n=len(L), sigma_chin=pm.sigma_star_chin(th), se_sigma_chin=pr.sigma_chin.std(ddof=1),
                                 sigma_kappa=pm.sigma_star_kappa(p), se_sigma_kappa=pr.sigma_kappa.std(ddof=1), kappa=p[5],
                                 a_chin=th[4] / (th[3] + th[4]), E_chin=float(np.exp(th[2])), E_kappa=float(np.exp(p[2])),
                                 qlr_kappa1=2 * len(L) * np.log(fth / fp), B=len(pr)))
        # finite-grid bias of the primary spec under the design's own parametric truths (noise-free)
        for lab, Lt, truth in (("Chinchilla form fitted to the design", np.exp(pm.pred_chin(th, N, Dd)), pm.sigma_star_chin(th)),
                               ("kappa family fitted to the design", np.exp(pm.pred_kappa(p, N, Dd)), pm.sigma_star_kappa(p))):
            dt = df.copy()
            dt["L"] = Lt
            rt = design_estimate(dt, order, H_PRIMARY)
            if not rt.get("ok"):
                out["bias"].append(dict(design=name, truth=lab, sigma_true=truth, ok=False))
                continue
            Sb = rt["S"]
            out["bias"].append(dict(design=name, truth=lab, sigma_true=truth, ok=True, k=len(rt["valid"]),
                                    sigma_pooled_equal_w=float(2 / (2 + np.mean(Sb))),
                                    bias_pooled=float(2 / (2 + np.mean(Sb)) - truth),
                                    bias_budget_min=float(np.min(rt["sigma"] - truth)),
                                    bias_budget_max=float(np.max(rt["sigma"] - truth))))
        # Monte Carlo coverage of the design-conditional bootstrap (selected designs): truth = the design's own
        # kappa-family fit, Gaussian noise with the robust (MAD) s.d. of its residuals, full estimator + bootstrap
        if name in MC_DESIGNS:
            Lt = np.exp(pm.pred_kappa(p, N, Dd))
            resid = L - Lt
            nsd = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
            mc = [x for x in rc.pmap(_mc_one, list(range(MC_R)), dict(df=df, Ltrue=Lt, noise_sd=nsd, order=order, h=H_PRIMARY,
                                                                       seed0=rc.seed_of(f"mc|{name}"), center="path",
                                                                       B=MC_B), chunksize=4)
                  if "_error" not in x and "fail" not in x]
            mc = pd.DataFrame(mc)
            truth = pm.sigma_star_kappa(p)
            out["mc"].append(dict(design=name, truth=truth, R=len(mc), R_target=MC_R, B=MC_B, noise_sd=nsd,
                                  mean_sigma_fe=mc.sigma_fe.mean(), bias_fe=mc.sigma_fe.mean() - truth,
                                  sd_sigma_fe=mc.sigma_fe.std(ddof=1), mean_boot_se_fe=float(((mc.hi - mc.lo) / 3.92).mean()),
                                  coverage_fe=float(np.mean((mc.lo <= truth) & (truth <= mc.hi))),
                                  mean_sigma_re=mc.sigma_re.mean(), bias_re=mc.sigma_re.mean() - truth,
                                  sd_sigma_re=mc.sigma_re.std(ddof=1), mean_se_re=mc.se_re.mean(),
                                  coverage_re=float(np.mean(np.abs(mc.sigma_re - truth) <= 1.96 * mc.se_re))))
            log(f"    MC {name}: bias FE {out['mc'][-1]['bias_fe']:+.4f} RE {out['mc'][-1]['bias_re']:+.4f}; coverage "
                f"FE {out['mc'][-1]['coverage_fe']:.2f}, RE {out['mc'][-1]['coverage_re']:.2f} (R={len(mc)})")
    for k in ("summary", "budgets", "invalid", "sens", "param", "bias", "mc", "order"):
        out[k] = pd.DataFrame(out[k])
    return out
