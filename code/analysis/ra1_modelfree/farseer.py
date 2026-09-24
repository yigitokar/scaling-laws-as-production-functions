"""farseer.py -- model-free local w and the extrapolation test on Farseer (Task 2; R1 c7.2-7.3, R2 Major 3, R4 M3),
and model-free sigma* along Farseer's local expansion path (R1 c6(b)).

Local surface: kernel-weighted local quadratic of f = ln L in (x, z) = (ln N, ln D) (Fan and Gijbels 1996; m2's
local_quad), Gaussian product kernel, bandwidths from leave-one-out CV (h = m * sd). Local elasticities f_x, f_z;
w_local = f_x/f_z (invariant to any monotone transform of L, hence E-free and form-free); local sigma from the
gradient and Hessian (m2's sigma_from_derivs). Along an isocost c' = ln(C/6) the local expansion path is the root of
ln w_local(x, c' - x) = 0; sigma* there is the model-free sigma*(C).

Parametric comparators (each fitted in the same parameter convention, then evaluated at the same (N, D) points):
  chin_full   Chinchilla form, Huber (delta = 1e-3), all 404 runs;
  chin_M100   Chinchilla form, Huber, runs with M = D/N <= 100 only ('Chinchilla-like support' extrapolation test);
  kappa_full  kappa family (outer exponent free), Huber, all runs;  kappa_M100 on M <= 100;
  eq3_full    Farseer's own Eq. 3 (Li et al. 2025), NLS on ln L, all runs;  eq3_M100 on M <= 100.
Conventions: 'ne' = non-embedding N (Farseer's); 'emb' = N incl. both untied embeddings (N + 131,072 h).
Statistic: Delta = ln w_param - ln w_local at grid points inside each size's observed D range (kernel n_eff >= 15;
25 as sensitivity), averaged within ln M bins.
Note (review): sigma along the local path is a second-derivative object; a local quadratic is the natural order for
first derivatives only (p - nu = 1), so the path sigma* carries O(h) smoothing bias at the design boundary and is
bandwidth-sensitive (see sensitivity()).

Inference (design-conditional): wild cluster bootstrap by model size (25 clusters, Webb weights) of the ln-loss
residuals around the pilot local-quadratic surface; every estimator is re-fitted in every draw (parametric fits
warm-started). Basic intervals: Delta_hat - quantiles(Delta* - Delta_0), Delta_0 = the statistic on the noise-free
bootstrap population. Smoothing bias of the local estimator is NOT in these intervals (bandwidth sensitivity reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

import parametric as pm
import ra1_common as rc

B_FAR = 999
if rc.QUICK:
    B_FAR = 20
# Review fix (ra1_modelfree_review.md, item M-A): on the primary CV grid the LOO-CV argmin of h_N is the grid's
# smallest multiple (0.2 sd) in both conventions, i.e. a boundary solution. The extended grid below locates the
# interior CV optimum; it is reported as a sensitivity (the primary bandwidth is kept: derivative estimation calls
# for more smoothing than the function-CV optimum), together with h_N x 0.75.
EXT_MULTS = (0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.45, 0.6, 0.8, 1.0)
MBINS = [0, 16, 64, 256, 1024, 4000]
MBIN_LAB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
C_PATH = np.array([1e18, 2e18, 5e18, 1e19, 2e19, 5e19, 1e20, 2e20, 5e20, 1e21])
SPECS = ["chin_full", "chin_M100", "kappa_full", "kappa_M100", "eq3_full", "eq3_M100"]


def load():
    import m2_data as md
    fa = md.load_farseer()
    return fa


# ============================================================================ local quadratic machinery
def _design(x, z, x0, z0):
    dx, dz = x - x0, z - z0
    return np.column_stack([np.ones_like(dx), dx, dz, dx * dx, dz * dz, dx * dz])


def local_at(x, z, f, x0, z0, hx, hz):
    """Coefficients [f, f_x, f_z, f_xx/2, f_zz/2, f_xz] and n_eff at (x0, z0)."""
    w = np.exp(-0.5 * (((x - x0) / hx) ** 2 + ((z - z0) / hz) ** 2))
    X = _design(x, z, x0, z0)
    sw = np.sqrt(w)
    c = np.linalg.lstsq(X * sw[:, None], f * sw, rcond=None)[0]
    return c, float(w.sum() ** 2 / np.sum(w ** 2))


def local_many(x, z, f, pts, hx, hz):
    out = np.zeros((len(pts), 7))
    for i, (x0, z0) in enumerate(pts):
        c, ne = local_at(x, z, f, x0, z0, hx, hz)
        out[i, :6], out[i, 6] = c, ne
    return out


def sigma_local(c):
    import m2_est as me
    return me.sigma_from_derivs(c[..., 1], c[..., 2], 2 * c[..., 3], 2 * c[..., 4], c[..., 5])


def pilot(x, z, f, hx, hz):
    """Fitted values and leverages of the local quadratic at the design points (bootstrap population)."""
    n = len(f)
    fit, lev = np.zeros(n), np.zeros(n)
    for i in range(n):
        w = np.exp(-0.5 * (((x - x[i]) / hx) ** 2 + ((z - z[i]) / hz) ** 2))
        X = _design(x, z, x[i], z[i])
        WX = X * w[:, None]
        A = np.linalg.solve(X.T @ WX, WX.T)
        fit[i] = A[0] @ f
        lev[i] = A[0, i]
    return fit, lev


def cv_bandwidth(x, z, f, mults=(0.2, 0.3, 0.45, 0.6, 0.8, 1.0)):
    import itertools
    sx, sz = x.std(), z.std()
    best = None
    table = []
    for mx, mz in itertools.product(mults, mults):
        hx, hz = mx * sx, mz * sz
        err = []
        for i in range(len(f)):
            w = np.exp(-0.5 * (((x - x[i]) / hx) ** 2 + ((z - z[i]) / hz) ** 2))
            w[i] = 0.0
            X = _design(x, z, x[i], z[i])
            sw = np.sqrt(w)
            c = np.linalg.lstsq(X * sw[:, None], f * sw, rcond=None)[0]
            err.append(f[i] - c[0])
        rmse = float(np.sqrt(np.mean(np.square(err))))
        table.append(dict(mx=mx, mz=mz, hx=hx, hz=hz, loocv_rmse=rmse))
        if best is None or rmse < best[0]:
            best = (rmse, hx, hz, mx, mz)
    return best, pd.DataFrame(table)


# ============================================================================ parametric w
def eq3_lnL(th, N, D):
    import m2_est as me
    return me.farseer_lnL(th, N, D)


def eq3_w(th, N, D, h=1e-4):
    fx = (eq3_lnL(th, N * np.exp(h), D) - eq3_lnL(th, N * np.exp(-h), D)) / (2 * h)
    fz = (eq3_lnL(th, N, D * np.exp(h)) - eq3_lnL(th, N, D * np.exp(-h))) / (2 * h)
    return fx / fz


def fit_eq3(N, D, L, init=None):
    import m2_est as me
    if init is None:
        st = me.farseer_starts(N, D, L)
        th, f = me.fit_farseer(N, D, L, "nls", starts=[st])
        return th, f
    y = np.log(L)

    def obj(q):
        r = eq3_lnL(q, N, D) - y
        return 0.5 * float(np.sum(r * r)) if np.all(np.isfinite(r)) else 1e10

    r = minimize(obj, init, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
    return r.x, r.fun


def fit_all_param(N, D, L, M, warm=None):
    """All six parametric comparators; warm = dict of previous estimates (bootstrap)."""
    out = {}
    sub = M <= 100
    for tag, s in (("full", np.ones(len(L), bool)), ("M100", sub)):
        if warm is None:
            th, _ = pm.fit_chin(N[s], D[s], L[s])
            p, _ = pm.fit_kappa(N[s], D[s], L[s], th_chin=th)
            te, _ = fit_eq3(N[s], D[s], L[s])
        else:
            m = rc.sl.fit_chinchilla(N[s], D[s], L[s], delta=1e-3, init=warm[f"chin_{tag}"])
            th = m.theta
            p, _ = pm.fit_kappa(N[s], D[s], L[s], init=warm[f"kappa_{tag}"])
            te, _ = fit_eq3(N[s], D[s], L[s], init=warm[f"eq3_{tag}"])
        out[f"chin_{tag}"], out[f"kappa_{tag}"], out[f"eq3_{tag}"] = th, p, te
    return out


def w_param(est, key, N, D):
    if key.startswith("chin"):
        return pm.wedge_chin(est[key], N, D)
    if key.startswith("kappa"):
        return pm.wedge_kappa(est[key], N, D)
    return eq3_w(est[key], N, D)


def sigma_param_star(est, key, C):
    """sigma* of a parametric comparator at compute C (constant for chin/kappa; path-dependent for Eq. 3)."""
    if key.startswith("chin"):
        return np.full(len(C), pm.sigma_star_chin(est[key]))
    if key.startswith("kappa"):
        return np.full(len(C), pm.sigma_star_kappa(est[key]))
    import m2_est as me
    out = []
    for Cc in C:
        cp = np.log(Cc / 6)
        g = lambda xx: np.log(eq3_w(est[key], np.exp(np.array([xx])), np.exp(np.array([cp - xx]))))[0]
        try:
            x0 = brentq(g, cp / 2 - 6, cp / 2 + 4)
            f = lambda a, b: float(eq3_lnL(est[key], np.exp(np.array([a])), np.exp(np.array([b])))[0])
            out.append(me.numderiv_sigma(f, x0, cp - x0, h=1e-3)[0])
        except ValueError:
            out.append(np.nan)
    return np.array(out)


# ============================================================================ local path (model-free sigma*(C))
def local_path(x, z, f, hx, hz, C_levels, hull):
    """For each C: root of ln w_local on the isocost inside the support; returns x*, sigma*, M*, neff."""
    rows = []
    for Cc in C_levels:
        cp = np.log(Cc / 6)
        lo, hi = hull(cp)
        if lo is None:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue

        def g(xx):
            c, _ = local_at(x, z, f, xx, cp - xx, hx, hz)
            return np.log(c[1] / c[2])

        try:
            glo, ghi = g(lo), g(hi)
            if np.sign(glo) == np.sign(ghi):
                rows.append((np.nan, np.nan, np.nan, np.nan))
                continue
            x0 = brentq(g, lo, hi, xtol=1e-6)
        except (ValueError, FloatingPointError):
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        c, ne = local_at(x, z, f, x0, cp - x0, hx, hz)
        rows.append((x0, float(sigma_local(c)), float(np.exp(cp - 2 * x0)), ne))
    return np.array(rows)


def make_hull(x, z, xs_sizes, dmin, dmax):
    """Isocost segment inside the design: x between sizes whose observed D range contains d = c' - x."""
    def hull(cp):
        ok = [xx for xx, a, b in zip(xs_sizes, dmin, dmax) if np.log(0.95 * a) <= cp - xx <= np.log(1.05 * b)]
        if len(ok) < 3:
            return None, None
        return min(ok), max(ok)
    return hull


# ============================================================================ evaluation grid
def build_grid(fa, n_m=24):
    Ns = np.sort(fa.N.unique())
    Ms = np.exp(np.linspace(0.0, np.log(2570.0), n_m))
    rows = []
    for n0 in Ns:
        s = fa.N == n0
        dmin, dmax = fa.D[s].min(), fa.D[s].max()
        ne = float(fa.N_emb[s].iloc[0])
        for m0 in Ms:
            d0 = n0 * m0
            if 0.9 * dmin <= d0 <= 1.1 * dmax:
                rows.append(dict(N=n0, N_emb=ne, D=d0, M=m0, M_emb=d0 / ne))
    return pd.DataFrame(rows)


def statistic(data, est, grid, bw, conv_list=("ne", "emb"), path=True):
    """All quantities of the Farseer stage for one data set (point estimate or bootstrap draw)."""
    out = {}
    D = data["D"]
    z = np.log(D)
    for conv in conv_list:
        Nc = data["N"] if conv == "ne" else data["N_emb"]
        x = np.log(Nc)
        f = np.log(data["L"])
        hx, hz = bw[conv]
        gN = grid["N"].values if conv == "ne" else grid["N_emb"].values
        pts = list(zip(np.log(gN), np.log(grid["D"].values)))
        loc = local_many(x, z, f, pts, hx, hz)
        wl = loc[:, 1] / loc[:, 2]
        out[f"lnw_local_{conv}"] = np.log(wl)
        out[f"sigma_local_{conv}"] = sigma_local(loc[:, :6])
        out[f"neff_{conv}"] = loc[:, 6]
        for key in SPECS:
            out[f"lnw_{key}_{conv}"] = np.log(w_param(est[conv], key, gN, grid["D"].values))
        if path:
            xs_sizes = np.log(np.sort(np.unique(Nc)))
            dmin = [D[Nc == v].min() for v in np.sort(np.unique(Nc))]
            dmax = [D[Nc == v].max() for v in np.sort(np.unique(Nc))]
            hull = make_hull(x, z, xs_sizes, dmin, dmax)
            pth = local_path(x, z, f, hx, hz, C_PATH, hull)
            out[f"path_x_{conv}"], out[f"path_sigma_{conv}"] = pth[:, 0], pth[:, 1]
            out[f"path_M_{conv}"], out[f"path_neff_{conv}"] = pth[:, 2], pth[:, 3]
            for key in SPECS:
                out[f"path_sigma_{key}_{conv}"] = sigma_param_star(est[conv], key, C_PATH)
    return out


NEFF_MIN = 15


def summaries(stat, grid, neff_min=None):
    """Bin averages of Delta = ln w_param - ln w_local over grid points with kernel n_eff >= neff_min (primary 15,
    which reaches Farseer's largest ratios M ~ 2,570 at the smallest sizes; 25 as sensitivity); linearity regression
    of ln w_local on ln(M/M*_local(C))."""
    neff_min = NEFF_MIN if neff_min is None else neff_min
    res = {}
    for conv in ("ne", "emb"):
        Mv = grid["M"].values if conv == "ne" else grid["M_emb"].values
        ok = stat[f"neff_{conv}"] >= neff_min
        b = np.digitize(Mv, MBINS) - 1
        for key in SPECS:
            dlt = stat[f"lnw_{key}_{conv}"] - stat[f"lnw_local_{conv}"]
            for j, lab in enumerate(MBIN_LAB):
                s = ok & (b == j)
                res[f"delta|{key}|{conv}|{lab}"] = float(np.mean(dlt[s])) if s.sum() else np.nan
            s = ok & (Mv >= 1000)
            res[f"delta|{key}|{conv}|M>=1000"] = float(np.mean(dlt[s])) if s.sum() else np.nan
        for j, lab in enumerate(MBIN_LAB):
            s = ok & (b == j)
            res[f"lnw_local|{conv}|{lab}"] = float(np.mean(stat[f"lnw_local_{conv}"][s])) if s.sum() else np.nan
        # linearity: ln w_local = b1 u + b2 u^2, u = ln(M / M*_local(C)), M*_local(C) interpolated on the local path
        if f"path_M_{conv}" in stat:
            pc = np.log(C_PATH)
            pm_ = np.log(stat[f"path_M_{conv}"])
            g = np.isfinite(pm_)
            gN = grid["N"].values if conv == "ne" else grid["N_emb"].values
            cg = np.log(6 * gN * grid["D"].values)
            inside = ok & (cg >= pc[g].min()) & (cg <= pc[g].max())
            if g.sum() >= 3 and inside.sum() > 10:
                lMs = np.interp(cg[inside], pc[g], pm_[g])
                u = np.log(Mv[inside]) - lMs
                y = stat[f"lnw_local_{conv}"][inside]
                X = np.column_stack([u, u * u])
                bb = np.linalg.lstsq(X, y, rcond=None)[0]
                b1 = np.linalg.lstsq(u[:, None], y, rcond=None)[0][0]
                res[f"lin|{conv}|b1_only"], res[f"lin|{conv}|b1"], res[f"lin|{conv}|b2"] = float(b1), float(bb[0]), float(bb[1])
                sp = stat[f"path_sigma_{conv}"][g]
                res[f"lin|{conv}|path_1_over_sigma_minus_1"] = float(np.mean(1 / sp - 1))
                res[f"lin|{conv}|n"] = int(inside.sum())
    return res


# ============================================================================ bootstrap worker
def _boot_worker(seed, data, est, grid, bw, fit_pilot, eres, clusters, scheme):
    rng = np.random.default_rng(seed)
    if scheme == "cluster_webb":
        ug = np.unique(clusters)
        eta = dict(zip(ug, rc.webb(rng, len(ug))))
        v = np.array([eta[g] for g in clusters])
    else:
        v = rc.rademacher(rng, len(eres))
    Lb = np.exp(fit_pilot + v * eres)
    d2 = dict(data, L=Lb)
    e2 = {}
    for conv in ("ne", "emb"):
        Nc = data["N"] if conv == "ne" else data["N_emb"]
        e2[conv] = fit_all_param(Nc, data["D"], Lb, data["D"] / Nc, warm=est[conv])
    st = statistic(d2, e2, grid, bw)
    sm = summaries(st, grid)
    sig = {f"sigma_{k}_{c}": (pm.sigma_star_chin(e2[c][k]) if k.startswith("chin") else pm.sigma_star_kappa(e2[c][k]))
           for c in ("ne", "emb") for k in ("chin_full", "chin_M100", "kappa_full", "kappa_M100")}
    keep = {k: v_ for k, v_ in st.items() if k.startswith("path_") or k.startswith("lnw_local") or k.startswith("sigma_local")}
    return dict(sm=sm, sig=sig, st=keep)


def run(log=rc.log, B=B_FAR):
    fa = load()
    data = dict(N=fa.N.values.astype(float), N_emb=fa.N_emb.values.astype(float), D=fa.D.values.astype(float),
                L=fa.L.values.astype(float))
    z = np.log(data["D"])
    f = np.log(data["L"])
    bw, cvtab = {}, []
    for conv in ("ne", "emb"):
        Nc = data["N"] if conv == "ne" else data["N_emb"]
        best, tab = cv_bandwidth(np.log(Nc), z, f)
        bw[conv] = (best[1], best[2])
        cvtab.append(tab.assign(conv=conv))
        log(f"  Farseer CV bandwidth ({conv}): mx={best[3]}, mz={best[4]} (hx={best[1]:.3f}, hz={best[2]:.3f}), LOO RMSE {best[0]:.5f}")
    bw_ext, cvext = {}, []
    for conv in ("ne", "emb"):
        Nc = data["N"] if conv == "ne" else data["N_emb"]
        best, tab = cv_bandwidth(np.log(Nc), z, f, mults=EXT_MULTS)
        bw_ext[conv] = (best[1], best[2])
        cvext.append(tab.assign(conv=conv))
        log(f"  Farseer CV bandwidth, extended grid ({conv}): mx={best[3]}, mz={best[4]} (hx={best[1]:.3f}, "
            f"hz={best[2]:.3f}), LOO RMSE {best[0]:.5f}")
    grid = build_grid(fa)
    est = {}
    for conv in ("ne", "emb"):
        Nc = data["N"] if conv == "ne" else data["N_emb"]
        est[conv] = fit_all_param(Nc, data["D"], data["L"], data["D"] / Nc)
        log(f"  Farseer parametric ({conv}): sigma* chin {pm.sigma_star_chin(est[conv]['chin_full']):.3f} "
            f"chin_M100 {pm.sigma_star_chin(est[conv]['chin_M100']):.3f} kappa {pm.sigma_star_kappa(est[conv]['kappa_full']):.3f} "
            f"kappa_M100 {pm.sigma_star_kappa(est[conv]['kappa_M100']):.3f}")
    st0 = statistic(data, est, grid, bw)
    sm0 = summaries(st0, grid)
    # bootstrap population: pilot local quadratic (non-embedding coordinates) and leverage-adjusted residuals
    x = np.log(data["N"])
    fit_p, lev = pilot(x, z, f, *bw["ne"])
    eres = (f - fit_p) / np.sqrt(np.clip(1 - lev, 0.1, 1))
    clusters = fa.N.values
    # intra-cluster correlation of pilot residuals (by model size; by D level)
    icc = {}
    for lab, g in (("model size N", fa.N.values), ("token budget D", np.round(np.log(fa.D.values), 2))):
        dfr = pd.DataFrame(dict(g=g, e=f - fit_p))
        between = dfr.groupby("g").e.mean().var()
        icc[lab] = float(between / dfr.e.var())
    # noise-free population statistic Delta_0
    data0 = dict(data, L=np.exp(fit_p))
    est0 = {c: fit_all_param(data["N"] if c == "ne" else data["N_emb"], data["D"], data0["L"],
                             data["D"] / (data["N"] if c == "ne" else data["N_emb"]), warm=est[c]) for c in ("ne", "emb")}
    st_0 = statistic(data0, est0, grid, bw)
    sm_0 = summaries(st_0, grid)
    log(f"  Farseer point estimates done; residual ICC {icc}; starting {B} wild cluster draws")
    payload = dict(data=data, est=est, grid=grid, bw=bw, fit_pilot=fit_p, eres=eres, clusters=clusters,
                   scheme="cluster_webb")
    seeds = [rc.seed_of(f"farseer|{i}") for i in range(B)]
    draws = rc.pmap(_boot_worker, seeds, payload, chunksize=8)
    nfail = sum("_error" in d for d in draws)
    draws = [d for d in draws if "_error" not in d]
    # run-level wild (Rademacher) for comparison of standard errors (fewer draws)
    B2 = max(20, B // 2)
    payload2 = dict(payload, scheme="run_rademacher")
    seeds2 = [rc.seed_of(f"farseer_run|{i}") for i in range(B2)]
    draws2 = [d for d in rc.pmap(_boot_worker, seeds2, payload2, chunksize=8) if "_error" not in d]
    log(f"  Farseer bootstrap done: {len(draws)} cluster draws ({nfail} failed), {len(draws2)} run-level draws")
    return dict(fa=fa, data=data, grid=grid, bw=bw, cv=pd.concat(cvtab), est=est, st0=st0, sm0=sm0, sm_pop=sm_0,
                st_pop=st_0, draws=draws, draws_run=draws2, icc=icc, B=B, B2=B2, nfail=nfail,
                bw_ext=bw_ext, cv_ext=pd.concat(cvext))


# ============================================================================ tables from the stage output
def tables(R):
    sm0, sm_pop = R["sm0"], R["sm_pop"]
    D = pd.DataFrame([d["sm"] for d in R["draws"]])
    D2 = pd.DataFrame([d["sm"] for d in R["draws_run"]])
    rows = []
    for key in sm0:
        if not key.startswith("delta|"):
            continue
        _, spec, conv, mb = key.split("|")
        est = sm0[key]
        dd = D[key].values - sm_pop[key]
        dd = dd[np.isfinite(dd)]
        se = float(np.std(dd, ddof=1)) if len(dd) > 4 else np.nan
        lo, hi = (est - np.percentile(dd, 97.5), est - np.percentile(dd, 2.5)) if len(dd) > 4 else (np.nan, np.nan)
        d2 = D2[key].values
        rows.append(dict(spec=spec, conv=conv, M_bin=mb, delta=est, se=se, lo=lo, hi=hi,
                         se_run_wild=float(np.nanstd(d2, ddof=1)), delta_pop=sm_pop[key],
                         ratio=float(np.exp(est)), ratio_lo=float(np.exp(lo)), ratio_hi=float(np.exp(hi))))
    delta = pd.DataFrame(rows)
    # local ln w by bin
    lw = []
    for key in sm0:
        if key.startswith("lnw_local|"):
            _, conv, mb = key.split("|")
            dd = D[key].values - sm_pop[key]
            lw.append(dict(conv=conv, M_bin=mb, lnw_local=sm0[key], w_local=np.exp(sm0[key]),
                           se=float(np.nanstd(dd, ddof=1)), lo=sm0[key] - np.nanpercentile(dd, 97.5),
                           hi=sm0[key] - np.nanpercentile(dd, 2.5)))
    lw = pd.DataFrame(lw)
    # linearity
    lin = []
    for conv in ("ne", "emb"):
        for par in ("b1_only", "b1", "b2", "path_1_over_sigma_minus_1"):
            key = f"lin|{conv}|{par}"
            if key not in sm0:
                continue
            dd = D[key].values - sm_pop[key]
            dd = dd[np.isfinite(dd)]
            lin.append(dict(conv=conv, param=par, est=sm0[key], se=float(np.std(dd, ddof=1)),
                            lo=sm0[key] - np.percentile(dd, 97.5), hi=sm0[key] - np.percentile(dd, 2.5),
                            p_zero=float((1 + np.sum(np.abs(dd) >= abs(sm0[key]))) / (len(dd) + 1)), n=sm0.get(f"lin|{conv}|n")))
    # test b1 (slope at the path, linear fit) against the path's 1/sigma* - 1 (same draws)
    for conv in ("ne", "emb"):
        k1, k2 = f"lin|{conv}|b1_only", f"lin|{conv}|path_1_over_sigma_minus_1"
        if k1 in sm0:
            e = sm0[k1] - sm0[k2]
            dd = (D[k1] - D[k2]).values - (sm_pop[k1] - sm_pop[k2])
            dd = dd[np.isfinite(dd)]
            lin.append(dict(conv=conv, param="b1_only minus path (1/sigma*-1)", est=e, se=float(np.std(dd, ddof=1)),
                            lo=e - np.percentile(dd, 97.5), hi=e - np.percentile(dd, 2.5),
                            p_zero=float((1 + np.sum(np.abs(dd) >= abs(e))) / (len(dd) + 1)), n=sm0.get(f"lin|{conv}|n")))
    lin = pd.DataFrame(lin)
    # path sigma*(C)
    st0, st_pop = R["st0"], R["st_pop"]
    P = []
    for conv in ("ne", "emb"):
        arr = np.array([d["st"][f"path_sigma_{conv}"] for d in R["draws"]])
        for j, Cc in enumerate(C_PATH):
            e = st0[f"path_sigma_{conv}"][j]
            dd = arr[:, j] - st_pop[f"path_sigma_{conv}"][j]
            dd = dd[np.isfinite(dd)]
            row = dict(conv=conv, C=Cc, sigma_local_path=e, M_star_local=st0[f"path_M_{conv}"][j],
                       N_star_local=np.exp(st0[f"path_x_{conv}"][j]), neff=st0[f"path_neff_{conv}"][j],
                       se=float(np.std(dd, ddof=1)) if len(dd) > 4 else np.nan,
                       lo=e - np.percentile(dd, 97.5) if len(dd) > 4 else np.nan,
                       hi=e - np.percentile(dd, 2.5) if len(dd) > 4 else np.nan, nvalid=len(dd))
            for key in SPECS:
                row[f"sigma_{key}"] = st0[f"path_sigma_{key}_{conv}"][j]
            P.append(row)
    path = pd.DataFrame(P)
    # pooled path sigma* (mean over C levels with a valid root; same draws)
    pool = []
    for conv in ("ne", "emb"):
        arr = np.array([d["st"][f"path_sigma_{conv}"] for d in R["draws"]])
        e0 = st0[f"path_sigma_{conv}"]
        g = np.isfinite(e0) & np.isfinite(st_pop[f"path_sigma_{conv}"])
        e = float(np.mean(e0[g]))
        dd = np.nanmean(arr[:, g], axis=1) - float(np.mean(st_pop[f"path_sigma_{conv}"][g]))
        dd = dd[np.isfinite(dd)]
        lc = np.log10(C_PATH[g])
        X = np.column_stack([np.ones(g.sum()), lc - lc.mean()])
        bsl = np.linalg.lstsq(X, e0[g], rcond=None)[0][1]
        bd = np.array([np.linalg.lstsq(X, a[g], rcond=None)[0][1] if np.all(np.isfinite(a[g])) else np.nan for a in arr])
        bpop = np.linalg.lstsq(X, st_pop[f"path_sigma_{conv}"][g], rcond=None)[0][1]
        bdd = bd[np.isfinite(bd)] - bpop
        # random effects across compute levels (DerSimonian-Laird on sigma*(C) with per-level bootstrap variances)
        from isoflop import dersimonian_laird
        vC = np.array([np.nanvar(arr[:, j] - st_pop[f"path_sigma_{conv}"][j], ddof=1) for j in np.where(g)[0]])
        dl = dersimonian_laird(e0[g], vC)
        pool.append(dict(conv=conv, k_C=int(g.sum()), Cmin=C_PATH[g].min(), Cmax=C_PATH[g].max(), sigma_mean=e,
                         sigma_re=dl["mu"], se_re=dl["se"], tau=dl["tau"], I2=dl["I2"], p_Q=dl["p_Q"],
                         se=float(np.std(dd, ddof=1)), lo=e - np.percentile(dd, 97.5), hi=e - np.percentile(dd, 2.5),
                         drift_per_decade=float(bsl), se_drift=float(np.std(bdd, ddof=1)),
                         lo_drift=float(bsl - np.percentile(bdd, 97.5)), hi_drift=float(bsl - np.percentile(bdd, 2.5)),
                         p_drift=float((1 + np.sum(np.abs(bdd) >= abs(bsl))) / (len(bdd) + 1)),
                         B=len(R["draws"])))
    pool = pd.DataFrame(pool)
    # parametric sigma* with wild-cluster SEs
    S = pd.DataFrame([d["sig"] for d in R["draws"]])
    sg = []
    for conv in ("ne", "emb"):
        for key in ("chin_full", "chin_M100", "kappa_full", "kappa_M100"):
            e = pm.sigma_star_chin(R["est"][conv][key]) if key.startswith("chin") else pm.sigma_star_kappa(R["est"][conv][key])
            sg.append(dict(conv=conv, spec=key, sigma_star=e, se_wild_cluster=float(S[f"sigma_{key}_{conv}"].std(ddof=1)),
                           lo=rc.pct(S[f"sigma_{key}_{conv}"], 2.5), hi=rc.pct(S[f"sigma_{key}_{conv}"], 97.5),
                           kappa=R["est"][conv][key][5] if key.startswith("kappa") else 1.0))
    sg = pd.DataFrame(sg)
    # grid-level point estimates (for the figure)
    g = R["grid"].copy()
    for conv in ("ne", "emb"):
        g[f"lnw_local_{conv}"] = st0[f"lnw_local_{conv}"]
        g[f"sigma_local_{conv}"] = st0[f"sigma_local_{conv}"]
        g[f"neff_{conv}"] = st0[f"neff_{conv}"]
        arr = np.array([d["st"][f"lnw_local_{conv}"] for d in R["draws"]])
        g[f"se_lnw_local_{conv}"] = np.nanstd(arr, axis=0, ddof=1)
        for key in SPECS:
            g[f"lnw_{key}_{conv}"] = st0[f"lnw_{key}_{conv}"]
    return dict(delta=delta, lnw=lw, lin=lin, path=path, pool=pool, sigma=sg, grid=g)


def sensitivity(R):
    """Point-estimate sensitivity of the bin-averaged Delta, the linearity regression and the local-path sigma* to the
    local bandwidth and to the n_eff cut. Wider (x1.5, x2) AND narrower bandwidths are reported (review fix M-A):
    h_N x 0.75 (h_D unchanged) and the extended-grid LOO-CV optimum (EXT_MULTS), which lies below the primary grid.
    'path_sigma_mean_poolC' averages over the compute levels used in the primary pooled path summary."""
    rows = []
    bw0 = R["bw"]
    sc = lambda fx, fz: {c: (fx * bw0[c][0], fz * bw0[c][1]) for c in bw0}
    variants = [("primary (CV bandwidth, n_eff >= 15)", None, NEFF_MIN), ("n_eff >= 25", None, 25),
                ("bandwidth x 1.5", sc(1.5, 1.5), NEFF_MIN), ("bandwidth x 2", sc(2.0, 2.0), NEFF_MIN),
                ("h_N x 0.75 (h_D unchanged)", sc(0.75, 1.0), NEFF_MIN)]
    if R.get("bw_ext") is not None:
        variants.append(("extended-grid CV optimum", R["bw_ext"], NEFF_MIN))
    g_pool = {c: np.isfinite(R["st0"][f"path_sigma_{c}"]) & np.isfinite(R["st_pop"][f"path_sigma_{c}"])
              for c in ("ne", "emb")}
    for lab, bw, nmin in variants:
        st = R["st0"] if bw is None else statistic(R["data"], R["est"], R["grid"], bw, path=True)
        sm = summaries(st, R["grid"], neff_min=nmin)
        for k, v in sm.items():
            if k.startswith("delta|") or k.startswith("lin|") or k.startswith("lnw_local|"):
                rows.append(dict(variant=lab, key=k, value=v))
        bwu = bw0 if bw is None else bw
        for c in ("ne", "emb"):
            rows.append(dict(variant=lab, key=f"h_N|{c}", value=float(bwu[c][0])))
            rows.append(dict(variant=lab, key=f"h_D|{c}", value=float(bwu[c][1])))
            if f"path_sigma_{c}" in st:
                ps = np.asarray(st[f"path_sigma_{c}"], float)
                rows.append(dict(variant=lab, key=f"path_sigma_mean|{c}", value=float(np.nanmean(ps))))
                q = ps[g_pool[c]]
                rows.append(dict(variant=lab, key=f"path_sigma_mean_poolC|{c}", value=float(np.nanmean(q))))
                rows.append(dict(variant=lab, key=f"path_sigma_min_poolC|{c}", value=float(np.nanmin(q))))
                rows.append(dict(variant=lab, key=f"path_sigma_max_poolC|{c}", value=float(np.nanmax(q))))
    return pd.DataFrame(rows)
