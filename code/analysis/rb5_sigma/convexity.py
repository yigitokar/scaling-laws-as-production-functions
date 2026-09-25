"""convexity.py -- the extrapolation decomposition (T1.9; R2 NM2 requests 1 and 3, R4 minor 14), Farseer's first-
derivative estimator restricted to budgets <= 3e20 (input of T1.4), and the numbers of Figure 5 (T16).

Local wedge (unchanged estimators). Farseer: ra1's kernel-weighted local quadratic of ln L in (ln N, ln D), primary CV
bandwidth, non-embedding N, 25 sizes x 24 log-spaced M up to 2,570, n_eff >= 15. Marin (three corpora) and Llama 3:
rb1's local quadratic in (ln M, ln C) at its primary bandwidths, 14 log-spaced M per budget line, n_eff >= 8. The
local expansion path M*_local(C) is the root of ln w_local on each isocost; u = ln(M/M*_local(C)) with M*_local
interpolated in ln C, at evaluation points inside the path's compute range. ln w_local = b1 u + b2 u^2 by OLS (ra1/rb1).

Decomposition (R2 NM2(1)). The linear extrapolation from the local slope at the path is ln w_lin = b1 u (the tangent
of panels d-f of Figure 5; 1/(1+b1) is the local first-derivative sigma*). At each evaluation point
    Delta_param = ln(w_param/w_local) = [ln w_param - b1 u] + [b1 u - ln w_local]
                = rotation-and-location component + Delta_lin,
where Delta_lin = ln(w_lin/w_local) is the gap of the linear extrapolation from the model-free slope, and
-Delta_lin = ln w_local - b1 u is the convexity component (positive: ln w bends up beyond the path). Bin means use the
evaluation points inside the path's compute range (the published Delta uses all evaluation points; both are given).

Bootstrap. The published draws are regenerated exactly: rb1's wild cluster bootstrap by budget (Webb weights, rb1's
seeds 'extrap|<design>|i', B = 999) and ra1's wild cluster bootstrap by model size (ra1's seeds 'farseer|i', B = 999),
with the parametric comparators re-fitted (warm-started) in every draw; the Eq. (3) comparator and the embedding
convention of ra1 are not re-fitted (they do not enter here and use no random numbers). Reproduction is checked
against the published bootstrap moments. Intervals are percentile intervals: the 2.5 and 97.5 percent quantiles of the
draws of the statistic (R4 minor 14), which contain the point estimate whenever the estimate lies inside the bootstrap
distribution; the basic intervals of ra1/rb1 are kept for comparison. Smoothing bias is not in any interval.

Leave-corner-out (R2 NM2(3)). Marin's highest-M bins rest on the design's corner: the smallest model on the largest
budgets. Two checks re-estimate everything (local surface at the primary bandwidths, parametric comparators, path)
without the corner runs: (a) each budget's largest-M run left out; (b) the runs with M > 1,000 left out (the only runs
that support the >= 1,024 bin). Bootstrap as above with this module's seeds.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import sigcommon as cm

rc = cm.rc
B_EX = 99 if cm.QUICK else 999
SPECS = ["chin_full", "chin_M100", "kappa_full", "kappa_M100"]
MBIN_LAB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
MARIN = ["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]
CUT_3E20 = 3e20 * (1 + 1e-9)


# ============================================================================ common: statistics from per-point arrays
def point_stats(M, C, lnw_local, neff, lnMstar_at_C, inside_C, lnw_param, neff_min, mbins, cut_C=None):
    """Bin means of the decomposition and the quadratic of ln w_local in u, from per-point arrays.
    M, C: evaluation points; lnMstar_at_C: ln M*_local interpolated at C (nan outside the path range);
    inside_C: point's compute inside the path range; lnw_param: dict spec -> array."""
    ok = (neff >= neff_min) & np.isfinite(lnw_local)
    ins = ok & inside_C & np.isfinite(lnMstar_at_C)
    if cut_C is not None:
        ins = ins & (C <= cut_C)
    b = np.digitize(M, mbins) - 1
    res = {}
    if ins.sum() <= 10:
        return res
    u = np.log(M[ins]) - lnMstar_at_C[ins]
    y = lnw_local[ins]
    Xq = np.column_stack([u, u * u])
    b1, b2 = np.linalg.lstsq(Xq, y, rcond=None)[0]
    res.update(b1=float(b1), b2=float(b2), sigma_slope=float(1.0 / (1.0 + b1)), n_inside=int(ins.sum()),
               umax=float(u.max()), umin=float(u.min()))
    uu = np.full(len(M), np.nan)
    uu[ins] = u
    dlin = b1 * uu - lnw_local                     # Delta_lin = ln(w_lin / w_local)
    for j, lab in enumerate(MBIN_LAB):
        s = ins & (b == j)
        res[f"n_in|{lab}"] = int(s.sum())
        res[f"dlin|{lab}"] = float(np.mean(dlin[s])) if s.sum() else np.nan
        res[f"lnw_local_in|{lab}"] = float(np.mean(lnw_local[s])) if s.sum() else np.nan
        res[f"u_in|{lab}"] = float(np.mean(uu[s])) if s.sum() else np.nan
        for sp, lp in lnw_param.items():
            dp = lp - lnw_local
            res[f"dparam_in|{sp}|{lab}"] = float(np.mean(dp[s])) if s.sum() else np.nan
            res[f"rot|{sp}|{lab}"] = float(np.mean((lp - b1 * uu)[s])) if s.sum() else np.nan
            s2 = ok & (b == j)
            res[f"dparam_all|{sp}|{lab}"] = float(np.mean(dp[s2])) if s2.sum() else np.nan
    s = ins & (M >= 256)
    res["n_in|M>=256"] = int(s.sum())
    res["dlin|M>=256"] = float(np.mean(dlin[s])) if s.sum() else np.nan
    for sp, lp in lnw_param.items():
        res[f"dparam_in|{sp}|M>=256"] = float(np.mean((lp - lnw_local)[s])) if s.sum() else np.nan
        res[f"rot|{sp}|M>=256"] = float(np.mean((lp - b1 * uu)[s])) if s.sum() else np.nan
    return res


# ============================================================================ rb1 designs (Marin x3, Llama 3)
def _rb1_setup(name, drop=None):
    import extrap as ex
    X = pickle.load(open(cm.RB1_EXTRAP_PKL, "rb"))
    R = X[name]
    d = ex.load_design(name)
    if drop is not None:
        d = d[~drop(d)].reset_index(drop=True)
    hx, hc = R["bw"][:2]
    grid = R["grid"] if drop is None else ex.build_grid(d)
    budgets = R["budgets"] if drop is None else [(Cb, g["u"].min(), g["u"].max()) for Cb, g in d.groupby("C_b")]
    x, c, f = d["u"].values, d["cc"].values, d["f"].values
    est = ex.fit_param(d)                             # rb1's cold start = the warm start of its draws
    fit_p, lev = ex.pilot(x, c, f, hx, hc)
    eres = (f - fit_p) / np.sqrt(np.clip(1 - lev, 0.1, 1))
    return dict(name=name, d=d, grid=grid, budgets=budgets, bw=(hx, hc), est=est, fit_p=fit_p, eres=eres, R=R)


def _rb1_arrays(st, grid, budgets):
    Cb = np.array([bb[0] for bb in budgets])
    lp = st["path_lnMstar"]
    g = np.isfinite(lp)
    lC = np.log(grid["C_b"].values)
    if g.sum() >= 3:
        lMs = np.interp(lC, np.log(Cb[g]), lp[g])
        inside = (lC >= np.log(Cb[g]).min() - 1e-9) & (lC <= np.log(Cb[g]).max() + 1e-9)
    else:
        lMs, inside = np.full(len(lC), np.nan), np.zeros(len(lC), bool)
    return lMs, inside


def _rb1_stats(st, S):
    import extrap as ex
    grid = S["grid"]
    lMs, inside = _rb1_arrays(st, grid, S["budgets"])
    lp = {sp: st[f"lnw_{sp}"] for sp in SPECS}
    return point_stats(grid["M"].values, grid["C_b"].values, st["lnw_local"], st["neff"], lMs, inside, lp,
                       ex.NEFF_MIN, ex.MBINS)


def _rb1_draw(seed, S):
    """rb1's extrap._boot_worker (cluster_webb scheme), with per-point arrays kept."""
    import extrap as ex
    rng = np.random.default_rng(seed)
    d = S["d"]
    g = pd.factorize(d["C_b"].values)[0]
    v = rng.choice(rc.WEBB, size=g.max() + 1)[g]
    fstar = S["fit_p"] + v * S["eres"]
    ds = d.copy()
    ds["L"] = np.exp(fstar)
    est_b = ex.fit_param(ds, warm=S["est"])
    st = ex.statistic(ds, fstar, est_b, S["grid"], S["bw"], S["budgets"])
    sm = ex.summaries(st, S["grid"], S["budgets"])
    return dict(mine=_rb1_stats(st, S), sm=sm)


def run_rb1(name, drop=None, tag="primary", log=cm.log, B=B_EX):
    import extrap as ex
    S = _rb1_setup(name, drop)
    d, f = S["d"], S["d"]["f"].values
    st0 = ex.statistic(d, f, S["est"], S["grid"], S["bw"], S["budgets"])
    sm0 = ex.summaries(st0, S["grid"], S["budgets"])
    mine0 = _rb1_stats(st0, S)
    d0 = d.copy()
    d0["L"] = np.exp(S["fit_p"])
    est0 = ex.fit_param(d0, warm=S["est"])
    stp = ex.statistic(d0, S["fit_p"], est0, S["grid"], S["bw"], S["budgets"])
    minep = _rb1_stats(stp, S)
    smp = ex.summaries(stp, S["grid"], S["budgets"])
    if tag == "primary":
        seeds = [cm.cm1.seed_of(f"extrap|{name}|{i}") for i in range(B)]      # rb1's seeds (reproduction)
    else:
        seeds = [cm.seed_of(f"extrap|{tag}|{name}|{i}") for i in range(B)]
    payload = dict(S={k: v for k, v in S.items() if k != "R"})
    draws = cm.pmap(_rb1_draw, seeds, payload, chunksize=25)
    nfail = sum("_error" in x for x in draws)
    draws = [x for x in draws if "_error" not in x]
    log(f"    [{name} | {tag}] n = {len(d)}; b2 {mine0.get('b2', np.nan):+.4f}; dlin 256-1,024 "
        f"{mine0.get('dlin|256-1,024', np.nan):+.3f}; draws {len(draws)} ({nfail} failed)")
    return dict(name=name, tag=tag, n=len(d), mine0=mine0, minep=minep, sm0=sm0, smp=smp,
                draws_mine=[x["mine"] for x in draws], draws_sm=[x["sm"] for x in draws], nfail=nfail,
                R_pub=S["R"] if tag == "primary" else None, n_dropped=int(len(ex.load_design(name)) - len(d)),
                Mmax_eval=float(np.nanmax(np.where((st0["neff"] >= ex.NEFF_MIN) & np.isfinite(st0["lnw_local"]),
                                                   S["grid"]["M"].values, np.nan))))


def check_rb1(res):
    """Reproduction of rb1's published draws: max |diff| of every summary key over all draws."""
    R = res["R_pub"]
    pub = R["draws"]
    mx = 0.0
    keys = [k for k in pub[0] if isinstance(pub[0][k], float)]
    n = min(len(pub), len(res["draws_sm"]))
    for i in range(n):
        a, b = pub[i], res["draws_sm"][i]
        for k in keys:
            if np.isfinite(a.get(k, np.nan)) or np.isfinite(b.get(k, np.nan)):
                mx = max(mx, abs(a.get(k, np.nan) - b.get(k, np.nan)) if np.isfinite(a.get(k, np.nan)) and np.isfinite(b.get(k, np.nan)) else np.inf)
    d0 = max(abs(R["sm0"][k] - res["sm0"][k]) for k in keys if np.isfinite(R["sm0"].get(k, np.nan)))
    return dict(design=res["name"], draws_compared=n, max_abs_diff_draws=mx, max_abs_diff_point=d0)


# ============================================================================ Farseer
def _far_setup():
    import farseer as fs
    Rf = pickle.load(open(cm.RA1_FARSEER_PKL, "rb"))
    fa = fs.load()
    data = dict(N=fa.N.values.astype(float), N_emb=fa.N_emb.values.astype(float), D=fa.D.values.astype(float),
                L=fa.L.values.astype(float))
    bw = Rf["bw"]
    grid = fs.build_grid(fa)
    x, z, f = np.log(data["N"]), np.log(data["D"]), np.log(data["L"])
    fit_p, lev = fs.pilot(x, z, f, *bw["ne"])
    eres = (f - fit_p) / np.sqrt(np.clip(1 - lev, 0.1, 1))
    pth = pd.read_csv(cm.RA1_FAR_PATH)
    valid = pth[(pth.conv == "ne") & (pth.nvalid > 0)].C.values.astype(float)
    return dict(data=data, bw=bw, grid=grid, est=Rf["est"], fit_p=fit_p, eres=eres, clusters=fa.N.values, Rf=Rf,
                valid_levels=valid)


def _far_statistic(data, est_ne, grid, bw, with_param=True):
    """ra1's farseer.statistic for the non-embedding convention (chin/kappa comparators only)."""
    import farseer as fs
    D = data["D"]
    z = np.log(D)
    Nc = data["N"]
    x = np.log(Nc)
    f = np.log(data["L"])
    hx, hz = bw["ne"]
    gN = grid["N"].values
    pts = list(zip(np.log(gN), np.log(grid["D"].values)))
    loc = fs.local_many(x, z, f, pts, hx, hz)
    out = dict(lnw_local=np.log(loc[:, 1] / loc[:, 2]), neff=loc[:, 6])
    if with_param:
        for key in SPECS:
            out[f"lnw_{key}"] = np.log(fs.w_param(est_ne, key, gN, grid["D"].values))
    xs_sizes = np.log(np.sort(np.unique(Nc)))
    dmin = [D[Nc == v].min() for v in np.sort(np.unique(Nc))]
    dmax = [D[Nc == v].max() for v in np.sort(np.unique(Nc))]
    hull = fs.make_hull(x, z, xs_sizes, dmin, dmax)
    pth = fs.local_path(x, z, f, hx, hz, fs.C_PATH, hull)
    out["path_M"], out["path_sigma"] = pth[:, 2], pth[:, 1]
    return out


def _far_stats(st, grid, valid_levels):
    """valid_levels: ra1's path levels with bootstrap support (5e18-1e21; the 2e18 root exists only in the point
    estimate), over which ra1 takes the plain path mean and the random-effects path mean."""
    import farseer as fs
    pc = np.log(fs.C_PATH)
    pm_ = np.log(st["path_M"])
    g = np.isfinite(pm_)
    cg = np.log(6 * grid["N"].values * grid["D"].values)
    lMs = np.interp(cg, pc[g], pm_[g]) if g.sum() >= 3 else np.full(len(cg), np.nan)
    inside = (cg >= pc[g].min()) & (cg <= pc[g].max()) if g.sum() >= 3 else np.zeros(len(cg), bool)
    lp = {sp: st[f"lnw_{sp}"] for sp in SPECS if f"lnw_{sp}" in st}
    Mv, Cv = grid["M"].values, 6 * grid["N"].values * grid["D"].values
    out = point_stats(Mv, Cv, st["lnw_local"], st["neff"], lMs, inside, lp, fs.NEFF_MIN, fs.MBINS)
    # T1.4 input: the first-derivative estimator on path points at or below 3e20, and the Hessian path there
    r3 = point_stats(Mv, Cv, st["lnw_local"], st["neff"], lMs, inside, {}, fs.NEFF_MIN, fs.MBINS, cut_C=CUT_3E20)
    out["b1_le3e20"] = r3.get("b1", np.nan)
    out["sigma_slope_le3e20"] = r3.get("sigma_slope", np.nan)
    out["n_le3e20"] = r3.get("n_inside", 0)
    ps = st["path_sigma"]
    vl = np.array([any(np.isclose(C, v) for v in valid_levels) for C in fs.C_PATH])
    m3 = vl & (fs.C_PATH <= CUT_3E20)
    out["path_sigma_mean_le3e20"] = float(np.mean(ps[m3])) if np.all(np.isfinite(ps[m3])) else np.nan
    out["path_sigma_mean_all"] = float(np.mean(ps[vl])) if np.all(np.isfinite(ps[vl])) else np.nan
    for Cc, s_ in zip(fs.C_PATH, ps):
        out[f"path_sigma|{Cc:.0e}"] = float(s_)
    # published-style bin means over all points (checks against ra1's Delta)
    ok = (st["neff"] >= fs.NEFF_MIN)
    b = np.digitize(Mv, fs.MBINS) - 1
    for sp in lp:
        for j, lab in enumerate(MBIN_LAB):
            s = ok & (b == j)
            out[f"pub_delta|{sp}|{lab}"] = float(np.mean((lp[sp] - st["lnw_local"])[s])) if s.sum() else np.nan
    return out


def _far_draw(seed, S):
    """ra1's farseer._boot_worker (cluster_webb), non-embedding convention, chin/kappa comparators."""
    import parametric as pm
    rng = np.random.default_rng(seed)
    cl = S["clusters"]
    ug = np.unique(cl)
    eta = dict(zip(ug, rc.webb(rng, len(ug))))
    v = np.array([eta[g] for g in cl])
    Lb = np.exp(S["fit_p"] + v * S["eres"])
    data = S["data"]
    N, D = data["N"], data["D"]
    M = D / N
    warm = S["est"]["ne"]
    e2 = {}
    for tag, s in (("full", np.ones(len(Lb), bool)), ("M100", M <= 100)):
        e2[f"chin_{tag}"] = rc.sl.fit_chinchilla(N[s], D[s], Lb[s], delta=1e-3, init=warm[f"chin_{tag}"]).theta
        e2[f"kappa_{tag}"], _ = pm.fit_kappa(N[s], D[s], Lb[s], init=warm[f"kappa_{tag}"])
    st = _far_statistic(dict(data, L=Lb), e2, S["grid"], S["bw"])
    return _far_stats(st, S["grid"], S["valid_levels"])


def run_farseer(log=cm.log, B=B_EX):
    S = _far_setup()
    st0 = _far_statistic(S["data"], S["est"]["ne"], S["grid"], S["bw"])
    m0 = _far_stats(st0, S["grid"], S["valid_levels"])
    # bootstrap population (ra1: parametric comparators re-fitted on the noise-free pilot)
    import parametric as pm
    data0 = dict(S["data"], L=np.exp(S["fit_p"]))
    N, D = S["data"]["N"], S["data"]["D"]
    M = D / N
    e0 = {}
    for tag, s in (("full", np.ones(len(N), bool)), ("M100", M <= 100)):
        e0[f"chin_{tag}"] = rc.sl.fit_chinchilla(N[s], D[s], data0["L"][s], delta=1e-3, init=S["est"]["ne"][f"chin_{tag}"]).theta
        e0[f"kappa_{tag}"], _ = pm.fit_kappa(N[s], D[s], data0["L"][s], init=S["est"]["ne"][f"kappa_{tag}"])
    stp = _far_statistic(data0, e0, S["grid"], S["bw"])
    mp = _far_stats(stp, S["grid"], S["valid_levels"])
    seeds = [rc.seed_of(f"farseer|{i}") for i in range(B)]              # ra1's seeds (reproduction)
    payload = dict(S={k: v for k, v in S.items() if k != "Rf"})
    draws = cm.pmap(_far_draw, seeds, payload, chunksize=25)
    nfail = sum("_error" in x for x in draws)
    draws = [x for x in draws if "_error" not in x]
    log(f"    [Farseer] b1 {m0.get('b1'):.5f} b2 {m0.get('b2'):+.5f}; dlin 256-1,024 {m0.get('dlin|256-1,024'):+.3f}; "
        f"FD <= 3e20 {m0.get('sigma_slope_le3e20'):.4f}; draws {len(draws)} ({nfail} failed)")
    return dict(name="Farseer", tag="primary", n=len(N), mine0=m0, minep=mp, draws_mine=draws, nfail=nfail,
                valid_levels=S["valid_levels"])


def check_farseer(res):
    """Reproduction against ra1's published bootstrap moments (s.e. of b1, b2; Delta s.e. by bin; path s.e.)."""
    D = pd.DataFrame(res["draws_mine"])
    lin = pd.read_csv(cm.cm1.RA1_FAR_LIN).set_index(["conv", "param"])
    dl = pd.read_csv(cm.RA1_FAR_DELTA)
    dl = dl[dl.conv == "ne"].set_index(["spec", "M_bin"])
    path = pd.read_csv(cm.RA1_FAR_PATH)
    path = path[(path.conv == "ne") & (path.nvalid > 0)].set_index("C")
    rows = []
    for p in ("b1", "b2"):
        rows.append(dict(item=f"se {p}", mine=float(D[p].std(ddof=1)), published=float(lin.loc[("ne", p), "se"]),
                         point_mine=res["mine0"][p], point_published=float(lin.loc[("ne", p), "est"])))
    for sp in SPECS:
        for lab in MBIN_LAB:
            k = f"pub_delta|{sp}|{lab}"
            if (sp, lab) in dl.index and k in D:
                rows.append(dict(item=f"se Delta {sp} {lab}", mine=float(D[k].std(ddof=1)), published=float(dl.loc[(sp, lab), "se"]),
                                 point_mine=res["mine0"][k], point_published=float(dl.loc[(sp, lab), "delta"])))
    for C in path.index:
        k = f"path_sigma|{C:.0e}"
        if k in D:
            rows.append(dict(item=f"se path sigma {C:.0e}", mine=float(D[k].std(ddof=1)), published=float(path.loc[C, "se"]),
                             point_mine=res["mine0"][k], point_published=float(path.loc[C, "sigma_local_path"])))
    T = pd.DataFrame(rows)
    T["abs_diff_se"] = np.abs(T.mine - T.published)
    T["abs_diff_point"] = np.abs(T.point_mine - T.point_published)
    return T


def farseer_restricted(res):
    """T1.4 input: Farseer's first-derivative estimate on path points <= 3e20 and the Hessian path RE over its levels
    <= 3e20, with standard errors from the (reproduced) draws; s.e. of the FD estimate = max(own draw s.d., the s.d. of
    the plain mean of the path levels), as rb1 does for the full path."""
    import farseer as fs
    from meta import meta
    D = pd.DataFrame(res["draws_mine"])
    m0 = res["mine0"]
    se_fd = float(D["sigma_slope_le3e20"].std(ddof=1))
    se_pm = float(D["path_sigma_mean_le3e20"].std(ddof=1))
    se_fd_all = float(D["sigma_slope"].std(ddof=1))
    se_pm_all = float(D["path_sigma_mean_all"].std(ddof=1))
    vlev = [float(C) for C in res["valid_levels"]]
    lev = [C for C in vlev if C <= CUT_3E20]
    y = np.array([m0[f"path_sigma|{C:.0e}"] for C in lev])
    s = np.array([D[f"path_sigma|{C:.0e}"].std(ddof=1) for C in lev])
    mt = meta(y, s)
    ya = np.array([m0[f"path_sigma|{C:.0e}"] for C in vlev])
    sa = np.array([D[f"path_sigma|{C:.0e}"].std(ddof=1) for C in vlev])
    ma = meta(ya, sa)
    return dict(fd_le3e20=dict(sigma=float(m0["sigma_slope_le3e20"]), se=max(se_fd, se_pm), se_fd_draws=se_fd,
                               se_pathmean_draws=se_pm, n_points=int(m0["n_le3e20"])),
                fd_all=dict(sigma=float(m0["sigma_slope"]), se=max(se_fd_all, se_pm_all), se_fd_draws=se_fd_all,
                            se_pathmean_draws=se_pm_all),
                hessian_re_le3e20=dict(sigma=mt["mu_re"], se=mt["se_re"], k=len(y), levels=" ".join(f"{C:.0e}" for C in lev),
                                       Q=mt["Q"]),
                hessian_re_all=dict(sigma=ma["mu_re"], se=ma["se_re"], k=len(ya)))


# ============================================================================ tables
def _pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.std(v, ddof=1)), len(v)) if len(v) > 10 else (np.nan,) * 3 + (len(v),)


def decomposition_table(results):
    """Per design (and Marin pooled) x bin: Delta_lin (and the convexity component), the parametric gap on the same
    points and its rotation component, with percentile intervals from the draws."""
    rows = []
    for res in results:
        D = pd.DataFrame(res["draws_mine"])
        m0 = res["mine0"]
        for lab in MBIN_LAB + ["M>=256"]:
            k = f"dlin|{lab}"
            if not np.isfinite(m0.get(k, np.nan)):
                continue
            lo, hi, se, nd = _pct(D[k]) if k in D else (np.nan,) * 4
            row = dict(design=res["name"], variant=res["tag"], M_bin=lab, n_points=m0.get(f"n_in|{lab}", np.nan),
                       u_mean=m0.get(f"u_in|{lab}", np.nan), delta_lin=m0[k], lo_pct=lo, hi_pct=hi, se=se, n_draws=nd,
                       convexity=-m0[k], convexity_lo=-hi, convexity_hi=-lo)
            for sp in ("kappa_full", "chin_full", "chin_M100"):
                kd, kr = f"dparam_in|{sp}|{lab}", f"rot|{sp}|{lab}"
                row[f"delta_{sp}_inside"] = m0.get(kd, np.nan)
                row[f"rotation_{sp}"] = m0.get(kr, np.nan)
                if kr in D:
                    lo_r, hi_r, se_r, _ = _pct(D[kr])
                    row[f"rotation_{sp}_lo"], row[f"rotation_{sp}_hi"] = lo_r, hi_r
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_marin(results, tag):
    """Equal-weight mean of Marin's three corpora, index-wise for the draws (independent designs' draws)."""
    rs = [r for r in results if r["name"] in MARIN and r["tag"] == tag]
    if len(rs) != 3:
        return None
    keys = set(rs[0]["mine0"]).intersection(*[set(r["mine0"]) for r in rs[1:]])
    m0 = {k: float(np.mean([r["mine0"][k] for r in rs])) for k in keys if isinstance(rs[0]["mine0"][k], (float, int))}
    n = min(len(r["draws_mine"]) for r in rs)
    dr = []
    for i in range(n):
        dr.append({k: float(np.mean([r["draws_mine"][i].get(k, np.nan) for r in rs])) for k in keys})
    for lab in MBIN_LAB + ["M>=256"]:
        m0[f"n_in|{lab}"] = int(sum(r["mine0"].get(f"n_in|{lab}", 0) for r in rs))
        vals = [r["mine0"].get(f"dlin|{lab}", np.nan) for r in rs]
        if any(not np.isfinite(v) for v in vals):          # a bin missing in any corpus: pooled mean over available corpora
            v_ok = [v for v in vals if np.isfinite(v)]
            m0[f"dlin|{lab}"] = float(np.mean(v_ok)) if v_ok else np.nan
            for i in range(n):
                vv = [r["draws_mine"][i].get(f"dlin|{lab}", np.nan) for r in rs]
                vv = [v for v, v0 in zip(vv, vals) if np.isfinite(v0) and np.isfinite(v)]
                dr[i][f"dlin|{lab}"] = float(np.mean(vv)) if vv else np.nan
    return dict(name="Marin, three corpora pooled", tag=tag, mine0=m0, draws_mine=dr)


def lin_table(results):
    """b1, b2 and the local first-derivative sigma* with percentile intervals."""
    rows = []
    for res in results:
        D = pd.DataFrame(res["draws_mine"])
        for k in ("b1", "b2", "sigma_slope"):
            if k not in res["mine0"]:
                continue
            lo, hi, se, nd = _pct(D[k]) if k in D else (np.nan,) * 4
            rows.append(dict(design=res["name"], variant=res["tag"], param=k, est=res["mine0"][k], se=se, lo_pct=lo,
                             hi_pct=hi, n_inside=res["mine0"].get("n_inside"), umax=res["mine0"].get("umax"), n_draws=nd))
    return pd.DataFrame(rows)


def delta_percentile_table(rb1_results):
    """Percentile intervals for the published Delta = ln(w_param/w_local) (all evaluation points), beside the basic
    intervals: rb1's designs from the regenerated draws (identical to the published ones), Marin pooled index-wise, and
    Farseer from ra1's published basic interval and population value (percentile = [pop + est - hi, pop + est - lo])."""
    rows = []
    pub = pd.read_csv(cm.RB1_EXTRAP_DELTA)
    for res in rb1_results:
        if res["tag"] != "primary":
            continue
        Dd = pd.DataFrame(res["draws_sm"])
        for sp in SPECS:
            for lab in MBIN_LAB + ["M>=256"]:
                k = f"delta|{sp}|{lab}"
                est = res["sm0"].get(k, np.nan)
                if not np.isfinite(est):
                    continue
                v = Dd[k].values
                v = v[np.isfinite(v)]
                pop = res["smp"].get(k, np.nan)
                p = pub[(pub.design == res["name"]) & (pub.spec == sp) & (pub.M_bin == lab)].iloc[0]
                rows.append(dict(design=res["name"], spec=sp, M_bin=lab, delta=est, delta_pop=pop,
                                 se=float(np.std(v - pop)), lo_pct=float(np.percentile(v, 2.5)), hi_pct=float(np.percentile(v, 97.5)),
                                 lo_basic=est - np.percentile(v - pop, 97.5), hi_basic=est - np.percentile(v - pop, 2.5),
                                 lo_basic_pub=p.lo, hi_basic_pub=p.hi, n_points=p.n_points, M_max_evaluated=p.M_max_evaluated,
                                 n_draws=len(v)))
    # Marin pooled (index-wise mean of the corpora's draws; the pooled estimate is the mean of the corpora's estimates)
    ms = [r for r in rb1_results if r["tag"] == "primary" and r["name"] in MARIN]
    if len(ms) == 3:
        n = min(len(r["draws_sm"]) for r in ms)
        for sp in SPECS:
            for lab in MBIN_LAB + ["M>=256"]:
                k = f"delta|{sp}|{lab}"
                ests = [r["sm0"].get(k, np.nan) for r in ms]
                if not all(np.isfinite(ests)):
                    continue
                v = np.array([np.mean([r["draws_sm"][i].get(k, np.nan) for r in ms]) for i in range(n)])
                v = v[np.isfinite(v)]
                pop = float(np.mean([r["smp"].get(k, np.nan) for r in ms]))
                est = float(np.mean(ests))
                rows.append(dict(design="Marin, three corpora pooled", spec=sp, M_bin=lab, delta=est, delta_pop=pop,
                                 se=float(np.std(v - pop)), lo_pct=float(np.percentile(v, 2.5)),
                                 hi_pct=float(np.percentile(v, 97.5)), lo_basic=est - np.percentile(v - pop, 97.5),
                                 hi_basic=est - np.percentile(v - pop, 2.5), n_draws=len(v)))
    fd = pd.read_csv(cm.RA1_FAR_DELTA)
    fd = fd[(fd.conv == "ne") & fd.spec.isin(SPECS)]
    for _, r in fd.iterrows():
        if not np.isfinite(r.delta):
            continue
        rows.append(dict(design="Farseer", spec=r.spec, M_bin=r.M_bin, delta=r.delta, delta_pop=r.delta_pop, se=r.se,
                         lo_pct=r.delta_pop + r.delta - r.hi, hi_pct=r.delta_pop + r.delta - r.lo, lo_basic=r.lo,
                         hi_basic=r.hi, lo_basic_pub=r.lo, hi_basic_pub=r.hi, n_draws=999))
    T = pd.DataFrame(rows)
    T["pct_contains_estimate"] = (T.lo_pct <= T.delta) & (T.delta <= T.hi_pct)
    T["basic_contains_estimate"] = (T.lo_basic <= T.delta) & (T.delta <= T.hi_basic)
    return T


def corner_drops():
    """Leave-corner-out definitions (functions of an rb1 design frame)."""
    def largest_M_per_budget(d):
        idx = d.groupby("C_b")["M"].idxmax()
        return d.index.isin(idx.values)

    def M_over_1000(d):
        return d["M"].values > 1000.0
    return {"corner_largestM": largest_M_per_budget, "corner_M1000": M_over_1000}


def run(log=cm.log):
    out = dict(rb1=[], checks=[])
    for name in MARIN + ["Llama 3"]:
        r = run_rb1(name, log=log)
        out["checks"].append(check_rb1(r))
        out["rb1"].append(r)
    for tag, fn in corner_drops().items():
        for name in MARIN:
            out["rb1"].append(run_rb1(name, drop=fn, tag=tag, log=log))
    out["farseer"] = run_farseer(log=log)
    out["far_check"] = check_farseer(out["farseer"])
    out["far_restricted"] = farseer_restricted(out["farseer"])
    return out


def tables(out):
    res = [out["farseer"]] + out["rb1"]
    pooled = [pooled_marin(out["rb1"], t) for t in ("primary", "corner_largestM", "corner_M1000")]
    res_all = res + [p for p in pooled if p is not None]
    dec = decomposition_table(res_all)
    lin = lin_table(res_all)
    dpt = delta_percentile_table(out["rb1"])
    chk = pd.DataFrame(out["checks"])
    # leave-corner-out: published-style Delta (all points) by bin for the variants
    lco = []
    for r in out["rb1"]:
        if r["tag"] == "primary":
            continue
        Dd = pd.DataFrame(r["draws_sm"])
        for sp in ("kappa_full", "chin_full", "chin_M100"):
            for lab in MBIN_LAB + ["M>=256"]:
                k = f"delta|{sp}|{lab}"
                e = r["sm0"].get(k, np.nan)
                if not np.isfinite(e):
                    continue
                v = Dd[k].values
                v = v[np.isfinite(v)]
                lco.append(dict(design=r["name"], variant=r["tag"], spec=sp, M_bin=lab, delta=e, lo_pct=float(np.percentile(v, 2.5)),
                                hi_pct=float(np.percentile(v, 97.5)), n_points=r["sm0"].get(f"n|{lab}", np.nan),
                                n_runs=r["n"], n_dropped=r["n_dropped"], M_max_evaluated=r["Mmax_eval"]))
        lco.append(dict(design=r["name"], variant=r["tag"], spec="convexity b2", M_bin="", delta=r["sm0"].get("lin|b2", np.nan),
                        n_runs=r["n"], n_dropped=r["n_dropped"], M_max_evaluated=r["Mmax_eval"]))
    return dict(decomposition=dec, lin=lin, delta_pct=dpt, check_rb1=chk, check_farseer=out["far_check"],
                corner=pd.DataFrame(lco))
