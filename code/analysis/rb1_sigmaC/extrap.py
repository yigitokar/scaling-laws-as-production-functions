"""extrap.py -- second-recipe extrapolation checks on the IsoFLOP designs of Marin (three corpora; M up to 3,706) and
Llama 3 (M up to 643) (task 4; R2 Major 5.1; revision_plan_v3 Sec. 1 last bullet).

Local (model-free) wedge. In the coordinates of an IsoFLOP design, x = ln M and c = ln C (C = 6ND exactly in the
FLOP-implied parameter convention N_F = C/(6D) used by ra1 for Llama 3 and Marin), n = (c - ln 6 - x)/2 and
d = (c - ln 6 + x)/2, so for any smooth loss index f = ln L:
        f_n = f_c - f_x,  f_d = f_c + f_x,   w = eps_N / eps_D = f_n / f_d = (f_c - f_x) / (f_c + f_x).
f_x is the slope along each isocost (observed directly on the profile) and f_c the slope across budgets at fixed M.
Both come from a kernel-weighted local quadratic of ln L in (x, c) with a Gaussian product kernel (Fan and Gijbels
1996), bandwidths from leave-one-out cross-validation over multiples of the coordinate standard deviations. w is
invariant to any monotone transform of L (E-free, form-free, unit-free: Llama 3's digitized loss units do not matter).
Evaluation: on every budget line (isocost), at log-spaced M inside the budget's observed M range; points with a kernel
effective sample size below NEFF_MIN are dropped. The local expansion path M*_local(C_b) is the root of ln w_local = 0
on each budget line (for an IsoFLOP profile this is the profile's minimum).

Parametric comparators (fitted by Huber loss on ln L in the same FLOP-implied convention; ra1's parametric.py):
  chin_full  Chinchilla form (kappa = 1), all runs;       chin_M100  same, runs with M <= 100 only;
  kappa_full kappa family (outer exponent free), all runs; kappa_M100 same, M <= 100.
Statistic: Delta = ln(w_param / w_local) averaged within M bins (<16, 16-64, 64-256, 256-1,024, >=1,024), as on
Farseer (ra1 H4). Convexity: ln w_local = b1 u + b2 u^2, u = ln(M / M*_local(C)) (b2 > 0: convex, as on Farseer).
Inference (design-conditional): wild cluster bootstrap by budget (Webb six-point weights) of leverage-adjusted
residuals around the pilot local-quadratic surface; every local quantity and every parametric fit is recomputed in each
draw (parametric fits warm-started); basic intervals centred on the statistic of the noise-free pilot population, as
in ra1. A run-level Rademacher wild bootstrap is the comparison. Smoothing bias is not in the intervals (bandwidth
sensitivity reported).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import rb1common as cm

rc = cm.rc
DESIGNS = ["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Llama 3"]
MBINS = [0, 16, 64, 256, 1024, 1e9]
MBIN_LAB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
SPECS = ["chin_full", "chin_M100", "kappa_full", "kappa_M100"]
NEFF_MIN = 8
M_CUT = 100.0
MULTS = (0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0)
B_EX = 999
B_RUN = 499
if cm.QUICK:
    B_EX, B_RUN = 16, 8


def load_design(name, screen=True):
    """IsoFLOP design in the FLOP-implied convention (ra1's loader). screen=True applies the monotonicity screen:
    where a model configuration is trained at several budgets (Marin), a run whose loss exceeds the lowest loss of the
    same configuration trained on fewer tokens (by > 1e-4 nats) is dropped as a failed run. On these data it drops one
    run (Marin DCLM, 157M configuration at 1.8e20 FLOP: 3.769 against 3.605 at half the tokens); Llama 3 has no
    repeated configurations."""
    d = rc.isoflop_designs()[name].copy()
    d["M"] = d["D"] / d["N"]
    d["u"] = np.log(d["M"].values)                      # x = ln M
    d["cc"] = np.log(6.0 * d["N"].values * d["D"].values)   # c = ln C (= ln C_b in the FLOP-implied convention)
    d["f"] = np.log(d["L"].values)
    d["flag_nonmonotone"] = False
    if "N_cfg" in d and d["N_cfg"].notna().all():
        cfg = d["N_cfg"].round(-5)
        for _, g in d.assign(cfg=cfg).groupby("cfg"):
            g = g.sort_values("D")
            best = np.inf
            for i, r in g.iterrows():
                if r["L"] > best + 1e-4:
                    d.loc[i, "flag_nonmonotone"] = True
                best = min(best, r["L"])
    if screen:
        d = d[~d["flag_nonmonotone"]]
    return d.reset_index(drop=True)


def hc_min(d):
    """Smallest admissible c-bandwidth: half the largest gap between adjacent budgets in ln C, so that every kernel
    spans at least two budgets and f_c (the slope across budgets at fixed M) is identified."""
    g = np.diff(np.sort(np.log(d["C_b"].unique())))
    return 0.5 * float(g.max())


# ============================================================================ local quadratic in (x = ln M, c = ln C)
def _X(x, c, x0, c0):
    dx, dc = x - x0, c - c0
    return np.column_stack([np.ones_like(dx), dx, dc, dx * dx, dc * dc, dx * dc])


def local_at(x, c, f, x0, c0, hx, hc):
    """Coefficients [f, f_x, f_c, f_xx/2, f_cc/2, f_xc] and kernel n_eff at (x0, c0)."""
    k = np.exp(-0.5 * (((x - x0) / hx) ** 2 + ((c - c0) / hc) ** 2))
    X = _X(x, c, x0, c0)
    sk = np.sqrt(k)
    coef = np.linalg.lstsq(X * sk[:, None], f * sk, rcond=None)[0]
    return coef, float(k.sum() ** 2 / np.sum(k ** 2))


def lnw_from(coef):
    fx, fc = coef[..., 1], coef[..., 2]
    fn, fd = fc - fx, fc + fx
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((fn < 0) & (fd < 0), np.log(fn / fd), np.nan)


def local_many(x, c, f, pts, hx, hc):
    out = np.zeros((len(pts), 7))
    for i, (x0, c0) in enumerate(pts):
        coef, ne = local_at(x, c, f, x0, c0, hx, hc)
        out[i, :6], out[i, 6] = coef, ne
    return out


def pilot(x, c, f, hx, hc):
    n = len(f)
    fit, lev = np.zeros(n), np.zeros(n)
    for i in range(n):
        k = np.exp(-0.5 * (((x - x[i]) / hx) ** 2 + ((c - c[i]) / hc) ** 2))
        X = _X(x, c, x[i], c[i])
        WX = X * k[:, None]
        A = np.linalg.pinv(X.T @ WX) @ WX.T
        fit[i] = A[0] @ f
        lev[i] = A[0, i]
    return fit, lev


def cv_bandwidth(x, c, f, mults=MULTS, hcmin=0.0):
    sx, sc = x.std(), c.std()
    rows = []
    for mx, mc in itertools.product(mults, mults):
        hx, hc = mx * sx, mc * sc
        if hc < hcmin:
            continue
        err = []
        for i in range(len(f)):
            k = np.exp(-0.5 * (((x - x[i]) / hx) ** 2 + ((c - c[i]) / hc) ** 2))
            k[i] = 0.0
            X = _X(x, c, x[i], c[i])
            sk = np.sqrt(k)
            coef = np.linalg.lstsq(X * sk[:, None], f * sk, rcond=None)[0]
            err.append(f[i] - coef[0])
        rows.append(dict(mx=mx, mc=mc, hx=hx, hc=hc, loocv_rmse=float(np.sqrt(np.mean(np.square(err))))))
    T = pd.DataFrame(rows)
    b = T.loc[T.loocv_rmse.idxmin()]
    return (float(b.hx), float(b.hc), float(b.mx), float(b.mc)), T


def build_grid(d, n_per=14):
    rows = []
    for Cb, g in d.groupby("C_b"):
        lo, hi = g["u"].min(), g["u"].max()
        for x0 in np.linspace(lo, hi, n_per):
            rows.append(dict(C_b=Cb, cc=float(np.log(Cb)), u=float(x0), M=float(np.exp(x0)),
                             N=float(np.sqrt(Cb / 6.0 / np.exp(x0))), D=float(np.sqrt(Cb / 6.0 * np.exp(x0)))))
    return pd.DataFrame(rows)


def local_path(x, c, f, hx, hc, budgets):
    """Root of ln w_local on each budget line inside its observed M range: returns ln M*_local(C_b)."""
    out = []
    for Cb, lo, hi in budgets:
        c0 = np.log(Cb)

        def g(x0):
            coef, _ = local_at(x, c, f, x0, c0, hx, hc)
            v = lnw_from(coef)
            return float(v) if np.isfinite(v) else np.nan

        xs = np.linspace(lo, hi, 25)
        gs = np.array([g(v) for v in xs])
        root = np.nan
        ok = np.isfinite(gs)
        for i in range(len(xs) - 1):
            if ok[i] and ok[i + 1] and np.sign(gs[i]) != np.sign(gs[i + 1]):
                try:
                    root = brentq(g, xs[i], xs[i + 1], xtol=1e-6)
                except ValueError:
                    root = np.nan
                break
        out.append(root)
    return np.array(out)


# ============================================================================ parametric comparators
def fit_param(d, warm=None):
    import parametric as pm
    N, D, L = d["N"].values, d["D"].values, d["L"].values
    sub = d["M"].values <= M_CUT
    out = {}
    for tag, s in (("full", np.ones(len(L), bool)), ("M100", sub)):
        if warm is None:
            th, _ = pm.fit_chin(N[s], D[s], L[s])
            p, _ = pm.fit_kappa(N[s], D[s], L[s], th_chin=th)
        else:
            th = rc.sl.fit_chinchilla(N[s], D[s], L[s], delta=1e-3, init=warm[f"chin_{tag}"]).theta
            p, _ = pm.fit_kappa(N[s], D[s], L[s], init=warm[f"kappa_{tag}"])
        out[f"chin_{tag}"], out[f"kappa_{tag}"] = th, p
    return out


def lnw_param(est, key, N, D):
    import parametric as pm
    if key.startswith("chin"):
        return np.log(pm.wedge_chin(est[key], N, D))
    return np.log(pm.wedge_kappa(est[key], N, D))


def sigma_param(est):
    import parametric as pm
    return {k: (pm.sigma_star_chin(v) if k.startswith("chin") else pm.sigma_star_kappa(v)) for k, v in est.items()}


# ============================================================================ statistic
def statistic(d, f, est, grid, bw, budgets):
    x, c = d["u"].values, d["cc"].values
    hx, hc = bw
    pts = list(zip(grid["u"].values, grid["cc"].values))
    loc = local_many(x, c, f, pts, hx, hc)
    out = dict(lnw_local=lnw_from(loc[:, :6]), neff=loc[:, 6], fx=loc[:, 1], fc=loc[:, 2])
    for key in SPECS:
        out[f"lnw_{key}"] = lnw_param(est, key, grid["N"].values, grid["D"].values)
    out["path_lnMstar"] = local_path(x, c, f, hx, hc, budgets)
    return out


def summaries(st, grid, budgets, neff_min=None):
    neff_min = NEFF_MIN if neff_min is None else neff_min
    res = {}
    Mv = grid["M"].values
    ok = (st["neff"] >= neff_min) & np.isfinite(st["lnw_local"])
    b = np.digitize(Mv, MBINS) - 1
    for key in SPECS:
        dl = st[f"lnw_{key}"] - st["lnw_local"]
        for j, lab in enumerate(MBIN_LAB):
            s = ok & (b == j)
            res[f"delta|{key}|{lab}"] = float(np.mean(dl[s])) if s.sum() else np.nan
        s = ok & (Mv >= 256)
        res[f"delta|{key}|M>=256"] = float(np.mean(dl[s])) if s.sum() else np.nan
    for j, lab in enumerate(MBIN_LAB):
        s = ok & (b == j)
        res[f"lnw_local|{lab}"] = float(np.mean(st["lnw_local"][s])) if s.sum() else np.nan
        res[f"n|{lab}"] = int(s.sum())
    res["n_invalid_local"] = int(np.sum((st["neff"] >= neff_min) & ~np.isfinite(st["lnw_local"])))
    # convexity of ln w_local in u = ln(M / M*_local(C_b)); M*_local interpolated in ln C over budgets with a root
    Cb = np.array([bb[0] for bb in budgets])
    lp = st["path_lnMstar"]
    g = np.isfinite(lp)
    res["n_path"] = int(g.sum())
    if g.sum() >= 3:
        lMs = np.interp(np.log(grid["C_b"].values), np.log(Cb[g]), lp[g])
        inside = ok & (np.log(grid["C_b"].values) >= np.log(Cb[g]).min() - 1e-9) & (np.log(grid["C_b"].values) <= np.log(Cb[g]).max() + 1e-9)
        u = np.log(Mv[inside]) - lMs[inside]
        y = st["lnw_local"][inside]
        if inside.sum() > 10:
            X = np.column_stack([u, u * u])
            bb = np.linalg.lstsq(X, y, rcond=None)[0]
            res["lin|b1"], res["lin|b2"] = float(bb[0]), float(bb[1])
            res["lin|b1_only"] = float(np.linalg.lstsq(u[:, None], y, rcond=None)[0][0])
            res["lin|n"] = int(inside.sum())
            res["lin|umax"] = float(u.max())
            res["lin|convexity_at_umax"] = float(bb[1] * u.max() ** 2)
            res["sigma_slope"] = float(1.0 / (1.0 + bb[0]))
            s2 = Mv[inside] < 1024
            if s2.sum() > 10:
                b2_ = np.linalg.lstsq(X[s2], y[s2], rcond=None)[0]
                res["lin|b2_Mlt1024"] = float(b2_[1])
        res["path_Mstar_median"] = float(np.exp(np.median(lp[g])))
    return res


def _boot_worker(seed, d, est, grid, bw, budgets, fit_pilot, eres, scheme):
    rng = np.random.default_rng(seed)
    if scheme == "cluster_webb":
        g = pd.factorize(d["C_b"].values)[0]
        v = rng.choice(rc.WEBB, size=g.max() + 1)[g]
    else:
        v = rng.choice(np.array([-1.0, 1.0]), size=len(eres))
    fstar = fit_pilot + v * eres
    ds = d.copy()
    ds["L"] = np.exp(fstar)
    est_b = fit_param(ds, warm=est)
    st = statistic(ds, fstar, est_b, grid, bw, budgets)
    sm = summaries(st, grid, budgets)
    sp = sigma_param(est_b)
    return dict(sm=sm, sig={k: float(v_) for k, v_ in sp.items()})


def select_bandwidths(log=cm.log):
    """Primary bandwidths. Marin's three corpora share one design, so they share one bandwidth: the multiple pair
    (m_x, m_c) minimizing the mean leave-one-out RMSE across the three corpora, subject to h_c >= hc_min (every kernel
    spans two budgets). Llama 3: its own CV optimum under the same constraint. Each corpus's own CV optimum is a
    sensitivity."""
    out, cvs = {}, []
    for name in DESIGNS:
        d = load_design(name)
        x, c, f = d["u"].values, d["cc"].values, d["f"].values
        own, T = cv_bandwidth(x, c, f, hcmin=hc_min(d))
        cvs.append(T.assign(design=name, hc_min=hc_min(d)))
        out[name] = dict(own=own, sx=x.std(), sc=c.std(), hcmin=hc_min(d))
    CV = pd.concat(cvs, ignore_index=True)

    def one_pct(T):
        """At the CV-optimal x-multiple, the smallest c-multiple whose LOO RMSE is within 1% of the minimum (the
        criterion is flat in m_c beyond about 1 s.d.; the rule favours locality in compute)."""
        g = T.groupby(["mx", "mc"]).loocv_rmse.mean()
        mx_, mc_ = g.idxmin()
        row = g.xs(mx_, level="mx")
        return mx_, float(row[row <= 1.01 * g.min()].index.min()), mc_
    mx, mc, mc_min = one_pct(CV[CV.design.str.startswith("Marin")])
    for name in DESIGNS:
        o = out[name]
        if name.startswith("Marin"):
            o["primary"] = (mx * o["sx"], mc * o["sc"], mx, mc)
            o["cv_argmin_pooled"] = (mx, mc_min)
        else:
            mx_l, mc_l, mc_lmin = one_pct(CV[CV.design == name])
            o["primary"] = (mx_l * o["sx"], mc_l * o["sc"], mx_l, mc_l)
            o["cv_argmin_pooled"] = (mx_l, mc_lmin)
        log(f"  [{name}] bandwidth: primary m=({o['primary'][2]}, {o['primary'][3]}) h=({o['primary'][0]:.3f}, "
            f"{o['primary'][1]:.3f}); own CV m=({o['own'][2]}, {o['own'][3]}); hc_min {o['hcmin']:.3f}")
    return out, CV


def run_design(name, bwsel, log=cm.log, B=B_EX, B2=B_RUN):
    d = load_design(name)
    d_raw = load_design(name, screen=False)
    x, c, f = d["u"].values, d["cc"].values, d["f"].values
    hx, hc, mx, mc = bwsel["primary"]
    grid = build_grid(d)
    budgets = [(Cb, g["u"].min(), g["u"].max()) for Cb, g in d.groupby("C_b")]
    est = fit_param(d)
    st0 = statistic(d, f, est, grid, (hx, hc), budgets)
    sm0 = summaries(st0, grid, budgets)
    fit_p, lev = pilot(x, c, f, hx, hc)
    eres = (f - fit_p) / np.sqrt(np.clip(1 - lev, 0.1, 1))
    d0 = d.copy()
    d0["L"] = np.exp(fit_p)
    est0 = fit_param(d0, warm=est)
    st_pop = statistic(d0, fit_p, est0, grid, (hx, hc), budgets)
    sm_pop = summaries(st_pop, grid, budgets)
    # sensitivity (point estimates)
    sens = {}
    variants = [("h x 0.75", (0.75 * hx, 0.75 * hc), 8), ("h x 1.5", (1.5 * hx, 1.5 * hc), 8),
                ("h_x x 0.75", (0.75 * hx, hc), 8), ("h_c x 1.5", (hx, 1.5 * hc), 8),
                ("n_eff >= 10", (hx, hc), 10), ("n_eff >= 6", (hx, hc), 6)]
    if bwsel["own"][:2] != bwsel["primary"][:2]:
        variants.append(("own-corpus CV bandwidth", tuple(bwsel["own"][:2]), 8))
    for lab, bw_, nm in variants:
        stv = statistic(d, f, est, grid, bw_, budgets)
        sens[lab] = summaries(stv, grid, budgets, neff_min=nm)
    if len(d_raw) != len(d):
        grid_r = build_grid(d_raw)
        bud_r = [(Cb, g["u"].min(), g["u"].max()) for Cb, g in d_raw.groupby("C_b")]
        est_r = fit_param(d_raw, warm=est)
        stv = statistic(d_raw, d_raw["f"].values, est_r, grid_r, (hx, hc), bud_r)
        sens["all runs (no monotonicity screen)"] = summaries(stv, grid_r, bud_r)
    log(f"  [{name}] n = {len(d)} (screened out {len(d_raw) - len(d)}); Delta(kappa_full) by bin " +
        ", ".join(f"{b}: {sm0.get(f'delta|kappa_full|{b}', np.nan):+.3f}" for b in MBIN_LAB) +
        f"; b2 = {sm0.get('lin|b2', np.nan):+.4f}; invalid local {sm0['n_invalid_local']}")
    payload = dict(d=d, est=est, grid=grid, bw=(hx, hc), budgets=budgets, fit_pilot=fit_p, eres=eres, scheme="cluster_webb")
    seeds = [cm.seed_of(f"extrap|{name}|{i}") for i in range(B)]
    draws = rc.pmap(_boot_worker, seeds, payload, procs=cm.N_PROC, chunksize=4)
    nfail = sum("_error" in dd for dd in draws)
    draws = [dd for dd in draws if "_error" not in dd]
    payload2 = dict(payload, scheme="run_rademacher")
    seeds2 = [cm.seed_of(f"extrap_run|{name}|{i}") for i in range(B2)]
    draws2 = [dd for dd in rc.pmap(_boot_worker, seeds2, payload2, procs=cm.N_PROC, chunksize=4) if "_error" not in dd]
    log(f"  [{name}] bootstrap: {len(draws)} cluster draws ({nfail} failed), {len(draws2)} run-level draws")
    return dict(name=name, d=d, d_raw=d_raw, grid=grid, bw=(hx, hc, mx, mc), est=est, st0=st0, sm0=sm0, st_pop=st_pop,
                sm_pop=sm_pop, draws=draws, draws_run=draws2, sens=sens, budgets=budgets, nfail=nfail,
                sigma_param=sigma_param(est), flagged=d_raw[d_raw.flag_nonmonotone])


def _mmax(R, lab):
    """Review addition: largest M at which the local wedge is evaluated (n_eff >= NEFF_MIN, finite) in bin lab."""
    g, st0 = R["grid"], R["st0"]
    ok = (st0["neff"] >= NEFF_MIN) & np.isfinite(st0["lnw_local"])
    Mv = g["M"].values
    if lab == "M>=256":
        s = ok & (Mv >= 256)
    else:
        j = MBIN_LAB.index(lab)
        s = ok & (Mv >= MBINS[j]) & (Mv < MBINS[j + 1])
    return float(Mv[s].max()) if s.any() else np.nan


def tables(R):
    """Delta by M bin with basic 95% intervals, convexity, parametric sigma*, for one design."""
    sm0, smp = R["sm0"], R["sm_pop"]
    Dd = pd.DataFrame([dd["sm"] for dd in R["draws"]])
    Dr = pd.DataFrame([dd["sm"] for dd in R["draws_run"]])
    rows = []
    for key in SPECS:
        for lab in MBIN_LAB + ["M>=256"]:
            k = f"delta|{key}|{lab}"
            est = sm0.get(k, np.nan)
            v = Dd[k].values - smp.get(k, np.nan) if k in Dd else np.array([])
            v = v[np.isfinite(v)]
            vr = Dr[k].values - smp.get(k, np.nan) if k in Dr else np.array([])
            vr = vr[np.isfinite(vr)]
            ok = len(v) > 10 and np.isfinite(est)
            rows.append(dict(design=R["name"], spec=key, M_bin=lab, delta=est,
                             lo=est - np.percentile(v, 97.5) if ok else np.nan,
                             hi=est - np.percentile(v, 2.5) if ok else np.nan,
                             se_cluster=float(np.std(v)) if ok else np.nan,
                             se_run=float(np.std(vr)) if len(vr) > 10 else np.nan,
                             boot_bias=float(np.mean(v)) if ok else np.nan,
                             # review addition: the centre of the basic interval (estimate minus bootstrap bias);
                             # the bias comes from the parametric (Huber) refits, not from the local wedge
                             delta_bc=est - float(np.mean(v)) if ok else np.nan,
                             lo_run=est - np.percentile(vr, 97.5) if len(vr) > 10 and np.isfinite(est) else np.nan,
                             hi_run=est - np.percentile(vr, 2.5) if len(vr) > 10 and np.isfinite(est) else np.nan,
                             ratio=np.exp(est) if np.isfinite(est) else np.nan,
                             n_points=sm0.get(f"n|{lab}", np.nan) if lab != "M>=256" else np.nan,
                             M_max_evaluated=_mmax(R, lab),
                             lnw_local=sm0.get(f"lnw_local|{lab}", np.nan) if lab != "M>=256" else np.nan,
                             n_draws=len(v)))
    T = pd.DataFrame(rows)
    lin = []
    for k in ("lin|b1", "lin|b2", "lin|b2_Mlt1024", "lin|b1_only", "sigma_slope", "lin|convexity_at_umax"):
        est = sm0.get(k, np.nan)
        v = Dd[k].values - smp.get(k, np.nan) if k in Dd else np.array([])
        v = v[np.isfinite(v)]
        ok = len(v) > 10 and np.isfinite(est)
        lo = est - np.percentile(v, 97.5) if ok else np.nan
        hi = est - np.percentile(v, 2.5) if ok else np.nan
        p0 = float((1 + np.sum(np.abs(v) >= abs(est))) / (len(v) + 1)) if ok else np.nan
        lin.append(dict(design=R["name"], param=k, est=est, se=float(np.std(v)) if ok else np.nan, lo=lo, hi=hi,
                        p_zero=p0, n=sm0.get("lin|n", np.nan), umax=sm0.get("lin|umax", np.nan)))
    Lt = pd.DataFrame(lin)
    sens = []
    for lab, sm in R["sens"].items():
        for key in ("kappa_full", "chin_M100", "chin_full", "kappa_M100"):
            for b in ("256-1,024", ">=1,024", "M>=256"):
                sens.append(dict(design=R["name"], variant=lab, spec=key, M_bin=b, delta=sm.get(f"delta|{key}|{b}", np.nan)))
        sens.append(dict(design=R["name"], variant=lab, spec="convexity b2", M_bin="", delta=sm.get("lin|b2", np.nan)))
        sens.append(dict(design=R["name"], variant=lab, spec="convexity b2, M < 1,024", M_bin="", delta=sm.get("lin|b2_Mlt1024", np.nan)))
    S = pd.DataFrame(sens)
    P = pd.DataFrame([dict(design=R["name"], spec=k, sigma_star=v) for k, v in R["sigma_param"].items()] +
                     [dict(design=R["name"], spec="local first-derivative 1/(1+b1)", sigma_star=sm0.get("sigma_slope", np.nan))])
    return T, Lt, S, P
