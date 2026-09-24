"""chin_inf.py -- inference fixes on the Chinchilla data (Task 4; R4 M1, M5, M6; R1 c9(a)-(c)).

(a) Design facts: 245 digitized runs; 137 on the nine IsoFLOP profiles (|dev| <= 0.045 dex around the offset-corrected
    nominal budget); 108 off-profile runs (Hoffmann et al.'s Approach-1 training-horizon runs, pooled into Approach 3);
    FLOP 1.4e18-1.3e22. Baseline sample n = 240 (132 on profile, 108 off).
(b) Bootstrap schemes for the reference Huber (delta = 1e-3; LAD in practice) Chinchilla-form fit and the
    kappa-free fit, n = 240, B = 999 each:
      pairs            resample runs;
      cluster9         m1's scheme: every run assigned to the nearest nominal budget (9 clusters);
      cluster_trunk    IsoFLOP runs clustered by budget, off-profile runs by model-size trunk (9 + 35 clusters);
      cluster_single   IsoFLOP runs by budget, off-profile runs as singletons (9 + 108 clusters);
      wild_fhh         design-conditional wild bootstrap for LAD-type fits (Feng, He and Hu 2011): y* = yhat + w |r~|,
                       w = +-1, r~ the FHH leverage/sparsity-adjusted residual; yhat = the kappa-free fit (the
                       better-fitting surface); the (N, D) design is held fixed;
      wild_cl_webb     the same with one Webb (2023) weight per cluster of cluster_trunk (few-cluster sensitivity).
(c) Tests of kappa = 1 and CES (alpha = beta): Wald with each bootstrap SE; Gaussian LR (chi2_1); Laplace quasi-LR;
    and restricted wild (FHH) bootstrap p-values (the null model generates the data), B = 999 (floor 0.001).
(d) kappa-free profile over sigma* in (0.05, 0.99), LR relative to the unconstrained optimum, Huber (Laplace QLR and
    Koenker-Bassett LR) and Gaussian, full design (n = 240) and on-path band (|ln N - ln N*(C)| <= 0.15, n = 41);
    whether the 95% sets hit the grid boundary; restricted wild-bootstrap calibration of the profile LR at selected
    null values (B = 299, floor 1/300).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import parametric as pm
import kprofile as pf
import ra1_common as rc

B_CH = 999
B_CAL = 299
if rc.QUICK:
    B_CH, B_CAL = 30, 20
SCHEMES = ["pairs", "cluster9", "cluster_trunk", "cluster_single", "wild_fhh", "wild_cl_webb"]
C_EVAL = (1e21, 5.76e23)
CH70 = (70e9, 1.4e12)
SIG_GRID = np.round(np.r_[np.arange(0.05, 0.59, 0.05), np.arange(0.60, 0.805, 0.01), np.arange(0.82, 0.985, 0.02),
                          0.99], 3)


def data240():
    df = rc.load_chinchilla_all()
    df = df[df["rank_worst"] > 5].reset_index(drop=True)
    return df


def design_facts():
    df = rc.load_chinchilla_all()
    d240 = df[df["rank_worst"] > 5]
    rows = []
    for lab, d in (("all digitized runs", df), ("baseline sample (5 highest-loss runs dropped)", d240)):
        rows.append(dict(sample=lab, runs=len(d), on_isoflop_profiles=int(d["iso"].sum()),
                         off_profile=int((~d["iso"]).sum()), n_budgets=int(d.loc[d["iso"], "budget"].nunique()),
                         off_profile_trunks=int(d.loc[~d["iso"], "trunk"].nunique()),
                         flop_min=float(d["C"].min()), flop_max=float(d["C"].max()),
                         iso_flop_min=float(d.loc[d["iso"], "C"].min()), iso_flop_max=float(d.loc[d["iso"], "C"].max()),
                         off_flop_min=float(d.loc[~d["iso"], "C"].min()), off_flop_max=float(d.loc[~d["iso"], "C"].max()),
                         N_min=float(d["N"].min()), N_max=float(d["N"].max()),
                         c_offset_dex=float(df.attrs.get("c_offset_dex", np.nan))))
    return pd.DataFrame(rows)


def cluster_ids(df, scheme):
    if scheme == "cluster9":
        return df["cl"].values
    if scheme in ("cluster_trunk", "wild_cl_webb"):
        return np.where(df["iso"], "b" + df["cl"].astype(str), "t" + df["trunk"].astype(str))
    if scheme == "cluster_single":
        return np.where(df["iso"], "b" + df["cl"].astype(str), "s" + df.index.astype(str))
    return np.arange(len(df)).astype(str)


def derived(th, p):
    m = rc.sl.Chinchilla.from_theta(th)
    d = dict(E=m.E, alpha=m.alpha, beta=m.beta, a=m.a_N, gamma=m.gamma, sigma_star=m.sigma_star,
             w70b=float(m.wedge(*CH70)), kappa=p[5], sigma_kappa=pm.sigma_star_kappa(p), E_kappa=float(np.exp(p[2])),
             alpha_minus_beta=m.alpha - m.beta)
    for C in C_EVAL:
        d[f"Mstar_{C:.3g}"] = float(m.D_opt(C) / m.N_opt(C))
    return d


def fit_pair(N, D, L, th_starts, p_starts):
    best = None
    for st in th_starts:
        m = rc.sl.fit_chinchilla(N, D, L, delta=1e-3, init=np.asarray(st, float))
        if best is None or m.extra["objective"] < best[1]:
            best = (m.theta, m.extra["objective"])
    th = best[0]
    import m2_est as me
    bq = None
    for st in list(p_starts) + [np.r_[th, 1.0]]:
        r = me.fit_q(N, D, L, "huber", init=st)
        if bq is None or r[1] < bq[1]:
            bq = r
    return th, best[1], bq[0], bq[1]


def _boot_one(i, N, D, L, scheme, groups, th0, p0, yhat, rfhh, clus, seed0, th_extra):
    rng = np.random.default_rng(seed0 + i)
    if scheme in ("pairs",):
        idx = rng.integers(0, len(L), len(L))
        Nb, Db, Lb = N[idx], D[idx], L[idx]
    elif scheme.startswith("cluster"):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        Nb, Db, Lb = N[idx], D[idx], L[idx]
    elif scheme == "wild_fhh":
        w = rc.fhh(rng, len(L))
        Nb, Db, Lb = N, D, np.exp(yhat + w * np.abs(rfhh))
    elif scheme == "wild_cl_webb":
        ug, inv = np.unique(clus, return_inverse=True)
        w = rc.webb(rng, len(ug))[inv]
        Nb, Db, Lb = N, D, np.exp(yhat + w * np.abs(rfhh))
    th, f1, p, f2 = fit_pair(Nb, Db, Lb, [th0] + th_extra, [p0])
    return derived(th, p)


def _restricted_one(i, N, D, L, null, th_r, p_r, yhat_r, rfhh_r, seed0, th_u, p_u):
    """Restricted wild (FHH) bootstrap draw under H0 ('kappa1' or 'ces'); returns the Laplace QLR and Gaussian LR."""
    import m2_est as me
    rng = np.random.default_rng(seed0 + i)
    w = rc.fhh(rng, len(L))
    Lb = np.exp(yhat_r + w * np.abs(rfhh_r))
    n = len(L)
    out = {}
    for kind, delta in (("huber", 1e-3), ("gauss", None)):
        # unrestricted
        if null == "kappa1":
            m = rc.sl.fit_chinchilla(N, D, Lb, delta=delta, init=th_r)
            r_obj = m.extra["objective"]
            u = min(me.fit_q(N, D, Lb, kind if kind == "huber" else "nls", init=p_u)[1],
                    me.fit_q(N, D, Lb, kind if kind == "huber" else "nls", init=np.r_[m.theta, 1.0])[1])
        else:  # ces: restricted alpha = beta
            mc = fit_ces(N, D, Lb, th_r, delta)
            r_obj = mc[1]
            u = min(rc.sl.fit_chinchilla(N, D, Lb, delta=delta, init=th_u).extra["objective"],
                    rc.sl.fit_chinchilla(N, D, Lb, delta=delta, init=mc[0]).extra["objective"])
        if kind == "huber":
            out["qlr_laplace"] = 2 * n * np.log(r_obj / u)
        else:
            out["lr_gauss"] = n * np.log(r_obj / u)
    return out


def fit_ces(N, D, L, init, delta=1e-3):
    from scipy.optimize import minimize
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    w = np.ones_like(lL)
    f = lambda q: rc.sl._obj(np.r_[q[:3], q[3], q[3]], lN, lD, lL, delta, w)
    best = None
    for st in ([init[0], init[1], init[2], 0.5 * (init[3] + init[4])], [6.0, 6.0, 0.5, 0.3], [6.0, 8.0, 0.6, 0.36]):
        r = minimize(f, np.asarray(st, float), method="L-BFGS-B", options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    q = best.x
    return np.r_[q[:3], q[3], q[3]], float(best.fun)


def _cal_one(i, kd_args, S0, full_r, yhat_r, rfhh_r, full_u, seed0):
    """Calibration draw for the profile LR at sigma0 (restricted wild FHH population)."""
    rng = np.random.default_rng(seed0 + i)
    N, D = kd_args
    w = rc.fhh(rng, len(yhat_r))
    Lb = np.exp(yhat_r + w * np.abs(rfhh_r))
    kd = pf.KappaData(N, D, Lb)
    fr, f_r, _ = pf.fit_S(kd, S0, [pf.start_at_S(full_r, S0, kd)], "huber")
    cands = [full_u, fr]
    fu = pf.fit_free(kd, cands, "huber")
    f_u = min(fu[1], f_r)
    return dict(qlr=2 * kd.n * np.log(f_r / f_u))


def run(log=rc.log):
    out = {}
    out["facts"] = design_facts()
    df = data240()
    N, D, L = df.N.values, df.D.values, df.L.values
    n = len(L)
    thB = rc.sl.BESIROGLU.theta
    thH = rc.sl.HOFFMANN.theta
    th, fth = pm.fit_chin(N, D, L, starts=[thB, thH], grid="fast")
    p, fp = pm.fit_kappa(N, D, L, th_chin=th)
    point = derived(th, p)
    log(f"  Chinchilla n=240: Huber sigma* {point['sigma_star']:.4f} kappa {point['kappa']:.4f} sigma*_k {point['sigma_kappa']:.4f}")
    # FHH residuals for the design-conditional wild bootstrap. The bootstrap population is the kappa-free fit (the
    # better-fitting surface; kappa = 1 is rejected): drawing from the kappa = 1 fit would centre kappa* at 1.
    from kappa_rob import kappa_jac
    yhat = pm.pred_kappa(p, N, D)
    r = np.log(L) - yhat
    rfhh, lev, f0 = pm.fhh_residuals(r, kappa_jac(p, N, D))
    r_chin = np.log(L) - pm.pred_chin(th, N, D)
    out["fhh_info"] = dict(f0=f0, max_lev=float(lev.max()), share_linear=float(np.mean(np.abs(r_chin) > 1e-3)),
                           mad_sd=float(1.4826 * np.median(np.abs(r_chin - np.median(r_chin)))),
                           sd=float(np.std(r_chin, ddof=5)), population="kappa-free fit")
    # Review fix (m-E): the wild population is the kappa-free fit, so the Chinchilla-form objects' bootstrap draws centre
    # on their pseudo-true values under that population, not on the point estimates. Record those values and report
    # basic intervals (logs for M* and w) for the wild schemes alongside the percentile intervals.
    thP, _, pP, _ = fit_pair(N, D, np.exp(yhat), [th, thB, thH], [p])
    pop = derived(thP, pP)
    rows = []
    draws = {}
    for sch in SCHEMES:
        clus = cluster_ids(df, sch)
        ug = np.unique(clus)
        groups = [np.where(clus == g)[0] for g in ug]
        payload = dict(N=N, D=D, L=L, scheme=sch, groups=groups, th0=th, p0=p, yhat=yhat, rfhh=rfhh, clus=clus,
                       seed0=rc.seed_of(f"chin|{sch}"), th_extra=[thB, thH])
        res = rc.pmap(_boot_one, list(range(B_CH)), payload, chunksize=16)
        ok = [x for x in res if "_error" not in x]
        Dd = pd.DataFrame(ok)
        draws[sch] = Dd
        row = dict(scheme=sch, n_clusters=len(ug), B=B_CH, n_fail=B_CH - len(ok))
        for k_, v in point.items():
            row[k_] = v
            row[f"se_{k_}"] = float(Dd[k_].std(ddof=1))
            row[f"rse_{k_}"] = rc.rse(Dd[k_])
            row[f"lo_{k_}"] = rc.pct(Dd[k_], 2.5)
            row[f"hi_{k_}"] = rc.pct(Dd[k_], 97.5)
        if sch.startswith("wild"):
            for k_, v in point.items():
                dv = Dd[k_].values.astype(float)
                dv = dv[np.isfinite(dv)]
                row[f"pop_{k_}"] = pop[k_]
                if k_.startswith("Mstar") or k_ == "w70b":
                    q = np.log(dv[dv > 0])
                    e_, p_ = np.log(v), np.log(pop[k_])
                    row[f"lo_basic_{k_}"] = float(np.exp(e_ - (np.percentile(q, 97.5) - p_)))
                    row[f"hi_basic_{k_}"] = float(np.exp(e_ - (np.percentile(q, 2.5) - p_)))
                else:
                    row[f"lo_basic_{k_}"] = float(v - (np.percentile(dv, 97.5) - pop[k_]))
                    row[f"hi_basic_{k_}"] = float(v - (np.percentile(dv, 2.5) - pop[k_]))
        z = (point["kappa"] - 1) / row["se_kappa"]
        row["p_kappa1_wald"] = float(2 * norm.sf(abs(z)))
        row["p_kappa1_boot"] = float(rc.boot_p_two_sided(Dd["kappa"], point["kappa"], 1.0))
        z2 = point["alpha_minus_beta"] / row["se_alpha_minus_beta"]
        row["p_ces_wald"] = float(2 * norm.sf(abs(z2)))
        row["p_ces_boot"] = float(rc.boot_p_two_sided(Dd["alpha_minus_beta"], point["alpha_minus_beta"], 0.0))
        rows.append(row)
        log(f"  Chinchilla bootstrap {sch:15s} ({len(ug)} clusters): se(beta) {row['se_beta']:.4f} "
            f"M*(5.76e23) [{row['lo_Mstar_5.76e+23']:.1f}, {row['hi_Mstar_5.76e+23']:.1f}] "
            f"w70B [{row['lo_w70b']:.2f}, {row['hi_w70b']:.2f}] p(kappa=1) {row['p_kappa1_wald']:.4f} fails {row['n_fail']}")
    out["schemes"] = pd.DataFrame(rows)
    out["draws"] = draws
    # ---------------- restricted tests (kappa = 1, CES)
    import m2_est as me
    tests = []
    # observed statistics
    mg = rc.sl.fit_chinchilla(N, D, L, delta=None, init=th)
    thg = mg.theta
    pg = min((me.fit_q(N, D, L, "nls", th_chin=thg), me.fit_q(N, D, L, "nls", init=p)), key=lambda t: t[1])
    LR_k_g = n * np.log(mg.extra["objective"] / pg[1])
    QLR_k = 2 * n * np.log(fth / fp)
    thc, fc = fit_ces(N, D, L, th)
    thcg, fcg = fit_ces(N, D, L, thg, None)
    LR_c_g = n * np.log(fcg / mg.extra["objective"])
    QLR_c = 2 * n * np.log(fc / fth)
    # Review fix (m-D): heteroskedasticity-robust (sandwich) Wald tests for the Gaussian (NLS) fits, a calibration
    # check of the Gaussian LR that does not rely on the restricted bootstrap population.
    from kappa_rob import kappa_jac

    def sandwich_wald(theta, J, r, g, val):
        A_ = np.linalg.inv(J.T @ J)
        Hd = np.sum((J @ A_) * J, axis=1)
        k_ = J.shape[1]
        out_ = {}
        for lab, wts in (("homosk", None), ("hc1", r ** 2 * n / (n - k_)), ("hc3", r ** 2 / (1 - Hd) ** 2)):
            V = A_ * float(r @ r) / (n - k_) if wts is None else A_ @ (J.T * wts) @ J @ A_
            out_[lab] = float(val ** 2 / (g @ V @ g))
        return out_
    rg = np.log(L) - pm.pred_chin(thg, N, D)
    W_c = sandwich_wald(thg, pm.jac_log_chin(thg, N, D), rg, np.array([0, 0, 0, 1.0, -1.0]), thg[3] - thg[4])
    rk = np.log(L) - pm.pred_kappa(pg[0], N, D)
    W_k = sandwich_wald(pg[0], kappa_jac(pg[0], N, D), rk, np.eye(6)[5], pg[0][5] - 1.0)
    # Koenker-Bassett LR (sparsity from the unrestricted residuals)
    kd = pf.KappaData(N, D, L)
    full_u = kd.from_raw(p)
    f0u = pf.f0_kernel(np.log(L) - pf.predict(full_u, kd.x, kd.z))
    S1 = lambda y_hat: float(np.sum(np.abs(np.log(L) - y_hat)))
    KB_k = 4 * f0u * (S1(pm.pred_chin(th, N, D)) - S1(pf.predict(full_u, kd.x, kd.z)))
    # CES (alpha = beta) is tested inside the kappa = 1 family: unrestricted = the kappa = 1 fit, not the kappa-free
    # bootstrap population above.
    KB_c = 4 * pf.f0_kernel(r_chin) * (S1(pm.pred_chin(thc, N, D)) - S1(pm.pred_chin(th, N, D)))
    for null, th_r, stat_q, stat_g, kb, Wg in (("kappa1", th, QLR_k, LR_k_g, KB_k, W_k), ("ces", thc, QLR_c, LR_c_g, KB_c, W_c)):
        yr = pm.pred_chin(th_r, N, D)
        rr = np.log(L) - yr
        rf, _, _ = pm.fhh_residuals(rr, pm.jac_log_chin(th_r, N, D))
        payload = dict(N=N, D=D, L=L, null=null, th_r=th_r, p_r=p, yhat_r=yr, rfhh_r=rf,
                       seed0=rc.seed_of(f"chin_restricted|{null}"), th_u=th, p_u=p)
        res = [x for x in rc.pmap(_restricted_one, list(range(B_CH)), payload, chunksize=16) if "_error" not in x]
        q_ = np.array([x["qlr_laplace"] for x in res])
        g_ = np.array([x["lr_gauss"] for x in res])
        tests.append(dict(null=null, B=len(res), stat_laplace_qlr=stat_q, p_chi2_laplace=float(chi2.sf(stat_q, 1)),
                          p_boot_laplace=float((1 + np.sum(q_ >= stat_q)) / (len(q_) + 1)),
                          crit95_boot_laplace=float(np.percentile(q_, 95)),
                          stat_gauss_lr=stat_g, p_chi2_gauss=float(chi2.sf(stat_g, 1)),
                          p_boot_gauss=float((1 + np.sum(g_ >= stat_g)) / (len(g_) + 1)),
                          crit95_boot_gauss=float(np.percentile(g_, 95)),
                          stat_kb=kb, p_chi2_kb=float(chi2.sf(kb, 1)),
                          p_floor=1.0 / (len(res) + 1),
                          **{f"wald_gauss_{k_}": v for k_, v in Wg.items()},
                          **{f"p_wald_gauss_{k_}": float(chi2.sf(v, 1)) for k_, v in Wg.items()}))
        log(f"  restricted wild bootstrap {null}: QLR {stat_q:.2f} (boot p {tests[-1]['p_boot_laplace']:.4f}); "
            f"Gaussian LR {stat_g:.2f} (chi2 p {tests[-1]['p_chi2_gauss']:.4f}, boot p {tests[-1]['p_boot_gauss']:.4f}); KB {kb:.2f}")
    out["tests"] = pd.DataFrame(tests)
    out["point"] = dict(theta=th, p=p, theta_gauss=thg, p_gauss=pg[0], ces=thc, ces_gauss=thcg)
    # ---------------- profile over sigma*
    C = 6 * N * D
    band = np.abs(np.log(N) - np.log(rc.sl.Chinchilla.from_theta(th).N_opt(C))) <= 0.15
    prof_rows, sets, cal_rows = [], [], []
    for samp, s in (("full design (n=240)", np.ones(n, bool)), ("on-path band (n=%d)" % band.sum(), band)):
        kd = pf.KappaData(N[s], D[s], L[s])
        ths, _ = pm.fit_chin(N[s], D[s], L[s])
        cands_raw = [me.fit_q(N[s], D[s], L[s], "huber", th_chin=ths)[0], p]
        for kind in ("huber", "gauss"):
            starts = [kd.from_raw(c) for c in cands_raw]
            if kind == "gauss":
                starts += [kd.from_raw(me.fit_q(N[s], D[s], L[s], "nls", th_chin=ths)[0])]
            fu = pf.fit_free(kd, starts, kind)
            pr = pf.profile_sigma(kd, SIG_GRID, fu[0], kind)
            # refine the unconstrained optimum from the best profile point
            best_i = int(np.argmin([x["obj"] for x in pr]))
            fu2 = pf.fit_free(kd, [fu[0], pr[best_i]["full"]], kind)
            fmin = min(fu2[1], min(x["obj"] for x in pr))
            full_u_s = fu2[0]
            su = 2 / (2 + full_u_s[3] + full_u_s[4])
            obj = np.array([x["obj"] for x in pr])
            if kind == "huber":
                LR = 2 * kd.n * np.log(obj / fmin)
                f0 = pf.f0_kernel(kd.y - pf.predict(full_u_s, kd.x, kd.z))
                s1u = pf.l1(full_u_s, kd)
                KB = np.array([4 * f0 * (pf.l1(x["full"], kd) - s1u) for x in pr])
                KB = np.maximum(KB, 0)
            else:
                LR = kd.n * np.log(obj / fmin)
                KB = np.full(len(pr), np.nan)
            for x, lr, kb in zip(pr, LR, KB):
                prof_rows.append(dict(sample=samp, objective=kind, sigma_star=x["sigma_star"], obj=x["obj"], LR=lr,
                                      LR_KB=kb, kappa=float(np.exp(x["full"][5])), E=float(np.exp(x["full"][2])),
                                      a_share=float(x["full"][4] / (x["full"][3] + x["full"][4])),
                                      kappa_at_bound=bool(np.exp(x["full"][5]) > 290 or np.exp(x["full"][5]) < 1.1e-3)))
            for lab, stat in (("Laplace QLR" if kind == "huber" else "Gaussian LR", LR), ("Koenker-Bassett LR", KB)):
                if np.all(np.isnan(stat)):
                    continue
                cs = pf.conf_set(SIG_GRID, stat)
                sets.append(dict(sample=samp, objective=kind, statistic=lab, sigma_hat_unconstrained=su,
                                 kappa_hat=float(np.exp(full_u_s[5])), E_hat=float(np.exp(full_u_s[2])),
                                 grid_lo=SIG_GRID.min(), grid_hi=SIG_GRID.max(), max_LR=float(np.nanmax(stat)),
                                 argmin_grid=float(SIG_GRID[int(np.nanargmin(stat))]), **{f"set_{k_}": v for k_, v in cs.items()}))
            log(f"  profile {samp} {kind}: sigma_hat {su:.3f}; max LR {np.nanmax(LR):.1f}; set {pf.conf_set(SIG_GRID, LR)}")
            # bootstrap calibration of the Laplace QLR at selected nulls
            if kind == "huber":
                cs = pf.conf_set(SIG_GRID, LR)
                if samp.startswith("full"):
                    s0s = [x for x in (cs["lo"], cs["hi"], 0.74) if np.isfinite(x)]
                else:
                    s0s = [0.30, 0.74]
                for s0 in s0s:
                    j = int(np.argmin(np.abs(SIG_GRID - s0)))
                    S0 = 2 / SIG_GRID[j] - 2
                    full_r = pr[j]["full"]
                    yr = pf.predict(full_r, kd.x, kd.z)
                    rr = kd.y - yr
                    # FHH adjustment with the numerical Jacobian of the restricted model (5 free parameters)
                    eps = 1e-6
                    qr = pr[j]["q"]
                    Jc = []
                    for k_ in range(len(qr)):
                        dq = np.zeros(len(qr))
                        dq[k_] = eps
                        fp_ = np.array([qr[0] + dq[0], qr[1] + dq[1], qr[2] + dq[2], S0 * (qr[3] + dq[3]),
                                        S0 * (1 - qr[3] - dq[3]), qr[4] + dq[4]])
                        Jc.append((pf.predict(fp_, kd.x, kd.z) - yr) / eps)
                    rf, _, _ = pm.fhh_residuals(rr, np.column_stack(Jc))
                    payload = dict(kd_args=(N[s], D[s]), S0=S0, full_r=full_r, yhat_r=yr, rfhh_r=rf, full_u=full_u_s,
                                   seed0=rc.seed_of(f"cal|{samp}|{s0}"))
                    res = [x["qlr"] for x in rc.pmap(_cal_one, list(range(B_CAL)), payload, chunksize=8) if "_error" not in x]
                    res = np.array(res)
                    cal_rows.append(dict(sample=samp, sigma0=float(SIG_GRID[j]), LR_obs=float(LR[j]), B=len(res),
                                         crit95_boot=float(np.percentile(res, 95)), crit95_chi2=3.8415,
                                         p_boot=float((1 + np.sum(res >= LR[j])) / (len(res) + 1)),
                                         p_chi2=float(chi2.sf(LR[j], 1)), p_floor=1 / (len(res) + 1)))
                    log(f"    calibration {samp} sigma0={SIG_GRID[j]}: LR {LR[j]:.2f}, boot crit {cal_rows[-1]['crit95_boot']:.2f}, "
                        f"p_boot {cal_rows[-1]['p_boot']:.3f}")
    out["profile"] = pd.DataFrame(prof_rows)
    out["sets"] = pd.DataFrame(sets)
    out["calibration"] = pd.DataFrame(cal_rows)
    out["band_n"] = int(band.sum())
    return out
