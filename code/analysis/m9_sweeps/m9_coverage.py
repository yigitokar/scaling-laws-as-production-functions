"""m9_coverage.py -- size and coverage of the plan's inference on this design (design-only Monte Carlo, run after
the power calculation and before any estimation on the sweep results).

The power calculation (m9_power) showed that the wild cluster bootstrap by width (Webb weights, residuals of the
unrestricted fit; 'WCU') understates the sampling s.d. of the tilt by about one third and its nominal 95% interval
covers 73-78%. This stage measures, with the same DGP (Chinchilla truth, s = 0.005, rho = 0.5):
  * coverage of the plan's basic intervals for sigma*_kappa, model-free sigma* (P and P6) and the Q3 slope
    (FineWeb-Edu design, convention P), relative to the noise-free value of each estimator on the design;
  * size and power of the tilt test by WCU (plan) and by the wild cluster RESTRICTED bootstrap ('WCR': residuals of
    the Hicks-neutral fit, chi = 0 imposed; Cameron, Gelbach and Miller 2008; MacKinnon and Webb 2017);
  * the ratio of the Monte Carlo s.d. to the mean bootstrap s.e. (a calibration factor).
Each replication runs single-threaded inside one of at most 4 worker processes.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import m9_common as mc
import m9_est as est
import m9_power as pw
import m9_stages as st

B_COV = 199


def cov_rep(item, design=None, tp=None, s=0.005, rho=0.5, chi=0.22, B=B_COV, full=True):
    """One replication: every statistic with the plan's wild cluster bootstrap on the raw residuals ('wcu') and on
    CR2 leverage-adjusted cluster residuals ('cr2'), same Webb draws; for the tilt also the restricted versions."""
    rep, seed = item
    rng = np.random.default_rng(seed)
    data = {}
    for r in ("web", "edu"):
        df = design[r]
        lt = pw.true_lnL(tp, df.N_P.values, df.D.values, chi if r == "edu" else 0.0)
        data[r] = np.exp(lt + pw.noise(df, s, rho, rng))
    out = dict(rep=rep)
    de = design["edu"]
    V = ("wcu", "cr2")
    if full:
        # ---- Q1, FineWeb-Edu, convention P
        m = de.is_main.values
        N, D, L = de.N_P.values[m], de.D.values[m], data["edu"][m]
        clus = de.arch.values[m]
        uni = sorted(set(clus))
        th, _ = est.fit_chin(N, D, L, "huber")
        p, _ = est.fit_kappa(N, D, L, "huber", th_chin=th)
        prep, _ = st.mf_prepare(N, D, L)
        f = np.log(L)
        pilot = est.pilot_fit(prep["x"], prep["z"], f, prep["hx"], prep["hz"])
        ms = st.mf_stats(prep, f, "P")
        lok = {pc: np.isfinite(ms[f"_path_{pc}"][:, 1]) for pc in ("P", "P6")}
        ms0 = st.mf_stats(prep, pilot, "P", lok)
        yk = est.pred_kappa(p, N, D)
        ek = {"wcu": f - yk}
        ek["cr2"] = est.cr2_residuals(ek["wcu"], est.hat_from_jac(est.jac_kappa(p, N, D)), clus)
        ep = {"wcu": f - pilot}
        ep["cr2"] = est.cr2_residuals(ep["wcu"], est.smoother_matrix(prep["x"], prep["z"], prep["hx"], prep["hz"]), clus)
        dr = {(k, v): [] for k in ("sig_kappa", "sig_mf_P", "sig_mf_P6") for v in V}
        for b in range(B):
            sh = st.Shock(seed + 1000 + b, "wild", uni)
            for v in V:
                pb, _ = est.fit_kappa(N, D, np.exp(yk + sh.errors(ek[v], clus, "edu")), "huber", init=p)
                dr[("sig_kappa", v)].append(est.sigma_star_kappa(pb))
                mb = st.mf_stats(prep, pilot + sh.errors(ep[v], clus, "edu"), "P", lok)
                dr[("sig_mf_P", v)].append(mb["sig_mf_P"])
                dr[("sig_mf_P6", v)].append(mb["sig_mf_P6"])
        pt = dict(sig_kappa=est.sigma_star_kappa(p), sig_mf_P=ms["sig_mf_P"], sig_mf_P6=ms["sig_mf_P6"])
        pop = dict(sig_kappa=pt["sig_kappa"], sig_mf_P=ms0["sig_mf_P"], sig_mf_P6=ms0["sig_mf_P6"])
        for (k, v), d_ in dr.items():
            lo, hi = mc.basic_ci(pt[k], d_, pop[k])
            out.update({f"{k}_est": pt[k], f"{k}_{v}_lo": lo, f"{k}_{v}_hi": hi, f"{k}_{v}_se": mc.sd_(d_)})
        # ---- Q3, FineWeb-Edu, convention P (main + hiM)
        N, D, M = de.N_P.values, de.D.values, de.M_P.values
        f = np.log(data["edu"])
        restr = de.is_main.values & (M <= 100)
        clus = de.arch.values
        uni = sorted(set(clus))
        prep3, _ = st.mf_prepare(N, D, np.exp(f))
        core = st.q3_core(N, D, M, f, restr, prep3)
        pop3 = st.q3_core(N, D, M, core["pilot"], restr, prep3, fits_warm=core["fits"])
        e3 = {"wcu": f - core["pilot"]}
        e3["cr2"] = est.cr2_residuals(e3["wcu"], est.smoother_matrix(prep3["x"], prep3["z"], prep3["hx"], prep3["hz"]), clus)
        k3 = "slope|chin_r|local"
        d3 = {v: [] for v in V}
        for b in range(B):
            sh = st.Shock(seed + 5000 + b, "wild", uni)
            for v in V:
                rr = st.q3_core(N, D, M, core["pilot"] + sh.errors(e3[v], clus, "edu"), restr, prep3, fits_warm=core["fits"])
                d3[v].append(rr["stats"].get(k3, np.nan))
        out["q3_est"] = core["stats"][k3]
        for v in V:
            lo, hi = mc.basic_ci(core["stats"][k3], d3[v], pop3["stats"][k3])
            out.update({f"q3_{v}_lo": lo, f"q3_{v}_hi": hi, f"q3_{v}_se": mc.sd_(d3[v])})
    # ---- tilt (validation set 1), convention P: unrestricted (plan) and restricted (null imposed), raw and CR2
    dw = design["web"]
    me_, mw_ = de.is_main.values, dw.is_main.values
    N = np.r_[dw.N_P.values[mw_], de.N_P.values[me_]]
    D = np.r_[dw.D.values[mw_], de.D.values[me_]]
    L = np.r_[data["web"][mw_], data["edu"][me_]]
    g = np.r_[np.zeros(mw_.sum(), int), np.ones(me_.sum(), int)]
    clus = np.r_[dw.arch.values[mw_], de.arch.values[me_]]
    uni = sorted(set(clus))
    starts, _ = est.ce_starts(N, D, L, g)
    th, _ = est.fit_ce(N, D, L, g, "huber", starts=starts)
    chi_hat = est.ce_tilt(th)
    yh = est.pred_ce(th, N, D, g)
    th0, yh0 = est.fit_ce_null(N, D, L, g, th)
    y = np.log(L)
    import m2_est as me
    H = est.hat_from_jac(est.jac_ce(th, N, D, g))
    H0 = est.hat_from_jac(est.num_jac(lambda q: me.panel_pred("hicksE", q, np.log(N), np.log(D), g, 2), th0))
    E = {"wcu": y - yh, "cr2": est.cr2_residuals(y - yh, H, clus)}
    E0 = {"wcu": y - yh0, "cr2": est.cr2_residuals(y - yh0, H0, clus)}
    du = {v: [] for v in V}
    dr_ = {v: [] for v in V}
    for b in range(B):
        sh = st.Shock(seed + 9000 + b, "wild", uni)
        w = np.array([sh.w[c] for c in clus])
        for v in V:
            tb, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), yh + w * E[v], g, 2, "huber", starts=[th], tight=False)
            du[v].append(est.ce_tilt(tb))
            tr, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), yh0 + w * E0[v], g, 2, "huber", starts=[th], tight=False)
            dr_[v].append(est.ce_tilt(tr))
    out["chi_est"] = chi_hat
    for v in V:
        lo, hi = mc.basic_ci(chi_hat, du[v], chi_hat)
        drv = np.asarray(dr_[v])
        out.update({f"chi_{v}_lo": lo, f"chi_{v}_hi": hi, f"chi_{v}_se": mc.sd_(du[v]),
                    f"chi_{v}_p_wcr": float((1 + np.sum(np.abs(drv) >= abs(chi_hat))) / (len(drv) + 1)),
                    f"chi_{v}_se_wcr": mc.sd_(drv)})
    return out


def run(R=40, B=B_COV, log=mc.log):
    design = pw.mc_design()
    tp = pw.truth_params("chin", design)
    if mc.QUICK:
        R, B = 4, 29
    # noise-free values of each estimator on the design (targets for coverage)
    nf = cov_rep((0, 1), design=design, tp=tp, s=1e-6, rho=0.5, chi=0.22, B=21)
    rows = []
    for chi, full in ((0.22, True), (0.0, False)):
        t0 = time.time()
        items = [(i, mc.seed_of(f"coverage|{chi}|{i}")) for i in range(R)]
        res = mc.pmap(cov_rep, items, dict(design=design, tp=tp, s=0.005, rho=0.5, chi=chi, B=B, full=full))
        bad = [r for r in res if "_error" in r]
        if bad:
            log(f"  coverage chi={chi}: {len(bad)} failed; first: {bad[0]['_error']}")
        d = pd.DataFrame([r for r in res if "_error" not in r])
        d.to_csv(os.path.join(mc.PROC, f"coverage_reps_chi{chi}.csv"), index=False)
        stats = [("sig_kappa", 0.737, nf["sig_kappa_est"]), ("sig_mf_P", np.nan, nf["sig_mf_P_est"]),
                 ("sig_mf_P6", 0.737, nf["sig_mf_P6_est"]), ("q3", np.nan, nf["q3_est"]),
                 ("chi", chi, nf["chi_est"] if chi else 0.0)]
        for k, truth, nfv in stats:
            if f"{k}_est" not in d:
                continue
            for v in ("wcu", "cr2"):
                est_, lo, hi, se_b = d[f"{k}_est"], d[f"{k}_{v}_lo"], d[f"{k}_{v}_hi"], d[f"{k}_{v}_se"]
                row = dict(statistic=k, scheme=v, chi_true=chi, n_rep=len(d), noise_free_value=nfv, truth=truth,
                           mc_sd=float(est_.std(ddof=1)), mean_boot_se=float(np.nanmean(se_b)),
                           calibration_factor=float(est_.std(ddof=1) / np.nanmean(se_b)),
                           cover_noise_free=float(np.mean((lo <= nfv) & (hi >= nfv))),
                           cover_truth=float(np.mean((lo <= truth) & (hi >= truth))) if np.isfinite(truth) else np.nan,
                           mean_halfwidth=float(np.nanmean((hi - lo) / 2)))
                if k == "chi":
                    row.update(reject_ci=float(np.mean((lo > 0) | (hi < 0))),
                               reject_wcr_p=float(np.mean(d[f"chi_{v}_p_wcr"] < 0.05)))
                rows.append(row)
        log(f"  coverage chi={chi}: {len(d)} reps in {time.time() - t0:.0f}s")
    out = pd.DataFrame(rows)
    mc.write_csv(out, "m9_sweeps_power_coverage")
    return out


if __name__ == "__main__":
    import sys
    run(R=int(sys.argv[1]) if len(sys.argv) > 1 else 40)


def cov_rep_mf(item, design=None, tp=None, s=0.005, rho=0.5, chi=0.22, B=99):
    """Model-free sigma* (P, P6) and Q3 slope with CR2 residuals AND the LOO-CV bandwidth re-selected in every draw
    ('cr2cv'); same DGP and seeds as cov_rep (so the point estimates coincide)."""
    rep, seed = item
    rng = np.random.default_rng(seed)
    data = {}
    for r in ("web", "edu"):
        df = design[r]
        lt = pw.true_lnL(tp, df.N_P.values, df.D.values, chi if r == "edu" else 0.0)
        data[r] = np.exp(lt + pw.noise(df, s, rho, rng))
    out = dict(rep=rep)
    de = design["edu"]
    m = de.is_main.values
    N, D, L = de.N_P.values[m], de.D.values[m], data["edu"][m]
    clus = de.arch.values[m]
    uni = sorted(set(clus))
    prep, _ = st.mf_prepare(N, D, L)
    f = np.log(L)
    pilot = est.pilot_fit(prep["x"], prep["z"], f, prep["hx"], prep["hz"])
    ms = st.mf_stats(prep, f, "P")
    lok = {pc: np.isfinite(ms[f"_path_{pc}"][:, 1]) for pc in ("P", "P6")}
    ms0 = st.mf_stats(prep, pilot, "P", lok)
    ep = est.cr2_residuals(f - pilot, est.smoother_matrix(prep["x"], prep["z"], prep["hx"], prep["hz"]), clus)
    dr = {"sig_mf_P": [], "sig_mf_P6": []}
    for b in range(B):
        sh = st.Shock(seed + 1000 + b, "wild", uni)
        fb = pilot + sh.errors(ep, clus, "edu")
        pb, _ = st.mf_prepare(N, D, np.exp(fb))
        mb = st.mf_stats(pb, fb, "P", lok)
        dr["sig_mf_P"].append(mb["sig_mf_P"])
        dr["sig_mf_P6"].append(mb["sig_mf_P6"])
    for k in dr:
        lo, hi = mc.basic_ci(ms[k], dr[k], ms0[k])
        out.update({f"{k}_est": ms[k], f"{k}_cr2cv_lo": lo, f"{k}_cr2cv_hi": hi, f"{k}_cr2cv_se": mc.sd_(dr[k])})
    N, D, M = de.N_P.values, de.D.values, de.M_P.values
    f = np.log(data["edu"])
    restr = de.is_main.values & (M <= 100)
    clus = de.arch.values
    uni = sorted(set(clus))
    prep3, _ = st.mf_prepare(N, D, np.exp(f))
    core = st.q3_core(N, D, M, f, restr, prep3)
    pop3 = st.q3_core(N, D, M, core["pilot"], restr, prep3, fits_warm=core["fits"])
    e3 = est.cr2_residuals(f - core["pilot"], est.smoother_matrix(prep3["x"], prep3["z"], prep3["hx"], prep3["hz"]), clus)
    k3 = "slope|chin_r|local"
    d3 = []
    for b in range(B):
        sh = st.Shock(seed + 5000 + b, "wild", uni)
        fb = core["pilot"] + sh.errors(e3, clus, "edu")
        pb, _ = st.mf_prepare(N, D, np.exp(fb))
        rr = st.q3_core(N, D, M, fb, restr, pb, fits_warm=core["fits"])
        d3.append(rr["stats"].get(k3, np.nan))
    lo, hi = mc.basic_ci(core["stats"][k3], d3, pop3["stats"][k3])
    out.update(q3_est=core["stats"][k3], q3_cr2cv_lo=lo, q3_cr2cv_hi=hi, q3_cr2cv_se=mc.sd_(d3))
    return out


def run_mf(R=40, B=99, log=mc.log):
    design = pw.mc_design()
    tp = pw.truth_params("chin", design)
    if mc.QUICK:
        R, B = 4, 21
    nf = pd.read_csv(os.path.join(mc.TABLES, "m9_sweeps_power_coverage.csv"))
    items = [(i, mc.seed_of(f"coverage|0.22|{i}")) for i in range(R)]
    t0 = time.time()
    res = mc.pmap(cov_rep_mf, items, dict(design=design, tp=tp, s=0.005, rho=0.5, chi=0.22, B=B))
    bad = [r for r in res if "_error" in r]
    if bad:
        log(f"  coverage (cr2cv): {len(bad)} failed; first: {bad[0]['_error']}")
    d = pd.DataFrame([r for r in res if "_error" not in r])
    d.to_csv(os.path.join(mc.PROC, "coverage_reps_cr2cv.csv"), index=False)
    rows = []
    for k, truth in (("sig_mf_P", np.nan), ("sig_mf_P6", 0.737), ("q3", np.nan)):
        nfv = float(nf[(nf.statistic == k) & (nf.chi_true == 0.22)].noise_free_value.values[0])
        est_, lo, hi, se_b = d[f"{k}_est"], d[f"{k}_cr2cv_lo"], d[f"{k}_cr2cv_hi"], d[f"{k}_cr2cv_se"]
        rows.append(dict(statistic=k, scheme="cr2cv", chi_true=0.22, n_rep=len(d), noise_free_value=nfv, truth=truth,
                         mc_sd=float(est_.std(ddof=1)), mean_boot_se=float(np.nanmean(se_b)),
                         calibration_factor=float(est_.std(ddof=1) / np.nanmean(se_b)),
                         cover_noise_free=float(np.mean((lo <= nfv) & (hi >= nfv))),
                         cover_truth=float(np.mean((lo <= truth) & (hi >= truth))) if np.isfinite(truth) else np.nan,
                         mean_halfwidth=float(np.nanmean((hi - lo) / 2))))
    out = pd.concat([nf[nf.scheme != "cr2cv"], pd.DataFrame(rows)], ignore_index=True)
    mc.write_csv(out, "m9_sweeps_power_coverage")
    log(f"  coverage (cr2cv): {len(d)} reps in {time.time() - t0:.0f}s")
    return out
