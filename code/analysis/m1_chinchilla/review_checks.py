"""review_checks.py -- additions from the independent review (2026-09-23), stage `checks` of run.py.

Cheap post-hoc checks computed from the module's saved outputs (bootstrap draws and CSVs), plus one small
size-matched bootstrap control. Run after the horse, duality, fdep, labs and syslr stages and before `report`.

1. estimator_differences   Paired-bootstrap tests of estimator differences. All estimators in the horse race use the
                           same index draws, so the SD of (theta_e - theta_huber) across draws is the correct s.e. of the
                           difference (comparing the difference with the marginal s.e.s ignores their 0.9+ correlation).
2. duality_bootcal         Bootstrap-calibrated p-values for the duality Wald tests. The chi2 p-values in
                           m1_chinchilla_duality_tests.csv depend strongly on the covariance estimator (robust vs SD)
                           because parabola argmins in resampled designs are heavy-tailed. Here the critical value comes
                           from the bootstrap distribution of the same quadratic form, W*_b = (d*_b - d_hat)' V^+
                           (d*_b - d_hat), so the p-value no longer relies on chi2 tails; both V choices are reported.
                           Slope-only (1 d.f.) tests are added, and for the "A3 (this fit)" nulls a slope t-statistic
                           that adds the primal's own sampling variance to the classical OLS s.e. of the A2 slope
                           (the OLS t in duality_tests treats the primal's slope as known).
3. labs_design_decomp      Decomposes the analytic-vs-design gap in the IsoFLOP tests into (i) finite-grid parabola
                           bias (design evaluated with D = C_b/(6N), i.e. exact 6ND = C_b) and (ii) the FLOP-accounting
                           mismatch of the actual design (for Marin, 6ND differs from the nominal budget by -7% to +35%,
                           strongly correlated with N within each budget).
4. fdep_size_control       The on-path band has n = 41 vs n = 240 for the full design, so part of the "SE explosion" is
                           sample size. Control: random subsamples of the full design with the same n (R = 4, pairs
                           bootstrap B = 100, same 75 starting values); report band SE / control SE.
5. system_lr_smallsample   The nested system LR has only n2 = 7-22 argmin observations; an F-form small-sample
                           calibration, F = (exp(LR/n2) - 1)(n2 - 2)/2 ~ F(2, n2 - 2), treats the whole LR as coming from
                           the argmin equation (an approximation, reported as a sensitivity check only).
"""
from __future__ import annotations

import itertools
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import f as f_dist

import boot
import duality
import estimators as est
import fdep
from common import (BESI_PUB, DERIVED_KEYS, HOFF_TEX, PROC, SEED, TABLES, load_chinchilla, load_isoflop, sl,
                    theta_of)

R = lambda f: pd.read_csv(os.path.join(TABLES, f))
BOOT_COLS = ["lnA", "lnB", "lnE", "alpha", "beta"] + DERIVED_KEYS   # column order of boot_chinchilla_*.npy


# ----------------------------------------------------------------------------- 1. estimator differences
def estimator_differences(log=print):
    hr = R("m1_chinchilla_horse_race.csv").set_index(["n", "estimator"])
    rows = []
    for n in (240, 245):
        for sch in ("pairs", "cluster"):
            H = np.load(os.path.join(PROC, f"boot_chinchilla_n{n}_huber_{sch}.npy"))
            for e in ("lad_log", "gauss_log", "nls_lev"):
                G = np.load(os.path.join(PROC, f"boot_chinchilla_n{n}_{e}_{sch}.npy"))
                m_ = min(len(G), len(H))
                for k in ("alpha", "beta", "a", "gamma", "sigma_star", "Mstar_1e+21", "Mstar_5.76e+23", "w_chin70b"):
                    j = BOOT_COLS.index(k)
                    d = G[:m_, j] - H[:m_, j]
                    pt = float(hr.loc[(n, e), k] - hr.loc[(n, "huber"), k])
                    sd = float(np.std(d, ddof=1))
                    rows.append(dict(n=n, scheme=sch, estimator=e, vs="huber", param=k, diff=pt, sd_diff=sd,
                                     z=pt / sd, p=float(2 * norm.sf(abs(pt / sd))),
                                     lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                                     corr=float(np.corrcoef(G[:m_, j], H[:m_, j])[0, 1]), B=m_))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_estimator_diffs.csv"), index=False)
    s = out[(out.n == 240) & (out.param.isin(["beta", "sigma_star", "Mstar_5.76e+23"]))]
    log("  paired estimator differences (n=240):\n" + s[["scheme", "estimator", "param", "diff", "sd_diff", "z", "p"]]
        .round(4).to_string())
    return out


# ----------------------------------------------------------------------------- 2. bootstrap-calibrated duality tests
SUBSETS = {"slope": [0], "path": [0, 1], "frontier": [2, 3], "all": [0, 1, 2, 3]}


def _bootcal(dhat, Dd, cov):
    """chi2 and bootstrap-calibrated p-values for every subset; Dd = bootstrap analogues of dhat."""
    Dd = Dd[np.all(np.isfinite(Dd), axis=1)]
    V = duality.robust_cov(Dd) if cov == "robust" else np.cov(Dd.T)
    out = {}
    for nm, ix in SUBSETS.items():
        Vs = np.atleast_2d(V[np.ix_(ix, ix)])
        P = np.linalg.pinv(Vs)
        d = dhat[ix]
        W = float(d @ P @ d)
        C = Dd[:, ix] - d                              # recentred at the full-sample estimate
        Wb = np.einsum("bi,ij,bj->b", C, P, C)
        out[f"W_{nm}"] = W
        out[f"p_{nm}_chi2"] = float(chi2.sf(W, len(ix)))
        out[f"p_{nm}_boot"] = float((1 + np.sum(Wb >= W)) / (len(Wb) + 1))
    out["B"] = len(Dd)
    return out


def _nulls_for(tag_file, df, lnC_ref, lit):
    """Rebuild (null label, dhat, bootstrap analogues) exactly as duality.duality_for does."""
    z = np.load(os.path.join(PROC, f"duality_draws_{tag_file}.npz"))
    pt2 = z["pt2"]
    N, D = df.N.values, df.D.values
    iso = df.iso.values.astype(bool)
    lab, lnC = df.budget.values, np.log(df.C.values)
    res = {"stratified pairs": [("A3 (this fit), analytic", pt2 - z["an"], z["A2b"] - z["A3an"]),
                                ("A3 (this fit), design-consistent", pt2 - z["de"], z["A2b"] - z["A3de"])],
           "wild, fixed design": [("A3 (this fit), analytic", pt2 - z["an"], z["WA2"] - z["Wan"]),
                                  ("A3 (this fit), design-consistent", pt2 - z["de"], z["WA2"] - z["Wde"])]}
    for tag, th in lit.items():
        ml = sl.Chinchilla.from_theta(th)
        pa = duality.vec(duality.analytic_objects(ml, lnC_ref))
        pd_ = duality.vec(duality.design_objects(ml, N[iso], D[iso], lab[iso], lnC[iso], lnC_ref))
        res["stratified pairs"] += [(f"{tag} (fixed), analytic", pt2 - pa, z["A2b"] - pa),
                                    (f"{tag} (fixed), design", pt2 - pd_, z["A2b"] - z[f"{tag}de"])]
        res["wild, fixed design"] += [(f"{tag} (fixed), analytic", pt2 - pa, z["WA2"] - pa),
                                      (f"{tag} (fixed), design", pt2 - pd_, z["WA2"] - pd_)]
    return res, z


def duality_bootcal(log=print):
    import labs
    lit = {"Hoffmann": theta_of(HOFF_TEX), "Besiroglu": theta_of(sl.BESIROGLU)}
    datasets = []
    for k in (5, 0):
        df = load_chinchilla(k)
        datasets.append((f"Chinchilla n={len(df)}", df, float(np.mean(np.log(df.loc[df.iso, "C"]))), lit))
    for exp, lab in labs.EXPERIMENTS.items():
        df = load_isoflop(exp)
        datasets.append((lab, df, float(np.mean(np.log(df.C.values))), {}))
    # classical OLS s.e. of the A2 slope, from the main test table
    te = pd.concat([R("m1_chinchilla_duality_tests.csv"), R("m1_chinchilla_labs_duality_tests.csv")], ignore_index=True)
    rows = []
    for label, df, lnC_ref, lt in datasets:
        tag_file = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        res, z = _nulls_for(tag_file, df, lnC_ref, lt)
        se_ols = float(te[te.dataset == label].se_a_ols.iloc[0])
        for bs, lst in res.items():
            for nm, dhat, Dd in lst:
                for cov in ("robust", "sd"):
                    r = dict(dataset=label, null=nm, bootstrap=bs, cov=cov, d_a=dhat[0], d_lnNstar_ref=dhat[1],
                             d_gamma=dhat[2], d_Lstar_ref=dhat[3])
                    r.update(_bootcal(dhat, Dd, cov))
                    if nm.startswith("A3 (this fit)") and bs == "stratified pairs":
                        # slope t with the primal's sampling variance added (independence; conservative if the
                        # A2 and A3 slopes are positively correlated across draws)
                        key = "A3de" if "design" in nm else "A3an"
                        sd3 = float(np.std(z[key][:, 0], ddof=1))
                        rse3 = float(boot.rse(z[key][:, 0]))
                        r.update(se_a_ols=se_ols, sd_a3_strat=sd3, rse_a3_strat=rse3,
                                 t_a_ols_plus_a3=float(dhat[0] / np.hypot(se_ols, rse3)),
                                 p_a_ols_plus_a3=float(2 * norm.sf(abs(dhat[0] / np.hypot(se_ols, rse3)))))
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_duality_bootcal.csv"), index=False)
    classical_path_F(datasets, log=log)
    show = out[out["null"].str.contains("design")]
    log("  bootstrap-calibrated duality tests (design nulls):\n" +
        show[["dataset", "null", "bootstrap", "cov", "W_path", "p_path_chi2", "p_path_boot", "p_slope_boot",
              "p_frontier_boot"]].round(4).to_string())
    return out


def classical_path_F(datasets, log=print):
    """Bootstrap-free benchmark: per-budget argmin differences e_b = lnN*_b(data) - lnN*_b(null, design-consistent),
    regressed on [1, x_b - xbar] (x = ln C_b); H0: both coefficients are zero, F(2, n_b - 2) under iid normal argmin
    errors. The null technology is treated as known (exact for the published parameter sets; for the module's own
    refit it ignores the primal's sampling error, so it over-rejects there)."""
    rows = []
    for label, df, lnC_ref, lt in datasets:
        N, D, L = df.N.values, df.D.values, df.L.values
        iso = df.iso.values.astype(bool)
        lab, lnC = df.budget.values, np.log(df.C.values)
        obs, used = duality.a2_minima(np.log(N[iso]), L[iso], lab[iso], lnC[iso], return_labels=True)
        techs = dict(lt)
        if label.startswith("Chinchilla"):
            m3 = est.fit_huber(N, D, L, inits=[theta_of(BESI_PUB), theta_of(HOFF_TEX)])
        else:
            import labs
            exp = [e for e, lb in labs.EXPERIMENTS.items() if lb == label][0]
            r3 = R("m1_chinchilla_labs_technology.csv")
            r3 = r3[(r3.experiment == exp) & r3.approach.str.startswith("A3")].iloc[0]
            m3 = sl.Chinchilla(E=r3.E, A=r3.A, B=r3.B, alpha=r3.alpha, beta=r3.beta)
        techs = {"A3 (this fit)": theta_of(m3), **techs}
        keep = iso & np.isin(lab, used)
        for tag, th in techs.items():
            ml = sl.Chinchilla.from_theta(th)
            nul, used0 = duality.a2_minima(np.log(N[keep]), ml.loss(N[keep], D[keep]), lab[keep], lnC[keep],
                                           return_labels=True)
            if len(nul) != len(obs):
                continue
            e = obs[:, 1] - nul[:, 1]
            x = obs[:, 0] - obs[:, 0].mean()
            X = np.column_stack([np.ones_like(x), x])
            b, *_ = np.linalg.lstsq(X, e, rcond=None)
            res = e - X @ b
            nb = len(e)
            s2 = float(res @ res / (nb - 2))
            V = s2 * np.linalg.inv(X.T @ X)
            F = float(b @ np.linalg.solve(V, b) / 2)
            rows.append(dict(dataset=label, null=f"{tag}, design-consistent argmins", n_budgets=nb,
                             level_gap=float(b[0]), se_level=float(np.sqrt(V[0, 0])), slope_gap=float(b[1]),
                             se_slope=float(np.sqrt(V[1, 1])), t_level=float(b[0] / np.sqrt(V[0, 0])),
                             t_slope=float(b[1] / np.sqrt(V[1, 1])), F=F, df2=nb - 2,
                             p_F=float(f_dist.sf(F, 2, nb - 2))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_duality_classicalF.csv"), index=False)
    log("  classical fixed-design F tests of the path restrictions:\n" + out.round(4).to_string())
    return out


# ----------------------------------------------------------------------------- 3. design decomposition for labs
def labs_design_decomp(log=print):
    import labs
    lt = R("m1_chinchilla_labs_technology.csv")
    rows = []
    for exp, lab in labs.EXPERIMENTS.items():
        df = load_isoflop(exp)
        r3 = lt[(lt.experiment == exp) & lt.approach.str.startswith("A3")].iloc[0]
        m3 = sl.Chinchilla(E=r3.E, A=r3.A, B=r3.B, alpha=r3.alpha, beta=r3.beta)
        N, D, C = df.N.values, df.D.values, df.C.values
        lab_, lnC = df.budget.values, np.log(C)
        lnC_ref = float(np.mean(lnC))
        an = duality.vec(duality.analytic_objects(m3, lnC_ref))
        de_act = duality.vec(duality.design_objects(m3, N, D, lab_, lnC, lnC_ref))
        de_6nd = duality.vec(duality.design_objects(m3, N, C / (6 * N), lab_, lnC, lnC_ref))
        a2 = duality.vec(duality.a2_objects(np.log(N), df.L.values, lab_, lnC, lnC_ref,
                                            a1_x0=[m3.E, np.log(m3.K), m3.gamma]))
        r6 = np.log(6 * N * D / C)
        for i, obj in enumerate(duality.OBJ_NAMES):
            rows.append(dict(dataset=lab, object=obj, A2_estimate=a2[i], A3_analytic=an[i],
                             A3_design_exact6ND=de_6nd[i], A3_design_actual=de_act[i],
                             finite_grid_bias=de_6nd[i] - an[i], flop_accounting_effect=de_act[i] - de_6nd[i],
                             gap_analytic=a2[i] - an[i], gap_design=a2[i] - de_act[i],
                             ln6ND_over_C_min=float(r6.min()), ln6ND_over_C_max=float(r6.max())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_labs_design_decomp.csv"), index=False)
    log("  design decomposition (slope a):\n" + out[out.object == "a"].round(4).to_string())
    return out


# ----------------------------------------------------------------------------- 4. size-matched control for fdep
R_CTRL, B_CTRL = 4, 100


def fdep_size_control(log=print):
    full = load_chinchilla(5)
    N, D, L = full.N.values, full.D.values, full.L.values
    fd = R("m1_chinchilla_fdep.csv")
    ref = est.fit_huber(N, D, L, inits=[theta_of(BESI_PUB)])
    th = theta_of(ref)
    Nbar, Dbar = float(np.exp(np.mean(np.log(N)))), float(np.exp(np.mean(np.log(D))))
    grid_starts = [np.array(s) for s in itertools.product(*fdep.SMALL_GRID.values())]
    inits = [th, theta_of(HOFF_TEX), theta_of(BESI_PUB)] + grid_starts
    band = fd[fd["sample"] == "on-path band |dlnN|<=0.15"].iloc[0]
    n_b = int(band.n)
    keys = ["alpha", "beta", "a", "gamma", "sigma_star", "E"]
    rng = np.random.default_rng(SEED + 4242)
    rows = []
    for r in range(R_CTRL):
        sub = np.sort(rng.choice(len(L), n_b, replace=False))
        Ns, Ds, Ls = N[sub], D[sub], L[sub]
        csn = est.cond_stats(est.jac_log_norm(est.to_norm(th, Nbar, Dbar), Ns, Ds, Nbar, Dbar))
        idxs = boot.draws(n_b, B_CTRL, "pairs", SEED + 4300 + r)
        out = boot.pmap(fdep._boot_sub, idxs, dict(N=Ns, D=Ds, L=Ls, inits=inits))
        sm = boot.summarize(out, keys)
        vd = fdep.variance_decomp(Ns, Ds, ref.alpha, ref.beta)
        # Gaussian profile LR over sigma* (kappa free) on the same random subsample: is the on-path flatness a
        # design effect or just small n?
        prof = fdep.profile_sigma(Ns, Ds, Ls, ref, kind="gauss")
        inside = prof[prof.LR <= 3.84].sigma_star
        row = dict(sample=f"random subsample {r + 1} (n={n_b})", n=n_b, share_trans=vd["share_trans"],
                   **{f"svn{i + 1}": v / np.sqrt(n_b) for i, v in enumerate(csn["sv"])}, cond_norm=csn["cond"],
                   **{f"se_{k}": sm[k]["se"] for k in keys}, nfail=sm["_nfail"],
                   profile_LR_max=float(prof.LR.max()), profile_set_lo=float(inside.min()),
                   profile_set_hi=float(inside.max()))
        rows.append(row)
        log(f"  size control {r + 1}: se alpha {sm['alpha']['se']:.4f} beta {sm['beta']['se']:.4f} a {sm['a']['se']:.4f} "
            f"gamma {sm['gamma']['se']:.4f}; cond_norm {csn['cond']:.0f}; transverse share {vd['share_trans']:.3f}; "
            f"profile LR max {prof.LR.max():.1f}, 95% set [{inside.min():.2f}, {inside.max():.2f}]")
    ctrl = pd.DataFrame(rows)
    med = ctrl.drop(columns=["sample"]).median(numeric_only=True)
    summ = dict(sample=f"median of {R_CTRL} random subsamples (n={n_b})", **med.to_dict())
    fullrow = fd[fd["sample"].str.startswith("full")].iloc[0]
    comp = []
    for k in keys:
        if f"se_{k}" not in band:
            continue
        comp.append(dict(param=k, se_full_n240=float(fullrow[f"se_{k}"]), se_band=float(band[f"se_{k}"]),
                         se_random_same_n=float(med[f"se_{k}"]),
                         ratio_band_to_full=float(band[f"se_{k}"] / fullrow[f"se_{k}"]),
                         ratio_band_to_random=float(band[f"se_{k}"] / med[f"se_{k}"]),
                         sqrt_n_benchmark=float(np.sqrt(fullrow.n / n_b))))
    comp = pd.DataFrame(comp)
    out = pd.concat([ctrl, pd.DataFrame([summ])], ignore_index=True)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_fdep_sizecontrol.csv"), index=False)
    comp.to_csv(os.path.join(TABLES, "m1_chinchilla_fdep_se_ratios.csv"), index=False)
    log("  on-path SE inflation vs size-matched random subsamples:\n" + comp.round(4).to_string())
    return out, comp


# ----------------------------------------------------------------------------- 6. warm-start sensitivity of the bootstrap
N_WS = 40


def _ws_fit(idx, N, D, L, inits, **_):
    mw = est.fit_huber(N[idx], D[idx], L[idx], inits=inits)
    mg = est.fit_huber(N[idx], D[idx], L[idx], grid=sl.FAST_GRID)
    mg = mg if mg.extra["objective"] < mw.extra["objective"] else mw     # grid + warm starts
    return dict(alpha_w=mw.alpha, beta_w=mw.beta, obj_w=mw.extra["objective"], alpha_g=mg.alpha, beta_g=mg.beta,
                obj_g=mg.extra["objective"])


def warmstart_check(log=print):
    """Re-fit the first N_WS horse-race bootstrap draws (Huber, n = 240, same seeds) with the 432-start FAST_GRID in
    addition to the warm starts; report how often the grid finds a lower objective and how much the exponents move."""
    df = load_chinchilla(5)
    N, D, L = df.N.values, df.D.values, df.L.values
    hr = R("m1_chinchilla_horse_race.csv")
    h = hr[(hr.n == 240) & (hr.estimator == "huber")].iloc[0]
    th = np.array([np.log(h.A), np.log(h.B), np.log(h.E), h.alpha, h.beta])
    inits = [th, th, theta_of(HOFF_TEX), theta_of(BESI_PUB)]         # as in horse_race.run
    k = 5
    rows = []
    for sch, idxs in (("pairs", boot.draws(len(L), N_WS, "pairs", SEED + 11 * k)),
                      ("cluster", boot.draws(boot.groups_of(df.cl.values), N_WS, "cluster", SEED + 13 * k + 1))):
        out = [o for o in boot.pmap(_ws_fit, idxs, dict(N=N, D=D, L=L, inits=inits)) if "_error" not in o]
        o = pd.DataFrame(out)
        better = o.obj_g < o.obj_w * (1 - 1e-9)
        rows.append(dict(scheme=sch, draws=len(o), share_grid_better=float(better.mean()),
                         max_abs_dalpha=float(np.abs(o.alpha_g - o.alpha_w).max()),
                         max_abs_dbeta=float(np.abs(o.beta_g - o.beta_w).max()),
                         sd_beta_warm=float(o.beta_w.std(ddof=1)), sd_beta_grid=float(o.beta_g.std(ddof=1)),
                         sd_alpha_warm=float(o.alpha_w.std(ddof=1)), sd_alpha_grid=float(o.alpha_g.std(ddof=1))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_warmstart_check.csv"), index=False)
    log("  warm-start check (Huber, n=240):\n" + out.round(5).to_string())
    return out


# ----------------------------------------------------------------------------- 5. small-sample LR calibration
def system_lr_smallsample(log=print):
    s = R("m1_chinchilla_system_lr.csv")
    Fv = (np.exp(s.LR.clip(lower=0) / s.n_argmin) - 1) * (s.n_argmin - 2) / 2
    s["F_approx"] = Fv
    s["df2"] = s.n_argmin - 2
    s["p_F_approx"] = f_dist.sf(Fv, 2, s.n_argmin - 2)
    out = s[["dataset", "likelihood", "n_loss", "n_argmin", "LR", "LR_p", "F_approx", "df2", "p_F_approx"]]
    out.to_csv(os.path.join(TABLES, "m1_chinchilla_system_lr_smallsample.csv"), index=False)
    log("  system LR small-sample calibration:\n" + out.round(4).to_string())
    return out


def run(log=print):
    estimator_differences(log=log)
    duality_bootcal(log=log)
    labs_design_decomp(log=log)
    system_lr_smallsample(log=log)
    warmstart_check(log=log)
    fdep_size_control(log=log)
