"""Step 1 of ra4_obsfix: reliability of ln C (R4 M7; R1 minor 44; R2 minor 25), design-matched comparison with the
production models removed from the experimental benchmark (R1 10(b)), reconciliation of Table 8's inference
(R1 9(e)), Anderson-Rubin sets for every IV (R1 minor 45), EIV rows with the justified reliability.

Reuses m4_observational code (panel.build_panel, experiments.build_olmo_ladder/build_gadre, lalonde.in_hull,
lalonde.exp_surface_data) by import; the estimators and inference are re-implemented in infer.py so that
(i) both sources of uncertainty enter every test and (ii) the small-sample conventions are explicit.

Outputs (output/tables/ra4_obsfix_*.csv, data/processed/ra4_obsfix/*):
  matched_long.csv          every (sample, surface, output, estimator, parameter) row with all inference variants
  compute_only.csv          EIV / reverse / IV rows (with AR sets) by sample
  inference_reconcile.csv   Table 8 family-FE rows: CR1 (all k), CR1 (nested), CV3, WCR variants, G*, leave-one-out
  leverage_familyFE.csv     partial-leverage shares by developer and family (family FE, theta_N and theta_D)
  reliability_models.csv    model-level compute comparisons (hardware-time reconstruction, Epoch methods)
  reliability_summary.csv   sigma_u bounds and implied reliabilities by sample (pooled / within family)
  surfaces.csv              surface fits (n, R2, Tobit scale, censoring counts)
  matched_hull_members.csv  which observational models are in which support
  bench_thetaC_draws.npz    benchmark theta_C draws (OLS, C only) for the TFP step
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ra4common as rc  # noqa: E402

sys.path.insert(0, rc.M4DIR)
import experiments as ex  # noqa: E402  (m4)
import lalonde as ll  # noqa: E402      (m4)
import panel  # noqa: E402              (m4)

import infer as inf  # noqa: E402

OUTPUTS = {"y_hellaswag": "HellaSwag", "y_core": "Composite", "y_arc_c": "ARC-Challenge", "y_winogrande": "Winogrande"}
ESTIMATORS = [("OLS", None, ()), ("Year FE", "year", ()), ("Developer FE", "developer", ()),
              ("Family FE", "family", ()), ("Lab x period FE", "devyear", ()),
              ("Notability-propensity control", None, ("ps", "ps2"))]
PARAMS = [("theta_N", ["n", "d"], "n"), ("theta_D", ["n", "d"], "d"), ("theta_C", ["c"], "c")]
SIGMA_CONF = np.log(3) / 1.645            # Epoch 'Confident' class: 90% interval within a factor of 3
MFU_ASSUMED = 0.30                          # Epoch's default utilization assumption (level only; cancels in SDs)


# ============================================================================== data
def load():
    df = panel.build_panel(verbose=False)
    df["devyear"] = df.developer.astype(str) + "_" + df.year.astype(str)
    lad = ex.build_olmo_ladder()
    gad = ex.build_gadre()
    return df, lad, gad


def surface_runs(lad, gad, which):
    """'legacy' = m4 (OLMo ladder + OLMo-2 7B/13B targets + Gadre N>=0.1B); 'strict' = without OLMo-2 targets."""
    L = lad if which == "legacy" else lad[~lad.target]
    E = ll.exp_surface_data(L, gad, "ladder+gadre").reset_index(drop=True)
    # clusters for the design-conditional wild bootstrap: runs of the same design and model size (shared trunk/seed)
    E["scl"] = E.ds + "_" + np.round(E.n, 3).astype(str)
    return E


# ============================================================================== item 1: reliability of ln C
def peak_flops():
    h = pd.read_csv(os.path.join(rc.RAW, "epoch_models", "ml_hardware.csv"))
    pk = h["Tensor-FP16/BF16 performance (FLOP/s)"].fillna(h["FP16 (half precision) performance (FLOP/s)"])
    return dict(zip(h["Hardware name"].str.strip(), pk))


def reliability(df, samples):
    """Which compute comparisons are independent of (N, D)?  Epoch's 'Training compute (FLOP)' is itself 6ND
    ('Operation counting'), a developer-reported FLOP count (usually 6ND), a hardware-time estimate, or a geometric
    mean of 6ND and hardware time.  Only hardware time x peak throughput x assumed utilization is independent of
    reported N and D.  We reconstruct it from Epoch's structured fields (chip-hours, or hardware quantity x training
    hours; hardware peak dense 16-bit FLOP/s) with a COMMON utilization assumption: Epoch's 'Hardware utilization'
    field is back-calculated from its own compute estimate for most rows, so using it would make the comparison
    circular.  sd(ln C_hw - ln 6ND) then bounds the error s.d. of ln 6ND from above (it also contains the
    dispersion of true utilization and any error in chip-hours)."""
    e = pd.read_csv(os.path.join(rc.RAW, "epoch_models", "all_ai_models.csv"), low_memory=False).drop_duplicates("Model")
    pk = peak_flops()
    m = df[df.main & df.epoch_name.notna()].merge(
        e[["Model", "Training chip-hours", "Hardware quantity", "Training time (hours)", "Training hardware",
           "Training compute estimation method", "Confidence"]], left_on="epoch_name", right_on="Model", how="left")
    m["chip_hours"] = m["Training chip-hours"].fillna(m["Hardware quantity"] * m["Training time (hours)"])
    hw = m["Training hardware"].fillna("").astype(str).str.split(",").str[0].str.strip()
    m["peak"] = [pk.get(h, np.nan) for h in hw]
    m["C_hw"] = m.chip_hours * 3600 * m.peak * MFU_ASSUMED
    meth = m["C_method"].fillna("").astype(str)
    m["method_class"] = np.select(
        [meth.eq("Operation counting"), meth.str.startswith("Reported") & ~meth.str.contains("Hardware"),
         meth.eq("Hardware"), meth.str.contains("Hardware")],
        ["Epoch 6ND (operation counting)", "Developer-reported FLOP", "Epoch hardware-time", "Epoch mixed (6ND and hardware)"],
        default="Other/unknown")
    m["dev_epoch"] = np.log(m.C_epoch / m.C)
    m["dev_hw"] = np.log(m.C_hw / m.C)
    # m4's 'independent' set (Hardware* or Reported* methods), which Table 8 / Section VI.A used
    m["m4_indep"] = meth.str.startswith("Hardware") | meth.str.startswith("Reported")
    cols = ["model", "family", "developer", "N", "D", "C", "C_epoch", "C_method", "method_class", "Confidence",
            "Training hardware", "chip_hours", "peak", "C_hw", "dev_epoch", "dev_hw", "m4_indep"]
    M = m[cols].copy()
    M.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_reliability_models.csv"), index=False)

    rng = np.random.default_rng(rc.SEED + 1)

    def sd_ci(x, fam, B=9999):
        x, fam = np.asarray(x, float), np.asarray(fam)
        ok = np.isfinite(x)
        x, fam = x[ok], fam[ok]
        fams = pd.unique(fam)
        idx = {f: np.where(fam == f)[0] for f in fams}
        sds = []
        for _ in range(B):
            pick = rng.choice(fams, len(fams))
            ii = np.concatenate([idx[f] for f in pick])
            sds.append(np.std(x[ii], ddof=1) if len(ii) > 1 else np.nan)
        lo, hi = np.nanpercentile(sds, [2.5, 97.5])
        mad = 1.4826 * np.median(np.abs(x - np.median(x)))
        return dict(n=int(len(x)), n_fam=int(len(fams)), mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)),
                    sd_lo=float(lo), sd_hi=float(hi), sd_mad=float(mad))

    hwm = M[np.isfinite(M.dev_hw)]
    within = hwm.groupby("family").filter(lambda g: len(g) >= 2)
    dev_w = within.dev_hw - within.groupby("family").dev_hw.transform("mean")
    k_w = within.groupby("family").size()
    sd_within = float(np.sqrt((dev_w ** 2).sum() / (len(within) - len(k_w)))) if len(k_w) else np.nan
    comps = {
        "hardware-time reconstruction (independent of N, D)": sd_ci(hwm.dev_hw, hwm.family),
        "m4 'independent' set (Hardware*/Reported* Epoch methods)": sd_ci(M[M.m4_indep].dev_epoch, M[M.m4_indep].family),
        "Epoch operation counting (Epoch's own 6ND)": sd_ci(M[M.method_class == "Epoch 6ND (operation counting)"].dev_epoch,
                                                            M[M.method_class == "Epoch 6ND (operation counting)"].family),
        "developer-reported FLOP": sd_ci(M[M.method_class == "Developer-reported FLOP"].dev_epoch,
                                         M[M.method_class == "Developer-reported FLOP"].family),
    }
    sig_hw = comps["hardware-time reconstruction (independent of N, D)"]["sd"]
    sig_hw_hi = comps["hardware-time reconstruction (independent of N, D)"]["sd_hi"]
    sig_m4 = comps["m4 'independent' set (Hardware*/Reported* Epoch methods)"]["sd"]
    rows = []
    for sname, d in samples.items():
        c = d.c.to_numpy()
        var_p = float(np.var(c, ddof=1))
        ct = c - d.groupby("family").c.transform("mean").to_numpy()
        F = d.family.nunique()
        var_w_correct = float((ct ** 2).sum() / (len(d) - F))         # within variance net of the F family means
        var_w_table8 = float(np.var(ct, ddof=1))                       # what the Table 8 code used (n-1 denominator)
        for lab, s in [("hardware-time sd (upper bound on sigma_u)", sig_hw),
                       ("hardware-time sd, upper 95% bootstrap bound", sig_hw_hi),
                       ("m4 'independent' set sd (as in Section VI.A)", sig_m4),
                       ("Epoch 'Confident' class, ln3/1.645 (as in Table 8)", SIGMA_CONF)]:
            rows.append(dict(sample=sname, n=len(d), families=F, sigma_u=s, var_c_pooled=var_p, var_c_within=var_w_correct,
                             var_c_within_table8=var_w_table8, reliability_pooled=1 - s ** 2 / var_p,
                             reliability_within=1 - s ** 2 / var_w_correct,
                             reliability_within_table8_formula=1 - s ** 2 / var_w_table8, sigma_label=lab))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_reliability_summary.csv"), index=False)
    C = pd.DataFrame([dict(comparison=k, **v) for k, v in comps.items()])
    C["sd_within_family_hw"] = np.nan
    C.loc[C.comparison.str.startswith("hardware"), "sd_within_family_hw"] = sd_within
    C.loc[C.comparison.str.startswith("hardware"), "n_within_families"] = len(k_w)
    C.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_reliability_comparisons.csv"), index=False)
    return dict(sigma_hw=sig_hw, sigma_hw_hi=sig_hw_hi, sigma_m4=sig_m4, sd_within_hw=sd_within, comps=comps,
                models=M, summary=R)


# ============================================================================== surfaces and benchmark draws
def fit_surfaces(E, out, rng, center, tobit=True):
    """OLS quadratic surface (m4 specification) with design-conditional wild-cluster bootstrap draws, and a Tobit
    (left-censored at the logit floor) version with parametric-bootstrap draws.  Returns dict with point
    coefficient vectors (5 slope/curvature terms) and draw matrices (5 x B)."""
    E = E.dropna(subset=[out])
    Z, F = inf.surface_design(E, center)
    y = E[out].to_numpy(float)
    b, Bw = inf.surface_wild(Z, y, rc.B_SURF, rng, clusters=E.scl.to_numpy())
    _, Br = inf.surface_wild(Z, y, 2000, rng, clusters=None)          # run-level Rademacher (robustness)
    r2 = 1 - np.sum((y - Z @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    res = dict(ols=dict(b=b[1:6], draws=Bw[1:6, :], draws_run=Br[1:6, :], R2=float(r2)), n=len(E),
               n_cens=int((y <= rc.FLOOR_LOGIT + 1e-9).sum()))
    if not tobit:
        return res
    bt, s, r = inf.tobit_fit(Z, y, rc.FLOOR_LOGIT)
    mu = Z @ bt
    draws = []
    for _ in range(rc.B_TOBIT):
        ys = np.maximum(rc.FLOOR_LOGIT, mu + s * rng.standard_normal(len(y)))
        bb, _, _ = inf.tobit_fit(Z, ys, rc.FLOOR_LOGIT, x0=np.r_[bt, np.log(s)])
        draws.append(bb[1:6])
    res["tobit"] = dict(b=bt[1:6], draws=np.array(draws).T, s=s,
                        converged=bool(np.max(np.abs(r.jac)) < 1e-4))   # BFGS 'precision loss' flag is benign here
    return res


def _calibrate_level(y, mu):
    """Level a minimizing sum (y - max(floor, mu + a))^2: the harness-specific intercept of the observational
    outputs is unknown, so the benchmark's level is matched to the observed outputs before clipping (levels are a
    Hicks-neutral nuisance; only the clipping depends on them)."""
    from scipy.optimize import minimize_scalar
    f = lambda a: np.sum((y - np.maximum(rc.FLOOR_LOGIT, mu + a)) ** 2)
    a0 = float(np.mean(y - mu))
    r = minimize_scalar(f, bounds=(a0 - 8, a0 + 8), method="bounded", options=dict(xatol=1e-8))
    return float(r.x)


def bench_outputs(F_obs, S, surf_kind, y_obs):
    """Benchmark outputs y*_i on the observational design and their bootstrap draws (n x B).
    'ols'  : m4's surface (OLS on the clipped experimental logits), slope terms only (levels are irrelevant).
    'tobit': latent surface from the censored (Tobit) fit, level calibrated to the observed outputs, then clipped
             at the same floor as the observed logits, so both sides are censored alike."""
    b_s, D_s = S[surf_kind]["b"], S[surf_kind]["draws"]
    if surf_kind == "ols":
        return F_obs @ b_s, F_obs @ D_s
    mu = F_obs @ b_s
    ystar = np.maximum(rc.FLOOR_LOGIT, mu + _calibrate_level(y_obs, mu))
    MU = F_obs @ D_s
    Y = np.empty_like(MU)
    for i in range(MU.shape[1]):
        Y[:, i] = np.maximum(rc.FLOOR_LOGIT, MU[:, i] + _calibrate_level(y_obs, MU[:, i]))
    return ystar, Y


def notability_ps(d):
    import statsmodels.formula.api as smf
    try:
        x = d.assign(notable_i=d.notable.astype(int))
        if x.notable_i.nunique() < 2:
            raise ValueError("no variation in notability")
        pm = smf.logit("notable_i ~ n + d + t", data=x).fit(disp=0, maxiter=200)
        ps = pm.predict(x).to_numpy()
        if not np.all(np.isfinite(ps)) or np.nanstd(ps) < 1e-6:
            raise ValueError("degenerate propensity")
        return ps
    except Exception as exc:  # perfect separation in small samples
        rc.log(f"  notability propensity unavailable: {exc}")
        return None


# ============================================================================== estimator rows
def estimator_rows(d, out, surf_kind, S, center, rng, sample, surface):
    """All (estimator, parameter) rows for one sample x surface x output."""
    rows = []
    F_obs = inf.obs_features(d.n, d.d, center)
    ystar, Ystar = bench_outputs(F_obs, S, surf_kind, d[out].to_numpy(float))
    dd = d.copy()
    dd["ystar"] = ystar
    dd["diff"] = dd[out].to_numpy() - ystar
    ps = notability_ps(dd)
    if ps is not None:
        dd["ps"], dd["ps2"] = ps, ps ** 2
    g_dev = inf.cluster_ids(dd.developer)
    g_fam = inf.cluster_ids(dd.family)
    for est, fe, extra in ESTIMATORS:
        if extra and ps is None:
            continue
        for par, xs, col in PARAMS:
            X, names, k_all, k_nest = inf.build_X(dd, xs, fe, extra)
            j = names.index(col)
            fit = inf.LinFit(X, dd[out].to_numpy(), g_dev, j)
            base = dict(sample=sample, surface=surface, surface_kind=surf_kind, output=out, estimator=est, param=par,
                        n_obs=len(dd), n_dev=int(dd.developer.nunique()), n_fam=int(dd.family.nunique()),
                        n_fam_ge2=int((dd.family.value_counts() >= 2).sum()), identified=bool(fit.identified))
            if not fit.identified:
                rows.append(base)
                continue
            b_obs = float(fit.b[j])
            bench = float(fit.h @ ystar)
            bdraw = fit.h @ Ystar
            bench_se = float(np.std(bdraw, ddof=1))
            se_all, se_nest = fit.se_cr1(k_all), fit.se_cr1(k_nest)
            _, loo, cv3 = inf.jackknife(dd, "diff", xs, fe, col, "developer", extra)
            _, _, cv3_y = inf.jackknife(dd, out, xs, fe, col, "developer", extra)   # CV3 of Obs. itself (reviewer)
            bias = b_obs - bench
            # inference variants for H0: bias = 0
            w_comb = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest, bench_draws=bdraw[rng.integers(0, len(bdraw), rc.B_WILD)],
                                   bench_var=bench_se ** 2)
            w_fixed = inf.wild_test(fit, bench, rc.B_WILD, rng, k_all)             # Table 8 / m4 protocol
            w_sub = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest, bench_draws=bdraw[rng.integers(0, len(bdraw), rc.B_WILD)],
                                  bench_var=bench_se ** 2, wlevel=g_fam)
            from scipy.stats import norm, t as tdist
            G = fit.G
            # reviewer addition: restricted wild-cluster bootstrap-t STUDENTIZED BY CV3 on the bias outcome y - y*,
            # with benchmark draws; own seeded stream so that the builder's random draws above are unchanged
            rng_v3 = np.random.default_rng([rc.SEED, 3, zlib.crc32(f"{sample}|{surface}|{out}|{surf_kind}|{est}|{par}".encode())])
            w_v3 = inf.wild_test_cv3(X, dd["diff"].to_numpy(float), g_dev, j, rc.B_WILD, rng_v3,
                                     bench_draws=bdraw[rng_v3.integers(0, len(bdraw), rc.B_WILD)] - bench,
                                     bench_var=bench_se ** 2)
            se_b_nest = np.sqrt(se_nest ** 2 + bench_se ** 2)
            se_b_cv3 = np.sqrt(cv3 ** 2 + bench_se ** 2) if np.isfinite(cv3) else np.nan
            rows.append(dict(base, obs=b_obs, obs_se_cr1_allk=se_all, obs_se_cr1_nested=se_nest, obs_se_cv3=cv3,
                             bench=bench, bench_se=bench_se, bench_lo=float(np.percentile(bdraw, 2.5)),
                             bench_hi=float(np.percentile(bdraw, 97.5)),
                             bench_se_runlevel=float(np.std(fit.h @ (F_obs @ S[surf_kind]["draws_run"]), ddof=1))
                             if surf_kind == "ols" else np.nan,
                             bias=bias, bias_se_table8=float(np.sqrt(se_all ** 2 + bench_se ** 2)),
                             bias_se_nested=float(se_b_nest), bias_se_cv3=float(se_b_cv3) if np.isfinite(se_b_cv3) else np.nan,
                             p_normal_table8=float(2 * norm.sf(abs(bias) / np.sqrt(se_all ** 2 + bench_se ** 2))),
                             p_t_nested=float(2 * tdist.sf(abs(bias) / se_b_nest, G - 1)),
                             p_t_cv3=float(2 * tdist.sf(abs(bias) / se_b_cv3, G - 1)) if np.isfinite(se_b_cv3) else np.nan,
                             p_wcr_combined=w_comb["p"], p_wcr_benchfixed_allk=w_fixed["p"], p_wcr_subcluster_family=w_sub["p"],
                             B_wild=rc.B_WILD, G_star=fit.g_star(), loo_bias_min=float(np.nanmin(loo.values)),
                             loo_bias_max=float(np.nanmax(loo.values)), loo_argmin=str(loo.idxmin()) if loo.notna().any() else "",
                             loo_argmax=str(loo.idxmax()) if loo.notna().any() else "", loo_n_unidentified=int(loo.isna().sum()),
                             mde80=float(2.8 * se_b_nest),
                             mde80_cv3=float(2.8 * se_b_cv3) if np.isfinite(se_b_cv3) else np.nan,
                             p_wcr_cv3_combined=w_v3["p"] if w_v3 else np.nan,
                             obs_se_cv3_y=float(cv3_y) if np.isfinite(cv3_y) else np.nan,
                             cv3_check=w_v3["cv3"] if w_v3 else np.nan,
                             mde80_cv3_tG1=float((tdist.ppf(0.975, G - 1) + tdist.ppf(0.80, G - 1)) * se_b_cv3)
                             if np.isfinite(se_b_cv3) else np.nan))
    return rows, dd


# ============================================================================== compute-only rows: EIV, reverse, IV
def ar_set(dd, out, z, fe, grid, crit):
    """Anderson-Rubin set for theta_C: {theta: |t_CR1(z)| < crit in y - theta*c on (controls, z)}."""
    g = inf.cluster_ids(dd.developer)
    X, names, k_all, k_nest = inf.build_X(dd, [z], fe)
    j = names.index(z)
    fit = inf.LinFit(X, np.zeros(len(dd)), g, j)
    y, c = dd[out].to_numpy(float), dd.c.to_numpy(float)
    acc = []
    for th in grid:
        yy = y - th * c
        b = fit.h @ yy
        u = yy - X @ (fit.XtXi @ (X.T @ yy))
        se = np.sqrt(fit.vjj(u[:, None], k_nest)[0])
        if abs(b / se) < crit:
            acc.append(th)
    if not acc:
        return np.nan, np.nan, "empty"
    acc = np.array(acc)
    step = grid[1] - grid[0]
    unb = acc.min() <= grid[0] + 1e-9 or acc.max() >= grid[-1] - 1e-9
    gaps = np.any(np.diff(acc) > 1.5 * step)
    return float(acc.min()), float(acc.max()), "unbounded" if unb else ("disjoint" if gaps else "interval")


def compute_only_rows(dd, out, S, surf_kind, center, rng, sample, surface, rel):
    rows = []
    F_obs = inf.obs_features(dd.n, dd.d, center)
    ystar = F_obs @ S[surf_kind]["b"]
    Ystar = F_obs @ S[surf_kind]["draws"]
    g = inf.cluster_ids(dd.developer)
    from scipy.stats import t as tdist
    # --- EIV-corrected OLS and family FE, H0: b_obs = lambda * bench
    sig = {"hardware-time sd (upper bound)": rel["sigma_hw"], "m4 'independent' set sd": rel["sigma_m4"],
           "Epoch 'Confident' class": SIGMA_CONF}
    for est, fe in [("OLS", None), ("Family FE", "family")]:
        X, names, k_all, k_nest = inf.build_X(dd, ["c"], fe)
        j = names.index("c")
        fit = inf.LinFit(X, dd[out].to_numpy(), g, j)
        c = dd.c.to_numpy()
        if fe is None:
            vc = np.var(c, ddof=1)
            vc8 = vc
        else:
            ct = c - dd.groupby(fe).c.transform("mean").to_numpy()
            vc = (ct ** 2).sum() / (len(dd) - dd[fe].nunique())
            vc8 = np.var(ct, ddof=1)
        bench = float(fit.h @ ystar)
        bdraw = fit.h @ Ystar
        bse = float(np.std(bdraw, ddof=1))
        se_nest = fit.se_cr1(k_nest)
        for lab, s in sig.items():
            for formula, v in [("correct", vc), ("table8", vc8)]:
                if formula == "table8" and lab != "Epoch 'Confident' class":
                    continue
                lam = 1 - s ** 2 / v
                if lam <= 0:
                    rows.append(dict(sample=sample, surface=surface, output=out, estimator=f"{est}, EIV ({lab})", formula=formula,
                                     reliability=lam, sigma_u=s))
                    continue
                w = inf.wild_test(fit, lam * bench, rc.B_WILD, rng, k_nest,
                                  bench_draws=lam * bdraw[rng.integers(0, len(bdraw), rc.B_WILD)], bench_var=(lam * bse) ** 2)
                rows.append(dict(sample=sample, surface=surface, output=out, estimator=f"{est}, EIV ({lab})", formula=formula,
                                 reliability=lam, sigma_u=s, obs=float(fit.b[j] / lam), obs_se=se_nest / lam, bench=bench,
                                 bench_se=bse, bias=float(fit.b[j] / lam - bench),
                                 bias_se=float(np.sqrt((se_nest / lam) ** 2 + bse ** 2)), p_wcr_combined=w["p"],
                                 n_obs=len(dd), n_dev=int(dd.developer.nunique())))
    # --- reverse (Nerlove) regression: c on y; theta = 1/r;  H0: r_obs = r_bench
    for est, fe in [("Reverse regression", None), ("Reverse regression, family FE", "family")]:
        X, names, k_all, k_nest = inf.build_X(dd.assign(yo=dd[out]), ["yo"], fe)
        j = names.index("yo")
        fit = inf.LinFit(X, dd.c.to_numpy(), g, j)
        Xs, _, _, _ = inf.build_X(dd.assign(yo=ystar), ["yo"], fe)
        # benchmark: same regression with y* as regressor (nonlinear in y*): r_b = (M y*)'(M c)/(M y*)'(M y*)
        Mproj = lambda A: A - Xs[:, [0] + list(range(2, Xs.shape[1]))] @ np.linalg.lstsq(
            Xs[:, [0] + list(range(2, Xs.shape[1]))], A, rcond=None)[0]
        cm_ = Mproj(dd.c.to_numpy()[:, None])[:, 0]
        ym = Mproj(ystar[:, None])[:, 0]
        r_b = float(ym @ cm_ / (ym @ ym))
        YM = Mproj(Ystar)
        r_draws = (YM * cm_[:, None]).sum(0) / (YM ** 2).sum(0)
        rse = float(np.std(r_draws, ddof=1))
        se_nest = fit.se_cr1(k_nest)
        w = inf.wild_test(fit, r_b, rc.B_WILD, rng, k_nest, bench_draws=r_draws[rng.integers(0, len(r_draws), rc.B_WILD)],
                          bench_var=rse ** 2)
        r_o = float(fit.b[j])
        th_draws = 1 / r_draws
        # Nerlove/Hall bracket: with no transmission bias, 1/r = theta_forward / R^2_forward (noise in y attenuates r)
        Xf, nf, _, _ = inf.build_X(dd, ["c"], fe)
        ff = inf.LinFit(Xf, dd[out].to_numpy(), g, nf.index("c"))
        yv = dd[out].to_numpy()
        R2f = float(1 - (ff.u @ ff.u) / (Mproj(yv[:, None])[:, 0] @ Mproj(yv[:, None])[:, 0]))
        rows.append(dict(sample=sample, surface=surface, output=out, estimator=est, formula="",
                         forward=float(ff.b[nf.index("c")]), R2_forward_within=R2f,
                         bracket_implied=float(ff.b[nf.index("c")]) / R2f, r_obs=r_o, r_bench=r_b,
                         t_r_scale=float((r_o - r_b) / np.hypot(se_nest, rse)), obs=1 / r_o,
                         obs_se=se_nest / r_o ** 2, bench=1 / r_b, bench_se=float(np.std(th_draws, ddof=1)),
                         bias=1 / r_o - 1 / r_b, bias_se=float(np.sqrt((se_nest / r_o ** 2) ** 2 + np.var(th_draws, ddof=1))),
                         p_wcr_combined=w["p"], n_obs=len(dd), n_dev=int(dd.developer.nunique())))
    # --- IV: 2SLS with cluster-robust first-stage F and Anderson-Rubin sets (t(G-1) and normal critical values)
    grid = np.round(np.arange(-3.0, 4.0 + 1e-9, 0.005), 4)
    for est, z, fe in [("IV: frontier FLOP/$", "z_frontier", None), ("IV: frontier FLOP/$ + developer FE", "z_frontier", "developer"),
                       ("IV: own-hardware FLOP/$", "z_own", None)]:
        dz = dd.dropna(subset=[z])
        if len(dz) < 8 or dz[z].nunique() < 2:
            rows.append(dict(sample=sample, surface=surface, output=out, estimator=est, formula="", n_obs=len(dz),
                             note="instrument unavailable (fewer than 8 models or no variation)"))
            continue
        gz = inf.cluster_ids(dz.developer)
        X, names, k_all, k_nest = inf.build_X(dz, [z], fe)
        j = names.index(z)
        fs = inf.LinFit(X, dz.c.to_numpy(), gz, j)
        if not fs.identified:
            rows.append(dict(sample=sample, surface=surface, output=out, estimator=est, formula="", n_obs=len(dz),
                             note="instrument collinear with controls"))
            continue
        F = float((fs.b[j] / fs.se_cr1(k_nest)) ** 2)
        red = inf.LinFit(X, dz[out].to_numpy(), gz, j)
        theta_iv = float(red.b[j] / fs.b[j])
        Fo = inf.obs_features(dz.n, dz.d, center)
        ys_z = Fo @ S[surf_kind]["b"]
        bench_iv = float((fs.h @ ys_z) / fs.b[j])
        bench_draws = (fs.h @ (Fo @ S[surf_kind]["draws"])) / fs.b[j]
        G = fs.G
        lo_t, hi_t, typ_t = ar_set(dz, out, z, fe, grid, tdist.ppf(0.975, G - 1))
        lo_n, hi_n, typ_n = ar_set(dz, out, z, fe, grid, 1.959964)
        rows.append(dict(sample=sample, surface=surface, output=out, estimator=est, formula="", obs=theta_iv, bench=bench_iv,
                         bench_se=float(np.std(bench_draws, ddof=1)), bias=theta_iv - bench_iv, F_first=F, n_obs=len(dz),
                         n_dev=int(dz.developer.nunique()), n_values_z=int(dz[z].nunique()),
                         ar_lo_t=lo_t, ar_hi_t=hi_t, ar_type_t=typ_t, ar_lo_n=lo_n, ar_hi_n=hi_n, ar_type_n=typ_n,
                         ar_grid=f"[{grid[0]}, {grid[-1]}] step 0.005"))
    return rows


# ============================================================================== item 2: reconcile Table 8 inference
def reconcile(d_leg, out, S_leg, center, rng):
    """Family FE rows of Table 8 (legacy benchmark, 57 models): why the wild p-values disagree with the SEs."""
    dd = d_leg.copy()
    F_obs = inf.obs_features(dd.n, dd.d, center)
    ystar = F_obs @ S_leg["ols"]["b"]
    Ystar = F_obs @ S_leg["ols"]["draws"]
    g = inf.cluster_ids(dd.developer)
    fam_n = dd.family.value_counts()
    rows, lev_rows = [], []
    for par, col in [("theta_N", "n"), ("theta_D", "d")]:
        X, names, k_all, k_nest = inf.build_X(dd, ["n", "d"], "family")
        j = names.index(col)
        fit = inf.LinFit(X, dd[out].to_numpy(), g, j)
        bench = float(fit.h @ ystar)
        bdraw = fit.h @ Ystar
        bse = float(np.std(bdraw, ddof=1))
        bias = float(fit.b[j]) - bench
        se_all, se_nest = fit.se_cr1(k_all), fit.se_cr1(k_nest)
        # drop singleton families (they contribute nothing to within slopes; they only change n and k)
        keep = dd.family.map(fam_n).to_numpy() >= 2
        d2 = dd[keep]
        X2, n2, k2_all, k2_nest = inf.build_X(d2, ["n", "d"], "family")
        fit2 = inf.LinFit(X2, d2[out].to_numpy(), inf.cluster_ids(d2.developer), n2.index(col))
        _, loo, cv3 = inf.jackknife(dd.assign(diff=dd[out] - ystar), "diff", ["n", "d"], "family", col)
        from scipy.stats import norm, t as tdist
        variants = []
        variants.append(("CR1, k counts 29 family dummies (Table 8)", se_all, bse,
                         2 * norm.sf(abs(bias) / np.hypot(se_all, bse)), "normal"))
        variants.append(("CR1, singleton families dropped (n=%d, k=%d)" % (len(d2), k2_all), fit2.se_cr1(k2_all), bse,
                         2 * norm.sf(abs(bias) / np.hypot(fit2.se_cr1(k2_all), bse)), "normal"))
        variants.append(("CR1, nested fixed effects not counted", se_nest, bse,
                         2 * tdist.sf(abs(bias) / np.hypot(se_nest, bse), fit.G - 1), "t(G-1)"))
        variants.append(("CV3 jackknife", cv3, bse, 2 * tdist.sf(abs(bias) / np.hypot(cv3, bse), fit.G - 1)
                         if np.isfinite(cv3) else np.nan, "t(G-1)"))
        w1 = inf.wild_test(fit, bench, rc.B_WILD, rng, k_all)
        w2 = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest)
        w3 = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest, bench_draws=bdraw[rng.integers(0, len(bdraw), rc.B_WILD)],
                           bench_var=bse ** 2)
        w4 = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest, bench_draws=bdraw[rng.integers(0, len(bdraw), rc.B_WILD)],
                           bench_var=bse ** 2, restricted=False)
        w5 = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest, bench_draws=bdraw[rng.integers(0, len(bdraw), rc.B_WILD)],
                           bench_var=bse ** 2, wlevel=inf.cluster_ids(dd.family))
        w6 = inf.wild_test(fit, bench, rc.B_WILD, rng, k_nest, bench_draws=bdraw[rng.integers(0, len(bdraw), rc.B_WILD)],
                           bench_var=bse ** 2, wlevel=np.arange(len(dd)))
        rng_v3 = np.random.default_rng([rc.SEED, 4, zlib.crc32(par.encode())])
        w7 = inf.wild_test_cv3(X, (dd[out] - ystar).to_numpy(float), g, j, rc.B_WILD, rng_v3,
                               bench_draws=bdraw[rng_v3.integers(0, len(bdraw), rc.B_WILD)] - bench, bench_var=bse ** 2)
        for lab, se, bs, p, ref in variants:
            rows.append(dict(param=par, variant=lab, bias=bias, obs_se=se, bench_se=bs, bias_se=float(np.hypot(se, bs)),
                             p=float(p), reference=ref))
        for lab, w in [("WCR, benchmark fixed (Table 8 protocol)", w1), ("WCR, benchmark fixed, nested k", w2),
                       ("WCR, benchmark uncertainty included", w3), ("WCU, benchmark uncertainty included", w4),
                       ("WCR subcluster (weights by family), benchmark included", w5),
                       ("WCR subcluster (weights by model), benchmark included", w6),
                       ("WCR studentized by CV3, benchmark included", w7)]:
            rows.append(dict(param=par, variant=lab, bias=bias, p=w["p"], reference=f"bootstrap-t, B={rc.B_WILD}, Webb",
                             T=w["T"]))
        rows.append(dict(param=par, variant="Effective number of clusters G* (rho=0)", bias=bias, value=fit.g_star(),
                         G=fit.G))
        rows.append(dict(param=par, variant="Leave-one-developer-out bias: min / max", bias=bias,
                         loo_min=float(np.nanmin(loo)), loo_max=float(np.nanmax(loo)), loo_argmin=str(loo.idxmin()),
                         loo_argmax=str(loo.idxmax())))
        for lab_by, col_by in [("developer", "developer"), ("family", "family")]:
            sh = fit.leverage_shares(dd[col_by].to_numpy())
            for k_, v_ in sh.items():
                lev_rows.append(dict(param=par, by=lab_by, unit=k_, share=float(v_)))
        for k_, v_ in loo.items():
            lev_rows.append(dict(param=par, by="loo_bias_developer_dropped", unit=k_, share=float(v_)))
        rows.append(dict(param=par, variant="dof factor (n-1)/(n-k): all k / nested / singletons dropped", bias=bias,
                         value=(fit.n - 1) / (fit.n - k_all), value2=(fit.n - 1) / (fit.n - k_nest),
                         value3=(fit2.n - 1) / (fit2.n - k2_all), n=fit.n, k_all=k_all, k_nested=k_nest,
                         singleton_families=int((fam_n == 1).sum())))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_inference_reconcile.csv"), index=False)
    pd.DataFrame(lev_rows).to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_leverage_familyFE.csv"), index=False)
    return R


# ============================================================================== main
def main():
    t0 = time.time()
    rng = np.random.default_rng(rc.SEED)
    df, lad, gad = load()
    E = {"strict": surface_runs(lad, gad, "strict"), "legacy": surface_runs(lad, gad, "legacy")}
    centers = {k: (v.n.mean(), v.d.mean()) for k, v in E.items()}
    obs_all = df[df.main].copy()
    hull = {k: ll.in_hull(v, obs_all[["n", "d"]].values) for k, v in E.items()}
    mem = obs_all[["model", "family", "developer", "N", "D", "M", "date"]].assign(in_strict_hull=hull["strict"],
                                                                                 in_legacy_hull=hull["legacy"])
    mem.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_matched_hull_members.csv"), index=False)
    # samples: (sample label, observational rows, surface used)
    SAMPLES = [("strict support (20)", obs_all[hull["strict"]], "strict"),
               ("legacy support (57)", obs_all[hull["legacy"]], "legacy"),
               ("legacy support, strict surface (extrapolated)", obs_all[hull["legacy"]], "strict")]
    rc.log(f"supports: strict {hull['strict'].sum()} models, legacy {hull['legacy'].sum()} models "
           f"({time.time() - t0:.0f}s)")

    # ---- item 1: reliability
    rel = reliability(df, {"main (128)": obs_all, "legacy support (57)": obs_all[hull["legacy"]],
                           "strict support (20)": obs_all[hull["strict"]]})
    rc.log(f"reliability: sigma_hw={rel['sigma_hw']:.3f} (upper 95% {rel['sigma_hw_hi']:.3f}); m4 set {rel['sigma_m4']:.3f}")

    # ---- surfaces
    surf, surf_rows = {}, []
    for sk, Ek in E.items():
        for out in OUTPUTS:
            S = fit_surfaces(Ek, out, rng, centers[sk], tobit=(out != "y_core"))
            surf[(sk, out)] = S
            surf_rows.append(dict(surface=sk, output=out, n_runs=S["n"], n_censored=S["n_cens"], R2_ols=S["ols"]["R2"],
                                  tobit_scale=S["tobit"]["s"] if "tobit" in S else np.nan,
                                  tobit_converged=S["tobit"]["converged"] if "tobit" in S else None,
                                  center_N=float(np.exp(centers[sk][0])), center_D=float(np.exp(centers[sk][1])),
                                  n_clusters_wild=int(Ek.scl.nunique()),
                                  **{f"ols_{k}": v for k, v in zip(["n", "d", "n2", "d2", "nd"], S["ols"]["b"])},
                                  **({f"tobit_{k}": v for k, v in zip(["n", "d", "n2", "d2", "nd"], S["tobit"]["b"])}
                                     if "tobit" in S else {})))
    pd.DataFrame(surf_rows).to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_surfaces.csv"), index=False)
    rc.log(f"surfaces fitted ({time.time() - t0:.0f}s)")

    # ---- design-matched rows
    allrows, co_rows, bench_draws = [], [], {}
    for sample, dsamp, sk in SAMPLES:
        for out in OUTPUTS:
            d = dsamp.dropna(subset=[out]).copy()
            kinds = ["ols"] + (["tobit"] if out in ("y_arc_c", "y_winogrande", "y_hellaswag") else [])
            for kind in kinds:
                rows, dd = estimator_rows(d, out, kind, surf[(sk, out)], centers[sk], rng, sample, sk)
                for r in rows:
                    r["obs_n_at_floor"] = int((d[out] <= rc.FLOOR_LOGIT + 1e-9).sum())
                allrows += rows
                if kind == "ols":
                    co_rows += compute_only_rows(dd, out, surf[(sk, out)], "ols", centers[sk], rng, sample, sk, rel) \
                        if out == "y_hellaswag" else []
                    if out in ("y_hellaswag", "y_core"):
                        X, names, _, _ = inf.build_X(dd, ["c"], None)
                        fit = inf.LinFit(X, dd[out].to_numpy(), inf.cluster_ids(dd.developer), names.index("c"))
                        Fo = inf.obs_features(dd.n, dd.d, centers[sk])
                        bench_draws[f"{sk}|{sample}|{out}"] = fit.h @ (Fo @ surf[(sk, out)]["ols"]["draws"])
                        bench_draws[f"{sk}|{sample}|{out}|point"] = np.array([fit.h @ (Fo @ surf[(sk, out)]["ols"]["b"])])
            rc.log(f"  {sample} / {out} done ({time.time() - t0:.0f}s)")
    M = pd.DataFrame(allrows)
    # reviewer addition: conservative few-cluster p-value = the largest of three procedures that all include the
    # benchmark's uncertainty (CV3 with t(G-1); WCR studentized by CR1; WCR studentized by CV3).  Rejecting only when
    # all three reject is valid if any one of them is; it keeps the p-value coherent with the CV3 s.e. in Table E1.
    M["p_conservative"] = M[["p_t_cv3", "p_wcr_combined", "p_wcr_cv3_combined"]].max(axis=1, skipna=True)
    M["p_min_of_three"] = M[["p_t_cv3", "p_wcr_combined", "p_wcr_cv3_combined"]].min(axis=1, skipna=True)
    M.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_matched_long.csv"), index=False)
    CO = pd.DataFrame(co_rows)
    CO.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_compute_only.csv"), index=False)
    np.savez(os.path.join(rc.PROC, "bench_thetaC_draws.npz"), **{k.replace("|", "__"): v for k, v in bench_draws.items()})

    # ---- item 2: Table 8 reconciliation on the legacy benchmark (the design Table 8 used)
    d_leg = obs_all[hull["legacy"]].dropna(subset=["y_hellaswag"]).copy()
    R = reconcile(d_leg, "y_hellaswag", surf[("legacy", "y_hellaswag")], centers["legacy"], rng)
    rc.log(f"reconciliation done ({time.time() - t0:.0f}s)")

    # ---- headline json
    fam_hull = {k: obs_all[hull[k]].family.value_counts() for k in hull}
    H = dict(runtime_sec=time.time() - t0, B_wild=rc.B_WILD, B_surf=rc.B_SURF, B_tobit=rc.B_TOBIT,
             supports={k: dict(models=int(hull[k].sum()), families=int(obs_all[hull[k]].family.nunique()),
                               developers=int(obs_all[hull[k]].developer.nunique()),
                               families_ge2=int((fam_hull[k] >= 2).sum()), models_in_families_ge2=int(fam_hull[k][fam_hull[k] >= 2].sum()),
                               singleton_families=int((fam_hull[k] == 1).sum()),
                               max_N=float(obs_all[hull[k]].N.max()), max_D=float(obs_all[hull[k]].D.max()),
                               date_min=str(obs_all[hull[k]].date.min())[:10], date_max=str(obs_all[hull[k]].date.max())[:10])
                       for k in hull},
             n_exp_runs={k: int(len(v)) for k, v in E.items()},
             reliability=dict(sigma_hw=rel["sigma_hw"], sigma_hw_hi=rel["sigma_hw_hi"], sigma_m4=rel["sigma_m4"],
                              sd_within_hw=rel["sd_within_hw"], comparisons=rel["comps"]),
             m4_inputs=dict(obs_panel_rows=int(len(df)), main=int(df.main.sum())))
    rc.dump_json(H, "matched_headline.json")
    rc.log(f"step_matched finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
