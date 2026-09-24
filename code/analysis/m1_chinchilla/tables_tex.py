"""Paper-ready LaTeX fragments (booktabs, threeparttable, AER style) built from the module's CSV outputs."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import TABLES
from texutil import num, se, write_table

R = lambda f: pd.read_csv(os.path.join(TABLES, f))
EST_COLS = ["huber", "lad_log", "gauss_log", "nls_lev", "vpnls", "norm"]
EST_HEAD = ["Huber", "LAD", "Gaussian", "NLS", "VPNLS", "KMW"]


def _sci(x, p=2):
    m, e = f"{x:.{p}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _pp(p):
    if p is None or not np.isfinite(p):
        return ""
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- T1 horse race
def t_horse():
    hr = R("m1_chinchilla_horse_race.csv")
    params = [("E", "$E$", 3), ("A", "$A$", 0), ("B", "$B$", 0), ("alpha", r"$\alpha$", 3), ("beta", r"$\beta$", 3),
              ("a", r"$a=\beta/(\alpha+\beta)$", 3), ("gamma", r"$\gamma=\alpha\beta/(\alpha+\beta)$", 3),
              ("sigma_star", r"$\sigma^*=2/(2+\alpha+\beta)$", 3), ("Mstar_1e+21", r"$M^*$ at $C=10^{21}$", 1),
              ("Mstar_5.76e+23", r"$M^*$ at $C=5.76\times10^{23}$", 1), ("Mstar_1e+26", r"$M^*$ at $C=10^{26}$", 1),
              ("w_chin70b", r"$w$ at Chinchilla-70B", 2)]
    body = []
    # robust (IQR/1.349) bootstrap s.e. for the heavy-tailed objects, computed from the stored draws
    from common import DERIVED_KEYS, PROC
    import boot as _b
    cols = ["th0", "th1", "th2", "alpha", "beta"] + DERIVED_KEYS
    heavy = {"A", "B", "Mstar_1e+21", "Mstar_5.76e+23", "Mstar_1e+26", "w_chin70b"}

    def rob(n, e, sch, k):
        M = np.load(os.path.join(PROC, f"boot_chinchilla_n{n}_{e}_{sch}.npy"))
        return float(_b.rse(M[:, cols.index(k)]))

    for n, lab in ((240, "Panel A. Besiroglu sample (5 highest-loss runs dropped), $n=240$"),
                   (245, "Panel B. All digitized runs, $n=245$")):
        body.append(("PANEL", lab, 7))
        sub = hr[hr.n == n].set_index("estimator")
        for k, name, p in params:
            body.append([name + (r"$^\dagger$" if k in heavy else "")] + [num(sub.loc[e, k], p) for e in EST_COLS])
            if k in heavy:
                body.append([""] + [f"{se(rob(n, e, 'pairs', k), p)} {se(rob(n, e, 'cluster', k), p, '[]')}" for e in EST_COLS])
            else:
                body.append([""] + [f"{se(sub.loc[e, f'se_pairs_{k}'], p)} {se(sub.loc[e, f'se_cluster_{k}'], p, '[]')}"
                                    for e in EST_COLS])
        body.append(["Objective at optimum"] + [f"{sub.loc[e, 'objective']:.4g}" for e in EST_COLS])
        body.append([r"Scaled condition number of $J$"] + [num(sub.loc[e, "cond_scaled"], 0) for e in EST_COLS])
        if n == 240:
            body.append("MID")
    share240 = hr[(hr.n == 240) & (hr.estimator == "huber")].share_linear_huber.iloc[0]
    share245 = hr[(hr.n == 245) & (hr.estimator == "huber")].share_linear_huber.iloc[0]
    ed = R("m1_chinchilla_estimator_diffs.csv")
    ed = ed[(ed.n == 240) & (ed.scheme == "pairs")].set_index(["estimator", "param"])
    pdiff = ("Paired bootstrap of estimator differences ($n=240$, same draws, s.d.\\ of the difference): "
             + "; ".join(f"{lab} $-$ Huber: $\\Delta\\beta={ed.loc[(e, 'beta')]['diff']:.3f}$ ({ed.loc[(e, 'beta')].sd_diff:.3f}), "
                         f"$\\Delta\\sigma^*={ed.loc[(e, 'sigma_star')]['diff']:.3f}$ ({ed.loc[(e, 'sigma_star')].sd_diff:.3f}), "
                         f"$\\Delta M^*(5.76\\times10^{{23}})={ed.loc[(e, 'Mstar_5.76e+23')]['diff']:.1f}$ "
                         f"({ed.loc[(e, 'Mstar_5.76e+23')].sd_diff:.1f})"
                         for e, lab in (("lad_log", "LAD"), ("gauss_log", "Gaussian"), ("nls_lev", "NLS levels"))) + ". ")
    notes = (r"Estimates of $L=E+AN^{-\alpha}+BD^{-\beta}$ on Epoch's digitization of the Chinchilla runs "
             r"(\citealt{besiroglu2024chinchilla}; $N$ = parameters, $D=C/(6N)$ tokens, $L$ in nats/token). "
             r"Columns: (1) Hoffmann--Besiroglu Huber loss ($\delta=10^{-3}$) on log loss, log-sum-exp parameterization, "
             r"4,500-point grid of starts; (2) least absolute deviations on log loss; (3) Gaussian NLS on log loss; "
             r"(4) NLS in levels; (5) variable projection in levels (given $\alpha,\beta$, $(E,A,B)$ are concentrated out "
             r"by non-negative least squares; same objective as (4)); (6) Huber objective with inputs normalized at "
             r"their geometric means (Klump--McAdam--Willman), a reparameterization of (1). "
             r"Bootstrap standard errors (400 draws, warm-started) in parentheses (pairs) and brackets (cluster bootstrap "
             r"over the 9 reconstructed IsoFLOP budgets; few clusters, cf. \citealt{cameron2008bootstrap}). "
             r"$M^*=D^*/N^*$ is the cost-minimizing tokens-per-parameter ratio; $w=\varepsilon_N/\varepsilon_D$ at "
             r"$(N,D)=(70\text{B},1.4\text{T})$ equals one if Chinchilla-70B is compute-optimal. "
             f"Share of Huber residuals in the linear region ($|r|>\\delta$): {share240:.3f} ($n=240$), {share245:.3f} ($n=245$). "
             + pdiff +
             r"$^\dagger$Heavy-tailed bootstrap distributions: robust standard errors (interquartile range/1.349) are reported; "
             r"bootstrap SDs and percentile intervals are in the CSV. "
             r"Scaled condition number: Belsley condition number of the Jacobian of $\ln\hat L$ (columns scaled to unit "
             r"length), in the raw parameters for (1)--(5) and in the normalized parameters for (6).")
    write_table("m1_chinchilla_horse_race", "Estimator Horse Race on the Chinchilla Data", "tab:m1_horse_race",
                "l" + "c" * 6, [[""] + [f"({i + 1})" for i in range(6)], [""] + EST_HEAD], body, notes, size="\\scriptsize")


# ----------------------------------------------------------------------------- T2 selection
def t_selection():
    kt = R("m1_chinchilla_selection_k.csv")
    ru = R("m1_chinchilla_selection_rules.csv")
    sw = R("m1_chinchilla_selection_swing.csv").set_index("param")
    body = [("PANEL", "Panel A. Drop the $k$ highest-loss runs (truncation on the outcome); Huber-LSE", 8)]
    for k in (0, 1, 2, 3, 4, 5, 6, 8, 10, 15):
        r = kt[kt.k == k].iloc[0]
        body.append([str(k), str(int(r.n)), num(r.Lmax_kept, 2), f"{num(r.huber_alpha)} {se(r.huber_se_alpha)}",
                     f"{num(r.huber_beta)} {se(r.huber_se_beta)}", f"{num(r.huber_a)} {se(r.huber_se_a)}",
                     f"{num(r['huber_Mstar_1e+21'], 1)} {se(r['huber_se_Mstar_1e+21'], 1)}", num(r.huber_E)])
    body.append(("PANEL", "Panel B. Alternative exclusion rules; Huber-LSE", 8))
    for _, r in ru.iterrows():
        if "Gaussian" in r.rule or "Hausman" in r.rule:
            continue
        body.append([r.rule.replace("<=", "LEQ").replace("<", "$<$").replace("LEQ", r"$\le$")
                     .replace("|residual|", r"$|$residual$|$"), str(int(r.n)), "",
                     f"{num(r.alpha)} {se(r.se_alpha)}", f"{num(r.beta)} {se(r.se_beta)}", f"{num(r.a)} {se(r.se_a)}",
                     f"{num(r['Mstar_1e+21'], 1)} {se(r['se_Mstar_1e+21'], 1)}", num(r.E)])
    body.append(("PANEL", "Panel C. Gaussian likelihood and the Hausman--Wise truncated-regression correction", 8))
    for _, r in ru.iterrows():
        if not ("Gaussian" in r.rule or "Hausman" in r.rule):
            continue
        body.append([r.rule.replace("<", "$<$").replace("Lbar", r"$\bar L$"), str(int(r.n)), "", num(r.alpha), num(r.beta),
                     num(r.a), num(r["Mstar_1e+21"], 1), num(r.E)])
    b = sw.loc["beta"]
    a = sw.loc["a"]
    notes = (r"Huber-LSE ($\delta=10^{-3}$) estimates; pairs-bootstrap standard errors (200 draws) in parentheses. "
             r"Panel A drops the $k$ runs with the highest loss from the 245 digitized runs ($k=5$ is the "
             r"\citet{besiroglu2024chinchilla} sample); ``Max $L$'' is the largest retained loss. Panel B: selection on "
             r"regressors ($D/N$, budget) is ignorable under correct specification, selection on the outcome is not "
             r"\citep{hausman1977social}. Panel C: Gaussian NLS on log loss, and the truncated-normal MLE that conditions "
             r"on $L<\bar L$ ($\bar L$ = 5th-highest loss, 3.447). Paired bootstrap of the $k=0\to5$ swing "
             r"(400 draws of the 245 runs; the rule $L<\bar L$ applied inside each draw): "
             f"$\\Delta\\beta={b['diff']:.3f}$ (s.e. {b.se_diff:.3f}; 95\\% interval [{b.lo_diff:.3f}, {b.hi_diff:.3f}]), "
             f"$\\Delta a={a['diff']:.3f}$ (s.e. {a.se_diff:.3f}).")
    write_table("m1_chinchilla_selection", "Selection on the Outcome: Dropping High-Loss Runs", "tab:m1_selection",
                "lccccccc", [["$k$ / rule", "$n$", "Max $L$", r"$\alpha$", r"$\beta$", "$a$", r"$M^*(10^{21})$", "$E$"]],
                body, notes, size="\\scriptsize")


# ----------------------------------------------------------------------------- T3 duality
def t_duality():
    ob = R("m1_chinchilla_duality_objects.csv")
    te = R("m1_chinchilla_duality_tests.csv")
    rp = R("m1_chinchilla_revealed_pref_70b.csv")
    sy = R("m1_chinchilla_system.csv")
    slr = R("m1_chinchilla_system_lr.csv")
    sls = R("m1_chinchilla_system_lr_smallsample.csv")
    names = {"a": r"Path slope $a$", "lnNstar_ref": r"Path level $\ln N^*(\bar C)$", "gamma": r"Frontier exponent $\gamma$",
             "Lstar_ref": r"Frontier level $L^*(\bar C)$"}
    bc = R("m1_chinchilla_duality_bootcal.csv")
    cf = R("m1_chinchilla_duality_classicalF.csv")
    body = [("PANEL", r"Panel A. Dual objects, $n=240$ (standard errors: within-budget stratified bootstrap, 300 draws)", 7)]
    o = ob[ob.dataset == "Chinchilla n=240"].set_index("object")
    for k in ("a", "lnNstar_ref", "gamma", "Lstar_ref"):
        r = o.loc[k]
        body.append([names[k], f"{num(r.A2_A1, 4)} {se(r.A2_A1_se, 4)}", num(r.A2_A1_Efixed, 4),
                     f"{num(r.A3_analytic, 4)} {se(r.A3_analytic_se, 4)}", f"{num(r.A3_design, 4)} {se(r.A3_design_se, 4)}", "", ""])
    body.append(("PANEL", r"Panel B. Wald tests of the cross-equation restrictions: statistic (p-value)", 7))
    body.append([r"\textit{Null technology}", r"\textit{Path} ($\chi^2_2$)", r"\textit{Path}, boot.\ $p$",
                 r"\textit{Frontier} ($\chi^2_2$)", r"\textit{All} ($\chi^2_4$)", r"$a_{A2}-a_0$", r"$t$ (OLS)"])
    for ds, bs in (("Chinchilla n=240", "stratified pairs"), ("Chinchilla n=240", "wild, fixed design"),
                   ("Chinchilla n=245", "stratified pairs")):
        body.append([f"\\textit{{{ds.replace('Chinchilla ', '$')}$, {bs} bootstrap}}", "", "", "", "", "", ""])
        for _, r in te[(te.dataset == ds) & (te.bootstrap == bs) & (te["cov"] == "robust")].iterrows():
            q = bc[(bc.dataset == ds) & (bc.bootstrap == bs) & (bc["null"] == r.null)]
            pb = "/".join(_pp(q[q["cov"] == c].p_path_boot.iloc[0]) for c in ("robust", "sd")) if len(q) else ""
            body.append([r"\quad " + r.null, f"{num(r.W_path, 1)} ({_pp(r.p_path)})", pb,
                         f"{num(r.W_front, 1)} ({_pp(r.p_front)})",
                         f"{num(r.W_all, 1)} ({_pp(r.p_all)})", num(r.a_A2 - r.a_null, 3),
                         f"{num(r.t_a_ols, 2)} ({_pp(r.p_a_ols)})"])
    c240 = cf[cf.dataset == "Chinchilla n=240"].set_index("null")
    fstr = "; ".join(f"{k.split(',')[0]} $F={c240.loc[k].F:.2f}$ ($p={c240.loc[k].p_F:.3f}$)" for k in c240.index)
    notes = (r"Approach 2 (conditional factor demand): per-budget parabola of $L$ in $\ln N$ on the IsoFLOP-profile runs, "
             r"argmins regressed on $\ln(C/6)$. Approach 1 (inverse cost function): $L^*_b=E+K(C_b/6)^{-\gamma}$ fitted "
             r"by NLS to the 9 per-budget minima (column 2 fixes $E$ at the Approach-3 value). Approach 3: Huber-LSE primal. "
             r"``Analytic'': closed forms $a=\beta/(\alpha+\beta)$, $\gamma=\alpha\beta/(\alpha+\beta)$, $N^*=G(C/6)^a$, "
             r"$L^*=E+K(C/6)^{-\gamma}$. ``Design'': the Approach-2/1 procedure applied to the Approach-3 fitted values at "
             r"the observed design (removes the finite-grid bias of parabola fits, \citealt{czech2026problems}). "
             r"$\bar C$ = geometric mean of the IsoFLOP budgets. Wald statistics use the bootstrap covariance of the "
             r"difference; for published parameter sets \citep{hoffmann2022training,besiroglu2024chinchilla} only the "
             r"Approach-2/1 side is random. Path = $(a,\ln N^*(\bar C))$; frontier = $(\gamma,L^*(\bar C))$. "
             r"Wald covariances are outlier-robust (IQR scales, Spearman correlations) because parabola argmins in "
             r"resampled designs are heavy-tailed; SD-based versions are in the CSV. Main scheme: within-budget stratified pairs "
             r"bootstrap (300 draws); the fixed-design wild bootstrap (400 Rademacher draws on the Approach-3 log residuals) "
             r"is a sensitivity check (it flips the sign of the gross outliers at the $10^{19}$ budget). Last two columns: "
             r"difference between the Approach-2 path slope and the null's slope, and its $t$-statistic using the classical "
             r"OLS standard error of the 9-point path regression (7 d.f.; it treats the null slope as known). "
             r"``Boot.\ $p$'': bootstrap-calibrated $p$-value of the path statistic (critical value from the bootstrap "
             r"distribution of the same quadratic form recentred at the estimate, instead of $\chi^2_2$), with the robust / "
             r"SD covariance; the $\chi^2$ $p$-values depend strongly on the covariance choice because the resampled "
             r"argmins are heavy-tailed, the calibrated ones do not. Bootstrap-free benchmark (fixed design, "
             r"iid normal argmin errors): regressing the 9 argmin gaps on $[1,\ln C_b]$ and testing both coefficients, "
             + fstr + r" ($F_{2,7}$).")
    write_table("m1_chinchilla_duality", "Duality: Approaches 1, 2 and 3 as Cost Function, Factor Demand and Primal",
                "tab:m1_duality", "lcccccc",
                [["", "A2/A1", r"A1, $E$ fixed", "A3 analytic", "A3 design", "", ""]], body, notes, size="\\scriptsize")
    # ---- T4 revealed preference + system
    body = [("PANEL", r"Panel A. Revealed-preference test at Chinchilla-70B ($N=70$B, $D=1.4$T): $w=\varepsilon_N/\varepsilon_D$", 4)]
    for _, r in rp.iterrows():
        if np.isfinite(r.w):
            ci = f"[{num(r.lo, 2)}, {num(r.hi, 2)}]" if np.isfinite(r.lo) else ""
            body.append([r.technology, num(r.w, 3), se(r.se, 3) + " " + ci, num(r.Mstar_at_chin70b_C, 1)])
        else:
            body.append([r.technology.replace("ln N_70B - ln N*_A2(C_70B)", r"$\ln N_{70B}-\ln N^*_{A2}(C_{70B})$"),
                         num(r.lnN_gap, 3), "", num(r.Mstar_at_chin70b_C, 1)])
    body.append(("PANEL", "Panel B. System estimator: loss equation + IsoFLOP-argmin equation (cross-equation restricted)", 4))
    for _, r in sy.iterrows():
        nlab = r.dataset.replace("chinchilla_", "").replace("n", "$n=") + "$"
        body.append([f"{nlab}, {r.likelihood} loss equation", "", "", ""])
        body.append([r"\quad $\alpha$, $\beta$", f"{num(r.alpha)} {se(r.se_alpha)}", f"{num(r.beta)} {se(r.se_beta)}", ""])
        body.append([r"\quad $a$, $\sigma^*$", f"{num(r.a)} {se(r.se_a)}", f"{num(r.sigma_star)} {se(r.se_sigma_star)}", ""])
        body.append([r"\quad $M^*(10^{21})$, $M^*(5.76\times10^{23})$", f"{num(r['Mstar_1e+21'], 1)} {se(r['se_Mstar_1e+21'], 1)}",
                     f"{num(r['Mstar_5.76e+23'], 1)} {se(r['se_Mstar_5.76e+23'], 1)}", ""])
        q = slr[(slr.dataset == r.dataset) & (slr.likelihood == r.likelihood)].iloc[0]
        body.append([r"\quad Nested LR test of the 2 restrictions ($\chi^2_2$)", f"{num(q.LR, 2)}", f"p = {_pp(q.LR_p)}", ""])
        qs = sls[(sls.dataset == r.dataset) & (sls.likelihood == r.likelihood)].iloc[0]
        body.append([r"\quad Small-sample ($F$-form) calibration", f"{num(qs.F_approx, 2)}",
                     f"p = {_pp(qs.p_F_approx)}", ""])
    h = R("m1_chinchilla_horse_race.csv")
    h = h[(h.n == 240) & (h.estimator == "huber")].iloc[0]
    notes = (r"Panel A: $w$ at Chinchilla-70B under each technology; $w=1$ iff the chosen model is cost-minimizing "
             r"(training-compute minimization, no inference demand). Published parameter sets have no usable standard errors. "
             r"For the refits: bootstrap s.e. and percentile 95\% interval (within-budget stratified, 300 draws); for the "
             f"$n=240$ Huber refit the pairs-bootstrap interval is [{h.lo_pairs_w_chin70b:.2f}, {h.hi_pairs_w_chin70b:.2f}] "
             f"and the 9-cluster interval [{h.lo_cluster_w_chin70b:.2f}, {h.hi_cluster_w_chin70b:.2f}] "
             r"(Table \ref{tab:m1_horse_race}). "
             r"Last column: $M^*=D^*/N^*$ at Chinchilla's compute ($5.88\times10^{23}$ FLOP); Chinchilla-70B has $M=20$. "
             r"The A2 rows report the log gap between 70B and the Approach-2 path extrapolated to Chinchilla's compute. "
             r"Panel B: concentrated quasi-likelihood system in the spirit of \citet{leonledesma2010identifying}: "
             r"loss equation (Laplace or Gaussian errors in $\ln L$) plus the per-budget parabola argmins, predicted "
             r"design-consistently from the same $(E,A,B,\alpha,\beta)$. The nested LR test frees a level shift and a slope "
             r"shift of the argmin equation relative to the primal's prediction (2 restrictions). With only 9 argmins the "
             r"$\chi^2_2$ approximation is generous; the $F$-form calibration $F=(e^{LR/n_2}-1)(n_2-2)/2\sim F_{2,n_2-2}$ "
             r"attributes the whole statistic to the argmin equation (an approximation).")
    write_table("m1_chinchilla_revealed_system", "Revealed-Preference Test at Chinchilla-70B and System Estimates",
                "tab:m1_revealed_system", "lccc", [["Technology / object", "Estimate", "S.e. [95\\% CI]", r"$M^*$ at $C_{70B}$"]],
                body, notes, size="\\scriptsize")


# ----------------------------------------------------------------------------- T5 identification
def t_ident():
    fd = R("m1_chinchilla_fdep.csv")
    vd = R("m1_chinchilla_design_variance.csv").set_index("sample")
    pr = R("m1_chinchilla_profile_sigma.csv")
    body = []
    for _, r in fd.iterrows():
        v = vd.loc[r["sample"]]
        body.append([r["sample"].replace("|dlnN|<=", r"$|\Delta\ln N|\le$ ").replace("full (n=240)", "Full design")
                     .replace("on-path", "On-path"), str(int(r.n)), num(v.share_trans, 3),
                     _sci(r.svn3), _sci(r.svn4), _sci(r.svn5), num(r.cond_norm, 0),
                     f"{num(r.alpha)} {se(r.se_alpha)}", f"{num(r.beta)} {se(r.se_beta)}", f"{num(r.a)} {se(r.se_a)}",
                     f"{num(r.gamma)} {se(r.se_gamma)}"])
    # size-matched control (review): random subsamples of the full design with the band's n (review_checks.py)
    sc = R("m1_chinchilla_fdep_sizecontrol.csv")
    c = sc[sc["sample"].str.startswith("median")].iloc[0]
    body.append([f"Random subsamples, same $n$ (median of {len(sc) - 1})", str(int(c.n)), num(c.share_trans, 3),
                 _sci(c.svn3), _sci(c.svn4), _sci(c.svn5), num(c.cond_norm, 0), se(c.se_alpha), se(c.se_beta),
                 se(c.se_a), se(c.se_gamma)])
    ctrl_prof = (f"random full-design subsamples of the same size (n = {int(c.n)}): {sc[~sc['sample'].str.startswith('median')].profile_LR_max.min():.1f}"
                 f"--{sc[~sc['sample'].str.startswith('median')].profile_LR_max.max():.1f}")
    rng = []
    for (s, o), g in pr.groupby(["sample", "objective"]):
        if o != "gauss":
            continue
        inside = g[g.LR <= 3.84].sigma_star
        pretty = s.replace("full (n=240)", "full design ($n=240$)").replace("on-path band |dlnN|<=0.15", r"on-path band $|\Delta\ln N|\le 0.15$ ($n=41$)")
        rng.append(f"{pretty}: {g.LR.max():.1f} (95\\% set for $\\sigma^*$: [{inside.min():.2f}, {inside.max():.2f}] within the grid [0.50, 0.95])")
    notes = (r"On-path samples: runs with $|\ln N-\ln N^*(C)|\le h$ around the reference expansion path (Huber-LSE, "
             r"$n=240$), and the nine Approach-2 argmins used as pseudo-observations. ``Transverse share'': share of the "
             r"variance of centred $(\ln N,\ln D)$ along $(\alpha,-\beta)$, the only direction in which $\ln R$ has "
             r"curvature (the Hessian of $\ln R$ in $(\ln N,\ln D)$ is rank one with null direction $(\beta,\alpha)$, the "
             r"expansion path). $s_3$--$s_5$: the three smallest singular values of the Jacobian of $\ln\hat L$ "
             r"with respect to the KMW-normalized parameters $(\ln a',\ln b',\ln E,\alpha,\beta)$, evaluated at the "
             r"full-sample estimate and divided by $\sqrt n$; cond.: their condition number. On an exactly optimal path the "
             r"loss equation depends on $(N,D)$ only through $C$, so the Jacobian has rank 3 and two singular values "
             r"vanish; in the on-path samples the two that collapse are $s_3$ and $s_5$ ($s_4$, the near-collinearity of "
             r"$E$ with the level parameters, is small in every design). Estimates: Huber-LSE on the "
             r"subsample (75 starting values), pairs-bootstrap s.e. (200 draws). The last row is a sample-size control: "
             r"random subsamples of the full design with the band's $n$ (100 bootstrap draws each; s.e. only), which "
             r"separates the loss of transverse variation from the loss of observations. Maximum Gaussian profile LR "
             r"statistic over $\sigma^*\in[0.50,0.95]$ in $L=E+(AN^{-a_1}+BD^{-b_1})^\kappa$: " + "; ".join(rng) +
             "; " + ctrl_prof + ".")
    write_table("m1_chinchilla_identification", "Functional Dependence: What On-Path Data Identify", "tab:m1_ident",
                "lcccccccccc", [["Sample", "$n$", "Transv. share", "$s_3/\\sqrt n$", "$s_4/\\sqrt n$", "$s_5/\\sqrt n$",
                                 "Cond.", r"$\alpha$", r"$\beta$", "$a$", r"$\gamma$"]], body, notes, size="\\scriptsize")


# ----------------------------------------------------------------------------- T6 specification tests
def t_spec():
    ces = R("m1_chinchilla_spec_ces.csv")
    kap = R("m1_chinchilla_spec_kappa.csv")
    tl = R("m1_chinchilla_spec_translog.csv")
    body = [("PANEL", r"Panel A. CES: $H_0:\alpha=\beta$ (Huber-LSE)", 6)]
    body.append(["", r"$\hat\alpha-\hat\beta$", "s.e.", "$z$ (p)", r"Restricted $\rho$ ($\sigma$)", r"Restricted $M^*(10^{21})$"])
    for _, r in ces.iterrows():
        body.append([f"$n={r['sample'][1:]}$, {r.scheme} bootstrap", num(r["diff"], 4), num(r.se_diff, 4),
                     f"{num(r.z, 2)} ({_pp(r.p)})", f"{num(r.ces_rho, 3)} ({num(r.ces_sigma, 3)})", num(r.ces_Mstar, 1)])
    body.append(("PANEL", r"Panel B. Outer exponent: $L=E+(AN^{-a_1}+BD^{-b_1})^\kappa$, $H_0:\kappa=1$", 6))
    body.append(["", r"$\hat\kappa$ (Huber)", "s.e. [95\\% CI]", "$z$ (p)", r"$\sigma^*$ ($\kappa$ free)", "Gaussian LR (p)"])
    for _, r in kap.iterrows():
        body.append([f"$n={r['sample'][1:]}$, {r.scheme} bootstrap", num(r.kappa_huber, 3),
                     f"{num(r.se_kappa, 3)} [{num(r.ci_lo, 2)}, {num(r.ci_hi, 2)}]", f"{num(r.z_kappa1, 2)} ({_pp(r.p_kappa1)})",
                     f"{num(r.sigma_star_huber, 3)} {se(r.se_sigma_star_gen, 3)}", f"{num(r.LR_gauss, 1)} ({_pp(r.p_LR_gauss)})"])
    body.append(("PANEL", r"Panel C. Rank-one curvature: translog of $\ln(L-E)$, $E$ fixed at the Huber estimate", 6))
    body.append(["", r"$b_{ND}$ ($t$)", r"$\tau=b_{NN}b_{DD}-b_{ND}^2$", "s.e.", r"$z$ vs.\ $\tau_{\text{model}}$ (p)",
                 r"$b_{ND}/\sqrt{b_{NN}b_{DD}}$"])
    for _, r in tl.iterrows():
        body.append([f"$n={r['sample'][1:]}$, {r.inference}", f"{num(r.bND, 4)} ({num(r.t_bND, 1)})",
                     _sci(r.tau), _sci(r.se_tau),
                     f"{num(r.z_tau_model, 2)} ({_pp(r.p_tau_model)})" if np.isfinite(r.get("z_tau_model", np.nan))
                     else f"{num(r.z_tau0, 2)} ({_pp(r.p_tau0)})", num(r.rank1_corr, 3)])
    notes = (r"Panel A: bootstrap Wald test of $\alpha=\beta$ (300 draws; pairs or cluster over the 9 IsoFLOP budgets) and "
             r"the restricted (homothetic CES) fit, $\sigma=1/(1+\rho)$. Panel B: the Kaplan-type nesting; $\kappa=1$ is "
             r"Chinchilla; $\sigma^*=2/(2+a_1+b_1)$ depends only on the inner exponents (isoquants). Bootstrap draws are "
             r"warm-started; the Gaussian LR compares Gaussian NLS fits with and without the restriction. Panel C: OLS of "
             r"$z=\ln(L-E)$ on centred $(n,d)=(\ln N,\ln D)$, $\tfrac12 n^2$, $\tfrac12 d^2$, $nd$. Chinchilla implies a "
             r"rank-one Hessian of $\ln R$ at every point, hence $\tau=0$, $b_{ND}<0$ and $b_{ND}/\sqrt{b_{NN}b_{DD}}=-1$; "
             r"$\tau_{\text{model}}$ is $\tau$ computed on the fitted values of the Huber estimate at the same design "
             r"(approximation error of the translog). Bootstrap rows re-estimate $E$ in every draw. Sensitivity rows fix $E$ "
             r"at alternative values (the test is sensitive to $E$).")
    write_table("m1_chinchilla_spec_tests", "Specification Tests: CES, Outer Exponent, and Rank-One Curvature",
                "tab:m1_spec", "lccccc", [["", "(1)", "(2)", "(3)", "(4)", "(5)"]], body, notes, size="\\scriptsize")


# ----------------------------------------------------------------------------- T7 labs
def t_labs():
    lt = R("m1_chinchilla_labs_technology.csv")
    te = R("m1_chinchilla_labs_duality_tests.csv")
    sy = R("m1_chinchilla_labs_system.csv")
    exps = list(dict.fromkeys(lt.experiment))
    heads = [lt[lt.experiment == e].label.iloc[0].replace("Marin 2026-03, ", "Marin ").replace("Meta ", "").replace(" IsoFLOPs", "")
             for e in exps]
    a3 = {e: lt[(lt.experiment == e) & lt.approach.str.startswith("A3")].iloc[0] for e in exps}
    a2 = {e: lt[(lt.experiment == e) & lt.approach.str.startswith("A2")].iloc[0] for e in exps}
    body = [("PANEL", "Panel A. Primal (Approach 3, Huber-LSE on IsoFLOP runs); cluster-bootstrap s.e.", len(exps) + 1)]
    body.append(["Runs (budgets)"] + [f"{int(a3[e].n)} ({int(a3[e].n_budgets)})" for e in exps])
    for k, nm, p in (("E", "$E$", 3), ("alpha", r"$\alpha$", 3), ("beta", r"$\beta$", 3), ("a", "$a$", 3),
                     ("gamma", r"$\gamma$", 3), ("sigma_star", r"$\sigma^*$", 3), ("Mstar_1e+21", r"$M^*(10^{21})$", 1)):
        body.append([nm] + [num(a3[e][k], p) for e in exps])
        body.append([""] + [se(a3[e][f"se_cluster_{k}"], p) for e in exps])
    body.append([r"$M^*(3.8\times10^{25})$"] + [num(a3[e]["Mstar_3.8e25"], 1) for e in exps])
    body.append(("PANEL", r"Panel B. Approach 2 argmins + Approach 1 frontier; $\alpha,\beta$ via $\kappa=1$; stratified s.e.", len(exps) + 1))
    for k, nm, p in (("a", "$a$", 3), ("gamma", r"$\gamma$", 3), ("alpha", r"$\alpha=\gamma/a$", 3),
                     ("beta", r"$\beta=\gamma/(1-a)$", 3)):
        body.append([nm] + [num(a2[e][k], p) for e in exps])
        body.append([""] + [se(a2[e][f"rse_strat_{k}"], p) for e in exps])
    body.append([r"$M^*(3.8\times10^{25})$"] + [num(a2[e]["Mstar_3.8e25"], 1) for e in exps])
    body.append(("PANEL", "Panel C. Cross-equation tests: statistic (p-value)", len(exps) + 1))
    for nul, nm in (("A3 (this fit), design-consistent", r"Wald, path ($\chi^2_2$), design"),
                    ("A3 (this fit), analytic", r"Wald, path ($\chi^2_2$), analytic")):
        row = [nm]
        for e, h in zip(exps, heads):
            q = te[(te.dataset == a3[e].label) & (te.null == nul) & (te.bootstrap == "stratified pairs") & (te["cov"] == "robust")]
            row.append(f"{num(q.W_path.iloc[0], 1)} ({_pp(q.p_path.iloc[0])})" if len(q) else "")
        body.append(row)
    bc = R("m1_chinchilla_duality_bootcal.csv")
    row = [r"Path, design: bootstrap-calibrated $p$ (robust/SD)"]
    for e in exps:
        q = bc[(bc.dataset == a3[e].label) & (bc["null"] == "A3 (this fit), design-consistent") &
               (bc.bootstrap == "stratified pairs")]
        row.append("/".join(_pp(q[q["cov"] == c].p_path_boot.iloc[0]) for c in ("robust", "sd")) if len(q) else "")
    body.append(row)
    row = [r"Slope, design: $t$ (OLS + primal s.e.)"]
    for e in exps:
        q = bc[(bc.dataset == a3[e].label) & (bc["null"] == "A3 (this fit), design-consistent") &
               (bc.bootstrap == "stratified pairs") & (bc["cov"] == "robust")]
        row.append(f"{num(q.t_a_ols_plus_a3.iloc[0], 2)} ({_pp(q.p_a_ols_plus_a3.iloc[0])})" if len(q) else "")
    body.append(row)
    slr = R("m1_chinchilla_system_lr.csv")
    sls = R("m1_chinchilla_system_lr_smallsample.csv")
    row = [r"System nested LR ($\chi^2_2$), Laplace"]
    row2 = [r"\quad small-sample ($F$-form) $p$"]
    for e in exps:
        q = slr[(slr.dataset == e) & (slr.likelihood == "laplace")]
        row.append(f"{num(q.LR.iloc[0], 1)} ({_pp(q.LR_p.iloc[0])})" if len(q) else "")
        q2 = sls[(sls.dataset == e) & (sls.likelihood == "laplace")]
        row2.append(_pp(q2.p_F_approx.iloc[0]) if len(q2) else "")
    body.append(row)
    body.append(row2)
    ll = a2[exps[0]]
    ob = R("m1_chinchilla_labs_duality_objects.csv")
    se_ols_ll = float(ob[(ob.dataset == ll.label) & (ob.object == "a")].A2_se_ols.iloc[0])
    ncl = [int(a3[e].n_budgets) for e in exps]
    notes = (r"IsoFLOP compilations from \citet{czech2026problems}: Llama 3 digitized from \citet{grattafiori2024llama} "
             r"Fig.~2 ($N=C/(6D)$; loss units not stated), Marin 2026-03 Llama-2-architecture ladders on three corpora "
             r"(Paloma loss; the nominal budgets differ from $6ND$ by $-7\%$ to $+35\%$), and the \citet{li2025misfitting} "
             r"FineWeb/C4 sweep (checkpoint-interpolated IsoFLOPs). "
             r"Loss levels ($E$) are not comparable across columns; exponents are. Panel A s.e.: cluster bootstrap over "
             f"budgets (200 draws; {min(ncl)}--{max(ncl)} clusters). Panel B s.e.: robust (IQR/1.349) within-budget "
             r"stratified bootstrap (the classical OLS s.e. of the Llama 3 path slope from the scatter of its 10 argmins is "
             f"{se_ols_ll:.3f}; the stratified bootstrap understates it). "
             r"Panel C: the Wald $p$-values use $\chi^2_2$ critical values and the robust covariance; the "
             r"bootstrap-calibrated $p$-values and the slope $t$-statistic that adds the primal's own bootstrap s.e. to the "
             r"OLS s.e. of the Approach-2 slope are more conservative. "
             r"$M^*(3.8\times10^{25})$ extrapolates 3.5 (Llama 3) to 5 (Marin) orders of magnitude beyond the largest budget "
             r"and is shown only to illustrate how far the primal and dual extrapolations diverge. "
             r"Our Approach-2 fit reproduces Meta's published law: $D^*=" f"{ll.meta_repro_D_coef:.4f}" r"\,C^{"
             f"{ll.meta_repro_D_exp:.4f}" r"}$ vs.\ $0.299\,C^{0.537}$.")
    write_table("m1_chinchilla_labs", "Lab-Own Technologies from IsoFLOP Experiments", "tab:m1_labs",
                "l" + "c" * len(exps), [[""] + heads], body, notes, size="\\scriptsize")


def run(log=print):
    for f in (t_horse, t_selection, t_duality, t_ident, t_spec, t_labs):
        f()
        log(f"  table {f.__name__} written")
