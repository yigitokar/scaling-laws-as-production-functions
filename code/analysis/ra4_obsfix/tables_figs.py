"""Step 5 of ra4_obsfix: appendix-ready LaTeX tables (Online Appendix E) and figures, from the CSV/JSON outputs of
the other steps (no estimation here).  AEA style: booktabs, AEA.cls `tablenotes` (not threeparttable), \\footnotesize.
Figures follow code/analysis/aer_style.py (at most three colors per panel, no twin axes)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ra4common as rc  # noqa: E402

import aer_style  # noqa: E402

F = lambda x, nd=3: rc.fmt(x, nd)
Fm = lambda x, nd=3: rc.fmt_m(x, nd)
P = lambda p: rc.fmt_p(p, rc.B_WILD)
T = lambda name: os.path.join(rc.TAB, f"{rc.PREFIX}_{name}")
S_STRICT, S_LEG, S_EXT = "strict support (20)", "legacy support (57)", "legacy support, strict surface (extrapolated)"


def J(name):
    return json.load(open(os.path.join(rc.PROC, name)))


# =========================================================================================== Table E1: design-matched
def _cells(g, par, bias_se_col="bias_se_cv3"):
    r = g[g.param == par]
    if not len(r) or not bool(r.identified.iloc[0]):
        return None
    return r.iloc[0]


def matched_pair(g, label, show_obs=True):
    c, nn, dd = _cells(g, "theta_C"), _cells(g, "theta_N"), _cells(g, "theta_D")
    def top(x, k):
        return F(x[k]) if x is not None else "n.i."
    def se(x, k):
        return f"({F(x[k])})" if x is not None and np.isfinite(x[k]) else ""
    def bias_bot(x):
        if x is None:
            return ""
        return f"({F(x.bias_se_cv3)}) [{P(x.p_conservative)}]"
    t = [label, top(c, "obs"), top(c, "bench"), top(c, "bias"), top(nn, "bias"), top(dd, "bias")]
    b = ["", se(c, "obs_se_cv3_y"), se(c, "bench_se"), bias_bot(c), bias_bot(nn), bias_bot(dd)]
    return [" & ".join(t), " & ".join(b)]


def table_E1():
    M = pd.read_csv(T("matched_long.csv"))
    H = J("matched_headline.json")
    st, lg = H["supports"]["strict"], H["supports"]["legacy"]
    est_lab = {"OLS": "OLS", "Year FE": "Year FE", "Developer FE": "Developer FE", "Family FE": "Family FE",
               "Lab x period FE": "Lab $\\times$ period FE", "Notability-propensity control": "Notability-propensity control"}
    body = []
    body.append(rc.panel_row(f"Panel A. HellaSwag, strict benchmark ({st['models']} models, "
                             f"{st['families']} families, {st['developers']} developers)", 6))
    g0 = M[(M["sample"] == S_STRICT) & (M.output == "y_hellaswag") & (M.surface_kind == "ols")]
    for e, lab in est_lab.items():
        g = g0[g0.estimator == e]
        if len(g):
            body += matched_pair(g, lab)
    body.append("\\addlinespace[3pt]")
    body.append(rc.panel_row(f"Panel B. HellaSwag, benchmark with OLMo-2 ({lg['models']} models, "
                             f"{lg['families']} families, {lg['developers']} developers)", 6))
    g0 = M[(M["sample"] == S_LEG) & (M.output == "y_hellaswag") & (M.surface_kind == "ols")]
    for e, lab in est_lab.items():
        g = g0[g0.estimator == e]
        if len(g):
            body += matched_pair(g, lab)
    body.append("\\addlinespace[3pt]")
    body.append(rc.panel_row("Panel C. OLS, other outputs and supports", 6))
    specs = [(S_EXT, "y_hellaswag", "ols", "HellaSwag, strict surface, 57"),
             (S_STRICT, "y_arc_c", "tobit", "ARC-Challenge, 20"),
             (S_LEG, "y_arc_c", "tobit", "ARC-Challenge, 57, with OLMo-2"),
             (S_STRICT, "y_winogrande", "tobit", "Winogrande, 20"),
             (S_LEG, "y_winogrande", "tobit", "Winogrande, 57, with OLMo-2")]
    for smp, out, kind, lab in specs:
        g = M[(M["sample"] == smp) & (M.output == out) & (M.surface_kind == kind) & (M.estimator == "OLS")]
        body += matched_pair(g, lab)
    hdr = [" & \\multicolumn{3}{c}{Compute only, $\\theta_C$} & $\\theta_N$ & $\\theta_D$",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}",
           " & Obs. & Exp. & Bias & Bias & Bias",
           " & (1) & (2) & (3) & (4) & (5)"]
    s = pd.read_csv(T("surfaces.csv"))
    ns = int(s[(s.surface == "strict") & (s.output == "y_hellaswag")].n_runs.iloc[0])
    nl = int(s[(s.surface == "legacy") & (s.output == "y_hellaswag")].n_runs.iloc[0])
    ncens = int(s[(s.surface == "strict") & (s.output == "y_arc_c")].n_censored.iloc[0])
    ncw = int(s[(s.surface == "strict") & (s.output == "y_winogrande")].n_censored.iloc[0])
    ni = " n.i.: not identified." if any("n.i." in r for r in body) else ""
    gsel = lambda smp: M[(M["sample"] == smp) & (M.output == "y_hellaswag") & (M.surface_kind == "ols")].G_star.dropna()
    gs, gl = (gsel(S_STRICT).min(), gsel(S_STRICT).max()), (gsel(S_LEG).min(), gsel(S_LEG).max())
    notes = (
        "Output: log-odds of above-chance accuracy. Obs.: estimator on the observed outputs of released base models. Exp.: the "
        "same estimator, with the same fixed effects and controls, on the outputs that the experimental technology produces for "
        "the same models: a quadratic in $(\\ln N,\\ln D)$ fitted to the OLMo ladder and \\citet{gadre2024language} "
        f"($N\\geq0.1$B; {ns} runs); Panel B adds the released OLMo-2 7B and 13B models ({nl} runs), as the original Table 8 did. "
        "Each panel uses the models inside the convex hull of its own runs; `strict surface, 57' evaluates the Panel A surface on "
        f"the Panel B models, {lg['models'] - st['models']} of which lie outside its support. Bias $=$ Obs.\\ $-$ Exp. Parentheses: CV3 jackknife s.e.\\ by developer (Obs.; bias: of Obs.\\ $-$ Exp., with the benchmark's s.e.\\ added "
        "in quadrature) and the design-conditional "
        f"wild-cluster bootstrap s.e.\\ of the surface ({rc.B_SURF:,} draws; Exp.). Brackets: the largest of three few-cluster "
        "$p$-values for zero bias that include the benchmark's uncertainty (CV3 with $t(G-1)$; restricted wild-cluster "
        f"bootstrap-$t$, Webb weights, {rc.B_WILD:,} draws, studentized by CR1 or by CV3). Effective numbers of clusters: {gs[0]:.1f}--{gs[1]:.1f} of "
        f"{st['developers']} in Panel A (no $p$-value is reliable there); {gl[0]:.1f}--{gl[1]:.1f} of "
        f"{lg['developers']} in Panel B.{ni} ARC-Challenge and Winogrande: Tobit surface censored at the floor "
        f"({ncens} and {ncw} of {ns} runs are at it), level matched to the observed outputs, counterfactual clipped at the same "
        "floor. Notability-propensity control (formerly `OP-style selection'): quadratic in the estimated propensity of Epoch "
        "notability; Panel A has no notable model.")
    src = ("Authors' calculations. Observational outputs: Open LLM Leaderboard v1 via \\citet{ruan2024observational} and "
           "\\citet{maiapolo2024sloth}; parameter counts from Hugging Face; release dates from \\citet{epochai2026data}. "
           "Experimental runs: \\citet{bhagia2024establishing}, \\citet{teamolmo2024olmo} (Panel B only) and "
           "\\citet{gadre2024language}.")
    rc.write_aea_table(T("tabE1_matched.tex"), "Design-Matched Comparison of Observational and Experimental Elasticities",
                       "tab:app-matched", "@{}l ccc c c@{}", hdr, body, notes, src, tabcolsep="4pt")


# =========================================================================================== Table E2: compute-only
def table_E2():
    CO = pd.read_csv(T("compute_only.csv"))
    body = []
    for smp, lab in [(S_LEG, "Panel A. HellaSwag, 57 models (benchmark including OLMo-2)"),
                     (S_STRICT, "Panel B. HellaSwag, 20 models (strictly experimental benchmark)")]:
        body.append(rc.panel_row(lab, 6))
        g = CO[CO["sample"] == smp]
        for est, nm in [("OLS, EIV (hardware-time sd (upper bound))", "OLS, EIV, $\\sigma_u=0.29$"),
                        ("Family FE, EIV (hardware-time sd (upper bound))", "Family FE, EIV, $\\sigma_u=0.29$"),
                        ("Family FE, EIV (Epoch 'Confident' class)", "Family FE, EIV, $\\sigma_u=0.67$")]:
            if smp == S_STRICT and "Confident" in est:
                continue
            for formula in (["correct", "table8"] if "Confident" in est else ["correct"]):
                r = g[(g.estimator == est) & (g.formula == formula)]
                if not len(r):
                    continue
                r = r.iloc[0]
                nm2 = nm + (", Table 8" if formula == "table8" else "")
                body.append(" & ".join([nm2, F(r.obs), F(r.bench), F(r.bias), F(r.reliability), ""]))
                body.append(" & ".join(["", f"({F(r.obs_se)})", f"({F(r.bench_se)})", f"({F(r.bias_se)}) [{P(r.p_wcr_combined)}]", "", ""]))
        for est, nm in [("Reverse regression", "Reverse regression"), ("Reverse regression, family FE", "Reverse, family FE")]:
            r = g[g.estimator == est].iloc[0]
            body.append(" & ".join([nm, F(r.obs), F(r.bench), F(r.bias), F(r.R2_forward_within),
                                    f"${Fm(r.forward)}/{Fm(r.R2_forward_within)}$"]))
            body.append(" & ".join(["", f"({F(r.obs_se)})", f"({F(r.bench_se)})", "", "", ""]))
        for est, nm in [("IV: frontier FLOP/$", "IV, frontier FLOP/\\$"), ("IV: frontier FLOP/$ + developer FE", "IV, frontier, developer FE"),
                        ("IV: own-hardware FLOP/$", "IV, own hardware")]:
            r = g[g.estimator == est]
            if not len(r) or not np.isfinite(r.iloc[0].get("obs", np.nan)):
                body.append(" & ".join([nm, "\\multicolumn{5}{l}{fewer than 8 models with the instrument}"]))
                continue
            r = r.iloc[0]
            def ar(lo, hi, typ):
                return "$(-\\infty,\\infty)$" if typ == "unbounded" else f"$[{Fm(lo, 2)},{Fm(hi, 2)}]$"
            body.append(" & ".join([nm + f" ($n={int(r.n_obs)}$)", F(r.obs), F(r.bench), F(r.bias), F(r.F_first, 2 if r.F_first < 10 else 1),
                                    ar(r.ar_lo_t, r.ar_hi_t, r.ar_type_t)]))
            body.append(" & ".join(["", "", f"({F(r.bench_se)})", "", "", ar(r.ar_lo_n, r.ar_hi_n, r.ar_type_n)]))
        body.append("\\addlinespace[3pt]")
    body = body[:-1]
    hdr = [" & Obs. & Exp. & Bias & $\\lambda$, $R^2$, $F$ & AR set; $\\hat\\theta/R^2$", " & (1) & (2) & (3) & (4) & (5)"]
    notes = (
        "Compute-only estimators of $\\theta_C$ on the design-matched samples of Table~\\ref{tab:app-matched}. EIV: slope divided "
        "by the reliability $\\lambda=1-\\sigma_u^2/\\text{Var}(\\ln C)$ (column 4), with $\\text{Var}(\\ln C)$ pooled or within "
        "family; the within-family variance is $\\sum\\tilde c_i^2/(n-F)$, net of the $F$ family means. $\\sigma_u=0.29$ is the "
        "standard deviation of $\\ln(C_{\\text{hw}}/6ND)$ across the 27 main-sample models with hardware-time records "
        "(Table~\\ref{tab:app-reliability}), an upper bound on the error in $\\ln 6ND$; $\\sigma_u=\\ln3/1.645$ is Epoch's "
        "`Confident' class, which describes the uncertainty of Epoch's own estimate, not of $6ND$. `Table 8': the "
        "within-family variance computed with an $n-1$ denominator, which is what the original Table 8 used (reliability 0.514 "
        "instead of 0.766 for the same $\\sigma_u$). Exp.\\ applies the same estimator to the experimental outputs, which are "
        "built from the same measured inputs and need no correction. Brackets: wild-cluster bootstrap $p$-values as in "
        "Table~\\ref{tab:app-matched}. Reverse regression: $1/b$ from regressing $\\ln C$ on the output. With no transmission "
        "bias, $1/b=\\hat\\theta_C/R^2$ of the forward regression, so its gap to Exp.\\ measures the output variance not "
        "explained by compute (productivity dispersion and noise), not a bias; column 5 gives the bracket. IV: $\\ln C$ "
        "instrumented by the log FLOP/s per dollar of the best data-center accelerator released six months before the model, "
        "or of the model's own training hardware (Epoch); $F$: cluster-robust first-stage statistic; Anderson--Rubin sets by "
        "inversion over $\\theta_C\\in[-3,4]$ with $t(G-1)$ critical values (second line: normal critical values); "
        "$(-\\infty,\\infty)$: the set covers the whole grid.")
    src = "Authors' calculations; see Table~\\ref{tab:app-matched}; hardware records from \\citet{epochai2026data}."
    rc.write_aea_table(T("tabE2_compute.tex"), "Compute-Only Estimators: Measurement Error, Reverse Regression and Instruments",
                       "tab:app-compute", "@{}l cccc l@{}", hdr, body, notes, src, tabcolsep="3pt")


# =========================================================================================== Table E3: inference
def table_E3():
    R = pd.read_csv(T("inference_reconcile.csv"))
    body = []
    order = [("CR1, k counts 29 family dummies (Table 8)", "CR1, all $k$ (Table 8), normal", "normal"),
             ("CR1, singleton families dropped", "CR1, singletons dropped, normal", "normal"),
             ("CR1, nested fixed effects not counted", "CR1, nested $k$, $t(18)$", "$t(18)$"),
             ("CV3 jackknife", "CV3 jackknife, $t(18)$", "$t(18)$")]
    for key, lab, ref in order:
        rn = R[(R.param == "theta_N") & R.variant.str.startswith(key)].iloc[0]
        rd = R[(R.param == "theta_D") & R.variant.str.startswith(key)].iloc[0]
        body.append(" & ".join([lab, F(rn.obs_se), F(rn.bias_se), P(rn.p), F(rd.obs_se), F(rd.bias_se), P(rd.p)]))
    body.append("\\addlinespace[2pt]")
    for key, lab in [("WCR, benchmark fixed (Table 8 protocol)", "WCR, benchmark fixed (Table 8)"),
                     ("WCR, benchmark fixed, nested k", "WCR, benchmark fixed, nested $k$"),
                     ("WCR, benchmark uncertainty included", "WCR, with benchmark draws"),
                     ("WCU, benchmark uncertainty included", "WCU, with benchmark draws"),
                     ("WCR subcluster (weights by family)", "WCR, family weights"),
                     ("WCR subcluster (weights by model)", "WCR, model weights"),
                     ("WCR studentized by CV3", "WCR, CV3-studentized, with benchmark draws")]:
        rn = R[(R.param == "theta_N") & (R.variant.str.startswith(key))].iloc[0]
        rd = R[(R.param == "theta_D") & (R.variant.str.startswith(key))].iloc[0]
        body.append(" & ".join([lab, "", "", P(rn.p), "", "", P(rd.p)]))
    body.append("\\addlinespace[2pt]")
    gn = R[(R.param == "theta_N") & R.variant.str.startswith("Effective")].iloc[0]
    gd = R[(R.param == "theta_D") & R.variant.str.startswith("Effective")].iloc[0]
    body.append(" & ".join(["Effective clusters $G^*$ (of 19)", "\\multicolumn{3}{c}{" + F(gn.value, 1) + "}",
                            "\\multicolumn{3}{c}{" + F(gd.value, 1) + "}"]))
    ln = R[(R.param == "theta_N") & R.variant.str.startswith("Leave")].iloc[0]
    ld = R[(R.param == "theta_D") & R.variant.str.startswith("Leave")].iloc[0]
    body.append(" & ".join(["Bias, one developer left out", f"\\multicolumn{{3}}{{c}}{{$[{Fm(ln.loo_min)},{Fm(ln.loo_max)}]$}}",
                            f"\\multicolumn{{3}}{{c}}{{$[{Fm(ld.loo_min)},{Fm(ld.loo_max)}]$}}"]))
    b_n, b_d = R[R.param == "theta_N"].bias.iloc[0], R[R.param == "theta_D"].bias.iloc[0]
    hdr = [f" & \\multicolumn{{3}}{{c}}{{$\\theta_N$ bias $={Fm(b_n)}$}} & \\multicolumn{{3}}{{c}}{{$\\theta_D$ bias $={Fm(b_d)}$}}",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           " & SE Obs. & SE bias & $p$ & SE Obs. & SE bias & $p$"]
    lev = pd.read_csv(T("leverage_familyFE.csv"))
    ld_ = lev[(lev.param == "theta_D") & (lev.by == "developer")].sort_values("share", ascending=False)
    fam = lev[(lev.param == "theta_D") & (lev.by == "family")].sort_values("share", ascending=False)
    dof = R[R.variant.str.startswith("dof")].iloc[0]
    bs_n = R[(R.param == "theta_N") & R.variant.str.startswith("CR1, k counts")].bench_se.iloc[0]
    bs_d = R[(R.param == "theta_D") & R.variant.str.startswith("CR1, k counts")].bench_se.iloc[0]
    sh_d = lev[(lev.param == "theta_D") & (lev.by == "developer")].share
    nzero, nsmall = int((sh_d < 1e-12).sum()), int(((sh_d >= 1e-12) & (sh_d < 1e-3)).sum())
    notes = (
        "Family fixed-effects rows of the original Table 8 (HellaSwag, 57 models, benchmark including OLMo-2). The bias and the "
        "benchmark are as published; only inference changes. SE (bias) adds the design-conditional bootstrap standard error of "
        f"the benchmark ({F(bs_n)} for $\\theta_N$, {F(bs_d)} for $\\theta_D$). The CR1 small-sample factor $\\frac{{G}}{{G-1}}\\frac{{n-1}}{{n-k}}$ "
        f"is {F(dof.value, 2)} on the variance when $k={int(dof.k_all)}$ counts the family dummies ({int(dof.singleton_families)} of the "
        f"30 families are singletons), {F(dof.value3, 2)} with singletons dropped, and {F(dof.value2, 2)} when fixed effects nested "
        "within developer clusters are not counted \\citep{cameron2015practitioner}. A bootstrap-$t$ is invariant to this constant; a "
        "normal reference is not, which is why the original table paired a $p$-value of 0.02 with a $t$-ratio of 1.6. WCR/WCU: "
        f"restricted/unrestricted wild-cluster bootstrap-$t$, developer clusters, Webb weights, {rc.B_WILD:,} draws "
        "\\citep{cameron2008bootstrap,webb2023reworking}; subcluster: weights drawn by family or by model with developer-clustered "
        "standard errors \\citep{mackinnon2018wild}; CV3-studentized: the bootstrap statistic uses the CV3 standard error of "
        "$\\text{Obs.}-\\text{Exp.}$ in the sample and in every draw. CV3: jackknife by developer \\citep{mackinnon2023cluster}. $G^*$: "
        "\\citet{carter2017asymptotic} with $\\rho=0$. Identifying variation for $\\theta_D$ (partial leverage) by developer: "
        + ", ".join(f"{u} {100 * v:.0f}\\%" for u, v in zip(ld_.unit.head(4), ld_.share.head(4)))
        + f"; {nzero} of 19 developers contribute nothing (they have only singleton families) and {nsmall} more under 0.1\\%; by family: "
        + ", ".join(f"{u} {100 * v:.0f}\\%" for u, v in zip(fam.unit.head(3), fam.share.head(3))) + ".")
    rc.write_aea_table(T("tabE3_inference.tex"), "Reconciling the Inference of the Original Table 8 (Family Fixed Effects)",
                       "tab:app-inference", "@{}l ccc ccc@{}", hdr, body, notes,
                       "Authors' calculations; see Table~\\ref{tab:app-matched}.", tabcolsep="3pt")


# =========================================================================================== Table E4: reliability
def table_E4():
    C = pd.read_csv(T("reliability_comparisons.csv"))
    R = pd.read_csv(T("reliability_summary.csv"))
    body = [rc.panel_row("Panel A. Discrepancy between two compute measures, $\\ln(C_{\\text{alt}}/6ND)$", 6)]
    body.append(" & Models (fam.) & Mean & S.d. & 95\\% interval & MAD s.d.")
    body.append("\\cmidrule(lr){2-6}")
    lab = {"hardware-time reconstruction (independent of N, D)": "Hardware time (independent)",
           "m4 'independent' set (Hardware*/Reported* Epoch methods)": "Epoch `Hardware'/`Reported'",
           "Epoch operation counting (Epoch's own 6ND)": "Epoch operation counting",
           "developer-reported FLOP": "Developer-reported FLOP"}
    for _, r in C.iterrows():
        body.append(" & ".join([lab[r.comparison], f"{int(r.n)} ({int(r.n_fam)})", F(r["mean"]), F(r.sd),
                                f"$[{Fm(r.sd_lo)},{Fm(r.sd_hi)}]$", F(r.sd_mad)]))
    hw = C[C.comparison.str.startswith("hardware")].iloc[0]
    body.append("\\addlinespace[4pt]")
    body.append(rc.panel_row("Panel B. Implied reliability of $\\ln C$, $\\lambda=1-\\sigma_u^2/\\text{Var}(\\ln C)$", 6))
    body.append(" & Models & $\\sigma_u$ & Pooled & Within family & Within, $n-1$")
    body.append("\\cmidrule(lr){2-6}")
    sig_lab = {"hardware-time sd (upper bound on sigma_u)": "Hardware-time s.d.",
               "hardware-time sd, upper 95% bootstrap bound": "\\quad upper 95\\% bound",
               "m4 'independent' set sd (as in Section VI.A)": "Section VI.A set",
               "Epoch 'Confident' class, ln3/1.645 (as in Table 8)": "Epoch `Confident' (Table 8)"}
    for smp, nlab in [("legacy support (57)", "57"), ("strict support (20)", "20"), ("main (128)", "128")]:
        for _, r in R[R["sample"] == smp].iterrows():
            body.append(" & ".join([sig_lab[r.sigma_label], nlab, F(r.sigma_u), F(r.reliability_pooled), F(r.reliability_within),
                                    F(r.reliability_within_table8_formula)]))
        body.append("\\addlinespace[2pt]")
    body = body[:-1]
    notes = (
        "Panel A compares $C=6ND$ (reported $N$ and $D$; exact Hugging Face parameter counts where available) with other compute "
        "measures for main-sample models matched to Epoch \\citep{epochai2026data}. Only hardware time is independent of reported $N$ "
        "and $D$: chip-hours (or chips $\\times$ training hours) $\\times$ peak dense 16-bit FLOP/s $\\times$ a common utilization of 0.3. "
        "Epoch's utilization field is back-calculated from its own compute estimate for most rows and is not used. Epoch's reported "
        "compute is itself $6ND$ (operation counting), a developer-reported count (usually $6ND$), or a geometric mean of $6ND$ and "
        "hardware time; the `Hardware'/`Reported' set used in Section VI.A mixes all three. The hardware-time s.d.\\ also contains "
        "the dispersion of true utilization and any error in chip-hours, so it bounds the error s.d.\\ of $\\ln 6ND$ from above; its "
        f"within-family value is {F(hw.sd_within_family_hw)} ({int(hw.n_within_families)} families with two or more records). "
        "Intervals: family-cluster bootstrap, 9,999 draws; MAD s.d.: $1.4826\\times$ median absolute deviation. Panel B: within-family "
        "variance $\\sum\\tilde c_i^2/(n-F)$, net of the $F$ family means; the last column divides by $n-1$, as the code behind the "
        "original Table 8 did. Within-family reliabilities assume errors independent across a family's models; an error in a "
        "family-wide token count is absorbed by the family effect.")
    rc.write_aea_table(T("tabE4_reliability.tex"), "How Well Is Training Compute Measured?", "tab:app-reliability",
                       "@{}l ccccc@{}", [], body, notes, "Authors' calculations.", tabcolsep="5pt")


# =========================================================================================== Table E5: Ho et al.
def table_E5():
    H = J("progress_headline.json")
    A = pd.read_csv(os.path.join(rc.TAB, "m5_progress_table7_panelA.csv")).set_index("code")
    att = pd.read_csv(os.path.join(rc.TAB, "m5_progress_attenuation_panelA.csv")).set_index("code")
    rep, ms, ph, corr = H["replication"], H["multistart"], H["phi"], H["boot_corr"]
    neu = H["dmr"]["neutrality"]

    def ci(lo, hi, nd=1):
        return f"$[{Fm(lo, nd)},{Fm(hi, nd)}]$" if np.isfinite(hi) else f"$[{Fm(lo, nd)},\\infty)$"

    def row(lab, r, obj=""):
        return " & ".join([lab, F(r.TC_months, 1), ci(r.TC_lo, r.TC_hi), F(r.mse, 4), obj])

    body = [rc.panel_row("Panel A. Replication of Ho et al.'s preferred model (231 observations, 144 papers)", 5)]
    body.append(" & ".join(["Published (bootstrap median)", "8.4", "$[4.5,14.3]$", "--", "0.0518129"]))
    t = rep["TC_boot_iid100"]
    body.append(" & ".join(["Their code and protocol, median", F(t[1], 2), ci(t[0], t[2]), F(A.loc["A1"].mse, 4),
                            F(rep["objective"], 7)]))
    t = rep["TC_boot_cluster"]
    body.append(" & ".join(["\\quad point estimate, cluster interval", F(rep["TC_point"], 2), ci(t[0], t[2]), "", ""]))
    t = rep["conv_TC_boot_cluster"]
    body.append(" & ".join(["Same objective, run to convergence", F(rep["conv_TC"], 2), ci(t[0], t[2]),
                            F(A.loc["A1c"].mse, 4), F(rep["conv_objective"], 7)]))
    body.append(" & ".join(["\\quad share of 300 starts below published", "", "", "",
                            F(ms["share_below_published"], 2)]))
    body.append(row("Unpenalized least squares (NLS)", A.loc["A2"]))
    lo, hi = H["dmr"]["TC_profile_ci_interp"]
    body.append(" & ".join(["\\quad profile-likelihood interval", "", ci(lo, hi), "", ""]))
    body.append(row("Benchmark-specific $E_b$ estimated", A.loc["A5"]))
    body.append("\\addlinespace[3pt]")
    body.append(rc.panel_row("Panel B. The ridge: what the data identify", 5))
    c1, c2 = corr["boot_m7_cluster"]["corr_alpha_year_beta_year"], corr["boot_m7_conv_cluster"]["corr_alpha_year_beta_year"]
    body.append(" & ".join(["Corr$(\\alpha g_N,\\beta g_D)$, their code / converged",
                            f"\\multicolumn{{2}}{{c}}{{{F(c1, 2)} / {F(c2, 2)}}}", "", ""]))
    lo, hi = ph["phi_ci_interp"]
    tl, th = ph["TC_range_over_interp_ci"]
    body.append(" & ".join(["$\\phi=g_N/g_C$, 95\\% profile interval", f"\\multicolumn{{2}}{{c}}{{$[{Fm(lo, 2)},{Fm(hi, 2)}]$}}", "", ""]))
    body.append(" & ".join(["\\quad range of $T_C$ over that interval", f"\\multicolumn{{2}}{{c}}{{{F(tl, 1)}--{F(th, 1)}}}", "", ""]))
    body.append(" & ".join(["Hicks neutrality $p$: converged / NLS",
                            f"\\multicolumn{{2}}{{c}}{{{F(neu['L1']['boot_p_two_sided'], 2)} / {F(neu['NLS']['boot_p_two_sided'], 2)}}}", "", ""]))
    body.append(row("Hicks neutrality imposed, converged", A.loc["A3"]))
    body.append(row("Hicks neutrality imposed, NLS", A.loc["A4"]))
    body.append(row("Effective data (converged)", att.loc["a3"]))
    r = A.loc["A7"]
    body.append(" & ".join(["Experimental exponents imposed", "$g_C<0$", ci(r.TC_lo, r.TC_hi), F(r.mse, 4), ""]))
    r = A.loc["A6"]
    body.append(" & ".join(["Year terms $\\div$ exponents (Whitfill)", F(r.TC_months, 1), ci(r.TC_lo, r.TC_hi), "--", ""]))
    hdr = [" & $T_C$ (months) & 95\\% interval & MSE & Objective"]
    notes = (
        "Ho et al.'s model: $\\ln\\text{ppl}=\\exp(\\alpha_{0b}-\\alpha g_Nt-\\alpha\\ln N)+\\exp(\\beta_{0b}-\\beta g_Dt-\\beta\\ln D)$, "
        "irreducible loss zero; $T_C=12\\ln2/(g_N+g_D)$. Their estimator minimizes MSE $+\\,0.0025\\sum|\\theta|$ by SLSQP from zero "
        f"with scipy's default tolerance, which stops after {H['dmr']['slsqp_default_nit']['nit']} iterations; `converged' runs the "
        "same objective and algorithm to convergence. The objective gap (0.0518 against 0.0507) is almost all penalty: MSE changes "
        "by 0.27 percent, and the penalty falls on constants whose values depend on an arbitrary normalization. Intervals: "
        "paper-cluster bootstrap (400 draws), percentiles of $g_N+g_D$ mapped to $T_C$ ($\\infty$ when non-positive growth lies "
        "inside); the first two rows use Ho et al.'s iid bootstrap. Profile intervals invert an iid Gaussian likelihood ratio and are "
        "indicative, because papers are clustered. The $\\phi$ interval is located on a grid of step 0.025 near its boundaries; the "
        "original draft's coarser grid gave $[-0.39,3.37]$, and its `5 to 38 months' was computed over the grid points $[-0.25,3.25]$. "
        "Hicks neutrality: $\\alpha g_N=\\beta g_D$. Effective data: $D$ measured as effective data under repetition "
        "\\citep{muennighoff2023scaling}.")
    rc.write_aea_table(T("tabE5_progress.tex"), "Algorithmic Progress in the Data of Ho et al.\\ (2024): Replication and Ridge",
                       "tab:app-progress", "@{}l cccc@{}", hdr, body, notes,
                       "Authors' calculations from the data and code of \\citet{ho2024algorithmic}.", tabcolsep="4pt")


# =========================================================================================== Table E6: allocative
TECH_LABEL = {
    "Besiroglu et al. (2024)": ("Chinchilla sweep", "\\citet{besiroglu2024chinchilla}"),
    "Hoffmann et al. (2022), TeX": ("Chinchilla sweep", "\\citet{hoffmann2022training}"),
    "Hoffmann et al. (2022), rounded": ("Chinchilla sweep", "\\citet{hoffmann2022training}, rounded"),
    "Chinchilla, kappa free (inner exponents)": ("Chinchilla sweep", "Chinchilla, $\\kappa$ free"),
    "Farseer grid (m5 fit, N incl. embeddings)": ("Sweeps of Section III", "Farseer, total $N$"),
    "m2: farseer (all)": ("Sweeps of Section III", "Farseer, non-embedding $N$"),
    "m2: gadre (C4)": ("Sweeps of Section III", "\\citet{gadre2024language}, C4"),
    "m2: olmo_ladder (all)": ("Sweeps of Section III", "OLMo ladder"),
    "m2: datablations (single_epoch)": ("Sweeps of Section III", "\\citet{muennighoff2023scaling}"),
    "m1: llama_3": ("IsoFLOP designs of Online Appendix D", "Llama 3 IsoFLOPs (Meta)"),
    "m1: marin_202603__comma__llama_2": ("IsoFLOP designs of Online Appendix D", "Marin, Comma"),
    "m1: marin_202603__dclm__llama_2": ("IsoFLOP designs of Online Appendix D", "Marin, DCLM"),
    "m1: marin_202603__nemotron__llama_2": ("IsoFLOP designs of Online Appendix D", "Marin, Nemotron-CC"),
    "m1: misfitting__fineweb_c4__transformer": ("IsoFLOP designs of Online Appendix D", "(Mis)Fitting, FineWeb/C4"),
    "Ho et al. (2024) model 7": ("Cross-lab regression", "\\citet{ho2024algorithmic}, model 7"),
}


def table_E6():
    R = pd.read_csv(T("allocative.csv"))
    K = pd.read_csv(T("kaplan_counterfactual.csv"))
    g = R[R["sample"] == "C >= 1e23"]
    body, grp = [], None
    for tech, (group, lab) in TECH_LABEL.items():
        r = g[g.technology == tech]
        if not len(r):
            continue
        r = r.iloc[0]
        if group != grp:
            body.append(rc.panel_row(group, 8))
            grp = group
        k = K[(K.technology == tech) & np.isclose(K.C, 5e26)]
        kap = F(k.CEG.iloc[0], 1) if len(k) else "--"
        body.append(" & ".join([lab, f"{100 * r.share_w_lt1_era1:.0f} / {100 * r.share_w_lt1_era2:.0f}",
                                F(r.gain_untrunc, 2), f"$[{Fm(r.gain_untrunc_lo, 2)},{Fm(r.gain_untrunc_hi, 2)}]$",
                                F(r.gain_trunc, 2), f"$[{Fm(r.gain_trunc_lo, 2)},{Fm(r.gain_trunc_hi, 2)}]$",
                                F(r.share_of_era1_gap_closed, 2) if np.isfinite(r.share_of_era1_gap_closed) else "--", kap]))
    body.append("\\addlinespace[3pt]")
    body.append(rc.panel_row("Other samples, \\citet{besiroglu2024chinchilla} technology", 8))
    for smp, lab in [("C >= 1e23, cleaned", "$C\\geq10^{23}$, cleaned"), ("Top-5 compute per year", "Top five per year"),
                     ("Epoch frontier flag", "Epoch frontier models")]:
        r = R[(R["sample"] == smp) & (R.technology == "Besiroglu et al. (2024)")].iloc[0]
        body.append(" & ".join([lab + f" ({int(r.n_era1)}, {int(r.n_era2)})", f"{100 * r.share_w_lt1_era1:.0f} / {100 * r.share_w_lt1_era2:.0f}",
                                F(r.gain_untrunc, 2), f"$[{Fm(r.gain_untrunc_lo, 2)},{Fm(r.gain_untrunc_hi, 2)}]$",
                                F(r.gain_trunc, 2), f"$[{Fm(r.gain_trunc_lo, 2)},{Fm(r.gain_trunc_hi, 2)}]$",
                                F(r.share_of_era1_gap_closed, 2), ""]))
    hdr = [" & $w<1$ (\\%) & \\multicolumn{2}{c}{Untruncated} & \\multicolumn{2}{c}{Truncated} & Gap & Kaplan",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
           " & era 1/2 & Gain & 95\\% CI & Gain & 95\\% CI & closed & CEG",
           " & (1) & (2) & (3) & (4) & (5) & (6) & (7)"]
    b = g[g.technology == "Besiroglu et al. (2024)"].iloc[0]
    notes = (
        f"Models in the Epoch database with $C\\geq10^{{23}}$ FLOP: {int(b.n_era1)} released in 2020--21 (era 1) and {int(b.n_era2)} "
        "in 2022--24 (era 2); dense, at most four epochs, $D=C/6N$. Farrell cost efficiency against the training-only frontier, "
        "$CE=C_{\\min}(L(N,D))/6ND$, depends only on the wedge $w=\\varepsilon_N/\\varepsilon_D$ within the Chinchilla family. "
        "Untruncated gain: ratio of era geometric means of $CE$ (as in the original Table 9). Truncated gain: the same with "
        "$CE(\\min(w,1))$, which counts only the elimination of under-training: over-training ($w>1$) can reflect inference "
        "demand and is not treated as inefficiency, whereas $w<1$ cannot (Proposition 2). Column 1: share of models with "
        "$w<1$. Column 6: $\\ln(\\text{truncated gain})$ divided by the era-1 log compute lost to under-training. Column 7: compute-"
        "equivalent gain of abandoning the Kaplan rule ($N\\propto C^{0.73}$ from GPT-3) at $5\\times10^{26}$ FLOP, the counterfactual "
        "of \\citet{gundlach2025origin}; the Kaplan allocations have $w<1$ under every technology, so truncation does not change it. "
        "Intervals: percentile bootstrap over models within era (999 draws); only the \\citet{besiroglu2024chinchilla} row also "
        "draws the technology's parameters, so the other intervals hold the technology fixed, and with seven era-1 models all "
        "intervals are rough. A wedge below one is counted as allocative error; by Proposition~\\ref{prop:A-wedge}(v) it can also "
        "reflect a binding data constraint or data-augmenting lab productivity. "
        "Technologies from other sweeps use each sweep's own $N$ and $D$ conventions and loss corpus, so their levels are indicative. "
        "Group labels give where each technology is estimated: the Chinchilla sweep and the sweeps of Section III, the IsoFLOP designs "
        "of Online Appendix D (never estimated in Section III), and Ho et al.'s cross-lab model, under which every model has $w>1$.")
    rc.write_aea_table(T("tabE6_allocative.tex"), "Allocative Gains from Rebalancing, with and without Truncating the Wedge at One",
                       "tab:app-allocative", "@{}l c cc cc cc@{}", hdr, body, notes,
                       "Authors' calculations from the Epoch AI models database \\citep{epochai2026data}.", tabcolsep="2.6pt")


# =========================================================================================== Table E7: TFP units
def table_E7():
    Tt = pd.read_csv(T("tfp_units.csv"))
    tl = Tt[(Tt.output == "y_hellaswag") & Tt.benchmark.str.startswith("legacy")]
    ts = Tt[(Tt.output == "y_hellaswag") & Tt.benchmark.str.startswith("strict")]
    body = []
    meas = [("Family effects, inputs netted with benchmark theta_C", "Family effects, $\\theta_C$ netting"),
            ("Family effects, within-family (theta_N, theta_D) netting", "Family effects, within netting"),
            ("Within-developer residuals (Mertens design), own theta_C", "Within-developer residuals")]
    for smp, lab in [("All dense base models", "All 128 models (38 families)"),
                     ("Drop distilled and synthetic-data", "Without distilled and synthetic-data models"),
                     ("Also drop code-specialised", "Also without code families (32 families)")]:
        body.append(rc.panel_row(lab, 7))
        for mk, ml in meas:
            a = tl[(tl["sample"] == smp) & (tl.measure == mk)].iloc[0]
            b = ts[(ts["sample"] == smp) & (ts.measure == mk)].iloc[0]
            own = "own" in mk
            body.append(" & ".join([ml, F(a.odds_ratio, 2), F(a.compute_eq, 1), F(a.loss_Besiroglu, 2),
                                    "" if own else F(b.compute_eq, 1), "" if own else F(b.loss_Besiroglu, 2), F(a.theta_conv, 2)]))
            body.append(" & ".join(["", f"$[{Fm(a.odds_lo, 2)},{Fm(a.odds_hi, 2)}]$", f"$[{Fm(a.compute_eq_lo, 1)},{Fm(a.compute_eq_hi, 1)}]$",
                                    f"$[{Fm(a.loss_Besiroglu_lo, 2)},{Fm(a.loss_Besiroglu_hi, 2)}]$",
                                    "" if own else f"$[{Fm(b.compute_eq_lo, 1)},{Fm(b.compute_eq_hi, 1)}]$",
                                    "" if own else f"$[{Fm(b.loss_Besiroglu_lo, 2)},{Fm(b.loss_Besiroglu_hi, 2)}]$", ""]))
        body.append("\\addlinespace[2pt]")
    body.append(" & ".join(["Manufacturing \\citep{syverson2004product}", "1.92", "1.92", "1.92", "1.92", "1.92", "$\\approx1$"]))
    hdr = [" & Output units & Compute & Loss & Compute & Loss & ",
           " & odds & \\multicolumn{2}{c}{$\\theta_C=0.33^{a}$} & \\multicolumn{2}{c}{$\\theta_C=0.43$} & $\\theta$ used",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
           " & (1) & (2) & (3) & (4) & (5) & (6)"]
    notes = (
        "90--10 ratios of Hicks-neutral productivity on HellaSwag log-odds $q$. A gap $\\Delta$ between the 90th and 10th "
        "percentiles is reported as an odds ratio $e^{\\Delta}$ (column 1, output units), as compute-equivalents $e^{\\Delta/\\theta_C}$ "
        "(input units: the compute the 10th-percentile family needs to match the 90th), and in reducible-loss units "
        "$e^{\\gamma\\Delta/\\theta_C}$ with $\\gamma=0.178$ \\citep{besiroglu2024chinchilla}; $\\gamma=0.155$ "
        "\\citep{hoffmann2022training} is in the replication files. $\\theta_C=0.33$: the design-matched benchmark on 57 models "
        "(benchmark including OLMo-2); $\\theta_C=0.43$: the strictly experimental benchmark on 20 models "
        "(Table~\\ref{tab:app-matched}). Family effects net inputs either with the benchmark $\\theta_C$ or with the within-family "
        "$(\\theta_N,\\theta_D)$; within-developer residuals follow \\citet{mertens2026secret} (developer and year effects, single-model "
        "developers dropped) and use the regression's own $\\theta_C$ (column 6) in columns 2--3 ($^{a}$). Brackets: 90 percent intervals from a developer-"
        f"cluster bootstrap ({rc.B_TFP} draws) that also draws $\\theta_C$ from its benchmark bootstrap. Manufacturing's 1.92 is an "
        "output-unit ratio and, with returns to scale near one, also an input-equivalent ratio; for LLMs the two differ by the "
        "factor $1/\\theta_C$ in logs, because returns to compute are small. Family effects also absorb benchmark contamination, "
        "similarity of training data to the benchmark and omitted teacher compute (distillation, synthetic data).")
    rc.write_aea_table(T("tabE7_tfp.tex"), "Productivity Dispersion across Model Families in Output and Input Units",
                       "tab:app-tfp-units", "@{}l cccccc@{}", hdr, body, notes,
                       "Authors' calculations; outputs from \\citet{ruan2024observational} and \\citet{maiapolo2024sloth}.",
                       tabcolsep="3pt")


# =========================================================================================== figures
def fig_support():
    import matplotlib.pyplot as plt
    sys.path.insert(0, rc.M4DIR)
    import experiments as ex  # noqa
    import lalonde as ll  # noqa
    from scipy.spatial import ConvexHull
    lad, gad = ex.build_olmo_ladder(), ex.build_gadre()
    mem = pd.read_csv(T("matched_hull_members.csv"))
    aer_style.use()
    fig, ax = plt.subplots(figsize=(aer_style.WIDTH_FULL * 0.62, 3.1))
    for which, col, ls, lab in [("legacy", aer_style.ORANGE, "--", "support with OLMo-2 7B/13B"),
                                ("strict", aer_style.BLUE, "-", "strictly experimental support")]:
        L = lad if which == "legacy" else lad[~lad.target]
        E = ll.exp_surface_data(L, gad)
        pts = E[["n", "d"]].to_numpy() / np.log(10)
        hull = ConvexHull(pts)
        v = np.r_[hull.vertices, hull.vertices[0]]
        ax.plot(pts[v, 0], pts[v, 1], color=col, ls=ls, lw=1.2, label=lab)
    E = ll.exp_surface_data(lad[~lad.target], gad)
    ax.scatter(E.n / np.log(10), E.d / np.log(10), marker="x", s=10, color=aer_style.INK2, lw=0.7, label="designed runs")
    tg = lad[lad.target]
    ax.scatter(tg.n / np.log(10), tg.d / np.log(10), marker="*", s=60, color=aer_style.ORANGE, zorder=4, label="OLMo-2 7B/13B")
    n10, d10 = np.log10(mem.N), np.log10(mem.D)
    ins, leg = mem.in_strict_hull.astype(bool), mem.in_legacy_hull.astype(bool)
    ax.scatter(n10[~leg], d10[~leg], s=9, color=aer_style.MUTED, alpha=0.6, lw=0, label="released models, outside")
    ax.scatter(n10[leg & ~ins], d10[leg & ~ins], s=14, facecolor="none", edgecolor=aer_style.ORANGE, lw=0.8,
               label="released, inside OLMo-2 support only (37)")
    ax.scatter(n10[ins], d10[ins], s=14, color=aer_style.BLUE, lw=0, label="released, inside strict support (20)")
    for m in (20, 200, 2000):
        xs = np.linspace(7.8, 11.6, 10)
        ax.plot(xs, xs + np.log10(m), color=aer_style.GRID, lw=0.8, zorder=0)
        ax.text(7.87, 7.87 + np.log10(m) + 0.02, f"$D/N={m}$", fontsize=6.3, color=aer_style.INK2, rotation=37,
                rotation_mode="anchor", bbox=dict(fc="white", ec="none", pad=0.2))
    ax.set_xlabel("$\\log_{10}$ parameters $N$")
    ax.set_ylabel("$\\log_{10}$ tokens $D$")
    ax.set_xlim(7.8, 11.6)
    ax.set_ylim(8.5, 13.4)
    ax.legend(fontsize=6.3, loc="lower right", handletextpad=0.3, labelspacing=0.3)
    aer_style.savefig(fig, f"{rc.PREFIX}_fig_support")


def fig_matched():
    import matplotlib.pyplot as plt
    M = pd.read_csv(T("matched_long.csv"))
    aer_style.use()
    fig, axs = plt.subplots(1, 3, figsize=(aer_style.WIDTH_FULL, 2.7), sharex=False)
    ests = ["OLS", "Year FE", "Developer FE", "Family FE", "Lab x period FE"]
    labs = ["OLS", "Year FE", "Dev. FE", "Family FE", "Lab$\\times$period"]
    for ax, par, title in [(axs[0], "theta_C", "(a) HellaSwag, $\\theta_C$ bias"), (axs[1], "theta_D", "(b) HellaSwag, $\\theta_D$ bias")]:
        for k, (smp, col, off, lab) in enumerate([(S_STRICT, aer_style.BLUE, -0.15, "strict benchmark, 20 models"),
                                                  (S_LEG, aer_style.ORANGE, 0.15, "with OLMo-2, 57 models")]):
            for i, e in enumerate(ests):
                r = M[(M["sample"] == smp) & (M.output == "y_hellaswag") & (M.surface_kind == "ols") & (M.estimator == e) & (M.param == par)]
                if not len(r) or not bool(r.identified.iloc[0]):
                    continue
                r = r.iloc[0]
                se = r.bias_se_cv3
                ax.errorbar(r.bias, i + off, xerr=1.96 * se, fmt="o", color=col, ms=3.5, lw=1.0, capsize=0,
                            label=lab if i == 0 else None)
        ax.axvline(0, color=aer_style.INK2, lw=0.8)
        ax.set_yticks(range(len(ests)))
        ax.set_yticklabels(labs if ax is axs[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel("bias (Obs. $-$ Exp.)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    ax = axs[2]
    outs = [("y_hellaswag", "ols", "HellaSwag"), ("y_arc_c", "tobit", "ARC-C"), ("y_winogrande", "tobit", "Winogrande"), ("y_core", "ols", "Composite")]
    for smp, col, off in [(S_STRICT, aer_style.BLUE, -0.15), (S_LEG, aer_style.ORANGE, 0.15)]:
        for i, (o, kind, _) in enumerate(outs):
            r = M[(M["sample"] == smp) & (M.output == o) & (M.surface_kind == kind) & (M.estimator == "OLS") & (M.param == "theta_C")].iloc[0]
            ax.errorbar(r.bias, i + off, xerr=1.96 * r.bias_se_cv3, fmt="o", color=col, ms=3.5, lw=1.0, capsize=0)
    ax.axvline(0, color=aer_style.INK2, lw=0.8)
    ax.set_yticks(range(len(outs)))
    ax.set_yticklabels([o[2] for o in outs])
    ax.invert_yaxis()
    ax.set_title("(c) OLS $\\theta_C$ bias by output", loc="left")
    ax.set_xlabel("bias (Obs. $-$ Exp.)")
    fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
    aer_style.savefig(fig, f"{rc.PREFIX}_fig_matched")


def fig_reliability():
    import matplotlib.pyplot as plt
    Mo = pd.read_csv(T("reliability_models.csv"))
    CO = pd.read_csv(T("compute_only.csv"))
    R = pd.read_csv(T("reliability_summary.csv"))
    aer_style.use()
    fig, axs = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    h = Mo[np.isfinite(Mo.dev_hw)].sort_values("dev_hw").reset_index(drop=True)
    y = np.arange(len(h))
    ax.scatter(h.dev_hw, y, s=12, color=aer_style.BLUE, label="hardware time vs $6ND$", zorder=3)
    ax.scatter(h.dev_epoch, y, s=12, marker="D", facecolor="none", edgecolor=aer_style.ORANGE, lw=0.8,
               label="Epoch's reported compute vs $6ND$")
    ax.axvline(0, color=aer_style.INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([m.split("/")[-1].replace("-hf", "").replace("-deduped", " (dd)") for m in h.model], fontsize=5.2)
    ax.set_xlabel("$\\ln(C_{\\mathrm{alt}}/6ND)$")
    ax.set_title("(a) Compute comparisons, 27 models", loc="left")
    ax.legend(fontsize=6.0, loc="lower right")
    ax = axs[1]
    # EIV-corrected family-FE and OLS theta_C as a function of sigma_u on the 57-model support
    g = CO[(CO["sample"] == S_LEG) & CO.estimator.str.startswith("Family FE, EIV") & (CO.formula == "correct")]
    r57 = R[R["sample"] == "legacy support (57)"].iloc[0]
    sig = np.linspace(0, 0.8, 161)
    for est, col, vcol, lab in [("OLS", aer_style.BLUE, "var_c_pooled", "pooled OLS"), ("Family FE", aer_style.ORANGE, "var_c_within", "family FE")]:
        base = CO[(CO["sample"] == S_LEG) & (CO.estimator == f"{est}, EIV (hardware-time sd (upper bound))")].iloc[0]
        b_raw = base.obs * base.reliability
        lam = 1 - sig ** 2 / r57[vcol]
        ax.plot(sig, b_raw / lam, color=col, label=f"{lab}, EIV-corrected")
        ax.axhline(base.bench, color=col, lw=0.8, ls=":")
    for s_, lab in [(0.292, "hardware\ntime bound"), (0.668, "Epoch\n`Confident'")]:
        ax.axvline(s_, color=aer_style.INK2, lw=0.7, ls="--")
        ax.text(s_ + 0.01, 0.265, lab, fontsize=6.2, color=aer_style.INK2, va="bottom")
    ax.plot([], [], color=aer_style.INK2, lw=0.8, ls=":", label="design-matched benchmark")
    ax.scatter([0.668], [0.858], color=aer_style.INK, s=14, zorder=4)
    ax.annotate("original Table 8 row (0.858):\nwrong within-family variance", (0.668, 0.858), xytext=(-150, -2),
                textcoords="offset points", fontsize=6.2, va="center")
    ax.set_xlabel("assumed error s.d. $\\sigma_u$ of $\\ln C$")
    ax.set_ylabel("$\\theta_C$ (HellaSwag log-odds)")
    ax.set_ylim(0.25, 0.95)
    ax.set_title("(b) EIV correction, 57 models", loc="left")
    ax.legend(fontsize=6.2, loc="center left")
    fig.tight_layout(w_pad=0.8)
    aer_style.savefig(fig, f"{rc.PREFIX}_fig_reliability")


def fig_allocative():
    import matplotlib.pyplot as plt
    R = pd.read_csv(T("allocative.csv"))
    g = R[R["sample"] == "C >= 1e23"].set_index("technology")
    techs = [t for t in TECH_LABEL if t in g.index and not t.startswith("Ho et al")]
    labs = [TECH_LABEL[t][1].replace("\\citet{besiroglu2024chinchilla}", "Besiroglu et al.").replace(
        "\\citet{hoffmann2022training}", "Hoffmann et al.").replace("\\citet{gadre2024language}", "Gadre et al.").replace(
        "\\citet{muennighoff2023scaling}", "Muennighoff et al.").replace("\\kappa", "\\kappa").replace("\\ ", " ") for t in techs]
    aer_style.use()
    fig, ax = plt.subplots(figsize=(aer_style.WIDTH_FULL * 0.72, 3.4))
    y = np.arange(len(techs))
    for col, off, key, lab in [(aer_style.ORANGE, 0.15, "untrunc", "untruncated (over-training counted as waste)"),
                               (aer_style.BLUE, -0.15, "trunc", "wedge truncated at one")]:
        v = g.loc[techs, f"gain_{key}"].to_numpy()
        lo, hi = g.loc[techs, f"gain_{key}_lo"].to_numpy(), g.loc[techs, f"gain_{key}_hi"].to_numpy()
        ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", color=col, ms=3.5, lw=1.0, capsize=0, label=lab)
    from matplotlib.ticker import NullFormatter
    ax.axvline(1, color=aer_style.INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("realized allocative gain, 2020\u201321 to 2022\u201324 ($C\\geq10^{23}$ FLOP)")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    aer_style.savefig(fig, f"{rc.PREFIX}_fig_allocative")


def fig_tfp_units():
    import matplotlib.pyplot as plt
    Tt = pd.read_csv(T("tfp_units.csv"))
    t = Tt[(Tt.output == "y_hellaswag") & Tt.benchmark.str.startswith("legacy") &
           (Tt.measure == "Family effects, inputs netted with benchmark theta_C")].set_index("sample")
    aer_style.use()
    fig, ax = plt.subplots(figsize=(aer_style.WIDTH_FULL * 0.62, 2.4))
    rows = [("odds_ratio", "odds_lo", "odds_hi", "output units: odds"), ("loss_Besiroglu", "loss_Besiroglu_lo", "loss_Besiroglu_hi",
            "output units: reducible loss"), ("compute_eq", "compute_eq_lo", "compute_eq_hi", "input units: compute")]
    smps = [("All dense base models", aer_style.ORANGE, 0.18, "all 128 models"),
            ("Also drop code-specialised", aer_style.BLUE, -0.18, "without code, distilled, synthetic")]
    for s, col, off, lab in smps:
        for i, (k, lo, hi, _) in enumerate(rows):
            r = t.loc[s]
            ax.errorbar(r[k], i + off, xerr=[[r[k] - r[lo]], [r[hi] - r[k]]], fmt="o", color=col, ms=3.5, lw=1.0, capsize=0,
                        label=lab if i == 0 else None)
    from matplotlib.ticker import NullFormatter
    ax.axvline(1.92, color=aer_style.INK2, lw=0.9, ls="--")
    ax.text(2.0, -0.55, "manufacturing (Syverson): 1.92", fontsize=6.4, color=aer_style.INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlim(1, 250)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[3] for r in rows], fontsize=7)
    ax.set_xlabel("90\u201310 ratio of family productivity (HellaSwag), 90% interval")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    aer_style.savefig(fig, f"{rc.PREFIX}_fig_tfp_units")


def main():
    table_E1()
    table_E2()
    table_E3()
    table_E4()
    table_E5()
    table_E6()
    table_E7()
    fig_support()
    fig_matched()
    fig_reliability()
    fig_allocative()
    fig_tfp_units()
    rc.log("tables and figures written")


if __name__ == "__main__":
    main()
