"""Generate the condensed Online Appendix C tables (Monte Carlo) from the module CSVs.

Writes paper/tables/appC_designs.tex, appC_industry.tex, appC_transmission.tex.
Every number is read from output/tables/m6_montecarlo_*.csv (no hand-typed estimates).
Run: .venv/bin/python code/paper/make_appendix_cd_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "output" / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)


def f(x, d=3):
    """Format with a true minus sign; '--' for missing."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        # avoid "-0.000"
        if float(s) == 0:
            s = s[1:]
        else:
            s = "$-$" + s[1:]
    return s


def sci(x):
    if not np.isfinite(x) or x > 1e15:
        return "$\\infty$"
    e = int(np.floor(np.log10(x)))
    m = x / 10**e
    return f"${m:.1f}\\times10^{{{e}}}$"


# ---------------------------------------------------------------------------------------------
# Table C1: Design A
# ---------------------------------------------------------------------------------------------
def table_designs():
    s = pd.read_csv(T / "m6_montecarlo_designA_summary.csv")
    p = pd.read_csv(T / "m6_montecarlo_designA_profile.csv").set_index("cell")
    dg = pd.read_csv(T / "m6_montecarlo_designA_diagnostics.csv").set_index("cell")
    cells = [
        ("onpath", "(i) On path, $s=0.02$"),
        ("iso16", "(ii) IsoFLOP, $\\pm16\\times$"),
        ("iso4", "(ii$'$) IsoFLOP, $\\pm4\\times$"),
        ("fact", "(iii) Factorial grid"),
        ("opt_s0", "(iv) Optimizing, $s=0$"),
        ("opt_s0.1", "(iv) Optimizing, $s=0.1$"),
        ("opt_s0.3", "(iv) Optimizing, $s=0.3$"),
        ("opt_s1", "(iv) Optimizing, $s=1$"),
        ("kaplan", "(v) Kaplan beliefs"),
    ]

    def get(cell, est, par, col):
        r = s[(s.cell == cell) & (s.estimator == est) & (s.param == par)]
        return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan

    L = []
    L.append("\\begin{table}[tp]")
    L.append("\\centering")
    L.append("\\caption{Monte Carlo Design A: What Alternative Designs Identify at Equal Compute}")
    L.append("\\label{tab:app-mc-designs}")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\begin{tabular}{@{}lcccccc@{}}")
    L.append("\\toprule")
    L.append(" & \\multicolumn{4}{c}{Primal estimator ($\\kappa=1$)} & \\multicolumn{2}{c}{Dual estimator} \\\\")
    L.append("\\cmidrule(lr){2-5}\\cmidrule(l){6-7}")
    L.append("Design & $a$ & $\\gamma$ & $\\sigma^*$ & $\\ln M^*(10^{24})$ & $\\sigma^*$ & $\\ln M^*(10^{24})$ \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{7}{@{}l}{\\textit{Panel A. Bias [root-mean-squared error]}} \\\\[1pt]")
    for cell, lab in cells:
        row1 = [lab]
        row2 = [""]
        for par, d in [("a", 3), ("gamma", 3), ("sigma_star", 3), ("lnMstar", 2)]:
            rm = get(cell, "primal", par, "rmse")
            row1.append(f(get(cell, "primal", par, "bias"), d))
            row2.append("[" + f(rm, 1 if (par == "lnMstar" and rm >= 10) else d) + "]")
        for par, d in [("sigma_star", 3), ("lnMstar", 2)]:
            b = get(cell, "dual", par, "bias")
            r = get(cell, "dual", par, "rmse")
            row1.append(f(b, d))
            row2.append("--" if np.isnan(r) else "[" + f(r, d) + "]")
        L.append(" & ".join(row1) + " \\\\")
        L.append(" & ".join(row2) + " \\\\[1pt]")
    L.append("\\midrule")
    L.append("\\multicolumn{7}{@{}l}{\\textit{Panel B. Identification diagnostics and inference for $\\sigma^*$}} \\\\[1pt]")
    L.append(" & cond$(J'J)$ & Corner & Wald & Bootstrap & $\\kappa$ free: & $\\kappa$ free: \\\\")
    L.append(" & normalized & fits (\\%) & coverage (\\%) & coverage (\\%) & CI width & flat (\\%) \\\\")
    L.append("\\cmidrule(l){2-7}")
    for cell, lab in cells:
        cond = float(dg.loc[cell, "cond_normalized"])
        corner = get(cell, "primal", "sigma_star", "corner_share")
        wc = get(cell, "primal", "sigma_star", "wald_cover")
        wcomp = get(cell, "primal", "sigma_star", "wald_computable")
        bc = get(cell, "primal", "sigma_star", "boot_cover")
        width = float(p.loc[cell, "median_ci_width"])
        flat = float(p.loc[cell, "share_flat_everywhere"])
        lo, hi = float(p.loc[cell, "median_ci_lo"]), float(p.loc[cell, "median_ci_hi"])
        wtxt = f"{100*wc:.0f}" + ("$^{\\dagger}$" if wcomp < 0.95 else "")
        btxt = "--" if np.isnan(bc) else f"{100*bc:.0f}" + ("$^{\\S}$" if cell in ("onpath", "opt_s0.3") else "")
        wid = "full" if (lo <= 0.5 + 1e-9 and hi >= 0.95 - 1e-9) else f"{width:.3f}"
        L.append(" & ".join([lab, sci(cond), f"{100*corner:.0f}", wtxt, btxt, wid, f"{100*flat:.0f}"]) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\par\\smallskip")
    # numbers for the note, read from the CSVs
    bc = pd.read_csv(T / "m6_montecarlo_designA_bootcheck.csv")
    on = bc[(bc.cell == "onpath") & (bc.param == "sigma_star")].iloc[0]
    note = (
        "\\begin{tablenotes}"
        "Each design has 90 runs, 10 at each of the nine Chinchilla IsoFLOP budgets "
        "($6\\times10^{18}$ to $3\\times10^{21}$ FLOP), and the same total compute, "
        "$\\sum 6ND=5.1\\times10^{22}$ FLOP; 500 replications per design. The truth is the "
        "\\citet{besiroglu2024chinchilla} technology ($a=0.513$, $\\gamma=0.178$, $\\sigma^*=0.737$, "
        "$\\ln M^*(10^{24})=2.90$); log-loss noise has standard deviation 0.0075. Designs: (i) runs within "
        "$s=0.02$ of the expansion path; (ii) ten sizes log-spaced between $N^*/16$ and $16N^*$ (or $N^*/4$ and "
        "$4N^*$) at each budget; (iii) a $9\\times10$ grid in $(N,D)$ spanning $256\\times$ in each input; "
        "(iv) optimizing labs with allocation errors of standard deviation $s$ in $\\ln(D/N)$ at given compute; "
        "(v) labs that follow Kaplan's rule $N\\propto C^{0.73}$ plus errors with $s=0.1$. Primal: Huber "
        "($\\delta=10^{-3}$) fit of $\\ln L$ with nine starting values, one at the truth, inside a parameter box. "
        "Dual: $a$ from the (noisy) optima or IsoFLOP minima, $\\gamma$ from the frontier, "
        "$\\alpha=\\gamma/a$, $\\beta=\\gamma/(1-a)$; not computed for the factorial grid. Panel B: condition "
        "number of $J'J$ at the truth with $A$ and $B$ normalized at the geometric means of $N$ and $D$; "
        "corner = estimate on the box boundary or one power-law term below 2 percent of reducible loss; "
        "Wald = LAD-sandwich delta-method 95 percent interval ($^{\\dagger}$: covariance singular or corner in "
        "more than 5 percent of replications, coverage among computable ones); bootstrap = pairs-bootstrap "
        "percentile interval (149 draws, 150 replications), each draw warm-started at the replication's "
        "estimate ($^{\\S}$: in a separate check with 60 on-path replications, restarting every draw from all "
        f"nine starting values raises coverage from {100*on.warm_cover:.0f} to {100*on.multi_cover:.0f} "
        f"percent but widens the median interval from {on.warm_median_width:.2f} to "
        f"{on.multi_median_width:.2f}); $\\kappa$ free = 95 percent profile-likelihood interval for "
        "$\\sigma^*$ in $L=E+(AN^{-a_1}+BD^{-b_1})^{\\kappa}$ on the grid $[0.50,0.95]$, median width, "
        "and the share of replications whose interval is the whole grid (``full'')."
        "\\end{tablenotes}"
    )
    L.append(note)
    L.append("\\begin{tablenotes}[Source]Authors' simulations; replication code "
             "\\texttt{code/analysis/m6\\_montecarlo}.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appC_designs.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------
# Table C2: Design B (industry)
# ---------------------------------------------------------------------------------------------
def table_industry():
    s = pd.read_csv(T / "m6_montecarlo_industry_summary.csv")
    sc = ["exog", "funding", "target", "predet"]
    ests = [
        ("E1_pooled_huber", "Pooled Huber, $E$ free (ML practice)"),
        ("E2_pooled_nls", "Pooled NLS + generation FE"),
        ("E3_lab_fe", "Lab FE"),
        ("E5_labgen_fe", "Lab $\\times$ generation FE"),
        ("E6_acf", "ACF, current $c$ as instrument"),
        ("E6b_acf_lagonly", "ACF, lagged instruments only"),
        ("E7_iv_path", "IV (compute price) + path$^{\\ddagger}$"),
        ("E9_foc_system", "FOC system, $T$ ignored"),
        ("E10_foc_system_T", "FOC system, $T$ proxied"),
    ]
    pars = [
        ("gamma", "Panel A. Frontier elasticity $\\gamma$ (truth 0.178)", 3),
        ("a", "Panel B. Allocation exponent $a$ (truth 0.513)", 3),
        ("tfp_growth", "Panel C. TFP growth per generation (truth 0.150)", 3),
        ("lnMstar", "Panel D. $\\ln M^*(10^{24})$ (truth 2.898)", 2),
    ]

    def cell(est, par, scen):
        r = s[(s.estimator == est) & (s.param == par) & (s.scenario == scen)].iloc[0]
        if est == "E7_iv_path":
            return float(r.median_bias), float(r.mae)
        return float(r.bias), float(r.rmse)

    L = []
    L.append("\\begin{table}[tp]")
    L.append("\\centering")
    L.append("\\caption{Monte Carlo Design B: Production-Function Estimators in a Simulated AI Industry}")
    L.append("\\label{tab:app-mc-industry}")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{2.5pt}")
    L.append("\\begin{tabular}{@{}lcccc@{}}")
    L.append("\\toprule")
    L.append("Compute rule: & (a) Exogenous & (b) Funding & (c) Target & (d) Predetermined \\\\")
    L.append("\\midrule")
    for par, title, d in pars:
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for est, lab in ests:
            vals = [cell(est, par, x) for x in sc]
            L.append(lab + " & " + " & ".join(f"{f(b, d)} [{f(r, d)}]" for b, r in vals) + " \\\\")
        if par != "lnMstar":
            L.append("\\addlinespace")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\par\\smallskip")
    meta = pd.read_csv(T / "m6_montecarlo_industry_meta.csv").set_index("scenario")
    note = (
        "\\begin{tablenotes}"
        "Bias [root-mean-squared error] over 400 replications per compute rule. Simulated industry: 40 labs, "
        "6 generations, 1 to 4 models per lab-generation with compute offsets of about $8\\times$; "
        "\\citet{besiroglu2024chinchilla} technology; Hicks-neutral productivity "
        "$\\omega_{ft}=0.15(t-1)+x_{ft}$ with $x_{ft}=0.8x_{f,t-1}+\\xi_{ft}$ and sd$(x)=0.25$; seed noise with "
        "standard deviation 0.05 in $y=-\\ln(L-E)$. Compute rules: (a) budgets independent of $\\omega$; "
        "(b) $\\ln C$ rises by $2x_{ft}$; (c) labs buy the compute that reaches a loss target; (d) $\\ln C$ "
        "rises by $2x_{f,t-1}$. Allocation minimizes lifetime compute with log-normal inference demand "
        f"(mean $\\ln w={meta.loc['exog','mean_ln_w']:.2f}$) plus allocation errors (standard deviation 0.2 "
        "in $\\ln(D/N)$). $E$ is known to every estimator except the first. Both ACF rows use current "
        "within-lab-generation deviations of $n$ and $d$ as instruments, valid here by construction; without "
        "them the lagged moment set is under-identified. $^{\\ddagger}$Median bias [median absolute error], "
        "because just-identified 2SLS has no finite moments; median first-stage $F$ is "
        f"{min(meta.loc[x,'median_first_stage_F'] for x in ('exog','funding','predet')):.0f} to "
        f"{max(meta.loc[x,'median_first_stage_F'] for x in ('exog','funding','predet')):.0f} "
        f"under rules (a), (b), and (d) and {meta.loc['target','median_first_stage_F']:.0f} under (c). "
        "FOC system: the within-family loss equation stacked with the training-only first-order condition "
        "($w=1$), ignoring or proxying inference demand $T$ (proxy with log noise of standard deviation 0.3). "
        "TFP growth: mean change in generation means of $y-\\hat F(n,d)$."
        "\\end{tablenotes}"
    )
    L.append(note)
    L.append("\\begin{tablenotes}[Source]Authors' simulations; replication code "
             "\\texttt{code/analysis/m6\\_montecarlo}.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appC_industry.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------
# Table C3: transmission-bias formula
# ---------------------------------------------------------------------------------------------
def table_transmission():
    t = pd.read_csv(T / "m6_montecarlo_transmission_check.csv")
    on = t[t.allocation.str.startswith("on expansion")].reset_index(drop=True)
    lt = t[t.allocation.str.startswith("lifetime")].reset_index(drop=True)
    names = {"exog": "Exogenous", "funding": "Funding, $\\lambda=$", "predet": "Predetermined, $\\lambda=$",
             "target": "Target, sd$(\\bar y)=$"}
    L = []
    L.append("\\begin{table}[tp]")
    L.append("\\centering")
    L.append("\\caption{Transmission Bias in the Returns to Compute: Closed Form and Simulation}")
    L.append("\\label{tab:app-mc-transmission}")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\begin{tabular}{@{}lccccccc@{}}")
    L.append("\\toprule")
    L.append(" & & \\multicolumn{3}{c}{On the path ($T=0$), $n=10^6$} & $T>0$ & "
             "\\multicolumn{2}{c}{$n=240$} \\\\")
    L.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\\cmidrule(l){7-8}")
    L.append("Compute rule & & Formula & Simulated & Reverse & Simulated & Mean & Coverage (\\%) \\\\")
    L.append("\\midrule")
    for i in range(len(on)):
        r, q = on.iloc[i], lt.iloc[i]
        lab = names[r.rule]
        par = "" if r.rule == "exog" else (f"{r.param:g}")
        rev = "--" if r.gamma_reverse < 0 else f(r.gamma_reverse, 3)
        L.append(" & ".join([lab, par, f(r.slope_formula, 4), f(r.slope_sim, 4), rev, f(q.slope_sim, 4),
                             f(r.finite_mean, 3), f"{100*r.finite_cover_true_gamma:.0f}"]) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\par\\smallskip")
    note = (
        "\\begin{tablenotes}"
        "Slope of $y=-\\ln(L-E)$ on $c=\\ln C$ across labs. Formula: "
        "$\\gamma+\\pi_1\\mathrm{Var}(\\omega)/(\\pi_1^2\\mathrm{Var}(\\omega)+\\mathrm{Var}(\\eta))$ "
        "(Proposition~\\ref{prop:transmission}) with $\\gamma=0.1783$, sd$(\\omega)=0.25$, sd$(\\eta)=1$, "
        "persistence 0.8. Funding: $c=\\lambda\\omega+\\eta$; predetermined: $c=\\lambda\\omega_{-1}+\\eta$; "
        "target: labs buy the compute that reaches a target $\\bar y$ with the stated dispersion (a common "
        "target gives slope 0). Simulated: OLS on $10^6$ draws, seed noise 0.05 (simulation standard errors "
        "at most 0.0003). Reverse: $\\mathrm{Var}(y)/\\mathrm{Cov}(c,y)$; ``--'' where $\\mathrm{Cov}(c,y)=0$ "
        "and it is undefined. $T>0$: lifetime-optimal allocations with log-normal inference demand; under "
        "target rules an extra term of $-0.002$ to $-0.001$ appears because high-$T$ labs buy more compute. "
        "$n=240$: mean OLS slope and coverage of the nominal 95 percent OLS interval over 2,000 samples of 240 "
        "lab-generations; coverage rates are specific to sd$(\\eta)=1$ and rise with more exogenous "
        "dispersion in compute."
        "\\end{tablenotes}"
    )
    L.append(note)
    L.append("\\begin{tablenotes}[Source]Authors' simulations; replication code "
             "\\texttt{code/analysis/m6\\_montecarlo}.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appC_transmission.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    # v2 (2026-09-24): the Online Appendix C tables are now generated by make_appendix_c_tables_v2.py (Monte Carlo
    # re-run without the truth among the starting values).  table_designs() above is the v1 table and must not
    # overwrite the v2 appC_designs.tex; table_industry()/table_transmission() are reused by the v2 script.
    import runpy
    runpy.run_path(str(Path(__file__).with_name("make_appendix_c_tables_v2.py")), run_name="__main__")


# =============================================================================================
# Online Appendix D tables generated from CSVs (the others are hand-condensed from output/tables/*.tex)
# =============================================================================================
def pse(x, se, d=3):
    return f(x, d) + ("" if (se is None or np.isnan(se)) else f" ({f(se, d)})")


def table_selection():
    k = pd.read_csv(T / "m1_chinchilla_selection_k.csv").set_index("k")
    r = pd.read_csv(T / "m1_chinchilla_selection_rules.csv")
    L = ["\\begin{table}[tp]", "\\centering",
         "\\caption{Chinchilla Data: Dropping High-Loss Runs and Other Exclusion Rules}",
         "\\label{tab:app-selection}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "Sample & $n$ & $\\alpha$ & $\\beta$ & $a$ & $M^*(10^{21})$ & $w$ (70B) \\\\", "\\midrule",
         "\\multicolumn{7}{@{}l}{\\textit{Panel A. Drop the $k$ highest-loss runs (Huber)}} \\\\"]
    for kk in [0, 1, 2, 3, 4, 5, 6, 8, 10, 15]:
        x = k.loc[kk]
        L.append(" & ".join([f"$k={kk}$" + (" (Besiroglu)" if kk == 5 else ""), f"{int(x.n)}",
                             pse(x.huber_alpha, x.huber_se_alpha), pse(x.huber_beta, x.huber_se_beta),
                             pse(x.huber_a, x.huber_se_a), pse(x["huber_Mstar_1e+21"], x["huber_se_Mstar_1e+21"], 1),
                             pse(x.huber_w_chin70b, x.huber_se_w_chin70b, 2)]) + " \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{7}{@{}l}{\\textit{Panel B. Alternative exclusion rules (Huber)}} \\\\")
    labs = {
        "drop the 6 lowest-loss runs (L <= 5th lowest; ties)": "Drop 6 lowest-loss runs",
        "drop 5 largest |residual| (n=245 fit)": "Drop 5 largest residuals",
        "drop D/N < 0.4 (selection on X)": "Drop $D/N<0.4$",
        "drop D/N < 1": "Drop $D/N<1$",
        "drop D/N < 2": "Drop $D/N<2$",
        "drop the 1e19 IsoFLOP budget": "Drop the $10^{19}$ budget",
        "IsoFLOP-profile runs only": "IsoFLOP runs only",
        "IsoFLOP runs only, 5 highest loss dropped": "IsoFLOP runs, $k=5$",
    }
    for _, x in r.iterrows():
        if x.rule not in labs:
            continue
        L.append(" & ".join([labs[x.rule], f"{int(x.n)}", pse(x.alpha, x.se_alpha), pse(x.beta, x.se_beta),
                             pse(x.a, x.se_a), pse(x["Mstar_1e+21"], x["se_Mstar_1e+21"], 1),
                             pse(x.w_chin70b, x.se_w_chin70b, 2)]) + " \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{7}{@{}l}{\\textit{Panel C. Gaussian likelihood and truncated regression}} \\\\")
    labs = {"Gaussian MLE, n=245": "Gaussian, all runs",
            "Gaussian MLE, n=240 (ignores truncation)": "Gaussian, $k=5$",
            "Hausman-Wise truncated MLE, n=240 (L < Lbar)": "Truncated Gaussian, $k=5$"}
    for _, x in r.iterrows():
        if x.rule not in labs:
            continue
        L.append(" & ".join([labs[x.rule], f"{int(x.n)}", f(x.alpha), f(x.beta), f(x.a),
                             f(x["Mstar_1e+21"], 1), f(x.w_chin70b, 2)]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    sw = pd.read_csv(T / "m1_chinchilla_selection_swing.csv").set_index("param")
    L.append(
        "\\begin{tablenotes}Huber ($\\delta=10^{-3}$) estimates on the 245 digitized Chinchilla runs, with "
        "pairs-bootstrap standard errors (200 draws) in parentheses. $w$ is the wedge $\\varepsilon_N/\\varepsilon_D$ "
        "at Chinchilla-70B ($N=70$B, $D=1.4$T), which equals 1 if that model is training-compute-optimal. Panel A: "
        "$k=5$ is the sample of \\citet{besiroglu2024chinchilla}; the five dropped runs are the most data-starved runs of "
        "the $10^{19}$ IsoFLOP ($D/N=0.04$--0.40), with log residuals 9 to 45 times the residual standard deviation. "
        "Panel B: selection on the regressor ($D/N$ or budget) instead of the outcome. Panel C: the truncated-Gaussian "
        "likelihood \\citep{hausman1977social} conditions on $L$ below the fifth-highest loss (3.447). A paired "
        "bootstrap of the $k=0\\to5$ change (400 draws, rule applied within each draw) gives "
        f"$\\Delta\\beta={f(sw.loc['beta','diff'])}$ (s.e.\\ {f(sw.loc['beta','se_diff'])}) and "
        f"$\\Delta a={f(sw.loc['a','diff'])}$ (s.e.\\ {f(sw.loc['a','se_diff'])}).\\end{{tablenotes}}")
    L.append("\\begin{tablenotes}[Source]Epoch AI digitization of \\citet{hoffmann2022training}, Figure 4; "
             "authors' estimates.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appD_selection.tex").write_text("\n".join(L) + "\n")


def table_homothetic():
    h = pd.read_csv(T / "m3_wedge_homothetic.csv")
    models = [("B:meta-llama/Meta-Llama-3-8B", "Llama 3 8B"), ("B:meta-llama/Meta-Llama-3-70B", "Llama 3 70B"),
              ("B:meta-llama/Llama-3.1-405B", "Llama 3.1 405B"), ("B:Qwen/Qwen3-8B-Base", "Qwen3 8B"),
              ("B:google/gemma-3-27b-pt", "Gemma 3 27B")]
    ms = [16.0, 20.0, 24.2, 41.0, 192.0]
    L = ["\\begin{table}[tp]", "\\centering", "\\caption{Revealed Inference Demand under Homothetic Technologies}",
         "\\label{tab:app-homothetic}", "\\footnotesize", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         " & \\multicolumn{5}{c}{Compute-optimal tokens per parameter, $M^*$} \\\\", "\\cmidrule(l){2-6}",
         "Model & 16 & 20 & 24.2 & 41 & 192 \\\\", "\\midrule"]
    for rho, lab in [(0.30, "$\\rho=0.30$"), (0.3527, "$\\rho=0.3527$"), (0.40, "$\\rho=0.40$")]:
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{Panel: {lab}}}}} \\\\")
        for uid, name in models:
            row = [name]
            for m in ms:
                x = h[(h.uid == uid) & np.isclose(h.Mstar, m) & np.isclose(h.rho, rho)]
                row.append(f(float(x.TD.iloc[0]), 1))
            L.append(" & ".join(row) + " \\\\")
        if rho != 0.40:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    L.append(
        "\\begin{tablenotes}Entries are revealed lifetime inference tokens per training token, "
        "$T/D=3[(M/M^*)^{\\rho}-1]$, under a homothetic technology with $\\alpha=\\beta=\\rho$, for which "
        "$w=(M/M^*)^{\\rho}$. $M^*=16$ \\citep{porian2024resolving}, 20 (the Chinchilla rule of thumb), 24.2 (our "
        "CES fit to the Chinchilla data), 41 (Meta's Llama 3 law at $3.8\\times10^{25}$ FLOP), and 192 "
        "\\citep{hu2024minicpm}; $\\rho=0.3527$ is the tied exponent of \\citet{muennighoff2023scaling}. Negative "
        "entries mean the model is under-trained relative to that technology.\\end{tablenotes}")
    L.append("\\begin{tablenotes}[Source]Model sizes and token counts from developers' reports and Hugging Face; "
             "authors' calculations.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appD_homothetic.tex").write_text("\n".join(L) + "\n")


def table_lalonde_robust():
    ests = [("OLS", "OLS"), ("Year FE", "Year FE"), ("Developer FE", "Developer FE"), ("Family FE", "Family FE"),
            ("Lab x period FE", "Lab $\\times$ period FE"), ("OP-style selection control", "OP-style control")]
    L = ["\\begin{table}[tp]", "\\centering",
         "\\caption{Design-Matched Benchmark: Other Outputs and a Cleaner Sample}",
         "\\label{tab:app-lalonde-robust}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         " & $\\theta_N$ & $\\theta_D$ & \\multicolumn{3}{c}{$\\theta_C$ (regression on $\\ln C$ only)} \\\\",
         "\\cmidrule(l){4-6}",
         "Estimator & Bias & Bias & Observational & Experimental & Bias \\\\", "\\midrule"]
    panels = [("arc", "Panel A. ARC-Challenge log odds (57 models)"),
              ("wino", "Panel B. Winogrande log odds (57 models)"),
              ("clean", "Panel C. HellaSwag log odds, restricted sample (49 models)")]
    for key, title in panels:
        d = pd.read_csv(T / f"m4_observational_table6_lalonde_{key}.csv")
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for e, lab in ests:
            g = d[d.estimator == e].set_index("param")
            r1 = [lab, f(g.loc["theta_N", "bias"]), f(g.loc["theta_D", "bias"]), f(g.loc["theta_C", "obs"]),
                  f(g.loc["theta_C", "bench"]), f(g.loc["theta_C", "bias"])]
            r2 = ["", f"({f(g.loc['theta_N', 'bias_se'])})", f"({f(g.loc['theta_D', 'bias_se'])})",
                  f"({f(g.loc['theta_C', 'obs_se'])})", f"({f(g.loc['theta_C', 'bench_se'])})",
                  f"({f(g.loc['theta_C', 'bias_se'])})"]
            L.append(" & ".join(r1) + " \\\\")
            L.append(" & ".join(r2) + " \\\\")
        if key != "clean":
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    wb = pd.read_csv(T / "m4_observational_table6_lalonde_clean_wildboot.csv")
    ols = wb[(wb.estimator == "OLS") & (wb.param == "theta_C")].iloc[0]
    L.append(
        "\\begin{tablenotes}Same design-matched comparison as Table~\\ref{tab:lalonde}. Panels A and B: 57 models from "
        "19 developers; Panel C excludes Qwen1.5 (token counts imputed by size), distilled, synthetic-data, and code "
        "models (49 models, 16 developers). Each estimator is applied to "
        "the observed benchmark log odds of the models inside the convex hull of the experimental runs "
        "(``Observational'') and, on the same models, to counterfactual outputs from the experimental technology, a "
        "quadratic surface fitted to the OLMo ladder and the \\citet{gadre2024language} testbed "
        "(``Experimental''). Bias $=$ Observational $-$ Experimental. Standard errors in parentheses: clustered by "
        "developer for the observational estimates, from 300 bootstrap refits of the surface for the benchmark; the "
        "bias standard error treats the two as independent. With 16 to 19 developer clusters these standard errors "
        f"can overstate precision; in Panel C the wild-cluster bootstrap $p$-value of the OLS $\\theta_C$ bias is "
        f"{ols.p_wild:.2f} (normal $p$ {ols.p_cr1_normal:.2f}). $\\theta_N$ and $\\theta_D$ are the elasticities "
        "of the success odds with respect to parameters and tokens; the OP-style row controls for a quadratic in the "
        "estimated probability of Epoch notability.\\end{tablenotes}")
    L.append("\\begin{tablenotes}[Source]Open LLM Leaderboard evaluations via \\citet{ruan2024observational} and "
             "\\citet{maiapolo2024sloth}; OLMo ladder \\citep{bhagia2024establishing}; \\citet{gadre2024language}; "
             "authors' estimates.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appD_lalonde_robust.tex").write_text("\n".join(L) + "\n")


def table_attenuation():
    A = pd.read_csv(T / "m5_progress_attenuation_panelA.csv")
    B = pd.read_csv(T / "m5_progress_attenuation_panelB.csv")
    C = pd.read_csv(T / "m5_progress_attenuation_mc.csv")

    def tc(x):
        return "--" if (x is None or np.isnan(x)) else f"{x:.1f}"

    def tcci(lo, hi):
        hi_s = "$\\infty$" if (np.isnan(hi) or hi > 1e3 or hi < 0) else f"{hi:.1f}"
        return f"[{lo:.1f}, {hi_s}]"

    labs = {
        "A0": "Baseline (model 7)", "A1": "Benchmark-specific perplexity",
        "a1": "(a) Tokens seen, epochs $=1$", "a2": "(a) Tokens seen, epochs imputed",
        "a3": "(a) Effective data", "a4": "(a) Word-level vocabulary",
        "a5": "(a) Vocabulary control", "b1": "(b) Published 2018 or later", "b2": "(b) Transformers only",
        "b3": "(b) No year terms", "c1": "(c) Benchmark-specific $E$",
        "d1": "(d) All models per paper", "d2": "(d) Top model per paper", "d3": "(d) Flagged outliers kept",
        "d4": "(d) No zero-shot evaluations",
    }
    L = ["\\begin{table}[tp]", "\\centering",
         "\\caption{Why Is the Cross-Lab Data Elasticity Small? Channels of Attenuation}",
         "\\label{tab:app-attenuation}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         " & & & & & & \\multicolumn{2}{c}{Default stop} \\\\", "\\cmidrule(l){7-8}",
         " & $n$ & $\\hat\\alpha$ & $\\hat\\beta$ & $\\hat\\gamma$ & $T_C$ [95\\% CI] & $\\hat\\beta$ & $T_C$ \\\\",
         "\\midrule",
         "\\multicolumn{8}{@{}l}{\\textit{Panel A. Cross-lab model on alternative data}} \\\\"]
    for _, x in A.iterrows():
        tcs = "--" if x.code == "b3" else f"{tc(x.TC_months)} {tcci(x.TC_lo, x.TC_hi)}"
        if abs(x.beta_data) < 1e-6:  # beta_data = 0: g_D and hence T_C undefined
            tcs = "n.d."
        tcd = "--" if x.code == "b3" else tc(x.TC_default)
        L.append(" & ".join([labs[x.code], f"{int(x.n)}", f(x.alpha_param), f(x.beta_data), f(x.gamma), tcs,
                             f(x.beta_default), tcd]) + " \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{8}{@{}l}{\\textit{Panel B. Chinchilla sweep}} \\\\")
    bl = ["$E$ estimated, Huber", "$E=0$, Huber", "$E=0$, least squares", "Total-loss elasticities"]
    for (_, x), lab in zip(B.iterrows(), bl):
        g = f(x.gamma)
        if pd.notna(x.gamma_lo):
            g += f" [{f(x.gamma_lo)}, {f(x.gamma_hi)}]"
        L.append(" & ".join([lab, "240", f(x.alpha), f(x.beta), "\\multicolumn{2}{l}{" + g + "}", "", ""]) + " \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{8}{@{}l}{\\textit{Panel C. Monte Carlo on the cross-lab design}} \\\\")
    L.append(" & & \\multicolumn{3}{c}{Global optimum} & MC & \\multicolumn{2}{c}{Authors' code} \\\\")

    L.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\\cmidrule(l){7-8}")
    L.append(" & & $\\hat\\alpha$ & $\\hat\\beta$ & $\\hat\\gamma$ & $T_C$ [5, 95] & $\\hat\\beta$ & $T_C$ \\\\")
    cl = {"S0": "Correct specification", "S1": "(c) $E$ dropped", "S1h": "(c) and neutrality imposed",
          "S2": "(c)+(a) $D$ = dataset size", "S3": "(c)+(a)+(d) best 3",
          "S3r": "(c)+(a) random 3", "S4": "(c)+(a), $\\rho(\\psi_D,\\ln D)=-0.5$",
          "S5": "(c)+(a), $\\rho(\\psi_D,\\ln D)=+0.5$"}
    for _, x in C.iterrows():
        L.append(" & ".join([cl[x.code], "", f(x.alpha_param_gl_pseudo), f(x.beta_data_gl_pseudo),
                             f(x.gamma_gl_pseudo),
                             f"{x.TC_gl_mc_med:.1f} [{x.TC_gl_mc_lo:.1f}, {x.TC_gl_mc_hi:.1f}]",
                             f(x.beta_data_ho_pseudo), f"{x.TC_ho_pseudo:.1f}"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    L.append(
        "\\begin{tablenotes}Panel A: the model and penalized objective of \\citet{ho2024algorithmic} (mean squared "
        "error plus $0.0025\\sum|\\theta|$, $E=0$), minimized to convergence on the indicated data; the last two "
        "columns (default stop) stop the optimizer at the default tolerance used in the authors' code. $T_C$ is the doubling time of "
        "effective compute, with a 95 percent paper-cluster bootstrap interval (400 draws, each started at zero; "
        "``$\\infty$'' when nonpositive growth is inside the interval). Rows: (a) measurement of data, with $D$ as "
        "tokens seen (missing epoch counts set to 1 or imputed from dataset size), as effective data under repetition "
        "\\citep[$R^*=15.4$;][]{muennighoff2023scaling}, or with word-level vocabularies only or a vocabulary control; "
        "33 percent of epoch counts are imputed; (b) time and collinearity; (c) benchmark-specific irreducible loss "
        "estimated by NLS; (d) selection, using all models of each paper, only the best, keeping flagged outliers, or "
        "dropping zero-shot evaluations. Panel B: the "
        "\\citet{besiroglu2024chinchilla} estimator on the Chinchilla sweep with $E$ estimated or set to zero, with "
        "95 percent bootstrap intervals for $\\gamma=\\alpha\\beta/(\\alpha+\\beta)$; the last row gives the sample "
        "means of the total-loss elasticities $\\alpha u/L$, $\\beta v/L$, and $\\gamma(L-E)/L$ implied by the full "
        "model. Panel C: pseudo-true values (at the global optimum and with the authors' code) from noise-free data when the truth is the Besiroglu technology with "
        "Hicks-neutral progress ($T_C=12$ months) evaluated at the actual inputs of the sample; MC median and "
        "5th--95th percentiles of $T_C$ from 204 replications with noise of standard deviation 0.215; rows switch on "
        "(c) dropping $E$, (a) measuring $D$ as dataset size, (d) keeping the best three models per paper, and a "
        "data-augmenting productivity shock $\\psi_D$ correlated with $\\ln D$. n.d.: not "
        "defined because $\\hat\\beta=0$.\\end{tablenotes}")
    L.append("\\begin{tablenotes}[Source]Data of \\citet{ho2024algorithmic}; Epoch AI digitization of "
             "\\citet{hoffmann2022training}; authors' estimates.\\end{tablenotes}")
    L.append("\\end{table}")
    (OUT / "appD_attenuation.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    table_selection()
    table_homothetic()
    table_lalonde_robust()
    table_attenuation()
    print("wrote", sorted(p.name for p in OUT.glob("appD_*.tex")))
