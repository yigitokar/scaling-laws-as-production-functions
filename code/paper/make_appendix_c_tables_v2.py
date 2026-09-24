"""Online Appendix C tables, version 2 (2026-09-24; Monte Carlo re-run with no estimator started at the truth).

Writes paper/tables/appC_designs.tex      (Table C1. Design A: bias/RMSE, diagnostics and inference; v2 numbers)
       paper/tables/appC_starts.tex       (Table C2. Design A: v1 start set vs nine and 64 random starts)
       paper/tables/appC_noise.tex        (Table C3. Design A: noise level, digitization, heteroskedastic/clustered noise)
       paper/tables/appC_industry.tex     (Table C4. Design B; v2 numbers, estimators started from data-based grids)
       paper/tables/appC_transmission.tex (Table C5. Formula check; label updated to the v2 appendix)
Every number is read from output/tables/m6_montecarlo_*.csv (v1 numbers for Table C2 from
data/processed/m6_montecarlo/v1_backup/tables); no hand-typed estimates.
Supersedes the appC part of code/paper/make_appendix_cd_tables.py.
Run: .venv/bin/python code/paper/make_appendix_c_tables_v2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "output" / "tables"
V1 = ROOT / "data" / "processed" / "m6_montecarlo" / "v1_backup" / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
SRC = ("\\begin{tablenotes}[Source]Authors' simulations; replication code "
       "\\texttt{code/analysis/m6\\_montecarlo}.\\end{tablenotes}")


def f(x, d=3):
    """Format with a true minus sign; '--' for missing; no negative zero."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = s[1:] if float(s) == 0 else "$-$" + s[1:]
    return s


def pc(x):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.0f}"


S = pd.read_csv(T / "m6_montecarlo_designA_summary.csv")
P = pd.read_csv(T / "m6_montecarlo_designA_profile.csv").set_index("cell")
BC = pd.read_csv(T / "m6_montecarlo_designA_bootcheck.csv")
S1 = pd.read_csv(V1 / "m6_montecarlo_designA_summary.csv")
SC = pd.read_csv(T / "m6_montecarlo_designA_startcheck.csv").set_index("cell")
GRID_LO, GRID_HI = 0.05, 0.99


def get(cell, est, par, col, s=S):
    r = s[(s.cell == cell) & (s.estimator == est) & (s.param == par)]
    return float(r[col].iloc[0]) if len(r) and col in r and pd.notna(r[col].iloc[0]) else np.nan


def ci_bounds(cell):
    lo, hi, w = (float(P.loc[cell, k]) for k in ("median_ci_lo", "median_ci_hi", "median_ci_width"))
    d = 3 if w < 0.1 else 2
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def rm(x, big=10):
    """RMSE formatting: 4 decimals below 0.01, 3 below 1, 2 below 10, 1 above."""
    if not np.isfinite(x):
        return "--"
    return f(x, 4 if x < 0.01 else 3 if x < 1 else 2 if x < big else 1)


# ---------------------------------------------------------------------------------------------
# Table C1: Design A
# ---------------------------------------------------------------------------------------------
def table_designs():
    cells = [("onpath", "(i) On path, $v=0.02$"), ("iso16", "(ii) IsoFLOP, $\\pm16\\times$"),
             ("iso4", "(ii$'$) IsoFLOP, $\\pm4\\times$"), ("fact", "(iii) Factorial grid"),
             ("opt_s0", "(iv) Optimizing, $v=0$"), ("opt_s0.1", "(iv) Optimizing, $v=0.1$"),
             ("opt_s0.3", "(iv) Optimizing, $v=0.3$"), ("opt_s1", "(iv) Optimizing, $v=1$"),
             ("kaplan", "(v) Kaplan beliefs")]
    L = ["\\begin{table}[tp]", "\\centering",
         "\\caption{Monte Carlo Design A: What Alternative Designs Identify at Equal Compute}",
         "\\label{tab:app-mc-designs}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.8pt}",
         "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & \\multicolumn{4}{c}{Primal estimator ($\\kappa=1$)} & \\multicolumn{2}{c}{Dual estimator} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(l){6-7}",
         "Design & $a$ & $\\gamma$ & $\\sigma^*$ & $\\ln M^*(10^{24})$ & $\\sigma^*$ & $\\ln M^*(10^{24})$ \\\\",
         "\\midrule", "\\multicolumn{7}{@{}l}{\\textit{Panel A. Bias [root-mean-squared error]}} \\\\[1pt]"]
    for cell, lab in cells:
        r1, r2 = [lab], [""]
        for par, d in [("a", 3), ("gamma", 3), ("sigma_star", 3), ("lnMstar", 2)]:
            rmse = get(cell, "primal", par, "rmse")
            bias = get(cell, "primal", par, "bias")
            r1.append(f(bias, 1 if (par == "lnMstar" and abs(bias) >= 10) else d))
            r2.append("[" + f(rmse, 1 if (par == "lnMstar" and rmse >= 10) else d) + "]")
        for par, d in [("sigma_star", 3), ("lnMstar", 2)]:
            b, r = get(cell, "dual", par, "bias"), get(cell, "dual", par, "rmse")
            r1.append(f(b, d))
            r2.append("--" if np.isnan(r) else "[" + f(r, d) + "]")
        L += [" & ".join(r1) + " \\\\", " & ".join(r2) + " \\\\[1pt]"]
    L += ["\\midrule",
          "\\multicolumn{7}{@{}l}{\\textit{Panel B. Inference for $\\sigma^*$ and the profile likelihood with $\\kappa$ free}} \\\\[1pt]",
          " & Corner & Wald & Bootstrap & \\multicolumn{3}{c}{95\\% profile set, $\\kappa$ free} \\\\",
          "\\cmidrule(l){5-7}",
          " & fits (\\%) & coverage (\\%) & coverage (\\%) & Median bounds & Flat (\\%) & Flat$'$ (\\%) \\\\",
          "\\cmidrule(l){2-7}"]
    for cell, lab in cells:
        wc, wcomp = get(cell, "primal", "sigma_star", "wald_cover"), get(cell, "primal", "sigma_star", "wald_computable")
        bcv = get(cell, "primal", "sigma_star", "boot_cover")
        wtxt = pc(wc) + ("$^{\\dagger}$" if wcomp < 0.95 else "")
        btxt = "--" if np.isnan(bcv) else pc(bcv) + ("$^{\\S}$" if cell in ("onpath", "opt_s0.3") else "")
        L.append(" & ".join([lab, pc(get(cell, "primal", "sigma_star", "corner_share")), wtxt, btxt, ci_bounds(cell),
                             pc(float(P.loc[cell, "share_flat_everywhere"])),
                             pc(float(P.loc[cell, "share_flat_on_v1_grid"]))]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    on = BC[(BC.cell == "onpath") & (BC.param == "sigma_star")].iloc[0]
    s3 = BC[(BC.cell == "opt_s0.3") & (BC.param == "sigma_star")].iloc[0]
    R_bc, B_bc = int(BC.R.iloc[0]), int(BC.B.iloc[0])
    wcomp_on = get("onpath", "primal", "sigma_star", "wald_computable")
    note = (
        "\\begin{tablenotes}"
        "Each design has 90 runs on nine budgets matching Chinchilla's nominal IsoFLOP budgets "
        "($6\\times10^{18}$ to $3\\times10^{21}$ FLOP), and the same total compute, "
        "$\\sum 6ND=5.1\\times10^{22}$ FLOP; 500 replications per design. The truth is the "
        "\\citet{besiroglu2024chinchilla} technology ($a=0.513$, $\\gamma=0.178$, $\\sigma^*=0.737$, "
        "$\\ln M^*(10^{24})=2.90$); log-loss noise is normal with standard deviation 0.0075. Designs: (i) runs "
        "with allocation errors of standard deviation $v=0.02$ around the expansion path; (ii) ten sizes log-spaced "
        "between $N^*/16$ and $16N^*$ (or $N^*/4$ and $4N^*$) at each budget; (iii) a $9\\times10$ grid in $(N,D)$ "
        "spanning $256\\times$ in each input; (iv) optimizing labs with allocation errors of standard deviation $v$ "
        "in $\\ln(D/N)$ at given compute; (v) labs that follow Kaplan's rule $N\\propto C^{0.73}$ plus errors with "
        "$v=0.1$. Primal: Huber ($\\delta=10^{-3}$) fit of $\\ln L$ from nine starting values drawn at random over "
        "the region of \\citet{hoffmann2022training}'s initialization grid, none at the truth, inside a parameter "
        "box (Table~\\ref{tab:app-mc-starts} compares start sets). Dual: $a$ from the (noisy) optima or IsoFLOP "
        "minima, $\\gamma$ from the frontier (eight starting values on a grid), $\\alpha=\\gamma/a$, "
        "$\\beta=\\gamma/(1-a)$; not computed for the factorial grid. Panel B: corner = estimate on the box "
        "boundary or one power-law term below 2 percent of reducible loss at every design point; Wald = "
        "LAD-sandwich delta-method 95 percent interval ($^{\\dagger}$: covariance singular or corner in more than "
        f"5 percent of replications, {pc(1 - wcomp_on)} percent on the path; coverage among computable ones); "
        "bootstrap = pairs-bootstrap percentile interval (149 draws, first 150 replications), each draw "
        f"warm-started at the replication's estimate ($^{{\\S}}$: in a separate check with {R_bc} replications and "
        f"{B_bc} draws, restarting every draw also from the nine starting values moves on-path coverage from "
        f"{pc(on.warm_cover)} to {pc(on.multi_cover)} percent and the median width from "
        f"{on.warm_median_width:.2f} to {on.multi_median_width:.2f}; at $v=0.3$ both give "
        f"{pc(s3.multi_cover)} percent); 95 percent profile set = likelihood-ratio confidence set for $\\sigma^*$ in "
        "$L=E+(AN^{-a_1}+BD^{-b_1})^{\\kappa}$ on a 29-point grid over $[0.05,0.99]$, with the likelihood ratio "
        "measured against the unrestricted optimum (first 150 replications): medians of the lower and upper "
        "bounds, share of replications whose set is the whole grid (flat) and share whose set contains all of "
        "$[0.50,0.95]$, the range searched in the first version (flat$'$). Monte Carlo standard errors of coverage "
        "rates near 95 percent are 1.0, 1.8 and 2.8 percentage points with 500, 150 and 60 replications."
        "\\end{tablenotes}")
    L += [note, SRC, "\\end{table}"]
    (OUT / "appC_designs.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------
# Table C2: starting values
# ---------------------------------------------------------------------------------------------
def table_starts():
    cells = [("onpath", "On path, $v=0.02$"), ("opt_s0", "Optimizing, $v=0$"), ("opt_s0.05", "Optimizing, $v=0.05$"),
             ("opt_s0.1", "Optimizing, $v=0.1$"), ("opt_s0.2", "Optimizing, $v=0.2$"), ("kaplan", "Kaplan beliefs")]
    L = ["\\begin{table}[tp]", "\\centering",
         "\\caption{Monte Carlo Design A: Sensitivity of the Primal Estimator to Its Starting Values}",
         "\\label{tab:app-mc-starts}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{Truth $+$ 8 random (first version)} & \\multicolumn{3}{c}{9 random (baseline)}"
         " & \\multicolumn{3}{c}{64 random} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
         " & Corner & \\multicolumn{2}{c}{RMSE} & Corner & \\multicolumn{2}{c}{RMSE} & Corner & \\multicolumn{2}{c}{RMSE} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}\\cmidrule(l){9-10}",
         "Design & (\\%) & $\\sigma^*$ & $\\ln M^*$ & (\\%) & $\\sigma^*$ & $\\ln M^*$ & (\\%) & $\\sigma^*$ & $\\ln M^*$ \\\\",
         "\\midrule"]
    for cell, lab in cells:
        v1 = [pc(get(cell, "primal", "sigma_star", "corner_share", S1)), rm(get(cell, "primal", "sigma_star", "rmse", S1)),
              rm(get(cell, "primal", "lnMstar", "rmse", S1))]
        v2 = [pc(get(cell, "primal", "sigma_star", "corner_share")), rm(get(cell, "primal", "sigma_star", "rmse")),
              rm(get(cell, "primal", "lnMstar", "rmse"))]
        r = SC.loc[cell]
        v64 = [pc(r.s64_corner_share), rm(r.s64_sigma_star_rmse), rm(r.s64_lnMstar_rmse)]
        L.append(" & ".join([lab] + v1 + v2 + v64) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    ab = [("0" if SC.loc[c, "s9_share_obj_above_truth"] == 0 else f"{100 * SC.loc[c, 's9_share_obj_above_truth']:.1f}") for c, _ in cells[:4]]
    above9 = ", ".join(ab[:3]) + " and " + ab[3]
    note = (
        "\\begin{tablenotes}"
        "Design A (Table~\\ref{tab:app-mc-designs}), 500 replications per design, identical simulated data in every "
        "column. Primal estimator: Huber ($\\delta=10^{-3}$) fit of $\\ln L$ with $\\kappa=1$, minimized by L-BFGS-B "
        "from each starting value in the set; the lowest objective is kept. First version: the true parameters plus "
        "eight values drawn at random over the region of \\citet{hoffmann2022training}'s initialization grid "
        "($\\ln A,\\ln B\\in[0,25]$, $\\ln E\\in[-1,1]$, $\\alpha,\\beta\\in[0.1,1.2]$). Baseline (all other tables "
        "and Figure~\\ref{fig:designs}): the same eight plus a ninth random value. 64 random: the baseline nine "
        "plus 55 further random values. Corner = estimate on the box boundary or one power-law term below 2 percent "
        "of reducible loss. RMSE of $\\hat\\sigma^*$ and of $\\ln\\hat M^*(10^{24})$. The objective at the estimate "
        f"exceeds the objective at the true parameters in {above9} percent of replications (on path, $v=0$, 0.05 "
        "and 0.1) with nine random starts, and in none with 64. For the IsoFLOP and factorial designs and for "
        "$v\\ge0.2$ the first-version and baseline estimates are identical up to optimizer tolerance "
        "(Table~\\ref{tab:app-mc-designs})."
        "\\end{tablenotes}")
    L += [note, SRC, "\\end{table}"]
    (OUT / "appC_starts.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------
# Table C3: Design A under alternative noise
# ---------------------------------------------------------------------------------------------
def table_noise():
    rows = [
        ("Normal, sd 0.005", [("iso16_sd0.005", "iso"), ("opt_s0.3_sd0.005", "s03")]),
        ("Normal, sd 0.0075 (baseline)", [("onpath", "on"), ("iso16", "iso"), ("fact", "fact"), ("opt_s0.1", "s01"),
                                          ("opt_s0.3", "s03"), ("opt_s1", "s1")]),
        ("Normal, sd 0.015", [("iso16_sd0.015", "iso"), ("opt_s0.3_sd0.015", "s03")]),
        ("Digitized as in the Epoch extraction", [("iso16_digit", "iso")]),
        ("Heteroskedastic and clustered", [("onpath_hc", "on"), ("iso16_hc", "iso"), ("fact_hc", "fact"),
                                           ("opt_s0.1_hc", "s01"), ("opt_s0.3_hc", "s03"), ("opt_s1_hc", "s1")]),
    ]
    dname = {"on": "On path, $v=0.02$", "iso": "IsoFLOP, $\\pm16\\times$", "fact": "Factorial grid",
             "s01": "Optimizing, $v=0.1$", "s03": "Optimizing, $v=0.3$", "s1": "Optimizing, $v=1$"}
    L = ["\\begin{table}[tp]", "\\centering",
         "\\caption{Monte Carlo Design A: Sensitivity to the Noise Process}",
         "\\label{tab:app-mc-noise}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{Primal estimator ($\\kappa=1$)} & \\multicolumn{4}{c}{95\\% profile set, $\\kappa$ free} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-8}",
         " & RMSE & RMSE & Wald & Covers & Flat & Flat$'$ & Median \\\\",
         "Design & $\\sigma^*$ & $\\ln M^*(10^{24})$ & coverage (\\%) & truth (\\%) & (\\%) & (\\%) & width \\\\",
         "\\midrule"]
    for k, (title, cl) in enumerate(rows):
        L.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{Noise: {title}}}}} \\\\[1pt]")
        for cell, d in cl:
            rm_s, rm_m = get(cell, "primal", "sigma_star", "rmse"), get(cell, "primal", "lnMstar", "rmse")
            wc, wcomp = get(cell, "primal", "sigma_star", "wald_cover"), get(cell, "primal", "sigma_star", "wald_computable")
            L.append(" & ".join([dname[d], rm(rm_s), rm(rm_m),
                                 pc(wc) + ("$^{\\dagger}$" if wcomp < 0.95 else ""),
                                 pc(float(P.loc[cell, "cover_truth"])), pc(float(P.loc[cell, "share_flat_everywhere"])),
                                 pc(float(P.loc[cell, "share_flat_on_v1_grid"])),
                                 f"{float(P.loc[cell, 'median_ci_width']):.3f}"]) + " \\\\")
        if k < len(rows) - 1:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip"]
    note = (
        "\\begin{tablenotes}"
        "Design A (Table~\\ref{tab:app-mc-designs}) under alternative noise processes for $\\ln L$; 500 "
        "replications per cell for the primal estimator and 150 for the profile likelihood. Normal: independent "
        "normal noise with the stated standard deviation; 0.0075 is the residual standard deviation of the "
        "\\citet{besiroglu2024chinchilla} fit on the Chinchilla extraction and 0.005 is close to its MAD-based "
        "value (0.0049). Digitized: losses rounded to a 0.012-nat grid, 2 percent pixel error in $N$ and $C$, and "
        "$D$ imputed as $C/(6N)$. Heteroskedastic and clustered: the standard deviation for run $i$ is "
        "proportional to $(N_iD_i)^{-1/4}$, larger for small models and for short runs, and scaled so that the "
        "design-average variance equals $0.0075^2$; at a given budget the two effects offset, so the standard "
        "deviation falls with compute (4.7 times from the smallest to the largest budget) and, in the factorial grid, "
        "varies along both inputs. Half of each run's variance is a shock shared by its cluster (correlation 0.5): "
        "the nine compute budgets (on-path, optimizing and IsoFLOP designs; shared data order) or the nine model "
        "sizes (factorial grid; one training trunk per size). Wald: LAD-sandwich interval assuming independent "
        "errors ($^{\\dagger}$: singular or corner in more than 5 percent of replications). Covers truth: share of "
        "replications whose 95 percent profile set, calibrated by $\\chi^2_1$ as if noise were independent, contains "
        "the true $\\sigma^*$. Flat: set is the whole grid $[0.05,0.99]$; flat$'$: set contains all of $[0.50,0.95]$."
        "\\end{tablenotes}")
    L += [note, SRC, "\\end{table}"]
    (OUT / "appC_noise.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------
# Table C4: Design B (industry); Table C5: transmission formula (numbers from the module CSVs)
# ---------------------------------------------------------------------------------------------
def tables_industry_transmission():
    import importlib.util
    spec = importlib.util.spec_from_file_location("old", ROOT / "code" / "paper" / "make_appendix_cd_tables.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.table_industry()
    old.table_transmission()
    p = OUT / "appC_transmission.tex"
    p.write_text(p.read_text().replace("Proposition~\\ref{prop:transmission}", "Proposition~\\ref{prop:A-transmission}"))
    p = OUT / "appC_industry.tex"
    txt = p.read_text()
    add = (" Every estimator is started from data-based values, none at the truth: nine random values for the "
           "pooled Huber fit, and nine combinations of $\\alpha,\\beta\\in\\{0.2,0.5,1\\}$ with levels matched to "
           "mean output for the nonlinear least-squares estimators, keeping the lowest criterion; ACF starts from "
           "pooled NLS and the FOC systems from the lab $\\times$ generation fit.")
    anchor = "TFP growth: mean change in generation means of $y-\\hat F(n,d)$."
    assert anchor in txt
    txt = txt.replace(anchor, anchor + add)
    p.write_text(txt)


if __name__ == "__main__":
    table_designs()
    table_starts()
    table_noise()
    tables_industry_transmission()
    print("wrote", sorted(p.name for p in OUT.glob("appC_*.tex")))
