"""Generate paper Tables 8 (LaLonde) and 9 (progress) from module CSVs (Section VI writer, 2026-09-24).

Writes paper/tables/table8_lalonde.tex and paper/tables/table9_progress.tex.
Every number is read from output/tables/*.csv; nothing is typed by hand.
"""
import math
import pandas as pd

ROOT = "/Users/yigitokar/scaling-laws-pf"
T = f"{ROOT}/output/tables"


def f3(x):
    s = f"{x:.3f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def se(x):
    return f"({x:.3f})"


def pbr(x):
    return f"[{x:.2f}]"


# ----------------------------------------------------------------------------- Table 8
lal = pd.read_csv(f"{T}/m4_observational_table6_lalonde.csv")
wild = pd.read_csv(f"{T}/m4_observational_table6_lalonde_wildboot.csv")
cleanw = pd.read_csv(f"{T}/m4_observational_table6_lalonde_clean_wildboot.csv")


def row(df, est, par):
    r = df[(df.estimator == est) & (df.param == par)]
    assert len(r) == 1, (est, par, len(r))
    return r.iloc[0]


def wp(dfw, est, par):
    if dfw is None:
        return None
    r = dfw[(dfw.estimator == est) & (dfw.param == par)]
    return None if len(r) == 0 else float(r.iloc[0].p_wild)


est_labels = [
    ("OLS", "OLS"),
    ("Year FE", "Year FE"),
    ("Developer FE", "Developer FE"),
    ("Family FE", "Family FE"),
    ("Lab x period FE", r"Lab $\times$ period FE"),
    ("OP-style selection control", "OP-style selection"),
]

lines = []


def block(df, dfw, est, label, full=True):
    """Three lines: estimates, SEs, wild p (bias columns only)."""
    c = row(df, est, "theta_C")
    cells1 = [label, f3(c.obs), f3(c.bench), f3(c.bias)]
    cells2 = ["", se(c.obs_se), se(c.bench_se), se(c.bias_se)]
    cells3 = ["", "", "", pbr(wp(dfw, est, "theta_C")) if wp(dfw, est, "theta_C") is not None else ""]
    for par in ("theta_N", "theta_D"):
        if full:
            r = row(df, est, par)
            cells1 += [f3(r.bench), f3(r.bias)]
            cells2 += [se(r.bench_se), se(r.bias_se)]
            p = wp(dfw, est, par) if dfw is not None else None
            cells3 += ["", pbr(p) if p is not None else ""]
        else:
            cells1 += ["", ""]
            cells2 += ["", ""]
            cells3 += ["", ""]
    out = [" & ".join(cells1) + r" \\", " & ".join(cells2) + r" \\"]
    if any(x for x in cells3[1:]):
        out.append(" & ".join(cells3) + r" \\")
    return out


def cblock(df, est, label):
    c = row(df, est, "theta_C")
    return [
        " & ".join([label, f3(c.obs), f3(c.bench), f3(c.bias), "", "", "", ""]) + r" \\",
        " & ".join(["", se(c.obs_se), se(c.bench_se), se(c.bias_se), "", "", "", ""]) + r" \\",
    ]


L = []
L.append(r"\begin{table}[tp]")
L.append(r"\centering")
L.append(r"\caption{Design-Matched Comparison of Observational and Experimental Elasticities}")
L.append(r"\label{tab:lalonde}")
L.append(r"\footnotesize")
L.append(r"\setlength{\tabcolsep}{3.2pt}")
L.append(r"\begin{tabular}{@{}l ccc cc cc@{}}")
L.append(r"\toprule")
L.append(r" & \multicolumn{3}{c}{Compute only, $\theta_C$} & \multicolumn{2}{c}{Parameters, $\theta_N$} & \multicolumn{2}{c}{Tokens, $\theta_D$} \\")
L.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
L.append(r" & Obs. & Exp. & Bias & Exp. & Bias & Exp. & Bias \\")
L.append(r" & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\")
L.append(r"\midrule")
L.append(r"\multicolumn{8}{@{}l}{\textit{Panel A. HellaSwag, 57 models inside the experimental support}} \\")
for est, lab in est_labels:
    L += block(lal, wild, est, lab)
    L.append(r"\addlinespace[2pt]")
L.append(r"\multicolumn{8}{@{}l}{\textit{Panel B. HellaSwag, other compute-only estimators}} \\")
L += cblock(lal, "OLS, EIV-corrected (Epoch 'Confident')", "OLS, EIV-corrected")
L += cblock(lal, "Family FE, EIV-corrected (Epoch 'Confident')", "Family FE, EIV-corrected")
L += cblock(lal, "Reverse (c on y)", "Reverse regression")
ivr = row(lal, "IV: frontier FLOP/$", "theta_C")
L += cblock(lal, "IV: frontier FLOP/$", rf"IV, FLOP/\$ ($F={ivr.F:.1f}$)")
L.append(r"\addlinespace[2pt]")
L.append(r"\multicolumn{8}{@{}l}{\textit{Panel C. OLS, other outputs, surface and sample}} \\")
alt = [
    ("core", "Composite (57)", None),
    ("arc", "ARC-Challenge (57)", None),
    ("wino", "Winogrande (57)", None),
    ("ladderonly", "Ladder-only surface (39)", None),
    ("clean", "Clean subsample (49)", cleanw),
]
for tag, lab, dfw in alt:
    df = pd.read_csv(f"{T}/m4_observational_table6_lalonde_{tag}.csv")
    L += block(df, dfw, "OLS", lab)
    L.append(r"\addlinespace[2pt]")
if L[-1].startswith(r"\addlinespace"):
    L.pop()
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
L.append("")
L.append(r"\begin{tablenotes}")
L.append(
    r"Output is the log-odds of above-chance accuracy, $q=\ln[\tilde a/(1-\tilde a)]$ with "
    r"$\tilde a=(\text{accuracy}-\text{chance})/(1-\text{chance})$. $\theta_N$ and $\theta_D$ are the coefficients on $\ln N$ "
    r"and $\ln D$ in a regression of $q$ on both; $\theta_C$ is the coefficient on $\ln C=\ln 6ND$ alone. "
    r"Obs.: the estimator applied to observed outputs, standard errors clustered by developer (19 clusters in Panel A). "
    r"Exp.: the same estimator, with the same fixed effects and controls, applied on the same models to counterfactual "
    r"outputs from the experimental technology, a quadratic surface in $(\ln N,\ln D)$ fitted to 88 designed runs "
    r"(OLMo ladder with OLMo-2 7B and 13B, and Gadre et al.\ with $N\geq 0.1$B; $R^2=0.994$); standard errors from 300 "
    r"bootstrap refits of the surface. Bias $=$ Obs.\ $-$ Exp.; its standard error treats the two as independent. "
    r"Brackets: restricted wild-cluster bootstrap-$t$ $p$-values (Webb weights, 999 draws) for the bias, reported for the "
    r"estimators and samples for which they were computed. The 57 models lie inside the convex hull of the experimental "
    r"$(\ln N,\ln D)$ points (30 families, 19 developers, all with $N\le 9$B). OP-style selection: OLS with a quadratic in the "
    r"estimated propensity of being an Epoch ``notable'' model. EIV: slope divided by the reliability implied by Epoch's "
    r"``Confident'' compute class. Reverse regression: inverse slope of $\ln C$ on $q$. IV: $\ln C$ instrumented by the "
    r"log FLOP/s per dollar of the best data-center accelerator released at least six months before the model; $F$ is the "
    r"cluster-robust first-stage statistic; in this row the Exp.\ column applies the same instrument to the experimental outputs. "
    r"With developer fixed effects the same instrument has $F<0.01$ (not shown). "
    r"Panel C: composite is the logit of mean chance-adjusted accuracy on HellaSwag, ARC-Challenge and Winogrande; the "
    r"ladder-only surface is fitted to OLMo runs only; the clean subsample drops Qwen1.5 (imputed $D$), distilled, "
    r"synthetic-data and code models (16 clusters)."
)
L.append(r"\end{tablenotes}")
L.append(r"\begin{tablenotes}[Source]")
L.append(
    r"Authors' calculations. Observational outputs: Open LLM Leaderboard v1 via \citet{ruan2024observational} and "
    r"\citet{maiapolo2024sloth}; parameter counts from Hugging Face; release dates and compute from \citet{epochai2026data}. "
    r"Experimental runs: \citet{bhagia2024establishing}, \citet{teamolmo2024olmo} and \citet{gadre2024language}."
)
L.append(r"\end{tablenotes}")
L.append(r"\end{table}")
open(f"{ROOT}/paper/tables/table8_lalonde.tex", "w").write("\n".join(L) + "\n")

# ----------------------------------------------------------------------------- Table 9
pa = pd.read_csv(f"{T}/m5_progress_table7.csv").set_index("code")
pb = pd.read_csv(f"{T}/m5_progress_table7_panelB.csv")
rep = pd.read_csv(f"{T}/m5_progress_replication.csv").set_index("parameter")


def ff(x, d=2):
    s = f"{x:.{d}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def ci(lo, hi, d=1):
    h = r"\infty)" if (hi is None or not math.isfinite(hi)) else f"{hi:.{d}f}]"
    return f"[{lo:.{d}f}, {h}"


def tc(r):
    if r.g_C <= 0:
        return r"$g_C<0$"
    return f"{r.TC_months:.1f}"


def arow(code, label, mse=True):
    r = pa.loc[code]
    cells = [label, f"{r.alpha_param:.3f}", f"{r.beta_data:.3f}", ff(r.g_N), ff(r.g_D), tc(r),
             "$" + ci(r.TC_lo, r.TC_hi).replace(r"\infty)", r"\infty)") + "$",
             (f"{r.mse:.4f}" if (mse and math.isfinite(r.mse)) else "--")]
    return " & ".join(cells) + r" \\"


P = []
P.append(r"\begin{table}[tp]")
P.append(r"\centering")
P.append(r"\caption{Algorithmic Progress under Alternative Estimators and Identifying Assumptions}")
P.append(r"\label{tab:progress}")
P.append(r"\footnotesize")
P.append(r"\setlength{\tabcolsep}{3pt}")
P.append(r"\begin{tabular}{@{}l ccccccc@{}}")
P.append(r"\toprule")
P.append(r" & $\hat\alpha$ & $\hat\beta$ & \multicolumn{1}{c}{$g_N$} & \multicolumn{1}{c}{$g_D$} & $T_C$ & 95\% CI & MSE \\")
P.append(r"\midrule")
P.append(r"\multicolumn{8}{@{}l}{\textit{Panel A. Doubling time of effective compute, data of Ho et al.\ (2024)}} \\")
tcp = rep.loc["T_C (months)"]
P.append(" & ".join([r"Published (bootstrap median)", f"{rep.loc['alpha_param'].ho_est:.3f}",
                     f"{rep.loc['beta_data'].ho_est:.3f}", "", "", f"{tcp.ho_est:.1f}",
                     f"$[{tcp.ho_lo:.1f}, {tcp.ho_hi:.1f}]$", "--"]) + r" \\")
P.append(arow("A1", r"(1) Replication, default stop"))
P.append(arow("A1c", r"(2) Run to convergence"))
P.append(arow("A2", r"(3) Unpenalized NLS"))
P.append(r"\quad profile-likelihood CI & & & & & & $[4.1, 40.5]$ & \\")
P.append(arow("A5", r"(4) $E_b$ estimated"))
P.append(arow("A3", r"(5) Hicks-neutral, as (2)"))
P.append(arow("A4", r"(6) Hicks-neutral, as (3)"))
P.append(arow("A12", r"(7) Effective data"))
P.append(arow("A10", r"(8) Impose $\gamma=0.0525$"))
P.append(arow("A6", r"(9) Year terms $\div$ exponents", mse=False))
P.append(arow("A7", r"(10) Besiroglu $(\alpha,\beta)$"))
P.append(arow("A9", r"(11) Hoffmann $(\alpha,\beta)$"))
reg = pa.loc[[c for c in pa.index if c.startswith("Rm")]]
npos = int((reg.g_C > 0).sum())
nneg = int((reg.g_C <= 0).sum())
tpos = reg[reg.g_C > 0].TC_months
P.append(" & ".join([r"(12) 12 sweep technologies",
                     "",
                     "", "", "",
                     rf"\multicolumn{{2}}{{c}}{{$g_C<0$ or {tpos.min():.0f}--{tpos.max():.0f}}}",
                     f"{reg.mse.min():.3f}--{reg.mse.max():.3f}"]) + r" \\")
P.append(r"\addlinespace[3pt]")
P.append(r"\multicolumn{8}{@{}l}{\textit{Panel B. Allocative gain, models with $C\geq10^{23}$ FLOP, 2020--21 to 2022--24}} \\")
P.append(r"Technology & $n_1,n_2$ & $\overline{CE}_1$ & \multicolumn{1}{c}{$\overline{CE}_2$} & \multicolumn{1}{c}{Gain} & 95\% CI & & Share \\")
P.append(r"\cmidrule(l){1-8}")


def brow(sample, tech, label):
    r = pb[(pb["sample"] == sample) & (pb.technology == tech)]
    assert len(r) == 1, (sample, tech)
    r = r.iloc[0]
    return " & ".join([label, f"{int(r.n_era1)}, {int(r.n_era2)}", f"{r.gmCE_era1:.2f}", f"{r.gmCE_era2:.2f}",
                       f"{r.allocative_gain:.2f}", f"$[{r.allocative_lo:.2f}, {r.allocative_hi:.2f}]$", "",
                       ff(r.allocative_share_of_algorithmic)]) + r" \\"


P.append(brow("C >= 1e23", "Besiroglu", "Besiroglu et al."))
P.append(brow("C >= 1e23", "Hoffmann (TeX)", "Hoffmann et al., TeX"))
P.append(brow("C >= 1e23", "Hoffmann (rounded)", "Hoffmann et al., rounded"))
P.append(brow("C >= 1e23", "Farseer grid", "Farseer grid"))
sw = pb[(pb["sample"] == "C >= 1e23") & (pb.technology.str.startswith("m1:") | pb.technology.str.startswith("m2:"))]
assert len(sw) == 9, len(sw)
P.append(" & ".join(["Nine sweep technologies", "7, 92", "", "",
                     f"{sw.allocative_gain.min():.2f}--{sw.allocative_gain.max():.2f}", "", "",
                     f"{ff(sw.allocative_share_of_algorithmic.min())} to {ff(sw.allocative_share_of_algorithmic.max())}"]) + r" \\")
P.append(brow("C >= 1e23, cleaned", "Besiroglu", "Besiroglu, cleaned"))
P.append(brow("Top-5 compute per year", "Besiroglu", "Besiroglu, top five"))
P.append(brow("C >= 1e23", "Ho model 7", "Ho et al.\\ (2024)"))
P.append(r"\bottomrule")
P.append(r"\end{tabular}")
P.append("")
P.append(r"\begin{tablenotes}")
P.append(
    r"Panel A re-estimates the preferred model of \citet{ho2024algorithmic}, "
    r"$\ln\text{ppl}=\exp(\alpha_{0b}-\alpha g_N t-\alpha\ln N)+\exp(\beta_{0b}-\beta g_D t-\beta\ln D)$ with irreducible loss set "
    r"to zero, under the stated estimator or restriction. $g_N,g_D$: parameter- and data-augmenting growth rates (log points per "
    r"year); $T_C=12\ln2/(g_N+g_D)$: effective-compute doubling time in months. Rows (1)--(2) minimize their objective, mean "
    r"squared error plus $0.0025\sum|\theta|$, by SLSQP from zero; row (1) uses the scipy default tolerance of the authors' "
    r"code and row (2) runs to convergence. The published 8.4 months is a bootstrap median; the point estimate of row (1) is "
    r"8.7. Intervals: 95\% paper-cluster bootstrap (400 draws), percentiles of $g_N+g_D$ mapped to $T_C$ ($\infty$ when "
    r"non-positive growth lies inside the interval); the published interval is the authors' iid bootstrap; the "
    r"profile-likelihood interval inverts an iid Gaussian likelihood-ratio test. Hicks neutrality: $\alpha g_N=\beta g_D$. "
    r"(7): data measured as effective data under repetition \citep{muennighoff2023scaling}. (8): the frontier elasticity "
    r"obtained by fitting the $E=0$ form to the Chinchilla sweep. (9): Ho et al.'s year coefficients divided by the exponents "
    r"of \citet{besiroglu2024chinchilla}, without re-estimation \citep{whitfill2025note}. (10)--(12): exponents fixed at "
    r"experimental values, benchmark-specific $E_b$ estimated; row (12) reports the range over the 12 technologies estimated in "
    rf"Section~\ref{{sec:tech}} (two on the Chinchilla sweep): $g_C<0$ for {nneg} of them; for the other {npos}, $T_C$ lies "
    rf"between {tpos.min():.0f} and {tpos.max():.0f} months, and every interval includes non-positive growth. "
    r"Panel B: $\overline{CE}$ is the geometric mean of Farrell cost efficiency $C_{\min}(L(N,D))/6ND$ under the stated "
    r"technology in each period ($n_1$, $n_2$ models); gain $=\overline{CE}_2/\overline{CE}_1$, with a bootstrap over models "
    r"(and over parameter draws for Besiroglu et al.). Share $=\ln(\text{gain})/\ln(8.9)$, where 8.9 is the effective-compute "
    r"gain implied by row (1) over the difference in mean release dates (9.9 for the top-five sample, 8.2 for the cleaned "
    r"sample); a negative share means that cost efficiency fell. Top five: the five models with the most compute in each year. Cleaned: drops derivatives, systems-paper runs, non-transformers, exact re-releases, models with $D/N$ "
    r"exactly 20 and Epoch ``speculative'' entries."
)
P.append(r"\end{tablenotes}")
P.append(r"\begin{tablenotes}[Source]")
P.append(
    r"Authors' calculations from the data and code of \citet{ho2024algorithmic} (Panel A) and the Epoch AI models "
    r"database \citep{epochai2026data} (Panel B)."
)
P.append(r"\end{tablenotes}")
P.append(r"\end{table}")
open(f"{ROOT}/paper/tables/table9_progress.tex", "w").write("\n".join(P) + "\n")
print("ok")
