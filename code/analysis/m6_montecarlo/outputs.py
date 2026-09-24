"""Tables (CSV + LaTeX) and figures for module m6_montecarlo."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import design_a as A  # noqa: E402
import design_b as B  # noqa: E402
import mc_lib as ml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aer_style as st  # noqa: E402

if ml.QUICK:                                   # smoke-test outputs never overwrite the paper outputs
    _savefig = st.savefig
    st.savefig = lambda fig, name: _savefig(fig, name, folder=ml.FIG)

MOD = "m6_montecarlo"
TAB = ml.TAB


def tpath(name, ext):
    return os.path.join(TAB, f"{MOD}_{name}.{ext}")


def fmt(x, d=3):
    """Number for LaTeX: sensible precision, no negative zero, math minus sign."""
    if x is None or not np.isfinite(float(x)):
        return "--"
    x = float(x)
    ax = abs(x)
    dd = 0 if ax >= 100 else (1 if ax >= 10 else d)
    out = f"{ax:.{dd}f}"
    if x < 0 and float(out) != 0.0:
        return "$-$" + out
    return out


def pct(x):
    return "--" if not np.isfinite(x) else f"{100 * x:.0f}"


def tex_wrap(body, colspec, caption, label, notes):
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering", r"\begin{threeparttable}", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\small", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", body, r"\bottomrule",
        r"\end{tabular}", r"\begin{tablenotes}[flushleft]\footnotesize", rf"\item \textit{{Notes:}} {notes}",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", ""])


# ============================================================================= Design A

A_ROWS = [("onpath", r"(i) On-path, $v=0.02$"), ("iso16", r"(ii) IsoFLOP, $\pm16\times$"),
          ("iso4", r"(ii$'$) IsoFLOP, $\pm4\times$"), ("fact", r"(iii) Factorial grid"),
          ("opt_s0", r"(iv) Optimizing labs, $v=0$"), ("opt_s0.1", r"(iv) Optimizing labs, $v=0.1$"),
          ("opt_s0.3", r"(iv) Optimizing labs, $v=0.3$"), ("opt_s1", r"(iv) Optimizing labs, $v=1$"),
          ("kaplan", r"(v) Kaplan-belief labs, $N\propto C^{0.73}$")]


def bootnote(bc):
    """Footnote text for the warm-start vs multi-start bootstrap check (review addition)."""
    if bc is None or bc.empty:
        return ""
    parts = []
    for cell, lab in [("onpath", "on-path ($v=0.02$)"), ("opt_s0.3", "$v=0.3$")]:
        r = bc[(bc.cell == cell) & (bc.param == "sigma_star")]
        if len(r):
            r = r.iloc[0]
            parts.append(rf"{lab}: {pct(r.warm_cover)}\% warm-start vs {pct(r.multi_cover)}\% multi-start coverage "
                         rf"(median widths {fmt(r.warm_median_width, 3)} vs {fmt(r.multi_median_width, 3)})")
    R = int(bc.R.iloc[0]); B_ = int(bc.B.iloc[0])
    return (r" ($^{\S}$: in a separate check on " + f"{R}" + r" fresh replications with " + f"{B_}" +
            r" draws that also restarts every draw from the estimator's nine random starting values, " + "; ".join(parts) + ")")


def design_a_tables(dfa, bc=None, tag="designA"):
    """tag = file stem: 'designA' (v1, run.py) or 'designA_v2' (v2 re-run, run_v2.py; the v1 files stay intact)."""
    summ = A.summarize(dfa)
    summ.to_csv(tpath(f"{tag}_summary", "csv"), index=False)
    diag = A.design_diagnostics()
    diag.to_csv(tpath(f"{tag}_diagnostics", "csv"), index=False)
    prof = A.profile_summary(dfa)
    prof.to_csv(tpath(f"{tag}_profile", "csv"), index=False)

    get = lambda cell, est, par, col: summ.loc[(summ.cell == cell) & (summ.estimator == est) & (summ.param == par), col]
    val = lambda *a: float(get(*a).iloc[0]) if len(get(*a)) else np.nan
    lines = [r" & \multicolumn{6}{c}{Primal estimator (Huber-LSE on $\ln L$, $\kappa=1$): bias [RMSE]} \\",
             r"\cmidrule(lr){2-7}",
             r"Design & $\alpha$ & $\beta$ & $a$ & $\gamma$ & $\sigma^*$ & $\ln M^*(10^{24})$ \\", r"\midrule",
             r"\multicolumn{7}{l}{\textit{Panel A. Sampling distribution of the primal (Approach-3) estimator}} \\"]
    for cell, lab in A_ROWS:
        b = [fmt(val(cell, "primal", p, "bias"), 2 if p == "lnMstar" else 3) for p in ml.PARAMS]
        r = ["[" + fmt(val(cell, "primal", p, "rmse"), 2 if p == "lnMstar" else 3) + "]" for p in ml.PARAMS]
        lines += [lab + " & " + " & ".join(b) + r" \\", " & " + " & ".join(r) + r" \\[2pt]"]
    lines += [r"\midrule",
              r"\multicolumn{7}{l}{\textit{Panel B. Dual estimator ($a$ from optima, $\gamma$ from $L^*(C)$, $\alpha=\gamma/a$, $\beta=\gamma/(1-a)$): RMSE}} \\"]
    for cell, lab in A_ROWS:
        if cell == "fact":
            continue
        r = [fmt(val(cell, "dual", p, "rmse"), 2 if p == "lnMstar" else 3) for p in ml.PARAMS]
        lines.append(lab + " & " + " & ".join(r) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{7}{l}{\textit{Panel C. Identification diagnostics and inference for $\sigma^*$}} \\",
              r" & cond$(J'J)$ & corner & Wald & bootstrap & profile CI & flat \\",
              r" & normalized & share (\%) & cover (\%) & cover (\%) & width & share (\%) \\", r"\midrule"]
    for cell, lab in A_ROWS:
        d = diag[diag.cell == cell]
        cond = d.cond_normalized.iloc[0] if len(d) else np.nan
        if not np.isfinite(cond):
            cstr = "--"
        elif cond > 1e15:
            cstr = r"$\infty$"
        else:
            m_, e_ = f"{cond:.1e}".split("e")
            cstr = rf"${m_}\times10^{{{int(e_)}}}$"
        g = summ[(summ.cell == cell) & (summ.estimator == "primal") & (summ.param == "sigma_star")].iloc[0]
        pr = prof[prof.cell == cell].iloc[0]
        wald = pct(g.wald_cover) + ("$^{\\dagger}$" if g.wald_computable < 0.95 else "")
        boot = pct(g.boot_cover) if "boot_cover" in g and np.isfinite(g.get("boot_cover", np.nan)) else "--"
        if bc is not None and cell in set(bc.cell) and boot != "--":
            boot += r"$^{\S}$"
        full_w = float(A.SIG_GRID[-1] - A.SIG_GRID[0]) - 1e-3
        wtxt = "full" if pr.median_ci_width >= full_w else fmt(pr.median_ci_width, 3)
        lines.append(f"{lab} & {cstr} & {pct(g.corner_share)} & {wald} & {boot} & {wtxt} & {pct(pr.share_flat_everywhere)}" + r" \\")
    t = A.cells()
    notes = (rf"{A.R_MAIN} replications per design; every design has 90 runs and the same total training compute "
             r"$\sum 6ND = 5.1\times10^{22}$ FLOP (10 runs at each of Chinchilla's nine IsoFLOP budgets, $6\times10^{18}$--$3\times10^{21}$). "
             r"Truth: Besiroglu et al.\ (2024) parameters ($\alpha=0.348$, $\beta=0.366$, $a=0.513$, $\gamma=0.178$, $\sigma^*=0.737$, "
             r"$\ln M^*(10^{24})=2.90$, i.e.\ $M^*=18.1$ tokens per parameter). Log-loss noise sd $=0.0075$ (residual sd of the Besiroglu fit on the "
             r"Chinchilla extraction). $v$ is the sd of allocation errors in $\ln(D/N)$ at fixed compute. Panel A: bias with RMSE in brackets. "
             r"The primal estimator is the Hoffmann/Besiroglu Huber($10^{-3}$) fit of $\ln L$, L-BFGS-B from nine starting values "
             r"drawn at random from Hoffmann et al.'s grid region (none at the truth; v2) within the box $\alpha,\beta\in[10^{-3},5]$; `corner' = estimate on the box boundary or one power-law term "
             r"$<2\%$ of reducible loss at every design point. Panel B: the dual estimator uses the noisy optima (on-path/optimizing designs) or "
             r"IsoFLOP parabola minima (Approach 2), and identifies $\alpha,\beta$ only through $\kappa=1$ and optimality. Panel C: condition number of "
             r"$J'J$ at the truth with $A,B$ normalized at the geometric means of $N,D$; Wald = LAD-sandwich/delta-method 95\% CI coverage for "
             r"$\sigma^*$ ($^{\dagger}$: covariance numerically singular or corner in $>5\%$ of replications; coverage over computable replications); "
             rf"bootstrap = pairs-bootstrap percentile CI ({A.B_BOOT} draws, first {A.R_BOOT} replications), each draw warm-started at the replication's estimate"
             + bootnote(bc) + r"; "
             rf"profile CI = median width of the 95\% profile-likelihood "
             rf"CI for $\sigma^*$ in the $\kappa$-generalized model $L=E+(AN^{{-a_1}}+BD^{{-b_1}})^{{\kappa}}$ with $\kappa$ free, on the grid $[0.05,0.99]$ (first {A.R_PROF} replications), inverting LR$=n\ln(\mathrm{{SSR}}(\sigma^*)/\mathrm{{SSR}}_{{\min}})$ against the $\chi^2_1$ critical value, with $\mathrm{{SSR}}_{{\min}}$ the unrestricted ($\kappa$ and $\sigma^*$ free) minimum; "
             r"`full' = the median CI is the entire grid; CI endpoints are linear interpolations of LR between grid points (spacing 0.004--0.005 next to the truth); flat = share of replications whose profile CI is the entire grid. "
             r"Dual estimator: frontier fitted from a $2\times2\times2$ grid of starting values (none at the truth).")
    tex = tex_wrap("\n".join(lines), "l" + "c" * 6,
                   r"Monte Carlo: what alternative experimental designs identify (common compute budget)",
                   "tab:mc_designs", notes)
    open(tpath(tag, "tex"), "w").write(tex)
    return summ, diag, prof


# ============================================================================= Design B

IND_MAIN = ["exog", "funding", "target", "predet"]
IND_VAR = ["exog_sel", "predet_sel", "predet_me", "predet_psiD", "funding_all", "funding_flag"]
IND_EST = ["E1_pooled_huber", "E1b_pooled_huber_Etrue", "E2_pooled_nls", "E3_lab_fe", "E4_mundlak", "E5_labgen_fe", "E6_acf",
           "E6b_acf_lagonly", "E7_iv_path", "E8_heckman", "E9_foc_system", "E10_foc_system_T"]
EST_SHORT = {"E1_pooled_huber": "Pooled Huber-LSE, $E$ free (ML practice)",
             "E1b_pooled_huber_Etrue": "Pooled Huber-LSE, $E$ at truth", "E2_pooled_nls": "Pooled NLS + generation FE",
             "E3_lab_fe": "Lab FE", "E4_mundlak": "Mundlak (CRE)", "E5_labgen_fe": r"Lab $\times$ generation FE",
             "E6_acf": "ACF GMM ($c$ predetermined)", "E6b_acf_lagonly": "ACF GMM (lagged instruments)",
             "E7_iv_path": "IV (compute price) + path", "E8_heckman": "Heckman selection correction",
             "E9_foc_system": "FOC system, $T$ ignored", "E10_foc_system_T": "FOC system, $T$ proxied"}
SC_SHORT = {"exog": "(a) Exogenous", "funding": "(b) Funding", "target": "(c) Target", "predet": "(d) Predet.",
            "exog_sel": "(a)+select.", "predet_sel": "(d)+select.", "predet_me": "(d)+ME in $D$",
            "predet_psiD": r"(d)+$\psi_D$", "funding_all": "(b)+sel.+ME", "funding_flag": "(b) flagships"}
PAR_TEX = {"alpha": r"$\alpha$", "beta": r"$\beta$", "a": r"allocation exponent $a$", "gamma": r"frontier elasticity $\gamma$",
           "sigma_star": r"$\sigma^*$", "lnMstar": r"$\ln M^*(10^{24})$", "sd_omega": r"TFP dispersion sd$(\omega)$",
           "tfp_growth": r"TFP growth per generation"}


# Just-identified 2SLS has no finite moments, so its Monte Carlo mean and RMSE are driven by a few extreme
# draws (e.g. a single gamma-hat of 2.7 under the target rule).  For these estimators the tables and the
# industry figure report the MEDIAN bias and the MEDIAN absolute error instead (review fix, 2026-09-23).
ROBUST_EST = ("E7_iv_path",)
ROBUST_NOTE = (r"$^{\ddagger}$Just-identified 2SLS has no finite moments; for this row the table reports the median bias "
               r"and the median absolute error instead of the mean bias and the RMSE. ")


def industry_tex(summ, scen, params, name, caption, label, extra_note="", ests=None, size=r"\small"):
    ests = IND_EST if ests is None else ests
    sub = summ.set_index(["scenario", "estimator", "param"])
    k = len(scen)
    head = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{SC_SHORT[s]}}}" for s in scen) + r" \\"
    cm = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(k))
    sub_head = "Estimator & " + " & ".join(["Bias & RMSE"] * k) + r" \\"
    lines = [head, cm, sub_head, r"\midrule"]
    tr = B.truth_row()
    for j, p in enumerate(params):
        if j:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * k}}}{{l}}{{\textit{{Panel {chr(65 + j)}. {PAR_TEX[p]} (truth {fmt(tr[p], 3)})}}}} \\")
        for e in ests:
            cells = []
            rob = e in ROBUST_EST
            for s in scen:
                try:
                    r = sub.loc[(s, e, p)]
                    cells += [fmt(r.median_bias if rob else r.bias, 3), fmt(r.mae if rob else r.rmse, 3)]
                except KeyError:
                    cells += ["--", "--"]
            if all(c == "--" for c in cells):
                continue
            lines.append(EST_SHORT[e] + (r"$^{\ddagger}$" if rob else "") + " & " + " & ".join(cells) + r" \\")
    notes = (rf"{B.R_IND} replications per scenario. Simulated industry: 40 labs $\times$ 6 generations, 1--4 models per "
             r"lab-generation (tier compute offsets $\approx 8\times$), Chinchilla technology at the Besiroglu et al.\ (2024) truth, Hicks-neutral "
             r"productivity $\omega_{ft}=\delta_t+x_{ft}$, $x_{ft}=0.8x_{f,t-1}+\xi_{ft}$, sd$(x)=0.25$, TFP growth $0.15$ per generation; "
             r"seed noise sd $0.05$ in $y=-\ln(L-E)$. Compute rules: (a) budget independent of $\omega$; (b) $c_{ft}$ rises by $2x_{ft}$; "
             r"(c) labs buy the compute that reaches a loss target; (d) $c_{ft}$ rises by $2x_{f,t-1}$ (provisioned a generation ahead). "
             r"Allocation: lifetime-compute optimum with log-normal inference demand $T$ (mean $\ln w\approx0.46$) plus allocation error "
             r"(sd $0.2$ in $\ln D/N$). $E$ is known to every estimator except the pooled Huber-LSE with $E$ free (ML practice). TFP growth and sd$(\omega)$ are computed from "
             r"lab-generation means of $y-\hat F(n,d)$. Both ACF rows use current within-lab-generation deviations of $n$ and $d$ "
             r"as instruments (valid here because tier offsets and inference demand are independent of $\omega$); without them the "
             r"lagged-instrument moment set is under-identified. `--': not identified in that design. "
             + (ROBUST_NOTE if any(e in ROBUST_EST for e in ests) else "") + extra_note)
    tex = tex_wrap("\n".join(lines), "l" + "rr" * k, caption, label, notes).replace(r"\small", size, 1)
    open(tpath(name, "tex"), "w").write(tex)


def industry_tables(dfb):
    summ = B.summarize(dfb)
    summ.to_csv(tpath("industry_summary", "csv"), index=False)
    meta = (dfb[dfb.estimator == "E2_pooled_nls"].groupby("scenario")
            [["n_trained", "n_released", "release_rate", "corr_c_omega", "mean_ln_w", "sample_sd_x"]].mean())
    fs = dfb[dfb.estimator == "E7_iv_path"].groupby("scenario")["first_stage_F"].median().rename("median_first_stage_F")
    imr = dfb[dfb.estimator == "E8_heckman"].groupby("scenario")["imr_coef"].mean().rename("mean_imr_coef")
    fail = dfb.assign(failed=dfb["gamma"].isna()).groupby(["scenario", "estimator"])["failed"].mean().unstack()
    meta = meta.join(fs).join(imr)
    meta.to_csv(tpath("industry_meta", "csv"))
    fail.to_csv(tpath("industry_failures", "csv"))
    main_est = ["E1_pooled_huber", "E2_pooled_nls", "E3_lab_fe", "E5_labgen_fe", "E6_acf", "E6b_acf_lagonly",
                "E7_iv_path", "E9_foc_system", "E10_foc_system_T"]
    industry_tex(summ, IND_MAIN, ["gamma", "a", "tfp_growth", "lnMstar"], "industry",
                 r"Monte Carlo: production-function estimators in a simulated AI industry",
                 "tab:mc_industry", ests=main_est,
                 extra_note=r"Mundlak (CRE) estimates are close to lab FE, and the Heckman estimator coincides with pooled NLS "
                            r"when there is no selection; both are reported in the appendix version of this table.")
    industry_tex(summ, IND_MAIN, ["gamma", "a", "sigma_star", "tfp_growth", "sd_omega", "lnMstar"], "industry_full",
                 r"Monte Carlo in the simulated industry: all estimators and objects", "tab:mc_industry_full",
                 size=r"\footnotesize")
    industry_tex(summ, ["exog_sel", "predet_sel", "predet_me", "predet_psiD", "funding_flag"],
                 ["gamma", "a", "tfp_growth", "sd_omega", "lnMstar"],
                 "industry_variants", r"Monte Carlo: selection, measurement error, factor-biased heterogeneity, flagships only",
                 "tab:mc_industry_variants",
                 r"Selection: a model is released only if its $y$ beats the lab's previous same-tier model by $0.5$ or clears the "
                 r"generation median (open-weight labs: bar lowered by $0.25$); about 18\% of trained models are not released. "
                 r"The Heckman correction uses all trained models' inputs and the open-weight policy as the exclusion restriction. "
                 r"ME: classical error (sd $0.15$) in $\ln D$, with reported $C=6ND$. $\psi_D$: lab-specific data-augmenting productivity "
                 r"(sd $0.3$) that the labs' allocation reflects. Flagships: one model per lab-generation under the funding rule; "
                 r"within-family estimators are not identified.", size=r"\footnotesize")
    industry_tex(summ, IND_MAIN, ["alpha", "beta", "sigma_star"], "industry_exponents",
                 r"Monte Carlo: exponents and on-path elasticity of substitution in the simulated industry",
                 "tab:mc_industry_exponents", size=r"\footnotesize")
    return summ, meta


# ============================================================================= verification tables

def verification_tables(ver, sel, chk):
    ver.to_csv(tpath("transmission_check", "csv"), index=False)
    sel.to_csv(tpath("selection_check", "csv"), index=False)
    chk.to_csv(tpath("identities_check", "csv"), index=False)
    rule_lab = {"exog": "Exogenous", "funding": r"Funding, $\lambda$", "predet": r"Predetermined, $\lambda$",
                "target": r"Target, sd$(\bar y)$"}
    lines = [r" & & \multicolumn{3}{c}{On expansion path ($T=0$)} & \multicolumn{2}{c}{Lifetime optimum ($T>0$)} & \multicolumn{2}{c}{$n=240$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Rule & Param. & Formula & Simulated & (s.e.) & Simulated & Misalloc. & Mean & Cover (\%) \\", r"\midrule"]
    on = ver[ver.allocation.str.startswith("on")].reset_index(drop=True)
    off = ver[~ver.allocation.str.startswith("on")].reset_index(drop=True)
    for i in range(len(on)):
        r, o = on.iloc[i], off.iloc[i]
        par = "--" if r.rule == "exog" else f"{r.param:g}"
        lines.append(f"{rule_lab[r.rule]} & {par} & {fmt(r.slope_formula, 4)} & {fmt(r.slope_sim, 4)} & ({fmt(r.slope_sim_se, 4)}) & "
                     f"{fmt(o.slope_sim, 4)} & {fmt(o.misallocation_term, 4)} & {fmt(r.finite_mean, 3)} & {100 * r.finite_cover_true_gamma:.0f}" + r" \\")
    notes = (r"Slope of $y=-\ln(L-E)$ on $c=\ln C$ (the frontier elasticity; the slope of $\ln(L-E)$ is its negative). "
             r"Formula: model\_spec Proposition 2, $\gamma+\pi_1\mathrm{Var}(\omega)/(\pi_1^2\mathrm{Var}(\omega)+\mathrm{Var}(\eta))$ with "
             r"$\gamma=0.1783$, sd$(\omega)=0.25$, $\rho=0.8$, sd$(\eta)=1$; exogenous: $\pi_1=0$; funding: $c=\lambda\omega+\eta$; predetermined: "
             r"$c=\lambda\omega_{-1}+\eta$ ($\mathrm{Cov}(c,\omega)=\lambda\rho\mathrm{Var}(\omega)$); target: $c=(\bar y-\omega+\ln K)/\gamma+\ln 6$ with "
             r"target dispersion sd$(\bar y)$ (the slope is $\gamma\,\mathrm{Var}(\bar y)/(\mathrm{Var}(\bar y)+\mathrm{Var}(\omega))$, $\to0$ for a common target). "
             r"Simulated: OLS on $10^6$ draws with seed noise sd $0.05$. $T>0$: inference demand $T=3D^*(c)\theta$, $\ln\theta\sim N(0,0.8^2)$, "
             r"allocation error sd $0.1$; `Misalloc.' $=-\mathrm{Cov}(c,\Delta)/\mathrm{Var}(c)$, where $\Delta$ is the output lost to over-training, "
             r"the extra term that appears when the misallocation loss is correlated with compute (only under the target rule, where labs with higher "
             r"$T$ buy more compute to reach the same loss). $n=240$: mean OLS slope over 2{,}000 samples of 240 lab-generations and the "
             r"coverage of the nominal 95\% OLS CI for $\gamma$.")
    tex = tex_wrap("\n".join(lines), "llccccccc",
                   r"Transmission bias of the returns to compute: closed form versus simulation", "tab:mc_transmission", notes)
    open(tpath("transmission", "tex"), "w").write(tex)


# ============================================================================= figures

def _s_of(cell):
    for c in A.cells():
        if c["name"] == cell:
            return c.get("s")
    return None


def fig2(summ, prof, dfa):
    st.use()
    fig, ax = plt.subplots(2, 2, figsize=(st.WIDTH_FULL, 4.9))
    s_cells = [f"opt_s{s:g}" for s in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]]
    S0 = 0.01                     # plotting position of s = 0 on the log axis
    xs = np.array([max(_s_of(c), S0) for c in s_cells])

    def ser(cell, est, par, col="rmse"):
        g = summ[(summ.cell == cell) & (summ.estimator == est) & (summ.param == par)]
        return float(g[col].iloc[0]) if len(g) else np.nan

    def ref_lines(a, par, col="rmse", src="summ"):
        for cell, colr, ls_, lab in [("iso16", st.ORANGE, "--", r"IsoFLOP $\pm16\times$"), ("fact", st.AQUA, ":", "Factorial grid")]:
            v = ser(cell, "primal", par, col) if src == "summ" else float(prof.loc[prof.cell == cell, col].iloc[0])
            a.axhline(v, color=colr, ls=ls_, lw=1.2, label=lab)

    def s_axis(a):
        a.set_xscale("log")
        a.set_xticks([S0, 0.05, 0.1, 0.3, 1, 2])
        a.set_xticklabels(["0", "0.05", "0.1", "0.3", "1", "2"])
        a.minorticks_off()
        a.set_xlabel(r"allocation error sd $v$ in $\ln(D/N)$ (optimizing labs)")

    # A: RMSE sigma*
    a = ax[0, 0]
    y = [ser(c, "primal", "sigma_star") for c in s_cells]
    yd = [ser(c, "dual", "sigma_star") for c in s_cells]
    a.plot(xs, y, "-o", color=st.BLUE, label="Primal (Huber-LSE)")
    a.plot(xs, yd, "-s", color=st.INK2, ms=3, lw=1.0, label=r"Dual ($\kappa=1$ + optimality)")
    ref_lines(a, "sigma_star")
    a.set_yscale("log")
    s_axis(a)
    a.set_ylabel(r"RMSE of $\hat\sigma^*$")
    a.set_title(r"A. The better labs optimize, the less their data reveal", loc="left")
    a.legend(loc="upper right", fontsize=6.5)

    # B: profile LR over sigma* (kappa free)
    a = ax[0, 1]
    grid = A.SIG_GRID
    cols = [f"median_LR_{s:.3f}" for s in grid]
    for cell, colr, lab, ls_ in [("onpath", st.INK2, r"On-path ($s=0.02$)", "-"), ("opt_s0.3", st.BLUE, r"Optimizing, $s=0.3$", "-"),
                                 ("iso16", st.ORANGE, r"IsoFLOP $\pm16\times$", "--"), ("fact", st.AQUA, "Factorial", ":")]:
        v = prof.loc[prof.cell == cell, cols].to_numpy(float).ravel()
        a.plot(grid, v, ls_, color=colr, label=lab, marker="o", ms=2.5)
    a.axhline(A.CHI2_95, color=st.MUTED, lw=0.8)
    a.text(0.505, A.CHI2_95 * 1.15, r"$\chi^2_1$ 95% critical value", fontsize=7, color=st.INK2)
    a.axvline(ml.TRUE["sigma_star"], color=st.MUTED, lw=0.8, ls=":")
    a.set_yscale("symlog", linthresh=1)
    a.set_ylim(-0.05, 6000)
    a.set_xlabel(r"$\sigma^*$ imposed ($\kappa$ free)")
    a.set_ylabel("median profile LR statistic")
    a.set_title(r"B. Profile likelihood of $\sigma^*$ when $\kappa$ is free", loc="left")
    a.legend(loc="upper center", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.8)

    # C: RMSE ln M*
    a = ax[1, 0]
    y = [ser(c, "primal", "lnMstar") for c in s_cells]
    yd = [ser(c, "dual", "lnMstar") for c in s_cells]
    a.plot(xs, y, "-o", color=st.BLUE, label="Primal (Huber-LSE)")
    FLOOR = 3e-3
    yd = np.maximum(np.array(yd, float), FLOOR)          # s = 0: the dual estimate of M* is exact (RMSE ~ 1e-14)
    a.plot(xs, yd, "-s", color=st.INK2, ms=3, lw=1.0, label="Dual (optimality imposed)")
    if np.any(yd <= FLOOR):
        # label below-right of the floor point so that it does not collide with the rising dual line
        a.annotate("exact at $s=0$", xy=(xs[0], FLOOR), xytext=(xs[0] * 1.35, FLOOR * 0.62), fontsize=6.5,
                   color=st.INK2, va="center")
    ref_lines(a, "lnMstar")
    a.set_yscale("log")
    a.set_ylim(FLOOR * 0.45, None)
    s_axis(a)
    a.set_ylabel(r"RMSE of $\ln\hat M^*(10^{24})$")
    a.set_title(r"C. Compute-optimal tokens per parameter $M^*$", loc="left")

    # D: sampling distributions of sigma*
    a = ax[1, 1]
    cells = [("onpath", "On-path\n$s{=}0.02$"), ("opt_s0.1", "Optim.\n$s{=}0.1$"), ("opt_s0.3", "Optim.\n$s{=}0.3$"),
             ("opt_s1", "Optim.\n$s{=}1$"), ("kaplan", "Kaplan\nbeliefs"), ("iso16", "IsoFLOP\n$\\pm16\\times$"), ("fact", "Factorial")]
    data = [dfa.loc[dfa.cell == c, "p_sigma_star"].to_numpy(float) for c, _ in cells]
    bp = a.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55)
    for b in bp["boxes"]:
        b.set(facecolor="#dbe8f8", edgecolor=st.INK2, lw=0.8)
    for k in ("whiskers", "caps", "medians"):
        for l_ in bp[k]:
            l_.set(color=st.INK2 if k != "medians" else st.BLUE, lw=0.9 if k != "medians" else 1.4)
    a.axhline(ml.TRUE["sigma_star"], color=st.ORANGE, lw=1.0, ls="--", label=r"truth $\sigma^*=0.737$")
    a.set_xticks(range(1, len(cells) + 1))
    a.set_xticklabels([l for _, l in cells], fontsize=6.5)
    a.set_ylabel(r"$\hat\sigma^*$ (primal): 5-25-50-75-95 pct.")
    a.set_ylim(0.25, 1.0)
    a.legend(loc="lower right", fontsize=7)
    a.set_title(r"D. Sampling distribution of $\hat\sigma^*$", loc="left")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    st.savefig(fig, f"{MOD}_fig2_designs")


def fig_ci_width(summ, prof):
    st.use()
    fig, ax = plt.subplots(1, 2, figsize=(st.WIDTH_FULL, 2.4))
    s_cells = [f"opt_s{s:g}" for s in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]]
    xs = np.array([max(_s_of(c), 0.01) for c in s_cells])
    w = [float(prof.loc[prof.cell == c, "median_ci_width"].iloc[0]) for c in s_cells]
    fl = [float(prof.loc[prof.cell == c, "share_flat_everywhere"].iloc[0]) for c in s_cells]
    for a, v, lab, col in [(ax[0], w, r"median width of 95% profile CI for $\sigma^*$", "median_ci_width"),
                           (ax[1], fl, r"share of replications with a flat profile", "share_flat_everywhere")]:
        a.plot(xs, v, "-o", color=st.BLUE, label="Optimizing labs")
        for cell, colr, ls_, l2 in [("iso16", st.ORANGE, "--", r"IsoFLOP $\pm16\times$"), ("fact", st.AQUA, ":", "Factorial")]:
            a.axhline(float(prof.loc[prof.cell == cell, col].iloc[0]), color=colr, ls=ls_, label=l2)
        a.set_xscale("log")
        a.set_xticks([0.01, 0.05, 0.1, 0.3, 1, 2])
        a.set_xticklabels(["0", "0.05", "0.1", "0.3", "1", "2"])
        a.minorticks_off()
        a.set_xlabel(r"allocation error sd $v$")
        a.set_ylabel(lab, fontsize=7.5)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    st.savefig(fig, f"{MOD}_profile_ci")


def fig_industry(summ):
    st.use()
    params = [("gamma", r"bias in $\hat\gamma$ (truth 0.178)"), ("tfp_growth", r"bias in TFP growth (truth 0.15)"),
              ("lnMstar", r"bias in $\ln\hat M^*(10^{24})$")]
    ests = [e for e in IND_EST if e != "E1b_pooled_huber_Etrue"]      # E1b is a decomposition device (tables only)
    fig, ax = plt.subplots(1, 3, figsize=(st.WIDTH_FULL, 3.6), sharey=True)
    scen = [("exog", st.INK2, "o", "(a) exogenous"), ("funding", st.ORANGE, "s", "(b) funding"),
            ("target", st.AQUA, "D", "(c) target"), ("predet", st.BLUE, "^", "(d) predetermined")]
    yy = np.arange(len(ests))[::-1]
    for a, (p, lab) in zip(ax, params):
        for k, (s, col, mk, sl) in enumerate(scen):
            v = []
            for e in ests:
                g = summ[(summ.scenario == s) & (summ.estimator == e) & (summ.param == p)]
                stat = "median_bias" if e in ROBUST_EST else "bias"     # 2SLS: median (no finite moments)
                v.append(float(g[stat].iloc[0]) if len(g) else np.nan)
            a.plot(v, yy + (k - 1.5) * 0.17, mk, color=col, ms=3.6, ls="none", label=sl)
        a.axvline(0, color=st.INK, lw=0.7)
        a.set_xlabel(lab)
        a.grid(axis="y", visible=False)
        a.tick_params(axis="x", labelsize=7)
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels([EST_SHORT[e].replace("$", "").replace(r"\times", "x") + (" (median)" if e in ROBUST_EST else "")
                           for e in ests], fontsize=7)
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    st.savefig(fig, f"{MOD}_industry_bias")


def fig_transmission(ver, sel):
    st.use()
    fig, ax = plt.subplots(1, 2, figsize=(st.WIDTH_FULL, 2.7))
    a = ax[0]
    for rule, col, mk in [("exog", st.INK2, "o"), ("funding", st.ORANGE, "s"), ("predet", st.BLUE, "^"), ("target", st.AQUA, "D")]:
        g = ver[(ver.rule == rule) & ver.allocation.str.startswith("on")]
        h = ver[(ver.rule == rule) & ~ver.allocation.str.startswith("on")]
        a.plot(g.slope_formula, g.slope_sim, mk, color=col, ms=4, ls="none", label=rule)
        a.plot(h.slope_formula, h.slope_sim, mk, mfc="none", color=col, ms=6, ls="none")
    lim = [-0.01, max(ver.slope_formula.max(), ver.slope_sim.max()) + 0.02]
    a.plot(lim, lim, color=st.MUTED, lw=0.8)
    a.axhline(ml.TRUE["gamma"], color=st.MUTED, lw=0.6, ls=":")
    a.set_xlabel("closed-form plim (model_spec Prop. 2)")
    a.set_ylabel(r"simulated OLS slope of $y$ on $\ln C$")
    a.legend(fontsize=7, title="filled: on path; hollow: $T>0$", title_fontsize=6.5)
    a.set_title("A. Transmission bias: formula vs simulation", loc="left")
    b = ax[1]
    for rule, col, mk in [("exog", st.INK2, "o"), ("funding", st.ORANGE, "s"), ("target", st.AQUA, "D")]:
        g = sel[sel.rule == rule]
        b.plot(100 * g.share_dropped, g.slope_y_on_c, "-" + mk, color=col, ms=4, label=rule)
    b.axhline(ml.TRUE["gamma"], color=st.MUTED, lw=0.8, ls=":")
    b.set_xlabel(r"share of runs not released (absolute threshold on $y$), %")
    b.set_ylabel(r"OLS slope among released")
    b.set_title("B. Selection on output (Prop. 3)", loc="left")
    b.legend(fontsize=7)
    fig.tight_layout()
    st.savefig(fig, f"{MOD}_transmission_check")


def make_all(dfa, dfb, ver, sel, chk, dfc):
    bc = A.bootcheck_summary(dfc)
    bc.to_csv(tpath("designA_bootcheck", "csv"), index=False)
    summ_a, diag, prof = design_a_tables(dfa, bc)
    summ_b, meta = industry_tables(dfb)
    verification_tables(ver, sel, chk)
    fig2(summ_a, prof, dfa)
    fig_ci_width(summ_a, prof)
    fig_industry(summ_b)
    fig_transmission(ver, sel)


# ============================================================================= v2 (2026-09-24): Design A re-run

S_LIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
S0_POS = 0.01                                  # plotting position of s = 0 on the log axis


def fig2_v2(summ, prof):
    """Two-panel paper figure (Section II, Figure 2 of v2).
    A: RMSE of sigma*-hat (top) and ln M*-hat (bottom) against the allocation-error sd s, primal (kappa = 1 Huber-LSE)
       and dual estimators, with IsoFLOP +-16x and factorial-grid reference lines (primal estimator, same compute).
    B: median kappa-free profile LR over sigma* on the grid (0.05, 0.99), LR relative to the unrestricted optimum."""
    st.use()
    fig = plt.figure(figsize=(st.WIDTH_FULL, 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.12, wspace=0.28)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    b = fig.add_subplot(gs[:, 1])
    s_cells = [f"opt_s{s:g}" for s in S_LIST]
    xs = np.array([max(s, S0_POS) for s in S_LIST])

    def ser(cell, est, par, col="rmse"):
        g = summ[(summ.cell == cell) & (summ.estimator == est) & (summ.param == par)]
        return float(g[col].iloc[0]) if len(g) else np.nan

    FLOOR = 3e-3
    for a, par, ylab in [(a1, "sigma_star", r"RMSE of $\hat\sigma^*$"), (a2, "lnMstar", r"RMSE of $\ln\hat M^*(10^{24})$")]:
        yp = np.array([ser(c, "primal", par) for c in s_cells])
        yd = np.array([ser(c, "dual", par) for c in s_cells])
        exact = yd < FLOOR
        a.plot(xs, yp, "-o", color=st.BLUE, ms=3.6, lw=1.3, label=r"Primal ($\kappa=1$ Huber fit)")
        a.plot(xs, np.maximum(yd, FLOOR), "--s", color=st.INK2, ms=3.2, lw=1.0, mfc="white",
               label=r"Dual ($\kappa=1$ + optimality)")
        if exact.any():
            a.annotate("dual exact at $v=0$", xy=(xs[0], FLOOR), xytext=(xs[0] * 1.3, FLOOR * 1.9), fontsize=7.2,
                       color=st.INK2, va="center")
        for cell, colr, ls_, lab in [("iso16", st.ORANGE, "--", r"IsoFLOP $\pm16\times$"), ("fact", st.AQUA, ":", "Factorial grid")]:
            a.axhline(ser(cell, "primal", par), color=colr, ls=ls_, lw=1.2, label=lab)
        a.set_yscale("log")
        a.set_ylabel(ylab, fontsize=8)
        a.set_ylim(FLOOR * 0.6, None)
    a1.set_title(r"A. Precision against allocation error, same compute", loc="left")
    a1.legend(loc="upper right", fontsize=7.2, ncol=1, handlelength=2.2, borderaxespad=0.2)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.set_xscale("log")
    a2.set_xticks([S0_POS, 0.05, 0.1, 0.3, 1, 2])
    a2.set_xticklabels(["0", "0.05", "0.1", "0.3", "1", "2"])
    a2.minorticks_off()
    a2.set_xlabel(r"allocation error sd $v$ in $\ln(D/N)$ (optimizing labs)")

    grid = A.SIG_GRID
    cols = [f"median_LR_{x:.3f}" for x in grid]
    for cell, colr, lab, ls_, mk in [("onpath", st.INK2, r"On-path ($v=0.02$)", "-", "o"),
                                     ("opt_s0.3", st.BLUE, r"Optimizing labs, $v=0.3$", "-", "o"),
                                     ("iso16", st.ORANGE, r"IsoFLOP $\pm16\times$", "--", "o")]:
        v = prof.loc[prof.cell == cell, cols].to_numpy(float).ravel()
        b.plot(grid, np.maximum(v, 0.0), ls_, color=colr, label=lab, marker=mk, ms=2.4, lw=1.3)
    b.axhline(A.CHI2_95, color=st.MUTED, lw=0.8)
    b.text(0.06, A.CHI2_95 * 1.18, r"$\chi^2_1$ 95% critical value", fontsize=7.2, color=st.INK2)
    b.axvline(ml.TRUE["sigma_star"], color=st.MUTED, lw=0.8, ls=":")
    b.text(ml.TRUE["sigma_star"] + 0.012, 2600, r"true $\sigma^*$", fontsize=7.2, color=st.INK2, ha="left")
    b.set_yscale("symlog", linthresh=1)
    b.set_ylim(-0.05, 9000)
    b.set_xlim(0.0, 1.0)
    b.set_xlabel(r"$\sigma^*$ imposed ($\kappa$ free)")
    b.set_ylabel("median LR statistic (vs. unrestricted optimum)", fontsize=8)
    b.set_title(r"B. Profile likelihood of $\sigma^*$, $\kappa$ free", loc="left")
    b.legend(loc="upper left", fontsize=7.2, bbox_to_anchor=(0.0, 1.0), handlelength=2.0, borderaxespad=0.1)
    st.savefig(fig, f"{MOD}_fig2_v2")


def fig_ci_width_v2(prof):
    """Appendix figure (v2): the 95% kappa-free profile set for sigma* against the allocation-error sd v.
    A: median lower and upper bounds (optimizing labs; IsoFLOP +-16x for reference).  B: share of replications whose set
    is the whole grid (0.05, 0.99), contains all of [0.50, 0.95] (the v1 range), contains every sigma* >= 0.90, and
    contains every sigma* <= 0.50."""
    st.use()
    fig, ax = plt.subplots(1, 2, figsize=(st.WIDTH_FULL, 2.6))
    s_cells = [f"opt_s{s:g}" for s in S_LIST]
    xs = np.array([max(s, S0_POS) for s in S_LIST])
    val = lambda c, col: float(prof.loc[prof.cell == c, col].iloc[0])
    a = ax[0]
    lo = np.array([val(c, "median_ci_lo") for c in s_cells])
    hi = np.array([val(c, "median_ci_hi") for c in s_cells])
    a.fill_between(xs, lo, hi, color=st.BLUE, alpha=0.18, lw=0)
    a.plot(xs, lo, "-o", color=st.BLUE, ms=3.5, label="Optimizing labs, median bounds")
    a.plot(xs, hi, "-o", color=st.BLUE, ms=3.5)
    for k in ("median_ci_lo", "median_ci_hi"):
        a.axhline(val("iso16", k), color=st.ORANGE, ls="--", lw=1.1, label=r"IsoFLOP $\pm16\times$" if k.endswith("lo") else None)
    a.axhline(ml.TRUE["sigma_star"], color=st.MUTED, ls=":", lw=0.9)
    a.text(0.012, ml.TRUE["sigma_star"] + 0.015, r"true $\sigma^*$", fontsize=7, color=st.INK2, ha="left", va="bottom")
    a.set_ylim(0.0, 1.02)
    a.set_ylabel(r"95% profile set for $\sigma^*$", fontsize=8)
    a.set_title(r"A. Median bounds of the set, $\kappa$ free", loc="left")
    a.legend(fontsize=7, loc="lower right")
    b = ax[1]
    for col, colr, ls_, lab in [("share_flat_everywhere", st.INK2, "-", "whole grid (0.05, 0.99)"),
                                ("share_flat_on_v1_grid", st.BLUE, "-", r"all of [0.50, 0.95]"),
                                ("share_accept_all_above_090", st.ORANGE, "--", r"all $\sigma^*\geq0.90$"),
                                ("share_accept_all_below_050", st.AQUA, ":", r"all $\sigma^*\leq0.50$")]:
        b.plot(xs, [val(c, col) for c in s_cells], ls_, marker="o", ms=3.2, color=colr, label=lab)
    b.set_ylim(-0.02, 1.3)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.set_ylabel("share of replications", fontsize=8)
    b.set_title("B. The set contains ...", loc="left")
    b.legend(fontsize=6.8, loc="upper center", ncol=2, columnspacing=1.0, handlelength=2.0, borderaxespad=0.1)
    for a in ax:
        a.set_xscale("log")
        a.set_xticks([S0_POS, 0.05, 0.1, 0.3, 1, 2])
        a.set_xticklabels(["0", "0.05", "0.1", "0.3", "1", "2"])
        a.minorticks_off()
        a.set_xlabel(r"allocation error sd $v$ in $\ln(D/N)$")
    fig.tight_layout()
    st.savefig(fig, f"{MOD}_profile_ci_v2")


V1_TABLES = os.path.join(ml.ROOT, "data", "processed", "m6_montecarlo", "v1_backup", "tables")


def v1v2_compare(summ, prof, bc):
    """v1 (truth among the nine starts; grid [0.50, 0.95]) vs v2 (no truth start; grid (0.05, 0.99)): same data."""
    rows = []
    s1 = pd.read_csv(os.path.join(V1_TABLES, "m6_montecarlo_designA_summary.csv"))
    p1 = pd.read_csv(os.path.join(V1_TABLES, "m6_montecarlo_designA_profile.csv"))
    b1 = pd.read_csv(os.path.join(V1_TABLES, "m6_montecarlo_designA_bootcheck.csv"))
    stats = ["bias", "median_bias", "rmse", "mae", "q05", "q95", "wald_cover", "boot_cover", "corner_share"]
    for _, r2 in summ.iterrows():
        m = s1[(s1.cell == r2.cell) & (s1.estimator == r2.estimator) & (s1.param == r2.param)]
        if m.empty:
            continue
        r1 = m.iloc[0]
        for k in stats:
            if k in r1 and k in r2 and np.isfinite(r1[k]) and np.isfinite(r2[k]):
                rows.append(dict(block="summary", cell=r2.cell, estimator=r2.estimator, param=r2.param, stat=k,
                                 v1=r1[k], v2=r2[k], diff=r2[k] - r1[k]))
    for _, r2 in prof.iterrows():
        m = p1[p1.cell == r2.cell]
        if m.empty:
            continue
        r1 = m.iloc[0]
        for k1, k2 in [("cover_truth", "cover_truth"), ("share_flat_everywhere", "share_flat_on_v1_grid"),
                       ("share_flat_everywhere", "share_flat_everywhere"), ("median_ci_width", "median_ci_width"),
                       ("share_sig_hat_off_grid", "share_sig_hat_off_grid")]:
            rows.append(dict(block="profile", cell=r2.cell, estimator="profile", param="sigma_star",
                             stat=(k2 if k1 == k2 else f"{k2} (v1: {k1})"), v1=r1[k1], v2=r2[k2], diff=r2[k2] - r1[k1]))
    for _, r2 in bc.iterrows():
        m = b1[(b1.cell == r2.cell) & (b1.param == r2.param)]
        if m.empty:
            continue
        r1 = m.iloc[0]
        for k in ["warm_cover", "warm_median_width", "multi_cover", "multi_median_width"]:
            rows.append(dict(block="bootcheck", cell=r2.cell, estimator="primal", param=r2.param, stat=k,
                             v1=r1[k], v2=r2[k], diff=r2[k] - r1[k]))
    out = pd.DataFrame(rows)
    out.to_csv(tpath("designA_v2_vs_v1", "csv"), index=False)
    return out


def make_design_a_v2(dfa, dfc):
    """Design A outputs after the v2 re-run (no truth start; extended profile grid; het./clustered cells)."""
    # v2 numbers go to the canonical Design A file names (m6_montecarlo_designA*.csv|.tex); the v1 files are kept in
    # data/processed/m6_montecarlo/v1_backup/tables and compared in m6_montecarlo_designA_v2_vs_v1.csv.
    bc = A.bootcheck_summary(dfc)
    bc.to_csv(tpath("designA_bootcheck", "csv"), index=False)
    summ_a, diag, prof = design_a_tables(dfa, bc, tag="designA")
    fig2_v2(summ_a, prof)
    fig_ci_width_v2(prof)
    if not ml.QUICK and os.path.isdir(V1_TABLES):
        v1v2_compare(summ_a, prof, bc)
    return summ_a, diag, prof, bc


def make_design_b_v2(dfb):
    """Design B outputs after the v2 re-run (no estimator started at the truth); comparison with v1 (same data)."""
    summ_b, meta = industry_tables(dfb)
    fig_industry(summ_b)
    v1 = os.path.join(ml.ROOT, "data", "processed", "m6_montecarlo", "v1_backup", "tables",
                      "m6_montecarlo_industry_summary.csv")
    if not ml.QUICK and os.path.exists(v1):
        s1 = pd.read_csv(v1)
        m = summ_b.merge(s1, on=["scenario", "estimator", "param"], suffixes=("_v2", "_v1"))
        keep = ["scenario", "estimator", "param", "truth_v2", "bias_v1", "bias_v2", "median_bias_v1", "median_bias_v2",
                "rmse_v1", "rmse_v2", "mae_v1", "mae_v2"]
        m = m[[k for k in keep if k in m]].rename(columns={"truth_v2": "truth"})
        m["diff_bias"] = m.bias_v2 - m.bias_v1
        m["diff_rmse"] = m.rmse_v2 - m.rmse_v1
        m.to_csv(tpath("industry_v2_vs_v1", "csv"), index=False)
    return summ_b, meta
