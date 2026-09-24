"""ra5_theory -- Theory v2 for the revision (referee round 1): verify every new or changed formal claim of
paper/sections/appendix_proofs.tex and generate the module's tables and figures.

Usage:  /Users/yigitokar/scaling-laws-pf/.venv/bin/python code/analysis/ra5_theory/run.py

Deterministic (fixed seeds); about 2 minutes on 4 CPU processes. Prints PASS/FAIL per claim and exits 1 on any FAIL.
  1. symbolic_v2.run_all(): sympy proofs (generalized wedge and its cases, family FOC, model-free sigma* and w,
     quasi-homotheticity, DMR sign, partial-identification sign lemma, twin equivalence on the path).
  2. numeric_v2.run_all(): brute-force developer problems (12 conduct cases, family problems), model-free estimators on a
     non-separable technology, finite-grid bias, singular-information geometry and Monte Carlo rates, Fisher information
     with E estimated, DMR counterexample, sharpness of the partial-identification bounds.
  3. partial_id.run(): illustration of Proposition A11 on the verified sample (anchors: Chinchilla and Llama 3 IsoFLOP minima).
Outputs: output/tables/ra5_theory_*.csv/.tex, output/figures/ra5_theory_*.pdf/.png, data/processed/ra5_theory/*.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ra5common  # noqa: E402
from ra5common import PROC, RESULTS, TABLES, as_records  # noqa: E402


def _esc(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _sig(x):
    """Compact positive number: 2 decimals below 1, 1 decimal below 100, integer with thousands separator above."""
    if x < 1:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:,.0f}".replace(",", "{,}")


def _fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    return f"${s}$" if x < 0 else s


def write_table(path, caption, label, colspec, header_rows, body_rows, notes, source, size=r"\footnotesize", tabcolsep="4pt"):
    """AEA layout: booktabs tabular, then the AEA.cls tablenotes environment (no threeparttable)."""
    L = [r"\begin{table}[tp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size, rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
         rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    L += header_rows
    L.append(r"\midrule")
    L += body_rows
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", rf"\begin{{tablenotes}}{notes}\end{{tablenotes}}",
          rf"\begin{{tablenotes}}[Source]{source}\end{{tablenotes}}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


WEDGE_ROWS = [
    # (case, first-order condition, what s = (w-1)/w measures, effect on T-hat relative to T)
    (r"Lifetime compute, developer serves ($\lambda=p=\phi=1$)", r"$1+T/(3D)$", r"Serving share of lifetime compute", r"Equal"),
    (r"(a) Serving cost borne at rate $\lambda$, price ratio $p$, cost elasticity $\phi$", r"$1+\lambda\phi p\,T/(3D)$",
     r"$\lambda\phi X_S/(X_T+\lambda\phi X_S)$", r"$\times\phi$"),
    (r"(a$'$) Open release, adoption value $v(A)$, users bear serving", r"$1+v'A|\varepsilon_{A,P}|/X_T$",
     r"Adoption value of compactness", r"Not $T$"),
    (r"(b) Binding memory or tier cap, multiplier $\nu$", r"$(1+\nu)(1+m_N)$", r"Upper bound on serving share", r"Overstates"),
    (r"(c) Binding latency SLO, $\tau(N)\le\bar\tau$", r"$(1+\nu_\tau)(1+m_N)$", r"Upper bound on serving share", r"Overstates"),
    (r"(d) Data cost $c_D$ per token", r"$(1+m_N)/(1+m_D)$", r"$(X_S-X_D)/(X_T+X_S)$", r"Understates"),
    (r"(d$'$) Binding data cap, multiplier $\mu$", r"$(1+m_N)/(1+m_D+\mu)$", r"Below serving share; $w<1$ possible", r"Understates"),
    (r"(e) Logit distillation, teacher $N_T$", r"$\dfrac{1+T/(3D)}{1+N_T/(3N)}$", r"$(X_S-X_{\rm teach})/(X_T+X_S)$", r"Understates"),
    (r"(f) Dollar cost $\propto N^{1+\delta}D$; serving $\propto N^{1+\delta_S}$", r"$1+\delta+\lambda(1+\delta_S)pT/(3D)$",
     r"Mixes $\delta$ with serving", r"$+3D\delta+\lambda\delta_SpT$"),
    (r"(g) Family token budget, sizes chosen", r"$\sum_i\pi_i(1+m_{N,i})=(1+m_D)\bar w_H$", r"$1-1/\bar w_H=\sum_i\omega_is_i$",
     r"Only $\sum_i\pi_i\hat T_i$ identified"),
    (r"Lab's technology differs (factor bias, beliefs, output)", r"$\hat w=w\,w_L/w_{L'}$", r"Contaminated", r"Over iff $w_L>w_{L'}$"),
]


def main():
    t0 = time.time()
    print("=" * 100 + "\nra5_theory: symbolic verification (sympy)\n" + "=" * 100)
    import symbolic_v2
    symbolic_v2.run_all()
    print("=" * 100 + "\nra5_theory: numerical / Monte Carlo verification\n" + "=" * 100)
    import numeric_v2
    out = numeric_v2.run_all()
    print("=" * 100 + "\nra5_theory: partial-identification illustration (verified sample)\n" + "=" * 100)
    import partial_id
    pi = partial_id.run()
    s = pi["sign"]
    ok_nest = True
    for a, g in s.groupby("anchor"):
        g = g.set_index("assumption")
        ok_nest &= g.loc["Normal inputs only", "share_not_identified"] >= g.loc["Slope in cross-sweep range", "share_not_identified"] - 1e-12
        ok_nest &= g.loc["Slope in cross-sweep range", "share_not_identified"] >= g.loc["Cross-sweep range + M* nondecreasing", "share_not_identified"] - 1e-12
        ok_nest &= g.loc["Parametric extrapolation (point)", "share_not_identified"] == 0
    ra5common.record("PI.illus", "Prop. A11, illustration", "Nested assumptions give nested identified sets: the share of verified-sample models whose sign "
                     "of w-1 is not identified falls weakly from normal inputs to the cross-sweep slope range to monotonicity, and is zero "
                     "under point extrapolation", "NEW", "numeric", ok_nest,
                     "; ".join(f"{a.split(' (')[0]}: " + ", ".join(f"{v:.2f}" for v in g["share_not_identified"]) for a, g in s.groupby("anchor")))

    # ------------------------------------------------------------------ tables: register
    reg = pd.DataFrame(as_records())
    reg.to_csv(os.path.join(TABLES, "ra5_theory_claims.csv"), index=False)
    from claims_tex import CLAIMS_TEX
    # Review fix: a single 38-row float overflowed the page; the register is split into two floats (symbolic, numerical)
    # written to the same file (labels tab:ra5-claims and tab:ra5-claims-num).
    def _claim_rows(sel):
        out_rows = []
        for c in RESULTS:
            if sel(c):
                stmt = CLAIMS_TEX.get(c.cid, _esc(c.cid))
                out_rows.append(rf"{_esc(c.cid)} & {_esc(c.result)} & {stmt} & {c.verdict.title()}{'' if c.passed else ' (FAIL)'} \\")
        return out_rows
    colspec = r"@{}l >{\raggedright\arraybackslash}p{0.13\textwidth} >{\raggedright\arraybackslash}p{0.47\textwidth} l@{}"
    note = (r"Each statement was checked symbolically (sympy; identities reduced to zero, or evaluated at random rational points with "
            r"40-digit precision) and numerically by a method that does not use the closed form under test: direct maximization of the "
            r"developer's objective, isoquant tracing, profiling of the least-squares criterion, Monte Carlo, or finite-difference Fisher "
            r"information. ``Corrected'' marks results that revise a statement of the first draft. The full register, with error metrics, "
            r"is \texttt{ra5\_theory\_claims.csv}.")
    src = r"Authors' calculations; \texttt{code/analysis/ra5\_theory}."
    pa = os.path.join(TABLES, "ra5_theory_claims.tex")
    pb = os.path.join(TABLES, "_ra5_theory_claims_num.tmp")
    write_table(pa, "Verification of the Revised Formal Results: Symbolic Checks", "tab:ra5-claims", colspec,
                [r"ID & Result & Statement tested & Verdict \\"], _claim_rows(lambda c: c.method == "sympy"), note, src, size=r"\scriptsize")
    write_table(pb, "Verification of the Revised Formal Results: Numerical Checks", "tab:ra5-claims-num", colspec,
                [r"ID & Result & Statement tested & Verdict \\"], _claim_rows(lambda c: c.method != "sympy"), note, src, size=r"\scriptsize")
    with open(pa, "a") as fa, open(pb) as fb:
        fa.write("\n" + fb.read())
    os.remove(pb)

    # ------------------------------------------------------------------ tables: generalized wedge
    wc = out["wedge_cases"]
    wc.to_csv(os.path.join(TABLES, "ra5_theory_wedge_cases.csv"), index=False)
    body = [rf"{a} & {b} & {c} & {d} \\" for a, b, c, d in WEDGE_ROWS]
    write_table(os.path.join(TABLES, "ra5_theory_wedge_cases.tex"), "What the Wedge Identifies under Alternative Developer Objectives",
                "tab:ra5-wedge-cases", r"@{}>{\raggedright\arraybackslash}p{0.31\textwidth} >{\raggedright\arraybackslash}p{0.21\textwidth} >{\raggedright\arraybackslash}p{0.24\textwidth} >{\raggedright\arraybackslash}p{0.14\textwidth}@{}",
                [r"Developer objective or cost structure & Wedge $w=\varepsilon_N/\varepsilon_D$ & $s=(w-1)/w$ measures & $\hat T$ vs.\ $\lambda pT$ \\"],
                body,
                r"The developer chooses $(N,D)$ to maximize $V(L,N)$ minus training expenditure $X_T=c_T\,6ND$ (and data cost $c_DD$). "
                r"$m_N\equiv-(\partial V/\partial\ln N)_L/X_T$ is the value of compactness at fixed quality per unit of training expenditure; "
                r"$m_D\equiv c_DD/X_T$; $X_S$ is serving expenditure, $\lambda$ the share of it the developer internalizes, $p$ the price of a "
                r"serving FLOP relative to a training FLOP, $\phi$ the size elasticity of serving cost; $\varepsilon_{A,P}$ is the elasticity "
                r"of adoption with respect to users' serving cost; $\nu,\nu_\tau,\mu\ge0$ are normalized multipliers; $\delta$ ($\delta_S$) is "
                r"the size elasticity of the price of a training (serving) FLOP; $\pi_i\propto\omega_i/w_i$ with $\omega_i$ the member's compute "
                r"share, $\bar w_H$ the compute-weighted harmonic mean of member wedges; $w_L$ and $w_{L'}$ are the log-MRTS of the "
                r"econometrician's and the lab's technology. The last column compares $\hat T\equiv3D(\hat w-1)$ with $\lambda pT$, the "
                r"internalized serving demand in training-FLOP units (rows other than (a) and (f) take $\phi=1$, $\eta=0$); ``Over iff $w_L>w_{L'}$'' "
                r"means that $\hat T$ overstates, which under factor bias is $\chi<0$. "
                rf"Every row was verified by brute-force maximization of the objective (maximum relative error "
                rf"{max(wc['relerr_general'].max(), wc['relerr_closed'].dropna().max()):.0e}; Online Appendix A, Propositions A8--A9).",
                r"Authors' derivations; \texttt{code/analysis/ra5\_theory}.", size=r"\scriptsize", tabcolsep="3pt")
    out["misspec"].to_csv(os.path.join(TABLES, "ra5_theory_misspec.csv"), index=False)
    fam = out["family"]
    fam.to_csv(os.path.join(TABLES, "ra5_theory_family.csv"), index=False)

    # ------------------------------------------------------------------ tables: model-free, finite grid
    out["modelfree"].to_csv(os.path.join(TABLES, "ra5_theory_modelfree.csv"), index=False)
    fg = out["finite_grid"]
    fg.to_csv(os.path.join(TABLES, "ra5_theory_finite_grid.csv"), index=False)
    body = []
    for r in fg.itertuples():
        body.append(rf"{r.half_width_lnN:.2f} & {r.N_ratio_span:,.1f} & {100 * r.rel_bias_curv_quadratic:.2f} & {r.lead_order_ratio:.3f} & "
                    rf"{r.sigma_star_quadratic:.4f} & {r.sigma_star_quartic:.4f} & {_fmt(r.argmin_bias, 5)} \\")
    write_table(os.path.join(TABLES, "ra5_theory_finite_grid.tex"), r"Finite-Grid Bias of the Model-Free $\sigma^*$ Estimator",
                "tab:ra5-finite-grid", r"@{}ccccccc@{}",
                [r"Half-width & $N$ span & Curvature & Actual/leading & $\hat\sigma^*$, & $\hat\sigma^*$, & Argmin \\",
                 r"($\ln N$) & (ratio) & bias (\%) & order bias & quadratic & quartic & bias ($\ln N$) \\"], body,
                rf"Nine equally spaced runs in $\ln N$, centered at the minimum of the \citet{{besiroglu2024chinchilla}} IsoFLOP profile at "
                rf"$10^{{21}}$ FLOP, noise-free. The model-free estimator is $1/\hat\sigma^*-1=\hat L_{{nn}}/(2|dL^*/dc|)$ with $\hat L_{{nn}}$ "
                rf"twice the quadratic (or quartic) coefficient and the exact frontier slope. True $\sigma^*={fg['sigma_star_true'].iloc[0]:.4f}$. "
                r"Leading-order bias of the quadratic coefficient: $(f^{(4)}/24)\,\kappa_4$ with $\kappa_4=(m_6-m_2m_4)/(m_4-m_2^2)$, "
                r"$m_k$ the grid's $k$-th moment; argmin bias $-(f'''/6)(m_4/m_2)/f''$ (Online Appendix A, Remark after Proposition A10). "
                r"Bias only: with seed and evaluation noise, quartic fits and narrow grids have higher variance.",
                r"Authors' calculations; \texttt{code/analysis/ra5\_theory}.")

    # ------------------------------------------------------------------ tables: singular information
    out["si_geom"].to_csv(os.path.join(TABLES, "ra5_theory_singular_geometry.csv"), index=False)
    out["si_split"].to_csv(os.path.join(TABLES, "ra5_theory_singular_split.csv"), index=False)
    mc, mcs = out["si_mc"], out["si_mc_slopes"]
    mc.to_csv(os.path.join(TABLES, "ra5_theory_singular_mc.csv"), index=False)
    mcs.to_csv(os.path.join(TABLES, "ra5_theory_singular_mc_slopes.csv"), index=False)
    body = []
    for tech, tag in [("Hoffmann (alpha != beta)", r"\textit{Panel A. $\alpha\neq\beta$ (Hoffmann et al.)}"),
                      ("equal exponents (alpha = beta)", r"\textit{Panel B. $\alpha=\beta$}")]:
        body.append(rf"\multicolumn{{6}}{{@{{}}l}}{{{tag}}} \\")
        for r in mc[mc["technology"] == tech].sort_values("tau", ascending=False).itertuples():
            body.append(rf"$10^{{{int(round(np.log10(r.tau)))}}}$ & {r.q25_unres:.1e} & {r.q75_unres:.1e} & {r.q90_unres:.1e} & "
                        rf"{r.q75_restricted:.1e} & {r.share_split:.2f} \\")
        sl = mcs[mcs["technology"] == tech].iloc[0]
        body.append(rf"Slope in $\ln\tau$ & {sl.slope_q25_unres:.2f} & {sl.slope_q75_unres:.2f} & {sl.slope_q90_unres:.2f} & "
                    rf"{sl.slope_q75_restricted:.2f} & \\[2pt]")
    write_table(os.path.join(TABLES, "ra5_theory_singular.tex"), r"Estimating $\sigma^*$ from On-Path Data under $\kappa=1$: Rates",
                "tab:ra5-singular", r"@{}lccccc@{}",
                [r" & \multicolumn{3}{c}{Unrestricted: quantiles of $|\hat\sigma^*-\sigma^*|$} & Restricted & Share with \\",
                 r"\cmidrule(lr){2-4}", r"Noise s.d.\ $\tau$ & 25th & 75th & 90th & 75th pct. & split rates \\"], body,
                rf"On-path data: 40 compute levels over four decades, Chinchilla technology ($\kappa=1$), Gaussian noise of s.d.\ $\tau$ on $L$ "
                rf"(a smaller $\tau$ mimics a larger sample, $n_{{\rm eff}}\propto\tau^{{-2}}$); {int(mc['R'].iloc[0])} replications per cell. "
                r"Unrestricted: least squares over $(E,A,B,\alpha,\beta)$ with $A,B$ bounded away from zero, solved by variable projection; "
                r"the error is the larger of the two exactly equivalent twin estimates. Restricted: imposes that the fitted technology's allocation "
                r"exponent equals the observed path slope, $\beta/(\alpha+\beta)=a$. Theory (Online Appendix A, Proposition A1(iii)): slope "
                r"$1/2$ ($n^{-1/4}$) for the upper quantiles of the unrestricted estimator, slope 1 ($n^{-1/2}$) for the restricted one; the "
                r"$n^{-1/4}$ rate is established by this simulation and a heuristic argument, not by a limit theorem.",
                r"Authors' simulations; \texttt{code/analysis/ra5\_theory}.")
    out["info_E"].to_csv(os.path.join(TABLES, "ra5_theory_info_Eunknown.csv"), index=False)

    # ------------------------------------------------------------------ tables: partial identification
    pi["bounds"].to_csv(os.path.join(TABLES, "ra5_theory_pi_bounds.csv"), index=False)
    pi["sign"].to_csv(os.path.join(TABLES, "ra5_theory_pi_sign.csv"), index=False)
    pi["models"].to_csv(os.path.join(TABLES, "ra5_theory_pi_models.csv"), index=False)
    pi["anchors"].to_csv(os.path.join(TABLES, "ra5_theory_pi_anchors.csv"), index=False)
    bd = pi["bounds"]
    body = []
    for aname, tag in [("Chinchilla (Epoch extraction)", "Panel A. Anchor: Chinchilla IsoFLOP minima (largest budget $2.9\\times10^{21}$ FLOP)"),
                       ("Meta Llama 3 (digitized)", "Panel B. Anchor: Llama 3 IsoFLOP minima (largest budget $10^{22}$ FLOP)")]:
        body.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{tag}}}}} \\")
        for r in s[s["anchor"] == aname].itertuples():
            b24 = bd[(bd["anchor"] == aname) & (bd["assumption"] == r.assumption) & (bd["C"] == 1e24)].iloc[0]
            er = f"$[{r.e_lo:.3f},{r.e_hi:.3f}]$" if r.e_lo != r.e_hi else f"${r.e_lo:.3f}$"
            ms = (f"[{_sig(b24.Mstar_lo)}, {_sig(b24.Mstar_hi)}]" if b24.Mstar_lo != b24.Mstar_hi else _sig(b24.Mstar_lo))
            wl, wh = r.median_w_lo_sig60_75, r.median_w_hi_sig60_75
            body.append(rf"{_esc(r.assumption).replace('M*', '$M^*$')} & {er} & {ms} & {r.share_over_identified:.2f} & "
                        rf"{r.share_under_identified:.2f} & {r.share_not_identified:.2f} & [{wl:.2f}, {wh:.1f}] \\")
    write_table(os.path.join(TABLES, "ra5_theory_pi_sign.tex"), r"Partial Identification of $M^*(C)$ beyond the Design: Illustration",
                "tab:ra5-pi", r"@{}>{\raggedright\arraybackslash}p{0.23\textwidth}cccccc@{}",
                [r" & Path slope & $M^*(10^{24})$ & \multicolumn{3}{c}{Share of models} & Median bounds \\",
                 r"\cmidrule(lr){4-6}", r"Assumption on $M^*(C)$ & $e$ & bounds & $w>1$ & $w<1$ & not identified & on $w$ \\"], body,
                r"Identified sets from Online Appendix A, Proposition A11: $\ln M^*(C)\in[\underline\mu+e_L\ln(C/C_J),\ \bar\mu+e_U\ln(C/C_J)]$ "
                r"for $C>C_J$, where $[\underline\mu,\bar\mu]$ is a 95 percent interval for $\ln M^*$ at the largest budget $C_J$ (path fitted "
                r"to the IsoFLOP minima by OLS; wild bootstrap-$t$ with HC2-rescaled residuals, HC2 standard errors and Webb weights, "
                r"$B=9{,}999$; simulated coverage at these nine- and ten-point designs is about 90 percent) and $e=d\ln M^*/d\ln C=1-2a$. "
                r"Normal inputs: $e\in[-1,1]$; cross-sweep range: $a\in[0.37,0.57]$; point extrapolation uses the fitted path without "
                r"sampling error. Shares are of the verified-sample models with $C>C_J$ (149 for Chinchilla, 137 for Llama 3), in each "
                r"model's own parameter and token conventions (units are not harmonized). The sign of $w-1$ equals the sign of "
                r"$\ln M-\ln M^*(C)$ for any technology with single-peaked IsoFLOP profiles; median bounds on $w$ use "
                r"$\ln w=(1/\sigma^*-1)\ln(M/M^*(C))$ with $\sigma^*\in[0.6,0.75]$ (the medians of the models' lower and upper bounds). "
                r"Illustration of the logic only; the paper's application with harmonized units and lab-own anchors is in Section IV.",
                r"Authors' calculations from \texttt{output/tables/m1\_chinchilla\_a2\_minima.csv}, "
                r"\texttt{m1\_chinchilla\_labs\_a2\_minima.csv} and \texttt{m3\_wedge\_models.csv}; \texttt{code/analysis/ra5\_theory}.",
                size=r"\scriptsize", tabcolsep="2.5pt")

    # ------------------------------------------------------------------ figures
    import figures_v2
    figures_v2.fig_pi_cone(pi)
    figures_v2.fig_singular(numeric_v2.si_curves(), mc)

    # ------------------------------------------------------------------ summary
    n_pass = sum(c.passed for c in RESULTS)
    n_fail = len(RESULTS) - n_pass
    verdicts = pd.Series([c.verdict for c in RESULTS]).value_counts().to_dict()
    with open(os.path.join(PROC, "run_summary.txt"), "w") as f:
        f.write(f"{n_pass} PASS, {n_fail} FAIL out of {len(RESULTS)} checks; verdicts {verdicts}\n")
    print("=" * 100)
    print(f"ra5_theory summary: {n_pass} PASS, {n_fail} FAIL out of {len(RESULTS)} checks; verdicts {verdicts}; {time.time() - t0:.0f}s")
    print("=" * 100)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
