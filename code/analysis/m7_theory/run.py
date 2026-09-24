"""m7_theory — verify every formal claim of the paper's theory and generate the module's tables/figures.

Usage:  /Users/yigitokar/scaling-laws-pf/.venv/bin/python code/analysis/m7_theory/run.py

What it does (deterministic; about 4 minutes on one CPU):
  1. symbolic.run_all(): sympy proofs of every algebraic claim in paper/notes/model_spec.md and SYNTHESIS P1-P8,
     plus the new results (SOC => sigma<1, wedge sufficient statistics, Harberger form, information decomposition, proxy).
  2. numeric.run_all(): brute-force optimization, Monte Carlo and Fisher-information checks of the same claims,
     counterexamples for the claims found false/imprecise, a bootstrap illustration of the partial-identification result,
     and re-computation of the SYNTHESIS §2 ledger numbers that are pure applications of the formulas.
  3. Writes output/tables/m7_theory_*.csv/.tex, output/figures/m7_theory_*.pdf/.png, data/processed/m7_theory/*.csv.
Prints PASS/FAIL per claim; exits with status 1 if any check fails.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common  # noqa: E402
from common import PROC, TABLES, RESULTS, as_records  # noqa: E402


def _tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("^", r"\^{}").replace("~", r"\~{}"))


def write_tex(path, header, rows, colspec, caption, label, notes, font=r"\small", ragged_cols=()):
    r"""booktabs + threeparttable table. ragged_cols: indices of p{} columns typeset ragged-right (rows then end with
    \tabularnewline because \raggedright redefines \\; main.tex does not load the array package)."""
    lines = [r"\begin{table}[htbp]", r"\centering", font, r"\begin{threeparttable}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append(r"\midrule")
        elif isinstance(r, str):
            lines.append(r)
        else:
            cells = [(r"\raggedright " + x) if i in ragged_cols else x for i, x in enumerate(r)]
            lines.append(" & ".join(cells) + (r" \tabularnewline" if ragged_cols else r" \\"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize", rf"\item \textit{{Notes:}} {notes}",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# Condensed, paper-facing claim register (LaTeX). Statements are written by hand here so that the table reads well;
# the full machine-generated register is m7_theory_claims.csv.
CLAIMS_TEX = [
    ("Model and elasticities", None),
    ("MS.T", "model spec", r"$y=\omega+F(n+\psi_N,d+\psi_D)-\epsilon$ holds exactly given $E$", "Imprecise"),
    ("MS.SIG", "model spec", r"$\sigma=(\alpha u+\beta v)/(\alpha u(1+\beta)+\beta v(1+\alpha))$; $\sigma^*=2/(2+\alpha+\beta)$", "Verified"),
    ("MS.SIGW", "new", r"$\sigma(w)=(1+w)/(1+\alpha+w(1+\beta))$: local $\sigma$ is a function of the wedge", "New"),
    ("SOC", "new", r"Interior compute optimum under $C=6ND$ (strict SOC) $\Leftrightarrow$ $0<\sigma<1$ at the optimum", "New"),
    ("Allocation and geometry", None),
    ("L1", "Lemma 1", r"$n^*,d^*,y^*$ with $(\omega,\psi_N,\psi_D)$; better data $\Rightarrow$ more $N$, lower $D/N$", "Verified (given $c$)"),
    ("L1.target", "Lemma 1", r"At a fixed loss target: $\partial n^*/\partial\psi_D=0$; $\omega$ moves the mix if $\alpha\neq\beta$", "Imprecise"),
    ("L2", "Lemma 2 / P2", r"Hessian of $\ln R$ rank one, null vector $(\beta,\alpha)$", "Verified"),
    ("L2.dec", "new", r"$\ln R=-(\alpha n+\beta d)/2+g(\tau)$; transverse coordinate $=-\ln w$", "New"),
    ("Identification", None),
    ("P1.path", "P1", r"Duality: $C^*(R)$, $N^*$, $D^*$, $K$ (both forms)", "Verified"),
    ("P2.rank", "Prop. 1 / P2", r"On-path Jacobian rank 3; $\sigma^*$ first-order estimable only if $\alpha=\beta$", "Verified"),
    ("P5.DMR.num", "Props. 1, 5; P2, P3", r"Explicit $\kappa$-family: identical paths/frontiers for all $\sigma^*\in(0,1)$", "Verified"),
    ("P5.ident", "Prop. 5 / P3", r"$g_N,g_D$ identified separately; bias magnitude not, bias \emph{sign} identified", "Corrected"),
    ("INFO.dec", "new", r"$y=y^*(c)-\gamma\Phi_S+\omega-\epsilon$, $\Phi=\ln(C/C_{\min})$; $\mathcal I(S)\lesssim(\gamma/Ss)^2\sum\Phi_i^2$, $S=\alpha+\beta$", "New"),
    ("INFO.num", "new", r"$\mathcal I(\sigma^*)\sim v^4$, $\mathcal I(\ln M^*)\sim v^2$ in transverse dispersion $v$", "New"),
    ("P2.plim", "Prop. 2 / P5", r"OLS slope $\gamma+\pi_1V_\omega/(\pi_1^2V_\omega+V_\eta)$ for any sign of $\pi_1$", "Verified"),
    ("P2.bounds", "P5", r"Forward/reverse bracket $\gamma$ iff $-(1+V_\epsilon/V_\omega)/\gamma\le\pi_1\le0$", "Corrected"),
    ("P3.sel", "Prop. 3", r"Outcome-based release attenuates the slope; in $[0,\gamma]$ if log-concave", "Verified"),
    ("P3.net", "Prop. 3", r"Net sign under funding rule: $>0$ iff $r>r^*$ (closed form)", "New"),
    ("P3.cex", "Prop. 3", r"$E[\omega\mid c,\text{released}]$ need not fall in $c$ when $\pi_1>0$", "Corrected"),
    ("PROXY", "new", r"$\ln(D/N)\mid c$ inverts $\chi=\beta\psi_D-\alpha\psi_N$; flat in Hicks-neutral $\omega$", "New"),
    ("Wedges and technical change", None),
    ("P4.foc", "Prop. 4 / P6", r"$w=1+T/(3D)$; invariant to $\omega$, $\epsilon$, $E$", "Verified"),
    ("P4.contam", "Prop. 4", r"$\hat w=w\,e^{\alpha\psi_N-\beta\psi_D}$", "Verified"),
    ("HOM", "P6", r"$T/D=3[(M/M^*)^\rho-1]$; $C/C_{\min}=\cosh(\tfrac{\rho}{2}\ln\tfrac{M}{M^*})^{2/\rho}$", "Verified"),
    ("W1", "new", r"Any $\alpha,\beta$: $w=(M/M^*(C))^{1/\sigma^*-1}$", "New"),
    ("W2", "new", r"$C/C_{\min}=((\alpha+\beta w)/(\alpha+\beta))^{1/\gamma}w^{-1/\alpha}$; $\approx\exp[(\ln w)^2/(2(\alpha+\beta))]$", "New"),
    ("D.constr", "task 2(d)", r"Binding $D\le\bar D$: $w=1/(\theta_D+\mu)<1+T/3D$; binding $N\le\bar N$: $w>1+T/3D$", "Verified"),
    ("PI.mono", "new", r"Concave, supermodular $F$: $w$ monotone in $(-n,d)$; lower bound on $T$ off-support", "New"),
    ("P5.rates", "Prop. 5 / P3", r"Rates of $K_t$, $D^*/N^*$; Hicks bias $\beta g_D-\alpha g_N$", "Verified"),
    ("P4S.CEG", "P4", r"Constant frontier CEG iff equal $E$ and equal $\gamma$ (not equal $\alpha,\beta$)", "False; corrected"),
    ("P4S.CEG.rule", "P4", r"At a common rule $D=mN$: equal $E$ and exponents necessary, not sufficient (need a common rescaling of $N$)", "False; corrected"),
    ("P7", "P7", r"Sahal: naive slope $\gamma/(1-s_A)$; with noise $\gamma[1+(g_A/g)R^2]$", "Verified"),
    ("P8", "P8", r"Hall-type bias $(\varepsilon_N-\varepsilon_D)(\Delta n-\Delta d)/2$", "Verified"),
]


def main():
    t0 = time.time()
    print("=" * 100)
    print("m7_theory: symbolic verification (sympy)")
    print("=" * 100)
    import symbolic
    symbolic.run_all()
    print("=" * 100)
    print("m7_theory: numerical / Monte Carlo verification")
    print("=" * 100)
    import numeric
    out = numeric.run_all()

    # ------------------------------------------------------------------ tables
    reg = pd.DataFrame(as_records())
    reg.to_csv(os.path.join(TABLES, "m7_theory_claims.csv"), index=False)
    out["ledger"].to_csv(os.path.join(TABLES, "m7_theory_ledger_checks.csv"), index=False)
    out["fisher"].to_csv(os.path.join(TABLES, "m7_theory_fisher_information.csv"), index=False)
    out["pi_band"].to_csv(os.path.join(TABLES, "m7_theory_pi_band.csv"), index=False)
    pd.DataFrame([out["pi_summary"]]).to_csv(os.path.join(TABLES, "m7_theory_pi_summary.csv"), index=False)
    out["dmr_family"].to_csv(os.path.join(TABLES, "m7_theory_dmr_family.csv"), index=False)
    out["constraints"].to_csv(os.path.join(TABLES, "m7_theory_constraints.csv"), index=False)
    out["harberger"].to_csv(os.path.join(TABLES, "m7_theory_harberger.csv"), index=False)
    out["transmission"].to_csv(os.path.join(TABLES, "m7_theory_transmission_mc.csv"), index=False)
    pd.concat([out["selection_i"], out["selection_net"]], axis=0, ignore_index=True).to_csv(
        os.path.join(TABLES, "m7_theory_selection_mc.csv"), index=False)
    out["wedge_cases"].to_csv(os.path.join(PROC, "wedge_bruteforce_cases.csv"), index=False)

    # claims register (LaTeX, condensed)
    status = {c.cid: c for c in RESULTS}
    rows = []
    for item in CLAIMS_TEX:
        if item[1] is None:
            rows.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{item[0]}}}}} \\")
            continue
        cid, src, stmt, verdict = item
        chk = status.get(cid)
        mark = "" if chk is None else ("" if chk.passed else " (check failed)")
        rows.append((cid.replace(".num", "").replace("_", r"\_"), src, stmt, verdict + mark))
    write_tex(os.path.join(TABLES, "m7_theory_claims.tex"), ["ID", "Source", "Statement (as verified or corrected)", "Verdict"], rows,
              r"@{}l p{0.13\textwidth} p{0.47\textwidth} p{0.14\textwidth}@{}", "Verification of the formal claims", "tab:m7_claims",
              r"Each statement was checked symbolically (sympy) and numerically (brute-force optimization, Monte Carlo with "
              r"$n\geq 4\times10^5$, or Fisher-information computations); scripts in \texttt{code/analysis/m7\_theory}. "
              r"``Source'' refers to the project's model specification (Lemma/Proposition numbers of the draft) and the literature "
              r"synthesis (P1--P8). ``Corrected'' and ``False'' entries give the corrected statement; the original wording is in the "
              r"module memo. The full register with error metrics is \texttt{m7\_theory\_claims.csv}.", font=r"\footnotesize",
              ragged_cols=(1, 2, 3))

    # observationally equivalent family (LaTeX)
    ddf = out["dmr_family"]
    rows = []
    for r in ddf.itertuples():
        rows.append((f"{r.S:.3f}", f"{r.sigma_star:.3f}", f"{r.kappa:.3f}", f"{r.hicks_bias:.3f}", f"{r.pct_above_frontier_M5x:.1f}"))
    write_tex(os.path.join(TABLES, "m7_theory_dmr_family.tex"),
              [r"$S=a_1+b_1$", r"$\sigma^*$", r"$\kappa$", r"Hicks bias $b_1g_D-a_1g_N$", r"Loss above frontier at $M=5M^*$ (\%)"], rows,
              "@{}ccccc@{}", "Observationally equivalent technologies on the expansion path", "tab:m7_dmr",
              r"Members of the family $L=E+(A N^{-a_1}+B D^{-b_1})^{\kappa}$ with $a_1=(1-a)S$, $b_1=aS$, $\kappa=\gamma/(a(1-a)S)$ and $(A,B)$ "
              r"chosen to reproduce the Besiroglu et al. (2024) expansion path and frontier ($a=0.513$, $\gamma=0.178$). With factor-augmenting "
              r"progress $g_N=0.25$, $g_D=0.60$ per year, brute-force compute-optimal frontiers coincide across members to "
              r"$10^{-14}$ in $\ln R^*$ and the argmin $n^*$ to $4\times10^{-7}$ (optimizer tolerance) at six dates and seven budgets. "
              r"The row with $\kappa=1$ is the Chinchilla law. Hicks bias is $d\ln\mathrm{MRTS}/dt$ per year at fixed inputs. Last column: "
              r"reducible loss of a run with five times the compute-optimal tokens per parameter, in percent above the common frontier at the "
              r"same compute ($C=10^{22}$).")

    # wedge sufficient statistic + partial identification example (LaTeX)
    hb = out["harberger"]
    pib = out["pi_band"]
    pis = out["pi_summary"]
    rows = []
    def _num(x, fmt=".2f"):
        # AER typesetting: a true minus sign for negative entries
        return f"${x:{fmt}}$" if x < 0 else f"{x:{fmt}}"

    for r in hb.itertuples():
        rows.append((f"{r.w:.2f}", _num(3 * (r.w - 1)), f"{r.C_over_Cmin_exact:.2f}", f"{r.C_over_Cmin_harberger:.2f}"))
    rows.append("MIDRULE")
    rows.append(r"\multicolumn{4}{l}{\textit{Revealed $T/D$ at $C=7.2\times10^{23}$ (Chinchilla technology), bootstrap 95\% band}} \\")
    rows.append((r"$M$", r"$T/D$", r"[2.5\%, 97.5\%]", r"se$(\ln\hat w)$"))
    for Mv in (20, 100, 341, 1000, 1875, 5000):
        i = int(np.argmin(np.abs(pib["M"].values - Mv)))
        rr = pib.iloc[i]
        rows.append((f"{rr.M:.0f}", _num(rr.T_over_D), f"[{_num(rr.lo95)}, {_num(rr.hi95)}]", f"{rr.se_ln_w:.3f}"))
    write_tex(os.path.join(TABLES, "m7_theory_wedge.tex"), [r"$w$", r"$T/D=3(w-1)$", r"$C/C_{\min}$ exact", r"$C/C_{\min}$ Harberger"], rows,
              "@{}cccc@{}", "The wedge as a sufficient statistic, and extrapolation uncertainty", "tab:m7_wedge",
              r"Top panel: Farrell allocative loss $C/C_{\min}=((\alpha+\beta w)/(\alpha+\beta))^{1/\gamma}w^{-1/\alpha}$ and its second-order "
              r"(Harberger) approximation $\exp[(\ln w)^2/(2(\alpha+\beta))]$ at the Besiroglu et al. exponents. Bottom panel: $T/D$ implied at "
              rf"fixed compute by the Huber fit to the Epoch Chinchilla extraction ($n=240$), with a pairs bootstrap ($B={pis['B_boot']}$). "
              rf"The design's tokens-per-parameter ratios reach $M={pis['M_design_max']:.0f}$, but only at compute up to "
              rf"${pis['C_design_max'] / 10 ** np.floor(np.log10(pis['C_design_max'])):.1f}\times10^{{{int(np.floor(np.log10(pis['C_design_max'])))}}}$ FLOP, "
              rf"so every entry is an extrapolation in compute. At $M=1875$ the s.d. of $\ln\hat w$ is {pis['sd_lnw_at_1875']:.3f}, "
              rf"of which the compute-optimal ratio $M^*$ alone contributes {pis['sd_lnw_Mstar_part']:.3f} and $\alpha+\beta$ alone "
              rf"{pis['sd_lnw_S_part']:.3f}. Technology in MassiveText tokens; illustrative for other corpora.")

    # ------------------------------------------------------------------ figures
    import figures
    figures.fig_geometry()
    sd_ch = figures.fig_information(out["fisher"])
    figures.fig_wedge(out["pi_band"], out["pi_summary"])
    figures.fig_bias()
    with open(os.path.join(PROC, "chinchilla_transverse_sd.txt"), "w") as f:
        f.write(f"sd(ln w) across the 240 Chinchilla-extraction runs (estimation sample) under Besiroglu technology: {sd_ch:.4f}\n")

    # ------------------------------------------------------------------ summary
    n_pass = sum(c.passed for c in RESULTS)
    n_fail = len(RESULTS) - n_pass
    verdicts = pd.Series([c.verdict for c in RESULTS]).value_counts().to_dict()
    print("=" * 100)
    print(f"m7_theory summary: {n_pass} PASS, {n_fail} FAIL out of {len(RESULTS)} checks; verdicts {verdicts}; {time.time() - t0:.0f}s")
    print("=" * 100)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
