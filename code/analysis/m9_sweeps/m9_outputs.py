"""m9_outputs.py -- tables (csv + AER LaTeX: booktabs, footnotesize, AEA.cls tablenotes) and figures (aer_style: at most
three colors per panel; edu = BLUE, web = ORANGE) for module m9_sweeps."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import m9_common as mc
import m9_est as est
import m9_stages as st

import aer_style as S  # noqa: E402

COL = {"edu": S.BLUE, "web": S.ORANGE}
NAME = {"edu": "FineWeb-Edu", "web": "FineWeb", "wiki": "WikiText-103"}
CONV_LAB = {"P": "non-embedding $N$, actual FLOPs", "T": "total $N$, $C=6ND$"}
PENDING = "pending"
SAMPLE_TEX = {"main": "Main grid", "no25": "Drop $D=25$M", "floor4": "Four-layer floor"}


def _g(df, **kw):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        if k not in df:
            return pd.DataFrame()
        m &= (df[k] == v).values
    return df[m]


def _cell(df, stat, **kw):
    g = _g(df, statistic=stat, **kw)
    if g.empty:
        return None
    return g.iloc[0]


def fmt_est_ci(row, p=3, se=False):
    if row is None or not np.isfinite(row["estimate"]):
        return "--", ""
    a = mc.texnum(row["estimate"], p)
    b = mc.texci(row["ci_lo"], row["ci_hi"], p)
    if se and np.isfinite(row.get("se", np.nan)):
        a += f" ({mc.texnum(row['se'], p)})"
    return a, b


# ============================================================================ tables
def t_power(ctx):
    pw = ctx["power"]
    base = pw[pw.cell.isin(["chin_s002_r5", "chin_s005_r5", "chin_s010_r5"]) & (pw.chi_true == 0.22)]
    null = pw[pw.cell.isin(["chin_s002_r5", "chin_s005_r5", "chin_s010_r5"]) & (pw.chi_true == 0.0)]
    cols = [("chin_s002_r5", "0.002"), ("chin_s005_r5", "0.005"), ("chin_s010_r5", "0.010")]

    def v(stat, corpus, conv, what="mc_se", src=base, p=3):
        out = []
        for c, _ in cols:
            g = src[(src.cell == c) & (src.statistic == stat) & (src.corpus == corpus) & (src.conv == conv)]
            out.append(mc.texnum(g[what].values[0], p) if len(g) and np.isfinite(g[what].values[0]) else "--")
        return out

    lines = ["& \\multicolumn{3}{c}{Log-loss noise s.d.} \\\\", "\\cmidrule(lr){2-4}",
             "Statistic & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule",
             "\\multicolumn{4}{l}{\\emph{Panel A. Expected standard errors (Monte Carlo s.d.), FineWeb-Edu design}} \\\\"]
    for lab, stat, conv in (("$\\sigma^*_\\kappa$, convention P", "sig_kappa", "P"), ("$\\sigma^*_\\kappa$, convention T", "sig_kappa", "T"),
                            ("$\\sigma^*$ ($\\kappa=1$), convention P", "sig_chin", "P"),
                            ("Model-free $\\sigma^*$, P (FLOP path)", "sig_modelfree_P", "P"),
                            ("Model-free $\\sigma^*$, P ($w=1$)", "sig_modelfree_P6", "P"),
                            ("Model-free $\\sigma^*$, T", "sig_modelfree_T", "T"),
                            ("Q3 slope, P (with high-$M$ runs)", "q3_slope", "P"), ("Q3 slope, T (with high-$M$ runs)", "q3_slope", "T"),
                            ("Q3 slope, P (main grid only)", "q3_slope_mainonly", "P")):
        lines.append(f"{lab} & " + " & ".join(v(stat, "edu", conv)) + " \\\\")
    lines.append("\\multicolumn{4}{l}{\\emph{Panel B. Between-corpus tilt $\\hat\\chi$ (true $\\chi=0.22$)}} \\\\")
    lines.append("S.e. of $\\hat\\chi$, P & " + " & ".join(v("tilt_chi", "val1", "P")) + " \\\\")
    lines.append("S.e. of $\\hat\\chi$, T & " + " & ".join(v("tilt_chi", "val1", "T")) + " \\\\")
    lines.append("Power, one validation set (normal approx.), P & " + " & ".join(v("tilt_chi", "val1", "P", "power_normal", p=2)) + " \\\\")
    lines.append("Minimum detectable $\\chi$ (80\\% power), P & " + " & ".join(v("tilt_chi", "val1", "P", "mde80_normal")) + " \\\\")
    lines.append("Pr(``factor-biased''), wild bootstrap rule, P & " + " & ".join(v("tilt_chi_bootstrap", "val1+val2", "P", "decide_factor_biased", p=2)) + " \\\\")
    lines.append("Mean 95\\% CI half-width, wild bootstrap, P & " + " & ".join(v("tilt_chi_bootstrap", "val1+val2", "P", "ci_halfwidth_mean")) + " \\\\")
    lines.append("\\multicolumn{4}{l}{\\emph{Panel C. Under $\\chi=0$}} \\\\")
    lines.append("Pr(``factor-biased''), wild bootstrap rule, P & " + " & ".join(v("tilt_chi_bootstrap", "val1+val2", "P", "decide_factor_biased", src=null, p=2)) + " \\\\")
    lines.append("Pr(``neutral''), wild bootstrap rule, P & " + " & ".join(v("tilt_chi_bootstrap", "val1+val2", "P", "decide_neutral", src=null, p=2)) + " \\\\")
    notes = ("Monte Carlo from the design only (code/sweep/run\\_grid.py), computed before any estimation on the sweep "
             "results. Truth: Chinchilla form with Besiroglu et al.'s exponents ($\\sigma^*=0.737$), anchored at $M^*=20$ "
             "and 3.8 nats at the design's central compute; FineWeb-Edu has a data-augmenting shift $\\chi$ relative to "
             "FineWeb. Noise: $\\ln L$ error with a common within-trunk component (share $\\rho=0.5$ of the variance) and a "
             "second validation set with noise correlated 0.8. 150 replications per cell (Panels A--B, rows 1--4); the "
             "wild-bootstrap rows use 60 replications with 199 Webb draws each and the plan's decision rule on two "
             "validation sets. Sensitivity cells ($\\rho=0$, 0.9; $M^*=60$; $\\kappa$-family truth) are in "
             "m9\\_sweeps\\_power.csv.")
    mc.write_tex("m9_sweeps_power", "Power of the Controlled Experiment (Computed before Estimation)", "tab:m9-power",
                 "lccc", lines, notes)


def t_design(ctx):
    inv, ds = ctx["inv"], ctx["ds"]
    tags = ["main", "hiM", "lrsweep", "lrcal", "lrcorner", "seedcorner", "seeds"]
    lines = ["& \\multicolumn{2}{c}{FineWeb-Edu} & \\multicolumn{2}{c}{FineWeb} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Runs (tag) & Planned & Completed & Planned & Completed \\\\", "\\midrule",
             "\\multicolumn{5}{l}{\\emph{Panel A. Endpoints}} \\\\"]
    for t in tags:
        vals = []
        for r in mc.CORPORA:
            g = inv[(inv.regime == r) & (inv.tag == t)]
            vals += [str(int(g.planned.values[0])), str(int(g.completed.values[0]))] if len(g) else ["0", "0"]
        lines.append(f"\\texttt{{{t}}} & " + " & ".join(vals) + " \\\\")
    lines.append("\\multicolumn{5}{l}{\\emph{Panel B. Main grid: design statistics (convention P / T)}} \\\\")

    def dv(r, key, p=0, sample="main"):
        out = []
        for conv in mc.CONVS:
            g = ds[(ds.corpus == r) & (ds.conv == conv) & (ds.sample == sample)]
            out.append(mc.texnum(g[key].values[0], p) if len(g) else "--")
        return " / ".join(out)

    for lab, key, p in (("$M$ minimum", "M_min", 2), ("$M$ maximum", "M_max", 0), ("sd$(\\ln M\\mid\\ln C)$", "sd_lnM_given_lnC", 2),
                        ("Endpoints with $M>100$", "n_M_gt_100", 0), ("Endpoints with $M>1{,}000$", "n_M_gt_1000", 0)):
        lines.append(f"{lab} & \\multicolumn{{2}}{{c}}{{{dv('edu', key, p)}}} & \\multicolumn{{2}}{{c}}{{{dv('web', key, p)}}} \\\\")
    lines.append("\\multicolumn{5}{l}{\\emph{Panel C. Main grid plus high-$M$ runs (P / T)}} \\\\")
    for lab, key, p in (("$M$ maximum", "M_max", 0), ("Endpoints with $M>1{,}000$", "n_M_gt_1000", 0)):
        lines.append(f"{lab} & \\multicolumn{{2}}{{c}}{{{dv('edu', key, p, 'main+hiM')}}} & \\multicolumn{{2}}{{c}}{{{dv('web', key, p, 'main+hiM')}}} \\\\")
    notes = ("Planned endpoints from code/sweep/run\\_grid.py after the trimming of the high-$M$ runs; completed "
             "endpoints are those in data/processed/sweep/results.jsonl at the time of the run. P: non-embedding $N$ "
             "with actual training FLOPs; T: total $N$ (tied embedding) with $C=6ND$. $M=D/N$.")
    mc.write_tex("m9_sweeps_design", "The Controlled Experiment: Design and Completed Runs", "tab:m9-design", "lcccc", lines, notes)


def t_sigma(ctx):
    q1 = ctx["q1"]
    if q1 is None or q1.empty:
        return
    ests = [("Model-free, FLOP", "sig_mf_P", "P"), ("Model-free, $w=1$", "sig_mf_P6", "P"),
            ("Model-free, $w=1$", "sig_mf_T", "T"), ("$\\kappa$ family", "sig_kappa", None),
            ("$\\kappa$ family, FLOP path", "sig_kappa_Ppath", "P"), ("$\\kappa=1$", "sig_chin", None),
            ("$\\kappa$ family, NLS", "sig_kappa_nls", None), ("$\\kappa=1$, NLS", "sig_chin_nls", None),
            ("$\\hat\\kappa$", "kappa", None)]
    have_web = not _g(q1, corpus="web").empty
    corp = ["edu", "web", "edu-web"] if have_web else ["edu", "web"]
    hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{'Edu $-$ Web' if c == 'edu-web' else NAME[c]}}}" for c in corp)
    lines = [f"& {hdr} \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(corp))),
             "Estimator & " + " & ".join(["P & T"] * len(corp)) + " \\\\", "\\midrule",
             f"\\multicolumn{{{1 + 2 * len(corp)}}}{{l}}{{\\emph{{Panel A. Main grid, own validation set}}}} \\\\"]
    for sample in mc.SAMPLES:
        if sample != "main":
            lines.append(f"\\multicolumn{{{1 + 2 * len(corp)}}}{{l}}{{\\emph{{Panel {'B' if sample == 'no25' else 'C'}. "
                         f"{SAMPLE_TEX[sample]}}}}} \\\\")
        for lab, stat, only in ests:
            if sample != "main" and stat in ("sig_kappa_nls", "sig_chin_nls", "kappa", "sig_mf_P6", "sig_kappa_Ppath"):
                continue
            r1, r2, r3 = [lab], [""], [""]
            for c in corp:
                for conv in mc.CONVS:
                    if only is not None and conv != only:
                        r1.append("")
                        r2.append("")
                        r3.append("")
                        continue
                    row = _cell(q1, stat, corpus=c, conv=conv, sample=sample, scheme="wild")
                    row2 = _cell(q1, stat, corpus=c, conv=conv, sample=sample, scheme="cr2")
                    if row is None and c == "web":
                        r1.append(PENDING if stat != "kappa" else "--")
                        r2.append("")
                        r3.append("")
                        continue
                    a, b = fmt_est_ci(row, 3 if stat != "kappa" else 2)
                    _, b2 = fmt_est_ci(row2, 3 if stat != "kappa" else 2)
                    if c == "edu-web" and row is not None and np.isfinite(row.get("p_equal", np.nan)):
                        b = f"$p={row['p_equal']:.2f}$"
                        b2 = f"$p={row2['p_equal']:.2f}$" if row2 is not None and np.isfinite(row2.get("p_equal", np.nan)) else ""
                    r1.append(a)
                    r2.append(b)
                    r3.append(b2.replace("[", "\\{").replace("]", "\\}") if b2.startswith("[") else b2)
            lines.append(" & ".join(r1) + " \\\\")
            lines.append(" & ".join(r2) + " \\\\")
            if sample == "main" and stat != "kappa":
                lines.append(" & ".join(r3) + " \\\\")
    notes = ("$\\sigma^*$ is the elasticity of substitution between parameters and data on the compute-optimal path. "
             "Model-free: kernel-weighted local quadratic of $\\ln L$ in $(\\ln N,\\ln D)$ (Gaussian kernel, "
             "leave-one-out bandwidth), evaluated where the local wedge equals the path value on each isocost "
             "(convention P: $w=\\eta(N)=d\\ln \\mathrm{FLOPs}/d\\ln N$ on actual-FLOP isocosts, or $w=1$ on $6ND$ "
             "isocosts; T: $w=1$ on $6N_{\\rm tot}D$ isocosts), averaged over the compute levels whose isocost crosses at "
             "least three widths; the plan's identity $1/\\sigma^*-1=L_{nn}|_C/(2|dL^*/d\\ln C|)$ (generalized for "
             "P) gives the same numbers. $\\kappa$ family: $L=E+(AN^{-a_1}+BD^{-b_1})^{\\kappa}$, "
             "$\\sigma^*_\\kappa=2/(2+a_1+b_1)$. Brackets: 95 percent basic intervals from a wild cluster bootstrap by "
             "trunk (Webb weights, 999 draws; eight clusters per corpus), as pre-specified; braces (Panel A): the "
             "pre-declared size-corrected companion, the same bootstrap on CR2 leverage-adjusted cluster residuals "
             "with the bandwidth re-selected in each draw. On this design the plan's intervals under-cover (coverage "
             "0.83 for $\\sigma^*_\\kappa$ and 0.40--0.45 for the model-free estimates at nominal 0.95); the companion "
             "reaches 0.88 and 0.65--0.70 (Online Appendix power table). Difference column: $p$-value of $H_0$: "
             "equal $\\sigma^*$, from joint draws with a common weight per width. Output: bits per byte on the "
             "corpus's own validation set. ``pending'': the FineWeb grid was incomplete at the time of this run.")
    mc.write_tex("m9_sweeps_sigma", "Elasticity of Substitution by Corpus (Q1)", "tab:m9-sigma",
                 "l" + "cc" * len(corp), lines, notes + " Parametric fits by Huber loss unless marked NLS (Gaussian).",
                 colsep="1pt")


def t_tilt(ctx):
    q2, dec = ctx["q2"], ctx["dec"]
    lines = ["Val. set & Conv. & $\\hat\\chi$ & 95\\% CI & WCR $p$ & $\\alpha$ & $\\beta$ & "
             "$M^*$ ratio & $\\hat w$ factor & Wald $p$ \\\\", "\\midrule"]
    VN = {"edu": "Edu", "web": "Web", "wiki": "Wiki"}
    if q2 is None or q2.empty:
        lines.append("\\multicolumn{10}{l}{Pending: the FineWeb main grid had " +
                     str(int(((ctx['df'].regime == 'web') & (ctx['df'].tag == 'main')).sum())) +
                     " of 44 endpoints at the time of this run.} \\\\")
    else:
        for v in mc.VALSETS:
            for conv in mc.CONVS:
                row = _cell(q2, "chi", valset=v, conv=conv, sample="main", scheme="wild")
                if row is None:
                    continue
                row2 = _cell(q2, "chi", valset=v, conv=conv, sample="main", scheme="cr2")
                a = _cell(q2, "alpha", valset=v, conv=conv, sample="main", scheme="wild")
                b = _cell(q2, "beta", valset=v, conv=conv, sample="main", scheme="wild")
                mr = _cell(q2, "Mstar_ratio_w1", valset=v, conv=conv, sample="main", scheme="wild")
                wf = _cell(q2, "what_factor", valset=v, conv=conv, sample="main", scheme="wild")
                wd = _cell(q2, "wald_equal_exponents", valset=v, conv=conv, sample="main", scheme="wild")
                lines.append(f"{VN[v]} & {conv} & {mc.texnum(row.estimate)} & {mc.texci(row.ci_lo, row.ci_hi)} & "
                             f"{mc.texnum(row2.p_wcr, 3) if row2 is not None else '--'} & "
                             f"{mc.texnum(a.estimate)} & {mc.texnum(b.estimate)} & {mc.texnum(mr.estimate, 2)} & "
                             f"{mc.texnum(wf.estimate, 2)} & {mc.texnum(wd.p_zero, 2) if wd is not None else '--'} \\\\")
                if row2 is not None:
                    ci2 = mc.texci(row2.ci_lo, row2.ci_hi).replace("[", "\\{").replace("]", "\\}")
                    lines.append(f"& & & {ci2} & & & & & & \\\\")
        if dec is not None and len(dec) and "conv" in dec:
            lines.append("\\midrule")
            for _, d in dec[dec["sample"] == "main"].iterrows():
                lab = "plan rule" if d.scheme == "wild" else "size-corrected rule"
                lines.append(f"\\multicolumn{{10}}{{l}}{{Decision ({lab}), convention {d.conv}: {d.decision}}} \\\\")
    notes = ("Corpus-pair model with common exponents and corpus-specific $(E,A,B)$, Huber loss on $\\ln L$ (bits per "
             "byte) of both corpora's main-grid models evaluated on the same validation set. "
             "$\\hat\\chi=\\Delta\\ln(A/B)$, FineWeb-Edu minus FineWeb ($>0$: data-augmenting for FineWeb-Edu). "
             "$M^*$ ratio $=e^{-2\\hat\\chi/(\\alpha+\\beta)}$ on the $w=1$ path; $\\hat w$ factor $=e^{-\\hat\\chi}$ "
             "(Proposition 4); $M^*$ ratio: $M^*_{\\rm edu}/M^*_{\\rm web}$. Validation sets: FineWeb-Edu (Edu), FineWeb "
             "(Web), WikiText-103 (Wiki). Brackets: wild cluster bootstrap by width (Webb weights, 999 draws) on the fit's residuals, "
             "as pre-specified. Braces and WCR $p$: the pre-declared size-corrected companion (CR2 cluster residuals; "
             "restricted bootstrap with $\\chi=0$ imposed through the Hicks-neutral fit), which has size 0.05 on this "
             "design, where the plan's interval rejects a true $\\chi=0$ about 20 percent of the time. Wald test of equal "
             "$(\\alpha,\\beta)$ from separate fits with bootstrap covariance. Decision rule (pre-registered): "
             "factor-biased if the interval excludes zero on both corpora's validation sets with the same sign; neutral "
             "if it contains zero on both with half-width below 0.10; otherwise inconclusive (size-corrected rule: "
             "WCR $p<0.05$ in place of the interval).")
    mc.write_tex("m9_sweeps_tilt", "Is Data Quality Factor-Neutral? The Between-Corpus Tilt (Q2)", "tab:m9-tilt",
                 "llcccccccc", lines, notes)


def t_extrap(ctx):
    q3 = ctx["q3"]
    lines = ["& & & & & \\multicolumn{3}{c}{Mean $\\ln(w_{\\rm r}/w_{\\rm loc})$, $M$ in} & \\\\",
             "\\cmidrule(lr){6-8}",
             "Corpus & Conv. & Sample & Slope & 95\\% CI & 100--316 & 316--1{,}000 & $>1{,}000$ & $n$ \\\\", "\\midrule"]
    short = {"edu": "Edu", "web": "Web"}
    if q3 is None or q3.empty:
        lines.append("\\multicolumn{9}{l}{Pending.} \\\\")
    else:
        for r in mc.CORPORA:
            for conv in mc.CONVS:
                for sample in mc.SAMPLES:
                    row = _cell(q3, "slope|chin_r|local", corpus=r, conv=conv, sample=sample, scheme="wild")
                    if row is None:
                        if sample == "main":
                            lines.append(f"{short[r]} & {conv} & {SAMPLE_TEX[sample]} & {PENDING} & & & & & \\\\")
                        continue
                    bins = [_cell(q3, f"bin{j}|chin_r|local", corpus=r, conv=conv, sample=sample, scheme="wild") for j in range(3)]
                    n = _cell(q3, "n_eval|chin_r|local", corpus=r, conv=conv, sample=sample, scheme="wild")
                    lines.append(f"{short[r]} & {conv} & {SAMPLE_TEX[sample].replace('Four-layer floor', '4-layer floor')} & "
                                 f"{mc.texnum(row.estimate)} & {mc.texci(row.ci_lo, row.ci_hi)} & " +
                                 " & ".join(mc.texnum(b.estimate, 2) if b is not None else "--" for b in bins) +
                                 f" & {int(n.estimate) if n is not None else '--'} \\\\")
                    row2 = _cell(q3, "slope|chin_r|local", corpus=r, conv=conv, sample=sample, scheme="cr2")
                    if row2 is not None:
                        ci2 = mc.texci(row2.ci_lo, row2.ci_hi).replace("[", "\\{").replace("]", "\\}")
                        lines.append(f"& & & & {ci2} & & & & \\\\")
        lines.append("\\midrule")
        lines.append("\\multicolumn{9}{l}{\\emph{Comparators (main grid): slope of $\\ln(w_{\\rm a}/w_{\\rm b})$ in $\\ln M$}} \\\\")
        for r in mc.CORPORA:
            for conv in mc.CONVS:
                for key, lab in (("slope|kappa_r|local", "$\\kappa$, $M\\le100$ vs local"),
                                 ("slope|chin_f|local", "full support vs local"),
                                 ("slope|chin_r|chin_f", "$M\\le100$ vs full support")):
                    row = _cell(q3, key, corpus=r, conv=conv, sample="main", scheme="wild")
                    if row is None:
                        continue
                    lines.append(f"\\multicolumn{{3}}{{l}}{{{short[r]}, {conv}: {lab}}} & {mc.texnum(row.estimate)} & "
                                 f"{mc.texci(row.ci_lo, row.ci_hi)} & & & & \\\\")
    notes = ("Q3 of the plan. Edu: FineWeb-Edu; Web: FineWeb. The Chinchilla form (Huber) is fitted on main-grid endpoints "
             "with $M\\le100$ and predicts the wedge $w=\\varepsilon_N/\\varepsilon_D$ at every endpoint with $M>100$, "
             "including the four-layer high-$M$ runs once available; $w_{\\rm local}$ comes from the local quadratic on the "
             "full grid. Slope: OLS of $\\ln(w_{\\rm r}/w_{\\rm local})$ on $\\ln M$. A negative slope means the restricted "
             "fit increasingly understates the wedge (overstates the value of extra tokens) as $M$ grows. Brackets: wild "
             "cluster bootstrap by trunk around the local-quadratic surface (999 draws, as pre-specified); braces: the "
             "pre-declared companion (CR2 cluster residuals, bandwidth re-selected in each draw). Both exclude the smoothing "
             "bias of the local estimator. On this design the slope's null value (Chinchilla form true in non-embedding "
             "units) is about $-0.01$ to $-0.05$ in convention P and $-0.08$ to $-0.10$ in T without the high-$M$ runs "
             "($+0.11$ with them); the plan's interval covers about 55 percent of the time (companion: 70--75 percent). "
             "Levels need not transfer to production scale; the sign is the object.")
    mc.write_tex("m9_sweeps_extrap", "Extrapolating the Wedge beyond the Chinchilla Support (Q3)", "tab:m9-extrap",
                 "lllcccccc", lines, notes)


def t_noise_lr(ctx):
    q4, q5 = ctx["q4"], ctx["q5"]
    lines = ["& FineWeb-Edu & FineWeb \\\\", "\\midrule", "\\multicolumn{3}{l}{\\emph{Panel A. Noise (Q4): s.d. of $\\ln L$}} \\\\"]
    for lab, key, filt in (("Seed s.d., pooled (own validation set)", "pooled_sd_lnL", dict()),
                           ("Within-trunk correlation of seed deviations", "rho_within_trunk", dict())):
        vals = []
        for r in mc.CORPORA:
            g = q4[(q4.corpus == r) & (q4.valset == r)] if len(q4) and "valset" in q4 else pd.DataFrame()
            vals.append(mc.texnum(g[key].values[0], 4) if len(g) and key in g and np.isfinite(g[key].values[0]) else PENDING)
        lines.append(f"{lab} & " + " & ".join(vals) + " \\\\")
    for conv in mc.CONVS:
        for fit, lab in (("chin_huber", "Chinchilla"), ("kappa_huber", "$\\kappa$ family")):
            vals = []
            for r in mc.CORPORA:
                g = q4[(q4.corpus == r) & (q4.get("conv") == conv) & (q4.get("fit") == fit)] if len(q4) and "fit" in q4 else pd.DataFrame()
                vals.append(mc.texnum(g.residual_sd.values[0], 4) if len(g) else PENDING)
            lines.append(f"Residual s.d., {lab}, {conv} & " + " & ".join(vals) + " \\\\")
    lines.append("\\multicolumn{3}{l}{\\emph{Panel B. Learning rate (Q5)}} \\\\")
    cal = q5.get("calibration", pd.DataFrame())
    for d in (128, 320, 512):
        vals = []
        for r in mc.CORPORA:
            g = cal[(cal.corpus == r) & (cal.d == d)] if len(cal) else pd.DataFrame()
            vals.append(f"{g.lr_star_quad.values[0] * 1e3:.2f}" if len(g) else PENDING)
        lines.append(f"Optimal peak LR ($\\times10^{{-3}}$) at $D=50$M, width {d} & " + " & ".join(vals) + " \\\\")
    for d in (128, 320, 512):
        vals = []
        for r in mc.CORPORA:
            g = cal[(cal.corpus == r) & (cal.d == d)] if len(cal) else pd.DataFrame()
            vals.append(mc.texnum(g.excess_rule_vs_quad.values[0], 4) if len(g) and np.isfinite(g.excess_rule_vs_quad.values[0]) else "--")
        lines.append(f"Excess $\\ln L$ of the rule at $D=50$M, width {d} & " + " & ".join(vals) + " \\\\")
    cor = q5.get("corners", pd.DataFrame())
    for d, D_t in ((128, 800_000_000), (128, 400_000_000), (128, 25_000_000), (640, 25_000_000), (640, 50_000_000)):
        vals = []
        for r in mc.CORPORA:
            g = cor[(cor.corpus == r) & (cor.d == d) & (cor.D_target == D_t) & (cor.valset == r)] if len(cor) else pd.DataFrame()
            vals.append(mc.texnum(g.excess_rule_quad.values[0], 4) if len(g) and "excess_rule_quad" in g and np.isfinite(g.excess_rule_quad.values[0]) else PENDING)
        lines.append(f"Excess $\\ln L$ of the rule, corner ({d}, {D_t / 1e6:.0f}M) & " + " & ".join(vals) + " \\\\")
    dr = q5.get("drift", pd.DataFrame())
    vals = []
    for r in mc.CORPORA:
        g = dr[dr.corpus == r] if len(dr) else pd.DataFrame()
        vals.append(f"{mc.texnum(g.dlnLRstar_dlnD.values[0])} ({mc.texnum(g.se.values[0])})" if len(g) else PENDING)
    lines.append("$d\\ln \\mathrm{LR}^*/d\\ln D$ at width 128 & " + " & ".join(vals) + " \\\\")
    ab = q5.get("abias", pd.DataFrame())
    for conv in mc.CONVS:
        vals = []
        for r in mc.CORPORA:
            g = ab[(ab.corpus == r) & (ab.conv == conv)] if len(ab) else pd.DataFrame()
            vals.append(mc.texnum(g.a_obs_minus_a.values[0], 4) if len(g) else PENDING)
        lines.append(f"Implied bias $a_{{\\rm obs}}-a$, {conv} & " + " & ".join(vals) + " \\\\")
    notes = ("Panel A: seed replicates (seeds 1--2 at widths 128/256/384, $D\\in\\{50,100,200\\}$M; seed 1 at width 128, "
             "$D\\in\\{200,400,800\\}$M) together with the seed-0 run of the same cell. Residual s.d.: Huber fits on the "
             "main grid (own validation set). Panel B: optimal LR from a quadratic in $\\ln$ LR through the four "
             "calibration LRs (FineWeb-Edu: \\texttt{lrsweep}; FineWeb: \\texttt{lrcal}), the rule used for both corpora "
             "being $3.07\\times10^{-3}(d/256)^{-0.90}$. Corners: LR $\\times\\{0.5,1,2\\}$; excess = $\\ln L$ at the rule "
             "minus the minimum of the quadratic (or the best tested LR). Bias bound: Online Appendix equation "
             "(app-flexbias), $a_{\\rm obs}-a=-(\\iota_n-\\iota_d)E/[(\\alpha+\\beta)R^*]$, with the gradient from "
             "the two corners at similar compute, (640, 25M) and (128, 800M).")
    mc.write_tex("m9_sweeps_noise_lr", "Noise and Learning-Rate Checks (Q4--Q5)", "tab:m9-noise-lr", "lcc", lines, notes)


def t_onpath(ctx):
    q6 = ctx["q6"]
    if q6 is None or q6.empty:
        return
    lines = ["Corpus & Conv. & Subsample & $n$ & cond$(J'J)$ & sd $\\ln(M/M^*)$ & 95\\% set, $\\sigma^*_\\kappa$ & Flat? \\\\",
             "\\midrule"]
    lab = {"full": "Full grid", "onpath1": "On path, 1/bin", "onpath2": "On path, 2/bin"}
    for _, r in q6.iterrows():
        setv = f"[{r.set_lo:.2f}, {r.set_hi:.2f}]" if np.isfinite(r.set_lo) else "empty"
        lines.append(f"{NAME[r.corpus]} & {r.conv} & {lab.get(r.subset, r.subset)} & {r.n} & {mc.texnum(r.cond_normalized, 0)} & "
                     f"{r.sd_lnM_dev_from_path:.2f} & {setv} & {'yes' if r.flat else 'no'} \\\\")
    notes = ("Q6. On-path subsample: in each doubling bin of compute, the endpoint(s) nearest the $\\kappa$-family "
             "expansion path fitted on the full grid. Condition number of $J'J$ for the Chinchilla form at the full-grid "
             "estimate (KMW-normalized). Profile: Gaussian likelihood ratio over $\\sigma^*_\\kappa$ on a grid over "
             "(0.05, 0.99), relative to the unconstrained $\\kappa$-family optimum (Section II Monte Carlo machinery); "
             "``flat'' means no grid value is rejected at 5 percent.")
    mc.write_tex("m9_sweeps_onpath", "Functional Dependence with Real Losses (Q6)", "tab:m9-onpath", "lllccccc", lines, notes)


def all_tables(ctx):
    for f in (t_power, t_design, t_sigma, t_tilt, t_extrap, t_noise_lr, t_onpath):
        try:
            f(ctx)
        except Exception as e:  # never lose the other tables
            import traceback
            mc.log(f"table {f.__name__} failed: {e!r}\n{traceback.format_exc()}")


# ============================================================================ figures
def _ax_style(ax):
    from matplotlib.ticker import NullFormatter
    ax.tick_params(length=2.5)
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_formatter(NullFormatter())
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_formatter(NullFormatter())


def fig_main(ctx, conv_b="T", name="m9_sweeps_fig", valset="edu"):
    """(a) bpb vs D by width for both corpora on a common output with fitted curves; (b) isoquants + expansion paths."""
    import matplotlib.pyplot as plt
    S.use()
    df = ctx["df"]
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 3.0))
    ax = axs[0]
    col = f"bpb_{valset}"
    fits = {}
    for r in mc.CORPORA:
        s = mc.main_sample(df, r, "main")
        s = s[np.isfinite(s[col])]
        if s.empty:
            continue
        # fitted curve on the common output: CE fit if available, else a separate Huber fit (convention P)
        if (valset, "P") in ctx["q2fits"]:
            thr = est.ce_corpus_theta(ctx["q2fits"][(valset, "P")], 1 if r == "edu" else 0)
        elif len(s) >= st.MIN_POINTS and s.d.nunique() >= st.MIN_WIDTHS:
            thr, _ = est.fit_chin(s.N_P.values, s.D.values, s[col].values, "huber", grid="fast")
        else:
            thr = None
        fits[r] = thr
        for d_, g in s.groupby("d"):
            g = g.sort_values("D")
            ax.plot(g.D, g[col], "o", ms=2.6, color=COL[r], alpha=0.9, mec="none")
            if thr is not None:
                Dg = np.exp(np.linspace(np.log(g.D.min() * 0.9), np.log(g.D.max() * 1.1), 60))
                ax.plot(Dg, np.exp(est.pred_chin(thr, np.full_like(Dg, g.N_P.values[0]), Dg)), "-", lw=0.9, color=COL[r])
            if r == "edu" and d_ in (128, 192, 256, 384, 640):
                ax.annotate(str(d_), (g.D.values[-1], g[col].values[-1]), xytext=(3, 0), textcoords="offset points",
                            fontsize=6, color=S.INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens $D$")
    ax.set_ylabel(f"Loss on {NAME[valset]} validation (bits per byte)")
    ax.set_title("(a) Loss against data, by width", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL[r], marker="o", ms=3, lw=0.9 if fits.get(r) is not None else 0,
                      label=NAME[r] + ("" if fits.get(r) is not None else " (grid incomplete; no fit)"))
               for r in mc.CORPORA]
    ax.legend(handles=handles, loc="upper right")
    _ax_style(ax)
    # (b) isoquants and expansion paths in convention conv_b
    ax = axs[1]
    for r in mc.CORPORA:
        s = mc.main_sample(df, r, "main")
        s = s[np.isfinite(s[col])]
        if s.empty:
            continue
        ax.plot(s[f"N_{conv_b}"], s.D, "o", ms=2.2, color=S.MUTED, mec="none", alpha=0.6, zorder=1)
    Nmin = min(mc.main_sample(df, r, "main")[f"N_{conv_b}"].min() for r in mc.CORPORA if len(mc.main_sample(df, r, "main")))
    Nmax = max(mc.main_sample(df, r, "main")[f"N_{conv_b}"].max() for r in mc.CORPORA if len(mc.main_sample(df, r, "main")))
    lnN = np.linspace(np.log(Nmin / 2), np.log(Nmax * 2), 300)
    levels = None
    for r in mc.CORPORA:
        s = mc.main_sample(df, r, "main")
        s = s[np.isfinite(s[col])]
        if (valset, conv_b) in ctx["q2fits"]:
            thr = est.ce_corpus_theta(ctx["q2fits"][(valset, conv_b)], 1 if r == "edu" else 0)
        elif len(s) >= st.MIN_POINTS and s.d.nunique() >= st.MIN_WIDTHS:
            thr, _ = est.fit_chin(s[f"N_{conv_b}"].values, s.D.values, s[col].values, "huber", grid="fast")
        else:
            continue
        m = mc.sl.Chinchilla.from_theta(thr)
        if levels is None:
            levels = np.quantile(s[col].values, [0.2, 0.5, 0.8])
        for j, lv in enumerate(levels):
            R = lv - m.E
            u = m.A * np.exp(-m.alpha * lnN)
            ok = R - u > 0
            Dq = ((R - u[ok]) / m.B) ** (-1 / m.beta)
            ax.plot(np.exp(lnN[ok]), Dq, "-", lw=1.0, color=COL[r], zorder=3)
            if r == "edu":
                # label each isoquant where it crosses D = 1.2e9 (inside the frame), in bits per byte
                k = int(np.argmin(np.abs(np.log(Dq) - np.log(1.2e9))))
                ax.annotate(f"{lv:.2f}" + (" bpb" if j == 0 else ""), (np.exp(lnN[ok][k]), Dq[k]), fontsize=6,
                            color=S.INK2, xytext=(3, 2), textcoords="offset points")
        Cg = np.exp(np.linspace(np.log(3e14), np.log(3e17), 50))
        ax.plot(m.N_opt(Cg), m.D_opt(Cg), "--", lw=1.6, color=COL[r], zorder=4)
        ax.annotate(f"{NAME[r]} expansion path ($w=1$)", (m.N_opt(1.5e17), m.D_opt(1.5e17)), fontsize=6.5,
                    color=COL[r], xytext=(-4, 4), textcoords="offset points", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1.5e7, 3e9)
    ax.set_xlim(Nmin / 2, Nmax * 2)
    ax.set_xlabel("Parameters $N$ (" + ("total, incl. embedding" if conv_b == "T" else "non-embedding") + ")")
    ax.set_ylabel("Training tokens $D$")
    ax.set_title("(b) Isoquants and expansion paths", loc="left")
    _ax_style(ax)
    fig.tight_layout()
    S.savefig(fig, name, folder=mc.FIGS)


def fig_sigma(ctx):
    import matplotlib.pyplot as plt
    S.use()
    lv, q1 = ctx["q1lvl"], ctx["q1"]
    if lv is None or lv.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.7), sharey=True)
    for ax, (conv, path, title) in zip(axs, (("P", "P", "(a) Non-embedding $N$, actual FLOPs"), ("T", "T", "(b) Total $N$, $C=6ND$"))):
        for k, r in enumerate(mc.CORPORA):
            g = lv[(lv.corpus == r) & (lv.conv == conv) & (lv.path == path) & (lv["sample"] == "main") & (lv.scheme == "wild")]
            if g.empty:
                continue
            off = 1.0 + 0.04 * (k - 0.5)
            ax.errorbar(g.C * off, g.sigma, yerr=[g.sigma - g.ci_lo, g.ci_hi - g.sigma], fmt="o", ms=3, lw=0.8,
                        color=COL[r], capsize=0, label=f"{NAME[r]}, model-free")
            kap = _cell(q1, "sig_kappa_Ppath" if conv == "P" else "sig_kappa", corpus=r, conv=conv, sample="main", scheme="wild")
            chn = _cell(q1, "sig_chin_Ppath" if conv == "P" else "sig_chin", corpus=r, conv=conv, sample="main", scheme="wild")
            if kap is not None:
                ax.axhline(kap.estimate, color=COL[r], lw=1.0, ls="-", alpha=0.8, label=f"{NAME[r]}, $\\kappa$ family")
            if chn is not None:
                ax.axhline(chn.estimate, color=COL[r], lw=1.0, ls=":", label=f"{NAME[r]}, $\\kappa=1$")
        ax.axhline(0.70, color=S.MUTED, lw=0.8, ls="--")
        ax.annotate("0.70 (public designs)", (0.02, 0.70), xycoords=("axes fraction", "data"), fontsize=6, color=S.INK2,
                    xytext=(0, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Compute $C$ (FLOPs)")
        ax.set_title(title, loc="left")
        _ax_style(ax)
    axs[0].set_ylabel("$\\sigma^*$ on the compute-optimal path")
    axs[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    S.savefig(fig, "m9_sweeps_sigma", folder=mc.FIGS)


def fig_wedge(ctx):
    import matplotlib.pyplot as plt
    S.use()
    pts, q3 = ctx["q3pts"], ctx["q3"]
    if pts is None or pts.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.8), sharey=True)
    for ax, conv in zip(axs, mc.CONVS):
        for r in mc.CORPORA:
            g = pts[(pts.corpus == r) & (pts.conv == conv) & (pts["sample"] == "main") & (pts.M > 100)]
            if g.empty:
                continue
            for tag, mk in (("main", "o"), ("hiM", "s")):
                h = g[g.tag == tag]
                if h.empty:
                    continue
                ax.errorbar(h.M, h.delta, yerr=[h.delta - h.delta_lo, h.delta_hi - h.delta], fmt=mk, ms=3, lw=0.7,
                            color=COL[r], capsize=0, mfc=COL[r] if tag == "main" else "white",
                            label=f"{NAME[r]}" + (", high-$M$ runs" if tag == "hiM" else ""))
            row = _cell(q3, "slope|chin_r|local", corpus=r, conv=conv, sample="main", scheme="wild")
            lvl = _cell(q3, "level100|chin_r|local", corpus=r, conv=conv, sample="main", scheme="wild")
            if row is not None and lvl is not None:
                Mg = np.exp(np.linspace(np.log(100), np.log(g.M.max() * 1.1), 50))
                ax.plot(Mg, lvl.estimate + row.estimate * np.log(Mg / 100), "-", color=COL[r], lw=1.2)
                ax.annotate(f"slope {row.estimate:.2f} [{row.ci_lo:.2f}, {row.ci_hi:.2f}]", (0.03, 0.06 + 0.09 * (r == "web")),
                            xycoords="axes fraction", fontsize=6.5, color=S.INK2)
        ax.axhline(0, color=S.INK2, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens per parameter $M=D/N$")
        ax.set_title("(a) Non-embedding $N$" if conv == "P" else "(b) Total $N$", loc="left")
        _ax_style(ax)
    axs[0].set_ylabel("$\\ln(w_{M\\leq100}/w_{\\rm local})$")
    axs[0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    S.savefig(fig, "m9_sweeps_wedge", folder=mc.FIGS)


def fig_lr(ctx):
    import matplotlib.pyplot as plt
    S.use()
    q5 = ctx["q5"]
    cal, cor = q5.get("calibration", pd.DataFrame()), q5.get("corners", pd.DataFrame())
    df = ctx["df"]
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.7))
    ax = axs[0]
    for r, tag in (("edu", "lrsweep"), ("web", "lrcal")):
        s = df[(df.regime == r) & (df.tag == tag)]
        for d_, g in s.groupby("d"):
            g = g.sort_values("lr")
            y = mc.own_output(g)
            ax.plot(g.lr, np.log(y) - np.log(y).min(), "-o", ms=2.5, lw=0.9, color=COL[r])
            ax.annotate(str(d_), (g.lr.values[-1], (np.log(y) - np.log(y).min())[-1]), fontsize=6, color=S.INK2,
                        xytext=(2, 0), textcoords="offset points")
        rule = df[(df.regime == r) & (df.tag == "main") & (df.D_target == 50_000_000)]
    import run_grid as rg
    for d_ in (128, 320, 512):
        ax.axvline(rg.lr_rule(d_), color=S.MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.005, 0.08)
    ax.set_xlabel("Peak learning rate")
    ax.set_ylabel("$\\ln L - \\min \\ln L$ (own validation set)")
    ax.set_title("(a) Calibration at $D=50$M, widths 128/320/512", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COL[r], marker="o", ms=3, label=NAME[r]) for r in mc.CORPORA] +
              [Line2D([], [], color=S.MUTED, ls=":", label="rule")], loc="upper center", fontsize=6.5)
    _ax_style(ax)
    ax = axs[1]
    if len(cor) and "excess_rule_quad" in cor:
        for r in mc.CORPORA:
            for d_, mk in ((128, "o"), (640, "s")):
                g = cor[(cor.corpus == r) & (cor.valset == r) & (cor.d == d_)].sort_values("D_target")
                if g.empty or "excess_rule_quad" not in g:
                    continue
                ax.plot(g.D_target, g.excess_rule_quad, "-" + mk, ms=3, lw=0.9, color=COL[r],
                        mfc=COL[r] if d_ == 128 else "white", label=f"{NAME[r]}, width {d_}")
    q4 = ctx["q4"]
    if len(q4) and "pooled_sd_lnL" in q4:
        for r in mc.CORPORA:
            g = q4[(q4.corpus == r) & (q4.valset == r)]
            if len(g):
                ax.axhline(g.pooled_sd_lnL.values[0], color=COL[r], lw=0.7, ls="--")
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens $D$")
    ax.set_ylabel("Excess $\\ln L$ of the LR rule")
    ax.set_title("(b) Corners: LR $\\times\\{0.5, 1, 2\\}$", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)
    _ax_style(ax)
    fig.tight_layout()
    S.savefig(fig, "m9_sweeps_lr", folder=mc.FIGS)


def fig_profile(ctx):
    import matplotlib.pyplot as plt
    S.use()
    pr = ctx["q6prof"]
    if pr is None or pr.empty:
        return
    corp = [r for r in mc.CORPORA if (pr.corpus == r).any()]
    fig, axs = plt.subplots(len(corp), 2, figsize=(S.WIDTH_FULL, 2.5 * len(corp)), squeeze=False, sharey=True)
    for i, r in enumerate(corp):
        for j, conv in enumerate(mc.CONVS):
            ax = axs[i, j]
            for sub, c, lab in (("full", S.BLUE, "full grid"), ("onpath1", S.ORANGE, "on path (1 per bin)"),
                                ("onpath2", S.AQUA, "on path (2 per bin)")):
                g = pr[(pr.corpus == r) & (pr.conv == conv) & (pr.subset == sub)]
                if len(g):
                    ax.plot(g.sigma_star, np.minimum(g.LR, 40), "-", color=c, lw=1.2, label=lab)
            ax.axhline(3.84, color=S.MUTED, ls="--", lw=0.7)
            ax.set_ylim(-0.5, 40)
            ax.set_title(f"({'abcd'[2 * i + j]}) {NAME[r]}, {'non-embedding' if conv == 'P' else 'total'} $N$", loc="left")
            ax.set_xlabel("$\\sigma^*_\\kappa$ (restricted)")
            _ax_style(ax)
        axs[i, 0].set_ylabel("Likelihood ratio")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    S.savefig(fig, "m9_sweeps_profile", folder=mc.FIGS)


def fig_power(ctx):
    import matplotlib.pyplot as plt
    S.use()
    pw = ctx["power"]
    cells = [("chin_s002_r5", 0.002), ("chin_s005_r5", 0.005), ("chin_s010_r5", 0.010)]
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.6))
    ax = axs[0]
    for (stat, corpus, conv), c, lab in ((("sig_kappa", "edu", "P"), S.BLUE, "$\\sigma^*_\\kappa$"),
                                         (("sig_modelfree_P", "edu", "P"), S.ORANGE, "model-free $\\sigma^*$"),
                                         (("tilt_chi", "val1", "P"), S.AQUA, "tilt $\\hat\\chi$")):
        ys = []
        for cell, _ in cells:
            g = pw[(pw.cell == cell) & (pw.chi_true == 0.22) & (pw.statistic == stat) & (pw.corpus == corpus) & (pw.conv == conv)]
            ys.append(g.mc_se.values[0] if len(g) else np.nan)
        ax.plot([s for _, s in cells], ys, "-o", color=c, ms=3, lw=1.2, label=lab)
    ax.set_xlabel("Log-loss noise s.d.")
    ax.set_ylabel("Expected standard error")
    ax.set_title("(a) Precision, convention P", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    _ax_style(ax)
    ax = axs[1]
    for (stat, key, chi), c, lab in ((("tilt_chi_bootstrap", "decide_factor_biased", 0.22), S.BLUE, "'factor-biased' when $\\chi=0.22$"),
                                     (("tilt_chi_bootstrap", "decide_factor_biased", 0.0), S.ORANGE, "'factor-biased' when $\\chi=0$"),
                                     (("tilt_chi_bootstrap", "reject_val1", 0.0), S.AQUA, "one-set rejection when $\\chi=0$")):
        ys = []
        for cell, _ in cells:
            g = pw[(pw.cell == cell) & (pw.chi_true == chi) & (pw.statistic == stat) & (pw.conv == "P")]
            ys.append(g[key].values[0] if len(g) else np.nan)
        ax.plot([s for _, s in cells], ys, "-o", color=c, ms=3, lw=1.2, label=lab)
    ax.axhline(0.05, color=S.MUTED, lw=0.8, ls="--")
    ax.annotate("nominal 0.05", (0.0021, 0.05), fontsize=6.5, color=S.INK2, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Log-loss noise s.d.")
    ax.set_ylabel("Probability")
    ax.set_title("(b) Plan's tilt rule (wild bootstrap, 8 clusters)", loc="left")
    ax.legend(loc="center right", fontsize=6.5)
    _ax_style(ax)
    fig.tight_layout()
    S.savefig(fig, "m9_sweeps_power", folder=mc.FIGS)


def all_figures(ctx):
    for f, kw in ((fig_main, {}), (fig_main, dict(conv_b="P", name="m9_sweeps_fig_P")),
                  (fig_main, dict(valset="web", name="m9_sweeps_fig_webval")),
                  (fig_sigma, {}), (fig_wedge, {}), (fig_lr, {}), (fig_profile, {}), (fig_power, {})):
        try:
            f(ctx, **kw)
        except Exception as e:
            import traceback
            mc.log(f"figure {f.__name__} {kw} failed: {e!r}\n{traceback.format_exc()}")
