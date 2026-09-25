"""exhibits.py -- the breakdown-frontier figure (S3), the regenerated appendix sign table (S5) and the paper numbers.

Figure: output/figures/rb5_sign_breakdown.{pdf,png}; WP4b inputs it in Appendix E with \\label{fig:app-breakdown}.
Table: output/tables/rb5_sign_table.tex (label tab:app-sign); WP4b copies it into paper/tables/appF_sign.tex.
No em dashes anywhere in generated text (author's rule); rounding half-up.
"""
from __future__ import annotations

import datetime
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

import rb5common as K


def r2(x, nd=2):
    if x is None or not np.isfinite(x):
        return "--"
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    s = f"{v:.{nd}f}"
    return "0.00" if s in ("-0.00",) else s


def fmt_num(x, nd=1):
    """Numbers for LaTeX text: one decimal below 100, integers above (thousands separated by {,})."""
    if abs(x) >= 100:
        return f"{round(x):,d}".replace(",", "{,}")
    return r2(x, nd)


# ------------------------------------------------------------------------------------------------ figure (S3)
def figure(FR, n_models, n_small, n_dec, units_label, experiment_factor=None, name="rb5_sign_breakdown"):
    import matplotlib.pyplot as plt
    import aer_style as S
    S.use()
    panels = [("all models", f"a. All models ({n_models})"),
              ("models below 15B", f"b. Models below 15 billion parameters ({n_small})"),
              (f"decisions ({units_label})", f"c. Allocation decisions ({n_dec})"),
              ("training compute (models)", "d. Training compute (models)")]
    sets = [("S1", "Primary: joint bands of the designs' paths", S.BLUE, "-"),
            ("S2(c)", "PI-1 anchors, slopes at their 95 percent intervals", S.ORANGE, "--")]
    fig, axes = plt.subplots(2, 2, figsize=(S.WIDTH_FULL, 4.6), sharex=True, sharey=True)
    for ax, (ser, title) in zip(axes.ravel(), panels):
        for key, lab, col, ls in sets:
            g = FR[(FR["set"] == key) & (FR["series"] == ser)].sort_values("tau")
            ax.plot(g["tau"], g["share"], color=col, ls=ls, lw=1.4, drawstyle="steps-post", label=lab)
        ax.axhline(0.5, color=S.MUTED, lw=0.6, ls=":")
        for t in K.TAU_MARKS:
            ax.axvline(t, color=S.MUTED, lw=0.6, ls=":")
        if experiment_factor is not None and np.isfinite(experiment_factor):
            ax.axvline(experiment_factor, color=S.AQUA, lw=1.0, ls="-.", label="Experiment's implied factor")
        ax.set_xscale("log")
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 1.3, 1.84, 3.4, 5, 10, 20])
        ax.set_xticklabels(["1", "1.3", "1.84", "3.4", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel(r"Tilt allowance $\tau$ on $M^*$ (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Share with $w>1$ identified")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    S.savefig(fig, name)


# ------------------------------------------------------------------------------------------------ table (S5)
TAU_U = [1.0, 1.3, 1.84, 3.4, 4.4, 10.0]
TAU_C = [1.0, 1.84, 3.4, 10.0]


def _row(SH, set_key, level, label, year="all"):
    g = SH[(SH["set"] == set_key) & (SH["level"] == level) & (SH["year"] == year)].set_index("tau")
    u = [r2(g.loc[t, "share_gt1"]) for t in TAU_U]
    c = [r2(g.loc[t, "share_gt1_cw"]) for t in TAU_C]
    return label + " & " + " & ".join(u) + " & & " + " & ".join(c) + r" \\"


def table(SH, V, info, units_label, n_units, n_units56, n_models, n_small, years_n, path_note):
    rows_models = [
        ("S1", "Primary: joint bands of the paths"),
        ("S1", f"\\quad Models below 15B parameters ($n={n_small}$)", "models below 15B"),
        ("S2(a)", "PI-1: point slopes, Comma excluded"),
        ("S2(a)+Comma", "\\quad PI-1 with Marin Comma"),
        ("S2(b)", "$t$-interval levels, Comma admitted"),
        ("S2(c)", "PI-1, slopes at 95 percent intervals"),
        ("S2(d)", "Normal inputs only ($|e|\\le1$)"),
        ("S2(e)", "Every technology as an anchor"),
        ("S2(f)-total", "All anchors in total parameters"),
        ("S2(f)-NF", "All in FLOP-effective parameters"),
    ]
    lines = []
    lines.append(r"\multicolumn{12}{@{}l}{\textit{Panel A. Models (" + str(n_models) + r")}} \\[1pt]")
    for r in rows_models:
        key, lab = r[0], r[1]
        lev = r[2] if len(r) > 2 else "models"
        lines.append(_row(SH, key, lev, lab))
    lines.append(r"\addlinespace")
    dec = f"decisions ({units_label})"
    lines.append(r"\multicolumn{12}{@{}l}{\textit{Panel B. Allocation decisions (" + str(n_units) + r")}} \\[1pt]")
    for r in rows_models:
        if len(r) > 2:
            continue
        lines.append(_row(SH, r[0], dec, r[1]))
    lines.append(_row(SH, "S1", "decisions (56)", f"\\quad Primary, {n_units56} family-label units"))
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{12}{@{}l}{\textit{Panel C. Primary, by release year (models)}} \\[1pt]")
    for y, n in years_n:
        lines.append(_row(SH, "S1", "models", f"{y} ($n={n}$)", year=str(y)))
    body = "\n".join(lines)
    today = datetime.date.today().isoformat()
    head = (
        "% Online Appendix table: share of the clean sample for which w > 1 is identified beyond the designs, by tilt\n"
        f"% allowance. Generated by code/analysis/rb5_sign/run.py (module rb5_sign, round 3, WP4a) on {today} from\n"
        "% output/tables/rb5_sign_shares.csv; half-up rounding. WP4b copies it into paper/tables/appF_sign.tex.\n")
    tex = head + r"""\begin{table}[tp]
\centering
\caption{Sign Identification beyond the Designs, by Tilt Allowance}
\label{tab:app-sign}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrcrrrr@{}}
\toprule
& \multicolumn{6}{c}{Share identified, tilt allowance $\tau$} & & \multicolumn{4}{c}{Share of compute, $\tau$} \\
\cmidrule(lr){2-7}\cmidrule(lr){9-12}
Identified set & 1 & 1.3 & 1.84 & 3.4 & 4.4 & 10 & & 1 & 1.84 & 3.4 & 10 \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular*}
\par\smallskip
\begin{tablenotes}
""" + path_note + r"""
\end{tablenotes}
\begin{tablenotes}[Source]
Author's calculations.
\end{tablenotes}
\end{table}
"""
    K.assert_no_em_dash(tex, "rb5_sign_table.tex")
    with open(os.path.join(K.TABLES, "rb5_sign_table.tex"), "w") as f:
        f.write(tex)
    return tex


def table_note(N):
    """The note, with every number taken from the computed objects N (a dict)."""
    s = (
        r"Share of the clean sample for which $w>1$ is identified beyond the designs "
        r"(Proposition~\ref{prop:A-pi}(iii), (vii) and (viii)): $\ln M$ exceeds the upper end of the identified set for "
        r"$\ln M^*(C)$ at the model's compute by at least $\ln\tau$; no model is identified as $w<1$ in any row. "
        r"Tilt allowances $\tau$: 1.3, tokenizer differences; 1.84, the largest tilt across DataDecide's recipes, 0.26, "
        r"which corresponds to $\tau\approx1.8$ under the reference exponents (we use 1.84); 3.4, the same tilt under "
        r"DataDecide's own exponents; 4.4, both 1.3 and 3.4; Figure~\ref{fig:app-breakdown} traces $\tau$ up to 20. "
        r"\textit{Primary:} one path per study, a line through the logged IsoFLOP minima at its bracketed budgets, "
        r"extrapolated with a simultaneous 95 percent band for its level and slope (wild bootstrap-$t$, Webb weights, "
        rf"{N['B']:,} draws, within-study pooled residual variance, critical value at least the normal-theory one) over "
        rf"{N['band_lo']} (Marin: from $3\times10^{{20}}$) to {N['band_hi']} FLOP; brackets give pointwise 95 percent "
        rf"intervals. Chinchilla: {N['chin_n']} budgets to $3\times10^{{21}}$ FLOP, total "
        rf"parameters as digitized (the convention of the models' $M$), level {N['chin_M']} [{N['chin_Mlo']}, "
        rf"{N['chin_Mhi']}] at $3\times10^{{21}}$ FLOP, slope ${N['chin_e']}$ $[{N['chin_elo']},{N['chin_ehi']}]$. "
        rf"Llama~3: {N['llama_n']} budgets to $10^{{21}}$, FLOP-implied $N_F=C/(6D)$, level {N['llama_M']} "
        rf"[{N['llama_Mlo']}, {N['llama_Mhi']}], slope ${N['llama_e']}$ $[{N['llama_elo']},{N['llama_ehi']}]$. "
        rf"Marin: DCLM and Nemotron-CC ({N['dclm_n']} budgets each) and Comma ({N['comma_n']}) to $3\times10^{{20}}$, "
        rf"FLOP-implied, corpus levels {N['dclm_M']}, {N['nemo_M']} and {N['comma_M']} and one slope, ${N['marin_e']}$ "
        rf"$[{N['marin_elo']},{N['marin_ehi']}]$, with a pooled residual variance ({N['marin_df']} degrees of freedom). "
        r"DeepSeek's published law enters as a point path in its own convention. The identified set is the union of the "
        rf"bands widened by $\ln\tau$; for $M^*(10^{{24}})$ it is [{N['ms24_lo']}, {N['ms24_hi']}] and for "
        rf"$M^*(10^{{25}})$ [{N['ms25_lo']}, {N['ms25_hi']}]. "
        r"\textit{PI-1} (version 3): bootstrap-$t$ intervals for each design's level, slopes bounded by the point "
        rf"estimates of the paths and published laws, $[{N['pi1_eL']},{N['pi1_eU']}]$; Marin Comma (five budgets, "
        rf"interval $[{N['comma_pi1_lo']},{N['comma_pi1_hi']}]$) was excluded by a rule of at least six budgets chosen "
        r"after that interval was seen. "
        r"\textit{$t$-interval levels:} levels on $t$-intervals with $k-2$ degrees of freedom, Comma admitted with its "
        rf"own slope, ${N['comma_t_e']}$; slopes in $[{N['tb_eL']},{N['tb_eU']}]$. "
        rf"\textit{{Slopes at 95 percent intervals:}} PI-1's levels, slopes in $[{N['c_eL']},{N['c_eU']}]$. "
        r"\textit{Normal inputs only:} the primary levels, $|e|\le1$. "
        r"\textit{Every technology:} the primary bands plus every in-set technology's in-support $M^*$, slopes in "
        rf"$[{N['e_eL']},{N['e_eU']}]$. "
        r"\textit{Conventions:} models' $M$ is in total parameters except in the rows in FLOP-effective parameters. In "
        rf"total parameters, Marin's minima are converted with its measured ratio of FLOP-implied to configuration "
        rf"parameters ({N['marin_r_lo']} to {N['marin_r_hi']} at the anchors), and Llama~3's band is shifted by the range "
        rf"of $\ln(N_F/N)$ at its anchor over three FLOP accountings, $[{N['l3_lr_lo']},{N['l3_lr_hi']}]$. In "
        r"FLOP-effective parameters, Chinchilla's minima and the models' parameters follow Hoffmann et al.'s count "
        r"(attention at 4,096 tokens for the models). "
        r"Compute-weighted: shares of training compute $6ND$. Allocation decisions: members of one family that share "
        r"one token budget (within 10 percent) form one decision; a decision is identified only if every member is."
    )
    K.assert_no_em_dash(s, "table note")
    return s
