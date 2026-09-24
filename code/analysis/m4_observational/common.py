"""Shared helpers for module m4_observational (observational vs experimental production functions).

Output measure used throughout (Ruan et al. 2024 / Mertens et al. 2026 convention, with a chance floor):
    adj = (acc - chance) / (1 - chance)       chance-adjusted accuracy in [0, 1]
    y   = logit(clip(adj, FLOOR, 1 - FLOOR))  "log odds of success above chance"
In a log-linear specification y = a + theta_N ln N + theta_D ln D, theta_x is the elasticity of the
odds of above-chance success with respect to input x (Cobb-Douglas in the odds), and a family intercept
is Hicks-neutral TFP on the odds scale (SYNTHESIS Sec. 4.E).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed", "m4_observational")
TABDIR = os.path.join(ROOT, "output", "tables")
FIGDIR = os.path.join(ROOT, "output", "figures")
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
for d in (PROC, TABDIR, FIGDIR):
    os.makedirs(d, exist_ok=True)

PREFIX = "m4_observational"
SEED = 20260923

# chance levels of the benchmarks (multiple-choice guessing rates)
CHANCE = {"hellaswag": 0.25, "arc_c": 0.25, "winogrande": 0.5, "mmlu": 0.25, "gsm8k": 0.0, "truthfulqa": 0.0,
          "arc_e": 0.25, "piqa": 0.5}
FLOOR = 0.01  # clip chance-adjusted accuracy to [FLOOR, 1-FLOOR] before the logit
CORE = ["hellaswag", "arc_c", "winogrande"]   # benchmarks measured in BOTH worlds (observational + experimental)


def adj_acc(acc, task):
    c = CHANCE[task]
    return (np.asarray(acc, float) - c) / (1.0 - c)


def logit_adj(acc, task, floor=FLOOR):
    a = np.clip(adj_acc(acc, task), floor, 1 - floor)
    return np.log(a / (1 - a))


def composite(df, tasks=CORE, floor=FLOOR):
    """Composite output: logit of the mean chance-adjusted accuracy over `tasks` (requires all present)."""
    A = np.column_stack([adj_acc(df[t], t) for t in tasks])
    m = np.nanmean(A, axis=1)
    m[np.isnan(A).any(axis=1)] = np.nan
    m = np.clip(m, floor, 1 - floor)
    return np.log(m / (1 - m))


def add_outputs(df):
    """Adds y_<task> (chance-adjusted logits) and y_core (composite of CORE tasks) to a frame with raw accs."""
    for t in CHANCE:
        if t in df:
            df["y_" + t] = logit_adj(df[t], t)
    if all(t in df for t in CORE):
        df["y_core"] = composite(df)
    return df


# ----------------------------------------------------------------------------- regression helpers
def ols(y, X, cluster=None, hc="HC1"):
    """OLS with HC1 or one-way cluster-robust (CR1) covariance. X: DataFrame (constant added by caller)."""
    import statsmodels.api as sm
    m = sm.OLS(np.asarray(y, float), X.astype(float), missing="drop")
    if cluster is None:
        return m.fit(cov_type=hc)
    g = pd.factorize(pd.Series(cluster))[0]
    return m.fit(cov_type="cluster", cov_kwds={"groups": g})


def demean(df, cols, by):
    """Within transformation (for one-way FE)."""
    out = df[cols] - df.groupby(by)[cols].transform("mean")
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def write_tex_table(path, header, rows, caption, label, notes, colspec=None, panel_rows=None, size="\\small"):
    """Writes a booktabs + threeparttable LaTeX fragment (AER style: no vertical rules).
    rows: list of lists of strings; a row equal to ['\\midrule'] is emitted verbatim;
    a row starting with '#' is a panel header spanning all columns."""
    ncol = len(header)
    colspec = colspec or ("l" + "c" * (ncol - 1))
    L = ["\\begin{table}[!htbp]", "\\centering", size, "\\begin{threeparttable}",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == ["\\midrule"]:
            L.append("\\midrule")
        elif len(r) == 1 and r[0].startswith("#"):
            L.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{r[0][1:]}}}}} \\\\")
        else:
            L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\begin{tablenotes}[flushleft]", "\\footnotesize",
          f"\\item \\textit{{Notes:}} {notes}", "\\end{tablenotes}", "\\end{threeparttable}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
