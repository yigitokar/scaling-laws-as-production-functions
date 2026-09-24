"""Minimal AER-style LaTeX table writer (booktabs + threeparttable, no vertical rules)."""
from __future__ import annotations

import math
import os

from common import TABLES


def num(x, p=3):
    if x is None:
        return ""
    try:
        if isinstance(x, str):
            return x
        if not math.isfinite(float(x)):
            return ""
    except (TypeError, ValueError):
        return str(x)
    x = float(x)
    if abs(x) >= 1e4:
        out = f"{abs(x):,.0f}".replace(",", "{,}")
    else:
        out = f"{abs(x):.{p}f}"
    neg = x < 0 and float(out.replace("{,}", "")) != 0
    return ("$-$" if neg else "") + out


def se(x, p=3, br="()"):
    s = num(x, p)
    return f"{br[0]}{s}{br[1]}" if s else ""


def sci(x, p=2):
    if x is None or not math.isfinite(float(x)):
        return ""
    m, e = f"{float(x):.{p}e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def write_table(name, caption, label, colspec, header_rows, body_rows, notes, size="\\small", landscape=False):
    """header_rows / body_rows: lists of already-formatted LaTeX row strings (without trailing \\\\) or the
    special tokens 'MID' (\\midrule) and ('PANEL', text, ncols)."""
    L = []
    L.append("\\begin{table}[!htbp]")
    L.append("\\centering")
    L.append(size)
    L.append(f"\\caption{{{caption}}}")
    L.append(f"\\label{{{label}}}")
    L.append("\\begin{threeparttable}")
    L.append(f"\\begin{{tabular}}{{{colspec}}}")
    L.append("\\toprule")
    for r in header_rows:
        L.append(_row(r))
    L.append("\\midrule")
    for r in body_rows:
        L.append(_row(r))
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\begin{tablenotes}[flushleft]")
    L.append("\\footnotesize")
    L.append(f"\\item \\textit{{Notes:}} {notes}")
    L.append("\\end{tablenotes}")
    L.append("\\end{threeparttable}")
    L.append("\\end{table}")
    txt = "\n".join(L) + "\n"
    if landscape:
        txt = "\\begin{landscape}\n" + txt + "\\end{landscape}\n"
    with open(os.path.join(TABLES, name + ".tex"), "w") as f:
        f.write(txt)
    return txt


def _row(r):
    if r == "MID":
        return "\\midrule"
    if isinstance(r, tuple) and r[0] == "PANEL":
        return f"\\multicolumn{{{r[2]}}}{{l}}{{\\textit{{{r[1]}}}}} \\\\"
    if isinstance(r, tuple) and r[0] == "RAW":
        return r[1]
    if isinstance(r, (list, tuple)):
        return " & ".join(str(c) for c in r) + " \\\\"
    return r + " \\\\"
