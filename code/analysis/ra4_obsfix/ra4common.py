"""Shared paths and helpers for revision module ra4_obsfix (Online Appendix E fixes).

Named ra4common (not common) because the module imports code from m4_observational and m5_progress, each of which
has its own `common.py`; the steps that need m4 and the steps that need m5 run in separate processes (see run.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ANALYSIS = os.path.join(ROOT, "code", "analysis")
M4DIR = os.path.join(ANALYSIS, "m4_observational")
M5DIR = os.path.join(ANALYSIS, "m5_progress")
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed", "ra4_obsfix")
TAB = os.path.join(ROOT, "output", "tables")
FIG = os.path.join(ROOT, "output", "figures")
for _d in (PROC, TAB, FIG):
    os.makedirs(_d, exist_ok=True)
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

PREFIX = "ra4_obsfix"
SEED = 20260924
B_WILD = 9999          # wild (cluster) bootstrap draws for p-values; p-value floor 1/B
B_SURF = 9999          # design-conditional wild bootstrap draws of the experimental surface (linear, vectorized)
B_TOBIT = 499          # parametric bootstrap draws of the censored (Tobit) surfaces
B_ALLOC = 999          # bootstrap draws for the allocative gains
B_TFP = 999            # developer-cluster bootstrap draws for TFP dispersion
FLOOR_LOGIT = float(np.log(0.01 / 0.99))   # m4 clips chance-adjusted accuracy to [0.01, 0.99]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def log(msg):
    print(f"[ra4 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(x, nd=3, dash=""):
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "$\\infty$" if xf == np.inf else dash
    if abs(xf) < 0.5 * 10 ** (-nd):
        xf = 0.0
    s = f"{xf:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def fmt_m(x, nd=3):
    """Number for use INSIDE math mode (plain minus sign)."""
    s = fmt(x, nd)
    return s.replace("$-$", "-").replace("$\\infty$", "\\infty")


def fmt_p(p, B=None):
    """p-value string; at the floor 1/B report '<' the floor."""
    if p is None or not np.isfinite(p):
        return ""
    if B is not None and p <= 1.0 / B + 1e-12:
        return f"$<${1.0 / B:.4f}".rstrip("0")
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def file_info(path):
    if not os.path.exists(path):
        return dict(file=os.path.relpath(path, ROOT), exists=False)
    return dict(file=os.path.relpath(path, ROOT), exists=True, sha256=sha256(path),
                mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path))))


def dump_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    with open(os.path.join(PROC, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def write_aea_table(path, caption, label, colspec, header_rows, body_rows, notes, source=None, tabcolsep="3pt",
                    size="\\footnotesize", placement="tp"):
    """AEA-style LaTeX table (booktabs; AEA.cls `tablenotes` environment, not threeparttable).
    header_rows / body_rows: lists of already-formatted LaTeX row strings WITHOUT the trailing '\\\\' for normal rows;
    a row that starts with a rule or spacing command ('\\midrule', '\\addlinespace', '\\cmidrule') is emitted verbatim."""
    L = [f"\\begin{{table}}[{placement}]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    verb = ("\\midrule", "\\addlinespace", "\\cmidrule", "\\toprule", "\\bottomrule")
    for r in header_rows:
        L.append(r if r.startswith(verb) else r + " \\\\")
    if header_rows:          # reviewer fix: no \\toprule\\midrule double rule when a table has no header rows
        L.append("\\midrule")
    for r in body_rows:
        L.append(r if r.startswith(verb) else r + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "", "\\begin{tablenotes}", notes, "\\end{tablenotes}"]
    if source:
        L += ["\\begin{tablenotes}[Source]", source, "\\end{tablenotes}"]
    L.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def panel_row(text, ncol):
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{text}}}}}"


def texesc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
