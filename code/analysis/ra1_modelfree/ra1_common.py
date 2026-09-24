"""ra1_common.py -- paths, seeds, data loaders, bootstrap weights and parallel helpers for module ra1_modelfree.

Module ra1_modelfree ("The technology from designed variation, model-free") answers referee requests R1 c6, c7, c9;
R2 Major 8, 9(a)(b); R3 M6; R4 M1, M3, M5, M6. It reuses the loaders and estimators of modules m1_chinchilla,
m2_techpanel and m8_measurement (imported, never edited) and the shared library code/analysis/sl.py.

Conventions (paper/notes/model_spec.md): L = loss, n = ln N, d = ln D, c = ln C, C = 6ND, M = D/N,
a = beta/(alpha+beta), gamma = alpha beta/(alpha+beta), sigma* = 2/(2+alpha+beta), w = eps_N/eps_D.
Model-free sigma* (R1 comment 7):  1/sigma* - 1 = L_nn|_C / (2 |dL*/d ln C|)  at any compute-optimal point.
"""
from __future__ import annotations

import os
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
M1DIR = os.path.join(ANALYSIS, "m1_chinchilla")
M2DIR = os.path.join(ANALYSIS, "m2_techpanel")
M8DIR = os.path.join(ANALYSIS, "m8_measurement")
# module directory first (so that e.g. `figures`, `tables` resolve to this module's files); m1/m2 code appended
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
for _p in (ANALYSIS, M2DIR, M1DIR):
    if _p not in sys.path:
        sys.path.append(_p)

import sl  # noqa: E402  (shared library; not edited)

RAW = os.path.join(ROOT, "data", "raw")
_OUT = os.environ.get("RA1_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "ra1_modelfree")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

PY = os.path.join(ROOT, ".venv", "bin", "python")
N_PROC = 5                   # at most 5 CPU processes (two MLX queues are using the GPU and some CPU)
SEED = 20260924              # master seed; every draw derives from it via seed_of(key)
QUICK = bool(os.environ.get("RA1_QUICK"))


def seed_of(key: str) -> int:
    return SEED + zlib.crc32(key.encode()) % 1_000_000


_T0 = time.time()
_LOGF = None


def log(*a):
    global _LOGF
    msg = time.strftime("%H:%M:%S") + f" [{time.time() - _T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOGF is None:
        _LOGF = open(os.path.join(PROC, "run_log.txt"), "a")
    _LOGF.write(msg + "\n")
    _LOGF.flush()


# ============================================================================ bootstrap weights
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def rademacher(rng, size):
    return rng.choice(np.array([-1.0, 1.0]), size=size)


def webb(rng, size):
    """Webb (2023) six-point weights: +-sqrt(1/2), +-1, +-sqrt(3/2), each with probability 1/6."""
    return rng.choice(WEBB, size=size)


def fhh(rng, size, tau=0.5):
    """Feng, He and Hu (2011) wild-bootstrap weights for quantile (here median/LAD-type) regression:
    -2 tau with probability tau and 2(1 - tau) with probability 1 - tau; the bootstrap response is
    yhat + w |r~| (quantreg::boot.rq, bsmethod = 'wild'). At tau = 0.5 the weights are +-1."""
    return np.where(rng.random(size) < tau, -2.0 * tau, 2.0 * (1.0 - tau))


def boot_p_two_sided(draws, est, null):
    """Symmetric bootstrap p-value for H0: theta = null from the bootstrap distribution of theta* - theta_hat."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    B = len(d)
    return (1 + np.sum(np.abs(d - est) >= abs(est - null))) / (B + 1)


def rse(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float((np.percentile(v, 75) - np.percentile(v, 25)) / 1.349) if len(v) > 4 else np.nan


def pct(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) > 4 else np.nan


# ============================================================================ parallel map (spawn, <= 5 processes)
_G = {}


def _init(payload):
    import warnings
    warnings.filterwarnings("ignore")
    _G.clear()
    _G.update(payload)


def _work(args):
    import warnings
    warnings.filterwarnings("ignore")
    fn, item = args
    try:
        return fn(item, **_G)
    except Exception as e:  # recorded, never silently dropped
        return {"_error": repr(e)}


def pmap(fn, items, payload, procs=N_PROC, chunksize=4):
    """fn(item, **payload) for every item in spawn-started worker processes."""
    from multiprocessing import get_context
    ctx = get_context("spawn")
    with ctx.Pool(procs, initializer=_init, initargs=(payload,)) as pool:
        return pool.map(_work, [(fn, it) for it in items], chunksize=chunksize)


# ============================================================================ data: IsoFLOP designs
CHIN_BUDGETS = np.array([6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21])
MARIN = {"marin_202603__comma__llama_2": "Marin, Comma", "marin_202603__dclm__llama_2": "Marin, DCLM",
         "marin_202603__nemotron__llama_2": "Marin, Nemotron-CC"}


def load_chinchilla_all():
    """All 245 digitized Chinchilla runs with m1's budget reconstruction (offset-corrected nearest budget;
    IsoFLOP-profile membership |dev| <= 0.045 dex) and off-profile model-size trunks."""
    import common as m1c  # m1_chinchilla/common.py
    df = m1c.load_chinchilla(0)
    # off-profile trunks: model sizes (digitized N) grouped where consecutive log10 N differ by < 0.005 dex
    off = ~df["iso"].values
    ln = np.log10(df["N"].values)
    order = np.argsort(ln[off])
    lo = ln[off][order]
    g = np.r_[0, np.cumsum(np.diff(lo) > 0.005)]
    trunk = np.full(len(df), -1)
    idx_off = np.where(off)[0][order]
    trunk[idx_off] = g
    df["trunk"] = trunk
    return df


def isoflop_designs(chin_drop5=False):
    """Dict name -> DataFrame(x = ln N_conv, L, budget, c = ln C_budget, N, D) for every IsoFLOP design.
    Parameter conventions (so that C_budget = 6 N D holds exactly along each profile):
      Chinchilla  N total (Hoffmann), D = C/(6N) constructed from digitized C; runs at their nominal budget.
      Llama 3     N = C/(6D) (constructed by the digitization; FLOP-implied).
      Marin       N_eff = C_budget/(6D) (FLOP-weighted; Marin's budgets are 3 x forward FLOPs, differ from
                  6 N_config D by -7% to +35%); N_config kept as N_cfg for robustness.
      Porian      N = 'standard' count (non-embedding body + head), D = C/(6N) read along constant-LR tuned runs."""
    out = {}
    ch = load_chinchilla_all()
    iso = ch[ch["iso"]].copy()
    if chin_drop5:
        iso = iso[iso["rank_worst"] > 5]
    iso["C_b"] = iso["budget"]
    out["Chinchilla"] = iso
    raw = pd.read_csv(os.path.join(RAW, "isoflop_experiments", "isoflop_experiments.csv"))
    ll = raw[raw.experiment == "llama_3"].rename(columns={"params": "N", "tokens": "D", "loss": "L"}).copy()
    ll["C_b"] = ll["budget"]
    out["Llama 3"] = ll
    for exp, lab in MARIN.items():
        m = raw[raw.experiment == exp].rename(columns={"params": "N_cfg", "tokens": "D", "loss": "L"}).copy()
        m["C_b"] = m["budget"]
        m["N"] = m["C_b"] / (6.0 * m["D"])
        out[lab] = m
    por = pd.read_csv(os.path.join(PROC, "porian_isoflop_points.csv"))
    for ds, lab in (("rw", "Porian, RefinedWeb"), ("owt2", "Porian, OpenWebText2")):
        p = por[por.dataset == ds].rename(columns={"n": "N", "t": "D", "loss": "L"}).copy()
        p["C_b"] = p["C"]
        out[lab] = p
    for k, df in out.items():
        df = df.copy()
        df["x"] = np.log(df["N"].values)
        df["c"] = np.log(df["C_b"].values)
        df["b"] = pd.factorize(df["C_b"], sort=True)[0]
        out[k] = df.sort_values(["b", "x"]).reset_index(drop=True)
    return out


DESIGN_META = {
    "Chinchilla": dict(study="Hoffmann et al. (2022)", conv="total", loss="MassiveText val., nats/token (digitized)"),
    "Llama 3": dict(study="Grattafiori et al. (2024)", conv="FLOP-implied, C/(6D)", loss="val. loss, units unstated (digitized)"),
    "Marin, Comma": dict(study="Marin (2026)", conv="FLOP-implied, C/(6D)", loss="Paloma macro, nats/token"),
    "Marin, DCLM": dict(study="Marin (2026)", conv="FLOP-implied, C/(6D)", loss="Paloma macro, nats/token"),
    "Marin, Nemotron-CC": dict(study="Marin (2026)", conv="FLOP-implied, C/(6D)", loss="Paloma macro, nats/token"),
    "Porian, RefinedWeb": dict(study="Porian et al. (2024)", conv="non-emb. + head", loss="val. loss, nats/token"),
    "Porian, OpenWebText2": dict(study="Porian et al. (2024)", conv="non-emb. + head", loss="val. loss, nats/token"),
    "Farseer": dict(study="Li et al. (2025)", conv="non-embedding", loss="English bits/char"),
}


# ============================================================================ data: sweeps (m2 loaders)
def sweep_panels():
    """The seven sweep-corpus technologies of m2 (Table 4), harmonized by m2_data."""
    import m2_data as md
    ch = md.load_chinchilla()
    fa = md.load_farseer()
    ga = md.load_gadre()
    ol = md.load_olmo()
    db = md.load_datablations()
    P = {"Chinchilla": ch, "Farseer": fa}
    for c in ("C4", "RedPajama", "RefinedWeb"):
        P[f"Gadre, {c}"] = ga[ga.corpus == c].reset_index(drop=True)
    P["OLMo ladder"] = ol
    P["Muennighoff"] = db[db.epochs <= 1.0001].reset_index(drop=True)
    return P


def texnum(x, p=3):
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return ""
    if abs(x) >= 1e4:
        s = f"{abs(x):,.0f}".replace(",", "{,}")
    else:
        s = f"{abs(x):.{p}f}"
    return ("$-$" if x < 0 and float(s.replace("{,}", "")) != 0 else "") + s


def write_tex(name, caption, label, colspec, lines, notes, source="Authors' calculations.", size="\\footnotesize"):
    """AER-style table (booktabs, tabular*, AEA.cls tablenotes -- not threeparttable)."""
    L = ["\\begin{table}[tp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         "\\setlength{\\tabcolsep}{3pt}",
         f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}", "\\toprule"]
    L += lines
    L += ["\\bottomrule", "\\end{tabular*}", "", "\\begin{tablenotes}", notes, "\\end{tablenotes}", "",
          "\\begin{tablenotes}[Source]", source, "\\end{tablenotes}", "\\end{table}"]
    with open(os.path.join(TABLES, name + ".tex"), "w") as f:
        f.write("\n".join(L) + "\n")
