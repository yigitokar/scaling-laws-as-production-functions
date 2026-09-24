"""m9_common.py -- paths, design, loaders, units and small helpers for module m9_sweeps (our two-corpus experiment).

Binding plan: paper/notes/m9_preanalysis_plan.md (committed 2026-09-24 03:12, before any estimation).
Design source of truth: code/sweep/run_grid.py (imported; it defines SIZES, D_GRID, D_MAX and the job lists) and the
architecture formulas of code/sweep/gpt_mlx.py (re-implemented below WITHOUT importing MLX; run.py asserts that they
reproduce the N and FLOP fields of every endpoint in results.jsonl).

Parameter conventions (plan Section 2):
  P  non-embedding N (field N_nonemb); compute = actual training FLOPs incl. unembedding matmul and attention (field C).
     Because C = fpt(N) D is not 6 N D, the compute-optimal path under P satisfies w = eta(N) = d ln fpt / d ln N
     (not w = 1). "P6" is the same N with the textbook cost 6 N D (path w = 1), reported for comparability.
  T  total N incl. the tied embedding (field N_total); C = 6 N_total D (within 3-5% of actual FLOPs here).
Output: loss in bits per byte = nats/token / (ln 2 x bytes/token of the SCORED validation tokens).
"""
from __future__ import annotations

import json
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
SWEEP_CODE = os.path.join(ROOT, "code", "sweep")
RA1DIR = os.path.join(ANALYSIS, "ra1_modelfree")
M2DIR = os.path.join(ANALYSIS, "m2_techpanel")
M6DIR = os.path.join(ANALYSIS, "m6_montecarlo")
# this module's directory first; reused modules appended (never inserted ahead, so their files cannot shadow ours)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
for _p in (ANALYSIS, M2DIR, RA1DIR, M6DIR, SWEEP_CODE):
    if _p not in sys.path:
        sys.path.append(_p)

import sl  # noqa: E402  (shared library; read-only)

SWEEP = os.path.join(ROOT, "data", "processed", "sweep")
_OUT = os.environ.get("M9_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "m9_sweeps")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

N_PROC = int(os.environ.get("M9_PROCS", 4))  # at most 4 CPU processes (two MLX training queues share the machine)
SEED = 20260924
QUICK = bool(os.environ.get("M9_QUICK"))
V, SEQ = 8192, 256          # vocabulary, context (train_sweep.py defaults)
CORPORA = ("edu", "web")
VALSETS = ("edu", "web", "wiki")
CONVS = ("P", "T")


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


# ============================================================================ architecture (gpt_mlx.py formulas)
def arch(d, L, vocab=V, seq=SEQ, mlp=4):
    """N_nonemb, N_emb, N_total and training FLOPs per token for width d, depth L (gpt_mlx.count_params and
    flops_per_token, copied so that MLX is never imported)."""
    d = np.asarray(d, float)
    L = np.asarray(L, float)
    per_block = 3 * d * d + d * d + 2 * mlp * d * d + 2 * d
    ne = L * per_block + d
    emb = vocab * d
    fpt = 6 * ne + 6 * vocab * d + 6 * L * seq * d
    return dict(N_nonemb=ne, N_emb=emb, N_total=ne + emb, fpt=fpt)


class LadderCost:
    """The design's shape ladder (depth = d/64) as a smooth curve in d: x(d) = ln N_nonemb, G(d) = ln fpt.
    eta(x) = dG/dx is the cost elasticity of non-embedding parameters (convention P); eta'(x) its derivative."""

    def __init__(self, dmin=64.0, dmax=1024.0, n=4000):
        self.d = np.exp(np.linspace(np.log(dmin), np.log(dmax), n))
        a = arch(self.d, self.d / 64.0)
        self.x = np.log(a["N_nonemb"])
        self.G = np.log(a["fpt"])
        self.eta_grid = np.gradient(self.G, self.x)
        self.deta_grid = np.gradient(self.eta_grid, self.x)

    def G_of_x(self, x):
        return np.interp(x, self.x, self.G)

    def eta(self, x):
        return np.interp(x, self.x, self.eta_grid)

    def deta(self, x):
        return np.interp(x, self.x, self.deta_grid)


LADDER = LadderCost()


def design_cells():
    """All planned endpoints (from run_grid.jobs), as a DataFrame with regime, tag, d, L, lr, seed, D_target."""
    import run_grid as rg  # code/sweep/run_grid.py (no MLX import)
    rows = []
    kinds = ["main", "lrcal", "lrcorner", "seedcorner", "seeds", "hiM"]
    for regime in CORPORA:
        for kind in kinds + (["lrsweep"] if regime == "edu" else []):
            if kind == "lrcal" and regime == "edu":
                continue
            for j in rg.jobs(kind, regime):
                for D in str(j["D"]).split(","):
                    rows.append(dict(regime=j["regime"], tag=j["tag"], d=j["d"], L=j["L"], lr=j["lr"],
                                     seed=j.get("seed", 0), D_target=int(float(D)), ext=j.get("ext", 0)))
    df = pd.DataFrame(rows)
    a = arch(df.d.values, df.L.values)
    for k, v in a.items():
        df[k] = v
    return df


# ============================================================================ bits per byte
def token_bytes():
    """Raw UTF-8 byte length of every token of the shared byte-level BPE (<eot> = 0 bytes). Reproduces
    meta.json's full-file bytes/token exactly (checked in run.py)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(SWEEP, "tokenizer.json"))
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    inv = {chr(c): b for b, c in zip(bs, cs)}
    blen = np.zeros(tok.get_vocab_size(), int)
    for s, i in tok.get_vocab().items():
        blen[i] = len(bytes(inv[ch] for ch in s))
    blen[tok.token_to_id("<eot>")] = 0
    return blen


def bytes_per_token_table(val_tokens=1_048_576):
    """bytes/token of (i) the full validation file (= meta.json) and (ii) the tokens actually scored by
    train_sweep.evaluate (targets of the first min(4096, n) windows of 257 tokens)."""
    cache = os.path.join(PROC, "bytes_per_token.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    blen = token_bytes()
    meta = json.load(open(os.path.join(SWEEP, "meta.json")))
    out = {}
    for r in VALSETS:
        arr = np.memmap(os.path.join(SWEEP, f"{r}_val.bin"), dtype=np.uint16, mode="r")
        nseq = min(val_tokens // SEQ, (len(arr) - 1) // SEQ)
        tgt = np.concatenate([arr[i * SEQ + 1: i * SEQ + SEQ + 1] for i in range(nseq)])
        out[r] = dict(full=float(blen[arr].sum() / len(arr)), meta=float(meta[r]["val"]["bytes_per_token"]),
                      scored=float(blen[tgt].sum() / len(tgt)), scored_tokens=int(len(tgt)))
    json.dump(out, open(cache, "w"), indent=2)
    return out


def to_bpb(nats, valset, which="scored"):
    return np.asarray(nats, float) / (np.log(2.0) * bytes_per_token_table()[valset][which])


# ============================================================================ data
def load_results():
    """results.jsonl -> DataFrame with bpb losses, conventions and cluster ids. Duplicate endpoints (identical
    regime, tag, d, L, lr, seed, D_target) keep the first record (interrupted trunks re-run only missing budgets)."""
    path = os.environ.get("M9_RESULTS", os.path.join(SWEEP, "results.jsonl"))   # override only for pipeline tests
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    key = ["regime", "tag", "d", "L", "lr", "seed", "D_target"]
    df["n_records"] = df.groupby(key)["tokens"].transform("size")
    df = df.drop_duplicates(key, keep="first").reset_index(drop=True)
    for r in VALSETS:
        col = f"loss_{r}"
        if col not in df:
            df[col] = np.nan
        df[f"bpb_{r}"] = to_bpb(df[col].values, r)
        df[f"bpb_{r}_meta"] = to_bpb(df[col].values, r, "meta")
    df["D"] = df["tokens"].astype(float)
    df["N_P"] = df["N_nonemb"].astype(float)
    df["N_T"] = df["N_total"].astype(float)
    df["C_P"] = df["C"].astype(float)
    df["C_T"] = 6.0 * df["N_T"] * df["D"]
    df["M_P"] = df["D"] / df["N_P"]
    df["M_T"] = df["D"] / df["N_T"]
    df["ext"] = df.get("ext", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["arch"] = df["d"].astype(str) + "x" + df["L"].astype(str)
    df["on_ladder"] = df["L"] * 64 == df["d"]
    # the rule LR at each width (for lrcorner multipliers)
    import run_grid as rg
    df["lr_rule"] = [round(rg.lr_rule(d), 6) for d in df["d"]]
    df["lr_mult"] = df["lr"] / df["lr_rule"]
    return df


def own_output(df):
    """Loss on the training corpus's own validation set (bpb)."""
    return np.where(df["regime"].values == "edu", df["bpb_edu"].values, df["bpb_web"].values)


def main_sample(df, regime, variant="main", with_hiM=False):
    """Plan Section 2 samples: 'main' (tag main), 'no25' (drop D = 25M), 'floor4' (drop d in {128, 192}).
    with_hiM adds the hiM endpoints that are not duplicates of main-grid cells (Q3 full support)."""
    s = df[(df.regime == regime) & (df.tag == "main") & (df.seed == 0)].copy()
    if with_hiM:
        h = df[(df.regime == regime) & (df.tag == "hiM") & (df.seed == 0)].copy()
        keys = set(zip(s.d, s.L, s.lr.round(6), s.D_target))
        h["dup_of_main"] = [(a, b, round(c, 6), e) in keys for a, b, c, e in zip(h.d, h.L, h.lr, h.D_target)]
        s = pd.concat([s, h[~h.dup_of_main]], ignore_index=True)
    if variant == "no25":
        s = s[s.D_target > 25_000_000]
    elif variant == "floor4":
        s = s[s.L >= 4]
    return s.reset_index(drop=True)


SAMPLES = ("main", "no25", "floor4")
SAMPLE_LABEL = {"main": "Main grid", "no25": "Drop D = 25M", "floor4": "Four-layer floor"}


def clusters_of(df):
    """Trunk clusters: one per architecture (d, L). hiM (256,4) shares width 256's cluster (its trunk reproduces the
    main-grid trunk up to 1.7B tokens: same seed, LR and data order); hiM (128,4) is its own cluster."""
    return df["arch"].values


# ============================================================================ bootstrap weights and summaries
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def webb(rng, size):
    return rng.choice(WEBB, size=size)


def cluster_weights(rng, clusters, universe=None):
    """Webb weight per cluster (universe = list of all cluster labels, so that the same label gets the same weight
    across data sets in one draw, e.g. width j in both corpora)."""
    u = np.unique(clusters) if universe is None else np.asarray(universe)
    w = dict(zip(u, webb(rng, len(u))))
    return np.array([w[c] for c in clusters])


def basic_ci(est, draws, pop, level=0.95):
    """Basic bootstrap interval est - (q(draws) - pop); pop = statistic on the noise-free bootstrap population."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if len(d) < 20 or not np.isfinite(est):
        return np.nan, np.nan
    lo, hi = np.percentile(d, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(est - (hi - pop)), float(est - (lo - pop))


def boot_p(draws, est, pop, null=0.0):
    """Symmetric bootstrap p-value for H0: theta = null, from the distribution of theta* - pop."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return np.nan
    return float((1 + np.sum(np.abs(d - pop) >= abs(est - null))) / (len(d) + 1))


def sd_(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if len(v) > 2 else np.nan


# ============================================================================ parallel map (spawn, <= 4 processes)
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
        import traceback
        return {"_error": repr(e), "_tb": traceback.format_exc()}


def pmap(fn, items, payload, procs=N_PROC, chunksize=1):
    from multiprocessing import get_context
    if procs <= 1:
        _init(payload)
        return [_work((fn, it)) for it in items]
    ctx = get_context("spawn")
    with ctx.Pool(procs, initializer=_init, initargs=(payload,)) as pool:
        return pool.map(_work, [(fn, it) for it in items], chunksize=chunksize)


# ============================================================================ LaTeX
def texnum(x, p=3):
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1e4:
        s = f"{abs(x):,.0f}".replace(",", "{,}")
    else:
        s = f"{abs(x):.{p}f}"
    return ("$-$" if x < 0 and float(s.replace("{,}", "")) != 0 else "") + s


def texci(lo, hi, p=3):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    return f"[{texnum(lo, p)}, {texnum(hi, p)}]"


def write_tex(name, caption, label, colspec, lines, notes, source="Authors' calculations.", size="\\footnotesize",
              colsep="3pt"):
    """AER-style table (booktabs, tabular*, AEA.cls tablenotes -- not threeparttable)."""
    L = ["\\begin{table}[tp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         f"\\setlength{{\\tabcolsep}}{{{colsep}}}",
         f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}", "\\toprule"]
    L += lines
    L += ["\\bottomrule", "\\end{tabular*}", "", "\\begin{tablenotes}", notes, "\\end{tablenotes}", "",
          "\\begin{tablenotes}[Source]", source, "\\end{tablenotes}", "\\end{table}"]
    with open(os.path.join(TABLES, name + ".tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_csv(df, name):
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=False)
