"""rb4_arch.py -- Hoffmann et al. (2022) architectures (Table A9), parameter counts and FLOPs per token (Appendix F).

Sources (arXiv 2203.15556, TeX source in data/raw/rb4_chinflop; the tables below are transcribed from it and
cross-checked against the TeX at every run when the source is present):
  Table A9 ("All models"): Parameters (million), d_model, ffw_size, kv_size, n_heads, n_layers -- 50 models.
  Table A4 ("FLOP comparison"): ratio of their FLOPs per sequence to 6ND for six models.
  Appendix F: forward-pass FLOPs per sequence (seq_len = 2,048; vocab = 32,000; factor 2 per multiply-accumulate):
    embeddings              2 S V d
    attention (per layer)   QKV 2*3 S d (kv h); logits 2 S^2 (kv h); softmax 3 h S^2; reduction 2 S^2 (kv h);
                            final linear 2 S (kv h) d
    dense (per layer)       2 S (d f + d f)
    final logits            2 S d V
    total forward = embeddings + L (attention + dense) + logits; backward = 2 x forward.

Parameter count. Table A9's counts are reproduced (mean |error| 0.8M, max 0.8 percent) by
    T = L (4 d kvh + 2 d f)  +  L d kvh  +  V d,
i.e. attention and MLP weights, one relative-position projection W_R per layer (Transformer-XL relative encodings,
as in Gopher) and ONE V x d vocabulary matrix (tied input/output, or input only). Dropping W_R misses by 7 percent;
counting two vocabulary matrices misses by up to 36 percent ([review]: was "31", the figure without W_R). Remaining differences are layer norms, relative
biases and rounding to 1M.

FLOP accountings (per token, forward + backward = 3 x forward):
  T4  Appendix F WITHOUT the embedding and final-logit terms. This is the count that reproduces every ratio of
      Table A4 to the two printed decimals (F_T4/(6 T)); the formula as printed does not (misses by up to 0.55).
      Primary: it is the count Hoffmann et al. evidently implemented, and (rb4_estim.coords_test) the one under which
      the digitized FLOP coordinates of their Figure 4 are consistent with the nine nominal budgets.
  A   Appendix F as printed (embedding "matmul" + final logits).
  X   executed FLOPs: A without the embedding term (a lookup costs no multiply-adds) -- the unembedding matmul is
      real. This is the concept of our own experiment's count (code/sweep/gpt_mlx.flops_per_token), up to the
      attention coefficient.
  6T  6 x total parameters (the Kaplan approximation, total count).
  6P  6 x non-embedding parameters P = T - V d (W_R kept: it is a per-layer weight matrix).
N_F = C/(6D) = F/6 for every accounting: with C = F(N) D the token count cancels, so N_F does not depend on how D
is recovered (see the memo).
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

import rb4common as cm

V, SEQ = 32000, 2048

# (Parameters (million), d_model, ffw_size, kv_size, n_heads, n_layers) -- Hoffmann et al. (2022), Table A9
TABLE_A9 = [
    (44, 512, 2048, 64, 8, 8), (57, 576, 2304, 64, 9, 9), (74, 640, 2560, 64, 10, 10), (90, 640, 2560, 64, 10, 13),
    (106, 640, 2560, 64, 10, 16), (117, 768, 3072, 64, 12, 12), (140, 768, 3072, 64, 12, 15),
    (163, 768, 3072, 64, 12, 18), (175, 896, 3584, 64, 14, 14), (196, 896, 3584, 64, 14, 16),
    (217, 896, 3584, 64, 14, 18), (251, 1024, 4096, 64, 16, 16), (278, 1024, 4096, 64, 16, 18),
    (306, 1024, 4096, 64, 16, 20), (425, 1280, 5120, 128, 10, 18), (489, 1280, 5120, 128, 10, 21),
    (509, 1408, 5632, 128, 11, 18), (552, 1280, 5120, 128, 10, 24), (587, 1408, 5632, 128, 11, 21),
    (632, 1536, 6144, 128, 12, 19), (664, 1408, 5632, 128, 11, 24), (724, 1536, 6144, 128, 12, 22),
    (816, 1536, 6144, 128, 12, 25), (893, 1792, 7168, 128, 14, 20), (1018, 1792, 7168, 128, 14, 23),
    (1143, 1792, 7168, 128, 14, 26), (1266, 2048, 8192, 128, 16, 22), (1424, 2176, 8704, 128, 17, 22),
    (1429, 2048, 8192, 128, 16, 25), (1593, 2048, 8192, 128, 16, 28), (1609, 2176, 8704, 128, 17, 25),
    (1731, 2304, 9216, 128, 18, 24), (1794, 2176, 8704, 128, 17, 28), (2007, 2304, 9216, 128, 18, 28),
    (2283, 2304, 9216, 128, 18, 32), (2298, 2560, 10240, 128, 20, 26), (2639, 2560, 10240, 128, 20, 30),
    (2980, 2560, 10240, 128, 20, 34), (3530, 2688, 10752, 128, 22, 36), (3802, 2816, 11264, 128, 22, 36),
    (4084, 2944, 11776, 128, 22, 36), (4516, 3072, 12288, 128, 24, 36), (6796, 3584, 14336, 128, 28, 40),
    (9293, 4096, 16384, 128, 32, 42), (11452, 4352, 17408, 128, 32, 47), (12295, 4608, 18432, 128, 36, 44),
    (12569, 4608, 18432, 128, 32, 47), (13735, 4864, 19456, 128, 32, 47), (14940, 4992, 19968, 128, 32, 49),
    (16183, 5120, 20480, 128, 40, 47),
]
# (label, num_layers, d_model, ffw_size, num_heads, k/q size, reported ratio Ours/6ND) -- Table A4
TABLE_A4 = [("73M", 10, 640, 2560, 10, 64, 1.03), ("305M", 20, 1024, 4096, 16, 64, 1.10),
            ("552M", 24, 1280, 5120, 10, 128, 1.08), ("1.1B", 26, 1792, 7168, 14, 128, 1.04),
            ("1.6B", 28, 2048, 8192, 16, 128, 1.03), ("6.8B", 40, 3584, 14336, 28, 128, 0.99)]
# other architectures used as checks (Rae et al. 2021, Table 1; Hoffmann et al. 2022, Table 4)
OTHER = {"Gopher 280B": dict(d=16384, ffw=65536, kv=128, h=128, L=80, D=300e9, N_rep=280e9, C_rep=5.76e23),
         "Chinchilla 70B": dict(d=8192, ffw=32768, kv=128, h=64, L=80, D=1.4e12, N_rep=70e9, C_rep=np.nan)}

ACCOUNTINGS = ("T4", "A", "X", "6T", "6P")
ACC_LABEL = {"T4": "N_F, Hoffmann's implemented count (reproduces Table A4)",
             "A": "N_F, Appendix F as printed (embedding matmul + logits)",
             "X": "N_F, executed FLOPs (unembedding matmul; embedding lookup free)",
             "6T": "T, total parameters (C = 6TD)", "6P": "P, non-embedding parameters (C = 6PD)"}


# ============================================================================ source check / fetch
def fetch_source():
    """Download the arXiv source (public) if it is not present, and verify its checksum."""
    import hashlib
    import tarfile
    import urllib.request
    os.makedirs(cm.RAW_RB4, exist_ok=True)
    if not os.path.exists(cm.ARXIV_TGZ):
        req = urllib.request.Request(cm.ARXIV_URL, headers={"User-Agent": "Mozilla/5.0 (research replication)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(cm.ARXIV_TGZ, "wb") as f:
            f.write(r.read())
    h = hashlib.sha256(open(cm.ARXIV_TGZ, "rb").read()).hexdigest()
    if not os.path.exists(cm.ARXIV_TEX):
        with tarfile.open(cm.ARXIV_TGZ) as t:
            t.extractall(os.path.join(cm.RAW_RB4, "src"))
    return h


def parse_tex():
    """Table A9 and Table A4 parsed from the TeX source (for the transcription check)."""
    src = open(cm.ARXIV_TEX).read()
    i0 = src.index("Parameters (million) &  d\\_model")
    i1 = src.index("\\bottomrule", i0)
    a9 = []
    for line in src[i0:i1].splitlines()[2:]:
        line = line.replace("\\cr", "").replace("\\\\", "").strip()
        if not line or line.startswith("\\"):
            continue
        p = [q.strip() for q in line.split("&")]
        a9.append((int(p[0].replace(",", "")), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])))
    j0 = src.index("FLOP Ratio (Ours/$6ND$)")
    j1 = src.index("\\bottomrule", j0)
    a4 = []
    for line in src[j0:j1].splitlines()[2:]:
        line = line.replace("\\\\", "").strip()
        if not line:
            continue
        p = [q.strip() for q in line.split("&")]
        a4.append((p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])))
    return a9, a4


# ============================================================================ counts
def params(d, ffw, kv, h, L):
    kvh = kv * h
    blocks = L * (4 * d * kvh + 2 * d * ffw)
    wr = L * d * kvh
    emb = V * d
    return dict(blocks=blocks, W_R=wr, emb=emb, T=blocks + wr + emb, P=blocks + wr, P_noWR=blocks)


def fwd_terms(d, ffw, kv, h, L, seq=SEQ):
    """Appendix F forward FLOPs per TOKEN (per-sequence terms divided by seq_len), by component."""
    kvh = kv * h
    return dict(embeddings=2 * V * d,
                attn_linear=L * (2 * 3 * d * kvh + 2 * kvh * d),
                attn_quadratic=L * (2 * seq * kvh + 2 * seq * kvh),
                softmax=L * 3 * h * seq,
                dense=L * 2 * (2 * d * ffw),
                logits=2 * d * V)


def flops_per_token(d, ffw, kv, h, L, kind, seq=SEQ):
    """Training FLOPs per token (forward + backward) under accounting `kind` (module docstring)."""
    t = fwd_terms(d, ffw, kv, h, L, seq)
    core = t["attn_linear"] + t["attn_quadratic"] + t["softmax"] + t["dense"]
    p = params(d, ffw, kv, h, L)
    if kind == "T4":
        return 3.0 * core
    if kind == "A":
        return 3.0 * (core + t["embeddings"] + t["logits"])
    if kind == "X":
        return 3.0 * (core + t["logits"])
    if kind == "6T":
        return 6.0 * p["T"]
    if kind == "6P":
        return 6.0 * p["P"]
    raise ValueError(kind)


def arch_table():
    """One row per Table A9 model: counts, per-token FLOPs under each accounting, N_F = F/6 and r = F/(6T)."""
    rows = []
    for i, (nm, d, f, kv, h, L) in enumerate(TABLE_A9):
        p = params(d, f, kv, h, L)
        t = fwd_terms(d, f, kv, h, L)
        r = dict(row=i, N_rep_M=nm, d_model=d, ffw_size=f, kv_size=kv, n_heads=h, n_layers=L, kvh=kv * h,
                 T=p["T"], P=p["P"], P_noWR=p["P_noWR"], W_R=p["W_R"], emb=p["emb"],
                 T_err_rel=p["T"] / (nm * 1e6) - 1.0, emb_share=p["emb"] / p["T"], WR_share=p["W_R"] / p["T"])
        for k in ACCOUNTINGS:
            F = flops_per_token(d, f, kv, h, L, k)
            r[f"F_{k}"] = F
            r[f"NF_{k}"] = F / 6.0
            r[f"r_{k}"] = F / (6.0 * p["T"])
        fwd_all = sum(t.values())
        for k, v in t.items():
            r[f"share_fwd_{k}"] = v / fwd_all
        rows.append(r)
    return pd.DataFrame(rows)


def table_a4_check():
    """Reproduce Table A4: ratio of FLOPs to 6ND under each accounting, with N the architecture total (and the
    table's rounded label as a check)."""
    rows = []
    for lab, L, d, f, h, kv, rep in TABLE_A4:
        p = params(d, f, kv, h, L)
        n_lab = float(lab[:-1]) * (1e6 if lab.endswith("M") else 1e9)
        r = dict(model=lab, n_layers=L, d_model=d, ffw_size=f, n_heads=h, kq_size=kv, ratio_reported=rep,
                 T_arch=p["T"], T_label=n_lab)
        for k in ("T4", "A", "X"):
            F = flops_per_token(d, f, kv, h, L, k)
            r[f"ratio_{k}"] = F / (6 * p["T"])
            r[f"ratio_{k}_labelN"] = F / (6 * n_lab)
        rows.append(r)
    out = pd.DataFrame(rows)
    summ = []
    for k in ("T4", "A", "X"):
        for col, lab in ((f"ratio_{k}", "architecture T"), (f"ratio_{k}_labelN", "rounded label N")):
            e = out[col].round(2) - out.ratio_reported
            summ.append(dict(accounting=k, N_in_6ND=lab, exact_2dp=int((e.abs() < 1e-9).sum()), n=len(out),
                             max_abs_err=float(np.abs(out[col] - out.ratio_reported).max()),
                             rmse=float(np.sqrt(np.mean((out[col] - out.ratio_reported) ** 2)))))
    return out, pd.DataFrame(summ)


def other_checks():
    """Gopher and Chinchilla totals under each accounting (Hoffmann et al. quote 6.3e23 for Gopher's training
    compute under their count, against Rae et al.'s 5.76e23)."""
    rows = []
    for nm, a in OTHER.items():
        p = params(a["d"], a["ffw"], a["kv"], a["h"], a["L"])
        r = dict(model=nm, T_arch=p["T"], N_reported=a["N_rep"], D=a["D"], C_6TD=6 * p["T"] * a["D"])
        for k in ("T4", "A", "X"):
            r[f"C_{k}"] = flops_per_token(a["d"], a["ffw"], a["kv"], a["h"], a["L"], k) * a["D"]
        r["C_quoted_by_hoffmann"] = 6.3e23 if nm.startswith("Gopher") else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def transcription_check():
    """Compare the transcribed tables with the TeX source (if present). Returns a dict."""
    out = dict(source_present=os.path.exists(cm.ARXIV_TEX))
    if not out["source_present"]:
        return out
    a9, a4 = parse_tex()
    out["a9_rows"] = len(a9)
    out["a9_equal"] = [tuple(r) for r in a9] == [tuple(r) for r in TABLE_A9]
    out["a4_equal"] = [tuple(r) for r in a4] == [tuple(r) for r in TABLE_A4]
    return out
