"""m2_data.py -- harmonized loaders for the m2_techpanel module.

Every loader returns a pandas DataFrame with (at least) the columns
    dataset, subset, run_id, cluster, N, D, L
where
    N  = parameter count in the convention documented in NCONV[dataset]
    D  = training tokens actually processed (incl. repetitions, if any)
    L  = held-out loss in the units documented in LUNITS[dataset]
    run_id  = one independent training run (checkpoints of the same run share run_id)
    cluster = bootstrap/cluster unit (see CLUSTER[dataset])
Raw files are read from data/raw/; nothing is downloaded here (see code/data/download_m2_techpanel.sh).
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")

# ----------------------------------------------------------------------------- documentation of conventions
NCONV = {
    "chinchilla": "total (Hoffmann et al. 2022; incl. embeddings)",
    "farseer": "non-embedding (excl. input and output embeddings; vocab 65,536 untied)",
    "gadre": "total incl. embeddings (open_lm `params`; D = 20 x multiplier x params)",
    "olmo_ladder": "excl. input embedding only (incl. LM head; OLMo MODEL_PARAMS; vocab 100,352)",
    "datablations": "Muennighoff PARAMS_MAP (as used by the authors; convention UNVERIFIED)",
    "datadecide": "excl. input embedding only (incl. LM head; DataDecide MODEL_TO_PARAMS; vocab 50,304)",
}
LUNITS = {
    "chinchilla": "nats/token, MassiveText val (SentencePiece 32k); digitized",
    "farseer": "bits/char, IntelliValSet-Raw English (web/paper/book average)",
    "gadre": "nats/token, C4 validation (GPT-NeoX tokenizer)",
    "olmo_ladder": "nats/token, C4-en validation (dolma2 tokenizer)",
    "datablations": "nats/token, C4 validation (GPT-2 tokenizer)",
    "datadecide": "nats/token, C4-en validation = ln(perplexity) (GPT-NeoX-20B tokenizer)",
}
DDEF = {
    "chinchilla": "C/(6N) (imputed by Epoch from digitized FLOP)",
    "farseer": "tokens processed (single epoch)",
    "gadre": "tokens processed (single epoch), 20 x M x N_total",
    "olmo_ladder": "tokens processed at final step (single epoch)",
    "datablations": "tokens processed (single-epoch subset: D = unique tokens)",
    "datadecide": "tokens at checkpoint = step x batch x 2048 (intermediate checkpoints; LR schedule not complete)",
}
CLUSTER = {
    "chinchilla": "run (pairs)",
    "farseer": "run (pairs)",
    "gadre": "run (pairs); pooled/neutrality models: (size, multiplier) cell",
    "olmo_ladder": "(size, multiplier) cell (reruns clustered with originals)",
    "datablations": "run (pairs)",
    "datadecide": "run (size x seed) within recipe; pooled/neutrality models: (size, seed) cell",
}


def _std(df, dataset, subset):
    df = df.copy()
    df["dataset"], df["subset"] = dataset, subset
    df["M"] = df["D"] / df["N"]
    df["C6"] = 6.0 * df["N"] * df["D"]
    cols = ["dataset", "subset", "run_id", "cluster", "N", "D", "L", "M", "C6"]
    return df[cols + [c for c in df.columns if c not in cols]].reset_index(drop=True)


# ----------------------------------------------------------------------------- Epoch Chinchilla (reference)
def load_chinchilla():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
    import sl
    df = sl.chinchilla_extraction(os.path.join(RAW, "epoch_chinchilla", "svg_extracted_data.csv"))
    df["run_id"] = np.arange(len(df))
    df["cluster"] = df["run_id"]
    return _std(df, "chinchilla", "all")


# ----------------------------------------------------------------------------- Farseer (Li et al. 2025b)
def load_farseer():
    """1222_full.csv, 404 runs. The column labelled 'D/N' takes 5 distinct values unrelated to D/N
    (3.52, 10.05, 28.2, 321.7, 454.9) -- it is NOT D/N; we recompute D/N from the D and N columns.
    'N' = non-embedding parameters (N_add_emb - N = 131,072 x h = two untied 65,536-row embeddings).
    'L' = IntelliValSet_Raw|en = English BPC (bits/char), the output Farseer fits. 'loss'/'smooth loss'
    are training losses in nats/token (bilingual training mix 2049_sc)."""
    raw = pd.read_csv(os.path.join(RAW, "farseer", "1222_full.csv"))
    df = pd.DataFrame({
        "N": raw["N"].astype(float), "D": raw["D"].astype(float), "L": raw["L"].astype(float),
        "N_emb": raw["N_add_emb"].astype(float), "L_train_smooth": raw["smooth loss"].astype(float),
        "lr": raw["lr"], "bs": raw["bs"], "C_farseer": raw["C"].astype(float),
        "DN_col_mislabeled": raw["D/N"],
    })
    df["run_id"] = np.arange(len(df))
    df["cluster"] = df["run_id"]
    return _std(df, "farseer", "all")


# ----------------------------------------------------------------------------- Gadre et al. (2024)
GADRE_CORPUS = {"c4_original": "C4", "rpj": "RedPajama", "rw_original": "RefinedWeb"}
GADRE_INDIST = {"c4_original": "paloma_c4_en", "rpj": "paloma_redpajama", "rw_original": "paloma_falcon-refinedweb"}


def load_gadre():
    """104 models (3 corpora). L = C4-validation loss (common output, 174M tokens); in-distribution
    Paloma loss in L_indist. se_lnL = token-level 95% CI half-width / (1.96 L) (delta method)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "gadre", "models", "*.json"))):
        d = json.load(open(f))
        h = d["hyperparameters"]
        res = {}
        for r in d["results"]:
            v = r["val_data"][0]
            key = "c4_val" if "c4_val/" in v else v.split("val_tok_mult/")[-1].split("/")[0]
            res[key] = r
        c4 = res["c4_val"]
        ind = res[GADRE_INDIST[d["dataset_name"]]]
        rows.append(dict(
            name=d["name"], corpus=GADRE_CORPUS[d["dataset_name"]], shape=d["name"].split("-", 1)[1].rsplit("-", 1)[0],
            mult=h["chinchilla_multiplier"], N=float(h["params"]), N_ne=float(h["params_no_embed"]),
            D=float(h["tokens"]), L=c4["loss"],
            se_lnL=(c4["loss_tokens_upper_95"] - c4["loss_tokens_lower_95"]) / (2 * 1.96 * c4["loss"]),
            L_indist=ind["loss"],
            se_lnL_indist=(ind["loss_tokens_upper_95"] - ind["loss_tokens_lower_95"]) / (2 * 1.96 * ind["loss"]),
        ))
    df = pd.DataFrame(rows)
    df["run_id"] = df["name"]
    df["cell"] = df["shape"] + "|" + df["mult"].astype(str)   # same (N, M) configuration across corpora
    df["cluster"] = df["run_id"]
    return _std(df, "gadre", "all")


# ----------------------------------------------------------------------------- OLMo ladder (Bhagia et al. 2024)
OLMO_PARAMS = {"190M": 190354176, "370M": 371262464, "760M": 758220288, "1B": 1279395840, "3B": 3169537280}


def load_olmo():
    """30 runs (5 sizes x {0.5,1,2,5,10}xC + 5 reruns at 1xC); final logged step. N from OLMo-ladder
    src/scaling/utils.py MODEL_PARAMS (= num_params(include_embedding=False): excludes the input
    embedding only). L = C4-en validation CE (nats/token). L_bpb = mean of the 11 validation-split
    downstream task BPB columns (bits/byte), an alternative tokenizer-free output."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "olmo_ladder", "*xC*.csv"))):
        base = os.path.basename(f)[:-4]
        size, mult = base.split("-")[0], base.split("-")[1]
        t = pd.read_csv(f)
        last = t.iloc[-1]
        bpb_cols = [c for c in t.columns if c.startswith("eval/downstream_bpb/") and "_val_" in c]
        rows.append(dict(run=base, size=size, mult=float(mult.replace("xC", "")), rerun=base.endswith("rerun"),
                         N=float(OLMO_PARAMS[size]), D=float(last["throughput/total_tokens"]),
                         L=float(last["eval/c4_en-validation/CrossEntropyLoss"]),
                         L_pile=float(last["eval/pile-validation/CrossEntropyLoss"]),
                         L_bpb=float(last[bpb_cols].mean()),
                         flops_logged=float(last["throughput/total_training_Gflops"]) * 1e9))
    df = pd.DataFrame(rows)
    df["run_id"] = df["run"]
    df["cluster"] = df["size"] + "|" + df["mult"].astype(str)
    return _std(df, "olmo_ladder", "all")


# ----------------------------------------------------------------------------- Muennighoff datablations
def load_datablations():
    """Final C4 validation losses hard-coded in huggingface/datablations plotstables/return_alloc.ipynb
    (NAMES_TO_VAL_LOSSES, 316 entries). We execute the notebook's own parsing cell to map model names
    (e.g. '2b855b9b' = N 2.8B, D 55B, U 9B unique) to (N, D, U); seed replicates, dedup/perplexity
    ablations and unfinished runs (loss 0) are excluded exactly as in the notebook (229 runs).
    ColPret's tokens_per_epoch / epochs columns are NOT used (they mis-parse names such as '1b25')."""
    nb = json.load(open(os.path.join(RAW, "datablations", "return_alloc.ipynb")))
    ns = {}
    exec("".join(nb["cells"][1]["source"]), ns)
    src = "".join(nb["cells"][2]["source"]).replace("import matplotlib.pyplot as plt", "")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src, ns)
    df = pd.DataFrame(dict(name=ns["names"], N=np.array(ns["model_params"], float), D=np.array(ns["tokens"], float),
                           U=np.array(ns["unique_tokens"], float), L=np.array(ns["losses"], float)))
    df["epochs"] = df["D"] / df["U"]
    df["run_id"] = df["name"]
    df["cluster"] = df["run_id"]
    return _std(df, "datablations", "all")


# ----------------------------------------------------------------------------- DataDecide (Magnusson et al. 2025)
DD_BATCH = {"4M": 32, "6M": 32, "8M": 32, "10M": 32, "14M": 32, "16M": 32, "20M": 64, "60M": 96, "90M": 160,
            "150M": 192, "300M": 320, "530M": 448, "750M": 576, "1B": 704}
DD_PARAMS = {"4M": 3744832, "6M": 6010464, "8M": 8538240, "10M": 9900432, "14M": 14380224, "16M": 16004560,
             "20M": 19101888, "60M": 57078144, "90M": 97946640, "150M": 151898880, "300M": 319980544,
             "530M": 530074944, "750M": 681297408, "1B": 1176832000}
DD_LAST = {"4M": 5725, "6M": 9182, "8M": 13039, "10M": 15117, "14M": 21953, "16M": 24432, "20M": 14584,
           "60M": 29042, "90M": 29901, "150M": 38157, "300M": 45787, "530M": 57786, "750M": 63589, "1B": 69369}
DD_SEQ = 2048
DD_EVALS = ["wikitext_103", "pile", "m2d2_s2orc", "ice", "dolma_wiki", "dolma_stack", "dolma_reddit", "dolma_pes2o",
            "dolma_common-crawl", "dolma_books", "c4_en"]


def load_datadecide(min_M=5.0):
    """DataDecide-ppl-results: 25 recipes x 14 sizes x 3 seeds (1B: default + large aux 2/3 full runs;
    small aux 2/3 of 530M/750M/1B are truncated runs). Batch sizes, non-embedding N and full-schedule
    last steps from github.com/allenai/DataDecide utils/constants.py. D = step x batch x 2048.
    Final checkpoints have D ~ 100 N (85-110), so only intermediate checkpoints break collinearity;
    their loss is measured before the cosine LR schedule has finished (see memo).
    Filters: step <= full-schedule last step (drops a few post-schedule steps), D/N >= min_M
    (drops warm-up transients; warm-up = N tokens), duplicate steps removed."""
    raw = pd.read_parquet(os.path.join(RAW, "datadecide", "ppl_results.parquet"))
    df = raw.rename(columns={"data": "recipe", "params": "size"})
    df["N"] = df["size"].map(DD_PARAMS).astype(float)
    df["D"] = df["step"] * df["size"].map(DD_BATCH) * DD_SEQ
    df["L"] = np.log(df["eval/c4_en-validation/Perplexity"])
    df["L_avg11"] = np.mean([np.log(df[f"eval/{e}-validation/Perplexity"]) for e in DD_EVALS], axis=0)
    df["frac"] = df["step"] / df["size"].map(DD_LAST)
    df = df[(df["step"] <= df["size"].map(DD_LAST)) & (df["D"] / df["N"] >= min_M)]
    df = df.drop_duplicates(["recipe", "size", "seed", "step"])
    df["run_id"] = df["recipe"] + "|" + df["size"] + "|" + df["seed"]
    df["cell"] = df["size"] + "|" + df["seed"]
    df["cluster"] = df["run_id"]
    keep = ["recipe", "size", "seed", "step", "frac", "cell", "run_id", "cluster", "N", "D", "L", "L_avg11"]
    return _std(df[keep], "datadecide", "all")


def load_all():
    return dict(chinchilla=load_chinchilla(), farseer=load_farseer(), gadre=load_gadre(), olmo_ladder=load_olmo(),
                datablations=load_datablations(), datadecide=load_datadecide())


def offpath_sd(N, D):
    """sd(ln(D/N) | ln C): residual s.d. of an OLS of ln(D/N) on (1, ln 6ND)."""
    r = np.log(np.asarray(D) / np.asarray(N))
    lc = np.log(6.0 * np.asarray(N) * np.asarray(D))
    X = np.c_[np.ones_like(lc), lc]
    b = np.linalg.lstsq(X, r, rcond=None)[0]
    return float(np.std(r - X @ b, ddof=2))
