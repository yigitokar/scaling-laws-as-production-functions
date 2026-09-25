"""base.py -- rebuild module ra2_wedge's objects (clean sample, technologies with draws, wedges) by import, without
touching ra2's output files, and apply this module's audit of documented token counts.

ra2_wedge is imported read-only. Two ra2 functions write into data/processed/ra2_wedge (sample.build writes
audit_flags.csv; ra2common.log appends to ra2's run log): both are redirected to this module's folder.

Token-count audit (this module; data/processed/rb2_decisions/d_audit.csv): the StableLM-Alpha base models (3B, 7B) were
trained on 800B tokens of a 1.5T-token dataset (Stability-AI/StableLM README, 'Training Tokens' column), and their
parameter counts are 3,638,525,952 and 7,869,358,080 (same table); m3/ra2 used the dataset size (1.5T) and nominal
sizes. Corrected here in the primary analysis; the uncorrected values are kept as a sensitivity row.
Round 3 (fix list W24(f)): MPT-30B's D is 1.05T (1T at 2k context + 50B at 8k; MosaicML's release post). Conventions that
change no count (OLMo 2 mid-training accounting, Yi's 3T, Qwen2.5's unaudited per-member budgets) are logged in D_NOTES.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

import rb2common as C

import sample as SM  # noqa: E402  (ra2_wedge)
import modelfree as MF  # noqa: E402
import techs as TT  # noqa: E402
import wedges as WG  # noqa: E402
import analysis_ra2 as AN  # noqa: E402

# ------------------------------------------------------------------ side-effect redirection (ra2 files untouched)
_REPLAY = os.path.join(C.PROC, "ra2_replay")
os.makedirs(_REPLAY, exist_ok=True)
SM.PROC = _REPLAY
for _m in (SM, TT, AN):
    _m.log = C.log
C.R2C.log = C.log
TT.N_PROC = C.N_PROC

D_AUDIT = [
    # model, field, ra2 value, corrected value, source key (data/raw/rb2_decisions), quote
    ("stablelm-base-alpha-3b", "D", 1.5e12, 8.0e11, "stablelm_github",
     "| 3B | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-3b/) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/) | 800B | 3,638,525,952 |"),
    ("stablelm-base-alpha-3b", "N", 3.0e9, 3638525952.0, "stablelm_github",
     "| 3B | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-3b/) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/) | 800B | 3,638,525,952 |"),
    ("stablelm-base-alpha-7b", "D", 1.5e12, 8.0e11, "stablelm_github",
     "| 7B | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-7b) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b) | 800B | 7,869,358,080 |"),
    ("stablelm-base-alpha-7b", "N", 7.0e9, 7869358080.0, "stablelm_github",
     "| 7B | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-7b) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b) | 800B | 7,869,358,080 |"),
    # round 3 (fix list W24(f); citations_4 #11): MPT-30B trained 1T tokens at 2k context and 50B more at 8k
    ("mpt-30b", "D", 1.0e12, 1.05e12, "mpt30b_blog",
     "we first pre-trained on 1T tokens using sequences that were 2k tokens long, and continued training for an additional 50B tokens using sequences that were 8k tokens long."),
]

# Round 3 (W24; X8(m)): conventions checked against the saved sources that do not change a count. Logged in d_audit.csv
# with corrected_value = ra2_value. (model, field, value, source key, quote, note)
D_NOTES = [
    ("OLMo-2-1124-13B", "D", 5.15e12, "olmo2_report",
     "OLMo 2 13B is trained on 5.6 trillion tokens (5 trillion for pretraining stage)",
     "convention: pretraining tokens plus the mean mid-training budget of the averaged runs (5T + (3 x 100B + 300B)/4); "
     "the report's total, 5.6T, adds every averaged run"),
    ("OLMo-2-0325-32B", "D", 6.15e12, "olmo2_report",
     "OLMo 2 32B is trained on 6.6 trillion tokens (6.06 trillion for pretraining stage)",
     "convention as for the 13B: 6T (release post: 'up to 6T tokens', 1.5 epochs) + 150B; the report's total is 6.6T and "
     "it gives 6.06T for the pretraining stage"),
    ("OLMo-2-1124-7B", "D", 4.05e12, "olmo2_report",
     "OLMo 2 7B is trained on 4 .05 trillion tokens (3.90 trillion for pretraining stage)",
     "convention: 4T stage 1 (release post: one epoch, up to 4T) + 50B (three averaged 50B runs); the report's 4.05T total "
     "adds the three runs to a 3.90T pretraining stage, which would give 3.95T under the convention"),
    ("OLMo-2-0425-1B", "D", 4.05e12, "olmo2_report",
     "We pretrain OLMo 2 1B to 4 trillion tokens on OLMo 2 Mix 1124 and perform a single 50B token anneal on Dolmino Mix 1124.",
     "4T + one 50B anneal (no averaging)"),
    ("Yi-6B", "D", 3.0e12, "yi_report", "we overtrain the model on more tokens (3T) than the compute optimal (around 1T).",
     "kept at 3T (the report's over-training statement and the model card); the report also gives 3.1T"),
    ("Yi-34B", "D", 3.0e12, "yi_report", "Our model is trained on 3.1T tokens",
     "kept at 3T; the 3.1T of this sentence would raise M by 3 percent"),
    ("Qwen2.5-0.5B", "D", 1.8e13, "qwen25_blog",
     "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens.",
     "flag: per-member budgets not stated ('up to 18 trillion'); the same flag applies to all seven Qwen2.5 sizes"),
]


def apply_audit(B):
    B = B.copy()
    B["d_audit"] = ""
    for model, fld, old, new, src, q in D_AUDIT:
        i = B.index[B["model"] == model]
        if not len(i):
            continue
        assert np.isclose(B.loc[i[0], fld], old, rtol=1e-6), (model, fld, B.loc[i[0], fld])
        if fld == "N":
            # keep the embedding split consistent: shift the non-embedding count by the correction
            B.loc[i, "N_nonemb"] = B.loc[i, "N_nonemb"] + (new - B.loc[i, "N"])
        B.loc[i, fld] = new
        B.loc[i, "d_audit"] = B.loc[i, "d_audit"] + f"{fld} "
    B["M"] = B["D"] / B["N"]
    B["Cmp"] = 6 * B["N"] * B["D"]
    B["emb_share"] = 1 - B["N_nonemb"] / B["N"]
    return B


# [review] ra2's chunked wild bootstraps (techs.chin_q_wild and the non-embedding refit; DeepSeek's law takes the chin_q
# draws row by row) give every draw a fixed seed but return the draws in an order that depends on the number of worker
# processes (chunks seeds[i::n_proc]). The cache techs_cache.pkl was built with 4 workers. Reordering to that order makes
# a rebuild reproduce the cache bit for bit with any worker count (checked 2026-09-25 with 2 workers); the order matters
# only for statistics that pair draws of different technologies by index ('lab-own where available').
CACHE_WORKERS = 4
CHUNKED = ("chin_q", "chin_ne", "deepseek")


def _worker_order(n, k):
    return [i for c in range(k) for i in range(c, n, k)]


def _canonical_draw_order(T, used, target=CACHE_WORKERS):
    for key in CHUNKED:
        t = T.get(key)
        if t is None or t.draws is None or used == target:
            continue
        d = np.asarray(t.draws)
        by_seed = np.empty_like(d)
        by_seed[_worker_order(len(d), used)] = d
        t.draws = by_seed[_worker_order(len(d), target)]
    return T


def _build_techs():
    C.log("base: ra2 model-free designs (wild bootstrap B = 999) and technologies (kappa-free wild bootstrap B = 399)")
    mf_tab, mf_pb, mf_draws = MF.run_all(B=999)
    T, M = TT.build(mf_tab, mf_draws, B_wild=399, mf_pb=mf_pb)
    T = _canonical_draw_order(T, TT.N_PROC)                                                   # [review]
    return dict(T=T, M=M, mf_tab=mf_tab, mf_pb=mf_pb, mf_draws=mf_draws)


def load(rebuild=False):
    """Returns dict(d, B, B0, T, M, mf_tab, mf_pb, mf_draws, L, L0, keys, exante). B = clean-sample-flagged verified
    sample with the token audit applied; B0 = ra2's inputs (no audit). The technology registry (the only slow part,
    about 4 minutes on 4 processes) is cached in data/processed/rb2_decisions/techs_cache.pkl."""
    cache = os.path.join(C.PROC, "techs_cache.pkl")
    if rebuild or not os.path.exists(cache):
        X = _build_techs()
        with open(cache, "wb") as f:
            pickle.dump(X, f)
    else:
        with open(cache, "rb") as f:
            X = pickle.load(f)
    d, B0 = SM.build()
    clean = B0["clean"]
    cls, _ = SM.family_classes(B0, clean)
    B0.loc[clean, "fam_class"] = cls
    B0["fam_class"] = B0["fam_class"].fillna("")
    B = apply_audit(B0)
    cls, _ = SM.family_classes(B, B["clean"])
    B.loc[B["clean"], "fam_class"] = cls
    T, M = X["T"], X["M"]
    keys = list(M.loc[M["in_set"] | M["sensitivity"], "key"])
    exante = list(M.loc[M["in_set"], "key"])
    L = WG.wedge_long(B, T, keys)
    L0 = WG.wedge_long(B0, T, [C.REF])
    return dict(d=d, B=B, B0=B0, keys=keys, exante=exante, L=L, L0=L0, **X)


def audit_table():
    """Changed counts (D_AUDIT) and checked conventions that change nothing (D_NOTES), with verified quotes."""
    import evidence as E
    rows = []
    for model, fld, old, new, src, q in D_AUDIT:
        assert E.verify(src, q), (model, src)
        rows.append(dict(model=model, field=fld, ra2_value=old, corrected_value=new, changed=True, source_key=src,
                         url=E.SOURCES[src]["url"], quote=q, note=""))
    for model, fld, val, src, q, note in D_NOTES:
        assert E.verify(src, q), (model, src)
        rows.append(dict(model=model, field=fld, ra2_value=val, corrected_value=val, changed=False, source_key=src,
                         url=E.SOURCES[src]["url"], quote=q, note=note))
    return pd.DataFrame(rows)
