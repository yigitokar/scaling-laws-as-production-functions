"""posttrain.py -- post-training compute as a size-dependent training cost (R2 Major 7).

Post-training (SFT, preference tuning, RL including rollouts) is planned when N is chosen, scales with N at given
post-training tokens and rollouts, and does not scale with D. With post-training expenditure X_P = x_P X_T (x_P = the
ratio of post-training to pre-training compute of the same model), the first-order condition in N gains X_P and the
one in D does not, so w = 1 + x_P + m_N (it acts like delta > 0 in Prop. A8(f)): the value of compactness is
m_N = w - 1 - x_P and s_corrected = (w - 1 - x_P)/(w - x_P). Mid-training and long-context tokens are counted in D
where the developer reports them (OLMo 2/3), so they are not part of x_P.

Disclosed budgets (quotes verified against the saved reports; data/raw/rb2_decisions):
  DeepSeek-V3: post-training 5K of 2,664K pre-training H800 GPU hours -> x_P = 0.0019.
  Tulu 3 on Llama 3.1: final SFT 8B on 32 GPUs x 6 h; 70B on 64 GPUs x 50 h -> x_P ~ 1e-4 (FLOP estimate at 40% MFU).
  Olmo 3 Think 32B: pre-training ~47 days on a 1,024-GPU cluster (9.5 days of it on 512 GPUs); RL 5 days on 224 GPUs
    (8 learner + 20 inference nodes) -> x_P(RL) = 0.026; Olmo 3.1 continued RL 21 more days -> 0.135; all
    post-training (SFT, DPO, RL, sweeps) took ~9 days of the cluster -> upper bound 0.21; with the 21-day Olmo 3.1
    continuation on 224 GPUs -> 0.32 (round 3, fix list W18(c)).
Round 3: the scenario grid includes x_P = 0.32, and the compute-weighted aggregates are also given with m/(1 + m)
(columns s_agg_<year>_mm), the definition of the medians.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb2common as C
import evidence as E

H100_FLOPS = 989e12
MFU = 0.40

SOURCES = [
    ("DeepSeek-V3 (all post-training)", "deepseek_v3_report",
     "Combined with 119K GPU hours for the context length extension and 5K GPU hours for post-training, DeepSeek-V3 costs only 2.788M GPU hours for its full training.",
     5e3 / 2664e3, "GPU hours, H800"),
    ("Tulu 3 SFT on Llama 3.1 8B (final run)", "tulu3_report",
     "The final 8B model is trained on 32 GPUs for 6 hours and the 70B model was trained on 64 GPUs for 50 hours.",
     32 * 6 * 3600 * H100_FLOPS * MFU / (6 * 8.03e9 * 15e12), "FLOPs at 40% MFU / 6ND"),
    ("Tulu 3 SFT on Llama 3.1 70B (final run)", "tulu3_report",
     "The final 8B model is trained on 32 GPUs for 6 hours and the 70B model was trained on 64 GPUs for 50 hours.",
     64 * 50 * 3600 * H100_FLOPS * MFU / (6 * 70.6e9 * 15e12), "FLOPs at 40% MFU / 6ND"),
    ("Olmo 3 Think 32B, RL (initial release)", "olmo3_report",
     "The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days with at least a day of training time lost due to stability issues.",
     5 * 224 / (9.5 * 512 + (47 - 9.5) * 1024), "GPU days; 224 GPUs"),
    ("Olmo 3.1 Think 32B, RL incl. 21-day continuation", "olmo3_report",
     "we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.",
     (5 + 21) * 224 / (9.5 * 512 + (47 - 9.5) * 1024), "GPU days"),
    ("Olmo 3 Think 32B, all post-training, upper bound", "olmo3_report",
     "Post-training: ∼9 days(SFT, DPO, and RL)",
     9 * 1024 / (9.5 * 512 + (47 - 9.5) * 1024), "GPU days; 9 days x 1,024 GPUs"),
    # round 3 (fix list W18(c); numbers_wedge #16): the 9 cluster-days plus the 21-day Olmo 3.1 RL continuation
    ("Olmo 3.1 Think 32B, all post-training incl. 21-day continuation, upper bound", "olmo3_report",
     "we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.",
     (9 * 1024 + 21 * 224) / (9.5 * 512 + (47 - 9.5) * 1024), "GPU days; 9 days x 1,024 GPUs + 21 days x 224 GPUs"),
]
EXTRA_QUOTES = [("olmo3_report", "Inference dominated our computational costs, using 8 H100 nodes for training and 20 nodes for inference for the 32B OlmoRL reasoner model.")]


def disclosed():
    rows = []
    for lab, key, q, x, unit in SOURCES:
        if not E.verify(key, q):
            raise AssertionError(f"post-training quote not found: {lab}")
        rows.append(dict(budget=lab, x_P=x, unit=unit, source_key=key, quote=q))
    for key, q in EXTRA_QUOTES:
        if not E.verify(key, q):
            raise AssertionError("post-training quote not found (RL nodes)")
    return pd.DataFrame(rows)


def effect(Bc, UT, wcol="w_ref"):
    """Median s, share w > 1 and the 2023-2025 compute-weighted aggregate after removing x_P from w - 1."""
    rows = []
    for x in (0.0, 0.002, 0.026, 0.135, 0.21, 0.32):
        w = Bc[wcol].values
        m = w - 1 - x
        s = m / (1 + m)
        wu = UT["w_ref"].values
        mu = wu - 1 - x
        su = mu / (1 + mu)
        row = dict(x_P=x, median_s_models=float(np.median(s)), share_mN_gt0_models=float(np.mean(m > 0)),
                   median_s_units=float(np.median(su)), share_mN_gt0_units=float(np.mean(mu > 0)))
        for y in (2023, 2024, 2025):
            g = Bc["year"].values == y
            mp = np.maximum(m[g], 0)
            row[f"s_agg_{y}"] = float((mp * Bc["Cmp"].values[g]).sum() / ((1 + mp + x) * Bc["Cmp"].values[g]).sum())
            # round 3 (W19(b), W23(q)): the same aggregate with m/(1 + m), post-training compute outside the cost base
            row[f"s_agg_{y}_mm"] = float((mp * Bc["Cmp"].values[g]).sum() / ((1 + mp) * Bc["Cmp"].values[g]).sum())
            row[f"median_s_{y}"] = float(np.median(s[g]))
        rows.append(row)
    # 2025 models only get the reasoning-era RL budget (x_P = 0.135), earlier models the DeepSeek-V3 level (0.002)
    x_era = np.where(Bc["year"].values >= 2025, 0.135, 0.002)
    m = Bc[wcol].values - 1 - x_era
    s = m / (1 + m)
    row = dict(x_P=np.nan, scenario="x_P = 0.135 for 2025 releases, 0.002 before", median_s_models=float(np.median(s)),
               share_mN_gt0_models=float(np.mean(m > 0)))
    for y in (2023, 2024, 2025):
        g = Bc["year"].values == y
        mp = np.maximum(m[g], 0)
        row[f"s_agg_{y}"] = float((mp * Bc["Cmp"].values[g]).sum() / ((1 + mp + x_era[g]) * Bc["Cmp"].values[g]).sum())
        row[f"s_agg_{y}_mm"] = float((mp * Bc["Cmp"].values[g]).sum() / ((1 + mp) * Bc["Cmp"].values[g]).sum())
        row[f"median_s_{y}"] = float(np.median(s[g]))
    rows.append(row)
    return pd.DataFrame(rows)
