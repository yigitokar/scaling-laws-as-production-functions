"""frame.py -- the coding frame of rb5_units: the 49 budget-level decision units of the clean sample with members,
sizes, token budgets, release dates and the keys of the saved primary sources that concern them. No wedge, share or
other outcome is written (the frame is what the blind second coder sees; protocol Sections 2 and 4).

Unit labels follow rb2_decisions' sub-group scheme so that other modules can merge on them: a family whose members all
share one budget keeps the family name (e.g. 'Llama-2'); a budget group inside a family with several budgets is
'<family>#k' (k-th budget in ascending D); a model that shares its budget with no family member is its own unit, labelled
by its uid.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import r5common as C

# saved primary sources (data/raw/rb2_decisions/<key>.txt) by family
SOURCES_BY_FAMILY = {
    "Apertus": ["apertus_report", "apertus_card"],
    "BTLM": ["btlm_blog"],
    "DeepSeek-LLM": ["deepseek_llm_report", "deepseek_llm_readme"],
    "Falcon": ["falcon_report", "falcon180b_blog"],
    "Falcon2": ["falcon2_news"],
    "Falcon3": ["falcon3_blog"],
    "Gemma": ["gemma_report", "gemma_blog"],
    "Gemma-2": ["gemma2_blog"],
    "Granite-3.0": ["granite30_report", "granite30_announcement"],
    "H2O-Danube1": ["danube_card"],
    "H2O-Danube3": ["danube3_report", "danube3_card"],
    "Llama": ["llama1_report", "llama1_blog"],
    "Llama-2": ["llama2_report", "llama2_blog"],
    "Llama-3 herd": ["llama3_report", "llama3_blog", "llama31_blog"],
    "MAP-Neo": ["neo_readme"],
    "MPT": ["mpt7b_blog", "mpt30b_blog"],
    "Marin": ["marin_card"],
    "OLMo-1": ["olmo_report"],
    "OLMo-1.7": [],
    "OLMo-2": ["olmo2_report", "olmo2_blog", "olmo2_32b_blog"],
    "Olmo-3": ["olmo3_report", "olmo3_blog"],
    "Phi-1.5": ["phi15_card"],
    "Phi-2": ["phi2_blog"],
    "Qwen": ["qwen_report", "qwen_readme"],
    "Qwen2": ["qwen2_report", "qwen2_blog"],
    "Qwen2.5": ["qwen25_report", "qwen25_blog"],
    "Qwen3": ["qwen3_report", "qwen3_blog"],
    "SmolLM": ["smollm_blog"],
    "SmolLM2": ["smollm2_report", "smollm2_card"],
    "SmolLM3": ["smollm3_blog"],
    "StableLM-2": ["stablelm2_report"],
    "StableLM-3B-4E1T": ["stablelm_github"],
    "StableLM-alpha": ["stablelm_github"],
    "StableLM-alpha-v2": ["stablelm_github"],
    "TinyLlama": ["tinyllama_readme"],
    "Yi": ["yi_report", "yi_readme"],
}


def clean_sample():
    """The 77-model clean sample with the audited counts (rb2_decisions' clean_models.csv; the frame uses only
    identifiers, N, D and dates)."""
    B = pd.read_csv(os.path.join(C.PROC_RB2, "clean_models.csv"))
    return B


def build(B=None):
    B = clean_sample() if B is None else B
    B = B.copy()
    B["gen"] = B["gen"] if "gen" in B else B["family"]
    grp = C.budget_groups(B)
    B["group"] = B["uid"].map(grp)
    size = B.groupby("group")["uid"].transform("count")
    nfam = B.groupby("gen")["uid"].transform("count")
    ngrp = B.groupby("gen")["group"].transform("nunique")
    B["unit"] = np.where(size >= 2, np.where(ngrp == 1, B["gen"], B["group"]), B["uid"])
    B["unit_kind"] = np.where(size >= 2, "common-budget", np.where(nfam >= 2, "size-specific member", "singleton"))
    man = C.manifest()
    rows = []
    for u, g in B.groupby("unit", sort=False):
        g = g.sort_values("N")
        keys = SOURCES_BY_FAMILY.get(g["gen"].iloc[0], [])
        missing = [k for k in keys if not os.path.exists(os.path.join(C.RAW, f"{k}.txt"))]
        assert not missing, (u, missing)
        rows.append(dict(
            unit=u, unit_kind=g["unit_kind"].iloc[0], family=g["gen"].iloc[0], developer=g["dev"].iloc[0],
            n_members=len(g), members="; ".join(g["model"]), uids="; ".join(g["uid"]),
            N_B="; ".join(f"{x / 1e9:.3g}" for x in g["N"]), D_T="; ".join(f"{x / 1e12:.3g}" for x in g["D"]),
            release_dates="; ".join(pd.to_datetime(g["date"]).dt.date.astype(str)),
            year=int(pd.to_datetime(g["date"]).dt.year.min()),
            other_family_members="; ".join(f"{m} (D = {d / 1e12:.3g}T)" for m, d in
                                          B.loc[(B["gen"] == g["gen"].iloc[0]) & (B["unit"] != u), ["model", "D"]].values),
            source_keys="; ".join(keys),
            source_urls="; ".join(man.get(k, {}).get("url", "") for k in keys)))
    F = pd.DataFrame(rows).sort_values(["unit_kind", "family", "unit"]).reset_index(drop=True)
    return F, B[["uid", "model", "gen", "dev", "unit", "unit_kind"]]


if __name__ == "__main__":
    F, M = build()
    F.to_csv(os.path.join(C.PROC, "coding_frame.csv"), index=False)
    print(F[["unit", "unit_kind", "members", "D_T"]].to_string())
    print(len(F), F["unit_kind"].value_counts().to_dict())
