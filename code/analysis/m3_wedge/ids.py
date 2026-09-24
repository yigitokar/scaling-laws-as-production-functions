"""ids.py -- lists of Hugging Face repository ids that the download script (code/data/download_m3_wedge.sh) fetches.

Usage:  python ids.py base|instruct|config|cards
  base     : every Sample-B base model (canonical current id), for the HF model API (downloads, safetensors, createdAt)
  instruct : official post-trained counterparts (usage aggregation)
  config   : "<model id> <repo to read config.json from>" (ungated mirror for gated repos)
  cards    : Sample-B ids whose model card is checked for the training-token statement
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW  # noqa: E402
import curated as cu  # noqa: E402


def sample_b_ids():
    o = pd.read_csv(os.path.join(RAW, "obsscaling", "base_llm_benchmark_eval.csv"))
    ids = []
    for m in o["Model"]:
        if cu.OBS_FIXES.get(m, {}).get("drop"):
            continue
        if "pythia" in m.lower():
            continue  # replaced by the standard suite below
        ids.append(cu.CANONICAL.get(m, m))
    ids += [f"EleutherAI/pythia-{m}" for m in cu.PYTHIA_STD]
    ids += [c["hf"] for c in cu.CURATED]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def main(which):
    base = sample_b_ids()
    if which == "base":
        print("\n".join(base))
    elif which == "instruct":
        out = []
        for b in base:
            out += cu.INSTRUCT.get(b, [])
        print("\n".join(dict.fromkeys(out)))
    elif which == "config":
        for b in base:
            print(b, cu.CONFIG_MIRROR.get(b, b))
    elif which == "cards":
        print("\n".join(c["hf"] for c in cu.CURATED))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "base")
