"""Build the observational (cross-lab) panel of base LLMs.

Sources (data/raw):
  obsscaling/base_llm_benchmark_eval.csv   Ruan et al. (2024): 148 base models, Open LLM Leaderboard v1 metrics
  sloth/data_v2.csv                        Maia Polo et al.: adds base models (Pythia standard, Yi-9B, ...) and D
  hf_meta/*.json                           HF model API: exact parameter count (safetensors.total), upload date
  epoch_models/all_ai_models.csv           Epoch AI: publication date, organization, confidence, hardware, notability
  epoch_models/ml_hardware.csv             Epoch AI: accelerator release dates/prices/FLOP/s (IV construction)

Data-construction decisions (all documented in the memo):
  * Units: N = total parameters (exact safetensors count where available, else reported); D = pretraining tokens
    processed (incl. repeated epochs and, for continued-pretraining models, the parent's tokens); C = 6ND (FLOP).
  * Known errors in the source CSVs are corrected (dictionary FIXES below, each with its source).
  * MoE/hybrid/RNN architectures are flagged; the main sample is dense transformers.
  * Output: OLLM-v1 accuracies (same harness and prompts for every model): HellaSwag (10-shot acc_norm),
    ARC-Challenge (25-shot acc_norm), Winogrande (5-shot), MMLU (5-shot), GSM8K (5-shot), TruthfulQA (MC2).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from common import PROC, RAW, add_outputs

# ----------------------------------------------------------------------------- family metadata
# family -> (developer, country, line, generation, arch, domain, special)
#   line/generation: product line and generation index for the dynamic-panel estimators
#   arch: dense | moe | rnn | hybrid ; domain: general | code ; special: '' | distilled | synthetic | continued
FAM = {
    "Llama": ("Meta", "US", "Meta-Llama", 1, "dense", "general", ""),
    "Llama-2": ("Meta", "US", "Meta-Llama", 2, "dense", "general", ""),
    "Llama-3": ("Meta", "US", "Meta-Llama", 3, "dense", "general", ""),
    "Llama-3.1": ("Meta", "US", "Meta-Llama", 4, "dense", "general", ""),
    "CodeLlama": ("Meta", "US", "Meta-CodeLlama", 1, "dense", "code", "continued"),
    "OPT": ("Meta", "US", "Meta-OPT", 1, "dense", "general", ""),
    "XGLM": ("Meta", "US", "Meta-XGLM", 1, "dense", "general", ""),
    "Qwen": ("Alibaba", "China", "Alibaba-Qwen", 1, "dense", "general", ""),
    "Qwen1.5": ("Alibaba", "China", "Alibaba-Qwen", 2, "dense", "general", ""),
    "Qwen2": ("Alibaba", "China", "Alibaba-Qwen", 3, "dense", "general", ""),
    "Mistral": ("Mistral", "France", "Mistral", 1, "dense", "general", ""),
    "Mixtral": ("Mistral", "France", "Mistral-MoE", 1, "moe", "general", ""),
    "DeepSeek-LLM": ("DeepSeek", "China", "DeepSeek-LLM", 1, "dense", "general", ""),
    "DeepSeek-MoE": ("DeepSeek", "China", "DeepSeek-MoE", 1, "moe", "general", ""),
    "DeepSeek-V2": ("DeepSeek", "China", "DeepSeek-MoE", 2, "moe", "general", ""),
    "DeepSeek-Coder": ("DeepSeek", "China", "DeepSeek-Coder", 1, "dense", "code", ""),
    "DeepSeek-Coder-V2": ("DeepSeek", "China", "DeepSeek-Coder", 2, "moe", "code", "continued"),
    "Yi": ("01.AI", "China", "01AI-Yi", 1, "dense", "general", ""),
    "Yi-1.5": ("01.AI", "China", "01AI-Yi", 2, "dense", "general", "continued"),
    "Yi-200K": ("01.AI", "China", "01AI-Yi200K", 1, "dense", "general", "continued"),
    "Gemma": ("Google", "US", "Google-Gemma", 1, "dense", "general", ""),
    "Gemma-2": ("Google", "US", "Google-Gemma", 2, "dense", "general", "distilled"),  # 2B/9B distilled (27B not)
    "RecurrentGemma": ("Google", "US", "Google-RecurrentGemma", 1, "hybrid", "general", ""),
    "Jamba": ("AI21", "Israel", "AI21-Jamba", 1, "hybrid", "general", ""),
    "Falcon": ("TII", "UAE", "TII-Falcon", 1, "dense", "general", ""),
    "Phi": ("Microsoft", "US", "Microsoft-Phi", 1, "dense", "general", "synthetic"),
    "Pythia": ("EleutherAI", "US", "EleutherAI", 2, "dense", "general", ""),
    "GPT-Neo/J": ("EleutherAI", "US", "EleutherAI", 1, "dense", "general", ""),
    "BLOOM": ("BigScience", "Multinational", "BigScience-BLOOM", 1, "dense", "general", ""),
    "MPT": ("MosaicML", "US", "MosaicML-MPT", 1, "dense", "general", ""),
    "StarCoder": ("BigCode", "Multinational", "BigCode-StarCoder", 1, "dense", "code", ""),
    "StarCoder2": ("BigCode", "Multinational", "BigCode-StarCoder", 2, "dense", "code", ""),
    "OpenLlama": ("OpenLM Research", "US", "OpenLM-OpenLlama", 1, "dense", "general", ""),
    "OpenLlamaV2": ("OpenLM Research", "US", "OpenLM-OpenLlama", 2, "dense", "general", ""),
    "GPT-2": ("OpenAI", "US", "OpenAI-GPT2", 1, "dense", "general", ""),
    "InternLM2": ("Shanghai AI Lab", "China", "Shanghai-InternLM", 2, "dense", "general", ""),
    "DeciLM": ("Deci", "Israel", "Deci", 1, "dense", "general", ""),
    "StableLM": ("Stability AI", "UK", "Stability-StableLM", None, "dense", "general", ""),  # gen set per model
    "RWKV": ("RWKV Foundation", "US", "RWKV", 1, "rnn", "general", ""),
    "RedPajama-INCITE-Base": ("Together", "US", "Together-RedPajama", 1, "dense", "general", ""),
    "Amber": ("LLM360", "UAE", "LLM360-Amber", 1, "dense", "general", ""),
    "Codegen": ("Salesforce", "US", "Salesforce-Codegen", 1, "dense", "general", ""),
    "SmolLM": ("Hugging Face", "US", "HF-SmolLM", 1, "dense", "general", "synthetic"),
    "Cerebras-GPT": ("Cerebras", "US", "Cerebras-GPT", 1, "dense", "general", ""),
    "BTLM": ("Cerebras", "US", "Cerebras-BTLM", 1, "dense", "general", ""),
    "H2O-Danube": ("H2O.ai", "US", "H2O-Danube", None, "dense", "general", ""),  # gen set per model
    "OLMo": ("AI2", "US", "AI2-OLMo", None, "dense", "general", ""),             # gen set per model
    "TinyLlama": ("TinyLlama", "Singapore", "TinyLlama", 1, "dense", "general", ""),
}
GEN_OVERRIDE = {  # model id -> generation index for lines whose ObsScaling family pools generations
    "stabilityai/stablelm-base-alpha-3b": 1, "stabilityai/stablelm-base-alpha-7b": 1,
    "stabilityai/stablelm-base-alpha-7b-v2": 2, "stabilityai/stablelm-2-1_6b": 3,
    "h2oai/h2o-danube-1.8b-base": 1, "h2oai/h2o-danube2-1.8b-base": 2, "h2oai/h2o-danube3-500m-base": 3,
    "h2oai/h2o-danube3-4b-base": 3, "allenai/OLMo-1B-hf": 1, "allenai/OLMo-7B-hf": 1, "allenai/OLMo-7B-0424-hf": 2,
}
# models distilled from a larger teacher (value-added TFP overstated; GNR gross-output issue)
DISTILLED = {"google/gemma-2-2b", "google/gemma-2-9b"}

# ----------------------------------------------------------------------------- corrections to source CSVs
# id -> dict(N=params in B, D=tokens in T, note)
FIXES = {
    "cerebras/btlm-3b-8k-base": dict(D=0.627, note="ObsScaling lists 627T; BTLM trained on 627B tokens (Epoch: 6.27e11)"),
    "openlm-research/open_llama_7b_v2": dict(N=7.0, note="ObsScaling lists 13B; open_llama_7b_v2 is a 7B model"),
    "openlm-research/open_llama_13b": dict(D=1.0, note="ObsScaling lists 0.6T; final OpenLLaMA-13B trained on 1T (Epoch: 1e12)"),
    "google/gemma-2b": dict(D=3.0, note="ObsScaling/Sloth list 6T; Gemma report and Epoch: 2B trained on 3T"),
    "Qwen/Qwen2-72B": dict(D=7.0, note="missing in ObsScaling; Qwen2 report/Epoch/Sloth: 7T"),
    "Qwen/Qwen2-7B": dict(D=7.0, note="missing in ObsScaling; Qwen2 report/Epoch/Sloth: 7T"),
    "Qwen/Qwen2-1.5B": dict(D=7.0, note="missing in ObsScaling; Qwen2 report/Epoch/Sloth: 7T"),
    "Qwen/Qwen2-0.5B": dict(D=12.0, note="missing in ObsScaling; Qwen2 report/Epoch/Sloth: 12T"),
    "internlm/internlm2-20b": dict(D=2.6, note="missing in ObsScaling; Epoch: 2.6e12"),
    "RWKV/rwkv-4-169m-pile": dict(D=0.33, note="Pile, 330B tokens (Sloth; Epoch RWKV-4 14B)"),
    "RWKV/rwkv-4-430m-pile": dict(D=0.33, note="Pile"), "RWKV/rwkv-4-1b5-pile": dict(D=0.33, note="Pile"),
    "RWKV/rwkv-4-3b-pile": dict(D=0.33, note="Pile"), "RWKV/rwkv-4-7b-pile": dict(D=0.33, note="Pile"),
    "RWKV/rwkv-4-14b-pile": dict(D=0.33, note="Pile"),
    # MoE: use ACTIVE parameters as the flow input (C = 6 N_active D); total kept in N_total
    "mistralai/Mixtral-8x7B-v0.1": dict(N=12.9, note="MoE: active 12.9B of 46.7B total; D undisclosed"),
    "mistralai/Mixtral-8x22B-v0.1": dict(N=39.0, note="MoE: active 39B of 141B total; D undisclosed"),
    "deepseek-ai/DeepSeek-V2": dict(N=21.0, note="MoE: active 21B of 236B"),
    "deepseek-ai/DeepSeek-Coder-V2-Base": dict(N=21.0, note="MoE: active 21B of 236B"),
    "deepseek-ai/deepseek-moe-16b-base": dict(N=2.8, note="MoE: active 2.8B of 16.4B"),
    "Qwen/Qwen2-57B-A14B": dict(N=14.0, note="MoE: active 14B of 57B"),
}
MOE_ACTIVE = {k for k, v in FIXES.items() if "MoE" in v.get("note", "")}

# Sloth base models that are NOT in ObsScaling (name in Sloth -> (HF id, family))
SLOTH_EXTRA = {
    "pythia-70m": ("EleutherAI/pythia-70m", "Pythia"), "pythia-160m": ("EleutherAI/pythia-160m", "Pythia"),
    "pythia-410m": ("EleutherAI/pythia-410m", "Pythia"), "pythia-1b": ("EleutherAI/pythia-1b", "Pythia"),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", "Pythia"), "pythia-2.8b": ("EleutherAI/pythia-2.8b", "Pythia"),
    "pythia-6.9b": ("EleutherAI/pythia-6.9b", "Pythia"), "pythia-12b": ("EleutherAI/pythia-12b", "Pythia"),
    "yi-9b": ("01-ai/Yi-9B", "Yi"), "recurrentgemma-2b": ("google/recurrentgemma-2b", "RecurrentGemma"),
}
SLOTH_D = {"EleutherAI/pythia-" + m: 0.3 for m in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]}
SLOTH_D.update({"01-ai/Yi-9B": 3.8, "google/recurrentgemma-2b": 2.0})
SLOTH_N = {"EleutherAI/pythia-" + m: v for m, v in
           zip(["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"], [0.07, 0.16, 0.41, 1.0, 1.4, 2.8, 6.9, 12.0])}
SLOTH_N.update({"01-ai/Yi-9B": 9.0, "google/recurrentgemma-2b": 2.0})

# ----------------------------------------------------------------------------- Epoch crosswalk (HF id -> Epoch 'Model')
EPOCH_NAME = {
    "meta-llama/Llama-2-7b-hf": "Llama 2-7B", "meta-llama/Llama-2-13b-hf": "Llama 2-13B",
    "meta-llama/Llama-2-70b-hf": "Llama 2-70B", "huggyllama/llama-7b": "LLaMA-7B", "huggyllama/llama-13b": "LLaMA-13B",
    "huggyllama/llama-30b": "LLaMA-33B", "huggyllama/llama-65b": "LLaMA-65B",
    "meta-llama/Meta-Llama-3-70B": "Llama 3-70B", "meta-llama/Meta-Llama-3-8B": "Llama 3-8B",
    "meta-llama/Meta-Llama-3.1-405B-FP8": "Llama 3.1-405B", "meta-llama/Meta-Llama-3.1-70B": "Llama 3.1-70B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama 3.1-8B",
    "Qwen/Qwen1.5-110B": "Qwen1.5-110B", "Qwen/Qwen1.5-72B": "Qwen1.5-72B", "Qwen/Qwen1.5-32B": "Qwen1.5-32B",
    "Qwen/Qwen1.5-14B": "Qwen1.5-14B", "Qwen/Qwen1.5-7B": "Qwen1.5-7B", "Qwen/Qwen-72B": "Qwen-72B",
    "Qwen/Qwen-14B": "Qwen-14B", "Qwen/Qwen-7B": "Qwen-7B", "Qwen/Qwen2-72B": "Qwen2-72B",
    "Qwen/Qwen2-57B-A14B": "Qwen2-57B-A14B", "Qwen/Qwen2-7B": "Qwen2-7B", "Qwen/Qwen2-1.5B": "Qwen2-1.5B",
    "Qwen/Qwen2-0.5B": "Qwen2-0.5B", "mistralai/Mistral-7B-v0.1": "Mistral 7B",
    "mistralai/Mixtral-8x7B-v0.1": "Mixtral 8x7B", "mistralai/Mixtral-8x22B-v0.1": "Mixtral 8x22B",
    "mistralai/Mistral-Nemo-Base-2407": "Mistral NeMo", "deepseek-ai/DeepSeek-V2": "DeepSeek-V2 (MoE-236B)",
    "deepseek-ai/DeepSeek-Coder-V2-Base": "DeepSeek-Coder-V2 236B", "01-ai/Yi-34B": "Yi-34B",
    "01-ai/Yi-1.5-34B": "Yi-1.5-34B", "01-ai/Yi-1.5-9B": "Yi-1.5-9B", "google/gemma-7b": "Gemma 7B",
    "google/gemma-2b": "Gemma 2B", "google/gemma-2-27b": "Gemma 2 27B", "google/gemma-2-9b": "Gemma 2 9B",
    "google/gemma-2-2b": "Gemma 2 2B", "ai21labs/Jamba-v0.1": "Jamba", "tiiuae/falcon-180B": "Falcon-180B",
    "tiiuae/falcon-40b": "Falcon-40B", "tiiuae/falcon-7b": "Falcon-7B", "microsoft/phi-2": "Phi-2",
    "microsoft/phi-1_5": "Phi-1.5",
    **{f"EleutherAI/pythia-{m}{v}": f"Pythia-{m}" for m in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
       for v in ["", "-deduped"]},
    "bigscience/bloom-560m": "BLOOM-560M", "bigscience/bloom-1b1": "BLOOM-1B", "bigscience/bloom-3b": "BLOOM-3B",
    "bigscience/bloom-7b1": "BLOOM-7.1B", "bigscience/bloom": "BLOOM-176B", "EleutherAI/gpt-neox-20b": "GPT-NeoX-20B",
    "EleutherAI/gpt-neo-2.7B": "GPT-Neo-2.7B", "EleutherAI/gpt-neo-1.3B": "GPT-Neo-1.3B",
    "EleutherAI/gpt-neo-125m": "GPT-Neo-125M", "EleutherAI/gpt-j-6b": "GPT-J-6B", "facebook/opt-6.7b": "OPT-6.7B",
    "facebook/opt-1.3b": "OPT-1.3B", "facebook/opt-350m": "OPT-350M", "facebook/opt-13b": "OPT-13B",
    "facebook/opt-2.7b": "OPT-2.7B", "facebook/opt-30b": "OPT-30B", "facebook/opt-66b": "OPT-66B",
    "mosaicml/mpt-30b": "MPT-30B", "mosaicml/mpt-7b": "MPT-7B", "facebook/xglm-564M": "XGLM",
    "facebook/xglm-7.5B": "XGLM-7.5B", "codellama/CodeLlama-7b-hf": "Code Llama-7B",
    "codellama/CodeLlama-13b-hf": "Code Llama-13B", "codellama/CodeLlama-34b-hf": "Code Llama-34B",
    "codellama/CodeLlama-70b-hf": "Code Llama-70B", "bigcode/starcoderbase": "StarCoder",
    "bigcode/starcoder2-15b": "StarCoder 2 15B", "bigcode/starcoder2-7b": "StarCoder 2 7B",
    "bigcode/starcoder2-3b": "StarCoder 2 3B", "deepseek-ai/deepseek-coder-1.3b-base": "DeepSeek Coder 1.3B",
    "deepseek-ai/deepseek-coder-6.7b-base": "DeepSeek Coder 6.7B", "deepseek-ai/deepseek-coder-33b-base": "DeepSeek Coder 33B",
    "openlm-research/open_llama_13b": "OpenLLaMA-13B", "internlm/internlm2-20b": "InternLM2-20B",
    "deepseek-ai/deepseek-llm-67b-base": "DeepSeek LLM 67B", "deepseek-ai/deepseek-llm-7b-base": "DeepSeek LLM 7B",
    "deepseek-ai/deepseek-moe-16b-base": "DeepSeekMoE-16B", "stabilityai/stablelm-base-alpha-7b-v2": "StableLM-Base-Alpha-7B",
    "stabilityai/stablelm-2-1_6b": "StableLM-2-1.6B", "RWKV/rwkv-4-14b-pile": "RWKV-4 14B",
    "togethercomputer/RedPajama-INCITE-7B-Base": "RedPajama-INCITE-7B-Base", "LLM360/Amber": "Amber",
    "HuggingFaceTB/SmolLM-1.7B": "SmolLM-1.7B", "cerebras/btlm-3b-8k-base": "BTLM-3B",
    "allenai/OLMo-1B-hf": "OLMo-1B", "allenai/OLMo-7B-hf": "OLMo-7B", "allenai/OLMo-7B-0424-hf": "OLMo 1.7-7B",
}
# release dates where neither Epoch nor the HF upload date is the release (HF migrations/re-uploads)
DATE_OVERRIDE = {
    "EleutherAI/gpt-neo-125m": "2021-03-21", "EleutherAI/gpt-neo-1.3B": "2021-03-21",
    "EleutherAI/gpt-neo-2.7B": "2021-03-21", "facebook/xglm-1.7B": "2021-12-20", "facebook/xglm-4.5B": "2021-12-20",
    "facebook/xglm-564M": "2021-12-20", "facebook/opt-125m": "2022-05-02", "openai-community/gpt2": "2019-02-14",
    "openai-community/gpt2-medium": "2019-02-14", "openai-community/gpt2-large": "2019-02-14",
    "openai-community/gpt2-xl": "2019-02-14", "Salesforce/codegen-6B-nl": "2022-03-25",
    "Salesforce/codegen-16B-nl": "2022-03-25", "cerebras/Cerebras-GPT-111M": "2023-03-28",
    "cerebras/Cerebras-GPT-256M": "2023-03-28", "cerebras/Cerebras-GPT-590M": "2023-03-28",
    "cerebras/Cerebras-GPT-1.3B": "2023-03-28", "cerebras/Cerebras-GPT-2.7B": "2023-03-28",
    "cerebras/Cerebras-GPT-6.7B": "2023-03-28", "cerebras/btlm-3b-8k-base": "2023-07-24",
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": "2023-05-05",
}
# Open LLM Leaderboard v1 scores missing in ObsScaling, taken from Sloth data_v2 (same leaderboard)
FILL_FROM_SLOTH = {"meta-llama/Meta-Llama-3-8B": "meta-llama-3-8b", "meta-llama/Meta-Llama-3-70B": "meta-llama-3-70b"}


def hf_param_count(d):
    """Exact parameter count from the HF API `safetensors` block, counting FLOATING-POINT tensors only.
    [review fix] `safetensors.total` also counts non-parameter buffers stored in the checkpoint: GPT-NeoX/GPT-Neo
    models (Pythia, GPT-NeoX-20B, GPT-Neo) store the causal-mask buffer `attention.bias` as U8 (2048x2048 per
    layer), so e.g. Pythia-70m has total = 95.6M = 70.4M weights + 25.2M mask bytes (+36%)."""
    st = d.get("safetensors") or {}
    par = st.get("parameters") or {}
    nf = sum(v for k, v in par.items() if str(k).upper().startswith(("F", "BF")))
    return float(nf) if nf > 0 else (float(st["total"]) if st.get("total") else np.nan)


def _hf_meta():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "hf_meta", "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "id" not in d:
            continue
        rows.append(dict(model=d["id"], hf_created=(d.get("createdAt") or "")[:10] or None,
                         N_exact=hf_param_count(d), hf_downloads=d.get("downloadsAllTime")))
    h = pd.DataFrame(rows)
    # HF ids that changed case/path since ObsScaling was compiled
    alias = {"IFM/Amber": "LLM360/Amber", "meta-llama/Llama-3.1-405B-FP8": "meta-llama/Meta-Llama-3.1-405B-FP8",
             "meta-llama/Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B", "meta-llama/Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
             "openai-community/gpt2": "openai-community/gpt2"}
    h["model"] = h["model"].replace(alias)
    return h


def hardware_table():
    h = pd.read_csv(os.path.join(RAW, "epoch_models", "ml_hardware.csv"))
    h["date"] = pd.to_datetime(h["Release date"], errors="coerce")
    flops = h["Tensor-FP16/BF16 performance (FLOP/s)"].fillna(h["FP16 (half precision) performance (FLOP/s)"])
    h["flops16"] = flops
    h["price"] = h["Release price (USD)"]
    h["ppd"] = h["flops16"] / h["price"]          # FLOP/s per USD at release price
    return h


def frontier_ppd(dates, lag_months=6):
    """ln of the best FP16/BF16 FLOP/s per USD among accelerators with a release price, released at least
    `lag_months` before `dates` (a cost shifter for training compute available at the training date)."""
    h = hardware_table().dropna(subset=["date", "ppd"])
    # data-center training accelerators only (list price >= $5,000); consumer GPUs have higher FLOP/$ but are
    # not what frontier pretraining runs use
    h = h[h.Type.isin(["GPU", "TPU"]) & (h.price >= 5000)]
    out = []
    for t in pd.to_datetime(dates):
        if pd.isna(t):
            out.append(np.nan)
            continue
        cut = t - pd.DateOffset(months=lag_months)
        s = h[h.date <= cut]
        out.append(np.log(s.ppd.max()) if len(s) else np.nan)
    return np.array(out)


def own_hardware_ppd(hw_strings):
    """ln FLOP/s per USD of the (first-listed) training accelerator recorded by Epoch, where priced."""
    h = hardware_table()
    names = h["Hardware name"].str.lower()
    out = []
    for s in hw_strings:
        if not isinstance(s, str):
            out.append(np.nan)
            continue
        first = s.split(",")[0].strip().lower()
        m = h[names == first]
        if not len(m):
            m = h[names.str.startswith(first[:18])]
        v = m.ppd.dropna()
        out.append(np.log(v.iloc[0]) if len(v) else np.nan)
    return np.array(out)


def size_tier(N_b):
    edges = [0, 0.25, 0.75, 2.2, 5.0, 10.5, 20, 45, 90, 1e9]
    labels = ["<0.25B", "0.25-0.75B", "0.75-2.2B", "2.2-5B", "5-10.5B", "10.5-20B", "20-45B", "45-90B", ">90B"]
    return pd.cut(N_b, edges, labels=labels).astype(str)


def build_panel(verbose=True):
    o = pd.read_csv(os.path.join(RAW, "obsscaling", "base_llm_benchmark_eval.csv"))
    o = o.rename(columns={"Model": "model", "Model Family": "family", "Model Size (B)": "N_rep",
                          "Pretraining Data Size (T)": "D_rep", "MMLU": "mmlu", "ARC-C": "arc_c",
                          "HellaSwag": "hellaswag", "Winograd": "winogrande", "TruthfulQA": "truthfulqa",
                          "GSM8K": "gsm8k", "XWinograd": "xwinograd", "HumanEval": "humaneval"})
    o["source"] = "ObsScaling"
    s = pd.read_csv(os.path.join(RAW, "sloth", "data_v2.csv"))
    s = s[(~s.Instruct) & s.Model.isin(SLOTH_EXTRA)].copy()
    ex = pd.DataFrame({"model": [SLOTH_EXTRA[m][0] for m in s.Model], "family": [SLOTH_EXTRA[m][1] for m in s.Model],
                       "mmlu": s.MMLU.values, "arc_c": s.ARC.values, "hellaswag": s.HellaSwag.values,
                       "winogrande": s.Winogrande.values, "truthfulqa": s.TruthfulQA.values, "gsm8k": s.GSM8K.values})
    ex["N_rep"] = ex.model.map(SLOTH_N)
    ex["D_rep"] = ex.model.map(SLOTH_D)
    ex["source"] = "Sloth"
    # [review fix] Sloth's rows for pythia-70m/1b/1.4b/2.8b/6.9b (standard) carry scores IDENTICAL, to 6 decimals on all
    # six benchmarks, to ObsScaling's *-deduped rows: they are copies of the deduped evaluations, not separate
    # observations. Keep a Sloth addition only if its score vector does not duplicate an ObsScaling row.
    sc = ["mmlu", "arc_c", "hellaswag", "winogrande", "truthfulqa", "gsm8k"]
    okeys = set(map(tuple, o[sc].round(6).itertuples(index=False, name=None)))
    dup = np.array([tuple(r) in okeys for r in ex[sc].round(6).itertuples(index=False, name=None)])
    if verbose and dup.any():
        print(f"[panel] dropping {int(dup.sum())} Sloth rows that duplicate ObsScaling scores: {list(ex.model[dup])}")
    ex = ex[~dup]
    df = pd.concat([o.drop(columns=["FLOPs (1E21)"]), ex], ignore_index=True)

    # fill missing OLLM-v1 scores from Sloth (same leaderboard numbers)
    s_all = pd.read_csv(os.path.join(RAW, "sloth", "data_v2.csv")).set_index("Model")
    for k, v in FILL_FROM_SLOTH.items():
        i = df.model == k
        for col, sc in [("arc_c", "ARC"), ("hellaswag", "HellaSwag"), ("mmlu", "MMLU"), ("winogrande", "Winogrande"),
                        ("gsm8k", "GSM8K"), ("truthfulqa", "TruthfulQA")]:
            if df.loc[i, col].isna().all():
                df.loc[i, col] = s_all.loc[v, sc]

    # corrections
    df["fix_note"] = ""
    for k, v in FIXES.items():
        i = df.model == k
        if "N" in v:
            df.loc[i, "N_rep"] = v["N"]
        if "D" in v:
            df.loc[i, "D_rep"] = v["D"]
        df.loc[i, "fix_note"] = v["note"]

    # family metadata
    meta = pd.DataFrame.from_dict(FAM, orient="index",
                                  columns=["developer", "country", "line", "gen", "arch", "domain", "special"])
    df = df.join(meta, on="family")
    for k, g in GEN_OVERRIDE.items():
        df.loc[df.model == k, "gen"] = g
    df.loc[df.model.isin(["google/gemma-2-27b"]), "special"] = ""       # 27B trained from scratch (Gemma 2 report)
    df.loc[df.model.isin(["01-ai/Yi-9B"]), "special"] = "continued"
    df["distilled"] = df.model.isin(DISTILLED)
    df["moe_active"] = df.model.isin(MOE_ACTIVE)

    # HF metadata: exact parameter counts and upload dates
    h = _hf_meta()
    df = df.merge(h, on="model", how="left")
    use_exact = df.N_exact.notna() & ~df.moe_active & (df.arch == "dense")
    df["N"] = np.where(use_exact, df.N_exact, df.N_rep * 1e9)
    df["N_source"] = np.where(use_exact, "HF safetensors", "reported")
    df["D"] = df.D_rep * 1e12
    df["C"] = 6 * df.N * df.D
    df["n"], df["d"], df["c"] = np.log(df.N), np.log(df.D), np.log(df.C)
    df["n_rep"] = np.log(df.N_rep * 1e9)
    df["M"] = df.D / df.N

    # Epoch join
    e = pd.read_csv(os.path.join(RAW, "epoch_models", "all_ai_models.csv"), low_memory=False)
    e = e.drop_duplicates("Model")
    ecols = {"Model": "epoch_name", "Publication date": "epoch_date", "Organization": "epoch_org",
             "Training compute (FLOP)": "C_epoch", "Confidence": "epoch_conf",
             "Training compute estimation method": "C_method", "Training hardware": "hardware",
             "Notability criteria": "notability", "Parameters": "N_epoch",
             "Training dataset size (total)": "D_epoch", "Hardware quantity": "hw_qty",
             "Training time (hours)": "train_hours", "Hardware utilization (MFU)": "mfu"}
    e = e[list(ecols)].rename(columns=ecols)
    df["epoch_name"] = df.model.map(EPOCH_NAME)
    df = df.merge(e, on="epoch_name", how="left")
    df["in_epoch"] = df.epoch_date.notna()
    df["notable"] = df.notability.notna()

    # release date: Epoch publication date > manual override > HF upload date
    d0 = pd.to_datetime(df.epoch_date, errors="coerce")
    d1 = pd.to_datetime(df.model.map(DATE_OVERRIDE), errors="coerce")
    d2 = pd.to_datetime(df.hf_created, errors="coerce")
    df["date"] = d0.fillna(d1).fillna(d2)
    df["date_source"] = np.where(d0.notna(), "Epoch", np.where(d1.notna(), "manual", np.where(d2.notna(), "HF upload", "")))
    df["year"] = df.date.dt.year
    df["t"] = (df.date - pd.Timestamp("2023-01-01")).dt.days / 365.25
    df["half"] = df.date.dt.year.astype("Int64").astype(str) + "H" + np.where(df.date.dt.month <= 6, "1", "2")

    # instruments: frontier price-performance at training date; own-hardware price-performance
    df["z_frontier"] = frontier_ppd(df.date, lag_months=6)
    df["z_own"] = own_hardware_ppd(df.hardware)
    df["china_post"] = ((df.country == "China") & (df.date >= "2023-10-17")).astype(int)  # Oct-2023 export-control update

    df["tier"] = size_tier(df.N / 1e9)
    df = add_outputs(df)
    # sample flags
    df["has_nd"] = df.N.notna() & df.D.notna()
    df["main"] = df.has_nd & (df.arch == "dense") & df.y_core.notna()
    # overlap with the experimental support (N <= 14B params, D <= 5T tokens; see experiments.py)
    df["overlap"] = df.main & (df.N <= 14e9) & (df.D <= 5.0e12)
    df = df.sort_values(["developer", "family", "N"]).reset_index(drop=True)
    df.to_csv(os.path.join(PROC, "obs_panel.csv"), index=False)
    if verbose:
        print(f"[panel] rows={len(df)}  with N,D={df.has_nd.sum()}  main(dense, core outputs)={df.main.sum()}  "
              f"overlap={df.overlap.sum()}  families(main)={df[df.main].family.nunique()}  "
              f"developers(main)={df[df.main].developer.nunique()}  in Epoch={df.in_epoch.sum()}  notable={df.notable.sum()}")
    return df


def write_hf_ids():
    """List of HF ids whose metadata the download script fetches (run once before download_m4_observational.sh)."""
    o = pd.read_csv(os.path.join(RAW, "obsscaling", "base_llm_benchmark_eval.csv"))
    ids = list(o.Model) + [v[0] for v in SLOTH_EXTRA.values()]
    with open(os.path.join(PROC, "hf_ids.txt"), "w") as f:
        f.write("\n".join(dict.fromkeys(ids)) + "\n")


if __name__ == "__main__":
    build_panel()
