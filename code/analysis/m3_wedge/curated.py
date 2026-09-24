"""curated.py -- hand-curated inputs for the verified open-weight choices sample (Sample B).

Sample B = pretrained decoder-only open-weight models whose training-token count D is documented by the developer
(model card or technical report) and cross-checked against at least one other source (Epoch AI, ObsScaling, Sloth).
It is built from three blocks:
  (1) ObsScaling base models (Ruan et al. 2024) with the corrections in OBS_FIXES;
  (2) Sloth additions (Maia Polo et al.): the standard (non-deduplicated) Pythia suite;
  (3) CURATED: families released after ObsScaling (mid-2024 to 2026) or missing from it.

Every D in CURATED carries `src` and, where the model card states it, `card_re`: a regular expression that must match
the saved model card (data/raw/m3_hf/cards/<id>.md). build_choices.py checks every regex and records D_card_verified.
Units: D = tokens processed in pretraining (incl. repeated epochs, incl. mid-training/annealing stages of the same
run); N = total parameters of the language model (exact safetensors float count where available; MoE: total and
active kept separately; the wedge uses ACTIVE parameters, which enter both 6ND and 2NT).
"""

# ----------------------------------------------------------------------------- (1) ObsScaling corrections
# id -> dict(N=params in B, D=tokens in T, note, drop=True to exclude)
OBS_FIXES = {
    # corrections documented by module m4_observational (panel.FIXES), re-checked here
    "cerebras/btlm-3b-8k-base": dict(D=0.627, note="ObsScaling lists 627T; BTLM trained on 627B tokens (Epoch 6.27e11)"),
    "openlm-research/open_llama_7b_v2": dict(N=7.0, note="ObsScaling lists 13B; a 7B model"),
    "openlm-research/open_llama_13b": dict(D=1.0, note="ObsScaling 0.6T; final OpenLLaMA-13B trained on 1T (Epoch 1e12)"),
    "google/gemma-2b": dict(D=3.0, note="ObsScaling/card: 6T total dataset; Gemma report and Epoch: 2B trained on 3T"),
    "Qwen/Qwen2-72B": dict(D=7.0, note="missing in ObsScaling; Qwen2 report and Epoch: 7T"),
    "Qwen/Qwen2-7B": dict(D=7.0, note="missing in ObsScaling; Qwen2 report and Epoch: 7T"),
    "Qwen/Qwen2-1.5B": dict(D=7.0, note="missing in ObsScaling; Qwen2 report and Epoch: 7T"),
    "Qwen/Qwen2-0.5B": dict(D=12.0, note="missing in ObsScaling; Qwen2 report and Epoch: 12T"),
    "RWKV/rwkv-4-169m-pile": dict(D=0.33, note="Pile, 330B tokens (Sloth; Epoch RWKV-4 14B)"),
    "RWKV/rwkv-4-430m-pile": dict(D=0.33, note="Pile, 330B tokens"),
    "RWKV/rwkv-4-1b5-pile": dict(D=0.33, note="Pile, 330B tokens"),
    "RWKV/rwkv-4-3b-pile": dict(D=0.33, note="Pile, 330B tokens"),
    "RWKV/rwkv-4-7b-pile": dict(D=0.33, note="Pile, 330B tokens"),
    "RWKV/rwkv-4-14b-pile": dict(D=0.33, note="Pile, 330B tokens"),
    # new in m3
    "TinyLlama/TinyLlama_v1.1": dict(D=2.0, note="ObsScaling 3T is TinyLlama v1.0; the v1.1 card states 2T (1.5T + 0.5T stages)"),
    "meta-llama/Meta-Llama-3.1-405B-FP8": dict(D=15.6, note="405B: 15.6T per Llama 3 herd report (card: ~15T for all 3.1 sizes)"),
    **{f"facebook/opt-{m}": dict(D=0.3, note="ObsScaling 0.18T is the corpus size; OPT models were trained for ~300B tokens "
                                              "(1.67 epochs; OPT logbook, Epoch 3.0e11)")
       for m in ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b"]},
    # duplicates / out of scope
    "RWKV/rwkv-raven-14b": dict(drop="instruction-tuned RWKV (derived model)"),
    "01-ai/Yi-6B-200K": dict(drop="long-context continuation of Yi-6B (same pretraining choice)"),
    "01-ai/Yi-34B-200K": dict(drop="long-context continuation of Yi-34B (same pretraining choice)"),
    "Qwen/Qwen2-57B-A14B": dict(drop="MoE upcycled from a dense Qwen2 model; D for the MoE stage only"),
    "Qwen/Qwen1.5-110B": dict(drop="D not disclosed"),
    "mistralai/Mistral-7B-v0.1": dict(drop="D not disclosed"),
    "mistralai/Mixtral-8x7B-v0.1": dict(drop="D not disclosed"),
    "mistralai/Mixtral-8x22B-v0.1": dict(drop="D not disclosed"),
    "mistralai/Mistral-Nemo-Base-2407": dict(drop="D not disclosed"),
    "ai21labs/Jamba-v0.1": dict(drop="D not disclosed"),
    "openai-community/gpt2": dict(drop="D not disclosed in tokens (WebText ~40GB, epochs unknown)"),
    "openai-community/gpt2-medium": dict(drop="D not disclosed in tokens"),
    "openai-community/gpt2-large": dict(drop="D not disclosed in tokens"),
    "openai-community/gpt2-xl": dict(drop="D not disclosed in tokens"),
    "internlm/internlm2-20b": dict(drop="D ambiguous (report: 2.0-2.6T across sizes; Sloth 2.15T; Epoch 2.6T)"),
    "internlm/internlm2-7b": dict(drop="D ambiguous (report: 2.0-2.6T across sizes; Sloth 2.15T)"),
    "Deci/DeciLM-7B": dict(drop="D not disclosed"),
}
# Pythia: ObsScaling carries the deduplicated suite; Sloth adds the standard suite. Both saw 299.9B tokens (the
# deduplicated runs repeat 1.5 epochs of 207B). We keep the standard suite and drop the deduplicated duplicates.
PYTHIA_STD = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]

# MoE active parameters (B) for ObsScaling rows (ObsScaling mixes total and active)
OBS_MOE = {
    "deepseek-ai/DeepSeek-V2": dict(N_active=21.0, N_total=236.0),
    "deepseek-ai/deepseek-moe-16b-base": dict(N_active=2.8, N_total=16.4),
    "deepseek-ai/DeepSeek-Coder-V2-Base": dict(N_active=21.0, N_total=236.0),
}

# family metadata for ObsScaling families: family -> (lab, domain, special)
#   domain: general | code ; special: '' | continued | synthetic | distilled
OBS_FAMILY = {
    "Llama": ("Meta", "general", ""), "Llama-2": ("Meta", "general", ""), "Llama-3": ("Meta", "general", ""),
    "Llama-3.1": ("Meta", "general", ""), "CodeLlama": ("Meta", "code", "continued"), "OPT": ("Meta", "general", ""),
    "XGLM": ("Meta", "general", ""), "Qwen": ("Alibaba", "general", ""), "Qwen1.5": ("Alibaba", "general", ""),
    "Qwen2": ("Alibaba", "general", ""), "DeepSeek-LLM": ("DeepSeek", "general", ""),
    "DeepSeek-MoE": ("DeepSeek", "general", ""), "DeepSeek-V2": ("DeepSeek", "general", ""),
    "DeepSeek-Coder": ("DeepSeek", "code", ""), "DeepSeek-Coder-V2": ("DeepSeek", "code", "continued"),
    "Yi": ("01.AI", "general", ""), "Yi-1.5": ("01.AI", "general", "continued"), "Gemma": ("Google", "general", ""),
    "Gemma-2": ("Google", "general", "distilled"), "Falcon": ("TII", "general", ""),
    "Phi": ("Microsoft", "general", "synthetic"), "Pythia": ("EleutherAI", "general", ""),
    "GPT-Neo/J": ("EleutherAI", "general", ""), "BLOOM": ("BigScience", "general", ""), "MPT": ("MosaicML", "general", ""),
    "StarCoder": ("BigCode", "code", ""), "StarCoder2": ("BigCode", "code", ""),
    "OpenLlama": ("OpenLM Research", "general", ""), "OpenLlamaV2": ("OpenLM Research", "general", ""),
    "StableLM": ("Stability AI", "general", ""), "RWKV": ("RWKV", "general", ""),
    "RedPajama-INCITE-Base": ("Together", "general", ""), "Amber": ("LLM360", "general", ""),
    "Codegen": ("Salesforce", "general", ""), "SmolLM": ("Hugging Face", "general", "synthetic"),
    "Cerebras-GPT": ("Cerebras", "general", ""), "BTLM": ("Cerebras", "general", ""),
    "H2O-Danube": ("H2O.ai", "general", ""), "OLMo": ("AI2", "general", ""), "TinyLlama": ("TinyLlama", "general", ""),
}
# model-level overrides of the family 'special' flag (Gemma 2: only the 2B and 9B were distilled; the 27B was
# trained from scratch -- Gemma 2 report)
SPECIAL_OVERRIDE = {"google/gemma-2-27b": ""}
# models whose own family label should be split into release generations (for within-family analysis)
FAMILY_SPLIT = {
    "stabilityai/stablelm-base-alpha-3b": "StableLM-alpha", "stabilityai/stablelm-base-alpha-7b": "StableLM-alpha",
    "stabilityai/stablelm-base-alpha-7b-v2": "StableLM-alpha-v2", "stabilityai/stablelm-2-1_6b": "StableLM-2",
    "h2oai/h2o-danube-1.8b-base": "H2O-Danube1", "h2oai/h2o-danube2-1.8b-base": "H2O-Danube2",
    "h2oai/h2o-danube3-500m-base": "H2O-Danube3", "h2oai/h2o-danube3-4b-base": "H2O-Danube3",
    "allenai/OLMo-1B-hf": "OLMo-1", "allenai/OLMo-7B-hf": "OLMo-1", "allenai/OLMo-7B-0424-hf": "OLMo-1.7",
    "tiiuae/falcon-rw-1b": "Falcon-RW", "microsoft/phi-1_5": "Phi-1.5", "microsoft/phi-2": "Phi-2",
    "openlm-research/open_llama_3b_v2": "OpenLlamaV2", "openlm-research/open_llama_7b_v2": "OpenLlamaV2",
    "Salesforce/codegen-6B-nl": "Codegen-NL", "Salesforce/codegen-16B-nl": "Codegen-NL",
}

# ----------------------------------------------------------------------------- (3) curated additions
# Keys: hf (id), family, lab, D (T tokens), src, card_re (regex on the saved card; None if the card is silent),
#       N (B; only when the checkpoint count is not the LM count), N_active (B, MoE), arch, special, base_released,
#       date (release; used only when neither Epoch nor HF upload date applies), mm (multimodal checkpoint), note
_Q25 = "Qwen2.5 technical report (18T tokens pre-training corpus); Epoch 1.8e13"
_Q3 = "Qwen3 card and technical report: pre-trained on 36T tokens"
_G3 = "Gemma 3 card: 27B 14T, 12B 12T, 4B 4T, 1B 2T, 270M 6T; distillation for all sizes (Gemma 3 report)"
CURATED = [
    # --- Meta
    dict(hf="meta-llama/Llama-3.2-1B", family="Llama-3.2", lab="Meta", D=9.0, src="card: up to 9T tokens",
         card_re=r"up to 9 trillion tokens", special="distilled",
         note="pruned from Llama 3.1 8B and trained with 3.1 8B/70B logits (card)"),
    dict(hf="meta-llama/Llama-3.2-3B", family="Llama-3.2", lab="Meta", D=9.0, src="card: up to 9T tokens",
         card_re=r"up to 9 trillion tokens", special="distilled",
         note="pruned from Llama 3.1 8B and trained with 3.1 8B/70B logits (card)"),
    dict(hf="meta-llama/Llama-4-Scout-17B-16E", family="Llama-4", lab="Meta", D=40.0, src="card: ~40T tokens",
         card_re=r"Scout was pretrained on \\?~40 trillion tokens", arch="moe", N_active=17.0, mm=True,
         note="MoE 17B active / 109B total; multimodal tokens"),
    dict(hf="meta-llama/Llama-4-Maverick-17B-128E", family="Llama-4", lab="Meta", D=22.0, src="card: ~22T tokens",
         card_re=r"Maverick was pretrained on \\?~22 trillion tokens", arch="moe", N_active=17.0, mm=True,
         special="distilled", note="MoE 17B active / 400B total; codistilled from Behemoth (Meta blog)"),
    # --- Qwen 2.5 (all sizes share the 18T corpus)
    *[dict(hf=f"Qwen/Qwen2.5-{s}", family="Qwen2.5", lab="Alibaba", D=18.0, src=_Q25, card_re=None)
      for s in ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]],
    # --- Qwen 3 (dense bases released up to 14B; 32B and 235B-A22B released post-trained only)
    *[dict(hf=f"Qwen/Qwen3-{s}-Base", family="Qwen3", lab="Alibaba", D=36.0, src=_Q3,
           card_re=r"pre-trained on 36 trillion tokens") for s in ["0.6B", "1.7B", "4B", "8B", "14B"]],
    dict(hf="Qwen/Qwen3-32B", family="Qwen3", lab="Alibaba", D=36.0, src=_Q3, card_re=None, base_released=False),
    dict(hf="Qwen/Qwen3-30B-A3B-Base", family="Qwen3-MoE", lab="Alibaba", D=36.0, src=_Q3,
         card_re=r"pre-trained on 36 trillion tokens", arch="moe", N_active=3.3, note="MoE 3.3B active / 30.5B total"),
    dict(hf="Qwen/Qwen3-235B-A22B", family="Qwen3-MoE", lab="Alibaba", D=36.0, src=_Q3, card_re=None, arch="moe",
         N_active=22.0, base_released=False, note="MoE 22B active / 235B total"),
    # --- Google Gemma 3 (all distilled; 4B+ checkpoints include a 417M SigLIP vision encoder)
    dict(hf="google/gemma-3-270m", family="Gemma-3", lab="Google", D=6.0, src=_G3,
         card_re=r"270M with 6 trillion tokens", special="distilled"),
    dict(hf="google/gemma-3-1b-pt", family="Gemma-3", lab="Google", date="2025-03-12", D=2.0, src=_G3,
         card_re=r"1B with 2 trillion tokens", special="distilled"),
    dict(hf="google/gemma-3-4b-pt", family="Gemma-3", lab="Google", date="2025-03-12", D=4.0, src=_G3,
         card_re=r"4B model was trained with 4 trillion tokens", special="distilled", mm=True, vision_B=0.417),
    dict(hf="google/gemma-3-12b-pt", family="Gemma-3", lab="Google", date="2025-03-12", D=12.0, src=_G3,
         card_re=r"12B model was trained with 12 trillion tokens", special="distilled", mm=True, vision_B=0.417),
    dict(hf="google/gemma-3-27b-pt", family="Gemma-3", lab="Google", date="2025-03-12", D=14.0, src=_G3,
         card_re=r"27B model was trained with 14 trillion tokens", special="distilled", mm=True, vision_B=0.417),
    # --- AI2 OLMo 2 / OLMoE / Olmo 3 (D = stage 1 + stage 2; stage-2 soups: mean ingredient length)
    dict(hf="allenai/OLMo-2-0425-1B", family="OLMo-2", lab="AI2", D=4.05, src="card: stage 1 4T + stage 2 50B",
         card_re=r"OLMo 2-1B\]\(https://huggingface.co/allenai/OLMo-2-0425-1B\) \| 4 Trillion"),
    dict(hf="allenai/OLMo-2-1124-7B", family="OLMo-2", lab="AI2", D=4.05, src="card: stage 1 4T + stage 2 50B (3-run soup)",
         card_re=r"4 trillion tokens<br>\(1 epoch\)"),
    dict(hf="allenai/OLMo-2-1124-13B", family="OLMo-2", lab="AI2", D=5.15,
         src="card: stage 1 5T (1.2 epochs) + stage 2 100B x3 / 300B x1 soup (mean 150B)",
         card_re=r"5 trillion tokens<br>\(1.2 epochs\)"),
    dict(hf="allenai/OLMo-2-0325-32B", family="OLMo-2", lab="AI2", D=6.15,
         src="card: stage 1 6T (1.5 epochs) + stage 2 100B x3 / 300B x1 soup (mean 150B)",
         card_re=r"6 trillion tokens<br>\(1.5 epoch\)"),
    dict(hf="allenai/OLMoE-1B-7B-0924", family="OLMoE", lab="AI2", D=5.13, src="card: 5,033B pretraining + 100B annealing",
         card_re=r"tokens5033B", arch="moe", N_active=1.3, note="MoE 1.3B active / 6.9B total"),
    dict(hf="allenai/Olmo-3-1025-7B", family="Olmo-3", lab="AI2", D=6.03, src="card: stage 1 5.93T + stage 2 100B",
         card_re=r"5\.93T tokens"),
    dict(hf="allenai/Olmo-3-1125-32B", family="Olmo-3", lab="AI2", D=5.60, src="card: stage 1 5.50T + stage 2 100B",
         card_re=r"5\.50T tokens"),
    # --- Hugging Face SmolLM2 / SmolLM3
    dict(hf="HuggingFaceTB/SmolLM2-135M", family="SmolLM2", lab="Hugging Face", D=2.0, src="card: 2T",
         card_re=r"trained on 2 trillion tokens", special="synthetic"),
    dict(hf="HuggingFaceTB/SmolLM2-360M", family="SmolLM2", lab="Hugging Face", D=4.0, src="card: 4T",
         card_re=r"trained on 4 trillion tokens", special="synthetic"),
    dict(hf="HuggingFaceTB/SmolLM2-1.7B", family="SmolLM2", lab="Hugging Face", D=11.0, src="card: 11T",
         card_re=r"trained on 11 trillion tokens", special="synthetic"),
    dict(hf="HuggingFaceTB/SmolLM3-3B-Base", family="SmolLM3", lab="Hugging Face", D=11.2, src="card: 11.2T",
         card_re=r"pretrained on 11\.2T tokens", special="synthetic"),
    # --- Microsoft Phi-3 / 3.5 / 4 (released post-trained only; heavy synthetic data)
    dict(hf="microsoft/Phi-3-mini-4k-instruct", family="Phi-3", lab="Microsoft", D=4.9, src="card (June 2024 update): 4.9T",
         card_re=r"Training data: 4\.9T tokens", special="synthetic", base_released=False),
    dict(hf="microsoft/Phi-3-small-8k-instruct", family="Phi-3", lab="Microsoft", D=4.8, src="card: 4.8T",
         card_re=r"Training data: 4\.8T tokens", special="synthetic", base_released=False),
    dict(hf="microsoft/Phi-3-medium-4k-instruct", family="Phi-3", lab="Microsoft", D=4.8, src="card: 4.8T",
         card_re=r"Training data: 4\.8T tokens", special="synthetic", base_released=False),
    dict(hf="microsoft/Phi-3.5-mini-instruct", family="Phi-3.5", lab="Microsoft", D=3.4, src="card: 3.4T",
         card_re=r"Training data:\*\* 3\.4T tokens", special="synthetic", base_released=False),
    dict(hf="microsoft/Phi-3.5-MoE-instruct", family="Phi-3.5", lab="Microsoft", D=4.9, src="card: 4.9T",
         card_re=r"Training data:\*\* 4\.9T tokens", special="synthetic", base_released=False, arch="moe", N_active=6.6,
         note="MoE 6.6B active / 41.9B total"),
    dict(hf="microsoft/phi-4", family="Phi-4", lab="Microsoft", D=9.8, src="card: 9.8T", card_re=r"9\.8T tokens",
         special="synthetic", base_released=False),
    dict(hf="microsoft/Phi-4-mini-instruct", family="Phi-4", lab="Microsoft", D=5.0, src="card: 5T",
         card_re=r"Training data:\*\* 5T tokens", special="synthetic", base_released=False),
    # --- DeepSeek MoE generation 2/3, Moonshot, Zhipu
    dict(hf="deepseek-ai/DeepSeek-V2-Lite", family="DeepSeek-V2", lab="DeepSeek", D=5.7, src="card: 5.7T",
         card_re=r"5\.7T tokens", arch="moe", N_active=2.4, note="MoE 2.4B active / 15.7B total"),
    dict(hf="deepseek-ai/DeepSeek-V3-Base", family="DeepSeek-V3", lab="DeepSeek", D=14.8, src="card: 14.8T",
         card_re=r"on 14\.8T tokens", arch="moe", N_active=37.0, note="MoE 37B active / 671B total"),
    dict(hf="moonshotai/Kimi-K2-Base", family="Kimi-K2", lab="Moonshot", D=15.5, src="card: 15.5T",
         card_re=r"on 15\.5T tokens", arch="moe", N_active=32.0, note="MoE 32B active / 1.04T total"),
    dict(hf="zai-org/GLM-4.5-Base", family="GLM-4.5", lab="Zhipu", D=23.0, src="GLM-4.5 report via Epoch (2.31e13)",
         card_re=None, arch="moe", N_active=32.0, note="MoE 32B active / 355B total; card silent on D"),
    # --- others with documented D
    dict(hf="tiiuae/Falcon3-7B-Base", family="Falcon3", lab="TII", D=14.0, src="card: 14 teratokens",
         card_re=r"Pretrained on 14 Teratokens"),
    dict(hf="tiiuae/falcon-11B", family="Falcon2", lab="TII", D=5.5, src="Epoch 5.5e12; card: over 5,000B tokens",
         card_re=r"over 5,000B tokens"),
    dict(hf="ibm-granite/granite-3.0-2b-base", family="Granite-3.0", lab="IBM", D=12.0, src="card: 10T + 2T",
         card_re=r"# Training tokens \| \*\*12T\*\* \| 12T"),
    dict(hf="ibm-granite/granite-3.0-8b-base", family="Granite-3.0", lab="IBM", D=12.0, src="card: 10T + 2T",
         card_re=r"# Training tokens \| 12T \| \*\*12T\*\*"),
    dict(hf="swiss-ai/Apertus-8B-2509", family="Apertus", lab="Swiss AI", D=15.0, src="card: 15T",
         card_re=r"pretrained on 15T tokens"),
    dict(hf="swiss-ai/Apertus-70B-2509", family="Apertus", lab="Swiss AI", D=15.0, src="card: 15T",
         card_re=r"pretrained on 15T tokens"),
    dict(hf="marin-community/marin-8b-base", family="Marin", lab="Marin", D=12.7, src="card: 12.7T",
         card_re=r"\| 12\.7T"),
    dict(hf="apple/DCLM-7B", family="DCLM", lab="Apple", D=2.5, src="card: 2.5T", card_re=r"Total Training Tokens:\*\* 2\.5T"),
    dict(hf="m-a-p/neo_7b", family="MAP-Neo", lab="M-A-P", D=4.42, src="card: 3.7T + 720B decay",
         card_re=r"3\.7T tokens were learned"),
    dict(hf="LLM360/K2", family="LLM360-K2", lab="LLM360", D=1.4, src="card: 1.4T", card_re=r"Tokens: 1\.4T"),
    dict(hf="stabilityai/stablelm-3b-4e1t", family="StableLM-3B-4E1T", lab="Stability AI", D=4.0,
         src="card: 1T tokens x 4 epochs", card_re=r"1 trillion tokens of diverse English and code datasets for 4 epochs"),
]

# official post-trained counterparts (usage validation aggregates base + official instruct/chat downloads)
INSTRUCT = {
    "meta-llama/Llama-2-7b-hf": ["meta-llama/Llama-2-7b-chat-hf"], "meta-llama/Llama-2-13b-hf": ["meta-llama/Llama-2-13b-chat-hf"],
    "meta-llama/Llama-2-70b-hf": ["meta-llama/Llama-2-70b-chat-hf"],
    "meta-llama/Meta-Llama-3-8B": ["meta-llama/Meta-Llama-3-8B-Instruct"],
    "meta-llama/Meta-Llama-3-70B": ["meta-llama/Meta-Llama-3-70B-Instruct"],
    "meta-llama/Llama-3.1-8B": ["meta-llama/Llama-3.1-8B-Instruct"],
    "meta-llama/Llama-3.1-70B": ["meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"],
    "meta-llama/Llama-3.1-405B": ["meta-llama/Llama-3.1-405B-Instruct", "meta-llama/Llama-3.1-405B-Instruct-FP8", "meta-llama/Llama-3.1-405B-FP8"],
    "meta-llama/Llama-3.2-1B": ["meta-llama/Llama-3.2-1B-Instruct"], "meta-llama/Llama-3.2-3B": ["meta-llama/Llama-3.2-3B-Instruct"],
    "meta-llama/Llama-4-Scout-17B-16E": ["meta-llama/Llama-4-Scout-17B-16E-Instruct"],
    "meta-llama/Llama-4-Maverick-17B-128E": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"],
    "google/gemma-2b": ["google/gemma-2b-it", "google/gemma-1.1-2b-it"], "google/gemma-7b": ["google/gemma-7b-it", "google/gemma-1.1-7b-it"],
    "google/gemma-2-2b": ["google/gemma-2-2b-it"], "google/gemma-2-9b": ["google/gemma-2-9b-it"],
    "google/gemma-2-27b": ["google/gemma-2-27b-it"],
    "google/gemma-3-270m": ["google/gemma-3-270m-it"], "google/gemma-3-1b-pt": ["google/gemma-3-1b-it"],
    "google/gemma-3-4b-pt": ["google/gemma-3-4b-it"], "google/gemma-3-12b-pt": ["google/gemma-3-12b-it"],
    "google/gemma-3-27b-pt": ["google/gemma-3-27b-it"],
    "Qwen/Qwen3-0.6B-Base": ["Qwen/Qwen3-0.6B"], "Qwen/Qwen3-1.7B-Base": ["Qwen/Qwen3-1.7B"], "Qwen/Qwen3-4B-Base": ["Qwen/Qwen3-4B"],
    "Qwen/Qwen3-8B-Base": ["Qwen/Qwen3-8B"], "Qwen/Qwen3-14B-Base": ["Qwen/Qwen3-14B"], "Qwen/Qwen3-30B-A3B-Base": ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507"],
    "Qwen/Qwen3-235B-A22B": ["Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507"],
    "Qwen/Qwen-7B": ["Qwen/Qwen-7B-Chat"], "Qwen/Qwen-14B": ["Qwen/Qwen-14B-Chat"], "Qwen/Qwen-72B": ["Qwen/Qwen-72B-Chat"],
    "allenai/OLMo-2-0425-1B": ["allenai/OLMo-2-0425-1B-Instruct"], "allenai/OLMo-2-1124-7B": ["allenai/OLMo-2-1124-7B-Instruct"],
    "allenai/OLMo-2-1124-13B": ["allenai/OLMo-2-1124-13B-Instruct"], "allenai/OLMo-2-0325-32B": ["allenai/OLMo-2-0325-32B-Instruct"],
    "allenai/OLMoE-1B-7B-0924": ["allenai/OLMoE-1B-7B-0924-Instruct"],
    "allenai/Olmo-3-1025-7B": ["allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-7B-Think"],
    "allenai/Olmo-3-1125-32B": ["allenai/Olmo-3-32B-Think", "allenai/Olmo-3.1-32B-Instruct"],
    "allenai/OLMo-7B-hf": ["allenai/OLMo-7B-Instruct-hf"], "allenai/OLMo-7B-0424-hf": ["allenai/OLMo-7B-0424-Instruct-hf"],
    "HuggingFaceTB/SmolLM-135M": ["HuggingFaceTB/SmolLM-135M-Instruct"], "HuggingFaceTB/SmolLM-360M": ["HuggingFaceTB/SmolLM-360M-Instruct"],
    "HuggingFaceTB/SmolLM-1.7B": ["HuggingFaceTB/SmolLM-1.7B-Instruct"],
    "HuggingFaceTB/SmolLM2-135M": ["HuggingFaceTB/SmolLM2-135M-Instruct"], "HuggingFaceTB/SmolLM2-360M": ["HuggingFaceTB/SmolLM2-360M-Instruct"],
    "HuggingFaceTB/SmolLM2-1.7B": ["HuggingFaceTB/SmolLM2-1.7B-Instruct"], "HuggingFaceTB/SmolLM3-3B-Base": ["HuggingFaceTB/SmolLM3-3B"],
    "deepseek-ai/deepseek-llm-7b-base": ["deepseek-ai/deepseek-llm-7b-chat"], "deepseek-ai/deepseek-llm-67b-base": ["deepseek-ai/deepseek-llm-67b-chat"],
    "deepseek-ai/deepseek-moe-16b-base": ["deepseek-ai/deepseek-moe-16b-chat"], "deepseek-ai/DeepSeek-V2": ["deepseek-ai/DeepSeek-V2-Chat"],
    "deepseek-ai/DeepSeek-V2-Lite": ["deepseek-ai/DeepSeek-V2-Lite-Chat"], "deepseek-ai/DeepSeek-V3-Base": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-0528"],
    "moonshotai/Kimi-K2-Base": ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-0905"], "zai-org/GLM-4.5-Base": ["zai-org/GLM-4.5"],
    "01-ai/Yi-6B": ["01-ai/Yi-6B-Chat"], "01-ai/Yi-34B": ["01-ai/Yi-34B-Chat"], "01-ai/Yi-1.5-6B": ["01-ai/Yi-1.5-6B-Chat"],
    "01-ai/Yi-1.5-9B": ["01-ai/Yi-1.5-9B-Chat"], "01-ai/Yi-1.5-34B": ["01-ai/Yi-1.5-34B-Chat"],
    "tiiuae/falcon-7b": ["tiiuae/falcon-7b-instruct"], "tiiuae/falcon-40b": ["tiiuae/falcon-40b-instruct"],
    "tiiuae/falcon-180B": ["tiiuae/falcon-180B-chat"], "tiiuae/Falcon3-7B-Base": ["tiiuae/Falcon3-7B-Instruct"],
    "mosaicml/mpt-7b": ["mosaicml/mpt-7b-instruct", "mosaicml/mpt-7b-chat"], "mosaicml/mpt-30b": ["mosaicml/mpt-30b-instruct", "mosaicml/mpt-30b-chat"],
    "ibm-granite/granite-3.0-2b-base": ["ibm-granite/granite-3.0-2b-instruct"], "ibm-granite/granite-3.0-8b-base": ["ibm-granite/granite-3.0-8b-instruct"],
    "swiss-ai/Apertus-8B-2509": ["swiss-ai/Apertus-8B-Instruct-2509"], "swiss-ai/Apertus-70B-2509": ["swiss-ai/Apertus-70B-Instruct-2509"],
    "marin-community/marin-8b-base": ["marin-community/marin-8b-instruct"], "LLM360/K2": ["LLM360/K2-Chat"],
    "stabilityai/stablelm-2-1_6b": ["stabilityai/stablelm-2-1_6b-chat", "stabilityai/stablelm-2-zephyr-1_6b"],
    "h2oai/h2o-danube3-4b-base": ["h2oai/h2o-danube3-4b-chat"], "h2oai/h2o-danube3-500m-base": ["h2oai/h2o-danube3-500m-chat"],
    "h2oai/h2o-danube2-1.8b-base": ["h2oai/h2o-danube2-1.8b-chat"], "h2oai/h2o-danube-1.8b-base": ["h2oai/h2o-danube-1.8b-chat"],
    "TinyLlama/TinyLlama_v1.1": [], "LLM360/Amber": ["LLM360/AmberChat"],
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": ["togethercomputer/RedPajama-INCITE-7B-Chat", "togethercomputer/RedPajama-INCITE-7B-Instruct"],
    "togethercomputer/RedPajama-INCITE-Base-3B-v1": ["togethercomputer/RedPajama-INCITE-Chat-3B-v1", "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"],
}
# Qwen 1.5 / 2 / 2.5: official post-trained repos follow "<base>-Chat" (1.5) and "<base>-Instruct" (2, 2.5)
for _s in ["0.5B", "1.8B", "4B", "7B", "14B", "32B", "72B"]:
    INSTRUCT[f"Qwen/Qwen1.5-{_s}"] = [f"Qwen/Qwen1.5-{_s}-Chat"]
for _s in ["0.5B", "1.5B", "7B", "72B"]:
    INSTRUCT[f"Qwen/Qwen2-{_s}"] = [f"Qwen/Qwen2-{_s}-Instruct"]
for _s in ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]:
    INSTRUCT[f"Qwen/Qwen2.5-{_s}"] = [f"Qwen/Qwen2.5-{_s}-Instruct"]

# Official post-trained releases include later post-training updates of the same pretrained base by the same lab
# (Llama 3.3 70B on Llama 3.1 70B; Qwen3 -2507 updates; DeepSeek-V3-0324 and R1 on V3-Base; Kimi-K2-0905), because
# their usage is usage of the same pretraining choice. Models with additional pretraining (e.g. DeepSeek-V3.1) are not.
# Hub aliases used only when matching OpenRouter's hugging_face_id (the Hub redirects them to the same repository,
# so they must NOT be added to INSTRUCT: downloads would be double counted)
OR_ALIAS = {"meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct"}

# ObsScaling ids that were renamed/moved on the Hub: ObsScaling id -> current canonical id (HF API lookups)
CANONICAL = {
    "meta-llama/Meta-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B": "meta-llama/Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B-FP8": "meta-llama/Llama-3.1-405B",
}

# ungated mirrors used only to read config.json (vocab size, width, tied embeddings) of gated repos
CONFIG_MIRROR = {
    "meta-llama/Llama-2-7b-hf": "NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf": "NousResearch/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf": "NousResearch/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B": "NousResearch/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B": "NousResearch/Meta-Llama-3-70B",
    "meta-llama/Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B", "meta-llama/Llama-3.1-70B": "unsloth/Meta-Llama-3.1-70B",
    "meta-llama/Llama-3.1-405B": "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B", "meta-llama/Llama-3.2-3B": "unsloth/Llama-3.2-3B",
    "google/gemma-2b": "unsloth/gemma-2b", "google/gemma-7b": "unsloth/gemma-7b", "google/gemma-2-2b": "unsloth/gemma-2-2b",
    "google/gemma-2-9b": "unsloth/gemma-2-9b", "google/gemma-2-27b": "unsloth/gemma-2-27b",
    "google/gemma-3-270m": "unsloth/gemma-3-270m", "google/gemma-3-1b-pt": "unsloth/gemma-3-1b-pt",
    "google/gemma-3-4b-pt": "unsloth/gemma-3-4b-pt", "google/gemma-3-12b-pt": "unsloth/gemma-3-12b-pt",
    "google/gemma-3-27b-pt": "unsloth/gemma-3-27b-pt",
}

# HF id -> Epoch 'Model' name (for release date, organization, confidence and the D cross-check)
EPOCH_NAME = {
    "meta-llama/Llama-2-7b-hf": "Llama 2-7B", "meta-llama/Llama-2-13b-hf": "Llama 2-13B", "meta-llama/Llama-2-70b-hf": "Llama 2-70B",
    "huggyllama/llama-7b": "LLaMA-7B", "huggyllama/llama-13b": "LLaMA-13B", "huggyllama/llama-30b": "LLaMA-33B",
    "huggyllama/llama-65b": "LLaMA-65B", "meta-llama/Meta-Llama-3-70B": "Llama 3-70B", "meta-llama/Meta-Llama-3-8B": "Llama 3-8B",
    "meta-llama/Meta-Llama-3.1-405B-FP8": "Llama 3.1-405B", "meta-llama/Meta-Llama-3.1-70B": "Llama 3.1-70B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama 3.1-8B", "meta-llama/Llama-3.2-1B": "Llama 3.2 1B", "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
    "meta-llama/Llama-4-Scout-17B-16E": "Llama 4 Scout", "meta-llama/Llama-4-Maverick-17B-128E": "Llama 4 Maverick",
    "Qwen/Qwen1.5-72B": "Qwen1.5-72B", "Qwen/Qwen1.5-14B": "Qwen1.5-14B", "Qwen/Qwen1.5-7B": "Qwen1.5-7B",
    "Qwen/Qwen-72B": "Qwen-72B", "Qwen/Qwen-14B": "Qwen-14B", "Qwen/Qwen-7B": "Qwen-7B", "Qwen/Qwen2-72B": "Qwen2-72B",
    "Qwen/Qwen2-7B": "Qwen2-7B", "Qwen/Qwen2-1.5B": "Qwen2-1.5B", "Qwen/Qwen2-0.5B": "Qwen2-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B", "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B": "Qwen2.5-14B", "Qwen/Qwen2.5-32B": "Qwen2.5-32B", "Qwen/Qwen2.5-72B": "Qwen2.5-72B",
    "Qwen/Qwen3-0.6B-Base": "Qwen3-0.6B", "Qwen/Qwen3-1.7B-Base": "Qwen3-1.7B", "Qwen/Qwen3-4B-Base": "Qwen3-4B",
    "Qwen/Qwen3-8B-Base": "Qwen3-8B", "Qwen/Qwen3-14B-Base": "Qwen3-14B", "Qwen/Qwen3-32B": "Qwen3-32B",
    "Qwen/Qwen3-30B-A3B-Base": "Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V2": "DeepSeek-V2 (MoE-236B)", "deepseek-ai/DeepSeek-V3-Base": "DeepSeek-V3",
    "01-ai/Yi-34B": "Yi-34B", "01-ai/Yi-6B": "Yi 6B", "01-ai/Yi-1.5-34B": "Yi-1.5-34B", "01-ai/Yi-1.5-9B": "Yi-1.5-9B",
    "google/gemma-7b": "Gemma 7B", "google/gemma-2b": "Gemma 2B", "google/gemma-2-27b": "Gemma 2 27B",
    "google/gemma-2-9b": "Gemma 2 9B", "google/gemma-2-2b": "Gemma 2 2B", "google/gemma-3-270m": "Gemma 3 270M",
    "tiiuae/falcon-180B": "Falcon-180B", "tiiuae/falcon-40b": "Falcon-40B", "tiiuae/falcon-7b": "Falcon-7B",
    "tiiuae/falcon-11B": "Falcon 2 11B", "tiiuae/Falcon3-7B-Base": "Falcon3-7B",
    "microsoft/phi-2": "Phi-2", "microsoft/phi-1_5": "Phi-1.5", "microsoft/phi-4": "Phi-4", "microsoft/Phi-4-mini-instruct": "Phi-4 Mini",
    "microsoft/Phi-3-mini-4k-instruct": "phi-3-mini 3.8B", "microsoft/Phi-3-small-8k-instruct": "phi-3-small 7.4B",
    "microsoft/Phi-3-medium-4k-instruct": "phi-3-medium 14B", "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
    "microsoft/Phi-3.5-MoE-instruct": "Phi-3.5-MoE",
    **{f"EleutherAI/pythia-{m}": f"Pythia-{m}" for m in PYTHIA_STD},
    "bigscience/bloom-560m": "BLOOM-560M", "bigscience/bloom-1b1": "BLOOM-1B", "bigscience/bloom-3b": "BLOOM-3B",
    "bigscience/bloom-7b1": "BLOOM-7.1B", "bigscience/bloom": "BLOOM-176B", "EleutherAI/gpt-neox-20b": "GPT-NeoX-20B",
    "EleutherAI/gpt-neo-2.7B": "GPT-Neo-2.7B", "EleutherAI/gpt-j-6b": "GPT-J-6B", "facebook/opt-6.7b": "OPT-6.7B",
    "facebook/opt-1.3b": "OPT-1.3B", "facebook/opt-350m": "OPT-350M", "facebook/opt-2.7b": "OPT-2.7B",
    "facebook/opt-30b": "OPT-30B", "facebook/opt-66b": "OPT-66B", "mosaicml/mpt-30b": "MPT-30B", "mosaicml/mpt-7b": "MPT-7B",
    "facebook/xglm-7.5B": "XGLM-7.5B", "openlm-research/open_llama_13b": "OpenLLaMA-13B",
    "deepseek-ai/deepseek-llm-67b-base": "DeepSeek LLM 67B", "deepseek-ai/deepseek-llm-7b-base": "DeepSeek LLM 7B",
    "deepseek-ai/deepseek-moe-16b-base": "DeepSeekMoE-16B", "stabilityai/stablelm-base-alpha-7b-v2": "StableLM-Base-Alpha-7B",
    "stabilityai/stablelm-2-1_6b": "StableLM-2-1.6B", "stabilityai/stablelm-3b-4e1t": "StableLM-3B-4E1T",
    "RWKV/rwkv-4-14b-pile": "RWKV-4 14B", "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": "RedPajama-INCITE-7B-Base",
    "LLM360/Amber": "Amber", "HuggingFaceTB/SmolLM-1.7B": "SmolLM-1.7B", "cerebras/btlm-3b-8k-base": "BTLM-3B",
    "allenai/OLMo-1B-hf": "OLMo-1B", "allenai/OLMo-7B-hf": "OLMo-7B", "allenai/OLMo-7B-0424-hf": "OLMo 1.7-7B",
    "allenai/OLMo-2-1124-7B": "OLMo 2 Furious 7B", "allenai/OLMo-2-1124-13B": "OLMo 2 Furious 13B",
    "allenai/OLMo-2-0325-32B": "OLMo 2 32B", "allenai/OLMoE-1B-7B-0924": "OLMoE", "allenai/Olmo-3-1125-32B": "Olmo 3",
    "ibm-granite/granite-3.0-8b-base": "Granite 3.0 8B", "ibm-granite/granite-3.0-2b-base": "Granite 3.0 2B",
    "swiss-ai/Apertus-70B-2509": "Apertus 70B", "marin-community/marin-8b-base": "Marin 8B", "apple/DCLM-7B": "DCLM 7B",
    "m-a-p/neo_7b": "MAP-Neo", "moonshotai/Kimi-K2-Base": "Kimi K2", "zai-org/GLM-4.5-Base": "GLM-4.5",
    "TinyLlama/TinyLlama_v1.1": None,
}
