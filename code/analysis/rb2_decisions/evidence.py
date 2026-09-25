"""evidence.py -- hand-coded, source-verified evidence for (1) the reading of each common-D family's token budget and
(2) the model-level serving code. Every quoted passage is checked against the primary source saved by fetch_sources.py
(data/raw/rb2_decisions/<key>.txt; match after removing all whitespace, case-insensitive). run.py fails if a quote is
not found, so the CSVs cannot drift from the sources.

(1) FAMILY READINGS (R2 Major 2; R1 New 1; R3 N3). For each family whose members share one token budget
    (max/min D <= 1.10 in the clean sample) the reading of the common D:
      cap      -- the report says every size was trained on the whole curated corpus and gives evidence that the corpus
                  binds (more data would still help, or data had to be expanded synthetically). Under a binding D cap
                  (Prop. A8 with a data cap) w = (1 + m_N)/(1 + mu), mu >= 0, so a member's w - 1 is a LOWER bound on
                  its m_N and each member's N is its own decision given the cap;
      choice   -- the budget is stated as chosen (trade-off, a selection from a larger pool, a budget that differs
                  across model classes of the same data); the family condition of Prop. A9 applies and only the
                  pi-weighted family share is revealed; member wedges are not revealed preferences;
      menu     -- sizes are stated to be set by a hardware or configuration menu; by Prop. A9(iv)/A8(iii)(b) the
                  family share is only an UPPER bound;
      cap+menu -- both margins constrained: neither bound direction holds; the family is uninformative about m_N.
(2) MODEL-LEVEL SERVING (R1 New 1(d); R3 N4(a)). serve_model = 1 if a first-party managed API (priced per token) or a
    first-party consumer product offered THIS size (any post-trained variant of the same pretraining run) within 180
    days of the base release, documented by a dated first-party page, an archived catalog (Wayback), or dated press.
    Self-deploy catalogs (the customer rents the accelerators: Vertex AI Model Garden, Azure AI Studio catalog) and
    free research demos (Hugging Face Spaces, Ai2 Playground) are coded separately and are 0 in the primary code.
    'ambiguous' = the developer's product used the family but the size is undisclosed (primary 0; sensitivity 1).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

import rb2common as C
from sources import SOURCES

_TXT = {}


def _norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def source_text(key):
    if key not in _TXT:
        p = os.path.join(C.RAW, f"{key}.txt")
        _TXT[key] = _norm(open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
    return _TXT[key]


def verify(key, quote):
    t = source_text(key)
    if t is None:
        return False
    return _norm(quote) in t


def manifest():
    p = os.path.join(C.RAW, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ============================================================================ (1) family readings
FAMILY = {
    "Llama-3 herd": dict(
        reading="cap", D_margin="cap (whole curated corpus)", N_margin="free (flagship sized by Meta's law for its compute)",
        evidence=[
            ("llama3_report", "We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2."),
            ("llama3_report", "While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal."),
            ("llama3_blog", "Both our 8B and 70B parameter models continued to improve log-linearly after we trained them on up to 15T tokens."),
        ],
        note="Every size trained on the whole ~15T corpus while still improving: the corpus binds for the 8B and 70B. "
             "The 405B was sized by Meta's law for its compute (D = 15.6T). Member wedges are lower bounds on m_N."),
    "Qwen3": dict(
        reading="cap", D_margin="cap (whole corpus; expanded with PDF extraction and synthetic data)", N_margin="unstated",
        evidence=[
            ("qwen3_report", "All Qwen3 models are trained on a large and diverse dataset consisting of 119 languages and dialects, with a total of 36 trillion tokens."),
            ("qwen3_report", "all Qwen3 models are trained on over 30 trillion tokens using a sequence length of 4,096 tokens."),
            ("qwen3_report", "To efficiently expand the training data, we employ a multi-modal approach: Qwen2.5-VL (Bai et al., 2025) is finetuned to extract text from extensive PDF documents."),
        ],
        note="Every size trained on the whole corpus; the corpus had to be expanded (PDF extraction, synthetic data): "
             "evidence that data bind. Member wedges are lower bounds on m_N."),
    "Qwen2.5": dict(
        reading="cap", D_margin="cap (corpus size stated for every member; tokens processed per member not stated)", N_margin="unstated",
        evidence=[
            ("qwen25_blog", "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens."),
            ("qwen25_report", "we have developed a larger and higher-quality pre-training dataset, expanding from the 7 trillion tokens used in Qwen2 (Yang et al., 2024a) to 18 trillion tokens."),
        ],
        note="Weak cap reading: the corpus (up to 18T) is stated for every size, but the tokens each member processed are "
             "not reported (R1 minor 16; R2 minor 16). Flagged D_unaudited."),
    "DeepSeek-LLM": dict(
        reading="cap+menu", D_margin="cap (the dataset at the time)", N_margin="menu (two prevalent open-source configurations)",
        evidence=[
            ("deepseek_llm_report", "we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding."),
            ("deepseek_llm_report", "in two prevalent used open-source configurations, 7B and 67B."),
        ],
        note="Both margins constrained: sizes are standard configurations and D is the whole dataset. Neither bound holds."),
    "Yi": dict(
        reading="cap+menu", D_margin="cap (whole constructed corpus; not saturated)", N_margin="menu (34B chosen to fit 24G GPU memory)",
        evidence=[
            ("yi_report", "when choosing model scale, the desiderata is to have small enough model that is feasible for inference on consumer-grade hardware like the RTX 4090 where the bounding factor is its limited 24G memory"),
            ("yi_report", "Our model is trained on 3.1T tokens, yet we belive with larger amount of data, we can continue improve the model performance (i.e., the model have not saturated at 3.1T)"),
            ("yi_report", "we overtrain the model on more tokens (3T) than the compute optimal (around 1T). The benefit is from the inference side, as we achieve stronger performance with reduced serving cost"),
        ],
        note="Size set by a memory tier and D by the corpus; the report states an inference motive, but the wedge bounds "
             "nothing in either direction."),
    "MPT": dict(
        reading="menu", D_margin="chosen (1T sampled from a larger mix, to match LLaMA)", N_margin="menu (single-GPU deployment)",
        evidence=[
            ("mpt30b_blog", "The size of MPT-30B was also specifically chosen to make it easy to deploy on a single GPU\u2014either 1x NVIDIA A100-80GB in 16-bit precision or 1x NVIDIA A100-40GB in 8-bit precision."),
            ("mpt7b_blog", "Trained on a large amount of data (1T tokens like LLaMA vs. 300B for Pythia, 300B for OpenLLaMA, and 800B for StableLM)."),
            ("mpt7b_blog", "the model was pre-trained on 1T tokens sampled according to this mix."),
        ],
        note="Sizes from a serving-hardware menu; the family share is an upper bound (Prop. A9(iv))."),
    "Llama-2": dict(
        reading="choice", D_margin="chosen (performance-cost trade-off)", N_margin="unstated",
        evidence=[
            ("llama2_report", "We trained on 2 trillion tokens of data as this provides a good performance–cost trade-off, up-sampling the most factual sources in an effort to increase knowledge and dampen hallucinations."),
            ("llama2_report", "We observe that after pretraining on 2T Tokens, the models still did not show any sign of saturation."),
        ],
        note="D chosen for the family; only the family share is revealed (Prop. A9)."),
    "Granite-3.0": dict(
        reading="choice", D_margin="chosen (dense 12T; MoE siblings 10T of the same data)", N_margin="unstated",
        evidence=[
            ("granite30_report", "Dense Models: 2B and 8B parameter models, trained on 12 trillion tokens in total."),
            ("granite30_report", "Mixture-of-Expert (MoE) Models: Sparse 1B and 3B MoE models, with 400M and 800M activated parameters respectively, trained on 10 trillion tokens in total."),
        ],
        note="No statement that the corpus was exhausted; token budgets differ by model class within the release."),
    "Apertus": dict(
        reading="choice", D_margin="chosen (15T in five stages)", N_margin="unstated",
        evidence=[
            ("apertus_report", "The Apertus models are 8B-scale and 70B-scale models (Section 2) pretrained on 15T tokens (Section 3) using up to 4096 GPUs (Section 6)."),
            ("apertus_report", "We train the model on 15T tokens (∼0.3T masked due to Goldfish Loss) divided into five stages"),
        ],
        note="No statement that the compliant corpus was exhausted."),
    "Olmo-3": dict(
        reading="choice", D_margin="chosen (6T mix drawn from a 9T pool; 32B schedule truncated at 5.5T)", N_margin="unstated",
        evidence=[
            ("olmo3_report", "Composition of Dolma 3 Mix including our 9T pool of data, the 6T mix we used for final model training, and the 150B mix we used for experimentation."),
            ("olmo3_report", "The learning rate schedule is a cosine schedule over one epoch (5.93T tokens), truncated at 5.5T tokens."),
        ],
        note="Budget chosen below the pool; the 32B run was cut at 5.5T of a 5.93T schedule (a deadline, not a data cap)."),
    "StableLM-alpha": dict(
        reading="choice", D_margin="chosen (800B of a 1.5T dataset)", N_margin="unstated",
        evidence=[
            ("stablelm_github", "StableLM-Alpha models are trained on a new dataset that builds on [The Pile](https://pile.eleuther.ai/), which contains 1.5 trillion tokens, roughly 3x the size of The Pile."),
            ("stablelm_github", "| 7B | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-7b) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b) | 800B | 7,869,358,080 |"),
        ],
        note="Token count corrected to 800B (ra2 used the 1.5T dataset size); 800B < 1.5T, so no cap."),
}
BOUND = {"cap": "lower bound (members)", "choice": "family share revealed (members not)", "menu": "upper bound",
         "cap+menu": "none (uninformative)"}

# common-D subgroups inside 'size-specific' families (sensitivity: subgroup-level decision units)
SUBGROUP_EVIDENCE = [
    ("Qwen2", "qwen2_report", "All Qwen2 dense models, excluding Qwen2-0.5B, were pre-trained on this large-scale dataset of over 7 trillion tokens."),
    ("Llama", "llama1_report", "LLaMA-33B and LLaMA- 65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens."),
]


def family_readings(Bc, Wf):
    """Bc: clean sample with fam_class; Wf: family-level table (gen -> W_f, s_f). Returns the readings table."""
    man = manifest()
    rows = []
    for gen, g in Bc[Bc["fam_class"] == "common-D"].groupby("gen"):
        spec = FAMILY[gen]
        ev = spec["evidence"]
        ok = [verify(k, q) for k, q in ev]
        if not all(ok):
            bad = [q[:80] for (k, q), o in zip(ev, ok) if not o]
            raise AssertionError(f"quote not found in saved source for {gen}: {bad}")
        wf = Wf.set_index("gen").loc[gen]
        rows.append(dict(
            family=gen, developer=g["dev"].iloc[0], members="; ".join(g.sort_values("N")["model"]), n_members=len(g),
            D_T=", ".join(f"{x/1e12:.2f}" for x in g.sort_values("N")["D"]), D_ratio=g["D"].max() / g["D"].min(),
            reading=spec["reading"], D_margin=spec["D_margin"], N_margin=spec["N_margin"],
            what_is_revealed=BOUND[spec["reading"]], W_family_ref=wf["W_f"], s_family_ref=wf["s_f"],
            **{f"quote_{i+1}": q for i, (k, q) in enumerate(ev)},
            **{f"source_{i+1}": k for i, (k, q) in enumerate(ev)},
            **{f"url_{i+1}": SOURCES[k]["url"] for i, (k, q) in enumerate(ev)},
            **{f"retrieved_{i+1}": man.get(k, {}).get("retrieved", "") for i, (k, q) in enumerate(ev)},
            quotes_verified=True, note=spec["note"]))
    for fam, k, q in SUBGROUP_EVIDENCE:
        if not verify(k, q):
            raise AssertionError(f"subgroup quote not found: {fam}")
    return pd.DataFrame(rows)


# ============================================================================ (2) model-level serving
# Alibaba: archived first-party catalogs (Wayback captures) listing the open-weight sizes as priced API models.
_Q1 = "wb_dashscope_qwen_20240202"
_Q2 = "wb_dashscope_qwen_20240717"
_Q25 = "wb_modelstudio_zh_20240930"
_Q3 = "wb_modelstudio_zh_20250429"
ALI = {
    "Qwen-7B": (_Q1, "2024-02-02", "qwen-7b-chat 通义千问对外开源的7B规模参数量的经过人类指令对齐的chat模型"),
    "Qwen-14B": (_Q1, "2024-02-02", "qwen-14b-chat 通义千问对外开源的14B规模参数量的经过人类指令对齐的chat模型"),
    "Qwen-72B": (_Q1, "2024-02-02", "qwen-72b-chat 通义千问对外开源的72B规模参数量的经过人类指令对齐的chat模型"),
    **{m: (_Q2, "2024-07-17", "灵积平台上基于Qwen2开源的0.5B、1.5B、7B、72B和57B-A14B MoE模型的instruct版本，并进行了针对性的推理性能优化，为广大开发者提供便捷的API服务")
       for m in ("Qwen2-0.5B", "Qwen2-1.5B", "Qwen2-7B", "Qwen2-72B")},
    "Qwen2.5-72B": (_Q25, "2024-09-30", "qwen2.5-72b-instruct 131,072 129,024 8,192 0.004元 0.012元"),
    "Qwen2.5-32B": (_Q25, "2024-09-30", "qwen2.5-32b-instruct 0.0035元 0.007元"),
    "Qwen2.5-14B": (_Q25, "2024-09-30", "qwen2.5-14b-instruct 0.002元 0.006元"),
    "Qwen2.5-7B": (_Q25, "2024-09-30", "qwen2.5-7b-instruct 0.001元 0.002元"),
    "Qwen2.5-3B": (_Q25, "2024-09-30", "qwen2.5-3b-instruct 32,768 30,720 限时免费"),
    "Qwen2.5-1.5B": (_Q25, "2024-09-30", "限时免费 qwen2.5-1.5b-instruct qwen2.5-0.5b-instruct"),
    "Qwen2.5-0.5B": (_Q25, "2024-09-30", "限时免费 qwen2.5-1.5b-instruct qwen2.5-0.5b-instruct"),
    "Qwen3-14B-Base": (_Q3, "2025-04-29", "qwen3-14b 非思考 - 0.001 元 0.004 元"),
    "Qwen3-8B-Base": (_Q3, "2025-04-29", "qwen3-8b 非思考 - 0.0005 元 0.002 元"),
    "Qwen3-4B-Base": (_Q3, "2025-04-29", "qwen3-4b 非思考 - 0.0003 元 0.0012 元"),
    "Qwen3-1.7B-Base": (_Q3, "2025-04-29", "qwen3-1.7b 非思考 32,768 30,720 - 0.0012 元"),
    "Qwen3-0.6B-Base": (_Q3, "2025-04-29", "qwen3-0.6b 非思考 30,720 - 0.0012 元"),
}
ALI_BILLING = (_Q1, "我们将根据模型输入和输出的token数量计费")

# (serve_model, channel, demo, source key, evidence date, quote, ambiguous, note)
S = {}
for m, (k, dt, q) in ALI.items():
    S[m] = (1, "managed API (per-token, Alibaba Cloud DashScope / Model Studio)", 0, k, dt, q, False,
            "archived first-party catalog (Wayback capture date = evidence date)")
S.update({
    "Meta-Llama-3-8B": (0, "consumer product uses the family; size undisclosed", 0, "llama3_blog", "2024-04-18",
                        "Meta AI, built with Llama 3 technology, is now one of the world’s leading AI assistants", True,
                        "Meta AI 'built with Llama 3 technology'; no first-party statement that the 8B served (Llama API launched Apr 2025)"),
    "Meta-Llama-3-70B": (0, "consumer product uses the family; size undisclosed", 0, "llama3_blog", "2024-04-18",
                         "Meta AI, built with Llama 3 technology, is now one of the world’s leading AI assistants", True,
                         "as for the 8B"),
    "Llama-3.1-405B": (1, "consumer product (Meta AI on WhatsApp and meta.ai)", 0, "llama31_blog", "2024-07-23",
                       "Try Llama 3.1 405B in the US on WhatsApp and at meta.ai by asking a challenging math or coding question.", False, ""),
    "Llama-2-7b-hf": (0, "none documented (Meta AI launched later on a custom model)", 0, "meta_ai_launch_2023", "2023-09-27",
                      "It’s powered by a custom model that leverages technology from Llama 2 and our latest large language model (LLM) research.", False,
                      "Meta AI launched 71 days after Llama 2 on a custom model, not a released size"),
    "llama-7b": (0, "none (research release)", 0, "llama1_blog", "2023-02-24",
                 "we are releasing our model under a noncommercial license focused on research use cases.", False, ""),
    "deepseek-llm-67b-base": (1, "consumer product (DeepSeek Chat)", 0, "vb_deepseek_chat_2023", "2023-12-01",
                              "Only the 67B version is available through this interface.", False,
                              "dated press report (VentureBeat) two days after the weights release (2023-11-29; the sample date is the report date, 2024-01-05); the API (deepseek-chat) does not name the size"),
    "deepseek-llm-7b-base": (0, "none documented (the chat product served the 67B only)", 0, "vb_deepseek_chat_2023", "2023-12-01",
                             "Only the 67B version is available through this interface.", False, ""),
    "gemma-2b": (0, "self-deploy catalog (Vertex AI, GKE)", 0, "gemma_blog", "2024-02-21",
                 "Pre-trained and instruction-tuned Gemma models can run on your laptop, workstation, or Google Cloud with easy deployment on Vertex AI and Google Kubernetes Engine (GKE).", False,
                 "the customer rents the accelerators; Google does not serve Gemma 1 per token"),
    "gemma-2-27b": (1, "first-party product (Google AI Studio)", 0, "gemma2_blog", "2024-06-27",
                    "Gemma 2 is now available in Google AI Studio, so you can test out its full performance capabilities at 27B without hardware requirements.", False, ""),
    "granite-3.0-2b-base": (1, "managed API (IBM watsonx)", 0, "granite30_announcement", "2024-10-21",
                            "Granite 3.0 8B Instruct and Granite 3.0 2B Instruct, as well as both Guardian 3.0 safety models, are available today for commercial use on the IBM watsonx platform", False, ""),
    "mpt-7b": (1, "managed API (MosaicML Inference, hosted endpoints)", 0, "mpt7b_blog", "2023-05-05",
               "Start with our managed endpoints for models like MPT-7B-Instruct", False, ""),
    "mpt-30b": (1, "managed API (MosaicML Inference, per-1K-token pricing)", 0, "mpt30b_blog", "2023-06-22",
                "Talk to our hosted endpoints for MPT-30B-Instruct (and MPT-7B-Instruct) using our Python API, with standard pricing per-1K-tokens.", False, ""),
    "phi-1_5": (0, "none (research release)", 0, "phi15_card", "2023-09-11",
                "The intention behind crafting this open-source model is to provide the research community with a non-restricted small model", False, ""),
    "phi-2": (0, "self-deploy catalog (Azure AI Studio model catalog)", 0, "phi2_blog", "2023-12-12",
              "We have made Phi-2 (opens in new tab) available in the Azure AI Studio model catalog to foster research and development on language models.", False,
              "catalog deployment on customer compute; Phi models-as-a-service (per token) began with Phi-3"),
    "Yi-34B": (0, "none documented at release (API early access to some applicants; size unnamed)", 0, "yi_readme", "2023-12-15",
               "Yi APIs (Yi official) - [Early access has been granted](https://x.com/01AI_Yi/status/1735728934560600536?s=20) to some applicants.", True,
               "01.AI's API platform opened in 2024; the early-access model is not named"),
    "Yi-6B": (0, "none documented", 0, "yi_readme", "2023-12-15",
              "Yi APIs (Yi official) - [Early access has been granted](https://x.com/01AI_Yi/status/1735728934560600536?s=20) to some applicants.", False, ""),
    "OLMo-2-1124-13B": (0, "research demo (Ai2 Playground)", 1, "olmo2_blog", "2024-11-26",
                        "Play with OLMo 2-Instruct-13B, our most capable OLMo 2 model, in the Ai2 playground.", False, ""),
    "OLMo-2-1124-7B": (0, "none at release (Ai2 Playground from Mar 2025)", 0, "olmo2_blog", "2024-11-26",
                       "Play with OLMo 2-Instruct-13B, our most capable OLMo 2 model, in the Ai2 playground.", False, ""),
    "OLMo-2-0325-32B": (0, "research demo (Ai2 Playground)", 1, "olmo2_32b_blog", "2025-03-13",
                        "all models are available on the Ai2 playground", False, ""),
    "Olmo-3-1025-7B": (0, "research demo (Ai2 Playground); third-party API (OpenRouter)", 1, "olmo3_blog", "2025-11-20",
                       "Try Olmo 3 on the Ai2 Playground | Use Olmo 3 via OpenRouter", False, "API courtesy of inference partners (third party)"),
    "Olmo-3-1125-32B": (0, "research demo (Ai2 Playground); third-party API (OpenRouter)", 1, "olmo3_blog", "2025-11-20",
                        "Try Olmo 3 on the Ai2 Playground | Use Olmo 3 via OpenRouter", False, "API courtesy of inference partners (third party)"),
    "SmolLM2-1.7B": (0, "none (on-device target)", 0, "smollm2_card", "2024-10-31",
                     "They are capable of solving a wide range of tasks while being lightweight enough to run on-device.", False, ""),
    "SmolLM-1.7B": (0, "none (local/on-device target)", 0, "smollm_blog", "2024-07-16",
                    "Our models are designed to be small and can run locally on various hardware configurations.", False, ""),
    "h2o-danube3-4b-base": (0, "none (on-device target; runs in H2O's on-device app)", 0, "danube3_card", "2024-07-04",
                            "Can be run natively and fully offline on phones", False, ""),
    "falcon-180B": (0, "research demo (Hugging Face Space)", 1, "falcon180b_blog", "2023-09-06",
                    "interact with the model on the Falcon Chat Demo Space", False, "TII operated no first-party API"),
    "stablelm-base-alpha-7b": (0, "research demo (Hugging Face Space)", 1, "stablelm_github", "2023-04-20",
                               "Try to chat with our 7B model, `StableLM-Tuned-Alpha-7B`, on [Hugging Face Spaces]", False, ""),
    "btlm-3b-8k-base": (0, "none (edge/mobile target)", 0, "btlm_blog", "2023-09-20",
                        "BTLM fits in mobile and edge devices with as little as 3GB of memory", False, ""),
    "marin-8b-base": (0, "none (research release)", 0, "marin_card", "2025-05-19",
                      "You can use Marin with the standard HuggingFace Transformers library", False, ""),
    "Apertus-8B-2509": (0, "none (self-deployment; hosting by third parties)", 0, "apertus_card", "2025-09-02",
                        "Deployment of the models is directly supported by the newest versions of [Transformers]", False, ""),
})
# developer-level defaults for the remaining models (same rule; the developer operated no first-party per-token API or
# product serving these sizes at their release). Source = the model's release page.
DEFAULT = {
    "Meta": "llama-7b", "Microsoft": "phi-1_5", "AI2": None, "Hugging Face": "SmolLM-1.7B", "H2O.ai": "h2o-danube3-4b-base",
    "TII": None, "Stability AI": None, "Swiss AI": "Apertus-8B-2509", "Cerebras": "btlm-3b-8k-base", "Marin": "marin-8b-base",
    "M-A-P": None, "TinyLlama": None, "01.AI": "Yi-6B", "Google": "gemma-2b",
}
FAMILY_SHARE = {  # models not listed above inherit the entry of a sibling of the same release
    "Llama-2-13b-hf": "Llama-2-7b-hf", "Llama-2-70b-hf": "Llama-2-7b-hf",
    "llama-13b": "llama-7b", "llama-30b": "llama-7b", "llama-65b": "llama-7b",
    "gemma-7b": "gemma-2b", "granite-3.0-8b-base": "granite-3.0-2b-base",
    "SmolLM2-135M": "SmolLM2-1.7B", "SmolLM2-360M": "SmolLM2-1.7B", "SmolLM-135M": "SmolLM-1.7B",
    "SmolLM-360M": "SmolLM-1.7B", "h2o-danube3-500m-base": "h2o-danube3-4b-base", "Apertus-70B-2509": "Apertus-8B-2509",
}
NONE_ROWS = {  # developers with no first-party serving and no better release quote in hand (ra2 coding source)
    "AI2": "non-profit research institute; models released on Hugging Face (research demo only, coded separately)",
    "TII": "no first-party commercial LLM API; Falcon served by third parties",
    "Stability AI": "LLM releases not offered through a first-party commercial API",
    "M-A-P": "open research community release", "TinyLlama": "academic project release",
    "H2O.ai": "enterprise software vendor; Danube positioned for on-device use",
    "Hugging Face": "hub/platform; SmolLM not served as a first-party product",
}


def serving_table(Bc):
    man = manifest()
    rows = []
    for _, r in Bc.iterrows():
        m = r["model"]
        origin = "model"
        key = m if m in S else FAMILY_SHARE.get(m)
        if key is not None and key != m:
            origin = "same release"
        if key is None and r["dev"] in DEFAULT and DEFAULT[r["dev"]] is not None:
            key, origin = DEFAULT[r["dev"]], "same developer"
        if key is not None:
            sv, ch, demo, src, dt, q, amb, note = S[key]
            if key != m:
                note = (note + "; " if note else "") + f"evidence from the {origin} ({key})"
                demo = 0
                if origin == "same developer":
                    dt = ""          # no dated offer of this model: the lag is undefined
            if not verify(src, q):
                raise AssertionError(f"serving quote not found for {m}: {src}: {q[:70]}")
            url = SOURCES[src]["url"]
            retrieved = man.get(src, {}).get("retrieved", "")
        else:
            sv, ch, demo, src, dt, q, amb, origin = 0, "none documented", 0, "", "", "", False, "developer coding (ra2)"
            note = NONE_ROWS.get(r["dev"], "")
            url = retrieved = ""
        rel = pd.Timestamp(r["date"]).normalize()
        lag = (pd.Timestamp(dt) - rel).days if dt else np.nan
        rows.append(dict(model=m, uid=r["uid"], developer=r["dev"], release_date=rel.date().isoformat(), N_B=r["N"] / 1e9,
                         serve_developer=int(r["serve"]), serve_model=int(sv), serve_ambiguous=bool(amb),
                         serve_model_hi=int(sv or amb), channel=ch, demo=int(demo), evidence_origin=origin, evidence_date=dt,
                         lag_days=lag, source_key=src, url=url, retrieved=retrieved, quote=q,
                         quote_verified=bool(src) and verify(src, q) if src else False, note=note))
    out = pd.DataFrame(rows)
    # evidence must be within the release window (180 days) for serve_model = 1
    bad = out[(out["serve_model"] == 1) & ((out["lag_days"] > 180) | (out["lag_days"] < -60))]
    assert bad.empty, bad[["model", "lag_days"]]
    return out
