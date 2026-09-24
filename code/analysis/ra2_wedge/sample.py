"""sample.py -- the clean inference-demand sample, the family-level token budget classification, and the conduct
covariates (developer serving footprint, stated deployment target), with sources.

Starting point: m3_wedge Sample B core (173 general-purpose open-weight base models with documented N and D;
data/processed/m3_wedge/wedge_models_wide.csv). The clean sample is defined EX ANTE by the sequential steps below
(revision_plan A; R2 Majors 4-5, 10; R1 c1-2, minor 24-25; R3 M5.5-6). Every flag is audited against the model card
(data/raw/ra2_cards, data/raw/m3_hf/cards) or the developer's technical report; the source of each flag is saved in
data/processed/ra2_wedge/audit_flags.csv.

  (a) one observation per pretraining run: continued pretraining / re-releases of an earlier released base with the
      same N are dropped, the first release is kept (Llama 3.1 8B/70B vs Llama 3; Qwen1.5-72B vs Qwen-72B; OpenLlama
      v2 3B/7B vs v1; Yi-1.5 6B/34B continue Yi 6B/34B; H2O-Danube2 continues Danube-1.8B);
  (b) research suites and replications whose D is fixed by research design (common budget or fixed tokens/parameter
      rule set by the study, not by deployment): Pythia, OPT, BLOOM, RWKV-4-Pile, XGLM, Cerebras-GPT, GPT-Neo/J,
      CodeGen-NL, Falcon-RW (b1); LLaMA-recipe replications and dataset-benchmark models: OpenLlama, RedPajama-INCITE,
      LLM360 Amber and K2, DCLM-7B (b2);
  (c) mixture-of-experts (reported separately with [total-N, active-N] bounds);
  (d) distilled, pruned or initialized from another model (teacher/parent compute is an omitted input);
  (e) multimodal pretraining (D includes non-text tokens);
  (f) instruction-tuned-only releases (no base checkpoint released);
  (g) non-transformer or hybrid architectures (different FLOP and serving accounting);
  (h) D not documented in a primary source (Qwen1.5: token counts only in the ObsScaling compilation).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from ra2common import M3_PROC, PROC, RAW, log

# ----------------------------------------------------------------------------- audit (model -> step, reason, source)
STEP_ORDER = ["a_dedupe", "b1_research_suite", "b2_replication", "c_moe", "d_distilled_pruned", "e_multimodal",
              "f_instruct_only", "g_nontransformer", "h_D_undocumented"]
STEP_LABEL = {
    "start": "Verified sample (m3 Sample B core)",
    "a_dedupe": "(a) one observation per pretraining run",
    "b1_research_suite": "(b1) drop research suites (D fixed by design)",
    "b2_replication": "(b2) drop replications / dataset benchmarks",
    "c_moe": "(c) drop mixture-of-experts",
    "d_distilled_pruned": "(d) drop distilled / pruned / derived init.",
    "e_multimodal": "(e) drop multimodal pretraining",
    "f_instruct_only": "(f) drop instruction-tuned-only releases",
    "g_nontransformer": "(g) drop non-transformer / hybrid",
    "h_D_undocumented": "(h) drop D without primary source (= clean sample)",
}

_suite = {
    "Pythia": "Pythia suite: all sizes trained on the same 300B tokens in the same order for research "
              "(biderman2023pythia)",
    "OPT": "OPT suite: all sizes trained for 300B tokens to replicate GPT-3 (zhang2022opt)",
    "BLOOM": "BigScience BLOOM: all sizes trained on ~341B ROOTS tokens (176B: 366B); research release "
             "(hf.co/bigscience/bloom card)",
    "RWKV": "RWKV-4 Pile suite: all sizes trained on the Pile (330B tokens) (hf.co/RWKV/rwkv-4-*-pile cards)",
    "XGLM": "XGLM suite: all sizes trained on 500B tokens (hf.co/facebook/xglm-* cards)",
    "Cerebras-GPT": "Cerebras-GPT: every size at 20 tokens/parameter by design (dey2023cerebras)",
    "GPT-Neo/J": "EleutherAI GPT-Neo/J/NeoX: trained on the Pile to fixed budgets (300-472B) (hf.co/EleutherAI cards)",
    "Codegen-NL": "CodeGen-NL: natural-language stage of the CodeGen code models, all sizes on the Pile "
                  "(hf.co/Salesforce/codegen-*-nl cards)",
    "Falcon-RW": "Falcon-RW-1B: RefinedWeb ablation model trained for 350B tokens (hf.co/tiiuae/falcon-rw-1b card)",
}
_repl = {
    "OpenLlama": "OpenLLaMA: reproduction of LLaMA trained to LLaMA's 1T tokens (hf.co/openlm-research cards)",
    "OpenLlamaV2": "OpenLLaMA v2: reproduction of LLaMA at 1T tokens (hf.co/openlm-research cards)",
    "RedPajama-INCITE-Base": "RedPajama-INCITE: open reproduction of the LLaMA recipe (hf.co/togethercomputer cards)",
    "Amber": "LLM360 Amber: LLaMA-7B replication at 1.26T tokens (hf.co/LLM360/Amber card)",
    "LLM360-K2": "LLM360 K2: LLaMA-65B-scale replication at 1.4T tokens (hf.co/LLM360/K2 card)",
    "DCLM": "DCLM-7B: dataset-benchmark model trained for 2.5T tokens to evaluate DCLM-Baseline (hf.co/apple/DCLM-7B)",
}

AUDIT = {
    # (a) duplicates of a pretraining run
    "Llama-3.1-8B": ("a_dedupe", "continued pretraining (long context) of the Llama 3 8B run; same N and D",
                     "grattafiori2024llama; m3 Table 6 notes"),
    "Llama-3.1-70B": ("a_dedupe", "continued pretraining of the Llama 3 70B run; same N and D", "grattafiori2024llama"),
    "Qwen1.5-72B": ("a_dedupe", "same N and D as Qwen-72B (ObsScaling); not a separate allocation decision",
                    "R2 Major 10"),
    "open_llama_3b_v2": ("a_dedupe", "same N and D as open_llama_3b (v2 retrained on a new mix at the same 1T)",
                         "hf.co/openlm-research/open_llama_3b_v2"),
    "open_llama_7b_v2": ("a_dedupe", "same N and D as open_llama_7b", "hf.co/openlm-research/open_llama_7b_v2"),
    "Yi-1.5-6B": ("a_dedupe", "'continuously pre-trained on Yi with a high-quality corpus of 500B tokens'",
                  "hf.co/01-ai/Yi-1.5-6B card"),
    "Yi-1.5-34B": ("a_dedupe", "'continuously pre-trained on Yi with a high-quality corpus of 500B tokens'",
                   "hf.co/01-ai/Yi-1.5-34B card"),
    "h2o-danube2-1.8b-base": ("a_dedupe", "'The base model was initialized from H2O-Danube-1.8B' (+2T tokens)",
                              "arXiv:2401.16818, Sec. 7"),
    # (d) distilled / pruned / derived initialization
    "gemma-2-2b": ("d_distilled_pruned", "'we train the 2B and 9B models with knowledge distillation'",
                   "gemmateam2024gemma2 (arXiv:2408.00118)"),
    "gemma-2-9b": ("d_distilled_pruned", "'we train the 2B and 9B models with knowledge distillation'",
                   "gemmateam2024gemma2 (arXiv:2408.00118)"),
    "gemma-3-270m": ("d_distilled_pruned", "Gemma 3 family: 'The Gemma 3 models are trained with distillation' "
                     "(270M released later; card silent)", "gemmateam2025gemma3 (arXiv:2503.19786)"),
    "gemma-3-1b-pt": ("d_distilled_pruned", "'The Gemma 3 models are trained with distillation'",
                      "gemmateam2025gemma3 (arXiv:2503.19786)"),
    "gemma-3-4b-pt": ("d_distilled_pruned", "'trained with distillation'; tokens include images",
                      "gemmateam2025gemma3 (arXiv:2503.19786)"),
    "gemma-3-12b-pt": ("d_distilled_pruned", "'trained with distillation'; tokens include images",
                       "gemmateam2025gemma3 (arXiv:2503.19786)"),
    "gemma-3-27b-pt": ("d_distilled_pruned", "'trained with distillation'; tokens include images",
                       "gemmateam2025gemma3 (arXiv:2503.19786)"),
    "Llama-3.2-1B": ("d_distilled_pruned", "pruned from Llama 3.1 8B; logits of 3.1 8B/70B as targets "
                     "('Knowledge distillation was used after pruning')", "hf.co/meta-llama/Llama-3.2-1B card"),
    "Llama-3.2-3B": ("d_distilled_pruned", "pruned from Llama 3.1 8B; logits of 3.1 8B/70B as targets",
                     "hf.co/meta-llama/Llama-3.2-3B card"),
    "Yi-1.5-9B": ("d_distilled_pruned", "continued from Yi-9B, which was depth-upscaled from Yi-6B (derived "
                  "initialization)", "hf.co/01-ai/Yi-1.5-9B card; Yi-9B card"),
    # (f) instruction-tuned-only releases
    "Phi-3-mini-4k-instruct": ("f_instruct_only", "no base checkpoint released", "hf.co/microsoft (m3 base_released)"),
    "Phi-3-small-8k-instruct": ("f_instruct_only", "no base checkpoint released", "hf.co/microsoft"),
    "Phi-3-medium-4k-instruct": ("f_instruct_only", "no base checkpoint released", "hf.co/microsoft"),
    "Phi-3.5-mini-instruct": ("f_instruct_only", "no base checkpoint released", "hf.co/microsoft"),
    "Phi-4-mini-instruct": ("f_instruct_only", "no base checkpoint released", "hf.co/microsoft"),
    "phi-4": ("f_instruct_only", "released checkpoint is post-trained (SFT + DPO); no base", "hf.co/microsoft/phi-4"),
    "Qwen3-32B": ("f_instruct_only", "Qwen3-32B base not released; the checkpoint is post-trained",
                  "hf.co/Qwen/Qwen3-32B card"),
    # (h) D without a primary source
    **{f"Qwen1.5-{s}": ("h_D_undocumented", "token count only in ObsScaling; Qwen1.5 card/blog state none",
                        "hf.co/Qwen/Qwen1.5-* cards (no token statement); R2 Major 10")
       for s in ["0.5B", "1.8B", "4B", "7B", "14B", "32B"]},
}
# multimodal-pretraining flags (applied after (d); only models not already removed are counted in (e))
MULTIMODAL = {
    "gemma-3-4b-pt": "Gemma 3 4B-27B: 'The increase in tokens accounts for the mix of images and text' (arXiv:2503.19786)",
    "gemma-3-12b-pt": "same", "gemma-3-27b-pt": "same",
    "Llama-4-Scout-17B-16E": "Llama 4: early-fusion multimodal pretraining; card: ~40T multimodal tokens",
    "Llama-4-Maverick-17B-128E": "Llama 4: early-fusion multimodal pretraining; card: ~22T multimodal tokens",
}
DISTILLED_EXTRA = {"Llama-4-Maverick-17B-128E": "codistilled from Llama 4 Behemoth (Meta blog)"}
NONTRANSFORMER_RE = r"rwkv|mamba|hawk|griffin|retnet|hgrn|hymba|samba|hybrid h3|nemotron-h|falcon-h1|granite-4\.0-h|jamba|kimi linear"

# family generation (for the common-D classification): families that share one pretraining data release
GENERATION = {"Llama-3": "Llama-3 herd", "Llama-3.1": "Llama-3 herd"}

# ----------------------------------------------------------------------------- conduct covariates
# Developer serving footprint: 1 if, at the model's release, the developer operated a commercial API or consumer
# product that served its own LLMs to third parties at scale (it bears serving cost on its own models).
# (developer, first year with serving=1, source)
SERVE = {
    # Meta: date-level threshold (review fix). Meta AI (first-party assistant serving Llama) launched 2023-09-27, so
    # under the stated rule Llama 1 (Feb 2023) and Llama 2 (Jul 2023) are coded 0; the builder's year-level threshold
    # coded both 1, contradicting the source note and the memo ("Llama 1 ... coded 0").
    "Meta": ("2023-09-27", "Meta AI assistant (launched 2023-09-27) serves Llama; Llama API (Apr 2025). Llama 1 (Feb 2023, "
                           "research) and Llama 2 (Jul 2023, served by partners, not first-party) coded 0"),
    "Alibaba": (2023, "Alibaba Cloud Model Studio / DashScope API serves Qwen (Tongyi Qianwen API, 2023)"),
    "Google": (2023, "Google Cloud Vertex AI / Gemini API; Gemma served on Vertex AI and AI Studio"),
    "Microsoft": (2023, "Azure AI (Foundry) model catalog serves Phi models"),
    "DeepSeek": (2024, "DeepSeek API platform (platform.deepseek.com); DeepSeek LLM released Jan 2024 with chat service"),
    "IBM": (2023, "watsonx.ai (GA Jul 2023) serves Granite"),
    "MosaicML": (2023, "MosaicML Inference API served MPT (2023; Databricks)"),
    "01.AI": (2024, "Yi API platform (2024); Yi-6B/34B released Nov 2023 before the API"),
    "Moonshot": (2023, "Kimi API platform"), "Zhipu": (2023, "BigModel / Z.ai API"),
    "Together": (2023, "Together AI inference cloud serves RedPajama-INCITE and other models"),
    "Mistral AI": (2023, "La Plateforme API"),
}
NO_SERVE = {
    "TII": "no first-party commercial LLM API (Falcon served by third parties)",
    "Hugging Face": "hub/platform; SmolLM not served as a first-party product (third-party hosting only)",
    "H2O.ai": "enterprise software (h2oGPTe); Danube positioned for on-device use",
    "AI2": "non-profit research institute (demo playground only)",
    "Stability AI": "LLM releases not offered as a first-party commercial API (API business: image models)",
    "Swiss AI": "public research consortium (ETH/EPFL/CSCS); hosting by third parties",
    "Marin": "open research lab (Stanford CRFM)", "M-A-P": "open research community",
    "TinyLlama": "academic project (SUTD)", "LLM360": "research release", "OpenLM Research": "research release",
    "EleutherAI": "non-profit research", "BigScience": "research workshop", "Cerebras": "hardware vendor; "
    "Cerebras Inference launched Aug 2024, after its model releases", "RWKV": "open research", "Salesforce":
    "research release", "Apple": "on-device products; DCLM is a research artifact",
}
# Stated deployment target (model level), from the model card: 'ondevice' = the card names on-device, mobile/phone or
# edge deployment as an intended use; 'local' = card states deployment on limited-resource local hardware (laptops,
# desktops); else 'server/unspecified'.
DEPLOY = {
    **{m: ("ondevice", "card: 'lightweight enough to run on-device'") for m in ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]},
    **{m: ("ondevice", "SmolLM blog/card: small models for on-device use (HuggingFaceTB/SmolLM-*)") for m in ["SmolLM-135M", "SmolLM-360M", "SmolLM-1.7B"]},
    "SmolLM3-3B-Base": ("ondevice", "SmolLM3 card: small model family designed for local/on-device deployment"),
    "h2o-danube3-500m-base": ("ondevice", "card: 'Can be run natively and fully offline on phones'"),
    "h2o-danube3-4b-base": ("ondevice", "card: 'Can be run natively and fully offline on phones'"),
    "Llama-3.2-1B": ("ondevice", "card: 'mobile AI powered writing assistants', 'on-device use-cases'"),
    "Llama-3.2-3B": ("ondevice", "card: 'mobile AI powered writing assistants', 'on-device use-cases'"),
    "gemma-3-270m": ("ondevice", "Gemma 3 270M release: compact model for on-device/task-specific use"),
    "Phi-3-mini-4k-instruct": ("ondevice", "card: optimized inference on 'mobile CPUs'"),
    **{m: ("local", "card: 'deploy them in environments with limited resources such as a laptop, desktop'")
       for m in ["gemma-2b", "gemma-7b", "gemma-2-2b", "gemma-2-9b", "gemma-2-27b", "gemma-3-1b-pt", "gemma-3-4b-pt",
                 "gemma-3-12b-pt", "gemma-3-27b-pt"]},
}

DEV_ALIAS = {"Meta AI": "Meta", "Microsoft Research": "Microsoft", "Cerebras Systems": "Cerebras",
             "Allen Institute for AI": "AI2", "Swiss AI Initiative": "Swiss AI", "Zhipu AI": "Zhipu",
             "Z.ai (Zhipu AI)": "Zhipu", "Mistral": "Mistral AI"}


def developer(lab):
    s = str(lab).split(",")[0].strip()
    return DEV_ALIAS.get(s, s)


def serve_flag(dev, year, date=None):
    """1 if the developer served its own LLMs first-party at the model's release. SERVE thresholds are a year (int)
    or a date ('YYYY-MM-DD') compared with the release date."""
    if dev in SERVE:
        y0, src = SERVE[dev]
        if isinstance(y0, str):
            if date is None or pd.isna(date):
                return int(year > int(y0[:4])), src
            return int(pd.Timestamp(date) >= pd.Timestamp(y0)), src
        return int(year >= y0), src
    if dev in NO_SERVE:
        return 0, NO_SERVE[dev]
    return np.nan, "not coded"


# ----------------------------------------------------------------------------- architecture from config.json
def _cfg(hf):
    f = os.path.join(RAW, "m3_hf", "configs", hf.replace("/", "__") + ".json")
    if not os.path.exists(f):
        return {}
    try:
        c = json.load(open(f))
    except Exception:
        return {}
    if "text_config" in c and isinstance(c["text_config"], dict):
        c = {**c, **c["text_config"]}
    return c


# Architectures whose config.json could not be downloaded (gated/authentication errors for MosaicML, Cerebras and
# Falcon-180B): values from the public model cards / config files of the same checkpoints.
ARCH_MANUAL = {
    "mosaicml/mpt-7b": dict(n_layer=32, d_model=4096, vocab=50432, n_heads=32, n_kv_heads=32, tied_cfg=True),
    "mosaicml/mpt-30b": dict(n_layer=48, d_model=7168, vocab=50432, n_heads=64, n_kv_heads=64, tied_cfg=True),
    "cerebras/btlm-3b-8k-base": dict(n_layer=32, d_model=2560, vocab=50257, n_heads=32, n_kv_heads=32, tied_cfg=True),
    "tiiuae/falcon-180B": dict(n_layer=80, d_model=14848, vocab=65024, n_heads=232, n_kv_heads=8, tied_cfg=True),
}


def arch_row(hf):
    if hf in ARCH_MANUAL:
        return dict(ARCH_MANUAL[hf])
    c = _cfg(hf)
    g = lambda *ks: next((c[k] for k in ks if k in c and c[k] is not None), np.nan)
    L = g("num_hidden_layers", "n_layer", "n_layers", "num_layers")
    d = g("hidden_size", "n_embd", "n_embed", "d_model", "dim")
    V = g("vocab_size", "padded_vocab_size")
    heads = g("num_attention_heads", "n_head", "n_heads")
    kv = g("num_key_value_heads", "n_kv_heads", "multi_query_group_num")
    tied = c.get("tie_word_embeddings", c.get("weight_tying", c.get("tied_embeddings", np.nan)))
    return dict(n_layer=L, d_model=d, vocab=V, n_heads=heads, n_kv_heads=kv if np.isfinite(kv) else heads, tied_cfg=tied)


# ----------------------------------------------------------------------------- build
def load_m3():
    d = pd.read_csv(os.path.join(M3_PROC, "wedge_models_wide.csv"))
    # keep m3's inputs only (its wedge columns are recomputed here)
    drop = [c for c in d.columns if c.startswith(("w_", "wlo_", "whi_", "Mx_", "CE_", "band_"))
            or c in ("techs_used", "pt_min", "pt_max", "n_tech", "lnw", "TD", "T", "Mstar_eq3_ownC")]
    d = d.drop(columns=drop)
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    d["Cmp"] = 6 * d["N"] * d["D"]
    for c in ["core", "code", "moe", "open_weights", "distilled", "synthetic", "continued", "mm", "base_released"]:
        d[c] = d[c].fillna(False).astype(bool)
    d["core"] = d["core"] & ~d["code"]
    d["prod"] = d["core"] & (d["Cmp"] >= 1e21)
    return d


def fill_embeddings(B):
    """N_emb / N_nonemb for Sample-B rows where m3 left them missing (configs with MPT/BTLM/BLOOM key names)."""
    B = B.copy()
    for i, r in B.iterrows():
        a = arch_row(r["hf"])
        for k, v in a.items():
            B.loc[i, k] = v
        if (pd.isna(r["N_nonemb"]) or pd.isna(r["N_emb"])) and not r["moe"] and np.isfinite(a["vocab"]) and np.isfinite(a["d_model"]):
            tied = a["tied_cfg"] if isinstance(a["tied_cfg"], bool) else True
            emb = a["vocab"] * a["d_model"] * (1 if tied else 2)
            B.loc[i, "N_emb"] = emb
            B.loc[i, "N_nonemb"] = r["N"] - emb
            B.loc[i, "tied"] = tied
            B.loc[i, "emb_filled"] = True
    B["emb_filled"] = B.get("emb_filled", False)
    B["emb_share"] = 1 - B["N_nonemb"] / B["N"]
    # output-head parameters (FLOP-bearing): vocab x width (tied or not, the head matmul costs 2 V d per token)
    B["N_head"] = B["vocab"] * B["d_model"]
    return B


def build():
    d = load_m3()
    B = d[(d["sample"] == "B") & d["core"]].copy().reset_index(drop=True)
    B = fill_embeddings(B)
    B["dev"] = B["lab"].map(developer)
    # --- flags with sources
    rows = []
    for i, r in B.iterrows():
        fam = r["family"]
        step, reason, src = None, "", ""
        if r["model"] in AUDIT and AUDIT[r["model"]][0] == "a_dedupe":
            step, reason, src = AUDIT[r["model"]]
        elif fam in _suite:
            step, reason, src = "b1_research_suite", _suite[fam], "model card / paper"
        elif fam in _repl:
            step, reason, src = "b2_replication", _repl[fam], "model card"
        elif r["moe"]:
            step, reason, src = "c_moe", f"MoE: {r['N_active']/1e9:.1f}B active / {r['N_total']/1e9:.1f}B total", "m3 (card)"
        elif r["model"] in AUDIT and AUDIT[r["model"]][0] == "d_distilled_pruned":
            step, reason, src = AUDIT[r["model"]]
        elif r["model"] in MULTIMODAL:
            step, reason, src = "e_multimodal", MULTIMODAL[r["model"]], "card / report"
        elif r["model"] in AUDIT and AUDIT[r["model"]][0] == "f_instruct_only":
            step, reason, src = AUDIT[r["model"]]
        elif re.search(NONTRANSFORMER_RE, r["model"], re.I):
            step, reason, src = "g_nontransformer", "non-transformer / hybrid", "architecture"
        elif r["model"] in AUDIT and AUDIT[r["model"]][0] == "h_D_undocumented":
            step, reason, src = AUDIT[r["model"]]
        # all applicable flags (not only the first): for robustness tables and the audit file
        flags = dict(
            f_dup=r["model"] in AUDIT and AUDIT[r["model"]][0] == "a_dedupe",
            f_suite=fam in _suite, f_repl=fam in _repl, f_moe=bool(r["moe"]),
            f_distilled=(r["model"] in AUDIT and AUDIT[r["model"]][0] == "d_distilled_pruned") or r["model"] in DISTILLED_EXTRA,
            f_multimodal=r["model"] in MULTIMODAL,
            f_instruct_only=(r["model"] in AUDIT and AUDIT[r["model"]][0] == "f_instruct_only") or not r["base_released"],
            f_nontransformer=bool(re.search(NONTRANSFORMER_RE, r["model"], re.I)),
            f_D_undoc=r["model"] in AUDIT and AUDIT[r["model"]][0] == "h_D_undocumented",
            f_m3_distilled=bool(r["distilled"]), f_synthetic=bool(r["synthetic"]))
        dep, dep_src = DEPLOY.get(r["model"], ("server/unspecified", "card: no on-device or local statement found"))
        sv, sv_src = serve_flag(r["dev"], r["year"], r["date"])
        rows.append(dict(uid=r["uid"], model=r["model"], hf=r["hf"], family=fam, dev=r["dev"], year=r["year"],
                         drop_step=step or "", drop_reason=reason, drop_source=src, deploy=dep, deploy_source=dep_src,
                         serve=sv, serve_source=sv_src, **flags))
    F = pd.DataFrame(rows)
    B = B.merge(F.drop(columns=["model", "hf", "family", "year", "dev"]), on="uid")
    B["clean"] = B["drop_step"] == ""
    B["clean_ag"] = ~B["drop_step"].isin(STEP_ORDER[:8])         # steps (a)-(g) only (h kept)
    B["ondevice"] = (B["deploy"] == "ondevice").astype(int)
    B["local"] = B["deploy"].isin(["ondevice", "local"]).astype(int)
    # family generation and common-D classification (on the clean sample and on all 173)
    B["gen"] = B["family"].map(lambda f: GENERATION.get(f, f))
    F.to_csv(os.path.join(PROC, "audit_flags.csv"), index=False)
    return d, B


def family_classes(B, mask, tol=1.10):
    """Common-D (max/min D <= tol, >= 2 members) vs size-specific D (> tol) vs singleton, within `mask`."""
    X = B[mask]
    g = X.groupby("gen")["D"].agg(["size", "min", "max"])
    cls = np.where(g["size"] < 2, "singleton", np.where(g["max"] / g["min"] <= tol, "common-D", "size-specific-D"))
    m = dict(zip(g.index, cls))
    return X["gen"].map(m), g.assign(cls=cls, ratio=g["max"] / g["min"])


def cleaning_steps(B):
    """Sequential drop order (a)..(h); returns a list of (label, mask after the step)."""
    out = [("start", pd.Series(True, index=B.index))]
    keep = pd.Series(True, index=B.index)
    for s in STEP_ORDER:
        keep = keep & (B["drop_step"] != s)
        out.append((s, keep.copy()))
    return out


def universe(d, B):
    """Production-scale universe for the open-vs-closed test and trends (m3 'prod' rows: 6ND >= 1e21, 2019-2026),
    with the clean-sample exclusions applied to Sample-B rows and name-based exclusions to Sample-A rows."""
    U = d[d["prod"] & d["year"].between(2019, 2026)].copy()
    drop_uid = set(B.loc[~B["clean"], "uid"])
    U["drop_B"] = U["uid"].isin(drop_uid)
    U["nontransformer"] = U["model"].str.contains(NONTRANSFORMER_RE, case=False, regex=True)
    U["confident"] = (U["sample"] == "B") | (U["confidence"] == "Confident")
    U["dev"] = U["lab"].where(U["lab"].notna(), U["org"]).map(developer)
    import analysis as m3an   # m3's developer map (parent developer clusters)
    U["dev"] = m3an.developer(U)
    return U
