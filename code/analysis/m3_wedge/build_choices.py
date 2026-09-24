"""build_choices.py -- the choices dataset (N, D) of pretrained decoder-only language models.

Two samples (both written to data/processed/m3_wedge/):
  Sample A  "Epoch universe": every pretrained language model in Epoch AI's all_ai_models.csv (snapshot 2026-09-23)
            released 2019-01 .. 2026-09 that is a decoder-only autoregressive text model, not derived from another
            model, with N and D (tokens) recoverable. Open AND closed models. Used for trends and heterogeneity.
  Sample B  "verified open-weight": open-weight base models with a documented token count (ObsScaling + Sloth +
            curated card-verified additions), exact N from Hugging Face safetensors, cross-checked against Epoch.
            Used for Figure 4, Table 5, the within-family analysis and the usage validation.
The final file choices.csv stacks Sample B and the Sample-A rows not matched to Sample B (flag in_B / in_A).
Excluded Epoch rows are kept with their reason in choices_excluded.csv.

Units. N = total parameters (MoE: N_total and N_active; the wedge uses N_active, which is the parameter count
that enters both training compute 6ND and inference compute 2NT). D = tokens processed during pretraining (dataset
size x epochs). Epoch's "Training dataset size" is sometimes in WORDS for older models; see words_to_tokens().
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from common import HFRAW, PROC, RAW, log
import curated as cu

EPOCH = os.path.join(RAW, "epoch_models", "all_ai_models.csv")
DATE_MIN, DATE_MAX = "2019-01-01", "2026-09-23"

# ----------------------------------------------------------------------------- architecture classification (names)
ENCODER_RE = re.compile(r"BERT|RoBERTa|ELECTRA|DeBERTa|XLM-?R|XLMR|XLNet|ALBERT|ConSERT|\bE5\b|-E5-|gte-|modernbert|RoFormer|"
                        r"Longformer|BigBird|Funnel|MPNet|MacBERT|NEZHA|Embedding|SimCSE|ERNIE(?! ?(Bot|4|X|-4))",
                        re.I)
ENCDEC_RE = re.compile(r"(^|[^A-Za-z])m?T5\b|T5-|ByT5|mT5|UL2|Flan|BART|AlexaTM|CodeT5|LongT5|NLLB|MADLAD|^Switch$|ST-MoE|"
                       r"^PLUG$|FRED-T5|CPM-2|M6-|^M6\b|Unified-IO|PanGu-Σ|Meena|BlenderBot|Generative BST|PLATO|"
                       r"Transformer \(big\)|Primer.*T5", re.I)
NONTEXT_RE = re.compile(r"CLIP|ALIGN|BLIP|Kosmos|Gato|NÜWA|BEIT|LLaVA|PaLI|VideoPoet|GAIA|Chameleon|Molmo|NVILA|-VL\b|VL-|"
                        r"Omni|Emu\d|ONE-PEACE|data2vec|LIMoE|Flamingo|MM1|OCR|Docling|Qwerty|Whisper|wav2vec|MolmoAct|"
                        r"FragLlama|BiomedGPT|MinerU", re.I)
SSM_RE = re.compile(r"Mamba|RWKV|Hyena|StripedHyena|HGRN|xLSTM|Griffin|Hawk|RetNet|GLA Transformer|Samba|Hymba|"
                    r"Nemotron-H|Granite-4\.0-H|Falcon-H1|Jamba|Kimi Linear|Qwen3-Next|MiniMax-Text|Falcon Mamba|"
                    r"RecurrentGemma|SRU", re.I)
MOE_RE = re.compile(r"MoE|Mixtral|-A\d+(\.\d+)?B|A\d+B\b|DeepSeek-V[2-9]|DeepSeek V[2-9]|Kimi K|GLM-4\.[5-9]|GLM-5|Arctic|DBRX|"
                    r"Grok|Switch|GLaM|MiniMax|Hunyuan-Large|Hunyuan-TurboS|ERNIE-4\.5|dots\.llm1|^Ling|Ling-|LongCat|"
                    r"Llama 4|gpt-oss|Jamba|OLMoE|Pangu.*MoE|Pangu Pro|Inkling|MiMo-V2|Solar Open|Arcee Trinity|Nemotron 3|"
                    r"Kimi Linear|JIUTIAN-139MoE|GigaChat Lite|^Aria$|Qwen3-Next|Qwen3-Coder-480B|Motif-3|A\.X K2|Ring-|"
                    r"Composer|DeepSeekMoE|Step-3|Qwen2-57B|Mistral Small 4|MAI-|Doubao|Grok|Gemini", re.I)
DERIVED_RE = re.compile(r"Instruct|-Chat|Chat\b|\bchat\b|-it\b| IT\b|RLHF|Thinking|Reason|-R1\b|\bR1\b|Coder-Next|"
                        r"Rumination|Z1|distill", re.I)


def pnum(x):
    """Epoch numeric fields sometimes hold comma-joined alternatives ('2590000000000,2592000000000'): take the max."""
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return max(float(t) for t in str(x).split(","))
        except ValueError:
            return np.nan


ACTIVE_RE = [re.compile(p, re.I) for p in [
    r"(\d+(?:\.\d+)?)\s*(B|billion)\s*(?:parameters\s*)?(?:are\s*)?(?:active|activated|activation)",
    r"(\d+(?:\.\d+)?)\s*(B|billion)\s*(?:params|parameters)?\s*(?:used|active)\s*(?:on average|per)",
    r"activ(?:e|ated)\s*(?:parameters|params)\s*(?:per token)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(B|billion)",
    r"with\s*(\d+(?:\.\d+)?)\s*(B|billion)\s*(?:activated|active)",
    r"of which\s*(\d+(?:\.\d+)?)\s*(B|billion)\s*(?:are\s*)?activ",
]]
# MoE models whose Epoch notes do not state active parameters (value from the developer's report)
MOE_ACTIVE_MANUAL = {"K-EXAONE": 23e9}
# Epoch rows removed by hand (derived releases, research prototypes, non-autoregressive or unofficial figures)
MANUAL_EXCLUDE = {
    "Llama 3.3 70B": "post-training update of Llama 3.1 70B (same pretraining)",
    "DeepSeek-V2.5": "merge of DeepSeek-V2-Chat and Coder-V2 (derived)",
    "DeepSeek-V3 (Mar 2025)": "post-training update of DeepSeek-V3",
    "Qwen3-235B-A22B (Jul 2025)": "post-training update of Qwen3-235B-A22B",
    "Qwen3-30B-A3B (Jul 2025)": "post-training update of Qwen3-30B-A3B",
    "GLM-4.6": "continued training of GLM-4.5 (active parameters assumed by Epoch)",
    "Granite 3.1 2B": "long-context continuation of Granite 3.0", "Granite 3.1 8B": "long-context continuation of Granite 3.0",
    "TinyLlama-1.1B (1T token checkpoint)": "intermediate checkpoint of TinyLlama 3T",
    "Aya Expanse 8B": "post-trained from Cohere Command models (D is post-training data)",
    "Aya Expanse 32B": "post-trained from Cohere Command models (D is post-training data)",
    "Apriel Nemotron 15B": "upscaled/continued from Mistral NeMo (derived)",
    "LLaDA": "masked diffusion language model (not autoregressive)",
    "GPT-4 (Mar 2023)": "unofficial (leaked) N and D; MoE active parameters unknown",
    "Llama 2-34B": "never released", "MegaScale (530B)": "systems-paper benchmark run, not a model release",
    "MegaScale (175B)": "systems-paper benchmark run, not a model release",
    "MegaScale (Production)": "systems-paper benchmark run, not a model release",
    "BitNet b1.58": "quantization research run", "OpenDiLoCo 150M": "distributed-training research run",
    "OpenDiLoCo 1.1B": "distributed-training research run", "DISTRO": "distributed-training research run",
    "NASA SMD": "encoder (RoBERTa-based INDUS)", "Vega v2": "encoder (SuperGLUE model)",
    "XY-LENTXL": "encoder (multilingual XLM-E)", "DeepNet": "translation encoder-decoder",
    "Spec-Drafter": "speculative-decoding draft model", "LMRec": "recommendation model",
    "Qwen2-57B-A14B": "MoE upcycled from a dense Qwen2 model (D for the MoE stage only)",
    "Stable LM 2 12B": "D ambiguous (2T tokens 'for two epochs': 2T or 4T)",
}
# Epoch D errors corrected from the primary source. [Review m3] The last five were found by comparing Epoch's
# dataset-size field with its own operation-counted / reported compute (C/(6ND) far from 1); in each case Epoch's own
# notes quote the primary source with the correct number of tokens processed.
EPOCH_D_FIX = {
    "GPT-3 175B (davinci)": (3.0e11, "Brown et al. (2020) Table D.1: all GPT-3 models trained for 300B tokens; "
                                     "Epoch records 238B words x 0.6 epochs"),
    "GLM-130B": (4.0e11, "GLM-130B paper (quoted in Epoch's notes): trained on over 400B tokens (200B Chinese + 200B "
                         "English); Epoch's dataset field says 152B; its compute notes use 6 x 400B x 130B"),
    "Cerebras-GPT-13B": (2.571e11, "Dey et al. (2023): 13B trained on 257B tokens (20 tokens/parameter); Epoch's 257.1B "
                                   "is already tokens seen (0.69 epochs of the Pile), so size x epochs double-counts"),
    "BloombergGPT": (5.69e11, "Wu et al. (2023): 569B tokens trained = 0.8 epochs of the 708.9B-token corpus; Epoch's "
                              "569B is already tokens seen, so size x epochs double-counts"),
    "Polyglot-Ko-12.8B": (1.67e11, "Epoch compute notes: 'trained for 167 billion tokens'; the dataset field (96B) is "
                                   "a Korean word count"),
    "GPT-SW3": (1.017e11, "Epoch notes: 97,000 steps x 512 x 2,048 = 101.7B tokens processed; the dataset field is "
                          "already tokens seen, so x 5 epochs over-counts"),
}


def active_from_notes(txt, N_total=np.inf):
    """Smallest 'active/activated parameters' figure in Epoch's parameter notes that is below the total."""
    if not isinstance(txt, str):
        return np.nan
    cands = []
    for p in ACTIVE_RE:
        for m in p.finditer(txt):
            v = float(m.group(1)) * 1e9
            if v < 0.8 * N_total:
                cands.append(v)
    return min(cands) if cands else np.nan


def active_from_name(name):
    m = re.search(r"A(\d+(?:\.\d+)?)B\b", name)
    return float(m.group(1)) * 1e9 if m else np.nan


WORDS_CONV_RE = re.compile(r"0\.75\s*words|words\s*(per|/)\s*token|tokens?\s*(per|/)\s*word|4/3|3 words per 4 tokens|"
                           r"=\s*[\d.,]+\s*[TBMK]?\s*words|[\d.,]+\s*(T|B|M|trillion|billion|million)\s*words\b", re.I)


def words_to_tokens(row, N_eff):
    """Return (D_tokens, flag). Epoch records older language-model dataset sizes in words (notes typically show the
    conversion '... tokens * 0.75 = ... words'). Rule: if the notes contain a words conversion, and operation-counted
    compute implies tokens = C/(6 N) that are 1.2-1.45x the recorded size, use C/(6N); if the notes contain a
    conversion but compute is not informative, multiply by 4/3 (Epoch's 0.75 words/token); otherwise keep."""
    DS, ep = row["DS"], row["Epochs"] if pd.notna(row["Epochs"]) else 1.0
    notes = str(row.get("Dataset size notes") or "")
    D = DS * ep
    C = row["Training compute (FLOP)"]
    oc = "Operation counting" in str(row.get("Training compute estimation method") or "")
    if not WORDS_CONV_RE.search(notes):
        return D, ""
    r = C / (6 * N_eff * D) if pd.notna(C) and pd.notna(N_eff) else np.nan
    if oc and np.isfinite(r) and 1.2 <= r <= 1.45:
        return C / (6 * N_eff), "words->tokens via C/(6N)"
    if re.search(r"tokens?\s*(\*|x)\s*0\.75|0\.75\s*words\s*(per|/)\s*token|words/token|3 words per 4 tokens|"
                 r"4/3 tokens per word|tokens per word", notes, re.I):
        # the recorded number is the WORD count obtained from a token count
        if np.isfinite(r) and 0.9 <= r <= 1.1:
            return D, "notes mention words but C/(6ND)~1: kept as tokens"
        return D * 4.0 / 3.0, "words->tokens x4/3"
    return D, "notes mention words; kept"


# ----------------------------------------------------------------------------- Epoch universe (Sample A)
def build_epoch():
    df = pd.read_csv(EPOCH)
    df["DS"] = df["Training dataset size (total)"].map(pnum)
    df["date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    lm = df[df["Domain"].fillna("").str.contains("Language")].copy()
    rows, excl = [], []
    for i, r in lm.iterrows():
        name = str(r["Model"])
        dom = str(r["Domain"])
        reason = ""
        if not (pd.Timestamp(DATE_MIN) <= r["date"] <= pd.Timestamp(DATE_MAX)):
            reason = "date outside 2019-01..2026-09"
        elif pd.notna(r["Base model"]):
            reason = "derived model (fine-tune / continued pretraining / post-training of a base model)"
        elif pd.isna(r["Parameters"]):
            reason = "N missing"
        elif re.search(r"Vision|Image|Video|Audio|Speech|Robotics|3D|Biology|Medicine|Multimodal", dom):
            reason = "multimodal domain (D includes non-text tokens)"
        elif ENCODER_RE.search(name):
            reason = "encoder (masked LM / embedding model)"
        elif ENCDEC_RE.search(name):
            reason = "encoder-decoder or dialogue seq2seq"
        elif NONTEXT_RE.search(name):
            reason = "multimodal / non-text model"
        elif DERIVED_RE.search(name):
            reason = "post-trained / derived variant (name)"
        elif name in MANUAL_EXCLUDE:
            reason = "manual: " + MANUAL_EXCLUDE[name]
        if reason:
            excl.append(dict(model=name, org=r["Organization"], date=r["date"], reason=reason))
            continue
        moe = bool(MOE_RE.search(name)) or name in MOE_ACTIVE_MANUAL or \
            bool(re.search(r"mixture[- ]of[- ]experts|\bMoE\b|sparse", str(r["Parameters notes"]), re.I))
        N_total = float(r["Parameters"])
        N_act = np.nan
        if moe:
            N_act = MOE_ACTIVE_MANUAL.get(name, active_from_name(name))
            if np.isnan(N_act):
                N_act = active_from_notes(r["Parameters notes"], N_total)
            if np.isnan(N_act) and pd.notna(r["DS"]) and pd.notna(r["Training compute (FLOP)"]) and \
                    "Operation counting" in str(r["Training compute estimation method"]) and r["Confidence"] != "Speculative":
                N_act = r["Training compute (FLOP)"] / (6 * r["DS"] * (r["Epochs"] if pd.notna(r["Epochs"]) else 1))
                if not (0.005 * N_total < N_act < 0.8 * N_total):
                    N_act = np.nan
            if np.isnan(N_act):
                excl.append(dict(model=name, org=r["Organization"], date=r["date"], reason="MoE with unknown active parameters"))
                continue
        N_eff = N_act if moe else N_total
        D_flag = ""
        if pd.notna(r["DS"]):
            D, D_flag = words_to_tokens(r, N_eff)
            D_src = "Epoch dataset size x epochs"
        elif pd.notna(r["Training compute (FLOP)"]) and "Operation counting" in str(r["Training compute estimation method"]) \
                and r["Confidence"] in ("Confident", "Likely"):
            D = r["Training compute (FLOP)"] / (6 * N_eff)
            D_src, D_flag = "Epoch C/(6N) (operation counting)", "D from compute"
        else:
            excl.append(dict(model=name, org=r["Organization"], date=r["date"], reason="D (tokens) not recoverable"))
            continue
        if name in EPOCH_D_FIX:
            D, D_flag = EPOCH_D_FIX[name][0], "fixed: " + EPOCH_D_FIX[name][1]
        uniq = r["DS"] if pd.notna(r["DS"]) else D
        if uniq < 5e9 or D < 1e9:
            excl.append(dict(model=name, org=r["Organization"], date=r["date"],
                             reason="benchmark-scale research model (unique corpus < 5e9 tokens: WikiText/PTB/enwik8-type)"))
            continue
        if (pd.notna(r["Epochs"]) and r["Epochs"] > 20) or D / N_eff > 3e4:
            excl.append(dict(model=name, org=r["Organization"], date=r["date"],
                             reason=f"implausible: epochs = {r['Epochs']}, D/N = {D / N_eff:.0f}"))
            continue
        task = str(r["Task"])
        code = task.strip() == "Code generation" or bool(re.search(r"Code|Coder|StarCoder|CodeGen|Codestral|SantaCoder|DeciCoder|Refact|PolyCoder|CodeGeeX", name))
        acc = str(r["Model accessibility"])
        openw = (str(r["Open model weights?"]) == "Yes") or acc.startswith("Open weights")
        arch = "moe" if moe else ("ssm/hybrid" if SSM_RE.search(name) else "dense")
        C = r["Training compute (FLOP)"]
        rows.append(dict(
            model=name, epoch_model=name, org=r["Organization"], org_cat=r["Organization categorization"],
            country=r["Country (of organization)"], date=r["date"], open_weights=openw, accessibility=acc,
            arch=arch, moe=moe, N_total=N_total, N_active=N_act if moe else N_total, N=N_eff, D=D, D_src=D_src,
            D_flag=D_flag, D_epoch=r["DS"] * (r["Epochs"] if pd.notna(r["Epochs"]) else 1), epochs=r["Epochs"],
            C_epoch=C, C_method=r["Training compute estimation method"], confidence=r["Confidence"],
            # [Review m3] consistency of D with Epoch's own compute: C/(6ND) should be ~1 when compute is
            # operation-counted or reported; |ln r| > ln 1.5 flags a residual unit/epoch problem (robustness check)
            C_D_ratio=C / (6 * N_eff * D) if pd.notna(C) else np.nan,
            D_C_mismatch=bool(pd.notna(C) and re.search(r"Operation counting|Reported", str(r["Training compute estimation method"]))
                              and abs(np.log(C / (6 * N_eff * D))) > np.log(1.5)),
            code=code, distilled=bool(re.search(r"distill", str(r["Parameters notes"]) + str(r["Dataset size notes"]), re.I)),
            notable=pd.notna(r["Notability criteria"]), frontier=bool(r["Frontier model"] is True),
            hf_dev=r["Hugging Face developer id"],
            # core = production-scale (6ND >= 1e21 FLOP), not code-specialised, N and D not 'Speculative' in Epoch
            core=bool(6 * N_eff * D >= 1e21 and not code and r["Confidence"] != "Speculative")))
    A = pd.DataFrame(rows)
    X = pd.DataFrame(excl)
    return A, X


# ----------------------------------------------------------------------------- Hugging Face metadata
def _hf_path(i):
    return os.path.join(HFRAW, "api", i.replace("/", "__") + ".json")


def hf_meta(i):
    p = _hf_path(i)
    if not os.path.exists(p):
        return {}
    try:
        d = json.load(open(p))
    except json.JSONDecodeError:
        return {}
    if "error" in d:
        return {}
    st = d.get("safetensors") or {}
    par = st.get("parameters") or {}
    # floating-point tensors only: GPT-NeoX/GPT-Neo checkpoints also store U8 causal-mask buffers (m4 review fix)
    nf = sum(v for k, v in par.items() if str(k).upper().startswith(("F", "BF")))
    return dict(hf_found=True, N_hf=float(nf) if nf > 0 else np.nan, hf_created=d.get("createdAt"),
                dl30=d.get("downloads"), dl_all=d.get("downloadsAllTime"), likes=d.get("likes"))


def hf_config_emb(i):
    """Embedding parameters from config.json: vocab x width x (1 if tied else 2)."""
    p = os.path.join(HFRAW, "configs", i.replace("/", "__") + ".json")
    if not os.path.exists(p):
        return np.nan, None
    try:
        c = json.load(open(p))
    except json.JSONDecodeError:
        return np.nan, None
    tc = c.get("text_config") if isinstance(c.get("text_config"), dict) else c
    V = tc.get("vocab_size") or c.get("vocab_size") or c.get("padded_vocab_size") or tc.get("padded_vocab_size")
    H = tc.get("hidden_size") or tc.get("d_model") or tc.get("n_embd") or c.get("hidden_size") or c.get("n_embd")
    tie = tc.get("tie_word_embeddings", c.get("tie_word_embeddings", True))
    if not (V and H):
        return np.nan, None
    return float(V) * float(H) * (1 if tie else 2), bool(tie)


# ----------------------------------------------------------------------------- Sample B
def build_sample_b():
    o = pd.read_csv(os.path.join(RAW, "obsscaling", "base_llm_benchmark_eval.csv"))
    rows, dropped = [], []
    for _, r in o.iterrows():
        mid = r["Model"]
        fx = cu.OBS_FIXES.get(mid, {})
        if fx.get("drop"):
            dropped.append(dict(model=mid, reason=fx["drop"]))
            continue
        if "pythia" in mid.lower():
            continue
        fam = r["Model Family"]
        lab, dom, special = cu.OBS_FAMILY.get(fam, ("", "general", ""))
        special = cu.SPECIAL_OVERRIDE.get(mid, special)
        N_B = fx.get("N", r["Model Size (B)"])
        D_T = fx.get("D", r["Pretraining Data Size (T)"])
        if pd.isna(D_T):
            dropped.append(dict(model=mid, reason="D missing in ObsScaling and no documented fix"))
            continue
        moe = cu.OBS_MOE.get(mid)
        rows.append(dict(hf=cu.CANONICAL.get(mid, mid), obs_id=mid, family=cu.FAMILY_SPLIT.get(mid, fam), lab=lab,
                         domain=dom, special=special, N_rep=(moe["N_total"] if moe else N_B) * 1e9,
                         N_active_rep=(moe["N_active"] * 1e9 if moe else np.nan), D=D_T * 1e12,
                         D_src="ObsScaling" + (f"; fix: {fx['note']}" if "note" in fx else ""),
                         D_obs=r["Pretraining Data Size (T)"] * 1e12, arch="moe" if moe else ("ssm/hybrid" if fam == "RWKV" else "dense"),
                         base_released=True, mm=False, card_re=None))
    for m, nb in zip(cu.PYTHIA_STD, [0.07, 0.16, 0.41, 1.0, 1.4, 2.8, 6.9, 12.0]):
        rows.append(dict(hf=f"EleutherAI/pythia-{m}", obs_id=None, family="Pythia", lab="EleutherAI", domain="general",
                         special="", N_rep=nb * 1e9, N_active_rep=np.nan, D=299.892736e9,
                         D_src="Biderman et al. (2023): 299,892,736,000 tokens (Sloth 0.25-0.3T; ObsScaling 0.3T)",
                         D_obs=0.3e12, arch="dense", base_released=True, mm=False, card_re=None))
    for c in cu.CURATED:
        rows.append(dict(hf=c["hf"], obs_id=None, family=c["family"], lab=c["lab"], domain="general",
                         special=c.get("special", ""), N_rep=np.nan,
                         N_active_rep=c.get("N_active", np.nan) * 1e9 if "N_active" in c else np.nan,
                         D=c["D"] * 1e12, D_src=c["src"], D_obs=np.nan, arch=c.get("arch", "dense"),
                         base_released=c.get("base_released", True), mm=c.get("mm", False), card_re=c.get("card_re"),
                         vision=c.get("vision_B", 0.0) * 1e9, note=c.get("note", ""), date_cur=c.get("date")))
    B = pd.DataFrame(rows)
    # --- Hugging Face metadata: exact N, upload date; usage (base + official instruct)
    meta = B["hf"].map(hf_meta).apply(pd.Series)
    B = pd.concat([B, meta], axis=1)
    inst_dl30, inst_all, inst_n = [], [], []
    for h in B["hf"]:
        s30 = sall = 0.0
        k = 0
        for j in cu.INSTRUCT.get(h, []):
            mj = hf_meta(j)
            if mj:
                k += 1
                s30 += mj.get("dl30") or 0
                sall += mj.get("dl_all") or 0
        inst_dl30.append(s30), inst_all.append(sall), inst_n.append(k)
    B["dl30_instruct"], B["dlall_instruct"], B["n_instruct_found"] = inst_dl30, inst_all, inst_n
    B["dl30_total"] = B["dl30"].fillna(0) + B["dl30_instruct"]
    B["dlall_total"] = B["dl_all"].fillna(0) + B["dlall_instruct"]
    # --- N: exact float count from safetensors when present (minus vision tower for multimodal checkpoints),
    #     else the reported size; MoE: total from HF, active from the card/report
    B["N_total"] = np.where(B["N_hf"].notna(), B["N_hf"] - B["vision"].fillna(0), B["N_rep"])
    B["N_src"] = np.where(B["N_hf"].notna(), "HF safetensors (float tensors)", "reported")
    B["moe"] = B["arch"].eq("moe")
    B["N_active"] = np.where(B["moe"], B["N_active_rep"], B["N_total"])
    B["N"] = B["N_active"]
    # sanity: HF count vs reported size (ObsScaling) -- flag > 25% gaps
    B["N_gap"] = np.where(B["N_rep"].notna() & B["N_hf"].notna(), B["N_hf"] / B["N_rep"] - 1, np.nan)
    # --- embedding parameters (for the non-embedding Farseer technology)
    emb = B["hf"].map(hf_config_emb)
    B["N_emb"] = [e[0] for e in emb]
    B["tied"] = [e[1] for e in emb]
    B["N_nonemb"] = np.where(B["moe"], np.nan, B["N_total"] - B["N_emb"])
    # --- card verification of D
    ver = []
    for _, r in B.iterrows():
        if not isinstance(r["card_re"], str):
            ver.append(np.nan)
            continue
        p = os.path.join(HFRAW, "cards", r["hf"].replace("/", "_") + ".md")
        t = open(p, errors="ignore").read() if os.path.exists(p) else ""
        t = re.sub(r"\s+", " ", t)   # cards wrap lines; match on whitespace-normalised text
        ver.append(bool(re.search(r["card_re"], t)))
    B["D_card_verified"] = ver
    return B, pd.DataFrame(dropped)


def attach_epoch(B, A_all):
    """Match Sample B to Epoch rows (release date, organization, confidence, Epoch D for the cross-check)."""
    ep = pd.read_csv(EPOCH)
    ep["DS"] = ep["Training dataset size (total)"].map(pnum)
    ep = ep.set_index("Model")
    out = []
    for _, r in B.iterrows():
        nm = cu.EPOCH_NAME.get(r["obs_id"] if isinstance(r["obs_id"], str) else r["hf"]) or cu.EPOCH_NAME.get(r["hf"])
        d = dict(epoch_model=None, epoch_date=pd.NaT, epoch_D=np.nan, epoch_DC=np.nan, org=None, country=None,
                 confidence=None)
        if nm and nm in ep.index:
            e = ep.loc[nm]
            if isinstance(e, pd.DataFrame):
                e = e.iloc[0]
            Neff = r["N"]
            d.update(epoch_model=nm, epoch_date=pd.to_datetime(e["Publication date"], errors="coerce"),
                     epoch_D=e["DS"] * (e["Epochs"] if pd.notna(e["Epochs"]) else 1),
                     epoch_DC=e["Training compute (FLOP)"] / (6 * Neff) if pd.notna(e["Training compute (FLOP)"]) else np.nan,
                     org=e["Organization"], country=e["Country (of organization)"], confidence=e["Confidence"])
        out.append(d)
    E = pd.DataFrame(out, index=B.index)
    B = pd.concat([B, E], axis=1)
    hfd = pd.to_datetime(B["hf_created"], errors="coerce", utc=True).dt.tz_localize(None)
    dcur = pd.to_datetime(B["date_cur"], errors="coerce") if "date_cur" in B else pd.Series(pd.NaT, index=B.index)
    B["date"] = B["epoch_date"].fillna(dcur).fillna(hfd)
    B["date_src"] = np.where(B["epoch_date"].notna(), "Epoch publication date",
                             np.where(dcur.notna(), "release date (developer announcement)", "HF upload date"))
    # HF-migrated repositories: upload date is not the release date (from Epoch / papers; see m4 DATE_OVERRIDE)
    over = {"EleutherAI/gpt-neo-125m": "2021-03-21", "EleutherAI/gpt-neo-1.3B": "2021-03-21", "facebook/xglm-1.7B": "2021-12-20",
            "facebook/xglm-4.5B": "2021-12-20", "facebook/xglm-564M": "2021-12-20", "facebook/opt-125m": "2022-05-02",
            "facebook/opt-13b": "2022-05-02", "Salesforce/codegen-6B-nl": "2022-03-25", "Salesforce/codegen-16B-nl": "2022-03-25",
            "cerebras/Cerebras-GPT-111M": "2023-03-28", "cerebras/Cerebras-GPT-256M": "2023-03-28",
            "cerebras/Cerebras-GPT-590M": "2023-03-28", "cerebras/Cerebras-GPT-1.3B": "2023-03-28",
            "cerebras/Cerebras-GPT-2.7B": "2023-03-28", "cerebras/Cerebras-GPT-6.7B": "2023-03-28",
            "cerebras/btlm-3b-8k-base": "2023-07-24", "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": "2023-05-05",
            "mosaicml/mpt-7b": "2023-05-05", "mosaicml/mpt-30b": "2023-06-22", "tiiuae/falcon-rw-1b": "2023-04-26",
            "EleutherAI/pythia-70m": "2023-02-13", "EleutherAI/pythia-160m": "2023-02-13", "EleutherAI/pythia-410m": "2023-02-13",
            "EleutherAI/pythia-1b": "2023-02-13", "EleutherAI/pythia-1.4b": "2023-02-13", "EleutherAI/pythia-2.8b": "2023-02-13",
            "EleutherAI/pythia-6.9b": "2023-02-13", "EleutherAI/pythia-12b": "2023-02-13"}
    for k, v in over.items():
        if k in set(B["hf"]) and B.loc[B["hf"] == k, "epoch_date"].isna().all():
            B.loc[B["hf"] == k, ["date", "date_src"]] = [pd.Timestamp(v), "release date (paper/Epoch sibling)"]
    # D cross-check vs Epoch (tokens); |log10 ratio| > 0.1 flagged
    B["D_vs_epoch"] = np.log10(B["D"] / B["epoch_D"])
    return B


def main():
    log("build_choices: Epoch universe")
    A, X = build_epoch()
    log(f"  Epoch candidates kept {len(A)}, excluded {len(X)}")
    log("build_choices: Sample B (verified open-weight)")
    B, Bdrop = build_sample_b()
    B = attach_epoch(B, A)
    B["in_B"] = True
    B["open_weights"] = True
    B["model"] = B["hf"].str.split("/").str[1]
    B["M"] = B["D"] / B["N"]
    B["C6"] = 6 * B["N"] * B["D"]
    B["code"] = B["domain"].eq("code")
    B["distilled"] = B["special"].eq("distilled")
    B["synthetic"] = B["special"].eq("synthetic")
    B["continued"] = B["special"].eq("continued")
    # mark Sample-A rows matched to Sample B (the verified values replace Epoch's)
    matched = set(B["epoch_model"].dropna())
    A["in_B"] = A["epoch_model"].isin(matched)
    A["M"] = A["D"] / A["N"]
    A["C6"] = 6 * A["N"] * A["D"]
    A["sample"] = "A"
    B["sample"] = "B"
    B["in_A"] = B["epoch_model"].isin(set(A["epoch_model"]))
    A.to_csv(os.path.join(PROC, "sampleA_epoch.csv"), index=False)
    X.to_csv(os.path.join(PROC, "choices_excluded.csv"), index=False)
    Bdrop.to_csv(os.path.join(PROC, "sampleB_dropped.csv"), index=False)
    cols_B = ["model", "hf", "family", "lab", "org", "country", "date", "date_src", "open_weights", "arch", "moe",
              "N_total", "N_active", "N", "N_src", "N_rep", "N_gap", "N_emb", "tied", "N_nonemb", "D", "D_src",
              "D_card_verified", "D_obs", "epoch_model", "epoch_D", "epoch_DC", "D_vs_epoch", "confidence", "M", "C6",
              "code", "distilled", "synthetic", "continued", "mm", "base_released", "dl30", "dl_all", "dl30_instruct",
              "dlall_instruct", "n_instruct_found", "dl30_total", "dlall_total", "likes", "hf_found", "in_A", "sample", "note"]
    B[cols_B].to_csv(os.path.join(PROC, "sampleB_verified.csv"), index=False)
    # stacked choices file: B + unmatched A
    Au = A[~A["in_B"]].copy()
    Au["family"] = None
    Au["lab"] = Au["org"]
    Au["in_A"] = True
    Au["synthetic"] = False
    Au["continued"] = False
    Au["mm"] = False
    common_cols = ["model", "family", "lab", "org", "date", "open_weights", "arch", "moe", "N_total", "N_active", "N",
                   "D", "D_src", "M", "C6", "confidence", "code", "distilled", "synthetic", "continued", "mm", "sample",
                   "in_A", "in_B"]
    Bc = B.copy()
    Bc["uid"] = "B:" + Bc["hf"]
    Bc["core"] = True
    Au["uid"] = "A:" + Au["epoch_model"]
    ch = pd.concat([Bc[common_cols + ["uid", "core", "hf", "N_nonemb", "N_emb", "tied", "D_card_verified", "base_released",
                                      "dl30_total", "dlall_total", "dl30", "dl_all", "likes", "epoch_model", "note"]],
                    Au[common_cols + ["uid", "core", "D_flag", "accessibility", "org_cat", "country", "notable", "frontier",
                                      "C_epoch", "C_method", "C_D_ratio", "D_C_mismatch", "epoch_model"]]],
                   ignore_index=True, sort=False)
    ch = ch.loc[:, ~ch.columns.duplicated()]
    ch["year"] = pd.to_datetime(ch["date"]).dt.year
    ch.to_csv(os.path.join(PROC, "choices.csv"), index=False)
    log(f"  Sample B: {len(B)} models ({B['family'].nunique()} families); dropped {len(Bdrop)}; "
        f"card-verified D: {int(B['D_card_verified'].fillna(False).sum())} of {int(B['D_card_verified'].notna().sum())} checked")
    return A, B, X, ch


if __name__ == "__main__":
    main()
