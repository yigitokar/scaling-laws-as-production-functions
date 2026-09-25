"""coderB.py: [review] copied unchanged (except the ROOT line) from coder B's session scratchpad on 2026-09-25 so that
the blind coding ships with the module; it reproduces data/processed/rb5_units/readings_coderB.csv byte for byte.

coderB.py: blind second coding (coder B) of the rb5 reading protocol (paper/notes/rb5_reading_protocol.md,
version 1 with Amendments 1-3). Codes were assigned from the saved primary sources in data/raw/rb2_decisions/ only.
Writes data/processed/rb5_units/readings_coderB.csv in the protocol's Section 10 format and checks every quote
(whitespace removed, case folded) against the saved text before writing.
"""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # [review] was an absolute path
RAW = os.path.join(ROOT, "data", "raw", "rb2_decisions")
PROC = os.path.join(ROOT, "data", "processed", "rb5_units")
PROTOCOL = "rb5_reading_protocol v1 (2026-09-25) with Amendments 1-3"

_T = {}


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def found(key, q):
    if key not in _T:
        _T[key] = norm(open(os.path.join(RAW, key + ".txt"), encoding="utf-8", errors="replace").read())
    return norm(q) in _T[key]


# Each unit: D, D_rule, N, N_rule (common-budget only), cap (list of members with own cap = 1), R (set of flags),
# syn (0/1), ev = [(role, source_key, quote)], note.
U = {}

# ---------------------------------------------------------------- common-budget units
U["Apertus"] = dict(
    D="choice", D_rule="K2", N="free", N_rule="not stated", cap=[], R=set(), syn=1,
    ev=[("D", "apertus_report", "Note that not nec- essarily all tokens of each stage data were consumed, due to the stage duration."),
        ("D", "apertus_report", "Stage 2 (5T - 9T tokens) FineWeb-HQ (33% highest quality) 4064 FineWeb-2-HQ (33% highest quality) and FineWeb-2 (random 33% sample of remaining languages) 3557 FineWeb-Edu (Score-3) 1179"),
        ("synthetic", "apertus_report", "For task data we rely on EuroBlocks-SFT-Synthetic- 112427 (Martins et al., 2025) for multilingual instruction and task data"),
        ("synthetic", "apertus_report", "English as well as multilingual instruction and task data, the Data Provenance Initiative subset of Flan and the Euroblocks."),
        ("synthetic", "apertus_report", "We include a small amount of synthetically gener- ated examples into the corpus to conduct scientific research in pretraining data poison- ing")],
    note="K2: stage mixes (Table 6) hold more tokens than each stage consumed. 'DCLM-edu is limited in size' concerns one component, not the budget: not coded C. "
         "Synthetic: EuroBlocks-SFT-Synthetic (stage 5) and poisoning examples are called synthetic; the generator is not named (coded 1 because the protocol equates synthetic with model-generated). "
         "Card: 'running on-device with MLX' is deployment support, not a size rationale (R_DEV 0). GQA for inference efficiency is architecture (Amendment 3).")

U["DeepSeek-LLM"] = dict(
    D="cap", D_rule="C1", N="menu", N_rule="prevalent configurations (7B, 67B)",
    cap=["deepseek-llm-7b-base", "deepseek-llm-67b-base"], R=set(), syn=0,
    ev=[("D+own_cap", "deepseek_llm_report", "To support the pre-training phase, we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding."),
        ("N", "deepseek_llm_report", "facilitate the scaling of large scale models in two prevalent used open- source configurations, 7B and 67B."),
        ("N", "deepseek_llm_report", "These layer adjustments, while maintaining parameter consistency with other open-source models, also facilitate model pipeline partitioning to optimize training and inference.")],
    note="C1 is the protocol's own example wording ('the dataset currently consists of X tokens'); own cap 1 for both members by Amendment 1. "
         "R_CO not coded: 'Guided by the scaling laws, we introduce DeepSeek LLM' does not say the sizes or the 2T budget were set by the law (sizes are prevalent configurations). GQA for inference cost is architecture.")

U["Granite-3.0"] = dict(
    D="choice", D_rule="K2", N="free", N_rule="not stated", cap=[], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("D", "granite30_report", "In stage 1, our dense and MoE models are trained on 10 trillion and 8 trillion tokens, respectively."),
        ("D", "granite30_report", "The FineWeb dataset consists of more than 15T tokens of cleaned and deduplicated English web data from CommonCrawl"),
        ("D", "granite30_report", "This is a 4T token / 3B document pretraining dataset that achieves strong performance on language model benchmarks"),
        ("R_INF", "granite30_announcement", "the new IBM Granite 3.0 models deliver state-of-the-art performance relative to model size while maximizing safety, speed and cost-efficiency for enterprise use cases."),
        ("R_DEV", "granite30_report", "these models target enterprise use cases, including on-premise and on-device settings."),
        ("synthetic", "granite30_report", "Some of the data sources for stage 2 are the same as the stage 1 data sources, mixed with a small amount of high-quality open-source and synthetic corpora with permissive licenses."),
        ("synthetic", "granite30_report", "Language Instructions [P2][IBM-Synthetic]: Synthetic dataset of instruction-response pairs created using EvolInstruct to improve complex reasoning and conversation skills."),
        ("synthetic", "granite30_report", "We only use open-source teacher models with a non-restrictive license to generate instructions and their respective responses.")],
    note="K2 is inferred from stated sizes: the 10T stage-1 budget draws on FineWeb (listed P1, more than 15T tokens) and the 2T stage 2 on DCLM-Baseline (listed P2, 4T tokens); the sources do not say in words that the budget is part of a larger pool. "
         "No C evidence. MoE models (10T) are not same-data evidence for K3 (fewer tokens).")

U["Llama#1"] = dict(
    D="choice", D_rule="K3", N="free", N_rule="not stated", cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("D", "llama1_report", "LLaMA-33B and LLaMA- 65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens."),
        ("D", "llama1_report", "The pre-training runs on 1T tokens have the same sampling proportion."),
        ("R_INF", "llama1_report", "The focus of this work is to train a series of language models that achieve the best possible per- formance at various inference budgets, by training on more tokens than what is typically used."),
        ("R_DEV", "llama1_report", "We believe that this model will help democratize the access and study of LLMs, since it can be run on a single GPU."),
        ("R_DEV", "llama1_report", "This model runs on a single V100 GPU during inference.")],
    note="R_DEV statements single out LLaMA-13B (a member of this unit). R_CO not coded: 'inspired by the Chinchilla scaling laws' describes the approach, and the same page sets budgets beyond the Chinchilla allocation for inference.")

U["Llama#2"] = dict(
    D="cap or choice", D_rule="size only", N="free", N_rule="not stated", cap=[], R={"R_INF"}, syn=0,
    ev=[("D", "llama1_report", "Overall, our entire training dataset contains roughly 1.4T tokens after tokenization. For most of our training data, each token is used only once dur- ing training, with the exception of the Wikipedia and Books domains, over which we perform ap- proximately two epochs."),
        ("D", "llama1_report", "LLaMA-33B and LLaMA- 65B were trained on 1.4T tokens."),
        ("R_INF", "llama1_report", "The focus of this work is to train a series of language models that achieve the best possible per- formance at various inference budgets, by training on more tokens than what is typically used.")],
    note="Budget equals the stated entire dataset (1.4T); two epochs on Wikipedia and Books is upsampling (Section 6), so no C2. "
         "Single-GPU statements single out LLaMA-13B (other unit), so R_DEV 0.")

U["Llama-2"] = dict(
    D="choice", D_rule="K1", N="free", N_rule="not stated", cap=[], R=set(), syn=0,
    ev=[("D", "llama2_report", "We trained on 2 trillion tokens of data as this provides a good performance–cost trade-off")],
    note="Protocol K1 example. GQA 'for improved inference scalability' of 34B/70B is architecture (Amendment 3). The saved llama2_blog text is an unrelated Meta developer page (not used).")

U["Llama-3 herd"] = dict(
    D="choice", D_rule="K1", N="free", N_rule="not stated", cap=[], R={"R_CO", "R_INF"}, syn=0,
    ev=[("D", "llama3_report", "Extrapolation of the resulting scaling law to3.8× 1025 FLOPs suggests training a 402B parameter model on 16.55T tokens."),
        ("D", "llama3_report", "Based on this observation, we ultimately decided to train a flagship model with 405B parameters."),
        ("D", "llama3_report", "We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2."),
        ("R_CO", "llama3_report", "While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal."),
        ("R_INF", "llama3_report", "The resulting models perform better than compute-optimal models at the same inference budget."),
        ("R_INF", "llama3_blog", "smaller models are generally preferred because they are much more efficient during inference.")],
    note="K1 from the flagship (a member): its size and tokens follow from the compute-optimal allocation of a 3.8e25 FLOP budget. "
         "The corpus is 'about 15T' and the 8B/70B used it once (size-only evidence); 'longer than compute-optimal' and 'continued to improve' are not token-margin evidence. "
         "Borderline: a coder who does not read the flagship allocation as K1 would code 'cap or choice' (size only).")

U["MPT"] = dict(
    D="choice", D_rule="K2", N="menu", N_rule="named GPU (single A100-80GB / A100-40GB)", cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("D", "mpt7b_blog", "the model was pre-trained on 1T tokens sampled according to this mix."),
        ("D", "mpt30b_blog", "We collected 1T tokens of pre-training data from ten different open-source text corpora. We tokenized the text using the EleutherAI GPT-NeoX-20B tokenizer and sampled according to the above ratios."),
        ("N+R_DEV", "mpt30b_blog", "The size of MPT-30B was also specifically chosen to make it easy to deploy on a single GPU"),
        ("R_INF", "mpt30b_blog", "Other comparable LLMs such as Falcon-40B have larger parameter counts and cannot be served on a single datacenter GPU (today); this necessitates 2+ GPUs, which increases the minimum inference system cost.")],
    note="K2 wording 'sampled according to this mix' (protocol example 'sampled from a larger mix'). Menu: MPT-30B size set to fit one named GPU.")

U["OLMo-2#1"] = dict(
    D="choice", D_rule="K1; K3", N="free", N_rule="not stated", cap=[], R=set(), syn=1,
    ev=[("D", "olmo2_report", "Similar toOLMo 27B, we use 2000 steps of warmup, set the schedule to 5 trillion tokens but truncate at the 4 trillion mark."),
        ("D", "olmo2_report", "Accordingly, for the 7B variant, we stop the schedule at 4T tokens and then switch to mid-training"),
        ("D", "olmo2_32b_blog", "OLMo 2 7B is trained for one epoch , up to 4T tokens, OLMo 2 13B is trained for 1.3 epochs , up to 5T tokens, OLMo 2 32B is trained for 1.5 epochs , up to 6T tokens."),
        ("synthetic", "olmo2_report", "We generated approximately 6.5B tokens of synthetic math data from rewritten versions of Tiny-GSM"),
        ("synthetic", "olmo2_report", "Then we pass all of these annotated examples to Qwen2.5-7B-Instruct to be rewritten in the style of MIND"),
        ("synthetic", "olmo2_report", "perform a single 50B token anneal onDolmino Mix 1124.")],
    note="K1: cosine schedules planned for 5T and truncated at 4T (1B and 7B). K3: 13B and 32B trained on more tokens of the same OLMo 2 Mix 1124. "
         "7B 'approximately one epoch' does not count as own cap. The 1B is described as an experimentation size (no listed rationale).")

U["Olmo-3"] = dict(
    D="choice", D_rule="K2; K1", N="free", N_rule="not stated", cap=[], R={"R_DEV"}, syn=1,
    ev=[("D", "olmo3_blog", "Olmo 3 is pretrained on Dolma 3 , a new ~9.3-trillion-token corpus drawn from web pages, science PDFs processed with olmOCR , codebases, math problems and solutions, and encyclopedic text. From this pool, we construct Dolma 3 Mix , a 5.9-trillion-token (~6T) pretraining mix"),
        ("D", "olmo3_report", "The learning rate schedule is a cosine schedule over one epoch (5.93T tokens), truncated at 5.5T tokens."),
        ("R_DEV", "olmo3_blog", "Olmo 3 is a family of compact, dense models at 7 billion and 32 billion parameters that can run on everything from laptops to research clusters."),
        ("synthetic", "olmo3_report", "Using these annotations as foundation, we promptGPT-4.1 and o4-mini to generate thinking traces for each capability-targeted task.")],
    note="K2 is the protocol example (6T mix drawn from a 9T pool); K1 from the 32B schedule truncated at 5.5T. Quality-aware upsampling inside the mix is not own cap. "
         "The 32B 'small enough that a wide audience can fine-tune and deploy them on accessible hardware' was also read; R_DEV rests on the laptop statement for the family.")

U["Qwen#2"] = dict(
    D="cap or choice", D_rule="size only", N="free", N_rule="not stated", cap=[], R=set(), syn=0,
    ev=[("D", "qwen_report", "Finally, we have built a dataset of up to 3 trillion tokens."),
        ("D", "qwen_readme", "which are trained on 3T tokens and support 32k context")],
    note="Budget (3T for 14B and 72B) equals the stated dataset; no C or K evidence (7B and 1.8B used fewer tokens). "
         "Borderline rationale not coded: 'providing more comprehensive and powerful LLMs at developer- or application-friendly scales' names no device, memory or inference cost.")

U["Qwen2#1"] = dict(
    D="choice", D_rule="K1; K3", N="menu", N_rule="consumer devices (smartphones, earphones, smart glasses) for Qwen2-1.5B",
    cap=[], R={"R_DEV"}, syn=1,
    ev=[("D", "qwen2_report", "Considering training costs, we opted to use the higher-quality 7 trillion token dataset for training larger models, leaving further exploration for future model iterations."),
        ("D", "qwen2_report", "An attempt to further relax the quality threshold resulted in a 12 trillion token dataset."),
        ("D", "qwen2_report", "Qwen2-0.5B were pre-trained using the 12 trillion token dataset."),
        ("N+R_DEV", "qwen2_report", "The smaller models, specifically Qwen2-0.5B and Qwen2-1.5B, are designed for easy deployment on portable devices such as smartphones, earphones, and smart glasses."),
        ("R_DEV", "qwen2_report", "Conversely, the larger models cater to deployment across GPUs of varying scales."),
        ("synthetic", "qwen2_report", "Moreover, these models are utilized to synthesize high-quality pre-training data.")],
    note="K1 (training costs) and K3 (the 0.5B trained on the 12T relaxed-threshold dataset). Menu read from the named consumer devices for the 1.5B member; a reader who treats this as a general on-device statement would code N free (reading choice).")

U["Qwen2.5"] = dict(
    D="cap or choice", D_rule="size only", N="free", N_rule="not stated", cap=[], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("D", "qwen25_report", "we have developed a larger and higher-quality pre-training dataset, expanding from the 7 trillion tokens used in Qwen2 (Yang et al., 2024a) to 18 trillion tokens."),
        ("D", "qwen25_blog", "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens."),
        ("R_INF", "qwen25_report", "Qwen2.5 brings back the 3B, 14B, and 32B models, which are more cost-effective for resource-limited scenarios"),
        ("R_DEV", "qwen25_report", "For edge-side models, we compare Qwen2.5-0.5B, 1.5B, and 3B against estab- lished baselines"),
        ("synthetic", "qwen25_report", "To generate high-quality synthetic data, particularly in mathematics, code, and knowledge domains, we leverage both Qwen2-72B-Instruct (Yang et al., 2024a) and Qwen2- Math-72B-Instruct")],
    note="Size only: every member trained on the stated 18T dataset; no C or K. 'Edge-side' and 'resource-limited scenarios' are general, not a named device (N free).")

U["Qwen3"] = dict(
    D="cap or choice", D_rule="size only", N="free", N_rule="not stated", cap=[], R={"R_DEV"}, syn=1,
    ev=[("D", "qwen3_report", "All Qwen3 models are trained on a large and diverse dataset consisting of119 languages and dialects, with a total of 36 trillion tokens."),
        ("R_DEV", "qwen3_report", "For edge-side models, we take similar-sized Qwen2.5, Llama-3, and Gemma-3 base models as the baselines."),
        ("synthetic", "qwen3_report", "we employ Qwen2.5 (Yang et al., 2024b), Qwen2.5-Math (Yang et al., 2024c), and Qwen2.5-Coder (Hui et al., 2024) models to synthesize trillions of text tokens in different formats")],
    note="Size only (36T dataset, all members). Edge-side statement covers the 8B, 4B, 1.7B and 0.6B members.")

U["SmolLM#1"] = dict(
    D="cap or choice", D_rule="C2 and K1/K3 both", N="free", N_rule="not stated",
    cap=["SmolLM-135M", "SmolLM-360M"], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("D+own_cap", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens) Python-Edu : educational Python samples from The Stack (4B tokens) FineWeb-Edu (deduplicated) : educational web samples from FineWeb (220B tokens)"),
        ("D+own_cap", "smollm_blog", "135M and 360M models, each trained on 600B tokens from Smollm-Corpus"),
        ("D", "smollm_blog", "Therefore, we decided to train the 1.7B model on 1 trillion tokens and the 135M and 360M models on 600B tokens, as the performance gains after 400B tokens begin to slow on some benchmarks for these smaller models."),
        ("R_INF", "smollm_blog", "These approaches enable novel applications while dramatically reducing inference costs and improving user privacy."),
        ("R_DEV", "smollm_blog", "Our models are designed to be small and can run locally on various hardware configurations."),
        ("R_DEV", "smollm_blog", "These memory requirements make our models suitable for deployment on a wide range of devices, from smartphones to laptops."),
        ("synthetic", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens)")],
    note="C2: stated unique corpus 28B+4B+220B = 252B; budget 600B exceeds it (about 2.4 passes). K1 (stopped where gains slow) and K3 (1.7B on 1T of the same corpus) are also present, so 'cap or choice' with both quotes. "
         "N free: the iPhone memory remark says the models are suitable for devices, not that a size was set to fit one.")

U["StableLM-alpha"] = dict(
    D="choice", D_rule="K2", N="free", N_rule="not stated", cap=[], R=set(), syn=0,
    ev=[("D", "stablelm_github", "StableLM-Alpha models are trained on a new dataset that builds on"),
        ("D", "stablelm_github", "which contains 1.5 trillion tokens, roughly 3x the size of The Pile."),
        ("D", "stablelm_github", "| 800B | 3,638,525,952 |")],
    note="K2: 800B tokens drawn from a stated 1.5T-token dataset.")

U["Yi"] = dict(
    D="choice", D_rule="K1", N="menu", N_rule="consumer GPU memory (RTX 4090, 24G)", cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("D", "yi_report", "since 34B is smaller than the conventional 70B used by Chinchilla [30] and LLaMA [77], we increase the pretrain data scale to 3.1T tokens to compensate for the decreased compute flops."),
        ("D", "yi_report", "For pretraining, we construct 3.1 trillion tokens of English and Chinese corpora using a cascaded data deduplication and quality filtering pipeline."),
        ("N+R_DEV", "yi_report", "when choosing model scale, the desiderata is to have small enough model that is feasible for inference on consumer-grade hardware like the RTX 4090 where the bounding factor is its limited 24G memory"),
        ("R_INF", "yi_report", "The benefit is from the inference side, as we achieve stronger performance with reduced serving cost")],
    note="K1 read from 'increase the pretrain data scale ... to compensate for the decreased compute flops' (budget set by the compute given the size). "
         "Borderline: the same passage is also 'beyond compute-optimal for inference', which the protocol treats as non-evidence; under that reading D is 'cap or choice' (size only: the 3.1T constructed corpus) and the reading 'cap or choice, menu'. "
         "'we prefer 3T tokens ... over 10T tokens without extensive filtering' is curation, not coded.")

# ---------------------------------------------------------------- singletons
U["B:cerebras/btlm-3b-8k-base"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("R_DEV", "btlm_blog", "Unlike large models like GPT-3 that runs from the cloud, BTLM fits in mobile and edge devices with as little as 3GB of memory"),
        ("R_DEV", "btlm_blog", "While a 3B model would comfortably fit on almost all mobile devices, prior 3B sized models substantially underperformed their 7B counterparts."),
        ("R_INF", "btlm_blog", "BTLM-3B has a 58% smaller memory footprint and 2x faster inference than 7B models.")],
    note="No repetition stated (627B tokens from SlimPajama).")

U["B:tiiuae/falcon-11B"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("R_DEV", "falcon2_news", "these models can run efficiently on just one graphics processing unit (GPU), making them highly scalable, and easy to deploy and integrate into lighter infrastructures like laptops and other devices."),
        ("R_INF", "falcon2_news", "In addition to reducing computing power requirements and meeting sustainability criteria, these models offer enhanced flexibility, seamlessly integrating into edge AI infrastructure")],
    note="R_INF rests on a quoted executive statement about smaller, more efficient models in the release news (borderline).")

U["B:tiiuae/Falcon3-7B-Base"] = dict(
    cap=[], R=set(), syn=0, ev=[],
    note="Not stated. 'Compact and efficient alternatives' singles out Falcon3-1B/3B (not this unit). 14T tokens, no corpus size or repetition stated; no synthetic data documented.")

U["B:google/gemma-2-27b"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("R_DEV+R_INF", "gemma2_blog", "The 27B Gemma 2 model is designed to run inference efficiently at full precision on a single Google Cloud TPU host, NVIDIA A100 80GB Tensor Core GPU , or NVIDIA H100 Tensor Core GPU , significantly reducing costs while maintaining high performance.")],
    note="Only the release blog is saved; no corpus size, repetition or synthetic data stated.")

U["B:h2oai/h2o-danube-1.8b-base"] = dict(
    cap=[], R=set(), syn=0, ev=[],
    note="Not stated: only the model card is saved (the technical report it cites is not among the saved sources); the card gives no budget, corpus, repetition, rationale or synthetic-data statement.")

U["B:m-a-p/neo_7b"] = dict(
    cap=[], R=set(), syn=0, ev=[],
    note="Not stated: README gives 4.5T tokens and a research-transparency purpose only.")

U["B:marin-community/marin-8b-base"] = dict(
    cap=[], R=set(), syn=1,
    ev=[("synthetic", "marin_card", "Jellyfish (First Cooldown)*: Higher quality data (~Dolmino+Fine Math)"),
        ("synthetic", "marin_card", "MathCoder2 Synthetic")],
    note="Synthetic: the Dolmino-Mix-1124 used in the cooldown lists 'MathCoder2 Synthetic'; the generator is not named (coded 1 as for other 'synthetic' sources). No repetition or rationale stated.")

U["B:allenai/OLMo-7B-0424-hf"] = dict(
    cap=[], R=set(), syn=0, ev=[],
    note="No saved source concerns this model beyond passing mentions of OLMo-0424 in olmo2_report (data mixing and curricula, stability); all variables not stated.")

U["B:microsoft/phi-1_5"] = dict(
    cap=["phi-1_5"], R=set(), syn=1,
    ev=[("own_cap", "phi15_card", "Dataset size: 30B tokens"),
        ("own_cap", "phi15_card", "Training tokens: 150B tokens"),
        ("synthetic", "phi15_card", "augmented with a new data source that consists of various NLP synthetic texts")],
    note="Own cap by C2 (150B tokens over a 30B-token dataset, five passes). Stated purpose is research on safety (no listed rationale). Synthetic generator not named.")

U["B:microsoft/phi-2"] = dict(
    cap=["phi-2"], R=set(), syn=1,
    ev=[("own_cap", "phi2_blog", "trained on 1.4T tokens from multiple passes on a mixture of Synthetic and Web datasets for NLP and coding."),
        ("synthetic", "phi2_blog", "Our training data mixture contains synthetic datasets specifically created to teach the model common sense reasoning and general knowledge")],
    note="'With its compact size, Phi-2 is an ideal playground for researchers' is a research rationale, not a listed flag. Synthetic generator not named.")

U["B:HuggingFaceTB/SmolLM3-3B-Base"] = dict(
    cap=[], R={"R_INF"}, syn=1,
    ev=[("R_INF", "smollm3_blog", "Small language models are becoming increasingly important as users seek capable models that can be deployed efficiently."),
        ("R_INF", "smollm3_blog", "SmolLM3 sits in the efficiency sweet spot."),
        ("synthetic", "smollm3_blog", "Introducing FineMath4+, InfiWebMath4+, and MegaMath (including Qwen Q&A, Pro synthetic rewrites, and text-code interleaved blocks)"),
        ("synthetic", "smollm3_blog", "with reasoning traces from R1.")],
    note="No repetition stated for the 11.2T pretraining. R1 traces are in the reasoning mid-training stage; stage-2 MegaMath (Qwen Q&A, synthetic rewrites) is in pretraining.")

U["B:stabilityai/stablelm-2-1_6b"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("R_DEV+R_INF", "stablelm2_report", "This model represents a substantial leap towards making advanced generation capabilities available directly on-device without the computational overhead of larger models."),
        ("R_DEV", "stablelm2_report", "Given its appealing small size, we also provide throughput measurements on a number of edge devices.")],
    note="Table 1 repeats selected sources (up to 6 epochs) and subsamples others; this is upsampling, not own cap. Restruct-v1 instruction data is not described as model-generated.")

U["B:stabilityai/stablelm-3b-4e1t"] = dict(
    cap=["stablelm-3b-4e1t"], R=set(), syn=0,
    ev=[("own_cap", "stablelm_github", "we train on 1 trillion (1T) tokens for 4 epochs following the observations of")],
    note="Purpose is research on repeated tokens; the 'Go smol or go home' inspiration for the token count does not state an inference-cost reason (R_NONE).")

U["B:stabilityai/stablelm-base-alpha-7b-v2"] = dict(
    cap=[], R=set(), syn=0, ev=[],
    note="Not stated: 1.1T tokens (1T stage 1 plus 100B at 4096 context), no corpus size, repetition or rationale.")

U["B:TinyLlama/TinyLlama_v1.1"] = dict(
    cap=["TinyLlama_v1.1"], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("own_cap", "tinyllama_readme", "Combined Dataset Size | Around 950B tokens"),
        ("own_cap", "tinyllama_readme", "Total Tokens During Training | 3 trillion (slightly more than 3 epochs/1430k steps)"),
        ("R_DEV", "tinyllama_readme", "Deployment on edge devices with restricted memory and computational capacities, for functionalities like real-time machine translation without an internet connection"),
        ("R_DEV", "tinyllama_readme", "This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint."),
        ("R_INF", "tinyllama_readme", "The fact that TinyLlama is a relatively small model with grouped query attention means it is also fast during inference.")],
    note="Own cap by Amendment 2: frame budget 2T against the README's ~950B combined dataset (the README describes the 3T v1.0 run, itself 'slightly more than 3 epochs').")

# ---------------------------------------------------------------- size-specific members
U["B:tiiuae/falcon-180B"] = dict(
    cap=["falcon-180B"], R={"R_CO"}, syn=0,
    ev=[("own_cap+R_CO", "falcon_report", "our 40B and 180B models are trained closer to the pretraining optimality suggested by Hoffmann et al. (2022)."),
        ("own_cap", "falcon_report", "At the time, this decision was partially motivated by constraints over data availability: we elected early on to strictly stick to a single epoch of training, and the total stock of RefinedWeb tokens was unknown to us when we started training"),
        ("R_CO", "falcon_report", "Falcon-180B is the first publicly documented GPT-3- sized model to follow the updated scaling law recommendations of Hoffmann et al. (2022), with a total pretraining length of 3,500 billion tokens, without any upsampling.")],
    note="Own cap by Amendment 1 (C3/C1: the budget decision was partly motivated by data available at the start). R_INF not coded: 'nears the performance of PaLM-2-Large at a reduced pretraining and inference cost' is a comparison, and the report says the 180B makes deployment more challenging. "
         "The HF blog's 'less than an epoch' is lower-precedence and not a correction.")

U["B:tiiuae/falcon-40b"] = dict(
    cap=["falcon-40b"], R={"R_CO"}, syn=0,
    ev=[("own_cap+R_CO", "falcon_report", "our 40B and 180B models are trained closer to the pretraining optimality suggested by Hoffmann et al. (2022)."),
        ("own_cap", "falcon_report", "At the time, this decision was partially motivated by constraints over data availability: we elected early on to strictly stick to a single epoch of training, and the total stock of RefinedWeb tokens was unknown to us when we started training")],
    note="Own cap by Amendment 1, as for the 180B.")

U["B:tiiuae/falcon-7b"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("R_INF", "falcon_report", "this is not the case for longer pretraining, offering an axis for decoupling training and inference compute."),
        ("R_INF", "falcon_report", "While our 7B follows a similar idea, our 40B and 180B models are trained closer to the pretraining optimality"),
        ("R_DEV", "falcon_report", "Falcon-7B can efficiently run on consumer hardware (e.g., Apple M2)")],
    note="The data-availability statement concerns the 40B and 180B only; the 7B used a single epoch (no own cap).")

U["B:google/gemma-2b"] = dict(
    cap=[], R={"R_DEV"}, syn=0,
    ev=[("R_DEV", "gemma_report", "a 2 billion param- eter model for CPU and on-device applications."),
        ("R_DEV", "gemma_blog", "Gemma models are capable of running directly on a developer laptop or desktop computer.")],
    note="No corpus size or repetition stated; synthetic data only in instruction tuning.")

U["B:google/gemma-7b"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=0,
    ev=[("R_INF+R_DEV", "gemma_report", "Gemma comes in two sizes: a 7 billion param- eter model for efficient deployment and develop- ment on GPU and TPU"),
        ("R_DEV", "gemma_blog", "Gemma models are capable of running directly on a developer laptop or desktop computer.")],
    note="'Each size is designed to address different computational constraints' supports both flags.")

U["B:h2oai/h2o-danube3-4b-base"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("R_DEV", "danube3_report", "Thanks to its compact architecture, H2O-Danube3 can be efficiently run on a modern smartphone, enabling local inference and rapid processing capabilities even on mobile devices."),
        ("R_INF", "danube3_report", "particularly aiming at efficient inference on consumer hardware and edge devices also allowing for full offline applications."),
        ("synthetic", "danube3_report", "academic texts, synthetic texts and other higher quality textual data is increasing.")],
    note="Synthetic texts in the later pretraining stages; generator not named. No corpus size or repetition stated.")

U["B:h2oai/h2o-danube3-500m-base"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("R_DEV+R_INF", "danube3_report", "we release H2O-Danube3-500M with 500M trainable parameters for edge devices with limited compute or for custom fine-tuning tasks that require low memory footprint or high throughput at low cost."),
        ("R_DEV", "danube3_report", "Thanks to its compact architecture, H2O-Danube3 can be efficiently run on a modern smartphone, enabling local inference and rapid processing capabilities even on mobile devices."),
        ("synthetic", "danube3_report", "academic texts, synthetic texts and other higher quality textual data is increasing.")],
    note="As for the 4B; the stage sizes for the 500M are given separately (2.8T and 1.15T).")

U["B:allenai/OLMo-1B-hf"] = dict(
    cap=["OLMo-1B-hf"], R=set(), syn=0,
    ev=[("own_cap", "olmo_report", "All of our released models have been trained to at least 2T tokens (a single epoch over our training data), and some have been trained beyond that by starting a second epoch over the data with a differ- ent shuffling order."),
        ("own_cap", "olmo_report", "We built our training dataset out of a 2T-token sam- ple from our open dataset, Dolma")],
    note="Own cap by Amendment 2: frame budget 3T against a stated 2T-token single epoch (the report's table gives 2T for the 1B run).")

U["B:allenai/OLMo-7B-hf"] = dict(
    cap=["OLMo-7B-hf"], R=set(), syn=0,
    ev=[("own_cap", "olmo_report", "All of our released models have been trained to at least 2T tokens (a single epoch over our training data), and some have been trained beyond that by starting a second epoch over the data with a differ- ent shuffling order."),
        ("own_cap", "olmo_report", "The checkpoint used for evaluating OLMo-7B is trained until 2.46T tokens on the Dolma (Soldaini et al., 2024) dataset")],
    note="2.46T over a 2T epoch (23 percent beyond, a second epoch started).")

U["B:allenai/OLMo-2-0325-32B"] = dict(
    cap=["OLMo-2-0325-32B"], R=set(), syn=1,
    ev=[("own_cap", "olmo2_32b_blog", "OLMo 2 32B is trained for 1.5 epochs , up to 6T tokens."),
        ("synthetic", "olmo2_report", "We generated approximately 6.5B tokens of synthetic math data from rewritten versions of Tiny-GSM"),
        ("synthetic", "olmo2_report", "Then we pass all of these annotated examples to Qwen2.5-7B-Instruct to be rewritten in the style of MIND"),
        ("synthetic", "olmo2_32b_blog", "For OLMo 2 13B and 32B , we repeat this process using two samples of 100B and 300B tokens.")],
    note="Own cap: 1.5 epochs over OLMo-Mix-1124 (3.9T).")

U["B:allenai/OLMo-2-1124-13B"] = dict(
    cap=["OLMo-2-1124-13B"], R=set(), syn=1,
    ev=[("own_cap", "olmo2_32b_blog", "OLMo 2 13B is trained for 1.3 epochs , up to 5T tokens"),
        ("own_cap", "olmo2_blog", "OLMo 2 13B is trained for 1.2 epochs up to 5T tokens."),
        ("synthetic", "olmo2_report", "We generated approximately 6.5B tokens of synthetic math data from rewritten versions of Tiny-GSM"),
        ("synthetic", "olmo2_report", "Then we pass all of these annotated examples to Qwen2.5-7B-Instruct to be rewritten in the style of MIND"),
        ("synthetic", "olmo2_32b_blog", "For OLMo 2 13B and 32B , we repeat this process using two samples of 100B and 300B tokens.")],
    note="Own cap: 1.2 to 1.3 epochs over OLMo-Mix-1124 (the two blogs differ on the decimal; both exceed one epoch).")

U["B:Qwen/Qwen-7B"] = dict(
    cap=[], R=set(), syn=0, ev=[],
    note="Not stated: 2.4T tokens (updated from 2.2T) on a dataset 'of up to 3 trillion tokens'; no repetition; the 'developer- or application-friendly scales' remark is not coded (see Qwen#2).")

U["B:Qwen/Qwen2-0.5B"] = dict(
    cap=[], R={"R_DEV"}, syn=1,
    ev=[("R_DEV", "qwen2_report", "The smaller models, specifically Qwen2-0.5B and Qwen2-1.5B, are designed for easy deployment on portable devices such as smartphones, earphones, and smart glasses."),
        ("synthetic", "qwen2_report", "Moreover, these models are utilized to synthesize high-quality pre-training data.")],
    note="12T budget equals the 12T dataset (no repetition, own cap 0).")

U["B:HuggingFaceTB/SmolLM-1.7B"] = dict(
    cap=["SmolLM-1.7B"], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("own_cap", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens) Python-Edu : educational Python samples from The Stack (4B tokens) FineWeb-Edu (deduplicated) : educational web samples from FineWeb (220B tokens)"),
        ("own_cap", "smollm_blog", "1.7B model, trained on 1T tokens from Smollm-Corpus"),
        ("R_INF", "smollm_blog", "These approaches enable novel applications while dramatically reducing inference costs and improving user privacy."),
        ("R_DEV", "smollm_blog", "These memory requirements make our models suitable for deployment on a wide range of devices, from smartphones to laptops."),
        ("synthetic", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens)")],
    note="Own cap by C2: 1T tokens over a stated 252B-token corpus (about 4 passes).")

U["B:HuggingFaceTB/SmolLM2-1.7B"] = dict(
    cap=["SmolLM2-1.7B"], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("own_cap", "smollm2_report", "When building SmolLM2, we trained on 11 trillion tokens (approximately two epochs on our collected datasets)"),
        ("R_INF", "smollm2_report", "the resulting performance gains and reduced inference costs make ex- tended training a worthwhile trade-off"),
        ("R_INF+R_DEV", "smollm2_report", "These small LMs are computationally inexpensive and can be run on a wider range of devices (e.g. mobile phones) while providing satisfactory performance on many important tasks."),
        ("R_DEV", "smollm2_card", "They are capable of solving a wide range of tasks while being lightweight enough to run on-device."),
        ("synthetic", "smollm2_report", "Cosmopedia v2 (Allal et al., 2024) at 4%, which provides 30B tokens of high- quality synthetic textbooks, blog posts, and stories."),
        ("synthetic", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens)")],
    note="Own cap: approximately two epochs over the collected datasets.")

U["B:HuggingFaceTB/SmolLM2-135M"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("R_INF+R_DEV", "smollm2_report", "These small LMs are computationally inexpensive and can be run on a wider range of devices (e.g. mobile phones) while providing satisfactory performance on many important tasks."),
        ("R_DEV", "smollm2_card", "They are capable of solving a wide range of tasks while being lightweight enough to run on-device."),
        ("synthetic", "smollm2_report", "We incorporated Stack-Edu from the start, alongside InfiMM-WebMath, FineMath, and Cosmopedia."),
        ("synthetic", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens)")],
    note="No epoch count or unique-corpus size is stated for the 2T single-stage mix of the small models (own cap 0).")

U["B:HuggingFaceTB/SmolLM2-360M"] = dict(
    cap=[], R={"R_INF", "R_DEV"}, syn=1,
    ev=[("R_INF+R_DEV", "smollm2_report", "These small LMs are computationally inexpensive and can be run on a wider range of devices (e.g. mobile phones) while providing satisfactory performance on many important tasks."),
        ("R_DEV", "smollm2_card", "They are capable of solving a wide range of tasks while being lightweight enough to run on-device."),
        ("synthetic", "smollm2_report", "We incorporated Stack-Edu from the start, alongside InfiMM-WebMath, FineMath, and Cosmopedia."),
        ("synthetic", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens)")],
    note="No epoch count or unique-corpus size is stated for the 4T single-stage mix (own cap 0).")

READING = {("choice", "free"): "choice", ("choice", "menu"): "menu", ("cap", "free"): "cap", ("cap", "menu"): "cap+menu",
           ("cap or choice", "free"): "cap or choice", ("cap or choice", "menu"): "cap or choice, menu"}


def main():
    F = pd.read_csv(os.path.join(PROC, "coding_frame.csv"))
    man = json.load(open(os.path.join(RAW, "manifest.json")))
    miss = sorted(set(F["unit"]) - set(U))
    extra = sorted(set(U) - set(F["unit"]))
    if miss or extra:
        sys.exit(f"units missing {miss} extra {extra}")
    bad = []
    rows = []
    kmax = max(len(u["ev"]) for u in U.values())
    for _, f in F.iterrows():
        u = U[f["unit"]]
        members = [m.strip() for m in f["members"].split(";")]
        for m in u["cap"]:
            if m not in members:
                bad.append(f"{f['unit']}: own-cap member {m} not in unit")
        r = dict(unit=f["unit"], unit_kind=f["unit_kind"], members=f["members"])
        if f["unit_kind"] == "common-budget":
            r.update(D_margin=u["D"], D_rule=u["D_rule"], N_margin=u["N"], N_rule=u["N_rule"],
                     reading=READING[(u["D"], u["N"])])
        else:
            r.update(D_margin="", D_rule="", N_margin="", N_rule="", reading="")
        r.update(own_cap=int(bool(u["cap"])), own_cap_members="; ".join(u["cap"]),
                 R_CO=int("R_CO" in u["R"]), R_INF=int("R_INF" in u["R"]), R_DEV=int("R_DEV" in u["R"]),
                 R_NONE=int(not u["R"]), synthetic=u["syn"])
        for i in range(kmax):
            if i < len(u["ev"]):
                role, key, q = u["ev"][i]
                if not found(key, q):
                    bad.append(f"{f['unit']}: quote {i + 1} not found in {key}: {q[:70]}")
                if len(q.split()) > 70:
                    bad.append(f"{f['unit']}: quote {i + 1} has {len(q.split())} words")
                if "\u2014" in q:
                    bad.append(f"{f['unit']}: quote {i + 1} contains an em dash")
                mm = man.get(key, {})
                r[f"quote_{i + 1}"], r[f"source_{i + 1}"] = q, key
                r[f"url_{i + 1}"], r[f"retrieved_{i + 1}"] = mm.get("final_url", mm.get("url", "")), mm.get("retrieved", "")
                r[f"quote_role_{i + 1}"] = role
            else:
                for c in ("quote", "source", "url", "retrieved", "quote_role"):
                    r[f"{c}_{i + 1}"] = ""
        r.update(protocol_version=PROTOCOL, coder="B", note=u["note"])
        # every non-zero code carries a quote of that role
        roles = " ".join(e[0] for e in u["ev"])
        need = []
        if f["unit_kind"] == "common-budget":
            need += ["D"] + (["N"] if u["N"] == "menu" else [])
        need += (["own_cap"] if u["cap"] else []) + sorted(u["R"]) + (["synthetic"] if u["syn"] else [])
        for n in need:
            if n not in roles:
                bad.append(f"{f['unit']}: code {n} has no quote")
        rows.append(r)
    if bad:
        print("\n".join(bad))
        sys.exit(1)
    cols = (["unit", "unit_kind", "members", "D_margin", "D_rule", "N_margin", "N_rule", "reading", "own_cap",
             "own_cap_members", "R_CO", "R_INF", "R_DEV", "R_NONE", "synthetic"]
            + [f"quote_{i + 1}" for i in range(kmax)] + [f"source_{i + 1}" for i in range(kmax)]
            + [f"url_{i + 1}" for i in range(kmax)] + [f"retrieved_{i + 1}" for i in range(kmax)]
            + [f"quote_role_{i + 1}" for i in range(kmax)] + ["protocol_version", "coder", "note"])
    out = pd.DataFrame(rows)[cols]
    for c in out.columns:
        if out[c].dtype == object and out[c].astype(str).str.contains("\u2014").any():
            print("em dash in column", c)
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROC, "readings_coderB.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}: {len(out)} units, {sum(len(u['ev']) for u in U.values())} quotes, all verified")


if __name__ == "__main__":
    main()
