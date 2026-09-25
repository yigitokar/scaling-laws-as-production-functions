"""coderA.py -- coder A's codes for the 49 budget-level decision units, under paper/notes/rb5_reading_protocol.md
(version 1 with Amendments 1-3). Every quote is verbatim from the saved source data/raw/rb2_decisions/<key>.txt and is
checked by verify_readings.py (whitespace removed, case folded) before the file is written.

Evidence roles: D (token margin of a common budget), N (size margin), CAP (a member's own binding cap), CO / INF / DEV
(sizing rationales R-CO, R-INF, R-DEV), SYN (synthetic-data flag), NONEV (recorded because the round-2 coding used it, but
not evidence under the protocol).

Fields: kind; D (choice / cap / cap or choice / ''), D_rule; N (free / menu / ''), N_rule; own_cap (0/1; for a common
budget the members with an own cap are listed); R_CO, R_INF, R_DEV; syn (0/1); ev = [(role, key, quote)]; note.
"""
from __future__ import annotations

U = {}

# ================================================================ common-budget units (17)
U["Apertus"] = dict(
    kind="common-budget", D="choice", D_rule="K2", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("D", "apertus_report", "We train the model on 15T tokens (∼0.3T masked due to Goldfish Loss) divided into five stages"),
        ("D", "apertus_report", "For all other languages, we subsample documents at random."),
        ("SYN", "apertus_report", "For task data we rely on EuroBlocks-SFT-Synthetic-112427 (Martins et al., 2025) for multilingual instruction and task data")],
    note="K2: the multilingual web data are a random subsample of a larger usable pool. Synthetic by label "
         "(EuroBlocks-SFT-Synthetic in the pretraining task data); the saved report does not describe how it was generated.")
U["DeepSeek-LLM"] = dict(
    kind="common-budget", D="cap", D_rule="C1", N="menu", N_rule="standard configurations", own_cap=0, R_CO=1, R_INF=0,
    R_DEV=0, syn=0,
    ev=[("D", "deepseek_llm_report", "we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding."),
        ("N", "deepseek_llm_report", "in two prevalent used open-source configurations, 7B and 67B."),
        ("CO", "deepseek_llm_report", "Under the guidance of our scaling laws, we build from scratch open-source large language models")],
    note="C1: the budget is the whole dataset available at the time; sizes are standard configurations.")
U["Granite-3.0"] = dict(
    kind="common-budget", D="cap or choice", D_rule="size only", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=1, R_DEV=1,
    syn=1,
    ev=[("D", "granite30_report", "Dense Models: 2B and 8B parameter models, trained on 12 trillion tokens in total."),
        ("DEV", "granite30_report", "For accessibility, being able to fine-tune a pretrained LLM for on-premise use requires models with lower hardware requirements."),
        ("DEV", "granite30_report", "including the potential to be run on constrained compute resources."),
        ("INF", "granite30_announcement", "deliver state-of-the-art performance relative to model size while maximizing safety, speed and cost-efficiency for enterprise use cases."),
        ("SYN", "granite30_report", "mixed with a small amount of high-quality open-source and synthetic corpora with permissive licenses."),
        ("NONEV", "granite30_report", "Mixture-of-Expert (MoE) Models: Sparse 1B and 3B MoE models, with 400M and 800M activated parameters respectively, trained on 10 trillion tokens in total.")],
    note="Only the dense budget is stated; the MoE siblings trained on fewer tokens (10T), which does not show that the "
         "dense budget was below the pool (K3 needs more tokens of the same data). Synthetic corpora in the stage-2 mix.")
U["Llama#1"] = dict(
    kind="common-budget", D="choice", D_rule="K3; K2", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("D", "llama1_report", "LLaMA-33B and LLaMA-65B were trained on 1.4T tokens. The smaller models were trained on 1.0T tokens."),
        ("D", "llama1_report", "Overall, our entire training dataset contains roughly 1.4T tokens after tokenization."),
        ("INF", "llama1_report", "However, this objective disregards the inference budget, which becomes critical when serving a language model at scale."),
        ("DEV", "llama1_report", "We believe that this model will help democratize the access and study of LLMs, since it can be run on a single GPU.")],
    note="The 7B and 13B used 1.0T of a 1.4T dataset on which the 33B and 65B trained longer. The single-GPU statement "
         "concerns LLaMA-13B (Amendment 3).")
U["Llama#2"] = dict(
    kind="common-budget", D="cap or choice", D_rule="size only", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=1, R_DEV=0,
    syn=0,
    ev=[("D", "llama1_report", "Overall, our entire training dataset contains roughly 1.4T tokens after tokenization."),
        ("D", "llama1_report", "LLaMA-33B and LLaMA-65B were trained on 1.4T tokens."),
        ("INF", "llama1_report", "However, this objective disregards the inference budget, which becomes critical when serving a language model at scale."),
        ("NONEV", "llama1_report", "with the exception of the Wikipedia and Books domains, over which we perform ap- proximately two epochs.")],
    note="Budget equals the stated dataset; no statement that more data were unavailable or that the budget was chosen "
         "below it. Two epochs of Wikipedia and books is upsampling of selected sources (not repetition).")
U["Llama-2"] = dict(
    kind="common-budget", D="choice", D_rule="K1", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0,
    ev=[("D", "llama2_report", "We trained on 2 trillion tokens of data as this provides a good performance–cost trade-off"),
        ("NONEV", "llama2_report", "We observe that after pretraining on 2T Tokens, the models still did not show any sign of saturation.")],
    note="K1: a stated performance-cost trade-off.")
U["Llama-3 herd"] = dict(
    kind="common-budget", D="cap or choice", D_rule="size only", N="free", N_rule="", own_cap=0, R_CO=1, R_INF=1, R_DEV=0,
    syn=0,
    ev=[("D", "llama3_report", "We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2."),
        ("D", "llama3_blog", "Llama 3 is pretrained on over 15T tokens that were all collected from publicly available sources."),
        ("CO", "llama3_report", "While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal."),
        ("INF", "llama3_report", "The resulting models perform better than compute-optimal models at the same inference budget."),
        ("INF", "llama3_blog", "smaller models are generally preferred because they are much more efficient during inference."),
        ("NONEV", "llama3_blog", "Both our 8B and 70B parameter models continued to improve log-linearly after we trained them on up to 15T tokens.")],
    note="Report and release post give the corpus size only; 'still improving' is a positive marginal product of data, "
         "true under every reading (protocol 5.1). The 405B at 15.6T exceeds 'about 15T' by 4 percent (not C2).")
U["MPT"] = dict(
    kind="common-budget", D="choice", D_rule="K2", N="menu", N_rule="single-GPU deployment", own_cap=0, R_CO=0, R_INF=1,
    R_DEV=1, syn=0,
    ev=[("D", "mpt7b_blog", "the model was pre-trained on 1T tokens sampled according to this mix."),
        ("D", "mpt30b_blog", "we used 1T tokens from the same 10 data subsets as the MPT-7B model ( Table 1 ), but in slightly different proportions."),
        ("D", "mpt30b_blog", "we first pre-trained on 1T tokens using sequences that were 2k tokens long, and continued training for an additional 50B tokens using sequences that were 8k tokens long."),
        ("N", "mpt30b_blog", "The size of MPT-30B was also specifically chosen to make it easy to deploy on a single GPU"),
        ("INF", "mpt30b_blog", "this necessitates 2+ GPUs, which increases the minimum inference system cost.")],
    note="K2 rests on sampling from the ten source subsets with changed proportions at the same budget; the saved texts "
         "do not give the subsets' sizes, so this is the weakest choice code. MPT-30B's budget is 1.05T (1T at 2k context "
         "plus 50B at 8k).")
U["OLMo-2#1"] = dict(
    kind="common-budget", D="choice", D_rule="K1; K3", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("D", "olmo2_report", "Similar to OLMo 2 7B, we use 2000 steps of warmup, set the schedule to 5 trillion tokens but truncate at the 4 trillion mark."),
        ("D", "olmo2_32b_blog", "OLMo 2 32B is trained for 1.5 epochs , up to 6T tokens."),
        ("SYN", "olmo2_report", "Having generated the problems, we then generate multi-step math solutions using GPT-4o.")],
    note="Schedule planned for 5T and truncated at 4T (K1); the 13B and 32B trained on more tokens of the same mix (K3). "
         "The 7B made about one pass ('approximately one epoch'), which is not repetition.")
U["Olmo-3"] = dict(
    kind="common-budget", D="choice", D_rule="K2; K1", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=1, syn=1,
    ev=[("D", "olmo3_report", "Composition of Dolma 3 Mix including our 9T pool of data, the 6T mix we used for final model training, and the 150B mix we used for experimentation."),
        ("D", "olmo3_report", "The learning rate schedule is a cosine schedule over one epoch (5.93T tokens), truncated at 5.5T tokens."),
        ("DEV", "olmo3_blog", "Olmo 3 is a family of compact, dense models at 7 billion and 32 billion parameters that can run on everything from laptops to research clusters."),
        ("DEV", "olmo3_blog", "32B models are big enough to support strong, competitive performance, but still small enough that a wide audience can fine-tune and deploy them on accessible hardware."),
        ("SYN", "olmo3_report", "We collect Wikipedia passages and prompted Qwen2.5 32B Instruct to generate QA pairs")],
    note="6T mix drawn from a 9T pool (K2); the 32B schedule truncated (K1).")
U["Qwen#2"] = dict(
    kind="common-budget", D="cap or choice", D_rule="size only", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=0,
    syn=0,
    ev=[("D", "qwen_report", "Finally, we have built a dataset of up to 3 trillion tokens."),
        ("D", "qwen_readme", "which are trained on 3T tokens and support 32k context")],
    note="Qwen-14B (3.0T) and Qwen-72B (3T) trained on the whole stated dataset; no statement of availability or choice.")
U["Qwen2#1"] = dict(
    kind="common-budget", D="choice", D_rule="K1; K2", N="menu", N_rule="portable devices (1.5B)", own_cap=0, R_CO=0,
    R_INF=0, R_DEV=1, syn=1,
    ev=[("D", "qwen2_report", "An attempt to further relax the quality threshold resulted in a 12 trillion token dataset."),
        ("D", "qwen2_report", "Considering training costs, we opted to use the higher-quality 7 trillion token dataset for training larger models, leaving further exploration for future model iterations."),
        ("N", "qwen2_report", "The smaller models, specifically Qwen2-0.5B and Qwen2-1.5B, are designed for easy deployment on portable devices such as smartphones, earphones, and smart glasses."),
        ("DEV", "qwen2_report", "Conversely, the larger models cater to deployment across GPUs of varying scales."),
        ("SYN", "qwen2_report", "Moreover, these models are utilized to synthesize high-quality pre-training data.")],
    note="A larger (12T) dataset existed and 7T was chosen for cost (K1, K2); the 1.5B's size targets portable devices.")
U["Qwen2.5"] = dict(
    kind="common-budget", D="cap or choice", D_rule="size only", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=1, R_DEV=1,
    syn=1,
    ev=[("D", "qwen25_blog", "all models are pretrained on our latest large-scale dataset, encompassing up to 18 trillion tokens."),
        ("INF", "qwen25_report", "Qwen2.5 brings back the 3B, 14B, and 32B models, which are more cost-effective for resource-limited scenarios"),
        ("DEV", "qwen25_report", "These enhancements make them particularly well-suited for edge-side applications in highly resource-constrained environments."),
        ("SYN", "qwen25_report", "To generate high-quality synthetic data, particularly in mathematics, code, and knowledge domains, we leverage both Qwen2-72B-Instruct")],
    note="Corpus size only ('up to 18 trillion'); the tokens each member processed are not stated (R1 minor 12), so the "
         "common-budget classification itself is unaudited.")
U["Qwen3"] = dict(
    kind="common-budget", D="cap or choice", D_rule="size only", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=1,
    syn=1,
    ev=[("D", "qwen3_report", "All Qwen3 models are trained on a large and diverse dataset consisting of 119 languages and dialects, with a total of 36 trillion tokens."),
        ("DEV", "qwen3_report", "For edge-side models, we take similar-sized Qwen2.5, Llama-3, and Gemma-3 base models as the baselines."),
        ("SYN", "qwen3_report", "models to synthesize trillions of text tokens in different formats, including textbooks, question-answering, instructions, and code snippets"),
        ("NONEV", "qwen3_report", "To efficiently expand the training data, we employ a multi-modal approach: Qwen2.5-VL (Bai et al., 2025) is finetuned to extract text from extensive PDF documents.")],
    note="Corpus size only. Data expansion (PDF extraction, synthesis) is corpus-building effort, not evidence that the "
         "corpus bound (protocol 5.1; R1 minor 12). The 0.6B-8B are called edge-side models.")
U["SmolLM#1"] = dict(
    kind="common-budget", D="cap or choice", D_rule="C2 and K1", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=1, R_DEV=1,
    syn=1,
    ev=[("D", "smollm_blog", "135M and 360M models, each trained on 600B tokens from Smollm-Corpus"),
        ("D", "smollm_blog", "Cosmopedia v2 : A collection of synthetic textbooks and stories generated by Mixtral (28B tokens) Python-Edu : educational Python samples from The Stack (4B tokens)"),
        ("D", "smollm_blog", "In Smollm-Corpus we include 220B deduplicated tokens from FineWeb."),
        ("D", "smollm_blog", "Therefore, we decided to train the 1.7B model on 1 trillion tokens and the 135M and 360M models on 600B tokens, as the performance gains after 400B tokens begin to slow on some benchmarks for these smaller models."),
        ("DEV", "smollm_blog", "Our models are designed to be small and can run locally on various hardware configurations."),
        ("INF", "smollm_blog", "These approaches enable novel applications while dramatically reducing inference costs and improving user privacy."),
        ("SYN", "smollm_blog", "A collection of synthetic textbooks and stories generated by Mixtral")],
    note="C2: 600B tokens from a corpus of 28 + 4 + 220 = 252B unique tokens (about 2.4 passes); K1: the length was set "
         "where gains slowed. Both kinds of evidence, so 'cap or choice' (protocol 5.1); the repetition itself makes the "
         "share a lower bound.")
U["StableLM-alpha"] = dict(
    kind="common-budget", D="choice", D_rule="K2", N="free", N_rule="", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0,
    ev=[("D", "stablelm_github", "StableLM-Alpha models are trained on a new dataset that builds on [The Pile](https://pile.eleuther.ai/), which contains 1.5 trillion tokens, roughly 3x the size of The Pile."),
        ("D", "stablelm_github", "| 3B | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-3b/) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/) | 800B | 3,638,525,952 |")],
    note="800B tokens of a 1.5T dataset (K2).")
U["Yi"] = dict(
    kind="common-budget", D="choice", D_rule="K1; K2", N="menu", N_rule="24G consumer GPU (34B)", own_cap=0, R_CO=0,
    R_INF=1, R_DEV=1, syn=0,
    ev=[("D", "yi_report", "we increase the pretrain data scale to 3.1T tokens to compensate for the decreased compute flops."),
        ("D", "yi_report", "we prefer 3T tokens over sophasticated engineering over 10T tokens without extensive filtering."),
        ("N", "yi_report", "when choosing model scale, the desiderata is to have small enough model that is feasible for inference on consumer-grade hardware like the RTX 4090 where the bounding factor is its limited 24G memory"),
        ("INF", "yi_report", "The benefit is from the inference side, as we achieve stronger performance with reduced serving cost"),
        ("NONEV", "yi_report", "Our model is trained on 3.1T tokens, yet we belive with larger amount of data, we can continue improve the model performance (i.e., the model have not saturated at 3.1T)")],
    note="The budget is stated as set for compute (K1) and preferred over 10T unfiltered tokens (K2); 'not saturated' is "
         "not cap evidence. The 34B's size was set by a 24G memory limit (menu). Report gives 3.1T; the analysis keeps 3T "
         "(model card; X8(m)).")

# ================================================================ size-specific members (17) and singletons (15)
_FALCON_CAP = ("CAP", "falcon_report", "At the time, this decision was partially motivated by constraints over data availability: we elected early on to strictly stick to a single epoch of training, and the total stock of RefinedWeb tokens was unknown to us when we started training")
_FALCON_CO = ("CO", "falcon_report", "While our 7B follows a similar idea, our 40B and 180B models are trained closer to the pretraining optimality suggested by Hoffmann et al. (2022).")
U["B:tiiuae/falcon-180B"] = dict(kind="size-specific member", own_cap=1, own_rule="C1/C3 (Amendment 1)", R_CO=1, R_INF=0,
                                 R_DEV=0, syn=0, ev=[_FALCON_CAP, _FALCON_CO],
                                 note="The 40B and 180B budgets were partly set by the data available at the time.")
U["B:tiiuae/falcon-40b"] = dict(kind="size-specific member", own_cap=1, own_rule="C1/C3 (Amendment 1)", R_CO=1, R_INF=0,
                                R_DEV=0, syn=0, ev=[_FALCON_CAP, _FALCON_CO], note="As Falcon-180B.")
U["B:tiiuae/falcon-7b"] = dict(
    kind="size-specific member", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("INF", "falcon_report", "Under that paradigm, inference costs can become predominant, shadowing pretraining costs"),
        ("INF", "falcon_report", "While our 7B follows a similar idea"),
        ("DEV", "falcon_report", "Falcon-7B can efficiently run on consumer hardware (e.g., Apple M2)")],
    note="The 7B follows the idea of decoupling training and inference compute by longer training.")
_GEMMA = "Each size is designed to address different compu- tational"
U["B:google/gemma-2b"] = dict(
    kind="size-specific member", own_cap=0, R_CO=0, R_INF=0, R_DEV=1, syn=0,
    ev=[("DEV", "gemma_report", "and a 2 billion param- eter model for CPU and on-device applications."),
        ("DEV", "gemma_report", _GEMMA)], note="")
U["B:google/gemma-7b"] = dict(
    kind="size-specific member", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("INF", "gemma_report", "a 7 billion param- eter model for efficient deployment and develop- ment on GPU and TPU"),
        ("DEV", "gemma_report", _GEMMA)], note="")
_DANUBE3 = [("INF", "danube3_report", "particularly aiming at efficient inference on consumer hardware and edge devices also allowing for full offline applications."),
            ("DEV", "danube3_report", "H2O-Danube3 can be efficiently run on a modern smartphone, enabling local inference"),
            ("SYN", "danube3_report", "synthetic texts and other higher quality textual data is increasing.")]
U["B:h2oai/h2o-danube3-4b-base"] = dict(kind="size-specific member", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=1,
                                        ev=_DANUBE3, note="Synthetic by label ('synthetic texts' in later stages).")
U["B:h2oai/h2o-danube3-500m-base"] = dict(kind="size-specific member", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=1,
                                          ev=_DANUBE3, note="As the 4B.")
_OLMO1_EP = ("CAP", "olmo_report", "All of our released models have been trained to at least 2T tokens (a single epoch over our training data), and some have been trained beyond that by starting a second epoch over the data")
U["B:allenai/OLMo-1B-hf"] = dict(kind="size-specific member", own_cap=1, own_rule="C2 (Amendment 2: frame budget 3T)",
                                 R_CO=0, R_INF=0, R_DEV=0, syn=0, ev=[_OLMO1_EP],
                                 note="The saved report's table gives 2T for the 1B; the released OLMo-1B-hf budget in the "
                                      "frame is 3T, 1.5 passes over the 2T corpus.")
U["B:allenai/OLMo-7B-hf"] = dict(
    kind="size-specific member", own_cap=1, own_rule="C2", R_CO=0, R_INF=0, R_DEV=0, syn=0,
    ev=[_OLMO1_EP, ("CAP", "olmo_report", "The checkpoint used for evaluating OLMo-7B is trained until 2.46T tokens on the Dolma (Soldaini et al., 2024) dataset")],
    note="2.46T tokens over a 2T epoch (1.23 passes).")
_OLMO2_SYN = ("SYN", "olmo2_report", "Having generated the problems, we then generate multi-step math solutions using GPT-4o.")
U["B:allenai/OLMo-2-1124-13B"] = dict(
    kind="size-specific member", own_cap=1, own_rule="C2", R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("CAP", "olmo2_blog", "OLMo 2 7B is trained for approximately one epoch on this dataset, while OLMo 2 13B is trained for 1.2 epochs up to 5T tokens."),
        _OLMO2_SYN], note="1.2 epochs of the stage-1 mix (1.3 in the 32B release post).")
U["B:allenai/OLMo-2-0325-32B"] = dict(
    kind="size-specific member", own_cap=1, own_rule="C2", R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("CAP", "olmo2_32b_blog", "OLMo 2 32B is trained for 1.5 epochs , up to 6T tokens."), _OLMO2_SYN],
    note="1.5 epochs of the stage-1 mix.")
U["B:Qwen/Qwen-7B"] = dict(kind="size-specific member", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0, ev=[],
                           note="Trained on 2.4T of a dataset of up to 3T (the 14B trained on 3.0T); no rationale stated.")
U["B:Qwen/Qwen2-0.5B"] = dict(
    kind="size-specific member", own_cap=0, R_CO=0, R_INF=0, R_DEV=1, syn=1,
    ev=[("DEV", "qwen2_report", "The smaller models, specifically Qwen2-0.5B and Qwen2-1.5B, are designed for easy deployment on portable devices such as smartphones, earphones, and smart glasses."),
        ("SYN", "qwen2_report", "Moreover, these models are utilized to synthesize high-quality pre-training data.")],
    note="Trained on the 12T dataset (one pass).")
U["B:HuggingFaceTB/SmolLM-1.7B"] = dict(
    kind="size-specific member", own_cap=1, own_rule="C2", R_CO=0, R_INF=1, R_DEV=1, syn=1,
    ev=[("CAP", "smollm_blog", "1.7B model, trained on 1T tokens from Smollm-Corpus"),
        ("CAP", "smollm_blog", "In Smollm-Corpus we include 220B deduplicated tokens from FineWeb."),
        ("DEV", "smollm_blog", "Our models are designed to be small and can run locally on various hardware configurations."),
        ("INF", "smollm_blog", "These approaches enable novel applications while dramatically reducing inference costs and improving user privacy."),
        ("SYN", "smollm_blog", "A collection of synthetic textbooks and stories generated by Mixtral")],
    note="1T tokens from a 252B-token corpus (about 4 passes).")
_SMOL2 = [("DEV", "smollm2_report", "These small LMs are computationally inexpensive and can be run on a wider range of devices (e.g. mobile phones) while providing satisfactory performance on many important tasks."),
          ("INF", "smollm2_report", "This enormity results in enormous computa- tional costs, both during training and for inference"),
          ("SYN", "smollm2_report", "which provides 30B tokens of high- quality synthetic textbooks, blog posts, and stories.")]
U["B:HuggingFaceTB/SmolLM2-1.7B"] = dict(
    kind="size-specific member", own_cap=1, own_rule="C2", R_CO=0, R_INF=1, R_DEV=1, syn=1,
    ev=[("CAP", "smollm2_report", "we trained on 11 trillion tokens (approximately two epochs on our collected datasets)")] + _SMOL2,
    note="About two passes over the collected data.")
U["B:HuggingFaceTB/SmolLM2-135M"] = dict(
    kind="size-specific member", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=1,
    ev=[("NONEV", "smollm2_report", "SmolLM2-135M (135M parameters, trained on 2T tokens)")] + _SMOL2,
    note="No repetition stated for the 2T budget.")
U["B:HuggingFaceTB/SmolLM2-360M"] = dict(
    kind="size-specific member", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=1,
    ev=[("NONEV", "smollm2_report", "SmolLM2-360M (360M parameters, trained on 4T tokens)")] + _SMOL2,
    note="No repetition stated for the 4T budget.")
# singletons
U["B:cerebras/btlm-3b-8k-base"] = dict(
    kind="singleton", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("DEV", "btlm_blog", "BTLM fits in mobile and edge devices with as little as 3GB of memory"),
        ("INF", "btlm_blog", "BTLM-3B has a 58% smaller memory footprint and 2x faster inference than 7B models."),
        ("NONEV", "btlm_blog", "The model was trained on 627B tokens from the SlimPajama dataset")],
    note="627B tokens of the 627B-token SlimPajama (one pass).")
U["B:tiiuae/falcon-11B"] = dict(
    kind="singleton", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("INF", "falcon2_news", "developers are recognizing the myriad benefits of smaller, more efficient models. In addition to reducing computing power requirements"),
        ("DEV", "falcon2_news", "these models can run efficiently on just one graphics processing unit (GPU), making them highly scalable, and easy to deploy and integrate into lighter infrastructures like laptops")],
    note="")
U["B:tiiuae/Falcon3-7B-Base"] = dict(kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0,
                                     ev=[("NONEV", "falcon3_blog", "The Falcon3-7B-Base is trained on the largest amount of data ensuring comprehensive coverage of concepts and knowledge")],
                                     note="No sizing rationale for the 7B (the compact 1B and 3B are distilled).")
U["B:google/gemma-2-27b"] = dict(
    kind="singleton", own_cap=0, R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("DEV", "gemma2_blog", "The 27B Gemma 2 model is designed to run inference efficiently at full precision on a single Google Cloud TPU host"),
        ("INF", "gemma2_blog", "And that’s now achievable on a single NVIDIA H100 Tensor Core GPU or TPU host, significantly reducing deployment costs.")],
    note="")
U["B:h2oai/h2o-danube-1.8b-base"] = dict(kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0, ev=[],
                                         note="Only the model card is saved; it states no rationale or data detail.")
U["B:m-a-p/neo_7b"] = dict(kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0, ev=[],
                           note="The saved README states no rationale or repetition.")
U["B:marin-community/marin-8b-base"] = dict(
    kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("SYN", "marin_card", "[MathCoder2 Synthetic](https://arxiv.org/abs/2310.03731)")],
    note="Dolmino-Mix-1124 (with MathCoder2 Synthetic) enters the cooldown data.")
U["B:allenai/OLMo-7B-0424-hf"] = dict(kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0, ev=[],
                                      note="No saved source for OLMo 1.7; every variable 'not stated'.")
U["B:microsoft/phi-1_5"] = dict(
    kind="singleton", own_cap=1, own_rule="C2", R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("CAP", "phi15_card", "Dataset size: 30B tokens"), ("CAP", "phi15_card", "Training tokens: 150B tokens"),
        ("SYN", "phi15_card", "augmented with a new data source that consists of various NLP synthetic texts."),
        ("NONEV", "phi15_card", "The intention behind crafting this open-source model is to provide the research community with a non-restricted small model")],
    note="150B tokens from a 30B-token dataset (5 passes). Research rationale only.")
U["B:microsoft/phi-2"] = dict(
    kind="singleton", own_cap=1, own_rule="C2", R_CO=0, R_INF=0, R_DEV=0, syn=1,
    ev=[("CAP", "phi2_blog", "trained on 1.4T tokens from multiple passes on a mixture of Synthetic and Web datasets for NLP and coding."),
        ("SYN", "phi2_blog", "Our training data mixture contains synthetic datasets specifically created to teach the model common sense reasoning and general knowledge")],
    note="Multiple passes stated.")
U["B:HuggingFaceTB/SmolLM3-3B-Base"] = dict(
    kind="singleton", own_cap=0, R_CO=0, R_INF=1, R_DEV=0, syn=1,
    ev=[("INF", "smollm3_blog", "Small language models are becoming increasingly important as users seek capable models that can be deployed efficiently."),
        ("SYN", "smollm3_blog", "MegaMath (including Qwen Q&A, Pro synthetic rewrites, and text-code interleaved blocks)")],
    note="")
U["B:stabilityai/stablelm-2-1_6b"] = dict(
    kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=1, syn=0,
    ev=[("DEV", "stablelm2_report", "Given its appealing small size, we also provide throughput measurements on a number of edge devices."),
        ("NONEV", "stablelm2_report", "Similarly, if epochs are below one, the number of tokens shown is a subset of the total dataset.")],
    note="Selected sources upsampled (several epochs) and others subsampled; not repetition of the corpus.")
U["B:stabilityai/stablelm-3b-4e1t"] = dict(
    kind="singleton", own_cap=1, own_rule="C2", R_CO=0, R_INF=0, R_DEV=0, syn=0,
    ev=[("CAP", "stablelm_github", "we train on 1 trillion (1T) tokens for 4 epochs"),
        ("NONEV", "stablelm_github", "Further inspiration for the token count is taken from \"Go smol or go home\"")],
    note="Four epochs of a 1T corpus. The token count follows de Vries (2023); no inference or device reason is stated.")
U["B:stabilityai/stablelm-base-alpha-7b-v2"] = dict(kind="singleton", own_cap=0, R_CO=0, R_INF=0, R_DEV=0, syn=0,
                                                    ev=[("NONEV", "stablelm_github", "scheduling 1 trillion tokens at context length 2048 followed by 100 billion tokens at 4096.")],
                                                    note="")
U["B:TinyLlama/TinyLlama_v1.1"] = dict(
    kind="singleton", own_cap=1, own_rule="C2 (Amendment 2: frame budget 2T)", R_CO=0, R_INF=1, R_DEV=1, syn=0,
    ev=[("CAP", "tinyllama_readme", "| Combined Dataset Size | Around 950B tokens |"),
        ("CAP", "tinyllama_readme", "| Total Tokens During Training | 3 trillion (slightly more than 3 epochs/1430k steps) |"),
        ("DEV", "tinyllama_readme", "This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint."),
        ("INF", "tinyllama_readme", "The fact that TinyLlama is a relatively small model with grouped query attention means it is also fast during inference.")],
    note="The README describes the project's 3T run over a 950B-token dataset; v1.1's 2T budget (frame) is about two "
         "passes over the same data sources, which the saved README does not state for v1.1 itself.")
