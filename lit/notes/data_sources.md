# Data sources for estimating scaling-law production functions

**Strand:** data scouting. **Compiled:** 2026-09-23. **Companion BibTeX:** `lit/bib/data.bib`.

**How things were checked.** Every dataset below was checked this session against its live source: GitHub API listings via authenticated `gh api`, raw-file headers, Hugging Face Hub and `datasets-server` API (info, statistics, first-rows), Epoch AI CSV/zip endpoints, the OpenRouter public API, the Wayback CDX API, and the arXiv and Crossref APIs for bibliographic facts. Row counts, column names and ranges are the values **observed on 2026-09-23**. Small CSVs were streamed into memory to compute diagnostics. No large file was saved to disk. Anything I could not confirm is labeled **UNVERIFIED**.

**Notation.**
- N: parameters. D: training tokens actually processed.
- C: training FLOP, often approximated as 6ND.
- L: loss (nats/token unless stated otherwise).
- M = D/N: tokens per parameter.
- "Off-frontier" means variation in D/N *holding C fixed*, as in an IsoFLOP sweep.
- Mapping to IO: ln(L − E) plays the role of (negative) log output; N and D are the inputs; C is cost.

---

## 0. TL;DR

1. **The best controlled "plant-level" data for an IO-style Chinchilla replication are:**
   - **Farseer** (404 runs, 25 N values × 55 D values, D/N 0.3 to 2,570, tuned hyperparameters).
   - **Epoch's Chinchilla extraction** (245 points, the canonical benchmark).
   - **Gadre et al.** (104 models, 3 corpora, M from 5 to 640, with per-model 95% CIs on the loss).
   - **Step Law** (1,911 dense runs = 17 (N,D) cells × a learning-rate/batch-size grid).
   - **Porian et al.** (tuned vs untuned hyperparameters, several FLOP-accounting conventions).
   - **Muennighoff datablations** (about 230 to 316 final losses with repeated data, recoverable from notebooks and ColPret).
2. **Several datasets show functional dependence *by construction*.** This is a clean empirical illustration of Ackerberg, Caves and Frazer (2015):
   - Every final DataDecide model has D = 100·N.
   - Every Pythia model has D ≈ 300B.
   - In ObsScaling, FLOPs = 6ND exactly for 100% of rows.
   - In Epoch's AI-models database, C lies within ±12% of 6·N·D·epochs for 53% of language models (the "operation counting" method). So log C, log N and log D are exactly collinear in about half the cross-section. That collinearity comes from how the data were constructed, not from optimization.
3. **Cross-lab panels with N, D, lab/family and date:**
   - Sloth `data_v2` (175 models with N and D, 33 families, Open LLM Leaderboard v1/v2 scores).
   - ObsScaling (123 of 148 base models with N and D, 39 families).
   - Epoch all-models (589 language models with N, D and C; 480 of them since 2020; org, country, date, confidence grade, accessibility, hardware, cost).
   - The Epoch Capabilities Index (268 models, 2023-02 to 2026-09; 104 name-match to a compute estimate, 70 to N and D).
4. **Selection is structural.**
   - Epoch "Notable" selects partly on the *outcome* (SOTA, citations, users). That is truncation on Y, which biases slopes.
   - "Frontier" (top-10 by compute) and "Large-scale" (>1e23 FLOP) select on an *input*, which is benign for E[Y|X].
   - Closed labs don't disclose N and D, so missingness is non-random, like Olley–Pakes exit.
5. **Output measurement is heterogeneous** (nats per token depend on the tokenizer; benchmark accuracy is bounded; Elo is relative).
   - Common-output bridges exist: bits-per-byte (bpb) columns in the OLMo ladder CSVs, WikiText-103 loss in the OLMo ladder, DataDecide-ppl and Ho et al., Paloma subsets in Gadre et al., and IRT-type latent indices (Epoch ECI, ObsScaling PCs, Sloth skills). These are the IO analogs of quantity-based output (TFPQ) versus revenue-based output (TFPR).
6. **Seed-replicate data identify the variance of the pure i.i.d. shock ε,** separately from persistent productivity ω and misspecification:
   - PolyPythias (10 seeds × 5 sizes).
   - DataDecide (3 seeds × 350 configurations).
   - OLMo-ladder reruns (5 pairs).
   - Choshen et al. report about 4% ARE as the achievable floor.
7. **A second, independent measurement of C** exists for 126 language models in Epoch: hardware count × hours × peak FLOP/s × MFU. It can be set against 6ND, which gives a repeated-measurement errors-in-variables correction for the input C.
8. **The revealed inference wedge θ = (αAN^−α)/(βBD^−β) is very sensitive to which Chinchilla estimates are used** (see §7). For Llama-3-8B, θ = 2.9 with Hoffmann et al. Approach-3 values and 5.2 with Besiroglu et al. values. The implied anticipated inference demand is 5.8 to 12.7 × D_train.
9. **Demand and price side:**
   - OpenRouter public API: 456 models, 181 with a Hugging Face id, per-provider prices. About 31 monthly Wayback snapshots from 2023-07 to 2026-09 allow a price panel.
   - Hugging Face API: downloads, all-time downloads, and an exact safetensors parameter count.
   - LMArena CC-BY panel: 1.14M rows, 422 models.
   - Epoch "frontier" `API prices`: only 16 non-empty rows.
   - Artificial Analysis needs an API key and has attribution terms.
10. **Licensing:**
    - Clean: Epoch (CC-BY), LMArena (CC-BY-4.0), DataDecide (ODC-BY), Pythia, datablations, OLMo-ladder and ObsScaling (Apache-2.0), Gadre, Porian, ColPret, Sloth, Gemstones and boardlaw (MIT).
    - **No license file:** Step Law, Farseer, and Epoch's `analyzing-chinchilla` repo. Cite them and ask the authors before redistributing derived data.

---

## 1. Ranked recommendations by research task

### (1) Replicate Chinchilla with IO-style estimators (NLS/Huber vs. log-linear OLS, translog, CES, GMM, bootstrap/cluster inference)

| Rank | Dataset | Why |
|---|---|---|
| 1 | **Farseer `1222_full.csv`** | Densest (N,D) design available. 404 runs; N from 99.6M to 6.37B; D from 1B to 512B; C (6ND) from 1.2e18 to 3.5e21; D/N from 0.31 to 2,570. Hyperparameters set by Step Law (so the flexible inputs are "optimized"). One data recipe, so there is no cross-plant TFP. Near-zero corr(log N, log D) = −0.04, and sd(log D/N given log C) = 1.96. This is a near-orthogonal factorial design. |
| 2 | **Epoch Chinchilla extraction** | The object everyone replicates. 245 points (N from 5.7e7 to 1.6e10; C from 1.4e18 to 1.3e22; L from 2.08 to 5.01). Besiroglu et al. estimates: E = 1.8172 (0.03), A = 482.01 (124.58), B = 2085.43 (1293.23), α = 0.3478 (0.02), β = 0.3658 (0.02), with n = 240 after excluding 5 outliers at D/N < 0.4. Hoffmann et al. Approach 3: E = 1.6934, A = 406.4, B = 410.7, α = 0.3392, β = 0.2849. |
| 3 | **Gadre et al. (mlfoundations/scaling)** | 104 models on 3 corpora (C4, RedPajama, RefinedWeb = 3 "plants"). M ∈ {5, 10, 20, 40, 80, 160, 320, 640}. Each model has 8 loss evaluations **with token-level 95% CIs**, so the output's measurement-error variance is known and can be used for WLS/GLS or a misspecification test. They also have 46 downstream evaluations. Their Eq. 4 imposes α = β, i.e., an exactly homothetic CES: L(C,M) = E + (aM^η + bM^−η)C^−η. |
| 4 | **Step Law dense** | 1,911 runs: 5 N values (215M to 1.07B) × 15 D values (4B to 100B), giving 17 (N,D) cells. Inside each cell there is a grid of 26 learning rates × 13 batch sizes. The minimum over hyperparameters is the *frontier*; the rest measures *inefficiency*. That supports stochastic-frontier estimation or a "flexible input" first-order condition, as in Gandhi–Navarro–Rivers. |
| 5 | **Porian et al.** | Kaplan-vs-Chinchilla "natural experiment" on input measurement and flexible-input optimization. The exponent a in N* ∝ C^a changes as follows (95% CIs): 0.864 (0.82–0.90) to 0.699 when last-layer FLOPs are counted; to 0.603 with scaled warmup; to 0.518 (0.49–0.54) with tuned optimizer on OWT2; 0.497 (0.49–0.50) on RefinedWeb. Models span 16 architectures from 55M to 901M. The file stores several parameter- and FLOP-count conventions side by side (`params_active`, `params_no_embed`, `flops_per_token_att`, `flops_per_token_cc`, and others). |
| 6 | **Datablations (Muennighoff)** | Adds the *input-quality/depreciation* dimension (repeated epochs). The fitted law has α = β = 0.3527, R_D* = 15.39, R_N* = 5.31. Up to 4 epochs of repetition is about as good as unique data. |

**IO estimators to try:** log-linear OLS with E profiled out; the Huber–log-sum-exp M-estimator of Hoffmann and Besiroglu (the "reference" estimator); translog in (log N, log D) with fixed effects; CES by NLS (Kmenta approximation as a check); GMM that uses IsoFLOP design as instruments; cluster bootstrap by compute budget or by corpus.

### (2) Demonstrate the functional-dependence (ACF) problem
1. **Chinchilla or Farseer "on-frontier" subsample.**
   - Keep only the per-C argmin model (IsoFLOP minima) and try to estimate (α, β) separately. N*(C) and D*(C) are deterministic, so only a combination is identified. This is the scaling-law version of ACF's collinearity result for labor.
   - Then add the off-frontier IsoFLOP points: identification returns.
   - Report the condition number and the ratio of bootstrap SEs between the two.
2. **DataDecide** (all final checkpoints satisfy D = 100N, so log D = log N + log 100). Perfect collinearity: α and β are not separately identified from final models. Intermediate checkpoints break collinearity, but the LR-schedule state changes along the run, so D is confounded with the optimizer state. That makes it a "bad instrument" example.
3. **Pythia** (D ≈ 299.9B for all 8 sizes; 154 checkpoints). β is unidentified from final models. The same checkpoint caveat applies (cosine schedule to 10%).
4. **Construction-induced dependence in cross-model data.**
   - ObsScaling: `FLOPs (1E21)` = 6ND for 100% of rows.
   - Epoch: 53% of language models have C = 6·N·D·epochs within ±12%. Its `Training compute estimation method` field is "Operation counting" for 325 of 589.
   - Any regression that uses C alongside N and D in these rows is collinear by construction.
5. **Step Law MoE subset.** D/N ≤ 9.3, and sd(log D/N given log C) is about 0.002, so there is no off-frontier data variation. It varies only sparsity (N_active/N_total from 0.087 to 0.576).

### (3) Cross-lab TFP dispersion and simultaneity (Marschak–Andrews)
1. **Sloth `data/data_v2.csv`.**
   - 188 rows, 175 with N and D, 33 families, date, instruct flag, and Open LLM Leaderboard v1+v2 scores plus BBH sub-tasks.
   - Sloth's own model is θ_ik(s,t) = α_ik + β_kᵀ(log s, log t, log s·log t): a **translog in (N, D) with family fixed effects**. The authors themselves describe α_ik as a family "efficiency" term, i.e., Hicks-neutral TFP.
2. **ObsScaling `eval_results/base_llm_benchmark_eval.csv`.**
   - 148 base models (123 with N and D), 39 families.
   - Model: S_m = θ_f log C_m + ν_f with family-specific (θ_f, ν_f). That is heterogeneous *slopes* too: non-neutral technology across firms.
   - "Phi is a clear outlier in compute efficiency" because synthetic-data (teacher) FLOPs are not counted. This is exactly the value-added vs gross-output problem of Gandhi, Navarro and Rivers (2020).
3. **Epoch all-models.**
   - 589 language models with N, D and C (480 since 2020), org, country, date, and a Confidence grade (Confident 420 / Likely 128 / Speculative 41). Confidence can serve as a measurement-error weight or class.
   - Also available: hardware, hours, MFU (43), cost (96).
   - Top orgs since 2020 by count: Meta AI 16, Alibaba 15, Google 14, Nvidia 7, OpenAI 7, DeepMind 7, DeepSeek 6.
4. **Epoch Capabilities Index (ECI)** plus AI-models compute. ECI has 268 models (2023-02-24 to 2026-09-09) with CIs; 219 match by exact name to AI Models; 104 have compute; 70 have N and D.
5. **Within-lab controlled "multi-plant" data**, useful for isolating data-recipe TFP from scale:
   - DataDecide: 25 corpora × 14 sizes × 3 seeds = 1,050 runs.
   - open-sci-ref: 8 corpora × 4 sizes × 3 token budgets.
   - Gadre: 3 corpora.
6. **Simultaneity design.** Lab-level TFP (data quality, architecture) raises the optimal C, so OLS of log L on log C is biased toward steeper returns. Candidate instruments and proxies:
   - Hardware prices and availability from Epoch `ml_hardware.csv` (176 accelerators with release price and FLOP/s), i.e., cost shifters.
   - Lagged own-family scale.
   - Reported hyperparameters as a proxy variable (idea in §8).

### (4) Algorithmic progress as TFP growth; factor-augmenting change
1. **Ho et al. (2024) dataset** (Google Sheet, CSV export works; `gid=2087221150`).
   - 408 rows × 36 columns: System, date, N, D, epochs, "Inferred_compute (6ND)", perplexity on WT103/WT2/PTB, zero-shot flag, Include?, Outlier?, Architecture, Base model, Organizations, **Tokenizer, Vocabulary**. 231 models were used in the paper.
   - Paper (Table 2, Eq. 8): α_param = 0.068 [0.045, 0.127], β_data = 0.040 [0.023, 0.062], α_year = 0.004 [−0.058, 0.032], β_year = 0.036 [−0.002, 0.080]. Effective-compute doubling time 8.4 months [4.5, 14.3].
   - The authors state that **α_year and β_year are negatively correlated; only their combination (effective compute) is identified.** This is the scaling-law version of the Diamond–McFadden–Rodriguez non-identification of the elasticity of substitution and the bias of technical change.
2. **Epoch all-models column `WikiText and Penn Treebank data`** (404 flagged rows) links Ho's rows to the full database.
3. **Modern vintages.** Sloth and ObsScaling (dates, 2023–2024), ECI (2023–2026, IRT-like latent capability), and Densing Law (capacity density doubling in roughly 3 months, no data link).
4. **Controlled sweeps of different vintages:** Chinchilla (2022), Gadre/Porian (2024), OLMo ladder (2024; includes OLMo-2 7B/13B targets), Farseer (2025). Factor bias can be identified only if a **common output metric** is re-measured. OLMo-ladder CSVs and DataDecide-ppl include WikiText-103 and Paloma subsets, and the OLMo ladder has bpb columns. See §8 idea 3.

### (5) Revealed inference-cost wedge from over-training
- **Inputs:** Epoch all-models (N, D, epochs, org, date, `Model accessibility`, `Open model weights?`). Combine with a controlled-sweep estimate of (A, B, α, β), or better a lab- or family-specific one, to compute θ_i (§7).
- **Demand and price side:**
  - OpenRouter `/api/v1/models` (456 models; `pricing.prompt/completion` per token; `hugging_face_id` for 181; `created` date).
  - `/api/v1/models/{id}/endpoints` gives per-provider prices, e.g., Llama-3.1-70B: DeepInfra $0.40/M tokens (fp8) vs Bedrock $0.72/M.
  - Wayback CDX shows 31 monthly snapshots of the OpenRouter API (2023-07-26 to 2026-09-02).
  - Hugging Face model API (`downloads` for the last 30 days, `downloadsAllTime`, `safetensors.total` = exact N).
  - LMArena `text` panel (vote counts as a usage proxy).
  - Epoch frontier `API prices` (16 rows).
  - Artificial Analysis (key required).
- **Descriptive preview** (Epoch, language models with C > 1e21): median D/N by year was 3.7 (2021, n=33), 14.9 (2022, 46), 44.8 (2023, 109), 200 (2024, 101), 87.7 (2025, 77). For 2023+, open-weights median D/N is 109 (n=246) vs API-only 30 (n=21).
- **Caveat on that preview:** it is confounded by disclosure selection (API labs rarely disclose N and D), by MoE total-parameter counting, and by dataset-size units.

---

## 2. Machine-readable table

Columns:
- `off_frontier`: whether there is variation in D/N at fixed C.
- `common_output`: whether all units share one output metric.
- `tasks`: which of tasks (1)–(5) the dataset supports.

```csv
id,name,kind,primary_url,raw_or_api_url,format,n_units,key_vars,license,off_frontier,common_output,seeds,tasks,bibkey,quirks
A01,Epoch Chinchilla extraction,A,https://github.com/epoch-research/analyzing-chinchilla,https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv,CSV 23KB,245 runs,"Model Size(N), Training FLOP(C), loss; D=C/(6N) derived",none (no LICENSE file),yes (IsoFLOP),yes (MassiveText loss),no,1;2,besiroglu2024chinchilla,"digitized from Hoffmann Fig.4; loss via 256-color map (~0.01 quantization); 5 outliers D/N<0.4; D is constructed"
A02,Porian et al. runs,A,https://github.com/formll/resolving-scaling-law-discrepancies,https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz,pandas pickle (xz) 9.0MB + summary pickles,"runs across 16 archs 55M-901M (row count to compute on load)","width, depth, lr, bs, beta2, warmup, decay, dataset(RW/OWT2), seed, train/val loss curves, FLOP conventions",MIT,yes (IsoFLOP + checkpoints at preset FLOPs),within dataset,partial (seed col),1;2,porian2024resolving,"pickle requires pandas; multiple param/FLOP conventions (params_active, params_no_embed, flops_per_token_att/cc); tuned vs untuned hparams"
A03,Gadre et al. over-training testbed,A,https://github.com/mlfoundations/scaling,https://api.github.com/repos/mlfoundations/scaling/contents/exp_data/models,104 JSON (~9KB each) + exp_data/evals,104 models,"N (params, params_no_embed), tokens, M multiplier, lr, wd, bs, 8 loss evals w/ 95% CI, 46 downstream evals",MIT,yes (M=5..640),yes within val set,no,1;2;3,gadre2024language,"3 corpora (c4 34, rpj 35, rw 35); multiplier in JSON is x20 tokens/param; loss CIs allow known-variance weighting"
A04,Muennighoff datablations,A,https://github.com/huggingface/datablations,notebooks plotstables/return_alloc.ipynb & contours.ipynb (hard-coded dicts); ColPret aggregated_eval/datablations_*.csv,ipynb dicts / CSV,"~316-353 named final losses; ColPret contour 229 rows; curves 9,260 rows","N, total tokens D, unique tokens U (epochs), val loss (C4/OSCAR)",Apache-2.0,yes (IsoFLOP + IsoLoss),yes per corpus,some (seeds repos),1;4,muennighoff2023scaling,"TensorBoard.dev links dead; parse names like 2b84b4b; ColPret columns tokens_per_epoch and flops are internally inconsistent -> rebuild from names"
A05,Pythia suite evals,A,https://github.com/EleutherAI/pythia,https://api.github.com/repos/EleutherAI/pythia/contents/evals/pythia-v1,JSON per checkpoint,"8 sizes x {std,dedup} x 154 ckpts; seeds 1-9 for std","N, step (tokens = step x 2,097,152), benchmark evals",Apache-2.0,no at final (D fixed 300B); checkpoints only,yes (Pile),yes (seeds 1-9),2;3,biderman2023pythia,"all final models D~=299.9B; checkpoints confounded by cosine LR schedule; dedup = 1.5 epochs of 207B"
A06,PolyPythias evals,A,https://huggingface.co/datasets/EleutherAI/polypythias-evals,https://huggingface.co/api/datasets/EleutherAI/polypythias-evals/tree/main,per-model folders,50 runs (5 sizes x 10 seeds),"size, seed, checkpoint evals",no license field on card,no,yes,yes (10),2 (noise variance),vanderwal2025polypythias,"use to estimate Var(eps) of pure seed shock"
A07,OLMo ladder runs,A,https://github.com/allenai/OLMo-ladder,https://raw.githubusercontent.com/allenai/OLMo-ladder/main/src/scripts/paper/data/ladder-runs/760M-2xC.csv,CSV per run (~100-375KB),"30 ladder runs (190M,370M,760M,1B,3B x 0.5,1,2,5,10xC + 1xC reruns) + OLMo-2 7B/13B target evals","_step, total_tokens, total_training_Gflops, lr, bs, CE loss on C4/Dolma/Pile/WikiText-103/Paloma, task bpb, task accuracy",Apache-2.0,yes (0.5-10x Chinchilla),yes (bpb & CE on fixed sets),rerun pairs,1;2;4,bhagia2024establishing,"paper Eq.1-2: L(N,D)=A/N^a+B/D^b+E then sigmoid to accuracy; N reported non-embedding in paper"
A08,DataDecide,A,https://huggingface.co/datasets/allenai/DataDecide-eval-results,https://huggingface.co/datasets/allenai/DataDecide-ppl-results,"Parquet (1.41M rows eval; 22,709 rows ppl)",1050 runs = 25 recipes x 14 sizes (4M-1B) x 3 seeds,"params, data recipe, seed, step, tokens, compute, chinchilla, OLMES task metrics, perplexities (C4, Dolma subsets, Pile, WikiText-103, Paloma)",ODC-BY,no at final (D=100N); checkpoints only,yes,yes (3),2;3;4,magnusson2025datadecide,"perfect log N/log D collinearity at final ckpts; recipes = 25 'plants' with same architecture"
A09,IBM ColPret (Hitchhiker),A+B,https://github.com/IBM/ColPret,https://api.github.com/repos/IBM/ColPret/contents/aggregated_eval,CSV/npy (pythia.csv 25.7MB; Amber.csv 27.6MB; olmo.csv; t5_pile; opt; K2; datablations; overtrain),485 models / 40+ families / ~1.9M steps,"model_name, family, tokens_seen, flops, num_params, data, checkpoint, loss cols, seed, epochs, arch",MIT,mixed,no (family-specific losses),some,1;2;3,choshen2024hitchhikers,"harmonized many sources but columns sometimes inconsistent (see datablations); ARE floor ~4% from seed noise; drop first ~10B tokens"
A10,Step Law dense+MoE,A,https://github.com/step-law/steplaw,https://raw.githubusercontent.com/step-law/steplaw/main/data/dense_lr_bs_loss.csv,CSV 344KB + 233KB,"1,911 dense runs; 708 MoE runs","dense: h, ffnh, numh, numl, lr, bs, iters, loss, smooth loss, N, D; MoE adds topk, nume, N(total), Na(active), Na/N",NONE (no license file),dense yes (D/N 19-466); MoE no,yes (own val set),no,1;2;5(MoE),li2025predictablea,"17 (N,D) cells with LR x BS grid -> frontier vs inefficiency; training logs on W&B"
A11,Farseer grid,A,https://github.com/Farseer-Scaling-Law/Farseer,https://raw.githubusercontent.com/Farseer-Scaling-Law/Farseer/main/ipynb/data/1222_full.csv,CSV 143KB (+~50 auxiliary CSVs),404 runs,"N, D, h, ffnh, numh, numl, lr, bs, loss, smooth loss, BPC on IntelliValSet en/zh web/paper/book",NONE (no license file),"yes (D/N 0.31-2,570)",yes (BPC on fixed val set),no,1;2,li2025predictableb,"column 'D/N' is NOT D/N (it is ~M/N); recompute from D and N; single recipe 2049_sc; bilingual ablation files"
A12,Gemstones,A,https://github.com/mcleish7/gemstone-scaling-laws,https://huggingface.co/Gemstone-Models,JSON/JSONL/parquet + 22 HF model repos,22 width x depth shapes; >4000 ckpts (paper),"width, depth, checkpoint, val loss (FineWeb-Edu, DCLM), task losses",MIT,partial (checkpoints; lr/cooldown ablations),yes,no,1;2,mcleish2025gemstones,"shape heterogeneity at fixed N -> N is not a sufficient statistic for 'capital'; W&B private, processed frames provided"
A13,open-sci-ref-0.01,A,https://huggingface.co/datasets/open-sci/open-sci-ref-0.01-logs,https://github.com/LAION-AI/open-sci-ref-0.01,tar.gz 44MB logs,"8 corpora x {0.13,0.4,1.3,1.7}B x {50B,300B,1T}","N, D, corpus, training logs, intermediate ckpts",Apache-2.0,yes (token budgets),yes,partial,3;4,nezhurina2025opensciref,"data-recipe TFP at matched (N,D)"
A14,Jones boardlaw (Hex),A,https://github.com/andyljones/boardlaw,https://f002.backblazeb2.com/file/boardlaw/output/experiments/eval/database.sql,SQLite 69.5MB,"many agents (board sizes 3-9)","train compute, test-time nodes, Elo from trials",MIT,yes (train vs test compute),Elo within board size,no,5 (train-vs-test isoquant),jones2021scaling,"Elo relative; direct train/test compute trade-off data"
A15,Large Language Monkeys samples,A,https://huggingface.co/datasets/ScalingIntelligence/monkey_business,datasets-server info,Parquet,"configs by task x model (e.g., MATH_Gemma-2B 128 problems)","per-problem samples + is_corrects (coverage vs k)",MIT,n/a,yes per task,n/a,5,brown2024large,"test-time compute input = #samples; model size = train-side input"
B01,Epoch AI models (all/notable/large-scale/frontier),B,https://epoch.ai/data/ai-models,https://epoch.ai/data/all_ai_models.csv,CSV (6.8MB all; zip 3.3MB),"3,620 all; 1,072 notable; 532 large-scale; 137 frontier","Model, Organization, date, Domain, Parameters, Training compute (FLOP), Training dataset size (total), Epochs, Confidence, compute estimation method, hardware, quantity, hours, MFU, cost (2023 USD), accessibility, open weights, country, base model, post-training compute, HF developer id",CC-BY 4.0,n/a (cross-section),no,no,3;4;5,epoch2026aimodels,"C often = 6ND by construction (53%); dataset size != tokens seen (use Epochs); MoE params total vs active mixed; notable = selection on outcome; living data -> pin snapshot"
B02,Epoch Benchmarking Hub + ECI,B,https://epoch.ai/benchmarks,https://epoch.ai/data/benchmark_data.zip,zip 2.3MB (~70 CSVs),"ECI 268 models; per-benchmark 5-313 rows; model_metadata 1,089 versions (348 with compute)","Model version, score, stderr, release date, org, country, training compute; ECI (eci, CI), EDI (benchmark difficulty, slope)",CC-BY 4.0 (external data keep own licenses),n/a,latent index (ECI),no,3;4,epoch2026benchmarking,"ECI name-join to AI Models 219/268; compute for 104; IRT-style specifics UNVERIFIED"
B03,Ho et al. algorithmic-progress sheet,B,https://github.com/epoch-research/lm-algorithmic-progress,https://docs.google.com/spreadsheets/d/11m8O_mU0cUkOB_5wluPne4PNsuvsKNbbVAzbYNy-NXY/export?format=csv&gid=2087221150,Google Sheet CSV export,408 rows (231 used in paper),"System, date, N, D, epochs, 6ND, ppl WT103/WT2/PTB, zero-shot, include, outlier, architecture, org, tokenizer, vocabulary",code MIT; sheet no explicit license (paper CC BY 4.0),n/a,3 benchmarks (dataset FE),no,4,ho2024algorithmic,"perplexity tokenizer-dependent; <=3 models per paper filter; alpha_year/beta_year not separately identified"
B04,ObsScaling eval results,B,https://github.com/ryoungj/ObsScaling,https://raw.githubusercontent.com/ryoungj/ObsScaling/main/eval_results/base_llm_benchmark_eval.csv,CSV,"148 base (123 with N,D); 92 sub-10B; 72 LB-v2; 27 instruct","Model, family, N (B), D (T), FLOPs(1e21)=6ND, MMLU, ARC-C, HellaSwag, Winogrande, TruthfulQA, GSM8K, XWinograd, HumanEval",Apache-2.0,n/a,yes (benchmarks),no,3;4,ruan2024observational,"FLOPs = 6ND exactly; MoE size inconsistent (Mixtral-8x7B 45 total vs 8x22B 39 active)"
B05,Sloth data,B,https://github.com/felipemaiapolo/sloth,https://raw.githubusercontent.com/felipemaiapolo/sloth/main/data/data_v2.csv,CSV (+ LB v1/v2 raw 1.75MB/0.2MB),"188 rows (175 with N,D), 33 families","Model, Family, Instruct, date, #Params, D (T), FLOPs, IFEval, BBH(+subtasks), MATH L5, GPQA, MUSR, MMLU-PRO",MIT,n/a,yes (benchmarks),no,3;4,maiapolo2024sloth,"mix of base & instruct; D partly from HF/leaderboard metadata"
B06,Open LLM Leaderboard v2/v1 contents,B,https://huggingface.co/datasets/open-llm-leaderboard/contents,https://huggingface.co/api/datasets/open-llm-leaderboard/contents/parquet/default/train/0.parquet,Parquet (4.1MB v2; 5.5MB v1),"v2 4,576 rows (275 pretrained, 58 cont.-pretrained); v1 7,260","#Params (B), Architecture, Type, Base Model, MoE, Merged, precision, IFEval, BBH, MATH L5, GPQA, MUSR, MMLU-PRO (+raw), CO2 cost, dates",no license field,n/a,yes,no,3,openllmleaderboard2025contents,"no D; #Params=-1 means missing; mostly fine-tunes/merges (intermediate-input products); archived (no longer updated)"
B07,LMArena leaderboard dataset,B,https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset,datasets-server config=text split=full,Parquet,"text/full 1,135,923 rows; 422 models; 62 orgs; 2023-05-08..2026-09-13","model_name, organization, license, rating, CI, variance, vote_count, rank, category, publish date",CC-BY-4.0,n/a,relative (Bradley-Terry/Elo),n/a,3;5,lmarena2026leaderboard,"Elo is relative (TFPR-like): anchor drift; style-control variant; many categories"
B08,OpenRouter models API,B,https://openrouter.ai/api/v1/models,https://openrouter.ai/api/v1/models/{id}/endpoints ; https://web.archive.org/cdx/search/cdx?url=openrouter.ai/api/v1/models,JSON,"456 models (181 with HF id); 31 monthly Wayback snapshots","prompt/completion price per token, context, created, hugging_face_id, per-provider prices, quantization",ToS UNVERIFIED,n/a,n/a,n/a,5,openrouter2026models,"current snapshot only; history via Wayback; provider competition for open models ~ marginal-cost pricing"
B09,Hugging Face model API,B,https://huggingface.co/api/models/{id},?expand[]=downloads&expand[]=downloadsAllTime&expand[]=safetensors,JSON,any public model,"downloads (30d), downloadsAllTime, likes, safetensors.total (exact N)",HF ToS,n/a,n/a,n/a,5 (usage), , "exact N vs reported N (measurement error); downloads = demand proxy for open weights"
B10,Epoch ML hardware,B,https://epoch.ai/data/machine-learning-hardware,https://epoch.ai/data/ml_hardware.csv,CSV,176 accelerators,"release date, release price, FP16/BF16/FP8/FP4 FLOP/s, memory, bandwidth, TDP, price-performance",CC-BY (page-level UNVERIFIED),n/a,n/a,n/a,3;5 (cost shifters/IV),epoch2026mlhardware,"build hardware-based C = qty x hours x peak x MFU (126 LMs matchable)"
B11,Artificial Analysis API,B,https://artificialanalysis.ai/documentation,API (x-api-key),JSON,UNVERIFIED count,"intelligence index, benchmark scores, price per 1M in/out/blended, output speed, TTFT",attribution required; ToS UNVERIFIED,n/a,index,n/a,5,artificialanalysis2026api,"needs account+key (user must create); 1,000 req/day"
B12,HELM raw results,B,https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/,"gs://crfm-helm-public/{classic,lite,capabilities,mmlu}/benchmark_output",JSON (huge: hundreds GB; Classic >5TB),many models x scenarios,"per-instance predictions, metrics",UNVERIFIED,n/a,yes,no,3,liang2023holistic,"do not bulk download; use summary JSON only"
```

---

## 3. Dataset cards: (A) controlled sweeps

### A01. Epoch's Chinchilla extraction (Besiroglu, Erdil, Barnett, You 2024)
- **Where:** `https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv` (23 KB).
  - Columns: `x, y, color, Model Size, Training FLOP, hex_color, loss`.
  - The repo also holds `data_extraction.ipynb` and `data_analysis.ipynb`. It has no LICENSE file; last push 2024-05-14.
- **Content:** 245 points digitized from Hoffmann et al. Fig. 4 (Approach 3 scatter). N from 5.73e7 to 1.62e10; C from 1.4e18 to 1.3e22; loss from 2.08 to 5.01 (MassiveText validation loss).
  - D is **not observed**; it is imputed as C/(6N). Any "D" regressor is a deterministic function of (C, N).
  - Diagnostics: corr(log N, log D) = +0.24; sd(log D/N given log C) = 1.50, i.e., rich off-frontier variation.
- **Measurement error in output:** loss was recovered from a 256-level color map, so quantization error is at most about ±0.01 (known, roughly uniform). There is further y-coordinate error from missing ticks.
- **Estimation facts (Besiroglu et al.):**
  - Huber loss on log L with δ = 1e−3, LSE parameterization, BFGS, 4,000 bootstrap replications.
  - They find Hoffmann's reported CIs implausibly narrow. For a = β/(α+β), Hoffmann reports 0.454–0.455; the replication SE is 0.018, so the 80% CI is about 50× wider. Reaching Hoffmann's precision would need about 600,000 runs.
  - Hoffmann's Approach 3 implies about 70 tokens per parameter, inconsistent with the paper's own 20. The replication gives about 20 (19–25.6 depending on outliers) and is consistent with Approaches 1–2.
- **IO use:** the benchmark replication. Compare Huber-LSE NLS, log-linear OLS with profiled E, translog, CES-NLS and GMM. Show that the SEs on α and β are driven by the IsoFLOP (off-frontier) spread.

### A02. Porian et al. (2024), formll
- **Where:** `data/experiment_results.pickle.xz` (8.98 MB). Summary pickles: `summary_df*.pickle.xz`. Loader: `data.py`. MIT license. Checkpoints are on HF at `formll/resolving-scaling-law-discrepancies`.
- **Each row is a training run.** Columns:
  - `hparams` (tuned/not), `warmup`, `decay` (constant / Chinchilla / cosine), `dataset` (RefinedWeb/OWT2).
  - Loss curves: `train/loss`, `val/loss` (pandas Series).
  - `width`, `depth`, `lr`, `bs`, `beta2`, `seed`, `seq_len`, `vocab_size`, `warmup_tokens`, `optimizer`, `precision`, `qk_norm`, `z_loss_coefficient`, `independent_wd`, `max_step`.
- **Derived in `data.py`:** `params_active` = 12w²d + Vw; `params_no_embed`; `params_all`; FLOPs per token with and without attention and with and without embeddings (`flops_per_token_att`, `flops_per_token_cc`, `flops_per_token_no_att`).
  - This is ideal for **input-measurement sensitivity**: the same run can be re-expressed under different "capital" definitions.
- **Results** (compute-optimal exponent a, N* ∝ C^a, 95% CIs): Kaplan reproduction 0.864 (0.82, 0.90) → count last layer 0.699 (0.66, 0.72) → correct warmup 0.603 (0.57, 0.63) → tuned optimizer 0.518 (0.49, 0.54) on OWT2; 0.497 (0.49, 0.50) on RefinedWeb.
  - Careful LR decay is *not* necessary for Chinchilla scaling.
  - Estimation is IsoFLOP with "noise-and-interpolate" bootstrap (Akima interpolation) and weighted log-space regression.
- **IO reading:** the Kaplan–Chinchilla discrepancy is explained by (i) input mismeasurement (omitted last-layer FLOPs, i.e., mismeasured capital services), (ii) an adjustment-cost-like fixed warmup that penalizes small plants, and (iii) *non-optimized flexible inputs* (learning rate, batch size, β2) whose inefficiency correlates with scale. Point (iii) is transmission-type bias through an omitted input.

### A03. Gadre et al. (2024), mlfoundations/scaling
- **Where:** `exp_data/models/*.json` (104 files, about 9 KB each). Also `exp_data/evals/*` (104), `exp_data/eval_metadata.csv`, `grid.json`, and `scaling/laws.py`. MIT license; weights at `huggingface.co/mlfoundations/scaling`.
- **Per model:** dataset (c4_original 34, rpj 35, rw_original 35); `hyperparameters.params`, `params_no_embed`, `tokens`, `chinchilla_multiplier` ∈ {0.25, 0.5, 1, 2, 4, 8, 16, 32}. The multiplier scales 20 tokens/param, so M runs from 5 to 640. Also lr, wd, global batch, warmup.
  - `results[]`: 8 validation losses (in-distribution val, C4, Paloma subsets, etc.), each with `loss_tokens_lower_95/upper_95` and `loss_sequences_lower_95/upper_95`.
- **Paper Eq. 4:** L(C,M) = E + (aM^η + bM^−η)C^−η, obtained by imposing α = β.
  - Fit grid: N ∈ {0.011, 0.079, 0.154, 0.411}B at M = 20, plus 0.011B at M = 320.
  - Held out: 1.4B at M up to 640 and 6.9B at M = 20; about 0.7% relative error.
  - Downstream (Eq. 5): Err(L) = ε − k·exp(−γL).
- **IO use:**
  - Three corpora serve as three "plants" with a common design, so one can test common elasticities vs plant-specific TFP.
  - Known output sampling error allows GLS, and a test of whether residual variance exceeds measurement noise (misspecification or unobserved heterogeneity).
  - The α = β restriction is testable.

### A04. Muennighoff et al. (2023), datablations
- **Where:** repo `huggingface/datablations` (Apache-2.0); models at `huggingface.co/datablations`. There is **no tabular results file** in the repo.
  - Final validation losses are hard-coded as `NAMES_TO_VAL_LOSSES` dictionaries in `plotstables/return_alloc.ipynb` (316 entries) and `plotstables/contours.ipynb` (353 entries). Names such as `2b84b4b` encode (N=2.8B, D=4B, U=4B).
  - The TensorBoard.dev links are dead (service shut down).
  - IBM ColPret re-tabulated them: `aggregated_eval/datablations_contour_losses.csv` (229 rows; N 7.1e6–8.67e9; D 1e8–9e11; epochs 1–9,000), `datablations_losses.csv` (9,260 curve points), `datablations_code_losses.csv` (295 rows).
  - **Quirk:** in ColPret's datablations tables, `tokens_per_epoch` is constant at 1.5e9 even where names imply U = 4B, and the `flops` column does not equal 6ND. Rebuild N, D and U from the model names.
- **Fitted law** (README): L = E + A/(U_N + U_N R_N*(1−e^(−R_N/R_N*)))^α + B/(U_D + U_D R_D*(1−e^(−R_D/R_D*)))^β, with α = β = 0.3527, R_D* = 15.39, R_N* = 5.31, and (log A, log B, log E) = (6.255, 7.305, 0.625).
  - Up to about 4 epochs, repeated data ≈ unique data. The value of further repetition decays to zero.
  - Scope: 400 runs, up to 9B parameters and 900B tokens, on C4 and OSCAR.
- **IO reading:** D_eff = U(1 + R*(1 − e^(−R/R*))) is an *effective capital stock with decaying marginal services*, a depreciation or vintage structure. Effective data is capped at about (1 + R_D*)·U ≈ 16.4·U. Excess parameters decay in the same way (R_N*), like idle capacity.

### A05/A06. Pythia and PolyPythias
- **Pythia:** 8 sizes (70M–12B), each in standard and deduplicated versions. Each model saw 299,892,736,000 tokens (about 1 epoch of the Pile; about 1.5 epochs of the 207B-token deduped Pile), with 154 checkpoints (steps 0, 1, 2, 4, …, 512, then every 1,000; batch 2,097,152 tokens).
  - Evals are in `evals/pythia-v1/<model>/` (1,832 JSON files in `evals/`); there are also OPT and BLOOM baselines.
  - Seeds 1–9 exist for the standard models (`EleutherAI/pythia-[size]-seed[num]`).
- **PolyPythias** (ICLR 2025): `huggingface.co/datasets/EleutherAI/polypythias-evals`, folders `pythia-{14m,…}-seed{0..9}`, 50 runs in total.
- **IO use:** (i) at final checkpoints D is constant, so the D-elasticity is not identified; (ii) the seed dispersion at fixed (N, D) estimates Var(ε), the non-persistent shock, which bounds achievable fit. Choshen et al. put the best achievable ARE at about 4%.

### A07. OLMo ladder (Bhagia et al. 2024; COLM 2025)
- **Where:** `github.com/allenai/OLMo-ladder` (Apache-2.0), `src/scripts/paper/data/ladder-runs/*.csv`, 41 files:
  - 30 ladder runs: {190M, 370M, 760M, 1B, 3B} × {0.5, 1, 2, 5, 10}×C, plus a 1×C rerun for each size.
  - OLMo-2 7B (`peteish7*`) and 13B (`peteish13*`) target evaluations.
- **Columns:** `_step`, `throughput/total_tokens`, `throughput/total_training_Gflops`, `optim/learning_rate_group0`, `learning_rate_peak`, `batch_size_in_tokens`; CE loss on C4, Dolma subsets, ICE, M2D2-S2ORC, Pile and WikiText-103 validation; task bpb (MMLU, HellaSwag, ARC, BoolQ, CSQA, OBQA, PIQA, SIQA, Winogrande); task accuracy.
- **Paper:** 4×4 = 16 ladder models in the main text; Eq. 1 L(N,D) = A/N^α + B/D^β + E on task loss; Eq. 2 Acc(L) = a/(1+e^(−k(L−L0))) + b.
  - Example (MMLU): L = 38.07/N^0.23 + 100.09/D^0.24 + 0.45.
- **IO use:** a clean 5×5 factorial off the frontier with a *measured* FLOP column; rerun pairs give ε-variance; bpb gives a tokenizer-free output measure; the two-step form is exactly "latent output → bounded observed output".

### A08. DataDecide (Magnusson et al., ICML 2025)
- **Where:** `allenai/DataDecide-eval-results` (1,410,750 rows; ODC-BY; features: chinchilla, compute, data, metrics, params, seed, step, task, tokens). `allenai/DataDecide-ppl-results` (22,709 rows; perplexity on C4, Dolma subsets, ICE, M2D2, Pile, WikiText-103).
- **Design:** 25 corpora × 14 sizes (4M, 6M, 8M, 10M, 14M, 16M, 20M, 60M, 90M, 150M, 300M, 530M, 750M, 1B) × 3 seeds ("default", "small/large aux 2/3"). Final models are trained to 100 tokens per parameter.
- **IO use:**
  - A *multi-plant* experiment: same technology, different material quality. It estimates Hicks-neutral vs factor-biased effects of data quality, i.e., whether recipes shift A or B, α or β.
  - It also demonstrates functional dependence (D = 100N at the final checkpoint).

### A09. IBM ColPret (Choshen, Zhang, Andreas 2024)
- **Where:** `github.com/IBM/ColPret` (MIT). Loader: `util/read_data.py`, `get_data()` with columns `model_name, model_type, scaled_set, tokens_seen, flops, num_params, data, checkpoint, loss_cols, original_paper, seed` (+ `epochs`, `arch` ∈ {dec, enc, enc-dec, moe, ssm}).
  - `aggregated_eval/` holds Pythia (25.7 MB), Amber (27.6 MB), K2, OLMo, OPT, T5-Pile, LLM360 Mamba, Revisiting (lang/vision), datablations, and an 'LLMs 0-shot' table.
  - `raw_data/` holds overtrain (Gadre), `chinchila_extracted.csv` (same as Epoch), `IBM_MoE.csv`, GPT-3 figure extractions, bimix-law, RedPajama, and `loss_size_flops_llama_sh.npy`.
- **Scale:** 485 models, 40+ families, about 1.9M evaluated steps.
- **Recommendations:** use intermediate checkpoints; drop the first ~10B tokens; the ARE floor is about 4% because of seed noise; transfer scaling parameters across families.
- **IO use:** a ready-made multi-family panel; the family is the "firm". Watch for harmonization errors (see A04).

### A10. Step Law (Li et al. 2025, Part I)
- **Where:** `github.com/step-law/steplaw` (no license). Files:
  - `data/dense_lr_bs_loss.csv`: 1,911 rows; columns `h, ffnh, numh, numl, lr, bs, ti, loss, smooth loss, exp_name, D, N, D/N`.
  - `data/moe_lr_bs_loss.csv`: 708 rows; adds `seq_len, topk, nume, moeh, N, Na, M, Na/N, kvcache, total_time`.
  - `data/1004_fitted_lr_bs_scaling_model_parameters.csv`: 1,000 bootstrap fits of lr = e^c·N^a·D^b and bs = e^c·D^d.
  - Training logs on W&B `billzid/predictable-scale`; checkpoints at `huggingface.co/StepLaw`.
- **Dense design:** N ∈ {214.7M, 268.3M, 429.3M, 536.9M, 1.074B}; D ∈ 15 values from 4B to 100B; 17 unique (N,D) cells; LR from 2.4e−4 to 2.2e−2 (26 values); BS from 16 to 2,048 (13 values).
- **MoE design:** total N ≈ 2.15B, active N_a from 0.19B to 1.24B, D ∈ {2, 4, 8, 20}B.
- **Paper:** about 3,700 LLMs, about 100T tokens, about 1M H800 GPU-hours; the hyperparameter landscape is convex with a broad optimum.
- **IO use:**
  - (i) Frontier vs inefficiency: the stochastic-frontier distribution of (L − min over hyperparameters of L) given (N, D).
  - (ii) Optimal LR and BS are *policy functions* of the state (N, D). This is the analog of a flexible-input demand, and the first-order-condition logic of GNR/LP applies.
  - (iii) The MoE file separates "capital stock" (total N) from "capital services" (active N_a).

### A11. Farseer (Li et al. 2025, Part II)
- **Where:** `github.com/Farseer-Scaling-Law/Farseer` (no license). The main file is `ipynb/data/1222_full.csv` (404 rows; README: "Primary dataset (~400+ samples)"), plus bilingual ablations (`bilingual_full.csv`, `bilingual_bpc.csv`) and about 50 auxiliary CSVs (LR/BS grids, MoE sparsity, RoPE, code, math, `chinchilla_svg_extracted_data.csv`).
- **Columns:** `N, D, h, ffnh, numh, numl, lr, bs, ti, loss, smooth loss, seq_len=2048, data_recipe=2049_sc`, and BPC on IntelliValSet (zh/en × web-data/paper/book; raw and processed).
  - **Quirk:** the columns `D/N` (five values: 3.52, 10.05, 28.2, 321.7, 454.9) and `new_D/N` do **not** equal D/N (they look like M/N, FLOPs per token over N). `round(D/N)` is the true D/N. Recompute from `D` and `N`.
- **Design range:** N from 9.96e7 to 6.37e9; D from 1e9 to 5.12e11; C (6ND) from 1.2e18 to 3.5e21; about 400 unique (N,D) cells; D/N from 0.31 to 2,570; corr(log N, log D) = −0.04.
- **IO use:** the best dataset for estimating the *shape* of the production function (CES vs translog vs Farseer's own form) and its returns to scale with minimal collinearity.

### A12–A15. Other controlled resources
- **Gemstones** (NeurIPS 2025): 22 width × depth shapes (e.g., 256×23 to 3072×12) on HF `Gemstone-Models`, over 4,000 checkpoints up to 2B parameters.
  - Repo `mcleish7/gemstone-scaling-laws` (MIT) has `benchmarks/fineweb_edu_losses_combined.jsonl`, `fineweb_dclm_losses_combined.jsonl`, `light_eval_df.parquet`, per-model validation losses, and task losses in Bhagia format. The authors' W&B is private, so they provide processed frames and a cache on HF `smcleish/scaling-laws-cache`.
  - Key claim: scaling-law prescriptions are highly sensitive to design choices and to which checkpoints are used.
  - IO use: *composition* of the capital stock (width vs depth) at fixed N, a test of whether N is a sufficient aggregate.
- **open-sci-ref-0.01:** 8 corpora (C4, Pile, SlimPajama, FineWeb-Edu-1.4T, DCLM-baseline, Nemotron-CC-HQ, HPLT-2.0-en, CommonCorpus) × {0.13, 0.4, 1.3, 1.7}B × {50B, 300B, 1T} tokens. Logs in `open-sci/open-sci-ref-0.01-logs` (44 MB tar.gz, Apache-2.0); 137 model repos under HF `open-sci`.
- **Jones (2021) boardlaw:** SQLite evaluation database (69,455,872 bytes, available as of 2026-09-23) at the backblaze URL in the table. Elo per agent vs train compute and test-time search nodes on Hex (board sizes 3–9) gives a *direct train-compute vs test-compute isoquant*. MIT.
- **Large Language Monkeys:** `ScalingIntelligence/monkey_business` (MIT). Configs by task × model, e.g., CodeContests (140 problems) and MATH (128) for Gemma-2B/7B and Llama-3-8B/70B, each with per-problem samples and `is_corrects`. Coverage as a function of the number of samples k at different model sizes: an inference-compute vs model-size isoquant.
- **No public run-level data located:**
  - Kaplan et al. (2020).
  - Hoffmann et al. (2022) raw runs.
  - Sardana et al. (2024): 47 models from 150M to 6B at 10–10,000 tokens/param.
  - Kumar et al. (precision; 465+ runs).
  - Busbridge et al. (distillation; students 143M–12.6B).
  - Hägele et al. (code only: `epfml/schedules-and-scaling`).
  - MiniCPM, DeepSeek LLM, Llama 3 (IsoFLOP in figures only).
  - Possible route: digitize the IsoFLOP figures, as Epoch did for Chinchilla.

---

## 4. Dataset cards: (B) observational / cross-lab

### B01. Epoch AI "Data on AI Models" (CC-BY 4.0)
- **Files:** `https://epoch.ai/data/{all_ai_models,notable_ai_models,large_scale_ai_models,frontier_ai_models}.csv` and `ai_models.zip`. The old `/data/epochdb/...` paths 301-redirect.

| Subset | Rows | Columns | Inclusion rule |
|---|---|---|---|
| All | 3,620 | 57 | Everything in the database (1950 to 2026-09-21) |
| Notable | 1,072 | 47 | SOTA, ≥1,000 citations, ≥1M monthly users, or historical significance: **selection partly on outcomes** |
| Large-scale | 532 | 34 | >1e23 FLOP |
| Frontier | 137 | 61 | Top-10 by training compute at release |

- **Key columns:** Model; Organization; Organization categorization; Country; Publication date; Domain; Task; Parameters; Training compute (FLOP) (+ notes, lower and upper bound); Training dataset size (total); Epochs; Confidence (Confident within 3×, …, Speculative within 30×); Training compute estimation method; Training hardware; Hardware quantity; Training time (hours); Hardware utilization (MFU, HFU); Training chip-hours; Training compute cost (2023 USD / cloud / upfront); Training power draw; Model accessibility; Open model weights?; Base model; Finetune compute; Post-training compute; Batch size; Numerical format; Frontier model; Hugging Face developer id; WikiText and Penn Treebank data.
- **Coverage of language models:**
  - 2,070 rows in the Language domain; 589 have N, D and C (≥2020: 480); notable subset: 236 (167 since 2020).
  - Cost (2023 USD) for 96 of the 589; MFU for 43.
  - Estimation method: "Operation counting" 325; Hardware 51; mixed 90; Reported 31; Third-party 8.
- **Compute is often constructed from N and D.** With epochs, log10(C / (6·N·D·epochs)) has quantiles p5 = −1.22, p25 = 0.00, p50 = 0.00, p75 = +0.10, p95 = +2.74. 53% are within ±0.05 (±12%) and 65% within ±0.3.
  - Large negative deviations are typically MoE (compute uses active parameters while `Parameters` is the total), e.g., "DeepSeek V4.1 Flash" at 748B total with C = 4.32e24 at 45T tokens.
  - Large positive deviations are typically dataset-size unit issues (words, examples, fine-tune data) or multi-stage training.
- **Tokens per parameter** (C > 1e21, by year): 2021 median 3.7 → 2022 14.9 → 2023 44.8 → 2024 200 → 2025 87.7. 2019–2020 values look contaminated by dataset-size units.
- **Hardware-based second measurement of C:** 126 language models have hardware type (matched to `ml_hardware.csv` peak BF16/FP16 FLOP/s), quantity and hours.
  - Using MFU where reported (43 models) and 0.4 otherwise, log10(C_hw / 6ND) has median +0.21, IQR [0.00, 0.75], p10 −0.58, p90 +2.09.
  - Usable as a noisy repeated measurement (errors-in-variables IV à la two-indicator models). The MFU imputation is a strong assumption.
- **Pitfalls:**
  - Living database: pin a snapshot date (all numbers here are from 2026-09-23).
  - Rows exist for models dated up to 2026-09.
  - Organizations are sometimes comma-joined multi-affiliation strings.

### B02. Epoch Capabilities & Benchmarking (CC-BY 4.0)
- **Zip** `https://epoch.ai/data/benchmark_data.zip` (2.3 MB; "updated Sep 23, 2026") contains about 70 benchmark CSVs.
  - Epoch-run benchmarks: GPQA Diamond 313 rows, MATH L5 108, OTIS Mock AIME 291, FrontierMath variants, SWE-bench Verified 35, SimpleQA Verified 80, chess puzzles 224, and others. Columns: `Model version, mean_score, stderr, Release date, Organization, Country, Training compute (FLOP), …`.
  - External benchmarks: MMLU 249, GSM8K 235, BoolQ 206, ARC-AGI 247, ARC-AGI-2 227, WeirdML 173, SciCode 174, METR time horizons 50, WebDev Arena 131, Terminal-Bench 204, and others.
  - Epoch Capabilities Index: `eci_scores.csv` (268 models; eci with CI), `edi_scores.csv` (58 benchmarks; difficulty `edi`, `estimated_slope_scaled`, anchor flag), and `processed_data_for_eci.csv` (2,780 model × benchmark scores).
  - `benchmark_metadata.csv` (86; random baseline, ceiling, scale) and `model_metadata.csv` (1,089 versions; 348 with compute).
- **Access** also via the `epochai` Python client (Airtable).
- **ECI join:** matched by exact name to AI Models for 219 of 268; with compute 104 (Confident 66 / Likely 28 / Speculative 10); with N 131; with N and D 70.
- **IO use:** a latent-output measure with benchmark-difficulty "prices", comparable across vintages. It is the closest analog to a quantity index.
  - **UNVERIFIED:** the exact IRT specification of ECI. The files suggest a two-parameter structure (difficulty plus slope).

### B03. Ho et al. (2024) algorithmic-progress data
- **Sheet:** `https://docs.google.com/spreadsheets/d/11m8O_mU0cUkOB_5wluPne4PNsuvsKNbbVAzbYNy-NXY/export?format=csv&gid=2087221150`, linked from the repo README. 408 rows.
- **Columns:** System, Author(s), Publication date, Year, Reference, Citations, Peer reviewed?, Link, Parameters, Hardware, Training Compute, Epoch, Epoch (pretrain), Epoch (finetune), uncertain, Pretrain Dataset Size, Inferred_compute (6ND), Finetune Dataset Size, Dataset Size, Dataset(s), Perplexity (WT103), Perplexity (WT2), Perplexity (PTB), Zero-shot?, Include?, Uses Cache, Outlier?, Architecture, Base Model, GitHub, Organizations, Organization Categorization, Comments, Complete row, Tokenizer, Vocabulary.
- **Paper, Eq. 3:** L = E + A/N_eff^α·e^(−α_year(Y−Y0)) + B/D_eff^β·e^(−β_year(Y−Y0)), with N_eff = N·exp(α′(Y−Y0)) and D_eff = D·exp(β′(Y−Y0)).
- **Preferred Eq. 8** adds benchmark fixed effects in the constants: L = exp[α′_const − α_year(Y−Y0) − α_param·log(N/N0)] + exp[β′_const − β_year(Y−Y0) − β_data·log(D/D0)].
- **Estimation:** NLS; about 90 specifications compared by leave-one-out CV; bootstrap CIs; at most 3 models per paper.
- **Estimates:** see §1(4).
  - Shapley decomposition (RNN 2012 to GPT-3 2021): parameter scaling 48.6%, data scaling 32.4%, parameter efficiency 2.1%, data efficiency 16.8%.
  - Overall, compute explains 60–95% of gains and algorithms 5–40%.
- **Output measurement:** perplexity is per-token and tokenizer-specific; the sheet records Tokenizer and Vocabulary, which enables a byte-level normalization check.

### B04. ObsScaling (Ruan, Maddison, Hashimoto; NeurIPS 2024)
- **Files** in `eval_results/`: `base_llm_benchmark_eval.csv` (148 models; 123 with N and D; 39 families), `base_llm_benchmark_eval_sub_10b.csv` (92), `base_llm_leaderboard_v2_eval.csv` (72; BBH, GPQA, IFEval, MATH-hard, MMLU-Pro, MuSR), `instruct_llm_benchmark_eval.csv` (27; Arena-Elo, MTBench), `base_llm_post_training_eval.csv`, `base_llm_emergent_capability_eval.csv`, `instruct_llm_agent_eval.csv`. Apache-2.0.
- **Model:**
  - σ^−1(E_m) ≈ βᵀS_m + α.
  - S_m ≈ θ_f·log C_m + ν_f (family-specific).
  - B_{i,m} ≈ γ_iᵀS_m.
  - PC-1 explains about 80% of variance and the top 3 about 97%.
  - Original sample: 77 models in 21 families (47 train / 30 holdout).
- **Quirks:**
  - FLOPs = 6ND exactly (100%).
  - MoE sizes are inconsistent: Mixtral-8x7B is listed at 45 (≈ total) but Mixtral-8x22B at 39 (active); DeepSeek-V2 at 21 (active); Qwen2-57B-A14B at 24.
  - Several models lack D.

### B05. Sloth (Maia Polo et al.; NeurIPS 2025)
- **Files** in `data/`: `data_v2.csv` (188 rows; 175 with N and D; 33 families; also `Instruct`, `date`, `FLOPs (1E21)`, LB-v2 scores and BBH subtasks), `data_v1.csv`, `tokens.csv`, `training_tokens.csv`, `open-llm-leaderboard_old.csv` (1.75 MB), `open-llm-leaderboard_new.csv`, `subscenario_scores.csv` (7.6 MB), and `test_time_scaling_data.npy` (102 MB, not needed). MIT.
- **Model:** θ_ik(s,t) = α_ik + β_kᵀ(log s, log t, log s·log t); η_i = Λθ_i + b; benchmark-specific links σ_j. Here α_ik is a family-level efficiency that "absorbs all hidden factors specific to family".
- **Sample:** 164 models, 30 families, 12 benchmarks.

### B06. Open LLM Leaderboard (Hugging Face)
- **v2 `open-llm-leaderboard/contents`:** 4,576 rows, last modified 2025-03-20 (archived).
  - `#Params (B)` has min −1 (the missing-value code) and median 8.03.
  - `Type`: pretrained 275, continuously pretrained 58, chat 718, fine-tuned 1,785, merges 1,724, multimodal 9, other 7. `MoE` True 76; `Merged` True 715; `Official Providers` 470.
  - Scores (normalized and raw) for IFEval, BBH, MATH L5, GPQA, MuSR and MMLU-PRO; CO₂ cost; Base Model lineage.
- **v1 `open-llm-leaderboard-old/contents`:** 7,260 rows; ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K.
- **No D.** Merge with Epoch, ObsScaling or Sloth for tokens.
- **IO use:** the derivative-product market (fine-tunes and merges use a base model as an *intermediate input*), and quality-ladder distributions.

### B07. LMArena leaderboard dataset (CC-BY-4.0)
- `lmarena-ai/leaderboard-dataset`. The `text` config has `latest` (10,606) and `full` (1,135,923) splits.
- Also configs `text_style_control`, `text_factuality`, `vision`, `search`, `webdev`, `agent*`, `image_edit`, `text_to_image`, `document`, and others.
- **Columns:** model_name, organization, license, rating, rating_lower, rating_upper, variance, vote_count, rank, category (29 categories), leaderboard_publish_date (2023-05-08 to 2026-09-13). 422 models, 62 organizations.
- **IO reading:** Bradley–Terry ratings are identified only up to location. Comparisons over time need anchoring (like a price deflator), which makes them TFPR-like. Vote counts are a crude usage and exposure proxy.

### B08–B11. Prices, usage, hardware
- **OpenRouter:**
  - `GET https://openrouter.ai/api/v1/models` returned 456 models, each with `pricing.prompt/completion/input_cache_read` (USD per token), `architecture.tokenizer`, `created`, `hugging_face_id` (181 non-empty), `context_length`, `knowledge_cutoff`.
  - Per-provider prices: `/api/v1/models/{author}/{slug}/endpoints` (e.g., Llama-3.1-70B-Instruct: DeepInfra 4e−7/4e−7 at fp8; Bedrock 7.2e−7/7.2e−7).
  - History: Wayback CDX has 32 captures collapsed by month (first 2023-07-26, last 2026-09-02).
  - **UNVERIFIED:** terms of use for redistribution.
- **Hugging Face model API:** `https://huggingface.co/api/models/{id}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=safetensors`. Example Llama-3.1-8B: downloads 555,281 (30 days), all-time 28,056,284, safetensors.total 8,030,261,248, which is exact N.
- **Epoch ML hardware:** `https://epoch.ai/data/ml_hardware.csv`, 176 accelerators. Columns: release date, release price, BF16/FP8/FP4 FLOP/s, memory and bandwidth, TDP, price-performance, "ML models".
  - Also available: `gpu_clusters.csv` (304 KB) and `ai_companies.csv` (11 rows; revenue, valuation, staff).
- **Epoch inference-price insight** (Cottier, Snodin, Owen, Adamczewski, 2025-03-12): the price to reach fixed performance fell 9× to 900× per year depending on benchmark (e.g., about 40×/yr at GPT-4-level GPQA Diamond). No download link was found.
- **Artificial Analysis:** free API with an x-api-key; returns intelligence/coding/math indices, benchmark scores, price per 1M tokens (input/output/blended), output speed and TTFT; 1,000 requests/day; attribution mandatory. **The user must create the account; I did not.**
- **HELM:** raw results in `gs://crfm-helm-public/<project>/benchmark_output`, hundreds of GB (Classic over 5 TB). Use summary files only.

---

## 5. Diagnostics computed this session (identification-relevant)

| Dataset | n | corr(log N, log D) | sd(log(D/N) given log C) | D/N range | Note |
|---|---|---|---|---|---|
| Chinchilla (Epoch extract) | 245 | +0.244 | 1.495 | 0.036–341 | D = C/(6N) constructed |
| Step Law dense (all hparams) | 1,911 | +0.162 | 0.783 | 19–466 | 17 (N,D) cells |
| Step Law MoE (total N) | 708 | −0.013 | 0.002 | 0.93–9.3 | only sparsity varies |
| Farseer `1222_full` | 404 | −0.041 | 1.962 | 0.31–2,570 | ≈ orthogonal design |
| Datablations contour (ColPret) | 229 | +0.328 | 2.128 | 0.036–63,800 | includes repetition |
| ObsScaling base (cross-lab) | 123 | +0.453 | 1.718 | 2.1–209,000 | FLOPs = 6ND for 100% |
| Sloth `data_v2` (cross-lab) | 175 | +0.355 | 1.532 | 2.1–24,000 | 33 families |
| DataDecide (final ckpts) | 350 configs | +1.000 | 0 | 100 | D = 100N |
| Pythia (final) | 8 | n/a (D constant) | n/a | ~25–4,300 | D ≈ 300B |

Reading the table:
- Cross-lab data show *moderate* positive corr(log N, log D) (0.35–0.45). That is consistent with labs scaling both inputs together along their perceived expansion path, but it is far from the perfect collinearity of any single lab's compute-optimal ray.
- The large sd(log D/N given log C) in cross-lab data comes mostly from heterogeneous over-training choices (inference wedge) and heterogeneous beliefs (Kaplan-era vs Chinchilla-era), which is itself identifying variation *if* those choices are orthogonal to TFP. Whether they are is a testable question.

---

## 6. Data pitfalls ↔ IO problems

| Data pitfall (where observed) | IO analog | Strength | Remedy / estimator |
|---|---|---|---|
| Compute-optimal models only: N*(C), D*(C) deterministic (on-frontier subsets; DataDecide D = 100N; Pythia D fixed) | ACF (2015) functional dependence / collinearity of a flexible input with the state | exact | Use IsoFLOP (off-frontier) variation as the "experiment"; report identified combinations (e.g., a = β/(α+β)) |
| C imputed as 6ND (ObsScaling 100%; Epoch 53%) | Deterministic accounting identity in inputs (e.g., capital constructed from investment by perpetual inventory) | exact | Never regress on C together with N and D; treat 6ND as a cost identity, not data |
| Selection into "Notable" by SOTA/citations/users | Selection on the dependent variable (truncation) | close | Use Frontier/Large-scale subsets (selection on inputs); Heckman-type or bounds |
| Closed labs don't report N and D (only 21 API-only LMs since 2023 with N, D and C > 1e21) | Olley–Pakes attrition/selection on productivity; non-random disclosure | close | Model the disclosure probability (open weights, org type); partial identification |
| Unreleased or failed runs, restarts after loss spikes (e.g., pythia-1b switched to bf16 after an fp16 spike) | Survivorship / exit | close | Controlled sweeps (A-data) have no selection; compare A-based vs B-based elasticities |
| MoE total vs active parameters mixed within a column (Epoch, ObsScaling) | Capital stock vs capital services; utilization | close | Use active N for flow services, total N for memory/"capital"; Step Law MoE separates both |
| Embedding vs non-embedding N (Kaplan vs Chinchilla; Porian's conventions) | Definition of the capital aggregate; measurement error in K | close | Sensitivity across conventions (Porian columns), Pearce & Song reconciliation |
| Dataset size ≠ tokens seen; epochs missing; units vary | Input measured as stock not flow; hours vs employees | close | D_seen = size × epochs; drop non-token units; Muennighoff's effective data |
| Repeated data (epochs), data filtering and quality | Depreciation; input quality / vintage (effective units) | close | D_eff = U(1 + R*(1 − e^(−R/R*))); corpus fixed effects (DataDecide, open-sci-ref) |
| Synthetic or distilled data (Phi outlier in ObsScaling) | Gross output vs value added; omitted intermediate inputs (GNR 2020) | close | Add teacher FLOPs as an intermediate input; gross-output production function |
| Loss in nats per token depends on tokenizer; different eval corpora | Output measured in firm-specific units; quantity vs revenue deflation (TFPQ vs TFPR) | close | Bits per byte on a common corpus (OLMo-ladder bpb; WikiText-103 in several datasets) |
| Benchmarks bounded in [0,1]; emergence | Bounded, nonlinear transform of latent output; threshold measurement | close | Two-step latent models (Bhagia Eq. 2; ObsScaling σ^−1; Sloth links; ECI) |
| Elo / Arena ratings relative, anchor drift | Revenue-based productivity with an unknown price index | loose | Anchor models across snapshots; use ratings only within snapshot |
| Seeds: same (N,D), different loss (PolyPythias, DataDecide, OLMo reruns) | i.i.d. shock ε vs persistent ω (OP/LP decomposition) | exact | Estimate Var(ε) from replicates; test whether residual variance exceeds Var(ε) |
| Checkpoints of one run used as separate observations | Serially dependent within-plant panel; schedule state = unobserved state variable | close | Cluster by run; use WSD/constant-LR designs (Hägele) or completed runs only |
| Hyperparameters tuned per scale (Porian; Step Law) | Flexible (static) inputs chosen optimally; transmission bias if omitted | close | Condition on optimized hyperparameters (frontier) or model the policy (GNR FOC) |
| Two measurements of C (6ND vs hardware × hours × FLOP/s × MFU; 126 LMs) | Repeated measurements for errors-in-variables | close | IV with the second indicator; Schennach/Hausman-type corrections |
| Hardware release prices and FLOP/$ over time | Input price shifters (cost-side instruments; Nerlove) | loose | IV for scale in cross-lab regressions; dual (cost-function) estimation |

---

## 7. Back-of-envelope: revealed inference wedge (my computation; for the modeling strand to redo properly)

**Derivation.**
- The lab minimizes training plus inference FLOPs, 6ND + 2N·D_inf, subject to L(N,D) = ℓ. This is Sardana et al., Eq. 3.
- With L = E + AN^−α + BD^−β, the first-order conditions give
  θ ≡ (αAN^−α)/(βBD^−β) = 1 + D_inf/(3D),
  so the implied inference demand is **D_inf = 3D(θ − 1)**.
- With training cost only, θ = 1. This mirrors De Loecker–Warzynski: the ratio of output elasticities to cost shares reveals a wedge. Note that αAN^−α is the elasticity of reducible loss with respect to N, times the reducible loss.

**Results.** (Llama-3.1-405B uses 15.6T tokens; D values are publicly reported token counts.)

| Model | D/N | θ (Hoffmann A3) | D_inf / D (Hoffmann A3) | θ (Besiroglu) | D_inf / D (Besiroglu) |
|---|---|---|---|---|---|
| Chinchilla-70B | 20 | 0.71 | −0.86 | 1.03 | 0.09 |
| Gopher-280B | 1.1 | 0.29 | −2.14 | 0.36 | −1.91 |
| GPT-3-175B | 1.7 | 0.34 | −1.99 | 0.43 | −1.72 |
| Llama-2-7B | 286 | 1.72 | 2.17 | 2.62 | 4.85 |
| Llama-2-70B | 29 | 0.79 | −0.63 | 1.17 | 0.52 |
| Llama-3-8B | 1,875 | 2.92 | 5.77 | 5.22 | 12.65 |
| Llama-3-70B | 214 | 1.40 | 1.20 | 2.45 | 4.36 |
| Llama-3.1-405B | 38.5 | 0.78 | −0.66 | 1.35 | 1.06 |
| Qwen1.5-0.5B | 4,800 | 4.44 | 10.3 | 7.00 | 18.0 |

The compute-optimal D/N at C = 1e24 is 62 under Hoffmann A3 and 18 under Besiroglu. This is the same 70-vs-20 inconsistency that Besiroglu et al. highlight.

**Lessons for the paper.**
1. The wedge is only as good as the elasticity estimates. Estimator choice alone moves θ by about 2×.
2. θ < 1 (GPT-3, Gopher) is not a "negative markup". It reveals *beliefs* (Kaplan-era production function) or constraints. This is the analog of the DLW critique that wedges also absorb adjustment costs and optimization frictions.
3. A lab-specific technology (A_j, B_j) changes θ. Estimate family-specific elasticities where possible (Sloth/ObsScaling families, or controlled sweeps per lab: OLMo ladder for AI2, Step/Farseer for StepFun, Porian/Gadre for OpenLM).
4. Data scarcity (a Muennighoff-style shadow price on D) pushes θ the other way. Over-training in the presence of repeated data understates the wedge.

---

## 8. Contribution ideas grounded in available data

1. **"IsoFLOP designs are instruments" paper section.**
   - Using Farseer and Chinchilla, show formally and numerically that on-frontier data identify only a = β/(α+β) (or the frontier elasticity), while off-frontier variation identifies α and β separately.
   - Quantify with the condition number and bootstrap SEs, comparing full design vs frontier-only vs DataDecide-final (perfect collinearity).
2. **A known-noise output model.** Gadre CIs plus PolyPythias, DataDecide and OLMo rerun seed variance give Var(ε). Test whether Chinchilla-form residuals exceed seed noise (misspecification) and estimate the "TFP" of corpora/plants net of noise.
3. **Common-output bridge for algorithmic progress.**
   - Use WikiText-103 and/or bpb (present in the OLMo ladder, DataDecide-ppl and Ho et al.) to put 2012–2025 models and sweeps on one tokenizer-free output scale.
   - Then estimate factor-augmenting progress with *within-vintage off-frontier variation* (controlled sweeps from 2022, 2024 and 2025). This addresses the α_year/β_year non-identification flagged by Ho et al. (the Diamond–McFadden–Rodriguez analog).
4. **Hyperparameters as proxy variables.** Step Law shows optimal LR and BS are monotone policy functions of (N, D). Across open-weight releases, reported peak LR and batch size (from tech reports or configs) could proxy unobserved lab productivity, as in Levinsohn–Petrin, provided the monotonicity/scalar-unobservable conditions hold. This is speculative and needs a scalar-unobservable argument.
5. **Revealed inference wedge panel.** θ_i for all open-weight LMs in Epoch, with family-specific elasticities. Correlate with ex-post demand (HF downloads, OpenRouter provider count and prices over time, LMArena votes). This tests whether over-training anticipates demand, a markup-like wedge validated against outcomes.
6. **Gross-output vs value-added LLM production.** Add teacher/synthetic-data FLOPs (distillation) as intermediate inputs. The Phi outlier is the motivating fact (ObsScaling). Busbridge et al. give the controlled benchmark, but its data are not public; the literature's functional form could be used.
7. **Train-time vs test-time isoquants.** Jones boardlaw (SQLite) and Monkey Business (coverage vs k at 4 model sizes). Estimate the marginal rate of technical substitution between training compute and inference compute.
8. **Measurement-error correction for C.** Use the 126 Epoch LMs with a hardware-time-based second measure as a repeated indicator. Show how measurement error attenuates the cross-lab compute elasticity.

---

## 9. Warnings, uncertainties and things not verified

- **Licenses.** Step Law, Farseer and Epoch's `analyzing-chinchilla` have *no license file*. Treat as "all rights reserved": cite, don't redistribute raw files, and ask the authors. PolyPythias and Open LLM Leaderboard HF cards have no license field. OpenRouter and Artificial Analysis terms were not read in full.
- **Living data.** Epoch (AI models, benchmarks, ECI) and LMArena change weekly. Freeze snapshots with dates in `data/`. Numbers here are from 2026-09-23. The Epoch benchmark page lists ECI leaders and models dated through 2026-09, and the database contains 2026 model names. Treat these as data, not as claims I verified independently.
- **Porian row count** not computed (the pickle needs pandas). Farseer's `D/N` column is mislabeled. ColPret's datablations `tokens_per_epoch` and `flops` columns look wrong.
- **DataDecide:** "350 models" on the card = 25 recipes × 14 sizes; the 3 seeds give about 1,050 runs (seed labels "default", "small aux 2/3", "large aux 2/3"; large-scale seeds appear only for some sizes, judging by the frequencies). Confirm when loading.
- **ECI** construction details (IRT variant) are UNVERIFIED. `model_versions` is empty for some top rows, so use name joins.
- **Sardana et al.** use α = 0.336 and β = 0.283 in their reproduction of Chinchilla, while Besiroglu cite Hoffmann's Approach 3 as α = 0.3392, β = 0.2849. That is a rounding and reporting inconsistency across papers; use Besiroglu's table as the canonical transcription.
- **Kaplan (2020) constants** (from ar5iv HTML): the joint L(N,D) fit (Table 2) has α_N = 0.076, α_D = 0.103, N_c = 6.4e13, D_c = 1.8e13. The separate fits (Table 5) have α_N = 0.076, N_c = 8.8e13, α_D = 0.095, D_c = 5.4e13. Compute-efficient scaling is N ∝ C^0.73 and D ∝ C^0.27. Eq. 1.5: L = [(N_c/N)^(α_N/α_D) + D_c/D]^α_D. This is a CES-type aggregator with *unequal* inner exponents (N^−0.74 vs D^−1) and no irreducible term.
- **CES mapping.** With α = β = ρ, Y ≡ (L − E)^(−1/ρ) = (AN^−ρ + BD^−ρ)^(−1/ρ) is exactly CES and CRS in (N, D), with σ = 1/(1 + ρ).
  - Muennighoff's fitted α = β = 0.353 gives σ ≈ 0.74.
  - Gadre also imposes α = β.
  - Besiroglu's unrestricted α = 0.348, β = 0.366 cannot reject equality at conventional levels, judging by the SEs (my reading; to be tested formally).
  - With α ≠ β the isoquants are non-homothetic (a CES-like "additive power" form). This is **my derivation**, not from a fetched source.
- **Descriptive D/N trends and θ computations** in §4/§7 are my own quick calculations on public data. They are illustrative, not paper-ready. Model token counts for Llama/Qwen come from ObsScaling/Epoch rows (Llama-3 15T, Llama-2 2T, Qwen1.5-0.5B 2.4T), Chinchilla 1.4T, Gopher and GPT-3 300B.
- **Venues** marked UNVERIFIED in `data.bib`: Pythia (ICML 2023?), Muennighoff (NeurIPS 2023?), Chatbot Arena (ICML 2024?), Choshen (ICML year).
- **Search budget:** the WebSearch budget for the session was exhausted near the end. Remaining checks used direct fetches and APIs. Items not checked for public data: Kumar precision runs, Busbridge distillation runs, DeepSeek/Llama IsoFLOP raw data, and a public OpenRouter token-usage (rankings) API.

---

## 10. Source URLs used (all fetched or queried 2026-09-23)
- Epoch data: https://epoch.ai/data/ai-models ; https://epoch.ai/data/all_ai_models.csv ; https://epoch.ai/data/notable_ai_models.csv ; https://epoch.ai/data/large_scale_ai_models.csv ; https://epoch.ai/data/frontier_ai_models.csv ; https://epoch.ai/data/ai_models.zip ; https://epoch.ai/benchmarks ; https://epoch.ai/benchmarks/use-this-data ; https://epoch.ai/data/benchmark_data.zip ; https://epoch.ai/data/ml_hardware.csv ; https://epoch.ai/data/gpu_clusters.csv ; https://epoch.ai/data/ai_companies.csv ; https://epoch.ai/data ; https://epoch.ai/data-insights/llm-inference-price-trends
- GitHub: epoch-research/analyzing-chinchilla ; epoch-research/lm-algorithmic-progress ; formll/resolving-scaling-law-discrepancies ; mlfoundations/scaling ; huggingface/datablations ; ryoungj/ObsScaling ; EleutherAI/pythia ; allenai/OLMo-ladder ; allenai/OLMo ; IBM/ColPret ; step-law/steplaw ; Farseer-Scaling-Law/Farseer ; felipemaiapolo/sloth ; mcleish7/gemstone-scaling-laws ; LAION-AI/open-sci-ref-0.01 ; andyljones/boardlaw ; epfml/schedules-and-scaling ; OpenBMB/MiniCPM ; KempnerInstitute/loss-to-loss-olmo
- Hugging Face: datasets open-llm-leaderboard/contents, open-llm-leaderboard-old/contents, lmarena-ai/leaderboard-dataset, allenai/DataDecide-eval-results, allenai/DataDecide-ppl-results, EleutherAI/polypythias-evals, ScalingIntelligence/monkey_business, open-sci/open-sci-ref-0.01-logs ; model orgs Gemstone-Models, open-sci, allenai (OLMo-Ladder-760M-0.5xC) ; models API meta-llama/Llama-3.1-8B
- Other: https://openrouter.ai/api/v1/models ; Wayback CDX for openrouter.ai/api/v1/models ; https://artificialanalysis.ai/documentation ; https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/ ; https://andyljones.com/boardlaw/ ; Google Sheet export of Ho et al. data
- Papers (arXiv abs/HTML/API): 2203.15556, 2001.08361 (ar5iv), 2404.10102, 2406.19146, 2403.08540, 2305.16264, 2304.01373, 2503.09543, 2412.04403, 2504.11393, 2410.11840, 2503.04715, 2506.10972, 2411.04330, 2502.08606, 2405.18392, 2405.10938, 2412.06540, 2401.04757, 2403.05812, 2401.00448, 2408.03314, 2104.03113, 2407.21787, 2403.04132, 2211.09110, 2502.06857, 2509.09009, 2412.04315, 2402.00838, 2501.00656, 2404.06395, 2406.12907, 2401.02954, 2407.21783, 2411.12925, 2402.07871, 2501.12370, 2407.13623, 2511.21622, 2507.07931, 2405.21015, 2202.05924, 2211.04325, 2212.05153, 2005.04305, 2408.00724, 1712.00409, 1909.12673, 2304.15004, 2206.07682, 2102.01293
- Crossref (IO references): DOIs 10.3982/ECTA13408, 10.2307/2171831, 10.1111/1467-937X.00246, 10.1257/aer.102.6.2437, 10.1086/707736, 10.2307/1905432, 10.1086/697204, 10.1257/aer.98.1.394, 10.1257/jel.49.2.326
