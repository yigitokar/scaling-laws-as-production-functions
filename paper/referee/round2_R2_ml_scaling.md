# Round-2 Referee Report: Referee 2 (ML scaling laws)

**Manuscript:** "What Optimizing Labs Reveal: Scaling Laws as Production Functions" (Okar and Claude). Revised version 2, working paper of September 2026, 175 pages.
**Round-1 report:** `paper/referee/R2_ml_scaling.md` (recommendation: major revision).
**Referee perspective:** I am a machine-learning researcher who has trained and fitted scaling laws at a frontier lab. As in round 1, I focus on the ML content: the curvature estimates, FLOP and parameter conventions, the released-model sample, extrapolation, and the design of the authors' own experiment. I leave the IO econometrics to the other referees.

**Recommendation:** Major revision (second round). The paper has improved a great deal, and most of my round-1 concerns are resolved or well on the way. I recommend major rather than minor revision for three reasons:
- The paper's lead number, "σ\* ≈ 0.7", conflicts with the paper's own estimates at the largest compute budgets.
- Part of the headline wedge evidence contradicts Proposition 2(iv).
- The pre-registered experiment is still a set of placeholders in the introduction, Section III.E, Section IV.D and the conclusion. The authors' own power analysis shows that the experiment's inference plan must change, and the paper does not yet say so.

Page numbers refer to the compiled version-2 `main.pdf`. The numbers I cite come from the paper, from `output/tables/*.csv` (mainly `ra1_modelfree_isoflop_budgets.csv`, `ra1_modelfree_isoflop_summary.csv` and `ra2_wedge_models.csv`), from `output/memos/m9_sweeps.md` §0, and from the sweep code in `code/sweep/`. Where a number is my own recomputation, I say so. I looked at the experiment's code, its data fields and its pre-registration only. I did not use its loss values.

---

## 1. What changed

The paper has been refocused around one idea: optimization is "double-edged" for measurement. The main changes relevant to my report are these.

1. **A model-free estimator of σ\*** (Proposition 4). It is the curvature of the IsoFLOP profile divided by twice the slope of the frontier. It is applied to Chinchilla, Llama 3, three Marin ladders, Porian et al. and Farseer's local path (Table 1, Figure 3). The headline is a random-effects mean of 0.695 [0.673, 0.717].
2. **A new reference technology**, the Chinchilla fit with κ free (σ\*_κ = 0.701). The wedge is reported as an expenditure share, s = (w − 1)/w. Token counts T now require a price ratio p.
3. **A "clean inference-demand sample" of 77 models.** It drops re-releases, research suites, replications, MoE models, distilled and pruned models, multimodal models, models without a base checkpoint, non-transformers and undocumented token counts. The robustness appendix (F) reports 32 technologies, conventions, MoE bounds, lab-own technologies, partial identification, conduct tests, family budgets, the cost side and validation.
4. **Proposition 5.** It gives partial identification of M\*(C) beyond the designs, and identification of the *sign* of w − 1 from the location of the IsoFLOP minimum alone. The paper reports 86 percent of models identified as over-trained under IsoFLOP anchors.
5. **A direct check of parametric extrapolation in M on Farseer** (Figure 6, Table F6). Parametric forms understate the local wedge at high M.
6. **A redesigned experiment with a committed pre-analysis plan.** It adds four-layer runs to high M, learning-rate corners, a FineWeb learning-rate calibration, seeds in both corpora and WikiText-103 evaluation. The plan is `paper/notes/m9_preanalysis_plan.md`, commit bd5c0ad, and a power calculation was run before estimation.
7. **Other changes.** Terminology is corrected ("IsoFLOP-minima frontier"; Hoffmann's Approach 1 also estimates factor demands). There is a practitioner table of the compute cost of over-training (Table D14). The observational material has moved to Appendix E.

## 2. Overall assessment

This is a much better paper. Several parts are now things I would cite and use:
- the model-free σ\* estimator and its invariance to E, κ and loss units;
- the local-wedge check of parametric extrapolation on Farseer;
- the sign result of Proposition 5(iii);
- Table F1, which states what the wedge identifies under each objective;
- the family-budget condition of Proposition 2(iv);
- the cleaning table (Table F2) and the MoE bounds (Table F4).

The authors took the two routes I offered in round 1 seriously. The abstract no longer reports "2.2×". Levels are presented as conditional on a technology, and the ordinal content is stated plainly: the R² of ln ŵ on ln M is 0.9999 (p. 28).

What remains, in order of importance:

1. **The curvature at frontier scale.** The paper's own budget-level estimates put σ\* at 0.56–0.66 at the largest budgets (6×10²⁰ FLOP and above) of Chinchilla, Llama 3 and Farseer. The headline "about 0.7" is an average over 10¹⁸–10²² FLOP, and its interval leaves out the FLOP-accounting and convention uncertainty that the paper itself quantifies (new Major 1).
2. **Common-D families.** Table 2 and the Llama 3 narrative in Section IV.C treat member wedges of common-token-budget families as revealed preferences. Proposition 2(iv) says they are not. The data-cap reading of a common D is also missing (new Major 2).
3. **The experiment.** There is no neutral evaluation for the main grid, and WikiText is not neutral for FineWeb-Edu in any case. The pre-registered inference does not work on this design, by the authors' own Monte Carlo, and this deviation is not disclosed. There are no tuning checks at the high-M shapes (new Major 3).
4. **Level claims.** The abstract and conclusion still state some level claims more strongly than the paper's own ranges support (new Major 4).
5. **Extrapolation in M.** The "conservative" direction rests on one recipe (new Major 5).
6. **Smaller items.** Lab-own technologies are applied inconsistently. There are gaps in the ML content (post-training compute; loss is not capability). Literature that is already in the bib is still not cited (new Majors 6–8).

---

## 3. Status of my round-1 major comments

| # | Round-1 comment | Status | What remains |
|---|---|---|---|
| 1 | Headline wedge used the κ = 1 curvature the paper rejects; curvature uncertainty understated | **Largely resolved** | The reference is now κ-free (σ\* = 0.701), with σ-sensitivity reported (p. 30; F3). New issue: which σ\* applies at frontier compute (new Major 1). |
| 2 | FLOPs are not costs; who pays; what T contains | **Largely resolved** | s is an expenditure share; T needs p; conduct is split by who serves. Post-training and mid-training compute on the training side are not discussed (new Major 7). |
| 3 | Extrapolation manufactures robustness; ordinal content; levels need external checks | **Partially resolved** | In-support versus beyond-design results, the sign result and the Farseer check are all done. The "conservative" claim still rests on one recipe. There is no second check at high M, and no cooldown of released checkpoints (new Major 5). |
| 4 | Conventions: embeddings, MoE, distillation, multimodal, tokenizer | **Resolved except (e)** | Non-embedding technologies, head FLOPs, MoE bounds and exclusions are all in place. D is still not normalized by bytes (acceptable for released models; see Major 3(f) for the experiment). |
| 5 | Rivals: fixed-D families, stakes, tiers, latency, sunk compute, capability ≠ loss | **Partially resolved** | Proposition 2(iv), multi-tier bunching with placebos, a serving-footprint test and on-device coding are all added. The tests are candidly low-powered. New inconsistency on common-D families (new Major 2). Capability ≠ loss is not discussed (new Major 7). |
| 6 | Reference technology; circular stated-intent checks | **Largely resolved** | Modern laws (DeepSeek, MiniCPM, Meta, Marin) are in the set, lab-own technologies are primary for their models, and the stated-intent checks are reframed ("on Meta's path by construction"). Chinchilla κ-free is defensible as the reference because its curvature matches the model-free estimate. Lab-own curvature is applied inconsistently (new Major 6). |
| 7 | Experiment cannot answer its questions as designed | **Partially resolved** | High-M four-layer runs, LR corners, FineWeb LR calibration, seeds in both corpora, WikiText, a committed PAP and an ex-ante power calculation are all added. Remaining: no neutral output on the main grid; inference failures found by the authors' own power analysis and not disclosed; no LR or weight-decay checks at the high-M shapes; the M–size confound; D not in bytes; no 100–300M anchors (new Major 3). |
| 8 | Evidence on σ narrower than presented | **Largely resolved; new issue** | The model-free estimates replace the κ = 1 range. "Five studies" is stated, DataDecide is out of the headline, the κ-robustness table (D10) and the practitioner table (D14) are added. New issue: drift of σ\* with scale, and accounting uncertainty (new Major 1). |
| 9 | Chinchilla and Kaplan facts | **Largely resolved** | (a) and (d) are fixed. (b) Chinchilla's D is still C/(6N), not rebuilt from the architecture table; this is disclosed (B2). (c) Kaplan's joint law is still presented as a κ-family member without qualification (App. A, p. 55; Minor 15). |
| 10 | Data handling in the released-model sample | **Largely resolved** | Deduplication, instruct-only models, suites and non-transformers are handled (F2). Corpus size versus tokens processed (Qwen2.5 18T, Qwen3 36T) and repeated tokens (e.g., stablelm-3b-4e1t) are not audited, as disclosed on p. 174 (Minor 16). |
| 11 | Usage validation weak | **Partially resolved (as far as public data allow)** | HF derivative counts from the model tree are added. OpenRouter per-model volumes cannot be collected (ToS). The paper now says plainly that levels cannot be validated. The developer-serving test has 18 clusters (p = 0.39). |
| 12 | Missing ML literature | **Partially resolved** | Bian, Roberts, Busbridge, DeepSeek LLM, MiniCPM, de Vries, Hägele, and Clark, Krajewski and Abnar (appendix) are now cited. Still uncited although in `references.bib`: Villalobos and Atkinson (2023), Erdil (2024, 2025), Kumar et al. (2024), Lourie et al. (2026), Bergsma et al. (2025), Tissue et al. (2024) and Goyal et al. (2024) (new Major 8). |

**Round-1 minor comments.** Most are addressed:
- starting values no longer include the truth;
- the Monte Carlo has a clustered-noise variant;
- the 44M model is explained;
- the trunk sharing of data order is stated;
- the 1.57 coincidence is confirmed;
- the training-versus-validation-loss step in the Porian decomposition is quantified (at most 0.005);
- the tokenizer effect is recomputed (8 percent);
- rank = rank in M is stated;
- only confident closed-model rows are used;
- 405B leave-one-out is reported;
- benchmark harmonization and hardware-based compute reliability are documented.

Not addressed:
- Minor 14: DataDecide tilts from final checkpoints only.
- Minor 26: Figure 1 still uses total N with no non-embedding marker.
- Minor 29, in part: Table 2 has no embedding-share column.

---

## 4. New major comments

### Major 1. The curvature that the wedge needs is at frontier compute, and the paper's own estimates put it below 0.70 there

The paper calls σ\* ≈ 0.70 "the best-measured object" (p. 35). It recommends 0.7 "for substitution between parameters and data in frontier-scale language models" (p. 41). The abstract reports "about 0.7" without qualification. Four pieces of the paper's own evidence point lower at scale.

**(a) Budget-level estimates at the largest budgets.** These come from `ra1_modelfree_isoflop_budgets.csv`, plotted in Figure 3(b):
- Chinchilla at 6×10²⁰, 10²¹ and 3×10²¹ FLOP gives σ\*_b = 0.564, 0.617 and 0.579.
- Llama 3 at 3×10²⁰, 6×10²⁰ and 10²¹ gives 0.618, 0.585 and 0.609.
- Farseer's local path at its largest compute level, 10²¹, gives 0.658 (s.e. 0.011; `ra1_modelfree_farseer_path.csv`), against 0.72–0.74 at 2–5×10²⁰.
- At 10²¹ FLOP, then, all three designs that reach that budget give 0.61–0.66.
- Between-budget heterogeneity is significant within both designs: Cochran Q = 15.6 (p = 0.025) for Chinchilla, and Q = 50.7 with I² = 0.86 for Llama 3.
- The inverse-variance (fixed-effect) pooled estimate for Llama 3 is 0.628, against the random-effects 0.660 reported in Table 1. The largest valid budget carries 62 percent of the fixed-effect weight.

**(b) Drift.** The paper reports drift of −0.072 per decade (s.e. 0.026) in Chinchilla and −0.053 (0.010) in Llama 3 (p. 23). It concludes only that "if it is real, σ\* at frontier compute is below 0.70." The designs that show no drift, Marin's three ladders, stop at 3×10²⁰ FLOP.

**(c) The reference technology itself.** The reference's σ\*_κ falls when the smallest runs are removed: 0.690 without the smallest 10 percent of N and 0.648 without the smallest 25 percent (Table D10, p. 123).

**(d) The homogeneity result.** The between-design homogeneity (Q = 4.1, τ = 0) compares design means taken over different compute windows. With τ = 0 the weights are inverse variances. By my calculation from Table 1's standard errors:
- about a third of the weight is on the three Marin corpora, which share one codebase and all stop at 3×10²⁰ FLOP;
- about 40 percent is on Farseer's local path, which stays below about 10²¹ FLOP;
- only about a quarter is on the two designs that reach budgets of 10²¹–10²².

"No detectable heterogeneity" across designs is therefore compatible with a common downward drift in compute.

**Accounting and convention uncertainty.** Separately, the headline interval [0.67, 0.72] leaves out uncertainty the paper quantifies itself:
- FLOP accounting shifts σ\* by 0.44η, about ±0.04 for |η| ≤ 0.1 (p. 21).
- Marin's configuration count gives 0.64–0.66 (p. 21).
- Counting embeddings moves Farseer's σ\*_κ from 0.710 to 0.668 (p. 24).
- The authors' own power memo (m9 §0.1, point 3) finds that the same technology has σ\* lower by 0.04–0.08 in total-N than in non-embedding units, "by construction."

σ\* is an elasticity with respect to a *measured* input, so the convention is part of the estimand. Table 1 mixes conventions across rows: total N, C/(6D), non-embedding N, and non-embedding plus head.

**Why this matters for Section IV.** In my recomputation, holding the reference M\*(C) fixed, the median share s on the clean sample is:

| σ\* | median s | Llama 3 8B ŵ | Qwen3 0.6B ŵ |
|---|---|---|---|
| 0.701 (reference) | 0.75 | 6.7 | 29 |
| 0.673 | 0.79 | – | – |
| 0.660 | 0.81 | 10.0 | 59 |
| 0.648 | 0.83 | – | – |
| 0.60 | 0.88 | 19.6 | 197 |

The magnitude bounds of Prop. 5(iv) use 1/σ\* − 1 ∈ [0.40, 0.52], that is σ\* ∈ [0.66, 0.71] (p. 33, Table F7). That range excludes the values the largest budgets point to. Section V's data-wall conclusion is not affected: it already includes σ\* = 0.60. The data-demand translation D/D\* = w^{σ\*/[2(1−σ\*)]} (p. 37) is affected.

**Requests.**
1. **Report the budget-level estimates.** Put the per-budget σ\*_b, with window diagnostics (points left and right of the minimum), in an appendix table. Report the fixed-effect pooled estimate beside the random-effects estimate.
2. **Pool across designs by compute.** Run a meta-regression of σ\*_b on log C with design fixed effects, pooling Chinchilla, Llama 3, Marin and Farseer's local path. Report the implied σ\* at 10²², 10²³ and 10²⁴ FLOP with intervals.
3. **Check that the drift is not an artifact of the estimator.** Use:
   - symmetric windows (several budgets have 9–10 points on one side and 2–4 on the other);
   - quartic fits;
   - a frontier slope from a power-law frontier with E profiled, rather than the derivative of a log-cubic at the edge of its support;
   - leave-the-largest-budget-out;
   - a Monte Carlo under a truth whose σ\* drifts, to show that the estimator detects it.
4. **Pin down the FLOP-accounting term.** Rebuild Chinchilla's FLOPs per token for each model in Hoffmann et al.'s architecture table (their Appendix F count) to obtain η by budget. This was round-1 Major 9(b); it would turn the ±0.04 into a number.
5. **Rewrite the headline as conditional.** Present σ\* as conditional on compute window and convention. For example: "about 0.70 averaged over 10¹⁸–10²² FLOP in FLOP-implied parameter units; about 0.6–0.66 at 6×10²⁰–3×10²¹ FLOP, the largest budgets observed." Change the abstract, p. 3 and p. 41 accordingly.
6. **Carry the lower curvature into the wedge bounds.** Widen [k_L, k_U] in Prop. 5(iv) to include the curvature implied at the models' own compute, and report the median s and the Llama 3 8B wedge under it.

### Major 2. Common-D families: member wedges are used as revealed preferences, against Proposition 2(iv); and the data-cap reading is missing

Proposition 2(iv) (p. 10) and the discussion on p. 11 state that, when a family trains every size on one D, "member-level wedges are not revealed preferences." Table F9 lists the Llama 3 herd as such a family, with a family share of 0.29. Yet:

- **Table 2 (p. 31).** 10 of the 16 Panel A rows belong to common-D families, by the paper's own coding in `ra2_wedge_models.csv`: Llama 3 8B, 70B and 405B; Llama 2 70B; Qwen2.5 7B and 72B; Qwen3 0.6B and 14B; DeepSeek LLM 7B and 67B. The table presents their ŵ and s as "revealed wedges and expenditure shares."
- **The Meta narrative (p. 30).** Section IV.C builds its sharpest case on Meta: "Llama 3 8B … s = 0.88 … Meta served Llama 3 … so this is a planned serving share … the flagship was trained as if for training efficiency and the smallest model as if for serving." Under Proposition 2(iv), the 8B's wedge is fixed by the technology given the 405B's D.

The inconsistency can be resolved in either direction, but the paper must choose. Practitioners will raise a third reading of a common D, which the paper does not consider: **a binding cap on curated unique tokens.** Llama 3's 15T and Qwen3's 36T are each developer's full pretraining corpus. Labs routinely train every size on the whole curated corpus, single-epoch, because more data of equal quality is not available. Under Proposition 2(iii), a binding token cap adds μ to the denominator. The small members' measured wedges are then *lower bounds* on m_N, not uninformative. This fits Meta's stated intent that the smaller models were deliberately trained beyond compute-optimal for inference. Under this reading the 8B highlight survives as a bound. Under the family-choice reading it does not.

The knife-edge argument on p. 11 ("a common budget under member-by-member choice would be a knife-edge, so its prevalence suggests that D is chosen at the family level") ignores the cap. It is a corner, not a knife-edge.

**Requests.**
1. Add the cap case to Proposition 2(iv) and Table F1. Under a cap, a member's wedge is a lower bound on its m_N.
2. For each common-D family, say which reading the paper adopts (family choice, menu, or cap) and why. Model cards and technical reports often say whether the corpus was exhausted.
3. In Table 2, flag common-D rows and show the family share beside the member values. Revise the p. 30 narrative to match the chosen reading.
4. Report the headline median over decision units: the 45 models from families with size-specific budgets and singletons, plus one family-level share for each of the 11 common-D families. The paper already reports that dropping the 11 families leaves s at 0.75; the decision-unit version is the natural companion.

### Major 3. The controlled experiment: design gaps and an inference plan the authors' own power analysis shows must change

I comment on the design only; the results are pending. The experiment is now much better. The four-layer high-M runs, the LR corners, the FineWeb LR calibration, the seeds in both corpora, the PAP committed before estimation, and a power calculation run *before* touching the results are exactly what I asked for. Six problems remain.

**(a) The main grid has no neutral output, and WikiText is not neutral for FineWeb-Edu.**
- In `code/sweep/run_grid.py`, the main-grid jobs (tag `main`, both corpora) use the default `--evals edu,web`. FineWeb-Edu seed replicates also use `edu,web`.
- Only `lrcal`, `lrcorner`, `seedcorner`, `hiM` and FineWeb seeds are evaluated on WikiText. `results.jsonl` confirms that no main-grid endpoint has `loss_wiki`.
- Checkpoints are not saved, so the main grid cannot be re-evaluated.
- The tilt in Q2, estimated on the 44-endpoint main grids, therefore cannot be checked on WikiText. Yet p. 26 states that a corpus difference "counts as quality only if it has the same sign on the other corpus's validation set and on WikiText-103." The PAP says "where available."
- WikiText-103 is Wikipedia Good and Featured articles. FineWeb-Edu is selected by a classifier trained to find educational, textbook-like prose, so Wikipedia is closer to the edu corpus than to FineWeb. On WikiText the edu corpus has a distribution-match advantage, the very confound the rule is meant to remove.

*Requests:*
1. Retrain a subset with a genuinely mixed neutral evaluation. For example, run 4 widths × both corpora, trunk plus branches, and evaluate on several Paloma domains (C4-100-domains, books, forum text) rather than one encyclopedic set.
2. Save checkpoints for every remaining run.
3. Until then, state on p. 26 and in B4 that the main-grid neutrality verdict rests on the two in-distribution validation sets. Drop "and on WikiText-103" as a condition for the main grid.

**(b) The pre-registered inference does not work on this design, and the paper does not disclose the change.**
`output/memos/m9_sweeps.md` §0 was written before estimation, which is commendable. It finds the following on this 8-width design:
- The plan's wild cluster bootstrap covers 72–78 percent of the time for the tilt and 40–65 percent for model-free σ\*.
- The two-set decision rule returns "factor-biased" under χ = 0 in 8–17 percent of replications.
- The Q3 slope has a non-zero null: −0.02 to −0.06 in the non-embedding convention, and +0.11 in total-N with the high-M runs, even when the Chinchilla form is true.
- The memo therefore adopts a "pre-declared remedy" (D8): CR2 residuals, restricted wild bootstrap tests for the tilt, and calibration factors of 1.4–1.7 for model-free σ\* and Q3.

That is a deviation in the estimator and in the decision rule. Yet p. 26 and B4 (p. 92) say there is "one deviation … [that] changes no estimator or decision rule." The text also describes the Q3 slope as the object "whose sign is the object." The memo shows that the sign is not the right statistic.

*Requests:*
1. List every deviation in B4 with its timestamp: the high-M trimming, the dropped (256,4) duplicates and D8.
2. Report the plan's inference and the remedied inference side by side, as the memo proposes.
3. Restate Q3 as a comparison against the design-specific null.
4. Say before the results are reported what this design *can* distinguish. By the memo's numbers, model-free σ\* has a Monte Carlo s.d. of 0.013–0.021 at s = 0.005, before calibration. The design separates 0.70 from Porian's 0.51 easily, but not 0.70 from 0.65, which is the comparison Major 1 makes relevant.

**(c) Tuning at the high-M shapes, where Q3 is decided.**
- The LR corners are the two-layer width-128 model over the main ladder and width 640 at 25 and 50M tokens.
- The four-layer `hiM` shapes, which carry the high-M comparison, have no LR check.
- Weight decay is fixed at λ = 0.1. In MLX's AdamW it is multiplied by the learning rate, so the EMA timescale is 1/(η·λ) steps: about 1,750 steps at width 128 and 7,400 at width 640.
- Relative to run length, that timescale ranges from about 4.8 runs (width 640, 25M tokens, 1,526 steps) to about 0.009 runs (width 128, 3.2B tokens, 195K steps), a factor of more than 500 across the design.
- Wang and Aitchison (2024) and Bergsma et al. (2025, in the bib) find that the optimal timescale scales with the length of training. A fixed λ therefore makes tuning quality vary with D/N.
- The paper's own flexible-input formula (App. D) shows the consequence. Inefficiency that worsens with D lowers the measured ε_D at high M and raises the local wedge. That biases Q3 toward "parametric forms understate w," which is Farseer's direction and the paper's expected answer.
- The fixed 250-step warmup is 16 percent of the 25M-token runs and 0.13 percent of the 3.2B-token runs.

*Requests:*
1. At the (128,4) shape at 1.6B and 3.2B tokens, on both corpora, add LR × {0.5, 2} and one run with λ scaled to hold the timescale-to-run-length ratio at its main-grid median.
2. Report Q5's excess-loss statistic there.
3. State the MLX weight-decay coupling in B4.

**(d) The M–size confound is reduced but not broken.**
- Above M = 1,000 in non-embedding units, only width-128 shapes appear (0.39–0.79M non-embedding parameters; 57–73 percent of parameters in the embedding).
- The (256,4) shape reaches 1,017, on FineWeb-Edu only.
- In total-N units the design stops at 1,743, below the 20 clean models that lie beyond Farseer's range.

The experiment therefore adds a second recipe at the ratios Farseer already covers, not new range. Training the (256,4) shape on both corpora to 6.4–12.8B tokens would help. That is about 1–2.5×10¹⁷ FLOP per run, within the paper's hardware budget. Otherwise, say plainly that the experiment's reach in M is Farseer's.

**(e) "Our experiment … can separate schedule from scale" (p. 21).** Separating the two requires unannealed and annealed validation losses at the same (N, D). The trunks log only a single-batch training loss every 1,000 steps (`train_sweep.py`), which is too noisy for the purpose. Evaluate each trunk on the validation sets at every branch point for the runs still in the queue; this costs nothing. Otherwise, narrow the claim: annealed runs at small scale can show only whether Porian's 0.51 is a small-scale effect.

**(f) D in bytes.** The primary output is bits per byte, which is a constant rescaling within each validation set. It does not affect the tilt on a common validation set, but D is still in tokens. Converting D to bytes shifts ln B_edu − ln B_web by β·ln(3.89/3.74), about 0.014–0.016. That is comparable to the power memo's minimum detectable tilt at low noise (0.020). Report the tilt with D in bytes as well.

### Major 4. The abstract and conclusion still make level claims that the paper's own ranges do not support

1. **"Most of lifetime cost since 2024" is a level claim.** The abstract says open-weight models "released since 2024 were trained as if compactness … accounted for most of lifetime cost." On the clean sample, the 2024 aggregate share ranges from 0.05 to 0.74 across the 32 ex-ante technologies. The 2025 share ranges from 0.24 to 0.96 (Table 3, Panel C). The robust statement is the rise, which holds "under every ex-ante technology" (p. 35). The abstract should say that, with the level attributed to the reference technology.
2. **"Up from near zero before 2023."** The 2019–22 figure of 0.04 comes from truncating w < 1 to w⁺ = 1. It is dominated by OPT-175B and GLM-130B, which carry 92 percent of that period's compute (p. 35). Their w < 1 reflects allocation by the Kaplan law, a belief error in the paper's own framework (Prop. 2(iii); App. E). It is not evidence of a zero value of compactness. Two fixes are possible: compute the 2019–22 share under the Kaplan law as the believed technology, where I expect s ≈ 0 for the right reason, or start the headline trend in 2023 (0.35 → 0.82).
3. **The sign result.** The conclusion says the sign is identified for 86 percent of the clean sample "under weak assumptions" (p. 41). Page 33 says "with no assumption on curvature or functional form." The 86 percent depends on two things: the anchor set, which admits IsoFLOP anchors and DeepSeek's law, and the path-slope range [−0.16, 0.19]. It is 77 percent with own-lab anchors and 23 percent when every ex-ante technology's in-support M\* is admitted. Also, the union of the 32 technologies' intervals lies above one for only 18 percent (p. 30). Name the assumptions in the same sentence and give the 23 percent alongside.
4. **"Ex ante."** Page 28 says the clean sample "is defined ex ante, before any wedge was computed," and F3 (p. 159) says the technology set "was fixed before any wedge was computed." Version 1 computed and published wedges for all 173 verified models (median 3.19). The cleaning steps implement round-1 referee requests, including my own Major 5 list. The procedure is fine, but these sentences are not accurate. Say that the rules were taken from the round-1 reports and fixed before the version-2 wedges under the new reference were computed, if that is what happened.
5. **The trend is ordinal too.** At given compute, every technology ranks models by M. The trend in s therefore largely restates the rise in M across release years, weighted by a few flagships; Llama 3.1 405B carries 39–55 percent of 2024 compute. Say so where the trend is called "more robust than the level" (p. 3).

### Major 5. The "parametric extrapolation is conservative" conclusion rests on one recipe at the smallest model sizes

This is a clear improvement over round 1, but the evidence has three limitations:
- It is one design, Farseer, with one data recipe and bits-per-character output.
- The M ≥ 1,024 bin contains only 0.10–0.34B models.
- Hyperparameters follow the Step Law rules, whose inefficiency the paper finds to be data-biased (App. D). The size of the gap also depends on the bandwidth: ln(w_param/w_local) for κ free runs from −0.25 to −0.67 in Table F6.

The paper states the believed-technology caveat correctly (Remark A12). Two further checks are feasible with data in hand:

1. **Marin's ladders reach M = 3,706 on three corpora** (Table 1). IsoFLOP designs with 7–8 budgets give local variation in two directions, so the local wedge of Proposition 4(i) and the fit-on-M ≤ 100 test can be run there. Llama 3's profiles reach M = 643. Report whether ln w is convex in ln(M/M\*(C)) on these designs.
2. **Bound the tuning share of Farseer's convexity.** Apply the flexible-input bias formula of App. D6 to the Step Law inefficiency to bound how much of the convexity could come from D-dependent mis-tuning.

Until a second recipe agrees, describe the direction as "in one recipe" wherever it is summarized (p. 4, p. 32, p. 41).

### Major 6. Lab-own technologies: apply one rule for curvature and for valid budgets

The introduction reports the lab-own median (0.61), and it is the paper's main answer to the factor-bias critique. But the rule differs across labs:
- Meta and Marin combine the lab's path with the design's *model-free* curvature.
- DeepSeek uses the reference curvature.
- AI2 uses the OLMo ladder with κ free, σ\* = 0.544 (1/σ\* − 1 = 0.84). Section III calls such small-sweep κ-free curvatures heterogeneous and possibly E–κ confounded (p. 24).
- The result for AI2 is extreme. OLMo 2 1B has ŵ = 20.2, against 2.0 under the ladder with κ = 1 (p. 165), and AI2 contributes 9 of the 22 lab-own models.
- Separately, Meta's path uses all ten budgets, although the paper's own validity rule drops the two unbracketed ones. On the eight valid budgets, the 405B's wedge is 1.64, not 0.97, and the 8B's is 12.2. The claim that the 405B is "on Meta's path by construction" rests on the invalid budgets.

**Requests.**
1. Use one rule: the lab's path plus the common model-free σ\*, with lab-specific curvatures as sensitivity.
2. Report the lab-own median without AI2.
3. Either use the eight-budget Meta path, or justify ten budgets explicitly as "the law Meta planned with" and flag in Table 2 that it violates the Section III validity rule.

### Major 7. What the wedge leaves out on the training side, and the gap between loss and capability

Round-1 Major 2(c) and 5(f) are only partly addressed.

- **Post-training and mid-training compute.** X_T is the final pretraining run. For 2024–25 releases, mid-training and annealing on curated data, SFT, and especially RL post-training can be material. Some of this compute scales with N, and all of it is planned when N is chosen. RL rollouts enter X_S (p. 11). The gradient compute of post-training, however, is a size-dependent training cost. It acts like δ > 0 and raises the measured wedge without any serving demand. This matters most for the 2025 end of the trend. At a minimum, discuss the direction. Better, bound it with disclosed post-training budgets (e.g., DeepSeek, OLMo 2 and Tülu report them).
- **Loss is not capability.** The ordinality claim on p. 6 extends to "accuracy on a benchmark that is monotone in loss." Known departures bear directly on the rivals in Section IV.E:
  - Kumar et al. (2024) find that over-trained models degrade more under post-training quantization. For quantized local deployment, which is the premise of the tier story, the value of extra tokens in *deployed* quality is lower than in loss.
  - Knowledge capacity per parameter and post-training responsiveness are also not functions of pretraining loss alone.
  
  One paragraph in IV.E and a sentence in the limitations would suffice.

### Major 8. Literature already in the bibliography that the paper should engage

- **The forward problem.** Villalobos and Atkinson (2023, `villalobos2023trading`) and Erdil (2024, `erdil2024optimally`) are the closest precedents for the training–inference trade-off that Proposition 2 inverts. They belong in the related-literature paragraph on p. 4 with Sardana, Bian and Roberts. Erdil (2025, `erdil2025inference`) bears on p and on who pays.
- **The experiment's tuning.** Lourie et al. (2026, `lourie2026small`) on tuning in small-scale scaling experiments, and Bergsma et al. (2025, `bergsma2025power`) on weight decay and batch size (Major 3(c)).
- **Annealing.** Tissue et al. (2024, `tissue2024scaling`) on the learning-rate annealing law. This is the natural tool for the Porian "unannealed" argument (p. 21) and for DataDecide's checkpoints.
- **Quantization.** Kumar et al. (2024, `kumar2024scaling`); see Major 7.
- **Data filtering.** Goyal et al. (2024, `goyal2024scaling`) on data filtering whose value depends on compute. This bears on the DataDecide tilt and on the FineWeb-Edu versus FineWeb design.

---

## 5. Minor comments

1. **Abstract (p. 1).** "Inverting first-order conditions with this technology" suggests the inversion uses only the model-free σ\*. It also uses the Chinchilla zero point M\*(C). Say "with a reference technology."
2. **P. 3 and p. 21.** "No detectable heterogeneity across designs" should say that the test has low power below a between-design s.d. of 0.02 (as p. 21 does) and that the design means average over different compute windows (Major 1).
3. **P. 9 and p. 12, Figure 1 and the Llama 3 8B example.** Both still use Besiroglu's κ = 1 parameters (w ≈ 5.2, s ≈ 0.81) and total N. Use the reference technology (6.7, 0.85) or Meta's own (8.4, 0.88), and add a non-embedding marker to Figure 1 (round-1 Minor 26).
4. **P. 20, Table 1.** Give the number of valid budgets per design in the table (Llama 3: 8 of 10) and add a column for the parameter convention.
5. **P. 23.** "With σ\* = 0.70 … ten times the compute-optimal tokens per parameter needs 73 percent more compute." Add "in the homothetic case"; the α/β split moves this by up to 19 percent (Table D14).
6. **P. 26 and B4 (p. 92).** The pre-registration is described as committed "when part of the FineWeb-Edu grid had finished" and when raw endpoint losses "had appeared in run logs." That disclosure is good. Also state which of the plan's primary choices, such as the primary convention for Q3, were made after those logs existed.
7. **P. 26 and B4 (p. 91).** "Seed replicates in both corpora": the FineWeb-Edu seeds are not evaluated on WikiText (`run_grid.py`). Say so.
8. **P. 29, Figure 5.** The x-axis reads "planned serving share of lifetime cost." Change it to "expenditure share s = (w − 1)/w," consistent with p. 11 and p. 28.
9. **P. 31, Table 2.** Add the embedding share (round-1 Minor 29) and a common-D flag (Major 2). For DeepSeek LLM, the lab-own interval "reflects only the reference curvature" (p. 165); mark this in the table.
10. **P. 32–33.** "Applying Farseer's gaps … raises their median s … from 0.68 to 0.71 or 0.72." Say that this transfers gaps measured on 0.10–0.34B models to models of 1–15B.
11. **P. 33.** State that the Llama 3 anchor at 10²² is the path's fitted value, not an observed bracketed minimum; the 10²² budget fails the Section III validity rule.
12. **P. 34, cost side.** "It never depends on p" is correct for s. Add that ϕ < 1 is the realistic case for large models with grouped-query attention, whose decode cost is dominated by weight and KV-cache traffic, and that ϕ < 1 raises s (Table F10). The range [0.5, 1.25] is fine.
13. **P. 35.** Say that the Hugging Face derivative elasticity with respect to planned T/D (0.60, p = 0.086) is below one, so derivative counts rise less than proportionally with planned demand, whatever the mechanism.
14. **P. 40, Section V.D.** γ = 0.136 comes from Meta's profiles, whose loss units are not stated. γ is cardinal in reducible loss and needs E. With 3.2 decades of budgets, E and γ are weakly separated (the paper says so for Marin, p. 161). Report the E-profile range, or use γ from designs with known units only.
15. **App. A, Definition of the generalized family (p. 55).** Kaplan's joint law is presented as the κ-family member with κ = α_D ≈ 0.103. Qualify it as in round-1 Major 9(c): it was fitted to early-stopped runs on D unique tokens, in the data-constrained regime, with non-embedding N.
16. **App. F10, "What is not done" (p. 174).** Two additions:
    - A Muennighoff effective-data robustness for models with disclosed repetition, e.g., stablelm-3b-4e1t (four epochs) and TinyLlama. The effect is probably small, but it is easy.
    - A sensitivity dropping Qwen2.5 and Qwen3 member rows, whose D is a corpus size stated for every size. The "all but common-D families" row covers this implicitly; say so.
17. **App. B2 (p. 87).** "We have not rebuilt D from Hoffmann et al.'s architecture table." This is feasible and is the cleanest way to settle η for the headline (Major 1, request 4).
18. **App. D4 / round-1 Minor 14.** Estimating DataDecide recipe tilts from final checkpoints only, where the schedule is complete, is still the cheapest test of the "unfinished schedule affects all recipes alike" assumption.
19. **Table F3 (p. 162).** MiniCPM's M\*(10²¹) = 192 lies outside its own design (10–60), as the paper notes. The median of 0.17 it produces sets the bottom of the "0.17–0.85" range that the introduction cites (p. 4). Report the range with and without technologies whose zero point lies outside their own design.
20. **P. 41, Conclusion, "What economists should use."** Give γ with its compute window and units caveat (Minor 14), and give σ\* with its window and convention (Major 1).
21. **Replication.** Include the per-budget σ\* file and the sweep's `results.jsonl` schema, including which endpoints carry which evaluations, in the package documentation.

---

## 6. Recommendation and priorities

**Recommendation: major revision.** The revision answers most of what I asked for in round 1, and much of it very well. The model-free σ\* estimator, the expenditure-share reading, the clean sample with MoE and distillation handled, the sign result and the Farseer check make Sections I–IV credible to an ML audience. The remaining work is well defined and mostly feasible with data and code the authors already have.

The paper cannot be accepted while the experiment is a placeholder in the introduction, Section III.E, Section IV.D and the conclusion. Its lead number also needs to be reconciled with its own budget-level evidence.

**Priorities for the next version:**
1. **σ\* at scale** (Major 1). Report the budget-level estimates, a meta-regression on compute, estimator diagnostics and the FLOP-accounting rebuild. Restate the headline with its window and convention, and carry it into the wedge's magnitude bounds.
2. **Common-D consistency** (Major 2). Add the cap case, and fix Table 2 and the Meta narrative.
3. **The experiment** (Major 3). Add a neutral evaluation on a retrained subset, disclose all deviations and the remedied inference, and add LR and weight-decay checks at the (128,4) high-M corner. Finish and report Q1–Q6.
4. **Level language** (Major 4). Correct the abstract, the "ex ante" sentences, the sign-result wording and the 2019–22 benchmark.
5. **A second recipe for the extrapolation check** (Major 5), on Marin.
6. **Lab-own rule, post-training and capability, literature** (Majors 6–8).
