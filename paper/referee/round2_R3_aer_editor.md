# Round-2 Referee Report R3: Co-Editor and Generalist Perspective

**Manuscript:** "What Optimizing Labs Reveal: Scaling Laws as Production Functions," by Yigit Okar and Claude (revised version 2).
**Version reviewed:** compiled draft of 2026-09-24 (`paper/main.pdf`, 175 pages).
- Main text: pp. 1–42.
- References: pp. 42–53.
- Online Appendix: pp. 54–175 (A 54–81, B 82–96, C 97–104, D 105–133, E 134–155, F 156–175).

I also read the LaTeX sources, the revision plan, paper plan v2, integration log v2, the ra1–ra5 memos, the output tables, and the pre-analysis plan. Page numbers below refer to the PDF.
**Journal:** *American Economic Review*
**Round-1 recommendation:** Reject and resubmit (refocus).
**Round-2 recommendation:** **Major revision.**

**Scope note.** The authors' controlled experiment is still running. It appears as TBD-m9 placeholders on pp. 3, 26–27, 33, 40–41 and in Appendices B4, C and E. As instructed, I comment on its design, its pre-analysis plan and the way the paper will absorb its results, not on results.

**Disclosure.** This report was prepared by an AI system (Claude) acting as a simulated referee in the authors' internal review workflow. The AER's editorial policy asks referees to disclose AI use, so I say so here.

---

## 1. Summary of the Revision

The authors have done what I asked for in round 1: they rebuilt the paper around one idea. The new version argues that optimization cuts both ways for measurement:

- Cost-minimizing developers place runs on the expansion path, where the data carry almost no information about curvature. Information about the compute-optimal ratio *M\** is second order, and about σ\* fourth order, in allocation errors (Proposition 3).
- The same optimization makes choices informative about objectives. A generalized first-order condition turns over-training into a revealed value of compactness, reported as the expenditure share *s* = (*w* − 1)/*w* (Proposition 2).
- Designed IsoFLOP variation breaks the circle. A new model-free estimator reads σ\* off the ratio of IsoFLOP curvature to the frontier slope (Proposition 4). Beyond the designs, the *sign* of *w* − 1 remains identified without σ\* or a functional form (Proposition 5).

Empirically, the revision reports:

- **Curvature.** A model-free σ\* of 0.695 [0.673, 0.717] across six IsoFLOP/Farseer designs from three laboratories (Table 1), with the Chinchilla (κ = 1) form overstating σ\* where the data can tell.
- **Portability.** *a* and *M\** do not travel across designs.
- **Revealed value of compactness** on a clean sample of 77 open-weight base models:
  - median *s* = 0.75 under a κ-free reference; 0.61 under lab-own technologies;
  - range 0.17–0.85 across 32 ex-ante technologies;
  - sign identified for 86 percent of the sample;
  - a compute-weighted share rising from 0.04 (2019–2022) to 0.82 (2025).
- **Economic implications (new Section V).** Data-demand growth, the data wall, the serving share of AI compute, and growth models.

Former Section VI is now Online Appendix E. The experiment has a pre-analysis plan committed before estimation.

## 2. Overall Assessment

### 2.1 What has improved

This is a much better paper, and in parts a different one.

1. **One paper, one frame.** The introduction (pp. 1–5) follows the order I proposed: stakes, question, idea, results, what is robust, closest papers, roadmap. The title now names the idea. The main text fell from about 21,400 to about 14,800 words, and from 59 to 42 pages.
2. **A genuinely new estimator.** Proposition 4 is the most useful addition. It identifies σ\* without a functional form, using objects every scaling study already estimates. Because it is invariant to monotone transformations of output, it can be applied to digitized profiles in unknown units, such as Llama 3's. Economists who work with designed variation along isocosts will recognize the idea. Scaling-law practitioners can use it tomorrow.
3. **Honest measurement.** Several changes make the revealed-demand section honest:
   - the κ-free reference;
   - the 32-technology ex-ante band shown as ECDFs;
   - the lab-own technologies;
   - the statement that ŵ is a monotone transformation of *M* (R² = 0.9999, p. 28);
   - the partial-identification treatment of the zero point (Proposition 5; pp. 33);
   - the Farseer check on the direction of extrapolation error (Figure 6).
4. **The economics of the wedge.** Proposition 2 now covers:
   - an internalization share λ and a relative price *p*;
   - size-dependent FLOP prices;
   - data costs, distillation and binding caps;
   - common token budgets within a family.

   The conduct tests (open versus closed; serving footprint; tier bunching) are the right tests, even though they have little power.
5. **Section V answers "why should economists care?"** Its answer is candid: the least portable object (*a*) drives data-demand forecasts, and σ\* barely matters for the cost of a data wall.
6. **Positioning.** The related-literature section is now well done: Hao & Merrill; Kricheli et al., including the reconciliation on p. 14; Marschak–Andrews; Bond–Söderbom; Box–Lucas; the forward-problem papers.
7. **Pre-analysis plan.** The experiment now has one, and the design addresses most of my round-1 concerns.

### 2.2 What still stands between this paper and the AER

Four problems remain, in decreasing order of importance.

1. **Editorial blockers** (Section 3): the byline, and the unfinished experiment.
2. **The three headline claims still outrun the paper's own evidence in specific places.**
   - *σ\* ≈ 0.7 "for frontier-scale language models."* It is an average over budgets of 10^18.5–10^21 FLOP. Within the two designs that reach 10^21, σ\* falls with compute to about 0.60 at their largest budgets (new comment N1).
   - *The trend "from near zero before 2023."* This baseline has three problems (N2):
     - it rests on eight models, two of which carry 92 percent of the compute;
     - one of those two belongs to a family the paper excludes elsewhere as a research suite;
     - by the paper's own Proposition 2(iii), all eight reflect pre-Chinchilla beliefs, not a low value of compactness.
   - *"For developers that serve them, lower serving cost."* The paper's own tests do not support this reading. Its showcase example, Llama 3 8B, is one that Proposition 2(iv) says is not a revealed preference (N3, N4).
3. **The revealed-demand result is now mostly ordinal.** After the authors' commendable retreat to what is robust, the robust empirical content is:
   - σ\* (in a compute range);
   - the sign of over-training;
   - the rise in over-training since 2023.

   The second and third are visible in tokens per parameter. The case for a general-interest journal must therefore rest on the methodological contribution and on a sharper statement of what the economic objects deliver (N5, N6).
4. **Length.** The main text is focused, but the total manuscript grew from 133 to 175 pages. The Online Appendix doubled to 122 pages, and one of its appendices is a separate paper (N7).

None of this requires a new paper. All of it must be fixed before I could recommend acceptance.

---

## 3. Status of Round-1 Comments

### 3.1 Editorial matters

| # | Round-1 request | Status | What remains |
|---|---|---|---|
| E1 | Authorship: AI cannot be listed as an author | **Not resolved** | The byline still reads "Yigit Okar and Claude" (p. 1). The disclosure footnote is good. But the AER's editorial policy now states it explicitly: "AI software, such as chatbots or other large language models, may not be listed as an author." AI use in drafting must instead be described at submission. This is a hard constraint, not a matter of taste. The editor cannot send the paper out with this byline. Move Claude's role to the acknowledgment/disclosure footnote, where the current text already describes it well. |
| E2 | Remove journal running heads | **Resolved** | Heads now read "WORKING PAPER / SEPTEMBER 2026". |
| E3 | Remove placeholders | **Partially resolved** | "[LOCATION TBD]" is gone. TBD-m9 blocks remain in the introduction (p. 3), III.E (p. 26), Figure 4 (p. 27), IV.D (p. 33), the conclusion (pp. 40–41), and Appendices B4, C and E5. The paper cannot be refereed with its own experiment as a placeholder. See N8 for how to prepare the text so the results can be absorbed whatever their sign. |
| E4 | Pre-register the experiment's analysis | **Resolved, with caveats** | Commit bd5c0ad, 03:12 on 24 September, predates estimation. Caveats are in N8: the repository is private; raw losses of part of one corpus had been seen; the power calculation is uncommitted; there is no ex-ante map from outcomes to headline claims. |
| E5 | Share the Ho et al. critique with the original authors and say so | **Not resolved** | The critique moved to Appendix E (pp. 134–155). The paper still says nothing about contact with the authors. This remains standard practice and should be done before submission. |

### 3.2 Major comments

| # | Round-1 comment | Status | What remains |
|---|---|---|---|
| M1 | Cut to one paper; about 13,000 words; ≤4 formal results; 6–7 exhibits | **Partially resolved** | The main text is focused: about 14,800 words, 10 exhibits (one a placeholder), 6 formal results. Former Section VI became Online Appendix E rather than a companion paper, and the Online Appendix grew from 63 to 122 pages. See N7. |
| M2 | Reframe introduction, abstract and title | **Largely resolved** | The frame, order, title, closest-papers paragraph and number density (about 20 numbers in the results paragraphs) are all fine. Remaining: the abstract still headlines two things the paper does not stand behind: a level ("most of lifetime cost") and the "near zero before 2023" baseline (N2). Its serving parenthetical is not supported (N4). |
| M3 | Separate the mechanical from the identified; express the result as a cost share; validate levels; lab-own technologies; ECDFs | **Largely resolved** | Done: IV.A states the object; *s* is reported; lab-own technologies cover 22 models; Figure 5 shows ECDFs. Levels cannot be validated: OpenRouter per-model volumes are unavailable, and the disclosure comparison is "a loose check, not a validation" (p. 40). I accept that. It follows that levels do not belong in the abstract (N2). |
| M4 | Reference technology imposes the rejected κ = 1 | **Resolved** | The reference is now κ-free. The band includes the κ-free members. The full range 0.17–0.85 is reported in the introduction. |
| M5 | Economics of the wedge (who pays, λ, *p*, data cost, test-time compute, family *D*, distillation) | **Resolved in the theory, partially in the application** | Proposition 2 is now general and the conduct tests are the right ones. Two problems remain. The paper does not apply Proposition 2(iv) to its own exhibits (N3). The "planned serving expenditure" label still rests on a developer-level code that the tests do not support (N4). |
| M6 | σ: complementarity is not a finding; "near 0.7" is not common; heterogeneity test; counting; drop the capital–labor band | **Resolved** | The model-free estimator changes the picture: 0.66–0.71 on annealed IsoFLOP designs. The heterogeneity statistics are reported. Complementarity is correctly called implied (p. 8, p. 23). The counting is consistent, and the capital–labor band is gone. **New issue:** within-design drift with compute (N1). |
| M7 | Show the objects matter economically | **Resolved in form** | Section V is a real addition. N6 asks for its integration with Section IV, where the paper's own results bear on its own forecasts. |
| M8 | Positioning | **Resolved** | |
| M9 | Experiment design | **Largely resolved in design** | Both parameter conventions, with non-embedding primary for Q3 and actual FLOPs counted; learning-rate checks at the corners and on the second corpus; seeds in both corpora; WikiText-103 as a neutral set; four-layer high-*M* runs; wild cluster bootstrap by trunk plus a seed-calibrated parametric bootstrap; "two corpora" language; a statement that the experiment tests direction, not production-scale levels. Remaining: the fixed 250-step warmup is handled only by a drop-*D* = 25M robustness check. Other items are in N8. |
| M10 | Move Section VI to a companion paper | **Partially resolved** | It is out of the main text, abstract and introduction (one sentence, p. 4), but it lives on as a 22-page Online Appendix E (N7). |
| M11 | Fewer formal results; notation; scope of the identification theory | **Partially resolved** | The scope statement (p. 14) is good. Several notation clashes are fixed: 𝔼 versus *E*; τ_C; κ in figures; "verified sample". Remaining: six formal results, and new clashes (minor 8). |
| M12 | Exhibits | **Largely resolved** | Figure 2 has two panels, and the truth has been removed from the start set. The ECDFs replace the wedge curve. Table 1 combines designs with σ\*. Remaining: ten exhibits; roughly 5-pt fonts in Figure 3(b) and Figure 7; Figure 5's axis label (minor 13–14). |

### 3.3 Minor comments

Most of the 35 round-1 minor comments are resolved, or moot because material was cut. Those still open:

- **3/5.** Resolved.
- **6.** Llama 3 8B figures: still three *M* values and three wedges in the main text; see minor 6 below.
- **16.** No tokenizer or byte normalization of *D*. Appendix F says it was not done. It matters for levels, not for signs.
- **31.** Version dates for 2026 preprints: please confirm.
- **33.** Online Appendix split: not done; it grew (N7).
- **34.** Archived snapshots: the data statement mentions download scripts. Please also state that dated snapshots and checksums of the Hugging Face and Epoch data are in the package.

---

## 4. New Major Comments

### N1. σ\* falls with compute inside the two designs that reach 10^21 FLOP; "about 0.7 for frontier-scale models" is an average over smaller budgets.

**Evidence in the paper.**
- Section III.B (p. 23) reports that σ\* falls by 0.072 per decade in Chinchilla (s.e. 0.026) and by 0.053 in Llama 3 (s.e. 0.010). It calls this "evidence of non-homotheticity, although weak." For Llama 3 the *t*-statistic is about 5.
- Figure 3(b) (p. 22) shows the pattern plainly. In the authors' file `ra1_modelfree_isoflop_budgets.csv`, the model-free estimates at the largest budgets are:
  - Chinchilla: 0.617 at 10^21 and 0.579 at 3 × 10^21;
  - Llama 3: 0.585 at 6 × 10^20 and 0.609 at 10^21.
- The Marin designs, which are flat at about 0.70, stop at 3 × 10^20.

**Referee computation.** I regressed the 36 per-budget estimates (excluding Porian) on log10 compute with design fixed effects.

| Specification | Slope per decade (s.e.) | Fitted σ\* at 10^21 |
|---|---|---|
| Inverse-variance weights, all designs | −0.051 (0.009) | 0.60–0.64 |
| Unweighted | −0.029 (0.011) | 0.63–0.68 |
| Chinchilla and Llama 3 only | −0.058 (0.013) | 0.60–0.62 |
| Marin only | −0.001 (0.023) | 0.70–0.71 |

The homogeneity result (Q = 4.1, p = 0.54) compares design *averages* taken over different compute windows. It does not speak to this gradient.

**Why it matters.**
1. The abstract and conclusion (p. 41) recommend "about 0.7" for "frontier-scale language models." The evidence nearest frontier scale says about 0.6, and possibly lower if the drift continues.
2. σ\* sets the scale of the wedge. The paper's own calculation (p. 30) shows that lowering σ\* from 0.80 to 0.60 at the reference zero point raises the median *s* from 0.55 to 0.88. The revealed-demand levels are computed with σ\* = 0.70 at compute two to four decades above where 0.70 was measured.
3. A σ\* that drifts with compute is a failure of quasi-homotheticity. Every closed form used in the inversion (equations 5 and 8) assumes quasi-homotheticity. This is a second failure of the κ family, alongside the convexity of ln *w* in Farseer (p. 32).
4. The headline interval [0.67, 0.72] leaves out two sources of uncertainty the paper documents on p. 21:
   - FLOP accounting, about ±0.04; in Marin's configuration count the estimates fall to 0.64–0.66;
   - Farseer's bandwidth sensitivity: the mean is 0.679 at the cross-validated bandwidth.

**Required actions.**
- (a) Report σ\* at a common compute level where the IsoFLOP designs overlap (e.g., 10^20), and at each design's largest bracketed budget. Report the meta-regression above, with its caveats (digitization error in Llama 3's per-budget standard errors; Marin's shorter range), in the main text.
- (b) State the headline as conditional on compute: e.g., "about 0.7 at 10^18–10^21 FLOP, declining toward 0.6 at the largest budgets of the two largest designs." Change the conclusion's "for frontier-scale language models" accordingly.
- (c) Show the revealed-demand results under the top-budget σ\* (about 0.60) as well as 0.70. Make clear in IV.C that the level is bounded below by the reference, not bracketed by it.
- (d) Carry the FLOP-accounting and bandwidth uncertainty into the interval reported in the introduction, or state them beside it.
- (e) The controlled experiment runs at 10^15–10^17 FLOP and cannot settle this. It can, however, report whether σ\* drifts across its own six budgets. Label that analysis as exploratory (it is not in the plan).

### N2. The trend headline: the pre-2023 baseline is not the same object, and "most of lifetime cost" is a level.

The paper now says "the trend is more robust than the level" (p. 3). The abstract states that models released since 2024 were trained as if compactness "accounted for most of lifetime cost, up from near zero before 2023." Four problems remain.

1. **The baseline is eight models, and two of them decide it.** I rebuilt the aggregate's 2019–2022 universe with the authors' code (`ra3_econ/share.py`, `ra2_wedge/sample.py`). Its eight models are:
   - CPM-Ant;
   - NeMo Megatron GPT 20B;
   - GLM-130B;
   - BLOOM-1.7B;
   - OPT-175B;
   - EMDR;
   - AraGPT2-Mega;
   - Grover-Mega.

   OPT-175B alone has 46 percent of the compute, and with GLM-130B 92 percent (p. 35).
2. **The baseline violates the paper's own sample rule.** The clean sample drops "research suites and replications whose *D* was fixed by design, such as Pythia, OPT and BLOOM" (p. 28). Every OPT size up to 66B and every BLOOM size, including the 176B, carries `drop_step = b1_research_suite` in the verified sample. Yet OPT-175B, a replication of GPT-3's allocation, and BLOOM-1.7B enter the baseline, because the universe applies the exclusions only to verified-sample rows.
3. **The pre-Chinchilla period is a belief regime, not a low value of compactness.** By Proposition 2(iii), optimizing against another technology lowers the measured wedge. The paper itself reads Gopher's *w* = 0.36 as "plausibly the law of Kaplan et al. (2020)" (p. 11). Appendix F reports that closed models had *negative* median shares in 2021–2023 (p. 172).

   In the aggregate, *w*⁺ = max(*w*, 1) sets every such model to zero. "Near zero before 2023" is therefore mostly the Kaplan-to-Chinchilla belief correction, mechanically truncated. It is not revealed evidence that developers placed little value on compactness.
4. **"Most" is a level, and it is not robust for 2024.** On the clean sample, the 2024 share ranges from 0.05 to 0.74 across the 32 technologies, with a median across technologies of 0.51 (Table 3, panel C; `ra3_econ_inference_share.csv`). Also, "it rises under every technology we consider" (p. 3) is verified only for 2023 → 2025 on the clean sample (Appendix F), not for the 0.04 baseline quoted in the same sentence.

A smaller point: the 2025 aggregate is dense-only. The open-weight frontier moved to mixture-of-experts in 2025, and those models are excluded; the largest dense 2025 run is Apertus-70B at 6 × 10^24 FLOP. I added the four MoE models of Table 2, panel B, at their bounds with active-parameter compute. The 2025 share moves only to 0.78–0.82, so this is not decisive. But the text should say "dense," and the aggregate should include MoE with bounds.

**Required actions.**
- (a) Drop the pre-2023 baseline from the abstract and introduction. State the trend within the post-Chinchilla regime, where it is robust: 2023 → 2025, rising under all 32 technologies on the clean sample.
- (b) If a longer series is kept, apply the clean-sample exclusions to the universe, so that OPT-175B and BLOOM go. Report the 2019–2022 wedges under a Kaplan-belief technology, the "believed technology" case of Proposition 2(iii), and say what remains.
- (c) Replace "most of lifetime cost" in the abstract with an ordinal or conditional statement. For example: "the value developers placed on compact models rose steeply between 2023 and 2025 under every technology we consider; under our reference it exceeded training cost for the median model." That last clause is equivalent to *w* > 2, and it says what "most" means without the word "cost," which misdescribes non-serving developers.

### N3. Common token budgets: Proposition 2(iv) is not applied to the paper's own exhibits or showcase.

Proposition 2(iv) (p. 10) and its discussion (p. 11) state a clean result. When a family trains every size on one token budget, member wedges differ only through the technology, "member-level wedges are not revealed preferences," and the family reveals one weighted number. This is exactly the right response to my round-1 M5(5). The paper then does not use it where it matters.

- **Table 2 (p. 31).** Ten of the sixteen models in panel A belong to common-*D* families: Llama 3 8B/70B/405B, Llama 2 70B, Qwen2.5 7B/72B, Qwen3 0.6B/14B, DeepSeek LLM 7B/67B. That is 32 of the 77 clean models (`fam_class` in `ra2_wedge_models.csv`). The table does not flag them.
- **The showcase contradicts the proposition.** Section IV.C (p. 30) calls Meta "the sharpest case":
  - Llama 3 8B has *s* = 0.88, "a planned serving share";
  - "within one developer and one data pipeline, ... the flagship was trained as if for training efficiency and the smallest model as if for serving."

  But the Llama 3 herd shares a 15T-token budget. By Proposition 2(iv), the gap between 405B's *w* ≈ 1 and 8B's *w* ≈ 8 is fixed by ln *w*_i − ln *w*_j = −α(ln *N*_i − ln *N*_j). Appendix F (p. 172) reports what the herd reveals: a family share of **0.29**.
- **The aggregate survives.** I recomputed the median with one observation per allocation decision: the family share for the 11 common-*D* families and member shares for the other 45 models. The median is 0.74, against 0.75 (referee computation from `ra2_wedge_family_level_W.csv`). The paper's own exclusion check on p. 34 agrees. So this is about exhibits, narrative and conduct tests, not the headline median.

**Required actions.**
- (a) Rebuild Table 2 around decisions. For common-*D* families, report the family share, and show member wedges only as illustrations of the mechanical gradient.
- (b) Replace the Llama 3 narrative. The accurate statement is that the Llama 3 herd, sized at one 15T budget, reveals a share of about 0.3. A size-specific family, such as OLMo 2 or SmolLM2, is the right showcase for member-level revealed values.
- (c) Run the conduct tests (IV.E) at the decision level, so that common-*D* families enter once.

### N4. The "planned serving expenditure" reading is still not supported, and the abstract asserts it.

The abstract says compactness is, "for developers that serve them, lower serving cost." The evidence in Section IV.E and Appendix F points the other way, or nowhere.

- **The serving code is a developer-level label applied to every model** (Appendix F, p. 170). It codes Qwen3 0.6B, a model whose use is overwhelmingly local or on-device, as "planned serving." The same holds for small Gemma and Qwen2.5 models. Whether the *specific size* was offered on the developer's own API at release is observable and not coded.
- **Serving barely matters.** Serving developers have a median *s* of 0.80 against 0.74 for others, and the coefficient has *p* = 0.39 (p. 34).
- **On-device models are the most over-trained.** In the authors' file, the 9 clean models with an on-device target have a median *s* of **0.90**, against 0.74 for server or unspecified models. That fits memory tiers and adoption, not serving cost.
- **Open-weight models are more over-trained than closed ones** (+0.52, *p* = 0.011). The paper itself says this is inconsistent with serving cost alone.
- **Half the sample may sit under a binding cap.** Fifty-one percent of production-scale open models bunch at memory-tier windows. By Proposition 2(iii), where such a cap binds, *w* − 1 only bounds the value of compactness from above.

**Required actions.**
- (a) Code serving at the model level: whether this size was listed on the developer's own API or product at release, with archived price lists or model catalogs. Re-run the test at that level and at the decision level (N3).
- (b) Unless the model-level test supports it, remove the serving parenthetical from the abstract. Describe *s* throughout as the revealed value of compactness, with planned serving expenditure as one of its interpretations.
- (c) Rename the "clean inference-demand sample." It includes developers the paper itself says do not reveal inference demand.
- (d) Report the on-device contrast descriptively in the main text. With two developers it cannot be tested, but it is informative about which mechanism dominates.

### N5. State the contribution the paper can now defend.

After the revision, the paper's robust empirical findings are:
1. a model-free σ\* of about 0.7 on annealed IsoFLOP designs (but see N1);
2. the sign of over-training, identified for 86 percent of models;
3. the rise in over-training after 2023.

(2) and (3) are known qualitatively in the ML community, and they are visible in tokens per parameter (R² = 0.9999, p. 28). The cardinal economic content, how much developers value compactness, is explicitly conditional and unvalidated. That is the right scientific stance, but the paper has not yet said clearly why the AER should publish what remains.

I think the case can be made, and the paper should make it in its introduction and conclusion.

- **The methodological contribution is general.** It has three parts:
  - a clean, quantified version of Marschak–Andrews/Bond–Söderbom, with rates tied to economic wedges;
  - an estimator of the elasticity of substitution from designed variation along isocosts that needs no functional form;
  - a revealed-preference result, Proposition 5(iii), that signs a first-order-condition wedge without knowing the curvature.

  The paper should say where else these apply: any setting with a known multiplicative cost and designed isocost variation (agronomic factorial trials; hardware and chip-design sweeps; dose-combination trials). Proposition 5(iii) is of independent interest to the markup literature: the sign of a wedge needs only the location of the optimum.
- **The economic payoff should be stated in one place, including what the paper cannot do.** Section V finds that the best-measured object, σ\*, matters least for data-wall costs, while the least portable objects, *a* and *M\**, matter most for data-demand forecasts. That is a finding, not an embarrassment. It tells economists which scaling-law parameters their models should treat as uncertain, and what laboratories would need to publish (IsoFLOP minima at frontier compute) to resolve it. The introduction should lead with this rather than bury it in "These objects matter" (p. 3).
- **Say where σ\* does matter:** it sets the scale of the revealed value of compactness (p. 30), and it converts projected serving demand into data demand (p. 37). Those are the paper's reasons to care about σ\*. They should be stated as such in Section V's opening paragraph.

### N6. Section V should use Section IV's results.

The two halves meet in Section V but do not talk to each other.

1. **The data-demand forecasts ignore the rising wedge.** At given compute, *D*/*D\** = *w*^{σ\*/[2(1−σ\*)]} (p. 37). The paper's own aggregate *s* rises from 0.35 (2023) to 0.82 (2025), which means *w* rises from about 1.5 to about 5.6. At σ\* = 0.70 that raises *D*/*D\** about 4.5-fold in two years, roughly 2.1 times a year. That is comparable to the 2.0–2.8 times a year the paper attributes to compute growth.

   The aggregate concerns many models, and the frontier run is closer to the path. Still, the forecast in V.A holds each technology's frontier *D*/*D\** fixed at the level of the four disclosed dense frontier runs. A scenario in which frontier over-training follows the post-2023 trend belongs in Table 3. This is also where σ\* enters the forecast (the multiple is 4.7 at σ\* = 0.74 and 2.3 at 0.60 for *w* = 3), which answers the "why σ\*" question in N5.
2. **The data wall bites earlier for over-trained models, and it feeds back on measured wedges.**
   - Scarcity is defined at compute-optimal use, *r* = *D\**(*C*)/*U*. Over-trained models process *w*^{1.17} times *D\**, so the relevant scarcity is higher. The paper notes that frontier runs at observed data use already process 0.6–1.2 times the stock (p. 37). The implied cost of the wall at *observed* allocations should be reported.
   - By Proposition 2(iii), a binding token cap adds μ to the denominator of *w*. As the wall approaches, measured wedges will *understate* the value of compactness. That is a prediction for the trend worth stating.
3. **The fleet comparison.** Its conclusion is a loose check (p. 40). Keep it short, and state in the text the service-life assumption (one year) that drives the 0.18–0.34 fleet shares.

### N7. Length: the main text is now right; the package is not.

- The main text is about 14,800 words by the authors' count and 42 pages. It has 10 exhibits (one a placeholder) and 6 formal results. The experiment's results will add perhaps 600–900 words. That is acceptable for the AER if the exhibits shrink:
  - merge Figure 4 into Figure 3 once the experiment is done, as the integration log proposes;
  - reduce Figure 7 to one or two panels, or move it to the appendix, since Table 3 carries the numbers.
- **Formal results.** Proposition 1 is standard Shephard–Uzawa duality, and its content is needed only for the definitions of *a*, γ and *M\**. Fold it into the text, keeping part (iii) as a sentence with the proof in Appendix A. Proposition 3(ii)'s rate statement is heuristic by the authors' own account (p. 13, "should then not be expected to shrink faster"; "rests on such simulations and a heuristic argument, not on a limit theorem"). A heuristic claim should not sit inside a proposition; move it to a remark. That leaves four formal results in the main text: Lemma 1 and Propositions 2–4, with Proposition 5 possibly merged into 4 as "beyond the design."
- **The Online Appendix is 122 pages, up from 63.**
  - Appendix E (22 pages) is a separate paper with its own findings: a design-matched benchmark, TFP dispersion, an industry Monte Carlo, the Ho et al. replication, and the allocative decomposition. As an appendix it will not be refereed as a paper, and it contains a replication critique (E5) that deserves proper refereeing. Spin it off, as I asked in round 1.
  - Appendix A (28 pages) and Appendix D (29 pages) can be split between material referees need and a supplementary archive in the replication package. Examples of the latter: the claims register, the 61-check verification tallies, and the practitioner overhead table.
  - A target of about 60 pages is reasonable.

### N8. The experiment: make the pre-analysis plan credible and the paper outcome-proof.

The plan (`paper/notes/m9_preanalysis_plan.md`) is good: a primary output, both conventions, estimators, two inference schemes, six questions with decision rules, and a commitment to report all answers. Five points remain.

- (a) **Verifiability.** The plan sits in a private git repository, and a commit time there can be rewritten. Report the commit hash in the paper; the text on p. 26 gives only the date. Deposit the plan now with a public timestamp (OSF, or a public tagged release with the same hash). Readers can then check it was unchanged, even though the deposit postdates the commit.
- (b) **What had been seen.** The plan was written after raw endpoint losses for part of the FineWeb-Edu grid had appeared in run logs. The appendix says so (Appendix B4). The main text (p. 26) should say it too, in one clause.
- (c) **The power calculation.** The plan requires it "BEFORE touching the results." The m9 memo says it was computed from 15:52 to 16:44, before `run.py` was first run on the sweep results. It is not committed. Commit the power output, and the memo section that records the order of work, before any further estimation, and cite the hash.
- (d) **Outcome-proof text.** In round 1 I asked the plan to say how results would be reported "whether they support or contradict Sections IV–V." The plan commits to reporting every answer, but not to how the headline changes. The integration log notes that "if m9 contradicts 'about 0.7', the abstract, introduction and conclusion must change together." Before the FineWeb results are in, write down in a dated amendment what the abstract and introduction will say under each outcome of Q1 (σ\* by corpus) and Q3 (the sign of the extrapolation slope), and state that the amendment postdates the preliminary FineWeb-Edu estimates in the m9 memo.
- (e) **Neutral-set coverage.** The plan evaluates *new* runs on WikiText-103 ("for runs that have it"). The main text states the quality rule as if WikiText were available for every endpoint (p. 26). Say which endpoints have WikiText losses.

---

## 5. Minor Comments

1. **Abstract, first sentence.** "choose model size and training data to minimize training and serving costs" states a model the paper tests and does not support (N4). Something like "trade off model quality, training cost and the value of a compact model" is accurate.
2. **Abstract.** "Designed experiments give ... about 0.7": after N1, add the compute range. "Inverting first-order conditions with this technology" is misleading, because the *level* result depends on *M\**(*C*), not on σ\* alone. Say "with a technology estimated from designed experiments."
3. **p. 2.** "For a developer that serves its own models, *s* is the planned serving share." By Proposition 2(ii), even serving developers internalize only λ of serving cost when others also serve their open weights. For open-weight releases by serving developers, *s* is λ-weighted. Say so.
4. **p. 3.** "rose from 0.04 in 2019–2022 to 0.82 in 2025, and it rises under every technology we consider." The second clause refers to 2023–2025 on the clean sample (N2).
5. **pp. 4 and 33, sign identification.**
   - The 86 percent share relies on admitting only IsoFLOP designs as anchors. Add in the introduction that it falls to 23 percent if every sweep's *M\** is admitted, and give the reason IsoFLOP anchors are the right ones (model-free *M\** at the largest budget).
   - Update the identified set for *M\**(10^24): [2.3, 89] uses ra2's narrow anchors. With wild bootstrap-*t* anchors it is roughly [1.5, 110] (integration log §4, open issue 3). The 86 percent is unaffected.
   - Reconcile in one sentence with p. 30, where "the union of the 32 technologies' 95 percent intervals lies above one for only 18 percent."
6. **Llama 3 8B numbers.** The main text has:
   - *M* ≈ 1,875 (p. 12, nominal 8B) versus 1,868 (p. 18, Table 2);
   - *w* ≈ 5.2 (Besiroglu), 6.73 (reference) and 8.4 (Meta's own).

   Label each by its technology at first use, and use exact *N* = 8.03B except in Figure 1. After N3 most of these should go.
7. **p. 13, Proposition 3(ii).** See N7: move the rate statement to a remark or conjecture.
8. **Notation.** *v* is both the data-loss term in eq. (2) and the allocation-error scale in Proposition 3(iii) and Figure 2. *c* is log compute, while *c*_T and *c*_D are prices (Proposition 2). *e* is the path slope in Proposition 5 and also the exponential in eq. (1). *b* ≡ 1 − *a* (p. 12) sits next to the loss coefficient *B*. Rename, e.g., ς_v for the allocation-error scale and *q*_T, *q*_D for prices.
9. **p. 21.** "Three laboratories" gives six estimates, but three are Marin's, with one recipe, one size ladder and one output. Say that the effective number of independent labs is three, next to the one-estimate-per-study mean of 0.693.
10. **p. 23.** "evidence of non-homotheticity, although weak." With a Llama 3 *t* of about 5, rephrase (N1).
11. **p. 28.** "The clean inference-demand sample is defined ex ante, before any wedge was computed." Wedges for all 173 verified models were computed and reported in version 1 (under κ = 1). Say "defined before wedges were recomputed under the revised technologies."
12. **p. 29, Figure 5.** The x-axis reads "planned serving share of lifetime cost." That contradicts the labelling rule of p. 11 and IV.A. Use "expenditure share *s* = (*w* − 1)/*w*."
13. **Figure legibility.**
    - Figure 3(b) and Figure 7 have fonts of about 5 pt.
    - In Figure 7, the y-axis labels of panels (b) and (c) overlap the neighboring panels.
    - Page 39 is half blank.
    - Figure 1 lacks the non-embedding marker R2 asked for.
14. **p. 31, Table 2.** Add a column flagging common-*D* families, with the family share (N3). Make "Serves" model-level (N4). Brackets are "95 percent design-conditional wild-bootstrap interval of one technology, not the uncertainty of the wedge." Consider dropping them, since the Band column carries the relevant uncertainty.
15. **p. 34.** Open versus closed: add in the introduction (p. 4) the caveat that only eight closed allocations have been disclosed since 2023 (six in 2024, two in 2025).
16. **p. 35.** Hugging Face derivatives: Appendix F reports the elasticity with respect to planned *T*/*D* (0.60, *p* = 0.086). That is the only check that touches levels, so it belongs in the main text beside the 0.80 elasticity with respect to *M*.
17. **p. 36, footnote 1.** Compute growth of 5.2-fold for January 2018 to May 2024, against Epoch's published 4.2-fold for the same window, is attributed to "probably a vintage effect." Five-year data demand is very sensitive to this: 4.2^{5(1−a)} versus 5.06^{5(1−a)}. Report the V.A growth rates at 4.2 as well, or reconcile.
18. **p. 37.** The shadow value of about $3 per million tokens would mean more beside an observed price: disclosed data-licensing deals, or the cost of synthetic-token generation. Optional.
19. **p. 40.** State the one-year service life behind the fleet conversion in the text, and give the share at two and three years.
20. **p. 41, "What economists should use."** Condition the σ\* recommendation on compute (N1). The γ recommendation ("the design ... closest to the application") still gives no guidance on how to choose. The large designs give 0.14–0.18, so say which to use for growth calibrations at 10^25–10^27 FLOP, and why.
21. **p. 41, Extensions.** The Sutton remark is now a clause. Fine.
22. **Title and terminology.** The title says "Labs." The text uses "developers" for released models and "laboratories" for designs. Either define "labs" in the first paragraph to cover both, or align the title.
23. **p. 26.** Cite the commit hash, not only the date (N8).
24. **Online Appendix F, p. 170.** Two serving dates are "judgment calls" (phi-1.5; 01.AI). After N4(a), these will be replaced by model-level evidence.
25. **References.** Confirm version dates for the 2026 preprints (round-1 minor 31).
26. **\thanks footnote.** Keep the disclosure text, which is clear and accurate. The AER additionally asks for a description of AI use at submission (E1).

---

## 6. Recommendation

**Major revision.**

The paper now has a clear idea and a genuine methodological contribution, the model-free estimator and the sign result. It has an honest measurement strategy and an economics section. It is a paper I can imagine in the AER.

Before I could recommend acceptance, the following must be done.

1. **Editorial:**
   - comply with the AEA's authorship policy (E1);
   - complete the experiment and integrate it under an outcome-proof pre-analysis plan with a public timestamp (E3, N8);
   - share the Ho et al. critique with its authors, or spin it off with Appendix E (E5, N7).
2. **Headline claims:**
   - condition σ\* on compute and propagate the documented uncertainty (N1);
   - restate the trend within the post-Chinchilla regime, without the OPT-dominated baseline or a level claim in the abstract (N2);
   - apply Proposition 2(iv) to the exhibits and replace the Llama 3 showcase (N3);
   - drop or substantiate the serving interpretation with model-level coding (N4).
3. **Contribution and integration:**
   - state the general methodological contribution and the economic payoff, including what cannot be identified (N5);
   - connect Section V to the paper's own revealed-demand results (N6).
4. **Length:** four formal results, eight or fewer main exhibits, and an Online Appendix of about 60 pages, with Appendix E as a companion paper (N7).

---

## 7. Summary for the Editor

The authors took the round-1 reports seriously and rebuilt the paper around the frame I proposed: optimization removes the information in outcomes and creates it in choices. The new model-free estimator of the elasticity of substitution (Proposition 4) and the sign-identification result (Proposition 5) are real contributions. The revealed-demand analysis is now careful and honest about what is ordinal and what is conditional. The main text is focused, and the new economics section makes the case that these objects matter.

What remains is fixable but not cosmetic.

- **Editorial.** The byline violates the AER's explicit policy on AI authorship, and the authors' own experiment is still a placeholder.
- **Claims.** Three headline claims go beyond the paper's evidence:
  - "σ\* ≈ 0.7" is a budget average, and the largest budgets say about 0.6;
  - "near zero before 2023" is an OPT-dominated, pre-Chinchilla baseline that the paper's own rules would exclude;
  - "planned serving cost" is contradicted by the paper's own conduct tests and by Proposition 2(iv) applied to its showcase example.
- **Size.** The package has grown to 175 pages, and a separate paper sits in Appendix E.

I would be glad to see the next version.

---

### Referee computations (reproducible from the authors' files)

- **Per-budget σ\* on compute.**
  - File: `output/tables/ra1_modelfree_isoflop_budgets.csv`. Designs: Chinchilla, Llama 3 and Marin ×3 (36 budgets; Porian excluded, as in the paper's summary).
  - Regression: σ̂\*_b on log10 *C*_b − 21 with design fixed effects.
  - Inverse-variance weights (weights 1/se²; s.e. scaled by the residual variance when above one): slope −0.0507 (0.0094); design intercepts at 10^21 of 0.601 (Llama 3) to 0.639 (Marin, Nemotron-CC).
  - Unweighted: −0.029 (0.011).
  - Chinchilla and Llama 3 only (weighted): −0.058 (0.013), with intercepts 0.620 and 0.595.
  - Marin only: −0.001 (0.023).
  - Largest-budget estimates: Chinchilla 0.617 (10^21) and 0.579 (3 × 10^21); Llama 3 0.585 (6 × 10^20) and 0.609 (10^21).
- **The 2019–2022 baseline.**
  - Universe rebuilt with `ra2_wedge/sample.py` (`build`, `universe`) and the filter of `ra3_econ/share.py::ra2_universe` (confident, non-MoE, transformer, open weights, clean exclusions on verified rows). Audit output was redirected to the session scratchpad.
  - The 2019–2022 rows: CPM-Ant, NeMo Megatron GPT 20B, GLM-130B, BLOOM-1.7B, OPT-175B, EMDR, AraGPT2-Mega, Grover-Mega.
  - `ra3_econ_inference_share.csv` gives OPT-175B's compute share (0.462) and the leave-OPT-out share (0.071).
  - In the verified sample, opt-125m through opt-66b and bloom-560m through bloom (176B) have `drop_step = b1_research_suite`.
- **2024 across technologies.** From `ra3_econ_inference_share.csv` (clean sample): 2024 range [0.054, 0.739], with a median across technologies of 0.512; 2025 range [0.240, 0.956], median 0.768.
- **MoE in the 2025 aggregate.**
  - Starting point: the 2025 dense aggregate (*m* = 4.528, ΣC = 2.86 × 10^25).
  - Added: DeepSeek-V3, Kimi K2, Qwen3-235B-A22B and Qwen3-30B-A3B, with compute 6·*N*_active·*D* and *w* at the Table 2 panel B bounds (truncated at 1).
  - Result: *s* = 0.775 (lower bounds) and 0.818 (upper bounds).
- **Common-*D* families.**
  - `fam_class` in `ra2_wedge_models.csv` (clean rows): 32 common-*D*, 30 size-specific, 15 singletons.
  - In Table 2 panel A, 10 of 16 models are common-*D*.
  - Family shares from `ra2_wedge_family_level_W.csv`: median across the 11 families 0.63; Llama 3 herd 0.294.
  - Median with one observation per decision (11 family shares plus 45 member shares): 0.736.
- **Serving and deployment.** From `ra2_wedge_models.csv` (clean rows), median `s_ref`:
  - by `serve`: 0.798 (33 models) versus 0.737 (44 models);
  - by `deploy`: on-device 0.895 (9 models), local 0.774 (3), server or unspecified 0.736 (65).
- **AEA policy.** AER Editorial Policy (aeaweb.org/journals/aer/editorial-policy), accessed 24 September 2026. It states: "Artificial intelligence (AI) software, such as chatbots or other large language models, may not be listed as an author."
