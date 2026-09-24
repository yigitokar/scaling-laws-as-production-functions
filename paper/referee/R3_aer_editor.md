# Referee Report R3: Co-Editor and Generalist Perspective

**Manuscript:** "Scaling Laws as Production Functions," by Yigit Okar and Claude.
**Version reviewed:** compiled draft of 2026-09-24 (`paper/main.pdf`, 133 pages). The main text runs from p. 1 to p. 59, the references from p. 60 to p. 70, and the Online Appendix from p. 71 to p. 133. I also read the LaTeX sources (`paper/sections/*.tex`, `paper/tables/*.tex`), the paper plan, the module memos and their reviews, the literature synthesis, and the sweep code.
**Journal:** *American Economic Review*
**Recommendation:** **Reject and resubmit.** I would welcome a substantially refocused resubmission. I would not support publishing the current manuscript even after local repairs. It contains three to four papers, its headline numbers are not robust by its own account, and it does not yet show why economists should care about the objects it estimates.

**Scope note.** The authors' controlled experiment (Sections III.B and IV.E; red "TBD-m9" placeholders; Figure 5) is still running. As instructed, I comment on its design (Major Comment 9) and not on the missing results.

---

## 1. Summary

The paper argues that a neural scaling law, relating a language model's loss to its parameters *N*, training tokens *D* and compute *C* ≈ 6*ND*, is a production function, and that fitting one is production-function estimation. It builds a dictionary between scaling-law practice and empirical IO (Table 1). It then gives:

- a duality result: Chinchilla's Approaches 1–3 estimate a cost function, conditional factor demands and a primal technology (Proposition 1);
- a lemma: with multiplicative cost 6*ND*, any interior compute optimum requires 0 < σ < 1 (Lemma 1);
- a revealed-demand result: under lifetime-compute minimization the ratio of output elasticities is *w* = 1 + *T*/(3*D*), so over-training reveals anticipated inference *T* (Proposition 2);
- identification results: on-path data identify the expansion path and the frontier elasticity but not curvature or *M\**. Information about *M\** is second order, and about curvature fourth order, in log-wedge deviations: "the better labs optimize, the less their data reveal" (Propositions 3–4). Further results cover biased technical change (Prop. 5), transmission, selection and proxies (Prop. 6), and partial identification of *T* (Prop. 7).

Empirically, the paper:

- re-estimates the Chinchilla data (estimator comparison, duality tests, rejection of κ = 1, and flat on-path profile likelihoods);
- estimates the technology in seven public sweeps. σ\* is 0.73–0.83 under the Chinchilla form and 0.51–0.71 with κ free, with Farseer at about 0.70;
- tests whether data quality is neutral, and decomposes the Kaplan–Chinchilla allocation gap into input mismeasurement and untuned flexible inputs;
- applies the inversion to 173 open-weight models. The median ŵ is 3.19, the share of models under-trained falls from 47 percent (2021) to 0 (2024), and over-trained models are downloaded more at fixed compute;
- adds a section on observational production functions: a LaLonde-style design-matched benchmark, TFP dispersion, a replication showing that the Ho et al. (2024) point estimate is an optimizer artifact on a Diamond–McFadden–Rodriguez ridge, and an allocative-versus-technical decomposition of the Kaplan→Chinchilla rebalancing.

## 2. Overall Assessment

### 2.1 Strengths

The work is careful, candid and unusually well documented. Numbers are traced to code, caveats are stated, and the authors repeatedly retreat from claims their evidence does not support. Three ideas are genuinely interesting to economists:

1. **The rate result** (Prop. 4). Optimizing behavior removes the variation that identifies curvature, and the loss of information can be quantified: information about *M\** is O(v²) and about σ\* is O(v⁴) in the dispersion of log wedges. The Monte Carlo (Figure 2) and the flat on-path profile likelihood in the Chinchilla data (Table 3, Panel D) are compelling illustrations.
2. **Lemma 1.** Because isocosts are straight lines in logs, an interior compute optimum *requires* σ < 1. This is elegant, easy to explain, and has a clear economic intuition (the implicit price of each input rises with the quantity of the other).
3. **The inversion** (Prop. 2). Reading a DLW-style wedge off labs' allocation choices to recover planned deployment, without usage data, is a new idea. As far as I can tell, and consistent with the authors' own search (`lit/notes/novelty.md`, C7), nobody has done it.

The setting is also attractive in principle. The cost function is an engineering identity, prices drop out, and designed input variation (IsoFLOP sweeps) exists alongside observational data.

### 2.2 The central problems

1. **The manuscript is three to four papers** (Major Comment 1). By my count, the main text has about 21,400 words (introduction 2,410; I 2,570; II 3,690; III 1,180; IV 3,790; V 3,300; VI 2,750; VII 1,000). It fills 59 pages in this format and carries 9 figures (one a placeholder), 9 tables, 7 propositions and a lemma, plus a 63-page Online Appendix. The results paragraphs of the introduction (pp. 2–4) contain roughly 110 numbers. No reader can tell what the paper is *about*.
2. **The introduction, abstract and title sell the wrong things in the wrong order** (Major Comment 2):
   - they lead with the mapping, which is the least novel element;
   - they then present the identification theory in heavy notation;
   - they headline two numbers the paper itself calls non-robust: "near 0.7", and "about 2.2 times".
3. **The economic significance of σ, *M\** and *w* is asserted, not demonstrated** (Major Comment 7). Two paragraphs of the conclusion (p. 59) are all a general reader gets.
4. **The headline result has two layers, one mechanical and one fragile** (Major Comments 3–5):
   - Its robust content is nearly mechanical: ln ŵ is a linear function of ln(*D/N*) with R² = 0.998 in the authors' own file.
   - Its cardinal content depends on the two objects Section II shows are weakly identified. It is computed under a technology (κ = 1) that Section IV rejects.
   - It rests on an economic model (lifetime-compute minimization by the party that pays for inference) that does not describe open-weight developers, who do not pay for their users' inference.
5. **"σ ≈ 0.7" is overstated as a common parameter** (Major Comment 6), and gross complementarity is presented as a finding although it is implied by Lemma 1 and imposed by the functional forms.
6. **The controlled experiment, as designed, cannot resolve the extrapolation problem that motivates it,** and it has flexible-input asymmetries that could masquerade as factor bias (Major Comment 9).

### 2.3 Direct answers to the questions put to me

- **Is the contribution clear?** No. The closing statement of contribution (p. 5) lists four items: the "systematic mapping", identification results, estimates with inference, and the inversion. Of these, the mapping is a device rather than a contribution. The paper never says which of the other three is *the* contribution.
- **Is it important?** Potentially. The inversion and the identification rates could interest a broad audience. As written, the importance rests on claims (σ matters for growth models; *w* measures planned deployment) that are never quantified.
- **Is it well positioned against the recent literature?** Partly (Major Comment 8):
  - Hao & Merrill (2026) are credited for the σ formula but not engaged on the demand side.
  - The contrast with Kricheli et al. (2026), "behavioral versus design collinearity", is less consequential than claimed, because current observational models lie far off the path.
  - Whitfill (2025), Mertens et al. (2026) and Ho et al. (2024) are engaged mainly in Section VI, which I recommend moving out.
  - Important antecedents are missing. In IO: Bond & Söderbom 2005. In econometrics: Sargan 1983 and Rotnitzky et al. 2000. In the economics of data: Jones & Tonetti 2020. On the forward inference problem: Bian et al. 2025 and Roberts et al. 2026.
- **Too long or unfocused?** Yes, decisively. Section 6 below gives a concrete plan to reach about 12,500–13,500 words and 7 exhibits.
- **Does the introduction sell the right result?** No (Major Comment 2). I propose a unifying frame: *optimization destroys the information in outcomes but creates information in choices.* It puts the identification theory and the inversion in one story, with the technology estimates as the bridge.

---

## 3. Editorial Matters (to resolve before any resubmission)

**E1. Authorship.**
- The byline lists "Claude," an AI system, as a co-author (title page footnote, p. 1).
- COPE's position statement on authorship and AI tools (2023) holds that AI tools cannot meet authorship criteria, because they cannot take responsibility for the work. Most economics journals follow it.
- The authors should check the AEA's current policy. In all likelihood they will need to list only human authors and describe the AI system's role in the acknowledgment or disclosure footnote.
- This must be settled before submission. It is not a matter for referees to adjudicate.

**E2. Submission format.** The manuscript is typeset as a published AER article: running heads read "THE AMERICAN ECONOMIC REVIEW MONTH 2026" and "VOL. VOL NO. ISSUE", with `\pubMonth{Month}` and `\pubVolume{Vol}`. Submit in the AEA submission layout without journal running heads or volume placeholders.

**E3. Placeholders.**
- The replication statement reads "[LOCATION TBD]" (p. 1).
- Five red TBD-m9 blocks appear in the introduction (p. 4), Section III.B (p. 27), Section IV.E (p. 40, including the whole of Figure 5), Section VI (p. 58) and the conclusion (pp. 58–59).
- The experiment is presented as one of the paper's pillars (Table 2, Panel B). An editor cannot send out a paper whose own experiment is a placeholder.

**E4. Pre-register the experiment's analysis now** (details in Major Comment 9(j)). Results are not yet in, and the analysis offers many forks: two outputs, two parameter conventions, nats versus bits per byte, Huber versus NLS, κ free or fixed, and several clustering choices. A dated analysis plan deposited before the results are seen would greatly increase the credibility of whatever the experiment shows.

**E5. Replication critiques.** Section VI.B (pp. 55–57) documents that the published Ho et al. (2024) point estimate is not the minimizer of their objective. Standard practice for a critique of this kind is to share it with the original authors and to say so in the paper.

---

## 4. Major Comments

### M1. The manuscript is three papers. Cut it to one.

The current structure pursues four agendas, each with its own data, estimators and literature:

- (a) an identification theory with a Monte Carlo (Section II, pp. 14–23);
- (b) a technology-estimation paper, which itself contains a sub-paper on input measurement and flexible inputs (Section IV.D and Table 5, pp. 36–39) aimed at ML practitioners;
- (c) a revealed-demand paper (Section V, pp. 40–50);
- (d) a paper on observational scaling and algorithmic progress (Section VI, pp. 50–58), which in turn bundles a LaLonde benchmark, TFP dispersion, an industry Monte Carlo, a forensic replication of Ho et al., a DMR ridge analysis, a Sahal-bias correction and an allocative-efficiency decomposition.

The conclusion adds five recommendations and two "numerical points" for ML practitioners (p. 59). None of these pieces is bad, but together they bury the paper's best ideas.

**Required action.** Pick one paper for the AER and move the rest to (i) a companion paper, (ii) the Online Appendix, or (iii) the replication package. My recommended cut:

| Current material | Pages | Action |
|---|---|---|
| Table 1 (18-row dictionary) | p. 9 | Move to the Online Appendix; keep at most 6 rows in the text, or a single sentence per analogy where it is used. |
| Prop. 1 (duality) with Approaches 1–3 | pp. 8–10 | Keep, shortened to about half a page. The "Approaches as estimators" reading can be two sentences. |
| Farrell/Harberger discussion | pp. 13–14 | Keep one sentence (the CE formula) and move the rest to the appendix. |
| Props. 3–4 | pp. 15–18 | Merge into one proposition. Keep. |
| Prop. 5 (DMR), Prop. 6 (transmission/selection/proxies) | pp. 20–22 | Move to the Online Appendix, or to the companion paper with Section VI. They are used mainly there. |
| Prop. 7 (partial identification) | pp. 22–23 | Keep as a remark inside Section V. |
| Figure 3 (Chinchilla design) | p. 29 | Move to the appendix; it duplicates Figure 1(a). |
| Table 3, Panels C–D | p. 30 | Move to the appendix. Replace Panel D with a small two-line figure: the profile likelihood on-path versus the full design. |
| Section IV.D and Table 5 (Porian decomposition, Step Law SFA, LR demand) | pp. 36–39 | Move to the Online Appendix or an ML-venue companion. Keep one paragraph saying that *a* and *M\** move by 0.04–0.17 with counting conventions and tuning. |
| Section V.C (stated intent), V.E over-identification test, V.H aggregate | pp. 45–50 | Condense to one paragraph each, or move to the appendix. |
| Section VI in full, with Tables 8–9 and Figures 8–9 | pp. 50–58 | Move to a companion paper, e.g., "What Cross-Lab Data Can Identify about Algorithmic Progress". It has its own audience (Whitfill; Mertens et al.; Ho et al.; Gundlach et al.) and its own findings. |
| Conclusion: "Recommendations for scaling studies" | p. 59 | Cut to two sentences. Put the list in the appendix or the companion paper. |

**Target.** About 12,500–13,500 words of main text; no more than 4 formal results; 6–7 exhibits; an introduction of no more than 1,800 words. Section 6 of this report gives the outline.

### M2. The introduction, abstract and title sell the wrong result, in the wrong order.

**Diagnosis.**

1. *Title and first two paragraphs (p. 1).* "Scaling Laws as Production Functions" announces the mapping. The authors' own novelty audit correctly says the mapping is not new as an idea (`lit/notes/novelty.md`: Hao & Merrill 2026, Epoch, a 2022 LessWrong post), and the introduction concedes this on p. 5 ("We do not claim to be the first..."). A title that names the least novel element invites a desk reject.
2. *Order.* The results arrive in this sequence:
   - identification (pp. 2–3), dense with ε_N = ε_D, γ = αβ/(α+β), "fourth order" and DMR;
   - technology (p. 3);
   - revealed demand (pp. 3–4);
   - a fourth block on observational scaling (p. 4).

   The only result a general economist will remember, that over-training reveals planned deployment, starts about 1,100 words in.
3. *Count.* Line 23 of `introduction.tex` promises "three sets of results", but four italic blocks follow. The conclusion (p. 58) repeats "does three things" and then adds a fourth.
4. *Headlined numbers the paper disowns:*
   - The abstract states "an elasticity of substitution near 0.7". Five of seven κ-free estimates lie between 0.51 and 0.62 (Table 4); see M6.
   - The abstract states that the median model's "lifetime inference compute will be about 2.2 times its training compute". The introduction says "The level is not robust" (p. 3), and Section V says "we ... rest our conclusions on signs and ranks" (p. 42).
   - An AER abstract should headline what the paper stands behind.
5. *A theorem presented as a finding.* "In seven public training sweeps, parameters and data are gross complements" (abstract; p. 3). Lemma 1 implies σ < 1 at any interior compute optimum, and the fitted families impose it (p. 11). The paper itself calls this "a check on the data rather than a discovery" (Section IV.B, p. 34).
6. *Imprecision in the abstract.*
   - "The less their data reveal about ... its optimal input mix". Under the maintained optimality assumption, optimal choices *do* reveal the optimal mix (Prop. 3(i)). What they fail to reveal is *M\** *without* assuming optimality.
   - "Once the data choose the curvature" is jargon for "once the outer exponent κ is freed".
7. *Density.* About 110 numbers appear in the results paragraphs of pp. 2–4, often three or four per sentence, with intervals. AER introductions typically carry a small number of headline estimates.

**Proposed frame.** One idea unifies the paper's best parts: *optimization is double-edged for measurement.*

- It removes the variation in outcomes that identifies the technology. That is Section II, with the rate result as its sharpest form.
- It makes choices informative about objectives. That is Section V.
- Designed experiments break the circle: they supply the technology, and choices then reveal what labs planned. The technology estimates (Section IV) are the bridge, and σ\* and *M\** matter because they are exactly what the inversion needs.

This turns the identification theory from a digression into the reason for the empirical design.

**Required actions.**

- **Rewrite the introduction** in this order:
  1. the stakes: investment, and why planned deployment matters (energy, capex, market structure);
  2. the question;
  3. the idea (double-edged optimization);
  4. three results with at most 12–15 numbers in total;
  5. what is robust and what is not, in one paragraph;
  6. the five or six closest papers and what is new relative to each, in two paragraphs;
  7. a roadmap.

  Move the IO citation catalog (p. 5, about 30 citations) into Section I, where each analogy is used.
- **Retitle.** Some options:
  - "What Optimizing Labs Reveal: The Technology of Language-Model Training and the Demand for Inference";
  - "Over-Training and the Revealed Demand for AI Inference";
  - "Optimization, Identification, and Revealed Demand in Language-Model Training".
- **Rewrite the abstract** to state robust claims. A template is in Section 7 below.

### M3. Revealed inference demand: separate the mechanical from the identified, and defend the level.

**Evidence.**

1. *The robust content is nearly mechanical.* I regressed ln ŵ (reference technology) on ln *M* alone in the authors' file (`output/tables/m3_wedge_models.csv`, core Sample B, n = 173). The slope is 0.358, which is (α+β)/2, and R² = 0.998. The reference ŵ is therefore, to three decimals, (*M*/19.2)^0.358.
   - Figure 6 is thus essentially a relabeled x-axis, and the Spearman correlations of 0.95–1.00 across technologies (p. 43) are essentially rank correlations of *D/N*, adjusted for a weak compute trend in *M\**.
   - The ordinal findings (small siblings more over-trained, open more than closed, the switch after 2022) are statements about tokens per parameter. They are visible in model cards and in Epoch's database, and the paper itself calls the within-family result "close to mechanical" (p. 46).
2. *The cardinal content lives entirely in M\*(C) and σ\*.* Section II shows these are the two weakly identified objects: Prop. 3(iii)–(iv), Prop. 4, and Prop. 7 with the Llama-3-8B decomposition on p. 23. The paper's response is to report levels "as ranges" (p. 42), yet those ranges are wide:
   - the median ŵ runs from 1.91 to 4.03 across the six common technologies (p. 43);
   - it is 3.85 with κ free on Farseer, 4.11 under NLS in levels, and 4.78 when the five Chinchilla outliers are kept;
   - the implied inference/training ratio *w* − 1 therefore runs from 0.9 to 3.8. That is a factor of four in the headline quantity.
3. *The validation cannot distinguish demand from accessibility.* The authors say so themselves (p. 48). At fixed *C*, higher *M* means a smaller model; smaller models are downloaded more because more users can run them. OpenRouter lists 21 of 164 models.

**Required actions.**

- (a) **Say what the object is.** State plainly that, within a homothetic family, ŵ is a monotone transformation of *M*/*M\**(*C*). The contribution is the *zero point* (*M\**) and the *scale* (1/σ\* − 1) that turn *D/N* into an economic quantity. Structure Section V around defending those two numbers, not around the ranks.
- (b) **Choose the estimand economists care about and express it as a cost share.** By Prop. 2 and the caveat on p. 41, *w* − 1 is the ratio of planned inference *cost* to training cost, in training-FLOP equivalents, when inference FLOPs cost *p* times training FLOPs. Report (*w* − 1)/*w*, the planned inference share of lifetime compute cost:
  - 0.69 at the reference median;
  - 0.48–0.79 across the technologies above.

  This is directly comparable to industry disclosures. Stop calling it "compute" unless *p* is calibrated. Decoding is memory-bound, so *p* > 1 is likely, and "2.2 times its training compute" (abstract) then overstates inference FLOPs.
- (c) **Validate levels, not just ranks.**
  - *Per-model inference volumes.* OpenRouter publishes per-model token volumes, which Demirer et al. (2025) use. Test whether planned *T*/*D* is proportional to realized OpenRouter tokens across the models served there, at least up to a common scale. The current test uses the number of providers.
  - *Aggregates.* The compute-weighted aggregate (eq. 13, p. 49) implies an inference share of planned lifetime compute. Compare it, carefully and with caveats, to firm disclosures. For example, Patterson et al. (2022) report that about three-fifths of Google's ML energy in 2019–2021 went to inference, and Wu et al. (2022) report that inference accounted for most of Meta's AI infrastructure capacity. Agreement in orders of magnitude would be meaningful external validation; disagreement would be informative.
  - *Model-card coding.* Code each model's stated deployment target (on-device or edge; server or API; research release) from model cards. Stated-intent evidence is currently five hand-picked cases (p. 45).
- (d) **Make the lab-own technologies primary where they exist.** Use Meta's IsoFLOP law for Meta and AI2's ladder for AI2, and extend to any lab that publishes IsoFLOP profiles (e.g., Marin). Report the headline for this subsample separately. Within-lab calibration of *M\** is the paper's own proposed remedy (p. 23, p. 46).
- (e) **Report the sensitivity band as a distribution.** Use one ECDF of ŵ per technology, with the κ-free members included, rather than as vertical gray bars around a curve (Figure 6).

### M4. The reference technology imposes a restriction that the paper rejects.

Section IV states that κ = 1 "is rejected in all seven sweeps" (p. 34) and recommends σ ≈ 0.7 (p. 59). Yet Section V computes the headline wedges under the Chinchilla-form Huber refit (σ\* = 0.737; p. 41), and every member of the "technology band" imposes κ = 1 (Table 6 notes). The authors' file confirms that the band never uses the κ-free Farseer column (`w_farseer_q`). The κ-free Farseer result (median 3.85) appears only as an aside (p. 43).

With κ free, the inner exponents imply 1/σ\* − 1 ≈ 0.43 instead of 0.357 on the Chinchilla data (σ\*_κ = 0.700; Table 3, Panel B). Holding *M\** fixed, this would raise the median ln ŵ by about 20 percent.

**Required action.** Either:
- adopt the preferred (κ-free) technology as the reference, which with the Chinchilla data may require re-estimating *M\**(*C*) under the κ family; or
- explain why the rejected form is the right reference for wedges.

Include the κ-free members in the band. Report the full range of medians in the introduction: at least 1.9 to 4.8 if the NLS-levels and outlier variants are credible, not "1.9 to 4.0" (p. 3). Otherwise say why those variants are excluded.

### M5. The economics of the wedge: who pays for inference, at what price, and under what constraints?

Proposition 2 assumes that the developer minimizes 6*ND* + 2*NT*, so that it bears the inference cost at the training price. This is an economics journal, and the economic model is the paper's weakest link.

1. **Open-weight developers do not pay for their users' inference.** For Meta, Alibaba, Google (Gemma) or Hugging Face (SmolLM), the serving FLOPs are paid by downstream users. A developer minimizes lifetime compute only to the extent that it *internalizes* users' serving costs, through adoption, ecosystem value, cloud revenue or reputation. The revealed *T* is then "internalized inference", λ*T*, with an unknown internalization rate λ, not "anticipated inference".

   This matters for the paper's own findings:
   - Open-weight models are *more* over-trained than closed ones: +0.57 log points, or +0.44 at given ln *C* (p. 46). Closed API providers bear their own serving costs in full, so a model of internalized cost predicts the opposite, all else equal.
   - The developer ranking (p. 46) is hard to square with demand and easy to square with the memory/accessibility rival of Prop. 2(iv), which raises *w* without *T*: Hugging Face 6.6, Alibaba 4.8, Google 2.3, Meta 1.8, Cerebras 1.0. Hugging Face, whose SmolLM models are positioned for on-device use, tops it.
   - The hardware-tier test (p. 49) captures only one discrete version of that rival: bunching at 6.5–9.5B parameters, with an effect of 0.11 (0.06).
2. **The price of an inference FLOP.** The paper notes on p. 41 that *w* = 1 + *pT*/(3*D*), but then reports *T* and "compute" as if *p* = 1. See M3(b).
3. **Omitted data costs.** Tokens are free in the objective. Licensing, filtering (e.g., classifier compute for FineWeb-Edu) and synthetic-data generation (teacher inference) are not. A positive per-token price lowers *w* for a given *T*, so it understates *T*. This rival has a known sign and should be added to Prop. 2(iv), together with the data and memory constraints.
4. **Test-time compute and latency.**
   - Roberts et al. (2026, "Test-Time Scaling Makes Overtraining Compute-Optimal") show that pass@k and reasoning make over-training optimal. This is inference demand, but it changes the interpretation of the 2024–2025 rise in ŵ (Figure 7): part of the rise may reflect more tokens per query rather than more queries.
   - Bian et al. (2025) treat latency constraints, which act like memory constraints (higher *w*, no *T*). Neither paper is cited in the main text.
5. **Family-level data choices.** In the authors' file, 24 of 44 multi-model families train all sizes on token counts that differ by at most 10 percent. These families hold 48 percent of Sample B. For them *D* is a family-level pipeline choice (Llama 3: 15T; Qwen2.5: 18T; Qwen3: 36T), and the size-specific first-order condition in *D* is not the operative margin. The "Pythia warning" on p. 45 generalizes.
   - Report the headline for families with size-specific *D*: the authors' 18-family "informative" subsample of p. 47, or its analog.
   - Discuss what a common-*D* rule reveals.
6. **Distillation and pruning.** Several small models were trained by distillation or pruned from larger ones (e.g., Gemma 2/3 small models; Llama 3.2 1B/3B). Their technology differs, and teacher compute is an omitted intermediate input. That is the gross-output problem the paper itself raises in Section VI. Only 5.8 percent of Sample B is flagged as distilled, which looks low for 2024–2025 small models. Audit the flag.

**Required action.** Add a short model section of about 1.5 pages, extending Hao & Merrill's (2026) profit-maximizing developer to σ > 0, since their Leontief approximation rules out over-training. It should include:
- (i) a developer that bears a share λ of serving cost: λ = 1 for a closed API provider, λ < 1 for an open-weight developer;
- (ii) a user-side memory or latency constraint;
- (iii) a per-token data cost.

Derive what the inversion identifies in each case. Then test the model's distinguishing predictions: open versus closed; developer type (API business versus research or on-device); stated deployment target. Without this, the "anticipated inference demand" label is not warranted for open-weight models, which are the paper's sample.

### M6. The elasticity of substitution: what is shown and what is claimed.

1. *Complementarity.* The paper does not find it empirically (see M2, item 5). The only form-free evidence is Farseer's local quadratic (median 0.690). Say so, and stop listing "gross complements in every sweep" as a result.
2. *"Near 0.7" is not a common value.* The κ-free estimates in Table 4 (p. 34) are:

   | Sweep | σ\*_κ (s.e.) |
   |---|---|
   | Chinchilla | 0.701 (0.014) |
   | Farseer | 0.710 (0.003) |
   | Gadre, C4 | 0.607 (0.054) |
   | Gadre, RedPajama | 0.623 (0.059) |
   | Gadre, RefinedWeb | 0.594 (0.062) |
   | OLMo ladder | 0.544 (0.021) |
   | Muennighoff | 0.511 (0.056) |

   Five of seven lie between 0.51 and 0.62. The differences are statistically large: OLMo against Farseer gives z ≈ 7.8, and Muennighoff against Farseer gives z ≈ 3.5. Either σ differs across recipes and designs, or the κ family is misspecified in some sweeps.
   - Report a formal heterogeneity test and a random-effects summary with the between-sweep standard deviation.
   - Make the abstract match the conclusion's more honest "0.5 to 0.8" (p. 59).
3. *The κ family is one alternative among many.*
   - κ̂ ranges from 0.22 (OLMo) to 0.77 (Chinchilla).
   - In the small sweeps the inner scale parameters are nearly unidentified (the module review reports *A*′, *B*′ of order 10⁶–10⁹).
   - The paper's own out-of-sample comparison favors Farseer's non-homothetic form (Table 4, Panel B).

   Present the κ-free estimate as a specification check. Anchor the level on the most flexible forms the data support.
4. *Counting.* "Seven sweeps" includes three corpora from one study (Gadre et al.) and sometimes DataDecide:
   - σ\* is reported as "0.73–0.83 ... standard errors between 0.006 and 0.034" on p. 3, but "0.735 ... to 0.828 (DataDecide), with standard errors between 0.005 and 0.034" on p. 34;
   - Table 4 has eight rows.

   Use "five studies; seven study-by-corpus designs" consistently.
5. *The capital–labor comparison.* The shaded 0.4–0.7 band in Figure 4 and the text on p. 35 invite a comparison between objects with different inputs, outputs and units. Either drop it or explain what economic inference it supports.

### M7. Show that the estimated objects matter for an economic question.

A generalist will ask why a 0.03 difference in σ\*, or a factor of 20 in *M\**, should matter to economists. The paper currently answers in two paragraphs of the conclusion (p. 59). It needs a short section (about 1,000 words, one exhibit) with quantitative applications built from its own estimates. Three natural candidates:

1. **The allocation exponent and the growth of data demand.** Compute-optimal data demand grows like *C*^(1−*a*). At the 4–5× annual compute growth the paper cites (p. 1), the paper's own cross-sweep range *a* ∈ [0.37, 0.57] (p. 3) implies that data demand grows 1.9× to 2.6× per year at 4.5×.
   - Over five years that is 25× versus 114×, a factor of about 4.5 in cumulative data needs.
   - This is the input behind "running out of data" forecasts (Villalobos et al. 2024) and the policy debate on data licensing.
   - It also turns the paper's identification result (that *a* and *M\** are the least portable objects) into an economically consequential finding.
2. **σ\*, *M\** and the cost of a data wall.** Compute the compute-equivalent cost of a binding cap on unique high-quality tokens, and the shadow value of a token at frontier allocations, under the estimated technologies (κ = 1 versus κ free).
   - If the difference between σ\* = 0.70 and 0.74 barely moves the answer, say so: that is informative too.
   - Connect to the economics of data (Jones & Tonetti 2020; Farboodi & Veldkamp) and to data-licensing markets.
3. **The inference share.** Over time and aggregated, (*w* − 1)/*w* is a revealed-preference indicator of expected deployment, relevant for AI energy and capital-expenditure planning and for the inference market (Demirer et al. 2025).
   - The aggregate on p. 49 is labelled "illustrative" and ranges from 0.67 to 3.26 across technologies.
   - After M3–M5 it should either become a measurement or be dropped.

The frontier elasticity γ is the paper's most robust object (1/γ ≈ 5.6; pp. 10, 59). Showing its use in one growth-model calibration (e.g., the GATE-type models the paper cites) would also help.

### M8. Positioning relative to the closest papers.

- **Hao & Merrill (2026).** Credit is given for the σ formula and the Leontief simplification (p. 11). But their model is richer than this paper's on the demand side: a profit-maximizing firm, the lifetime cost 6*nd* + 2*nt*, and a quality-threshold demand. The paper should frame Prop. 2 as the inversion of an HM-type problem once σ > 0, and borrow their demand structure for M5.
- **Kricheli et al. (2026).**
  - The distinction "their collinearity arises from design; ours from optimizing behavior" (pp. 5, 16) is less consequential than claimed. Observational models today lie far *off* the path: the benchmark panel's off-path spread is 1.64, above Chinchilla's 1.35 (p. 27), and the paper says so. The functional-dependence problem therefore bites mainly in single-lab compute-optimal ladders, which are designs.
  - The genuinely new element relative to Kricheli is the rate result (Prop. 4). Lead with it.
  - Reconcile the two results explicitly. Kricheli's ill-conditioning worsens as |α − β| → 0 on a single ray, while Prop. 3(iii) says σ\* is first-order identified on the path *iff* α = β. As I read them, the first concerns the levels *A* and *B* and the second the exponents. Readers will otherwise see a contradiction.
- **The IO antecedents of "the better labs optimize, the less their data reveal."** The idea that optimizing firms facing common prices generate data that cannot identify the technology is well known in IO: Bond & Söderbom (2005) for Cobb–Douglas without adjustment costs, and Ackerberg, Caves & Frazer (2015). Frame the slogan as a quantitative refinement (the O(v²)/O(v⁴) rates). Relate the rates to the econometrics of singular or near-singular information (Sargan 1983; Rotnitzky, Cox, Bottai & Robins 2000) and to optimal design (D-optimality, which Kricheli et al. discuss).
- **The forward inference problem.** Sardana et al. (2024) are cited. Bian et al. (2025) on latency, Roberts et al. (2026) on test-time scaling, and de Vries (2023) are in the authors' notes but not the main text. All bear directly on the interpretation of *w* (M5).
- **Whitfill (2025), Mertens et al. (2026), Ho et al. (2024), Gundlach et al. (2025).** These are engaged in Section VI. If that section moves to a companion paper, keep one paragraph in the introduction that states the relation in a sentence each and points to the companion.
- **Placement.** The related-literature discussion (p. 5) begins after four pages of results and cites about 70 works. For the AER, name the five or six closest papers on p. 2, say exactly what is new relative to each, and leave the catalog to the body.

### M9. The controlled experiment: design comments (results not yet available).

The experiment is well conceived as a *controlled* comparison: a shared tokenizer, a common output, annealed endpoints at every budget, and common random numbers. Its stated purposes (`paper/notes/m9_spec.md`) are:
1. κ-free σ per lab on a common output;
2. neutrality of data quality;
3. an extrapolation check of ε_D at high *M*, which is the key open issue for Section V;
4. seed noise;
5. functional dependence with real losses.

As designed, it can deliver (1), (2), (4) and (5) with limited power, and (3) only in a narrow sense. Specific concerns:

- (a) **Parameter conventions limit the high-*M* check.**
  - Under non-embedding counts *M* reaches 2,031; including the tied embedding it reaches only 555 (Table 2, note a).
  - The reference technology and Sample B use total parameters. Under the consistent convention the experiment therefore extends the design support only from Chinchilla's *M* = 341 to 555.
  - In the authors' file, 44 percent of Sample B lies above *M* = 555 and 18 percent above 2,031.
  - The TBD text in Section V.B (p. 44) targets *M* ≈ 1,000–2,000, which is valid only under the non-embedding convention.
  - At widths 128–192 the embedding is the majority of parameters (8,192 × 128 ≈ 1.0M against 0.39M non-embedding at width 128), so the convention will dominate *a* and *M\**. The paper's own Section IV.D warns about exactly this.
  - Pre-specify the convention for the extrapolation test, and report both conventions.
- (b) **The experiment cannot address extrapolation in compute.**
  - All runs use less than 10¹⁷ FLOP (p. 27), while 75 percent of Sample B exceeds 1.3 × 10²² FLOP (p. 42).
  - Whether a high-*M* bias in ε_D measured at 10¹⁵–10¹⁷ FLOP transfers seven orders of magnitude up is an assumption. Non-homotheticity, which the paper calls "the most serious rival" (p. 49), is precisely the failure of that transfer.
  - Frame the check accordingly. Consider adding public high-*M* runs as external checks: Sardana et al. trained up to about 10,000 tokens per parameter, and the Marin profiles reach *M* ≈ 2,750 (Table 2).
- (c) **Learning rate tuned for one lab at one budget.**
  - Per `code/sweep/run_grid.py`, the LR rule was calibrated only on the FineWeb-Edu lab, at *D* = 50M. It used three widths (128, 320, 512) and four LRs on a factor-2 grid, fitted with a quadratic in log LR.
  - The rule is then applied to both labs, to all six budgets (25M–800M) and extrapolated to width 640.
  - By the paper's own flexible-input result (Section IV.D, pp. 38–39), inefficiency whose gradient is transverse to the path biases *a* and *M\**. Step Law and Bjorck et al. show the optimal LR depends on *D*.
  - Worse, a lab-specific LR mismatch is a *factor-biased flexible-input inefficiency*. It will be indistinguishable from factor-biased data quality in the neutrality test, which is purpose (2).
  - Run at least a three-point LR check for the FineWeb lab and at the corner cells (smallest and largest *M*) in both labs. Then report that the endpoint loss is flat in LR near the rule at those cells (an envelope argument), or re-tune.
- (d) **A fixed 250-step warmup is Kaplan's mechanism in miniature.**
  - With batches of 16,384 tokens, warmup is about 16 percent of steps at *D* = 25M and about 0.5 percent at 800M. With the 20 percent cooldown, short runs spend a smaller share of steps at peak LR: about 64 percent, against 80 percent.
  - This is a *D*-dependent inefficiency of exactly the kind the paper blames for about half of the Kaplan–Chinchilla gap (p. 38).
  - Scale warmup with steps, or show robustness to dropping the 25M budget.
- (e) **Branching creates strong within-width dependence.**
  - Every endpoint at a given width shares a trunk up to 0.8*D*_k. The six endpoints of a width are therefore not independent draws, and residuals will be correlated within width.
  - Valid inference requires clustering by width: 8 clusters per lab, the same few-cluster problem as Chinchilla's 9 budgets (p. 31).
  - Estimating σ\* with κ free relies on fourth-order information (Prop. 4). With effectively 8 trunks per lab it may be imprecise.
  - Pre-specify few-cluster inference (wild-cluster bootstrap with Webb weights, or randomization inference). Consider independent seed trunks for at least two budgets at every width.
- (f) **Seed replicates cover one lab only** (FineWeb-Edu; widths 128/256/384; *D* = 50–200M). The neutrality test compares labs, and noise may differ by corpus. Replicate in both labs, including the high-*M* cells where the extrapolation claim is made.
- (g) **The "two labs" label oversells.** They are two corpora, one a classifier-filtered subset of the other, with identical training recipes. What is tested is one particular quality filter, not "data quality" or "labs".
  - Use "two corpora" in the text.
  - Add a third, neutral evaluation set drawn from neither training distribution (e.g., a Paloma or Wikipedia subset), and report bits per byte. Otherwise a common output measured on either lab's own validation set confounds quality with distribution match, a concern the paper raises for DataDecide on p. 36.
- (h) **FLOP accounting.**
  - The run records compute including attention and unembedding (`m9_spec.md`), while the text uses *C* = 6*ND* (p. 26).
  - At width 128 the unembedding alone costs about 6 × 8,192 × 128 ≈ 6.3M FLOPs per token, against 6*N*_nonemb ≈ 2.3M.
  - State which *C* defines the IsoFLOP geometry and *M\**(*C*), and report both.
- (i) **Power.** The Gadre comparison had 3 corpora and 104 runs, yet it could not distinguish Hicks-neutral from factor-augmenting alternatives (*p* = 0.13–0.15, p. 36). Two corpora with 44 endpoints each may face the same problem. Use the paper's own Monte Carlo machinery (Section II.B; Appendix C) to compute, *before* seeing results, the expected standard errors of σ\*_κ, of the neutrality LR test, and of *w*_extrap/*w*_full under the design, using the observed seed-noise SD.
- (j) **Pre-analysis plan.** Deposit a dated plan now (E4). It should fix:
  - the primary output (evaluation set and units);
  - the parameter convention;
  - the estimator (Huber or NLS);
  - the inference method;
  - the exact tests for purposes (1)–(3);
  - how the results will be reported whether they support or contradict Sections IV–V.

If budget forces a choice, I would reallocate compute from the neutrality question, where Gadre and DataDecide already give the public evidence, to purpose (3). Run replicated high-*M* cells at the largest affordable width, which is where the experiment can move the paper's headline.

### M10. Section VI belongs in a companion paper.

Section VI (pp. 50–58) answers a different question: what cross-lab data reveal about the technology and about algorithmic progress. Its results are either null or not robust:

- The LaLonde-style test is a failure to reject a *joint* null. That null combines a common technology up to Hicks-neutral shifts, a one-to-one mapping across evaluation harnesses, and no transmission or selection.
  - The 95 percent interval for the OLS bias is about ±23 percent of the benchmark (p. 52).
  - It covers only models with at most 9B parameters.
  - Its "experimental truth" is a fitted quadratic surface from other recipes (OLMo and Gadre).
- The IO remedies that the introduction's framing leads readers to expect (control functions, dynamic panels, instruments) turn out to be unavailable in public data (p. 53). The "laboratory for IO" pitch (pp. 5, 59) is therefore not delivered.
- The allocative gain from the Kaplan→Chinchilla rebalancing is "not even [robust in] sign" (p. 58).
- The Ho et al. finding is valuable, but it is a replication comment: the published point estimate is not the minimizer, and estimators give 6.1–10.2 months.
- TFP dispersion and the industry Monte Carlo add further topics.

**Action.** Move Section VI, with Tables 8–9, Figures 8–9 and Props. 5–6, to a companion paper, where it can be developed properly. For example, that paper could apply ACF-type estimators within families, as the industry Monte Carlo suggests, and turn the Ho et al. ridge into a general statement about identifying biased progress.

If the authors insist on keeping a trace in this paper, limit it to one paragraph in Section V or VI and remove it from the abstract and introduction.

### M11. Formal apparatus and notation.

- **Too many formal results.** Seven propositions and a lemma appear in pp. 8–23 before any data.
  - Keep Prop. 1 (compressed), Lemma 1, Prop. 2, and a merged Prop. 3–4.
  - Move Props. 5–7 to the appendix, keeping a remark where each is used.
  - Move "Every statement was also checked symbolically ... the replication package reports 61 checks" (p. 15) to the replication documentation.
- **Scope of the identification theory.** Prop. 3 assumes on-path samples of labs that minimize *training* compute. The paper's own evidence shows that post-2022 labs minimize *lifetime* compute and sit far off the training path (Section V). State early, in Section II, which data the theory applies to: single-lab ladders, compute-optimal-era releases, and designs like DataDecide's. State also the role that heterogeneous inference demand plays as an "exclusion restriction" in observational data (p. 16). This is where the paper's two halves meet, and it deserves a sentence in the introduction.
- **Notation clashes.**
  - *E* denotes both irreducible loss and the expectation operator, e.g., "E[y | c] ... outcomes identify E" in Prop. 3(ii)–(iii) (p. 15).
  - *a* is the allocation exponent while *a*₁ is the inner exponent on *N*, and likewise *b* versus *b*₁ (eq. 5).
  - *T* is inference tokens while *T_C* is a doubling time.
  - κ is written *q* in Figure 4's legend.
  - "Labs" means experimenters in designed sweeps, developers of released models, and the two corpora in the experiment.
  - "Sample B" appears without a Sample A in the main text.

  Rename throughout, for example *L*∞ for irreducible loss, *s*_N for the allocation exponent, and "verified sample" for Sample B.

### M12. Exhibits: fewer, and each with one message.

- **Figure 1** (p. 12): good. It can carry the identification idea as well, so Figure 3 becomes redundant.
- **Figure 2** (p. 19):
  - Four dense panels with small fonts. Panel A's title restates the slogan, although the dual estimator's RMSE *rises* with *s*.
  - The notes admit that one of nine starting values is the truth, "which flatters the near-path designs". Remove that start from the Monte Carlo rather than footnote it.
  - Keep two panels: RMSE of σ\* and ln *M\** against *s*, with IsoFLOP reference lines; and the κ-free profile.
- **Figure 4** (p. 33):
  - Panel (b), 25 recipe labels at tiny size, conveys one number (homogeneity across recipes: 0.79–0.84). Replace it with a sentence.
  - Panel (a): relabel *q* as κ, drop the capital–labor band (M6), and order the sweeps by design spread *s_M*, so readers see that identification tracks design.
- **Figure 6** (p. 42): a near-deterministic curve with about 15 labels and gray bars (M3). Replace it with ECDFs of ŵ, or of the inference cost share, by technology. Add a separate small panel for the within-family relative wedges.
- **Figure 7** (p. 45): report counts by group and year on the plot. The closed-model series in 2025 rests on 3 models (p. 46); show intervals or truncate.
- **Figures 8 and 9 and Tables 8 and 9** (pp. 51–57): move with Section VI (M10). Figure 8 duplicates Table 8, and Figure 9's fonts are unreadable in print.
- **Table 3** (p. 30): four heterogeneous panels, two kinds of standard errors (parentheses and brackets), and a note of about 200 words. Keep Panels A–B only.
- **Table 4** (p. 34): eleven columns with two-row entries at footnote size. Split it into (i) the κ-free and nonparametric σ with *s_M*, and (ii) the κ = 1 allocation objects (*a*, γ, *M\**), to make the "what is portable" point visually.
- **Table 6** (p. 44): add the inference cost share (*w* − 1)/*w*. Drop CE or move it to the notes.
- **Table 2** (p. 25): trim to the datasets used in the main text after the cut.
- **Table notes throughout:** many exceed 150 words (Tables 3, 4, 8, 9). Move estimator details to the appendix and keep notes to definitions and sources.

---

## 5. Minor Comments

1. **Abstract.** It has about 105 words, which is fine, but it uses "gross complements" (theorem-implied; M2), "near 0.7" (M6), "once the data choose the curvature" (jargon) and "2.2 times its training compute" (not robust; compute versus cost, M3(b)). See the template in Section 7.
2. **p. 1.** "Economists have studied the estimation of production functions for eighty years" is fine as rhetoric, but the next sentence ("many of the difficulties ... are instances of problems that literature has named and, in part, solved") promises a transfer of *solutions*. The paper mostly delivers diagnoses (M10). Temper it.
3. **p. 2, the Monte Carlo sentence.** The comparison (RMSE 0.28 versus 0.004) is striking. Say in the same sentence that the near-path precision under κ = 1 is functional-form information (the paper says this on p. 19).
4. **p. 3.** "the entire range [0.50, 0.95]" should read "the entire grid searched, [0.50, 0.95]". The interval is bounded by the grid.
5. **p. 3.** Reconcile the standard-error ranges for σ\* between p. 3 (0.006–0.034) and p. 34 (0.005–0.034). The difference comes from including DataDecide.
6. **Llama 3 8B figures.** *M* is 1,875 on p. 13 and 1,868 in Table 6; *T*/*D* is 12.7 on p. 13 and 12.8 on p. 43; the introduction's "6 under Meta's own scaling law" is 6.3 on p. 43. Use one set of numbers (exact *N* = 8.03B) throughout.
7. **p. 4.** "log downloads rise about one for one with log tokens per parameter". The coefficient is on ln *M* at fixed ln *C*, where ln *N* = (ln *C* − ln 6 − ln *M*)/2. The "one for one" reading invites misinterpretation, so give the equivalent elasticity with respect to size.
8. **p. 4, Ho et al.** "We replicate ... exactly" and "[the point estimate] is not the minimizer of their objective" are important claims. If any of this stays in the introduction, give the objective values (0.0518 versus 0.0507, from p. 56) and the fact that the MSE difference is only 0.27 percent in the same sentence, so readers see the magnitude immediately.
9. **p. 7, eq. (1).** Productivity terms ω, ψ_N and ψ_D are introduced here but used only in Sections II.D, IV.C and VI. After the cut, introduce them where needed.
10. **p. 8.** "Three analogies break down" is a useful paragraph. Keep it, and move it before the dictionary.
11. **p. 13.** The Harberger approximation "overstates the loss by 23 percent at *w* = 5.2". If the exact formula is used, drop the approximation from the main text.
12. **p. 16.** "(standard errors of 125 and 1,293)" for *A* and *B* is more convincing with the corresponding point estimates beside it.
13. **p. 19, Figure 2 notes.** Remove the true parameter from the primal's starting values (see M12). Also report Monte Carlo standard errors for coverage rates (e.g., 67 percent from how many replications?).
14. **p. 22.** "In an illustrative calibration with γ = 0.178 and π₁ = 2" should say where π₁ = 2 comes from, or be dropped.
15. **p. 26.** State the context length (256 tokens), batch size, and precision (fp32 or bf16 in MLX) in the text. They matter for comparability with production models.
16. **p. 27.** "Measurement is a smaller problem". The paper's own ∂ln *w*/∂ln *D* = β ≈ 0.37 means a 20 percent tokenizer difference moves ŵ by 7 percent (p. 41). Say whether *D* is harmonized to a common tokenizer anywhere. If not, report sensitivity to a bytes-per-token normalization.
17. **p. 28.** "Huber ... is least absolute deviations in practice". This is a nice diagnostic, but it belongs in the appendix for this audience.
18. **p. 31.** The truncated-Gaussian argument for the five outliers maintains Gaussian errors that the outliers contradict (the paper says so). Say directly that the swing is leverage in the *D/N* → 0 corner, and move the Hausman–Wise digression to the appendix.
19. **p. 32.** The revealed-preference test at Chinchilla-70B (*w* = 1.04) is an elegant validation. Consider promoting it to the within-lab validation in Section V.
20. **pp. 35–36, data quality.** "Whether 'better' data are data-augmenting is therefore not identified here; under common exponents a lower *M\** given the tilt is an identity". If this is an identity, the text on factor bias moving *M\** threefold should be stated as descriptive.
21. **p. 42.** "Seventy-five percent use more compute than the largest budget of any design in the band". Clarify that this refers to Chinchilla's 1.3 × 10²² FLOP, and give the corresponding share for the κ-free and Farseer technologies.
22. **p. 43.** Qwen3 0.6B: "its level is not credible, but its rank is". Given M3, the rank is the rank of *M*. Consider dropping extreme-*M* models (e.g., *M* > 10,000) from level statistics.
23. **p. 45, stated intent.** Five hand-picked cases, "without statistical power". Either systematize the check through model-card coding (M3(c)) or move it to a footnote.
24. **p. 46.** The open-weight premium compares *disclosed* allocations: 64 closed models, only 3 of them from 2025. Put this caveat in the introduction wherever the premium is mentioned, or drop the premium from the introduction.
25. **p. 47, over-identification test.** It uses a homoskedastic *F* on ln *T̂* for *T̂* > 0 only, which selects on the outcome, and the paper calls it partly mechanical. Report a version that does not condition on *T̂* > 0, or move it to the appendix.
26. **p. 48, Table 7.** With 20–26 developer clusters, report wild-cluster bootstrap *p*-values alongside the clustered standard errors. State whether downloads of post-trained variants are summed consistently across families.
27. **p. 49.** "about a tenth of the median ln ŵ of 1.16" appears in the hardware-tier discussion. The benchmark 1.16 = ln 3.19 is the Sample B median, but the regression sample is different: production-scale open-weight models released 2023–2026, n = 238. Use the median of the regression sample, or say which sample the benchmark comes from.
28. **p. 59, "What economists should use".** The recommendation of "a range of about 0.5 to 0.8" for σ conflicts with the abstract (M6). The recommendation to take γ "from the sweep closest to the application" needs one sentence of guidance on how to choose.
29. **p. 59, "Implications for models of AI and growth".** "Because σ < 1, a binding constraint on one input ... lowers the return to the other faster than a Cobb–Douglas aggregate would imply". Quantify it (M7), or cut it.
30. **p. 59, Extensions.** The Sutton endogenous-sunk-cost remark is interesting but unsupported here. Cut it, or reduce it to a clause.
31. **References.** Several key references are working papers or arXiv preprints from 2026 (Hao & Merrill; Kricheli et al.; Konig et al.; Czech et al.; Marin). Give version dates, since these papers are revised frequently.
32. **JEL codes.** C51, D24, L86 and O33 are appropriate. After the refocus, consider L63 (microelectronics; computers) or O30 if deployment and innovation become central.
33. **Online Appendix** (63 pages, more than 25 tables). Split it into (i) material referees need to evaluate the main claims and (ii) a supplementary archive. The claims register (Table D-claims) and the verification tallies belong in the replication package.
34. **Data availability.**
    - Hugging Face downloads, OpenRouter listings and LMArena votes are time-varying. Archive the exact snapshots (dated September 2026, pp. 27 and 48) in the replication package.
    - For sources that cannot be redistributed, provide checksums of the downloaded files, so that the download scripts reproduce exactly the analysis inputs.
35. **Style.** AER prose favors the result first and the caveat second. Many sentences here lead with the caveat or embed three or four numbers with intervals, e.g., p. 43: "The lower bound of the bootstrap interval exceeds 1 for 87 percent of models (80 percent with the budget-clustered bootstrap of Section IV), and the whole technology band lies above 1 for 79 percent (136 of 173; median band [1.77, 6.92])".
    - Allow at most one interval per sentence in the main text.
    - Reduce "in the spirit of", which is used for LaLonde, Dehejia–Wahba, GNR, Raval, Leon-Ledesma and others.
    - Avoid chaining analogies to IO papers in place of stating the result.

---

## 6. A Suggested Structure for the Resubmission

(Word targets are approximate; "Exh." counts main-text figures and tables.)

| Section | Content | Words | Exh. |
|---|---|---|---|
| Introduction | Stakes; the double-edged role of optimization; three results; what is robust; closest papers; roadmap | 1,800 | 0 |
| I. The Training Problem | Technology, multiplicative cost, duality (short), Lemma 1 (σ < 1), Prop. 2 (inversion) with the internalization/memory/data-cost extensions of M5 | 2,300 | Fig. 1 |
| II. What Optimizing Labs' Data Identify | Merged Prop. 3–4 (functional dependence plus rates); Monte Carlo (2 panels); Chinchilla on-path evidence; scope (which data the theory governs) | 1,800 | Fig. 2 (2 panels + inset) |
| III. The Technology from Designed Variation | Data (brief); Chinchilla (short); sweeps; κ; heterogeneity; the controlled experiment as the new evidence; one paragraph on conventions and flexible inputs | 2,800 | Table 1 (data + estimates), Fig. 3 (σ by sweep; experiment) |
| IV. What Over-Training Reveals | Measurement under the preferred technology; results as cost shares; lab-own technologies; family-level *D*; validation (OpenRouter volumes, aggregates, model cards); rivals (open versus closed, memory, test-time) | 3,000 | Fig. 4 (ECDFs by technology), Table 2 (selected models), Fig. 5 (trends) |
| V. Economic Implications | Data-demand growth and the allocation exponent; the data wall and the shadow value of data; the inference share of AI compute | 1,000 | Table 3 or Fig. 6 |
| VI. Conclusion | | 600 | |
| **Total** | | **≈13,300** | **7** |

**Companion paper:** current Section VI, Props. 5–6 and Tables 8–9 and Figures 8–9 (observational production functions, algorithmic progress, allocative versus technical change).
**Online Appendix:** Table 1 (dictionary), Section IV.D and Table 5, the duality tests, the selection rules, and the robustness tables.

---

## 7. A Template Abstract (numbers in brackets to be filled after M3–M6)

> Language-model developers choose model size and training data to minimize the cost of training and serving. We show that such optimization cuts both ways for measurement. It removes the variation that identifies the training technology: information about the optimal input mix is second order, and about curvature fourth order, in developers' allocation errors. At the same time, it makes developers' choices informative about their objectives. Using designed training experiments, we estimate an elasticity of substitution between parameters and data of [0.5–0.7]. Inverting developers' first-order conditions, we find that open-weight models released since 2024 are built as if serving would account for [X–Y] percent of their lifetime compute costs, up from near zero in 2021.

---

## 8. Summary for the Editor

The manuscript contains three good ideas: optimization-induced loss of identification with explicit rates; the requirement σ < 1 at interior compute optima; and the inversion of the inference-aware first-order condition. It also contains a large body of careful empirical work. In its current form, however:

- it is several papers at once;
- its introduction, abstract and title emphasize its least novel element and its least robust numbers;
- its headline measurement is almost entirely a transformation of tokens per parameter, with a zero point and a scale that the paper itself shows are weakly identified and computed under a rejected restriction;
- its economic model does not fit open-weight developers, who do not bear their users' inference costs;
- it never shows why the estimated objects matter for an economic question.

A refocused paper could be a strong AER submission. It should have one frame, three results, an explicit developer model, validated levels, a quantitative economic application, a completed and pre-registered experiment designed to address the extrapolation problem, and Section VI spun off.

---

### Works mentioned in this report and not cited in the manuscript's main text

- Bian, S., M. Yan and S. Venkataraman (2025). "Scaling Inference-Efficient Language Models." ICML 2025; arXiv:2501.18107.
- Bond, S. and M. Söderbom (2005). "Adjustment Costs and the Identification of Cobb Douglas Production Functions." IFS Working Paper W05/04.
- COPE Council (2023). "Authorship and AI tools." COPE position statement.
- de Vries, H. (2023). "Go smol or go home." Blog post.
- Farboodi, M. and L. Veldkamp (2021). "A Growth Model of the Data Economy." NBER Working Paper 28427.
- Jones, C. I. and C. Tonetti (2020). "Nonrivalry and the Economics of Data." *American Economic Review* 110(9): 2819–2858.
- Patterson, D., et al. (2022). "The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink." *Computer* 55(7).
- Roberts, et al. (2026). "Test-Time Scaling Makes Overtraining Compute-Optimal." arXiv:2604.01411.
- Rotnitzky, A., D. R. Cox, M. Bottai and J. Robins (2000). "Likelihood-Based Inference with Singular Information Matrix." *Bernoulli* 6(2): 243–284.
- Sargan, J. D. (1983). "Identification and Lack of Identification." *Econometrica* 51(6): 1605–1633.
- Villalobos, P., et al. (2024). "Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data." ICML 2024 (arXiv:2211.04325).
- Wu, C.-J., et al. (2022). "Sustainable AI: Environmental Implications, Challenges and Opportunities." *Proceedings of MLSys* 4.

### Computations by the referee (reproducible from the authors' files)

- **ln ŵ on ln M.** OLS of ln ŵ (reference technology, column `w_chin`) on ln *M* in `output/tables/m3_wedge_models.csv`, restricted to `sample == "B"` and `core == True` (n = 173): slope 0.358, intercept −1.059, R² = 0.998. The implied *M\** is exp(1.059/0.358) ≈ 19.2. Adding ln *C* raises R² to 1.000 (to three decimals).
- **Coverage of the experiment.** Share of the same 173 models with *M* ≤ 341 (Chinchilla's maximum *M*): 45.1 percent. With *M* ≤ 555 (the experiment's maximum *M* counting total parameters): 56.1 percent. With *M* ≤ 2,031 (non-embedding count): 82.1 percent.
- **Family-level token counts.** Of 44 families with more than one model, 24 have max(*D*)/min(*D*) ≤ 1.10. They contain 48 percent of the 173 models.
- **Inference cost shares.** (*w* − 1)/*w* at the median ŵ under each technology: 0.48 (Hoffmann, 1.91), 0.53 (Meta, 2.12), 0.69 (reference, 3.19), 0.74 (Farseer κ-free, 3.85), 0.75 (Gadre, 4.03), 0.79 (Chinchilla with outliers, 4.78).
- **Data-demand growth.** Growth of compute-optimal data demand, (4.5)^(1−*a*) per year: 2.58 at *a* = 0.37 and 1.91 at *a* = 0.57. Over five years: 114× versus 25×.
- **Heterogeneity in σ\*_κ.** From Table 4: z-statistics for OLMo and for Muennighoff against Farseer, treating the estimates as independent: (0.710 − 0.544)/√(0.003² + 0.021²) ≈ 7.8 and (0.710 − 0.511)/√(0.003² + 0.056²) ≈ 3.5.
- **Experiment design details** are read from `code/sweep/run_grid.py` and `paper/notes/m9_spec.md`:
  - LR calibration: `lrsweep` jobs at widths 128, 320 and 512, LR ∈ {1, 2, 4, 8} × 10⁻³, *D* = 50M, FineWeb-Edu only;
  - warmup: 250 steps;
  - seed replicates: FineWeb-Edu only, widths 128/256/384, *D* ∈ {50, 100, 200}M.
