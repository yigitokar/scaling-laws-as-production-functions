# Revision plan v3 (after referee round 2) — lead author, 2026-09-24 ~18:00

Round-2 reports: paper/referee/round2_R1..R4 (all "major revision"; R3 up from reject-and-resubmit). They converge on
five substantive fixes and a length/exhibit cut. Binding for round-3 analysis modules and writers.

## 1. Headline claims (abstract ≤ 100 words; no level claims; conditioned statements)
Proposed abstract (edit to final numbers, keep the logic):
"Language-model developers trade off model quality, training cost, and the value of a compact model. We show that
optimization cuts both ways for measurement: it removes the variation that identifies the training technology —
information about curvature is fourth order in allocation errors — but makes developers' choices informative about
their objectives. From designed training experiments we estimate, without functional-form restrictions, an elasticity
of substitution between parameters and data of about 0.7 at 10^19–10^21 FLOP, declining at larger budgets. Inverting
first-order conditions, most open-weight models released since 2023 are trained beyond training-cost minimization —
robustly in sign, conditionally in size — and the implied value of compactness rose steeply through 2025."
- σ*: "≈0.70 at 10^19–10^21 FLOP (study-level interval, Marin one study), declining toward ≈0.6 at the largest budgets
  of the two largest designs"; FLOP-accounting (±0.04) and convention ranges stated beside the interval. (R1 N2; R3 N1)
- Revealed object: "the value of compactness" m_N (w − 1 per unit training cost); share s = (w − 1)/w. "Planned serving
  expenditure" only as ONE interpretation, supported where a model-level serving code says the size was served by
  its developer; tier-window models give upper bounds; open-weight premium read as evidence for adoption/tier
  mechanisms, not serving-cost internalization. Rename "clean inference-demand sample" → "clean sample". (R1 N1; R3 N4)
- Decision units: common-D families reveal ONE family-level number (Prop. 2(iv)); member wedges are not revealed
  preferences — except under a binding curated-token CAP, where they are lower bounds (add the cap case; R2 M2).
  Headline median over decision units. Replace the Llama 3 8B showcase with a size-specific family (OLMo 2 or SmolLM2);
  Llama 3 herd = family share ≈ 0.29 with the flagship's sign unidentified. (R1 N1; R2 M2; R3 N3)
- Sign identification: report the share identified as a function of a tilt allowance on M* (1, 1.3, 1.84, 3.4, 4.4),
  compute-weighted as well as unweighted, normal-inputs-only on the harmonized sample; widen anchor intervals (wild
  bootstrap-t); drop "weak assumptions". (R1 N3; R2 M4.3; R4 R2-M3)
- Trend: start in 2023 on the clean sample under one set of rules (unweighted, decision-level, compute-weighted,
  leave-one-developer-out); drop "near zero before 2023" (OPT/GLM dominated; a Kaplan-belief regime); optional
  robustness: pre-2023 under the Kaplan law as the believed technology. Note that the trend largely restates the rise
  in M. (R1 N4; R2 M4; R3 N2; R4 R2-M1)
- "Parametric extrapolation is conservative": state "in one recipe (Farseer)" unless Marin/Llama 3 (and our experiment,
  per Amendment 1) agree. (R2 M5)

## 2. Structure, formal results, exhibits, length
- Main text ≈ 15,000 words incl. the experiment; FOUR formal results: Lemma 1 (σ<1), Prop 1 (generalized wedge incl.
  family budgets + cap), Prop 2 (identification + information; the n^{-1/4} rate moved to a Remark), Prop 3 (model-free
  σ* and w, and the beyond-design sign result). Duality becomes text (part (iii) as a sentence).
- ≤ 8 main exhibits: Fig 1 geometry; Fig 2 Monte Carlo (2 panels); Table 1 designs & σ* (+ σ*(C) meta-regression row);
  Fig 3 σ* by design and budget + experiment panel (merged); Fig 4 ECDF of s (decision units); Table 2 decisions
  (rebuilt: decision units, family shares, model-level serving, tier flags); Fig 5 extrapolation (Farseer + Marin);
  Table 3 economics. Economics figure → appendix.
- Online Appendix ≈ 60 pages: A proofs (keep what the main text needs; verification tallies → replication docs);
  B data; C Monte Carlo; D technology (trimmed); F revealed-demand robustness. Appendix E (observational & progress)
  becomes a separate companion paper (paper/companion/), with one sentence in the introduction.
- Notation clashes: allocation-error scale ς (not v); prices q_T, q_D (not c_T, c_D); path slope ε or e_p (not e);
  b ≡ 1 − a avoided (write 1 − a). Define "labs" to cover designers and developers in the first paragraph.
- Literature to engage (R2 M8): villalobos2023trading, erdil2024optimally, erdil2025inference, lourie2026small,
  bergsma2025power, tissue2024scaling, kumar2024scaling, goyal2024scaling; Bond & Söderbom 2005 already in.
- Post-training compute (acts like δ > 0) and loss ≠ capability (Kumar et al. quantization) discussed in IV.E and
  limitations (R2 M7).
- Section V integrates Section IV: data-demand scenario with rising wedges (D/D* = w^{σ*/[2(1−σ*)]}); data wall at
  observed allocations; cap feedback (measured wedges understate as the wall approaches); fleet comparison with 1/2/3-
  year service lives; compute growth 4.2× and 5.06×; γ guidance for growth calibrations. (R3 N6, minors 17–20)
- Say why the AER should publish (R3 N5): general methodology (designed isocost variation + multiplicative known cost:
  agronomy factorial trials, chip-design sweeps, dose-combination trials; sign-of-wedge result for markups) and the
  economic payoff (what matters and what labs would need to publish).
- Editorial: byline per the user's decision (pending); Ho et al. critique shared with authors — user's decision (note in
  the companion paper); PAP commit hashes (bd5c0ad plan; a5f47f8 amendment 1 + power) cited in III.E; public timestamp
  pending the user's decision.
