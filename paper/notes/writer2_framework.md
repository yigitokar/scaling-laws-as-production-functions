# Writer notes (v2): Section I, "The Training Problem" (framework)

Writer: framework section writer (v2), 2026-09-24. Replaces `writer_framework.md` (v1).

## Files written
- `paper/sections/framework.tex`: Section I, subsections A–D. It overwrites the v1 file; v1 is kept in `sections/v1/framework.tex`.
- `paper/tables/table1_dictionary.tex`: Table 1, a compact dictionary. It has 8 rows (v1 had 18) and is `\input` from framework.tex.
- Figure 1 is `output/figures/m7_theory_geometry.pdf`, included directly. The file is unchanged; the notes are revised.
- No new bib keys. Every key exists in `paper/references.bib`, and the compile shows no citation warnings.

Length, by a crude detex that counts each inline math group and each citation as one word:
- text: about 2,620 words, of which 533 are formal statements (Props. 1–2, Lemma 1);
- figure notes: 124 words;
- Table 1: about 380 words.

The plan target was about 2,300. The overrun comes from v2 content the referees asked for: the generalized wedge, conduct, family budgets, and the DLW critiques. The prose was cut twice, from about 3,000 words.

Compile: `code/paper/test_section.sh framework`.
- No errors and no citation warnings. There is one overfull box of 0.86pt.
- Built together with `appendix_proofs`, every appendix label resolves. The remaining undefined references are listed under open issue 1.

## Labels defined
- Section: `sec:framework`.
- Subsections:
  - `sec:framework:tech` (I.A);
  - `sec:framework:duality` (I.B);
  - `sec:framework:sigma` (I.C);
  - `sec:framework:wedge` (I.D).
  - v1's `sec:framework:dict` is gone, because the dictionary is now part of I.A.
- Equations:
  - `eq:tech` (1): the technology, including κ and productivity;
  - `eq:elast` (2): output elasticities;
  - `eq:sigmaw` (3): σ(w) and σ*;
  - `eq:gw` (4): the generalized wedge;
  - `eq:wM` (5): ln w = (1/σ* − 1) ln(M/M*(C)).
  - v1's `eq:sigma` (the σ formula in u, v) was dropped.
- Results:
  - `prop:duality` (Proposition 1);
  - `lemma:sigma` (Lemma 1: interior compute optima require 0 < σ < 1);
  - `prop:wedge` (Proposition 2: generalized wedge, serving case, constraints and misspecification, family budgets).
- Exhibits: `tab:dictionary` (Table 1), `fig:geometry` (Figure 1).

Proposition 2 has four parts. They map to theory_main_text_v2 as follows:
- (i) general formula;
- (ii) serving case (a);
- (iii) caps, data cap, distillation, and ŵ = w·w_L/w_{L'};
- (iv) family budgets (Prop. A9).

Section IV writers can cite `Proposition~\ref{prop:wedge}(ii)` for the serving reading and `(iv)` for families.

## Numbers used (value, then source; every value rechecked against the CSVs)

| Number in text | Value | Source |
|---|---|---|
| Besiroglu α, β | 0.348, 0.366 | `m7_theory_ledger_checks.csv` (published 0.3478, 0.3658) |
| γ (Besiroglu) | 0.178; 1/γ = 5.6 | ledger 0.17829; 1/0.17829 = 5.61 |
| σ* (Besiroglu) | 0.737 | ledger 0.73703 |
| σ range over all mixes | 0.732–0.742 | 1/(1+β) = 0.7321, 1/(1+α) = 0.7418 (Lemma A1(ii)) |
| 8B/15T run (nominal Llama 3 8B, total N = 8e9) | M ≈ 1,875; w ≈ 5.2; s ≈ 0.81; w − 1 = 4.2 | ledger "Besiroglu w, Llama-3 8B" 5.2175; s = 1 − 1/5.2175 = 0.808 |
| C/C_min for that run | 5.5; training-cost efficiency 0.18 | ledger "Besiroglu C/C_min, Llama-3 8B" 5.5127; 1/5.51 = 0.181 |
| Llama 3 8B under Meta's own technology (Section IV) | w ≈ 8.4 | `ra2_wedge_models.csv`, `w_lab` = 8.356 (s_lab 0.880); ra2 memo H3 |
| Gopher (Besiroglu) | w = 0.36 | ledger 0.3622 |
| Clean inference-demand sample, common-D families | 32 of 77 models in 11 families | `ra2_wedge_family_split.csv` (common-D: n = 32, n_families = 11); `ra2_wedge_models.csv` (clean = 77) |
| Special cases of the wedge | eleven | `output/tables/ra5_theory_wedge_cases.tex` (11 rows) |

Numbers dropped from v1:
- the 1.9×10^14 tokens figure (R4 minor 9; do-not-claim: cardinal T without p);
- the T/D = 12.7 figure;
- the Harberger 23% approximation (R3 minor 11);
- Kaplan σ* = 0.535 and the Hao–Merrill 0.762 (R2 Major 9c);
- Hoffmann's a = 0.46–0.50;
- GPT-3's w = 0.43;
- "173 open-weight models".

## Referee comments addressed in Section I

**R1 (IO econometrician)**
- **1 (conduct).** Proposition 2 now covers any objective V(L, N). The text sets out three conduct models (developer serves; open release valued through adoption; tier caps), gives their distinguishing predictions, and cites the conduct-testing literature (Berry–Haile, Backus–Conlon–Sinkinson, Duarte et al.). "Planned serving expenditure" is used only for developers that serve their models. T is a present value, via the appendix.
- **2(b) (family D).** Prop. 2(iv) covers this, along with the clean-sample count (32/77 models, 11 families), "member wedges are not revealed preferences", and the knife-edge argument.
- **3(a)–(d) (DLW).** The wedge is Raval's ratio statistic. Both critiques appear as the misspecification case: Demirer/Raval factor bias, and Bond et al. output concept. Designed experiments remove the Doraszelski–Jaumandreu circularity at the cost of external validity. De Ridder et al. is cited on levels versus ranks.
- **5 (cost side).** "FLOP-accounting approximation, not an identity." Size-dependent FLOP prices (δ), data costs and serving-price ratios are explicit in Prop. 2. Heterogeneous wedges are presented as the Bond–Söderbom analogue of relative-price variation.
- **6(d).** σ* < 1 is "implied ... not a finding". The empirical content is the U-shaped IsoFLOP profiles, and the level comes from Prop. 4.
- **8(a), (d), (e), (g), (h) (analogies).**
  - Table 1 uses Marschak–Andrews / Bond–Söderbom for on-path collinearity, not ACF.
  - IsoFLOP sweeps are "assigned variation ..., not a shock to relative prices".
  - There is no GNR label and no TFPR row.
  - Nerlove is cited only for the returns-to-scale measure, "found to vary with scale".
  - Scale-correlated mismeasurement is attributed to Basu–Fernald.
- **Minors.**
  - 7: noise placement matters only for estimation, and the rates are unchanged with noise on ln L (ra5, Prop. A2(iv)).
  - 8: timing and information is recast as belief error, covered by ŵ = w·w_L/w_{L'}.
  - 9: quasi-homotheticity.
  - 10: Approach 1 also estimates the factor demands.
  - 11: the empirical content of Lemma 1 is U-shaped profiles.
  - 12: floors and deadlines enter the signs paragraph and the appendix pointer.
  - 13: "training-cost efficiency".
  - 47: Hicks = Allen–Uzawa = Morishima.
  - 51: the serving-cost elasticity and the family budget are in the appendix, and the family budget is in Prop. 2(iv).

**R2 (ML scaling)**
- **Major 2a.** The expenditure share s needs no p; a count of T needs p, which is "likely above one" because decoding is memory-bound.
- **Major 2b.** λ internalization, and the adoption reading for open weights.
- **Major 2c.** RL rollouts and synthetic data enter X_S.
- **Major 4c.** Distillation is a data cost with m_D = N_T/(3N) (Busbridge et al.).
- **Major 5.** Tiers, latency (Bian et al.), fixed-D families and the output concept are all covered.
- **Major 9a.** "IsoFLOP-minima frontier"; Hoffmann's Approach 1 also estimates the factor demands.
- **Major 9c.** The Kaplan σ comparison was removed from the text. Kaplan appears only as the κ-form reference in Table 1.
- **Major 9d.** IsoFLOP minima are "outcomes of inputs that an experimenter assigned, not choices".
- **Minor 3.** 2N FLOPs per token, with the memory-bandwidth caveat.
- **Minor 4.** M* depends on the tokenizer and on whether embeddings are counted. The claim that the exponents are unit-free was dropped.
- **Minor 5.** No T level is given.
- **Minor 6.** The Harberger approximation was dropped.
- **Minor 26.** The Figure 1 notes say "(total) parameters". The figure itself has no non-embedding marker; see open issue 6.

**R3 (editor)**
- **M1.** The dictionary is cut from 18 rows to 8. The duality discussion is short. Farrell/Harberger is reduced to one sentence, with the CE formula in the appendix.
- **M2(5).** Complementarity is not presented as a finding.
- **M3(a)–(b).** The text states that the technology supplies the zero point M*(C) and the scale 1/σ* − 1, and that ranks at given compute are ranks of M. s is the headline object.
- **M5.** λ, memory and latency, data cost, test-time scaling (Roberts et al.) and latency (Bian et al.) are all covered.
- **M8.** Prop. 2 is framed as the inversion of a Hao–Merrill-type problem with σ > 0, and the Leontief limit rules out over-training.
- **M11.** Notation: E is irreducible loss, and 𝔼 is not used in Section I. a₁ and b₁ are absent; the κ family uses α and β as inner exponents. θ is not used. "Developer" and "experimenter" are used consistently.
- **Minors.**
  - 9: the productivity terms are kept compact. They are used in Prop. 1(iii), Prop. 2(iii) and Sections II–III.
  - 10: the "three analogies break down" paragraph now comes before the table.
  - 11: Harberger removed.

**R4 (auditor)**
- **Minor 7 / C7.** The Llama 3 8B illustration is labelled "nominally Llama 3 8B", with total N and Besiroglu parameters, and it points to Section IV's own-technology estimate (8.4).
- **Minor 8 / C8.** Gopher's technology is named.
- **Minor 9.** No T level is given.
- **Minor 5.** Section I defines d ≡ ln D. Width must not be called d in Section III (see open issues).

## Open issues for the integrator

1. **Undefined references until other files exist.**
   - `sec:ident`, `sec:tech`, `sec:wedge` (other writers).
   - `prop:ident`, `prop:modelfree`: Section II writer. They are cited in Lemma 1's discussion and in Table 1.
   - `app:additional`: Appendix D. It is cited in I.B as containing duality tests on the Chinchilla data (v1's `appD_duality`). If Appendix D drops the duality tests, delete the clause "Online Appendix~\ref{app:additional} reports such tests" in framework.tex (I.B, second paragraph).
2. **`tab:ra5-wedge-cases` is referenced but not yet `\input` anywhere.** Section I.D says "Online Appendix A tabulates eleven cases (Table~\ref{tab:ra5-wedge-cases})".
   - Fix: copy `output/tables/ra5_theory_wedge_cases.tex` to `paper/tables/` and `\input` it in `appendix_proofs.tex`, right after Remark A-conduct (`rem:A-conduct`).
   - If it goes into Appendix F instead, change "Online Appendix~\ref{app:proofs}" to "Online Appendix~\ref{app:wedge}" in that sentence.
3. **Notation clash in Appendix A (not my file).** Definition A1 and Lemma A1(i) use s ≡ ε_N/(ε_N+ε_D) for the elasticity share. The main text uses s ≡ (w − 1)/w, as do Prop. A8 and ra2. Suggest renaming the appendix elasticity share, e.g., to ς or s_ε, in Definition A1 and Lemma A1(i).
4. **Main-text result count.** Section I has three formal results (Prop. 1, Lemma 1, Prop. 2). With Props. 3–5 in Section II, the main text has six, against the revision plan's "≤ 5". This is the lead author's call, as ra5 open issue 7 notes. Lemma 1 could become a remark without breaking any label, provided `lemma:sigma` is kept.
5. **Equation (5), `eq:wM`, is defined here.** Section IV's writer should cite `\eqref{eq:wM}` rather than re-derive it, to avoid a duplicate equation.
6. **Figure 1 is unchanged** (m7, Besiroglu κ = 1 parameters). Two referee requests would need a new figure from the figure owner: R2 minor 26 (a non-embedding-N marker) and R3 M12 (Figure 1 carrying the identification idea, which panel (b) already does). The notes now state total N. Figure 1(b) shows σ* = 0.60, 0.74 and 0.85. If Section II quotes off-path losses for other members (R4 minor 6: v1 used 0.89 and 0.33), align them with the plotted members.
7. **Section III must not use d for model width** (R4 minor 5), because Section I defines d ≡ ln D.
8. **Length.** About 2,620 words against a target of about 2,300. If the paper runs long, cut in this order:
   - (i) the DLW paragraph's final two sentences, moving Hao–Merrill and Sardana to the introduction;
   - (ii) the Gopher sentence;
   - (iii) the Hicks/Allen/Morishima sentence.
9. **Table 1 cites Kaplan et al. (2020) for the κ family.** Kaplan's joint law has the outer-exponent form, but it was fitted to early-stopped runs (R2 Major 9c). The text no longer compares Kaplan's σ; if the lead author prefers, replace the citation with `hoffmann2022training` alone.

## Placeholders
None. Section I has no m9-dependent content.
