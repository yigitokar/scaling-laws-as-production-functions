# Writer notes: Section II "What Scaling Data Identify" (`paper/sections/identification.tex`)

Writer: section "identification", 2026-09-24. Compiles alone with `code/paper/test_section.sh identification`
(no LaTeX errors, no overfull boxes, no citation warnings; the only undefined references point to other sections).
Rendered pages were checked: Figure 2 fits `\textwidth`, and every display fits the measure.

Length: about 3,420 words of main text (752 of them inside the five propositions) plus about 210 words of figure notes.
The target was about 2,800. Candidate cuts, if needed:
- trim Proposition 6(iii) to its first sentence, since selection is fully treated in Online Appendix A, Prop. A6;
- drop the Section IV preview sentence at the end of II.A, since Section IV reports the same numbers;
- shorten the Kaplan-belief sentence in II.B.

## 1. Structure, labels, and mapping to the appendix

| Main text | Label | Content | Appendix counterpart (appendix_proofs.tex) |
|---|---|---|---|
| eq. (family) | `eq:ident-family` | generalized κ-family with ω, ψ_N, ψ_D, ε | Def. A1/A2 (`def:tech`, `def:kappa`) |
| eq. (geometry) | `eq:ident-geometry` | ln R = κ[−(a₁n+b₁d)/2 + g(τ)] | Lemma A4 (`lem:geometry`), cited via `\ref` |
| Proposition 3 | `prop:fdep` | rank-one Hessian; on-path identification (i)–(iv) | Lemma A4(i) + Prop. A1 (`prop:fd`) |
| eq. (decomp) | `eq:ident-decomp` | y = γ(c − ln6) − ln K − γΦ − ε | Prop. A2(i) |
| Proposition 4 | `prop:info` | information: ln G at O(v²), S or σ* at O(v⁴) | Prop. A2 (appendix label `prop:info`, which CLASHES) |
| Figure 2 | `fig:designs` | `m6_montecarlo_fig2_designs.pdf` | Online Appendix C (m6) |
| Proposition 5 | `prop:dmr` | drift identity; g_N, g_D identified; neutrality iff M* constant; CEG iff equal E and γ | Prop. A3 (`prop:dmr`, which CLASHES) + Cor. A2 (`cor:ceg`) |
| eq. (drift) | `eq:ident-drift` | ∂ln M*/∂t = −[σ*/(1−σ*)]𝓑_t | eq. `eq:A-drift` |
| Proposition 6 | `prop:transmission` | transmission, allocation exponent, selection, proxies | Props. A5 (`prop:transmission`, which CLASHES), A6 (`prop:selection`), A7 (`prop:proxy`) |
| Proposition 7 | `prop:pi` | partial identification of T | Prop. A9 (`prop:pi`, which CLASHES) |

Subsection labels: `sec:ident-fdep`, `sec:ident-info`, `sec:ident-dmr`, `sec:ident-obs`, `sec:ident-pi`.

## 2. Number ledger (every number in the section)

| Number in text | Value | Source (verified) |
|---|---|---|
| Kaplan joint law outer exponent | κ ≈ 0.103 | appendix_proofs.tex Def. A2; theory_main_text.md Def. 1 |
| verification checks | 61 | `output/tables/m7_theory_claims.csv` (61 rows); m7 review |
| equivalent-family loss above frontier at M = 5M*, C = 1e22 | 1.4% (σ* = 0.89), 4.1% (0.74, κ = 1), 18.2% (0.33) | `output/tables/m7_theory_dmr_family.csv`, column `pct_above_frontier_M5x` (1.449, 4.128, 18.214) |
| Figure 1(b) members | σ* = 0.60, 0.74, 0.85 | `output/figures/m7_theory_geometry.png` legend; m7 memo §3 |
| SE of α, β; SE of A, B (Besiroglu) | 0.02; 125 and 1,293 | appendix_proofs.tex Remark after Prop. A1; SYNTHESIS §5.1 |
| κ-free profile LR, Chinchilla data | ≤ 2 (max 1.96) on the 41-run on-path band; 79–101 in random n = 41 subsamples; 424 in the full design | `output/tables/m1_chinchilla_profile_sigma.csv` (max LR 1.962 band, 424.49 full); `m1_chinchilla_fdep_sizecontrol.csv` (`profile_LR_max` 79.0–101.4) |
| Chinchilla transverse spread | sd(ln w) = 0.48 (240 runs, Besiroglu technology) | `data/processed/m7_theory/chinchilla_transverse_sd.txt` (0.4797) |
| Information slopes, 9 × 5 factorial | 4.00 (σ*), 2.00 (ln M* at central compute) | m7 memo finding 5; claims row INFO.num |
| MC truth | σ* = 0.737; M*(1e24) = 18.1 | m6 memo header (0.7370; ln M* = 2.898) |
| MC noise sd | 0.0075 in ln L | m6 memo §2.1 |
| MC design | 90 runs, 9 budgets 6e18–3e21, total 5.1e22 FLOP; IsoFLOP ±16×; 9 × 10 factorial; Kaplan N ∝ C^0.73 with s = 0.1 | m6 memo §2.2; `m6_montecarlo_designA_diagnostics.csv` (total_compute 5.106e22) |
| s = 0: corner share; RMSE σ̂*; RMSE ln M̂* | 51%; 0.28; 24 | `m6_montecarlo_designA_summary.csv` opt_s0 primal (corner 0.510; 0.2819; 23.995) |
| RMSE σ̂* at s = 0.05, 0.1, 0.2 | 0.067, 0.025, 0.015 | same CSV (0.06695, 0.02462, 0.01469) |
| RMSE ln M̂* at s = 0.1, 0.3, 1 | 4.5, 2.0, 0.59 | same CSV (4.491, 1.975, 0.590) |
| 5–95% range of ln M̂* at s = 0.3 | [−0.40, 6.34], a factor of about 850 in M̂* | same CSV (q05 −0.399, q95 6.344; exp(6.743) = 848) |
| IsoFLOP ±16× RMSE σ̂*, ln M̂* | 0.0036, 0.16 | same CSV iso16 primal (0.00357, 0.1619) |
| κ = 1 primal σ̂* RMSE for 0.1 ≤ s ≤ 0.3 | 0.013–0.025 | same CSV (0.0246, 0.0147, 0.0131) |
| dual σ̂* RMSE for 0.1 ≤ s ≤ 0.3 | about 0.012 | same CSV dual (0.0118, 0.0117, 0.0121) |
| κ-free profile CI = whole grid | 88% on-path; 65% at s = 0.3 | `m6_montecarlo_designA_profile.csv` `share_flat_everywhere` (0.880; 0.653) |
| κ-free median CI width | 0.10 at s = 1; 0.016 IsoFLOP; 0.019 factorial | same CSV (0.1044; 0.0161; 0.0186) |
| flat share at s = 0.3 by noise sd 0.005 / 0.0075 / 0.015 | 38 / 65 / 90% | same CSV (0.380; 0.653; 0.900) |
| dual on path RMSE σ*, ln M* | 0.011, 0.009 | summary CSV onpath dual (0.01128; 0.00945) |
| Kaplan-belief dual bias a, σ*, ln M* | +0.22, −0.09, −3.9 (M* about 50× understated) | summary CSV kaplan dual (a 0.2173; σ* −0.0876; ln M* −3.917; exp(3.917) = 50) |
| on-path Wald singular or corner | 35% | summary CSV onpath `wald_computable` 0.652 |
| warm-start bootstrap coverage σ* on path | 67% | summary CSV onpath `boot_cover` 0.673 |
| multi-start bootstrap coverage and width | 92%, median width 0.43 (60 replications) | `m6_montecarlo_designA_bootcheck.csv` onpath sigma_star (0.9167; 0.4285) |
| designed-variation coverage | 95–97% | summary CSV: iso16 Wald 96.0, boot 96.7; fact Wald 94.6, boot 96.7 |
| Llama 3 8B at about 100 × M* (Chinchilla technology) | 1,875 / 17.8 ≈ 105 | `m7_theory_pi_summary.csv` (`Mstar_full` = 17.81 at C = 7.2e23); m6 review §6.1 |
| DMR family σ* range; bias range | 0.33–0.89; 0.05–0.74 per year (g_N = 0.25, g_D = 0.60) | `m7_theory_dmr_family.csv` (`hicks_bias` 0.0464–0.7429); dmr_family.tex note |
| frontier equivalence tolerance | 1e−14 in ln R*; 6 dates × 7 budgets | dmr_family.tex note; m7 memo finding 3 |
| transmission sim vs closed form | within 4e−4 (on path) | `m6_montecarlo_transmission_check.csv` (max abs diff 4.2e−4 on-path rows) |
| funding illustration | γ = 0.178, π₁ = 2: forward 0.248, reverse 0.403 | `m7_theory_transmission_mc.csv` (0.2475; 0.4029) |
| selection with funding (λ = 2) | net bias +0.100, −0.006 (50% dropped), −0.046 (75% dropped) | `m6_montecarlo_selection_check.csv` (0.1000; −0.0064; −0.0461) |
| PI illustration | 300 bootstrap draws; M = 1,875; C = 7.2e23; sd(ln ŵ) = 0.161; M* part 0.143; α+β part 0.047 | `m7_theory_pi_summary.csv` (0.1611; 0.1430; 0.0466; B_boot 300) |
| design support | M = 341 only at C ≤ 1.3e22; 55× compute extrapolation | same CSV (`M_design_max` 341.1; `C_design_max` 1.2956e22; 7.2e23/1.3e22 = 55.6) |
| nonparametric bound | w_b = 0.76 | same CSV (`wb_max_refit` 0.766; `wb_max_besiroglu` 0.763) |

Deliberately NOT quoted: the m7 illustration's T/D = 12.8 [9.1, 18.9] at M = 1,875. Section V (m3) reports
Llama 3 8B T/D = 12.8 [8.9, 21.4] at M = 1,868 (exact HF parameter count), and two different bands would confuse
readers. Only the variance decomposition, which is the point of Proposition 7, is quoted.

## 3. Claims and the caveats attached to them in the text

- Prop. 3: behavioral (not design-induced) functional dependence. Credit to `kricheli2026tokens` for the
  design-induced statistics (per `lit/notes/novelty.md`). We do not claim to be first to note collinearity.
- Prop. 3(iii): "second order" identification is stated as in Prop. A1 (Jacobian rank 3). No convergence rates are
  claimed (m7 open issue).
- Prop. 4: stated for S = α+β, then σ* (m7 review fix 3). The text gives the caveat "local, Gaussian noise, E known".
  Only slopes are quoted, never unit-noise SE levels (m7 review §4.2).
- MC: the warm-start qualifier is attached to the 67% coverage, and the multi-start result (92%, width 0.43) is given
  (m6 review issue 4). The flat-profile share is tied to noise and design size (m6 review §6.1). The κ = 1
  precision at 0.1 ≤ s ≤ 0.3 is called functional-form information, restricted to that range (m6 review issue 5).
  The truth-as-start caveat is in the figure notes. "On-path problems concern ladders and sweeps; released models
  are far off-path" (m6 review §6.1).
- Prop. 5: the sign of the Hicks bias IS identified (m7 correction). The fixed-relative-price caveat, the IsoFLOP-minima
  requirement and several budgets per date are stated. The general bias formula is βg_D − αg_N, with the textbook
  form only for α = β. CEG: constant iff equal E and γ. We do not say "iff equal exponents". The common-rule nuance
  (equal exponents necessary, not sufficient) was cut from the main text for length; it is in Cor. A2.
- Prop. 6: bracket condition as in A5(iii); predetermined regime via effective π₁ = response × ρ_ω (consistent with
  m6 claim 6: γ + λρVar(ω)/Var(c)). Selection closed forms are labeled Gaussian. The proxy result is part (iv).
  "Section VI examines how far either route goes with public data." We do not claim IO remedies work on public
  cross-lab data.
- Prop. 7: heavy extrapolation caveat (55× in compute), and the nonparametric bound is uninformative on Chinchilla
  support. The Sardana direction (conservative) is stated as in the appendix remark.

## 4. Cross-references assumed to exist elsewhere

- `sec:framework` (Section I) must define y = −ln(L−E), G, K, M*(C), the wedge w = ε_N/ε_D, the sufficient statistic
  ln w = (1/σ*−1) ln(M/M*(C)) and the Farrell allocative loss C/C_min as a function of w alone. The Harberger
  approximation Φ ≈ σ*(ln w)²/(4(1−σ*)) is restated here.
- `prop:duality` (Prop. 1): duality, including G ≡ (αA/(βB))^{1/(α+β)} and the Approaches 1/2/3 reading. I cite it for
  the cross-equation restriction G^{α+β} = αA/(βB).
- `prop:wedge` (Prop. 2): w = 1 + T/(3D), i.e. T = 3D(w − 1).
- `lemma:sigma` (Lemma 1): interior compute optima imply 0 < σ < 1.
- `fig:geometry` (Figure 1 = m7_theory_geometry): panel (b) shows σ* = 0.60, 0.74, 0.85 members of the equivalent
  family. That figure labels its panels "(a)/(b)", while Figure 2 uses "A.–D.". I wrote "panel (b)" and "panel A".
- `sec:tech` (Section IV): must report the Chinchilla κ-free profile (LR ≤ 1.96 on-path band vs 79–101 random vs 424
  full) and apply Prop. 5(iii) to data quality (corpora in place of dates).
- `sec:obs` (Section VI): the DMR ridge in the Ho et al. data, and whether dynamic-panel or within-family routes are
  feasible.
- `sec:wedge` (Section V): within-family calibration of M*(C), the GNR direction.
- `tab:data` (Table 2): reports sd(ln M | ln C), the transverse variation, for each dataset.
- `lem:geometry` (appendix Lemma A4): cited by `\ref`. Its label is unique.

## 5. Placeholders

None. The section needs no m9 numbers.

## 6. New bib keys

None. Every key used exists in `paper/references.bib`: ackerberg2015identification, besiroglu2024chinchilla,
diamond1978measurement, gandhi2020identification, ho2024algorithmic, hoffmann2022training, kaplan2020scaling,
klepper1984consistent, konig2026validity, kricheli2026tokens, levinsohn2003estimating, marschak1944random,
olley1996dynamics, sardana2024beyond, whitfill2025note.

## 7. Open issues for the integrator

1. **Label clashes (must fix).** `prop:info`, `prop:dmr`, `prop:transmission` and `prop:pi` are defined both here (as
   instructed) and in `appendix_proofs.tex`. `prop:wedge` will clash once framework.tex defines it. A test compile
   with the appendix gives "Label multiply defined", and the last definition wins, so main-text `\ref`s would print
   the appendix numbers (A2, A3, ...). Rename the appendix labels (for example `prop:A-info`, `prop:A-dmr`,
   `prop:A-transmission`, `prop:A-pi`, `prop:A-wedge`) and their internal references in the appendix. The appendix
   also refers internally to `prop:wedge` (its own A8) in Lemma A3(vi), Prop. A2's remark, Prop. A3's remark and
   Prop. A7.
2. **Proof pointers.** The section says once that "Proofs are in Online Appendix A" and cites Lemma A4 by `\ref`. After
   the relabeling, the integrator may add "Proof: Online Appendix A, Proposition A1" lines after each proposition,
   using the mapping in §1.
3. **Macros.** `\E`, `\Var` and `\Cov` are defined at the top of identification.tex with `\providecommand`, the same
   definitions as the appendix. Suggest moving them to the main.tex preamble. A later `\newcommand{\E}` in another
   section would otherwise error.
4. **Notation changes relative to the appendix.**
   - The Hicks bias is 𝓑_t (`\mathcal B_t`) here. The appendix's B_t clashes with the loss coefficient B.
   - Noise variance is V_ε here. The appendix Prop. A2 uses s², which clashes with the MC allocation-error sd s used
     in Figure 2.
   - Productivity persistence is ρ_ω, to avoid clashing with the homothetic ρ = (α+β)/2.

   Suggest aligning the appendix.
5. **Duplicate definition risk.** The generalized family (`eq:ident-family`) and the derived S, a, b, γ, σ* are defined
   here. If framework.tex also defines the κ-family, keep one definition.
6. **Word count** is about 20% over target. See the candidate cuts at the top.
7. The plan's "M* only by assuming optimality" is stated as in Prop. A1(iii): A/B and the technology's own M* are
   identified only by imposing optimality, and the observed M*(C) is revealed by choices. No discrepancy with the memos.
