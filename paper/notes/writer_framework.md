# Writer notes — Section I "Scaling Laws and the Production Problem" (framework)

Writer: framework section writer, 2026-09-24.

## Files written
- `paper/sections/framework.tex`: Section I, with subsections A–E.
- `paper/tables/table1_dictionary.tex`: Table 1, the dictionary (18 rows in 4 panels). It is `\input` from framework.tex.
- Figure 1 is `output/figures/m7_theory_geometry.pdf`, included directly.
- No new bib keys. Every citation key exists in `paper/references.bib`, and the test compile reports no citation warnings.

Word count, by a crude detex:
- about 2,470 words of main text, including about 450 words of formal statements (Propositions 1–2, Lemma 1);
- 142 words of figure notes;
- the table is extra.

The target was about 2,200. Prose was cut twice; the rest is the formal statements.

Compiles cleanly with `code/paper/test_section.sh framework`:
- no errors;
- no float-too-large warning;
- one overfull box of 0.1–0.3pt, which can be ignored.

The only undefined references are to other sections, and to appendix labels when compiled without the appendix.

With `appendix_proofs.tex` added (my own scratch build, main.tex order with `\appendix`), every appendix reference resolves, **except the label collision below**.

## Labels defined here
- Sections: `sec:framework`; subsections `sec:framework:tech` (I.A), `sec:framework:dict` (I.B), `sec:framework:duality` (I.C), `sec:framework:sigma` (I.D), `sec:framework:wedge` (I.E).
- Equations: `eq:tech` (technology, eq. 1), `eq:elast` (output elasticities), `eq:sigma` (σ formula), `eq:sigmaw` (σ(w) and σ*).
- Formal results:
  - `prop:duality` (Proposition 1);
  - `lemma:sigma` (Lemma 1: interior compute optima ⇒ 0 < σ < 1);
  - `prop:wedge` (Proposition 2: revealed inference demand).
- Table and figure: `tab:dictionary` (Table 1), `fig:geometry` (Figure 1).

## Cross-references I assume exist elsewhere
- `sec:ident` (II), `sec:tech` (IV), `sec:obs` (VI), `sec:wedge` (V).
- Appendix A labels (they exist in appendix_proofs.tex):

  | Label | Result |
  |---|---|
  | `lem:sigma` | Lemma A1 |
  | `lem:soc` | Lemma A2 |
  | `cor:complements` | Corollary A1 |
  | `lem:alloc` | Lemma A3, including part (v) |
  | `lem:geometry` | Lemma A4 |
  | `lem:ce` | Lemma A5 |
  | `cor:suff` | Corollary A3 |
  | `prop:fd` | Proposition A1(iv), cited in the Figure 1 notes |

## OPEN ISSUES FOR THE INTEGRATOR (priority order)

1. **Label collision (must fix).** `appendix_proofs.tex` defines `\label{prop:info}`, `prop:dmr`, `prop:transmission`, `prop:wedge` and `prop:pi`. The writer assignment reserves exactly these labels for main-text Propositions 4, 5, 6, 2 and 7.
   - I verified this in a combined build: LaTeX warns "Label `prop:wedge' multiply defined", and every main-text `Proposition~\ref{prop:wedge}` prints "Proposition A8".
   - Fix: rename the appendix labels (e.g., `prop:A-info`, `prop:A-dmr`, `prop:A-transmission`, `prop:A-wedge`, `prop:A-pi`), and update the appendix's internal references to them. For example, Lemma A3(vi) and Corollary A3 cite `Proposition~\ref{prop:wedge}` meaning A8.
   - For the same reason, my proof pointer for Proposition 2 does not use the colliding label. It reads "Online Appendix A (the proposition on the inference wedge, Lemmas A4 and A5, and Corollary A3)". Once the appendix label is renamed, the integrator may replace that phrase with `Proposition~\ref{prop:A-wedge}`.
2. **Appendix subsection references render badly.** With `\appendix`, AEA.cls sets `\thesubsection` to `\thesection\arabic`, while `\p@subsection` prepends `\thesection.`. So `\ref{app:wedge}` prints "A.A8".
   - I therefore do not reference appendix subsections.
   - Other writers may; fixing `\p@subsection` inside the appendix, or avoiding such references, is needed.
3. **Notation clash on R.** Following the assignment, the main text defines R ≡ L − E (reducible loss) and y = −ln R. The appendix defines R ≡ u + v, the ω-free sum, with L − E = e^{−ω+ε}R. The two coincide when ω = ε = 0. Suggest renaming the appendix object (e.g., R̃ or Q), or adding one sentence there.
4. **Numbering differs from `theory_main_text.md`.** I followed the plan and assignment: Prop 1 = duality, Lemma 1 = complementarity, Prop 2 = wedge. theory_main_text.md proposed Proposition 1 = complementarity, Lemma 2 = duality and Proposition 6 = wedge.

   | Main text | Appendix |
   |---|---|
   | Proposition 1 | Lemma A3 |
   | Lemma 1 | Lemma A2 + Corollary A1 |
   | Proposition 2 | Proposition A8, with Lemma A4(iii), Lemma A5 and Corollary A3 |
   | σ formula and σ(w) (text, not a numbered result) | Lemma A1 |

5. **The Hoffmann approach exponents** ("a between 0.46 and 0.50 across the three approaches", Section I.C) are the published values of Hoffmann et al. (2022, Table 2: 0.50 / 0.49 / 0.46). They are not in our CSVs. The m1 memo quotes 0.49 (0.462, 0.534) for Approach 2.
6. **The Llama 3 8B illustration is not the Section V estimate.** Sections I.E and Figure 1 use the Besiroglu *published* parameters and nominal N = 8e9, D = 15e12:
   - w = 5.22, T/D = 12.65, C/C_min = 5.51, CE = 0.18.

   Section V's reference-technology estimate (m3) is w = 5.27 [3.97, 8.12] and T/D = 12.8, using the Huber refit and the exact HF parameter count (M = 1,868). My text labels its number "under the Besiroglu et al. (2024) parameters", which is consistent with Figure 1. The intro should quote Section V's numbers.
7. **Evidence for σ < 1 when the form does not impose it.** The sentence in I.D reads: "The families we fit impose σ < 1, but on the best-designed sweep, estimates from forms that do not impose it also imply σ < 1 (Section IV)."
   - Source: m2 memo claim 1 and H1c (Farseer nonparametric local σ: median 0.690 [0.679, 0.699], pointwise range 0.61–0.99; Farseer Eq. 3 median 0.706); m2 review, "H1a σ < 1 holds under every form and in every sweep".
   - The technology writer should make sure Section IV states this.
8. **Table 1 layout.** It fills most of a page at `\footnotesize`, with at most two references per row. aea.bst prints three-author citations in full, which drove the height. If the integrator adds rows or references, it will overflow.
9. **Figure 1 is placed in I.D.** Its panel B (three observationally equivalent technologies) is also natural for Section II to cite (`fig:geometry`).

## Numbers used (value → source)

**Besiroglu et al. (2024) published parameters** (α = 0.3478, β = 0.3658). Sources: `output/tables/m7_theory_ledger_checks.csv` and SYNTHESIS §2.1; recomputed.
- α = 0.348, β = 0.366 (rounded).
- σ* = 0.737 (ledger 0.73703).
- σ range [0.732, 0.742] = [1/(1+β), 1/(1+α)] (computed: 0.7322, 0.7419).
- γ = 0.178 (0.17829); 1/γ = 5.6 (5.609).
- Llama-3-8B-configuration run (N = 8e9, D = 15e12):
  - M = 1,875;
  - w = 5.2 (5.2175);
  - T/D = 12.7 (12.65);
  - T ≈ 1.9e14 (12.65 × 1.5e13);
  - C/C_min = 5.5 (5.513);
  - Farrell CE = 0.18 (1/5.513).
- Gopher: w = 0.36 (0.3622), C/C_min = 2.0 (2.012).
- GPT-3: w = 0.43 (0.4265).

**Other numbers**
- **Hoffmann (TeX-precision) σ\* = 0.762** (ledger 0.7622). It equals Hao & Merrill's averaged-exponent value 0.7622; recomputed.
- **Kaplan joint law:** σ ∈ [0.500, 0.575], σ* = 0.535, outer exponent κ ≈ 0.103, inner exponents 0.738 and 1, E = 0. Source: appendix_proofs.tex Lemma A1(v) and Definition A2.
- **Harberger approximation:** within 1% for |ln w| ≤ 0.7, and overstates by 23% at w = 5.2. Sources: m7 memo §5, `m7_theory_harberger.csv` (exact 5.518 vs approximation 6.776 at w = 5.22, ratio 1.228). Recomputed: 0.5–0.9% at ln w = ±0.7.
- **Figure 1 content:** panel B uses σ* = 0.60, 0.74, 0.85 at C = 1e22, and panel A uses isoquants at the frontier losses for C = 1e20…1e24. Source: `code/analysis/m7_theory/figures.py::fig_geometry`.
- **173 open-weight models** (Section V sample). Source: paper_plan.md and m3 memo (Sample B).

## Claims made, and the caveats kept
- **Duality and the Chinchilla approaches** (cost function, factor demand, primal). The interpretation is ours; the algebra is Hoffmann et al.'s. The text says Section IV tests the gaps.
- **The σ formula is credited to hao2026theory.** Our additions: σ(w), σ*, ordinality, the Kaplan contrast, and the observation that their averaged value equals σ*.
- **Lemma 1** is stated as in appendix Lemma A2 and Corollary A1: strict SOC ⇔ 0 < σ < 1; σ > 1 or σ < 0 ⇒ output minimum on the isocost; σ > 1 everywhere ⇒ corner. It is a local statement at the optimum. The text notes that the fitted families impose σ < 1.
- **Proposition 2.**
  - (i) The wedge formula and its invariances (to ω, ε, E and κ).
  - (ii) The sufficient statistic, exact in the generalized κ family only. It is not proven for Farseer-type non-homothetic forms (m7 open issue 5), which is why the statement says "generalized family".
  - (iii) The Farrell ratio, written with exponent (α+β)/(αβ) so that it holds for any κ; this equals 1/γ at κ = 1.
  - (iv) Data and memory constraints and factor-bias contamination.
- **"Not a markup"** is stated explicitly. The DLW and Hsieh–Klenow analogies are drawn with the difference spelled out (shadow cost of a second use of the same input), and the Raval/Demirer contamination analog is noted.
- **Wedges below one** are attributed to data constraints, data-augmenting productivity, or belief error (Kaplan), following appendix Proposition A8(v).
- **Part (iii) of Proposition 1 holds at given compute.** The loss-target contrast is from Lemma A3(v).
- **No "Do not claim" item is violated.** The text does not say "first", does not call w a markup, does not say "σ more stable than a", and does not say the sign of the bias cannot be identified. The Table 1 row for algorithmic progress says "Drift of M* signs the bias; its size and σ* are not identified", which is the corrected DMR statement.

## Placeholders
None. The section has no m9-dependent content.
