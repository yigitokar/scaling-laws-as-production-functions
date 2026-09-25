# Memo: module rb3_econ2, integrating Section V (economics) with Section IV (revealed wedges)

Module owner: rb3_econ2 (Claude). Date: 2026-09-24. Entry point: `code/analysis/rb3_econ2/run.py`.

**How to run it.** `.venv/bin/python code/analysis/rb3_econ2/run.py`
- One command regenerates every number, table and figure below.
- CPU only, one process, about 5 seconds.
- Deterministic. A second run into a scratch root (`RB3_OUTPUT_ROOT`) reproduced all 33 table files byte for byte (after the independent review; the figure now also follows `RB3_OUTPUT_ROOT`).
- Independent review: `output/memos/rb3_econ2_review.md` (fixes applied there are reflected below).
- The run stops if either check file fails:
  - `rb3_econ2_wedge_checks.csv`: rb2's by-year aggregates are reproduced exactly;
  - `rb3_econ2_wall_checks.csv`: 13 numerical checks of the wall solver.

**Inputs (all read-only).**
- ra3_econ's `growth.py`, `demand.py` and `wall.py` are imported. ra3's output root is redirected to a sandbox, so no ra3 file is written.
- rb2_decisions' reviewed clean sample and decision units: `data/processed/rb2_decisions/clean_models.csv` and `decision_units.csv`.
- ra2's 32 ex-ante technologies.
- ra1's per-budget IsoFLOP minima.
- m1/m2 registries.
- Epoch's database, in two vintages (see H4).
- `sl.py` is not used or edited.

**Scope.**
- Binding plan: `paper/notes/revision_plan_v3.md` §2, "Section V integrates Section IV".
- Requests covered:
  - R3 round 2: N6 (1)–(3) and minors 17–20;
  - R1 round 2: minors 17–18;
  - R2 round 2: minors 14 and 20;
  - R4 round 2: minors 9–10.

**Notation.**
- w = ε_N/ε_D; m_N = w − 1; s = (w − 1)/w; k = 1/σ\* − 1.
- e(σ\*) = σ\*/[2(1 − σ\*)], so that at given compute D/D\*(C) = w^e.
- D is tokens processed and U is unique tokens.
- r = D\*(C)/U is scarcity at compute-optimal use; D/U is scarcity at the observed allocation.
- Reference technology: Chinchilla refit with κ free (σ\* = 0.7006, a = 0.504, γ = 0.165).

**Labels.**
- [E]: estimated here.
- [U]: from an upstream module.
- [L]: literature or a posted price, checked against a saved primary copy.

---------------------------------------------------------------------------------------------------

## 1. Headline findings

### H1. The revealed wedge rose about 1.9-fold a year over 2023–2025, and the frontier data multiple with it [E/U]
File: `rb3_econ2_wedge_trend.csv`. The sample is rb2's clean sample: 77 models in 56 decision units. The trend is an OLS regression of ln w (reference technology) on release date. The 95% interval is a wild cluster bootstrap-t by developer (18 clusters, Webb weights, B = 9,999).

| Estimator | n | w per year [95% CI] | D/D\* per year from M/M\* (σ\*-free) | value-based at σ\* = 0.60; 0.74 |
|---|---|---|---|---|
| **Decision units, OLS (primary)** | 56 | **1.91 [1.03, 3.48]** | **2.13 [1.04, 4.30]** | 1.62; 2.51 |
| Models, OLS | 77 | 2.00 [0.96, 4.33] | 2.24 | 1.68; 2.67 |
| Decision units, compute-weighted | 56 | 2.22 [0.40, 12.8] | 2.54 | 1.82; 3.10 |
| Models, compute-weighted | 77 | 1.96 [1.01, 3.85] | 2.20 | 1.66; 2.60 |
| Units above 10²⁴ FLOP (6 developers; indicative) | 11 | 1.80 [0.85, 3.93] | 1.99 | 1.56; 2.31 |
| Compute-weighted aggregate w = 1/(1 − s), units | 56 | 2.01 (1.34 → 2.18 → 5.39) | 2.26 | 1.69; 2.69 |
| Median wedge, units | 56 | 1.78 (1.97 → 5.23 → 6.26) | 1.97 | 1.54; 2.28 |

- **Leave one developer out** (units, OLS): 1.73–2.30. Without Alibaba it is 1.86; without Meta, 1.77.
- **The 32 ex-ante technologies.** The data-multiple growth is 1.93–2.33 per year, while the revealed-value growth is 1.16–4.12 per year (`_wedge_trend_by_tech.csv`).
  - This is the mechanism R3 asked about. The data trend is a trend in tokens per parameter. It needs no curvature.
  - The *value* that trend reveals scales with k = 1/σ\* − 1.
- **R3's own calculation is reproduced.** On the clean decision units the aggregate w goes from 1.34 (2023) to 5.39 (2025). At σ\* = 0.70 that is 2.26-fold a year in D/D\*; R3 found 2.1 on the universe.

### H2. If frontier over-training follows that trend, frontier data demand grows 3.6–5.6-fold a year instead of 2.2 [E]
Files: `rb3_econ2_scenarios.csv`, main table Panel B.
- **Anchor.** The frontier's observed data multiple is D/D\* = 1.25 under the reference technology, i.e. w_F = 1.21. It is the median over the four disclosed dense runs above 10²⁵ FLOP, whose D/D\* are Llama 3.1 405B 1.35, Nemotron-4 1.12, Pangu Ultra 2.15 and Aramco 1.15 (wedges 1.29, 1.10, 1.92, 1.13). The frontier is close to the path.
- **Scenario.** From September 2026 the frontier's value of compactness grows at 1.91 a year, and D/D\* = w^e(σ\*).
  - At σ\* = 0.70 this is the same as extrapolating M/M\*, so it needs no curvature.
  - σ\* enters only when the value of compactness is the primitive being projected, e.g. from serving demand. This answers R3 N5 and N6(1) on "why σ\*".

| Frontier scenario (reference path, a = 0.50) | σ\* | D/D\* per year | D per year, C at 4.2×; 5.06× | 5-year multiple at 4.2×; 5.06× | Frontier reaches 100T (5.06×) |
|---|---|---|---|---|---|
| Wedge held at 1.21 (Section V.A) | – | 1.00 | 2.04; 2.24 | 35; 56 | 2027.4 |
| Wedge rises 1.91× a year | 0.60 | 1.62 [1.02, 2.55] | 3.31; 3.63 | 399; 634 | 2027.1 |
| | 0.70 | 2.13 [1.04, 4.29] | 4.34; **4.76** | 1,536; 2,441 | 2027.0 |
| | 0.74 | 2.51 [1.05, 5.90] | 5.12; 5.62 | 3,522; 5,597 | 2027.0 |
| Rise stops at the 2025 aggregate wedge (5.39) | 0.60 | ×3.1 once | – | 108; 172 | 2027.1 |
| | 0.70 | ×5.7 once | – | 201; **320** | 2027.0 |
| | 0.74 | ×8.4 once | – | 295; 469 | 2027.0 |

- **Scale.** The wedge trend adds as much to data-demand growth as compute growth does: a factor of 2.1 a year at σ\* = 0.70 against 2.0–2.2 from compute.
- **The bounded "catch-up" version is the plausible one.** The frontier's wedge reaches the 2025 open-weight aggregate after 2.3 years and stays there. Five-year frontier data demand is then 3.1–8.4 times the V.A forecast (σ\* 0.60 to 0.74).
  - *The target rests on one family* (review addition; `rb3_econ2_scenarios_catchup_sensitivity.csv`). The 2025 aggregate (5.39) comes from seven decision units, and Alibaba's Qwen3 family, a cap-reading lower bound, carries most of their compute. Without it the 2025 aggregate is 3.13, the one-off multiple is 2.0/3.0/3.9 (σ\* 0.60/0.70/0.74) and five-year demand at 5.06× is 114/170/216 times today's (not 172/320/469). Leaving out each 2025 developer in turn, the σ\* = 0.70 five-year multiple runs from 170 to 451.
- **Panel B is notional demand.** It is the data the frontier would process without the wall. Panel C shows what the wall does to it: in 2029, on the rising path, the capped frontier processes 15.1 epochs of the stock rather than 19.2 (R\*_D = 15.4).
- **Uncertainty.**
  - The 95% interval for g_w runs from 1.03 (no rise) to 3.48.
  - Combining the a-range (0.36–0.57) with σ\* (0.60–0.74), per-year data-demand growth in the trend scenario spans 3.3–7.1 at 5.06× compute.
- **Dates barely move.** Anchored at observed data use, frontier runs reach the 100T stock in 2027.4 with the wedge held and 2027.0–2027.1 with it rising. The effective 320T stock moves from 2028.9 to 2027.7–2028.0.
  - Sensitivity: if the wedge trend began at the anchor runs' median release date (2024.5) rather than today, frontier D/D\* would already be 3.7–9.7 today. The 100T stock would then have been reached in 2025.8–2026.2 (`_scenarios_start_sensitivity.csv`). We start the rise today, the conservative choice.

### H3. The data wall at observed allocations: zero today, small by 2028, 5–10% of lifetime cost by 2029 if repeated data decay slowly (23–48% under the data-only fit), and measured wedges start to understate [E]
Files: `rb3_econ2_frontier_wall.csv`, `_wall_observed_grid.csv`, `_wall_understatement.csv`, `_wall_today.csv`; main table Panel C; appendix table `app_wall`.

**Model.** The developer minimizes lifetime cost, 6ND + ΦN, with a value of compactness proportional to N, at the loss of its uncapped choice.
- Without a cap, the choice has ε_N/ε_D = w and Φ = 6D(w − 1).
- The technology is ra3's κ-family wall technology combined with Muennighoff et al.'s repetition model.
- Checks:
  - with w = 1 the model reproduces ra3's wall exactly;
  - the uncapped optimum equals the closed form (N\*w^−e, D\*w^e), to within 4×10⁻⁷;
  - scale invariance holds at 10²⁵ vs 10²⁷ FLOP (difference ≤ 1.4×10⁻⁹);
  - Prop. 1(iii)'s μ equals 1/η − 1 (η = d ln D′/d ln D) under soft repetition, to within 5×10⁻⁷ (13 checks in all).

**(a) The wall binds earlier for over-trained models.** Take extra lifetime cost at compute-optimal scarcity r (same compute, same U), with data decay only:

| w | r = 0.71 | r = 1.19 | r = 2 | r = 4 |
|---|---|---|---|---|
| 1 | 0 | 0.1% | 1.6% | 7.4% |
| 2 | 0.3% | 1.6% | 4.4% | 11.9% |
| 5.39 | 1.9% | 3.9% | 7.3% | 16.1% |

(Grid points r = 2^(k/4); r = 0.71 and 1.19 are 2^(−1/2) and 2^(1/4).)

With R\*_D = 2.9 the same rows are 0/7.0/39.8% (w = 1, r = 1/2/4) against 13.8/33.1/75.7% (w = 5.39).
- An over-trained model processes w^e times D\* and faces the wall at w^e-times-lower compute.
- The proportional cost per unit of D/U is lower, because such a developer can give up some compactness. Its training compute falls while N rises.
- At very high r the lines cross (w = 5.39 against w = 1 between r = 19 and 23): compact models absorb extreme scarcity more cheaply in lifetime terms.

**(b) Today's frontier at observed data use.**
- In September 2026, frontier runs at observed data multiples process 0.49–1.09 times the 100T stock (0.62 under the reference). The wall costs nothing there, and below 0.1% for Farseer's own form.
- At the stock's lower 95% bound (22T), D/U is 2.2–4.9 and the extra lifetime cost is 1.8–8.6% (data decay only) or 3.3–32% (full model).

**(c) The projected frontier (reference technology, 100T stock, 5.06×).**

| Year | Wedge held: D/U; extra lifetime cost R\*_D = 15.4 / 2.9 / full | Wedge rises: w; D/U; extra cost 15.4 / 2.9 / full |
|---|---|---|
| 2028 | 1.6; 0.6% / 2.7% / 0.7% | 2.76; 4.2; 2.8% / 13.2% / 2.8% |
| 2029 | 3.5; 4.6% / 23% / 13% | 5.27; 19.2; 10.5% / 48% / 10.5% |
| 2030 | 7.4; 16% / 107% / 85% | 10.1; 87; 25% / 96% / 25% |

- In dollars (amortized, $10⁻¹⁸ per FLOP), the 2029 extra lifetime cost is about $2 billion with the wedge held and $21 billion with it rising (data decay only).
- The shadow value of a unique token at the frontier is $1.9 per million tokens (2028, wedge held) and $31 (2029). With the wedge rising it is $8 and $187.
- Under 4.2× compute growth everything arrives later: with the wedge held, D/U = 2.8 and the cost is 3.0% in 2029.

**(d) Measured wedges understate the value of compactness as the wall binds (Prop. 1(iii)).** The ratio ŵ/(1 + m_N) is the wedge an econometrician computes from the capped choice, treating tokens processed as fresh, relative to one plus the value of compactness at that choice. It depends on D/U and hardly on w. Here D/U is the tokens the developer would process without the wall over the stock (w = 5.39; exact grid points after the review):

| D/U | 1.5 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Hard cap (Prop. 1(iii)) | 0.58 | 0.38 | 0.14 | 0.04 | 0.01 |
| R\*_D = 2.9 | 0.90 | 0.83 | 0.67 | 0.49 | 0.31 |
| R\*_D = 15.4 | 0.98 | 0.96 | 0.92 | 0.85 | 0.74 |

**Two ratios, two questions (review addition).** 1 + m_N = 1 + Φ/(6D) at the capped choice exceeds the no-wall wedge w, because the capped developer processes fewer tokens. Part of the understatement above is therefore the rise of 1 + m_N, not a fall in the measured wedge. For the *trend* prediction the right object is ŵ/w, the measured wedge relative to the wedge the same developer (same Φ) would reveal without the wall:

| D/U | 1.5 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Hard cap | 0.81 | 0.69 | 0.47 | 0.30 | 0.18 |
| R\*_D = 2.9 | 0.96 | 0.93 | 0.85 | 0.74 | 0.60 |
| R\*_D = 15.4 | 0.98 | 0.98 | 0.96 | 0.93 | 0.87 |

- On the rising-wedge path the measured frontier wedge understates 1 + m_N by 9% in 2028 and 30% in 2029 under the most favourable repetition model. Under R\*_D = 2.9 the figures are 34% and 73%.
- Relative to its no-wall path, the measured frontier wedge falls short by 5% (2028) and 15% (2029) under R\*_D = 15.4, and by 20% and 44% under R\*_D = 2.9. The measured wedge still rises (2.62 → 4.48 → 6.44 in 2028–2030 against 2.76 → 5.27 → 10.07 without the wall, R\*_D = 15.4).
- Under R\*_D = 2.9 it can fall outright: on the catch-up path, with the value of compactness constant after 2029, the measured wedge drops from 2.94 (2029) to 2.09 (2030). With the wedge held, the measured frontier wedge falls below 1 by 2029 (0.91): the frontier would look under-trained.
- **Prediction for the trend.** As frontier and open-weight releases approach the stock, the measured wedge trend should flatten even if the value of compactness keeps rising. A flattening in 2026–2027 releases is therefore not evidence of a falling value of compactness.
- **Exception: Muennighoff's full model.** Excess parameters also decay there, which acts as a size penalty (like ν in Prop. 1(iii)). It *raises* the measured wedge once N exceeds the compute-optimal size for U: 1.06 in 2029 with the wedge held. For over-trained models the full model coincides with data decay only.

### H4. The 5.06× vs 4.2× gap is a database-vintage effect; our estimator reproduces Epoch's 4.2× on the May 2024 database [E/L]
Files: `rb3_econ2_compute_vintage*.csv`; appendix table `app_growth`, Panel A.

The estimator is ra3's, reproduced exactly: running top-10 at release, OLS of log₁₀ C on date for January 2018 to May 2024, wild cluster bootstrap-t by developer. It is applied to two copies of Epoch's public file:
- **May 2024.** `all_systems.csv` as captured by the Internet Archive on **31 May 2024**, three days after Sevilla and Roldán's post (sha256 in `code/data/download_rb3_econ2.sh`).
  - Result: **4.13× [3.75, 4.55]**, 58 runs, 22 developers.
  - Epoch's published figure is 4.2× (90% CI 3.6–4.9). A test of 4.2× gives p = 0.73.
- **September 2026.** The 23 September 2026 snapshot, same window.
  - Result: **5.23× [4.02, 6.80]**, 78 runs.
  - This reproduces ra3's 5.23 exactly; 4.2× is rejected at p = 0.038.

**Decomposition** (two-factor Shapley on ln growth):
- **Compute revisions: 64%.** 145 models revised by more than 5%, 54 given an estimate, 21 estimates withdrawn. Examples: Mistral Large 2×10²⁵ and GPT-3.5 (text-davinci-003) lose their estimates; FLAN 137B is revised from 4.9×10²² to 2.0×10²⁴.
- **Backfilled entries: 36%.** 978 entries released before June 2024 but added later, e.g. GPT-2 124M–774M, GPT-3 13B, GPT-4 (Jun 2023), Aramco Metabrain.
- One case straddles the two: the 2.6×10²⁴ FLOP estimate of "GPT-3.5 (text-davinci-003)" (Nov 2022) now sits on a separate entry, "GPT-3.5 (davinci-002)" (Mar 2022). The decomposition counts it as a withdrawal plus an addition.
- *Verified by the reviewer:* the saved file's SHA-1 (base32) equals the Internet Archive's CDX digest for the 2024-05-31 03:46:17 capture; its latest "Last modified" stamp is 2024-05-29. An independent reimplementation gives 4.13 (58 runs), 5.23 (78) and 5.06 (92). Epoch's page states "4.2 x/year (90% CI: 3.6x to 4.9x) after 2018" for models in the top 10 at release (28 May 2024).
- **Notable models only:** 4.16 (2024 file) and 4.95 (2026 file).

**Reading and data-demand growth.**
- The paper's footnote guess ("probably a vintage effect") is confirmed and quantified. Forecasts should use the current database (5.06×, 2018–2026) and report 4.2× beside it.
- Data-demand growth at a = 0.36–0.57 is 1.87–2.52× a year at 4.2 (23–101× over five years), against 2.02–2.84× at 5.06 (34–183×).
- Reference (a = 0.50): 2.04 vs 2.24 per year, 35× vs 56× over five years.
- R4 minor 9's IsoFLOP model-free paths (a = 0.35–0.50; Porian et al.'s unannealed profiles excluded, as in Section III) give 2.04–2.53× at 4.2 and 2.24–2.86× at 5.06. These are now stated in the Table 3 notes, together with OLMo's bootstrap interval for a ([0.11, 0.82]) and the fact that Farseer's local slope lies beyond its design.

### H5. γ for growth calibrations at 10²⁵–10²⁷ FLOP: use the κ-free Chinchilla joint fit, γ = 0.165, i.e. 66× compute per halving of reducible loss [E/U]
Files: `rb3_econ2_gamma*.csv`; appendix table `app_gamma`, Panels A–B.

| Technology | E from | γ (s.e.) | 2^(1/γ) [95% CI] | Years at 4.2×; 5.06× |
|---|---|---|---|---|
| **Chinchilla, κ free (reference)** | joint fit | **0.165 (0.006)** | **66 [50, 90]** | 2.9; 2.6 |
| Chinchilla, κ = 1 (rejected, §III) | joint fit | 0.178 (0.006) | 49 [39, 63] | 2.7; 2.4 |
| Chinchilla, IsoFLOP frontier | frontier | 0.168 (0.007) | 62 [44, 91] | 2.9; 2.5 |
| Llama 3, Meta's frontier (A1) | frontier | 0.136 (0.006) | 161 [108, 259] | 3.5; 3.1 |
| Llama 3, joint on IsoFLOP runs, κ = 1 | joint | 0.149 (0.005) | 104 [77, 146] | 3.2; 2.9 |
| Farseer, κ = 1 | joint | 0.148 (0.009) | 109 [66, 209] | 3.3; 2.9 |
| Farseer, κ free (E = 0.27) | joint | 0.096 (0.003) | 1,351 | 5.0; 4.4 |

Why this design, in four points:
1. **γ is cardinal in reducible loss, so it needs E.** Only Chinchilla has known loss units (nats per token on MassiveText) and an E identified by off-path runs in a joint fit that the data do not reject (κ free).
2. **Two identification routes agree only in this design.** The E-profile of Chinchilla's nine model-free IsoFLOP minima gives a 95% set E ∈ [1.68, 1.80]. That set contains the joint-fit E = 1.757, and γ over it is [0.156, 0.179]. The two routes use the same 240 runs but different variation (the frontier's curvature versus off-path runs), so they are separate checks, not independent samples.
   - Llama 3's frontier gives E ∈ [0.52, 0.59] in unstated units and γ ∈ [0.111, 0.143].
   - Marin's two-decade ladders cannot separate E from γ: 0.08–0.12, and 0.05–0.16 for Comma.
   - Farseer's κ-free fit puts E at 0.27, the E–γ ridge; its γ of 0.096 is not a frontier elasticity anyone should use.
3. **No drift in γ with compute inside Chinchilla, unlike σ\*.** At the joint E, local γ is 0.169 below 10²⁰ FLOP and 0.174 above; the drift of successive-pair slopes is −0.0006 per decade.
4. **R4 minor 10.** κ = 1 is rejected, so lead with κ free: 66×, not 49×.

Conversions and caveats:
- **Guidance.** Central value 0.165. Report the range 0.14–0.18 (49–161× per halving; 2.4–3.1 years at 5.06×, 2.7–3.5 at 4.2×), with Meta's 0.136 as the low end.
- **Extrapolation.** Applying γ at 10²⁵–10²⁷ extrapolates 3–5 decades beyond the design (6×10¹⁸–1.3×10²²).
- **Per percent.** A 1% cut in reducible loss needs 6.0% more compute (1/γ).
- **Total loss.** In Chinchilla's units, reducible loss is only 4.4% of total loss at 10²⁶ FLOP. The total-loss elasticity is 0.007, i.e. γ(L\* − E)/L\*. A calibration that maps *total* loss or capability into output must say which loss it uses.

### H6. The fleet comparison with 1-, 2- and 3-year service lives [E]
Files: `rb3_econ2_fleet.csv`; appendix table `app_gamma`, Panel C.
- The 2025 compute-weighted wedge of the decision units implies m = w − 1 = 4.39 (models: 4.85).
- Planned serving is spread over a service life L, and training compute grows 5.06× a year, so φ = (1 − e^(−gL))/(gL) is 0.49, 0.30 and 0.20 for L = 1, 2, 3.

| Service life | R&D multiple ρ = 1 | ρ = 4.4 | ρ = 10.4 |
|---|---|---|---|
| 1 year | 0.68 | 0.33 | 0.17 |
| 2 years | 0.57 | 0.23 | 0.11 |
| 3 years | 0.47 | 0.17 | 0.08 |

- **Longer lives lower the fleet share.** The models in service are older vintages, trained when compute was smaller.
- At 4.2× compute growth, 1-year shares are 0.18–0.35. With inference FLOPs at twice the price (p = 2), 1-year shares at 5.06× are 0.29–0.50.
- *m rests on Qwen3* (review addition). Without Alibaba the 2025 aggregate gives m = 2.13, and 1-year shares at ρ = 4.4–10.4 fall to 0.09–0.19 (5.06×, p = 1); see the `2025, decision units excl. Alibaba` rows of `rb3_econ2_fleet.csv`.
- **Text for the paper.** State that the 0.17–0.33 range assumes a one-year service life; it falls to 0.11–0.23 at two years and 0.08–0.17 at three.
- R&D and service lives both push fleet shares below the disclosed 0.4–0.7. The comparison is a loose check.

### H7. The shadow value of a unique token beside the cost of a synthetic token (optional; R3 minor 18) [E/L]
Files: `rb3_econ2_synthetic_*.csv`; appendix table `app_gamma`, Panel D.
- **The shadow value itself** (reference technology, compute-optimal, data decay only, $1.0×10⁻¹⁸ per FLOP):

  | Compute | r = 2 | r = 4 | r = 8 |
  |---|---|---|---|
  | 10²⁶ FLOP | $0.5/M | $2.8/M | $14/M |
  | Frontier in September 2026 | $1.6/M | $8.8/M | $45/M |

  Cloud prices are 3.7× higher.
- **Posted API prices for generated tokens** (DeepSeek, retrieved 2026-09-24 17:15 UTC, saved copy): $0.60–$1.20 per million output tokens for V4.1-Flash and $1.98–$3.96 for V4-Pro.
  - At the 2026 frontier the shadow value exceeds these prices once r > 1.5–2.9; at 10²⁶ FLOP, once r > 2.1–4.6.
- **Self-generation** by a model of the trainee's size at p = 1–2 costs p/3 of the processing cost of a token. That equals the shadow value at r = 3.5–4.8; a generator one-tenth the size breaks even at r = 1.8.
- **Reading.** Synthetic data become economic at moderate scarcity, r ≈ 2–5, if a synthetic token were worth a fresh one. It is not, so these break-even scarcities are lower bounds.
- The API price is one provider's posted price on one date, at the low end for a large model. The reviewer verified it against the saved page and the live page (24 September 2026). It is not a market-wide cost of synthetic data.
- **Licensing prices were not converted per token, because the conversion is not verifiable.** Reddit's S-1 reports $203.0 million of licensing contracts over 2–3 years and a corpus of "over one billion posts and over 16 billion comments". That is about $0.012 per post or comment across all licensees. Tokens per item are not disclosed, so we do not convert.

---------------------------------------------------------------------------------------------------

## 2. Methods

- **Trend (trend.py).**
  - ln w_ref regressed on decimal release date, on rb2's decision units (family W for common-D families) and models, OLS and WLS (compute weights).
  - Inference: ra3's `growth.wild_cluster` (CR1; WCU percentile-t; Webb; B = 9,999; clusters = developers).
  - Endpoint growth of the compute-weighted aggregate w = 1/(1 − s_agg) and of the median.
  - Leave one developer out.
  - M-based growth under each ra2 ex-ante technology: ln(M/M\*_j) = ln w_j/k_j, on ra2's clean rows. The StableLM-Alpha token audit of rb2 is not applied here (2 of 77 models).
- **Scenarios (scenario.py).**
  - D_F(t) = m_D(t) D\*(C_F(t)).
  - C_F follows ra3's fitted top-10 trend to today and grows at 4.2× or 5.06× afterwards.
  - m_D0 is the median D/D\* of ra3's four dense anchor runs (reference path).
  - Trend: m_D = m_D0 g_w^(e(σ\*)(t − now)). Catch-up: the w ratio is capped at 5.39/1.21.
  - Year of reaching a stock: root of m_D(t) D\*(C_F(t)) = S₀ 1.05^(t − 2024).
  - Panel A technologies use ra3's path objects unchanged. The check chin_q reproduces ra3's 2027.36 and 56×.
- **Wall at observed allocations (wallobs.py).**
  - A vectorized grid (4,001 points over 30 log-units of N) plus bounded Brent.
  - The corner is evaluated exactly under a hard cap.
  - The shadow value is a central difference in ln U at fixed loss.
  - The measured wedge is ε_N/ε_D of the κ family at (N_c, D_c processed); it is also computed at effective data.
  - Curvature variants are members of the κ-free reference's observationally equivalent family (same path and frontier).
  - Repetition specifications:
    - D15, data decay only with R\*_D = 15.4 (ra3's primary; the most favourable case);
    - D3, their data-only fit with R\*_D = 2.9;
    - DN, their full model, in which excess parameters also decay;
    - hard cap, no useful repetition.
- **Vintage (vintage.py).**
  - Matching: by normalized name (Unicode kept), then by (date, first organization) for renamed entries (38 of them, e.g. GPT-4 → "GPT-4 (Mar 2023)").
  - Counterfactual databases: 2024 file + 2026 compute; 2024 file + later entries.
  - Two-factor Shapley decomposition.
- **γ (gamma.py).**
  - E-profile: for E on a grid of 1,201 values, NLS in levels of the per-budget minima on (ln K, γ).
  - 95% set: SSR ≤ SSR_min [1 + F(0.95; 1, n − 3)/(n − 3)].
  - Local γ: at the joint-fit E, by halves of the budget range, and the drift of successive-pair slopes per decade.
- **Fleet (fleet.py).** s_fleet = p m φ/(p m φ + ρ), as in ra3.
- **Synthetic (synth.py).** ra3's `wall_point`; break-even r by log-linear interpolation and Brent.

---------------------------------------------------------------------------------------------------

## 3. Inventory

**Code** (`code/analysis/rb3_econ2/`): `run.py`, `rb3common.py`, `trend.py`, `scenario.py`, `wallobs.py`, `vintage.py`, `fleet.py`, `gamma.py`, `synth.py`, `exhibits.py`. The download and verification record is `code/data/download_rb3_econ2.sh`.

**Paper exhibits.**

| File | Content |
|---|---|
| `output/tables/rb3_econ2_table.tex` | **Revised Table 3** (label `tab:econ`), "Economic Implications: Data Demand, the Data Wall and the Revealed Wedge". Panel A: compute-optimal data demand under six technologies, at 4.2× and 5.06×. Panel B: **the rising-wedge scenario rows** (σ\* 0.60/0.70/0.74; catch-up in parentheses). Panel C: the data wall at compute-optimal *and* observed allocations (R\*_D 15.4/2.9/full next to σ\*; shadow $/M; measured-wedge ratio). Compiled in AEA.cls at footnotesize: fits one page, no overfull boxes. |
| `output/figures/rb3_econ2_figure.pdf/.png` | **Revised economics figure (appendix)**, 2×2: (a) frontier data demand with rising wedges against the stocks (gray band: 22T–490T stock, growing 0–10% a year, as in ra3); (b) the wall binds earlier for over-trained models; (c) measured wedges understate as the wall binds (solid ŵ/(1 + m_N), dashed ŵ/w, added in review); (d) shadow value vs the API price of generated tokens. |
| `output/tables/rb3_econ2_app_growth.tex` | Appendix (`tab:app-econ-growth`): A, compute growth by database vintage and the decomposition; B, wedge-trend estimators; C, the wedge by year and the data multiple it implies. |
| `output/tables/rb3_econ2_app_wall.tex` | Appendix (`tab:app-econ-wall`): A, lifetime cost by w and r; B, ŵ/(1 + m_N) / ŵ/w by D/U and repetition model; C, today's frontier by technology and stock; D, compute-optimal r_max, shadow values, σ_CU and γ_eff/γ (moved from Table 3). |
| `output/tables/rb3_econ2_app_gamma.tex` | Appendix (`tab:app-econ-gamma`): A, γ candidates and compute per halving; B, E-profiles; C, fleet shares by service life, ρ and p; D, shadow value vs synthetic-token costs. |

**CSVs** (`output/tables/rb3_econ2_*`):
- `wedge_trend`, `wedge_trend_lodo`, `wedge_trend_by_tech`, `wedge_by_year`, `wedge_checks`;
- `demand`, `demand_all_technologies`, `scenarios`, `scenarios_start_sensitivity`, `scenarios_catchup_sensitivity` (review addition), `scenario_constants`, `frontier_anchor_runs`;
- `wall_compute_optimal`, `wall_main_rows`, `wall_observed_grid`, `wall_understatement`, `frontier_wall`, `wall_today`, `wall_checks`;
- `compute_vintage`, `compute_vintage_decomposition`, `compute_vintage_changes`, `compute_vintage_info`;
- `fleet`;
- `gamma`, `gamma_eprofiles`, `gamma_local`;
- `synthetic_shadow`, `synthetic_breakeven`.

**Processed data** (`data/processed/rb3_econ2/`): `summary.json`, `gamma_eprofile_grid.csv`, and `ra3_sandbox/` (empty directories created by importing ra3).

**Raw data** (`data/raw/rb3_econ2/`), not redistributed:
- `epoch_all_systems_20240531.csv` (used);
- `epoch_notable_ai_models_20240620.csv` and `epoch_large_scale_ai_models_20240619.csv` (reference only);
- `deepseek_pricing_20260924.html`.

**Bibliography** (`lit/bib/extra_round3_rb3_econ2.bib`, verified against the saved copies):
- `epochai2024dataarchive`: Epoch file via the Internet Archive, 31 May 2024;
- `deepseek2026pricing`: API pricing page, accessed 24 September 2026.

Existing keys used: epochai2026data, sevilla2024training, villalobos2022run, muennighoff2023scaling, denain2026final. The integrator must merge the two new keys into `paper/references.bib`; `tab:app-econ-gamma` cites deepseek2026pricing and (after the review) `tab:app-econ-growth` cites epochai2024dataarchive, epochai2026data and sevilla2024training (all four compile with no undefined citations against references.bib plus the extra bib).

---------------------------------------------------------------------------------------------------

## 4. Claims for the paper (evidence; caveat)

1. **Opening of Section V: where σ\* matters.**
   - *Claim:* "σ\* converts the value of compactness into data. At given compute, D/D\* = w^{σ\*/[2(1−σ\*)]}. If the value of compactness keeps rising at its 2023–2025 pace, frontier data demand grows 3.6-, 4.8- or 5.6-fold a year at σ\* = 0.60, 0.70 or 0.74, against 2.2-fold with the wedge held. Extrapolating tokens per parameter instead needs no curvature."
   - *Evidence:* H1–H2; Panel B.
   - *Caveat:* the trend is two years long. Its 95% interval runs from 1.03 to 3.48, and the 2024→2025 step of the medians rests on one or two developers (rb2 H6). The frontier itself is near the path (w = 1.21). The 5-year "trend" multiples (634–5,597×) are mechanical; quote the catch-up version (172–469×, i.e. 3.1–8.4 times the V.A forecast) as the plausible range. Its target, the 2025 aggregate wedge, depends on Qwen3: without Alibaba it is 3.1 and the catch-up range is 114–216× (170–451× at σ\* = 0.70 across leave-one-developer-out targets).
2. **The data wall at observed allocations.**
   - *Claim:* "At observed data use, today's frontier runs process 0.5–1.1 times the quality-adjusted stock, so the wall costs nothing yet. With the wedge held, the frontier processes 3.5 times the stock by 2029, and the wall costs 5% of lifetime cost if repeated data decay slowly and 23% under Muennighoff et al.'s data-only fit. With the wedge rising, the frontier processes 19 times the stock and the costs are 10% and 48%."
   - *Evidence:* H3(b)–(c); Panel C.
   - *Caveat:* the result is conditional on the repetition model (R\*_D from ≤ 9B-parameter runs) and on a value of compactness linear in N. The stock's 95% interval (22T–490T) moves the date by several years.
3. **The wall binds earlier for over-trained models.**
   - *Claim:* "At the same compute and stock, a model with the 2025 aggregate wedge processes seven times the compute-optimal data (at σ\* = 0.70). It pays 3.9% more lifetime cost where a compute-optimal run pays nothing, although its proportional cost per unit of scarcity is lower: it can give back some compactness."
   - *Evidence:* H3(a); appendix `app_wall` Panel A; figure (b).
4. **Prediction: measured wedges will understate.**
   - *Claim:* "By Proposition 1(iii) a binding token cap adds μ to the denominator of w. When a developer would process four times the unique stock, its measured wedge understates one plus the value of compactness at its capped choice by 8% if repeated data decay slowly, by a third under the data-only fit, and by 86% under a hard cap. Relative to the wedge it would reveal without the wall, the shortfall is 4%, 15% and 53%. As frontier and open-weight releases approach the stock, the measured trend should flatten even if the value of compactness keeps rising."
   - *Evidence:* H3(d); figure (c) (solid: ŵ/(1 + m_N); dashed: ŵ/w); appendix `app_wall` Panel B.
   - *Caveat:* under Muennighoff's full model, excess-parameter decay acts as a size penalty and can raise measured wedges of near-compute-optimal models.
   - *Wording:* do not describe the ŵ/(1 + m_N) figures as the fall in measured wedges. Part of them is the rise of 1 + m_N when the capped developer processes fewer tokens.
5. **σ\* and the wall.**
   - *Claim:* "σ\* barely moves the cost of a cap when repeated data decay slowly (7.7, 7.4 and 7.2% at r = 4 for σ\* = 0.60, 0.70, 0.74). It matters more under the data-only fit (51, 40 and 35%)."
   - *Evidence:* Panel C (`_wall_main_rows.csv`).
   - *Note:* this refines ra3's "σ\* barely matters", which was stated under the favourable repetition model only.
6. **Compute growth (footnote).**
   - *Claim:* "Epoch's published 4.2-fold rate for 2018 to May 2024 is reproduced by our estimator on Epoch's database as it stood on 31 May 2024 (4.1-fold). On the September 2026 database the same window gives 5.2-fold: two-thirds of the difference are later revisions of compute estimates, one third entries added after May 2024. We use the current database (5.06-fold for 2018–2026) and report 4.2-fold beside it: over five years, compute-optimal data demand grows 35-fold rather than 56-fold under the reference path."
   - *Evidence:* H4; Panel A.
7. **γ for growth models.**
   - *Claim:* "For growth calibrations at 10²⁵–10²⁷ FLOP use γ = 0.165 (95 percent CI 0.154–0.177): halving reducible loss takes 66 times the compute (50–90), 2.6 years of frontier compute growth at 5.06-fold or 2.9 at 4.2-fold. It is the one design whose irreducible loss is pinned down twice, by the joint fit and by the profile of its IsoFLOP minima, in stated loss units, and its local elasticity does not drift with compute. Other designs give 0.14–0.18 (49–161 times)."
   - *Evidence:* H5.
   - *Caveat:* extrapolation over 3–5 decades; loss is not capability. At 10²⁶ FLOP reducible loss is about 4 percent of total loss in these units.
8. **Fleet (one sentence).**
   - *Claim:* "Assuming each model is served for one year, the 2025 wedge implies fleet shares of 0.17–0.33 at realistic R&D multiples, falling to 0.11–0.23 at two years and 0.08–0.17 at three."
   - *Evidence:* H6.
   - *Caveat:* it reads the value of compactness as planned serving expenditure, which is one interpretation (plan v3 §1). The 2025 wedge rests largely on Qwen3: without Alibaba the one-year range is 0.09–0.19.
9. **Shadow value vs synthetic tokens (optional).**
   - *Claim:* "At today's frontier the shadow value of a unique token exceeds the posted API price of a generated token ($0.60–$3.96 per million) once compute-optimal data are 1.5–2.9 times the stock; generating with the trainee's own compute breaks even at 3.5–4.8 times."
   - *Evidence:* H7.
   - *Caveat:* synthetic tokens are imperfect substitutes, so these are lower bounds. The prices are those of 24 September 2026. Do not use a per-token licensing price.

---------------------------------------------------------------------------------------------------

## 5. Robustness

- **Trend.**
  - Estimator: g_w runs 1.78–2.22 (the table in H1).
  - Leave one developer out: 1.73–2.30.
  - Units above 10²⁴ FLOP: 1.80.
  - Under all 32 technologies the M-based D/D\* growth is 1.93–2.33, while the w-growth is 1.16–4.12. That contrast is the σ\* point.
- **Scenario anchor.**
  - With the wedge trend starting in mid-2024, frontier D/D\* today would be 3.7–9.7 and the 100T stock would already be reached (2025.8–2026.2). We start today.
  - Four anchor runs only; closed-frontier data use is not disclosed.
- **Wall.**
  - All objects depend on (w, r) only (checked).
  - The primary repetition model is the most favourable. The data-only fit multiplies costs by about 4–5 at moderate scarcity.
  - A hard cap gives ŵ/(1 + m_N) = 0.38 at D/U = 2 (ŵ/w = 0.69).
  - Stock uncertainty dominates dates: at 22T the wall already costs 3.1% today (reference); at 490T it costs nothing through 2029 and 0.4% in 2030.
- **Vintage.**
  - Name matching could misattribute a few renamed entries. Only 38 matches rely on (date, organization), and the frontier-relevant renames were inspected (`_compute_vintage_changes.csv`).
  - Notable-only samples tell the same story: 4.16 (2024) vs 4.95 (2026).
  - Epoch's own estimator (piecewise, with an estimated break) was not replicated exactly. A breakpoint search on the 2024 file gave 3.3–4.1 depending on AlphaGo's inclusion (exploratory, not in outputs).
- **γ.**
  - The Chinchilla frontier profile's E-set does not touch the grid boundary.
  - Llama 3's frontier profile uses ra1's 8 bracketed budgets (to 10²¹); Meta's A1 uses 10 budgets (to 10²²).
  - Marin's profiles are uninformative (two decades).
- **Fleet.** A service life of 1–3 years and p = 1–2 span 0.08–0.50 at ρ = 4.4–10.4.

---------------------------------------------------------------------------------------------------

## 6. Referee comments addressed

| Comment | Response |
|---|---|
| **R3 N6(1)**: forecasts ignore the rising wedge; add a scenario row to Table 3 | Table 3 Panel B, H2: rising-wedge rows at σ\* = 0.60/0.70/0.74, with a bounded catch-up variant and 4.2×/5.06× compute. R3's computation is reproduced (2.1–2.3× a year in D/D\*). The trend's σ\*-free reading (tokens per parameter) is stated beside the value-based reading, which is where σ\* enters. |
| **R3 N6(2)**: the wall at observed allocations; measured wedges understate as the wall approaches | H3, Table 3 Panel C, appendix `app_wall`, figure (b)–(c). The lifetime-cost wall at observed allocations is reported today, along the projected frontier, and by w and r. The understatement is quantified for four repetition models, with μ = 1/η − 1 verified. Both ŵ/(1 + m_N) (Prop. 1(iii)) and ŵ/w (the shortfall from the no-wall path, added in review) are reported. A testable prediction for 2026–27 releases is stated. |
| **R3 N6(3)**, **R3 minor 19**: fleet comparison; state the service life; give 2 and 3 years | H6, appendix `app_gamma` Panel C. One year (φ = 0.49): 0.17–0.33. Two years: 0.11–0.23. Three years: 0.08–0.17. Moved out of Table 3; one sentence in the text (Claim 8). |
| **R3 minor 17**: 4.2× vs 5.06×; report both or reconcile | Both are reported in Table 3 Panels A–B and the appendix. Reconciled (H4): our estimator gives 4.13 on Epoch's 31 May 2024 file. The gap is 64% compute revisions and 36% backfilled entries. |
| **R3 minor 18**: $3/M beside an observed price (optional) | H7: posted API prices for generated tokens ($0.60–3.96/M) and self-generation cost give break-even scarcities of 1.5–4.8. Licensing is not converted per token (not verifiable). |
| **R3 minor 20**: which γ for 10²⁵–10²⁷ and why | H5, appendix `app_gamma` Panels A–B, Claim 7: γ = 0.165 (66× per halving), with four reasons: E identified twice, known units, no drift, κ = 1 rejected. The σ\* half of minor 20 (condition σ\* on compute) belongs to rb1_sigmaC, not to this module. |
| **R3 N5** (part): say where σ\* matters | Claims 1 and 5: σ\* converts the value of compactness into data demand, and it matters for the wall under fast data decay. |
| **R1 minor 17**: Section V long; fleet to appendix | The fleet comparison is now in the appendix, with one main-text sentence. Table 3 drops the inference-share panel, whose trend is in Section IV / rb2. |
| **R1 minor 18**: R\*_D sensitivity next to σ\* | Table 3 Panel C shows R\*_D = 15.4, 2.9 and the full model side by side with σ\* = 0.60/0.70/0.74. |
| **R2 minor 14 / 20**; **R4 minor 10**: γ units, E-profile, lead with the reference | H5: E-profiles for five designs; guidance restricted to the design with known units; the κ-free reference leads (66×, not 49×). |
| **R4 minor 9**: the a-range and the top of the growth range | Panel A keeps Farseer's own form and OLMo. In the review the Table 3 notes were extended to flag both (Farseer's slope lies beyond its design; OLMo's a has a bootstrap interval of [0.11, 0.82]) and to give the IsoFLOP model-free path range (a = 0.35–0.50): 2.24–2.86× a year at 5.06 and 2.04–2.53× at 4.2 (`_demand_all_technologies.csv`). Before the review the notes flagged only Farseer. |
| **R2 Major 1** (σ\* at frontier scale), as it bears on Section V | σ\* = 0.60, the top-budget value, is carried through Panels B–C and the wall. |

---------------------------------------------------------------------------------------------------

## 7. Open issues

1. **Integration.**
   - Table 3 changed. The old Panel C (aggregate inference share, 2019–2025) is dropped: R3 N2 and R4 R2-M1 drop the pre-2023 baseline, and rb2's trend table carries 2023–2025. Section V.C's text must be rewritten around the fleet sentence (Claim 8).
   - The compute-optimal wall numbers in the old Panel B (7.2/7.4/7.7%, r_max 117/83/47) came from κ = 1 equivalents. The new table uses κ-free equivalents: 7.7/7.4/7.2%, r_max 47/83/119. The appendix keeps r_max, σ_CU and γ_eff.
   - The γ numbers in V.D should switch to the reference: 66×, with 49–161× as the range.
   - Merge the two new bib keys into `paper/references.bib`.
2. **m9 (our experiment).** It does not enter Section V. If m9's σ\* differs from 0.70, the σ\* rows already span 0.60–0.74.
3. **Frontier anchor.** Four disclosed dense runs anchor the level. Closed-frontier D is unknown, so the scenario starts the rise today, which may understate current over-training.
4. **Repetition at frontier scale.** R\*_D comes from ≤ 9B models. The wall's costs and the understatement prediction are conditional on it.
5. **Epoch's piecewise estimator.** Not replicated. Our fixed-2018 OLS on the May 2024 file gives 4.13 against their 4.2, so the replication gap is small.
6. **Licensing prices.** No verifiable per-token data-licensing price was found. Reddit's S-1 gives contract value and item counts, not tokens.
7. **Authorship caveat (carried from ra3).** No Anthropic-specific figures are used or should be introduced here.
