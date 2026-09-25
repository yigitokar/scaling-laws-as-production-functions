# Memo: round-3 re-run of module rb3_econ2 (Section V, Table 3, Online Appendix F)

Work package WP5 of `paper/notes/round3_fixlist.md` (items E1-E6). Date: 2026-09-25. Entry point unchanged:
`.venv/bin/python code/analysis/rb3_econ2/run.py` (CPU, one process, about 15 seconds, deterministic: a second run into a
scratch root reproduced all 36 output tables and `summary.json` byte for byte). The version-3 memo
`output/memos/rb3_econ2.md` describes the module; this memo records what changed in round 3 and why. Log with every
number and its source: `paper/notes/round3_WP5_log.md`.

## 1. What changed in the inputs, and why

| Change | Why | Where |
|---|---|---|
| Decision units: the 49 budget-level units of rb5_units (`data/processed/rb5_units/units_primary.csv`, codes file `readings_final.csv`) instead of version 3's 56 family-label units | Fix list D-2: members of a family that share one token budget (max/min D at most 1.10) are one decision; this is what Proposition 1(iv) licenses (R1 round 3, New 1(a)(3)) | `trend.load_units()` builds the unit table (unit wedge `W_unit`, compute `C_unit`, year `unit_year`, date of the first member) and checks it against rb2's `decision_units.csv`: same 49 labels, wedges and compute to machine precision, same years, developers, dates and member counts (`rb3_econ2_wedge_checks.csv`, 14 checks, all pass) |
| Model-free IsoFLOP path slopes and the E-profiles of gamma read rb4's minima (`data/processed/rb4_chinflop/override_new_isoflop_budgets.csv`) | Fix list E1(a); audit numbers_technology #7: Chinchilla's path in FLOP-effective parameters (the primary convention since rb4), a = 0.489 (0.502 in total parameters) | `rb3common.RA1_BUDGETS`; ra3's `demand.ra1_path_slopes` is replaced in ra3's module namespace for this process only (no ra3 file is written) |
| The 32-technology trend uses the audited token counts | Fix list E6 ("unless E1 reruns on audited counts, which it should"); audit numbers_appF #3 | rb2's long table of wedges under every technology (`data/processed/rb2_decisions/exhibits_cache.pkl`, objects `L`, `M`, `exante`); the reference row now equals the "Models, OLS" row exactly (checked) |
| The high-wedge point of the wall grids follows the 2025 compute-weighted wedge of the decision units (5.38; was a hard-coded 5.39) | The 2025 aggregate on the 49 units is 5.3843 | `rb3common.W_AGG25`, set by `run.py` from the by-year table |

## 2. New analyses (E1(b)-(c))

**Trend variants** (`rb3_econ2_wedge_trend.csv`; OLS of ln w, reference technology, on release date; 95 percent wild
cluster bootstrap-t intervals by developer, Webb weights, B = 9,999):

| Trend | n | w per year [95% CI] | D/D* per year at sigma* = 0.70 (from M/M*) |
|---|---|---|---|
| Decision units, OLS (primary) | 49 | 1.86 [0.96, 3.54] | 2.06 [0.95, 4.39] |
| Developer fixed effects | 49 | 1.59 [0.47, 5.59] | 1.72 |
| Fixed effects and log compute | 49 | 2.70 [1.39, 5.21] | 3.19 |
| Units above 1e24 FLOP (6 developers) | 11 | 1.76 [0.78, 4.15] | 1.93 |
| Compute-weighted units | 49 | 2.21 [0.38, 13.1] | 2.53 |
| Flagship runs above 1e25 FLOP (descriptive) | 4 | 1.70 (1.31 without Pangu Ultra) | 1.86 |
| Version 3's 56 family-label units | 56 | 1.91 [1.03, 3.48] | 2.13 |

- Nine of the 18 developers have units at more than one date and identify the fixed-effects slope.
- The flagship row is an OLS slope over four runs (Aramco Metabrain 2024-03, w 1.13; Nemotron-4 340B 2024-06, 1.10;
  Llama 3.1 405B 2024-07, 1.29; Pangu Ultra 2025-04, 1.92), with no interval.
- The 56-unit row reproduces version 3 (1.91 [1.03, 3.48]); the point moves in the fourth decimal because of the
  MPT-30B token audit (1.05T).

**Scenario by trend** (`rb3_econ2_scenarios_by_trend.csv`; the frontier's wedge, w = 1.21, D/D* = 1.25, grows at the
trend from September 2026; the capped scenario stops at the 2025 wedge, 5.38):

| Trend | sigma* | D/D* per year | D per year, 4.2x; 5.06x | Five years, capped: 4.2x; 5.06x | Five years, uncapped: 4.2x; 5.06x | Years to the 2025 wedge |
|---|---|---|---|---|---|---|
| Decision units | 0.60 | 1.59 | 3.24; 3.56 | 108; 171 | 359; 571 | 2.4 |
| | 0.70 | 2.06 | 4.20; 4.61 | 201; 319 | 1,305; 2,074 | 2.4 |
| | 0.74 | 2.41 | 4.92; 5.40 | 294; 468 | 2,887; 4,588 | 2.4 |
| Within developers | 0.70 | 1.72 | 3.51; 3.85 | 201; 319 | 529; 841 | 3.2 |
| Above 1e24 FLOP | 0.70 | 1.93 | 3.94; 4.32 | 201; 319 | 945; 1,502 | 2.6 |
| Compute-weighted | 0.70 | 2.52 | 5.14; 5.64 | 201; 319 | 3,580; 5,690 | 1.9 |
| Flagships | 0.70 | 1.86 | 3.78; 4.15 | 201; 319 | 774; 1,230 | 2.8 |

Every trend reaches the 2025 wedge within five years, so the capped five-year multiple does not depend on the trend.
Without Alibaba the 2025 wedge is 3.11 and the capped five-year multiples at 5.06x are 114, 169 and 215.

## 3. What changed in the headline numbers (version 3 -> round 3)

| Quantity | Version 3 (56 units) | Round 3 (49 units) | Source |
|---|---|---|---|
| Growth of w a year, primary | 1.91 [1.03, 3.48] | **1.86 [0.96, 3.54]** | `_wedge_trend.csv` units_ols |
| Data multiple a year at sigma* = 0.70 | 2.13 [1.04, 4.29] | **2.06 [0.95, 4.38]** (M-based 2.06 [0.95, 4.39]) | `_scenarios_by_trend.csv`; `_wedge_trend.csv` |
| Frontier data demand a year at 5.06x (trend, sigma* = 0.70) | 4.76 | **4.61** | `_scenarios_by_trend.csv` |
| Compute-weighted w, 2023 / 2024 / 2025 | 1.34 / 2.18 / 5.39 | **1.32 / 2.17 / 5.38** | `_wedge_by_year.csv` |
| Units by year | 19 / 30 / 7 | **16 / 27 / 6** | `_wedge_by_year.csv` |
| Capped five-year multiple at 5.06x | 172 / 320 / 469 | **171 / 319 / 468** | `_scenarios.csv` catchup |
| Same, without Alibaba | 114 / 170 / 216 | **114 / 169 / 215** | `_scenarios_catchup_sensitivity.csv` |
| Years to reach the 2025 wedge (primary) | 2.3 | **2.4** | `_scenarios.csv` |
| 2029 frontier, wedge rising: w; D/U; extra cost (15.4 / 2.9) | 5.27; 19.2; 10.5% / 48.4% | **4.95; 17.9; 10.4% / 48.3%** | `_frontier_wall.csv` |
| Leave one developer out (units, OLS) | 1.73-2.30 | **1.66-2.26** | `_wedge_trend_lodo.csv` |
| 32 technologies, D/D* a year (models) | 1.93-2.33 (pre-audit counts) | **1.97-2.38** (audited; 2.24 under the reference) | `_wedge_trend_by_tech.csv` |
| Model-free IsoFLOP path growth, 4.2x; 5.06x | 2.04-2.53; 2.24-2.86 | **2.06-2.53; 2.26-2.86** (a = 0.35-0.50) | `_demand_all_technologies.csv` |
| Start-in-mid-2024 sensitivity: D/D* today | 3.7-9.7 (6.7 at 0.70) | **3.5-8.9 (6.2 at 0.70)** | `_scenarios_start_sensitivity.csv` |
| D/D* of the 2025 wedge (Table F-wall labels) | (7.18) at w = 5.39 | **(7.16) at w = 5.38**, at the solver's sigma* = 0.700553 | `_wall_observed_grid.csv` |
| Chinchilla E-profile (minima) | T: E 1.74 [1.68, 1.80], gamma 0.167 [0.156, 0.179] | **N_F: E 1.74 [1.68, 1.80], gamma 0.168 [0.157, 0.180]** | `_gamma_eprofiles.csv`, `_gamma_eprofiles_T.csv` |
| Local gamma drift per decade | -0.0006 (T) | **-0.0007 (N_F)**; local gamma 0.169 / 0.173 | `_gamma_local.csv` |

Unchanged: compute growth (5.06 [4.27, 6.02]; 4.13 on the 2024 file; Shapley 0.64 / 0.36), Panel A of Table 3, the
dates at which frontier runs reach the stocks, today's frontier (0.5-1.1 times the 100T stock; 2.2-4.9 at 22T), the
compute-optimal wall (7.7 / 7.4 / 7.2 and 51 / 40 / 35 percent at r = 4), the wedge-held frontier path (3.5 times the
stock in 2029; 4.6 and 23 percent; about $2 billion), the measured-wedge shortfalls at D/U = 4 (4, 15 and 53 percent),
fleet shares (0.17-0.33, 0.11-0.23, 0.08-0.17; 0.09-0.19 without Alibaba; 0.68 with no R&D multiple), gamma (0.165
(0.006), 66 [50, 90] times per halving), the synthetic-token break-evens (1.5-2.9).

## 4. What it means for the paper

- **The primary trend's interval now includes no growth** (lower end 0.96; 0.956 to 0.985 across bootstrap seeds, see
  the log). So do those of the within-developer, above-1e24 FLOP and compute-weighted trends. Only the within-developer
  trend conditional on log compute (2.70 [1.39, 5.21]) and two model-level or version-3 rows exclude it. Section V now
  says the wedge trend matches compute growth's contribution to data demand only at the point estimate.
- **The data multiple for the median decision** (fix list H7 slot): about 2.1 a year (2.06), 95 percent interval 0.95
  to 4.4. The introduction should not say "has multiplied tokens ... as much as compute growth adds" without the interval.
- **Table 3, panel B** is now "Scenario: the frontier adopts the open-weight trend", with rows for the primary (three
  curvatures), within developers, above 1e24 FLOP, compute-weighted and the four flagships, annual rates with intervals,
  and only the capped five-year multiples. The uncapped multiples are in the new Online Appendix table
  `tab:app-econ-trend`.
- **Page fit.** With the new rows, Table 3 and Table F1 no longer fit a page. Table F1 was split into a vintage table
  (`tab:app-econ-growth`) and a trend table (`tab:app-econ-trend`, new). Table 3 drops its r = 16 row and the 2026 and
  2028 frontier rows (the text cites neither; both are in Table F-wall and `_frontier_wall.csv`), and Table 3 and Table
  F-gamma use `\arraystretch` 0.96. Section V and Appendix F compile together with no overfull box and no float too
  large.

## 5. Inventory of changed or new files

- Code: `code/analysis/rb3_econ2/{rb3common,trend,scenario,wallobs,gamma,run,exhibits}.py`.
- New outputs: `output/tables/rb3_econ2_scenarios_by_trend.csv`, `rb3_econ2_gamma_eprofiles_T.csv`,
  `rb3_econ2_app_trend.tex`; changed: the other `rb3_econ2_*` tables whose inputs changed (section 1), the figure
  `output/figures/rb3_econ2_figure.{pdf,png}`, `data/processed/rb3_econ2/{summary.json, gamma_eprofile_grid.csv}`.
- Paper: `paper/sections/economics.tex`, `paper/sections/appendix_econ.tex`, `paper/tables/table3_econ.tex`,
  `appG_econ_growth.tex`, `appG_econ_trend.tex` (new), `appG_econ_wall.tex`, `appG_econ_gamma.tex` (each a copy of the
  generated table plus two comment lines).

## 6. Open issues

- The wild-bootstrap endpoints move with the seed by about 0.02 at the lower end and 0.1 at the upper end (18 clusters);
  the conclusion that the primary interval includes no growth holds at every seed tried.
- The within-developer slope rests on nine developers; the flagship trend rests on one 2025 run (Pangu Ultra).
- rb3 now reads rb2's exhibit cache (a pickle) for the audited 32-technology wedges; the replication run order must be
  rb4, rb5_units, rb2, then rb3.
