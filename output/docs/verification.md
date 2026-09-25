# Verification of the formal results (supplementary archive to Online Appendix A)

Version 3 of the paper, 2026-09-24. Referee R3 (round 2, N7) asked that the verification tallies and the claims register be
moved out of Online Appendix A into the replication package. This file is that archive. Appendix A now states only the
results the main text uses; its introduction points here.

## 1. What is verified, and how

Every algebraic identity in Appendix A was checked symbolically (SymPy), and every comparative-static, bias and
information statement numerically: by brute-force optimization of the developer's objective, by simulation, or by
direct computation of Fisher information. Each check is a row of a claims register with its method, its metric
(tolerance or maximum error) and a pass flag.

| Register | Code (entry point) | Claims | Verdicts | Methods | All checks pass |
|---|---|---|---|---|---|
| `output/tables/m7_theory_claims.csv` (version 1 theory) | `code/analysis/m7_theory/run.py` | 61 | 31 verified, 21 new, 6 imprecise, 3 false | 31 symbolic, 22 numeric, 6 Monte Carlo, 1 symbolic + numeric, 1 bootstrap | yes (61/61) |
| `output/tables/m7_theory_ledger_checks.csv` (numbers quoted in the text) | same | 53 | -- | recomputation against the text ledger | yes (53/53) |
| `output/tables/ra5_theory_claims.csv` (version 2 theory) | `code/analysis/ra5_theory/run.py` | 38 | 32 new, 5 corrected, 1 verified | 20 symbolic, 17 numeric, 1 Monte Carlo | yes (38/38) |
| Version 3 addition: binding data cap in a family (Prop. A4(vi) of version 3) | `code/paper/verify_family_cap.py` | 4 checks | new | brute force | yes |

Verdicts classify the *original* statement that was checked (for example, in the model specification or an earlier
draft); the pass flag refers to the corrected statement that the paper now makes. The three "false" rows of the version-1
register concern the compute-equivalent-gain corollary (frontier gains are constant in compute iff E and the frontier
elasticity gamma coincide, not iff (alpha, beta, E) coincide); the corrected statement moved with the observational
material to the companion paper. The five "corrected" rows of the version-2 register are the size-dependent FLOP price
case of the generalized wedge, the sign of the Diamond-McFadden-Rodriguez drift (companion paper), and the on-path
geometry of Proposition A1(iii) (quartic criterion along opposite-rate directions; the split A/B case).

## 2. Numbering map (the registers use version-2 numbers)

| Version 2 (registers) | Version 3 (paper) | Label | Main text |
|---|---|---|---|
| Lemma A1, A2; Corollary A1 | Lemma A1, A2; Corollary A1 (renamed "Complementarity, sigma < 1") | lem:sigma, lem:soc, cor:complements | Lemma 1 |
| Lemma A3 (i)-(iv), (vi) | Lemma A3 (i)-(iv); (vi) dropped | lem:alloc | duality text, Section I |
| Lemma A3 (v) | companion paper | -- | -- |
| Lemma A4, A5 | Lemma A4, A5 | lem:geometry, lem:ce | Propositions 1-3 |
| Proposition A1 (i), (ii), (iii)(a)-(b), (iv) | Proposition A1 | prop:fd | Proposition 2 |
| Proposition A1 (iii)(c) (the n^{-1/4} rate) | Remark A3 | rem:A-rate | Proposition 2, remark |
| Proposition A2 | Proposition A2 | prop:A-info | Proposition 2 |
| Proposition A3 (DMR), Corollary A2 (CEG), Proposition A4 (Farrell), A5-A7 (transmission, selection, proxies), Corollaries A4-A5 (growth accounting) | companion paper (`paper/companion/proofs_from_appendixA_v2.tex`) | -- | -- |
| Proposition A8 | Proposition A3 | prop:A-wedge | Proposition 1 |
| Corollary A3 | Corollary A2 | cor:suff | Proposition 1 |
| Proposition A9 (i)-(v) | Proposition A4 (i)-(v); new part (vi), binding data cap | prop:A-family | Proposition 1 (family budgets, cap case) |
| Proposition A10 | Proposition A5 | prop:A-modelfree | Proposition 3 |
| Proposition A11 | Proposition A6 | prop:A-pi | Proposition 3 (beyond-design part; alias prop:pi) |
| Remark after A10 (finite grids) | Remark A6 | rem:A-grid | Section II |
| Remark "illustration at frontier scale" | dropped (superseded by the anchors of Online Appendix E, placed at each design's largest bracketed budget) | -- | -- |
| Remark on parametric projection | Remark A7 | rem:A-pi-param | Section IV |
| Remark on conduct | Remark A4 | rem:A-conduct | Section IV |

## 3. The version-3 check (`code/paper/verify_family_cap.py`)

Setting: three family members on a common token budget with the Chinchilla technology of Besiroglu et al. (2024);
member i values loss at Pi_i and plans serving T_i; sizes are chosen at interior optima given the budget.

1. At the unconstrained family optimum D* the family condition of Proposition A4(iii) holds: sum_i pi_i (1 + m_N,i) =
   W_H to 1e-6 (mu = 1.6e-7).
2. With the budget capped at 0.3, 0.5 and 0.8 D*, the cap's normalized multiplier mu is positive (3.60, 1.38, 0.32) and
   sum_i pi_i (1 + m_N,i) = (1 + mu) W_H holds to numerical precision, so W_H - 1 bounds the pi-weighted value of
   compactness from below (Proposition A4(vi), first part).
3. Every member whose own marginal value of data exceeds its cost (rho_i >= 1) has w_i - 1 <= m_N,i (second part); a
   member with rho_i < 1 can have w_i - 1 > m_N,i (at 0.3 D*: 4.68 against 4.20), which is why the member bound needs the
   premise that each member would train longer at its own size.
4. With a binding size cap on a member that wants more data (rho = 1.005 at 0.5 D*), w - 1 = 8.00 exceeds m_N = 2.52:
   the member bound fails when a tier cap also binds (third part).

Run: `.venv/bin/python code/paper/verify_family_cap.py` (deterministic, under one second; prints "ALL CHECKS PASSED").

## 4. Other verification records cited by the paper

- Monte Carlo of the paper's identification results: `output/memos/m6_montecarlo_v2.md` (Online Appendix C).
- Partial identification and anchors: `output/memos/rb2_decisions.md` and its review (Online Appendix E).
- The controlled experiment's power, size and coverage, computed before estimation: `output/tables/m9_sweeps_power.csv`,
  `output/tables/m9_sweeps_power_coverage.csv` (Online Appendix B, Table B3), committed in `a5f47f8`.
