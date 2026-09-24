# Module m9_sweeps — our controlled two-lab experiment (spec)

Data (produced by code/sweep/*.py on the M5 Max, MLX):
- data/processed/sweep/results.jsonl — one row per (regime ∈ {edu, web}, width d, depth L, lr, seed, D_target) endpoint;
  tags: "main" (grid), "seeds" (replicates: seeds 1-2 at d ∈ {128,256,384}, D ∈ {50,100,200}M, edu only),
  "lrsweep" (LR calibration, D = 50M, edu). Fields: N_nonemb, N_emb, N_total, tokens (D actually trained),
  C (flops incl. attention & unembedding), C_6ND_nonemb, C_6ND_total, loss_edu, loss_web (nats/token on 1,048,576
  held-out tokens of each corpus), train_loss_trunk, etc.
- data/processed/sweep/results_curves.jsonl — trunk training-loss curves (constant-LR phase) every 1,000 steps.
- data/processed/sweep/meta.json — bytes/token of each validation set (edu 3.8857, web 3.7426) for bits-per-byte.
- data/processed/sweep/lr_fit.json — LR calibration: lr*(d) = 3.07e-3 (d/256)^-0.90.
- Design: widths d ∈ {128,192,256,320,384,448,512,640}, depth d/64, head dim 64, seq 256, batch 64×256 tokens,
  AdamW(0.9, 0.99), wd 0.1, clip 1, warmup 250 steps, WSD with 1−sqrt cooldown over the last 20% (branched from the
  constant-LR trunk at 0.8·D_k), D_k ∈ {25,50,100,200,400,800}M (≤400M for d ∈ {448,512}; ≤200M for d = 640).
  Same data order across sizes (common random numbers). Shared 8,192-token byte-level BPE trained on both corpora.
  "Labs": edu = FineWeb-Edu sample-10BT (educational-quality filtered); web = FineWeb sample-10BT.

Tasks
1. Describe the design (runs per lab; N range; D range; M range under both N conventions; design statistic
   sd(ln M | ln C); compute range). Loss in nats/token and bits-per-byte; cross-evaluation matrix (each lab's
   models on both validation sets).
2. Technology per lab, on (a) own validation set and (b) a COMMON output (both labs evaluated on the same
   validation set — do both edu-val and web-val): Huber-LSE (sl.fit_chinchilla) and Gaussian NLS; generalized
   outer exponent (q/κ free, as in m2's code: code/analysis/m2_techpanel); report E, A, B (normalized), α, β, a,
   γ, σ*, σ*_q, M*(C) at 1e15/1e16/1e17, with bootstrap SEs clustered by model width (8 clusters: use wild or
   pairs and report both; few clusters!). Both N conventions (non-embedding vs total incl. tied embedding):
   quantify how the convention moves a and M* vs σ (Pearce–Song / m8 Proposition M at small scale, where the
   embedding share is 20–73%).
3. Neutrality of data quality (Doraszelski–Jaumandreu-style bias test) with a COMMON output: nest and test
   (i) Hicks-neutral in reducible loss (common α, β; A_r/B_r constant), (ii) data-augmenting (only B_r differs),
   (iii) parameter-augmenting (only A_r), (iv) E-shift only, (v) exponents differ. Report the implied
   factor-augmenting productivity χ = βψ_D − αψ_N and its effect on M* and on the revealed wedge ŵ
   (model_spec Prop. 4: ŵ = w·e^{−χ}). Is higher-quality data (FineWeb-Edu) data-augmenting (Lemma 1: then the
   edu lab should choose FEWER tokens per parameter)? Compare with m2's DataDecide/Gadre results.
4. Extrapolation check for the wedge (the key open issue of m3): fit the technology on a "Chinchilla-like"
   support (M ≤ 100 or ≤ 341 in the convention used by the wedge module) and predict (i) the loss and (ii) the
   local elasticity ε_D and the wedge w at high M (up to ~2,000). Compare with the full-grid fit and with local
   nonparametric elasticities (finite differences along D at fixed N). Direction: does the restricted fit
   overstate the value of extra tokens (Sardana et al. 2024), i.e., is ŵ understated when extrapolating? Report the
   ratio w_extrap/w_full at each N as a function of M.
5. Seed noise: SD of log loss across seeds (and data orders) at fixed (N, D) → measurement-error variance of
   output; compare with residual SD of the fitted technology (misspecification vs noise).
6. Functional dependence with real losses: select the on-path subsample (points nearest the fitted
   expansion path at each compute level) and show the κ-free profile over σ* is flat vs curved on the full grid
   (as m1 did for Chinchilla), and the Jacobian condition numbers.
7. (If time) Learning curves: constant-LR trunk loss vs tokens (Wright-curve exponent) vs the cooldown
   endpoints' β — the "unfinished output" bias of reading exponents off training curves.

Deliverables: code/analysis/m9_sweeps/run.py; tables output/tables/m9_*.csv/.tex; figures output/figures/m9_*
(Figure 9 for the paper: 2-panel — (a) loss vs D by width for both labs with fitted curves on a common output,
(b) isoquants of both labs + expansion paths); memo output/memos/m9_sweeps.md (same structure as other memos),
then an independent review (output/memos/m9_sweeps_review.md).
