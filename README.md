# What Optimizing Labs Reveal: Scaling Laws as Production Functions

Replication package and working materials for the paper by **Yigit Okar** (Care AI).
The research was carried out in close collaboration with **Claude** (Anthropic), which designed and ran the analysis
and drafted the paper with the author (see the title footnote of the paper).

**Idea.** Neural scaling laws — loss as a function of parameters N, training tokens D and compute C ≈ 6ND — are
production functions, and fitting them is production-function estimation. Optimization is double-edged for
measurement: cost-minimizing labs generate data that cannot identify the technology's curvature or optimal mix, but
their choices reveal their objectives. The paper (i) characterizes what scaling data identify, (ii) estimates the
elasticity of substitution between parameters and data without functional-form restrictions (IsoFLOP curvature
divided by twice the frontier slope), and (iii) inverts developers' first-order conditions to recover the revealed
value of compact models from over-training.

## Pre-registration of the controlled experiment
The analysis plan for our controlled two-corpus experiment is `paper/notes/m9_preanalysis_plan.md`, committed in the
authors' working repository as commit `bd5c0ad` (2026-09-24 03:12 +03, before any estimation on the experiment's
results). The power/coverage study, the pre-declared inference correction (deviation D8) and an outcome-to-headline
amendment (`paper/notes/m9_preanalysis_amendment1.md`) were committed as `a5f47f8` (17:44 +03; one wording fix to the stated writing time in `b2e9bf8` at 17:45), before any estimate on
the second corpus. This public repository was first pushed on 2026-09-24 with those files unchanged; its first commit
provides a public timestamp.

## Layout
```
paper/            LaTeX source (AEA.cls, AER style), sections/, tables/, compiled main.pdf
paper/notes/      plans, pre-analysis plan + amendment, writer and integration logs
paper/referee/    internal referee reports (rounds 1–2) used to revise the paper
lit/              literature notes by strand, synthesis, bibliography
code/analysis/    estimation library (sl.py) and one folder per analysis module (run.py regenerates each module)
code/sweep/       MLX transformer and warmup-stable-decay scaling sweeps (Apple silicon)
code/data/        download scripts for all public datasets
output/           tables (CSV + LaTeX), figures (PDF + PNG), memos (one per module, with independent reviews)
data/processed/   our own experiment's results (sweep/), the verified model samples (m3_wedge/, ra2_wedge/), m9 outputs
```

## Reproducing
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
bash code/data/download_public.sh            # plus code/data/download_*.sh for module-specific pulls
python code/analysis/<module>/run.py         # m1…m8, ra1…ra5, m9_sweeps
python code/sweep/prep_data.py && python code/sweep/run_grid.py main edu   # the controlled experiment (GPU, MLX)
```
Data excluded from this repository: raw downloads (re-downloadable with the scripts; several sources — Epoch AI's
Chinchilla digitization, the Farseer and Step Law run files — have no redistribution license), harmonized copies of
those data, tokenized training corpora, and model checkpoints.

## License
Code: MIT (see LICENSE). Paper text and figures: © the author. Third-party data remain under their original terms.
