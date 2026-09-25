# What Optimizing Labs Reveal: Scaling Laws as Production Functions

Replication package and working materials for the paper by **Yigit Okar** (Care AI). The research was carried out in
close collaboration with **Claude** (Anthropic), which designed and ran the analysis and drafted the paper with the
author; the author posed the research question and made the project's decisions on scope, disclosure and publication
(see the title footnote of the paper). The companion paper, "Observational Production Functions for Language Models and
the Measurement of Algorithmic Progress", is in `paper/companion/`.

**Idea.** Neural scaling laws (loss as a function of parameters N, training tokens D and compute C, about 6ND) are
production functions, and fitting them is production-function estimation. Optimization cuts both ways for measurement:
cost-minimizing labs generate data that cannot identify the technology's curvature or optimal mix, but their choices
reveal their objectives. The paper (i) characterizes what scaling data identify, (ii) estimates the elasticity of
substitution between parameters and data without functional-form restrictions (IsoFLOP curvature divided by twice the
frontier slope), and (iii) inverts developers' first-order conditions to recover the revealed value of compactness
from over-training.

**Versions.** The paper cites release `v4` of this repository, tagged at submission. The first public snapshot
(commit `59c5707`, pushed at 18:10 +03 on 24 September 2026) holds paper version 2, the pre-analysis plan and
Amendment 1. The version-3 snapshot, with Amendment 2, was pushed at [[X1: time]].

## Pre-analysis plan of the controlled experiment
The record below matches Online Appendix B4 of the paper, which gives the details and the table of deviations.
- **Plan.** `paper/notes/m9_preanalysis_plan.md`, committed in the author's working repository as commit `bd5c0ad` at
  03:12 (+03) on 24 September 2026; its header gives 03:15 as the time of writing, three minutes after the commit. By
  then part of the FineWeb-Edu main grid had finished and its raw endpoint losses were in the run logs, but no model
  had been fitted.
- **Power, size and coverage** of the plan's inference were computed from the design alone between 15:52 and 17:03,
  before the first estimation at 17:12. The size-corrected CR2 scheme (deviation D8) was written into the analysis memo
  at about 17:05 and first committed at 17:44 in `a5f47f8`; its timing rests on our account, and the coverage outputs
  that fix its calibration factors carry file times of 16:55 to 17:02.
- **Amendment 1** (`paper/notes/m9_preanalysis_amendment1.md`), committed at 17:44 in `a5f47f8`, maps each outcome of
  the questions Q1 to Q3 to the sentences of the paper that it would change. Its committed header misstated the time of
  writing as about 18:10, after the commit itself; `b2e9bf8`, at 17:45, changed it to 17:40. It was written before any
  FineWeb technology estimate or tilt, but after the FineWeb learning-rate calibration and the matched-cell loss
  differences between corpora had been computed, and after preliminary FineWeb-Edu estimates (listed in Appendix B4).
  It also replaced total parameters by FLOP-effective parameters for comparisons with the public designs; because the
  two counts differ by only 3 to 7 percent in this design, FineWeb-Edu's arms of Q1 and Q3 are exploratory.
- **Amendment 2** (`paper/notes/m9_preanalysis_amendment2.md`), committed at [[X1: time]] and pushed to this repository
  at [[X1: time]], before any FineWeb technology estimate, restates the outcome map against the current version of the
  paper, defines when an estimate lies inside or outside the band of Amendment 1, fixes the calibration factors, sets
  the conditions for a robust factor-biased verdict and declares the retraining of the FineWeb-Edu main-grid widths 384
  and 512 with every evaluation set (queue E, launched at 11:34:53 on 25 September 2026, commit `5599198`).
- **Neutral evaluation sets computed after training (deviation D13).** For runs whose job lists predate the trainer
  update of 17:44 on 24 September, the missing C4, PG-19 and WikiText-103 losses are computed after training from the
  saved endpoint weights with the same evaluation code (`code/sweep/eval_ckpt.py`, commit `4088638` at 23:07; merged
  into the analysis in `642f9f9` at 00:49 on 25 September); reloading reproduces the recorded losses exactly.
- **Timestamps.** The working repository's history is not public, because it contains third-party files without
  redistribution licenses. The plan and Amendment 1 carry the commit hashes and times given above, and this repository's
  first commit (`59c5707`) contains both files unchanged. This repository timestamps Amendment 2 and the version-3
  snapshot.

## Layout
```
paper/            LaTeX source (AER style): main.tex, sections/, tables/, references.bib, compiled main.pdf. To compile, put AEA.cls and aea.bst from the AEA's LaTeX template (aeaweb.org/journals/templates) in paper/ and paper/companion/
paper/companion/  the companion paper (observational.tex, compiled observational.pdf, its tables)
paper/notes/      the pre-analysis plan, Amendments 1 and 2, the experiment's specification (m9_spec.md) and the
                  coding protocol of the budget readings (rb5_reading_protocol.md)
code/analysis/    estimation library (sl.py, aer_style.py) and one folder per analysis module
code/paper/       scripts that build the paper's figures and several of its tables
code/sweep/       MLX transformer and warmup-stable-decay scaling sweeps (Apple silicon) and the queue scripts
code/data/        download scripts for the public datasets
code/tools/       make_manifest.py (checksums of the raw downloads)
output/           tables (CSV and LaTeX), figures (PDF and PNG), memos (one per module, most with an independent
                  review), sensitivity runs, and docs/verification.md (the verification archive of Appendix A)
data/raw/         MANIFEST.sha256 only (the raw downloads themselves are not redistributed)
data/processed/   our experiment's records (sweep/, m9_sweeps/) and the verified model samples and decision units
                  (m3_wedge/, ra2_wedge/, rb1_sigmaC/ to rb4_chinflop/, rb5_*)
```

## Reproducing
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
bash code/data/download_public.sh            # then the module-specific scripts code/data/download_*.sh
python code/tools/make_manifest.py --verify  # compare the downloads with the files we used (see below)
python code/analysis/<module>/run.py         # one module at a time, in the order below
python code/sweep/prep_data.py && python code/sweep/run_grid.py main edu   # the controlled experiment (GPU, MLX)
```
Modules read upstream outputs, so run them in this order: `m1_chinchilla` to `m8_measurement`, `ra1_modelfree`,
`ra2_wedge`, `ra3_econ`, `ra4_obsfix`, `ra5_theory`, `rb4_chinflop`, `rb1_sigmaC`, `rb2_decisions`, `rb3_econ2`, then
the round-3 modules `rb5_*` (each memo in `output/memos/` names its entry script), and `m9_sweeps` once the experiment
has finished. The modules write the tables and figures to `output/`; `code/paper/make_*.py` build Figures 1 and 3
and several appendix and companion tables. The paper compiles with Tectonic
(`tectonic -X compile main.tex` in `paper/`, and `observational.tex` in `paper/companion/`). Random seeds are fixed in
the code.

## Checksums of the raw downloads
`data/raw/MANIFEST.sha256`, written by `code/tools/make_manifest.py`, lists every file we downloaded (2,513 files,
18.3 GB, including the sources that carry no redistribution license) with its SHA-256, size in bytes, retrieval time
(UTC), source URL, the terms recorded for the source and a kind: `fixed` for a file at a stable location, `living` for
an API response, live database or web page (a new download will differ, and the checksum identifies the copy we used),
and `derived` for a file made locally from another download. After downloading, check the files with
`python code/tools/make_manifest.py --verify` (optionally followed by subdirectories of `data/raw`), which reports
identical, mismatched, changed living and missing files and exits with status 1 on a mismatch of a fixed file, or with
standard tools from `data/raw`:
`awk -F'\t' '!/^#/ {print $1"  "$4}' MANIFEST.sha256 | shasum -a 256 -c`.
Most files are fetched by the scripts in `code/data`, and the source documents of `rb2_decisions` and `rb4_chinflop`
by those modules. A few were downloaded outside the scripts (the FineWeb and FineWeb-Edu sample-10BT shards, the C4,
PG-19 and WikiText-103 validation files, Epoch AI's Chinchilla digitization and its model database); the manifest gives
their source URLs, and for the Hugging Face files the SHA-256 matches the one the Hub records.

## Records of the controlled experiment
All records are in `data/processed/sweep/`; losses are in nats per token, and the analysis converts them to bits per
byte.
- `results.jsonl`: one row per annealed endpoint. A field `loss_<set>` is present for each evaluation set evaluated
  during training, with sets `edu` and `web` (the FineWeb-Edu and FineWeb validation sets), `wiki` (WikiText-103), `c4`
  and `pg19`. The sets are fixed when a queue's job list is generated (`code/sweep/run_grid.py`): job lists generated
  before the trainer update of 17:44 on 24 September 2026 (the FineWeb-Edu main grid, the FineWeb main grid started at
  15:37 and the first FineWeb-Edu high-M job started at 17:33) carry the two in-distribution sets, and that high-M
  job also WikiText-103 (the FineWeb main grid's width-640 endpoint at 200 million tokens, rerun after the reboot of
  03:17 on 25 September from a regenerated list, carries all five); the FineWeb-Edu learning-rate calibration (`lrsweep`) carries FineWeb-Edu only; the
  learning-rate corners and the FineWeb calibration add WikiText-103; the later job lists carry all five sets, except
  the FineWeb-Edu seed replicates, which carry the two in-distribution sets.
- `results_posthoc.jsonl`: losses computed after training from saved endpoint weights (deviation D13;
  `code/sweep/eval_ckpt.py`, run by `code/sweep/queue_d.sh`). Each row names the checkpoint (`ckpt`), lists the sets
  computed after training (`added`) and gives, for the sets recorded during training, the difference between the
  reloaded model's loss and the recorded one (`recheck_diff`, zero when the reload is exact). The analysis
  (`code/analysis/m9_sweeps/m9_common.py`, `merge_posthoc`) fills only missing losses from this file and flags each
  filled loss with `posthoc_<set>`. Endpoints without saved weights (the runs launched before the trainer update,
  among them the FineWeb-Edu main grid, the FineWeb main grid at widths 128 to 256, the FineWeb-Edu (128,4) high-M
  shape and the learning-rate calibrations and corners) keep only the sets evaluated during training; queue E (tag
  `mainre`) retrains the FineWeb-Edu main-grid widths 384 and 512 with all five sets.
- `results_trunk.jsonl`: the unannealed (constant learning rate) trunk's validation loss at every branch point except
  the last (`anneal` = 0), for runs launched after the trainer update of 17:44 on 24 September, on the sets of the run's
  job list.
- `results_curves.jsonl`, `grid_log.txt`, `lr_fit.json`, `meta.json`, `tokenizer.json` and the queue logs (`*.log`)
  document the runs. The saved endpoint weights (`ckpt/`) and the tokenized corpora (`*.bin`) are not in the
  repository because of their size; the SHA-256 hashes of the checkpoints behind the post hoc evaluations will be
  published with release `v4`.

## Data not in this repository
Raw downloads are re-downloadable with the scripts and checkable against `data/raw/MANIFEST.sha256`. Several sources
have no license that permits redistribution (Epoch AI's digitization of the Chinchilla runs, the Farseer and Step Law
run files, and the raw digitized Llama 3 IsoFLOP points), so neither they nor our harmonized copies of them are here;
nor are the tokenized training corpora or model checkpoints. The Llama 3 and Marin points we use come from the
open-athena IsoFLOP compilation (Apache-2.0). FineWeb, FineWeb-Edu and C4 (ODC-BY), WikiText-103 (CC BY-SA) and PG-19
(Apache-2.0) are not redistributed.

## License
Code: MIT (see LICENSE). Paper text and figures: copyright the author. Third-party data remain under their original
terms.
