#!/usr/bin/env bash
# Additional public data used by module m2_techpanel (snapshot of 2026-09-23).
# Everything else the module uses is fetched by download_public.sh (Farseer, Gadre, OLMo ladder,
# ColPret, DataDecide-ppl, Epoch Chinchilla extraction).
set -u
cd "$(dirname "$0")/../../data/raw"
get() {  # get URL OUTFILE
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then echo "exists  $2"; return; fi
  if curl -sfL --retry 5 --retry-delay 3 -A "Mozilla/5.0" "$1" -o "$2"; then echo "ok      $2 ($(wc -c <"$2") bytes)"; else echo "FAILED  $1"; rm -f "$2"; fi
}

# ---- Muennighoff et al. (2023) datablations: final C4 validation losses are hard-coded in these notebooks
#      (NAMES_TO_VAL_LOSSES, PARAMS_MAP, TOKENS_MAP). Apache-2.0.
get https://raw.githubusercontent.com/huggingface/datablations/main/plotstables/return_alloc.ipynb datablations/return_alloc.ipynb
get https://raw.githubusercontent.com/huggingface/datablations/main/plotstables/contours.ipynb     datablations/contours.ipynb

# ---- Kricheli et al. (2026) collinear vs non-collinear designs (HF dataset TPPIsCriticalFor/colinear_scaling_models,
#      GPL-2.0). Only the pre-extracted per-epoch *training* losses and the authors' fit summaries are public in tabular
#      form; no validation losses (val_losses arrays in training_metrics/*.npz are empty). Downloaded for documentation;
#      not used for technology estimates (see memo, Section 5).
for f in c4 cosmopedia pes2o redpajama wikipedia wikipedia_bf16 wikipedia_bigtpp; do
  get "https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models/resolve/main/extracted_losses/$f.csv" kricheli/$f.csv
done
get https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models/resolve/main/README.md kricheli/README.md
get https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models/resolve/main/full_coverage_results/full_coverage_all_seeds.parquet kricheli/full_coverage_all_seeds.parquet

# ---- Metadata consulted (not saved; values are hard-coded with provenance in code/analysis/m2_techpanel/m2_data.py):
#   OLMo ladder parameter counts: gh api repos/allenai/OLMo-ladder/contents/src/scaling/utils.py  (MODEL_PARAMS)
#   OLMo ladder model configs:    gh api repos/allenai/OLMo-ladder/contents/src/ladder/ladder.py
#   DataDecide batch sizes / non-embedding N / last steps: gh api repos/allenai/DataDecide/contents/utils/constants.py
#   Farseer fitting code (output = IntelliValSet_Raw|en, raw N and D):
#     gh api repos/Farseer-Scaling-Law/Farseer/contents/scalinglaw_utils/scaling_law_fiting/read_data_big_exp.py
echo done
