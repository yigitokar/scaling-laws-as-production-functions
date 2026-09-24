#!/usr/bin/env bash
# Download the public scaling-law datasets used in the paper (snapshot of 2026-09-23).
# Sources and licenses are documented in lit/notes/data_sources.md.
set -u
cd "$(dirname "$0")/../../data/raw"
get() {  # get URL OUTFILE
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then echo "exists  $2"; return; fi
  if curl -sfL --retry 5 --retry-delay 3 -A "Mozilla/5.0" "$1" -o "$2"; then echo "ok      $2 ($(wc -c <"$2") bytes)"; else echo "FAILED  $1"; rm -f "$2"; fi
}
gh_dir() {  # gh_dir OWNER/REPO PATH OUTDIR [pattern]
  gh api "repos/$1/contents/$2" --jq '.[] | select(.type=="file") | .download_url' 2>/dev/null | grep -E "${4:-.}" | while read -r u; do
    get "$u" "$3/$(basename "$u")"
  done
}

# ---- (A) controlled sweeps -------------------------------------------------
# Farseer (Li et al. 2025): 404-run (N, D) grid
get https://raw.githubusercontent.com/Farseer-Scaling-Law/Farseer/main/ipynb/data/1222_full.csv farseer/1222_full.csv
gh_dir Farseer-Scaling-Law/Farseer ipynb/data farseer '\.csv$'
# Step Law (Li et al. 2025 Part I): LR x BS grids inside (N, D) cells
get https://raw.githubusercontent.com/step-law/steplaw/main/data/dense_lr_bs_loss.csv steplaw/dense_lr_bs_loss.csv
gh_dir step-law/steplaw data steplaw '\.csv$'
# Gadre et al. 2024: over-training testbed (104 models, 3 corpora)
gh_dir mlfoundations/scaling exp_data/models gadre/models '\.json$'
gh api repos/mlfoundations/scaling/contents/exp_data --jq '.[] | "\(.type) \(.path)"' > gadre/exp_data_listing.txt 2>/dev/null
# Porian et al. 2024: tuned vs untuned, several FLOP conventions
get https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/data/experiment_results.pickle.xz porian/experiment_results.pickle.xz
gh api repos/formll/resolving-scaling-law-discrepancies/contents/data --jq '.[] | "\(.type) \(.path) \(.size)"' > porian/data_listing.txt 2>/dev/null
# Muennighoff et al. 2023 via IBM ColPret aggregation; ColPret itself (Choshen et al. 2024)
gh api repos/IBM/ColPret/contents/aggregated_eval --jq '.[] | "\(.type) \(.path) \(.size) \(.download_url)"' > colpret_listing.txt 2>/dev/null
gh_dir IBM/ColPret aggregated_eval colpret 'datablations|overtrain|olmo|opt|t5|K2'
# OLMo ladder (Bhagia et al. 2024)
gh_dir allenai/OLMo-ladder src/scripts/paper/data/ladder-runs olmo_ladder '\.csv$'
# DataDecide perplexity results (Magnusson et al. 2025)
get https://huggingface.co/datasets/allenai/DataDecide-ppl-results/resolve/main/data/train-00000-of-00001.parquet datadecide/ppl_results.parquet

# ---- (B) observational / cross-lab -----------------------------------------
get https://epoch.ai/data/benchmark_data.zip epoch_bench/benchmark_data.zip
get https://epoch.ai/data/ml_hardware.csv epoch_models/ml_hardware.csv
get "https://docs.google.com/spreadsheets/d/11m8O_mU0cUkOB_5wluPne4PNsuvsKNbbVAzbYNy-NXY/export?format=csv&gid=2087221150" ho2024/algorithmic_progress.csv
get https://raw.githubusercontent.com/ryoungj/ObsScaling/main/eval_results/base_llm_benchmark_eval.csv obsscaling/base_llm_benchmark_eval.csv
gh_dir ryoungj/ObsScaling eval_results obsscaling '\.csv$'
get https://raw.githubusercontent.com/felipemaiapolo/sloth/main/data/data_v2.csv sloth/data_v2.csv
gh_dir felipemaiapolo/sloth data sloth '\.csv$'
get https://huggingface.co/api/datasets/open-llm-leaderboard/contents/parquet/default/train/0.parquet openllm/contents_v2.parquet
get https://openrouter.ai/api/v1/models openrouter/models_2026-09-23.json
echo done
