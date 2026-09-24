#!/usr/bin/env bash
# download_m8_measurement.sh -- module m8_measurement needs NO data beyond code/data/download_public.sh:
#   data/raw/porian/experiment_results.pickle.xz   (Porian et al. 2024; MIT)
#   data/raw/steplaw/dense_lr_bs_loss.csv          (Step Law; no license file: use, do not redistribute)
#   data/raw/steplaw/1004_fitted_lr_bs_scaling_model_parameters.csv
#   data/raw/epoch_chinchilla/svg_extracted_data.csv (Besiroglu et al. 2024; no license file)
#
# For reference only (not data, not used at run time): Porian et al.'s analysis code, read to port their
# IsoFLOP pipeline into code/analysis/m8_measurement/porian.py. Fetched into a scratch directory with:
REF_DIR="${1:-/tmp/porian_repo}"
mkdir -p "$REF_DIR"
for f in README.md analysis.py configs.py data.py make_paper.ipynb paper_figures.py paper_tables.py plotting.py requirements.txt utils.py; do
  curl -sL -o "$REF_DIR/$f" "https://raw.githubusercontent.com/formll/resolving-scaling-law-discrepancies/main/$f"
done
