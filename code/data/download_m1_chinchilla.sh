#!/usr/bin/env bash
# Additional public data for module m1_chinchilla (snapshot of 2026-09-23).
#   (1) open-athena/isoflop-experiments (Hugging Face, Apache-2.0; 814 rows): IsoFLOP points for Chinchilla
#       (two derived versions of the Epoch digitization), Llama 3 (Czech digitization of Grattafiori et al. 2024
#       Fig. 2), Marin 2026-03 scaling ladders (3 corpora, W&B export) and the (Mis)Fitting FineWeb/C4 sweep.
#       Dataset commit used: ba52cfacd7886583f4e285b8326d8ad051dee565 (lastModified 2026-03-27); the URLs below are
#       pinned to it (round-3 integration, at WP6's request; the local files match that revision exactly).
#   (2) eric-czech/llama3_isoflop_extraction (GitHub, no license file; used for cross-checking only, not
#       redistributed): the raw digitized Llama 3 IsoFLOP points.
# The Epoch Chinchilla extraction itself is downloaded by download_public.sh (epoch_chinchilla/).
set -u
cd "$(dirname "$0")/../../data/raw"
mkdir -p isoflop_experiments llama3_isoflop
curl -sfL https://huggingface.co/datasets/open-athena/isoflop-experiments/resolve/ba52cfacd7886583f4e285b8326d8ad051dee565/isoflop_experiments.csv \
  -o isoflop_experiments/isoflop_experiments.csv
curl -sfL https://huggingface.co/datasets/open-athena/isoflop-experiments/resolve/ba52cfacd7886583f4e285b8326d8ad051dee565/README.md \
  -o isoflop_experiments/README.md
for f in README.md extract_isoflops_points.py isoflops_points.csv; do
  curl -sfL "https://raw.githubusercontent.com/eric-czech/llama3_isoflop_extraction/main/$f" -o "llama3_isoflop/$f"
done
# Besiroglu et al. (2024) analysis notebook (used to diagnose which objective produced their published estimates)
[ -s epoch_chinchilla/data_analysis.ipynb ] || curl -sfL \
  https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data_analysis.ipynb \
  -o epoch_chinchilla/data_analysis.ipynb
wc -l isoflop_experiments/isoflop_experiments.csv llama3_isoflop/isoflops_points.csv
