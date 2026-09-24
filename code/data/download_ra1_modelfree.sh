#!/usr/bin/env bash
# download_ra1_modelfree.sh -- module ra1_modelfree needs NO data beyond those fetched by
#   code/data/download_public.sh, download_m1_chinchilla.sh, download_m2_techpanel.sh and download_m8_measurement.sh:
#   data/raw/epoch_chinchilla/svg_extracted_data.csv        (Epoch digitization of Hoffmann et al. 2022; no license file)
#   data/raw/isoflop_experiments/isoflop_experiments.csv    (open-athena/isoflop-experiments: Llama 3, Marin x3; Apache-2.0)
#   data/raw/porian/experiment_results.pickle.xz            (Porian et al. 2024; MIT)
#   data/raw/farseer/1222_full.csv                          (Farseer, Li et al. 2025; no license file)
#   data/raw/gadre, data/raw/olmo_ladder, data/raw/datablations (m2 sweeps)
# Non-data references used for verification only (not needed at run time):
#   - Crossref metadata for the new bibliography entries in lit/bib/extra_ra1_modelfree.bib, e.g.
#       curl -s "https://api.crossref.org/works/10.1093/biomet/asr052"
#   - The Feng-He-Hu wild-bootstrap weights as implemented in R's quantreg (boot.rq, bsmethod = "wild"):
#       curl -s https://raw.githubusercontent.com/cran/quantreg/master/R/boot.R | grep -n -A14 'bsmethod == "wild"'
#   - de Vries (2023), "Go smol or go home" (parameters and the 30%-size / 100%-overhead point):
#       https://www.harmdevries.com/post/model-size-vs-compute-overhead/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for f in data/raw/epoch_chinchilla/svg_extracted_data.csv data/raw/isoflop_experiments/isoflop_experiments.csv \
         data/raw/porian/experiment_results.pickle.xz data/raw/farseer/1222_full.csv; do
  test -f "$ROOT/$f" && echo "ok  $f" || echo "MISSING $f (run the download script of the module that owns it)"
done
