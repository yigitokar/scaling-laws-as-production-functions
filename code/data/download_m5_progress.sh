#!/usr/bin/env bash
# Module m5_progress: extra public downloads (snapshot of 2026-09-23).
# Ho et al. (2024) replication code, epoch-research/lm-algorithmic-progress (MIT license),
# pinned to commit 29c7d852b6cd19ae11bb6662e9c1e61c03b1e4a7 (last push 2024-03-28).
# The data sheet itself (ho2024/algorithmic_progress.csv) is downloaded by download_public.sh.
set -u
cd "$(dirname "$0")/../../data/raw"
mkdir -p ho2024/code
SHA=29c7d852b6cd19ae11bb6662e9c1e61c03b1e4a7
for f in LICENSE README.md appendices.ipynb cross-validation.ipynb section3.ipynb "sections1%262.ipynb"; do
  out="ho2024/code/$(printf '%b' "${f//%/\\x}")"
  if [ -s "$out" ]; then echo "exists  $out"; continue; fi
  curl -sfL --retry 5 "https://raw.githubusercontent.com/epoch-research/lm-algorithmic-progress/${SHA}/${f}" -o "$out" \
    && echo "ok      $out" || echo "FAILED  $f"
done
