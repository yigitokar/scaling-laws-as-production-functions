#!/usr/bin/env bash
# Public sources for module rb3_econ2 (round 3; Section V integration). Accessed 2026-09-24. Run from anywhere.
#
# 1. Epoch AI's public database files as captured by the Internet Archive in May/June 2024, used to reconcile our
#    frontier compute growth (5.06x, database of 2026-09-23) with Epoch's published 4.2x for ~2018 to May 2024
#    (Sevilla and Roldan, posted 2024-05-28). Only all_systems.csv (capture 2024-05-31 03:46:17 UTC) enters the analysis;
#    the two June captures are kept for reference.
#      sha256 epoch_all_systems_20240531.csv          07bedfca2bf2abc0f59554e1774f302d518b692d3a0ba6c8e3fe2540cceadb76
#      sha256 epoch_notable_ai_models_20240620.csv    00ebbfe746f9649ff865e0f17e41cbcb653eddb1af6a0bdfffdff3bac45dbdd3
#      sha256 epoch_large_scale_ai_models_20240619.csv 3fbf17530388fd507f26a798cc308429a90caa6c7efbffd98a23a24fd960ff38
#    Epoch AI's terms of use apply; the archived copies are not redistributed with the package (download script only).
# 2. DeepSeek API pricing page (retrieved 2026-09-24 17:15 UTC), for the optional comparison of the shadow value of a
#    unique token with posted prices of generated tokens. Prices change: the saved copy is the record.
#      sha256 deepseek_pricing_20260924.html           210f102275ccf1a6542f08a3bc9e4b4c7c83278cb74b35217bffa112df6363b2
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/raw/rb3_econ2"
mkdir -p "$OUT"
get() { [ -s "$2" ] && return 0; curl -sfL --retry 4 --retry-delay 3 -m 180 "$1" -o "$2" || { echo "FAILED $1"; rm -f "$2"; }; }

get "http://web.archive.org/web/20240531034617id_/https://epochai.org/data/epochdb/all_systems.csv" "$OUT/epoch_all_systems_20240531.csv"
get "http://web.archive.org/web/20240620060236id_/https://epochai.org/data/epochdb/notable_ai_models.csv" "$OUT/epoch_notable_ai_models_20240620.csv"
get "http://web.archive.org/web/20240619212353id_/https://epochai.org/data/epochdb/large_scale_ai_models.csv" "$OUT/epoch_large_scale_ai_models_20240619.csv"
get "https://api-docs.deepseek.com/quick_start/pricing" "$OUT/deepseek_pricing_20260924.html"
( cd "$OUT" && shasum -a 256 epoch_*.csv )
