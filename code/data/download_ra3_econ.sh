#!/usr/bin/env bash
# Public sources for module ra3_econ (economic implications). Accessed 2026-09-24. Run from anywhere.
#
# The module's computations use only data already in data/raw (Epoch AI model database, snapshot 2026-09-23, fetched by
# code/data/download_public.sh; sha256 all_ai_models.csv = fff4038ac5cd2f671a79dcbb6a8096dabb0d1e65a6d29ad1171e142637d1a195,
# frontier_ai_models.csv = 2b93b016517331b5a71ef207cc95cd66acc2d81f44c570cad7983197b91a5b59) and upstream module outputs.
# The PDFs below were downloaded only to VERIFY the external parameters hard-coded in
# code/analysis/ra3_econ/ra3common.py (stock of public text, repetition half-lives, disclosed inference shares,
# DeepSeek's allocation law, GATE's training-inference trade-off). They are arXiv versions; do not redistribute.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/raw/ra3_econ_lit"
mkdir -p "$OUT"
get() { [ -s "$2" ] && return 0; curl -sfL --retry 4 --retry-delay 3 -m 120 "$1" -o "$2" || { echo "FAILED $1"; rm -f "$2"; }; }

get https://arxiv.org/pdf/2211.04325 "$OUT/villalobos2024_arxiv2211.04325.pdf"   # Villalobos et al. (ICML 2024): stock 100T/320T, 2028
get https://arxiv.org/pdf/2305.16264 "$OUT/muennighoff2023_arxiv2305.16264.pdf"  # Muennighoff et al. (NeurIPS 2023): R*_D, R*_N
get https://arxiv.org/pdf/2204.05149 "$OUT/patterson2022_arxiv2204.05149.pdf"    # Patterson et al. (2022): 3/5 inference
get https://arxiv.org/pdf/2111.00364 "$OUT/wu2022_arxiv2111.00364.pdf"           # Wu et al. (MLSys 2022): 10:20:70
get https://arxiv.org/pdf/2503.04941 "$OUT/erdil2025gate_arxiv2503.04941.pdf"    # Erdil et al. (2025), GATE
get https://arxiv.org/pdf/2401.02954 "$OUT/bi2024deepseek_arxiv2401.02954.pdf"   # DeepSeek LLM: D_opt = 5.8316 C^0.4757

# Web pages checked (not archived; quoted facts in ra3common.py):
#   https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year   (Sevilla and Roldan 2024)
#   https://epoch.ai/data-insights/openai-compute-spend                                  (You 2025; press-reported)
#   https://epoch.ai/gradient-updates/r-and-d-vs-training-compute                         (Denain and Wu 2026)
#   https://www.sec.gov/Archives/edgar/data/1713445/000162828024006294/reddits-1q423.htm (Reddit S-1, $203.0M licensing)
#   Nvidia Q4 FY2024 earnings call (21 February 2024), CFO remarks: ~40% of data-center revenue for AI inference
#   (quoted in https://www.fool.com/investing/2024/03/03/90-billion-reasons-why-buying-nvidia-stock-is-a-no/).
ls -la "$OUT"
