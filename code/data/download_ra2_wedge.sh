#!/usr/bin/env bash
# Extra public data for module ra2_wedge (re-specified inversion). Snapshot: 2026-09-24. Run from anywhere.
# Everything here is public: Hugging Face model API (model-tree derivative counts), Hugging Face model cards (README,
# kept only to audit training flags and stated deployment targets; do not redistribute), and arXiv PDFs of the papers
# whose published numbers are used (DeepSeek LLM, MiniCPM, Patterson et al. 2022, Wu et al. 2022, OpenRouter/a16z
# State of AI).
# OpenRouter per-model token volumes: NOT downloaded. The documented API (/api/v1/models, /endpoints) has no usage
# field, and OpenRouter's Terms of Service (last updated 2026-08-31, Section 7) prohibit scripts/crawlers that scrape
# information from the Site, which covers the rankings/activity pages. We use only the author-level totals printed in
# the published State of AI report (arXiv:2601.10088, Table 1).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/data/raw"
PY="$ROOT/.venv/bin/python"
IDS="$ROOT/code/analysis/m3_wedge/ids.py"
get() { mkdir -p "$(dirname "$2")"; [ -s "$2" ] && return 0; curl -sfL --retry 4 --retry-delay 3 -m 60 "$1" -o "$2" || { echo "FAILED $1"; rm -f "$2"; }; }
export -f get

# (1) Hugging Face model tree: number of derivative repositories by relation (finetune, adapter, quantized, merge)
#     for every Sample-B base model and its official post-trained counterparts (expand[]=childrenModelCount).
{ $PY "$IDS" base; $PY "$IDS" instruct; } | sort -u | while read -r id; do
  echo "https://huggingface.co/api/models/${id}?expand[]=childrenModelCount $RAW/ra2_hf_tree/$(echo "$id" | sed 's#/#__#g').json"
done | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'

# (2) Model cards (README.md) of every Sample-B base model (audit of distillation/pruning, multimodal pretraining,
#     instruction-tuned-only releases, stated deployment target). The README of gated repositories is public.
$PY "$IDS" base | while read -r id; do
  echo "https://huggingface.co/${id}/resolve/main/README.md $RAW/ra2_cards/$(echo "$id" | sed 's#/#__#g').md"
done | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'

# (3) Papers whose published numbers enter the module (verification of numbers; text extracted with PyMuPDF)
for id in 2401.02954 2404.06395 2204.05149 2111.00364 2601.10088; do
  get "https://arxiv.org/pdf/${id}" "$RAW/ra2_papers/${id}.pdf"
done
"$PY" - "$RAW/ra2_papers" <<'EOF'
import os, sys, pymupdf
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if f.endswith(".pdf") and not os.path.exists(os.path.join(d, f[:-4] + ".txt")):
        doc = pymupdf.open(os.path.join(d, f))
        open(os.path.join(d, f[:-4] + ".txt"), "w").write("\n".join(p.get_text() for p in doc))
EOF
echo done-ra2
