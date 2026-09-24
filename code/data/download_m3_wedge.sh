#!/usr/bin/env bash
# Extra public data for module m3_wedge (revealed inference demand). Snapshot: 2026-09-23/24. Run from anywhere.
# Everything here is public metadata (Hugging Face model API and model cards, OpenRouter public API, LMArena CC-BY-4.0
# leaderboard dataset). Model cards are kept only to verify the stated training-token counts; do not redistribute.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/data/raw"
PY="$ROOT/.venv/bin/python"
IDS="$ROOT/code/analysis/m3_wedge/ids.py"
get() { mkdir -p "$(dirname "$2")"; [ -s "$2" ] && return 0; curl -sfL --retry 4 --retry-delay 3 -m 60 "$1" -o "$2" || { echo "FAILED $1"; rm -f "$2"; }; }
export -f get

# (1) Hugging Face model API: 30-day downloads, all-time downloads, exact parameter counts (safetensors), upload date,
#     likes -- for every Sample-B base model and its official post-trained counterparts.
{ $PY "$IDS" base; $PY "$IDS" instruct; } | sort -u | while read -r id; do
  echo "https://huggingface.co/api/models/${id}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=safetensors&expand[]=createdAt&expand[]=likes $RAW/m3_hf/api/$(echo "$id" | sed 's#/#__#g').json"
done | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'

# (2) Model cards (README.md) of the curated additions: the training-token statements are checked by regex in
#     build_choices.py (the README of gated repositories is public).
$PY "$IDS" cards | while read -r id; do
  echo "https://huggingface.co/${id}/resolve/main/README.md $RAW/m3_hf/cards/$(echo "$id" | sed 's#/#_#g').md"
done | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'

# (3) config.json (vocabulary size, width, tied embeddings -> embedding parameters). Gated repositories (Meta, Google)
#     are read from ungated mirrors listed in curated.CONFIG_MIRROR (NousResearch/unsloth copies of the same config).
$PY "$IDS" config | while read -r id src; do
  echo "https://huggingface.co/${src}/resolve/main/config.json $RAW/m3_hf/configs/$(echo "$id" | sed 's#/#__#g').json"
done | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'

# (4) OpenRouter per-model endpoints (provider count, per-provider prices, quantization) for every model in the
#     2026-09-23 models snapshot that lists a Hugging Face id.
"$PY" - "$RAW" <<'EOF' | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'
import json, sys, os
raw = sys.argv[1]
d = json.load(open(os.path.join(raw, "openrouter", "models_2026-09-23.json")))["data"]
for m in d:
    if m.get("hugging_face_id") and not m["id"].endswith((":free", ":batch")):
        slug = m["id"]
        print(f"https://openrouter.ai/api/v1/models/{slug}/endpoints {raw}/m3_openrouter/endpoints/{slug.replace('/', '__')}.json")
EOF

# (5) LMArena leaderboard dataset (text arena, full history of published leaderboards; CC-BY-4.0)
get "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/refs%2Fconvert%2Fparquet/text/full/0000.parquet" \
    "$RAW/m3_lmarena/text_full.parquet"
echo done-m3
