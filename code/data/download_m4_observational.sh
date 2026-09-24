#!/usr/bin/env bash
# Extra public data for module m4_observational (snapshot 2026-09-23). Run from anywhere.
# Licenses: DataDecide (ODC-BY), Pythia evals (Apache-2.0), Gadre et al. mlfoundations/scaling (MIT).
set -u
RAW="$(cd "$(dirname "$0")/../../data/raw" && pwd)"
get() { mkdir -p "$(dirname "$2")"; [ -s "$2" ] && return; curl -sfL --retry 5 --retry-delay 3 "$1" -o "$2" || { echo "FAILED $1"; rm -f "$2"; }; }
export -f get

# (1) DataDecide downstream eval results (Magnusson et al. 2025): 4 parquet shards, ~700 MB total
for i in 0 1 2 3; do
  get "https://huggingface.co/datasets/allenai/DataDecide-eval-results/resolve/main/data/train-0000${i}-of-00004.parquet" \
      "$RAW/datadecide_eval/train-0000${i}-of-00004.parquet"
done

# (2) Pythia per-checkpoint lm-eval-harness results (Biderman et al. 2023), zero- and five-shot, standard + deduped
for m in 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b; do for v in "" "-deduped"; do for s in zero-shot five-shot; do
  gh api "repos/EleutherAI/pythia/contents/evals/pythia-v1/pythia-${m}${v}/${s}" --jq '.[] | select(.type=="file") | .download_url' 2>/dev/null \
  | while read -r u; do echo "$u $RAW/pythia_evals/pythia-${m}${v}/${s}/$(basename "$u")"; done
done; done; done | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'

# (3) Gadre et al. (2024) downstream "heavy" evals for the 104 over-training testbed models
gh api repos/mlfoundations/scaling/contents/exp_data/evals --jq '.[] | .download_url' \
  | while read -r u; do echo "$u $RAW/gadre/evals/$(basename "$u")"; done | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'
get https://raw.githubusercontent.com/mlfoundations/scaling/main/exp_data/eval_metadata.csv "$RAW/gadre/eval_metadata.csv"
echo done

# (4) Hugging Face model-card metadata (createdAt = upload date; safetensors.total = exact parameter count)
#     for every base model in the observational panel (ids listed by run.py in data/processed/m4_observational/hf_ids.txt)
IDS="$(cd "$(dirname "$0")/../.." && pwd)/data/processed/m4_observational/hf_ids.txt"
if [ -s "$IDS" ]; then
  while read -r id; do
    f="$RAW/hf_meta/$(echo "$id" | tr '/' '__').json"
    echo "https://huggingface.co/api/models/${id}?expand[]=createdAt&expand[]=safetensors&expand[]=downloadsAllTime $f"
  done < "$IDS" | xargs -P 6 -n 2 bash -c 'get "$0" "$1"'
fi
# OLMo-2 targets of the OLMo ladder (exact parameter counts for the experimental benchmark)
for id in allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-13B; do
  get "https://huggingface.co/api/models/${id}?expand[]=createdAt&expand[]=safetensors&expand[]=downloadsAllTime" "$RAW/hf_meta/$(echo "$id" | tr '/' '_').json"
done
echo done-hf
