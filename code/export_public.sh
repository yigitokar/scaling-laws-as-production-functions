#!/usr/bin/env bash
# Build the public replication snapshot in ../scaling-laws-pf-public, a separate git repository pushed to
# github.com/yigitokar/scaling-laws-as-production-functions. This script only writes the working tree there; it never
# commits or pushes (the author does both, after checking the tree).
#
# The export is a whitelist: only the paths in INCLUDE are copied, minus the patterns in EXCLUDE. It then mirrors the
# result into the target, so files that are no longer exported (for example internal notes of earlier snapshots) are
# removed from the target's working tree. The target's .git directory is never touched.
#
# Included: every analysis module under code/analysis (m1-m9, ra1-ra5, rb1-rb4, rb5_*, the shared library), the paper
#   code (code/paper), the training code and queues (code/sweep), the download scripts (code/data), code/tools, the
#   outputs (tables, figures, memos, sensitivity runs, the verification archive), our own processed data (the
#   controlled experiment and the model samples), the checksum manifest of the raw downloads, the paper sources with
#   the compiled main.pdf, the companion paper with its PDF, and from paper/notes only the pre-analysis plan, its
#   amendments, the experiment's specification and the coding protocol of the budget readings (rb5_reading_protocol.md,
#   which Online Appendix E5 says is in the replication package; added at round-3 integration).
# Excluded: raw downloads (re-downloadable; several sources have no redistribution license) except their checksum
#   manifest; harmonized copies of unlicensed third-party data (the processed folders not listed below); tokenized
#   corpora (*.bin), checkpoints (ckpt/) and caches (*.pkl); build artifacts; internal referee reports, audits and
#   notes (paper/referee, paper/audit, paper/notes other than the five files, draft correspondence); working
#   copies of sections and tables (sections/v*, tables/v*, tables/unused_v*); lit/ (literature notes); STATUS.md and
#   the agent workflow scripts (code/workflows); and output files that list digitized third-party runs.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$SRC/../scaling-laws-pf-public}"
cd "$SRC"

INCLUDE=(
  README.md LICENSE requirements.txt
  code/analysis code/paper code/sweep code/data code/tools code/export_public.sh
  data/raw/MANIFEST.sha256
  data/processed/sweep data/processed/m9_sweeps
  data/processed/m3_wedge data/processed/ra2_wedge
  data/processed/rb1_sigmaC data/processed/rb2_decisions data/processed/rb3_econ2 data/processed/rb4_chinflop
  data/processed/rb5_*
  output/tables output/figures output/memos output/sensitivity output/docs
  paper/main.tex paper/main.pdf paper/main.bbl paper/references.bib
  paper/sections paper/tables paper/figures
  paper/companion
  paper/notes/m9_preanalysis_plan.md paper/notes/m9_preanalysis_amendment1.md
  paper/notes/m9_preanalysis_amendment2.md paper/notes/m9_spec.md
  paper/notes/rb5_reading_protocol.md
)

# Extended regular expressions on the path relative to the repository root.
EXCLUDE=(
  '(^|/)__pycache__/' '\.pyc$' '(^|/)\.DS_Store$'
  '\.pkl$' '\.bin$' '(^|/)ckpt/'
  '^paper/.*\.(aux|log|blg|out|toc|synctex\.gz|fls|fdb_latexmk)$'
  '^paper/_build/' '^paper/referee/' '^paper/audit/' '^paper/aea_template/'
  '^paper/notes/ho_et_al_note_draft\.md$'
  '^paper/sections/v[^/]*/' '^paper/tables/v[^/]*/' '^paper/tables/unused_v[^/]*/'
  '^paper/companion/[^/]*_from_[^/]*_v2\.tex$'
  # Output files with digitized third-party run-level rows (Epoch AI's Chinchilla digitization, no license):
  '^output/tables/m1_chinchilla_selection_worst_points\.csv$'
  # The AEA's class and bibliography style carry the AEA's copyright notice: not redistributed (README says where to get them).
  '(^|/)AEA\.cls$' '(^|/)aea\.bst$'
  # (N, D) coordinates of Farseer's runs (no redistribution license), although without losses:
  '^output/tables/ra1_modelfree_farseer_grid\.csv$'
  # Stale module table superseded by the paper's tables (round-3 fix list R4):
  '^output/tables/m1_chinchilla_horse_race\.tex$'
)

LIST="$(mktemp -t export_list.XXXXXX)"
STAGE="$(mktemp -d -t export_stage.XXXXXX)"
trap 'rm -rf "$STAGE" "$LIST" "$LIST.all"' EXIT

: > "$LIST.all"
for p in "${INCLUDE[@]}"; do
  # Unmatched globs (for example rb5_* before a package has created it) expand to nothing.
  for q in $p; do
    if [ -e "$q" ]; then
      find "$q" -type f >> "$LIST.all"
    else
      case "$q" in *'*'*) ;; *) echo "note: not present yet, skipped: $q" ;; esac
    fi
  done
done
EXRE="$(IFS='|'; echo "${EXCLUDE[*]}")"
LC_ALL=C sort -u "$LIST.all" | grep -Ev "$EXRE" > "$LIST"

tar -cf - -T "$LIST" | tar -xf - -C "$STAGE"

cat > "$STAGE/.gitignore" <<'EOF'
# Written by code/export_public.sh for the public replication package.
.venv/
/tools/
__pycache__/
*.pyc
.DS_Store
paper/**/*.aux
paper/**/*.log
data/raw/*
!data/raw/MANIFEST.sha256
data/processed/sweep/*.bin
data/processed/sweep/ckpt/
EOF

mkdir -p "$DST"
rsync -a --delete --exclude='.git/' "$STAGE/" "$DST/"

# Warn about exported CSV files that look like run-level third-party data (loss with parameters and tokens per row),
# other than our own experiment's outputs.
while IFS= read -r f; do
  case "$f" in *.csv) ;; *) continue ;; esac
  case "$f" in data/processed/sweep/*|data/processed/m9_sweeps/*|output/tables/m9_sweeps_*) continue ;; esac
  hdr="$(head -1 "$f" | tr -d '\r' | tr 'A-Z' 'a-z')"
  if echo ",$hdr," | grep -Eq ',(l|loss|lnl|final_loss|eval_loss|val_loss),' \
     && echo ",$hdr," | grep -Eq ',(n|params|n_params|n_total|n_nonemb),' \
     && echo ",$hdr," | grep -Eq ',(d|tokens),'; then
    echo "check: run-level columns (N, D, loss) in $f"
  fi
done < "$LIST"

n="$(wc -l < "$LIST" | tr -d ' ')"
echo "exported $n files to $DST ($(du -sh "$DST" | cut -f1) including .git)"
