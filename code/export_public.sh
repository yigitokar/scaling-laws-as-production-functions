#!/usr/bin/env bash
# Build the public replication snapshot in ../scaling-laws-pf-public (a separate git repo pushed to GitHub).
# Excludes: raw data (re-downloadable; some sources have no redistribution license), harmonized copies of
# unlicensed third-party data (Farseer, Step Law, Epoch's Chinchilla extraction), large binaries, build artifacts.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$SRC/../scaling-laws-pf-public"
mkdir -p "$DST"
rsync -a --delete --exclude='.git/' \
  --exclude='.venv/' --exclude='tools/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  --exclude='data/raw/' \
  --exclude='data/processed/sweep/*.bin' --exclude='data/processed/sweep/ckpt/' \
  --exclude='data/processed/m1_chinchilla/' --exclude='data/processed/m2_techpanel/' \
  --exclude='data/processed/m4_observational/' --exclude='data/processed/m5_progress/' \
  --exclude='data/processed/m6_montecarlo/' --exclude='data/processed/m7_theory/' \
  --exclude='data/processed/m8_measurement/' --exclude='data/processed/ra1_modelfree/' \
  --exclude='data/processed/ra3_econ/' --exclude='data/processed/ra4_obsfix/' --exclude='data/processed/ra5_theory/' \
  --exclude='paper/_build/' --exclude='paper/aea_template/test_*' --exclude='*.aux' --exclude='*.log' \
  --exclude='*.synctex.gz' \
  "$SRC/" "$DST/"
cp "$SRC/.gitignore" "$DST/.gitignore"
echo "exported to $DST ($(du -sh "$DST" | cut -f1))"
