#!/usr/bin/env bash
# Referee-driven additions to the experiment (run concurrently with overnight.sh):
# LR calibration for the FineWeb corpus; LR x{0.5,2} at the high-M and low-M corners (both corpora);
# seed replicates at the high-M corner (both) and FineWeb seeds; high-M runs with a 4-layer floor on extension data.
cd "$(dirname "$0")/../.."
LOG=data/processed/sweep/extras.log
PY=.venv/bin/python
run() { echo "=== $* start $(date)" >> $LOG; $PY code/sweep/run_grid.py "$@" 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG; echo "=== $* end $(date)" >> $LOG; }
run lrcal web
run lrcorner edu
run lrcorner web
run seedcorner edu
run seedcorner web
run seeds web
until [ -s data/processed/sweep/edu_train_ext.bin ] && [ -s data/processed/sweep/web_train_ext.bin ] && grep -q train_ext data/processed/sweep/meta.json; do sleep 30; done
run hiM edu
run hiM web
run hiM2 edu
echo "ALL DONE $(date)" >> $LOG
