#!/usr/bin/env bash
# Overnight chain: wait for LR calibration -> fit LR rule -> main grids (edu, web) -> seed replicates.
cd "$(dirname "$0")/../.."
LOG=data/processed/sweep/overnight.log
while pgrep -f "run_grid.py lrsweep" >/dev/null; do sleep 20; done
echo "lrsweep finished $(date)" >> $LOG
.venv/bin/python code/sweep/fit_lr.py >> $LOG 2>&1
for job in "main edu" "main web" "seeds edu"; do
  echo "=== $job start $(date)" >> $LOG
  .venv/bin/python code/sweep/run_grid.py $job 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG
  echo "=== $job end $(date)" >> $LOG
done
echo "ALL DONE $(date)" >> $LOG
