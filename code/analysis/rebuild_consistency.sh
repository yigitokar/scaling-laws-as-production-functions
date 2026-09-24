#!/usr/bin/env bash
# Consistency rebuild after the sl.py E_fixed fix: m1 (uses E_fixed; writes registry) -> m3 and m5 (read registries).
cd "$(dirname "$0")/../.."
LOG=output/rebuild_consistency.log
echo "start $(date)" > $LOG
for m in m1_chinchilla m3_wedge m5_progress; do
  echo "=== $m $(date)" >> $LOG
  .venv/bin/python code/analysis/$m/run.py >> $LOG 2>&1; echo "exit $? $(date)" >> $LOG
done
echo "ALL DONE $(date)" >> $LOG
