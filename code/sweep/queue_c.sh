#!/usr/bin/env bash
# Queue C (round-2 referee additions): starts after queue B finishes.
cd "$(dirname "$0")/../.."
LOG=data/processed/sweep/extras.log
until grep -q "ALL DONE" $LOG; do sleep 60; done
for job in "hiMlr edu" "hiMwd edu"; do echo "=== $job start $(date)" >> $LOG; .venv/bin/python code/sweep/run_grid.py $job 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG; echo "=== $job end $(date)" >> $LOG; done
echo "QUEUE C DONE $(date)" >> $LOG
