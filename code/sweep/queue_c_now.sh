#!/usr/bin/env bash
# Queue C without waiting for queue B (restart after the 2026-09-25 03:17 reboot): small d=128 models run
# concurrently with queues A and B. Writes the same "QUEUE C DONE" marker that queue_d.sh waits for.
cd "$(dirname "$0")/../.."
LOG=data/processed/sweep/extras.log
for job in "hiMlr edu" "hiMwd edu"; do echo "=== $job start $(date)" >> $LOG; .venv/bin/python code/sweep/run_grid.py $job 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG; echo "=== $job end $(date)" >> $LOG; done
echo "QUEUE C DONE $(date)" >> $LOG
