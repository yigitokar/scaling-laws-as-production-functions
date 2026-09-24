#!/usr/bin/env bash
# Restart both GPU queues after an interruption (completed endpoints are skipped by train_sweep.py).
# Queue A: remaining main web grid, then edu seeds.  Queue B (priority order): LR corners (web), high-M runs, seeds.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
qa() { LOG=data/processed/sweep/overnight.log; for job in "main web" "seeds edu"; do echo "=== $job start $(date)" >> $LOG; $PY code/sweep/run_grid.py $job 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG; echo "=== $job end $(date)" >> $LOG; done; echo "QUEUE A DONE $(date)" >> $LOG; }
qb() { LOG=data/processed/sweep/extras.log; for job in "lrcorner web" "hiM edu" "hiM web" "seedcorner edu" "seedcorner web" "seeds web"; do echo "=== $job start $(date)" >> $LOG; $PY code/sweep/run_grid.py $job 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG; echo "=== $job end $(date)" >> $LOG; done; echo "ALL DONE $(date)" >> $LOG; }
qa & qb & wait
