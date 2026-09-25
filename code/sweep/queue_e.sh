#!/usr/bin/env bash
# Queue E (Amendment 2, X3(a)): FineWeb-Edu main-grid widths 384 and 512 retrained with all five evaluation sets,
# trunk evaluations and saved checkpoints. Approved by the author on 2026-09-25; runs concurrently with queues A-D.
cd "$(dirname "$0")/../.."
LOG=data/processed/sweep/queue_e.log
echo "=== edure start $(date)" >> $LOG
.venv/bin/python code/sweep/run_grid.py edure 2>&1 | grep --line-buffered -E "RUN \{|->|all done|Error|Traceback|nan" >> $LOG
echo "QUEUE E DONE $(date)" >> $LOG
