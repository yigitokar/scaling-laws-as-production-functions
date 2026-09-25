#!/usr/bin/env bash
# Queue D: after queues A and C finish, evaluate saved checkpoints on the neutral sets they were not trained with
# (main FineWeb grid d>=384 and the first high-M job; their job lists predated the evals change). See eval_ckpt.py.
cd "$(dirname "$0")/../.."
until grep -q "QUEUE A DONE" data/processed/sweep/overnight.log && grep -q "QUEUE C DONE" data/processed/sweep/extras.log; do sleep 120; done
LOG=data/processed/sweep/posthoc.log
echo "=== posthoc start $(date)" >> $LOG
cd code/sweep && ../../.venv/bin/python eval_ckpt.py >> ../../$LOG 2>&1
echo "QUEUE D DONE $(date)" >> ../../$LOG
