# Status (2026-09-24 ~08:00) — paused when the laptop lid closed

1. Paper v1 drafted and refereed (paper/referee/R1–R4). v2 rewrite in progress (plan: paper/notes/paper_plan_v2.md).
   Workflow rewrite-paper-v2 (run wf_acfe781f-db1): writers → integrator → round-2 referees (paper/referee/round2_*).
2. GPU experiment queues (resume automatically after wake): code/sweep/overnight.sh (log data/processed/sweep/overnight.log)
   and code/sweep/extras.sh (log extras.log). Check with: pgrep -fl "overnight.sh|extras.sh|train_sweep".
3. Next: m9 analysis per paper/notes/m9_preanalysis_plan.md → fill TBD-m9 placeholders → final referee pass → audit.

## Update 2026-09-24 ~09:30 (second pause)
- Edu main grid complete (44 endpoints); web grid running (overnight.sh); extras.sh running.
- Rewrite workflow wf_acfe781f-db1: 7 writers done; mcfix + appendix_D_F failed during the first sleep; integrator then referees round 2 were running.
- Background agents launched: Monte Carlo fix (m6 v2: figure m6_montecarlo_fig2_v2, appendix_mc.tex) and m9 pipeline build (power calc first; memo output/memos/m9_sweeps.md).
- To do on resume: check which of these finished (outputs above); re-run failures; run the Appendix D/F writer after the integrator; then m9 final run + review; fill TBD-m9; final referee pass + audit.

## Update 2026-09-24 15:45 (after a machine reboot at 15:34)
- GPU queues restarted with code/sweep/resume_queues.sh (queue A: main web → seeds edu; queue B: lrcorner web → hiM edu → hiM web → seedcorner → seeds web). Completed endpoints are skipped automatically; to restart after any interruption: `nohup code/sweep/resume_queues.sh >/dev/null 2>&1 &`.
- Workflow rewrite-paper-v2-continue (run wf_44d55abe-2cb; script in session scratchpad rewrite_v2b.js): MC fix + Appendix D/F + m9 pipeline build (power first) → integrator → referee round 2.
