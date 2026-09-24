Reviewer checks for module m5_progress (independent review, 2026-09-23). Not part of the pipeline: `run.py` does not
call them and they write no outputs (they print to stdout). Run after `run.py` with the project venv:

- `check_obj.py`: re-implements Ho et al.'s `model_7` + `residuals` from their notebook, without using `common.HoModel`,
  and evaluates the objective at the published and the converged parameter vectors (0.0518129 vs 0.0507227; MSE and
  gradient).
- `check_nit.py`: SLSQP iteration count at scipy's default stop (nit 18, nfev 209, as printed in Ho's notebook) and the
  MSE / L1 decomposition of the objective.
- `check_profile.py`: re-optimizes the 1-D profile likelihood at the CI boundaries with 60 global random starts, checks
  that the NLS minimum is global, and prints the E_b estimates vs their upper bounds in row A7 (6 processes, ~1 min).
- `check_tex.py`: brace/environment/math/column-count check of the LaTeX fragments (no TeX compiler on this machine).

See output/memos/m5_progress_review.md.
