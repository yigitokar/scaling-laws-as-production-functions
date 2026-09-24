"""boot.py -- deterministic, parallel (6-process) bootstrap machinery.

Draws are index arrays generated in the parent from np.random.default_rng(seed); workers only fit.
Schemes:
  pairs    i.i.d. resampling of runs (Efron)
  cluster  resampling of whole IsoFLOP budgets (9 clusters for Chinchilla; few-cluster caveat, Cameron et al. 2008)
  strat    resampling of runs within each budget (keeps the design; used for Approach-2 objects)
Every replication is warm-started from a short list of candidate optima (full-sample estimate, Hoffmann, Besiroglu):
the grid search of the full-sample fit is not repeated (the module's time budget does not allow it). The effect of
this shortcut is checked in review_checks.warmstart_check (40 pairs + 40 cluster draws re-fitted with the 432-start
FAST_GRID added; output m1_chinchilla_warmstart_check.csv).
"""
from __future__ import annotations

import os
import warnings
from multiprocessing import get_context

import numpy as np

from common import N_PROC

_G = {}


def draws(n_or_groups, B, scheme, seed):
    """Return a list of B index arrays."""
    rng = np.random.default_rng(seed)
    out = []
    if scheme == "pairs":
        n = n_or_groups
        for _ in range(B):
            out.append(rng.integers(0, n, n))
    elif scheme == "cluster":
        groups = n_or_groups
        for _ in range(B):
            pick = rng.integers(0, len(groups), len(groups))
            out.append(np.concatenate([groups[i] for i in pick]))
    elif scheme == "strat":
        groups = n_or_groups
        for _ in range(B):
            out.append(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]))
    else:
        raise ValueError(scheme)
    return out


def groups_of(labels):
    labels = np.asarray(labels)
    return [np.where(labels == g)[0] for g in np.unique(labels)]


def _init(payload):
    warnings.filterwarnings("ignore")
    _G.update(payload)


def _work(args):
    fn, idx = args
    warnings.filterwarnings("ignore")
    try:
        return fn(idx, **_G)
    except Exception as e:  # a failed replication is recorded, never silently dropped
        return {"_error": repr(e)}


def pmap(fn, idx_list, payload, procs=N_PROC, chunksize=4):
    """Apply fn(idx, **payload) to every index array, in `procs` spawn-started processes."""
    ctx = get_context("spawn")
    with ctx.Pool(procs, initializer=_init, initargs=(payload,)) as pool:
        return pool.map(_work, [(fn, idx) for idx in idx_list], chunksize=chunksize)


def summarize(rows, keys):
    """Bootstrap SD, percentile 95% CI, number of valid draws."""
    out = {}
    ok = [r for r in rows if "_error" not in r]
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in ok], float)
        v = v[np.isfinite(v)]
        if len(v) < 5:
            out[k] = dict(se=np.nan, lo=np.nan, hi=np.nan, rse=np.nan, nvalid=len(v))
            continue
        out[k] = dict(se=float(np.std(v, ddof=1)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                      rse=float((np.percentile(v, 75) - np.percentile(v, 25)) / 1.349),
                      nvalid=int(len(v)), sd_log=float(np.std(np.log(v), ddof=1)) if np.all(v > 0) else np.nan)
    out["_nfail"] = len(rows) - len(ok)
    return out


def rse(a, axis=0):
    """Robust (IQR-based) bootstrap standard error: (q75 - q25)/1.349; equals the SD under normality."""
    return (np.percentile(a, 75, axis=axis) - np.percentile(a, 25, axis=axis)) / 1.349


def mat(rows, keys):
    ok = [r for r in rows if "_error" not in r]
    return np.array([[r.get(k, np.nan) for k in keys] for r in ok], float)


os.environ.setdefault("OMP_NUM_THREADS", "1")
