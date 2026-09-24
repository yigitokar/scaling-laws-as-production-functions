"""common.py -- paths, data loaders and small helpers for module m1_chinchilla.

Data conventions (see memo, Section 2):
  * Chinchilla (Epoch digitization of Hoffmann et al. 2022, Fig. 4 left): N = total parameters incl. embeddings
    (Hoffmann convention), C = training FLOPs read off the figure, D = C/(6N) tokens (constructed, single epoch),
    L = MassiveText validation loss in nats/token (read from a 256-level colour map, +/-0.01 quantisation).
  * Llama 3 IsoFLOPs (Czech digitization of Grattafiori et al. 2024, Fig. 2): C = nominal budget, D = tokens read off
    the x-axis, N = C/(6D) (constructed), L = "validation loss" as plotted (units not stated in the report).
  * Marin 2026-03 IsoFLOPs (W&B export via open-athena): Llama-2 architecture, C = 6ND nominal budget,
    N and D from run configs, L = Paloma macro-average loss (nats/token, Llama-2 tokenizer).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

import sl  # noqa: E402  (shared library; never edited here)

RAW = os.path.join(ROOT, "data", "raw")
# M1_OUTPUT_ROOT (optional) redirects every output, e.g. for a --quick smoke test that must not overwrite results
_OUT = os.environ.get("M1_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "m1_chinchilla")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
MEMOS = os.path.join(_OUT, "output", "memos")
for _d in (PROC, TABLES, FIGS, MEMOS):
    os.makedirs(_d, exist_ok=True)

N_PROC = 6            # CPU processes (GPU is busy: never use MLX here)
SEED = 20260923       # master seed; every random draw in the module derives from it

CHIN_BUDGETS = np.array([6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21])
C_EVAL = (1e21, 5.76e23, 1e26)          # compute levels at which M* = D*/N* is reported
CHIN70B = (70e9, 1.4e12)                # Chinchilla-70B: N, D (Hoffmann et al. 2022)

HOFF_TEX = sl.HOFFMANN                  # Hoffmann A3, TeX precision (via Besiroglu et al.)
HOFF_ROUND = sl.Chinchilla(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
BESI_PUB = sl.BESIROGLU                 # Besiroglu et al. (2024) published point estimates


# ----------------------------------------------------------------------------- Chinchilla
def load_chinchilla(drop_worst=0):
    """All 245 digitized points (drop_worst=0) or the Besiroglu sample (drop_worst=5).

    Adds: reconstructed IsoFLOP budget (nearest of the 9 Chinchilla budgets after removing the common
    digitization offset of log10 C), `iso` = IsoFLOP-profile member flag, `cl` = cluster id for the
    cluster bootstrap, and `rank_worst` (1 = highest loss)."""
    df = sl.chinchilla_extraction(path=os.path.join(RAW, "epoch_chinchilla", "svg_extracted_data.csv"), drop_worst=0)
    lc = np.log10(df["C"].values)
    lb = np.log10(CHIN_BUDGETS)
    j = np.abs(lc[:, None] - lb[None, :]).argmin(1)
    dev = lc - lb[j]
    # Digitized C sits systematically left of the nominal budgets (see memo): estimate the common offset from
    # the points within 0.1 dex of a budget and define IsoFLOP membership as |dev - offset| <= 0.045 dex (~11%).
    offset = float(np.median(dev[np.abs(dev) < 0.1]))
    j = np.abs((lc - offset)[:, None] - lb[None, :]).argmin(1)
    dev = lc - offset - lb[j]
    df["budget"] = CHIN_BUDGETS[j]
    df["cl"] = j
    df["dev"] = dev
    df["iso"] = np.abs(dev) <= 0.045
    df["rank_worst"] = df["L"].rank(ascending=False, method="first").astype(int)
    df.attrs["c_offset_dex"] = offset
    df = df.sort_values("L").reset_index(drop=True)
    if drop_worst:
        df = df[df["rank_worst"] > drop_worst].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- IsoFLOP compilation
def load_isoflop(experiment):
    """Rows of open-athena/isoflop-experiments for one experiment, with cluster = budget."""
    df = pd.read_csv(os.path.join(RAW, "isoflop_experiments", "isoflop_experiments.csv"))
    df = df[df["experiment"] == experiment].copy()
    df = df.rename(columns={"params": "N", "tokens": "D", "loss": "L"})
    df["C"] = df["budget"]
    df["cl"] = pd.factorize(df["budget"], sort=True)[0]
    df["iso"] = True
    return df[["N", "D", "C", "L", "budget", "cl", "iso"]].sort_values(["budget", "N"]).reset_index(drop=True)


def load_llama3_raw():
    """Czech (2026) digitization of Llama 3 Fig. 2, used to cross-check the open-athena rows."""
    df = pd.read_csv(os.path.join(RAW, "llama3_isoflop", "isoflops_points.csv"))
    df = df.rename(columns={"compute_budget": "budget", "training_tokens": "D", "validation_loss": "L"})
    df["C"] = df["budget"]
    df["N"] = df["C"] / (6 * df["D"])
    return df


# ----------------------------------------------------------------------------- derived technology objects
def derived(m: sl.Chinchilla):
    """Dictionary of all objects reported for a technology (model_spec notation)."""
    d = dict(E=m.E, A=m.A, B=m.B, alpha=m.alpha, beta=m.beta, a=m.a_N, gamma=m.gamma, sigma_star=m.sigma_star,
             G=m.G, K=m.K, lnG=np.log(m.G), lnK=np.log(m.K))
    for C in C_EVAL:
        d[f"Mstar_{C:.3g}"] = float(m.D_opt(C) / m.N_opt(C))
    d["w_chin70b"] = float(m.wedge(*CHIN70B))
    return d


DERIVED_KEYS = ["E", "A", "B", "alpha", "beta", "a", "gamma", "sigma_star", "lnG", "lnK",
                "Mstar_1e+21", "Mstar_5.76e+23", "Mstar_1e+26", "w_chin70b"]


def theta_of(m):
    return np.array([np.log(m.A), np.log(m.B), np.log(m.E), m.alpha, m.beta])


def from_theta(th, **extra):
    return sl.Chinchilla.from_theta(np.asarray(th, float), **extra)


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{p}f}"
