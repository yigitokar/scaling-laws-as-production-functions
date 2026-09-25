"""rb3common.py -- paths, constants and helpers for module rb3_econ2 (round 3): integrating Section V (economics) with
Section IV (revealed value of compactness). Requests: R3 round-2 N6 (1)-(3) and minors 17-20; R1 round-2 minors 17-18;
R2 round-2 minor 14/20 and Major 1 (sigma* at frontier scale); R4 round-2 minors 9-10. Binding plan:
paper/notes/revision_plan_v3.md, Section 2 ("Section V integrates Section IV").

Builds on module ra3_econ (code/analysis/ra3_econ: growth.py, demand.py, wall.py, imported read-only; ra3's outputs are
never written) and on module rb2_decisions' reviewed clean sample and decision units (data/processed/rb2_decisions,
read-only). The shared library code/analysis/sl.py is not used or edited.

Notation (paper/notes/model_spec.md): C = 6ND; M = D/N; a = path slope (N* ~ C^a, D* ~ C^(1-a)); gamma = frontier
elasticity of reducible loss; sigma* = on-path elasticity of substitution; k = 1/sigma* - 1 = S/2 (ln w = k ln(M/M*));
w = eps_N/eps_D (wedge); m_N = w - 1 (value of compactness per unit of training cost); s = (w - 1)/w; D is tokens
processed, U unique tokens; r = D*(C)/U (scarcity at compute-optimal use); r_obs = D/U (scarcity at the observed
allocation). The data multiple of a wedge at given compute is D/D*(C) = w^e(sigma*), e = sigma*/[2(1 - sigma*)].
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
RA3 = os.path.join(ANALYSIS, "ra3_econ")
_OUT = os.environ.get("RB3_OUTPUT_ROOT", ROOT)
# ra3's modules write nothing at import time except (exist_ok) directory creation; point their output root at a scratch
# folder inside this module's processed directory so that nothing of ra3's is ever touched.
os.environ.setdefault("RA3_OUTPUT_ROOT", os.path.join(_OUT, "data", "processed", "rb3_econ2", "ra3_sandbox"))
for _p in (ANALYSIS, RA3):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

import aer_style  # noqa: E402,F401

P = "rb3_econ2_"
SEED = 20260926
B_WCB = 9999                        # wild cluster bootstrap draws (Webb weights)
N_PROC = 4                          # CPU budget (GPU queues are running); the module runs single-process
RAW = os.path.join(ROOT, "data", "raw", "rb3_econ2")
PROC = os.path.join(_OUT, "data", "processed", "rb3_econ2")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
for _d in (PROC, TABLES, FIGS):
    os.makedirs(_d, exist_ok=True)

RB2_PROC = os.path.join(ROOT, "data", "processed", "rb2_decisions")
RB2_MODELS = os.path.join(RB2_PROC, "clean_models.csv")            # 77 clean models, token audit applied, w_ref
# Round 3 (fix list D-2, E1): the primary decision units are the budget-level units of module rb5_units (members of a
# family whose token budgets are within 10 percent form one decision; 49 units). trend.py builds the unit table from
# the model-level file RB5_UNITS (unit, W_unit, C_unit, unit_year; the unit's date is its first member's) and checks it
# against rb2_decisions' unit table RB2_UNITS, which rb2 writes from the same partition.
RB5_UNITS = os.path.join(ROOT, "data", "processed", "rb5_units", "units_primary.csv")
RB2_UNITS = os.path.join(RB2_PROC, "decision_units.csv")           # 49 budget-level units (rb2's copy; check only)
RB2_UNITS56 = os.path.join(RB2_PROC, "decision_units_family56.csv")  # version 3's 56 family-label units (robustness)
# rb2's exhibit cache: the long table of wedges under every technology on the audited token counts (L) and the
# registry (M). Used for the 32-technology trend (round 3, fix list E6: audited counts instead of ra2's inputs).
RB2_CACHE = os.path.join(RB2_PROC, "exhibits_cache.pkl")
RB2_TREND = os.path.join(ROOT, "output", "tables", "rb2_decisions_trend_by_tech.csv")
RA2_MODELS = os.path.join(ROOT, "output", "tables", "ra2_wedge_models.csv")
RA2_TECHS = os.path.join(ROOT, "output", "tables", "ra2_wedge_technologies.csv")
M3_MODELS = os.path.join(ROOT, "output", "tables", "m3_wedge_models.csv")
RB1_STUDY = os.path.join(ROOT, "output", "tables", "rb1_sigmaC_study_level.csv")
RB1_PRED = os.path.join(ROOT, "output", "tables", "rb1_sigmaC_metareg_predictions.csv")
# Round 3 (fix list E1(a); audit numbers_technology #7): per-budget model-free IsoFLOP minima of module rb4_chinflop,
# which re-estimates Chinchilla in FLOP-effective parameters (N_F, the primary convention since rb4); the other designs
# are ra1's minima unchanged. Used for the model-free path slopes (Chinchilla a = 0.489, was 0.502 in total parameters)
# and for the E-profiles of gamma.py. ra1's total-parameter file is kept for the convention comparison.
RA1_BUDGETS = os.path.join(ROOT, "data", "processed", "rb4_chinflop", "override_new_isoflop_budgets.csv")
RA1_BUDGETS_T = os.path.join(ROOT, "output", "tables", "ra1_modelfree_isoflop_budgets.csv")
M1_REG = os.path.join(ROOT, "output", "tables", "technology_registry_m1.csv")
M2_REG = os.path.join(ROOT, "output", "tables", "technology_registry_m2.csv")

# Epoch AI database vintages. Current: snapshot 2026-09-23 (code/data/download_public.sh; used by ra3). Archived:
# Wayback Machine captures of Epoch's public CSVs (code/data/download_rb3_econ2.sh; sha256 recorded there).
EPOCH_ALL_2026 = os.path.join(ROOT, "data", "raw", "epoch_models", "all_ai_models.csv")
EPOCH_ALL_2024 = os.path.join(RAW, "epoch_all_systems_20240531.csv")   # capture 2024-05-31 03:46 UTC
SNAPSHOT_2024 = "2024-05-31"
SNAPSHOT_2026 = "2026-09-23"
EPOCH_WINDOW_END = "2024-05-31"   # Sevilla and Roldan (2024), posted 28 May 2024: '4.2x/year (90% CI 3.6-4.9) after 2018'

# scenario constants
SIGMAS = (0.60, 0.70, 0.74)       # curvature sensitivity (task; R3 N6(1)): top budgets / reference / kappa = 1 refit
# The 2025 compute-weighted wedge of the decision units, rounded to two decimals, is the high-wedge point of the wall
# grids (wallobs.py) and exhibits. run.py sets it from trend.py's by-year table before the wall is solved (round 3: 5.38
# on the 49 budget-level units; 5.39 on version 3's 56 units, the default here).
W_AGG25 = 5.39
G_EPOCH = 4.2                     # Epoch's published frontier growth for ~2018-May 2024 (sevilla2024training)
T_NOW = 2026 + (266 / 365.25)     # 2026-09-24
T0 = 2024.0
HORIZON = 5.0                     # years
LIFE_YEARS = (1.0, 2.0, 3.0)      # service lives for the fleet comparison (R3 N6(3), minor 19)
RHO = (1.0, 4.4, 10.4)            # R&D multiples: none; final runs 22.6% and 9.6% of R&D compute (denain2026final)
U_STOCK = 100e12                  # quality-adjusted stock of public text in 2024 (villalobos2022run), 95%: 22T-490T
U_STOCK_LO, U_STOCK_HI = 22e12, 490e12
U_GROWTH = 0.05

# verified external prices (optional comparison; data/raw/rb3_econ2/deepseek_pricing_20260924.html, retrieved
# 2026-09-24 17:15 UTC): DeepSeek API, per 1M output tokens: deepseek-flash (DeepSeek-V4.1-Flash) $0.60 off-peak /
# $1.20 peak; deepseek-v4-pro (DeepSeek-V4-Pro-0813) $1.98 off-peak / $3.96 peak. Key: deepseek2026pricing.
API_OUTPUT_USD_PER_MTOK = dict(flash_offpeak=0.60, flash_peak=1.20, pro_offpeak=1.98, pro_peak=3.96)
# Reddit, Inc. Form S-1 (reddit2024s1): 'over one billion posts and over 16 billion comments through December 31,
# 2023'; data-licensing arrangements with aggregate contract value $203.0 million over two to three years (ra3).
REDDIT = dict(usd=203.0e6, items=17e9, years_lo=2.0, years_hi=3.0)


def log(msg):
    print(f"[rb3_econ2] {msg}", flush=True)


def dec_year(dates):
    d = pd.to_datetime(dates)
    return (d.dt.year + (d.dt.dayofyear - 1) / 365.25).astype(float)


def e_of(sigma):
    """Elasticity of D/D*(C) with respect to the wedge at given compute: sigma / (2 (1 - sigma))."""
    return sigma / (2.0 * (1.0 - sigma))


def k_of(sigma):
    return 1.0 / sigma - 1.0


def webb(rng, size):
    pts = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    return pts[rng.integers(0, 6, size=size)]


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:,.{d}f}"
