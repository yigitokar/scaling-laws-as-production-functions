"""ra3common.py -- paths, constants and verified external parameters for module ra3_econ (economic implications).

Notation follows paper/notes/model_spec.md: C = 6ND, M = D/N, a = beta/(alpha+beta) (path slope, N* ~ C^a, D* ~ C^(1-a)),
gamma = alpha beta/(alpha+beta) (frontier elasticity), sigma* = 2/(2+alpha+beta) (on-path elasticity of substitution;
in the kappa family it uses the inner exponents), w = eps_N/eps_D (wedge), s = (w-1)/w (planned inference share of
lifetime compute).

Every external number below was checked against the primary source on 2026-09-24 (PDFs archived in
data/raw/ra3_econ_lit/ by code/data/download_ra3_econ.sh); the bib key is given with each constant.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ANALYSIS, "..", ".."))
M3_DIR = os.path.join(ANALYSIS, "m3_wedge")
for _p in (ANALYSIS, M3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aer_style  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
_OUT = os.environ.get("RA3_OUTPUT_ROOT", ROOT)
PROC = os.path.join(_OUT, "data", "processed", "ra3_econ")
TABLES = os.path.join(_OUT, "output", "tables")
FIGS = os.path.join(_OUT, "output", "figures")
for _d in (PROC, TABLES, FIGS):
    os.makedirs(_d, exist_ok=True)

P = "ra3_econ_"                      # file prefix
SEED = 20260924
B_WCB = 9999                         # wild cluster bootstrap draws (p-value floor 1/(B+1) = 1e-4)
N_MC = 20000                         # Monte Carlo draws for exhaustion dates
SNAPSHOT = "2026-09-23"              # Epoch AI models snapshot (code/data/download_public.sh)
T_NOW = 2026 + (266 / 365.25)        # 2026-09-24 as a decimal year
T0 = 2024.0                          # centering year for trends and the stock base year (Villalobos et al.)

EPOCH_ALL = os.path.join(RAW, "epoch_models", "all_ai_models.csv")
EPOCH_FRONTIER = os.path.join(RAW, "epoch_models", "frontier_ai_models.csv")
M3_MODELS = os.path.join(ROOT, "output", "tables", "m3_wedge_models.csv")
M1_REG = os.path.join(ROOT, "output", "tables", "technology_registry_m1.csv")
M2_REG = os.path.join(ROOT, "output", "tables", "technology_registry_m2.csv")
M1_PROC = os.path.join(ROOT, "data", "processed", "m1_chinchilla")
M2_PROC = os.path.join(ROOT, "data", "processed", "m2_techpanel")

# ----------------------------------------------------------------------------- verified external parameters
# Villalobos et al. (2024, ICML; key villalobos2022run), Sec. 2.1-2.5, Table 1, Fig. 3, Figs. 1/5/6:
#   deduplicated indexed-web stock 510T tokens [95%: 130T, 2100T] (2024); quality-adjusted (10-40% usable) 100T
#   [22T, 490T]; repetition-adjusted ("effective") 320T [65T, 1700T]; stock growth 0-10%/yr; historical dataset growth
#   0.38 OOM/yr [0.27, 0.48]; median full-utilisation year 2028 (very likely by 2032) at ~5e28 FLOP; with 5x
#   over-training one year earlier at ~6e27 FLOP; compute-optimal = 20 tokens/parameter (D = sqrt(20 C/6)).
VILLALOBOS = dict(U_q=100e12, U_q_lo=22e12, U_q_hi=490e12, U_eff=320e12, U_eff_lo=65e12, U_eff_hi=1700e12,
                  base_year=2024.0, g_lo=0.0, g_hi=0.10, g_mid=0.05, D_growth_oom=0.38, year_median=2028,
                  C_exhaust=5e28, C_exhaust_ot5=6e27)
# Muennighoff et al. (2023, NeurIPS; muennighoff2023scaling), Sec. 3.1 eqs. (5)-(6), App. A: fitted R*_D = 15.387756,
# R*_N = 5.309743 (joint fit); "only decay D" variant R*_D = 2.9157 (their Table 1); D' plateaus at U(1 + R*_D).
MUENN = dict(RD=15.387756, RN=5.309743, RD_only=2.9157)
# Epoch AI (Sevilla and Roldan 2024; sevilla2024training): notable models 2010-May 2024 4.1x/yr [90%: 3.7, 4.6];
# frontier (running top-10 at release) 5.3x/yr [4.9, 5.7] over 2010-2024 and 4.2x/yr [3.6, 4.9] since ~2018;
# top-10 language models mid-2020-May 2024 5.0x/yr [80%: 3.1, 7.3]. Summary: "4-5x/year".
EPOCH_GROWTH = [
    dict(sample="Notable models, 2010-May 2024", g=4.1, lo=3.7, hi=4.6, ci="90%"),
    dict(sample="Frontier (running top-10), 2010-May 2024", g=5.3, lo=4.9, hi=5.7, ci="90%"),
    dict(sample="Frontier (running top-10), ~2018-May 2024", g=4.2, lo=3.6, hi=4.9, ci="90%"),
    dict(sample="Top-10 language models, mid-2020-May 2024", g=5.0, lo=3.1, hi=7.3, ci="80%"),
]
# Ho et al. (2024; ho2024algorithmic): effective compute from algorithms doubles every ~8.4 months (bootstrap median;
# module m5 reproduces 8.44 and shows it is fragile, profile 95% CI [4.1, 40.5] months).
ALGO_DOUBLING_MONTHS = 8.4
# DeepSeek LLM (bi2024deepseek, eq. 4): M_opt = 0.1715 C^0.5243, D_opt = 5.8316 C^0.4757 (M = non-embedding FLOPs/token).
DEEPSEEK = dict(a=0.5243, D0=5.8316, d_exp=0.4757)
# Kaplan et al. (2020; kaplan2020scaling): N_opt ~ C^0.73, D ~ C^0.27 (early-stopped, pre-Chinchilla; historical only).
KAPLAN_A = 0.73

# External disclosures on the inference share of AI compute/energy (checked against the primary text):
DISCLOSURES = [
    dict(source="Patterson et al. (2022)", key="patterson2022carbon", firm="Google", period="2019-2021",
         metric="share of ML energy (one week each April; incl. research, development, testing)", inference_share=0.60,
         note="about 3/5 inference, 2/5 training in all three years; ML = 10-15% of Google energy", kind="disclosure"),
    dict(source="Wu et al. (2022)", key="wu2022sustainable", firm="Facebook/Meta", period="2019-2021",
         metric="AI power capacity, experimentation:training:inference = 10:20:70", inference_share=0.70,
         note="0.78 of training+inference if experimentation is excluded", kind="disclosure"),
    dict(source="Wu et al. (2022)", key="wu2022sustainable", firm="Facebook/Meta", period="2019-2021",
         metric="carbon footprint of the production language (translation) model LM", inference_share=0.65,
         note="recommendation models split about evenly", kind="disclosure"),
    dict(source="Nvidia Q4 FY2024 earnings call (21 Feb 2024)", key="nvidia2024q4call", firm="Nvidia (customers)",
         period="Feb 2023-Jan 2024", metric="share of data-center revenue for AI inference (hardware flow)",
         inference_share=0.40, note="CFO: 'approximately 40% of data center revenue was for AI inference'",
         kind="disclosure"),
    dict(source="You (2025), Epoch AI, from press reports", key="you2025openai", firm="OpenAI", period="2024",
         metric="inference / (inference + all R&D compute), cloud spend", inference_share=1.8 / (1.8 + 5.0),
         note="$1.8B inference vs ~$5B R&D compute; not a firm disclosure", kind="press-reported"),
    dict(source="You (2025), Epoch AI, from press reports", key="you2025openai", firm="OpenAI", period="2024",
         metric="inference / (inference + final training runs of released models)",
         inference_share=1.8 / (1.8 + 0.386 + 0.083), note="final runs ~$0.47B (GPT-4.5 $386M + others $83M)",
         kind="press-reported"),
]
# Final training runs as a share of R&D compute (Denain and Wu 2026, Epoch AI; denain2026final): OpenAI 9.6% (2024),
# Z.ai 12.3% (H2 2024-H1 2025), MiniMax 22.6% (Q4 2024-Q3 2025) -> R&D multiple rho = 1/share in [4.4, 10.4].
FINAL_RUN_SHARE = dict(OpenAI=0.096, Zai=0.123, MiniMax=0.226)

# Reddit, Inc., Form S-1 (Feb 2024; reddit2024s1): data-licensing arrangements entered in January 2024 with aggregate
# contract value $203.0 million and terms of two to three years.
REDDIT_LICENSING_USD = 203.0e6

# ----------------------------------------------------------------------------- helpers
DEV_MAP = {"Meta AI": "Meta", "Facebook AI Research": "Meta", "Facebook": "Meta", "Facebook AI": "Meta",
           "Google DeepMind": "Google", "DeepMind": "Google", "Google Brain": "Google", "Google Research": "Google",
           "Microsoft Research": "Microsoft", "Microsoft Research Asia": "Microsoft", "IBM Research": "IBM",
           "Cerebras Systems": "Cerebras", "Technology Innovation Institute": "TII", "Huawei Noah's Ark Lab": "Huawei",
           "Z.ai (Zhipu AI)": "Zhipu", "Zhipu AI": "Zhipu", "Alibaba Group": "Alibaba",
           "Allen Institute for AI": "AI2", "NVIDIA": "Nvidia", "Mistral": "Mistral AI"}   # copied from m3 analysis.py


def log(msg):
    print(f"[ra3_econ] {msg}", flush=True)


def webb_weights(rng, size):
    """Webb (2023) six-point weights: +-sqrt(1/2), +-1, +-sqrt(3/2) with equal probability."""
    pts = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    return pts[rng.integers(0, 6, size=size)]


def lognormal_from_ci(median, lo, hi, rng, n):
    """Draw from a lognormal with the given median and (approximately symmetric in logs) 95% interval."""
    sd = (np.log(hi) - np.log(lo)) / (2 * 1.959964)
    return np.exp(np.log(median) + sd * rng.standard_normal(n))
