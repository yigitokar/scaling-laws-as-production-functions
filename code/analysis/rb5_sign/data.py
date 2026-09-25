"""data.py -- inputs of module rb5_sign (all read-only).

Models: the clean sample of rb2_decisions (77 dense open-weight base models, token audit applied), a parameter.
Units: decision units, a parameter (default [review]: WP4b's final 49 budget-level units,
data/processed/rb5_units/units_primary.csv, the same partition as rb2's units_subgroup.csv; robustness: the 56
family-label units, rb2's units_primary.csv).
IsoFLOP minima: ra1_modelfree's reviewed per-budget minima at bracketed budgets (Chinchilla in total parameters as
digitized; Llama 3 and Marin in the FLOP-implied convention N_F = C/(6D)); rb4_chinflop's Chinchilla minima in N_F
(variant NF_T4, Hoffmann et al.'s implemented count). Marin's configuration count N_cfg (a total count) from the
open-athena compilation, for the conversion of Marin's minima to total parameters.
Technologies: ra2's registry as cached by rb2_decisions (DeepSeek's published law; the in-set technologies' anchors
for the every-technology variant).
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

import rb5common as K

MARIN = {"Marin, Comma": "marin_202603__comma__llama_2", "Marin, DCLM": "marin_202603__dclm__llama_2",
         "Marin, Nemotron-CC": "marin_202603__nemotron__llama_2"}


def load_models(path=K.DEFAULT_MODELS):
    B = pd.read_csv(path)
    need = ["uid", "model", "dev", "year", "N", "D", "M", "Cmp", "N_nonemb", "N_head", "n_layer", "d_model", "gen",
            "Mstar_ref", "w_ref"]
    miss = [c for c in need if c not in B.columns]
    if miss:
        raise ValueError(f"models file lacks columns {miss}")
    B = B.copy()
    B["M"] = B["D"] / B["N"]
    B["Cmp"] = 6 * B["N"] * B["D"]
    # FLOP-effective count of released models (all-N_F variant only): Hoffmann et al.'s accounting at n_ctx tokens
    B["N_F"] = B["N_nonemb"] + 2 * B["N_head"] + 2 * B["n_layer"] * B["d_model"] * K.N_CTX_MODELS
    return B.reset_index(drop=True)


def load_units(path, B):
    U = pd.read_csv(path)
    if "unit" not in U.columns or "uid" not in U.columns:
        raise ValueError(f"units file {path} needs columns uid, unit")
    if "unit_type" not in U.columns:
        U["unit_type"] = np.where(U.groupby("unit")["uid"].transform("count") > 1, "family", "model")
    missing = set(B["uid"]) - set(U["uid"])
    extra = set(U["uid"]) - set(B["uid"])
    if extra:
        raise ValueError(f"units file {path} has {len(extra)} uids not in the models file")
    return U, sorted(missing)


def load_techs(path=K.TECHS_CACHE):
    with open(path, "rb") as f:
        return pickle.load(f)


def minima(design, source="ra1"):
    """Per-budget minima (budget C, ln C, ln M*, N*) of one design at its bracketed budgets."""
    if source == "ra1":
        b = pd.read_csv(K.RA1_BUDGETS)
        g = b[b["design"] == design].sort_values("budget_C")
    elif source == "rb4_nf":
        b = pd.read_csv(K.RB4_BUDGETS)
        g = b[b["variant"] == "NF_T4"].sort_values("budget_C")
        assert design == "Chinchilla"
    else:
        raise ValueError(source)
    if not len(g):
        raise ValueError(f"no minima for {design} ({source})")
    return pd.DataFrame(dict(design=design, C=g["budget_C"].values, lnC=np.log(g["budget_C"].values),
                             lnM=np.log(g["Mstar"].values), Nstar=g["Nstar"].values))


def marin_ratio_curve():
    """Marin's ratio r = N_F/N_cfg = C_b/(6 N_cfg D) by configuration (constant across budgets to 1e-5); returns
    (ln N_F, ln r) sorted, one point per configuration (median over budgets and corpora)."""
    raw = pd.read_csv(K.RAW_ISO)
    m = raw[raw["experiment"].isin(MARIN.values())].copy()
    m["r"] = m["budget"] / (6 * m["tokens"] * m["params"])
    g = m.groupby("params")["r"].median().reset_index()
    lnNF = np.log(g["params"].values * g["r"].values)
    o = np.argsort(lnNF)
    return lnNF[o], np.log(g["r"].values[o])


def marin_minima_total(design):
    """Marin minima converted to total (configuration) parameters: the optimal run at budget b has N*_cfg = N*_F / r
    and the same tokens, so ln M*_T = ln M*_F + ln r(N*_F), and its compute counted as 6 N_cfg D is C_b / r."""
    g = minima(design)
    x, lr = marin_ratio_curve()
    lnr = np.interp(np.log(g["Nstar"].values), x, lr)
    g = g.copy()
    g["lnr"] = lnr
    g["lnM"] = g["lnM"] + lnr
    g["lnC"] = g["lnC"] - lnr
    g["C"] = np.exp(g["lnC"])
    return g


def llama3_conversion():
    """Bound on ln(N_F / N_total) at Llama 3's anchor for a Llama-3-shaped model of the anchor's size (Llama 3.2 3B:
    d = 3072, 28 layers, 24 query and 8 key-value heads of dimension 128, feed-forward width 8192, vocabulary 128,256)
    under three accountings of the budget and tied or untied input embeddings. Returns a DataFrame and (lo, hi)."""
    d, L, V, kv, ff, nctx = 3072, 28, 128256, 1024, 8192, 8192
    n_nonemb = L * (2 * d * d + 2 * d * kv + 3 * d * ff)
    emb = V * d
    rows = []
    for tied in (True, False):
        n_tot = n_nonemb + emb * (1 if tied else 2)
        for acc, nf in [("6 N_total D", n_tot), ("non-embedding only (Kaplan)", n_nonemb),
                        ("non-embedding + head + attention at 8,192 tokens", n_nonemb + emb + L * nctx * d)]:
            rows.append(dict(embeddings="tied" if tied else "untied", accounting=acc, N_total=n_tot, N_F=nf,
                             ratio_NF_over_Ntotal=nf / n_tot, ln_ratio=np.log(nf / n_tot)))
    T = pd.DataFrame(rows)
    return T, (float(T["ln_ratio"].min()), float(T["ln_ratio"].max()))
