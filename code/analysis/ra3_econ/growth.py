"""growth.py -- growth of frontier training compute, 2018-2026, from Epoch AI's model database (snapshot 2026-09-23).

Frontier definitions (all computed here from all_ai_models.csv):
  top10      running top-10 by training compute at release, all domains (Epoch's 'frontier' definition);  PRIMARY
  top10_nsp  same, excluding Epoch confidence 'Speculative'
  top10_lang running top-10 among language-domain models only
  record     record-setting runs (compute at or above the running maximum)
  epochflag  Epoch's own 'Frontier model' flag (ends 2025-07)
  top10_2024 top-10, 2018 to May 2024 (the window of Sevilla and Roldan's 4.2x/yr)
Regression: log10 C on (t - 2024), t = decimal release date. Inference: developer-clustered (first listed
organization, parent-mapped); CR1 standard errors; wild cluster bootstrap with Webb weights (B = 9,999):
unrestricted percentile-t confidence intervals (WCU) and restricted p-values (WCR) for H0: growth = 4.2x and 4.5x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ra3common import B_WCB, DEV_MAP, EPOCH_ALL, SEED, SNAPSHOT, T0, webb_weights

START = "2018-01-01"


def load_epoch():
    d = pd.read_csv(EPOCH_ALL, low_memory=False)
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d["C"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    d = d.dropna(subset=["date", "C"])
    d = d[(d["C"] > 0) & (d["date"] <= pd.Timestamp(SNAPSHOT))].sort_values(["date", "C"]).reset_index(drop=True)
    d["t"] = d["date"].dt.year + (d["date"].dt.dayofyear - 1) / 365.25
    d["lc"] = np.log10(d["C"])
    org = d["Organization"].fillna("unknown").astype(str).str.split(",").str[0].str.strip()
    d["dev"] = org.replace(DEV_MAP)
    d["lang"] = d["Domain"].fillna("").str.contains("Language")

    def running_top(mask):
        flag = np.zeros(len(d), bool)
        idx = np.where(mask)[0]
        lc, dt = d["lc"].values, d["date"].values
        for j, i in enumerate(idx):
            prev = idx[:j]
            prev = prev[dt[prev] <= dt[i]]
            flag[i] = (lc[prev] > lc[i]).sum() < 10
        return flag

    d["top10"] = running_top(np.ones(len(d), bool))
    d["top10_lang"] = running_top(d["lang"].values)
    run_max = np.maximum.accumulate(np.r_[-np.inf, d["lc"].values[:-1]])
    d["record"] = d["lc"].values >= run_max
    d["epochflag"] = d["Frontier model"].astype(str).str.lower().eq("true")
    return d


def samples(d):
    base = d[d["date"] >= START]
    return {
        "top10": base[base["top10"]],
        "top10_nsp": base[base["top10"] & (base["Confidence"] != "Speculative")],
        "top10_lang": base[base["top10_lang"]],
        "record": base[base["record"]],
        "epochflag": base[base["epochflag"]],
        "top10_2024": base[base["top10"] & (base["date"] <= "2024-05-31")],
    }


LABELS = {"top10": "Running top-10 at release (all domains)",
          "top10_nsp": "Running top-10, excl. 'Speculative' compute",
          "top10_lang": "Running top-10 among language models",
          "record": "Record-setting runs",
          "epochflag": "Epoch 'Frontier model' flag (to 2025-07)",
          "top10_2024": "Running top-10, 2018 to May 2024"}


def _cr1(X, e, g, G):
    """CR1 covariance for OLS; X (n,k), e (..., n), g cluster index (n,), returns (..., k, k)."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    Z = np.zeros((n, G))
    Z[np.arange(n), g] = 1.0
    S = np.einsum("...n,nk,ng->...gk", e, X, Z, optimize=True)   # cluster scores
    meat = np.einsum("...gk,...gl->...kl", S, S)
    c = G / (G - 1) * (n - 1) / (n - k)
    return c * XtXi @ meat @ XtXi


def wild_cluster(y, X, g, B=B_WCB, seed=SEED, nulls=()):
    """OLS with CR1 SEs; WCU percentile-t CI for the slope; WCR p-values for slope nulls; unrestricted coef draws."""
    rng = np.random.default_rng(seed)
    G = int(g.max()) + 1
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    V = _cr1(X, e, g, G)
    se = np.sqrt(np.diag(V))
    out = dict(b0=b[0], b1=b[1], se_b1=se[1], se_b0=se[0], G=G, n=len(y))
    # unrestricted (WCU): percentile-t CI and coefficient draws
    v = webb_weights(rng, (B, G))
    ys = (X @ b)[None, :] + e[None, :] * v[:, g]
    bs = ys @ (XtXi @ X.T).T
    es = ys - bs @ X.T
    Vs = _cr1(X, es, g, G)
    ts = (bs[:, 1] - b[1]) / np.sqrt(Vs[:, 1, 1])
    q_lo, q_hi = np.quantile(ts, [0.975, 0.025])
    out.update(b1_lo=b[1] - q_lo * se[1], b1_hi=b[1] - q_hi * se[1], draws=bs)
    # restricted (WCR) p-values for H0: slope = b1_0
    for name, b10 in nulls:
        yr = y - b10 * X[:, 1]
        b0r = yr.mean()
        er = yr - b0r
        vr = webb_weights(rng, (B, G))
        ysr = (b0r + b10 * X[:, 1])[None, :] + er[None, :] * vr[:, g]
        bsr = ysr @ (XtXi @ X.T).T
        esr = ysr - bsr @ X.T
        Vsr = _cr1(X, esr, g, G)
        tsr = (bsr[:, 1] - b10) / np.sqrt(Vsr[:, 1, 1])
        t_obs = (b[1] - b10) / se[1]
        out[f"p_{name}"] = (1 + np.sum(np.abs(tsr) >= abs(t_obs))) / (B + 1)
        out[f"t_{name}"] = t_obs
    return out


def fit_all(d):
    rows, draws = [], {}
    for j, (k, s) in enumerate(samples(d).items()):
        y = s["lc"].values
        X = np.c_[np.ones(len(s)), s["t"].values - T0]
        g = pd.factorize(s["dev"])[0]
        r = wild_cluster(y, X, g, seed=SEED + j, nulls=[("4.2x", np.log10(4.2)), ("4.5x", np.log10(4.5))])
        draws[k] = r.pop("draws")
        lvl = lambda t: 10 ** (r["b0"] + r["b1"] * (t - T0))  # noqa: E731
        rows.append(dict(sample=k, label=LABELS[k], n=r["n"], clusters=r["G"],
                         t_min=float(s["t"].min()), t_max=float(s["t"].max()),
                         growth=10 ** r["b1"], growth_lo=10 ** r["b1_lo"], growth_hi=10 ** r["b1_hi"],
                         oom_per_year=r["b1"], se_oom_cr1=r["se_b1"], p_wcr_4p2=r["p_4.2x"], p_wcr_4p5=r["p_4.5x"],
                         C_trend_2024=lvl(2024.0), C_trend_2026=lvl(2026.0), C_trend_now=lvl(2026 + 266 / 365.25),
                         C_max_obs=float(s["C"].max()), model_max_obs=s.loc[s["C"].idxmax(), "Model"],
                         b0=r["b0"], b1=r["b1"]))
    return pd.DataFrame(rows), draws
