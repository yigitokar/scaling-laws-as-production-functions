"""labown.py -- lab-own technologies under ONE rule (R2 Major 6): the lab's own expansion path plus the common
model-free curvature sigma* (rb1_sigmaC's study-level value if present, else ra1's 0.695); lab-specific curvatures,
Meta's 10-budget planning law and AI2's kappa = 1 ladder are sensitivity rows.

Paths (primary):
  Meta     -- Llama 3 IsoFLOP minima on the 8 budgets whose minimum is bracketed (ra1; 6e18-1e21); wild bootstrap-t.
              Sensitivity: the 10-budget path that reproduces Meta's published planning law (ra2 'meta_mf').
  Marin    -- Nemotron-CC ladder minima, 7 bracketed budgets (ra1); wild bootstrap-t.
  AI2      -- the OLMo ladder's expansion path under kappa free (m2 draws; a = b1/(a1 + b1)); kappa = 1 as sensitivity.
  DeepSeek -- the published allocation law (Bi et al. 2024, Eq. 4; non-embedding FLOPs per token); no path draws.
Wedge: ln w = k [ln M - ln M*(C)], k = 1/sigma* - 1, in each path's own parameter convention.
Joint draws: path draws (bootstrap-t adjusted, or m2 pairs draws for AI2) x independent normal curvature draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb2common as C
import paths as P
import wedges as WG  # ra2

LABS = {"Meta": "meta", "Marin": "marin", "AI2": "ai2", "DeepSeek": "deepseek"}
LAB_SPECIFIC_SIGMA = {"meta": ("ra1 Llama 3 model-free", 0.6595), "marin": ("ra1 Marin Nemotron-CC model-free", 0.7049),
                      "ai2": ("OLMo ladder kappa free", None), "deepseek": ("reference (DeepSeek published no curvature)", None)}


def _k_draws(sig, se, B, seed, df=None):
    """Curvature draws k = 1/sigma - 1. Review fix: rb1's study-level interval is HKSJ, i.e. mean +/- t_{k-1} x se with
    k = 4 studies (t_3 = 3.18); normal draws with that se gave a 95% range of +/-0.025 instead of rb1's +/-0.040. With
    df given, sigma = mean + se x t_df, which reproduces rb1's interval; normal draws only for ra1's fallback."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(B) if df is None else rng.standard_t(df, B)
    s = sig + se * z
    return 1 / s - 1


def build(X, B=9999):
    """Returns (per-model table for lab-own models, draws dict uid -> (B,) ln w draws, sensitivity rows, meta)."""
    Bf, T = X["B"], X["T"]
    sg = C.common_sigma()
    k0 = 1 / sg["sigma"] - 1
    kd = _k_draws(sg["sigma"], sg["se"], B, C.SEED + 11, sg.get("df"))
    meta8 = P.Path(P.ra1_minima("Llama 3"), B=B, seed=C.SEED + 21, label="Meta, 8 bracketed budgets")
    marin = P.Path(P.ra1_minima("Marin, Nemotron-CC"), B=B, seed=C.SEED + 22, label="Marin Nemotron-CC, 7 budgets")
    olq, ol1 = T["olmo_q"], T["olmo"]
    ds = T["deepseek"]
    m10 = T["meta_mf"]                                     # 10-budget path (Meta's published law) + ra1 curvature
    rows, draws = [], {}
    X_ = Bf[Bf["clean"] & Bf["dev"].isin(LABS)].copy()
    for _, r in X_.iterrows():
        lab = LABS[r["dev"]]
        one = pd.DataFrame([r])
        D = r["D"]
        if lab in ("meta", "marin"):
            path = meta8 if lab == "meta" else marin
            N = r["N"]
            lnC = np.log(6 * N * D)
            lnM = np.log(D / N)
            lms = float(path.lnMstar(lnC)[0])
            lms_d = path.adjusted_draws(lnC)[:, 0]
            conv = "total"
        elif lab == "ai2":
            N = WG.n_used(one, "olmo")[0]
            lnC = np.log(6 * N * D)
            lnM = np.log(D / N)
            lms = float(C.ln_mstar(np.exp(lnC), olq.alpha, olq.beta, olq.lnG))
            dr = olq.draws
            lms_all = C.ln_mstar(np.exp(lnC), dr[:, 0], dr[:, 1], dr[:, 2])
            lms_d = lms_all[np.arange(B) % len(lms_all)]
            conv = "olmo"
        else:
            N = WG.n_used(one, "ds")[0]
            lnC = np.log(6 * N * D)
            lnM = np.log(D / N)
            lms = float(C.ln_mstar(np.exp(lnC), ds.alpha, ds.beta, ds.lnG))
            lms_d = np.full(B, lms)
            conv = "ds"
        lw = k0 * (lnM - lms)
        lwd = kd * (lnM - lms_d)
        draws[r["uid"]] = lwd
        # sensitivities (points)
        sens = {}
        ks = LAB_SPECIFIC_SIGMA[lab][1]
        if lab == "ai2":
            ks = olq.sigma_star
        if ks is not None:
            sens["lab-specific curvature"] = np.exp((1 / ks - 1) * (lnM - lms))
        if lab == "meta":
            Nt = r["N"]
            sens["Meta 10-budget planning law"] = np.exp(k0 * (np.log(D / Nt) - C.ln_mstar(6 * Nt * D, m10.alpha, m10.beta, m10.lnG)))
            sens["Meta 10-budget law, Meta curvature (ra2 primary)"] = np.exp(
                (m10.alpha + m10.beta) / 2 * (np.log(D / Nt) - C.ln_mstar(6 * Nt * D, m10.alpha, m10.beta, m10.lnG)))
        if lab == "ai2":
            sens["OLMo ladder kappa = 1 path"] = np.exp(k0 * (lnM - C.ln_mstar(np.exp(lnC), ol1.alpha, ol1.beta, ol1.lnG)))
            sens["OLMo ladder kappa free, own curvature (ra2 primary)"] = np.exp(
                (olq.alpha + olq.beta) / 2 * (lnM - lms))
        rows.append(dict(uid=r["uid"], model=r["model"], dev=r["dev"], lab_path=lab, conv=conv, M_conv=np.exp(lnM),
                         Mstar_lab=np.exp(lms), w_lab1=np.exp(lw), w_lab1_lo=np.exp(C.pct(lwd, 2.5)),
                         w_lab1_hi=np.exp(C.pct(lwd, 97.5)), s_lab1=C.share(np.exp(lw)),
                         **{f"w_sens_{k}": float(v) for k, v in sens.items()}))
    tabl = pd.DataFrame(rows)
    meta = dict(sigma=sg, k=k0, meta8_a_lnG=meta8.a_lnG(), meta8_Mstar_1e21=float(np.exp(meta8.beta[0])),
                marin_a_lnG=marin.a_lnG(), B=B)
    return tabl, draws, meta
