"""meta.py -- heterogeneity of sigma* across technologies (Task 3; R1 c6(a), R3 M6.2).

Cochran (1954) Q test of homogeneity, DerSimonian and Laird (1986) random-effects mean and between-technology SD tau,
Higgins and Thompson (2002) I^2, and the Hartung-Knapp (2001) / Sidik-Jonkman (2002) small-k interval (t_{k-1} with the
HKSJ variance). Inputs: point estimates and bootstrap standard errors (treated as known within-study variances).
Leave-one-out and 'five studies' (Gadre's three corpora combined by inverse variance) variants are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t as tdist


def meta(y, se):
    y, se = np.asarray(y, float), np.asarray(se, float)
    v = se ** 2
    k = len(y)
    w = 1 / v
    mu_f = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - mu_f) ** 2))
    df = k - 1
    tau2 = max(0.0, (Q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se_re = float(np.sqrt(1 / np.sum(ws)))
    q_hk = float(np.sum(ws * (y - mu) ** 2) / df) if df > 0 else np.nan
    se_hk = float(np.sqrt(max(q_hk, 1.0) / np.sum(ws))) if df > 0 else np.nan   # 'modified' HKSJ (q >= 1)
    tq = tdist.ppf(0.975, df) if df > 0 else np.nan
    return dict(k=k, mu_fixed=mu_f, se_fixed=float(np.sqrt(1 / np.sum(w))), Q=Q, df=df, p_Q=float(chi2.sf(Q, df)),
                I2=float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0, tau2=float(tau2), tau=float(np.sqrt(tau2)),
                mu_re=mu, se_re=se_re, lo_re=mu - 1.96 * se_re, hi_re=mu + 1.96 * se_re,
                se_hksj=se_hk, lo_hksj=mu - tq * se_hk, hi_hksj=mu + tq * se_hk,
                pi_lo=mu - tdist.ppf(0.975, max(k - 2, 1)) * np.sqrt(tau2 + se_re ** 2),
                pi_hi=mu + tdist.ppf(0.975, max(k - 2, 1)) * np.sqrt(tau2 + se_re ** 2))


def pairwise_z(y, se, names):
    rows = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            z = (y[i] - y[j]) / np.sqrt(se[i] ** 2 + se[j] ** 2)
            rows.append(dict(a=names[i], b=names[j], diff=y[i] - y[j], z=z, p=2 * norm.sf(abs(z))))
    return pd.DataFrame(rows)


def run_set(label, df, ycol, secol, study_col=None):
    """Full-set meta-analysis plus leave-one-out and (if study_col) study-level aggregation."""
    y, se = df[ycol].values, df[secol].values
    out = [dict(set=label, variant="all technologies", members="; ".join(df["technology"]), **meta(y, se))]
    for i in range(len(df)):
        m = np.ones(len(df), bool)
        m[i] = False
        out.append(dict(set=label, variant=f"leave out {df['technology'].iloc[i]}",
                        members="; ".join(df["technology"][m]), **meta(y[m], se[m])))
    if study_col is not None:
        g = df.groupby(study_col, sort=False)
        ys, ses, names = [], [], []
        for s, d in g:
            w = 1 / d[secol].values ** 2
            ys.append(float(np.sum(w * d[ycol].values) / np.sum(w)))
            ses.append(float(np.sqrt(1 / np.sum(w))))
            names.append(s)
        out.append(dict(set=label, variant="one estimate per study (inverse-variance within study)",
                        members="; ".join(names), **meta(np.array(ys), np.array(ses))))
    return pd.DataFrame(out)


SWEEP_KEYS = {"chinchilla|all": ("Chinchilla", "Hoffmann et al. (2022)"), "farseer|all": ("Farseer", "Li et al. (2025)"),
              "gadre|C4": ("Gadre, C4", "Gadre et al. (2025)"), "gadre|RedPajama": ("Gadre, RedPajama", "Gadre et al. (2025)"),
              "gadre|RefinedWeb": ("Gadre, RefinedWeb", "Gadre et al. (2025)"),
              "olmo_ladder|all": ("OLMo ladder", "Bhagia et al. (2024)"),
              "datablations|single_epoch": ("Muennighoff", "Muennighoff et al. (2023)")}


def run_all(iso, far, kap):
    import os
    import ra1_common as rc
    t3 = pd.read_csv(os.path.join(rc.ROOT, "output", "tables", "m2_table3_technology.csv"))
    t3 = t3[(t3.estimator == "huber") & t3.key.isin(SWEEP_KEYS)].copy()
    t3["technology"] = t3.key.map(lambda k: SWEEP_KEYS[k][0])
    t3["study"] = t3.key.map(lambda k: SWEEP_KEYS[k][1])
    t3 = t3.set_index("technology").loc[[v[0] for v in SWEEP_KEYS.values()]].reset_index()
    ktab, _ = kap
    kb = ktab[ktab.variant == "baseline"].set_index("tech")
    t3["sigma_kappa_ra1"] = t3.technology.map(kb["sigma_kappa"])
    t3["se_sigma_kappa_wild"] = t3.technology.map(kb["se_sigma_kappa"])
    sets = []
    sets.append(run_set("sigma*_kappa, seven sweep-corpus technologies (m2 Table 4; pairs/cell bootstrap SEs)",
                        t3, "sigma_star_q", "se_sigma_star_q", "study"))
    sets.append(run_set("sigma*_kappa, seven sweep-corpus technologies (design-conditional wild SEs)",
                        t3, "sigma_kappa_ra1", "se_sigma_kappa_wild", "study"))
    sets.append(run_set("sigma* under kappa = 1 (Chinchilla form), seven technologies (m2 Table 4 SEs)",
                        t3, "sigma_star", "se_sigma_star", "study"))
    # model-free designs: per-design random-effects summary across budgets (DL), Farseer: mean over the local path
    s = iso["summary"].copy()
    mf = pd.DataFrame(dict(technology=s.design, study=s.meta_study, y=s.sigma_re, se=s.se_sigma_re))
    fp = far["pool"]
    fpn = fp[fp.conv == "ne"].iloc[0]
    mf = pd.concat([mf, pd.DataFrame([dict(technology="Farseer (local path)", study="Li et al. (2025)", y=fpn.sigma_re,
                                           se=fpn.se_re)])], ignore_index=True)
    sets.append(run_set("model-free sigma*, IsoFLOP designs and Farseer's local path", mf, "y", "se", "study"))
    mfx = mf[~mf.technology.str.startswith("Porian")].reset_index(drop=True)
    sets.append(run_set("model-free sigma*, excluding Porian et al. (unannealed constant-LR profiles)", mfx, "y", "se", "study"))
    # review sensitivity (M-A): Farseer's local-path sigma* at narrower bandwidths (point estimate = plain mean over the
    # pooled compute levels; the primary SE is kept, so this row shows the shift in the point estimate only)
    sv = far.get("sens")
    if sv is not None:
        for var, lab in (("extended-grid CV optimum", "extended-grid CV bandwidth"),
                         ("h_N x 0.75 (h_D unchanged)", "h_N x 0.75")):
            sel = sv[(sv.variant == var) & (sv.key == "path_sigma_mean_poolC|ne")]
            if len(sel):
                mfx2 = mfx.copy()
                mfx2.loc[mfx2.technology == "Farseer (local path)", "y"] = float(sel.value.iloc[0])
                sets.append(run_set(f"model-free sigma*, excluding Porian; Farseer path at the {lab} (plain mean over C; "
                                    "primary SE)", mfx2, "y", "se", "study"))
    # parametric sigma* on the same IsoFLOP runs
    par = iso["param"].copy()
    par["technology"] = par.design
    par["study"] = par.design.map(lambda d: rc.DESIGN_META[d]["study"])
    sets.append(run_set("sigma*_kappa on the IsoFLOP designs (same runs, wild SEs)", par, "sigma_kappa", "se_sigma_kappa", "study"))
    sets.append(run_set("sigma* under kappa = 1 on the IsoFLOP designs (same runs, wild SEs)", par, "sigma_chin", "se_sigma_chin", "study"))
    het = pd.concat(sets, ignore_index=True)
    pz = pairwise_z(t3.sigma_star_q.values, t3.se_sigma_star_q.values, list(t3.technology))
    pz2 = pairwise_z(t3.sigma_kappa_ra1.values, t3.se_sigma_kappa_wild.values, list(t3.technology))
    return dict(het=het, inputs_sweeps=t3[["technology", "study", "n_obs", "sigma_star", "se_sigma_star", "sigma_star_q",
                                           "se_sigma_star_q", "sigma_kappa_ra1", "se_sigma_kappa_wild"]],
                inputs_modelfree=mf, pairwise_m2=pz, pairwise_wild=pz2)
