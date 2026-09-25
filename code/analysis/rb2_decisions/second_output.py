"""second_output.py -- the output-concept test of Bond et al. (2021) with a second OUTPUT on the same runs
(R1 New 5; Prop. A8(ii): w_hat = w w_L / w_L' when the developer optimizes L' and the econometrician uses L).

Data: the OLMo ladder (Bhagia et al. 2024; 30 runs, 5 sizes x {0.5, 1, 2, 5, 10} x Chinchilla + 5 reruns; final logged
step; m2_techpanel's loader). Outputs: C4-en validation cross-entropy (nats/token; m2's primary) and task bits per
byte (mean over the validation splits of the downstream tasks), and by task family: knowledge (the four MMLU
subject groups), science QA (ARC-Easy, ARC-Challenge, OpenBookQA), commonsense (HellaSwag, PIQA, SocialIQA,
WinoGrande, CommonsenseQA), reading comprehension (BoolQ).
Technologies: Chinchilla form (kappa = 1; m2's Huber estimator) and kappa free (inner exponents; m2's fit_q), each
fitted on each output. The statistic is ln(w_task / w_C4) for every clean-sample model, in OLMo's parameter
convention (excludes the input embedding only). Inference: pairs bootstrap by (size, multiplier) cell, the SAME
resample for both outputs (B = 399, 4 processes), so the distribution of the ratio reflects the joint sampling.
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import rb2common as C

FAMILIES = {
    "knowledge (MMLU)": ["mmlu_stem_val", "mmlu_humanities_val", "mmlu_social_sciences_val", "mmlu_other_val"],
    "science QA (ARC, OBQA)": ["arc_easy_val", "arc_challenge_val", "openbookqa_val"],
    "commonsense (HS, PIQA, SIQA, WG, CSQA)": ["hellaswag_val", "piqa_val", "socialiqa_val", "winogrande_val", "csqa_val"],
    "reading comprehension (BoolQ)": ["boolq_val"],
}


def load():
    import m2_data as md  # m2_techpanel (read-only)
    df = md.load_olmo()
    fam = {}
    for f in sorted(glob.glob(os.path.join(C.ROOT, "data", "raw", "olmo_ladder", "*xC*.csv"))):
        base = os.path.basename(f)[:-4]
        last = pd.read_csv(f).iloc[-1]
        row = {}
        for k, pats in FAMILIES.items():
            cols = [c for c in last.index if c.startswith("eval/downstream_bpb/") and any(p + "_" in c for p in pats)]
            row[k] = float(last[cols].mean())
            row[k + " n"] = len(cols)
        fam[base] = row
    F = pd.DataFrame.from_dict(fam, orient="index")
    df = df.merge(F, left_on="run", right_index=True)
    return df


def _lnG_chin(th):
    lnA, lnB, lnE, al, be = th
    return (np.log(al) + lnA - np.log(be) - lnB) / (al + be)


def _lnG_q(th):
    a, b, e, al, be, q = th
    return (np.log(al) + a - np.log(be) - b) / (al + be)


def fit_both(N, D, L, th0=None):
    """Chinchilla (Huber) and kappa-free fits; returns ((alpha, beta, lnG) chin, (a1, b1, lnG) kappa-free, thetas)."""
    import m2_est as me
    if th0 is None:
        th, f = me.fit_chin(N, D, L, "huber", grid="default")
        # m2 level-preserving starts as a check
        best = me.fit_chin_multi(N, D, L, "huber", starts=me.level_starts(N, D, L))
        if best is not None and best[1] < f:
            th, f = best
    else:
        th, f = me.fit_chin(N, D, L, "huber", grid="fast", init=th0[0])
        best = me.fit_chin_multi(N, D, L, "huber", starts=[th0[0]])
        if best is not None and best[1] < f:
            th, f = best
    thq, fq = me.fit_q(N, D, L, "huber", th_chin=th, init=None if th0 is None else th0[1])
    return (th[3], th[4], _lnG_chin(th)), (thq[3], thq[4], _lnG_q(thq)), (th, thq)


def _boot_worker(args):
    seeds, df, outputs, th0s = args
    import m2_est as me
    out = []
    cells = df["cluster"].values
    ug = np.unique(cells)
    for s in seeds:
        rng = np.random.default_rng(s)
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.where(cells == g)[0] for g in pick])
        d = df.iloc[idx]
        row = []
        for o in outputs:
            try:
                c1, cq, _ = fit_both(d["N"].values, d["D"].values, d[o].values, th0=th0s[o])
                row += list(c1) + list(cq)
            except Exception:  # noqa: BLE001
                row += [np.nan] * 6
        out.append(row)
    return out


def run(Bc_olmoN, D, B=399):
    """Bc_olmoN: clean-sample N in OLMo's convention; D tokens. Returns (per-model table, summary, fits)."""
    df = load()
    outputs = ["L", "L_bpb"] + list(FAMILIES)
    fits, th0s = {}, {}
    for o in outputs:
        c1, cq, ths = fit_both(df["N"].values, df["D"].values, df[o].values)
        fits[o] = dict(chin=c1, kfree=cq)
        th0s[o] = ths
    seeds = [C.SEED + 5000 + i for i in range(B)]
    chunks = [seeds[i::C.N_PROC] for i in range(C.N_PROC)]
    with ProcessPoolExecutor(C.N_PROC) as ex:
        parts = list(ex.map(_boot_worker, [(c, df, outputs, th0s) for c in chunks]))
    dr = np.array([r for p in parts for r in p])            # B x (6 * n_outputs)
    lnN, lnD = np.log(Bc_olmoN), np.log(D)
    rows_t, per_model = [], {}
    for j, o in enumerate(outputs):
        for form, off in (("kappa = 1", 0), ("kappa free", 3)):
            p = fits[o]["chin" if off == 0 else "kfree"]
            al, be, lnG = p
            rows_t.append(dict(output=o, form=form, alpha=al, beta=be, lnG=lnG, S2=(al + be) / 2, sigma_star=2 / (2 + al + be),
                               a=be / (al + be), Mstar_1e21=float(np.exp(C.ln_mstar(1e21, al, be, lnG))),
                               Mstar_1e23=float(np.exp(C.ln_mstar(1e23, al, be, lnG))),
                               n_draws_ok=int(np.isfinite(dr[:, 6 * j + off: 6 * j + off + 3]).all(1).sum())))
            per_model[(o, form)] = (C.log_w(lnN, lnD, al, be, lnG),
                                    C.log_w(lnN[None, :], lnD[None, :], dr[:, 6 * j + off: 6 * j + off + 1],
                                            dr[:, 6 * j + off + 1: 6 * j + off + 2], dr[:, 6 * j + off + 2: 6 * j + off + 3]))
    techs = pd.DataFrame(rows_t)
    summ = []
    ratios = {}
    for o in outputs[1:]:
        for form in ("kappa = 1", "kappa free"):
            pt = per_model[(o, form)][0] - per_model[("L", form)][0]
            d = per_model[(o, form)][1] - per_model[("L", form)][1]
            ok = np.isfinite(d).all(1)
            d = d[ok]
            med = np.median(d, axis=1)
            ratios[(o, form)] = pt
            summ.append(dict(output=o, form=form, n_models=len(pt), median_lnratio=float(np.median(pt)),
                             median_lnratio_lo=C.pct(med, 2.5), median_lnratio_hi=C.pct(med, 97.5),
                             q10_lnratio=float(np.percentile(pt, 10)), q90_lnratio=float(np.percentile(pt, 90)),
                             min_lnratio=float(pt.min()), max_lnratio=float(pt.max()),
                             share_models_ci_excludes_0=float(np.mean((np.percentile(d, 2.5, axis=0) > 0) |
                                                                      (np.percentile(d, 97.5, axis=0) < 0))),
                             median_s_C4=float(np.median(C.share(np.exp(per_model[("L", form)][0])))),
                             median_s_output=float(np.median(C.share(np.exp(per_model[(o, form)][0])))),
                             share_w_gt1_C4=float(np.mean(per_model[("L", form)][0] > 0)),
                             share_w_gt1_output=float(np.mean(per_model[(o, form)][0] > 0)), B_ok=int(ok.sum())))
    return techs, pd.DataFrame(summ), ratios, per_model, df
