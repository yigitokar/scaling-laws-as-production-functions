"""variants.py -- the primary identified set (S1) and its sensitivities (S2(a)-(f)) as lists of anchored paths.

S1 (decision D-3): one path per study, each with a simultaneous 95 percent band that carries level and slope error
jointly: Chinchilla (9 bracketed budgets, 6e18 to 3e21, total parameters as digitized); Llama 3 (8, up to 1e21,
FLOP-implied N_F); Marin (DCLM 7, Nemotron-CC 7, Comma 5 budgets up to 3e20, FLOP-implied; common slope, corpus
levels, pooled residual variance); DeepSeek's published law as a point path in its own convention. Models enter in
total parameters, the convention of released models' M (Llama 3 and Marin are converted in S2(f)).

S2 (fix list):
 (a) PI-1 as published in version 3 (rb2 code, read-only): bootstrap-t anchor intervals at the largest bracketed
     budget, path slopes bounded by the hull of point estimates [-0.156, 0.162], Comma excluded by the six-budget rule
     chosen after its interval was seen; also PI-1 with Comma (the disclosure).
 (b) every anchor on a t-interval with k - 2 degrees of freedom, Comma admitted with its own slope (R2 NM1); (b') also
     with the slopes' t-intervals.
 (c) PI-1's anchor intervals crossed with the hull of the anchors' slope confidence intervals (R1 New 3(a), R4 N2).
 (d) normal inputs only, |e| <= 1, from the S1 studies' anchor intervals (and from PI-1's, as published).
 (e) every in-set technology's in-support zero point admitted as an anchor, beside the S1 bands (and PI-4 as
     published).
 (f) conventions: all anchors in total parameters; all anchors and models in FLOP-effective parameters; Chinchilla's
     anchor alone in N_F.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import rb5common as K
import data as DA
import bands as BD
import idset as ID

import signid as SI          # rb2_decisions (read-only)
import techs as TT           # ra2_wedge (read-only)
import analysis_ra2 as AN    # ra2_wedge (read-only)


class ConvView:
    """A fitted path placed on another parameter convention (models' axes), without refitting."""

    def __init__(self, base, conv, name=None):
        self.base, self.conv = base, conv
        self.name = name or base.name
        self.kind = base.kind

    def band(self, c):
        return self.base.band(c)


def marin_frame(total=False):
    gs = []
    for des in ("Marin, DCLM", "Marin, Nemotron-CC", "Marin, Comma"):
        g = DA.marin_minima_total(des) if total else DA.minima(des)
        g["group"] = des.replace("Marin, ", "")
        gs.append(g)
    return pd.concat(gs, ignore_index=True)


def published_slopes(X):
    """Path slopes e = 1 - 2a of the published laws used by PI-1 (DeepSeek's three data-quality exponents, MiniCPM)."""
    return [1 - 2 * a for a in TT.DEEPSEEK["table4_a"].values()] + [1 - 2 * X["T"]["minicpm"].a]


def deepseek_point(X):
    t = X["T"]["deepseek"]
    return BD.PointPath("DeepSeek LLM law", t.alpha, t.beta, t.lnG, conv="ds")


def deepseek_anchor(X, eL, eU):
    t = X["T"]["deepseek"]
    x = float(np.log(3e20))
    v = float(-2 * t.lnG + (1 - 2 * t.a) * (x - np.log(6.0)))
    return BD.AnchorPath("DeepSeek LLM law (anchor at 3e20)", x, v, v, eL, eU, conv="ds")


def primary(X, B, *, studentize="pooled", simultaneous=True, marin_mean=False, B_boot=K.B_BOOT, seed=K.SEED,
            crit_rule="max"):
    cr = ID.c_extent(B, ("total", "nf"))
    extra = np.concatenate([ID.model_axes(B, cv)[1] for cv in ("total", "nf")] + [[np.log(1e24), np.log(1e25)]])
    kw = dict(B=B_boot, c_range=cr, studentize=studentize, simultaneous=simultaneous, extra_c=extra,
              crit_rule=crit_rule)
    chin = BD.StudyPath("Chinchilla", DA.minima("Chinchilla"), conv="total", seed=seed + 1, **kw)
    llama = BD.StudyPath("Llama 3", DA.minima("Llama 3"), conv="total", seed=seed + 2, **kw)
    marin = BD.StudyPath("Marin", marin_frame(), conv="total", seed=seed + 3, mean_level=marin_mean, **kw)
    return dict(chin=chin, llama=llama, marin=marin, ds=deepseek_point(X))


def paths_list(P, keys=("chin", "llama", "marin", "ds")):
    return [P[k] for k in keys]


# ------------------------------------------------------------------------------------------------ S2 variants
def pi1_published(X, B, include_comma=False):
    """Version 3's PI-1 exactly (rb2 signid.anchors / e_range / ra2 pi_bounds)."""
    A, _ = SI.anchors(X, include_comma=include_comma)
    e = SI.e_range(A, X)
    PBr = SI.pi_models(B, A, e, (0.40, 0.52))
    return A, e, PBr


def anchor_paths_from_rb2(A, e):
    out = []
    for _, a in A.iterrows():
        out.append(BD.AnchorPath(a["design"], np.log(a["C0"]), a["lnMs_lo"], a["lnMs_hi"], e[0], e[1],
                                 conv="total" if a["conv"] == "total" else a["conv"]))
    return out


def t_interval_anchors(X, with_slope_ci=False, level=K.LEVEL):
    """S2(b): each IsoFLOP design (Comma included) on its own line; anchor at its largest bracketed budget with a
    t-interval on k - 2 degrees of freedom; e over the designs' point slopes (or their t-intervals) and the published
    laws."""
    rows, anchors = [], []
    for des in ("Chinchilla", "Llama 3", "Marin, DCLM", "Marin, Nemotron-CC", "Marin, Comma"):
        g = DA.minima(des)
        c0 = g["lnC"].max()
        Xd = np.column_stack([np.ones(len(g)), g["lnC"].values - c0])
        XtXi = np.linalg.inv(Xd.T @ Xd)
        b = XtXi @ Xd.T @ g["lnM"].values
        r = g["lnM"].values - Xd @ b
        df = len(g) - 2
        s = np.sqrt(r @ r / df)
        tq = stats.t.ppf(0.5 + level / 2, df)
        se0, se1 = s * np.sqrt(XtXi[0, 0]), s * np.sqrt(XtXi[1, 1])
        rows.append(dict(design=des, n_budgets=len(g), df=df, C0=float(np.exp(c0)), Mstar=float(np.exp(b[0])),
                         Mstar_lo=float(np.exp(b[0] - tq * se0)), Mstar_hi=float(np.exp(b[0] + tq * se0)), e=b[1],
                         e_lo=b[1] - tq * se1, e_hi=b[1] + tq * se1, t_q=tq))
        anchors.append((des, c0, b[0] - tq * se0, b[0] + tq * se0))
    T = pd.DataFrame(rows)
    pub = published_slopes(X)
    if with_slope_ci:
        eL, eU = min(T["e_lo"].min(), min(pub)), max(T["e_hi"].max(), max(pub))
    else:
        eL, eU = min(T["e"].min(), min(pub)), max(T["e"].max(), max(pub))
    paths = [BD.AnchorPath(d, c0, lo, hi, eL, eU, conv="total") for d, c0, lo, hi in anchors]
    paths.append(deepseek_anchor(X, eL, eU))
    return paths, T, (eL, eU)


def normal_inputs_new(X, P):
    """S2(d) on the S1 studies' anchors: each corpus level's pointwise bootstrap-t interval at c_J, e in [-1, 1]."""
    out = []
    for k in ("chin", "llama", "marin"):
        sp = P[k]
        ai = sp.anchor_interval()
        for _, r in ai.iterrows():
            out.append(BD.AnchorPath(f"{sp.name} {r['group']}", sp.c_J, r["lnMs_lo"], r["lnMs_hi"], -1.0, 1.0,
                                     conv="total"))
    out.append(deepseek_anchor(X, -1.0, 1.0))
    return out


def every_technology(X, P):
    """S2(e): the S1 bands plus every in-set technology's in-support M* (ra2's anchors of kind 'all', each in its own
    convention and with its own interval), extrapolated with e over all in-set technologies' paths and the S1 slopes."""
    A = AN.pi_anchors(X["T"], X["M"], X["mf_tab"], X["mf_draws"])
    Aall = A[A["kind"] == "all"]
    M = X["M"]
    e_s = [float(P[k].beta[-1]) for k in ("chin", "llama", "marin")]
    e_t = [1 - 2 * a for a in M.loc[M["in_set"], "a"]]
    eL, eU = min(e_s + e_t), max(e_s + e_t)
    out = paths_list(P)
    for _, a in Aall.iterrows():
        out.append(BD.AnchorPath(f"tech {a['anchor']}", np.log(a["C0"]), a["lnMs_lo"], a["lnMs_hi"], eL, eU,
                                 conv=a["conv"]))
    return out, (eL, eU), A


def pi4_published(X, B, A_pi1, e_pi1):
    """rb2's PI-4 exactly: PI-1 anchors plus ra2's 'all' anchors, e over all in-set technologies and PI-1's range."""
    A_ra2 = AN.pi_anchors(X["T"], X["M"], X["mf_tab"], X["mf_draws"])
    A4 = pd.concat([A_pi1, A_ra2[A_ra2["kind"] == "all"]], ignore_index=True)
    M = X["M"]
    a_all = list(M.loc[M["in_set"], "a"]) + [(1 - v) / 2 for v in e_pi1]
    e_all = (1 - 2 * max(a_all), 1 - 2 * min(a_all))
    PB4 = AN.pi_bounds(B, pd.Series(True, index=B.index), A4, e_all, (0.40, 0.52),
                       {"PI-4": dict(kinds=["iso", "all"], e=e_all)})
    return PB4, e_all


def conventions(X, B, P, *, B_boot=K.B_BOOT, seed=K.SEED):
    """S2(f). (i) all anchors in total parameters: Chinchilla as digitized; Marin's minima converted with its measured
    configuration-to-FLOP ratio (refitted); Llama 3's band shifted by the bound on ln(N_F/N_total) at its anchor.
    (ii) all anchors and models in FLOP-effective parameters: Chinchilla's minima in N_F (rb4, Hoffmann's count);
    Llama 3 and Marin as estimated; models' N_F by Hoffmann's accounting at 4,096 tokens.
    (iii) Chinchilla's anchor alone in N_F, models in total parameters (audit numbers_conclusion F6)."""
    cr = ID.c_extent(B, ("total", "nf"))
    extra = np.concatenate([ID.model_axes(B, cv)[1] for cv in ("total", "nf")] + [[np.log(1e24), np.log(1e25)]])
    kw = dict(B=B_boot, c_range=cr, extra_c=extra)
    conv_T, (lr_lo, lr_hi) = DA.llama3_conversion()
    marin_T = BD.StudyPath("Marin (total parameters)", marin_frame(total=True), conv="total", seed=seed + 13, **kw)
    llama_T = BD.ShiftedPath(P["llama"], lr_lo, lr_hi, name="Llama 3 (total parameters, conversion bound)")
    total = [P["chin"], llama_T, marin_T, P["ds"]]
    chin_F = BD.StudyPath("Chinchilla (N_F)", DA.minima("Chinchilla", source="rb4_nf"), conv="nf", seed=seed + 11, **kw)
    nf = [chin_F, ConvView(P["llama"], "nf"), ConvView(P["marin"], "nf"), P["ds"]]
    swap = [ConvView(chin_F, "total", "Chinchilla (N_F) vs total M"), P["llama"], P["marin"], P["ds"]]
    info = dict(llama3_conversion=conv_T, llama3_lnr=(lr_lo, lr_hi), marin_T=marin_T, chin_F=chin_F)
    return dict(total=total, nf=nf, swap=swap), info
