"""porian.py -- re-implementation of the Porian et al. (2024) IsoFLOP pipeline and the
measurement / flexible-input decomposition of the Kaplan-Chinchilla gap in the allocation exponent a.

Data: data/raw/porian/experiment_results.pickle.xz (975 runs, MIT license).
Reference code: github.com/formll/resolving-scaling-law-discrepancies (analysis.py, data.py, configs.py).
We port their steps (parameter/FLOP conventions, proportional sliding-window smoothing of the train loss,
log-log interpolation of the loss at each FLOP budget, Akima interpolation across model sizes,
"noise-and-interpolate" bootstrap of the IsoFLOP argmin, weighted log-OLS of N*(C) on C) so that every
number is regenerated from the raw runs. We skip their saturating loss-curve fits (not needed for a).

Econometric reading of each step (paper Section 'Measurement and flexible inputs'):
  step 1 -> 2  (count last-layer FLOPs): *input measurement*. N and C are measured without the unembedding
                (head) matrix, whose share of N falls with scale -> regressor-correlated measurement error
                (Collard-Wexler & De Loecker 2016), Nerlove (1963)-type spurious small-scale economies.
  step 2 -> 3  (warmup scaled with N): *flexible input not optimized* (warmup is a fixed 1.57B tokens).
  step 3 -> 4  (cosine decay matched to budget): flexible input (schedule).
  step 4 -> 5  (tune LR, batch size, beta2 per size): flexible inputs concentrated out (envelope).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

from common import RAW, PROC, wls_loglog

PICKLE = os.path.join(RAW, "porian", "experiment_results.pickle.xz")
FLOP_VALS = 1e17 * 2 ** np.arange(-3.0, 9, 1.0)          # Porian configs.FLOP_VALS (1.25e16 ... 2.56e19)
CHINCHILLA_FLOPS = 5.88e23
SEQ_LEN = 2048

# (loss key, FLOPs-per-token key, N key) for each counting convention; first four are Porian's ISOFLOP_ARGS
COUNTS = {
    ("kaplan", "train"): ("train/loss_smoothed", "flops_per_token_no_att_no_embed", "params_no_embed"),
    ("kaplan", "val"): ("val/loss", "flops_per_token_no_att_no_embed", "params_no_embed"),
    ("standard", "val"): ("val/loss", "flops_per_token", "params"),
    ("standard", "train"): ("train/loss_smoothed", "flops_per_token", "params"),
    ("attention", "train"): ("train/loss_smoothed", "flops_per_token_att", "eff_params_att"),
    # extension: Pearce-Song "total" convention (precise body + input embedding + head + positional; C = 6 N_total D)
    ("total", "train"): ("train/loss_smoothed", "flops_per_token_total", "params_all_precise"),
}
RW_SEED = dict(noise_low=0.002, noise_high=0.05, l_threshold_high=7, l_threshold_low=3)
OWT2_SEED = dict(noise_low=0.01, noise_high=0.1, l_threshold_high=6, l_threshold_low=3)

# Porian Table 1 (a, 95% CI) for the five Figure-1 steps + 'Kaplan adjusted' (literature values, for comparison)
PORIAN_REPORTED = {
    ("rw", 1): (0.835, 0.82, 0.85), ("rw", 2): (0.706, 0.69, 0.72), ("rw", 3): (0.602, 0.59, 0.62),
    ("rw", 4): (0.571, 0.56, 0.59), ("rw", 5): (0.497, 0.49, 0.50),
    ("owt2", 1): (0.864, 0.82, 0.90), ("owt2", 2): (0.699, 0.66, 0.72), ("owt2", 3): (0.603, 0.57, 0.63),
    ("owt2", 4): (0.574, 0.54, 0.61), ("owt2", 5): (0.518, 0.49, 0.54),
    ("rw", "kaplan_adj"): (0.717, np.nan, np.nan),
}

STEPS = [  # (step id, label, hparams, warmup, decay, count, loss)
    (1, "Reproduce Kaplan (non-embedding, no head FLOPs)", "base", "long", "kaplan", "kaplan", "train"),
    (2, "+ count last-layer (head) FLOPs", "base", "long", "kaplan", "standard", "val"),
    (3, "+ warmup scaled to model size", "base", "short", "kaplan", "standard", "val"),
    (4, "+ cosine decay matched to budget", "base", "short", "chinchilla", "standard", "val"),
    (5, "+ tuned LR, batch, beta2 (constant LR)", "tuned", "short", "const", "standard", "val"),
]


# ----------------------------------------------------------------------------- data processing (port of data.py)

def precise_flops_per_token_chinchilla(width, depth, seq_len=2048, vocab_size=50432, num_heads=4):
    width, depth = width.astype(float), depth.astype(float)
    embeddings = 2 * seq_len * width
    attention = 2 * 3 * seq_len * width ** 2
    attention = attention + 2 * seq_len * seq_len * width + 3 * num_heads * seq_len * seq_len \
        + 2 * seq_len * seq_len * width + 2 * seq_len * width ** 2
    dense_block = 4 * seq_len * width * (4 * width)
    final_logits = 2 * seq_len * width * vocab_size
    fwd = embeddings + depth * attention + depth * dense_block + final_logits
    return 3 * fwd / seq_len


def precise_param_count_open_lm(width, depth, vocab_size=50432):
    d_ff = 256 * (((2 * 4 * width / 3).astype(int) + 256 - 1) // 256)
    return (4 * width + 3 * d_ff) * width * depth + vocab_size * width


def _prop_window_smooth(x: pd.Series, p=0.05, compensate=True):
    """Porian's proportional_sliding_window_filter (window half-width = p * index position)."""
    cs = x.cumsum().values
    csp = np.concatenate([[0.0], cs])
    inds = np.arange(len(x))
    up = np.minimum(inds + np.floor(p * inds).astype(int), len(x) - 1)
    dn = np.maximum(0, inds - np.floor(p * inds).astype(int))
    idx_new = np.interp((up + dn) / 2, inds, x.index.values.astype(float))
    vals = (cs[up] - csp[dn]) / (up - dn + 1)
    if compensate:
        idx_new = idx_new - np.diff(idx_new, prepend=0) / 2
    return pd.Series(vals, index=idx_new)


def load_porian():
    df = pd.read_pickle(PICKLE, compression="xz").reset_index(drop=True)
    w, dep, V, S = df.width, df.depth, df.vocab_size, df.seq_len
    df["params_active"] = (12 * w ** 2 * dep + V * w).astype(float)
    df["params_active_precise"] = precise_param_count_open_lm(w, dep).astype(float)
    df["params_no_embed"] = precise_param_count_open_lm(w, dep, vocab_size=0).astype(float)
    df["params_all"] = (12 * w ** 2 * dep + (S + 2 * V) * w).astype(float)   # Porian's definition (approximate body)
    # REVIEW FIX: Porian's params_all uses the approximate body 12 w^2 depth, which differs from the precise SwiGLU
    # body count (params_no_embed) by up to -25% for the small architectures. For the Pearce-Song "total" convention
    # we therefore add the input embedding, positional embedding and head to the *precise* body, so that "total"
    # differs from "standard" only by the input + positional embeddings.
    df["params_all_precise"] = (df.params_no_embed + (S + 2 * V) * w).astype(float)
    df["flops_per_token_att"] = 6 * df.params_active_precise + 6 * S * w * dep
    df["flops_per_token_cc"] = precise_flops_per_token_chinchilla(w, dep)
    df["flops_per_token_no_att"] = 6 * df.params_active_precise
    df["flops_per_token_no_att_no_embed"] = 6 * df.params_no_embed
    df["flops_per_token"] = df.flops_per_token_no_att
    df["flops_per_token_total"] = 6 * df.params_all_precise
    df["params"] = df.flops_per_token / 6
    df["eff_params_att"] = df.flops_per_token_att / 6
    sm = []
    for s in df["train/loss"]:
        s = s.dropna() if s is not None else s
        sm.append(None if s is None or len(s) == 0 else _prop_window_smooth(s))
    df["train/loss_smoothed"] = sm
    return df


# ----------------------------------------------------------------------------- IsoFLOP pipeline (port of analysis.py)

def fetch_flop(sub, C, loss_key, fpt_key, n_key, tol=0.1):
    """Loss of every run at training FLOPs C (log-log interpolation along the run's loss curve), as in
    Porian's fetch_flop: runs whose nearest logged point is > tol away (relative) from C are dropped."""
    out = []
    for _, r in sub.iterrows():
        s = r[loss_key]
        if s is None or len(s) == 0:
            continue
        s = s.dropna().groupby(level=0).mean().sort_index()
        if len(s) == 0:
            continue
        fl = s.index.values.astype(float) * SEQ_LEN * r["bs"] * r[fpt_key]
        lv = s.values
        k = np.searchsorted(fl, C)
        if k > 0:
            k = k - 1 + int(np.argmin(np.abs(np.log(fl[k - 1:k + 1] / C))))
        k = min(k, len(fl) - 1)
        if np.exp(abs(np.log(fl[k] / C))) - 1 > tol:
            continue
        if len(fl) > 1:
            lo, hi = max(0, k - 5), k + 5
            loss = float(np.exp(np.interp(np.log(C), np.log(fl[lo:hi]), np.log(lv[lo:hi]))))
        else:
            loss = float(lv[k])
        out.append(dict(n=float(r[n_key]), t=C / r[fpt_key], loss=loss, width=r["width"], depth=r["depth"],
                        lr=r["lr"], bs=r["bs"]))
    return pd.DataFrame(out)


def _noise_sd(loss, noise_low, noise_high, l_threshold_high, l_threshold_low):
    """Porian's get_noise_for_loss rule (thresholds are on log loss, so in practice sd = noise_low)."""
    ll = np.log(loss)
    lo, hi = np.log(l_threshold_low), np.log(l_threshold_high)
    out = np.interp(ll, [lo, hi], [np.log(noise_low), np.log(noise_high)])
    out = np.where(ll >= l_threshold_high, np.log(noise_high), out)
    out = np.where(ll <= l_threshold_low, np.log(noise_low), out)
    return np.exp(out)


def isoflop_argmin(pts: pd.DataFrame, rng, seed_cfg, iters=1000, mult=25, min_std_factor=0.33):
    """Akima interpolation of log loss on log N over one IsoFLOP; bootstrap by adding calibrated seed noise.
    Returns dict with point argmin, bootstrap draws (edge argmins dropped) and Porian's inflated log-sd."""
    p = pts.loc[pts.groupby("n").loss.idxmin()].sort_values("n")
    if len(p) < 3:
        return None
    x, y = np.log(p.n.values), p.loss.values
    m = (len(p) - 1) * mult
    grid = np.linspace(x.min(), x.max(), m)
    f0 = np.exp(Akima1DInterpolator(x, np.log(y))(grid))
    i0 = int(np.argmin(f0))
    sd = _noise_sd(y, **seed_cfg)
    noise = rng.standard_normal((iters, len(y))) * sd
    stars, lstars = [], []
    for b in range(iters):
        yb = y + noise[b]
        fb = Akima1DInterpolator(x, np.log(yb))(grid)
        ib = int(np.argmin(fb))
        if 0 < ib < m - 1:
            stars.append(np.exp(grid[ib]))
            lstars.append(np.exp(fb[ib]))
    res = dict(n_pts=len(p), n_point=np.exp(grid[i0]) if 0 < i0 < m - 1 else np.nan, edge=not (0 < i0 < m - 1),
               loss_point=f0[i0], n_min=p.n.min(), n_max=p.n.max())
    if len(stars) < iters // 2:
        res.update(valid=False, stars=None, std=np.nan)
        return res
    std = np.std(np.log(stars))
    min_std = min_std_factor * mult * (grid[1] - grid[0])
    std = max(std, min_std) * (iters / len(stars))
    res.update(valid=True, stars=np.array(stars), std=std, n_median=float(np.median(stars)),
               loss_median=float(np.median(lstars)))
    return res


def run_config(df, dataset, hparams, warmup, decay, count, loss, seed=42, iters=1000, flop_vals=FLOP_VALS):
    """Full pipeline for one configuration: per-C argmins, weighted power law N* = k C^a, bootstrap CI."""
    loss_key, fpt_key, n_key = COUNTS[(count, loss)]
    sub = df.query("dataset==@dataset and hparams==@hparams and warmup==@warmup and decay==@decay")
    seed_cfg = RW_SEED if dataset == "rw" else OWT2_SEED
    rng = np.random.default_rng(seed)
    rows, pts_all = [], []
    for C in flop_vals:
        pts = fetch_flop(sub, C, loss_key, fpt_key, n_key)
        if len(pts) == 0:
            continue
        pts["C"] = C
        pts_all.append(pts)
        r = isoflop_argmin(pts, rng, seed_cfg, iters=iters)
        if r is None:
            continue
        r["C"] = C
        rows.append(r)
    per_c = pd.DataFrame(rows)
    ok = per_c[per_c.valid & ~per_c.edge].copy() if len(per_c) else per_c
    out = dict(dataset=dataset, hparams=hparams, warmup=warmup, decay=decay, count=count, loss=loss,
               n_budgets=len(ok))
    if len(ok) >= 3:
        w = 1.0 / ok["std"].values ** 2
        a, k, r2 = wls_loglog(ok.C.values, ok.n_median.values, w)
        B = min(len(s) for s in ok.stars)
        boots = np.array([wls_loglog(ok.C.values, np.array([s[i] for s in ok.stars]), w)[:2] for i in range(B)])
        a_unw, _, _ = wls_loglog(ok.C.values, ok.n_median.values)
        out.update(a=a, a_lo=np.quantile(boots[:, 0], 0.025), a_hi=np.quantile(boots[:, 0], 0.975),
                   a_se=np.std(boots[:, 0], ddof=1), r2=r2, a_unweighted=a_unw, logk=k,
                   N_at_chinchilla=np.exp(k) * CHINCHILLA_FLOPS ** a,
                   N_at_chinchilla_lo=np.quantile(np.exp(boots[:, 1]) * CHINCHILLA_FLOPS ** boots[:, 0], 0.025),
                   N_at_chinchilla_hi=np.quantile(np.exp(boots[:, 1]) * CHINCHILLA_FLOPS ** boots[:, 0], 0.975),
                   C_min=ok.C.min(), C_max=ok.C.max(), n_boot=B)
    per_c = per_c.drop(columns=["stars"], errors="ignore")
    for k_, v in [("dataset", dataset), ("hparams", hparams), ("warmup", warmup), ("decay", decay),
                  ("count", count), ("loss", loss)]:
        per_c[k_] = v
    pts_all = pd.concat(pts_all, ignore_index=True) if pts_all else pd.DataFrame()
    return out, per_c, pts_all


# ----------------------------------------------------------------------------- analyses

def porian_steps(df, iters=1000):
    """Reproduce Porian's Table 1 (both datasets) + 'Kaplan adjusted' + attention-accounting variants."""
    res, perc, pts = [], [], {}
    for ds in ("rw", "owt2"):
        for sid, lab, hp, wu, dc, cnt, ls in STEPS:
            o, pc, pt = run_config(df, ds, hp, wu, dc, cnt, ls, iters=iters)
            rep = PORIAN_REPORTED.get((ds, sid), (np.nan,) * 3)
            o.update(step=sid, label=lab, a_porian=rep[0], a_porian_lo=rep[1], a_porian_hi=rep[2])
            res.append(o)
            perc.append(pc.assign(step=sid))
            pts[(ds, sid)] = pt
    o, pc, _ = run_config(df, "rw", "tuned", "long", "const", "kaplan", "train", iters=iters)
    o.update(step="kaplan_adj", label="Tuned, long warmup, Kaplan count (Kaplan adjusted)", a_porian=0.717,
             a_porian_lo=np.nan, a_porian_hi=np.nan)
    res.append(o)
    perc.append(pc.assign(step="kaplan_adj"))
    return pd.DataFrame(res), pd.concat(perc, ignore_index=True), pts


def measurement_grid(df, iters=1000):
    """Holding the runs fixed, vary only the counting convention for N and C (train loss throughout,
    so that the same loss curves are read at the same token counts up to the FLOP relabeling)."""
    res = []
    run_sets = [("Kaplan setup (untuned, long warmup)", "base", "long", "kaplan"),
                ("Tuned (short warmup, const LR)", "tuned", "short", "const")]
    for ds in ("rw", "owt2"):
        for rlab, hp, wu, dc in run_sets:
            for cnt in ("kaplan", "standard", "attention", "total"):
                o, _, _ = run_config(df, ds, hp, wu, dc, cnt, "train", iters=iters)
                o.update(run_set=rlab)
                res.append(o)
    return pd.DataFrame(res)


def architecture_shares(df):
    """Head (unembedding) share s = V w / N_std along Porian's architecture ladder, and its elasticity
    theta = dln(V w)/dln N_noembed (the omitted component's scale elasticity)."""
    a = df[["width", "depth", "params_no_embed", "params", "params_all_precise", "vocab_size"]].drop_duplicates()
    a = a.sort_values("params").reset_index(drop=True)
    a["head"] = a.vocab_size * a.width
    a["s_head"] = a["head"] / a["params"]
    a["s_embed_total"] = (a.params_all_precise - a.params_no_embed) / a.params_all_precise
    theta = np.polyfit(np.log(a.params_no_embed), np.log(a["head"]), 1)[0]
    return a, theta


def inefficiency_gradient(pts_base, pts_tuned, per_c_base, per_c_tuned, techno):
    """Flexible-input inefficiency delta(N, C) = L_untuned - L_tuned on matched (N, C) cells, its slope along
    each IsoFLOP (Delta = d delta / d ln N at fixed C = delta_N - delta_D), and the first-order prediction of
    the argmin shift  ln N*_obs - ln N*  ~=  -Delta / f''(ln N*),  f = tuned IsoFLOP loss curve.
    techno: fitted Chinchilla on tuned runs (gives f'' = (alpha+beta) gamma R at the optimum)."""
    rows = []
    for C in sorted(set(pts_base.C) & set(pts_tuned.C)):
        b = pts_base[pts_base.C == C].groupby("n").loss.min()
        t = pts_tuned[pts_tuned.C == C].groupby("n").loss.min()
        common = b.index.intersection(t.index)
        pb = per_c_base[per_c_base.C == C]
        pt = per_c_tuned[per_c_tuned.C == C]
        if len(common) < 3 or len(pb) == 0 or len(pt) == 0 or not bool(pt.valid.iloc[0]) or not bool(pb.valid.iloc[0]):
            continue
        x = np.log(common.values.astype(float))
        dlt = (b.loc[common] - t.loc[common]).values
        xs = np.log(pt.n_median.iloc[0])
        # local slope of delta along the IsoFLOP at the tuned optimum: weighted linear fit around xs
        wts = np.exp(-0.5 * ((x - xs) / 0.75) ** 2)
        Z = np.c_[np.ones_like(x), x - xs]
        coef = np.linalg.lstsq(Z * np.sqrt(wts)[:, None], dlt * np.sqrt(wts), rcond=None)[0]
        Delta = coef[1]
        # curvature of the tuned IsoFLOP curve at its optimum: (i) from the fitted technology, (ii) local quadratic
        N_opt = np.exp(xs)
        D_opt = C / (6 * N_opt)
        u, v = techno.uv(N_opt, D_opt)
        f2_model = techno.alpha ** 2 * u + techno.beta ** 2 * v
        tl = t.loc[common].values
        wq = np.exp(-0.5 * ((x - xs) / 1.0) ** 2)
        Zq = np.c_[np.ones_like(x), x - xs, (x - xs) ** 2]
        cq = np.linalg.lstsq(Zq * np.sqrt(wq)[:, None], tl * np.sqrt(wq), rcond=None)[0]
        f2_local = 2 * cq[2]
        wq5 = np.exp(-0.5 * ((x - xs) / 0.5) ** 2)
        cq5 = np.linalg.lstsq(Zq * np.sqrt(wq5)[:, None], tl * np.sqrt(wq5), rcond=None)[0]
        f2_local05 = 2 * cq5[2]
        rows.append(dict(C=C, n_common=len(common), lnN_tuned=xs, lnN_untuned=np.log(pb.n_median.iloc[0]),
                         shift_obs=np.log(pb.n_median.iloc[0]) - xs, delta_at_opt=coef[0], Delta=Delta,
                         f2_model=f2_model, f2_local=f2_local, shift_pred_model=-Delta / f2_model,
                         shift_pred_local=-Delta / f2_local if f2_local > 0 else np.nan,
                         f2_local05=f2_local05, shift_pred_local05=-Delta / f2_local05 if f2_local05 > 0 else np.nan,
                         R_opt=u + v))
    return pd.DataFrame(rows)


def fit_tuned_technology(pts_tuned):
    """Chinchilla technology on the tuned RefinedWeb runs (all IsoFLOP points, standard count, val loss)."""
    from common import sl
    p = pts_tuned.copy()
    p["D"] = p.t
    m = sl.fit_chinchilla(p.n.values, p.D.values, p.loss.values, grid=sl.FAST_GRID)
    return m, p


def synthetic_count_check(df, dataset, hparams, warmup, decay, tech, iters=300, seed=11):
    """Semi-synthetic check of the measurement bias: replace every observed loss by the loss implied by a
    smooth technology `tech` (Chinchilla in the standard count N_std = N_noembed + V w) at exactly the same
    (architecture, FLOP budget) points, and run the IsoFLOP pipeline in the standard and in Kaplan's count.
    The difference of the two exponents is the pure measurement bias implied by `tech` on Porian's design."""
    rng = np.random.default_rng(seed)
    seed_cfg = RW_SEED if dataset == "rw" else OWT2_SEED
    arch = df[["params_no_embed", "params"]].drop_duplicates().set_index("params_no_embed")["params"]
    out = {}
    for count in ("standard", "kaplan"):
        loss_key, fpt_key, n_key = COUNTS[(count, "train")]
        sub = df.query("dataset==@dataset and hparams==@hparams and warmup==@warmup and decay==@decay")
        rows = []
        for C in FLOP_VALS:
            pts = fetch_flop(sub, C, loss_key, fpt_key, n_key)
            if len(pts) == 0:
                continue
            Nstd = pts.n.values if count == "standard" else arch.loc[pts.n.values].values
            pts["loss"] = tech.loss(Nstd, pts.t.values)
            r = isoflop_argmin(pts, rng, seed_cfg, iters=iters)
            if r is None or not r["valid"] or r["edge"]:
                continue
            rows.append(dict(C=C, n=r["n_median"], std=r["std"]))
        pc = pd.DataFrame(rows)
        a, _, _ = wls_loglog(pc.C.values, pc.n.values, 1 / pc["std"].values ** 2)
        out[count] = a
    return out
