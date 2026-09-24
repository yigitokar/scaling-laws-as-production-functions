"""m9_power.py -- power calculation for the two-corpus experiment (plan Section 4), computed from the DESIGN ONLY.

This stage never reads data/processed/sweep/results.jsonl. The design (widths, depths, D ladder, budget caps, hiM
cells) comes from code/sweep/run_grid.py and the architecture formulas; the technology is a public one.

Data-generating process (all in nats/token; bits per byte only rescale E, A, B):
  Truth "chin"  Chinchilla form with Besiroglu et al.'s exponents (alpha = 0.3478, beta = 0.3658; sigma* = 0.737),
                in non-embedding N (convention P). Levels anchored so that at the design's geometric-mean compute
                C_c (of 6 N_nonemb D over the main grid) the w = 1 path has M* = 20 and L = 3.8 nats, with E = 1.9.
  Truth "kappa" kappa family with sigma*_k = 0.70 (the model-free consensus), a1 = b1 = 0.4286, kappa = 0.6, same anchor.
  Corpora       web = truth; edu = truth with a data-augmenting shift ln B_edu = ln B_web - chi (chi = 0 or 0.22,
                DataDecide's tilt), so Delta ln(A/B) = chi exactly.
  Noise         ln L_obs = ln L + e, e = s (sqrt(rho) u_trunk + sqrt(1 - rho) v_endpoint): a common within-trunk
                component (one trunk per architecture; hiM (256,4) shares width 256's trunk), s in {0.002, 0.005, 0.01},
                rho = 0.5 (0 and 0.9 as sensitivity; m6 mc_lib.het_cluster_noise with het = 0). A second validation set
                has the same truth and noise correlated 0.8 with the first (same models, different text).
Estimators: exactly those of run.py (Huber-LSE Chinchilla form; kappa family; model-free local-quadratic path with
LOO-CV bandwidth; corpus-pair CE model; Q3 restricted fit on M <= 100). Expected standard errors are Monte Carlo
standard deviations across replications. Power of the tilt test: (a) normal approximation with the Monte Carlo s.e.;
(b) the actual procedure -- 95% basic interval from a wild cluster bootstrap by width with Webb weights (B = 199 per
replication, fewer than the 999 of the real analysis, for time) -- and the plan's decision rule on two validation sets.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import m9_common as mc
import m9_est as est

BES_ALPHA, BES_BETA = 0.3478, 0.3658
L_ANCHOR, E_ANCHOR, M_ANCHOR = 3.8, 1.9, 20.0
RHO_VAL = 0.8
C_LEVELS = np.exp(np.linspace(np.log(3e14), np.log(6e16), 12))
BW_MULTS = (0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5)


# ============================================================================ design (no results)
def mc_design():
    """Main + hiM endpoints per corpus from run_grid.py (duplicates of main cells removed). D = tokens actually
    trained (ceil(D_target / 16,384) steps)."""
    cells = mc.design_cells()
    out = {}
    for r in mc.CORPORA:
        m = cells[(cells.regime == r) & (cells.tag == "main")].copy()
        h = cells[(cells.regime == r) & (cells.tag == "hiM")].copy()
        keys = set(zip(m.d, m.L, m.D_target))
        h = h[[(a, b, c) not in keys for a, b, c in zip(h.d, h.L, h.D_target)]]
        df = pd.concat([m, h], ignore_index=True)
        df["D"] = np.ceil(df.D_target / 16384.0) * 16384.0
        df["N_P"], df["N_T"] = df["N_nonemb"], df["N_total"]
        df["M_P"], df["M_T"] = df.D / df.N_P, df.D / df.N_T
        df["C_P"] = df.fpt * df.D
        df["arch"] = df.d.astype(str) + "x" + df.L.astype(str)
        df["is_main"] = df.tag == "main"
        out[r] = df.reset_index(drop=True)
    return out


def anchor(design):
    m = design["web"][design["web"].is_main]
    Cc = float(np.exp(np.mean(np.log(6 * m.N_P * m.D))))
    Ns = np.sqrt(Cc / (6 * M_ANCHOR))
    return Cc, Ns, M_ANCHOR * Ns


def truth_params(kind, design, M_anchor=M_ANCHOR):
    Cc, _, _ = anchor(design)
    Ns = np.sqrt(Cc / (6 * M_anchor))
    Ds = M_anchor * Ns
    R = L_ANCHOR - E_ANCHOR
    if kind == "chin":
        al, be = BES_ALPHA, BES_BETA
        u, v = be * R / (al + be), al * R / (al + be)
        return dict(kind=kind, lnA=np.log(u) + al * np.log(Ns), lnB=np.log(v) + be * np.log(Ds), lnE=np.log(E_ANCHOR),
                    a1=al, b1=be, k=1.0, Cc=Cc)
    k, a1 = 0.6, 0.5 * (2 / 0.70 - 2)
    uv = R ** (1 / k) / 2
    return dict(kind=kind, lnA=np.log(uv) + a1 * np.log(Ns), lnB=np.log(uv) + a1 * np.log(Ds), lnE=np.log(E_ANCHOR),
                a1=a1, b1=a1, k=k, Cc=Cc)


def true_lnL(tp, N, D, chi):
    lnB = tp["lnB"] - chi
    inner = np.logaddexp(tp["lnA"] - tp["a1"] * np.log(N), lnB - tp["b1"] * np.log(D))
    return np.logaddexp(tp["k"] * inner, tp["lnE"])


def noise(df, s, rho, rng):
    import mc_lib as ml  # m6 machinery: common within-cluster shock (het = 0)
    return ml.het_cluster_noise(df.N_P.values, df.D.values, s, rng, df.arch.values, het=0.0, rho=rho)


# ============================================================================ one replication
def _fit_block(N, D, L, conv_x, conv, M, is_main, C_levels, want_q3=True):
    """Q1 and Q3 statistics for one corpus and convention."""
    out = {}
    Nm, Dm, Lm = N[is_main], D[is_main], L[is_main]
    th, _ = est.fit_chin(Nm, Dm, Lm, "huber")
    p, _ = est.fit_kappa(Nm, Dm, Lm, "huber", th_chin=th)
    out["sig_chin"] = est.sigma_star_chin(th)
    out["sig_kappa"] = est.sigma_star_kappa(p)
    out["theta"] = th
    # model-free on the main grid
    x, z, f = np.log(Nm), np.log(Dm), np.log(Lm)
    (rm, hx, hz, _, _), _ = est.cv_bandwidth(x, z, f, BW_MULTS)
    sx = np.sort(np.unique(x))
    dmin = [Dm[x == v].min() for v in sx]
    dmax = [Dm[x == v].max() for v in sx]
    paths = conv_x
    for pc in paths:
        hull = est.make_hull(pc, sx, dmin, dmax)
        pth = est.local_path(x, z, f, hx, hz, pc, C_levels, hull)
        out[f"sig_mf_{pc}"] = float(np.nanmean(pth[:, 1])) if np.isfinite(pth[:, 1]).any() else np.nan
        out[f"n_levels_{pc}"] = int(np.isfinite(pth[:, 1]).sum())
    if want_q3:
        sub = is_main & (M <= 100)
        thr, _ = est.fit_chin(N[sub], D[sub], L[sub], "huber")
        xa, za, fa = np.log(N), np.log(D), np.log(L)
        (_, hxa, hza, _, _), _ = est.cv_bandwidth(xa, za, fa, BW_MULTS)
        c = est.local_coefs(xa, za, fa, list(zip(xa, za)), hxa, hza)
        lw = np.where((c[:, 1] < 0) & (c[:, 2] < 0), est.lnw_local(c), np.nan)
        q3 = est.q3_stat(N, D, M, lw, thr)
        out["q3_slope"], out["q3_mean"], out["q3_n"] = q3["slope"], q3["mean"], q3["n"]
        # same statistic without hiM points
        cm = est.local_coefs(x, z, f, list(zip(x, z)), hx, hz)
        lwm = np.where((cm[:, 1] < 0) & (cm[:, 2] < 0), est.lnw_local(cm), np.nan)
        q3m = est.q3_stat(Nm, Dm, M[is_main], lwm, thr)
        out["q3_slope_mainonly"] = q3m["slope"]
    return out


def ce_block(Ne, De, Le, Nw, Dw, Lw, clus_e, clus_w, B=0, rng=None, th_sep=None):
    """CE tilt (Huber) and, if B > 0, its wild-cluster bootstrap basic 95% interval (Webb weights by trunk; the same
    weight for a width in both corpora)."""
    N = np.r_[Nw, Ne]
    D = np.r_[Dw, De]
    L = np.r_[Lw, Le]
    g = np.r_[np.zeros(len(Nw), int), np.ones(len(Ne), int)]
    if th_sep is None:
        starts, th_sep = est.ce_starts(N, D, L, g)
    else:
        starts = [mc_embed(th_sep, g)]
    th, _ = est.fit_ce(N, D, L, g, "huber", starts=starts)
    chi = est.ce_tilt(th)
    res = dict(chi=chi, alpha=th[0], beta=th[1])
    if B:
        import m2_est as me
        yhat = est.pred_ce(th, N, D, g)
        e = np.log(L) - yhat
        cl = np.r_[clus_w, clus_e]
        uni = np.unique(cl)
        draws = []
        for _ in range(B):
            v = mc.cluster_weights(rng, cl, uni)
            Ls = np.exp(yhat + v * e)
            try:
                tb, _, _ = me.fit_panel_ls("CE", np.log(N), np.log(D), np.log(Ls), g, 2, "huber", starts=[th], tight=False)
                draws.append(est.ce_tilt(tb))
            except Exception:
                draws.append(np.nan)
        lo, hi = mc.basic_ci(chi, draws, chi)
        res.update(lo=lo, hi=hi, se_boot=mc.sd_(draws))
    return res


def mc_embed(th_sep, g):
    import m2_est as me
    P = np.vstack([th_sep[r] for r in np.asarray(g, int)])
    return me.embed("CE", P, np.asarray(g, int), 2)


def one_rep(item, design=None, tp=None, s=None, rho=None, chi=None, B=0, full=True):
    """item = replication index. Returns a flat dict of statistics."""
    rep, seed = item
    rng = np.random.default_rng(seed)
    out = dict(rep=rep)
    data = {}
    for r in ("web", "edu"):
        df = design[r]
        lt = true_lnL(tp, df.N_P.values, df.D.values, chi if r == "edu" else 0.0)
        e1 = noise(df, s, rho, rng)
        e2 = RHO_VAL * e1 + np.sqrt(1 - RHO_VAL ** 2) * noise(df, s, rho, rng)
        data[r] = (np.exp(lt + e1), np.exp(lt + e2))
    t0 = time.time()
    for conv in ("P", "T"):
        seps = {}
        for r in ("web", "edu"):
            df = design[r]
            N = df[f"N_{conv}"].values
            M = df[f"M_{conv}"].values
            if full:
                blk = _fit_block(N, df.D.values, data[r][0], ("P", "P6") if conv == "P" else ("T",), conv, M,
                                 df.is_main.values, C_LEVELS, want_q3=True)
                seps[r] = blk.pop("theta")
                for k, v in blk.items():
                    out[f"{k}|{r}|{conv}"] = v
        # corpus-pair tilt on both validation sets (main grids)
        for vi in (0, 1):
            de, dw = design["edu"], design["web"]
            me_, mw_ = de.is_main.values, dw.is_main.values
            ce = ce_block(de[f"N_{conv}"].values[me_], de.D.values[me_], data["edu"][vi][me_],
                          dw[f"N_{conv}"].values[mw_], dw.D.values[mw_], data["web"][vi][mw_],
                          de.arch.values[me_], dw.arch.values[mw_], B=B if conv == "P" else 0, rng=rng,
                          th_sep=[seps["web"], seps["edu"]] if (full and vi == 0) else None)
            for k, v in ce.items():
                out[f"ce_{k}|val{vi + 1}|{conv}"] = v
    out["secs"] = time.time() - t0
    return out


# ============================================================================ driver
CELLS = [  # (name, truth, s, rho, M_anchor, R, R_boot)
    ("chin_s002_r5", "chin", 0.002, 0.5, 20.0),
    ("chin_s005_r5", "chin", 0.005, 0.5, 20.0),
    ("chin_s010_r5", "chin", 0.010, 0.5, 20.0),
    ("chin_s005_r0", "chin", 0.005, 0.0, 20.0),
    ("chin_s005_r9", "chin", 0.005, 0.9, 20.0),
    ("chin_s005_M60", "chin", 0.005, 0.5, 60.0),
    ("kappa_s005_r5", "kappa", 0.005, 0.5, 20.0),
]
CHI_ALT = 0.22


def run(R=200, R_boot=80, B=199, log=mc.log):
    design = mc_design()
    Cc, Ns, Ds = anchor(design)
    log(f"power: anchor C_c = {Cc:.3e}, N* = {Ns:.3e}, D* = {Ds:.3e}; R = {R}, R_boot = {R_boot}, B = {B}")
    if mc.QUICK:
        R, R_boot, B = 8, 4, 49
    rows, reps = [], []
    for name, kind, s, rho, Ma in CELLS:
        tp = truth_params(kind, design, Ma)
        for chi in (CHI_ALT, 0.0):
            baseline = name in ("chin_s002_r5", "chin_s005_r5", "chin_s010_r5")
            if chi == 0.0 and not baseline:
                continue
            t0 = time.time()
            items = [(i, mc.seed_of(f"power|{name}|{chi}|{i}")) for i in range(R)]
            res = mc.pmap(one_rep, items, dict(design=design, tp=tp, s=s, rho=rho, chi=chi, B=0, full=True))
            bad = [r for r in res if "_error" in r]
            res = [r for r in res if "_error" not in r]
            if bad:
                log(f"  {name} chi={chi}: {len(bad)} failed replications, first: {bad[0]['_error']}")
            boot = []
            if baseline:
                items_b = [(i, mc.seed_of(f"powerboot|{name}|{chi}|{i}")) for i in range(R_boot)]
                boot = mc.pmap(one_rep, items_b, dict(design=design, tp=tp, s=s, rho=rho, chi=chi, B=B, full=False))
                boot = [r for r in boot if "_error" not in r]
            df = pd.DataFrame(res)
            df["cell"], df["chi_true"] = name, chi
            reps.append(df)
            bdf = pd.DataFrame(boot)
            rows += summarize(name, kind, s, rho, Ma, chi, tp, df, bdf)
            log(f"  {name} chi={chi}: {len(df)} reps + {len(bdf)} boot reps in {time.time() - t0:.0f}s")
    # noise-free evaluation (smoothing / misspecification bias of each estimator on the design), s = 1e-5
    for kind in ("chin", "kappa"):
        tp = truth_params(kind, design)
        r0 = one_rep((0, mc.seed_of(f"power|noisefree|{kind}")), design=design, tp=tp, s=1e-5, rho=0.5, chi=CHI_ALT)
        df0 = pd.DataFrame([r0])
        for rr in summarize(f"{kind}_noisefree", kind, 0.0, 0.5, M_ANCHOR, CHI_ALT, tp, df0, pd.DataFrame()):
            rr["mc_se"] = np.nan
            rows.append(rr)
    out = pd.DataFrame(rows)
    mc.write_csv(out, "m9_sweeps_power")
    pd.concat(reps, ignore_index=True).to_csv(os.path.join(mc.PROC, "power_replications.csv"), index=False)
    return out


def truth_sigma_P(tp):
    """sigma of the true technology on its actual-FLOP compute-optimal path (w = eta), mean over C_LEVELS that the
    model-free estimator reaches in the design (lnM* inside the grid)."""
    wf = lambda N, D: est.w_kappa([tp["lnA"], tp["lnB"], tp["lnE"], tp["a1"], tp["b1"], tp["k"]], N, D)
    pp = est.param_path("P", wf, tp["a1"], tp["b1"], C_LEVELS, 12.9, 17.7)
    ok = np.isfinite(pp[:, 1]) & (pp[:, 0] > 12.9) & (pp[:, 0] < 17.7)
    return float(np.mean(pp[ok, 1]))


def summarize(name, kind, s, rho, Ma, chi, tp, df, bdf):
    true_sig = 2 / (2 + tp["a1"] + tp["b1"])
    true_sig_P = truth_sigma_P(tp)
    rows = []

    def add(stat, corpus, conv, v, truth=np.nan, extra=None):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        d = dict(cell=name, truth=kind, noise_sd=s, rho_trunk=rho, M_anchor=Ma, chi_true=chi, statistic=stat,
                 corpus=corpus, conv=conv, n_rep=int(ok.sum()), mean=float(np.mean(v[ok])) if ok.any() else np.nan,
                 mc_se=float(np.std(v[ok], ddof=1)) if ok.sum() > 2 else np.nan, true_value=truth,
                 rmse=float(np.sqrt(np.mean((v[ok] - truth) ** 2))) if np.isfinite(truth) and ok.any() else np.nan)
        if extra:
            d.update(extra)
        rows.append(d)

    for conv in ("P", "T"):
        for r in ("edu", "web"):
            for st in ("sig_kappa", "sig_chin"):
                k = f"{st}|{r}|{conv}"
                if k in df:
                    tv = true_sig if conv == "P" and (st == "sig_kappa" or kind == "chin") else np.nan
                    add(st, r, conv, df[k], tv)
            for pc in (("P", "P6") if conv == "P" else ("T",)):
                k = f"sig_mf_{pc}|{r}|{conv}"
                if k in df:
                    tv = {"P": true_sig_P, "P6": true_sig}.get(pc, np.nan)
                    add(f"sig_modelfree_{pc}", r, conv, df[k], tv,
                        dict(levels_mean=float(df[f"n_levels_{pc}|{r}|{conv}"].mean())))
            for st in ("q3_slope", "q3_slope_mainonly"):
                k = f"{st}|{r}|{conv}"
                if k in df:
                    add(st, r, conv, df[k], np.nan, dict(n_points=float(df.get(f"q3_n|{r}|{conv}", pd.Series([np.nan])).mean())))
        # tilt
        for vi in (1, 2):
            k = f"ce_chi|val{vi}|{conv}"
            if k in df:
                v = df[k].values
                se = float(np.std(v, ddof=1))
                pw = float(np.mean(np.abs(v) / se > 1.96)) if chi != 0 else float(np.mean(np.abs(v - np.mean(v)) / se > 1.96))
                add("tilt_chi", f"val{vi}", conv, v, chi if (kind == "chin" and conv == "P") else np.nan,
                    dict(power_normal=pw, mde80_normal=2.80 * se))
        if len(bdf) and f"ce_lo|val1|{conv}" in bdf:
            lo1, hi1 = bdf[f"ce_lo|val1|{conv}"].values, bdf[f"ce_hi|val1|{conv}"].values
            lo2, hi2 = bdf[f"ce_lo|val2|{conv}"].values, bdf[f"ce_hi|val2|{conv}"].values
            ex1 = (lo1 > 0) | (hi1 < 0)
            ex2 = (lo2 > 0) | (hi2 < 0)
            same = np.sign(lo1 + hi1) == np.sign(lo2 + hi2)
            biased = ex1 & ex2 & same
            neutral = (~ex1) & (~ex2) & ((hi1 - lo1) / 2 < 0.10) & ((hi2 - lo2) / 2 < 0.10)
            add("tilt_chi_bootstrap", "val1+val2", conv, bdf[f"ce_chi|val1|{conv}"], chi if kind == "chin" else np.nan,
                dict(reject_val1=float(np.mean(ex1)), decide_factor_biased=float(np.mean(biased)),
                     decide_neutral=float(np.mean(neutral)), decide_inconclusive=float(np.mean(~biased & ~neutral)),
                     ci_halfwidth_mean=float(np.mean((hi1 - lo1) / 2)),
                     se_boot_mean=float(np.nanmean(bdf[f"ce_se_boot|val1|{conv}"])),
                     cover_true=float(np.mean((lo1 <= chi) & (hi1 >= chi)))))
    return rows


if __name__ == "__main__":
    import sys
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    Rb = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    run(R=R, R_boot=Rb)
