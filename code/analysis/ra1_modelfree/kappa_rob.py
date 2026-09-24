"""kappa_rob.py -- robustness of the kappa-free elasticity sigma*_kappa (Task 5; R2 Major 8) and design-conditional
standard errors for the seven sweep-corpus technologies (inputs to the heterogeneity analysis, Task 3).

For each technology: kappa-free fit L = E + (A N^-a1 + B D^-b1)^kappa by Huber (delta = 1e-3) on ln L (m2_est.fit_q,
started from the Chinchilla-form fit), sigma*_kappa = 2/(2 + a1 + b1). Variants:
  baseline; drop the smallest model sizes; conventions (Farseer N incl. embeddings); output (OLMo task BPB);
  sample (Chinchilla with the five highest-loss runs; Muennighoff D/N >= 0.4; Farseer M <= 1,000 / M >= 2);
  E fixed at the Chinchilla-form estimate; and the E-kappa trade-off: profile over E (fit the other parameters at each
  E; Laplace quasi-LR relative to the unconstrained optimum), with the range of sigma*_kappa over the E values inside
  the chi2_1 95% set (descriptive; the set is not bootstrap-calibrated).
Standard errors: design-conditional wild bootstrap (Feng-He-Hu weights on leverage/sparsity-adjusted residuals of the
kappa-free fit); Webb wild cluster by model size for Farseer (residual ICC 0.27 by size) and by (size, multiplier)
cell for the OLMo ladder. B = 499 for baseline rows, 199 for variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import parametric as pm
import kprofile as pf
import ra1_common as rc

B_BASE, B_VAR = 499, 199
if rc.QUICK:
    B_BASE, B_VAR = 20, 10


def fit_one(N, D, L):
    th, fth = pm.fit_chin(N, D, L)
    p, fp = pm.fit_kappa(N, D, L, th_chin=th)
    return th, fth, p, fp


def _wild_one(item, V, seed_base):
    import m2_est as me
    j, i = item
    v = V[j]
    rng = np.random.default_rng(seed_base[j] + i)
    if v["webb"]:
        ug, inv = np.unique(v["clus"], return_inverse=True)
        w = rc.webb(rng, len(ug))[inv]
    else:
        w = rc.fhh(rng, len(v["yhat"]))
    Lb = np.exp(v["yhat"] + w * np.abs(v["rfhh"]))
    best = None
    for st in (v["p"], np.r_[v["th"], 1.0]):
        r = me.fit_q(v["N"], v["D"], Lb, "huber", init=st)
        if best is None or r[1] < best[1]:
            best = r
    return dict(j=j, sigma_kappa=pm.sigma_star_kappa(best[0]), kappa=best[0][5])


def kappa_jac(p, N, D, eps=1e-6):
    base = pm.pred_kappa(p, N, D)
    J = []
    for k in range(6):
        q = np.array(p, float).copy()
        q[k] += eps
        J.append((pm.pred_kappa(q, N, D) - base) / eps)
    return np.column_stack(J)


def e_profile(N, D, L, p, n_grid=36):
    kd = pf.KappaData(N, D, L)
    full_u = kd.from_raw(p)
    fu = pf.fit_free(kd, [full_u], "huber")
    E_hat = float(np.exp(fu[0][2]))
    Lmin = float(np.min(L))
    grid = np.unique(np.r_[np.linspace(0.02, 0.995, n_grid) * Lmin, E_hat])
    fulls, objs = [], []
    prev = fu[0]
    for E in grid:                      # forward sweep, warm-started
        full, f = pf.fit_E(kd, E, [fu[0], prev])
        prev = full
        fulls.append(full)
        objs.append(f)
    for i in range(len(grid) - 2, -1, -1):   # backward sweep
        full, f = pf.fit_E(kd, grid[i], [fulls[i + 1]])
        if f < objs[i] - 1e-14:
            fulls[i], objs[i] = full, f
    # Review fix (m-F): the 36-point grid (spacing ~0.028 min L) resolves the chi2 E set with only 2-7 points, and
    # sigma*_kappa at the first excluded neighbours can differ materially (Gadre RedPajama). Refine the grid with 25
    # points between the excluded neighbours of the coarse set, warm-started from the nearest coarse fits.
    fmin0 = min(fu[1], min(objs))
    qlr0 = 2 * len(L) * np.log(np.array(objs) / fmin0)
    ins = np.where(qlr0 <= 3.8415)[0]
    fine = np.array([])
    if len(ins):
        lo_i, hi_i = max(ins.min() - 1, 0), min(ins.max() + 1, len(grid) - 1)
        fine = np.setdiff1d(np.linspace(grid[lo_i], grid[hi_i], 27)[1:-1], grid)
    for E in fine:
        j = int(np.argmin(np.abs(grid - E)))
        full, f = pf.fit_E(kd, E, [fulls[j], fu[0]])
        fulls.append(full)
        objs.append(f)
    Eall = np.r_[grid, fine]
    df = pd.DataFrame(dict(E=Eall, obj=objs, sigma_kappa=[2 / (2 + q[3] + q[4]) for q in fulls],
                           kappa=[float(np.exp(q[5])) for q in fulls], fine=np.r_[np.zeros(len(grid), bool),
                                                                                  np.ones(len(fine), bool)]))
    df = df.sort_values("E").reset_index(drop=True)
    fmin = min(fu[1], df.obj.min())
    df["QLR"] = 2 * len(L) * np.log(df.obj / fmin)
    return df, E_hat


def variants(P):
    V = []
    for name, d in P.items():
        base = dict(tech=name, N=d.N.values.astype(float), D=d.D.values.astype(float), L=d.L.values.astype(float))
        clus = d["N"].values if name == "Farseer" else (d["cluster"].values if name == "OLMo ladder" else None)
        webb = name in ("Farseer", "OLMo ladder")
        V.append(dict(base, variant="baseline", clus=clus, webb=webb, B=B_BASE))
        sizes = np.sort(d.N.unique())
        if name == "Chinchilla":
            for q in (10, 25):
                s = d.N.values >= np.percentile(d.N.values, q)
                V.append(dict(tech=name, variant=f"drop runs with N below its {q}th percentile", N=base["N"][s],
                              D=base["D"][s], L=base["L"][s], clus=None, webb=False, B=B_VAR))
        else:
            for k in ((1, 2, 5) if name == "Farseer" else (1, 2) if len(sizes) > 4 else (1,)):
                s = d.N.values > sizes[k - 1]
                V.append(dict(tech=name, variant=f"drop the {k} smallest model size{'s' if k > 1 else ''}", N=base["N"][s],
                              D=base["D"][s], L=base["L"][s], clus=None if clus is None else clus[s], webb=webb, B=B_VAR))
        if name == "Farseer":
            V.append(dict(tech=name, variant="N incl. both embeddings", N=d.N_emb.values.astype(float), D=base["D"],
                          L=base["L"], clus=clus, webb=True, B=B_VAR))
            for lab, s in (("drop M > 1,000", d.M.values <= 1000), ("drop M < 2", d.M.values >= 2)):
                V.append(dict(tech=name, variant=lab, N=base["N"][s], D=base["D"][s], L=base["L"][s], clus=clus[s],
                              webb=True, B=B_VAR))
        if name == "OLMo ladder":
            V.append(dict(tech=name, variant="output: task bits per byte", N=base["N"], D=base["D"],
                          L=d.L_bpb.values.astype(float), clus=clus, webb=True, B=B_VAR))
        if name == "Muennighoff":
            s = d.M.values >= 0.4
            V.append(dict(tech=name, variant="drop D/N < 0.4", N=base["N"][s], D=base["D"][s], L=base["L"][s], clus=None,
                          webb=False, B=B_VAR))
        if name == "Chinchilla":
            ch = rc.sl.chinchilla_extraction(path=f"{rc.RAW}/epoch_chinchilla/svg_extracted_data.csv", drop_worst=0)
            V.append(dict(tech=name, variant="all 245 runs (outliers kept)", N=ch.N.values, D=ch.D.values, L=ch.L.values,
                          clus=None, webb=False, B=B_VAR))
    return V


def run(log=rc.log):
    P = rc.sweep_panels()
    rows, eprof, V = [], [], []
    base_th = {}
    for v in variants(P):
        N, D, L = v["N"], v["D"], v["L"]
        if v["variant"] == "baseline":
            th, fth, p, fp = fit_one(N, D, L)
            base_th[v["tech"]] = th
        else:   # warm starts from the technology's baseline fit (plus m2's level-preserving starts)
            import m2_est as me
            th, fth = pm.fit_chin(N, D, L, starts=[base_th[v["tech"]]] + me.level_starts(N, D, L), grid=None)
            p, fp = pm.fit_kappa(N, D, L, th_chin=th)
        row = dict(tech=v["tech"], variant=v["variant"], n=len(L), n_sizes=len(np.unique(np.round(np.log(N), 4))),
                   sigma_kappa=pm.sigma_star_kappa(p), kappa=p[5], a1=p[3], b1=p[4], E_kappa=float(np.exp(p[2])),
                   sigma_chin=pm.sigma_star_chin(th), E_chin=float(np.exp(th[2])),
                   qlr_kappa1=2 * len(L) * np.log(fth / fp))
        # E fixed at the Chinchilla-form E (the E-kappa trade-off in one number)
        kd = pf.KappaData(N, D, L)
        fullE, _ = pf.fit_E(kd, float(np.exp(th[2])) if np.exp(th[2]) < L.min() else 0.999 * L.min(), [kd.from_raw(p)])
        row["sigma_kappa_E_chin"] = 2 / (2 + fullE[3] + fullE[4])
        row["kappa_E_chin"] = float(np.exp(fullE[5]))
        row["inference"] = ("wild cluster (Webb), " + ("model size" if v["tech"] == "Farseer" else "(size, multiplier) cell")
                            if v["webb"] else "wild (Feng-He-Hu), run level")
        if v["variant"] == "baseline":
            ep, Ehat = e_profile(N, D, L, p)
            ep["tech"] = v["tech"]
            eprof.append(ep)
            ins = ep[ep.QLR <= 3.8415]
            row.update(E_set_lo=float(ins.E.min()), E_set_hi=float(ins.E.max()),
                       sigma_kappa_over_E_set_lo=float(ins.sigma_kappa.min()),
                       sigma_kappa_over_E_set_hi=float(ins.sigma_kappa.max()),
                       sigma_kappa_E_grid_min=float(ep.sigma_kappa.min()), sigma_kappa_E_grid_max=float(ep.sigma_kappa.max()))
        rows.append(row)
        yhat = pm.pred_kappa(p, N, D)
        rf, _, _ = pm.fhh_residuals(np.log(L) - yhat, kappa_jac(p, N, D))
        V.append(dict(N=N, D=D, yhat=yhat, rfhh=rf, p=p, th=th, webb=v["webb"],
                      clus=v["clus"] if v["clus"] is not None else np.arange(len(L)), B=v["B"]))
        log(f"  kappa robustness {v['tech']:18s} {v['variant']:42s} n={len(L):4d} sigma*_k {row['sigma_kappa']:.3f} "
            f"kappa {p[5]:.3f} | E fixed at the Chinchilla-form E: {row['sigma_kappa_E_chin']:.3f}")
    # one parallel pass over all (variant, draw) pairs
    items = [(j, i) for j, v in enumerate(V) for i in range(v["B"])]
    seed_base = [rc.seed_of(f"kwild|{r['tech']}|{r['variant']}") for r in rows]
    res = [x for x in rc.pmap(_wild_one, items, dict(V=V, seed_base=seed_base), chunksize=8) if "_error" not in x]
    R = pd.DataFrame(res)
    for j, row in enumerate(rows):
        s = R.loc[R.j == j, "sigma_kappa"].values
        row.update(se_sigma_kappa=float(np.std(s, ddof=1)), lo=rc.pct(s, 2.5), hi=rc.pct(s, 97.5), B=len(s))
    log(f"  kappa robustness: {len(res)} wild draws over {len(rows)} variants")
    return pd.DataFrame(rows), pd.concat(eprof, ignore_index=True)
