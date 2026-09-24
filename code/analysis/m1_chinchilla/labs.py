"""Lab-own technologies from IsoFLOP compilations (open-athena/isoflop-experiments): Meta's Llama 3 (Czech
digitization of Grattafiori et al. 2024, Fig. 2), Marin 2026-03 ladders (3 corpora), and the (Mis)Fitting FineWeb/C4
sweep (checkpoint-interpolated IsoFLOPs; reported with a caveat).

For each: Approach 3 (Huber-LSE primal, all IsoFLOP runs), Approach 2 (per-budget parabola argmins, Meta's own
procedure) and Approach 1 (frontier through the minima), the cross-equation tests of duality.py, the system estimator,
and a cluster (budget) bootstrap of the primal. The Approach-2/1 technology is also mapped into (alpha, beta) under the
kappa = 1 restriction: alpha = gamma/a, beta = gamma/(1-a) (model_spec Proposition 1).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import boot
import duality
import estimators as est
from common import PROC, SEED, TABLES, derived, load_isoflop, sl, theta_of

EXPERIMENTS = {
    "llama_3": "Meta Llama 3 IsoFLOPs",
    "marin_202603__comma__llama_2": "Marin 2026-03, Comma",
    "marin_202603__dclm__llama_2": "Marin 2026-03, DCLM",
    "marin_202603__nemotron__llama_2": "Marin 2026-03, Nemotron-CC",
    "misfitting__fineweb_c4__transformer": "(Mis)Fitting FineWeb/C4",
}
META_PUB = dict(b=0.537, coef=0.299)   # Llama 3 Fig. 3 label: D*(C) = 0.299 C^0.537 (tokens); text: (0.53, 0.29)
B_LAB = 300


def _boot_a3(idx, N, D, L, inits, **_):
    m = est.fit_huber(N[idx], D[idx], L[idx], inits=inits)
    d = derived(m)
    d.update(lnA=np.log(m.A), lnB=np.log(m.B), lnE=np.log(m.E))
    return d


def run(B=B_LAB, log=print):
    rows, tests, objs, sysrows, draws, o2s = [], [], [], [], {}, {}
    for i, (exp, lab) in enumerate(EXPERIMENTS.items()):
        df = load_isoflop(exp)
        lnC_ref = float(np.mean(np.log(df.C.values)))
        r = duality.duality_for(df, lab, lnC_ref, B, SEED + 2000 + 17 * i, lit=None, do_system=True, log=log,
                                grid=sl.DEFAULT_GRID)
        m3 = r["m3"]
        # cluster bootstrap (budget) of the primal
        idxs = boot.draws(boot.groups_of(df.cl.values), B, "cluster", SEED + 2100 + 17 * i)
        out = boot.pmap(_boot_a3, idxs, dict(N=df.N.values, D=df.D.values, L=df.L.values,
                                               inits=[theta_of(m3), theta_of(sl.BESIROGLU), theta_of(sl.HOFFMANN)]))
        keys = ["E", "A", "B", "alpha", "beta", "a", "gamma", "sigma_star", "Mstar_1e+21", "Mstar_5.76e+23", "Mstar_1e+26",
                "lnG", "lnK"]
        scl = boot.summarize(out, keys)
        # stratified draws of the primal (from duality_for) -> covariance for the registry
        reps = r["reps"]
        Th = np.array([[np.log(q["A3_A"]), np.log(q["A3_B"]), np.log(q["A3_E"]), q["A3_alpha"], q["A3_beta"]] for q in reps])
        draws[(exp, "A3_huber", "strat")] = Th
        draws[(exp, "A3_huber", "cluster")] = boot.mat(out, ["lnA", "lnB", "lnE", "alpha", "beta"])
        d3 = derived(m3)
        row = dict(experiment=exp, label=lab, n=len(df), n_budgets=df.budget.nunique(), approach="A3 primal (Huber-LSE)")
        row.update(d3)
        for k in keys:
            row[f"se_strat_{k}"] = float(np.std([q[f"A3_{k}"] for q in reps], ddof=1)) if f"A3_{k}" in reps[0] else np.nan
            row[f"se_cluster_{k}"] = scl[k]["se"]
            row[f"rse_cluster_{k}"] = scl[k]["rse"]
        row["Mstar_3.8e25"] = float(m3.D_opt(3.8e25) / m3.N_opt(3.8e25))
        rows.append(row)
        # Approach 2 + 1 technology, mapped to (alpha, beta) under kappa = 1
        o2 = r["o2"]
        o2s[exp] = o2
        a, g = o2["a"], o2["gamma"]
        A2 = np.array([[q["A2_a"], q["A2_gamma"], q["A2_E_A1"], q["A2_lnK"], q["A2_lnG"]] for q in reps])
        al_d, be_d = A2[:, 1] / A2[:, 0], A2[:, 1] / (1 - A2[:, 0])
        C38 = 3.8e25
        Mst = lambda aa, lnG, C: np.exp((1 - 2 * aa) * np.log(C / 6) - 2 * lnG)
        row2 = dict(experiment=exp, label=lab, n=len(df), n_budgets=df.budget.nunique(),
                    approach="A2 argmins + A1 frontier (alpha, beta via kappa = 1)",
                    a=a, gamma=g, E=o2["E_A1"], lnK=o2["lnK"], lnG=o2["lnG"], G=np.exp(o2["lnG"]), K=np.exp(o2["lnK"]),
                    alpha=g / a, beta=g / (1 - a), sigma_star=2 / (2 + g / a + g / (1 - a)),
                    **{f"Mstar_{C:.3g}": float(Mst(a, o2["lnG"], C)) for C in (1e21, 5.76e23, 1e26)},
                    **{"Mstar_3.8e25": float(Mst(a, o2["lnG"], C38))},
                    se_strat_a=float(np.std(A2[:, 0], ddof=1)), se_strat_gamma=float(np.std(A2[:, 1], ddof=1)),
                    se_strat_E=float(np.std(A2[:, 2], ddof=1)), se_strat_lnG=float(np.std(A2[:, 4], ddof=1)),
                    se_strat_lnK=float(np.std(A2[:, 3], ddof=1)),
                    se_strat_alpha=float(np.std(al_d, ddof=1)), se_strat_beta=float(np.std(be_d, ddof=1)),
                    se_strat_sigma_star=float(np.std(2 / (2 + al_d + be_d), ddof=1)),
                    rse_strat_a=float(boot.rse(A2[:, 0])), rse_strat_gamma=float(boot.rse(A2[:, 1])),
                    rse_strat_alpha=float(boot.rse(al_d)), rse_strat_beta=float(boot.rse(be_d)),
                    rse_strat_lnG=float(boot.rse(A2[:, 4])), rse_strat_E=float(boot.rse(A2[:, 2])),
                    path_resid_sd=r["extra"]["A2_path_resid_sd"])
        if exp == "llama_3":
            mins = o2["mins"]
            lnD = mins[:, 0] - np.log(6) - mins[:, 1]
            bb, cc = np.polyfit(mins[:, 0], lnD, 1)
            row2.update(meta_repro_D_coef=float(np.exp(cc)), meta_repro_D_exp=float(bb),
                        meta_pub_D_coef=META_PUB["coef"], meta_pub_D_exp=META_PUB["b"],
                        meta_pub_Dstar_3_8e25=META_PUB["coef"] * C38 ** META_PUB["b"],
                        meta_repro_Dstar_3_8e25=float(np.exp(cc) * C38 ** bb))
        rows.append(row2)
        draws[(exp, "A2A1", "strat")] = A2
        tests.append(r["tests"])
        objs.append(r["objs"])
        for lik, s in r["system"].items():
            q = dict(experiment=exp, label=lab, likelihood=lik, LR=s["LR"], LR_p=s["LR_p"], n_loss=s["n1"], n_argmin=s["n2"])
            q.update(s["derived"])
            q.update({f"se_{k}": v for k, v in s["se"].items()})
            sysrows.append(q)
        log(f"  lab {lab}: A3 a={m3.a_N:.3f} (A2 a={a:.3f}); A3 gamma={m3.gamma:.3f} (A1 {g:.3f}); "
            f"alpha={m3.alpha:.3f} ({scl['alpha']['se']:.3f}) beta={m3.beta:.3f} ({scl['beta']['se']:.3f})")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TABLES, "m1_chinchilla_labs_technology.csv"), index=False)
    duality.save_minima(o2s, "m1_chinchilla_labs_a2_minima.csv")
    tt = pd.concat(tests, ignore_index=True)
    tt.to_csv(os.path.join(TABLES, "m1_chinchilla_labs_duality_tests.csv"), index=False)
    pd.concat(objs, ignore_index=True).to_csv(os.path.join(TABLES, "m1_chinchilla_labs_duality_objects.csv"), index=False)
    pd.DataFrame(sysrows).to_csv(os.path.join(TABLES, "m1_chinchilla_labs_system.csv"), index=False)
    for (exp, a, sch), M in draws.items():
        np.save(os.path.join(PROC, f"boot_{exp}_{a}_{sch}.npy"), M)
    return tab, tt, pd.DataFrame(sysrows), draws
