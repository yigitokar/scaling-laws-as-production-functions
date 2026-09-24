"""Step 2 of ra4_obsfix: allocative-efficiency gains with the wedge truncated at one (R1 comment 10(d); R4 M4(d)).

Farrell cost efficiency against the training-only frontier, CE_i = C_min(L(N_i, D_i)) / 6 N_i D_i, is a function of
the model's wedge w_i = eps_N/eps_D alone (Chinchilla family; E and loss units cancel), with CE(1) = 1.  Deliberate
over-training (w > 1) lowers CE but can be rationalized by inference demand (Proposition 2); w < 1 cannot
(Proposition 2(v)).  The truncated measure counts only the elimination of under-training:

    CE_i^tr = CE(min(w_i, 1)) = CE_i if w_i < 1, and 1 otherwise.

The Kaplan->Chinchilla realized gain is the ratio of era geometric means (2022-24 vs 2020-21) of CE^tr, reported
next to the untruncated gain of m5_progress (Table 9 panel B) for every technology the project has estimated, with
provenance.  Reuses m5_progress (s4_allocative.load_epoch, cleaned_mask, kaplan_ce, farseer_technology; the
Besiroglu bootstrap draws and Ho et al.'s paper-cluster bootstrap draws cached by m5) by import.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ra4common as rc  # noqa: E402

sys.path.insert(0, rc.M5DIR)
import common as m5c  # noqa: E402  (m5_progress/common.py)
import s4_allocative as s4  # noqa: E402

import sl  # noqa: E402

M5PROC = os.path.join(rc.ROOT, "data", "processed", "m5_progress")
SAMPLES = ["C >= 1e23", "C >= 1e23, cleaned", "Top-5 compute per year", "Epoch frontier flag"]


def technologies():
    """(label -> (sl.Chinchilla, provenance dict)).  Registry technologies are read from the m1/m2 registries as they
    exist at run time (file hashes are recorded)."""
    T = {}
    T["Besiroglu et al. (2024)"] = (m5c.BESIROGLU, dict(group="Chinchilla sweep", source="published refit (Besiroglu et al. 2024)",
                                                       where="Section III (reference technology)", N_conv="total"))
    T["Hoffmann et al. (2022), TeX"] = (m5c.HOFFMANN_TEX, dict(group="Chinchilla sweep", source="published Approach 3",
                                                               where="Section III", N_conv="total"))
    T["Hoffmann et al. (2022), rounded"] = (m5c.HOFFMANN_RND, dict(group="Chinchilla sweep", source="published Eq. 10",
                                                                   where="Section III", N_conv="total"))
    info = {}
    # kappa-free Chinchilla: the wedge and CE depend only on the inner aggregator (R4 M2 recomputation)
    reg2 = os.path.join(rc.TAB, "technology_registry_m2.csv")
    info["technology_registry_m2.csv"] = rc.file_info(reg2)
    info["technology_registry_m1.csv"] = rc.file_info(os.path.join(rc.TAB, "technology_registry_m1.csv"))
    if os.path.exists(reg2):
        r2 = pd.read_csv(reg2)
        g = r2[(r2.dataset == "chinchilla") & (r2.role == "generalized_form")]
        if len(g):
            g = g.iloc[0]
            T["Chinchilla, kappa free (inner exponents)"] = (
                sl.Chinchilla(E=float(g.E), A=float(g.A), B=float(g.B), alpha=float(g.alpha), beta=float(g.beta)),
                dict(group="Chinchilla sweep", source="m2 registry, generalized (kappa-free) Huber fit",
                     where="Section III (Table 4, kappa free)", N_conv="total"))
    far = s4.farseer_technology(cache=True)
    T["Farseer grid (m5 fit, N incl. embeddings)"] = (far, dict(group="Farseer", source="m5 fit of the 404-run grid",
                                                                where="Online Appendix E (m5)", N_conv="total incl. embeddings"))
    regt, reginfo = s4.registry_chinchillas()
    where = {"m1": "Online Appendix D (m1 registry; not estimated in Section III)", "m2": "Section III / Table 4 (m2 registry)"}
    for lab, tech in regt.items():
        tag = lab[:2]
        T[lab] = (tech, dict(group=f"{tag} registry", source=f"{tag} registry, primary Huber row", where=where.get(tag, ""),
                             N_conv="sweep convention (see registry)"))
    return T, info


def ho_setup():
    df = m5c.load_ho()
    d = m5c.ho_arrays(df)
    m = m5c.HoModel(m5c.HoSpec(delta=0.0025), d)
    x = m5c.HO_M7_X
    gC = m.rates(x)["g_C"]
    p = os.path.join(M5PROC, "boot_m7_cluster.npy")
    if os.path.exists(p):
        bt = np.load(p)
        gC_draws = np.array([m.rates(b[:-1])["g_C"] for b in bt])
        src = rc.file_info(p)
    else:   # fall back to the point rate only
        gC_draws, src = np.array([gC]), dict(file=p, exists=False)
    return m, x, gC, gC_draws, src


def besiroglu_draws():
    p = os.path.join(M5PROC, "besiroglu_boot_theta.npy")
    mine = os.path.join(rc.PROC, "besiroglu_boot_theta.npy")
    # reviewer fix: prefer m5's current file (so the draws cannot go stale and the provenance is the same on every run);
    # the local copy is only a fallback when m5's cache is absent
    if os.path.exists(p):
        th = np.load(p)
        np.save(mine, th)
        return th, dict(rc.file_info(p), copied_to=os.path.relpath(mine, rc.ROOT))
    if os.path.exists(mine):
        return np.load(mine), rc.file_info(mine)
    # recompute exactly as m5 does (pairs bootstrap of the Chinchilla extraction, seed 13, 200 draws)
    df = sl.chinchilla_extraction(os.path.join(rc.RAW, "epoch_chinchilla", "svg_extracted_data.csv"))
    full = sl.fit_chinchilla(df.N, df.D, df.L, init=m5c.BESIROGLU.theta)
    rng = np.random.default_rng(13)
    th = []
    for _ in range(200):
        idx = rng.integers(0, len(df), len(df))
        dd = df.iloc[idx]
        th.append(sl.fit_chinchilla(dd.N, dd.D, dd.L, init=full.theta).theta)
    th = np.array(th)
    np.save(mine, th)
    return th, rc.file_info(mine)


def ce_and_w(T, N, D):
    ce = T.cost_efficiency(N, D)
    w = T.wedge(N, D)
    return ce, w, np.where(w < 1, ce, 1.0)


def main():
    t0 = time.time()
    ep, steps = s4.load_epoch()
    T, reginfo = technologies()
    hm, hx, gC_ho, gC_draws, ho_src = ho_setup()
    thetas, th_src = besiroglu_draws()
    e1 = ((ep.date >= s4.ERA1[0]) & (ep.date <= s4.ERA1[1])).to_numpy()
    e2 = ((ep.date >= s4.ERA2[0]) & (ep.date <= s4.ERA2[1])).to_numpy()
    big = (ep.C >= s4.C_LARGE).to_numpy()
    smask = {"C >= 1e23": big, "C >= 1e23, cleaned": big & s4.cleaned_mask(ep).to_numpy(),
             "Top-5 compute per year": ep.top5_year.to_numpy(), "Epoch frontier flag": ep.frontier_flag.to_numpy()}
    N, D = ep.N.to_numpy(float), ep.D.to_numpy(float)
    per_model = ep[["Model", "Organization", "date", "N", "D", "C", "M"]].copy()
    rows = []
    tech_items = list(T.items())
    # Ho et al. model 7 technology, evaluated at each model's date (attenuated exponents; WT103 word units)
    ho_ce = np.array([hm.technology(hx, y, bench="wt103").cost_efficiency(n, d) for y, n, d in zip(ep.year, N, D)])
    ho_w = np.array([hm.technology(hx, y, bench="wt103").wedge(n, d) for y, n, d in zip(ep.year, N, D)])
    for lab, (tech, prov) in tech_items + [("Ho et al. (2024) model 7", (None, dict(group="Cross-lab regression",
                                           source="Ho et al. model 7 at each release date", where="Online Appendix E",
                                           N_conv="Ho sheet (word-level losses)")))]:
        if tech is None:
            ce, w = ho_ce, ho_w
            ce_tr = np.where(w < 1, ce, 1.0)
        else:
            ce, w, ce_tr = ce_and_w(tech, N, D)
        per_model[f"w|{lab}"] = w
        per_model[f"CE|{lab}"] = ce
        for sname, sm in smask.items():
            i1, i2 = np.where(e1 & sm)[0], np.where(e2 & sm)[0]
            if len(i1) < 2 or len(i2) < 2:
                continue
            gm = lambda v, ii: float(np.exp(np.mean(np.log(v[ii]))))
            g_u = gm(ce, i2) / gm(ce, i1)
            g_t = gm(ce_tr, i2) / gm(ce_tr, i1)
            dt = float(ep.year.to_numpy()[i2].mean() - ep.year.to_numpy()[i1].mean())
            algo = float(np.exp(gC_ho * dt))
            rng = np.random.default_rng([rc.SEED, zlib.crc32(f"{sname}|{lab}".encode())])
            bu, bt_, sh, gap_share = [], [], [], []
            for b in range(rc.B_ALLOC):
                j1, j2 = rng.choice(i1, len(i1)), rng.choice(i2, len(i2))
                if lab.startswith("Besiroglu"):
                    Tb = sl.Chinchilla.from_theta(thetas[b % len(thetas)])
                    c1, w1_, t1 = ce_and_w(Tb, N[j1], D[j1])
                    c2, w2_, t2 = ce_and_w(Tb, N[j2], D[j2])
                else:
                    c1, c2, t1, t2 = ce[j1], ce[j2], ce_tr[j1], ce_tr[j2]
                gu = np.exp(np.log(c2).mean() - np.log(c1).mean())
                gt = np.exp(np.log(t2).mean() - np.log(t1).mean())
                bu.append(gu)
                bt_.append(gt)
                gcb = gC_draws[b % len(gC_draws)]
                sh.append(np.log(gt) / (gcb * dt) if gcb > 0 else np.nan)
                gap_share.append(np.log(gt) / -np.log(np.exp(np.log(t1).mean())) if np.log(t1).mean() < 0 else np.nan)
            q = lambda v: np.nanpercentile(v, [2.5, 97.5])
            gap1 = 1.0 / gm(ce_tr, i1)                       # era-1 compute lost to under-training (gm)
            rows.append(dict(sample=sname, technology=lab, **{f"prov_{k}": v for k, v in prov.items()},
                             n_era1=len(i1), n_era2=len(i2), medM_era1=float(np.median(ep.M.to_numpy()[i1])),
                             medM_era2=float(np.median(ep.M.to_numpy()[i2])),
                             share_w_lt1_era1=float(np.mean(w[i1] < 1)), share_w_lt1_era2=float(np.mean(w[i2] < 1)),
                             gmCE_era1=gm(ce, i1), gmCE_era2=gm(ce, i2), gmCEtr_era1=gm(ce_tr, i1), gmCEtr_era2=gm(ce_tr, i2),
                             gain_untrunc=g_u, gain_untrunc_lo=q(bu)[0], gain_untrunc_hi=q(bu)[1],
                             gain_trunc=g_t, gain_trunc_lo=q(bt_)[0], gain_trunc_hi=q(bt_)[1],
                             overtraining_component=g_u / g_t, era1_undertraining_cost=gap1,
                             share_of_era1_gap_closed=float(np.log(g_t) / np.log(gap1)) if gap1 > 1 else np.nan,
                             share_of_era1_gap_closed_lo=q(gap_share)[0], share_of_era1_gap_closed_hi=q(gap_share)[1],
                             years_between=dt, ho_algorithmic_gain=algo,
                             share_of_ho_trunc=float(np.log(g_t) / np.log(algo)), share_of_ho_trunc_lo=q(sh)[0],
                             share_of_ho_trunc_hi=q(sh)[1], share_of_ho_untrunc=float(np.log(g_u) / np.log(algo)),
                             B=rc.B_ALLOC))
        rc.log(f"  {lab} done ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_allocative.csv"), index=False)
    per_model.to_csv(os.path.join(rc.PROC, "allocative_per_model.csv"), index=False)

    # Kaplan-rule counterfactual (Gundlach et al.'s CEG of rebalancing) and the wedge of the Kaplan allocation
    Cs = np.array([3.14e23, 3.8e25, 5e26])
    kap = []
    for lab, (tech, prov) in tech_items:
        ce_k, Nk, Dk = s4.kaplan_ce(tech, Cs)
        wk = tech.wedge(Nk, Dk)
        for c, v, ww in zip(Cs, ce_k, wk):
            kap.append(dict(technology=lab, C=c, CEG=1 / v, w_kaplan=ww, truncation_binds=bool(ww > 1)))
    K = pd.DataFrame(kap)
    K.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_kaplan_counterfactual.csv"), index=False)
    rc.dump_json(dict(runtime_sec=time.time() - t0, sample_steps=steps, n_sample=int(len(ep)), gC_ho=gC_ho,
                      TC_ho_point=12 * np.log(2) / gC_ho, n_gC_draws=int(len(gC_draws)), ho_boot_source=ho_src,
                      besiroglu_draws_source=th_src, registries=reginfo, B=rc.B_ALLOC,
                      technologies={k: v[1] for k, v in T.items()}), "allocative_headline.json")
    rc.log(f"step_alloc finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
