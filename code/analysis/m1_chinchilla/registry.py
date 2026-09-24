"""Task 7 -- technology registry for downstream modules: output/tables/technology_registry_m1.csv.

One row per (dataset, subset, estimator). Parameter covariance (bootstrap, theta = ln A, ln B, ln E, alpha, beta) is
saved as data/processed/m1_chinchilla/cov_<row_id>.npy and the underlying draws as boot_*.npy (paths in the CSV).
SE columns are bootstrap SDs of the level parameters (E, A, B, alpha, beta); the `se_scheme` column says which
bootstrap (pairs / cluster by IsoFLOP budget / within-budget stratified). A and B have heavily skewed bootstrap
distributions: use the covariance of (ln A, ln B) for any delta-method work.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import BESI_PUB, HOFF_ROUND, HOFF_TEX, PROC, ROOT, TABLES, derived

CHIN_CONV = dict(N_convention="total parameters incl. embeddings (Hoffmann et al. 2022)",
                 D_convention="tokens = C/(6N), C read off Hoffmann Fig. 4 (single epoch)",
                 loss_units="nats/token, MassiveText validation loss (Hoffmann tokenizer); digitized, +/-0.01")
LAB_CONV = {
    "llama_3": dict(N_convention="N = C/(6D) (constructed from the nominal budget)",
                    D_convention="training tokens read off Llama 3 Fig. 2 x-axis (digitized)",
                    loss_units="'validation loss' as plotted in Llama 3 Fig. 2 (held-out NLL; units/normalisation not "
                               "stated; not comparable to MassiveText nats)"),
    "marin": dict(N_convention="parameters of the Llama-2-architecture run (open-athena export)",
                  D_convention="training tokens from run configs; the nominal IsoFLOP budget (3 x forward FLOPs) differs "
                               "from 6ND by -7% to +35%, strongly correlated with N within budgets (review check)",
                  loss_units="Paloma macro-average loss, nats/token (Llama-2 tokenizer)"),
    "misfitting": dict(N_convention="parameters (Marghi/Li et al. sweep)",
                       D_convention="tokens at the interpolated checkpoint, D = C/(6N) (not end-of-schedule runs)",
                       loss_units="C4 validation loss, nats/token (FineWeb-trained)"),
}
EST_NAME = {"huber": "Huber-LSE delta=1e-3", "lad_log": "LAD (log loss)", "gauss_log": "Gaussian NLS (log loss)",
            "nls_lev": "NLS (levels)", "vpnls": "VPNLS (levels)", "norm": "KMW-normalized Huber"}
COLS = ["row_id", "dataset", "subset", "estimator", "source", "n", "E", "A", "B", "alpha", "beta", "se_scheme",
        "se_E", "se_A", "se_B", "se_alpha", "se_beta", "se2_scheme", "se2_E", "se2_alpha", "se2_beta", "a", "se_a",
        "gamma", "se_gamma", "sigma_star", "se_sigma_star", "G", "K", "Mstar_1e21", "Mstar_5.76e23", "Mstar_1e26",
        "w_chin70b", "N_convention", "D_convention", "loss_units", "cov_file", "draws_file", "citation", "notes"]


def _cov(draws_path, row_id):
    if draws_path is None or not os.path.exists(draws_path):
        return "", ""
    M = np.load(draws_path)[:, :5]
    M = M[np.all(np.isfinite(M), axis=1)]
    cov = np.cov(M.T)
    cp = os.path.join(PROC, f"cov_{row_id}.npy")
    np.save(cp, cov)
    rel = lambda p: os.path.relpath(p, ROOT)
    return rel(cp), rel(draws_path)


def _rng(v):
    lo, hi = f"{v.min():.2f}", f"{v.max():.2f}"
    return lo if lo == hi else f"{lo}-{hi}"


def _lit_row(row_id, m, subset, citation, notes):
    d = derived(m)
    return dict(row_id=row_id, dataset="chinchilla_massivetext", subset=subset, estimator="published point estimate",
                source="literature", n=np.nan, E=m.E, A=m.A, B=m.B, alpha=m.alpha, beta=m.beta, a=d["a"], gamma=d["gamma"],
                sigma_star=d["sigma_star"], G=m.G, K=m.K, Mstar_1e21=d["Mstar_1e+21"], **{"Mstar_5.76e23": d["Mstar_5.76e+23"]},
                Mstar_1e26=d["Mstar_1e+26"], w_chin70b=d["w_chin70b"], citation=citation, notes=notes, **CHIN_CONV)


def run(log=print):
    rows = []
    # ---------------- Chinchilla horse race
    hr = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_horse_race.csv"))
    for _, r in hr.iterrows():
        n = int(r.n)
        rid = f"chin_n{n}_{r.estimator}"
        dp = os.path.join(PROC, f"boot_chinchilla_n{n}_{r.estimator}_pairs.npy")
        cf, df_ = _cov(dp, rid)
        _cov(os.path.join(PROC, f"boot_chinchilla_n{n}_{r.estimator}_cluster.npy"), rid + "_cluster")
        ref = (n == 240 and r.estimator == "huber")
        rows.append(dict(row_id=rid, dataset="chinchilla_massivetext", subset=f"Epoch digitization, n={n}" +
                         (" (5 highest-loss runs dropped, Besiroglu sample)" if n == 240 else " (all points)"),
                         estimator=EST_NAME[r.estimator], source="estimated (this module)", n=n,
                         E=r.E, A=r.A, B=r.B, alpha=r.alpha, beta=r.beta, se_scheme="pairs bootstrap",
                         se_E=r.se_pairs_E, se_A=r.se_pairs_A, se_B=r.se_pairs_B, se_alpha=r.se_pairs_alpha,
                         se_beta=r.se_pairs_beta, se2_scheme="cluster bootstrap (9 IsoFLOP budgets)", se2_E=r.se_cluster_E,
                         se2_alpha=r.se_cluster_alpha, se2_beta=r.se_cluster_beta, a=r.a, se_a=r.se_pairs_a,
                         gamma=r.gamma, se_gamma=r.se_pairs_gamma, sigma_star=r.sigma_star, se_sigma_star=r.se_pairs_sigma_star,
                         G=np.exp(r.lnG), K=np.exp(r.lnK), Mstar_1e21=r["Mstar_1e+21"], **{"Mstar_5.76e23": r["Mstar_5.76e+23"]},
                         Mstar_1e26=r["Mstar_1e+26"], w_chin70b=r.w_chin70b, cov_file=cf, draws_file=df_,
                         citation="besiroglu2024chinchilla; hoffmann2022training",
                         notes=("REFERENCE technology (Hoffmann/Besiroglu estimator on the Besiroglu sample). " if ref else "") +
                               ("Cluster covariance in cov_" + rid + "_cluster.npy."), **CHIN_CONV))
    # ---------------- system estimators
    sy = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_system.csv"))
    slr = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_system_lr.csv"))
    for _, r in sy.iterrows():
        q = slr[(slr.dataset == r.dataset) & (slr.likelihood == r.likelihood)].iloc[0]
        n = 240 if r.dataset.endswith("240") else 245
        rid = f"chin_n{n}_system_{r.likelihood}"
        cf, df_ = _cov(os.path.join(PROC, f"boot_chinchilla_n{n}_system_{r.likelihood}_strat.npy"), rid)
        rows.append(dict(row_id=rid, dataset="chinchilla_massivetext", subset=f"Epoch digitization, n={n}",
                         estimator=f"System: loss eq. ({r.likelihood}) + IsoFLOP-argmin eq., cross-equation restricted",
                         source="estimated (this module)", n=n, E=r.E, A=r.A, B=r.B, alpha=r.alpha, beta=r.beta,
                         se_scheme="within-budget stratified bootstrap", se_E=r.se_E, se_A=r.se_A, se_B=r.se_B,
                         se_alpha=r.se_alpha, se_beta=r.se_beta, a=r.a, se_a=r.se_a, gamma=r.gamma, se_gamma=r.se_gamma,
                         sigma_star=r.sigma_star, se_sigma_star=r.se_sigma_star, G=np.exp(r.lnG), K=np.exp(r.lnK),
                         Mstar_1e21=r["Mstar_1e+21"], **{"Mstar_5.76e23": r["Mstar_5.76e+23"]}, Mstar_1e26=r["Mstar_1e+26"],
                         w_chin70b=r.w_chin70b, cov_file=cf, draws_file=df_,
                         citation="leonledesma2010identifying; hoffmann2022training",
                         notes=f"Nested LR test of the 2 cross-equation restrictions: {q.LR:.2f} (p = {q.LR_p:.3f}).", **CHIN_CONV))
    # ---------------- literature
    cfh = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_duality_classicalF.csv"))
    cfh = cfh[(cfh.dataset == "Chinchilla n=240") & cfh["null"].str.startswith("Hoffmann")].iloc[0]
    bch = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_duality_bootcal.csv"))
    bch = bch[(bch.dataset == "Chinchilla n=240") & (bch["null"] == "Hoffmann (fixed), design")]
    pb_s = bch[bch.bootstrap == "stratified pairs"].p_path_boot
    pb_w = bch[bch.bootstrap == "wild, fixed design"].p_path_boot
    rows.append(_lit_row("hoffmann_A3_tex", HOFF_TEX, "Hoffmann et al. (2022) Approach 3, TeX-source precision",
                         "hoffmann2022training; besiroglu2024chinchilla",
                         "Published values (E, alpha, beta at TeX precision via Besiroglu et al.). Reported CIs are not "
                         "usable (Besiroglu et al.: ~50x too narrow). Implies w(70B) = 0.71, outside the refit's pairs "
                         "95% interval but inside its 9-cluster interval. Its factor-demand (path) restrictions: classical "
                         f"fixed-design F test p = {cfh.p_F:.3f}; bootstrap-calibrated p = {_rng(pb_s)} "
                         f"(stratified), {_rng(pb_w)} (wild) (memo, H6)."))
    rows.append(_lit_row("hoffmann_A3_rounded", HOFF_ROUND, "Hoffmann et al. (2022) Approach 3, rounded (Eq. 10)",
                         "hoffmann2022training", "Rounded published values."))
    lit_b = _lit_row("besiroglu_published", BESI_PUB, "Besiroglu et al. (2024) replication, n=240", "besiroglu2024chinchilla",
                     "Published values; published SEs E 0.03, A 124.58, B 1293.23, alpha 0.02, beta 0.02. They are the "
                     "optimum of a Huber-likelihood with a free scale (effectively LAD), not of the Huber(1e-3) "
                     "objective; our LAD row reproduces them to 3-4 digits (memo).")
    lit_b.update(se_scheme="published (4000 pairs bootstrap draws)", se_E=0.03, se_A=124.58, se_B=1293.23, se_alpha=0.02,
                 se_beta=0.02)
    rows.append(lit_b)
    # ---------------- labs
    lt = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_labs_technology.csv"))
    lob = pd.read_csv(os.path.join(TABLES, "m1_chinchilla_labs_duality_objects.csv"))
    for _, r in lt.iterrows():
        key = "llama_3" if r.experiment == "llama_3" else ("marin" if r.experiment.startswith("marin") else "misfitting")
        conv = LAB_CONV[key]
        is_a3 = r.approach.startswith("A3")
        rid = f"{r.experiment}_{'A3_huber' if is_a3 else 'A2A1_kappa1'}"
        if is_a3:
            cf, df_ = _cov(os.path.join(PROC, f"boot_{r.experiment}_A3_huber_strat.npy"), rid)
            _cov(os.path.join(PROC, f"boot_{r.experiment}_A3_huber_cluster.npy"), rid + "_cluster")
            row = dict(estimator="Huber-LSE delta=1e-3 (primal, Approach 3)", E=r.E, A=r.A, B=r.B, alpha=r.alpha, beta=r.beta,
                       se_scheme="within-budget stratified bootstrap", se_E=r.se_strat_E, se_A=r.se_strat_A, se_B=r.se_strat_B,
                       se_alpha=r.se_strat_alpha, se_beta=r.se_strat_beta, se2_scheme="cluster bootstrap (budgets)",
                       se2_E=r.se_cluster_E, se2_alpha=r.se_cluster_alpha, se2_beta=r.se_cluster_beta,
                       a=r.a, se_a=r.se_strat_a, gamma=r.gamma, se_gamma=r.se_strat_gamma, sigma_star=r.sigma_star,
                       se_sigma_star=r.se_strat_sigma_star, G=r.G, K=r.K, w_chin70b=np.nan,
                       notes="Primal fit on IsoFLOP runs (transverse variation identifies alpha, beta without kappa = 1). "
                             "Cluster covariance in cov_" + rid + "_cluster.npy. w_chin70b not reported (different units/data).")
            cf_, df2 = cf, df_
        else:
            cf_, df2 = "", os.path.relpath(os.path.join(PROC, f"boot_{r.experiment}_A2A1_strat.npy"), ROOT)
            # review fix: the within-budget stratified bootstrap does not capture budget-level scatter of the argmins
            # around the path (Llama 3: SD 0.011 vs classical OLS s.e. 0.018), so se_a is the larger of the two.
            se_ols_a = float(lob[(lob.dataset == r.label) & (lob.object == "a")].A2_se_ols.iloc[0])
            row = dict(estimator="Approach 2 argmins (a, G) + Approach 1 frontier (E, K, gamma); alpha = gamma/a, "
                                 "beta = gamma/(1-a) under kappa = 1",
                       E=r.E, A=np.nan, B=np.nan, alpha=r.alpha, beta=r.beta, se_scheme="within-budget stratified bootstrap "
                       "(se_a: max of the robust IQR/1.349 bootstrap s.e. and the classical OLS s.e. of the path slope)",
                       se_E=r.se_strat_E, se_alpha=r.se_strat_alpha, se_beta=r.se_strat_beta, a=r.a,
                       se_a=max(float(r.rse_strat_a), se_ols_a),
                       gamma=r.gamma, se_gamma=r.se_strat_gamma, sigma_star=r.sigma_star, se_sigma_star=r.se_strat_sigma_star,
                       G=r.G, K=r.K, w_chin70b=np.nan,
                       notes="Only Approach-2/1 objects (a, G, E, K, gamma) are estimated here; alpha and beta follow from "
                             "the kappa = 1 functional-form restriction (model_spec Prop. 1), A and B are not reported. "
                             "Draws columns: a, gamma, E_A1, lnK, lnG. "
                             f"Path-slope s.e.: stratified bootstrap SD {float(r.se_strat_a):.4f} (robust {float(r.rse_strat_a):.4f}), "
                             f"classical OLS {se_ols_a:.4f}; the stratified draws understate slope uncertainty when the OLS "
                             "s.e. is larger, and their SD is inflated by failed parabolas when it is much larger. " +
                             ("Reproduces Meta's published law D* = 0.299 C^0.537 (memo)." if r.experiment == "llama_3" else ""))
        row.update(row_id=rid, dataset=r.experiment, subset=r.label + f" ({int(r.n_budgets)} budgets)", source="estimated (this module)",
                   n=int(r.n), Mstar_1e21=r["Mstar_1e+21"], **{"Mstar_5.76e23": r["Mstar_5.76e+23"]}, Mstar_1e26=r["Mstar_1e+26"],
                   cov_file=cf_, draws_file=df2,
                   citation={"llama_3": "grattafiori2024llama; czech2026problems", "marin": "czech2026problems; marin2026ladders",
                             "misfitting": "li2025misfitting; czech2026problems"}[key], **conv)
        rows.append(row)
    reg = pd.DataFrame(rows)
    for c in COLS:
        if c not in reg:
            reg[c] = np.nan
    reg = reg[COLS]
    reg.to_csv(os.path.join(TABLES, "technology_registry_m1.csv"), index=False)
    log(f"  registry: {len(reg)} rows")
    return reg
