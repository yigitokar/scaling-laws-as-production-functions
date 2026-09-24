"""technologies.py -- the technologies used to invert (N, D) into revealed inference demand.

Every technology is reduced to what the wedge needs: (alpha, beta, lnG), because under the Chinchilla form
  ln w = (alpha+beta) ln G - alpha ln N + beta ln D,        G = (alpha A/(beta B))^(1/(alpha+beta)),
and cost efficiency depends on (w, alpha, beta) only (m7 Lemma A5). A, B, E are never needed separately.
For each technology we carry the full-sample point estimate and, where the upstream module saved them, its bootstrap
draws (so the band reflects joint uncertainty in alpha, beta and G), its parameter-count convention and its design
support. Sources (all read-only):
  m1 registry + draws: Chinchilla refit on the Besiroglu sample (Huber, n = 240; pairs and 9-cluster bootstraps),
                       Meta Llama 3 IsoFLOPs: Approach-2/1 (Meta's own law; alpha = gamma/a, beta = gamma/(1-a) under
                       kappa = 1) and the primal (Approach 3) fitted to the same points.
  m2 registry + draws: Farseer (non-embedding N; incl.-embedding N variant; q-family inner aggregator),
                       Gadre et al. RefinedWeb / C4 / RedPajama, OLMo ladder (AI2's own sweep).
  literature points:   Hoffmann et al. (2022) A3 at TeX precision; Besiroglu et al. (2024) published values (sl.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import M1_PROC, M2_PROC, REG_M1, REG_M2, lnG_of, sl


@dataclass
class Tech:
    key: str
    label: str
    alpha: float
    beta: float
    lnG: float
    n_conv: str                       # 'total' | 'nonemb' | 'olmo' (MODEL_PARAMS: excludes the input embedding only)
    source: str
    draws: np.ndarray | None = None   # (B, 3): alpha, beta, lnG
    support: dict = field(default_factory=dict)   # M_min, M_max, N_min, N_max, C_max
    note: str = ""

    @property
    def a(self):
        return self.beta / (self.alpha + self.beta)

    @property
    def sigma_star(self):
        return 2 / (2 + self.alpha + self.beta)

    def mstar(self, C):
        return np.exp(-2 * self.lnG + (1 - 2 * self.a) * np.log(np.asarray(C, float) / 6))


def _from_theta_draws(arr, order):
    """Bootstrap draws -> (alpha, beta, lnG). order 'm1' = (lnA, lnB, lnE, alpha, beta); 'm2' = (E, A, B, alpha, beta)."""
    arr = np.asarray(arr, float)
    if order == "m1":
        A, B = np.exp(arr[:, 0]), np.exp(arr[:, 1])
    else:
        A, B = arr[:, 1], arr[:, 2]
    al, be = arr[:, 3], arr[:, 4]
    ok = np.isfinite(A) & np.isfinite(B) & (al > 0) & (be > 0)
    return np.c_[al, be, lnG_of(A, B, al, be)][ok]


def _support_from_m2(row):
    return dict(M_min=row["M_min"], M_max=row["M_max"], N_min=row["N_min"], N_max=row["N_max"],
                D_min=row["D_min"], D_max=row["D_max"])


def load_all():
    r1 = pd.read_csv(REG_M1).set_index("row_id")
    r2 = pd.read_csv(REG_M2)
    techs = {}
    # support of the Chinchilla (n = 240) design, from the m2 panel of the same data
    chin_sup = r2[(r2.dataset == "chinchilla") & (r2.subset == "all") & (r2.estimator == "huber")].iloc[0]
    chin_sup = dict(_support_from_m2(chin_sup), C_max=1.30e22)
    # --- Chinchilla refit (reference): m1 Huber on the Besiroglu sample, pairs bootstrap (B = 400)
    row = r1.loc["chin_n240_huber"]
    d_pairs = _from_theta_draws(np.load(os.path.join(M1_PROC, "boot_chinchilla_n240_huber_pairs.npy"))[:, :5], "m1")
    d_clu = _from_theta_draws(np.load(os.path.join(M1_PROC, "boot_chinchilla_n240_huber_cluster.npy"))[:, :5], "m1")
    base = dict(alpha=row.alpha, beta=row.beta, lnG=lnG_of(row.A, row.B, row.alpha, row.beta))
    techs["chin"] = Tech("chin", "Chinchilla refit (Besiroglu sample)", n_conv="total", draws=d_pairs, support=chin_sup,
                         source="m1 chin_n240_huber; pairs bootstrap B=400", **base)
    techs["chin_cl"] = Tech("chin_cl", "Chinchilla refit, 9-cluster bootstrap", n_conv="total", draws=d_clu,
                            support=chin_sup, source="m1 chin_n240_huber; cluster bootstrap B=400", **base)
    # --- estimator / sample sensitivity on the same Chinchilla data (not in the band): NLS in levels (n = 240) and
    #     Huber on all 245 points (the 5 gross outliers kept); m1 horse race
    for k, rid, bf, lab in [("chin_nls", "chin_n240_nls_lev", "boot_chinchilla_n240_nls_lev_pairs.npy", "Chinchilla, NLS in levels"),
                            ("chin245", "chin_n245_huber", "boot_chinchilla_n245_huber_pairs.npy", "Chinchilla, all 245 points")]:
        rr = r1.loc[rid]
        techs[k] = Tech(k, lab, rr.alpha, rr.beta, lnG_of(rr.A, rr.B, rr.alpha, rr.beta), n_conv="total",
                        draws=_from_theta_draws(np.load(os.path.join(M1_PROC, bf))[:, :5], "m1"), support=chin_sup,
                        source=f"m1 {rid}; pairs bootstrap")
    # --- literature points
    for k, lab, m in [("besi", "Besiroglu et al. (published)", sl.BESIROGLU), ("hoff", "Hoffmann et al. A3 (TeX)", sl.HOFFMANN)]:
        techs[k] = Tech(k, lab, m.alpha, m.beta, float(np.log(m.G)), n_conv="total", support=chin_sup,
                        source="literature point estimate (sl.py)")
    # --- Farseer (m2): non-embedding N (headline row), incl.-embedding N variant, q-family inner aggregator
    def m2row(ds, sub, est):
        return r2[(r2.dataset == ds) & (r2.subset == sub) & (r2.estimator == est)].iloc[0]
    fs = m2row("farseer", "all", "huber")
    techs["farseer"] = Tech("farseer", "Farseer (non-emb. N)", fs.alpha, fs.beta, lnG_of(fs.A, fs.B, fs.alpha, fs.beta),
                            n_conv="nonemb", draws=_from_theta_draws(np.load(os.path.join(M2_PROC, "boot", "farseer__all__huber.npy")), "m2"),
                            support=dict(_support_from_m2(fs), C_max=3.5e21), source="m2 farseer/all/huber; pairs B=400")
    fe = m2row("farseer", "N_incl_emb", "huber")
    techs["farseer_emb"] = Tech("farseer_emb", "Farseer (N incl. emb.)", fe.alpha, fe.beta, lnG_of(fe.A, fe.B, fe.alpha, fe.beta),
                                n_conv="total", draws=_from_theta_draws(np.load(os.path.join(M2_PROC, "boot", "farseer__N_incl_emb__huber.npy")), "m2"),
                                support=dict(_support_from_m2(fe), C_max=4.1e21), source="m2 farseer/N_incl_emb/huber; pairs B=200")
    fq = m2row("farseer", "all", "huber_q")
    qd = np.load(os.path.join(M2_PROC, "boot", "farseer__all__huber_q.npy"))
    techs["farseer_q"] = Tech("farseer_q", "Farseer, q-family (inner)", fq.alpha, fq.beta, lnG_of(fq.A, fq.B, fq.alpha, fq.beta),
                              n_conv="nonemb", draws=_from_theta_draws(qd[:, :5], "m2"), support=dict(_support_from_m2(fs), C_max=3.5e21),
                              source="m2 farseer/all/huber_q; the wedge uses the inner aggregator (q cancels)",
                              note=f"q = {fq.q:.3f}")
    # --- Gadre et al. over-training testbed (M up to 640)
    for sub, k in [("RefinedWeb", "gadre_rw"), ("C4", "gadre_c4"), ("RedPajama", "gadre_rp")]:
        g = m2row("gadre", sub, "huber")
        techs[k] = Tech(k, f"Gadre et al. {sub}", g.alpha, g.beta, lnG_of(g.A, g.B, g.alpha, g.beta), n_conv="total",
                        draws=_from_theta_draws(np.load(os.path.join(M2_PROC, "boot", f"gadre__{sub}__huber.npy")), "m2"),
                        support=dict(_support_from_m2(g), C_max=8.0e21), source=f"m2 gadre/{sub}/huber; B=400")
    # --- OLMo ladder (AI2's own sweep; N excludes the input embedding only)
    ol = m2row("olmo_ladder", "all", "huber")
    techs["olmo"] = Tech("olmo", "OLMo ladder (AI2)", ol.alpha, ol.beta, lnG_of(ol.A, ol.B, ol.alpha, ol.beta), n_conv="olmo",
                         draws=_from_theta_draws(np.load(os.path.join(M2_PROC, "boot", "olmo_ladder__all__huber.npy")), "m2"),
                         support=dict(_support_from_m2(ol), C_max=1.2e22), source="m2 olmo_ladder/all/huber; B=400")
    # --- Meta Llama 3 IsoFLOPs (m1): Approach 2/1 (Meta's own law) and the primal on the same points
    llama_sup = dict(M_min=0.44, M_max=643.0, N_min=5.9e7, N_max=1.7e10, C_max=1e22)
    ra = r1.loc["llama_3_A2A1_kappa1"]
    dA = np.load(os.path.join(M1_PROC, "boot_llama_3_A2A1_strat.npy"))   # columns: a, gamma, E_A1, lnK, lnG
    a_, g_ = dA[:, 0], dA[:, 1]
    okA = (a_ > 0) & (a_ < 1) & (g_ > 0)
    techs["meta_a2"] = Tech("meta_a2", "Meta Llama 3 law (A2/A1)", ra.alpha, ra.beta, float(np.log(ra.G)), n_conv="total",
                            draws=np.c_[g_ / a_, g_ / (1 - a_), dA[:, 4]][okA], support=llama_sup,
                            source="m1 llama_3_A2A1_kappa1; stratified B=200",
                            note="alpha = gamma/a, beta = gamma/(1-a) under kappa = 1; reproduces D* = 0.299 C^0.537")
    r3 = r1.loc["llama_3_A3_huber"]
    techs["meta_a3"] = Tech("meta_a3", "Meta Llama 3 IsoFLOPs, primal (A3)", r3.alpha, r3.beta,
                            lnG_of(r3.A, r3.B, r3.alpha, r3.beta), n_conv="total",
                            draws=_from_theta_draws(np.load(os.path.join(M1_PROC, "boot_llama_3_A3_huber_strat.npy"))[:, :5], "m1"),
                            support=llama_sup, source="m1 llama_3_A3_huber; stratified B=200")
    return techs


# technologies whose 95% bootstrap intervals (or points) form the partial-identification band for every model
BAND_SET = ["chin", "besi", "hoff", "farseer", "farseer_emb", "gadre_rw"]
# narrower 'core' band: technologies whose compute-optimal ratio at 1e21-1e24 FLOP is 17-25 (drops the two extremes,
# Hoffmann's A3, which fails its own IsoFLOP argmins marginally (m1), and Gadre RefinedWeb, whose M* is 1-3)
CORE_SET = ["chin", "besi", "farseer", "farseer_emb"]
# lab-own technologies added to the band for that lab's models
LAB_TECH = {"Meta": ["meta_a2", "meta_a3"], "AI2": ["olmo"]}
# homothetic CES grid (SYNTHESIS 2.3): M* in {16 (Porian), 20 (Chinchilla rule), 24.2 (m1 CES fit), 41 (Meta law at
# 3.8e25), 192 (MiniCPM)}, rho in {0.30, 0.3527 (Muennighoff alpha=beta), 0.40}
HOMO_MSTAR = [16.0, 20.0, 24.2, 41.0, 192.0]
HOMO_RHO = [0.30, 0.3527, 0.40]


def tech_table(techs):
    rows = []
    for k, t in techs.items():
        rows.append(dict(key=k, label=t.label, alpha=t.alpha, beta=t.beta, a=t.a, sigma_star=t.sigma_star, lnG=t.lnG,
                         Mstar_1e21=float(t.mstar(1e21)), Mstar_1e24=float(t.mstar(1e24)), Mstar_1e25=float(t.mstar(1e25)),
                         n_draws=0 if t.draws is None else len(t.draws), n_conv=t.n_conv, source=t.source, note=t.note,
                         **{f"support_{kk}": vv for kk, vv in t.support.items()}))
    return pd.DataFrame(rows)
