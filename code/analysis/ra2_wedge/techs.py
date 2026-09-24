"""techs.py -- every technology used to invert (N, D), under an EX-ANTE inclusion rule (R1 c4b; R2 Major 6a; R3 M4).

Inclusion rule (fixed before looking at wedges):
  IN  (i)   every technology estimated in m1/m2/m8 from FINAL-checkpoint losses of a designed sweep, primary estimator
            (Huber-LSE on log loss), each corpus and each parameter-count convention as its own row, under the
            Chinchilla form (kappa = 1) and, where estimated, the kappa-free form (inner exponents; the wedge depends
            on the inner aggregator only);
      (ii)  for IsoFLOP designs, the Approach-2 path (A2/A1, kappa = 1 curvature) and the model-free lab path with
            model-free curvature (this module);
      (iii) literature point estimates on the Chinchilla data (Hoffmann A3; Besiroglu) and published lab allocation
            laws: Llama 3 (Meta; = m1 A2/A1, which reproduces D* = 0.299 C^0.537), DeepSeek LLM (Bi et al. 2024,
            Eq. 4, verified from arXiv:2401.02954), MiniCPM (Hu et al. 2024, Fig. 10 'Average' box, verified from
            arXiv:2404.06395).
  OUT (a)   technologies identified from checkpoints taken before the learning-rate schedule completes
            (DataDecide: 'LR schedule not complete'; (Mis)Fitting: IsoFLOPs interpolated from per-checkpoint logs)
            -- the checkpoint artifact is a function of D/N and contaminates curvature and M*;
      (b)   estimator/sample variants on a design already represented (NLS in levels, all-245-points, cluster
            draws): reported as SENSITIVITY rows, not in the set.
Reference technology: Chinchilla kappa-free (inner exponents a1 = 0.4243, b1 = 0.4309, sigma*_kappa = 0.700; m1
spec_kappa, Huber, n = 240). Its uncertainty: design-conditional wild bootstrap (Rademacher = Feng, He and Hu 2011
weights for median-type regression) on log-loss residuals, B = 399; m2's pairs draws (B = 200) as a check.
Lab-own technologies (primary for that lab's models): Meta = Meta's A2 path + Meta's model-free IsoFLOP curvature
(ra1's reviewed estimate; this module's bias-corrected estimator is a sensitivity row); Marin = Marin's own A2 path +
ra1's model-free curvature (three corpora); AI2 = OLMo ladder (kappa-free
primary, kappa = 1 alternative); DeepSeek = DeepSeek LLM's published path (non-embedding FLOPs/token units) with the
reference curvature (DeepSeek did not publish curvature).
Every technology is reduced to (alpha, beta, ln G) [+ draws]: hybrids use alpha = S(1-a), beta = S a.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ra2common import (M1_PROC, M2_PROC, N_PROC, PROC, RA1_SIGMA, RA1_SUMMARY, ROOT, SEED, TABLES, ab_from_S_a, lnG_of,
                        log, sl)

sys.path.insert(0, os.path.join(ROOT, "code", "analysis", "m2_techpanel"))
import technologies as m3tech  # noqa: E402  (m3_wedge, read-only)
from technologies import Tech  # noqa: E402

REG_M1 = os.path.join(ROOT, "output", "tables", "technology_registry_m1.csv")
REG_M2 = os.path.join(ROOT, "output", "tables", "technology_registry_m2.csv")

REFERENCE = "chin_q"
# published lab laws (verified from the papers; see module docstring)
DEEPSEEK = dict(Mbase=0.1715, a=0.5243, Dbase=5.8316, b=0.4757, Cmax=3e20, lseq=4096,
                table4_a={"early data": 0.450, "current data": 0.524, "OpenWebText2": 0.578})
MINICPM = dict(alpha=0.29, beta=0.23, Mstar_1e21=191.87, Cmax=1.44e21)
LLAMA3_PUBLISHED = dict(Dcoef=0.299, Dexp=0.537)   # D*(C) = 0.299 C^0.537 (grattafiori2024llama, Fig. 3)


# ----------------------------------------------------------------------------- helpers
def _m2(r2, ds, sub, est):
    return r2[(r2.dataset == ds) & (r2.subset == sub) & (r2.estimator == est)].iloc[0]


def _draws_m2(fname, q=False):
    arr = np.load(os.path.join(M2_PROC, "boot", fname))
    A, B, al, be = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    ok = np.isfinite(A) & np.isfinite(B) & (al > 0) & (be > 0) & (A > 0) & (B > 0)
    return np.c_[al, be, lnG_of(A, B, al, be)][ok]


def _support(row, Cmax):
    return dict(M_min=row["M_min"], M_max=row["M_max"], N_min=row["N_min"], N_max=row["N_max"], C_max=Cmax)


# ----------------------------------------------------------------------------- kappa-free reference: wild bootstrap
def _chin_data():
    return sl.chinchilla_extraction(os.path.join(ROOT, "data", "raw", "epoch_chinchilla", "svg_extracted_data.csv"))


def _q_wild_worker(args):
    import m2_est as me
    seeds, N, D, yhat, res, th0 = args
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        v = rng.choice([-1.0, 1.0], size=len(yhat))
        L = np.exp(yhat + res * v)
        try:
            th, f = me.fit_q(N, D, L, "huber", init=th0)
            a, b, e, al, be, q = th
            out.append([al, be, (np.log(al) + a - np.log(be) - b) / (al + be), q])
        except Exception:
            out.append([np.nan] * 4)
    return out


def chin_q_wild(B=399):
    """kappa-free Chinchilla (m1 spec_kappa n240 Huber point; refit here to get the residuals) and a wild bootstrap."""
    import m2_est as me
    df = _chin_data()
    N, D, L = df.N.values, df.D.values, df.L.values
    r1 = pd.read_csv(REG_M1).set_index("row_id").loc["chin_n240_huber"]
    th_chin = np.array([np.log(r1.A), np.log(r1.B), np.log(r1.E), r1.alpha, r1.beta])
    th, f = me.fit_q(N, D, L, "huber", th_chin=th_chin)
    a, b, e, al, be, q = th
    # fitted log loss
    x, z = np.log(N), np.log(D)
    s = np.logaddexp(a - al * x, b - be * z)
    yhat = np.logaddexp(q * s, e)
    res = np.log(L) - yhat
    seeds = [SEED + 1000 + i for i in range(B)]
    chunks = [seeds[i::N_PROC] for i in range(N_PROC)]
    with ProcessPoolExecutor(N_PROC) as ex:
        parts = list(ex.map(_q_wild_worker, [(c, N, D, yhat, res, th) for c in chunks]))
    dr = np.array([r for p in parts for r in p])
    point = dict(alpha=al, beta=be, lnG=(np.log(al) + a - np.log(be) - b) / (al + be), q=q, E=np.exp(e), lnA=a, lnB=b)
    return point, dr


# ----------------------------------------------------------------------------- Chinchilla non-embedding (m8 refit)
OMEGA_PS, THETA_PS = 47491.0, 1.0 / 3.0   # Pearce & Song (2024) Eq. 11 map, as in m8_measurement/embed.py


def nonembed_from_total(NT, omega=OMEGA_PS, theta=THETA_PS):
    """Invert N_T = N_E + omega N_E^theta for N_E (copy of m8_measurement.embed.nonembed_from_total)."""
    from scipy.optimize import brentq
    out = []
    for n in np.asarray(NT, float):
        g = lambda y, n=n: np.exp(y) + omega * np.exp(y) ** theta - n
        out.append(np.exp(brentq(g, 0.0, np.log(n))))
    return np.array(out)


def _ne_worker(args):
    seeds, NE, D, yhat, res, th0 = args
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        v = rng.choice([-1.0, 1.0], size=len(yhat))
        L = np.exp(yhat + res * v)
        try:
            m = sl.fit_chinchilla(NE, D, L, delta=1e-3, init=th0)
            out.append([m.alpha, m.beta, np.log(m.G)])
        except Exception:
            out.append([np.nan] * 3)
    return out


def chin_nonemb_wild(B=399):
    """Chinchilla with non-embedding N (Pearce-Song map N_T = N_E + 47,491 N_E^(1/3), as m8), Huber, wild bootstrap."""
    df = _chin_data()
    NE = nonembed_from_total(df.N.values)
    m = sl.fit_chinchilla(NE, df.D.values, df.L.values, delta=1e-3, grid=sl.FAST_GRID)
    m = sl.fit_chinchilla(NE, df.D.values, df.L.values, delta=1e-3, init=m.theta)
    yhat = np.log(m.loss(NE, df.D.values))
    res = np.log(df.L.values) - yhat
    seeds = [SEED + 5000 + i for i in range(B)]
    chunks = [seeds[i::N_PROC] for i in range(N_PROC)]
    with ProcessPoolExecutor(N_PROC) as ex:
        parts = list(ex.map(_ne_worker, [(c, NE, df.D.values, yhat, res, m.theta) for c in chunks]))
    dr = np.array([r for p in parts for r in p])
    return m, dr, dict(M_min=float((df.D / NE).min()), M_max=float((df.D / NE).max()), N_min=float(NE.min()),
                       N_max=float(NE.max()), C_max=1.30e22)


# ----------------------------------------------------------------------------- ra1 model-free sigma (if present)
def ra1_sigma():
    """Return {design_key: dict(S2, sigma, se_sigma)} from module ra1's reviewed output
    (output/tables/ra1_modelfree_isoflop_summary.csv; primary = random-effects mean over valid budgets, 'sigma_re',
    with its standard error 'se_sigma_re'; ra1 memo H2) for Llama 3 and Marin's three corpora, else {}.
    Review note (ra2 reviewer, 2026-09-24): the builder's loader looked for 'ra1_modelfree_sigma.csv', which ra1 never
    wrote, so the lab-own technologies silently used ra2's own (bias-corrected) estimator. ra1's estimator is the
    reviewed one (finite-grid bias <= 0.006, Monte Carlo coverage 0.92-0.98 for the RE interval)."""
    out = {}
    for path in (RA1_SUMMARY, RA1_SIGMA):
        if not os.path.exists(path):
            continue
        try:
            t = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in t.columns}
        dcol = next((cols[c] for c in ("design", "dataset", "experiment", "sweep") if c in cols), None)
        scol = next((cols[c] for c in ("sigma_re", "sigma_star", "sigma", "sigma_star_mf", "sigma_mf") if c in cols), None)
        ecol = next((cols[c] for c in ("se_sigma_re", "se_sigma", "se") if c in cols), None)
        if dcol is None or scol is None:
            continue
        for _, r in t.iterrows():
            k = str(r[dcol]).lower()
            if "porian" in k or "farseer" in k:
                continue
            key = "meta" if "llama" in k else ("marin_comma" if "comma" in k else ("marin_dclm" if "dclm" in k else
                                                                                    ("marin_nemotron" if "nemotron" in k else None)))
            try:
                sig = float(r[scol])
                se = float(r[ecol]) if ecol else np.nan
            except (TypeError, ValueError):
                continue
            if key and np.isfinite(sig) and 0.3 < sig < 0.95 and key not in out:   # plausibility guard
                out[key] = dict(S2=1 / sig - 1, sigma=sig, se_sigma=se, source=os.path.basename(path))
        if out:
            break
    return out


# ----------------------------------------------------------------------------- build the registry
def build(mf_table, mf_draws, B_wild=399, mf_pb=None):
    """Returns (techs: dict key -> Tech, meta: DataFrame with set membership, family, form, lab)."""
    T = m3tech.load_all()   # m3: chin, chin_cl, chin_nls, chin245, besi, hoff, farseer, farseer_emb, farseer_q,
    #                              gadre_rw/c4/rp, olmo, meta_a2, meta_a3
    r1 = pd.read_csv(REG_M1).set_index("row_id")
    r2 = pd.read_csv(REG_M2)
    chin_sup = T["chin"].support
    # --- kappa-free Chinchilla: reference
    log("techs: kappa-free Chinchilla wild bootstrap")
    pq, drq = chin_q_wild(B_wild)
    okq = np.isfinite(drq).all(1)
    T["chin_q"] = Tech("chin_q", "Chinchilla, kappa free (reference)", pq["alpha"], pq["beta"], pq["lnG"], n_conv="total",
                       draws=drq[okq, :3], support=chin_sup,
                       source=f"m1 spec_kappa n240 Huber (refit); wild bootstrap B={B_wild} (Rademacher)",
                       note=f"kappa = {pq['q']:.3f}")
    T["chin_q_pairs"] = Tech("chin_q_pairs", "Chinchilla, kappa free (m2 pairs draws)", pq["alpha"], pq["beta"], pq["lnG"],
                             n_conv="total", draws=_draws_m2("chinchilla__all__huber_q.npy"), support=chin_sup,
                             source="m2 chinchilla/all/huber_q pairs draws B=200")
    # --- other kappa-free rows (m2)
    for sub, k in [("RefinedWeb", "gadre_rw_q"), ("C4", "gadre_c4_q"), ("RedPajama", "gadre_rp_q")]:
        g = _m2(r2, "gadre", sub, "huber_q")
        g1 = _m2(r2, "gadre", sub, "huber")
        T[k] = Tech(k, f"Gadre et al. {sub}, kappa free", g.alpha, g.beta, lnG_of(g.A, g.B, g.alpha, g.beta), n_conv="total",
                    draws=_draws_m2(f"gadre__{sub}__huber_q.npy"), support=_support(g1, 8.0e21), source=f"m2 gadre/{sub}/huber_q")
    o = _m2(r2, "olmo_ladder", "all", "huber_q")
    o1 = _m2(r2, "olmo_ladder", "all", "huber")
    T["olmo_q"] = Tech("olmo_q", "OLMo ladder (AI2), kappa free", o.alpha, o.beta, lnG_of(o.A, o.B, o.alpha, o.beta), n_conv="olmo",
                       draws=_draws_m2("olmo_ladder__all__huber_q.npy"), support=_support(o1, 1.2e22), source="m2 olmo_ladder/all/huber_q")
    mu = _m2(r2, "datablations", "single_epoch", "huber")
    T["muen"] = Tech("muen", "Muennighoff et al., single epoch", mu.alpha, mu.beta, lnG_of(mu.A, mu.B, mu.alpha, mu.beta),
                     n_conv="total", draws=_draws_m2("datablations__single_epoch__huber.npy"),
                     support=_support(mu, float(6 * mu.N_max * mu.D_max)), source="m2 datablations/single_epoch/huber",
                     note="N convention unverified (m2)")
    mq = _m2(r2, "datablations", "single_epoch", "huber_q")
    T["muen_q"] = Tech("muen_q", "Muennighoff et al., kappa free", mq.alpha, mq.beta, lnG_of(mq.A, mq.B, mq.alpha, mq.beta),
                       n_conv="total", draws=_draws_m2("datablations__single_epoch__huber_q.npy"),
                       support=_support(mu, float(6 * mu.N_max * mu.D_max)), source="m2 datablations/single_epoch/huber_q")
    # --- Chinchilla non-embedding (m8 refit; wild bootstrap here)
    log("techs: Chinchilla non-embedding wild bootstrap")
    mne, drne, sup_ne = chin_nonemb_wild(B_wild)
    okn = np.isfinite(drne).all(1)
    T["chin_ne"] = Tech("chin_ne", "Chinchilla, non-embedding N (m8)", mne.alpha, mne.beta, float(np.log(mne.G)), n_conv="nonemb",
                        draws=drne[okn], support=sup_ne, source=f"m8 refit (Pearce-Song map); wild bootstrap B={B_wild}")
    # --- Marin (m1): primal A3 and A2/A1 per corpus
    iso = pd.read_csv(os.path.join(ROOT, "data", "raw", "isoflop_experiments", "isoflop_experiments.csv"))
    for corp in ["comma", "dclm", "nemotron"]:
        g_ = iso[iso["experiment"] == f"marin_202603__{corp}__llama_2"]
        msup_c = dict(M_min=float((g_.tokens / g_.params).min()), M_max=float((g_.tokens / g_.params).max()), C_max=3e20)
        rid3 = f"marin_202603__{corp}__llama_2_A3_huber"
        rr = r1.loc[rid3]
        d3 = np.load(os.path.join(M1_PROC, f"boot_{rid3}_strat.npy"))
        A, Bq, al, be = np.exp(d3[:, 0]), np.exp(d3[:, 1]), d3[:, 3], d3[:, 4]
        ok = (al > 0) & (be > 0)
        msup = msup_c
        T[f"marin_{corp}_a3"] = Tech(f"marin_{corp}_a3", f"Marin {corp}, primal", rr.alpha, rr.beta, lnG_of(rr.A, rr.B, rr.alpha, rr.beta),
                                     n_conv="total", draws=np.c_[al, be, lnG_of(A, Bq, al, be)][ok], support=msup, source=f"m1 {rid3}")
        rid2 = f"marin_202603__{corp}__llama_2_A2A1_kappa1"
        ra = r1.loc[rid2]
        dA = np.load(os.path.join(M1_PROC, f"boot_marin_202603__{corp}__llama_2_A2A1_strat.npy"))
        a_, g_ = dA[:, 0], dA[:, 1]
        okA = (a_ > 0) & (a_ < 1) & (g_ > 0)
        T[f"marin_{corp}_a2"] = Tech(f"marin_{corp}_a2", f"Marin {corp}, A2/A1 law", ra.alpha, ra.beta, float(np.log(ra.G)),
                                     n_conv="total", draws=np.c_[g_ / a_, g_ / (1 - a_), dA[:, 4]][okA], support=msup,
                                     source=f"m1 {rid2}")
    # --- model-free lab technologies: own A2 path + own IsoFLOP curvature (bias-corrected; ra1 value if available)
    ra1 = ra1_sigma()
    mfsrc = {}
    for key, lab in [("meta", "Meta Llama 3"), ("marin_comma", "Marin comma"), ("marin_dclm", "Marin dclm"),
                     ("marin_nemotron", "Marin nemotron")]:
        row = mf_table[(mf_table.design == key) & (mf_table.deg == 2)].iloc[0]
        dr = mf_draws[key]
        okd = np.isfinite(dr).all(1)
        a_pt, lnG_pt = row["a"], row["lnG"]
        sup_mf = dict(T["meta_a2"].support) if key == "meta" else dict(T[f"{key}_a2"].support)
        # (1) this module's own estimator, bias-corrected by its parametric simulation (builder's primary; now a
        #     sensitivity row '<key>_mfbc')
        fac = row["mc_factor"]
        S2_bc = row["S2_bc"]
        al_bc, be_bc = ab_from_S_a(2 * S2_bc, a_pt)
        Sd_bc = 2 * dr[okd, 0] * fac
        ald_bc, bed_bc = ab_from_S_a(Sd_bc, dr[okd, 1])
        # (2) primary: ra1's reviewed model-free sigma* (RE mean over valid budgets) with ra1's standard error; the
        #     joint draws keep this module's path draws (a, ln G) and their correlation with the curvature draws, with
        #     the curvature draws re-centred on ra1's S/2 and rescaled to ra1's s.e. (delta method: sd(S/2) = se/sigma^2)
        if key in ra1:
            S2 = ra1[key]["S2"]
            sd_target = ra1[key]["se_sigma"] / ra1[key]["sigma"] ** 2
            s2d = dr[okd, 0]
            k_sd = sd_target / np.std(s2d, ddof=1) if np.isfinite(sd_target) else 1.0
            Sd = 2 * (S2 + (s2d - s2d.mean()) * k_sd)
            src = (f"ra1 model-free sigma* = {ra1[key]['sigma']:.3f} (s.e. {ra1[key]['se_sigma']:.3f}; "
                   f"{ra1[key]['source']}, RE over valid budgets)")
        else:
            S2, Sd = S2_bc, Sd_bc
            src = "own model-free estimate (bias-corrected; ra1 output not found)"
        mfsrc[key] = src
        al, be = ab_from_S_a(2 * S2, a_pt)
        ald, bed = ab_from_S_a(Sd, dr[okd, 1])
        k = f"{key}_mf"
        T[k] = Tech(k, f"{lab}: own path + model-free curvature", al, be, lnG_pt, n_conv="total",
                    draws=np.c_[ald, bed, dr[okd, 2]],
                    support=sup_mf,
                    source=f"A2 path (own quadratic argmins; wild bootstrap B={int(row['B'])}) + {src}",
                    note=f"S/2 = {S2:.3f}; sigma* = {1/(1+S2):.3f} (ra2 raw {1/(1+row['S2']):.3f}, ra2 bias-corr. "
                         f"{1/(1+S2_bc):.3f})")
        T[f"{key}_mfbc"] = Tech(f"{key}_mfbc", f"{lab}: model-free curvature, ra2 estimator bias-corrected", al_bc, be_bc,
                                lnG_pt, n_conv="total", draws=np.c_[ald_bc, bed_bc, dr[okd, 2]], support=sup_mf,
                                source="ra2 per-budget quadratic estimator x MC bias factor")
        # raw (not bias-corrected) variant: sensitivity
        al2, be2 = ab_from_S_a(2 * row["S2"], a_pt)
        ald2, bed2 = ab_from_S_a(2 * dr[okd, 0], dr[okd, 1])
        T[f"{key}_mfraw"] = Tech(f"{key}_mfraw", f"{lab}: model-free curvature, raw", al2, be2, lnG_pt, n_conv="total",
                                 draws=np.c_[ald2, bed2, dr[okd, 2]], support=sup_mf, source="raw model-free (ra2 estimator)")
    # --- review addition: Meta's path on the 8 IsoFLOP budgets whose minimum ra1 finds bracketed (ra1 drops 3e21 and
    #     1e22 as unbracketed); ra1 curvature; sensitivity row only (the 10-budget path reproduces Meta's published law)
    if mf_pb is not None and "meta_mf" in T:
        g = mf_pb[(mf_pb["design"] == "meta") & (mf_pb["budget"] <= 1.01e21)]
        a8, lnG8 = np.polyfit(np.log(g["budget"].values / 6.0), g["lnNstar"].values, 1)
        S_m = T["meta_mf"].alpha + T["meta_mf"].beta
        al8, be8 = ab_from_S_a(S_m, a8)
        T["meta_mf8"] = Tech("meta_mf8", "Meta Llama 3: path on ra1's 8 bracketed budgets + ra1 curvature", al8, be8,
                             float(lnG8), n_conv="total", support=dict(T["meta_a2"].support),
                             source="A2 path on budgets <= 1e21 (ra1-valid) + ra1 sigma*; point estimate",
                             note=f"a = {a8:.4f}")
    # --- DeepSeek LLM published law (units: non-embedding FLOPs/token M; N_ds = M/6) + reference curvature
    a_ds = DEEPSEEK["a"]
    lnG_ds = np.log(DEEPSEEK["Mbase"] / 6) + a_ds * np.log(6)
    Sref = T[REFERENCE].alpha + T[REFERENCE].beta
    al, be = ab_from_S_a(Sref, a_ds)
    Sd = T[REFERENCE].draws[:, 0] + T[REFERENCE].draws[:, 1]
    ald, bed = ab_from_S_a(Sd, a_ds)
    T["deepseek"] = Tech("deepseek", "DeepSeek LLM law + reference curvature", al, be, lnG_ds, n_conv="ds",
                         draws=np.c_[ald, bed, np.full(len(Sd), lnG_ds)], support=dict(C_max=DEEPSEEK["Cmax"]),
                         source="Bi et al. 2024 Eq. 4 (M_opt = 0.1715 C^0.5243, D_opt = 5.8316 C^0.4757; C = M D); "
                                "curvature: kappa-free Chinchilla draws", note="path draws not published")
    # --- MiniCPM published law: alpha = 0.29, beta = 0.23 (average of 5 corpora), M*(1e21) = 191.87, non-embedding N
    al, be = MINICPM["alpha"], MINICPM["beta"]
    a_m = be / (al + be)
    lnG_m = -(np.log(MINICPM["Mstar_1e21"]) - (1 - 2 * a_m) * np.log(1e21 / 6)) / 2
    T["minicpm"] = Tech("minicpm", "MiniCPM law (published)", al, be, lnG_m, n_conv="nonemb",
                        support=dict(M_min=10.0, M_max=60.0, C_max=MINICPM["Cmax"]),
                        source="Hu et al. 2024 Fig. 10 'Average': 7.15e-2 N^-0.29 + 3.00e-1 D^-0.23 + 0.31; D/N=191.87 at 1e21")
    # --- metadata: set membership and grouping
    meta = []
    EXANTE = ["chin_q", "chin", "besi", "hoff", "chin_ne", "farseer", "farseer_emb", "farseer_q", "gadre_rw", "gadre_c4",
              "gadre_rp", "gadre_rw_q", "gadre_c4_q", "gadre_rp_q", "olmo", "olmo_q", "muen", "muen_q",
              "meta_a2", "meta_a3", "meta_mf", "marin_comma_a3", "marin_dclm_a3", "marin_nemotron_a3",
              "marin_comma_a2", "marin_dclm_a2", "marin_nemotron_a2", "marin_comma_mf", "marin_dclm_mf",
              "marin_nemotron_mf", "deepseek", "minicpm"]
    SENS = ["chin_cl", "chin_q_pairs", "chin_nls", "chin245", "meta_mfraw", "marin_comma_mfraw", "marin_dclm_mfraw",
            "marin_nemotron_mfraw", "meta_mfbc", "marin_comma_mfbc", "marin_dclm_mfbc", "marin_nemotron_mfbc", "meta_mf8"]
    source_group = {"chin": "Chinchilla", "besi": "Chinchilla", "hoff": "Chinchilla", "farseer": "Farseer",
                    "gadre": "Gadre et al.", "olmo": "OLMo ladder", "muen": "Muennighoff et al.", "meta": "Meta Llama 3",
                    "marin": "Marin", "deepseek": "DeepSeek LLM", "minicpm": "MiniCPM"}
    for k, t in T.items():
        grp = next((v for p, v in source_group.items() if k.startswith(p)), "other")
        form = ("kappa free" if (k.endswith("_q") or k.startswith("chin_q")) else
                "model-free curvature" if "_mf" in k else "published law" if k in ("deepseek", "minicpm") else
                "A2/A1 path (kappa=1)" if k.endswith("_a2") else "kappa = 1")
        meta.append(dict(key=k, label=t.label, group=grp, form=form, n_conv=t.n_conv,
                         in_set=k in EXANTE, sensitivity=k in SENS,
                         excluded=k not in EXANTE and k not in SENS, alpha=t.alpha, beta=t.beta, lnG=t.lnG,
                         a=t.a, S2=(t.alpha + t.beta) / 2, sigma_star=t.sigma_star,
                         n_draws=0 if t.draws is None else len(t.draws), source=t.source, note=t.note))
    M = pd.DataFrame(meta)
    M.attrs["mf_source"] = mfsrc
    M.attrs["chin_q_point"] = pq
    return T, M


LAB_OWN = {   # lab -> (primary technology, alternatives)
    "Meta": ("meta_mf", ["meta_a2", "meta_a3", "meta_mfbc", "meta_mfraw", "meta_mf8"]),
    "AI2": ("olmo_q", ["olmo"]),
    "DeepSeek": ("deepseek", []),
    "Marin": ("marin_nemotron_mf", ["marin_comma_mf", "marin_dclm_mf", "marin_comma_a2", "marin_dclm_a2",
                                    "marin_nemotron_a2", "marin_nemotron_mfbc"]),
}
# Vintage of the lab-own technology (review addition; R1 minor 37): a model released on or after this date was planned
# when the lab's own law existed (contemporaneous); earlier releases are evaluated under a later (ex-post) lab law.
# Meta: the Llama 3 IsoFLOP law was fitted to plan Llama 3 (released 2024-04-18); Llama 1/2 predate it.
# AI2: the OLMo ladder (bhagia2024establishing, Dec 2024) was built with the OLMo 2 recipe (OLMo 2, Nov 2024);
#      OLMo 1/1.7 predate it. DeepSeek: the law is in the DeepSeek LLM report (Jan 2024). Marin: ladder 2026-03.
LAB_LAW_FROM = {"Meta": "2024-04-01", "AI2": "2024-11-01", "DeepSeek": "2024-01-01", "Marin": "2026-03-01"}
EXCLUDED_TECH = [
    ("DataDecide (25 recipes)", "identified from intermediate checkpoints before the cosine schedule completes "
                                "(m2: 'LR schedule not complete'); artifact is a function of D/N"),
    ("(Mis)Fitting FineWeb/C4 (m1)", "IsoFLOPs interpolated from per-checkpoint training logs (open-athena README)"),
]
