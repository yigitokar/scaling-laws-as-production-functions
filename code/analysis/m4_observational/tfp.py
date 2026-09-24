"""Lab/family productivity (TFP) dispersion in compute-equivalent units.

Family effects omega_f from y = a + theta_N n + theta_D d + omega_f + e (family FE, main sample), model-level
residuals within developer from y = a + theta_C c + developer FE + year FE + e (Mertens et al. 2026 design).
Compute-equivalent 90/10 ratio: exp((omega_90 - omega_10) / theta_C), i.e. how much more training compute the
p10 family needs to match the p90 family's output (TFP^(1/RTS), SYNTHESIS Sec. 4.E).
theta_C is either (a) the same regression's compute elasticity (as in Mertens et al.) or (b) the design-matched
experimental benchmark (lalonde.design_matched), which is free of transmission bias by construction.
Reducible-loss units: a compute factor R corresponds to R^gamma in reducible loss at fixed compute under a
Chinchilla technology (gamma = 0.155 Hoffmann, 0.178 Besiroglu; ASSUMED crosswalk, illustrative only), the unit in
which Syverson's (2004, 2011) manufacturing 90-10 TFP ratio of 1.92 is expressed (with RTS ~ 1).
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

import observational as ob
from common import PREFIX, RAW, TABDIR, fmt, logit_adj, write_tex_table

GAMMA = {"Hoffmann": 0.1548, "Besiroglu": 0.1783}   # SYNTHESIS ledger Sec. 2.1 (TeX-precision Hoffmann; Besiroglu refit)
SYVERSON = 1.92
CODE_FAMS = {"CodeLlama", "StarCoder", "StarCoder2", "DeepSeek-Coder"}
DISTILL_SYN = {"google/gemma-2-2b", "google/gemma-2-9b"}
SYN_FAMS = {"Phi", "SmolLM"}


def add_pc1(df, cols=("mmlu", "arc_c", "hellaswag", "winogrande", "gsm8k", "truthfulqa")):
    """ObsScaling-style first principal component of standardized benchmark logits (complete cases)."""
    Y = pd.DataFrame({c: logit_adj(df[c], c) for c in cols}, index=df.index)
    ok = Y.notna().all(axis=1) & df.main
    Z = (Y[ok] - Y[ok].mean()) / Y[ok].std()
    u, s, vt = np.linalg.svd(Z.values, full_matrices=False)
    w = vt[0] * np.sign(vt[0].sum())
    df["y_pc1"] = np.nan
    df.loc[ok, "y_pc1"] = Z.values @ w
    return df, float(s[0] ** 2 / (s ** 2).sum()), dict(zip(cols, w))


def _p9010(x, w=None):
    x = np.asarray(x, float)
    return np.nanpercentile(x, 90) - np.nanpercentile(x, 10)


def dispersion(d, out, theta_bench=None):
    """Returns dict of dispersion statistics for one sample/output."""
    # (1) family effects with (n, d)
    X = pd.concat([d[["n", "d"]], ob._dummies(d, "family")], axis=1)
    r = sm.OLS(d[out], sm.add_constant(X)).fit()
    fam = d.assign(res=r.resid + 0.0)
    # family effect = mean of (y - theta_N n - theta_D d) within family (equivalent to the FE incl. baseline)
    fam["w"] = d[out] - r.params["n"] * d.n - r.params["d"] * d.d
    om = fam.groupby("family").w.mean()
    th_fe = 0.5 * (r.params["n"] + r.params["d"])
    # (2) Mertens design: compute only, developer FE + year FE
    X2 = pd.concat([d[["c"]], ob._dummies(d, "developer"), ob._dummies(d, "year")], axis=1)
    r2 = sm.OLS(d[out], sm.add_constant(X2)).fit()
    # [review fix] residuals of single-model developers are identically 0 under developer FE; exclude them from the
    # within-developer dispersion (as in the ECI version below)
    multi = (d.developer.map(d.developer.value_counts()) >= 2).values
    res = r2.resid[multi]
    th_c = float(r2.params["c"])
    dev = d.assign(w=d[out] - th_c * d.c).groupby("developer").w.mean()
    out_d = dict(n_models=len(d), n_families=len(om), n_dev=len(dev), n_res=int(multi.sum()), theta_ray_fe=th_fe,
                 theta_c_mertens=th_c, fam_p9010=_p9010(om), fam_sd=float(om.std()), res_p9010=_p9010(res),
                 res_sd=float(res.std()), dev_p9010=_p9010(dev), dev_range=float(dev.max() - dev.min()))
    for lab, th in [("own", None), ("bench", theta_bench)]:
        if lab == "bench" and th is None:
            continue
        tf = th_fe if th is None else th
        tc = th_c if th is None else th
        out_d[f"fam_ce_{lab}"] = float(np.exp(out_d["fam_p9010"] / tf))
        out_d[f"res_ce_{lab}"] = float(np.exp(out_d["res_p9010"] / tc))
        out_d[f"dev_ce_{lab}"] = float(np.exp(out_d["dev_p9010"] / tc))
    if theta_bench is not None:
        # [review addition] internally consistent 'experimental technology' family effects: net inputs out with the
        # SAME compute elasticity used for the conversion (omega_f = family mean of y - theta_bench ln C), instead of
        # the within-family (theta_N, theta_D) split, which Table 6 finds biased
        omc = d.assign(w=d[out] - theta_bench * d.c).groupby("family").w.mean()
        out_d["famc_p9010"] = _p9010(omc)
        out_d["famc_ce_bench"] = float(np.exp(out_d["famc_p9010"] / theta_bench))
    return out_d, om, dev, r, r2


def boot_dispersion(d, out, theta_bench, B=400, seed=0):
    rng = np.random.default_rng(seed)
    devs = d.developer.unique()
    draws = []
    for _ in range(B):
        pick = rng.choice(devs, len(devs), replace=True)
        parts = []
        for j, g in enumerate(pick):
            x = d[d.developer == g].copy()
            x["developer"] = f"{g}#{j}"
            x["family"] = x.family + f"#{j}"
            parts.append(x)
        bd = pd.concat(parts, ignore_index=True)
        try:
            s, *_ = dispersion(bd, out, theta_bench)
            draws.append(s)
        except Exception:
            continue
    return pd.DataFrame(draws)


def run_tfp(df, bench_theta, B=400, seed=0):
    """bench_theta: dict output -> experimentally matched theta_C (OLS benchmark from lalonde.design_matched)."""
    df, share, w = add_pc1(df)
    samples = {
        "All dense base models": df.main,
        "Drop distilled (Gemma-2 2B/9B) and synthetic-data (Phi, SmolLM)": df.main & ~df.model.isin(DISTILL_SYN) & ~df.family.isin(SYN_FAMS),
        "Also drop code-specialised families": df.main & ~df.model.isin(DISTILL_SYN) & ~df.family.isin(SYN_FAMS | CODE_FAMS),
    }
    rows, fam_effects, dev_effects = [], {}, {}
    for out in ["y_hellaswag", "y_core", "y_pc1"]:
        for sname, m in samples.items():
            d = df[m].dropna(subset=[out]).copy()
            s, om, dev, r, r2 = dispersion(d, out, bench_theta.get(out))
            bs = boot_dispersion(d, out, bench_theta.get(out), B=B, seed=seed)
            for k in ["fam_ce_own", "res_ce_own", "dev_ce_own", "fam_ce_bench", "res_ce_bench", "dev_ce_bench",
                      "famc_ce_bench", "fam_p9010", "res_p9010", "theta_ray_fe", "theta_c_mertens"]:
                if k in bs:
                    s[k + "_lo"], s[k + "_hi"] = np.nanpercentile(bs[k], [5, 95])  # 90% interval (skewed ratios)
            s.update(output=out, sample=sname, theta_bench=bench_theta.get(out, np.nan), B_ok=len(bs))
            rows.append(s)
            if sname == "All dense base models":
                fam_effects[out] = om
                dev_effects[out] = dev
    T = pd.DataFrame(rows)
    for g, gam in GAMMA.items():
        for k in ["fam_ce_bench", "res_ce_bench", "fam_ce_own", "res_ce_own", "famc_ce_bench"]:
            if k in T:
                T[f"{k}_loss_{g}"] = T[k] ** gam
    T.to_csv(os.path.join(TABDIR, f"{PREFIX}_tfp_dispersion.csv"), index=False)
    return T, fam_effects, dev_effects, df, share, w


def eci_dispersion(B=400, seed=0):
    """Frontier sample: Epoch Capabilities Index vs log training compute (Epoch), developer + year FE."""
    z = zipfile.ZipFile(os.path.join(RAW, "epoch_bench", "benchmark_data.zip"))
    eci = pd.read_csv(io.BytesIO(z.read("epoch_capabilities_index/eci_scores.csv")))
    e = pd.read_csv(os.path.join(RAW, "epoch_models", "all_ai_models.csv"), low_memory=False).drop_duplicates("Model")
    m = eci.merge(e[["Model", "Organization", "Training compute (FLOP)", "Parameters", "Training dataset size (total)",
                     "Confidence"]], on="Model", how="left", suffixes=("", "_e"))
    m = m.dropna(subset=["Training compute (FLOP)"]).copy()
    org = m["Organization_e"].fillna(m["Organization"]).astype(str).str.split(",").str[0].str.strip()
    org = org.replace({"Microsoft Research": "Microsoft", "Google DeepMind": "Google", "Google": "Google",
                       "Hugging Face": "BigCode", "Meta AI": "Meta"})
    m["developer"], m["family"] = org, org
    m["c"] = np.log(m["Training compute (FLOP)"])
    m["year"] = pd.to_datetime(m.date).dt.year
    m["y"] = m.eci
    rows = []
    for conf_lab, mask in [("all", np.ones(len(m), bool)), ("Confident only", (m.Confidence == "Confident").values)]:
        d = m[mask].copy()
        X = pd.concat([d[["c"]], ob._dummies(d, "developer"), ob._dummies(d, "year")], axis=1)
        r = sm.OLS(d.y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.developer)[0]})
        th = float(r.params["c"])
        dev = d.assign(w=d.y - th * d.c).groupby("developer").w.mean()
        multi = d.developer.map(d.developer.value_counts()) >= 2
        res = r.resid[multi.values]
        s = dict(sample=conf_lab, n=len(d), n_dev=d.developer.nunique(), theta_c=th, theta_c_se=float(r.bse["c"]),
                 res_p9010=_p9010(res), res_ce=float(np.exp(_p9010(res) / th)), dev_p9010=_p9010(dev),
                 dev_ce=float(np.exp(_p9010(dev) / th)), pooled_slope=float(sm.OLS(d.y, sm.add_constant(d[["c"]])).fit().params["c"]))
        rng = np.random.default_rng(seed)
        bs = []
        devs = d.developer.unique()
        for _ in range(B):
            pick = rng.choice(devs, len(devs), replace=True)
            bd = pd.concat([d[d.developer == g].assign(developer=f"{g}#{j}") for j, g in enumerate(pick)], ignore_index=True)
            try:
                X = pd.concat([bd[["c"]], ob._dummies(bd, "developer"), ob._dummies(bd, "year")], axis=1)
                rb = sm.OLS(bd.y, sm.add_constant(X)).fit()
                tb = float(rb.params["c"])
                mb = bd.developer.map(bd.developer.value_counts()) >= 2
                bs.append(dict(res_ce=np.exp(_p9010(rb.resid[mb.values]) / tb), theta_c=tb))
            except Exception:
                continue
        bs = pd.DataFrame(bs)
        s["res_ce_lo"], s["res_ce_hi"] = np.nanpercentile(bs.res_ce, [5, 95])
        s["theta_c_lo"], s["theta_c_hi"] = np.nanpercentile(bs.theta_c, [5, 95])
        rows.append(s)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(TABDIR, f"{PREFIX}_eci_dispersion.csv"), index=False)
    return E, m


def _ci(lo, hi, nd=1):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{lo:.{nd}f}, " + (f"{hi:.{nd}f}]" if hi < 1000 else ">1000]")


def make_tfp_table(T, E, fname="tfp_table", B=None):
    rows = []
    lab = {"y_hellaswag": "HellaSwag logit", "y_core": "Composite (HS, ARC-C, WG)", "y_pc1": "PC1 of 6 benchmarks"}
    for out in ["y_hellaswag", "y_core", "y_pc1"]:
        rows.append([f"#{lab[out]}"])
        for _, r in T[T.output == out].iterrows():
            nm = {"All dense base models": "All dense base models",
                  "Drop distilled (Gemma-2 2B/9B) and synthetic-data (Phi, SmolLM)": "Drop distilled/synthetic",
                  "Also drop code-specialised families": "Also drop code families"}[r["sample"]]
            bench = f"{r.fam_ce_bench:.1f}" if np.isfinite(r.get("fam_ce_bench", np.nan)) else ""
            benchci = _ci(r.get("fam_ce_bench_lo", np.nan), r.get("fam_ce_bench_hi", np.nan))
            resb = f"{r.res_ce_bench:.1f}" if np.isfinite(r.get("res_ce_bench", np.nan)) else ""
            resbci = _ci(r.get("res_ce_bench_lo", np.nan), r.get("res_ce_bench_hi", np.nan))
            famc = f"{r.famc_ce_bench:.1f}" if np.isfinite(r.get("famc_ce_bench", np.nan)) else ""
            famcci = _ci(r.get("famc_ce_bench_lo", np.nan), r.get("famc_ce_bench_hi", np.nan))
            rows.append([nm, f"{int(r.n_models)}/{int(r.n_families)}", fmt(r.theta_ray_fe, 2), f"{r.fam_ce_own:.1f}", bench,
                         famc, fmt(r.theta_c_mertens, 2), f"{r.res_ce_own:.1f}", resb,
                         f"{r.fam_ce_bench_loss_Besiroglu:.2f}" if np.isfinite(r.get("fam_ce_bench_loss_Besiroglu", np.nan)) else ""])
            rows.append(["", "", _ci(r.theta_ray_fe_lo, r.theta_ray_fe_hi, 2), _ci(r.fam_ce_own_lo, r.fam_ce_own_hi),
                         benchci, famcci, _ci(r.theta_c_mertens_lo, r.theta_c_mertens_hi, 2),
                         _ci(r.res_ce_own_lo, r.res_ce_own_hi), resbci, ""])
    rows.append(["\\midrule"])
    rows.append(["#Frontier models: Epoch Capabilities Index (ECI) on Epoch training compute"])
    for _, r in E.iterrows():
        rows.append([f"ECI, {r['sample']}", f"{int(r.n)}/{int(r.n_dev)}", "", "", "", "", fmt(r.theta_c, 2),
                     f"{r.res_ce:.1f}", "", ""])
        rows.append(["", "", "", "", "", "", f"({fmt(r.theta_c_se, 2)})", _ci(r.res_ce_lo, r.res_ce_hi), "", ""])
    header = ["", "Models/fam.", "$\\hat\\theta_C$", "own $\\theta$", "exp.\\ $\\theta$", "exp.\\ $\\theta$, $\\ln C$",
              "$\\hat\\theta_C$", "own $\\theta$", "exp.\\ $\\theta$", "Loss units"]
    pre = ("\\multicolumn{2}{c}{} & \\multicolumn{4}{c}{Family effects, 90/10} & \\multicolumn{3}{c}{Within-developer residuals, 90/10} & \\\\ "
           "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}")
    notes = ("Compute-equivalent 90/10 ratios $\\exp((\\omega_{90}-\\omega_{10})/\\theta_C)$: the factor by which the 10th-"
             "percentile family (or model) must scale training compute to match the 90th percentile. Family effects "
             "$\\omega_f$ from $y=a+\\theta_N\\ln N+\\theta_D\\ln D+\\omega_f$ (family fixed effects; $\\theta_C=(\\theta_N+"
             "\\theta_D)/2$); within-developer residuals from $y=a+\\theta_C\\ln C+$ developer and release-year effects "
             "(the Mertens et al.\\ design). `own $\\theta$' uses the "
             "regression's own compute elasticity; `exp.\\ $\\theta$' the design-matched experimental benchmark (Table 6, OLS "
             "row); `exp.\\ $\\theta$, $\\ln C$' nets inputs out with the same benchmark elasticity ($\\omega_f$ = family mean of "
             "$y-\\theta_C\\ln C$) instead of the within-family $(\\theta_N,\\theta_D)$, so that netting and conversion use one "
             "technology. Within-developer residuals exclude developers with a single model (residual identically zero). "
             f"Brackets: 90\\% intervals from {B if B else ''} developer-cluster bootstrap draws; for the `exp.\\ $\\theta$' columns "
             "$\\theta_C$ is held at its point value, so these intervals omit benchmark uncertainty. Loss units: family 90/10 "
             "(exp.\\ $\\theta$) raised to $\\gamma=0.178$ (Besiroglu et al.\\ Chinchilla refit), i.e.\\ the equivalent "
             "reducible-loss ratio at fixed compute under an assumed Chinchilla technology; compare Syverson's 90/10 TFP "
             "ratio of 1.92 in U.S.\\ manufacturing and Mertens et al.'s within-developer 41$\\times$ (MMLU-Pro). With "
             "returns to scale near one, Syverson's 1.92 is also an input-equivalent ratio, so the compute-equivalent columns, "
             "not the loss-unit column, are the like-for-like comparison; the loss-unit column depends on the output index. "
             "Distilled: Gemma-2 2B/9B; synthetic-data: Phi-1.5/2, SmolLM; code families: CodeLlama, StarCoder(2), "
             "DeepSeek-Coder.")
    p = os.path.join(TABDIR, f"{PREFIX}_{fname}.tex")
    write_tex_table(p, header, rows, caption="Productivity dispersion across LLM families in compute-equivalent units",
                    label=f"tab:{fname}", notes=notes, colspec="lccccccccc", size="\\scriptsize")
    s = open(p).read().replace("\\toprule\n", "\\toprule\n" + pre + "\n", 1)
    # drop the automatic header midrule duplication
    open(p, "w").write(s)
