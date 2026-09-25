"""m9_power_v2.py -- design-only power study, version 2 (round-3 fix list, item X2), and the summaries of
m9_coverage_v2.py. Written 2026-09-25 for Amendment 2 of the pre-analysis plan. Reads NO endpoint loss: every input is
the design (code/sweep/run_grid.py, architecture formulas) or a simulated replication.

Stages (CPU only, at most 2 processes by default, run with nice -n 10; never imports MLX):
  python m9_power_v2.py design           # N_F / N_P and N_F / N_T by width; noise-free sigma* by convention and truth
  python m9_power_v2.py q3null R         # Q3 slope under the null by noise s.d. (0 to 0.015) and within-trunk share
  python m9_power_v2.py summary          # coverage, calibration factors, equality test, tilt tests (from the reps)
Outputs: output/tables/m9_sweeps_power_v2*.csv (the originals m9_sweeps_power*.csv are left as committed).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

import m9_common as mc
import m9_est as est
import m9_power as pw
import m9_coverage_v2 as cv

TABLES = cv.TABLES
PREFIX = cv.PREFIX
log = cv.log
Z975, Z80 = 1.959964, 0.841621
N_BOOT_MC = 2000     # resamples of replications for the Monte Carlo s.e. of a calibration factor


def out(name):
    return os.path.join(TABLES, f"{PREFIX}{name}.csv")


# ============================================================================ design facts and by-construction offsets
def design_facts():
    rows = []
    for d, L in [(128, 2), (192, 3), (256, 4), (320, 5), (384, 6), (448, 7), (512, 8), (640, 10), (128, 4)]:
        a = mc.arch(d, L)
        NF = a["fpt"] / 6.0
        rows.append(dict(width=d, layers=L, N_P=float(a["N_nonemb"]), N_T=float(a["N_total"]), N_F=float(NF),
                         NF_over_NP=float(NF / a["N_nonemb"]), NF_over_NT=float(NF / a["N_total"]),
                         grid="high-M" if (d, L) == (128, 4) else "main"))
    df = pd.DataFrame(rows)
    df.to_csv(out("_nf_ratios"), index=False)
    m = df[df.grid == "main"]
    log(f"design: N_F/N_T {m.NF_over_NT.min():.3f}-{m.NF_over_NT.max():.3f} on the main grid, "
        f"{df[df.grid == 'high-M'].NF_over_NT.iloc[0]:.3f} at (128,4); N_F/N_P {m.NF_over_NP.min():.3f}-"
        f"{m.NF_over_NP.max():.3f}")
    return df


def offsets():
    """Noise-free model-free sigma* in N_F, P (FLOP path), P6 and T under each truth, both corpora' designs (the
    coverage target and the by-construction offsets between conventions), and the noise-free Q3 slope (P)."""
    rows = []
    for kind, kap in (("chin", 1.0), ("kappa", 0.36), ("kappa", 0.60)):
        r = cv.noise_free(kind, kap)
        for corp in ("edu", "web"):
            v = {c: r[f"mf_{c}|{corp}|est"] for c in ("F", "P", "P6", "T")}
            rows.append(dict(truth=kind, kappa=kap, corpus=corp, sig_F=v["F"], sig_P=v["P"], sig_P6=v["P6"],
                             sig_T=v["T"], F_minus_T=v["F"] - v["T"], F_minus_P=v["F"] - v["P"],
                             F_minus_P6=v["F"] - v["P6"], P_minus_T=v["P"] - v["T"],
                             q3_slope_P=r[f"q3|{corp}|est"], q3_mean_P=r[f"q3mean|{corp}"]))
    df = pd.DataFrame(rows)
    df.to_csv(out("_offsets"), index=False)
    log("offsets: " + "; ".join(f"{a.truth}{'' if a.truth == 'chin' else a.kappa} {a.corpus}: F {a.sig_F:.3f} "
                                f"P {a.sig_P:.3f} P6 {a.sig_P6:.3f} T {a.sig_T:.3f}" for a in df.itertuples()))
    return df


# ============================================================================ Q3 null by noise and within-trunk share
Q3_NOISE = (0.0, 0.002, 0.005, 0.0075, 0.010, 0.015)
Q3_RHO = (0.0, 0.5, 0.9)


def q3_rep(item, design=None, tp=None, s=None, rho=None, chi=0.22):
    """Q3 slope (Chinchilla form on M <= 100 against the local wedge) at one noise draw, both corpora, conventions P
    and T, with the high-M runs and on the main grid only (m9_power._fit_block's two variants)."""
    rep, seed = item
    rng = np.random.default_rng(seed)
    o = dict(rep=rep)
    for r in ("edu", "web"):
        df = design[r]
        c_ = chi if r == "edu" else 0.0
        lt = pw.true_lnL(tp, df.N_P.values, df.D.values, c_)
        f = lt + (pw.noise(df, s, rho, rng) if s > 0 else 0.0)
        main = df.is_main.values
        for conv in ("P", "T"):
            N, D, M = df[f"N_{conv}"].values, df.D.values, df[f"M_{conv}"].values
            restr = main & (M <= 100)
            starts = ([cv.chin_start(tp, c_)] if conv == "P" else []) + list(
                est.me.level_starts(N[restr], D[restr], np.exp(f[restr])))
            prep = cv.mf_prepare_fast(N, D, np.exp(f))
            sl, mean_, th, _ = cv.q3_slope(N, D, M, f, restr, prep, starts)
            prepm = cv.mf_prepare_fast(N[main], D[main], np.exp(f[main]))
            slm, meanm, _, _ = cv.q3_slope(N[main], D[main], M[main], f[main], restr[main], prepm, [th])
            o.update({f"slope|{r}|{conv}": sl, f"mean|{r}|{conv}": mean_, f"slope_main|{r}|{conv}": slm,
                      f"mean_main|{r}|{conv}": meanm})
    return o


def run_q3null(R=500):
    design = cv.design_v2()
    cells = [("chin", 1.0, s, rho) for s in Q3_NOISE for rho in Q3_RHO if not (s == 0.0 and rho != 0.5)]
    cells += [("kappa", 0.36, 0.005, 0.5), ("kappa", 0.60, 0.005, 0.5), ("kappa", 0.36, 0.0, 0.5),
              ("kappa", 0.60, 0.0, 0.5)]
    for kind, kap, s, rho in cells:
        tp = cv.truth_v2(kind, kap, design)
        name = f"{kind}{'' if kind == 'chin' else int(round(kap * 100))}_s{s:g}_r{rho:g}"
        path = os.path.join(TABLES, f"{PREFIX}_reps_q3null_{name}.csv")
        n = 1 if s == 0.0 else R
        items = [(i, mc.seed_of(f"v2q3|{name}|{i}")) for i in range(n)]
        cv.run_items(q3_rep, items, dict(design=design, tp=tp, s=s, rho=rho), path)


def summarize_q3null():
    rows = []
    for fn in sorted(os.listdir(TABLES)):
        if not fn.startswith(f"{PREFIX}_reps_q3null_"):
            continue
        name = fn[len(f"{PREFIX}_reps_q3null_"):-4]
        truth, s_, r_ = name.split("_")
        d = pd.read_csv(os.path.join(TABLES, fn))
        for col in d.columns:
            if col == "rep":
                continue
            stat, corp, conv = col.split("|")
            v = d[col].values.astype(float)
            v = v[np.isfinite(v)]
            rows.append(dict(truth=truth, noise_sd=float(s_[1:]), rho_trunk=float(r_[1:]), statistic=stat,
                             corpus=corp, conv=conv, n_rep=len(v), mean=float(np.mean(v)) if len(v) else np.nan,
                             mc_se_of_mean=float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan,
                             sd=float(np.std(v, ddof=1)) if len(v) > 2 else np.nan))
    df = pd.DataFrame(rows).sort_values(["statistic", "conv", "corpus", "truth", "noise_sd", "rho_trunk"])
    df.to_csv(out("_q3null"), index=False)
    return df


# ============================================================================ coverage and calibration summaries
def _factor_se(est_, se_, rng):
    """Monte Carlo s.e. of sd(est)/mean(se) by resampling replications."""
    n = len(est_)
    vals = []
    for _ in range(N_BOOT_MC):
        i = rng.integers(0, n, n)
        vals.append(np.std(est_[i], ddof=1) / np.mean(se_[i]))
    return float(np.std(vals, ddof=1))


def cov_rows(cell, d, nf):
    rng = np.random.default_rng(mc.seed_of(f"v2factorse|{cell}"))
    rows = []
    stats = sorted({c.split("|")[0] for c in d.columns if "|" in c and c.split("|")[1] in ("edu", "web")
                    and c.endswith("|est")})
    for stat in stats:
        for corp in ("edu", "web", "pooled"):
            for sch in ("wcu", "cr2cv"):
                if corp == "pooled":
                    parts = []
                    for c_ in ("edu", "web"):
                        parts.append(pd.DataFrame(dict(
                            dev=d[f"{stat}|{c_}|est"] - nf[f"{stat}|{c_}|est"], est=d[f"{stat}|{c_}|est"],
                            lo=d[f"{stat}|{c_}|{sch}|lo"], hi=d[f"{stat}|{c_}|{sch}|hi"],
                            se=d[f"{stat}|{c_}|{sch}|se"], nf=nf[f"{stat}|{c_}|est"])))
                    g = pd.concat(parts, ignore_index=True)
                else:
                    g = pd.DataFrame(dict(dev=d[f"{stat}|{corp}|est"] - nf[f"{stat}|{corp}|est"],
                                          est=d[f"{stat}|{corp}|est"], lo=d[f"{stat}|{corp}|{sch}|lo"],
                                          hi=d[f"{stat}|{corp}|{sch}|hi"], se=d[f"{stat}|{corp}|{sch}|se"],
                                          nf=nf[f"{stat}|{corp}|est"]))
                g = g[np.isfinite(g.dev) & np.isfinite(g.lo) & np.isfinite(g.hi) & np.isfinite(g.se)]
                n = len(g)
                if n < 20:
                    continue
                sd = float(np.std(g.dev, ddof=1))
                mse = float(np.mean(g.se))
                cov_ = float(np.mean((g.lo <= g.nf) & (g.hi >= g.nf)))
                rows.append(dict(cell=cell, statistic=stat, corpus=corp, scheme=sch, n_rep=n,
                                 noise_free_value=float(g.nf.mean()), mean_estimate=float(g.est.mean()),
                                 mean_bias=float(g.dev.mean()), mc_sd=sd, mc_sd_se=sd / np.sqrt(2 * (n - 1)),
                                 mean_boot_se=mse, factor=sd / mse,
                                 factor_mc_se=_factor_se(g.dev.values, g.se.values, rng),
                                 coverage=cov_, coverage_mc_se=float(np.sqrt(cov_ * (1 - cov_) / n)),
                                 mean_halfwidth=float(np.mean((g.hi - g.lo) / 2)),
                                 _mid=((g.lo + g.hi) / 2).values, _hw=((g.hi - g.lo) / 2).values, _nf=g.nf.values))
    # equality (edu - web)
    for stat in sorted({c.split("|")[0][5:] for c in d.columns if c.startswith("diff_") and c.endswith("|est")}):
        k = f"diff_{stat}"
        nfd = nf[f"{k}|est"]
        for sch in ("wcu", "cr2cv"):
            g = pd.DataFrame(dict(est=d[f"{k}|est"], lo=d[f"{k}|{sch}|lo"], hi=d[f"{k}|{sch}|hi"],
                                  se=d[f"{k}|{sch}|se"]))
            g = g[np.isfinite(g).all(axis=1)]
            n = len(g)
            sd = float(np.std(g.est, ddof=1))
            mse = float(np.mean(g.se))
            cov_ = float(np.mean((g.lo <= nfd) & (g.hi >= nfd)))
            rej = float(np.mean((g.lo > 0) | (g.hi < 0)))
            rows.append(dict(cell=cell, statistic=k, corpus="edu-web", scheme=sch, n_rep=n, noise_free_value=nfd,
                             mean_estimate=float(g.est.mean()), mean_bias=float(g.est.mean() - nfd), mc_sd=sd,
                             mc_sd_se=sd / np.sqrt(2 * (n - 1)), mean_boot_se=mse, factor=sd / mse,
                             factor_mc_se=_factor_se(g.est.values - nfd, g.se.values, rng), coverage=cov_,
                             coverage_mc_se=float(np.sqrt(cov_ * (1 - cov_) / n)), reject_zero=rej,
                             reject_zero_mc_se=float(np.sqrt(rej * (1 - rej) / n)),
                             mean_halfwidth=float(np.mean((g.hi - g.lo) / 2)),
                             mde80_oracle=(Z975 + Z80) * sd, mde80_uncalibrated=Z975 * mse + Z80 * sd,
                             _mid=((g.lo + g.hi) / 2).values, _hw=((g.hi - g.lo) / 2).values,
                             _nf=np.full(n, nfd), _zero=True))
    return rows


def summarize_cov():
    nfs = {}
    rows = []
    for cell, (kind, kap, s, rho, convs) in cv.COV_CELLS.items():
        path = os.path.join(TABLES, f"{PREFIX}_reps_cov_{cell}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        key = (kind, kap)
        if key not in nfs:
            nfs[key] = cv.noise_free(kind, kap)
        for r in cov_rows(cell, d, nfs[key]):
            r.update(truth=kind, kappa=kap, noise_sd=s, rho_trunk=rho)
            rows.append(r)
    # fixed factors: the largest cr2cv (CR2-scheme) factor per family across cells, conventions and corpora
    fam = lambda st_: "q3" if st_ == "q3" else ("equality" if st_.startswith("diff_") else "modelfree")
    cr = [r for r in rows if r["scheme"] == "cr2cv" and r["corpus"] != "pooled"]
    fixed = {}
    for f_ in ("modelfree", "q3", "equality"):
        cand = [r for r in cr if fam(r["statistic"]) == f_]
        if not cand:
            continue
        top = max(cand, key=lambda r: r["factor"])
        fixed[f_] = dict(family=f_, largest_factor=top["factor"], largest_factor_mc_se=top["factor_mc_se"],
                         at_cell=top["cell"], at_statistic=top["statistic"], at_corpus=top["corpus"])
    if "q3" in fixed:
        fixed["q3"]["applied_factor"] = max(1.75, fixed["q3"]["largest_factor"])
    for f_ in ("modelfree", "equality"):
        if f_ in fixed:
            fixed[f_]["applied_factor"] = fixed[f_]["largest_factor"]
    # calibrated coverage (half-width times the applied factor about the midpoint) and calibrated equality tests
    for r in rows:
        f_ = fam(r["statistic"])
        c = fixed.get(f_, {}).get("applied_factor", np.nan)
        mid, hw, nf_ = r.pop("_mid"), r.pop("_hw"), r.pop("_nf")
        zero = r.pop("_zero", False)
        lo, hi = mid - c * hw, mid + c * hw
        cc = float(np.mean((lo <= nf_) & (hi >= nf_)))
        r.update(applied_factor=c, coverage_calibrated=cc, coverage_calibrated_mc_se=float(np.sqrt(cc * (1 - cc) / len(mid))))
        if zero:
            rz = float(np.mean((lo > 0) | (hi < 0)))
            r.update(reject_zero_calibrated=rz, reject_zero_calibrated_mc_se=float(np.sqrt(rz * (1 - rz) / len(mid))),
                     mde80_calibrated=Z975 * c * r["mean_boot_se"] + Z80 * r["mc_sd"])
    df = pd.DataFrame(rows)
    df.to_csv(out("_coverage"), index=False)
    fx = pd.DataFrame(list(fixed.values()))
    fx.to_csv(out("_factors"), index=False)
    return df, fx


# ============================================================================ tilt summaries
def summarize_tilt():
    rows, cal_rows = [], []
    for chi in (0.0, 0.22):
        path = os.path.join(TABLES, f"{PREFIX}_reps_tilt_chi{chi}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        n = len(d)

        def rate(mask):
            p = float(np.mean(mask))
            return p, float(np.sqrt(p * (1 - p) / n))

        base = dict(chi_true=chi, n_rep=n, noise_sd=0.005, rho_trunk=0.5, conv="P")
        for vi in ("val1", "val2"):
            e = d[f"chi|{vi}"].values
            sd = float(np.std(e, ddof=1))
            rows.append(dict(base, valset=vi, statistic="chi_hat", value=float(np.mean(e)), mc_se=sd / np.sqrt(n)))
            rows.append(dict(base, valset=vi, statistic="chi_hat_mc_sd", value=sd, mc_se=sd / np.sqrt(2 * (n - 1))))
            rows.append(dict(base, valset=vi, statistic="mde80_normal", value=(Z975 + Z80) * sd,
                             mc_se=(Z975 + Z80) * sd / np.sqrt(2 * (n - 1))))
            for v in ("wcu", "cr2"):
                lo, hi, se = d[f"{v}|{vi}|lo"].values, d[f"{v}|{vi}|hi"].values, d[f"{v}|{vi}|se"].values
                for nm, mask in ((f"{v}_interval_rejects_zero", (lo > 0) | (hi < 0)),
                                 (f"{v}_interval_covers_truth", (lo <= chi) & (hi >= chi)),
                                 (f"wcr_{'raw' if v == 'wcu' else 'cr2'}_p_below_0.05", d[f"{v}|{vi}|p_wcr"].values < 0.05)):
                    p, s_ = rate(mask)
                    rows.append(dict(base, valset=vi, statistic=nm, value=p, mc_se=s_))
                rows.append(dict(base, valset=vi, statistic=f"{v}_mean_boot_se", value=float(np.mean(se)), mc_se=np.nan))
                rows.append(dict(base, valset=vi, statistic=f"{v}_factor", value=sd / float(np.mean(se)), mc_se=np.nan))
                rows.append(dict(base, valset=vi, statistic=f"{v}_mean_halfwidth", value=float(np.mean((hi - lo) / 2)),
                                 mc_se=np.nan))
        # decision rules on two validation sets
        c1, c2 = d["chi|val1"].values, d["chi|val2"].values
        same = np.sign(c1) == np.sign(c2)
        lo1, hi1, lo2, hi2 = (d[f"wcu|{v}|{b}"].values for v in ("val1", "val2") for b in ("lo", "hi"))
        ex1, ex2 = (lo1 > 0) | (hi1 < 0), (lo2 > 0) | (hi2 < 0)
        plan_fb = ex1 & ex2 & (np.sign(lo1 + hi1) == np.sign(lo2 + hi2))
        plan_ne = ~ex1 & ~ex2 & ((hi1 - lo1) / 2 < 0.10) & ((hi2 - lo2) / 2 < 0.10)
        p1, p2 = d["cr2|val1|p_wcr"].values, d["cr2|val2|p_wcr"].values
        hw1 = (d["cr2|val1|hi"] - d["cr2|val1|lo"]).values / 2
        hw2 = (d["cr2|val2|hi"] - d["cr2|val2|lo"]).values / 2
        d8_fb = (p1 < 0.05) & (p2 < 0.05) & same
        d8_ne = (p1 >= 0.05) & (p2 >= 0.05) & (hw1 < 0.10) & (hw2 < 0.10)
        for nm, mask in (("plan_rule_factor_biased", plan_fb), ("plan_rule_neutral", plan_ne),
                         ("plan_rule_inconclusive", ~plan_fb & ~plan_ne), ("d8_rule_factor_biased", d8_fb),
                         ("d8_rule_neutral", d8_ne), ("d8_rule_inconclusive", ~d8_fb & ~d8_ne),
                         ("both_schemes_factor_biased", plan_fb & d8_fb), ("both_schemes_neutral", plan_ne & d8_ne)):
            p, s_ = rate(mask)
            rows.append(dict(base, valset="val1+val2", statistic=nm, value=p, mc_se=s_))
        cal_rows.append((chi, d, same))
    # CR2 interval calibrated by the tilt factor (Amendment 2, item 6): the largest ratio of the Monte Carlo s.d. of
    # chi-hat to the mean CR2 bootstrap s.e. over chi in {0, 0.22} and the two validation sets; half-width multiplied
    # about the midpoint; "excludes zero" / "neutral" (contains zero, half-width below 0.10) on both validation sets.
    facs = [np.std(d[f"chi|{vi}"], ddof=1) / np.mean(d[f"cr2|{vi}|se"]) for _, d, _ in cal_rows for vi in ("val1", "val2")]
    c = float(np.max(facs)) if facs else np.nan
    for chi, d, same in cal_rows:
        n = len(d)
        base = dict(chi_true=chi, n_rep=n, noise_sd=0.005, rho_trunk=0.5, conv="P")
        ex, ne = [], []
        for vi in ("val1", "val2"):
            lo, hi = d[f"cr2|{vi}|lo"].values, d[f"cr2|{vi}|hi"].values
            mid, hw = (lo + hi) / 2, c * (hi - lo) / 2
            e_ = ((mid - hw) > 0) | ((mid + hw) < 0)
            ex.append(e_)
            ne.append(~e_ & (hw < 0.10))
            p = float(np.mean(e_))
            rows.append(dict(base, valset=vi, statistic="cr2_calibrated_rejects_zero", value=p,
                             mc_se=float(np.sqrt(p * (1 - p) / n)), factor=c))
        for nm, mask in (("cr2_calibrated_rule_factor_biased", ex[0] & ex[1] & same),
                         ("cr2_calibrated_rule_neutral", ne[0] & ne[1]),
                         ("cr2_calibrated_rule_inconclusive", ~(ex[0] & ex[1] & same) & ~(ne[0] & ne[1]))):
            p = float(np.mean(mask))
            rows.append(dict(base, valset="val1+val2", statistic=nm, value=p, mc_se=float(np.sqrt(p * (1 - p) / n)),
                             factor=c))
    df = pd.DataFrame(rows)
    df.to_csv(out("_tilt"), index=False)
    return df


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import m9_power_v2 as mod
    what = sys.argv[1]
    if what == "design":
        mod.design_facts()
        mod.offsets()
    elif what == "q3null":
        mod.run_q3null(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
    elif what == "summary":
        mod.summarize_q3null()
        mod.summarize_tilt()
        mod.summarize_cov()
