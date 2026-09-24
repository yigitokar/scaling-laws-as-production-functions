"""Step 4 of ra4_obsfix: productivity (TFP) dispersion in both unit systems (R4 M11; R1 comment 10(e)).

A Hicks-neutral productivity gap Delta (p90 - p10 of family effects, in HellaSwag log-odds) can be cardinalized as
  (i)   output units, native:      exp(Delta)                  (odds of above-chance success);
  (ii)  input (compute) units:     exp(Delta / theta_C)        (compute the p10 family needs to match the p90 family);
  (iii) output units, reducible loss: exp(gamma * Delta / theta_C) = (ii)^gamma, gamma the frontier elasticity of
        reducible loss (0.178 Besiroglu, 0.155 Hoffmann).
Syverson's (2004) manufacturing 90/10 of 1.92 is an output-unit TFP ratio; with returns to scale near one it is also
an input-equivalent ratio.  For LLMs returns to compute are small (theta_C ~ 0.33-0.43 in log-odds, gamma ~ 0.16-0.18
in loss), so the two unit systems differ by a factor 1/theta_C (or 1/gamma) in logs: the comparison with
manufacturing is a cardinalization choice.  Intervals: developer-cluster bootstrap jointly with draws of the
benchmark theta_C from the design-conditional surface bootstrap of step_matched (m4 held theta_C fixed).
Reuses m4_observational (panel.build_panel; tfp sample definitions) by import.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ra4common as rc  # noqa: E402

sys.path.insert(0, rc.M4DIR)
import panel  # noqa: E402   (m4)
import tfp as m4tfp  # noqa: E402

GAMMA = {"Besiroglu": 0.1783, "Hoffmann": 0.1548}
SYVERSON = 1.92


def dummies(labels):
    return pd.get_dummies(pd.Series(labels).astype(str), drop_first=True, dtype=float).to_numpy()


def stats(d, out, theta_bench):
    """Dispersion statistics (log units) for one sample: family effects with the within-family (theta_N, theta_D)
    netting ('fam'), family effects netted with the benchmark theta_C ('famc'), and Mertens-design residuals within
    developer ('res', with the regression's own theta_C)."""
    y = d[out].to_numpy(float)
    n, dd_, c = d.n.to_numpy(float), d.d.to_numpy(float), d.c.to_numpy(float)
    X = np.column_stack([np.ones(len(d)), n, dd_, dummies(d.family)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    om = pd.Series(y - b[1] * n - b[2] * dd_).groupby(d.family.to_numpy()).mean()
    omc = pd.Series(y - theta_bench * c).groupby(d.family.to_numpy()).mean()
    X2 = np.column_stack([np.ones(len(d)), c, dummies(d.developer), dummies(d.year)])
    b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    res = y - X2 @ b2
    multi = d.developer.map(d.developer.value_counts()).to_numpy() >= 2
    p9010 = lambda v: float(np.percentile(v, 90) - np.percentile(v, 10))
    return dict(fam=p9010(om), famc=p9010(omc), res=p9010(res[multi]), theta_own=float(b2[1]),
                theta_fe_ray=float(0.5 * (b[1] + b[2])), n_fam=int(len(om)), n_models=int(len(d)))


def main():
    t0 = time.time()
    df = panel.build_panel(verbose=False)
    m = df.main
    samples = {"All dense base models": m,
               "Drop distilled and synthetic-data": m & ~df.model.isin(m4tfp.DISTILL_SYN) & ~df.family.isin(m4tfp.SYN_FAMS),
               "Also drop code-specialised": m & ~df.model.isin(m4tfp.DISTILL_SYN) & ~df.family.isin(m4tfp.SYN_FAMS | m4tfp.CODE_FAMS)}
    draws = np.load(os.path.join(rc.PROC, "bench_thetaC_draws.npz"))
    benches = {"y_hellaswag": {"legacy benchmark (Table 8 design, 57 models)": "legacy__legacy support (57)__y_hellaswag",
                               "strict benchmark (20 models)": "strict__strict support (20)__y_hellaswag"},
               "y_core": {"legacy benchmark (Table 8 design, 57 models)": "legacy__legacy support (57)__y_core",
                          "strict benchmark (20 models)": "strict__strict support (20)__y_core"}}
    rows = []
    rng = np.random.default_rng(rc.SEED + 7)
    for out, bset in benches.items():
        for blab, key in bset.items():
            th0 = float(draws[key + "__point"][0])
            thd = draws[key]
            for sname, sm in samples.items():
                d = df[sm].dropna(subset=[out]).copy()
                s0 = stats(d, out, th0)
                devs = d.developer.unique()
                gidx = {g: np.where(d.developer.to_numpy() == g)[0] for g in devs}
                B = []
                for b in range(rc.B_TFP):
                    pick = rng.choice(devs, len(devs), replace=True)
                    ii = np.concatenate([gidx[g] for g in pick])
                    tag = np.repeat(np.arange(len(pick)).astype(str), [len(gidx[g]) for g in pick])
                    bd = d.iloc[ii].copy()
                    bd["developer"] = bd.developer.to_numpy() + "#" + tag
                    bd["family"] = bd.family.to_numpy() + "#" + tag
                    th = float(thd[rng.integers(0, len(thd))])
                    try:
                        sb = stats(bd, out, th)
                    except Exception:
                        continue
                    sb["theta_b"] = th
                    B.append(sb)
                B = pd.DataFrame(B)
                for measure, delta, dB, conv, convB in [
                        ("Family effects, inputs netted with benchmark theta_C", s0["famc"], B.famc, th0, B.theta_b),
                        ("Family effects, within-family (theta_N, theta_D) netting", s0["fam"], B.fam, th0, B.theta_b),
                        ("Within-developer residuals (Mertens design), own theta_C", s0["res"], B.res, s0["theta_own"], B.theta_own)]:
                    q = lambda v: np.nanpercentile(v, [5, 95])
                    ce, ceB = np.exp(delta / conv), np.exp(dB / convB)
                    r = dict(output=out, benchmark=blab, sample=sname, measure=measure, n_models=s0["n_models"],
                             n_families=s0["n_fam"], theta_conv=conv, delta_logodds=delta,
                             delta_lo=q(dB)[0], delta_hi=q(dB)[1], odds_ratio=np.exp(delta),
                             odds_lo=np.exp(q(dB)[0]), odds_hi=np.exp(q(dB)[1]), compute_eq=ce, compute_eq_lo=q(ceB)[0],
                             compute_eq_hi=q(ceB)[1], B_ok=int(len(B)))
                    for gl, gv in GAMMA.items():
                        r[f"loss_{gl}"] = ce ** gv
                        r[f"loss_{gl}_lo"], r[f"loss_{gl}_hi"] = q(ceB ** gv)
                    # log-units ratio to manufacturing (Syverson 1.92)
                    r["log_ratio_to_syverson_compute"] = np.log(ce) / np.log(SYVERSON)
                    r["log_ratio_to_syverson_loss_Besiroglu"] = np.log(ce ** GAMMA["Besiroglu"]) / np.log(SYVERSON)
                    rows.append(r)
            rc.log(f"  {out} / {blab} done ({time.time() - t0:.0f}s)")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(rc.TAB, f"{rc.PREFIX}_tfp_units.csv"), index=False)
    rc.dump_json(dict(runtime_sec=time.time() - t0, B=rc.B_TFP, gamma=GAMMA, syverson=SYVERSON), "tfp_headline.json")
    rc.log(f"step_tfp finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
