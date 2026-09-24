"""Module m4_observational: observational (cross-lab) production functions vs experimental benchmarks.

Single entry point; regenerates every output of the module from data/raw (deterministic seeds):
    /Users/yigitokar/scaling-laws-pf/.venv/bin/python code/analysis/m4_observational/run.py [--fast]
Raw data needed beyond code/data/download_public.sh: run code/data/download_m4_observational.sh once
(DataDecide eval results, Pythia per-checkpoint evals, Gadre et al. downstream evals, HF model metadata).

Outputs (prefix m4_observational_):
  tables : exp_benchmarks, curvature, design_audit, table6_lalonde (design-matched, HellaSwag; Table 6),
           table6_lalonde_core / _ladderonly (robustness), table6b_global_overlap / _main (all estimators incl. IV,
           EIV, FD, ACF, reverse), iv_diagnostics, regimes_semisynthetic, tfp_table, tfp_dispersion, eci_dispersion,
           pythia_output_check, headline (json in data/processed)
  figures: fig5_lalonde, support, elasticity_by_M, tfp_families, regimes
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import common as cm  # noqa: E402
import experiments as ex  # noqa: E402
import figs  # noqa: E402
import lalonde as ll  # noqa: E402
import observational as ob  # noqa: E402
import panel  # noqa: E402
import semisynth  # noqa: E402
import tfp  # noqa: E402

FAST = "--fast" in sys.argv
B = 100 if FAST else 300          # bootstrap draws (surface refits, ACF, dispersion)
R_SIM = 100 if FAST else 400      # semi-synthetic replications


def exp_table(est):
    est.to_csv(os.path.join(cm.TABDIR, f"{cm.PREFIX}_exp_benchmarks.csv"), index=False)
    rows = []
    outs = [("y_hellaswag", "HellaSwag"), ("y_arc_c", "ARC-Challenge"), ("y_winogrande", "Winogrande"), ("y_core", "Composite")]
    dsets = ["OLMo ladder", "OLMo ladder + OLMo-2", "Gadre et al. (N>=0.1B)", "Gadre et al.", "DataDecide"]
    for ds in dsets:
        top, bot = [ds.replace(">=", "$\\geq$")], [""]
        for o, _ in outs:
            for par in ["theta_N", "theta_D", "theta_C_ray"]:
                r = est[(est.dataset == ds) & (est.output == o) & (est.param == par)]
                top.append(cm.fmt(r.est.iloc[0], 2) if len(r) else "")
                bot.append(f"({cm.fmt(r.se.iloc[0], 2)})" if len(r) else "")
        n = est[est.dataset == ds].n_obs.iloc[0]
        rows += [top + [str(int(n))], bot + [""]]
    rows.append(["\\midrule"])
    for sh in [0, 5]:
        ds = f"Pythia ({sh}-shot)"
        top, bot = [ds + ", ARC-C / WG"], [""]
        for o in ["y_hellaswag", "y_arc_c", "y_winogrande", "y_core"]:
            for par in ["theta_N", "theta_D_ckpt", "none"]:
                r = est[(est.dataset == ds) & (est.output == o) & (est.param == par)]
                top.append(cm.fmt(r.est.iloc[0], 2) if len(r) else "")
                bot.append(f"({cm.fmt(r.se.iloc[0], 2)})" if len(r) else "")
        n_fin = est[(est.dataset == ds) & (est.param == "theta_N")].n_obs.max()
        n_ck = est[(est.dataset == ds) & (est.param == "theta_D_ckpt")].n_obs.max()
        rows += [top + [f"{int(n_fin)} / {int(n_ck)}"], bot + [""]]
    header = [""] + ["$\\theta_N$", "$\\theta_D$", "$\\theta_C$"] * 4 + ["Runs"]
    pre = ("\\multicolumn{1}{c}{} & \\multicolumn{3}{c}{HellaSwag} & \\multicolumn{3}{c}{ARC-Challenge} & "
           "\\multicolumn{3}{c}{Winogrande} & \\multicolumn{3}{c}{Composite} & \\\\ "
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}")
    notes = ("Global log-linear fits $y=a+\\theta_N\\ln N+\\theta_D\\ln D$ on designed sweeps, "
             "$y=\\text{logit}((acc-\\text{chance})/(1-\\text{chance}))$; $\\theta_C=(\\theta_N+\\theta_D)/2$ is the elasticity "
             "along a proportional ray. OLMo ladder: 190M--3B $\\times$ 0.5--10$\\times$ Chinchilla tokens (30 runs incl.\\ "
             "reruns), OLMES 5-shot cloze formats; + OLMo-2 7B (3.9T) and 13B (5.0T). Gadre et al.: 3 corpora "
             "(corpus effects), 11M--6.9B, 5--640 tokens/parameter, LLM-foundry ICL evals. DataDecide: 25 recipes "
             "(recipe effects, SEs clustered by recipe), final checkpoints with $N\\geq$60M on the $D=100N$ ray, so only "
             "$\\theta_C$ is identified. Pythia: $\\theta_N$ from the 15 final models with released evals (no pythia-1b standard) at $D=300$B (dedup dummy); $\\theta_D$ from "
             "within-run checkpoints $\\geq$27B tokens (model effects; confounded by the cosine schedule), composite = ARC-C and "
             "Winogrande (the released Pythia evals contain no HellaSwag). HC1 standard errors in parentheses.")
    p = os.path.join(cm.TABDIR, f"{cm.PREFIX}_exp_benchmarks.tex")
    cm.write_tex_table(p, header, rows, "Experimental elasticities of benchmark log-odds (designed sweeps)",
                       "tab:m4_exp_benchmarks", notes, colspec="l" + "c" * 13, size="\\scriptsize")
    s = open(p).read().replace("\\toprule\n", "\\toprule\n" + pre + "\n", 1)
    open(p, "w").write(s)


def design_audit(df):
    m = df[df.main]
    rows = []
    for nm, d in [("All dense base models", m), ("Overlap (N<=14B, D<=5T)", m[m.overlap])]:
        fam = d.groupby("family")
        wn = d.n - fam.n.transform("mean")
        wd = d.d - fam.d.transform("mean")
        k = fam.size()
        dvar = fam.d.std().fillna(0) > 0.05
        rows.append(dict(sample=nm, models=len(d), families=d.family.nunique(), developers=d.developer.nunique(),
                         singleton_families=int((k == 1).sum()), sd_n=d.n.std(), sd_d=d.d.std(),
                         corr_nd=np.corrcoef(d.n, d.d)[0, 1], within_sd_n=wn.std(), within_sd_d=wd.std(),
                         within_corr_nd=np.corrcoef(wn, wd)[0, 1] if wd.std() > 0 else np.nan,
                         fams_with_D_variation=int(dvar.sum()), models_in_them=int(k[dvar].sum()),
                         # off-ray variation: SD of ln(D/N) after projecting on ln C (0 on a single expansion path)
                         sd_logM_given_c=float(np.std((d.d - d.n) - np.polyval(np.polyfit(d.c, d.d - d.n, 1), d.c))),
                         median_M=float(d.M.median()), share_C_equals_6ND=1.0, date_min=str(d.date.min())[:7],
                         date_max=str(d.date.max())[:7],
                         # [review addition] measurement of N: exact HF float-tensor counts vs the reported size
                         n_exact_N=int((d.N_source == "HF safetensors").sum()),
                         exact_over_reported_mean=float((d.N / (d.N_rep * 1e9) - 1)[d.N_source == "HF safetensors"].mean()),
                         exact_over_reported_sd=float((d.N / (d.N_rep * 1e9) - 1)[d.N_source == "HF safetensors"].std())))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(cm.TABDIR, f"{cm.PREFIX}_design_audit.csv"), index=False)
    lab = [("models", "Models", 0), ("families", "Families", 0), ("developers", "Developers", 0),
           ("singleton_families", "Singleton families", 0), ("sd_n", "SD $\\ln N$", 2), ("sd_d", "SD $\\ln D$", 2),
           ("corr_nd", "Corr$(\\ln N,\\ln D)$", 2), ("within_sd_n", "Within-family SD $\\ln N$", 2),
           ("within_sd_d", "Within-family SD $\\ln D$", 2), ("within_corr_nd", "Within-family corr", 2),
           ("fams_with_D_variation", "Families with $D$ variation", 0), ("models_in_them", "... models in them", 0),
           ("sd_logM_given_c", "SD $\\ln(D/N)\\mid\\ln C$", 2), ("median_M", "Median $D/N$", 0)]
    rows_t = [[l] + [(str(int(r[k])) if nd == 0 else cm.fmt(r[k], nd)) for _, r in A.iterrows()] for k, l, nd in lab]
    cm.write_tex_table(os.path.join(cm.TABDIR, f"{cm.PREFIX}_design_audit.tex"), ["", "All dense", "Overlap"], rows_t,
                       "Identifying variation in the observational panel", "tab:m4_design_audit",
                       "Observational panel: ObsScaling base models plus Sloth additions, dense transformers with known $N$ "
                       "and $D$ and all three core benchmarks. Within-family moments after removing family means. "
                       "`Families with $D$ variation': within-family SD of $\\ln D$ above 0.05; $\\theta_D$ under family "
                       "fixed effects is identified only from these families (Cerebras-GPT has $D=20N$ exactly, so it "
                       "identifies only $\\theta_N+\\theta_D$). Compute equals $6ND$ by construction in every row.",
                       colspec="lcc")
    return A


def pythia_output_check(df, pyt):
    """Output comparability: the same Pythia models measured by the observational pipeline (Open LLM Leaderboard,
    25-shot ARC-C, 5-shot Winogrande) vs the experimenters' own evals (lm-eval-harness 0/5-shot)."""
    import statsmodels.api as sm
    rows = []
    obs = df[(df.family == "Pythia") & df.main].copy()
    # [review fix] compare the SAME models in both sources: the leaderboard panel and the released Pythia evals
    # (the latter has no pythia-1b standard; the former has no duplicated Sloth copies after the panel fix)
    obs["base"] = obs.model.str.split("/").str[-1]
    common = sorted(set(obs.base) & set(pyt.model))
    obs = obs[obs.base.isin(common)]
    for out in ["y_arc_c", "y_winogrande"]:
        X = sm.add_constant(pd.concat([obs[["n"]], obs.model.str.contains("deduped").astype(float).rename("dedup")], axis=1))
        r = cm.ols(obs[out], X)
        rows.append(dict(source="Open LLM Leaderboard (observational)", output=out, theta_N=r.params["n"], se=r.bse["n"], n=len(obs),
                         models=";".join(common)))
        for sh in [0, 5]:
            p = pyt[(pyt.shots == sh) & (pyt.step == 143000) & pyt.model.isin(common)]
            X = sm.add_constant(pd.concat([p[["n"]], p.dedup.astype(float)], axis=1))
            r = cm.ols(p[out], X)
            rows.append(dict(source=f"Pythia evals, {sh}-shot (experimental)", output=out, theta_N=r.params["n"], se=r.bse["n"], n=len(p),
                             models=";".join(common)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(cm.TABDIR, f"{cm.PREFIX}_pythia_output_check.csv"), index=False)
    return P


def main():
    t0 = time.time()
    rng_seed = cm.SEED
    panel.write_hf_ids()
    df = panel.build_panel()
    lad, gad, pyt, dd_all, dd_fin = ex.build_all()
    est = ex.estimate_experiments(lad, gad, pyt, dd_all, dd_fin)
    exp_table(est)
    curv = ex.curvature_check(lad, gad)
    curv.to_csv(os.path.join(cm.TABDIR, f"{cm.PREFIX}_curvature.csv"), index=False)
    audit = design_audit(df)
    pchk = pythia_output_check(df, pyt)
    print(f"[run] data + experiments done ({time.time() - t0:.0f}s)")

    # ---- Table 6: design-matched LaLonde comparison
    labels = {"y_hellaswag": "HellaSwag logit", "y_core": "composite logit (HellaSwag, ARC-C, Winogrande)",
              "y_arc_c": "ARC-Challenge logit", "y_winogrande": "Winogrande logit"}
    fname_label = {"table6_lalonde_clean": "HellaSwag logit; excluding Qwen1.5, distilled, synthetic-data and code models"}
    matched = {}
    for out, surf, fname in [("y_hellaswag", "ladder+gadre", "table6_lalonde"),
                             ("y_hellaswag", "ladder", "table6_lalonde_ladderonly"),
                             ("y_core", "ladder+gadre", "table6_lalonde_core"),
                             ("y_arc_c", "ladder+gadre", "table6_lalonde_arc"),
                             ("y_winogrande", "ladder+gadre", "table6_lalonde_wino"),
                             ("y_hellaswag", "ladder+gadre|clean", "table6_lalonde_clean")]:
        dfx = df
        if surf.endswith("|clean"):
            # drop models whose per-size D is not officially documented (Qwen1.5: ObsScaling imputation), distilled
            # (Gemma-2 2B/9B) and synthetic-data (Phi, SmolLM) models, and code-specialised families
            surf = surf.split("|")[0]
            dfx = df.copy()
            drop = dfx.family.isin(["Qwen1.5", "Phi", "SmolLM"] + list(tfp.CODE_FAMS)) | dfx.model.isin(tfp.DISTILL_SYN)
            dfx["main"] = dfx.main & ~drop
        res, hull_obs, info = ll.design_matched(dfx, lad, gad, out=out, surface=surf, B=B, seed=rng_seed)
        ll.make_matched_table(res, fname_label.get(fname, labels[out]), fname, info)
        matched[(out, surf) if fname != "table6_lalonde_clean" else ("clean", surf)] = (res, hull_obs, info)
        print(f"[run] {fname}: hull n={info['n_hull']}  ({time.time() - t0:.0f}s)")
    res6, hull6, info6 = matched[("y_hellaswag", "ladder+gadre")]
    # [review addition] few-cluster inference for the Table 6 biases: restricted wild cluster bootstrap-t (Webb weights)
    wild6 = ll.wild_bias_table(res6, hull6, "y_hellaswag", "table6_lalonde", B=999, seed=rng_seed)
    resc, hullc, _ = matched[("clean", "ladder+gadre")]
    wildc = ll.wild_bias_table(resc, hullc, "y_hellaswag", "table6_lalonde_clean", B=999, seed=rng_seed)
    print(f"[run] wild-cluster bootstrap for Table 6 done ({time.time() - t0:.0f}s)")

    # ---- Table 6b: all estimators on the overlap and full samples (global experimental fits for reference)
    glob_rows = {}
    for sample in ["overlap", "main"]:
        allr, tr, iv, path = ll.run(df, lad, gad, dd_fin, est, out="y_hellaswag", sample=sample, B_acf=B, seed=rng_seed)
        ll.make_table(allr, tr, est, "y_hellaswag", sample, "HellaSwag logit", f"table6b_global_{sample}")
        allr.to_csv(os.path.join(cm.TABDIR, f"{cm.PREFIX}_table6b_global_{sample}_long.csv"), index=False)
        iv.to_csv(os.path.join(cm.TABDIR, f"{cm.PREFIX}_iv_diagnostics_{sample}.csv"), index=False)
        glob_rows[sample] = (allr, tr, iv)
        print(f"[run] table6b ({sample}) done ({time.time() - t0:.0f}s)")

    # ---- Semi-synthetic regimes (DataDecide)
    sim = semisynth.simulate(dd_fin, R=R_SIM, seed=rng_seed)
    print(f"[run] semi-synthetic regimes done ({time.time() - t0:.0f}s)")

    # ---- TFP dispersion (benchmark theta_C = design-matched OLS benchmark for each output)
    bench_theta = {}
    for out in ["y_hellaswag", "y_core"]:
        r = matched[(out, "ladder+gadre")][0]
        bench_theta[out] = float(r[(r.estimator == "OLS") & (r.param == "theta_C")].bench.iloc[0])
    T, fam_eff, dev_eff, df2, pc_share, pc_w = tfp.run_tfp(df, bench_theta, B=B, seed=rng_seed)
    E, eci_df = tfp.eci_dispersion(B=B, seed=rng_seed)
    tfp.make_tfp_table(T, E, B=B)
    print(f"[run] TFP dispersion done ({time.time() - t0:.0f}s)")

    # ---- figures
    figs.fig_lalonde(res6, est, out="y_hellaswag")
    figs.fig_support(df, lad, gad, set(hull6.model))
    E_ = ll.exp_surface_data(lad, gad, "ladder+gadre").dropna(subset=["y_hellaswag"])
    center = (E_.n.mean(), E_.d.mean())
    surf = ll.fit_surface(E_, "y_hellaswag", center)
    figs.fig_elasticity_by_M(surf.params, center, df, hull_pts=E_[["n", "d"]].values)
    figs.fig_tfp(fam_eff["y_hellaswag"], df, bench_theta["y_hellaswag"])
    figs.fig_regimes(sim)

    # ---- headline numbers for the memo
    def g(res, est_, par, k):
        r = res[(res.estimator == est_) & (res.param == par)]
        return float(r[k].iloc[0]) if len(r) else None
    H = dict(
        panel=dict(rows=len(df), main=int(df.main.sum()), overlap=int(df.overlap.sum()),
                   families=int(df[df.main].family.nunique()), developers=int(df[df.main].developer.nunique()),
                   hull=int(info6["n_hull"]), notable=int(df[df.main].notable.sum()), in_epoch=int(df[df.main].in_epoch.sum())),
        experiments=est[est.param != "R2"].to_dict(orient="records"),
        matched_hellaswag=res6.to_dict(orient="records"), matched_hellaswag_wild=wild6.to_dict(orient="records"),
        matched_info=dict(n_hull=info6["n_hull"], n_exp=info6["n_exp"], B=info6.get("B"), B_ok=info6.get("B_ok")),
        matched_core=matched[("y_core", "ladder+gadre")][0].to_dict(orient="records"),
        matched_ladderonly=matched[("y_hellaswag", "ladder")][0].to_dict(orient="records"),
        matched_clean=matched[("clean", "ladder+gadre")][0].to_dict(orient="records"),
        matched_clean_wild=wildc.to_dict(orient="records"),
        matched_arc=matched[("y_arc_c", "ladder+gadre")][0].to_dict(orient="records"),
        matched_wino=matched[("y_winogrande", "ladder+gadre")][0].to_dict(orient="records"),
        surface=dict(R2=info6["surface_R2"], n_exp=info6["n_exp"], params=info6["surface_params"]),
        global_overlap=glob_rows["overlap"][0].to_dict(orient="records"),
        global_truth_overlap={k: (float(v) if np.ndim(v) == 0 else None) for k, v in glob_rows["overlap"][1].items() if k != "V"},
        global_truth_main={k: (float(v) if np.ndim(v) == 0 else None) for k, v in glob_rows["main"][1].items() if k != "V"},
        global_main=glob_rows["main"][0].to_dict(orient="records"),
        iv_overlap=glob_rows["overlap"][2].to_dict(orient="records"),
        iv_main=glob_rows["main"][2].to_dict(orient="records"),
        regimes=sim.astype({"regime": str}).to_dict(orient="records"),
        tfp=T.to_dict(orient="records"), eci=E.to_dict(orient="records"), pc1_share=pc_share, pc1_weights=pc_w,
        pythia_check=pchk.to_dict(orient="records"), curvature=curv.to_dict(orient="records"),
        audit=audit.to_dict(orient="records"), runtime_sec=time.time() - t0)
    with open(os.path.join(cm.PROC, "headline.json"), "w") as f:
        json.dump(H, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[run] all outputs written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
