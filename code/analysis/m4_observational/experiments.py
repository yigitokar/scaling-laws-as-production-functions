"""Experimental benchmarks: designed sweeps evaluated on the SAME benchmarks as the observational panel.

Designs (each is a single-lab experiment in which (N, D) are set by the experimenter, not by a lab's response to
its own productivity -- the "LaLonde" ground truth):
  E1 OLMo ladder (Bhagia et al. 2024): 5 sizes x 5 token multipliers (0.5-10x Chinchilla) + 5 reruns = 30 final
     checkpoints, plus the two OLMo-2 targets (7B/4T, 13B/5T) trained with the same recipe. Factorial => theta_N,
     theta_D separately identified. OLMES 5-shot "rc" (cloze) formats with length normalisation.
  E2 Gadre et al. (2024) over-training testbed: 104 models, 3 corpora, N 11M-6.9B, M = D/N in 5..640 (factorial
     within corpus). LLM-foundry ICL evals (HellaSwag 10-shot, ARC-C 10-shot, Winogrande 0-shot).
  E3 Pythia (Biderman et al. 2023): 8 sizes x {standard, deduped} at fixed D = 300B tokens => theta_N only at the
     final checkpoint; within-run checkpoints give a (schedule-confounded) D-elasticity. lm-eval-harness 0/5-shot
     ARC-C (acc_norm) and Winogrande (no HellaSwag in the released evals).
  E4 DataDecide (Magnusson et al. 2025): 25 data recipes x 14 sizes x 3 seeds, final D = 100 N (a single ray),
     so only the ray elasticity theta_C is identified from final checkpoints (recipe fixed effects).

Units: N = total parameters (E1 ladder runs: training FLOP / (6 D); OLMo's FLOP counter charges 2 N_(excl. input
embedding) + 4 N_total + 12 L d_model seq per token, so this 'N' differs from the total parameter count by a few
percent -- it drops a third of the input embedding and adds attention FLOPs [review note]; E1 OLMo-2 targets: HF
floating-point tensor counts; E4 DataDecide: compute / (6 tokens) from the released columns; E2: `params` incl.
embeddings; E3: HF floating-point tensor counts); D = tokens processed.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from common import CORE, PROC, RAW, add_outputs, ols
from panel import hf_param_count

LADDER = os.path.join(RAW, "olmo_ladder")


# ----------------------------------------------------------------------------- data construction
def build_olmo_ladder():
    cols = {"hellaswag": "eval/downstream/hellaswag_val_rc_5shot_len_norm",
            "arc_c": "eval/downstream/arc_challenge_test_rc_5shot_len_norm",
            "winogrande": "eval/downstream/winogrande_val_rc_5shot_len_norm",
            "arc_e": "eval/downstream/arc_easy_test_rc_5shot_len_norm",
            "piqa": "eval/downstream/piqa_val_rc_5shot_len_norm"}
    rows = []
    for f in sorted(glob.glob(os.path.join(LADDER, "*xC*.csv"))):
        d = pd.read_csv(f).dropna(subset=[cols["hellaswag"]])
        last = d.iloc[-1]
        name = os.path.basename(f)[:-4]
        size, mult = re.match(r"(\w+)-([\d.]+)xC", name).groups()
        D = last["throughput/total_tokens"]
        C = last["throughput/total_training_Gflops"] * 1e9
        r = dict(run=name, size=size, mult=float(mult), rerun="rerun" in name, N=C / (6 * D), D=D, C=C, target=False)
        r.update({k: last[v] for k, v in cols.items()})
        rows.append(r)
    # OLMo-2 targets (same recipe family; final stage-1 checkpoint and, for 7B, the annealed model)
    hf = lambda i: hf_param_count(json.load(open(os.path.join(RAW, "hf_meta", i))))  # floating-point tensors only
    for tag, f, N in [("OLMo2-7B", "peteish7_eval_full.csv", hf("allenai_OLMo-2-1124-7B.json")),
                      ("OLMo2-13B", "peteish13_eval_final.csv", hf("allenai_OLMo-2-1124-13B.json"))]:
        d = pd.read_csv(os.path.join(LADDER, f)).dropna(subset=[cols["hellaswag"]])
        last = d.iloc[-1]
        D = last["_step"] * last["batch_size_in_tokens"]
        r = dict(run=tag, size=tag, mult=np.nan, rerun=False, N=float(N), D=float(D), C=6 * N * D, target=True)
        r.update({k: last[v] for k, v in cols.items()})
        rows.append(r)
    df = add_outputs(pd.DataFrame(rows))
    df["n"], df["d"], df["c"] = np.log(df.N), np.log(df.D), np.log(df.C)
    return df


def build_gadre():
    rows = []
    for f in glob.glob(os.path.join(RAW, "gadre", "models", "*.json")):
        m = json.load(open(f))
        hp = m["hyperparameters"]
        name = m["name"]
        # eval files keep the URL-encoded '=' (%3D) of the GitHub download names
        ev = os.path.join(RAW, "gadre", "evals", f"evaluation_{name.replace('=', '%3D')}_heavy.json")
        if not os.path.exists(ev):
            continue
        e = json.load(open(ev))["eval_metrics"]["icl"]
        rows.append(dict(run=name, corpus=m["dataset_name"], N=hp["params"], N_ne=hp["params_no_embed"],
                         D=hp["tokens"], mult=hp["chinchilla_multiplier"], hellaswag=e.get("hellaswag"),
                         arc_c=e.get("arc_challenge"), winogrande=e.get("winogrande"), arc_e=e.get("arc_easy"),
                         piqa=e.get("piqa"), mmlu=e.get("mmlu")))
    df = add_outputs(pd.DataFrame(rows))
    df["N"], df["D"] = df.N.astype(float), df.D.astype(float)   # int64 products overflow above 9.2e18
    df["C"] = 6 * df.N * df.D
    df["n"], df["d"], df["c"] = np.log(df.N), np.log(df.D), np.log(df.C)
    return df


def build_pythia():
    # [review fix] floating-point tensors only (safetensors.total includes the U8 causal-mask buffers of GPT-NeoX
    # checkpoints) and a deterministic choice when standard/deduped counts differ (they differ only by dtype rounding)
    exact = {}
    for f in sorted(glob.glob(os.path.join(RAW, "hf_meta", "EleutherAI_pythia-*.json"))):
        j = json.load(open(f))
        n = hf_param_count(j)
        if np.isfinite(n):
            k = j["id"].split("/")[-1].replace("-deduped", "")
            exact[k] = min(exact.get(k, np.inf), n)
    rows = []
    for mdir in sorted(glob.glob(os.path.join(RAW, "pythia_evals", "pythia-*"))):
        model = os.path.basename(mdir)
        base = model.replace("-deduped", "")
        for shot, sub in [(0, "zero-shot"), (5, "five-shot")]:
            for f in glob.glob(os.path.join(mdir, sub, "*.json")):
                step = int(re.search(r"step(\d+)", f).group(1))
                r = json.load(open(f))["results"]
                rows.append(dict(model=model, dedup=model.endswith("deduped"), shots=shot, step=step,
                                 N=exact.get(base, np.nan), D=step * 2_097_152,
                                 arc_c=r["arc_challenge"]["acc_norm"], winogrande=r["winogrande"]["acc"],
                                 arc_e=r["arc_easy"]["acc_norm"], piqa=r["piqa"]["acc_norm"]))
    df = pd.DataFrame(rows)
    df = df[df.step > 0]
    df = add_outputs(df)
    df["y_core2"] = _core2(df)
    df["N"], df["D"] = df.N.astype(float), df.D.astype(float)
    df["C"] = 6 * df.N * df.D
    df["n"], df["d"], df["c"] = np.log(df.N), np.log(df.D), np.log(df.C)
    return df


def _core2(df):
    """Composite over ARC-C and Winogrande only (Pythia evals lack HellaSwag)."""
    from common import composite
    return composite(df, tasks=["arc_c", "winogrande"])


def build_datadecide(force=False):
    out = os.path.join(PROC, "datadecide_tasks.parquet")
    if os.path.exists(out) and not force:
        df = pd.read_parquet(out)
    else:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        keep = {"hellaswag": "acc_per_char", "arc_challenge": "acc_per_char", "winogrande": "acc_raw",
                "arc_easy": "acc_per_char", "piqa": "acc_per_char"}
        parts = []
        for f in sorted(glob.glob(os.path.join(RAW, "datadecide_eval", "*.parquet"))):
            t = pq.read_table(f, columns=["params", "data", "task", "step", "seed", "tokens", "compute", "metrics"])
            t = t.filter(pc.is_in(t["task"], value_set=__import__("pyarrow").array(list(keep))))
            d = t.to_pandas()
            # `metrics` is a Python-dict repr string; pull the one metric we need with a regex (fast, exact)
            pats = {k: re.compile(r"'" + v + r"': ([-0-9.eE+]+|nan)") for k, v in keep.items()}
            d["acc"] = [float(pats[k].search(m).group(1)) for m, k in zip(d.metrics, d.task)]
            parts.append(d.drop(columns="metrics"))
        long = pd.concat(parts)
        df = long.pivot_table(index=["params", "data", "seed", "step", "tokens", "compute"], columns="task",
                              values="acc").reset_index()
        df = df.rename(columns={"arc_challenge": "arc_c", "arc_easy": "arc_e"})
        df.to_parquet(out)
    df = df[df["tokens"] > 0].copy()
    df = add_outputs(df)
    df["tokens"], df["compute"] = df["tokens"].astype(float), df["compute"].astype(float)
    df["N"] = df["compute"] / (6 * df["tokens"])
    df["D"], df["C"] = df["tokens"], df["compute"]
    df["n"], df["d"], df["c"] = np.log(df.N), np.log(df.D), np.log(df.C)
    fin = df.loc[df.groupby(["params", "data", "seed"]).step.idxmax()].copy()
    fin["final"] = True
    return df, fin


# ----------------------------------------------------------------------------- estimation
def _row(ds, out, par, res, name, sample, n_obs, extra=None):
    r = dict(dataset=ds, output=out, param=par, est=float(res.params[name]), se=float(res.bse[name]),
             n_obs=int(n_obs), sample=sample)
    if extra:
        r.update(extra)
    return r


def _ray(res, a=0.5):
    """theta_C along a ray with share a of ln C going to ln N: theta_C = a theta_N + (1-a) theta_D (delta method)."""
    b = res.params[["n", "d"]].values
    V = res.cov_params().loc[["n", "d"], ["n", "d"]].values
    w = np.array([a, 1 - a])
    return float(w @ b), float(np.sqrt(w @ V @ w))


def estimate_experiments(lad, gad, pyt, dd_all, dd_fin, outputs=("y_hellaswag", "y_arc_c", "y_winogrande", "y_core")):
    import statsmodels.api as sm
    R = []
    supp = {}

    def factorial(ds, df, out, sample, fe=None, cluster=None):
        d = df.dropna(subset=[out, "n", "d"])
        X = d[["n", "d"]].copy()
        if fe is not None:
            X = pd.concat([X, pd.get_dummies(d[fe], drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X)
        res = ols(d[out], X, cluster=None if cluster is None else d[cluster])
        ext = dict(N_min=d.N.min(), N_max=d.N.max(), C_min=d.C.min(), C_max=d.C.max(), corr_nd=np.corrcoef(d.n, d.d)[0, 1])
        R.append(_row(ds, out, "theta_N", res, "n", sample, len(d), ext))
        R.append(_row(ds, out, "theta_D", res, "d", sample, len(d), ext))
        est, se = _ray(res)
        R.append(dict(dataset=ds, output=out, param="theta_C_ray", est=est, se=se, n_obs=len(d), sample=sample, **ext))
        R.append(dict(dataset=ds, output=out, param="R2", est=res.rsquared, se=np.nan, n_obs=len(d), sample=sample, **ext))
        return res

    for out in outputs:
        # E1: OLMo ladder (30 finals) and with the two OLMo-2 targets
        factorial("OLMo ladder", lad[~lad.target], out, "30 final ckpts (190M-3B)")
        factorial("OLMo ladder + OLMo-2", lad, out, "30 finals + 7B/13B targets")
        # E2: Gadre, corpus FE; all sizes and N >= 100M
        factorial("Gadre et al.", gad, out, "104 models, corpus FE", fe="corpus")
        factorial("Gadre et al. (N>=0.1B)", gad[gad.N >= 1e8], out, "N>=0.1B, corpus FE", fe="corpus")
        # E4: DataDecide final checkpoints: ray elasticity with recipe FE, clustered by recipe
        d = dd_fin.dropna(subset=[out])
        d = d[d.N >= 5e7]  # below ~60M every task is at chance in DataDecide
        X = sm.add_constant(pd.concat([d[["c"]], pd.get_dummies(d["data"], drop_first=True, dtype=float)], axis=1))
        res = ols(d[out], X, cluster=d["data"])
        ext = dict(N_min=d.N.min(), N_max=d.N.max(), C_min=d.C.min(), C_max=d.C.max(), corr_nd=1.0)
        R.append(_row("DataDecide", out, "theta_C_ray", res, "c", "finals N>=60M, D=100N, recipe FE", len(d), ext))

    # E3: Pythia -- theta_N at fixed D (final checkpoints) and within-run D-elasticity from checkpoints
    for out in ["y_arc_c", "y_winogrande", "y_core2"]:
        for shots in (0, 5):
            p = pyt[(pyt.shots == shots)]
            fin = p[p.step == 143000].dropna(subset=[out, "n"])
            X = sm.add_constant(pd.concat([fin[["n"]], fin[["dedup"]].astype(float)], axis=1))
            res = ols(fin[out], X)
            ext = dict(N_min=fin.N.min(), N_max=fin.N.max(), C_min=fin.C.min(), C_max=fin.C.max(), corr_nd=np.nan)
            R.append(_row(f"Pythia ({shots}-shot)", out, "theta_N", res, "n", "16 finals, D=300B", len(fin), ext))
            ck = p[(p.step >= 13000)].dropna(subset=[out, "d"])
            X = sm.add_constant(pd.concat([ck[["d"]], pd.get_dummies(ck["model"], drop_first=True, dtype=float)], axis=1))
            res = ols(ck[out], X, cluster=ck["model"])
            ext = dict(N_min=ck.N.min(), N_max=ck.N.max(), C_min=ck.C.min(), C_max=ck.C.max(), corr_nd=np.nan)
            R.append(_row(f"Pythia ({shots}-shot)", out, "theta_D_ckpt", res, "d", "ckpts >= 27B tokens, model FE", len(ck), ext))
    est = pd.DataFrame(R)
    return est


def curvature_check(lad, gad):
    """Is the logit-linear (Cobb-Douglas-in-odds) form adequate over the experimental range? Adds squares and
    the interaction; reports the F-test p-value and the local elasticities at the sample means."""
    import statsmodels.api as sm
    rows = []
    for ds, df, fe in [("OLMo ladder + OLMo-2", lad, None), ("Gadre et al. (N>=0.1B)", gad[gad.N >= 1e8], "corpus")]:
        for out in ["y_hellaswag", "y_arc_c", "y_winogrande", "y_core"]:
            d = df.dropna(subset=[out]).copy()
            d["n2"], d["d2"], d["nd"] = (d.n - d.n.mean()) ** 2, (d.d - d.d.mean()) ** 2, (d.n - d.n.mean()) * (d.d - d.d.mean())
            X0 = d[["n", "d"]]
            if fe:
                X0 = pd.concat([X0, pd.get_dummies(d[fe], drop_first=True, dtype=float)], axis=1)
            X1 = pd.concat([X0, d[["n2", "d2", "nd"]]], axis=1)
            r0 = sm.OLS(d[out], sm.add_constant(X0)).fit()
            r1 = sm.OLS(d[out], sm.add_constant(X1)).fit()
            F = ((r0.ssr - r1.ssr) / 3) / (r1.ssr / r1.df_resid)
            from scipy.stats import f as fdist
            rows.append(dict(dataset=ds, output=out, F=F, p=1 - fdist.cdf(F, 3, r1.df_resid), n2=r1.params["n2"],
                             d2=r1.params["d2"], nd=r1.params["nd"], R2_lin=r0.rsquared, R2_quad=r1.rsquared))
    return pd.DataFrame(rows)


def build_all(verbose=True):
    lad, gad, pyt = build_olmo_ladder(), build_gadre(), build_pythia()
    dd_all, dd_fin = build_datadecide()
    lad.to_csv(os.path.join(PROC, "exp_olmo_ladder.csv"), index=False)
    gad.to_csv(os.path.join(PROC, "exp_gadre.csv"), index=False)
    pyt.to_csv(os.path.join(PROC, "exp_pythia.csv"), index=False)
    dd_fin.to_csv(os.path.join(PROC, "exp_datadecide_final.csv"), index=False)
    if verbose:
        print(f"[exp] OLMo ladder {len(lad)} runs; Gadre {len(gad)}; Pythia {len(pyt)} (model x ckpt x shots); "
              f"DataDecide {len(dd_all)} ckpts, {len(dd_fin)} finals")
    return lad, gad, pyt, dd_all, dd_fin


if __name__ == "__main__":
    lad, gad, pyt, dd_all, dd_fin = build_all()
    est = estimate_experiments(lad, gad, pyt, dd_all, dd_fin)
    pd.set_option("display.width", 220)
    print(est[est.param != "R2"].round(3).to_string())
    print(curvature_check(lad, gad).round(3).to_string())
