"""run.py -- module m3_wedge: over-training reveals anticipated inference demand (headline H2).

Regenerates every output of the module from raw data (data/raw) and the technology registries/bootstrap draws of
modules m1_chinchilla and m2_techpanel. Deterministic (no random draws are taken here: bands use the upstream
bootstrap draws). CPU only; runs in about 10 seconds.

  python code/analysis/m3_wedge/run.py            # all stages
Prerequisite for a fresh clone: bash code/data/download_m3_wedge.sh (Hugging Face metadata and model cards,
OpenRouter endpoints, LMArena leaderboard); the other raw files come from code/data/download_public.sh.

Stages
  1. build_choices : Sample A (Epoch universe) and Sample B (verified open-weight) -> data/processed/m3_wedge
  2. wedge         : w, T/D, T, CE per model x technology; partial-identification bands; homothetic grid; Farseer Eq. 3
  3. analysis      : within-family (GNR) revealed preference + over-identification; trends; usage validation;
                     rival wedges; aggregate lifetime-inference multiple; stated-intent checks
  4. outputs       : tables (output/tables/m3_wedge_*.csv|.tex), figures (output/figures/m3_wedge_*.pdf|.png),
                     headline numbers (data/processed/m3_wedge/headline.json)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROC, log  # noqa: E402
import analysis as AN  # noqa: E402
import build_choices  # noqa: E402
import figures as FG  # noqa: E402
import tables as TB  # noqa: E402
import wedge as WG  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    t0 = time.time()
    open(os.path.join(PROC, "run_log.txt"), "w").close()
    # 1. data
    A, B, X, ch = build_choices.main()
    Bdrop = pd.read_csv(os.path.join(PROC, "sampleB_dropped.csv"))
    ch = pd.read_csv(os.path.join(PROC, "choices.csv"))
    # 2. wedges
    techs, long, bd, homo, eq3, mstar_f = WG.main(ch)
    # 3. analysis
    log("analysis: assemble")
    d = AN.assemble(ch, long, bd, eq3)
    d.to_csv(os.path.join(PROC, "wedge_models_wide.csv"), index=False)
    log("analysis: within-family revealed preference")
    F, Sfam = AN.family_gnr(d, techs)
    over = AN.overid_tests(F, d)
    log("analysis: trends")
    Tr, labs, regs, U = AN.trends(d)
    U.attrs["robust"].to_csv(os.path.join(TB.TABLES, "m3_wedge_trends_robust.csv"), index=False)
    log("analysis: usage validation")
    V, Bv = AN.validation(d)
    Bv.to_csv(os.path.join(PROC, "validation_sample.csv"), index=False)
    log("analysis: rival wedges")
    R = AN.rivals(d, techs, mstar_f)
    log("analysis: aggregate")
    Ag, Ab = AN.aggregate(d, techs, long)
    stated = AN.stated_intent(d)
    # 4. outputs
    log("outputs: tables")
    TB.tech_table(WG.T.tech_table(techs))
    T5 = TB.table5(d, F)
    TB.family_tables(F, Sfam, over)
    TB.trends_tables(Tr, labs, regs)
    TB.validation_table(V)
    TB.aggregate_table(Ag, Ab)
    TB.rivals_table(R, over, stated)
    TB.homothetic_tex(d, homo)
    Bfull = pd.read_csv(os.path.join(PROC, "sampleB_verified.csv"))
    TB.data_audit(A, Bfull, X, Bdrop)
    # per-model result files (all technologies)
    keep = ["uid", "model", "hf", "family", "lab", "org", "date", "year", "sample", "core", "open_weights", "arch", "moe",
            "N_total", "N_active", "N", "N_nonemb", "D", "D_src", "M", "Cmp", "distilled", "synthetic", "code"]
    wcols = [c for c in d.columns if c.startswith(("w_", "wlo_", "whi_", "CE_", "Mx_"))]
    out = d[keep + wcols + ["band_lo", "band_hi", "band_lo_nohoff", "band_hi_nohoff", "band_lo_core", "band_hi_core", "techs_used", "w_eq3", "Mstar_eq3_ownC"]]
    out.to_csv(os.path.join(TB.TABLES, "m3_wedge_models.csv"), index=False)
    long.to_csv(os.path.join(TB.TABLES, "m3_wedge_models_long.csv"), index=False)
    log("outputs: figures")
    FG.fig4(d, F, techs)
    FG.fig_trends(Tr, d)
    FG.fig_family(F)
    FG.fig_validation(Bv)
    FG.fig_rivals(R, d)
    FG.fig_aggregate(Ab)
    # summaries quoted in the memo: wedges of Sample B by technology; flagships before / after 2024
    Bc0 = d[(d["sample"] == "B") & d["core"]]
    Lc = long[long["uid"].isin(Bc0["uid"])]
    rows = []
    for k in techs:
        # [Review m3] shares over the models for which the technology is defined (the builder divided by all 173,
        # counting undefined rows as 'no', which understated the Farseer non-embedding and OLMo shares)
        ok = Bc0[f"w_{k}"].notna()
        B_ = Bc0[ok]
        w = B_[f"w_{k}"]
        lk = Lc[(Lc["tech"] == k) & Lc["w"].notna()]
        rows.append(dict(tech=k, label=techs[k].label, n=int(ok.sum()), median_w=w.median(), share_w_gt1=(w > 1).mean(),
                         share_lo_gt1=(B_[f"wlo_{k}"] > 1).mean() if techs[k].draws is not None else np.nan,
                         share_Mx=B_[f"Mx_{k}"].astype(bool).mean(), share_Cx=lk["C_extrap"].mean(),
                         median_lo_ratio=(B_[f"wlo_{k}"] / w).median(), median_hi_ratio=(B_[f"whi_{k}"] / w).median()))
    rows.append(dict(tech="PI band", n=len(Bc0), median_w=np.nan, share_lo_gt1=(Bc0["band_lo"] > 1).mean(),
                     median_lo_ratio=Bc0["band_lo"].median(), median_hi_ratio=Bc0["band_hi"].median(), label="band_lo / band_hi medians"))
    pd.DataFrame(rows).to_csv(os.path.join(TB.TABLES, "m3_wedge_sampleB_by_tech.csv"), index=False)
    fl = F[F["flagship"]].merge(d[["uid", "year", "wlo_chin", "whi_chin", "band_lo", "band_hi"]], on="uid")
    fl["period"] = np.where(fl["year"] >= 2024, "2024+", "pre-2024")
    fl.groupby("period").apply(lambda g: pd.Series(dict(
        n=len(g), median_w=g["w_abs"].median(), share_ci_contains1=((g["wlo_chin"] <= 1) & (g["whi_chin"] >= 1)).mean(),
        share_ci_above1=(g["wlo_chin"] > 1).mean(), share_band_above1=(g["band_lo"] > 1).mean()))).reset_index().to_csv(
        os.path.join(TB.TABLES, "m3_wedge_flagships_by_period.csv"), index=False)
    # headline numbers (the memo quotes these)
    Bc = d[(d["sample"] == "B") & d["core"]]
    def row(model):
        r = d[d["model"] == model].iloc[0]
        return dict(M=r["M"], w=r["w_chin"], w_lo=r["wlo_chin"], w_hi=r["whi_chin"], TD=3 * (r["w_chin"] - 1),
                    T=3 * r["D"] * (r["w_chin"] - 1), CE=r["CE_chin"], band=[r["band_lo"], r["band_hi"]],
                    w_meta=r["w_meta_a2"], w_meta_a3=r["w_meta_a3"], w_hoff=r["w_hoff"], w_farseer=r["w_farseer"],
                    w_farseer_emb=r["w_farseer_emb"], w_gadre=r["w_gadre_rw"], w_besi=r["w_besi"], w_olmo=r.get("w_olmo"),
                    w_eq3=r["w_eq3"], CE_meta=r["CE_meta_a2"])
    H = dict(
        n_sampleA_kept=int(len(A)), n_sampleA_core=int(A["core"].sum()), n_excluded=int(len(X)), n_sampleB=int(len(Bfull)),
        n_sampleB_core=int(len(Bc)), n_families_B=int(Bfull["family"].nunique()),
        n_sampleB_below_1e21=int((Bc["Cmp"] < AN.PROD_C).sum()),
        n_prod_union=int((d["prod"] & d["year"].between(2019, 2026)).sum()),
        n_prod_open=int((d["prod"] & d["open_weights"] & d["year"].between(2019, 2026)).sum()),
        n_prod_clusters=int(d.loc[d["prod"] & d["year"].between(2019, 2026), "dev"].nunique()),
        n_prod_D_C_mismatch=int((d["prod"] & d["D_C_mismatch"]).sum()),
        sampleB_share_Cx_chin=float(long[(long["tech"] == "chin") & long["uid"].isin(Bc["uid"])]["C_extrap"].mean()),
        sampleB_median_w=float(Bc["w_chin"].median()), sampleB_share_w_gt1=float((Bc["w_chin"] > 1).mean()),
        sampleB_share_bandlo_gt1=float((Bc["band_lo"] > 1).mean()), sampleB_share_w_gt2=float((Bc["w_chin"] > 2).mean()),
        sampleB_share_Mx_chin=float(Bc["Mx_chin"].mean()), sampleB_share_Mx_farseer=float(Bc["Mx_farseer_emb"].mean()),
        sampleB_share_Mx_farseer_nonemb=float(Bc.loc[Bc["w_farseer"].notna(), "Mx_farseer"].astype(bool).mean()),
        models={m: row(m) for m in ["Meta-Llama-3-8B", "Meta-Llama-3-70B", "Llama-3.1-405B", "Llama-2-7b-hf", "Llama-2-70b-hf",
                                    "Qwen3-0.6B-Base", "Qwen2.5-72B", "gemma-3-27b-pt", "gemma-3-1b-pt", "DeepSeek-V3-Base",
                                    "SmolLM2-135M", "OLMo-2-0325-32B", "OLMo-2-1124-7B", "Qwen1.5-0.5B", "pythia-12b"]},
        chinchilla=dict(w=float(d.loc[d["uid"] == "A:Chinchilla", "w_chin"].iloc[0]),
                        w_hoff=float(d.loc[d["uid"] == "A:Chinchilla", "w_hoff"].iloc[0])),
        overid={k: v for k, v in over.items() if k != "siblings_wrel_lt1"},
        validation=V.to_dict("records"), aggregate=Ab.to_dict("records"),
        rivals=dict(tier=R["tier_reg"], distill=R["distill_reg"], factor_bias=R["factor_bias"]),
        runtime_s=time.time() - t0)
    with open(os.path.join(PROC, "headline.json"), "w") as f:
        json.dump(H, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
