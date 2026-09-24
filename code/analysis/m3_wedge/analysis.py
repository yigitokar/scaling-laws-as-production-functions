"""analysis.py -- within-family revealed preference (GNR direction), trends, usage validation, rival wedges and the
aggregate lifetime-inference multiple.  All inputs come from build_choices.py and wedge.py (data/processed/m3_wedge).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import HFRAW, PROC, RAW, ROOT, SEED, TABLES, ce_of_w, log, log_w
import curated as cu
import technologies as T

REF = "chin"   # reference technology: Chinchilla refit on the Besiroglu sample (m1 Huber, n = 240)
PROD_C = 1e21  # production-scale threshold on 6ND (FLOP), applied to BOTH samples in the trend/aggregate universe

# [Review m3] Developer clusters. Sample A carries Epoch's 'Organization' string and Sample B a curated lab name, so
# the same developer appeared under several labels ('Meta' vs 'Meta AI', 'Cerebras' vs 'Cerebras Systems', Google's
# units, ...), which inflated the number of clusters and split the lab table. Map to the parent developer (Alphabet
# units -> Google; conservative for clustering: fewer, larger clusters).
DEV_MAP = {"Meta AI": "Meta", "Facebook AI Research": "Meta", "Facebook": "Meta", "Facebook AI": "Meta",
           "Google DeepMind": "Google", "DeepMind": "Google", "Google Brain": "Google", "Google Research": "Google",
           "Microsoft Research": "Microsoft", "Microsoft Research Asia": "Microsoft", "IBM Research": "IBM",
           "Cerebras Systems": "Cerebras", "Technology Innovation Institute": "TII", "RWKV Foundation": "RWKV",
           "Huawei Noah's Ark Lab": "Huawei", "Z.ai (Zhipu AI)": "Zhipu", "Zhipu AI": "Zhipu",
           "Alibaba Group": "Alibaba", "Allen Institute for AI": "AI2", "NVIDIA": "Nvidia", "Mistral": "Mistral AI"}


def developer(df):
    """Canonical developer label (first listed organization, mapped to the parent developer)."""
    s = df["lab"].where(df["lab"].notna() & (df["lab"].astype(str) != ""), df.get("org"))
    s = s.fillna("other").astype(str).str.split(",").str[0].str.strip()
    return s.replace(DEV_MAP)


# ----------------------------------------------------------------------------- assemble
def assemble(ch, long, bd, eq3):
    W = long.pivot(index="uid", columns="tech", values="w").add_prefix("w_")
    Wlo = long.pivot(index="uid", columns="tech", values="w_lo").add_prefix("wlo_")
    Whi = long.pivot(index="uid", columns="tech", values="w_hi").add_prefix("whi_")
    Mx = long.pivot(index="uid", columns="tech", values="M_extrap").add_prefix("Mx_")
    CE = long.pivot(index="uid", columns="tech", values="CE").add_prefix("CE_")
    d = ch.merge(W, left_on="uid", right_index=True).merge(Wlo, left_on="uid", right_index=True) \
        .merge(Whi, left_on="uid", right_index=True).merge(Mx, left_on="uid", right_index=True) \
        .merge(CE, left_on="uid", right_index=True).merge(bd, on="uid").merge(eq3, on="uid", how="left")
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    d["Cmp"] = 6 * d["N"] * d["D"]
    d["lnw"] = np.log(d[f"w_{REF}"])
    d["TD"] = 3 * (d[f"w_{REF}"] - 1)
    d["T"] = d["TD"] * d["D"]
    # analysis universe: B rows (general-purpose) + core Epoch rows not matched to B
    d["core"] = d["core"].fillna(False).astype(bool) & ~d["code"].fillna(False).astype(bool)
    # [Review m3] production-scale universe for trends / rivals / aggregate: the Sample-A core already requires
    # 6ND >= 1e21, but Sample B did not (Pythia-70M, Cerebras-GPT-111M, ... entered the "production-scale" trends).
    # Apply the same threshold to both samples there; Sample-B-only analyses keep all general-purpose B models.
    d["prod"] = d["core"] & (d["Cmp"] >= PROD_C)
    d["D_C_mismatch"] = d["D_C_mismatch"].fillna(False).astype(bool) if "D_C_mismatch" in d else False
    d["dev"] = developer(d)
    d["open_weights"] = d["open_weights"].fillna(False).astype(bool)
    d["size_class"] = pd.cut(d["N"], [0, 4e9, 9e9, 16e9, 40e9, 80e9, np.inf],
                             labels=["<4B", "4-9B", "9-16B", "16-40B", "40-80B", ">80B"])
    return d


# ----------------------------------------------------------------------------- (3) within-family revealed preference
MIN_FAMILY = 2
D_VARIES_TOL = 1.10


def family_gnr(d, techs):
    """GNR direction: within a family trained on the same data/recipe, assume the largest model (flagship) is on its
    own training-optimal path (w_f = 1). With the Chinchilla form the family-specific A/B cancels and each sibling's
    wedge relative to the flagship depends only on the exponents:
        ln w_i^rel = -alpha ln(N_i/N_f) + beta ln(D_i/D_f),  T_i = 3 D_i (w_i^rel - 1).
    If the flagship itself serves inference (w_f > 1) these are lower bounds: w_i = w_f w_i^rel."""
    B = d[(d["sample"] == "B") & d["core"] & ~d["family"].isna()].copy()
    rows = []
    for fam, g in B.groupby("family"):
        if len(g) < MIN_FAMILY:
            continue
        # flagship = largest model by total parameters (MoE: total; ties broken by training compute)
        f = g.sort_values(["N_total", "Cmp"], ascending=False).iloc[0]
        fb = d.loc[d["uid"] == f["uid"]].iloc[0]
        # [Review m3] "D varies" needs real variation: Llama 3.1 (15T vs 15.6T for the 405B) and BLOOM (341B vs 366B)
        # were flagged by nunique() > 1 although D is common up to 4-7%. Require max/min > 1.10.
        d_varies = bool(g["D"].max() / g["D"].min() > D_VARIES_TOL)
        for _, r in g.iterrows():
            rec = dict(family=fam, uid=r["uid"], model=r["model"], lab=r["lab"], N=r["N"], D=r["D"], M=r["M"],
                       flagship=r["uid"] == f["uid"], flag_model=f["model"], N_f=f["N"], D_f=f["D"],
                       w_abs=r[f"w_{REF}"], w_flag_abs=f[f"w_{REF}"], D_varies=d_varies, n_fam=len(g),
                       flag_band_has1=bool(fb["band_lo"] <= 1 <= fb["band_hi"]))
            for k in ["chin", "farseer_emb", "gadre_rw", "hoff", "meta_a2"]:
                t = techs[k]
                lw = -t.alpha * np.log(r["N"] / f["N"]) + t.beta * np.log(r["D"] / f["D"])
                rec[f"wrel_{k}"] = np.exp(lw)
                if t.draws is not None:
                    LW = -t.draws[:, 0] * np.log(r["N"] / f["N"]) + t.draws[:, 1] * np.log(r["D"] / f["D"])
                    rec[f"wrel_{k}_lo"], rec[f"wrel_{k}_hi"] = np.exp(np.percentile(LW, [2.5, 97.5]))
            rec["TDrel"] = 3 * (rec[f"wrel_{REF}"] - 1)
            rec["Trel"] = rec["TDrel"] * r["D"]
            rows.append(rec)
    F = pd.DataFrame(rows)
    # family summary: flagship wedge under absolute technologies, implied family M*_f vs technology M*(C_f)
    S = []
    for fam, g in F.groupby("family"):
        f = g[g["flagship"]].iloc[0]
        sib = g[~g["flagship"]]
        S.append(dict(family=fam, lab=f["lab"], n=len(g), flagship=f["model"], N_f=f["N"], D_f=f["D"], M_f=f["M"],
                      w_flag_chin=f["w_abs"], w_flag_meta_a2=d.loc[d["uid"] == f["uid"], "w_meta_a2"].iloc[0],
                      D_varies=bool(f["D_varies"]), flag_band_has1=bool(f["flag_band_has1"]),
                      flag_band_lo=d.loc[d["uid"] == f["uid"], "band_lo"].iloc[0],
                      flag_band_hi=d.loc[d["uid"] == f["uid"], "band_hi"].iloc[0],
                      share_sib_wrel_gt1=float((sib[f"wrel_{REF}"] > 1).mean()) if len(sib) else np.nan,
                      spearman_Trel_N=stats.spearmanr(sib["N"], sib["Trel"]).statistic if len(sib) >= 3 else np.nan,
                      max_TDrel=sib["TDrel"].max() if len(sib) else np.nan))
    S = pd.DataFrame(S)
    return F, S


def overid_tests(F, d):
    """Raval-style over-identification. Demand model: planned lifetime tokens T_i = kappa_f N_i^(-eta) (smaller
    siblings are served more). (i) sign: every non-flagship sibling should have w_rel > 1; (ii) a common eta across
    families (F-test of slope heterogeneity) using the absolute-technology T_i (all members with T_i > 0) and the
    flagship-normalised T_i^rel (siblings only). Informative only where D varies within the family: with a common D,
    T_i/D is a mechanical function of N_i."""
    out = {}
    sib = F[~F["flagship"]]
    out["n_siblings"] = int(len(sib))
    out["share_siblings_wrel_gt1"] = float((sib[f"wrel_{REF}"] > 1).mean())
    out["siblings_wrel_lt1"] = sib.loc[sib[f"wrel_{REF}"] <= 1, ["family", "model", f"wrel_{REF}"]].to_dict("records")
    res = []
    for label, df, ycol in [("absolute (Chinchilla refit), all members", F.assign(Tabs=3 * F["D"] * (F["w_abs"] - 1)), "Tabs"),
                            ("flagship-normalised, siblings", sib, "Trel")]:
        for scope in ["all families", "families with size-varying D"]:
            x = df[(df[ycol] > 0)].copy()
            if scope != "all families":
                x = x[x["D_varies"]]
            x = x[x.groupby("family")["N"].transform("nunique") >= 2]   # slope needs >= 2 distinct sizes
            if x["family"].nunique() < 2 or len(x) < 6:
                continue
            x["lnT"], x["lnN"] = np.log(x[ycol]), np.log(x["N"])
            m0 = smf.ols("lnT ~ C(family) + lnN", data=x).fit(cov_type="cluster", cov_kwds=dict(groups=x["family"]))
            m1 = smf.ols("lnT ~ C(family) + C(family):lnN", data=x).fit()
            m0h = smf.ols("lnT ~ C(family) + lnN", data=x).fit()
            Fst, p, dfd = m1.compare_f_test(m0h)
            res.append(dict(spec=label, scope=scope, n=len(x), families=x["family"].nunique(),
                            eta=-m0.params["lnN"], se_eta=m0.bse["lnN"], F_homog=Fst, p_homog=p, df_diff=dfd,
                            family_slopes=json.dumps({f: round(-v, 3) for f, v in m1.params.items() if ":lnN" in f})))
    out["demand_tests"] = res
    return out


# ----------------------------------------------------------------------------- (4) trends and heterogeneity
def trends(d):
    U = d[d["prod"] & d["year"].between(2019, 2026)].copy()
    rows = []
    for (grp, key), g in [((c, v), gg) for c in ["year"] for v, gg in U.groupby("year")] + \
                         [(("open", v), gg) for v, gg in U.groupby("open_weights")] + \
                         [(("moe", v), gg) for v, gg in U.groupby("moe")] + \
                         [(("size", v), gg) for v, gg in U.groupby("size_class", observed=True)]:
        w = g[f"w_{REF}"]
        rows.append(dict(group=grp, value=str(key), n=len(g), median_w=w.median(), q25_w=w.quantile(0.25),
                         q75_w=w.quantile(0.75), share_w_lt1=(w < 1).mean(), share_w_gt2=(w > 2).mean(),
                         median_M=g["M"].median(), median_TD=(3 * (w - 1)).median(),
                         median_w_band_lo=g["band_lo"].median(), median_w_band_hi=g["band_hi"].median(),
                         median_N=g["N"].median(), median_D=g["D"].median()))
    # year x openness
    for (y, o), g in U.groupby(["year", "open_weights"]):
        w = g[f"w_{REF}"]
        rows.append(dict(group="year_x_open", value=f"{y}|{'open' if o else 'closed'}", n=len(g), median_w=w.median(),
                         q25_w=w.quantile(0.25), q75_w=w.quantile(0.75), share_w_lt1=(w < 1).mean(),
                         share_w_gt2=(w > 2).mean(), median_M=g["M"].median(), median_TD=(3 * (w - 1)).median()))
    Tr = pd.DataFrame(rows)
    # labs with >= 4 core models (canonical developer labels)
    L = U.assign(lab2=U["dev"])
    labs = L.groupby("lab2").filter(lambda g: len(g) >= 4).groupby("lab2").agg(
        n=("uid", "size"), median_w=(f"w_{REF}", "median"), median_M=("M", "median"),
        first=("year", "min"), last=("year", "max")).reset_index().sort_values("median_w")
    # regressions: ln w on year FE, openness, MoE, (ln C): conditional on C, ln w is proportional to ln(M/M*(C))
    U = U.assign(lnC=np.log(U["Cmp"]), open=U["open_weights"].astype(int), moe_=U["moe"].astype(int),
                 lab2=L["lab2"], yr=U["year"].astype(str))
    regs = []
    for sample, UU in [("main", U), ("drop Sample-A rows with |ln C/(6ND)| > ln 1.5", U[~U["D_C_mismatch"]])]:
        for f in ["lnw ~ C(yr) + open + moe_", "lnw ~ C(yr) + open + moe_ + lnC"]:
            m = smf.ols(f, data=UU).fit(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(UU["lab2"])[0]))
            for k in m.params.index:
                regs.append(dict(sample=sample, formula=f, term=k, coef=m.params[k], se=m.bse[k], p=m.pvalues[k],
                                 n=int(m.nobs), clusters=UU["lab2"].nunique(), r2=m.rsquared))
    # [Review m3] robustness of the by-year trend to Sample-A rows whose D disagrees with Epoch's own compute
    rob = []
    for (y, lab), g in [((y, "main"), gg) for y, gg in U.groupby("year")] + \
                       [((y, "drop D/C mismatch"), gg) for y, gg in U[~U["D_C_mismatch"]].groupby("year")]:
        w = g[f"w_{REF}"]
        rob.append(dict(year=y, sample=lab, n=len(g), median_w=w.median(), share_w_lt1=(w < 1).mean(),
                        share_w_gt2=(w > 2).mean(), median_M=g["M"].median()))
    U.attrs["robust"] = pd.DataFrame(rob)
    return Tr, labs, pd.DataFrame(regs), U


# ----------------------------------------------------------------------------- (5) usage validation
ARENA = {  # Sample-B base id -> LMArena model names of its official post-trained release(s)
    "meta-llama/Llama-2-7b-hf": ["llama-2-7b-chat"], "meta-llama/Llama-2-13b-hf": ["llama-2-13b-chat"],
    "meta-llama/Llama-2-70b-hf": ["llama-2-70b-chat"], "meta-llama/Meta-Llama-3-8B": ["llama-3-8b-instruct"],
    "meta-llama/Meta-Llama-3-70B": ["llama-3-70b-instruct"], "meta-llama/Llama-3.1-8B": ["llama-3.1-8b-instruct"],
    "meta-llama/Llama-3.1-70B": ["llama-3.1-70b-instruct"],
    "meta-llama/Llama-3.1-405B": ["llama-3.1-405b-instruct", "llama-3.1-405b-instruct-fp8", "llama-3.1-405b-instruct-bf16"],
    "meta-llama/Llama-3.2-1B": ["llama-3.2-1b-instruct"], "meta-llama/Llama-3.2-3B": ["llama-3.2-3b-instruct"],
    "meta-llama/Llama-4-Scout-17B-16E": ["llama-4-scout-17b-16e-instruct"],
    "meta-llama/Llama-4-Maverick-17B-128E": ["llama-4-maverick-17b-128e-instruct"],
    "google/gemma-2b": ["gemma-2b-it", "gemma-1.1-2b-it"], "google/gemma-7b": ["gemma-7b-it", "gemma-1.1-7b-it"],
    "google/gemma-2-2b": ["gemma-2-2b-it"], "google/gemma-2-9b": ["gemma-2-9b-it"], "google/gemma-2-27b": ["gemma-2-27b-it"],
    "google/gemma-3-1b-pt": ["gemma-3-1b-it"], "google/gemma-3-4b-pt": ["gemma-3-4b-it"],
    "google/gemma-3-12b-pt": ["gemma-3-12b-it"], "google/gemma-3-27b-pt": ["gemma-3-27b-it"],
    **{f"Qwen/Qwen1.5-{s}": [f"qwen1.5-{s.lower()}-chat"] for s in ["0.5B", "1.8B", "4B", "7B", "14B", "32B", "72B"]},
    "Qwen/Qwen2-72B": ["qwen2-72b-instruct"], "Qwen/Qwen2.5-72B": ["qwen2.5-72b-instruct"],
    "Qwen/Qwen2.5-7B": ["qwen2.5-7b-instruct"], "Qwen/Qwen2.5-32B": ["qwen2.5-32b-instruct"],
    "Qwen/Qwen3-32B": ["qwen3-32b"], "Qwen/Qwen3-30B-A3B-Base": ["qwen3-30b-a3b"], "Qwen/Qwen3-235B-A22B": ["qwen3-235b-a22b", "qwen3-235b-a22b-no-thinking"],
    "Qwen/Qwen3-8B-Base": ["qwen3-8b"], "Qwen/Qwen3-4B-Base": ["qwen3-4b"],
    "microsoft/Phi-3-mini-4k-instruct": ["phi-3-mini-4k-instruct", "phi-3-mini-4k-instruct-june-2024"],
    "microsoft/Phi-3-small-8k-instruct": ["phi-3-small-8k-instruct"], "microsoft/Phi-3-medium-4k-instruct": ["phi-3-medium-4k-instruct"],
    "microsoft/phi-4": ["phi-4"], "microsoft/Phi-3.5-mini-instruct": ["phi-3.5-mini-instruct"],
    "01-ai/Yi-34B": ["yi-34b-chat"], "01-ai/Yi-1.5-34B": ["yi-1.5-34b-chat"], "deepseek-ai/DeepSeek-V3-Base": ["deepseek-v3"],
    "deepseek-ai/DeepSeek-V2": ["deepseek-v2-api-0628"], "deepseek-ai/deepseek-llm-67b-base": ["deepseek-llm-67b-chat"],
    "moonshotai/Kimi-K2-Base": ["kimi-k2-0711-preview"], "zai-org/GLM-4.5-Base": ["glm-4.5"],
    "allenai/OLMo-2-0325-32B": ["olmo-2-0325-32b-instruct"], "allenai/OLMo-7B-hf": ["olmo-7b-instruct"],
    "tiiuae/falcon-180B": ["falcon-180b-chat"], "mosaicml/mpt-7b": ["mpt-7b-chat"], "mosaicml/mpt-30b": ["mpt-30b-chat"],
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": ["RWKV-4-Raven-14B"][:0],
    "HuggingFaceTB/SmolLM2-1.7B": ["smollm2-1.7b-instruct"], "ibm-granite/granite-3.0-8b-base": ["granite-3.0-8b-instruct"],
    "ibm-granite/granite-3.0-2b-base": ["granite-3.0-2b-instruct"],
}


def usage_data(d):
    B = d[(d["sample"] == "B") & d["core"]].copy()
    # OpenRouter: listed if the base id or an official instruct id appears as hugging_face_id in the 2026-09-23 snapshot
    orr = json.load(open(os.path.join(RAW, "openrouter", "models_2026-09-23.json")))["data"]
    hf2slug = {}
    for m in orr:
        h = (m.get("hugging_face_id") or "").lower()
        if h and not m["id"].endswith((":free", ":batch")):
            hf2slug.setdefault(h, []).append(m["id"])
    prov, price, listed = [], [], []
    for h in B["hf"]:
        ids = [h] + cu.INSTRUCT.get(h, []) + ([k for k, v in cu.CANONICAL.items() if v == h])
        ids += [cu.OR_ALIAS[i] for i in ids if i in cu.OR_ALIAS]
        slugs = sorted({s for i in ids for s in hf2slug.get(i.lower(), [])})
        # [Review m3] distinct providers across all of the model's listed releases (the table labels this "number of
        # distinct providers"; the builder summed per-release provider counts, double counting providers)
        provs, pmin = set(), np.nan
        for s in slugs:
            p = os.path.join(RAW, "m3_openrouter", "endpoints", s.replace("/", "__") + ".json")
            if os.path.exists(p):
                e = json.load(open(p)).get("data", {}).get("endpoints", [])
                provs |= {x.get("provider_name") for x in e}
                pr = [float(x["pricing"]["completion"]) for x in e if x.get("pricing") and float(x["pricing"]["completion"]) > 0]
                if pr:
                    pmin = np.nanmin([pmin, min(pr)]) if np.isfinite(pmin) else min(pr)
        listed.append(len(slugs) > 0), prov.append(len(provs)), price.append(pmin)
    B["or_listed"], B["or_providers"], B["or_min_price"] = listed, prov, price
    # LMArena (text arena, all published leaderboards): max cumulative vote count over the model's history
    ar = pd.read_parquet(os.path.join(RAW, "m3_lmarena", "text_full.parquet"), columns=["model_name", "category", "vote_count"])
    votes = ar[ar["category"] == "overall"].groupby("model_name")["vote_count"].max()
    B["arena_votes"] = [sum(votes.get(n, 0) for n in ARENA.get(h, [])) if ARENA.get(h) else np.nan for h in B["hf"]]
    B["arena_matched"] = [bool(ARENA.get(h)) and any(n in votes.index for n in ARENA.get(h, [])) for h in B["hf"]]
    B.loc[~B["arena_matched"], "arena_votes"] = np.nan
    return B


def validation(d):
    """Does over-training predict usage? w is a function of (N, D), so the informative comparison holds compute and
    release date fixed: at fixed C, a more over-trained model (higher M) has HIGHER loss than the compute-optimal one
    but is cheaper to serve, so a positive partial correlation with usage cannot come from quality."""
    B = usage_data(d)
    B = B[B["dl_all"].notna() | B["or_listed"]].copy()
    ref = pd.Timestamp("2026-09-23")
    B["age_m"] = ((ref - B["date"]).dt.days / 30.44).clip(lower=1)
    B = B.assign(lnM=np.log(B["M"]), lnC=np.log(B["Cmp"]), lnN=np.log(B["N"]), lnD=np.log(B["D"]),
                 ln_dlall=np.log1p(B["dlall_total"]), ln_dl30=np.log1p(B["dl30_total"]),
                 ln_likes=np.log1p(B["likes"]), ln_age=np.log(B["age_m"]), yr=B["year"].astype(str),
                 or_listed_i=B["or_listed"].astype(int), ln_prov=np.log1p(B["or_providers"]),
                 ln_votes=np.log1p(B["arena_votes"]), lab2=B["lab"].fillna("other"))
    specs = [
        ("ln_dlall", "all-time downloads (base + official instruct)"),
        ("ln_dl30", "30-day downloads (base + official instruct)"),
        ("ln_likes", "HF likes (base repo)"),
        ("or_listed_i", "listed on OpenRouter (2026-09-23), LPM"),
        ("ln_prov", "ln(1 + OpenRouter providers)"),
        ("ln_votes", "ln(1 + LMArena votes), matched models"),
    ]
    rhs = {"uncond": "lnw", "cond_N": "lnw + lnN + C(yr) + ln_age", "cond_C": "lnM + lnC + C(yr) + ln_age",
           "cond_C_lab": "lnM + lnC + C(yr) + ln_age + C(lab2)"}
    rows = []
    for y, lab in specs:
        for sk, r in rhs.items():
            x = B.dropna(subset=[y, "lnw", "lnM", "lnC", "ln_age"])
            if y == "ln_votes":
                x = x[x["arena_matched"]]
            if sk == "cond_C_lab":
                x = x[x.groupby("lab2")["lab2"].transform("size") >= 2]
            if len(x) < 15:
                continue
            m = smf.ols(f"{y} ~ {r}", data=x).fit(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(x["lab2"])[0]))
            key = "lnw" if "lnw" in r.split(" + ")[0] else "lnM"
            rows.append(dict(outcome=y, outcome_label=lab, spec=sk, key=key, coef=m.params[key], se=m.bse[key],
                             p=m.pvalues[key], n=int(m.nobs), labs=x["lab2"].nunique(), r2=m.rsquared,
                             coef_lnC=m.params.get("lnC", np.nan), se_lnC=m.bse.get("lnC", np.nan)))
    return pd.DataFrame(rows), B


# ----------------------------------------------------------------------------- (6) rival wedges
def rivals(d, techs, mstar_f):
    out = {}
    U = d[d["prod"]].copy()
    # (a) data scarcity / Kaplan-era beliefs: w < 1
    lo = U[U[f"w_{REF}"] < 1].sort_values("date")
    out["w_lt1"] = lo[["model", "org", "lab", "year", "N", "D", "M", f"w_{REF}", "band_lo", "band_hi", "open_weights"]]
    out["share_w_lt1_by_year"] = U.groupby("year").apply(lambda g: pd.Series(dict(
        n=len(g), share_lt1=(g[f"w_{REF}"] < 1).mean(),
        share_lt1_allband=(g["band_hi"] < 1).mean()))).reset_index()
    # (b) memory / hardware tiers: bunching of open-weight N and the wedge by tier
    O = U[U["open_weights"] & U["year"].between(2023, 2026)].copy()
    O["lnN"] = np.log10(O["N_total"])
    bins = np.arange(8, 12.6, 0.1)
    h, e = np.histogram(O["lnN"], bins)
    out["bunch_hist"] = pd.DataFrame(dict(lo=e[:-1], hi=e[1:], n=h))
    # tier comparison conditional on year and compute
    O["tier_consumer"] = O["N_total"].between(6.5e9, 9.5e9).astype(int)
    O["lnC"], O["yr"] = np.log(O["Cmp"]), O["year"].astype(str)
    O["lab2"] = O["dev"]
    m = smf.ols("lnw ~ tier_consumer + lnC + C(yr)", data=O).fit(cov_type="cluster",
                                                                 cov_kwds=dict(groups=pd.factorize(O["lab2"])[0]))
    out["tier_reg"] = dict(coef=m.params["tier_consumer"], se=m.bse["tier_consumer"], p=m.pvalues["tier_consumer"],
                           n=int(m.nobs), share_in_tier=O["tier_consumer"].mean(),
                           share_in_tier_lognormal=_lognormal_share(O["lnN"], np.log10(6.5e9), np.log10(9.5e9)))
    # (c) non-homotheticity: Farseer's own Eq. 3 vs the Chinchilla form fitted to the same sweep
    B = d[(d["sample"] == "B") & d["core"] & d["w_eq3"].notna()].copy()
    # [Review m3] "in range" = within Farseer's actual N support (non-embedding N <= largest Farseer model, 6.37e9).
    # The builder used 1.5x the largest model, which put 33 7-8B models "in range"; kept as a variant.
    sup = techs["farseer"].support
    B["in_farseer_N"] = B["N_nonemb"].between(sup["N_min"], sup["N_max"])
    B["in_farseer_N_1p5"] = B["N_nonemb"].between(sup["N_min"], 1.5 * sup["N_max"])
    B["above_farseer_N"] = B["N_nonemb"] > sup["N_max"]
    B["above_farseer_N_1p5"] = B["N_nonemb"] > 1.5 * sup["N_max"]
    out["eq3"] = B[["model", "N_nonemb", "D", "M", "w_farseer", "w_eq3", "Mstar_eq3_ownC", "in_farseer_N",
                    "above_farseer_N", "in_farseer_N_1p5", "above_farseer_N_1p5"]]
    Cg = np.array([1e19, 1e20, 1e21, 1e22, 1e23, 1e24])
    out["eq3_Mstar_curve"] = pd.DataFrame(dict(C=Cg, Mstar_eq3=mstar_f(Cg), Mstar_farseer_chin=techs["farseer"].mstar(Cg),
                                               Mstar_chin=techs["chin"].mstar(Cg), Mstar_meta_a2=techs["meta_a2"].mstar(Cg)))
    # (d) distillation / intermediate inputs
    Bd = d[(d["sample"] == "B") & d["core"]].copy()
    Bd = Bd.assign(lnC=np.log(Bd["Cmp"]), yr=Bd["year"].astype(str), dist=Bd["distilled"].fillna(False).astype(int),
                   syn=Bd["synthetic"].fillna(False).astype(int), lab2=Bd["lab"].fillna("other"))
    m = smf.ols("lnw ~ dist + syn + lnC + C(yr)", data=Bd).fit(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(Bd["lab2"])[0]))
    out["distill_reg"] = {k: dict(coef=m.params[k], se=m.bse[k], p=m.pvalues[k]) for k in ["dist", "syn", "lnC"]}
    out["distill_reg"]["n"] = int(m.nobs)
    # (e) factor-biased lab productivity (model_spec Prop. 4 / m7 A8): w_hat = w exp(alpha psi_N - beta psi_D).
    # m2's recipe tilts bound |alpha psi_N - beta psi_D| by ln(wedge_ratio) across recipes
    mag = pd.read_csv(os.path.join(ROOT, "output", "tables", "m2_neutrality_magnitudes.csv"))   # upstream (m2)
    out["m2_wedge_ratios"] = mag[["experiment", "estimator", "tilt_range", "tilt_range_lo", "tilt_range_hi", "wedge_ratio"]]
    Bb = d[(d["sample"] == "B") & d["core"]]
    rr = {}
    for lab, ratio in [("DataDecide NLS (primary)", float(mag.query("experiment=='datadecide' & estimator=='nls'")["wedge_ratio"].iloc[0])),
                       ("DataDecide Huber (primary)", float(mag.query("experiment=='datadecide' & estimator=='huber'")["wedge_ratio"].iloc[0])),
                       ("DataDecide M>=20 Huber (largest)", float(mag["wedge_ratio"].max()))]:
        rr[lab] = dict(ratio=ratio, share_w_above=(Bb[f"w_{REF}"] > ratio).mean(),
                       share_bandlo_above=(Bb["band_lo"] > ratio).mean(), n=len(Bb))
    out["factor_bias"] = rr
    return out


def _lognormal_share(x, a, b):
    mu, sd = np.mean(x), np.std(x, ddof=1)
    return float(stats.norm.cdf((b - mu) / sd) - stats.norm.cdf((a - mu) / sd))


# ----------------------------------------------------------------------------- (7) aggregate
def aggregate(d, techs, long):
    """Implied lifetime inference compute as a multiple of training compute for the open-weight ecosystem, by year:
    sum_i 2 N_i T_i / sum_i 6 N_i D_i = sum_i (w_i - 1) C_i / sum_i C_i  (compute-weighted mean of w - 1),
    raw and with T_i truncated at 0 (w < 1 cannot be inference demand). Bands: 95% bootstrap interval within each
    technology (draws) and the min/max over the band technologies."""
    U = d[d["prod"] & d["open_weights"] & d["year"].between(2019, 2026)].copy()
    rows = []
    for k in ["chin", "besi", "hoff", "farseer_emb", "gadre_rw", "meta_a2"]:
        t = techs[k]
        lnN, lnD = np.log(U["N"].values), np.log(U["D"].values)
        C = U["Cmp"].values
        yrs = U["year"].values
        draws = t.draws if t.draws is not None else np.array([[t.alpha, t.beta, t.lnG]])
        pt = np.exp(log_w(lnN, lnD, t.alpha, t.beta, t.lnG))
        Wd = np.exp(log_w(lnN[None, :], lnD[None, :], draws[:, 0:1], draws[:, 1:2], draws[:, 2:3]))
        for y in sorted(set(yrs)) + ["all"]:
            s = (yrs == y) if y != "all" else np.ones_like(yrs, bool)
            if s.sum() == 0:
                continue
            cw = C[s] / C[s].sum()
            mult = float(np.sum((pt[s] - 1) * cw))
            mult_tr = float(np.sum(np.maximum(pt[s] - 1, 0) * cw))
            md = ((Wd[:, s] - 1) * cw[None, :]).sum(1)
            mdt = (np.maximum(Wd[:, s] - 1, 0) * cw[None, :]).sum(1)
            rows.append(dict(tech=k, year=y, n=int(s.sum()), C_total=float(C[s].sum()), multiple=mult,
                             multiple_lo=float(np.percentile(md, 2.5)) if len(md) > 1 else np.nan,
                             multiple_hi=float(np.percentile(md, 97.5)) if len(md) > 1 else np.nan,
                             multiple_trunc=mult_tr,
                             multiple_trunc_lo=float(np.percentile(mdt, 2.5)) if len(mdt) > 1 else np.nan,
                             multiple_trunc_hi=float(np.percentile(mdt, 97.5)) if len(mdt) > 1 else np.nan,
                             median_w=float(np.median(pt[s])),
                             T_total=float(np.sum(3 * U["D"].values[s] * np.maximum(pt[s] - 1, 0)))))
    A = pd.DataFrame(rows)
    # band over technologies
    Bd = []
    for y, g in A.groupby("year"):
        lo = np.nanmin(np.r_[g["multiple_lo"].values, g["multiple"].values])
        hi = np.nanmax(np.r_[g["multiple_hi"].values, g["multiple"].values])
        lot = np.nanmin(np.r_[g["multiple_trunc_lo"].values, g["multiple_trunc"].values])
        hit = np.nanmax(np.r_[g["multiple_trunc_hi"].values, g["multiple_trunc"].values])
        ref = g[g["tech"] == REF].iloc[0]
        Bd.append(dict(year=y, n=int(ref["n"]), C_total=ref["C_total"], multiple_ref=ref["multiple"],
                       multiple_ref_lo=ref["multiple_lo"], multiple_ref_hi=ref["multiple_hi"], band_lo=lo, band_hi=hi,
                       multiple_trunc_ref=ref["multiple_trunc"], multiple_trunc_ref_lo=ref["multiple_trunc_lo"],
                       multiple_trunc_ref_hi=ref["multiple_trunc_hi"], band_trunc_lo=lot, band_trunc_hi=hit,
                       median_w_ref=ref["median_w"]))
    return A, pd.DataFrame(Bd)


def stated_intent(d):
    """Validation against developers' stated objectives (models whose papers state the allocation rule)."""
    cases = [
        ("Chinchilla", "A:Chinchilla", "training-compute-optimal by design", "hoffmann2022training"),
        ("Cerebras-GPT (111M-13B)", "Cerebras-GPT", "20 tokens per parameter, stated as compute-optimal", "dey2023cerebras"),
        ("Pythia (70M-12B)", "Pythia", "same 300B tokens for every size (research design)", "biderman2023pythia"),
        ("LLaMA-1 7B/13B", "LLaMA-small", "target: best performance at given inference budgets", "touvron2023llama"),
        ("Llama 3 8B/70B", "Llama-3", "smaller models trained far beyond compute-optimal for inference", "grattafiori2024llama"),
    ]
    rows = []
    for name, key, intent, cite in cases:
        if key.startswith("A:"):
            g = d[d["uid"] == key]
        elif key == "Cerebras-GPT":
            g = d[d["model"].astype(str).str.startswith("Cerebras-GPT")]
        elif key == "LLaMA-small":
            g = d[d["model"].isin(["llama-7b", "llama-13b"])]
        else:
            g = d[(d["family"] == key)]
        if len(g) == 0:
            continue
        rows.append(dict(case=name, n=len(g), intent=intent, cite=cite, M_min=g["M"].min(), M_max=g["M"].max(),
                         w_ref_min=g[f"w_{REF}"].min(), w_ref_max=g[f"w_{REF}"].max(),
                         band_lo=g["band_lo"].min(), band_hi=g["band_hi"].max(),
                         w_meta_a2_min=g["w_meta_a2"].min(), w_meta_a2_max=g["w_meta_a2"].max()))
    return pd.DataFrame(rows)
