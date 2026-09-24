"""Semi-synthetic 'lab market' on real experimental outcomes (DataDecide): transmission and selection bias by regime.

DataDecide trains the same architecture on 25 data recipes x 14 sizes x 3 seeds (final D = 100 N). Treat each recipe
as a lab with Hicks-neutral productivity omega_r (its recipe fixed effect) and let labs choose which sizes to
train/release according to a behavioural rule (model_spec Propositions 2-3). Outcomes are the REAL evaluated
HellaSwag accuracies of the chosen (recipe, size, seed) runs, so the technology and the noise are real; only the
choice rule is simulated.  Truth = within-recipe slope of y on ln C on the full factorial design.

Regimes:
  budget     : each lab gets K sizes drawn independently of omega                    (Prop. 2: OLS consistent)
  funding    : the size window rises with the lab's omega rank                         (Prop. 2: OLS overstates)
  target     : each lab trains up the size ladder until y >= y*, releases that model   (Prop. 2: OLS attenuated)
  selection  : random sizes as in `budget`, but a model is released only if y >= y_bar (Prop. 3: OLS understates)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import PREFIX, TABDIR


def _ols_slope(d, fe=False):
    X = d[["c"]].copy()
    if fe:
        if d.data.nunique() < 2 or d.groupby("data").c.nunique().max() < 2:
            return np.nan
        X = pd.concat([X, pd.get_dummies(d.data, drop_first=True, dtype=float)], axis=1)
    return float(sm.OLS(d.y, sm.add_constant(X)).fit().params["c"])


def simulate(dd_fin, out="y_hellaswag", R=500, K=3, seed=0):
    d = dd_fin[dd_fin.N >= 5e7].dropna(subset=[out]).copy()
    d["y"] = d[out]
    sizes = sorted(d.params.unique(), key=lambda s: d[d.params == s].N.iloc[0])
    d["s"] = d.params.map({s: i for i, s in enumerate(sizes)})
    truth = _ols_slope(d, fe=True)
    X = pd.concat([d[["c"]], pd.get_dummies(d.data, drop_first=True, dtype=float)], axis=1)
    r = sm.OLS(d.y, sm.add_constant(X)).fit()
    om = (d.y - r.params["c"] * d.c).groupby(d.data).mean()
    rank = om.rank(pct=True)
    labs = list(om.index)
    S = len(sizes)
    ystar = d[d.s == S // 2].y.median()        # capability target: median lab at the middle size
    ybar = d.y.quantile(0.4)                     # release bar for the selection regime
    rng = np.random.default_rng(seed)
    cell = {k: g for k, g in d.groupby(["data", "s"])}
    res = []
    for it in range(R):
        draws = {}
        for regime in ["budget", "funding", "target", "selection"]:
            rows = []
            for lab in labs:
                if regime in ("budget", "selection"):
                    ss = rng.choice(S, K, replace=False)
                elif regime == "funding":
                    c0 = int(np.clip(np.round(1 + (S - 3) * rank[lab] + rng.uniform(-1, 1)), 1, S - 2))
                    ss = [c0 - 1, c0, c0 + 1]
                else:
                    ss = []
                    for s in range(S):
                        g = cell[(lab, s)]
                        yv = g.sample(1, random_state=rng.integers(1e9)).iloc[0]
                        if yv.y >= ystar or s == S - 1:
                            rows.append(yv)
                            break
                    continue
                for s in ss:
                    g = cell[(lab, s)]
                    yv = g.sample(1, random_state=rng.integers(1e9)).iloc[0]
                    if regime == "selection" and yv.y < ybar:
                        continue
                    rows.append(yv)
            sd = pd.DataFrame(rows)
            res.append(dict(it=it, regime=regime, estimator="Pooled OLS", est=_ols_slope(sd)))
            res.append(dict(it=it, regime=regime, estimator="Recipe (lab) FE", est=_ols_slope(sd, fe=True)))
    R_ = pd.DataFrame(res)
    S_ = R_.groupby(["regime", "estimator"]).est.agg(mean="mean", sd="std", p5=lambda x: np.nanpercentile(x, 5),
                                                      p95=lambda x: np.nanpercentile(x, 95)).reset_index()
    S_["truth"] = truth
    S_["bias"] = S_["mean"] - truth
    S_["sd_omega"] = float(om.std())
    S_["sd_c"] = float(d.c.std())
    lab_names = {"budget": "Budget (exogenous compute)", "funding": "Funding (compute rises with TFP)",
                 "target": "Capability target", "selection": "Release only if y ≥ bar"}
    S_["regime"] = S_.regime.map(lab_names)
    order = [lab_names[k] for k in ["budget", "funding", "target", "selection"]]
    S_["regime"] = pd.Categorical(S_.regime, order)
    S_ = S_.sort_values(["regime", "estimator"])
    S_.to_csv(os.path.join(TABDIR, f"{PREFIX}_regimes_semisynthetic.csv"), index=False)
    return S_
