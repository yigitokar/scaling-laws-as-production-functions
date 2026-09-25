"""symwin.py -- review addition: two remaining estimator-artifact checks of the drift of sigma*_b with compute
(R2 Major 1 request 3): count-symmetric windows and a power-law frontier with E fitted.

R2: "several budgets have 9-10 points on one side and 2-4 on the other"; "a frontier slope from a power-law frontier
with E profiled, rather than the derivative of a log-cubic at the edge of its support".

Variants of ra1's primary estimator (isoflop.design_estimate: quadratic on the window |x - x0| <= 1 around the pooled
path centre, logcubic/logquad frontier), applied to each IsoFLOP design:
  primary           ra1's estimator unchanged (reproduces ra1_modelfree_isoflop_budgets.csv);
  symmetric         each valid budget's window trimmed to the k nearest runs on each side of the fitted argmin,
                    k = min(runs left, runs right) (so the quadratic sees equal numbers of runs on both sides); the
                    quadratic is refitted and the budget kept if the refitted argmin is still bracketed (>= 2 per side);
  power             primary windows; frontier E + K exp(-g c) with E fitted by least squares (ra1's 'power' kind);
  symmetric+power   both.
Statistic: the drift of sigma*_b on log10 C, by design (OLS) and pooled over the five IsoFLOP designs with design fixed
effects (OLS; classical s.e. -- point estimates are the object here: per-budget s.e. for the new variants would need
ra1's bootstrap, which is not re-run). Chinchilla and Llama 3 carry the drift; Marin is the control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rb1common as cm

rc = cm.rc
DESIGNS = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]


def _sym_fits(df, h):
    import isoflop as iso
    P = iso.prep(df)
    L = df["L"].values.astype(float)
    centres, _ = iso.path_centres(P, L)
    out = []
    for B_ in P:
        x, Lb = B_["x"], L[B_["idx"]]
        fb = iso.fit_budget_centered(x, Lb, 2, h, centres[B_["b"]])
        if not fb.get("ok"):
            continue
        xs, m = fb["xstar"], fb["mask"]
        xw, Lw = x[m], Lb[m]
        left, right = np.where(xw < xs - 1e-9)[0], np.where(xw > xs + 1e-9)[0]
        k = min(len(left), len(right))
        keep = np.r_[left[np.argsort(xs - xw[left])[:k]], right[np.argsort(xw[right] - xs)[:k]]]
        fs = iso.fit_budget_centered(xw[keep], Lw[keep], 2, np.inf, xs)
        out.append(dict(b=B_["b"], c=B_["c"], C=B_["C"], ok=bool(fs.get("ok")), curv=fs.get("curv", np.nan),
                        Lstar=fs.get("Lstar", np.nan), k_side=k, n_left_primary=len(left), n_right_primary=len(right),
                        curv_primary=fb["curv"], Lstar_primary=fb["Lstar"]))
    return pd.DataFrame(out)


def _sigma(cb, Ls, curv, fkind):
    import isoflop as iso
    if fkind == "auto":
        fkind = iso.fkind_auto(len(cb))
    fr = iso.frontier(np.asarray(cb), np.asarray(Ls), fkind)
    if fr is None:
        return None, fkind
    S = np.asarray(curv) / np.abs(fr["slope"])
    sig = np.where(fr["slope"] < 0, 2.0 / (2.0 + S), np.nan)
    return sig, fkind


def _ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = float(r @ r / max(len(y) - 2, 1))
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])) if len(y) > 2 else np.nan
    return float(b[1]), se


def run(log=cm.log):
    import isoflop as iso
    designs = rc.isoflop_designs()
    rows, budget_rows = [], []
    for name in DESIGNS:
        df = designs[name]
        r0 = iso.design_estimate(df)
        sym = _sym_fits(df, iso.H_PRIMARY)
        ok = sym[sym.ok]
        variants = {}
        variants["primary"] = (r0["cb"], r0["sigma"], r0["fkind"])
        rp = iso.design_estimate(df, fkind="power")
        variants["power"] = (rp["cb"], rp["sigma"], "power") if rp.get("ok") else (np.array([]), np.array([]), "power")
        for fk, lab in (("auto", "symmetric"), ("power", "symmetric+power")):
            sig, fk_ = _sigma(ok.c.values, ok.Lstar.values, ok.curv.values, fk)
            variants[lab] = (ok.c.values, sig, fk_) if sig is not None else (np.array([]), np.array([]), fk_)
        for lab, (cb, sig, fk) in variants.items():
            cb, sig = np.asarray(cb, float), np.asarray(sig, float)
            g = np.isfinite(sig)
            lc = cb[g] / np.log(10)
            sl, se = _ols(lc, sig[g]) if g.sum() >= 3 else (np.nan, np.nan)
            rows.append(dict(design=name, variant=lab, frontier=fk, k_budgets=int(g.sum()), drift_ols=sl, se_ols=se,
                             sigma_mean=float(np.mean(sig[g])) if g.any() else np.nan,
                             sigma_top=float(sig[g][np.argmax(cb[g])]) if g.any() else np.nan,
                             C_top=float(np.exp(cb[g].max())) if g.any() else np.nan,
                             mean_k_side=float(ok.k_side.mean()) if lab.startswith("symmetric") else np.nan))
            for c_, s_ in zip(cb[g], sig[g]):
                budget_rows.append(dict(design=name, variant=lab, C=float(np.exp(c_)), sigma=float(s_)))
        log(f"  [{name}] drift of sigma*_b: " + "; ".join(f"{r['variant']} {r['drift_ols']:+.3f} (k={r['k_budgets']})"
                                                           for r in rows if r["design"] == name))
    T = pd.DataFrame(rows)
    Bd = pd.DataFrame(budget_rows)
    # pooled over the five IsoFLOP designs, design fixed effects (OLS), and over Chinchilla + Llama 3
    for lab_s, des in (("five IsoFLOP designs, design FE (OLS)", DESIGNS),
                       ("Chinchilla + Llama 3, design FE (OLS)", ["Chinchilla", "Llama 3"]),
                       ("Marin x3, design FE (OLS)", DESIGNS[2:])):
        for v in ("primary", "symmetric", "power", "symmetric+power"):
            g = Bd[(Bd.variant == v) & Bd.design.isin(des)]
            D = pd.get_dummies(g.design).astype(float).values
            x = np.log10(g.C.values)
            X = np.column_stack([D, x])
            b = np.linalg.lstsq(X, g.sigma.values, rcond=None)[0]
            r = g.sigma.values - X @ b
            s2 = float(r @ r / (len(r) - X.shape[1]))
            se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
            T = pd.concat([T, pd.DataFrame([dict(design=lab_s, variant=v, frontier="", k_budgets=len(g),
                                                 drift_ols=float(b[-1]), se_ols=se)])], ignore_index=True)
    log("  pooled drift (design FE, OLS): " + "; ".join(
        f"{r.design[:14]}|{r.variant}: {r.drift_ols:+.3f} ({r.se_ols:.3f})" for r in T[T.design.str.contains("FE")].itertuples()))
    return T
