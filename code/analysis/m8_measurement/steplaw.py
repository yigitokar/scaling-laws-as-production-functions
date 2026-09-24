"""steplaw.py -- hyperparameters as flexible inputs: concentrated technology, inefficiency, SFA and
flexible-input demand functions, using the Step Law dense grid (Li et al. 2025, 'Predictable Scale I').

Data: data/raw/steplaw/dense_lr_bs_loss.csv, 1,911 runs = 17 (N, D) cells x (LR x batch) grid.
N = non-embedding parameters (Step Law convention); batch 'bs' in sequences of 2,048 tokens; loss =
'smooth loss' (smoothed final training loss; single epoch, so ~ validation loss). No license file:
we use but do not redistribute raw rows (processed outputs here are cell-level aggregates).

Economic reading. With flexible inputs h = (ln LR, ln B) chosen after (N, D), the scaling law of interest is
the *concentrated* (envelope) technology L*(N, D) = min_h L(N, D, h). A run with h != h*(N, D) has
technical inefficiency u = ln L(N, D, h) - ln L*(N, D) >= 0. If E[u | N, D] varies with (N, D), fitting a
scaling law to non-optimized runs is an omitted-variable (transmission-type) problem: the regressors are
correlated with the inefficiency term. On an IsoFLOP the researcher's FOC becomes
    alpha u_N - beta v_D  =  L (u_n - u_d)            (u_n = du/dln N, u_d = du/dln D),
so only the *difference* of inefficiency gradients along the isocost line distorts allocation; its drift
with C biases the allocation exponent: a_obs - a = -(1/(alpha+beta)) d z / d ln C,  z = L (u_n - u_d)/(gamma R).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from common import RAW, MINI_GRID, sl, fit_warm, fit_efixed, model_row, parallel_map

CSV = os.path.join(RAW, "steplaw", "dense_lr_bs_loss.csv")
SEQ = 2048
VOCAB = 65536            # Step Law paper: BPE vocabulary of 65,536 [L] (Porian N_std for his rule; count robustness)
DIVERGED = 0.30          # u = ln L - ln L*_cell above this = diverged/failed run (loss > 1.35x cell best)

# Published flexible-input rules (literature values; see memo for sources)
STEP_LAW = dict(lr_c=1.79, lr_N=-0.713, lr_D=0.307, bs_c=0.58, bs_D=0.571)          # li2025predictablea; B in tokens
BJORCK = dict(lr_c=1.55e-3, lr_N=-0.23, lr_D=-0.32, bs_tokens=0.5e6)               # bjorck2024scaling (batch fixed 0.5M tokens)
PORIAN_RULE = dict(lr_c=3.7, lr_N=-0.36, bs_c=0.00037, bs_N=0.703)                  # porian2024resolving; B in 2048-token seqs
DEEPSEEK = dict(lr_c=0.3118, lr_C=-0.1250, bs_c=0.2920, bs_C=0.3271)                # bi2024deepseek; B in tokens, C in FLOPs


def load_steplaw():
    d = pd.read_csv(CSV)
    d = d.rename(columns={"smooth loss": "L", "loss": "L_raw"})
    d["N"], d["D"] = d.N.astype(float), d.D.astype(float)    # avoid int64 overflow in 6ND
    d["x"] = np.log(d.lr)                     # log learning rate
    d["y"] = np.log(d.bs)                     # log batch size (sequences)
    d["lr_k"] = np.round(2 * np.log2(d.lr)) / 2   # grid index (the CSV mixes 4- and 5-digit spellings of the same LR)
    d["cell"] = d.groupby(["N", "D"]).ngroup()
    d["lnN"], d["lnD"] = np.log(d.N), np.log(d.D)
    d["C"] = 6 * d.N * d.D
    d["Lstar"] = d.groupby("cell").L.transform("min")
    d["u"] = np.log(d.L) - np.log(d.Lstar)
    d["diverged"] = d.u > DIVERGED
    return d


def cells(d):
    c = d.groupby("cell").agg(N=("N", "first"), D=("D", "first"), n_runs=("L", "size"),
                              n_div=("diverged", "sum"), Lstar=("L", "min")).reset_index()
    c["M"] = c.D / c.N
    c["C"] = 6 * c.N * c.D
    return c


# ----------------------------------------------------------------------------- within-cell optima

def quad_surface(g, thr=0.02, n_draw=200, rng=None):
    """Quadratic in (ln LR, ln B) for ln L on runs within thr of the cell minimum; argmin with parametric
    (coefficient-draw) uncertainty. Returns (x*, y*, ln L*_smooth, draws of (x*, y*))."""
    s = g[g.u < thr]
    X = np.c_[np.ones(len(s)), s.x, s.y, s.x ** 2, s.y ** 2, s.x * s.y]
    yv = np.log(s.L.values)
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ b
    s2 = res @ res / max(len(yv) - 6, 1)
    V = s2 * np.linalg.inv(X.T @ X)

    def argmin(bb):
        H = np.array([[2 * bb[3], bb[5]], [bb[5], 2 * bb[4]]])
        z = -np.linalg.solve(H, bb[1:3])
        return z, bb[0] + bb[1:3] @ z + 0.5 * z @ H @ z, np.linalg.eigvalsh(H).min()

    z, f, mineig = argmin(b)
    draws = []
    if rng is not None:
        for bb in rng.multivariate_normal(b, V, n_draw):
            zz, _, me = argmin(bb)
            if me > 0:
                draws.append(zz)
    return dict(x=z[0], y=z[1], lnL=f, min_eig=mineig, n=len(s), resid_sd=np.sqrt(s2), draws=np.array(draws))


def cell_optima(d, thr=0.02, seed=1):
    rng = np.random.default_rng(seed)
    rows, draws = [], {}
    for c, g in d.groupby("cell"):
        q = quad_surface(g, thr=thr, rng=rng)
        best = g.loc[g.L.idxmin()]
        rows.append(dict(cell=c, N=best.N, D=best.D, lr_grid=best.lr, bs_grid=best.bs, L_grid=best.L,
                         lr_smooth=np.exp(q["x"]), bs_smooth=np.exp(q["y"]), L_smooth=np.exp(q["lnL"]),
                         se_lnlr=np.std(q["draws"][:, 0], ddof=1), se_lnbs=np.std(q["draws"][:, 1], ddof=1),
                         n_quad=q["n"], resid_sd=q["resid_sd"], min_eig=q["min_eig"]))
        draws[c] = q["draws"]
    return pd.DataFrame(rows), draws


# ----------------------------------------------------------------------------- flexible-input demand functions

def _ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * r[:, None]).T @ (X * r[:, None])
    V_hc1 = XtXi @ meat @ XtXi * n / (n - k)
    return b, np.sqrt(np.diag(V_hc1)), r


def demand_functions(opt, draws, B=1000, seed=2):
    """ln LR* = c + e_N ln N + e_D ln D ; ln B*_tokens = c + f_N ln N + f_D ln D (and Step Law's D-only form).
    SEs: HC1 on point estimates, and a two-level bootstrap (resample the 17 cells; draw each cell's optimum
    from its within-cell sampling distribution) for the smoothed optima."""
    rng = np.random.default_rng(seed)
    out = []
    lnN, lnD = np.log(opt.N.values), np.log(opt.D.values)
    X = np.c_[np.ones(len(opt)), lnN, lnD]
    Xd = np.c_[np.ones(len(opt)), lnD]
    specs = [("lr", "grid argmin", np.log(opt.lr_grid.values), X), ("lr", "smoothed argmin", np.log(opt.lr_smooth.values), X),
             ("bs", "grid argmin", np.log(opt.bs_grid.values * SEQ), X), ("bs", "smoothed argmin", np.log(opt.bs_smooth.values * SEQ), X),
             ("bs", "grid argmin, D only", np.log(opt.bs_grid.values * SEQ), Xd),
             ("bs", "smoothed argmin, D only", np.log(opt.bs_smooth.values * SEQ), Xd)]
    cells_ = opt.cell.values
    for var, how, yv, XX in specs:
        b, se, _ = _ols(XX, yv)
        boots = []
        for _ in range(B):
            idx = rng.integers(0, len(opt), len(opt))
            if "smoothed" in how:
                j = 0 if var == "lr" else 1
                yb = np.array([draws[cells_[i]][rng.integers(0, len(draws[cells_[i]])), j] for i in idx])
                yb = yb + (np.log(SEQ) if var == "bs" else 0.0)
            else:
                yb = yv[idx]
            try:
                boots.append(np.linalg.lstsq(XX[idx], yb, rcond=None)[0])
            except np.linalg.LinAlgError:
                continue
        boots = np.array(boots)
        row = dict(var=var, method=how, n=len(yv), const=b[0], const_se=se[0], const_bse=boots[:, 0].std(ddof=1))
        if XX.shape[1] == 3:
            row.update(e_N=b[1], e_N_se=se[1], e_N_bse=boots[:, 1].std(ddof=1), e_D=b[2], e_D_se=se[2],
                       e_D_bse=boots[:, 2].std(ddof=1))
        else:
            row.update(e_N=np.nan, e_N_se=np.nan, e_N_bse=np.nan, e_D=b[1], e_D_se=se[1], e_D_bse=boots[:, 1].std(ddof=1))
        out.append(row)
    return pd.DataFrame(out)


def conditional_lr_demand(d, thr=0.03):
    """Conditional LR demand (batch held fixed): within each (cell, batch) slice, argmin over LR of a quadratic
    in ln LR fitted to runs within thr of the slice minimum (>= 4 LR values, interior optimum).
    Pooled regression ln LR*_{c,b} = k0 + k_N ln N + k_D ln D + k_B ln B, SEs clustered by cell."""
    rows = []
    for (c, bs), g in d[~d.diverged].groupby(["cell", "bs"]):
        g = g.sort_values("x")
        gmin = g.L.min()
        s = g[np.log(g.L) - np.log(gmin) < thr]
        if s.lr_k.nunique() < 4:
            continue
        p = np.polyfit(s.x, np.log(s.L), 2)
        if p[0] <= 0:
            continue
        xs = -p[1] / (2 * p[0])
        interior = (xs > g.x.min()) and (xs < g.x.max())
        # also the discrete argmin, and whether it is at the edge of the LR grid run for that slice
        xa = g.loc[g.L.idxmin(), "x"]
        edge = (xa <= g.x.min() + 1e-9) or (xa >= g.x.max() - 1e-9)
        rows.append(dict(cell=c, N=g.N.iloc[0], D=g.D.iloc[0], bs=bs, lnlr_star=xs, lnlr_grid=xa, interior=interior,
                         grid_edge=edge, n=len(s), L_min=gmin))
    sl_ = pd.DataFrame(rows)
    # add slice-level inefficiency relative to the cell optimum (for the envelope/Le Chatelier comparison)
    return sl_


def cluster_ols(df, ycol, xcols, cluster="cell"):
    X = np.c_[np.ones(len(df)), df[xcols].values]
    y = df[ycol].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = df[cluster].nunique()
    for _, idx in df.groupby(cluster).indices.items():
        sc = X[idx].T @ r[idx]
        meat += np.outer(sc, sc)
    n, k = X.shape
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V)), G


# ----------------------------------------------------------------------------- flexible-input policies

def snap(g, lr, bs):
    """Nearest run in the cell to a target (LR, batch), distance in log2 units."""
    dist = np.sqrt((np.log2(g.lr) - np.log2(lr)) ** 2 + (np.log2(g.bs) - np.log2(bs)) ** 2)
    i = dist.idxmin()
    return g.loc[i], float(dist.loc[i])


def rule_targets(N, D, rule, best_fixed=None):
    C = 6 * N * D
    if rule == "frontier":
        return None
    if rule == "best_fixed":
        return best_fixed
    if rule == "porian_base":           # Porian's untuned 'base' (Kaplan reproduction): LR 3e-3, 256 x 2048 tokens
        return 3e-3, 256
    if rule == "porian_rule":           # Porian tuned N-rule; his N counts the head: N_std = N_nonemb + V h
        return None
    if rule == "deepseek":
        return DEEPSEEK["lr_c"] * C ** DEEPSEEK["lr_C"], DEEPSEEK["bs_c"] * C ** DEEPSEEK["bs_C"] / SEQ
    if rule == "bjorck":   # Bjorck's constant is for N and D in billions (checked: 7B, 1T tokens -> 1.1e-4)
        return BJORCK["lr_c"] * (N / 1e9) ** BJORCK["lr_N"] * (D / 1e9) ** BJORCK["lr_D"], BJORCK["bs_tokens"] / SEQ
    if rule == "steplaw":
        return STEP_LAW["lr_c"] * N ** STEP_LAW["lr_N"] * D ** STEP_LAW["lr_D"], STEP_LAW["bs_c"] * D ** STEP_LAW["bs_D"] / SEQ
    raise ValueError(rule)


RULE_LABELS = {
    "frontier": "Frontier: min over (LR, batch) grid",
    "steplaw": "Step Law rule (in-sample), snapped to grid",
    "best_fixed": "One fixed (LR, batch) for all cells (best on average)",
    "porian_base": "Porian 'base' fixed: LR 3e-3, batch 0.5M tokens",
    "porian_rule": "Porian tuned N-rule (LR, batch in N)",
    "deepseek": "DeepSeek C-rule (LR, batch in C)",
    "bjorck": "Bjorck (N, D)-rule, batch 0.5M tokens",
}


def policy_sample(d, rule, best_fixed=None, h_embed=None):
    """One run per cell chosen by a flexible-input policy (nearest grid point to the rule's target)."""
    rows = []
    for c, g in d.groupby("cell"):
        N, D = g.N.iloc[0], g.D.iloc[0]
        if rule == "frontier":
            r, dist = g.loc[g.L.idxmin()], 0.0
        else:
            if rule == "porian_rule":
                Nstd = N + VOCAB * g.h.iloc[0]
                tgt = (PORIAN_RULE["lr_c"] * Nstd ** PORIAN_RULE["lr_N"], PORIAN_RULE["bs_c"] * Nstd ** PORIAN_RULE["bs_N"])
            else:
                tgt = rule_targets(N, D, rule, best_fixed)
            r, dist = snap(g, *tgt)
        rows.append(dict(cell=c, N=N, D=D, L=r.L, lr=r.lr, bs=r.bs, u=r.u, snap_dist=dist))
    return pd.DataFrame(rows)


def best_fixed_pair(d, min_cells=15):
    """(LR, batch) grid pair minimizing mean inefficiency across cells, among pairs run in >= min_cells cells."""
    g = d.groupby(["lr_k", "bs"]).agg(nc=("cell", "nunique"), ubar=("u", "mean")).reset_index()
    g = g[g.nc >= min_cells].sort_values("ubar")
    top = g.iloc[0]
    return 2.0 ** top.lr_k, float(top.bs), g


def ineff_gradient(ps):
    """Regress the policy's inefficiency u (cell level) on ln N, ln D: the scale correlation that biases
    scaling-law exponents. HC1 SEs, n = 17 cells."""
    X = np.c_[np.ones(len(ps)), np.log(ps.N), np.log(ps.D)]
    b, se, _ = _ols(X, ps.u.values)
    return dict(u_mean=ps.u.mean(), u_n=b[1], u_n_se=se[1], u_d=b[2], u_d_se=se[2])


E_GRID = (1.2, 1.4, 1.6)


def fit_policy(ps, init=None, E_list=E_GRID, full_grid=False, delta=None, mini=False):
    """Technology on a policy sample (17 cells). Estimator: NLS on log loss (delta=None; Gaussian QMLE).
    With n = 17 nearly noise-free points the Huber(1e-3) objective of sl.fit_chinchilla is effectively L1 and
    has many local optima, so NLS is the default here (Huber reported as a robustness row for the frontier).
    Returns {'free': model with E estimated, E: model with E held fixed}."""
    out = {}
    grid = MINI_GRID if mini else (sl.DEFAULT_GRID if full_grid else sl.FAST_GRID)
    out["free"] = fit_warm(ps.N.values, ps.D.values, ps.L.values, init=init, grid=grid, delta=delta)
    for E in E_list:
        th = None
        if init is not None:
            th = np.array(init, float).copy()
            th[2] = np.log(E)
        out[E] = fit_efixed(ps.N.values, ps.D.values, ps.L.values, E, delta=delta, init=th,
                            grid=MINI_GRID if mini else sl.FAST_GRID)
    return out


# ----------------------------------------------------------------------------- random-configuration Monte Carlo

def _mc_one(args):
    """One Monte Carlo draw: each cell tries k random non-diverged configurations and keeps the best."""
    d, k, seed, init, check = args
    rng = np.random.default_rng(seed)
    rows = []
    for c, g in d[~d.diverged].groupby("cell"):
        idx = rng.choice(len(g), size=min(k, len(g)), replace=False)
        r = g.iloc[idx].sort_values("L").iloc[0]
        rows.append(dict(cell=c, N=r.N, D=r.D, L=r.L, u=r.u))
    ps = pd.DataFrame(rows)
    fits = fit_policy(ps, init=init, E_list=(1.4,), mini=True)
    res = dict(k=k, seed=seed, **{f"ineff_{kk}": v for kk, v in ineff_gradient(ps).items()})
    if check:   # same draw re-fitted with the full FAST_GRID
        full = fit_policy(ps, init=init, E_list=(1.4,), mini=False)
        res["check_abs_diff_a_free"] = abs(full["free"].a_N - fits["free"].a_N)
        res["check_abs_diff_a_E14"] = abs(full[1.4].a_N - fits[1.4].a_N)
    for key, m in fits.items():
        tag = "free" if key == "free" else f"E{key:.1f}"
        res.update({f"{tag}_{p}": getattr(m, p) for p in ("E", "alpha", "beta")})
        res[f"{tag}_a"] = m.a_N
        res[f"{tag}_sigma_star"] = m.sigma_star
    return res


def random_config_mc(d, ks=(1, 2, 4, 16), R=200, init=None, seed0=1000):
    args = [(d, k, seed0 + 10000 * k + r, init, r < 3) for k in ks for r in range(R)]
    return pd.DataFrame(parallel_map(_mc_one, args))


# ----------------------------------------------------------------------------- stochastic frontier

def sfa_cell_fe(d, dist="halfnormal", het=True, sample=None, sv_fixed=None):
    """Cost-type stochastic frontier with cell fixed effects (the concentrated technology in each cell):
        ln L_ij = mu_c + v_ij + u_ij,  v ~ N(0, s_v^2),  u >= 0 half-normal (or exponential with mean s_u),
        s_u,ij = exp(g0 + g_n (ln N - mean) + g_d (ln D - mean))
    i.e. scale-dependent inefficiency as in heteroskedastic SFA (Caudill, Ford & Gropper 1995).
    MLE by L-BFGS-B; SEs from a numerical Hessian. sv_fixed calibrates the noise s.d. (e.g. from the
    residual s.d. of the within-cell quadratic surfaces) instead of estimating it."""
    s = d if sample is None else sample
    cell = s.cell.values
    ncell = int(cell.max()) + 1
    y = np.log(s.L.values)
    zn = s.lnN.values - s.lnN.mean()
    zd = s.lnD.values - s.lnD.mean()

    def unpack(p):
        mu = p[:ncell]
        lsv = np.log(sv_fixed) if sv_fixed is not None else p[ncell]
        g = p[ncell + (0 if sv_fixed is not None else 1):]
        return mu, lsv, g

    def nll(p):
        mu, lsv, g = unpack(p)
        sv = np.exp(np.clip(lsv, -20, 2))
        su = np.exp(np.clip(g[0] + (g[1] * zn + g[2] * zd if het else 0.0), -20, 2))
        e = y - mu[cell]
        if dist == "halfnormal":
            sig = np.sqrt(su ** 2 + sv ** 2)
            lam = su / sv
            ll = np.log(2) - np.log(sig) + norm.logpdf(e / sig) + norm.logcdf(lam * e / sig)
        else:  # exponential inefficiency (cost frontier: e = v + u)
            ll = -np.log(su) - e / su + 0.5 * (sv / su) ** 2 + norm.logcdf(e / sv - sv / su)
        return -np.sum(ll)

    mu0 = np.log(s.groupby("cell").L.min().reindex(range(ncell)).values)
    p0 = np.r_[mu0, ([] if sv_fixed is not None else [np.log(0.002)]), np.log(0.03), 0.0, 0.0]
    best = None
    for shift in (0.0, -0.003, 0.003):
        pp = p0.copy()
        pp[:ncell] += shift
        r = minimize(nll, pp, method="L-BFGS-B", options=dict(maxiter=20000, maxfun=200000))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    H = _num_hess(nll, p)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full_like(p, np.nan)
    mu, lsv, g = unpack(p)
    k0 = ncell + (0 if sv_fixed is not None else 1)
    return dict(mu=mu, mu_se=se[:ncell], sv=float(np.exp(lsv)), g=g, g_se=se[k0:], nll=best.fun, n=len(y),
                converged=bool(best.success), dist=dist, sv_fixed=sv_fixed is not None)


def _num_hess(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    fx = f(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k)
            ei[i] = h; ej[j] = h
            if i == j:
                H[i, i] = (f(x + ei) - 2 * fx + f(x - ei)) / h ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h ** 2)
    return H


# ----------------------------------------------------------------------------- review additions (m8 reviewer)

def lechatelier_decomp(opt_b, cond_b, kind="smoothed"):
    """Point decomposition d ln eta*/d ln D (unconditional) = e_D|B + e_B * f_D on one (re)sample of cells.
    kind='smoothed' uses the quadratic-surface optima and interior conditional slices; 'grid' uses grid argmins and
    non-edge slices. Also returns the fraction of the gap between Step Law's published +0.307 and Bjorck's -0.32
    that conditioning on the batch closes, (e_D_unc - e_D|B) / 0.627."""
    lnN, lnD = np.log(opt_b.N.values), np.log(opt_b.D.values)
    X = np.c_[np.ones(len(opt_b)), lnN, lnD]
    if kind == "smoothed":
        ylr, ybs = np.log(opt_b.lr_smooth.values), np.log(opt_b.bs_smooth.values * SEQ)
        cc, ycol = cond_b[cond_b.interior], "lnlr_star"
    else:
        ylr, ybs = np.log(opt_b.lr_grid.values), np.log(opt_b.bs_grid.values * SEQ)
        cc, ycol = cond_b[~cond_b.grid_edge], "lnlr_grid"
    bu = np.linalg.lstsq(X, ylr, rcond=None)[0]
    bb = np.linalg.lstsq(X, ybs, rcond=None)[0]
    Xc = np.c_[np.ones(len(cc)), np.log(cc.N), np.log(cc.D), np.log(cc.bs * SEQ)]
    bc = np.linalg.lstsq(Xc, cc[ycol].values, rcond=None)[0]
    eD_unc, eD_con, eB, fD = bu[2], bc[2], bc[3], bb[2]
    return dict(e_D_unconditional=eD_unc, e_D_conditional=eD_con, e_B=eB, f_D=fD, implied_unconditional=eD_con + eB * fD,
                share_from_batch=eB * fD / eD_unc,
                share_of_published_gap_closed=(eD_unc - eD_con) / (STEP_LAW["lr_D"] - BJORCK["lr_D"]))


def lechatelier_bootstrap(opt, cond, B=1000, seed=3):
    """Cell-cluster bootstrap (resample the 17 cells with all their batch slices) of the decomposition, for both the
    smoothed and the grid-argmin optima. Draws without enough (N, D) variation to fit the regressions are skipped."""
    rng = np.random.default_rng(seed)
    cells_ = opt.cell.values
    out = {}
    for kind in ("smoothed", "grid"):
        pt = lechatelier_decomp(opt, cond, kind)
        bs = []
        for _ in range(B):
            pick = rng.integers(0, len(cells_), len(cells_))
            ob = opt.iloc[pick]
            if ob.N.nunique() < 2 or ob.D.nunique() < 3:
                continue
            cb = pd.concat([cond[cond.cell == cells_[i]] for i in pick])
            bs.append(lechatelier_decomp(ob, cb, kind))
        bs = pd.DataFrame(bs)
        out[kind] = {k: dict(point=float(v), se=float(bs[k].std(ddof=1)), q05=float(bs[k].quantile(.05)),
                             q95=float(bs[k].quantile(.95))) for k, v in pt.items()}
        out[kind]["n_boot"] = int(len(bs))
    return out


def sfa_loglik_i(p, y, cell, zn, zd, ncell, dist):
    """Per-observation log likelihood of sfa_cell_fe (sigma_v estimated), parameter vector (mu_1..mu_C, ln s_v, g)."""
    mu, lsv, g = p[:ncell], p[ncell], p[ncell + 1:]
    sv = np.exp(np.clip(lsv, -20, 2))
    su = np.exp(np.clip(g[0] + g[1] * zn + g[2] * zd, -20, 2))
    e = y - mu[cell]
    if dist == "halfnormal":
        sig = np.sqrt(su ** 2 + sv ** 2)
        return np.log(2) - np.log(sig) + norm.logpdf(e / sig) + norm.logcdf((su / sv) * e / sig)
    return -np.log(su) - e / su + 0.5 * (sv / su) ** 2 + norm.logcdf(e / sv - sv / su)


def sfa_cluster_se(sample, r, dist, h=1e-5):
    """Cluster-robust (by cell) sandwich SEs for the inefficiency-variance coefficients g of an sfa_cell_fe fit
    with sigma_v estimated: V = H^-1 (sum_c s_c s_c') H^-1 * G/(G-1). The 17 cells are the natural clusters: the
    inefficiency distribution within a cell is set by one designed (LR, batch) grid."""
    cell = sample.cell.values
    ncell = int(cell.max()) + 1
    y = np.log(sample.L.values)
    zn = sample.lnN.values - sample.lnN.mean()
    zd = sample.lnD.values - sample.lnD.mean()
    p = np.r_[r["mu"], np.log(max(r["sv"], 1e-12)), r["g"]]
    k = len(p)
    sc = np.zeros((len(y), k))
    for j in range(k):
        ej = np.zeros(k)
        ej[j] = h
        sc[:, j] = (sfa_loglik_i(p + ej, y, cell, zn, zd, ncell, dist) - sfa_loglik_i(p - ej, y, cell, zn, zd, ncell, dist)) / (2 * h)
    H = _num_hess(lambda q: -np.sum(sfa_loglik_i(q, y, cell, zn, zd, ncell, dist)), p)
    Hi = np.linalg.pinv(H)
    Gs = pd.DataFrame(sc).groupby(cell).sum().values
    V = Hi @ (Gs.T @ Gs) @ Hi * ncell / (ncell - 1)
    return np.sqrt(np.clip(np.diag(V)[-3:], 0, None))


def cell_level_ineff_regression(nd):
    """Transparent cross-check of the SFA variance function: ln(mean iota_c) on centred ln N, ln D across the 17
    cells (HC1). For exponential inefficiency E[iota] = sigma_u, so the slopes are directly comparable to g_N, g_D."""
    cm = nd.groupby("cell").agg(u=("u", "mean"), lnN=("lnN", "first"), lnD=("lnD", "first"))
    X = np.c_[np.ones(len(cm)), cm.lnN - nd.lnN.mean(), cm.lnD - nd.lnD.mean()]
    b, se, _ = _ols(X, np.log(cm.u.values))
    return dict(g0=b[0], g0_se=se[0], g_n=b[1], g_n_se=se[1], g_d=b[2], g_d_se=se[2], n=len(cm))


def frontier_count_robustness(d, init=None):
    """Step Law's N is the non-embedding count (Kaplan's convention). Refit the frontier technology with the head
    counted (N + V h, Porian's 'standard' count) and with head + input embedding (N + 2 V h; untied embeddings
    assumed [A]); V = 65,536 from the Step Law paper [L]. Proposition M predicts a lower a once the omitted,
    scale-declining component is counted."""
    fr = policy_sample(d, "frontier")
    h = d.groupby("cell").h.first().reindex(fr.cell).values.astype(float)
    rows = []
    for lab, N in (("non-embedding (Step Law)", fr.N.values), ("+ head", fr.N.values + VOCAB * h),
                   ("+ head + input embedding", fr.N.values + 2 * VOCAB * h)):
        m = sl.fit_chinchilla(N, fr.D.values, fr.L.values, delta=None, grid=sl.FAST_GRID)
        e = fit_efixed(N, fr.D.values, fr.L.values, 1.4, delta=None, grid=sl.FAST_GRID)
        share = (N - fr.N.values) / N
        rows.append(dict(count=lab, share_min=share.min(), share_max=share.max(), free_E=m.E, free_a=m.a_N,
                         free_sigma_star=m.sigma_star, free_gamma=m.gamma, E14_a=e.a_N, E14_sigma_star=e.sigma_star,
                         E14_gamma=e.gamma, free_objective=m.extra["objective"]))
    return pd.DataFrame(rows)
