"""Figures for module m4_observational (AER style via aer_style)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aer_style as st
from common import PREFIX

LAB = {"theta_N": r"$\theta_N$ (elasticity w.r.t. $\ln N$)", "theta_D": r"$\theta_D$ (elasticity w.r.t. $\ln D$)",
       "theta_C": r"$\theta_C$ (compute-only regression)"}


def fig_lalonde(res, exp_global, out="y_hellaswag", name="fig5_lalonde"):
    """Figure 5: observational estimates (blue) vs the design-matched experimental benchmark (orange), 95% CIs.
    Gray band: range of global (not design-matched) linear experimental estimates."""
    st.use()
    # IV rows are omitted from the figure (first-stage F < 1 on this sample; see Table 6)
    order = ["OLS", "Year FE", "Developer FE", "Family FE", "Lab x period FE", "OP-style selection control",
             "OLS, EIV-corrected (Epoch 'Confident')",
             "Family FE, EIV-corrected (Epoch 'Confident')", "Reverse (c on y)", "Reverse (c on y), family FE"]
    short = {"OP-style selection control": "OP-style selection", "IV: frontier FLOP/$": "IV (frontier FLOP/\\$)",
             "IV: frontier FLOP/$ + developer FE": "IV + developer FE",
             "OLS, EIV-corrected (Epoch 'Confident')": "OLS, EIV-corrected",
             "Family FE, EIV-corrected (Epoch 'Confident')": "Family FE, EIV-corr.",
             "Reverse (c on y)": "Reverse regression", "Reverse (c on y), family FE": "Reverse, family FE",
             "Lab x period FE": "Lab " + "×" + " period FE"}
    fig, axes = plt.subplots(1, 3, figsize=(st.WIDTH_FULL, 3.6), sharey=True)
    ests = [e for e in order if (res.estimator == e).any()]
    ypos = {e: len(ests) - 1 - i for i, e in enumerate(ests)}
    for ax, par in zip(axes, ["theta_N", "theta_D", "theta_C"]):
        g = exp_global[(exp_global.output == out) & (exp_global.param == ({"theta_C": "theta_C_ray"}.get(par, par)))]
        g = g[g.dataset.isin(["OLMo ladder", "OLMo ladder + OLMo-2", "Gadre et al. (N>=0.1B)", "DataDecide"])]
        if len(g):
            ax.axvspan(g.est.min(), g.est.max(), color=st.GRID, alpha=0.9, lw=0, zorder=0)
        r = res[(res.param == par) & res.estimator.isin(ests)]
        for _, x in r.iterrows():
            y = ypos[x.estimator]
            ax.errorbar(x.obs, y + 0.15, xerr=1.96 * x.obs_se, fmt="o", color=st.BLUE, ms=3.5, lw=1.0, capsize=0)
            ax.errorbar(x.bench, y - 0.15, xerr=[[x.bench - x.bench_lo], [x.bench_hi - x.bench]], fmt="D",
                        color=st.ORANGE, ms=3.2, lw=1.0, capsize=0)
        ax.set_title(LAB[par], fontsize=8.5)
        ax.axvline(0, color=st.MUTED, lw=0.5)
        # [review fix] widen the theta_C axis so the EIV-corrected family-FE interval (upper end ~1.1) is not clipped
        lim = {"theta_N": (-0.1, 1.0), "theta_D": (-0.4, 0.8), "theta_C": (-0.1, 1.25)}[par]
        ax.set_xlim(*lim)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([short.get(e, e) for e in ypos])
    h1 = axes[0].errorbar([], [], xerr=[], fmt="o", color=st.BLUE, ms=3.5, label="Observational estimate")
    h2 = axes[0].errorbar([], [], xerr=[], fmt="D", color=st.ORANGE, ms=3.2, label="Experimental technology, same design")
    import matplotlib.patches as mp
    h3 = mp.Patch(color=st.GRID, label="Range of global experimental fits (not design-matched)")
    fig.legend(handles=[h1, h2, h3], loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.06), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    st.savefig(fig, f"{PREFIX}_{name}")


def fig_support(df, lad, gad, hull_models, name="support"):
    st.use()
    from scipy.spatial import ConvexHull
    fig, ax = plt.subplots(figsize=(st.WIDTH_HALF + 0.6, 3.1))
    m = df[df.main]
    ax.scatter(m.N, m.D, s=9, color=st.MUTED, alpha=0.6, lw=0, label=f"Observational base models (n={len(m)})")
    h = m[m.model.isin(hull_models)]
    ax.scatter(h.N, h.D, s=12, color=st.BLUE, lw=0, label=f"... inside experimental support (n={len(h)})")
    g = gad[gad.N >= 1e8]
    ax.scatter(g.N, g.D, s=10, marker="s", facecolor="none", edgecolor=st.AQUA, lw=0.8, label=f"Gadre et al. runs (n={len(g)})")
    ax.scatter(lad.N, lad.D, s=12, marker="D", color=st.ORANGE, lw=0, label=f"OLMo ladder + OLMo-2 (n={len(lad)})")
    P = np.vstack([np.c_[lad.n, lad.d], np.c_[g.n, g.d]])
    hull = ConvexHull(P)
    v = np.r_[hull.vertices, hull.vertices[0]]
    ax.plot(np.exp(P[v, 0]), np.exp(P[v, 1]), color=st.INK2, lw=0.8, ls="--")
    xs = np.logspace(7.5, 12, 50)
    for M in [20, 200, 2000]:
        ax.plot(xs, M * xs, color=st.MUTED, lw=0.5, ls=":")
        # label along the line near the left edge (sparse region), rotated to the line's on-screen angle
        x0 = 5.6e7
        ang = np.degrees(np.arctan(ax.get_window_extent().height / ax.get_window_extent().width * np.log10(6e11 / 5e7) / np.log10(3e13 / 1e9)))
        ax.text(x0, M * x0 * 1.35, f"D/N={M}", fontsize=6, color=st.INK2, ha="left", va="bottom", rotation=ang,
                rotation_mode="anchor")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e7, 6e11); ax.set_ylim(1e9, 3e13)
    ax.set_xlabel("Parameters N"); ax.set_ylabel("Training tokens D")
    ax.legend(loc="lower right", fontsize=6.2, handletextpad=0.3, borderaxespad=0.2)
    st.savefig(fig, f"{PREFIX}_{name}")


def fig_elasticity_by_M(surf_params, center, df, hull_pts=None, name="elasticity_by_M"):
    """Local experimental elasticities implied by the quadratic surface, as a function of tokens per parameter,
    at N = 1B and N = 7B, with the distribution of observational D/N as a rug."""
    st.use()
    p = surf_params
    n0, d0 = center
    fig, ax = plt.subplots(figsize=(st.WIDTH_HALF + 0.6, 2.8))
    Ms = np.logspace(np.log10(5), np.log10(3000), 100)
    for N, ls in [(1e9, "-"), (7e9, "--")]:
        n = np.log(N)
        d = np.log(Ms * N)
        x, z = n - n0, d - d0
        thN = p["n"] + 2 * p["n2"] * x + p["nd"] * z
        thD = p["d"] + 2 * p["d2"] * z + p["nd"] * x
        ins = np.ones_like(Ms, bool)
        if hull_pts is not None:   # draw only inside the experimental support (no extrapolation)
            from scipy.spatial import Delaunay
            ins = Delaunay(hull_pts).find_simplex(np.c_[np.full_like(d, n), d]) >= 0
        ax.plot(Ms[ins], thN[ins], color=st.BLUE, ls=ls, label=rf"$\theta_N$, N={N/1e9:.0f}B")
        ax.plot(Ms[ins], thD[ins], color=st.ORANGE, ls=ls, label=rf"$\theta_D$, N={N/1e9:.0f}B")
    m = df[df.main]
    ax.plot(m.M, np.full(len(m), -0.08), "|", color=st.MUTED, ms=6, alpha=0.7)
    ax.axvline(20, color=st.MUTED, lw=0.5, ls=":")
    ax.text(21, 0.93, "Chinchilla\nD/N=20", fontsize=6.5, color=st.INK2, va="top")
    ax.set_xscale("log")
    ax.set_ylim(-0.12, 1.0)
    ax.set_xlabel("Tokens per parameter D/N (rug: observational models)")
    ax.set_ylabel("Local elasticity of HellaSwag logit")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    st.savefig(fig, f"{PREFIX}_{name}")


def fig_tfp(om, df, theta, name="tfp_families"):
    st.use()
    fams = df[df.main].groupby("family").agg(dev=("developer", "first"), special=("special", "first"),
                                               domain=("domain", "first"), k=("model", "size"))
    o = (om - om.median()) / theta     # log compute-equivalent relative to the median family
    o = o.sort_values()
    fig, ax = plt.subplots(figsize=(st.WIDTH_HALF + 0.6, 4.6))
    for i, (f, v) in enumerate(o.items()):
        sp = fams.loc[f, "special"] if f in fams.index else ""
        code = fams.loc[f, "domain"] == "code" if f in fams.index else False
        col = st.ORANGE if sp in ("distilled", "synthetic") else (st.AQUA if code else st.BLUE)
        ax.plot(v / np.log(10), i, "o", color=col, ms=3.5)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels(o.index, fontsize=6)
    ax.axvline(0, color=st.MUTED, lw=0.5)
    ax.set_xlabel(r"Family effect, $\log_{10}$ compute-equivalent (median family = 0)")
    p10, p90 = np.percentile(o.values, [10, 90])
    ax.axvspan(p10 / np.log(10), p90 / np.log(10), color=st.GRID, alpha=0.6, lw=0, zorder=0)
    import matplotlib.lines as ml
    hs = [ml.Line2D([], [], marker="o", ls="", color=c, ms=3.5, label=l) for c, l in
          [(st.BLUE, "general"), (st.AQUA, "code-specialised"), (st.ORANGE, "distilled / synthetic data")]]
    ax.legend(handles=hs, fontsize=6.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    st.savefig(fig, f"{PREFIX}_{name}")


def fig_regimes(sim, name="regimes"):
    st.use()
    fig, ax = plt.subplots(figsize=(st.WIDTH_HALF + 0.6, 2.6))
    order = list(dict.fromkeys(sim.regime))
    for j, (est, col, off) in enumerate([("Pooled OLS", st.BLUE, 0.12), ("Recipe (lab) FE", st.ORANGE, -0.12)]):
        s = sim[sim.estimator == est].set_index("regime").loc[order]
        ax.errorbar(s["mean"], np.arange(len(order)) + off, xerr=[s["mean"] - s.p5, s.p95 - s["mean"]], fmt="o",
                    color=col, ms=3.5, lw=1, capsize=0, label=est)
    t = sim.truth.iloc[0]
    ax.axvline(t, color=st.INK2, lw=0.8, ls="--")
    ax.text(t + 0.003, len(order) - 0.6, "experimental truth", fontsize=6.5, color=st.INK2, ha="left")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel(r"Estimated $\theta_C$ (HellaSwag logit on $\ln C$)")
    ax.set_ylim(-0.5, len(order) - 0.3)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.grid(axis="y", visible=False)
    st.savefig(fig, f"{PREFIX}_{name}")
