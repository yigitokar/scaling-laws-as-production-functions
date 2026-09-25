"""figures_rb1.py -- figures for module rb1_sigmaC (project style: code/analysis/aer_style.py; at most three categorical
colors per panel, gray for context, one y-axis per panel, legends for two or more series).

  rb1_sigmaC_panel   (a) budget-level model-free sigma*_b against compute, six designs, with the pooled meta-regression
                     line and its CR2 band; (b) the identified set for sigma*(C) beyond the designs, with the training
                     compute of the clean-sample models. To merge into Figure 3 (revision_plan_v3).
  rb1_sigmaC_extrap_rb1version  [round 3: the paper's Figure 5, rb1_sigmaC_extrap, is drawn by rb5_sigma/figs_rb5.py]
                     Top: Delta = ln(w_param / w_local) by M bin on Farseer, Marin (three corpora) and Llama 3.
                     Bottom: local ln w against u = ln(M / M*_local(C)), with the fitted quadratic and its tangent at u=0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import rb1common as cm

st = cm.aer_style
BLUE, ORANGE, AQUA = st.BLUE, st.ORANGE, st.AQUA
INK, INK2, MUTED, GRID = st.INK, st.INK2, st.MUTED, st.GRID
MB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
MB_LAB = ["<16", "16–64", "64–256", "256–1k", "≥1k"]
STYLE = {"Chinchilla": dict(color=BLUE, marker="o", label="Chinchilla"),
         "Llama 3": dict(color=ORANGE, marker="s", label="Llama 3"),
         "Marin, Comma": dict(color=AQUA, marker="^", label="Marin (3 corpora)"),
         "Marin, DCLM": dict(color=AQUA, marker="^", label=None),
         "Marin, Nemotron-CC": dict(color=AQUA, marker="^", label=None),
         "Farseer (local path)": dict(color=INK2, marker="D", label="Farseer local path")}


def fig_panel(M, P):
    st.use()
    d = M["data"]
    d = d[~d.porian]
    line = M["primary_line"]
    Pr = M["preds"]
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_FULL, 3.25), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    ax = axes[0]
    jit = {"Chinchilla": -0.045, "Llama 3": 0.045, "Marin, Comma": -0.03, "Marin, DCLM": 0.0, "Marin, Nemotron-CC": 0.03,
           "Farseer (local path)": 0.0}
    for des, g in d.groupby("design"):
        s = STYLE[des]
        x = g.log10C.values + jit[des]
        y = g.y.values
        lo, hi = y - 1.96 * g.se.values, y + 1.96 * g.se.values
        ax.vlines(x, np.clip(lo, 0.40, None), np.clip(hi, None, 0.95), color=s["color"], lw=0.8, alpha=0.55, zorder=2)
        mfc = "white" if des.startswith("Farseer") else s["color"]
        ax.plot(x, y, ls="none", marker=s["marker"], ms=4.2, mfc=mfc, mec=s["color"], mew=0.9, zorder=3)
    # pooled lines
    cc = np.linspace(18.5, 21.5, 50)
    mu, b = line["mu"], line["beta"]
    cov = np.array(line["cov"])
    p = Pr[Pr.spec_key == "weighted"].set_index("log10C")
    # CR2 band interpolated from the prediction grid (19, 20, 21, 21.5, 22)
    lc = p.index.values
    lo_i = np.interp(cc, lc, p.lo_cr2.values)
    hi_i = np.interp(cc, lc, p.hi_cr2.values)
    ax.fill_between(cc, lo_i, hi_i, color=GRID, alpha=0.9, lw=0, zorder=1)
    ax.plot(cc, mu + b * (cc - 20), color=INK, lw=1.4, zorder=4)
    mu_u, b_u = M["unweighted_line"]["mu"], M["unweighted_line"]["beta"]
    ax.plot(cc, mu_u + b_u * (cc - 20), color=INK, lw=1.1, ls=":", zorder=4)
    ax.set_xlim(18.3, 21.75)
    ax.set_ylim(0.40, 0.95)
    ax.set_xlabel(r"$\log_{10}$ training compute (FLOP)")
    ax.set_ylabel(r"Model-free $\sigma^*_b$ by budget")
    ax.set_title("(a) Budget-level estimates and the pooled drift", loc="left")
    h = [Line2D([], [], ls="none", marker=STYLE[k]["marker"], color=STYLE[k]["color"],
                mfc="white" if k.startswith("Farseer") else STYLE[k]["color"], label=STYLE[k]["label"])
         for k in ("Chinchilla", "Llama 3", "Marin, Comma", "Farseer (local path)")]
    h += [Line2D([], [], color=INK, lw=1.4, label=f"Pooled, weighted ({b:+.3f}/decade)"),
          Line2D([], [], color=INK, lw=1.1, ls=":", label=f"Pooled, unweighted ({b_u:+.3f})")]
    leg_a = h
    # (b) identified set beyond the designs
    ax = axes[1]
    sig_top = P["sig_top"]
    labs = list(P["slopes"].keys())
    b_cl, b_pool = P["slopes"][labs[0]], P["slopes"][labs[1]]
    xx = np.linspace(21.0, 26.0, 60)
    lo_cl = np.clip(sig_top + b_cl * (xx - 21), 0.01, 0.999)
    lo_pl = np.clip(sig_top + b_pool * (xx - 21), 0.01, 0.999)
    ax.fill_between(xx, lo_cl, sig_top, color=BLUE, alpha=0.18, lw=0, label="Identified set (drift maintained)")
    ax.plot(xx, lo_cl, color=BLUE, lw=1.2, label=f"Lower bound: Chinchilla–Llama 3 drift ({b_cl:+.3f})")
    ax.plot(xx, lo_pl, color=BLUE, lw=1.1, ls="--", label=f"Lower bound: pooled drift ({b_pool:+.3f})")
    ax.plot(xx, np.full_like(xx, sig_top), color=INK, lw=1.2, label=f"Upper bound: top budgets ({sig_top:.2f})")
    ax.axhline(0.70, color=MUTED, lw=1.0, ls="-.", label="Design average (0.70; no drift)")
    ax.axvspan(18.5, 21.0, color=GRID, alpha=0.6, lw=0)
    ax.text(19.75, 0.50, "designs", ha="center", va="center", fontsize=7, color=INK2)
    # rug: training compute of the clean sample
    c, _ = __import__("pid").load_clean()
    ax.plot(c.log10C.values, np.full(len(c), 0.265), ls="none", marker="|", ms=7, color=INK2, alpha=0.7, mew=0.8)
    ax.text(18.6, 0.272, "clean-sample models", fontsize=6.5, color=INK2, va="bottom")
    ax.set_xlim(18.5, 26.0)
    ax.set_ylim(0.25, 0.95)
    ax.set_xlabel(r"$\log_{10}$ training compute (FLOP)")
    ax.set_ylabel(r"$\sigma^*(C)$")
    ax.set_title("(b) Beyond the designs: partial identification", loc="left")
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.6)
    fig.legend(handles=leg_a, loc="lower left", bbox_to_anchor=(0.06, -0.01), ncol=3, fontsize=6.6, handlelength=1.5,
               columnspacing=1.0)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    st.savefig(fig, f"{cm.PREFIX}_panel", folder=cm.FIGS)


def _farseer_u():
    g = pd.read_csv(cm.RA1_FAR_GRID)
    pth = pd.read_csv(cm.RA1_FAR_PATH)
    pth = pth[(pth.conv == "ne") & pth.M_star_local.notna() & (pth.nvalid > 0)]
    cg = np.log(6 * g.N * g.D)
    ok = (g.neff_ne >= 15) & g.lnw_local_ne.notna() & (cg >= np.log(pth.C.min())) & (cg <= np.log(pth.C.max()))
    lM = np.interp(cg[ok], np.log(pth.C.values), np.log(pth.M_star_local.values))
    return np.log(g.M[ok].values) - lM, g.lnw_local_ne[ok].values


def _design_u(v):
    grid, st0 = v["grid"], v["st0"]
    Cb = np.array([b[0] for b in v["budgets"]])
    lp = st0["path_lnMstar"]
    g = np.isfinite(lp)
    ok = (st0["neff"] >= 8) & np.isfinite(st0["lnw_local"])
    lMs = np.interp(np.log(grid["C_b"].values), np.log(Cb[g]), lp[g])
    inside = ok & (np.log(grid["C_b"].values) >= np.log(Cb[g]).min() - 1e-9) & (np.log(grid["C_b"].values) <= np.log(Cb[g]).max() + 1e-9)
    return np.log(grid["M"].values[inside]) - lMs[inside], st0["lnw_local"][inside]


def fig_extrap(X):
    st.use()
    fd = pd.read_csv(cm.RA1_FAR_DELTA)
    fd = fd[fd.conv == "ne"]
    lin_f = pd.read_csv(cm.RA1_FAR_LIN)
    pooled = pd.read_csv(f"{cm.TABLES}/{cm.PREFIX}_extrap_marin_pooled.csv")
    fig, axes = plt.subplots(2, 3, figsize=(st.WIDTH_FULL, 4.9), sharey="row")
    specs = [("kappa_full", BLUE, "o", r"$\kappa$ free, all runs"), ("chin_full", ORANGE, "s", "Chinchilla form, all runs"),
             ("chin_M100", AQUA, "^", r"Chinchilla form, fit on $M\leq100$")]
    xb = np.arange(len(MB))
    offs = [-0.2, 0.0, 0.2]
    panels = [("Farseer (bits/char; 404 runs)", "farseer"), ("Marin, three corpora (IsoFLOP)", "marin"),
              ("Llama 3 (IsoFLOP, digitized)", "llama")]
    for j, (title, key) in enumerate(panels):
        ax = axes[0, j]
        ax.axhline(0, color=INK2, lw=0.8)
        for (sp, col, mk, lab), o in zip(specs, offs):
            if key == "farseer":
                r = fd[fd.spec == sp].set_index("M_bin").reindex(MB)
                e, lo, hi = r.delta.values, r.lo.values, r.hi.values
            elif key == "marin":
                r = pooled[pooled.spec == sp].set_index("M_bin").reindex(MB)
                e, lo, hi = r.delta.values, r.lo.values, r.hi.values
                for nm in ("Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"):
                    t = X[nm]["T"]
                    tt = t[t.spec == sp].set_index("M_bin").reindex(MB)
                    ax.plot(xb + o, tt.delta.values, ls="none", marker=mk, ms=2.2, color=col, alpha=0.45, zorder=2)
            else:
                t = X["Llama 3"]["T"]
                r = t[t.spec == sp].set_index("M_bin").reindex(MB)
                e, lo, hi = r.delta.values, r.lo.values, r.hi.values
            okk = np.isfinite(e)
            ax.vlines((xb + o)[okk & np.isfinite(lo)], lo[okk & np.isfinite(lo)], hi[okk & np.isfinite(lo)], color=col, lw=1.0, zorder=3)
            ax.plot((xb + o)[okk], e[okk], ls="none", marker=mk, ms=4.5, color=col, mec="white", mew=0.5, zorder=4,
                    label=lab if j == 0 else None)
        ax.set_xticks(xb)
        ax.set_xticklabels(MB_LAB, fontsize=6.8, rotation=0)
        ax.set_title(f"({'abc'[j]}) {title}", loc="left", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"$\ln(w_{\rm param}/w_{\rm local})$")
        ax.set_xlabel(r"$M=D/N$ bin")
        ax.set_ylim(-2.2, 0.6)
    axes[0, 0].legend(loc="lower left", fontsize=6.3, handlelength=1.0)
    axes[0, 2].text(4, -1.9, "no\nsupport", ha="center", fontsize=6.5, color=MUTED)
    # bottom row: ln w_local against u, quadratic fit and tangent
    for j, (title, key) in enumerate(panels):
        ax = axes[1, j]
        if key == "farseer":
            u, y = _farseer_u()
            b1 = float(lin_f[(lin_f.conv == "ne") & (lin_f.param == "b1")].est.iloc[0])
            b2 = float(lin_f[(lin_f.conv == "ne") & (lin_f.param == "b2")].est.iloc[0])
            ax.plot(u, y, ls="none", marker="o", ms=1.8, color=MUTED, alpha=0.6, zorder=2)
        elif key == "marin":
            us, ys = [], []
            for nm, mk in (("Marin, Comma", "^"), ("Marin, DCLM", "v"), ("Marin, Nemotron-CC", "D")):
                u_, y_ = _design_u(X[nm])
                ax.plot(u_, y_, ls="none", marker=mk, ms=2.0, color=MUTED, alpha=0.7, zorder=2)
                us.append(u_); ys.append(y_)
            u, y = np.concatenate(us), np.concatenate(ys)
            Xm = np.column_stack([u, u * u])
            b1, b2 = np.linalg.lstsq(Xm, y, rcond=None)[0]
        else:
            u, y = _design_u(X["Llama 3"])
            L = X["Llama 3"]["L"].set_index("param")
            b1, b2 = float(L.loc["lin|b1", "est"]), float(L.loc["lin|b2", "est"])
            ax.plot(u, y, ls="none", marker="s", ms=1.8, color=MUTED, alpha=0.6, zorder=2)
        uu = np.linspace(np.nanmin(u), np.nanmax(u), 100)
        ax.plot(uu, b1 * uu + b2 * uu ** 2, color=BLUE, lw=1.4, zorder=3, label="Quadratic fit")
        ax.plot(uu, b1 * uu, color=ORANGE, lw=1.1, ls="--", zorder=3, label="Tangent at $u=0$ (linear form)")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.text(0.03, 0.97, f"$b_2={b2:+.3f}$", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_xlabel(r"$u=\ln(M/M^*_{\rm local}(C))$")
        if j == 0:
            ax.set_ylabel(r"Local $\ln w$")
            ax.legend(loc="lower right", fontsize=6.3, handlelength=1.6)
        ax.set_title(f"({'def'[j]}) {title.split(' (')[0]}", loc="left", fontsize=8)
    axes[1, 0].set_ylim(-3.5, 5.0)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    # [round 3, T16] The paper's Figure 5 (output/figures/rb1_sigmaC_extrap.*) is now drawn by module rb5_sigma
    # (figs_rb5.fig5: percentile intervals and the linear-from-local-slope reference). This rb1 version is kept under
    # its own name so that re-running rb1 cannot overwrite the paper's figure.
    st.savefig(fig, f"{cm.PREFIX}_extrap_rb1version", folder=cm.FIGS)


def make_all(M, P, X, T):
    fig_panel(M, P)
    fig_extrap(X)
    cm.log("  figures written")
