"""figs_rb5.py -- Figure 5 regenerated with the new numbers (T16) and the call of Figure 3's script (T10).

Figure 5 (the extrapolation check; same layout as rb1_sigmaC/figures_rb1.fig_extrap):
  top row (a)-(c): Delta = ln(w_param/w_local) by M bin for Farseer, Marin (three corpora pooled; corpora faint) and
    Llama 3, for the kappa-free form, the Chinchilla form and the Chinchilla form fitted on M <= 100, now with PERCENTILE
    intervals (R4 minor 14), and a fourth series, the linear extrapolation from the local slope at the path,
    Delta_lin = ln(w_lin/w_local) (T1.9; its negative is the convexity component);
  bottom row (d)-(f): local ln w against u = ln(M/M*_local(C)) with the fitted quadratic and its tangent at u = 0 (the
    linear extrapolation from the local slope at the path).
Numbers: output/tables/rb5_sigma_extrap_delta_pct.csv, _extrap_decomposition.csv, _extrap_lin.csv (this module) and
the local-wedge evaluation points of ra1/rb1 (unchanged).
Output: previews output/figures/rb5_sigma_preview_fig5.{pdf,png} (and ..._fig3) by default; with final=True the
paper's files output/figures/rb1_sigmaC_extrap.* and fig3_merged.* (after the independent review).
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sigcommon as cm

st = cm.aer_style
BLUE, ORANGE, AQUA = st.BLUE, st.ORANGE, st.AQUA
INK, INK2, MUTED, GRID = st.INK, st.INK2, st.MUTED, st.GRID
MB = ["M<16", "16-64", "64-256", "256-1,024", ">=1,024"]
MB_LAB = ["<16", "16–64", "64–256", "256–1k", "≥1k"]
MARIN = ["Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC"]


def _farseer_u():
    g = pd.read_csv(cm.cm1.RA1_FAR_GRID)
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


def fig5(tables_dir=cm.TABLES, name="rb5_sigma_preview_fig5", folder=cm.FIGS):
    st.use()
    dp = pd.read_csv(os.path.join(tables_dir, f"{cm.PREFIX}_extrap_delta_pct.csv"))
    dec = pd.read_csv(os.path.join(tables_dir, f"{cm.PREFIX}_extrap_decomposition.csv"))
    lin = pd.read_csv(os.path.join(tables_dir, f"{cm.PREFIX}_extrap_lin.csv"))
    X = pickle.load(open(cm.RB1_EXTRAP_PKL, "rb"))
    fig, axes = plt.subplots(2, 3, figsize=(st.WIDTH_FULL, 4.9), sharey="row")
    specs = [("kappa_full", BLUE, "o", r"$\kappa$ free, all runs"), ("chin_full", ORANGE, "s", "Chinchilla form, all runs"),
             ("chin_M100", AQUA, "^", r"Chinchilla form, fit on $M\leq100$")]
    xb = np.arange(len(MB))
    offs = [-0.24, -0.08, 0.08, 0.24]
    panels = [("Farseer (bits/char; 404 runs)", "Farseer"), ("Marin, three corpora (IsoFLOP)", "Marin, three corpora pooled"),
              ("Llama 3 (IsoFLOP, digitized)", "Llama 3")]
    for j, (title, key) in enumerate(panels):
        ax = axes[0, j]
        ax.axhline(0, color=INK2, lw=0.8)
        for (sp, col, mk, lab), o in zip(specs, offs[:3]):
            r = dp[(dp.design == key) & (dp.spec == sp)].set_index("M_bin").reindex(MB)
            e, lo, hi = r.delta.values, r.lo_pct.values, r.hi_pct.values
            if key.startswith("Marin"):
                for nm in MARIN:
                    t = dp[(dp.design == nm) & (dp.spec == sp)].set_index("M_bin").reindex(MB)
                    ax.plot(xb + o, t.delta.values, ls="none", marker=mk, ms=2.2, color=col, alpha=0.45, zorder=2)
            okk = np.isfinite(e)
            m = okk & np.isfinite(lo)
            ax.vlines((xb + o)[m], lo[m], hi[m], color=col, lw=1.0, zorder=3)
            ax.plot((xb + o)[okk], e[okk], ls="none", marker=mk, ms=4.5, color=col, mec="white", mew=0.5, zorder=4,
                    label=lab if j == 0 else None)
        # linear extrapolation from the local slope at the path (T1.9)
        r = dec[(dec.design == key) & (dec.variant == "primary")].set_index("M_bin").reindex(MB)
        e, lo, hi = r.delta_lin.values, r.lo_pct.values, r.hi_pct.values
        okk = np.isfinite(e)
        m = okk & np.isfinite(lo)
        ax.vlines((xb + offs[3])[m], lo[m], hi[m], color=INK2, lw=1.0, zorder=3)
        ax.plot((xb + offs[3])[okk], e[okk], ls="none", marker="D", ms=3.8, color=INK2, mec="white", mew=0.5, zorder=4,
                label="Linear from local slope at path" if j == 0 else None)
        ax.set_xticks(xb)
        ax.set_xticklabels(MB_LAB, fontsize=6.8, rotation=0)
        ax.set_title(f"({'abc'[j]}) {title}", loc="left", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"$\ln(w_{\rm param}/w_{\rm local})$")
        ax.set_xlabel(r"$M=D/N$ bin")
        ax.set_ylim(-2.2, 0.6)
    axes[0, 0].legend(loc="lower left", fontsize=6.0, handlelength=1.0, labelspacing=0.25)
    axes[0, 2].text(4, -1.9, "no\nsupport", ha="center", fontsize=6.5, color=MUTED)
    for j, (title, key) in enumerate(panels):
        ax = axes[1, j]
        L = lin[(lin.design == key) & (lin.variant == "primary")].set_index("param")
        if key == "Farseer":
            u, y = _farseer_u()
            ax.plot(u, y, ls="none", marker="o", ms=1.8, color=MUTED, alpha=0.6, zorder=2)
        elif key.startswith("Marin"):
            us, ys = [], []
            for nm, mk in zip(MARIN, ("^", "v", "D")):
                u_, y_ = _design_u(X[nm])
                ax.plot(u_, y_, ls="none", marker=mk, ms=2.0, color=MUTED, alpha=0.7, zorder=2)
                us.append(u_)
                ys.append(y_)
            u, y = np.concatenate(us), np.concatenate(ys)
            b1, b2 = np.linalg.lstsq(np.column_stack([u, u * u]), y, rcond=None)[0]
            L = pd.DataFrame(dict(est=[b1, b2]), index=["b1", "b2"])
        else:
            u, y = _design_u(X["Llama 3"])
            ax.plot(u, y, ls="none", marker="s", ms=1.8, color=MUTED, alpha=0.6, zorder=2)
        b1, b2 = float(L.loc["b1", "est"]), float(L.loc["b2", "est"])
        uu = np.linspace(np.nanmin(u), np.nanmax(u), 100)
        ax.plot(uu, b1 * uu + b2 * uu ** 2, color=BLUE, lw=1.4, zorder=3, label="Quadratic fit")
        ax.plot(uu, b1 * uu, color=INK2, lw=1.1, ls="--", zorder=3, label="Linear from local slope at path")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.text(0.03, 0.97, f"$b_2={b2:+.3f}$", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_xlabel(r"$u=\ln(M/M^*_{\rm local}(C))$")
        if j == 0:
            ax.set_ylabel(r"Local $\ln w$")
            ax.legend(loc="lower right", fontsize=6.0, handlelength=1.6)
        ax.set_title(f"({'def'[j]}) {title.split(' (')[0]}", loc="left", fontsize=8)
    axes[1, 0].set_ylim(-3.5, 5.0)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    st.savefig(fig, name, folder=folder)
    return os.path.join(folder, name + ".pdf")


def fig3(final=False):
    """Figure 3 via the paper's script (code/paper/make_fig3_merged.py, which reads this module's identified-set CSV
    for panel c)."""
    script = os.path.join(cm.ROOT, "code", "paper", "make_fig3_merged.py")
    out = "fig3_merged" if final else "rb5_sigma_preview_fig3"
    env = dict(os.environ, RB5_TABLES=cm.TABLES, FIG3_FOLDER=cm.FIGS)
    r = subprocess.run([sys.executable, script, "--out", out], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout.strip()


def make_all(final=False, log=cm.log):
    p5 = fig5(name="rb1_sigmaC_extrap" if final else "rb5_sigma_preview_fig5")
    s3 = fig3(final=final)
    log(f"  figures: {p5}; Figure 3: {s3}")
