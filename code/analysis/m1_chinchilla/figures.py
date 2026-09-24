"""Figures for module m1_chinchilla (AER style via aer_style; max 3 categorical colours per panel)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from common import BESI_PUB, CHIN70B, FIGS, HOFF_TEX, TABLES, load_chinchilla, sl, theta_of  # noqa: E402  (sets sys.path)
import aer_style as S  # noqa: E402
import estimators as est  # noqa: E402

R = lambda f: pd.read_csv(os.path.join(TABLES, f))
L10 = np.log10


def _ref():
    df = load_chinchilla(5)
    return est.fit_huber(df.N.values, df.D.values, df.L.values, inits=[theta_of(BESI_PUB), theta_of(HOFF_TEX)])


# ----------------------------------------------------------------------------- Figure 1
def fig_plane(ref):
    full = load_chinchilla(0)
    lt = R("m1_chinchilla_labs_technology.csv")
    ll = lt[(lt.experiment == "llama_3") & lt.approach.str.startswith("A2")].iloc[0]
    vd = R("m1_chinchilla_design_variance.csv").set_index("sample").loc["full (n=240)"]
    fig, ax = plt.subplots(figsize=(S.WIDTH_FULL, 6.0))
    xlo, xhi, ylo, yhi = 7.6, 11.3, 8.2, 12.6
    # isocost lines C = 6ND, labelled where they leave the plot through the left or the top edge
    for c in range(18, 25):
        xs = np.array([xlo, xhi])
        ys = c - np.log10(6) - xs
        ax.plot(xs, ys, color=S.GRID, lw=0.9, ls=(0, (4, 3)), zorder=1)
        yl = c - np.log10(6) - xlo
        if ylo + 0.2 < yl < yhi - 0.1:
            ax.text(xlo + 0.03, yl - 0.08, f"$C=10^{{{c}}}$", fontsize=6.3, color=S.MUTED, ha="left", va="top")
        else:
            xl = c - np.log10(6) - yhi
            if xlo < xl < xhi - 0.2:
                ax.text(xl + 0.06, yhi - 0.03, f"$C=10^{{{c}}}$", fontsize=6.3, color=S.MUTED, ha="left", va="top")
    # isoquants of the reference technology
    gx, gy = np.meshgrid(np.linspace(xlo, xhi, 300), np.linspace(ylo, yhi, 300))
    Lg = ref.loss(10 ** gx, 10 ** gy)
    levels = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.3]
    cs = ax.contour(gx, gy, Lg, levels=levels, colors=S.INK2, linewidths=0.6, zorder=2)
    ax.clabel(cs, fmt=lambda v: f"L={v:.1f}", fontsize=6, inline=True, inline_spacing=2)
    # runs coloured by loss
    norm = Normalize(vmin=2.05, vmax=3.45)
    keep = full.rank_worst > 5
    sc = ax.scatter(L10(full.N[keep]), L10(full.D[keep]), c=full.L[keep], cmap=S.SEQ, norm=norm, s=12, lw=0.3,
                    edgecolor=S.MUTED, zorder=4)
    ax.scatter(L10(full.N[~keep]), L10(full.D[~keep]), facecolor="none", edgecolor=S.INK, s=16, lw=0.7, zorder=4,
               label="5 highest-loss runs (dropped, n = 240)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("Validation loss (nats/token)")
    cb.outline.set_linewidth(0.4)
    # expansion paths
    C = np.logspace(17.5, 25.2, 200)
    paths = [(ref, S.BLUE, rf"Refit (Huber, n = 240): $a$ = {ref.a_N:.3f}"),
             (HOFF_TEX, S.ORANGE, rf"Hoffmann et al. A3: $a$ = {HOFF_TEX.a_N:.3f}")]
    for m, col, lab in paths:
        ax.plot(L10(m.N_opt(C)), L10(m.D_opt(C)), color=col, lw=1.5, label=lab, zorder=5)
    Nm = C / (6 * ll.meta_repro_D_coef * C ** ll.meta_repro_D_exp)
    ax.plot(L10(Nm), L10(C / (6 * Nm)), color=S.AQUA, lw=1.5, zorder=5,
            label=rf"Meta Llama 3 (A2 argmins): $a$ = {1 - ll.meta_repro_D_exp:.3f}")
    ax.plot(L10(CHIN70B[0]), L10(CHIN70B[1]), marker="*", ms=10, color=S.INK, zorder=6, lw=0)
    ax.annotate("Chinchilla-70B\n(70B params, 1.4T tokens)", (L10(CHIN70B[0]), L10(CHIN70B[1])), xytext=(-118, -34),
                textcoords="offset points", fontsize=7, color=S.INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6, shrinkA=0, shrinkB=5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=S.MUTED, lw=0.4, alpha=1.0), zorder=8)
    # along vs transverse directions at the path point with C = 1e20
    c0 = 1e20
    p0 = np.array([L10(ref.N_opt(c0)), L10(ref.D_opt(c0))])
    al, be = ref.alpha, ref.beta
    u = np.array([be, al]) / np.hypot(al, be)
    t = np.array([al, -be]) / np.hypot(al, be)
    for vec, txt, off in ((u, f"along path: var = {vd.var_along:.2f}\n(no curvature information)", (0.08, -0.33)),
                          (t, f"transverse: var = {vd.var_trans:.2f}\n(identifies $\\sigma$)", (0.12, 0.02))):
        ax.annotate("", xy=p0 + 0.62 * vec, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.0), zorder=7)
        q = p0 + 0.62 * vec + np.array(off)
        ax.text(q[0], q[1], txt, fontsize=6.8, color=S.INK, zorder=7, va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\log_{10}$ parameters $N$")
    ax.set_ylabel(r"$\log_{10}$ training tokens $D$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7, handlelength=1.8)
    ax.grid(False)
    S.savefig(fig, "m1_chinchilla_isoquants", folder=FIGS)


# ----------------------------------------------------------------------------- Figure 2: profile over sigma*
def fig_profile():
    pr = R("m1_chinchilla_profile_sigma.csv")
    kap = R("m1_chinchilla_spec_kappa.csv")
    fig, axs = plt.subplots(1, 2, figsize=(S.WIDTH_FULL, 2.7), sharex=True, sharey=True)
    for ax, obj, ttl in ((axs[0], "gauss", "(a) Gaussian profile likelihood ratio"),
                         (axs[1], "huber", "(b) Huber objective, quasi-LR scale")):
        for smp, col, lab in (("full (n=240)", S.BLUE, "Full design (n = 240)"),
                              ("on-path band |dlnN|<=0.15", S.ORANGE, r"On-path band $|\Delta\ln N|\leq 0.15$ (n = 41)")):
            g = pr[(pr["sample"] == smp) & (pr.objective == obj)].sort_values("sigma_star")
            if obj == "gauss":
                y = g.LR
            else:
                n = 240 if smp.startswith("full") else 41
                y = 2 * n * np.log(g.obj / g.obj.min())
            ax.plot(g.sigma_star, y, color=col, lw=1.4, label=lab)
        ax.axhline(3.84, color=S.MUTED, lw=0.8, ls="--")
        ax.text(0.505, 3.84 * 1.2, r"$\chi^2_1$ 5% critical value", fontsize=6.5, color=S.MUTED)
        ax.set_yscale("symlog", linthresh=4)
        ax.set_ylim(0, 800)
        ax.set_title(ttl, loc="left")
    k = kap[(kap["sample"] == "n240")].iloc[0]
    for ax in axs:
        ax.axvline(k.sigma_star_kappa1, color=S.INK2, lw=0.7, ls=":")
        ax.text(k.sigma_star_kappa1 + 0.006, 330, r"$\kappa=1$" + "\nestimate", fontsize=6.5, color=S.INK2, va="top")
    axs[0].set_ylabel("Profile LR statistic")
    fig.supxlabel(r"$\sigma^*=2/(2+a_1+b_1)$ held fixed; $E$, $A$, $B$, $a_1/(a_1+b_1)$ and $\kappa$ profiled out", fontsize=8, y=0.1)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.3, wspace=0.08)
    S.savefig(fig, "m1_chinchilla_profile_sigma", folder=FIGS)


# ----------------------------------------------------------------------------- Figure 3: selection
def fig_selection():
    kt = R("m1_chinchilla_selection_k.csv")
    fig, axs = plt.subplots(1, 3, figsize=(S.WIDTH_FULL, 2.3))
    for ax, k, lab in ((axs[0], "beta", r"(a) $\hat\beta$ (data exponent)"), (axs[1], "alpha", r"(b) $\hat\alpha$ (parameter exponent)"),
                       (axs[2], "Mstar_1e+21", r"(c) $M^*$ at $C=10^{21}$")):
        ax.fill_between(kt.k, kt[f"huber_lo_{k}"], kt[f"huber_hi_{k}"], color=S.BLUE, alpha=0.15, lw=0)
        ax.plot(kt.k, kt[f"huber_{k}"], color=S.BLUE, marker="o", ms=3, label="Huber-LSE (95% pairs band)")
        ax.plot(kt.k, kt[f"lad_{k}"], color=S.ORANGE, ls="--", lw=1.2, label="LAD")
        ax.plot(kt.k, kt[f"gauss_{k}"], color=S.AQUA, ls="-.", lw=1.2, label="Gaussian NLS")
        ax.axvline(5, color=S.MUTED, lw=0.7, ls=":")
        ax.set_title(lab, loc="left")
        ax.set_xlabel("k highest-loss runs dropped")
    y0, y1 = axs[1].get_ylim()
    axs[1].text(5.3, y0 + 0.04 * (y1 - y0), "k = 5: Besiroglu\net al. sample", fontsize=6.5, color=S.INK2, va="bottom")
    # figure-level legend below the panels (inside panel (a) it hid the Gaussian curve at k = 0-1)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.3, wspace=0.28)
    S.savefig(fig, "m1_chinchilla_selection", folder=FIGS)


# ----------------------------------------------------------------------------- Figure 4: duality
def fig_duality(ref):
    mn = R("m1_chinchilla_a2_minima.csv")
    lm = R("m1_chinchilla_labs_a2_minima.csv")
    lt = R("m1_chinchilla_labs_technology.csv")
    fig, axs = plt.subplots(2, 2, figsize=(S.WIDTH_FULL, 4.6))
    g = mn[mn.dataset == "chinchilla_n240"]
    C = np.logspace(18.5, 21.8, 50)
    ax = axs[0, 0]
    ax.plot(L10(g.C), L10(g.Nstar), "o", color=S.INK, ms=3.5, label="A2 argmins (per-budget parabola)")
    ax.plot(L10(C), L10(np.exp(g.lnG_A2.iloc[0]) * (C / 6) ** g.a_A2.iloc[0]), color=S.BLUE, lw=1.3,
            label=f"A2 path: a = {g.a_A2.iloc[0]:.3f}")
    ax.plot(L10(C), L10(ref.N_opt(C)), color=S.ORANGE, lw=1.3, label=f"A3 refit: a = {ref.a_N:.3f}")
    ax.plot(L10(C), L10(HOFF_TEX.N_opt(C)), color=S.AQUA, lw=1.3, ls="--", label=f"A3 Hoffmann: a = {HOFF_TEX.a_N:.3f}")
    ax.set_title("(a) Chinchilla: factor demand $N^*(C)$", loc="left")
    ax.set_ylabel(r"$\log_{10} N^*$")
    ax.legend(fontsize=6.3, loc="upper left")
    ax = axs[0, 1]
    ax.plot(L10(g.C), g.Lstar, "o", color=S.INK, ms=3.5, label="Per-budget minima")
    E, lnK, gm = g.E_A1.iloc[0], g.lnK_A1.iloc[0], g.gamma_A1.iloc[0]
    ax.plot(L10(C), E + np.exp(lnK) * (C / 6) ** -gm, color=S.BLUE, lw=1.3, label=rf"A1 frontier: $\gamma$ = {gm:.3f}")
    ax.plot(L10(C), ref.L_opt(C), color=S.ORANGE, lw=1.3, label=rf"A3 refit: $\gamma$ = {ref.gamma:.3f}")
    ax.plot(L10(C), HOFF_TEX.L_opt(C), color=S.AQUA, lw=1.3, ls="--", label=rf"A3 Hoffmann: $\gamma$ = {HOFF_TEX.gamma:.3f}")
    ax.set_title("(b) Chinchilla: inverse cost function $L^*(C)$", loc="left")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=6.3, loc="upper right")
    # Llama 3
    gl = lm[lm.dataset == "llama_3"]
    a3 = lt[(lt.experiment == "llama_3") & lt.approach.str.startswith("A3")].iloc[0]
    m3 = sl.Chinchilla(E=a3.E, A=a3.A, B=a3.B, alpha=a3.alpha, beta=a3.beta)
    C = np.logspace(18.5, 22.3, 50)
    ax = axs[1, 0]
    ax.plot(L10(gl.C), L10(gl.Nstar), "o", color=S.INK, ms=3.5, label="A2 argmins")
    ax.plot(L10(C), L10(np.exp(gl.lnG_A2.iloc[0]) * (C / 6) ** gl.a_A2.iloc[0]), color=S.BLUE, lw=1.3,
            label=f"A2 path (= Meta's law): a = {gl.a_A2.iloc[0]:.3f}")
    ax.plot(L10(C), L10(m3.N_opt(C)), color=S.ORANGE, lw=1.3, label=f"A3 on Meta's IsoFLOPs: a = {m3.a_N:.3f}")
    ax.set_title("(c) Llama 3: factor demand $N^*(C)$", loc="left")
    ax.set_ylabel(r"$\log_{10} N^*$")
    ax.set_xlabel(r"$\log_{10} C$ (FLOP)")
    ax.legend(fontsize=6.3, loc="upper left")
    ax = axs[1, 1]
    ax.plot(L10(gl.C), gl.Lstar, "o", color=S.INK, ms=3.5, label="Per-budget minima")
    E, lnK, gm = gl.E_A1.iloc[0], gl.lnK_A1.iloc[0], gl.gamma_A1.iloc[0]
    ax.plot(L10(C), E + np.exp(lnK) * (C / 6) ** -gm, color=S.BLUE, lw=1.3, label=rf"A1 frontier: $\gamma$ = {gm:.3f}")
    ax.plot(L10(C), m3.L_opt(C), color=S.ORANGE, lw=1.3, label=rf"A3: $\gamma$ = {m3.gamma:.3f}")
    ax.set_title("(d) Llama 3: inverse cost function $L^*(C)$", loc="left")
    ax.set_ylabel("Loss (Llama 3 units)")
    ax.set_xlabel(r"$\log_{10} C$ (FLOP)")
    ax.legend(fontsize=6.3, loc="upper right")
    fig.tight_layout()
    S.savefig(fig, "m1_chinchilla_duality", folder=FIGS)


def run(log=print):
    S.use()
    ref = _ref()
    for f, args in ((fig_plane, (ref,)), (fig_profile, ()), (fig_selection, ()), (fig_duality, (ref,))):
        f(*args)
        log(f"  figure {f.__name__} written")
