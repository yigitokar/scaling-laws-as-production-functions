"""Figures for the theory module (AER style via aer_style).  All inputs come from closed forms verified in
symbolic.py / numeric.py, or from the numeric outputs passed in by run.py."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.stats import norm

from common import PARAM_SETS, ROOT, ce_ratio_from_w, kappa_family_member, lnR, path_objects  # also puts code/analysis on sys.path
import aer_style  # noqa: E402
from aer_style import AQUA, BLUE, INK2, MUTED, ORANGE  # noqa: E402

BES = PARAM_SETS["Besiroglu"]


def _chinchilla_transverse_sd():
    """sd of ln w across the 240 Chinchilla-extraction runs (the Besiroglu et al. estimation sample, 5 highest-loss points
    dropped) under the Besiroglu technology (design's transverse spread). Reviewer fix: the original read all 245 raw rows
    (sd 0.534) although the docstring and the bootstrap use the 240-run sample (sd 0.480)."""
    import sl
    df = sl.chinchilla_extraction(path=os.path.join(ROOT, "data/raw/epoch_chinchilla/svg_extracted_data.csv"))
    N = df["N"].values
    D = df["D"].values
    al, be = BES["alpha"], BES["beta"]
    lnw = np.log(al * BES["A"] * N ** -al / (be * BES["B"] * D ** -be))
    return float(np.std(lnw)), float(np.min(lnw)), float(np.max(lnw))


def fig_geometry():
    """(a) The (ln N, ln D) plane: isoquants, isocosts, expansion path and transverse deviation.
    (b) Observationally equivalent technologies: identical on the path, different off it."""
    aer_style.use()
    fig, axes = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 2.9))
    ax = axes[0]
    A_, B_, al, be = BES["A"], BES["B"], BES["alpha"], BES["beta"]
    po = path_objects(A_, B_, al, be)
    lnN = np.linspace(np.log(1e8), np.log(1e12), 300)
    lnD = np.linspace(np.log(1e9), np.log(1e14), 300)
    NN, DD = np.meshgrid(lnN, lnD)
    R = A_ * np.exp(-al * NN) + B_ * np.exp(-be * DD)
    lev = np.sort(po["K"] * (np.array([1e20, 1e21, 1e22, 1e23, 1e24]) / 6) ** (-po["gamma"]))
    ax.contour(NN / np.log(10), DD / np.log(10), R, levels=lev, colors=MUTED, linewidths=0.8)
    for C in (1e20, 1e21, 1e22, 1e23, 1e24):
        lc = np.log(C / 6)
        ax.plot(lnN / np.log(10), (lc - lnN) / np.log(10), ls=":", color=INK2, lw=0.7)
    lcs = np.linspace(np.log(1e19 / 6), np.log(1e25 / 6), 50)
    ax.plot((np.log(po["G"]) + po["a"] * lcs) / np.log(10), (-np.log(po["G"]) + po["b"] * lcs) / np.log(10), color=BLUE, lw=1.6,
            label="Expansion path (Besiroglu)")
    # an over-trained run and its projection to the path at equal compute
    N0, D0 = 8e9, 15e12
    lc0 = np.log(N0 * D0)
    n_s = np.log(po["G"]) + po["a"] * lc0
    ax.plot([np.log10(N0), n_s / np.log(10)], [np.log10(D0), (lc0 - n_s) / np.log(10)], color=ORANGE, lw=1.2)
    ax.plot(np.log10(N0), np.log10(D0), "o", color=ORANGE, ms=4.5, label="Over-trained run (8B, 15T)")
    w0 = al * A_ * N0 ** -al / (be * B_ * D0 ** -be)
    ax.annotate(f"equal-compute deviation\nfrom the path: $w={w0:.1f}$", xy=(10.45, 12.62), xytext=(10.25, 13.45), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.text(8.12, 11.05, "isocost $6ND=C$", fontsize=6.5, color=INK2, rotation=-38)
    ax.text(8.55, 13.55, "isoquant", fontsize=6.5, color=INK2)
    ax.set_xlim(8, 12)
    ax.set_ylim(9, 14)
    ax.set_xlabel(r"$\log_{10} N$ (parameters)")
    ax.set_ylabel(r"$\log_{10} D$ (tokens)")
    ax.set_title("(a) Isoquants, isocosts and the expansion path", loc="left")
    ax.legend(loc="lower right", fontsize=7, framealpha=1.0, frameon=True, edgecolor="white")

    ax = axes[1]
    a, gam, G, K = po["a"], po["gamma"], po["G"], po["K"]
    lc = np.log(1e22 / 6)
    n_s = np.log(G) + a * lc
    x = np.linspace(-np.log(200), np.log(200), 400)  # ln(M/M*(C)) at fixed compute
    lnRstar = np.log(K) - gam * lc
    for S, col in ((2 * (1 / 0.60 - 1), ORANGE), (al + be, BLUE), (2 * (1 / 0.85 - 1), AQUA)):
        m = kappa_family_member(S, a, gam, G, K)
        pen = lnR(n_s - x / 2, lc - n_s + x / 2, m["A"], m["B"], m["a1"], m["b1"], m["kappa"]) - lnRstar
        ax.plot(x / np.log(10), 100 * (np.exp(pen) - 1), color=col, label=rf"$\sigma^*={m['sigma_star']:.2f}$")
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"$\log_{10}(M/M^*(C))$ at fixed compute")
    ax.set_ylabel("Reducible loss above frontier (%)")
    ax.set_title("(b) Same path and frontier, different curvature", loc="left")
    ax.legend(loc="upper center", fontsize=7)
    ax.set_ylim(0, 60)
    fig.tight_layout()
    aer_style.savefig(fig, "m7_theory_geometry")


def fig_information(fisher_df):
    aer_style.use()
    fig, ax = plt.subplots(figsize=(aer_style.WIDTH_HALF + 0.4, 2.8))
    S = BES["alpha"] + BES["beta"]
    sd_lnw = (S / 2) * np.sqrt(fisher_df["var_x"].values)
    ax.loglog(sd_lnw, fisher_df["se_sigma_star"], "o-", color=BLUE, ms=3.5, label=r"se($\hat\sigma^*$): slope $-2$")
    # reviewer fix: s.e. of ln M* at the design's central compute (1e20 FLOP), not of the intercept -2 ln G (= ln M* at C = 6 FLOP)
    ax.loglog(sd_lnw, fisher_df["se_lnMstar_center"], "s-", color=ORANGE, ms=3.5,
              label=r"se($\ln \hat M^*$) at $C=10^{20}$: slope $-1$")
    sd_ch, _, _ = _chinchilla_transverse_sd()
    ax.axvline(sd_ch, color=MUTED, lw=0.8, ls="--")
    ax.text(sd_ch * 0.93, 30, "Chinchilla\nextraction", ha="right", fontsize=7, color=INK2)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"sd of transverse deviations, sd($\ln w$)")
    ax.set_ylabel("Asymptotic s.e. (unit noise, 45 runs)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Information and the wedge dispersion", loc="left")
    fig.tight_layout()
    aer_style.savefig(fig, "m7_theory_information")
    return sd_ch


def fig_wedge(pi_band, pi_summary):
    aer_style.use()
    fig, axes = plt.subplots(1, 3, figsize=(aer_style.WIDTH_FULL, 2.4))
    ax = axes[0]
    r = np.logspace(-1, np.log10(300), 200)
    for s_, col in ((0.70, ORANGE), (0.737, BLUE), (0.78, AQUA)):
        ax.semilogx(r, 3 * (r ** (1 / s_ - 1) - 1), color=col, label=rf"$\sigma^*={s_}$")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"$M/M^*(C)$")
    ax.set_ylabel(r"Revealed $T/D = 3(w-1)$")
    ax.set_title(r"(a) $w=(M/M^*)^{1/\sigma^*-1}$", loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axes[1]
    al, be = BES["alpha"], BES["beta"]
    w = np.logspace(np.log10(0.25), np.log10(8), 200)
    ax.semilogx(w, ce_ratio_from_w(w, al, be), color=BLUE, label="Exact (Besiroglu)")
    ax.semilogx(w, np.exp(np.log(w) ** 2 / (2 * (al + be))), color=ORANGE, ls="--", label="Harberger approx.")
    ax.set_xlabel(r"Wedge $w=\varepsilon_N/\varepsilon_D$")
    ax.set_ylabel(r"$C/C_{\min}$")
    ax.set_title("(b) Allocative loss", loc="left")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=6.5, loc="upper center")
    ax.set_ylim(0.9, 12)
    ax = axes[2]
    M = pi_band["M"].values
    ax.fill_between(M, pi_band["lo95"], pi_band["hi95"], color=BLUE, alpha=0.2, lw=0, label="95% bootstrap band")
    ax.semilogx(M, pi_band["T_over_D"], color=BLUE, label="Point estimate")
    ax.axvspan(M.min(), pi_summary["M_design_max"], color=MUTED, alpha=0.15, lw=0)
    # reviewer fix: the design reaches these M only at C <= 1.3e22, so at C = 7.2e23 the shaded range is not a support
    ax.text(1.3, 16, "design $M$ range\n(only at\n" + r"$C\leq1.3{\times}10^{22}$)", fontsize=6.0, color=INK2)
    ax.axvline(1875, color=ORANGE, lw=0.8, ls="--")
    ax.text(1875 * 0.8, -3.2, "8B/15T", fontsize=6.5, color=ORANGE, ha="right")
    ax.set_xlabel(r"$M = D/N$ at $C=7.2\times10^{23}$")
    ax.set_ylabel(r"$T/D$")
    ax.set_title("(c) Extrapolation band", loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    aer_style.savefig(fig, "m7_theory_wedge")


def fig_bias():
    """plim of forward OLS (y on c) and of the reverse-regression estimate 1/delta, relative to gamma, as a function of the
    compute rule pi1 (Gaussian model), for three selection severities (retained variance share r of the outcome)."""
    aer_style.use()
    gam, Vw, Ve, Vh = 0.178, 0.04, 0.01, 1.0
    p1 = np.linspace(-9, 6, 400)
    Vc = p1 ** 2 * Vw + Vh
    Cov = gam * Vc + p1 * Vw
    Vy = gam ** 2 * Vc + 2 * gam * p1 * Vw + Vw + Ve
    delta = Cov / Vy
    W = Vc - delta ** 2 * Vy
    fig, axes = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 2.5))
    ax = axes[0]
    for r, col, lab in ((1.0, BLUE, "no selection"), (0.5, ORANGE, r"selection, $r=0.5$"), (0.15, AQUA, r"selection, $r=0.15$")):
        b = delta * r * Vy / (delta ** 2 * r * Vy + W)
        ax.plot(p1 * gam, b / gam, color=col, label=lab)
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.axvline(-1, color=MUTED, lw=0.6, ls=":")
    ax.axvline(0, color=MUTED, lw=0.6, ls=":")
    ax.text(-1.02, 1.6, "target", fontsize=6.5, ha="right", color=INK2)
    ax.text(0.05, 1.6, "exogenous", fontsize=6.5, color=INK2)
    ax.text(0.6, 0.2, "funding", fontsize=6.5, color=INK2)
    ax.set_xlabel(r"Compute response to productivity, $\gamma\pi_1$")
    ax.set_ylabel(r"plim forward OLS / $\gamma$")
    ax.set_title("(a) Transmission and selection", loc="left")
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.73))
    ax.set_ylim(-0.05, 1.8)
    ax = axes[1]
    ok = Cov > 1e-9
    ax.plot((p1 * gam)[ok], (1 / delta)[ok] / gam, color=BLUE, label=r"reverse $1/\delta$ (selection-proof if Gaussian)")
    b1 = delta * Vy / (delta ** 2 * Vy + W)
    ax.plot(p1 * gam, b1 / gam, color=ORANGE, label="forward (no selection)")
    lo = -(1 + Ve / Vw)
    ax.axvspan(lo, 0, color=MUTED, alpha=0.15, lw=0)
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.text(lo + 0.05, 5.5, r"bracket contains $\gamma$", fontsize=6.5, color=INK2)
    ax.set_ylim(0, 6)
    ax.set_xlabel(r"$\gamma\pi_1$")
    ax.set_ylabel(r"estimate / $\gamma$")
    ax.set_title("(b) Forward-reverse bounds", loc="left")
    ax.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    aer_style.savefig(fig, "m7_theory_bias")
