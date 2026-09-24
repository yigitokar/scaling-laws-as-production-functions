"""Redraw the algorithmic-progress ridge figure for Online Appendix E (paper v2).

Same content as m5_progress's `s2_dmr.figure_from_files()` (output/figures/m5_progress_dmr_ridge), with two changes:
  * notation follows paper v2: the doubling time of effective compute is tau_C (not T_C), and the parameter-augmenting
    share is written g_N/g_C (phi is used for another object in the main text);
  * panel (c) uses the refined profile over g_N/g_C from module ra4_obsfix (output/tables/ra4_obsfix_phi_profile.csv,
    m5's grid plus 31 re-optimized points near the interval boundaries) and shades the 95 percent profile interval
    read from data/processed/ra4_obsfix/progress_headline.json.
No estimation is redone: every input is a processed file written by m5_progress or ra4_obsfix.
Output: output/figures/appE_ridge.{pdf,png}.   Run: .venv/bin/python code/paper/make_appendix_e_ridge.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
import aer_style  # noqa: E402

PROC5 = os.path.join(ROOT, "data", "processed", "m5_progress")
PROC4 = os.path.join(ROOT, "data", "processed", "ra4_obsfix")
TAB = os.path.join(ROOT, "output", "tables")
LN2 = np.log(2.0)
AY_GRID = np.round(np.arange(-0.12, 0.0801, 0.005), 4)  # as in m5_progress/s2_dmr.py
BY_GRID = np.round(np.arange(-0.06, 0.1601, 0.005), 4)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    prof = pd.read_csv(os.path.join(PROC5, "dmr_profile_grid.csv"))
    floor = pd.read_csv(os.path.join(PROC5, "dmr_valley_floor.csv"))
    pdf = pd.read_csv(os.path.join(PROC5, "slsqp_path.csv"))
    summ = json.load(open(os.path.join(PROC5, "dmr_summary.json")))
    b1k = np.load(os.path.join(PROC5, "boot_m7_ho1000.npy"))[:, :-1]
    phi = pd.read_csv(os.path.join(TAB, "ra4_obsfix_phi_profile.csv")).sort_values("value")
    head = json.load(open(os.path.join(PROC4, "progress_headline.json")))
    lo, hi = head["phi"]["phi_ci_interp"]
    pts = summ["points"]

    aer_style.use()
    fig, axs = plt.subplots(1, 3, figsize=(aer_style.WIDTH_FULL, 2.55), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    # (a) profile LR region in (alpha_year, beta_year), bootstrap cloud, optimizer path
    ax = axs[0]
    P = prof.pivot(index="beta_year", columns="alpha_year", values="LR")
    X, Y = np.meshgrid(P.columns.values, P.index.values)
    lim = (AY_GRID.min(), AY_GRID.max())
    ax.scatter(b1k[:, 3], b1k[:, 8], s=1.5, color=aer_style.MUTED, alpha=0.35, lw=0, zorder=1)
    ax.contourf(X, Y, P.values, levels=[-1, 5.991], colors=[aer_style.BLUE], alpha=0.10, zorder=0)
    ax.contour(X, Y, P.values, levels=[5.991, 20], colors=[aer_style.INK, aer_style.MUTED],
               linewidths=[1.0, 0.6], linestyles=["-", ":"], zorder=2)
    ax.plot([], [], "-", color=aer_style.INK, lw=1.0, label="95% joint region (profile LR)")
    ax.plot(lim, lim, ls="--", color=aer_style.INK2, lw=0.7, zorder=2)
    ax.text(0.047, 0.066, "Hicks-\nneutral", fontsize=5.8, color=aer_style.INK2, ha="left", va="top")
    k0 = int(summ["slsqp_default_stop_iteration"])
    pp = pdf.iloc[max(k0 - 3, 0):]
    ax.plot(pp["alpha_year"], pp["beta_year"], ".", color=aer_style.ORANGE, ms=2.4, zorder=3,
            label="SLSQP iterates after the default stop")
    ax.plot(*pts["ho_default"], "o", color=aer_style.ORANGE, ms=5, mec="white", mew=0.6, zorder=4,
            label="Ho et al. (default stop)")
    ax.plot(*pts["ho_converged"], "D", color=aer_style.BLUE, ms=4.2, mec="white", mew=0.6, zorder=4,
            label="Same objective, converged")
    ax.plot(*pts["nls"], "s", color=aer_style.AQUA, ms=4.2, mec="white", mew=0.6, zorder=4, label="Unpenalized NLS")
    ax.plot([], [], "o", color=aer_style.MUTED, ms=2, label="Bootstrap draws (Ho code)")
    ax.set_xlim(*lim)
    ax.set_ylim(BY_GRID.min(), BY_GRID.max())
    ax.set_xlabel(r"$\alpha g_N$ (per year)")
    ax.set_ylabel(r"$\beta g_D$ (per year)")
    ax.set_title("(a) Profile LR, bootstrap, optimizer", loc="left")
    ax.legend(loc="lower left", fontsize=5.4, handletextpad=0.2, borderaxespad=0.15, labelspacing=0.25, markerscale=0.9)
    # (b) doubling time along the valley floor
    ax = axs[1]
    fl = floor[floor.g_C > 0]
    ax.plot(fl.alpha_year, 12 * LN2 / fl.g_C, color=aer_style.BLUE, lw=1.0)
    inside = fl[fl.LR <= 5.991]
    ax.plot(inside.alpha_year, 12 * LN2 / inside.g_C, color=aer_style.BLUE, lw=3.2, alpha=0.35, solid_capstyle="butt",
            label="inside 95% region")
    ax.axhline(8.4, color=aer_style.ORANGE, lw=0.9, ls=":")
    ax.text(AY_GRID.min() + 0.003, 8.9, "Ho et al.: 8.4", color=aer_style.ORANGE, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(3, 60)
    ax.set_yticks([3, 5, 10, 20, 40])
    ax.set_yticklabels(["3", "5", "10", "20", "40"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\alpha g_N$ along the valley floor")
    ax.set_ylabel(r"$\tau_C$ (months)")
    ax.set_title("(b) Doubling time along the ridge", loc="left")
    ax.legend(loc="upper left", fontsize=6.3)
    # (c) profile over the parameter-augmenting share g_N/g_C (refined grid)
    ax = axs[2]
    ax.axvspan(lo, hi, color=aer_style.BLUE, alpha=0.08, lw=0)
    ax.plot(phi.value, phi.LR, color=aer_style.INK)
    ax.axhline(3.841, color=aer_style.INK2, lw=0.8, ls="--")
    ax.text(phi.value.max(), 3.5, "95% cutoff", fontsize=6.0, color=aer_style.INK2, ha="right", va="top")
    ax.set_ylim(0, 14)
    ax.set_xlabel(r"parameter-augmenting share $g_N/g_C$")
    ax.set_ylabel("profile LR statistic")
    for _, rr in phi.iterrows():
        if np.any(np.isclose(rr.value, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])) and rr.g_C > 0 and rr.LR < 13:
            ax.annotate(f"{12 * LN2 / rr.g_C:.0f}", (rr.value, rr.LR), textcoords="offset points", xytext=(4, 3),
                        fontsize=6.3, ha="left", color=aer_style.BLUE)
    ax.text(0.97, 0.97, r"labels: $\tau_C$ (months)", transform=ax.transAxes, fontsize=6.3, color=aer_style.BLUE,
            va="top", ha="right")
    ax.set_title("(c) Profile over the factor bias", loc="left")
    fig.tight_layout(w_pad=0.6)
    aer_style.savefig(fig, "appE_ridge")
    print("wrote output/figures/appE_ridge.{pdf,png}; 95% interval for g_N/g_C:", round(lo, 3), round(hi, 3))


if __name__ == "__main__":
    main()
