"""Section II (identification), version 3: exhibits and numbers that the section writer generates.

1. Figure 2 re-rendered from the m6 v2 Monte Carlo outputs with the version-3 notation (allocation-error scale
   \\varsigma instead of v, which is the data-loss term in eq. (1) of Section I; R3 minor 8, R4 minor 14). The data,
   estimators and layout are those of m6's outputs.fig2_v2 (code/analysis/m6_montecarlo/outputs.py); only the axis and
   legend labels change. Output: output/figures/m6_montecarlo_fig2_v3.{pdf,png}.
   Inputs: output/tables/m6_montecarlo_designA_summary.csv, m6_montecarlo_designA_profile.csv (m6 v2, 2026-09-24).

2. Off-path losses of three members of the kappa family that share Besiroglu et al.'s expansion path and frontier
   (Figure 1, panel b; R4 minor 15: "no output file contains these numbers"). In the family, the excess of reducible
   loss over the frontier at equal compute depends only on the wedge: R/R* = exp(gamma * Phi), with
   Phi = [ln(1 - a + a w) - a ln w] / [a (1 - a) S], ln w = (S/2) ln(M/M*), S = 2/sigma* - 2 (Online Appendix
   Lemma A5 / Proposition A2(i)). A brute-force check minimizes compute at the run's loss on the member's own
   technology. Output: output/tables/ident_v3_equivalent_family.csv.

Run: .venv/bin/python code/paper/make_ident_v3.py   (CPU only, a few seconds; no project module is modified)
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis", "m6_montecarlo"))
import aer_style as st  # noqa: E402
import design_a as A  # noqa: E402  (grid of sigma* values and the chi2 critical value only)
import mc_lib as ml  # noqa: E402  (true parameters only)

TAB = os.path.join(ROOT, "output", "tables")
S_LIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]   # as in m6 outputs.py
S0_POS = 0.01                                          # plotting position of varsigma = 0 on the log axis


def fig2_v3():
    summ = pd.read_csv(os.path.join(TAB, "m6_montecarlo_designA_summary.csv"))
    prof = pd.read_csv(os.path.join(TAB, "m6_montecarlo_designA_profile.csv"))
    st.use()
    fig = plt.figure(figsize=(st.WIDTH_FULL, 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.12, wspace=0.28)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    b = fig.add_subplot(gs[:, 1])
    cells = [f"opt_s{s:g}" for s in S_LIST]
    xs = np.array([max(s, S0_POS) for s in S_LIST])

    def ser(cell, est, par, col="rmse"):
        g = summ[(summ.cell == cell) & (summ.estimator == est) & (summ.param == par)]
        return float(g[col].iloc[0]) if len(g) else np.nan

    floor = 3e-3
    for a, par, ylab in [(a1, "sigma_star", r"RMSE of $\hat\sigma^*$"),
                         (a2, "lnMstar", r"RMSE of $\ln\hat M^*(10^{24})$")]:
        yp = np.array([ser(c, "primal", par) for c in cells])
        yd = np.array([ser(c, "dual", par) for c in cells])
        a.plot(xs, yp, "-o", color=st.BLUE, ms=3.6, lw=1.3, label=r"Primal ($\kappa=1$ Huber fit)")
        a.plot(xs, np.maximum(yd, floor), "--s", color=st.INK2, ms=3.2, lw=1.0, mfc="white",
               label=r"Dual ($\kappa=1$ + optimality)")
        if (yd < floor).any():
            a.annotate(r"dual exact at $\varsigma=0$", xy=(xs[0], floor), xytext=(xs[0] * 1.3, floor * 1.9),
                       fontsize=7.2, color=st.INK2, va="center")
        for cell, colr, ls_, lab in [("iso16", st.ORANGE, "--", r"IsoFLOP $\pm16\times$"),
                                     ("fact", st.AQUA, ":", "Factorial grid")]:
            a.axhline(ser(cell, "primal", par), color=colr, ls=ls_, lw=1.2, label=lab)
        a.set_yscale("log")
        a.set_ylabel(ylab, fontsize=8)
        a.set_ylim(floor * 0.6, None)
    a1.set_title(r"A. Precision against allocation error, same compute", loc="left")
    a1.legend(loc="upper right", fontsize=7.2, ncol=1, handlelength=2.2, borderaxespad=0.2)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.set_xscale("log")
    a2.set_xticks([S0_POS, 0.05, 0.1, 0.3, 1, 2])
    a2.set_xticklabels(["0", "0.05", "0.1", "0.3", "1", "2"])
    a2.minorticks_off()
    a2.set_xlabel(r"allocation-error s.d. $\varsigma$ in $\ln(D/N)$ (optimizing labs)")

    grid = A.SIG_GRID
    cols = [f"median_LR_{x:.3f}" for x in grid]
    for cell, colr, lab, ls_ in [("onpath", st.INK2, r"On-path ($\varsigma=0.02$)", "-"),
                                 ("opt_s0.3", st.BLUE, r"Optimizing labs, $\varsigma=0.3$", "-"),
                                 ("iso16", st.ORANGE, r"IsoFLOP $\pm16\times$", "--")]:
        v = prof.loc[prof.cell == cell, cols].to_numpy(float).ravel()
        b.plot(grid, np.maximum(v, 0.0), ls_, color=colr, label=lab, marker="o", ms=2.4, lw=1.3)
    b.axhline(A.CHI2_95, color=st.MUTED, lw=0.8)
    b.text(0.06, A.CHI2_95 * 1.18, r"$\chi^2_1$ 95% critical value", fontsize=7.2, color=st.INK2)
    b.axvline(ml.TRUE["sigma_star"], color=st.MUTED, lw=0.8, ls=":")
    b.text(ml.TRUE["sigma_star"] + 0.012, 2600, r"true $\sigma^*$", fontsize=7.2, color=st.INK2, ha="left")
    b.set_yscale("symlog", linthresh=1)
    b.set_ylim(-0.05, 9000)
    b.set_xlim(0.0, 1.0)
    b.set_xlabel(r"$\sigma^*$ imposed ($\kappa$ free)")
    b.set_ylabel("median LR statistic (vs. unrestricted optimum)", fontsize=8)
    b.set_title(r"B. Profile likelihood of $\sigma^*$, $\kappa$ free", loc="left")
    b.legend(loc="upper left", fontsize=7.2, bbox_to_anchor=(0.0, 1.0), handlelength=2.0, borderaxespad=0.1)
    st.savefig(fig, "m6_montecarlo_fig2_v3")


def equivalent_family():
    """Excess reducible loss at M = 5 M*(C) for kappa-family members sharing Besiroglu's path and frontier."""
    alpha, beta = 0.3478, 0.3658                      # Besiroglu et al. (2024), as in Section I
    a = beta / (alpha + beta)
    gamma = alpha * beta / (alpha + beta)
    rows = []
    for sig in (0.85, 0.74, 0.60, 2.0 / (2.0 + alpha + beta)):
        S = 2.0 / sig - 2.0
        lnw = 0.5 * S * np.log(5.0)
        w = np.exp(lnw)
        phi = (np.log(1.0 - a + a * w) - a * lnw) / (a * (1.0 - a) * S)
        closed = np.expm1(gamma * phi)
        # brute force: member with inner exponents a1 = (1-a)S, b1 = aS, kappa_S = gamma/(a(1-a)S) on the common path;
        # normalize N*(C) = D*(C) = 1 at the chosen compute, so the member's terms are equal to b1 : a1 there.
        a1, b1 = (1.0 - a) * S, a * S
        kap = gamma / (a * (1.0 - a) * S)
        Aa, Bb = b1, a1                                # A N^-a1 : B D^-b1 = b1 : a1 at N = D = 1 (the FOC)
        R = lambda n, d: (Aa * np.exp(-a1 * n) + Bb * np.exp(-b1 * d)) ** kap
        u = np.log(5.0)                                # ln(M/M*) at fixed compute: d - n = u, n + d = 0
        r_run = R(-u / 2, u / 2)
        # least log-compute c' = n + d that attains r_run on the member's own technology (along its path n = a c', d = (1-a)c')
        f = lambda cp: R(a * cp, (1.0 - a) * cp) - r_run
        cmin = brentq(f, -5.0, 0.0)
        brute = r_run / R(0.0, 0.0) - 1.0
        rows.append(dict(sigma_star=round(sig, 4), S=S, ln_w=lnw, w=w, Phi_closed=phi, Phi_brute=-cmin,
                         excess_closed=closed, excess_brute=brute))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "ident_v3_equivalent_family.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    fig2_v3()
    equivalent_family()
    print("wrote output/figures/m6_montecarlo_fig2_v3.{pdf,png} and output/tables/ident_v3_equivalent_family.csv")
