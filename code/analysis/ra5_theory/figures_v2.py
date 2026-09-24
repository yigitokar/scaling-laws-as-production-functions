"""Figures for ra5_theory (AER style via aer_style; at most three colors per panel; no twin axes)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import aer_style
from aer_style import BLUE, INK2, MUTED, ORANGE

GRAY = "#b9b8b3"


def fig_pi_cone(pi, name="ra5_theory_pi_cone"):
    """(a) Identified sets for ln M*(C) beyond the largest Chinchilla budget, with verified-sample models.
    (b) Share of verified-sample models (beyond each anchor) whose sign of w-1 is identified, by assumption."""
    aer_style.use()
    fig, axes = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    ax = axes[0]
    aname = "Chinchilla (Epoch extraction)"
    fit = pi["fits"][aname]
    pts = pi["anchor_points"][aname]
    b = pi["sampleB"]
    l10 = np.log(10)
    ax.scatter(b["lnC"] / l10, b["lnM"] / l10, s=7, color=GRAY, lw=0, label="Verified-sample models", zorder=1)
    ax.scatter(pts["lnC"] / l10, pts["lnM"] / l10, s=14, color=BLUE, zorder=3, label="IsoFLOP minima (Chinchilla)")
    cJ = fit["cJ"]
    cin = np.linspace(pts["lnC"].min(), cJ, 20)
    ax.plot(cin / l10, (fit["mu"] + fit["e"] * (cin - cJ)) / l10, color=BLUE, lw=1.2, zorder=3)
    cc = np.linspace(cJ, np.log(1e26), 50)
    mu_lo, mu_hi = fit["mu_ci"]
    eL, eU = -0.14, 0.26
    ax.fill_between(cc / l10, (mu_lo + eL * (cc - cJ)) / l10, (mu_hi + eU * (cc - cJ)) / l10, color=BLUE, alpha=0.18, lw=0,
                    label=r"Identified set, $e\in[-0.14,0.26]$", zorder=2)
    ax.plot(cc / l10, (mu_lo - 1.0 * (cc - cJ)) / l10, color=ORANGE, ls="--", lw=1.1, label="Bounds from normal inputs only")
    ax.plot(cc / l10, (mu_hi + 1.0 * (cc - cJ)) / l10, color=ORANGE, ls="--", lw=1.1)
    ax.axvline(cJ / l10, color=INK2, lw=0.6, ls=":")
    ax.text(cJ / l10 + 0.08, 0.15, "largest budget", fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(18.5, 26)
    ax.set_ylim(0, 4.8)
    ax.set_xlabel(r"Training compute, $\log_{10}C$ (FLOP)")
    ax.set_ylabel(r"Tokens per parameter, $\log_{10}M$")
    ax.set_title("(a) Compute-optimal ratio beyond the design", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5, handlelength=1.6)

    ax = axes[1]
    s = pi["sign"]
    labels_short = {"Normal inputs only": "Normal inputs", "Slope in cross-sweep range": "Cross-sweep slope range",
                    "Cross-sweep range + M* nondecreasing": r"... + $M^*$ nondecreasing",
                    "Anchor's own slope persists (95% CI)": "Anchor's slope persists",
                    "Parametric extrapolation (point)": "Point extrapolation"}
    ylab, over, under, none = [], [], [], []
    for aname2, tag in [("Chinchilla (Epoch extraction)", "Chinchilla"), ("Meta Llama 3 (digitized)", "Llama 3")]:
        for r in s[s["anchor"] == aname2].itertuples():
            ylab.append(f"{tag}: {labels_short[r.assumption]}")
            over.append(r.share_over_identified)
            under.append(r.share_under_identified)
            none.append(r.share_not_identified)
    y = np.arange(len(ylab))[::-1]
    ax.barh(y, over, color=BLUE, height=0.7, label=r"$w>1$ identified")
    ax.barh(y, under, left=over, color=ORANGE, height=0.7, label=r"$w<1$ identified")
    ax.barh(y, none, left=np.array(over) + np.array(under), color=GRAY, height=0.7, label="Sign not identified")
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of models beyond the anchor")
    ax.set_title("(b) Is the sign of $w-1$ identified?", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), ncol=3, fontsize=6.5, handlelength=1.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    aer_style.savefig(fig, name)


def fig_singular(si_geom_curves, mc, name="ra5_theory_singular"):
    """(a) Noise-free on-path criterion along two rate directions (log-log): quartic vs quadratic.
    (b) Monte Carlo 75th percentile of |sigma*-hat - sigma*| against the noise s.d. (n_eff ~ tau^-2)."""
    aer_style.use()
    fig, axes = plt.subplots(1, 2, figsize=(aer_style.WIDTH_FULL, 2.7))
    ax = axes[0]
    for lab, (t, q), col in zip(["Rates move oppositely (1, $-1$)", "Rates move together (1, 1)"], si_geom_curves, [BLUE, ORANGE]):
        ax.loglog(t, q, "o-", color=col, ms=3.5, label=lab)
    ax.set_xlabel(r"Displacement $t$ of the decay rates")
    ax.set_ylabel("Profiled least-squares criterion")
    ax.set_title("(a) On-path criterion: quartic vs quadratic", loc="left")
    t = si_geom_curves[0][0]
    ax.text(t[3], si_geom_curves[0][1][3] * 1e3, "slope 4", fontsize=7, color=BLUE)
    ax.text(t[7], si_geom_curves[1][1][7] * 1e-3, "slope 2", fontsize=7, color=ORANGE)
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axes[1]
    for tech, ls in [("Hoffmann (alpha != beta)", "-"), ("equal exponents (alpha = beta)", "--")]:
        g = mc[mc["technology"] == tech].sort_values("tau")
        short = r"$\alpha\neq\beta$" if "!=" in tech else r"$\alpha=\beta$"
        ax.loglog(g["tau"], g["q75_unres"], ls, marker="o", ms=3.5, color=BLUE, label=f"Unrestricted, {short}")
        ax.loglog(g["tau"], g["q75_restricted"], ls, marker="s", ms=3.2, color=ORANGE, label=f"Path restriction imposed, {short}")
    ax.set_xlabel(r"Noise s.d. $\tau$ (effective sample size $\propto\tau^{-2}$)")
    ax.set_ylabel(r"75th pct. of $|\hat\sigma^*-\sigma^*|$")
    ax.set_title(r"(b) Rates: $n^{-1/4}$ vs $n^{-1/2}$", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.2, handlelength=3.2)
    fig.tight_layout()
    aer_style.savefig(fig, name)
