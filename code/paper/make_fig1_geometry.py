"""Figure 1 (version 3; round 3 marker order and x-range): the geometry of the training problem.

Panel (a) draws the paper's reference technology (Chinchilla, kappa free; Section IV and Online Appendix F) in the
(log10 N, log10 D) plane: isoquants at the frontier losses for C = 1e20, ..., 1e24 FLOP, isocosts 6ND = C, and the
compute-optimal expansion path. It marks the three SmolLM2 base models, a family with size-specific token budgets
(the paper's showcase in Section IV), at their total parameter counts (filled) and at their non-embedding counts
(hollow; Referee 2, round-1 minor 26 and round-2 minor 3), and joins SmolLM2 1.7B to the path at equal compute.

Panel (b) is unchanged from the v2 figure (code/analysis/m7_theory/figures.py::fig_geometry, panel b): three members of
the kappa family constructed to share the expansion path and the loss-compute frontier of the Besiroglu et al. (2024)
parameters, with sigma* = 0.60, 0.74 and 0.85. Section II quotes their excess loss at five times the compute-optimal
ratio (2.0, 4.1, 7.6 percent); those numbers are written to output/tables/fig1_geometry_numbers.csv (Referee 4,
round-2 minor 15: "no output file contains these numbers").

Inputs (read-only): data/processed/ra2_wedge/headline.json (the reference point estimate, key chin_q_point),
output/tables/ra2_wedge_models.csv (SmolLM2 parameter and token counts; the reference wedges it reports are
reproduced here to 1e-12). No shared library is edited. Deterministic; CPU only; runs in seconds.

Usage: .venv/bin/python code/paper/make_fig1_geometry.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis", "m7_theory"))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))

import matplotlib.pyplot as plt  # noqa: E402

import aer_style  # noqa: E402
from aer_style import AQUA, BLUE, INK2, MUTED, ORANGE  # noqa: E402
from common import PARAM_SETS, kappa_family_member, lnR, path_objects  # noqa: E402

LN10 = np.log(10.0)


def reference():
    """kappa-free Chinchilla reference: L = E + (A N^-alpha + B D^-beta)^kappa (ra2_wedge, techs.chin_q_wild point)."""
    h = json.load(open(os.path.join(ROOT, "data", "processed", "ra2_wedge", "headline.json")))
    p = h["chin_q_point"]
    return dict(A=float(np.exp(p["lnA"])), B=float(np.exp(p["lnB"])), alpha=float(p["alpha"]), beta=float(p["beta"]),
                kappa=float(p["q"]), E=float(p["E"]), lnG=float(p["lnG"]))


def smollm2():
    d = pd.read_csv(os.path.join(ROOT, "output", "tables", "ra2_wedge_models.csv"))
    d = d[(d.clean == True) & d.model.str.startswith("SmolLM2")].copy()  # noqa: E712
    return d.sort_values("N")[["model", "N", "N_nonemb", "D", "w_chin_q", "s_ref"]]


def main():
    ref = reference()
    A_, B_, al, be = ref["A"], ref["B"], ref["alpha"], ref["beta"]
    po = path_objects(A_, B_, al, be)          # inner (kappa = 1) path objects; kappa does not move the path
    assert abs(np.log(po["G"]) - ref["lnG"]) < 1e-9
    fam = smollm2()
    wedge = lambda N, D: al * A_ * N ** -al / (be * B_ * D ** -be)  # noqa: E731
    fam["w_check"] = wedge(fam.N.values, fam.D.values)
    assert np.max(np.abs(fam.w_check / fam.w_chin_q - 1)) < 1e-12

    aer_style.use()
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))  # AER text width is 5.33in: print at ~1:1 so labels stay >= 7pt

    # ------------------------------------------------------------------ (a) reference technology
    ax = axes[0]
    lnN = np.linspace(np.log(1e8), np.log(1e12), 400)
    lnD = np.linspace(np.log(1e9), np.log(1e14), 400)
    NN, DD = np.meshgrid(lnN, lnD)
    Q = A_ * np.exp(-al * NN) + B_ * np.exp(-be * DD)   # isoquants are level sets of the inner aggregate (ordinal)
    budgets = np.array([1e20, 1e21, 1e22, 1e23, 1e24])
    lev = np.sort(po["K"] * (budgets / 6) ** (-po["gamma"]))
    ax.contour(NN / LN10, DD / LN10, Q, levels=lev, colors=MUTED, linewidths=0.8)
    for C in budgets:
        lc = np.log(C / 6)
        ax.plot(lnN / LN10, (lc - lnN) / LN10, ls=":", color=INK2, lw=0.7)
    lcs = np.linspace(np.log(1e19 / 6), np.log(1e25 / 6), 50)
    ax.plot((np.log(po["G"]) + po["a"] * lcs) / LN10, (-np.log(po["G"]) + (1 - po["a"]) * lcs) / LN10, color=BLUE,
            lw=1.6, label="Expansion path (reference)")
    # SmolLM2: projection of the 1.7B model to the path at equal compute
    r = fam.iloc[-1]
    lc0 = np.log(r.N * r.D)                        # = ln(C/6): the isocost is n + d = lc0
    n_path = np.log(po["G"]) + po["a"] * lc0       # n*(C) = ln G + a ln(C/6)
    ax.plot([np.log10(r.N), n_path / LN10], [np.log10(r.D), (lc0 - n_path) / LN10], color=ORANGE, lw=1.2)
    # round 3 (numbers_framework #5): hollow non-embedding markers first, filled total-count markers on top (zorder),
    # so that the 1.7B model's filled marker, where the orange line starts, is visible
    ax.plot(np.log10(fam.N_nonemb), np.log10(fam.D), "o", mfc="white", mec=ORANGE, mew=1.0, ms=4.5, zorder=3,
            label="SmolLM2, non-embedding")
    ax.plot(np.log10(fam.N), np.log10(fam.D), "o", color=ORANGE, ms=4.5, zorder=4, label="SmolLM2, total parameters")
    for _, rr in fam.iterrows():
        tag = rr.model.replace("SmolLM2-", "")
        off = (5, 1) if tag == "135M" else (4, 3)
        ax.annotate(tag, xy=(np.log10(rr.N), np.log10(rr.D)), xytext=off, textcoords="offset points", fontsize=7,
                    color=INK2)
    mid = ((np.log10(r.N) + n_path / LN10) / 2, (np.log10(r.D) + (lc0 - n_path) / LN10) / 2)
    ax.annotate(f"equal-compute deviation\nfrom the path: $w={r.w_chin_q:.1f}$", xy=mid, xytext=(10.35, 13.35),
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.text(8.05, 10.72, "isocost $6ND=C$", fontsize=7, color=INK2, rotation=-38)
    ax.text(8.42, 13.62, "isoquant", fontsize=7, color=INK2)
    ax.set_xlim(7.85, 12)                          # round 3: the 135M hollow marker (log10 N = 8.03) is not clipped
    ax.set_ylim(9, 14)
    ax.set_xlabel(r"$\log_{10} N$ (parameters)")
    ax.set_ylabel(r"$\log_{10} D$ (tokens)")
    ax.set_title("(a) Isoquants, isocosts and the path", loc="left")
    ax.legend(loc="lower right", fontsize=7, framealpha=1.0, frameon=True, edgecolor="white")

    # ------------------------------------------------------------------ (b) unchanged v2 construction (Besiroglu)
    ax = axes[1]
    BES = PARAM_SETS["Besiroglu"]
    pb = path_objects(BES["A"], BES["B"], BES["alpha"], BES["beta"])
    a, gam, G, K = pb["a"], pb["gamma"], pb["G"], pb["K"]
    lc = np.log(1e22 / 6)
    n_s = np.log(G) + a * lc
    x = np.linspace(-np.log(200), np.log(200), 400)  # ln(M/M*(C)) at fixed compute
    lnRstar = np.log(K) - gam * lc
    rows = []
    for S, col in ((2 * (1 / 0.60 - 1), ORANGE), (BES["alpha"] + BES["beta"], BLUE), (2 * (1 / 0.85 - 1), AQUA)):
        m = kappa_family_member(S, a, gam, G, K)
        pen = lnR(n_s - x / 2, lc - n_s + x / 2, m["A"], m["B"], m["a1"], m["b1"], m["kappa"]) - lnRstar
        ax.plot(x / LN10, 100 * (np.exp(pen) - 1), color=col, label=rf"$\sigma^*={m['sigma_star']:.2f}$")
        for mult in (5.0, 10.0, 0.2):
            off = lnR(n_s - np.log(mult) / 2, lc - n_s + np.log(mult) / 2, m["A"], m["B"], m["a1"], m["b1"], m["kappa"])
            rows.append(dict(panel="b", object="pct_reducible_loss_above_frontier", sigma_star=m["sigma_star"],
                             kappa=m["kappa"], C=1e22, M_over_Mstar=mult,
                             value=100 * (np.exp(off - lnRstar) - 1)))
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"$\log_{10}(M/M^*(C))$ at fixed compute")
    ax.set_ylabel("Reducible loss above frontier (%)")
    ax.set_title("(b) Same path, different curvature", loc="left")
    ax.legend(loc="upper center", fontsize=7)
    ax.set_ylim(0, 60)
    fig.tight_layout()
    aer_style.savefig(fig, "fig1_geometry")

    # ------------------------------------------------------------------ numbers quoted in the text and notes
    for _, rr in fam.iterrows():
        C = 6 * rr.N * rr.D
        Mstar = po["G"] ** -2 * (C / 6) ** (1 - 2 * po["a"])
        for obj, val in (("C", C), ("M", rr.D / rr.N), ("Mstar_ref", Mstar), ("M_over_Mstar", rr.D / rr.N / Mstar),
                         ("w_ref", rr.w_chin_q), ("s_ref", rr.s_ref), ("N_total", rr.N), ("N_nonemb", rr.N_nonemb),
                         ("w_ref_nonemb_units", wedge(rr.N_nonemb, rr.D))):
            rows.append(dict(panel="a", object=f"{rr.model}:{obj}", value=val))
    S_ref = al + be
    for obj, val in (("sigma_star_ref", 2 / (2 + S_ref)), ("k_ref=1/sigma*-1", S_ref / 2),
                     ("sigma_min_over_mixes", min(1 / (1 + al), 1 / (1 + be))),
                     ("sigma_max_over_mixes", max(1 / (1 + al), 1 / (1 + be))),
                     ("a_ref", po["a"]), ("gamma_ref(kappa*alpha*beta/S)", ref["kappa"] * al * be / S_ref),
                     ("Gopher_w_ref(280B,300B)", wedge(280e9, 300e9))):
        rows.append(dict(panel="text", object=obj, value=val))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "output", "tables", "fig1_geometry_numbers.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
