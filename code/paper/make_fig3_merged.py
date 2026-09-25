"""Figure 3 of the paper (version 3): the elasticity of substitution by design, by budget, beyond the designs, and
in our controlled experiment (placeholder panel until the m9 results are final).

Reads only reviewed module outputs (no estimation here):
  (a) ra1_modelfree_isoflop_summary.csv (model-free, RE mean over budgets), ra1_modelfree_isoflop_param.csv (same-run
      parametric fits), rb1_sigmaC_conventions.csv (Farseer first-derivative path estimate and its path-mean s.e.),
      m2_table3_technology.csv (Farseer parametric fits, Huber), rb1_sigmaC_study_level.csv (study-level summary and
      the range of its variants).
  (b) rb1_sigmaC_budgets_input.csv (budget-level sigma*_b), rb1_sigmaC_metareg_predictions.csv (pooled lines, CR2 band).
  (c) rb1_sigmaC_pi_sigmaC.csv (identified set beyond the designs), data/processed/rb2_decisions/clean_models.csv
      (training compute of the clean sample, rug).
  (d) empty: TBD-m9.
[rb4 integration, 2026-09-24] Chinchilla in panel (a) follows rb1's switch RB1_CHINCHILLA (default 'rb4'): model-free
  from rb4's override summary (data/processed/rb4_chinflop/override_new_isoflop_summary.csv, N_F with FLOPs per token
  rebuilt from Hoffmann et al.'s architecture table) and same-run parametric fits from rb4_chinflop_param.csv (row
  'N_F (T4)'); with RB1_CHINCHILLA=ra1, ra1's total-N values. Panels (b)-(c) read rb1's primary CSVs, which carry the
  same switch.
[round 3, T10, 2026-09-25] Panel (c) reads module rb5_sigma's identified set (output/tables/rb5_sigma_pi_set.csv):
  the drift-agnostic upper line at 0.70 (dashed) beside the top-budget value, the Chinchilla-Llama 3 drift labelled,
  the values at 1e24 marked, shading from 1e21 with a tick at 3e21; the pooled-drift lower line is no longer drawn.
  Panel (d) already has panel (b)'s vertical scale (0.40-1.07). Option --out NAME writes NAME.{pdf,png} instead.
Writes output/figures/fig3_merged.{pdf,png}. Usage: .venv/bin/python code/paper/make_fig3_merged.py [--out NAME]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
import aer_style as st  # noqa: E402

T = os.path.join(ROOT, "output", "tables")
T5 = os.environ.get("RB5_TABLES", T)          # [round 3] module rb5_sigma's tables (panel c)
FIG_FOLDER = os.environ.get("FIG3_FOLDER", st.FIGDIR)
CHIN_REBUILT = os.environ.get("RB1_CHINCHILLA", "rb4").strip().lower() == "rb4"
BLUE, ORANGE, AQUA = st.BLUE, st.ORANGE, st.AQUA
INK, INK2, MUTED, GRID = st.INK, st.INK2, st.MUTED, st.GRID


def panel_a(ax):
    summ = pd.read_csv(os.path.join(T, "ra1_modelfree_isoflop_summary.csv")).set_index("design")
    par = pd.read_csv(os.path.join(T, "ra1_modelfree_isoflop_param.csv")).set_index("design")
    if CHIN_REBUILT:   # [rb4 integration] Chinchilla in N_F (rebuilt FLOPs per token), as in rb1's primary outputs
        o = pd.read_csv(os.path.join(ROOT, "data", "processed", "rb4_chinflop", "override_new_isoflop_summary.csv"))
        o = o.set_index("design")
        for c in ("sigma_re", "se_sigma_re"):
            summ.loc["Chinchilla", c] = o.loc["Chinchilla", c]
        q = pd.read_csv(os.path.join(T, "rb4_chinflop_param.csv")).set_index("convention").loc["N_F (T4)"]
        for c in ("sigma_kappa", "se_sigma_kappa", "sigma_chin", "se_sigma_chin"):
            par.loc["Chinchilla", c] = q[c]
    conv = pd.read_csv(os.path.join(T, "rb1_sigmaC_conventions.csv"))
    far_fd = conv[(conv.design == "Farseer") & (conv.convention == "non-embedding N, first-derivative path")].iloc[0]
    m2 = pd.read_csv(os.path.join(T, "m2_table3_technology.csv"))
    far_p = m2[(m2.key == "farseer|all") & (m2.estimator == "huber")].iloc[0]
    study = pd.read_csv(os.path.join(T, "rb1_sigmaC_study_level.csv"))
    head = study[study.variant.str.startswith("PRIMARY")].iloc[0]
    rng = (study.mean_re.min(), study.mean_re.max())
    # [round 3, WP2, T10] the gray band spans every study-level variant, including Chinchilla's windows, samples and
    # memberships propagated by module rb5_sigma (T1.5): 0.663-0.702 (rb5_sigma_study_ranges.csv, family "All: ...").
    rr = os.path.join(T5, "rb5_sigma_study_ranges.csv")
    if os.path.exists(rr):
        r5 = pd.read_csv(rr)
        r5 = r5[r5.family.str.startswith("All: ")]
        if len(r5):
            rng = (min(rng[0], float(r5.mean_min.iloc[0])), max(rng[1], float(r5.mean_max.iloc[0])))

    rows = ["Chinchilla", "Llama 3", "Marin, Comma", "Marin, DCLM", "Marin, Nemotron-CC", "Farseer",
            "Porian, RefinedWeb", "Porian, OpenWebText2"]
    labels = {"Farseer": "Farseer (path)", "Porian, RefinedWeb": "Porian, RefinedWeb*",
              "Porian, OpenWebText2": "Porian, OWT2*"}
    ypos = {r: i for i, r in enumerate(rows)}
    y_sum = len(rows) + 0.35
    for r in rows:
        y = ypos[r]
        if r == "Farseer":
            mf, mf_se = far_fd.sigma, far_fd.se
            kf, kf_se = far_p.sigma_star_q, far_p.se_sigma_star_q
            k1, k1_se = far_p.sigma_star, far_p.se_sigma_star
        else:
            mf, mf_se = summ.loc[r, "sigma_re"], summ.loc[r, "se_sigma_re"]
            kf, kf_se = par.loc[r, "sigma_kappa"], par.loc[r, "se_sigma_kappa"]
            k1, k1_se = par.loc[r, "sigma_chin"], par.loc[r, "se_sigma_chin"]
        por = r.startswith("Porian")
        for val, se, off, col, mk in ((mf, mf_se, -0.22, BLUE, "o"), (kf, kf_se, 0.0, AQUA, "D"),
                                      (k1, k1_se, 0.22, ORANGE, "s")):
            lo, hi = val - 1.96 * se, val + 1.96 * se
            ax.hlines(y + off, max(lo, 0.25), min(hi, 0.95), color=col, lw=1.1, alpha=0.55 if por else 1.0)
            ax.plot(val, y + off, marker=mk, ms=4.6 if mk == "o" else 4.0, color=col,
                    mfc="white" if por else col, mec=col, mew=0.9, ls="none", zorder=3)
            if lo < 0.25:
                ax.annotate("", xy=(0.25, y + off), xytext=(0.27, y + off),
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    # study-level summary (one estimate per study) and the range of variant means
    ax.axhline(len(rows) - 0.45, color=INK2, lw=0.6)
    ax.fill_betweenx([y_sum - 0.28, y_sum + 0.28], rng[0], rng[1], color=MUTED, alpha=0.45, lw=0, zorder=1)
    ax.hlines(y_sum, head.lo_hksj, head.hi_hksj, color=INK, lw=1.4, zorder=2)
    ax.plot(head.mean_re, y_sum, marker="D", ms=5.2, color=INK, ls="none", zorder=3)
    ax.set_yticks(list(range(len(rows))) + [y_sum])
    ax.set_yticklabels([labels.get(r, r) for r in rows] + ["Study-level mean"])
    ax.set_ylim(len(rows) + 1.0, -0.7)
    ax.set_xlim(0.25, 0.95)
    ax.set_xlabel(r"$\sigma^*$ (95 percent interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) By design", loc="left")
    h = [Line2D([], [], ls="none", marker="o", color=BLUE, label="Model-free"),
         Line2D([], [], ls="none", marker="D", color=AQUA, label=r"$\kappa$ free"),
         Line2D([], [], ls="none", marker="s", color=ORANGE, label=r"$\kappa=1$"),
         Patch(color=MUTED, alpha=0.45, label="Variant range")]
    ax.legend(handles=h, loc="upper left", fontsize=6.8, handlelength=1.0, borderaxespad=0.2, labelspacing=0.3,
              handletextpad=0.4)
    return head, rng


def panel_b(ax):
    d = pd.read_csv(os.path.join(T, "rb1_sigmaC_budgets_input.csv"))
    d = d[~d.porian.astype(bool)]
    pr = pd.read_csv(os.path.join(T, "rb1_sigmaC_metareg_predictions.csv"))
    style = {"Chinchilla": (BLUE, "o", "Chinchilla"), "Llama 3": (ORANGE, "s", "Llama 3"),
             "Marin, Comma": (AQUA, "^", "Marin (3 corpora)"), "Marin, DCLM": (AQUA, "^", None),
             "Marin, Nemotron-CC": (AQUA, "^", None), "Farseer (local path)": (INK2, "D", "Farseer local path")}
    jit = {"Chinchilla": -0.045, "Llama 3": 0.045, "Marin, Comma": -0.03, "Marin, DCLM": 0.0,
           "Marin, Nemotron-CC": 0.03, "Farseer (local path)": 0.0}
    for des, g in d.groupby("design"):
        col, mk, _ = style[des]
        x = g.log10C.values + jit[des]
        y, se = g.y.values, g.se.values
        ax.vlines(x, np.clip(y - 1.96 * se, 0.40, None), np.clip(y + 1.96 * se, None, 0.93), color=col, lw=0.8,
                  alpha=0.5, zorder=2)
        ax.plot(x, y, ls="none", marker=mk, ms=4.2, mfc="white" if des.startswith("Farseer") else col, mec=col,
                mew=0.9, zorder=3)
    w = pr[pr.spec_key == "weighted"].sort_values("log10C")
    u = pr[pr.spec_key == "unweighted"].sort_values("log10C")
    cc = np.linspace(18.5, 21.5, 61)
    ax.fill_between(cc, np.interp(cc, w.log10C, w.lo_cr2), np.interp(cc, w.log10C, w.hi_cr2), color=GRID, alpha=0.95,
                    lw=0, zorder=1)
    ax.plot(cc, np.interp(cc, w.log10C, w.est), color=INK, lw=1.4, zorder=4)
    ax.plot(cc, np.interp(cc, u.log10C, u.est), color=INK, lw=1.1, ls=":", zorder=4)
    sl = pd.read_csv(os.path.join(T, "rb1_sigmaC_metareg_slopes.csv"))
    bw = sl[(sl.spec_key == "weighted") & sl["sample"].str.startswith("all six")].slope.iloc[0]
    bu = sl[(sl.spec_key == "unweighted") & sl["sample"].str.startswith("all six")].slope.iloc[0]
    ax.set_xlim(18.3, 21.7)
    ax.set_ylim(0.40, 1.07)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel(r"$\log_{10}$ compute (FLOP)")
    ax.set_ylabel(r"Model-free $\sigma^*_b$ by budget")
    ax.set_title("(b) By budget, pooled drift", loc="left")
    h = [Line2D([], [], ls="none", marker=style[k][1], color=style[k][0],
                mfc="white" if k.startswith("Farseer") else style[k][0], label=style[k][2])
         for k in ("Chinchilla", "Llama 3", "Marin, Comma", "Farseer (local path)")]
    h += [Line2D([], [], color=INK, lw=1.4, label=f"Pooled ({bw:+.3f} per decade)"),
          Line2D([], [], color=INK, lw=1.1, ls=":", label=f"Unweighted ({bu:+.3f})")]
    ax.legend(handles=h, loc="upper center", fontsize=6.5, handlelength=1.2, ncol=2, columnspacing=0.6,
              labelspacing=0.25, borderaxespad=0.15)
    return bw, bu


def panel_c(ax):
    """[round 3, T10] Identified set beyond the designs from module rb5_sigma (rb5_sigma_pi_set.csv): lower bound the
    linear continuation of the Chinchilla-Llama 3 drift from the top-budget value placed at 1e21; upper bound the
    top-budget value if the decline at the top budgets is real, and 0.70 (the level where the designs overlap) if one is
    agnostic about it; values at 1e24 marked; Chinchilla's largest bracketed budget (3e21) ticked."""
    p = pd.read_csv(os.path.join(T5, "rb5_sigma_pi_set.csv"))
    p = p[p.anchor == "1e21 (rb1)"].sort_values("log10C")
    s_top, b = float(p.sigma_top.iloc[0]), float(p.drift.iloc[0])
    up_ag = float(p.upper_drift_agnostic.iloc[0])
    xx = np.linspace(21.0, 26.0, 101)
    lo = np.clip(s_top + b * (xx - 21.0), 0.0, 1.0)
    ax.fill_between(xx, lo, s_top, color=BLUE, alpha=0.20, lw=0, zorder=1)
    ax.fill_between(xx, s_top, up_ag, color=BLUE, alpha=0.08, lw=0, zorder=1)
    ax.plot(xx, lo, color=BLUE, lw=1.3, zorder=3)
    ax.plot(xx, np.full_like(xx, s_top), color=INK, lw=1.3, zorder=3)
    ax.plot(xx, np.full_like(xx, up_ag), color=INK2, lw=1.1, ls="--", zorder=3)
    ax.axvspan(18.5, 21.0, color=GRID, alpha=0.6, lw=0)
    ax.text(19.75, 0.47, "designs", ha="center", va="center", fontsize=7.0, color=INK2)
    x3 = np.log10(3e21)
    ax.plot([x3, x3], [0.25, 0.345], color=INK2, lw=0.8, ls=":", zorder=2)
    ax.text(x3 + 0.08, 0.335, r"$3\times10^{21}$", fontsize=6.3, color=INK2, va="bottom", ha="left")
    lo24 = float(np.clip(s_top + b * 3.0, 0.0, 1.0))
    for v, c in ((lo24, BLUE), (s_top, INK), (up_ag, INK2)):
        ax.plot(24.0, v, marker="o", ms=3.6, color=c, mec="white", mew=0.5, zorder=4)
        ax.text(24.0, v + 0.014, f"{v:.2f}", fontsize=6.4, color=c, va="bottom", ha="center",
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85), zorder=5)
    ax.text(24.0, 0.30, r"$10^{24}$", fontsize=6.3, color=INK2, ha="center", va="bottom")
    clean = pd.read_csv(os.path.join(ROOT, "data", "processed", "rb2_decisions", "clean_models.csv"))
    lc = np.log10(clean.Cmp.values)
    ax.plot(lc, np.full(len(lc), 0.275), ls="none", marker="|", ms=8, color=INK2, alpha=0.7, mew=0.8)
    ax.text(18.6, 0.29, "clean-sample\nmodels", fontsize=6.5, color=INK2, va="bottom", linespacing=0.95)
    ax.set_xlim(18.5, 26.0)
    ax.set_ylim(0.25, 1.12)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel(r"$\log_{10}$ compute (FLOP)")
    ax.set_ylabel(r"$\sigma^*(C)$")
    ax.set_title("(c) Beyond the designs", loc="left")
    h = [Patch(color=BLUE, alpha=0.20, label="Identified set, decline real"),
         Patch(color=BLUE, alpha=0.08, label="Added if drift-agnostic"),
         Line2D([], [], color=INK, lw=1.3, label=f"Upper: largest budgets ({s_top:.2f})"),
         Line2D([], [], color=INK2, lw=1.1, ls="--", label=f"Upper, drift-agnostic ({up_ag:.2f})"),
         Line2D([], [], color=BLUE, lw=1.3, label=f"Chinchilla–Llama 3 drift ({b:+.3f} per decade)".replace("-", "\u2212"))]
    ax.legend(handles=h, loc="upper right", fontsize=6.0, handlelength=1.5, labelspacing=0.2, borderaxespad=0.2)


def panel_d(ax):
    ax.set_xlim(14.0, 17.5)
    ax.set_ylim(0.40, 1.07)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel(r"$\log_{10}$ compute (FLOP)")
    ax.set_ylabel(r"Model-free $\sigma^*_b$ by budget")
    ax.set_title("(d) Controlled experiment", loc="left")
    ax.fill_between([14.0, 17.5], 0.40, 1.07, facecolor="none", edgecolor=MUTED, hatch="////", lw=0, alpha=0.35)
    ax.text(15.75, 0.72, "[TBD-m9]\nResults pending\n\nFineWeb-Edu and FineWeb,\nFLOP-effective parameters",
            ha="center", va="center", fontsize=7.5, color="#b0211d",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b0211d", lw=0.8))


def main(out="fig3_merged"):
    st.use()
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.0})
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.5),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], height_ratios=[1.12, 1.0]))
    head, rng = panel_a(axes[0, 0])
    bw, bu = panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    st.savefig(fig, out, folder=FIG_FOLDER)
    print(f"study-level {head.mean_re:.3f} [{head.lo_hksj:.3f}, {head.hi_hksj:.3f}]; variant range "
          f"{rng[0]:.3f}-{rng[1]:.3f}; pooled slopes {bw:+.4f} / {bu:+.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig3_merged", help="output name (default: the paper's fig3_merged)")
    main(ap.parse_args().out)
