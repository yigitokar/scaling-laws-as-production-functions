"""Shared figure style for the paper (print, light background, AER look).

Rules (from the project's dataviz checklist):
  - At most 3 categorical colors per panel (validated all-pairs CVD-safe): BLUE, ORANGE, AQUA.
    More series -> facet into small multiples or use gray + direct labels.
  - Sequential magnitude -> one hue light->dark (use SEQ cmap). Never rainbow.
  - One y-axis per panel (never twin axes). Thin marks: 1.2-1.5pt lines, markers >= 3.5pt.
  - Text is black/gray; color only on marks. Legends always present for >= 2 series.
  - Save vector PDF (for LaTeX) + 200-dpi PNG (for previews): use savefig(fig, "name").
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGDIR = os.path.join(ROOT, "output", "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CAT = [BLUE, ORANGE, AQUA]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#dbe8f8", "#8bb4e8", "#2a78d6", "#123f78"])
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ec", BLUE])

WIDTH_FULL, WIDTH_HALF = 6.5, 3.2  # inches (AER text width ~6.5in)


def use():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "lines.markersize": 4, "axes.prop_cycle": mpl.cycler(color=CAT),
        "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def savefig(fig, name, folder=FIGDIR):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name + ".pdf"))
    fig.savefig(os.path.join(folder, name + ".png"), dpi=200)
    plt.close(fig)
