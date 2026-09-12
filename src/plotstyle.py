"""
Shared Nature-journal-like plotting style: clean sans-serif type, thin axes,
no chart-junk, colorblind-safe palette (Okabe & Ito, 2008), constrained
layout everywhere to avoid overlapping text/legends.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.constrained_layout.use": True,
    })


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)


def panel_label(ax, text):
    """Bold panel label (a, b, c...) in the top-left corner, Nature-style."""
    ax.text(-0.14, 1.08, text, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")
