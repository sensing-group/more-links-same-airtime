"""Shared figure style: every plot is drawn at its printed width in the body font."""
from pathlib import Path

import matplotlib as mpl

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
COLUMN_W_IN = 241.14749 / 72.27          # acmart sigconf \columnwidth
TEXT_W_IN = 506.29498 / 72.27            # acmart sigconf \textwidth
FONT_PT = 8
SMALL_PT = 7
OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
             "grey": "#8C8C8C"}


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertinus Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Linux Libertine O",
        "mathtext.it": "Linux Libertine O:italic",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": SMALL_PT,
        "xtick.labelsize": SMALL_PT,
        "ytick.labelsize": SMALL_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.color": "#DDDDDD",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "savefig.bbox": "standard",
    })


def save(fig, stem: str) -> None:
    """Save at the exact figure size (no tight bounding box, so the width stays the printed width)."""
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
