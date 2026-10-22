#!/usr/bin/env python3
"""How much a further sounding repeats: correlation of a pair with itself
against the lag, and of two different pairs at the same instant (scripted
trials, occupied phases)."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
FIG_H_IN = 1.55
PHASES = (("occupied_static", "Seated", "green"), ("occupied_moving", "Walking", "vermilion"))


def main() -> None:
    ps.apply()
    d = json.loads((ANALYSIS_DIR / "sounding_correlation.json").read_text())["phases"]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN), layout="constrained")
    for key, label, colour in PHASES:
        lags = sorted(d[key]["within_pair_median_abs_corr"], key=float)
        ax.plot([float(x) for x in lags], [d[key]["within_pair_median_abs_corr"][x] for x in lags],
                marker="o", color=ps.OKABE_ITO[colour], label=f"Same pair, {label.lower()}")
    across = [v for key, _, _ in PHASES for v in d[key]["across_pairs_median_abs_corr"].values()]
    level = sum(across) / len(across)
    ax.axhline(level, color=ps.OKABE_ITO["black"], linestyle="--", linewidth=0.8)
    ax.text(2.0, level + 0.02, "two different pairs, same instant", ha="right", va="bottom",
            fontsize=ps.SMALL_PT, color=ps.OKABE_ITO["black"])
    ax.axhline(0, color=ps.OKABE_ITO["grey"], linewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2], labels=["0.125", "0.25", "0.5", "1", "2"])
    ax.set_xlabel("Time between soundings of a pair (s)")
    ax.set_ylabel("Absolute correlation")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="lower left")
    ps.save(fig, "fig_redundancy")


if __name__ == "__main__":
    main()
