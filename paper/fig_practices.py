#!/usr/bin/env python3
"""Threshold-detector J against the number of pairs on the office data, with
the original practice (raw variance, threshold fit on the scored data) and
with sound practice (per-pair standardization, threshold from held-out
sessions; max and vote rules)."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
FIG_H_IN = 1.7
SERIES = (("paper", "Raw variance, threshold set on the scored data", "vermilion", "o", "--"),
          ("z_max_cv", "Standardized, threshold from other sessions (max rule)", "blue", "s", "-"),
          ("vote_cv", "Standardized, threshold from other sessions (vote rule)", "green", "^", "-"))


def main() -> None:
    ps.apply()
    by_k = json.loads((ANALYSIS_DIR / "threshold_practices.json").read_text())["by_k"]
    ks = sorted(int(k) for k in by_k)
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN), layout="constrained")
    for key, label, colour, marker, style in SERIES:
        ax.plot(ks, [by_k[str(k)][key] for k in ks], marker=marker, linestyle=style,
                color=ps.OKABE_ITO[colour], label=label)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100], labels=["1", "10", "100"])
    ax.set_xlabel("Pairs (office mesh)")
    ax.set_ylabel("Threshold $J$")
    ax.set_ylim(0, 0.45)
    fig.legend(loc="outside upper center", ncol=1)
    ps.save(fig, "fig_practices")


if __name__ == "__main__":
    main()
