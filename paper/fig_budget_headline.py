#!/usr/bin/env python3
"""Headline figure: detection at a fixed airtime budget, spent on few fast or many slow pairs.

Reads the equal-budget diagonals from the analysis outputs (ANALYSIS_DIR) and
draws AUC against the number of pairs that share the budget.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
FIG_H_IN = 1.9
WIDE_W_IN = 0.62 * ps.TEXT_W_IN          # right half of Fig. 2 (minipage width)
WIDE_H_IN = 1.6
CHANCE = 0.5
# (label, json file, result key or window, budget in pair-soundings per second, colour, marker,
#  label anchor (pairs, AUC), horizontal alignment)
SERIES = [
    ("Office, occupancy, 20 s", "decision_windows/decision_window_ablation_w20s.json", "w20s", 2.0, "blue", "o", (2.1, 0.635), "left"),
    ("Trials, occupancy, 2 s", "scripted_trial_decision_windows.json", "occupancy|2", 8.0, "green", "s", (1.06, 0.68), "left"),
    ("Trials, motion, 1 s", "scripted_trial_decision_windows.json", "motion|1", 8.0, "vermilion", "^", (1.0, 0.94), "left"),
]


def diagonal(path: Path, key: str, budget: float):
    d = json.loads(path.read_text())
    cells = d["windows"][key]["cells"] if "windows" in d else d["results"][key]
    pts = []
    for cell, v in cells.items():
        k, f = cell.split("|")
        if v and abs(int(k) * float(f) - budget) < 1e-9:
            pts.append((int(k), v["auc"]))
    return sorted(pts)


def draw(width_in: float, height_in: float, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(width_in, height_in), layout="constrained")
    for label, fname, key, budget, colour, marker, (lx, ly), ha in SERIES:
        pts = diagonal(ANALYSIS_DIR / fname, key, budget)
        k, auc = zip(*pts)
        ax.plot(k, auc, marker=marker, color=ps.OKABE_ITO[colour])
        ax.text(lx, ly, label, color=ps.OKABE_ITO[colour], ha=ha, va="center", fontsize=ps.SMALL_PT)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    ax.set_xlabel("Pairs sharing the same airtime")
    ax.set_ylabel("Detection AUC")
    ax.axhline(CHANCE, color=ps.OKABE_ITO["grey"], linestyle=":", linewidth=0.8)
    ax.text(16, CHANCE + 0.01, "chance", color=ps.OKABE_ITO["grey"], ha="right", va="bottom", fontsize=ps.SMALL_PT)
    ax.set_ylim(CHANCE, 1.0)
    ps.save(fig, stem)


def main() -> None:
    ps.apply()
    draw(ps.COLUMN_W_IN, FIG_H_IN, "fig_budget_headline")
    draw(WIDE_W_IN, WIDE_H_IN, "fig_budget_wide")


if __name__ == "__main__":
    main()
