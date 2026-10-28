#!/usr/bin/env python3
"""Full-width figure: every allocation of a fixed budget, for each decision window.

Columns: office occupancy, trial occupancy, trial motion. Top row: classifier
AUC; bottom row: J of the label-free vote rule. A line ends where a link would
receive fewer than two soundings per decision.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
FIG_H_IN = 3.0
OFFICE_BUDGET = 2.0          # pair-soundings per second (office base 1 Hz per pair)
TRIAL_BUDGET = 8.0           # pair-soundings per second (trial base 8 Hz per pair)
OFFICE_WINDOWS = (5, 10, 20, 30, 60)
TRIAL_WINDOWS = (1, 2, 5, 10)
COLOURS = ("vermilion", "orange", "green", "blue", "purple")
CHANCE = 0.5
J_CHANCE = 0.0              # J of a detector whose alarms carry no information
J_MIN = -0.05                # held-out J can fall slightly below chance (office, 60 s)


def office_curve(w: int, metric: str):
    cells = json.loads((ANALYSIS_DIR / f"decision_windows/decision_window_ablation_w{w}s.json").read_text())["windows"][f"w{w}s"]["cells"]
    return diagonal(cells, OFFICE_BUDGET, metric)


def trial_curve(task: str, w: int, metric: str):
    cells = json.loads((ANALYSIS_DIR / "scripted_trial_decision_windows.json").read_text())["results"][f"{task}|{w}"]
    return diagonal(cells, TRIAL_BUDGET, metric)


def diagonal(cells: dict, budget: float, metric: str):
    pts = []
    for key, v in cells.items():
        k, f = key.split("|")
        if v and abs(int(k) * float(f) - budget) < 1e-9:
            pts.append((int(k), v[metric]))
    return sorted(pts)


def main() -> None:
    ps.apply()
    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXT_W_IN, FIG_H_IN), sharex="col", layout="constrained")
    panels = [("Office, occupancy", lambda w, m: office_curve(w, m), OFFICE_WINDOWS, "upper left"),
              ("Trials, occupancy", lambda w, m: trial_curve("occupancy", w, m), TRIAL_WINDOWS, "lower right"),
              ("Trials, motion", lambda w, m: trial_curve("motion", w, m), TRIAL_WINDOWS, "lower right")]
    for col, (title, curve, windows, legend_loc) in enumerate(panels):
        for w, colour in zip(windows, COLOURS):
            for row, metric in enumerate(("auc", "j_vote")):
                pts = curve(w, metric)
                if len(pts) < 2:
                    continue
                k, v = zip(*pts)
                axes[row, col].plot(k, v, marker="o", color=ps.OKABE_ITO[colour], label=f"{w} s")
        axes[0, col].set_title(title, loc="left")
        axes[0, col].axhline(CHANCE, color=ps.OKABE_ITO["grey"], linestyle=":", linewidth=0.8)
        axes[0, col].set_ylim(CHANCE, 1.0)
        axes[1, col].axhline(J_CHANCE, color=ps.OKABE_ITO["grey"], linestyle=":", linewidth=0.8)
        axes[1, col].set_ylim(J_MIN, 0.9)
        axes[1, col].set_xscale("log", base=2)
        axes[1, col].set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
        axes[1, col].set_xlabel("Links sharing the budget")
        axes[0, col].legend(title="Decision", loc=legend_loc, ncol=2, fontsize=ps.SMALL_PT,
                            title_fontsize=ps.SMALL_PT, handlelength=1.2, columnspacing=0.8)
    for row, level in enumerate((CHANCE, J_CHANCE)):
        axes[row, 0].text(1, level + 0.01, "chance", color=ps.OKABE_ITO["grey"], ha="left", va="bottom",
                          fontsize=ps.SMALL_PT)
    axes[0, 0].set_ylabel("Classifier AUC")
    axes[1, 0].set_ylabel("Vote rule $J$")
    ps.save(fig, "fig_budget_grid")


if __name__ == "__main__":
    main()
