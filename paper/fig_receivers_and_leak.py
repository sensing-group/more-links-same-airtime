#!/usr/bin/env python3
"""Two single-column figures from the analysis outputs (ANALYSIS_DIR).

fig_receivers: at the same airtime, one sensing pair against one transmitter
heard by all its receivers (scripted trials, per-pair and per-transmitter runs).
fig_schedule_leak: availability of pairs in occupied and empty windows, and the
AUC of the schedule alone against the CSI features (office mesh).
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
FIG_H_IN = 1.6
BUDGET_HZ = 2.0                         # soundings per second for the receivers comparison
# (task, decision window s, label)
RECEIVER_CASES = [("occupancy", 1, "Occupancy\n1 s"), ("motion", 1, "Motion\n1 s"),
                  ("occupancy", 5, "Occupancy\n5 s"), ("motion", 5, "Motion\n5 s")]
BAR_W = 0.36
CHANCE = 0.5                            # AUC of a detector that carries no information


def receivers() -> None:
    per_pair = json.loads((ANALYSIS_DIR / "scripted_trial_decision_windows.json").read_text())["results"]
    per_tx = json.loads((ANALYSIS_DIR / "scripted_trial_decision_windows_tx.json").read_text())["results"]
    key = f"1|{BUDGET_HZ:g}"
    pair = [per_pair[f"{t}|{w:g}"][key]["auc"] for t, w, _ in RECEIVER_CASES]
    tx = [per_tx[f"{t}|{w:g}"][key]["auc"] for t, w, _ in RECEIVER_CASES]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN), layout="constrained")
    x = np.arange(len(RECEIVER_CASES))
    ax.bar(x - BAR_W / 2, pair, BAR_W, color=ps.OKABE_ITO["grey"], label="One pair")
    ax.bar(x + BAR_W / 2, tx, BAR_W, color=ps.OKABE_ITO["blue"], label="One transmitter, all receivers")
    for xi, v in zip(x + BAR_W / 2, tx):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=ps.SMALL_PT)
    for xi, v in zip(x - BAR_W / 2, pair):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=ps.SMALL_PT)
    ax.set_xticks(x, [c[2] for c in RECEIVER_CASES])
    ax.set_ylabel("Detection AUC")
    ax.set_ylim(0, 1.04)
    ax.axhline(CHANCE, color=ps.OKABE_ITO["grey"], linestyle=":", linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    fig.legend(loc="outside upper center", ncol=2)
    ps.save(fig, "fig_receivers")


def schedule_leak() -> None:
    d = json.loads((ANALYSIS_DIR / "schedule_leak.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(ps.COLUMN_W_IN, FIG_H_IN), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    vals = [d["availability_empty"], d["availability_occupied"]]
    a1.bar([0, 1], vals, 0.6, color=[ps.OKABE_ITO["grey"], ps.OKABE_ITO["vermilion"]])
    for xi, v in enumerate(vals):
        a1.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=ps.SMALL_PT)
    a1.set_xticks([0, 1], ["Empty", "Occupied"])
    a1.set_ylabel("Pairs sounded enough")
    a1.set_ylim(0, 0.62)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_title("(a) Schedule", loc="left")
    a1.grid(axis="x", visible=False)
    names = ["Pair count\n(in session)", "Schedule\npattern", "CSI\nfeatures"]
    aucs = [float(np.median(d["available_pair_count_auc_within_session"])),
            d["pattern_only_auc_session_grouped"], d["csi_auc_session_grouped"]]
    cols = [ps.OKABE_ITO["green"], ps.OKABE_ITO["green"], ps.OKABE_ITO["blue"]]
    a2.bar(range(3), aucs, 0.6, color=cols)
    for xi, v in enumerate(aucs):
        a2.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=ps.SMALL_PT)
    a2.set_xticks(range(3), names)
    a2.set_ylabel("Occupancy AUC")
    a2.set_ylim(0, 1.0)
    a2.axhline(CHANCE, color=ps.OKABE_ITO["grey"], linestyle=":", linewidth=0.8, zorder=0)
    a2.set_title("(b) Schedule against CSI", loc="left")
    a2.grid(axis="x", visible=False)
    ps.save(fig, "fig_schedule_leak")


def main() -> None:
    ps.apply()
    receivers()
    schedule_leak()


if __name__ == "__main__":
    main()
