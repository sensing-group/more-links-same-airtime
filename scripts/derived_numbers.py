#!/usr/bin/env python3
"""Numbers the paper derives from other analysis outputs, recomputed from those outputs.

- 95% intervals of the mean gain from spreading (Section 5): percentile bootstrap over the six
  held-out trials (trials) or over the random two-pair subsets (office).
- Cover cost of hiding the adaptive schedule (Section 7): the ratio of occupied to empty
  availability in the sessions where occupied minutes have more available pairs.
- Lag at which a pair's own earlier sounding stops repeating it more than another pair does
  (Section 5): linear interpolation of the within-pair correlation against the across-pair level.

Reads ANALYSIS_DIR (default results/), writes OUT_PATH (default work/derived_numbers.json).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(ROOT / "results")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(ROOT / "work" / "derived_numbers.json")))
N_BOOT = 10000
SEED = 0
CI = (2.5, 97.5)
# (task|decision window s, few-fast cell, many-slow cell) along the budget of 8 soundings/s
TRIAL_GAINS = {"occupancy_2s": ("occupancy|2", "1|8", "8|1"), "motion_1s": ("motion|1", "1|8", "4|2")}
# (decision window, two-pair cell at 1 Hz, 16-pair cell) along the office budget of 2 soundings/s
OFFICE_GAINS = {"occupancy_20s": ("w20s", "2|1", "16|0.125"), "occupancy_60s": ("w60s", "2|1", "16|0.125")}


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return [round(float(v), 4) for v in np.percentile(means, CI)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "seed": SEED, "trial_gain": {}, "office_gain": {}}
    trials = json.loads((ANALYSIS_DIR / "scripted_trial_decision_windows.json").read_text())["results"]
    for name, (key, few, many) in TRIAL_GAINS.items():
        a, b = trials[key][few]["auc_by_trial"], trials[key][many]["auc_by_trial"]
        g = np.array([b[t] - a[t] for t in sorted(a)])
        out["trial_gain"][name] = {"mean": round(float(g.mean()), 4), "ci95": boot_ci(g, rng), "wins": int((g > 0).sum()), "n": len(g)}
    for name, (w, few, many) in OFFICE_GAINS.items():
        cells = json.loads((ANALYSIS_DIR / "decision_windows" / f"decision_window_ablation_{w}.json").read_text())["windows"][w]["cells"]
        g = cells[many]["auc"] - np.array(cells[few]["auc_subsets"])
        out["office_gain"][name] = {"mean": round(float(g.mean()), 4), "ci95": boot_ci(g, rng), "wins": int((g > 0).sum()), "n": len(g)}
    leak = json.loads((ANALYSIS_DIR / "schedule_leak.json").read_text())
    ratios = sorted(o / e for o, e in leak["per_session_availability_occupied_empty"] if o > e)
    out["cover_cost"] = {"min": round(ratios[0], 3), "max": round(ratios[-1], 3),
                         "median": round(float(np.median(ratios)), 3), "n_sessions": len(ratios)}
    corr = json.loads((ANALYSIS_DIR / "sounding_correlation.json").read_text())["phases"]
    out["crossing_lag_s"] = {}
    for phase, d in corr.items():
        lags = sorted(d["within_pair_median_abs_corr"], key=float)
        x = np.array([float(l) for l in lags]); y = np.array([d["within_pair_median_abs_corr"][l] for l in lags])
        level = float(np.median(list(d["across_pairs_median_abs_corr"].values())))
        i = int(np.argmax(y < level))
        out["crossing_lag_s"][phase] = round(float(x[i - 1] + (y[i - 1] - level) / (y[i - 1] - y[i]) * (x[i] - x[i - 1])), 3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
