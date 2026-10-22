#!/usr/bin/env python3
"""
How much a further sounding of the same pair repeats what that pair already
measured, compared with what a different pair measures at the same time.

Per scripted trial and pair, the subcarrier-mean amplitude is resampled onto a
uniform GRID_HZ grid and detrended by a ROLLING_S moving mean, so slow drift
does not count as shared information. In each occupied phase (seated, walking):
  - within a pair: correlation between the series and itself LAG seconds later
  - across pairs: correlation between two different pairs, same time, split by
    whether the pairs share the transmitter, the receiver, a node in different
    roles, or no node
Both are reported signed and absolute; redundancy depends on the size of a
correlation, not its sign, so the absolute values are the ones to compare.
Medians over pairs (and pair couples) and trials are written to OUT_PATH.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "sounding_correlation.json")))
GRID_HZ = 8.0
ROLLING_S = 5.0
LAGS_S = (0.125, 0.25, 0.5, 1.0, 2.0)
PHASES = ("occupied_static", "occupied_moving")
MAX_PAIR_COUPLES = 1200
COUPLE_GROUPS = ("same_transmitter", "same_receiver", "node_in_other_role", "no_shared_node")
RNG_SEED = 0


def detrended_series(ts, amp, grid):
    s = np.interp(grid, ts, amp.mean(axis=1))
    k = int(ROLLING_S * GRID_HZ)
    trend = np.convolve(s, np.ones(k) / k, mode="same")
    return s - trend


def couple_group(a, b) -> str:
    if a[0] == b[0]:
        return "same_transmitter"
    if a[1] == b[1]:
        return "same_receiver"
    return "node_in_other_role" if set(a) & set(b) else "no_shared_node"


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def main() -> None:
    set_vm_limit()
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    within = {p: {lag: [] for lag in LAGS_S} for p in PHASES}
    across = {p: {g: [] for g in COUPLE_GROUPS} for p in PHASES}
    for trial in st.TRIALS:
        links, phases = st.read_trial(st.TRIALS_DIR / trial)
        for start, end, label in phases:
            if label not in PHASES:
                continue
            grid = np.arange(start + st.PHASE_GUARD_S, end - st.PHASE_GUARD_S, 1 / GRID_HZ)
            series, ends = [], []
            for lid, (ts, amp) in links.items():
                inside = (ts >= grid[0]) & (ts <= grid[-1])
                if inside.sum() >= GRID_HZ * (grid[-1] - grid[0]) * 0.5:      # sounded at least at half the grid rate
                    series.append(detrended_series(ts, amp, grid))
                    ends.append(lid.split("->"))
            for s in series:
                for lag in LAGS_S:
                    d = int(round(lag * GRID_HZ))
                    within[label][lag].append(corr(s[:-d], s[d:]))      # signed; abs taken at summary
            couples = [(i, j) for i in range(len(series)) for j in range(i + 1, len(series))]
            for idx in rng.permutation(len(couples))[:MAX_PAIR_COUPLES]:
                i, j = couples[idx]
                across[label][couple_group(ends[i], ends[j])].append(corr(series[i], series[j]))
        print(f"{trial} done ({time.time() - t0:.0f}s)", flush=True)
        del links
    out = {"grid_hz": GRID_HZ, "rolling_s": ROLLING_S, "trials": list(st.TRIALS), "phases": {}}
    for p in PHASES:
        out["phases"][p] = {
            "within_pair_median_corr": {f"{lag:g}": round(float(np.nanmedian(within[p][lag])), 4) for lag in LAGS_S},
            "within_pair_median_abs_corr": {f"{lag:g}": round(float(np.nanmedian(np.abs(within[p][lag]))), 4) for lag in LAGS_S},
            "across_pairs_median_corr": {g: round(float(np.nanmedian(across[p][g])), 4) for g in COUPLE_GROUPS},
            "across_pairs_median_abs_corr": {g: round(float(np.nanmedian(np.abs(across[p][g]))), 4) for g in COUPLE_GROUPS},
            "n_couples": {g: len(across[p][g]) for g in COUPLE_GROUPS},
            "n_pairs": len(within[p][LAGS_S[0]]),
        }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
