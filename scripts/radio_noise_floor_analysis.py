#!/usr/bin/env python3
"""
Empty-room noise floor of a single pair in the scripted trials, against CSI quantization.

Per pair and trial, over the empty phases in DECISION_WINDOW_S windows:
  - temporal std of the frame's mean amplitude (mean over subcarriers), in CSI units
    and relative to the mean amplitude; the median over windows is the pair's floor
  - the std that rounding the I and Q values to integers alone would add to that mean,
    (1/sqrt(12)) / sqrt(n_subcarriers), for comparison
Laboratory trials mix two Espressif MAC families (FAMILY_A_PREFIX and the rest) in one
room; per pair, without training, the script also compares the families on the floor and
on the AUC of the z-scored subcarrier-variance feature (seated or walking against empty,
held-out empty windows, BASE_RATE_HZ soundings per second).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "radio_noise_floor.json")))
DECISION_WINDOW_S = 5.0
MIN_FRAMES_PER_WINDOW = 10
MIN_WINDOWS_PER_CLASS = 5
FAMILY_A_PREFIX = "90:70:69"
LAB_MARKER = "lab"                     # trials whose name contains this are the mixed-family laboratory
QUANT_STEP_STD = 1.0 / np.sqrt(12.0)   # std of rounding a real value to the nearest integer


def family(mac: str) -> str:
    return "A" if mac.startswith(FAMILY_A_PREFIX) else "B"


def empty_floor(ts, frame_mean, phases) -> tuple:
    """Median over empty windows of the temporal std of the frame mean, absolute and relative."""
    sd, rel = [], []
    for lo, hi, lab in phases:
        if lab != "empty_room":
            continue
        for a in np.arange(lo + st.PHASE_GUARD_S, hi - st.PHASE_GUARD_S - DECISION_WINDOW_S, DECISION_WINDOW_S):
            seg = frame_mean[(ts >= a) & (ts < a + DECISION_WINDOW_S)]
            if len(seg) >= MIN_FRAMES_PER_WINDOW:
                sd.append(seg.std())
                rel.append(seg.std() / seg.mean())
    return (float(np.median(sd)), float(np.median(rel))) if sd else (np.nan, np.nan)


def class_auc(z, labels, ok, positive: str) -> float:
    e, p = ok & (labels == "empty_room"), ok & (labels == positive)
    if e.sum() < MIN_WINDOWS_PER_CLASS or p.sum() < MIN_WINDOWS_PER_CLASS:
        return np.nan
    return float(roc_auc_score(np.r_[np.zeros(e.sum()), np.ones(p.sum())], np.r_[z[e], z[p]]))


def main() -> None:
    set_vm_limit()
    thr = st.FEATURE_NAMES.index(st.THRESHOLD_FEATURE)
    trials = [st.TRIALS_DIR / t for t in st.TRIALS if (st.TRIALS_DIR / t / "raw" / "csi_raw.jsonl.zst").exists()]
    rows = []
    for tdir in trials:
        links, phases = st.read_trial(tdir)
        F, labels, cal, lids = st.trial_features(links, phases, DECISION_WINDOW_S)
        Z = st.zscore(F, cal)[0, :, :, thr]                       # base rate, (pairs, windows)
        for li, lid in enumerate(lids):
            ts, amp = links[lid]
            floor, floor_rel = empty_floor(ts, amp.mean(axis=1), phases)
            ok = np.isfinite(Z[li]) & ~cal
            tx, rx = lid.split("->")
            rows.append({"trial": tdir.name, "pair": lid, "tx": family(tx), "rx": family(rx),
                         "empty_floor": floor, "empty_floor_rel": floor_rel, "mean_amp": float(amp.mean(dtype=np.float64)),
                         "presence_auc": class_auc(Z[li], labels, ok, "occupied_static"),
                         "walking_auc": class_auc(Z[li], labels, ok, "occupied_moving")})
        print(f"{tdir.name}: {len(lids)} pairs", flush=True)
        del links

    floors = np.array([r["empty_floor"] for r in rows])
    rels = np.array([r["empty_floor_rel"] for r in rows])
    quant = QUANT_STEP_STD / np.sqrt(st.N_SUBCARRIERS)
    out = {"decision_window_s": DECISION_WINDOW_S, "base_rate_hz": st.BASE_RATE_HZ, "trials": [t.name for t in trials],
           "n_pair_trials": len(rows), "empty_floor_median": float(np.nanmedian(floors)),
           "empty_floor_iqr": [float(np.nanpercentile(floors, 25)), float(np.nanpercentile(floors, 75))],
           "empty_floor_rel_median": float(np.nanmedian(rels)), "quantization_floor": float(quant),
           "floor_over_quantization": float(np.nanmedian(floors) / quant), "lab_families": {}}
    print(f"empty floor median {out['empty_floor_median']:.3f} units ({out['empty_floor_rel_median']:.1%} of the mean), "
          f"quantization {quant:.3f}, ratio {out['floor_over_quantization']:.1f}", flush=True)

    per = defaultdict(list)                                       # laboratory pairs, mean over trials
    for r in rows:
        if LAB_MARKER in r["trial"]:
            per[r["pair"]].append(r)
    pairs = [{"pair": p, "tx": rs[0]["tx"], "rx": rs[0]["rx"]}
             | {k: float(np.nanmean([r[k] for r in rs])) for k in ("empty_floor", "empty_floor_rel", "presence_auc", "walking_auc")}
             for p, rs in per.items()]
    for side in ("rx", "tx"):
        for metric in ("empty_floor", "empty_floor_rel", "presence_auc", "walking_auc"):
            a = [p[metric] for p in pairs if p[side] == "A" and np.isfinite(p[metric])]
            b = [p[metric] for p in pairs if p[side] == "B" and np.isfinite(p[metric])]
            res = {"family_a_median": float(np.median(a)), "family_b_median": float(np.median(b)),
                   "family_a_n": len(a), "family_b_n": len(b), "mannwhitney_p": float(mannwhitneyu(a, b).pvalue)}
            out["lab_families"][f"{side}|{metric}"] = res
            print(f"lab, by {side} family, {metric:15s}: {FAMILY_A_PREFIX} {res['family_a_median']:.3f} (n={len(a)}), "
                  f"others {res['family_b_median']:.3f} (n={len(b)}), p={res['mannwhitney_p']:.3g}", flush=True)
    out["rows"] = rows
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
