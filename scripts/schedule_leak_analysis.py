#!/usr/bin/env python3
"""
How much the office mesh's adaptive schedule reveals about occupancy.

On the rate-decimated office windows (extract_rate_decimated_features.py), a
pair is available in a window when it was sounded often enough to score. The
scheduler decided which pairs to sound, so availability is a property of the
schedule, observable without CSI. Reported:
  - availability (fraction of pair-windows) in occupied and in empty windows,
    pooled and per session
  - AUC of the number of available pairs (no training, no CSI), pooled and
    within each session that holds both classes
  - from rate_decimation_ablation.json: AUC of a classifier on the
    availability pattern alone against the CSI features, session-grouped CV
Empty windows are drawn as in rate_decimation_ablation.py (--empty-match time:
within each time-of-day slot, N_DRAWS times; numbers are means over draws).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from memlimit.memlimit import set_vm_limit  # noqa: E402
from rate_decimation_ablation import (MATCHED_CLASS_RATIO, N_DRAWS, RNG_SEED, balance, load,  # noqa: E402
                                     load_times, time_bin)

IN_DIR = Path(os.environ.get("IN_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "rate_decimated")))
ABLATION_JSON = IN_DIR / "rate_decimation_ablation.json"
OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "schedule_leak.json")))
PATTERN_K = "100"                    # pair count of the pattern-vs-CSI comparison


def leak(avail: np.ndarray, y: np.ndarray, g: np.ndarray, pick) -> dict:
    """Availability by class and AUC of the available-pair count, pooled and within each
    session holding both classes. pick(mask) returns the windows of a mask that are
    scored: all of them, or a time-matched draw."""
    per_window = avail.sum(axis=0)
    k = pick(np.ones(len(y), bool))
    out = {"occupied": float(avail[:, k[y[k] == 1]].mean()), "empty": float(avail[:, k[y[k] == 0]].mean()),
           "auc": float(roc_auc_score(y[k], per_window[k])), "per_session": {}}
    for s in np.unique(g):
        k = pick(g == s)
        if len(set(y[k])) == 2:
            out["per_session"][int(s)] = (float(avail[:, k[y[k] == 1]].mean()), float(avail[:, k[y[k] == 0]].mean()),
                                          float(roc_auc_score(y[k], per_window[k])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--empty-match", choices=("time", "none"), default="time",
                    help="draw empty windows within each MATCH_BIN_S slot of the day (N_DRAWS times), or use every window once")
    args = ap.parse_args()
    set_vm_limit()
    X, y, g, link_ids, _, _, sessions = load(IN_DIR)
    avail = np.isfinite(X[0, :, :, 0])                  # (pairs, windows)
    del X
    if args.empty_match == "time":
        rng = np.random.default_rng(RNG_SEED)
        slots = time_bin(load_times(IN_DIR))
        def pick(mask):
            idx = np.where(mask)[0]
            return idx[balance(y[idx], rng, slots[idx], MATCHED_CLASS_RATIO)]
        draws = [leak(avail, y, g, pick) for _ in range(N_DRAWS)]
    else:
        draws = [leak(avail, y, g, lambda mask: np.where(mask)[0])]
    # the sessions that can be matched depend on counts only, so they are the same in every draw
    assert all(d["per_session"].keys() == draws[0]["per_session"].keys() for d in draws)
    per_session = np.mean([list(d["per_session"].values()) for d in draws], axis=0)    # (sessions, 3)
    aucs = [d["auc"] for d in draws]
    abl = json.loads(ABLATION_JSON.read_text())["results"]
    out = {
        "empty_match": args.empty_match, "n_draws": len(draws),
        "n_windows": int(len(y)), "n_occupied": int(y.sum()), "n_sessions": len(sessions), "n_pairs": len(link_ids),
        "availability_occupied": round(float(np.mean([d["occupied"] for d in draws])), 4),
        "availability_empty": round(float(np.mean([d["empty"] for d in draws])), 4),
        "sessions_with_both_classes": len(per_session),
        "sessions_occupied_higher": int(sum(o > e for o, e, _ in per_session)),
        "per_session_availability_occupied_empty": [[round(float(o), 4), round(float(e), 4)] for o, e, _ in per_session],
        "available_pair_count_auc": round(float(np.mean(aucs)), 4),
        "available_pair_count_auc_draw_sd": round(float(np.std(aucs)), 4),
        "available_pair_count_auc_within_session": [round(float(a), 4) for _, _, a in per_session],
        "within_session_sessions": [sessions[s] for s in draws[0]["per_session"]],
        "pattern_vs_csi_k": int(PATTERN_K),
        "pattern_only_auc_session_grouped": abl["session_grouped"][PATTERN_K]["mask_only_auc"],
        "csi_auc_session_grouped": abl["session_grouped"][PATTERN_K]["lr_auc_mean"][0],
        "pattern_only_auc_stratified": abl["stratified"][PATTERN_K]["mask_only_auc"],
        "csi_auc_stratified": abl["stratified"][PATTERN_K]["lr_auc_mean"][0],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
