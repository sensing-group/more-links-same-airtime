#!/usr/bin/env python3
"""
Descriptive numbers of the office recordings that the paper states.

From the 60 s rate-decimated windows (IN_DIR, extract_rate_decimated_features.py):
  - sounding rate: per session, the median over link-windows with at least one frame
    of native frames per second; median and range over sessions
  - labeled windows, occupied windows, sessions with occupied windows, sessions occupied
    throughout
From the per-session release features (EWSN_DIR, ghost_*_ewsn.npz, 60 s windows):
  - transmitter-receiver identifier pairs, node identifiers, self-records (tx == rx)
  - raw amplitude variance per pair (median over labeled windows): largest pair against the
    median pair, and the share of windows in which the TOP_ARGMAX_PAIRS pairs most often
    holding the maximum over all pairs hold it
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402

DATA = REPO_ROOT / "data" / "csi-sense-trials"
IN_DIR = Path(os.environ.get("IN_DIR", str(DATA / "rate_decimated")))
EWSN_DIR = Path(os.environ.get("EWSN_DIR", str(DATA / "ewsn_compat")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(DATA / "office_session_stats.json")))
AMP_VAR_INDEX = 4                 # amplitude variance in the 11-feature release layout (mean part)
TOP_ARGMAX_PAIRS = 3


def rate_and_counts() -> dict:
    rates, n_win, n_pos, with_pos, all_pos = [], 0, 0, 0, 0
    files = sorted(IN_DIR.glob("rate_*.npz"))
    for f in files:
        z = np.load(f, allow_pickle=False)
        nf = np.asarray(z["native_frames"], float)
        rates.append(float(np.median(nf[nf > 0]) / float(z["window_s"])))
        y = z["labels"]
        y = y[y >= 0]
        n_win += len(y)
        n_pos += int(y.sum())
        with_pos += int(y.sum() > 0)
        all_pos += int(len(y) > 0 and y.all())
    return {"n_sessions": len(files), "median_session_rate_hz": float(np.median(rates)),
            "session_rate_range_hz": [float(min(rates)), float(max(rates))], "labeled_windows": n_win,
            "occupied_windows": n_pos, "sessions_with_occupied": with_pos, "sessions_occupied_throughout": all_pos}


def pair_stats() -> dict:
    files = sorted(EWSN_DIR.glob("ghost_*_ewsn.npz"))
    lids = sorted({str(l) for f in files for l in np.load(f, allow_pickle=False)["link_ids"]})
    li = {l: i for i, l in enumerate(lids)}
    cols = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        ok = d["labels"] >= 0
        a = np.full((len(lids), int(ok.sum())), np.nan, np.float32)
        for si, l in enumerate(d["link_ids"]):
            v = d["features"][si][ok]
            live = np.abs(v).sum(1) > 0
            a[li[str(l)], live] = v[live, AMP_VAR_INDEX]
        cols.append(a)
    A = np.concatenate(cols, axis=1)
    med = np.nanmedian(A, axis=1)
    has = np.isfinite(A).any(axis=0)
    winner = np.nanargmax(np.where(np.isfinite(A[:, has]), A[:, has], -np.inf), axis=0)
    top = Counter(winner.tolist()).most_common(TOP_ARGMAX_PAIRS)
    nodes = {n for l in lids for n in l.split("->")}
    return {"pairs": len(lids), "node_ids": len(nodes), "self_records": sum(a == b for a, b in (l.split("->") for l in lids)),
            "raw_amp_var_largest_pair_median": float(np.nanmax(med)), "raw_amp_var_median_pair": float(np.nanmedian(med)),
            "top_argmax_pairs_share": float(sum(c for _, c in top) / has.sum()), "windows_scored": int(has.sum())}


def main() -> None:
    set_vm_limit()
    out = {"rate_and_counts": rate_and_counts(), "pairs": pair_stats()}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
