#!/usr/bin/env python3
"""
Extract EWSN-compatible per-link features from the dataset-release/ (ghost paper dataset).

This is the office mesh (Dornbirn):
  - 14 ESP32-C3 nodes (12 to 16 node ids appear in later sessions)
  - 27 sessions, 4.1M windows
  - Ground truth: camera-based presence + person count

Memory safety:
  - Processes ONE session at a time (streaming windows)
  - Per-link accumulators are bounded (one list per link per aggregation window)
  - resource.setrlimit(RLIMIT_AS, 10 GB) at startup
  - gc.collect() between sessions
  - Peak memory estimate: ~2 GB for the largest session (620k windows)

Output:
  Per-session NPZ: (n_links, n_windows, 22) + labels + timestamps
  Combined NPZ: all sessions merged (for multi-env comparison)
"""
from __future__ import annotations

import gc
import gzip
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ======================================================================
# Memory safety
# ======================================================================

VM_LIMIT_BYTES = 10 * 1024 ** 3  # 10 GB hard cap


def _set_memory_limit() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(VM_LIMIT_BYTES, hard) if hard != resource.RLIM_INFINITY else VM_LIMIT_BYTES
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    except (ValueError, OSError):
        pass


def _rss_mb() -> float:
    """Current RSS in MB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


# ======================================================================
# Configuration
# ======================================================================

DATASET_DIR = Path(os.environ.get(
    "DATASET_DIR",
    str(Path(__file__).resolve().parent.parent / "dataset-release"),
))
OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "csi-sense-trials" / "ewsn_compat"),
))

# EWSN feature extraction parameters
N_SUBCARRIERS = 52
N_FEATURES = 11
FEATURE_NAMES = [
    "rssi_mean", "rssi_std", "rssi_min", "rssi_max",
    "amp_var", "amp_var_std", "nbvi",
    "temporal_diff_mean", "temporal_diff_std",
    "zcr", "energy",
]

# Aggregation window for EWSN features
WINDOW_S = 60.0  # 60-second windows (matches EWSN paper)
ENVIRONMENT_ID = "office"  # stored in every output file
MIN_PACKETS_PER_LINK_WINDOW = 5  # Minimum packets for a valid link-window

# The dataset windows are 250 ms with 100 ms stride.
# At ~3.57 Hz per link, expect ~3-4 packets per 250 ms window per active link.
# In a 60-second aggregation window, expect ~214 dataset windows → ~214 packets/link.

# Sessions to skip (too small)
MIN_SESSION_WINDOWS = 100  # Need at least 100 dataset windows for meaningful extraction

# Stride: process every Nth dataset window to reduce runtime.
# At stride=4 with 250ms windows, we still get ~60 packets/link per 60s agg window.
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "4"))

# RSS warning threshold
RSS_WARN_MB = 4000.0


# ======================================================================
# Feature extraction (EWSN-compatible, per-link per-packet)
# ======================================================================

def extract_per_packet_features(csi_frames: List[List[int]], rssi_mean: float) -> np.ndarray:
    """Extract 11 features from one link's CSI frames within one dataset window.

    This replicates the EWSN fusion_baselines.py feature set but operates
    on all frames in a dataset window (typically 3-10 frames per 250 ms window).
    """
    if not csi_frames or len(csi_frames) < 2:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # Compute per-subcarrier amplitudes: (n_frames, 52)
    arr = np.array(csi_frames, dtype=np.float32)
    i_vals = arr[:, 0::2]
    q_vals = arr[:, 1::2]
    amplitudes = np.sqrt(i_vals ** 2 + q_vals ** 2)

    # Mean amplitude across subcarriers per frame → 1D signal
    mean_amp = amplitudes.mean(axis=1)  # (n_frames,)

    if mean_amp.size < 2:
        return np.zeros(N_FEATURES, dtype=np.float32)

    rssi_m = float(np.mean(mean_amp))
    rssi_s = float(np.std(mean_amp))
    diff = np.diff(mean_amp)

    features = np.array([
        rssi_m,                                                          # rssi_mean
        rssi_s,                                                          # rssi_std
        float(np.min(mean_amp)),                                         # rssi_min
        float(np.max(mean_amp)),                                         # rssi_max
        float(np.var(mean_amp)),                                         # amp_var
        float(np.std(diff)) if diff.size > 0 else 0.0,                  # amp_var_std
        rssi_s / rssi_m if rssi_m > 0 else 0.0,                         # nbvi
        float(np.mean(np.abs(diff))) if diff.size > 0 else 0.0,         # temporal_diff_mean
        float(np.std(diff)) if diff.size > 0 else 0.0,                  # temporal_diff_std
        float(np.sum(np.abs(np.diff(np.sign(diff - np.mean(diff))))) /  # zcr
              (2 * len(diff))) if len(diff) > 1 else 0.0,
        float(np.sum(mean_amp ** 2)),                                    # energy
    ], dtype=np.float32)

    return features


# ======================================================================
# Session processing (streaming, memory-safe)
# ======================================================================

def discover_sessions(dataset_dir: Path) -> List[Path]:
    """Find all session directories with sample files."""
    sessions_dir = dataset_dir / "sessions"
    if not sessions_dir.exists():
        print(f"ERROR: {sessions_dir} not found", flush=True)
        return []
    return sorted(
        d for d in sessions_dir.iterdir()
        if d.is_dir() and list(d.glob("samples_*"))
    )


def stream_windows(session_dir: Path, stride: int = WINDOW_STRIDE):
    """Yield (window_dict) from a session directory, streaming.

    With stride > 1, only yields every Nth data window to reduce
    processing time on large sessions. stride=4 with 250ms dataset
    windows still provides ~60 packets/link per 60-second agg window.
    """
    files = sorted(session_dir.glob("samples_*.jsonl.gz"))
    if not files:
        files = sorted(session_dir.glob("samples_*.jsonl"))
    count = 0
    for fpath in files:
        opener = gzip.open if fpath.suffix == ".gz" else open
        with opener(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_metadata") or obj.get("_chunk_header") or obj.get("_footer"):
                    continue
                count += 1
                if stride > 1 and count % stride != 0:
                    continue
                yield obj


def process_session(
    session_dir: Path,
    window_s: float = WINDOW_S,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Process one session into EWSN-compatible features.

    Returns:
        link_ids: sorted list of link IDs seen
        features: (n_links, n_agg_windows, 22) float32
        labels: (n_agg_windows,) int8  — 0=empty, 1=occupied
        timestamps: (n_agg_windows,) float64
        n_dataset_windows: total dataset windows processed
        n_links: number of unique links
    """
    # Phase 1: Stream all windows, accumulate per-link features per agg window
    # Key: (link_id, agg_window_idx) → list of 11-feature vectors
    # Also track timestamps and labels per agg window

    t0 = time.time()
    link_features: Dict[str, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    agg_window_labels: Dict[int, List[int]] = defaultdict(list)  # agg_idx → [0 or 1 per dataset window]
    agg_window_timestamps: Dict[int, float] = {}

    first_ts = None
    n_dataset_windows = 0

    for window in stream_windows(session_dir):
        ts = window.get("timestamp", 0)
        if first_ts is None:
            first_ts = ts

        # Determine aggregation window index
        agg_idx = int((ts - first_ts) / window_s)
        if agg_idx not in agg_window_timestamps:
            agg_window_timestamps[agg_idx] = first_ts + agg_idx * window_s

        # Label from ground truth
        gt = window.get("ground_truth", {})
        label = 1 if gt.get("presence", False) else 0
        agg_window_labels[agg_idx].append(label)

        # Per-link features
        for link in window.get("csi_data", {}).get("links", []):
            tx = link.get("tx", "")
            rx = link.get("rx", "")
            if not tx or not rx:
                continue
            lid = f"{tx}->{rx}"
            csi_frames = link.get("csi_frames", [])
            rssi = link.get("rssi_mean", 0.0)
            if csi_frames and len(csi_frames) >= 2:
                feat = extract_per_packet_features(csi_frames, rssi)
                if not np.any(np.isnan(feat)):
                    link_features[lid][agg_idx].append(feat)

        n_dataset_windows += 1

        # Memory check every 50k windows
        if n_dataset_windows % 50000 == 0:
            rss = _rss_mb()
            print(f"    {n_dataset_windows:>8d} windows, RSS={rss:.0f} MB", flush=True)
            if rss > RSS_WARN_MB:
                print(f"    WARNING: RSS={rss:.0f} MB exceeds {RSS_WARN_MB:.0f} MB", flush=True)

    if n_dataset_windows < MIN_SESSION_WINDOWS:
        return [], np.zeros((0, 0, 0)), np.zeros(0), np.zeros(0), n_dataset_windows, 0

    # Phase 2: Build dense feature matrix
    link_ids = sorted(link_features.keys())
    n_links = len(link_ids)
    n_agg_windows = max(agg_window_timestamps.keys()) + 1 if agg_window_timestamps else 0

    if n_agg_windows == 0 or n_links == 0:
        return [], np.zeros((0, 0, 0)), np.zeros(0), np.zeros(0), n_dataset_windows, 0

    features = np.zeros((n_links, n_agg_windows, 2 * N_FEATURES), dtype=np.float32)
    for li, lid in enumerate(link_ids):
        for agg_idx, feat_list in link_features[lid].items():
            if len(feat_list) < MIN_PACKETS_PER_LINK_WINDOW:
                continue
            if agg_idx >= n_agg_windows:
                continue
            pkt_arr = np.array(feat_list, dtype=np.float32)
            features[li, agg_idx, :N_FEATURES] = pkt_arr.mean(axis=0)
            features[li, agg_idx, N_FEATURES:] = pkt_arr.std(axis=0)

    # Free the big accumulator
    del link_features
    gc.collect()

    # Labels: majority vote per aggregation window
    labels = np.full(n_agg_windows, -1, dtype=np.int8)
    for agg_idx, votes in agg_window_labels.items():
        if agg_idx < n_agg_windows and votes:
            labels[agg_idx] = 1 if sum(votes) > len(votes) / 2 else 0

    timestamps = np.array([
        agg_window_timestamps.get(i, first_ts + i * window_s)
        for i in range(n_agg_windows)
    ], dtype=np.float64)

    elapsed = time.time() - t0
    print(f"    {n_dataset_windows:,} windows → {n_links} links × {n_agg_windows} agg windows "
          f"in {elapsed:.1f}s (RSS={_rss_mb():.0f} MB)", flush=True)

    return link_ids, features, labels, timestamps, n_dataset_windows, n_links


# ======================================================================
# Main
# ======================================================================

def main():
    _set_memory_limit()

    print("=" * 70, flush=True)
    print("EXTRACT EWSN FEATURES FROM DATASET-RELEASE", flush=True)
    print(f"  Dataset: {DATASET_DIR}", flush=True)
    print(f"  Output:  {OUT_DIR}", flush=True)
    print(f"  Window:  {WINDOW_S}s", flush=True)
    print(f"  Stride:  {WINDOW_STRIDE} (process every {WINDOW_STRIDE}th dataset window)", flush=True)
    print(f"  VM limit: {VM_LIMIT_BYTES / 1024**3:.0f} GB", flush=True)
    print("=" * 70, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = discover_sessions(DATASET_DIR)
    print(f"Found {len(sessions)} sessions", flush=True)

    # Process each session independently
    session_summaries = []
    all_link_ids_union = set()

    for i, sess_dir in enumerate(sessions):
        sess_id = sess_dir.name
        print(f"\n[{i+1}/{len(sessions)}] {sess_id}", flush=True)

        link_ids, features, labels, timestamps, n_dw, n_links = process_session(sess_dir)

        if len(link_ids) == 0:
            print(f"    SKIP: too few windows ({n_dw})", flush=True)
            continue

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        n_unk = int((labels == -1).sum())
        print(f"    Labels: {n_pos} pos, {n_neg} neg, {n_unk} unlabeled", flush=True)

        # Save per-session NPZ
        out_path = OUT_DIR / f"ghost_{sess_id}_ewsn.npz"
        np.savez_compressed(
            out_path,
            link_ids=np.array(link_ids),
            features=features,
            labels=labels,
            timestamps=timestamps,
            feature_names=np.array(FEATURE_NAMES),
            trial_id=sess_id,
            environment_id=ENVIRONMENT_ID,
            window_s=WINDOW_S,
        )
        size_kb = out_path.stat().st_size / 1024
        print(f"    Saved: {out_path.name} ({size_kb:.0f} KB)", flush=True)

        session_summaries.append({
            "session_id": sess_id,
            "n_links": n_links,
            "n_windows": int(features.shape[1]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_dataset_windows": n_dw,
        })
        all_link_ids_union.update(link_ids)

        # Free memory before next session
        del link_ids, features, labels, timestamps
        gc.collect()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    total_pos = sum(s["n_pos"] for s in session_summaries)
    total_neg = sum(s["n_neg"] for s in session_summaries)
    total_win = sum(s["n_windows"] for s in session_summaries)
    print(f"  Sessions processed: {len(session_summaries)}", flush=True)
    print(f"  Total links (union): {len(all_link_ids_union)}", flush=True)
    print(f"  Total agg windows: {total_win} ({total_pos} pos, {total_neg} neg)", flush=True)

    # Save summary JSON
    summary_path = OUT_DIR / "ghost_dataset_extraction_summary.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_dir": str(DATASET_DIR),
        "window_s": WINDOW_S,
        "n_sessions": len(session_summaries),
        "total_links_union": len(all_link_ids_union),
        "total_agg_windows": total_win,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "link_ids_union": sorted(all_link_ids_union),
        "sessions": session_summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
