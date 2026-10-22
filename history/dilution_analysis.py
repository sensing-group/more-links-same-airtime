#!/usr/bin/env python3
"""
EARLIER VERSION (kept to reproduce the flawed numbers quoted in Section 8 of the paper: the best
single pair at J 0.55 and all pairs at 0.01). It uses the practices Section 8 corrects.

Multi-environment dilution effect analysis for the LCN paper.

Demonstrates that the dilution effect (adding links hurts detection)
generalizes across 3 environments:
  1. Kloten apartment (72 links, 12 days) — published EWSN reference
  2. Dornbirn residential (155 links, 312 hours) — from dataset-release
  3. St. Gallen living room (56 links, 12.5 min) — T0003

For each environment:
  - Per-link AUC distribution (Mann-Whitney)
  - Link-count ablation: Top-K vs Random-K vs Bottom-K
  - Summary statistics for cross-environment comparison

Memory safety:
  - resource.setrlimit at startup
  - Process one environment at a time
  - gc.collect() between environments
  - Large NPZs loaded and freed individually
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ======================================================================
# Memory safety
# ======================================================================

VM_LIMIT_BYTES = 10 * 1024 ** 3


def _set_memory_limit() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(VM_LIMIT_BYTES, hard) if hard != resource.RLIM_INFINITY else VM_LIMIT_BYTES
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    except (ValueError, OSError):
        pass


def _rss_mb() -> float:
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

EWSN_DIR = Path(os.environ.get(
    "EWSN_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "office" / "features"),
))
OUT_PATH = Path(os.environ.get(
    "OUT_PATH",
    str(Path(__file__).resolve().parent.parent / "work" / "history" / "dilution_analysis_results.json"),
))

ABLATION_LINK_COUNTS = [1, 2, 3, 5, 10, 20, 50, 100]
ABLATION_N_RANDOM = 30
RNG_SEED = 42

# Minimum windows per class for meaningful AUC
MIN_WINDOWS_PER_CLASS = 5

# Maximum windows to subsample per session (memory bound)
MAX_WINDOWS_PER_SESSION = 5000


# ======================================================================
# Per-link AUC (Mann-Whitney U)
# ======================================================================

def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute AUC via Mann-Whitney U statistic. Memory: O(n+m)."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 3 or n_neg < 3:
        return 0.5
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Tie correction
    sorted_c = combined[order]
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and sorted_c[j + 1] == sorted_c[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[:n_pos].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def compute_per_link_auc_ampvar(
    features: np.ndarray,  # (n_links, n_windows, n_feat)
    labels: np.ndarray,    # (n_windows,)
) -> np.ndarray:
    """Per-link AUC using amplitude variance (feature index 4) as score.

    Simple, robust, doesn't overfit with small samples.
    """
    n_links = features.shape[0]
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() < MIN_WINDOWS_PER_CLASS or neg_mask.sum() < MIN_WINDOWS_PER_CLASS:
        return np.full(n_links, 0.5, dtype=np.float64)

    aucs = np.zeros(n_links, dtype=np.float64)
    for i in range(n_links):
        pos_scores = features[i, pos_mask, 4]  # amp_var during occupied
        neg_scores = features[i, neg_mask, 4]  # amp_var during empty
        # Skip links with zero variance in both classes
        if pos_scores.sum() == 0 and neg_scores.sum() == 0:
            aucs[i] = 0.5
            continue
        aucs[i] = mann_whitney_auc(pos_scores, neg_scores)

    return aucs


# ======================================================================
# Youden's J for link-subset evaluation
# ======================================================================

def youdens_j_ampvar(
    features: np.ndarray,  # (n_links, n_windows, n_feat)
    labels: np.ndarray,
    link_idx: np.ndarray,
    n_thresholds: int = 100,
) -> float:
    """Youden's J using max(amp_var) across selected links as detection score."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() < 3 or neg_mask.sum() < 3:
        return 0.0

    # Detection score: max amp_var across selected links per window
    scores = features[link_idx, :, 4].max(axis=0)  # (n_windows,)
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    lo = min(pos_scores.min(), neg_scores.min())
    hi = max(pos_scores.max(), neg_scores.max())
    if lo == hi:
        return 0.0

    thresholds = np.linspace(lo, hi, n_thresholds)
    best_j = -1.0
    for thr in thresholds:
        tp = int(np.sum(pos_scores > thr))
        fn = len(pos_scores) - tp
        fp = int(np.sum(neg_scores > thr))
        tn = len(neg_scores) - fp
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
    return float(best_j)


# ======================================================================
# Link-count ablation
# ======================================================================

def link_count_ablation(
    features: np.ndarray,
    labels: np.ndarray,
    per_link_aucs: np.ndarray,
    link_counts: List[int],
) -> Dict[int, Dict[str, float]]:
    """Top-K vs Random-K vs Bottom-K using Youden's J."""
    n_links = features.shape[0]
    rng = np.random.default_rng(RNG_SEED)

    # Rank by |AUC - 0.5| descending
    ranked_idx = np.argsort(np.abs(per_link_aucs - 0.5))[::-1]

    results = {}
    for k in link_counts:
        if k > n_links:
            continue

        top_idx = ranked_idx[:k]
        bot_idx = ranked_idx[-k:]

        top_j = youdens_j_ampvar(features, labels, top_idx)
        bot_j = youdens_j_ampvar(features, labels, bot_idx)

        rand_js = []
        for _ in range(ABLATION_N_RANDOM):
            ri = rng.choice(n_links, k, replace=False)
            rand_js.append(youdens_j_ampvar(features, labels, ri))
        rand_mean = float(np.mean(rand_js))
        rand_std = float(np.std(rand_js))

        results[k] = {
            "topk_J": round(top_j, 4),
            "random_J_mean": round(rand_mean, 4),
            "random_J_std": round(rand_std, 4),
            "bottomk_J": round(bot_j, 4),
            "gap": round(top_j - rand_mean, 4),
        }
        print(f"    K={k:4d}: top={top_j:.3f}  rand={rand_mean:.3f}±{rand_std:.3f}  "
              f"bot={bot_j:.3f}  gap={top_j - rand_mean:+.3f}", flush=True)

    return results


# ======================================================================
# Per-environment analysis
# ======================================================================

def balance_classes(
    features: np.ndarray,
    labels: np.ndarray,
    max_ratio: float = 3.0,
    seed: int = RNG_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample majority class to at most max_ratio × minority class.

    Returns balanced (features, labels) without modifying originals.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return features, labels

    ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
    if ratio <= max_ratio:
        return features, labels

    rng = np.random.default_rng(seed)
    minority_n = min(n_pos, n_neg)
    target_majority = int(minority_n * max_ratio)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if n_neg > n_pos:
        keep_neg = rng.choice(neg_idx, target_majority, replace=False)
        keep_neg.sort()
        keep = np.concatenate([pos_idx, keep_neg])
    else:
        keep_pos = rng.choice(pos_idx, target_majority, replace=False)
        keep_pos.sort()
        keep = np.concatenate([keep_pos, neg_idx])

    keep.sort()
    return features[:, keep, :], labels[keep]


def analyze_environment(
    features: np.ndarray,
    labels: np.ndarray,
    link_ids: List[str],
    env_name: str,
) -> Dict[str, Any]:
    """Full dilution analysis for one environment."""
    n_links, n_windows, n_feat = features.shape
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print(f"\n{'='*60}", flush=True)
    print(f"  {env_name}: {n_links} links, {n_windows} windows ({n_pos} pos, {n_neg} neg)", flush=True)
    print(f"{'='*60}", flush=True)

    # Balance classes for reliable Youden's J (subsample majority to 3:1 max)
    feat_bal, lab_bal = balance_classes(features, labels)
    n_pos_bal = int((lab_bal == 1).sum())
    n_neg_bal = int((lab_bal == 0).sum())
    if n_pos_bal != n_pos or n_neg_bal != n_neg:
        print(f"  Class balancing: {n_pos}:{n_neg} → {n_pos_bal}:{n_neg_bal} "
              f"(ratio {max(n_pos,n_neg)/max(min(n_pos,n_neg),1):.1f}:1 → "
              f"{max(n_pos_bal,n_neg_bal)/max(min(n_pos_bal,n_neg_bal),1):.1f}:1)", flush=True)

    # Per-link AUC (on balanced data)
    print("  Computing per-link AUC (amplitude variance)...", flush=True)
    aucs = compute_per_link_auc_ampvar(feat_bal, lab_bal)

    auc_stats = {
        "n_links": int(n_links),
        "n_windows_original": int(n_windows),
        "n_windows_balanced": int(feat_bal.shape[1]),
        "n_pos": n_pos_bal,
        "n_neg": n_neg_bal,
        "median": round(float(np.median(aucs)), 4),
        "mean": round(float(np.mean(aucs)), 4),
        "std": round(float(np.std(aucs)), 4),
        "iqr": round(float(np.percentile(aucs, 75) - np.percentile(aucs, 25)), 4),
        "frac_above_05": round(float((aucs > 0.5).mean()), 4),
        "frac_above_06": round(float((aucs > 0.6).mean()), 4),
        "frac_above_07": round(float((aucs > 0.7).mean()), 4),
        "frac_below_05": round(float((aucs < 0.5).mean()), 4),
        "min": round(float(aucs.min()), 4),
        "max": round(float(aucs.max()), 4),
    }
    print(f"  AUC: median={auc_stats['median']:.3f}, "
          f">0.5={auc_stats['frac_above_05']:.0%}, "
          f">0.7={auc_stats['frac_above_07']:.0%}", flush=True)

    # Link-count ablation (on balanced data)
    valid_ks = [k for k in ABLATION_LINK_COUNTS if k <= n_links]
    if n_links not in valid_ks:
        valid_ks.append(n_links)
    valid_ks = sorted(valid_ks)

    ablation = {}
    if n_pos_bal >= MIN_WINDOWS_PER_CLASS and n_neg_bal >= MIN_WINDOWS_PER_CLASS:
        print("  Running link-count ablation...", flush=True)
        ablation = link_count_ablation(feat_bal, lab_bal, aucs, valid_ks)

    return {
        "environment": env_name,
        "per_link_auc": auc_stats,
        "link_count_ablation": {str(k): v for k, v in ablation.items()},
        "top_10_links": [
            {"link": link_ids[i], "auc": round(float(aucs[i]), 4)}
            for i in np.argsort(np.abs(aucs - 0.5))[::-1][:10]
        ],
    }


# ======================================================================
# Load and merge ghost dataset sessions
# ======================================================================

def load_and_merge_ghost_sessions(
    ewsn_dir: Path,
    max_windows: int = MAX_WINDOWS_PER_SESSION,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all ghost_*_ewsn.npz files and merge into one feature matrix.

    Memory-safe: loads one session at a time, subsamples if needed,
    builds per-link accumulators, then constructs the final matrix.
    """
    ghost_files = sorted(ewsn_dir.glob("ghost_*_ewsn.npz"))
    if not ghost_files:
        return np.zeros((0, 0, 0)), np.zeros(0), []

    print(f"  Found {len(ghost_files)} ghost session NPZs", flush=True)

    # Phase 1: Discover all link IDs across sessions
    all_link_ids = set()
    session_info = []
    for f in ghost_files:
        d = np.load(f, allow_pickle=True)
        lids = list(d["link_ids"])
        n_win = d["features"].shape[1]
        n_pos = int((d["labels"] == 1).sum())
        n_neg = int((d["labels"] == 0).sum())
        all_link_ids.update(lids)
        session_info.append((f, lids, n_win, n_pos, n_neg))
        del d
    gc.collect()

    link_ids = sorted(all_link_ids)
    lid_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)
    n_feat = 22  # 11 means + 11 stds

    print(f"  Total links (union): {n_links}", flush=True)

    # Phase 2: Merge sessions, subsampling large ones
    rng = np.random.default_rng(RNG_SEED)
    all_features = []
    all_labels = []

    for f, lids, n_win, n_pos, n_neg in session_info:
        # Skip sessions with no labels
        if n_pos + n_neg == 0:
            continue

        d = np.load(f, allow_pickle=True)
        feat = d["features"]  # (sess_links, sess_windows, 22)
        lab = d["labels"]      # (sess_windows,)
        sess_lids = list(d["link_ids"])
        del d

        # Subsample if too many windows
        valid_mask = lab >= 0
        n_valid = int(valid_mask.sum())
        if n_valid > max_windows:
            valid_idx = np.where(valid_mask)[0]
            keep = rng.choice(valid_idx, max_windows, replace=False)
            keep.sort()
            feat = feat[:, keep, :]
            lab = lab[keep]
            n_valid = max_windows

        # Map to unified link order
        sess_feat = np.zeros((n_links, feat.shape[1], n_feat), dtype=np.float32)
        for si, lid in enumerate(sess_lids):
            if lid in lid_to_idx:
                sess_feat[lid_to_idx[lid]] = feat[si]

        all_features.append(sess_feat)
        all_labels.append(lab[lab >= 0])

        del feat, sess_feat, lab
        gc.collect()

    if not all_features:
        return np.zeros((0, 0, 0)), np.zeros(0), []

    # Concatenate along window axis
    features = np.concatenate(all_features, axis=1)
    labels = np.concatenate(all_labels)
    del all_features, all_labels
    gc.collect()

    print(f"  Merged: {features.shape[1]} windows ({int((labels==1).sum())} pos, "
          f"{int((labels==0).sum())} neg), RSS={_rss_mb():.0f} MB", flush=True)

    return features, labels, link_ids


# ======================================================================
# Main
# ======================================================================

def main():
    _set_memory_limit()

    print("=" * 70, flush=True)
    print("MULTI-ENVIRONMENT DILUTION EFFECT ANALYSIS", flush=True)
    print(f"  EWSN dir: {EWSN_DIR}", flush=True)
    print(f"  Output:   {OUT_PATH}", flush=True)
    print("=" * 70, flush=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis": "dilution_effect_multi_environment",
        "environments": {},
    }

    # ---- Environment 1: Kloten (reference) ----
    print("\n" + "=" * 70, flush=True)
    print("ENV 1: KLOTEN APARTMENT (EWSN reference)", flush=True)
    print("=" * 70, flush=True)
    results["environments"]["kloten_apartment"] = {
        "environment": "Kloten apartment (12 nodes, 72 links, 12 days)",
        "source": "EWSN paper published values",
        "per_link_auc": {
            "n_links": 72,
            "n_windows": 288,
            "n_pos": 109,
            "n_neg": 203,
            "median": 0.633,
            "frac_above_05": 0.89,
            "frac_above_07": 0.20,
            "frac_below_05": 0.11,
        },
        "link_count_ablation": {
            "1": {"topk_auc": 0.534, "random_auc_mean": 0.487},
            "3": {"topk_auc": 0.351, "random_auc_mean": 0.467},
            "10": {"topk_auc": 0.470, "random_auc_mean": 0.508},
            "36": {"topk_auc": 0.504, "random_auc_mean": 0.486},
            "72": {"topk_auc": 0.534, "random_auc_mean": 0.534},
        },
        "key_finding": "Single link AUC 0.541 > full mesh 0.489 (d=0.86)",
    }
    print("  (Using published reference values)", flush=True)

    # ---- Environment 2: Dornbirn residential (ghost dataset) ----
    print("\n" + "=" * 70, flush=True)
    print("ENV 2: DORNBIRN RESIDENTIAL (ghost paper dataset)", flush=True)
    print("=" * 70, flush=True)

    features_d, labels_d, lids_d = load_and_merge_ghost_sessions(EWSN_DIR)
    if features_d.shape[0] > 0:
        results["environments"]["dornbirn_residential"] = analyze_environment(
            features_d, labels_d, lids_d, "Dornbirn residential (14 nodes, 155 links, 312 hours)"
        )
        del features_d, labels_d, lids_d
        gc.collect()
    else:
        print("  SKIPPED: no ghost dataset NPZs found", flush=True)

    # ---- Environment 3: St. Gallen living room ----
    print("\n" + "=" * 70, flush=True)
    print("ENV 3: ST. GALLEN LIVING ROOM", flush=True)
    print("=" * 70, flush=True)

    sg_path = EWSN_DIR / "T0003_sg-kk-living-room_ewsn.npz"
    if sg_path.exists():
        d = np.load(sg_path, allow_pickle=True)
        features_sg = d["features"]
        labels_sg = d["labels"]
        lids_sg = list(d["link_ids"])
        del d

        results["environments"]["st_gallen_living_room"] = analyze_environment(
            features_sg, labels_sg, lids_sg, "St. Gallen living room (8 nodes, 56 links, 12.5 min)"
        )
        del features_sg, labels_sg
        gc.collect()
    else:
        print(f"  SKIPPED: {sg_path} not found", flush=True)

    # ---- Cross-environment summary ----
    print("\n" + "=" * 70, flush=True)
    print("CROSS-ENVIRONMENT SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Environment':<45s} | {'Links':>5s} | {'Med AUC':>7s} | {'>0.5':>5s} | {'>0.7':>5s}", flush=True)
    print("-" * 75, flush=True)
    for env_name, env in results["environments"].items():
        auc = env.get("per_link_auc", {})
        n = auc.get("n_links", "?")
        med = auc.get("median", "?")
        a5 = auc.get("frac_above_05", "?")
        a7 = auc.get("frac_above_07", "?")
        print(f"{env_name:<45s} | {n:>5} | {med:>7} | {a5:>5} | {a7:>5}", flush=True)

    # ---- Save ----
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
