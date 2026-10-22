#!/usr/bin/env python3
"""
EX3: Temporal vs Spatial Decomposition of the Dilution Effect.

EWSN reviewer concern F3: "Can't separate temporal (TDMA undersampling)
from spatial (redundant links)."

Two-factor experiment:

  Factor A — Vary link count at FIXED per-link temporal resolution:
    For K in {1, 2, 5, 10, 20, 50, 100, ALL}:
      Subsample time windows to simulate fixed per-link rate
      (e.g., if full mesh has 1.4 Hz/link, keep 1.4 Hz for all K)
      Compute classifier AUC at each K
    If dilution still occurs → spatial component is real
    If dilution disappears → temporal undersampling was the whole story

  Factor B — Vary temporal resolution at FIXED link count:
    For K in {5, 20, 50}:
      For rate_fraction in {0.1, 0.2, 0.5, 1.0}:
        Subsample time windows proportionally
        Compute classifier AUC
    Shows how much temporal resolution alone affects detection

Output:
  2D results matrix (links × temporal resolution → AUC)
  JSON results + figures via figure-generator skill

Memory safety:
  resource.setrlimit at startup, gc.collect between trials, class balancing
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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
    str(Path(__file__).resolve().parent.parent / "data" / "csi-sense-trials" / "ewsn_compat"),
))
OUT_PATH = Path(os.environ.get(
    "OUT_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "csi-sense-trials" / "temporal_spatial_decomposition.json"),
))

# Link counts for Factor A
LINK_COUNTS = [1, 2, 5, 10, 20, 50, 100]

# Rate fractions for Factor B (1.0 = use all temporal windows)
RATE_FRACTIONS = [0.1, 0.2, 0.5, 1.0]

# Fixed link counts for Factor B
FACTOR_B_LINK_COUNTS = [5, 20, 50]

N_RANDOM_TRIALS = 20
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 1000
RNG_SEED = 42
MAX_CLASS_RATIO = 3.0
MAX_WINDOWS_PER_SESSION = 5000


# ======================================================================
# Data loading (shared with dilution_analysis.py)
# ======================================================================

def load_and_merge(ewsn_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ghost_files = sorted(ewsn_dir.glob("ghost_*_ewsn.npz"))
    if not ghost_files:
        return np.zeros((0, 0, 0)), np.zeros(0), []

    all_link_ids = set()
    session_info = []
    for f in ghost_files:
        d = np.load(f, allow_pickle=True)
        lids = list(d["link_ids"])
        all_link_ids.update(lids)
        session_info.append((f, lids, d["features"].shape[1],
                             int((d["labels"] == 1).sum()), int((d["labels"] == 0).sum())))
        del d
    gc.collect()

    link_ids = sorted(all_link_ids)
    lid_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)

    rng = np.random.default_rng(RNG_SEED)
    all_features = []
    all_labels = []

    for f, lids, n_win, n_pos, n_neg in session_info:
        if n_pos + n_neg == 0:
            continue
        d = np.load(f, allow_pickle=True)
        feat = d["features"]
        lab = d["labels"]
        sess_lids = list(d["link_ids"])
        del d

        valid_mask = lab >= 0
        n_valid = int(valid_mask.sum())
        if n_valid > MAX_WINDOWS_PER_SESSION:
            valid_idx = np.where(valid_mask)[0]
            keep = rng.choice(valid_idx, MAX_WINDOWS_PER_SESSION, replace=False)
            keep.sort()
            feat = feat[:, keep, :]
            lab = lab[keep]

        sess_feat = np.zeros((n_links, feat.shape[1], 22), dtype=np.float32)
        for si, lid in enumerate(sess_lids):
            if lid in lid_to_idx:
                sess_feat[lid_to_idx[lid]] = feat[si]

        all_features.append(sess_feat)
        all_labels.append(lab[lab >= 0])
        del feat, sess_feat, lab
        gc.collect()

    if not all_features:
        return np.zeros((0, 0, 0)), np.zeros(0), []

    features = np.concatenate(all_features, axis=1)
    labels = np.concatenate(all_labels)
    del all_features, all_labels
    gc.collect()

    return features, labels, link_ids


def balance_classes(features, labels):
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0 or max(n_pos, n_neg) / min(n_pos, n_neg) <= MAX_CLASS_RATIO:
        return features, labels

    rng = np.random.default_rng(RNG_SEED)
    minority_n = min(n_pos, n_neg)
    target = int(minority_n * MAX_CLASS_RATIO)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if n_neg > n_pos:
        keep = np.sort(np.concatenate([pos_idx, rng.choice(neg_idx, target, replace=False)]))
    else:
        keep = np.sort(np.concatenate([rng.choice(pos_idx, target, replace=False), neg_idx]))
    return features[:, keep, :], labels[keep]


# ======================================================================
# Classifier evaluation
# ======================================================================

def evaluate_auc(features, labels, link_idx, window_idx=None):
    """Compute LR AUC on a subset of links and optionally a subset of time windows."""
    X = np.hstack([features[i, :, :] for i in link_idx])  # (n_windows, K*22)
    y = labels

    if window_idx is not None:
        X = X[window_idx]
        y = y[window_idx]

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    actual_folds = min(N_FOLDS, n_pos, n_neg)
    if actual_folds < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RNG_SEED)
    fold_aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, random_state=RNG_SEED)
        clf.fit(X_tr, y[tr])
        prob = clf.predict_proba(X_te)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y[te], prob))
        except ValueError:
            fold_aucs.append(0.5)
    return float(np.mean(fold_aucs))


# ======================================================================
# Factor A: Vary links at fixed temporal resolution
# ======================================================================

def factor_a(features, labels, link_counts):
    """Vary link count while keeping temporal resolution fixed.

    For each K, we use ALL time windows (same temporal resolution for all K).
    This isolates the spatial component: if AUC drops with K → spatial dilution.
    If AUC rises with K → no spatial dilution.
    """
    n_links = features.shape[0]
    rng = np.random.default_rng(RNG_SEED)

    print("\n  FACTOR A: Vary links, fixed temporal resolution", flush=True)
    results = {}
    for k in link_counts:
        if k > n_links:
            continue
        rand_aucs = []
        for _ in range(N_RANDOM_TRIALS):
            ri = rng.choice(n_links, k, replace=False)
            rand_aucs.append(evaluate_auc(features, labels, ri))

        mean_auc = float(np.mean(rand_aucs))
        std_auc = float(np.std(rand_aucs))
        results[k] = {"mean_auc": round(mean_auc, 4), "std_auc": round(std_auc, 4)}
        print(f"    K={k:4d}: AUC={mean_auc:.3f}±{std_auc:.3f}", flush=True)

    return results


# ======================================================================
# Factor B: Vary temporal resolution at fixed link count
# ======================================================================

def factor_b(features, labels, link_counts, rate_fractions):
    """Vary temporal resolution while keeping link count fixed.

    For each K and rate_fraction, subsample time windows.
    This isolates the temporal component.
    """
    n_links, n_windows, _ = features.shape
    rng = np.random.default_rng(RNG_SEED)

    print("\n  FACTOR B: Vary temporal resolution, fixed link count", flush=True)
    results = {}
    for k in link_counts:
        if k > n_links:
            continue
        results[k] = {}

        # Pick a fixed random link subset for this K
        link_idx = rng.choice(n_links, k, replace=False)

        for rate_frac in rate_fractions:
            n_keep = max(10, int(n_windows * rate_frac))
            window_idx = np.sort(rng.choice(n_windows, n_keep, replace=False))

            auc = evaluate_auc(features, labels, link_idx, window_idx)
            results[k][rate_frac] = round(auc, 4)
            print(f"    K={k:3d}, rate={rate_frac:.1f} ({n_keep:4d} windows): AUC={auc:.3f}", flush=True)

    return results


# ======================================================================
# Factor AB: Full 2D grid (links × temporal resolution)
# ======================================================================

def factor_ab(features, labels, link_counts, rate_fractions):
    """Full 2D grid: vary both link count and temporal resolution.

    Produces a matrix where rows = link count, columns = rate fraction.
    Each cell = mean AUC over N_RANDOM_TRIALS random link subsets.
    """
    n_links, n_windows, _ = features.shape
    rng = np.random.default_rng(RNG_SEED)

    print("\n  FACTOR A×B: Full 2D grid", flush=True)
    grid = {}
    for k in link_counts:
        if k > n_links:
            continue
        grid[k] = {}
        for rate_frac in rate_fractions:
            n_keep = max(10, int(n_windows * rate_frac))
            trial_aucs = []
            for _ in range(min(N_RANDOM_TRIALS, 10)):  # Fewer trials for 2D grid (speed)
                link_idx = rng.choice(n_links, k, replace=False)
                window_idx = np.sort(rng.choice(n_windows, n_keep, replace=False))
                trial_aucs.append(evaluate_auc(features, labels, link_idx, window_idx))

            mean_auc = float(np.mean(trial_aucs))
            grid[k][rate_frac] = round(mean_auc, 4)
            print(f"    K={k:3d}, rate={rate_frac:.1f}: AUC={mean_auc:.3f} "
                  f"(RSS={_rss_mb():.0f} MB)", flush=True)

    return grid


# ======================================================================
# Main
# ======================================================================

def main():
    _set_memory_limit()

    print("=" * 70, flush=True)
    print("EX3: TEMPORAL vs SPATIAL DECOMPOSITION", flush=True)
    print(f"  EWSN dir: {EWSN_DIR}", flush=True)
    print(f"  Output:   {OUT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    features, labels, link_ids = load_and_merge(EWSN_DIR)
    if features.shape[0] == 0:
        print("ERROR: no data loaded", flush=True)
        return

    features, labels = balance_classes(features, labels)
    n_links, n_windows, _ = features.shape
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"  {n_links} links, {n_windows} windows ({n_pos} pos, {n_neg} neg), "
          f"RSS={_rss_mb():.0f} MB", flush=True)

    # Factor A
    print("\n" + "=" * 70, flush=True)
    valid_ks = [k for k in LINK_COUNTS if k <= n_links]
    results_a = factor_a(features, labels, valid_ks)

    # Factor B
    print("\n" + "=" * 70, flush=True)
    valid_b_ks = [k for k in FACTOR_B_LINK_COUNTS if k <= n_links]
    results_b = factor_b(features, labels, valid_b_ks, RATE_FRACTIONS)

    # Factor AB (2D grid)
    print("\n" + "=" * 70, flush=True)
    grid = factor_ab(features, labels, valid_ks, RATE_FRACTIONS)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("INTERPRETATION", flush=True)
    print("=" * 70, flush=True)

    # Check Factor A: does AUC increase or decrease with K?
    a_values = [(k, v["mean_auc"]) for k, v in sorted(results_a.items())]
    if len(a_values) >= 3:
        first_auc = a_values[0][1]
        last_auc = a_values[-1][1]
        if last_auc > first_auc + 0.05:
            print("  Factor A: AUC INCREASES with K → NO spatial dilution", flush=True)
            print(f"    K={a_values[0][0]}→K={a_values[-1][0]}: "
                  f"AUC {first_auc:.3f}→{last_auc:.3f} (+{last_auc-first_auc:.3f})", flush=True)
            spatial_dilution = False
        elif last_auc < first_auc - 0.05:
            print("  Factor A: AUC DECREASES with K → spatial dilution IS real", flush=True)
            spatial_dilution = True
        else:
            print("  Factor A: AUC roughly flat → spatial effect is weak", flush=True)
            spatial_dilution = False

    # Check Factor B: does AUC increase with temporal resolution?
    temporal_effect = False
    for k in valid_b_ks:
        if k in results_b:
            rates = sorted(results_b[k].items())
            if len(rates) >= 2:
                low_rate_auc = rates[0][1]
                high_rate_auc = rates[-1][1]
                if high_rate_auc > low_rate_auc + 0.05:
                    temporal_effect = True
                    print(f"  Factor B (K={k}): AUC increases from {low_rate_auc:.3f} "
                          f"to {high_rate_auc:.3f} with more temporal data", flush=True)

    if temporal_effect and not spatial_dilution:
        verdict = "temporal_dominant"
        verdict_msg = ("TDMA undersampling is the dominant factor. Adding links at fixed "
                       "temporal resolution does NOT degrade performance. The 'dilution effect' "
                       "is a scheduling artifact, not a spatial physics limit.")
    elif spatial_dilution and not temporal_effect:
        verdict = "spatial_dominant"
        verdict_msg = "Spatial dilution is real and independent of temporal resolution."
    elif spatial_dilution and temporal_effect:
        verdict = "both_contribute"
        verdict_msg = "Both spatial dilution and temporal undersampling contribute."
    else:
        verdict = "neither_strong"
        verdict_msg = "Neither spatial nor temporal effects are strong in isolation."

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  {verdict_msg}", flush=True)

    # Save
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "EX3_temporal_spatial_decomposition",
        "environment": "office",
        "n_links": n_links,
        "n_windows": n_windows,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "factor_a_fixed_temporal": {str(k): v for k, v in results_a.items()},
        "factor_b_fixed_links": {str(k): {str(r): v for r, v in rv.items()}
                                 for k, rv in results_b.items()},
        "factor_ab_grid": {str(k): {str(r): v for r, v in rv.items()}
                           for k, rv in grid.items()},
        "verdict": verdict,
        "verdict_msg": verdict_msg,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
