#!/usr/bin/env python3
"""
Two-factor ablation (link count K x per-link rate) on packet-decimated features.

Reads the per-session output of extract_rate_decimated_features.py. Every rate
shares the same eligible link-windows, the same link subsets and the same CV
splits, so the rate effect is a paired difference within each trial.

Reported per (K, rate):
  - logistic-regression AUC under stratified 5-fold CV (the paper's protocol)
    and under session-grouped 5-fold CV
  - a mask-only control: the same classifier on the eligibility mask alone
  - the OR-rule threshold detector: per-link z of subcarrier amplitude
    variance against training-fold empty windows, alarm if max z > Z_THRESHOLD;
    Youden's J and the AUC of the max-z score

Data is ~20 small npz files (tens of MB); everything fits in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from memlimit.memlimit import set_vm_limit  # noqa: E402

IN_DIR = Path(os.environ.get("IN_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "rate_decimated")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(IN_DIR / "rate_decimation_ablation.json")))

LINK_COUNTS = (1, 2, 5, 10, 20, 50, 100)
N_TRIALS = 20
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 1000
MAX_CLASS_RATIO = 3.0       # negatives kept per positive, as in the paper
Z_THRESHOLD = 2.0           # OR-rule per-link alarm threshold
THRESHOLD_FEATURE = "amp_var_subcarrier"
RNG_SEED = 42
# --complete-links: keep links eligible in at least this fraction of windows, then
# only the windows where all of them are eligible, so eligibility cannot carry occupancy
COMPLETE_LINK_MIN_FRACTION = 0.9

# links with no eligible window in a training fold give all-NaN columns; they are imputed, not errors
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")


def load(in_dir: Path):
    files = sorted(in_dir.glob("rate_*.npz"))
    if not files:
        sys.exit(f"no rate_*.npz in {in_dir}")
    heads = [np.load(f, allow_pickle=True) for f in files]
    link_ids = sorted({str(l) for h in heads for l in h["link_ids"]})
    li = {l: i for i, l in enumerate(link_ids)}
    rates = heads[0]["rate_fractions"]
    names = [str(n) for n in heads[0]["feature_names"]]
    X, Y, G = [], [], []
    for g, h in enumerate(heads):
        lab = h["labels"]
        ok = lab >= 0
        f = h["features"][:, :, ok, :]                       # (R, l, w, F)
        x = np.full((len(rates), len(link_ids), int(ok.sum()), len(names)), np.nan, np.float32)
        x[:, [li[str(l)] for l in h["link_ids"]]] = f
        X.append(x)
        Y.append(lab[ok].astype(int))
        G.append(np.full(int(ok.sum()), g))
    return (np.concatenate(X, axis=2), np.concatenate(Y), np.concatenate(G),
            link_ids, rates, names, [str(h["session_id"]) for h in heads])


def balance(y: np.ndarray, rng) -> np.ndarray:
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_neg = min(len(neg), int(MAX_CLASS_RATIO * len(pos)))
    return np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))


def lr_auc(X: np.ndarray, y: np.ndarray, splits) -> float:
    """Fold-wise mean imputation and standardisation; mean AUC over folds."""
    aucs = []
    for tr, te in splits:
        if len(set(y[te])) < 2:
            continue
        mu = np.nanmean(X[tr], axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        xtr = np.where(np.isfinite(X[tr]), X[tr], mu)
        xte = np.where(np.isfinite(X[te]), X[te], mu)
        sd = xtr.std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER)
        clf.fit((xtr - mu) / sd, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba((xte - mu) / sd)[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def or_rule(V: np.ndarray, y: np.ndarray, splits) -> tuple:
    """V: (windows, K) threshold feature. Returns (Youden J, AUC of max z)."""
    js, aucs = [], []
    for tr, te in splits:
        if len(set(y[te])) < 2:
            continue
        empty = V[tr][y[tr] == 0]
        mu = np.nanmean(empty, axis=0)
        sd = np.nanstd(empty, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.inf)
        z = (V[te] - np.where(np.isfinite(mu), mu, 0.0)) / sd
        score = np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=1)
        alarm = score > Z_THRESHOLD
        yt = y[te]
        js.append(alarm[yt == 1].mean() - alarm[yt == 0].mean())
        finite = np.where(np.isfinite(score), score, -1e9)
        aucs.append(roc_auc_score(yt, finite))
    return float(np.mean(js)), float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="report data shape and memory, fit nothing")
    ap.add_argument("--complete-links", action="store_true",
                    help=f"restrict to links eligible in >= {COMPLETE_LINK_MIN_FRACTION:.0%} of windows "
                         "and to windows where all of them are eligible")
    args = ap.parse_args()
    set_vm_limit()
    t0 = time.time()

    X, y, g, link_ids, rates, names, sessions = load(IN_DIR)
    rng = np.random.default_rng(RNG_SEED)
    keep = balance(y, rng)
    X, y, g = X[:, :, keep, :], y[keep], g[keep]
    link_counts = LINK_COUNTS
    if args.complete_links:
        e = np.isfinite(X[0, :, :, 0])
        links = np.where(e.mean(axis=1) >= COMPLETE_LINK_MIN_FRACTION)[0]
        wins = e[links].all(axis=0)
        X, y, g = X[:, links][:, :, wins], y[wins], g[wins]
        link_ids = [link_ids[i] for i in links]
        link_counts = tuple(k for k in LINK_COUNTS if k < len(links)) + (len(links),)
    n_r, n_l, n_w, n_f = X.shape
    elig = np.isfinite(X[0, :, :, 0]).T.astype(np.float32)        # (W, L), identical at every rate
    assert all(np.array_equal(np.isfinite(X[r, :, :, 0]), np.isfinite(X[0, :, :, 0])) for r in range(n_r))
    thr = names.index(THRESHOLD_FEATURE)
    print(f"{n_w} windows ({y.sum()} pos) from {len(set(g))} sessions "
          f"({len(set(g[y == 1]))} with positives), {n_l} links, rates {list(rates)}, "
          f"eligible link-windows {elig.mean():.3f}, X {X.nbytes / 1024 ** 2:.0f} MB", flush=True)
    if args.dry_run:
        return

    schemes = {
        "stratified": lambda s: list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(elig, y)),
        "session_grouped": lambda s: list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=s).split(elig, y, g)),
    }
    results = {name: {} for name in schemes}
    mask_varies = bool(elig.min() < 1)   # in complete-link mode the mask is constant
    for k in link_counts:
        if k > n_l:
            continue
        subsets = [rng.choice(n_l, k, replace=False) for _ in range(N_TRIALS)]
        for scheme, make in schemes.items():
            cell = {"lr_auc": [[] for _ in range(n_r)], "or_j": [[] for _ in range(n_r)],
                    "or_auc": [[] for _ in range(n_r)], "mask_auc": []}
            for t, sub in enumerate(subsets):
                splits = make(RNG_SEED + t)
                cell["mask_auc"].append(lr_auc(elig[:, sub], y, splits) if mask_varies else float("nan"))
                for r in range(n_r):
                    xr = X[r][sub].transpose(1, 0, 2).reshape(n_w, k * n_f)
                    cell["lr_auc"][r].append(lr_auc(xr, y, splits))
                    j, a = or_rule(X[r][sub][:, :, thr].T, y, splits)
                    cell["or_j"][r].append(j)
                    cell["or_auc"][r].append(a)
            lr = np.array(cell["lr_auc"])
            d = lr[0] - lr[-1]
            p = float(wilcoxon(lr[0], lr[-1]).pvalue) if np.any(d != 0) else 1.0
            results[scheme][k] = {
                "lr_auc_mean": [round(float(np.nanmean(v)), 4) for v in lr],
                "lr_auc_std": [round(float(np.nanstd(v)), 4) for v in lr],
                "rate_effect_full_minus_lowest": round(float(np.nanmean(d)), 4),
                "rate_effect_wilcoxon_p": p,
                "mask_only_auc": round(float(np.nanmean(cell["mask_auc"])), 4),
                "or_rule_j": [round(float(np.mean(v)), 4) for v in cell["or_j"]],
                "or_rule_auc": [round(float(np.mean(v)), 4) for v in cell["or_auc"]],
            }
            c = results[scheme][k]
            print(f"[{scheme:15s}] K={k:3d} LR AUC by rate {c['lr_auc_mean']} "
                  f"(full-low {c['rate_effect_full_minus_lowest']:+.3f}, p={p:.3g}) "
                  f"mask {c['mask_only_auc']:.3f} | OR J {c['or_rule_j']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "in_dir": str(IN_DIR),
        "sessions": sessions, "n_windows": int(n_w), "n_pos": int(y.sum()), "n_links": int(n_l),
        "rate_fractions": [float(r) for r in rates], "eligible_fraction": float(elig.mean()),
        "z_threshold": Z_THRESHOLD, "threshold_feature": THRESHOLD_FEATURE,
        "complete_links": args.complete_links, "link_ids": link_ids,
        "n_trials": N_TRIALS, "results": {s: {str(k): v for k, v in r.items()} for s, r in results.items()},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
