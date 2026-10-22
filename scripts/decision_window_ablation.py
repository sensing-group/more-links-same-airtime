#!/usr/bin/env python3
"""
Decision window x link count x per-link rate, on a fixed sounding budget.

Reads one directory of extract_rate_decimated_features.py output per decision
window (DW_DIR/w<W>s/rate_*.npz). One link set serves every window: the
N_COMPLETE_LINKS links most often eligible in the shortest window (chosen
without labels). Each window keeps only the windows where all of them are
eligible, so the eligibility mask is constant and cannot carry occupancy
(under adaptive TDMA it otherwise does). Empty windows are then drawn as in
rate_decimation_ablation.py: with --empty-match time (the default) matched by
time of day within each training and each test fold, N_DRAWS times; every number is the mean
over draws.

Per (W, K, rate) cell, with session-grouped 5-fold CV and paired link subsets:
  - classifier: L2 logistic regression on K x 12 features, AUC
  - threshold OR rule: per-link z of subcarrier amplitude variance against
    training-fold empty windows, score = max over K, tau fit on the training
    fold to maximise Youden's J, J on the test fold
  - threshold vote: number of links with z > Z_VOTE, tau fit the same way
A rate whose decimated frame count is below MIN_DECIMATED_FRAMES has NaN
features: the cell is reported as undecidable, not scored.

The equal-budget view holds K x f constant (links x Hz = frames per second the
channel spends on these links) and reads the grid along its diagonals. With
--budget-unit tx, K counts transmitters and each subset takes every chosen link
of K random transmitters (one transmission is heard by all its receivers).

Time-only control: the same classifier on the time of day alone (sin and cos of
the UTC hour), scored on the same folds; it shows how much of a result the
clock could explain. Under --empty-match time it should sit near 0.5.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
from rate_decimation_ablation import (N_DRAWS, balance, time_bin, load, load_times, lr_auc,  # noqa: E402
                                     make_splits, time_only_auc)

DW_DIR = Path(os.environ.get("DW_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "decision_windows")))
OUT_PATH = os.environ.get("OUT_PATH")   # default: DW_DIR/decision_window_ablation[_<window>].json

LINK_COUNTS = (1, 2, 4, 8, 16, 32, 64)
TX_COUNTS = (1, 2, 4, 8)
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))
N_FOLDS = 5
N_COMPLETE_LINKS = int(os.environ.get("N_COMPLETE_LINKS", "16"))
THRESHOLD_FEATURE = "amp_var_subcarrier"
Z_VOTE = 2.0
N_TAU_CANDIDATES = 200
RNG_SEED = 42

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", message="All-NaN slice")


def best_tau(score: np.ndarray, y: np.ndarray) -> float:
    cand = np.unique(score)
    if len(cand) > N_TAU_CANDIDATES:
        cand = np.quantile(cand, np.linspace(0, 1, N_TAU_CANDIDATES))
    js = [(score[y == 1] > t).mean() - (score[y == 0] > t).mean() for t in cand]
    return float(cand[int(np.argmax(js))])


def threshold_j(V: np.ndarray, y: np.ndarray, splits) -> tuple:
    """V: (windows, K) threshold feature, all finite. Returns (J of max rule, J of vote rule)."""
    jm, jv = [], []
    for tr, te in splits:
        if len(set(y[te])) < 2:
            continue
        empty = V[tr][y[tr] == 0]
        sd = empty.std(axis=0)
        sd[sd == 0] = np.inf
        z = (V - empty.mean(axis=0)) / sd
        zmax, vote = z.max(axis=1), (z > Z_VOTE).sum(axis=1).astype(float)
        for score, acc in ((zmax, jm), (vote, jv)):
            t = best_tau(score[tr], y[tr])
            acc.append((score[te][y[te] == 1] > t).mean() - (score[te][y[te] == 0] > t).mean())
    return float(np.mean(jm)), float(np.mean(jv))


def draw_subset(k: int, n_l: int, tx: np.ndarray, unit: str, rng) -> np.ndarray:
    if unit == "link":
        return rng.choice(n_l, k, replace=False)
    return np.where(np.isin(tx, rng.choice(np.unique(tx), k, replace=False)))[0]


def select_links(w_dir: Path) -> list:
    """The N_COMPLETE_LINKS links most often eligible in this window; labels unused."""
    X, _, _, link_ids, _, _, _ = load(w_dir)
    order = np.argsort(-np.isfinite(X[0, :, :, 0]).mean(axis=1), kind="stable")
    return [link_ids[i] for i in order[:N_COMPLETE_LINKS]]


def analyse_window(w_dir: Path, rng, chosen: list, unit: str, empty_match: str) -> dict:
    X, y, g, link_ids, rates, names, _ = load(w_dir)
    t = load_times(w_dir)
    links = np.array([link_ids.index(l) for l in chosen if l in link_ids])
    wins = np.isfinite(X[0, links, :, 0]).all(axis=0)           # windows with every chosen link; labels unused
    X, y, g, t = X[:, links][:, :, wins], y[wins], g[wins], t[wins]
    tx = np.array([link_ids[i].split("->")[0] for i in links])
    n_r, n_l, _, n_f = X.shape
    decidable = [bool(np.isfinite(X[r]).all()) for r in range(n_r)]
    thr = names.index(THRESHOLD_FEATURE)
    slots = time_bin(t) if empty_match == "time" else None
    n_draws = N_DRAWS if empty_match == "time" else 1
    pre = balance(y, rng) if slots is None else None
    n_units = len(np.unique(tx)) if unit == "tx" else n_l
    counts = TX_COUNTS if unit == "tx" else LINK_COUNTS
    ks = [k for k in counts if k < n_units] + [n_units]
    subsets = {k: [draw_subset(k, n_l, tx, unit, rng) for _ in range(N_TRIALS if k < n_units else 1)] for k in ks}
    # scores[(k, r)]: (draws, subsets, 3) for AUC, J of the max rule, J of the vote rule
    scores = {(k, r): np.full((n_draws, len(subsets[k]), 3), np.nan) for k in ks for r in range(n_r) if decidable[r]}
    time_only, n_pos = [], []
    for d in range(n_draws):
        splits = make_splits(y, g, True, RNG_SEED, slots, pre, rng)
        n_pos.append(int(sum(y[te].sum() for _, te in splits)))
        time_only.append(time_only_auc(t, y, splits))
        for (k, r), arr in scores.items():
            for i, sub in enumerate(subsets[k]):
                arr[d, i, 0] = lr_auc(X[r][sub].transpose(1, 0, 2).reshape(len(y), len(sub) * n_f), y, splits)
                arr[d, i, 1:] = threshold_j(X[r][sub][:, :, thr].T, y, splits)
        print(f"[{w_dir.name}] draw {d + 1}/{n_draws}: {n_pos[-1]} occupied test windows, "
              f"time-only AUC {time_only[-1]:.3f}", flush=True)
    info = {"n_windows": int(len(y)), "n_pos": int(y.sum()), "n_pos_matched": int(np.mean(n_pos)),
            "kept_fraction": float(wins.mean()), "n_sessions": int(len(set(g))),
            "n_sessions_with_pos": int(len(set(g[y == 1]))), "n_links": int(n_l),
            "rate_fractions": [float(r) for r in rates], "decidable": decidable, "budget_unit": unit,
            "empty_match": empty_match, "n_draws": n_draws,
            "time_only_auc": round(float(np.mean(time_only)), 4),
            "time_only_auc_draw_sd": round(float(np.std(time_only)), 4), "cells": {}}
    r4 = lambda v: round(float(v), 4)
    for (k, r), arr in scores.items():
        per_subset = arr.mean(axis=0)                            # (subsets, 3), mean over draws
        per_draw = arr.mean(axis=1)                              # (draws, 3), mean over subsets
        info["cells"][f"{k}|{rates[r]:g}"] = {
            "auc": r4(per_subset[:, 0].mean()), "auc_sd": r4(per_subset[:, 0].std()),
            "auc_draw_sd": r4(per_draw[:, 0].std()),
            "auc_subsets": [r4(x) for x in per_subset[:, 0]],
            "j_max": r4(per_subset[:, 1].mean()), "j_vote": r4(per_subset[:, 2].mean()),
            "j_vote_draw_sd": r4(per_draw[:, 2].std())}
    for k in ks:
        row = "  ".join(f"f={rates[r]:g}: {info['cells'][f'{k}|{rates[r]:g}']['auc']:.3f}"
                        if decidable[r] else f"f={rates[r]:g}: --" for r in range(n_r))
        print(f"  K={k:3d} AUC {row}", flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--window", help="analyse only this window directory, e.g. w5s")
    ap.add_argument("--budget-unit", choices=("link", "tx"), default="link",
                    help="spend the budget per link, or per transmitter")
    ap.add_argument("--empty-match", choices=("time", "none"), default="time",
                    help="draw empty windows within each MATCH_BIN_S slot of the day (repeated N_DRAWS times), or from any hour once")
    args = ap.parse_args()
    set_vm_limit()
    t0 = time.time()
    dirs = sorted(DW_DIR.glob("w*s"), key=lambda d: float(d.name[1:-1]))
    chosen = select_links(dirs[0])
    if args.window:
        dirs = [d for d in dirs if d.name == args.window]
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "dw_dir": str(DW_DIR),
           "n_trials": N_TRIALS, "windows": {}}
    out["links"] = chosen
    for d in dirs:
        # one generator per window, so a window's result does not depend on which others ran
        rng = np.random.default_rng([RNG_SEED, int(float(d.name[1:-1]))])
        out["windows"][d.name] = analyse_window(d, rng, chosen, args.budget_unit, args.empty_match)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    out_path = Path(OUT_PATH) if OUT_PATH else DW_DIR / (
        f"decision_window_ablation{'_tx' if args.budget_unit == 'tx' else ''}"
        f"{'_' + args.window if args.window else ''}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
