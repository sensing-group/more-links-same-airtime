#!/usr/bin/env python3
"""
Decision window x link count x per-link rate, on a fixed sounding budget.

Reads one directory of extract_rate_decimated_features.py output per decision
window (DW_DIR/w<W>s/rate_*.npz). One link set serves every window: the
N_COMPLETE_LINKS links most often eligible in the shortest window (chosen
without labels). Each window keeps only the windows where all of them are
eligible, so the eligibility mask is constant and cannot carry occupancy
(under adaptive TDMA it otherwise does).

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
clock could explain when the classes are not spread evenly over the day.
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
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
from rate_decimation_ablation import balance, load, lr_auc  # noqa: E402

DW_DIR = Path(os.environ.get("DW_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "decision_windows")))
OUT_PATH = os.environ.get("OUT_PATH")   # default: DW_DIR/decision_window_ablation[_<window>].json

LINK_COUNTS = (1, 2, 4, 8, 16, 32, 64)
TX_COUNTS = (1, 2, 4, 8)
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))
N_FOLDS = 5
N_COMPLETE_LINKS = int(os.environ.get("N_COMPLETE_LINKS", "16"))
SECONDS_PER_DAY = 86400.0
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


def load_times(in_dir: Path) -> np.ndarray:
    """Window start times in the order load() returns windows."""
    out = []
    for f in sorted(in_dir.glob("rate_*.npz")):
        h = np.load(f, allow_pickle=True)
        out.append(h["timestamps"][h["labels"] >= 0])
    return np.concatenate(out)


def time_only_auc(t: np.ndarray, y: np.ndarray, splits) -> float:
    phase = 2 * np.pi * (t % SECONDS_PER_DAY) / SECONDS_PER_DAY
    return lr_auc(np.column_stack([np.sin(phase), np.cos(phase)]), y, splits)


def draw_subset(k: int, n_l: int, tx: np.ndarray, unit: str, rng) -> np.ndarray:
    if unit == "link":
        return rng.choice(n_l, k, replace=False)
    return np.where(np.isin(tx, rng.choice(np.unique(tx), k, replace=False)))[0]


def select_links(w_dir: Path) -> list:
    """The N_COMPLETE_LINKS links most often eligible in this window; labels unused."""
    X, _, _, link_ids, _, _, _ = load(w_dir)
    order = np.argsort(-np.isfinite(X[0, :, :, 0]).mean(axis=1), kind="stable")
    return [link_ids[i] for i in order[:N_COMPLETE_LINKS]]


def analyse_window(w_dir: Path, rng, chosen: list, unit: str) -> dict:
    X, y, g, link_ids, rates, names, _ = load(w_dir)
    t = load_times(w_dir)
    keep = balance(y, rng)
    X, y, g, t = X[:, :, keep, :], y[keep], g[keep], t[keep]
    base_ok = np.isfinite(X[0, :, :, 0])
    links = np.array([link_ids.index(l) for l in chosen if l in link_ids])
    wins = base_ok[links].all(axis=0)
    X, y, g, t = X[:, links][:, :, wins], y[wins], g[wins], t[wins]
    tx = np.array([link_ids[i].split("->")[0] for i in links])
    n_r, n_l, n_w, n_f = X.shape
    decidable = [bool(np.isfinite(X[r]).all()) for r in range(n_r)]
    thr = names.index(THRESHOLD_FEATURE)
    splits = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=RNG_SEED).split(X[0, 0], y, g))
    info = {"n_windows": int(n_w), "n_pos": int(y.sum()), "kept_fraction": float(wins.mean()), "n_sessions": int(len(set(g))),
            "n_sessions_with_pos": int(len(set(g[y == 1]))), "n_links": int(n_l),
            "rate_fractions": [float(r) for r in rates], "decidable": decidable, "budget_unit": unit,
            "time_only_auc": round(time_only_auc(t, y, splits), 4), "cells": {}}
    print(f"[{w_dir.name}] {n_w} windows ({y.sum()} pos, {info['n_sessions_with_pos']} sessions with pos), "
          f"{n_l} complete links, decidable by rate {decidable}, time-only AUC {info['time_only_auc']:.3f}", flush=True)
    n_units = len(np.unique(tx)) if unit == "tx" else n_l
    counts = TX_COUNTS if unit == "tx" else LINK_COUNTS
    for k in [k for k in counts if k < n_units] + [n_units]:
        subsets = [draw_subset(k, n_l, tx, unit, rng) for _ in range(N_TRIALS if k < n_units else 1)]
        for r in range(n_r):
            if not decidable[r]:
                continue
            auc, jmax, jvote = [], [], []
            for sub in subsets:
                auc.append(lr_auc(X[r][sub].transpose(1, 0, 2).reshape(n_w, len(sub) * n_f), y, splits))
                a, b = threshold_j(X[r][sub][:, :, thr].T, y, splits)
                jmax.append(a)
                jvote.append(b)
            info["cells"][f"{k}|{rates[r]:g}"] = {
                "auc": round(float(np.mean(auc)), 4), "auc_sd": round(float(np.std(auc)), 4),
                "auc_subsets": [round(float(x), 4) for x in auc],
                "j_max": round(float(np.mean(jmax)), 4), "j_vote": round(float(np.mean(jvote)), 4)}
        row = "  ".join(f"f={rates[r]:g}: {info['cells'][f'{k}|{rates[r]:g}']['auc']:.3f}"
                        if decidable[r] else f"f={rates[r]:g}: --" for r in range(n_r))
        print(f"  K={k:3d} AUC {row}", flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--window", help="analyse only this window directory, e.g. w5s")
    ap.add_argument("--budget-unit", choices=("link", "tx"), default="link",
                    help="spend the budget per link, or per transmitter")
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
        out["windows"][d.name] = analyse_window(d, rng, chosen, args.budget_unit)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    out_path = Path(OUT_PATH) if OUT_PATH else DW_DIR / (
        f"decision_window_ablation{'_tx' if args.budget_unit == 'tx' else ''}"
        f"{'_' + args.window if args.window else ''}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
