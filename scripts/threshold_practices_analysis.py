#!/usr/bin/env python3
"""Threshold-detector evaluation practices on the original office features (ghost_*_ewsn.npz, amp_var = feature 4).

paper     : max of RAW amp_var over K links, tau chosen in-sample to maximise J (dilution_analysis.py)
z_max_cv  : per-link z vs training-fold EMPTY windows, zero-filled link-windows = missing,
            max over K, tau = argmax J on the TRAINING fold, J measured on the test fold
z_cfar_cv : same score, tau = (1-ALPHA) empirical quantile of the max over training EMPTY windows
z_mean_cv : mean of z over K (instead of max), tau = argmax J on training fold
vote_cv   : number of links with z > Z_VOTE, tau = argmax J on training fold
Folds: StratifiedGroupKFold by session (5). Random-K links, N_SUB subsets per K.
Empty windows as in rate_decimation_ablation.py: with --empty-match time (the default)
matched by time of day within each training and each test fold, N_DRAWS draws, and
"paper" is scored in-sample on the matched windows of each draw; --empty-match none
keeps CLASS_RATIO empty windows per occupied one from any hour, drawn once.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import temporal_spatial_decomposition as tsd  # noqa: E402
from memlimit.memlimit import set_vm_limit  # noqa: E402
from rate_decimation_ablation import N_DRAWS, balance, make_splits, time_bin  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "threshold_practices.json")))

ALPHA, Z_VOTE, N_SUB, SEED = 0.05, 2.0, 30, 42
KS = [1, 2, 3, 5, 10, 20, 50, 100, 226]
N_FOLDS = 5
N_TAU = 200
N_TAU_INSAMPLE = 100
CLASS_RATIO = 3

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--empty-match", choices=("time", "none"), default="time")
args = ap.parse_args()

set_vm_limit()
files = sorted(tsd.EWSN_DIR.glob("ghost_*_ewsn.npz"))
lids = sorted({l for f in files for l in np.load(f, allow_pickle=True)["link_ids"]})
li = {l: i for i, l in enumerate(lids)}
A, Y, G, T = [], [], [], []
for g, f in enumerate(files):
    d = np.load(f, allow_pickle=True); lab = d["labels"]; ok = lab >= 0
    T.append(d["timestamps"][ok])
    a = np.full((len(lids), ok.sum()), np.nan, np.float32)
    for si, l in enumerate(d["link_ids"]):
        v = d["features"][si][ok]
        live = np.abs(v).sum(1) > 0
        a[li[l], live] = v[live, 4]
    A.append(a); Y.append(lab[ok]); G.append(np.full(ok.sum(), g))
A = np.concatenate(A, 1); Y = np.concatenate(Y); G = np.concatenate(G); T = np.concatenate(T)
rng = np.random.default_rng(SEED)
if args.empty_match == "none":
    k = balance(Y, rng, ratio=CLASS_RATIO)
    draws = [make_splits(Y, G, True, SEED, None, k, rng)]
else:
    slots = time_bin(T)
    draws = [make_splits(Y, G, True, SEED, slots, None, rng) for _ in range(N_DRAWS)]

def best_tau(s, y):
    c = np.unique(s[np.isfinite(s)])
    if len(c) > N_TAU: c = np.quantile(c, np.linspace(0, 1, N_TAU))
    js = [(s[y == 1] > t).mean() - (s[y == 0] > t).mean() for t in c]
    return c[int(np.argmax(js))]

def j_at(s, y, t): return (s[y == 1] > t).mean() - (s[y == 0] > t).mean()

def scores(sub, tr, te):
    e = A[sub][:, tr][:, Y[tr] == 0]
    mu, sd = np.nanmean(e, 1, keepdims=True), np.nanstd(e, 1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.inf); mu = np.where(np.isfinite(mu), mu, 0)
    z = (A[sub] - mu) / sd
    zmax = np.nanmax(np.where(np.isfinite(z), z, -np.inf), 0)
    zmax = np.where(np.isfinite(zmax), zmax, -1e9)
    zmean = np.nan_to_num(np.nanmean(z, 0), nan=-1e9)
    vote = np.nansum(z > Z_VOTE, 0).astype(float)
    return zmax, zmean, vote

out = {}
for K in KS:
    if K > A.shape[0]: K = A.shape[0]
    r = {m: [] for m in ("paper", "z_max_cv", "z_cfar_cv", "z_mean_cv", "vote_cv")}
    for _ in range(N_SUB if K < A.shape[0] else 1):
        sub = rng.choice(A.shape[0], K, replace=False)
        raw_all = np.nan_to_num(A[sub], nan=0.0).max(0)
        for folds in draws:
            scored = np.concatenate([te for _, te in folds])      # every matched window, once
            raw, ys = raw_all[scored], Y[scored]
            r["paper"].append(max(j_at(raw, ys, t) for t in np.linspace(raw.min(), raw.max(), N_TAU_INSAMPLE)))
            jm, jc, jn, jv = [], [], [], []
            for tr, te in folds:
                zmax, zmean, vote = scores(sub, tr, te)
                ytr, yte = Y[tr], Y[te]
                jm.append(j_at(zmax[te], yte, best_tau(zmax[tr], ytr)))
                jc.append(j_at(zmax[te], yte, np.quantile(zmax[tr][ytr == 0], 1 - ALPHA)))
                jn.append(j_at(zmean[te], yte, best_tau(zmean[tr], ytr)))
                jv.append(j_at(vote[te], yte, best_tau(vote[tr], ytr)))
            r["z_max_cv"].append(np.mean(jm)); r["z_cfar_cv"].append(np.mean(jc))
            r["z_mean_cv"].append(np.mean(jn)); r["vote_cv"].append(np.mean(jv))
    out[K] = {m: float(np.mean(v)) for m, v in r.items()}
    print(f"K={K:3d} " + "  ".join(f"{m}={v:.3f}" for m, v in out[K].items()), flush=True)

n_scored = [sum(len(te) for _, te in f) for f in draws]
n_pos = [sum(int(Y[te].sum()) for _, te in f) for f in draws]
OUT_PATH.write_text(json.dumps({"empty_match": args.empty_match, "n_draws": len(draws),
                                "n_windows": int(np.mean(n_scored)), "n_pos": int(np.mean(n_pos)), "n_links": int(A.shape[0]),
                                "alpha": ALPHA, "z_vote": Z_VOTE, "by_k": {str(k): v for k, v in out.items()}}, indent=2))
print(f"wrote {OUT_PATH}")
