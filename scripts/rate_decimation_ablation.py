#!/usr/bin/env python3
"""
Two-factor ablation (link count K x per-link rate) on packet-decimated features.

Reads the per-session output of extract_rate_decimated_features.py. Every rate
shares the same eligible link-windows, the same link subsets and the same CV
splits, so the rate effect is a paired difference within each trial.

Empty windows: with --empty-match time (the default) the folds are drawn on all
windows first; then, separately in each training and each test fold, every
MATCH_BIN_S slot of the day keeps MATCHED_CLASS_RATIO empty windows per occupied
one, drawn from the same fold. Both classes then share one time-of-day profile in
every fold, so the clock carries no signal in either direction (matching across
folds lets a classifier learn the times of the held-out occupied windows as
empty; hour slots still leave a time-only AUC near 0.6 at 60 s windows). The draw
is repeated N_DRAWS times; every number is the mean over draws, with the spread
across draws beside it. --empty-match none keeps MAX_CLASS_RATIO empty windows
per occupied one from any hour, drawn once before the folds (the protocol of the
earlier analyses).

Reported per (K, rate):
  - logistic-regression AUC under stratified 5-fold CV (the paper's protocol)
    and under session-grouped 5-fold CV
  - a mask-only control: the same classifier on the eligibility mask alone
  - the OR-rule threshold detector: per-link z of subcarrier amplitude
    variance against training-fold empty windows, alarm if max z > Z_THRESHOLD;
    Youden's J and the AUC of the max-z score
and a time-only control: the same classifier on the time of day alone.

Data is one small npz file per session (tens of MB in all); everything fits in memory.
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
from ghost_eval.memlimit import set_vm_limit  # noqa: E402

IN_DIR = Path(os.environ.get("IN_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "rate_decimated")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(IN_DIR / "rate_decimation_ablation.json")))

LINK_COUNTS = (1, 2, 5, 10, 20, 50, 100)
N_TRIALS = 20
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 1000
MAX_CLASS_RATIO = 3.0       # negatives kept per positive, as in the paper
MATCHED_CLASS_RATIO = 1     # negatives per positive within a slot of a fold; 3 would drop many positives
MATCH_BIN_S = 900.0         # time-of-day slot (15 min) within which empty windows are matched
Z_THRESHOLD = 2.0           # OR-rule per-link alarm threshold
THRESHOLD_FEATURE = "amp_var_subcarrier"
RNG_SEED = 42
N_DRAWS = int(os.environ.get("N_DRAWS", "10"))   # empty-window draws under --empty-match time
SECONDS_PER_DAY = 86400.0
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


def load_times(in_dir: Path) -> np.ndarray:
    """Window start times (UTC epoch s) in the order load() returns windows."""
    out = []
    for f in sorted(in_dir.glob("rate_*.npz")):
        h = np.load(f, allow_pickle=True)
        out.append(h["timestamps"][h["labels"] >= 0])
    return np.concatenate(out)


def time_bin(t: np.ndarray) -> np.ndarray:
    """Index of the MATCH_BIN_S slot of the (UTC) day."""
    return ((t % SECONDS_PER_DAY) // MATCH_BIN_S).astype(int)


def balance(y: np.ndarray, rng, slots: np.ndarray | None = None, ratio: float = MAX_CLASS_RATIO) -> np.ndarray:
    """Every occupied window and `ratio` empty windows per occupied one. With slots,
    the empty windows are drawn within each time-of-day slot; a slot with too few
    empty windows keeps only as many occupied ones as it can match."""
    if slots is None:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(ratio * len(pos)))
        return np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
    keep = [np.zeros(0, int)]
    for h in np.unique(slots[y == 1]):
        pos, neg = np.where((y == 1) & (slots == h))[0], np.where((y == 0) & (slots == h))[0]
        n_pos = min(len(pos), int(len(neg) // ratio))
        if n_pos:
            keep += [rng.choice(pos, n_pos, replace=False), rng.choice(neg, int(ratio * n_pos), replace=False)]
    return np.sort(np.concatenate(keep))


def make_splits(y: np.ndarray, g: np.ndarray, grouped: bool, seed: int, slots: np.ndarray | None,
                pre: np.ndarray | None, rng) -> list:
    """Folds as (train, test) indices into the full window arrays. With slots, the
    folds are drawn on all windows and each side is slot-matched within itself;
    otherwise the folds are drawn on the pre-balanced windows `pre`."""
    idx = np.arange(len(y)) if slots is not None else pre
    cv = (StratifiedGroupKFold if grouped else StratifiedKFold)(N_FOLDS, shuffle=True, random_state=seed)
    folds = cv.split(np.zeros(len(idx)), y[idx], g[idx]) if grouped else cv.split(np.zeros(len(idx)), y[idx])
    if slots is None:
        return [(idx[tr], idx[te]) for tr, te in folds]
    match = lambda part: part[balance(y[part], rng, slots[part], MATCHED_CLASS_RATIO)]
    return [(match(tr), match(te)) for tr, te in folds]


def time_only_auc(t: np.ndarray, y: np.ndarray, splits) -> float:
    """The classifier on the time of day alone (sin and cos of the UTC hour)."""
    phase = 2 * np.pi * (t % SECONDS_PER_DAY) / SECONDS_PER_DAY
    return lr_auc(np.column_stack([np.sin(phase), np.cos(phase)]), y, splits)


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


def run_draw(X, y, t, elig, subsets, n_r, thr, splits: dict) -> dict:
    """One draw of empty windows: per scheme and K, per-subset scores of shape (rates, subsets).
    splits[scheme][i] are the folds of subset i (the same folds for every K)."""
    n_w, n_f = X.shape[2], X.shape[3]
    mask_varies = bool(elig.min() < 1)   # in complete-link mode the mask is constant
    out = {}
    for scheme, folds in splits.items():
        out[scheme] = {"time_only_auc": time_only_auc(t, y, folds[0])}
        for k, subs in subsets.items():
            cell = {"lr_auc": np.full((n_r, len(subs)), np.nan), "or_j": np.full((n_r, len(subs)), np.nan),
                    "or_auc": np.full((n_r, len(subs)), np.nan), "mask_auc": np.full(len(subs), np.nan)}
            for i, sub in enumerate(subs):
                if mask_varies:
                    cell["mask_auc"][i] = lr_auc(elig[:, sub], y, folds[i])
                for r in range(n_r):
                    cell["lr_auc"][r, i] = lr_auc(X[r][sub].transpose(1, 0, 2).reshape(n_w, k * n_f), y, folds[i])
                    cell["or_j"][r, i], cell["or_auc"][r, i] = or_rule(X[r][sub][:, :, thr].T, y, folds[i])
            out[scheme][k] = cell
    return out


def summarise(draws: list, scheme: str, k: int) -> dict:
    """Mean over draws and subsets; subset spread of the draw mean; draw spread of the subset mean."""
    lr = np.array([d[scheme][k]["lr_auc"] for d in draws])          # (draws, rates, subsets)
    per_subset = lr.mean(axis=0)
    d = per_subset[0] - per_subset[-1]
    mask = np.array([d_[scheme][k]["mask_auc"] for d_ in draws])
    or_j = np.array([d_[scheme][k]["or_j"] for d_ in draws])
    or_auc = np.array([d_[scheme][k]["or_auc"] for d_ in draws])
    r4 = lambda v: [round(float(x), 4) for x in v]
    return {
        "lr_auc_mean": r4(np.nanmean(lr, axis=(0, 2))),
        "lr_auc_std": r4(np.nanstd(per_subset, axis=1)),
        "lr_auc_draw_sd": r4(np.nanstd(np.nanmean(lr, axis=2), axis=0)),
        "rate_effect_full_minus_lowest": round(float(np.nanmean(d)), 4),
        "rate_effect_wilcoxon_p": float(wilcoxon(per_subset[0], per_subset[-1]).pvalue) if np.any(d != 0) else 1.0,
        "mask_only_auc": round(float(np.nanmean(mask)), 4),
        "mask_only_auc_draw_sd": round(float(np.nanstd(np.nanmean(mask, axis=1))), 4),
        "or_rule_j": r4(np.nanmean(or_j, axis=(0, 2))),
        "or_rule_auc": r4(np.nanmean(or_auc, axis=(0, 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="report data shape and memory, fit nothing")
    ap.add_argument("--complete-links", action="store_true",
                    help=f"restrict to links eligible in >= {COMPLETE_LINK_MIN_FRACTION:.0%} of windows "
                         "and to windows where all of them are eligible")
    ap.add_argument("--empty-match", choices=("time", "none"), default="time",
                    help="draw empty windows within each MATCH_BIN_S slot of the day (repeated N_DRAWS times), or from any hour once")
    args = ap.parse_args()
    set_vm_limit()
    t0 = time.time()

    X, y, g, link_ids, rates, names, sessions = load(IN_DIR)
    t = load_times(IN_DIR)
    rng = np.random.default_rng(RNG_SEED)
    link_counts = LINK_COUNTS
    if args.complete_links:       # chosen on all windows, without labels
        e = np.isfinite(X[0, :, :, 0])
        links = np.where(e.mean(axis=1) >= COMPLETE_LINK_MIN_FRACTION)[0]
        wins = e[links].all(axis=0)
        X, y, g, t = X[:, links][:, :, wins], y[wins], g[wins], t[wins]
        link_ids = [link_ids[i] for i in links]
        link_counts = tuple(k for k in LINK_COUNTS if k < len(links)) + (len(links),)
    n_r, n_l = X.shape[0], X.shape[1]
    assert all(np.array_equal(np.isfinite(X[r, :, :, 0]), np.isfinite(X[0, :, :, 0])) for r in range(n_r))
    thr = names.index(THRESHOLD_FEATURE)
    print(f"{X.shape[2]} windows ({y.sum()} pos) from {len(set(g))} sessions "
          f"({len(set(g[y == 1]))} with positives), {n_l} links, rates {list(rates)}, "
          f"X {X.nbytes / 1024 ** 2:.0f} MB, empty match {args.empty_match}", flush=True)
    if args.dry_run:
        return

    slots = time_bin(t) if args.empty_match == "time" else None
    n_draws = N_DRAWS if args.empty_match == "time" else 1
    pre = balance(y, rng) if slots is None else None
    subsets = {k: [rng.choice(n_l, k, replace=False) for _ in range(N_TRIALS)] for k in link_counts if k <= n_l}
    elig = np.isfinite(X[0, :, :, 0]).T.astype(np.float32)        # (W, L), identical at every rate
    draws, n_pos = [], []
    for d in range(n_draws):
        splits = {scheme: [make_splits(y, g, scheme == "session_grouped", RNG_SEED + i, slots, pre, rng)
                           for i in range(N_TRIALS)] for scheme in ("stratified", "session_grouped")}
        n_pos.append(int(sum(y[te].sum() for _, te in splits["session_grouped"][0])))
        draws.append(run_draw(X, y, t, elig, subsets, n_r, thr, splits))
        print(f"draw {d + 1}/{n_draws}: {n_pos[-1]} occupied test windows, time-only AUC "
              f"{draws[-1]['session_grouped']['time_only_auc']:.3f} session-grouped, "
              f"{draws[-1]['stratified']['time_only_auc']:.3f} stratified ({time.time() - t0:.0f}s)", flush=True)

    results = {}
    for scheme in ("stratified", "session_grouped"):
        to = [d[scheme]["time_only_auc"] for d in draws]
        results[scheme] = {"time_only_auc": round(float(np.mean(to)), 4), "time_only_auc_draw_sd": round(float(np.std(to)), 4)}
        for k in subsets:
            c = results[scheme][str(k)] = summarise(draws, scheme, k)
            print(f"[{scheme:15s}] K={k:3d} LR AUC by rate {c['lr_auc_mean']} (draw sd {c['lr_auc_draw_sd']}) "
                  f"mask {c['mask_only_auc']:.3f} | OR J {c['or_rule_j']}", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "in_dir": str(IN_DIR),
        "sessions": sessions, "empty_match": args.empty_match, "n_draws": n_draws,
        "n_windows": int(len(y)), "n_pos": int(y.sum()), "n_pos_matched": int(np.mean(n_pos)),
        "n_links": int(n_l), "rate_fractions": [float(r) for r in rates], "eligible_fraction": float(elig.mean()),
        "z_threshold": Z_THRESHOLD, "threshold_feature": THRESHOLD_FEATURE,
        "complete_links": args.complete_links, "link_ids": link_ids,
        "n_trials": N_TRIALS, "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
