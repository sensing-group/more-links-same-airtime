#!/usr/bin/env python3
"""
Decision window x link count x per-link rate on the scripted 8-node trials.

The trials run at ~20 Hz per link (the 14-node office at ~1 Hz), so they test
the regime where temporal resolution could matter. Each trial follows one
protocol: empty, occupied-static, occupied-moving, empty. Two tasks:
  occupancy : empty vs occupied (static or moving)
  motion    : occupied-moving vs occupied-static

Deployment-like protocol per trial: the first CAL_S seconds of the opening
empty phase calibrate every link; detection is scored on the rest. A link is
eligible in a window when it holds at least BASE_RATE_HZ * W frames; each
trial keeps its links eligible in >= COMPLETE_LINK_MIN_FRACTION of windows
and the windows where all of them are eligible, then equalizes and decimates
every link to each rate (evenly spaced frames).

With --budget-unit tx the unit of spending is a transmitter, not a link: one
transmission is heard by every receiver, so K counts transmitters and each
subset takes all complete links of K random transmitters.

Detectors, for random K-unit subsets:
  max / vote : per-link z of subcarrier amplitude variance against the
               calibration windows; score = max over links, or number of links
               with z > Z_VOTE; tau = (1 - ALPHA) quantile of the calibration
               scores (no labels used); Youden's J on the scored windows
  classifier : L2 logistic regression on link-order-free summaries (mean, sd,
               max, min, median over the K links of each z-scored feature),
               trained leave-one-trial-out; AUC on the held-out trial
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from memlimit.memlimit import set_vm_limit  # noqa: E402
from extract_rate_decimated_features import FEATURE_NAMES, decimated_indices, link_window_features  # noqa: E402

TRIALS_DIR = Path(os.environ.get("TRIALS_DIR", str(Path(__file__).resolve().parent.parent / "data" / "trials")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "scripted_trial_decision_windows.json")))
TRIALS = ("T0003_residence", "T0004_residence", "T0008_laboratory",
          "T0009_laboratory", "T0010_laboratory", "T0011_laboratory")

DECISION_WINDOWS_S = tuple(float(w) for w in os.environ.get("DECISION_WINDOWS_S", "1,2,5,10,30").split(","))
BASE_RATE_HZ = 8.0
RATES_HZ = (8.0, 4.0, 2.0, 1.0, 0.5)
LINK_COUNTS = (1, 2, 4, 8, 16, 32)
TX_COUNTS = (1, 2, 4, 8)          # --budget-unit tx: transmitters, each feeding all its receivers
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))
CAL_S = 200.0                 # calibration span at the start of the opening empty phase
PHASE_GUARD_S = 5.0           # windows this close to a phase boundary are dropped
COMPLETE_LINK_MIN_FRACTION = 0.9
MIN_FRAMES = 2
ALPHA = 0.05
Z_VOTE = 2.0
N_SUBCARRIERS = 52
THRESHOLD_FEATURE = "amp_var_subcarrier"
LR_C = 1.0
LR_MAX_ITER = 1000
HGB_MAX_ITER = 200                # gradient-boosted trees (--classifier hgb)
RNG_SEED = 42
OCCUPIED = ("occupied_static", "occupied_moving")
# tasks: occupancy (empty vs occupied), motion (seated vs walking), presence (empty vs seated)
TASKS = tuple(os.environ.get("TASKS", "occupancy,motion").split(","))
DEFAULT_TASKS = ("occupancy", "motion")

warnings.filterwarnings("ignore", message="Mean of empty slice")


def read_trial(tdir: Path):
    """Per-link time-ordered (ts array, (n, 52) amplitude array); phases from annotations."""
    per_link = defaultdict(list)
    with open(tdir / "raw" / "csi_raw.jsonl.zst", "rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), errors="replace")
        for line in text:
            try:
                d = json.loads(line)
            except ValueError:
                continue                         # a cut or corrupt line
            iq = d.get("iq_data")
            if "tx_mac" not in d or not iq or len(iq) != N_SUBCARRIERS:
                continue
            a = np.asarray(iq, dtype=np.float32)
            per_link[f"{d['tx_mac']}->{d['rx_mac']}"].append((d["wall_ts"], np.hypot(a[:, 0], a[:, 1])))
    links = {}
    for lid, rows in per_link.items():
        rows.sort(key=lambda r: r[0])
        links[lid] = (np.array([r[0] for r in rows]), np.stack([r[1] for r in rows]))
    phases = [json.loads(l) for l in open(tdir / "annotations.jsonl")]
    phases = sorted(((p["start_ts"], p["end_ts"], p["fine_label"]) for p in phases), key=lambda p: p[0])
    return links, phases


def window_label(lo: float, hi: float, phases) -> str | None:
    for s, e, lab in phases:
        if s + PHASE_GUARD_S <= lo and hi <= e - PHASE_GUARD_S:
            return lab
    return None


def trial_features(links, phases, w: float):
    """Features (R, L, n_win, F), window labels, calibration mask, link ids, for decision window w."""
    t0 = phases[0][0]
    t_end = max(ts[-1] for ts, _ in links.values())
    starts = np.arange(t0, t_end - w, w)
    labels = np.array([window_label(s, s + w, phases) or "" for s in starts])
    cal = (starts + w <= t0 + CAL_S) & (labels == "empty_room")
    use = cal | ((starts >= t0 + CAL_S) & (labels != ""))
    starts, labels, cal = starts[use], labels[use], cal[use]
    lids = sorted(links)
    base = int(round(BASE_RATE_HZ * w))
    keep_n = [int(round(r * w)) for r in RATES_HZ]
    F = np.full((len(RATES_HZ), len(lids), len(starts), len(FEATURE_NAMES)), np.nan, np.float32)
    for li, lid in enumerate(lids):
        ts, amp = links[lid]
        lo = np.searchsorted(ts, starts)
        hi = np.searchsorted(ts, starts + w)
        for wi in range(len(starts)):
            n = hi[wi] - lo[wi]
            if n < base:
                continue
            seg = amp[lo[wi]:hi[wi]]
            for r, k in enumerate(keep_n):
                if k >= MIN_FRAMES:
                    F[r, li, wi] = link_window_features(seg[decimated_indices(n, k)])
    return F, labels, cal, lids


def zscore(F, cal):
    mu = np.nanmean(F[:, :, cal, :], axis=2, keepdims=True)
    sd = np.nanstd(F[:, :, cal, :], axis=2, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.inf)
    return (F - mu) / sd


def summarise(Z):
    """Z: (K, n_win, F) -> (n_win, 5F) link-order-free summary."""
    return np.concatenate([Z.mean(0), Z.std(0), Z.max(0), Z.min(0), np.median(Z, 0)], axis=1)


def task_labels(labels, task):
    if task == "occupancy":
        y = np.where(np.isin(labels, OCCUPIED), 1, np.where(labels == "empty_room", 0, -1))
    elif task == "presence":
        y = np.where(labels == "occupied_static", 1, np.where(labels == "empty_room", 0, -1))
    else:
        y = np.where(labels == "occupied_moving", 1, np.where(labels == "occupied_static", 0, -1))
    return y


def n_units(p, unit: str) -> int:
    return len(set(p["tx"])) if unit == "tx" else p["n_links"]


def draw_subset(p, k: int, unit: str, rng) -> np.ndarray:
    """Link indices for k links, or for every complete link of k transmitters."""
    if unit == "link":
        return rng.choice(p["n_links"], k, replace=False)
    txs = rng.choice(sorted(set(p["tx"])), k, replace=False)
    return np.where(np.isin(p["tx"], txs))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="list trials and windows, read nothing")
    ap.add_argument("--budget-unit", choices=("link", "tx"), default="link",
                    help="spend the budget per link, or per transmitter (one transmission feeds all its receivers)")
    ap.add_argument("--classifier", choices=("lr", "hgb"), default="lr",
                    help="logistic regression, or gradient-boosted trees")
    args = ap.parse_args()
    set_vm_limit()
    t0 = time.time()
    trials = [TRIALS_DIR / t for t in TRIALS if (TRIALS_DIR / t / "raw" / "csi_raw.jsonl.zst").exists()]
    print(f"{len(trials)} trials, windows {DECISION_WINDOWS_S} s, rates {RATES_HZ} Hz", flush=True)
    if args.dry_run:
        return
    rng = np.random.default_rng(RNG_SEED)
    suffix = (("_tx" if args.budget_unit == "tx" else "") + ("_hgb" if args.classifier == "hgb" else "")
              + ("" if TASKS == DEFAULT_TASKS else "_" + "_".join(TASKS)))
    out_path = OUT_PATH.with_name(f"{OUT_PATH.stem}{suffix}.json")
    thr = FEATURE_NAMES.index(THRESHOLD_FEATURE)

    # per window: per trial (z-scored features restricted to complete links/windows, labels, cal mask)
    data = {w: [] for w in DECISION_WINDOWS_S}
    for tdir in trials:
        links, phases = read_trial(tdir)
        for w in DECISION_WINDOWS_S:
            F, labels, cal, lids = trial_features(links, phases, w)
            ok = np.isfinite(F[0, :, :, 0])
            sel = np.where(ok.mean(axis=1) >= COMPLETE_LINK_MIN_FRACTION)[0]
            wins = ok[sel].all(axis=0)
            Z = zscore(F[:, sel][:, :, wins], cal[wins])
            data[w].append({"trial": tdir.name, "Z": Z, "labels": labels[wins], "cal": cal[wins], "n_links": len(sel),
                            "tx": np.array([lids[i].split("->")[0] for i in sel])})
        print(f"  read {tdir.name}: {len(links)} links "
              f"complete per window {[d[-1]['n_links'] for d in [data[w] for w in DECISION_WINDOWS_S]]} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del links

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials": [t.name for t in trials],
           "rates_hz": RATES_HZ, "base_rate_hz": BASE_RATE_HZ, "cal_s": CAL_S, "alpha": ALPHA,
           "budget_unit": args.budget_unit, "results": {}}
    for w in DECISION_WINDOWS_S:
        per = data[w]
        n_min = min(n_units(p, args.budget_unit) for p in per)
        counts = TX_COUNTS if args.budget_unit == "tx" else LINK_COUNTS
        for task in TASKS:
            res = {}
            for k in [k for k in counts if k < n_min] + [n_min]:
                for r, rate in enumerate(RATES_HZ):
                    if int(round(rate * w)) < MIN_FRAMES:
                        res[f"{k}|{rate:g}"] = None          # fewer than MIN_FRAMES samples: undecidable
                        continue
                    jmax, jvote, aucs, aps = [], [], [], []
                    by_trial = defaultdict(list)
                    for _ in range(N_TRIALS if k < n_min else 1):
                        subs = [draw_subset(p, k, args.budget_unit, rng) for p in per]
                        Xs, ys = [], []
                        for p, sub in zip(per, subs):
                            Z = p["Z"][r][sub]                                  # (K, n_win, F)
                            y = task_labels(p["labels"], task)
                            test = (~p["cal"]) & (y >= 0)
                            zt = Z[:, :, thr]
                            for score, acc in ((zt.max(0), jmax), ((zt > Z_VOTE).sum(0).astype(float), jvote)):
                                tau = np.quantile(score[p["cal"]], 1 - ALPHA)
                                yt, st = y[test], score[test]
                                if len(set(yt)) == 2:
                                    acc.append((st[yt == 1] > tau).mean() - (st[yt == 0] > tau).mean())
                            Xs.append(np.nan_to_num(summarise(Z)[test]))
                            ys.append(y[test])
                        for i in range(len(per)):                                  # leave one trial out
                            if len(set(ys[i])) < 2:
                                continue
                            Xtr = np.concatenate([Xs[j] for j in range(len(per)) if j != i])
                            ytr = np.concatenate([ys[j] for j in range(len(per)) if j != i])
                            mu, sd = Xtr.mean(0), Xtr.std(0)
                            sd[sd == 0] = 1.0
                            clf = (HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER, random_state=RNG_SEED)
                                   if args.classifier == "hgb" else
                                   LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER)).fit((Xtr - mu) / sd, ytr)
                            prob = clf.predict_proba((Xs[i] - mu) / sd)[:, 1]
                            aucs.append(roc_auc_score(ys[i], prob))
                            aps.append(average_precision_score(ys[i], prob))
                            by_trial[per[i]["trial"]].append(aucs[-1])
                    res[f"{k}|{rate:g}"] = {"auc": round(float(np.mean(aucs)), 4),
                                            "j_max": round(float(np.mean(jmax)), 4),
                                            "j_vote": round(float(np.mean(jvote)), 4),
                                            "pr_auc": round(float(np.mean(aps)), 4),
                                            "auc_by_trial": {tr: round(float(np.mean(v)), 4) for tr, v in by_trial.items()}}
                row = "  ".join(f"{rate:g}Hz:" + (f"{res[f'{k}|{rate:g}']['auc']:.2f}/{res[f'{k}|{rate:g}']['j_vote']:.2f}"
                                                  if res[f"{k}|{rate:g}"] else "--") for rate in RATES_HZ)
                print(f"[{task:9s} W={w:g}s] K={k:2d} AUC/Jvote {row}", flush=True)
            out["results"][f"{task}|{w:g}"] = res
        out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
