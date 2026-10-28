#!/usr/bin/env python3
"""
Does the fixed script order explain the trial results?

Every scripted trial runs empty, seated, walking, empty in that order, so a window's position
in its trial predicts its label. Two checks, leave-one-trial-out as in
scripted_trial_decision_windows.py:
  - position only: logistic regression on the window's relative position in its trial
    (polynomial of degree POSITION_DEGREE), no CSI
  - the paper's classifier (same features, subsets, standardization and model), scored
    against the opening and against the final empty phase separately; a classifier that
    read elapsed time instead of the room would score the final empty phase as occupied
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from memlimit.memlimit import set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "trial_phase_order_check.json")))
DECISION_WINDOW_S = 5.0
TASKS = ("occupancy", "presence")
SPLITS = ((1, 8.0), (2, 4.0), (4, 2.0), (8, 1.0), (16, 0.5))
POSITION_DEGREE = 3
MIN_PHASE_WINDOWS = 3


def empty_phases(labels) -> dict:
    occupied = np.where(np.isin(labels, st.OCCUPIED))[0]
    idx, empty = np.arange(len(labels)), labels == "empty_room"
    return {"opening_empty": empty & (idx < occupied.min()), "final_empty": empty & (idx > occupied.max())}


def fit_score(Xtr, ytr, Xte) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=st.LR_C, max_iter=st.LR_MAX_ITER).fit((Xtr - mu) / sd, ytr)
    return clf.predict_proba((Xte - mu) / sd)[:, 1]


def phase_aucs(y, score, masks) -> dict:
    out = {"all": float(roc_auc_score(y, score))}
    for key, m in masks.items():
        sel = (y == 1) | m
        if m.sum() >= MIN_PHASE_WINDOWS:
            out[key] = float(roc_auc_score(y[sel], score[sel]))
    return out


def main() -> None:
    set_vm_limit()
    rng = np.random.default_rng(st.RNG_SEED)
    rates = list(st.RATES_HZ)
    per = []
    for t in st.TRIALS:
        links, phases = st.read_trial(st.TRIALS_DIR / t)
        F, labels, cal, _ = st.trial_features(links, phases, DECISION_WINDOW_S)
        ok = np.isfinite(F[0, :, :, 0])
        sel = np.where(ok.mean(axis=1) >= st.COMPLETE_LINK_MIN_FRACTION)[0]
        wins = ok[sel].all(axis=0)
        per.append({"trial": t, "Z": st.zscore(F[:, sel][:, :, wins], cal[wins]), "labels": labels[wins],
                    "cal": cal[wins], "n_links": len(sel)})
        del links
        print(f"read {t}", flush=True)
    out = {"decision_window_s": DECISION_WINDOW_S, "trials": list(st.TRIALS), "position_only": {}, "classifier": {}}
    for task in TASKS:
        rows = []
        for p in per:
            y = st.task_labels(p["labels"], task)
            test = (~p["cal"]) & (y >= 0)
            pos = np.arange(len(y)) / len(y)
            masks = {k: m[test] for k, m in empty_phases(p["labels"]).items()}
            rows.append((np.column_stack([pos ** d for d in range(1, POSITION_DEGREE + 1)])[test], y[test], masks))
        res = [phase_aucs(rows[i][1], fit_score(np.vstack([r[0] for j, r in enumerate(rows) if j != i]),
                                                np.concatenate([r[1] for j, r in enumerate(rows) if j != i]), rows[i][0]),
                          rows[i][2]) for i in range(len(rows))]
        out["position_only"][task] = {k: round(float(np.mean([r[k] for r in res if k in r])), 4) for k in res[0]}
        print(f"position only, {task}: {out['position_only'][task]}", flush=True)
        for k, rate in SPLITS:
            r = rates.index(rate)
            n_sub = st.N_TRIALS if k < min(p["n_links"] for p in per) else 1
            acc = []
            for _ in range(n_sub):
                Xs, ys, ms = [], [], []
                for p in per:
                    sub = rng.choice(p["n_links"], k, replace=False)
                    y = st.task_labels(p["labels"], task)
                    test = (~p["cal"]) & (y >= 0)
                    Xs.append(np.nan_to_num(st.summarise(p["Z"][r][sub])[test]))
                    ys.append(y[test])
                    ms.append({key: m[test] for key, m in empty_phases(p["labels"]).items()})
                for i in range(len(per)):
                    Xtr = np.concatenate([Xs[j] for j in range(len(per)) if j != i])
                    ytr = np.concatenate([ys[j] for j in range(len(per)) if j != i])
                    acc.append(phase_aucs(ys[i], fit_score(Xtr, ytr, Xs[i]), ms[i]))
            cell = {key: round(float(np.mean([a[key] for a in acc if key in a])), 4) for key in acc[0]}
            out["classifier"][f"{task}|{k}|{rate:g}"] = cell
            print(f"classifier {task} {k}|{rate:g}: {cell}", flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
