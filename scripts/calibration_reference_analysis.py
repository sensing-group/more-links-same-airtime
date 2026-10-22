#!/usr/bin/env python3
"""
False alarms of a label-free threshold detector against the number of pairs
and the length of the empty-room reference, on the scripted trials.

Per trial, the first CAL_S seconds of the opening empty phase are the
reference: each pair's amplitude-variance feature is z-scored against the
reference windows, the score of K pairs is their maximum, and the threshold
is the (1 - ALPHA) quantile of the reference scores (no labels). Reported per
(decision window, reference span, K): the alarm rate on held-out empty windows
(false alarms) and on occupied windows (detections), averaged over trials and
random pair subsets.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "calibration_reference.json")))
DECISION_WINDOWS_S = (5.0, 10.0, 30.0)
REFERENCE_SPANS_S = (60.0, 120.0, 200.0, 290.0)
PAIR_COUNTS = (1, 2, 4, 8, 16, 32)
N_DRAWS = 20
RATE_INDEX = 0                     # the highest emulated rate (8 Hz)
RNG_SEED = 0


def main() -> None:
    set_vm_limit()
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    thr = st.FEATURE_NAMES.index(st.THRESHOLD_FEATURE)
    acc = {}
    for tdir in (st.TRIALS_DIR / t for t in st.TRIALS):
        links, phases = st.read_trial(tdir)
        for w in DECISION_WINDOWS_S:
            for span in REFERENCE_SPANS_S:
                st.CAL_S = span
                F, labels, cal, _ = st.trial_features(links, phases, w)
                ok = np.isfinite(F[0, :, :, 0])
                sel = np.where(ok.mean(axis=1) >= st.COMPLETE_LINK_MIN_FRACTION)[0]
                wins = ok[sel].all(axis=0)
                Z = st.zscore(F[:, sel][:, :, wins], cal[wins])[RATE_INDEX][:, :, thr]
                lab, c = labels[wins], cal[wins]
                empty, occ = (~c) & (lab == "empty_room"), (~c) & np.isin(lab, st.OCCUPIED)
                for k in (k for k in PAIR_COUNTS if k <= len(sel)):
                    for _ in range(N_DRAWS):
                        score = Z[rng.choice(len(sel), k, replace=False)].max(axis=0)
                        tau = np.quantile(score[c], 1 - st.ALPHA)
                        cell = acc.setdefault((w, span, k), {"fa": [], "det": [], "n_ref": []})
                        cell["fa"].append(float((score[empty] > tau).mean()))
                        cell["det"].append(float((score[occ] > tau).mean()))
                        cell["n_ref"].append(int(c.sum()))
        print(f"{tdir.name} done ({time.time() - t0:.0f}s)", flush=True)
        del links
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "alpha": st.ALPHA, "trials": list(st.TRIALS),
           "rate_hz": st.RATES_HZ[RATE_INDEX], "cells": []}
    for (w, span, k), v in sorted(acc.items()):
        out["cells"].append({"window_s": w, "reference_s": span, "pairs": k,
                             "reference_windows": round(float(np.mean(v["n_ref"])), 1),
                             "false_alarm": round(float(np.mean(v["fa"])), 4),
                             "detection": round(float(np.mean(v["det"])), 4)})
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
