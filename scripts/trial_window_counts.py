#!/usr/bin/env python3
"""
Scored windows per scripted trial and decision window, with the selection rules of
scripted_trial_decision_windows.py: links available in at least COMPLETE_LINK_MIN_FRACTION
of windows, windows in which all of them are available, minus the calibration span.
The same windows counted per class (empty_room, occupied_static, occupied_moving) go to
CLASS_OUT_PATH, for the class balance of each task.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "trial_window_counts.json")))
CLASS_OUT_PATH = Path(os.environ.get("CLASS_OUT_PATH", str(OUT_PATH.with_name("trial_class_counts.json"))))
CLASSES = ("empty_room", "occupied_static", "occupied_moving")


def main() -> None:
    set_vm_limit()
    out, classes = {}, {}
    for t in st.TRIALS:
        links, phases = st.read_trial(st.TRIALS_DIR / t)
        out[t], classes[t] = {}, {}
        for w in st.DECISION_WINDOWS_S:
            F, labels, cal, _ = st.trial_features(links, phases, w)
            ok = np.isfinite(F[0, :, :, 0])
            sel = np.where(ok.mean(axis=1) >= st.COMPLETE_LINK_MIN_FRACTION)[0]
            wins = ok[sel].all(axis=0)
            out[t][f"{w:g}"] = int((wins & ~cal).sum())
            counts = Counter(labels[wins & ~cal].tolist())
            classes[t][f"{w:g}"] = {c: int(counts[c]) for c in CLASSES}
        print(t, out[t], flush=True)
        del links
    OUT_PATH.write_text(json.dumps(out, indent=2))
    CLASS_OUT_PATH.write_text(json.dumps(classes, indent=2))


if __name__ == "__main__":
    main()
