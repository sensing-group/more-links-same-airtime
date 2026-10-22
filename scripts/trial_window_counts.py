#!/usr/bin/env python3
"""
Scored windows per scripted trial and decision window, with the selection rules of
scripted_trial_decision_windows.py: pairs available in at least COMPLETE_LINK_MIN_FRACTION
of windows, windows in which all of them are available, minus the calibration span.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "trial_window_counts.json")))


def main() -> None:
    set_vm_limit()
    out = {}
    for t in st.TRIALS:
        links, phases = st.read_trial(st.TRIALS_DIR / t)
        out[t] = {}
        for w in st.DECISION_WINDOWS_S:
            F, labels, cal, _ = st.trial_features(links, phases, w)
            ok = np.isfinite(F[0, :, :, 0])
            sel = np.where(ok.mean(axis=1) >= st.COMPLETE_LINK_MIN_FRACTION)[0]
            wins = ok[sel].all(axis=0)
            out[t][f"{w:g}"] = int((wins & ~cal).sum())
        print(t, out[t], flush=True)
        del links
    OUT_PATH.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
