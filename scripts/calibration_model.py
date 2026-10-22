#!/usr/bin/env python3
"""
Finite-reference model of the label-free threshold's false alarms.

Each pair's empty-room feature is an independent standard normal. A reference
of m windows estimates each pair's mean and spread; the score of K pairs is the
largest standardized value; the threshold is the (1 - ALPHA) quantile of the m
reference scores; the false-alarm rate is the share of fresh empty windows
above it. The model is evaluated at every (pairs, reference) cell of
calibration_reference.json and written beside the measured rate, together with
the reference length the model needs for a target false-alarm rate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = Path(os.environ.get("IN_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "calibration_reference.json")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(REPO_ROOT / "data" / "csi-sense-trials" / "calibration_model.json")))
N_REPEATS = 4000           # simulated references per cell
N_TEST = 200               # fresh empty windows per reference
BATCH = 20
RNG_SEED = 0
TARGETS = (0.10, 0.15)     # false-alarm rates for the required-reference table
REQUIRED_PAIRS = (1, 2, 4, 8, 16, 32, 64)
M_MIN, M_MAX = 4, 1024     # bisection range of the reference length, windows


def model_false_alarm(k: int, m: int, alpha: float, rng) -> float:
    rates = []
    for _ in range(N_REPEATS // BATCH):
        ref = rng.standard_normal((BATCH, k, m))
        test = rng.standard_normal((BATCH, k, N_TEST))
        mu, sd = ref.mean(axis=2, keepdims=True), ref.std(axis=2, keepdims=True)
        tau = np.quantile(((ref - mu) / sd).max(axis=1), 1 - alpha, axis=1)
        rates.append((((test - mu) / sd).max(axis=1) > tau[:, None]).mean(axis=1))
    return float(np.mean(np.concatenate(rates)))


def required_reference(k: int, target: float, alpha: float, rng) -> int:
    """Smallest reference length (windows) whose modeled false-alarm rate is at most target."""
    lo, hi = M_MIN, M_MAX
    while lo < hi:
        mid = (lo + hi) // 2
        if model_false_alarm(k, mid, alpha, rng) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    d = json.loads(IN_PATH.read_text())
    for c in d["cells"]:
        m = int(round(c["reference_windows"]))
        c["model_false_alarm"] = round(model_false_alarm(c["pairs"], m, d["alpha"], rng), 4) if m >= 2 else None
    d["model"] = "independent Gaussian pairs, mean and spread estimated from the reference"
    d["required_reference_windows"] = {f"{t:g}": {str(k): required_reference(k, t, d["alpha"], rng) for k in REQUIRED_PAIRS}
                                       for t in TARGETS}
    OUT_PATH.write_text(json.dumps(d, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
