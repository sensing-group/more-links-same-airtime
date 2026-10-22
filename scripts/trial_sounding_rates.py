#!/usr/bin/env python3
"""Soundings per second of each pair in each scripted trial (Section 4: "about 20 times per second").

Counts the raw records of every transmitter-receiver pair over the trial's duration.
Reads TRIALS_DIR (default data/trials), writes OUT_PATH (default work/trial_sounding_rates.json).
"""
from __future__ import annotations

import io
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import zstandard

ROOT = Path(__file__).resolve().parent.parent
TRIALS_DIR = Path(os.environ.get("TRIALS_DIR", str(ROOT / "data" / "trials")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(ROOT / "work" / "trial_sounding_rates.json")))


def pair_rates(raw: Path) -> np.ndarray:
    counts, t0, t1 = Counter(), None, None
    with open(raw, "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue                         # a cut or corrupt line
            if "tx_mac" not in d:
                continue
            ts = d["wall_ts"]
            t0, t1 = (ts, ts) if t0 is None else (min(t0, ts), max(t1, ts))
            counts[(d["tx_mac"], d["rx_mac"])] += 1
    return np.array(sorted(c / (t1 - t0) for c in counts.values()))


def main() -> None:
    out = {}
    for tdir in sorted(p for p in TRIALS_DIR.iterdir() if (p / "raw" / "csi_raw.jsonl.zst").exists()):
        r = pair_rates(tdir / "raw" / "csi_raw.jsonl.zst")
        out[tdir.name] = {"n_pairs": len(r), "median_hz": round(float(np.median(r)), 2),
                          "min_hz": round(float(r.min()), 2), "max_hz": round(float(r.max()), 2)}
        print(tdir.name, out[tdir.name], flush=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
