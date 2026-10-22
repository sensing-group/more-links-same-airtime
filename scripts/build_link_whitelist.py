#!/usr/bin/env python3
"""The office link whitelist (results/decision_window_link_whitelist.json), an input of the
decision-window extraction in reproduce/office.sh.

A link enters the list when it yields a feature in at least MIN_FRACTION of the labelled 60 s
windows of an extraction at BASE_RATE_HZ (labels select the windows, never the links). The shipped
list came from the first 20 office sessions, two of which were still incomplete at the time
(history/first_office/sessions.txt); this script reproduces it from that extraction.

Usage: DATASET_DIR=data/office OUT_DIR=work/rate_decimated_base0p5 BASE_RATE_HZ=0.5 \
         RATE_FRACTIONS=1.0,0.5,0.2,0.1 python3 scripts/extract_rate_decimated_features.py
       IN_DIR=work/rate_decimated_base0p5 python3 scripts/build_link_whitelist.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = Path(os.environ.get("IN_DIR", str(ROOT / "work" / "rate_decimated_base0p5")))
OUT_PATH = Path(os.environ.get("OUT_PATH", str(ROOT / "work" / "decision_window_link_whitelist.json")))
MIN_FRACTION = 0.7


def main() -> None:
    files = sorted(IN_DIR.glob("rate_*.npz"))
    heads = [np.load(f, allow_pickle=True) for f in files]
    lids = sorted({str(x) for h in heads for x in h["link_ids"]})
    index = {x: i for i, x in enumerate(lids)}
    eligible = []
    for h in heads:
        ok = h["labels"] >= 0
        e = np.zeros((len(lids), int(ok.sum())), bool)
        e[[index[str(x)] for x in h["link_ids"]]] = np.isfinite(h["features"][0][:, ok, 0])
        eligible.append(e)
    fraction = np.concatenate(eligible, axis=1).mean(axis=1)
    whitelist = [lids[i] for i in np.where(fraction >= MIN_FRACTION)[0]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(whitelist, indent=1))
    print(f"{len(whitelist)} links from {len(files)} sessions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
