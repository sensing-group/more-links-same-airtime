#!/usr/bin/env python3
"""Compare a rerun (work/) with the shipped analysis outputs (results/).

Every JSON file under results/ is matched with the file at the same relative path under work/,
and every numeric leaf is compared. A value matches when it is within ATOL or within RTOL of the
shipped value. Exit status 0 means every compared value matched.

Usage: python3 reproduce/compare.py [--work work] [--results results] [--atol 1e-6] [--rtol 1e-6]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATOL = 1e-6
RTOL = 1e-6
MAX_REPORTED = 10   # mismatching values printed per file
# Outputs of optional steps (reproduce/apartment.sh needs the apartment labels): reported, not required
OPTIONAL = ("apartment/",)


def leaves(x, path=""):
    """Yield (path, value) for every numeric leaf of a JSON document."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def close(a: float, b: float, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= max(atol, rtol * abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--atol", type=float, default=ATOL)
    ap.add_argument("--rtol", type=float, default=RTOL)
    a = ap.parse_args()
    work, results = Path(a.work), Path(a.results)
    total_bad = total_missing = 0
    for ref in sorted(results.rglob("*.json")):
        rel = ref.relative_to(results)
        mine = work / rel
        if not mine.exists():
            optional = str(rel).startswith(OPTIONAL)
            print(f"{'SKIPPED ' if optional else 'MISSING '} {rel}{' (optional step not run)' if optional else ''}")
            total_missing += not optional
            continue
        want = dict(leaves(json.loads(ref.read_text())))
        got = dict(leaves(json.loads(mine.read_text())))
        absent = sorted(set(want) - set(got))
        bad = [(k, want[k], got[k]) for k in want if k in got and not close(got[k], want[k], a.atol, a.rtol)]
        status = "OK      " if not bad and not absent else "DIFFERS "
        print(f"{status} {rel}: {len(want)} values, {len(bad)} differ, {len(absent)} absent")
        for k, w, g in bad[:MAX_REPORTED]:
            print(f"           {k}: shipped {w:.6g}, rerun {g:.6g}")
        total_bad += len(bad) + len(absent)
    print(f"\n{'ALL MATCH' if total_bad == 0 and total_missing == 0 else 'NOT ALL MATCH'}: "
          f"{total_bad} differing or absent values, {total_missing} files not rerun")
    return 0 if total_bad == 0 and total_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
