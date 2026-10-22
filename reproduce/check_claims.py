#!/usr/bin/env python3
"""Check every data-derived number the paper prints against the analysis outputs.

reproduce/claims.csv maps each printed number to the JSON output that produces it. For each row the
expression in `expr` is evaluated against `source` (a path relative to the results directory), the
value is rounded to `decimals` and compared with `printed`. Exit status 0 means every claim matched.

Columns of claims.csv:
  id        unique claim identifier
  section   section of the paper (e.g. "5 Spending the sounding budget", "Table 3")
  context   a few words of the sentence or table cell, enough to find the number
  printed   the number as the paper prints it
  source    JSON output, relative to the results directory
  expr      Python expression giving the value (see helpers below)
  decimals  decimal places the paper rounds to (for a percentage, of the percent; -1 rounds to tens)
  note      other places the same number is printed, and remarks

Helpers available in `expr` (paths are '/'-separated keys from the JSON root; an integer segment
indexes a list; keys may contain '.' and '|'):
  v(path, src=None)           value at path in the row's source, or in results/<src> when given
  find(path, src=None, **kw)  the element of the list at path whose fields equal kw
  keys(path, src=None)        keys of the object at path
  wins(a, b)                  dicts: number of keys where a[k] > b[k]; a scalar and b a list: number
                              of elements of b below a
  median(xs), mean(xs)        of a sequence
  and the builtins abs, all, any, float, int, len, max, min, round, sorted, sum, tuple, zip.

Convention for `printed`:
  "0.94", "43%", "9,913", "$-$0.01"  one number; '%', thousands commas and a typeset minus are
                                     stripped. If expr returns a sequence, every element must round
                                     to this number (e.g. "stays near 0.1" over several cells).
  "0.13 to 0.19"                     a range; expr returns (low, high), each compared.
  "18/20"                            a count out of a total; expr returns (count, total).
  "<0.01", ">0.92"                   a bound the sentence states; the unrounded value (every element
                                     of a sequence) must satisfy it.
  "~0.91"                            "about" or "near": every element within one unit of the last
                                     printed decimal (one unit of 10 when decimals is -1).
  "six", "eleven"                    number words are read as digits.
Rounding is decimal, on the shortest representation of the value: a value exactly halfway between
two printed values (7.55 to one decimal) matches either, since the paper's tables format with
round-half-even and its text rounds half up.

Usage: python3 reproduce/check_claims.py [--claims reproduce/claims.csv] [--results results] [--verbose]
--results points the check at a rerun (e.g. work/) instead of the shipped outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "reproduce" / "claims.csv"
RESULTS = ROOT / "results"
EQ_TOL = 1e-9          # float noise allowed when comparing a rounded value with the printed one
RELATIONS = ("<=", ">=", "<", ">", "~")
RANGE_SEP = " to "
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
SAFE_BUILTINS = {f.__name__: f for f in (abs, all, any, float, int, len, max, min, round, sorted, sum, tuple, zip)}


class Source:
    """JSON outputs under one results directory, loaded once each."""

    def __init__(self, results: Path):
        self.results = results
        self.cache: dict[str, object] = {}

    def load(self, rel: str):
        if rel not in self.cache:
            self.cache[rel] = json.loads((self.results / rel).read_text())
        return self.cache[rel]

    def get(self, rel: str, path: str):
        x = self.load(rel)
        for seg in [s for s in path.split("/") if s]:
            x = x[int(seg)] if isinstance(x, list) else x[seg]
        return x


def helpers(store: Source, default: str) -> dict:
    def v(path, src=None):
        return store.get(src or default, path)

    def find(path, src=None, **kw):
        hits = [e for e in v(path, src) if all(e.get(k) == want for k, want in kw.items())]
        if len(hits) != 1:
            raise LookupError(f"find({path!r}, {kw}) matched {len(hits)} elements")
        return hits[0]

    def keys(path, src=None):
        return list(v(path, src).keys())

    def wins(a, b):
        if isinstance(a, dict):
            return sum(a[k] > b[k] for k in b)
        return sum(a > x for x in b)

    return {"__builtins__": SAFE_BUILTINS, "v": v, "find": find, "keys": keys, "wins": wins,
            "median": statistics.median, "mean": statistics.fmean}


def parse_number(tok: str) -> float:
    tok = tok.strip().replace("$-$", "-").replace("−", "-").replace(",", "").rstrip("%").strip()
    return float(NUMBER_WORDS[tok.lower()]) if tok.lower() in NUMBER_WORDS else float(tok)


def parse_printed(printed: str) -> tuple[str, list[float]]:
    """Return (relation, numbers): relation is '=' or one of RELATIONS."""
    s = printed.strip()
    rel = next((r for r in RELATIONS if s.startswith(r)), "=")
    if rel != "=":
        s = s[len(rel):]
    if RANGE_SEP in s:
        parts = s.split(RANGE_SEP)
    elif "/" in s:
        parts = s.split("/")
    else:
        parts = [s]
    return rel, [parse_number(p) for p in parts]


def rounded(x: float, decimals: int) -> tuple[float, float]:
    """The two decimal roundings of x (equal unless x lies exactly halfway)."""
    q = Decimal(1).scaleb(-decimals)
    d = Decimal(repr(float(x)))
    return tuple(float(d.quantize(q, rounding=m)) + 0.0 for m in (ROUND_HALF_DOWN, ROUND_HALF_UP))


def as_list(x) -> list[float]:
    return [float(e) for e in x] if isinstance(x, (list, tuple)) else [float(x)]


def compare(rel: str, want: list[float], got: list[float], decimals: int) -> bool:
    if any(math.isnan(g) for g in got):
        return False
    if len(want) == 2:                       # range or count/total: element by element
        if len(got) != 2:
            return False
        pairs = list(zip(got, want))
    else:                                    # one number: every computed element against it
        pairs = [(g, want[0]) for g in got]
    unit = 10.0 ** (-decimals)
    for g, w in pairs:
        if rel == "<" and not g < w: return False
        if rel == ">" and not g > w: return False
        if rel == "<=" and not g <= w: return False
        if rel == ">=" and not g >= w: return False
        if rel == "~" and not abs(g - w) <= unit + EQ_TOL: return False
        if rel == "=" and not any(abs(r - w) <= EQ_TOL for r in rounded(g, decimals)): return False
    return True


def show(got: list[float], decimals: int) -> str:
    fmt = (lambda g: f"{rounded(g, decimals)[1]:.{max(decimals, 0)}f}")
    return ", ".join(f"{fmt(g)} ({g:.6g})" for g in got)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=str(CLAIMS))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--verbose", action="store_true", help="print passing claims too")
    a = ap.parse_args()
    store = Source(Path(a.results))
    rows = list(csv.DictReader(open(a.claims, newline="")))
    ids = [r["id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        print(f"duplicate claim ids: {', '.join(dup)}")
        return 1
    failed = 0
    for r in rows:
        decimals = int(r["decimals"])
        try:
            rel, want = parse_printed(r["printed"])
            got = as_list(eval(r["expr"], helpers(store, r["source"])))
            ok = compare(rel, want, got, decimals)
            detail = show(got, decimals)
        except Exception as e:  # a missing key or file is a failed claim, not a crash
            ok, detail = False, f"error: {type(e).__name__}: {e}"
        if not ok:
            failed += 1
        if not ok or a.verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {r['id']:<10} [{r['section']}] \"{r['context']}\": "
                  f"printed {r['printed']}, computed {detail}")
    print(f"\n{'ALL CLAIMS MATCH' if failed == 0 else 'NOT ALL CLAIMS MATCH'}: "
          f"{len(rows) - failed} of {len(rows)} claims match, {failed} differ")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
