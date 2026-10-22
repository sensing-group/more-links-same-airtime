#!/usr/bin/env python3
"""
Rate-decimated features for the 9-node apartment release, written in the format
of extract_rate_decimated_features.py (OUT_DIR/w<W>s/rate_<day>.npz), so that
decision_window_ablation.py analyses it unchanged with days as sessions.

The mesh ran a fixed round-robin schedule. One sounding is one trigger: the
burst frames a receiver records for one (tx, rx, seq) are averaged into one
amplitude vector (52 subcarriers).

Labels come from the door sensor, never from the camera (the camera label of
this release is a function of the hour of day: occupied 07:00-22:00, empty
otherwise, and it reports occupancy inside 420 door-confirmed empty minutes):
  empty    (0): minutes inside a door-confirmed empty period, EDGE_GUARD_S from its ends
  occupied (1): minutes of hours labelled lr_occupied=1, HOME_GUARD_S from any empty period
Only minutes whose UTC hour is in LABEL_HOURS_UTC are labelled, the hours in
which empty periods occur, so both classes come from the same time of day.

Every processed hour also writes a 1 Hz activity series (mean over pairs of the
normalised sounding-to-sounding change). --check-clock reads them and compares
the activity around door-open events at several clock offsets.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from ghost_eval.memlimit import safe_pool_map, set_vm_limit  # noqa: E402
from extract_rate_decimated_features import FEATURE_NAMES, decimated_indices, link_window_features  # noqa: E402

DATASET_DIR = Path(os.environ.get("DATASET_DIR", str(REPO_ROOT / "dataset-release-apartment" / "extracted" / "dataset_release")))
DOOR_EVENTS = Path(os.environ.get("DOOR_EVENTS", str(REPO_ROOT / "data" / "ground_truth" / "door_sensor_events.csv")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "apartment_windows")))

DECISION_WINDOWS_S = tuple(float(w) for w in os.environ.get("DECISION_WINDOWS_S", "10,20,30,60").split(","))
BASE_RATE_HZ = float(os.environ.get("BASE_RATE_HZ", "0.5"))
RATE_FRACTIONS = tuple(float(f) for f in os.environ.get("RATE_FRACTIONS", "1.0,0.5,0.25").split(","))
CLOCK_OFFSET_S = float(os.environ.get("CLOCK_OFFSET_S", "0"))   # added to the CSI clock to reach UTC
LABEL_HOURS_UTC = range(int(os.environ.get("LABEL_HOUR_FIRST", "10")), int(os.environ.get("LABEL_HOUR_LAST", "21")))
FILE_HOURS = range(LABEL_HOURS_UTC.start - 1, LABEL_HOURS_UTC.stop + 1)   # one spare hour each side
EXCLUDED_DAYS = ("2026-01-13",)     # 5 h of data, all at night
EDGE_GUARD_S = 300.0
HOME_GUARD_S = 1800.0
MIN_DECIMATED_FRAMES = 2
N_SUBCARRIERS = 52
TRIGGER_SPAN_S = 1.0                # frames of one (tx, rx, seq) farther apart than this are separate triggers
CLOCK_OFFSETS_S = (-7200, -3600, -1800, 0, 1800, 3600, 7200)
PERI_EVENT_S = 30.0
BASELINE_S = (600.0, 120.0)         # baseline: from 600 s to 120 s before the event
EST_BYTES_PER_WORKER = int(1.5 * 1024 ** 3)

log = logging.getLogger("apartment_budget")


def utc(s: str) -> float:
    dt = datetime.fromisoformat(s)
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def minute_labels() -> dict:
    """UTC minute epoch -> 1 (occupied) or 0 (empty), door sensor and hourly labels only."""
    lab = DATASET_DIR / "labels"
    empty = [(utc(p["start"]), utc(p["end"])) for p in json.load(open(lab / "confirmed_empty_periods.json"))]
    occupied_hours = {utc(r["timestamp"]) for r in csv.DictReader(open(lab / "living_room_occupancy_labels.csv"))
                      if r["lr_occupied"] == "1"}
    out = {}
    for a, b in empty:
        for m in range(int((a + EDGE_GUARD_S) // 60 + 1) * 60, int((b - EDGE_GUARD_S) // 60) * 60, 60):
            out[m] = 0
    for h in occupied_hours:
        for m in range(int(h), int(h) + 3600, 60):
            if all(m + 60 <= a - HOME_GUARD_S or m >= b + HOME_GUARD_S for a, b in empty):
                out[m] = 1
    keep = {m: v for m, v in out.items()
            if datetime.fromtimestamp(m, timezone.utc).hour in LABEL_HOURS_UTC
            and datetime.fromtimestamp(m, timezone.utc).strftime("%Y-%m-%d") not in EXCLUDED_DAYS}
    return keep


def read_triggers(path: Path):
    """Per pair: time-ordered sounding times (UTC) and (n, 52) mean burst amplitudes."""
    open_trig, done = {}, defaultdict(list)

    def flush(pair):
        t0, frames = open_trig.pop(pair)[1:]
        a = np.asarray(frames, dtype=np.float32)
        done[pair].append((t0, np.hypot(a[:, 0::2], a[:, 1::2]).mean(axis=0)))

    with open(path, "rb") as f:
        for line in f:
            try:
                d = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            csi = d.get("csi_data")
            if not csi or len(csi) != 2 * N_SUBCARRIERS:
                continue
            pair, t = (d["tx_mac"], d["rx_mac"]), utc(d["received_at"]) + CLOCK_OFFSET_S
            cur = open_trig.get(pair)
            if cur is not None and (cur[0] != d["seq"] or t - cur[1] > TRIGGER_SPAN_S):
                flush(pair)
                cur = None
            if cur is None:
                open_trig[pair] = (d["seq"], t, [csi])
            else:
                cur[2].append(csi)
    for pair in list(open_trig):
        flush(pair)
    out = {}
    for pair, rows in done.items():
        rows.sort(key=lambda r: r[0])
        out[f"{pair[0]}->{pair[1]}"] = (np.array([r[0] for r in rows]), np.stack([r[1] for r in rows]))
    return out


def activity_series(links: dict):
    """1 Hz series: mean over pairs of |change of subcarrier-mean amplitude| / that pair's median change."""
    acc = defaultdict(list)
    for ts, amp in links.values():
        d = np.abs(np.diff(amp.mean(axis=1)))
        med = np.median(d)
        if len(d) == 0 or med <= 0:
            continue
        for sec, v in zip(ts[1:].astype(np.int64), d / med):
            acc[sec].append(v)
    secs = np.array(sorted(acc), dtype=np.int64)
    return secs, np.array([np.mean(acc[s]) for s in secs], dtype=np.float32)


def process_hour(item):
    path, labels = item
    t0 = time.time()
    links = read_triggers(path)
    secs, act = activity_series(links)
    lids = sorted(links)
    res = {"file": f"{path.parent.name}/{path.name}", "secs": secs, "act": act, "windows": {}}
    if not links:
        return res
    lo = min(ts[0] for ts, _ in links.values())
    hi = max(ts[-1] for ts, _ in links.values())
    for w in DECISION_WINDOWS_S:
        starts = np.arange(np.ceil(lo / w) * w, hi - w, w)
        lab = np.array([labels.get(int(s // 60) * 60, -1) for s in starts])
        starts, lab = starts[lab >= 0], lab[lab >= 0]
        base = int(round(BASE_RATE_HZ * w))
        keep_n = [int(round(BASE_RATE_HZ * f * w)) for f in RATE_FRACTIONS]
        F = np.full((len(RATE_FRACTIONS), len(lids), len(starts), len(FEATURE_NAMES)), np.nan, np.float32)
        native = np.zeros((len(lids), len(starts)), np.int32)
        for li, lid in enumerate(lids):
            ts, amp = links[lid]
            a, b = np.searchsorted(ts, starts), np.searchsorted(ts, starts + w)
            for wi in range(len(starts)):
                n = b[wi] - a[wi]
                native[li, wi] = n
                if n < base:
                    continue
                seg = amp[a[wi]:b[wi]]
                for r, k in enumerate(keep_n):
                    if k >= MIN_DECIMATED_FRAMES:
                        F[r, li, wi] = link_window_features(seg[decimated_indices(n, k)])
        res["windows"][w] = (starts, lab, lids, F, native)
    rates = {lid: len(ts) / max(ts[-1] - ts[0], 1.0) for lid, (ts, _) in links.items()}
    res["median_pair_rate_hz"] = float(np.median(list(rates.values())))
    log.info("%s: %d pairs, median %.2f soundings/s, %.0fs", res["file"], len(lids),
             res["median_pair_rate_hz"], time.time() - t0)
    del links
    gc.collect()
    return res


def hour_files():
    files = []
    for day in sorted((DATASET_DIR / "csi_data").iterdir()):
        if day.name in EXCLUDED_DAYS:
            continue
        for f in sorted(day.glob("csi_*.jsonl")):
            if int(f.name[4:6]) in FILE_HOURS:
                files.append(f)
    return files


def check_clock() -> None:
    act_dir = OUT_DIR / "activity"
    series = [np.load(f) for f in sorted(act_dir.glob("*.npz"))]
    if not series:
        sys.exit(f"no activity series in {act_dir}; run the extraction first")
    secs = np.concatenate([s["secs"] for s in series])
    act = np.concatenate([s["act"] for s in series])
    order = np.argsort(secs)
    secs, act = secs[order], act[order]
    opens = [utc(r["timestamp"]) for r in csv.DictReader(open(DOOR_EVENTS)) if r["action"] == "Door opened"]
    print("offset_s  events  peri/baseline (median over events)")
    for off in CLOCK_OFFSETS_S:
        ratios = []
        for t in opens:
            c = t + off
            peri = act[(secs >= c - PERI_EVENT_S) & (secs <= c + PERI_EVENT_S)]
            base = act[(secs >= c - BASELINE_S[0]) & (secs <= c - BASELINE_S[1])]
            if len(peri) > 10 and len(base) > 60:
                ratios.append(peri.mean() / base.mean())
        print(f"{off:+8.0f}  {len(ratios):6d}  {np.median(ratios) if ratios else float('nan'):.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="count labelled minutes and files, read no CSI")
    ap.add_argument("--check-clock", action="store_true", help="compare door-event activity at clock offsets")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    set_vm_limit()
    if args.check_clock:
        check_clock()
        return
    labels = minute_labels()
    files = hour_files()
    n_occ = sum(labels.values())
    print(f"{len(files)} hour files; labelled minutes: {n_occ} occupied, {len(labels) - n_occ} empty, "
          f"hours {LABEL_HOURS_UTC.start}-{LABEL_HOURS_UTC.stop - 1} UTC", flush=True)
    if args.dry_run:
        by_day = defaultdict(lambda: [0, 0])
        for m, v in labels.items():
            by_day[datetime.fromtimestamp(m, timezone.utc).strftime("%Y-%m-%d")][v] += 1
        for d, (e, o) in sorted(by_day.items()):
            print(f"  {d}: {o} occupied, {e} empty minutes")
        return
    (OUT_DIR / "activity").mkdir(parents=True, exist_ok=True)
    per_day = defaultdict(list)
    rates = []
    for res in safe_pool_map(process_hour, [(f, labels) for f in files], args.workers, EST_BYTES_PER_WORKER):
        np.savez_compressed(OUT_DIR / "activity" / (res["file"].replace("/", "_") + ".npz"),
                            secs=res["secs"], act=res["act"])
        if res["windows"]:
            per_day[res["file"][:10]].append(res["windows"])
            rates.append(res["median_pair_rate_hz"])
    for w in DECISION_WINDOWS_S:
        wdir = OUT_DIR / f"w{w:g}s"
        wdir.mkdir(parents=True, exist_ok=True)
        for day, hours in per_day.items():
            parts = [h[w] for h in hours if len(h[w][0])]
            if not parts:
                continue
            lids = sorted({l for p in parts for l in p[2]})
            idx = {l: i for i, l in enumerate(lids)}
            n_win = sum(len(p[0]) for p in parts)
            F = np.full((len(RATE_FRACTIONS), len(lids), n_win, len(FEATURE_NAMES)), np.nan, np.float32)
            native = np.zeros((len(lids), n_win), np.int32)
            starts, labs, o = [], [], 0
            for s, lab, plids, pF, pnat in parts:
                rows = [idx[l] for l in plids]
                F[:, rows, o:o + len(s)] = pF
                native[rows, o:o + len(s)] = pnat
                starts.append(s)
                labs.append(lab)
                o += len(s)
            np.savez_compressed(wdir / f"rate_{day}.npz", link_ids=np.array(lids), labels=np.concatenate(labs),
                                timestamps=np.concatenate(starts), native_frames=native, features=F,
                                rate_fractions=np.array(RATE_FRACTIONS), feature_names=np.array(FEATURE_NAMES),
                                base_rate_hz=BASE_RATE_HZ, window_s=w, session_id=day)
    summary = {"files": len(files), "days": sorted(per_day), "median_pair_rate_hz": float(np.median(rates)),
               "base_rate_hz": BASE_RATE_HZ, "rate_fractions": RATE_FRACTIONS, "clock_offset_s": CLOCK_OFFSET_S,
               "label_hours_utc": [LABEL_HOURS_UTC.start, LABEL_HOURS_UTC.stop - 1],
               "labelled_minutes": {"occupied": n_occ, "empty": len(labels) - n_occ}}
    (OUT_DIR / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
