#!/usr/bin/env python3
"""Recompute wall_ts for all records in a trial using stored clock_sync offsets.

Usage: python -m tools.recompute_wall_ts --trial data/trials/T0001_walking_S01_lab-alpha/
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    print("zstandard required: pip install zstandard")
    sys.exit(1)


def recompute(trial_dir: Path):
    meta_path = trial_dir / "trial.json"
    if not meta_path.exists():
        print(f"trial.json not found in {trial_dir}")
        return False

    meta = json.loads(meta_path.read_text())
    clock_sync = meta.get("clock_sync", {})
    nodes_at_start = clock_sync.get("nodes_at_start", {})

    if not nodes_at_start:
        print("No clock_sync.nodes_at_start in trial.json — cannot recompute")
        return False

    # Build offset lookup: mac → offset
    offsets = {}
    for mac, info in nodes_at_start.items():
        offsets[mac] = info["offset"]

    raw_path = trial_dir / "raw" / "csi_raw.jsonl.zst"
    if not raw_path.exists():
        print(f"Raw file not found: {raw_path}")
        return False

    print(f"Recomputing wall_ts for {trial_dir.name}")
    print(f"  Offsets: {len(offsets)} nodes")

    # Read, recompute, write
    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=3)

    records = []
    fixed = 0
    with open(raw_path, "rb") as f:
        reader = dctx.stream_reader(f)
        text = reader.read().decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tx_mac = rec.get("tx_mac", "")
            firmware_ts_us = rec.get("firmware_ts_us", 0)
            offset = offsets.get(tx_mac)

            if offset is not None and firmware_ts_us > 0:
                new_wall_ts = offset + (firmware_ts_us / 1e6)
                if abs(new_wall_ts - rec.get("wall_ts", 0)) > 0.001:
                    fixed += 1
                rec["wall_ts"] = new_wall_ts

            records.append(rec)

    # Write back
    tmp_path = raw_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        writer = cctx.stream_writer(f)
        for rec in records:
            writer.write(json.dumps(rec, separators=(",", ":")).encode("utf-8"))
            writer.write(b"\n")
        writer.close()

    tmp_path.replace(raw_path)
    print(f"  {len(records)} records processed, {fixed} wall_ts values corrected")
    return True


def main():
    parser = argparse.ArgumentParser(description="Recompute wall_ts from clock offsets")
    parser.add_argument("--trial", required=True, help="Trial directory path")
    args = parser.parse_args()
    success = recompute(Path(args.trial))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
