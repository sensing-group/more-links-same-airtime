#!/usr/bin/env python3
"""Regenerate manifest.jsonl from all trial.json files.

Usage: python -m tools.rebuild_manifest [--data-root data/]
"""
import argparse
import json
from pathlib import Path


def rebuild(data_root: Path):
    trials_dir = data_root / "trials"
    if not trials_dir.exists():
        print(f"No trials directory at {trials_dir}")
        return

    entries = []
    for trial_json in sorted(trials_dir.glob("*/trial.json")):
        meta = json.loads(trial_json.read_text())
        entry = {
            "trial_id": meta["trial_id"],
            "activity": meta["activity"],
            "activities": meta.get("activities", [meta["activity"]]),
            "subject_id": meta["subject_id"],
            "environment_id": meta["environment_id"],
            "topology_id": meta["topology_id"],
            "repetition": meta.get("repetition", 1),
            "started_at": meta["started_at"],
            "duration_s": meta.get("duration_s"),
            "tags": meta.get("tags", []),
            "raw_packets": meta.get("collection", {}).get("raw_packets", 0),
            "cal_packets": meta.get("collection", {}).get("cal_packets", 0),
            "dropped": meta.get("collection", {}).get("dropped", 0),
            "path": str(trial_json.parent.relative_to(data_root)),
        }
        entries.append(entry)

    entries.sort(key=lambda e: e["trial_id"])

    manifest = data_root / "manifest.jsonl"
    with open(manifest, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    print(f"Wrote {len(entries)} entries to {manifest}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild manifest.jsonl")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    args = parser.parse_args()
    rebuild(Path(args.data_root))


if __name__ == "__main__":
    main()
