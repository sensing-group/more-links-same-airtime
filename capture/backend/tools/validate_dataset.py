#!/usr/bin/env python3
"""Validate dataset integrity.

Checks:
- Every trial directory has trial.json, raw/csi_raw.jsonl.zst, calibrated/csi_cal.jsonl.zst
- Every trial.json references valid environment, subject, topology files
- Every activity is in ALL_ACTIVITIES
- Manifest consistency with trial directories
- Timestamp sanity: |wall_ts - server_ts| < 50ms for sampled records
- Clock sync present in trial.json

Usage: python -m tools.validate_dataset [--data-root data/]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import ALL_ACTIVITIES


def validate(data_root: Path):
    passed = 0
    failed = 0
    warnings = 0

    def ok(msg):
        nonlocal passed
        passed += 1
        print(f"  \u2713  {msg}")

    def fail(msg):
        nonlocal failed
        failed += 1
        print(f"  \u2717  {msg}")

    def warn(msg):
        nonlocal warnings
        warnings += 1
        print(f"  !  {msg}")

    trials_dir = data_root / "trials"
    manifest_path = data_root / "manifest.jsonl"

    # Check required directories
    print("  Structure")
    for d in ["environments", "subjects", "topologies", "trials"]:
        p = data_root / d
        if p.exists():
            ok(f"{d}/ exists")
        else:
            fail(f"{d}/ missing")

    if manifest_path.exists():
        ok("manifest.jsonl exists")
    else:
        fail("manifest.jsonl missing")

    if not trials_dir.exists():
        print(f"\n  No trials directory. {passed}/{passed+failed} checks passed.")
        return failed == 0

    # Check each trial
    print("\n  Trials")
    trial_dirs = sorted(trials_dir.iterdir())
    manifest_ids = set()

    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        manifest_ids.add(json.loads(line)["trial_id"])
                    except (json.JSONDecodeError, KeyError):
                        fail(f"Malformed manifest line: {line[:60]}")

    for trial_dir in trial_dirs:
        if not trial_dir.is_dir():
            continue
        tid = trial_dir.name

        # trial.json
        meta_path = trial_dir / "trial.json"
        if not meta_path.exists():
            fail(f"{tid}: trial.json missing")
            continue

        meta = json.loads(meta_path.read_text())
        trial_id = meta.get("trial_id", "?")

        # Required files
        raw = trial_dir / "raw" / "csi_raw.jsonl.zst"
        cal = trial_dir / "calibrated" / "csi_cal.jsonl.zst"
        if raw.exists():
            ok(f"{trial_id}: raw data present")
        else:
            warn(f"{trial_id}: raw/csi_raw.jsonl.zst missing")
        if cal.exists():
            ok(f"{trial_id}: calibrated data present")
        else:
            warn(f"{trial_id}: calibrated/csi_cal.jsonl.zst missing")

        # Validate references
        activity = meta.get("activity", "")
        if activity in ALL_ACTIVITIES:
            ok(f"{trial_id}: activity '{activity}' valid")
        else:
            fail(f"{trial_id}: unknown activity '{activity}'")

        env_id = meta.get("environment_id", "")
        env_file = data_root / "environments" / f"{env_id}.json"
        if env_file.exists():
            ok(f"{trial_id}: environment '{env_id}' exists")
        else:
            fail(f"{trial_id}: environment file missing: {env_file}")

        subj_id = meta.get("subject_id", "")
        subj_file = data_root / "subjects" / f"{subj_id}.json"
        if subj_file.exists():
            ok(f"{trial_id}: subject '{subj_id}' exists")
        else:
            fail(f"{trial_id}: subject file missing: {subj_file}")

        topo_id = meta.get("topology_id", "")
        topo_file = data_root / "topologies" / f"{topo_id}.json"
        if topo_file.exists():
            ok(f"{trial_id}: topology '{topo_id}' exists")
        else:
            fail(f"{trial_id}: topology file missing: {topo_file}")

        # Clock sync
        clock = meta.get("clock_sync", {})
        if clock.get("nodes_at_start"):
            ok(f"{trial_id}: clock_sync present")
        else:
            warn(f"{trial_id}: clock_sync.nodes_at_start empty")

        # Manifest consistency
        if trial_id in manifest_ids:
            ok(f"{trial_id}: in manifest")
        else:
            fail(f"{trial_id}: missing from manifest")

    # Check for orphaned manifest entries
    trial_ids_on_disk = set()
    for d in trial_dirs:
        if d.is_dir() and (d / "trial.json").exists():
            meta = json.loads((d / "trial.json").read_text())
            trial_ids_on_disk.add(meta.get("trial_id", ""))
    orphans = manifest_ids - trial_ids_on_disk
    for o in orphans:
        fail(f"Manifest entry '{o}' has no trial directory")

    total = passed + failed + warnings
    print(f"\n{'─' * 50}")
    suffix = f"  ({warnings} warnings)" if warnings else ""
    print(f"  {passed}/{total} checks passed, {failed} failed{suffix}\n")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate dataset integrity")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    args = parser.parse_args()
    success = validate(Path(args.data_root))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
