#!/usr/bin/env python3
"""
Convert JSONL label files to legacy hour-level CSV format.

Reads labels_*.jsonl files from date directories and collapses them into
the hour-level CSV format used by the original evaluation pipeline, enabling
backward compatibility with existing analysis scripts.
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OCCUPIED_LABELS = {"occupied", "moving", "sedentary", "sleeping"}


def load_events(date_dirs: list[Path]) -> list[dict]:
    """Load all label events from the given date directories."""
    events = []
    for d in date_dirs:
        if not d.is_dir():
            logger.warning(f"Skipping non-directory: {d}")
            continue
        for f in sorted(d.glob("labels_*.jsonl")):
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return events


def collapse_to_hours(events: list[dict], env_id: str) -> pd.DataFrame:
    """Collapse events into hour-level rows."""
    # Group by (date, hour)
    slots: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for ev in events:
        ts = ev.get("ts")
        iso = ev.get("iso", "")
        if ts:
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
        elif iso:
            from datetime import datetime
            dt = datetime.fromisoformat(iso)
        else:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        hour = dt.hour
        slots[(date_str, hour)].append(ev)

    rows = []
    for (date_str, hour), evts in sorted(slots.items()):
        # Last-write-wins: sort by ts
        evts.sort(key=lambda e: e.get("ts", 0))
        labels = [e.get("label", "") for e in evts]
        non_empty = [l for l in labels if l]

        # Occupied if any label is in the occupied set
        lr_occupied = 1 if any(l in OCCUPIED_LABELS for l in labels) else 0

        # Most frequent non-empty label
        if non_empty:
            activity_type = Counter(non_empty).most_common(1)[0][0]
        else:
            activity_type = ""

        # Note from last event
        note = evts[-1].get("note", "") if evts else ""

        rows.append({
            "date": date_str,
            "hour": hour,
            "lr_occupied": lr_occupied,
            "activity_type": activity_type,
            "note": note,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL label files to legacy hour-level CSV"
    )
    parser.add_argument(
        "--input", nargs="+", type=Path, required=True,
        help="Date directories containing labels_*.jsonl files"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--env-id", type=str, default="ENV-A",
        help="Environment identifier (default: ENV-A)"
    )
    args = parser.parse_args()

    events = load_events(args.input)
    if not events:
        logger.error("No label events found")
        return

    logger.info(f"Loaded {len(events)} label events")

    df = collapse_to_hours(events, args.env_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(df)} hour rows to {args.output}")


if __name__ == "__main__":
    main()
