"""
Activity annotation store — JSONL within the trial directory.

Follows OPPORTUNITY dataset interval-label conventions:
  - Every label is an interval [start_ts, end_ts]
  - Point labels have end_ts == start_ts
  - merge_asof-compatible with CSI DataFrames
"""
import asyncio
import dataclasses
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schema import (
    ALL_ACTIVITIES,
    ACTIVITY_TAXONOMY,
    FINE_TO_COARSE,
    AnnotationCreate,
    AnnotationRecord,
)

logger = logging.getLogger(__name__)


class LabelStore:
    def __init__(self, base_dir: str = "data", recorder=None):
        self._base_dir = Path(base_dir)
        self._recorder = recorder
        self._lock = asyncio.Lock()

    def _annotations_file(self) -> Path:
        """Return the annotations file for the current trial, or a global fallback."""
        if self._recorder and self._recorder.trial_id:
            trial_dir = self._recorder.get_trial_dir()
            if trial_dir:
                f = trial_dir / "annotations.jsonl"
                f.parent.mkdir(parents=True, exist_ok=True)
                return f
        # Fallback: global annotations dir
        d = self._base_dir / "annotations"
        d.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return d / f"{date_str}.jsonl"

    async def add_annotation(self, req: AnnotationCreate) -> AnnotationRecord:
        if req.fine_label not in ALL_ACTIVITIES:
            raise ValueError(
                f"Unknown activity '{req.fine_label}'. Valid: {ALL_ACTIVITIES}"
            )

        now = time.time()
        start_ts = req.start_ts if req.start_ts is not None else now

        if req.end_ts is not None:
            end_ts = req.end_ts
        elif req.duration_s is not None:
            end_ts = start_ts + req.duration_s
        else:
            end_ts = start_ts

        duration_s = end_ts - start_ts

        rec = AnnotationRecord(
            annotation_id=uuid.uuid4().hex,
            trial_id=self._recorder.trial_id if self._recorder and self._recorder.trial_id else "",
            start_ts=start_ts,
            end_ts=end_ts,
            server_ts=now,
            duration_s=round(duration_s, 4),
            coarse_label=FINE_TO_COARSE[req.fine_label],
            fine_label=req.fine_label,
            annotator_id=req.annotator_id,
            confidence=req.confidence,
            note=req.note,
            source=req.source,
        )

        path = self._annotations_file()
        line = json.dumps(dataclasses.asdict(rec), separators=(",", ":"))
        async with self._lock:
            with open(path, "a") as f:
                f.write(line + "\n")
                f.flush()

        logger.info("Annotation: %s %s [%.1fs]", rec.fine_label, rec.annotation_id[:8], duration_s)
        return rec

    async def get_annotations(self, date_str: str = None, trial_id: str = None) -> list[dict]:
        # If a trial is active or specified, read from trial dir
        path = self._annotations_file()

        if date_str and not (self._recorder and self._recorder.trial_id):
            # Fallback mode: date-based file
            d = self._base_dir / "annotations"
            path = d / f"{date_str}.jsonl"

        if not path.exists():
            return []

        results = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if trial_id and rec.get("trial_id") != trial_id:
                        continue
                    results.append(rec)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed annotation line in %s", path)
                    continue

        results.sort(key=lambda r: r.get("server_ts", 0), reverse=True)
        return results

    async def delete_annotation(self, annotation_id: str) -> bool:
        path = self._annotations_file()
        if not path.exists():
            return False

        async with self._lock:
            lines = path.read_text().splitlines()
            kept = []
            found = False
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("annotation_id") == annotation_id:
                        found = True
                        continue
                    kept.append(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line during delete in %s", path)
                    kept.append(line)

            if not found:
                return False

            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    for line in kept:
                        f.write(line + "\n")
                os.replace(tmp_path, path)
            except Exception:
                os.unlink(tmp_path)
                raise

        return True

    async def get_taxonomy(self) -> dict:
        return ACTIVITY_TAXONOMY

    async def get_activities(self) -> list[str]:
        return ALL_ACTIVITIES
