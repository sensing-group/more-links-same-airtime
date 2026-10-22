"""
Append-only CSI data recorder with async queue flushing.

Trial-centric storage with hybrid anchored timestamps (schema v2.0).
Writes RawCSIRecord and CalibratedRecord to JSONL files, optionally zstd-compressed.
All disk I/O happens in a background flush loop — the UDP receive path never blocks.
"""
import asyncio
import dataclasses
import json
import logging
import os
import platform
import re
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from schema import (
    SCHEMA_VERSION, ALL_ACTIVITIES, RawCSIRecord, CalibratedRecord,
    TrialStartRequest,
)

logger = logging.getLogger(__name__)

try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False


def _serialize_record(record) -> bytes:
    """Serialize a frozen dataclass to a JSON line (bytes, no trailing newline)."""
    d = dataclasses.asdict(record)
    if _USE_ORJSON:
        return orjson.dumps(d)
    return json.dumps(d, separators=(",", ":")).encode("utf-8")


def next_trial_id(data_root: Path) -> str:
    """Generate the next trial ID (T0001, T0002, ...) from manifest and existing dirs."""
    manifest = data_root / "manifest.jsonl"
    num = 1
    if manifest.exists() and manifest.stat().st_size > 0:
        # Read last line efficiently
        with open(manifest, "rb") as f:
            f.seek(0, 2)
            pos = f.tell() - 2
            while pos > 0:
                f.seek(pos)
                if f.read(1) == b"\n":
                    break
                pos -= 1
            last_line = f.readline().decode().strip()
        if last_line:
            try:
                last_id = json.loads(last_line)["trial_id"]
                num = int(last_id[1:]) + 1
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    # Also skip any trial directories that exist but aren't in the manifest
    # (e.g. recordings that started but never cleanly stopped)
    trials_dir = data_root / "trials"
    if trials_dir.exists():
        existing = {p.name for p in trials_dir.iterdir() if p.is_dir()}
        while True:
            candidate = f"T{num:04d}"
            if not any(name == candidate or name.startswith(candidate + "_") for name in existing):
                break
            num += 1

    return f"T{num:04d}"


def _count_repetitions(data_root: Path, activity: str, subject_id: str, environment_id: str) -> int:
    """Count existing trials with the same (activity, subject, environment) combo."""
    manifest = data_root / "manifest.jsonl"
    if not manifest.exists():
        return 0
    count = 0
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if (entry.get("activity") == activity and
                    entry.get("subject_id") == subject_id and
                    entry.get("environment_id") == environment_id):
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


class _StreamWriter:
    """Manages a single JSONL file with streaming writes. Optionally zstd-compressed."""

    def __init__(self, path: Path, compress: bool = True):
        self._path = path
        self._compress = compress
        self._file = None
        self._zstd_writer = None

    @property
    def current_path(self) -> str:
        return str(self._path)

    def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._path, "ab")
        if self._compress:
            try:
                import zstandard as zstd
                cctx = zstd.ZstdCompressor(level=3)
                self._zstd_writer = cctx.stream_writer(fh, closefd=True)
                self._file = self._zstd_writer
            except Exception:
                fh.close()
                raise
        else:
            self._file = fh
        logger.info("Opened %s (compress=%s)", self._path, self._compress)

    def write_line(self, data: bytes):
        if self._file is None:
            return
        self._file.write(data)
        self._file.write(b"\n")

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._zstd_writer:
            self._zstd_writer.close()
            self._zstd_writer = None
        elif self._file:
            self._file.close()
        self._file = None


class CSIRecorder:
    def __init__(self, base_dir: str = "data", registry=None,
                 flush_interval_s: float = 2.0, buffer_size: int = 1000):
        self._base_dir = Path(base_dir)
        self._registry = registry
        self._flush_interval_s = flush_interval_s
        self._buffer_size = buffer_size

        self._trial_id: Optional[str] = None
        self._trial_dir: Optional[Path] = None
        self._trial_request: Optional[TrialStartRequest] = None
        self._raw_buf: asyncio.Queue = asyncio.Queue()
        self._cal_buf: asyncio.Queue = asyncio.Queue()
        self._recording = False
        self._stats = {"raw_written": 0, "cal_written": 0, "dropped": 0}
        self._flush_task: Optional[asyncio.Task] = None
        self._auto_stop_task: Optional[asyncio.Task] = None
        self._started_ts: float = 0.0
        self._scheduled_stop_ts: Optional[float] = None

        self._raw_writer: Optional[_StreamWriter] = None
        self._cal_writer: Optional[_StreamWriter] = None

        # Persistent compression preference. Applies to the next trial opened —
        # toggling mid-recording does not affect the currently-open writers.
        self._compress: bool = True

        # LRU cache: (tx, rx, seq) → record_id
        self._id_cache: OrderedDict = OrderedDict()
        self._id_cache_max = 2000

    def set_compress(self, enabled: bool) -> None:
        """Set compression preference for the next trial."""
        self._compress = bool(enabled)
        logger.info("Compression preference set to %s (applies on next trial)", self._compress)

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def trial_id(self) -> Optional[str]:
        return self._trial_id

    async def start(self):
        """Start the background flush loop (does not start recording)."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        """Stop recording and flush remaining data."""
        await self.stop_recording()
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

    async def start_recording(self, req: TrialStartRequest):
        """Begin a new trial recording."""
        if self._recording:
            return

        # Validate
        if req.activity not in ALL_ACTIVITIES:
            raise ValueError(f"Unknown activity '{req.activity}'. Valid: {ALL_ACTIVITIES}")
        if not re.match(r"^S\d{2}$", req.subject_id):
            raise ValueError(f"subject_id must match S00-S99, got '{req.subject_id}'")

        env_file = self._base_dir / "environments" / f"{req.environment_id}.json"
        if not env_file.exists():
            raise ValueError(f"Environment file not found: {env_file}")
        topo_file = self._base_dir / "topologies" / f"{req.topology_id}.json"
        if not topo_file.exists():
            raise ValueError(f"Topology file not found: {topo_file}")

        # Ensure data dirs exist
        (self._base_dir / "trials").mkdir(parents=True, exist_ok=True)
        (self._base_dir / "manifest.jsonl").touch()

        # Generate trial ID and directory
        self._trial_id = next_trial_id(self._base_dir)
        self._trial_request = req
        repetition = _count_repetitions(
            self._base_dir, req.activity, req.subject_id, req.environment_id
        ) + 1

        dir_name = f"{self._trial_id}_{req.environment_id}"
        self._trial_dir = self._base_dir / "trials" / dir_name
        (self._trial_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self._trial_dir / "calibrated").mkdir(parents=True, exist_ok=True)

        self._started_ts = time.time()
        now = datetime.now(timezone.utc)

        # Snapshot clock offsets at start
        clock_start = {}
        if self._registry:
            clock_start = self._registry.get_clock_offsets()

        meta = {
            "trial_id": self._trial_id,
            "activity": req.activity,
            "activities": [req.activity],
            "subject_id": req.subject_id,
            "environment_id": req.environment_id,
            "topology_id": req.topology_id,
            "repetition": repetition,
            "tags": req.tags,
            "started_at": now.isoformat(),
            "stopped_at": None,
            "duration_s": None,
            "clock_sync": {
                "method": "ema_anchored",
                "ema_alpha": 0.05,
                "nodes_at_start": clock_start,
                "nodes_at_stop": None,
            },
            "collection": {
                "hostname": platform.node(),
                "schema_version": SCHEMA_VERSION,
                "compress": self._compress,
                "raw_packets": 0,
                "cal_packets": 0,
                "dropped": 0,
            },
            "notes": req.notes,
        }
        (self._trial_dir / "trial.json").write_text(
            json.dumps(meta, indent=2)
        )

        self._stats = {"raw_written": 0, "cal_written": 0, "dropped": 0}

        ext = "jsonl.zst" if self._compress else "jsonl"
        self._raw_writer = _StreamWriter(
            self._trial_dir / "raw" / f"csi_raw.{ext}", compress=self._compress,
        )
        self._raw_writer.open()
        self._cal_writer = _StreamWriter(
            self._trial_dir / "calibrated" / f"csi_cal.{ext}", compress=self._compress,
        )
        self._cal_writer.open()

        self._recording = True
        logger.info("Recording started: trial=%s dir=%s", self._trial_id, dir_name)

        # Schedule auto-stop if a duration was requested
        self._scheduled_stop_ts = None
        if req.duration_s is not None and req.duration_s > 0:
            self._scheduled_stop_ts = self._started_ts + req.duration_s
            self._auto_stop_task = asyncio.create_task(
                self._auto_stop_after(req.duration_s)
            )
            logger.info(
                "Auto-stop scheduled in %.1fs (at %.0f)",
                req.duration_s, self._scheduled_stop_ts,
            )

    async def _auto_stop_after(self, delay_s: float):
        """Sleep, then stop the current recording if still running."""
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        if self._recording:
            logger.info("Auto-stop firing after %.1fs", delay_s)
            try:
                await self.stop_recording()
            except Exception:
                logger.exception("Auto-stop failed")

    async def stop_recording(self):
        """Stop recording, flush, finalize trial.json, append manifest."""
        if not self._recording:
            return
        self._recording = False

        # Cancel pending auto-stop (no-op if it's the one calling us)
        if self._auto_stop_task is not None:
            task = self._auto_stop_task
            self._auto_stop_task = None
            if task is not asyncio.current_task() and not task.done():
                task.cancel()
        self._scheduled_stop_ts = None

        await self._drain_and_write()

        if self._raw_writer:
            self._raw_writer.close()
            self._raw_writer = None
        if self._cal_writer:
            self._cal_writer.close()
            self._cal_writer = None

        if self._trial_dir:
            meta_path = self._trial_dir / "trial.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                now = datetime.now(timezone.utc)
                stopped_ts = time.time()
                meta["stopped_at"] = now.isoformat()
                meta["duration_s"] = round(stopped_ts - self._started_ts, 2)
                meta["collection"]["raw_packets"] = self._stats["raw_written"]
                meta["collection"]["cal_packets"] = self._stats["cal_written"]
                meta["collection"]["dropped"] = self._stats["dropped"]

                # Snapshot clock offsets at stop
                if self._registry:
                    meta["clock_sync"]["nodes_at_stop"] = self._registry.get_clock_offsets()

                meta_path.write_text(json.dumps(meta, indent=2))

                # Append to manifest
                manifest_entry = {
                    "trial_id": meta["trial_id"],
                    "activity": meta["activity"],
                    "activities": meta["activities"],
                    "subject_id": meta["subject_id"],
                    "environment_id": meta["environment_id"],
                    "topology_id": meta["topology_id"],
                    "repetition": meta["repetition"],
                    "started_at": meta["started_at"],
                    "duration_s": meta["duration_s"],
                    "tags": meta["tags"],
                    "raw_packets": self._stats["raw_written"],
                    "cal_packets": self._stats["cal_written"],
                    "dropped": self._stats["dropped"],
                    "path": str(self._trial_dir.relative_to(self._base_dir)),
                }
                manifest_path = self._base_dir / "manifest.jsonl"
                with open(manifest_path, "a") as f:
                    f.write(json.dumps(manifest_entry, separators=(",", ":")) + "\n")

        logger.info("Recording stopped: trial=%s raw=%d cal=%d dropped=%d",
                     self._trial_id, self._stats["raw_written"],
                     self._stats["cal_written"], self._stats["dropped"])

    def record_raw(self, sample, wall_ts: float, server_ts: float) -> None:
        """Enqueue a RawCSIRecord from a CSISample. Non-blocking."""
        if not self._recording:
            return
        if self._raw_buf.qsize() > 5000:
            self._stats["dropped"] += 1
            return

        record_id = uuid.uuid4().hex
        amp = sample.amplitude
        amp_mean = float(np.mean(amp))
        amp_std = float(np.std(amp))
        amp_var = float(np.var(amp))
        n = sample.num_carriers
        bw = "HT40" if n > 56 else "HT20"

        rec = RawCSIRecord(
            record_id=record_id,
            trial_id=self._trial_id,
            seq=sample.seq,
            wall_ts=wall_ts,
            server_ts=server_ts,
            firmware_ts_us=sample.timestamp_us,
            tx_mac=sample.tx_mac,
            rx_mac=sample.rx_mac,
            rssi=sample.rssi,
            noise_floor=sample.noise_floor,
            snr=float(sample.rssi - sample.noise_floor),
            num_subcarriers=n,
            bandwidth=bw,
            iq_data=sample.iq_data.astype(np.int8).tolist(),
            amplitude_mean=round(amp_mean, 6),
            amplitude_std=round(amp_std, 6),
            amplitude_var=round(amp_var, 6),
        )

        cache_key = (sample.tx_mac, sample.rx_mac, sample.seq)
        self._id_cache[cache_key] = (record_id, wall_ts)
        if len(self._id_cache) > self._id_cache_max:
            self._id_cache.popitem(last=False)

        self._raw_buf.put_nowait(rec)

    def record_calibrated(self, packet_record: dict, wall_ts: float) -> None:
        """Enqueue a CalibratedRecord from the detector's packet_info dict."""
        if not self._recording:
            return

        tx = packet_record["tx_mac"]
        rx = packet_record["rx_mac"]
        seq = packet_record["seq"]
        cache_key = (tx, rx, seq)
        cached = self._id_cache.get(cache_key)
        if cached:
            record_id, raw_wall_ts = cached
        else:
            record_id = uuid.uuid4().hex
            raw_wall_ts = packet_record.get("wall_ts", wall_ts)

        rssi = packet_record.get("rssi", 0)
        noise = packet_record.get("noise_floor", 0)

        rec = CalibratedRecord(
            record_id=record_id,
            trial_id=self._trial_id,
            seq=seq,
            wall_ts=raw_wall_ts,
            server_ts=wall_ts,
            tx_mac=tx,
            rx_mac=rx,
            amplitude_var=packet_record.get("amplitude_variance", 0.0),
            baseline_mean=packet_record.get("baseline_mean", 0.0),
            baseline_std=packet_record.get("baseline_std", 0.0),
            baseline_samples=packet_record.get("baseline_samples", 0),
            z_score=packet_record.get("z_score"),
            state=packet_record.get("state", "calibrating"),
            rssi=rssi,
            noise_floor=noise,
            snr=float(rssi - noise),
        )
        self._cal_buf.put_nowait(rec)

    def get_trial_dir(self) -> Optional[Path]:
        return self._trial_dir

    def get_stats(self) -> dict:
        return {
            "recording": self._recording,
            "trial_id": self._trial_id or "",
            "raw_written": self._stats["raw_written"],
            "cal_written": self._stats["cal_written"],
            "dropped": self._stats["dropped"],
            "queue_depth_raw": self._raw_buf.qsize(),
            "queue_depth_cal": self._cal_buf.qsize(),
            "current_raw_file": self._raw_writer.current_path if self._raw_writer else "",
            "current_cal_file": self._cal_writer.current_path if self._cal_writer else "",
            "trial_dir": str(self._trial_dir) if self._trial_dir else "",
            "compress": self._compress,
            "scheduled_stop_ts": self._scheduled_stop_ts,
            "elapsed_s": (time.time() - self._started_ts) if self._recording else 0.0,
            "remaining_s": (
                max(0.0, self._scheduled_stop_ts - time.time())
                if (self._recording and self._scheduled_stop_ts is not None)
                else None
            ),
        }

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(self._flush_interval_s)
            if self._recording:
                try:
                    await self._drain_and_write()
                except Exception:
                    logger.exception("Flush failed")

    async def _drain_and_write(self):
        raw_batch = []
        while not self._raw_buf.empty():
            try:
                raw_batch.append(self._raw_buf.get_nowait())
            except asyncio.QueueEmpty:
                break
        if raw_batch and self._raw_writer:
            for rec in raw_batch:
                self._raw_writer.write_line(_serialize_record(rec))
            self._raw_writer.flush()
            self._stats["raw_written"] += len(raw_batch)

        cal_batch = []
        while not self._cal_buf.empty():
            try:
                cal_batch.append(self._cal_buf.get_nowait())
            except asyncio.QueueEmpty:
                break
        if cal_batch and self._cal_writer:
            for rec in cal_batch:
                self._cal_writer.write_line(_serialize_record(rec))
            self._cal_writer.flush()
            self._stats["cal_written"] += len(cal_batch)
