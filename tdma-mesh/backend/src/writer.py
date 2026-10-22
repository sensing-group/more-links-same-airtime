#!/usr/bin/env python3
"""
CSI Data Writer - Persists CSI packets to disk as JSONL files.

Buffers packets in memory and flushes to date-partitioned JSONL files
with configurable rotation and optional gzip compression.
"""

import asyncio
import collections
import gzip
import json
import logging
from datetime import datetime
from pathlib import Path

from models import CSIPacket

logger = logging.getLogger(__name__)


class CSIWriter:
    """
    Async CSI packet writer with in-memory buffering and periodic flush.

    Packets are appended via write() (sync, never blocks) and flushed
    to disk every 2 seconds by a background task.
    """

    def __init__(self, data_dir: str, buffer_size: int,
                 rotation_minutes: int, compress: bool = False):
        self._data_dir = Path(data_dir)
        self._buffer_size = buffer_size
        self._rotation_minutes = rotation_minutes
        self._compress = compress

        self._buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._flush_task: asyncio.Task | None = None
        self._current_path: Path | None = None
        self._file_handle = None

        # Stats
        self.packets_written = 0
        self.packets_dropped = 0
        self.bytes_written = 0
        self.files_created = 0
        self.last_flush_time: datetime | None = None

    async def start(self) -> None:
        """Start the background flush task."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"CSI Writer started (dir={self._data_dir}, "
                    f"compress={self._compress})")

    async def stop(self) -> None:
        """Stop the flush task and do a final flush."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush of remaining packets
        self._do_flush()
        self._close_file()
        logger.info(f"CSI Writer stopped ({self.packets_written} packets written)")

    def write(self, packet: CSIPacket) -> None:
        """Append a packet to the buffer. Never blocks, never raises."""
        try:
            if len(self._buffer) >= self._buffer_size:
                self.packets_dropped += 1
                return
            self._buffer.append(packet)
        except Exception:
            self.packets_dropped += 1
            logger.debug("Failed to buffer CSI packet", exc_info=True)

    def get_stats(self) -> dict:
        """Return writer statistics."""
        return {
            "packets_written": self.packets_written,
            "packets_dropped": self.packets_dropped,
            "bytes_written": self.bytes_written,
            "files_created": self.files_created,
            "last_flush_time": (self.last_flush_time.isoformat()
                                if self.last_flush_time else None),
            "buffer_depth": len(self._buffer),
        }

    async def _flush_loop(self) -> None:
        """Background loop that flushes buffered packets every 2 seconds."""
        while True:
            try:
                await asyncio.sleep(0.5)
                self._do_flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("CSI flush error", exc_info=True)

    def _do_flush(self) -> None:
        """Drain the buffer and write packets to disk."""
        if not self._buffer:
            return

        # Atomically swap the buffer
        packets = list(self._buffer)
        self._buffer.clear()

        now = datetime.now()
        date_dir = self._data_dir / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Compute the rotation slot: floor to the nearest rotation interval
        slot_minute = (now.hour * 60 + now.minute) // self._rotation_minutes * self._rotation_minutes
        slot_h, slot_m = divmod(slot_minute, 60)
        ext = ".jsonl.gz" if self._compress else ".jsonl"
        filename = f"csi_{slot_h:02d}{slot_m:02d}00{ext}"
        target_path = date_dir / filename

        # Rotate only when the slot changes (i.e. a new time window)
        if self._current_path is not None and self._current_path != target_path:
            self._close_file()

        # Open a new file if needed
        if self._file_handle is None:
            self._open_file(target_path)

        try:
            for packet in packets:
                line = json.dumps(packet.to_dict(), ensure_ascii=False) + "\n"
                encoded = line.encode("utf-8")
                self._file_handle.write(encoded if self._compress else line)
                self.bytes_written += len(encoded)
                self.packets_written += 1
            self._file_handle.flush()
        except Exception:
            logger.warning("CSI write error", exc_info=True)

        self.last_flush_time = now

    def _open_file(self, path: Path) -> None:
        """Open a new output file."""
        try:
            if self._compress:
                self._file_handle = gzip.open(path, "ab")
            else:
                self._file_handle = open(path, "a", encoding="utf-8")
            self._current_path = path
            self.files_created += 1
            logger.info(f"Opened CSI file: {path}")
        except Exception:
            logger.warning(f"Failed to open {path}", exc_info=True)

    def _close_file(self) -> None:
        """Close the current file handle if open."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                logger.warning("Error closing CSI file", exc_info=True)
            finally:
                self._file_handle = None
                self._current_path = None
