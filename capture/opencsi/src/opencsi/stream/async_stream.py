"""
Async streaming interface for OpenCSI.

Provides an AsyncIterator interface for processing calibrated CSI frames.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from opencsi.hardware.base import CSIPacket

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_MAXSIZE = 1000


class CalibratedStream:
    """
    Async iterator for calibrated CSI frames.

    Wraps an AutoCalibrator and a packet source to provide an async
    stream of calibrated frames.

    Usage:
        async with CalibratedStream(calibrator, packet_source) as stream:
            async for frame in stream:
                process(frame)
    """

    def __init__(
        self,
        calibrator,  # AutoCalibrator - not typed to avoid circular imports
        packet_queue: Optional[asyncio.Queue] = None,
    ):
        """
        Initialize the calibrated stream.

        Args:
            calibrator: AutoCalibrator instance
            packet_queue: Optional asyncio.Queue for receiving packets
        """
        self._calibrator = calibrator
        self._packet_queue = packet_queue or asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)
        self._running = False
        self._frames_produced = 0

    async def __aenter__(self):
        """Async context manager entry."""
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._running = False
        return False

    def __aiter__(self) -> AsyncIterator:
        """Return async iterator."""
        return self

    async def __anext__(self):
        """Get next calibrated frame."""

        while self._running:
            try:
                # Get packet from queue with timeout
                packet = await asyncio.wait_for(
                    self._packet_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                # No packet available, try again
                continue

            if packet is None:
                # None signals end of stream
                raise StopAsyncIteration

            # Process packet
            timestamp = time.time()
            frame = self._calibrator.ingest([packet], timestamp)

            if frame is None:
                # No frame produced, try next packet
                continue

            self._frames_produced += 1
            return frame

        raise StopAsyncIteration

    async def put_packet(self, packet: CSIPacket) -> None:
        """Add a packet to the stream queue."""
        await self._packet_queue.put(packet)

    def put_packet_nowait(self, packet: CSIPacket) -> None:
        """Add a packet to the stream queue (non-blocking)."""
        try:
            self._packet_queue.put_nowait(packet)
        except asyncio.QueueFull:
            try:
                self._packet_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._packet_queue.put_nowait(packet)

    async def stop(self) -> None:
        """Stop the stream by sending sentinel."""
        self._running = False
        await self._packet_queue.put(None)

    def get_stats(self) -> dict:
        """Get stream statistics."""
        return {
            'running': self._running,
            'frames_produced': self._frames_produced,
            'queue_size': self._packet_queue.qsize(),
        }
