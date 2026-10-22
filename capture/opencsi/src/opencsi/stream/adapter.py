"""
Callback adapter for integrating OpenCSI with existing codebases.

Provides a simple interface for feeding CSI packets to the calibrator
and receiving calibrated frames via callbacks.
"""

import logging
from typing import Callable, Optional, List

from opencsi.hardware.base import CSIPacket

logger = logging.getLogger(__name__)


class CallbackAdapter:
    """
    Callback-based adapter for integrating AutoCalibrator with existing systems.

    This adapter wraps an AutoCalibrator instance and provides a simpler
    callback-based interface. Use this when integrating with UDP receivers
    or other packet sources.

    Usage:
        from opencsi import AutoCalibrator
        from opencsi.stream import CallbackAdapter

        calibrator = AutoCalibrator(node_positions=positions)

        def on_frame(frame):
            print(f"Got calibrated frame with {len(frame.links)} links")

        adapter = CallbackAdapter(calibrator, on_frame=on_frame)

        # In your packet receive loop:
        udp_receiver.on_packet(adapter.ingest)
    """

    def __init__(
        self,
        calibrator,  # AutoCalibrator - not typed to avoid circular imports
        on_frame: Optional[Callable] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize the callback adapter.

        Args:
            calibrator: AutoCalibrator instance
            on_frame: Callback for calibrated frames (CalibratedFrame)
            on_error: Callback for errors (receives Exception)
        """
        self._calibrator = calibrator
        self._on_frame = on_frame
        self._on_error = on_error
        self._packet_buffer: List[CSIPacket] = []
        self._packets_received = 0
        self._frames_emitted = 0

    def ingest(self, packet: CSIPacket, timestamp: Optional[float] = None) -> None:
        """
        Ingest a single CSI packet.

        This is the main entry point for feeding packets to the calibrator.
        When enough packets are collected, a calibrated frame is generated
        and the on_frame callback is invoked.

        Args:
            packet: CSIPacket from hardware extractor
            timestamp: Optional timestamp override (uses current time if None)
        """
        try:
            self._packets_received += 1
            self._packet_buffer.append(packet)

            # For now, process immediately on each packet
            # A more sophisticated implementation could batch packets
            import time
            ts = timestamp or time.time()

            frame = self._calibrator.ingest(self._packet_buffer, ts)
            self._packet_buffer.clear()

            if frame is not None and self._on_frame:
                self._frames_emitted += 1
                self._on_frame(frame)

        except Exception as e:
            if self._on_error:
                self._on_error(e)
            else:
                logger.exception(f"Error processing packet: {e}")

    def ingest_batch(
        self,
        packets: List[CSIPacket],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Ingest a batch of CSI packets at once.

        Args:
            packets: List of CSIPacket from hardware extractor
            timestamp: Optional timestamp override
        """
        try:
            self._packets_received += len(packets)

            import time
            ts = timestamp or time.time()

            frame = self._calibrator.ingest(packets, ts)

            if frame is not None and self._on_frame:
                self._frames_emitted += 1
                self._on_frame(frame)

        except Exception as e:
            if self._on_error:
                self._on_error(e)
            else:
                logger.exception(f"Error processing batch: {e}")

    def set_on_frame(self, callback: Callable) -> None:
        """Set the on_frame callback."""
        self._on_frame = callback

    def set_on_error(self, callback: Callable[[Exception], None]) -> None:
        """Set the on_error callback."""
        self._on_error = callback

    def get_stats(self) -> dict:
        """Get adapter statistics."""
        return {
            'packets_received': self._packets_received,
            'frames_emitted': self._frames_emitted,
            'pending_packets': len(self._packet_buffer),
        }

    def reset(self) -> None:
        """Reset adapter state."""
        self._packet_buffer.clear()
        self._packets_received = 0
        self._frames_emitted = 0
