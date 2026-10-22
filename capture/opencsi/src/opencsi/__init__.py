"""
OpenCSI - Self-calibrating WiFi CSI preprocessing library.

OpenCSI transforms raw, hardware-dependent WiFi Channel State Information (CSI)
into calibrated, hardware-agnostic Z-scores with reliability metadata.

Supports multiple hardware platforms:
    - ESP32 (WiFi 4/6)
    - Intel 5300
    - Nexmon (BCM43455, BCM4339)
    - Atheros (AR9380, QCA9558)

Quick Start:
    from opencsi import AutoCalibrator, get_extractor

    # Use the factory function for hardware-agnostic setup
    extractor = get_extractor("esp32")  # or "intel_5300", "nexmon", etc.

    calibrator = AutoCalibrator(
        node_positions={"AA:BB:CC:DD:EE:01": (0.0, 0.0), ...},
        hardware=extractor,
    )

    for raw_packet in udp_packets:
        packet = calibrator.hardware.parse(raw_packet, rx_mac)
        frame = calibrator.ingest([packet], timestamp)
        if frame and frame.motion_detected:
            print("Motion detected!")

    # Available extractors
    from opencsi import list_extractors
    print(list_extractors())  # ['esp32', 'intel_5300', 'nexmon', ...]

For more information, see: https://github.com/opencsi/opencsi

Submodule access (advanced users):
    Symbols removed from the top level are still importable from their
    submodules, e.g.::

        from opencsi.hardware.capabilities import FEATURE_MIMO
        from opencsi.stats.circular import circular_mean
        from opencsi.baseline.frequency_baseline import FrequencyAwareBaseline
"""

__version__ = "0.1.0"

# =============================================================================
# Core - Main calibration engine and types
# =============================================================================
from opencsi.core.calibrator import AutoCalibrator
from opencsi.core.config import AutoCalibratorConfig
from opencsi.core.types import (
    CalibratedFrame,
    CalibratedLink,
    RawCSIPacket,
    CalibrationSnapshot,
    QualityGrade,
)

# =============================================================================
# Hardware - Abstraction layer and registry
# =============================================================================
from opencsi.hardware.base import CSIExtractor, CSIPacket
from opencsi.hardware.registry import get_extractor, list_extractors

# =============================================================================
# Sanitization - Preprocessing level enum
# =============================================================================
from opencsi.sanitization.sanitizer import SanitizationLevel

# =============================================================================
# Streaming - Async calibration stream
# =============================================================================
from opencsi.stream.async_stream import CalibratedStream

# =============================================================================
# Configuration - Runtime config access
# =============================================================================
from opencsi.core.env import get_config

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    "__version__",
    # Core
    "AutoCalibrator",
    "AutoCalibratorConfig",
    "CalibratedFrame",
    "CalibratedLink",
    "RawCSIPacket",
    "CalibrationSnapshot",
    "QualityGrade",
    # Hardware
    "CSIExtractor",
    "CSIPacket",
    "get_extractor",
    "list_extractors",
    # Sanitization
    "SanitizationLevel",
    # Streaming
    "CalibratedStream",
    # Configuration
    "get_config",
]
