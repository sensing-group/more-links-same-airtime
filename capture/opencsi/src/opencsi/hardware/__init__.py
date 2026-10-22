"""Hardware abstraction layer for OpenCSI."""

from opencsi.hardware.base import (
    CSIExtractor,
    CSIPacket,
)
from opencsi.hardware.esp32 import (
    ESP32Extractor,
    MAGIC_BYTES,
    HEADER_SIZE,
    ESP32_DEFAULT_NOISE_FLOOR,
)
from opencsi.hardware.feitcsi import FeitCSIExtractor
from opencsi.hardware.profiles import (
    HardwareProfileLoader,
    get_hardware_profile,
    DEFAULT_PROFILES_PATH,
)

__all__ = [
    # Abstract base
    "CSIExtractor",
    "CSIPacket",
    # ESP32 implementation
    "ESP32Extractor",
    "MAGIC_BYTES",
    "HEADER_SIZE",
    "ESP32_DEFAULT_NOISE_FLOOR",
    # FeitCSI implementation (Intel AX200/AX210/AX211)
    "FeitCSIExtractor",
    # Hardware profiles
    "HardwareProfileLoader",
    "get_hardware_profile",
    "DEFAULT_PROFILES_PATH",
]
