"""
Intel iwlwifi CSI Extractor for WiFi 6/6E/7 Cards.

Supports modern Intel WiFi cards that use the iwlwifi driver:
    - Intel AX200 (WiFi 6)
    - Intel AX201 (WiFi 6)
    - Intel AX210 (WiFi 6E)
    - Intel AX211 (WiFi 6E)
    - Intel BE200 (WiFi 7) - Future support
    - Intel BE202 (WiFi 7) - Future support

CSI extraction on these cards is possible through:
    1. PicoScenes platform (https://ps.zpj.io/)
    2. IAX open-source tool (IEEE 2024)
    3. FeitCSI tool
    4. Direct iwlwifi vendor commands (IWL_MVM_VENDOR_CMD_CSI_EVENT)

This module provides a unified interface for parsing CSI data from
any of these sources.

References:
    - PicoScenes: https://ps.zpj.io/
    - IAX Paper: https://ieeexplore.ieee.org/document/10944229/
    - iwlwifi driver: https://wireless.wiki.kernel.org/en/users/drivers/iwlwifi

Usage:
    from opencsi.hardware.iwlwifi import IwlWifiExtractor

    # For AX210
    extractor = IwlWifiExtractor(device="ax210")

    # For BE200 (future)
    extractor = IwlWifiExtractor(device="be200")

Note:
    This is a stub implementation. Full support requires integration
    with one of the CSI extraction tools mentioned above.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Tuple

from opencsi.hardware.base import CSIExtractor, CSIPacket
from opencsi.hardware.capabilities import (
    FEATURE_PHASE_CALIBRATION,
    FEATURE_WIFI7,
    HardwareCapabilities,
    SubcarrierConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Device identifiers
DEVICE_AX200 = "ax200"
DEVICE_AX201 = "ax201"
DEVICE_AX210 = "ax210"
DEVICE_AX211 = "ax211"
DEVICE_BE200 = "be200"
DEVICE_BE202 = "be202"

SUPPORTED_DEVICES = {
    DEVICE_AX200,
    DEVICE_AX201,
    DEVICE_AX210,
    DEVICE_AX211,
    DEVICE_BE200,
    DEVICE_BE202,
}

# WiFi 6 devices (802.11ax)
WIFI6_DEVICES = {DEVICE_AX200, DEVICE_AX201}

# WiFi 6E devices (802.11ax with 6GHz)
WIFI6E_DEVICES = {DEVICE_AX210, DEVICE_AX211}

# WiFi 7 devices (802.11be)
WIFI7_DEVICES = {DEVICE_BE200, DEVICE_BE202}

# Subcarrier counts per bandwidth (802.11ax HE format)
# These are the data + pilot subcarriers for HE frames
HE_SUBCARRIERS_20MHZ = 242
HE_SUBCARRIERS_40MHZ = 484
HE_SUBCARRIERS_80MHZ = 996
HE_SUBCARRIERS_160MHZ = 1992

# Subcarrier counts for WiFi 7 (802.11be EHT format)
# 320 MHz is new in WiFi 7
EHT_SUBCARRIERS_320MHZ = 3984

# Pilot indices for HE (802.11ax) - approximate, varies by RU allocation
HE_PILOT_INDICES_20MHZ = (24, 92, 150, 218)
HE_PILOT_INDICES_40MHZ = (24, 92, 150, 218, 266, 334, 392, 460)
HE_PILOT_INDICES_80MHZ = (
    24, 92, 150, 218, 266, 334, 392, 460,
    536, 604, 662, 730, 778, 846, 904, 972
)
HE_PILOT_INDICES_160MHZ = (
    24, 92, 150, 218, 266, 334, 392, 460,
    536, 604, 662, 730, 778, 846, 904, 972,
    1020, 1088, 1146, 1214, 1262, 1330, 1388, 1456,
    1532, 1600, 1658, 1726, 1774, 1842, 1900, 1968
)

# CSI data source formats
class CSISource(Enum):
    """Source format for CSI data."""
    PICOSCENES = "picoscenes"
    IAX = "iax"
    FEITCSI = "feitcsi"
    IWLWIFI_RAW = "iwlwifi_raw"
    UNKNOWN = "unknown"


# =============================================================================
# Subcarrier Configurations
# =============================================================================

def _create_ax_subcarrier_configs() -> Dict[int, SubcarrierConfig]:
    """Create subcarrier configs for WiFi 6/6E (802.11ax)."""
    return {
        20: SubcarrierConfig(
            num_subcarriers=HE_SUBCARRIERS_20MHZ,
            pilot_indices=HE_PILOT_INDICES_20MHZ,
            dc_index=HE_SUBCARRIERS_20MHZ // 2,
        ),
        40: SubcarrierConfig(
            num_subcarriers=HE_SUBCARRIERS_40MHZ,
            pilot_indices=HE_PILOT_INDICES_40MHZ,
            dc_index=HE_SUBCARRIERS_40MHZ // 2,
        ),
        80: SubcarrierConfig(
            num_subcarriers=HE_SUBCARRIERS_80MHZ,
            pilot_indices=HE_PILOT_INDICES_80MHZ,
            dc_index=HE_SUBCARRIERS_80MHZ // 2,
        ),
        160: SubcarrierConfig(
            num_subcarriers=HE_SUBCARRIERS_160MHZ,
            pilot_indices=HE_PILOT_INDICES_160MHZ,
            dc_index=HE_SUBCARRIERS_160MHZ // 2,
        ),
    }


def _create_be_subcarrier_configs() -> Dict[int, SubcarrierConfig]:
    """Create subcarrier configs for WiFi 7 (802.11be)."""
    # WiFi 7 includes all WiFi 6 bandwidths plus 320 MHz
    configs = _create_ax_subcarrier_configs()

    # Add 320 MHz for WiFi 7
    # Pilot indices for EHT 320 MHz (placeholder - needs verification)
    eht_pilot_320 = tuple(range(24, EHT_SUBCARRIERS_320MHZ, 62))

    configs[320] = SubcarrierConfig(
        num_subcarriers=EHT_SUBCARRIERS_320MHZ,
        pilot_indices=eht_pilot_320,
        dc_index=EHT_SUBCARRIERS_320MHZ // 2,
    )

    return configs


# =============================================================================
# Hardware Capabilities
# =============================================================================

def get_iwlwifi_capabilities(device: str) -> HardwareCapabilities:
    """
    Get hardware capabilities for an Intel iwlwifi device.

    Args:
        device: Device identifier (ax200, ax210, be200, etc.)

    Returns:
        HardwareCapabilities for the specified device

    Raises:
        ValueError: If device is not supported
    """
    device = device.lower()

    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unknown iwlwifi device: {device}. "
            f"Supported: {sorted(SUPPORTED_DEVICES)}"
        )

    if device in WIFI6_DEVICES:
        # AX200, AX201 - WiFi 6 without 6GHz
        caps = HardwareCapabilities(
            hardware_type=f"iwlwifi_{device}",
            supported_bandwidths=[20, 40, 80, 160],
            supported_bands=["2.4GHz", "5GHz"],
            max_tx_antennas=2,
            max_rx_antennas=2,
            subcarrier_configs=_create_ax_subcarrier_configs(),
            supports_mimo=True,
            has_reliable_agc=True,
        )

    elif device in WIFI6E_DEVICES:
        # AX210, AX211 - WiFi 6E with 6GHz
        caps = HardwareCapabilities(
            hardware_type=f"iwlwifi_{device}",
            supported_bandwidths=[20, 40, 80, 160],
            supported_bands=["2.4GHz", "5GHz", "6GHz"],
            max_tx_antennas=2,
            max_rx_antennas=2,
            subcarrier_configs=_create_ax_subcarrier_configs(),
            supports_mimo=True,
            has_reliable_agc=True,
        )

    elif device in WIFI7_DEVICES:
        # BE200, BE202 - WiFi 7 with 320MHz
        caps = HardwareCapabilities(
            hardware_type=f"iwlwifi_{device}",
            supported_bandwidths=[20, 40, 80, 160, 320],
            supported_bands=["2.4GHz", "5GHz", "6GHz"],
            max_tx_antennas=2,
            max_rx_antennas=2,
            subcarrier_configs=_create_be_subcarrier_configs(),
            supports_mimo=True,
            has_reliable_agc=True,
        )

    else:
        # Should not reach here
        raise ValueError(f"Unhandled device: {device}")

    # Add explicit features that may not be auto-detected
    caps.add_feature(FEATURE_PHASE_CALIBRATION)

    if device in WIFI7_DEVICES:
        caps.add_feature(FEATURE_WIFI7)

    return caps


# =============================================================================
# IwlWifi Extractor
# =============================================================================

class IwlWifiExtractor(CSIExtractor):
    """
    CSI Extractor for Intel iwlwifi-based WiFi 6/6E/7 cards.

    This is a unified extractor that can handle CSI data from various
    sources (PicoScenes, IAX, FeitCSI, raw iwlwifi).

    Args:
        device: Intel device identifier (ax200, ax210, be200, etc.)
        source: CSI data source format (auto-detected if not specified)

    Example:
        # For AX210 with PicoScenes data
        extractor = IwlWifiExtractor(device="ax210", source=CSISource.PICOSCENES)

        # For BE200 (future)
        extractor = IwlWifiExtractor(device="be200")

    Note:
        This is currently a stub. Full implementation requires integration
        with the actual CSI extraction tools.
    """

    def __init__(
        self,
        device: str = DEVICE_AX210,
        source: CSISource = CSISource.UNKNOWN,
    ):
        """
        Initialize the iwlwifi extractor.

        Args:
            device: Device identifier (ax200, ax210, be200, etc.)
            source: CSI data source format
        """
        self.device = device.lower()
        self.source = source
        self._capabilities = get_iwlwifi_capabilities(self.device)

        logger.info(
            f"IwlWifiExtractor initialized for {self.device} "
            f"(source={source.value})"
        )

        if self.device in WIFI7_DEVICES:
            logger.warning(
                f"WiFi 7 device {self.device} selected. "
                "CSI extraction tools for WiFi 7 are not yet publicly available. "
                "This extractor is a placeholder for future support."
            )

    @property
    def capabilities(self) -> HardwareCapabilities:
        """Return hardware capabilities for this device."""
        return self._capabilities

    def parse(
        self,
        data: bytes,
        rx_mac: Optional[str] = None,
    ) -> Optional[CSIPacket]:
        """
        Parse CSI data from iwlwifi-based capture.

        Args:
            data: Raw CSI data bytes
            rx_mac: Receiver MAC address (optional)

        Returns:
            Parsed CSIPacket or None if parsing fails

        Note:
            Dispatches to the appropriate parser based on the configured
            CSI source. Currently supports FeitCSI; PicoScenes and IAX
            are stubs for future implementation.
        """
        if self.source == CSISource.FEITCSI:
            if not hasattr(self, '_feitcsi'):
                from opencsi.hardware.feitcsi import FeitCSIExtractor
                self._feitcsi = FeitCSIExtractor(device=self.device)
            return self._feitcsi.parse(data, rx_mac=rx_mac)

        raise NotImplementedError(
            f"IwlWifiExtractor.parse() is not yet implemented for "
            f"source={self.source.value}. "
            f"CSI extraction for {self.device} requires integration with "
            f"PicoScenes, IAX, or direct iwlwifi vendor commands. "
            f"See: https://ps.zpj.io/ for PicoScenes documentation."
        )

    def parse_picoscenes(
        self,
        data: bytes,
        rx_mac: Optional[str] = None,
    ) -> Optional[CSIPacket]:
        """
        Parse CSI data from PicoScenes format.

        Args:
            data: PicoScenes CSI data
            rx_mac: Receiver MAC address

        Returns:
            Parsed CSIPacket or None

        Note:
            Stub - requires PicoScenes Python bindings.
        """
        raise NotImplementedError(
            "PicoScenes parsing requires the PicoScenes Python API. "
            "Install via: pip install picoscenes (when available) "
            "or use their MATLAB tools."
        )

    def parse_iax(
        self,
        data: bytes,
        rx_mac: Optional[str] = None,
    ) -> Optional[CSIPacket]:
        """
        Parse CSI data from IAX format.

        Args:
            data: IAX CSI data
            rx_mac: Receiver MAC address

        Returns:
            Parsed CSIPacket or None

        Note:
            Stub - requires IAX format specification.
        """
        raise NotImplementedError(
            "IAX parsing not yet implemented. "
            "See: https://ieeexplore.ieee.org/document/10944229/"
        )

    @classmethod
    def list_supported_devices(cls) -> Tuple[str, ...]:
        """Return tuple of supported device identifiers."""
        return tuple(sorted(SUPPORTED_DEVICES))

    @classmethod
    def is_wifi7_device(cls, device: str) -> bool:
        """Check if device is a WiFi 7 card."""
        return device.lower() in WIFI7_DEVICES

    @classmethod
    def is_wifi6e_device(cls, device: str) -> bool:
        """Check if device is a WiFi 6E card (has 6GHz support)."""
        return device.lower() in WIFI6E_DEVICES

    def get_max_subcarriers(self) -> int:
        """Get maximum number of subcarriers for this device."""
        max_bw = max(self._capabilities.supported_bandwidths)
        return self._capabilities.get_num_subcarriers(max_bw)

    def get_default_profile(self) -> dict:
        """Return default calibration parameters for iwlwifi hardware.

        Returns:
            Dictionary with hardware-specific defaults.
        """
        if self.device in WIFI7_DEVICES:
            return {
                "typical_noise_floor_dbm": -91,
                "typical_amplitude_sigma": 1.8,
                "typical_phase_sigma_rad": 0.5,
                "typical_snr_db": 35,
            }
        # WiFi 6 and WiFi 6E share identical calibration profiles
        return {
            "typical_noise_floor_dbm": -92,
            "typical_amplitude_sigma": 2.0,
            "typical_phase_sigma_rad": 0.6,
            "typical_snr_db": 32,
        }

    def __repr__(self) -> str:
        return (
            f"IwlWifiExtractor(device={self.device!r}, "
            f"source={self.source.value!r})"
        )
