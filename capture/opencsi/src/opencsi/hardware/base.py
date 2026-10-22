"""
Abstract base classes for hardware-specific CSI extraction.

This module defines the interface that hardware-specific extractors
must implement to parse raw CSI bytes into a common format.

The key abstraction is CSIExtractor, which provides:
- A `capabilities` property describing hardware features
- A `parse()` method to convert raw bytes to CSIPacket
- Helper methods like `get_pilot_indices()` that query capabilities

Example:
    class MyExtractor(CSIExtractor):
        @property
        def capabilities(self) -> HardwareCapabilities:
            return HardwareCapabilities(...)

        def parse(self, data: bytes, rx_mac: str = None) -> Optional[CSIPacket]:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from opencsi.hardware.capabilities import HardwareCapabilities


@dataclass
class CSIPacket:
    """Hardware-agnostic CSI packet representation.

    This is the canonical CSI packet format used throughout OpenCSI.
    Hardware-specific extractors convert their native formats to this.

    Core fields (required):
        tx_mac, rx_mac: MAC addresses identifying the link
        timestamp_us: Packet timestamp in microseconds
        rssi: Received Signal Strength Indicator in dBm
        noise_floor: Noise floor estimate in dBm
        channel: WiFi channel number
        bandwidth_mhz: Channel bandwidth (20, 40, 80, 160, 320)
        n_subcarriers: Number of subcarriers in iq_data
        iq_data: Complex I/Q data, shape (n_subcarriers, 2)

    Extended fields (optional, for multi-hardware support):
        frequency_band: "2.4GHz", "5GHz", or "6GHz"
        n_tx_antennas: Number of TX antennas (1 for SISO)
        n_rx_antennas: Number of RX antennas (1 for SISO)
        antenna_sel: Antenna selection for MIMO (tx_idx, rx_idx)
        hardware_type: Hardware identifier (e.g., "esp32", "intel_5300")
    """
    # Core fields
    tx_mac: str
    rx_mac: str
    timestamp_us: int
    rssi: int
    noise_floor: int
    channel: int
    bandwidth_mhz: int
    n_subcarriers: int
    iq_data: np.ndarray           # Shape: (n_subcarriers, 2) for I/Q

    # AGC and pilot data
    rx_state: int = 0             # AGC state (hardware-specific)
    pilot_phases: Optional[np.ndarray] = None

    # Extended fields for multi-hardware support
    frequency_band: str = "2.4GHz"  # "2.4GHz", "5GHz", "6GHz"
    n_tx_antennas: int = 1
    n_rx_antennas: int = 1
    antenna_sel: Optional[Tuple[int, int]] = None  # (tx_idx, rx_idx) for MIMO
    hardware_type: str = "unknown"

    @property
    def amplitude(self) -> np.ndarray:
        """Compute amplitude from I/Q data."""
        iq = self.iq_data.astype(np.float32)
        return np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)

    @property
    def phase(self) -> np.ndarray:
        """Compute phase from I/Q data."""
        return np.arctan2(self.iq_data[:, 1], self.iq_data[:, 0])

    @property
    def complex_csi(self) -> np.ndarray:
        """Return CSI as complex array."""
        return self.iq_data[:, 0] + 1j * self.iq_data[:, 1]

    @property
    def snr(self) -> float:
        """Estimated SNR in dB."""
        return float(self.rssi - self.noise_floor)

    @property
    def link_key(self) -> tuple:
        """Return (tx_mac, rx_mac) tuple for link identification."""
        return (self.tx_mac, self.rx_mac)


class CSIExtractor(ABC):
    """Abstract interface for hardware-specific CSI extraction.

    Implement this class to add support for new hardware platforms.
    The key requirement is implementing the `capabilities` property,
    which describes the hardware's features and subcarrier configuration.

    Example:
        class MyHardwareExtractor(CSIExtractor):
            @property
            def capabilities(self) -> HardwareCapabilities:
                return HardwareCapabilities(
                    hardware_type="my_hardware",
                    supported_bandwidths=[20, 40],
                    supported_bands=["2.4GHz"],
                    subcarrier_configs={
                        20: SubcarrierConfig(52, (6, 20, 32, 46), 26),
                        40: SubcarrierConfig(114, (6, 20, 32, 46, 60, 74, 86, 100))
                    }
                )

            def parse(self, data: bytes, rx_mac: str = None) -> Optional[CSIPacket]:
                # Parse hardware-specific format
                ...
                return CSIPacket(...)

            def get_default_profile(self) -> dict:
                return {...}
    """

    @property
    @abstractmethod
    def capabilities(self) -> 'HardwareCapabilities':
        """Hardware capabilities descriptor.

        Returns a HardwareCapabilities object describing the hardware's
        supported bandwidths, bands, MIMO configuration, and subcarrier
        configurations for each bandwidth.

        This is the primary abstraction for multi-hardware support.

        Returns:
            HardwareCapabilities instance
        """
        ...

    @property
    def hardware_type(self) -> str:
        """Hardware identifier (e.g., 'esp32', 'intel_5300').

        Default implementation returns capabilities.hardware_type.

        Returns:
            String identifying the hardware platform.
        """
        return self.capabilities.hardware_type

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers for default bandwidth (20 MHz).

        For multi-bandwidth support, use get_num_subcarriers(bandwidth_mhz).

        Returns:
            Number of subcarriers in CSI data for 20 MHz.
        """
        return self.capabilities.get_num_subcarriers(20)

    def get_num_subcarriers(self, bandwidth_mhz: int = 20) -> int:
        """Get number of subcarriers for a given bandwidth.

        Args:
            bandwidth_mhz: Channel bandwidth in MHz

        Returns:
            Number of subcarriers for that bandwidth
        """
        return self.capabilities.get_num_subcarriers(bandwidth_mhz)

    def get_pilot_indices(self, bandwidth_mhz: int = 20) -> Tuple[int, ...]:
        """Get pilot subcarrier indices for a given bandwidth.

        Args:
            bandwidth_mhz: Channel bandwidth in MHz

        Returns:
            Tuple of pilot subcarrier indices
        """
        return self.capabilities.get_pilot_indices(bandwidth_mhz)

    def get_dc_index(self, bandwidth_mhz: int = 20) -> Optional[int]:
        """Get DC subcarrier index for a given bandwidth.

        Args:
            bandwidth_mhz: Channel bandwidth in MHz

        Returns:
            DC index or None if not applicable
        """
        return self.capabilities.get_dc_index(bandwidth_mhz)

    @abstractmethod
    def parse(self, data: bytes, rx_mac: Optional[str] = None) -> Optional[CSIPacket]:
        """Parse raw bytes into CSIPacket.

        Args:
            data: Raw bytes from the hardware.
            rx_mac: Optional receiver MAC address (may be extracted from data).

        Returns:
            CSIPacket if parsing succeeds, None if data is invalid.
        """
        ...

    @abstractmethod
    def get_default_profile(self) -> dict:
        """Return default calibration parameters for this hardware.

        Returns:
            Dictionary with hardware-specific defaults:
            - typical_noise_floor_dbm: Expected noise floor
            - typical_amplitude_sigma: Expected amplitude variance
            - typical_phase_sigma_rad: Expected phase variance
            - typical_snr_db: Expected SNR
        """
        ...
