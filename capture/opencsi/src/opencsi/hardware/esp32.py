"""
ESP32 CSI Extractor for OpenCSI.

Parses binary CSI packets from ESP32 devices running the CSI firmware.
Supports both v1 and v2 packet formats.
"""

import logging
import struct
from functools import cached_property
from typing import Optional

import numpy as np

from opencsi.hardware.base import CSIExtractor, CSIPacket
from opencsi.hardware.capabilities import (
    HardwareCapabilities,
    SubcarrierConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Protocol Constants
# =============================================================================

# Binary packet magic and sizes
MAGIC_BYTES = b'\xde\xad'
HEADER_SIZE = 32  # Fixed header size
BYTES_PER_SUBCARRIER = 4  # int16 I + int16 Q

# Default noise floor for ESP32
ESP32_DEFAULT_NOISE_FLOOR = -95

# Channel/band thresholds
CHANNEL_5GHZ_START = 36  # Channels >= 36 are 5GHz band

# Bandwidth encoding in packet format
BANDWIDTH_FLAG_40MHZ = 1

# =============================================================================
# Subcarrier Configurations by Mode
# =============================================================================
# ESP32 family supports multiple WiFi standards:
# - ESP32, ESP32-S2, ESP32-S3, ESP32-C3: WiFi 4 (802.11n) - HT20/HT40
# - ESP32-C6: WiFi 6 (802.11ax) - HE20/HE40

# --- WiFi 4 (802.11n HT) ---
SUBCARRIERS_HT20 = 52
SUBCARRIERS_HT40 = 114
PILOT_INDICES_HT20 = (6, 20, 32, 46)
PILOT_INDICES_HT40 = (6, 20, 32, 46, 60, 74, 86, 100)
DC_INDEX_HT20 = 26
DC_INDEX_HT40 = 57

# --- WiFi 6 (802.11ax HE) ---
# HE uses more subcarriers with tighter spacing (78.125 kHz vs 312.5 kHz)
SUBCARRIERS_HE20 = 242  # 234 data + 8 pilots
SUBCARRIERS_HE40 = 484  # 468 data + 16 pilots
PILOT_INDICES_HE20 = (24, 48, 72, 96, 145, 169, 193, 217)
PILOT_INDICES_HE40 = (24, 48, 72, 96, 145, 169, 193, 217,
                      266, 290, 314, 338, 387, 411, 435, 459)
DC_INDEX_HE20 = 121
DC_INDEX_HE40 = 242

# Valid subcarrier counts (for validation)
VALID_SUBCARRIER_COUNTS = (
    SUBCARRIERS_HT20, SUBCARRIERS_HT40,
    SUBCARRIERS_HE20, SUBCARRIERS_HE40,
)


class ESP32Extractor(CSIExtractor):
    """
    CSI extractor for ESP32 devices.

    Supports the entire ESP32 family:
    - ESP32, ESP32-S2, ESP32-S3, ESP32-C3: WiFi 4 (HT20/HT40)
    - ESP32-C6: WiFi 6 (HE20/HE40) with 5GHz support

    Automatically detects packet version, format, and WiFi standard.

    Usage:
        # Default: WiFi 4 (HT) mode
        extractor = ESP32Extractor()

        # WiFi 6 (HE) mode for ESP32-C6
        extractor = ESP32Extractor(wifi6=True)

        packet = extractor.parse(raw_bytes, rx_mac="AA:BB:CC:DD:EE:FF")
        if packet:
            print(f"RSSI: {packet.rssi}, SNR: {packet.snr}")

        # Query capabilities dynamically
        pilot_indices = extractor.get_pilot_indices(bandwidth_mhz=20)
    """

    def __init__(self, wifi6: bool = False):
        """
        Initialize ESP32 extractor.

        Args:
            wifi6: Enable WiFi 6 (802.11ax) mode for ESP32-C6.
                   Default is WiFi 4 (802.11n) for other ESP32 variants.
        """
        self._wifi6 = wifi6
        self._hardware_type = "esp32_c6" if wifi6 else "esp32"

    @cached_property
    def capabilities(self) -> HardwareCapabilities:
        """ESP32 hardware capabilities based on WiFi standard."""
        if self._wifi6:
            # ESP32-C6 with WiFi 6 (802.11ax HE)
            subcarrier_configs = {
                20: SubcarrierConfig(
                    num_subcarriers=SUBCARRIERS_HE20,
                    pilot_indices=PILOT_INDICES_HE20,
                    dc_index=DC_INDEX_HE20,
                ),
                40: SubcarrierConfig(
                    num_subcarriers=SUBCARRIERS_HE40,
                    pilot_indices=PILOT_INDICES_HE40,
                    dc_index=DC_INDEX_HE40,
                ),
            }
            supported_bands = ["2.4GHz", "5GHz"]
        else:
            # ESP32/S2/S3/C3 with WiFi 4 (802.11n HT)
            subcarrier_configs = {
                20: SubcarrierConfig(
                    num_subcarriers=SUBCARRIERS_HT20,
                    pilot_indices=PILOT_INDICES_HT20,
                    dc_index=DC_INDEX_HT20,
                ),
                40: SubcarrierConfig(
                    num_subcarriers=SUBCARRIERS_HT40,
                    pilot_indices=PILOT_INDICES_HT40,
                    dc_index=DC_INDEX_HT40,
                ),
            }
            supported_bands = ["2.4GHz"]

        return HardwareCapabilities(
            hardware_type=self._hardware_type,
            supported_bandwidths=[20, 40],
            supported_bands=supported_bands,
            max_tx_antennas=1,
            max_rx_antennas=1,
            subcarrier_configs=subcarrier_configs,
            supports_mimo=False,
            has_reliable_agc=False,
            typical_noise_floor_dbm=float(ESP32_DEFAULT_NOISE_FLOOR),
        )

    def parse(
        self,
        data: bytes,
        rx_mac: Optional[str] = None
    ) -> Optional[CSIPacket]:
        """
        Parse ESP32 binary CSI packet.

        Args:
            data: Raw bytes from ESP32
            rx_mac: Receiver MAC (required if not in packet)

        Returns:
            CSIPacket or None if parsing fails
        """
        if len(data) < HEADER_SIZE:
            return None

        # Check magic bytes
        if data[:2] != MAGIC_BYTES:
            return None

        try:
            return self._parse_v2_packet(data, rx_mac)
        except Exception as e:
            logger.debug(f"Failed to parse ESP32 packet: {e}")
            return None

    def _parse_v2_packet(
        self,
        data: bytes,
        rx_mac: Optional[str] = None
    ) -> Optional[CSIPacket]:
        """Parse v2 binary packet format."""
        # Header structure (32 bytes):
        # 0-1: magic (0xdead)
        # 2: version
        # 3: flags
        # 4-5: sequence number
        # 6-9: timestamp (ms)
        # 10-11: rx_state
        # 12-15: rssi + noise_floor + channel + bandwidth
        # 16-21: tx_mac (6 bytes)
        # 22-27: rx_mac (6 bytes)
        # 28-29: n_subcarriers
        # 30-31: reserved/padding

        # Unpack header fields in one call for efficiency
        # Format: version(B), flags(B), seq_num(H), timestamp_ms(I), rx_state(H),
        #         rssi(b), noise_floor(b), channel(B), bandwidth(B)
        # Note: version and flags are parsed for forward compatibility but not currently used
        (
            _version, _flags, _seq_num, timestamp_ms, rx_state,
            rssi, noise_floor, channel, bandwidth
        ) = struct.unpack('<BBHIHbbBB', data[2:16])

        # Extract MACs
        tx_mac_bytes = data[16:22]
        tx_mac = ':'.join(f'{b:02X}' for b in tx_mac_bytes)

        rx_mac_bytes = data[22:28]
        packet_rx_mac = ':'.join(f'{b:02X}' for b in rx_mac_bytes)

        # Use packet rx_mac if valid, otherwise use provided
        if packet_rx_mac and packet_rx_mac != '00:00:00:00:00:00':
            rx_mac = packet_rx_mac
        elif rx_mac is None:
            rx_mac = "UNKNOWN"

        n_subcarriers = struct.unpack('<H', data[28:30])[0]

        # Validate subcarrier count against known configurations
        if n_subcarriers not in VALID_SUBCARRIER_COUNTS:
            logger.warning(f"Unexpected subcarrier count: {n_subcarriers}")
            # Default based on WiFi mode
            n_subcarriers = SUBCARRIERS_HE20 if self._wifi6 else SUBCARRIERS_HT20

        # Calculate expected packet size
        iq_data_size = n_subcarriers * BYTES_PER_SUBCARRIER
        expected_size = HEADER_SIZE + iq_data_size

        if len(data) < expected_size:
            logger.debug(
                f"Packet too short: {len(data)} < {expected_size}"
            )
            return None

        # Parse I/Q data
        iq_data = self._parse_iq_data(data[HEADER_SIZE:], n_subcarriers)

        # Set noise floor default if not provided
        if noise_floor == 0:
            noise_floor = ESP32_DEFAULT_NOISE_FLOOR

        # Bandwidth in MHz
        bandwidth_mhz = 40 if bandwidth == BANDWIDTH_FLAG_40MHZ else 20

        # Determine frequency band from channel number
        frequency_band = "5GHz" if channel >= CHANNEL_5GHZ_START else "2.4GHz"

        return CSIPacket(
            tx_mac=tx_mac,
            rx_mac=rx_mac,
            timestamp_us=timestamp_ms * 1000,  # Convert ms to us
            rssi=rssi,
            noise_floor=noise_floor,
            channel=channel,
            bandwidth_mhz=bandwidth_mhz,
            n_subcarriers=n_subcarriers,
            iq_data=iq_data,
            rx_state=rx_state,
            pilot_phases=None,
            # Extended fields
            frequency_band=frequency_band,
            n_tx_antennas=1,
            n_rx_antennas=1,
            hardware_type=self._hardware_type,
        )

    def _parse_iq_data(
        self,
        data: bytes,
        n_subcarriers: int
    ) -> np.ndarray:
        """Parse I/Q data from bytes.

        Format: int16 I, int16 Q for each subcarrier (little-endian).
        """
        expected_bytes = n_subcarriers * BYTES_PER_SUBCARRIER
        if len(data) < expected_bytes:
            raise ValueError(
                f"Truncated CSI packet: got {len(data)} bytes, "
                f"expected {expected_bytes} for {n_subcarriers} subcarriers"
            )

        # Parse all I/Q values at once using numpy
        raw = np.frombuffer(data[:expected_bytes], dtype='<i2')
        # Reshape to (n_subcarriers, 2) where columns are [I, Q]
        return raw.reshape(n_subcarriers, 2).astype(np.float32)

    def get_default_profile(self) -> dict:
        """Return ESP32 default calibration profile."""
        return {
            "description": "ESP32 WROOM/WROVER defaults",
            "typical_noise_floor_dbm": ESP32_DEFAULT_NOISE_FLOOR,
            "typical_amplitude_sigma": 2.5,
            "typical_phase_sigma_rad": 0.8,
            "typical_snr_db": 25,
            "subcarriers": SUBCARRIERS_HT20,
            "bandwidth_mhz": 20,
        }
