"""
Atheros CSI Extractor for OpenCSI.

Parses CSI data from Atheros/Qualcomm WiFi chips using the Atheros CSI Tool.
Supports AR9380, AR9580, QCA9558 chipsets.

References:
- Atheros CSI Tool: https://wands.sg/research/wifi/AthesCSI/
- Xie et al., "Precise Power Delay Profiling with Commodity WiFi", MobiCom 2015.
- CSIKit: https://github.com/Gi-z/CSIKit (format reference)
- csiread: https://github.com/citysu/csiread (header format reference)

Supported Chips:
- AR9380: Common in enterprise APs
- AR9580: PCIe variant
- QCA9558: Used in OpenWrt routers
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

# Atheros CSI header size (25 bytes)
# Fields: timestamp(8) + csi_len(2) + tx_channel(2) + err_info(1) +
#         noise_floor(1) + rate(1) + bandwidth(1) + num_tones(1) +
#         nr(1) + nc(1) + rssi(1) + rssi1(1) + rssi2(1) + rssi3(1) +
#         payload_len(2) = 25
ATHEROS_HEADER_SIZE = 25

# Struct format: Q=timestamp, H=csi_len, H=tx_channel,
# B=err_info, B=noise_floor, B=rate, B=bandwidth, B=num_tones,
# B=nr, B=nc, B=rssi, B=rssi1, B=rssi2, B=rssi3, H=payload_len
ATHEROS_HEADER_FORMAT = '<QHHBbBBBBBBBBBH'

# Subcarrier counts (Atheros reports more than Intel)
ATHEROS_SUBCARRIERS_20MHZ = 56
ATHEROS_SUBCARRIERS_40MHZ = 114

# Pilot indices
ATHEROS_PILOTS_20MHZ = (7, 21, 35, 49)
ATHEROS_PILOTS_40MHZ = (7, 21, 35, 49, 63, 77, 91, 105)

# DC indices
ATHEROS_DC_20MHZ = 28
ATHEROS_DC_40MHZ = 57

# Default noise floor
ATHEROS_DEFAULT_NOISE_FLOOR = -95

# Bits per 10-bit symbol used in the CSI data
ATHEROS_BITS_PER_SYMBOL = 10


class AtherosExtractor(CSIExtractor):
    """
    CSI extractor for Atheros/Qualcomm WiFi chips.

    Parses the binary format captured via the modified ath9k driver.
    The CSI data uses 10-bit signed integers packed into byte groups
    (5 bytes = 4 values), with imaginary components stored before real.

    Supports:
        - AR9380
        - AR9580
        - QCA9558

    Usage:
        extractor = AtherosExtractor(chip="ar9380")
        packet = extractor.parse(raw_bytes)
    """

    def __init__(self, chip: str = "ar9380"):
        """
        Initialize Atheros extractor.

        Args:
            chip: Atheros chip variant ("ar9380", "ar9580", "qca9558")
        """
        self._chip = chip.lower()

    @cached_property
    def capabilities(self) -> HardwareCapabilities:
        """Atheros hardware capabilities."""
        return HardwareCapabilities(
            hardware_type=f"atheros_{self._chip}",
            supported_bandwidths=[20, 40],
            supported_bands=["2.4GHz", "5GHz"],
            max_tx_antennas=3,
            max_rx_antennas=3,
            subcarrier_configs={
                20: SubcarrierConfig(
                    num_subcarriers=ATHEROS_SUBCARRIERS_20MHZ,
                    pilot_indices=ATHEROS_PILOTS_20MHZ,
                    dc_index=ATHEROS_DC_20MHZ,
                ),
                40: SubcarrierConfig(
                    num_subcarriers=ATHEROS_SUBCARRIERS_40MHZ,
                    pilot_indices=ATHEROS_PILOTS_40MHZ,
                    dc_index=ATHEROS_DC_40MHZ,
                ),
            },
            supports_mimo=True,
            has_reliable_agc=True,
            typical_noise_floor_dbm=float(ATHEROS_DEFAULT_NOISE_FLOOR),
        )

    def parse(
        self,
        data: bytes,
        rx_mac: Optional[str] = None
    ) -> Optional[CSIPacket]:
        """
        Parse Atheros CSI data.

        Args:
            data: Raw bytes from Atheros CSI capture
            rx_mac: Receiver MAC address

        Returns:
            CSIPacket or None if parsing fails
        """
        if len(data) < ATHEROS_HEADER_SIZE:
            return None

        try:
            (
                timestamp,
                csi_len,
                tx_channel,
                err_info,
                noise_floor,
                rate,
                bandwidth,
                num_tones,
                nr,
                nc,
                rssi,
                rssi1,
                rssi2,
                rssi3,
                payload_len
            ) = struct.unpack(ATHEROS_HEADER_FORMAT, data[:ATHEROS_HEADER_SIZE])
        except struct.error:
            logger.debug("Failed to unpack Atheros header")
            return None

        # Validate data length
        expected_len = ATHEROS_HEADER_SIZE + csi_len
        if len(data) < expected_len:
            logger.debug(f"Incomplete packet: {len(data)} < {expected_len}")
            return None

        # Extract CSI payload
        payload = data[ATHEROS_HEADER_SIZE:ATHEROS_HEADER_SIZE + csi_len]

        # Unpack 10-bit signed integers
        # We process 5 bytes at a time to get 4 values
        # Number of values needed = num_tones * nr * nc * 2 (complex)
        required_values = num_tones * nr * nc * 2

        unpacked_values = []
        idx = 0
        limit = len(payload)

        # Iterate through payload in 5-byte chunks
        while idx + 4 < limit:
            b0 = payload[idx]
            b1 = payload[idx + 1]
            b2 = payload[idx + 2]
            b3 = payload[idx + 3]
            b4 = payload[idx + 4]

            # Unpack 4 values (10 bits each from 5 bytes)
            v0 = (b0 >> 0) | ((b1 & 0x3) << 8)
            v1 = (b1 >> 2) | ((b2 & 0xF) << 6)
            v2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
            v3 = (b3 >> 6) | ((b4 & 0xFF) << 2)

            unpacked_values.extend([v0, v1, v2, v3])
            idx += 5

        # Convert to numpy array for efficiency and handle sign extension
        raw_values = np.array(unpacked_values, dtype=np.int16)

        # 10-bit signed conversion: 0-511 positive, 512-1023 negative
        raw_values[raw_values >= 512] -= 1024

        if len(raw_values) < required_values:
            logger.debug(
                f"Insufficient unpacked values: "
                f"{len(raw_values)} < {required_values}"
            )
            return None

        # Truncate if we have extra values (due to 4-value chunks)
        csi_values = raw_values[:required_values]

        # Reshape to (num_tones, nr, nc, 2)
        # IMPORTANT: Atheros format stores [imaginary, real] pairs,
        # not [real, imaginary]
        try:
            csi_matrix = csi_values.reshape((num_tones, nc, nr, 2))
        except ValueError:
            logger.debug("Failed to reshape CSI matrix")
            return None

        # Select first stream (0, 0)
        # Swap I/Q: index 0 is imaginary, index 1 is real
        iq_stream = csi_matrix[:, 0, 0, :]
        iq_data = np.stack(
            [iq_stream[:, 1], iq_stream[:, 0]], axis=1
        ).astype(np.float32)

        # Determine bandwidth
        bandwidth_mhz = 40 if bandwidth == 1 else 20

        # Determine frequency band and channel
        frequency_band = "2.4GHz"
        channel = tx_channel

        # If tx_channel is frequency in MHz (> 1000)
        if tx_channel > 1000:
            if tx_channel > 5000:
                frequency_band = "5GHz"
                # Approximate channel number for 5GHz
                channel = (tx_channel - 5000) // 5
            else:
                frequency_band = "2.4GHz"
                # Approximate channel number for 2.4GHz
                channel = (tx_channel - 2407) // 5
                if channel < 1 or channel > 14:
                    channel = 0  # Invalid
        else:
            # Assumed to be channel number already
            if tx_channel > 14:
                frequency_band = "5GHz"
            else:
                frequency_band = "2.4GHz"

        return CSIPacket(
            tx_mac="00:00:00:00:00:00",  # TX MAC not in header
            rx_mac=rx_mac if rx_mac else "00:00:00:00:00:00",
            timestamp_us=timestamp,
            rssi=rssi,
            noise_floor=noise_floor,
            channel=channel,
            bandwidth_mhz=bandwidth_mhz,
            n_subcarriers=num_tones,
            iq_data=iq_data,
            rx_state=err_info,
            frequency_band=frequency_band,
            n_tx_antennas=nc,
            n_rx_antennas=nr,
            antenna_sel=(0, 0),
            hardware_type=self.hardware_type
        )

    def get_default_profile(self) -> dict:
        """Return Atheros default calibration profile."""
        return {
            "description": f"Atheros {self._chip.upper()} defaults",
            "typical_noise_floor_dbm": ATHEROS_DEFAULT_NOISE_FLOOR,
            "typical_amplitude_sigma": 2.5,
            "typical_phase_sigma_rad": 0.8,
            "typical_snr_db": 28,
            "subcarriers": ATHEROS_SUBCARRIERS_20MHZ,
            "bandwidth_mhz": 20,
            "supports_mimo": True,
        }
