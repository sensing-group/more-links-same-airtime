"""
FeitCSI Binary Format Parser for OpenCSI.

Parses CSI data captured by FeitCSI from Intel AX200/AX210/AX211 NICs.
FeitCSI uses the iwlwifi vendor netlink command (IWL_MVM_VENDOR_CMD_CSI_EVENT)
to extract CSI and writes it in a binary format: 272-byte packed header
followed by int16 I/Q pairs.

This module handles the FeitCSI binary protocol. Hardware capabilities
(subcarrier configs, MIMO, bands) are provided by iwlwifi.py.

References:
    - FeitCSI: https://github.com/mhutarern/FeitCSI
    - iwlwifi rs.h: rate_n_flags v2 bit field definitions

Usage:
    from opencsi.hardware.feitcsi import FeitCSIExtractor

    extractor = FeitCSIExtractor(device="ax210", channel=48)

    # Parse a single record (returns first MIMO stream)
    packet = extractor.parse(raw_bytes)

    # Parse all MIMO streams from a record
    packets = extractor.parse_all(raw_bytes)

    # Iterate over a .dat file
    for stream_packets in extractor.iter_file("capture.dat"):
        for packet in stream_packets:
            print(packet.amplitude)
"""

import logging
import struct
from functools import cached_property
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np

from opencsi.hardware.base import CSIExtractor, CSIPacket
from opencsi.hardware.capabilities import HardwareCapabilities
from opencsi.hardware.iwlwifi import DEVICE_AX210, get_iwlwifi_capabilities

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# FeitCSI header size (RawHeaderData struct, __attribute__((__packed__)))
FEITCSI_HEADER_SIZE = 272

# Bytes per subcarrier: int16 real + int16 imag
BYTES_PER_SUBCARRIER = 4

# Default noise floor estimate for AX200/AX210 (not provided by FeitCSI)
IWLWIFI_DEFAULT_NOISE_FLOOR = -92

# Default SNR estimate (dB) when raw RSSI cannot be converted to dBm
IWLWIFI_DEFAULT_SNR_DB = 32

# Plausible dBm range for RSSI values
RSSI_PLAUSIBLE_MIN_DBM = -120
RSSI_PLAUSIBLE_MAX_DBM = 0

# Channel number threshold for 5 GHz band
CHANNEL_5GHZ_START = 36
CHANNEL_6GHZ_START = 191

# =============================================================================
# rateNflag v2 bit field constants (from iwlwifi rs.h)
# =============================================================================

# Bits 10-8: modulation type
RATE_MCS_MOD_TYPE_POS = 8
RATE_MCS_MOD_TYPE_MSK = 0x7 << RATE_MCS_MOD_TYPE_POS

_FORMAT_CCK = 0 << RATE_MCS_MOD_TYPE_POS
_FORMAT_OFDM = 1 << RATE_MCS_MOD_TYPE_POS
_FORMAT_HT = 2 << RATE_MCS_MOD_TYPE_POS
_FORMAT_VHT = 3 << RATE_MCS_MOD_TYPE_POS
_FORMAT_HE = 4 << RATE_MCS_MOD_TYPE_POS
_FORMAT_EHT = 5 << RATE_MCS_MOD_TYPE_POS

_FORMAT_NAMES = {
    _FORMAT_CCK: "CCK",
    _FORMAT_OFDM: "OFDM",
    _FORMAT_HT: "HT",
    _FORMAT_VHT: "VHT",
    _FORMAT_HE: "HE",
    _FORMAT_EHT: "EHT",
}

# Bits 13-11: channel width
RATE_MCS_CHAN_WIDTH_POS = 11
RATE_MCS_CHAN_WIDTH_MSK = 0x7 << RATE_MCS_CHAN_WIDTH_POS

_BW_MAP = {
    0: 20,
    1: 40,
    2: 80,
    3: 160,
    4: 320,
}


# =============================================================================
# Helper functions
# =============================================================================


def _decode_format(rate_n_flag: int) -> str:
    """Decode WiFi format from rateNflag field.

    Args:
        rate_n_flag: Raw rateNflag value from FeitCSI header.

    Returns:
        Format string: "CCK", "OFDM", "HT", "VHT", "HE", or "EHT".
    """
    fmt = rate_n_flag & RATE_MCS_MOD_TYPE_MSK
    return _FORMAT_NAMES.get(fmt, "UNKNOWN")


def _decode_bandwidth_mhz(rate_n_flag: int) -> int:
    """Decode channel bandwidth from rateNflag field.

    Args:
        rate_n_flag: Raw rateNflag value from FeitCSI header.

    Returns:
        Bandwidth in MHz: 20, 40, 80, 160, or 320.
    """
    bw_val = (rate_n_flag & RATE_MCS_CHAN_WIDTH_MSK) >> RATE_MCS_CHAN_WIDTH_POS
    return _BW_MAP.get(bw_val, 20)


def _frequency_band(channel: int) -> str:
    """Determine frequency band from channel number."""
    if channel >= CHANNEL_6GHZ_START:
        return "6GHz"
    if channel >= CHANNEL_5GHZ_START:
        return "5GHz"
    return "2.4GHz"


def _estimate_rssi_dbm(raw_rssi: int) -> int:
    """Convert raw FeitCSI RSSI register value to estimated dBm.

    FeitCSI stores raw firmware energy values (uint32) from iwlwifi's
    rx_mpdu_desc. The exact conversion to dBm requires AGC state which
    is not in the header. We handle three cases:

    1. Negative dBm stored as unsigned 32-bit → signed reinterpretation
    2. Already-plausible dBm value → pass through
    3. Raw positive energy value → fall back to noise_floor + default SNR

    Args:
        raw_rssi: Unsigned 32-bit RSSI value from FeitCSI header.

    Returns:
        Estimated RSSI in dBm.
    """
    # Reinterpret as signed (handles negative dBm stored in uint32)
    signed_val = raw_rssi if raw_rssi < 0x80000000 else raw_rssi - 0x100000000
    if RSSI_PLAUSIBLE_MIN_DBM <= signed_val <= RSSI_PLAUSIBLE_MAX_DBM:
        return int(signed_val)

    # Raw energy register — use default estimate
    logger.debug(
        "FeitCSI raw RSSI %d (signed=%d) outside dBm range, "
        "using noise_floor + default SNR estimate",
        raw_rssi, signed_val,
    )
    return IWLWIFI_DEFAULT_NOISE_FLOOR + IWLWIFI_DEFAULT_SNR_DB


def _fix_160mhz_bug(
    iq_flat: np.ndarray,
    num_subcarriers: int,
    num_rx: int,
    num_tx: int,
    format_type: str,
    bandwidth_mhz: int,
) -> tuple:
    """Apply FeitCSI's 160 MHz firmware bug fix.

    The Intel firmware reports extra garbage subcarriers in 160 MHz mode:
    - VHT 160: indices 242-255 (14 garbage), 498 -> 484
    - HE 160: indices 996-1023 (28 garbage), 2020 -> 1992

    Args:
        iq_flat: Raw I/Q array, shape (num_rx * num_tx * num_subcarriers, 2).
        num_subcarriers: Reported subcarrier count.
        num_rx: Number of RX antennas.
        num_tx: Number of TX antennas.
        format_type: WiFi format string ("VHT", "HE", etc.).
        bandwidth_mhz: Channel bandwidth in MHz.

    Returns:
        Tuple of (fixed_iq, fixed_num_subcarriers).
    """
    if bandwidth_mhz != 160:
        return iq_flat, num_subcarriers

    if format_type == "VHT" and num_subcarriers != 484:
        new_num = 484
        garbage_start, garbage_end = 242, 256
    elif format_type == "HE" and num_subcarriers != 1992:
        new_num = 1992
        garbage_start, garbage_end = 996, 1024
    else:
        return iq_flat, num_subcarriers

    # Reshape to (num_rx, num_tx, num_subcarriers, 2), strip, reshape back
    iq = iq_flat.reshape(num_rx, num_tx, num_subcarriers, 2)
    keep = np.ones(num_subcarriers, dtype=bool)
    keep[garbage_start:garbage_end] = False
    iq_fixed = iq[:, :, keep, :]

    return iq_fixed.reshape(-1, 2), new_num


# =============================================================================
# FeitCSI Extractor
# =============================================================================


class FeitCSIExtractor(CSIExtractor):
    """CSI extractor for FeitCSI binary format.

    Parses the binary output of the FeitCSI tool, which captures CSI
    from Intel AX200/AX210/AX211 NICs via the patched iwlwifi driver.

    Each record is a 272-byte packed header followed by int16 I/Q pairs.
    The header contains metadata (timestamps, antenna counts, subcarrier
    count, source MAC, rate/format flags). The payload contains interleaved
    real/imaginary int16 values ordered as [rx][tx][subcarrier].

    Args:
        device: Intel device identifier (default "ax210").
        channel: WiFi channel number (not in FeitCSI header, must be supplied).
        rx_mac: Receiver MAC address (not in FeitCSI header, must be supplied).

    Usage:
        extractor = FeitCSIExtractor(device="ax210", channel=48)

        # Single record, first MIMO stream
        packet = extractor.parse(raw_bytes)

        # All MIMO streams
        packets = extractor.parse_all(raw_bytes)

        # Iterate .dat file
        for stream_packets in extractor.iter_file("capture.dat"):
            ...
    """

    def __init__(
        self,
        device: str = DEVICE_AX210,
        channel: int = 0,
        rx_mac: str = "00:00:00:00:00:00",
    ):
        self.device = device.lower()
        self.channel = channel
        self.rx_mac = rx_mac
        self._hw_caps = get_iwlwifi_capabilities(self.device)

    @cached_property
    def capabilities(self) -> HardwareCapabilities:
        """Hardware capabilities for the underlying Intel device."""
        return self._hw_caps

    def parse(
        self,
        data: bytes,
        rx_mac: Optional[str] = None,
    ) -> Optional[CSIPacket]:
        """Parse a single FeitCSI record, returning the first MIMO stream.

        Args:
            data: Raw bytes (272-byte header + I/Q payload).
            rx_mac: Override receiver MAC for this packet.

        Returns:
            CSIPacket for the first (rx=0, tx=0) stream, or None on error.
        """
        packets = self.parse_all(data, rx_mac=rx_mac)
        if not packets:
            return None
        return packets[0]

    def parse_all(
        self,
        data: bytes,
        rx_mac: Optional[str] = None,
    ) -> List[CSIPacket]:
        """Parse a single FeitCSI record, returning all MIMO streams.

        Args:
            data: Raw bytes (272-byte header + I/Q payload).
            rx_mac: Override receiver MAC for this packet.

        Returns:
            List of CSIPacket, one per (rx, tx) antenna pair.
            Empty list if parsing fails.
        """
        if len(data) < FEITCSI_HEADER_SIZE:
            logger.debug(
                f"Data too short for FeitCSI header: "
                f"{len(data)} < {FEITCSI_HEADER_SIZE}"
            )
            return []

        # Unpack header fields at known offsets
        csi_data_size = struct.unpack_from('<I', data, 0)[0]
        timestamp = struct.unpack_from('<Q', data, 12)[0]
        num_rx = data[46]
        num_tx = data[47]
        num_subcarriers = struct.unpack_from('<I', data, 52)[0]
        rssi_raw = struct.unpack_from('<I', data, 60)[0]
        src_mac_bytes = data[68:74]
        rate_n_flag = struct.unpack_from('<I', data, 92)[0]

        # Validate
        if num_rx == 0 or num_tx == 0 or num_subcarriers == 0:
            logger.debug(
                f"Invalid FeitCSI header: "
                f"rx={num_rx}, tx={num_tx}, sub={num_subcarriers}"
            )
            return []

        expected_payload = num_rx * num_tx * num_subcarriers * BYTES_PER_SUBCARRIER
        if csi_data_size < expected_payload:
            logger.debug(
                f"FeitCSI csiDataSize ({csi_data_size}) < "
                f"expected ({expected_payload})"
            )
            return []

        total_size = FEITCSI_HEADER_SIZE + csi_data_size
        if len(data) < total_size:
            logger.debug(
                f"Data too short: {len(data)} < {total_size}"
            )
            return []

        # Decode format and bandwidth
        format_type = _decode_format(rate_n_flag)
        bandwidth_mhz = _decode_bandwidth_mhz(rate_n_flag)

        # Parse I/Q payload
        payload = data[FEITCSI_HEADER_SIZE:FEITCSI_HEADER_SIZE + expected_payload]
        iq_flat = np.frombuffer(payload, dtype='<i2').reshape(-1, 2).astype(np.float32)

        # Apply 160 MHz firmware bug fix
        iq_flat, num_subcarriers = _fix_160mhz_bug(
            iq_flat, num_subcarriers, num_rx, num_tx,
            format_type, bandwidth_mhz,
        )

        # Reshape to (num_rx, num_tx, num_subcarriers, 2)
        iq_mimo = iq_flat.reshape(num_rx, num_tx, num_subcarriers, 2)

        # Build common fields
        tx_mac = ':'.join(f'{b:02X}' for b in src_mac_bytes)
        effective_rx_mac = rx_mac or self.rx_mac
        band = _frequency_band(self.channel)
        rssi = _estimate_rssi_dbm(rssi_raw)

        # Produce one CSIPacket per (rx, tx) pair
        packets = []
        for rx_idx in range(num_rx):
            for tx_idx in range(num_tx):
                packet = CSIPacket(
                    tx_mac=tx_mac,
                    rx_mac=effective_rx_mac,
                    timestamp_us=timestamp,
                    rssi=rssi,
                    noise_floor=IWLWIFI_DEFAULT_NOISE_FLOOR,
                    channel=self.channel,
                    bandwidth_mhz=bandwidth_mhz,
                    n_subcarriers=num_subcarriers,
                    iq_data=iq_mimo[rx_idx, tx_idx],
                    frequency_band=band,
                    n_tx_antennas=num_tx,
                    n_rx_antennas=num_rx,
                    antenna_sel=(tx_idx, rx_idx),
                    hardware_type=self._hw_caps.hardware_type,
                )
                packets.append(packet)

        return packets

    def iter_file(
        self,
        path: Union[str, Path],
        rx_mac: Optional[str] = None,
    ) -> Iterator[List[CSIPacket]]:
        """Iterate over records in a FeitCSI .dat file.

        Each yield is a list of CSIPackets (one per MIMO stream) from
        a single capture record.

        Args:
            path: Path to the .dat file.
            rx_mac: Override receiver MAC.

        Yields:
            List[CSIPacket] per record.
        """
        with open(Path(path), 'rb') as f:
            while True:
                header = f.read(FEITCSI_HEADER_SIZE)
                if len(header) < FEITCSI_HEADER_SIZE:
                    break

                csi_data_size = struct.unpack_from('<I', header, 0)[0]
                payload = f.read(csi_data_size)
                if len(payload) < csi_data_size:
                    logger.warning(
                        f"Truncated record in {path}: "
                        f"expected {csi_data_size}, got {len(payload)}"
                    )
                    break

                packets = self.parse_all(header + payload, rx_mac=rx_mac)
                if packets:
                    yield packets

    def get_default_profile(self) -> dict:
        """Return default calibration parameters for iwlwifi hardware."""
        return {
            "description": f"FeitCSI ({self.device.upper()}) defaults",
            "typical_noise_floor_dbm": IWLWIFI_DEFAULT_NOISE_FLOOR,
            "typical_amplitude_sigma": 2.0,
            "typical_phase_sigma_rad": 0.6,
            "typical_snr_db": IWLWIFI_DEFAULT_SNR_DB,
        }
