"""
Nexmon CSI Extractor for OpenCSI.

Parses CSI data from Broadcom WiFi chips using the Nexmon CSI firmware.
Supports BCM4339, BCM43455, BCM43456 (int16 I/Q) and BCM4358, BCM4366c0
(packed-float I/Q) chipsets.

References:
- Nexmon CSI: https://github.com/seemoo-lab/nexmon_csi
- nexcsi: https://github.com/nexmonster/nexcsi (header struct reference)
- csiread: https://github.com/citysu/csiread (float format reference)
- CSIKit: https://github.com/Gi-z/CSIKit (multi-chip support reference)
- Gringoli et al., "Free Your CSI: A Channel State Information
  Extraction Platform For Modern Wi-Fi Chipsets", WiNTECH 2019.

Supported Chips:
- BCM4339: Nexus 5 (20/40/80 MHz, int16)
- BCM43455: Raspberry Pi 3B+/4 (20/40/80 MHz, int16)
- BCM43456: Various devices (20/40/80 MHz, int16)
- BCM4358: Samsung Galaxy S6/Note 4 (20/40/80 MHz, packed float 9+5)
- BCM4366c0: Asus RT-AC86U (20/40/80 MHz, packed float 12+6)
"""

import logging
import struct
import time
from functools import cached_property
from typing import Optional, Tuple

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

# Nexmon UDP packet header size
NEXMON_HEADER_SIZE = 18

# Magic bytes for Nexmon packets
NEXMON_MAGIC = b'\x11\x11'

# Subcarrier counts by bandwidth
NEXMON_SUBCARRIERS_20MHZ = 64
NEXMON_SUBCARRIERS_40MHZ = 128
NEXMON_SUBCARRIERS_80MHZ = 256

# Pilot indices by bandwidth
NEXMON_PILOTS_20MHZ = (11, 25, 39, 53)
NEXMON_PILOTS_40MHZ = (11, 39, 53, 75, 89, 117)
NEXMON_PILOTS_80MHZ = (25, 53, 89, 117, 139, 167, 203, 231)

# DC indices
NEXMON_DC_20MHZ = 32
NEXMON_DC_40MHZ = 64
NEXMON_DC_80MHZ = 128

# Default noise floor
NEXMON_DEFAULT_NOISE_FLOOR = -93

# =============================================================================
# Chanspec Bitfield Constants (Broadcom chanspec format)
# =============================================================================

CHANSPEC_CHAN_MASK = 0x00FF
CHANSPEC_BW_MASK = 0x0C00
CHANSPEC_BW_SHIFT = 10
CHANSPEC_BAND_MASK = 0xC000
CHANSPEC_BAND_SHIFT = 14
CHANSPEC_BW_MAP = {0: 20, 1: 40, 2: 80, 3: 160}
CHANSPEC_BAND_5G = 0x03

# =============================================================================
# Core/Spatial Stream Bitfield Constants
# =============================================================================

CORE_MASK = 0x07
SPATIAL_STREAM_MASK = 0x07
SPATIAL_STREAM_SHIFT = 3

# =============================================================================
# Chip Variants and CSI Format
# =============================================================================

# Chips that use int16 I/Q format
INT16_CHIPS = frozenset({"bcm4339", "bcm43455", "bcm43456"})

# Chips that use packed-float I/Q format: (mantissa_bits, exponent_bits)
FLOAT_CHIPS = {
    "bcm4358": (9, 5),    # 14 bits per component
    "bcm4366c0": (12, 6),  # 18 bits per component
}

SUPPORTED_CHIPS = INT16_CHIPS | frozenset(FLOAT_CHIPS.keys())

# Bytes per subcarrier (same for all formats: int16 I+Q or packed-float uint32)
BYTES_PER_SUBCARRIER = 4

# =============================================================================
# Chip Auto-Detection from chip_ver field (bytes 16-17)
# =============================================================================

# Mapping from chip_ver uint16 → chip name.
# Source: CSIKit CHIPS dict, nexcsi, csiread.
CHIP_VER_MAP = {
    0x0001: "bcm4339",     # "0100" LE
    0x0065: "bcm43455",    # "6500" LE
    0xa6dc: "bcm43455",    # "dca6" LE
    0x0003: "bcm4358",     # "0300" LE
    0xdead: "bcm4358",     # "adde" LE
    0xe834: "bcm4366c0",   # "34e8" LE
    0x006a: "bcm4366c0",   # "6a00" LE
}

# Autoscaling target bit width (from Broadcom firmware, matches CSIKit)
FLOAT_AUTOSCALE_NBITS = 10


def _decode_chanspec(chanspec: int) -> Tuple[int, int, str]:
    """
    Decode Broadcom chanspec bitfield.

    Args:
        chanspec: Raw 16-bit chanspec value from Nexmon header.

    Returns:
        Tuple of (channel_number, bandwidth_mhz, frequency_band).
        bandwidth_mhz is 0 if the chanspec BW field is unrecognized.
        frequency_band is "5GHz" or "2.4GHz".
    """
    channel = chanspec & CHANSPEC_CHAN_MASK
    bw_code = (chanspec & CHANSPEC_BW_MASK) >> CHANSPEC_BW_SHIFT
    band_code = (chanspec & CHANSPEC_BAND_MASK) >> CHANSPEC_BAND_SHIFT

    bandwidth_mhz = CHANSPEC_BW_MAP.get(bw_code, 0)
    frequency_band = "5GHz" if band_code == CHANSPEC_BAND_5G else "2.4GHz"

    return channel, bandwidth_mhz, frequency_band


def _decode_core_spatial(css: int) -> Tuple[int, int]:
    """
    Extract core and spatial stream indices from core_and_spatial field.

    Args:
        css: Raw 16-bit core_and_spatial value from Nexmon header.

    Returns:
        Tuple of (core_index, spatial_stream_index).
    """
    core = css & CORE_MASK
    spatial_stream = (css >> SPATIAL_STREAM_SHIFT) & SPATIAL_STREAM_MASK
    return core, spatial_stream


def _unpack_float_csi(
    raw: bytes,
    n_subcarriers: int,
    nman: int,
    nexp: int,
) -> np.ndarray:
    """
    Unpack packed-float CSI data used by BCM4358 and BCM4366c0.

    Each subcarrier is stored as a 32-bit little-endian word with layout
    (MSB to LSB):
        [sign_r (1)][mantissa_r (nman-1)][sign_i (1)][mantissa_i (nman-1)][exponent (nexp)]

    Total used bits per subcarrier: 2*nman + nexp (remaining bits unused).
    The exponent is SIGNED (offset binary, bias = 2^(nexp-1)) and SHARED
    between the real and imaginary components.

    Autoscaling normalizes all values by the peak effective exponent
    (exponent + highest mantissa bit position) so the output dynamic
    range is bounded by FLOAT_AUTOSCALE_NBITS.

    Reference: CSIKit byteops.unpack_float_acphy, csiread Nexmon._parse_float,
    Broadcom firmware tof_unpack_csi.

    Args:
        raw: Raw bytes after the 18-byte header.
        n_subcarriers: Expected number of subcarriers.
        nman: Number of mantissa bits per component (including sign).
        nexp: Number of exponent bits (shared between I and Q).

    Returns:
        ndarray of shape (n_subcarriers, 2) with float32 I/Q values.
    """
    nbytes_needed = n_subcarriers * BYTES_PER_SUBCARRIER
    if len(raw) < nbytes_needed:
        raise ValueError(
            f"Need {nbytes_needed} bytes for {n_subcarriers} subcarriers, "
            f"got {len(raw)}"
        )

    # Each subcarrier is a 32-bit LE word
    H = np.frombuffer(raw[:nbytes_needed], dtype='<u4')

    # Bit masks
    iq_mask = np.uint32((1 << (nman - 1)) - 1)
    e_mask = np.uint32((1 << nexp) - 1)
    e_p = 1 << (nexp - 1)  # exponent bias
    sgnr_shift = nexp + 2 * nman - 1  # bit position of real sign
    sgni_shift = nexp + nman - 1      # bit position of imag sign

    # Extract unsigned mantissa values (nman-1 bits each, no sign)
    vi = ((H >> np.uint32(nexp + nman)) & iq_mask).astype(np.int64)
    vq = ((H >> np.uint32(nexp)) & iq_mask).astype(np.int64)

    # Extract exponent (nexp bits, make signed)
    e = (H & e_mask).astype(np.int32)
    signed_mask = e >= e_p
    e[signed_mask] -= (e_p << 1)

    # Extract sign bits
    sign_r = ((H >> np.uint32(sgnr_shift)) & np.uint32(1)).astype(bool)
    sign_i = ((H >> np.uint32(sgni_shift)) & np.uint32(1)).astype(bool)

    # Autoscaling: find highest set bit in (vi | vq) per subcarrier
    x = (vi | vq).astype(np.int64)
    highest_bit = np.zeros(n_subcarriers, dtype=np.int32)
    for bit_pos in range(nman - 1):
        has_bit = (x >> bit_pos) & 1
        highest_bit = np.where(has_bit, bit_pos + 1, highest_bit)

    # Effective exponent = signed_exponent + highest_bit
    # Only consider subcarriers with nonzero mantissa (matches CSIKit)
    effective_e = e + highest_bit
    nonzero = x > 0
    if np.any(nonzero):
        maxbit = int(np.max(effective_e[nonzero]))
    else:
        maxbit = -e_p  # all-zero mantissa: fallback
    shft = FLOAT_AUTOSCALE_NBITS - maxbit

    # Apply shared exponent + autoscale shift
    e_shifted = e + shft
    e_zero = -nman  # below this exponent, value clamps to 0

    # Vectorized exponent application using float power-of-2
    scale = np.power(2.0, e_shifted.astype(np.float64))
    scale[e_shifted < e_zero] = 0.0

    out_r = vi.astype(np.float64) * scale
    out_i = vq.astype(np.float64) * scale

    # Apply signs
    out_r[sign_r] *= -1.0
    out_i[sign_i] *= -1.0

    return np.stack([out_r, out_i], axis=1).astype(np.float32)


class NexmonExtractor(CSIExtractor):
    """
    CSI extractor for Nexmon-enabled Broadcom WiFi chips.

    Parses the Nexmon CSI UDP format: 18-byte header followed by
    interleaved I/Q data. Supports int16 format (BCM4339, BCM43455,
    BCM43456) and packed-float format (BCM4358, BCM4366c0).

    Chanspec decoding extracts channel, bandwidth, and band from the
    Broadcom chanspec bitfield. Core/spatial stream indices are parsed
    from the core_and_spatial header field.

    Usage:
        extractor = NexmonExtractor(chip="bcm43455")
        packet = extractor.parse(udp_payload)
    """

    def __init__(self, chip: str = "bcm43455"):
        """
        Initialize Nexmon extractor.

        Args:
            chip: Broadcom chip variant or "auto" for auto-detection.
                  Supported: "bcm4339", "bcm43455", "bcm43456" (int16 I/Q),
                  "bcm4358" (packed float 9+5), "bcm4366c0" (packed float 12+6),
                  "auto" (detect from chip_ver header field).

        Raises:
            ValueError: If chip is not "auto" and not supported.
        """
        self._chip = chip.lower()
        self._auto_detect = self._chip == "auto"
        if not self._auto_detect and self._chip not in SUPPORTED_CHIPS:
            raise ValueError(
                f"Unsupported chip '{chip}'. "
                f"Supported: {sorted(SUPPORTED_CHIPS)} or 'auto'"
            )

    @cached_property
    def capabilities(self) -> HardwareCapabilities:
        """Nexmon hardware capabilities."""
        chip_label = self._chip if not self._auto_detect else "auto"
        return HardwareCapabilities(
            hardware_type=f"nexmon_{chip_label}",
            supported_bandwidths=[20, 40, 80],
            supported_bands=["2.4GHz", "5GHz"],
            max_tx_antennas=1,
            max_rx_antennas=1,
            subcarrier_configs={
                20: SubcarrierConfig(
                    num_subcarriers=NEXMON_SUBCARRIERS_20MHZ,
                    pilot_indices=NEXMON_PILOTS_20MHZ,
                    dc_index=NEXMON_DC_20MHZ,
                ),
                40: SubcarrierConfig(
                    num_subcarriers=NEXMON_SUBCARRIERS_40MHZ,
                    pilot_indices=NEXMON_PILOTS_40MHZ,
                    dc_index=NEXMON_DC_40MHZ,
                ),
                80: SubcarrierConfig(
                    num_subcarriers=NEXMON_SUBCARRIERS_80MHZ,
                    pilot_indices=NEXMON_PILOTS_80MHZ,
                    dc_index=NEXMON_DC_80MHZ,
                ),
            },
            supports_mimo=False,
            has_reliable_agc=False,
            typical_noise_floor_dbm=float(NEXMON_DEFAULT_NOISE_FLOOR),
        )

    def parse(
        self,
        data: bytes,
        rx_mac: Optional[str] = None
    ) -> Optional[CSIPacket]:
        """
        Parse Nexmon CSI UDP packet.

        Parses the Nexmon CSI format which consists of an 18-byte header
        followed by interleaved I/Q data (int16 or packed-float depending
        on chip variant).

        Args:
            data: Raw UDP payload from Nexmon CSI.
            rx_mac: Receiver MAC address (optional, not in the header).

        Returns:
            CSIPacket or None if parsing fails.
        """
        if len(data) < NEXMON_HEADER_SIZE:
            logger.debug(f"Packet too short: {len(data)} bytes")
            return None

        try:
            # Unpack header (Little Endian, 18 bytes)
            (
                magic,
                rssi,
                fc,
                src_mac_bytes,
                seq,
                core_spatial,
                chanspec,
                chip_ver
            ) = struct.unpack('<HbB6sHHHH', data[:NEXMON_HEADER_SIZE])

            if magic != 0x1111:
                logger.debug(f"Invalid Nexmon magic bytes: {magic:#x}")
                return None

            tx_mac = ':'.join(f'{b:02x}' for b in src_mac_bytes)

            # Resolve chip identity for this packet
            resolved_chip = self._resolve_chip(chip_ver)

            # Decode chanspec for channel, bandwidth, and band
            channel, chanspec_bw, frequency_band = _decode_chanspec(chanspec)

            # Decode core/spatial stream
            core, spatial_stream = _decode_core_spatial(core_spatial)

            # Determine bandwidth: prefer chanspec, fall back to payload size
            payload = data[NEXMON_HEADER_SIZE:]
            bandwidth_mhz = chanspec_bw
            n_subcarriers = self._determine_subcarriers(
                payload, bandwidth_mhz
            )

            if bandwidth_mhz == 0:
                # Chanspec had invalid BW — infer from subcarrier count
                bandwidth_mhz = self._bw_from_subcarriers(n_subcarriers)

            # Determine frequency band from chanspec; if channel alone was
            # stored (legacy chanspec=channel), fall back to channel number
            if chanspec_bw == 0 and channel > 0:
                frequency_band = "5GHz" if channel > 14 else "2.4GHz"

            # Parse CSI I/Q data based on chip format
            iq_data = self._parse_iq(payload, n_subcarriers, resolved_chip)

            return CSIPacket(
                tx_mac=tx_mac,
                rx_mac=rx_mac or "00:00:00:00:00:00",
                timestamp_us=int(time.time() * 1e6),
                rssi=rssi,
                noise_floor=int(NEXMON_DEFAULT_NOISE_FLOOR),
                channel=channel,
                bandwidth_mhz=bandwidth_mhz,
                n_subcarriers=n_subcarriers,
                iq_data=iq_data,
                hardware_type=self.capabilities.hardware_type,
                rx_state=fc,
                antenna_sel=(core, spatial_stream),
                frequency_band=frequency_band,
            )

        except struct.error as e:
            logger.error(f"Failed to unpack Nexmon header: {e}")
            return None
        except ValueError as e:
            logger.error(f"Failed to parse CSI data: {e}")
            return None

    def _resolve_chip(self, chip_ver: int) -> str:
        """
        Resolve chip identity from header or configured value.

        In auto mode, looks up chip_ver in CHIP_VER_MAP. Falls back to
        bcm43455 (int16) if chip_ver is unknown.

        Args:
            chip_ver: Raw chip_ver uint16 from Nexmon header.

        Returns:
            Resolved chip name string.
        """
        if not self._auto_detect:
            return self._chip
        detected = CHIP_VER_MAP.get(chip_ver)
        if detected is not None:
            return detected
        logger.debug(
            f"Unknown chip_ver {chip_ver:#06x}, defaulting to bcm43455"
        )
        return "bcm43455"

    def _determine_subcarriers(
        self, payload: bytes, chanspec_bw: int
    ) -> int:
        """
        Determine number of subcarriers from payload and bandwidth.

        All chip formats use 4 bytes per subcarrier (int16: I16+Q16,
        packed-float: uint32 with shared exponent).

        Args:
            payload: CSI data bytes (after header).
            chanspec_bw: Bandwidth from chanspec (0 if invalid).

        Returns:
            Number of subcarriers.
        """
        n_from_payload = len(payload) // BYTES_PER_SUBCARRIER

        # If chanspec bandwidth is valid, use its expected count
        bw_to_nsub = {
            20: NEXMON_SUBCARRIERS_20MHZ,
            40: NEXMON_SUBCARRIERS_40MHZ,
            80: NEXMON_SUBCARRIERS_80MHZ,
        }
        if chanspec_bw in bw_to_nsub:
            expected = bw_to_nsub[chanspec_bw]
            if n_from_payload >= expected:
                return expected

        return n_from_payload

    def _bw_from_subcarriers(self, n_subcarriers: int) -> int:
        """
        Infer bandwidth from subcarrier count (fallback).

        Args:
            n_subcarriers: Number of subcarriers detected.

        Returns:
            Bandwidth in MHz.
        """
        if n_subcarriers <= NEXMON_SUBCARRIERS_20MHZ:
            return 20
        if n_subcarriers <= NEXMON_SUBCARRIERS_40MHZ:
            return 40
        return 80

    def _parse_iq(
        self, payload: bytes, n_subcarriers: int, chip: str
    ) -> np.ndarray:
        """
        Parse I/Q data from payload, routing to int16 or float format.

        Args:
            payload: CSI data bytes (after header).
            n_subcarriers: Number of subcarriers to extract.
            chip: Resolved chip name (may differ from self._chip in auto mode).

        Returns:
            ndarray of shape (n_subcarriers, 2) with I/Q values.
        """
        if chip in FLOAT_CHIPS:
            nman, nexp = FLOAT_CHIPS[chip]
            return _unpack_float_csi(payload, n_subcarriers, nman, nexp)

        # int16 path
        nbytes = n_subcarriers * BYTES_PER_SUBCARRIER
        iq_flat = np.frombuffer(payload[:nbytes], dtype='<i2')
        return iq_flat.reshape(n_subcarriers, 2).astype(np.float32)

    def get_default_profile(self) -> dict:
        """Return Nexmon default calibration profile."""
        return {
            "description": f"Nexmon {self._chip.upper()} defaults",
            "typical_noise_floor_dbm": NEXMON_DEFAULT_NOISE_FLOOR,
            "typical_amplitude_sigma": 2.8,
            "typical_phase_sigma_rad": 0.85,
            "typical_snr_db": 27,
            "subcarriers": NEXMON_SUBCARRIERS_20MHZ,
            "bandwidth_mhz": 20,
        }
