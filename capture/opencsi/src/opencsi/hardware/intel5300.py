"""
Intel WiFi Link 5300 CSI Extractor for OpenCSI.

Parses CSI data from Intel 5300 NICs using the Linux 802.11n CSI Tool.

References:
- Linux 802.11n CSI Tool: https://dhalperi.github.io/linux-80211n-csitool/
- Halperin et al., "Tool Release: Gathering 802.11n Traces with Channel
  State Information", ACM SIGCOMM CCR, 2011.
- read_bfee.c: https://github.com/dhalperi/linux-80211n-csitool-supplementary
- CSIKit: https://github.com/Gi-z/CSIKit (scaling reference)

Packet Format:
- 20-byte header followed by bit-packed CSI payload
- 30 subcarrier groups, each with 3-bit padding
- 8-bit signed I/Q values at non-byte-aligned positions
- Supports up to 3x3 MIMO configurations
"""

import logging
import struct
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

# Intel 5300 reports 30 "subcarrier groups" in HT20 mode
# Each group represents 2 adjacent subcarriers
INTEL_SUBCARRIER_GROUPS = 30

# Bits per complex I/Q value (8 real + 8 imaginary)
INTEL_BITS_PER_COMPLEX = 16

# Bits of padding at the start of each subcarrier group
INTEL_BITS_PER_GROUP_PADDING = 3

# Pilot subcarrier group positions (approximate)
INTEL_PILOT_GROUPS = (7, 21)

# Default noise floor for Intel 5300
INTEL_DEFAULT_NOISE_FLOOR = -92

# Fallback noise floor when noise == -127 (sentinel value)
INTEL_FALLBACK_NOISE_FLOOR = -92

# RSSI offset (hardware-specific constant from CSI Tool)
INTEL_RSSI_OFFSET_DB = 44

# Header size for CSI data (20 bytes)
INTEL_HEADER_SIZE = 20

# Antenna permutation mask (2 bits per antenna, 3 antennas)
ANTENNA_PERM_MASK = 0x03
ANTENNA_PERM_SHIFT = 2


def _extract_csi_bitpacked(
    payload: bytes,
    nrx: int,
    ntx: int,
) -> np.ndarray:
    """
    Extract CSI coefficients from bit-packed Intel 5300 format.

    Each of 30 subcarrier groups has 3 bits of padding, followed by
    nrx * ntx complex values. Each complex value is 16 bits: 8 for
    the real part and 8 for the imaginary part, at non-byte-aligned
    positions.

    Reference: read_bfee.c from linux-80211n-csitool-supplementary.

    Args:
        payload: CSI payload bytes (after 20-byte header).
        nrx: Number of receive antennas.
        ntx: Number of transmit antennas.

    Returns:
        Complex ndarray of shape (30, nrx, ntx).
    """
    csi = np.zeros((INTEL_SUBCARRIER_GROUPS, nrx, ntx), dtype=np.complex64)
    index = 0  # bit index into payload

    for i in range(INTEL_SUBCARRIER_GROUPS):
        index += INTEL_BITS_PER_GROUP_PADDING
        remainder = index % 8

        for j in range(nrx * ntx):
            byte_idx = index // 8

            # Extract 8-bit signed real value at non-byte-aligned position.
            # Mask to 8 bits before int8 cast — C truncates implicitly,
            # Python does not.
            raw_real = (
                (payload[byte_idx] >> remainder)
                | (payload[byte_idx + 1] << (8 - remainder))
            ) & 0xFF
            real_val = np.int8(raw_real) if raw_real < 128 else np.int8(raw_real - 256)

            # Extract 8-bit signed imaginary value
            raw_imag = (
                (payload[byte_idx + 1] >> remainder)
                | (payload[byte_idx + 2] << (8 - remainder))
            ) & 0xFF
            imag_val = np.int8(raw_imag) if raw_imag < 128 else np.int8(raw_imag - 256)

            # Map to (subcarrier, rx, tx)
            rx_idx = j // ntx
            tx_idx = j % ntx
            csi[i, rx_idx, tx_idx] = complex(float(real_val), float(imag_val))

            index += INTEL_BITS_PER_COMPLEX

    return csi


def _get_antenna_permutation(antenna_sel: int) -> Tuple[int, int, int]:
    """
    Extract antenna permutation from antenna_sel byte.

    The antenna_sel byte encodes which physical antenna maps to each
    logical antenna position using 2-bit segments.

    Reference: read_bfee.c antenna permutation.

    Args:
        antenna_sel: Raw antenna_sel byte from header.

    Returns:
        Tuple of 3 physical antenna indices (0-based).
    """
    perm0 = (antenna_sel >> (0 * ANTENNA_PERM_SHIFT)) & ANTENNA_PERM_MASK
    perm1 = (antenna_sel >> (1 * ANTENNA_PERM_SHIFT)) & ANTENNA_PERM_MASK
    perm2 = (antenna_sel >> (2 * ANTENNA_PERM_SHIFT)) & ANTENNA_PERM_MASK
    return perm0, perm1, perm2


def _apply_antenna_permutation(
    csi: np.ndarray,
    perm: Tuple[int, int, int],
    nrx: int,
) -> np.ndarray:
    """
    Apply antenna permutation to CSI matrix.

    Reorders the Rx dimension so that logical antenna indices
    correspond to physical antenna positions.

    Args:
        csi: Complex CSI array of shape (30, nrx, ntx).
        perm: Physical antenna indices from _get_antenna_permutation.
        nrx: Number of receive antennas.

    Returns:
        Permuted CSI array.
    """
    result = np.zeros_like(csi)
    for logical_idx in range(nrx):
        physical_idx = perm[logical_idx]
        if physical_idx < nrx:
            result[:, physical_idx, :] = csi[:, logical_idx, :]
        else:
            result[:, logical_idx, :] = csi[:, logical_idx, :]
    return result


def _scale_csi(
    csi: np.ndarray,
    rssi_a: int,
    rssi_b: int,
    rssi_c: int,
    noise: int,
    agc: int,
    nrx: int,
    ntx: int,
) -> np.ndarray:
    """
    Apply CSI amplitude scaling from Halperin et al.

    Normalizes CSI amplitudes using the relationship between RSSI
    (measured signal power) and CSI magnitude, accounting for AGC,
    noise floor, and MIMO configuration.

    Reference: get_scaled_csi.m from CSI Tool supplementary.

    Args:
        csi: Complex CSI array of shape (30, nrx, ntx).
        rssi_a, rssi_b, rssi_c: Per-antenna RSSI values.
        noise: Noise floor estimate from hardware.
        agc: AGC gain setting.
        nrx: Number of receive antennas.
        ntx: Number of transmit antennas.

    Returns:
        Scaled complex CSI array (same shape).
    """
    # Total RSS in linear (milliwatts)
    rssi_mag = 0.0
    for rssi_val in [rssi_a, rssi_b, rssi_c]:
        if rssi_val != 0:
            rssi_mag += 10.0 ** (rssi_val / 10.0)

    if rssi_mag == 0.0:
        return csi

    # RSS in dB, corrected for hardware offset and AGC
    rss_db = 10.0 * np.log10(rssi_mag) - INTEL_RSSI_OFFSET_DB - agc

    # RSS in linear power
    rss_power = 10.0 ** (rss_db / 10.0)

    # CSI power (mean across all entries)
    csi_sq = np.abs(csi) ** 2
    csi_pwr = np.sum(csi_sq)
    if csi_pwr == 0.0:
        return csi

    # Scale factor: ratio of measured power to CSI power
    scale = rss_power / (csi_pwr / INTEL_SUBCARRIER_GROUPS)

    # Noise power
    noise_val = noise if noise != -127 else INTEL_FALLBACK_NOISE_FLOOR
    thermal_noise_pwr = 10.0 ** (noise_val / 10.0)

    # Quantization error power
    quant_error_pwr = scale * nrx * ntx

    # Total noise
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    if total_noise_pwr <= 0.0:
        return csi

    return csi * np.sqrt(scale / total_noise_pwr)


class Intel5300Extractor(CSIExtractor):
    """
    CSI extractor for Intel WiFi Link 5300.

    Implements the binary format defined by the Linux 802.11n CSI Tool
    (read_bfee.c). Features:
    - Bit-packed CSI extraction with 3-bit per-group padding
    - Antenna permutation from antenna_sel byte
    - CSI amplitude scaling (AGC, RSSI, noise correction)
    - SISO link selection from MIMO matrix

    References:
        - read_bfee.c in the CSI tool source
        - get_scaled_csi.m for amplitude normalization

    Usage:
        extractor = Intel5300Extractor()
        packet = extractor.parse(raw_bytes)
    """

    def __init__(self, antenna_config: str = "3x3"):
        """
        Initialize Intel 5300 extractor.

        Args:
            antenna_config: MIMO configuration ("1x1", "2x2", "3x3")
        """
        self._antenna_config = antenna_config
        n_tx, n_rx = self._parse_antenna_config(antenna_config)
        self._n_tx = n_tx
        self._n_rx = n_rx

    def _parse_antenna_config(self, config: str) -> Tuple[int, int]:
        """Parse antenna configuration string."""
        parts = config.lower().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 3, 3  # Default to 3x3

    @cached_property
    def capabilities(self) -> HardwareCapabilities:
        """Intel 5300 hardware capabilities."""
        return HardwareCapabilities(
            hardware_type="intel_5300",
            supported_bandwidths=[20],
            supported_bands=["2.4GHz", "5GHz"],
            max_tx_antennas=3,
            max_rx_antennas=3,
            subcarrier_configs={
                20: SubcarrierConfig(
                    num_subcarriers=INTEL_SUBCARRIER_GROUPS,
                    pilot_indices=INTEL_PILOT_GROUPS,
                    dc_index=15,
                ),
            },
            supports_mimo=True,
            has_reliable_agc=True,
            typical_noise_floor_dbm=float(INTEL_DEFAULT_NOISE_FLOOR),
        )

    def parse(
        self,
        data: bytes,
        rx_mac: Optional[str] = None
    ) -> Optional[CSIPacket]:
        """
        Parse Intel 5300 CSI data.

        Implements the binary format from read_bfee.c with proper
        bit-level extraction, antenna permutation, and CSI scaling.

        Args:
            data: Raw bytes from Intel 5300
            rx_mac: Receiver MAC address (optional)

        Returns:
            CSIPacket or None if parsing fails
        """
        if len(data) < INTEL_HEADER_SIZE:
            return None

        try:
            # Unpack header (20 bytes)
            (
                timestamp_low,
                bfee_count,
                reserved1,
                nrx,
                ntx,
                rssi_a,
                rssi_b,
                rssi_c,
                noise,
                agc,
                antenna_sel,
                length,
                fake_rate_n_flags
            ) = struct.unpack('<IH2sBBBBBbBBHH', data[:INTEL_HEADER_SIZE])

            if nrx == 0 or ntx == 0:
                logger.debug(
                    f"Intel5300 packet with nrx={nrx}, ntx={ntx}. Ignoring."
                )
                return None

            # Validate payload length using read_bfee.c formula:
            # calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8
            calc_len = (
                INTEL_SUBCARRIER_GROUPS
                * (nrx * ntx * INTEL_BITS_PER_COMPLEX
                   + INTEL_BITS_PER_GROUP_PADDING)
                + 7
            ) // 8

            if length != calc_len:
                logger.warning(
                    f"Intel5300 payload length mismatch: "
                    f"got {length}, expected {calc_len} "
                    f"(nrx={nrx}, ntx={ntx})"
                )
                return None

            if len(data) < INTEL_HEADER_SIZE + length:
                logger.warning(
                    f"Intel5300 payload too short: "
                    f"need {INTEL_HEADER_SIZE + length}, got {len(data)}"
                )
                return None

            payload = data[INTEL_HEADER_SIZE:INTEL_HEADER_SIZE + length]

            # Extract CSI using bit-packed format (read_bfee.c algorithm)
            csi_complex = _extract_csi_bitpacked(payload, nrx, ntx)

            # Apply antenna permutation
            perm = _get_antenna_permutation(antenna_sel)
            csi_complex = _apply_antenna_permutation(
                csi_complex, perm, nrx
            )

            # Apply CSI scaling (AGC/RSSI/noise normalization)
            csi_complex = _scale_csi(
                csi_complex, rssi_a, rssi_b, rssi_c,
                noise, agc, nrx, ntx,
            )

            # Select the best SISO link from the MIMO matrix
            rssi_values = [rssi_a, rssi_b, rssi_c]
            best_rx_idx = 0
            max_rssi_raw = -1
            for i in range(min(nrx, 3)):
                if rssi_values[i] > max_rssi_raw:
                    max_rssi_raw = rssi_values[i]
                    best_rx_idx = i

            csi_rx = csi_complex[:, best_rx_idx, :]
            energy_tx = np.sum(np.abs(csi_rx), axis=0)
            best_tx_idx = int(np.argmax(energy_tx)) if ntx > 0 else 0

            csi_siso = csi_rx[:, best_tx_idx]

            # Convert to IQ format: (n_subcarriers, 2)
            iq_data = np.stack(
                [csi_siso.real, csi_siso.imag], axis=1
            ).astype(np.float32)

            # Calculate RSSI in dBm
            if max_rssi_raw == 0 and nrx > 0:
                rssi_dbm = -100
            else:
                rssi_dbm = int(max_rssi_raw) - INTEL_RSSI_OFFSET_DB - int(agc)

            selected_antenna = (best_tx_idx, best_rx_idx)

            return CSIPacket(
                tx_mac="00:00:00:00:00:00",
                rx_mac=rx_mac or "00:00:00:00:00:00",
                timestamp_us=timestamp_low,
                rssi=rssi_dbm,
                noise_floor=int(noise) if noise != -127 else INTEL_FALLBACK_NOISE_FLOOR,
                channel=0,
                bandwidth_mhz=20,
                n_subcarriers=INTEL_SUBCARRIER_GROUPS,
                iq_data=iq_data,
                rx_state=agc,
                antenna_sel=selected_antenna,
                frequency_band="2.4GHz",
                n_tx_antennas=ntx,
                n_rx_antennas=nrx,
                hardware_type="intel_5300",
            )

        except Exception as e:
            logger.error(f"Failed to parse Intel 5300 packet: {e}")
            return None

    def get_default_profile(self) -> dict:
        """Return Intel 5300 default calibration profile."""
        return {
            "description": "Intel WiFi Link 5300 defaults",
            "typical_noise_floor_dbm": INTEL_DEFAULT_NOISE_FLOOR,
            "typical_amplitude_sigma": 3.0,
            "typical_phase_sigma_rad": 0.9,
            "typical_snr_db": 30,
            "subcarriers": INTEL_SUBCARRIER_GROUPS,
            "bandwidth_mhz": 20,
            "supports_mimo": True,
        }
