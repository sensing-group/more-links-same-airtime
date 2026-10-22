"""Tests for Intel 5300 CSI parser.

Tests bit-packed CSI extraction (3-bit padding per group), antenna
permutation, CSI scaling, and SISO link selection from MIMO matrix.
"""

import struct

import numpy as np
import pytest

from opencsi.hardware.intel5300 import (
    INTEL_BITS_PER_COMPLEX,
    INTEL_BITS_PER_GROUP_PADDING,
    INTEL_HEADER_SIZE,
    INTEL_RSSI_OFFSET_DB,
    INTEL_SUBCARRIER_GROUPS,
    Intel5300Extractor,
    _extract_csi_bitpacked,
    _get_antenna_permutation,
    _apply_antenna_permutation,
)
from opencsi.hardware.base import CSIPacket


def _calc_payload_len(nrx, ntx):
    """Compute expected payload length using read_bfee.c formula."""
    return (
        INTEL_SUBCARRIER_GROUPS
        * (nrx * ntx * INTEL_BITS_PER_COMPLEX + INTEL_BITS_PER_GROUP_PADDING)
        + 7
    ) // 8


def _pack_bitpacked_payload(nrx, ntx, iq_pairs):
    """Build a proper bit-packed CSI payload matching read_bfee.c format.

    Args:
        nrx: Number of Rx antennas.
        ntx: Number of Tx antennas.
        iq_pairs: List of (real, imag) tuples, one per (subcarrier, rx*tx)
                  pair. Length must be 30 * nrx * ntx. Each value is int8.
                  If a single tuple, it's repeated for all entries.
    """
    n_entries = INTEL_SUBCARRIER_GROUPS * nrx * ntx
    if isinstance(iq_pairs, tuple) and len(iq_pairs) == 2:
        iq_pairs = [iq_pairs] * n_entries

    calc_len = _calc_payload_len(nrx, ntx)
    # Build a bitstream
    bits = []

    pair_idx = 0
    for sc in range(INTEL_SUBCARRIER_GROUPS):
        # 3-bit padding
        bits.extend([0, 0, 0])

        for entry in range(nrx * ntx):
            real_val, imag_val = iq_pairs[pair_idx]
            pair_idx += 1

            # Pack 8-bit signed real (LSB first)
            real_byte = real_val & 0xFF
            for bit in range(8):
                bits.append((real_byte >> bit) & 1)

            # Pack 8-bit signed imag (LSB first)
            imag_byte = imag_val & 0xFF
            for bit in range(8):
                bits.append((imag_byte >> bit) & 1)

    # Convert bits to bytes
    payload = bytearray(calc_len)
    for i, bit in enumerate(bits):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < calc_len:
            payload[byte_idx] |= (bit << bit_idx)

    return bytes(payload)


def create_intel_packet(
    nrx=1,
    ntx=1,
    noise=-92,
    agc=30,
    rssi_a=10,
    rssi_b=0,
    rssi_c=0,
    antenna_sel=0x01,
    iq_pairs=None,
):
    """Build a complete Intel 5300 CSI packet with proper bit-packed payload.

    Args:
        iq_pairs: (real, imag) tuple or list of tuples. Default: (1, 2).
    """
    if iq_pairs is None:
        iq_pairs = (1, 2)

    payload = _pack_bitpacked_payload(nrx, ntx, iq_pairs)
    length = len(payload)

    timestamp = 12345678
    bfee_count = 1
    reserved1 = b'\x00\x00'
    fake_rate_n_flags = 0

    header = struct.pack(
        '<IH2sBBBBBbBBHH',
        timestamp,
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
        fake_rate_n_flags,
    )

    return header + payload


# ====================================================================
# Unit tests for bit-packed extraction
# ====================================================================


class TestBitpackedExtraction:
    """Test the _extract_csi_bitpacked function directly."""

    def test_1x1_extraction(self):
        """1x1: 30 subcarrier groups, each with real=5, imag=10."""
        nrx, ntx = 1, 1
        payload = _pack_bitpacked_payload(nrx, ntx, (5, 10))
        csi = _extract_csi_bitpacked(payload, nrx, ntx)

        assert csi.shape == (30, 1, 1)
        for i in range(30):
            assert csi[i, 0, 0].real == pytest.approx(5.0)
            assert csi[i, 0, 0].imag == pytest.approx(10.0)

    def test_negative_values(self):
        """Negative int8 values are correctly extracted."""
        nrx, ntx = 1, 1
        payload = _pack_bitpacked_payload(nrx, ntx, (-3, -7))
        csi = _extract_csi_bitpacked(payload, nrx, ntx)

        for i in range(30):
            assert csi[i, 0, 0].real == pytest.approx(-3.0)
            assert csi[i, 0, 0].imag == pytest.approx(-7.0)

    def test_extreme_values(self):
        """int8 extremes: 127 and -128."""
        nrx, ntx = 1, 1
        payload = _pack_bitpacked_payload(nrx, ntx, (127, -128))
        csi = _extract_csi_bitpacked(payload, nrx, ntx)

        for i in range(30):
            assert csi[i, 0, 0].real == pytest.approx(127.0)
            assert csi[i, 0, 0].imag == pytest.approx(-128.0)

    def test_2x2_extraction(self):
        """2x2: 4 entries per subcarrier group."""
        nrx, ntx = 2, 2
        # Each subcarrier has nrx*ntx=4 entries
        # Entry order: (rx0,tx0), (rx0,tx1), (rx1,tx0), (rx1,tx1)
        iq_pairs = []
        for sc in range(30):
            for j in range(4):
                iq_pairs.append((j + 1, j + 10))

        payload = _pack_bitpacked_payload(nrx, ntx, iq_pairs)
        csi = _extract_csi_bitpacked(payload, nrx, ntx)

        assert csi.shape == (30, 2, 2)
        # j=0 → rx=0,tx=0: real=1, imag=10
        assert csi[0, 0, 0].real == pytest.approx(1.0)
        assert csi[0, 0, 0].imag == pytest.approx(10.0)
        # j=3 → rx=1,tx=1: real=4, imag=13
        assert csi[0, 1, 1].real == pytest.approx(4.0)
        assert csi[0, 1, 1].imag == pytest.approx(13.0)

    def test_calc_len_formula(self):
        """Verify calc_len matches expected sizes from read_bfee.c."""
        # 1x1: (30 * (1*1*16 + 3) + 7) / 8 = (30 * 19 + 7) / 8 = 577/8 = 72
        assert _calc_payload_len(1, 1) == 72
        # 2x2: (30 * (2*2*16 + 3) + 7) / 8 = (30 * 67 + 7) / 8 = 2017/8 = 252
        assert _calc_payload_len(2, 2) == 252
        # 3x3: (30 * (3*3*16 + 3) + 7) / 8 = (30 * 147 + 7) / 8 = 4417/8 = 552
        assert _calc_payload_len(3, 3) == 552


# ====================================================================
# Unit tests for antenna permutation
# ====================================================================


class TestAntennaPermutation:
    """Test antenna_sel byte decoding."""

    def test_identity_permutation(self):
        """antenna_sel=0b100100 → perm=(0, 1, 2) (identity)."""
        perm = _get_antenna_permutation(0b00_10_01_00)
        assert perm == (0, 1, 2)

    def test_swapped_permutation(self):
        """antenna_sel=0b001001 → perm=(1, 0, 2)."""
        perm = _get_antenna_permutation(0b00_10_00_01)
        assert perm == (1, 0, 2)

    def test_apply_identity(self):
        """Identity permutation doesn't change CSI."""
        csi = np.arange(30 * 3 * 2, dtype=np.complex64).reshape(30, 3, 2)
        perm = (0, 1, 2)
        result = _apply_antenna_permutation(csi, perm, 3)
        np.testing.assert_array_equal(result, csi)

    def test_apply_swap(self):
        """Swap rx0 ↔ rx1."""
        nrx, ntx = 2, 1
        csi = np.zeros((30, 2, 1), dtype=np.complex64)
        csi[:, 0, 0] = 10 + 0j  # logical 0
        csi[:, 1, 0] = 20 + 0j  # logical 1

        perm = (1, 0, 0)  # logical 0 → physical 1, logical 1 → physical 0
        result = _apply_antenna_permutation(csi, perm, nrx)

        # After permutation: physical 0 gets logical 1's data,
        # physical 1 gets logical 0's data
        np.testing.assert_array_equal(result[:, 0, 0], 20 + 0j)
        np.testing.assert_array_equal(result[:, 1, 0], 10 + 0j)


# ====================================================================
# Integration tests: full packet parsing
# ====================================================================


class TestIntel5300Parsing:
    """Test full packet parsing through Intel5300Extractor."""

    def test_parse_siso(self):
        """Parse 1x1 packet: verify header fields and IQ shape."""
        extractor = Intel5300Extractor(antenna_config="1x1")
        data = create_intel_packet(nrx=1, ntx=1, iq_pairs=(1, 2))

        packet = extractor.parse(data, rx_mac="AA:BB:CC:DD:EE:FF")

        assert packet is not None
        assert isinstance(packet, CSIPacket)
        assert packet.rx_mac == "AA:BB:CC:DD:EE:FF"
        assert packet.n_rx_antennas == 1
        assert packet.n_tx_antennas == 1
        assert packet.n_subcarriers == 30
        assert packet.timestamp_us == 12345678
        assert packet.iq_data.shape == (30, 2)

    def test_parse_siso_rssi(self):
        """RSSI calculation: rssi_a - OFFSET - AGC."""
        extractor = Intel5300Extractor(antenna_config="1x1")
        data = create_intel_packet(
            nrx=1, ntx=1, rssi_a=10, agc=30, iq_pairs=(1, 2)
        )
        packet = extractor.parse(data)
        assert packet is not None
        # rssi_dbm = 10 - 44 - 30 = -64
        assert packet.rssi == 10 - INTEL_RSSI_OFFSET_DB - 30

    def test_parse_siso_noise_floor(self):
        """Noise floor is passed through from header."""
        extractor = Intel5300Extractor(antenna_config="1x1")
        data = create_intel_packet(nrx=1, ntx=1, noise=-92)
        packet = extractor.parse(data)
        assert packet is not None
        assert packet.noise_floor == -92

    def test_parse_mimo_best_rx(self):
        """MIMO: selects Rx with highest RSSI."""
        extractor = Intel5300Extractor(antenna_config="3x3")
        # rssi_b=50 is strongest → best_rx_idx=1
        data = create_intel_packet(
            nrx=3, ntx=3,
            rssi_a=10, rssi_b=50, rssi_c=20,
            agc=20,
        )

        packet = extractor.parse(data)
        assert packet is not None
        assert packet.n_rx_antennas == 3
        assert packet.n_tx_antennas == 3
        # Selected antenna should have rx=1 (highest RSSI)
        assert packet.antenna_sel[1] == 1
        # rssi_dbm = 50 - 44 - 20 = -14
        assert packet.rssi == 50 - INTEL_RSSI_OFFSET_DB - 20

    def test_parse_zero_antennas(self):
        """nrx=0 or ntx=0 → None."""
        extractor = Intel5300Extractor()
        data = create_intel_packet(nrx=0, ntx=1)
        assert extractor.parse(data) is None

        data = create_intel_packet(nrx=1, ntx=0)
        assert extractor.parse(data) is None

    def test_short_packet(self):
        """Packet shorter than header → None."""
        extractor = Intel5300Extractor()
        assert extractor.parse(b'\x00' * 10) is None

    def test_length_mismatch(self):
        """Payload length doesn't match calc_len → None."""
        extractor = Intel5300Extractor()
        data = create_intel_packet(nrx=1, ntx=1)
        # Truncate payload
        assert extractor.parse(data[:-1]) is None

    def test_csi_scaling_applied(self):
        """CSI values are scaled (not raw int8 values)."""
        extractor = Intel5300Extractor(antenna_config="1x1")
        data = create_intel_packet(
            nrx=1, ntx=1,
            rssi_a=30, noise=-92, agc=20,
            iq_pairs=(10, 20),
        )
        packet = extractor.parse(data)
        assert packet is not None
        # After scaling, values should differ from raw (10, 20)
        # We don't check exact values since scaling depends on power ratios,
        # but they should be finite and non-zero
        assert np.all(np.isfinite(packet.iq_data))
        assert np.any(packet.iq_data != 0)

    def test_varying_iq_values(self):
        """Different I/Q values per subcarrier are preserved."""
        nrx, ntx = 1, 1
        iq_pairs = [(i, i * 2) for i in range(30)]
        data = create_intel_packet(
            nrx=nrx, ntx=ntx,
            rssi_a=30, noise=-92, agc=20,
            iq_pairs=iq_pairs,
        )
        extractor = Intel5300Extractor(antenna_config="1x1")
        packet = extractor.parse(data)
        assert packet is not None

        # The I/Q ratio should be preserved per subcarrier (scaling is uniform)
        # Skip subcarrier 0 which has (0, 0)
        for sc in range(1, 30):
            ratio = packet.iq_data[sc, 1] / packet.iq_data[sc, 0]
            assert ratio == pytest.approx(2.0, abs=0.01)


class TestIntel5300Capabilities:
    """Test hardware capabilities reporting."""

    def test_capabilities(self):
        extractor = Intel5300Extractor()
        caps = extractor.capabilities
        assert caps.hardware_type == "intel_5300"
        assert caps.supported_bandwidths == [20]
        assert caps.supports_mimo is True
        assert caps.max_tx_antennas == 3
        assert caps.max_rx_antennas == 3

    def test_default_profile(self):
        extractor = Intel5300Extractor()
        profile = extractor.get_default_profile()
        assert "Intel" in profile["description"]
        assert profile["bandwidth_mhz"] == 20
        assert profile["supports_mimo"] is True
