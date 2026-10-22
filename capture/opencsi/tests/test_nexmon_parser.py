import struct

import numpy as np
import pytest

from opencsi.hardware.nexmon import (
    BYTES_PER_SUBCARRIER,
    CHANSPEC_BAND_5G,
    CHANSPEC_BAND_SHIFT,
    CHANSPEC_BW_MAP,
    CHANSPEC_BW_SHIFT,
    CHIP_VER_MAP,
    FLOAT_AUTOSCALE_NBITS,
    FLOAT_CHIPS,
    INT16_CHIPS,
    NEXMON_HEADER_SIZE,
    NEXMON_MAGIC,
    NEXMON_SUBCARRIERS_20MHZ,
    NEXMON_SUBCARRIERS_40MHZ,
    NEXMON_SUBCARRIERS_80MHZ,
    SUPPORTED_CHIPS,
    NexmonExtractor,
    _decode_chanspec,
    _decode_core_spatial,
    _unpack_float_csi,
)
from opencsi.hardware.capabilities import (
    NEXMON_HT20_CONFIG,
    NEXMON_HT40_CONFIG,
    DEFAULT_VHT80_CONFIG,
    create_nexmon_capabilities,
)


# =============================================================================
# Helpers
# =============================================================================

def _build_chanspec(channel: int, bw_code: int, band_5g: bool = False) -> int:
    """Build a Broadcom chanspec value from components."""
    cs = channel & 0xFF
    cs |= (bw_code & 0x03) << CHANSPEC_BW_SHIFT
    if band_5g:
        cs |= CHANSPEC_BAND_5G << CHANSPEC_BAND_SHIFT
    return cs


def create_nexmon_packet(
    rssi: int = -50,
    channel: int = 36,
    subcarriers: int = 64,
    src_mac: bytes = b'\x00\x11\x22\x33\x44\x55',
    core_spatial: int = 0,
    chanspec: int | None = None,
    fc: int = 0,
    chip_ver: int = 0,
) -> bytes:
    """Build a synthetic Nexmon UDP packet with int16 I/Q data."""
    if chanspec is None:
        # Build proper chanspec: infer bw_code from subcarrier count
        if subcarriers <= NEXMON_SUBCARRIERS_20MHZ:
            bw_code = 0  # 20 MHz
        elif subcarriers <= NEXMON_SUBCARRIERS_40MHZ:
            bw_code = 1  # 40 MHz
        else:
            bw_code = 2  # 80 MHz
        band_5g = channel > 14
        chanspec = _build_chanspec(channel, bw_code, band_5g)

    header = struct.pack(
        "<HbB6sHHHH",
        0x1111,          # Magic
        rssi,            # RSSI (signed byte)
        fc,              # Frame Control
        src_mac,         # Source MAC
        100,             # Sequence
        core_spatial,    # Core/Spatial
        chanspec,        # Chanspec
        chip_ver,        # Chip Version
    )

    # Generate synthetic I/Q data: I=index, Q=index+1
    iq_data = np.zeros(subcarriers * 2, dtype=np.int16)
    for i in range(subcarriers):
        iq_data[2 * i] = i
        iq_data[2 * i + 1] = i + 1

    return header + iq_data.tobytes()


def _pack_float_subcarrier(
    i_mantissa: int,
    q_mantissa: int,
    exponent: int,
    nman: int,
    nexp: int,
) -> int:
    """
    Pack one subcarrier as a uint32 word (Broadcom packed-float format).

    Bit layout (LSB to MSB):
        [exponent (nexp)][imag_mantissa (nman-1)][imag_sign (1)]
        [real_mantissa (nman-1)][real_sign (1)][unused]

    Exponent is stored as offset binary (bias = 2^(nexp-1)):
        signed >= 0: stored as-is
        signed < 0:  stored as signed + 2 * bias

    Args:
        i_mantissa: Signed real mantissa.
        q_mantissa: Signed imaginary mantissa.
        exponent: Signed exponent value.
        nman: Mantissa bits per component (including sign).
        nexp: Exponent bits (shared).

    Returns:
        uint32 packed word.
    """
    e_p = 1 << (nexp - 1)
    max_man = (1 << (nman - 1)) - 1

    # Signs
    sign_r = 1 if i_mantissa < 0 else 0
    sign_i = 1 if q_mantissa < 0 else 0
    abs_i = min(abs(i_mantissa), max_man)
    abs_q = min(abs(q_mantissa), max_man)

    # Encode signed exponent to offset binary
    if exponent >= 0:
        e_unsigned = exponent
    else:
        e_unsigned = exponent + (e_p << 1)
    e_unsigned &= (1 << nexp) - 1

    # Build uint32 word
    word = e_unsigned
    word |= abs_q << nexp
    word |= sign_i << (nexp + nman - 1)
    word |= abs_i << (nexp + nman)
    word |= sign_r << (nexp + 2 * nman - 1)

    return word


def create_float_payload(
    chip: str,
    subcarriers: int,
    iq_values: list[tuple[int, int, int]],
) -> bytes:
    """
    Build packed-float CSI payload (uint32 LE words).

    Args:
        chip: Chip name (e.g., "bcm4358").
        subcarriers: Number of subcarriers.
        iq_values: List of (i_mantissa, q_mantissa, exponent) per subcarrier.

    Returns:
        Raw payload bytes (subcarriers * 4).
    """
    nman, nexp = FLOAT_CHIPS[chip]

    if len(iq_values) < subcarriers:
        # Pad with zeros
        iq_values = list(iq_values) + [(0, 0, 0)] * (subcarriers - len(iq_values))

    words = np.array(
        [_pack_float_subcarrier(i, q, e, nman, nexp)
         for i, q, e in iq_values[:subcarriers]],
        dtype='<u4',
    )
    return words.tobytes()


def create_float_nexmon_packet(
    chip: str,
    channel: int = 36,
    subcarriers: int = 64,
    iq_values: list[tuple[int, int, int]] | None = None,
    rssi: int = -50,
) -> bytes:
    """
    Build a synthetic Nexmon packet with packed-float I/Q data.

    Args:
        chip: Chip name (e.g., "bcm4358", "bcm4366c0").
        channel: WiFi channel number.
        subcarriers: Number of subcarriers.
        iq_values: List of (i_mantissa, q_mantissa, exponent) per subcarrier.
                   If None, generates a simple test pattern.
        rssi: RSSI value.

    Returns:
        Complete packet bytes (header + float CSI payload).
    """
    if iq_values is None:
        # Default: I=1, Q=2, exponent=0
        iq_values = [(1, 2, 0)] * subcarriers

    payload = create_float_payload(chip, subcarriers, iq_values)

    # Build chanspec
    bw_code = 0 if subcarriers <= 64 else (1 if subcarriers <= 128 else 2)
    band_5g = channel > 14
    chanspec = _build_chanspec(channel, bw_code, band_5g)

    header = struct.pack(
        "<HbB6sHHHH",
        0x1111,
        rssi,
        0,
        b'\x00\x11\x22\x33\x44\x55',
        100,
        0,
        chanspec,
        0,
    )

    return header + payload


def compute_expected_float_output(
    iq_values: list[tuple[int, int, int]],
    nman: int,
    nexp: int,
) -> np.ndarray:
    """
    Compute expected output of _unpack_float_csi for verification.

    Replicates the autoscaling algorithm to produce expected I/Q values.

    Returns:
        ndarray of shape (n, 2) with float32 I/Q values.
    """
    e_p = 1 << (nexp - 1)
    max_man = (1 << (nman - 1)) - 1

    n = len(iq_values)

    # Extract mantissa magnitudes and signed exponents
    vi_arr = np.array([min(abs(i), max_man) for i, q, e in iq_values], dtype=np.int64)
    vq_arr = np.array([min(abs(q), max_man) for i, q, e in iq_values], dtype=np.int64)
    sign_r = np.array([i < 0 for i, q, e in iq_values])
    sign_i = np.array([q < 0 for i, q, e in iq_values])
    e_arr = np.array([e for i, q, e in iq_values], dtype=np.int32)

    # Autoscaling: highest set bit in (vi | vq)
    x = vi_arr | vq_arr
    highest_bit = np.zeros(n, dtype=np.int32)
    for bit_pos in range(nman - 1):
        has_bit = (x >> bit_pos) & 1
        highest_bit = np.where(has_bit, bit_pos + 1, highest_bit)

    effective_e = e_arr + highest_bit
    nonzero = x > 0
    if np.any(nonzero):
        maxbit = int(np.max(effective_e[nonzero]))
    else:
        maxbit = -e_p

    shft = FLOAT_AUTOSCALE_NBITS - maxbit
    e_shifted = e_arr + shft

    # Apply exponent
    scale = np.power(2.0, e_shifted.astype(np.float64))
    e_zero = -nman
    scale[e_shifted < e_zero] = 0.0

    out_r = vi_arr.astype(np.float64) * scale
    out_i = vq_arr.astype(np.float64) * scale
    out_r[sign_r] *= -1.0
    out_i[sign_i] *= -1.0

    return np.stack([out_r, out_i], axis=1).astype(np.float32)


# =============================================================================
# Test: chanspec decoding
# =============================================================================

class TestChanspecDecoding:
    def test_20mhz_2_4ghz(self):
        chanspec = _build_chanspec(6, 0, band_5g=False)
        ch, bw, band = _decode_chanspec(chanspec)
        assert ch == 6
        assert bw == 20
        assert band == "2.4GHz"

    def test_20mhz_5ghz(self):
        chanspec = _build_chanspec(36, 0, band_5g=True)
        ch, bw, band = _decode_chanspec(chanspec)
        assert ch == 36
        assert bw == 20
        assert band == "5GHz"

    def test_40mhz_5ghz(self):
        chanspec = _build_chanspec(36, 1, band_5g=True)
        ch, bw, band = _decode_chanspec(chanspec)
        assert ch == 36
        assert bw == 40
        assert band == "5GHz"

    def test_80mhz_5ghz(self):
        chanspec = _build_chanspec(42, 2, band_5g=True)
        ch, bw, band = _decode_chanspec(chanspec)
        assert ch == 42
        assert bw == 80
        assert band == "5GHz"

    def test_160mhz(self):
        chanspec = _build_chanspec(50, 3, band_5g=True)
        ch, bw, band = _decode_chanspec(chanspec)
        assert ch == 50
        assert bw == 160
        assert band == "5GHz"

    def test_invalid_bw_returns_zero(self):
        # Raw chanspec with bits 9:8 = 0b11 is valid (160), but
        # a completely zeroed chanspec should return channel=0, bw=20
        ch, bw, band = _decode_chanspec(0x0000)
        assert ch == 0
        assert bw == 20  # bw_code=0 maps to 20

    def test_various_channels(self):
        for channel in [1, 6, 11, 36, 40, 44, 48, 149, 153, 157, 161]:
            chanspec = _build_chanspec(channel, 0, band_5g=(channel > 14))
            ch, bw, band = _decode_chanspec(chanspec)
            assert ch == channel


# =============================================================================
# Test: core/spatial stream decoding
# =============================================================================

class TestCoreSpatialDecoding:
    def test_core_0_ss_0(self):
        core, ss = _decode_core_spatial(0x0000)
        assert core == 0
        assert ss == 0

    def test_core_1_ss_0(self):
        core, ss = _decode_core_spatial(0x0001)
        assert core == 1
        assert ss == 0

    def test_core_0_ss_1(self):
        # spatial_stream in bits 3-5, so ss=1 means bit 3 set = 0x08
        core, ss = _decode_core_spatial(0x0008)
        assert core == 0
        assert ss == 1

    def test_core_2_ss_3(self):
        # core=2 (0x02), ss=3 (0x03 << 3 = 0x18)
        css = 0x02 | (0x03 << 3)
        core, ss = _decode_core_spatial(css)
        assert core == 2
        assert ss == 3

    def test_max_values(self):
        # core=7 (0x07), ss=7 (0x07 << 3 = 0x38)
        css = 0x07 | (0x07 << 3)
        core, ss = _decode_core_spatial(css)
        assert core == 7
        assert ss == 7


# =============================================================================
# Test: int16 parsing (existing + extended)
# =============================================================================

class TestInt16Parsing:
    def test_parsing_20mhz(self):
        extractor = NexmonExtractor()
        packet = create_nexmon_packet(subcarriers=64, channel=36, rssi=-60)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.n_subcarriers == 64
        assert parsed.bandwidth_mhz == 20
        assert parsed.rssi == -60
        assert parsed.channel == 36
        assert parsed.tx_mac == "00:11:22:33:44:55"
        assert parsed.hardware_type.startswith("nexmon_")

        # Verify IQ data
        expected_iq = np.zeros((64, 2), dtype=np.int16)
        for i in range(64):
            expected_iq[i, 0] = i
            expected_iq[i, 1] = i + 1
        np.testing.assert_array_equal(parsed.iq_data, expected_iq)

    def test_parsing_40mhz(self):
        extractor = NexmonExtractor()
        packet = create_nexmon_packet(subcarriers=128, channel=36)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.n_subcarriers == 128
        assert parsed.bandwidth_mhz == 40

    def test_parsing_80mhz(self):
        extractor = NexmonExtractor()
        packet = create_nexmon_packet(subcarriers=256, channel=36)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.n_subcarriers == 256
        assert parsed.bandwidth_mhz == 80

    def test_invalid_magic(self):
        extractor = NexmonExtractor()
        packet = create_nexmon_packet()
        bad_packet = b'\x00\x00' + packet[2:]
        parsed = extractor.parse(bad_packet)
        assert parsed is None

    def test_short_packet(self):
        extractor = NexmonExtractor()
        packet = b'\x11\x11'
        parsed = extractor.parse(packet)
        assert parsed is None

    def test_antenna_sel_populated(self):
        extractor = NexmonExtractor()
        # core=1, ss=2 -> css = 0x01 | (0x02 << 3) = 0x11 = 17
        css = 0x01 | (0x02 << 3)
        packet = create_nexmon_packet(core_spatial=css, channel=36)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.antenna_sel == (1, 2)

    def test_rx_state_is_frame_control(self):
        extractor = NexmonExtractor()
        packet = create_nexmon_packet(fc=0x80, channel=36)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.rx_state == 0x80

    def test_chanspec_bandwidth_used(self):
        """Chanspec BW should be used as primary bandwidth source."""
        extractor = NexmonExtractor()
        # 40 MHz chanspec but only 64 subcarriers of data
        chanspec = _build_chanspec(36, 1, band_5g=True)  # 40 MHz
        packet = create_nexmon_packet(
            subcarriers=64, chanspec=chanspec, channel=36
        )
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.n_subcarriers == 64

    def test_frequency_band_from_chanspec(self):
        extractor = NexmonExtractor()
        chanspec = _build_chanspec(36, 0, band_5g=True)
        packet = create_nexmon_packet(chanspec=chanspec, channel=36)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.frequency_band == "5GHz"

    def test_frequency_band_2_4ghz(self):
        extractor = NexmonExtractor()
        chanspec = _build_chanspec(6, 0, band_5g=False)
        packet = create_nexmon_packet(
            chanspec=chanspec, channel=6, subcarriers=64
        )
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.frequency_band == "2.4GHz"

    def test_all_int16_chips(self):
        for chip in INT16_CHIPS:
            extractor = NexmonExtractor(chip=chip)
            packet = create_nexmon_packet(subcarriers=64, channel=36)
            parsed = extractor.parse(packet)
            assert parsed is not None, f"Failed for chip {chip}"
            assert parsed.hardware_type == f"nexmon_{chip}"


# =============================================================================
# Test: packed-float CSI parsing (uint32 format with shared exponent)
# =============================================================================

class TestFloatCsiParsing:
    def test_bcm4358_iq_ratio_preserved(self):
        """BCM4358: I/Q ratio should be preserved after autoscaling."""
        extractor = NexmonExtractor(chip="bcm4358")
        # I=3, Q=2, exponent=0 (shared) → ratio 3:2
        iq_values = [(3, 2, 0)] * 4
        packet = create_float_nexmon_packet(
            "bcm4358", subcarriers=4, iq_values=iq_values
        )
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.n_subcarriers == 4
        # Ratio must be preserved (autoscaling is uniform)
        for i in range(4):
            ratio = parsed.iq_data[i, 0] / parsed.iq_data[i, 1]
            assert ratio == pytest.approx(3.0 / 2.0, rel=1e-5)

    def test_bcm4358_exact_values(self):
        """BCM4358: verify exact autoscaled output."""
        nman, nexp = FLOAT_CHIPS["bcm4358"]
        iq_values = [(3, 2, 0)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4358", 1, iq_values)
        result = _unpack_float_csi(payload, 1, nman, nexp)

        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_bcm4358_with_exponent(self):
        """BCM4358: exponent should scale relative to other subcarriers."""
        nman, nexp = FLOAT_CHIPS["bcm4358"]
        # Two subcarriers: same mantissa, different exponent
        iq_values = [(5, 1, 0), (5, 1, 3)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4358", 2, iq_values)
        result = _unpack_float_csi(payload, 2, nman, nexp)

        np.testing.assert_array_almost_equal(result, expected, decimal=1)
        # Sub 1 should be 2^3 = 8x larger than sub 0
        assert abs(result[1, 0]) == pytest.approx(abs(result[0, 0]) * 8.0, rel=1e-3)

    def test_bcm4358_negative_values(self):
        """BCM4358: sign bits should produce negative output."""
        nman, nexp = FLOAT_CHIPS["bcm4358"]
        iq_values = [(-5, 3, 0)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4358", 1, iq_values)
        result = _unpack_float_csi(payload, 1, nman, nexp)

        assert result[0, 0] < 0  # real is negative
        assert result[0, 1] > 0  # imag is positive
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_bcm4358_negative_exponent(self):
        """BCM4358: negative exponent should produce smaller values."""
        nman, nexp = FLOAT_CHIPS["bcm4358"]
        # exp=2 vs exp=-2 → 2^4 = 16x ratio
        iq_values = [(5, 1, 2), (5, 1, -2)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4358", 2, iq_values)
        result = _unpack_float_csi(payload, 2, nman, nexp)

        np.testing.assert_array_almost_equal(result, expected, decimal=1)
        # Sub 0 should be 2^4 = 16x larger than sub 1
        assert abs(result[0, 0]) == pytest.approx(abs(result[1, 0]) * 16.0, rel=1e-3)

    def test_bcm4366c0_exact_values(self):
        """BCM4366c0: verify exact autoscaled output."""
        nman, nexp = FLOAT_CHIPS["bcm4366c0"]
        iq_values = [(10, -5, 0)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4366c0", 1, iq_values)
        result = _unpack_float_csi(payload, 1, nman, nexp)

        np.testing.assert_array_almost_equal(result, expected, decimal=1)
        assert result[0, 0] > 0
        assert result[0, 1] < 0

    def test_bcm4366c0_with_exponent(self):
        """BCM4366c0: verify exponent scaling."""
        nman, nexp = FLOAT_CHIPS["bcm4366c0"]
        iq_values = [(7, 3, 0), (7, 3, 2)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4366c0", 2, iq_values)
        result = _unpack_float_csi(payload, 2, nman, nexp)

        np.testing.assert_array_almost_equal(result, expected, decimal=1)
        # Sub 1 should be 4x larger
        assert abs(result[1, 0]) == pytest.approx(abs(result[0, 0]) * 4.0, rel=1e-3)

    def test_zero_mantissa_produces_zero(self):
        """Zero mantissa should produce zero regardless of exponent."""
        nman, nexp = FLOAT_CHIPS["bcm4358"]
        iq_values = [(0, 0, 5)]
        payload = create_float_payload("bcm4358", 1, iq_values)
        result = _unpack_float_csi(payload, 1, nman, nexp)

        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.0)

    def test_float_subcarrier_count_matches_payload_size(self):
        """Float chips use 4 bytes/subcarrier (same as int16)."""
        extractor = NexmonExtractor(chip="bcm4358")
        # 64 subcarriers → 256 bytes payload
        packet = create_float_nexmon_packet("bcm4358", subcarriers=64)
        parsed = extractor.parse(packet)

        assert parsed is not None
        assert parsed.n_subcarriers == 64

    def test_float_packet_too_short(self):
        """Should return None for truncated float packets."""
        extractor = NexmonExtractor(chip="bcm4358")
        # Build header + only 10 bytes of payload (not enough for 64 subs)
        header = struct.pack(
            "<HbB6sHHHH",
            0x1111, -50, 0, b'\x00' * 6, 100, 0,
            _build_chanspec(36, 0, True), 0,
        )
        parsed = extractor.parse(header + b'\x00' * 10)

        # Should parse with only 2 subcarriers (10 // 4 = 2)
        assert parsed is not None
        assert parsed.n_subcarriers == 2


# =============================================================================
# Test: chip validation
# =============================================================================

class TestChipValidation:
    def test_unsupported_chip_raises(self):
        with pytest.raises(ValueError, match="Unsupported chip"):
            NexmonExtractor(chip="bcm9999")

    def test_supported_chips_all_construct(self):
        for chip in SUPPORTED_CHIPS:
            ext = NexmonExtractor(chip=chip)
            assert ext._chip == chip

    def test_case_insensitive(self):
        ext = NexmonExtractor(chip="BCM43455")
        assert ext._chip == "bcm43455"


# =============================================================================
# Test: guard indices in SubcarrierConfig
# =============================================================================

class TestGuardIndices:
    def test_nexmon_ht20_guard_indices(self):
        guards = NEXMON_HT20_CONFIG.guard_indices
        assert len(guards) == 9
        assert 0 in guards
        assert 63 in guards
        assert 32 not in guards

    def test_nexmon_ht40_guard_indices(self):
        guards = NEXMON_HT40_CONFIG.guard_indices
        assert len(guards) == 11
        assert 0 in guards
        assert 127 in guards
        assert 64 not in guards

    def test_vht80_guard_indices(self):
        guards = DEFAULT_VHT80_CONFIG.guard_indices
        assert len(guards) == 11
        assert 0 in guards
        assert 255 in guards
        assert 128 not in guards

    def test_guard_indices_sorted(self):
        """Guard indices should be in ascending order."""
        for config in [NEXMON_HT20_CONFIG, NEXMON_HT40_CONFIG, DEFAULT_VHT80_CONFIG]:
            guards = config.guard_indices
            assert guards == tuple(sorted(guards))

    def test_guard_indices_within_range(self):
        """All guard indices should be valid subcarrier indices."""
        for config in [NEXMON_HT20_CONFIG, NEXMON_HT40_CONFIG, DEFAULT_VHT80_CONFIG]:
            for idx in config.guard_indices:
                assert 0 <= idx < config.num_subcarriers

    def test_num_guards_property(self):
        assert NEXMON_HT20_CONFIG.num_guards == 9
        assert NEXMON_HT40_CONFIG.num_guards == 11
        assert DEFAULT_VHT80_CONFIG.num_guards == 11

    def test_create_nexmon_capabilities_uses_configs(self):
        caps = create_nexmon_capabilities()
        config_20 = caps.get_subcarrier_config(20)
        assert len(config_20.guard_indices) > 0


# =============================================================================
# Test: _unpack_float_csi edge cases
# =============================================================================

class TestUnpackFloatEdgeCases:
    def test_insufficient_bytes_raises(self):
        """Should raise ValueError if payload is too short."""
        with pytest.raises(ValueError, match="Need .* bytes"):
            _unpack_float_csi(b'\x00', n_subcarriers=64, nman=9, nexp=5)

    def test_large_mantissa_values(self):
        """Max mantissa should be decoded correctly."""
        nman, nexp = 9, 5
        max_man = (1 << (nman - 1)) - 1  # 255
        iq_values = [(max_man, -max_man, 0)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4358", 1, iq_values)
        result = _unpack_float_csi(payload, 1, nman, nexp)

        assert result[0, 0] > 0
        assert result[0, 1] < 0
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_shared_exponent(self):
        """I and Q in the same subcarrier share the same exponent."""
        nman, nexp = 9, 5
        # I=10, Q=3, exp=4 → both scaled by 2^(4 + shft)
        iq_values = [(10, 3, 4)]
        expected = compute_expected_float_output(iq_values, nman, nexp)

        payload = create_float_payload("bcm4358", 1, iq_values)
        result = _unpack_float_csi(payload, 1, nman, nexp)

        np.testing.assert_array_almost_equal(result, expected, decimal=1)
        # Ratio should be exactly 10/3 since they share the exponent
        ratio = result[0, 0] / result[0, 1]
        assert ratio == pytest.approx(10.0 / 3.0, rel=1e-5)


# =============================================================================
# Test: default profile
# =============================================================================

class TestDefaultProfile:
    def test_profile_structure(self):
        ext = NexmonExtractor(chip="bcm43455")
        profile = ext.get_default_profile()
        assert "description" in profile
        assert "typical_noise_floor_dbm" in profile
        assert "BCM43455" in profile["description"]

    def test_float_chip_profile(self):
        ext = NexmonExtractor(chip="bcm4358")
        profile = ext.get_default_profile()
        assert "BCM4358" in profile["description"]


# =============================================================================
# Test: chip auto-detection from chip_ver header field
# =============================================================================

class TestChipAutoDetection:
    """Test chip="auto" mode that detects chip from packet header."""

    def test_auto_mode_constructs(self):
        """chip="auto" should not raise."""
        ext = NexmonExtractor(chip="auto")
        assert ext._auto_detect is True

    def test_auto_detects_bcm4339(self):
        """chip_ver=0x0001 → BCM4339 (int16)."""
        ext = NexmonExtractor(chip="auto")
        packet_data = create_nexmon_packet(chip_ver=0x0001)
        packet = ext.parse(packet_data)
        assert packet is not None
        assert packet.iq_data.shape == (64, 2)

    def test_auto_detects_bcm43455(self):
        """chip_ver=0x0065 → BCM43455 (int16)."""
        ext = NexmonExtractor(chip="auto")
        packet_data = create_nexmon_packet(chip_ver=0x0065)
        packet = ext.parse(packet_data)
        assert packet is not None

    def test_auto_detects_bcm43455_alternate(self):
        """chip_ver=0xa6dc → BCM43455 (int16, alternate ID)."""
        ext = NexmonExtractor(chip="auto")
        packet_data = create_nexmon_packet(chip_ver=0xa6dc)
        packet = ext.parse(packet_data)
        assert packet is not None

    def test_auto_detects_bcm4358(self):
        """chip_ver=0x0003 → BCM4358 (float 9+5)."""
        ext = NexmonExtractor(chip="auto")
        # BCM4358 uses float format, so we need float-encoded payload
        nman, nexp = FLOAT_CHIPS["bcm4358"]
        n_subcarriers = NEXMON_SUBCARRIERS_20MHZ

        # Build float payload with known values
        payload = bytearray()
        for _ in range(n_subcarriers):
            word = _pack_float_subcarrier(10, 20, 0, nman, nexp)
            payload.extend(struct.pack('<I', word))

        chanspec = _build_chanspec(36, 0, band_5g=True)  # 20 MHz, 5 GHz
        header = struct.pack(
            "<HbB6sHHHH",
            0x1111, -50, 0,
            b'\x00\x11\x22\x33\x44\x55',
            100, 0, chanspec, 0x0003,
        )

        packet = ext.parse(header + bytes(payload))
        assert packet is not None
        assert packet.iq_data.shape == (n_subcarriers, 2)

    def test_auto_detects_bcm4366c0(self):
        """chip_ver=0xe834 → BCM4366c0 (float 12+6)."""
        ext = NexmonExtractor(chip="auto")
        nman, nexp = FLOAT_CHIPS["bcm4366c0"]
        n_subcarriers = NEXMON_SUBCARRIERS_20MHZ

        payload = bytearray()
        for _ in range(n_subcarriers):
            word = _pack_float_subcarrier(10, 20, 0, nman, nexp)
            payload.extend(struct.pack('<I', word))

        chanspec = _build_chanspec(36, 0, band_5g=True)
        header = struct.pack(
            "<HbB6sHHHH",
            0x1111, -50, 0,
            b'\x00\x11\x22\x33\x44\x55',
            100, 0, chanspec, 0xe834,
        )

        packet = ext.parse(header + bytes(payload))
        assert packet is not None
        assert packet.iq_data.shape == (n_subcarriers, 2)

    def test_auto_unknown_chip_ver_defaults_to_int16(self):
        """Unknown chip_ver → falls back to bcm43455 (int16)."""
        ext = NexmonExtractor(chip="auto")
        # 0xBEEF is not in CHIP_VER_MAP
        packet_data = create_nexmon_packet(chip_ver=0xBEEF)
        packet = ext.parse(packet_data)
        assert packet is not None
        assert packet.iq_data.shape == (64, 2)

    def test_auto_capabilities_label(self):
        """Auto mode uses 'nexmon_auto' as hardware_type."""
        ext = NexmonExtractor(chip="auto")
        assert ext.capabilities.hardware_type == "nexmon_auto"

    def test_all_chip_ver_map_entries_resolve(self):
        """Every entry in CHIP_VER_MAP maps to a supported chip."""
        for chip_ver, chip_name in CHIP_VER_MAP.items():
            assert chip_name in SUPPORTED_CHIPS, (
                f"CHIP_VER_MAP[{chip_ver:#x}] = '{chip_name}' "
                f"not in SUPPORTED_CHIPS"
            )
