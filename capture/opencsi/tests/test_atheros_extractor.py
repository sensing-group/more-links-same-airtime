"""Tests for Atheros CSI parser.

Tests the 25-byte header format (with rssi3 field) and 10-bit packed
CSI data with I/Q swap (Atheros stores [imaginary, real] pairs).
"""

import struct
import unittest

import numpy as np

from opencsi.hardware.atheros import (
    ATHEROS_HEADER_FORMAT,
    ATHEROS_HEADER_SIZE,
    AtherosExtractor,
)


def pack_10bit_values(values):
    """Pack a list of 10-bit unsigned values into bytes (4 values → 5 bytes)."""
    packed = bytearray()
    # Pad to multiple of 4
    padded = list(values)
    while len(padded) % 4 != 0:
        padded.append(0)

    for i in range(0, len(padded), 4):
        v0 = padded[i] & 0x3FF
        v1 = padded[i + 1] & 0x3FF
        v2 = padded[i + 2] & 0x3FF
        v3 = padded[i + 3] & 0x3FF

        b0 = v0 & 0xFF
        b1 = ((v0 >> 8) & 0x3) | ((v1 & 0x3F) << 2)
        b2 = ((v1 >> 6) & 0xF) | ((v2 & 0xF) << 4)
        b3 = ((v2 >> 4) & 0x3F) | ((v3 & 0x3) << 6)
        b4 = (v3 >> 2) & 0xFF

        packed.extend([b0, b1, b2, b3, b4])
    return bytes(packed)


def signed_to_10bit(val):
    """Convert signed value (-512..511) to 10-bit unsigned representation."""
    if val < 0:
        return 1024 + val
    return val


def build_atheros_packet(
    timestamp=1234567890,
    tx_channel=2437,
    err_info=0,
    noise_floor=95,
    rate=0,
    bandwidth=0,
    num_tones=56,
    nr=1,
    nc=1,
    rssi=30,
    rssi1=28,
    rssi2=25,
    rssi3=22,
    payload_len=100,
    csi_values=None,
):
    """Build a complete Atheros CSI packet with 25-byte header.

    Args:
        csi_values: List of signed 10-bit values. If None, generates sequential
                    pattern. Length must be num_tones * nr * nc * 2.
    """
    required_count = num_tones * nr * nc * 2

    if csi_values is None:
        csi_values = [(i % 1000) - 500 for i in range(required_count)]

    # Convert signed values to 10-bit unsigned representation
    unsigned_values = [signed_to_10bit(v) for v in csi_values]
    csi_payload = pack_10bit_values(unsigned_values)
    csi_len = len(csi_payload)

    # 25-byte header with rssi3 field
    header = struct.pack(
        ATHEROS_HEADER_FORMAT,
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
        payload_len,
    )
    assert len(header) == ATHEROS_HEADER_SIZE
    return header + csi_payload


class TestAtherosHeaderFormat(unittest.TestCase):
    """Test that the 25-byte header is parsed correctly."""

    def test_header_size_is_25(self):
        self.assertEqual(ATHEROS_HEADER_SIZE, 25)
        self.assertEqual(struct.calcsize(ATHEROS_HEADER_FORMAT), 25)

    def test_basic_parsing(self):
        """Parse a valid 1x1 packet and verify header fields."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(nr=1, nc=1))

        self.assertIsNotNone(packet)
        self.assertEqual(packet.timestamp_us, 1234567890)
        self.assertEqual(packet.rssi, 30)
        self.assertEqual(packet.noise_floor, 95)
        self.assertEqual(packet.n_subcarriers, 56)
        self.assertEqual(packet.n_rx_antennas, 1)
        self.assertEqual(packet.n_tx_antennas, 1)

    def test_channel_detection_2_4ghz_frequency(self):
        """tx_channel=2437 → channel 6, 2.4GHz."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(tx_channel=2437))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.channel, 6)
        self.assertEqual(packet.frequency_band, "2.4GHz")

    def test_channel_detection_5ghz_frequency(self):
        """tx_channel=5240 → channel 48, 5GHz."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(tx_channel=5240))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.channel, 48)
        self.assertEqual(packet.frequency_band, "5GHz")

    def test_channel_detection_channel_number(self):
        """tx_channel=6 → channel 6, 2.4GHz."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(tx_channel=6))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.channel, 6)
        self.assertEqual(packet.frequency_band, "2.4GHz")

    def test_bandwidth_20mhz(self):
        """bandwidth=0 → 20 MHz."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(bandwidth=0))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.bandwidth_mhz, 20)

    def test_bandwidth_40mhz(self):
        """bandwidth=1 → 40 MHz."""
        extractor = AtherosExtractor()
        packet = extractor.parse(
            build_atheros_packet(bandwidth=1, num_tones=114)
        )
        self.assertIsNotNone(packet)
        self.assertEqual(packet.bandwidth_mhz, 40)


class TestAtherosIQSwap(unittest.TestCase):
    """Test that I/Q values are correctly swapped.

    Atheros stores [imaginary, real] pairs. The parser should output
    [real, imaginary] in iq_data.
    """

    def test_iq_swap_simple(self):
        """Known I/Q values: verify swap from [imag, real] to [real, imag]."""
        # For a 1x1 packet, values are stored as:
        # tone_k: [imaginary, real] for stream (0, 0)
        #
        # After I/Q swap, iq_data[k, 0] = real, iq_data[k, 1] = imaginary.

        # Use simple known values: imag=100, real=200 for all tones
        num_tones = 56
        csi_values = []
        for _ in range(num_tones):
            csi_values.append(100)   # stored first → imaginary
            csi_values.append(200)   # stored second → real
        # After swap: iq_data[:, 0] = 200 (real), iq_data[:, 1] = 100 (imag)

        extractor = AtherosExtractor()
        packet = extractor.parse(
            build_atheros_packet(nr=1, nc=1, csi_values=csi_values)
        )
        self.assertIsNotNone(packet)
        self.assertEqual(packet.iq_data.shape, (56, 2))
        np.testing.assert_allclose(packet.iq_data[:, 0], 200.0)  # real
        np.testing.assert_allclose(packet.iq_data[:, 1], 100.0)  # imaginary

    def test_iq_swap_negative_values(self):
        """Negative 10-bit values are correctly sign-extended and swapped."""
        num_tones = 56
        csi_values = []
        for _ in range(num_tones):
            csi_values.append(-50)   # imaginary
            csi_values.append(-100)  # real
        # After swap: real=-100, imag=-50

        extractor = AtherosExtractor()
        packet = extractor.parse(
            build_atheros_packet(nr=1, nc=1, csi_values=csi_values)
        )
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(packet.iq_data[:, 0], -100.0)  # real
        np.testing.assert_allclose(packet.iq_data[:, 1], -50.0)   # imaginary


class TestAtherosMIMO(unittest.TestCase):
    """Test MIMO configurations."""

    def test_3x3_parsing(self):
        """3x3 MIMO: parser selects stream (0, 0)."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(nr=3, nc=3))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.iq_data.shape, (56, 2))
        self.assertEqual(packet.n_rx_antennas, 3)
        self.assertEqual(packet.n_tx_antennas, 3)

    def test_2x2_parsing(self):
        """2x2 MIMO."""
        extractor = AtherosExtractor()
        packet = extractor.parse(build_atheros_packet(nr=2, nc=2))
        self.assertIsNotNone(packet)
        self.assertEqual(packet.iq_data.shape, (56, 2))

    def test_3x3_selects_stream_0_0(self):
        """In 3x3, iq_data should be the (0,0) stream with I/Q swap."""
        num_tones = 56
        nr, nc = 3, 3
        csi_values = []
        for k in range(num_tones):
            for r in range(nr):
                for c in range(nc):
                    if r == 0 and c == 0:
                        csi_values.extend([10, 20])  # imag=10, real=20
                    else:
                        csi_values.extend([99, 88])  # other streams
        # Stream (0,0) after swap: real=20, imag=10
        extractor = AtherosExtractor()
        packet = extractor.parse(
            build_atheros_packet(nr=nr, nc=nc, csi_values=csi_values)
        )
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(packet.iq_data[:, 0], 20.0)  # real
        np.testing.assert_allclose(packet.iq_data[:, 1], 10.0)  # imaginary


class TestAtherosEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def test_too_short_packet(self):
        """Packet shorter than 25 bytes → None."""
        extractor = AtherosExtractor()
        self.assertIsNone(extractor.parse(b'\x00' * 24))

    def test_truncated_payload(self):
        """Header valid but CSI payload truncated → None."""
        extractor = AtherosExtractor()
        full = build_atheros_packet(nr=1, nc=1)
        # Cut off last byte of payload
        self.assertIsNone(extractor.parse(full[:-1]))

    def test_10bit_extremes(self):
        """10-bit signed extremes: 511 and -512."""
        num_tones = 56
        csi_values = []
        for k in range(num_tones):
            if k % 2 == 0:
                csi_values.extend([511, -512])   # max positive, min negative
            else:
                csi_values.extend([-512, 511])
        extractor = AtherosExtractor()
        packet = extractor.parse(
            build_atheros_packet(nr=1, nc=1, csi_values=csi_values)
        )
        self.assertIsNotNone(packet)
        # Even tones: imag=511, real=-512 → after swap: real=-512, imag=511
        self.assertAlmostEqual(packet.iq_data[0, 0], -512.0)
        self.assertAlmostEqual(packet.iq_data[0, 1], 511.0)

    def test_chip_variants(self):
        """Different chip names should all parse correctly."""
        for chip in ["ar9380", "ar9580", "qca9558"]:
            extractor = AtherosExtractor(chip=chip)
            packet = extractor.parse(build_atheros_packet(nr=1, nc=1))
            self.assertIsNotNone(packet, f"Failed for chip={chip}")
            self.assertIn(chip, extractor.hardware_type)


class TestAtherosCapabilities(unittest.TestCase):
    """Test hardware capabilities reporting."""

    def test_capabilities(self):
        extractor = AtherosExtractor(chip="ar9380")
        caps = extractor.capabilities
        self.assertEqual(caps.supported_bandwidths, [20, 40])
        self.assertIn("2.4GHz", caps.supported_bands)
        self.assertIn("5GHz", caps.supported_bands)
        self.assertTrue(caps.supports_mimo)
        self.assertEqual(caps.max_tx_antennas, 3)
        self.assertEqual(caps.max_rx_antennas, 3)

    def test_subcarrier_configs(self):
        extractor = AtherosExtractor()
        caps = extractor.capabilities
        self.assertEqual(caps.subcarrier_configs[20].num_subcarriers, 56)
        self.assertEqual(caps.subcarrier_configs[40].num_subcarriers, 114)

    def test_default_profile(self):
        extractor = AtherosExtractor()
        profile = extractor.get_default_profile()
        self.assertIn("AR9380", profile["description"])
        self.assertEqual(profile["bandwidth_mhz"], 20)


if __name__ == '__main__':
    unittest.main()
