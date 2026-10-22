"""Tests for FeitCSI binary format parser.

Tests parsing of FeitCSI's 272-byte header + I/Q payload format
into OpenCSI's CSIPacket abstraction. Covers format decoding,
single-record parsing, MIMO streams, 160 MHz firmware bug fix,
and .dat file iteration.
"""

import os
import struct
import tempfile

import numpy as np
import pytest

from opencsi.hardware.feitcsi import (
    FEITCSI_HEADER_SIZE,
    BYTES_PER_SUBCARRIER,
    IWLWIFI_DEFAULT_NOISE_FLOOR,
    IWLWIFI_DEFAULT_SNR_DB,
    FeitCSIExtractor,
    _decode_format,
    _decode_bandwidth_mhz,
    _estimate_rssi_dbm,
)
from opencsi.hardware.base import CSIPacket
from opencsi.hardware.iwlwifi import IwlWifiExtractor, CSISource


# ====================================================================
# Test helpers
# ====================================================================


def _build_feitcsi_header(
    csi_data_size=0,
    timestamp=1000000,
    num_rx=1,
    num_tx=1,
    num_subcarriers=52,
    rssi1=50,
    rssi2=0,
    src_mac=b'\xAA\xBB\xCC\xDD\xEE\xFF',
    rate_n_flag=0x100,  # OFDM 20MHz by default
):
    """Build a 272-byte FeitCSI header."""
    header = bytearray(FEITCSI_HEADER_SIZE)
    struct.pack_into('<I', header, 0, csi_data_size)
    struct.pack_into('<Q', header, 12, timestamp)
    header[46] = num_rx
    header[47] = num_tx
    struct.pack_into('<I', header, 52, num_subcarriers)
    struct.pack_into('<I', header, 60, rssi1)
    struct.pack_into('<I', header, 64, rssi2)
    header[68:74] = src_mac
    struct.pack_into('<I', header, 92, rate_n_flag)
    return bytes(header)


def _build_feitcsi_record(
    num_rx=1,
    num_tx=1,
    num_subcarriers=52,
    rate_n_flag=0x100,
    iq_value=(100, -50),
    **header_kwargs,
):
    """Build a complete FeitCSI record (header + payload)."""
    n_total = num_rx * num_tx * num_subcarriers
    csi_data_size = n_total * BYTES_PER_SUBCARRIER

    header = _build_feitcsi_header(
        csi_data_size=csi_data_size,
        num_rx=num_rx,
        num_tx=num_tx,
        num_subcarriers=num_subcarriers,
        rate_n_flag=rate_n_flag,
        **header_kwargs,
    )

    # Build I/Q payload: int16 real + int16 imag per subcarrier
    payload = bytearray(csi_data_size)
    for i in range(n_total):
        struct.pack_into('<hh', payload, i * 4, iq_value[0], iq_value[1])

    return header + bytes(payload)


# ====================================================================
# Task 1: Format decoding
# ====================================================================


class TestFormatDecoding:
    """Test rateNflag field decoding."""

    def test_decode_ofdm(self):
        assert _decode_format(0x100) == "OFDM"

    def test_decode_ht(self):
        assert _decode_format(0x200) == "HT"

    def test_decode_vht(self):
        assert _decode_format(0x300) == "VHT"

    def test_decode_he(self):
        assert _decode_format(0x400) == "HE"

    def test_decode_eht(self):
        assert _decode_format(0x500) == "EHT"

    def test_decode_cck(self):
        assert _decode_format(0x000) == "CCK"

    def test_decode_bandwidth_20(self):
        assert _decode_bandwidth_mhz(0x000) == 20

    def test_decode_bandwidth_40(self):
        assert _decode_bandwidth_mhz(0x800) == 40

    def test_decode_bandwidth_80(self):
        assert _decode_bandwidth_mhz(0x1000) == 80

    def test_decode_bandwidth_160(self):
        assert _decode_bandwidth_mhz(0x1800) == 160

    def test_decode_bandwidth_320(self):
        assert _decode_bandwidth_mhz(0x2000) == 320


# ====================================================================
# RSSI conversion
# ====================================================================


class TestRSSIConversion:
    """Test raw RSSI to dBm conversion."""

    def test_negative_dbm_passthrough(self):
        """Negative dBm value (-50) passes through."""
        assert _estimate_rssi_dbm(0xFFFFFFCE) == -50  # -50 as uint32

    def test_negative_dbm_small(self):
        """Typical weak signal (-85 dBm)."""
        raw = 0xFFFFFFAB  # -85 as uint32
        assert _estimate_rssi_dbm(raw) == -85

    def test_zero_is_plausible(self):
        """0 dBm is edge of plausible range."""
        assert _estimate_rssi_dbm(0) == 0

    def test_positive_raw_uses_default(self):
        """Positive raw energy value falls back to estimate."""
        result = _estimate_rssi_dbm(42)
        assert result == IWLWIFI_DEFAULT_NOISE_FLOOR + IWLWIFI_DEFAULT_SNR_DB

    def test_large_positive_uses_default(self):
        """Large raw value (not a negative dBm) falls back."""
        result = _estimate_rssi_dbm(5000)
        assert result == IWLWIFI_DEFAULT_NOISE_FLOOR + IWLWIFI_DEFAULT_SNR_DB

    def test_snr_is_reasonable(self):
        """Default estimate produces reasonable SNR."""
        rssi = _estimate_rssi_dbm(42)
        snr = rssi - IWLWIFI_DEFAULT_NOISE_FLOOR
        assert 0 < snr <= 50


# ====================================================================
# Task 2: Single-record parsing
# ====================================================================


class TestFeitCSIParsing:
    """Test single-record parsing."""

    def test_parse_siso_ofdm(self):
        """Parse 1x1 OFDM 20MHz record."""
        extractor = FeitCSIExtractor(channel=36)
        data = _build_feitcsi_record(
            num_rx=1, num_tx=1, num_subcarriers=52,
            rate_n_flag=0x100, iq_value=(100, -50),
            timestamp=5000000, rssi1=42,
        )
        packet = extractor.parse(data, rx_mac="11:22:33:44:55:66")

        assert packet is not None
        assert isinstance(packet, CSIPacket)
        assert packet.tx_mac == "AA:BB:CC:DD:EE:FF"
        assert packet.rx_mac == "11:22:33:44:55:66"
        assert packet.timestamp_us == 5000000
        assert packet.n_subcarriers == 52
        assert packet.n_rx_antennas == 1
        assert packet.n_tx_antennas == 1
        assert packet.bandwidth_mhz == 20
        assert packet.channel == 36
        assert packet.iq_data.shape == (52, 2)
        assert packet.iq_data[0, 0] == pytest.approx(100.0)
        assert packet.iq_data[0, 1] == pytest.approx(-50.0)
        assert packet.hardware_type == "iwlwifi_ax210"
        assert packet.antenna_sel == (0, 0)

    def test_parse_he_80mhz(self):
        """Parse HE 80 MHz (996 subcarriers, our real AX210 config)."""
        # HE=0x400, 80MHz=(2<<11)=0x1000
        rate_n_flag = 0x400 | 0x1000
        extractor = FeitCSIExtractor(channel=48)
        data = _build_feitcsi_record(
            num_rx=2, num_tx=1, num_subcarriers=996,
            rate_n_flag=rate_n_flag, iq_value=(200, 150),
        )
        packet = extractor.parse(data)

        assert packet is not None
        assert packet.n_subcarriers == 996
        assert packet.bandwidth_mhz == 80
        assert packet.n_rx_antennas == 2
        assert packet.n_tx_antennas == 1
        # parse() returns first stream (rx=0, tx=0)
        assert packet.iq_data.shape == (996, 2)
        assert packet.antenna_sel == (0, 0)

    def test_parse_extracts_src_mac(self):
        """Source MAC is correctly extracted from header."""
        extractor = FeitCSIExtractor()
        data = _build_feitcsi_record(
            src_mac=b'\x01\x02\x03\x04\x05\x06',
        )
        packet = extractor.parse(data)
        assert packet is not None
        assert packet.tx_mac == "01:02:03:04:05:06"

    def test_parse_short_data_returns_none(self):
        """Data shorter than header returns None."""
        extractor = FeitCSIExtractor()
        assert extractor.parse(b'\x00' * 100) is None

    def test_parse_truncated_payload_returns_none(self):
        """Header valid but payload truncated returns None."""
        extractor = FeitCSIExtractor()
        data = _build_feitcsi_record()
        assert extractor.parse(data[:-10]) is None

    def test_parse_frequency_band_5ghz(self):
        """Channel >= 36 -> 5GHz band."""
        extractor = FeitCSIExtractor(channel=36)
        data = _build_feitcsi_record()
        packet = extractor.parse(data)
        assert packet.frequency_band == "5GHz"

    def test_parse_frequency_band_24ghz(self):
        """Channel < 36 -> 2.4GHz band."""
        extractor = FeitCSIExtractor(channel=6)
        data = _build_feitcsi_record()
        packet = extractor.parse(data)
        assert packet.frequency_band == "2.4GHz"


# ====================================================================
# Task 3: MIMO stream handling
# ====================================================================


class TestFeitCSIMIMO:
    """Test MIMO stream handling."""

    def test_parse_all_2x1(self):
        """2 RX, 1 TX -> 2 packets."""
        rate_n_flag = 0x400 | 0x1000  # HE 80MHz
        extractor = FeitCSIExtractor(channel=48)

        # Build with distinct values per stream
        num_sub = 996
        payload = bytearray(2 * 1 * num_sub * 4)
        # rx0/tx0: (10, 20) for all subcarriers
        for i in range(num_sub):
            struct.pack_into('<hh', payload, i * 4, 10, 20)
        # rx1/tx0: (30, 40) for all subcarriers
        offset = num_sub * 4
        for i in range(num_sub):
            struct.pack_into('<hh', payload, offset + i * 4, 30, 40)

        header = _build_feitcsi_header(
            csi_data_size=len(payload),
            num_rx=2, num_tx=1, num_subcarriers=num_sub,
            rate_n_flag=rate_n_flag,
        )
        data = header + bytes(payload)

        packets = extractor.parse_all(data)

        assert len(packets) == 2
        # First packet: rx=0, tx=0
        assert packets[0].antenna_sel == (0, 0)
        assert packets[0].iq_data[0, 0] == pytest.approx(10.0)
        assert packets[0].iq_data[0, 1] == pytest.approx(20.0)
        # Second packet: rx=1, tx=0
        assert packets[1].antenna_sel == (0, 1)
        assert packets[1].iq_data[0, 0] == pytest.approx(30.0)
        assert packets[1].iq_data[0, 1] == pytest.approx(40.0)

    def test_parse_all_empty_on_invalid(self):
        """Invalid data returns empty list."""
        extractor = FeitCSIExtractor()
        assert extractor.parse_all(b'\x00' * 10) == []

    def test_parse_returns_first_stream(self):
        """parse() returns same as parse_all()[0]."""
        extractor = FeitCSIExtractor(channel=36)
        data = _build_feitcsi_record(num_rx=2, num_tx=1, num_subcarriers=52)

        single = extractor.parse(data)
        all_streams = extractor.parse_all(data)

        assert single is not None
        assert len(all_streams) == 2
        np.testing.assert_array_equal(single.iq_data, all_streams[0].iq_data)
        assert single.antenna_sel == all_streams[0].antenna_sel


# ====================================================================
# Task 4: 160 MHz firmware bug fix
# ====================================================================


class TestFeitCSIFirmwareBugFix:
    """Test 160 MHz firmware bug workaround."""

    def test_vht_160_strips_garbage(self):
        """VHT 160: 498 subcarriers stripped to 484."""
        # VHT=0x300, 160MHz=(3<<11)=0x1800
        rate_n_flag = 0x300 | 0x1800
        extractor = FeitCSIExtractor(channel=36)

        raw_sub = 498  # firmware reports 498 instead of 484
        data = _build_feitcsi_record(
            num_rx=1, num_tx=1, num_subcarriers=raw_sub,
            rate_n_flag=rate_n_flag,
        )
        packet = extractor.parse(data)
        assert packet is not None
        assert packet.n_subcarriers == 484

    def test_he_160_strips_garbage(self):
        """HE 160: 2020 subcarriers stripped to 1992."""
        # HE=0x400, 160MHz=0x1800
        rate_n_flag = 0x400 | 0x1800
        extractor = FeitCSIExtractor(channel=36)

        raw_sub = 2020
        data = _build_feitcsi_record(
            num_rx=1, num_tx=1, num_subcarriers=raw_sub,
            rate_n_flag=rate_n_flag,
        )
        packet = extractor.parse(data)
        assert packet is not None
        assert packet.n_subcarriers == 1992

    def test_no_fix_when_already_correct(self):
        """VHT 160 with 484 subcarriers needs no fix."""
        rate_n_flag = 0x300 | 0x1800
        extractor = FeitCSIExtractor(channel=36)
        data = _build_feitcsi_record(
            num_rx=1, num_tx=1, num_subcarriers=484,
            rate_n_flag=rate_n_flag,
        )
        packet = extractor.parse(data)
        assert packet is not None
        assert packet.n_subcarriers == 484

    def test_no_fix_for_80mhz(self):
        """80 MHz doesn't trigger the bug fix."""
        rate_n_flag = 0x400 | 0x1000
        extractor = FeitCSIExtractor(channel=36)
        data = _build_feitcsi_record(
            num_rx=1, num_tx=1, num_subcarriers=996,
            rate_n_flag=rate_n_flag,
        )
        packet = extractor.parse(data)
        assert packet is not None
        assert packet.n_subcarriers == 996


# ====================================================================
# Task 5: .dat file iteration
# ====================================================================


class TestFeitCSIFileIteration:
    """Test .dat file reading."""

    def test_iter_file_reads_records(self):
        """Read multiple records from a .dat file."""
        extractor = FeitCSIExtractor(channel=36)

        # Write 3 records to a temp file
        records = []
        for i in range(3):
            record = _build_feitcsi_record(
                num_rx=1, num_tx=1, num_subcarriers=52,
                timestamp=1000 * (i + 1),
                iq_value=(i * 10, i * 20),
            )
            records.append(record)

        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            for r in records:
                f.write(r)
            tmp_path = f.name

        try:
            packets = list(extractor.iter_file(tmp_path))
            assert len(packets) == 3
            # Each element is a list of CSIPackets (one per stream)
            assert len(packets[0]) == 1  # 1x1 SISO
            assert packets[0][0].timestamp_us == 1000
            assert packets[1][0].timestamp_us == 2000
            assert packets[2][0].timestamp_us == 3000
        finally:
            os.unlink(tmp_path)

    def test_iter_file_empty_file(self):
        """Empty file yields nothing."""
        extractor = FeitCSIExtractor()
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            tmp_path = f.name
        try:
            packets = list(extractor.iter_file(tmp_path))
            assert packets == []
        finally:
            os.unlink(tmp_path)

    def test_iter_file_mimo(self):
        """File with 2x1 MIMO records yields 2 packets per record."""
        extractor = FeitCSIExtractor(channel=36)
        record = _build_feitcsi_record(num_rx=2, num_tx=1, num_subcarriers=52)

        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            f.write(record)
            tmp_path = f.name

        try:
            packets = list(extractor.iter_file(tmp_path))
            assert len(packets) == 1
            assert len(packets[0]) == 2  # 2 RX streams
        finally:
            os.unlink(tmp_path)


# ====================================================================
# Task 7: Real data integration test
# ====================================================================


class TestFeitCSIRealData:
    """Integration test with real AX210 capture."""

    REAL_DATA_PATH = "/tmp/feitcsi_test.dat"

    @pytest.mark.skipif(
        not os.path.exists("/tmp/feitcsi_test.dat"),
        reason="Real FeitCSI data not available",
    )
    def test_parse_real_capture(self):
        """Parse first record from real AX210 HE 80MHz capture."""
        extractor = FeitCSIExtractor(device="ax210", channel=48)

        records = list(extractor.iter_file(self.REAL_DATA_PATH))
        assert len(records) > 0

        first = records[0]
        assert len(first) >= 1  # At least one stream

        packet = first[0]
        assert packet.n_subcarriers == 996
        assert packet.bandwidth_mhz == 80
        assert packet.n_rx_antennas == 2
        assert packet.n_tx_antennas == 1
        assert packet.iq_data.shape == (996, 2)
        # Verify data is non-trivial
        assert np.any(packet.iq_data != 0)
        # Verify amplitude is reasonable (not all zeros, not all max)
        amp = packet.amplitude
        assert np.mean(amp) > 0
        assert np.mean(amp) < 32768


# ====================================================================
# Task 6: IwlWifiExtractor delegation
# ====================================================================


class TestIwlWifiDelegation:
    """Test IwlWifiExtractor delegates to FeitCSI."""

    def test_parse_with_feitcsi_source(self):
        """IwlWifiExtractor(source=FEITCSI) delegates to FeitCSI."""
        extractor = IwlWifiExtractor(device="ax210", source=CSISource.FEITCSI)
        data = _build_feitcsi_record(
            num_rx=2, num_tx=1, num_subcarriers=52,
        )
        packet = extractor.parse(data, rx_mac="AA:BB:CC:DD:EE:FF")
        assert packet is not None
        assert isinstance(packet, CSIPacket)
        assert packet.rx_mac == "AA:BB:CC:DD:EE:FF"

    def test_parse_unknown_source_still_raises(self):
        """Non-FEITCSI source still raises NotImplementedError."""
        extractor = IwlWifiExtractor(device="ax210", source=CSISource.PICOSCENES)
        with pytest.raises(NotImplementedError):
            extractor.parse(b'\x00' * 300)
