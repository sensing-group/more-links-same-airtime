"""Binary packet parsing for CSI sensing protocol.

Supports HELLO, HEARTBEAT, CSI v1/v2/v5-batch, and TRIGGER packets.
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
import struct
import logging

import numpy as np
from binascii import crc32

from config import (
    BINARY_MAGIC,
    BATCH_HEADER_SIZE,
    CHIP_TYPES,
    CSI_V1_PACKET_SIZE,
    CSI_V2_PACKET_SIZE,
    HEARTBEAT_PACKET_SIZE,
    HELLO_PACKET_SIZE,
    MSG_TYPE_CSI_V2,
    MSG_TYPE_CSI_V5,
    MSG_TYPE_HEARTBEAT,
    MSG_TYPE_HELLO,
    MSG_TYPE_TRIGGER,
    NUM_SUBCARRIERS,
    SAMPLE_SIZE_V5,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger packet layout
# ---------------------------------------------------------------------------
TRIGGER_PACKET_SIZE = 32
TRIGGER_SAMPLES_PER_MEAS = 5
TRIGGER_SEND_ALL = 1

# ---------------------------------------------------------------------------
# IQ data size (52 subcarriers * 2 bytes per I/Q component)
# ---------------------------------------------------------------------------
IQ_DATA_BYTES = NUM_SUBCARRIERS * 2 * 2  # 208 bytes -> 52 pairs of (I, Q) as int8*2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mac_bytes_to_str(mac_bytes: bytes) -> str:
    """Convert 6 raw MAC bytes to colon-separated hex string."""
    return ":".join(f"{b:02X}" for b in mac_bytes)


def validate_crc32(data: bytes) -> bool:
    """Return True if the last 4 bytes are a valid CRC-32 of the preceding data."""
    if len(data) < 4:
        return False
    received = struct.unpack("<I", data[-4:])[0]
    computed = crc32(data[:-4]) & 0xFFFFFFFF
    return received == computed


def _parse_iq(raw: bytes) -> np.ndarray:
    """Parse 104 bytes of raw IQ into (52, 2) float32 array.

    Each subcarrier is stored as two signed int8 values: (I, Q).
    """
    iq_raw = np.frombuffer(raw[:NUM_SUBCARRIERS * 2], dtype=np.int8)
    return iq_raw.reshape(NUM_SUBCARRIERS, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HelloPacket:
    mac: str
    ip: str
    chip: str
    version: str
    uptime_s: int
    free_heap: int


@dataclass
class HeartbeatPacket:
    mac: str
    uptime_s: int
    free_heap: int


@dataclass
class CSISample:
    tx_mac: str
    rx_mac: str
    seq: int
    timestamp_us: int
    rssi: int
    noise_floor: int
    num_carriers: int
    iq_data: np.ndarray  # Shape: (52, 2) float32
    channel: int = 0
    bandwidth: str = ""
    _amplitude: np.ndarray = None  # cached

    @property
    def amplitude(self) -> np.ndarray:
        """Per-subcarrier amplitude: sqrt(I^2 + Q^2). Computed once, cached."""
        if self._amplitude is None:
            self._amplitude = np.sqrt(self.iq_data[:, 0] ** 2 + self.iq_data[:, 1] ** 2)
        return self._amplitude


@dataclass
class TriggerPacket:
    tx_mac: str
    seq: int
    burst_count: int
    slot_ms: int = 20
    slot_id: int = 0

    def to_bytes(self) -> bytes:
        """Serialise to 32-byte binary trigger packet.

        Layout:
          [0]     magic 0xC5
          [1]     type 0x10
          [2:8]   tx_mac (6 bytes)
          [8:12]  seq (uint32 LE)
          [12:14] burst_count (uint16 LE)
          [14:16] slot_ms (uint16 LE)
          [16]    samples_per_meas
          [17]    send_all_samples
          [18]    slot_id
          [19:28] reserved
          [28:32] crc32
        """
        packet = bytearray(TRIGGER_PACKET_SIZE)
        packet[0] = BINARY_MAGIC
        packet[1] = MSG_TYPE_TRIGGER
        packet[2:8] = bytes(int(b, 16) for b in self.tx_mac.split(":"))
        struct.pack_into("<I", packet, 8, self.seq)
        struct.pack_into("<H", packet, 12, self.burst_count)
        struct.pack_into("<H", packet, 14, self.slot_ms)
        packet[16] = TRIGGER_SAMPLES_PER_MEAS
        packet[17] = TRIGGER_SEND_ALL
        packet[18] = self.slot_id & 0xFF
        crc_val = crc32(bytes(packet[:-4])) & 0xFFFFFFFF
        struct.pack_into("<I", packet, 28, crc_val)
        return bytes(packet)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_hello(data: bytes, ip: str) -> Optional[HelloPacket]:
    """Parse binary HELLO (40 bytes).

    Layout:
      magic(1) type(1) mac(6) ip(4) chip(1) fw_major(1) fw_minor(1)
      fw_patch(1) fw_tag(8) uptime(4) free_heap(4) min_heap(4) crc(4)
    """
    if len(data) < HELLO_PACKET_SIZE:
        return None
    if not validate_crc32(data[:HELLO_PACKET_SIZE]):
        logger.debug("HELLO CRC mismatch")
        return None

    mac = mac_bytes_to_str(data[2:8])
    # data[8:12] contains the IP reported by the device; we use the UDP source instead.
    chip_id = data[12]
    chip = CHIP_TYPES.get(chip_id, f"UNKNOWN(0x{chip_id:02X})")
    fw_major = data[13]
    fw_minor = data[14]
    fw_patch = data[15]
    fw_tag = data[16:24].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    version = f"{fw_major}.{fw_minor}.{fw_patch}"
    if fw_tag:
        version += f"-{fw_tag}"
    uptime_s = struct.unpack_from("<I", data, 24)[0]
    free_heap = struct.unpack_from("<I", data, 28)[0]

    return HelloPacket(
        mac=mac,
        ip=ip,
        chip=chip,
        version=version,
        uptime_s=uptime_s,
        free_heap=free_heap,
    )


def parse_heartbeat(data: bytes) -> Optional[HeartbeatPacket]:
    """Parse binary HEARTBEAT (24 bytes).

    Layout:
      magic(1) type(1) mac(6) uptime(4) free_heap(4) min_heap(4) crc(4)
    """
    if len(data) < HEARTBEAT_PACKET_SIZE:
        return None
    if not validate_crc32(data[:HEARTBEAT_PACKET_SIZE]):
        logger.debug("HEARTBEAT CRC mismatch")
        return None

    mac = mac_bytes_to_str(data[2:8])
    uptime_s = struct.unpack_from("<I", data, 8)[0]
    free_heap = struct.unpack_from("<I", data, 12)[0]

    return HeartbeatPacket(mac=mac, uptime_s=uptime_s, free_heap=free_heap)


def parse_csi_v1(data: bytes, rx_mac: str) -> Optional[CSISample]:
    """Parse 128-byte v1 CSI packet.

    Layout:
      magic(2) tx_mac(6) seq(2) timestamp(4) rssi(1) noise(1)
      channel(1) bw(1) carriers(1) reserved(1) iq(104) crc(4)
    """
    if len(data) < CSI_V1_PACKET_SIZE:
        return None
    if not validate_crc32(data[:CSI_V1_PACKET_SIZE]):
        logger.debug("CSI v1 CRC mismatch")
        return None

    tx_mac = mac_bytes_to_str(data[2:8])
    seq = struct.unpack_from("<H", data, 8)[0]
    timestamp_us = struct.unpack_from("<I", data, 10)[0]
    rssi = struct.unpack_from("<b", data, 14)[0]  # signed
    noise_floor = struct.unpack_from("<b", data, 15)[0]  # signed
    channel = data[16]
    bw_byte = data[17]
    num_carriers = data[18]
    iq_data = _parse_iq(data[20:124])

    return CSISample(
        tx_mac=tx_mac,
        rx_mac=rx_mac,
        seq=seq,
        timestamp_us=timestamp_us,
        rssi=rssi,
        noise_floor=noise_floor,
        num_carriers=num_carriers,
        iq_data=iq_data,
        channel=channel,
        bandwidth="HT40" if bw_byte else "HT20",
    )


def parse_csi_v2(data: bytes) -> Optional[CSISample]:
    """Parse 148-byte v2 CSI packet (includes rx_mac).

    Layout:
      magic(2) tx_mac(6) rx_mac(6) seq(2) sample_idx(1) sample_count(1)
      timestamp(4) rssi(1) noise(1) rx_state(1) channel(1) bw(1) sig_mode(1)
      carriers(1) stbc(1) pilot(8) reserved(2) iq(104) crc(4)
    """
    if len(data) < CSI_V2_PACKET_SIZE:
        return None
    if not validate_crc32(data[:CSI_V2_PACKET_SIZE]):
        logger.debug("CSI v2 CRC mismatch")
        return None

    tx_mac = mac_bytes_to_str(data[2:8])
    rx_mac = mac_bytes_to_str(data[8:14])
    seq = struct.unpack_from("<H", data, 14)[0]
    # sample_idx = data[16], sample_count = data[17] — informational, not stored
    timestamp_us = struct.unpack_from("<I", data, 18)[0]
    rssi = struct.unpack_from("<b", data, 22)[0]
    noise_floor = struct.unpack_from("<b", data, 23)[0]
    # rx_state(24) channel(25) bw(26) sig_mode(27)
    channel = data[25]
    bw_byte = data[26]
    num_carriers = data[28]
    # stbc(29) pilot(30:38) reserved(38:40)
    iq_data = _parse_iq(data[40:144])

    return CSISample(
        tx_mac=tx_mac,
        rx_mac=rx_mac,
        seq=seq,
        timestamp_us=timestamp_us,
        rssi=rssi,
        noise_floor=noise_floor,
        num_carriers=num_carriers,
        iq_data=iq_data,
        channel=channel,
        bandwidth="HT40" if bw_byte else "HT20",
    )


def parse_csi_batch_v5(
    data: bytes,
    rx_mac: str,
    slot_resolver: Callable[[int], Optional[str]],
) -> List[CSISample]:
    """Parse v5 batch: header(8) + N * sample(107) + crc(4).

    Header layout:
      magic(2) slot_id(1) count(1) seq(2) timestamp_lo(2)

    Per-sample layout (107 bytes):
      delta_us(2) rssi(1) iq(104)
    """
    min_size = BATCH_HEADER_SIZE + SAMPLE_SIZE_V5 + 4  # header + 1 sample + crc
    if len(data) < min_size:
        return []
    if not validate_crc32(data):
        logger.debug("CSI v5 batch CRC mismatch")
        return []

    # Parse header
    slot_id = data[2]
    count = data[3]
    seq_base = struct.unpack_from("<H", data, 4)[0]
    timestamp_lo = struct.unpack_from("<H", data, 6)[0]

    # Resolve tx_mac from slot_id
    tx_mac = slot_resolver(slot_id)
    if tx_mac is None:
        logger.debug("v5 batch: no MAC for slot_id=%d", slot_id)
        return []

    expected_size = BATCH_HEADER_SIZE + count * SAMPLE_SIZE_V5 + 4
    if len(data) < expected_size:
        logger.debug(
            "v5 batch: expected %d bytes, got %d", expected_size, len(data)
        )
        return []

    samples: List[CSISample] = []

    for i in range(count):
        offset = BATCH_HEADER_SIZE + i * SAMPLE_SIZE_V5
        delta_us = struct.unpack_from("<H", data, offset)[0]
        rssi = struct.unpack_from("<b", data, offset + 2)[0]
        iq_data = _parse_iq(data[offset + 3 : offset + 3 + NUM_SUBCARRIERS * 2])

        timestamp_us = timestamp_lo + delta_us

        samples.append(
            CSISample(
                tx_mac=tx_mac,
                rx_mac=rx_mac,
                seq=seq_base + i,
                timestamp_us=timestamp_us,
                rssi=rssi,
                noise_floor=0,  # not included in v5
                num_carriers=NUM_SUBCARRIERS,
                iq_data=iq_data,
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def parse_packet(
    data: bytes,
    addr: tuple,
    ip_to_mac: dict,
    slot_resolver: Callable[[int], Optional[str]],
) -> tuple:
    """Dispatch incoming UDP data to the correct parser.

    Returns:
        ("hello", HelloPacket)
        ("heartbeat", HeartbeatPacket)
        ("csi", [CSISample, ...])
        (None, None)  on unknown / failed parse
    """
    if not data or data[0] != BINARY_MAGIC:
        return (None, None)

    msg_type = data[1] if len(data) > 1 else 0
    ip = addr[0]

    if msg_type == MSG_TYPE_HELLO:
        pkt = parse_hello(data, ip)
        return ("hello", pkt) if pkt else (None, None)

    if msg_type == MSG_TYPE_HEARTBEAT:
        # v1 CSI packets also use 0x02 as second byte — disambiguate by size
        if len(data) == CSI_V1_PACKET_SIZE:
            rx_mac = ip_to_mac.get(ip)
            if rx_mac is None:
                logger.debug("CSI v1 from unknown IP %s", ip)
                return (None, None)
            pkt = parse_csi_v1(data, rx_mac)
            return ("csi", [pkt]) if pkt else (None, None)
        pkt = parse_heartbeat(data)
        return ("heartbeat", pkt) if pkt else (None, None)

    if msg_type == MSG_TYPE_CSI_V2:
        # v2 carries rx_mac inside the packet
        pkt = parse_csi_v2(data)
        return ("csi", [pkt]) if pkt else (None, None)

    if msg_type == MSG_TYPE_CSI_V5:
        # v5 batch — resolve rx_mac from sender IP
        rx_mac = ip_to_mac.get(ip)
        if rx_mac is None:
            logger.debug("v5 batch from unknown IP %s", ip)
            return (None, None)
        samples = parse_csi_batch_v5(data, rx_mac, slot_resolver)
        return ("csi", samples) if samples else (None, None)

    # v1 fallback: msg_type not explicitly defined — check packet size
    if len(data) == CSI_V1_PACKET_SIZE:
        rx_mac = ip_to_mac.get(ip)
        if rx_mac is None:
            logger.debug("CSI v1 from unknown IP %s", ip)
            return (None, None)
        pkt = parse_csi_v1(data, rx_mac)
        return ("csi", [pkt]) if pkt else (None, None)

    logger.debug("Unknown msg_type=0x%02X len=%d", msg_type, len(data))
    return (None, None)
