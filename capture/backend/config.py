"""Configuration constants for CSI sensing backend.

All values are overridable via environment variables.
"""

import os

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
UDP_PORT = int(os.environ.get("UDP_PORT", "5000"))
TRIGGER_PORT = int(os.environ.get("TRIGGER_PORT", "5001"))
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
BROADCAST_ADDR = os.environ.get("BROADCAST_ADDR", "255.255.255.255")
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", "4096"))

# ---------------------------------------------------------------------------
# TDMA Timing
# ---------------------------------------------------------------------------
SLOT_DURATION_MS = int(os.environ.get("SLOT_DURATION_MS", "20"))
BURSTS_PER_TRIGGER = int(os.environ.get("BURSTS_PER_TRIGGER", "10"))
CYCLE_DELAY_MS = int(os.environ.get("CYCLE_DELAY_MS", "5"))
MIN_NODES_TO_START = int(os.environ.get("MIN_NODES_TO_START", "2"))

# ---------------------------------------------------------------------------
# Zone Configuration
# ---------------------------------------------------------------------------
ZONE_MAX_SIZE = int(os.environ.get("ZONE_MAX_SIZE", "6"))

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
MOTION_ZSCORE_THRESHOLD = float(os.environ.get("MOTION_ZSCORE_THRESHOLD", "2.5"))
ACTIVE_LINK_THRESHOLD = int(os.environ.get("ACTIVE_LINK_THRESHOLD", "3"))
BASELINE_LEARNING_SAMPLES = int(os.environ.get("BASELINE_LEARNING_SAMPLES", "200"))
# Per-link readiness / link-quality gating.
# A link is "viable" if its recent sample rate is above this floor. Dead or
# marginal links below this rate are excluded from calibration quorum,
# detection, and progress reporting so they cannot block the system.
MIN_LINK_RATE_HZ = float(os.environ.get("MIN_LINK_RATE_HZ", "0.5"))
# Fraction of viable links that must reach BASELINE_LEARNING_SAMPLES before
# the detector transitions from CALIBRATING to MONITORING.
READINESS_QUORUM_FRACTION = float(os.environ.get("READINESS_QUORUM_FRACTION", "0.7"))
# Minimum absolute number of viable links required for the quorum to be
# meaningful at all (avoids early flip with only a few nodes online).
READINESS_QUORUM_MIN_LINKS = int(os.environ.get("READINESS_QUORUM_MIN_LINKS", "10"))

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
HEARTBEAT_TIMEOUT_S = float(os.environ.get("HEARTBEAT_TIMEOUT_S", "30.0"))
HEARTBEAT_CHECK_INTERVAL_S = float(os.environ.get("HEARTBEAT_CHECK_S", "5.0"))

# ---------------------------------------------------------------------------
# Room (default dimensions in metres)
# ---------------------------------------------------------------------------
DEFAULT_ROOM_WIDTH = float(os.environ.get("ROOM_WIDTH", "5.0"))
DEFAULT_ROOM_HEIGHT = float(os.environ.get("ROOM_HEIGHT", "4.0"))

# ---------------------------------------------------------------------------
# Binary Protocol
# ---------------------------------------------------------------------------
BINARY_MAGIC = 0xC5

MSG_TYPE_HELLO = 0x01
MSG_TYPE_HEARTBEAT = 0x02
MSG_TYPE_CSI_V2 = 0x03
MSG_TYPE_CSI_V5 = 0x07
MSG_TYPE_TRIGGER = 0x10
MSG_TYPE_TEST = 0xFF

# ---------------------------------------------------------------------------
# Packet Sizes (bytes)
# ---------------------------------------------------------------------------
HELLO_PACKET_SIZE = 40
HEARTBEAT_PACKET_SIZE = 24
CSI_V1_PACKET_SIZE = 128
CSI_V2_PACKET_SIZE = 148
BATCH_HEADER_SIZE = 8
SAMPLE_SIZE_V5 = 107

# ---------------------------------------------------------------------------
# CSI
# ---------------------------------------------------------------------------
NUM_SUBCARRIERS = 52

# ---------------------------------------------------------------------------
# Chip Types
# ---------------------------------------------------------------------------
CHIP_TYPES = {
    0x01: "ESP32",
    0x02: "ESP32-S2",
    0x03: "ESP32-C3",
    0x04: "ESP32-S3",
    0x05: "ESP32-C2",
    0x06: "ESP32-C6",
}
