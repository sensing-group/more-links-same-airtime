"""
Synthetic CSI data generation for testing and examples.

Provides functions to generate realistic CSI packets with configurable
noise, motion injection, and node placement. All packets are compatible
with AutoCalibrator.ingest().

The synthetic data models frequency-selective fading: each link has a
stable per-subcarrier amplitude profile with small per-frame noise on
top. This produces low spectral entropy (structured spectrum) and low
temporal variance (stable over time), matching the quiet detection
requirements of AutoCalibrator.

Example:
    from opencsi.testing import generate_synthetic_csi, make_packet

    # Generate a full dataset with motion in frames 80-90
    positions, frames, timestamps = generate_synthetic_csi(
        n_frames=100,
        motion_frames=range(80, 90),
    )

    # Or create individual packets
    pkt = make_packet("AA:BB:CC:DD:EE:00", "AA:BB:CC:DD:EE:01")
"""

import time as _time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencsi.core.types import RawCSIPacket

# =============================================================================
# Module-level constants (configurable defaults)
# =============================================================================

DEFAULT_N_SUBCARRIERS = 52
"""ESP32 HT20 subcarrier count."""

DEFAULT_N_NODES = 4
"""Default number of simulated nodes."""

DEFAULT_RSSI = -50
"""Default RSSI in dBm (typical indoor)."""

DEFAULT_NOISE_FLOOR = -90
"""Default noise floor in dBm."""

DEFAULT_CHANNEL = 6
"""Default WiFi channel (2.4 GHz channel 6)."""

DEFAULT_FREQUENCY_BAND = "2.4GHz"
"""Default frequency band."""

DEFAULT_ROOM_DIMS = (5.0, 4.0)
"""Default room dimensions in meters (width, height)."""

IQ_AMPLITUDE_MEAN = 10.0
"""Mean of the I component for synthetic I/Q data."""

IQ_AMPLITUDE_STD = 2.0
"""Standard deviation of I/Q components."""

IQ_FRAME_NOISE_STD = 0.3
"""Small per-frame noise std added on top of the stable channel profile.

Kept small relative to IQ_AMPLITUDE_STD so temporal variance stays low
and the calibrator detects a quiet environment."""

MOTION_PERTURBATION_SCALE = 5.0
"""Default scale factor for motion injection noise."""

MAC_PREFIX = "AA:BB:CC:DD:EE"
"""Prefix for deterministic MAC address generation."""

DEFAULT_FRAME_INTERVAL_S = 0.05
"""Default inter-frame interval in seconds (50 ms / 20 Hz)."""

DEFAULT_N_FRAMES = 100
"""Default number of frames to generate."""

DEFAULT_SEED = 42
"""Default random seed for reproducibility."""

CHANNEL_PROFILE_N_TAPS = 3
"""Number of multipath taps for synthetic channel profile generation.

A small number of taps produces a smooth frequency-selective fading
pattern with low spectral entropy, mimicking real indoor channels."""


def make_mac(index: int) -> str:
    """Generate a deterministic MAC address from an integer index.

    Args:
        index: Node index (0-255).

    Returns:
        MAC string like "AA:BB:CC:DD:EE:00".

    Raises:
        ValueError: If index is outside [0, 255].
    """
    if not 0 <= index <= 255:
        raise ValueError(f"MAC index must be in [0, 255], got {index}")
    return f"{MAC_PREFIX}:{index:02X}"


def make_positions(
    n_nodes: int = DEFAULT_N_NODES,
    room_width: float = DEFAULT_ROOM_DIMS[0],
    room_height: float = DEFAULT_ROOM_DIMS[1],
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Generate node positions distributed around the room perimeter.

    Places nodes evenly along the rectangle perimeter. For 4 nodes in a
    5x4m room, this places one node at each corner. For more nodes,
    positions are interpolated along the perimeter.

    Args:
        n_nodes: Number of nodes.
        room_width: Room width in meters.
        room_height: Room height in meters.
        seed: Optional RNG seed (unused for deterministic placement,
              reserved for future jittered placement).

    Returns:
        Dict mapping MAC address -> (x, y) position in meters.
    """
    perimeter = 2 * (room_width + room_height)
    positions: Dict[str, Tuple[float, float]] = {}

    for i in range(n_nodes):
        # Walk along the perimeter
        dist = (i / n_nodes) * perimeter
        if dist <= room_width:
            # Bottom edge: left to right
            x, y = dist, 0.0
        elif dist <= room_width + room_height:
            # Right edge: bottom to top
            x, y = room_width, dist - room_width
        elif dist <= 2 * room_width + room_height:
            # Top edge: right to left
            x, y = room_width - (dist - room_width - room_height), room_height
        else:
            # Left edge: top to bottom
            x, y = 0.0, room_height - (dist - 2 * room_width - room_height)

        positions[make_mac(i)] = (round(x, 4), round(y, 4))

    return positions


def _make_channel_profile(
    rng: np.random.Generator,
    n_subcarriers: int,
) -> np.ndarray:
    """Generate a stable frequency-selective fading profile for one link.

    Uses a small number of complex multipath taps to create a smooth
    per-subcarrier amplitude profile. The result has low spectral entropy,
    matching real indoor WiFi channel characteristics.

    Args:
        rng: NumPy random generator instance.
        n_subcarriers: Number of subcarriers.

    Returns:
        Array of shape (n_subcarriers,) with positive amplitude values
        centered around IQ_AMPLITUDE_MEAN.
    """
    # Generate a few complex multipath taps (impulse response)
    taps = (
        rng.normal(0, 1, CHANNEL_PROFILE_N_TAPS)
        + 1j * rng.normal(0, 1, CHANNEL_PROFILE_N_TAPS)
    )
    # Compute frequency response via DFT at each subcarrier
    k = np.arange(n_subcarriers)
    freq_response = np.zeros(n_subcarriers, dtype=complex)
    for tap_idx in range(CHANNEL_PROFILE_N_TAPS):
        freq_response += taps[tap_idx] * np.exp(
            -2j * np.pi * tap_idx * k / n_subcarriers
        )
    # Convert to amplitude and scale to desired mean
    raw_amplitude = np.abs(freq_response)
    # Normalize: shift and scale so mean ≈ IQ_AMPLITUDE_MEAN, all positive
    if raw_amplitude.max() > 0:
        raw_amplitude = (
            raw_amplitude / raw_amplitude.mean() * IQ_AMPLITUDE_MEAN
        )
    else:
        raw_amplitude = np.full(n_subcarriers, IQ_AMPLITUDE_MEAN)
    return raw_amplitude.astype(np.float32)


def make_packet(
    tx_mac: str,
    rx_mac: str,
    timestamp: Optional[float] = None,
    n_subcarriers: int = DEFAULT_N_SUBCARRIERS,
    rssi: int = DEFAULT_RSSI,
    noise_floor: int = DEFAULT_NOISE_FLOOR,
    channel: int = DEFAULT_CHANNEL,
    frequency_band: str = DEFAULT_FREQUENCY_BAND,
    seed: Optional[int] = None,
    iq_mean: float = IQ_AMPLITUDE_MEAN,
    iq_std: float = IQ_AMPLITUDE_STD,
) -> RawCSIPacket:
    """Create a single synthetic RawCSIPacket with Gaussian I/Q data.

    Generates I/Q pairs drawn from N(iq_mean, iq_std) for the I component
    and N(0, iq_std) for the Q component. This gives amplitude approximately
    equal to iq_mean with random phase.

    Args:
        tx_mac: Transmitter MAC address.
        rx_mac: Receiver MAC address.
        timestamp: Packet timestamp in seconds. Defaults to current wall
            clock time if not provided.
        n_subcarriers: Number of subcarriers.
        rssi: RSSI in dBm.
        noise_floor: Noise floor in dBm.
        channel: WiFi channel number.
        frequency_band: "2.4GHz", "5GHz", or "6GHz".
        seed: Optional RNG seed for reproducibility.
        iq_mean: Mean of the I component.
        iq_std: Standard deviation of I/Q components.

    Returns:
        RawCSIPacket ready for AutoCalibrator.ingest().
    """
    rng = np.random.default_rng(seed)
    iq_data = np.empty((n_subcarriers, 2), dtype=np.float32)
    iq_data[:, 0] = rng.normal(iq_mean, iq_std, n_subcarriers)  # I
    iq_data[:, 1] = rng.normal(0.0, iq_std, n_subcarriers)      # Q

    return RawCSIPacket(
        tx_mac=tx_mac,
        rx_mac=rx_mac,
        iq_data=iq_data,
        timestamp=timestamp if timestamp is not None else _time.time(),
        rssi=rssi,
        noise_floor=noise_floor,
        channel=channel,
        frequency_band=frequency_band,
    )


def make_packet_dict(
    tx_mac: str,
    rx_mac: str,
    **kwargs,
) -> Dict[Tuple[str, str], RawCSIPacket]:
    """Convenience: make a single packet wrapped in the dict format expected by ingest().

    All keyword arguments are forwarded to make_packet().

    Args:
        tx_mac: Transmitter MAC address.
        rx_mac: Receiver MAC address.
        **kwargs: Additional arguments for make_packet().

    Returns:
        {(tx_mac, rx_mac): RawCSIPacket}
    """
    return {(tx_mac, rx_mac): make_packet(tx_mac, rx_mac, **kwargs)}


def generate_synthetic_csi(
    n_frames: int = DEFAULT_N_FRAMES,
    n_nodes: int = DEFAULT_N_NODES,
    n_subcarriers: int = DEFAULT_N_SUBCARRIERS,
    frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S,
    room_width: float = DEFAULT_ROOM_DIMS[0],
    room_height: float = DEFAULT_ROOM_DIMS[1],
    motion_frames: Optional[Sequence[int]] = None,
    motion_links: Optional[Sequence[Tuple[str, str]]] = None,
    motion_magnitude: float = MOTION_PERTURBATION_SCALE,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    Dict[str, Tuple[float, float]],              # node_positions
    List[Dict[Tuple[str, str], RawCSIPacket]],   # frames
    List[float],                                  # timestamps
]:
    """Generate a full synthetic CSI dataset.

    Creates n_nodes arranged around the room perimeter and generates
    n_frames of synthetic CSI packets for all unique TX-RX pairs.

    Each link gets a stable per-subcarrier amplitude profile (simulating
    frequency-selective fading) with small per-frame noise. This produces
    low spectral entropy and low temporal variance, allowing the
    AutoCalibrator's quiet detector to learn baselines.

    Motion injection: if motion_frames is provided, those frame indices
    will have amplitude perturbation applied. If motion_links is also
    provided, perturbation is limited to those links; otherwise all links
    are perturbed.

    Args:
        n_frames: Number of frames to generate.
        n_nodes: Number of nodes.
        n_subcarriers: Subcarriers per packet.
        frame_interval_s: Time between frames (seconds).
        room_width: Room width in meters.
        room_height: Room height in meters.
        motion_frames: Frame indices where motion occurs (None = no motion).
        motion_links: Specific (tx, rx) pairs to perturb (None = all links).
        motion_magnitude: Scale factor for motion perturbation.
        seed: Random seed for reproducibility.

    Returns:
        (node_positions, frames, timestamps) where:
        - node_positions: {MAC -> (x, y)}
        - frames: list of {(tx, rx) -> RawCSIPacket}
        - timestamps: list of float timestamps
    """
    rng = np.random.default_rng(seed)
    positions = make_positions(n_nodes, room_width, room_height)
    macs = list(positions.keys())

    # Build all directed link pairs (A->B and B->A are separate links)
    link_keys: List[Tuple[str, str]] = []
    for i in range(len(macs)):
        for j in range(len(macs)):
            if i != j:
                link_keys.append((macs[i], macs[j]))

    # Pre-generate a stable channel profile per link (frequency-selective fading)
    link_profiles: Dict[Tuple[str, str], np.ndarray] = {}
    for link_key in link_keys:
        link_profiles[link_key] = _make_channel_profile(rng, n_subcarriers)

    motion_frame_set = set(motion_frames) if motion_frames is not None else set()
    motion_link_set = set(motion_links) if motion_links is not None else None

    # Use wall-clock-relative timestamps so baseline age calculations in
    # AutoCalibrator (which use time.time()) produce realistic values.
    base_timestamp = _time.time()
    frames: List[Dict[Tuple[str, str], RawCSIPacket]] = []
    timestamps: List[float] = []

    for frame_idx in range(n_frames):
        ts = base_timestamp + frame_idx * frame_interval_s
        timestamps.append(ts)

        is_motion_frame = frame_idx in motion_frame_set
        frame_packets: Dict[Tuple[str, str], RawCSIPacket] = {}

        for link_key in link_keys:
            tx_mac, rx_mac = link_key
            profile = link_profiles[link_key]

            # I component: stable profile + small per-frame noise
            iq_data = np.empty((n_subcarriers, 2), dtype=np.float32)
            iq_data[:, 0] = profile + rng.normal(
                0, IQ_FRAME_NOISE_STD, n_subcarriers
            )
            # Q component: small noise (phase jitter)
            iq_data[:, 1] = rng.normal(0.0, IQ_FRAME_NOISE_STD, n_subcarriers)

            # Inject motion perturbation
            if is_motion_frame:
                should_perturb = (
                    motion_link_set is None
                    or (tx_mac, rx_mac) in motion_link_set
                )
                if should_perturb:
                    iq_data[:, 0] += rng.normal(
                        0,
                        IQ_AMPLITUDE_STD * motion_magnitude,
                        n_subcarriers,
                    )

            frame_packets[(tx_mac, rx_mac)] = RawCSIPacket(
                tx_mac=tx_mac,
                rx_mac=rx_mac,
                iq_data=iq_data,
                timestamp=ts,
                rssi=DEFAULT_RSSI,
                noise_floor=DEFAULT_NOISE_FLOOR,
                channel=DEFAULT_CHANNEL,
                frequency_band=DEFAULT_FREQUENCY_BAND,
            )

        frames.append(frame_packets)

    return positions, frames, timestamps
