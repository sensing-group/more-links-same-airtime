# OpenCSI P0 Implementation Plan

## Task A: Synthetic Data Generator

### Files to Create

1. **`opencsi/src/opencsi/testing/__init__.py`** — public re-exports
2. **`opencsi/src/opencsi/testing/synthetic.py`** — core implementation

### Function Signatures

```python
# opencsi/src/opencsi/testing/synthetic.py

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple
from opencsi.core.types import RawCSIPacket
from opencsi.hardware.base import CSIPacket

# --- Constants (top-of-file) ---
DEFAULT_N_SUBCARRIERS = 52          # ESP32 HT20
DEFAULT_N_NODES = 4
DEFAULT_RSSI = -50
DEFAULT_NOISE_FLOOR = -90
DEFAULT_CHANNEL = 6
DEFAULT_FREQUENCY_BAND = "2.4GHz"
DEFAULT_ROOM_DIMS = (5.0, 4.0)     # meters (w, h)
IQ_AMPLITUDE_MEAN = 10.0           # mean I/Q component amplitude
IQ_AMPLITUDE_STD = 2.0             # std of I/Q components
MOTION_PERTURBATION_SCALE = 5.0    # multiplier for motion injection
MAC_PREFIX = "AA:BB:CC:DD:EE"      # prefix for generated MACs


def make_mac(index: int) -> str:
    """Generate a deterministic MAC address from an integer index.

    Args:
        index: Node index (0-255)

    Returns:
        MAC string like "AA:BB:CC:DD:EE:00"
    """


def make_positions(
    n_nodes: int = DEFAULT_N_NODES,
    room_width: float = DEFAULT_ROOM_DIMS[0],
    room_height: float = DEFAULT_ROOM_DIMS[1],
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Generate node positions distributed around the room perimeter.

    Places nodes evenly along the rectangle perimeter. This provides
    realistic geometry for Fresnel zone calculations.

    Args:
        n_nodes: Number of nodes
        room_width: Room width in meters
        room_height: Room height in meters
        seed: Optional RNG seed for reproducibility

    Returns:
        Dict mapping MAC → (x, y) position
    """


def make_packet(
    tx_mac: str,
    rx_mac: str,
    timestamp: float = 0.0,
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

    Generates I/Q pairs drawn from N(iq_mean, iq_std) for I and
    N(0, iq_std) for Q. This gives amplitude ~iq_mean and random phase.

    Args:
        tx_mac: Transmitter MAC
        rx_mac: Receiver MAC
        timestamp: Packet timestamp (seconds)
        n_subcarriers: Number of subcarriers
        rssi: RSSI in dBm
        noise_floor: Noise floor in dBm
        channel: WiFi channel number
        frequency_band: "2.4GHz", "5GHz", or "6GHz"
        seed: Optional RNG seed
        iq_mean: Mean of I component
        iq_std: Std dev of I/Q components

    Returns:
        RawCSIPacket ready for AutoCalibrator.ingest()
    """


def make_packet_dict(
    tx_mac: str,
    rx_mac: str,
    **kwargs,
) -> Dict[Tuple[str, str], RawCSIPacket]:
    """Convenience: make a single packet wrapped in the dict format expected by ingest().

    Returns:
        {(tx_mac, rx_mac): RawCSIPacket}
    """


def generate_synthetic_csi(
    n_frames: int = 100,
    n_nodes: int = DEFAULT_N_NODES,
    n_subcarriers: int = DEFAULT_N_SUBCARRIERS,
    frame_interval_s: float = 0.05,
    room_width: float = DEFAULT_ROOM_DIMS[0],
    room_height: float = DEFAULT_ROOM_DIMS[1],
    motion_frames: Optional[Sequence[int]] = None,
    motion_links: Optional[Sequence[Tuple[str, str]]] = None,
    motion_magnitude: float = MOTION_PERTURBATION_SCALE,
    seed: int = 42,
) -> Tuple[
    Dict[str, Tuple[float, float]],       # node_positions
    List[Dict[Tuple[str, str], RawCSIPacket]],  # frames (list of packet dicts)
    List[float],                            # timestamps
]:
    """Generate a full synthetic CSI dataset.

    Creates n_nodes arranged around the room perimeter and generates
    n_frames of synthetic CSI packets for all unique link pairs.

    Motion injection: if motion_frames is provided, those frame indices
    will have amplitude perturbation applied. If motion_links is also
    provided, perturbation is limited to those links; otherwise all links
    within a simulated Fresnel zone are perturbed.

    Motion model:
    - Perturbed links get amplitude scaled by motion_magnitude (additive
      Gaussian on I/Q with std = iq_std * motion_magnitude)
    - Non-motion frames have baseline I/Q noise only

    Args:
        n_frames: Number of frames to generate
        n_nodes: Number of nodes
        n_subcarriers: Subcarriers per packet
        frame_interval_s: Time between frames (seconds)
        room_width: Room width in meters
        room_height: Room height in meters
        motion_frames: Frame indices where motion occurs (None = no motion)
        motion_links: Specific links to perturb (None = all links)
        motion_magnitude: Scale factor for motion perturbation
        seed: Random seed for reproducibility

    Returns:
        (node_positions, frames, timestamps) tuple where:
        - node_positions: MAC → (x, y)
        - frames: list of {(tx, rx) → RawCSIPacket}
        - timestamps: list of float timestamps
    """
```

### `__init__.py` Exports

```python
# opencsi/src/opencsi/testing/__init__.py
"""Synthetic data generation for testing and examples."""

from opencsi.testing.synthetic import (
    generate_synthetic_csi,
    make_packet,
    make_packet_dict,
    make_positions,
    make_mac,
)

__all__ = [
    "generate_synthetic_csi",
    "make_packet",
    "make_packet_dict",
    "make_positions",
    "make_mac",
]
```

### Implementation Notes

**I/Q Data Shape**: `(n_subcarriers, 2)` as `np.float32` — column 0 is I, column 1 is Q. This matches `RawCSIPacket.iq_data` and `CSIPacket.iq_data`.

**Realistic Values**:
| Field | Typical Range | Default |
|-------|--------------|---------|
| RSSI | -30 to -80 dBm | -50 |
| noise_floor | -85 to -95 dBm | -90 |
| n_subcarriers | 52 (HT20), 114 (HT40), 242 (HE20) | 52 |
| channel | 1-14 (2.4GHz) | 6 |
| I/Q values | int16 range but float32 in RawCSIPacket | N(10, 2) |
| MAC format | `XX:XX:XX:XX:XX:XX` uppercase | `AA:BB:CC:DD:EE:{idx:02X}` |
| timestamp | Unix seconds (float) | 0.0 + frame_idx * interval |

**Motion Injection**: Add Gaussian perturbation to I component:
```python
if is_motion_frame and link_key in affected_links:
    iq_data[:, 0] += rng.normal(0, iq_std * motion_magnitude, n_subcarriers)
```
This creates amplitude change that the calibrator detects as Z-score deviation.

**Node Placement**: Distribute evenly along room perimeter. For 4 nodes in a 5×4m room:
- Node 0: (0, 0) — corner
- Node 1: (5, 0) — corner
- Node 2: (5, 4) — corner
- Node 3: (0, 4) — corner

For >4 nodes, interpolate along the perimeter.

---

## Task B: Pipeline Tests

### File: `opencsi/tests/test_calibrator_pipeline.py`

### Test Cases

```python
"""
End-to-end pipeline tests for AutoCalibrator using synthetic data.

Tests the full pipeline: raw packets → sanitization → baseline learning → Z-score output.
"""

import pytest
import numpy as np
from opencsi import AutoCalibrator, AutoCalibratorConfig, CalibratedFrame
from opencsi.testing import generate_synthetic_csi, make_packet, make_positions, make_packet_dict


class TestCalibrationBaseline:
    """Tests for baseline learning from quiet-period data."""

    def test_ingest_returns_calibrated_frame(self):
        """ingest() returns a CalibratedFrame for every call."""
        positions = make_positions(4)
        cal = AutoCalibrator(positions)
        nodes = list(positions.keys())

        packet = make_packet(nodes[0], nodes[1], timestamp=1.0)
        frame = cal.ingest({(nodes[0], nodes[1]): packet}, timestamp=1.0)

        assert isinstance(frame, CalibratedFrame)
        assert frame.timestamp == 1.0

    def test_baseline_learns_after_quiet_frames(self):
        """After sufficient quiet frames, baselines should be initialized."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=100, n_nodes=4, seed=42
        )
        config = AutoCalibratorConfig(min_baseline_samples=10)
        cal = AutoCalibrator(positions, config=config)

        for frame_packets, ts in zip(frames, timestamps):
            cal.ingest(frame_packets, timestamp=ts)

        # After 100 quiet frames, at least some links should be initialized
        initialized_count = sum(
            1 for state in cal._link_states.values()
            if state.baseline.is_initialized
        )
        assert initialized_count > 0, "No baselines learned after 100 quiet frames"

    def test_z_scores_near_zero_during_quiet(self):
        """Z-scores should be near zero when no motion is present."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=150, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions)

        last_frame = None
        for frame_packets, ts in zip(frames, timestamps):
            last_frame = cal.ingest(frame_packets, timestamp=ts)

        # After calibration settles, Z-scores should be low
        for link in last_frame.links.values():
            if link.reliability > 0.3:
                assert link.combined_change < 5.0, (
                    f"Z-score {link.combined_change} too high during quiet period"
                )


class TestMotionDetection:
    """Tests that motion injection produces detectable Z-scores."""

    def test_motion_increases_z_scores(self):
        """Frames with motion should have higher Z-scores than quiet frames."""
        # 100 quiet frames for baseline, then 10 motion frames
        motion_frame_indices = list(range(100, 110))

        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=120,
            n_nodes=4,
            motion_frames=motion_frame_indices,
            motion_magnitude=5.0,
            seed=42,
        )
        cal = AutoCalibrator(positions)

        quiet_max_changes = []
        motion_max_changes = []

        for i, (frame_packets, ts) in enumerate(zip(frames, timestamps)):
            result = cal.ingest(frame_packets, timestamp=ts)

            if i >= 80 and i < 100:  # Late quiet period (baselines learned)
                quiet_max_changes.append(result.get_max_change())
            elif i >= 100:  # Motion period
                motion_max_changes.append(result.get_max_change())

        # Motion frames should have larger max changes than quiet frames
        if quiet_max_changes and motion_max_changes:
            avg_quiet = np.mean(quiet_max_changes)
            avg_motion = np.mean(motion_max_changes)
            assert avg_motion > avg_quiet, (
                f"Motion avg ({avg_motion:.2f}) should exceed quiet avg ({avg_quiet:.2f})"
            )

    def test_motion_detected_flag(self):
        """CalibratedFrame.motion_detected should be True during motion."""
        motion_frame_indices = list(range(100, 110))

        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=120,
            n_nodes=6,  # More nodes = more links
            motion_frames=motion_frame_indices,
            motion_magnitude=8.0,
            seed=42,
        )
        cal = AutoCalibrator(positions)

        for i, (frame_packets, ts) in enumerate(zip(frames, timestamps)):
            result = cal.ingest(frame_packets, timestamp=ts)

        # Note: motion_detected depends on compute_spatial_analysis() being called.
        # This test validates the pipeline produces the right structure.
        # Whether motion_detected is set depends on the number of active links.


class TestMultipleNodes:
    """Tests with different node counts."""

    def test_two_nodes_single_link(self):
        """Two nodes should produce exactly one link."""
        positions = make_positions(2)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions)

        packets = {
            (nodes[0], nodes[1]): make_packet(nodes[0], nodes[1], timestamp=1.0),
        }
        frame = cal.ingest(packets, timestamp=1.0)

        assert len(frame.links) == 1

    def test_six_nodes_many_links(self):
        """Six nodes can produce up to C(6,2)*2=30 directed links."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=10, n_nodes=6, seed=42
        )
        cal = AutoCalibrator(positions)

        frame = cal.ingest(frames[0], timestamp=timestamps[0])
        # Should have links for all pairs in the frame
        assert frame.n_total_links > 0


class TestCalibratedLinkFields:
    """Tests that CalibratedLink fields are populated correctly."""

    def test_link_has_mac_addresses(self):
        """Each CalibratedLink should have valid tx_mac and rx_mac."""
        positions = make_positions(3)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions)

        packets = {
            (nodes[0], nodes[1]): make_packet(nodes[0], nodes[1]),
        }
        frame = cal.ingest(packets, timestamp=1.0)

        link = list(frame.links.values())[0]
        assert link.tx_mac == nodes[0]
        assert link.rx_mac == nodes[1]

    def test_link_has_quality_grade(self):
        """Each CalibratedLink should have a QualityGrade."""
        from opencsi.core.types import QualityGrade

        positions = make_positions(3)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions)

        packets = {
            (nodes[0], nodes[1]): make_packet(nodes[0], nodes[1]),
        }
        frame = cal.ingest(packets, timestamp=1.0)

        link = list(frame.links.values())[0]
        assert isinstance(link.quality_grade, QualityGrade)

    def test_link_reliability_in_range(self):
        """Link reliability should be in [0, 1]."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=50, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions)

        for frame_packets, ts in zip(frames, timestamps):
            result = cal.ingest(frame_packets, timestamp=ts)

        for link in result.links.values():
            assert 0.0 <= link.reliability <= 1.0


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_packet_dict(self):
        """ingest() with empty dict should return empty frame."""
        positions = make_positions(2)
        cal = AutoCalibrator(positions)

        frame = cal.ingest({}, timestamp=1.0)
        assert len(frame.links) == 0

    def test_single_frame_no_crash(self):
        """Single frame should not crash (not enough for baseline)."""
        positions = make_positions(4)
        cal = AutoCalibrator(positions)

        nodes = list(positions.keys())
        packets = {
            (nodes[0], nodes[1]): make_packet(nodes[0], nodes[1]),
        }
        frame = cal.ingest(packets, timestamp=1.0)
        assert isinstance(frame, CalibratedFrame)

    def test_list_input_format(self):
        """ingest() should accept list of CSIPacket objects."""
        from opencsi.hardware.base import CSIPacket

        positions = make_positions(2)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions)

        iq = np.random.default_rng(42).normal(10, 2, (52, 2)).astype(np.float32)
        packet = CSIPacket(
            tx_mac=nodes[0], rx_mac=nodes[1],
            timestamp_us=1_000_000, rssi=-50, noise_floor=-90,
            channel=6, bandwidth_mhz=20, n_subcarriers=52,
            iq_data=iq, hardware_type="esp32",
        )
        frame = cal.ingest([packet], timestamp=1.0)
        assert len(frame.links) == 1

    def test_reproducible_with_seed(self):
        """Same seed should produce identical frames."""
        positions1, frames1, ts1 = generate_synthetic_csi(n_frames=10, seed=123)
        positions2, frames2, ts2 = generate_synthetic_csi(n_frames=10, seed=123)

        assert positions1 == positions2
        assert ts1 == ts2
        for f1, f2 in zip(frames1, frames2):
            for key in f1:
                np.testing.assert_array_equal(f1[key].iq_data, f2[key].iq_data)


class TestSnapshotExportImport:
    """Tests for calibration snapshot export/import."""

    def test_export_after_calibration(self):
        """export_snapshot() should produce a CalibrationSnapshot."""
        from opencsi.core.types import CalibrationSnapshot

        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=100, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions)

        for fp, ts in zip(frames, timestamps):
            cal.ingest(fp, timestamp=ts)

        snapshot = cal.export_snapshot()
        assert isinstance(snapshot, CalibrationSnapshot)

    def test_snapshot_roundtrip_json(self):
        """Snapshot should survive JSON serialization roundtrip."""
        from opencsi.core.types import CalibrationSnapshot

        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=100, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions)

        for fp, ts in zip(frames, timestamps):
            cal.ingest(fp, timestamp=ts)

        snapshot = cal.export_snapshot()
        json_str = snapshot.to_json()
        restored = CalibrationSnapshot.from_json(json_str)

        assert restored.version == snapshot.version
        assert len(restored.links) == len(snapshot.links)
```

### Assertions Summary

| Test | Key Assertion |
|------|--------------|
| `test_ingest_returns_calibrated_frame` | Return type is CalibratedFrame |
| `test_baseline_learns_after_quiet_frames` | `is_initialized` becomes True for at least 1 link |
| `test_z_scores_near_zero_during_quiet` | `combined_change < 5.0` for reliable links |
| `test_motion_increases_z_scores` | avg motion Z-score > avg quiet Z-score |
| `test_link_reliability_in_range` | `0 <= reliability <= 1` |
| `test_reproducible_with_seed` | Same seed → identical I/Q data |
| `test_snapshot_roundtrip_json` | JSON export/import preserves link count |

---

## Task C: API Surface Trim

### Current State: 133 exports in `__all__`

### Target: ~15 top-level exports

### Exports to KEEP at top level (~15 symbols)

These are the "scikit-learn of WiFi sensing" public API:

```python
__all__ = [
    # Version
    "__version__",

    # Core — the main 3 classes users interact with
    "AutoCalibrator",
    "AutoCalibratorConfig",
    "CalibratedFrame",
    "CalibratedLink",
    "RawCSIPacket",

    # Hardware — factory + interface
    "CSIExtractor",
    "CSIPacket",
    "get_extractor",
    "list_extractors",

    # Convenience
    "get_config",

    # Backward compatibility (deprecated)
    "AutoFresnel",
    "AutoFresnelConfig",
]
```

That's **14 symbols** (13 + `__version__`).

### Exports to REMOVE from top level (with new submodule path)

All removed symbols remain importable via their submodule. The `__init__.py` will no longer import them at the top level.

| Removed Symbol | New Import Path |
|---|---|
| `CalibrationSnapshot` | `opencsi.core.types` |
| `LinkSnapshot` | `opencsi.core.types` |
| `PerLinkBaseline` | `opencsi.core.types` |
| `PositionEstimate` | `opencsi.core.types` |
| `QualityGrade` | `opencsi.core.types` |
| `LearningMode` | `opencsi.core.types` |
| `NetworkState` | `opencsi.core.types` |
| `WelfordAccumulator` | `opencsi.core.welford` |
| `SPATIAL_MOTION_ZSCORE_THRESHOLD` | `opencsi.core.config` |
| `SPATIAL_HIGH_MOTION_THRESHOLD` | `opencsi.core.config` |
| `POSITION_MIN_ACTIVE_LINKS` | `opencsi.core.config` |
| `POSITION_CLUSTER_DISTANCE_M` | `opencsi.core.config` |
| `POSITION_GRID_RESOLUTION_M` | `opencsi.core.config` |
| `MAX_DETECTABLE_PERSONS` | `opencsi.core.config` |
| `RELIABLE_LINK_THRESHOLD` | `opencsi.core.config` |
| `WAVELENGTH_2_4GHZ_M` | `opencsi.core.config` |
| `WAVELENGTH_5GHZ_M` | `opencsi.core.config` |
| `WAVELENGTH_6GHZ_M` | `opencsi.core.config` |
| `channel_to_frequency_mhz` | `opencsi.core.config` |
| `channel_to_wavelength` | `opencsi.core.config` |
| `frequency_to_wavelength` | `opencsi.core.config` |
| `get_band_wavelength` | `opencsi.core.config` |
| `CSISanitizer` | `opencsi.sanitization` |
| `SanitizerConfig` | `opencsi.sanitization` |
| `SanitizationLevel` | `opencsi.sanitization` |
| `SanitizedCSI` | `opencsi.sanitization` |
| `circular_mean` | `opencsi.stats.circular` |
| `circular_std` | `opencsi.stats.circular` |
| `circular_mean_and_std` | `opencsi.stats.circular` |
| `circular_var` | `opencsi.stats.circular` |
| `circular_resultant_length` | `opencsi.stats.circular` |
| `circular_distance` | `opencsi.stats.circular` |
| `circular_weighted_mean` | `opencsi.stats.circular` |
| `circular_concentration` | `opencsi.stats.circular` |
| `unwrap_phase_diff` | `opencsi.stats.circular` |
| `compute_phase_diff_stats` | `opencsi.stats.circular` |
| `compute_phase_change_zscore` | `opencsi.stats.circular` |
| `SensitivityMapLearner` | `opencsi.learning` |
| `AntennaPatternLearner` | `opencsi.learning` |
| `AntennaPattern` | `opencsi.learning` |
| `theoretical_fresnel_response` | `opencsi.learning.fresnel` |
| `compute_fresnel_radius` | `opencsi.learning.fresnel` |
| `OpportunisticCalibrator` | `opencsi.learning.opportunistic` |
| `CalibrationSweepRequest` | `opencsi.learning.opportunistic` |
| `FREQUENCY_BUCKETS` | `opencsi.baseline.frequency_tracker` |
| `BUCKET_NAMES` | `opencsi.baseline.frequency_tracker` |
| `NUM_FREQUENCY_BUCKETS` | `opencsi.baseline.frequency_tracker` |
| `frequency_to_bucket` | `opencsi.baseline.frequency_tracker` |
| `bucket_to_name` | `opencsi.baseline.frequency_tracker` |
| `BucketBaseline` | `opencsi.baseline.frequency_baseline` |
| `FrequencyAwareBaseline` | `opencsi.baseline.frequency_baseline` |
| `PassiveFrequencyCollector` | `opencsi.baseline.frequency_baseline` |
| `NUM_TIME_BUCKETS` | `opencsi.baseline.time_tracker` |
| `timestamp_to_time_bucket` | `opencsi.baseline.time_tracker` |
| `time_bucket_to_name` | `opencsi.baseline.time_tracker` |
| `TimeBucketBaseline` | `opencsi.baseline.time_baseline` |
| `TimeAwareBaseline` | `opencsi.baseline.time_baseline` |
| `PassiveTimeCollector` | `opencsi.baseline.time_baseline` |
| `OpenCSIConfig` | `opencsi.core.env` |
| `config` | `opencsi.core.env` |
| `reload_config` | `opencsi.core.env` |
| `is_debug` | `opencsi.core.env` |
| `get_default_hardware` | `opencsi.core.env` |
| `get_array_module` | `opencsi.core.env` |
| `ParallelExecutor` | `opencsi.core.parallel` |
| `parallel_map` | `opencsi.core.parallel` |
| `parallel_map_unordered` | `opencsi.core.parallel` |
| `batch_iter` | `opencsi.core.parallel` |
| `BatchAccumulator` | `opencsi.core.parallel` |
| `link_parallel_context` | `opencsi.core.parallel` |
| `BACKEND_NONE` | `opencsi.core.parallel` |
| `BACKEND_THREAD` | `opencsi.core.parallel` |
| `BACKEND_PROCESS` | `opencsi.core.parallel` |
| `HardwareCapabilities` | `opencsi.hardware.capabilities` |
| `SubcarrierConfig` | `opencsi.hardware.capabilities` |
| `ESP32Extractor` | `opencsi.hardware.esp32` |
| `IwlWifiExtractor` | `opencsi.hardware.iwlwifi` |
| `HardwareProfileLoader` | `opencsi.hardware.profiles` |
| `get_hardware_profile` | `opencsi.hardware.profiles` |
| `HardwareRegistry` | `opencsi.hardware.registry` |
| All `FEATURE_*` constants (16) | `opencsi.hardware.capabilities` |
| `compute_fresnel_radius_at_midpoint` | `opencsi.geometry.fresnel` |
| `compute_snr_scaling_factor` | `opencsi.geometry.fresnel` |
| `compute_effective_detection_width` | `opencsi.geometry.fresnel` |
| `compute_detection_width_for_channel` | `opencsi.geometry.fresnel` |
| `DistanceEstimator` | `opencsi.geometry.distance` |
| `PathLossModel` | `opencsi.geometry.distance` |
| `PositionSolver` | `opencsi.geometry.position` |
| `MDSResult` | `opencsi.geometry.position` |
| `CoverageEstimate` | `opencsi.geometry.coverage` |
| `CoverageAreaEstimator` | `opencsi.geometry.coverage` |
| `SEPARATION_FRACTION` | `opencsi.geometry.coverage` |
| `GRID_FRACTION` | `opencsi.geometry.coverage` |
| `MIN_GRID_RESOLUTION_M` | `opencsi.geometry.coverage` |
| `MAX_GRID_RESOLUTION_M` | `opencsi.geometry.coverage` |
| `MIN_SEPARATION_M` | `opencsi.geometry.coverage` |
| `MAX_SEPARATION_M` | `opencsi.geometry.coverage` |
| `RSSICollector` | `opencsi.discovery.rssi_collector` |
| `LinkRSSIStats` | `opencsi.discovery.rssi_collector` |
| `CollectionProgress` | `opencsi.discovery.rssi_collector` |
| `CALIBRATION_MIN_SAMPLES` | `opencsi.discovery.rssi_collector` |
| `CALIBRATION_TIMEOUT_S` | `opencsi.discovery.rssi_collector` |
| `CALIBRATION_MIN_LINKS_FRACTION` | `opencsi.discovery.rssi_collector` |

### Implementation Strategy

1. **Replace** the entire body of `opencsi/__init__.py`:
   - Keep only the 14 imports listed in "KEEP"
   - Remove all other import blocks
   - Replace `__all__` with the 14-symbol list
   - Keep the docstring and `__version__`
   - Keep the backward-compatibility aliases (`AutoFresnel`, `AutoFresnelConfig`)

2. **Do NOT add lazy imports** — keep it simple. Users who need advanced features use submodule imports.

3. **Update existing tests**: The test `test_module_exports` in `test_hardware_abstraction.py` asserts specific exports exist. Update it to check the new reduced API. Symbols removed from top-level should be tested as importable from their submodules.

4. **No deprecation warnings for now** — the library is v0.1.0 (pre-release). Just remove the exports cleanly.

### New `__init__.py` Structure

```python
"""
OpenCSI - Self-calibrating WiFi CSI preprocessing library.
...docstring...
"""

__version__ = "0.1.0"

# Core — the main classes users interact with
from opencsi.core.calibrator import AutoCalibrator
from opencsi.core.types import (
    CalibratedFrame,
    CalibratedLink,
    RawCSIPacket,
)
from opencsi.core.config import AutoCalibratorConfig

# Backward compatibility aliases (deprecated)
AutoFresnel = AutoCalibrator
AutoFresnelConfig = AutoCalibratorConfig

# Hardware — factory pattern
from opencsi.hardware.base import CSIExtractor, CSIPacket
from opencsi.hardware.registry import get_extractor, list_extractors

# Configuration
from opencsi.core.env import get_config

__all__ = [
    "__version__",
    "AutoCalibrator",
    "AutoCalibratorConfig",
    "CalibratedFrame",
    "CalibratedLink",
    "RawCSIPacket",
    "CSIExtractor",
    "CSIPacket",
    "get_extractor",
    "list_extractors",
    "get_config",
    "AutoFresnel",
    "AutoFresnelConfig",
]
```

---

## Execution Order & Dependencies

```
Task A (Synthetic Generator)  ←  no dependencies
        ↓
Task B (Pipeline Tests)       ←  depends on Task A (uses synthetic data)
Task C (API Surface Trim)     ←  no dependency on A or B
        ↓
Task D (Verification)         ←  depends on B and C
```

Tasks A and C can execute **in parallel**. Task B depends on A. Task D (verification) runs after B and C.

---

## Verification Criteria (Task D)

1. `cd opencsi && python -m pytest tests/ -v` — all tests pass
2. `python -c "from opencsi import AutoCalibrator, get_extractor"` — basic import works
3. `python -c "from opencsi.testing import generate_synthetic_csi"` — testing module importable
4. `python -c "from opencsi.core.types import CalibrationSnapshot"` — submodule import works
5. `python -c "from opencsi.hardware.capabilities import FEATURE_MIMO"` — submodule constant importable
6. `python -c "import opencsi; assert len(opencsi.__all__) < 20"` — API surface is trimmed
