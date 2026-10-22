"""
End-to-end pipeline tests for AutoCalibrator using synthetic data.

Tests the full pipeline: raw packets -> sanitization -> baseline learning -> Z-score output.
"""

import json

import numpy as np
import pytest

from opencsi import AutoCalibrator, AutoCalibratorConfig, CalibratedFrame
from opencsi.core.types import (
    CalibrationSnapshot,
    CalibratedLink,
    QualityGrade,
)
from opencsi.testing import (
    generate_synthetic_csi,
    make_packet,
    make_positions,
    make_packet_dict,
)

# Synthetic data has high spectral entropy compared to real CSI.
# Relax the entropy threshold so the quiet detector accepts synthetic frames.
SYNTHETIC_ENTROPY_THRESHOLD = 6.0

# Frame counts for baseline and motion phases
QUIET_FRAMES = 80
MOTION_START = 80
TOTAL_FRAMES_WITH_MOTION = 110
MOTION_MAGNITUDE = 5.0

# Z-score thresholds
QUIET_ZSCORE_CEILING = 5.0
MOTION_ZSCORE_FLOOR = 2.0


def _make_config(**overrides) -> AutoCalibratorConfig:
    """Create an AutoCalibratorConfig tuned for synthetic data."""
    defaults = dict(entropy_low_threshold=SYNTHETIC_ENTROPY_THRESHOLD)
    defaults.update(overrides)
    return AutoCalibratorConfig(**defaults)


def _run_calibrator(
    n_frames=100,
    n_nodes=4,
    motion_frames=None,
    motion_magnitude=MOTION_MAGNITUDE,
    motion_links=None,
    config=None,
    seed=42,
):
    """Helper: create calibrator, feed synthetic frames, return (cal, results)."""
    config = config or _make_config()
    positions, frames, timestamps = generate_synthetic_csi(
        n_frames=n_frames,
        n_nodes=n_nodes,
        motion_frames=motion_frames,
        motion_links=motion_links,
        motion_magnitude=motion_magnitude,
        seed=seed,
    )
    cal = AutoCalibrator(positions, config=config)
    results = []
    for packets, ts in zip(frames, timestamps):
        results.append(cal.ingest(packets, timestamp=ts))
    return cal, results, positions


class TestCalibrationBaseline:
    """Tests for baseline learning from quiet-period data."""

    def test_ingest_returns_calibrated_frame(self):
        """ingest() returns a CalibratedFrame for every call."""
        positions = make_positions(4)
        cal = AutoCalibrator(positions, config=_make_config())
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
        config = _make_config(min_baseline_samples=10)
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
        cal = AutoCalibrator(positions, config=_make_config())

        last_frame = None
        for frame_packets, ts in zip(frames, timestamps):
            last_frame = cal.ingest(frame_packets, timestamp=ts)

        # After calibration settles, Z-scores should be low for reliable links
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
        cal = AutoCalibrator(positions, config=_make_config())

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
        """CalibratedFrame.motion_detected should be True during motion.

        motion_detected is a computed property that checks if any reliable
        link has combined_change > 2.0 and reliability >= 0.3.
        """
        motion_frame_indices = list(range(100, 120))

        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=120,
            n_nodes=6,  # More nodes = more links
            motion_frames=motion_frame_indices,
            motion_magnitude=8.0,
            seed=42,
        )
        cal = AutoCalibrator(positions, config=_make_config())

        any_motion_detected = False
        for i, (frame_packets, ts) in enumerate(zip(frames, timestamps)):
            result = cal.ingest(frame_packets, timestamp=ts)
            if i >= 100 and result.motion_detected:
                any_motion_detected = True

        # At least one frame during the motion period should detect motion
        assert any_motion_detected, (
            "motion_detected should be True for at least one frame during "
            "high-magnitude motion"
        )


class TestMultipleNodes:
    """Tests with different node counts."""

    def test_two_nodes_single_link(self):
        """Two nodes should produce exactly one link."""
        positions = make_positions(2)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions, config=_make_config())

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
        cal = AutoCalibrator(positions, config=_make_config())

        frame = cal.ingest(frames[0], timestamp=timestamps[0])
        # Should have links for all pairs in the frame
        assert frame.n_total_links > 0

    def test_all_directed_links_present(self):
        """With n nodes, each frame should contain n*(n-1) directed links."""
        n_nodes = 4
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=5, n_nodes=n_nodes, seed=42
        )
        expected_links = n_nodes * (n_nodes - 1)
        assert len(frames[0]) == expected_links


class TestCalibratedLinkFields:
    """Tests that CalibratedLink fields are populated correctly."""

    def test_link_has_mac_addresses(self):
        """Each CalibratedLink should have valid tx_mac and rx_mac."""
        positions = make_positions(3)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions, config=_make_config())

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
        cal = AutoCalibrator(positions, config=_make_config())

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
        cal = AutoCalibrator(positions, config=_make_config())

        for frame_packets, ts in zip(frames, timestamps):
            result = cal.ingest(frame_packets, timestamp=ts)

        for link in result.links.values():
            assert 0.0 <= link.reliability <= 1.0


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_packet_dict(self):
        """ingest() with empty dict should return empty frame."""
        positions = make_positions(2)
        cal = AutoCalibrator(positions, config=_make_config())

        frame = cal.ingest({}, timestamp=1.0)
        assert len(frame.links) == 0

    def test_single_frame_no_crash(self):
        """Single frame should not crash (not enough for baseline)."""
        positions = make_positions(4)
        cal = AutoCalibrator(positions, config=_make_config())

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
        cal = AutoCalibrator(positions, config=_make_config())

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
        """Same seed should produce identical I/Q data."""
        _, frames1, _ = generate_synthetic_csi(n_frames=10, seed=123)
        _, frames2, _ = generate_synthetic_csi(n_frames=10, seed=123)

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
        cal = AutoCalibrator(positions, config=_make_config())

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
        cal = AutoCalibrator(positions, config=_make_config())

        for fp, ts in zip(frames, timestamps):
            cal.ingest(fp, timestamp=ts)

        snapshot = cal.export_snapshot()
        json_str = snapshot.to_json()
        restored = CalibrationSnapshot.from_json(json_str)

        assert restored.version == snapshot.version
        assert len(restored.links) == len(snapshot.links)


class TestMakePacketDict:
    """Tests for the make_packet_dict convenience function."""

    def test_returns_correct_format(self):
        """make_packet_dict returns {(tx, rx): RawCSIPacket}."""
        from opencsi.core.types import RawCSIPacket

        d = make_packet_dict("AA:BB:CC:DD:EE:00", "AA:BB:CC:DD:EE:01", seed=42)

        assert len(d) == 1
        key = ("AA:BB:CC:DD:EE:00", "AA:BB:CC:DD:EE:01")
        assert key in d
        assert isinstance(d[key], RawCSIPacket)

    def test_works_with_ingest(self):
        """make_packet_dict output should be directly usable with ingest()."""
        positions = make_positions(2)
        nodes = list(positions.keys())
        cal = AutoCalibrator(positions, config=_make_config())

        packets = make_packet_dict(nodes[0], nodes[1])
        frame = cal.ingest(packets, timestamp=1.0)
        assert len(frame.links) == 1


class TestZScoreCorrectness:
    """Verify Z-scores are near zero during quiet and elevated during motion."""

    def test_z_scores_near_zero_for_all_reliable_links_during_quiet(self):
        """After baseline settles, every reliable link's Z-score < ceiling."""
        cal, results, _ = _run_calibrator(n_frames=120, motion_frames=None)

        last = results[-1]
        for link in last.links.values():
            if link.reliability > 0.3:
                assert link.combined_change < QUIET_ZSCORE_CEILING, (
                    f"Z-score {link.combined_change:.2f} too high during quiet "
                    f"for {link.tx_mac[-5:]}→{link.rx_mac[-5:]}"
                )

    def test_z_scores_elevated_during_motion(self):
        """Mean Z-score during motion should exceed mean Z during quiet."""
        cal, results, _ = _run_calibrator(
            n_frames=TOTAL_FRAMES_WITH_MOTION,
            motion_frames=range(MOTION_START, TOTAL_FRAMES_WITH_MOTION),
        )

        quiet_z = []
        for i in range(60, MOTION_START):
            for link in results[i].links.values():
                if link.reliability > 0.1:
                    quiet_z.append(link.combined_change)

        motion_z = []
        for i in range(MOTION_START, TOTAL_FRAMES_WITH_MOTION):
            for link in results[i].links.values():
                if link.reliability > 0.1:
                    motion_z.append(link.combined_change)

        if quiet_z and motion_z:
            assert np.mean(motion_z) > np.mean(quiet_z), (
                f"Mean motion Z ({np.mean(motion_z):.2f}) should exceed "
                f"mean quiet Z ({np.mean(quiet_z):.2f})"
            )

    def test_combined_change_exceeds_threshold_during_motion(self):
        """At least some links should have combined_change > threshold during motion."""
        cal, results, _ = _run_calibrator(
            n_frames=TOTAL_FRAMES_WITH_MOTION,
            motion_frames=range(MOTION_START, TOTAL_FRAMES_WITH_MOTION),
        )

        last_motion = results[-1]
        high_z_links = [
            link for link in last_motion.links.values()
            if link.combined_change > MOTION_ZSCORE_FLOOR
        ]
        assert len(high_z_links) > 0, (
            "No links exceeded Z-score threshold during motion injection"
        )


class TestReliabilityScoring:
    """Verify reliability scores are well-formed and quality grades assigned."""

    def test_reliability_in_valid_range(self):
        """All link reliabilities must be in [0, 1]."""
        cal, results, _ = _run_calibrator(n_frames=50)

        for link in results[-1].links.values():
            assert 0.0 <= link.reliability <= 1.0, (
                f"Reliability {link.reliability} out of [0, 1] range"
            )

    def test_quality_grades_are_enum_values(self):
        """Every link should have a QualityGrade enum value."""
        cal, results, _ = _run_calibrator(n_frames=50)

        for link in results[-1].links.values():
            assert isinstance(link.quality_grade, QualityGrade)

    def test_reliability_increases_with_more_samples(self):
        """More quiet samples should yield higher (or equal) mean reliability."""
        _, results_short, _ = _run_calibrator(n_frames=20)
        _, results_long, _ = _run_calibrator(n_frames=100)

        r_short = results_short[-1].mean_reliability
        r_long = results_long[-1].mean_reliability

        assert r_long >= r_short, (
            f"100-frame reliability ({r_long:.3f}) should be >= "
            f"20-frame reliability ({r_short:.3f})"
        )


class TestSnapshotRoundtrip:
    """Comprehensive CalibrationSnapshot export → JSON → reimport tests."""

    def test_json_roundtrip_preserves_link_count(self):
        """JSON export → import preserves the number of links."""
        cal, _, _ = _run_calibrator(n_frames=QUIET_FRAMES)

        snapshot = cal.export_snapshot()
        json_str = snapshot.to_json()

        parsed = json.loads(json_str)
        assert "links" in parsed

        restored = CalibrationSnapshot.from_json(json_str)
        assert len(restored.links) == len(snapshot.links)

    def test_json_roundtrip_preserves_version(self):
        """JSON roundtrip preserves the snapshot version."""
        cal, _, _ = _run_calibrator(n_frames=QUIET_FRAMES)

        snapshot = cal.export_snapshot()
        restored = CalibrationSnapshot.from_json(snapshot.to_json())
        assert restored.version == snapshot.version

    def test_json_roundtrip_preserves_amplitude_stats(self):
        """JSON roundtrip preserves per-link amplitude mean/std."""
        cal, _, _ = _run_calibrator(n_frames=QUIET_FRAMES)

        snapshot = cal.export_snapshot()
        restored = CalibrationSnapshot.from_json(snapshot.to_json())

        for link_key in snapshot.links:
            orig = snapshot.links[link_key]
            rest = restored.links[link_key]
            assert abs(orig.amplitude_mean - rest.amplitude_mean) < 1e-6
            assert abs(orig.amplitude_std - rest.amplitude_std) < 1e-6
            assert orig.quiet_samples == rest.quiet_samples
            assert orig.is_initialized == rest.is_initialized

    def test_from_snapshot_creates_functional_calibrator(self):
        """AutoCalibrator.from_snapshot() produces a working calibrator."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=QUIET_FRAMES, n_nodes=4, seed=42,
        )
        cal = AutoCalibrator(positions, config=_make_config())
        for packets, ts in zip(frames, timestamps):
            cal.ingest(packets, timestamp=ts)

        snapshot = cal.export_snapshot()
        cal2 = AutoCalibrator.from_snapshot(snapshot, positions, config=_make_config())

        # Feed one more frame — should not crash and should produce valid output
        _, extra_frames, extra_ts = generate_synthetic_csi(
            n_frames=1, n_nodes=4, seed=99,
        )
        frame = cal2.ingest(extra_frames[0], timestamp=extra_ts[0])
        assert isinstance(frame, CalibratedFrame)
        assert frame.n_total_links > 0


class TestMultiLink:
    """Verify multiple TX-RX pairs are calibrated independently."""

    def test_different_links_have_different_baselines(self):
        """Each link should develop its own baseline (different channel profiles)."""
        cal, _, _ = _run_calibrator(n_frames=QUIET_FRAMES, n_nodes=4)

        initialized_means = []
        for state in cal._link_states.values():
            if state.baseline.is_initialized:
                initialized_means.append(state.baseline.amplitude_mean_scalar)

        assert len(initialized_means) >= 2, (
            f"Expected >=2 initialized baselines, got {len(initialized_means)}"
        )
        unique_means = set(round(m, 4) for m in initialized_means)
        assert len(unique_means) > 1, (
            "All links have identical baselines — they should differ "
            "due to different synthetic channel profiles"
        )

    def test_six_nodes_produces_thirty_directed_links(self):
        """Six nodes should produce 6*5=30 directed links per frame."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=5, n_nodes=6, seed=42,
        )
        cal = AutoCalibrator(positions, config=_make_config())
        frame = cal.ingest(frames[0], timestamp=timestamps[0])
        assert frame.n_total_links == 30

    def test_motion_on_specific_links_affects_target_most(self):
        """Motion injected on specific links should affect those links most."""
        positions = make_positions(4)
        macs = list(positions.keys())
        target_link = (macs[0], macs[1])

        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=TOTAL_FRAMES_WITH_MOTION,
            n_nodes=4,
            motion_frames=range(MOTION_START, TOTAL_FRAMES_WITH_MOTION),
            motion_links=[target_link],
            motion_magnitude=MOTION_MAGNITUDE,
            seed=42,
        )
        cal = AutoCalibrator(positions, config=_make_config())
        for packets, ts in zip(frames, timestamps):
            result = cal.ingest(packets, timestamp=ts)

        last = result
        target_z = 0.0
        other_z = []
        for key, link in last.links.items():
            if key == target_link:
                target_z = link.combined_change
            else:
                other_z.append(link.combined_change)

        assert len(other_z) > 0, "Expected non-empty other_z list"
        assert target_z > 0, f"Expected target link Z > 0, got {target_z}"
        assert target_z > np.mean(other_z), (
            f"Target link Z ({target_z:.2f}) should exceed "
            f"other links mean Z ({np.mean(other_z):.2f})"
        )


# ============================================================================
# to_numpy() tests
# ============================================================================


class TestToNumpy:
    """Tests for CalibratedLink.to_numpy() and CalibratedFrame.to_numpy()."""

    def test_link_to_numpy_shape(self):
        """CalibratedLink.to_numpy() returns 1D array with correct length."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=50, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions, config=_make_config())
        for fp, ts in zip(frames, timestamps):
            result = cal.ingest(fp, timestamp=ts)

        assert len(result.links) > 0, "Expected non-empty links"
        link = next(iter(result.links.values()))
        arr = link.to_numpy()
        assert arr.ndim == 1
        assert arr.dtype == np.float64
        assert len(arr) == len(CalibratedLink.numpy_field_names())

    def test_link_to_numpy_values_match_fields(self):
        """to_numpy() values correspond to named fields."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=50, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions, config=_make_config())
        for fp, ts in zip(frames, timestamps):
            result = cal.ingest(fp, timestamp=ts)

        assert len(result.links) > 0, "Expected non-empty links"
        link = next(iter(result.links.values()))
        arr = link.to_numpy()
        names = CalibratedLink.numpy_field_names()

        idx = names.index('combined_change')
        assert arr[idx] == pytest.approx(link.combined_change)

        idx = names.index('reliability')
        assert arr[idx] == pytest.approx(link.reliability)

    def test_link_to_numpy_no_inf(self):
        """to_numpy() must not contain inf values (clamped for ML pipelines)."""
        link = CalibratedLink(
            tx_mac="AA:00", rx_mac="BB:00",
            baseline_age_s=float('inf'),
        )
        arr = link.to_numpy()
        assert np.all(np.isfinite(arr)), f"inf found in to_numpy: {arr}"

    def test_frame_to_numpy_shape(self):
        """CalibratedFrame.to_numpy() returns (n_links, n_features) array."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=50, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions, config=_make_config())
        for fp, ts in zip(frames, timestamps):
            result = cal.ingest(fp, timestamp=ts)

        data, keys = result.to_numpy()
        assert data.ndim == 2
        assert data.shape[0] == len(keys)
        assert data.shape[1] == len(CalibratedLink.numpy_field_names())
        assert data.shape[0] == len(result.links)

    def test_frame_to_numpy_min_reliability_filters(self):
        """min_reliability parameter filters low-reliability links."""
        positions, frames, timestamps = generate_synthetic_csi(
            n_frames=50, n_nodes=4, seed=42
        )
        cal = AutoCalibrator(positions, config=_make_config())
        for fp, ts in zip(frames, timestamps):
            result = cal.ingest(fp, timestamp=ts)

        data_all, keys_all = result.to_numpy(min_reliability=0.0)
        data_high, keys_high = result.to_numpy(min_reliability=0.99)
        assert len(keys_all) > 0, "Expected non-empty links at min_reliability=0.0"
        assert len(keys_high) <= len(keys_all)

    def test_frame_to_numpy_empty(self):
        """Empty frame returns (0, n_features) array."""
        positions = make_positions(2)
        cal = AutoCalibrator(positions, config=_make_config())
        frame = cal.ingest({}, timestamp=1.0)

        data, keys = frame.to_numpy()
        assert data.shape == (0, len(CalibratedLink.numpy_field_names()))
        assert keys == []

    def test_numpy_field_names_are_strings(self):
        """numpy_field_names() returns list of strings."""
        names = CalibratedLink.numpy_field_names()
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0


# ============================================================================
# JSON save/load round-trip tests
# ============================================================================


class TestJSONPersistence:
    """Tests for JSON-based save/load (replacing pickle)."""

    def test_save_load_roundtrip_json(self, tmp_path):
        """Save to JSON, load back, verify all fields survive."""
        cal, _, positions = _run_calibrator(n_frames=QUIET_FRAMES, n_nodes=4)
        state_path = str(tmp_path / "state.json")

        cal.save(state_path)

        # File should be valid JSON
        with open(state_path, 'r') as f:
            raw = json.load(f)
        assert raw['version'] == 7
        assert isinstance(raw['config'], dict)
        assert isinstance(raw['link_baselines'], dict)

        # Load back
        cal2 = AutoCalibrator.load(state_path, positions, config=_make_config())
        assert len(cal2._link_states) == len(cal._link_states)

        # Baselines should match
        for key in cal._link_states:
            orig = cal._link_states[key].baseline
            loaded = cal2._link_states[key].baseline
            assert abs(orig.amplitude_mean_scalar - loaded.amplitude_mean_scalar) < 1e-6
            assert abs(orig.amplitude_std_scalar - loaded.amplitude_std_scalar) < 1e-6
            assert orig.quiet_samples == loaded.quiet_samples
            assert orig.is_initialized == loaded.is_initialized

    def test_save_load_preserves_frame_counters(self, tmp_path):
        """Frame counters survive JSON round-trip."""
        cal, _, positions = _run_calibrator(n_frames=QUIET_FRAMES, n_nodes=4)
        state_path = str(tmp_path / "state.json")
        cal.save(state_path)

        cal2 = AutoCalibrator.load(state_path, positions, config=_make_config())
        assert cal2._total_frames_processed == cal._total_frames_processed
        assert cal2._quiet_frames == cal._quiet_frames
        assert cal2._active_frames == cal._active_frames

    def test_backward_compat_pickle_load(self, tmp_path):
        """Legacy pickle files can still be loaded."""
        import pickle

        cal, _, positions = _run_calibrator(n_frames=QUIET_FRAMES, n_nodes=4)

        # Manually create a pickle state file (simulating legacy format)
        from opencsi.core.calibrator_persistence import compact_baselines
        compacted = compact_baselines(cal)
        state = {
            'version': 6,
            'node_positions': {k: list(v) for k, v in positions.items()},
            'config': cal.config,  # Legacy: raw dataclass object
            'link_baselines': compacted,
            'sensitivity_learner': cal.sensitivity_learner.to_dict(),
            'antenna_learner': cal.antenna_learner.to_dict(),
            'frequency_collector': cal._passive_frequency_collector.to_dict(),
            'time_collector': cal._passive_time_collector.to_dict(),
            'total_frames': cal._total_frames_processed,
            'quiet_frames': cal._quiet_frames,
            'active_frames': cal._active_frames,
            'saved_at': 0.0,
            'calibrated_links': 0,
        }

        pkl_path = tmp_path / "legacy.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(state, f)

        # Load WITHOUT passing config — exercises the pickle config path
        cal2 = AutoCalibrator.load(str(pkl_path), positions)
        assert len(cal2._link_states) == len(cal._link_states)

        # Verify the raw dataclass config was preserved (not replaced with defaults)
        assert cal2.config.entropy_low_threshold == cal.config.entropy_low_threshold

    def test_loaded_calibrator_is_functional(self, tmp_path):
        """Loaded calibrator can ingest new frames without errors."""
        cal, _, positions = _run_calibrator(n_frames=QUIET_FRAMES, n_nodes=4)
        state_path = str(tmp_path / "state.json")
        cal.save(state_path)

        cal2 = AutoCalibrator.load(state_path, positions, config=_make_config())

        _, extra_frames, extra_ts = generate_synthetic_csi(
            n_frames=5, n_nodes=4, seed=99,
        )
        for packets, ts in zip(extra_frames, extra_ts):
            frame = cal2.ingest(packets, timestamp=ts)
            assert isinstance(frame, CalibratedFrame)

    def test_config_from_dict_roundtrip(self):
        """AutoCalibratorConfig.from_dict(cfg.to_dict()) reproduces config."""
        import dataclasses

        cfg = _make_config(baseline_learning_rate=0.42, min_baseline_samples=99)
        d = cfg.to_dict()
        cfg2 = AutoCalibratorConfig.from_dict(d)

        for field in dataclasses.fields(AutoCalibratorConfig):
            assert getattr(cfg, field.name) == getattr(cfg2, field.name), (
                f"Field {field.name} mismatch: {getattr(cfg, field.name)} != {getattr(cfg2, field.name)}"
            )

    def test_config_from_dict_ignores_unknown_keys(self):
        """from_dict() silently drops keys not in the dataclass."""
        d = {'baseline_learning_rate': 0.5, 'future_field': 'unknown'}
        cfg = AutoCalibratorConfig.from_dict(d)
        assert cfg.baseline_learning_rate == 0.5
        assert not hasattr(cfg, 'future_field')
