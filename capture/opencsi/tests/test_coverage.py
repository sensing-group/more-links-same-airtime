"""Tests for coverage area estimation.

Tests the CoverageAreaEstimator which derives room geometry and
recommended detection thresholds from node positions.
"""

import numpy as np
import pytest

from opencsi.geometry.coverage import (
    CoverageAreaEstimator,
    CoverageEstimate,
    MIN_SEPARATION_M,
    MAX_SEPARATION_M,
    MIN_GRID_RESOLUTION_M,
    MAX_GRID_RESOLUTION_M,
    SEPARATION_FRACTION,
    GRID_FRACTION,
    MIN_NODES_FOR_HULL,
)
from opencsi.geometry.distance import DistanceEstimator
from opencsi.geometry.position import PositionSolver


class TestCoverageEstimate:
    """Tests for the CoverageEstimate dataclass."""

    def test_coverage_estimate_properties(self):
        """Test computed properties of CoverageEstimate."""
        estimate = CoverageEstimate(
            hull_vertices=[0, 1, 2, 3],
            hull_area_m2=20.0,
            bbox_min=(0.0, 0.0),
            bbox_max=(5.0, 4.0),
            bbox_width_m=5.0,
            bbox_height_m=4.0,
            recommended_multiperson_separation_m=1.6,
            recommended_grid_resolution_m=0.4,
            position_stress=0.05,
            n_boundary_nodes=4,
            n_interior_nodes=0,
        )

        assert estimate.room_area_m2 == pytest.approx(20.0)
        assert estimate.shorter_dimension_m == pytest.approx(4.0)
        assert estimate.longer_dimension_m == pytest.approx(5.0)


class TestCoverageAreaEstimator:
    """Tests for the CoverageAreaEstimator class."""

    def test_coverage_from_square_room(self):
        """4 nodes at corners of 5×4m room."""
        positions = np.array([
            [0, 0],
            [5, 0],
            [5, 4],
            [0, 4],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        # Check dimensions (no margin)
        assert coverage.bbox_width_m == pytest.approx(5.0)
        assert coverage.bbox_height_m == pytest.approx(4.0)

        # Check hull
        assert coverage.n_boundary_nodes == 4
        assert coverage.n_interior_nodes == 0
        assert coverage.hull_area_m2 == pytest.approx(20.0)

        # Check thresholds: 40% of shorter dimension (4m) = 1.6m
        expected_separation = 4.0 * SEPARATION_FRACTION
        assert coverage.recommended_multiperson_separation_m == pytest.approx(
            expected_separation, rel=0.01
        )

        # Grid: 10% of 4m = 0.4m
        expected_grid = 4.0 * GRID_FRACTION
        assert coverage.recommended_grid_resolution_m == pytest.approx(
            expected_grid, rel=0.01
        )

    def test_coverage_with_margin(self):
        """Test that margin expands bounding box correctly."""
        positions = np.array([
            [0, 0],
            [5, 0],
            [5, 4],
            [0, 4],
        ], dtype=np.float64)

        # 10% margin
        estimator = CoverageAreaEstimator(margin_fraction=0.1)
        coverage = estimator.estimate_from_positions(positions)

        # Width should be 5 + 2*(5*0.1) = 6
        assert coverage.bbox_width_m == pytest.approx(6.0)
        # Height should be 4 + 2*(4*0.1) = 4.8
        assert coverage.bbox_height_m == pytest.approx(4.8)

    def test_coverage_with_interior_nodes(self):
        """Room with nodes not all on boundary."""
        positions = np.array([
            [0, 0],    # corner
            [5, 0],    # corner
            [5, 4],    # corner
            [0, 4],    # corner
            [2.5, 2],  # interior
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        assert coverage.n_boundary_nodes == 4
        assert coverage.n_interior_nodes == 1
        # Interior node doesn't affect bounding box
        assert coverage.bbox_width_m == pytest.approx(5.0)
        assert coverage.bbox_height_m == pytest.approx(4.0)

    def test_threshold_bounds_small_room(self):
        """Tiny room should clamp separation to minimum."""
        # 2×2m room (very small)
        positions = np.array([
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        # 40% of 2m = 0.8m, but min is 1.5m
        assert coverage.recommended_multiperson_separation_m == pytest.approx(
            MIN_SEPARATION_M
        )

        # 10% of 2m = 0.2m, but min is 0.25m
        assert coverage.recommended_grid_resolution_m == pytest.approx(
            MIN_GRID_RESOLUTION_M
        )

    def test_threshold_bounds_large_room(self):
        """Large room should clamp separation to maximum."""
        # 20×20m room (large)
        positions = np.array([
            [0, 0],
            [20, 0],
            [20, 20],
            [0, 20],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        # 40% of 20m = 8m, but max is 5m
        assert coverage.recommended_multiperson_separation_m == pytest.approx(
            MAX_SEPARATION_M
        )

        # 10% of 20m = 2m, but max is 1.0m
        assert coverage.recommended_grid_resolution_m == pytest.approx(
            MAX_GRID_RESOLUTION_M
        )

    def test_typical_room_sizes(self):
        """Test various typical room sizes."""
        test_cases = [
            # (width, height, expected_separation, expected_grid)
            (3.0, 3.0, 1.5, 0.3),    # Small room, separation clipped to min
            (4.0, 4.0, 1.6, 0.4),    # ~1.6m separation, 0.4m grid
            (5.0, 4.0, 1.6, 0.4),    # Rectangular, uses shorter dim
            (6.0, 5.0, 2.0, 0.5),    # Medium room
            (8.0, 6.0, 2.4, 0.6),    # Larger room
            (10.0, 8.0, 3.2, 0.8),   # Large room
            (12.0, 10.0, 4.0, 1.0),  # Very large, grid clipped to max
        ]

        for width, height, exp_sep, exp_grid in test_cases:
            positions = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ], dtype=np.float64)

            estimator = CoverageAreaEstimator(margin_fraction=0.0)
            coverage = estimator.estimate_from_positions(positions)

            # Clamp expected values
            exp_sep = np.clip(exp_sep, MIN_SEPARATION_M, MAX_SEPARATION_M)
            exp_grid = np.clip(exp_grid, MIN_GRID_RESOLUTION_M, MAX_GRID_RESOLUTION_M)

            assert coverage.recommended_multiperson_separation_m == pytest.approx(
                exp_sep, rel=0.01
            ), f"Failed for {width}×{height}m room"
            assert coverage.recommended_grid_resolution_m == pytest.approx(
                exp_grid, rel=0.01
            ), f"Failed for {width}×{height}m room"

    def test_collinear_nodes(self):
        """Collinear nodes should be handled gracefully."""
        # All nodes on a line
        positions = np.array([
            [0, 0],
            [2, 0],
            [4, 0],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        # Should not raise, but hull area should be 0
        assert coverage.hull_area_m2 == pytest.approx(0.0)
        # All nodes treated as boundary
        assert coverage.n_boundary_nodes == 3
        # Should use default height of 1m for degenerate case
        assert coverage.bbox_height_m == pytest.approx(1.0)

    def test_minimum_nodes_required(self):
        """Should raise error with fewer than 3 nodes."""
        positions = np.array([
            [0, 0],
            [5, 0],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator()

        with pytest.raises(ValueError, match=f"at least {MIN_NODES_FOR_HULL}"):
            estimator.estimate_from_positions(positions)

    def test_invalid_shape(self):
        """Should raise error for invalid position array shape."""
        estimator = CoverageAreaEstimator()

        # 1D array
        with pytest.raises(ValueError, match="must be .N, 2. array"):
            estimator.estimate_from_positions(np.array([0, 1, 2, 3]))

        # 3D coordinates
        with pytest.raises(ValueError, match="must be .N, 2. array"):
            estimator.estimate_from_positions(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))


class TestCoverageFromRSSI:
    """Tests for the full RSSI → positions → coverage pipeline."""

    def test_coverage_from_rssi_pipeline(self):
        """Full pipeline: synthetic RSSI → positions → thresholds."""
        # Generate synthetic positions
        true_positions = np.array([
            [0, 0],
            [5, 0],
            [5, 4],
            [0, 4],
        ], dtype=np.float64)

        # Generate synthetic RSSI from positions using path loss model
        n_nodes = len(true_positions)
        rssi_matrix = np.zeros((n_nodes, n_nodes))

        distance_estimator = DistanceEstimator(rssi_ref=-40.0, path_loss_exponent=2.5)

        # Compute pairwise distances and convert to RSSI
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = np.linalg.norm(true_positions[i] - true_positions[j])
                    rssi_matrix[i, j] = distance_estimator.model.distance_to_rssi(dist)
                else:
                    rssi_matrix[i, j] = 0  # Self-link

        # Run full pipeline
        position_solver = PositionSolver()
        estimator = CoverageAreaEstimator(margin_fraction=0.0)

        coverage, positions = estimator.estimate_from_rssi_matrix(
            rssi_matrix, distance_estimator, position_solver
        )

        # MDS recovers positions up to rotation/reflection/translation
        # Check that the pairwise distances are preserved
        recovered_distances = np.zeros((n_nodes, n_nodes))
        true_distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    recovered_distances[i, j] = np.linalg.norm(positions[i] - positions[j])
                    true_distances[i, j] = np.linalg.norm(true_positions[i] - true_positions[j])

        # Distances should be close (within 10% for synthetic data)
        np.testing.assert_allclose(
            recovered_distances, true_distances, rtol=0.1, atol=0.5
        )

        # Coverage dimensions should be close to original
        assert coverage.bbox_width_m == pytest.approx(5.0, rel=0.2)
        assert coverage.bbox_height_m == pytest.approx(4.0, rel=0.2)

        # Stress should be low for clean synthetic data
        assert coverage.position_stress < 0.2

    def test_coverage_from_rssi_with_noise(self):
        """Pipeline with noisy RSSI measurements."""
        # Generate synthetic positions
        true_positions = np.array([
            [0, 0],
            [5, 0],
            [5, 4],
            [0, 4],
            [2.5, 2],  # Interior node
        ], dtype=np.float64)

        n_nodes = len(true_positions)
        distance_estimator = DistanceEstimator(rssi_ref=-40.0, path_loss_exponent=2.5)

        # Generate RSSI with Gaussian noise
        np.random.seed(42)
        rssi_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = np.linalg.norm(true_positions[i] - true_positions[j])
                    clean_rssi = distance_estimator.model.distance_to_rssi(dist)
                    # Add noise (typical RSSI std is 2-4 dB)
                    rssi_matrix[i, j] = clean_rssi + np.random.normal(0, 3.0)
                else:
                    rssi_matrix[i, j] = 0

        # Symmetrize (average forward/reverse)
        rssi_matrix = (rssi_matrix + rssi_matrix.T) / 2

        position_solver = PositionSolver()
        estimator = CoverageAreaEstimator(margin_fraction=0.1)

        coverage, positions = estimator.estimate_from_rssi_matrix(
            rssi_matrix, distance_estimator, position_solver
        )

        # With 3dB noise, RSSI-based distances can have 10-30% error
        # MDS may produce distorted positions, so we check:
        # 1. Result is in reasonable range (not completely wrong)
        # 2. We get valid coverage metrics

        # Dimensions should be positive and within order of magnitude
        assert 2.0 < coverage.bbox_width_m < 15.0
        assert 2.0 < coverage.bbox_height_m < 12.0

        # The pipeline should complete without error
        assert coverage.position_stress >= 0.0

        # Thresholds should be derived (in valid range)
        assert MIN_SEPARATION_M <= coverage.recommended_multiperson_separation_m <= MAX_SEPARATION_M
        assert MIN_GRID_RESOLUTION_M <= coverage.recommended_grid_resolution_m <= MAX_GRID_RESOLUTION_M

    def test_rssi_matrix_validation(self):
        """Test validation of RSSI matrix inputs."""
        distance_estimator = DistanceEstimator()
        position_solver = PositionSolver()
        estimator = CoverageAreaEstimator()

        # Non-square matrix
        with pytest.raises(ValueError, match="must be square"):
            estimator.estimate_from_rssi_matrix(
                np.array([[1, 2, 3], [4, 5, 6]]),
                distance_estimator,
                position_solver,
            )

        # Too few nodes
        with pytest.raises(ValueError, match=f"at least {MIN_NODES_FOR_HULL}"):
            estimator.estimate_from_rssi_matrix(
                np.array([[-40, -50], [-50, -40]]),
                distance_estimator,
                position_solver,
            )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_nodes_same_position(self):
        """All nodes at same position should handle gracefully."""
        positions = np.array([
            [2.5, 2.0],
            [2.5, 2.0],
            [2.5, 2.0],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        # Should use default dimensions for degenerate case
        assert coverage.bbox_width_m == pytest.approx(1.0)
        assert coverage.bbox_height_m == pytest.approx(1.0)
        assert coverage.hull_area_m2 == pytest.approx(0.0)

    def test_triangle_nodes(self):
        """Minimum valid case: 3 nodes forming a triangle."""
        positions = np.array([
            [0, 0],
            [4, 0],
            [2, 3],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        assert coverage.n_boundary_nodes == 3
        assert coverage.n_interior_nodes == 0
        assert coverage.bbox_width_m == pytest.approx(4.0)
        assert coverage.bbox_height_m == pytest.approx(3.0)
        # Triangle area = 0.5 * base * height = 0.5 * 4 * 3 = 6
        assert coverage.hull_area_m2 == pytest.approx(6.0)

    def test_non_convex_arrangement(self):
        """Test L-shaped or non-convex node arrangements."""
        # L-shaped arrangement
        positions = np.array([
            [0, 0],
            [3, 0],
            [3, 2],
            [1, 2],
            [1, 4],
            [0, 4],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator(margin_fraction=0.0)
        coverage = estimator.estimate_from_positions(positions)

        # Bounding box covers full extent
        assert coverage.bbox_width_m == pytest.approx(3.0)
        assert coverage.bbox_height_m == pytest.approx(4.0)

        # Convex hull will include outer nodes
        # Interior nodes (if any) would be counted separately
        assert coverage.n_boundary_nodes >= 4  # At least the corners

    def test_mds_stress_passed_through(self):
        """Test that MDS stress is correctly passed to coverage estimate."""
        positions = np.array([
            [0, 0],
            [5, 0],
            [5, 4],
            [0, 4],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator()
        coverage = estimator.estimate_from_positions(positions, mds_stress=0.123)

        assert coverage.position_stress == pytest.approx(0.123)

    def test_default_stress_is_zero(self):
        """Test that default stress is 0 for user-provided positions."""
        positions = np.array([
            [0, 0],
            [5, 0],
            [5, 4],
            [0, 4],
        ], dtype=np.float64)

        estimator = CoverageAreaEstimator()
        coverage = estimator.estimate_from_positions(positions)

        assert coverage.position_stress == pytest.approx(0.0)
