"""Coverage Area Estimator from node positions.

Computes room geometry from RSSI-derived node positions and
derives recommended detection thresholds.

References:
    - MDS-MAP Algorithm: Shang et al., IEEE SECON 2003
    - Convex Hull Boundary: Renold & Chandrakala, IEEE 2017
    - SMACOF Refinement: De Leeuw & Mair, 2009
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

if TYPE_CHECKING:
    from opencsi.geometry.distance import DistanceEstimator
    from opencsi.geometry.ndf import NDFResult
    from opencsi.geometry.position import PositionSolver

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants for threshold derivation
# -----------------------------------------------------------------------------

# Multiperson separation: fraction of shorter room dimension
SEPARATION_FRACTION = 0.4

# Grid resolution: fraction of shorter room dimension
GRID_FRACTION = 0.1

# Grid resolution bounds (meters)
MIN_GRID_RESOLUTION_M = 0.25
MAX_GRID_RESOLUTION_M = 1.0

# Multiperson separation bounds (meters)
MIN_SEPARATION_M = 1.5
MAX_SEPARATION_M = 5.0

# Default bounding box margin (fraction of dimensions)
DEFAULT_MARGIN_FRACTION = 0.1

# Minimum nodes for 2D coverage estimation
MIN_NODES_FOR_HULL = 3


@dataclass
class CoverageEstimate:
    """Estimated coverage area metrics.

    Attributes:
        hull_vertices: Indices of nodes forming the convex hull boundary.
        hull_area_m2: Area enclosed by the convex hull in square meters.
        bbox_min: Minimum corner of axis-aligned bounding box (x_min, y_min).
        bbox_max: Maximum corner of axis-aligned bounding box (x_max, y_max).
        bbox_width_m: Width of bounding box in meters.
        bbox_height_m: Height of bounding box in meters.
        recommended_multiperson_separation_m: Recommended threshold for merging
            nearby peaks as same person, based on room dimensions.
        recommended_grid_resolution_m: Recommended grid cell size for position
            estimation, based on room dimensions.
        position_stress: MDS stress value (lower = better position fit).
            0.0 if positions were user-provided rather than RSSI-derived.
        n_boundary_nodes: Number of nodes on the convex hull boundary.
        n_interior_nodes: Number of nodes inside the convex hull.
    """

    hull_vertices: List[int]
    hull_area_m2: float
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    bbox_width_m: float
    bbox_height_m: float
    recommended_multiperson_separation_m: float
    recommended_grid_resolution_m: float
    position_stress: float
    n_boundary_nodes: int
    n_interior_nodes: int
    sensing_ndf: Optional[NDFResult] = field(default=None, repr=False)
    """Spatial sensing degrees of freedom analysis. Computed automatically
    when `compute_ndf=True` is passed to the estimator. Provides the number
    of independent spatial features the mesh can resolve (NDFResult.ndf),
    the singular value spectrum, and related metrics."""

    @property
    def room_area_m2(self) -> float:
        """Approximate room area from bounding box."""
        return self.bbox_width_m * self.bbox_height_m

    @property
    def shorter_dimension_m(self) -> float:
        """Shorter dimension of the bounding box."""
        return min(self.bbox_width_m, self.bbox_height_m)

    @property
    def longer_dimension_m(self) -> float:
        """Longer dimension of the bounding box."""
        return max(self.bbox_width_m, self.bbox_height_m)


class CoverageAreaEstimator:
    """Estimates coverage area and derives thresholds from node positions.

    This class takes node positions (either user-provided or RSSI-derived)
    and computes room geometry metrics plus recommended detection thresholds
    that scale with room size.

    Workflow:
        1. Take node positions (from MDS or user-provided)
        2. Compute convex hull to find boundary nodes
        3. Estimate room dimensions from bounding box
        4. Derive recommended detection thresholds

    Threshold derivation rationale:
        - multiperson_separation = 0.4 × min(width, height)
          A person's detection footprint is ~0.5-1m. Two people can't be
          closer than ~1m without physical contact. In a 5×4m room:
          0.4 × 4m = 1.6m separation. In a 10×8m room: 0.4 × 8m = 3.2m.

        - grid_resolution = 0.1 × min(width, height), clamped to [0.25, 1.0]
          Finer grids for smaller rooms, coarser for larger rooms.

    Usage:
        estimator = CoverageAreaEstimator()

        # From known positions
        coverage = estimator.estimate_from_positions(positions)

        # From RSSI (full pipeline)
        coverage, positions = estimator.estimate_from_rssi_matrix(
            rssi_matrix, distance_estimator, position_solver
        )
    """

    def __init__(
        self,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
        compute_ndf: bool = False,
    ):
        """Initialize the coverage area estimator.

        Args:
            margin_fraction: Expand bounding box by this fraction (default 10%)
                to account for sensing beyond node positions. Nodes typically
                don't sit exactly at room boundaries.
            compute_ndf: If True, also compute the mesh sensing degrees of
                freedom (NDF) via SVD of the Fresnel-weighted sensing matrix.
                This adds ~0.1s for typical meshes (14 nodes, 0.25m grid).
        """
        self.margin_fraction = margin_fraction
        self.compute_ndf = compute_ndf

    def estimate_from_positions(
        self,
        positions: np.ndarray,
        mds_stress: float = 0.0,
    ) -> CoverageEstimate:
        """Estimate coverage area from node positions.

        Args:
            positions: (N, 2) array of (x, y) node positions in meters.
            mds_stress: Stress value from MDS position solving (for confidence).
                Use 0.0 if positions are user-provided ground truth.

        Returns:
            CoverageEstimate with geometry and recommended thresholds.

        Raises:
            ValueError: If fewer than 3 positions are provided.
        """
        positions = np.asarray(positions, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"positions must be (N, 2) array, got shape {positions.shape}"
            )

        n_nodes = positions.shape[0]
        if n_nodes < MIN_NODES_FOR_HULL:
            raise ValueError(
                f"Need at least {MIN_NODES_FOR_HULL} nodes for coverage estimation, "
                f"got {n_nodes}"
            )

        # Compute convex hull
        hull_vertices, hull_area = self._compute_convex_hull(positions)

        # Compute bounding box with margin
        bbox_min, bbox_max = self._compute_bounding_box(positions)

        width = bbox_max[0] - bbox_min[0]
        height = bbox_max[1] - bbox_min[1]

        # Derive thresholds
        separation, grid_res = self._derive_thresholds(width, height)

        # Count boundary vs interior nodes
        n_boundary = len(hull_vertices)
        n_interior = n_nodes - n_boundary

        logger.info(
            "Coverage estimate: %.1f × %.1f m (area=%.1f m²), "
            "%d boundary + %d interior nodes",
            width,
            height,
            hull_area,
            n_boundary,
            n_interior,
        )
        logger.info(
            "Recommended thresholds: multiperson_separation=%.2f m, "
            "grid_resolution=%.2f m",
            separation,
            grid_res,
        )

        # Optionally compute NDF
        ndf_result = None
        if self.compute_ndf:
            from opencsi.geometry.ndf import analyze_mesh_topology
            try:
                ndf_result = analyze_mesh_topology(
                    positions,
                    grid_resolution_m=grid_res,
                    room_bounds=(bbox_min, bbox_max),
                )
                logger.info("Sensing NDF = %d from %d links", ndf_result.ndf, ndf_result.num_links)
            except Exception:
                logger.warning("NDF computation failed", exc_info=True)

        return CoverageEstimate(
            hull_vertices=hull_vertices,
            hull_area_m2=hull_area,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_width_m=width,
            bbox_height_m=height,
            recommended_multiperson_separation_m=separation,
            recommended_grid_resolution_m=grid_res,
            position_stress=mds_stress,
            n_boundary_nodes=n_boundary,
            n_interior_nodes=n_interior,
            sensing_ndf=ndf_result,
        )

    def estimate_from_rssi_matrix(
        self,
        rssi_matrix: np.ndarray,
        distance_estimator: DistanceEstimator,
        position_solver: PositionSolver,
    ) -> Tuple[CoverageEstimate, np.ndarray]:
        """Full pipeline: RSSI → distance → positions → coverage.

        Chains the DistanceEstimator and PositionSolver to convert an RSSI
        matrix into positions, then estimates coverage area.

        Args:
            rssi_matrix: (N, N) symmetric RSSI matrix in dBm.
                Diagonal should be 0 (self-link). Missing values should be NaN.
            distance_estimator: Configured DistanceEstimator instance.
            position_solver: Configured PositionSolver instance.

        Returns:
            Tuple of:
                - CoverageEstimate with geometry and thresholds
                - (N, 2) array of estimated positions

        Raises:
            ValueError: If RSSI matrix is invalid or positions can't be solved.
        """
        rssi_matrix = np.asarray(rssi_matrix, dtype=np.float64)

        if rssi_matrix.ndim != 2:
            raise ValueError(
                f"rssi_matrix must be 2D, got shape {rssi_matrix.shape}"
            )

        if rssi_matrix.shape[0] != rssi_matrix.shape[1]:
            raise ValueError(
                f"rssi_matrix must be square, got shape {rssi_matrix.shape}"
            )

        n_nodes = rssi_matrix.shape[0]
        if n_nodes < MIN_NODES_FOR_HULL:
            raise ValueError(
                f"Need at least {MIN_NODES_FOR_HULL} nodes, got {n_nodes}"
            )

        # Step 1: RSSI → distance
        logger.debug("Converting RSSI matrix to distances...")
        distance_matrix = distance_estimator.rssi_matrix_to_distances(rssi_matrix)

        # Step 2: distance → positions via SMACOF
        logger.debug("Solving positions via SMACOF...")
        mds_result = position_solver.solve_smacof(distance_matrix)

        if not mds_result.converged:
            logger.warning(
                "MDS did not converge (stress=%.4f, iterations=%d)",
                mds_result.stress,
                mds_result.iterations,
            )

        if mds_result.outlier_indices:
            logger.warning(
                "MDS detected %d outlier node(s): %s",
                len(mds_result.outlier_indices),
                mds_result.outlier_indices,
            )

        # Align positions to canonical orientation
        positions = position_solver.align_to_reference(mds_result.positions)

        # Step 3: positions → coverage estimate
        coverage = self.estimate_from_positions(
            positions,
            mds_stress=mds_result.stress,
        )

        return coverage, positions

    def _compute_convex_hull(
        self,
        positions: np.ndarray,
    ) -> Tuple[List[int], float]:
        """Compute convex hull vertices and area.

        Args:
            positions: (N, 2) array of positions.

        Returns:
            Tuple of (hull_vertices, hull_area_m2).
            If points are collinear, returns all points as boundary with area 0.
        """
        try:
            hull = ConvexHull(positions)
            return list(hull.vertices), float(hull.volume)  # 2D: volume = area
        except QhullError:
            # Points are collinear or degenerate
            logger.warning(
                "Convex hull failed (collinear points?), "
                "treating all nodes as boundary"
            )
            return list(range(len(positions))), 0.0

    def _compute_bounding_box(
        self,
        positions: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute axis-aligned bounding box with margin.

        The margin accounts for the fact that nodes don't typically sit
        exactly at room boundaries—the sensing area extends beyond them.

        Args:
            positions: (N, 2) array of positions.

        Returns:
            Tuple of (bbox_min, bbox_max) where each is (x, y).
        """
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)

        # Compute raw dimensions
        width = x_max - x_min
        height = y_max - y_min

        # Handle degenerate cases (all nodes in a line or point)
        # Expand to minimum 1m dimension centered at the midpoint
        if width < 1e-6:
            logger.warning("All nodes have same x-coordinate, using default width=1m")
            x_center = x_min
            x_min = x_center - 0.5
            x_max = x_center + 0.5
            width = 1.0
        if height < 1e-6:
            logger.warning("All nodes have same y-coordinate, using default height=1m")
            y_center = y_min
            y_min = y_center - 0.5
            y_max = y_center + 0.5
            height = 1.0

        # Apply margin
        margin_x = width * self.margin_fraction
        margin_y = height * self.margin_fraction

        bbox_min = (x_min - margin_x, y_min - margin_y)
        bbox_max = (x_max + margin_x, y_max + margin_y)

        return bbox_min, bbox_max

    def _derive_thresholds(
        self,
        width: float,
        height: float,
    ) -> Tuple[float, float]:
        """Derive multiperson separation and grid resolution.

        Thresholds scale with room size:
        - Multiperson separation: 40% of shorter dimension
        - Grid resolution: 10% of shorter dimension

        Both are clamped to reasonable absolute bounds.

        Args:
            width: Room width in meters.
            height: Room height in meters.

        Returns:
            Tuple of (multiperson_separation_m, grid_resolution_m).
        """
        shorter_dim = min(width, height)

        # Multiperson separation: 40% of shorter dimension
        # Rationale: In a 5m wide room, 2m separation is reasonable
        #            In a 3m wide room, 1.2m separation makes sense
        separation = shorter_dim * SEPARATION_FRACTION
        separation = float(np.clip(separation, MIN_SEPARATION_M, MAX_SEPARATION_M))

        # Grid resolution: 10% of shorter dimension, clamped
        grid_res = shorter_dim * GRID_FRACTION
        grid_res = float(np.clip(grid_res, MIN_GRID_RESOLUTION_M, MAX_GRID_RESOLUTION_M))

        return separation, grid_res
