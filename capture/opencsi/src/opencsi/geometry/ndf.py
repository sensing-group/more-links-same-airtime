"""
Numerical Degrees of Freedom (NDF) Analysis for Wireless Mesh Sensing.

Computes the spatial sensing degrees of freedom of an irregular wireless mesh
by constructing the Fresnel-weighted sensing matrix W and analyzing its
singular value spectrum.

Theory:
    Given N nodes at positions {p_i}, there are L = N(N-1)/2 unique links.
    Each link (i,j) measures a path-integrated perturbation through the
    sensing region, weighted by the Fresnel zone profile. The sensing matrix
    W has shape (L, V) where V is the number of voxels in the discretized
    region. The NDF equals the number of significant singular values of W
    (analogous to Bucci & Franceschetti's result for continuous apertures,
    but generalized to discrete irregular link topologies).

References:
    - Bucci & Franceschetti (1989): NDF of scattered fields from continuous apertures
    - Miller (2000): Communication modes between volumes via Green's function SVD
    - Wilson & Patwari (2010): Radio Tomographic Imaging weight model
    - Slepian (1961): ~2WT degrees of freedom for bandlimited signals

Usage:
    from opencsi.geometry.ndf import analyze_mesh_topology

    result = analyze_mesh_topology(node_positions)
    print(f"NDF = {result['ndf']} from {result['num_links']} links")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import svds

from opencsi.core.config import WAVELENGTH_2_4GHZ_M

# Lazy imports for optional RSSI pipeline (avoid circular imports)
# DistanceEstimator and PositionSolver are imported inside
# analyze_from_rssi_matrix() to keep ndf.py usable without them.

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Grid discretization
DEFAULT_GRID_RESOLUTION_M = 0.25
"""Default voxel size for the sensing region grid (meters).
Quarter-wavelength at 2.4 GHz is ~3cm, but 25cm provides practical
spatial resolution matching typical body dimensions."""

# Fresnel weight model (same physics as cs_rti.py)
FRESNEL_SIGMA_FRACTION = 0.5
"""Fresnel zone standard deviation = r_F * this fraction.
The Gaussian weight drops to e^(-2) at the first Fresnel radius."""

FRESNEL_TRUNCATION_SIGMA = 3.0
"""Truncate Gaussian weights beyond this many sigma for sparsity.
At 3 sigma, weight is exp(-4.5) ≈ 0.011, below perceptual significance."""

ENDPOINT_WEIGHT_SCALE = 4.0
"""Parabolic endpoint weight: scale * t * (1-t). Peaks at t=0.5 (midpoint),
zero at endpoints. Models the ellipsoidal Fresnel zone shape."""

# NDF threshold
DEFAULT_NDF_THRESHOLD = 0.01
"""Singular values below this fraction of sigma_max are considered noise.
Defines the effective rank of the sensing matrix."""

# Bounding box margin
ROOM_MARGIN_FRACTION = 0.1
"""Expand bounding box by this fraction beyond outermost nodes.
Accounts for sensing region extending beyond node positions."""

DEGENERATE_DIMENSION_M = 1.0
"""Minimum bounding box dimension (meters) to handle collinear nodes."""

MIN_LINK_LENGTH_M = 0.1
"""Minimum valid link length (meters). Links shorter than this are skipped
because the Fresnel zone model breaks down in the near field."""


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class NDFResult:
    """Result of mesh topology NDF analysis.

    Attributes:
        ndf: Number of spatial degrees of freedom (effective rank of W).
        singular_values: Full singular value spectrum in descending order.
        num_nodes: Number of nodes in the mesh.
        num_links: Number of valid links (after filtering short links).
        num_voxels: Number of grid cells in the sensing region.
        matrix_shape: Shape of the sensing matrix W as (links, voxels).
        grid_resolution_m: Grid cell size used for discretization.
        room_bounds: Bounding box as ((x_min, y_min), (x_max, y_max)).
        wavelength_m: Signal wavelength used for Fresnel zone computation.
        threshold: Singular value threshold used for NDF computation.
    """
    ndf: int
    singular_values: np.ndarray
    num_nodes: int
    num_links: int
    num_voxels: int
    matrix_shape: Tuple[int, int]
    grid_resolution_m: float
    room_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    wavelength_m: float
    threshold: float

    @property
    def redundancy_ratio(self) -> float:
        """Ratio of links to NDF. Values >> 1 indicate link redundancy."""
        if self.ndf == 0:
            return float('inf')
        return self.num_links / self.ndf

    @property
    def efficiency(self) -> float:
        """Fraction of grid cells that are independently resolvable (NDF/V)."""
        if self.num_voxels == 0:
            return 0.0
        return self.ndf / self.num_voxels


# =============================================================================
# Core functions
# =============================================================================

def build_sensing_matrix(
    node_positions: np.ndarray,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    room_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M,
) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build the Fresnel-weighted sensing matrix W for a node mesh.

    Constructs a sparse matrix where each row corresponds to a link and
    each column to a grid cell. The weight W[l, g] represents how much
    a perturbation at grid cell g affects the signal on link l, modeled
    by the first Fresnel zone profile.

    The weight model for link (tx, rx) at grid cell g:
        W[l, g] = exp(-d_perp^2 / (2 * sigma_F^2)) * 4*t*(1-t)

    where:
        sigma_F = (FRESNEL_SIGMA_FRACTION) * sqrt(wavelength * link_length / 4)
        d_perp = perpendicular distance from cell center to link axis
        t = projection parameter along link (0 at tx, 1 at rx)

    Note: This uses a fixed sigma_F (midpoint value) with a separate 4t(1-t)
    endpoint taper, rather than the physically exact position-dependent
    sigma_F(t) = sqrt(wavelength * link_length * t * (1-t)) / 2. The two
    models agree at the link midpoint (t=0.5) and differ most near endpoints.
    Empirical testing shows NDF differs by at most 2 between models across
    all tested topologies, confirming NDF is robust to the weight model choice.
    The fixed-sigma approximation is used for consistency with the CS-RTI
    reconstruction module (cs_rti.py).

    Args:
        node_positions: (N, 2) array of node (x, y) positions in meters.
        grid_resolution_m: Grid cell size in meters.
        room_bounds: Optional ((x_min, y_min), (x_max, y_max)). If None,
            auto-computed from node positions with margin.
        wavelength_m: Signal wavelength in meters.

    Returns:
        Tuple of:
            - W: Sparse CSR matrix of shape (num_links, num_cells)
            - grid_x: 1D array of grid cell center x-coordinates
            - grid_y: 1D array of grid cell center y-coordinates

    Raises:
        ValueError: If fewer than 2 nodes provided or positions wrong shape.
    """
    node_positions = np.asarray(node_positions, dtype=np.float64)

    if node_positions.ndim != 2 or node_positions.shape[1] != 2:
        raise ValueError(
            f"node_positions must be (N, 2), got shape {node_positions.shape}"
        )

    n_nodes = node_positions.shape[0]
    if n_nodes < 2:
        raise ValueError(f"Need at least 2 nodes, got {n_nodes}")

    # Auto-compute room bounds if not provided
    if room_bounds is None:
        room_bounds = _compute_room_bounds(node_positions)

    (x_min, y_min), (x_max, y_max) = room_bounds

    # Build grid
    grid_x = np.arange(x_min + grid_resolution_m / 2, x_max, grid_resolution_m)
    grid_y = np.arange(y_min + grid_resolution_m / 2, y_max, grid_resolution_m)

    if len(grid_x) == 0 or len(grid_y) == 0:
        raise ValueError(
            f"Grid is empty with bounds ({x_min:.2f},{y_min:.2f})-"
            f"({x_max:.2f},{y_max:.2f}) and resolution {grid_resolution_m:.2f}m"
        )

    nx = len(grid_x)
    ny = len(grid_y)
    num_cells = nx * ny

    # Pre-compute grid cell centers
    gx_mesh, gy_mesh = np.meshgrid(grid_x, grid_y, indexing="ij")
    cell_centers = np.column_stack([gx_mesh.ravel(), gy_mesh.ravel()])

    # Generate all unique link pairs
    rows = []
    cols = []
    data = []
    link_index = 0

    # Truncation threshold
    min_weight = np.exp(-0.5 * FRESNEL_TRUNCATION_SIGMA ** 2)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            tx_pos = node_positions[i]
            rx_pos = node_positions[j]
            link_vec = rx_pos - tx_pos
            link_length = np.linalg.norm(link_vec)

            if link_length < MIN_LINK_LENGTH_M:
                continue

            link_dir = link_vec / link_length

            # Fresnel zone parameters
            r_fresnel = np.sqrt(wavelength_m * link_length / 4.0)
            sigma_f = r_fresnel * FRESNEL_SIGMA_FRACTION

            # Bounding box pre-filter
            margin = FRESNEL_TRUNCATION_SIGMA * sigma_f
            bb_x_lo = min(tx_pos[0], rx_pos[0]) - margin
            bb_x_hi = max(tx_pos[0], rx_pos[0]) + margin
            bb_y_lo = min(tx_pos[1], rx_pos[1]) - margin
            bb_y_hi = max(tx_pos[1], rx_pos[1]) + margin

            candidate_mask = (
                (cell_centers[:, 0] >= bb_x_lo) &
                (cell_centers[:, 0] <= bb_x_hi) &
                (cell_centers[:, 1] >= bb_y_lo) &
                (cell_centers[:, 1] <= bb_y_hi)
            )
            candidate_indices = np.nonzero(candidate_mask)[0]

            if len(candidate_indices) == 0:
                link_index += 1
                continue

            candidates = cell_centers[candidate_indices]

            # Projection parameter t
            diff = candidates - tx_pos
            t_vals = diff @ link_dir / link_length

            # Only cells where projection falls within link
            valid = (t_vals >= 0.0) & (t_vals <= 1.0)
            if not np.any(valid):
                link_index += 1
                continue

            valid_indices = candidate_indices[valid]
            valid_candidates = candidates[valid]
            valid_t = t_vals[valid]

            # Perpendicular distance
            projections = tx_pos + np.outer(valid_t * link_length, link_dir)
            d_perp = np.linalg.norm(valid_candidates - projections, axis=1)

            # Gaussian decay
            weights = np.exp(-d_perp ** 2 / (2.0 * sigma_f ** 2))

            # Endpoint weight: parabolic, peaks at midpoint
            endpoint_w = ENDPOINT_WEIGHT_SCALE * valid_t * (1.0 - valid_t)
            weights *= endpoint_w

            # Apply truncation
            significant = weights >= min_weight
            if not np.any(significant):
                link_index += 1
                continue

            sig_indices = valid_indices[significant]
            sig_weights = weights[significant]

            rows.extend([link_index] * len(sig_indices))
            cols.extend(sig_indices.tolist())
            data.extend(sig_weights.tolist())

            link_index += 1

    num_links = link_index

    if not data:
        logger.warning("Sensing matrix is empty — no links with significant weights")
        W = scipy.sparse.csr_matrix((num_links, num_cells))
    else:
        W = scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_links, num_cells),
        )

    logger.info(
        "Sensing matrix: %d links x %d cells, %d nonzeros (%.1f%% sparse)",
        num_links, num_cells, len(data),
        100.0 * (1.0 - len(data) / max(num_links * num_cells, 1)),
    )

    return W, grid_x, grid_y


DENSE_SVD_THRESHOLD = 200
"""Use dense SVD when min(L, V) is below this value.
scipy.sparse.linalg.svds requires k < min(L, V), so it cannot compute
the last singular value. Dense SVD computes the full spectrum. For small
matrices (L < 200), dense SVD is fast and avoids the off-by-one issue."""


def compute_ndf(
    W: scipy.sparse.spmatrix,
    threshold: float = DEFAULT_NDF_THRESHOLD,
) -> Tuple[int, np.ndarray]:
    """Compute the number of spatial degrees of freedom from the sensing matrix.

    Performs SVD of the sensing matrix W and counts singular values above
    the threshold (relative to sigma_max). This is the discrete analogue
    of Bucci-Franceschetti's NDF for continuous apertures.

    For practical WiFi meshes (N < ~120 at 2.4 GHz), the singular value
    spectrum decays gradually — no Slepian-like step function appears
    because the system is far from the Bucci-Franceschetti saturation
    limit. The NDF indicates how many independent spatial features the
    mesh can resolve.

    Args:
        W: Sparse sensing matrix of shape (L, V).
        threshold: Relative threshold — singular values below
            threshold * sigma_max are counted as noise.

    Returns:
        Tuple of:
            - ndf: Number of degrees of freedom (effective rank)
            - singular_values: Array of singular values in descending order

    Raises:
        ValueError: If the sensing matrix is empty.
    """
    num_links, num_voxels = W.shape

    if num_links == 0 or num_voxels == 0:
        raise ValueError(
            f"Sensing matrix has zero dimension: {W.shape}"
        )

    min_dim = min(num_links, num_voxels)

    if min_dim < 2:
        logger.warning("Matrix too small for SVD (shape %s)", W.shape)
        return 0, np.array([])

    # Use dense SVD for small matrices to get the full spectrum.
    # scipy.sparse.linalg.svds requires k < min(L, V), so it cannot
    # compute the last singular value. For small meshes where all SVs
    # are significant, this would undercount NDF by 1.
    if min_dim <= DENSE_SVD_THRESHOLD:
        try:
            W_dense = W.toarray() if scipy.sparse.issparse(W) else np.asarray(W)
            _, sigma, _ = np.linalg.svd(W_dense, full_matrices=False)
        except Exception as e:
            logger.error("Dense SVD failed: %s", e)
            return 0, np.array([])
    else:
        # Large matrix — use sparse SVD (computes min_dim - 1 values)
        k = min_dim - 1
        try:
            _, sigma, _ = svds(W.astype(np.float64), k=k)
        except Exception as e:
            logger.error("Sparse SVD failed: %s", e)
            return 0, np.array([])
        # svds returns in ascending order — reverse to descending
        sigma = sigma[::-1]

    # Count values above threshold
    sigma_max = sigma[0]
    if sigma_max <= 0:
        return 0, sigma

    cutoff = threshold * sigma_max
    ndf = int(np.sum(sigma > cutoff))

    logger.info(
        "NDF = %d (sigma_max=%.4f, cutoff=%.4f, spectrum_size=%d)",
        ndf, sigma_max, cutoff, len(sigma),
    )

    return ndf, sigma


def compute_full_svd(
    W: scipy.sparse.spmatrix,
    threshold: float = DEFAULT_NDF_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the full SVD of the sensing matrix, returning U, sigma, Vt.

    Same dense/sparse logic as compute_ndf(), but returns the full
    decomposition for downstream diagnostics (per-voxel sensitivity,
    per-node marginality, etc.).

    Args:
        W: Sparse sensing matrix of shape (L, V).
        threshold: Not used for truncation here — included for API
            consistency. Callers apply their own threshold on sigma.

    Returns:
        Tuple of:
            - U: Left singular vectors, shape (L, min(L, V))
            - sigma: Singular values in descending order, shape (min(L, V),)
            - Vt: Right singular vectors, shape (min(L, V), V)

    Raises:
        ValueError: If the sensing matrix is empty or too small.
    """
    num_links, num_voxels = W.shape

    if num_links == 0 or num_voxels == 0:
        raise ValueError(f"Sensing matrix has zero dimension: {W.shape}")

    min_dim = min(num_links, num_voxels)

    if min_dim < 2:
        raise ValueError(f"Matrix too small for SVD (shape {W.shape})")

    if min_dim <= DENSE_SVD_THRESHOLD:
        W_dense = W.toarray() if scipy.sparse.issparse(W) else np.asarray(W)
        U, sigma, Vt = np.linalg.svd(W_dense, full_matrices=False)
    else:
        k = min_dim - 1
        U, sigma, Vt = svds(W.astype(np.float64), k=k)
        # svds returns in ascending order — reverse to descending
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

    return U, sigma, Vt


def analyze_mesh_topology(
    node_positions: np.ndarray,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    room_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M,
    threshold: float = DEFAULT_NDF_THRESHOLD,
) -> NDFResult:
    """Analyze the spatial sensing capability of a wireless mesh topology.

    End-to-end pipeline: builds the Fresnel-weighted sensing matrix for
    the given node positions and computes the NDF via SVD.

    This answers the research question: given N nodes at these positions,
    how many independent spatial features can the mesh resolve?

    Args:
        node_positions: (N, 2) array of node (x, y) positions in meters.
        grid_resolution_m: Grid cell size in meters.
        room_bounds: Optional bounding box. Auto-computed if None.
        wavelength_m: Signal wavelength in meters.
        threshold: Relative singular value threshold for NDF.

    Returns:
        NDFResult with NDF, singular values, and mesh metadata.

    Example:
        >>> positions = np.array([[0, 0], [3, 0], [3, 4], [0, 4]])
        >>> result = analyze_mesh_topology(positions)
        >>> print(f"NDF = {result.ndf}, links = {result.num_links}")
    """
    node_positions = np.asarray(node_positions, dtype=np.float64)
    n_nodes = node_positions.shape[0]

    # Build sensing matrix
    W, grid_x, grid_y = build_sensing_matrix(
        node_positions,
        grid_resolution_m=grid_resolution_m,
        room_bounds=room_bounds,
        wavelength_m=wavelength_m,
    )

    # Compute NDF
    ndf, sigma = compute_ndf(W, threshold=threshold)

    # Determine actual room bounds used
    if room_bounds is None:
        room_bounds = _compute_room_bounds(node_positions)

    return NDFResult(
        ndf=ndf,
        singular_values=sigma,
        num_nodes=n_nodes,
        num_links=W.shape[0],
        num_voxels=W.shape[1],
        matrix_shape=W.shape,
        grid_resolution_m=grid_resolution_m,
        room_bounds=room_bounds,
        wavelength_m=wavelength_m,
        threshold=threshold,
    )


def analyze_from_rssi_matrix(
    rssi_matrix: np.ndarray,
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M,
    threshold: float = DEFAULT_NDF_THRESHOLD,
) -> NDFResult:
    """Analyze mesh NDF starting from an RSSI matrix (no positions needed).

    Chains through the existing opencsi pipeline:
        RSSI matrix → DistanceEstimator → PositionSolver → NDF analysis

    This is the preferred entry point when node positions are not known
    precisely. MDS-derived positions have ~0.5m error, but NDF is robust
    to position noise (verified: <2% change at 0.5m std noise).

    Args:
        rssi_matrix: (N, N) symmetric RSSI matrix in dBm.
            Diagonal should be 0 (self-link). Missing values should be NaN.
        grid_resolution_m: Grid cell size in meters.
        wavelength_m: Signal wavelength in meters.
        threshold: Relative singular value threshold for NDF.

    Returns:
        NDFResult with NDF, singular values, and mesh metadata.

    Raises:
        ValueError: If RSSI matrix is invalid or positions can't be solved.

    Example:
        >>> # From a running system's pairwise RSSI measurements:
        >>> result = analyze_from_rssi_matrix(rssi_matrix)
        >>> print(f"NDF = {result.ndf}")
    """
    from opencsi.geometry.distance import DistanceEstimator
    from opencsi.geometry.position import PositionSolver

    rssi_matrix = np.asarray(rssi_matrix, dtype=np.float64)

    if rssi_matrix.ndim != 2 or rssi_matrix.shape[0] != rssi_matrix.shape[1]:
        raise ValueError(
            f"rssi_matrix must be square 2D, got shape {rssi_matrix.shape}"
        )

    # Step 1: RSSI → distance
    distance_estimator = DistanceEstimator()
    distance_matrix = distance_estimator.rssi_matrix_to_distances(rssi_matrix)

    # Step 2: distance → positions via SMACOF
    position_solver = PositionSolver()
    mds_result = position_solver.solve_smacof(distance_matrix)

    if not mds_result.converged:
        logger.warning(
            "MDS did not converge (stress=%.4f). NDF is robust to position "
            "noise, but results may be less precise.",
            mds_result.stress,
        )

    # Align to canonical orientation
    positions = position_solver.align_to_reference(mds_result.positions)

    logger.info(
        "Derived %d node positions from RSSI (MDS stress=%.4f)",
        len(positions), mds_result.stress,
    )

    # Step 3: positions → NDF
    return analyze_mesh_topology(
        positions,
        grid_resolution_m=grid_resolution_m,
        wavelength_m=wavelength_m,
        threshold=threshold,
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _compute_room_bounds(
    node_positions: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute bounding box from node positions with margin.

    Follows the same convention as CoverageAreaEstimator._compute_bounding_box.

    Args:
        node_positions: (N, 2) array of positions.

    Returns:
        ((x_min, y_min), (x_max, y_max)) with margin applied.
    """
    x_min, y_min = node_positions.min(axis=0)
    x_max, y_max = node_positions.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    # Handle degenerate cases
    if width < 1e-6:
        x_center = (x_min + x_max) / 2
        x_min = x_center - DEGENERATE_DIMENSION_M / 2
        x_max = x_center + DEGENERATE_DIMENSION_M / 2
        width = DEGENERATE_DIMENSION_M

    if height < 1e-6:
        y_center = (y_min + y_max) / 2
        y_min = y_center - DEGENERATE_DIMENSION_M / 2
        y_max = y_center + DEGENERATE_DIMENSION_M / 2
        height = DEGENERATE_DIMENSION_M

    margin_x = width * ROOM_MARGIN_FRACTION
    margin_y = height * ROOM_MARGIN_FRACTION

    return (
        (x_min - margin_x, y_min - margin_y),
        (x_max + margin_x, y_max + margin_y),
    )
