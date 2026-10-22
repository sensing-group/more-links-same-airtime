"""Position Solver using Multidimensional Scaling (MDS).

Reconstructs 2D node positions from pairwise distance estimates.
Uses classical MDS with optional SMACOF refinement for noisy data.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# SMACOF algorithm parameters
MDS_MAX_ITERATIONS = 300
SMACOF_STRESS_THRESHOLD = 0.1
SMACOF_CONVERGENCE_TOL = 1e-6
SMACOF_TIMEOUT_SECONDS = 30.0  # Wall-clock timeout for SMACOF

# Outlier detection
OUTLIER_RESIDUAL_THRESHOLD = 3.0  # Standard deviations

# Distance imputation
DEFAULT_FALLBACK_DISTANCE = 1.0  # Used when all distances are NaN


@dataclass
class MDSResult:
    """Result of MDS position estimation."""

    positions: np.ndarray  # (N, 2) array of (x, y) positions
    stress: float  # Normalized stress (lower is better, <0.1 is good)
    iterations: int  # Number of SMACOF iterations (0 for classical MDS)
    converged: bool  # Whether SMACOF converged
    outlier_indices: List[int]  # Indices of potential outlier nodes

    @property
    def position_dict(self) -> dict:
        """Return positions as dict with integer keys."""
        return {i: (self.positions[i, 0], self.positions[i, 1])
                for i in range(len(self.positions))}


class PositionSolver:
    """Solves 2D positions from distance matrix using MDS.

    Workflow:
    1. Classical MDS: Eigendecomposition of double-centered distance matrix
    2. SMACOF refinement: Iterative stress minimization for noisy data
    3. Outlier detection: Flag nodes with large residual errors

    Usage:
        solver = PositionSolver()

        # Classical MDS (fast, good for clean data)
        positions, stress = solver.solve_classical_mds(distance_matrix)

        # SMACOF (better for noisy data)
        result = solver.solve_smacof(distance_matrix)

        # With outlier detection
        clean_distances, outliers = solver.detect_and_remove_outliers(distance_matrix)
    """

    def __init__(
        self,
        max_iterations: int = MDS_MAX_ITERATIONS,
        stress_threshold: float = SMACOF_STRESS_THRESHOLD,
        convergence_tol: float = SMACOF_CONVERGENCE_TOL,
    ):
        self.max_iterations = max_iterations
        self.stress_threshold = stress_threshold
        self.convergence_tol = convergence_tol

    def _impute_missing_distances(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Impute missing (NaN) distances using shortest path.

        For missing links, estimate distance as minimum path through
        other nodes (triangle inequality).

        Raises:
            ValueError: If distance matrix has no valid distances after Floyd-Warshall.
        """
        D = distance_matrix.copy()
        n = D.shape[0]

        # Replace NaN with infinity for shortest path
        D_inf = np.where(np.isnan(D), np.inf, D)

        # Floyd-Warshall for shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if D_inf[i, k] + D_inf[k, j] < D_inf[i, j]:
                        D_inf[i, j] = D_inf[i, k] + D_inf[k, j]

        # Validate that at least some valid distances exist after Floyd-Warshall
        finite_mask = ~np.isinf(D_inf)
        np.fill_diagonal(finite_mask, False)  # Exclude diagonal
        valid_count = np.sum(finite_mask)

        if valid_count == 0:
            raise ValueError(
                "Distance matrix has no valid distances after Floyd-Warshall. "
                "Cannot solve positions without connectivity."
            )

        # Replace remaining infinities with mean distance
        # Handle NaN result when all original distances are NaN
        valid_original = distance_matrix[~np.isnan(distance_matrix) & ~np.isinf(distance_matrix)]
        if len(valid_original) > 0:
            mean_dist = np.nanmean(valid_original)
        else:
            mean_dist = DEFAULT_FALLBACK_DISTANCE
            logger.warning(
                "All distances are NaN/Inf, using fallback distance: %f",
                mean_dist,
            )

        if np.isnan(mean_dist):
            mean_dist = DEFAULT_FALLBACK_DISTANCE

        D_inf = np.where(np.isinf(D_inf), mean_dist, D_inf)

        return D_inf

    def solve_classical_mds(
        self,
        distance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Classical MDS via eigendecomposition.

        Converts distance matrix to positions using double centering
        and eigenvalue decomposition.

        Args:
            distance_matrix: (N, N) symmetric distance matrix.
                            Missing values (NaN) will be imputed.

        Returns:
            Tuple of:
            - positions: (N, 2) array of (x, y) coordinates
            - stress: Normalized stress value
        """
        # Handle missing values
        D = self._impute_missing_distances(distance_matrix)
        n = D.shape[0]

        # Ensure symmetry
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        # Double centering: B = -0.5 * H * D^2 * H
        # where H = I - (1/n) * 11^T is the centering matrix
        D_sq = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J

        # Eigendecomposition (use eigh for symmetric matrix)
        eigenvalues, eigenvectors = eigh(B)

        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top 2 dimensions
        # Handle negative eigenvalues (shouldn't happen with valid distances)
        eigenvalues = np.maximum(eigenvalues[:2], 0)

        # Positions: X = V * sqrt(Lambda)
        positions = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

        # Compute stress
        stress = self._compute_stress(D, positions)

        return positions, stress

    def solve_smacof(
        self,
        distance_matrix: np.ndarray,
        initial_positions: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> MDSResult:
        """SMACOF algorithm for MDS with noise handling.

        Iteratively minimizes stress by weighted least squares.
        Better than classical MDS for noisy distance estimates.

        Args:
            distance_matrix: (N, N) distance matrix with possible NaN values
            initial_positions: (N, 2) initial guess (uses classical MDS if None)
            weights: (N, N) weight matrix (higher = more reliable, 0 = ignore)

        Returns:
            MDSResult with positions, stress, and convergence info
        """
        # Handle missing values
        D = self._impute_missing_distances(distance_matrix)
        n = D.shape[0]

        # Ensure symmetry
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        # Set up weights (0 for missing/imputed, 1 for measured)
        if weights is None:
            weights = np.where(np.isnan(distance_matrix), 0.0, 1.0)
            weights = (weights + weights.T) / 2  # Symmetrize
            np.fill_diagonal(weights, 0)

        # Initialize positions
        if initial_positions is None:
            X, _ = self.solve_classical_mds(distance_matrix)
        else:
            X = initial_positions.copy()

        # SMACOF iterations with wall-clock timeout
        old_stress = np.inf
        converged = False
        iterations = 0
        timed_out = False
        start_time = time.monotonic()

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Check wall-clock timeout
            elapsed = time.monotonic() - start_time
            if elapsed > SMACOF_TIMEOUT_SECONDS:
                logger.warning(
                    "SMACOF timed out after %.1f seconds at iteration %d",
                    elapsed,
                    iterations,
                )
                timed_out = True
                break

            # Compute current inter-point distances
            D_X = self._compute_distance_matrix(X)

            # Compute stress
            stress = self._compute_weighted_stress(D, D_X, weights)

            # Check convergence
            if abs(old_stress - stress) < self.convergence_tol:
                converged = True
                break
            old_stress = stress

            # Guttman transform
            X = self._guttman_transform(X, D, D_X, weights)

        # Center positions at origin
        X -= X.mean(axis=0)

        # Detect outliers
        D_X = self._compute_distance_matrix(X)
        outliers = self._detect_outliers(D, D_X, weights)

        # Recalculate stress excluding outliers if any were found
        if outliers:
            # Create weight mask that excludes outlier nodes
            weights_clean = weights.copy()
            for idx in outliers:
                weights_clean[idx, :] = 0.0
                weights_clean[:, idx] = 0.0
            stress = self._compute_weighted_stress(D, D_X, weights_clean)
            logger.info(
                "Recalculated stress excluding %d outlier(s): %.4f",
                len(outliers),
                stress,
            )

        return MDSResult(
            positions=X,
            stress=stress,
            iterations=iterations,
            converged=converged and not timed_out,
            outlier_indices=outliers,
        )

    def _compute_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances from positions."""
        n = positions.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def _compute_stress(
        self,
        target_distances: np.ndarray,
        positions: np.ndarray,
    ) -> float:
        """Compute normalized stress (Stress-1).

        stress = sqrt(sum((d_ij - d_ij_hat)^2) / sum(d_ij^2))
        """
        D_hat = self._compute_distance_matrix(positions)
        n = target_distances.shape[0]

        numerator = 0.0
        denominator = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                diff = target_distances[i, j] - D_hat[i, j]
                numerator += diff ** 2
                denominator += target_distances[i, j] ** 2

        if denominator == 0:
            return 0.0

        return np.sqrt(numerator / denominator)

    def _compute_weighted_stress(
        self,
        target_distances: np.ndarray,
        current_distances: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute weighted stress."""
        n = target_distances.shape[0]

        numerator = 0.0
        denominator = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                w = weights[i, j]
                diff = target_distances[i, j] - current_distances[i, j]
                numerator += w * (diff ** 2)
                denominator += w * (target_distances[i, j] ** 2)

        if denominator == 0:
            return 0.0

        return np.sqrt(numerator / denominator)

    def _guttman_transform(
        self,
        X: np.ndarray,
        D: np.ndarray,
        D_X: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Apply Guttman transform (SMACOF update step)."""
        n = X.shape[0]

        # Build B matrix
        B = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if D_X[i, j] > 1e-10:
                        B[i, j] = -weights[i, j] * D[i, j] / D_X[i, j]
                    else:
                        B[i, j] = 0.0

        # Diagonal elements
        for i in range(n):
            B[i, i] = -np.sum(B[i, :])

        # Sum of weights for normalization
        V = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    V[i, j] = -weights[i, j]
            V[i, i] = -np.sum(V[i, :])

        # Regularize V for inversion
        V_reg = V + np.eye(n) * 1e-6

        # Update: X_new = V^{-1} * B * X
        X_new = np.linalg.solve(V_reg, B @ X)

        return X_new

    def _detect_outliers(
        self,
        target_distances: np.ndarray,
        current_distances: np.ndarray,
        weights: np.ndarray,
    ) -> List[int]:
        """Detect nodes with large residual errors."""
        n = target_distances.shape[0]

        # Compute per-node residual
        node_residuals = np.zeros(n)
        for i in range(n):
            residuals = []
            for j in range(n):
                if i != j and weights[i, j] > 0:
                    residuals.append(
                        abs(target_distances[i, j] - current_distances[i, j])
                    )
            if residuals:
                node_residuals[i] = np.mean(residuals)

        # Flag nodes with residual > threshold * std
        if node_residuals.std() > 0:
            z_scores = (node_residuals - node_residuals.mean()) / node_residuals.std()
            outliers = [i for i in range(n) if z_scores[i] > OUTLIER_RESIDUAL_THRESHOLD]
        else:
            outliers = []

        if outliers:
            logger.warning(f"Detected {len(outliers)} potential outlier node(s)")

        return outliers

    def detect_and_remove_outliers(
        self,
        distance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """Detect outliers and return cleaned distance matrix.

        First runs SMACOF to get position estimates, then identifies
        nodes with large residuals and returns matrix with those
        rows/columns zeroed out.

        Args:
            distance_matrix: (N, N) distance matrix

        Returns:
            Tuple of:
            - cleaned_matrix: Matrix with outlier links set to NaN
            - outlier_indices: Indices of outlier nodes
        """
        result = self.solve_smacof(distance_matrix)

        if not result.outlier_indices:
            return distance_matrix.copy(), []

        # Create weight matrix with outliers zeroed
        cleaned = distance_matrix.copy()
        for idx in result.outlier_indices:
            cleaned[idx, :] = np.nan
            cleaned[:, idx] = np.nan
        np.fill_diagonal(cleaned, 0)

        # Validate remaining matrix has valid data
        n = cleaned.shape[0]
        remaining_nodes = [i for i in range(n) if i not in result.outlier_indices]

        if len(remaining_nodes) < 3:
            logger.warning(
                "After outlier removal, only %d nodes remain. "
                "Need at least 3 for 2D position solving.",
                len(remaining_nodes),
            )
            # Return original matrix with warning rather than unusable result
            return distance_matrix.copy(), result.outlier_indices

        # Check that remaining nodes have valid distances between them
        valid_pairs = 0
        for i in remaining_nodes:
            for j in remaining_nodes:
                if i < j and not np.isnan(cleaned[i, j]):
                    valid_pairs += 1

        min_required_pairs = len(remaining_nodes) - 1  # At least a spanning tree
        if valid_pairs < min_required_pairs:
            logger.warning(
                "After outlier removal, only %d valid distance pairs remain "
                "(need at least %d). Returning original matrix.",
                valid_pairs,
                min_required_pairs,
            )
            return distance_matrix.copy(), result.outlier_indices

        return cleaned, result.outlier_indices

    def align_to_reference(
        self,
        positions: np.ndarray,
        reference_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Align positions using Procrustes analysis.

        MDS solutions are unique up to rotation, reflection, and translation.
        This aligns to a reference or canonical orientation.

        Args:
            positions: (N, 2) positions to align
            reference_positions: (N, 2) reference positions (optional)
                                If None, aligns to first two nodes on x-axis.

        Returns:
            Aligned (N, 2) positions
        """
        X = positions.copy()

        # Center at origin
        X -= X.mean(axis=0)

        if reference_positions is not None:
            # Procrustes alignment to reference
            Y = reference_positions.copy()
            Y -= Y.mean(axis=0)

            # Find optimal rotation using SVD (Procrustes)
            H = X.T @ Y
            U, S, Vt = np.linalg.svd(H)
            # Correct for reflection: ensure det(R) > 0
            d = np.linalg.det(Vt.T @ U.T)
            sign_matrix = np.eye(U.shape[1])
            sign_matrix[-1, -1] = np.sign(d)
            R = Vt.T @ sign_matrix @ U.T

            # Apply rotation
            X = X @ R
        else:
            # Default: align first two nodes along x-axis
            if len(X) >= 2:
                # Rotate so node 0 -> node 1 is along positive x
                dx = X[1, 0] - X[0, 0]
                dy = X[1, 1] - X[0, 1]
                angle = -np.arctan2(dy, dx)

                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                X = X @ R.T

                # Ensure positive y for first node not on x-axis
                if len(X) >= 3 and X[2, 1] < 0:
                    X[:, 1] = -X[:, 1]  # Flip y

        return X
