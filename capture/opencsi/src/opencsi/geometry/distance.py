"""Distance Estimator using path loss model.

Converts RSSI measurements to estimated distances by inverting the
log-distance path loss model.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Reference RSSI at 1 meter (typical for ESP32 at 2.4 GHz)
REFERENCE_RSSI_1M = -40.0  # dBm

# Path loss exponent (free space = 2.0, indoor = 2.5-4.0)
DEFAULT_PATH_LOSS_EXPONENT = 2.5

# Distance bounds for sanity checking
MIN_DISTANCE_M = 0.1
MAX_DISTANCE_M = 20.0


@dataclass
class PathLossModel:
    """Parameters for the log-distance path loss model.

    RSSI = RSSI_ref - 10 * n * log10(d)

    Where:
        RSSI_ref: RSSI at reference distance (1m)
        n: Path loss exponent
        d: Distance in meters
    """

    rssi_ref: float = REFERENCE_RSSI_1M
    path_loss_exponent: float = DEFAULT_PATH_LOSS_EXPONENT

    def rssi_to_distance(self, rssi: float) -> float:
        """Convert RSSI to estimated distance.

        d = 10^((RSSI_ref - RSSI) / (10 * n))
        """
        if np.isnan(rssi):
            return np.nan

        exponent = (self.rssi_ref - rssi) / (10 * self.path_loss_exponent)
        distance = 10 ** exponent

        # Clamp to reasonable bounds
        return np.clip(distance, MIN_DISTANCE_M, MAX_DISTANCE_M)

    def distance_to_rssi(self, distance: float) -> float:
        """Convert distance to expected RSSI (for verification)."""
        if distance <= 0:
            return self.rssi_ref

        return self.rssi_ref - 10 * self.path_loss_exponent * np.log10(distance)


class DistanceEstimator:
    """Estimates distances from RSSI measurements.

    Uses the log-distance path loss model with configurable parameters.
    Can also estimate the path loss exponent from known reference distances.

    Usage:
        estimator = DistanceEstimator(rssi_ref=-40.0, path_loss_exponent=2.5)

        # Single RSSI to distance
        distance = estimator.rssi_to_distance(-55.0)  # ~1.78m

        # Convert full RSSI matrix
        distance_matrix = estimator.rssi_matrix_to_distances(rssi_matrix)
    """

    def __init__(
        self,
        rssi_ref: float = REFERENCE_RSSI_1M,
        path_loss_exponent: float = DEFAULT_PATH_LOSS_EXPONENT,
    ):
        self.model = PathLossModel(
            rssi_ref=rssi_ref,
            path_loss_exponent=path_loss_exponent,
        )

    @property
    def rssi_ref(self) -> float:
        return self.model.rssi_ref

    @property
    def path_loss_exponent(self) -> float:
        return self.model.path_loss_exponent

    def rssi_to_distance(self, rssi: float) -> float:
        """Convert single RSSI value to distance estimate."""
        return self.model.rssi_to_distance(rssi)

    def rssi_matrix_to_distances(self, rssi_matrix: np.ndarray) -> np.ndarray:
        """Convert RSSI matrix to distance matrix.

        Args:
            rssi_matrix: (N, N) array of RSSI values in dBm.
                         Diagonal should be 0 (self-link).
                         Missing values should be NaN.

        Returns:
            (N, N) distance matrix in meters.
            Diagonal is 0. Missing values remain NaN.
        """
        n = rssi_matrix.shape[0]
        distances = np.zeros_like(rssi_matrix, dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    distances[i, j] = 0.0
                else:
                    distances[i, j] = self.model.rssi_to_distance(rssi_matrix[i, j])

        return distances

    def estimate_path_loss_exponent(
        self,
        rssi_pairs: np.ndarray,
        known_distances: np.ndarray,
        rssi_ref: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Estimate path loss exponent from known reference distances.

        Given some known distances (e.g., from manual measurement or
        previous calibration), estimate the best-fit path loss exponent.

        Args:
            rssi_pairs: Array of RSSI values (dBm)
            known_distances: Corresponding distances (m)
            rssi_ref: Reference RSSI at 1m (uses default if None)

        Returns:
            Tuple of (estimated_n, r_squared)
        """
        if rssi_ref is None:
            rssi_ref = self.model.rssi_ref

        # Filter out invalid data
        valid_mask = (
            ~np.isnan(rssi_pairs) &
            ~np.isnan(known_distances) &
            (known_distances > 0)
        )
        rssi_valid = rssi_pairs[valid_mask]
        dist_valid = known_distances[valid_mask]

        if len(rssi_valid) < 2:
            logger.warning("Not enough valid data points for path loss estimation")
            return self.model.path_loss_exponent, 0.0

        # Linear regression on: (RSSI_ref - RSSI) = 10*n*log10(d)
        # Let y = (RSSI_ref - RSSI) and x = 10*log10(d)
        # Then y = n * x

        x = 10 * np.log10(dist_valid)
        y = rssi_ref - rssi_valid

        # Least squares: n = sum(x*y) / sum(x^2)
        n_estimated = np.sum(x * y) / np.sum(x ** 2)

        # Compute R^2 using no-intercept formula (origin-forced regression).
        # Standard R² uses SS_tot = Σ(y - ȳ)² which can be negative for
        # no-intercept models. The correct formula uses SS_tot = Σ(y²).
        y_pred = n_estimated * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot_noint = np.sum(y ** 2)
        r_squared = 1 - (ss_res / ss_tot_noint) if ss_tot_noint > 0 else 0.0

        # Clamp to reasonable range
        n_estimated = np.clip(n_estimated, 1.5, 5.0)

        logger.info(
            f"Estimated path loss exponent: n={n_estimated:.2f}, R²={r_squared:.3f}"
        )

        return float(n_estimated), float(r_squared)

    def update_model(
        self,
        rssi_ref: Optional[float] = None,
        path_loss_exponent: Optional[float] = None,
    ) -> None:
        """Update the path loss model parameters."""
        if rssi_ref is not None:
            self.model.rssi_ref = rssi_ref
        if path_loss_exponent is not None:
            self.model.path_loss_exponent = path_loss_exponent

    def get_distance_uncertainty(self, rssi: float, rssi_std: float) -> float:
        """Estimate distance uncertainty from RSSI measurement noise.

        Uses error propagation to estimate distance uncertainty from
        RSSI measurement standard deviation.

        Args:
            rssi: Mean RSSI value (dBm)
            rssi_std: Standard deviation of RSSI (dB)

        Returns:
            Estimated distance standard deviation (m)
        """
        d = self.model.rssi_to_distance(rssi)

        # Derivative: dd/dRSSI = -d * ln(10) / (10 * n)
        # std_d = |dd/dRSSI| * std_RSSI
        coef = np.log(10) / (10 * self.model.path_loss_exponent)
        distance_std = d * coef * rssi_std

        return float(distance_std)
