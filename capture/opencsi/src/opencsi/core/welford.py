#!/usr/bin/env python3
"""
Welford's Online Algorithm for Mean/Variance.

This module implements numerically stable online statistics using Welford's
algorithm. This is particularly useful for calibration where we need to
continuously update statistics without storing all historical samples.

Reference:
    Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums
    of Squares and Products". Technometrics. 4 (3): 419-420.
"""

import math
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Module-Level Constants
# =============================================================================

# Default confidence level for confidence_interval()
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Z-scores for common confidence levels (two-tailed)
# Using normal approximation (valid for n > 30)
Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Minimum samples before computing confidence intervals
MIN_SAMPLES_FOR_CI = 2


@dataclass
class WelfordAccumulator:
    """
    Online mean/variance accumulator using Welford's algorithm.

    This class provides numerically stable computation of running mean and
    variance without storing individual samples. It supports:
    - Single-sample updates via update()
    - Parallel combination via combine() for aggregating distributed statistics
    - Confidence intervals using normal approximation
    - Serialization via to_dict()/from_dict()

    Attributes:
        n: Number of samples accumulated
        mean: Running mean
        m2: Sum of squared differences from the mean (for variance calculation)

    Example:
        >>> acc = WelfordAccumulator()
        >>> for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     acc.update(x)
        >>> print(f"mean={acc.mean}, std={acc.std()}")
        mean=3.0, std=1.5811388300841898
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from the mean

    def update(self, x: float) -> None:
        """
        Update statistics with a new sample.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: New sample value
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self, ddof: int = 1) -> float:
        """
        Compute sample variance.

        Args:
            ddof: Delta degrees of freedom. Use ddof=1 for sample variance
                  (Bessel's correction), ddof=0 for population variance.

        Returns:
            Sample variance, or 0.0 if not enough samples.
        """
        if self.n <= ddof:
            return 0.0
        return self.m2 / (self.n - ddof)

    def std(self, ddof: int = 1) -> float:
        """
        Compute sample standard deviation.

        Args:
            ddof: Delta degrees of freedom (see variance()).

        Returns:
            Sample standard deviation, or 0.0 if not enough samples.
        """
        return math.sqrt(self.variance(ddof))

    def confidence_interval(
        self,
        level: float = DEFAULT_CONFIDENCE_LEVEL
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for the mean.

        Uses normal approximation (Z-score), which is valid for n > 30.
        For smaller samples, this is an approximation.

        Args:
            level: Confidence level (0.90, 0.95, or 0.99). Default 0.95.

        Returns:
            Tuple of (lower_bound, upper_bound) for the mean.
            Returns (mean, mean) if not enough samples.
        """
        if self.n < MIN_SAMPLES_FOR_CI:
            return (self.mean, self.mean)

        # Get Z-score for confidence level
        z = Z_SCORES.get(level, Z_SCORES[DEFAULT_CONFIDENCE_LEVEL])

        # Standard error of the mean
        se = self.std() / math.sqrt(self.n)

        # Confidence interval
        margin = z * se
        return (self.mean - margin, self.mean + margin)

    @classmethod
    def combine(cls, *accumulators: 'WelfordAccumulator') -> 'WelfordAccumulator':
        """
        Combine multiple accumulators into one (parallel combination).

        This allows aggregating statistics computed on different data
        partitions without access to the original samples.

        Uses Chan's parallel algorithm for combining variances:
            Chan et al. (1983). "Algorithms for Computing the Sample Variance".

        Args:
            *accumulators: WelfordAccumulator instances to combine.

        Returns:
            New WelfordAccumulator with combined statistics.
        """
        if not accumulators:
            return cls()

        if len(accumulators) == 1:
            return cls(
                n=accumulators[0].n,
                mean=accumulators[0].mean,
                m2=accumulators[0].m2
            )

        # Start with a copy of the first accumulator
        combined = cls(
            n=accumulators[0].n,
            mean=accumulators[0].mean,
            m2=accumulators[0].m2
        )

        # Combine remaining accumulators
        for acc in accumulators[1:]:
            if acc.n == 0:
                continue

            if combined.n == 0:
                combined.n = acc.n
                combined.mean = acc.mean
                combined.m2 = acc.m2
                continue

            # Chan's parallel algorithm
            n_combined = combined.n + acc.n
            delta = acc.mean - combined.mean

            # New mean
            new_mean = (combined.n * combined.mean + acc.n * acc.mean) / n_combined

            # New m2 (sum of squared differences)
            # m2_combined = m2_a + m2_b + delta^2 * n_a * n_b / n_combined
            new_m2 = (
                combined.m2 + acc.m2 +
                delta * delta * combined.n * acc.n / n_combined
            )

            combined.n = n_combined
            combined.mean = new_mean
            combined.m2 = new_m2

        return combined

    def to_dict(self) -> dict:
        """
        Serialize to dictionary for persistence.

        Returns:
            Dict with n, mean, m2 fields.
        """
        return {
            'n': self.n,
            'mean': self.mean,
            'm2': self.m2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WelfordAccumulator':
        """
        Deserialize from dictionary.

        Args:
            d: Dictionary with n, mean, m2 fields.

        Returns:
            WelfordAccumulator instance.
        """
        return cls(
            n=d.get('n', 0),
            mean=d.get('mean', 0.0),
            m2=d.get('m2', 0.0),
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"WelfordAccumulator(n={self.n}, mean={self.mean:.4f}, std={self.std():.4f})"
