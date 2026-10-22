#!/usr/bin/env python3
"""
Circular Statistics for Auto-Fresnel.

Self-contained circular statistics module with NO external dependencies.
This is part of the Auto-Fresnel calibration package.

Key Functions:
- circular_mean: Mean direction (Yamartino method)
- circular_std: Circular standard deviation
- circular_var: Circular variance

References:
- Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics.
- Yamartino, R.J. (1984). A comparison of several single-pass estimators.
- Fisher, N.I. (1993). Statistical Analysis of Circular Data.
"""

import numpy as np
from typing import Tuple

# =============================================================================
# Module-Level Constants
# =============================================================================

# Numerical stability epsilon (avoid log(0), divide-by-zero)
EPSILON = 1e-10


def circular_mean(angles: np.ndarray) -> float:
    """
    Compute the mean direction of circular data (Yamartino method).

    For phase data, this correctly handles the wrap-around at ±π.
    The mean direction is the angle of the resultant vector.

    Args:
        angles: Array of angles in radians

    Returns:
        Mean angle in radians [-π, π]

    Example:
        >>> phases = np.array([0.1, -0.1, 6.18])  # 6.18 ≈ 0 - 0.1
        >>> circular_mean(phases)  # ≈ 0.0 (not 2.06 from linear mean)
    """
    if len(angles) == 0:
        return 0.0

    # Convert to unit vectors and average
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))

    # Mean direction is the angle of the resultant vector
    return float(np.arctan2(sin_mean, cos_mean))


def circular_resultant_length(angles: np.ndarray) -> float:
    """
    Compute the mean resultant length R.

    R is a measure of concentration:
    - R = 1: All angles identical (perfect concentration)
    - R = 0: Angles uniformly distributed (maximum dispersion)

    Args:
        angles: Array of angles in radians

    Returns:
        Mean resultant length R in [0, 1]
    """
    if len(angles) == 0:
        return 0.0

    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))

    # Resultant length normalized by n
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
    return float(R)


def circular_var(angles: np.ndarray) -> float:
    """
    Compute the circular variance V = 1 - R.

    Circular variance is bounded [0, 1]:
    - V = 0: All angles identical
    - V = 1: Angles uniformly distributed

    Args:
        angles: Array of angles in radians

    Returns:
        Circular variance in [0, 1]
    """
    R = circular_resultant_length(angles)
    return float(1.0 - R)


def circular_std(angles: np.ndarray) -> float:
    """
    Compute the circular standard deviation (Yamartino approximation).

    This is the standard measure of angular dispersion, defined as:
        σ = sqrt(-2 * ln(R))

    For small dispersions, this approximates the linear standard deviation.

    Args:
        angles: Array of angles in radians

    Returns:
        Circular standard deviation in radians [0, inf)

    Note:
        For very dispersed data (R→0), σ→inf.
        We clamp to sqrt(-2*ln(epsilon)) for numerical stability.
    """
    if len(angles) == 0:
        return 0.0

    R = circular_resultant_length(angles)

    # Clamp R to avoid log(0)
    R = max(R, EPSILON)

    # Yamartino formula
    return float(np.sqrt(-2.0 * np.log(R)))


def circular_mean_and_std(angles: np.ndarray) -> Tuple[float, float]:
    """
    Compute both circular mean and std efficiently.

    More efficient than calling circular_mean and circular_std separately.

    Args:
        angles: Array of angles in radians

    Returns:
        Tuple of (mean_angle, std) in radians
    """
    if len(angles) == 0:
        return 0.0, 0.0

    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)

    sin_mean = np.mean(sin_vals)
    cos_mean = np.mean(cos_vals)

    # Mean direction
    mean_angle = np.arctan2(sin_mean, cos_mean)

    # Resultant length
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    R = max(R, EPSILON)

    # Circular std
    std = np.sqrt(-2.0 * np.log(R))

    return float(mean_angle), float(std)


def circular_distance(angle1: float, angle2: float) -> float:
    """
    Compute the shortest angular distance between two angles.

    Properly handles wrap-around at ±π.

    Args:
        angle1: First angle in radians
        angle2: Second angle in radians

    Returns:
        Unsigned angular distance in [0, π]
    """
    diff = angle1 - angle2
    # Wrap to [-π, π]
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return float(np.abs(diff))


def unwrap_phase_diff(phase_diff: np.ndarray) -> np.ndarray:
    """
    Unwrap phase differences to minimize jumps.

    When computing Δφ = φ(t) - φ(t-1), large jumps near ±π
    indicate wrap-around, not actual large phase changes.

    Args:
        phase_diff: Array of phase differences

    Returns:
        Unwrapped phase differences in [-π, π]
    """
    # Wrap to [-π, π]
    wrapped = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
    return wrapped


def circular_weighted_mean(
    angles: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute weighted circular mean.

    Useful for SNR-weighted phase averaging across subcarriers.

    Args:
        angles: Array of angles in radians
        weights: Array of non-negative weights (e.g., SNR)

    Returns:
        Weighted mean angle in radians [-π, π]
    """
    if len(angles) == 0 or len(weights) == 0:
        return 0.0

    # Normalize weights
    weight_sum = np.sum(weights)
    if weight_sum < EPSILON:
        return circular_mean(angles)

    weights_norm = weights / weight_sum

    # Weighted sum of unit vectors
    sin_mean = np.sum(weights_norm * np.sin(angles))
    cos_mean = np.sum(weights_norm * np.cos(angles))

    return float(np.arctan2(sin_mean, cos_mean))


def circular_concentration(angles: np.ndarray) -> float:
    """
    Estimate the concentration parameter κ (kappa) of a von Mises distribution.

    κ measures how concentrated the data is around the mean:
    - κ = 0: Uniform distribution
    - κ → ∞: All data at the mean

    Uses Mardia-Jupp approximation for computational efficiency.

    Args:
        angles: Array of angles in radians

    Returns:
        Concentration parameter κ ≥ 0
    """
    R = circular_resultant_length(angles)

    if R < 0.53:
        # Low concentration: κ ≈ 2R + R³ + 5R⁵/6
        kappa = 2 * R + R**3 + (5 * R**5) / 6
    elif R < 0.85:
        # Medium concentration: κ ≈ -0.4 + 1.39R + 0.43/(1-R)
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R + EPSILON)
    else:
        # High concentration: κ ≈ 1/(2(1-R) - (1-R)² - (1-R)³)
        one_minus_R = 1 - R + EPSILON
        kappa = 1 / (2 * one_minus_R - one_minus_R**2 - one_minus_R**3)

    return float(max(0, kappa))


def compute_phase_change_zscore(
    current_phase: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    min_std: float = 0.01
) -> float:
    """
    Compute Z-score for phase change using circular statistics.

    For each subcarrier, computes how many standard deviations
    the current phase is from the baseline mean, using circular distance.

    Args:
        current_phase: Current phase array (n_subcarriers,)
        baseline_mean: Per-subcarrier baseline mean phase
        baseline_std: Per-subcarrier baseline std (circular)
        min_std: Minimum std to avoid division by zero

    Returns:
        Mean Z-score across subcarriers
    """
    if len(current_phase) == 0:
        return 0.0

    if len(current_phase) != len(baseline_mean) or len(current_phase) != len(baseline_std):
        return 0.0

    # Compute circular distance from baseline mean
    distances = np.array([
        circular_distance(current_phase[i], baseline_mean[i])
        for i in range(len(current_phase))
    ])

    # Compute Z-scores with floor on std
    stds = np.maximum(baseline_std, min_std)
    z_scores = distances / stds

    # Return mean Z-score (could also use max for sensitivity)
    return float(np.mean(z_scores))


def compute_phase_diff_stats(
    phase_current: np.ndarray,
    phase_previous: np.ndarray
) -> Tuple[float, float]:
    """
    Compute statistics of time-differential phase Δφ = φ(t) - φ(t-1).

    Time-differential phase removes SFO (becomes DC offset) and makes
    motion detection more robust.

    Args:
        phase_current: Current phase (n_subcarriers,)
        phase_previous: Previous phase (n_subcarriers,)

    Returns:
        Tuple of (mean_phase_diff, std_phase_diff)
    """
    if len(phase_current) != len(phase_previous) or len(phase_current) == 0:
        return 0.0, 0.0

    # Compute phase differences
    phase_diff = phase_current - phase_previous

    # Unwrap to handle ±π wrap-around
    phase_diff = unwrap_phase_diff(phase_diff)

    # Return circular statistics of the differences
    return circular_mean_and_std(phase_diff)
