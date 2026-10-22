"""
Fresnel Zone Geometry Calculations for OpenCSI.

This module provides Fresnel zone calculations for SNR-based detection
width scaling. Supports dynamic wavelength based on channel/frequency band.

The Fresnel zone is the ellipsoidal region around the line-of-sight path
where radio wave reflections contribute constructively or destructively.
For WiFi sensing, motion within the first Fresnel zone causes the most
significant signal perturbation.

Key relationships:
- First Fresnel zone radius at midpoint: r = sqrt(λ * d / 4)
- Larger wavelength (2.4 GHz) = larger detection zone
- Smaller wavelength (5/6 GHz) = smaller but more precise zone
"""

import math
from typing import Optional

import numpy as np

from opencsi.core.config import (
    WAVELENGTH_2_4GHZ_M,
    SNR_EXCELLENT_DB,
    SNR_GOOD_DB,
    SNR_FAIR_DB,
    SNR_FACTOR_EXCELLENT,
    SNR_FACTOR_GOOD,
    SNR_FACTOR_FAIR,
    SNR_FACTOR_MINIMUM,
    channel_to_wavelength,
    get_band_wavelength,
)


def compute_fresnel_radius_at_midpoint(
    link_length_m: float,
    n: int = 1,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M
) -> float:
    """
    Compute the maximum Fresnel zone radius (at link midpoint).

    The first Fresnel zone is widest at the midpoint between TX and RX.
    r_n = sqrt(n * λ * d / 4)

    Args:
        link_length_m: Distance between TX and RX in meters
        n: Fresnel zone number (1 = first zone)
        wavelength_m: Signal wavelength in meters

    Returns:
        Maximum Fresnel radius in meters
    """
    if link_length_m <= 0:
        return 0.0
    return math.sqrt(n * wavelength_m * link_length_m / 4)


def _interpolate(value: float, low: float, high: float, factor_low: float, factor_high: float) -> float:
    """Linear interpolation between two SNR thresholds."""
    fraction = (value - low) / (high - low)
    return factor_low + fraction * (factor_high - factor_low)


def compute_snr_scaling_factor(snr_db: float) -> float:
    """
    Compute scaling factor for effective detection width based on SNR.

    Higher SNR = larger effective detection zone (closer to theoretical).
    Lower SNR = reduced detection zone due to noise masking the signal.

    Args:
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Scaling factor [SNR_FACTOR_MINIMUM, SNR_FACTOR_EXCELLENT]
    """
    if snr_db >= SNR_EXCELLENT_DB:
        return SNR_FACTOR_EXCELLENT

    if snr_db >= SNR_GOOD_DB:
        return _interpolate(snr_db, SNR_GOOD_DB, SNR_EXCELLENT_DB, SNR_FACTOR_GOOD, SNR_FACTOR_EXCELLENT)

    if snr_db >= SNR_FAIR_DB:
        return _interpolate(snr_db, SNR_FAIR_DB, SNR_GOOD_DB, SNR_FACTOR_FAIR, SNR_FACTOR_GOOD)

    # Below fair: clamp to minimum for very low SNR
    fraction = max(0.0, snr_db / SNR_FAIR_DB)
    return SNR_FACTOR_MINIMUM + fraction * (SNR_FACTOR_FAIR - SNR_FACTOR_MINIMUM)


def compute_effective_detection_width(
    link_length_m: float,
    snr_db: float,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M
) -> float:
    """
    Compute effective detection width for a link based on SNR.

    Combines theoretical Fresnel radius with SNR-based scaling.

    Args:
        link_length_m: Distance between TX and RX in meters
        snr_db: Current SNR in dB
        wavelength_m: Signal wavelength in meters

    Returns:
        Effective detection width in meters
    """
    # Theoretical maximum at midpoint
    fresnel_radius = compute_fresnel_radius_at_midpoint(
        link_length_m, n=1, wavelength_m=wavelength_m
    )

    # Scale by SNR
    scale_factor = compute_snr_scaling_factor(snr_db)

    return fresnel_radius * scale_factor


def compute_detection_width_for_channel(
    link_length_m: float,
    snr_db: float,
    channel: Optional[int] = None,
    band: str = "2.4GHz",
) -> float:
    """
    Compute effective detection width using channel number for wavelength.

    Convenience function that computes wavelength from WiFi channel,
    enabling dynamic Fresnel zone calculation for multi-band systems.

    Args:
        link_length_m: Distance between TX and RX in meters
        snr_db: Current SNR in dB
        channel: WiFi channel number (if None, uses band default)
        band: Frequency band ("2.4GHz", "5GHz", "6GHz")

    Returns:
        Effective detection width in meters

    Examples:
        # 2.4 GHz channel 6, 5m link, 25 dB SNR
        width = compute_detection_width_for_channel(5.0, 25.0, channel=6)

        # 5 GHz channel 36, same link
        width_5g = compute_detection_width_for_channel(5.0, 25.0, channel=36, band="5GHz")
        # width_5g will be smaller (5 GHz has shorter wavelength)
    """
    if channel is not None:
        wavelength_m = channel_to_wavelength(channel, band)
    else:
        wavelength_m = get_band_wavelength(band)

    return compute_effective_detection_width(link_length_m, snr_db, wavelength_m)


# =============================================================================
# SNR-Aware Confidence Bounds (P1 Ghost Filter Enhancement)
# =============================================================================

# Uncertainty percentages by SNR tier (from RF physics research)
# Lower SNR = higher uncertainty in detection zone boundaries
# Reference: docs/4-research/in-progress/RF_PHYSICS_GHOST_FILTERING.md
UNCERTAINTY_EXCELLENT = 0.05   # ±5% at 25+ dB (high confidence)
UNCERTAINTY_GOOD = 0.10        # ±10% at 20 dB
UNCERTAINTY_FAIR = 0.40        # ±40% at 10 dB (low confidence)
UNCERTAINTY_POOR = 0.60        # ±60% at <10 dB (very low confidence)

# SNR thresholds for uncertainty tiers (aligned with research)
# These shadow the config.py values but are specific to uncertainty calculation
SNR_UNCERTAINTY_EXCELLENT = 25.0  # Above: ±5%
SNR_UNCERTAINTY_GOOD = 20.0       # Above: ±10%
SNR_UNCERTAINTY_FAIR = 10.0       # Above: ±40%

# Multi-link confirmation requirement thresholds
REQUIRE_MULTILINK_UNCERTAINTY = 0.25  # Above this uncertainty, require multi-link confirmation


def compute_fresnel_confidence_bounds(
    link_length_m: float,
    snr_db: float,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M
) -> tuple[float, float, float, float]:
    """
    Compute Fresnel detection width with confidence bounds based on SNR.

    Lower SNR = wider uncertainty bounds = less certain about detection zone.
    When uncertainty is high, the system should require multi-link confirmation
    to reduce false positives from single noisy links.

    Physics background:
    - At high SNR (25+ dB), the detected signal clearly dominates noise,
      so the effective Fresnel zone is close to theoretical.
    - At low SNR (10 dB), noise can mask/simulate signals, creating
      uncertainty about where detection actually occurred.

    Args:
        link_length_m: Distance between TX and RX in meters
        snr_db: Current SNR in dB
        wavelength_m: Signal wavelength in meters

    Returns:
        Tuple of (effective_width, lower_bound, upper_bound, uncertainty):
        - effective_width: Central estimate in meters
        - lower_bound: Conservative (smaller) bound in meters
        - upper_bound: Liberal (larger) bound in meters
        - uncertainty: Relative uncertainty [0, 1] (0=certain, 1=uncertain)

    Examples:
        # High SNR (25 dB) → tight bounds
        w, lo, hi, unc = compute_fresnel_confidence_bounds(5.0, 25.0)
        # unc ≈ 0.05 (±5%), requires no multi-link confirmation

        # Low SNR (10 dB) → wide bounds
        w, lo, hi, unc = compute_fresnel_confidence_bounds(5.0, 10.0)
        # unc ≈ 0.40 (±40%), should require multi-link confirmation
    """
    effective_width = compute_effective_detection_width(
        link_length_m, snr_db, wavelength_m
    )

    # Compute uncertainty based on SNR tier (using research-aligned thresholds)
    if snr_db >= SNR_UNCERTAINTY_EXCELLENT:
        uncertainty = UNCERTAINTY_EXCELLENT
    elif snr_db >= SNR_UNCERTAINTY_GOOD:
        # Interpolate between GOOD and EXCELLENT
        fraction = (snr_db - SNR_UNCERTAINTY_GOOD) / (SNR_UNCERTAINTY_EXCELLENT - SNR_UNCERTAINTY_GOOD)
        uncertainty = UNCERTAINTY_GOOD + fraction * (UNCERTAINTY_EXCELLENT - UNCERTAINTY_GOOD)
    elif snr_db >= SNR_UNCERTAINTY_FAIR:
        # Interpolate between FAIR and GOOD
        fraction = (snr_db - SNR_UNCERTAINTY_FAIR) / (SNR_UNCERTAINTY_GOOD - SNR_UNCERTAINTY_FAIR)
        uncertainty = UNCERTAINTY_FAIR + fraction * (UNCERTAINTY_GOOD - UNCERTAINTY_FAIR)
    else:
        # Below FAIR: interpolate toward POOR
        fraction = max(0.0, snr_db / SNR_UNCERTAINTY_FAIR)
        uncertainty = UNCERTAINTY_POOR - fraction * (UNCERTAINTY_POOR - UNCERTAINTY_FAIR)

    # Compute bounds
    lower_bound = effective_width * (1.0 - uncertainty)
    upper_bound = effective_width * (1.0 + uncertainty)

    return effective_width, lower_bound, upper_bound, uncertainty


def requires_multilink_confirmation(snr_db: float) -> bool:
    """
    Check if the SNR is low enough to require multi-link confirmation.

    When SNR is poor, single-link detections are unreliable and the system
    should require corroboration from multiple links before treating a
    detection as a real presence.

    Args:
        snr_db: Signal-to-noise ratio in dB

    Returns:
        True if multi-link confirmation should be required
    """
    # Compute uncertainty for this SNR level (using research-aligned thresholds)
    if snr_db >= SNR_UNCERTAINTY_EXCELLENT:
        uncertainty = UNCERTAINTY_EXCELLENT
    elif snr_db >= SNR_UNCERTAINTY_GOOD:
        fraction = (snr_db - SNR_UNCERTAINTY_GOOD) / (SNR_UNCERTAINTY_EXCELLENT - SNR_UNCERTAINTY_GOOD)
        uncertainty = UNCERTAINTY_GOOD + fraction * (UNCERTAINTY_EXCELLENT - UNCERTAINTY_GOOD)
    elif snr_db >= SNR_UNCERTAINTY_FAIR:
        fraction = (snr_db - SNR_UNCERTAINTY_FAIR) / (SNR_UNCERTAINTY_GOOD - SNR_UNCERTAINTY_FAIR)
        uncertainty = UNCERTAINTY_FAIR + fraction * (UNCERTAINTY_GOOD - UNCERTAINTY_FAIR)
    else:
        fraction = max(0.0, snr_db / SNR_UNCERTAINTY_FAIR)
        uncertainty = UNCERTAINTY_POOR - fraction * (UNCERTAINTY_POOR - UNCERTAINTY_FAIR)

    return uncertainty > REQUIRE_MULTILINK_UNCERTAINTY


# =============================================================================
# Near-Field Correction (Rank 7 RF Physics Enhancement)
# =============================================================================
#
# Physics basis (from RF_PHYSICS_TOPOLOGY_ANALYSIS.md):
# - Far-field boundary: d_far = 2·D²/λ
# - For ESP32 PCB antenna (D ≈ 5cm): d_far ≈ 0.04m
# - Conservative boundary: 0.5m to account for coupling effects
#
# In the near-field:
# - Fresnel zone model breaks down
# - Phase varies nonlinearly with distance
# - Amplitude varies as 1/d³ instead of 1/d²
#
# Reference: ITU-R P.526-15, Orfanidis "EM Waves and Antennas"

# Near-field boundary for ESP32 antennas (conservative estimate)
# Accounts for PCB antenna characteristics and coupling effects
NEAR_FIELD_BOUNDARY_M = 0.5
NEAR_FIELD_INVALID_DISTANCE_WEIGHT = 0.02


def near_field_correction(
    link_distance_m: float,
    boundary_m: float = NEAR_FIELD_BOUNDARY_M
) -> float:
    """
    Compute correction factor for near-field links.

    Links shorter than the near-field boundary have reduced reliability
    for Fresnel-based detection because the far-field assumption breaks down.

    Physics basis:
    - Far-field requires d >> λ and d >> antenna aperture
    - ESP32 PCB antenna: d_far ≈ 2·(0.05)²/0.125 ≈ 0.04m (theoretical)
    - We use 0.5m conservative boundary for practical sensing

    Args:
        link_distance_m: Distance between TX and RX in meters
        boundary_m: Near-field boundary distance (default: 0.5m)

    Returns:
        Correction factor [~0.1, 1.0]:
        - 1.0 for far-field links (d >= boundary)
        - Smoothly decreasing for near-field links
        - Approaches 0.5 at d=0 (invalid links get minimum weight)

    Example:
        # Far-field link (3m)
        factor = near_field_correction(3.0)  # Returns 1.0

        # Near-field link (0.3m)
        factor = near_field_correction(0.3)  # Returns ~0.69

        # Very short link (0.1m)
        factor = near_field_correction(0.1)  # Returns ~0.08 (severe penalty)
    """
    if link_distance_m >= boundary_m:
        return 1.0

    if link_distance_m <= 0:
        return NEAR_FIELD_INVALID_DISTANCE_WEIGHT

    # Smooth transition using tanh (sigmoid-like)
    # At ratio=0.5, returns ~0.75
    # At ratio=0, returns ~0.5
    # At ratio=1, returns ~1.0
    ratio = link_distance_m / boundary_m
    return 0.5 + 0.5 * float(np.tanh(4.0 * (ratio - 0.5)))


def compute_effective_detection_width_with_near_field(
    link_length_m: float,
    snr_db: float,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M,
    apply_near_field: bool = True
) -> float:
    """
    Compute effective detection width with optional near-field correction.

    This is an enhanced version of compute_effective_detection_width()
    that additionally applies near-field correction for short links.

    Args:
        link_length_m: Distance between TX and RX in meters
        snr_db: Current SNR in dB
        wavelength_m: Signal wavelength in meters
        apply_near_field: Whether to apply near-field correction

    Returns:
        Effective detection width in meters (reduced for near-field links)
    """
    base_width = compute_effective_detection_width(
        link_length_m, snr_db, wavelength_m
    )

    if apply_near_field:
        correction = near_field_correction(link_length_m)
        return base_width * correction

    return base_width


def is_near_field_link(link_distance_m: float) -> bool:
    """
    Check if a link is in the near-field region.

    Args:
        link_distance_m: Distance between TX and RX in meters

    Returns:
        True if the link is shorter than the near-field boundary
    """
    return link_distance_m < NEAR_FIELD_BOUNDARY_M
