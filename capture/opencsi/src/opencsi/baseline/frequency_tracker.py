"""
Frequency Tracking for OpenCSI AutoCalibrator.

CSI statistics vary with sampling frequency - baselines learned at 50Hz are
invalid at 2Hz. This module provides frequency bucket classification for
frequency-aware calibration.

Key insight: Dynamic schedulers vary sampling rates (2Hz-60Hz+).
We need separate baseline statistics for each frequency regime.
"""

from typing import Tuple, List

# =============================================================================
# Frequency Bucket Configuration
# =============================================================================

# Frequency buckets: (min_hz, max_hz) - exclusive upper bound
# These represent distinct sampling regimes with different noise characteristics
FREQUENCY_BUCKETS: List[Tuple[float, float]] = [
    (0.0, 2.0),           # Bucket 0: idle (very sparse sampling)
    (2.0, 8.0),           # Bucket 1: background (low priority)
    (8.0, 15.0),          # Bucket 2: normal (standard operation)
    (15.0, 30.0),         # Bucket 3: elevated (active tracking)
    (30.0, 60.0),         # Bucket 4: focus (high priority link)
    (60.0, float('inf')), # Bucket 5: burst (maximum rate)
]

# Human-readable names for each bucket
BUCKET_NAMES: List[str] = [
    "idle",
    "background",
    "normal",
    "elevated",
    "focus",
    "burst",
]

# Number of frequency buckets
NUM_FREQUENCY_BUCKETS = len(FREQUENCY_BUCKETS)

# Default bucket when frequency is unknown or insufficient data
DEFAULT_FREQUENCY_BUCKET = 2  # "normal"

# Reference frequencies for each bucket (geometric mean of bucket range)
# Used for interpolation when a bucket has no data
BUCKET_REFERENCE_FREQUENCIES: List[float] = [
    1.0,    # idle: ~1 Hz
    4.0,    # background: ~4 Hz
    11.0,   # normal: ~11 Hz
    22.0,   # elevated: ~22 Hz
    42.0,   # focus: ~42 Hz
    80.0,   # burst: ~80 Hz
]


def frequency_to_bucket(freq_hz: float) -> int:
    """
    Map instantaneous frequency to a bucket index.

    Args:
        freq_hz: Instantaneous sampling frequency in Hz

    Returns:
        Bucket index [0, NUM_FREQUENCY_BUCKETS-1]

    Examples:
        >>> frequency_to_bucket(1.5)
        0  # idle
        >>> frequency_to_bucket(10.0)
        2  # normal
        >>> frequency_to_bucket(50.0)
        4  # focus
        >>> frequency_to_bucket(100.0)
        5  # burst
    """
    if freq_hz <= 0:
        return 0  # Invalid frequency -> idle bucket

    for idx, (min_hz, max_hz) in enumerate(FREQUENCY_BUCKETS):
        if min_hz <= freq_hz < max_hz:
            return idx

    # Shouldn't reach here, but return highest bucket for very high frequencies
    return NUM_FREQUENCY_BUCKETS - 1


def bucket_to_name(bucket_idx: int) -> str:
    """
    Get human-readable name for a frequency bucket.

    Args:
        bucket_idx: Bucket index [0, NUM_FREQUENCY_BUCKETS-1]

    Returns:
        Human-readable bucket name

    Examples:
        >>> bucket_to_name(0)
        'idle'
        >>> bucket_to_name(2)
        'normal'
    """
    if 0 <= bucket_idx < len(BUCKET_NAMES):
        return BUCKET_NAMES[bucket_idx]
    return f"unknown_{bucket_idx}"


def bucket_to_frequency_range(bucket_idx: int) -> Tuple[float, float]:
    """
    Get the frequency range for a bucket.

    Args:
        bucket_idx: Bucket index

    Returns:
        (min_hz, max_hz) tuple for the bucket
    """
    if 0 <= bucket_idx < len(FREQUENCY_BUCKETS):
        return FREQUENCY_BUCKETS[bucket_idx]
    return (0.0, float('inf'))


def get_bucket_reference_frequency(bucket_idx: int) -> float:
    """
    Get the reference frequency for a bucket.

    This is used for interpolation when computing baselines for
    buckets that don't have enough samples.

    Args:
        bucket_idx: Bucket index

    Returns:
        Reference frequency in Hz for the bucket
    """
    if 0 <= bucket_idx < len(BUCKET_REFERENCE_FREQUENCIES):
        return BUCKET_REFERENCE_FREQUENCIES[bucket_idx]
    return 10.0  # Default to ~10 Hz


def compute_bucket_distance(bucket_a: int, bucket_b: int) -> int:
    """
    Compute the distance between two buckets.

    Used for interpolation weighting - closer buckets have more relevance.

    Args:
        bucket_a: First bucket index
        bucket_b: Second bucket index

    Returns:
        Absolute bucket distance
    """
    return abs(bucket_a - bucket_b)


def get_adjacent_buckets(bucket_idx: int) -> List[int]:
    """
    Get adjacent bucket indices for interpolation fallback.

    Returns buckets in order of proximity (closest first).

    Args:
        bucket_idx: Target bucket index

    Returns:
        List of adjacent bucket indices, closest first
    """
    adjacent = []

    # Add neighbors in order of proximity
    for distance in range(1, NUM_FREQUENCY_BUCKETS):
        # Lower neighbor
        lower = bucket_idx - distance
        if 0 <= lower < NUM_FREQUENCY_BUCKETS:
            adjacent.append(lower)

        # Upper neighbor
        upper = bucket_idx + distance
        if 0 <= upper < NUM_FREQUENCY_BUCKETS:
            adjacent.append(upper)

    return adjacent
