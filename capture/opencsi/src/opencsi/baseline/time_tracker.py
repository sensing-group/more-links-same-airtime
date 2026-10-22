"""
Time-of-day bucket tracking for diurnal baseline calibration.

Environmental conditions (temperature, HVAC, sunlight, occupancy patterns)
vary predictably throughout the day. This module provides time-bucket
classification to enable separate baselines for different time periods.

Similar to frequency_tracker.py but for temporal variation instead of
sampling rate variation.
"""

from datetime import datetime
from typing import List, Tuple

# =============================================================================
# Time-of-Day Buckets (24-hour cycle)
# =============================================================================

# Time buckets representing distinct environmental periods
TIME_BUCKETS: List[Tuple[int, int]] = [
    (0, 6),      # Bucket 0: night (midnight-6am) - minimal activity, HVAC low
    (6, 12),     # Bucket 1: morning (6am-noon) - activity ramps up, HVAC starts
    (12, 18),    # Bucket 2: afternoon (noon-6pm) - peak activity, stable temp
    (18, 24),    # Bucket 3: evening (6pm-midnight) - wind-down, variable
]

NUM_TIME_BUCKETS = 4
DEFAULT_TIME_BUCKET = 2  # afternoon (most common observation period)

# Reference hours for interpolation (geometric center of each bucket)
BUCKET_REFERENCE_HOURS = [3.0, 9.0, 15.0, 21.0]

# Human-readable names for logging
TIME_BUCKET_NAMES = ["night", "morning", "afternoon", "evening"]

# Interpolation scaling factors (relative variance expected per bucket)
# Night tends to be quieter, afternoon more variable
TIME_BUCKET_VARIANCE_SCALE = [0.7, 0.9, 1.0, 0.85]


def timestamp_to_time_bucket(timestamp: float) -> int:
    """
    Map Unix timestamp to time-of-day bucket.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        Bucket index (0-3)
    """
    try:
        hour_of_day = datetime.fromtimestamp(timestamp).hour
    except (OSError, ValueError):
        # Invalid timestamp - use default
        return DEFAULT_TIME_BUCKET

    for idx, (min_h, max_h) in enumerate(TIME_BUCKETS):
        if min_h <= hour_of_day < max_h:
            return idx

    # Should not reach here, but fallback to night if hour >= 24 somehow
    return 0


def time_bucket_to_name(bucket_idx: int) -> str:
    """Get human-readable name for time bucket."""
    if 0 <= bucket_idx < NUM_TIME_BUCKETS:
        return TIME_BUCKET_NAMES[bucket_idx]
    return f"unknown_{bucket_idx}"


def get_hour_of_day(timestamp: float) -> float:
    """
    Get fractional hour of day from timestamp.

    Args:
        timestamp: Unix timestamp

    Returns:
        Hour of day as float (0.0 - 24.0)
    """
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    except (OSError, ValueError):
        return 12.0  # Default to noon


def get_adjacent_time_buckets(bucket_idx: int) -> List[int]:
    """
    Get adjacent time buckets for interpolation (circular).

    Time wraps around: bucket 3 (evening) is adjacent to bucket 0 (night).

    Args:
        bucket_idx: Current bucket index

    Returns:
        List of adjacent bucket indices, ordered by proximity
    """
    if bucket_idx < 0 or bucket_idx >= NUM_TIME_BUCKETS:
        return [DEFAULT_TIME_BUCKET]

    prev_bucket = (bucket_idx - 1) % NUM_TIME_BUCKETS
    next_bucket = (bucket_idx + 1) % NUM_TIME_BUCKETS

    return [prev_bucket, next_bucket]


def compute_time_interpolation_weight(
    source_bucket: int,
    target_bucket: int,
    current_hour: float
) -> float:
    """
    Compute interpolation weight between two time buckets.

    Weight is based on how close current time is to each bucket's
    reference hour (geometric center).

    Args:
        source_bucket: Bucket with calibrated baseline
        target_bucket: Bucket needing interpolation
        current_hour: Current hour of day (0-24)

    Returns:
        Weight [0, 1] for interpolation scaling
    """
    source_ref = BUCKET_REFERENCE_HOURS[source_bucket]
    target_ref = BUCKET_REFERENCE_HOURS[target_bucket]

    # Handle circular distance (e.g., 23:00 to 03:00)
    def circular_distance(h1: float, h2: float) -> float:
        diff = abs(h1 - h2)
        return min(diff, 24.0 - diff)

    source_dist = circular_distance(current_hour, source_ref)
    target_dist = circular_distance(current_hour, target_ref)

    # Inverse distance weighting
    total_dist = source_dist + target_dist
    if total_dist < 0.01:
        return 1.0

    # Weight is inversely proportional to source distance
    weight = 1.0 - (source_dist / total_dist)

    # Apply variance scaling (some buckets naturally more variable)
    source_scale = TIME_BUCKET_VARIANCE_SCALE[source_bucket]
    target_scale = TIME_BUCKET_VARIANCE_SCALE[target_bucket]
    scale_ratio = target_scale / source_scale if source_scale > 0 else 1.0

    return min(1.0, weight * scale_ratio)
