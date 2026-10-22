"""
Time-of-day aware baseline calibration.

Environmental conditions vary by time of day:
- Night (0-6h): Minimal activity, HVAC low, stable
- Morning (6-12h): Activity ramps up, HVAC starts
- Afternoon (12-18h): Peak activity, temperature stable
- Evening (18-24h): Wind-down, variable conditions

This module maintains separate baselines for each time period,
enabling more accurate Z-score calculation by comparing current
measurements against the appropriate time-of-day baseline.

Follows the same pattern as frequency_baseline.py but for temporal
variation instead of sampling rate variation.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from opencsi.baseline.base import BaseBucketStats
from opencsi.baseline.time_tracker import (
    NUM_TIME_BUCKETS,
    DEFAULT_TIME_BUCKET,
    TIME_BUCKET_VARIANCE_SCALE,
    timestamp_to_time_bucket,
    time_bucket_to_name,
    get_adjacent_time_buckets,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Minimum samples per bucket before considered calibrated
TIME_MIN_QUIET_SAMPLES = 20

# EMA learning rate for time-aware baseline updates
TIME_BASELINE_LEARNING_RATE = 0.05

# Minimum time between updates to same bucket (prevents rapid fluctuation)
TIME_MIN_UPDATE_INTERVAL_S = 60.0

# Interpolation defaults when no calibration exists
TIME_DEFAULT_AMPLITUDE_STD = 1.0
TIME_DEFAULT_PHASE_STD = 1.0

# Health threshold - bucket considered healthy if this many quiet samples
TIME_BUCKET_HEALTH_SAMPLES = 50

# Number of recent samples to keep for running std calculation
TIME_AMPLITUDE_HISTORY_SIZE = 50


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimeBucketBaseline(BaseBucketStats):
    """
    Baseline statistics for a single time-of-day bucket.

    Inherits core statistics from BaseBucketStats and adds
    time-specific running sample history for std estimation.
    """

    # Recent samples for stability analysis
    recent_amplitude_samples: deque = field(
        default_factory=lambda: deque(maxlen=TIME_AMPLITUDE_HISTORY_SIZE)
    )

    @property
    def is_calibrated(self) -> bool:
        """True if bucket has enough samples to be trusted."""
        return self.is_initialized and self.quiet_samples >= TIME_MIN_QUIET_SAMPLES

    @property
    def health(self) -> float:
        """Calibration health [0, 1] based on sample count."""
        if not self.is_initialized:
            return 0.0
        return min(1.0, self.quiet_samples / TIME_BUCKET_HEALTH_SAMPLES)

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        d = self._base_to_dict()
        d['recent_amplitude_samples'] = list(self.recent_amplitude_samples)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'TimeBucketBaseline':
        """Deserialize from persistence."""
        baseline = cls()
        baseline._base_from_dict(data)
        saved_samples = data.get('recent_amplitude_samples', [])
        baseline.recent_amplitude_samples = deque(
            saved_samples, maxlen=TIME_AMPLITUDE_HISTORY_SIZE
        )
        return baseline


@dataclass
class TimeAwareBaseline:
    """Time-of-day aware baseline for a single link."""

    tx_mac: str = ""
    rx_mac: str = ""

    # Per-bucket baselines (4 time periods per day)
    bucket_baselines: List[TimeBucketBaseline] = field(
        default_factory=lambda: [TimeBucketBaseline() for _ in range(NUM_TIME_BUCKETS)]
    )

    def get_baseline(self, bucket_idx: int) -> TimeBucketBaseline:
        """Get baseline for specific time bucket."""
        if 0 <= bucket_idx < NUM_TIME_BUCKETS:
            return self.bucket_baselines[bucket_idx]
        return self.bucket_baselines[DEFAULT_TIME_BUCKET]

    def get_baseline_with_interpolation(
        self,
        bucket_idx: int,
        timestamp: float
    ) -> Tuple[TimeBucketBaseline, bool, int]:
        """
        Get baseline with interpolation if bucket not calibrated.

        Args:
            bucket_idx: Target time bucket
            timestamp: Current timestamp (for interpolation weight)

        Returns:
            (baseline, was_interpolated, source_bucket)
        """
        target = self.get_baseline(bucket_idx)

        # If calibrated, use directly
        if target.is_calibrated:
            return target, False, bucket_idx

        # Try adjacent buckets
        adjacent = get_adjacent_time_buckets(bucket_idx)

        for adj_bucket in adjacent:
            adj_baseline = self.get_baseline(adj_bucket)
            if adj_baseline.is_calibrated:
                interpolated = self._interpolate_baseline(
                    adj_baseline, adj_bucket, bucket_idx
                )
                return interpolated, True, adj_bucket

        # No calibration available - return theoretical default
        return self._get_default_baseline(bucket_idx), True, -1

    def _interpolate_baseline(
        self,
        source: TimeBucketBaseline,
        source_bucket: int,
        target_bucket: int,
    ) -> TimeBucketBaseline:
        """
        Interpolate baseline from source bucket to target bucket.

        Uses variance scaling to account for different activity levels
        at different times of day.
        """
        # Scale variance based on time-of-day activity patterns
        source_scale = TIME_BUCKET_VARIANCE_SCALE[source_bucket]
        target_scale = TIME_BUCKET_VARIANCE_SCALE[target_bucket]
        variance_ratio = target_scale / source_scale if source_scale > 0 else 1.0

        interpolated = TimeBucketBaseline()
        interpolated.amplitude_mean = source.amplitude_mean
        interpolated.amplitude_std = source.amplitude_std * np.sqrt(variance_ratio)
        interpolated.phase_std = source.phase_std * np.sqrt(variance_ratio)
        interpolated.is_initialized = True
        interpolated.quiet_samples = max(1, source.quiet_samples // 2)  # Reduce confidence

        return interpolated

    def _get_default_baseline(self, bucket_idx: int) -> TimeBucketBaseline:
        """Get theoretical default when no calibration exists."""
        default = TimeBucketBaseline()
        default.amplitude_mean = 0.5  # Arbitrary default
        default.amplitude_std = TIME_DEFAULT_AMPLITUDE_STD
        default.phase_std = TIME_DEFAULT_PHASE_STD
        default.is_initialized = False  # Mark as uncalibrated
        return default

    def get_calibration_health(self) -> float:
        """
        Overall calibration health [0, 1].

        1.0 = all buckets calibrated with many samples
        0.0 = no buckets calibrated
        """
        healths = [b.health for b in self.bucket_baselines]
        return float(np.mean(healths))

    def get_calibrated_buckets(self) -> List[int]:
        """Get list of calibrated bucket indices."""
        return [
            i for i, b in enumerate(self.bucket_baselines)
            if b.is_calibrated
        ]

    def get_uncalibrated_buckets(self) -> List[int]:
        """Get list of uncalibrated bucket indices."""
        return [
            i for i, b in enumerate(self.bucket_baselines)
            if not b.is_calibrated
        ]

    def update_bucket(
        self,
        bucket_idx: int,
        amplitude_mean: float,
        phase_std: float,
        timestamp: float,
        learning_rate: float = TIME_BASELINE_LEARNING_RATE
    ) -> bool:
        """
        Update a time bucket baseline with new sample (EMA).

        Args:
            bucket_idx: Time bucket to update
            amplitude_mean: Mean amplitude from current sample
            phase_std: Phase difference std from current sample
            timestamp: Sample timestamp
            learning_rate: EMA learning rate (default 0.05)

        Returns:
            True if update was applied, False if skipped
        """
        if bucket_idx < 0 or bucket_idx >= NUM_TIME_BUCKETS:
            return False

        baseline = self.bucket_baselines[bucket_idx]

        # Rate limit updates
        if (baseline.last_update_time > 0 and
                timestamp - baseline.last_update_time < TIME_MIN_UPDATE_INTERVAL_S):
            return False

        if not baseline.is_initialized:
            # First sample - initialize
            baseline.amplitude_mean = amplitude_mean
            # Estimate initial variance from single sample
            baseline.amplitude_std = max(0.1, abs(amplitude_mean) * 0.2)
            baseline.phase_std = phase_std if phase_std > 0 else TIME_DEFAULT_PHASE_STD
            baseline.is_initialized = True
            baseline.creation_time = timestamp
            baseline.quiet_samples = 1
            baseline.total_samples = 1
            baseline.last_update_time = timestamp
        else:
            # EMA update
            alpha = learning_rate
            baseline.amplitude_mean = (
                (1 - alpha) * baseline.amplitude_mean + alpha * amplitude_mean
            )

            # Update std using running variance estimate
            baseline.recent_amplitude_samples.append(amplitude_mean)
            if len(baseline.recent_amplitude_samples) >= 10:
                baseline.amplitude_std = max(
                    0.01, float(np.std(baseline.recent_amplitude_samples))
                )

            # Update phase std via variance EMA (same as amplitude)
            phase_var = (
                (1 - alpha) * baseline.phase_std ** 2 + alpha * phase_std ** 2
            )
            baseline.phase_std = max(math.sqrt(phase_var), 0.01)

            baseline.quiet_samples += 1
            baseline.total_samples += 1
            baseline.last_update_time = timestamp

        return True

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'tx_mac': self.tx_mac,
            'rx_mac': self.rx_mac,
            'bucket_baselines': [b.to_dict() for b in self.bucket_baselines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TimeAwareBaseline':
        """Deserialize from persistence."""
        baseline = cls()
        baseline.tx_mac = data.get('tx_mac', '')
        baseline.rx_mac = data.get('rx_mac', '')
        buckets = data.get('bucket_baselines', [])
        for i, bucket_data in enumerate(buckets):
            if i < NUM_TIME_BUCKETS:
                baseline.bucket_baselines[i] = TimeBucketBaseline.from_dict(bucket_data)
        return baseline


# =============================================================================
# Passive Time Collector (manages all links)
# =============================================================================

class PassiveTimeCollector:
    """
    Collects time-of-day aware baselines for all links.

    Passively learns baselines during quiet periods, building up
    per-bucket calibration over days of operation.

    Similar to PassiveFrequencyCollector but for time-of-day.
    """

    def __init__(
        self,
        learning_rate: float = TIME_BASELINE_LEARNING_RATE,
    ):
        self.learning_rate = learning_rate
        self._baselines: Dict[Tuple[str, str], TimeAwareBaseline] = {}
        self._total_samples = 0

    def get_or_create_baseline(
        self,
        link_key: Tuple[str, str]
    ) -> TimeAwareBaseline:
        """Get or create time-aware baseline for link."""
        if link_key not in self._baselines:
            self._baselines[link_key] = TimeAwareBaseline(
                tx_mac=link_key[0],
                rx_mac=link_key[1]
            )
        return self._baselines[link_key]

    def on_sample(
        self,
        link_key: Tuple[str, str],
        amplitude: float,
        phase_std: float,
        timestamp: float,
        is_quiet: bool
    ) -> None:
        """
        Process a new sample.

        Only updates baselines during quiet periods.

        Args:
            link_key: (tx_mac, rx_mac) tuple
            amplitude: Mean amplitude value
            phase_std: Phase difference std
            timestamp: Sample timestamp
            is_quiet: True if environment is quiet (safe to update baseline)
        """
        self._total_samples += 1

        if not is_quiet:
            # Still track total samples for coverage
            baseline = self.get_or_create_baseline(link_key)
            bucket_idx = timestamp_to_time_bucket(timestamp)
            baseline.bucket_baselines[bucket_idx].total_samples += 1
            return

        # Update baseline during quiet period
        baseline = self.get_or_create_baseline(link_key)
        bucket_idx = timestamp_to_time_bucket(timestamp)

        baseline.update_bucket(
            bucket_idx=bucket_idx,
            amplitude_mean=amplitude,
            phase_std=phase_std,
            timestamp=timestamp,
            learning_rate=self.learning_rate
        )

    def get_baseline_with_interpolation(
        self,
        link_key: Tuple[str, str],
        timestamp: float
    ) -> Tuple[TimeBucketBaseline, bool, int]:
        """
        Get appropriate baseline for link at given time.

        Returns:
            (baseline, was_interpolated, source_bucket)
        """
        if link_key not in self._baselines:
            # No baseline - return default
            default = TimeBucketBaseline()
            return default, True, -1

        bucket_idx = timestamp_to_time_bucket(timestamp)
        return self._baselines[link_key].get_baseline_with_interpolation(
            bucket_idx, timestamp
        )

    def get_calibration_health(self, link_key: Tuple[str, str]) -> float:
        """Get calibration health for link [0, 1]."""
        if link_key not in self._baselines:
            return 0.0
        return self._baselines[link_key].get_calibration_health()

    def get_overall_health(self) -> float:
        """Get mean calibration health across all links."""
        if not self._baselines:
            return 0.0
        healths = [b.get_calibration_health() for b in self._baselines.values()]
        return float(np.mean(healths))

    def get_stats(self) -> dict:
        """Get collector statistics."""
        total_links = len(self._baselines)
        calibrated_links = sum(
            1 for b in self._baselines.values()
            if b.get_calibration_health() > 0.5
        )

        bucket_coverage = [0] * NUM_TIME_BUCKETS
        for baseline in self._baselines.values():
            for i, bucket in enumerate(baseline.bucket_baselines):
                if bucket.is_calibrated:
                    bucket_coverage[i] += 1

        return {
            'total_samples': self._total_samples,
            'total_links': total_links,
            'calibrated_links': calibrated_links,
            'overall_health': self.get_overall_health(),
            'bucket_coverage': {
                time_bucket_to_name(i): count
                for i, count in enumerate(bucket_coverage)
            },
        }

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'learning_rate': self.learning_rate,
            'total_samples': self._total_samples,
            'baselines': {
                f"{k[0]}->{k[1]}": v.to_dict()
                for k, v in self._baselines.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PassiveTimeCollector':
        """Deserialize from persistence."""
        collector = cls(
            learning_rate=data.get('learning_rate', TIME_BASELINE_LEARNING_RATE)
        )
        collector._total_samples = data.get('total_samples', 0)

        baselines_data = data.get('baselines', {})
        for key_str, baseline_data in baselines_data.items():
            # Parse "tx_mac->rx_mac" format
            if '->' in key_str:
                tx, rx = key_str.split('->', 1)
                link_key = (tx, rx)
                collector._baselines[link_key] = TimeAwareBaseline.from_dict(baseline_data)

        return collector
