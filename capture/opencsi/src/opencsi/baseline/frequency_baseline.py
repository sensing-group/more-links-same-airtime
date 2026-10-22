"""
Frequency-Aware Baseline Management for OpenCSI AutoCalibrator.

This module provides frequency-binned baseline statistics. CSI statistics vary
with sampling frequency - baselines learned at 50Hz are invalid at 2Hz.

Key Components:
- BucketBaseline: Statistics for a single frequency bucket
- FrequencyAwareBaseline: Collection of bucket baselines with interpolation
- PassiveFrequencyCollector: Mode 1 - always-running passive collection

Design Principle: GENERALIZABLE. Works with ANY scheduler. Calibration requests
are optional hints, not requirements.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from opencsi.baseline.base import BaseBucketStats
from opencsi.baseline.frequency_tracker import (
    NUM_FREQUENCY_BUCKETS,
    DEFAULT_FREQUENCY_BUCKET,
    frequency_to_bucket,
    bucket_to_name,
    get_adjacent_buckets,
    get_bucket_reference_frequency,
    BUCKET_REFERENCE_FREQUENCIES,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Tiered retention policy for baseline samples
RECENT_SAMPLES_TO_KEEP = 20  # Keep recent samples at full resolution
HOURLY_AGGREGATES_TO_KEEP = 24  # Keep hourly aggregates for the last 24 hours
DAILY_AGGREGATES_TO_KEEP = 30  # Keep daily aggregates for the last 30 days

# Minimum samples required for a bucket baseline to be considered initialized
MIN_BUCKET_SAMPLES_FOR_INIT = 5

# Minimum quiet samples for confident baseline
MIN_BUCKET_QUIET_SAMPLES_FOR_CONFIDENCE = 20

# Baseline age decay constant (1-hour half-life)
BASELINE_AGE_DECAY_LAMBDA = 1.0 / 3600.0

# Maximum age before baseline is considered stale (24 hours)
MAX_BASELINE_AGE_S = 86400.0

# Minimum standard deviation threshold (avoid divide-by-zero)
MIN_STD_THRESHOLD = 0.1

# EMA learning rate for passive collection (slow, conservative)
PASSIVE_LEARNING_RATE = 0.05

# EMA learning rate for quiet-period updates (faster)
QUIET_LEARNING_RATE = 0.1

# IPI (Inter-Packet Interval) history size for frequency estimation
IPI_HISTORY_SIZE = 20


@dataclass
class AggregatedStats:
    """Statistical aggregate of baseline data over a time period."""
    mean_amplitude: float
    std_amplitude: float
    sample_count: int
    timestamp: float
    period: str  # 'hourly' or 'daily'

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'mean_amplitude': self.mean_amplitude,
            'std_amplitude': self.std_amplitude,
            'sample_count': self.sample_count,
            'timestamp': self.timestamp,
            'period': self.period,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AggregatedStats':
        """Deserialize from dictionary."""
        return cls(
            mean_amplitude=data['mean_amplitude'],
            std_amplitude=data['std_amplitude'],
            sample_count=data['sample_count'],
            timestamp=data['timestamp'],
            period=data['period'],
        )


@dataclass
class BucketBaseline(BaseBucketStats):
    """
    Baseline statistics for a single frequency bucket.

    Each frequency bucket has its own statistics because CSI noise
    characteristics vary significantly with sampling rate.

    Inherits core statistics from BaseBucketStats and adds
    frequency-specific data retention (hourly/daily aggregation).
    """
    # Data retention and aggregation
    last_hourly_aggregation_time: float = 0.0
    last_daily_aggregation_time: float = 0.0
    recent_samples: deque = field(default_factory=lambda: deque(maxlen=RECENT_SAMPLES_TO_KEEP))  # (timestamp, amplitude)
    hourly_aggregates: List[AggregatedStats] = field(default_factory=list)
    daily_aggregates: List[AggregatedStats] = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        """Check if baseline is too old to be reliable."""
        return self.age_seconds > MAX_BASELINE_AGE_S

    @property
    def has_confident_baseline(self) -> bool:
        """Check if we have enough quiet samples for confidence."""
        return (self.is_initialized and
                self.quiet_samples >= MIN_BUCKET_QUIET_SAMPLES_FOR_CONFIDENCE)

    def get_sample_confidence(self) -> float:
        """
        Get confidence factor based on sample count.

        Returns:
            Confidence in [0, 1] based on quiet sample count
        """
        if not self.is_initialized:
            return 0.0

        return min(1.0, self.quiet_samples / MIN_BUCKET_QUIET_SAMPLES_FOR_CONFIDENCE)

    def get_recency_factor(self) -> float:
        """
        Get recency factor (exponential decay with age).

        Returns:
            Recency factor in [0, 1]
        """
        if not self.is_initialized:
            return 0.0

        age = self.age_seconds
        if age > MAX_BASELINE_AGE_S:
            return 0.0

        return math.exp(-BASELINE_AGE_DECAY_LAMBDA * age)

    def _maybe_aggregate(self, timestamp: float):
        """
        Aggregate recent samples into hourly/daily stats if needed.
        This method is designed to be testable by using the provided timestamp.
        """
        # Initialize aggregation times if they are not set
        if self.last_hourly_aggregation_time == 0:
            self.last_hourly_aggregation_time = timestamp
        if self.last_daily_aggregation_time == 0:
            self.last_daily_aggregation_time = timestamp

        # Hourly aggregation
        if timestamp - self.last_hourly_aggregation_time >= 3600:
            if self.recent_samples:
                amplitudes = [s[1] for s in self.recent_samples]
                mean_amp = float(np.mean(amplitudes))
                std_amp = float(np.std(amplitudes))

                hourly_stat = AggregatedStats(
                    mean_amplitude=mean_amp,
                    std_amplitude=std_amp,
                    sample_count=len(self.recent_samples),
                    timestamp=timestamp,
                    period='hourly'
                )
                self.hourly_aggregates.append(hourly_stat)
                self.hourly_aggregates = self.hourly_aggregates[-HOURLY_AGGREGATES_TO_KEEP:]
                self.recent_samples.clear()
            self.last_hourly_aggregation_time = timestamp

        # Daily aggregation (based on hourly data)
        if timestamp - self.last_daily_aggregation_time >= 86400:
            # Use hourly aggregates from the last 24 hours
            relevant_hourly = [
                h for h in self.hourly_aggregates
                if timestamp - h.timestamp < 86400
            ]
            if relevant_hourly:
                mean_amp = float(np.mean([h.mean_amplitude for h in relevant_hourly]))
                # Law of total variance: Var[X] = E[Var] + Var[E]
                hourly_vars = np.array([h.std_amplitude ** 2 for h in relevant_hourly])
                hourly_means = np.array([h.mean_amplitude for h in relevant_hourly])
                combined_var = float(np.mean(hourly_vars) + np.var(hourly_means))
                std_amp = float(np.sqrt(combined_var))
                sample_count = sum(h.sample_count for h in relevant_hourly)

                daily_stat = AggregatedStats(
                    mean_amplitude=mean_amp,
                    std_amplitude=std_amp,
                    sample_count=sample_count,
                    timestamp=timestamp,
                    period='daily'
                )
                self.daily_aggregates.append(daily_stat)
                self.daily_aggregates = self.daily_aggregates[-DAILY_AGGREGATES_TO_KEEP:]
            self.last_daily_aggregation_time = timestamp

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = self._base_to_dict()
        d.update({
            'last_hourly_aggregation_time': self.last_hourly_aggregation_time,
            'last_daily_aggregation_time': self.last_daily_aggregation_time,
            'recent_samples': list(self.recent_samples),
            'hourly_aggregates': [h.to_dict() for h in self.hourly_aggregates],
            'daily_aggregates': [d_.to_dict() for d_ in self.daily_aggregates],
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'BucketBaseline':
        """Deserialize from dictionary."""
        instance = cls(
            last_hourly_aggregation_time=data.get('last_hourly_aggregation_time', data.get('last_aggregation_time', 0.0)),
            last_daily_aggregation_time=data.get('last_daily_aggregation_time', data.get('last_aggregation_time', 0.0)),
            recent_samples=deque(data.get('recent_samples', []), maxlen=RECENT_SAMPLES_TO_KEEP),
        )
        instance._base_from_dict(data)
        instance.hourly_aggregates = [
            AggregatedStats.from_dict(h) for h in data.get('hourly_aggregates', [])
        ]
        instance.daily_aggregates = [
            AggregatedStats.from_dict(d) for d in data.get('daily_aggregates', [])
        ]
        return instance


@dataclass
class FrequencyAwareBaseline:
    """
    Frequency-aware baseline statistics for a single link.

    Maintains separate baseline statistics for each frequency bucket,
    with interpolation fallback for uncalibrated buckets.
    """
    tx_mac: str
    rx_mac: str

    # Per-bucket baselines
    bucket_baselines: List[BucketBaseline] = field(
        default_factory=lambda: [BucketBaseline() for _ in range(NUM_FREQUENCY_BUCKETS)]
    )

    @property
    def link_key(self) -> Tuple[str, str]:
        """Return (tx_mac, rx_mac) tuple."""
        return (self.tx_mac, self.rx_mac)

    def get_baseline(self, bucket_idx: int) -> BucketBaseline:
        """
        Get baseline for a specific bucket.

        Args:
            bucket_idx: Frequency bucket index

        Returns:
            BucketBaseline for the requested bucket
        """
        if 0 <= bucket_idx < len(self.bucket_baselines):
            return self.bucket_baselines[bucket_idx]
        return self.bucket_baselines[DEFAULT_FREQUENCY_BUCKET]

    def get_baseline_with_interpolation(
        self, bucket_idx: int
    ) -> Tuple[BucketBaseline, bool, int]:
        """
        Get baseline with interpolation fallback for uncalibrated buckets.

        If the requested bucket doesn't have data, interpolates from
        adjacent buckets that do have data.

        Args:
            bucket_idx: Target frequency bucket index

        Returns:
            Tuple of (baseline, was_interpolated, bucket_used)
        """
        # First, check if target bucket has data
        target_baseline = self.get_baseline(bucket_idx)
        if target_baseline.is_initialized:
            return target_baseline, False, bucket_idx

        # Need to interpolate - find nearest bucket with data
        adjacent = get_adjacent_buckets(bucket_idx)

        for adj_bucket in adjacent:
            adj_baseline = self.get_baseline(adj_bucket)
            if adj_baseline.is_initialized:
                # Use this bucket's data with interpolation adjustment
                interpolated = self._interpolate_baseline(
                    adj_baseline, adj_bucket, bucket_idx
                )
                return interpolated, True, adj_bucket

        # No calibrated buckets - return theoretical defaults
        return self._get_theoretical_default(bucket_idx), True, -1

    def _interpolate_baseline(
        self,
        source_baseline: BucketBaseline,
        source_bucket: int,
        target_bucket: int
    ) -> BucketBaseline:
        """
        Interpolate baseline from source bucket to target bucket.

        Adjusts statistics based on frequency ratio. Higher frequencies
        generally have lower variance (more averaging).

        Args:
            source_baseline: Baseline to interpolate from
            source_bucket: Source bucket index
            target_bucket: Target bucket index

        Returns:
            Interpolated BucketBaseline
        """
        source_freq = get_bucket_reference_frequency(source_bucket)
        target_freq = get_bucket_reference_frequency(target_bucket)

        # Frequency ratio for scaling
        # Higher frequency -> more samples averaged -> lower std
        freq_ratio = source_freq / max(target_freq, 0.1)
        std_scale = np.sqrt(freq_ratio)  # Variance scales linearly, std scales as sqrt

        # Create interpolated baseline
        return BucketBaseline(
            amplitude_mean=source_baseline.amplitude_mean,
            amplitude_std=max(source_baseline.amplitude_std * std_scale, MIN_STD_THRESHOLD),
            phase_std=max(source_baseline.phase_std * std_scale, MIN_STD_THRESHOLD),
            quiet_samples=0,  # Not trusted - interpolated
            total_samples=0,
            last_update_time=source_baseline.last_update_time,
            creation_time=source_baseline.creation_time,
            is_initialized=True,  # Usable but not directly calibrated
        )

    def _get_theoretical_default(self, bucket_idx: int) -> BucketBaseline:
        """
        Get theoretical default baseline when no calibration data exists.

        Uses typical values based on ESP32 CSI characteristics.

        Args:
            bucket_idx: Target bucket index

        Returns:
            BucketBaseline with theoretical defaults
        """
        # Typical ESP32 CSI values (from hardware profiles)
        ref_freq = get_bucket_reference_frequency(bucket_idx)

        # Scale std based on frequency (higher freq = lower noise due to averaging)
        freq_scale = np.sqrt(10.0 / max(ref_freq, 1.0))

        return BucketBaseline(
            amplitude_mean=10.0,  # Typical amplitude (arbitrary units)
            amplitude_std=max(2.5 * freq_scale, MIN_STD_THRESHOLD),
            phase_std=max(0.8 * freq_scale, MIN_STD_THRESHOLD),
            quiet_samples=0,
            total_samples=0,
            last_update_time=0.0,
            creation_time=0.0,
            is_initialized=False,  # Mark as not calibrated
        )

    def update_bucket(
        self,
        bucket_idx: int,
        amplitude: float,
        phase_std: float,
        is_quiet: bool,
        learning_rate: float,
        timestamp: float
    ):
        """
        Update baseline for a specific frequency bucket.

        Args:
            bucket_idx: Frequency bucket to update
            amplitude: Mean amplitude of current sample
            phase_std: Phase differential std of current sample
            is_quiet: Whether environment is currently quiet
            learning_rate: EMA learning rate
            timestamp: Current timestamp
        """
        if not 0 <= bucket_idx < len(self.bucket_baselines):
            return

        baseline = self.bucket_baselines[bucket_idx]
        baseline.total_samples += 1

        if is_quiet:
            baseline.quiet_samples += 1

            if not baseline.is_initialized:
                # First initialization
                baseline.amplitude_mean = amplitude
                baseline.amplitude_std = max(abs(amplitude) * 0.1, MIN_STD_THRESHOLD)
                baseline.phase_std = max(phase_std, MIN_STD_THRESHOLD)
                baseline.is_initialized = True
                baseline.creation_time = timestamp
            else:
                # EMA update for quiet samples
                alpha = learning_rate
                old_mean = baseline.amplitude_mean
                baseline.amplitude_mean = (
                    (1 - alpha) * old_mean +
                    alpha * amplitude
                )
                # Update std using EMA of squared deviation (variance), then sqrt
                # Use OLD mean for deviation — using new mean underestimates variance
                deviation_sq = (amplitude - old_mean) ** 2
                variance = (
                    (1 - alpha) * baseline.amplitude_std ** 2 +
                    alpha * deviation_sq
                )
                baseline.amplitude_std = max(
                    math.sqrt(variance), MIN_STD_THRESHOLD
                )
                if phase_std > 0:
                    phase_var = (
                        (1 - alpha) * baseline.phase_std ** 2 +
                        alpha * phase_std ** 2
                    )
                    baseline.phase_std = max(
                        math.sqrt(phase_var), MIN_STD_THRESHOLD
                    )

            baseline.last_update_time = timestamp

        # Store recent sample for aggregation (deque auto-evicts oldest)
        baseline.recent_samples.append((timestamp, amplitude))

        # Trigger aggregation logic
        baseline._maybe_aggregate(timestamp)

    def get_calibration_confidence(self, bucket_idx: int) -> float:
        """
        Get overall calibration confidence for a bucket.

        Confidence = sample_factor x quiet_factor x recency_factor

        Args:
            bucket_idx: Frequency bucket index

        Returns:
            Confidence score in [0, 1]
        """
        baseline = self.get_baseline(bucket_idx)

        if not baseline.is_initialized:
            return 0.0

        sample_factor = baseline.get_sample_confidence()
        recency_factor = baseline.get_recency_factor()

        # Quiet ratio factor
        if baseline.total_samples > 0:
            quiet_ratio = baseline.quiet_samples / baseline.total_samples
        else:
            quiet_ratio = 0.0

        return sample_factor * recency_factor * (0.5 + 0.5 * quiet_ratio)

    def get_uncalibrated_buckets(self) -> List[int]:
        """
        Get list of buckets that need calibration.

        Returns:
            List of bucket indices that are not initialized or have low confidence
        """
        uncalibrated = []
        for idx, baseline in enumerate(self.bucket_baselines):
            if not baseline.is_initialized or self.get_calibration_confidence(idx) < 0.3:
                uncalibrated.append(idx)
        return uncalibrated

    def get_calibration_health(self) -> float:
        """
        Get overall frequency calibration health.

        Returns:
            Health score in [0, 1] based on bucket coverage and confidence
        """
        if not self.bucket_baselines:
            return 0.0

        confidences = [
            self.get_calibration_confidence(idx)
            for idx in range(NUM_FREQUENCY_BUCKETS)
        ]

        # Weight by typical bucket usage (normal/elevated are most important)
        weights = [0.05, 0.1, 0.3, 0.3, 0.2, 0.05]

        weighted_health = sum(c * w for c, w in zip(confidences, weights))
        return weighted_health

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            'tx_mac': self.tx_mac,
            'rx_mac': self.rx_mac,
            'bucket_baselines': [b.to_dict() for b in self.bucket_baselines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FrequencyAwareBaseline':
        """Deserialize from dictionary."""
        instance = cls(
            tx_mac=data.get('tx_mac', ''),
            rx_mac=data.get('rx_mac', ''),
        )

        bucket_data = data.get('bucket_baselines', [])
        for idx, bd in enumerate(bucket_data):
            if idx < len(instance.bucket_baselines):
                instance.bucket_baselines[idx] = BucketBaseline.from_dict(bd)

        return instance


@dataclass
class LinkFrequencyState:
    """Per-link state for frequency tracking."""
    last_sample_time: float = 0.0
    recent_ipis: deque = field(default_factory=lambda: deque(maxlen=IPI_HISTORY_SIZE))
    frequency_baseline: Optional[FrequencyAwareBaseline] = None

    def compute_instantaneous_frequency(self, timestamp: float) -> float:
        """
        Compute instantaneous frequency from inter-packet interval.

        Args:
            timestamp: Current sample timestamp

        Returns:
            Instantaneous frequency in Hz
        """
        if self.last_sample_time <= 0:
            return 10.0  # Default to normal bucket

        ipi = timestamp - self.last_sample_time

        if ipi <= 0:
            return 10.0

        return 1.0 / ipi

    def update_ipi(self, timestamp: float):
        """
        Update IPI tracking with new sample.

        Args:
            timestamp: Current sample timestamp
        """
        if self.last_sample_time > 0:
            ipi = timestamp - self.last_sample_time
            if ipi > 0:
                self.recent_ipis.append(ipi)

        self.last_sample_time = timestamp

    def get_mean_frequency(self) -> float:
        """
        Get mean frequency over recent samples.

        Returns:
            Mean frequency in Hz, or default if insufficient data
        """
        if len(self.recent_ipis) < 3:
            return 10.0

        mean_ipi = float(np.mean(self.recent_ipis))
        return 1.0 / mean_ipi if mean_ipi > 0 else 10.0


class PassiveFrequencyCollector:
    """
    Mode 1 - Always-running passive collection that bins by frequency.

    This collector observes all samples, computes instantaneous frequency,
    and updates frequency-aware baselines. It works with ANY scheduler -
    no cooperation required.

    Key principle: Generalizable. Passive collection happens regardless
    of what the scheduler is doing.
    """

    def __init__(
        self,
        learning_rate: float = PASSIVE_LEARNING_RATE,
        quiet_learning_rate: float = QUIET_LEARNING_RATE,
    ):
        """
        Initialize the passive frequency collector.

        Args:
            learning_rate: EMA learning rate for non-quiet samples
            quiet_learning_rate: EMA learning rate for quiet samples
        """
        self._learning_rate = learning_rate
        self._quiet_learning_rate = quiet_learning_rate

        # Per-link frequency state
        self._link_states: Dict[Tuple[str, str], LinkFrequencyState] = {}

        # Statistics
        self._total_samples = 0
        self._samples_per_bucket: List[int] = [0] * NUM_FREQUENCY_BUCKETS

    def get_or_create_state(
        self, link_key: Tuple[str, str]
    ) -> LinkFrequencyState:
        """Get or create frequency state for a link."""
        if link_key not in self._link_states:
            tx_mac, rx_mac = link_key
            state = LinkFrequencyState()
            state.frequency_baseline = FrequencyAwareBaseline(
                tx_mac=tx_mac,
                rx_mac=rx_mac
            )
            self._link_states[link_key] = state
        return self._link_states[link_key]

    def on_sample(
        self,
        link_key: Tuple[str, str],
        amplitude: float,
        phase_std: float,
        timestamp: float,
        is_quiet: bool
    ) -> Tuple[int, float]:
        """
        Process a new sample, bin by frequency, update baseline.

        Args:
            link_key: (tx_mac, rx_mac) tuple
            amplitude: Mean amplitude of the sample
            phase_std: Phase differential standard deviation
            timestamp: Sample timestamp
            is_quiet: Whether environment is quiet

        Returns:
            Tuple of (bucket_idx, instantaneous_frequency)
        """
        state = self.get_or_create_state(link_key)

        # Compute instantaneous frequency
        freq_hz = state.compute_instantaneous_frequency(timestamp)
        bucket_idx = frequency_to_bucket(freq_hz)

        # Update IPI tracking
        state.update_ipi(timestamp)

        # Update frequency-aware baseline
        if state.frequency_baseline:
            learning_rate = self._quiet_learning_rate if is_quiet else self._learning_rate
            state.frequency_baseline.update_bucket(
                bucket_idx=bucket_idx,
                amplitude=amplitude,
                phase_std=phase_std,
                is_quiet=is_quiet,
                learning_rate=learning_rate,
                timestamp=timestamp
            )

        # Update statistics
        self._total_samples += 1
        if 0 <= bucket_idx < len(self._samples_per_bucket):
            self._samples_per_bucket[bucket_idx] += 1

        return bucket_idx, freq_hz

    def get_frequency_baseline(
        self, link_key: Tuple[str, str]
    ) -> Optional[FrequencyAwareBaseline]:
        """Get frequency-aware baseline for a link."""
        if link_key in self._link_states:
            return self._link_states[link_key].frequency_baseline
        return None

    def get_link_frequency_state(
        self, link_key: Tuple[str, str]
    ) -> Optional[LinkFrequencyState]:
        """Get frequency state for a link."""
        return self._link_states.get(link_key)

    def get_link_count(self) -> int:
        """Get number of links being tracked."""
        return len(self._link_states)

    def get_stats(self) -> dict:
        """Get collector statistics."""
        return {
            'total_samples': self._total_samples,
            'samples_per_bucket': {
                bucket_to_name(i): count
                for i, count in enumerate(self._samples_per_bucket)
            },
            'links_tracked': len(self._link_states),
        }

    def get_calibration_gaps(self) -> Dict[Tuple[str, str], List[int]]:
        """
        Get calibration gaps (uncalibrated buckets) for all links.

        Returns:
            Dict mapping link_key to list of uncalibrated bucket indices
        """
        gaps = {}
        for link_key, state in self._link_states.items():
            if state.frequency_baseline:
                uncalibrated = state.frequency_baseline.get_uncalibrated_buckets()
                if uncalibrated:
                    gaps[link_key] = uncalibrated
        return gaps

    def get_overall_health(self) -> float:
        """
        Get overall frequency calibration health across all links.

        Returns:
            Health score in [0, 1]
        """
        if not self._link_states:
            return 0.0

        healths = []
        for state in self._link_states.values():
            if state.frequency_baseline:
                healths.append(state.frequency_baseline.get_calibration_health())

        return float(np.mean(healths)) if healths else 0.0

    def remove_link(self, link_key: Tuple[str, str]):
        """
        Remove a link's frequency state to free memory.

        Called during link pruning to prevent unbounded memory growth.

        Args:
            link_key: (tx_mac, rx_mac) tuple
        """
        if link_key in self._link_states:
            del self._link_states[link_key]

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'link_states': {
                f"{k[0]}->{k[1]}": {
                    'last_sample_time': v.last_sample_time,
                    'recent_ipis': list(v.recent_ipis),  # Serialize deque to list
                    'frequency_baseline': (
                        v.frequency_baseline.to_dict()
                        if v.frequency_baseline else None
                    ),
                }
                for k, v in self._link_states.items()
            },
            'total_samples': self._total_samples,
            'samples_per_bucket': self._samples_per_bucket,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PassiveFrequencyCollector':
        """Deserialize from persistence."""
        collector = cls()
        collector._total_samples = data.get('total_samples', 0)
        collector._samples_per_bucket = data.get(
            'samples_per_bucket', [0] * NUM_FREQUENCY_BUCKETS
        )

        for key_str, state_data in data.get('link_states', {}).items():
            parts = key_str.split('->')
            if len(parts) != 2:
                continue

            link_key = (parts[0], parts[1])
            state = LinkFrequencyState()
            state.last_sample_time = state_data.get('last_sample_time', 0.0)

            # Restore IPI history
            saved_ipis = state_data.get('recent_ipis', [])
            state.recent_ipis = deque(saved_ipis, maxlen=IPI_HISTORY_SIZE)

            freq_baseline_data = state_data.get('frequency_baseline')
            if freq_baseline_data:
                state.frequency_baseline = FrequencyAwareBaseline.from_dict(
                    freq_baseline_data
                )

            collector._link_states[link_key] = state

        return collector
