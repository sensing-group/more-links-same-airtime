"""
Shared base class for bucket-level baseline statistics.

Both frequency-bucketed and time-bucketed baselines share the same
core statistics (amplitude mean/std, phase std, sample counts, timing).
This module extracts those into a reusable dataclass to eliminate
duplication between BucketBaseline and TimeBucketBaseline.
"""

import time as _time
from dataclasses import dataclass


@dataclass
class BaseBucketStats:
    """
    Core statistics shared by all bucket-level baselines.

    This base class captures the common fields that appear in both
    frequency-bucketed (BucketBaseline) and time-bucketed (TimeBucketBaseline)
    baselines. Subclasses add dimension-specific behavior (aggregation,
    retention policies, etc.).

    Attributes:
        amplitude_mean: Running mean of amplitude (scalar, EMA-updated).
        amplitude_std: Running std of amplitude.
        phase_std: Running std of phase differences (circular).
        quiet_samples: Count of samples during quiet (no-motion) periods.
        total_samples: Count of all samples (quiet + non-quiet).
        last_update_time: Unix timestamp of last baseline update.
        creation_time: Unix timestamp when this bucket was first initialized.
        is_initialized: True once at least one quiet sample has been ingested.
    """
    amplitude_mean: float = 0.0
    amplitude_std: float = 1.0
    phase_std: float = 1.0
    quiet_samples: int = 0
    total_samples: int = 0
    last_update_time: float = 0.0
    creation_time: float = 0.0
    is_initialized: bool = False

    @property
    def age_seconds(self) -> float:
        """Seconds since last baseline update."""
        if self.last_update_time == 0:
            return float('inf')
        return _time.time() - self.last_update_time

    def _base_to_dict(self) -> dict:
        """Serialize shared fields to dictionary."""
        return {
            'amplitude_mean': self.amplitude_mean,
            'amplitude_std': self.amplitude_std,
            'phase_std': self.phase_std,
            'quiet_samples': self.quiet_samples,
            'total_samples': self.total_samples,
            'last_update_time': self.last_update_time,
            'creation_time': self.creation_time,
            'is_initialized': self.is_initialized,
        }

    def _base_from_dict(self, data: dict) -> None:
        """Restore shared fields from dictionary."""
        self.amplitude_mean = data.get('amplitude_mean', 0.0)
        self.amplitude_std = data.get('amplitude_std', 1.0)
        self.phase_std = data.get('phase_std', 1.0)
        self.quiet_samples = data.get('quiet_samples', 0)
        self.total_samples = data.get('total_samples', 0)
        self.last_update_time = data.get('last_update_time', 0.0)
        self.creation_time = data.get('creation_time', 0.0)
        self.is_initialized = data.get('is_initialized', False)
