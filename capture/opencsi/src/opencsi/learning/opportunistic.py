"""
Opportunistic Calibration for OpenCSI AutoCalibrator.

Mode 2 - During extended quiet periods (5+ min), requests calibration sweeps
to fill frequency bucket gaps.

Key Principle: GENERALIZABLE. This is an optional enhancement that works with
ANY scheduler. Calibration requests are hints, not requirements. The system
continues to function if the scheduler ignores or cannot fulfill requests.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from opencsi.baseline.frequency_baseline import PassiveFrequencyCollector
from opencsi.baseline.frequency_tracker import (
    NUM_FREQUENCY_BUCKETS,
    bucket_to_name,
    BUCKET_REFERENCE_FREQUENCIES,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Minimum quiet duration before generating calibration request
MIN_QUIET_DURATION_FOR_CALIBRATION_S = 300.0  # 5 minutes

# Minimum priority threshold to generate a request
MIN_CALIBRATION_PRIORITY = 0.3

# Priority decay - reduce priority after recent calibration attempts
CALIBRATION_ATTEMPT_COOLDOWN_S = 1800.0  # 30 minutes

# Maximum frequency targets per request
MAX_FREQUENCY_TARGETS = 3

# Default sweep duration per frequency target
SWEEP_DURATION_PER_TARGET_S = 30.0

# Request expiry time
REQUEST_EXPIRY_S = 600.0  # 10 minutes


@dataclass
class CalibrationSweepRequest:
    """
    A request for the scheduler to perform calibration sweeps.

    This is an OPTIONAL hint to the scheduler. The system continues to work
    if the scheduler ignores or cannot fulfill the request.
    """
    # Priority [0, 1] - higher = more urgent
    priority: float

    # Links that need calibration (may be empty for global sweep)
    links: List[Tuple[str, str]]

    # Target frequencies to sweep (Hz)
    frequency_targets: List[float]

    # Bucket indices with calibration gaps
    bucket_gaps: List[int]

    # Suggested sweep duration (seconds)
    duration_s: float

    # Human-readable reason for the request
    reason: str

    # Request metadata
    created_at: float = field(default_factory=time.time)
    request_id: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return time.time() - self.created_at > REQUEST_EXPIRY_S

    def to_dict(self) -> dict:
        """Serialize for API/logging."""
        return {
            'priority': round(self.priority, 3),
            'links': [f"{tx}->{rx}" for tx, rx in self.links],
            'frequency_targets': [round(f, 1) for f in self.frequency_targets],
            'bucket_gaps': [bucket_to_name(b) for b in self.bucket_gaps],
            'duration_s': round(self.duration_s, 1),
            'reason': self.reason,
            'created_at': self.created_at,
            'request_id': self.request_id,
            'is_expired': self.is_expired,
        }


class OpportunisticCalibrator:
    """
    Generates calibration sweep requests during extended quiet periods.

    Mode 2 of frequency-aware calibration. Monitors quiet state and generates
    requests to fill frequency bucket gaps when conditions are favorable.

    Key Design:
    - Requests are OPTIONAL hints to the scheduler
    - System works correctly even if requests are ignored
    - Non-blocking polling interface (no async requirements)
    - Integrates with CalibrationPolicy's trigger system
    """

    def __init__(
        self,
        passive_collector: PassiveFrequencyCollector,
        min_quiet_duration_s: float = MIN_QUIET_DURATION_FOR_CALIBRATION_S,
    ):
        """
        Initialize the opportunistic calibrator.

        Args:
            passive_collector: Reference to the passive frequency collector
            min_quiet_duration_s: Minimum quiet duration before generating request
        """
        self._passive_collector = passive_collector
        self._min_quiet_duration_s = min_quiet_duration_s

        # Quiet tracking
        self._quiet_since: Optional[float] = None
        self._last_quiet_check_time: float = 0.0

        # Request state
        self._pending_request: Optional[CalibrationSweepRequest] = None
        self._request_counter = 0

        # Calibration history
        self._last_calibration_attempt: float = 0.0
        self._calibration_attempts: List[float] = []

    def on_frame(self, is_quiet: bool, timestamp: float):
        """
        Process a frame update.

        Call this for every frame to track quiet duration and
        potentially generate calibration requests.

        Args:
            is_quiet: Whether the environment is currently quiet
            timestamp: Current timestamp
        """
        self._last_quiet_check_time = timestamp

        if is_quiet:
            if self._quiet_since is None:
                self._quiet_since = timestamp
                logger.debug("OpportunisticCalibrator: Quiet period started")

            # Check if we've been quiet long enough
            quiet_duration = timestamp - self._quiet_since
            if quiet_duration >= self._min_quiet_duration_s:
                self._check_and_generate_request(timestamp)
        else:
            if self._quiet_since is not None:
                quiet_duration = timestamp - self._quiet_since
                logger.debug(
                    f"OpportunisticCalibrator: Quiet period ended "
                    f"after {quiet_duration:.1f}s"
                )
            self._quiet_since = None

            # Clear pending request if room becomes occupied
            if self._pending_request:
                logger.debug(
                    "OpportunisticCalibrator: Clearing pending request "
                    "(room no longer quiet)"
                )
                self._pending_request = None

    def _check_and_generate_request(self, timestamp: float):
        """Check calibration gaps and generate request if needed."""
        # Don't generate new request if one is pending
        if self._pending_request and not self._pending_request.is_expired:
            return

        # Check cooldown from last attempt
        if timestamp - self._last_calibration_attempt < CALIBRATION_ATTEMPT_COOLDOWN_S:
            return

        # Analyze calibration gaps
        gaps = self._passive_collector.get_calibration_gaps()
        if not gaps:
            return  # All links are well-calibrated

        # Compute priority based on gap severity
        priority = self._compute_priority(gaps)
        if priority < MIN_CALIBRATION_PRIORITY:
            return

        # Generate request
        request = self._generate_request(gaps, priority, timestamp)
        if request:
            self._pending_request = request
            logger.info(
                f"OpportunisticCalibrator: Generated calibration request "
                f"(priority={priority:.2f}, buckets={request.bucket_gaps})"
            )

    def _compute_priority(
        self, gaps: Dict[Tuple[str, str], List[int]]
    ) -> float:
        """
        Compute calibration priority based on gap severity.

        Higher priority for:
        - More links with gaps
        - Gaps in commonly-used buckets (normal, elevated)
        - Older gaps

        Args:
            gaps: Dict mapping link_key to list of uncalibrated bucket indices

        Returns:
            Priority score [0, 1]
        """
        if not gaps:
            return 0.0

        # Count unique uncalibrated buckets across all links
        all_buckets = set()
        for bucket_list in gaps.values():
            all_buckets.update(bucket_list)

        # Weight by bucket importance (normal/elevated most important)
        bucket_weights = {
            0: 0.05,  # idle
            1: 0.1,   # background
            2: 0.35,  # normal
            3: 0.3,   # elevated
            4: 0.15,  # focus
            5: 0.05,  # burst
        }

        weighted_gaps = sum(
            bucket_weights.get(b, 0.1) for b in all_buckets
        )

        # Scale by fraction of links affected
        link_fraction = len(gaps) / max(
            1, self._passive_collector.get_link_count()
        )

        # Combine factors
        priority = min(1.0, weighted_gaps * (0.5 + 0.5 * link_fraction))

        return priority

    def _generate_request(
        self,
        gaps: Dict[Tuple[str, str], List[int]],
        priority: float,
        timestamp: float
    ) -> Optional[CalibrationSweepRequest]:
        """
        Generate a calibration sweep request.

        Args:
            gaps: Dict mapping link_key to uncalibrated bucket indices
            priority: Computed priority
            timestamp: Current timestamp

        Returns:
            CalibrationSweepRequest or None
        """
        # Find most common uncalibrated buckets
        bucket_counts: Dict[int, int] = {}
        for bucket_list in gaps.values():
            for bucket in bucket_list:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        if not bucket_counts:
            return None

        # Sort by count (most common first)
        sorted_buckets = sorted(
            bucket_counts.keys(),
            key=lambda b: bucket_counts[b],
            reverse=True
        )

        # Take top N buckets
        target_buckets = sorted_buckets[:MAX_FREQUENCY_TARGETS]

        # Convert to target frequencies
        frequency_targets = [
            BUCKET_REFERENCE_FREQUENCIES[b]
            for b in target_buckets
            if b < len(BUCKET_REFERENCE_FREQUENCIES)
        ]

        # Compute duration
        duration_s = len(target_buckets) * SWEEP_DURATION_PER_TARGET_S

        # Get links with gaps in target buckets
        links_needing_calibration = [
            link_key
            for link_key, bucket_list in gaps.items()
            if any(b in target_buckets for b in bucket_list)
        ]

        # Generate request ID
        self._request_counter += 1
        request_id = f"freq_cal_{self._request_counter}_{int(timestamp)}"

        # Build reason string
        bucket_names = [bucket_to_name(b) for b in target_buckets]
        reason = f"Fill frequency calibration gaps: {', '.join(bucket_names)}"

        return CalibrationSweepRequest(
            priority=priority,
            links=links_needing_calibration,
            frequency_targets=frequency_targets,
            bucket_gaps=target_buckets,
            duration_s=duration_s,
            reason=reason,
            created_at=timestamp,
            request_id=request_id,
        )

    def get_calibration_request(self) -> Optional[CalibrationSweepRequest]:
        """
        Get pending calibration request (non-blocking poll).

        This is called by CalibrationPolicy to check for frequency
        calibration triggers.

        Returns:
            CalibrationSweepRequest if one is pending, None otherwise
        """
        if self._pending_request:
            if self._pending_request.is_expired:
                logger.debug(
                    "OpportunisticCalibrator: Request expired, clearing"
                )
                self._pending_request = None
                return None
            return self._pending_request
        return None

    def mark_request_attempted(self, request_id: str = "", timestamp: Optional[float] = None):
        """
        Mark that a calibration request was attempted.

        Call this when the scheduler acknowledges a request,
        regardless of whether it succeeds.

        Args:
            request_id: Optional request ID for tracking
            timestamp: Optional timestamp (defaults to time.time())
        """
        ts = timestamp if timestamp is not None else time.time()
        self._last_calibration_attempt = ts
        self._calibration_attempts.append(ts)

        # Clean old attempts
        cutoff = ts - 86400  # Keep last 24 hours
        self._calibration_attempts = [
            t for t in self._calibration_attempts if t > cutoff
        ]

        # Clear pending request
        if self._pending_request:
            logger.info(
                f"OpportunisticCalibrator: Request {request_id or 'unknown'} "
                f"marked as attempted"
            )
            self._pending_request = None

    def clear_request(self):
        """Clear any pending request."""
        self._pending_request = None

    @property
    def quiet_duration_s(self) -> float:
        """Get current quiet duration in seconds."""
        if self._quiet_since is None:
            return 0.0
        return self._last_quiet_check_time - self._quiet_since

    @property
    def has_pending_request(self) -> bool:
        """Check if there's a pending calibration request."""
        return (
            self._pending_request is not None
            and not self._pending_request.is_expired
        )

    def get_status(self) -> dict:
        """Get calibrator status for API/debugging."""
        return {
            'quiet_since': self._quiet_since,
            'quiet_duration_s': round(self.quiet_duration_s, 1),
            'min_quiet_duration_s': self._min_quiet_duration_s,
            'has_pending_request': self.has_pending_request,
            'pending_request': (
                self._pending_request.to_dict()
                if self._pending_request
                else None
            ),
            'last_calibration_attempt': self._last_calibration_attempt,
            'attempts_last_24h': len(self._calibration_attempts),
        }
