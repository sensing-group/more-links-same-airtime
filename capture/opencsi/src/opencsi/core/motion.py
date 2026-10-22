"""
Motion Detection Module for OpenCSI.

Provides pluggable motion detection algorithms based on CSI Z-scores.
Users can implement custom MotionDetector subclasses for specialized use cases.

Usage:
    from opencsi.core.motion import MotionDetector, ThresholdMotionDetector

    # Use default threshold-based detector
    detector = ThresholdMotionDetector(threshold=2.0)

    # Check for motion
    if detector.detect(z_score=3.5, reliability=0.8):
        print("Motion detected!")

    # Or implement your own
    class MyDetector(MotionDetector):
        def detect(self, z_score, reliability=1.0, **kwargs):
            return z_score > self.custom_threshold
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MOTION_THRESHOLD = 2.0  # Z-score threshold for motion
DEFAULT_HIGH_MOTION_THRESHOLD = 5.0  # Z-score for definite motion
CONSECUTIVE_SAMPLES_FOR_MOTION = 2  # Consecutive high samples needed
QUIET_VARIANCE_THRESHOLD = 0.5  # Variance threshold for quiet detection
QUIET_ENTROPY_THRESHOLD = 0.8  # Entropy threshold for quiet detection
MIN_RELIABILITY_FOR_DETECTION = 0.3  # Minimum reliability to trust detection

# Confidence computation constants
CONFIDENCE_Z_DIVISOR_NO_DETECTION = 5.0  # Divisor for Z-score when no detection
CONFIDENCE_MAX_NO_DETECTION = 0.9  # Max confidence cap when no detection
CONFIDENCE_Z_DIVISOR_DETECTION = 10.0  # Divisor for Z-score when detected

# Adaptive detector constants
DEFAULT_TARGET_QUIET_RATIO = 0.9  # Target ratio of quiet samples
DEFAULT_ADAPTATION_RATE = 0.01  # How fast threshold adapts
DEFAULT_MIN_ADAPTIVE_THRESHOLD = 1.5  # Minimum adaptive threshold
DEFAULT_MAX_ADAPTIVE_THRESHOLD = 5.0  # Maximum adaptive threshold
QUIET_RATIO_MARGIN = 0.05  # Margin for quiet ratio adjustment


# =============================================================================
# Motion Detection Result
# =============================================================================

@dataclass
class MotionResult:
    """Result of motion detection for a single link or frame."""
    detected: bool
    confidence: float  # 0.0 to 1.0
    z_score: float
    reliability: float
    is_quiet: bool = False
    consecutive_count: int = 0


# =============================================================================
# Abstract Base Class
# =============================================================================

class MotionDetector(ABC):
    """
    Abstract base class for motion detection algorithms.

    Subclass this to implement custom motion detection logic.
    The calibrator will call detect() for each link on each frame.

    Example:
        class MLMotionDetector(MotionDetector):
            def __init__(self, model_path: str):
                self.model = load_model(model_path)

            def detect(self, z_score, reliability=1.0, **kwargs):
                features = kwargs.get('features', [z_score])
                return self.model.predict([features])[0] > 0.5
    """

    @abstractmethod
    def detect(
        self,
        z_score: float,
        reliability: float = 1.0,
        **kwargs
    ) -> bool:
        """
        Detect motion based on Z-score and optional context.

        Args:
            z_score: Combined amplitude/phase Z-score
            reliability: Link reliability score (0 to 1)
            **kwargs: Additional context (variance, entropy, history, etc.)

        Returns:
            True if motion is detected
        """
        ...

    def detect_with_result(
        self,
        z_score: float,
        reliability: float = 1.0,
        **kwargs
    ) -> MotionResult:
        """
        Detect motion and return detailed result.

        Args:
            z_score: Combined Z-score
            reliability: Link reliability
            **kwargs: Additional context

        Returns:
            MotionResult with detection details
        """
        detected = self.detect(z_score, reliability, **kwargs)
        confidence = self._compute_confidence(z_score, reliability, detected)

        return MotionResult(
            detected=detected,
            confidence=confidence,
            z_score=z_score,
            reliability=reliability,
            is_quiet=kwargs.get('is_quiet', False),
            consecutive_count=kwargs.get('consecutive_count', 0),
        )

    def _compute_confidence(
        self,
        z_score: float,
        reliability: float,
        detected: bool
    ) -> float:
        """Compute confidence in the detection result."""
        if not detected:
            return 1.0 - min(
                abs(z_score) / CONFIDENCE_Z_DIVISOR_NO_DETECTION,
                CONFIDENCE_MAX_NO_DETECTION
            )

        # Higher Z-score and reliability = higher confidence
        z_confidence = min(abs(z_score) / CONFIDENCE_Z_DIVISOR_DETECTION, 1.0)
        return z_confidence * reliability


# =============================================================================
# Threshold-Based Detector (Default)
# =============================================================================

class ThresholdMotionDetector(MotionDetector):
    """
    Simple threshold-based motion detector.

    Detects motion when Z-score exceeds threshold and reliability is sufficient.
    This is the default detector used by AutoCalibrator.

    Args:
        threshold: Z-score threshold for motion (default: 2.0)
        high_threshold: Z-score for definite motion (default: 5.0)
        min_reliability: Minimum reliability to trust detection (default: 0.3)
        require_consecutive: Number of consecutive high samples needed (default: 2)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_MOTION_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_MOTION_THRESHOLD,
        min_reliability: float = MIN_RELIABILITY_FOR_DETECTION,
        require_consecutive: int = CONSECUTIVE_SAMPLES_FOR_MOTION,
    ):
        self.threshold = threshold
        self.high_threshold = high_threshold
        self.min_reliability = min_reliability
        self.require_consecutive = require_consecutive

        # Track consecutive high samples per link
        self._consecutive_counts: Dict[Tuple[str, str], int] = {}

    def detect(
        self,
        z_score: float,
        reliability: float = 1.0,
        **kwargs
    ) -> bool:
        """
        Detect motion using threshold logic.

        Returns True if:
        - Z-score exceeds high_threshold (definite motion), OR
        - Z-score exceeds threshold AND reliability >= min_reliability
          AND consecutive high samples >= require_consecutive
        """
        link_key = kwargs.get('link_key')

        # Definite motion - high Z-score (keep incrementing consecutive count)
        if abs(z_score) >= self.high_threshold:
            if link_key:
                self._consecutive_counts[link_key] = (
                    self._consecutive_counts.get(link_key, 0) + 1
                )
            return True

        # Check threshold with reliability
        if abs(z_score) >= self.threshold and reliability >= self.min_reliability:
            if link_key:
                count = self._consecutive_counts.get(link_key, 0) + 1
                self._consecutive_counts[link_key] = count
                return count >= self.require_consecutive
            return True

        # Below threshold - reset consecutive count
        if link_key:
            self._consecutive_counts[link_key] = 0
        return False

    def reset_link(self, link_key: Tuple[str, str]) -> None:
        """Reset consecutive count for a link."""
        self._consecutive_counts.pop(link_key, None)

    def get_consecutive_count(self, link_key: Tuple[str, str]) -> int:
        """Get consecutive high sample count for a link."""
        return self._consecutive_counts.get(link_key, 0)


# =============================================================================
# Quiet Period Detector
# =============================================================================

class QuietPeriodDetector:
    """
    Detects quiet periods (empty room) based on variance and entropy.

    Quiet periods are used for baseline learning. A period is considered
    quiet when both variance and entropy are below their thresholds.

    Args:
        variance_threshold: Maximum variance for quiet (default: 0.5)
        entropy_threshold: Maximum entropy for quiet (default: 0.8)
        min_consecutive: Minimum consecutive quiet samples (default: 5)
    """

    def __init__(
        self,
        variance_threshold: float = QUIET_VARIANCE_THRESHOLD,
        entropy_threshold: float = QUIET_ENTROPY_THRESHOLD,
        min_consecutive: int = 5,
    ):
        self.variance_threshold = variance_threshold
        self.entropy_threshold = entropy_threshold
        self.min_consecutive = min_consecutive

        self._consecutive_quiet: Dict[Tuple[str, str], int] = {}

    def is_quiet(
        self,
        variance: float,
        entropy: float,
        link_key: Optional[Tuple[str, str]] = None,
    ) -> bool:
        """
        Check if current sample indicates a quiet period.

        Args:
            variance: Current signal variance
            entropy: Current spectral entropy
            link_key: Optional link identifier for tracking

        Returns:
            True if quiet period detected
        """
        is_sample_quiet = (
            variance < self.variance_threshold and
            entropy < self.entropy_threshold
        )

        if link_key is None:
            return is_sample_quiet

        # Track consecutive quiet samples
        if is_sample_quiet:
            count = self._consecutive_quiet.get(link_key, 0) + 1
            self._consecutive_quiet[link_key] = count
            return count >= self.min_consecutive
        else:
            self._consecutive_quiet[link_key] = 0
            return False

    def reset_link(self, link_key: Tuple[str, str]) -> None:
        """Reset quiet tracking for a link."""
        self._consecutive_quiet.pop(link_key, None)

    def get_consecutive_quiet(self, link_key: Tuple[str, str]) -> int:
        """Get consecutive quiet sample count for a link."""
        return self._consecutive_quiet.get(link_key, 0)


# =============================================================================
# Adaptive Motion Detector
# =============================================================================

class AdaptiveMotionDetector(MotionDetector):
    """
    Adaptive motion detector that adjusts threshold based on environment.

    Tracks recent Z-score distribution and adapts threshold to maintain
    a target false positive rate. Useful for environments with varying
    noise levels.

    Args:
        initial_threshold: Starting threshold (default: 2.0)
        target_quiet_ratio: Target ratio of quiet samples (default: 0.9)
        adaptation_rate: How fast to adapt (default: 0.01)
        min_threshold: Minimum allowed threshold (default: 1.5)
        max_threshold: Maximum allowed threshold (default: 5.0)
    """

    def __init__(
        self,
        initial_threshold: float = DEFAULT_MOTION_THRESHOLD,
        target_quiet_ratio: float = DEFAULT_TARGET_QUIET_RATIO,
        adaptation_rate: float = DEFAULT_ADAPTATION_RATE,
        min_threshold: float = DEFAULT_MIN_ADAPTIVE_THRESHOLD,
        max_threshold: float = DEFAULT_MAX_ADAPTIVE_THRESHOLD,
    ):
        self.threshold = initial_threshold
        self.target_quiet_ratio = target_quiet_ratio
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._recent_detections: deque = deque(maxlen=100)

    def detect(
        self,
        z_score: float,
        reliability: float = 1.0,
        **kwargs
    ) -> bool:
        """Detect motion with adaptive threshold."""
        detected = abs(z_score) >= self.threshold and reliability >= MIN_RELIABILITY_FOR_DETECTION

        # Track detection for adaptation
        self._recent_detections.append(1 if detected else 0)

        # Adapt threshold periodically
        if len(self._recent_detections) >= 50:
            self._adapt_threshold()

        return detected

    def _adapt_threshold(self) -> None:
        """Adapt threshold based on recent detection rate."""
        if len(self._recent_detections) < 10:
            return

        detection_rate = sum(self._recent_detections) / len(self._recent_detections)
        quiet_ratio = 1.0 - detection_rate

        # Adjust threshold to approach target quiet ratio
        if quiet_ratio < self.target_quiet_ratio:
            # Too many detections - increase threshold
            self.threshold = min(
                self.threshold * (1 + self.adaptation_rate),
                self.max_threshold
            )
        elif quiet_ratio > self.target_quiet_ratio + QUIET_RATIO_MARGIN:
            # Too few detections - decrease threshold
            self.threshold = max(
                self.threshold * (1 - self.adaptation_rate),
                self.min_threshold
            )

    @property
    def current_threshold(self) -> float:
        """Get current adapted threshold."""
        return self.threshold


# =============================================================================
# Factory Function
# =============================================================================

def get_motion_detector(
    detector_type: str = "threshold",
    **kwargs
) -> MotionDetector:
    """
    Factory function for creating motion detectors.

    Args:
        detector_type: "threshold" or "adaptive"
        **kwargs: Arguments passed to detector constructor

    Returns:
        MotionDetector instance

    Example:
        detector = get_motion_detector("adaptive", target_quiet_ratio=0.85)
    """
    detectors = {
        "threshold": ThresholdMotionDetector,
        "adaptive": AdaptiveMotionDetector,
    }

    if detector_type not in detectors:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Available: {list(detectors.keys())}"
        )

    return detectors[detector_type](**kwargs)
