"""Core calibration engine for OpenCSI."""

from opencsi.core.calibrator import AutoCalibrator
from opencsi.core.types import (
    CalibrationSnapshot,
    CalibratedFrame,
    CalibratedLink,
    LinkSnapshot,
    PerLinkBaseline,
    QualityGrade,
    LearningMode,
    NetworkState,
    RawCSIPacket,
)
from opencsi.core.config import AutoCalibratorConfig
from opencsi.core.welford import WelfordAccumulator

# Pluggable components for modular usage
from opencsi.core.baseline import (
    BaselineManager,
    BaselineProvider,
    LinkBaseline,
    WindowedBaselineManager,
    get_baseline_manager,
)
from opencsi.core.reliability import (
    ReliabilityCalculator,
    ReliabilityScorer,
    ReliabilityBreakdown,
    get_reliability_calculator,
)
from opencsi.core.motion import (
    MotionDetector,
    MotionResult,
    ThresholdMotionDetector,
    AdaptiveMotionDetector,
    QuietPeriodDetector,
    get_motion_detector,
)

__all__ = [
    # Main calibrator
    "AutoCalibrator",
    # Data types
    "CalibrationSnapshot",
    "CalibratedFrame",
    "CalibratedLink",
    "LinkSnapshot",
    "PerLinkBaseline",
    "QualityGrade",
    "LearningMode",
    "NetworkState",
    "RawCSIPacket",
    # Configuration
    "AutoCalibratorConfig",
    # Statistics
    "WelfordAccumulator",
    # Baseline management (pluggable)
    "BaselineManager",
    "BaselineProvider",
    "LinkBaseline",
    "WindowedBaselineManager",
    "get_baseline_manager",
    # Reliability scoring (pluggable)
    "ReliabilityCalculator",
    "ReliabilityScorer",
    "ReliabilityBreakdown",

    "get_reliability_calculator",
    # Motion detection (pluggable)
    "MotionDetector",
    "MotionResult",
    "ThresholdMotionDetector",
    "AdaptiveMotionDetector",
    "QuietPeriodDetector",
    "get_motion_detector",
]
