"""Spatial learning components for OpenCSI."""

from opencsi.learning.fresnel import (
    WIFI_WAVELENGTH_M,
    compute_angle,
    compute_distance,
    point_to_line_distance,
    compute_fresnel_radius,
    theoretical_fresnel_response,
)
from opencsi.learning.antenna_pattern import (
    AntennaPattern,
    AntennaPatternLearner,
)
from opencsi.learning.sensitivity_map import (
    SensitivityMapLearner,
)
from opencsi.learning.opportunistic import (
    CalibrationSweepRequest,
    OpportunisticCalibrator,
    MIN_QUIET_DURATION_FOR_CALIBRATION_S,
    MIN_CALIBRATION_PRIORITY,
    CALIBRATION_ATTEMPT_COOLDOWN_S,
    MAX_FREQUENCY_TARGETS,
    SWEEP_DURATION_PER_TARGET_S,
    REQUEST_EXPIRY_S,
)

__all__ = [
    # Fresnel geometry
    "WIFI_WAVELENGTH_M",
    "compute_angle",
    "compute_distance",
    "point_to_line_distance",
    "compute_fresnel_radius",
    "theoretical_fresnel_response",
    # Antenna pattern learning
    "AntennaPattern",
    "AntennaPatternLearner",
    # Sensitivity map learning
    "SensitivityMapLearner",
    # Opportunistic calibration
    "CalibrationSweepRequest",
    "OpportunisticCalibrator",
    "MIN_QUIET_DURATION_FOR_CALIBRATION_S",
    "MIN_CALIBRATION_PRIORITY",
    "CALIBRATION_ATTEMPT_COOLDOWN_S",
    "MAX_FREQUENCY_TARGETS",
    "SWEEP_DURATION_PER_TARGET_S",
    "REQUEST_EXPIRY_S",
]
