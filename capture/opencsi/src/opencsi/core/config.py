"""
Configuration and constants for OpenCSI AutoCalibrator.

This module contains the configuration dataclass and module-level constants
used throughout the calibration system.
"""

import dataclasses
from dataclasses import dataclass


# =============================================================================
# Configuration Constants
# =============================================================================

RELIABLE_LINK_THRESHOLD = 0.1
"""Minimum reliability for a link to be counted as 'reliable'.

This matches MIN_RELIABILITY (0.1) to ensure that links at the reliability
floor are still counted as usable. The floor represents "calibrated but stale"
links that remain valid for detection even when baselines are old.

Note: ESP32-C3 hardware typically achieves reliability in the 0.15-0.28 range
when baselines are fresh. After extended periods without quiet detection,
reliability decays to the floor but baselines remain valid for anomaly detection."""

# =============================================================================
# Wavelength and Frequency Constants
# =============================================================================
# WiFi operates in multiple frequency bands with different wavelengths.
# Wavelength affects Fresnel zone size: larger wavelength = larger zone.

# Speed of light in meters per second (for frequency-to-wavelength conversion)
SPEED_OF_LIGHT_M_S = 299792458.0

# Common wavelengths (pre-computed for efficiency)
WAVELENGTH_2_4GHZ_M = 0.125   # WiFi 2.4 GHz (~2.4 GHz center)
WAVELENGTH_5GHZ_M = 0.060     # WiFi 5 GHz (~5 GHz center)
WAVELENGTH_6GHZ_M = 0.050     # WiFi 6E/7 6 GHz (~6 GHz center)

# Channel number boundaries for band detection
CHANNEL_2_4GHZ_MAX = 14       # Channels 1-14 are 2.4 GHz
CHANNEL_5GHZ_MIN = 36         # Channels 36+ are 5 GHz
CHANNEL_6GHZ_MIN = 1          # 6 GHz uses channels 1-233 (different numbering)

# Base frequencies for channel-to-frequency conversion (MHz)
FREQ_BASE_2_4GHZ_MHZ = 2407   # Channel 1 = 2412 MHz
FREQ_STEP_2_4GHZ_MHZ = 5      # 5 MHz per channel
FREQ_CHANNEL_14_MHZ = 2484.0  # Channel 14 exception (Japan only)
FREQ_BASE_5GHZ_MHZ = 5000     # Approximate base for 5 GHz
FREQ_STEP_5GHZ_MHZ = 5        # 5 MHz per channel
FREQ_BASE_6GHZ_MHZ = 5950     # 6 GHz (WiFi 6E/7) base frequency
FREQ_STEP_6GHZ_MHZ = 5        # 5 MHz per channel

# Unit conversion
MHZ_TO_HZ = 1e6               # MHz to Hz multiplier

# =============================================================================
# Effective Detection Width Constants (SNR-based Fresnel zone scaling)
# =============================================================================
# These constants define how SNR affects the effective detection area.
# Higher SNR = larger effective detection width (closer to theoretical Fresnel zone).
# Lower SNR = reduced detection area due to noise masking the signal.

# SNR thresholds for detection width scaling (in dB)
SNR_EXCELLENT_DB = 25.0      # Above this: full theoretical Fresnel zone
SNR_GOOD_DB = 15.0           # Above this: 70%+ of theoretical zone
SNR_FAIR_DB = 5.0            # Above this: 40%+ of theoretical zone
SNR_POOR_DB = 0.0            # Below SNR_FAIR: minimum 20% of theoretical zone

# SNR scaling factors (multiply by Fresnel radius)
SNR_FACTOR_EXCELLENT = 1.0   # Full zone at excellent SNR
SNR_FACTOR_GOOD = 0.7        # 70% at good SNR
SNR_FACTOR_FAIR = 0.4        # 40% at fair SNR
SNR_FACTOR_MINIMUM = 0.2     # Never go below 20%

# Weak link threshold (meters) - below this, link has limited detection utility
WEAK_LINK_DETECTION_WIDTH_M = 0.3


# =============================================================================
# Frequency/Wavelength Conversion Functions
# =============================================================================

def channel_to_frequency_mhz(channel: int, band: str = "2.4GHz") -> float:
    """
    Convert WiFi channel number to center frequency in MHz.

    Args:
        channel: WiFi channel number
        band: Frequency band ("2.4GHz", "5GHz", or "6GHz")

    Returns:
        Center frequency in MHz

    Examples:
        >>> channel_to_frequency_mhz(1)        # 2.4 GHz channel 1
        2412.0
        >>> channel_to_frequency_mhz(6)        # 2.4 GHz channel 6
        2437.0
        >>> channel_to_frequency_mhz(36, "5GHz")
        5180.0
        >>> channel_to_frequency_mhz(149, "5GHz")
        5745.0
    """
    if band == "2.4GHz" or channel <= CHANNEL_2_4GHZ_MAX:
        # 2.4 GHz: Channel 1 = 2412 MHz, 5 MHz spacing
        # Exception: Channel 14 = 2484 MHz (Japan only)
        if channel == 14:
            return FREQ_CHANNEL_14_MHZ
        return float(FREQ_BASE_2_4GHZ_MHZ + channel * FREQ_STEP_2_4GHZ_MHZ)

    elif band == "5GHz" or channel >= CHANNEL_5GHZ_MIN:
        # 5 GHz: More complex channel plan
        # UNII-1: 36, 40, 44, 48 -> 5180, 5200, 5220, 5240 MHz
        # UNII-2A: 52, 56, 60, 64 -> 5260, 5280, 5300, 5320 MHz
        # UNII-2C: 100-144 -> 5500-5720 MHz
        # UNII-3: 149-165 -> 5745-5825 MHz
        return float(FREQ_BASE_5GHZ_MHZ + channel * FREQ_STEP_5GHZ_MHZ)

    else:
        # 6 GHz (WiFi 6E/7): Channels 1-233
        # Channel 1 = 5955 MHz, 5 MHz spacing
        return float(FREQ_BASE_6GHZ_MHZ + channel * FREQ_STEP_6GHZ_MHZ)


def channel_to_wavelength(channel: int, band: str = "2.4GHz") -> float:
    """
    Convert WiFi channel number to wavelength in meters.

    This is the primary function for computing Fresnel zone parameters
    from packet metadata.

    Args:
        channel: WiFi channel number
        band: Frequency band ("2.4GHz", "5GHz", or "6GHz")

    Returns:
        Wavelength in meters

    Examples:
        >>> channel_to_wavelength(1)           # 2.4 GHz
        0.1242...
        >>> channel_to_wavelength(36, "5GHz")  # 5 GHz
        0.0579...
        >>> channel_to_wavelength(1, "6GHz")   # 6 GHz
        0.0503...
    """
    freq_mhz = channel_to_frequency_mhz(channel, band)
    return frequency_to_wavelength(freq_mhz)


def frequency_to_wavelength(frequency_mhz: float) -> float:
    """
    Convert frequency in MHz to wavelength in meters.

    Args:
        frequency_mhz: Frequency in MHz

    Returns:
        Wavelength in meters
    """
    freq_hz = frequency_mhz * MHZ_TO_HZ
    return SPEED_OF_LIGHT_M_S / freq_hz


def get_band_wavelength(band: str) -> float:
    """
    Get approximate wavelength for a frequency band.

    Uses pre-computed center-frequency wavelengths for efficiency.

    Args:
        band: Frequency band ("2.4GHz", "5GHz", or "6GHz")

    Returns:
        Approximate wavelength in meters
    """
    if band == "5GHz":
        return WAVELENGTH_5GHZ_M
    elif band == "6GHz":
        return WAVELENGTH_6GHZ_M
    else:
        return WAVELENGTH_2_4GHZ_M


@dataclass
class AutoCalibratorConfig:
    """
    Configuration for AutoCalibrator (OpenCSI calibration layer).

    The complete reliability formula is multiplicative:
        R = Q_cv × Q_age × Q_rate × Q_rssi × Q_sample

    Where:
    - Q_cv   = 1/(1+CV), CV = σ/μ capped at 3.0 — baseline stability
    - Q_age  = e^(-λ×t_age), λ = reliability_decay_lambda — baseline freshness
    - Q_rate = frequency bucket calibration confidence [0, 1]
    - Q_rssi = sigmoid(RSSI) — signal quality
    - Q_sample = quiet_samples / min_baseline_samples — ramp-up confidence

    See opencsi.core.reliability.ReliabilityCalculator for implementation.

    The calibrator wraps the sanitization layer internally.
    Sanitization settings can be configured here.
    """

    # === Sanitization (wrapped internally) ===
    enable_agc_correction: bool = True
    """Enable AGC (automatic gain control) correction."""

    enable_phase_correction: bool = True
    """Enable phase error removal (SFO/PDD/CFO)."""

    enable_outlier_removal: bool = True
    """Enable Hampel outlier removal."""

    enable_hamming_window: bool = True
    """Enable Hamming windowing for spectral leakage reduction."""

    enable_nbvi_selection: bool = False
    """Enable NBVI-based automatic subcarrier selection."""

    nbvi_n_subcarriers: int = 12
    """Number of subcarriers to select (ESPectre uses 12)."""

    nbvi_min_quiet_samples: int = 50
    """Minimum quiet samples before NBVI selection is computed."""

    nbvi_recompute_interval_s: float = 300.0
    """Interval between NBVI recomputations (seconds)."""

    # === AGC Warmup ===
    agc_warmup_samples: int = 20
    """Minimum samples before AGC warmup completes. AGC needs time to stabilize after boot."""

    agc_warmup_timeout_s: float = 5.0
    """Maximum warmup duration (seconds). Warmup completes when EITHER sample count OR timeout is reached."""

    # === Baseline Learning ===
    quiet_detection_window_s: float = 30.0
    """Window for detecting quiet periods (seconds)."""

    quiet_variance_threshold: float = 5.0
    """Variance threshold for 'quiet' classification.
    Uses temporal variance (stability of mean amplitude over time)."""

    baseline_learning_rate: float = 0.1
    """How fast to update baselines (0-1). Higher = faster adaptation."""

    min_baseline_samples: int = 10
    """Minimum quiet samples before baseline is considered valid."""

    # === Sensitivity Map Learning ===
    sensitivity_grid_resolution_m: float = 0.25
    """Spatial resolution for sensitivity maps (meters)."""

    sensitivity_learning_rate: float = 0.05
    """How fast to update sensitivity maps (0-1)."""

    min_observations_for_confidence: int = 10
    """Minimum observations before trusting learned sensitivity."""

    # === Antenna Pattern Learning ===
    antenna_learning_rate: float = 0.02
    """How fast to update antenna patterns (slow, needs many observations)."""

    antenna_angular_resolution_deg: float = 10.0
    """Angular resolution for antenna pattern (degrees)."""

    # === Reliability Scoring ===
    reliability_decay_lambda: float = 1.0 / 86400.0
    """Decay constant λ for baseline age factor. Default: 1/86400 (24-hour half-life).

    This controls how quickly calibration reliability decays over time. A 24-hour
    half-life means reliability drops to 50% after 24 hours of no baseline updates.
    The previous 1-hour half-life was too aggressive for long-term operation.

    Override via RELIABILITY_DECAY_HALF_LIFE_S environment variable."""

    reliability_history_window_s: float = 300.0
    """Window for computing link reliability scores (seconds)."""

    # === Learning Mode Transitions ===
    mode_transition_min_baseline_coverage: float = 0.5
    """Fraction of links that must have learned baselines to exit BLIND mode."""

    mode_transition_min_sensitivity_coverage: float = 0.1
    """Fraction of grid cells with observations to consider ONLINE mode active."""

    supervised_calibration_timeout_s: float = 900.0
    """Timeout for supervised calibration mode (15 minutes default)."""

    # === Quiet Detection Thresholds ===
    variance_low_threshold: float = 5.0
    """Low temporal variance threshold - below this is definitely quiet."""

    variance_high_threshold: float = 20.0
    """High variance threshold for entropy-based quiet detection."""

    entropy_low_threshold: float = 2.0
    """Spectral entropy threshold (bits). Low entropy = periodic source."""

    temporal_amplitude_history_size: int = 50
    """Number of frames to keep in temporal amplitude history for quiet detection."""

    min_temporal_samples: int = 10
    """Minimum samples needed before computing temporal variance."""

    # === Drift Detection ===
    drift_threshold_sigma: float = 2.0
    """Baseline drift threshold in σ units for flagging stale calibration."""

    drift_window_s: float = 30.0
    """Time window for drift detection (seconds)."""

    drift_link_fraction: float = 0.3
    """Fraction of links that must show drift to trigger global drift flag."""

    # === Output Normalization (SNR-based weighting) ===
    snr_weight_transition_db: float = 15.0
    """SNR at which phase/amplitude weighting is 50/50."""

    # === Per-Link Quiet Gating ===
    per_link_activity_zscore_threshold: float = 1.5
    """Z-score threshold for per-link baseline gating."""

    # === Persistence ===
    auto_save_interval_s: float = 300.0
    """Auto-save calibration state every N seconds."""

    state_file_path: str = "calibration_state.json"
    """Default path for state persistence."""

    def to_dict(self) -> dict:
        """Serialize all config fields for persistence (JSON-safe)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'AutoCalibratorConfig':
        """Reconstruct config from a dict, ignoring unknown keys.

        This allows forward compatibility: if a newer version added fields
        that don't exist in this version, they are silently dropped.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})
