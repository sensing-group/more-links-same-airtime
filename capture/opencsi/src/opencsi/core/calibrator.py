#!/usr/bin/env python3
"""
AutoCalibrator: Self-Calibrating CSI Normalization Layer.

This module transforms raw, hardware-dependent CSI data into calibrated,
hardware-agnostic Z-scores with reliability metadata.

**Architecture:**
AutoCalibrator WRAPS the sanitization layer internally. It accepts raw CSI
packets and handles all preprocessing before calibration. This ensures
consistent sanitization across all users and simplifies the API.

Key Features:
1. WRAPS sanitization layer (AGC, phase correction, outliers)
2. Learns per-link baselines during quiet periods (auto-detected)
3. Normalizes measurements to Z-scores (standard deviations from baseline)
4. Provides reliability scores for each link (multiplicative formula)
5. Detects environment drift (furniture moved, etc.)
6. Supports hardware profiles for cold-start initialization
7. Learns spatial sensitivity maps through detection feedback
8. Learns effective antenna patterns through inverse Fresnel imaging

Usage:
    from opencsi import AutoCalibrator

    # Initialize with node positions (required)
    node_positions = {
        "AA:BB:CC:DD:EE:01": (0.0, 0.0),
        "AA:BB:CC:DD:EE:02": (5.0, 0.0),
    }
    calibrator = AutoCalibrator(node_positions)

    # In your processing loop - pass RAW packets:
    for packets in incoming_packets:
        frame = calibrator.ingest(packets, timestamp)
        for link in frame.links.values():
            if link.combined_change > 2.0 and link.is_reliable:
                print(f"Motion detected on {link.tx_mac}->{link.rx_mac}")

    # Save state on shutdown
    calibrator.save("calibration_state.json")
"""

import logging
import math
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from opencsi.core.types import (
    CalibratedFrame,
    CalibratedLink,
    CalibrationSnapshot,
    PerLinkBaseline,
    QualityGrade,
    RawCSIPacket,
)
from opencsi.core.welford import WelfordAccumulator
from opencsi.core.config import (
    AutoCalibratorConfig,
    SNR_FACTOR_MINIMUM,
    WAVELENGTH_2_4GHZ_M,
    channel_to_wavelength,
)
from opencsi.core.baseline import (
    BaselineProvider,
    get_baseline_manager,
)
from opencsi.core.reliability import (
    ReliabilityScorer,
    get_reliability_calculator,
)
from opencsi.core.motion import (
    MotionDetector,
    QuietPeriodDetector,
    get_motion_detector,
)
from opencsi.learning import (
    SensitivityMapLearner,
    AntennaPatternLearner,
    AntennaPattern,
    compute_distance,
)
from opencsi.geometry.fresnel import (
    compute_fresnel_radius_at_midpoint,
    compute_snr_scaling_factor,
)
from opencsi.sanitization import (
    CSISanitizer,
    SanitizerConfig,
    SanitizationLevel,
    SanitizedCSI,
)
from opencsi.baseline import (
    PassiveFrequencyCollector,
    PassiveTimeCollector,
    frequency_to_bucket,
    bucket_to_name,
)
from opencsi.learning.opportunistic import (
    OpportunisticCalibrator,
    CalibrationSweepRequest,
)
from opencsi.core.nbvi import NBVISubcarrierSelector
from opencsi.core.calibrator_persistence import (
    export_snapshot as _export_snapshot,
    restore_from_snapshot as _restore_from_snapshot,
    get_link_confidence as _get_link_confidence,
    save_state as _save_state,
    load_calibrator as _load_calibrator,
    compact_baselines as _compact_baselines,
    try_load_state as _try_load_state,
    reconstruct_welford as _reconstruct_welford,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Module-Level Constants
# =============================================================================

# --- Reliability Formula Constants ---
MIN_RELIABILITY = 0.1  # Floor for reliability score [0, 1]
UNINITIALIZED_BASELINE_RELIABILITY = 0.3  # Reliability when baseline not learned

# --- Detection Feedback ---
MIN_FEEDBACK_CONFIDENCE = 0.9  # Minimum confidence to accept detection feedback

# --- Default RSSI for API calls without actual RSSI data ---
GOOD_RSSI_DBM = -60.0  # RSSI where quality is "good"

# --- Q_rate Calculation (Sample rate quality factor) ---
REFERENCE_SAMPLE_RATE_HZ = 20.0  # Expected nominal rate

# --- SNR-Weighted Combination ---
SNR_SIGMOID_WIDTH = 5.0  # Sigmoid width for SNR-based phase/amplitude weighting

# --- Baseline Learning ---
MIN_STD_THRESHOLD = 0.1  # Minimum standard deviation (avoid divide-by-zero)
DEFAULT_SNR_ASSUMPTION = 20.0  # Assumed SNR when not provided

# --- Antenna/Sensitivity Caps ---
MAX_ANTENNA_GAIN_FACTOR = 2.0  # Cap on antenna gain correction effect

# --- Position Change Detection ---
SIGNIFICANT_POSITION_CHANGE_M = 0.1  # Position change threshold for stale detection

# --- Numerical Constants ---
NUMERICAL_EPSILON = 1e-10  # Small value to avoid log(0), divide-by-zero


# --- Baseline Gating ---
DEFAULT_GATING_RATIO = 0.0  # Default gating ratio when no events recorded

# --- AGC Warmup ---
AGC_WARMUP_SAMPLES_DEFAULT = 20
AGC_WARMUP_TIMEOUT_S_DEFAULT = 5.0

# --- Calibration Maturity ---
MATURITY_MIN_SAMPLES = 20
MATURITY_RELIABILITY_WEIGHT = 0.6
MATURITY_HEALTH_WEIGHT = 0.4

# --- Link Pruning Configuration ---
LINK_STALE_TIMEOUT_S = 3600.0  # Prune links not seen for 1 hour
LINK_PRUNE_INTERVAL_S = 300.0  # Check for stale links every 5 minutes
MAX_TRACKED_LINKS = 1000  # Hard limit on tracked links to prevent OOM

# --- Subcarrier Ghost Discrimination ---
# Minimum std for baseline to be considered initialized for coherence computation
SC_BASELINE_STD_MIN = 0.01
# Z-score threshold per subcarrier to count as "significantly affected"
SC_AFFECTED_ZSCORE_THRESHOLD = 1.5

# Doppler feature extraction from per-subcarrier phase differences.
# Sparsity threshold is per-link: DOPPLER_SPARSITY_SIGMA × baseline phase_std_scalar.
# This adapts to each link's noise floor rather than using a fixed radian threshold.
DOPPLER_SPARSITY_SIGMA = 2.0       # number of baseline σ for "active" subcarrier
DOPPLER_SPARSITY_FALLBACK = 0.3    # radians — used when baseline not yet initialized


def _compute_subcarrier_features(
    amplitude: np.ndarray,
    baseline: 'PerLinkBaseline',
) -> Tuple[float, float]:
    """
    Compute subcarrier-level features for ghost discrimination.

    A real human body is a broadband scatterer — it perturbs all subcarriers
    coherently (spectral shape is preserved, many subcarriers change).
    Narrowband interference or HVAC perturbation affects only a few subcarriers.

    Args:
        amplitude: Current per-subcarrier amplitude (n_subcarriers,)
        baseline: Learned baseline with per-subcarrier mean/std

    Returns:
        (subcarrier_coherence, amplitude_flatness, per_sc_zscore):
        - coherence: Pearson correlation between current and baseline spectral
          shape [0, 1]. High = shape preserved = broadband scatterer (human).
        - flatness: Fraction of subcarriers with significant perturbation [0, 1].
          High = many subcarriers affected = broadband (human).
        - per_sc_zscore: Per-subcarrier Z-score vector (n_subcarriers,).
          The spatial frequency signature of the perturbation.
    """
    if (not baseline.is_initialized
            or baseline.amplitude_mean is None
            or baseline.amplitude_std is None
            or len(amplitude) != baseline.n_subcarriers
            or baseline.n_subcarriers < 4):
        return 0.0, 0.0, np.zeros(0)

    baseline_mean = baseline.amplitude_mean
    baseline_std = baseline.amplitude_std

    # --- Subcarrier Coherence ---
    # Pearson correlation between current and baseline spectral shape.
    # A broadband perturbation (human) preserves the spectral shape (correlation
    # stays high); narrowband interference distorts it (correlation drops).
    current_centered = amplitude - (amplitude.sum() / len(amplitude))
    baseline_centered = baseline_mean - (baseline_mean.sum() / len(baseline_mean))
    current_norm = float(np.sqrt((current_centered * current_centered).sum()))
    baseline_norm = float(np.sqrt((baseline_centered * baseline_centered).sum()))

    if current_norm < SC_BASELINE_STD_MIN or baseline_norm < SC_BASELINE_STD_MIN:
        coherence = 0.0
    else:
        corr = float(np.dot(current_centered, baseline_centered) / (current_norm * baseline_norm))
        coherence = max(0.0, corr)  # Clamp negative correlation to 0

    # --- Amplitude Flatness ---
    # Fraction of subcarriers with |amplitude - baseline_mean| > threshold * baseline_std.
    # Broadband scatterer (human) → most subcarriers change → high flatness.
    # Narrowband interference → few subcarriers spike → low flatness.
    safe_std = np.maximum(baseline_std, SC_BASELINE_STD_MIN)
    per_sc_zscore = np.abs(amplitude - baseline_mean) / safe_std
    n_affected = int(np.sum(per_sc_zscore > SC_AFFECTED_ZSCORE_THRESHOLD))
    flatness = n_affected / baseline.n_subcarriers

    return coherence, flatness, per_sc_zscore


def _compute_effective_detection_width(
    link_distance_m: float,
    snr_db: float,
    wavelength_m: float = WAVELENGTH_2_4GHZ_M
) -> Tuple[float, float, float]:
    """
    Compute effective detection width for a link based on geometry and SNR.

    This is a thin wrapper that returns all three components needed by the calibrator.

    Args:
        link_distance_m: Distance between TX and RX in meters
        snr_db: Signal-to-noise ratio in dB
        wavelength_m: Signal wavelength in meters (default: 2.4GHz)

    Returns:
        Tuple of (fresnel_radius_m, snr_factor, effective_width_m)
    """
    fresnel_radius = compute_fresnel_radius_at_midpoint(link_distance_m, wavelength_m=wavelength_m)
    snr_factor = compute_snr_scaling_factor(snr_db)
    effective_width = 2 * fresnel_radius * snr_factor
    return fresnel_radius, snr_factor, effective_width


@dataclass
class LinkState:
    """Internal state for a single link."""
    baseline: PerLinkBaseline
    recent_variances: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_entropies: deque = field(default_factory=lambda: deque(maxlen=100))
    last_change_zscore: float = 0.0
    consecutive_high_change: int = 0
    sample_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_zscores: deque = field(default_factory=lambda: deque(maxlen=30))
    baseline_updates_gated: int = 0
    last_sample_time: float = 0.0
    recent_ipis: deque = field(default_factory=lambda: deque(maxlen=20))
    current_frequency_bucket: int = 2  # Default to "normal"
    current_frequency_hz: float = 10.0
    last_activity_time: float = field(default_factory=time.time)
    # Frequency metadata for hardware-agnostic Fresnel calculations
    channel: int = 6  # WiFi channel number
    frequency_band: str = "2.4GHz"  # "2.4GHz", "5GHz", or "6GHz"


class AutoCalibrator:
    """
    Self-calibrating CSI normalization layer.

    Transforms raw CSI into hardware-agnostic Z-scores by:
    1. SANITIZING raw CSI (AGC, phase correction, outliers) - WRAPPED INTERNALLY
    2. Auto-detecting quiet periods (empty room or periodic sources)
    3. Learning per-link baseline statistics during quiet periods
    4. Normalizing all measurements to Z-scores
    5. Tracking reliability based on multiplicative formula
    6. Learning spatial sensitivity maps
    7. Learning effective antenna patterns

    Attributes:
        node_positions: MAC address -> (x, y) position mapping
        config: AutoCalibratorConfig instance
        sanitizer: CSISanitizer instance (wrapped internally)
        sensitivity_learner: SensitivityMapLearner for spatial learning
        antenna_learner: AntennaPatternLearner for antenna patterns
    """

    def __init__(
        self,
        node_positions: Dict[str, Tuple[float, float]],
        config: Optional[AutoCalibratorConfig] = None,
        hardware_profile: Optional[dict] = None,
        baseline_manager: Optional[BaselineProvider] = None,
        reliability_calculator: Optional[ReliabilityScorer] = None,
        motion_detector: Optional[MotionDetector] = None,
        quiet_detector: Optional[QuietPeriodDetector] = None,
    ):
        """
        Initialize the calibration layer.

        Args:
            node_positions: MAC address -> (x, y) position mapping.
            config: Optional configuration (learning rates, thresholds, etc.)
            hardware_profile: Optional hardware defaults for cold start
            baseline_manager: Pluggable baseline manager (default: EMA-based)
            reliability_calculator: Pluggable reliability scorer (default: multiplicative)
            motion_detector: Pluggable motion detector (default: threshold-based)
            quiet_detector: Pluggable quiet period detector (default: variance/entropy)
        """
        if not node_positions:
            raise ValueError("node_positions cannot be empty")

        self.node_positions = node_positions.copy()
        self.config = config or AutoCalibratorConfig()
        self._hardware_profile = hardware_profile

        # === PLUGGABLE COMPONENTS ===
        self.baseline_manager: BaselineProvider = baseline_manager or get_baseline_manager(
            "ema",
            ema_alpha=self.config.baseline_learning_rate,
            min_samples=self.config.min_baseline_samples,
        )
        self.reliability_calculator: ReliabilityScorer = (
            reliability_calculator or get_reliability_calculator("default")
        )
        self.motion_detector: MotionDetector = motion_detector or get_motion_detector(
            "threshold",
            threshold=self.config.drift_threshold_sigma,
        )
        self.quiet_detector: QuietPeriodDetector = quiet_detector or QuietPeriodDetector(
            variance_threshold=self.config.variance_low_threshold,
            entropy_threshold=self.config.entropy_low_threshold,
        )

        # === SANITIZER (wrapped internally) ===
        sanitizer_config = SanitizerConfig(
            level=SanitizationLevel.STANDARD,
            enable_agc_correction=self.config.enable_agc_correction,
            enable_phase_correction=self.config.enable_phase_correction,
            enable_outlier_removal=self.config.enable_outlier_removal,
            enable_hamming_window=self.config.enable_hamming_window,
            enable_nbvi_selection=self.config.enable_nbvi_selection,
        )
        self.sanitizer = CSISanitizer(config=sanitizer_config)

        # Per-link state
        self._link_states: Dict[Tuple[str, str], LinkState] = {}

        # Link pruning state
        self._last_prune_time = time.time()

        # Global state
        self._is_quiet = True
        self._last_quiet_time = time.time()
        self._last_save_time = time.time()
        self._drift_detected = False
        self._total_frames_processed = 0
        self._current_frame: Optional[CalibratedFrame] = None

        # Statistics
        self._quiet_frames = 0
        self._active_frames = 0

        # Force quiet period state
        self._force_quiet_until: float = 0.0

        # Baseline freeze state
        self._baseline_frozen = False

        # AGC warmup state
        self._agc_warmup_complete: bool = False
        self._agc_warmup_samples: int = 0
        self._agc_warmup_start: float = time.time()

        # Temporal stability tracking for quiet detection
        self._recent_mean_amplitudes: deque = deque(
            maxlen=self.config.temporal_amplitude_history_size
        )

        # Initialize learning components
        self.sensitivity_learner = SensitivityMapLearner(
            node_positions=node_positions,
            resolution=self.config.sensitivity_grid_resolution_m,
            learning_rate=self.config.sensitivity_learning_rate,
            min_observations=self.config.min_observations_for_confidence
        )

        self.antenna_learner = AntennaPatternLearner(
            node_positions=node_positions,
            angular_resolution_deg=self.config.antenna_angular_resolution_deg,
            learning_rate=self.config.antenna_learning_rate
        )

        # Frequency-aware calibration (Mode 1: Passive collection)
        self._passive_frequency_collector = PassiveFrequencyCollector()

        # Time-of-day aware calibration
        self._passive_time_collector = PassiveTimeCollector(
            learning_rate=self.config.baseline_learning_rate
        )

        # Frequency-aware calibration (Mode 2: Opportunistic sweeps)
        self._opportunistic_calibrator = OpportunisticCalibrator(
            passive_collector=self._passive_frequency_collector
        )

        # NBVI subcarrier selection
        if self.config.enable_nbvi_selection:
            self._nbvi_selector = NBVISubcarrierSelector(
                n_subcarriers=self.config.nbvi_n_subcarriers,
                min_quiet_samples=self.config.nbvi_min_quiet_samples,
                recompute_interval_s=self.config.nbvi_recompute_interval_s
            )
        else:
            self._nbvi_selector = None

        logger.info(
            f"AutoCalibrator initialized with {len(node_positions)} nodes, "
            f"NBVI={'enabled' if self._nbvi_selector else 'disabled'}"
        )

    def _check_agc_warmup(self, n_samples: int, timestamp: float) -> None:
        """Check and update AGC warmup state."""
        if self._agc_warmup_complete:
            return

        self._agc_warmup_samples += n_samples
        elapsed = timestamp - self._agc_warmup_start

        if (self._agc_warmup_samples >= self.config.agc_warmup_samples or
                elapsed >= self.config.agc_warmup_timeout_s):
            self._agc_warmup_complete = True
            logger.info(
                f"AGC warmup complete: {self._agc_warmup_samples} samples in {elapsed:.1f}s"
            )

    def ingest(
        self,
        raw_packets: Union[Dict[Tuple[str, str], RawCSIPacket], Dict[Tuple[str, str], dict], List],
        timestamp: Optional[float] = None,
        hardware_id: Optional[str] = None
    ) -> CalibratedFrame:
        """
        Primary method. Takes RAW CSI, sanitizes it, returns calibrated CSI.

        Args:
            raw_packets: Dict mapping (tx_mac, rx_mac) to RawCSIPacket or dict,
                        or list of CSIPacket objects from hardware extractor
            timestamp: Frame timestamp (uses current time if None)
            hardware_id: Optional hardware identifier for profile lookup

        Returns:
            CalibratedFrame with normalized values and reliability metadata
        """
        timestamp = timestamp or time.time()
        self._total_frames_processed += 1

        # Handle list of CSIPackets (from hardware extractor)
        if isinstance(raw_packets, list):
            raw_packets = self._convert_packet_list(raw_packets)

        # Infer subcarrier count from the first valid packet
        n_subcarriers = 0
        if raw_packets:
            for packet in raw_packets.values():
                iq_data = None
                if isinstance(packet, RawCSIPacket):
                    iq_data = packet.iq_data
                elif isinstance(packet, dict):
                    iq_data = packet.get('iq_data')

                if iq_data is not None and hasattr(iq_data, 'shape'):
                    n_subcarriers = iq_data.shape[0]
                    break

        if n_subcarriers == 0 and raw_packets:
            logger.warning("Could not determine subcarrier count. Aborting frame.")
            return CalibratedFrame(timestamp=timestamp, links={})

        # Check AGC warmup
        self._check_agc_warmup(len(raw_packets), timestamp)

        # Create output frame
        frame = CalibratedFrame(timestamp=timestamp, links={})

        # Collect variance and entropy for quiet detection
        all_variances = []
        all_entropies = []

        # Store sanitized data for baseline updates
        sanitized_data_cache: Dict[Tuple[str, str], SanitizedCSI] = {}
        amplitude_means_cache: Dict[Tuple[str, str], float] = {}
        amplitude_mean_sum = 0.0

        # Process each link
        for link_key, raw_packet in raw_packets.items():
            tx_mac, rx_mac = link_key

            # === STEP 1: SANITIZE RAW CSI ===
            sanitized = self._sanitize_packet(raw_packet, tx_mac, rx_mac, timestamp, n_subcarriers)
            sanitized_data_cache[link_key] = sanitized

            # Ensure link state exists
            if link_key not in self._link_states:
                self._link_states[link_key] = self._create_link_state(tx_mac, rx_mac)

            state = self._link_states[link_key]
            state.last_activity_time = timestamp
            state.sample_timestamps.append(timestamp)

            # Extract channel/frequency_band from packet for wavelength calculation
            if isinstance(raw_packet, RawCSIPacket):
                state.channel = raw_packet.channel
                state.frequency_band = raw_packet.frequency_band
            elif isinstance(raw_packet, dict):
                state.channel = raw_packet.get('channel', 6)
                state.frequency_band = raw_packet.get('frequency_band', "2.4GHz")

            # Compute instantaneous frequency
            freq_state = self._passive_frequency_collector.get_or_create_state(link_key)
            freq_hz = freq_state.compute_instantaneous_frequency(timestamp)
            bucket_idx = frequency_to_bucket(freq_hz)
            state.current_frequency_bucket = bucket_idx
            state.current_frequency_hz = freq_hz

            # Extract measurements from sanitized data
            amplitude = sanitized.amplitude
            phase = sanitized.phase
            rssi = sanitized.rssi
            snr = sanitized.snr
            phase_diff_std = sanitized.phase_diff_std if sanitized.phase_diff_available else 0.0
            sample_rate = self._estimate_sample_rate(state)

            # Compute variance (inline to avoid np.var dispatch overhead)
            amp_mean = float(amplitude.sum() / len(amplitude))
            variance = float(((amplitude - amp_mean) ** 2).sum() / len(amplitude))
            state.recent_variances.append(variance)
            amplitude_means_cache[link_key] = amp_mean
            amplitude_mean_sum += amp_mean

            # Compute spectral entropy
            spectral_entropy = self._compute_spectral_entropy(amplitude)
            if spectral_entropy > 0:
                state.recent_entropies.append(spectral_entropy)
                all_entropies.append(spectral_entropy)

            all_variances.append(variance)

            # === STEP 2: CALIBRATE against learned baseline ===
            if self._nbvi_selector is not None:
                amplitude = self._nbvi_selector.apply_selection(amplitude, link_key)

            calibrated_link = self._calibrate_link(
                state=state,
                amplitude=amplitude,
                phase=phase,
                phase_diff_std=phase_diff_std,
                snr=snr,
                rssi=rssi,
                sample_rate=sample_rate,
                timestamp=timestamp
            )

            frame.links[link_key] = calibrated_link

        # Compute frame mean amplitude for quiet detection
        link_count = len(amplitude_means_cache)
        frame_mean_amplitude = amplitude_mean_sum / link_count if link_count > 0 else 0.0

        # Finalize frame with shared post-processing
        return self._finalize_frame(
            frame=frame,
            sanitized_items=sanitized_data_cache,
            all_entropies=all_entropies,
            frame_mean_amplitude=frame_mean_amplitude,
            amplitude_means=amplitude_means_cache,
            n_subcarriers=n_subcarriers,
            timestamp=timestamp
        )

    def ingest_sanitized(
        self,
        sanitized_data: Dict[Tuple[str, str], dict],
        timestamp: Optional[float] = None
    ) -> CalibratedFrame:
        """
        Alternative method for users who want to use their own sanitization.

        **Note**: The recommended API is ingest() which handles sanitization
        internally. Only use this if you have specific sanitization requirements.

        Args:
            sanitized_data: Dict mapping (tx_mac, rx_mac) to sanitized CSI data.
                           Each value should have: amplitude, phase, rssi, snr,
                           phase_diff_std (optional), sample_rate (optional)
            timestamp: Frame timestamp (uses current time if None)

        Returns:
            CalibratedFrame with normalized values and reliability metadata
        """
        timestamp = timestamp or time.time()
        self._total_frames_processed += 1

        # Infer subcarrier count from the first valid packet
        n_subcarriers = 0
        if sanitized_data:
            first_packet = next(iter(sanitized_data.values()), None)
            if first_packet and 'amplitude' in first_packet:
                n_subcarriers = len(first_packet['amplitude'])

        if n_subcarriers == 0 and sanitized_data:
            logger.warning("Could not determine subcarrier count from sanitized CSI data. Aborting frame.")
            return CalibratedFrame(timestamp=timestamp, links={})

        # Check AGC warmup (blocks baseline updates until stable)
        self._check_agc_warmup(len(sanitized_data), timestamp)

        # Create output frame
        frame = CalibratedFrame(
            timestamp=timestamp,
            links={}
        )

        # Collect variance and entropy for quiet detection
        all_variances = []
        all_entropies = []

        # Process each link
        for link_key, data in sanitized_data.items():
            tx_mac, rx_mac = link_key

            # Ensure link state exists
            if link_key not in self._link_states:
                self._link_states[link_key] = self._create_link_state(tx_mac, rx_mac)

            state = self._link_states[link_key]

            # Update activity timestamp for link pruning
            state.last_activity_time = timestamp

            # Track sample timing for rate quality
            state.sample_timestamps.append(timestamp)

            # Extract channel/frequency_band from dict for wavelength calculation
            state.channel = data.get('channel', 6)
            state.frequency_band = data.get('frequency_band', "2.4GHz")

            # === Compute instantaneous frequency BEFORE calibration ===
            freq_state = self._passive_frequency_collector.get_or_create_state(link_key)
            freq_hz = freq_state.compute_instantaneous_frequency(timestamp)
            bucket_idx = frequency_to_bucket(freq_hz)
            state.current_frequency_bucket = bucket_idx
            state.current_frequency_hz = freq_hz

            # Extract measurements from sanitized data
            amplitude = data.get('amplitude', np.zeros(n_subcarriers))
            phase = data.get('phase', np.zeros(n_subcarriers))
            rssi = data.get('rssi', -50)
            snr = data.get('snr', DEFAULT_SNR_ASSUMPTION)
            phase_diff_std = data.get('phase_diff_std', 0.0)
            spectral_entropy = data.get('spectral_entropy', 0.0)
            sample_rate = data.get('sample_rate', self._estimate_sample_rate(state))

            # Compute variance (inline to avoid np.var dispatch)
            _amp_m = float(amplitude.sum() / len(amplitude))
            variance = float(((amplitude - _amp_m) ** 2).sum() / len(amplitude))
            state.recent_variances.append(variance)

            if spectral_entropy > 0:
                state.recent_entropies.append(spectral_entropy)
                all_entropies.append(spectral_entropy)

            all_variances.append(variance)

            # Apply NBVI subcarrier weighting if enabled (weights stable subcarriers higher)
            if self._nbvi_selector is not None:
                amplitude = self._nbvi_selector.apply_selection(amplitude, link_key)

            # Compute calibrated measurements
            calibrated_link = self._calibrate_link(
                state=state,
                amplitude=amplitude,
                phase=phase,
                phase_diff_std=phase_diff_std,
                snr=snr,
                rssi=rssi,
                sample_rate=sample_rate,
                timestamp=timestamp
            )

            frame.links[link_key] = calibrated_link

        # Compute frame mean amplitude for quiet detection
        all_amplitudes = [d.get('amplitude', np.zeros(1)).mean() for d in sanitized_data.values()]
        frame_mean_amplitude = float(np.mean(all_amplitudes)) if all_amplitudes else 0.0

        # Finalize frame with shared post-processing
        return self._finalize_frame(
            frame=frame,
            sanitized_items=sanitized_data,
            all_entropies=all_entropies,
            frame_mean_amplitude=frame_mean_amplitude,
            amplitude_means=None,
            n_subcarriers=n_subcarriers,
            timestamp=timestamp
        )

    def _convert_packet_list(self, packets: List) -> Dict[Tuple[str, str], RawCSIPacket]:
        """Convert list of CSIPacket objects to dict format."""
        result = {}
        for packet in packets:
            if hasattr(packet, 'tx_mac') and hasattr(packet, 'rx_mac'):
                link_key = (packet.tx_mac, packet.rx_mac)
                result[link_key] = RawCSIPacket.from_csi_packet(packet)
        return result

    def _sanitize_packet(
        self,
        raw_packet: Union[RawCSIPacket, dict],
        tx_mac: str,
        rx_mac: str,
        timestamp: float,
        n_subcarriers: int
    ) -> SanitizedCSI:
        """Sanitize a raw CSI packet using the internal sanitizer."""
        if isinstance(raw_packet, RawCSIPacket):
            return self.sanitizer.sanitize(
                iq_data=raw_packet.iq_data,
                tx_mac=raw_packet.tx_mac,
                rx_mac=raw_packet.rx_mac,
                timestamp=raw_packet.timestamp,
                rssi=raw_packet.rssi,
                noise_floor=raw_packet.noise_floor,
                rx_state=raw_packet.rx_state,
                pilot_phases=raw_packet.pilot_phases
            )
        else:
            # Dict format
            iq_data = raw_packet.get('iq_data')
            if iq_data is None:
                amplitude = raw_packet.get('amplitude', np.zeros(n_subcarriers))
                phase = raw_packet.get('phase', np.zeros(n_subcarriers))
                return self._create_mock_sanitized(
                    amplitude, phase, tx_mac, rx_mac, timestamp,
                    raw_packet.get('rssi', -50),
                    raw_packet.get('noise_floor', -90),
                    raw_packet.get('snr', DEFAULT_SNR_ASSUMPTION),
                    raw_packet.get('phase_diff_std', 0.0)
                )

            return self.sanitizer.sanitize(
                iq_data=iq_data,
                tx_mac=tx_mac,
                rx_mac=rx_mac,
                timestamp=timestamp,
                rssi=raw_packet.get('rssi', -50),
                noise_floor=raw_packet.get('noise_floor', -90),
                rx_state=raw_packet.get('rx_state', 0),
                pilot_phases=raw_packet.get('pilot_phases')
            )

    def _create_mock_sanitized(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        tx_mac: str,
        rx_mac: str,
        timestamp: float,
        rssi: int,
        noise_floor: int,
        snr: float,
        phase_diff_std: float
    ) -> SanitizedCSI:
        """Create a mock SanitizedCSI from pre-sanitized data."""
        complex_csi = amplitude * np.exp(1j * phase)
        return SanitizedCSI(
            amplitude=amplitude,
            phase=phase,
            complex_csi=complex_csi,
            tx_mac=tx_mac,
            rx_mac=rx_mac,
            timestamp=timestamp,
            rssi=rssi,
            noise_floor=noise_floor,
            snr=snr,
            phase_diff_std=phase_diff_std,
            phase_diff_available=phase_diff_std > 0
        )

    def _finalize_frame(
        self,
        frame: CalibratedFrame,
        sanitized_items: Dict[Tuple[str, str], Union[SanitizedCSI, dict]],
        all_entropies: List[float],
        frame_mean_amplitude: float,
        amplitude_means: Optional[Dict[Tuple[str, str], float]],
        n_subcarriers: int,
        timestamp: float
    ) -> CalibratedFrame:
        """
        Finalize frame processing with shared post-processing logic.

        This method consolidates common logic between ingest() and ingest_sanitized().
        """
        # Update temporal amplitude history for quiet detection
        self._recent_mean_amplitudes.append(frame_mean_amplitude)

        # Compute temporal variance
        if len(self._recent_mean_amplitudes) >= self.config.min_temporal_samples:
            _tmp = np.array(list(self._recent_mean_amplitudes))
            _m = _tmp.sum() / len(_tmp)
            temporal_variance = float(((_tmp - _m) ** 2).sum() / len(_tmp))
        else:
            temporal_variance = float('inf')

        mean_entropy = float(sum(all_entropies) / len(all_entropies)) if all_entropies else 0.0
        self._is_quiet = self._detect_quiet(temporal_variance, mean_entropy)

        # Frequency-aware calibration (Mode 1: Passive Collection)
        if self._agc_warmup_complete:
            self._update_frequency_baselines(
                frame, sanitized_items, timestamp, amplitude_means, n_subcarriers
            )

        # Update quiet/active frame counters and baselines
        if self._is_quiet:
            self._quiet_frames += 1
            self._last_quiet_time = timestamp
            if self._agc_warmup_complete:
                self._update_baselines_with_gating(
                    frame, sanitized_items, timestamp, n_subcarriers
                )
        else:
            self._active_frames += 1

        # Detect drift
        self._drift_detected = self._detect_drift()

        # Update frame metadata
        oldest_baseline = self._get_oldest_baseline_age()
        frame.calibration_age_s = oldest_baseline
        frame.environment_stable = self._is_quiet
        frame.baseline_drift_detected = self._drift_detected
        frame.update_aggregate_stats()

        # Attach per-subcarrier phase-diff and Doppler features from sanitized data.
        # The phase_diff vector is already computed by the sanitizer but was
        # previously collapsed to a single scalar (phase_diff_std).
        # Sparsity threshold is per-link: k × baseline_phase_std (adapts to noise floor).
        max_doppler = 0.0
        doppler_sum = 0.0
        doppler_count = 0
        for link_key, sanitized in sanitized_items.items():
            if link_key not in frame.links:
                continue
            link = frame.links[link_key]
            phase_diff = getattr(sanitized, 'phase_diff', None)
            if phase_diff is not None and len(phase_diff) > 0:
                link.per_subcarrier_phase_diff = phase_diff
                energy = float(np.sum(phase_diff ** 2))
                link.doppler_energy = energy

                # Per-link adaptive threshold from baseline phase noise
                state = self._link_states.get(link_key)
                if state and state.baseline.is_initialized and state.baseline.phase_std_scalar > 1e-6:
                    threshold = DOPPLER_SPARSITY_SIGMA * state.baseline.phase_std_scalar
                else:
                    threshold = DOPPLER_SPARSITY_FALLBACK
                link.doppler_sparsity = float(
                    np.mean(np.abs(phase_diff) > threshold)
                )

                doppler_sum += energy
                doppler_count += 1
                if energy > max_doppler:
                    max_doppler = energy

        if doppler_count > 0:
            frame.mean_doppler_energy = doppler_sum / doppler_count
            frame.max_doppler_energy = max_doppler

        # Compute system-wide calibration maturity
        frame.calibration_maturity = self._compute_calibration_maturity(frame)

        # Expose AGC warmup status
        frame.agc_warmup_complete = self._agc_warmup_complete

        # Mode 2: Opportunistic calibration
        self._opportunistic_calibrator.on_frame(self._is_quiet, timestamp)

        # Update frame with frequency calibration metadata
        self._update_frame_frequency_metadata(frame)

        # Store current frame for feedback API
        self._current_frame = frame

        # Prune stale links
        self._prune_stale_links(timestamp)

        # Auto-save if needed
        self._maybe_auto_save()

        return frame

    def _compute_spectral_entropy(self, amplitude: np.ndarray) -> float:
        """Compute spectral entropy for quiet detection."""
        power = amplitude ** 2
        total_power = np.sum(power)
        if total_power < NUMERICAL_EPSILON:
            return 0.0

        p = power / total_power
        p = np.clip(p, NUMERICAL_EPSILON, 1.0)
        entropy = -np.sum(p * np.log2(p))
        return float(entropy)

    def _create_link_state(self, tx_mac: str, rx_mac: str) -> LinkState:
        """Create new link state with hardware profile defaults."""
        baseline = PerLinkBaseline(
            tx_mac=tx_mac,
            rx_mac=rx_mac,
            creation_time=time.time()
        )

        if self._hardware_profile:
            baseline.amplitude_std_scalar = self._hardware_profile.get(
                'typical_amplitude_sigma', 2.5
            )
            baseline.phase_std_scalar = self._hardware_profile.get(
                'typical_phase_sigma_rad', 0.8
            )
            baseline.quiet_samples = 1

        return LinkState(baseline=baseline)

    def _estimate_sample_rate(self, state: LinkState) -> float:
        """Estimate sample rate from recent timestamps."""
        if len(state.sample_timestamps) < 2:
            return REFERENCE_SAMPLE_RATE_HZ

        timestamps = state.sample_timestamps
        n = len(timestamps)
        if n < 2:
            return REFERENCE_SAMPLE_RATE_HZ

        total_span = timestamps[-1] - timestamps[0]
        if total_span > NUMERICAL_EPSILON:
            return (n - 1) / total_span
        return REFERENCE_SAMPLE_RATE_HZ

    def _calibrate_link(
        self,
        state: LinkState,
        amplitude: np.ndarray,
        phase: np.ndarray,
        phase_diff_std: float,
        snr: float,
        rssi: int,
        sample_rate: float,
        timestamp: float
    ) -> CalibratedLink:
        """Compute calibrated measurements for a single link."""
        baseline = state.baseline

        # Compute Z-scores
        amplitude_zscore = baseline.compute_amplitude_zscore(amplitude)
        phase_zscore = baseline.compute_phase_zscore(phase_diff_std)

        # Combined change with proper SNR-based weighting
        combined_change = self._compute_combined_change(
            amplitude_zscore, phase_zscore, snr
        )

        # Compute reliability
        reliability = self._compute_reliability(
            baseline, snr, rssi, sample_rate, state
        )

        # Determine quality grade
        quality_grade = QualityGrade.from_reliability(reliability, snr)

        # Track Z-scores for drift detection
        state.recent_zscores.append(combined_change)
        state.last_change_zscore = combined_change

        # Use pluggable motion detector for consecutive tracking
        link_key = (baseline.tx_mac, baseline.rx_mac)
        motion_detected = self.motion_detector.detect(
            z_score=combined_change,
            reliability=reliability,
            link_key=link_key,
        )

        # Track consecutive high readings (sync with motion detector state)
        if hasattr(self.motion_detector, 'get_consecutive_count'):
            state.consecutive_high_change = self.motion_detector.get_consecutive_count(link_key)
        elif combined_change > self.config.drift_threshold_sigma:
            state.consecutive_high_change += 1
        else:
            state.consecutive_high_change = 0

        # Get frequency calibration info
        freq_bucket = state.current_frequency_bucket
        freq_hz = state.current_frequency_hz
        freq_interpolated = False
        freq_confidence = 1.0

        freq_baseline = self._passive_frequency_collector.get_frequency_baseline(link_key)
        if freq_baseline:
            _, freq_interpolated, _ = freq_baseline.get_baseline_with_interpolation(freq_bucket)
            freq_confidence = freq_baseline.get_calibration_confidence(freq_bucket)

        # Compute baseline gating ratio
        total_baseline_events = state.baseline_updates_gated + state.baseline.quiet_samples
        if total_baseline_events > 0:
            gating_ratio = state.baseline_updates_gated / total_baseline_events
        else:
            gating_ratio = DEFAULT_GATING_RATIO

        # Compute wavelength from channel for frequency-aware Fresnel calculations
        channel = state.channel
        frequency_band = state.frequency_band
        wavelength_m = channel_to_wavelength(channel, frequency_band)

        # Compute effective detection width using actual wavelength
        link_distance = 0.0
        fresnel_radius = 0.0
        snr_scale_factor = SNR_FACTOR_MINIMUM
        effective_width = 0.0

        tx_pos = self.node_positions.get(baseline.tx_mac)
        rx_pos = self.node_positions.get(baseline.rx_mac)
        if tx_pos is not None and rx_pos is not None:
            link_distance = compute_distance(tx_pos, rx_pos)
            fresnel_radius, snr_scale_factor, effective_width = _compute_effective_detection_width(
                link_distance, snr, wavelength_m
            )

        # Compute subcarrier-level ghost discrimination features
        sc_coherence, amp_flatness, sc_zscore = _compute_subcarrier_features(
            amplitude, baseline
        )

        return CalibratedLink(
            tx_mac=baseline.tx_mac,
            rx_mac=baseline.rx_mac,
            amplitude_change=amplitude_zscore,
            phase_change=phase_zscore,
            combined_change=combined_change,
            reliability=reliability,
            baseline_age_s=baseline.age_seconds,
            snr_estimate=snr,
            quality_grade=quality_grade,
            raw_amplitude=amplitude,
            raw_phase=phase,
            amplitude_mean=float(amplitude.sum() / len(amplitude)),
            phase_diff_std=phase_diff_std,
            baseline_samples=baseline.quiet_samples,
            frequency_bucket=freq_bucket,
            frequency_hz=freq_hz,
            calibration_interpolated=freq_interpolated,
            bucket_confidence=freq_confidence,
            baseline_gating_ratio=gating_ratio,
            link_distance_m=link_distance,
            fresnel_radius_m=fresnel_radius,
            effective_detection_width_m=effective_width,
            snr_scaling_factor=snr_scale_factor,
            channel=channel,
            frequency_band=frequency_band,
            wavelength_m=wavelength_m,
            subcarrier_coherence=sc_coherence,
            amplitude_flatness=amp_flatness,
            per_subcarrier_zscore=sc_zscore if len(sc_zscore) > 0 else None,
        )

    def _compute_combined_change(
        self,
        amplitude_zscore: float,
        phase_zscore: float,
        snr: float
    ) -> float:
        """Compute SNR-weighted combination of amplitude and phase change."""
        transition = self.config.snr_weight_transition_db

        phase_weight = 1.0 / (1.0 + math.exp(-(snr - transition) / SNR_SIGMOID_WIDTH))
        amplitude_weight = 1.0 - phase_weight

        combined = amplitude_weight * amplitude_zscore + phase_weight * phase_zscore
        return combined

    def _compute_reliability(
        self,
        baseline: PerLinkBaseline,
        snr: float,
        rssi: int,
        sample_rate: float,
        state: LinkState
    ) -> float:
        """Compute link reliability score [0, 1].

        Delegates to pluggable ReliabilityCalculator which implements:
            R = Q_cv * Q_age * Q_rate * Q_rssi * Q_sample
        """
        if not baseline.is_initialized:
            return UNINITIALIZED_BASELINE_RELIABILITY

        q_rate = self._compute_q_rate(state)

        return self.reliability_calculator.compute(
            rssi_dbm=float(rssi),
            amplitude_std=baseline.amplitude_std_scalar,
            amplitude_mean=baseline.amplitude_mean_scalar,
            baseline_age_s=baseline.age_seconds,
            decay_lambda=self.config.reliability_decay_lambda,
            q_rate=q_rate,
            quiet_samples=baseline.quiet_samples,
            min_baseline_samples=self.config.min_baseline_samples,
        )

    def _compute_q_rate(self, state: LinkState) -> float:
        """Quality factor based on frequency bucket stability."""
        link_key = (state.baseline.tx_mac, state.baseline.rx_mac)
        freq_baseline = self._passive_frequency_collector.get_frequency_baseline(link_key)

        if not freq_baseline:
            return 0.1

        return freq_baseline.get_calibration_confidence(state.current_frequency_bucket)

    def _detect_quiet(self, variance: float, entropy: float) -> bool:
        """Detect if environment is in quiet state.

        Uses pluggable QuietPeriodDetector for variance/entropy thresholds.
        """
        if time.time() < self._force_quiet_until:
            return True

        # Use pluggable quiet detector
        is_quiet = self.quiet_detector.is_quiet(variance, entropy)
        if is_quiet:
            return True

        # Legacy high-variance low-entropy check (periodic sources)
        if (variance > self.config.variance_high_threshold and
                entropy < self.config.entropy_low_threshold):
            return True

        return False

    def _update_baseline_if_quiet(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
        phase_diff_std: float,
        timestamp: float
    ):
        """Update link baseline during quiet period.

        Updates both the PerLinkBaseline (scalar aggregates) and the pluggable BaselineManager (per-subcarrier).
        """
        if self._baseline_frozen:
            return

        if link_key not in self._link_states:
            return

        state = self._link_states[link_key]
        baseline = state.baseline

        alpha = self.config.baseline_learning_rate

        current_amp_mean = float(amplitude.sum() / len(amplitude))
        current_amp_std = float(np.std(amplitude))
        current_amp_variance = current_amp_std ** 2

        # Update pluggable baseline manager
        self.baseline_manager.update(
            link_key=link_key,
            amplitude=amplitude,
            phase=phase,
            is_quiet=True,
        )

        # Update PerLinkBaseline scalar aggregates
        if not baseline.is_initialized:
            baseline.amplitude_mean_scalar = current_amp_mean
            baseline.amplitude_variance_scalar = current_amp_variance
            baseline.amplitude_std_scalar = max(current_amp_std, MIN_STD_THRESHOLD)
            baseline.phase_std_scalar = max(phase_diff_std, MIN_STD_THRESHOLD)
            baseline.is_initialized = True
            baseline.quiet_samples = 1
            baseline.last_update_time = timestamp
            baseline.creation_time = timestamp

            # Initialize per-subcarrier arrays from first quiet sample
            baseline.ensure_arrays(len(amplitude))
            baseline.amplitude_mean[:] = amplitude
            baseline.amplitude_std[:] = np.maximum(np.abs(amplitude) * 0.1, SC_BASELINE_STD_MIN)
        else:
            baseline.amplitude_mean_scalar = (
                (1 - alpha) * baseline.amplitude_mean_scalar +
                alpha * current_amp_mean
            )

            baseline.amplitude_variance_scalar = (
                (1 - alpha) * baseline.amplitude_variance_scalar +
                alpha * current_amp_variance
            )

            baseline.amplitude_std_scalar = max(
                math.sqrt(baseline.amplitude_variance_scalar),
                MIN_STD_THRESHOLD
            )

            if phase_diff_std > 0:
                baseline.phase_std_scalar = (
                    (1 - alpha) * baseline.phase_std_scalar +
                    alpha * phase_diff_std
                )

            # Update per-subcarrier arrays via EMA
            if (baseline.amplitude_mean is not None
                    and len(amplitude) == baseline.n_subcarriers):
                # Compute deviation from OLD mean before updating it
                deviation = np.abs(amplitude - baseline.amplitude_mean)
                baseline.amplitude_mean[:] = (
                    (1 - alpha) * baseline.amplitude_mean + alpha * amplitude
                )
                baseline.amplitude_std[:] = np.maximum(
                    (1 - alpha) * baseline.amplitude_std + alpha * deviation,
                    SC_BASELINE_STD_MIN,
                )

            baseline.quiet_samples += 1
            baseline.last_update_time = timestamp

    def _detect_drift(self) -> bool:
        """Detect if environment has changed (drift)."""
        drift_count = 0

        for state in self._link_states.values():
            if len(state.recent_zscores) >= 10:
                recent = list(state.recent_zscores)
                mean_zscore = np.mean(recent)
                if mean_zscore > self.config.drift_threshold_sigma:
                    drift_count += 1

        drift_fraction = drift_count / max(1, len(self._link_states))
        return drift_fraction > self.config.drift_link_fraction

    def _get_oldest_baseline_age(self) -> float:
        """Get age of oldest baseline."""
        if not self._link_states:
            return float('inf')

        ages = [
            state.baseline.age_seconds
            for state in self._link_states.values()
            if state.baseline.is_initialized
        ]
        return max(ages) if ages else float('inf')

    def _maybe_auto_save(self):
        """Auto-save state if interval has passed."""
        now = time.time()
        if now - self._last_save_time >= self.config.auto_save_interval_s:
            try:
                self.save(self.config.state_file_path)
                self._last_save_time = now
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")

    def _cleanup_link_related_structures(self, link_key: Tuple[str, str]) -> None:
        """Clean up all structures associated with a link."""
        if self.sensitivity_learner:
            self.sensitivity_learner.remove_link(link_key)
        if self._passive_frequency_collector:
            self._passive_frequency_collector.remove_link(link_key)
        if self.sanitizer:
            self.sanitizer.clear_link_history(link_key[0], link_key[1])
        # Clean up motion/quiet detector per-link tracking to prevent memory leak
        if hasattr(self.motion_detector, 'remove_link'):
            self.motion_detector.remove_link(link_key)
        if hasattr(self.quiet_detector, 'reset_link'):
            self.quiet_detector.reset_link(link_key)

    def _prune_stale_links(self, timestamp: float):
        """Prune links that haven't been seen recently."""
        if timestamp - self._last_prune_time < LINK_PRUNE_INTERVAL_S:
            return

        self._last_prune_time = timestamp
        cutoff = timestamp - LINK_STALE_TIMEOUT_S

        stale_links = [
            link_key for link_key, state in self._link_states.items()
            if state.last_activity_time < cutoff
        ]

        if stale_links:
            logger.info(f"Pruning {len(stale_links)} stale links")
            for link_key in stale_links:
                del self._link_states[link_key]
                self._cleanup_link_related_structures(link_key)

        if len(self._link_states) > MAX_TRACKED_LINKS:
            sorted_links = sorted(
                self._link_states.items(),
                key=lambda x: x[1].last_activity_time
            )
            excess = len(self._link_states) - MAX_TRACKED_LINKS
            links_to_remove = [k for k, _ in sorted_links[:excess]]

            logger.warning(f"Hard limit exceeded, pruning {excess} oldest links")
            for link_key in links_to_remove:
                del self._link_states[link_key]
                self._cleanup_link_related_structures(link_key)

    def _is_link_individually_quiet(
        self,
        frame: CalibratedFrame,
        link_key: Tuple[str, str]
    ) -> bool:
        """Check whether a specific link is individually quiet."""
        calibrated_link = frame.links.get(link_key)
        if calibrated_link is None:
            return True
        return abs(calibrated_link.combined_change) <= self.config.per_link_activity_zscore_threshold

    def _update_baselines_with_gating(
        self,
        frame: CalibratedFrame,
        sanitized_items: Dict[Tuple[str, str], Union[SanitizedCSI, dict]],
        timestamp: float,
        n_subcarriers: int
    ) -> None:
        """Update baselines for all links during a quiet period, gating active links."""
        for link_key, sanitized in sanitized_items.items():
            if not self._is_link_individually_quiet(frame, link_key):
                state = self._link_states.get(link_key)
                if state is not None:
                    state.baseline_updates_gated += 1
                continue

            if isinstance(sanitized, SanitizedCSI):
                amplitude = sanitized.amplitude
                phase = sanitized.phase
                phase_diff_std = sanitized.phase_diff_std if sanitized.phase_diff_available else 0.0
            else:
                amplitude = sanitized.get('amplitude', np.zeros(n_subcarriers))
                phase = sanitized.get('phase', np.zeros(n_subcarriers))
                phase_diff_std = sanitized.get('phase_diff_std', 0.0)

            if self._nbvi_selector is not None:
                self._nbvi_selector.update_quiet_sample(link_key, amplitude, timestamp)

            self._update_baseline_if_quiet(
                link_key, amplitude, phase, phase_diff_std, timestamp
            )

    def _update_frequency_baselines(
        self,
        frame: CalibratedFrame,
        sanitized_items: Dict[Tuple[str, str], Union[SanitizedCSI, dict]],
        timestamp: float,
        amplitude_means: Optional[Dict[Tuple[str, str], float]] = None,
        n_subcarriers: int = 0
    ) -> None:
        """Update frequency-aware baselines for all links."""
        for link_key, sanitized in sanitized_items.items():
            if amplitude_means and link_key in amplitude_means:
                amplitude = amplitude_means[link_key]
            elif isinstance(sanitized, SanitizedCSI):
                amplitude = float(np.mean(sanitized.amplitude))
            else:
                amplitude = float(np.mean(sanitized.get('amplitude', np.zeros(n_subcarriers))))

            if isinstance(sanitized, SanitizedCSI):
                phase_diff_std = sanitized.phase_diff_std if sanitized.phase_diff_available else 0.0
            else:
                phase_diff_std = sanitized.get('phase_diff_std', 0.0)

            link_is_quiet = self._is_quiet and self._is_link_individually_quiet(frame, link_key)

            self._passive_frequency_collector.on_sample(
                link_key=link_key,
                amplitude=amplitude,
                phase_std=phase_diff_std,
                timestamp=timestamp,
                is_quiet=link_is_quiet
            )

            self._passive_time_collector.on_sample(
                link_key=link_key,
                amplitude=amplitude,
                phase_std=phase_diff_std,
                timestamp=timestamp,
                is_quiet=link_is_quiet
            )

    def _compute_calibration_maturity(self, frame: CalibratedFrame) -> float:
        """Compute system-wide calibration maturity [0, 1]."""
        if not frame.links:
            return 0.0

        maturity_sum = 0.0
        link_count = 0

        for link in frame.links.values():
            link_count += 1
            state = self._link_states.get((link.tx_mac, link.rx_mac))
            if state is None:
                continue

            sample_completeness = min(
                state.baseline.quiet_samples / MATURITY_MIN_SAMPLES, 1.0
            )

            health = 1.0 - getattr(link, 'baseline_gating_ratio', 0.0)

            maturity_sum += sample_completeness * (
                MATURITY_RELIABILITY_WEIGHT * link.reliability +
                MATURITY_HEALTH_WEIGHT * health
            )

        return maturity_sum / link_count

    def _update_frame_frequency_metadata(self, frame: CalibratedFrame) -> None:
        """Populate frequency calibration metadata fields."""
        frame.frequency_calibration_health = self._passive_frequency_collector.get_overall_health()
        gaps = self._passive_frequency_collector.get_calibration_gaps()
        all_uncalibrated: set = set()
        for bucket_list in gaps.values():
            all_uncalibrated.update(bucket_list)
        frame.uncalibrated_buckets = list(all_uncalibrated)
        frame.calibration_request_pending = self._opportunistic_calibrator.has_pending_request

    # =========================================================================
    # Public API
    # =========================================================================

    def provide_detection_feedback(
        self,
        position: Tuple[float, float],
        confidence: float
    ):
        """
        Application tells calibrator about a detection.

        Only call this for HIGH-CONFIDENCE detections (>0.9).

        Args:
            position: (x, y) where application believes someone is
            confidence: How confident the application is [0, 1]
        """
        if confidence < MIN_FEEDBACK_CONFIDENCE:
            return

        if self._current_frame is None:
            logger.warning("provide_detection_feedback called before any frame ingested")
            return

        link_responses = {}
        for link_key, link in self._current_frame.links.items():
            link_responses[link_key] = link.combined_change

            self.sensitivity_learner.on_detection_at_position(
                position=position,
                link_key=link_key,
                response_magnitude=link.combined_change
            )

        self.antenna_learner.on_consistent_detection(
            position=position,
            link_responses=link_responses
        )

    def get_link_reliability(self, tx_mac: str, rx_mac: str) -> float:
        """Get learned reliability score for a link."""
        link_key = (tx_mac, rx_mac)
        if link_key not in self._link_states:
            return 0.0

        state = self._link_states[link_key]
        sample_rate = self._estimate_sample_rate(state)
        rssi = int(GOOD_RSSI_DBM)
        snr = DEFAULT_SNR_ASSUMPTION

        return self._compute_reliability(
            state.baseline, snr, rssi, sample_rate, state
        )

    def get_effective_sensitivity(
        self,
        tx_mac: str,
        rx_mac: str,
        position: Tuple[float, float]
    ) -> float:
        """Get learned sensitivity of this link at a spatial position."""
        link_key = (tx_mac, rx_mac)

        sensitivity = self.sensitivity_learner.get_sensitivity(link_key, position)
        antenna_gain = self.antenna_learner.get_effective_gain(tx_mac, rx_mac, position)

        return sensitivity * min(antenna_gain, MAX_ANTENNA_GAIN_FACTOR)

    def get_baseline_drift(self, tx_mac: str, rx_mac: str) -> float:
        """Detect if environment has changed since calibration."""
        link_key = (tx_mac, rx_mac)
        if link_key not in self._link_states:
            return 0.0

        state = self._link_states[link_key]

        if len(state.recent_zscores) < 5:
            return 0.0

        recent = list(state.recent_zscores)
        mean_zscore = np.mean(recent)
        return float(mean_zscore)

    def get_antenna_pattern(self, node_mac: str) -> Optional[AntennaPattern]:
        """Get learned effective antenna pattern for a node."""
        return self.antenna_learner.get_pattern(node_mac)

    def get_all_links(self) -> List[Tuple[str, str]]:
        """Get all link keys tracked by calibrator."""
        return list(self._link_states.keys())

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics for debugging."""
        n_links = len(self._link_states)
        n_sensitivity_maps = len(getattr(self.sensitivity_learner, 'sensitivity_maps', {}))
        n_freq_states = len(getattr(self._passive_frequency_collector, '_link_states', {}))
        phase_corrector = getattr(self.sanitizer, 'phase_corrector', None)
        n_phase_histories = len(getattr(phase_corrector, '_link_phase_history', {}))

        est_per_link_kb = 15
        est_per_sensitivity_kb = 20

        return {
            'tracked_links': n_links,
            'sensitivity_maps': n_sensitivity_maps,
            'frequency_states': n_freq_states,
            'phase_histories': n_phase_histories,
            'max_tracked_links': MAX_TRACKED_LINKS,
            'estimated_memory_mb': (
                (n_links * est_per_link_kb +
                 n_sensitivity_maps * est_per_sensitivity_kb) / 1024
            ),
            'last_prune_time': self._last_prune_time,
            'prune_interval_s': LINK_PRUNE_INTERVAL_S,
            'stale_timeout_s': LINK_STALE_TIMEOUT_S,
        }

    def force_prune_links(self, max_age_s: float = None):
        """Force immediate pruning of stale links."""
        now = time.time()
        max_age = max_age_s if max_age_s is not None else LINK_STALE_TIMEOUT_S
        cutoff = now - max_age

        stale_links = [
            link_key for link_key, state in self._link_states.items()
            if state.last_activity_time < cutoff
        ]

        logger.info(f"Force pruning: found {len(stale_links)} stale links")

        for link_key in stale_links:
            del self._link_states[link_key]
            self._cleanup_link_related_structures(link_key)

        self._last_prune_time = now
        return len(stale_links)

    def reset_baseline(self, tx_mac: str = None, rx_mac: str = None):
        """Reset baseline for specific link or all links."""
        if tx_mac is not None and rx_mac is not None:
            link_key = (tx_mac, rx_mac)
            if link_key in self._link_states:
                self._link_states[link_key] = self._create_link_state(tx_mac, rx_mac)
                # Reset pluggable baseline manager for this link
                self.baseline_manager.reset(link_key)
                logger.info(f"Reset baseline for {tx_mac}->{rx_mac}")
        else:
            self._link_states.clear()
            # Reset all baselines in pluggable manager
            self.baseline_manager.reset()
            logger.info("Reset all baselines")

    def force_quiet_period(self, duration_s: float = 10.0):
        """Signal that the room is definitely empty."""
        self._force_quiet_until = time.time() + duration_s
        logger.info(f"Force quiet period for {duration_s}s")
        self._is_quiet = True

    def force_agc_warmup_complete(self) -> None:
        """Force AGC warmup to complete immediately."""
        if not self._agc_warmup_complete:
            self._agc_warmup_complete = True
            logger.info(f"AGC warmup forced complete at {self._agc_warmup_samples} samples")

    def get_frequency_calibration_request(self) -> Optional[CalibrationSweepRequest]:
        """Get pending frequency calibration request."""
        return self._opportunistic_calibrator.get_calibration_request()

    def mark_calibration_request_attempted(self, request_id: str = ""):
        """Mark that a calibration request was attempted."""
        self._opportunistic_calibrator.mark_request_attempted(request_id)

    def get_frequency_calibration_health(self) -> float:
        """Get overall frequency calibration health."""
        return self._passive_frequency_collector.get_overall_health()

    def update_node_position(self, node_mac: str, position: Tuple[float, float]):
        """Update position for a node."""
        old_pos = self.node_positions.get(node_mac)
        self.node_positions[node_mac] = position

        self.sensitivity_learner.update_node_position(node_mac, position)
        self.antenna_learner.update_node_position(node_mac, position)

        if old_pos is not None:
            dist = compute_distance(old_pos, position)
            if dist > SIGNIFICANT_POSITION_CHANGE_M:
                logger.info(
                    f"Node {node_mac} position changed by {dist:.2f}m"
                )

    def freeze_baseline(self, freeze: bool = True):
        """Freeze or unfreeze baseline updates."""
        was_frozen = self._baseline_frozen
        self._baseline_frozen = freeze

        # Also freeze the pluggable baseline manager
        self.baseline_manager.freeze(freeze)

        if freeze and not was_frozen:
            logger.info("AutoCalibrator: Baseline updates FROZEN")
        elif not freeze and was_frozen:
            logger.info("AutoCalibrator: Baseline updates UNFROZEN")

    def is_baseline_frozen(self) -> bool:
        """Check if baseline updates are currently frozen."""
        return self._baseline_frozen

    def get_baseline_zscore(
        self,
        tx_mac: str,
        rx_mac: str,
        amplitude: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Z-scores using the pluggable baseline manager.

        Args:
            tx_mac: Transmitter MAC address
            rx_mac: Receiver MAC address
            amplitude: Current amplitude array
            phase: Current phase array

        Returns:
            Tuple of (amplitude_zscore, phase_zscore)
        """
        link_key = (tx_mac, rx_mac)
        return self.baseline_manager.compute_zscore(link_key, amplitude, phase)

    def get_motion_detected(
        self,
        tx_mac: str,
        rx_mac: str,
        z_score: float,
        reliability: float = 1.0
    ) -> bool:
        """Check motion detection using the pluggable motion detector.

        Args:
            tx_mac: Transmitter MAC address
            rx_mac: Receiver MAC address
            z_score: Combined Z-score
            reliability: Link reliability

        Returns:
            True if motion detected
        """
        link_key = (tx_mac, rx_mac)
        return self.motion_detector.detect(
            z_score=z_score,
            reliability=reliability,
            link_key=link_key,
        )

    def get_pluggable_components(self) -> dict:
        """Get references to the pluggable components for advanced usage.

        Returns:
            Dict with keys: baseline_manager, reliability_calculator,
            motion_detector, quiet_detector
        """
        return {
            'baseline_manager': self.baseline_manager,
            'reliability_calculator': self.reliability_calculator,
            'motion_detector': self.motion_detector,
            'quiet_detector': self.quiet_detector,
        }

    def get_status(self) -> dict:
        """Get calibrator status for debugging/API."""
        calibrated_links = sum(
            1 for s in self._link_states.values()
            if s.baseline.is_initialized
        )

        freq_health = self._passive_frequency_collector.get_overall_health()
        freq_stats = self._passive_frequency_collector.get_stats()

        return {
            'total_links': len(self._link_states),
            'calibrated_links': calibrated_links,
            'is_quiet': self._is_quiet,
            'drift_detected': self._drift_detected,
            'baseline_frozen': self._baseline_frozen,
            'total_frames': self._total_frames_processed,
            'quiet_frames': self._quiet_frames,
            'active_frames': self._active_frames,
            'oldest_baseline_age_s': round(self._get_oldest_baseline_age(), 1),
            'last_quiet_time': self._last_quiet_time,
            'node_count': len(self.node_positions),
            'config': self.config.to_dict(),
            'sanitizer': self.sanitizer.get_stats(),
            'frequency_calibration': {
                'health': round(freq_health, 3),
                'samples_per_bucket': freq_stats.get('samples_per_bucket', {}),
                'total_samples': freq_stats.get('total_samples', 0),
            },
            'per_link_gating': {
                'enabled': self.config.per_link_activity_zscore_threshold > 0,
                'threshold': self.config.per_link_activity_zscore_threshold,
                'links_gated': {
                    str(k): v.baseline_updates_gated
                    for k, v in self._link_states.items()
                    if v.baseline_updates_gated > 0
                },
                'total_gated_events': sum(
                    s.baseline_updates_gated for s in self._link_states.values()
                ),
            },
            'nbvi_selection': {
                'enabled': self._nbvi_selector is not None,
                'stats': self._nbvi_selector.get_stats() if self._nbvi_selector else {},
            },
            'agc_warmup': {
                'complete': self._agc_warmup_complete,
                'samples': self._agc_warmup_samples,
                'config_samples': self.config.agc_warmup_samples,
                'config_timeout_s': self.config.agc_warmup_timeout_s,
            },
            'time_calibration': self._passive_time_collector.get_stats(),
            'pluggable_components': {
                'baseline_manager': type(self.baseline_manager).__name__,
                'baseline_stats': self.baseline_manager.get_stats() if hasattr(self.baseline_manager, 'get_stats') else {},
                'reliability_calculator': type(self.reliability_calculator).__name__,
                'motion_detector': type(self.motion_detector).__name__,
                'motion_threshold': getattr(self.motion_detector, 'threshold', None),
                'quiet_detector': type(self.quiet_detector).__name__,
            },
        }

    def get_link_status(self, tx_mac: str, rx_mac: str) -> Optional[dict]:
        """Get detailed status for a specific link."""
        link_key = (tx_mac, rx_mac)
        if link_key not in self._link_states:
            return None

        state = self._link_states[link_key]
        baseline = state.baseline

        freq_baseline = self._passive_frequency_collector.get_frequency_baseline(link_key)
        freq_info = {}
        if freq_baseline:
            freq_info = {
                'frequency_health': round(freq_baseline.get_calibration_health(), 3),
                'uncalibrated_buckets': [
                    bucket_to_name(b) for b in freq_baseline.get_uncalibrated_buckets()
                ],
            }

        return {
            'tx_mac': tx_mac,
            'rx_mac': rx_mac,
            'is_initialized': baseline.is_initialized,
            'quiet_samples': baseline.quiet_samples,
            'baseline_age_s': round(baseline.age_seconds, 1),
            'amplitude_mean': round(baseline.amplitude_mean_scalar, 3),
            'amplitude_std': round(baseline.amplitude_std_scalar, 3),
            'phase_std': round(baseline.phase_std_scalar, 3),
            'reliability': round(self.get_link_reliability(tx_mac, rx_mac), 3),
            'drift_sigma': round(self.get_baseline_drift(tx_mac, rx_mac), 3),
            'sensitivity_coverage': round(
                self.sensitivity_learner.get_coverage(link_key), 3
            ),
            'current_frequency_bucket': bucket_to_name(state.current_frequency_bucket),
            'current_frequency_hz': round(state.current_frequency_hz, 1),
            'frequency_calibration': freq_info,
        }

    # =========================================================================
    # Snapshot Export/Import (for external persistence)
    # =========================================================================

    def _reconstruct_welford_from_baseline(
        self, baseline: PerLinkBaseline
    ) -> Optional[WelfordAccumulator]:
        """Reconstruct WelfordAccumulator from baseline (delegates to persistence module)."""
        return _reconstruct_welford(baseline)

    def export_snapshot(self) -> CalibrationSnapshot:
        """Export portable calibration snapshot (JSON-serializable)."""
        return _export_snapshot(self)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CalibrationSnapshot,
        node_positions: Dict[str, Tuple[float, float]],
        config: Optional[AutoCalibratorConfig] = None,
        hardware_profile: Optional[dict] = None,
    ) -> 'AutoCalibrator':
        """Restore AutoCalibrator from a snapshot."""
        return _restore_from_snapshot(cls, snapshot, node_positions, config, hardware_profile)

    def get_link_confidence(
        self, tx_mac: str, rx_mac: str
    ) -> Optional[Tuple[float, float]]:
        """Get 95% confidence interval for link's amplitude mean."""
        return _get_link_confidence(self, tx_mac, rx_mac)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str, force: bool = False):
        """Persist learned calibration state with safe atomic writes."""
        _save_state(self, path, force)

    def _compact_baselines_for_save(self) -> dict:
        """Compact baseline data for efficient storage."""
        return _compact_baselines(self)

    @classmethod
    def _try_load_state(cls, path: Path) -> Optional[dict]:
        """Try to load state from a file, returning None if invalid."""
        return _try_load_state(path)

    @classmethod
    def load(
        cls,
        path: str,
        node_positions: Dict[str, Tuple[float, float]],
        config: Optional[AutoCalibratorConfig] = None,
        hardware_profile: Optional[dict] = None,
    ) -> 'AutoCalibrator':
        """Load previously learned calibration with backup recovery."""
        return _load_calibrator(cls, path, node_positions, config, hardware_profile)
