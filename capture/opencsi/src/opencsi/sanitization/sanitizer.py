"""
Internal CSI Sanitizer for OpenCSI AutoCalibrator.

Self-contained sanitization module with NO external dependencies.
This is part of the OpenCSI calibration package.

Implements essential CSI preprocessing:
1. AGC (Automatic Gain Control) correction
2. Phase correction (SFO, PDD, CFO removal)
3. Outlier removal (Hampel filter)
4. Time-differential phase computation

References:
- Samsung Research: "Optimal Preprocessing of WiFi CSI" (IEEE 2024)
- Tsinghua University: WiFi Sensing Tutorial
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from opencsi.stats.circular import circular_mean_and_std, circular_resultant_length

if TYPE_CHECKING:
    from opencsi.hardware.capabilities import HardwareCapabilities

logger = logging.getLogger(__name__)


# =============================================================================
# Module-Level Constants
# =============================================================================

# --- Numerical Stability ---
EPSILON = 1e-10

# --- Default Subcarrier Configuration (HT20 mode, backward compatibility) ---
# These are used when no HardwareCapabilities are provided
DEFAULT_NUM_SUBCARRIERS = 52
DEFAULT_PILOT_INDICES = (6, 20, 32, 46)  # HT20 pilot positions
DEFAULT_DC_SUBCARRIER_INDEX = 26  # DC subcarrier (center)

# --- AGC Correction ---
AGC_DEFAULT_GAIN = 1.0
AGC_HISTORY_SIZE = 100
AGC_GAIN_TABLE = {
    0: AGC_DEFAULT_GAIN,
    1: 1.2,
    2: 1.5,
    3: 2.0,
    4: 2.5,
    5: 3.0,
}

# --- Hampel Filter (Outlier Removal) ---
HAMPEL_WINDOW_DEFAULT = 5
HAMPEL_THRESHOLD_DEFAULT = 3.0
MAD_SCALE_FACTOR = 1.4826  # MAD to std conversion
# Fallback relative tolerance when MAD is zero
HAMPEL_ZERO_MAD_RELATIVE_TOLERANCE = 0.01

# --- Savitzky-Golay Smoothing ---
SAVGOL_WINDOW_DEFAULT = 3
SAVGOL_ORDER = 2

# --- DensePose-Style Phase Sanitization ---
# Reference: "DensePose From WiFi" (arXiv 2301.00250)
# Temporal median filter kernel size (odd number)
PHASE_TEMPORAL_MEDIAN_KERNEL = 5
# Uniform smoothing kernel size
PHASE_UNIFORM_SMOOTH_KERNEL = 3

# --- Edge Subcarriers ---
EDGE_SUBCARRIERS_TO_EXCLUDE = 2

# --- Quality Score Thresholds ---
QUALITY_MAX_OUTLIER_PENALTY = 0.5
QUALITY_OUTLIER_PENALTY_SCALE = 0.1
QUALITY_LOW_SNR_THRESHOLD = 10
QUALITY_MEDIUM_SNR_THRESHOLD = 20
QUALITY_LOW_SNR_PENALTY = 0.2
QUALITY_MEDIUM_SNR_PENALTY = 0.1

# --- Regression Fitting ---
# Use middle portion of subcarriers for robust linear fit
REGRESSION_START_FRACTION = 0.25
REGRESSION_END_FRACTION = 0.75


class SanitizationLevel(Enum):
    """Sanitization intensity levels."""
    NONE = 0        # Raw CSI (for debugging)
    BASIC = 1       # AGC correction only
    STANDARD = 2    # AGC + phase linear error removal
    FULL = 3        # All corrections including outlier removal
    DENSEPOSE = 4   # DensePose-style: unwrap → median filter → detrend → smooth


@dataclass
class SanitizedCSI:
    """
    Sanitized CSI data ready for calibration.

    Contains both amplitude and phase information after
    error correction and noise reduction.
    """
    # Core data
    amplitude: np.ndarray           # (n_subcarriers,) - corrected amplitude
    phase: np.ndarray               # (n_subcarriers,) - sanitized phase in radians

    # Complex representation
    complex_csi: np.ndarray         # (n_subcarriers,) - complex H(f)

    # Metadata
    tx_mac: str
    rx_mac: str
    timestamp: float
    rssi: int
    noise_floor: int
    snr: float                      # Signal-to-noise ratio estimate

    # Quality metrics
    agc_correction_applied: bool = False
    phase_correction_applied: bool = False
    hamming_window_applied: bool = False
    outliers_removed: int = 0
    quality_score: float = 1.0      # 0-1, higher is better

    # Time-differential phase
    phase_diff: Optional[np.ndarray] = None       # Δφ(t) = φ(t) - φ(t-1)
    phase_diff_mean: float = 0.0                  # Mean of phase_diff
    phase_diff_std: float = 0.0                   # Std of phase_diff (circular)
    phase_diff_available: bool = False            # True if phase_diff computed

    @property
    def amplitude_db(self) -> np.ndarray:
        """Amplitude in dB scale."""
        return 20 * np.log10(self.amplitude + EPSILON)

    @property
    def phase_degrees(self) -> np.ndarray:
        """Phase in degrees."""
        return np.degrees(self.phase)

    def to_dict(self) -> dict:
        """Serialize for logging/storage."""
        return {
            'tx_mac': self.tx_mac,
            'rx_mac': self.rx_mac,
            'timestamp': self.timestamp,
            'rssi': self.rssi,
            'snr': self.snr,
            'amplitude': self.amplitude.tolist(),
            'phase': self.phase.tolist(),
            'quality_score': self.quality_score,
            'phase_diff_available': self.phase_diff_available,
            'phase_diff_std': self.phase_diff_std
        }


@dataclass
class SanitizerConfig:
    """
    Configuration for CSI sanitizer.

    Supports dynamic hardware configuration via hardware_capabilities.
    When hardware_capabilities is provided, pilot indices and subcarrier
    counts are queried dynamically based on bandwidth.

    Example:
        # Static configuration (backward compatible)
        config = SanitizerConfig()

        # Dynamic configuration with hardware capabilities
        from opencsi.hardware.esp32 import ESP32Extractor
        extractor = ESP32Extractor(wifi6=True)
        config = SanitizerConfig(hardware_capabilities=extractor.capabilities)
    """
    level: SanitizationLevel = SanitizationLevel.STANDARD

    # AGC correction
    enable_agc_correction: bool = True
    agc_reference_power: float = AGC_DEFAULT_GAIN
    use_rx_state_for_agc: bool = True

    # Phase correction
    enable_phase_correction: bool = True
    use_pilot_subcarriers: bool = True
    phase_unwrap: bool = True

    # Outlier removal
    enable_outlier_removal: bool = True
    hampel_window: int = HAMPEL_WINDOW_DEFAULT
    hampel_threshold: float = HAMPEL_THRESHOLD_DEFAULT

    # Temporal smoothing
    enable_temporal_smoothing: bool = False
    smoothing_window: int = SAVGOL_WINDOW_DEFAULT

    # Subcarrier selection
    exclude_edge_subcarriers: bool = True
    edge_subcarriers_to_exclude: int = EDGE_SUBCARRIERS_TO_EXCLUDE
    exclude_dc_subcarrier: bool = True

    # NBVI-based subcarrier selection
    enable_nbvi_selection: bool = False
    nbvi_n_subcarriers: int = 12
    nbvi_alpha: float = 0.5

    # Hamming windowing
    enable_hamming_window: bool = True

    # DensePose-style phase sanitization (arXiv 2301.00250)
    # When level=DENSEPOSE, uses: unwrap → median filter → linear detrend → smooth
    phase_temporal_median_kernel: int = PHASE_TEMPORAL_MEDIAN_KERNEL
    phase_uniform_smooth_kernel: int = PHASE_UNIFORM_SMOOTH_KERNEL

    # Hardware capabilities for dynamic configuration
    # When set, pilot indices and subcarrier counts are queried from capabilities
    hardware_capabilities: Optional['HardwareCapabilities'] = None

    def get_pilot_indices(self, bandwidth_mhz: int = 20) -> Tuple[int, ...]:
        """
        Get pilot subcarrier indices for a given bandwidth.

        If hardware_capabilities is set, queries it dynamically.
        Otherwise, returns the default HT20 pilot indices.

        Args:
            bandwidth_mhz: Channel bandwidth in MHz

        Returns:
            Tuple of pilot subcarrier indices
        """
        if self.hardware_capabilities is not None:
            try:
                return self.hardware_capabilities.get_pilot_indices(bandwidth_mhz)
            except KeyError:
                # Fall back to default if bandwidth not supported
                pass
        return DEFAULT_PILOT_INDICES

    def get_num_subcarriers(self, bandwidth_mhz: int = 20) -> int:
        """
        Get number of subcarriers for a given bandwidth.

        If hardware_capabilities is set, queries it dynamically.
        Otherwise, returns the default HT20 subcarrier count.

        Args:
            bandwidth_mhz: Channel bandwidth in MHz

        Returns:
            Number of subcarriers
        """
        if self.hardware_capabilities is not None:
            try:
                return self.hardware_capabilities.get_num_subcarriers(bandwidth_mhz)
            except KeyError:
                pass
        return DEFAULT_NUM_SUBCARRIERS

    def get_dc_index(self, bandwidth_mhz: int = 20) -> Optional[int]:
        """
        Get DC subcarrier index for a given bandwidth.

        Args:
            bandwidth_mhz: Channel bandwidth in MHz

        Returns:
            DC subcarrier index or None
        """
        if self.hardware_capabilities is not None:
            try:
                return self.hardware_capabilities.get_dc_index(bandwidth_mhz)
            except KeyError:
                pass
        return DEFAULT_DC_SUBCARRIER_INDEX


class AGCCorrector:
    """
    Corrects Automatic Gain Control (AGC) variations.

    AGC causes random per-packet amplitude scaling that masks
    the actual channel variations we want to sense.
    """

    def __init__(self, config: SanitizerConfig):
        self.config = config
        self._power_history: deque = deque(maxlen=AGC_HISTORY_SIZE)

    def correct(
        self,
        amplitude: np.ndarray,
        rx_state: int = 0,
        rssi: int = -50
    ) -> Tuple[np.ndarray, float]:
        """
        Apply AGC correction to amplitude data.

        Args:
            amplitude: Raw amplitude (n_subcarriers,)
            rx_state: Firmware AGC/antenna state indicator
            rssi: Received signal strength

        Returns:
            (corrected_amplitude, estimated_gain)
        """
        if not self.config.enable_agc_correction:
            return amplitude, 1.0

        # Method 1: Use rx_state if available and configured
        if self.config.use_rx_state_for_agc and rx_state > 0:
            gain = self._estimate_gain_from_rx_state(rx_state)
            corrected = amplitude / gain
            return corrected, gain

        # Method 2: Power normalization (baseline method)
        total_power = np.sum(amplitude ** 2)
        if total_power > 0:
            gain = np.sqrt(total_power / (len(amplitude) * self.config.agc_reference_power))
            corrected = amplitude / gain
        else:
            corrected = amplitude
            gain = 1.0

        # Track power history
        self._power_history.append(total_power)

        return corrected, gain

    def _estimate_gain_from_rx_state(self, rx_state: int) -> float:
        """Estimate gain from rx_state field."""
        return AGC_GAIN_TABLE.get(rx_state, AGC_DEFAULT_GAIN)


class PhaseCorrector:
    """
    Corrects phase errors in CSI data.

    Phase errors include:
    1. SFO (Sampling Frequency Offset) - linear slope across subcarriers
    2. PDD (Packet Detection Delay) - constant offset
    3. CFO (Central Frequency Offset) - common phase error

    Also supports time-differential phase (Δφ) computation.

    Supports dynamic pilot indices via SanitizerConfig.hardware_capabilities.
    """

    def __init__(self, config: SanitizerConfig):
        self.config = config
        self._last_phase: Optional[np.ndarray] = None
        # Per-link previous phase for time-differential computation
        self._link_phase_history: Dict[Tuple[str, str], np.ndarray] = {}

    def correct(
        self,
        phase: np.ndarray,
        pilot_phases: Optional[np.ndarray] = None,
        bandwidth_mhz: int = 20
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply phase corrections.

        Args:
            phase: Raw phase (n_subcarriers,) in radians
            pilot_phases: Pilot subcarrier I/Q values (n_pilots, 2) if available
            bandwidth_mhz: Channel bandwidth for pilot index lookup

        Returns:
            (corrected_phase, correction_info)
        """
        if not self.config.enable_phase_correction:
            return phase, {'applied': False}

        correction_info = {'applied': True}
        corrected = phase.copy()

        # Step 1: Unwrap phase (remove 2π discontinuities)
        # Uses fast_unwrap to avoid np.unwrap dispatch overhead on small arrays
        if self.config.phase_unwrap:
            from opencsi.core.fast_math import fast_unwrap
            corrected = fast_unwrap(corrected)
            correction_info['unwrapped'] = True

        # Step 2: Remove linear phase error (SFO + PDD)
        if self.config.use_pilot_subcarriers and pilot_phases is not None:
            # Get pilot indices for this bandwidth
            pilot_indices = self.config.get_pilot_indices(bandwidth_mhz)
            corrected, slope, offset = self._remove_linear_error_pilots(
                corrected, pilot_phases, pilot_indices
            )
            correction_info['slope_removed'] = float(slope)
            correction_info['offset_removed'] = float(offset)
        else:
            # Fallback: linear regression on all subcarriers
            corrected, slope, offset = self._remove_linear_error_regression(corrected)
            correction_info['slope_removed'] = float(slope)
            correction_info['offset_removed'] = float(offset)

        # Step 3: Remove residual CFO (common phase error)
        # Use fast_median to avoid np.median dispatch overhead on small arrays
        from opencsi.core.fast_math import fast_median
        cfo_estimate = fast_median(corrected)
        corrected = corrected - cfo_estimate
        correction_info['cfo_removed'] = float(cfo_estimate)

        # Wrap back to [-π, π]
        corrected = np.mod(corrected + np.pi, 2 * np.pi) - np.pi

        self._last_phase = corrected
        return corrected, correction_info

    def _remove_linear_error_pilots(
        self,
        phase: np.ndarray,
        pilot_phases: np.ndarray,
        pilot_indices: Tuple[int, ...] = DEFAULT_PILOT_INDICES
    ) -> Tuple[np.ndarray, float, float]:
        """
        Remove linear phase error using pilot subcarriers.

        Args:
            phase: Unwrapped phase array (n_subcarriers,)
            pilot_phases: Pilot I/Q values (n_pilots, 2)
            pilot_indices: Pilot subcarrier positions (dynamic based on bandwidth)

        Returns:
            (corrected_phase, slope, offset)
        """
        # Extract pilot phases from I/Q
        pilot_phase_values = np.arctan2(pilot_phases[:, 1], pilot_phases[:, 0])
        pilot_phase_values = np.unwrap(pilot_phase_values)

        # Use dynamic pilot indices
        pilot_idx_array = np.array(pilot_indices)

        # Validate pilot count matches
        if len(pilot_phase_values) != len(pilot_idx_array):
            # Fall back to regression if mismatch
            return self._remove_linear_error_regression(phase)

        # Least squares fit through pilot positions
        A = np.vstack([pilot_idx_array, np.ones(len(pilot_idx_array))]).T
        slope, offset = np.linalg.lstsq(A, pilot_phase_values, rcond=None)[0]

        # Remove linear component from all subcarriers
        n_subcarriers = len(phase)
        subcarrier_indices = np.arange(n_subcarriers)
        linear_component = slope * subcarrier_indices + offset
        corrected = phase - linear_component

        return corrected, slope, offset

    def _remove_linear_error_regression(
        self,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Remove linear phase error using regression on middle subcarriers.

        Uses closed-form linear regression instead of np.linalg.lstsq
        for ~10× speedup on 52-element arrays.
        """
        n = len(phase)

        if n < 4:
            return phase.copy(), 0.0, 0.0

        # Robust linear fit (use middle subcarriers to avoid edge effects)
        mid_start = int(n * REGRESSION_START_FRACTION)
        mid_end = int(n * REGRESSION_END_FRACTION)

        x = np.arange(mid_start, mid_end, dtype=np.float64)
        y = phase[mid_start:mid_end]

        # Closed-form: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        m = len(x)
        sx = float(x.sum())
        sy = float(y.sum())
        sxy = float((x * y).sum())
        sxx = float((x * x).sum())
        denom = m * sxx - sx * sx
        if abs(denom) < 1e-12:
            return phase.copy(), 0.0, sy / m if m > 0 else 0.0
        slope = (m * sxy - sx * sy) / denom
        offset = (sy - slope * sx) / m

        # Remove linear component
        subcarrier_indices = np.arange(n, dtype=np.float64)
        corrected = phase - (slope * subcarrier_indices + offset)

        return corrected, slope, offset

    def correct_densepose_style(
        self,
        phase: np.ndarray,
        median_kernel: int = PHASE_TEMPORAL_MEDIAN_KERNEL,
        smooth_kernel: int = PHASE_UNIFORM_SMOOTH_KERNEL
    ) -> Tuple[np.ndarray, dict]:
        """
        DensePose-style phase sanitization (arXiv 2301.00250).

        Four-step process proven effective for WiFi-based pose estimation:
        1. Phase unwrapping: Correct ±π discontinuities
        2. Temporal median filter: Remove outliers (across subcarriers)
        3. Linear detrending: Remove systematic drift (SFO/PDD)
        4. Uniform smoothing: Reduce high-frequency noise

        Reference: "DensePose From WiFi" Section on Phase Sanitization
        - α₁ = (Φ_F - Φ_1) / (2πF) where F = num subcarriers
        - α₀ = (1/F) Σ ϕ_f
        - ϕ̂_f = ϕ_f - (α₁·f + α₀)

        Args:
            phase: Raw phase array (n_subcarriers,) in radians
            median_kernel: Kernel size for median filter (odd number)
            smooth_kernel: Kernel size for uniform smoothing (odd number)

        Returns:
            Tuple of (sanitized_phase, correction_info)
        """
        from scipy.ndimage import median_filter, uniform_filter1d

        correction_info = {
            'method': 'densepose',
            'steps_applied': []
        }

        n_subcarriers = len(phase)
        corrected = phase.copy()

        # Step 1: Phase unwrapping (correct ±π discontinuities)
        corrected = np.unwrap(corrected)
        correction_info['steps_applied'].append('unwrap')

        # Step 2: Median filter (remove outliers across subcarriers)
        # Ensures kernel size is odd and at least 3
        kernel = max(3, median_kernel | 1)
        corrected = median_filter(corrected, size=kernel, mode='reflect')
        correction_info['steps_applied'].append(f'median_filter(k={kernel})')

        # Step 3: Linear detrending (DensePose formula)
        # α₁ = (Φ_F - Φ_1) / (F - 1)   (phase slope in radians/subcarrier)
        # α₀ = mean(Φ) - α₁ · mean(f)  (intercept ensuring zero-mean residual)
        F = n_subcarriers
        alpha_1 = (corrected[-1] - corrected[0]) / max(F - 1, 1)
        alpha_0 = np.mean(corrected) - alpha_1 * (F - 1) / 2.0

        # Apply detrending: ϕ̂_f = ϕ_f - (α₁·f + α₀)
        subcarrier_indices = np.arange(n_subcarriers)
        linear_component = alpha_1 * subcarrier_indices + alpha_0
        corrected = corrected - linear_component

        correction_info['alpha_1'] = float(alpha_1)
        correction_info['alpha_0'] = float(alpha_0)
        correction_info['steps_applied'].append('linear_detrend')

        # Step 4: Uniform smoothing (reduce high-frequency noise)
        smooth_k = max(3, smooth_kernel | 1)
        corrected = uniform_filter1d(corrected, size=smooth_k, mode='reflect')
        correction_info['steps_applied'].append(f'uniform_smooth(k={smooth_k})')

        # Wrap back to [-π, π] for consistency
        corrected = np.mod(corrected + np.pi, 2 * np.pi) - np.pi

        self._last_phase = corrected
        return corrected, correction_info

    def compute_phase_diff(
        self,
        corrected_phase: np.ndarray,
        tx_mac: str,
        rx_mac: str
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Compute time-differential phase Δφ = φ(t) - φ(t-1).

        Args:
            corrected_phase: Current corrected phase (n_subcarriers,)
            tx_mac: Transmitter MAC address
            rx_mac: Receiver MAC address

        Returns:
            Tuple of (phase_diff array or None, stats dict)
        """
        link_key = (tx_mac, rx_mac)
        stats = {'has_previous': False, 'phase_diff_mean': 0.0, 'phase_diff_std': 0.0}

        # Check if we have previous phase for this link
        if link_key in self._link_phase_history:
            prev_phase = self._link_phase_history[link_key]

            if len(prev_phase) == len(corrected_phase):
                # Compute phase difference
                phase_diff = corrected_phase - prev_phase

                # Wrap to [-π, π] to handle wrap-around
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

                # Use fast circular statistics (avoids np.mean dispatch overhead)
                from opencsi.core.fast_math import fast_circular_mean_std
                mean_val, std_val = fast_circular_mean_std(phase_diff)

                stats['has_previous'] = True
                stats['phase_diff_mean'] = float(mean_val)
                stats['phase_diff_std'] = float(std_val)

                # Update history and return
                self._link_phase_history[link_key] = corrected_phase.copy()
                return phase_diff, stats

        # No previous phase - store current and return None
        self._link_phase_history[link_key] = corrected_phase.copy()
        return None, stats

    def get_phase_diff_circular_stats(
        self,
        phase_diff: np.ndarray
    ) -> dict:
        """Compute circular statistics of phase difference."""
        if phase_diff is None or len(phase_diff) == 0:
            return {
                'circular_mean': 0.0,
                'circular_std': 0.0,
                'resultant_length': 0.0
            }

        mean_val, std_val = circular_mean_and_std(phase_diff)
        R = circular_resultant_length(phase_diff)

        return {
            'circular_mean': mean_val,
            'circular_std': std_val,
            'resultant_length': R
        }

    def clear_link_history(self, tx_mac: str = None, rx_mac: str = None):
        """Clear phase history for a specific link or all links."""
        if tx_mac is not None and rx_mac is not None:
            link_key = (tx_mac, rx_mac)
            self._link_phase_history.pop(link_key, None)
        else:
            self._link_phase_history.clear()


class OutlierRemover:
    """
    Removes outliers and noise from CSI data.

    Uses Hampel filter: robust outlier detection based on
    Median Absolute Deviation (MAD).
    """

    def __init__(self, config: SanitizerConfig):
        self.config = config

    def remove_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply Hampel filter to remove outliers.

        Args:
            data: 1D array of values (e.g., amplitude per subcarrier)

        Returns:
            Tuple of (filtered_data, number_of_outliers_replaced)
        """
        if not self.config.enable_outlier_removal:
            return data, 0

        filtered = data.copy()
        n_outliers = 0

        window = self.config.hampel_window
        threshold = self.config.hampel_threshold

        n_samples = len(data)
        kernel_size = 2 * window + 1

        # Use efficient vectorized computation for the bulk of the data
        # Indices where the window is fully within bounds
        if n_samples >= kernel_size:
            # Create sliding window view: (n_valid_windows, kernel_size)
            # This covers indices from `window` to `n_samples - window - 1`
            v = sliding_window_view(data, window_shape=kernel_size)

            # Calculate medians for each window
            medians = np.median(v, axis=1)

            # Calculate MAD for each window
            # MAD = median(|x_i - median(X)|)
            abs_devs = np.abs(v - medians[:, np.newaxis])
            mads = np.median(abs_devs, axis=1)

            # Scale MAD to estimate std
            sigmas = MAD_SCALE_FACTOR * mads

            # Get the center data points corresponding to these windows
            # The center of the window starting at i is at index i + window
            center_indices = np.arange(window, n_samples - window)
            center_data = data[center_indices]

            diffs = np.abs(center_data - medians)

            # Vectorized outlier check
            # Case 1: sigma != 0
            mask_nz = sigmas != 0
            outlier_mask = np.zeros(len(center_indices), dtype=bool)
            outlier_mask[mask_nz] = diffs[mask_nz] > threshold * sigmas[mask_nz]

            # Case 2: sigma == 0 (fallback relative tolerance)
            mask_z = sigmas == 0
            if np.any(mask_z):
                min_sigmas = np.maximum(
                    np.abs(medians[mask_z]) * HAMPEL_ZERO_MAD_RELATIVE_TOLERANCE,
                    EPSILON
                )
                outlier_mask[mask_z] = diffs[mask_z] > threshold * min_sigmas

            # Apply corrections
            filtered[center_indices[outlier_mask]] = medians[outlier_mask]
            n_outliers += np.sum(outlier_mask)

            # Indices processed by vectorization
            processed_indices = set(center_indices)
        else:
            processed_indices = set()

        # Handle edges (and small arrays) with the original loop to preserve exact behavior
        # Left edge: 0 to window - 1
        # Right edge: n_samples - window to n_samples - 1
        # Or all indices if array is too small
        edge_indices = [i for i in range(n_samples) if i not in processed_indices]

        for i in edge_indices:
            # Window bounds (shrinking at edges)
            start = max(0, i - window)
            end = min(n_samples, i + window + 1)

            # Local median and MAD
            local_data = data[start:end]
            median = np.median(local_data)
            mad = np.median(np.abs(local_data - median))

            # Scale MAD to estimate std
            sigma = MAD_SCALE_FACTOR * mad

            # Check if outlier
            if sigma == 0:
                # Fallback for zero variance: use relative tolerance
                min_sigma = max(abs(median) * HAMPEL_ZERO_MAD_RELATIVE_TOLERANCE, EPSILON)
                if np.abs(data[i] - median) > threshold * min_sigma:
                    filtered[i] = median
                    n_outliers += 1
            elif np.abs(data[i] - median) > threshold * sigma:
                filtered[i] = median
                n_outliers += 1

        return filtered, n_outliers


# Cache hamming windows by size — np.hamming() creates a new array
# every call, wasting ~5ms/frame on 780 identical 52-element windows.
_hamming_cache: Dict[int, np.ndarray] = {}


def apply_hamming_window(data: np.ndarray) -> np.ndarray:
    """
    Apply Hamming window across subcarriers to reduce spectral leakage.

    Args:
        data: CSI data array (n_subcarriers,) or (n_links, n_subcarriers)

    Returns:
        Windowed data with same shape as input
    """
    if data.ndim == 1:
        n = len(data)
        if n not in _hamming_cache:
            _hamming_cache[n] = np.hamming(n)
        return data * _hamming_cache[n]
    elif data.ndim == 2:
        n = data.shape[-1]
        if n not in _hamming_cache:
            _hamming_cache[n] = np.hamming(n)
        return data * _hamming_cache[n][np.newaxis, :]
    else:
        return data


class CSISanitizer:
    """
    Complete CSI sanitization pipeline for OpenCSI AutoCalibrator.

    Self-contained with NO external dependencies.
    """

    def __init__(self, config: Optional[SanitizerConfig] = None):
        """Initialize sanitizer with configuration."""
        self.config = config or SanitizerConfig()
        self.agc_corrector = AGCCorrector(self.config)
        self.phase_corrector = PhaseCorrector(self.config)
        self.outlier_remover = OutlierRemover(self.config)

        # Statistics
        self._total_sanitized = 0
        self._total_outliers_removed = 0

    def sanitize(
        self,
        iq_data: np.ndarray,
        tx_mac: str,
        rx_mac: str,
        timestamp: float,
        rssi: int = -50,
        noise_floor: int = -90,
        rx_state: int = 0,
        pilot_phases: Optional[np.ndarray] = None,
        bandwidth_mhz: int = 20
    ) -> SanitizedCSI:
        """
        Complete sanitization pipeline.

        Args:
            iq_data: Raw I/Q data (n_subcarriers, 2) or complex (n_subcarriers,)
            tx_mac: Transmitter MAC address
            rx_mac: Receiver MAC address
            timestamp: Packet timestamp
            rssi: Received signal strength indicator
            noise_floor: Estimated noise floor
            rx_state: Firmware AGC/antenna state
            pilot_phases: Pilot subcarrier I/Q values (n_pilots, 2)
            bandwidth_mhz: Channel bandwidth for dynamic pilot lookup

        Returns:
            SanitizedCSI object with corrected data
        """
        # Convert to complex if needed
        if iq_data.ndim == 2 and iq_data.shape[1] == 2:
            complex_csi = iq_data[:, 0] + 1j * iq_data[:, 1]
        elif np.iscomplexobj(iq_data):
            complex_csi = iq_data
        else:
            # Assume real amplitude data
            complex_csi = iq_data.astype(complex)

        # Extract amplitude and phase
        amplitude = np.abs(complex_csi)
        phase = np.angle(complex_csi)

        # Track corrections applied
        agc_applied = False
        phase_applied = False
        hamming_applied = False
        outliers_removed = 0

        # Step 1: AGC correction (all levels except NONE)
        if self.config.level.value >= SanitizationLevel.BASIC.value:
            amplitude, _ = self.agc_corrector.correct(amplitude, rx_state, rssi)
            agc_applied = True

        # Step 2: Phase correction
        if self.config.level == SanitizationLevel.DENSEPOSE:
            # DensePose-style: unwrap → median filter → linear detrend → smooth
            phase, _ = self.phase_corrector.correct_densepose_style(
                phase,
                median_kernel=self.config.phase_temporal_median_kernel,
                smooth_kernel=self.config.phase_uniform_smooth_kernel
            )
            phase_applied = True
        elif self.config.level.value >= SanitizationLevel.STANDARD.value:
            # Standard: pilot-based or regression linear error removal
            phase, _ = self.phase_corrector.correct(phase, pilot_phases, bandwidth_mhz)
            phase_applied = True

        # Step 3: Outlier removal (FULL level or DENSEPOSE)
        if self.config.level.value >= SanitizationLevel.FULL.value:
            amplitude, outliers_removed = self.outlier_remover.remove_outliers(amplitude)
            self._total_outliers_removed += outliers_removed

        # Step 4: Hamming windowing
        if self.config.enable_hamming_window:
            amplitude = apply_hamming_window(amplitude)
            hamming_applied = True

        # Compute time-differential phase
        phase_diff, diff_stats = self.phase_corrector.compute_phase_diff(
            phase, tx_mac, rx_mac
        )

        # Reconstruct complex CSI
        complex_csi = amplitude * np.exp(1j * phase)

        # Estimate SNR
        snr = float(rssi - noise_floor)

        # Compute quality score
        quality_score = self._compute_quality_score(
            outliers_removed, snr, len(amplitude)
        )

        self._total_sanitized += 1

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
            agc_correction_applied=agc_applied,
            phase_correction_applied=phase_applied,
            hamming_window_applied=hamming_applied,
            outliers_removed=outliers_removed,
            quality_score=quality_score,
            phase_diff=phase_diff,
            phase_diff_mean=diff_stats['phase_diff_mean'],
            phase_diff_std=diff_stats['phase_diff_std'],
            phase_diff_available=diff_stats['has_previous']
        )

    def _compute_quality_score(
        self,
        outliers_removed: int,
        snr: float,
        n_subcarriers: int
    ) -> float:
        """Compute overall quality score [0, 1]."""
        # Start with perfect score
        score = 1.0

        # Penalty for outliers
        outlier_fraction = outliers_removed / max(n_subcarriers, 1)
        outlier_penalty = min(
            QUALITY_MAX_OUTLIER_PENALTY,
            outlier_fraction * QUALITY_OUTLIER_PENALTY_SCALE
        )
        score -= outlier_penalty

        # Penalty for low SNR
        if snr < QUALITY_LOW_SNR_THRESHOLD:
            score -= QUALITY_LOW_SNR_PENALTY
        elif snr < QUALITY_MEDIUM_SNR_THRESHOLD:
            score -= QUALITY_MEDIUM_SNR_PENALTY

        return max(0.0, min(1.0, score))

    def get_stats(self) -> dict:
        """Get sanitizer statistics."""
        return {
            'total_sanitized': self._total_sanitized,
            'total_outliers_removed': self._total_outliers_removed,
            'config': {
                'level': self.config.level.name,
                'agc_correction': self.config.enable_agc_correction,
                'phase_correction': self.config.enable_phase_correction,
                'outlier_removal': self.config.enable_outlier_removal,
                'hamming_window': self.config.enable_hamming_window
            }
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._total_sanitized = 0
        self._total_outliers_removed = 0

    def clear_link_history(self, tx_mac: str = None, rx_mac: str = None):
        """Clear phase history for time-differential computation."""
        self.phase_corrector.clear_link_history(tx_mac, rx_mac)
