"""
Core data types for OpenCSI calibration system.

This module defines the output data structures for the calibration layer:
- CalibratedLink: Per-link normalized measurements with reliability metadata
- CalibratedFrame: Collection of calibrated links with frame-level metadata
- PerLinkBaseline: Learned baseline statistics for one link
- RawCSIPacket: Input format for calibrator
"""

import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from opencsi.core.config import (
    WEAK_LINK_DETECTION_WIDTH_M,
    RELIABLE_LINK_THRESHOLD,
    WAVELENGTH_2_4GHZ_M,
)

# =============================================================================
# Quality Grade Thresholds
# =============================================================================
# These thresholds define the quality grade boundaries based on reliability and SNR.
# Adjusting these affects how links are classified for weighting and filtering.

# Maximum baseline_age_s value for numpy export (avoids inf in ML pipelines)
MAX_BASELINE_AGE_FOR_NUMPY_S = 1e6  # ~11.5 days

QUALITY_GRADE_A_RELIABILITY = 0.8
QUALITY_GRADE_A_SNR_DB = 25.0
QUALITY_GRADE_B_RELIABILITY = 0.6
QUALITY_GRADE_B_SNR_DB = 15.0
QUALITY_GRADE_C_RELIABILITY = 0.3
QUALITY_GRADE_C_SNR_DB = 5.0

# Default subcarrier count for backward compatibility (ESP32 802.11n HT20)
DEFAULT_N_SUBCARRIERS = 52


# Fields exported by CalibratedLink.to_numpy() — order defines column indices
CALIBRATED_LINK_NUMPY_FIELDS = [
    'amplitude_change', 'phase_change', 'combined_change',
    'reliability', 'snr_estimate', 'baseline_age_s',
    'amplitude_mean', 'phase_diff_std', 'baseline_samples',
    'baseline_gating_ratio',
    'link_distance_m', 'fresnel_radius_m', 'effective_detection_width_m',
    'snr_scaling_factor',
    'frequency_hz', 'bucket_confidence',
]


class QualityGrade(Enum):
    """Link quality grades based on reliability and SNR."""
    A = "A"  # Excellent: high SNR, stable baseline, low CV
    B = "B"  # Good: decent SNR, stable baseline
    C = "C"  # Fair: usable but noisy or aging baseline
    F = "F"  # Poor: unreliable, stale baseline, or low SNR

    @classmethod
    def from_reliability(cls, reliability: float, snr: float) -> 'QualityGrade':
        """
        Assign quality grade based on reliability and SNR.

        Args:
            reliability: Link reliability score [0, 1]
            snr: Signal-to-noise ratio in dB

        Returns:
            QualityGrade enum value
        """
        if reliability >= QUALITY_GRADE_A_RELIABILITY and snr >= QUALITY_GRADE_A_SNR_DB:
            return cls.A
        elif reliability >= QUALITY_GRADE_B_RELIABILITY and snr >= QUALITY_GRADE_B_SNR_DB:
            return cls.B
        elif reliability >= QUALITY_GRADE_C_RELIABILITY and snr >= QUALITY_GRADE_C_SNR_DB:
            return cls.C
        else:
            return cls.F


class LearningMode(Enum):
    """
    Calibrator learning mode.

    The system progresses through these modes as it gathers data:
    - BLIND: Day 0, uses theoretical Fresnel zones
    - SUPERVISED: Optional user-guided calibration walk
    - ONLINE: Continuous refinement from high-confidence detections
    """
    BLIND = 0       # Uses theoretical isotropic Fresnel, hardware profile defaults
    SUPERVISED = 1  # User walking calibration path (optional)
    ONLINE = 2      # Automatic refinement from detection feedback


@dataclass
class NetworkState:
    """
    Network health state for complete reliability calculation.

    This information comes from the scheduler and network layer.
    Applications should inject this periodically via update_network_state().

    The complete reliability formula is:
    R = (1/(1+CV)) × e^(-λ×t_age) × Q_rate × Q_rssi × Q_latency × Q_load × Q_interference
    """
    # Per-link latency metrics
    link_latencies_ms: Dict[Tuple[str, str], float] = field(default_factory=dict)
    """Round-trip latency per link in milliseconds."""

    link_jitter_ms: Dict[Tuple[str, str], float] = field(default_factory=dict)
    """Latency jitter (std) per link in milliseconds."""

    # Network load metrics
    network_load: float = 0.0
    """Overall network utilization [0, 1]. 0 = idle, 1 = saturated."""

    active_link_count: int = 0
    """Number of currently active links in schedule."""

    # Per-link interference (Fresnel zone overlap)
    interference_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    """Interference score per link [0, 1]. 0 = isolated, 1 = heavily overlapped."""

    # Timestamp
    updated_at: float = 0.0
    """When this state was last updated (Unix timestamp)."""

    def get_latency_quality(self, link_key: Tuple[str, str]) -> float:
        """
        Compute Q_latency for a link.

        Returns 1.0 if link not in state (assume healthy).
        """
        if link_key not in self.link_latencies_ms:
            return 1.0

        latency = self.link_latencies_ms.get(link_key, 50.0)
        jitter = self.link_jitter_ms.get(link_key, 5.0)

        jitter_factor = max(0.0, 1.0 - jitter / 20.0)
        latency_factor = max(0.0, 1.0 - latency / 200.0)

        return jitter_factor * latency_factor

    def get_load_quality(self) -> float:
        """Compute Q_load based on network congestion."""
        return max(0.0, 1.0 - self.network_load ** 2)

    def get_interference_quality(self, link_key: Tuple[str, str]) -> float:
        """Compute Q_interference for a link."""
        if link_key not in self.interference_scores:
            return 1.0
        return max(0.0, 1.0 - self.interference_scores.get(link_key, 0.0))


@dataclass
class CalibratedLink:
    """
    Calibrated measurements for a single TX-RX link.

    All measurements are normalized to Z-scores (standard deviations
    from the learned baseline). This makes them hardware-agnostic
    and directly comparable across different links and hardware.
    """
    tx_mac: str
    rx_mac: str

    # Normalized measurements (Z-scores)
    amplitude_change: float = 0.0
    phase_change: float = 0.0
    combined_change: float = 0.0

    # Reliability metadata
    reliability: float = 0.0
    baseline_age_s: float = float('inf')
    snr_estimate: float = 0.0
    quality_grade: QualityGrade = QualityGrade.F

    # Raw values for debugging
    raw_amplitude: Optional[np.ndarray] = None
    raw_phase: Optional[np.ndarray] = None

    # Additional metrics
    amplitude_mean: float = 0.0
    phase_diff_std: float = 0.0
    baseline_samples: int = 0

    # Frequency-aware calibration metadata
    frequency_bucket: int = 2
    frequency_hz: float = 10.0
    calibration_interpolated: bool = False
    bucket_confidence: float = 1.0

    # Baseline gating metadata
    baseline_gating_ratio: float = 0.0

    # Effective detection metrics
    link_distance_m: float = 0.0
    fresnel_radius_m: float = 0.0
    effective_detection_width_m: float = 0.0
    snr_scaling_factor: float = 1.0

    # Frequency metadata for hardware-agnostic localization
    channel: int = 6  # WiFi channel number
    frequency_band: str = "2.4GHz"  # "2.4GHz", "5GHz", or "6GHz"
    wavelength_m: float = WAVELENGTH_2_4GHZ_M  # Signal wavelength in meters

    # Subcarrier-level ghost discrimination features
    # High coherence = spectral shape preserved = broadband scatterer (human)
    # Low coherence = spectral shape distorted = narrowband interference (ghost)
    subcarrier_coherence: float = 0.0
    # High flatness = many subcarriers affected = broadband (human)
    # Low flatness = few subcarriers affected = narrowband (ghost)
    amplitude_flatness: float = 0.0
    # Per-subcarrier Z-score vector — the spatial frequency signature.
    # Shape (n_subcarriers,). Each element is |amplitude_k - baseline_k| / std_k.
    # Preserves which subcarriers are perturbed (vs scalar mean that erases this).
    # Empty array when baseline is not initialized.
    per_subcarrier_zscore: Optional[np.ndarray] = None

    # Per-subcarrier phase difference Δφ(t) = φ(t) - φ(t-1).
    # Shape (n_subcarriers,). Carries per-delay-tap Doppler velocity.
    # Moving person → coherent phase shift. Static ghost → zero.
    per_subcarrier_phase_diff: Optional[np.ndarray] = None
    # Total Doppler energy: Σ(Δφ²). High = motion, low = static.
    doppler_energy: float = 0.0
    # Fraction of subcarriers with |Δφ| > threshold. High = broadband motion.
    doppler_sparsity: float = 0.0

    @property
    def link_key(self) -> Tuple[str, str]:
        """Return (tx_mac, rx_mac) tuple for dict keys."""
        return (self.tx_mac, self.rx_mac)

    @property
    def is_reliable(self) -> bool:
        """Quick check if link is reliable enough for detection."""
        return self.reliability >= 0.5 and self.quality_grade != QualityGrade.F

    @property
    def has_motion(self) -> bool:
        """Quick check if combined_change indicates motion (> 2σ)."""
        return self.combined_change > 2.0

    @property
    def is_weak_link(self) -> bool:
        """Check if link has limited detection utility."""
        return self.effective_detection_width_m < WEAK_LINK_DETECTION_WIDTH_M

    def to_numpy(self) -> np.ndarray:
        """Export scalar measurements as a 1D float64 array.

        Column order matches CALIBRATED_LINK_NUMPY_FIELDS.
        Use ``CalibratedLink.numpy_field_names()`` to get column labels.

        Infinite values (e.g. ``baseline_age_s`` before first update) are
        clamped to ``MAX_BASELINE_AGE_FOR_NUMPY_S`` so downstream ML
        pipelines never encounter inf/NaN.

        Returns:
            1D array of shape ``(n_features,)``
        """
        values = [getattr(self, f) for f in CALIBRATED_LINK_NUMPY_FIELDS]
        arr = np.array(values, dtype=np.float64)
        np.clip(arr, -MAX_BASELINE_AGE_FOR_NUMPY_S, MAX_BASELINE_AGE_FOR_NUMPY_S, out=arr)
        return arr

    @staticmethod
    def numpy_field_names() -> List[str]:
        """Return field names corresponding to ``to_numpy()`` columns."""
        return list(CALIBRATED_LINK_NUMPY_FIELDS)

    def to_dict(self) -> dict:
        """Serialize for logging/API."""
        return {
            'tx_mac': self.tx_mac,
            'rx_mac': self.rx_mac,
            'amplitude_change': round(self.amplitude_change, 3),
            'phase_change': round(self.phase_change, 3),
            'combined_change': round(self.combined_change, 3),
            'reliability': round(self.reliability, 3),
            'baseline_age_s': round(min(self.baseline_age_s, 1e9), 1),
            'snr_estimate': round(self.snr_estimate, 1),
            'quality_grade': self.quality_grade.value,
            'frequency_bucket': self.frequency_bucket,
            'frequency_hz': round(self.frequency_hz, 1),
            'calibration_interpolated': self.calibration_interpolated,
            'bucket_confidence': round(self.bucket_confidence, 3),
            'baseline_gating_ratio': round(self.baseline_gating_ratio, 3),
            'link_distance_m': round(self.link_distance_m, 3),
            'fresnel_radius_m': round(self.fresnel_radius_m, 3),
            'effective_detection_width_m': round(self.effective_detection_width_m, 3),
            'snr_scaling_factor': round(self.snr_scaling_factor, 3),
            'is_weak_link': self.is_weak_link,
            'channel': self.channel,
            'frequency_band': self.frequency_band,
            'wavelength_m': round(self.wavelength_m, 4),
        }


@dataclass
class CalibratedFrame:
    """
    Collection of calibrated links for a single time instant.

    This is the primary output of the calibration layer, containing
    all calibrated link measurements plus frame-level metadata.
    """
    timestamp: float = 0.0
    links: Dict[Tuple[str, str], CalibratedLink] = field(default_factory=dict)

    # Frame-level metadata
    calibration_age_s: float = float('inf')
    environment_stable: bool = False
    baseline_drift_detected: bool = False

    # Aggregate statistics
    mean_reliability: float = 0.0
    n_reliable_links: int = 0
    n_total_links: int = 0

    # Frequency-aware calibration
    frequency_calibration_health: float = 1.0
    uncalibrated_buckets: List[int] = field(default_factory=list)
    calibration_request_pending: bool = False

    # Calibration maturity
    calibration_maturity: float = 0.0

    # Frame-level Doppler aggregates (from per-subcarrier phase-diff)
    mean_doppler_energy: float = 0.0
    max_doppler_energy: float = 0.0

    # AGC warmup status
    agc_warmup_complete: bool = False

    def to_numpy(
        self,
        min_reliability: float = 0.0,
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Export all links as a 2D numpy array.

        Each row is a link's scalar measurements (see
        ``CalibratedLink.numpy_field_names()`` for column labels).

        Args:
            min_reliability: Only include links with reliability >= this.

        Returns:
            ``(data, link_keys)`` where ``data`` has shape
            ``(n_links, n_features)`` and ``link_keys[i]`` is the
            ``(tx_mac, rx_mac)`` for row *i*.
        """
        filtered = [
            (k, v) for k, v in self.links.items()
            if v.reliability >= min_reliability
        ]
        if not filtered:
            n_features = len(CALIBRATED_LINK_NUMPY_FIELDS)
            return np.empty((0, n_features), dtype=np.float64), []

        keys, vals = zip(*filtered)
        data = np.stack([v.to_numpy() for v in vals])
        return data, list(keys)

    def get_link(self, tx_mac: str, rx_mac: str) -> Optional[CalibratedLink]:
        """Get calibrated link by MAC addresses."""
        return self.links.get((tx_mac, rx_mac))

    def get_motion_links(
        self,
        threshold: float = 2.0,
        min_reliability: float = 0.5
    ) -> Dict[Tuple[str, str], CalibratedLink]:
        """Get links with motion above threshold."""
        return {
            key: link for key, link in self.links.items()
            if link.combined_change >= threshold and link.reliability >= min_reliability
        }

    def get_max_change(self, min_reliability: float = 0.3) -> float:
        """Get maximum combined_change across reliable links."""
        reliable_changes = [
            link.combined_change for link in self.links.values()
            if link.reliability >= min_reliability
        ]
        return max(reliable_changes) if reliable_changes else 0.0

    def get_mean_change(self, min_reliability: float = 0.3) -> float:
        """Get reliability-weighted mean of combined_change."""
        total_weight = 0.0
        weighted_sum = 0.0

        for link in self.links.values():
            if link.reliability >= min_reliability:
                weighted_sum += link.combined_change * link.reliability
                total_weight += link.reliability

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_reliable_links(
        self,
        min_reliability: float = RELIABLE_LINK_THRESHOLD
    ) -> Dict[Tuple[str, str], CalibratedLink]:
        """Get all links meeting reliability threshold."""
        return {
            key: link for key, link in self.links.items()
            if link.reliability >= min_reliability
        }

    @property
    def motion_detected(self) -> bool:
        """True if any reliable link shows Z-score above threshold."""
        return any(
            link.combined_change > 2.0 and link.reliability >= 0.3
            for link in self.links.values()
        )

    def get_subcarrier_ghost_features(
        self,
        min_reliability: float = 0.3,
        motion_threshold: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Get aggregate subcarrier coherence and amplitude flatness for motion links.

        These features discriminate ghosts (narrowband interference) from real
        human motion (broadband scatterer):
        - **Coherence**: spectral shape preserved → human. Shape distorted → ghost.
        - **Flatness**: many subcarriers affected → human. Few spikes → ghost.

        Args:
            min_reliability: Minimum link reliability to include
            motion_threshold: Minimum combined_change Z-score to include

        Returns:
            (mean_coherence, mean_flatness) across motion links.
            Both are 0.0 if no qualifying links exist.
        """
        coherences = []
        flatnesses = []
        for link in self.links.values():
            if link.reliability >= min_reliability and link.combined_change >= motion_threshold:
                coherences.append(link.subcarrier_coherence)
                flatnesses.append(link.amplitude_flatness)

        if not coherences:
            return 0.0, 0.0

        return float(np.mean(coherences)), float(np.mean(flatnesses))

    def update_aggregate_stats(self):
        """Recompute aggregate statistics from links."""
        self.n_total_links = len(self.links)

        if self.n_total_links == 0:
            self.mean_reliability = 0.0
            self.n_reliable_links = 0
            return

        reliabilities = [link.reliability for link in self.links.values()]
        self.mean_reliability = float(np.mean(reliabilities))
        self.n_reliable_links = sum(1 for r in reliabilities if r >= RELIABLE_LINK_THRESHOLD)

    def to_dict(self) -> dict:
        """Serialize for logging/API."""
        return {
            'timestamp': self.timestamp,
            'n_links': len(self.links),
            'n_reliable_links': self.n_reliable_links,
            'mean_reliability': round(self.mean_reliability, 3),
            'calibration_age_s': round(min(self.calibration_age_s, 1e9), 1),
            'environment_stable': self.environment_stable,
            'baseline_drift_detected': self.baseline_drift_detected,
            'max_change': round(self.get_max_change(), 3),
            'mean_change': round(self.get_mean_change(), 3),
            'motion_detected': self.motion_detected,
            'frequency_calibration_health': round(self.frequency_calibration_health, 3),
            'uncalibrated_buckets': self.uncalibrated_buckets,
            'calibration_request_pending': self.calibration_request_pending,
            'calibration_maturity': round(self.calibration_maturity, 3),
            'agc_warmup_complete': self.agc_warmup_complete,
            'links': {
                f"{k[0][-8:]}->{k[1][-8:]}": v.to_dict()
                for k, v in self.links.items()
            }
        }


@dataclass
class PerLinkBaseline:
    """
    Learned baseline statistics for one link.

    Supports dynamic subcarrier counts for multi-hardware compatibility.
    Arrays are lazily initialized on first use via ensure_arrays().

    Example:
        baseline = PerLinkBaseline(tx_mac="AA:...", rx_mac="BB:...")

        # Arrays auto-resize when processing packets with different subcarrier counts
        baseline.ensure_arrays(n_subcarriers=242)  # WiFi 6 HE20
    """
    tx_mac: str
    rx_mac: str

    # Amplitude statistics (per-subcarrier) - lazily initialized
    amplitude_mean: Optional[np.ndarray] = None
    amplitude_std: Optional[np.ndarray] = None

    # Phase statistics (from sanitized phase, circular) - lazily initialized
    phase_mean: Optional[np.ndarray] = None
    phase_std: Optional[np.ndarray] = None

    # Aggregate statistics
    amplitude_mean_scalar: float = 0.0
    amplitude_std_scalar: float = 1.0
    amplitude_variance_scalar: float = 1.0
    phase_std_scalar: float = 1.0

    # Metadata
    quiet_samples: int = 0
    last_update_time: float = 0.0
    creation_time: float = 0.0
    is_initialized: bool = False

    # Dynamic array sizing
    n_subcarriers: int = 0

    def __post_init__(self):
        """Initialize with default arrays for backward compatibility."""
        if self.amplitude_mean is None:
            self.ensure_arrays(DEFAULT_N_SUBCARRIERS)

    def ensure_arrays(self, n_subcarriers: int) -> None:
        """
        Ensure arrays are allocated for the given subcarrier count.

        If the subcarrier count changes (e.g., switching bandwidths),
        arrays are reallocated and the baseline is reset.

        Args:
            n_subcarriers: Number of subcarriers for this link
        """
        if self.n_subcarriers == n_subcarriers and self.amplitude_mean is not None:
            return  # Already correctly sized

        # Reallocate arrays
        self.amplitude_mean = np.zeros(n_subcarriers)
        self.amplitude_std = np.ones(n_subcarriers)
        self.phase_mean = np.zeros(n_subcarriers)
        self.phase_std = np.ones(n_subcarriers)
        self.n_subcarriers = n_subcarriers

        # Reset baseline if size changed
        if self.is_initialized:
            self.is_initialized = False
            self.quiet_samples = 0

    @property
    def age_seconds(self) -> float:
        """Seconds since last baseline update."""
        if self.last_update_time == 0:
            return float('inf')
        return _time.time() - self.last_update_time

    @property
    def link_key(self) -> Tuple[str, str]:
        """Return (tx_mac, rx_mac) tuple."""
        return (self.tx_mac, self.rx_mac)

    def compute_amplitude_zscore(self, current_amplitude: np.ndarray) -> float:
        """Compute Z-score for amplitude deviation."""
        if not self.is_initialized:
            return 0.0

        # Ensure arrays match incoming data
        if len(current_amplitude) != self.n_subcarriers:
            self.ensure_arrays(len(current_amplitude))
            return 0.0  # Need to re-learn baseline

        current_mean = float(current_amplitude.sum() / len(current_amplitude))
        if self.amplitude_std_scalar < 1e-6:
            return 0.0

        return abs(current_mean - self.amplitude_mean_scalar) / self.amplitude_std_scalar

    def compute_phase_zscore(self, phase_diff_std: float) -> float:
        """Compute Z-score for phase change."""
        if not self.is_initialized or self.phase_std_scalar < 1e-6:
            return 0.0

        effective_std = max(self.phase_std_scalar, 0.01)
        return max(0, phase_diff_std - self.phase_std_scalar) / effective_std


@dataclass
class RawCSIPacket:
    """
    Raw CSI packet for calibrator input.

    This is the expected input format for AutoCalibrator.ingest().
    Can be constructed from hardware-specific packets or raw I/Q data.
    """
    tx_mac: str
    rx_mac: str
    iq_data: np.ndarray  # (n_subcarriers, 2) - I/Q pairs
    timestamp: float
    rssi: int = -50
    noise_floor: int = -90
    rx_state: int = 0  # AGC/antenna state
    pilot_phases: Optional[np.ndarray] = None  # (4, 2) - pilot I/Q

    # Frequency metadata for hardware-agnostic Fresnel calculations
    channel: int = 6  # WiFi channel number (default: 2.4GHz channel 6)
    frequency_band: str = "2.4GHz"  # "2.4GHz", "5GHz", or "6GHz"

    @classmethod
    def from_csi_packet(cls, packet: 'CSIPacket') -> 'RawCSIPacket':
        """Construct from hardware CSIPacket."""
        return cls(
            tx_mac=packet.tx_mac,
            rx_mac=packet.rx_mac,
            iq_data=packet.iq_data,
            timestamp=packet.timestamp_us / 1e6 if hasattr(packet, 'timestamp_us') else 0.0,
            rssi=packet.rssi,
            noise_floor=packet.noise_floor,
            rx_state=packet.rx_state,
            pilot_phases=packet.pilot_phases,
            channel=getattr(packet, 'channel', 6),
            frequency_band=getattr(packet, 'frequency_band', "2.4GHz"),
        )


# =============================================================================
# Calibration Snapshot Types (for persistence)
# =============================================================================


@dataclass
class LinkSnapshot:
    """
    Point-in-time calibration state for one link.

    This is a portable, JSON-serializable representation of a link's
    calibration state. No numpy arrays - only primitive types.

    Used for:
    - Persistence to SQLite
    - Export/import across deployments
    - Trend analysis over time
    """
    tx_mac: str
    rx_mac: str
    timestamp: float

    # Amplitude statistics
    amplitude_mean: float
    amplitude_std: float
    amplitude_ci: Optional[Tuple[float, float]]  # 95% confidence interval

    # Phase statistics
    phase_std: float

    # Calibration state
    quiet_samples: int
    is_initialized: bool

    # Optional Welford accumulator state (for combining snapshots)
    welford_n: Optional[int] = None
    welford_mean: Optional[float] = None
    welford_m2: Optional[float] = None
    baseline_updates_gated: int = 0

    @property
    def link_key(self) -> str:
        """Get link key string for persistence."""
        return f"{self.tx_mac}->{self.rx_mac}"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'tx_mac': self.tx_mac,
            'rx_mac': self.rx_mac,
            'timestamp': self.timestamp,
            'amplitude_mean': self.amplitude_mean,
            'amplitude_std': self.amplitude_std,
            'amplitude_ci_low': self.amplitude_ci[0] if self.amplitude_ci else None,
            'amplitude_ci_high': self.amplitude_ci[1] if self.amplitude_ci else None,
            'phase_std': self.phase_std,
            'quiet_samples': self.quiet_samples,
            'is_initialized': self.is_initialized,
            'welford_n': self.welford_n,
            'welford_mean': self.welford_mean,
            'welford_m2': self.welford_m2,
            'baseline_updates_gated': self.baseline_updates_gated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LinkSnapshot':
        """Deserialize from dictionary."""
        ci_low = d.get('amplitude_ci_low')
        ci_high = d.get('amplitude_ci_high')
        amplitude_ci = (ci_low, ci_high) if ci_low is not None and ci_high is not None else None

        return cls(
            tx_mac=d['tx_mac'],
            rx_mac=d['rx_mac'],
            timestamp=d['timestamp'],
            amplitude_mean=d['amplitude_mean'],
            amplitude_std=d['amplitude_std'],
            amplitude_ci=amplitude_ci,
            phase_std=d['phase_std'],
            quiet_samples=d['quiet_samples'],
            is_initialized=d['is_initialized'],
            welford_n=d.get('welford_n'),
            welford_mean=d.get('welford_mean'),
            welford_m2=d.get('welford_m2'),
            baseline_updates_gated=d.get('baseline_updates_gated', 0),
        )


@dataclass
class CalibrationSnapshot:
    """
    Portable calibration export (no numpy arrays).

    This is a complete, JSON-serializable snapshot of calibration state
    that can be persisted, exported, and imported across deployments.

    Design follows OpenCV/scikit-learn patterns where the library provides
    serialization primitives, but persistence policy is left to the application.

    Usage:
        # Export from AutoCalibrator
        snapshot = calibrator.export_snapshot()

        # Serialize to JSON
        json_str = snapshot.to_json()

        # Store in SQLite, send over network, etc.
        # ...

        # Later, restore calibrator state
        snapshot = CalibrationSnapshot.from_json(json_str)
        calibrator = AutoCalibrator.from_snapshot(snapshot, node_positions)
    """
    version: int
    timestamp: float
    links: Dict[str, LinkSnapshot]

    # Current version of the snapshot format
    CURRENT_VERSION: int = 1

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'links': {key: link.to_dict() for key, link in self.links.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CalibrationSnapshot':
        """Deserialize from dictionary."""
        links = {
            key: LinkSnapshot.from_dict(link_dict)
            for key, link_dict in d.get('links', {}).items()
        }
        return cls(
            version=d.get('version', 1),
            timestamp=d.get('timestamp', 0.0),
            links=links,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> 'CalibrationSnapshot':
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(s))
