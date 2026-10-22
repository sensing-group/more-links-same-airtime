"""
Baseline Management Module for OpenCSI.

Manages per-link baseline statistics for CSI calibration.
Baselines represent the "empty room" state and are used to compute Z-scores.

Usage:
    from opencsi.core.baseline import BaselineManager

    manager = BaselineManager()

    # Update baseline during quiet periods
    manager.update(link_key, amplitude, phase, is_quiet=True)

    # Compute Z-score relative to baseline
    z_score = manager.compute_zscore(link_key, amplitude, phase)

    # Check baseline health
    if manager.is_initialized(link_key):
        print(f"Baseline age: {manager.get_age(link_key):.1f}s")
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Baseline learning
MIN_SAMPLES_FOR_BASELINE = 10
EMA_ALPHA_BASELINE = 0.05  # Exponential moving average alpha
MIN_STD_THRESHOLD = 0.1  # Minimum std to avoid divide-by-zero

# Drift detection
DRIFT_THRESHOLD_SIGMA = 3.0  # Sigma change indicating drift
DRIFT_CHECK_INTERVAL_S = 60.0

# Baseline staleness
BASELINE_STALE_TIMEOUT_S = 3600.0  # 1 hour
BASELINE_WARN_AGE_S = 1800.0  # 30 minutes


# =============================================================================
# Baseline Data Structure
# =============================================================================

@dataclass
class LinkBaseline:
    """
    Per-link baseline statistics.

    Tracks mean and standard deviation of amplitude and phase
    during quiet (empty room) periods.
    """
    # Amplitude statistics
    amplitude_mean: Optional[np.ndarray] = None
    amplitude_std: Optional[np.ndarray] = None

    # Phase statistics
    phase_mean: Optional[np.ndarray] = None
    phase_std: Optional[np.ndarray] = None

    # Metadata
    sample_count: int = 0
    last_update_time: float = 0.0
    creation_time: float = field(default_factory=time.time)

    # Quality tracking
    initialization_complete: bool = False
    drift_detected: bool = False
    last_drift_check: float = 0.0

    @property
    def age_seconds(self) -> float:
        """Time since last update in seconds."""
        if self.last_update_time == 0:
            return float('inf')
        return time.time() - self.last_update_time

    @property
    def is_stale(self) -> bool:
        """Check if baseline is too old to be reliable."""
        return self.age_seconds > BASELINE_STALE_TIMEOUT_S

    def ensure_arrays(self, n_subcarriers: int) -> None:
        """Initialize arrays if needed or resize if n_subcarriers changed."""
        if self.amplitude_mean is None or len(self.amplitude_mean) != n_subcarriers:
            self.amplitude_mean = np.zeros(n_subcarriers)
            self.amplitude_std = np.ones(n_subcarriers)
            self.phase_mean = np.zeros(n_subcarriers)
            self.phase_std = np.ones(n_subcarriers)
            self.sample_count = 0
            self.initialization_complete = False


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaselineProvider(ABC):
    """
    Abstract base class for baseline management.

    Subclass to implement custom baseline learning strategies.
    """

    @abstractmethod
    def update(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
        is_quiet: bool = True,
        **kwargs
    ) -> bool:
        """
        Update baseline with new sample.

        Args:
            link_key: (tx_mac, rx_mac) tuple
            amplitude: Amplitude array
            phase: Phase array
            is_quiet: Whether this is a quiet period sample
            **kwargs: Additional context

        Returns:
            True if baseline was updated
        """
        ...

    @abstractmethod
    def compute_zscore(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Z-scores relative to baseline.

        Args:
            link_key: (tx_mac, rx_mac) tuple
            amplitude: Current amplitude
            phase: Current phase

        Returns:
            Tuple of (amplitude_zscore, phase_zscore)
        """
        ...

    @abstractmethod
    def is_initialized(self, link_key: Tuple[str, str]) -> bool:
        """Check if baseline is initialized for this link."""
        ...


# =============================================================================
# EMA Baseline Manager (Default)
# =============================================================================

class BaselineManager(BaselineProvider):
    """
    Manages per-link baselines using exponential moving average.

    Updates baseline statistics during quiet periods using EMA,
    which allows gradual adaptation to environmental changes
    while maintaining stability.

    Args:
        ema_alpha: EMA smoothing factor (default: 0.05)
        min_samples: Minimum samples before baseline is valid (default: 10)
        min_std: Minimum standard deviation to avoid div-by-zero (default: 0.1)
    """

    def __init__(
        self,
        ema_alpha: float = EMA_ALPHA_BASELINE,
        min_samples: int = MIN_SAMPLES_FOR_BASELINE,
        min_std: float = MIN_STD_THRESHOLD,
    ):
        self.ema_alpha = ema_alpha
        self.min_samples = min_samples
        self.min_std = min_std

        self._baselines: Dict[Tuple[str, str], LinkBaseline] = {}
        self._frozen: bool = False

    def update(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
        is_quiet: bool = True,
        **kwargs
    ) -> bool:
        """
        Update baseline using EMA during quiet periods.

        Returns:
            True if baseline was updated
        """
        if self._frozen:
            return False

        if not is_quiet:
            return False

        baseline = self._get_or_create_baseline(link_key, len(amplitude))
        baseline.ensure_arrays(len(amplitude))

        # Update amplitude statistics with EMA
        if baseline.sample_count == 0:
            baseline.amplitude_mean = amplitude.copy()
            baseline.amplitude_std = np.ones_like(amplitude)
            baseline.phase_mean = phase.copy()
            baseline.phase_std = np.ones_like(phase)
        else:
            # EMA update for mean — store old values for variance calculation
            old_amp_mean = baseline.amplitude_mean
            old_phase_mean = baseline.phase_mean
            baseline.amplitude_mean = (
                self.ema_alpha * amplitude +
                (1 - self.ema_alpha) * old_amp_mean
            )
            baseline.phase_mean = (
                self.ema_alpha * phase +
                (1 - self.ema_alpha) * old_phase_mean
            )

            # Update std using running variance estimate
            # Use OLD mean for deviation — using new mean underestimates variance
            amp_diff = amplitude - old_amp_mean
            phase_diff = phase - old_phase_mean

            baseline.amplitude_std = np.sqrt(
                self.ema_alpha * amp_diff**2 +
                (1 - self.ema_alpha) * baseline.amplitude_std**2
            )
            baseline.phase_std = np.sqrt(
                self.ema_alpha * phase_diff**2 +
                (1 - self.ema_alpha) * baseline.phase_std**2
            )

        # Enforce minimum std
        baseline.amplitude_std = np.maximum(baseline.amplitude_std, self.min_std)
        baseline.phase_std = np.maximum(baseline.phase_std, self.min_std)

        baseline.sample_count += 1
        baseline.last_update_time = time.time()

        if baseline.sample_count >= self.min_samples:
            baseline.initialization_complete = True

        return True

    def compute_zscore(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute mean Z-scores across subcarriers.

        Returns:
            Tuple of (amplitude_zscore, phase_zscore)
        """
        baseline = self._baselines.get(link_key)

        if baseline is None or not baseline.initialization_complete:
            # No baseline - return zero (no deviation from unknown baseline)
            return 0.0, 0.0

        # Guard against shape mismatch (e.g., subcarrier count change)
        if len(amplitude) != len(baseline.amplitude_mean):
            return 0.0, 0.0

        # Compute per-subcarrier Z-scores
        amp_z = (amplitude - baseline.amplitude_mean) / baseline.amplitude_std
        phase_z = (phase - baseline.phase_mean) / baseline.phase_std

        # Return mean absolute Z-score across subcarriers
        return float(np.mean(np.abs(amp_z))), float(np.mean(np.abs(phase_z)))

    def compute_zscore_array(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-subcarrier Z-scores.

        Returns:
            Tuple of (amplitude_zscore_array, phase_zscore_array)
        """
        baseline = self._baselines.get(link_key)

        if baseline is None or not baseline.initialization_complete:
            return np.zeros_like(amplitude), np.zeros_like(phase)

        # Guard against shape mismatch (e.g., subcarrier count change)
        if len(amplitude) != len(baseline.amplitude_mean):
            return np.zeros_like(amplitude), np.zeros_like(phase)

        amp_z = (amplitude - baseline.amplitude_mean) / baseline.amplitude_std
        phase_z = (phase - baseline.phase_mean) / baseline.phase_std

        return amp_z, phase_z

    def is_initialized(self, link_key: Tuple[str, str]) -> bool:
        """Check if baseline is initialized."""
        baseline = self._baselines.get(link_key)
        return baseline is not None and baseline.initialization_complete

    def get_age(self, link_key: Tuple[str, str]) -> float:
        """Get baseline age in seconds."""
        baseline = self._baselines.get(link_key)
        return baseline.age_seconds if baseline else float('inf')

    def is_stale(self, link_key: Tuple[str, str]) -> bool:
        """Check if baseline is stale."""
        baseline = self._baselines.get(link_key)
        return baseline is None or baseline.is_stale

    def get_baseline(self, link_key: Tuple[str, str]) -> Optional[LinkBaseline]:
        """Get baseline for a link."""
        return self._baselines.get(link_key)

    def reset(self, link_key: Optional[Tuple[str, str]] = None) -> None:
        """Reset baseline(s)."""
        if link_key is None:
            self._baselines.clear()
        else:
            self._baselines.pop(link_key, None)

    def freeze(self, freeze: bool = True) -> None:
        """Freeze/unfreeze baseline updates."""
        self._frozen = freeze

    @property
    def is_frozen(self) -> bool:
        """Check if baselines are frozen."""
        return self._frozen

    def get_all_links(self) -> List[Tuple[str, str]]:
        """Get all tracked link keys."""
        return list(self._baselines.keys())

    def get_stats(self) -> dict:
        """Get baseline statistics."""
        initialized = sum(1 for b in self._baselines.values() if b.initialization_complete)
        stale = sum(1 for b in self._baselines.values() if b.is_stale)

        return {
            'total_links': len(self._baselines),
            'initialized': initialized,
            'stale': stale,
            'frozen': self._frozen,
        }

    def _get_or_create_baseline(
        self,
        link_key: Tuple[str, str],
        n_subcarriers: int
    ) -> LinkBaseline:
        """Get or create baseline for a link."""
        if link_key not in self._baselines:
            self._baselines[link_key] = LinkBaseline()

        baseline = self._baselines[link_key]
        baseline.ensure_arrays(n_subcarriers)
        return baseline


# =============================================================================
# Windowed Baseline Manager
# =============================================================================

class WindowedBaselineManager(BaselineProvider):
    """
    Baseline manager using sliding window of recent quiet samples.

    Computes baseline from last N quiet samples rather than EMA.
    More responsive to recent changes but requires more memory.

    Args:
        window_size: Number of samples to keep (default: 100)
        min_samples: Minimum samples before baseline is valid (default: 10)
        min_std: Minimum standard deviation to avoid div-by-zero (default: 0.1)
    """

    def __init__(
        self,
        window_size: int = 100,
        min_samples: int = MIN_SAMPLES_FOR_BASELINE,
        min_std: float = MIN_STD_THRESHOLD,
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        self.min_std = min_std

        self._windows: Dict[Tuple[str, str], deque] = {}
        self._frozen: bool = False

    def update(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
        is_quiet: bool = True,
        **kwargs
    ) -> bool:
        """Update baseline window."""
        if self._frozen or not is_quiet:
            return False

        if link_key not in self._windows:
            self._windows[link_key] = deque(maxlen=self.window_size)

        self._windows[link_key].append((amplitude.copy(), phase.copy()))
        return True

    def compute_zscore(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        phase: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute Z-scores from windowed baseline."""
        window = self._windows.get(link_key)

        if window is None or len(window) < self.min_samples:
            return 0.0, 0.0

        # Filter window to matching subcarrier count (handles bandwidth changes)
        n_sc = len(amplitude)
        matching = [(s[0], s[1]) for s in window if len(s[0]) == n_sc]
        if len(matching) < self.min_samples:
            return 0.0, 0.0

        # Compute statistics from window
        amplitudes = np.array([s[0] for s in matching])
        phases = np.array([s[1] for s in matching])

        amp_mean = np.mean(amplitudes, axis=0)
        amp_std = np.maximum(np.std(amplitudes, axis=0), self.min_std)
        phase_mean = np.mean(phases, axis=0)
        phase_std = np.maximum(np.std(phases, axis=0), self.min_std)

        amp_z = np.mean(np.abs((amplitude - amp_mean) / amp_std))
        phase_z = np.mean(np.abs((phase - phase_mean) / phase_std))

        return float(amp_z), float(phase_z)

    def is_initialized(self, link_key: Tuple[str, str]) -> bool:
        """Check if enough samples collected."""
        window = self._windows.get(link_key)
        return window is not None and len(window) >= self.min_samples

    def reset(self, link_key: Optional[Tuple[str, str]] = None) -> None:
        """Reset window(s)."""
        if link_key is None:
            self._windows.clear()
        else:
            self._windows.pop(link_key, None)

    def freeze(self, freeze: bool = True) -> None:
        """Freeze/unfreeze updates."""
        self._frozen = freeze

    @property
    def is_frozen(self) -> bool:
        return self._frozen


# =============================================================================
# Factory Function
# =============================================================================

def get_baseline_manager(
    manager_type: str = "ema",
    **kwargs
) -> BaselineProvider:
    """
    Factory function for creating baseline managers.

    Args:
        manager_type: "ema" or "windowed"
        **kwargs: Arguments passed to manager constructor

    Returns:
        BaselineProvider instance

    Example:
        manager = get_baseline_manager("windowed", window_size=200)
    """
    managers = {
        "ema": BaselineManager,
        "windowed": WindowedBaselineManager,
    }

    if manager_type not in managers:
        raise ValueError(
            f"Unknown manager type: {manager_type}. "
            f"Available: {list(managers.keys())}"
        )

    return managers[manager_type](**kwargs)
