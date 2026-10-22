"""
NBVI (Normalized Baseline Variability Index) Subcarrier Selection.

Automatically selects the k most stable subcarriers per-link based on
variance during quiet periods. Unstable subcarriers (high NBVI) are
down-weighted rather than discarded, preserving spatial information
while reducing noise.

Reference: ESPectre (Nguyen et al., 2021) - NBVI metric for subcarrier
quality assessment in CSI-based sensing.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

NBVI_DEFAULT_N_SUBCARRIERS = 12  # ESPectre default
NBVI_DEFAULT_MIN_QUIET_SAMPLES = 50  # Samples needed before computing NBVI
NBVI_DEFAULT_RECOMPUTE_INTERVAL_S = 300.0  # Recompute every 5 minutes
NBVI_EMA_ALPHA = 0.3  # EMA smoothing for NBVI score updates
NBVI_UNSELECTED_WEIGHT = 0.1  # Weight for non-selected subcarriers
NBVI_HISTORY_BUFFER_MULTIPLIER = 2  # History deque maxlen multiplier

# Small epsilon to avoid log(0) and divide-by-zero
NUMERICAL_EPSILON = 1e-10


class NBVISubcarrierSelector:
    """
    NBVI (Normalized Baseline Variability Index) subcarrier selection.

    Automatically selects the k most stable subcarriers per-link based on
    variance during quiet periods.
    """

    def __init__(
        self,
        n_subcarriers: int = NBVI_DEFAULT_N_SUBCARRIERS,
        min_quiet_samples: int = NBVI_DEFAULT_MIN_QUIET_SAMPLES,
        recompute_interval_s: float = NBVI_DEFAULT_RECOMPUTE_INTERVAL_S
    ):
        self.n_subcarriers = n_subcarriers
        self.min_quiet_samples = min_quiet_samples
        self.recompute_interval_s = recompute_interval_s

        self._link_nbvi: Dict[Tuple[str, str], np.ndarray] = {}
        self._link_selected: Dict[Tuple[str, str], List[int]] = {}
        self._link_quiet_history: Dict[Tuple[str, str], deque] = {}
        self._last_recompute: Dict[Tuple[str, str], float] = {}

    def update_quiet_sample(
        self,
        link_key: Tuple[str, str],
        amplitude: np.ndarray,
        timestamp: float
    ) -> bool:
        """Add a quiet-period sample for NBVI computation."""
        if link_key not in self._link_quiet_history:
            maxlen = self.min_quiet_samples * NBVI_HISTORY_BUFFER_MULTIPLIER
            self._link_quiet_history[link_key] = deque(maxlen=maxlen)

        self._link_quiet_history[link_key].append(amplitude.copy())

        history_len = len(self._link_quiet_history[link_key])
        last_recompute = self._last_recompute.get(link_key, 0.0)
        time_since_recompute = timestamp - last_recompute

        if (history_len >= self.min_quiet_samples and
                time_since_recompute >= self.recompute_interval_s):
            self._recompute_nbvi(link_key, timestamp)
            return True

        return False

    def _recompute_nbvi(self, link_key: Tuple[str, str], timestamp: float):
        """Recompute NBVI scores and selected subcarriers for a link."""
        history = self._link_quiet_history[link_key]
        if len(history) < self.min_quiet_samples:
            return

        try:
            samples = np.array(list(history))
        except ValueError:
            logger.warning(f"NBVI: inconsistent sample sizes for {link_key}, clearing history")
            self._link_quiet_history[link_key].clear()
            return

        variances = np.var(samples, axis=0)
        mean_amplitudes = np.mean(np.abs(samples), axis=0)

        nbvi_scores = np.where(
            mean_amplitudes > NUMERICAL_EPSILON,
            np.std(samples, axis=0) / mean_amplitudes,
            np.inf
        )

        if link_key in self._link_nbvi:
            prev_scores = self._link_nbvi[link_key]
            if len(prev_scores) == len(nbvi_scores):
                # Only EMA-blend finite values; inf entries stay inf until
                # a valid score replaces them (prevents permanent corruption)
                both_finite = np.isfinite(nbvi_scores) & np.isfinite(prev_scores)
                blended = np.where(
                    both_finite,
                    NBVI_EMA_ALPHA * nbvi_scores + (1 - NBVI_EMA_ALPHA) * prev_scores,
                    np.where(np.isfinite(nbvi_scores), nbvi_scores, prev_scores),
                )
                nbvi_scores = blended

        self._link_nbvi[link_key] = nbvi_scores

        valid_indices = np.where(np.isfinite(nbvi_scores))[0]
        if len(valid_indices) < self.n_subcarriers:
            selected = valid_indices.tolist()
        else:
            sorted_indices = valid_indices[np.argsort(nbvi_scores[valid_indices])]
            selected = sorted_indices[:self.n_subcarriers].tolist()

        self._link_selected[link_key] = selected
        self._last_recompute[link_key] = timestamp

    def get_selected_subcarriers(self, link_key: Tuple[str, str]) -> Optional[List[int]]:
        """Get selected subcarrier indices for a link."""
        return self._link_selected.get(link_key)

    def _build_weight_array(
        self,
        selected: List[int],
        n_subcarriers: int
    ) -> np.ndarray:
        """Build a weight array from selected subcarrier indices."""
        weights = np.full(n_subcarriers, NBVI_UNSELECTED_WEIGHT)
        valid_indices = np.array([i for i in selected if 0 <= i < n_subcarriers])
        if len(valid_indices) > 0:
            weights[valid_indices] = 1.0
        return weights

    def apply_selection(
        self,
        amplitude: np.ndarray,
        link_key: Tuple[str, str]
    ) -> np.ndarray:
        """Apply subcarrier selection as weights."""
        selected = self._link_selected.get(link_key)
        if selected is None or len(selected) == 0:
            return amplitude

        weights = self._build_weight_array(selected, len(amplitude))
        return amplitude * weights

    def get_subcarrier_weights(self, link_key: Tuple[str, str], n_subcarriers: int) -> Optional[np.ndarray]:
        """Get the weight array for subcarriers."""
        selected = self._link_selected.get(link_key)
        if selected is None:
            return None
        return self._build_weight_array(selected, n_subcarriers)

    def get_stats(self) -> Dict:
        """Get statistics about NBVI selection state."""
        return {
            'links_with_selection': len(self._link_selected),
            'links_with_history': len(self._link_quiet_history),
            'avg_selected_count': np.mean([
                len(s) for s in self._link_selected.values()
            ]) if self._link_selected else 0,
        }

    def clear_link(self, link_key: Tuple[str, str]):
        """Clear NBVI state for a specific link."""
        self._link_nbvi.pop(link_key, None)
        self._link_selected.pop(link_key, None)
        self._link_quiet_history.pop(link_key, None)
        self._last_recompute.pop(link_key, None)
