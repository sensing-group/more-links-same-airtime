"""
Antenna Pattern Learning for OpenCSI AutoCalibrator.

This module learns effective antenna gain patterns from observed signals.
Traditional approach: Use known antenna pattern to predict signal.
Our approach: Use observed signals to reconstruct antenna pattern.

This implements "Inverse Fresnel Imaging" - by observing how links respond
differently than theoretical predictions, we infer the actual antenna
characteristics.

References:
- docs/research/AUTO_FRESNEL_CALIBRATION_LAYER.md §Mechanism 3
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from opencsi.learning.fresnel import (
    compute_angle,
    theoretical_fresnel_response,
)

logger = logging.getLogger(__name__)

# Type alias for link key
LinkKey = Tuple[str, str]


@dataclass
class AntennaPattern:
    """
    Polar gain pattern for a single antenna.

    Stores learned gain factors at discrete angular bins around the antenna.
    Enables lookup of effective gain at any angle through interpolation.

    Attributes:
        node_mac: MAC address of the node
        angular_resolution_deg: Resolution of angular bins
        gains: Array of gain factors per angle bin
        observation_counts: Number of observations per bin
    """
    node_mac: str
    angular_resolution_deg: float = 10.0

    # Derived from resolution
    n_bins: int = field(init=False)

    # Per-angle statistics
    gains: np.ndarray = field(init=False)
    observation_counts: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize arrays based on angular resolution."""
        self.n_bins = int(360.0 / self.angular_resolution_deg)
        # Initialize gains to 1.0 (isotropic assumption)
        self.gains = np.ones(self.n_bins, dtype=np.float64)
        self.observation_counts = np.zeros(self.n_bins, dtype=np.int32)

    def _angle_to_bin(self, angle_rad: float) -> int:
        """Convert angle in radians to bin index."""
        angle_deg = math.degrees(angle_rad) % 360.0
        return int(angle_deg / self.angular_resolution_deg) % self.n_bins

    def update(self, angle_rad: float, gain_factor: float, learning_rate: float = 0.02):
        """
        Update gain at a specific angle.

        Uses EMA to smoothly incorporate new observations.

        Args:
            angle_rad: Angle in radians
            gain_factor: Observed gain factor (response / expected)
            learning_rate: EMA coefficient (α)
        """
        bin_idx = self._angle_to_bin(angle_rad)

        # EMA update
        self.gains[bin_idx] = (
            (1 - learning_rate) * self.gains[bin_idx] +
            learning_rate * gain_factor
        )
        self.observation_counts[bin_idx] += 1

    def get_gain(self, angle_rad: float) -> float:
        """
        Get gain factor at a specific angle.

        Uses linear interpolation between adjacent bins.

        Args:
            angle_rad: Angle in radians

        Returns:
            Gain factor at this angle (1.0 = nominal)
        """
        angle_deg = math.degrees(angle_rad) % 360.0
        bin_float = angle_deg / self.angular_resolution_deg

        bin_low = int(bin_float) % self.n_bins
        bin_high = (bin_low + 1) % self.n_bins
        frac = bin_float - int(bin_float)

        # Linear interpolation
        gain = (1 - frac) * self.gains[bin_low] + frac * self.gains[bin_high]
        return float(gain)

    def has_observations(self, angle_rad: float, min_obs: int = 5) -> bool:
        """Check if we have enough observations at this angle."""
        bin_idx = self._angle_to_bin(angle_rad)
        return self.observation_counts[bin_idx] >= min_obs

    def get_total_observations(self) -> int:
        """Get total observation count across all angles."""
        return int(np.sum(self.observation_counts))

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'node_mac': self.node_mac,
            'angular_resolution_deg': self.angular_resolution_deg,
            'gains': self.gains.tolist(),
            'observation_counts': self.observation_counts.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AntennaPattern':
        """Restore from serialized data."""
        pattern = cls(
            node_mac=data['node_mac'],
            angular_resolution_deg=data['angular_resolution_deg']
        )
        pattern.gains = np.array(data['gains'], dtype=np.float64)
        pattern.observation_counts = np.array(data['observation_counts'], dtype=np.int32)
        return pattern


class AntennaPatternLearner:
    """
    Learns effective antenna gain patterns from observed signals.

    Traditional approach: Use known antenna pattern to predict signal.
    Our approach: Use observed signals to reconstruct antenna pattern.

    This implements "Inverse Fresnel Imaging" - by observing how links
    respond differently than theoretical predictions, we infer the actual
    antenna characteristics.

    Attributes:
        node_positions: MAC -> (x, y) position mapping
        antenna_patterns: MAC -> AntennaPattern mapping
        config: Learning parameters
    """

    def __init__(
        self,
        node_positions: Dict[str, Tuple[float, float]],
        angular_resolution_deg: float = 10.0,
        learning_rate: float = 0.02,
        min_expected_response: float = 0.1
    ):
        """
        Initialize the antenna pattern learner.

        Args:
            node_positions: MAC address -> (x, y) position mapping
            angular_resolution_deg: Angular bin size for patterns
            learning_rate: EMA coefficient for pattern updates
            min_expected_response: Minimum expected response to consider update
        """
        self.node_positions = node_positions.copy()
        self.angular_resolution_deg = angular_resolution_deg
        self.learning_rate = learning_rate
        self.min_expected_response = min_expected_response

        # Initialize patterns for all known nodes
        self.antenna_patterns: Dict[str, AntennaPattern] = {}
        for mac in node_positions:
            self.antenna_patterns[mac] = AntennaPattern(
                node_mac=mac,
                angular_resolution_deg=angular_resolution_deg
            )

        logger.debug(
            f"AntennaPatternLearner initialized for {len(node_positions)} nodes, "
            f"resolution={angular_resolution_deg}°"
        )

    def on_consistent_detection(
        self,
        position: Tuple[float, float],
        link_responses: Dict[LinkKey, float]
    ):
        """
        Update antenna patterns based on a consistent detection.

        Called when multiple links agree on a detection position.
        We use the ratio of observed vs expected response to learn
        the actual antenna gain at that angle.

        Args:
            position: Consensus (x, y) position estimate
            link_responses: Dict of (tx_mac, rx_mac) -> combined_change
        """
        for link_key, response in link_responses.items():
            tx_mac, rx_mac = link_key

            # Skip if node positions unknown
            if tx_mac not in self.node_positions or rx_mac not in self.node_positions:
                continue

            tx_pos = self.node_positions[tx_mac]
            rx_pos = self.node_positions[rx_mac]

            # Compute expected response from theoretical Fresnel
            expected = theoretical_fresnel_response(tx_pos, rx_pos, position)

            # Only update if we expected a meaningful response
            if expected < self.min_expected_response:
                continue

            # The ratio reveals the effective antenna gain
            gain_factor = response / expected

            # Clamp gain factor to reasonable range (0.1 to 10x)
            gain_factor = max(0.1, min(10.0, gain_factor))

            # Update TX antenna pattern
            angle_from_tx = compute_angle(tx_pos, position)
            if tx_mac in self.antenna_patterns:
                self.antenna_patterns[tx_mac].update(
                    angle_from_tx, gain_factor, self.learning_rate
                )

            # Update RX antenna pattern
            angle_from_rx = compute_angle(rx_pos, position)
            if rx_mac in self.antenna_patterns:
                self.antenna_patterns[rx_mac].update(
                    angle_from_rx, gain_factor, self.learning_rate
                )

    def get_antenna_gain(self, node_mac: str, angle_rad: float) -> float:
        """
        Get learned antenna gain at a specific angle.

        Args:
            node_mac: Node MAC address
            angle_rad: Angle in radians

        Returns:
            Gain factor (1.0 = nominal isotropic)
        """
        if node_mac not in self.antenna_patterns:
            return 1.0  # Unknown node, assume isotropic
        return self.antenna_patterns[node_mac].get_gain(angle_rad)

    def get_effective_gain(
        self,
        tx_mac: str,
        rx_mac: str,
        target_pos: Tuple[float, float]
    ) -> float:
        """
        Get combined effective gain for a link at a target position.

        Combines the TX and RX antenna gains for the angles toward
        the target position.

        Args:
            tx_mac: Transmitter MAC
            rx_mac: Receiver MAC
            target_pos: (x, y) position of interest

        Returns:
            Combined gain factor
        """
        tx_gain = 1.0
        rx_gain = 1.0

        if tx_mac in self.node_positions and tx_mac in self.antenna_patterns:
            angle_from_tx = compute_angle(self.node_positions[tx_mac], target_pos)
            tx_gain = self.antenna_patterns[tx_mac].get_gain(angle_from_tx)

        if rx_mac in self.node_positions and rx_mac in self.antenna_patterns:
            angle_from_rx = compute_angle(self.node_positions[rx_mac], target_pos)
            rx_gain = self.antenna_patterns[rx_mac].get_gain(angle_from_rx)

        return tx_gain * rx_gain

    def get_pattern(self, node_mac: str) -> Optional[AntennaPattern]:
        """Get the antenna pattern for a node."""
        return self.antenna_patterns.get(node_mac)

    def update_node_position(self, node_mac: str, position: Tuple[float, float]):
        """Update position for a node (e.g., after recalibration)."""
        self.node_positions[node_mac] = position
        if node_mac not in self.antenna_patterns:
            self.antenna_patterns[node_mac] = AntennaPattern(
                node_mac=node_mac,
                angular_resolution_deg=self.angular_resolution_deg
            )

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'node_positions': {k: list(v) for k, v in self.node_positions.items()},
            'angular_resolution_deg': self.angular_resolution_deg,
            'learning_rate': self.learning_rate,
            'min_expected_response': self.min_expected_response,
            'patterns': {
                mac: pattern.to_dict()
                for mac, pattern in self.antenna_patterns.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AntennaPatternLearner':
        """Restore from serialized data."""
        node_positions = {k: tuple(v) for k, v in data['node_positions'].items()}
        learner = cls(
            node_positions=node_positions,
            angular_resolution_deg=data['angular_resolution_deg'],
            learning_rate=data['learning_rate'],
            min_expected_response=data['min_expected_response']
        )
        # Restore patterns
        for mac, pattern_data in data.get('patterns', {}).items():
            learner.antenna_patterns[mac] = AntennaPattern.from_dict(pattern_data)
        return learner
