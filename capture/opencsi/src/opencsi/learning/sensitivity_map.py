"""
Sensitivity Map Learning for OpenCSI AutoCalibrator.

This module implements spatial sensitivity grid learning. Instead of assuming
theoretical elliptical Fresnel zones, it learns where each link can actually
detect changes through observations.

References:
- docs/research/AUTO_FRESNEL_CALIBRATION_LAYER.md §Mechanism 2
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

from opencsi.learning.fresnel import (
    compute_distance,
    theoretical_fresnel_response,
)

logger = logging.getLogger(__name__)

# Type alias for link key
LinkKey = Tuple[str, str]


class SensitivityMapLearner:
    """
    Learns where each link can actually detect changes.

    Instead of assuming theoretical elliptical Fresnel zones, this class
    learns a 2D grid of actual sensitivity values for each link through
    observations.

    When the system detects motion at a position (via higher-level algorithms),
    we update the sensitivity maps to reflect how each link responded.

    Attributes:
        node_positions: MAC -> (x, y) position mapping
        resolution: Grid resolution in meters
        bounds: (x_min, x_max, y_min, y_max) of the grid
        sensitivity_maps: Per-link 2D grids
        observation_counts: Per-link observation counts
    """

    def __init__(
        self,
        node_positions: Dict[str, Tuple[float, float]],
        resolution: float = 0.25,
        learning_rate: float = 0.05,
        min_observations: int = 10,
        margin: float = 0.5
    ):
        """
        Initialize the sensitivity map learner.

        Args:
            node_positions: MAC address -> (x, y) position mapping
            resolution: Grid cell size in meters
            learning_rate: EMA coefficient for sensitivity updates
            min_observations: Min observations before trusting learned value
            margin: Extra space around nodes for grid bounds
        """
        self.node_positions = node_positions.copy()
        self.resolution = resolution
        self.learning_rate = learning_rate
        self.min_observations = min_observations

        # Compute grid bounds from node positions
        if node_positions:
            xs = [p[0] for p in node_positions.values()]
            ys = [p[1] for p in node_positions.values()]
            self.bounds = (
                min(xs) - margin,
                max(xs) + margin,
                min(ys) - margin,
                max(ys) + margin
            )
        else:
            self.bounds = (0, 10, 0, 10)  # Default

        # Grid dimensions
        self.n_x = max(1, int((self.bounds[1] - self.bounds[0]) / resolution))
        self.n_y = max(1, int((self.bounds[3] - self.bounds[2]) / resolution))

        # Per-link grids (initialized on first use)
        self.sensitivity_maps: Dict[LinkKey, np.ndarray] = {}
        self.observation_counts: Dict[LinkKey, np.ndarray] = {}

        logger.debug(
            f"SensitivityMapLearner initialized: grid={self.n_x}x{self.n_y}, "
            f"resolution={resolution}m"
        )

    def _ensure_link_grids(self, link_key: LinkKey):
        """Initialize grids for a link if not already present."""
        if link_key not in self.sensitivity_maps:
            # Initialize with theoretical values
            tx_mac, rx_mac = link_key
            self.sensitivity_maps[link_key] = self._compute_theoretical_grid(
                tx_mac, rx_mac
            )
            self.observation_counts[link_key] = np.zeros(
                (self.n_y, self.n_x), dtype=np.int32
            )

    def _compute_theoretical_grid(
        self,
        tx_mac: str,
        rx_mac: str
    ) -> np.ndarray:
        """Compute theoretical Fresnel sensitivity for entire grid."""
        grid = np.zeros((self.n_y, self.n_x), dtype=np.float64)

        tx_pos = self.node_positions.get(tx_mac)
        rx_pos = self.node_positions.get(rx_mac)

        if tx_pos is None or rx_pos is None:
            # Unknown positions, return uniform 0.5
            return grid + 0.5

        for iy in range(self.n_y):
            for ix in range(self.n_x):
                x = self.bounds[0] + (ix + 0.5) * self.resolution
                y = self.bounds[2] + (iy + 0.5) * self.resolution
                grid[iy, ix] = theoretical_fresnel_response(tx_pos, rx_pos, (x, y))

        return grid

    def _position_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert (x, y) position to grid indices."""
        ix = int((position[0] - self.bounds[0]) / self.resolution)
        iy = int((position[1] - self.bounds[2]) / self.resolution)
        # Clamp to valid range
        ix = max(0, min(self.n_x - 1, ix))
        iy = max(0, min(self.n_y - 1, iy))
        return ix, iy

    def _grid_to_position(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid indices to (x, y) position (cell center)."""
        x = self.bounds[0] + (ix + 0.5) * self.resolution
        y = self.bounds[2] + (iy + 0.5) * self.resolution
        return x, y

    def on_detection_at_position(
        self,
        position: Tuple[float, float],
        link_key: LinkKey,
        response_magnitude: float
    ):
        """
        Update sensitivity map when we observe a response.

        Called when detection occurs at a position. The response magnitude
        tells us how sensitive this link actually is at this position.

        Args:
            position: (x, y) where detection occurred
            link_key: (tx_mac, rx_mac) tuple
            response_magnitude: Combined change Z-score from this link
        """
        self._ensure_link_grids(link_key)

        ix, iy = self._position_to_grid(position)

        # EMA update
        current = self.sensitivity_maps[link_key][iy, ix]
        self.sensitivity_maps[link_key][iy, ix] = (
            (1 - self.learning_rate) * current +
            self.learning_rate * response_magnitude
        )
        self.observation_counts[link_key][iy, ix] += 1

    def get_sensitivity(
        self,
        link_key: LinkKey,
        position: Tuple[float, float]
    ) -> float:
        """
        Query learned sensitivity at a position.

        Returns the learned value if we have enough observations,
        otherwise falls back to theoretical Fresnel sensitivity.

        Args:
            link_key: (tx_mac, rx_mac) tuple
            position: (x, y) query position

        Returns:
            Sensitivity value (higher = more sensitive to motion)
        """
        self._ensure_link_grids(link_key)

        ix, iy = self._position_to_grid(position)

        # Return learned value if we have enough observations
        if self.observation_counts[link_key][iy, ix] >= self.min_observations:
            return float(self.sensitivity_maps[link_key][iy, ix])
        else:
            # Fall back to theoretical
            return self._get_theoretical_sensitivity(link_key, position)

    def _get_theoretical_sensitivity(
        self,
        link_key: LinkKey,
        position: Tuple[float, float]
    ) -> float:
        """Compute theoretical Fresnel sensitivity for a position."""
        tx_mac, rx_mac = link_key
        tx_pos = self.node_positions.get(tx_mac)
        rx_pos = self.node_positions.get(rx_mac)

        if tx_pos is None or rx_pos is None:
            return 0.5  # Unknown positions, return neutral

        return theoretical_fresnel_response(tx_pos, rx_pos, position)

    def get_observation_count(
        self,
        link_key: LinkKey,
        position: Tuple[float, float]
    ) -> int:
        """Get number of observations at a position."""
        if link_key not in self.observation_counts:
            return 0
        ix, iy = self._position_to_grid(position)
        return int(self.observation_counts[link_key][iy, ix])

    def is_learned(
        self,
        link_key: LinkKey,
        position: Tuple[float, float]
    ) -> bool:
        """Check if sensitivity at position is learned (not theoretical)."""
        return self.get_observation_count(link_key, position) >= self.min_observations

    def get_coverage(self, link_key: LinkKey) -> float:
        """
        Get fraction of grid cells that have learned values.

        Returns:
            Coverage fraction [0, 1]
        """
        if link_key not in self.observation_counts:
            return 0.0

        learned_cells = np.sum(self.observation_counts[link_key] >= self.min_observations)
        total_cells = self.n_x * self.n_y
        return learned_cells / total_cells

    def get_total_observations(self, link_key: LinkKey) -> int:
        """Get total observations for a link."""
        if link_key not in self.observation_counts:
            return 0
        return int(np.sum(self.observation_counts[link_key]))

    def get_sensitivity_map(self, link_key: LinkKey) -> Optional[np.ndarray]:
        """Get the full sensitivity grid for a link."""
        return self.sensitivity_maps.get(link_key)

    def update_node_position(self, node_mac: str, position: Tuple[float, float]):
        """
        Update position for a node.

        Note: This invalidates learned sensitivity maps for affected links.
        The theoretical fallback will be used until new observations arrive.
        """
        old_pos = self.node_positions.get(node_mac)
        self.node_positions[node_mac] = position

        if old_pos is not None:
            # Check if position changed significantly
            if compute_distance(old_pos, position) > self.resolution:
                # Invalidate affected links by resetting observation counts
                for link_key in list(self.observation_counts.keys()):
                    if node_mac in link_key:
                        logger.info(f"Resetting sensitivity map for {link_key} due to position change")
                        # Recompute theoretical grid
                        tx_mac, rx_mac = link_key
                        self.sensitivity_maps[link_key] = self._compute_theoretical_grid(
                            tx_mac, rx_mac
                        )
                        self.observation_counts[link_key].fill(0)

    def remove_link(self, link_key: LinkKey):
        """
        Remove a link's sensitivity data to free memory.

        Called during link pruning to prevent unbounded memory growth.

        Args:
            link_key: (tx_mac, rx_mac) tuple
        """
        if link_key in self.sensitivity_maps:
            del self.sensitivity_maps[link_key]
        if link_key in self.observation_counts:
            del self.observation_counts[link_key]

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'node_positions': {k: list(v) for k, v in self.node_positions.items()},
            'resolution': self.resolution,
            'learning_rate': self.learning_rate,
            'min_observations': self.min_observations,
            'bounds': list(self.bounds),
            'n_x': self.n_x,
            'n_y': self.n_y,
            'sensitivity_maps': {
                f"{k[0]}->{k[1]}": v.tolist()
                for k, v in self.sensitivity_maps.items()
            },
            'observation_counts': {
                f"{k[0]}->{k[1]}": v.tolist()
                for k, v in self.observation_counts.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SensitivityMapLearner':
        """Restore from serialized data."""
        node_positions = {k: tuple(v) for k, v in data['node_positions'].items()}
        learner = cls(
            node_positions=node_positions,
            resolution=data['resolution'],
            learning_rate=data['learning_rate'],
            min_observations=data['min_observations']
        )
        learner.bounds = tuple(data['bounds'])
        learner.n_x = data['n_x']
        learner.n_y = data['n_y']

        # Restore maps
        for key_str, grid in data.get('sensitivity_maps', {}).items():
            parts = key_str.split('->')
            if len(parts) == 2:
                link_key = (parts[0], parts[1])
                learner.sensitivity_maps[link_key] = np.array(grid, dtype=np.float64)

        for key_str, counts in data.get('observation_counts', {}).items():
            parts = key_str.split('->')
            if len(parts) == 2:
                link_key = (parts[0], parts[1])
                learner.observation_counts[link_key] = np.array(counts, dtype=np.int32)

        return learner
