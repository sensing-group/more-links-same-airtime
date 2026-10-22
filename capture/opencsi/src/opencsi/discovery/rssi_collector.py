"""RSSI Collector for calibration phase.

Collects RSSI measurements from all TX-RX pairs during a calibration window
to build a complete RSSI matrix for position estimation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Minimum RSSI samples required per link for reliable distance estimation
CALIBRATION_MIN_SAMPLES = 50

# Maximum time to wait for calibration data (seconds)
CALIBRATION_TIMEOUT_S = 30.0

# Minimum fraction of possible links that must have data
CALIBRATION_MIN_LINKS_FRACTION = 0.7

# RSSI outlier rejection bounds (dBm)
RSSI_MIN_VALID = -100
RSSI_MAX_VALID = -10


@dataclass
class LinkRSSIStats:
    """Statistics for RSSI samples on a single link."""

    tx_mac: str
    rx_mac: str
    samples: List[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        if not self.samples:
            return float('nan')
        return float(np.mean(self.samples))

    @property
    def median(self) -> float:
        if not self.samples:
            return float('nan')
        return float(np.median(self.samples))

    @property
    def std(self) -> float:
        if not self.samples:
            return float('nan')
        return float(np.std(self.samples))

    def add_sample(self, rssi: int) -> None:
        """Add an RSSI sample with outlier rejection."""
        if RSSI_MIN_VALID <= rssi <= RSSI_MAX_VALID:
            self.samples.append(rssi)


@dataclass
class CollectionProgress:
    """Current state of RSSI collection."""

    elapsed_s: float
    timeout_s: float
    nodes_discovered: int
    total_links_possible: int
    links_with_data: int
    links_complete: int  # Have >= CALIBRATION_MIN_SAMPLES
    min_samples_per_link: int
    is_complete: bool

    @property
    def progress_fraction(self) -> float:
        """Overall progress as fraction [0, 1]."""
        if self.total_links_possible == 0:
            return 0.0
        # Weight: 70% link coverage, 30% sample count
        link_progress = self.links_complete / self.total_links_possible
        time_progress = min(1.0, self.elapsed_s / self.timeout_s)
        return 0.7 * link_progress + 0.3 * time_progress


class RSSICollector:
    """Collects pairwise RSSI measurements during calibration.

    Usage:
        collector = RSSICollector()
        collector.start_collection()

        # Feed RSSI samples from CSI packets
        collector.add_rssi_sample("AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66", -45)

        # Check progress
        progress = collector.get_progress()

        # When complete, get the RSSI matrix
        if collector.is_collection_complete():
            rssi_matrix, mac_list = collector.get_link_rssi_matrix()
    """

    def __init__(
        self,
        min_samples: int = CALIBRATION_MIN_SAMPLES,
        timeout_s: float = CALIBRATION_TIMEOUT_S,
        min_links_fraction: float = CALIBRATION_MIN_LINKS_FRACTION,
    ):
        self.min_samples = min_samples
        self.timeout_s = timeout_s
        self.min_links_fraction = min_links_fraction

        # Collection state
        self._collecting = False
        self._start_time: Optional[float] = None
        self._links: Dict[Tuple[str, str], LinkRSSIStats] = {}
        self._discovered_macs: Set[str] = set()

    def start_collection(self) -> None:
        """Begin RSSI collection phase."""
        self._collecting = True
        self._start_time = time.monotonic()
        self._links.clear()
        self._discovered_macs.clear()
        logger.info("RSSI calibration collection started")

    def stop_collection(self) -> None:
        """Stop RSSI collection."""
        self._collecting = False
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        logger.info(
            f"RSSI collection stopped after {elapsed:.1f}s, "
            f"{len(self._links)} links, {len(self._discovered_macs)} nodes"
        )

    def add_rssi_sample(self, tx_mac: str, rx_mac: str, rssi: int) -> None:
        """Add an RSSI sample from a CSI packet.

        Args:
            tx_mac: Transmitter MAC address
            rx_mac: Receiver MAC address
            rssi: RSSI value in dBm (typically -90 to -20)
        """
        if not self._collecting:
            return

        # Normalize MAC format (uppercase, colon-separated)
        tx_mac = tx_mac.upper()
        rx_mac = rx_mac.upper()

        # Skip self-links (shouldn't happen but be safe)
        if tx_mac == rx_mac:
            return

        # Track discovered nodes
        self._discovered_macs.add(tx_mac)
        self._discovered_macs.add(rx_mac)

        # Get or create link stats
        link_key = (tx_mac, rx_mac)
        if link_key not in self._links:
            self._links[link_key] = LinkRSSIStats(tx_mac=tx_mac, rx_mac=rx_mac)

        self._links[link_key].add_sample(rssi)

    def get_progress(self) -> CollectionProgress:
        """Get current collection progress."""
        elapsed = 0.0
        if self._start_time:
            elapsed = time.monotonic() - self._start_time

        n_nodes = len(self._discovered_macs)
        # For N nodes, there are N*(N-1) directed links
        total_links_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 0

        links_with_data = len(self._links)
        links_complete = sum(
            1 for link in self._links.values()
            if link.count >= self.min_samples
        )

        min_samples = min(
            (link.count for link in self._links.values()),
            default=0
        )

        return CollectionProgress(
            elapsed_s=elapsed,
            timeout_s=self.timeout_s,
            nodes_discovered=n_nodes,
            total_links_possible=total_links_possible,
            links_with_data=links_with_data,
            links_complete=links_complete,
            min_samples_per_link=min_samples,
            is_complete=self.is_collection_complete(),
        )

    def is_collection_complete(self) -> bool:
        """Check if we have enough data for calibration."""
        if not self._collecting:
            return False

        # Check timeout
        if self._start_time:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self.timeout_s:
                return True

        # Need at least 3 nodes for 2D positioning
        n_nodes = len(self._discovered_macs)
        if n_nodes < 3:
            return False

        # Check link coverage
        total_links_possible = n_nodes * (n_nodes - 1)
        links_complete = sum(
            1 for link in self._links.values()
            if link.count >= self.min_samples
        )

        coverage = links_complete / total_links_possible if total_links_possible > 0 else 0
        return coverage >= self.min_links_fraction

    def get_link_rssi_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Build symmetric RSSI matrix from collected samples.

        Returns:
            Tuple of:
            - rssi_matrix: (N, N) array where element [i,j] is median RSSI
              from node i to node j. Diagonal is 0. Missing links are NaN.
            - mac_list: List of MAC addresses in matrix order.
        """
        # Sort MACs for consistent ordering
        mac_list = sorted(self._discovered_macs)
        mac_to_idx = {mac: i for i, mac in enumerate(mac_list)}

        n = len(mac_list)
        rssi_matrix = np.full((n, n), np.nan, dtype=np.float64)
        np.fill_diagonal(rssi_matrix, 0.0)  # Self-links have no attenuation

        for (tx_mac, rx_mac), stats in self._links.items():
            if stats.count > 0:
                i = mac_to_idx.get(tx_mac)
                j = mac_to_idx.get(rx_mac)
                if i is not None and j is not None:
                    # Use median for robustness to outliers
                    rssi_matrix[i, j] = stats.median

        return rssi_matrix, mac_list

    def get_symmetrized_rssi_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Get RSSI matrix with symmetric averaging.

        For distance estimation, we want symmetric distances.
        This averages RSSI(A->B) with RSSI(B->A) for each pair.

        Returns:
            Tuple of (symmetric_rssi_matrix, mac_list)
        """
        rssi_matrix, mac_list = self.get_link_rssi_matrix()
        n = len(mac_list)

        symmetric = np.full((n, n), np.nan, dtype=np.float64)
        np.fill_diagonal(symmetric, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                rssi_ij = rssi_matrix[i, j]
                rssi_ji = rssi_matrix[j, i]

                if np.isnan(rssi_ij) and np.isnan(rssi_ji):
                    continue
                elif np.isnan(rssi_ij):
                    avg = rssi_ji
                elif np.isnan(rssi_ji):
                    avg = rssi_ij
                else:
                    avg = (rssi_ij + rssi_ji) / 2.0

                symmetric[i, j] = avg
                symmetric[j, i] = avg

        return symmetric, mac_list

    def get_link_stats(self) -> Dict[Tuple[str, str], LinkRSSIStats]:
        """Get raw link statistics for debugging."""
        return dict(self._links)

    def get_discovered_macs(self) -> Set[str]:
        """Get set of discovered MAC addresses."""
        return set(self._discovered_macs)
