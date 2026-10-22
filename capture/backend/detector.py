"""Presence detection via CSI amplitude variance Z-scores."""
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from config import (
    ACTIVE_LINK_THRESHOLD,
    BASELINE_LEARNING_SAMPLES,
    BURSTS_PER_TRIGGER,
    CYCLE_DELAY_MS,
    MIN_LINK_RATE_HZ,
    MOTION_ZSCORE_THRESHOLD,
    READINESS_QUORUM_FRACTION,
    READINESS_QUORUM_MIN_LINKS,
    SLOT_DURATION_MS,
)

logger = logging.getLogger(__name__)

WELFORD_MIN_SAMPLES = 10
VARIANCE_FLOOR = 1e-9
RATE_WINDOW_S = 5.0
EWMA_ALPHA = 0.2  # ~5-sample effective window (~100ms at 50Hz per link)

# Throttle the calibration-completeness check (O(n_links) per call).
CALIBRATION_CHECK_INTERVAL_S = 1.0


class DetectorState(Enum):
    """Detector lifecycle states."""

    CALIBRATING = "calibrating"
    MONITORING = "monitoring"


@dataclass
class LinkBaseline:
    """Online statistics for a single TX->RX link's amplitude variance.

    Uses Welford's algorithm for numerically stable running mean/variance.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Running sum of squared deviations

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def update(self, value: float):
        """Welford's online update: incorporate a new sample."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2


class PresenceDetector:
    """Per-link CSI variance -> Z-score -> occupancy detection.

    Detection pipeline:
    1. CSI sample -> amplitude = sqrt(I^2 + Q^2) -> variance across subcarriers
    2. Baseline phase: accumulate variance per link using Welford's algorithm
    3. Detection: Z = (recent_variance - baseline_mean) / baseline_std
    4. Link is active if |Z| > MOTION_ZSCORE_THRESHOLD
    5. Room is occupied if active_links >= ACTIVE_LINK_THRESHOLD
    """

    def __init__(self):
        self._baselines: Dict[Tuple[str, str], LinkBaseline] = {}
        self._ewma_variance: Dict[Tuple[str, str], float] = {}
        self._state = DetectorState.CALIBRATING
        self._total_samples = 0
        self._calibration_samples = 0
        self._recent_packets: List[dict] = []
        self._max_packet_history = 100
        self._recorder = None
        # Calibration timing
        self._calibration_start: float = time.time()
        self._calibration_duration: float | None = None
        self._baseline_frozen: bool = False
        # Per-node and per-link timestamps for rate calculation
        self._tx_timestamps: Dict[str, List[float]] = defaultdict(list)
        self._rx_timestamps: Dict[str, List[float]] = defaultdict(list)
        self._link_timestamps: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        # Measured cycle rate tracking
        self._cycle_timestamps: List[float] = []
        self._measured_cycle_hz: float = 0.0
        # Throttle for calibration-complete check (called from hot path)
        self._last_calibration_check: float = 0.0
        # Simulation: inject a virtual occupant at (x, y) metres
        self._simulation: dict | None = None

    def set_recorder(self, recorder) -> None:
        self._recorder = recorder

    def on_csi_sample(self, sample, wall_ts: float = None):
        """Process a single CSI sample.

        Args:
            sample: Object with .tx_mac, .rx_mac, .amplitude (ndarray of shape (N,)).
            wall_ts: Reconstructed wall-clock timestamp (from clock offset + firmware_ts).
        """
        if wall_ts is None:
            wall_ts = time.time()

        link_key = (sample.tx_mac, sample.rx_mac)

        now = time.time()
        self._tx_timestamps[sample.tx_mac].append(now)
        self._rx_timestamps[sample.rx_mac].append(now)
        self._link_timestamps[link_key].append(now)

        amp_variance = float(np.var(sample.amplitude))
        prev = self._ewma_variance.get(link_key, amp_variance)
        self._ewma_variance[link_key] = EWMA_ALPHA * amp_variance + (1 - EWMA_ALPHA) * prev

        if link_key not in self._baselines:
            self._baselines[link_key] = LinkBaseline()

        baseline = self._baselines[link_key]
        if not (self._baseline_frozen and self._state == DetectorState.MONITORING):
            baseline.update(amp_variance)

        z_score = None
        if self._state == DetectorState.MONITORING and baseline.count >= WELFORD_MIN_SAMPLES and baseline.std >= VARIANCE_FLOOR:
            z_score = (self._ewma_variance[link_key] - baseline.mean) / baseline.std

        packet_info = {
            "tx_mac": sample.tx_mac,
            "rx_mac": sample.rx_mac,
            "timestamp_us": sample.timestamp_us,
            "seq": sample.seq,
            "rssi": sample.rssi,
            "noise_floor": sample.noise_floor,
            "amplitude_variance": round(amp_variance, 4),
            "baseline_mean": round(baseline.mean, 4),
            "baseline_std": round(baseline.std, 4),
            "baseline_samples": baseline.count,
            "z_score": round(z_score, 2) if z_score is not None else None,
            "state": self._state.value,
            "wall_ts": wall_ts,
        }

        self._recent_packets.append(packet_info)
        if len(self._recent_packets) > self._max_packet_history:
            self._recent_packets.pop(0)

        if self._recorder is not None and self._recorder.recording:
            self._recorder.record_calibrated(packet_info, wall_ts=now)

        self._total_samples += 1
        if self._state == DetectorState.CALIBRATING:
            self._check_calibration_complete()

    def _check_calibration_complete(self):
        """Transition to MONITORING once a quorum of viable links is ready.

        A global min(count) gate would be held hostage by any single marginal
        link; instead, we only consider links whose recent rate is above the
        viability floor, and flip state when READINESS_QUORUM_FRACTION of them
        have reached BASELINE_LEARNING_SAMPLES.

        Called from the hot path (on_csi_sample) so throttled to once per
        CALIBRATION_CHECK_INTERVAL_S.
        """
        if not self._baselines:
            return
        now = time.time()
        if now - self._last_calibration_check < CALIBRATION_CHECK_INTERVAL_S:
            return
        self._last_calibration_check = now

        viable_keys = [k for k in self._baselines if self._link_is_viable(k)]
        if len(viable_keys) < READINESS_QUORUM_MIN_LINKS:
            return
        ready_count = sum(
            1 for k in viable_keys
            if self._baselines[k].count >= BASELINE_LEARNING_SAMPLES
        )
        ready_fraction = ready_count / len(viable_keys)
        if ready_fraction >= READINESS_QUORUM_FRACTION:
            self._state = DetectorState.MONITORING
            self._calibration_duration = now - self._calibration_start
            self._calibration_samples = self._total_samples
            logger.info(
                "Calibration complete. %d/%d viable links ready (%.0f%%) in %.1fs. "
                "%d non-viable links excluded.",
                ready_count, len(viable_keys), ready_fraction * 100,
                self._calibration_duration,
                len(self._baselines) - len(viable_keys),
            )

    def get_active_links(self) -> List[dict]:
        """Return list of active links with their Z-scores.

        Returns empty list during calibration phase.
        """
        if self._state != DetectorState.MONITORING:
            return []
        active: List[dict] = []
        for link_key, baseline in self._baselines.items():
            if baseline.count < WELFORD_MIN_SAMPLES or baseline.std < VARIANCE_FLOOR:
                continue
            # Exclude dead/marginal links from detection — they can't
            # contribute useful evidence.
            if not self._link_is_viable(link_key):
                continue
            tx, rx = link_key
            recent = self._ewma_variance.get(link_key, baseline.mean)
            z = (recent - baseline.mean) / baseline.std
            if abs(z) > MOTION_ZSCORE_THRESHOLD:
                active.append({"tx": tx, "rx": rx, "z_score": round(z, 2)})
        return active

    def get_state(self) -> dict:
        """Get current detection state for API / WebSocket consumers."""
        active = self.get_active_links()
        occupied = len(active) >= ACTIVE_LINK_THRESHOLD
        progress = self._calibration_progress()

        return {
            "state": self._state.value,
            "occupied": occupied,
            "occupancy": 1 if occupied else 0,
            "active_links": len(active),
            "active_link_details": active,
            "total_links": len(self._baselines),
            "calibration_progress": round(progress, 2),
            "total_samples": self._total_samples,
        }

    def _calibration_progress(self) -> float:
        """Return calibration progress as a fraction in [0.0, 1.0].

        Averaged across viable links so a single dead link cannot pin the
        value to zero. Non-viable links are ignored entirely.
        """
        if self._state != DetectorState.CALIBRATING:
            return 1.0
        if not self._baselines:
            return 0.0
        target = BASELINE_LEARNING_SAMPLES
        viable_progress = [
            min(self._baselines[k].count / target, 1.0)
            for k in self._baselines
            if self._link_is_viable(k)
        ]
        if not viable_progress:
            return 0.0
        return sum(viable_progress) / len(viable_progress)

    def get_recent_packets(self, limit: int = 50) -> List[dict]:
        """Get recent packet information with calibration stats."""
        return self._recent_packets[-limit:]

    def _trim_and_rate(self, timestamps: List[float], cutoff: float) -> float:
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
        return len(timestamps) / RATE_WINDOW_S

    def _link_rate(self, link_key: Tuple[str, str]) -> float:
        """Current sample rate (Hz) for one link over RATE_WINDOW_S."""
        ts = self._link_timestamps.get(link_key)
        if not ts:
            return 0.0
        cutoff = time.time() - RATE_WINDOW_S
        return self._trim_and_rate(ts, cutoff)

    def _link_is_viable(self, link_key: Tuple[str, str]) -> bool:
        """A link is viable if it's currently delivering samples at a useful rate.

        Non-viable links are excluded from calibration quorum, detection, and
        adaptive scheduling — a dead or marginal link cannot block the system.
        """
        return self._link_rate(link_key) >= MIN_LINK_RATE_HZ

    def _link_is_ready(self, link_key: Tuple[str, str]) -> bool:
        """A viable link that has enough baseline samples to score against."""
        b = self._baselines.get(link_key)
        if b is None or b.count < BASELINE_LEARNING_SAMPLES:
            return False
        return self._link_is_viable(link_key)

    def _compute_rates(self) -> tuple:
        """Compute all rate data in a single pass."""
        now = time.time()
        cutoff = now - RATE_WINDOW_S

        tx_rates: Dict[str, float] = {}
        for mac, ts in self._tx_timestamps.items():
            tx_rates[mac] = self._trim_and_rate(ts, cutoff)

        rx_rates: Dict[str, float] = {}
        for mac, ts in self._rx_timestamps.items():
            rx_rates[mac] = self._trim_and_rate(ts, cutoff)

        link_rates: Dict[Tuple[str, str], float] = {}
        for link_key, ts in self._link_timestamps.items():
            link_rates[link_key] = self._trim_and_rate(ts, cutoff)

        return tx_rates, rx_rates, link_rates

    def get_node_rates(self) -> Dict[str, dict]:
        """Return per-node CSI yield and per-link sensing rates."""
        tx_rates, rx_rates, link_rates = self._compute_rates()
        n_nodes = len(set(tx_rates) | set(rx_rates))
        n_peers = max(n_nodes - 1, 1)

        all_macs = sorted(set(tx_rates) | set(rx_rates))
        node_rates: Dict[str, dict] = {}
        for mac in all_macs:
            tx_yield = tx_rates.get(mac, 0.0)
            rx_yield = rx_rates.get(mac, 0.0)
            mac_links = {k: v for k, v in link_rates.items() if mac in k}
            link_vals = list(mac_links.values()) if mac_links else [0.0]
            min_link = min(link_vals)
            avg_link = sum(link_vals) / len(link_vals)
            # Capture rate: what fraction of bursts are received as CSI
            # Expected per-link = BURSTS_PER_TRIGGER × cycles/s
            cycle_s = (n_nodes * SLOT_DURATION_MS + CYCLE_DELAY_MS) / 1000.0
            expected_link = BURSTS_PER_TRIGGER / cycle_s if cycle_s > 0 else 0.0
            capture_rate = avg_link / expected_link if expected_link > 0 else 0.0
            node_rates[mac] = {
                "tx_yield": round(tx_yield, 1),
                "rx_yield": round(rx_yield, 1),
                "avg_link_rate": round(avg_link, 1),
                "min_link_rate": round(min_link, 1),
                "link_count": len(mac_links),
                "capture_rate": round(capture_rate, 2),
            }

        return node_rates

    def get_link_rates(self) -> Dict[str, float]:
        """Return per-link CSI sampling rates (samples/sec)."""
        now = time.time()
        cutoff = now - RATE_WINDOW_S
        rates: Dict[str, float] = {}
        for (tx, rx), ts in self._link_timestamps.items():
            rate = self._trim_and_rate(ts, cutoff)
            rates[f"{tx}->{rx}"] = round(rate, 1)
        return rates

    def get_stats_summary(self) -> dict:
        """Return statistics summary for the stats panel."""
        node_rates = self.get_node_rates()
        link_rates = self.get_link_rates()
        link_values = list(link_rates.values())
        avg_link_rate = sum(link_values) / len(link_values) if link_values else 0.0
        min_link_rate = min(link_values) if link_values else 0.0

        n_nodes = len(node_rates)
        cycle_s = (n_nodes * SLOT_DURATION_MS + CYCLE_DELAY_MS) / 1000.0
        expected_link_rate = BURSTS_PER_TRIGGER / cycle_s if cycle_s > 0 else 0.0
        system_capture_rate = avg_link_rate / expected_link_rate if expected_link_rate > 0 else 0.0

        # Measured cycle rate: each link sees bursts_per_trigger samples per cycle
        measured_cycles_per_sec = 0.0
        if avg_link_rate > 0 and BURSTS_PER_TRIGGER > 0:
            measured_cycles_per_sec = avg_link_rate / BURSTS_PER_TRIGGER

        link_baselines = {}
        viable_count = 0
        ready_count = 0
        for link_key, b in self._baselines.items():
            tx, rx = link_key
            viable = self._link_is_viable(link_key)
            if viable:
                viable_count += 1
                if b.count >= BASELINE_LEARNING_SAMPLES:
                    ready_count += 1
            link_baselines[f"{tx}->{rx}"] = {
                "tx": tx,
                "rx": rx,
                "count": b.count,
                "mean": round(b.mean, 4),
                "std": round(b.std, 4),
                "viable": viable,
            }

        return {
            "calibration_duration_s": round(self._calibration_duration, 2) if self._calibration_duration else None,
            "calibrating": self._state == DetectorState.CALIBRATING,
            "baseline_frozen": self._baseline_frozen,
            "baseline_target": BASELINE_LEARNING_SAMPLES,
            "viable_links": viable_count,
            "ready_links": ready_count,
            "min_link_rate_hz": MIN_LINK_RATE_HZ,
            "readiness_quorum_fraction": READINESS_QUORUM_FRACTION,
            "link_baselines": link_baselines,
            "calibration_elapsed_s": round(time.time() - self._calibration_start, 2) if self._state == DetectorState.CALIBRATING else None,
            "total_links": len(self._baselines),
            "calibration_samples": self._total_samples if self._state == DetectorState.CALIBRATING else self._calibration_samples,
            "total_samples": self._total_samples,
            "node_rates": node_rates,
            "link_rates": link_rates,
            "avg_link_rate": round(avg_link_rate, 1),
            "min_link_rate": round(min_link_rate, 1),
            "expected_link_rate": round(expected_link_rate, 1),
            "system_capture_rate": round(system_capture_rate, 2),
            "tdma": {
                "n_nodes": n_nodes,
                "slot_ms": SLOT_DURATION_MS,
                "cycle_delay_ms": CYCLE_DELAY_MS,
                "bursts_per_trigger": BURSTS_PER_TRIGGER,
                "cycle_ms": round(cycle_s * 1000),
                "cycles_per_sec": round(1.0 / cycle_s, 2) if cycle_s > 0 else 0,
                "measured_cycles_per_sec": round(measured_cycles_per_sec, 2),
            },
        }

    def get_calibration_weights(self) -> Dict[str, int]:
        """Return per-TX-MAC slot multipliers for adaptive calibration.

        Returns a dict of mac -> number of TX slots it should get per cycle.
        - Fully calibrated nodes: 0 slots (skip them)
        - Lagging nodes: proportional to how far behind their worst link is
        - Returns empty dict when not calibrating or no baselines yet.
        """
        if self._state != DetectorState.CALIBRATING or not self._baselines:
            return {}
        target = BASELINE_LEARNING_SAMPLES

        # Find worst (lowest count) link per TX mac, ignoring non-viable
        # links — otherwise a dead pair pulls its TX into permanent slot
        # reinforcement that can't fix anything.
        tx_worst: Dict[str, int] = {}
        for link_key, b in self._baselines.items():
            if not self._link_is_viable(link_key):
                continue
            tx = link_key[0]
            worst = tx_worst.get(tx, target)
            tx_worst[tx] = min(worst, b.count)

        # If no node has finished yet, equal scheduling
        any_done = any(w >= target for w in tx_worst.values())
        if not any_done:
            return {}

        weights: Dict[str, int] = {}
        for mac, worst_count in tx_worst.items():
            if worst_count >= target:
                weights[mac] = 0  # fully calibrated, skip
            else:
                # More slots for nodes further behind (1-4 slots)
                remaining = (target - worst_count) / target
                weights[mac] = max(1, min(4, round(remaining * 4)))
        return weights

    @property
    def baseline_frozen(self) -> bool:
        return self._baseline_frozen

    def toggle_baseline_freeze(self) -> bool:
        """Toggle baseline freeze. Returns new frozen state."""
        self._baseline_frozen = not self._baseline_frozen
        logger.info("Baseline frozen: %s", self._baseline_frozen)
        return self._baseline_frozen

    def reset_baselines(self):
        """Clear all baselines and restart calibration."""
        self._baselines.clear()
        self._ewma_variance.clear()
        self._recent_packets.clear()
        self._tx_timestamps.clear()
        self._rx_timestamps.clear()
        self._link_timestamps.clear()
        self._state = DetectorState.CALIBRATING
        self._total_samples = 0
        self._calibration_samples = 0
        self._calibration_start = time.time()
        self._calibration_duration = None
        self._baseline_frozen = False
        self._last_calibration_check = 0.0
        self._simulation = None
        logger.info("Baselines reset, re-calibrating")

    # --- Simulation ---

    def set_simulation(self, pos: dict | None):
        """Set or clear a virtual occupant position {"x": float, "y": float}."""
        self._simulation = pos

    def get_simulation(self) -> dict | None:
        return self._simulation

    @staticmethod
    def _point_to_segment_dist(
        px: float, py: float,
        ax: float, ay: float, bx: float, by: float,
    ) -> float:
        """Minimum distance from point (px,py) to segment (a,b)."""
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def apply_simulation(
        self,
        node_positions: dict[str, tuple[float, float]],
        fresnel_radius: float = 0.8,
    ):
        """Inflate EWMA variance for links near the simulated occupant.

        Must be called just before get_state() in the broadcast loop so
        the snapshot is consistent.  No-op during calibration.
        """
        sim = self._simulation
        if sim is None or self._state != DetectorState.MONITORING:
            return
        sx, sy = sim["x"], sim["y"]
        for link_key, baseline in self._baselines.items():
            if baseline.count < WELFORD_MIN_SAMPLES or baseline.std < VARIANCE_FLOOR:
                continue
            tx_pos = node_positions.get(link_key[0])
            rx_pos = node_positions.get(link_key[1])
            if tx_pos is None or rx_pos is None:
                continue
            dist = self._point_to_segment_dist(
                sx, sy, tx_pos[0], tx_pos[1], rx_pos[0], rx_pos[1],
            )
            if dist < fresnel_radius:
                factor = 5.0 * (1.0 - dist / fresnel_radius)
                self._ewma_variance[link_key] = baseline.mean + factor * baseline.std
