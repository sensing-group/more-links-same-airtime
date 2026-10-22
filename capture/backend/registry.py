"""Node registry with health monitoring and clock synchronization."""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from config import HEARTBEAT_CHECK_INTERVAL_S, HEARTBEAT_TIMEOUT_S

logger = logging.getLogger(__name__)

CLOCK_EMA_ALPHA = 0.05  # Slow EMA to smooth server-side jitter


class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class NodeInfo:
    """Metadata for a single ESP32 node."""

    mac: str
    ip: str
    chip: str = "ESP32"
    version: str = "1.0"
    status: NodeStatus = NodeStatus.ONLINE
    last_seen: datetime = field(default_factory=datetime.now)
    position: Optional[Tuple[float, float]] = None
    uptime_s: int = 0
    free_heap: int = 0
    channel: int = 0
    bandwidth: str = ""

    # Clock synchronization
    clock_offset: float = 0.0        # server_wall_ts - firmware_uptime_s
    clock_offset_samples: int = 0
    clock_drift_ppm: float = 0.0
    last_sync_ts: float = 0.0


class NodeRegistry:
    """Tracks ESP32 nodes and monitors their health via heartbeats."""

    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None
        self._on_change: list[Callable] = []

    async def start(self):
        """Start the background health-check loop."""
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("Node registry started")

    async def stop(self):
        """Stop the health-check loop."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

    async def register_node(
        self,
        mac: str,
        ip: str,
        chip: str = "ESP32",
        version: str = "1.0",
    ) -> NodeInfo:
        """Register a new node or update an existing one."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                node = self._nodes[mac]
                node.ip = ip
                node.chip = chip
                node.version = version
                node.last_seen = datetime.now()
                node.status = NodeStatus.ONLINE
            else:
                node = NodeInfo(mac=mac, ip=ip, chip=chip, version=version)
                self._nodes[mac] = node
                logger.info("New node: %s (%s) at %s", mac, chip, ip)
        await self._notify_change()
        return node

    async def update_heartbeat(
        self, mac: str, uptime_s: int, free_heap: int
    ):
        """Update heartbeat fields for a node."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                node = self._nodes[mac]
                node.last_seen = datetime.now()
                node.uptime_s = uptime_s
                node.free_heap = free_heap
                if node.status == NodeStatus.OFFLINE:
                    node.status = NodeStatus.ONLINE

    async def update_clock_offset(self, mac: str, uptime_s: int):
        """Recompute clock offset from HELLO/HEARTBEAT uptime field."""
        mac = mac.upper()
        async with self._lock:
            node = self._nodes.get(mac)
            if node is None:
                return

            server_now = time.time()
            new_offset = server_now - uptime_s

            if node.clock_offset_samples == 0:
                node.clock_offset = new_offset
            else:
                # EMA update: smooth out server-side jitter
                prev_offset = node.clock_offset
                node.clock_offset = (
                    (1 - CLOCK_EMA_ALPHA) * prev_offset +
                    CLOCK_EMA_ALPHA * new_offset
                )
                # Estimate drift in ppm from offset change
                dt = server_now - node.last_sync_ts
                if dt > 0 and node.clock_offset_samples > 1:
                    drift_s_per_s = (new_offset - prev_offset) / dt
                    node.clock_drift_ppm = drift_s_per_s * 1e6

            node.clock_offset_samples += 1
            node.last_sync_ts = server_now

    def update_rf_info(self, mac: str, channel: int, bandwidth: str):
        """Update channel and bandwidth from a CSI sample (non-async, called from sync callback)."""
        mac = mac.upper()
        node = self._nodes.get(mac)
        if node and channel > 0:
            node.channel = channel
            node.bandwidth = bandwidth

    def get_clock_offset(self, mac: str) -> Optional[float]:
        """Return the clock offset for a single node, or None if not synced."""
        node = self._nodes.get(mac.upper())
        if node and node.clock_offset_samples > 0:
            return node.clock_offset
        return None

    def get_clock_offsets(self) -> dict:
        """Return {mac: {offset, samples, drift_ppm}} for all synced nodes."""
        return {
            mac: {
                "offset": node.clock_offset,
                "samples": node.clock_offset_samples,
                "drift_ppm": round(node.clock_drift_ppm, 2),
            }
            for mac, node in self._nodes.items()
            if node.clock_offset_samples > 0
        }

    def get_online_nodes(self) -> List[NodeInfo]:
        """Return all nodes currently marked ONLINE."""
        return [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]

    def get_all_nodes(self) -> List[NodeInfo]:
        """Return all registered nodes."""
        return list(self._nodes.values())

    def get_node(self, mac: str) -> Optional[NodeInfo]:
        """Look up a node by MAC address."""
        return self._nodes.get(mac.upper())

    def get_ip_to_mac(self) -> Dict[str, str]:
        """Return a mapping of IP addresses to MAC addresses."""
        return {n.ip: n.mac for n in self._nodes.values()}

    async def set_position(self, mac: str, x: float, y: float) -> bool:
        """Set the physical position of a node. Returns True if found."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                self._nodes[mac].position = (x, y)
                return True
        return False

    def on_change(self, callback: Callable):
        """Register a callback invoked when the node set changes."""
        self._on_change.append(callback)

    async def _notify_change(self):
        """Invoke all registered change callbacks."""
        for cb in self._on_change:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()
            except Exception:
                logger.error("Change callback error", exc_info=True)

    async def _health_loop(self):
        """Periodically check for nodes that have stopped sending heartbeats."""
        while True:
            await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL_S)
            now = datetime.now()
            async with self._lock:
                changed = False
                for node in self._nodes.values():
                    elapsed = (now - node.last_seen).total_seconds()
                    if (
                        elapsed > HEARTBEAT_TIMEOUT_S
                        and node.status == NodeStatus.ONLINE
                    ):
                        node.status = NodeStatus.OFFLINE
                        logger.warning(
                            "Node %s offline (no heartbeat for %.0fs)",
                            node.mac,
                            elapsed,
                        )
                        changed = True
            if changed:
                await self._notify_change()
