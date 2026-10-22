"""Zone-based round-robin TDMA scheduler."""
import asyncio
import logging
import math
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    BROADCAST_ADDR,
    BURSTS_PER_TRIGGER,
    CYCLE_DELAY_MS,
    DEFAULT_ROOM_HEIGHT,
    DEFAULT_ROOM_WIDTH,
    MIN_NODES_TO_START,
    SLOT_DURATION_MS,
    TRIGGER_PORT,
    ZONE_MAX_SIZE,
)
from protocol import TriggerPacket

logger = logging.getLogger(__name__)

# --- Module-level constants ---
NODE_WAIT_INTERVAL_S = 1.0
SLOT_ID_MAX = 255
SCHEDULER_ERROR_BACKOFF_S = 0.5


@dataclass
class Zone:
    """A spatial grouping of nodes for round-robin scheduling."""

    zone_id: int
    node_macs: List[str] = field(default_factory=list)
    centroid: Tuple[float, float] = (0.0, 0.0)

    def __len__(self) -> int:
        return len(self.node_macs)


class TDMAScheduler:
    """Zone-based round-robin TDMA scheduler.

    Zone assignment (greedy position clustering):
    1. Estimate zone count from node count: zones = ceil(N / ZONE_MAX_SIZE)
    2. Sort nodes by spatial position (x then y), split into contiguous chunks
    3. Cap each zone at ZONE_MAX_SIZE
    4. Fallback: single zone if no positions available

    Scheduling loop:
    - For each zone: trigger each node sequentially, wait SLOT_DURATION_MS between
    - After all zones complete one cycle: wait CYCLE_DELAY_MS before next cycle
    """

    def __init__(self, get_online_nodes, get_calibration_weights=None):
        """
        Args:
            get_online_nodes: Callable returning List[NodeInfo] (each has .mac, .position).
            get_calibration_weights: Optional callable returning Dict[str, int] of
                mac -> slot multiplier for adaptive calibration (0 = skip, >1 = extra slots).
        """
        self._get_online_nodes = get_online_nodes
        self._get_calibration_weights = get_calibration_weights
        self._zones: List[Zone] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sock: Optional[socket.socket] = None
        self._seq = 0
        # Slot ID tracking for v5 batch resolution
        self._slot_id_map: Dict[int, str] = {}  # slot_id -> tx_mac
        self._mac_to_slot: Dict[str, int] = {}  # tx_mac -> slot_id
        self._next_slot_id = 0
        # Room dimensions for zone computation
        self.room_width = DEFAULT_ROOM_WIDTH
        self.room_height = DEFAULT_ROOM_HEIGHT

    # ---- Lifecycle ----

    async def start(self):
        """Start the scheduling loop."""
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setblocking(False)
        self._broadcast_addr = (BROADCAST_ADDR, TRIGGER_PORT)
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TDMA Scheduler started")

    async def stop(self):
        """Stop the scheduling loop and release resources."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._sock:
            self._sock.close()
            self._sock = None
        logger.info("TDMA Scheduler stopped")

    # ---- Slot ID management ----

    def get_slot_tx_mac(self, slot_id: int) -> Optional[str]:
        """Resolve slot_id to tx_mac (for v5 batch CSI)."""
        return self._slot_id_map.get(slot_id)

    def _assign_slot_id(self, mac: str) -> int:
        """Assign a stable slot_id to a MAC, reusing existing assignment."""
        if mac in self._mac_to_slot:
            return self._mac_to_slot[mac]
        sid = self._next_slot_id % (SLOT_ID_MAX + 1)
        self._next_slot_id += 1
        self._slot_id_map[sid] = mac
        self._mac_to_slot[mac] = sid
        return sid

    # ---- Zone computation ----

    def update_zones(self):
        """Recompute zones from current node positions.

        Positioned nodes are sorted spatially and split into contiguous chunks
        of at most ZONE_MAX_SIZE.  Unpositioned nodes are distributed to the
        smallest existing zones.
        """
        nodes = self._get_online_nodes()

        # Save old zone state for comparison
        old_zone_signature = self._get_zone_signature()

        if not nodes:
            self._zones = []
            if old_zone_signature is not None:
                logger.info("Zones cleared (no online nodes)")
            return

        positioned = [(n.mac, n.position) for n in nodes if n.position]
        unpositioned = [n.mac for n in nodes if not n.position]

        if not positioned:
            zone = Zone(zone_id=0, node_macs=[n.mac for n in nodes])
            self._zones = [zone]
            new_zone_signature = self._get_zone_signature()
            if old_zone_signature != new_zone_signature:
                logger.info("Single zone with %d nodes (no positions)", len(nodes))
            return

        self._zones = self._cluster_positioned(positioned)
        self._distribute_unpositioned(unpositioned)

        # Only log if zones actually changed
        new_zone_signature = self._get_zone_signature()
        if old_zone_signature != new_zone_signature:
            logger.info(
                "Zones updated: %d zones, %s nodes each",
                len(self._zones),
                [len(z) for z in self._zones],
            )

    def _get_zone_signature(self) -> Optional[Tuple]:
        """Get a hashable signature of current zone state for change detection."""
        if not self._zones:
            return None
        # Signature based purely on node groupings, independent of zone IDs or order
        return tuple(sorted(tuple(sorted(z.node_macs)) for z in self._zones))

    def _cluster_positioned(
        self, positioned: List[Tuple[str, Tuple[float, float]]]
    ) -> List[Zone]:
        """Sort positioned nodes spatially and split into zone-sized chunks."""
        n_zones = max(1, math.ceil(len(positioned) / ZONE_MAX_SIZE))
        sorted_nodes = sorted(positioned, key=lambda item: (item[1][0], item[1][1]))

        zones: List[Zone] = []
        chunk_size = math.ceil(len(sorted_nodes) / n_zones)
        for i in range(n_zones):
            chunk = sorted_nodes[i * chunk_size : (i + 1) * chunk_size]
            if not chunk:
                continue
            macs = [mac for mac, _ in chunk]
            positions = [pos for _, pos in chunk]
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            zones.append(Zone(zone_id=i, node_macs=macs, centroid=(cx, cy)))
        return zones

    def _distribute_unpositioned(self, unpositioned: List[str]):
        """Assign unpositioned nodes to the smallest existing zone."""
        for mac in unpositioned:
            smallest = min(self._zones, key=len)
            smallest.node_macs.append(mac)

    # ---- Scheduling loop ----

    async def _run_loop(self):
        """Main scheduling loop: cycle through zones, triggering each node."""
        while self._running:
            try:
                nodes = self._get_online_nodes()
                if len(nodes) < MIN_NODES_TO_START:
                    logger.debug(
                        "Waiting for nodes (%d/%d)", len(nodes), MIN_NODES_TO_START
                    )
                    await asyncio.sleep(NODE_WAIT_INTERVAL_S)
                    continue

                self.update_zones()
                if not self._zones:
                    await asyncio.sleep(NODE_WAIT_INTERVAL_S)
                    continue

                # Get adaptive weights during calibration
                weights = {}
                if self._get_calibration_weights:
                    weights = self._get_calibration_weights()

                if weights:
                    await self._trigger_cycle_weighted(weights)
                else:
                    await self._trigger_cycle()

                await asyncio.sleep(CYCLE_DELAY_MS / 1000.0)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler error")
                await asyncio.sleep(SCHEDULER_ERROR_BACKOFF_S)

    async def _trigger_cycle(self):
        """Trigger every node in every zone once.

        Each trigger is broadcast to ALL nodes.  The tx_mac field inside
        the packet tells nodes which one should transmit; every other
        node enters RX mode and collects CSI.
        """
        for zone in self._zones:
            if not self._running:
                break
            for mac in zone.node_macs:
                if not self._running:
                    break
                slot_id = self._assign_slot_id(mac)
                await self._send_trigger(mac, slot_id)
                await asyncio.sleep(SLOT_DURATION_MS / 1000.0)

    async def _trigger_cycle_weighted(self, weights: Dict[str, int]):
        """Trigger cycle with per-node slot multipliers for adaptive calibration.

        Nodes with weight 0 are skipped entirely. Nodes with weight >1
        get that many consecutive TX slots per cycle.
        """
        for zone in self._zones:
            if not self._running:
                break
            for mac in zone.node_macs:
                if not self._running:
                    break
                n_slots = weights.get(mac, 1)
                if n_slots <= 0:
                    continue  # fully calibrated, skip
                slot_id = self._assign_slot_id(mac)
                for _ in range(n_slots):
                    if not self._running:
                        break
                    await self._send_trigger(mac, slot_id)
                    await asyncio.sleep(SLOT_DURATION_MS / 1000.0)

    async def _send_trigger(self, mac: str, slot_id: int):
        """Broadcast a UDP trigger packet to all nodes."""
        self._seq += 1
        trigger = TriggerPacket(
            tx_mac=mac,
            seq=self._seq,
            burst_count=BURSTS_PER_TRIGGER,
            slot_ms=SLOT_DURATION_MS,
            slot_id=slot_id,
        )
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._sock.sendto, trigger.to_bytes(), self._broadcast_addr
            )
        except Exception as exc:
            logger.debug("Trigger broadcast failed for %s: %s", mac, exc)

    # ---- Public query API ----

    def get_zones_info(self) -> List[dict]:
        """Return zone metadata for API / WebSocket consumers."""
        return [
            {
                "zone_id": z.zone_id,
                "nodes": z.node_macs,
                "centroid": z.centroid,
            }
            for z in self._zones
        ]
