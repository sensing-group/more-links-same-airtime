"""Async UDP receiver for ESP32 CSI packets."""
import asyncio
import logging
from typing import Callable, Dict, Optional

from config import UDP_PORT, BUFFER_SIZE
from protocol import parse_packet

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TASKS = 100


class UDPProtocol(asyncio.DatagramProtocol):
    """Asyncio protocol that dispatches incoming datagrams to CSIReceiver."""

    def __init__(self, receiver: "CSIReceiver"):
        self.receiver = receiver
        self.transport = None
        self._active_tasks: set = set()
        self._dropped = 0

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple):
        if len(self._active_tasks) >= MAX_CONCURRENT_TASKS:
            self._dropped += 1
            if self._dropped % 1000 == 1:
                logger.warning(
                    "Task queue full, dropped %d total", self._dropped
                )
            return
        task = asyncio.create_task(self._handle(data, addr))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _handle(self, data: bytes, addr: tuple):
        try:
            await self.receiver._handle_packet(data, addr)
        except Exception:
            logger.error("Packet error from %s", addr, exc_info=True)

    def error_received(self, exc):
        logger.error("UDP error: %s", exc)


class CSIReceiver:
    """Async UDP server that receives and dispatches ESP32 CSI packets.

    Supports HELLO, HEARTBEAT, and CSI packet types via registered callbacks.
    """

    def __init__(self):
        self.transport = None
        self._on_hello: list[Callable] = []
        self._on_heartbeat: list[Callable] = []
        self._on_csi: list[Callable] = []
        self._ip_to_mac: Dict[str, str] = {}
        self._slot_resolver: Optional[Callable] = None
        # Stats
        self.packets_received = 0
        self.packets_parsed = 0

    async def start(self, host: str = "0.0.0.0", port: int = UDP_PORT):
        """Start listening for UDP packets."""
        loop = asyncio.get_event_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self), local_addr=(host, port)
        )
        logger.info("CSI Receiver listening on %s:%d", host, port)

    async def stop(self):
        """Stop the receiver and close the transport."""
        if self.transport:
            self.transport.close()
        logger.info("CSI Receiver stopped")

    def on_hello(self, callback: Callable):
        """Register a callback for HELLO packets."""
        self._on_hello.append(callback)

    def on_heartbeat(self, callback: Callable):
        """Register a callback for HEARTBEAT packets."""
        self._on_heartbeat.append(callback)

    def on_csi(self, callback: Callable):
        """Register a callback for CSI data packets."""
        self._on_csi.append(callback)

    def set_slot_resolver(self, resolver: Callable):
        """Set the TDMA slot resolver used during packet parsing."""
        self._slot_resolver = resolver

    def register_ip_mac(self, ip: str, mac: str):
        """Register IP-to-MAC mapping (called when HELLO received)."""
        self._ip_to_mac[ip] = mac

    def get_stats(self) -> Dict[str, int]:
        """Return receiver statistics."""
        return {
            "packets_received": self.packets_received,
            "packets_parsed": self.packets_parsed,
            "known_nodes": len(self._ip_to_mac),
        }

    async def _handle_packet(self, data: bytes, addr: tuple):
        """Parse and dispatch a single packet to registered callbacks."""
        ip = addr[0]
        self.packets_received += 1

        ptype, parsed = parse_packet(
            data,
            addr,
            self._ip_to_mac,
            self._slot_resolver or (lambda x: None),
        )
        if ptype is None:
            return
        self.packets_parsed += 1

        if ptype == "hello":
            self.register_ip_mac(ip, parsed.mac)
            await self._invoke_callbacks(self._on_hello, parsed)
        elif ptype == "heartbeat":
            await self._invoke_callbacks(self._on_heartbeat, parsed)
        elif ptype == "csi":
            samples = parsed if isinstance(parsed, list) else [parsed]
            for sample in samples:
                await self._invoke_callbacks(self._on_csi, sample)

    @staticmethod
    async def _invoke_callbacks(callbacks: list[Callable], arg):
        """Invoke a list of callbacks, handling both sync and async."""
        for cb in callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(arg)
            else:
                cb(arg)
