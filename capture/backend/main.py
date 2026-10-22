"""CSI-Sense entrypoint — wires all components and starts the server."""
import asyncio
import logging
import signal
import sys
import time

import uvicorn

from config import UDP_PORT, WEB_PORT
from receiver import CSIReceiver
from registry import NodeRegistry
from scheduler import TDMAScheduler
from detector import PresenceDetector
from recorder import CSIRecorder
from labeler import LabelStore
from init_data import init_data_dir
from server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("csi-sense")


async def main():
    init_data_dir("data")

    registry = NodeRegistry()
    receiver = CSIReceiver()
    detector = PresenceDetector()
    scheduler = TDMAScheduler(
        get_online_nodes=registry.get_online_nodes,
        get_calibration_weights=detector.get_calibration_weights,
    )
    recorder = CSIRecorder(base_dir="data", registry=registry)
    labeler = LabelStore(base_dir="data", recorder=recorder)

    detector.set_recorder(recorder)

    async def on_hello(hello_pkt):
        await registry.register_node(hello_pkt.mac, hello_pkt.ip,
                                     hello_pkt.chip, hello_pkt.version)
        await registry.update_clock_offset(hello_pkt.mac, hello_pkt.uptime_s)

    async def on_heartbeat(hb_pkt):
        await registry.update_heartbeat(hb_pkt.mac, hb_pkt.uptime_s, hb_pkt.free_heap)
        await registry.update_clock_offset(hb_pkt.mac, hb_pkt.uptime_s)

    def on_csi(sample):
        # Compute wall_ts once, share with detector and recorder.
        # firmware_ts_us in v5 batches is a truncated 16-bit relative timestamp,
        # not the full monotonic uptime. We can only use clock_offset anchoring
        # when firmware_ts_us is a plausible absolute uptime (> 1 second = 1e6 µs).
        server_ts = time.time()
        clock_offset = registry.get_clock_offset(sample.tx_mac)
        if (clock_offset is not None and sample.timestamp_us > 1_000_000):
            wall_ts = clock_offset + (sample.timestamp_us / 1e6)
        else:
            wall_ts = server_ts

        if sample.channel:
            registry.update_rf_info(sample.rx_mac, sample.channel, sample.bandwidth)
        detector.on_csi_sample(sample, wall_ts=wall_ts)
        if recorder.recording:
            recorder.record_raw(sample, wall_ts=wall_ts, server_ts=server_ts)

    receiver.on_hello(on_hello)
    receiver.on_heartbeat(on_heartbeat)
    receiver.on_csi(on_csi)
    receiver.set_slot_resolver(scheduler.get_slot_tx_mac)

    app = create_app(registry, detector, scheduler, receiver,
                     recorder=recorder, labeler=labeler)

    await registry.start()
    await receiver.start()
    await scheduler.start()
    await recorder.start()

    shutdown_event = asyncio.Event()

    def handle_signal(sig):
        logger.info(f"Received {sig.name}, shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal, sig)

    config = uvicorn.Config(app, host="0.0.0.0", port=WEB_PORT, log_level="info")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    logger.info(f"CSI-Sense running — UDP:{UDP_PORT} HTTP:{WEB_PORT}")

    await shutdown_event.wait()

    server.should_exit = True
    await server_task
    await recorder.stop()
    await scheduler.stop()
    await receiver.stop()
    await registry.stop()
    logger.info("CSI-Sense stopped")


if __name__ == "__main__":
    asyncio.run(main())
