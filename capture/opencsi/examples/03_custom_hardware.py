#!/usr/bin/env python3
"""Example: implementing a custom CSI hardware extractor.

Shows how to subclass CSIExtractor for a new hardware platform and
feed parsed packets into the calibrator via RawCSIPacket.from_csi_packet().
"""

from typing import Optional

import numpy as np

from opencsi import AutoCalibrator, AutoCalibratorConfig, CSIExtractor, CSIPacket
from opencsi.core.types import RawCSIPacket
from opencsi.hardware.capabilities import (
    HardwareCapabilities,
    SubcarrierConfig,
)
from opencsi.testing import make_positions

N_SUBCARRIERS = 64


class MyCustomExtractor(CSIExtractor):
    """Example extractor for a hypothetical 64-subcarrier radio."""

    @property
    def capabilities(self) -> HardwareCapabilities:
        return HardwareCapabilities(
            hardware_type="my_custom_radio",
            supported_bandwidths=[20],
            supported_bands=["2.4GHz"],
            subcarrier_configs={
                20: SubcarrierConfig(
                    num_subcarriers=N_SUBCARRIERS,
                    pilot_indices=(7, 21, 43, 57),
                    dc_index=32,
                ),
            },
        )

    def parse(self, data: bytes, rx_mac: Optional[str] = None) -> Optional[CSIPacket]:
        """Parse 264-byte raw payload into CSIPacket.

        Format: [tx_mac:6][rssi:1][channel:1][IQ:64*2*2=256 bytes]
        (simplified for this example)
        """
        if len(data) < 264:
            return None

        tx_mac_bytes = data[:6]
        tx_mac = ":".join(f"{b:02X}" for b in tx_mac_bytes)
        rssi = int.from_bytes(data[6:7], "little", signed=True)
        channel = data[7]

        iq_raw = np.frombuffer(data[8:8 + N_SUBCARRIERS * 4], dtype=np.int16)
        iq_data = iq_raw.reshape(N_SUBCARRIERS, 2).astype(np.float32)

        return CSIPacket(
            tx_mac=tx_mac,
            rx_mac=rx_mac or "00:00:00:00:00:00",
            timestamp_us=0,
            rssi=rssi,
            noise_floor=-90,
            channel=channel,
            bandwidth_mhz=20,
            n_subcarriers=N_SUBCARRIERS,
            iq_data=iq_data,
            hardware_type="my_custom_radio",
        )

    def get_default_profile(self) -> dict:
        return {
            "typical_noise_floor_dbm": -90,
            "typical_amplitude_sigma": 2.5,
            "typical_phase_sigma_rad": 0.3,
            "typical_snr_db": 30,
        }


# --- Demo: use the custom extractor with AutoCalibrator ---

extractor = MyCustomExtractor()
print(f"Hardware: {extractor.hardware_type}")
print(f"Subcarriers: {extractor.num_subcarriers}")
print(f"Pilots: {extractor.get_pilot_indices()}")

# Set up calibrator with the extractor's default profile
positions = make_positions(n_nodes=3)
cal = AutoCalibrator(
    positions,
    hardware_profile=extractor.get_default_profile(),
)

# Simulate receiving a raw packet, parsing it, and feeding to calibrator
rng = np.random.default_rng(42)
nodes = list(positions.keys())

# Build a fake raw byte payload matching the parse format
tx_mac_bytes = bytes.fromhex(nodes[0].replace(":", ""))
rssi_byte = (-50 & 0xFF).to_bytes(1, "little")
channel_byte = (6).to_bytes(1, "little")
iq_bytes = rng.normal(10, 2, (N_SUBCARRIERS, 2)).astype(np.int16).tobytes()
raw_data = tx_mac_bytes + rssi_byte + channel_byte + iq_bytes

# Parse and feed to calibrator
packet = extractor.parse(raw_data, rx_mac=nodes[1])
if packet:
    raw_pkt = RawCSIPacket.from_csi_packet(packet)
    frame = cal.ingest({(raw_pkt.tx_mac, raw_pkt.rx_mac): raw_pkt}, timestamp=1.0)
    print(f"\nIngested packet: {frame.n_total_links} link(s)")
    print("Custom extractor registered and working.")
else:
    print("Parse failed (unexpected)")
