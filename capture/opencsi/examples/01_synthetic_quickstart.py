#!/usr/bin/env python3
"""Quickstart: generate synthetic CSI, calibrate, and inspect results."""

from opencsi import AutoCalibrator, AutoCalibratorConfig
from opencsi.testing import generate_synthetic_csi

# Synthetic data has high spectral entropy; relax threshold for quiet detection
config = AutoCalibratorConfig(entropy_low_threshold=6.0)

# Generate 100 frames from 4 nodes (no motion)
positions, frames, timestamps = generate_synthetic_csi(n_frames=100, n_nodes=4)

# Create calibrator and feed all frames
cal = AutoCalibrator(positions, config=config)
for packets, ts in zip(frames, timestamps):
    result = cal.ingest(packets, timestamp=ts)

# Inspect the last frame
print(f"Links: {result.n_total_links}  Reliable: {result.n_reliable_links}")
print(f"Max Z-score: {result.get_max_change():.2f}")
print(f"Motion detected: {result.motion_detected}")

for key, link in list(result.links.items())[:3]:
    print(f"  {link.tx_mac[-5:]}→{link.rx_mac[-5:]}  "
          f"Z={link.combined_change:.2f}  R={link.reliability:.3f}  "
          f"Q={link.quality_grade.value}")
