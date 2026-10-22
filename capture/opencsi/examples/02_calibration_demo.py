#!/usr/bin/env python3
"""Demo: baseline learning followed by motion detection.

First 80 frames are quiet (baseline learning), then frames 80-99 have
motion injected. Shows how Z-scores change when motion is present.
"""

from opencsi import AutoCalibrator, AutoCalibratorConfig
from opencsi.testing import generate_synthetic_csi

QUIET_FRAMES = 80
TOTAL_FRAMES = 100
MOTION_MAGNITUDE = 5.0

# Synthetic data has high spectral entropy; relax threshold for quiet detection
config = AutoCalibratorConfig(entropy_low_threshold=6.0)

# Generate dataset: quiet baseline, then motion
positions, frames, timestamps = generate_synthetic_csi(
    n_frames=TOTAL_FRAMES,
    n_nodes=4,
    motion_frames=range(QUIET_FRAMES, TOTAL_FRAMES),
    motion_magnitude=MOTION_MAGNITUDE,
    seed=42,
)

cal = AutoCalibrator(positions, config=config)

print("--- Baseline Learning Phase ---")
for i, (packets, ts) in enumerate(zip(frames, timestamps)):
    result = cal.ingest(packets, timestamp=ts)

    if i == QUIET_FRAMES - 1:
        print(f"Frame {i:3d}: max_Z={result.get_max_change():5.2f}  "
              f"reliable_links={result.n_reliable_links}")
        print(f"\n--- Motion Injection Phase (magnitude={MOTION_MAGNITUDE}) ---")

    if i >= QUIET_FRAMES:
        print(f"Frame {i:3d}: max_Z={result.get_max_change():5.2f}  "
              f"motion_links={len(result.get_motion_links(min_reliability=0.1))}")

print(f"\nFinal: motion_detected={result.motion_detected}, "
      f"motion_links={len(result.get_motion_links())}")
