# OpenCSI

**Self-calibrating WiFi CSI preprocessing library with automatic baseline learning.**

OpenCSI transforms raw, hardware-dependent WiFi Channel State Information (CSI) into calibrated, hardware-agnostic Z-scores with reliability metadata. It automatically learns per-link baselines during quiet periods and adapts to different sampling frequencies and environmental conditions.

## Features

- **Automatic Calibration**: Learns baselines during detected quiet periods - no manual calibration required
- **Hardware Agnostic**: Normalizes measurements to Z-scores that are comparable across different hardware
- **Frequency-Aware**: Maintains separate baselines for different sampling rates (2Hz to 60Hz+)
- **Time-of-Day Aware**: Handles diurnal variations in environmental conditions
- **Reliability Scoring**: Provides per-link reliability metrics based on SNR, baseline freshness, and sample rate
- **Persistence**: Save and load calibration state (JSON) for fast startup

## Installation

```bash
pip install opencsi
```

Or install from source:

```bash
git clone https://github.com/opencsi/opencsi.git
cd opencsi
pip install -e .
```

## Quick Start

```python
from opencsi import AutoCalibrator, ESP32Extractor

# Define node positions in your deployment
node_positions = {
    "AA:BB:CC:DD:EE:01": (0.0, 0.0),
    "AA:BB:CC:DD:EE:02": (5.0, 0.0),
    "AA:BB:CC:DD:EE:03": (2.5, 4.0),
}

# Create calibrator
calibrator = AutoCalibrator(
    node_positions=node_positions,
    hardware=ESP32Extractor(),
)

# In your processing loop
for raw_packet in udp_packets:
    # Parse hardware-specific packet
    packet = calibrator.hardware.parse(raw_packet, rx_mac)
    if packet is None:
        continue

    # Ingest and get calibrated output
    frame = calibrator.ingest([packet], timestamp)
    if frame is None:
        continue

    # Use calibrated measurements
    for link_key, link in frame.links.items():
        if link.combined_change > 2.0 and link.reliability > 0.5:
            print(f"Motion detected on {link_key}: {link.combined_change:.1f}σ")

# Save state on shutdown
calibrator.save("calibration_state.json")
```

## Core Concepts

### Calibrated Output

OpenCSI outputs `CalibratedFrame` objects containing:

- **amplitude_change**: Z-score of amplitude deviation from baseline
- **phase_change**: Normalized phase deviation metric
- **combined_change**: SNR-weighted combination (the primary detection metric)
- **reliability**: Link trustworthiness score [0, 1]
- **quality_grade**: Letter grade (A/B/C/F) for quick assessment

### Quiet Detection

The system automatically detects quiet periods using:

- Temporal variance analysis (stability of mean amplitude over time)
- Spectral entropy (distinguishes HVAC from human activity)
- Per-link activity gating (protects baselines from contamination)

### Frequency-Aware Calibration

CSI statistics vary with sampling frequency. OpenCSI maintains separate baselines for 6 frequency buckets:

| Bucket | Range | Use Case |
|--------|-------|----------|
| idle | 0-2 Hz | Sparse sampling |
| background | 2-8 Hz | Low priority links |
| normal | 8-15 Hz | Standard operation |
| elevated | 15-30 Hz | Active tracking |
| focus | 30-60 Hz | High priority |
| burst | 60+ Hz | Maximum rate |

## Deployment Topologies

OpenCSI calibrates any set of TX→RX links. It doesn't care what the endpoints are — ESP32 nodes, routers, access points, laptops. If a device produces or receives CSI packets, it's a valid endpoint. What you *do* with the calibrated Z-scores (motion detection, localization, gesture recognition) is up to your application.

### Single Node + Router

A single ESP32 paired with a WiFi router produces one calibrated link. OpenCSI will learn the baseline, compute Z-scores, and track reliability — just like any other link. Your application decides what a Z-score spike means.

```python
node_positions = {
    "AA:BB:CC:DD:EE:01": (0.0, 0.0),   # ESP32 sensor
    "FF:EE:DD:CC:BB:00": (3.5, 0.0),   # Router
}
calibrator = AutoCalibrator(node_positions)
```

### Scaling

Links are calibrated independently. More links means more data for your application, but OpenCSI's per-link output is the same whether you have 1 link or 156.

| Nodes | Directed Links |
|-------|---------------|
| 1 + router | 1-2 |
| 2 | 2 |
| 3 | 6 |
| 4 | 12 |
| 6 | 30 |
| 13 | 156 |

### Asymmetric Links

Links are **directed** — `A→B` is calibrated independently from `B→A`. If your protocol only produces one direction (e.g., nodes transmit to a central receiver), each link is still fully calibrated.

### Node Positions

`node_positions` maps MAC addresses to (x, y) coordinates. These are used internally for distance-based reliability weighting. If a TX or RX MAC is *not* in `node_positions`, the link still gets calibrated Z-scores but distance-dependent quality factors are unavailable.

## Hardware Support

### ESP32 (Built-in)

```python
from opencsi.hardware import ESP32Extractor

extractor = ESP32Extractor()
packet = extractor.parse(raw_bytes, rx_mac)
```

Supports:
- v1 protocol (128-byte packets, magic 0xC5 0x02)
- v2 protocol (148-byte packets, magic 0xC5 0x03)
- v5 batch protocol (variable length, magic 0xC5 0x07)

### Custom Hardware

Implement the `CSIExtractor` interface:

```python
from opencsi.hardware import CSIExtractor, CSIPacket

class MyHardwareExtractor(CSIExtractor):
    @property
    def hardware_type(self) -> str:
        return "my_hardware"

    @property
    def num_subcarriers(self) -> int:
        return 52

    def parse(self, data: bytes, rx_mac: str) -> Optional[CSIPacket]:
        # Parse your hardware's packet format
        ...

    def get_default_profile(self) -> dict:
        return {"typical_snr_db": 25, ...}
```

## API Reference

### AutoCalibrator

The main entry point for CSI calibration.

```python
AutoCalibrator(
    node_positions: Dict[str, Tuple[float, float]],
    config: Optional[AutoCalibratorConfig] = None,
    hardware: Optional[CSIExtractor] = None,
    hardware_profile: Optional[dict] = None,
)
```

Key methods:

- `ingest(packets, timestamp)` → `CalibratedFrame`: Process raw packets
- `save(path)` / `load(path)`: Persist calibration state

### CalibratedFrame

Output from calibration containing all per-link measurements.

```python
frame.links: Dict[Tuple[str, str], CalibratedLink]  # Core output
frame.calibration_maturity: float   # 0-1, system readiness
frame.n_reliable_links: int         # Links above reliability threshold
frame.mean_reliability: float       # Average link reliability
```

### CalibratedLink

Per-link calibrated measurements.

```python
link.combined_change: float  # Primary detection metric (Z-score)
link.reliability: float      # Link trustworthiness [0, 1]
link.quality_grade: QualityGrade  # A/B/C/F
link.snr_estimate: float     # Signal-to-noise ratio (dB)
link.baseline_age_s: float   # Seconds since baseline update
```

## Architecture

```
opencsi/
├── core/           # AutoCalibrator, types, config
├── sanitization/   # AGC correction, phase correction, outliers
├── baseline/       # Frequency and time-aware baselines
├── hardware/       # CSIExtractor ABC, ESP32 implementation
├── stats/          # Circular statistics
└── testing/        # Synthetic data generation for tests
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Citation

If you use OpenCSI in your research, please cite:

```bibtex
@software{opencsi,
  title = {OpenCSI: Self-calibrating WiFi CSI Preprocessing Library},
  year = {2024},
  url = {https://github.com/opencsi/opencsi}
}
```
