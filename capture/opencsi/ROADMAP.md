# OpenCSI Roadmap

This document outlines the development roadmap for OpenCSI, a hardware-agnostic WiFi CSI preprocessing library.

---

## Current Version (v0.2.x)

### Completed Features

- **Hardware Abstraction Layer (HAL)**
  - `HardwareCapabilities` descriptor for dynamic configuration
  - `SubcarrierConfig` for bandwidth-specific settings
  - Plugin registry with entry_points support
  - Factory function `get_extractor()` with environment variable default

- **Multi-Hardware Support**
  - ESP32 (WiFi 4/6, including C6 variant)
  - Intel 5300 (stub implementation)
  - Nexmon BCM43455 (stub implementation)
  - Atheros AR9380 (stub implementation)

- **Feature Detection API**
  - `capabilities.has_feature(FEATURE_MIMO)` queries
  - Auto-detection from hardware capabilities
  - Custom feature registration

- **Environment Configuration**
  - `OPENCSI_DEFAULT_HARDWARE` - Default extractor
  - `OPENCSI_LOG_LEVEL` - Logging verbosity
  - `OPENCSI_DEBUG` - Debug mode
  - `OPENCSI_ARRAY_BACKEND` - NumPy/CuPy selection

- **Wavelength-Aware Processing**
  - Channel-to-wavelength conversion
  - Multi-band Fresnel zone calculations
  - Dynamic subcarrier configuration

---

## Short Term (v0.3.x)

### Multi-Hardware Fusion

**Goal:** Combine CSI data from heterogeneous deployments (e.g., ESP32 + Intel 5300).

```python
# Planned API
from opencsi import MultiHardwareFusion

fusion = MultiHardwareFusion()
fusion.add_source("esp32_1", get_extractor("esp32"))
fusion.add_source("intel_1", get_extractor("intel_5300"))

# Unified output regardless of source hardware
frame = fusion.ingest(packets)
```

**Key Challenges:**
- Subcarrier count normalization (interpolation/decimation)
- Timestamp synchronization across hardware
- SNR-weighted fusion

### Streaming API

**Goal:** Process CSI packets without buffering entire sequences.

```python
# Planned API
from opencsi import StreamingCalibrator

async for frame in StreamingCalibrator(hardware):
    yield frame.motion_detected, frame.position
```

**Benefits:**
- Lower memory footprint
- Real-time processing
- Backpressure handling

### Parallelization Support

**Goal:** Enable parallel processing at multiple levels for high-throughput deployments.

#### Parallelization Levels

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OpenCSI Parallelization Stack                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 4: Distributed    │ Multiple machines (Redis/ZMQ coordination)   │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 3: Multi-Process  │ ProcessPoolExecutor for CPU-bound batches    │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 2: Multi-Thread   │ ThreadPoolExecutor for I/O-bound links       │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 1: Vectorized     │ NumPy/CuPy SIMD/GPU (already implemented)    │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 0: Single-Thread  │ Sequential processing (baseline)             │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Environment Variables

```bash
# Control parallelization
export OPENCSI_PARALLEL_BACKEND=thread    # thread, process, or none
export OPENCSI_MAX_WORKERS=4              # Number of workers (default: CPU count)
export OPENCSI_BATCH_SIZE=32              # Packets per batch
```

#### Planned API

```python
from opencsi import ParallelCalibrator, get_extractor

# Automatic parallelization across links
calibrator = ParallelCalibrator(
    node_positions=positions,
    hardware=get_extractor(),
    parallel_backend="thread",  # or "process", "none"
    max_workers=4,
)

# Process multiple packets in parallel
frames = calibrator.ingest_batch(packets)  # Returns List[CalibratedFrame]

# Async interface for real-time systems
async for frame in calibrator.stream(packet_source):
    if frame.motion_detected:
        await notify(frame.position)
```

#### Link-Level Parallelism

```python
# Process each TX-RX link independently
from opencsi.parallel import LinkParallelProcessor

processor = LinkParallelProcessor(max_workers=8)

# Each link's baseline/calibration runs in separate thread
results = processor.map(
    links=["link_A", "link_B", "link_C", "link_D"],
    func=calibrate_link,
    packets=grouped_packets,
)
```

#### Batch Processing

```python
from opencsi import BatchProcessor

# Accumulate packets and process in batches
batch = BatchProcessor(batch_size=64, timeout_ms=100)

for packet in udp_stream:
    batch.add(packet)
    if batch.ready():
        frames = calibrator.ingest_batch(batch.flush())
```

#### Performance Targets

| Configuration | Throughput | Latency | Use Case |
|--------------|------------|---------|----------|
| Single-thread | 1k pkt/s | <1ms | Low-power edge |
| Multi-thread (4) | 4k pkt/s | <5ms | Real-time sensing |
| Multi-process (8) | 10k pkt/s | <20ms | Offline analysis |
| GPU + Thread | 50k pkt/s | <10ms | High-density deployment |

#### When to Use Each Backend

| Backend | Best For | Avoid When |
|---------|----------|------------|
| `none` | Single-link, low latency | Many links |
| `thread` | I/O-bound, many links | CPU-bound ops |
| `process` | CPU-bound batches | Shared state needed |
| `cupy` | Large arrays, FFT | Small packets |

### Community Extractor Guide

- Step-by-step guide for adding new hardware
- Validation script for extractor compliance
- Entry points tutorial for plugins

---

## Medium Term (v0.4.x)

### GPU Acceleration

**Goal:** Optional CuPy backend for high-throughput deployments.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenCSI Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  CSI Packets → Sanitization → Calibration → Position Solver │
├─────────────────────────────────────────────────────────────┤
│                    Array Backend Abstraction                 │
│  ┌───────────────┐              ┌───────────────┐           │
│  │    NumPy      │   ← swap →   │    CuPy       │           │
│  │    (CPU)      │              │    (GPU)      │           │
│  └───────────────┘              └───────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

#### Environment Variable

```bash
# Enable GPU acceleration
export OPENCSI_ARRAY_BACKEND=cupy
```

#### Implementation Plan

| Phase | Component | Effort | Speedup |
|-------|-----------|--------|---------|
| 1 | FFT in sanitizer | 2 days | 10-50x |
| 2 | Correlation in baseline | 2 days | 10-20x |
| 3 | Matrix operations in position solver | 3 days | 5-10x |
| 4 | Batch processing for multiple links | 2 days | 2-5x |

#### Code Pattern

```python
# Current (works with both backends)
from opencsi.core.env import get_array_module

xp = get_array_module()  # Returns numpy or cupy

# Array operations use xp instead of np
data = xp.fft.fft(csi_matrix)
mean = xp.mean(data, axis=0)
```

#### Requirements

- CuPy installation: `pip install cupy-cuda12x`
- CUDA-capable GPU
- Driver version 450.80.02+ (CUDA 11.x) or 525.60.13+ (CUDA 12.x)

#### Fallback Behavior

```python
# Automatic fallback if CuPy not available
if config.array_backend == "cupy":
    try:
        import cupy
    except ImportError:
        logger.warning("CuPy not available, falling back to NumPy")
        # Continues with NumPy
```

### Phase Calibration Framework

**Goal:** Hardware-specific phase offset correction.

```python
# Planned API
calibrator = PhaseCalibrator(hardware)
calibrator.learn_offsets(reference_packets)
corrected_csi = calibrator.apply(raw_csi)
```

### Doppler Processing

**Goal:** Extract velocity information from CSI phase.

```python
# Planned API
doppler = DopplerExtractor(sample_rate_hz=100)
velocity_spectrum = doppler.process(csi_sequence)
```

---

## Long Term (v1.0.x)

### FFI Bindings (C API)

**Goal:** Enable use from embedded systems and other languages.

```c
// Planned C API
opencsi_handle_t handle = opencsi_create("esp32");
opencsi_packet_t* packet = opencsi_parse(handle, raw_data, len);
opencsi_frame_t* frame = opencsi_calibrate(handle, packet);
bool motion = opencsi_get_motion(frame);
opencsi_destroy(handle);
```

**Target Platforms:**
- ESP32 (direct integration)
- Raspberry Pi (low-latency)
- Any platform with C compiler

### pcap/Wireshark Integration

**Goal:** Import CSI from standard packet captures.

```python
# Planned API
from opencsi.io import PcapReader

for packet in PcapReader("capture.pcap", hardware="nexmon"):
    frame = calibrator.ingest([packet])
```

### Distributed Calibration

**Goal:** Coordinate calibration across multiple nodes.

```python
# Planned API
coordinator = DistributedCalibrator(
    nodes=["esp32_1", "esp32_2", "esp32_3"],
    backend="redis"
)
coordinator.sync_baselines()
```

### Machine Learning Integration

**Goal:** Export features for ML models.

```python
# Planned API
from opencsi.ml import FeatureExtractor

extractor = FeatureExtractor(
    features=["amplitude_variance", "phase_diff", "doppler_bins"]
)
X = extractor.transform(frames)
# X is sklearn/pytorch compatible
```

---

## Version Timeline

| Version | Target Date | Key Features |
|---------|-------------|--------------|
| v0.2.0 | Q1 2025 | HAL, Feature Detection, Environment Config |
| v0.3.0 | Q2 2025 | Multi-Hardware Fusion, Streaming API |
| v0.4.0 | Q3 2025 | GPU Acceleration, Doppler Processing |
| v0.5.0 | Q4 2025 | Phase Calibration, ML Features |
| v1.0.0 | Q1 2026 | FFI Bindings, Stable API |

---

## Contributing

### Adding New Hardware

1. Create `opencsi/hardware/your_hardware.py`
2. Implement `CSIExtractor` with `capabilities` property
3. Add profile to `profiles.json`
4. Register via entry_points in `pyproject.toml`
5. Run validation: `python -m opencsi.scripts.validate_extractor your_hardware`

### GPU Acceleration Contributions

Focus areas for GPU speedup:
- FFT operations in sanitizer
- Correlation computations in baseline
- Matrix inversions in position solver
- Batch processing for many links

### Reporting Issues

Include:
- Hardware type and firmware version
- OpenCSI version (`python -c "import opencsi; print(opencsi.__version__)"`)
- Environment variables (`env | grep OPENCSI`)
- Minimal reproduction script

---

## Design Principles

Learned from OpenCV, OpenMPI, and similar projects:

1. **Zero-Configuration Start**: `get_extractor()` just works
2. **Progressive Disclosure**: Simple API for beginners, full control for experts
3. **Backend Abstraction**: Same code runs on CPU/GPU
4. **Plugin Architecture**: Community extensions without core changes
5. **Graceful Degradation**: Always fall back to working state
