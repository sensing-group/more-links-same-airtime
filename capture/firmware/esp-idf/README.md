# ESP32-C3 WiFi CSI Firmware

ESP-IDF firmware for WiFi Channel State Information (CSI) capture on XIAO ESP32-C3 boards.

## Current Version

**v5.2.3-FOVEATED** (January 2026)

Thread safety fixes based on external code review. Adds mutex protection for batch send buffer.

## Directory Structure

```
firmware/esp-idf/
├── main/
│   ├── main.c           # Main firmware source (see header for version history)
│   └── CMakeLists.txt   # Component configuration
├── CMakeLists.txt       # Project configuration
├── sdkconfig.defaults   # Default build configuration
├── dependencies.lock    # ESP-IDF dependencies
└── README.md            # This file
```

## Documentation

All firmware documentation is in `docs/firmware/`:

- `BUILD_NOTES.md` - General build notes and configuration
- `BUILD_WINDOWS.md` - Windows-specific build procedure
- `DEBUG_TRIGGER_CRASH.md` - Debug notes for v4.1.2 stack overflow fix
- `FIRMWARE_v4.1.3_OPTIMIZATIONS.md` - Optimization guide for v4.1.3

## Quick Build

### Prerequisites
- ESP-IDF v5.5.2
- Python 3.11
- CMake, Ninja

### Build Commands

```bash
# Clean build
rm -rf build sdkconfig

# Set target and build
idf.py set-target esp32c3
idf.py build

# Flash
idf.py -p COMx flash monitor
```

For Windows-specific instructions, see `docs/firmware/BUILD_WINDOWS.md`.

## Version History

See `main/main.c` header for complete version history.

| Version | Tag | Key Changes |
|---------|-----|-------------|
| 5.2.3 | FOVEATED | Thread safety: mutex for batch send, reverted ping-pong (race condition) |
| 5.2.2 | FOVEATED | Performance: CRC outside spinlock, trigger state optimization |
| 5.2.1 | FOVEATED | Bug fixes: tick overflow, CSI callback context, nested spinlock, UDP tracking |
| 5.2.0 | FOVEATED | 40-50 Hz focus mode, reduced rate limit (15ms), doubled ring buffer |
| 5.1.1 | HIGHFREQ | 30ms rate limit, 10-packet bursts |
| 4.1.3 | OPTIMIZED | Trigger rate limiting, reduced burst count |
| 4.1.2 | STABLE | Fixed stack overflow in trigger processing |
| 4.1.1 | PUREBIN | Faster burst timing (200µs delay) |
| 4.1.0 | PUREBIN | Removed cJSON, pure binary protocol |
| 4.0 | BINARY | Zero-malloc binary protocol |
| 3.6 | RELIABLE | Task watchdog, heap monitoring |
