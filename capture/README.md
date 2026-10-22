# CSI-Sense

Minimal WiFi CSI (Channel State Information) sensing system for room-level presence detection using ESP32 microcontrollers. Designed to be understood in an afternoon.

## How It Works

ESP32 nodes take turns transmitting WiFi frames while all other nodes listen. The channel response between each transmitter-receiver pair (a "link") changes when a person moves through or stands in the signal path. By monitoring the variance of these channel measurements across all links and comparing to a learned baseline, the system detects room occupancy.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Backend (Python)            │
                    │                                         │
  ESP32 nodes       │  ┌──────────┐    ┌──────────────────┐  │
  ───UDP:5000──────▶│  │ Receiver │───▶│    Detector       │  │
  (CSI data)        │  │ (async)  │    │ (Welford Z-score) │  │
                    │  └──────────┘    └────────┬─────────┘  │
                    │        │                  │             │
                    │        ▼                  ▼             │
  ESP32 nodes       │  ┌──────────┐    ┌──────────────────┐  │    ┌──────────┐
  ◀──UDP:5001──────┐│  │ Registry │    │  Server (FastAPI) │──┼───▶│ Frontend │
  (triggers)       ││  │ (nodes)  │    │  REST + WebSocket │  │    │ React+SVG│
                   ││  └──────────┘    └──────────────────┘  │    └──────────┘
                   ││        ▲                                │
                   ││        │                                │
                   ││  ┌─────┴────┐                           │
                   │└──│Scheduler │                           │
                   │   │  (TDMA)  │                           │
                   │   └──────────┘                           │
                   │                                          │
                   └──────────────────────────────────────────┘
```

### Data Flow

1. **Scheduler** broadcasts a TDMA trigger packet to `255.255.255.255:5001` every 20ms. The packet contains the MAC address of which node should transmit.
2. **ESP32 nodes** all receive the broadcast trigger. The node whose MAC matches enters TX mode and sends a burst of WiFi frames. All other nodes enter RX mode and collect CSI measurements from those frames.
3. Each RX node batches its CSI samples and sends them back to the server via UDP on port 5000.
4. **Receiver** parses the binary packets, resolves TX/RX MACs, and dispatches `CSISample` objects to callbacks.
5. **Detector** computes the amplitude variance across 52 subcarriers for each TX-RX link. During calibration it learns per-link baseline statistics (Welford's online algorithm). During monitoring it computes Z-scores and flags links with |Z| > threshold as "active".
6. **Server** exposes the state via REST API and pushes 2 Hz updates over WebSocket to the frontend.

### TDMA Protocol

The system uses Time-Division Multiple Access so that only one node transmits at a time, preventing packet collisions:

```
Time ──▶
┌────────┬────────┬────────┬────────┬────────┬───
│ Node A │ Node B │ Node C │  ...   │ Node N │ (cycle delay) → repeat
│   TX   │   TX   │   TX   │        │   TX   │
│ all RX │ all RX │ all RX │        │ all RX │
└────────┴────────┴────────┴────────┴────────┴───
  20ms     20ms     20ms              20ms     5ms
```

Each trigger is a 32-byte binary packet (magic `0xC5`, type `0x10`) containing the designated transmitter's MAC, a sequence number, burst count, slot duration, and CRC-32. Triggers are **broadcast** (not unicast) so every node knows who should transmit.

### Binary Packet Types

| Type | Magic | ID | Size | Description |
|------|-------|----|------|-------------|
| HELLO | 0xC5 | 0x01 | 40 B | Node registration (MAC, chip, firmware version) |
| HEARTBEAT | 0xC5 | 0x02 | 24 B | Health check (uptime, free heap) |
| CSI v1 | 0xC5 | 0x02 | 128 B | Single CSI sample (disambiguated by size) |
| CSI v2 | 0xC5 | 0x03 | 148 B | Single CSI sample with embedded RX MAC |
| CSI v5 | 0xC5 | 0x07 | variable | Batched CSI samples (header + N*107 B + CRC) |
| TRIGGER | 0xC5 | 0x10 | 32 B | TDMA scheduling command |

All packets use CRC-32 for integrity validation.

### Detection Pipeline

```
CSI Sample (52 subcarriers × I,Q)
    │
    ▼
Amplitude = sqrt(I² + Q²) per subcarrier
    │
    ▼
Variance across 52 subcarriers → single scalar per link
    │
    ├── Calibration: Welford update (mean, std) ── needs 200 samples/link
    │
    └── Monitoring: Z = (variance - mean) / std
                         │
                         ▼
                    |Z| > 2.5 → link is "active"
                         │
                         ▼
                    active_links ≥ 3 → room "occupied"
```

## File Structure

```
csi-sense/
├── backend/
│   ├── config.py          # Constants with env variable overrides
│   ├── protocol.py        # Binary packet parsing (HELLO, HEARTBEAT, CSI v1/v2/v5, TRIGGER)
│   ├── receiver.py        # Async UDP server (asyncio.DatagramProtocol)
│   ├── registry.py        # Node tracking, health monitoring, IP↔MAC mapping
│   ├── scheduler.py       # Zone-based round-robin TDMA with broadcast triggers
│   ├── detector.py        # Per-link Welford statistics → Z-score → occupancy
│   ├── server.py          # FastAPI REST API + WebSocket + static file serving
│   ├── main.py            # Entrypoint: wires components, starts services
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx         # Layout shell
│       ├── RoomMap.tsx     # SVG room map with draggable nodes and active links
│       ├── StatusBar.tsx   # Occupancy indicator + connection status
│       ├── ConfigPanel.tsx # Room dimension controls
│       ├── store.ts        # Zustand state store
│       ├── useWebSocket.ts # Auto-reconnecting WebSocket hook
│       └── types.ts        # TypeScript interfaces
├── firmware/esp-idf/       # ESP32 firmware (ESP-IDF 5.x, flash separately)
├── opencsi/                # Vendored CSI processing library
├── Dockerfile              # Multi-stage build (Node → Python)
├── docker-compose.yml      # network_mode: host for ESP32 UDP
├── .env.example            # Configuration template
└── config.json.example     # Runtime config template (room dimensions)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- ESP32-C3 (or ESP32/ESP32-S3) nodes flashed with the firmware from `firmware/esp-idf/`
- All devices on the same WiFi network

### 1. Configure

```bash
cp .env.example .env
cp config.json.example config.json
# Edit .env to set room dimensions, ports, detection thresholds
# config.json stores runtime room dimensions (auto-updated via API)
```

### 2. Build and Run

```bash
sudo docker compose up --build -d
```

The server starts on port 8000 with `network_mode: host` so it can receive UDP broadcasts from ESP32 nodes.

### 3. Verify

```bash
# Check logs
sudo docker logs csi_sense

# Check nodes
curl http://localhost:8000/api/nodes

# Check detection state
curl http://localhost:8000/api/detection

# Check receiver stats
curl http://localhost:8000/api/stats
```

### 4. Use the Dashboard

Open **http://localhost:8000** in a browser.

1. Nodes appear as circles once ESP32s connect (~5 seconds)
2. Drag nodes to match their physical positions in the room
3. Keep the room **empty** during calibration (~5 minutes for all 155 links)
4. Once calibration completes, walk through the room to trigger detection

### Running Without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
pip install ../opencsi
python main.py

# Frontend (separate terminal, dev mode with hot reload)
cd frontend
npm install
npm run dev
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/nodes` | List all registered nodes with status, position, chip info |
| GET | `/api/detection` | Current occupancy state, active links, calibration progress |
| GET | `/api/room` | Room dimensions |
| PUT | `/api/room` | Update room dimensions (JSON: `{"width": 5.0, "height": 4.0}`) |
| PUT | `/api/nodes/{mac}/position` | Set node position (JSON: `{"x": 1.5, "y": 2.0}`) |
| POST | `/api/calibration/reset` | Clear baselines and restart calibration |
| GET | `/api/scheduler` | Current zone assignments |
| GET | `/api/stats` | Receiver packet counts + detection summary |
| WS | `/ws/live` | 2 Hz JSON broadcast of nodes, detection, room, zones |

## Configuration

All settings are overridable via environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `UDP_PORT` | 5000 | Port for receiving CSI data from nodes |
| `TRIGGER_PORT` | 5001 | Port for sending TDMA triggers to nodes |
| `WEB_PORT` | 8000 | HTTP and WebSocket port |
| `BROADCAST_ADDR` | 255.255.255.255 | Trigger broadcast address |
| `SLOT_DURATION_MS` | 20 | Time per TDMA slot (ms) |
| `BURSTS_PER_TRIGGER` | 10 | WiFi frames per TX burst |
| `CYCLE_DELAY_MS` | 5 | Pause between full TDMA cycles (ms) |
| `MIN_NODES_TO_START` | 2 | Minimum nodes before scheduling begins |
| `ZONE_MAX_SIZE` | 6 | Maximum nodes per scheduling zone |
| `MOTION_ZSCORE_THRESHOLD` | 2.5 | Z-score threshold for link activity |
| `ACTIVE_LINK_THRESHOLD` | 3 | Minimum active links to declare occupancy |
| `BASELINE_LEARNING_SAMPLES` | 200 | Samples per link before detection starts |
| `HEARTBEAT_TIMEOUT_S` | 30.0 | Seconds before marking a node offline |
| `ROOM_WIDTH` | 5.0 | Room width in meters |
| `ROOM_HEIGHT` | 4.0 | Room height in meters |

## Troubleshooting

**No nodes appearing**: Check that ESP32 nodes are on the same network and the firmware's `SERVER_IP` matches this machine's IP. Verify UDP port 5000 is not blocked.

**Nodes connect but no CSI data**: Triggers must be **broadcast** to `255.255.255.255:5001`. If running without Docker, ensure your firewall allows UDP broadcast. Check `curl localhost:8000/api/stats` — `packets_received` should increase rapidly.

**Calibration is slow**: Calibration requires the slowest link (worst node pair) to accumulate 200 samples. With 13 nodes this takes ~5 minutes. You can lower `BASELINE_LEARNING_SAMPLES` for faster startup at the cost of noisier baselines.

**False positives after calibration**: If someone was in the room during calibration, the baseline includes their presence. Reset with `curl -X POST localhost:8000/api/calibration/reset` while the room is empty.

**Port already in use**: Another service is using port 5000 or 8000. Either stop it or change ports via environment variables.

## Dependencies

- **Backend**: Python 3.10+, FastAPI, uvicorn, numpy
- **Frontend**: React 18, Zustand, Vite, TypeScript
- **Firmware**: ESP-IDF 5.x (ESP32, ESP32-C3, ESP32-S3)

## License

MIT
