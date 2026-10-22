"""FastAPI server with REST API and WebSocket live updates."""
import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import WEB_PORT, DEFAULT_ROOM_WIDTH, DEFAULT_ROOM_HEIGHT
from schema import AnnotationCreate, TrialStartRequest
from experiment import ExperimentRunner, ExperimentConfig
from localizer import estimate as localize

logger = logging.getLogger(__name__)

# WebSocket broadcast interval
WS_BROADCAST_HZ = 2
WS_BROADCAST_INTERVAL_S = 1.0 / WS_BROADCAST_HZ

# Config persistence file
CONFIG_FILE = Path("config.json")


def _mac_key(mac: str) -> str:
    """Return last 4 hex chars of a MAC address (e.g. 'AA:BB:CC:DD:EE:FF' -> 'EE:FF')."""
    raw = mac.replace(":", "")[-4:].upper()
    return f"{raw[:2]}:{raw[2:]}"


class RoomConfig(BaseModel):
    width: float = DEFAULT_ROOM_WIDTH
    height: float = DEFAULT_ROOM_HEIGHT


class PositionUpdate(BaseModel):
    x: float
    y: float


class LabelUpdate(BaseModel):
    label: str


class RoomFeature(BaseModel):
    id: str
    type: str  # "door", "window", "table", "wall"
    x: float
    y: float
    width: float
    height: float | None = None
    rotation: float = 0
    flip: bool = False


def create_app(registry, detector, scheduler, receiver=None,
               recorder=None, labeler=None) -> FastAPI:
    app = FastAPI(title="CSI-Sense", version="2.0.0")

    room, node_positions, node_labels, features = _load_config()

    experiment = ExperimentRunner(recorder=recorder, labeler=labeler)

    ws_clients: set = set()

    async def _sync_node_positions():
        """Apply config positions to new nodes and add new nodes to config."""
        changed = False
        for node in registry.get_all_nodes():
            key = _mac_key(node.mac)
            if key not in node_positions:
                node_positions[key] = (node.position[0], node.position[1]) if node.position else None
                changed = True
            elif node.position is None and node_positions[key] is not None:
                x, y = node_positions[key]
                await registry.set_position(node.mac, x, y)
        if changed:
            _save_config(room, node_positions, node_labels, features)

    registry.on_change(_sync_node_positions)

    @app.get("/api/nodes")
    async def get_nodes():
        nodes = registry.get_all_nodes()
        return [{"mac": n.mac, "ip": n.ip, "chip": n.chip, "version": n.version,
                 "status": n.status.value, "position": n.position,
                 "label": node_labels.get(_mac_key(n.mac)),
                 "last_seen": n.last_seen.isoformat(), "uptime_s": n.uptime_s,
                 "free_heap": n.free_heap, "channel": n.channel,
                 "bandwidth": n.bandwidth} for n in nodes]

    @app.get("/api/detection")
    async def get_detection():
        return detector.get_state()

    @app.get("/api/room")
    async def get_room():
        return {"width": room.width, "height": room.height}

    @app.put("/api/room")
    async def set_room(config: RoomConfig):
        room.width = config.width
        room.height = config.height
        scheduler.room_width = config.width
        scheduler.room_height = config.height
        _save_config(room, node_positions, node_labels, features)
        return {"width": room.width, "height": room.height}

    @app.put("/api/nodes/{mac}/position")
    async def set_position(mac: str, pos: PositionUpdate):
        success = await registry.set_position(mac, pos.x, pos.y)
        if not success:
            raise HTTPException(404, f"Node {mac} not found")
        node_positions[_mac_key(mac.upper())] = (pos.x, pos.y)
        _save_config(room, node_positions, node_labels, features)
        return {"mac": mac, "position": [pos.x, pos.y]}

    @app.put("/api/nodes/{mac}/label")
    async def set_label(mac: str, body: LabelUpdate):
        key = _mac_key(mac.upper())
        node_labels[key] = body.label
        _save_config(room, node_positions, node_labels, features)
        return {"mac": mac, "label": body.label}

    @app.get("/api/features")
    async def get_features():
        return features

    @app.put("/api/features")
    async def set_features(items: list[RoomFeature]):
        features.clear()
        features.extend([f.model_dump() for f in items])
        _save_config(room, node_positions, node_labels, features)
        return features

    @app.post("/api/calibration/reset")
    async def reset_calibration():
        detector.reset_baselines()
        return {"status": "reset"}

    @app.post("/api/baseline/freeze")
    async def toggle_baseline_freeze():
        frozen = detector.toggle_baseline_freeze()
        return {"baseline_frozen": frozen}

    # --- Simulation ---

    class SimulationUpdate(BaseModel):
        enabled: bool
        x: float = 0.0
        y: float = 0.0

    @app.post("/api/simulation")
    async def set_simulation(body: SimulationUpdate):
        if body.enabled:
            detector.set_simulation({"x": body.x, "y": body.y})
        else:
            detector.set_simulation(None)
        return {"simulation": detector.get_simulation()}

    @app.get("/api/simulation")
    async def get_simulation():
        return {"simulation": detector.get_simulation()}

    @app.get("/api/scheduler")
    async def get_scheduler():
        return {"zones": scheduler.get_zones_info()}

    @app.get("/api/stats")
    async def get_stats():
        stats = receiver.get_stats() if receiver else {}
        stats["detection"] = detector.get_state()
        stats["stats"] = detector.get_stats_summary()
        return stats

    @app.get("/api/packets")
    async def get_packets(limit: int = 50):
        """Get recent packet data with calibration statistics."""
        return {"packets": detector.get_recent_packets(limit)}

    @app.get("/api/recording/status")
    async def recording_status():
        if recorder is None:
            return {"recording": False, "error": "recorder not configured"}
        return recorder.get_stats()

    @app.post("/api/recording/start")
    async def recording_start(req: TrialStartRequest):
        if recorder is None:
            raise HTTPException(500, "recorder not configured")
        try:
            await recorder.start_recording(req)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return recorder.get_stats()

    @app.post("/api/recording/stop")
    async def recording_stop():
        if recorder is None:
            raise HTTPException(500, "recorder not configured")
        await recorder.stop_recording()
        return recorder.get_stats()

    @app.put("/api/recording/compress")
    async def recording_compress(req: dict):
        if recorder is None:
            raise HTTPException(500, "recorder not configured")
        enabled = bool(req.get("enabled", True))
        recorder.set_compress(enabled)
        return recorder.get_stats()

    @app.get("/api/annotations/taxonomy")
    async def annotations_taxonomy():
        if labeler is None:
            raise HTTPException(500, "labeler not configured")
        return await labeler.get_taxonomy()

    @app.get("/api/annotations/activities")
    async def annotations_activities():
        if labeler is None:
            raise HTTPException(500, "labeler not configured")
        return await labeler.get_activities()

    @app.post("/api/annotations")
    async def annotations_create(req: AnnotationCreate):
        if labeler is None:
            raise HTTPException(500, "labeler not configured")
        try:
            rec = await labeler.add_annotation(req)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return dataclasses.asdict(rec)

    @app.get("/api/annotations")
    async def annotations_list(date: str = None, trial_id: str = None):
        if labeler is None:
            raise HTTPException(500, "labeler not configured")
        return await labeler.get_annotations(date_str=date, trial_id=trial_id)

    @app.delete("/api/annotations/{annotation_id}")
    async def annotations_delete(annotation_id: str):
        if labeler is None:
            raise HTTPException(500, "labeler not configured")
        found = await labeler.delete_annotation(annotation_id)
        if not found:
            raise HTTPException(404, "Annotation not found")
        return {"deleted": True}

    @app.get("/api/experiment/status")
    async def experiment_status():
        return experiment.get_status()

    @app.post("/api/experiment/start")
    async def experiment_start(config: ExperimentConfig):
        try:
            await experiment.start(config)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        return experiment.get_status()

    @app.post("/api/experiment/stop")
    async def experiment_stop():
        await experiment.stop()
        return experiment.get_status()

    # --- Metadata endpoints for environments, subjects, topologies ---

    def _list_json(subdir: str):
        d = Path(f"data/{subdir}")
        if not d.exists():
            return []
        return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]

    def _save_json(subdir: str, id_field: str, data: dict):
        d = Path(f"data/{subdir}")
        d.mkdir(parents=True, exist_ok=True)
        fpath = d / f"{data[id_field]}.json"
        fpath.write_text(json.dumps(data, indent=2))
        return data

    def _delete_json(subdir: str, item_id: str):
        fpath = Path(f"data/{subdir}/{item_id}.json")
        if not fpath.exists():
            raise HTTPException(404, f"{item_id} not found")
        fpath.unlink()
        return {"deleted": True}

    @app.get("/api/environments")
    async def list_environments():
        return _list_json("environments")

    @app.post("/api/environments")
    async def create_environment(data: dict):
        if not data.get("environment_id"):
            raise HTTPException(422, "environment_id is required")
        return _save_json("environments", "environment_id", data)

    @app.delete("/api/environments/{env_id}")
    async def delete_environment(env_id: str):
        return _delete_json("environments", env_id)

    @app.get("/api/subjects")
    async def list_subjects():
        return _list_json("subjects")

    @app.post("/api/subjects")
    async def create_subject(data: dict):
        if not data.get("subject_id"):
            raise HTTPException(422, "subject_id is required")
        return _save_json("subjects", "subject_id", data)

    @app.delete("/api/subjects/{subject_id}")
    async def delete_subject(subject_id: str):
        return _delete_json("subjects", subject_id)

    @app.get("/api/topologies")
    async def list_topologies():
        return _list_json("topologies")

    @app.post("/api/topologies")
    async def create_topology(data: dict):
        if not data.get("topology_id"):
            raise HTTPException(422, "topology_id is required")
        return _save_json("topologies", "topology_id", data)

    @app.delete("/api/topologies/{topo_id}")
    async def delete_topology(topo_id: str):
        return _delete_json("topologies", topo_id)

    @app.get("/api/rf-info")
    async def rf_info():
        """Return detected channel/bandwidth from live nodes."""
        nodes = registry.get_online_nodes()
        channels = set(n.channel for n in nodes if n.channel > 0)
        bandwidths = set(n.bandwidth for n in nodes if n.bandwidth)
        return {
            "channels": sorted(channels),
            "bandwidths": sorted(bandwidths),
            "num_nodes": len(nodes),
        }

    @app.websocket("/ws/live")
    async def websocket_live(ws: WebSocket):
        await ws.accept()
        ws_clients.add(ws)
        try:
            while True:
                await ws.receive_text()  # Keep alive
        except WebSocketDisconnect:
            ws_clients.discard(ws)
        except Exception:
            ws_clients.discard(ws)

    async def _broadcast_loop():
        """Broadcast state to all WebSocket clients at WS_BROADCAST_HZ."""
        while True:
            await asyncio.sleep(WS_BROADCAST_INTERVAL_S)
            if not ws_clients:
                continue

            try:
                nodes = registry.get_all_nodes()
                node_pos = {n.mac: n.position for n in nodes if n.position}

                # Apply simulation before reading detection state
                detector.apply_simulation(node_pos)

                detection = detector.get_state()
                loc = localize(
                    detection["active_link_details"], node_pos,
                    room.width, room.height,
                )

                state = {
                    "nodes": [{"mac": n.mac, "ip": n.ip, "status": n.status.value,
                               "position": n.position, "chip": n.chip,
                               "label": node_labels.get(_mac_key(n.mac)),
                               "channel": n.channel, "bandwidth": n.bandwidth} for n in nodes],
                    "detection": detection,
                    "room": {"width": room.width, "height": room.height},
                    "zones": scheduler.get_zones_info(),
                    "recent_packets": detector.get_recent_packets(20),
                    "recording": recorder.get_stats() if recorder else {"recording": False},
                    "experiment": experiment.get_status(),
                    "stats": detector.get_stats_summary(),
                    "features": features,
                    "localization": loc,
                    "simulation": detector.get_simulation(),
                }
                msg = json.dumps(state)
            except Exception:
                logger.exception("Failed to build broadcast state")
                continue

            dead = set()
            for ws in list(ws_clients):  # snapshot to avoid mutation during iteration
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            ws_clients.difference_update(dead)

    @app.on_event("startup")
    async def on_startup():
        async def _broadcast_watchdog():
            while True:
                task = asyncio.create_task(_broadcast_loop())
                try:
                    await task
                except Exception:
                    logger.error("Broadcast loop crashed, restarting", exc_info=True)
        asyncio.create_task(_broadcast_watchdog())

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:
        @app.get("/")
        async def root():
            return {"message": "CSI-Sense API. Frontend not built yet."}

    return app


def _load_config() -> tuple[RoomConfig, dict, dict]:
    """Load room config, node positions, and node labels from config.json."""
    node_positions: dict[str, Optional[tuple[float, float]]] = {}
    node_labels: dict[str, str] = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            room = RoomConfig(width=data.get("width", DEFAULT_ROOM_WIDTH),
                              height=data.get("height", DEFAULT_ROOM_HEIGHT))
            for key, val in data.get("nodes", {}).items():
                k = key.replace(":", "").upper()
                normalized = f"{k[:2]}:{k[2:]}" if ":" not in key else key.upper()
                pos = None
                if val.get("x") is not None and val.get("y") is not None:
                    pos = (float(val["x"]), float(val["y"]))
                if normalized not in node_positions or (pos is not None and node_positions[normalized] is None):
                    node_positions[normalized] = pos
                if val.get("label"):
                    node_labels[normalized] = val["label"]
            feat_list = data.get("features", [])
            return room, node_positions, node_labels, feat_list
        except Exception:
            pass
    return RoomConfig(), node_positions, node_labels, []


def _save_config(room: RoomConfig, node_positions: dict, node_labels: dict, features: list):
    """Persist room config, node positions, node labels, and features to config.json."""
    nodes_out = {}
    for key, pos in sorted(node_positions.items()):
        entry: dict = {}
        if pos is not None:
            entry["x"] = round(pos[0], 2)
            entry["y"] = round(pos[1], 2)
        else:
            entry["x"] = None
            entry["y"] = None
        if key in node_labels:
            entry["label"] = node_labels[key]
        nodes_out[key] = entry
    out: dict = {"width": room.width, "height": room.height, "nodes": nodes_out}
    if features:
        out["features"] = features
    CONFIG_FILE.write_text(json.dumps(out, indent=2) + "\n")
