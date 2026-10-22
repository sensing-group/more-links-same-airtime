"""Initialize data directory structure on first run.

Creates required subdirectories and copies template files if they don't exist.
Called from main.py before starting the server.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_DIRS = ["trials", "environments", "subjects", "topologies"]

TEMPLATES = {
    "environments/lab-alpha.json": {
        "environment_id": "lab-alpha",
        "description": "",
        "dimensions_m": {"length": 0, "width": 0, "height": 0},
        "furniture": "",
        "wall_material": "",
        "floor_material": "",
        "known_interference": [],
        "photos": [],
        "created_at": "2026-03-31T00:00:00+00:00",
    },
    "subjects/S01.json": {
        "subject_id": "S01",
        "height_cm": None,
        "weight_kg": None,
        "age_range": "",
        "notes": "",
        "created_at": "2026-03-31T00:00:00+00:00",
    },
    "topologies/topo-10node.json": {
        "topology_id": "topo-10node",
        "description": "10 ESP32 nodes",
        "num_nodes": 10,
        "height_profile": "variable",
        "nodes": [],
        "channel": 6,
        "bandwidth": "HT20",
        "tx_power_dbm": 20,
        "antenna_type": "onboard",
        "created_at": "2026-03-31T00:00:00+00:00",
    },
}


def init_data_dir(base_dir: str = "data"):
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    for d in REQUIRED_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Touch manifest
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        manifest.touch()

    # Write templates only if file doesn't exist
    for rel_path, content in TEMPLATES.items():
        path = root / rel_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2))
            logger.info("Created template: %s", path)
