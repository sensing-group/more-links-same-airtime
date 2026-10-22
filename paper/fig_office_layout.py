#!/usr/bin/env python3
"""Office mesh layout: 14 wall nodes (positions digitized from the deployment
map, not surveyed) and the 16 fixed pairs used by the budget analysis, coloured
by transmitter. The room is drawn lying down (its long side horizontal)."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import paper_style as ps

ROOT = Path(__file__).resolve().parent.parent
ROOM_CONFIG = Path(os.environ.get("ROOM_CONFIG", str(ROOT / "paper" / "room_config_office.json")))
PAIRS_JSON = Path(os.environ.get("ANALYSIS_DIR", str(ROOT / "results"))) / "decision_windows" / "decision_window_ablation_w20s.json"
FIG_H_IN = 2.5
LABEL_OFFSET_PT = 6        # node label distance from the node, outward from its wall
MARGIN_X, MARGIN_Y = 0.38, 0.3   # metres around the room for the outward labels
# Tables read off the same deployment map, in room coordinates (x0, y0, width, height), metres
TABLES = ((0.0, 2.05, 0.8, 1.6), (0.0, 1.03, 1.6, 0.6))
TX_COLOURS = ("vermilion", "blue")


def to_plot(x: float, y: float):
    """Room (x across, y along) -> plot (along, across)."""
    return y, x


def outward(px: float, py: float, h: float, w: float) -> dict:
    """Place a node label outside the room, away from the wall the node sits on."""
    walls = {"bottom": py, "top": w - py, "left": px, "right": h - px}
    side = min(walls, key=walls.get)
    d = LABEL_OFFSET_PT
    return {"bottom": dict(xytext=(0, -d), ha="center", va="top"),
            "top": dict(xytext=(0, d), ha="center", va="bottom"),
            "left": dict(xytext=(-d, 0), ha="right", va="center"),
            "right": dict(xytext=(d, 0), ha="left", va="center")}[side]


def main() -> None:
    ps.apply()
    cfg = json.loads(ROOM_CONFIG.read_text())
    nodes = {n["mac"]: n for n in cfg["nodes"]}
    pairs = [l.split("->") for l in json.loads(PAIRS_JSON.read_text())["links"]]
    txs = sorted({a for a, _ in pairs}, key=lambda m: nodes[m]["label"])
    colour = {tx: ps.OKABE_ITO[c] for tx, c in zip(txs, TX_COLOURS)}
    w, h = cfg["room"]["width"], cfg["room"]["height"]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, FIG_H_IN), layout="constrained")
    ax.add_patch(Rectangle((0, 0), h, w, fill=False, linewidth=1.0, edgecolor=ps.OKABE_ITO["black"]))
    for x0, y0, tw, th in TABLES:
        px, py = to_plot(x0, y0)
        ax.add_patch(Rectangle((px, py), th, tw, facecolor="#E6E6E6", edgecolor="#AAAAAA", linewidth=0.5))
    for a, b in pairs:
        (xa, ya), (xb, yb) = to_plot(nodes[a]["x"], nodes[a]["y"]), to_plot(nodes[b]["x"], nodes[b]["y"])
        ax.plot([xa, xb], [ya, yb], color=colour[a], linewidth=0.7, alpha=0.75, zorder=2)
    for mac, n in nodes.items():
        px, py = to_plot(n["x"], n["y"])
        is_tx = mac in colour
        ax.scatter(px, py, s=46 if is_tx else 30, zorder=3, edgecolor=ps.OKABE_ITO["black"], linewidth=0.5,
                   color=colour.get(mac, "white"))
        ax.annotate(n["label"], (px, py), textcoords="offset points", fontsize=ps.SMALL_PT, **outward(px, py, h, w))
    for tx in txs:
        ax.plot([], [], color=colour[tx], label=f"Pairs sounded by {nodes[tx]['label']}")
    fig.legend(loc="outside lower center", ncol=2)
    ax.set_xlim(-MARGIN_X, h + MARGIN_X)
    ax.set_ylim(-MARGIN_Y, w + MARGIN_Y)
    ax.set_aspect("equal")
    ax.axis("off")
    ps.save(fig, "fig_office_layout")


if __name__ == "__main__":
    main()
