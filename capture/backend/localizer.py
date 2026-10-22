"""Pseudo-localization via weighted centroid of active-link intersections."""
import math
from typing import Dict, List, Optional, Tuple

# Minimum angle (degrees) between two links for a valid intersection.
MIN_ANGLE_DEG = 20.0
# Room-bounds margin (metres) — allow intersections slightly outside walls.
BOUNDS_MARGIN_M = 0.3
# Fresnel-zone half-width used only for labelling; not needed here but
# kept as documentation of the physical model.
FRESNEL_RADIUS_M = 0.8


def _line_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Intersection of infinite lines (p1-p2) and (p3-p4).

    Returns None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def _angle_between(
    d1: Tuple[float, float], d2: Tuple[float, float],
) -> float:
    """Acute angle (radians) between two direction vectors."""
    dot = d1[0] * d2[0] + d1[1] * d2[1]
    m1 = math.hypot(*d1)
    m2 = math.hypot(*d2)
    if m1 < 1e-12 or m2 < 1e-12:
        return 0.0
    cos_a = max(-1.0, min(1.0, abs(dot) / (m1 * m2)))
    return math.acos(cos_a)


def estimate(
    active_links: List[dict],
    node_positions: Dict[str, Tuple[float, float]],
    room_width: float,
    room_height: float,
) -> Optional[dict]:
    """Compute pseudo-localisation from active-link intersections.

    Args:
        active_links: List of {"tx": mac, "rx": mac, "z_score": float}.
        node_positions: {mac: (x_m, y_m)} for positioned nodes.
        room_width:  Room X extent in metres.
        room_height: Room Y extent in metres.

    Returns:
        {"x", "y", "radius", "n_intersections"} or None.
    """
    # --- Deduplicate directed → undirected, keep higher |z| ---
    undir: Dict[frozenset, dict] = {}
    for link in active_links:
        key = frozenset((link["tx"], link["rx"]))
        if key not in undir or abs(link["z_score"]) > abs(undir[key]["z_score"]):
            undir[key] = link
    # Filter to links where both endpoints are positioned
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    for key, link in undir.items():
        tx_pos = node_positions.get(link["tx"])
        rx_pos = node_positions.get(link["rx"])
        if tx_pos is None or rx_pos is None:
            continue
        segments.append((tx_pos, rx_pos, abs(link["z_score"])))

    if len(segments) < 2:
        return None

    min_angle_rad = math.radians(MIN_ANGLE_DEG)
    lo_x, hi_x = -BOUNDS_MARGIN_M, room_width + BOUNDS_MARGIN_M
    lo_y, hi_y = -BOUNDS_MARGIN_M, room_height + BOUNDS_MARGIN_M

    points: List[Tuple[float, float, float]] = []  # (x, y, weight)

    for i in range(len(segments)):
        p1, p2, z1 = segments[i]
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        for j in range(i + 1, len(segments)):
            p3, p4, z2 = segments[j]
            d2 = (p4[0] - p3[0], p4[1] - p3[1])

            angle = _angle_between(d1, d2)
            if angle < min_angle_rad:
                continue

            pt = _line_intersection(p1, p2, p3, p4)
            if pt is None:
                continue

            if not (lo_x <= pt[0] <= hi_x and lo_y <= pt[1] <= hi_y):
                continue

            weight = min(z1, z2) * math.sin(angle)
            if weight <= 0:
                continue
            points.append((pt[0], pt[1], weight))

    if not points:
        return None

    total_w = sum(w for _, _, w in points)
    if total_w < 1e-12:
        return None

    cx = sum(x * w for x, _, w in points) / total_w
    cy = sum(y * w for _, y, w in points) / total_w

    # Weighted std deviation as confidence radius
    if len(points) == 1:
        radius = 1.0
    else:
        var = sum(w * ((x - cx) ** 2 + (y - cy) ** 2) for x, y, w in points) / total_w
        radius = math.sqrt(max(0.0, var))

    # Clamp estimate inside room bounds
    cx = max(0.0, min(room_width, cx))
    cy = max(0.0, min(room_height, cy))

    return {
        "x": round(cx, 3),
        "y": round(cy, 3),
        "radius": round(max(radius, 0.1), 3),
        "n_intersections": len(points),
    }
