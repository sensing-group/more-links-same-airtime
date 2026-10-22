"""
Fresnel Zone Geometry Calculations for OpenCSI.

This module provides Fresnel zone calculations used by the spatial learning
components. Fresnel zones define the ellipsoidal region where signal
propagation is affected.

Key Functions:
- compute_fresnel_radius: Calculate Fresnel radius at a point
- theoretical_fresnel_response: Expected response at a position
- compute_angle/compute_distance: Geometry helpers
"""

import math
from typing import Tuple

# WiFi parameters
WIFI_WAVELENGTH_M = 0.125  # 2.4 GHz wavelength


def compute_angle(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
    """
    Compute angle from one position to another.

    Args:
        from_pos: (x, y) source position
        to_pos: (x, y) target position

    Returns:
        Angle in radians [-π, π], where 0 is +x direction
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return math.atan2(dy, dx)


def compute_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two positions."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx * dx + dy * dy)


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    Compute perpendicular distance from a point to a line segment.

    Args:
        point: (x, y) query point
        line_start: (x, y) line segment start
        line_end: (x, y) line segment end

    Returns:
        Distance from point to line segment
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line vector
    dx = x2 - x1
    dy = y2 - y1

    # Length squared
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-10:
        # Degenerate line (start == end)
        return compute_distance(point, line_start)

    # Parameter t for closest point on infinite line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return compute_distance(point, (closest_x, closest_y))


def compute_fresnel_radius(
    tx_pos: Tuple[float, float],
    rx_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
    n: int = 1,
    wavelength: float = WIFI_WAVELENGTH_M
) -> float:
    """
    Compute the radius of the nth Fresnel zone at a given point.

    The Fresnel zone is an ellipsoid around the TX-RX line. At any point,
    the radius depends on where along the line the point projects.

    Args:
        tx_pos: Transmitter position (x, y)
        rx_pos: Receiver position (x, y)
        target_pos: Point where we want the radius
        n: Fresnel zone number (1 = first zone)
        wavelength: Signal wavelength in meters

    Returns:
        Fresnel radius in meters at the target position
    """
    d_tx = compute_distance(tx_pos, target_pos)
    d_rx = compute_distance(rx_pos, target_pos)
    d_total = compute_distance(tx_pos, rx_pos)

    if d_total < 1e-6:
        return 0.0

    # Distance along line from TX to projection of target
    # Using the ellipse parameter
    d1 = d_tx
    d2 = d_rx

    # Fresnel radius formula: r_n = sqrt(n * λ * d1 * d2 / (d1 + d2))
    radius = math.sqrt(n * wavelength * d1 * d2 / (d1 + d2 + 1e-10))
    return radius


def theoretical_fresnel_response(
    tx_pos: Tuple[float, float],
    rx_pos: Tuple[float, float],
    target_pos: Tuple[float, float]
) -> float:
    """
    Compute theoretical Fresnel zone response at a position.

    Models the expected signal disturbance from an object at the target
    position, assuming isotropic antennas and first Fresnel zone.

    Args:
        tx_pos: Transmitter position (x, y)
        rx_pos: Receiver position (x, y)
        target_pos: Object position (x, y)

    Returns:
        Expected response magnitude [0, 1]
        1.0 = on the TX-RX line (maximum disturbance)
        0.0 = far outside Fresnel zone
    """
    # Distance from target to TX-RX line
    perp_dist = point_to_line_distance(target_pos, tx_pos, rx_pos)

    # Fresnel radius at this point
    fresnel_r = compute_fresnel_radius(tx_pos, rx_pos, target_pos, n=1)

    if fresnel_r < 1e-6:
        return 0.0

    # Normalized distance (0 = on line, 1 = at Fresnel boundary)
    normalized_dist = perp_dist / fresnel_r

    # Response drops off as Gaussian (approximation)
    # Objects at Fresnel boundary cause ~0.6 of maximum disturbance
    response = math.exp(-0.5 * normalized_dist * normalized_dist)

    return response
