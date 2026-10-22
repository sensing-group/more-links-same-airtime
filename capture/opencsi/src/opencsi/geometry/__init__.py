"""Geometry utilities for OpenCSI (Fresnel zones, position solving, coverage)."""

from opencsi.geometry.fresnel import (
    compute_fresnel_radius_at_midpoint,
    compute_snr_scaling_factor,
    compute_effective_detection_width,
    compute_fresnel_confidence_bounds,
    requires_multilink_confirmation,
    near_field_correction,
    compute_effective_detection_width_with_near_field,
    is_near_field_link,
    NEAR_FIELD_BOUNDARY_M,
)
from opencsi.geometry.distance import (
    PathLossModel,
    DistanceEstimator,
    REFERENCE_RSSI_1M,
    DEFAULT_PATH_LOSS_EXPONENT,
    MIN_DISTANCE_M,
    MAX_DISTANCE_M,
)
from opencsi.geometry.position import (
    MDSResult,
    PositionSolver,
    MDS_MAX_ITERATIONS,
    SMACOF_STRESS_THRESHOLD,
    SMACOF_CONVERGENCE_TOL,
    SMACOF_TIMEOUT_SECONDS,
    OUTLIER_RESIDUAL_THRESHOLD,
    DEFAULT_FALLBACK_DISTANCE,
)
from opencsi.geometry.coverage import (
    CoverageEstimate,
    CoverageAreaEstimator,
    SEPARATION_FRACTION,
    GRID_FRACTION,
    MIN_GRID_RESOLUTION_M,
    MAX_GRID_RESOLUTION_M,
    MIN_SEPARATION_M,
    MAX_SEPARATION_M,
    DEFAULT_MARGIN_FRACTION,
    MIN_NODES_FOR_HULL,
)
from opencsi.geometry.ndf import (
    NDFResult,
    build_sensing_matrix,
    compute_ndf,
    analyze_mesh_topology,
    analyze_from_rssi_matrix,
    DEFAULT_GRID_RESOLUTION_M as NDF_DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_NDF_THRESHOLD,
)

__all__ = [
    # Fresnel zone calculations
    "compute_fresnel_radius_at_midpoint",
    "compute_snr_scaling_factor",
    "compute_effective_detection_width",
    "compute_fresnel_confidence_bounds",
    "requires_multilink_confirmation",
    "near_field_correction",
    "compute_effective_detection_width_with_near_field",
    "is_near_field_link",
    "NEAR_FIELD_BOUNDARY_M",
    # Distance estimation
    "PathLossModel",
    "DistanceEstimator",
    "REFERENCE_RSSI_1M",
    "DEFAULT_PATH_LOSS_EXPONENT",
    "MIN_DISTANCE_M",
    "MAX_DISTANCE_M",
    # Position solving
    "MDSResult",
    "PositionSolver",
    "MDS_MAX_ITERATIONS",
    "SMACOF_STRESS_THRESHOLD",
    "SMACOF_CONVERGENCE_TOL",
    "SMACOF_TIMEOUT_SECONDS",
    "OUTLIER_RESIDUAL_THRESHOLD",
    "DEFAULT_FALLBACK_DISTANCE",
    # Coverage area estimation
    "CoverageEstimate",
    "CoverageAreaEstimator",
    "SEPARATION_FRACTION",
    "GRID_FRACTION",
    "MIN_GRID_RESOLUTION_M",
    "MAX_GRID_RESOLUTION_M",
    "MIN_SEPARATION_M",
    "MAX_SEPARATION_M",
    "DEFAULT_MARGIN_FRACTION",
    "MIN_NODES_FOR_HULL",
    # NDF analysis
    "NDFResult",
    "build_sensing_matrix",
    "compute_ndf",
    "analyze_mesh_topology",
    "analyze_from_rssi_matrix",
    "NDF_DEFAULT_GRID_RESOLUTION_M",
    "DEFAULT_NDF_THRESHOLD",
]
