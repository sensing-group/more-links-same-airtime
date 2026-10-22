"""
Canonical record schemas for CSI data storage.

All timestamps are Unix float64 seconds (UTC).
All IQ values are int8 (as captured from firmware).

Schema version 2.0: trial-centric storage with hybrid anchored timestamps.
"""
import dataclasses
from typing import Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.0"


@dataclasses.dataclass(frozen=True, slots=True)
class RawCSIRecord:
    record_id:       str      # uuid4 hex, no dashes
    trial_id:        str      # "T0001"
    seq:             int      # firmware packet sequence number

    # Timestamps (three-field hybrid system)
    wall_ts:         float    # PRIMARY — reconstructed UTC (offset + firmware_ts_us/1e6)
    server_ts:       float    # Fallback — time.time() on server at receipt (UTC, jittery)
    firmware_ts_us:  int      # Raw — device monotonic µs since boot (for reproducibility)

    tx_mac:          str
    rx_mac:          str
    rssi:            int      # dBm
    noise_floor:     int      # dBm
    snr:             float    # rssi - noise_floor

    num_subcarriers: int
    bandwidth:       str      # "HT20" or "HT40"
    iq_data:         list     # [[I0,Q0], [I1,Q1], ...] per-subcarrier int8 pairs

    amplitude_mean:  float
    amplitude_std:   float
    amplitude_var:   float


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedRecord:
    record_id:           str
    trial_id:            str      # "T0001"
    seq:                 int

    # Timestamps (two-field: wall_ts propagated from raw, server_ts for fallback)
    wall_ts:             float    # PRIMARY — from corresponding RawCSIRecord
    server_ts:           float    # Fallback

    tx_mac:              str
    rx_mac:              str

    amplitude_var:       float
    baseline_mean:       float
    baseline_std:        float
    baseline_samples:    int
    z_score:             Optional[float]
    state:               str     # "calibrating" | "monitoring"
    rssi:                int
    noise_floor:         int
    snr:                 float


@dataclasses.dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """
    Interval annotation following OPPORTUNITY dataset conventions.
    Labels are intervals [start_ts, end_ts]. Point annotations use end_ts == start_ts.
    """
    annotation_id:   str
    trial_id:        str      # "T0001"

    start_ts:        float    # UTC
    end_ts:          float    # UTC; == start_ts for point label
    server_ts:       float
    duration_s:      float    # end_ts - start_ts

    coarse_label:    str      # e.g. "locomotion"
    fine_label:      str      # e.g. "walking"

    annotator_id:    str
    confidence:      float    # 0.0–1.0
    note:            str
    source:          str      # "manual" | "auto" | "import"


@dataclasses.dataclass(frozen=True, slots=True)
class TrialMetadata:
    """Full trial metadata written to trial.json."""
    trial_id:        str       # "T0001"
    activity:        str       # Primary fine_label or "mixed"
    activities:      list[str] # All fine_labels present
    subject_id:      str       # "S01"
    environment_id:  str       # "lab-alpha"
    topology_id:     str       # "topo-9node-variable-height"
    repetition:      int       # Which repetition of this combo
    tags:            list[str] # Flexible: ["baseline", "los", "nlos"]

    started_at:      str       # ISO 8601 with timezone
    stopped_at:      str       # ISO 8601
    duration_s:      float

    clock_sync:      dict      # Per-node offsets at start/stop
    collection:      dict      # CollectionMeta as dict
    notes:           str


@dataclasses.dataclass(frozen=True, slots=True)
class CollectionMeta:
    hostname:        str
    schema_version:  str
    compress:        bool
    raw_packets:     int
    cal_packets:     int
    dropped:         int


class AnnotationCreate(BaseModel):
    fine_label:    str
    start_ts:      Optional[float] = None
    end_ts:        Optional[float] = None
    duration_s:    Optional[float] = None
    annotator_id:  str = "default"
    confidence:    float = Field(default=1.0, ge=0.0, le=1.0)
    note:          str = ""
    source:        str = "manual"


class TrialStartRequest(BaseModel):
    """Required parameters to start a new trial recording."""
    activity:        str
    subject_id:      str
    environment_id:  str
    topology_id:     str
    tags:            list[str] = []
    notes:           str = ""
    duration_s:      float | None = None  # if set, recording auto-stops after this many seconds


ACTIVITY_TAXONOMY: dict[str, list[str]] = {
    "locomotion": [
        "walking", "pacing",
    ],
    "stationary": [
        "sitting", "standing", "working_at_desk", "phone_call", "reading",
    ],
    "transition": [
        "sitting_down", "standing_up", "entering_room", "leaving_room",
    ],
    "exercise": [
        "push_ups", "jumping_jacks",
    ],
    "environment": [
        "empty_room", "multiple_people",
        "occupied_static", "occupied_moving",
    ],
    "composite": [
        "mixed",
    ],
    "other": [
        "other",
    ],
}

FINE_TO_COARSE: dict[str, str] = {
    fine: coarse
    for coarse, fines in ACTIVITY_TAXONOMY.items()
    for fine in fines
}

ALL_ACTIVITIES: list[str] = [f for fines in ACTIVITY_TAXONOMY.values() for f in fines]
