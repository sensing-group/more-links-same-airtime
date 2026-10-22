"""
Persistence helpers for AutoCalibrator.

Handles save/load (JSON), snapshot export/import (JSON-safe),
and Welford accumulator reconstruction from baseline statistics.

All functions take the calibrator instance (or cls) as the first
argument. The AutoCalibrator class delegates to these functions.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from opencsi.core.types import (
    CalibrationSnapshot,
    LinkSnapshot,
    PerLinkBaseline,
)
from opencsi.core.config import AutoCalibratorConfig
from opencsi.core.welford import WelfordAccumulator
from opencsi.baseline import PassiveFrequencyCollector, PassiveTimeCollector

if TYPE_CHECKING:
    from opencsi.core.calibrator import AutoCalibrator

logger = logging.getLogger(__name__)

# Cap stored quiet_samples to prevent unbounded growth
MAX_STORED_QUIET_SAMPLES = 10000


# =============================================================================
# Welford Reconstruction
# =============================================================================

def reconstruct_welford(baseline: PerLinkBaseline) -> Optional[WelfordAccumulator]:
    """
    Reconstruct a WelfordAccumulator from stored baseline statistics.

    This is an approximation since we store EMA, not exact Welford state.
    Used for computing confidence intervals and exporting snapshots.

    Args:
        baseline: PerLinkBaseline with stored statistics.

    Returns:
        WelfordAccumulator if baseline is initialized with >1 samples,
        None otherwise.
    """
    if not baseline.is_initialized or baseline.quiet_samples < 2:
        return None

    # Estimate m2 from variance: m2 = variance * (n - 1)
    m2 = baseline.amplitude_variance_scalar * (baseline.quiet_samples - 1)

    return WelfordAccumulator(
        n=baseline.quiet_samples,
        mean=baseline.amplitude_mean_scalar,
        m2=m2,
    )


# =============================================================================
# Snapshot Export/Import
# =============================================================================

def export_snapshot(calibrator: 'AutoCalibrator') -> CalibrationSnapshot:
    """
    Export portable calibration snapshot (JSON-serializable).

    Creates a complete snapshot of all calibration state suitable for
    external persistence (SQLite, cloud storage, etc.).

    Args:
        calibrator: AutoCalibrator instance.

    Returns:
        CalibrationSnapshot with all link calibration data.
    """
    timestamp = time.time()
    links: Dict[str, LinkSnapshot] = {}

    for link_key, state in calibrator._link_states.items():
        tx_mac, rx_mac = link_key
        baseline = state.baseline

        acc = reconstruct_welford(baseline)
        if acc is not None:
            amplitude_ci = acc.confidence_interval(0.95)
            welford_n = acc.n
            welford_mean = acc.mean
            welford_m2 = acc.m2
        else:
            amplitude_ci = None
            welford_n = None
            welford_mean = None
            welford_m2 = None

        link_snap = LinkSnapshot(
            tx_mac=tx_mac,
            rx_mac=rx_mac,
            timestamp=timestamp,
            amplitude_mean=baseline.amplitude_mean_scalar,
            amplitude_std=baseline.amplitude_std_scalar,
            amplitude_ci=amplitude_ci,
            phase_std=baseline.phase_std_scalar,
            quiet_samples=baseline.quiet_samples,
            is_initialized=baseline.is_initialized,
            welford_n=welford_n,
            welford_mean=welford_mean,
            welford_m2=welford_m2,
            baseline_updates_gated=state.baseline_updates_gated,
        )
        links[link_snap.link_key] = link_snap

    return CalibrationSnapshot(
        version=CalibrationSnapshot.CURRENT_VERSION,
        timestamp=timestamp,
        links=links,
    )


def restore_from_snapshot(
    cls,
    snapshot: CalibrationSnapshot,
    node_positions: Dict[str, Tuple[float, float]],
    config: Optional[AutoCalibratorConfig] = None,
    hardware_profile: Optional[dict] = None,
) -> 'AutoCalibrator':
    """
    Restore AutoCalibrator from a snapshot.

    Creates a new AutoCalibrator instance with baseline statistics
    restored from the snapshot.

    Args:
        cls: AutoCalibrator class (for construction).
        snapshot: CalibrationSnapshot to restore from.
        node_positions: MAC -> (x, y) position mapping.
        config: Optional configuration overrides.
        hardware_profile: Optional hardware profile for cold start.

    Returns:
        AutoCalibrator instance with restored baseline statistics.
    """
    instance = cls(
        node_positions=node_positions,
        config=config,
        hardware_profile=hardware_profile,
    )

    for _link_key_str, link_snap in snapshot.links.items():
        tx_mac = link_snap.tx_mac
        rx_mac = link_snap.rx_mac
        link_key = (tx_mac, rx_mac)

        link_state = instance._create_link_state(tx_mac, rx_mac)
        link_state.baseline.amplitude_mean_scalar = link_snap.amplitude_mean
        link_state.baseline.amplitude_std_scalar = link_snap.amplitude_std
        link_state.baseline.amplitude_variance_scalar = link_snap.amplitude_std ** 2
        link_state.baseline.phase_std_scalar = link_snap.phase_std
        link_state.baseline.quiet_samples = link_snap.quiet_samples
        link_state.baseline.is_initialized = link_snap.is_initialized
        link_state.baseline.last_update_time = link_snap.timestamp
        link_state.baseline.creation_time = link_snap.timestamp
        link_state.baseline_updates_gated = link_snap.baseline_updates_gated

        instance._link_states[link_key] = link_state

    if instance._link_states:
        instance._agc_warmup_complete = True

    logger.info(
        f"Restored AutoCalibrator from snapshot: "
        f"{len(instance._link_states)} links, "
        f"snapshot_time={snapshot.timestamp}"
    )
    return instance


def get_link_confidence(
    calibrator: 'AutoCalibrator',
    tx_mac: str,
    rx_mac: str,
) -> Optional[Tuple[float, float]]:
    """
    Get 95% confidence interval for link's amplitude mean.

    Args:
        calibrator: AutoCalibrator instance.
        tx_mac: Transmitter MAC address.
        rx_mac: Receiver MAC address.

    Returns:
        Tuple of (lower, upper) bounds for 95% CI, or None if
        link not found or not enough samples.
    """
    link_key = (tx_mac, rx_mac)
    if link_key not in calibrator._link_states:
        return None

    state = calibrator._link_states[link_key]
    acc = reconstruct_welford(state.baseline)
    if acc is None:
        return None

    return acc.confidence_interval(0.95)


# =============================================================================
# JSON Persistence (v7+), with legacy pickle fallback for reading
# =============================================================================

# State file format version.  Bump when the schema changes.
STATE_FORMAT_VERSION = 7

def compact_baselines(calibrator: 'AutoCalibrator') -> dict:
    """Compact baseline data for efficient storage.

    Aggregation policy:
    - Cap quiet_samples at MAX_STORED_QUIET_SAMPLES
    - Only store essential statistics (mean, std, variance)
    - Discard per-sample history (already aggregated into EMA)
    """
    compacted = {}

    for k, v in calibrator._link_states.items():
        capped_samples = min(
            v.baseline.quiet_samples,
            MAX_STORED_QUIET_SAMPLES,
        )

        compacted[f"{k[0]}->{k[1]}"] = {
            'tx_mac': v.baseline.tx_mac,
            'rx_mac': v.baseline.rx_mac,
            'amplitude_mean_scalar': v.baseline.amplitude_mean_scalar,
            'amplitude_std_scalar': v.baseline.amplitude_std_scalar,
            'amplitude_variance_scalar': v.baseline.amplitude_variance_scalar,
            'phase_std_scalar': v.baseline.phase_std_scalar,
            'quiet_samples': capped_samples,
            'last_update_time': v.baseline.last_update_time,
            'creation_time': v.baseline.creation_time,
            'is_initialized': v.baseline.is_initialized,
            'baseline_updates_gated': v.baseline_updates_gated,
        }

    return compacted


def _try_read_existing_state(path_obj: Path) -> Optional[dict]:
    """Read an existing state file (JSON or legacy pickle) for safety checks."""
    if not path_obj.exists():
        return None
    try:
        with open(path_obj, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    # Legacy pickle fallback
    try:
        with open(path_obj, 'rb') as f:
            state = pickle.load(f)
        logger.warning(
            f"Read legacy pickle state from {path_obj}. "
            "Re-save to migrate to JSON format."
        )
        return state
    except Exception:
        return None


def save_state(calibrator: 'AutoCalibrator', path: str, force: bool = False):
    """Persist learned calibration state as JSON with safe atomic writes.

    Uses backup rotation to prevent data loss:
    1. Keeps .bak backup of previous good state
    2. Only overwrites if new state has calibrated links (unless force=True)
    3. Uses atomic write (temp file + rename)

    Args:
        calibrator: AutoCalibrator instance.
        path: Path to save state file.
        force: If True, save even if no calibrated links.
    """
    calibrated_count = sum(
        1 for s in calibrator._link_states.values()
        if s.baseline.is_initialized
    )

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Safety check: don't overwrite good state with empty state
    if not force and calibrated_count == 0 and path_obj.exists():
        existing = _try_read_existing_state(path_obj)
        if existing is not None:
            existing_calibrated = sum(
                1 for bl in existing.get('link_baselines', {}).values()
                if bl.get('is_initialized', False)
            )
            if existing_calibrated > 0:
                logger.warning(
                    f"Skipping save: would overwrite {existing_calibrated} calibrated links with 0"
                )
                return

    compacted_baselines = compact_baselines(calibrator)

    state = {
        'version': STATE_FORMAT_VERSION,
        'node_positions': {k: list(v) for k, v in calibrator.node_positions.items()},
        'config': calibrator.config.to_dict(),
        'link_baselines': compacted_baselines,
        'sensitivity_learner': calibrator.sensitivity_learner.to_dict(),
        'antenna_learner': calibrator.antenna_learner.to_dict(),
        'frequency_collector': calibrator._passive_frequency_collector.to_dict(),
        'time_collector': calibrator._passive_time_collector.to_dict(),
        'total_frames': calibrator._total_frames_processed,
        'quiet_frames': calibrator._quiet_frames,
        'active_frames': calibrator._active_frames,
        'saved_at': time.time(),
        'calibrated_links': calibrated_count,
    }

    temp_path = path_obj.with_suffix('.json.tmp')
    backup_path = path_obj.with_suffix('.bak')

    try:
        with open(temp_path, 'w') as f:
            json.dump(state, f, indent=2)

        if path_obj.exists():
            try:
                path_obj.rename(backup_path)
            except OSError:
                pass

        temp_path.rename(path_obj)

        logger.info(
            f"Saved calibration state to {path} "
            f"({calibrated_count}/{len(calibrator._link_states)} calibrated)"
        )

    except Exception as e:
        logger.error(f"Failed to save calibration state: {e}")
        if temp_path.exists():
            temp_path.unlink()


def try_load_state(path: Path) -> Optional[dict]:
    """Try to load state from a file, returning None if invalid.

    Tries JSON first, then falls back to legacy pickle for backward
    compatibility.  Validates that the state has calibrated links.
    """
    if not path.exists():
        return None

    state = None

    # --- Try JSON first ---
    try:
        with open(path, 'r') as f:
            state = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    except Exception as e:
        logger.debug(f"JSON load failed for {path}: {e}")

    # --- Fall back to legacy pickle ---
    if state is None:
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            logger.warning(
                f"Loaded legacy pickle state from {path}. "
                "Re-save to migrate to JSON format."
            )
        except Exception as e:
            logger.warning(f"Failed to load state from {path}: {e}")
            return None

    if not isinstance(state, dict):
        logger.warning(f"Invalid state format in {path}")
        return None

    baselines = state.get('link_baselines', {})
    calibrated = sum(
        1 for bl in baselines.values()
        if bl.get('is_initialized', False)
    )

    if len(baselines) == 0:
        logger.warning(f"State file {path} has no links")
        return None

    if calibrated == 0:
        logger.info(
            f"State file {path} has {len(baselines)} links but none calibrated yet"
        )

    logger.info(
        f"Loaded state from {path}: {calibrated}/{len(baselines)} calibrated"
    )
    return state


def load_calibrator(
    cls,
    path: str,
    node_positions: Dict[str, Tuple[float, float]],
    config: Optional[AutoCalibratorConfig] = None,
    hardware_profile: Optional[dict] = None,
) -> 'AutoCalibrator':
    """Load previously learned calibration with backup recovery.

    Tries to load from:
    1. Primary state file
    2. Backup file (.bak) if primary is corrupted/empty
    3. Fresh instance if both fail

    Args:
        cls: AutoCalibrator class (for construction).
        path: Path to state file.
        node_positions: MAC -> (x, y) position mapping.
        config: Optional configuration overrides.
        hardware_profile: Optional hardware profile for cold start.

    Returns:
        AutoCalibrator instance (restored or fresh).
    """
    from opencsi.learning import SensitivityMapLearner, AntennaPatternLearner, compute_distance
    from opencsi.core.calibrator import SIGNIFICANT_POSITION_CHANGE_M

    path_obj = Path(path)
    backup_path = path_obj.with_suffix('.bak')

    state = try_load_state(path_obj)

    if state is None:
        logger.warning(f"Primary state file failed, trying backup: {backup_path}")
        state = try_load_state(backup_path)
        if state is not None:
            logger.info("Recovered from backup file")

    if state is None:
        logger.warning("No valid state file found, creating fresh instance")
        return cls(
            node_positions=node_positions,
            config=config,
            hardware_profile=hardware_profile,
        )

    try:
        if config is not None:
            loaded_config = config
        elif 'config' in state and isinstance(state['config'], dict):
            loaded_config = AutoCalibratorConfig.from_dict(state['config'])
        elif 'config' in state and isinstance(state['config'], AutoCalibratorConfig):
            # Legacy pickle: config was stored as a raw dataclass object
            loaded_config = state['config']
        else:
            loaded_config = AutoCalibratorConfig()
        instance = cls(
            node_positions=node_positions,
            config=loaded_config,
            hardware_profile=hardware_profile,
        )

        # Check for position changes
        saved_positions = {
            k: tuple(v) for k, v in state.get('node_positions', {}).items()
        }
        position_changed = False
        for mac, new_pos in node_positions.items():
            if mac in saved_positions:
                old_pos = saved_positions[mac]
                if compute_distance(old_pos, new_pos) > SIGNIFICANT_POSITION_CHANGE_M:
                    logger.warning(f"Node {mac} position changed: {old_pos} -> {new_pos}")
                    position_changed = True

        # Restore link states
        for key_str, baseline_data in state.get('link_baselines', {}).items():
            parts = key_str.split('->')
            if len(parts) != 2:
                continue

            link_key = (parts[0], parts[1])
            tx_mac = baseline_data['tx_mac']
            rx_mac = baseline_data['rx_mac']

            link_state = instance._create_link_state(tx_mac, rx_mac)
            link_state.baseline.amplitude_mean_scalar = baseline_data.get(
                'amplitude_mean_scalar', 0.0
            )
            link_state.baseline.amplitude_std_scalar = baseline_data.get(
                'amplitude_std_scalar', 1.0
            )
            link_state.baseline.amplitude_variance_scalar = baseline_data.get(
                'amplitude_variance_scalar',
                link_state.baseline.amplitude_std_scalar ** 2,
            )
            link_state.baseline.phase_std_scalar = baseline_data.get(
                'phase_std_scalar', 1.0
            )
            link_state.baseline.quiet_samples = baseline_data.get('quiet_samples', 0)
            link_state.baseline.last_update_time = baseline_data.get(
                'last_update_time', 0.0
            )
            link_state.baseline.creation_time = baseline_data.get('creation_time', 0.0)
            link_state.baseline.is_initialized = baseline_data.get(
                'is_initialized', False
            )
            link_state.baseline_updates_gated = baseline_data.get(
                'baseline_updates_gated', 0
            )

            instance._link_states[link_key] = link_state

        # Restore sensitivity learner
        if not position_changed and 'sensitivity_learner' in state:
            try:
                instance.sensitivity_learner = SensitivityMapLearner.from_dict(
                    state['sensitivity_learner']
                )
            except Exception as e:
                logger.warning(f"Failed to restore sensitivity learner: {e}")

        # Restore antenna learner
        if not position_changed and 'antenna_learner' in state:
            try:
                instance.antenna_learner = AntennaPatternLearner.from_dict(
                    state['antenna_learner']
                )
            except Exception as e:
                logger.warning(f"Failed to restore antenna learner: {e}")

        # Restore frequency collector
        if 'frequency_collector' in state:
            try:
                instance._passive_frequency_collector = PassiveFrequencyCollector.from_dict(
                    state['frequency_collector']
                )
            except Exception as e:
                logger.warning(f"Failed to restore frequency collector: {e}")

        # Restore time collector
        if 'time_collector' in state:
            try:
                instance._passive_time_collector = PassiveTimeCollector.from_dict(
                    state['time_collector']
                )
            except Exception as e:
                logger.warning(f"Failed to restore time collector: {e}")

        # Restore statistics
        instance._total_frames_processed = state.get('total_frames', 0)
        instance._quiet_frames = state.get('quiet_frames', 0)
        instance._active_frames = state.get('active_frames', 0)

        instance._agc_warmup_complete = True

        logger.info(
            f"Loaded calibration state from {path} "
            f"({len(instance._link_states)} links, "
            f"positions_changed={position_changed})"
        )
        return instance

    except Exception as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return cls(
            node_positions=node_positions,
            config=config,
            hardware_profile=hardware_profile,
        )
