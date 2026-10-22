"""
Experiment runner — timed activity data collection with auto-labeling.

State machine:
  idle → baseline (empty room) → [position → active] × N → done → idle

Each phase transition emits a "beep" event via the status dict so the frontend
can play an audio cue. Annotations are created automatically with precise timestamps.

Supports multiple activities in sequence: after the shared baseline, each activity
gets its own position (get-ready) + active phase. A configurable transition time
between activities lets the subject get into the next position.
"""
import asyncio
import dataclasses
import logging
import time
from typing import Optional

from pydantic import BaseModel

from schema import AnnotationCreate, TrialStartRequest, ALL_ACTIVITIES

logger = logging.getLogger(__name__)


class ActivityStep(BaseModel):
    activity: str          # fine_label from taxonomy
    activity_s: float = 30.0
    note: str = ""


class ExperimentConfig(BaseModel):
    # Multi-activity list (preferred)
    activities: list[ActivityStep] = []

    # Legacy single-activity fields (auto-converted to activities list)
    activity: str = ""
    activity_s: float = 30.0

    baseline_s: float = 30.0         # empty room period (once at start)
    position_s: float = 10.0         # get-in-position before first activity
    transition_s: float = 10.0       # get-in-position between activities
    note: str = ""

    # Trial metadata (required for recording)
    subject_id: str = ""
    environment_id: str = ""
    topology_id: str = ""
    tags: list[str] = []

    def get_steps(self) -> list[ActivityStep]:
        """Return normalised activity list, supporting legacy single-activity config."""
        if self.activities:
            return self.activities
        if self.activity:
            return [ActivityStep(activity=self.activity, activity_s=self.activity_s, note=self.note)]
        return []


@dataclasses.dataclass
class _PhaseInfo:
    name: str           # "baseline" | "position" | "active" | "done"
    duration_s: float
    started_at: float = 0.0


class ExperimentRunner:
    def __init__(self, recorder=None, labeler=None):
        self._recorder = recorder
        self._labeler = labeler
        self._task: Optional[asyncio.Task] = None

        self._phase: str = "idle"          # idle | baseline | position | active | done
        self._config: Optional[ExperimentConfig] = None
        self._phase_start: float = 0.0
        self._phase_duration: float = 0.0
        self._experiment_start: float = 0.0
        self._total_elapsed: float = 0.0     # frozen at experiment end
        self._beep: Optional[str] = None   # set briefly on phase transitions for frontend
        self._activity_index: int = 0      # current activity (0-based)
        self._activity_count: int = 0      # total activities
        self._current_activity: str = ""   # current activity label

    def get_status(self) -> dict:
        now = time.time()
        elapsed = now - self._phase_start if self._phase != "idle" else 0
        remaining = max(0, self._phase_duration - elapsed) if self._phase not in ("idle", "done") else 0

        status = {
            "phase": self._phase,
            "activity": self._current_activity,
            "elapsed_s": round(elapsed, 1),
            "remaining_s": round(remaining, 1),
            "phase_duration_s": self._phase_duration,
            "total_elapsed_s": round(self._total_elapsed, 1) if self._phase in ("idle", "done") else round(now - self._experiment_start, 1),
            "activity_index": self._activity_index,
            "activity_count": self._activity_count,
        }
        if self._config:
            status["config"] = self._config.model_dump()

        # Consume beep (read-once)
        if self._beep:
            status["beep"] = self._beep
            self._beep = None

        return status

    async def start(self, config: ExperimentConfig):
        if self._phase != "idle":
            raise RuntimeError("Experiment already running")

        steps = config.get_steps()
        if not steps:
            raise ValueError("No activities configured")
        for step in steps:
            if step.activity not in ALL_ACTIVITIES:
                raise ValueError(f"Unknown activity '{step.activity}'. Valid: {ALL_ACTIVITIES}")

        self._config = config
        self._activity_count = len(steps)
        self._activity_index = 0
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """Abort experiment."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._recorder and self._recorder.recording:
            await self._recorder.stop_recording()
        self._phase = "idle"
        self._config = None
        self._task = None
        self._current_activity = ""
        logger.info("Experiment aborted")

    async def _run(self):
        cfg = self._config
        steps = cfg.get_steps()
        try:
            self._experiment_start = time.time()
            activity_names = ", ".join(s.activity for s in steps)

            if self._recorder and not self._recorder.recording:
                # Determine primary activity for trial naming
                primary = steps[0].activity if len(steps) == 1 else "mixed"
                trial_req = TrialStartRequest(
                    activity=primary,
                    subject_id=cfg.subject_id,
                    environment_id=cfg.environment_id,
                    topology_id=cfg.topology_id,
                    tags=cfg.tags,
                    notes=f"experiment: {activity_names}",
                )
                await self._recorder.start_recording(trial_req)

            self._current_activity = steps[0].activity
            await self._enter_phase("baseline", cfg.baseline_s)
            if self._labeler and self._recorder:
                try:
                    await self._labeler.add_annotation(AnnotationCreate(
                        fine_label="empty_room",
                        start_ts=self._phase_start,
                        duration_s=cfg.baseline_s,
                        source="experiment",
                        note=f"baseline for {activity_names}",
                    ))
                except Exception:
                    logger.exception("Failed to label baseline")
            await asyncio.sleep(cfg.baseline_s)

            for i, step in enumerate(steps):
                self._activity_index = i
                self._current_activity = step.activity

                wait_s = cfg.position_s if i == 0 else cfg.transition_s
                self._beep = "position"
                await self._enter_phase("position", wait_s)
                await asyncio.sleep(wait_s)

                self._beep = "start"
                activity_start = time.time()
                await self._enter_phase("active", step.activity_s)
                await asyncio.sleep(step.activity_s)
                activity_end = time.time()

                if self._labeler:
                    try:
                        await self._labeler.add_annotation(AnnotationCreate(
                            fine_label=step.activity,
                            start_ts=activity_start,
                            end_ts=activity_end,
                            source="experiment",
                            note=step.note or cfg.note,
                        ))
                    except Exception:
                        logger.exception("Failed to label activity %s", step.activity)

                logger.info("Activity %d/%d done: %s (%.0fs)",
                            i + 1, len(steps), step.activity, step.activity_s)

            self._beep = "done"
            self._phase = "done"
            self._phase_start = time.time()
            self._phase_duration = 0
            self._total_elapsed = time.time() - self._experiment_start

            try:
                if self._recorder and self._recorder.recording:
                    await self._recorder.stop_recording()
            except Exception:
                logger.exception("Failed to stop recording at experiment end")

            logger.info("Experiment complete: %d activities (%.0fs baseline)",
                        len(steps), cfg.baseline_s)

            # Stay in "done" for a few seconds so frontend can show it
            await asyncio.sleep(5)
            self._phase = "idle"
            self._config = None
            self._current_activity = ""

        except asyncio.CancelledError:
            self._total_elapsed = time.time() - self._experiment_start
            self._phase = "idle"
            self._config = None
            self._current_activity = ""
            raise

    async def _enter_phase(self, name: str, duration: float):
        self._phase = name
        self._phase_start = time.time()
        self._phase_duration = duration
        logger.info("Experiment phase: %s (%.0fs) — activity %d/%d: %s",
                     name, duration, self._activity_index + 1,
                     self._activity_count, self._current_activity)
