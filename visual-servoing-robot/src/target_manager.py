"""Manual multi-target selection with isolated image-plane estimator state.

``TargetManager`` has no PyBullet, world-coordinate, depth, camera-pose, or
visual-servo dependency.  It consumes only the RGB detections for the stable
RED/GREEN/BLUE IDs and their timestamps.  The Stage 16 runner is responsible
for adapting the one selected RGB detection to the already frozen controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from multi_target_detector import DetectedTarget, TARGET_CLASSES
from target_motion_estimator import MotionEstimate, TargetMotionEstimator


@dataclass(frozen=True)
class TargetSwitchEvent:
    """One explicit manual or protocol target-selection change."""

    timestamp_s: float
    previous_target_id: str
    selected_target_id: str
    previous_class: str
    selected_class: str
    source: str


class TargetManager:
    """Own target selection and separate vision-only velocity histories.

    Every recognised colour receives its own ``TargetMotionEstimator``.  A
    switch never transfers a prior target's velocity to the new target.  The
    new selected target deliberately uses its current measurement for one
    frame, even when its independent estimator already has valid history.
    """

    def __init__(
        self,
        ema_alpha: float,
        prediction_horizon_s: float,
        initial_target_id: str = "target_1",
    ) -> None:
        self._class_by_id = {config.target_id: config.class_name for config in TARGET_CLASSES}
        if initial_target_id not in self._class_by_id:
            raise ValueError(f"Unknown initial target id: {initial_target_id}")
        self.available_targets: dict[str, DetectedTarget] = {}
        self.estimators = {
            target_id: TargetMotionEstimator(ema_alpha, prediction_horizon_s)
            for target_id in self._class_by_id
        }
        self.selected_target_id = initial_target_id
        self.selected_class = self._class_by_id[initial_target_id]
        self.selection_mode = "MANUAL"
        self.last_selected_target: str | None = None
        self.last_selection_source = "MANUAL_DEFAULT"
        self._requires_current_switch_fallback = True

    @property
    def known_target_ids(self) -> tuple[str, ...]:
        return tuple(self._class_by_id)

    @property
    def requires_current_switch_fallback(self) -> bool:
        return self._requires_current_switch_fallback

    def update_detections(
        self,
        timestamp_s: float,
        detections: Sequence[DetectedTarget],
        selected_only: bool = False,
    ) -> Mapping[str, MotionEstimate | None]:
        """Update estimator state from the current RGB detections.

        ``selected_only`` is used by Stage 16 WARM-UP: all colours remain
        available to the GUI, while only the manually selected target updates
        an estimator or can affect control. During normal tracking, every
        target resumes its own isolated estimator history. No target's
        measurement or velocity is used for another target.
        """

        self.available_targets = {detection.target_id: detection for detection in detections}
        estimates: dict[str, MotionEstimate | None] = {}
        for target_id, estimator in self.estimators.items():
            if selected_only and target_id != self.selected_target_id:
                estimates[target_id] = None
                continue
            detection = self.available_targets.get(target_id)
            if detection is None or not detection.valid or detection.centroid is None:
                estimator.target_lost()
                estimates[target_id] = None
            else:
                estimates[target_id] = estimator.update(timestamp_s, *detection.centroid)
        return estimates

    def selected_detection(self) -> DetectedTarget | None:
        return self.available_targets.get(self.selected_target_id)

    def selected_is_available(self) -> bool:
        detection = self.selected_detection()
        return bool(detection is not None and detection.valid)

    def select_target(
        self,
        target_id: str,
        timestamp_s: float,
        source: str = "MANUAL_KEY",
    ) -> TargetSwitchEvent | None:
        """Select a known target without resetting the robot or simulation."""

        if target_id not in self._class_by_id:
            raise ValueError(f"Unknown target id: {target_id}")
        if target_id == self.selected_target_id:
            return None
        previous_target_id = self.selected_target_id
        previous_class = self.selected_class
        self.last_selected_target = previous_target_id
        self.selected_target_id = target_id
        self.selected_class = self._class_by_id[target_id]
        self.last_selection_source = source
        # A target can retain its own estimator history, but the first command
        # after a switch must never be a prediction carried from the old
        # target.  Force exactly one current-measurement controller frame.
        self._requires_current_switch_fallback = True
        return TargetSwitchEvent(
            timestamp_s=float(timestamp_s),
            previous_target_id=previous_target_id,
            selected_target_id=target_id,
            previous_class=previous_class,
            selected_class=self.selected_class,
            source=source,
        )

    def controller_source_for_selected(
        self,
        estimate: MotionEstimate | None,
    ) -> str:
        """Return the allowed controller measurement source for this frame."""

        if not self.selected_is_available():
            return "TARGET_LOST"
        if self._requires_current_switch_fallback:
            self._requires_current_switch_fallback = False
            return "CURRENT_SWITCH_FALLBACK"
        if estimate is None or not estimate.estimator_valid:
            return "CURRENT_FALLBACK"
        return "PREDICTED"
