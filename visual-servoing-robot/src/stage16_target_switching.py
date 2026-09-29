"""Stage 16 manual multi-target switching with frozen predictive visual servo.

The selected controller input originates solely from the selected target's RGB
centroid.  Target world positions in this module create fixed coloured spheres
and are read afterwards only to audit that the experiment scene stayed static.
They never flow into target selection, estimation, visual servo, IK, or Panda
motor targets.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import simulation as sim
import stage10_evaluation as stage10
import stage13_evaluation as stage13
from camera_observation import EyeInHandRgbDisplay, RedTargetDetection
from multi_target_detector import (
    DetectedTarget,
    MultiTargetDetector,
    annotate_multi_target_detections,
)
from stage15_multitarget_evaluation import TARGET_SPECS, _create_coloured_sphere, _set_target_pose
from target_manager import TargetManager, TargetSwitchEvent


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / "logs"
LOG_PATH = LOG_DIRECTORY / "stage16_target_switching.csv"
SUMMARY_PATH = LOG_DIRECTORY / "stage16_summary.csv"

# These static positions are an experimental scene definition only.  They are
# intentionally not supplied to target selection, estimators, controller input,
# visual servo, or IK.  Separation is greater than twice the 0.04 m radius.
TARGET_WORLD_POSITIONS: dict[str, tuple[float, float, float]] = {
    "target_1": (0.350, -0.250, sim.GROUND_TARGET_RADIUS),  # RED
    "target_2": (0.450, -0.250, sim.GROUND_TARGET_RADIUS),  # GREEN
    "target_3": (0.350, -0.360, sim.GROUND_TARGET_RADIUS),  # BLUE
}

KEY_TO_TARGET_ID = {ord("1"): "target_1", ord("2"): "target_2", ord("3"): "target_3"}
STAGE16_PREDICTION_ALPHA = stage13.PREDICTION_ALPHA
# This is the best Stage 13 online horizon already evaluated with the frozen
# alpha.  Stage 16 does not tune it.
STAGE16_PREDICTION_HORIZON_S = 0.30
AUTOMATED_EVALUATION_SWITCH_SEQUENCE: tuple[tuple[float, str], ...] = (
    (1.0, "target_2"),  # RED -> GREEN
    (10.0, "target_3"),  # GREEN -> BLUE
    (19.0, "target_1"),  # BLUE -> RED
    (28.0, "target_3"),  # RED -> BLUE
    (37.0, "target_2"),  # BLUE -> GREEN
)
STAGE16_EVALUATION_DURATION_SECONDS = 47.0


@dataclass
class SwitchAttempt:
    index: int
    event: TargetSwitchEvent
    settle_frames: int = 0
    response_time_s: float | None = None
    target_lost_events: int = 0
    completed: bool = False


@dataclass
class Stage16Metrics:
    selected_frames: dict[str, int] = field(default_factory=lambda: {spec.target_id: 0 for spec in TARGET_SPECS})
    selected_detected_frames: dict[str, int] = field(default_factory=lambda: {spec.target_id: 0 for spec in TARGET_SPECS})
    post_convergence_errors: dict[str, list[float]] = field(
        default_factory=lambda: {spec.target_id: [] for spec in TARGET_SPECS}
    )
    selected_target_lost_events: int = 0
    max_locked_deviation: float = 0.0
    max_target_gt_position_deviation_m: float = 0.0


def _class_by_id(target_id: str) -> str:
    return next(spec.class_name for spec in TARGET_SPECS if spec.target_id == target_id)


def _empty_controller_detection(image_center: tuple[int, int]) -> RedTargetDetection:
    return RedTargetDetection(
        detected=False, bounding_box=None, contour_area=0.0, centroid=None,
        image_center=image_center, pixel_error=None, annotated_rgba_buffer=b"",
    )


def _controller_detection_from_selected(
    detection: DetectedTarget | None,
    image_center: tuple[int, int],
    rgba_buffer: object,
) -> RedTargetDetection:
    """Adapt one RGB detection to the existing frozen controller interface."""

    if detection is None or not detection.valid or detection.centroid is None:
        return _empty_controller_detection(image_center)
    u, v = detection.centroid
    return RedTargetDetection(
        detected=True,
        bounding_box=detection.bounding_box,
        contour_area=detection.area,
        centroid=(u, v),
        image_center=image_center,
        pixel_error=(u - image_center[0], v - image_center[1]),
        annotated_rgba_buffer=rgba_buffer.tobytes() if hasattr(rgba_buffer, "tobytes") else bytes(rgba_buffer),
    )


def _manual_keyboard_input(
    manager: TargetManager,
    timestamp_s: float,
    client_id: int,
) -> tuple[TargetSwitchEvent | None, bool]:
    """Read the only allowed manual selection/quit commands.

    ``--stage16-selection`` calls this function once each physics step. No
    elapsed-time, READY, convergence, detection, or target-position condition
    can change the selected target.
    """

    events = p.getKeyboardEvents(physicsClientId=client_id)
    quit_key_codes = (ord("q"), ord("Q"), 27, getattr(p, "B3G_ESCAPE", 27))
    if any(events.get(key_code, 0) & p.KEY_WAS_TRIGGERED for key_code in quit_key_codes):
        return None, True
    for key_code, target_id in KEY_TO_TARGET_ID.items():
        if events.get(key_code, 0) & p.KEY_WAS_TRIGGERED:
            return manager.select_target(target_id, timestamp_s, source="MANUAL_KEYBOARD"), False
    return None, False


def _annotated_overlay(
    rgba_buffer: object,
    width: int,
    height: int,
    detections: Sequence[DetectedTarget],
    manager: TargetManager,
    mode_title: str,
    phase: str,
    source: str,
    servo_state: str,
) -> bytes:
    """Display all RGB detections and the selected target's *raw* error only."""

    annotated = annotate_multi_target_detections(rgba_buffer, width, height, detections)
    rgba = np.frombuffer(annotated, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    selected = manager.selected_detection()
    selected_name = manager.selected_class
    image_center = (width // 2, height // 2)
    availability = " ".join(
        f"{_class_by_id(target_id)[0]}:{'Y' if manager.available_targets.get(target_id, None) and manager.available_targets[target_id].valid else 'N'}"
        for target_id in manager.known_target_ids
    )
    selected_is_valid = bool(selected is not None and selected.valid and selected.centroid is not None)
    if selected_is_valid:
        assert selected is not None and selected.centroid is not None
        u_selected, v_selected = selected.centroid
        raw_ex = u_selected - image_center[0]
        raw_ey = v_selected - image_center[1]
        raw_error_text = f"Raw selected error: ex={raw_ex:+d}  ey={raw_ey:+d}"
        selected_pixel_text = f"Selected pixel: ({u_selected}, {v_selected})"
        display_servo_state = servo_state
    else:
        raw_ex = raw_ey = None
        raw_error_text = "Raw selected error: ex=N/A  ey=N/A"
        selected_pixel_text = "Selected pixel: N/A"
        # Do not leave a prior RGB error onscreen when the selected target is lost.
        display_servo_state = "HOLD (TARGET LOST)"

    cv2.rectangle(bgr, (4, 4), (635, 116), (25, 25, 25), thickness=-1)
    cv2.putText(bgr, f"{mode_title} | Phase: {phase}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"Available: {availability} | Selected: {selected_name} | Selection source: {manager.last_selection_source}", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"Image center: {image_center} | Controller: {source}", (10, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"{selected_pixel_text} | {raw_error_text}", (10, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"Servo: {display_servo_state} | [1] RED [2] GREEN [3] BLUE [Q/ESC] Quit", (10, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    if selected_is_valid and selected is not None and selected.bounding_box is not None:
        x, y, box_width, box_height = selected.bounding_box
        cv2.rectangle(bgr, (x - 3, y - 3), (x + box_width + 3, y + box_height + 3), (0, 255, 255), 4)
        text_x = max(5, x)
        text_y = min(height - 30, y + box_height + 22)
        cv2.putText(bgr, f"SELECTED: {selected_name}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"ex: {raw_ex:+d}  ey: {raw_ey:+d}", (text_x, min(height - 8, text_y + 21)), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(bgr, f"SELECTED {selected_name}: TARGET LOST - HOLD | ex: N/A  ey: N/A", (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()


def _csv_fields() -> tuple[str, ...]:
    return (
        "time_s", "available_red", "available_green", "available_blue",
        "selected_target_id", "selected_class", "switch_event", "previous_target",
        "switch_source", "u_selected", "v_selected", "raw_ex", "raw_ey", "error_norm",
        "u_pred", "v_pred", "controller_source", "target_detected", "servo_state",
        "ik_command_issued", "prediction_clamped", "ground_truth_used_for_selection",
        "ground_truth_used_for_controller", "ground_truth_used_for_evaluation",
    )


def _write_log_row(
    timestamp_s: float,
    manager: TargetManager,
    raw_measurement: RedTargetDetection,
    estimate: object | None,
    source: str,
    switch_event: TargetSwitchEvent | None,
    context: stage10.ReadyContext,
    ik_command_issued: bool,
    prediction_clamped: bool,
) -> dict[str, object]:
    if raw_measurement.detected:
        assert raw_measurement.centroid is not None and raw_measurement.pixel_error is not None
        u, v = raw_measurement.centroid
        ex, ey = raw_measurement.pixel_error
        error_norm: float | str = sqrt(ex * ex + ey * ey)
    else:
        u = v = ex = ey = error_norm = ""
    available = {
        target_id: int(bool(manager.available_targets.get(target_id, None) and manager.available_targets[target_id].valid))
        for target_id in manager.known_target_ids
    }
    return {
        "time_s": f"{timestamp_s:.6f}",
        "available_red": available["target_1"], "available_green": available["target_2"], "available_blue": available["target_3"],
        "selected_target_id": manager.selected_target_id, "selected_class": manager.selected_class,
        "switch_event": int(switch_event is not None),
        "previous_target": switch_event.previous_target_id if switch_event is not None else "",
        "switch_source": switch_event.source if switch_event is not None else "",
        "u_selected": u, "v_selected": v, "raw_ex": ex, "raw_ey": ey, "error_norm": error_norm,
        "u_pred": getattr(estimate, "u_pred", ""), "v_pred": getattr(estimate, "v_pred", ""),
        "controller_source": source, "target_detected": int(raw_measurement.detected),
        "servo_state": context.servo.state, "ik_command_issued": int(ik_command_issued),
        "prediction_clamped": int(prediction_clamped),
        # All three fields document the deliberately one-way GT audit.
        "ground_truth_used_for_selection": False,
        "ground_truth_used_for_controller": False,
        "ground_truth_used_for_evaluation": True,
    }


def _summary_fields() -> tuple[str, ...]:
    return (
        "record_type", "switch_index", "previous_target", "selected_target", "switch_source",
        "switch_time_s", "response_time_s", "switch_result", "target_lost_events",
        "selected_detection_rate_percent", "mean_tracking_error_after_convergence",
        "switch_success_rate_percent", "mean_switch_response_time_s", "rmse_switch_response_time_s",
        "max_switch_response_time_s", "locked_joint_max_deviation_rad", "target_gt_max_position_deviation_m",
        "result",
    )


def _target_position_evaluation_only(
    target_body_ids: dict[str, int],
    client_id: int,
) -> float:
    """Read world poses only after control for a static-scene evaluation audit."""

    errors = []
    for target_id, body_id in target_body_ids.items():
        world_position, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
        errors.append(float(np.linalg.norm(np.asarray(world_position) - np.asarray(TARGET_WORLD_POSITIONS[target_id]))))
    return max(errors, default=0.0)


def _record_switch(
    attempts: list[SwitchAttempt],
    event: TargetSwitchEvent | None,
    active_attempt: SwitchAttempt | None,
) -> SwitchAttempt | None:
    if event is None:
        return active_attempt
    if active_attempt is not None and not active_attempt.completed:
        active_attempt.completed = True  # completed with no response -> unsuccessful
    attempt = SwitchAttempt(index=len(attempts) + 1, event=event)
    attempts.append(attempt)
    print(
        f"Stage 16 switch {attempt.index}: {event.previous_class} -> {event.selected_class} "
        f"at t={event.timestamp_s:.2f}s ({event.source})."
    )
    return attempt


def _advance_switch_convergence(
    attempt: SwitchAttempt | None,
    measurement: RedTargetDetection,
    timestamp_s: float,
) -> bool:
    if attempt is None or attempt.completed:
        return False
    if not measurement.detected:
        attempt.settle_frames = 0
        return False
    assert measurement.pixel_error is not None
    ex, ey = measurement.pixel_error
    attempt.settle_frames = (
        attempt.settle_frames + 1
        if abs(ex) <= sim.TWO_D_TOL_X_PIXELS and abs(ey) <= sim.TWO_D_TOL_Y_PIXELS
        else 0
    )
    if attempt.settle_frames < sim.TWO_D_PRECISION_SETTLE_FRAMES_REQUIRED:
        return False
    attempt.response_time_s = timestamp_s - attempt.event.timestamp_s
    attempt.completed = True
    print(
        f"Stage 16 switch {attempt.index}: {attempt.event.selected_class} centered in "
        f"{attempt.response_time_s:.3f}s."
    )
    return True


def _build_summary(
    attempts: Sequence[SwitchAttempt],
    metrics: Stage16Metrics,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    response_times = [attempt.response_time_s for attempt in attempts if attempt.response_time_s is not None]
    for attempt in attempts:
        target_id = attempt.event.selected_target_id
        target_frames = metrics.selected_frames[target_id]
        rows.append(
            {
                "record_type": "SWITCH", "switch_index": attempt.index,
                "previous_target": attempt.event.previous_target_id, "selected_target": target_id,
                "switch_source": attempt.event.source, "switch_time_s": attempt.event.timestamp_s,
                "response_time_s": attempt.response_time_s if attempt.response_time_s is not None else "",
                "switch_result": "PASS" if attempt.response_time_s is not None else "FAIL",
                "target_lost_events": attempt.target_lost_events,
                "selected_detection_rate_percent": 100.0 * metrics.selected_detected_frames[target_id] / target_frames if target_frames else 0.0,
                "mean_tracking_error_after_convergence": "",
                "switch_success_rate_percent": "", "mean_switch_response_time_s": "",
                "rmse_switch_response_time_s": "", "max_switch_response_time_s": "",
                "locked_joint_max_deviation_rad": metrics.max_locked_deviation,
                "target_gt_max_position_deviation_m": metrics.max_target_gt_position_deviation_m,
                "result": "PASS" if attempt.response_time_s is not None else "FAIL",
            }
        )
    for target_id in metrics.selected_frames:
        errors = metrics.post_convergence_errors[target_id]
        rows.append(
            {
                "record_type": "TARGET", "switch_index": "", "previous_target": "", "selected_target": target_id,
                "switch_source": "", "switch_time_s": "", "response_time_s": "", "switch_result": "",
                "target_lost_events": "",
                "selected_detection_rate_percent": 100.0 * metrics.selected_detected_frames[target_id] / metrics.selected_frames[target_id] if metrics.selected_frames[target_id] else 0.0,
                "mean_tracking_error_after_convergence": sum(errors) / len(errors) if errors else "",
                "switch_success_rate_percent": "", "mean_switch_response_time_s": "",
                "rmse_switch_response_time_s": "", "max_switch_response_time_s": "",
                "locked_joint_max_deviation_rad": metrics.max_locked_deviation,
                "target_gt_max_position_deviation_m": metrics.max_target_gt_position_deviation_m,
                "result": "OBSERVED",
            }
        )
    success_rate = 100.0 * len(response_times) / len(attempts) if attempts else 0.0
    mean_response = sum(response_times) / len(response_times) if response_times else None
    rmse_response = sqrt(sum(value * value for value in response_times) / len(response_times)) if response_times else None
    rows.append(
        {
            "record_type": "OVERALL", "switch_index": "", "previous_target": "", "selected_target": "",
            "switch_source": "", "switch_time_s": "", "response_time_s": "", "switch_result": "",
            "target_lost_events": metrics.selected_target_lost_events,
            "selected_detection_rate_percent": "", "mean_tracking_error_after_convergence": "",
            "switch_success_rate_percent": success_rate, "mean_switch_response_time_s": mean_response if mean_response is not None else "",
            "rmse_switch_response_time_s": rmse_response if rmse_response is not None else "",
            "max_switch_response_time_s": max(response_times) if response_times else "",
            "locked_joint_max_deviation_rad": metrics.max_locked_deviation,
            "target_gt_max_position_deviation_m": metrics.max_target_gt_position_deviation_m,
            "result": "PASS" if len(attempts) == 5 and len(response_times) == 5 and metrics.max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT else "FAIL",
        }
    )
    return rows


def _create_multitarget_ready_context(
    client_id: int,
    camera_display: EyeInHandRgbDisplay,
    detector: MultiTargetDetector,
    manager: TargetManager,
    mode_title: str,
    realtime: bool = True,
    perform_warmup: bool = True,
) -> tuple[stage10.ReadyContext, dict[str, int]]:
    """Create all Stage 16 targets before a RED-only WARM-UP.

    This deliberately does not call Stage 10's single-red ready helper. The
    frozen controller primitives are reused, but RED, GREEN, and BLUE exist
    before the first physics and RGB frame. During WARM-UP only the selected
    RED estimator/controller path is updated; GREEN and BLUE remain visible
    detector overlays and cannot affect an IK command.
    """

    p.resetSimulation(physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, 0, physicsClientId=client_id)
    p.setTimeStep(sim.TIME_STEP, physicsClientId=client_id)
    sim.enable_pybullet_camera_debug_previews(client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=client_id)
    initial_positions = sim.print_panda_arm_joint_configuration(robot_id, client_id)
    arm_joint_indices = sim.get_panda_arm_joint_indices(robot_id, client_id)
    arm_joint_names = sim.get_panda_arm_joint_names(robot_id, client_id)
    locked_indices = [arm_joint_names[name] for name in sim.BASELINE_LOCKED_JOINT_NAMES]
    locked_positions = {index: initial_positions[index] for index in locked_indices}
    active_indices = [index for index in arm_joint_indices if index not in locked_indices]

    # Three static target bodies are created before entering the WARM-UP loop.
    target_body_ids = {
        spec.target_id: _create_coloured_sphere(spec, client_id)
        for spec in TARGET_SPECS
    }
    for target_id, position in TARGET_WORLD_POSITIONS.items():
        _set_target_pose(target_body_ids[target_id], position, client_id)

    axis_ids: list[int] = []
    sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
    active_hold = sim.capture_active_joint_hold_targets(robot_id, active_indices, client_id)
    context = stage10.ReadyContext(
        robot_id=robot_id,
        target_body_id=target_body_ids["target_1"],
        arm_joint_indices=arm_joint_indices,
        active_joint_indices=active_indices,
        locked_joint_indices=locked_indices,
        locked_initial_positions=locked_positions,
        commanded_joint_targets=sim.build_hold_joint_targets(arm_joint_indices, locked_positions, active_hold),
        active_hold_targets=active_hold,
        servo=sim.TwoDimensionalVisualServo(),
        camera_axis_debug_item_ids=axis_ids,
        debug_text_id=None,
        warmup_duration_s=0.0,
        warmup_initial_error=None,
        ready_ex=None,
        ready_ey=None,
        max_locked_deviation=0.0,
    )
    ready_frames = 0
    warmup_initial_error: float | None = None
    first_rgb_reported = False
    warmup_steps = max(1, round(sim.STAGE9_WARMUP_MAX_SIMULATION_SECONDS / sim.TIME_STEP))
    print("Stage 16 scene initialized: RED + GREEN + BLUE exist before WARM-UP; selected target: RED.")

    if not perform_warmup:
        # Stage 21's interactive showcase starts immediately. The same frozen
        # controller aligns RED in its ordinary tracking loop, while 1/2/3 and
        # quit input are already responsive from the first displayed frame.
        print("Stage 21 integration: immediate manual tracking; no experiment READY gate.")
        return context, target_body_ids

    for step in range(warmup_steps):
        stage10._step_physics(context, client_id)
        if step % sim.CAMERA_UPDATE_INTERVAL_STEPS != 0:
            if realtime:
                time.sleep(sim.TIME_STEP)
            continue

        sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
        camera_position, camera_orientation = sim.get_camera_optical_center_pose(robot_id, client_id)
        frame = sim.render_live_eye_in_hand_rgb_frame(camera_position, camera_orientation, client_id)
        detections = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
        estimates = manager.update_detections(step * sim.TIME_STEP, detections, selected_only=True)
        raw_measurement = _controller_detection_from_selected(
            manager.selected_detection(), (frame.image_width // 2, frame.image_height // 2), frame.rgba_buffer
        )
        estimate = estimates[manager.selected_target_id]
        source = manager.controller_source_for_selected(estimate)
        if source == "PREDICTED":
            controller_measurement, source, _ = stage13._predicted_measurement(raw_measurement, estimate)
        else:
            controller_measurement = raw_measurement
        stage10._control_latest_measurement(context, controller_measurement, client_id)

        if raw_measurement.detected:
            assert raw_measurement.pixel_error is not None
            ex, ey = raw_measurement.pixel_error
            if warmup_initial_error is None:
                warmup_initial_error = sqrt(ex * ex + ey * ey)
            ready_frames = ready_frames + 1 if abs(ex) <= 2 and abs(ey) <= 2 else 0
        else:
            ready_frames = 0

        if not first_rgb_reported:
            visible_classes = {detection.class_name for detection in detections if detection.valid}
            print(
                "Stage 16 first RGB frame: "
                f"RED={'YES' if 'RED' in visible_classes else 'NO'}, "
                f"GREEN={'YES' if 'GREEN' in visible_classes else 'NO'}, "
                f"BLUE={'YES' if 'BLUE' in visible_classes else 'NO'}; "
                "controller=RED only."
            )
            first_rgb_reported = True

        camera_display.show(
            frame,
            _annotated_overlay(
                frame.rgba_buffer, frame.image_width, frame.image_height, detections,
                manager, mode_title, "WARM-UP", source, context.servo.state,
            ),
        )
        context.debug_text_id = sim.update_motion_debug_text(
            f"{mode_title}\n"
            "Phase: WARM-UP | Selected: RED | RED controls; GREEN/BLUE observe only.\n"
            "Targets present: RED + GREEN + BLUE",
            context.debug_text_id,
            client_id,
        )
        if ready_frames >= sim.TWO_D_PRECISION_SETTLE_FRAMES_REQUIRED:
            context.warmup_duration_s = (step + 1) * sim.TIME_STEP
            context.warmup_initial_error = warmup_initial_error
            context.ready_ex, context.ready_ey = raw_measurement.pixel_error
            context.servo = sim.TwoDimensionalVisualServo()
            print(
                f"Stage 16 WARM-UP -> READY after {context.warmup_duration_s:.2f}s; "
                f"RED ex/ey=({context.ready_ex}, {context.ready_ey})."
            )
            return context, target_body_ids
        if realtime:
            time.sleep(sim.TIME_STEP)

    raise RuntimeError("Stage 16 WARM-UP did not reach RED READY within the protocol limit.")


def run_stage16_target_selection() -> None:
    """Run the indefinite Stage 16 manual-selection demonstration.

    The selected target is RED at startup and can change only through a
    ``MANUAL_KEYBOARD`` event from ``1``, ``2``, or ``3``. This mode contains
    no timer-based switch schedule, trial counter, output CSV, or fixed run
    duration. It ends only when the user presses Q/Escape, closes the Tk RGB
    viewer, or closes the PyBullet GUI.
    """

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 16 manual selection.")
    display: EyeInHandRgbDisplay | None = None
    try:
        mode_title = "STAGE 16 - MANUAL TARGET SELECTION"
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB - Stage 16 Manual Target Selection")
        manager = TargetManager(
            STAGE16_PREDICTION_ALPHA, STAGE16_PREDICTION_HORIZON_S, initial_target_id="target_1"
        )
        detector = MultiTargetDetector()
        context, _ = _create_multitarget_ready_context(
            client_id, display, detector, manager, mode_title
        )
        print("Stage 16 Manual Target Selection: ACTIVE")
        print("Selected target: RED (MANUAL_DEFAULT).")
        print("Keys: 1=RED, 2=GREEN, 3=BLUE, Q/Escape=Quit.")
        print("No scheduled switch sequence, fixed duration, trial count, or automatic target selection is active.")

        step = 0
        while p.isConnected(client_id) and display.is_open:
            timestamp_s = step * sim.TIME_STEP
            switch_event, quit_requested = _manual_keyboard_input(manager, timestamp_s, client_id)
            if quit_requested:
                print("Stage 16 manual mode: user requested exit.")
                break
            if switch_event is not None:
                print(
                    f"Manual selection: {switch_event.previous_class} -> "
                    f"{switch_event.selected_class} ({switch_event.source})."
                )

            # Frozen Stage 13/10 motor path. The current hold target remains
            # in force when the selected target is centred or currently lost.
            stage10._step_physics(context, client_id)
            if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                sim.update_camera_reference_axes(
                    context.robot_id, client_id, context.camera_axis_debug_item_ids
                )
                camera_position, camera_orientation = sim.get_camera_optical_center_pose(
                    context.robot_id, client_id
                )
                frame = sim.render_live_eye_in_hand_rgb_frame(
                    camera_position, camera_orientation, client_id
                )
                detections = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
                estimates = manager.update_detections(timestamp_s, detections)
                raw_measurement = _controller_detection_from_selected(
                    manager.selected_detection(),
                    (frame.image_width // 2, frame.image_height // 2),
                    frame.rgba_buffer,
                )
                estimate = estimates[manager.selected_target_id]
                source = manager.controller_source_for_selected(estimate)
                prediction_clamped = False
                if source == "PREDICTED":
                    controller_measurement, source, prediction_clamped = stage13._predicted_measurement(
                        raw_measurement, estimate
                    )
                else:
                    controller_measurement = raw_measurement

                # The controller receives the selected target only. If that
                # target is absent, the frozen controller requests HOLD; it
                # cannot select another visible colour.
                stage10._control_latest_measurement(context, controller_measurement, client_id)
                display.show(
                    frame,
                    _annotated_overlay(
                        frame.rgba_buffer, frame.image_width, frame.image_height, detections,
                        manager, mode_title, "READY / TRACKING", source, context.servo.state,
                    ),
                )
                context.debug_text_id = sim.update_motion_debug_text(
                    f"{mode_title}\n"
                    f"Selected: {manager.selected_class} | Selection: {manager.last_selection_source}\n"
                    f"Controller: {source} | Servo: {context.servo.state}\n"
                    "[1] RED | [2] GREEN | [3] BLUE | [Q/ESC] Quit",
                    context.debug_text_id,
                    client_id,
                )
                # Keep the local name explicit for review: the display uses
                # raw error, while this flag is evaluation-free telemetry only.
                _ = prediction_clamped
            step += 1
            time.sleep(sim.TIME_STEP)
    finally:
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


def run_stage16_evaluation() -> None:
    """Run the finite, scheduled Stage 16 switching evaluation only."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 16.")
    display: EyeInHandRgbDisplay | None = None
    try:
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB - Stage 16 Switching Evaluation")
        manager = TargetManager(
            STAGE16_PREDICTION_ALPHA, STAGE16_PREDICTION_HORIZON_S, initial_target_id="target_1"
        )
        detector = MultiTargetDetector()
        context, target_body_ids = _create_multitarget_ready_context(
            client_id, display, detector, manager, "STAGE 16 - SWITCHING EVALUATION"
        )
        attempts: list[SwitchAttempt] = []
        active_attempt: SwitchAttempt | None = None
        metrics = Stage16Metrics(max_locked_deviation=context.max_locked_deviation)
        scheduled_index = 0
        selected_was_detected = False
        pending_switch_event: TargetSwitchEvent | None = None
        print("Stage 16 Evaluation: scheduled multi-target switching")
        print(
            f"Frozen predictive measurement: alpha={STAGE16_PREDICTION_ALPHA:.2f}, "
            f"tau={STAGE16_PREDICTION_HORIZON_S:.2f}s."
        )
        print("Evaluation selection source: AUTOMATED_EVALUATION; robot pose is never reset after READY.")

        total_steps = round(STAGE16_EVALUATION_DURATION_SECONDS / sim.TIME_STEP)
        with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for step in range(total_steps):
                timestamp_s = step * sim.TIME_STEP
                event = None
                if scheduled_index < len(AUTOMATED_EVALUATION_SWITCH_SEQUENCE):
                    switch_time, target_id = AUTOMATED_EVALUATION_SWITCH_SEQUENCE[scheduled_index]
                    if timestamp_s >= switch_time:
                        event = manager.select_target(target_id, timestamp_s, source="AUTOMATED_EVALUATION")
                        scheduled_index += 1
                if event is not None:
                    active_attempt = _record_switch(attempts, event, active_attempt)
                    pending_switch_event = event

                # Frozen Stage 13/10 motor path: keep locks and currently
                # commanded active targets at every physics step.
                stage10._step_physics(context, client_id)
                metrics.max_locked_deviation = max(metrics.max_locked_deviation, context.max_locked_deviation)

                if step % sim.CAMERA_UPDATE_INTERVAL_STEPS != 0:
                    time.sleep(sim.TIME_STEP)
                    continue

                sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
                camera_position, camera_orientation = sim.get_camera_optical_center_pose(context.robot_id, client_id)
                frame = sim.render_live_eye_in_hand_rgb_frame(camera_position, camera_orientation, client_id)
                detections = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
                estimates = manager.update_detections(timestamp_s, detections)
                raw_measurement = _controller_detection_from_selected(
                    manager.selected_detection(), (frame.image_width // 2, frame.image_height // 2), frame.rgba_buffer
                )
                estimate = estimates[manager.selected_target_id]
                source = manager.controller_source_for_selected(estimate)
                prediction_clamped = False
                if source == "PREDICTED":
                    controller_measurement, source, prediction_clamped = stage13._predicted_measurement(
                        raw_measurement, estimate
                    )
                else:
                    controller_measurement = raw_measurement

                # This is the sole controller call.  ``controller_measurement``
                # originates from the selected target only; other detections
                # remain display/estimator observations.
                ik_command_issued, _ = stage10._control_latest_measurement(
                    context, controller_measurement, client_id
                )

                selected_id = manager.selected_target_id
                metrics.selected_frames[selected_id] += 1
                metrics.selected_detected_frames[selected_id] += int(raw_measurement.detected)
                if selected_was_detected and not raw_measurement.detected:
                    metrics.selected_target_lost_events += 1
                    if active_attempt is not None and not active_attempt.completed:
                        active_attempt.target_lost_events += 1
                selected_was_detected = raw_measurement.detected
                just_converged = _advance_switch_convergence(active_attempt, raw_measurement, timestamp_s)
                if active_attempt is not None and active_attempt.completed and raw_measurement.detected:
                    assert raw_measurement.pixel_error is not None
                    ex, ey = raw_measurement.pixel_error
                    metrics.post_convergence_errors[selected_id].append(sqrt(ex * ex + ey * ey))

                # Explicit post-control, evaluation-only audit that all scene
                # objects stayed at their scheduled static world positions.
                metrics.max_target_gt_position_deviation_m = max(
                    metrics.max_target_gt_position_deviation_m,
                    _target_position_evaluation_only(target_body_ids, client_id),
                )
                writer.writerow(
                    _write_log_row(
                        timestamp_s, manager, raw_measurement, estimate, source, pending_switch_event,
                        context, ik_command_issued, prediction_clamped,
                    )
                )
                pending_switch_event = None
                display.show(
                    frame,
                    _annotated_overlay(
                        frame.rgba_buffer, frame.image_width, frame.image_height, detections,
                        manager, "STAGE 16 - SWITCHING EVALUATION", "READY / TRACKING", source, context.servo.state,
                    ),
                )
                context.debug_text_id = sim.update_motion_debug_text(
                    "STAGE 16 - SWITCHING EVALUATION\n"
                    f"Phase: READY / TRACKING | Selected: {manager.selected_class} | Source: {source} | Servo: {context.servo.state}\n"
                    "Scheduled evaluation switching is active.",
                    context.debug_text_id,
                    client_id,
                )
                time.sleep(sim.TIME_STEP)

        if active_attempt is not None and not active_attempt.completed:
            active_attempt.completed = True
        summaries = _build_summary(attempts, metrics)
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=_summary_fields())
            writer.writeheader()
            writer.writerows(summaries)
        from plot_stage16 import generate_stage16_figures

        figure_paths = generate_stage16_figures()
        overall = next(row for row in summaries if row["record_type"] == "OVERALL")
        print("\n===== STAGE 16 TARGET SWITCHING SUMMARY =====")
        for attempt in attempts:
            result = "PASS" if attempt.response_time_s is not None else "FAIL"
            response = f"{attempt.response_time_s:.3f}s" if attempt.response_time_s is not None else "not converged"
            print(
                f"Switch {attempt.index}: {attempt.event.previous_class}->{attempt.event.selected_class}: "
                f"{response}, lost={attempt.target_lost_events}, {result}"
            )
        print(f"Switch success rate: {overall['switch_success_rate_percent']:.1f}%")
        print(f"Selected-target lost events: {metrics.selected_target_lost_events}")
        print(f"Locked joint max deviation: {metrics.max_locked_deviation:.6f} rad")
        print("Controller source audit: only selected RGB detection -> current fallback/prediction -> frozen Stage 13 control.")
        print("GT audit: selection=False, controller=False, evaluation=True")
        print("CSV:", LOG_PATH)
        print("Summary:", SUMMARY_PATH)
        print("Figures:", *figure_paths)
        print("Stage 16:", overall["result"])
    finally:
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)
