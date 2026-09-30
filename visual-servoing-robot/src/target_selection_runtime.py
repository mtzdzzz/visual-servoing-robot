"""Shared RGB-only selected-target adaptation, input, overlay and scene setup."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import time
from typing import Mapping, Sequence

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import robotics_core as sim
import config
import predictive_measurement as predictive
import visual_servo_runtime as runtime
from camera_observation import EyeInHandRgbDisplay, RedTargetDetection
from multi_target_detector import DetectedTarget, MultiTargetDetector, annotate_multi_target_detections
from scene_factory import create_coloured_sphere, set_static_body_pose
from target_manager import TargetManager, TargetSwitchEvent

KEY_TO_TARGET_ID = {ord("1"): "target_1", ord("2"): "target_2", ord("3"): "target_3"}
CLASS_BY_ID = {"target_1": "RED", "target_2": "GREEN", "target_3": "BLUE"}


@dataclass(frozen=True)
class TargetSceneSpec:
    target_id: str
    class_name: str
    rgba: tuple[float, float, float, float]


DEFAULT_TARGET_SPECS = (
    TargetSceneSpec("target_1", "RED", (1., 0., 0., 1.)),
    TargetSceneSpec("target_2", "GREEN", (0., 1., 0., 1.)),
    TargetSceneSpec("target_3", "BLUE", (0., 0., 1., 1.)),
)
DEFAULT_TARGET_WORLD_POSITIONS = {
    "target_1": (0.350, -0.250, sim.GROUND_TARGET_RADIUS),
    "target_2": (0.450, -0.250, sim.GROUND_TARGET_RADIUS),
    "target_3": (0.350, -0.360, sim.GROUND_TARGET_RADIUS),
}
PREDICTION_ALPHA = predictive.PREDICTION_ALPHA
PREDICTION_HORIZON_S = config.PREDICTION_HORIZON_S


def controller_detection_from_selected(detection: DetectedTarget | None, image_center: tuple[int, int],
                                       rgba_buffer: object) -> RedTargetDetection:
    if detection is None or not detection.valid or detection.centroid is None:
        return RedTargetDetection(False, None, 0.0, None, image_center, None, b"")
    u, v = detection.centroid
    return RedTargetDetection(True, detection.bounding_box, detection.area, (u, v), image_center,
                              (u - image_center[0], v - image_center[1]),
                              rgba_buffer.tobytes() if hasattr(rgba_buffer, "tobytes") else bytes(rgba_buffer))


def manual_keyboard_input(manager: TargetManager, timestamp_s: float,
                          client_id: int) -> tuple[TargetSwitchEvent | None, bool]:
    events = p.getKeyboardEvents(physicsClientId=client_id)
    quit_codes = (ord("q"), ord("Q"), 27, getattr(p, "B3G_ESCAPE", 27))
    if any(events.get(code, 0) & p.KEY_WAS_TRIGGERED for code in quit_codes):
        return None, True
    for key_code, target_id in KEY_TO_TARGET_ID.items():
        if events.get(key_code, 0) & p.KEY_WAS_TRIGGERED:
            return manager.select_target(target_id, timestamp_s, source="MANUAL_KEYBOARD"), False
    return None, False


def annotated_overlay(rgba_buffer: object, width: int, height: int,
                      detections: Sequence[DetectedTarget], manager: TargetManager,
                      mode_title: str, phase: str, source: str, servo_state: str) -> bytes:
    annotated = annotate_multi_target_detections(rgba_buffer, width, height, detections)
    rgba = np.frombuffer(annotated, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    selected = manager.selected_detection()
    selected_name = manager.selected_class
    image_center = (width // 2, height // 2)
    availability = " ".join(
        f"{CLASS_BY_ID[target_id][0]}:{'Y' if manager.available_targets.get(target_id) and manager.available_targets[target_id].valid else 'N'}"
        for target_id in manager.known_target_ids
    )
    valid = bool(selected is not None and selected.valid and selected.centroid is not None)
    if valid:
        assert selected is not None and selected.centroid is not None
        u_selected, v_selected = selected.centroid
        raw_ex, raw_ey = u_selected - image_center[0], v_selected - image_center[1]
        error_text = f"Raw selected error: ex={raw_ex:+d}  ey={raw_ey:+d}"
        pixel_text = f"Selected pixel: ({u_selected}, {v_selected})"
        displayed_servo = servo_state
    else:
        raw_ex = raw_ey = None
        error_text, pixel_text = "Raw selected error: ex=N/A  ey=N/A", "Selected pixel: N/A"
        displayed_servo = "HOLD (TARGET LOST)"
    cv2.rectangle(bgr, (4, 4), (635, 116), (25, 25, 25), thickness=-1)
    lines = (
        f"{mode_title} | Phase: {phase}",
        f"Available: {availability} | Selected: {selected_name} | Selection source: {manager.last_selection_source}",
        f"Image center: {image_center} | Controller: {source}",
        f"{pixel_text} | {error_text}",
        f"Servo: {displayed_servo} | [1] RED [2] GREEN [3] BLUE [Q/ESC] Quit",
    )
    for index, line in enumerate(lines):
        cv2.putText(bgr, line, (10, 25 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .52 if index == 0 else .40, (255, 255, 255), 1, cv2.LINE_AA)
    if valid and selected is not None and selected.bounding_box is not None:
        x, y, box_width, box_height = selected.bounding_box
        cv2.rectangle(bgr, (x - 3, y - 3), (x + box_width + 3, y + box_height + 3), (0, 255, 255), 4)
        text_x, text_y = max(5, x), min(height - 30, y + box_height + 22)
        cv2.putText(bgr, f"SELECTED: {selected_name}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    .52, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"ex: {raw_ex:+d}  ey: {raw_ey:+d}", (text_x, min(height - 8, text_y + 21)),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(bgr, f"SELECTED {selected_name}: TARGET LOST - HOLD | ex: N/A  ey: N/A",
                    (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()


def create_multitarget_ready_context(
    client_id: int, camera_display: EyeInHandRgbDisplay, detector: MultiTargetDetector,
    manager: TargetManager, mode_title: str, target_specs: Sequence[object],
    target_world_positions: Mapping[str, Sequence[float]], realtime: bool = True,
    perform_warmup: bool = True,
) -> tuple[runtime.ReadyContext, dict[str, int]]:
    """Create all targets before the optional selected-RED warm-up."""
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
    target_body_ids = {
        spec.target_id: create_coloured_sphere(sim.GROUND_TARGET_RADIUS, spec.rgba, client_id)
        for spec in target_specs
    }
    for target_id, position in target_world_positions.items():
        set_static_body_pose(target_body_ids[target_id], position, client_id)
    axis_ids: list[int] = []
    sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
    active_hold = sim.capture_active_joint_hold_targets(robot_id, active_indices, client_id)
    context = runtime.ReadyContext(
        robot_id, target_body_ids["target_1"], arm_joint_indices, active_indices, locked_indices,
        locked_positions, sim.build_hold_joint_targets(arm_joint_indices, locked_positions, active_hold),
        active_hold, sim.TwoDimensionalVisualServo(), axis_ids, None, 0.0, None, None, None, 0.0,
    )
    print("Multi-target scene initialized: RED + GREEN + BLUE exist before WARM-UP; selected target: RED.")
    if not perform_warmup:
        return context, target_body_ids
    ready_frames, warmup_initial_error = 0, None
    warmup_steps = max(1, round(sim.STAGE9_WARMUP_MAX_SIMULATION_SECONDS / sim.TIME_STEP))
    for step in range(warmup_steps):
        runtime.step_physics(context, client_id)
        if step % sim.CAMERA_UPDATE_INTERVAL_STEPS != 0:
            if realtime:
                time.sleep(sim.TIME_STEP)
            continue
        sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
        position, orientation = sim.get_camera_optical_center_pose(robot_id, client_id)
        frame = sim.render_live_eye_in_hand_rgb_frame(position, orientation, client_id)
        detections = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
        estimates = manager.update_detections(step * sim.TIME_STEP, detections, selected_only=True)
        raw = controller_detection_from_selected(manager.selected_detection(),
            (frame.image_width // 2, frame.image_height // 2), frame.rgba_buffer)
        estimate = estimates[manager.selected_target_id]
        source = manager.controller_source_for_selected(estimate)
        measurement = predictive.predicted_measurement(raw, estimate)[0] if source == "PREDICTED" else raw
        runtime.control_latest_measurement(context, measurement, client_id)
        if raw.detected:
            ex, ey = raw.pixel_error
            if warmup_initial_error is None:
                warmup_initial_error = sqrt(ex * ex + ey * ey)
            ready_frames = ready_frames + 1 if abs(ex) <= 2 and abs(ey) <= 2 else 0
        else:
            ready_frames = 0
        camera_display.show(frame, annotated_overlay(frame.rgba_buffer, frame.image_width, frame.image_height,
            detections, manager, mode_title, "WARM-UP", source, context.servo.state))
        if ready_frames >= sim.TWO_D_PRECISION_SETTLE_FRAMES_REQUIRED:
            context.warmup_duration_s = (step + 1) * sim.TIME_STEP
            context.warmup_initial_error = warmup_initial_error
            context.ready_ex, context.ready_ey = raw.pixel_error
            context.servo = sim.TwoDimensionalVisualServo()
            return context, target_body_ids
        if realtime:
            time.sleep(sim.TIME_STEP)
    raise RuntimeError("Multi-target WARM-UP did not reach RED READY within the protocol limit.")
