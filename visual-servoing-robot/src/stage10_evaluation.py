"""Stage 10 robustness experiments built on the frozen Stage 9 controller.

This module owns only experiment scheduling, CSV logging, and deterministic
measurement-noise injection.  It never changes visual-servo gains, joint
groups, camera mounting, IK parameters, target trajectory parameters, or
detector thresholds.
"""

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
import io
from math import pi, sqrt
from pathlib import Path
import random
import time
from typing import Callable, Sequence

import pybullet as p
import pybullet_data

import simulation as sim
from camera_observation import EyeInHandRgbDisplay, RedTargetDetection, create_manual_draggable_red_ground_target


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / "logs"
TARGET_LOST_DURATION_SECONDS = 14.0
TARGET_LOST_HIDE_START_SECONDS = 4.0
TARGET_LOST_HIDE_END_SECONDS = 6.5
TARGET_LOST_HIDDEN_WORLD = (-0.80, 0.80, sim.GROUND_TARGET_RADIUS)
NOISE_DURATION_SECONDS = sim.STAGE9_DURATION_SECONDS
NOISE_LEVELS_PIXELS = (0.0, 1.0, 3.0)
NOISE_RANDOM_SEED = 20260911


@dataclass
class ReadyContext:
    robot_id: int
    target_body_id: int
    arm_joint_indices: list[int]
    active_joint_indices: list[int]
    locked_joint_indices: list[int]
    locked_initial_positions: dict[int, float]
    commanded_joint_targets: list[float]
    active_hold_targets: dict[int, float]
    servo: sim.TwoDimensionalVisualServo
    camera_axis_debug_item_ids: list[int]
    debug_text_id: int | None
    warmup_duration_s: float
    warmup_initial_error: float | None
    ready_ex: int | None
    ready_ey: int | None
    max_locked_deviation: float


def _percentile(values: Sequence[float], percentile: float = 0.95) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def _metric(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_error": None, "rmse": None, "p95_error": None, "max_error": None}
    return {
        "mean_error": sum(values) / len(values),
        "rmse": sqrt(sum(value * value for value in values) / len(values)),
        "p95_error": _percentile(values),
        "max_error": max(values),
    }


def _build_noisy_measurement(
    detection: RedTargetDetection,
    random_generator: random.Random,
    standard_deviation_pixels: float,
) -> tuple[RedTargetDetection, float, float]:
    """Return an RGB-derived measurement with deterministic pixel noise.

    The raw detector has already consumed the Eye-in-Hand RGB frame.  This
    function has no target-body or target-world input and is the sole source
    of Stage 10B disturbance.
    """
    if not detection.detected:
        return detection, 0.0, 0.0
    assert detection.centroid is not None
    noise_u = random_generator.gauss(0.0, standard_deviation_pixels)
    noise_v = random_generator.gauss(0.0, standard_deviation_pixels)
    u, v = detection.centroid
    cx, cy = detection.image_center
    measured_centroid = (u + noise_u, v + noise_v)
    measured_error = (measured_centroid[0] - cx, measured_centroid[1] - cy)
    return (
        RedTargetDetection(
            detected=True,
            bounding_box=detection.bounding_box,
            contour_area=detection.contour_area,
            centroid=measured_centroid,  # Controller measurement, not ground truth.
            image_center=detection.image_center,
            pixel_error=measured_error,
            annotated_rgba_buffer=detection.annotated_rgba_buffer,
        ),
        noise_u,
        noise_v,
    )


def _create_ready_context(client_id: int, camera_display: EyeInHandRgbDisplay, label: str) -> ReadyContext:
    """Reset a repeatable robot state and verify visual READY before a test."""
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
    target_body_id = create_manual_draggable_red_ground_target(client_id)
    target_orientation = (0.0, 0.0, 0.0, 1.0)
    p.resetBasePositionAndOrientation(
        target_body_id, sim.STAGE9_TARGET_CENTER_WORLD, target_orientation, physicsClientId=client_id
    )
    axis_ids: list[int] = []
    sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
    active_hold = sim.capture_active_joint_hold_targets(robot_id, active_indices, client_id)
    commanded = sim.build_hold_joint_targets(arm_joint_indices, locked_positions, active_hold)
    servo = sim.TwoDimensionalVisualServo()
    ready_frames = 0
    warmup_initial_error: float | None = None
    max_locked_deviation = 0.0
    debug_text_id = sim.update_motion_debug_text(
        f"STAGE 10\n{label}\nPhase: WARM-UP", None, client_id
    )
    warmup_steps = max(1, round(sim.STAGE9_WARMUP_MAX_SIMULATION_SECONDS / sim.TIME_STEP))
    for step in range(warmup_steps):
        p.resetBasePositionAndOrientation(
            target_body_id, sim.STAGE9_TARGET_CENTER_WORLD, target_orientation, physicsClientId=client_id
        )
        sim.apply_constrained_arm_position_control(
            robot_id, arm_joint_indices, commanded, locked_positions, client_id
        )
        p.stepSimulation(physicsClientId=client_id)
        deviations = sim.get_locked_joint_deviations(robot_id, locked_positions, client_id)
        max_locked_deviation = max(max_locked_deviation, max(map(abs, deviations.values()), default=0.0))
        if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
            sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
        raw_detection = sim.update_live_eye_in_hand_camera(
            robot_id, step, client_id, camera_display, f"STAGE 10 {label} | Phase: WARM-UP"
        )
        if raw_detection is not None:
            if raw_detection.detected:
                assert raw_detection.pixel_error is not None
                ex, ey = raw_detection.pixel_error
                if warmup_initial_error is None:
                    warmup_initial_error = sqrt(ex * ex + ey * ey)
                ready_frames = ready_frames + 1 if abs(ex) <= 2 and abs(ey) <= 2 else 0
            else:
                ready_frames = 0
            with contextlib.redirect_stdout(io.StringIO()):
                command = sim._stage9_start_dynamic_tracking_from_current_pose(
                    servo, raw_detection, robot_id, locked_indices, locked_positions, client_id
                )
            if command is not None and not servo.hold_requested:
                commanded = list(command.joint_targets)
            if servo.hold_requested:
                active_hold = sim.capture_active_joint_hold_targets(robot_id, active_indices, client_id)
                commanded = sim.build_hold_joint_targets(arm_joint_indices, locked_positions, active_hold)
                servo.hold_requested = False
            if ready_frames >= sim.TWO_D_PRECISION_SETTLE_FRAMES_REQUIRED:
                assert raw_detection.pixel_error is not None
                return ReadyContext(
                    robot_id, target_body_id, arm_joint_indices, active_indices, locked_indices,
                    locked_positions, commanded, active_hold, sim.TwoDimensionalVisualServo(), axis_ids,
                    debug_text_id, (step + 1) * sim.TIME_STEP, warmup_initial_error,
                    raw_detection.pixel_error[0], raw_detection.pixel_error[1], max_locked_deviation,
                )
        time.sleep(sim.TIME_STEP)
    raise RuntimeError(f"Stage 10 {label}: WARM-UP did not reach READY within the protocol limit.")


def _control_latest_measurement(
    context: ReadyContext, measurement: RedTargetDetection, client_id: int
) -> tuple[bool, bool]:
    """Apply the frozen latest-observation controller and return command flags."""
    if context.servo.state.startswith("FAIL"):
        return False, False
    if context.servo.state == "TARGET LOST / HOLD" and measurement.detected:
        context.servo = sim.TwoDimensionalVisualServo()
    with contextlib.redirect_stdout(io.StringIO()):
        command = sim._stage9_start_dynamic_tracking_from_current_pose(
            context.servo, measurement, context.robot_id, context.locked_joint_indices,
            context.locked_initial_positions, client_id
        )
    ik_command_issued = command is not None and not context.servo.hold_requested
    nonzero_correction = bool(
        ik_command_issued
        and command is not None
        and (abs(command.camera_delta_c[0]) > 1e-12 or abs(command.camera_delta_c[1]) > 1e-12)
    )
    if ik_command_issued and command is not None:
        context.commanded_joint_targets = list(command.joint_targets)
    if context.servo.hold_requested:
        context.active_hold_targets = sim.capture_active_joint_hold_targets(
            context.robot_id, context.active_joint_indices, client_id
        )
        context.commanded_joint_targets = sim.build_hold_joint_targets(
            context.arm_joint_indices, context.locked_initial_positions, context.active_hold_targets
        )
        context.servo.hold_requested = False
    return ik_command_issued, nonzero_correction


def _stage10_csv_fields() -> tuple[str, ...]:
    return (
        "time_s", "experiment", "condition", "phase", "target_detected",
        "target_u", "target_v", "measurement_u", "measurement_v", "noise_u_px", "noise_v_px",
        "ex", "ey", "measurement_ex", "measurement_ey", "error_norm", "servo_state",
        "ik_command_issued", "nonzero_correction_issued", "ee_x", "ee_y", "ee_z",
        "camera_x", "camera_y", "camera_z", "target_world_x_log_only",
        "target_world_y_log_only", "target_world_z_log_only",
    )


def _stage10_log_row(
    simulation_time_s: float,
    experiment: str,
    condition: str,
    phase: str,
    raw_detection: RedTargetDetection,
    measurement: RedTargetDetection,
    noise_u: float,
    noise_v: float,
    context: ReadyContext,
    target_world_position: Sequence[float],
    ik_command_issued: bool,
    nonzero_correction_issued: bool,
    client_id: int,
) -> dict[str, object]:
    ee_position, _ = sim.get_camera_reference_pose(context.robot_id, client_id)
    camera_position, _ = sim.get_camera_optical_center_pose(context.robot_id, client_id)
    if raw_detection.detected:
        assert raw_detection.centroid is not None and raw_detection.pixel_error is not None
        target_u, target_v = raw_detection.centroid
        ex, ey = raw_detection.pixel_error
        error_norm: float | str = sqrt(ex * ex + ey * ey)
    else:
        target_u = target_v = ex = ey = error_norm = ""
    if measurement.detected:
        assert measurement.centroid is not None and measurement.pixel_error is not None
        measurement_u, measurement_v = measurement.centroid
        measurement_ex, measurement_ey = measurement.pixel_error
    else:
        measurement_u = measurement_v = measurement_ex = measurement_ey = ""
    return {
        "time_s": f"{simulation_time_s:.6f}",
        "experiment": experiment,
        "condition": condition,
        "phase": phase,
        "target_detected": int(raw_detection.detected),
        "target_u": target_u,
        "target_v": target_v,
        "measurement_u": measurement_u,
        "measurement_v": measurement_v,
        "noise_u_px": f"{noise_u:.6f}",
        "noise_v_px": f"{noise_v:.6f}",
        "ex": ex,
        "ey": ey,
        "measurement_ex": measurement_ex,
        "measurement_ey": measurement_ey,
        "error_norm": error_norm,
        "servo_state": context.servo.state,
        "ik_command_issued": int(ik_command_issued),
        "nonzero_correction_issued": int(nonzero_correction_issued),
        "ee_x": f"{ee_position[0]:.9f}",
        "ee_y": f"{ee_position[1]:.9f}",
        "ee_z": f"{ee_position[2]:.9f}",
        "camera_x": f"{camera_position[0]:.9f}",
        "camera_y": f"{camera_position[1]:.9f}",
        "camera_z": f"{camera_position[2]:.9f}",
        # World data is scheduling/log-only and never reaches the controller.
        "target_world_x_log_only": f"{target_world_position[0]:.9f}",
        "target_world_y_log_only": f"{target_world_position[1]:.9f}",
        "target_world_z_log_only": f"{target_world_position[2]:.9f}",
    }


def _apply_target_pose(target_body_id: int, position: Sequence[float], client_id: int) -> None:
    p.resetBasePositionAndOrientation(
        target_body_id, position, (0.0, 0.0, 0.0, 1.0), physicsClientId=client_id
    )


def _step_physics(context: ReadyContext, client_id: int) -> None:
    sim.apply_constrained_arm_position_control(
        context.robot_id, context.arm_joint_indices, context.commanded_joint_targets,
        context.locked_initial_positions, client_id,
    )
    p.stepSimulation(physicsClientId=client_id)
    deviations = sim.get_locked_joint_deviations(
        context.robot_id, context.locked_initial_positions, client_id
    )
    context.max_locked_deviation = max(
        context.max_locked_deviation, max(map(abs, deviations.values()), default=0.0)
    )


def _episode_metrics(errors: Sequence[float], detected_frames: int, total_frames: int) -> dict[str, float | int | None]:
    values = _metric(errors)
    values["detection_rate"] = 100.0 * detected_frames / total_frames if total_frames else 0.0
    values["frames"] = total_frames
    return values


def _stage10_overlay(
    experiment: str,
    condition: str,
    phase: str,
    simulation_time_s: float,
    duration_s: float,
    servo: sim.TwoDimensionalVisualServo,
    detection: RedTargetDetection | None,
) -> str:
    """Create display-only status text from the current RGB observation."""
    if detection is not None and detection.detected:
        assert detection.centroid is not None and detection.pixel_error is not None
        ex, ey = detection.pixel_error
        target_text = f"Target: {detection.centroid} | Detection: YES"
        error_text = f"ex={ex:+.1f} ey={ey:+.1f} Error={sqrt(ex * ex + ey * ey):.1f}px"
    else:
        target_text = "Target: N/A | Detection: NO"
        error_text = "ex=N/A ey=N/A Error=N/A"
    return (
        f"STAGE 10 - {experiment}\nCondition: {condition} | Phase: {phase}\n"
        f"Time: {simulation_time_s:.1f}/{duration_s:.1f}s | {target_text}\n"
        f"{error_text} | Servo: {servo.state}"
    )


def _target_lost_summary(
    context: ReadyContext,
    errors: Sequence[float],
    detected_frames: int,
    total_frames: int,
    lost_event_count: int,
    lost_duration_s: float | None,
    reacquisition_time_s: float | None,
    recovery_time_s: float | None,
    active_hold_max_deviation: float,
    completed: bool,
    log_path: Path,
) -> dict[str, object]:
    metrics = _episode_metrics(errors, detected_frames, total_frames)
    passed = (
        completed
        and lost_event_count >= 1
        and reacquisition_time_s is not None
        and recovery_time_s is not None
        and active_hold_max_deviation < 0.01
        and context.max_locked_deviation < 0.005
    )
    return {
        "experiment": "TARGET_LOST_RECOVERY",
        "condition": "predefined out-of-view target for 2.5 s",
        "duration_s": TARGET_LOST_DURATION_SECONDS,
        "warmup_duration_s": context.warmup_duration_s,
        "warmup_initial_error": context.warmup_initial_error,
        "ready_ex": context.ready_ex,
        "ready_ey": context.ready_ey,
        **metrics,
        "lost_event_count": lost_event_count,
        "lost_duration_s": lost_duration_s,
        "reacquisition_time_s": reacquisition_time_s,
        "recovery_time_s": recovery_time_s,
        "active_hold_max_deviation_rad": active_hold_max_deviation,
        "locked_joint_max_deviation_rad": context.max_locked_deviation,
        "result": "PASS" if passed else "FAIL",
        "log_path": str(log_path),
    }


def _run_target_lost_recovery_trial(
    client_id: int,
    camera_display: EyeInHandRgbDisplay,
) -> dict[str, object]:
    """Run an out-of-view interval and verify vision-only automatic recovery."""
    context = _create_ready_context(client_id, camera_display, "TARGET LOST / RECOVERY")
    log_path = LOG_DIRECTORY / "stage10_target_lost.csv"
    target_orientation = (0.0, 0.0, 0.0, 1.0)
    total_steps = max(1, round(TARGET_LOST_DURATION_SECONDS / sim.TIME_STEP))
    total_frames = detected_frames = lost_event_count = 0
    errors: list[float] = []
    previously_detected = True
    lost_started_at: float | None = None
    lost_duration_s: float | None = None
    reacquisition_time_s: float | None = None
    recovery_start_s: float | None = None
    recovery_time_s: float | None = None
    recovery_ready_frames = 0
    lost_hold_reference: dict[int, float] | None = None
    active_hold_max_deviation = 0.0
    latest_detection: RedTargetDetection | None = None

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=_stage10_csv_fields())
        writer.writeheader()
        for step in range(total_steps):
            simulation_time_s = step * sim.TIME_STEP
            is_hidden = TARGET_LOST_HIDE_START_SECONDS <= simulation_time_s < TARGET_LOST_HIDE_END_SECONDS
            phase = "TARGET LOST" if is_hidden else (
                "RECOVERY" if simulation_time_s >= TARGET_LOST_HIDE_END_SECONDS else "TRACKING"
            )
            target_world_position = (
                TARGET_LOST_HIDDEN_WORLD if is_hidden else sim.STAGE9_TARGET_CENTER_WORLD
            )
            _apply_target_pose(context.target_body_id, target_world_position, client_id)
            p.resetBaseVelocity(
                context.target_body_id,
                linearVelocity=(0.0, 0.0, 0.0),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client_id,
            )
            _step_physics(context, client_id)
            if lost_hold_reference is not None:
                current_active = sim.capture_active_joint_hold_targets(
                    context.robot_id, context.active_joint_indices, client_id
                )
                active_hold_max_deviation = max(
                    active_hold_max_deviation,
                    max(
                        abs(current_active[index] - lost_hold_reference[index])
                        for index in context.active_joint_indices
                    ),
                )
            if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                sim.update_camera_reference_axes(
                    context.robot_id, client_id, context.camera_axis_debug_item_ids
                )
            raw_detection = sim.update_live_eye_in_hand_camera(
                context.robot_id,
                step,
                client_id,
                camera_display,
                _stage10_overlay(
                    "TARGET LOST / RECOVERY", "OUT OF VIEW", phase, simulation_time_s,
                    TARGET_LOST_DURATION_SECONDS, context.servo, latest_detection,
                ),
            )
            if raw_detection is None:
                time.sleep(sim.TIME_STEP)
                continue

            latest_detection = raw_detection
            total_frames += 1
            if raw_detection.detected:
                detected_frames += 1
                assert raw_detection.pixel_error is not None
                ex, ey = raw_detection.pixel_error
                errors.append(sqrt(ex * ex + ey * ey))
            if previously_detected and not raw_detection.detected:
                lost_event_count += 1
                lost_started_at = simulation_time_s
                lost_hold_reference = sim.capture_active_joint_hold_targets(
                    context.robot_id, context.active_joint_indices, client_id
                )
                active_hold_max_deviation = 0.0
                print(f"Stage 10A: TARGET LOST at t={simulation_time_s:.3f} s; holding CURRENT pose.")
            if not previously_detected and raw_detection.detected:
                if lost_started_at is not None:
                    lost_duration_s = simulation_time_s - lost_started_at
                if simulation_time_s >= TARGET_LOST_HIDE_END_SECONDS:
                    reacquisition_time_s = simulation_time_s - TARGET_LOST_HIDE_END_SECONDS
                    recovery_start_s = simulation_time_s
                    recovery_ready_frames = 0
                    print(
                        "Stage 10A: target re-detected from CURRENT pose; "
                        f"reacquisition={reacquisition_time_s:.3f} s."
                    )
                lost_hold_reference = None
            previously_detected = raw_detection.detected

            ik_command_issued, nonzero_correction = _control_latest_measurement(
                context, raw_detection, client_id
            )
            if raw_detection.detected and recovery_start_s is not None and recovery_time_s is None:
                assert raw_detection.pixel_error is not None
                ex, ey = raw_detection.pixel_error
                recovery_ready_frames = (
                    recovery_ready_frames + 1 if abs(ex) <= 2 and abs(ey) <= 2 else 0
                )
                if recovery_ready_frames >= sim.TWO_D_PRECISION_SETTLE_FRAMES_REQUIRED:
                    recovery_time_s = simulation_time_s - recovery_start_s
                    print(f"Stage 10A: recovered precision tracking in {recovery_time_s:.3f} s.")
            writer.writerow(
                _stage10_log_row(
                    simulation_time_s, "TARGET_LOST_RECOVERY", "OUT_OF_VIEW", phase,
                    raw_detection, raw_detection, 0.0, 0.0, context, target_world_position,
                    ik_command_issued, nonzero_correction, client_id,
                )
            )
            context.debug_text_id = sim.update_motion_debug_text(
                _stage10_overlay(
                    "TARGET LOST / RECOVERY", "OUT OF VIEW", phase, simulation_time_s,
                    TARGET_LOST_DURATION_SECONDS, context.servo, raw_detection,
                ),
                context.debug_text_id,
                client_id,
            )
            time.sleep(sim.TIME_STEP)

    summary = _target_lost_summary(
        context, errors, detected_frames, total_frames, lost_event_count, lost_duration_s,
        reacquisition_time_s, recovery_time_s, active_hold_max_deviation, True, log_path,
    )
    print("\n===== Stage 10A Target Lost / Recovery =====")
    print("Lost event count:", summary["lost_event_count"])
    print("Lost duration:", _format_metric(summary["lost_duration_s"], "s"))
    print("Reacquisition time:", _format_metric(summary["reacquisition_time_s"], "s"))
    print("Recovery time:", _format_metric(summary["recovery_time_s"], "s"))
    print("Active HOLD maximum deviation:", _format_metric(summary["active_hold_max_deviation_rad"], "rad"))
    print("Result:", summary["result"])
    return summary


def _run_noise_trial(
    standard_deviation_pixels: float,
    client_id: int,
    camera_display: EyeInHandRgbDisplay,
) -> dict[str, object]:
    """Run the frozen MEDIUM motion with noise only on the RGB measurement."""
    condition = f"{standard_deviation_pixels:g} px"
    context = _create_ready_context(client_id, camera_display, f"VISUAL NOISE {condition}")
    suffix = str(int(standard_deviation_pixels))
    log_path = LOG_DIRECTORY / f"stage10_noise_{suffix}.csv"
    random_generator = random.Random(NOISE_RANDOM_SEED)
    target_orientation = (0.0, 0.0, 0.0, 1.0)
    frequency_hz = 0.10  # Stage 9 MEDIUM trajectory; frozen experiment variable.
    total_steps = max(1, round(NOISE_DURATION_SECONDS / sim.TIME_STEP))
    total_frames = detected_frames = lost_event_count = 0
    errors: list[float] = []
    previously_detected = True
    latest_detection: RedTargetDetection | None = None
    ik_command_count = nonzero_correction_count = 0

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=_stage10_csv_fields())
        writer.writeheader()
        for step in range(total_steps):
            simulation_time_s = step * sim.TIME_STEP
            target_world_position = sim._stage9_target_world_position(simulation_time_s, frequency_hz)
            _apply_target_pose(context.target_body_id, target_world_position, client_id)
            p.resetBaseVelocity(
                context.target_body_id,
                linearVelocity=sim._stage9_target_world_velocity(simulation_time_s, frequency_hz),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client_id,
            )
            _step_physics(context, client_id)
            if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                sim.update_camera_reference_axes(
                    context.robot_id, client_id, context.camera_axis_debug_item_ids
                )
            raw_detection = sim.update_live_eye_in_hand_camera(
                context.robot_id,
                step,
                client_id,
                camera_display,
                _stage10_overlay(
                    "VISUAL NOISE", condition, "TRACKING", simulation_time_s,
                    NOISE_DURATION_SECONDS, context.servo, latest_detection,
                ),
            )
            if raw_detection is None:
                time.sleep(sim.TIME_STEP)
                continue

            latest_detection = raw_detection
            total_frames += 1
            if raw_detection.detected:
                detected_frames += 1
                assert raw_detection.pixel_error is not None
                ex, ey = raw_detection.pixel_error
                errors.append(sqrt(ex * ex + ey * ey))
            elif previously_detected:
                lost_event_count += 1
            previously_detected = raw_detection.detected
            measurement, noise_u, noise_v = _build_noisy_measurement(
                raw_detection, random_generator, standard_deviation_pixels
            )
            ik_command_issued, nonzero_correction = _control_latest_measurement(
                context, measurement, client_id
            )
            ik_command_count += int(ik_command_issued)
            nonzero_correction_count += int(nonzero_correction)
            writer.writerow(
                _stage10_log_row(
                    simulation_time_s, "VISUAL_NOISE", condition, "TRACKING", raw_detection,
                    measurement, noise_u, noise_v, context, target_world_position,
                    ik_command_issued, nonzero_correction, client_id,
                )
            )
            context.debug_text_id = sim.update_motion_debug_text(
                _stage10_overlay(
                    "VISUAL NOISE", condition, "TRACKING", simulation_time_s,
                    NOISE_DURATION_SECONDS, context.servo, raw_detection,
                ),
                context.debug_text_id,
                client_id,
            )
            time.sleep(sim.TIME_STEP)

    metrics = _episode_metrics(errors, detected_frames, total_frames)
    result = "PASS" if total_frames and detected_frames == total_frames and context.max_locked_deviation < 0.005 else "FAIL"
    summary: dict[str, object] = {
        "experiment": "VISUAL_NOISE",
        "condition": condition,
        "duration_s": NOISE_DURATION_SECONDS,
        "warmup_duration_s": context.warmup_duration_s,
        "warmup_initial_error": context.warmup_initial_error,
        "ready_ex": context.ready_ex,
        "ready_ey": context.ready_ey,
        **metrics,
        "lost_event_count": lost_event_count,
        "lost_duration_s": None,
        "reacquisition_time_s": None,
        "recovery_time_s": None,
        "active_hold_max_deviation_rad": None,
        "locked_joint_max_deviation_rad": context.max_locked_deviation,
        "ik_command_count": ik_command_count,
        "nonzero_correction_count": nonzero_correction_count,
        "result": result,
        "log_path": str(log_path),
    }
    print(f"\n===== Stage 10B Visual Noise: {condition} =====")
    print("Mean Error:", _format_metric(summary["mean_error"], "px"))
    print("RMSE:", _format_metric(summary["rmse"], "px"))
    print("P95:", _format_metric(summary["p95_error"], "px"))
    print("Max Error:", _format_metric(summary["max_error"], "px"))
    print("Target Lost Count:", summary["lost_event_count"])
    print("Detection Rate:", _format_metric(summary["detection_rate"], "%"))
    print("Result:", result)
    return summary


def _format_metric(value: object, unit: str) -> str:
    return f"{float(value):.3f} {unit}" if value is not None else "N/A"


def _write_stage10_summary(summaries: Sequence[dict[str, object]]) -> Path:
    """Write a single audit-friendly summary without altering trial CSV data."""
    path = LOG_DIRECTORY / "stage10_summary.csv"
    fields = (
        "experiment", "condition", "duration_s", "warmup_duration_s", "warmup_initial_error",
        "ready_ex", "ready_ey", "frames", "mean_error", "rmse", "p95_error", "max_error",
        "detection_rate", "lost_event_count", "lost_duration_s", "reacquisition_time_s",
        "recovery_time_s", "active_hold_max_deviation_rad", "locked_joint_max_deviation_rad",
        "ik_command_count", "nonzero_correction_count", "result",
    )
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
    return path


def run_stage10_robustness_evaluation() -> None:
    """Run Stage 10A/10B without changing the accepted control baseline."""
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 10.")
    camera_display: EyeInHandRgbDisplay | None = None
    try:
        camera_display = EyeInHandRgbDisplay("Eye-in-Hand RGB - Stage 10")
        print("Stage 10: Robustness and Disturbance Evaluation")
        print("Frozen baseline: controller gains/bands, locks, IK, camera and HSV are unchanged.")
        print("Controller input: fresh Eye-in-Hand RGB/OpenCV detection only.")
        print("Target world data is schedule/log-only; it is not passed into controller functions.")
        summaries = [_run_target_lost_recovery_trial(client_id, camera_display)]
        summaries.extend(
            _run_noise_trial(noise_level, client_id, camera_display)
            for noise_level in NOISE_LEVELS_PIXELS
        )
        summary_path = _write_stage10_summary(summaries)
        stage_passed = all(summary["result"] == "PASS" for summary in summaries)
        print("\n===== Stage 10 Completion =====")
        for summary in summaries:
            print(f"{summary['experiment']} ({summary['condition']}): {summary['result']}")
        print("Summary CSV:", summary_path)
        print("Controller use of target world position: CONFIRMED ABSENT.")
        print("Stage 10:", "PASS" if stage_passed else "FAIL")
    finally:
        if camera_display is not None:
            camera_display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)
