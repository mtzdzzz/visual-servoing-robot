"""Shared motor/visual-servo runtime used by experiments and demos.

The formulas and constants remain owned by :mod:`robotics_core`; this module
only provides the accepted context lifecycle and latest-observation adapter.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
from math import sqrt
import time

import pybullet as p
import pybullet_data

import robotics_core as sim
from camera_observation import (
    EyeInHandRgbDisplay, RedTargetDetection, create_manual_draggable_red_ground_target,
)


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


def create_ready_context(
    client_id: int, camera_display: EyeInHandRgbDisplay, label: str,
) -> ReadyContext:
    """Create the accepted single-target scene and reach raw-RGB READY."""
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
        target_body_id, sim.STAGE9_TARGET_CENTER_WORLD, target_orientation,
        physicsClientId=client_id,
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
        f"VISUAL SERVO\n{label}\nPhase: WARM-UP", None, client_id
    )
    warmup_steps = max(1, round(sim.STAGE9_WARMUP_MAX_SIMULATION_SECONDS / sim.TIME_STEP))
    for step in range(warmup_steps):
        p.resetBasePositionAndOrientation(
            target_body_id, sim.STAGE9_TARGET_CENTER_WORLD, target_orientation,
            physicsClientId=client_id,
        )
        sim.apply_constrained_arm_position_control(
            robot_id, arm_joint_indices, commanded, locked_positions, client_id
        )
        p.stepSimulation(physicsClientId=client_id)
        deviations = sim.get_locked_joint_deviations(robot_id, locked_positions, client_id)
        max_locked_deviation = max(
            max_locked_deviation, max(map(abs, deviations.values()), default=0.0)
        )
        if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
            sim.update_camera_reference_axes(robot_id, client_id, axis_ids)
        raw_detection = sim.update_live_eye_in_hand_camera(
            robot_id, step, client_id, camera_display,
            f"{label} | Phase: WARM-UP",
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
                commanded = sim.build_hold_joint_targets(
                    arm_joint_indices, locked_positions, active_hold
                )
                servo.hold_requested = False
            if ready_frames >= sim.TWO_D_PRECISION_SETTLE_FRAMES_REQUIRED:
                assert raw_detection.pixel_error is not None
                return ReadyContext(
                    robot_id, target_body_id, arm_joint_indices, active_indices,
                    locked_indices, locked_positions, commanded, active_hold,
                    sim.TwoDimensionalVisualServo(), axis_ids, debug_text_id,
                    (step + 1) * sim.TIME_STEP, warmup_initial_error,
                    raw_detection.pixel_error[0], raw_detection.pixel_error[1],
                    max_locked_deviation,
                )
        time.sleep(sim.TIME_STEP)
    raise RuntimeError(f"{label}: WARM-UP did not reach READY within the protocol limit.")


def control_latest_measurement(
    context: ReadyContext, measurement: RedTargetDetection, client_id: int,
) -> tuple[bool, bool]:
    """Apply the frozen latest-observation controller and return command flags."""
    if context.servo.state.startswith("FAIL"):
        return False, False
    if context.servo.state == "TARGET LOST / HOLD" and measurement.detected:
        context.servo = sim.TwoDimensionalVisualServo()
    with contextlib.redirect_stdout(io.StringIO()):
        command = sim._stage9_start_dynamic_tracking_from_current_pose(
            context.servo, measurement, context.robot_id,
            context.locked_joint_indices, context.locked_initial_positions, client_id,
        )
    issued = command is not None and not context.servo.hold_requested
    nonzero = bool(
        issued and command is not None
        and (abs(command.camera_delta_c[0]) > 1e-12 or abs(command.camera_delta_c[1]) > 1e-12)
    )
    if issued and command is not None:
        context.commanded_joint_targets = list(command.joint_targets)
    if context.servo.hold_requested:
        context.active_hold_targets = sim.capture_active_joint_hold_targets(
            context.robot_id, context.active_joint_indices, client_id
        )
        context.commanded_joint_targets = sim.build_hold_joint_targets(
            context.arm_joint_indices, context.locked_initial_positions,
            context.active_hold_targets,
        )
        context.servo.hold_requested = False
    return issued, nonzero


def step_physics(context: ReadyContext, client_id: int) -> None:
    """Apply the accepted motor interface and update locked-joint telemetry."""
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

