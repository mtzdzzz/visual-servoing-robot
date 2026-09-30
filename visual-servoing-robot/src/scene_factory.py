"""Shared, behavior-preserving PyBullet scene construction primitives."""
from __future__ import annotations

import time
from typing import Sequence

import pybullet as p
import pybullet_data

import robotics_core as sim
import visual_servo_runtime as runtime
from camera_observation import create_manual_draggable_red_ground_target


def set_static_body_pose(body_id: int, position: Sequence[float], client_id: int) -> None:
    p.resetBasePositionAndOrientation(body_id, position, (0.0, 0.0, 0.0, 1.0), physicsClientId=client_id)
    p.resetBaseVelocity(body_id, linearVelocity=(0.0, 0.0, 0.0), angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client_id)


def create_coloured_sphere(radius: float, rgba: Sequence[float], client_id: int) -> int:
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client_id)
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                             basePosition=(0.0, 0.0, radius), physicsClientId=client_id)


def create_box_obstacle(dimensions: Sequence[float], rgba: Sequence[float], client_id: int) -> int:
    half = [float(value) / 2.0 for value in dimensions]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client_id)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=client_id)
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                             basePosition=(0.42, -0.30, half[2]), physicsClientId=client_id)


def run_hold_steps(context: runtime.ReadyContext, number_of_steps: int, client_id: int, realtime: bool = True) -> None:
    for step in range(number_of_steps):
        runtime.step_physics(context, client_id)
        if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
            sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
        if realtime:
            time.sleep(sim.TIME_STEP)


def freeze_at_current_pose(context: runtime.ReadyContext, client_id: int) -> None:
    context.active_hold_targets = sim.capture_active_joint_hold_targets(
        context.robot_id, context.active_joint_indices, client_id
    )
    context.commanded_joint_targets = sim.build_hold_joint_targets(
        context.arm_joint_indices, context.locked_initial_positions, context.active_hold_targets
    )


def create_static_localization_context(client_id: int) -> runtime.ReadyContext:
    """Build the established camera-visible Stage 14 HOLD scene."""
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
    _, active_start_positions = sim.STAGE7_TRIAL_ACTIVE_START_POSITIONS[0]
    for joint_index, target_position in zip(active_indices, active_start_positions):
        p.resetJointState(robot_id, joint_index, target_position, physicsClientId=client_id)
    target_body_id = create_manual_draggable_red_ground_target(client_id)
    set_static_body_pose(target_body_id, sim.STAGE9_TARGET_CENTER_WORLD, client_id)
    active_hold = sim.capture_active_joint_hold_targets(robot_id, active_indices, client_id)
    context = runtime.ReadyContext(
        robot_id=robot_id, target_body_id=target_body_id, arm_joint_indices=arm_joint_indices,
        active_joint_indices=active_indices, locked_joint_indices=locked_indices,
        locked_initial_positions=locked_positions,
        commanded_joint_targets=sim.build_hold_joint_targets(arm_joint_indices, locked_positions, active_hold),
        active_hold_targets=active_hold, servo=sim.TwoDimensionalVisualServo(),
        camera_axis_debug_item_ids=[], debug_text_id=None, warmup_duration_s=0.0,
        warmup_initial_error=None, ready_ex=None, ready_ey=None, max_locked_deviation=0.0,
    )
    run_hold_steps(context, round(1.0 / sim.TIME_STEP), client_id)
    freeze_at_current_pose(context, client_id)
    sim.update_camera_reference_axes(robot_id, client_id, context.camera_axis_debug_item_ids)
    return context
