"""Inspect a Panda arm or run safe PyBullet motion demonstrations.

This module is intentionally limited to robot loading and inspection.  It does
not create a camera or implement perception, visual servoing, PID, ROS, or
Gazebo integration.
"""

from __future__ import annotations

import argparse
from math import sqrt
import time
from typing import Sequence

import pybullet as p
import pybullet_data


JOINT_TYPE_NAMES = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
    p.JOINT_POINT2POINT: "point2point",
    p.JOINT_GEAR: "gear",
}

CONTROLLED_JOINT_INDEX = 0  # Panda's first revolute joint: panda_joint1.
SAFE_TARGET_POSITION = 0.30  # radians; well inside panda_joint1's limits.
MAX_JOINT_VELOCITY = 0.06  # radians/second; reaches the target in about 5 seconds.
POSITION_GAIN = 0.03
MAX_MOTOR_FORCE = 5.0
TIME_STEP = 1.0 / 240.0

# Verified from the loaded franka_panda/panda.urdf: index 7 is panda_link8.
# E is the Panda's fixed wrist/flange link, before panda_hand.  It is the
# stable mechanical reference to which a future eye-in-hand camera is mounted.
END_EFFECTOR_LINK_INDEX = 7
END_EFFECTOR_LINK_NAME = "panda_link8"
CAMERA_REFERENCE_FRAME_NAME = "C"
# MVP camera-reference transform.  The virtual camera origin C coincides with
# the selected end-effector frame E: T_E_C = Identity.
T_E_C_POSITION = (0.0, 0.0, 0.0)
T_E_C_ORIENTATION = (0.0, 0.0, 0.0, 1.0)
# A tested, reachable world position for the future camera origin C, in metres.
SAFE_CAMERA_ORIGIN_TARGET_POSITION = (0.45, 0.00, 0.55)
DEFAULT_CAMERA_REFERENCE_TARGET_SEQUENCE = (
    ("Camera Origin Test", SAFE_CAMERA_ORIGIN_TARGET_POSITION),
)
MAX_IK_JOINT_VELOCITY = 0.20  # radians/second
MAX_IK_MOTOR_FORCE = 20.0
IK_VELOCITY_GAIN = 0.8
TRAJECTORY_MAX_WAYPOINT_SPEED = 0.18  # radians/second
POSITION_STATUS_INTERVAL_SECONDS = 1.0
MIN_TRAJECTORY_DURATION = 5.0  # seconds; Stage 3 visual-motion minimum
POST_TRAJECTORY_SETTLE_SECONDS = 3.0
STARTUP_PAUSE_SECONDS = 3.0
DEBUG_STATUS_TEXT_POSITION = (0.0, 0.0, 1.35)
DEBUG_TARGET_MARKER_SIZE = 0.045
CAMERA_AXIS_LENGTH = 0.10


def _decode_name(value: bytes | str) -> str:
    """Convert PyBullet's byte-string names into ordinary Python strings."""
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a compact, dependency-free text table."""
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[column]) for column, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def print_panda_structure(robot_id: int, client_id: int) -> None:
    """Print every Panda joint and its parent/child link information.

    ``getJointInfo`` supplies the joint index, name, type, position limits,
    child link name, and parent-link index used below.
    """
    base_link_name = _decode_name(p.getBodyInfo(robot_id, physicsClientId=client_id)[0])
    joint_count = p.getNumJoints(robot_id, physicsClientId=client_id)
    child_link_names: dict[int, str] = {-1: base_link_name}
    joint_infos = []

    for joint_index in range(joint_count):
        joint_info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        joint_infos.append(joint_info)
        child_link_names[joint_index] = _decode_name(joint_info[12])

    rows: list[list[str]] = []
    for joint_info in joint_infos:
        joint_index = joint_info[0]
        joint_name = _decode_name(joint_info[1])
        joint_type = JOINT_TYPE_NAMES.get(joint_info[2], f"unknown ({joint_info[2]})")
        lower_limit, upper_limit = joint_info[8], joint_info[9]
        parent_link_index = joint_info[16]
        parent_link_name = child_link_names.get(parent_link_index, "unknown")
        child_link_name = _decode_name(joint_info[12])

        rows.append(
            [
                str(joint_index),
                joint_name,
                joint_type,
                f"{parent_link_name} ({parent_link_index})",
                child_link_name,
                f"{lower_limit:.6f}",
                f"{upper_limit:.6f}",
            ]
        )

    print("Panda robot structure")
    print(f"Base link: {base_link_name}")
    print(f"Joint count: {joint_count}")
    print(
        _format_table(
            [
                "Index",
                "Joint name",
                "Joint type",
                "Parent link (index)",
                "Child link",
                "Lower limit",
                "Upper limit",
            ],
            rows,
        )
    )


def inspect_panda_structure() -> None:
    """Start a headless PyBullet client, load Panda, and print its structure."""
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Unable to connect to PyBullet in DIRECT mode.")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=client_id,
        )
        if robot_id < 0:
            raise RuntimeError("Unable to load franka_panda/panda.urdf.")

        print_panda_structure(robot_id, client_id)
    finally:
        p.disconnect(physicsClientId=client_id)


def get_panda_arm_joint_indices(robot_id: int, client_id: int) -> list[int]:
    """Read the seven Panda revolute arm joints from the loaded URDF."""
    arm_joint_indices = []
    joint_count = p.getNumJoints(robot_id, physicsClientId=client_id)
    for joint_index in range(joint_count):
        joint_info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        joint_name = _decode_name(joint_info[1])
        if joint_info[2] == p.JOINT_REVOLUTE and joint_name.startswith("panda_joint"):
            arm_joint_indices.append(joint_index)

    if len(arm_joint_indices) != 7:
        raise RuntimeError(
            "Expected seven Panda revolute arm joints, "
            f"but found {arm_joint_indices}."
        )
    return arm_joint_indices


def get_link_index_by_name(robot_id: int, link_name: str, client_id: int) -> int:
    """Find a PyBullet link index from the child-link name read from the URDF."""
    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        joint_info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        if _decode_name(joint_info[12]) == link_name:
            return joint_index
    raise RuntimeError(f"Link {link_name!r} was not found in the loaded Panda URDF.")


def get_end_effector_reference_pose(
    robot_id: int,
    client_id: int,
) -> tuple[list[float], list[float]]:
    """Read the selected E-frame world position and orientation from PyBullet."""
    loaded_link_name = _decode_name(
        p.getJointInfo(
            robot_id,
            END_EFFECTOR_LINK_INDEX,
            physicsClientId=client_id,
        )[12]
    )
    if loaded_link_name != END_EFFECTOR_LINK_NAME:
        raise RuntimeError(
            "Configured end-effector reference does not match the loaded URDF: "
            f"expected {END_EFFECTOR_LINK_NAME!r}, got {loaded_link_name!r}."
        )
    link_state = p.getLinkState(
        robot_id,
        END_EFFECTOR_LINK_INDEX,
        computeForwardKinematics=True,
        physicsClientId=client_id,
    )
    return list(link_state[4]), list(link_state[5])


def get_camera_reference_pose(
    robot_id: int,
    client_id: int,
) -> tuple[list[float], list[float]]:
    """Compose the world pose of C from E and the fixed transform T_E_C."""
    end_effector_position, end_effector_orientation = get_end_effector_reference_pose(
        robot_id,
        client_id,
    )
    camera_position, camera_orientation = p.multiplyTransforms(
        end_effector_position,
        end_effector_orientation,
        T_E_C_POSITION,
        T_E_C_ORIENTATION,
    )
    return list(camera_position), list(camera_orientation)


def print_camera_reference_inspection(robot_id: int, client_id: int) -> None:
    """Print the candidate link poses and current E/C-frame MVP definition."""
    print("\nPanda end-effector reference inspection:")
    for link_name in ("panda_link7", "panda_link8", "panda_hand"):
        link_index = get_link_index_by_name(robot_id, link_name, client_id)
        joint_info = p.getJointInfo(robot_id, link_index, physicsClientId=client_id)
        link_state = p.getLinkState(
            robot_id,
            link_index,
            computeForwardKinematics=True,
            physicsClientId=client_id,
        )
        print(
            f"  {link_name}: link index={link_index}, "
            f"joint={_decode_name(joint_info[1])}, parent link index={joint_info[16]}"
        )
        print("    position:", [round(value, 4) for value in link_state[4]])
        print("    orientation xyzw:", [round(value, 4) for value in link_state[5]])

    end_effector_position, end_effector_orientation = get_end_effector_reference_pose(
        robot_id,
        client_id,
    )
    print("Current E frame:")
    print(f"  link index: {END_EFFECTOR_LINK_INDEX}")
    print(f"  link name: {END_EFFECTOR_LINK_NAME}")
    print("  world position:", [round(value, 4) for value in end_effector_position])
    print("  world orientation xyzw:", [round(value, 4) for value in end_effector_orientation])
    camera_position, camera_orientation = get_camera_reference_pose(robot_id, client_id)
    print(
        "Camera reference C MVP: T_E_C = Identity "
        f"(translation={list(T_E_C_POSITION)}, orientation={list(T_E_C_ORIENTATION)})."
    )
    print("  C world position:", [round(value, 4) for value in camera_position])
    print("  C world orientation xyzw:", [round(value, 4) for value in camera_orientation])


def calculate_position_only_ik(
    robot_id: int,
    target_position: Sequence[float],
    client_id: int,
) -> tuple[list[int], list[float]]:
    """Calculate q1...q7 for a Camera-reference XYZ target without orientation."""
    if len(target_position) != 3:
        raise ValueError("target_position must contain exactly [x, y, z].")

    arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
    movable_joint_indices = []
    lower_limits = []
    upper_limits = []
    rest_poses = []

    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        joint_info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        if joint_info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue
        movable_joint_indices.append(joint_index)
        lower_limits.append(joint_info[8])
        upper_limits.append(joint_info[9])
        rest_poses.append(
            p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        )

    ik_solution = p.calculateInverseKinematics(
        robot_id,
        END_EFFECTOR_LINK_INDEX,
        # T_E_C is Identity in this MVP, therefore this E-link target is also
        # the desired world position of the future Camera origin C.
        targetPosition=target_position,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=[upper - lower for lower, upper in zip(lower_limits, upper_limits)],
        restPoses=rest_poses,
        maxNumIterations=300,
        residualThreshold=1e-5,
        physicsClientId=client_id,
    )
    if len(ik_solution) != len(movable_joint_indices):
        raise RuntimeError(
            "PyBullet IK returned an unexpected number of movable-joint values: "
            f"{len(ik_solution)}."
        )

    solution_by_joint = dict(zip(movable_joint_indices, ik_solution))
    return arm_joint_indices, [solution_by_joint[index] for index in arm_joint_indices]


def _smoothstep(progress: float) -> float:
    """Ease a zero-to-one trajectory progress with zero endpoint velocity."""
    bounded_progress = max(0.0, min(1.0, progress))
    return bounded_progress * bounded_progress * (3.0 - 2.0 * bounded_progress)


def calculate_trajectory_duration(
    start_positions: Sequence[float],
    target_positions: Sequence[float],
) -> float:
    """Choose a duration that keeps every smooth waypoint below the speed cap."""
    if len(start_positions) != len(target_positions):
        raise ValueError("Start and target joint-position lists must have equal lengths.")

    largest_joint_change = max(
        (abs(target - start) for start, target in zip(start_positions, target_positions)),
        default=0.0,
    )
    # The maximum derivative of smoothstep is 1.5, so this bounds the
    # commanded waypoint speed rather than only its average speed.
    return max(
        MIN_TRAJECTORY_DURATION,
        1.5 * largest_joint_change / TRAJECTORY_MAX_WAYPOINT_SPEED,
    )


def apply_arm_position_control(
    robot_id: int,
    arm_joint_indices: Sequence[int],
    waypoint_positions: Sequence[float],
    client_id: int,
) -> None:
    """Apply one low-speed POSITION_CONTROL waypoint to the seven arm joints."""
    if len(arm_joint_indices) != 7 or len(waypoint_positions) != 7:
        raise ValueError("POSITION_CONTROL requires exactly seven Panda arm targets.")

    for joint_index, waypoint_position in zip(arm_joint_indices, waypoint_positions):
        joint_info = p.getJointInfo(
            robot_id,
            joint_index,
            physicsClientId=client_id,
        )
        p.setJointMotorControl2(
            robot_id,
            joint_index,
            p.POSITION_CONTROL,
            targetPosition=waypoint_position,
            force=min(MAX_IK_MOTOR_FORCE, joint_info[10]),
            positionGain=POSITION_GAIN,
            velocityGain=IK_VELOCITY_GAIN,
            maxVelocity=MAX_IK_JOINT_VELOCITY,
            physicsClientId=client_id,
        )


def get_end_effector_position(robot_id: int, client_id: int) -> list[float]:
    """Read the current future-Camera-origin world position from PyBullet."""
    camera_position, _ = get_camera_reference_pose(robot_id, client_id)
    return camera_position


def print_position_verification(
    target_position: Sequence[float],
    robot_id: int,
    client_id: int,
    label: str,
    point_name: str | None = None,
) -> float:
    """Print target, measured Camera-reference position, component error, and norm."""
    actual_position = get_end_effector_position(robot_id, client_id)
    position_error = [
        target - actual for target, actual in zip(target_position, actual_position)
    ]
    error_norm = sqrt(sum(component * component for component in position_error))

    print(f"\n{label}")
    if point_name is not None:
        print(f"Current Target Point: {point_name}")
    print("Target Position:", [round(value, 4) for value in target_position])
    print("Actual Position:", [round(value, 4) for value in actual_position])
    print("Position Error:", [round(value, 4) for value in position_error])
    print(f"Error Norm: {error_norm:.4f} m")
    return error_norm


def add_target_markers(
    target_sequence: Sequence[tuple[str, Sequence[float]]],
    client_id: int,
) -> None:
    """Draw persistent cross markers and labels for every planned XYZ target."""
    marker_colors = ([0.2, 0.9, 0.3], [0.2, 0.6, 1.0], [1.0, 0.6, 0.1])
    for point_index, (point_name, target_position) in enumerate(target_sequence):
        x, y, z = target_position
        color = marker_colors[point_index % len(marker_colors)]
        marker_segments = (
            ((x - DEBUG_TARGET_MARKER_SIZE, y, z), (x + DEBUG_TARGET_MARKER_SIZE, y, z)),
            ((x, y - DEBUG_TARGET_MARKER_SIZE, z), (x, y + DEBUG_TARGET_MARKER_SIZE, z)),
            ((x, y, z - DEBUG_TARGET_MARKER_SIZE), (x, y, z + DEBUG_TARGET_MARKER_SIZE)),
        )
        for start, end in marker_segments:
            p.addUserDebugLine(
                start,
                end,
                lineColorRGB=color,
                lineWidth=3,
                lifeTime=0,
                physicsClientId=client_id,
            )
        p.addUserDebugText(
            f"Point {point_name}",
            (x, y, z + DEBUG_TARGET_MARKER_SIZE * 1.5),
            textColorRGB=color,
            textSize=1.4,
            lifeTime=0,
            physicsClientId=client_id,
        )


def update_camera_reference_axes(
    robot_id: int,
    client_id: int,
    debug_item_ids: list[int],
) -> None:
    """Draw the moving Camera-reference C axes from the current E-frame pose."""
    camera_position, camera_orientation = get_camera_reference_pose(robot_id, client_id)
    axis_definitions = (
        ((CAMERA_AXIS_LENGTH, 0.0, 0.0), [1.0, 0.0, 0.0]),  # X: red
        ((0.0, CAMERA_AXIS_LENGTH, 0.0), [0.0, 1.0, 0.0]),  # Y: green
        ((0.0, 0.0, CAMERA_AXIS_LENGTH), [0.0, 0.0, 1.0]),  # Z: blue
    )
    new_debug_item_ids = []
    for axis_index, (axis_end_in_camera, color) in enumerate(axis_definitions):
        axis_end_world, _ = p.multiplyTransforms(
            camera_position,
            camera_orientation,
            axis_end_in_camera,
            (0.0, 0.0, 0.0, 1.0),
        )
        replace_item_id = (
            debug_item_ids[axis_index]
            if axis_index < len(debug_item_ids) and debug_item_ids[axis_index] >= 0
            else -1
        )
        new_debug_item_ids.append(
            p.addUserDebugLine(
                camera_position,
                axis_end_world,
                lineColorRGB=color,
                lineWidth=4,
                lifeTime=0,
                replaceItemUniqueId=replace_item_id,
                physicsClientId=client_id,
            )
        )

    label_position, _ = p.multiplyTransforms(
        camera_position,
        camera_orientation,
        (0.0, 0.0, CAMERA_AXIS_LENGTH * 1.25),
        (0.0, 0.0, 0.0, 1.0),
    )
    replace_label_id = (
        debug_item_ids[3] if len(debug_item_ids) > 3 and debug_item_ids[3] >= 0 else -1
    )
    new_debug_item_ids.append(
        p.addUserDebugText(
            f"Camera Reference {CAMERA_REFERENCE_FRAME_NAME} (T_E_C = I)",
            label_position,
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=0,
            replaceItemUniqueId=replace_label_id,
            physicsClientId=client_id,
        )
    )
    debug_item_ids[:] = new_debug_item_ids


def update_motion_debug_text(
    text: str,
    previous_text_id: int | None,
    client_id: int,
) -> int:
    """Replace the single persistent GUI status text without leaving stale labels."""
    if previous_text_id is not None and previous_text_id >= 0:
        p.removeUserDebugItem(previous_text_id, physicsClientId=client_id)
    return p.addUserDebugText(
        text,
        DEBUG_STATUS_TEXT_POSITION,
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.6,
        lifeTime=0,
        physicsClientId=client_id,
    )


def wait_with_static_arm(
    duration: float,
    robot_id: int,
    client_id: int,
    camera_axis_debug_item_ids: list[int],
) -> bool:
    """Advance the fixed timestep while deliberately sending no arm commands."""
    for _ in range(round(duration / TIME_STEP)):
        if not p.isConnected(client_id):
            return False
        p.stepSimulation(physicsClientId=client_id)
        update_camera_reference_axes(
            robot_id,
            client_id,
            camera_axis_debug_item_ids,
        )
        time.sleep(TIME_STEP)
    return True


def move_arm_smoothly_to_ik_target(
    robot_id: int,
    point_name: str,
    target_position: Sequence[float],
    client_id: int,
    debug_text_id: int | None,
    camera_axis_debug_item_ids: list[int],
) -> tuple[list[int], list[float], int, float] | None:
    """Reach one future-Camera-origin target smoothly and hold it briefly."""
    arm_joint_indices, joint_targets = calculate_position_only_ik(
        robot_id,
        target_position,
        client_id,
    )
    start_positions = [
        p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        for joint_index in arm_joint_indices
    ]
    trajectory_duration = calculate_trajectory_duration(start_positions, joint_targets)
    print(f"\n=== Moving to {point_name} ===")
    print(f"Future Camera-origin target: {list(target_position)} m")
    print("Calculated q1...q7:", [round(position, 4) for position in joint_targets])
    print(f"Smooth trajectory duration: {trajectory_duration:.1f} s")
    debug_text_id = update_motion_debug_text(
        f"Moving to {point_name}",
        debug_text_id,
        client_id,
    )

    simulation_step = 0
    status_interval_steps = max(
        1,
        round(POSITION_STATUS_INTERVAL_SECONDS / TIME_STEP),
    )
    final_report_step = round(
        (trajectory_duration + POST_TRAJECTORY_SETTLE_SECONDS) / TIME_STEP
    )
    motion_complete_step = round(trajectory_duration / TIME_STEP)
    motion_duration_printed = False
    while p.isConnected(client_id):
        elapsed_time = simulation_step * TIME_STEP
        interpolation = _smoothstep(elapsed_time / trajectory_duration)
        waypoint_positions = [
            start + interpolation * (target - start)
            for start, target in zip(start_positions, joint_targets)
        ]
        # This is the only motor command for this motion.  It contains only
        # the seven Panda revolute arm joints; no gripper joint is touched.
        apply_arm_position_control(
            robot_id,
            arm_joint_indices,
            waypoint_positions,
            client_id,
        )
        p.stepSimulation(physicsClientId=client_id)
        update_camera_reference_axes(
            robot_id,
            client_id,
            camera_axis_debug_item_ids,
        )
        simulation_step += 1

        if simulation_step % status_interval_steps == 0:
            print_position_verification(
                target_position,
                robot_id,
                client_id,
                label=(
                    f"{point_name} position status at "
                    f"{simulation_step * TIME_STEP:.1f} s:"
                ),
                point_name=point_name,
            )
        if not motion_duration_printed and simulation_step >= motion_complete_step:
            debug_text_id = update_motion_debug_text(
                f"Reached {point_name}",
                debug_text_id,
                client_id,
            )
            print(
                f"{point_name} actual motion time: "
                f"{motion_complete_step * TIME_STEP:.2f} s"
            )
            motion_duration_printed = True
        if simulation_step >= final_report_step:
            print_position_verification(
                target_position,
                robot_id,
                client_id,
                label=f"{point_name} Final Position Result:",
                point_name=point_name,
            )
            print(
                f"{point_name} held for {POST_TRAJECTORY_SETTLE_SECONDS:.1f} s; "
                "segment complete."
            )
            return (
                arm_joint_indices,
                joint_targets,
                debug_text_id,
                motion_complete_step * TIME_STEP,
            )
        time.sleep(TIME_STEP)

    return None


def run_position_ik_gui_example(
    target_sequence: Sequence[tuple[str, Sequence[float]]],
) -> None:
    """Move Panda through a sequence of position-only IK targets.

    No target orientation is passed to ``calculateInverseKinematics``.  The
    solver therefore computes only a position solution.  Its q1...q7 solution
    is converted into smooth intermediate waypoints, each sent in the fixed
    simulation loop to Panda's seven revolute arm joints using
    ``POSITION_CONTROL``.
    """
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI.")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.setGravity(0, 0, 0, physicsClientId=client_id)
        p.setTimeStep(TIME_STEP, physicsClientId=client_id)
        p.loadURDF("plane.urdf", physicsClientId=client_id)
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=client_id,
        )

        end_effector_name = _decode_name(
            p.getJointInfo(
                robot_id,
                END_EFFECTOR_LINK_INDEX,
                physicsClientId=client_id,
            )[12]
        )
        if end_effector_name != END_EFFECTOR_LINK_NAME:
            raise RuntimeError(
                "The configured end-effector link no longer matches the selected "
                f"E frame {END_EFFECTOR_LINK_NAME!r}; "
                f"loaded {end_effector_name!r} instead."
            )

        print_camera_reference_inspection(robot_id, client_id)
        print("Close the PyBullet GUI window to finish the example.")

        add_target_markers(target_sequence, client_id)
        camera_axis_debug_item_ids: list[int] = []
        update_camera_reference_axes(robot_id, client_id, camera_axis_debug_item_ids)
        debug_text_id = update_motion_debug_text(
            f"Starting in {STARTUP_PAUSE_SECONDS:.0f} seconds",
            None,
            client_id,
        )
        print(f"Holding still for {STARTUP_PAUSE_SECONDS:.1f} s before the first target.")
        if not wait_with_static_arm(
            STARTUP_PAUSE_SECONDS,
            robot_id,
            client_id,
            camera_axis_debug_item_ids,
        ):
            return

        final_arm_joint_indices: list[int] | None = None
        final_joint_targets: list[float] | None = None
        motion_durations: list[tuple[str, float]] = []
        for point_name, target_position in target_sequence:
            result = move_arm_smoothly_to_ik_target(
                robot_id,
                point_name,
                target_position,
                client_id,
                debug_text_id,
                camera_axis_debug_item_ids,
            )
            if result is None:
                return
            (
                final_arm_joint_indices,
                final_joint_targets,
                debug_text_id,
                motion_duration,
            ) = result
            motion_durations.append((point_name, motion_duration))

        if final_arm_joint_indices is None or final_joint_targets is None:
            raise RuntimeError("The IK target sequence must contain at least one point.")

        sequence_name = " -> ".join(point_name for point_name, _ in target_sequence)
        print(f"\n{sequence_name} sequence complete. Holding the final position.")
        print("Actual motion durations:")
        for point_name, motion_duration in motion_durations:
            print(f"  {point_name}: {motion_duration:.2f} s")
        while p.isConnected(client_id):
            apply_arm_position_control(
                robot_id,
                final_arm_joint_indices,
                final_joint_targets,
                client_id,
            )
            p.stepSimulation(physicsClientId=client_id)
            update_camera_reference_axes(
                robot_id,
                client_id,
                camera_axis_debug_item_ids,
            )
            time.sleep(TIME_STEP)
    finally:
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


def run_single_joint_gui_example() -> None:
    """Move only Panda's first revolute joint slowly in a visible GUI window.

    Gravity is disabled so the remaining, intentionally uncontrolled joints
    stay still instead of falling.  No motor-control command is sent to any
    joint other than ``CONTROLLED_JOINT_INDEX``.
    """
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI.")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.setGravity(0, 0, 0, physicsClientId=client_id)
        p.setTimeStep(TIME_STEP, physicsClientId=client_id)
        p.loadURDF("plane.urdf", physicsClientId=client_id)
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=client_id,
        )

        joint_info = p.getJointInfo(
            robot_id,
            CONTROLLED_JOINT_INDEX,
            physicsClientId=client_id,
        )
        joint_name = _decode_name(joint_info[1])
        if joint_info[2] != p.JOINT_REVOLUTE:
            raise RuntimeError(
                f"Joint {CONTROLLED_JOINT_INDEX} ({joint_name}) is not revolute."
            )

        lower_limit, upper_limit = joint_info[8], joint_info[9]
        target_position = max(lower_limit, min(SAFE_TARGET_POSITION, upper_limit))
        print(
            f"Controlling only joint {CONTROLLED_JOINT_INDEX} ({joint_name}) "
            f"toward {target_position:.2f} rad at up to {MAX_JOINT_VELOCITY:.2f} rad/s."
        )
        print("Close the PyBullet GUI window to finish the example.")

        p.setJointMotorControl2(
            robot_id,
            CONTROLLED_JOINT_INDEX,
            p.POSITION_CONTROL,
            targetPosition=target_position,
            force=MAX_MOTOR_FORCE,
            positionGain=POSITION_GAIN,
            maxVelocity=MAX_JOINT_VELOCITY,
            physicsClientId=client_id,
        )

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(TIME_STEP)
    finally:
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print Panda's joint/link structure in headless DIRECT mode.",
    )
    parser.add_argument(
        "--single-joint",
        action="store_true",
        help="Run the earlier single-joint GUI motion example instead of IK.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Run one custom future-Camera-origin XYZ target in metres.",
    )
    arguments = parser.parse_args()

    if arguments.inspect:
        inspect_panda_structure()
    elif arguments.single_joint:
        run_single_joint_gui_example()
    else:
        target_sequence = (
            (("Custom Camera Origin", arguments.target),)
            if arguments.target is not None
            else DEFAULT_CAMERA_REFERENCE_TARGET_SEQUENCE
        )
        run_position_ik_gui_example(target_sequence)
