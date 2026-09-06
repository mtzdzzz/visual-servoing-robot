"""Inspect Panda or run the RGB-detection hold loop in PyBullet.

The default loop preserves the accepted Stage 3.1 locked-joint baseline and
fixed hand-eye camera. It performs Stage 5 image detection but deliberately
does not implement visual-servo motion, PID, ROS, Gazebo, or YOLO.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import acos, degrees, radians, sqrt
import time
from typing import Sequence

import pybullet as p
import pybullet_data

from camera_observation import (
    CAMERA_OFFSET_DEBUG_OUTPUT_PATH,
    STAGE4_FINAL_RGB_OUTPUT_PATH,
    STAGE4_SECOND_POSE_RGB_OUTPUT_PATH,
    EyeInHandRgbDisplay,
    add_camera_diagnostic_debug_lines,
    capture_eye_in_hand_rgb,
    capture_forced_look_at_rgb,
    create_red_ground_target,
    detect_red_target_from_live_rgb,
    print_camera_coordinate_definition,
    print_camera_observation,
    print_camera_render_diagnostics,
    print_camera_target_alignment,
    render_live_eye_in_hand_rgb_frame,
)


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
CAMERA_DISPLAY_FREQUENCY_HZ = 20.0
CAMERA_UPDATE_INTERVAL_STEPS = max(
    1,
    round(1.0 / (TIME_STEP * CAMERA_DISPLAY_FREQUENCY_HZ)),
)
TARGET_DETECTION_PRINT_INTERVAL_STEPS = max(1, round(1.0 / TIME_STEP))

# Verified from the loaded franka_panda/panda.urdf: index 7 is panda_link8.
# E is the Panda's fixed wrist/flange link, before panda_hand.  It is the
# stable mechanical reference to which a future eye-in-hand camera is mounted.
END_EFFECTOR_LINK_INDEX = 7
END_EFFECTOR_LINK_NAME = "panda_link8"
CAMERA_REFERENCE_FRAME_NAME = "C"
# Stage 3.1's position-IK reference remains at E exactly as validated.  It is
# deliberately separate from the physical camera optical center below, so the
# camera mounting investigation cannot change the accepted robot trajectory.
T_E_IK_REFERENCE_POSITION = (0.0, 0.0, 0.0)
T_E_IK_REFERENCE_ORIENTATION = (0.0, 0.0, 0.0, 1.0)

# Physical Eye-in-Hand camera mounting transform. Panda's hand and fingers
# extend along E's local +Z axis, while E is within the hand collision
# geometry. Candidate checks at 0.03/0.05/0.07 m along +Y_E selected 0.07 m:
# it is the nearest requested offset with positive clearance at every tested
# pose. This is a local E-frame translation, not a world-axis assumption.
CAMERA_TRANSLATION_CANDIDATES_METRES = (0.03, 0.05, 0.07)
CAMERA_TRANSLATION_DIRECTION_E = (0.0, 1.0, 0.0)
CAMERA_SELECTED_TRANSLATION_METRES = 0.07
T_E_C_POSITION = tuple(
    component * CAMERA_SELECTED_TRANSLATION_METRES
    for component in CAMERA_TRANSLATION_DIRECTION_E
)
# This manually selected, fixed hand-eye rotation faces the ground work area
# from the startup pose and remains unchanged at every later pose.  It is not
# calculated from the red target position and no look-at operation is used.
T_E_C_ROTATION_RPY_DEGREES = (0.0, -15.0, -20.0)
T_E_C_ORIENTATION = p.getQuaternionFromEuler(
    tuple(radians(angle) for angle in T_E_C_ROTATION_RPY_DEGREES)
)
CAMERA_CLEARANCE_PROBE_RADIUS = 0.01  # metres; diagnostic only, not rendered

# Archived Stage 3/4 Cartesian test targets.  They remain defined only for
# the explicitly disabled legacy demo below and are never consulted by the
# default RGB-detection startup path.
SAFE_CAMERA_ORIGIN_TARGET_POSITION = (0.09, -0.09, 1.08)
DEFAULT_CAMERA_REFERENCE_TARGET_SEQUENCE = (
    ("Camera Origin Test", SAFE_CAMERA_ORIGIN_TARGET_POSITION),
)
SECONDARY_CAMERA_REFERENCE_TARGET = (
    "Small Camera-Follow Check",
    (0.092, -0.092, 1.078),
)
LEGACY_FIXED_MOTION_ENABLED = False
VISUAL_SERVO_MOTION_ENABLED = False
# This is the accepted Stage 3.1 baseline.  The values are recorded from the
# loaded Panda at startup and applied on every control step; they are not IK
# targets.
BASELINE_LOCKED_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
)
LOCKING_STRATEGIES = (
    (
        "A",
        (
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
        ),
    ),
)
LOCKED_JOINT_DEVIATION_LIMIT = 0.005  # radians
CAMERA_TARGET_ERROR_LIMIT = 0.01  # metres
CONSTRAINED_IK_MAX_ITERATIONS = 250
CONSTRAINED_IK_POSITION_TOLERANCE = 0.001  # metres
CONSTRAINED_IK_JACOBIAN_DELTA = 1e-4  # radians
CONSTRAINED_IK_MAX_STEP = 0.08  # radians per numerical iteration
CONSTRAINED_IK_DAMPING = 0.002
MAX_IK_JOINT_VELOCITY = 0.20  # radians/second
MAX_IK_MOTOR_FORCE = 20.0
IK_VELOCITY_GAIN = 0.8
TRAJECTORY_MAX_WAYPOINT_SPEED = 0.18  # radians/second
# Both the startup hold and the IK-motion phase report their lock error at the
# same cadence.  Keeping this independent of the 240 Hz physics rate prevents
# terminal spam while still exposing a control overwrite immediately.
CONTROL_STATUS_INTERVAL_SECONDS = 0.5
MIN_TRAJECTORY_DURATION = 5.0  # seconds; Stage 3 visual-motion minimum
POST_TRAJECTORY_SETTLE_SECONDS = 3.0
STARTUP_PAUSE_SECONDS = 3.0
DEBUG_STATUS_TEXT_POSITION = (0.0, 0.0, 1.35)
DEBUG_TARGET_MARKER_SIZE = 0.045
CAMERA_AXIS_LENGTH = 0.10


@dataclass(frozen=True)
class LockingStrategyEvaluation:
    """Measured result of one locked-joint motion strategy."""

    name: str
    locked_joint_names: tuple[str, ...]
    active_joint_names: tuple[str, ...]
    final_camera_error: float
    max_locked_deviation: float
    motion_stable: bool
    reachable: bool


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


def get_panda_arm_joint_names(robot_id: int, client_id: int) -> dict[str, int]:
    """Read Panda's seven revolute joint names and indices from the URDF."""
    joint_names = {}
    for joint_index in get_panda_arm_joint_indices(robot_id, client_id):
        joint_info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        joint_names[_decode_name(joint_info[1])] = joint_index
    return joint_names


def print_panda_arm_joint_configuration(robot_id: int, client_id: int) -> dict[int, float]:
    """Print all arm-joint names, limits, and recorded initial positions."""
    initial_positions: dict[int, float] = {}
    print("\nPanda revolute arm-joint inspection:")
    for joint_index in get_panda_arm_joint_indices(robot_id, client_id):
        joint_info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        initial_position = p.getJointState(
            robot_id,
            joint_index,
            physicsClientId=client_id,
        )[0]
        initial_positions[joint_index] = initial_position
        print(
            f"  index={joint_index}, name={_decode_name(joint_info[1])}, "
            f"type=revolute, limits=({joint_info[8]:.4f}, {joint_info[9]:.4f}), "
            f"initial={initial_position:.4f} rad"
        )
    return initial_positions


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
    """Return the unchanged Stage 3.1 position-IK reference at E.

    This function remains the reference used by the accepted constrained-IK
    controller.  The physical optical center C is intentionally calculated by
    ``get_camera_optical_center_pose`` so camera mounting work cannot alter
    the robot control baseline.
    """
    end_effector_position, end_effector_orientation = get_end_effector_reference_pose(
        robot_id,
        client_id,
    )
    ik_reference_position, ik_reference_orientation = p.multiplyTransforms(
        end_effector_position,
        end_effector_orientation,
        T_E_IK_REFERENCE_POSITION,
        T_E_IK_REFERENCE_ORIENTATION,
    )
    return list(ik_reference_position), list(ik_reference_orientation)


def get_camera_optical_center_pose(
    robot_id: int,
    client_id: int,
) -> tuple[list[float], list[float]]:
    """Compose the physical camera optical-center C pose from E and T_E_C."""
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


def get_measured_hand_eye_transform(
    robot_id: int,
    client_id: int,
) -> tuple[list[float], list[float]]:
    """Recover T_E_C from current world poses for a rigid-mount audit."""
    end_effector_position, end_effector_orientation = get_end_effector_reference_pose(
        robot_id,
        client_id,
    )
    camera_position, camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    inverse_end_effector_position, inverse_end_effector_orientation = p.invertTransform(
        end_effector_position,
        end_effector_orientation,
    )
    measured_translation, measured_orientation = p.multiplyTransforms(
        inverse_end_effector_position,
        inverse_end_effector_orientation,
        camera_position,
        camera_orientation,
    )
    return list(measured_translation), list(measured_orientation)


def _quaternion_distance_degrees(
    first_orientation: Sequence[float],
    second_orientation: Sequence[float],
) -> float:
    """Return the smallest angle between two unit quaternions in degrees."""
    absolute_dot = abs(
        sum(first * second for first, second in zip(first_orientation, second_orientation))
    )
    return degrees(2.0 * acos(max(-1.0, min(1.0, absolute_dot))))


def print_hand_eye_transform_validation(
    label: str,
    measured_transform: tuple[Sequence[float], Sequence[float]],
    comparison_transform: tuple[Sequence[float], Sequence[float]] | None = None,
) -> None:
    """Print the fixed mount transform and, optionally, a second-pose delta."""
    measured_translation, measured_orientation = measured_transform
    translation_error = sqrt(
        sum(
            (measured - configured) ** 2
            for measured, configured in zip(measured_translation, T_E_C_POSITION)
        )
    )
    rotation_error = _quaternion_distance_degrees(
        measured_orientation,
        T_E_C_ORIENTATION,
    )
    print(f"\n{label} hand-eye transform validation:")
    print("  measured T_E_C translation:", [round(value, 6) for value in measured_translation])
    print("  measured T_E_C rotation xyzw:", [round(value, 6) for value in measured_orientation])
    print(f"  configured translation error: {translation_error:.8f} m")
    print(f"  configured rotation error: {rotation_error:.8f} deg")
    if comparison_transform is not None:
        comparison_translation, comparison_orientation = comparison_transform
        pose_translation_delta = sqrt(
            sum(
                (measured - comparison) ** 2
                for measured, comparison in zip(measured_translation, comparison_translation)
            )
        )
        pose_rotation_delta = _quaternion_distance_degrees(
            measured_orientation,
            comparison_orientation,
        )
        print(f"  two-pose T_E_C translation delta: {pose_translation_delta:.8f} m")
        print(f"  two-pose T_E_C rotation delta: {pose_rotation_delta:.8f} deg")


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
    ik_reference_position, _ = get_camera_reference_pose(robot_id, client_id)
    camera_position, camera_orientation = get_camera_optical_center_pose(robot_id, client_id)
    print(
        "Stage 3.1 IK reference remains at E "
        f"(translation={list(T_E_IK_REFERENCE_POSITION)}):",
        [round(value, 4) for value in ik_reference_position],
    )
    print(
        "Physical camera optical center C: "
        f"T_E_C translation={list(T_E_C_POSITION)}, "
        "rotation RPY degrees="
        f"{list(T_E_C_ROTATION_RPY_DEGREES)}, "
        f"orientation xyzw={list(T_E_C_ORIENTATION)}"
    )
    print("  C world position:", [round(value, 4) for value in camera_position])
    print("  C world orientation xyzw:", [round(value, 4) for value in camera_orientation])


def _link_name_for_debug(robot_id: int, link_index: int, client_id: int) -> str:
    """Return a readable link name, including PyBullet's base pseudo-link."""
    if link_index == -1:
        return "base"
    return _decode_name(
        p.getJointInfo(robot_id, link_index, physicsClientId=client_id)[12]
    )


def print_camera_mount_geometry_inspection(
    robot_id: int,
    target_world_position: Sequence[float],
    client_id: int,
) -> None:
    """Inspect E/C against the loaded wrist and hand geometry without motors.

    A 1 cm collision-only sphere is momentarily placed at C to measure signed
    clearance from the actual Panda collision meshes.  It is removed before
    returning and never participates in physics stepping or motor control.
    """
    end_effector_position, _ = get_end_effector_reference_pose(robot_id, client_id)
    camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
    hand_link_index = get_link_index_by_name(robot_id, "panda_hand", client_id)

    print("\nCamera mount / Panda geometry inspection:")
    print("E world position:", [round(value, 6) for value in end_effector_position])
    print("C world position:", [round(value, 6) for value in camera_position])
    print("T_E_C translation:", [round(value, 6) for value in T_E_C_POSITION])
    print(
        "distance(E, C): "
        f"{sqrt(sum((camera - end) ** 2 for camera, end in zip(camera_position, end_effector_position))):.6f} m"
    )
    print(
        "camera-to-target distance: "
        f"{sqrt(sum((target - camera) ** 2 for target, camera in zip(target_world_position, camera_position))):.6f} m"
    )

    for link_index in (END_EFFECTOR_LINK_INDEX, hand_link_index):
        link_name = _link_name_for_debug(robot_id, link_index, client_id)
        aabb_min, aabb_max = p.getAABB(
            robot_id,
            link_index,
            physicsClientId=client_id,
        )
        print(
            f"  {link_name} (link {link_index}) world collision AABB:",
            "min=",
            [round(value, 4) for value in aabb_min],
            "max=",
            [round(value, 4) for value in aabb_max],
        )
        visual_shapes = [
            shape
            for shape in p.getVisualShapeData(robot_id, physicsClientId=client_id)
            if shape[1] == link_index
        ]
        collision_shapes = p.getCollisionShapeData(
            robot_id,
            link_index,
            physicsClientId=client_id,
        )
        print(
            f"    visual shapes={len(visual_shapes)}, collision shapes={len(collision_shapes)}"
        )
        for collision_shape in collision_shapes:
            print(
                "    collision local position:",
                [round(value, 4) for value in collision_shape[5]],
                "asset:",
                _decode_name(collision_shape[4]),
            )

    probe_shape_id = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=CAMERA_CLEARANCE_PROBE_RADIUS,
        physicsClientId=client_id,
    )
    probe_body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=probe_shape_id,
        basePosition=camera_position,
        physicsClientId=client_id,
    )
    try:
        closest_points = p.getClosestPoints(
            robot_id,
            probe_body_id,
            distance=0.5,
            physicsClientId=client_id,
        )
        if not closest_points:
            print("  C clearance probe: no Panda geometry found within 0.5 m.")
            return
        nearest_point = min(closest_points, key=lambda point: point[8])
        signed_surface_distance = nearest_point[8]
        nearest_link_index = nearest_point[3]
        print(
            "  C 1-cm probe signed clearance to Panda: "
            f"{signed_surface_distance:.6f} m "
            f"(nearest {_link_name_for_debug(robot_id, nearest_link_index, client_id)})."
        )
        print(
            "  C outside Panda collision geometry: "
            f"{signed_surface_distance >= 0.0}"
        )
    finally:
        p.removeBody(probe_body_id, physicsClientId=client_id)


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


def _invert_3x3(matrix: list[list[float]]) -> list[list[float]] | None:
    """Return a 3×3 inverse without adding a numerical-library dependency."""
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant) < 1e-12:
        return None
    return [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2])
            / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])
            / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2])
            / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0])
            / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2])
            / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
            / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1])
            / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
            / determinant,
        ],
    ]


def _set_arm_joint_positions(
    robot_id: int,
    arm_joint_indices: Sequence[int],
    joint_positions: Sequence[float],
    client_id: int,
) -> None:
    for joint_index, joint_position in zip(arm_joint_indices, joint_positions):
        p.resetJointState(
            robot_id,
            joint_index,
            joint_position,
            physicsClientId=client_id,
        )


def calculate_constrained_position_ik(
    robot_id: int,
    target_position: Sequence[float],
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> tuple[list[int], list[float], float]:
    """Solve position IK while allowing numerical updates only on active joints.

    PyBullet's ``calculateInverseKinematics`` still provides the nominal full
    solution, but it is never executed as-is.  A damped finite-difference
    position-IK refinement holds every locked joint at its recorded initial
    angle and computes targets solely for the active joints.
    """
    arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
    locked_joint_indices = list(locked_joint_indices)
    active_joint_indices = [
        joint_index for joint_index in arm_joint_indices if joint_index not in locked_joint_indices
    ]
    if not active_joint_indices:
        raise ValueError("At least one Panda arm joint must remain active for IK.")

    saved_positions = [
        p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        for joint_index in arm_joint_indices
    ]
    # Required nominal PyBullet IK result.  It is intentionally not sent to
    # motors, because it may move the joints selected for locking.
    calculate_position_only_ik(robot_id, target_position, client_id)

    joint_limits = {
        joint_index: (
            p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[8],
            p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[9],
        )
        for joint_index in arm_joint_indices
    }
    candidate_positions = dict(zip(arm_joint_indices, saved_positions))
    candidate_positions.update(locked_initial_positions)

    final_error = float("inf")
    try:
        for _ in range(CONSTRAINED_IK_MAX_ITERATIONS):
            _set_arm_joint_positions(
                robot_id,
                arm_joint_indices,
                [candidate_positions[index] for index in arm_joint_indices],
                client_id,
            )
            actual_position = get_end_effector_position(robot_id, client_id)
            position_error = [
                target - actual for target, actual in zip(target_position, actual_position)
            ]
            final_error = sqrt(sum(component * component for component in position_error))
            if final_error <= CONSTRAINED_IK_POSITION_TOLERANCE:
                break

            jacobian = [[0.0 for _ in active_joint_indices] for _ in range(3)]
            for column, joint_index in enumerate(active_joint_indices):
                candidate_positions[joint_index] += CONSTRAINED_IK_JACOBIAN_DELTA
                _set_arm_joint_positions(
                    robot_id,
                    arm_joint_indices,
                    [candidate_positions[index] for index in arm_joint_indices],
                    client_id,
                )
                perturbed_position = get_end_effector_position(robot_id, client_id)
                candidate_positions[joint_index] -= CONSTRAINED_IK_JACOBIAN_DELTA
                for row in range(3):
                    jacobian[row][column] = (
                        perturbed_position[row] - actual_position[row]
                    ) / CONSTRAINED_IK_JACOBIAN_DELTA

            normal_matrix = [
                [
                    sum(
                        jacobian[row][column] * jacobian[other_row][column]
                        for column in range(len(active_joint_indices))
                    )
                    + (CONSTRAINED_IK_DAMPING if row == other_row else 0.0)
                    for other_row in range(3)
                ]
                for row in range(3)
            ]
            inverse_normal_matrix = _invert_3x3(normal_matrix)
            if inverse_normal_matrix is None:
                break
            damped_error = [
                sum(inverse_normal_matrix[row][column] * position_error[column] for column in range(3))
                for row in range(3)
            ]
            joint_updates = [
                sum(jacobian[row][column] * damped_error[row] for row in range(3))
                for column in range(len(active_joint_indices))
            ]
            largest_update = max((abs(update) for update in joint_updates), default=0.0)
            update_scale = min(
                1.0,
                CONSTRAINED_IK_MAX_STEP / max(largest_update, 1e-12),
            )
            for joint_index, joint_update in zip(active_joint_indices, joint_updates):
                lower_limit, upper_limit = joint_limits[joint_index]
                candidate_positions[joint_index] = max(
                    lower_limit,
                    min(
                        upper_limit,
                        candidate_positions[joint_index] + joint_update * update_scale,
                    ),
                )
            for joint_index in locked_joint_indices:
                candidate_positions[joint_index] = locked_initial_positions[joint_index]
    finally:
        _set_arm_joint_positions(robot_id, arm_joint_indices, saved_positions, client_id)

    return (
        arm_joint_indices,
        [candidate_positions[index] for index in arm_joint_indices],
        final_error,
    )


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


def apply_constrained_arm_position_control(
    robot_id: int,
    arm_joint_indices: Sequence[int],
    waypoint_positions: Sequence[float],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> None:
    """Control all seven arm joints while forcing locked ones to their initial q."""
    if len(arm_joint_indices) != 7 or len(waypoint_positions) != 7:
        raise ValueError("Constrained POSITION_CONTROL requires seven Panda arm targets.")
    constrained_waypoints = [
        locked_initial_positions.get(joint_index, waypoint_position)
        for joint_index, waypoint_position in zip(arm_joint_indices, waypoint_positions)
    ]
    apply_arm_position_control(
        robot_id,
        arm_joint_indices,
        constrained_waypoints,
        client_id,
    )


def get_locked_joint_deviations(
    robot_id: int,
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> dict[int, float]:
    """Measure the signed locked-joint deviations from their recorded starts."""
    return {
        joint_index: p.getJointState(
            robot_id,
            joint_index,
            physicsClientId=client_id,
        )[0]
        - initial_position
        for joint_index, initial_position in locked_initial_positions.items()
    }


def print_locked_joint_status(
    phase: str,
    robot_id: int,
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> float:
    """Print each locked joint's measured state and immutable target."""
    if phase not in {"HOLD", "MOVE"}:
        raise ValueError("phase must be either 'HOLD' or 'MOVE'.")
    joint_names = {
        joint_index: _decode_name(
            p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[1]
        )
        for joint_index in locked_initial_positions
    }
    deviations = get_locked_joint_deviations(
        robot_id,
        locked_initial_positions,
        client_id,
    )
    print(f"phase = {phase}")
    for joint_index in sorted(deviations):
        locked_target = locked_initial_positions[joint_index]
        current_position = locked_target + deviations[joint_index]
        print(
            f"  {joint_names[joint_index]}: "
            f"current angle = {current_position:.6f} rad, "
            f"locked target angle = {locked_target:.6f} rad, "
            f"deviation = {deviations[joint_index]:.6f} rad"
        )
    return max((abs(deviation) for deviation in deviations.values()), default=0.0)


def evaluate_locking_strategy(
    strategy_name: str,
    locked_joint_names: tuple[str, ...],
    target_position: Sequence[float],
) -> LockingStrategyEvaluation:
    """Run one strategy in DIRECT mode and return measured feasibility metrics."""
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Unable to create a DIRECT client for strategy evaluation.")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.setGravity(0, 0, 0, physicsClientId=client_id)
        p.setTimeStep(TIME_STEP, physicsClientId=client_id)
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=client_id,
        )
        arm_joint_names = get_panda_arm_joint_names(robot_id, client_id)
        locked_joint_indices = [arm_joint_names[name] for name in locked_joint_names]
        arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
        active_joint_names = tuple(
            name for name, index in arm_joint_names.items() if index not in locked_joint_indices
        )
        locked_initial_positions = {
            joint_index: p.getJointState(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )[0]
            for joint_index in locked_joint_indices
        }
        _, joint_targets, _ = calculate_constrained_position_ik(
            robot_id,
            target_position,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )
        start_positions = [
            p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
            for joint_index in arm_joint_indices
        ]
        trajectory_duration = calculate_trajectory_duration(start_positions, joint_targets)
        total_steps = round(
            (trajectory_duration + POST_TRAJECTORY_SETTLE_SECONDS) / TIME_STEP
        )
        motion_steps = round(trajectory_duration / TIME_STEP)
        max_locked_deviation = 0.0
        settled_errors = []
        for simulation_step in range(total_steps):
            interpolation = _smoothstep(simulation_step * TIME_STEP / trajectory_duration)
            waypoint_positions = [
                start + interpolation * (target - start)
                for start, target in zip(start_positions, joint_targets)
            ]
            apply_constrained_arm_position_control(
                robot_id,
                arm_joint_indices,
                waypoint_positions,
                locked_initial_positions,
                client_id,
            )
            p.stepSimulation(physicsClientId=client_id)
            deviations = get_locked_joint_deviations(
                robot_id,
                locked_initial_positions,
                client_id,
            )
            max_locked_deviation = max(
                max_locked_deviation,
                max((abs(deviation) for deviation in deviations.values()), default=0.0),
            )
            if simulation_step >= motion_steps:
                actual_position = get_end_effector_position(robot_id, client_id)
                settled_errors.append(
                    sqrt(
                        sum(
                            (target - actual) ** 2
                            for target, actual in zip(target_position, actual_position)
                        )
                    )
                )

        final_camera_error = settled_errors[-1]
        settling_variation = max(settled_errors) - min(settled_errors)
        motion_stable = (
            settling_variation < 0.003
            and max_locked_deviation < LOCKED_JOINT_DEVIATION_LIMIT
        )
        reachable = (
            final_camera_error < CAMERA_TARGET_ERROR_LIMIT
            and max_locked_deviation < LOCKED_JOINT_DEVIATION_LIMIT
        )
        return LockingStrategyEvaluation(
            name=strategy_name,
            locked_joint_names=locked_joint_names,
            active_joint_names=active_joint_names,
            final_camera_error=final_camera_error,
            max_locked_deviation=max_locked_deviation,
            motion_stable=motion_stable,
            reachable=reachable,
        )
    finally:
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


def print_locking_strategy_comparison(
    evaluations: Sequence[LockingStrategyEvaluation],
) -> None:
    """Print the requested A/B comparison before choosing a GUI strategy."""
    print("\nConstrained IK strategy comparison:")
    for evaluation in evaluations:
        print(f"Strategy {evaluation.name}:")
        print("  locked joints:", list(evaluation.locked_joint_names))
        print("  active joints:", list(evaluation.active_joint_names))
        print(f"  final camera position error: {evaluation.final_camera_error:.4f} m")
        print(f"  locked joint max deviation: {evaluation.max_locked_deviation:.6f} rad")
        print(f"  motion stable: {'yes' if evaluation.motion_stable else 'no'}")
        print(f"  reachable: {'yes' if evaluation.reachable else 'no'}")


def choose_locking_strategy(
    evaluations: Sequence[LockingStrategyEvaluation],
) -> LockingStrategyEvaluation | None:
    """Prefer the most restrictive stable strategy that still reaches C accurately."""
    for evaluation in evaluations:
        if evaluation.reachable and evaluation.motion_stable:
            return evaluation
    return None


def get_end_effector_position(robot_id: int, client_id: int) -> list[float]:
    """Read the unchanged Stage 3.1 IK reference world position at E."""
    ik_reference_position, _ = get_camera_reference_pose(robot_id, client_id)
    return ik_reference_position


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
    """Draw distinct E and physical-camera-C frames plus their mount offset.

    This is GUI-only visualization.  The existing callers execute it during
    HOLD and MOVE, but this function never sends an IK or motor command.
    """
    end_effector_position, end_effector_orientation = get_end_effector_reference_pose(
        robot_id,
        client_id,
    )
    camera_position, camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    axis_definitions = (
        ((CAMERA_AXIS_LENGTH, 0.0, 0.0), [1.0, 0.0, 0.0]),  # X: red
        ((0.0, CAMERA_AXIS_LENGTH, 0.0), [0.0, 1.0, 0.0]),  # Y: green
        ((0.0, 0.0, CAMERA_AXIS_LENGTH), [0.0, 0.0, 1.0]),  # Z: blue
    )
    new_debug_item_ids: list[int] = []

    def replacement_id(index: int) -> int:
        return (
            debug_item_ids[index]
            if index < len(debug_item_ids) and debug_item_ids[index] >= 0
            else -1
        )

    # Physical camera optical-center C axes.
    for axis_index, (axis_end_in_camera, color) in enumerate(axis_definitions):
        axis_end_world, _ = p.multiplyTransforms(
            camera_position,
            camera_orientation,
            axis_end_in_camera,
            (0.0, 0.0, 0.0, 1.0),
        )
        new_debug_item_ids.append(
            p.addUserDebugLine(
                camera_position,
                axis_end_world,
                lineColorRGB=color,
                lineWidth=4,
                lifeTime=0,
                replaceItemUniqueId=replacement_id(axis_index),
                physicsClientId=client_id,
            )
        )

    label_position, _ = p.multiplyTransforms(
        camera_position,
        camera_orientation,
        (0.0, 0.0, CAMERA_AXIS_LENGTH * 1.25),
        (0.0, 0.0, 0.0, 1.0),
    )
    new_debug_item_ids.append(
        p.addUserDebugText(
            f"Camera optical center {CAMERA_REFERENCE_FRAME_NAME}",
            label_position,
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=0,
            replaceItemUniqueId=replacement_id(3),
            physicsClientId=client_id,
        )
    )

    # E remains marked separately at its original panda_link8 frame origin.
    for axis_index, (axis_end_in_end_effector, color) in enumerate(axis_definitions):
        axis_end_world, _ = p.multiplyTransforms(
            end_effector_position,
            end_effector_orientation,
            tuple(component * 0.65 for component in axis_end_in_end_effector),
            (0.0, 0.0, 0.0, 1.0),
        )
        new_debug_item_ids.append(
            p.addUserDebugLine(
                end_effector_position,
                axis_end_world,
                lineColorRGB=color,
                lineWidth=3,
                lifeTime=0,
                replaceItemUniqueId=replacement_id(4 + axis_index),
                physicsClientId=client_id,
            )
        )
    end_effector_label_position, _ = p.multiplyTransforms(
        end_effector_position,
        end_effector_orientation,
        (0.0, 0.0, CAMERA_AXIS_LENGTH * 0.8),
        (0.0, 0.0, 0.0, 1.0),
    )
    new_debug_item_ids.append(
        p.addUserDebugText(
            f"E: {END_EFFECTOR_LINK_NAME}",
            end_effector_label_position,
            textColorRGB=[1.0, 0.2, 1.0],
            textSize=1.2,
            lifeTime=0,
            replaceItemUniqueId=replacement_id(7),
            physicsClientId=client_id,
        )
    )
    new_debug_item_ids.append(
        p.addUserDebugLine(
            end_effector_position,
            camera_position,
            lineColorRGB=[1.0, 0.2, 1.0],
            lineWidth=4,
            lifeTime=0,
            replaceItemUniqueId=replacement_id(8),
            physicsClientId=client_id,
        )
    )
    debug_item_ids[:] = new_debug_item_ids


def update_camera_translation_candidate_markers(
    robot_id: int,
    client_id: int,
    debug_item_ids: list[int],
) -> None:
    """Mark the requested local +Y_E camera-offset candidates in the GUI.

    E itself remains visible through ``update_camera_reference_axes``.  Each
    candidate C is labelled, given a small C-frame cross, and connected to E;
    all positions are recomputed from the latest E pose every physics step.
    """
    end_effector_position, end_effector_orientation = get_end_effector_reference_pose(
        robot_id,
        client_id,
    )
    candidate_colours = (
        [1.0, 0.45, 0.0],  # 0.03 m: orange
        [0.0, 1.0, 1.0],  # 0.05 m: cyan
        [1.0, 1.0, 0.0],  # 0.07 m selected: yellow
    )
    new_debug_item_ids: list[int] = []

    def replacement_id(index: int) -> int:
        return (
            debug_item_ids[index]
            if index < len(debug_item_ids) and debug_item_ids[index] >= 0
            else -1
        )

    for candidate_index, (offset, colour) in enumerate(
        zip(CAMERA_TRANSLATION_CANDIDATES_METRES, candidate_colours)
    ):
        camera_position, camera_orientation = p.multiplyTransforms(
            end_effector_position,
            end_effector_orientation,
            tuple(component * offset for component in CAMERA_TRANSLATION_DIRECTION_E),
            T_E_C_ORIENTATION,
        )
        marker_start, _ = p.multiplyTransforms(
            camera_position,
            camera_orientation,
            (-0.018, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        marker_end, _ = p.multiplyTransforms(
            camera_position,
            camera_orientation,
            (0.018, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        label_position, _ = p.multiplyTransforms(
            camera_position,
            camera_orientation,
            (0.0, 0.0, 0.03),
            (0.0, 0.0, 0.0, 1.0),
        )
        is_selected = abs(offset - CAMERA_SELECTED_TRANSLATION_METRES) < 1e-9
        label = f"C offset {offset:.2f} m" + (" (selected)" if is_selected else "")
        base_index = candidate_index * 3
        new_debug_item_ids.append(
            p.addUserDebugLine(
                end_effector_position,
                camera_position,
                lineColorRGB=colour,
                lineWidth=2,
                lifeTime=0,
                replaceItemUniqueId=replacement_id(base_index),
                physicsClientId=client_id,
            )
        )
        new_debug_item_ids.append(
            p.addUserDebugLine(
                marker_start,
                marker_end,
                lineColorRGB=colour,
                lineWidth=5,
                lifeTime=0,
                replaceItemUniqueId=replacement_id(base_index + 1),
                physicsClientId=client_id,
            )
        )
        new_debug_item_ids.append(
            p.addUserDebugText(
                label,
                label_position,
                textColorRGB=colour,
                textSize=1.0,
                lifeTime=0,
                replaceItemUniqueId=replacement_id(base_index + 2),
                physicsClientId=client_id,
            )
        )
    debug_item_ids[:] = new_debug_item_ids


def update_live_eye_in_hand_camera(
    robot_id: int,
    simulation_step: int,
    client_id: int,
    camera_display: EyeInHandRgbDisplay,
) -> None:
    """Refresh the fixed hand-eye RGB camera from the latest E pose.

    This function is called after every physics step in both HOLD and MOVE.
    It always reads ``getLinkState`` through ``get_camera_optical_center_pose``
    to obtain current T_W_E(t) and composes T_W_C(t) = T_W_E(t) @ T_E_C.
    Rendering and display are intentionally rate-limited to 20 Hz so the
    240 Hz simulation/control loop remains stable.
    """
    camera_position, camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    if simulation_step % CAMERA_UPDATE_INTERVAL_STEPS != 0:
        return
    live_frame = render_live_eye_in_hand_rgb_frame(
        camera_position,
        camera_orientation,
        client_id,
    )
    red_detection = detect_red_target_from_live_rgb(live_frame)
    camera_display.show(live_frame, red_detection.annotated_rgba_buffer)
    if simulation_step % TARGET_DETECTION_PRINT_INTERVAL_STEPS == 0:
        if red_detection.detected:
            assert red_detection.centroid is not None
            assert red_detection.pixel_error is not None
            assert red_detection.bounding_box is not None
            assert red_detection.contour_area is not None
            print(
                "Red target detection: "
                f"target pixel = {red_detection.centroid}; "
                f"image center = {red_detection.image_center}; "
                f"ex = {red_detection.pixel_error[0]}, "
                f"ey = {red_detection.pixel_error[1]}; "
                f"bbox = {red_detection.bounding_box}; "
                f"area = {red_detection.contour_area:.1f} px"
            )
        else:
            print(
                "Red target detection: Target not detected; "
                f"image center = {red_detection.image_center}"
            )


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


def enable_pybullet_camera_debug_previews(client_id: int) -> None:
    """Show PyBullet's diagnostic previews for the live camera render stream.

    They are GUI diagnostics only. The Stage 5 detector still receives only
    the RGBA RGB frame returned by the Eye-in-Hand getCameraImage call.
    """
    for preview_flag in (
        p.COV_ENABLE_RGB_BUFFER_PREVIEW,
        p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
    ):
        p.configureDebugVisualizer(
            preview_flag,
            1,
            physicsClientId=client_id,
        )


def wait_with_static_arm(
    duration: float,
    robot_id: int,
    arm_joint_indices: Sequence[int],
    initial_arm_positions: dict[int, float],
    locked_initial_positions: dict[int, float],
    client_id: int,
    camera_axis_debug_item_ids: list[int],
    camera_offset_candidate_debug_item_ids: list[int],
    camera_display: EyeInHandRgbDisplay,
) -> bool:
    """Hold Panda while keeping the Eye-in-Hand camera active every frame."""
    static_waypoint_positions = [
        initial_arm_positions[joint_index] for joint_index in arm_joint_indices
    ]
    status_interval_steps = max(
        1,
        round(CONTROL_STATUS_INTERVAL_SECONDS / TIME_STEP),
    )
    for simulation_step in range(round(duration / TIME_STEP)):
        if not p.isConnected(client_id):
            return False
        # Never leave the arm passive during the startup hold.  This uses the
        # exact same POSITION_CONTROL gains, force and maxVelocity as MOVE;
        # apply_constrained_arm_position_control overrides every locked target
        # with locked_initial_positions on every physics step.
        apply_constrained_arm_position_control(
            robot_id,
            arm_joint_indices,
            static_waypoint_positions,
            locked_initial_positions,
            client_id,
        )
        p.stepSimulation(physicsClientId=client_id)
        update_camera_reference_axes(
            robot_id,
            client_id,
            camera_axis_debug_item_ids,
        )
        update_camera_translation_candidate_markers(
            robot_id,
            client_id,
            camera_offset_candidate_debug_item_ids,
        )
        # Camera rendering is deliberately independent of robot motion,
        # legacy demos, visual-servo state, and target-detection outcome.
        update_live_eye_in_hand_camera(
            robot_id,
            simulation_step,
            client_id,
            camera_display,
        )
        if (simulation_step + 1) % status_interval_steps == 0:
            print_locked_joint_status(
                "HOLD",
                robot_id,
                locked_initial_positions,
                client_id,
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
    camera_offset_candidate_debug_item_ids: list[int],
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    camera_display: EyeInHandRgbDisplay,
) -> tuple[list[int], list[float], int, float, float, float] | None:
    """Reach one unchanged Stage 3.1 IK-reference target and hold briefly."""
    arm_joint_indices, joint_targets, constrained_ik_error = calculate_constrained_position_ik(
        robot_id,
        target_position,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
    )
    if constrained_ik_error >= CAMERA_TARGET_ERROR_LIMIT:
        raise RuntimeError(
            "The selected locked-joint strategy cannot reach the Camera target "
            f"accurately enough (constrained IK error {constrained_ik_error:.4f} m)."
        )
    start_positions = [
        p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        for joint_index in arm_joint_indices
    ]
    trajectory_duration = calculate_trajectory_duration(start_positions, joint_targets)
    print(f"\n=== Moving to {point_name} ===")
    print(f"Stage 3.1 IK-reference target: {list(target_position)} m")
    print("Constrained q1...q7:", [round(position, 4) for position in joint_targets])
    print(f"Smooth trajectory duration: {trajectory_duration:.1f} s")
    debug_text_id = update_motion_debug_text(
        f"Moving to {point_name}",
        debug_text_id,
        client_id,
    )

    simulation_step = 0
    status_interval_steps = max(
        1,
        round(CONTROL_STATUS_INTERVAL_SECONDS / TIME_STEP),
    )
    final_report_step = round(
        (trajectory_duration + POST_TRAJECTORY_SETTLE_SECONDS) / TIME_STEP
    )
    motion_complete_step = round(trajectory_duration / TIME_STEP)
    motion_duration_printed = False
    max_locked_deviation = 0.0
    while p.isConnected(client_id):
        elapsed_time = simulation_step * TIME_STEP
        interpolation = _smoothstep(elapsed_time / trajectory_duration)
        waypoint_positions = [
            start + interpolation * (target - start)
            for start, target in zip(start_positions, joint_targets)
        ]
        # This is the only motor command for this motion.  It contains only
        # the seven Panda revolute arm joints; no gripper joint is touched.
        apply_constrained_arm_position_control(
            robot_id,
            arm_joint_indices,
            waypoint_positions,
            locked_initial_positions,
            client_id,
        )
        p.stepSimulation(physicsClientId=client_id)
        update_camera_reference_axes(
            robot_id,
            client_id,
            camera_axis_debug_item_ids,
        )
        update_camera_translation_candidate_markers(
            robot_id,
            client_id,
            camera_offset_candidate_debug_item_ids,
        )
        update_live_eye_in_hand_camera(
            robot_id,
            simulation_step,
            client_id,
            camera_display,
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
            max_locked_deviation = max(
                max_locked_deviation,
                print_locked_joint_status(
                    "MOVE",
                    robot_id,
                    locked_initial_positions,
                    client_id,
                ),
            )
        else:
            deviations = get_locked_joint_deviations(
                robot_id,
                locked_initial_positions,
                client_id,
            )
            max_locked_deviation = max(
                max_locked_deviation,
                max((abs(deviation) for deviation in deviations.values()), default=0.0),
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
            final_camera_error = print_position_verification(
                target_position,
                robot_id,
                client_id,
                label=f"{point_name} Final Position Result:",
                point_name=point_name,
            )
            print(f"Locked joint max deviation: {max_locked_deviation:.6f} rad")
            if (
                final_camera_error >= CAMERA_TARGET_ERROR_LIMIT
                or max_locked_deviation >= LOCKED_JOINT_DEVIATION_LIMIT
            ):
                print("Warning: final constrained-motion acceptance criteria were not met.")
            print(
                f"{point_name} held for {POST_TRAJECTORY_SETTLE_SECONDS:.1f} s; "
                "segment complete."
            )
            return (
                arm_joint_indices,
                joint_targets,
                debug_text_id,
                motion_complete_step * TIME_STEP,
                final_camera_error,
                max_locked_deviation,
            )
        time.sleep(TIME_STEP)

    return None


def run_rgb_detection_hold_gui() -> None:
    """Run the Stage 6-ready observation loop without any fixed robot motion.

    Startup and the continuous loop use the accepted constrained
    POSITION_CONTROL baseline purely to hold the recorded initial pose. A
    future visual-servo controller may consume the live (ex, ey) output, but
    until that controller is deliberately enabled, it supplies no joint or
    Cartesian target at all.
    """
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI.")

    camera_display: EyeInHandRgbDisplay | None = None
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.setGravity(0, 0, 0, physicsClientId=client_id)
        p.setTimeStep(TIME_STEP, physicsClientId=client_id)
        enable_pybullet_camera_debug_previews(client_id)
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
        initial_arm_positions = print_panda_arm_joint_configuration(robot_id, client_id)
        arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
        arm_joint_names = get_panda_arm_joint_names(robot_id, client_id)
        locked_joint_indices = [
            arm_joint_names[joint_name] for joint_name in BASELINE_LOCKED_JOINT_NAMES
        ]
        locked_initial_positions = {
            joint_index: initial_arm_positions[joint_index]
            for joint_index in locked_joint_indices
        }
        active_joint_names = [
            joint_name
            for joint_name, joint_index in arm_joint_names.items()
            if joint_index not in locked_joint_indices
        ]
        print("Stage 3.1 baseline locked joints:", list(BASELINE_LOCKED_JOINT_NAMES))
        print("Stage 3.1 baseline active joints:", active_joint_names)

        # This creates only the visual red sphere. Its world position is never
        # passed to IK, a motor command, or the RGB detector.
        create_red_ground_target(client_id)
        print_camera_coordinate_definition()
        camera_display = EyeInHandRgbDisplay()
        print(
            "Live Eye-in-Hand RGB refresh: "
            f"{CAMERA_DISPLAY_FREQUENCY_HZ:.0f} Hz "
            f"(every {CAMERA_UPDATE_INTERVAL_STEPS} simulation steps)."
        )

        camera_axis_debug_item_ids: list[int] = []
        camera_offset_candidate_debug_item_ids: list[int] = []
        update_camera_reference_axes(robot_id, client_id, camera_axis_debug_item_ids)
        update_camera_translation_candidate_markers(
            robot_id,
            client_id,
            camera_offset_candidate_debug_item_ids,
        )
        debug_text_id = update_motion_debug_text(
            f"Initializing: Robot HOLD | Camera ACTIVE ({STARTUP_PAUSE_SECONDS:.0f} s)",
            None,
            client_id,
        )
        print("Initialization phase:")
        print("Robot: HOLD")
        print("Eye-in-Hand Camera: ACTIVE")
        print(f"Holding initial Panda pose for {STARTUP_PAUSE_SECONDS:.1f} s.")
        if not wait_with_static_arm(
            STARTUP_PAUSE_SECONDS,
            robot_id,
            arm_joint_indices,
            initial_arm_positions,
            locked_initial_positions,
            client_id,
            camera_axis_debug_item_ids,
            camera_offset_candidate_debug_item_ids,
            camera_display,
        ):
            return

        debug_text_id = update_motion_debug_text(
            "RGB detection active: holding initial Panda pose",
            debug_text_id,
            client_id,
        )
        print("Legacy Fixed Motion: DISABLED")
        robot_mode = "VISUAL SERVO" if VISUAL_SERVO_MOTION_ENABLED else "HOLD"
        print("After initialization:")
        print(f"Robot: {robot_mode}")
        print("Eye-in-Hand Camera: ACTIVE")
        print(
            "Visual Servo Motion Source: "
            f"{'ENABLED' if VISUAL_SERVO_MOTION_ENABLED else 'NOT YET ENABLED'}"
        )
        print(
            "RGB detection is observation-only. The detected (ex, ey) is "
            "available for a future visual-servo controller; no motion command "
            "is generated in this stage."
        )
        print("Close the PyBullet GUI window to finish the example.")

        hold_targets = [
            initial_arm_positions[joint_index] for joint_index in arm_joint_indices
        ]
        simulation_step = 0
        status_interval_steps = max(
            1,
            round(CONTROL_STATUS_INTERVAL_SECONDS / TIME_STEP),
        )
        while p.isConnected(client_id):
            # Holding targets are the startup angles, not an IK solution. The
            # constrained controller re-applies the recorded locked targets at
            # every step while the active joints remain at their start angles.
            apply_constrained_arm_position_control(
                robot_id,
                arm_joint_indices,
                hold_targets,
                locked_initial_positions,
                client_id,
            )
            p.stepSimulation(physicsClientId=client_id)
            update_camera_reference_axes(
                robot_id,
                client_id,
                camera_axis_debug_item_ids,
            )
            update_camera_translation_candidate_markers(
                robot_id,
                client_id,
                camera_offset_candidate_debug_item_ids,
            )
            update_live_eye_in_hand_camera(
                robot_id,
                simulation_step,
                client_id,
                camera_display,
            )
            simulation_step += 1
            if simulation_step % status_interval_steps == 0:
                print_locked_joint_status(
                    "HOLD",
                    robot_id,
                    locked_initial_positions,
                    client_id,
                )
            time.sleep(TIME_STEP)
    finally:
        if camera_display is not None:
            camera_display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


def run_position_ik_gui_example(
    target_sequence: Sequence[tuple[str, Sequence[float]]],
) -> None:
    """Move Panda to one Stage 3.1 IK-reference target with a lock plan.

    No target orientation is passed to ``calculateInverseKinematics``.  The
    solver therefore computes only a position solution.  Its q1...q7 solution
    is converted into smooth intermediate waypoints, each sent in the fixed
    simulation loop to Panda's seven revolute arm joints using
    ``POSITION_CONTROL``.
    """
    if not LEGACY_FIXED_MOTION_ENABLED:
        raise RuntimeError(
            "Legacy fixed Cartesian/IK motion is disabled. "
            "Run the default RGB-detection hold loop instead."
        )
    if len(target_sequence) != 1:
        raise ValueError("This constrained Stage 3 check accepts exactly one target point.")

    target_name, target_position = target_sequence[0]
    evaluations = [
        evaluate_locking_strategy(strategy_name, locked_joint_names, target_position)
        for strategy_name, locked_joint_names in LOCKING_STRATEGIES
    ]
    print_locking_strategy_comparison(evaluations)
    selected_strategy = choose_locking_strategy(evaluations)
    if selected_strategy is None:
        print(
            "No locking strategy met both the Camera-target and locked-joint "
            "acceptance criteria. GUI motion will not be started."
        )
        return
    print(
        f"\nSelected strategy {selected_strategy.name}: "
        f"lock {list(selected_strategy.locked_joint_names)}, "
        f"active {list(selected_strategy.active_joint_names)}"
    )

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI.")

    camera_display: EyeInHandRgbDisplay | None = None
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
        initial_arm_positions = print_panda_arm_joint_configuration(robot_id, client_id)
        arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
        arm_joint_names = get_panda_arm_joint_names(robot_id, client_id)
        locked_joint_indices = [
            arm_joint_names[joint_name]
            for joint_name in selected_strategy.locked_joint_names
        ]
        locked_initial_positions = {
            joint_index: initial_arm_positions[joint_index]
            for joint_index in locked_joint_indices
        }
        print("Close the PyBullet GUI window to finish the example.")

        # Observation setup only: it creates no motor command and does not
        # change the Stage 3.1 joint-control or IK configuration.
        red_target_id, red_target_world_position = create_red_ground_target(client_id)
        print(
            "Target world position:",
            [round(value, 4) for value in red_target_world_position],
        )
        print_camera_coordinate_definition()
        print_camera_mount_geometry_inspection(
            robot_id,
            red_target_world_position,
            client_id,
        )
        startup_hand_eye_transform = get_measured_hand_eye_transform(robot_id, client_id)
        print_hand_eye_transform_validation(
            "Startup fixed camera",
            startup_hand_eye_transform,
        )
        static_camera_reference_position, static_camera_reference_orientation = (
            get_camera_optical_center_pose(robot_id, client_id)
        )
        static_observation = capture_eye_in_hand_rgb(
            static_camera_reference_position,
            static_camera_reference_orientation,
            red_target_world_position,
            red_target_id,
            client_id,
            STAGE4_FINAL_RGB_OUTPUT_PATH,
        )
        print("\nStatic-arm camera attachment check:")
        print_camera_observation(static_observation, red_target_world_position)
        print_camera_target_alignment(static_observation, red_target_world_position)
        print_camera_render_diagnostics(static_observation, red_target_world_position)
        camera_diagnostic_debug_item_ids = add_camera_diagnostic_debug_lines(
            static_observation,
            red_target_world_position,
            client_id,
        )

        camera_display = EyeInHandRgbDisplay()
        print(
            "Live Eye-in-Hand RGB refresh: "
            f"{CAMERA_DISPLAY_FREQUENCY_HZ:.0f} Hz "
            f"(every {CAMERA_UPDATE_INTERVAL_STEPS} simulation steps)."
        )

        add_target_markers(target_sequence, client_id)
        camera_axis_debug_item_ids: list[int] = []
        camera_offset_candidate_debug_item_ids: list[int] = []
        update_camera_reference_axes(robot_id, client_id, camera_axis_debug_item_ids)
        update_camera_translation_candidate_markers(
            robot_id,
            client_id,
            camera_offset_candidate_debug_item_ids,
        )
        debug_text_id = update_motion_debug_text(
            f"Starting in {STARTUP_PAUSE_SECONDS:.0f} seconds",
            None,
            client_id,
        )
        print(f"Holding still for {STARTUP_PAUSE_SECONDS:.1f} s before the first target.")
        if not wait_with_static_arm(
            STARTUP_PAUSE_SECONDS,
            robot_id,
            arm_joint_indices,
            initial_arm_positions,
            locked_initial_positions,
            client_id,
            camera_axis_debug_item_ids,
            camera_offset_candidate_debug_item_ids,
            camera_display,
        ):
            return

        final_arm_joint_indices: list[int] | None = None
        final_joint_targets: list[float] | None = None
        result = move_arm_smoothly_to_ik_target(
            robot_id,
            target_name,
            target_position,
            client_id,
            debug_text_id,
            camera_axis_debug_item_ids,
            camera_offset_candidate_debug_item_ids,
            locked_joint_indices,
            locked_initial_positions,
            camera_display,
        )
        if result is None:
            return
        (
            final_arm_joint_indices,
            final_joint_targets,
            debug_text_id,
            motion_duration,
            final_camera_error,
            max_locked_deviation,
        ) = result

        if final_arm_joint_indices is None or final_joint_targets is None:
            raise RuntimeError("The IK target sequence must contain at least one point.")

        print(f"\n{target_name} constrained motion complete. Holding the final position.")
        print(f"Actual motion duration: {motion_duration:.2f} s")
        print(f"Final camera position error: {final_camera_error:.4f} m")
        print(f"Locked joint max deviation: {max_locked_deviation:.6f} rad")

        # Render once after the validated Stage 3.1 motion has completed.  The
        # renderer reads the offset optical-center C pose but sends no joint
        # commands and no IK request.
        camera_reference_position, camera_reference_orientation = (
            get_camera_optical_center_pose(robot_id, client_id)
        )
        print_camera_mount_geometry_inspection(
            robot_id,
            red_target_world_position,
            client_id,
        )
        observation = capture_eye_in_hand_rgb(
            camera_reference_position,
            camera_reference_orientation,
            red_target_world_position,
            red_target_id,
            client_id,
        )
        print("\nPost-motion camera attachment check:")
        print_camera_observation(observation, red_target_world_position)
        print_camera_target_alignment(observation, red_target_world_position)
        print_camera_render_diagnostics(observation, red_target_world_position)
        camera_diagnostic_debug_item_ids = add_camera_diagnostic_debug_lines(
            observation,
            red_target_world_position,
            client_id,
            camera_diagnostic_debug_item_ids,
        )
        if observation.target_visible_pixel_count == 0:
            raise RuntimeError(
                "The red target was not visible in the RGB preview; "
                "the camera-observation check did not pass."
            )
        forced_look_at_observation = capture_forced_look_at_rgb(
            camera_reference_position,
            camera_reference_orientation,
            red_target_world_position,
            red_target_id,
            client_id,
            CAMERA_OFFSET_DEBUG_OUTPUT_PATH,
        )
        print("\nForced look-at camera diagnostic (no Panda pose change):")
        print_camera_observation(forced_look_at_observation, red_target_world_position)
        print_camera_render_diagnostics(
            forced_look_at_observation,
            red_target_world_position,
        )
        camera_diagnostic_debug_item_ids = add_camera_diagnostic_debug_lines(
            forced_look_at_observation,
            red_target_world_position,
            client_id,
            camera_diagnostic_debug_item_ids,
        )
        print(
            "Forced look-at red target visible pixels: "
            f"{forced_look_at_observation.target_visible_pixel_count}"
        )

        # Stage 4 fixed-mount check: use the same accepted position-only IK
        # and motor controller for a nearby target.  No camera orientation is
        # recomputed; only T_W_E changes as the arm moves.
        secondary_target_name, secondary_target_position = SECONDARY_CAMERA_REFERENCE_TARGET
        secondary_result = move_arm_smoothly_to_ik_target(
            robot_id,
            secondary_target_name,
            secondary_target_position,
            client_id,
            debug_text_id,
            camera_axis_debug_item_ids,
            camera_offset_candidate_debug_item_ids,
            locked_joint_indices,
            locked_initial_positions,
            camera_display,
        )
        if secondary_result is None:
            return
        (
            final_arm_joint_indices,
            final_joint_targets,
            debug_text_id,
            secondary_motion_duration,
            secondary_camera_error,
            secondary_max_locked_deviation,
        ) = secondary_result
        print(f"\n{secondary_target_name} complete.")
        print(f"Actual secondary motion duration: {secondary_motion_duration:.2f} s")
        print(f"Secondary Camera-reference position error: {secondary_camera_error:.4f} m")
        print(f"Secondary locked joint max deviation: {secondary_max_locked_deviation:.6f} rad")

        second_camera_position, second_camera_orientation = get_camera_optical_center_pose(
            robot_id,
            client_id,
        )
        second_hand_eye_transform = get_measured_hand_eye_transform(robot_id, client_id)
        print_camera_mount_geometry_inspection(
            robot_id,
            red_target_world_position,
            client_id,
        )
        print_hand_eye_transform_validation(
            "Second robot pose",
            second_hand_eye_transform,
            startup_hand_eye_transform,
        )
        second_pose_observation = capture_eye_in_hand_rgb(
            second_camera_position,
            second_camera_orientation,
            red_target_world_position,
            red_target_id,
            client_id,
            STAGE4_SECOND_POSE_RGB_OUTPUT_PATH,
        )
        print("\nSecond-pose fixed-camera RGB check:")
        print_camera_observation(second_pose_observation, red_target_world_position)
        print_camera_render_diagnostics(second_pose_observation, red_target_world_position)
        camera_diagnostic_debug_item_ids = add_camera_diagnostic_debug_lines(
            second_pose_observation,
            red_target_world_position,
            client_id,
            camera_diagnostic_debug_item_ids,
        )
        if second_pose_observation.target_visible_pixel_count == 0:
            raise RuntimeError(
                "The red target was not visible from the fixed camera at the "
                "second robot pose."
            )

        final_hold_step = 0
        final_hold_status_interval_steps = max(
            1,
            round(CONTROL_STATUS_INTERVAL_SECONDS / TIME_STEP),
        )
        while p.isConnected(client_id):
            apply_constrained_arm_position_control(
                robot_id,
                final_arm_joint_indices,
                final_joint_targets,
                locked_initial_positions,
                client_id,
            )
            p.stepSimulation(physicsClientId=client_id)
            update_camera_reference_axes(
                robot_id,
                client_id,
                camera_axis_debug_item_ids,
            )
            update_camera_translation_candidate_markers(
                robot_id,
                client_id,
                camera_offset_candidate_debug_item_ids,
            )
            update_live_eye_in_hand_camera(
                robot_id,
                final_hold_step,
                client_id,
                camera_display,
            )
            final_hold_step += 1
            if final_hold_step % final_hold_status_interval_steps == 0:
                print_locked_joint_status(
                    "HOLD",
                    robot_id,
                    locked_initial_positions,
                    client_id,
                )
            time.sleep(TIME_STEP)
    finally:
        if camera_display is not None:
            camera_display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


def run_single_joint_gui_example() -> None:
    """Move only Panda's first revolute joint slowly in a visible GUI window.

    Gravity is disabled so the remaining, intentionally uncontrolled joints
    stay still instead of falling.  No motor-control command is sent to any
    joint other than ``CONTROLLED_JOINT_INDEX``.
    """
    if not LEGACY_FIXED_MOTION_ENABLED:
        raise RuntimeError(
            "Legacy single-joint motion is disabled during Stage 6 preparation."
        )
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
    arguments = parser.parse_args()

    if arguments.inspect:
        inspect_panda_structure()
    else:
        run_rgb_detection_hold_gui()
