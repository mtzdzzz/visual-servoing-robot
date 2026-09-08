"""Inspect Panda or run one single-axis visual-servo check in PyBullet.

The default loop preserves the accepted Stage 3.1 locked-joint baseline and
fixed hand-eye camera. The selected Stage 6 mode performs RGB detection and
one bounded proportional Camera-frame axis validation; it does not implement
PID, two-axis control, ROS, Gazebo, grasping, TCP compensation, or YOLO.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
    RedTargetDetection,
    add_camera_diagnostic_debug_lines,
    add_servo_state_overlay,
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
# Stage 6A camera/servo updates are independent of the 240 Hz physics loop.
# Keep the last verified 20 Hz baseline after the 30 Hz fine-deadband trial
# could not complete under the existing active-joint waypoint tolerance.
CAMERA_DISPLAY_FREQUENCY_HZ = 20.0
CAMERA_UPDATE_INTERVAL_STEPS = max(
    1,
    round(1.0 / (TIME_STEP * CAMERA_DISPLAY_FREQUENCY_HZ)),
)
TARGET_DETECTION_PRINT_INTERVAL_STEPS = max(1, round(1.0 / TIME_STEP))
SINGLE_AXIS_CALIBRATION_STEP_METRES = 0.002
# Stable Stage 6A baseline.  The speed-tuning trials may temporarily raise
# these two values, but a rejected trial must leave these safe values intact.
SINGLE_AXIS_SERVO_KP_METRES_PER_PIXEL = 0.0001
SINGLE_AXIS_SERVO_MAX_STEP_METRES = 0.002
# A proportional (not fixed-distance) Camera-X step uses a progressively
# smaller clamp near the image center.  The global MAX_STEP above remains a
# hard upper bound, so the baseline naturally stays at 2 mm in every region.
SINGLE_AXIS_SERVO_MAX_STEP_FAST_METRES = 0.010
SINGLE_AXIS_SERVO_MAX_STEP_MEDIUM_METRES = 0.005
SINGLE_AXIS_SERVO_MAX_STEP_FINE_METRES = 0.002
SINGLE_AXIS_SERVO_FAST_ERROR_PIXELS = 100
SINGLE_AXIS_SERVO_MEDIUM_ERROR_PIXELS = 30
# The requested <8 px deadband was tested, but pixel quantisation stalled at
# exactly 8 px under the fixed Stage 3.1 waypoint tolerance.  Preserve the
# last demonstrated stable <10 px baseline rather than leaving a step-limit
# HOLD configuration active.
SINGLE_AXIS_SERVO_EX_GOAL_PIXELS = 10
SINGLE_AXIS_OVERSHOOT_HOLD_PIXELS = 20
SINGLE_AXIS_CALIBRATION_MIN_RESPONSE_PIXELS = 1
# The locked-joint numerical IK leaves a measured residual of about 1.3 mm for
# a 2 mm Camera-X command; accepting up to 2 mm lets RGB feedback close the
# loop from the actual pose without weakening the 2 mm physical step limit.
SINGLE_AXIS_CAMERA_TARGET_TOLERANCE_METRES = 0.002
# A waypoint is complete only after the *active* joints have reached the
# commanded IK target.  Camera-position residual is deliberately not used as
# completion criteria, because the permitted 2 mm planning residual could
# otherwise acknowledge a 2 mm command before the robot moves at all.
# The accepted Stage 3.1 low-gain controller has a measured 0.0046 rad
# active-joint steady-state error on this small waypoint, so 0.006 rad is the
# practical completion band.  A separate four-render-frame minimum below
# guarantees the waypoint has first been physically applied.
SINGLE_AXIS_ACTIVE_JOINT_TARGET_TOLERANCE_RAD = 0.006
SINGLE_AXIS_MIN_WAYPOINT_RENDER_FRAMES = 4
SINGLE_AXIS_CAMERA_IK_CORRECTION_ITERATIONS = 3
SINGLE_AXIS_COMMAND_TIMEOUT_RENDER_FRAMES = 160
SINGLE_AXIS_MAX_CONSECUTIVE_ERROR_INCREASES = 3
SINGLE_AXIS_MAX_CONTROL_STEPS = 120
# Stage 6B reuses the accepted Stage 6A numeric baseline.  These distinct
# names make the vertical controller auditable without changing any Stage 6A
# parameter, sign, or control path.
VERTICAL_SERVO_KP_METRES_PER_PIXEL = SINGLE_AXIS_SERVO_KP_METRES_PER_PIXEL
VERTICAL_SERVO_MAX_STEP_METRES = SINGLE_AXIS_SERVO_MAX_STEP_METRES
VERTICAL_SERVO_GOAL_PIXELS = SINGLE_AXIS_SERVO_EX_GOAL_PIXELS
# Measured by the accepted Stage 6A/6B +/-2 mm RGB calibration at the current
# baseline start pose.  Stage 6C consumes these values directly and does not
# repeat or guess either sign.
STAGE6A_CALIBRATED_HORIZONTAL_SIGN = -1
STAGE6B_CALIBRATED_VERTICAL_SIGN = -1
TWO_D_TOL_X_PIXELS = SINGLE_AXIS_SERVO_EX_GOAL_PIXELS
TWO_D_TOL_Y_PIXELS = VERTICAL_SERVO_GOAL_PIXELS
TWO_D_ERROR_NORM_INCREASE_TOLERANCE_PIXELS = 3.0
TWO_D_MAX_CONSECUTIVE_NORM_INCREASES = 3
TWO_D_CAMERA_ORIENTATION_STEP_LIMIT_DEGREES = 10.0
# Each axis retains its Stage 6A/6B 2 mm maximum.  A simultaneous command is
# additionally norm-limited to 2 mm, preventing its 2.83 mm resultant from
# violating the accepted constrained-IK tolerance. This is a safety reduction,
# never a speed increase or a change to either single-axis parameter.
TWO_D_MAX_RESULTANT_STEP_METRES = min(
    SINGLE_AXIS_SERVO_MAX_STEP_METRES,
    VERTICAL_SERVO_MAX_STEP_METRES,
)

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
# Select one accepted Stage 6 path. Stage 6C combines only the existing
# calibrated Camera-X/ex and Camera-Y/ey axes; it does not redefine either.
VISUAL_SERVO_MODE = "TWO_D"
VISUAL_SERVO_MOTION_ENABLED = True
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
# The default Stage 6A path does not command a startup pose.  Archived legacy
# demos may still use this duration explicitly, but the RGB/servo loop starts
# from the loaded/current Panda pose immediately.
STARTUP_PAUSE_SECONDS = 0.0
DEBUG_STATUS_TEXT_POSITION = (0.0, 0.0, 1.35)
DEBUG_TARGET_MARKER_SIZE = 0.045
CAMERA_AXIS_LENGTH = 0.10
HORIZONTAL_RECALIBRATION_KEY_CODES = (ord("r"), ord("R"))
VERTICAL_RECALIBRATION_KEY_CODES = (ord("r"), ord("R"))


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


@dataclass(frozen=True)
class CameraAxisMotionCommand:
    """One bounded Camera-frame translation executed through constrained IK."""

    label: str
    camera_delta_c: tuple[float, float, float]
    camera_delta_w: tuple[float, float, float]
    target_camera_position: tuple[float, float, float]
    joint_targets: tuple[float, ...]
    camera_target_error: float


@dataclass
class SingleAxisVisualServo:
    """Stage 6A state machine: calibrate Camera X, then close only on ex."""

    state: str = "CALIBRATION: WAITING FOR TARGET"
    initial_u: int | None = None
    initial_ex: int | None = None
    plus_x_u: int | None = None
    plus_x_ex: int | None = None
    minus_x_u: int | None = None
    minus_x_ex: int | None = None
    control_direction: int | None = None
    calibration_initial_joint_targets: tuple[float, ...] | None = None
    calibration_initial_camera_position: tuple[float, float, float] | None = None
    pending_command: CameraAxisMotionCommand | None = None
    pending_render_frames: int = 0
    motion_completed: bool = False
    previous_abs_ex: int | None = None
    consecutive_error_increases: int = 0
    control_steps: int = 0
    control_started_at: float | None = None
    recent_control_history: list[tuple[int, float, tuple[float, float, float]]] = field(
        default_factory=list
    )
    first_servo_command_logged: bool = False
    hold_requested: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state.startswith(
            (
                "HORIZONTAL CENTERED",
                "DIVERGING",
                "FAIL",
                "TARGET LOST",
                "HOLD",
                "OVERSHOOT",
            )
        )

    def request_hold(self, state: str) -> None:
        self.state = state
        self.pending_command = None
        self.hold_requested = True

    def record_completed_p_step(
        self,
        ex: int,
        command_delta_x: float,
        camera_position: Sequence[float],
    ) -> None:
        """Keep the last five measured horizontal-control samples for safety logs."""
        self.recent_control_history.append(
            (ex, command_delta_x, tuple(camera_position))
        )
        del self.recent_control_history[:-5]


@dataclass
class VerticalVisualServo:
    """Stage 6B state machine: calibrate Camera Y, then close only on ey."""

    state: str = "CALIBRATION: WAITING FOR TARGET"
    initial_v: int | None = None
    initial_ey: int | None = None
    plus_y_v: int | None = None
    plus_y_ey: int | None = None
    minus_y_v: int | None = None
    minus_y_ey: int | None = None
    control_direction: int | None = None
    calibration_initial_joint_targets: tuple[float, ...] | None = None
    calibration_initial_camera_position: tuple[float, float, float] | None = None
    pending_command: CameraAxisMotionCommand | None = None
    pending_render_frames: int = 0
    motion_completed: bool = False
    previous_abs_ey: int | None = None
    consecutive_error_increases: int = 0
    control_steps: int = 0
    control_started_at: float | None = None
    recent_control_history: list[tuple[int, float, tuple[float, float, float]]] = field(
        default_factory=list
    )
    first_servo_command_logged: bool = False
    hold_requested: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state.startswith(
            (
                "VERTICAL CENTERED",
                "DIVERGING",
                "FAIL",
                "TARGET LOST",
                "HOLD",
                "OVERSHOOT",
            )
        )

    def request_hold(self, state: str) -> None:
        self.state = state
        self.pending_command = None
        self.hold_requested = True

    def record_completed_p_step(
        self,
        ey: int,
        command_delta_y: float,
        camera_position: Sequence[float],
    ) -> None:
        """Keep the last five vertical-control samples for safety logs."""
        self.recent_control_history.append(
            (ey, command_delta_y, tuple(camera_position))
        )
        del self.recent_control_history[:-5]


@dataclass
class TwoDimensionalVisualServo:
    """Stage 6C state machine: simultaneous calibrated Camera-X/Y tracking."""

    state: str = "2D WAITING FOR TARGET"
    initial_pixel: tuple[int, int] | None = None
    initial_ex: int | None = None
    initial_ey: int | None = None
    initial_error_norm: float | None = None
    horizontal_sign: int = STAGE6A_CALIBRATED_HORIZONTAL_SIGN
    vertical_sign: int = STAGE6B_CALIBRATED_VERTICAL_SIGN
    pending_command: CameraAxisMotionCommand | None = None
    pending_camera_orientation: tuple[float, float, float, float] | None = None
    pending_render_frames: int = 0
    motion_completed: bool = False
    previous_error_norm: float | None = None
    consecutive_norm_increases: int = 0
    control_steps: int = 0
    control_started_at: float | None = None
    recent_control_history: list[
        tuple[int, int, float, float, float, tuple[float, float, float]]
    ] = field(default_factory=list)
    max_camera_orientation_step_degrees: float = 0.0
    orientation_stable: bool = True
    last_ex: int | None = None
    last_ey: int | None = None
    last_error_norm: float | None = None
    last_delta_x: float = 0.0
    last_delta_y: float = 0.0
    first_servo_command_logged: bool = False
    hold_requested: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state.startswith(
            (
                "2D CENTERED",
                "DIVERGING",
                "FAIL",
                "TARGET LOST",
                "HOLD",
                "OVERSHOOT",
            )
        )

    def request_hold(self, state: str) -> None:
        self.state = state
        self.pending_command = None
        self.pending_camera_orientation = None
        self.hold_requested = True

    def record_completed_p_step(
        self,
        ex: int,
        ey: int,
        error_norm: float,
        delta_x: float,
        delta_y: float,
        camera_position: Sequence[float],
    ) -> None:
        """Keep recent 2D samples for the norm-based divergence diagnosis."""
        self.recent_control_history.append(
            (
                ex,
                ey,
                error_norm,
                delta_x,
                delta_y,
                tuple(camera_position),
            )
        )
        del self.recent_control_history[:-5]


def get_visual_servo_overlay_text(
    servo: SingleAxisVisualServo | VerticalVisualServo | TwoDimensionalVisualServo,
) -> str:
    """Return concise RGB-overlay status without affecting detection/control."""
    if not isinstance(servo, TwoDimensionalVisualServo):
        return servo.state
    if (
        servo.last_ex is None
        or servo.last_ey is None
        or servo.last_error_norm is None
    ):
        return servo.state
    return (
        f"{servo.state} | ex={servo.last_ex} ey={servo.last_ey} "
        f"norm={servo.last_error_norm:.1f} dx={servo.last_delta_x:.4f} "
        f"dy={servo.last_delta_y:.4f}"
    )


def get_single_axis_adaptive_step_limit(absolute_ex: int | float) -> float:
    """Return the Camera-X P-step limit for the current image error.

    ``SINGLE_AXIS_SERVO_MAX_STEP_METRES`` is the trial's global safety bound.
    The zone limit only lowers that bound as the target nears the image centre;
    it never alters the calibrated Camera-X sign or any Panda motor setting.
    """
    if absolute_ex > SINGLE_AXIS_SERVO_FAST_ERROR_PIXELS:
        zone_limit = SINGLE_AXIS_SERVO_MAX_STEP_FAST_METRES
    elif absolute_ex > SINGLE_AXIS_SERVO_MEDIUM_ERROR_PIXELS:
        zone_limit = SINGLE_AXIS_SERVO_MAX_STEP_MEDIUM_METRES
    else:
        zone_limit = SINGLE_AXIS_SERVO_MAX_STEP_FINE_METRES
    return min(SINGLE_AXIS_SERVO_MAX_STEP_METRES, zone_limit)


def calculate_single_axis_adaptive_p_step(ex: int | float) -> tuple[float, float, float]:
    """Return raw P step, zone limit, and bounded unsigned-sign P step.

    The returned step still requires the measured calibration sign before it
    becomes a Camera-frame command.  This helper is deliberately pure so the
    Stage 6A speed policy can be checked without touching robot state.
    """
    raw_step = SINGLE_AXIS_SERVO_KP_METRES_PER_PIXEL * ex
    step_limit = get_single_axis_adaptive_step_limit(abs(ex))
    applied_step = max(-step_limit, min(step_limit, raw_step))
    return raw_step, step_limit, applied_step


def calculate_vertical_adaptive_p_step(ey: int | float) -> tuple[float, float, float]:
    """Return the Stage 6B raw, limited, Camera-Y P step for ``ey`` only."""
    raw_step = VERTICAL_SERVO_KP_METRES_PER_PIXEL * ey
    step_limit = min(
        VERTICAL_SERVO_MAX_STEP_METRES,
        get_single_axis_adaptive_step_limit(abs(ey)),
    )
    applied_step = max(-step_limit, min(step_limit, raw_step))
    return raw_step, step_limit, applied_step


def get_locked_joint_report_phase(
    servo: SingleAxisVisualServo | VerticalVisualServo | TwoDimensionalVisualServo,
) -> str:
    """Map Stage 6A detail states onto the strict HOLD/MOVE report contract."""
    return "MOVE" if servo.pending_command is not None and not servo.is_terminal else "HOLD"


def horizontal_recalibration_requested(client_id: int) -> bool:
    """Return True only when the user requests a safe Stage 6A recalibration."""
    keyboard_events = p.getKeyboardEvents(physicsClientId=client_id)
    return any(
        keyboard_events.get(key_code, 0) & p.KEY_WAS_TRIGGERED
        for key_code in HORIZONTAL_RECALIBRATION_KEY_CODES
    )


def vertical_recalibration_requested(client_id: int) -> bool:
    """Return True only for a safe Stage 6B current-pose recalibration."""
    keyboard_events = p.getKeyboardEvents(physicsClientId=client_id)
    return any(
        keyboard_events.get(key_code, 0) & p.KEY_WAS_TRIGGERED
        for key_code in VERTICAL_RECALIBRATION_KEY_CODES
    )


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


def camera_frame_delta_to_world(
    camera_position: Sequence[float],
    camera_orientation: Sequence[float],
    camera_delta_c: Sequence[float],
) -> list[float]:
    """Calculate delta_p_W = R_W_C @ delta_p_C for one Camera-frame offset."""
    if len(camera_delta_c) != 3:
        raise ValueError("Camera-frame translation must contain exactly [x, y, z].")
    endpoint, _ = p.multiplyTransforms(
        camera_position,
        camera_orientation,
        camera_delta_c,
        (0.0, 0.0, 0.0, 1.0),
    )
    return [
        endpoint[index] - camera_position[index]
        for index in range(3)
    ]


def calculate_constrained_camera_position_ik(
    robot_id: int,
    target_camera_position: Sequence[float],
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> tuple[list[int], list[float], float]:
    """Solve a small C-position target using the unchanged constrained E IK.

    The existing Stage 3.1 solver remains responsible for all joint-limit,
    active-joint, and locked-joint logic. This wrapper corrects its E-frame
    target against the measured physical C position so the fixed T_E_C offset
    is respected without changing that accepted baseline solver.
    """
    if len(target_camera_position) != 3:
        raise ValueError("target_camera_position must contain exactly [x, y, z].")

    current_camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
    current_end_effector_position, _ = get_camera_reference_pose(robot_id, client_id)
    end_effector_target = [
        end_effector + (target_camera - current_camera)
        for end_effector, target_camera, current_camera in zip(
            current_end_effector_position,
            target_camera_position,
            current_camera_position,
        )
    ]
    arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
    saved_positions = [
        p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        for joint_index in arm_joint_indices
    ]
    joint_targets: list[float] | None = None
    final_camera_error = float("inf")
    try:
        for _ in range(SINGLE_AXIS_CAMERA_IK_CORRECTION_ITERATIONS):
            (
                arm_joint_indices,
                joint_targets,
                _,
            ) = calculate_constrained_position_ik(
                robot_id,
                end_effector_target,
                locked_joint_indices,
                locked_initial_positions,
                client_id,
            )
            _set_arm_joint_positions(
                robot_id,
                arm_joint_indices,
                joint_targets,
                client_id,
            )
            predicted_camera_position, _ = get_camera_optical_center_pose(
                robot_id,
                client_id,
            )
            camera_error_vector = [
                target - actual
                for target, actual in zip(
                    target_camera_position,
                    predicted_camera_position,
                )
            ]
            final_camera_error = sqrt(
                sum(component * component for component in camera_error_vector)
            )
            if final_camera_error <= SINGLE_AXIS_CAMERA_TARGET_TOLERANCE_METRES:
                break
            end_effector_target = [
                target + error
                for target, error in zip(end_effector_target, camera_error_vector)
            ]
    finally:
        _set_arm_joint_positions(
            robot_id,
            arm_joint_indices,
            saved_positions,
            client_id,
        )

    if joint_targets is None:
        raise RuntimeError("Camera-frame constrained IK did not produce joint targets.")
    return arm_joint_indices, joint_targets, final_camera_error


def plan_camera_x_axis_motion(
    robot_id: int,
    label: str,
    camera_delta_x: float,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand:
    """Plan one bounded local Camera +X or -X translation through IK."""
    if abs(camera_delta_x) > SINGLE_AXIS_SERVO_MAX_STEP_METRES + 1e-12:
        raise ValueError(
            "A Stage 6A Camera X command may not exceed "
            f"{SINGLE_AXIS_SERVO_MAX_STEP_METRES:.4f} m."
        )
    camera_position, camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    camera_delta_c = (camera_delta_x, 0.0, 0.0)
    camera_delta_w = camera_frame_delta_to_world(
        camera_position,
        camera_orientation,
        camera_delta_c,
    )
    target_camera_position = [
        position + delta
        for position, delta in zip(camera_position, camera_delta_w)
    ]
    _, joint_targets, camera_target_error = calculate_constrained_camera_position_ik(
        robot_id,
        target_camera_position,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
    )
    if camera_target_error > SINGLE_AXIS_CAMERA_TARGET_TOLERANCE_METRES:
        raise RuntimeError(
            "Constrained Camera X target is not reachable within tolerance: "
            f"{camera_target_error:.6f} m."
        )
    print(
        f"{label}: delta_p_C={list(camera_delta_c)} m; "
        f"delta_p_W={[round(value, 6) for value in camera_delta_w]} m; "
        f"target C={[round(value, 6) for value in target_camera_position]} m; "
        f"IK error={camera_target_error:.6f} m"
    )
    return CameraAxisMotionCommand(
        label=label,
        camera_delta_c=camera_delta_c,
        camera_delta_w=tuple(camera_delta_w),
        target_camera_position=tuple(target_camera_position),
        joint_targets=tuple(joint_targets),
        camera_target_error=camera_target_error,
    )


def plan_camera_y_axis_motion(
    robot_id: int,
    label: str,
    camera_delta_y: float,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand:
    """Plan one bounded local Camera +Y or -Y translation through IK.

    Camera +Y is the fixed camera-frame up-vector used by ``getCameraImage``.
    It is only a candidate for reducing image ``ey``; the calibration below,
    rather than this geometric convention, determines the control sign.
    """
    if abs(camera_delta_y) > VERTICAL_SERVO_MAX_STEP_METRES + 1e-12:
        raise ValueError(
            "A Stage 6B Camera Y command may not exceed "
            f"{VERTICAL_SERVO_MAX_STEP_METRES:.4f} m."
        )
    camera_position, camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    camera_delta_c = (0.0, camera_delta_y, 0.0)
    camera_delta_w = camera_frame_delta_to_world(
        camera_position,
        camera_orientation,
        camera_delta_c,
    )
    target_camera_position = [
        position + delta
        for position, delta in zip(camera_position, camera_delta_w)
    ]
    _, joint_targets, camera_target_error = calculate_constrained_camera_position_ik(
        robot_id,
        target_camera_position,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
    )
    if camera_target_error > SINGLE_AXIS_CAMERA_TARGET_TOLERANCE_METRES:
        raise RuntimeError(
            "Constrained Camera Y target is not reachable within tolerance: "
            f"{camera_target_error:.6f} m."
        )
    print(
        f"{label}: delta_p_C={list(camera_delta_c)} m; "
        f"delta_p_W={[round(value, 6) for value in camera_delta_w]} m; "
        f"target C={[round(value, 6) for value in target_camera_position]} m; "
        f"IK error={camera_target_error:.6f} m"
    )
    return CameraAxisMotionCommand(
        label=label,
        camera_delta_c=camera_delta_c,
        camera_delta_w=tuple(camera_delta_w),
        target_camera_position=tuple(target_camera_position),
        joint_targets=tuple(joint_targets),
        camera_target_error=camera_target_error,
    )


def plan_camera_xy_motion(
    robot_id: int,
    label: str,
    camera_delta_x: float,
    camera_delta_y: float,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand:
    """Plan one current-pose-relative Stage 6C Camera-frame XY increment."""
    if abs(camera_delta_x) > SINGLE_AXIS_SERVO_MAX_STEP_METRES + 1e-12:
        raise ValueError(
            "A Stage 6C Camera X component may not exceed "
            f"{SINGLE_AXIS_SERVO_MAX_STEP_METRES:.4f} m."
        )
    if abs(camera_delta_y) > VERTICAL_SERVO_MAX_STEP_METRES + 1e-12:
        raise ValueError(
            "A Stage 6C Camera Y component may not exceed "
            f"{VERTICAL_SERVO_MAX_STEP_METRES:.4f} m."
        )
    camera_position, camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    camera_delta_c = (camera_delta_x, camera_delta_y, 0.0)
    camera_delta_w = camera_frame_delta_to_world(
        camera_position,
        camera_orientation,
        camera_delta_c,
    )
    target_camera_position = [
        position + delta
        for position, delta in zip(camera_position, camera_delta_w)
    ]
    _, joint_targets, camera_target_error = calculate_constrained_camera_position_ik(
        robot_id,
        target_camera_position,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
    )
    if camera_target_error > SINGLE_AXIS_CAMERA_TARGET_TOLERANCE_METRES:
        raise RuntimeError(
            "Constrained Camera XY target is not reachable within tolerance: "
            f"{camera_target_error:.6f} m."
        )
    print(
        f"{label}: delta_p_C={list(camera_delta_c)} m; "
        f"delta_p_W={[round(value, 6) for value in camera_delta_w]} m; "
        f"target C={[round(value, 6) for value in target_camera_position]} m; "
        f"IK error={camera_target_error:.6f} m"
    )
    return CameraAxisMotionCommand(
        label=label,
        camera_delta_c=camera_delta_c,
        camera_delta_w=tuple(camera_delta_w),
        target_camera_position=tuple(target_camera_position),
        joint_targets=tuple(joint_targets),
        camera_target_error=camera_target_error,
    )


def _request_camera_x_motion(
    servo: SingleAxisVisualServo,
    robot_id: int,
    label: str,
    camera_delta_x: float,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand | None:
    """Plan a new x-only Camera command or stop safely if constrained IK fails."""
    try:
        command = plan_camera_x_axis_motion(
            robot_id,
            label,
            camera_delta_x,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )
    except RuntimeError as error:
        print(f"{label}: control direction FAIL / HOLD ({error})")
        servo.request_hold("FAIL: CAMERA IK / HOLD")
        return None
    servo.pending_command = command
    servo.pending_render_frames = 0
    return command


def _request_camera_y_motion(
    servo: VerticalVisualServo,
    robot_id: int,
    label: str,
    camera_delta_y: float,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand | None:
    """Plan a Stage 6B Y-only command or safely HOLD if constrained IK fails."""
    try:
        command = plan_camera_y_axis_motion(
            robot_id,
            label,
            camera_delta_y,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )
    except RuntimeError as error:
        print(f"{label}: vertical control FAIL / HOLD ({error})")
        servo.request_hold("FAIL: CAMERA IK / HOLD")
        return None
    servo.pending_command = command
    servo.pending_render_frames = 0
    return command


def _request_camera_xy_motion(
    servo: TwoDimensionalVisualServo,
    robot_id: int,
    label: str,
    camera_delta_x: float,
    camera_delta_y: float,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand | None:
    """Plan one Stage 6C XY command or safely HOLD if constrained IK fails."""
    try:
        command = plan_camera_xy_motion(
            robot_id,
            label,
            camera_delta_x,
            camera_delta_y,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )
    except RuntimeError as error:
        print(f"{label}: 2D control FAIL / HOLD ({error})")
        servo.request_hold("FAIL: CAMERA IK / HOLD")
        return None
    _, camera_orientation = get_camera_optical_center_pose(robot_id, client_id)
    servo.pending_command = command
    servo.pending_camera_orientation = tuple(camera_orientation)
    servo.pending_render_frames = 0
    return command


def _request_calibration_restore(
    servo: SingleAxisVisualServo | VerticalVisualServo,
    label: str,
) -> CameraAxisMotionCommand | None:
    """Return exactly to the joint pose captured before the calibration probe."""
    if (
        servo.calibration_initial_joint_targets is None
        or servo.calibration_initial_camera_position is None
    ):
        print(f"{label}: calibration start pose is unavailable; HOLD.")
        servo.request_hold("FAIL: CALIBRATION START POSE / HOLD")
        return None
    command = CameraAxisMotionCommand(
        label=label,
        camera_delta_c=(0.0, 0.0, 0.0),
        camera_delta_w=(0.0, 0.0, 0.0),
        target_camera_position=servo.calibration_initial_camera_position,
        joint_targets=servo.calibration_initial_joint_targets,
        camera_target_error=0.0,
    )
    servo.pending_command = command
    servo.pending_render_frames = 0
    print(f"{label}: restoring the captured pre-calibration active-joint pose.")
    return command


def get_active_joint_target_error(
    robot_id: int,
    joint_targets: Sequence[float],
    locked_joint_indices: Sequence[int],
    client_id: int,
) -> float:
    """Return the largest active-joint error to one seven-joint waypoint."""
    arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
    if len(joint_targets) != len(arm_joint_indices):
        raise ValueError("A Stage 6A waypoint must contain seven arm targets.")
    locked_set = set(locked_joint_indices)
    active_errors = [
        abs(
            p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
            - target_position
        )
        for joint_index, target_position in zip(arm_joint_indices, joint_targets)
        if joint_index not in locked_set
    ]
    if not active_errors:
        raise RuntimeError("Stage 6A requires at least one active Panda joint.")
    return max(active_errors)


def update_single_axis_visual_servo(
    servo: SingleAxisVisualServo,
    detection: RedTargetDetection,
    robot_id: int,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand | None:
    """Advance the Camera-X calibration and ex-only P loop by one RGB frame."""
    if servo.is_terminal:
        return None
    if not detection.detected:
        print("Target lost: control paused, HOLD current pose.")
        servo.request_hold("TARGET LOST / HOLD")
        return None

    assert detection.centroid is not None
    assert detection.pixel_error is not None
    u, _ = detection.centroid
    ex, ey = detection.pixel_error

    if servo.pending_command is not None:
        active_joint_error = get_active_joint_target_error(
            robot_id,
            servo.pending_command.joint_targets,
            locked_joint_indices,
            client_id,
        )
        servo.pending_render_frames += 1
        if servo.pending_render_frames < SINGLE_AXIS_MIN_WAYPOINT_RENDER_FRAMES:
            return None
        if active_joint_error > SINGLE_AXIS_ACTIVE_JOINT_TARGET_TOLERANCE_RAD:
            if servo.pending_render_frames > SINGLE_AXIS_COMMAND_TIMEOUT_RENDER_FRAMES:
                print(
                    f"{servo.pending_command.label}: control direction FAIL / HOLD "
                    "(active-joint waypoint timeout, "
                    f"error={active_joint_error:.6f} rad)."
                )
                servo.request_hold("FAIL: ACTIVE WAYPOINT TIMEOUT / HOLD")
            return None
        current_camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
        motion_start_position = [
            target - delta
            for target, delta in zip(
                servo.pending_command.target_camera_position,
                servo.pending_command.camera_delta_w,
            )
        ]
        actual_camera_displacement = [
            actual - start
            for actual, start in zip(
                current_camera_position,
                motion_start_position,
            )
        ]
        print(
            f"{servo.pending_command.label}: reached active-joint waypoint; "
            f"active joint error={active_joint_error:.6f} rad; "
            f"u={u}, ex={ex}, ey={ey} (ey is observation only)."
        )
        if servo.pending_command.label.startswith("P step"):
            servo.record_completed_p_step(
                ex,
                servo.pending_command.camera_delta_c[0],
                current_camera_position,
            )
            print(
                "P control completion: "
                f"ex={ex}, abs(ex)={abs(ex)}, "
                f"commanded delta={servo.pending_command.camera_delta_c[0]:.6f} m, "
                "actual camera displacement="
                f"{[round(value, 6) for value in actual_camera_displacement]} m, "
                "servo state=P CONTROL: EX ONLY"
            )
        servo.pending_command = None
        servo.pending_render_frames = 0
        servo.motion_completed = True

    if servo.state == "CALIBRATION: WAITING FOR TARGET":
        arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
        servo.initial_u = u
        servo.initial_ex = ex
        servo.calibration_initial_joint_targets = tuple(
            p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
            for joint_index in arm_joint_indices
        )
        camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
        servo.calibration_initial_camera_position = tuple(camera_position)
        servo.state = "CALIBRATION: +X 2 mm"
        print(
            "Servo State = CALIBRATING / continuous P loop paused; "
            f"u0={u}, Initial ex={ex}, ey={ey}."
        )
        return _request_camera_x_motion(
            servo,
            robot_id,
            "Calibration +X",
            SINGLE_AXIS_CALIBRATION_STEP_METRES,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )

    if servo.state == "CALIBRATION: +X 2 mm":
        servo.plus_x_u = u
        servo.plus_x_ex = ex
        servo.state = "CALIBRATION: RESTORE INITIAL"
        print(f"Calibration +2 mm: u_plus={u}, +2mm ex={ex}, ey={ey}.")
        return _request_calibration_restore(servo, "Calibration restore initial")

    if servo.state == "CALIBRATION: RESTORE INITIAL":
        servo.state = "CALIBRATION: -X 2 mm"
        return _request_camera_x_motion(
            servo,
            robot_id,
            "Calibration -X",
            -SINGLE_AXIS_CALIBRATION_STEP_METRES,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )

    if servo.state == "CALIBRATION: -X 2 mm":
        servo.minus_x_u = u
        servo.minus_x_ex = ex
        assert servo.initial_u is not None
        assert servo.initial_ex is not None
        assert servo.plus_x_u is not None
        assert servo.plus_x_ex is not None
        plus_x_response = servo.plus_x_ex - servo.initial_ex
        minus_x_response = servo.minus_x_ex - servo.initial_ex
        signed_camera_x_response = servo.plus_x_ex - servo.minus_x_ex
        plus_x_abs_error = abs(servo.plus_x_ex)
        minus_x_abs_error = abs(servo.minus_x_ex)
        initial_abs_error = abs(servo.initial_ex)
        print(
            "Calibration result: "
            f"u0={servo.initial_u}, Initial ex={servo.initial_ex}; "
            f"u_plus={servo.plus_x_u}, +2mm ex={servo.plus_x_ex}; "
            f"u_minus={servo.minus_x_u}, -2mm ex={servo.minus_x_ex}."
        )
        if (
            abs(signed_camera_x_response)
            < SINGLE_AXIS_CALIBRATION_MIN_RESPONSE_PIXELS
            or min(plus_x_abs_error, minus_x_abs_error)
            >= initial_abs_error
        ):
            print(
                "Calibration response does not identify a Camera-X direction "
                "that reduces |ex|: FAIL / HOLD."
            )
            servo.request_hold("FAIL: CALIBRATION DIRECTION / HOLD")
            return None

        # The P law below is delta_x = sign * Kp * ex.  The sign is derived
        # from the measured d(ex)/d(Camera-X), never guessed from the setup.
        servo.control_direction = -1 if signed_camera_x_response > 0 else 1
        improving_axis = "+X" if plus_x_abs_error < minus_x_abs_error else "-X"
        print(
            f"Correct direction sign: {servo.control_direction:+d}; "
            f"{improving_axis} reduced |ex| more; "
            f"d(ex)/dX samples: +X={plus_x_response:+d}, "
            f"-X={minus_x_response:+d}."
        )
        servo.state = "CALIBRATION: RESTORE AFTER -X"
        return _request_calibration_restore(servo, "Calibration restore after -X")

    if servo.state == "CALIBRATION: RESTORE AFTER -X":
        servo.state = "P CONTROL: EX ONLY"
        servo.previous_abs_ex = None
        servo.control_started_at = time.monotonic()
        print("Direction Calibration: PASS. Starting horizontal ex-only P control.")

    if servo.state != "P CONTROL: EX ONLY":
        return None

    absolute_ex = abs(ex)
    if absolute_ex < SINGLE_AXIS_SERVO_EX_GOAL_PIXELS:
        convergence_time = (
            time.monotonic() - servo.control_started_at
            if servo.control_started_at is not None
            else 0.0
        )
        print(
            "Servo State = HORIZONTAL CENTERED; "
            f"Initial ex={servo.initial_ex}, Final ex={ex}, "
            f"Convergence time={convergence_time:.2f} s."
        )
        servo.request_hold("HORIZONTAL CENTERED")
        return None
    if (
        servo.initial_ex is not None
        and ex * servo.initial_ex < 0
        and absolute_ex > SINGLE_AXIS_OVERSHOOT_HOLD_PIXELS
    ):
        print(
            "Servo State = OVERSHOOT / HOLD; target crossed the image centre "
            f"by {absolute_ex} px (limit "
            f"{SINGLE_AXIS_OVERSHOOT_HOLD_PIXELS} px). Robot = HOLD."
        )
        servo.request_hold("OVERSHOOT / HOLD")
        return None
    if servo.previous_abs_ex is not None and absolute_ex > servo.previous_abs_ex:
        servo.consecutive_error_increases += 1
        if (
            servo.consecutive_error_increases
            >= SINGLE_AXIS_MAX_CONSECUTIVE_ERROR_INCREASES
        ):
            print(
                "Servo State = DIVERGING; |ex| increased on three consecutive "
                "completed P steps. Robot = HOLD."
            )
            print("Recent 5 horizontal control samples (ex, command, camera position):")
            for history_ex, history_command, history_camera_position in (
                servo.recent_control_history
            ):
                print(
                    f"  ex={history_ex}, command={history_command:.6f} m, "
                    "camera position="
                    f"{[round(value, 6) for value in history_camera_position]}"
                )
            servo.request_hold("DIVERGING")
            return None
    else:
        servo.consecutive_error_increases = 0
    servo.previous_abs_ex = absolute_ex
    if servo.control_steps >= SINGLE_AXIS_MAX_CONTROL_STEPS:
        print("Single-axis control step limit reached: HOLD.")
        servo.request_hold("HOLD: STEP LIMIT")
        return None

    assert servo.control_direction is not None
    raw_delta_x, step_limit, applied_delta_x = calculate_single_axis_adaptive_p_step(
        ex
    )
    commanded_delta_x = servo.control_direction * applied_delta_x
    servo.control_steps += 1
    print(
        f"P control {servo.control_steps}: u={u}, cx={detection.image_center[0]}, "
        f"ex={ex}, abs(ex)={absolute_ex}, "
        f"raw_step={raw_delta_x:.6f} m, step_limit={step_limit:.6f} m, "
        f"applied_step={commanded_delta_x:.6f} m, "
        "servo state=P CONTROL: EX ONLY"
    )
    return _request_camera_x_motion(
        servo,
        robot_id,
        f"P step {servo.control_steps}: ex={ex} px, ey={ey} px",
        commanded_delta_x,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
    )


def update_vertical_visual_servo(
    servo: VerticalVisualServo,
    detection: RedTargetDetection,
    robot_id: int,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand | None:
    """Advance Stage 6B Camera-Y calibration and the ey-only P loop.

    ``ex`` is read and logged solely for visibility.  No state in this
    function uses it to construct a Cartesian, IK, or joint command.
    """
    if servo.is_terminal:
        return None
    if not detection.detected:
        print("Target lost: vertical control paused, HOLD current pose.")
        servo.request_hold("TARGET LOST / HOLD")
        return None

    assert detection.centroid is not None
    assert detection.pixel_error is not None
    u, v = detection.centroid
    ex, ey = detection.pixel_error

    if servo.pending_command is not None:
        active_joint_error = get_active_joint_target_error(
            robot_id,
            servo.pending_command.joint_targets,
            locked_joint_indices,
            client_id,
        )
        servo.pending_render_frames += 1
        if servo.pending_render_frames < SINGLE_AXIS_MIN_WAYPOINT_RENDER_FRAMES:
            return None
        if active_joint_error > SINGLE_AXIS_ACTIVE_JOINT_TARGET_TOLERANCE_RAD:
            if servo.pending_render_frames > SINGLE_AXIS_COMMAND_TIMEOUT_RENDER_FRAMES:
                print(
                    f"{servo.pending_command.label}: vertical control FAIL / HOLD "
                    "(active-joint waypoint timeout, "
                    f"error={active_joint_error:.6f} rad)."
                )
                servo.request_hold("FAIL: ACTIVE WAYPOINT TIMEOUT / HOLD")
            return None
        current_camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
        motion_start_position = [
            target - delta
            for target, delta in zip(
                servo.pending_command.target_camera_position,
                servo.pending_command.camera_delta_w,
            )
        ]
        actual_camera_displacement = [
            actual - start
            for actual, start in zip(
                current_camera_position,
                motion_start_position,
            )
        ]
        print(
            f"{servo.pending_command.label}: reached active-joint waypoint; "
            f"active joint error={active_joint_error:.6f} rad; "
            f"v={v}, ey={ey}, u={u}, ex={ex} (ex is observation only)."
        )
        if servo.pending_command.label.startswith("Vertical P step"):
            servo.record_completed_p_step(
                ey,
                servo.pending_command.camera_delta_c[1],
                current_camera_position,
            )
            print(
                "Vertical P completion: "
                f"ey={ey}, abs(ey)={abs(ey)}, "
                f"commanded delta={servo.pending_command.camera_delta_c[1]:.6f} m, "
                "actual camera displacement="
                f"{[round(value, 6) for value in actual_camera_displacement]} m, "
                "servo state=P CONTROL: EY ONLY"
            )
        servo.pending_command = None
        servo.pending_render_frames = 0
        servo.motion_completed = True

    if servo.state == "CALIBRATION: WAITING FOR TARGET":
        arm_joint_indices = get_panda_arm_joint_indices(robot_id, client_id)
        servo.initial_v = v
        servo.initial_ey = ey
        servo.calibration_initial_joint_targets = tuple(
            p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
            for joint_index in arm_joint_indices
        )
        camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
        servo.calibration_initial_camera_position = tuple(camera_position)
        servo.state = "CALIBRATION: +Y 2 mm"
        print(
            "Servo State = VERTICAL CALIBRATING / continuous P loop paused; "
            f"Initial v={v}, Initial ey={ey}, ex={ex} (ex is observation only)."
        )
        return _request_camera_y_motion(
            servo,
            robot_id,
            "Vertical calibration +Y",
            SINGLE_AXIS_CALIBRATION_STEP_METRES,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )

    if servo.state == "CALIBRATION: +Y 2 mm":
        servo.plus_y_v = v
        servo.plus_y_ey = ey
        servo.state = "CALIBRATION: RESTORE INITIAL"
        print(f"Vertical calibration +2 mm: v_plus={v}, +2mm ey={ey}, ex={ex}.")
        return _request_calibration_restore(servo, "Vertical calibration restore initial")

    if servo.state == "CALIBRATION: RESTORE INITIAL":
        servo.state = "CALIBRATION: -Y 2 mm"
        return _request_camera_y_motion(
            servo,
            robot_id,
            "Vertical calibration -Y",
            -SINGLE_AXIS_CALIBRATION_STEP_METRES,
            locked_joint_indices,
            locked_initial_positions,
            client_id,
        )

    if servo.state == "CALIBRATION: -Y 2 mm":
        servo.minus_y_v = v
        servo.minus_y_ey = ey
        assert servo.initial_v is not None
        assert servo.initial_ey is not None
        assert servo.plus_y_v is not None
        assert servo.plus_y_ey is not None
        plus_y_response = servo.plus_y_ey - servo.initial_ey
        minus_y_response = servo.minus_y_ey - servo.initial_ey
        signed_camera_y_response = servo.plus_y_ey - servo.minus_y_ey
        plus_y_abs_error = abs(servo.plus_y_ey)
        minus_y_abs_error = abs(servo.minus_y_ey)
        initial_abs_error = abs(servo.initial_ey)
        print(
            "Vertical calibration result: "
            f"v0={servo.initial_v}, Initial ey={servo.initial_ey}; "
            f"v_plus={servo.plus_y_v}, +2mm ey={servo.plus_y_ey}; "
            f"v_minus={servo.minus_y_v}, -2mm ey={servo.minus_y_ey}."
        )
        if (
            abs(signed_camera_y_response)
            < SINGLE_AXIS_CALIBRATION_MIN_RESPONSE_PIXELS
            or min(plus_y_abs_error, minus_y_abs_error)
            >= initial_abs_error
        ):
            print(
                "Vertical calibration response does not identify a Camera-Y "
                "direction that reduces |ey|: FAIL / HOLD."
            )
            servo.request_hold("FAIL: CALIBRATION DIRECTION / HOLD")
            return None

        # delta_y = sign * Kp_y * ey. The sign comes only from the measured
        # RGB response d(ey)/d(Camera-Y), never from an assumed frame sign.
        servo.control_direction = -1 if signed_camera_y_response > 0 else 1
        improving_axis = "+Y" if plus_y_abs_error < minus_y_abs_error else "-Y"
        print(
            f"Vertical correct direction sign: {servo.control_direction:+d}; "
            f"{improving_axis} reduced |ey| more; "
            f"d(ey)/dY samples: +Y={plus_y_response:+d}, "
            f"-Y={minus_y_response:+d}."
        )
        servo.state = "CALIBRATION: RESTORE AFTER -Y"
        return _request_calibration_restore(servo, "Vertical calibration restore after -Y")

    if servo.state == "CALIBRATION: RESTORE AFTER -Y":
        servo.state = "P CONTROL: EY ONLY"
        servo.previous_abs_ey = None
        servo.control_started_at = time.monotonic()
        print("Vertical Direction Calibration: PASS. Starting ey-only P control.")

    if servo.state != "P CONTROL: EY ONLY":
        return None

    absolute_ey = abs(ey)
    if absolute_ey < VERTICAL_SERVO_GOAL_PIXELS:
        convergence_time = (
            time.monotonic() - servo.control_started_at
            if servo.control_started_at is not None
            else 0.0
        )
        print(
            "Servo State = VERTICAL CENTERED; "
            f"Initial ey={servo.initial_ey}, Final ey={ey}, "
            f"Convergence time={convergence_time:.2f} s."
        )
        servo.request_hold("VERTICAL CENTERED")
        return None
    if (
        servo.initial_ey is not None
        and ey * servo.initial_ey < 0
        and absolute_ey > SINGLE_AXIS_OVERSHOOT_HOLD_PIXELS
    ):
        print(
            "Servo State = OVERSHOOT / HOLD; target crossed the image centre "
            f"by {absolute_ey} px (limit "
            f"{SINGLE_AXIS_OVERSHOOT_HOLD_PIXELS} px). Robot = HOLD."
        )
        servo.request_hold("OVERSHOOT / HOLD")
        return None
    if servo.previous_abs_ey is not None and absolute_ey > servo.previous_abs_ey:
        servo.consecutive_error_increases += 1
        if servo.consecutive_error_increases >= SINGLE_AXIS_MAX_CONSECUTIVE_ERROR_INCREASES:
            print(
                "Servo State = DIVERGING; |ey| increased on three consecutive "
                "completed P steps. Robot = HOLD."
            )
            print("Recent 5 vertical control samples (ey, command, camera position):")
            for history_ey, history_command, history_camera_position in (
                servo.recent_control_history
            ):
                print(
                    f"  ey={history_ey}, command={history_command:.6f} m, "
                    "camera position="
                    f"{[round(value, 6) for value in history_camera_position]}"
                )
            servo.request_hold("DIVERGING")
            return None
    else:
        servo.consecutive_error_increases = 0
    servo.previous_abs_ey = absolute_ey
    if servo.control_steps >= SINGLE_AXIS_MAX_CONTROL_STEPS:
        print("Vertical control step limit reached: HOLD.")
        servo.request_hold("HOLD: STEP LIMIT")
        return None

    assert servo.control_direction is not None
    raw_delta_y, step_limit, applied_delta_y = calculate_vertical_adaptive_p_step(ey)
    commanded_delta_y = servo.control_direction * applied_delta_y
    servo.control_steps += 1
    print(
        f"Vertical P control {servo.control_steps}: v={v}, cy={detection.image_center[1]}, "
        f"ey={ey}, abs(ey)={absolute_ey}, ex={ex} (observation only), "
        f"raw_step={raw_delta_y:.6f} m, step_limit={step_limit:.6f} m, "
        f"applied_step={commanded_delta_y:.6f} m, "
        "servo state=P CONTROL: EY ONLY"
    )
    return _request_camera_y_motion(
        servo,
        robot_id,
        f"Vertical P step {servo.control_steps}: ey={ey} px, ex={ex} px",
        commanded_delta_y,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
    )


def _quaternion_distance_degrees(
    first_orientation: Sequence[float],
    second_orientation: Sequence[float],
) -> float:
    """Return the smallest rotation angle between two unit quaternions."""
    dot_product = abs(
        sum(first * second for first, second in zip(first_orientation, second_orientation))
    )
    bounded_dot_product = max(-1.0, min(1.0, dot_product))
    return degrees(2.0 * acos(bounded_dot_product))


def update_two_dimensional_visual_servo(
    servo: TwoDimensionalVisualServo,
    detection: RedTargetDetection,
    robot_id: int,
    locked_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    client_id: int,
) -> CameraAxisMotionCommand | None:
    """Advance Stage 6C's current-pose-relative calibrated XY P controller."""
    if servo.is_terminal:
        return None
    if not detection.detected:
        print("Target lost: 2D control paused, HOLD current pose.")
        servo.last_delta_x = 0.0
        servo.last_delta_y = 0.0
        servo.request_hold("TARGET LOST / HOLD")
        return None

    assert detection.centroid is not None
    assert detection.pixel_error is not None
    u, v = detection.centroid
    ex, ey = detection.pixel_error
    error_norm = sqrt(ex * ex + ey * ey)
    servo.last_ex = ex
    servo.last_ey = ey
    servo.last_error_norm = error_norm

    if servo.pending_command is not None:
        active_joint_error = get_active_joint_target_error(
            robot_id,
            servo.pending_command.joint_targets,
            locked_joint_indices,
            client_id,
        )
        servo.pending_render_frames += 1
        if servo.pending_render_frames < SINGLE_AXIS_MIN_WAYPOINT_RENDER_FRAMES:
            return None
        if active_joint_error > SINGLE_AXIS_ACTIVE_JOINT_TARGET_TOLERANCE_RAD:
            if servo.pending_render_frames > SINGLE_AXIS_COMMAND_TIMEOUT_RENDER_FRAMES:
                print(
                    f"{servo.pending_command.label}: 2D control FAIL / HOLD "
                    "(active-joint waypoint timeout, "
                    f"error={active_joint_error:.6f} rad)."
                )
                servo.request_hold("FAIL: ACTIVE WAYPOINT TIMEOUT / HOLD")
            return None
        current_camera_position, current_camera_orientation = get_camera_optical_center_pose(
            robot_id,
            client_id,
        )
        motion_start_position = [
            target - delta
            for target, delta in zip(
                servo.pending_command.target_camera_position,
                servo.pending_command.camera_delta_w,
            )
        ]
        actual_camera_displacement = [
            actual - start
            for actual, start in zip(
                current_camera_position,
                motion_start_position,
            )
        ]
        orientation_step_degrees = 0.0
        if servo.pending_camera_orientation is not None:
            orientation_step_degrees = _quaternion_distance_degrees(
                servo.pending_camera_orientation,
                current_camera_orientation,
            )
            servo.max_camera_orientation_step_degrees = max(
                servo.max_camera_orientation_step_degrees,
                orientation_step_degrees,
            )
            if orientation_step_degrees > TWO_D_CAMERA_ORIENTATION_STEP_LIMIT_DEGREES:
                servo.orientation_stable = False
                print(
                    "Camera orientation warning: one Stage 6C waypoint changed "
                    f"world camera orientation by {orientation_step_degrees:.3f} deg "
                    f"(limit {TWO_D_CAMERA_ORIENTATION_STEP_LIMIT_DEGREES:.1f} deg)."
                )
        print(
            f"{servo.pending_command.label}: reached active-joint waypoint; "
            f"active joint error={active_joint_error:.6f} rad; "
            f"ex={ex}, ey={ey}, error_norm={error_norm:.3f}; "
            "actual camera displacement="
            f"{[round(value, 6) for value in actual_camera_displacement]} m; "
            f"camera orientation step={orientation_step_degrees:.3f} deg."
        )
        servo.record_completed_p_step(
            ex,
            ey,
            error_norm,
            servo.pending_command.camera_delta_c[0],
            servo.pending_command.camera_delta_c[1],
            current_camera_position,
        )
        servo.pending_command = None
        servo.pending_camera_orientation = None
        servo.pending_render_frames = 0
        servo.motion_completed = True

    if servo.state == "2D WAITING FOR TARGET":
        servo.initial_pixel = (u, v)
        servo.initial_ex = ex
        servo.initial_ey = ey
        servo.initial_error_norm = error_norm
        servo.previous_error_norm = None
        servo.control_started_at = time.monotonic()
        servo.state = "2D TRACKING"
        print(
            "Stage 6C start from CURRENT pose: "
            f"target=({u}, {v}), ex={ex}, ey={ey}, error_norm={error_norm:.3f}; "
            f"using verified signs X={servo.horizontal_sign:+d}, "
            f"Y={servo.vertical_sign:+d}."
        )

    if servo.state != "2D TRACKING":
        return None

    if abs(ex) < TWO_D_TOL_X_PIXELS and abs(ey) < TWO_D_TOL_Y_PIXELS:
        convergence_time = (
            time.monotonic() - servo.control_started_at
            if servo.control_started_at is not None
            else 0.0
        )
        servo.last_delta_x = 0.0
        servo.last_delta_y = 0.0
        print(
            "Servo State = 2D CENTERED; "
            f"Initial (ex, ey)=({servo.initial_ex}, {servo.initial_ey}), "
            f"Final (ex, ey)=({ex}, {ey}), error_norm={error_norm:.3f}, "
            f"Convergence time={convergence_time:.2f} s."
        )
        servo.request_hold("2D CENTERED")
        return None

    if (
        servo.previous_error_norm is not None
        and error_norm
        > servo.previous_error_norm + TWO_D_ERROR_NORM_INCREASE_TOLERANCE_PIXELS
    ):
        servo.consecutive_norm_increases += 1
        if (
            servo.consecutive_norm_increases >= TWO_D_MAX_CONSECUTIVE_NORM_INCREASES
            and len(servo.recent_control_history) >= 5
        ):
            print(
                "Servo State = DIVERGING; 2D error norm increased materially on "
                "three consecutive completed control periods. Robot = HOLD."
            )
            print("Recent 5 Stage 6C samples (ex, ey, norm, dx, dy, camera position):")
            for sample_ex, sample_ey, sample_norm, sample_dx, sample_dy, sample_camera in (
                servo.recent_control_history
            ):
                print(
                    f"  ex={sample_ex}, ey={sample_ey}, norm={sample_norm:.3f}, "
                    f"dx={sample_dx:.6f} m, dy={sample_dy:.6f} m, "
                    "camera position="
                    f"{[round(value, 6) for value in sample_camera]}"
                )
            servo.last_delta_x = 0.0
            servo.last_delta_y = 0.0
            servo.request_hold("DIVERGING")
            return None
    else:
        servo.consecutive_norm_increases = 0
    servo.previous_error_norm = error_norm

    if servo.control_steps >= SINGLE_AXIS_MAX_CONTROL_STEPS:
        print("Stage 6C control step limit reached: HOLD.")
        servo.last_delta_x = 0.0
        servo.last_delta_y = 0.0
        servo.request_hold("HOLD: STEP LIMIT")
        return None

    if abs(ex) < TWO_D_TOL_X_PIXELS:
        raw_delta_x = 0.0
        applied_delta_x = 0.0
    else:
        raw_delta_x, _, applied_delta_x = calculate_single_axis_adaptive_p_step(ex)
    if abs(ey) < TWO_D_TOL_Y_PIXELS:
        raw_delta_y = 0.0
        applied_delta_y = 0.0
    else:
        raw_delta_y, _, applied_delta_y = calculate_vertical_adaptive_p_step(ey)

    commanded_delta_x = servo.horizontal_sign * applied_delta_x
    commanded_delta_y = servo.vertical_sign * applied_delta_y
    resultant_step = sqrt(
        commanded_delta_x * commanded_delta_x
        + commanded_delta_y * commanded_delta_y
    )
    if resultant_step > TWO_D_MAX_RESULTANT_STEP_METRES:
        resultant_scale = TWO_D_MAX_RESULTANT_STEP_METRES / resultant_step
        commanded_delta_x *= resultant_scale
        commanded_delta_y *= resultant_scale
    servo.last_delta_x = commanded_delta_x
    servo.last_delta_y = commanded_delta_y
    servo.control_steps += 1
    current_camera_position, current_camera_orientation = get_camera_optical_center_pose(
        robot_id,
        client_id,
    )
    print(
        f"2D control {servo.control_steps}: target=({u}, {v}), center="
        f"{detection.image_center}, ex={ex}, ey={ey}, error_norm={error_norm:.3f}, "
        f"raw_dx={raw_delta_x:.6f} m, raw_dy={raw_delta_y:.6f} m, "
        f"dx={commanded_delta_x:.6f} m, dy={commanded_delta_y:.6f} m, "
        f"resultant={min(resultant_step, TWO_D_MAX_RESULTANT_STEP_METRES):.6f} m, "
        "servo state=2D TRACKING; Camera orientation="
        f"{[round(value, 6) for value in current_camera_orientation]}."
    )
    return _request_camera_xy_motion(
        servo,
        robot_id,
        f"2D P step {servo.control_steps}: ex={ex} px, ey={ey} px",
        commanded_delta_x,
        commanded_delta_y,
        locked_joint_indices,
        locked_initial_positions,
        client_id,
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


def get_current_arm_hold_targets(
    robot_id: int,
    arm_joint_indices: Sequence[int],
    client_id: int,
) -> list[float]:
    """Capture the current joint pose for an active-joint HOLD command.

    The constrained controller still overrides locked joints with their
    immutable startup targets. Active joints use these live measurements, so
    a stopped visual-servo command holds its reached pose rather than home.
    """
    return [
        p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        for joint_index in arm_joint_indices
    ]


def capture_active_joint_hold_targets(
    robot_id: int,
    active_joint_indices: Sequence[int],
    client_id: int,
) -> dict[int, float]:
    """Save the active joints' actual reached angles as their HOLD targets."""
    return {
        joint_index: p.getJointState(
            robot_id,
            joint_index,
            physicsClientId=client_id,
        )[0]
        for joint_index in active_joint_indices
    }


def build_hold_joint_targets(
    arm_joint_indices: Sequence[int],
    locked_initial_positions: dict[int, float],
    active_hold_targets: dict[int, float],
) -> list[float]:
    """Build the seven-joint HOLD target without using active startup angles."""
    targets = []
    for joint_index in arm_joint_indices:
        if joint_index in locked_initial_positions:
            targets.append(locked_initial_positions[joint_index])
        else:
            try:
                targets.append(active_hold_targets[joint_index])
            except KeyError as error:
                raise RuntimeError(
                    f"Missing current HOLD target for active joint {joint_index}."
                ) from error
    return targets


def print_active_joint_hold_status(
    servo_state: str,
    robot_id: int,
    active_joint_indices: Sequence[int],
    initial_arm_positions: dict[int, float],
    active_hold_targets: dict[int, float],
    commanded_joint_targets: Sequence[float],
    arm_joint_indices: Sequence[int],
    client_id: int,
) -> None:
    """Report the live active-joint HOLD source and detect stale home targets."""
    target_by_joint = dict(zip(arm_joint_indices, commanded_joint_targets))
    print(f"Servo State: {servo_state}")
    for joint_index in active_joint_indices:
        joint_name = _decode_name(
            p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[1]
        )
        current_position = p.getJointState(
            robot_id,
            joint_index,
            physicsClientId=client_id,
        )[0]
        initial_position = initial_arm_positions[joint_index]
        hold_target = active_hold_targets[joint_index]
        commanded_target = target_by_joint[joint_index]
        print(
            f"  {joint_name}: current q={current_position:.6f}, "
            f"commanded target q={commanded_target:.6f}, "
            f"initial q={initial_position:.6f}, "
            f"hold target q={hold_target:.6f}"
        )
        if (
            abs(commanded_target - initial_position) < 1e-9
            and abs(hold_target - initial_position) >= 1e-9
        ):
            print(
                "  ERROR: active HOLD target was overwritten by initial q; "
                "inspect the main-loop command source."
            )


def print_visual_servo_current_pose_takeover(
    robot_id: int,
    arm_joint_indices: Sequence[int],
    initial_arm_positions: dict[int, float],
    detection: RedTargetDetection,
    first_command: CameraAxisMotionCommand,
    client_id: int,
    controlled_axis: str = "ex",
) -> bool:
    """Print evidence that the first P target was derived from current C pose."""
    current_joint_positions = {
        _decode_name(p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[1]):
        p.getJointState(robot_id, joint_index, physicsClientId=client_id)[0]
        for joint_index in arm_joint_indices
    }
    initial_joint_positions = {
        _decode_name(p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[1]):
        initial_arm_positions[joint_index]
        for joint_index in arm_joint_indices
    }
    current_end_effector_position, _ = get_camera_reference_pose(robot_id, client_id)
    current_camera_position, _ = get_camera_optical_center_pose(robot_id, client_id)
    command_source_camera_position = [
        target - delta
        for target, delta in zip(
            first_command.target_camera_position,
            first_command.camera_delta_w,
        )
    ]
    source_error = sqrt(
        sum(
            (current - source) ** 2
            for current, source in zip(
                current_camera_position,
                command_source_camera_position,
            )
        )
    )
    source_is_current_pose = source_error < 1e-8
    active_joint_target = {
        _decode_name(p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[1]):
        target_position
        for joint_index, target_position in zip(arm_joint_indices, first_command.joint_targets)
        if _decode_name(p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)[1])
        not in BASELINE_LOCKED_JOINT_NAMES
    }
    assert detection.centroid is not None
    assert detection.pixel_error is not None
    print("\nVisual Servo first takeover:")
    print("  Current q:", {name: round(value, 6) for name, value in current_joint_positions.items()})
    print("  Initial q:", {name: round(value, 6) for name, value in initial_joint_positions.items()})
    print("  Current EE position:", [round(value, 6) for value in current_end_effector_position])
    print("  Current Camera position:", [round(value, 6) for value in current_camera_position])
    print("  Target pixel:", detection.centroid)
    if controlled_axis not in {"ex", "ey", "2D"}:
        raise ValueError("controlled_axis must be 'ex', 'ey', or '2D'.")
    if controlled_axis == "2D":
        print("  ex, ey:", detection.pixel_error)
    else:
        controlled_error = detection.pixel_error[0 if controlled_axis == "ex" else 1]
        print(f"  {controlled_axis}:", controlled_error)
    print(
        "  First servo target: Camera position=",
        [round(value, 6) for value in first_command.target_camera_position],
        "; active joint targets=",
        {name: round(value, 6) for name, value in active_joint_target.items()},
    )
    print("  Visual Servo Start Source: CURRENT POSE")
    print(
        "  First Servo Command Based On Current Pose:",
        "PASS" if source_is_current_pose else "FAIL",
        f"(source error={source_error:.12f} m)",
    )
    return source_is_current_pose


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
    servo_state: str = "INITIALIZING / HOLD",
) -> RedTargetDetection | None:
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
        return None
    live_frame = render_live_eye_in_hand_rgb_frame(
        camera_position,
        camera_orientation,
        client_id,
    )
    red_detection = detect_red_target_from_live_rgb(live_frame)
    camera_display.show(
        live_frame,
        add_servo_state_overlay(
            live_frame,
            red_detection.annotated_rgba_buffer,
            servo_state,
        ),
    )
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
    return red_detection


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
    """Run the selected one-axis Stage 6 visual-servo loop without home motion.

    Startup and the continuous loop use the accepted constrained
    POSITION_CONTROL baseline to hold locked joints and the current active pose.
    The selected controller consumes one live pixel-error component and
    supplies only incremental Cartesian targets from the current camera pose.
    """
    if VISUAL_SERVO_MODE not in {"HORIZONTAL_EX", "VERTICAL_EY", "TWO_D"}:
        raise RuntimeError(
            "VISUAL_SERVO_MODE must be 'HORIZONTAL_EX', 'VERTICAL_EY', or 'TWO_D'."
        )
    two_dimensional_mode = VISUAL_SERVO_MODE == "TWO_D"
    vertical_mode = VISUAL_SERVO_MODE == "VERTICAL_EY"
    stage_name = "Stage 6C" if two_dimensional_mode else "Stage 6B" if vertical_mode else "Stage 6A"
    controlled_axis = "2D" if two_dimensional_mode else "ey" if vertical_mode else "ex"
    candidate_axis_name = (
        "Camera-X + Camera-Y"
        if two_dimensional_mode
        else "Camera-Y"
        if vertical_mode
        else "Camera-X"
    )
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
        active_joint_indices = [
            joint_index
            for joint_index in arm_joint_indices
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
            "Initializing: Robot HOLD at CURRENT pose | Camera ACTIVE",
            None,
            client_id,
        )
        print("Initialization phase:")
        print("Robot: HOLD at CURRENT pose")
        print("Eye-in-Hand Camera: ACTIVE")
        print("Startup initial-pose motion: DISABLED")
        print("Waiting only for the first valid Eye-in-Hand RGB detection.")

        debug_text_id = update_motion_debug_text(
            (
                "Stage 6C: 2D tracking waiting for RGB target"
                if two_dimensional_mode
                else f"{stage_name}: {candidate_axis_name} calibration waiting for RGB target"
            ),
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
            (
                "Stage 6C uses ex + ey only with the previously calibrated "
                "Camera-X/Y signs; no new axis sign is guessed."
                if two_dimensional_mode
                else "Stage 6B uses ey only. ex remains visible for observation and is "
                "never used to create a robot command."
                if vertical_mode
                else "Stage 6A uses ex only. ey remains visible for observation and is "
                "never used to create a robot command."
            )
        )
        if not two_dimensional_mode:
            print(
                f"If {stage_name} is safely HOLDing after DIVERGING/TARGET LOST, press R "
                f"to calibrate {candidate_axis_name} again from the current active-joint pose."
            )
        print("Close the PyBullet GUI window to finish the example.")

        # The default path never commands startup active-joint targets. From
        # the first control step onward, active HOLD targets are measured q.
        active_hold_targets = capture_active_joint_hold_targets(
            robot_id,
            active_joint_indices,
            client_id,
        )
        commanded_joint_targets = build_hold_joint_targets(
            arm_joint_indices,
            locked_initial_positions,
            active_hold_targets,
        )
        initial_servo_state = (
            "CALIBRATION: WAITING FOR TARGET"
            if VISUAL_SERVO_MOTION_ENABLED
            else "HOLD: VISUAL SERVO PAUSED"
        )
        single_axis_servo: (
            SingleAxisVisualServo | VerticalVisualServo | TwoDimensionalVisualServo
        ) = (
            TwoDimensionalVisualServo(
                state=(
                    "2D WAITING FOR TARGET"
                    if VISUAL_SERVO_MOTION_ENABLED
                    else "HOLD: VISUAL SERVO PAUSED"
                )
            )
            if two_dimensional_mode
            else VerticalVisualServo(state=initial_servo_state)
            if vertical_mode
            else SingleAxisVisualServo(state=initial_servo_state)
        )
        simulation_step = 0
        status_interval_steps = max(
            1,
            round(CONTROL_STATUS_INTERVAL_SECONDS / TIME_STEP),
        )
        while p.isConnected(client_id):
            if (
                VISUAL_SERVO_MOTION_ENABLED
                and not two_dimensional_mode
                and single_axis_servo.is_terminal
                and (
                    vertical_recalibration_requested(client_id)
                    if vertical_mode
                    else horizontal_recalibration_requested(client_id)
                )
            ):
                # Do not reset any Panda joint.  The current reached active
                # pose becomes HOLD while a fresh local +/-2 mm sign test is
                # performed from that exact pose.
                active_hold_targets = capture_active_joint_hold_targets(
                    robot_id,
                    active_joint_indices,
                    client_id,
                )
                commanded_joint_targets = build_hold_joint_targets(
                    arm_joint_indices,
                    locked_initial_positions,
                    active_hold_targets,
                )
                single_axis_servo = VerticalVisualServo() if vertical_mode else SingleAxisVisualServo()
                debug_text_id = update_motion_debug_text(
                    f"{stage_name}: recalibrating {candidate_axis_name} from current HOLD pose",
                    debug_text_id,
                    client_id,
                )
                print(
                    f"{stage_name} recalibration requested: Robot HOLD; "
                    "continuous P loop paused for current-pose sign calibration."
                )
            # In PAUSED/HOLD, active_hold_targets is intentionally immutable:
            # it was captured after initialization (or when the servo entered
            # HOLD).  Every control step below therefore holds that reached
            # pose, never the program-start active-joint angles.
            # Before the first RGB-calibration result this is a pure HOLD. Once
            # calibrated, only the selected RGB-derived Camera command may
            # replace these active-joint targets.
            apply_constrained_arm_position_control(
                robot_id,
                arm_joint_indices,
                commanded_joint_targets,
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
            detection = update_live_eye_in_hand_camera(
                robot_id,
                simulation_step,
                client_id,
                camera_display,
                get_visual_servo_overlay_text(single_axis_servo),
            )
            if detection is not None and VISUAL_SERVO_MOTION_ENABLED:
                previous_servo_state = single_axis_servo.state
                if two_dimensional_mode:
                    assert isinstance(single_axis_servo, TwoDimensionalVisualServo)
                    new_command = update_two_dimensional_visual_servo(
                        single_axis_servo,
                        detection,
                        robot_id,
                        locked_joint_indices,
                        locked_initial_positions,
                        client_id,
                    )
                elif vertical_mode:
                    assert isinstance(single_axis_servo, VerticalVisualServo)
                    new_command = update_vertical_visual_servo(
                        single_axis_servo,
                        detection,
                        robot_id,
                        locked_joint_indices,
                        locked_initial_positions,
                        client_id,
                    )
                else:
                    assert isinstance(single_axis_servo, SingleAxisVisualServo)
                    new_command = update_single_axis_visual_servo(
                        single_axis_servo,
                        detection,
                        robot_id,
                        locked_joint_indices,
                        locked_initial_positions,
                        client_id,
                    )
                if single_axis_servo.motion_completed:
                    # A completed calibration/P waypoint becomes the newest
                    # active HOLD pose before another command is considered.
                    # The saved values come from measured joint state, never
                    # from the program-start active-joint angles.
                    active_hold_targets = capture_active_joint_hold_targets(
                        robot_id,
                        active_joint_indices,
                        client_id,
                    )
                    single_axis_servo.motion_completed = False
                if new_command is not None:
                    if (
                        (
                            new_command.label.startswith("2D P step")
                            if two_dimensional_mode
                            else new_command.label.startswith("Vertical P step")
                            if vertical_mode
                            else new_command.label.startswith("P step")
                        )
                        and not single_axis_servo.first_servo_command_logged
                    ):
                        current_pose_source_ok = print_visual_servo_current_pose_takeover(
                            robot_id,
                            arm_joint_indices,
                            initial_arm_positions,
                            detection,
                            new_command,
                            client_id,
                            controlled_axis=controlled_axis,
                        )
                        single_axis_servo.first_servo_command_logged = True
                        if not current_pose_source_ok:
                            print(
                                "FAIL: first Visual Servo target was not derived "
                                "from the current camera pose. Robot HOLD."
                            )
                            single_axis_servo.request_hold(
                                "FAIL: FIRST TARGET NOT CURRENT POSE / HOLD"
                            )
                    if not single_axis_servo.hold_requested:
                        commanded_joint_targets = list(new_command.joint_targets)
                if single_axis_servo.hold_requested:
                    active_hold_targets = capture_active_joint_hold_targets(
                        robot_id,
                        active_joint_indices,
                        client_id,
                    )
                    commanded_joint_targets = build_hold_joint_targets(
                        arm_joint_indices,
                        locked_initial_positions,
                        active_hold_targets,
                    )
                    single_axis_servo.hold_requested = False
                if single_axis_servo.state != previous_servo_state:
                    debug_text_id = update_motion_debug_text(
                        single_axis_servo.state,
                        debug_text_id,
                        client_id,
                    )
            simulation_step += 1
            if simulation_step % status_interval_steps == 0:
                report_phase = get_locked_joint_report_phase(single_axis_servo)
                if report_phase == "HOLD":
                    print_active_joint_hold_status(
                        single_axis_servo.state,
                        robot_id,
                        active_joint_indices,
                        initial_arm_positions,
                        active_hold_targets,
                        commanded_joint_targets,
                        arm_joint_indices,
                        client_id,
                    )
                print_locked_joint_status(
                    report_phase,
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
