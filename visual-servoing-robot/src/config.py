"""Read-only project configuration inventory.

Authoritative definitions remain in their owning core modules.  This facade
returns those live values for audits and documentation, avoiding a second
copy/source of truth during the behavior-preserving Stage 22 refactor.
"""
from __future__ import annotations

from typing import Any

# Shared stable values previously repeated by integration/evaluation wrappers.
# Core camera/control values remain authoritative in their long-established
# owner modules and are exposed through ``parameter_snapshot`` below.
PREDICTION_HORIZON_S = 0.30
OBSTACLE_DIMENSIONS_M = (0.10, 0.10, 0.16)
OBSTACLE_SAFETY_MARGIN_M = 0.020
OBSTACLE_RGBA = (1.0, 0.82, 0.0, 1.0)
COLLISION_INTERPOLATION_RESOLUTION_RAD = 0.05
RRT_STEP_SIZE_RAD = 0.15
RRT_GOAL_BIAS_PROBABILITY = 0.10
RRT_MAX_ITERATIONS = 2500
RRT_MAX_PLANNING_TIME_S = 5.0
EXECUTION_RESOLUTION_RAD = 0.01
COMMAND_SPEED_RAD_S = 0.04
WAYPOINT_TOLERANCE_RAD = 0.0005
WAYPOINT_TIMEOUT_S = 5.0


def parameter_snapshot() -> dict[str, Any]:
    import robotics_core as sim
    import camera_observation as camera
    import predictive_measurement as prediction
    return {
        "Kp": sim.SINGLE_AXIS_SERVO_KP_METRES_PER_PIXEL,
        "prediction_alpha": prediction.PREDICTION_ALPHA,
        "prediction_tau_s": PREDICTION_HORIZON_S,
        "base_max_step_m": sim.SINGLE_AXIS_SERVO_MAX_STEP_METRES,
        "deadband_x_px": sim.TWO_D_TOL_X_PIXELS,
        "deadband_y_px": sim.TWO_D_TOL_Y_PIXELS,
        "camera_width_px": camera.CAMERA_IMAGE_WIDTH,
        "camera_height_px": camera.CAMERA_IMAGE_HEIGHT,
        "camera_fov_y_deg": camera.CAMERA_FOV_Y_DEGREES,
        "camera_near_m": camera.CAMERA_NEAR_PLANE,
        "camera_far_m": camera.CAMERA_FAR_PLANE,
        "T_E_C_position": sim.T_E_C_POSITION,
        "T_E_C_orientation": sim.T_E_C_ORIENTATION,
        "locked_joint_names": sim.BASELINE_LOCKED_JOINT_NAMES,
        "active_joint_definition": "Panda arm joints excluding locked_joint_names",
        "position_gain": sim.POSITION_GAIN,
        "velocity_gain": sim.IK_VELOCITY_GAIN,
        "motor_force": sim.MAX_IK_MOTOR_FORCE,
        "max_velocity": sim.MAX_IK_JOINT_VELOCITY,
        "occupancy_margin_m": OBSTACLE_SAFETY_MARGIN_M,
        "collision_resolution_rad": COLLISION_INTERPOLATION_RESOLUTION_RAD,
        "rrt_step_rad": RRT_STEP_SIZE_RAD,
        "rrt_goal_bias": RRT_GOAL_BIAS_PROBABILITY,
        "rrt_max_iterations": RRT_MAX_ITERATIONS,
        "rrt_max_time_s": RRT_MAX_PLANNING_TIME_S,
        "execution_resolution_rad": EXECUTION_RESOLUTION_RAD,
        "execution_waypoint_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
    }
