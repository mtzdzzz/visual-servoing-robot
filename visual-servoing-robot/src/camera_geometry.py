"""Authoritative Eye-in-Hand camera pose and render-frame geometry.

This module contains the Stage 14-validated transforms.  It deliberately has
no target, obstacle, controller, or ground-truth inputs.
"""
from __future__ import annotations

from math import acos, degrees
from typing import Sequence

import numpy as np
import pybullet as p

import robotics_core as sim
from camera_observation import LiveCameraFrame


def matrix_from_pose(position: Sequence[float], orientation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(p.getMatrixFromQuaternion(orientation), dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def compose_camera_to_world(robot_id: int, client_id: int) -> tuple[np.ndarray, list[float], list[float]]:
    """Return T_W_C = T_W_E @ T_E_C using only the live robot pose."""
    end_position, end_orientation = sim.get_end_effector_reference_pose(robot_id, client_id)
    camera_position, camera_orientation = p.multiplyTransforms(
        end_position, end_orientation, sim.T_E_C_POSITION, sim.T_E_C_ORIENTATION
    )
    return matrix_from_pose(camera_position, camera_orientation), list(camera_position), list(camera_orientation)


def _quaternion_angle_degrees(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(first, second)))
    return degrees(2.0 * acos(min(1.0, max(-1.0, dot))))


def validate_camera_pose_and_render_frame(
    robot_id: int, live_frame: LiveCameraFrame, client_id: int
) -> tuple[np.ndarray, float, float]:
    camera_to_world, composed_position, composed_orientation = compose_camera_to_world(robot_id, client_id)
    position_error = float(np.linalg.norm(np.asarray(composed_position) - np.asarray(live_frame.camera_world_position)))
    orientation_error = _quaternion_angle_degrees(composed_orientation, live_frame.camera_world_orientation)
    if position_error > 1e-6 or orientation_error > 1e-3:
        raise RuntimeError("Camera-pose audit failed: live render pose is not T_W_E @ T_E_C.")
    return camera_to_world, position_error, orientation_error


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise RuntimeError("Camera render frame contains a zero-length axis.")
    return vector / length


def render_to_camera_axis_transform(camera_to_world: np.ndarray, live_frame: LiveCameraFrame) -> np.ndarray:
    """Return the exact C_render -> physical C axis transform used by Stage 14."""
    eye = np.asarray(live_frame.render_parameters.eye_position, dtype=np.float64)
    target = np.asarray(live_frame.render_parameters.target_position, dtype=np.float64)
    forward = _normalize(target - eye)
    up_input = _normalize(np.asarray(live_frame.render_parameters.up_vector, dtype=np.float64))
    render_right = _normalize(np.cross(forward, up_input))
    render_up = _normalize(np.cross(render_right, forward))
    render_to_world = np.column_stack((render_right, render_up, forward))
    return camera_to_world[:3, :3].T @ render_to_world
