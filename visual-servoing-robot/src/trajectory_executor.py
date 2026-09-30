"""Non-blocking, motor-only execution of a validated path (no PyBullet resets).

The caller supplies CURRENT measurements and the existing motor interface.
Safety failures latch; neither a restart nor a return-home command exists here.
"""
from __future__ import annotations

import numpy as np
import config

EXECUTION_RESOLUTION_RAD = config.EXECUTION_RESOLUTION_RAD
COMMAND_SPEED_RAD_S = config.COMMAND_SPEED_RAD_S
WAYPOINT_TOLERANCE_RAD = config.WAYPOINT_TOLERANCE_RAD
WAYPOINT_TIMEOUT_S = config.WAYPOINT_TIMEOUT_S


def densify_path(path, resolution=EXECUTION_RESOLUTION_RAD):
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A path must contain at least two finite seven-joint waypoints")
    if resolution <= 0:
        raise ValueError("Resolution must be positive")
    dense = [values[0].copy()]
    for start, goal in zip(values[:-1], values[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / resolution)))
        dense.extend(start + (goal - start) * i / count for i in range(1, count + 1))
    return np.asarray(dense)


class TrajectoryExecutor:
    """All checks precede motor command publication; locks never take path values."""

    def __init__(self, path, locked_positions, joint_limits, edge_safe):
        self.path = densify_path(path)
        self.locked = dict(locked_positions)  # vector positions, not arbitrary body IDs
        self.active = [i for i in range(7) if i not in self.locked]
        limits = np.asarray(joint_limits)
        if np.any(self.path < limits[:, 0]) or np.any(self.path > limits[:, 1]):
            raise ValueError("Path violates joint limits")
        for index, target in self.locked.items():
            if not np.allclose(self.path[:, index], target, atol=1e-9, rtol=0):
                raise ValueError("Path modifies a locked joint")
        # Independently validate the EXACT dense execution path before any motion.
        if not all(edge_safe(a, b) for a, b in zip(self.path[:-1], self.path[1:])):
            raise ValueError("Final dense execution path validation FAILED")
        self.edge_safe = edge_safe
        self.index = 0
        self.state = "READY_TO_EXECUTE"
        self.reason = ""
        self.command = self.path[0].copy()
        self.last_valid_actual = self.path[0].copy()
        self.hold = None
        self.elapsed = 0.0
        self.waypoint_elapsed = 0.0
        self.dwell = 0.0

    def stop(self, actual, reason):
        if self.state != "SAFE_STOP":
            current = np.asarray(actual, dtype=float)
            self.hold = (current if current.shape == (7,) and np.isfinite(current).all()
                         else self.last_valid_actual).copy()
            self.reason = reason
            self.state = "SAFE_STOP"
        self.command = self.hold.copy()
        for i, value in self.locked.items():
            self.command[i] = value
        return self.command.copy()

    def update(self, actual, dt, current_collision, edge_start=None):
        actual = np.asarray(actual, dtype=float)
        if self.state == "SAFE_STOP":
            return self.stop(actual, self.reason)
        if actual.shape != (7,) or not np.isfinite(actual).all() or not np.isfinite(dt) or dt <= 0:
            return self.stop(actual, "INVALID_ROBOT_STATE")
        self.last_valid_actual = actual.copy()
        if max(abs(actual[i] - value) for i, value in self.locked.items()) >= 0.005:
            return self.stop(actual, "LOCKED_JOINT_DEVIATION")
        if current_collision:
            return self.stop(actual, "ACTUAL_CONFIGURATION_COLLISION")
        if self.state == "GOAL_REACHED":
            return self.command.copy()
        if self.state == "READY_TO_EXECUTE":
            if np.max(np.abs(actual[self.active] - self.path[0, self.active])) > 0.002:
                return self.stop(actual, "ROBOT_NOT_AT_VALIDATED_PATH_START")
            self.state = "TRAJECTORY_EXECUTION"
        self.elapsed += dt
        self.waypoint_elapsed += dt
        # Slow command progression without modifying motor maxVelocity/gains.
        if self.index > 0:
            self.dwell = max(abs(self.path[self.index] - self.path[self.index - 1])) / COMMAND_SPEED_RAD_S
        target = self.path[self.index]
        if self.waypoint_elapsed > WAYPOINT_TIMEOUT_S:
            return self.stop(actual, "WAYPOINT_TIMEOUT")
        arrived = np.max(np.abs(actual[self.active] - target[self.active])) <= WAYPOINT_TOLERANCE_RAD
        if arrived and self.waypoint_elapsed >= self.dwell:
            if self.index == len(self.path) - 1:
                self.state = "GOAL_REACHED"
                self.command = actual.copy()  # latch reached pose, never initial pose
                for i, value in self.locked.items():
                    self.command[i] = value
                return self.command.copy()
            self.index += 1
            self.waypoint_elapsed = 0.0
            target = self.path[self.index]
        # Check short upcoming segment from CURRENT q, not just planned q.
        if not self.edge_safe(actual if edge_start is None else edge_start, target):
            return self.stop(actual, "UPCOMING_SEGMENT_COLLISION")
        self.command = target.copy()
        for i, value in self.locked.items():
            self.command[i] = value
        return self.command.copy()
