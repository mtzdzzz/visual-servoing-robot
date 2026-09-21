"""Vision-occupancy collision checking for Stage 18.

``CollisionChecker`` owns a separate, headless Panda collision model.  Formal
checks accept only a supplied estimated world-frame AABB; this module never
accepts a PyBullet obstacle body ID, pose, or world-coordinate target.  Its
robot model represents every main Panda collision link by the link's current
world AABB.  This is deliberately a broad-phase, conservative approximation
suited to a later planner, not a full mesh narrow-phase or self-collision
system.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Sequence

import numpy as np
import pybullet as p
import pybullet_data


@dataclass(frozen=True)
class AxisAlignedBox:
    """Finite world-frame AABB used by the formal visual collision checker."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        low, high = np.asarray(self.minimum, dtype=np.float64), np.asarray(self.maximum, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("AABB bounds must be three finite values.")
        if np.any(low >= high):
            raise ValueError("AABB requires minimum < maximum for every axis.")

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in ((np.asarray(self.minimum) + np.asarray(self.maximum)) / 2.0))


@dataclass(frozen=True)
class ConfigurationCollisionResult:
    """A fresh, visual-occupancy collision decision for one Panda configuration."""

    collision: bool
    colliding_links: tuple[str, ...]
    minimum_clearance_m: float


@dataclass(frozen=True)
class PathCollisionResult:
    """Interpolated joint-space path result; no motion command is issued."""

    collision: bool
    num_samples: int
    first_collision_sample: int | None
    first_collision_fraction: float | None
    colliding_links: tuple[str, ...]
    minimum_clearance_m: float


class CollisionChecker:
    """Check Panda link-AABBs against an estimated obstacle occupancy AABB.

    The checker keeps its own ``DIRECT`` client, so temporary joint-state
    updates cannot change the GUI robot, active hold targets, motor commands,
    visual servo state, or Stage 16 target selection.  The model includes the
    Panda base, seven arm links, link8/hand, and finger collision links.  It
    implements robot-to-obstacle checking only; self-collision is explicitly
    outside Stage 18.
    """

    def __init__(self) -> None:
        self.client_id = p.connect(p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Unable to create the Stage 18 DIRECT collision client.")
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            self.robot_id = p.loadURDF(
                "franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.client_id
            )
            if self.robot_id < 0:
                raise RuntimeError("Unable to load Panda collision model.")
            self.arm_joint_indices = self._find_arm_joint_indices()
            self.arm_joint_limits = self._read_arm_joint_limits()
            self.link_indices, self.link_names = self._find_collision_link_indices()
            self.end_effector_link_index = self._find_link_index("panda_link8")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "client_id", -1) >= 0 and p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = -1

    def __enter__(self) -> "CollisionChecker":
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def check_configuration(
        self,
        joint_configuration: Sequence[float],
        estimated_occupancy: AxisAlignedBox,
    ) -> ConfigurationCollisionResult:
        """Formally check a configuration using only visual estimated occupancy."""

        return self._check_against_aabb(joint_configuration, estimated_occupancy)

    def evaluate_configuration_against_ground_truth(
        self,
        joint_configuration: Sequence[float],
        ground_truth_aabb: AxisAlignedBox,
    ) -> ConfigurationCollisionResult:
        """Evaluation-only oracle. Never call this to choose a formal result."""

        return self._check_against_aabb(joint_configuration, ground_truth_aabb)

    def check_path(
        self,
        q_start: Sequence[float],
        q_goal: Sequence[float],
        estimated_occupancy: AxisAlignedBox,
        max_joint_step_rad: float,
    ) -> PathCollisionResult:
        """Check all interpolated configurations of a path against visual occupancy."""

        return self._check_path_against_aabb(q_start, q_goal, estimated_occupancy, max_joint_step_rad)

    def evaluate_path_against_ground_truth(
        self,
        q_start: Sequence[float],
        q_goal: Sequence[float],
        ground_truth_aabb: AxisAlignedBox,
        max_joint_step_rad: float,
    ) -> PathCollisionResult:
        """Evaluation-only version of :meth:`check_path` using the true box AABB."""

        return self._check_path_against_aabb(q_start, q_goal, ground_truth_aabb, max_joint_step_rad)

    def end_effector_position(self, joint_configuration: Sequence[float]) -> tuple[float, float, float]:
        """Return a display-only link8 point for GUI path lines."""

        self._set_arm_configuration(joint_configuration)
        state = p.getLinkState(
            self.robot_id, self.end_effector_link_index, computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return tuple(float(value) for value in state[4])

    def _check_path_against_aabb(
        self,
        q_start: Sequence[float],
        q_goal: Sequence[float],
        obstacle: AxisAlignedBox,
        max_joint_step_rad: float,
    ) -> PathCollisionResult:
        start, goal = self._validated_configuration(q_start), self._validated_configuration(q_goal)
        if max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive.")
        max_delta = float(np.max(np.abs(goal - start)))
        num_samples = max(2, int(ceil(max_delta / max_joint_step_rad)) + 1)
        minimum_clearance = float("inf")
        for sample_index in range(num_samples):
            fraction = sample_index / (num_samples - 1)
            configuration = start + fraction * (goal - start)
            result = self._check_against_aabb(configuration, obstacle)
            minimum_clearance = min(minimum_clearance, result.minimum_clearance_m)
            if result.collision:
                return PathCollisionResult(
                    True, num_samples, sample_index, fraction, result.colliding_links, minimum_clearance
                )
        return PathCollisionResult(False, num_samples, None, None, (), minimum_clearance)

    def _check_against_aabb(
        self,
        joint_configuration: Sequence[float],
        obstacle: AxisAlignedBox,
    ) -> ConfigurationCollisionResult:
        self._set_arm_configuration(joint_configuration)
        colliding: list[str] = []
        minimum_clearance = float("inf")
        for link_index, link_name in zip(self.link_indices, self.link_names):
            bounds = p.getAABB(self.robot_id, linkIndex=link_index, physicsClientId=self.client_id)
            link_box = AxisAlignedBox(tuple(bounds[0]), tuple(bounds[1]))
            clearance = _aabb_clearance(link_box, obstacle)
            minimum_clearance = min(minimum_clearance, clearance)
            if _aabb_overlap(link_box, obstacle):
                colliding.append(link_name)
        return ConfigurationCollisionResult(bool(colliding), tuple(colliding), minimum_clearance)

    def _validated_configuration(self, joint_configuration: Sequence[float]) -> np.ndarray:
        values = np.asarray(joint_configuration, dtype=np.float64)
        if values.shape != (len(self.arm_joint_indices),) or not np.all(np.isfinite(values)):
            raise ValueError("Expected seven finite Panda arm joint positions.")
        for value, (lower, upper) in zip(values, self.arm_joint_limits):
            if value < lower - 1e-9 or value > upper + 1e-9:
                raise ValueError("Configuration lies outside Panda joint limits.")
        return values

    def _set_arm_configuration(self, joint_configuration: Sequence[float]) -> None:
        values = self._validated_configuration(joint_configuration)
        for joint_index, joint_value in zip(self.arm_joint_indices, values):
            p.resetJointState(self.robot_id, joint_index, float(joint_value), physicsClientId=self.client_id)

    def _find_arm_joint_indices(self) -> list[int]:
        indices: list[int] = []
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
            name = info[1].decode("utf-8")
            if info[2] == p.JOINT_REVOLUTE and name.startswith("panda_joint"):
                indices.append(index)
        if len(indices) != 7:
            raise RuntimeError(f"Expected 7 Panda revolute joints, found {indices}.")
        return indices

    def _read_arm_joint_limits(self) -> list[tuple[float, float]]:
        return [
            (float(info[8]), float(info[9]))
            for info in (
                p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
                for index in self.arm_joint_indices
            )
        ]

    def _find_collision_link_indices(self) -> tuple[list[int], list[str]]:
        # The fixed base is index -1.  All Panda arm, link8/hand and finger
        # links are included; non-Panda auxiliary links are excluded.
        indices, names = [-1], ["panda_link0"]
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
            child_name = info[12].decode("utf-8")
            if child_name.startswith("panda_link") or child_name in {
                "panda_hand", "panda_leftfinger", "panda_rightfinger"
            }:
                indices.append(index)
                names.append(child_name)
        return indices, names

    def _find_link_index(self, link_name: str) -> int:
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            if p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)[12].decode("utf-8") == link_name:
                return index
        raise RuntimeError(f"Panda collision model is missing {link_name}.")


def _aabb_overlap(first: AxisAlignedBox, second: AxisAlignedBox) -> bool:
    first_min, first_max = np.asarray(first.minimum), np.asarray(first.maximum)
    second_min, second_max = np.asarray(second.minimum), np.asarray(second.maximum)
    return bool(np.all(first_min <= second_max) and np.all(second_min <= first_max))


def _aabb_clearance(first: AxisAlignedBox, second: AxisAlignedBox) -> float:
    """Euclidean distance between two AABBs; zero means overlapping/touching."""

    first_min, first_max = np.asarray(first.minimum), np.asarray(first.maximum)
    second_min, second_max = np.asarray(second.minimum), np.asarray(second.maximum)
    separation = np.maximum(0.0, np.maximum(second_min - first_max, first_min - second_max))
    return float(sqrt(float(np.dot(separation, separation))))
