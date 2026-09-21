"""Joint-space RRT-Connect planner for the Stage 19 visual-AABB experiment.

The planner intentionally knows nothing about PyBullet obstacle bodies or
ground-truth poses.  It receives a Stage 18 :class:`CollisionChecker`, a
fresh Stage 17 *estimated* occupancy AABB, and a full seven-joint start/goal
configuration.  Only the caller-designated active joints are sampled; every
configuration assembled for collision checking restores the frozen joints to
their Stage 3.1 values.

Every attempted tree edge is checked with the existing interpolated
``CollisionChecker.check_path`` interface.  Therefore RRT nodes are not
accepted merely because their endpoints happen to be safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Sequence

import numpy as np

from collision_checker import AxisAlignedBox, CollisionChecker


class ExtendStatus(str, Enum):
    """Result of one tree extension in standard RRT-Connect terminology."""

    TRAPPED = "TRAPPED"
    ADVANCED = "ADVANCED"
    REACHED = "REACHED"


@dataclass(frozen=True)
class RRTConnectConfig:
    """Fixed, reproducible Stage 19 sampling parameters."""

    step_size_rad: float = 0.15
    edge_resolution_rad: float = 0.05
    goal_bias_probability: float = 0.10
    max_iterations: int = 2500
    max_planning_time_s: float = 5.0
    random_seed: int = 1

    def __post_init__(self) -> None:
        if self.step_size_rad <= 0.0:
            raise ValueError("step_size_rad must be positive.")
        if self.edge_resolution_rad <= 0.0:
            raise ValueError("edge_resolution_rad must be positive.")
        if not 0.0 <= self.goal_bias_probability <= 1.0:
            raise ValueError("goal_bias_probability must be within [0, 1].")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.max_planning_time_s <= 0.0:
            raise ValueError("max_planning_time_s must be positive.")


@dataclass(frozen=True)
class PlannerNode:
    """One active-joint node with a parent index in its own RRT tree."""

    active_configuration: tuple[float, ...]
    parent_index: int | None


@dataclass
class _Tree:
    """Mutable RRT tree whose root is either start or goal."""

    root_is_start: bool
    nodes: list[PlannerNode]


@dataclass(frozen=True)
class RRTConnectResult:
    """Planning result.  ``path`` always contains full seven-joint vectors."""

    success: bool
    path: tuple[tuple[float, ...], ...]
    planning_time_s: float
    iterations: int
    tree_a_nodes: int
    tree_b_nodes: int
    collision_checks: int
    failure_reason: str


class RRTConnectPlanner:
    """Bidirectional RRT-Connect over Panda's designated active joints only.

    ``q_start`` and ``q_goal`` are full seven-joint Panda configurations.  The
    planner preserves their common locked coordinates and samples only
    ``active_joint_positions``.  The input collision checker is the frozen
    Stage 18 checker; the planner never accesses PyBullet itself.
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        estimated_occupancy: AxisAlignedBox,
        active_joint_positions: Sequence[int],
        locked_joint_positions: dict[int, float],
        joint_limits: Sequence[tuple[float, float]],
        config: RRTConnectConfig,
    ) -> None:
        self.collision_checker = collision_checker
        self.estimated_occupancy = estimated_occupancy
        self.active_joint_positions = tuple(int(index) for index in active_joint_positions)
        self.locked_joint_positions = {
            int(index): float(value) for index, value in locked_joint_positions.items()
        }
        self.joint_limits = tuple((float(low), float(high)) for low, high in joint_limits)
        self.config = config
        if not self.active_joint_positions:
            raise ValueError("RRT-Connect requires at least one active joint.")
        if len(set(self.active_joint_positions)) != len(self.active_joint_positions):
            raise ValueError("active_joint_positions must be unique.")
        if len(self.joint_limits) != 7:
            raise ValueError("Stage 19 expects the seven Panda arm joint limits.")
        if any(index < 0 or index >= len(self.joint_limits) for index in self.active_joint_positions):
            raise ValueError("An active joint position is outside the seven-joint arm vector.")
        overlap = set(self.active_joint_positions).intersection(self.locked_joint_positions)
        if overlap:
            raise ValueError(f"A joint cannot be both active and locked: {sorted(overlap)}")
        self._active_lower = np.asarray(
            [self.joint_limits[index][0] for index in self.active_joint_positions], dtype=np.float64
        )
        self._active_upper = np.asarray(
            [self.joint_limits[index][1] for index in self.active_joint_positions], dtype=np.float64
        )
        self._collision_checks = 0

    def plan(
        self,
        q_start: Sequence[float],
        q_goal: Sequence[float],
    ) -> RRTConnectResult:
        """Plan a collision-free path from ``q_start`` to ``q_goal``.

        The caller must run the required direct-path check first.  This method
        intentionally does not shortcut that policy; it only performs
        bidirectional sampling after the caller established that RRT is needed.
        """

        start = self._validate_full_configuration(q_start, "q_start")
        goal = self._validate_full_configuration(q_goal, "q_goal")
        start_active, goal_active = self._active_from_full(start), self._active_from_full(goal)
        if np.allclose(start_active, goal_active, atol=1e-12):
            return RRTConnectResult(
                True, (tuple(float(value) for value in start),), 0.0, 0, 1, 1, 0, ""
            )

        # The planner never allows a colliding endpoint into either tree.
        if self._configuration_collision(start_active):
            return self._failure("START_CONFIGURATION_COLLISION", 0.0, 0, 1, 1)
        if self._configuration_collision(goal_active):
            return self._failure("GOAL_CONFIGURATION_COLLISION", 0.0, 0, 1, 1)

        self._collision_checks = 0
        rng = np.random.default_rng(self.config.random_seed)
        start_tree = _Tree(True, [PlannerNode(tuple(float(value) for value in start_active), None)])
        goal_tree = _Tree(False, [PlannerNode(tuple(float(value) for value in goal_active), None)])
        tree_a, tree_b = start_tree, goal_tree
        started = perf_counter()

        for iteration in range(1, self.config.max_iterations + 1):
            elapsed = perf_counter() - started
            if elapsed >= self.config.max_planning_time_s:
                return self._failure(
                    "MAX_PLANNING_TIME", elapsed, iteration - 1, len(start_tree.nodes), len(goal_tree.nodes)
                )

            # A fixed 10% goal bias remains independent of obstacle geometry.
            q_rand = goal_active if rng.random() < self.config.goal_bias_probability else self._sample_active(rng)
            status_a, node_a = self._extend(tree_a, q_rand)
            if status_a is not ExtendStatus.TRAPPED and node_a is not None:
                q_new = np.asarray(tree_a.nodes[node_a].active_configuration, dtype=np.float64)
                status_b, node_b = self._connect(tree_b, q_new)
                if status_b is ExtendStatus.REACHED and node_b is not None:
                    path = self._reconstruct_full_path(tree_a, node_a, tree_b, node_b)
                    return RRTConnectResult(
                        True,
                        tuple(path),
                        perf_counter() - started,
                        iteration,
                        len(start_tree.nodes),
                        len(goal_tree.nodes),
                        self._collision_checks,
                        "",
                    )
            # Standard RRT-Connect alternates the tree expanded first.
            tree_a, tree_b = tree_b, tree_a

        return self._failure(
            "MAX_ITERATIONS", perf_counter() - started, self.config.max_iterations,
            len(start_tree.nodes), len(goal_tree.nodes),
        )

    def _failure(
        self,
        reason: str,
        planning_time_s: float,
        iterations: int,
        start_nodes: int,
        goal_nodes: int,
    ) -> RRTConnectResult:
        return RRTConnectResult(
            False, (), planning_time_s, iterations, start_nodes, goal_nodes,
            self._collision_checks, reason,
        )

    def _validate_full_configuration(self, values: Sequence[float], label: str) -> np.ndarray:
        configuration = np.asarray(values, dtype=np.float64)
        if configuration.shape != (len(self.joint_limits),) or not np.all(np.isfinite(configuration)):
            raise ValueError(f"{label} must contain seven finite joint values.")
        for index, (value, limits) in enumerate(zip(configuration, self.joint_limits)):
            if value < limits[0] - 1e-9 or value > limits[1] + 1e-9:
                raise ValueError(f"{label}[{index}] violates its Panda joint limits.")
        for index, locked_value in self.locked_joint_positions.items():
            if not np.isclose(configuration[index], locked_value, atol=1e-9):
                raise ValueError(f"{label}[{index}] differs from its frozen locked-joint value.")
        return configuration

    def _active_from_full(self, configuration: np.ndarray) -> np.ndarray:
        return np.asarray([configuration[index] for index in self.active_joint_positions], dtype=np.float64)

    def _full_from_active(self, active_configuration: Sequence[float]) -> tuple[float, ...]:
        active = np.asarray(active_configuration, dtype=np.float64)
        if active.shape != self._active_lower.shape or not np.all(np.isfinite(active)):
            raise ValueError("Active configuration has an unexpected shape or non-finite value.")
        if np.any(active < self._active_lower - 1e-9) or np.any(active > self._active_upper + 1e-9):
            raise ValueError("Active configuration violates a Panda joint limit.")
        full = np.zeros(len(self.joint_limits), dtype=np.float64)
        for index, value in self.locked_joint_positions.items():
            full[index] = value
        for index, value in zip(self.active_joint_positions, active):
            full[index] = value
        return tuple(float(value) for value in full)

    def _sample_active(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self._active_lower, self._active_upper)

    @staticmethod
    def _nearest_node_index(tree: _Tree, target: np.ndarray) -> int:
        configurations = np.asarray([node.active_configuration for node in tree.nodes], dtype=np.float64)
        return int(np.argmin(np.linalg.norm(configurations - target, axis=1)))

    def _steer(self, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        delta = target - source
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.step_size_rad:
            return target.copy(), True
        return source + delta * (self.config.step_size_rad / distance), False

    def _edge_is_safe(self, source: np.ndarray, destination: np.ndarray) -> bool:
        self._collision_checks += 1
        result = self.collision_checker.check_path(
            self._full_from_active(source), self._full_from_active(destination),
            self.estimated_occupancy, self.config.edge_resolution_rad,
        )
        return not result.collision

    def _configuration_collision(self, active_configuration: np.ndarray) -> bool:
        self._collision_checks += 1
        return self.collision_checker.check_configuration(
            self._full_from_active(active_configuration), self.estimated_occupancy
        ).collision

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[ExtendStatus, int | None]:
        nearest_index = self._nearest_node_index(tree, target)
        nearest = np.asarray(tree.nodes[nearest_index].active_configuration, dtype=np.float64)
        candidate, reached = self._steer(nearest, target)
        if np.allclose(candidate, nearest, atol=1e-12) or not self._edge_is_safe(nearest, candidate):
            return ExtendStatus.TRAPPED, None
        tree.nodes.append(PlannerNode(tuple(float(value) for value in candidate), nearest_index))
        return (ExtendStatus.REACHED if reached else ExtendStatus.ADVANCED), len(tree.nodes) - 1

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[ExtendStatus, int | None]:
        """Repeatedly extend one tree until collision or exact connection."""

        while True:
            status, node_index = self._extend(tree, target)
            if status is not ExtendStatus.ADVANCED:
                return status, node_index

    @staticmethod
    def _root_to_node(tree: _Tree, node_index: int) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        current: int | None = node_index
        while current is not None:
            values.append(np.asarray(tree.nodes[current].active_configuration, dtype=np.float64))
            current = tree.nodes[current].parent_index
        return list(reversed(values))

    def _reconstruct_full_path(
        self,
        first_tree: _Tree,
        first_node: int,
        second_tree: _Tree,
        second_node: int,
    ) -> list[tuple[float, ...]]:
        """Return the correctly oriented q_start-to-q_goal path after a join."""

        first_branch = self._root_to_node(first_tree, first_node)
        second_branch = self._root_to_node(second_tree, second_node)
        if first_tree.root_is_start:
            start_to_join, goal_to_join = first_branch, second_branch
        else:
            start_to_join, goal_to_join = second_branch, first_branch
        active_path = start_to_join + list(reversed(goal_to_join[:-1]))
        return [self._full_from_active(active) for active in active_path]
