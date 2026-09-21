"""Stage 19: RRT-Connect planning using only Stage 17 visual occupancy.

This is a planning, validation, and visualization experiment.  The GUI Panda
is deliberately held at a safe start configuration throughout; no direct or
RRT path is sent to its motors.  A separate Stage 18 DIRECT collision model
checks every RRT edge against a newly captured Stage 17 estimated AABB.
Ground-truth obstacle geometry is retrieved only after all formal planning
results have been fixed, and is used only to audit final paths.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import numpy as np
import pybullet as p

import simulation as sim
import stage10_evaluation as stage10
import stage14_rgbd_evaluation as stage14
import stage17_obstacle_evaluation as stage17
import stage18_collision_evaluation as stage18
from camera_observation import EyeInHandRgbDisplay
from collision_checker import AxisAlignedBox, CollisionChecker, PathCollisionResult
from rrt_connect_planner import RRTConnectConfig, RRTConnectPlanner, RRTConnectResult
from stage15_multitarget_evaluation import TARGET_SPECS, _create_coloured_sphere, _set_target_pose


ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = ROOT / "outputs" / "logs"
PATH_DIRECTORY = ROOT / "outputs" / "paths"
LOG_PATH = LOG_DIRECTORY / "stage19_rrt_connect.csv"
SUMMARY_PATH = LOG_DIRECTORY / "stage19_summary.csv"

# The Stage 18 validated interpolation density is deliberately reused without
# relaxation.  The remaining values are fixed Stage 19 planner configuration,
# not visual-servo or robot motor parameters.
RRT_STEP_SIZE_RAD = 0.15
PATH_MAX_JOINT_STEP_RAD = stage18.PATH_MAX_JOINT_STEP_RAD
GOAL_BIAS_PROBABILITY = 0.10
MAX_ITERATIONS = 2500
MAX_PLANNING_TIME_S = 5.0
RRT_RANDOM_SEEDS = (1, 2, 3, 4, 5)

# Stage 17's yellow geometry and safety margin are reused unchanged.  The
# original low ground placement is outside the joint5--7 swept workspace once
# joints1--4 are frozen.  This Stage 19 *scene placement* lifts the same
# perceived cuboid into the camera-visible active-wrist workspace.  Its world
# pose is used solely to place the simulated object; planner input remains the
# fresh RGB-D estimated AABB returned by Stage 18's capture helper.
STAGE19_OBSTACLE_SCENE_POSITION = (0.10, -0.20, 0.80)

# The GUI robot is made current at this safe active-joint configuration before
# perception begins.  It is a static Stage 19 experiment setup, never a motor
# command or a Stage 3.1 baseline change.  joints1--4 retain their recorded
# locked values.
STAGE19_ACTIVE_START = (-0.4316, 0.3433, -1.1167)


@dataclass(frozen=True)
class PlanningScenario:
    """A predefined Cartesian goal.  IK creates its active joint goal online."""

    scenario_id: str
    scenario_type: str
    end_effector_goal_m: tuple[float, float, float]


# These Cartesian goal poses are fixed planning test targets, not obstacle
# poses or manually specified detours.  Every one is solved at runtime through
# the frozen constrained position-only IK.  RRT provides all intermediate
# waypoints after a formal direct-path collision decision.
PLANNING_SCENARIOS: tuple[PlanningScenario, ...] = (
    PlanningScenario("S01", "DIRECT_SAFE", (0.1324, -0.0133, 0.9944)),
    PlanningScenario("S02", "DIRECT_SAFE", (0.1212, 0.0662, 1.0216)),
    PlanningScenario("S03", "DIRECT_SAFE", (0.1109, 0.0469, 1.1015)),
    PlanningScenario("S04", "DIRECT_COLLISION", (0.1052, -0.0886, 1.0492)),
    PlanningScenario("S05", "DIRECT_COLLISION", (0.0711, -0.1108, 1.0762)),
    PlanningScenario("S06", "DIRECT_COLLISION", (0.0339, -0.1220, 1.0892)),
    PlanningScenario("S07", "DIRECT_COLLISION", (0.0085, -0.1201, 1.1015)),
    PlanningScenario("S08", "DIRECT_COLLISION", (-0.0146, -0.1120, 1.1132)),
    PlanningScenario("S09", "NEAR_BOUNDARY", (0.1112, -0.0644, 0.9813)),
    PlanningScenario("S10", "NEAR_BOUNDARY", (0.1196, -0.0653, 1.0079)),
)


@dataclass
class FormalPlan:
    """Vision-only planning result stored before ground-truth evaluation."""

    scenario: PlanningScenario
    seed: int | None
    q_start: tuple[float, ...]
    q_goal: tuple[float, ...] | None
    goal_ik_error_m: float | None
    direct_result: PathCollisionResult | None
    rrt_required: bool
    planning_result: RRTConnectResult | None
    final_path: tuple[tuple[float, ...], ...]
    final_path_estimated_safe: bool
    final_validation: PathCollisionResult | None
    path_file: Path | None


def _json_vector(values: Sequence[float] | None) -> str:
    if values is None:
        return ""
    return json.dumps([round(float(value), 8) for value in values], separators=(",", ":"))


def _path_length(path: Sequence[Sequence[float]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(np.asarray(last) - np.asarray(first)) for first, last in zip(path, path[1:])))


def _validate_complete_path(
    checker: CollisionChecker,
    path: Sequence[Sequence[float]],
    occupancy: AxisAlignedBox,
) -> PathCollisionResult:
    """Independently revalidate every full path edge against visual occupancy."""

    if not path:
        raise ValueError("An empty path cannot be validated.")
    if len(path) == 1:
        configuration = checker.check_configuration(path[0], occupancy)
        return PathCollisionResult(
            configuration.collision, 1, 0 if configuration.collision else None,
            0.0 if configuration.collision else None, configuration.colliding_links,
            configuration.minimum_clearance_m,
        )
    total_samples = 0
    min_clearance = float("inf")
    for index, (first, second) in enumerate(zip(path, path[1:])):
        result = checker.check_path(first, second, occupancy, PATH_MAX_JOINT_STEP_RAD)
        total_samples += result.num_samples
        min_clearance = min(min_clearance, result.minimum_clearance_m)
        if result.collision:
            return PathCollisionResult(
                True, total_samples, index, index / max(1, len(path) - 1),
                result.colliding_links, min_clearance,
            )
    return PathCollisionResult(False, total_samples, None, None, (), min_clearance)


def _evaluate_complete_path_against_gt(
    checker: CollisionChecker,
    path: Sequence[Sequence[float]],
    ground_truth_aabb: AxisAlignedBox,
) -> bool:
    """Evaluation-only full-path safety check with no effect on planning."""

    if not path:
        return False
    if len(path) == 1:
        return not checker.evaluate_configuration_against_ground_truth(path[0], ground_truth_aabb).collision
    return all(
        not checker.evaluate_path_against_ground_truth(
            first, second, ground_truth_aabb, PATH_MAX_JOINT_STEP_RAD
        ).collision
        for first, second in zip(path, path[1:])
    )


def _write_path(path: Sequence[Sequence[float]], scenario_id: str, seed_label: str) -> Path:
    PATH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path_file = PATH_DIRECTORY / f"stage19_{scenario_id.lower()}_seed_{seed_label}_path.csv"
    with path_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("waypoint_index", "q1", "q2", "q3", "q4", "q5", "q6", "q7"))
        writer.writeheader()
        for index, configuration in enumerate(path):
            writer.writerow({"waypoint_index": index, **{f"q{joint + 1}": f"{value:.10f}" for joint, value in enumerate(configuration)}})
    return path_file


def _draw_waypoint_path(
    path: Sequence[Sequence[float]],
    checker: CollisionChecker,
    client_id: int,
    colour: tuple[float, float, float],
    line_width: float,
    label: str | None = None,
) -> None:
    """Draw a display-only link8 polyline.  It is not a collision proxy."""

    if not path:
        return
    points = [checker.end_effector_position(configuration) for configuration in path]
    for first, second in zip(points, points[1:]):
        p.addUserDebugLine(first, second, lineColorRGB=colour, lineWidth=line_width, lifeTime=0, physicsClientId=client_id)
    if label:
        endpoint = points[-1]
        p.addUserDebugText(
            label, (endpoint[0], endpoint[1], endpoint[2] + 0.025), textColorRGB=colour,
            textSize=0.75, lifeTime=0, physicsClientId=client_id,
        )


def _set_stage19_current_start(context: stage10.ReadyContext, client_id: int) -> None:
    """Set the static planning-start configuration, then HOLD it unchanged."""

    if len(context.active_joint_indices) != len(STAGE19_ACTIVE_START):
        raise RuntimeError("Stage 19 expected exactly the three Stage 3.1 active joints.")
    for joint_index, position in zip(context.active_joint_indices, STAGE19_ACTIVE_START):
        p.resetJointState(context.robot_id, joint_index, position, physicsClientId=client_id)
    stage14._freeze_at_current_pose(context, client_id)
    stage14._run_hold_steps(context, round(0.50 / sim.TIME_STEP), client_id)
    stage14._freeze_at_current_pose(context, client_id)


def _full_current_configuration(context: stage10.ReadyContext, client_id: int) -> tuple[float, ...]:
    return tuple(
        float(p.getJointState(context.robot_id, joint_index, physicsClientId=client_id)[0])
        for joint_index in context.arm_joint_indices
    )


def _locked_vector_positions(
    context: stage10.ReadyContext,
    checker: CollisionChecker,
) -> tuple[dict[int, float], tuple[int, ...]]:
    """Map PyBullet indices to the seven-joint CollisionChecker vector order."""

    vector_index = {joint_index: index for index, joint_index in enumerate(checker.arm_joint_indices)}
    locked = {
        vector_index[joint_index]: context.locked_initial_positions[joint_index]
        for joint_index in context.locked_joint_indices
    }
    active = tuple(vector_index[joint_index] for joint_index in context.active_joint_indices)
    if active != (4, 5, 6) or set(locked) != {0, 1, 2, 3}:
        raise RuntimeError(
            "Stage 19 requires the frozen Stage 3.1 layout: locked joint1--4 and active joint5--7."
        )
    return locked, active


def _solve_goal_with_existing_ik(
    scenario: PlanningScenario,
    context: stage10.ReadyContext,
    client_id: int,
) -> tuple[tuple[float, ...], float]:
    """Use the existing constrained, position-only IK without motor commands."""

    arm_indices, targets, ik_error = sim.calculate_constrained_position_ik(
        context.robot_id,
        scenario.end_effector_goal_m,
        context.locked_joint_indices,
        context.locked_initial_positions,
        client_id,
        position_tolerance=0.002,
    )
    if list(arm_indices) != list(context.arm_joint_indices):
        raise RuntimeError("Existing constrained IK returned an unexpected Panda arm order.")
    return tuple(float(value) for value in targets), float(ik_error)


def _csv_fields() -> tuple[str, ...]:
    return (
        "scenario_id", "scenario_type", "seed", "goal_end_effector_x_m", "goal_end_effector_y_m",
        "goal_end_effector_z_m", "goal_ik_error_m", "q_start", "q_goal", "direct_path_safe",
        "direct_first_collision_fraction", "rrt_required", "planning_success", "planning_time_s",
        "iterations", "tree_a_nodes", "tree_b_nodes", "collision_checks", "num_waypoints",
        "direct_joint_distance", "rrt_path_length", "final_path_estimated_safe", "final_path_gt_safe",
        "failure_reason", "path_file", "estimated_obstacle_center", "estimated_aabb_min", "estimated_aabb_max",
        "gt_used_for_planning", "gt_used_for_final_path_evaluation", "self_collision_checking",
    )


def _summary_fields() -> tuple[str, ...]:
    return (
        "scenarios", "direct_safe_count", "rrt_required_count", "rrt_trials", "rrt_success_count",
        "rrt_failure_count", "rrt_planning_success_rate", "mean_planning_time_s", "median_planning_time_s",
        "max_planning_time_s", "mean_iterations", "mean_tree_nodes", "mean_raw_path_length",
        "final_path_gt_collision_count", "self_collision_checking", "path_max_joint_step_rad",
        "rrt_step_size_rad", "goal_bias_probability", "random_seed_count", "locked_joint_max_deviation_rad", "result",
    )


def _row_from_formal(
    record: FormalPlan,
    visual_occupancy: AxisAlignedBox,
    final_path_gt_safe: bool | None,
) -> dict[str, object]:
    direct = record.direct_result
    planned = record.planning_result
    path = record.final_path
    q_goal = record.q_goal
    return {
        "scenario_id": record.scenario.scenario_id,
        "scenario_type": record.scenario.scenario_type,
        "seed": "" if record.seed is None else record.seed,
        "goal_end_effector_x_m": f"{record.scenario.end_effector_goal_m[0]:.8f}",
        "goal_end_effector_y_m": f"{record.scenario.end_effector_goal_m[1]:.8f}",
        "goal_end_effector_z_m": f"{record.scenario.end_effector_goal_m[2]:.8f}",
        "goal_ik_error_m": "" if record.goal_ik_error_m is None else f"{record.goal_ik_error_m:.9f}",
        "q_start": _json_vector(record.q_start),
        "q_goal": _json_vector(q_goal),
        "direct_path_safe": "" if direct is None else int(not direct.collision),
        "direct_first_collision_fraction": "" if direct is None or direct.first_collision_fraction is None else f"{direct.first_collision_fraction:.9f}",
        "rrt_required": int(record.rrt_required),
        "planning_success": "" if planned is None else int(planned.success),
        "planning_time_s": "" if planned is None else f"{planned.planning_time_s:.9f}",
        "iterations": "" if planned is None else planned.iterations,
        "tree_a_nodes": "" if planned is None else planned.tree_a_nodes,
        "tree_b_nodes": "" if planned is None else planned.tree_b_nodes,
        "collision_checks": "" if planned is None else planned.collision_checks,
        "num_waypoints": len(path),
        "direct_joint_distance": "" if q_goal is None else f"{float(np.linalg.norm(np.asarray(q_goal) - np.asarray(record.q_start))):.9f}",
        "rrt_path_length": "" if not path else f"{_path_length(path):.9f}",
        "final_path_estimated_safe": int(record.final_path_estimated_safe),
        "final_path_gt_safe": "" if final_path_gt_safe is None else int(final_path_gt_safe),
        "failure_reason": "" if planned is None else planned.failure_reason,
        "path_file": "" if record.path_file is None else str(record.path_file.relative_to(ROOT)),
        "estimated_obstacle_center": _json_vector(visual_occupancy.center),
        "estimated_aabb_min": _json_vector(visual_occupancy.minimum),
        "estimated_aabb_max": _json_vector(visual_occupancy.maximum),
        "gt_used_for_planning": False,
        "gt_used_for_final_path_evaluation": True,
        "self_collision_checking": "NOT_IMPLEMENTED",
    }


def run_stage19_rrt_connect_planning() -> None:
    """Plan and validate Stage 19 paths without executing any candidate path."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PATH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 19.")
    display: EyeInHandRgbDisplay | None = None
    checker: CollisionChecker | None = None
    try:
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB-D - Stage 19 RRT-Connect")
        context = stage14._create_static_localization_context(client_id)
        _set_stage19_current_start(context, client_id)
        target_ids = {"target_1": context.target_body_id}
        for spec in TARGET_SPECS[1:]:
            target_ids[spec.target_id] = _create_coloured_sphere(spec, client_id)
        for target_id, target_position in stage17.TARGET_WORLD_POSITIONS.items():
            _set_target_pose(target_ids[target_id], target_position, client_id)
        obstacle_body_id = stage17._create_yellow_obstacle(client_id)
        stage17._set_static_body_pose(obstacle_body_id, STAGE19_OBSTACLE_SCENE_POSITION, client_id)
        p.addUserDebugText(
            "REAL OBSTACLE (Stage 19 scene)",
            (
                STAGE19_OBSTACLE_SCENE_POSITION[0], STAGE19_OBSTACLE_SCENE_POSITION[1],
                STAGE19_OBSTACLE_SCENE_POSITION[2] + stage17.OBSTACLE_DIMENSIONS_M[2] / 2.0 + 0.025,
            ),
            textColorRGB=(1.0, 0.85, 0.0), textSize=0.85, physicsClientId=client_id,
        )
        context.debug_text_id = sim.update_motion_debug_text(
            "STAGE 19 - RRT-CONNECT PLANNING\nRobot: HOLD | Planning and visual validation only; no candidate path executes",
            context.debug_text_id, client_id,
        )
        stage14._run_hold_steps(context, round(stage17.SCENE_SETTLE_SECONDS / sim.TIME_STEP), client_id)

        print("Stage 19: RRT-Connect Collision-Free Path Planning")
        print("Planning joints: panda_joint5, panda_joint6, panda_joint7. Locked: panda_joint1--4.")
        print(
            f"RRT step={RRT_STEP_SIZE_RAD:.3f} rad | edge resolution={PATH_MAX_JOINT_STEP_RAD:.3f} rad | "
            f"goal bias={GOAL_BIAS_PROBABILITY:.0%} | max iterations={MAX_ITERATIONS} | max time={MAX_PLANNING_TIME_S:.1f}s."
        )
        print("Formal planner input: fresh Stage 17 visual occupancy AABB only. GT is evaluation-only after planning.")
        visual_occupancy, _debug_box, captured_steps = stage18._capture_visual_occupancy(context, display, client_id)
        print(f"Stage 17 visual occupancy capture: {stage18.VISUAL_OCCUPANCY_FRAMES} valid frames / {captured_steps} physics steps.")
        print(f"Estimated occupancy: min={visual_occupancy.minimum}, max={visual_occupancy.maximum}")

        checker = CollisionChecker()
        locked_vector, active_vector = _locked_vector_positions(context, checker)
        q_start = _full_current_configuration(context, client_id)
        if checker.check_configuration(q_start, visual_occupancy).collision:
            raise RuntimeError("Stage 19 current start configuration collides with the visual occupancy.")
        start_ee = checker.end_effector_position(q_start)
        p.addUserDebugText(
            "Q_START", (start_ee[0], start_ee[1], start_ee[2] + 0.03),
            textColorRGB=(0.1, 0.9, 1.0), textSize=0.85, physicsClientId=client_id,
        )

        # -------- Formal phase: no obstacle ground truth is read in this block. --------
        formal_records: list[FormalPlan] = []
        primary_rrt_success_drawn = False
        for scenario in PLANNING_SCENARIOS:
            q_goal, ik_error = _solve_goal_with_existing_ik(scenario, context, client_id)
            goal_configuration = checker.check_configuration(q_goal, visual_occupancy)
            direct = checker.check_path(q_start, q_goal, visual_occupancy, PATH_MAX_JOINT_STEP_RAD)
            goal_ee = checker.end_effector_position(q_goal)
            p.addUserDebugText(
                scenario.scenario_id, (goal_ee[0], goal_ee[1], goal_ee[2] + 0.015),
                textColorRGB=(0.5, 0.8, 1.0), textSize=0.55, physicsClientId=client_id,
            )
            if not direct.collision:
                final_path = (q_start, q_goal)
                validation = _validate_complete_path(checker, final_path, visual_occupancy)
                path_file = _write_path(final_path, scenario.scenario_id, "direct") if not validation.collision else None
                formal_records.append(
                    FormalPlan(scenario, None, q_start, q_goal, ik_error, direct, False, None,
                               final_path, not validation.collision, validation, path_file)
                )
                _draw_waypoint_path(final_path, checker, client_id, (0.1, 0.7, 0.1), 1.0)
                print(f"{scenario.scenario_id} {scenario.scenario_type}: DIRECT_PATH SAFE; RRT not invoked.")
                continue

            _draw_waypoint_path((q_start, q_goal), checker, client_id, (0.95, 0.1, 0.1), 1.3, f"DIRECT COLLISION {scenario.scenario_id}")
            if goal_configuration.collision:
                # A colliding endpoint must never be admitted to an RRT tree.
                failure = RRTConnectResult(False, (), 0.0, 0, 0, 0, 0, "GOAL_CONFIGURATION_COLLISION")
                formal_records.append(
                    FormalPlan(scenario, None, q_start, q_goal, ik_error, direct, True, failure,
                               (), False, None, None)
                )
                print(f"{scenario.scenario_id} {scenario.scenario_type}: direct collision; unsafe goal, RRT rejected.")
                continue

            for seed in RRT_RANDOM_SEEDS:
                config = RRTConnectConfig(
                    step_size_rad=RRT_STEP_SIZE_RAD,
                    edge_resolution_rad=PATH_MAX_JOINT_STEP_RAD,
                    goal_bias_probability=GOAL_BIAS_PROBABILITY,
                    max_iterations=MAX_ITERATIONS,
                    max_planning_time_s=MAX_PLANNING_TIME_S,
                    random_seed=seed,
                )
                planner = RRTConnectPlanner(
                    checker, visual_occupancy, active_vector, locked_vector, checker.arm_joint_limits, config
                )
                result = planner.plan(q_start, q_goal)
                validation = _validate_complete_path(checker, result.path, visual_occupancy) if result.success else None
                valid = bool(result.success and validation is not None and not validation.collision)
                path_file = _write_path(result.path, scenario.scenario_id, str(seed)) if valid else None
                formal_records.append(
                    FormalPlan(scenario, seed, q_start, q_goal, ik_error, direct, True, result,
                               result.path if valid else (), valid, validation, path_file)
                )
                print(
                    f"{scenario.scenario_id} seed={seed}: RRT {'SUCCESS' if valid else 'FAIL'} | "
                    f"time={result.planning_time_s:.3f}s | iter={result.iterations} | "
                    f"nodes={result.tree_a_nodes + result.tree_b_nodes} | reason={result.failure_reason or 'NONE'}"
                )
                if valid and not primary_rrt_success_drawn:
                    _draw_waypoint_path(result.path, checker, client_id, (0.1, 0.95, 0.1), 2.0, "RRT SAFE PATH")
                    primary_rrt_success_drawn = True

        # -------- Evaluation-only phase: GT is intentionally first read here. --------
        gt_bounds = p.getAABB(obstacle_body_id, physicsClientId=client_id)
        ground_truth_aabb = AxisAlignedBox(tuple(gt_bounds[0]), tuple(gt_bounds[1]))
        print("GT safety audit begins only after every formal visual-AABB planning decision is complete.")
        gt_results: list[bool | None] = []
        for record in formal_records:
            gt_safe = (
                _evaluate_complete_path_against_gt(checker, record.final_path, ground_truth_aabb)
                if record.final_path_estimated_safe else None
            )
            gt_results.append(gt_safe)

        with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for record, gt_safe in zip(formal_records, gt_results):
                writer.writerow(_row_from_formal(record, visual_occupancy, gt_safe))

        direct_safe = [item for item in formal_records if not item.rrt_required and item.final_path_estimated_safe]
        rrt_trials = [item for item in formal_records if item.rrt_required and item.seed is not None]
        rrt_successes = [item for item in rrt_trials if item.final_path_estimated_safe]
        times = [item.planning_result.planning_time_s for item in rrt_trials if item.planning_result is not None]
        iterations = [item.planning_result.iterations for item in rrt_trials if item.planning_result is not None]
        nodes = [
            item.planning_result.tree_a_nodes + item.planning_result.tree_b_nodes
            for item in rrt_trials if item.planning_result is not None
        ]
        lengths = [_path_length(item.final_path) for item in rrt_successes]
        planned_path_gt_collisions = sum(
            item.final_path_estimated_safe and gt_safe is False
            for item, gt_safe in zip(formal_records, gt_results)
        )
        rrt_required_scenarios = {item.scenario.scenario_id for item in formal_records if item.rrt_required}
        success_rate = len(rrt_successes) / len(rrt_trials) if rrt_trials else 0.0
        stage_pass = (
            len(PLANNING_SCENARIOS) >= 10
            and len(direct_safe) >= 3
            and len(rrt_required_scenarios) >= 5
            and bool(rrt_trials)
            and success_rate > 0.0
            and all(item.final_path_estimated_safe for item in rrt_successes)
            and planned_path_gt_collisions == 0
            and context.max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
        )
        summary = {
            "scenarios": len(PLANNING_SCENARIOS),
            "direct_safe_count": len(direct_safe),
            "rrt_required_count": len(rrt_required_scenarios),
            "rrt_trials": len(rrt_trials),
            "rrt_success_count": len(rrt_successes),
            "rrt_failure_count": len(rrt_trials) - len(rrt_successes),
            "rrt_planning_success_rate": success_rate,
            "mean_planning_time_s": mean(times) if times else None,
            "median_planning_time_s": median(times) if times else None,
            "max_planning_time_s": max(times) if times else None,
            "mean_iterations": mean(iterations) if iterations else None,
            "mean_tree_nodes": mean(nodes) if nodes else None,
            "mean_raw_path_length": mean(lengths) if lengths else None,
            "final_path_gt_collision_count": planned_path_gt_collisions,
            "self_collision_checking": "NOT_IMPLEMENTED",
            "path_max_joint_step_rad": PATH_MAX_JOINT_STEP_RAD,
            "rrt_step_size_rad": RRT_STEP_SIZE_RAD,
            "goal_bias_probability": GOAL_BIAS_PROBABILITY,
            "random_seed_count": len(RRT_RANDOM_SEEDS),
            "locked_joint_max_deviation_rad": context.max_locked_deviation,
            "result": "PASS" if stage_pass else "FAIL",
        }
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=_summary_fields())
            writer.writeheader()
            writer.writerow({field: "" if summary[field] is None else summary[field] for field in _summary_fields()})

        from plot_stage19 import generate_stage19_figures

        figures = generate_stage19_figures()
        print("\n===== STAGE 19 RRT-CONNECT SUMMARY =====")
        print(f"Scenarios={len(PLANNING_SCENARIOS)} | direct-safe={len(direct_safe)} | RRT-required={len(rrt_required_scenarios)}")
        print(f"RRT success={len(rrt_successes)}/{len(rrt_trials)} ({success_rate:.1%})")
        print(
            f"Planning time mean/median/max = {summary['mean_planning_time_s'] or 0.0:.3f}/"
            f"{summary['median_planning_time_s'] or 0.0:.3f}/{summary['max_planning_time_s'] or 0.0:.3f} s"
        )
        print(f"Final path GT collision count: {planned_path_gt_collisions}")
        print("Ground truth used for planning: False")
        print("Ground truth used for final-path evaluation: True")
        print("Self-collision checking: NOT_IMPLEMENTED")
        print(f"Locked joint max deviation: {context.max_locked_deviation:.6f} rad")
        print("CSV:", LOG_PATH)
        print("Summary:", SUMMARY_PATH)
        print("Path directory:", PATH_DIRECTORY)
        print("Figures:", *figures)
        print("Stage 19:", "PASS" if stage_pass else "FAIL")
    finally:
        if checker is not None:
            checker.close()
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)
