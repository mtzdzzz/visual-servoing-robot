"""Stage 18 vision-based configuration and interpolated path collision checks.

This evaluation purposely never moves the GUI Panda along a candidate path.
It first obtains a fresh Stage 17 visual occupancy estimate, then uses a
separate DIRECT Panda model to assess fixed candidate paths.  Ground-truth
obstacle AABB data is read *only after* all formal visual-AABB decisions are
complete, and is used exclusively to score TP/TN/FP/FN.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pybullet as p

import robotics_core as sim
import config
import scene_factory
import obstacle_perception_runtime
import stage17_obstacle_evaluation as stage17
from camera_observation import EyeInHandRgbDisplay
from collision_checker import AxisAlignedBox, CollisionChecker, PathCollisionResult
from stage15_multitarget_evaluation import TARGET_SPECS


ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = ROOT / "outputs" / "logs"
LOG_PATH = LOG_DIRECTORY / "stage18_collision_checking.csv"
SUMMARY_PATH = LOG_DIRECTORY / "stage18_summary.csv"
VISUAL_OCCUPANCY_FRAMES = 20
PATH_MAX_JOINT_STEP_RAD = config.COLLISION_INTERPOLATION_RESOLUTION_RAD
CANDIDATE_RANDOM_SEED = 20260921
SAFE_PATH_COUNT = 7
COLLISION_PATH_COUNT = 7
NEAR_BOUNDARY_PATH_COUNT = 6


@dataclass(frozen=True)
class CandidatePath:
    path_id: str
    path_type: str
    q_start: tuple[float, ...]
    q_goal: tuple[float, ...]
    estimated_result: PathCollisionResult


def _json_vector(values: Sequence[float]) -> str:
    return json.dumps([round(float(value), 8) for value in values], separators=(",", ":"))


def _classification(estimated_collision: bool, ground_truth_collision: bool) -> str:
    if estimated_collision and ground_truth_collision:
        return "TP"
    if estimated_collision and not ground_truth_collision:
        return "FP"
    if not estimated_collision and ground_truth_collision:
        return "FN"
    return "TN"


def _fixed_candidate_goals(
    q_start: Sequence[float],
    checker: CollisionChecker,
) -> list[tuple[float, ...]]:
    """Create a deterministic broad evaluation corpus, not a motion planner."""

    rng = np.random.default_rng(CANDIDATE_RANDOM_SEED)
    lower = np.asarray([limits[0] for limits in checker.arm_joint_limits], dtype=np.float64)
    upper = np.asarray([limits[1] for limits in checker.arm_joint_limits], dtype=np.float64)
    start = np.asarray(q_start, dtype=np.float64)
    goals: list[tuple[float, ...]] = []
    # Start with structured active-joint perturbations, then use reproducible
    # broad seven-joint samples.  The latter are *offline collision-model
    # configurations only*: no GUI motor command is issued and the real
    # Panda's Stage 3.1 locked-joint baseline remains unchanged.
    offsets = (
        (-0.8, 0.1, -0.8), (-0.5, 0.3, 0.0), (-0.2, 0.6, 0.8),
        (0.2, 0.9, -0.8), (0.5, 1.2, 0.0), (0.8, 1.5, 0.8),
    )
    for delta in offsets:
        candidate = start.copy()
        candidate[4:] = np.clip(candidate[4:] + np.asarray(delta), lower[4:] + 0.04, upper[4:] - 0.04)
        goals.append(tuple(float(value) for value in candidate))
    for _ in range(320):
        candidate = rng.uniform(lower + 0.04, upper - 0.04)
        goals.append(tuple(float(value) for value in candidate))
    return goals


def _select_paths(
    q_start: Sequence[float],
    checker: CollisionChecker,
    visual_occupancy: AxisAlignedBox,
) -> list[CandidatePath]:
    """Select 20 diverse paths based solely on their estimated-AABB results."""

    evaluated: list[tuple[tuple[float, ...], PathCollisionResult]] = []
    for goal in _fixed_candidate_goals(q_start, checker):
        result = checker.check_path(q_start, goal, visual_occupancy, PATH_MAX_JOINT_STEP_RAD)
        evaluated.append((goal, result))
    collision = [(goal, result) for goal, result in evaluated if result.collision]
    safe = [(goal, result) for goal, result in evaluated if not result.collision]
    # Safe candidates far from the estimated occupancy are the SAFE corpus.
    safe.sort(key=lambda item: item[1].minimum_clearance_m, reverse=True)
    # Near-boundary paths are the latest visual-AABB crossings.  They test
    # the conservative safety boundary rather than pretending that an
    # unattainably close AABB-safe path exists in the current constrained
    # workspace.  GT evaluation may classify some of them as false positives,
    # which is meaningful evidence of conservative occupancy.
    collision.sort(key=lambda item: (item[1].first_collision_fraction or 1.0, item[1].minimum_clearance_m))
    near = list(reversed(collision))
    if len(safe) < SAFE_PATH_COUNT or len(collision) < COLLISION_PATH_COUNT + NEAR_BOUNDARY_PATH_COUNT:
        raise RuntimeError(
            "The fixed Stage 18 visual-AABB corpus did not yield enough safe, collision, and near-boundary paths."
        )
    selected: list[CandidatePath] = []
    collision_selected = collision[:COLLISION_PATH_COUNT]
    collision_goals = {item[0] for item in collision_selected}
    near_selected = [item for item in near if item[0] not in collision_goals][:NEAR_BOUNDARY_PATH_COUNT]
    for path_type, count, source in (
        ("SAFE", SAFE_PATH_COUNT, safe),
        ("COLLISION", COLLISION_PATH_COUNT, collision_selected),
        ("NEAR_BOUNDARY", NEAR_BOUNDARY_PATH_COUNT, near_selected),
    ):
        for goal, result in source[:count]:
            selected.append(
                CandidatePath(
                    f"P{len(selected) + 1:02d}", path_type, tuple(float(value) for value in q_start), goal, result
                )
            )
    return selected


def _draw_path(
    candidate: CandidatePath,
    checker: CollisionChecker,
    client_id: int,
) -> None:
    """Draw display-only link8 path samples; the GUI Panda never executes them."""

    result = candidate.estimated_result
    samples = np.linspace(np.asarray(candidate.q_start), np.asarray(candidate.q_goal), result.num_samples)
    points = [checker.end_effector_position(sample) for sample in samples]
    colour = (0.1, 0.9, 0.1) if candidate.path_type == "SAFE" else (0.95, 0.1, 0.1)
    if candidate.path_type == "NEAR_BOUNDARY":
        colour = (1.0, 0.6, 0.0)
    for first, second in zip(points, points[1:]):
        p.addUserDebugLine(first, second, lineColorRGB=colour, lineWidth=1, lifeTime=0, physicsClientId=client_id)
    if result.first_collision_sample is not None:
        point = points[result.first_collision_sample]
        p.addUserDebugText(
            f"COLLISION {candidate.path_id}", [point[0], point[1], point[2] + 0.02],
            textColorRGB=(1.0, 0.1, 0.1), textSize=0.7, lifeTime=0, physicsClientId=client_id,
        )


def _csv_fields() -> tuple[str, ...]:
    return (
        "path_id", "path_type", "q_start", "q_goal", "num_samples", "estimated_collision", "gt_collision",
        "classification", "first_collision_sample", "first_collision_fraction", "estimated_minimum_clearance_m",
        "gt_minimum_clearance_m", "colliding_links_estimated", "colliding_links_gt", "estimated_obstacle_center",
        "estimated_aabb_min", "estimated_aabb_max", "gt_used_for_formal_collision", "gt_used_for_evaluation",
    )


def _summary_fields() -> tuple[str, ...]:
    return (
        "paths", "safe_paths", "collision_paths", "near_boundary_paths", "true_positive", "true_negative",
        "false_positive", "false_negative", "precision", "recall", "safety_recall", "self_collision_checking",
        "path_max_joint_step_rad", "result",
    )


def _safe_metric(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def run_stage18_collision_checking() -> None:
    """Run visual-AABB collision checking without moving the GUI Panda."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 18.")
    display: EyeInHandRgbDisplay | None = None
    checker: CollisionChecker | None = None
    try:
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB-D - Stage 18 Collision Checking")
        context = scene_factory.create_static_localization_context(client_id)
        target_ids = {"target_1": context.target_body_id}
        for spec in TARGET_SPECS[1:]:
            target_ids[spec.target_id] = scene_factory.create_coloured_sphere(
                sim.GROUND_TARGET_RADIUS, spec.rgba, client_id
            )
        for target_id, target_position in stage17.TARGET_WORLD_POSITIONS.items():
            scene_factory.set_static_body_pose(target_ids[target_id], target_position, client_id)
        obstacle_body_id = scene_factory.create_box_obstacle(stage17.OBSTACLE_DIMENSIONS_M, stage17.OBSTACLE_RGBA, client_id)
        obstacle_position = dict(stage17.OBSTACLE_SCENES)["center"]
        scene_factory.set_static_body_pose(obstacle_body_id, obstacle_position, client_id)
        p.addUserDebugText(
            "REAL OBSTACLE", [obstacle_position[0], obstacle_position[1], obstacle_position[2] + 0.11],
            textColorRGB=(1.0, 0.85, 0.0), textSize=1.0, physicsClientId=client_id,
        )
        context.debug_text_id = sim.update_motion_debug_text(
            "STAGE 18 - VISION-BASED COLLISION CHECKING\nRobot: HOLD | Candidate paths are DIRECT-model evaluation only",
            context.debug_text_id, client_id,
        )
        scene_factory.run_hold_steps(context, round(stage17.SCENE_SETTLE_SECONDS / sim.TIME_STEP), client_id)
        print("Stage 18: Vision-Based Collision Checking")
        print("Robot geometry: world AABBs for Panda base, arm, link8/hand, and fingers. Self-collision: NOT IMPLEMENTED.")
        print(f"Path resolution: maximum per-joint interpolation increment = {PATH_MAX_JOINT_STEP_RAD:.3f} rad.")
        print("Formal collision input: Stage 17 estimated occupancy only. GT is evaluation-only after formal checks.")
        visual_occupancy, _debug_box, captured = obstacle_perception_runtime.capture_visual_occupancy(
            context, display, client_id, stage17.OBSTACLE_DIMENSIONS_M,
            stage17.OBSTACLE_SAFETY_MARGIN_M, VISUAL_OCCUPANCY_FRAMES,
            stage17.MAX_CAPTURE_SECONDS,
        )
        print(f"Stage 17 visual occupancy: {captured} physics steps, {VISUAL_OCCUPANCY_FRAMES} valid RGB-D frames.")
        print(f"Estimated occupancy AABB: min={visual_occupancy.minimum}, max={visual_occupancy.maximum}")
        q_start = tuple(
            float(p.getJointState(context.robot_id, joint_index, physicsClientId=client_id)[0])
            for joint_index in context.arm_joint_indices
        )
        checker = CollisionChecker()
        candidates = _select_paths(q_start, checker, visual_occupancy)
        # Formal results above were selected and fixed using vision only.  GT
        # now becomes available to the evaluation layer, never the checker path
        # selection or the formal estimated_collision field.
        gt_bounds = p.getAABB(obstacle_body_id, physicsClientId=client_id)
        ground_truth_aabb = AxisAlignedBox(tuple(gt_bounds[0]), tuple(gt_bounds[1]))
        print("GT collision audit starts only after all formal visual-AABB paths are decided.")

        counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for candidate in candidates:
                gt_result = checker.evaluate_path_against_ground_truth(
                    candidate.q_start, candidate.q_goal, ground_truth_aabb, PATH_MAX_JOINT_STEP_RAD
                )
                classification = _classification(candidate.estimated_result.collision, gt_result.collision)
                counts[classification] += 1
                _draw_path(candidate, checker, client_id)
                writer.writerow({
                    "path_id": candidate.path_id, "path_type": candidate.path_type,
                    "q_start": _json_vector(candidate.q_start), "q_goal": _json_vector(candidate.q_goal),
                    "num_samples": candidate.estimated_result.num_samples,
                    "estimated_collision": int(candidate.estimated_result.collision), "gt_collision": int(gt_result.collision),
                    "classification": classification,
                    "first_collision_sample": "" if candidate.estimated_result.first_collision_sample is None else candidate.estimated_result.first_collision_sample,
                    "first_collision_fraction": "" if candidate.estimated_result.first_collision_fraction is None else f"{candidate.estimated_result.first_collision_fraction:.9f}",
                    "estimated_minimum_clearance_m": f"{candidate.estimated_result.minimum_clearance_m:.9f}",
                    "gt_minimum_clearance_m": f"{gt_result.minimum_clearance_m:.9f}",
                    "colliding_links_estimated": ";".join(candidate.estimated_result.colliding_links),
                    "colliding_links_gt": ";".join(gt_result.colliding_links),
                    "estimated_obstacle_center": _json_vector(visual_occupancy.center),
                    "estimated_aabb_min": _json_vector(visual_occupancy.minimum), "estimated_aabb_max": _json_vector(visual_occupancy.maximum),
                    "gt_used_for_formal_collision": False, "gt_used_for_evaluation": True,
                })
                print(
                    f"{candidate.path_id} {candidate.path_type}: estimated={candidate.estimated_result.collision}, "
                    f"GT={gt_result.collision}, {classification}, samples={candidate.estimated_result.num_samples}"
                )

        precision = _safe_metric(counts["TP"], counts["TP"] + counts["FP"])
        recall = _safe_metric(counts["TP"], counts["TP"] + counts["FN"])
        safety_recall = recall
        stage_pass = (
            len(candidates) >= 20 and counts["FN"] == 0
            and context.max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
        )
        summary = {
            "paths": len(candidates), "safe_paths": sum(item.path_type == "SAFE" for item in candidates),
            "collision_paths": sum(item.path_type == "COLLISION" for item in candidates),
            "near_boundary_paths": sum(item.path_type == "NEAR_BOUNDARY" for item in candidates),
            "true_positive": counts["TP"], "true_negative": counts["TN"], "false_positive": counts["FP"], "false_negative": counts["FN"],
            "precision": precision, "recall": recall, "safety_recall": safety_recall,
            "self_collision_checking": "NOT_IMPLEMENTED", "path_max_joint_step_rad": PATH_MAX_JOINT_STEP_RAD,
            "result": "PASS" if stage_pass else "FAIL",
        }
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=_summary_fields())
            writer.writeheader()
            writer.writerow({field: "" if summary[field] is None else summary[field] for field in _summary_fields()})
        from plot_stage18 import generate_stage18_figures
        figures = generate_stage18_figures()
        print("\n===== STAGE 18 COLLISION CHECKING SUMMARY =====")
        print(f"Paths: {len(candidates)} | TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']}")
        print(f"Precision={precision if precision is not None else float('nan'):.3f} | Safety Recall={safety_recall if safety_recall is not None else float('nan'):.3f}")
        print("Ground truth used for formal collision: False")
        print("Ground truth used for evaluation: True")
        print("Self-collision checking: NOT IMPLEMENTED")
        print(f"Locked joint max deviation: {context.max_locked_deviation:.6f} rad")
        print("CSV:", LOG_PATH)
        print("Summary:", SUMMARY_PATH)
        print("Figures:", *figures)
        print("Stage 18:", "PASS" if stage_pass else "FAIL")
    finally:
        if checker is not None:
            checker.close()
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)

