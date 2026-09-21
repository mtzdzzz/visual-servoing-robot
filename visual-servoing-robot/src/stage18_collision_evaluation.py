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
import time
from typing import Sequence

import numpy as np
import pybullet as p

import simulation as sim
import stage10_evaluation as stage10
import stage14_rgbd_evaluation as stage14
import stage17_obstacle_evaluation as stage17
from camera_observation import (
    CAMERA_FAR_PLANE,
    CAMERA_FOV_Y_DEGREES,
    CAMERA_NEAR_PLANE,
    EyeInHandRgbDisplay,
    render_live_eye_in_hand_rgbd_frame,
)
from collision_checker import AxisAlignedBox, CollisionChecker, PathCollisionResult
from multi_target_detector import MultiTargetDetector
from obstacle_detector import YellowObstacleDetector
from obstacle_localization import ObstacleLocalizer
from rgbd_localization import camera_intrinsics_from_fov
from stage15_multitarget_evaluation import TARGET_SPECS, _create_coloured_sphere, _set_target_pose


ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = ROOT / "outputs" / "logs"
LOG_PATH = LOG_DIRECTORY / "stage18_collision_checking.csv"
SUMMARY_PATH = LOG_DIRECTORY / "stage18_summary.csv"
VISUAL_OCCUPANCY_FRAMES = 20
PATH_MAX_JOINT_STEP_RAD = 0.05
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


def _capture_visual_occupancy(
    context: stage10.ReadyContext,
    display: EyeInHandRgbDisplay,
    client_id: int,
) -> tuple[AxisAlignedBox, stage17.OccupancyDebugBox, int]:
    """Collect a fresh Stage-17 occupancy from RGB, mask, synchronized depth.

    No obstacle position/AABB/ID is read here.  The returned AABB is built
    from median vision-only Stage 17 occupancy bounds and is the *only*
    obstacle input supplied to formal collision decisions.
    """

    intrinsics = camera_intrinsics_from_fov(640, 480, CAMERA_FOV_Y_DEGREES)
    localizer = ObstacleLocalizer(
        intrinsics, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE,
        stage17.OBSTACLE_DIMENSIONS_M, stage17.OBSTACLE_SAFETY_MARGIN_M,
    )
    obstacle_detector = YellowObstacleDetector()
    target_detector = MultiTargetDetector()
    debug_box = stage17.OccupancyDebugBox(client_id, [])
    occupancy_minima: list[np.ndarray] = []
    occupancy_maxima: list[np.ndarray] = []
    render_to_camera: np.ndarray | None = None
    steps = 0
    maximum_steps = round(stage17.MAX_CAPTURE_SECONDS / sim.TIME_STEP)
    while len(occupancy_minima) < VISUAL_OCCUPANCY_FRAMES and steps < maximum_steps:
        stage10._step_physics(context, client_id)
        if steps % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
            sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
            camera_position, camera_orientation = sim.get_camera_optical_center_pose(context.robot_id, client_id)
            rgbd = render_live_eye_in_hand_rgbd_frame(camera_position, camera_orientation, client_id)
            frame = rgbd.live_rgb_frame
            target_detections = target_detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
            obstacle_detection = obstacle_detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
            camera_to_world, _, _ = stage14._validate_camera_pose_and_render_frame(context.robot_id, frame, client_id)
            current_axes = stage14._render_to_camera_axis_transform(camera_to_world, frame)
            if render_to_camera is None:
                render_to_camera = current_axes
            elif not np.allclose(render_to_camera, current_axes, atol=1e-6):
                raise RuntimeError("Stage 18 render-to-C axis transform changed unexpectedly.")
            localization = (
                localizer.localize(obstacle_detection.mask, rgbd.depth_buffer, camera_to_world, current_axes)
                if obstacle_detection.valid else localizer._invalid("OBSTACLE_NOT_DETECTED")
            )
            if localization.valid:
                assert localization.occupancy_aabb_min_m is not None and localization.occupancy_aabb_max_m is not None
                occupancy_minima.append(np.asarray(localization.occupancy_aabb_min_m, dtype=np.float64))
                occupancy_maxima.append(np.asarray(localization.occupancy_aabb_max_m, dtype=np.float64))
                debug_box.update(localization.occupancy_aabb_min_m, localization.occupancy_aabb_max_m)
            display.show(
                frame,
                stage17._overlay(frame, target_detections, obstacle_detection, localization, "center", 1),
            )
        steps += 1
        time.sleep(sim.TIME_STEP)
    if len(occupancy_minima) != VISUAL_OCCUPANCY_FRAMES:
        raise RuntimeError(
            f"Stage 18 needs {VISUAL_OCCUPANCY_FRAMES} valid visual occupancy frames; got {len(occupancy_minima)}."
        )
    occupancy = AxisAlignedBox(
        tuple(float(value) for value in np.median(np.stack(occupancy_minima), axis=0)),
        tuple(float(value) for value in np.median(np.stack(occupancy_maxima), axis=0)),
    )
    debug_box.update(occupancy.minimum, occupancy.maximum)
    return occupancy, debug_box, steps


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
        context = stage14._create_static_localization_context(client_id)
        target_ids = {"target_1": context.target_body_id}
        for spec in TARGET_SPECS[1:]:
            target_ids[spec.target_id] = _create_coloured_sphere(spec, client_id)
        for target_id, target_position in stage17.TARGET_WORLD_POSITIONS.items():
            _set_target_pose(target_ids[target_id], target_position, client_id)
        obstacle_body_id = stage17._create_yellow_obstacle(client_id)
        obstacle_position = dict(stage17.OBSTACLE_SCENES)["center"]
        stage17._set_static_body_pose(obstacle_body_id, obstacle_position, client_id)
        p.addUserDebugText(
            "REAL OBSTACLE", [obstacle_position[0], obstacle_position[1], obstacle_position[2] + 0.11],
            textColorRGB=(1.0, 0.85, 0.0), textSize=1.0, physicsClientId=client_id,
        )
        context.debug_text_id = sim.update_motion_debug_text(
            "STAGE 18 - VISION-BASED COLLISION CHECKING\nRobot: HOLD | Candidate paths are DIRECT-model evaluation only",
            context.debug_text_id, client_id,
        )
        stage14._run_hold_steps(context, round(stage17.SCENE_SETTLE_SECONDS / sim.TIME_STEP), client_id)
        print("Stage 18: Vision-Based Collision Checking")
        print("Robot geometry: world AABBs for Panda base, arm, link8/hand, and fingers. Self-collision: NOT IMPLEMENTED.")
        print(f"Path resolution: maximum per-joint interpolation increment = {PATH_MAX_JOINT_STEP_RAD:.3f} rad.")
        print("Formal collision input: Stage 17 estimated occupancy only. GT is evaluation-only after formal checks.")
        visual_occupancy, _debug_box, captured = _capture_visual_occupancy(context, display, client_id)
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
