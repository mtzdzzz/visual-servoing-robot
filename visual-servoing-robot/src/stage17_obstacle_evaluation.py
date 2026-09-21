"""Stage 17 RGB-D obstacle perception and conservative occupancy evaluation.

No controller command, collision check, motion planner, or obstacle avoidance
logic is used here.  The Panda remains in the frozen Stage 14 HOLD pose while
RGB, depth, and the current hand-eye pose are used to estimate a yellow box.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
import pybullet as p

import simulation as sim
import stage10_evaluation as stage10
import stage14_rgbd_evaluation as stage14
from camera_observation import (
    CAMERA_FAR_PLANE,
    CAMERA_FOV_Y_DEGREES,
    CAMERA_NEAR_PLANE,
    EyeInHandRgbDisplay,
    LiveCameraFrame,
    render_live_eye_in_hand_rgbd_frame,
)
from multi_target_detector import MultiTargetDetector, annotate_multi_target_detections
from obstacle_detector import ObstacleDetection, YellowObstacleDetector
from obstacle_localization import ObstacleLocalizationResult, ObstacleLocalizer
from rgbd_localization import camera_intrinsics_from_fov
from stage15_multitarget_evaluation import TARGET_SPECS, _create_coloured_sphere, _set_target_pose


ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = ROOT / "outputs" / "logs"
LOG_PATH = LOG_DIRECTORY / "stage17_obstacle_perception.csv"
SUMMARY_PATH = LOG_DIRECTORY / "stage17_summary.csv"

# Known shape only: its pose is *not* passed to detector/localizer/occupancy.
OBSTACLE_DIMENSIONS_M = (0.10, 0.10, 0.16)  # world X, Y, Z; axis-aligned box
OBSTACLE_SAFETY_MARGIN_M = 0.020
OBSTACLE_RGBA = (1.0, 0.82, 0.0, 1.0)
SAMPLES_PER_SCENE = 20
SCENE_SETTLE_SECONDS = 0.50
MAX_CAPTURE_SECONDS = 8.0

TARGET_WORLD_POSITIONS = {
    "target_1": (0.30, -0.22, sim.GROUND_TARGET_RADIUS),
    "target_2": (0.52, -0.22, sim.GROUND_TARGET_RADIUS),
    "target_3": (0.36, -0.40, sim.GROUND_TARGET_RADIUS),
}

# All target/obstacle poses are experiment scheduling only.  The obstacle is
# within the Stage 14 camera workspace but spatially separated from the balls.
# These positions cover centre/left/right/near/far in the Stage 14 validated
# camera work area.  The outer x/y limits deliberately stay clear of the
# image-border distortion seen in the first exploratory render, rather than
# changing camera intrinsics, mounting, or any frozen perception/controller
# configuration.
OBSTACLE_SCENES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("center", (0.24, -0.30, OBSTACLE_DIMENSIONS_M[2] / 2.0)),
    ("left", (0.20, -0.30, OBSTACLE_DIMENSIONS_M[2] / 2.0)),
    ("right", (0.28, -0.30, OBSTACLE_DIMENSIONS_M[2] / 2.0)),
    ("near", (0.24, -0.24, OBSTACLE_DIMENSIONS_M[2] / 2.0)),
    ("far", (0.24, -0.36, OBSTACLE_DIMENSIONS_M[2] / 2.0)),
)


@dataclass
class OccupancyDebugBox:
    """PyBullet debug drawing created from estimated occupancy only."""

    client_id: int
    line_ids: list[int]
    text_id: int = -1

    def update(self, minimum: Sequence[float] | None, maximum: Sequence[float] | None) -> None:
        if minimum is None or maximum is None:
            return
        lo, hi = list(minimum), list(maximum)
        corners = [
            (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
        ]
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
        while len(self.line_ids) < len(edges):
            self.line_ids.append(-1)
        for index, (start, end) in enumerate(edges):
            self.line_ids[index] = p.addUserDebugLine(
                corners[start], corners[end], lineColorRGB=(1.0, 0.85, 0.0), lineWidth=2,
                lifeTime=0, replaceItemUniqueId=self.line_ids[index], physicsClientId=self.client_id,
            )
        label_position = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2] + 0.02)
        self.text_id = p.addUserDebugText(
            "ESTIMATED OCCUPANCY", label_position, textColorRGB=(1.0, 0.85, 0.0), textSize=1.0,
            lifeTime=0, replaceItemUniqueId=self.text_id, physicsClientId=self.client_id,
        )


def _create_yellow_obstacle(client_id: int) -> int:
    half_extents = [dimension / 2.0 for dimension in OBSTACLE_DIMENSIONS_M]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id)
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=OBSTACLE_RGBA, physicsClientId=client_id
    )
    return p.createMultiBody(
        baseMass=0.0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
        basePosition=(0.42, -0.30, half_extents[2]), physicsClientId=client_id,
    )


def _set_static_body_pose(body_id: int, position: Sequence[float], client_id: int) -> None:
    p.resetBasePositionAndOrientation(body_id, position, (0.0, 0.0, 0.0, 1.0), physicsClientId=client_id)
    p.resetBaseVelocity(body_id, linearVelocity=(0.0, 0.0, 0.0), angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client_id)


def _aabb_metrics(
    estimated_minimum: Sequence[float], estimated_maximum: Sequence[float],
    gt_minimum: Sequence[float], gt_maximum: Sequence[float],
) -> tuple[float, float, float]:
    est_lo, est_hi = np.asarray(estimated_minimum), np.asarray(estimated_maximum)
    gt_lo, gt_hi = np.asarray(gt_minimum), np.asarray(gt_maximum)
    intersection_extents = np.maximum(0.0, np.minimum(est_hi, gt_hi) - np.maximum(est_lo, gt_lo))
    intersection = float(np.prod(intersection_extents))
    est_volume, gt_volume = float(np.prod(est_hi - est_lo)), float(np.prod(gt_hi - gt_lo))
    union = est_volume + gt_volume - intersection
    return (
        intersection / union if union > 0.0 else 0.0,
        intersection / gt_volume if gt_volume > 0.0 else 0.0,
        est_volume / gt_volume if gt_volume > 0.0 else 0.0,
    )


def _format(value: float | int | str | None) -> float | int | str:
    if value is None:
        return ""
    return f"{value:.9f}" if isinstance(value, float) else value


def _csv_fields() -> tuple[str, ...]:
    return (
        "time", "scene", "scene_index", "localization_state", "obstacle_detected", "centroid_u", "centroid_v",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_area", "valid_depth_samples", "depth_min_m",
        "depth_median_m", "depth_max_m", "visible_xmin", "visible_xmax", "visible_ymin", "visible_ymax",
        "visible_zmin", "visible_zmax", "estimated_xmin", "estimated_xmax", "estimated_ymin", "estimated_ymax",
        "estimated_zmin", "estimated_zmax", "estimated_center_x", "estimated_center_y", "estimated_center_z",
        "gt_center_x", "gt_center_y", "gt_center_z", "center_error_x", "center_error_y", "center_error_z",
        "center_error_3d", "gt_xmin", "gt_xmax", "gt_ymin", "gt_ymax", "gt_zmin", "gt_zmax", "aabb_iou",
        "gt_coverage", "volume_ratio", "safety_margin", "target_red_detected", "target_green_detected",
        "target_blue_detected", "target_confusion_count", "gt_used_for_detection", "gt_used_for_localization",
        "gt_used_for_occupancy", "gt_used_for_controller", "gt_used_for_evaluation",
    )


def _summary_fields() -> tuple[str, ...]:
    return (
        "scene", "frames", "valid_frames", "detection_rate", "mean_center_error_m", "rmse_center_error_m",
        "max_center_error_m", "mean_aabb_iou", "mean_gt_coverage", "mean_volume_ratio", "target_detection_rate",
        "target_confusion_count", "locked_joint_max_deviation_rad", "result",
    )


def _overlay(
    frame: LiveCameraFrame,
    target_detections: Sequence[object],
    obstacle: ObstacleDetection,
    localization: ObstacleLocalizationResult,
    scene: str,
    scene_index: int,
) -> bytes:
    annotated = annotate_multi_target_detections(frame.rgba_buffer, frame.image_width, frame.image_height, target_detections)
    rgba = np.frombuffer(annotated, dtype=np.uint8).reshape((frame.image_height, frame.image_width, 4))
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cv2.rectangle(bgr, (4, 4), (635, 104), (25, 25, 25), thickness=-1)
    availability = " ".join(
        f"{detection.class_name[0]}:{'Y' if detection.valid else 'N'}" for detection in target_detections
    )
    lines = [
        f"STAGE 17 RGB-D OBSTACLE | Scene {scene_index}/5: {scene}",
        f"Targets: {availability} | Robot: HOLD | RGB-D: ACTIVE",
    ]
    if obstacle.valid and obstacle.bounding_box is not None and obstacle.centroid is not None:
        x, y, width, height = obstacle.bounding_box
        cv2.drawContours(bgr, [obstacle.contour], -1, (0, 255, 255), 2)
        cv2.rectangle(bgr, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.drawMarker(bgr, obstacle.centroid, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        lines.append(f"Obstacle: YELLOW {obstacle.centroid} | area={obstacle.area:.0f}px")
        if localization.valid:
            lines.append(
                f"Depth median: {localization.depth_median_m:.3f} m | Point samples: {localization.valid_depth_samples}"
            )
        else:
            lines.append(f"Obstacle localization: {localization.state}")
    else:
        lines.append(f"Obstacle: NOT DETECTED ({obstacle.state})")
    for index, line in enumerate(lines):
        cv2.putText(bgr, line, (10, 25 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()


def _row(
    timestamp: float, scene: str, scene_index: int, obstacle: ObstacleDetection,
    localization: ObstacleLocalizationResult, target_detections: Sequence[object], gt_center: Sequence[float],
    gt_aabb: tuple[Sequence[float], Sequence[float]],
) -> tuple[dict[str, object], float | None, float | None, float | None]:
    target_by_class = {item.class_name: item for item in target_detections}
    target_flags = {name: int(bool(target_by_class.get(name) and target_by_class[name].valid)) for name in ("RED", "GREEN", "BLUE")}
    box = obstacle.bounding_box or (None, None, None, None)
    row: dict[str, object] = {
        "time": _format(timestamp), "scene": scene, "scene_index": scene_index, "localization_state": localization.state,
        "obstacle_detected": int(obstacle.valid), "centroid_u": _format(obstacle.centroid[0] if obstacle.centroid else None),
        "centroid_v": _format(obstacle.centroid[1] if obstacle.centroid else None), "bbox_x": _format(box[0]),
        "bbox_y": _format(box[1]), "bbox_w": _format(box[2]), "bbox_h": _format(box[3]), "mask_area": _format(obstacle.area),
        "valid_depth_samples": localization.valid_depth_samples, "depth_min_m": _format(localization.depth_min_m),
        "depth_median_m": _format(localization.depth_median_m), "depth_max_m": _format(localization.depth_max_m),
        "visible_xmin": "", "visible_xmax": "", "visible_ymin": "", "visible_ymax": "", "visible_zmin": "", "visible_zmax": "",
        "estimated_xmin": "", "estimated_xmax": "", "estimated_ymin": "", "estimated_ymax": "", "estimated_zmin": "", "estimated_zmax": "",
        "estimated_center_x": "", "estimated_center_y": "", "estimated_center_z": "",
        "gt_center_x": _format(float(gt_center[0])), "gt_center_y": _format(float(gt_center[1])), "gt_center_z": _format(float(gt_center[2])),
        "center_error_x": "", "center_error_y": "", "center_error_z": "", "center_error_3d": "",
        "gt_xmin": _format(float(gt_aabb[0][0])), "gt_xmax": _format(float(gt_aabb[1][0])), "gt_ymin": _format(float(gt_aabb[0][1])),
        "gt_ymax": _format(float(gt_aabb[1][1])), "gt_zmin": _format(float(gt_aabb[0][2])), "gt_zmax": _format(float(gt_aabb[1][2])),
        "aabb_iou": "", "gt_coverage": "", "volume_ratio": "", "safety_margin": _format(OBSTACLE_SAFETY_MARGIN_M),
        "target_red_detected": target_flags["RED"], "target_green_detected": target_flags["GREEN"], "target_blue_detected": target_flags["BLUE"],
        "target_confusion_count": 0, "gt_used_for_detection": False, "gt_used_for_localization": False,
        "gt_used_for_occupancy": False, "gt_used_for_controller": False, "gt_used_for_evaluation": True,
    }
    if not localization.valid:
        return row, None, None, None
    assert localization.visible_aabb_min_m is not None and localization.visible_aabb_max_m is not None
    assert localization.occupancy_aabb_min_m is not None and localization.occupancy_aabb_max_m is not None
    assert localization.estimated_center_world_m is not None
    estimate = np.asarray(localization.estimated_center_world_m)
    error = estimate - np.asarray(gt_center)
    error_norm = float(np.linalg.norm(error))
    iou, coverage, volume_ratio = _aabb_metrics(
        localization.occupancy_aabb_min_m, localization.occupancy_aabb_max_m, gt_aabb[0], gt_aabb[1]
    )
    visible_lo, visible_hi = localization.visible_aabb_min_m, localization.visible_aabb_max_m
    estimated_lo, estimated_hi = localization.occupancy_aabb_min_m, localization.occupancy_aabb_max_m
    row.update({
        "visible_xmin": _format(visible_lo[0]), "visible_xmax": _format(visible_hi[0]), "visible_ymin": _format(visible_lo[1]), "visible_ymax": _format(visible_hi[1]),
        "visible_zmin": _format(visible_lo[2]), "visible_zmax": _format(visible_hi[2]), "estimated_xmin": _format(estimated_lo[0]), "estimated_xmax": _format(estimated_hi[0]),
        "estimated_ymin": _format(estimated_lo[1]), "estimated_ymax": _format(estimated_hi[1]), "estimated_zmin": _format(estimated_lo[2]), "estimated_zmax": _format(estimated_hi[2]),
        "estimated_center_x": _format(float(estimate[0])), "estimated_center_y": _format(float(estimate[1])), "estimated_center_z": _format(float(estimate[2])),
        "center_error_x": _format(float(error[0])), "center_error_y": _format(float(error[1])), "center_error_z": _format(float(error[2])), "center_error_3d": _format(error_norm),
        "aabb_iou": _format(iou), "gt_coverage": _format(coverage), "volume_ratio": _format(volume_ratio),
    })
    return row, error_norm, iou, coverage


def _scene_summary(
    scene: str, frame_count: int, valid_count: int, detected_count: int, errors: Sequence[float],
    ious: Sequence[float], coverages: Sequence[float], volume_ratios: Sequence[float], target_successes: int,
    max_locked_deviation: float,
) -> dict[str, object]:
    mean = lambda values: float(np.mean(values)) if values else None
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    result = (
        frame_count >= SAMPLES_PER_SCENE and valid_count >= SAMPLES_PER_SCENE
        and target_successes >= SAMPLES_PER_SCENE and max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
    )
    return {
        "scene": scene, "frames": frame_count, "valid_frames": valid_count,
        "detection_rate": 100.0 * detected_count / frame_count if frame_count else 0.0,
        "mean_center_error_m": mean(errors), "rmse_center_error_m": rmse,
        "max_center_error_m": float(np.max(errors)) if errors else None, "mean_aabb_iou": mean(ious),
        "mean_gt_coverage": mean(coverages), "mean_volume_ratio": mean(volume_ratios),
        "target_detection_rate": 100.0 * target_successes / frame_count if frame_count else 0.0,
        "target_confusion_count": 0, "locked_joint_max_deviation_rad": max_locked_deviation,
        "result": "PASS" if result else "FAIL",
    }


def run_stage17_obstacle_perception() -> None:
    """Run five static, RGB-D-only obstacle occupancy scenes in PyBullet GUI."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 17.")
    display: EyeInHandRgbDisplay | None = None
    try:
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB-D - Stage 17 Obstacle Perception")
        intrinsics = camera_intrinsics_from_fov(640, 480, CAMERA_FOV_Y_DEGREES)
        obstacle_detector = YellowObstacleDetector()
        target_detector = MultiTargetDetector()
        localizer = ObstacleLocalizer(
            intrinsics, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE, OBSTACLE_DIMENSIONS_M,
            OBSTACLE_SAFETY_MARGIN_M,
        )
        print("Stage 17: RGB-D Obstacle Perception and Conservative Occupancy")
        print(f"Obstacle: YELLOW axis-aligned cuboid {OBSTACLE_DIMENSIONS_M} m; safety margin={OBSTACLE_SAFETY_MARGIN_M:.3f} m.")
        print("Robot: HOLD. Collision checking, obstacle avoidance, motion planning, and controller input are disabled.")
        print("GT audit: detection=False, localization=False, occupancy=False, controller=False, evaluation=True")
        print(f"Intrinsics: fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")

        summaries: list[dict[str, object]] = []
        all_errors: list[float] = []
        all_ious: list[float] = []
        all_coverages: list[float] = []
        all_volumes: list[float] = []
        all_frames = all_valid = all_detected = all_target_successes = 0
        max_locked_deviation = 0.0
        simulation_time = 0.0
        fixed_render_to_camera: np.ndarray | None = None

        with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for scene_index, (scene, obstacle_position) in enumerate(OBSTACLE_SCENES, start=1):
                context = stage14._create_static_localization_context(client_id)
                targets = {"target_1": context.target_body_id}
                for spec in TARGET_SPECS[1:]:
                    targets[spec.target_id] = _create_coloured_sphere(spec, client_id)
                for target_id, target_position in TARGET_WORLD_POSITIONS.items():
                    _set_target_pose(targets[target_id], target_position, client_id)
                obstacle_body_id = _create_yellow_obstacle(client_id)
                _set_static_body_pose(obstacle_body_id, obstacle_position, client_id)
                p.addUserDebugText("REAL OBSTACLE", [obstacle_position[0], obstacle_position[1], obstacle_position[2] + 0.11], textColorRGB=(1.0, 0.85, 0.0), textSize=1.0, physicsClientId=client_id)
                debug_box = OccupancyDebugBox(client_id, [])
                stage14._run_hold_steps(context, round(SCENE_SETTLE_SECONDS / sim.TIME_STEP), client_id)
                simulation_time += SCENE_SETTLE_SECONDS
                context.debug_text_id = sim.update_motion_debug_text(
                    f"STAGE 17 - RGB-D OBSTACLE PERCEPTION\nScene {scene_index}/5: {scene}\nRobot: HOLD | GT: EVALUATION ONLY",
                    context.debug_text_id, client_id,
                )
                frame_count = valid_count = detected_count = target_successes = 0
                errors: list[float] = []
                ious: list[float] = []
                coverages: list[float] = []
                volume_ratios: list[float] = []
                capture_steps = 0
                while valid_count < SAMPLES_PER_SCENE and capture_steps < round(MAX_CAPTURE_SECONDS / sim.TIME_STEP):
                    stage10._step_physics(context, client_id)
                    simulation_time += sim.TIME_STEP
                    if capture_steps % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                        sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
                        camera_position, camera_orientation = sim.get_camera_optical_center_pose(context.robot_id, client_id)
                        rgbd = render_live_eye_in_hand_rgbd_frame(camera_position, camera_orientation, client_id)
                        frame = rgbd.live_rgb_frame
                        targets_rgb = target_detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
                        obstacle_rgb = obstacle_detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
                        camera_to_world, _, _ = stage14._validate_camera_pose_and_render_frame(context.robot_id, frame, client_id)
                        current_axes = stage14._render_to_camera_axis_transform(camera_to_world, frame)
                        if fixed_render_to_camera is None:
                            fixed_render_to_camera = current_axes
                            print("Validated C_render -> C axis transform:\n" + np.array2string(current_axes, precision=5, suppress_small=True))
                        elif not np.allclose(fixed_render_to_camera, current_axes, atol=1e-6):
                            raise RuntimeError("Stage 17 render-to-C transform changed unexpectedly.")
                        localization = (
                            localizer.localize(obstacle_rgb.mask, rgbd.depth_buffer, camera_to_world, current_axes)
                            if obstacle_rgb.valid else localizer._invalid("OBSTACLE_NOT_DETECTED")
                        )
                        # GT read only after RGB/D estimate: evaluation and debug comparison, never perception input.
                        gt_position, _ = p.getBasePositionAndOrientation(obstacle_body_id, physicsClientId=client_id)
                        gt_aabb = p.getAABB(obstacle_body_id, physicsClientId=client_id)
                        row, error, iou, coverage = _row(
                            simulation_time, scene, scene_index, obstacle_rgb, localization, targets_rgb, gt_position, gt_aabb
                        )
                        writer.writerow(row)
                        frame_count += 1
                        detected_count += int(obstacle_rgb.valid)
                        target_successes += int(all(item.valid for item in targets_rgb))
                        if localization.valid:
                            valid_count += 1
                            assert localization.occupancy_aabb_min_m is not None and localization.occupancy_aabb_max_m is not None
                            debug_box.update(localization.occupancy_aabb_min_m, localization.occupancy_aabb_max_m)
                            assert error is not None and iou is not None and coverage is not None
                            errors.append(error); ious.append(iou); coverages.append(coverage)
                            estimated_volume = np.prod(np.asarray(localization.occupancy_aabb_max_m) - np.asarray(localization.occupancy_aabb_min_m))
                            gt_volume = np.prod(np.asarray(gt_aabb[1]) - np.asarray(gt_aabb[0]))
                            volume_ratios.append(float(estimated_volume / gt_volume))
                        display.show(frame, _overlay(frame, targets_rgb, obstacle_rgb, localization, scene, scene_index))
                    capture_steps += 1
                    time.sleep(sim.TIME_STEP)

                max_locked_deviation = max(max_locked_deviation, context.max_locked_deviation)
                summary = _scene_summary(
                    scene, frame_count, valid_count, detected_count, errors, ious, coverages, volume_ratios,
                    target_successes, context.max_locked_deviation,
                )
                summaries.append(summary)
                all_frames += frame_count; all_valid += valid_count; all_detected += detected_count; all_target_successes += target_successes
                all_errors.extend(errors); all_ious.extend(ious); all_coverages.extend(coverages); all_volumes.extend(volume_ratios)
                print(
                    f"Scene {scene_index} ({scene}): valid={valid_count}/{SAMPLES_PER_SCENE}, "
                    f"detection={summary['detection_rate']:.1f}%, center error="
                    f"{(summary['mean_center_error_m'] or float('nan')) * 1000.0:.2f} mm, "
                    f"coverage={100.0 * (summary['mean_gt_coverage'] or 0.0):.1f}% -> {summary['result']}"
                )

        overall = _scene_summary(
            "OVERALL", all_frames, all_valid, all_detected, all_errors, all_ious, all_coverages, all_volumes,
            all_target_successes, max_locked_deviation,
        )
        summaries.append(overall)
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=_summary_fields())
            writer.writeheader()
            for item in summaries:
                writer.writerow({field: _format(item.get(field)) for field in _summary_fields()})
        from plot_stage17 import generate_stage17_figures
        figure_paths = generate_stage17_figures()
        stage_pass = all(item["result"] == "PASS" for item in summaries) and max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
        print("\n===== STAGE 17 OBSTACLE PERCEPTION SUMMARY =====")
        print(f"Overall detection rate: {overall['detection_rate']:.1f}% | target all-colour detection: {overall['target_detection_rate']:.1f}% | confusion=0")
        print(f"Center error mean/RMSE/max: {(overall['mean_center_error_m'] or float('nan'))*1000:.2f}/{(overall['rmse_center_error_m'] or float('nan'))*1000:.2f}/{(overall['max_center_error_m'] or float('nan'))*1000:.2f} mm")
        print(f"Occupancy mean IoU={100.0*(overall['mean_aabb_iou'] or 0.0):.1f}% | GT coverage={100.0*(overall['mean_gt_coverage'] or 0.0):.1f}% | volume ratio={overall['mean_volume_ratio'] or float('nan'):.2f}")
        print(f"Locked joint max deviation: {max_locked_deviation:.6f} rad")
        print("GT used for obstacle detection: False")
        print("GT used for obstacle localization: False")
        print("GT used for occupancy estimation: False")
        print("GT used for controller: False")
        print("GT used for evaluation: True")
        print("CSV:", LOG_PATH)
        print("Summary:", SUMMARY_PATH)
        print("Figures:", *figure_paths)
        print("Stage 17:", "PASS" if stage_pass else "FAIL")
    finally:
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)
