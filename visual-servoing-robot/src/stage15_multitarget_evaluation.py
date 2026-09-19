"""Stage 15 multi-target RGB-D perception validation.

The frozen robot, camera and Stage 14 geometry remain unchanged.  This module
adds only RGB-based RED/GREEN/BLUE detection, passes each colour's own mask to
the Stage 14 localizer, and evaluates the resulting independent positions.
No result is selected or sent to visual servo, IK, or Panda motor control.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
import pybullet as p

import simulation as sim
import stage14_rgbd_evaluation as stage14
import stage10_evaluation as stage10
from camera_observation import (
    CAMERA_FAR_PLANE,
    CAMERA_FOV_Y_DEGREES,
    CAMERA_NEAR_PLANE,
    EyeInHandRgbDisplay,
    LiveCameraFrame,
    render_live_eye_in_hand_rgbd_frame,
)
from multi_target_detector import (
    TARGET_CLASSES,
    DetectedTarget,
    MultiTargetDetector,
    annotate_multi_target_detections,
)
from rgbd_localization import (
    CameraIntrinsics,
    DepthStatistics,
    RGBDLocalizationResult,
    RGBDTargetLocalizer,
    camera_intrinsics_from_fov,
)


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / "logs"
LOG_PATH = LOG_DIRECTORY / "stage15_multitarget.csv"
SUMMARY_PATH = LOG_DIRECTORY / "stage15_summary.csv"
TARGET_RADIUS_M = sim.GROUND_TARGET_RADIUS
SAMPLES_PER_SCENE = 20
SCENE_SETTLE_SECONDS = 0.50
MAX_CAPTURE_SECONDS = 8.0


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    class_name: str
    rgba: tuple[float, float, float, float]
    marker_rgb: tuple[float, float, float]


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("target_1", "RED", (1.0, 0.0, 0.0, 1.0), (1.0, 0.1, 0.1)),
    TargetSpec("target_2", "GREEN", (0.0, 1.0, 0.0, 1.0), (0.1, 1.0, 0.1)),
    TargetSpec("target_3", "BLUE", (0.0, 0.0, 1.0, 1.0), (0.1, 0.4, 1.0)),
)


# All centre separations are at least 0.10 m (> 2*radius).  Scene 2 places
# RED/GREEN near each other in image space while still avoiding collision.
SCENE_CONFIGURATIONS: tuple[tuple[str, dict[str, tuple[float, float, float]]], ...] = (
    (
        "dispersed",
        {
            "target_1": (0.30, -0.22, TARGET_RADIUS_M),
            "target_2": (0.41, -0.22, TARGET_RADIUS_M),
            "target_3": (0.35, -0.34, TARGET_RADIUS_M),
        },
    ),
    (
        "red_green_close",
        {
            "target_1": (0.31, -0.25, TARGET_RADIUS_M),
            "target_2": (0.41, -0.25, TARGET_RADIUS_M),
            "target_3": (0.36, -0.36, TARGET_RADIUS_M),
        },
    ),
    (
        "blue_farther",
        {
            "target_1": (0.30, -0.22, TARGET_RADIUS_M),
            "target_2": (0.42, -0.22, TARGET_RADIUS_M),
            "target_3": (0.36, -0.40, TARGET_RADIUS_M),
        },
    ),
    (
        "green_image_edge",
        {
            "target_1": (0.30, -0.22, TARGET_RADIUS_M),
            "target_2": (0.55, -0.25, TARGET_RADIUS_M),
            "target_3": (0.35, -0.35, TARGET_RADIUS_M),
        },
    ),
    (
        "three_depths",
        {
            "target_1": (0.30, -0.19, TARGET_RADIUS_M),
            "target_2": (0.36, -0.30, TARGET_RADIUS_M),
            "target_3": (0.44, -0.42, TARGET_RADIUS_M),
        },
    ),
)


@dataclass(frozen=True)
class LocalizedTarget:
    """One target's detector and RGB-D result, with evaluation-only GT fields."""

    target_id: str
    class_name: str
    detection: DetectedTarget
    localization: RGBDLocalizationResult
    ground_truth_world_m: tuple[float, float, float]
    error_3d_m: float | None


@dataclass
class TargetDebugMarker:
    """One colour-coded world marker based only on this target's estimate."""

    client_id: int
    class_name: str
    colour: tuple[float, float, float]
    line_ids: list[int]
    text_id: int = -1

    def update(self, position: Sequence[float] | None) -> None:
        if position is None:
            return
        center = list(position)
        half_size = 0.012
        while len(self.line_ids) < 3:
            self.line_ids.append(-1)
        for index, direction in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
            start = [center[axis] - half_size * direction[axis] for axis in range(3)]
            end = [center[axis] + half_size * direction[axis] for axis in range(3)]
            self.line_ids[index] = p.addUserDebugLine(
                start, end, lineColorRGB=self.colour, lineWidth=3, lifeTime=0,
                replaceItemUniqueId=self.line_ids[index], physicsClientId=self.client_id,
            )
        self.text_id = p.addUserDebugText(
            f"Estimated {self.class_name}",
            [center[0], center[1], center[2] + 0.035],
            textColorRGB=self.colour, textSize=1.0, lifeTime=0,
            replaceItemUniqueId=self.text_id, physicsClientId=self.client_id,
        )


def _create_coloured_sphere(spec: TargetSpec, client_id: int) -> int:
    """Create a static experiment target; position is scheduling/evaluation only."""

    collision_id = p.createCollisionShape(
        p.GEOM_SPHERE, radius=TARGET_RADIUS_M, physicsClientId=client_id
    )
    visual_id = p.createVisualShape(
        p.GEOM_SPHERE, radius=TARGET_RADIUS_M, rgbaColor=spec.rgba, physicsClientId=client_id
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=(0.0, 0.0, TARGET_RADIUS_M),
        physicsClientId=client_id,
    )


def _set_target_pose(body_id: int, position: Sequence[float], client_id: int) -> None:
    """Apply a static target pose only; it is not a perception/controller input."""

    p.resetBasePositionAndOrientation(
        body_id, position, (0.0, 0.0, 0.0, 1.0), physicsClientId=client_id
    )
    p.resetBaseVelocity(
        body_id, linearVelocity=(0.0, 0.0, 0.0), angularVelocity=(0.0, 0.0, 0.0),
        physicsClientId=client_id,
    )


def _empty_localization() -> RGBDLocalizationResult:
    return RGBDLocalizationResult(
        valid=False, state="TARGET_NOT_DETECTED", pixel=None,
        depth=DepthStatistics(0, None, None, None, None),
        surface_camera_m=None, surface_world_m=None,
        center_camera_m=None, center_world_m=None,
    )


def _overlay(
    frame: LiveCameraFrame,
    detections: Sequence[DetectedTarget],
    localized: Sequence[LocalizedTarget],
    scene_label: str,
    scene_index: int,
) -> bytes:
    """Add concise display-only status after frozen RGB detection/localization."""

    annotated = annotate_multi_target_detections(
        frame.rgba_buffer, frame.image_width, frame.image_height, detections
    )
    rgba = np.frombuffer(annotated, dtype=np.uint8).reshape((frame.image_height, frame.image_width, 4))
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cv2.putText(
        bgr, f"STAGE 15 MULTI-TARGET RGB-D | Scene {scene_index}: {scene_label}",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
    )
    detected_count = sum(item.detection.detected for item in localized)
    cv2.putText(
        bgr, f"Detected: {detected_count}/3 | Robot: HOLD", (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
    )
    colour_by_name = {spec.class_name: tuple(int(255 * value) for value in spec.marker_rgb[::-1]) for spec in TARGET_SPECS}
    for index, item in enumerate(localized):
        if item.localization.valid:
            assert item.localization.depth.median_m is not None
            text = (
                f"{item.class_name[0]} {item.detection.centroid} "
                f"Z={item.localization.depth.median_m:.3f}m"
            )
        else:
            text = f"{item.class_name[0]} not detected"
        cv2.putText(
            bgr, text, (8, 68 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, colour_by_name[item.class_name], 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()


def _csv_fields() -> tuple[str, ...]:
    return (
        "time_s", "scene", "scene_index", "target_id", "class_name", "detected", "detection_valid",
        "u", "v", "area_px", "confidence", "depth_m", "depth_sample_count",
        "camera_x", "camera_y", "camera_z", "camera_world_x", "camera_world_y", "camera_world_z",
        "world_surface_x", "world_surface_y", "world_surface_z",
        "world_est_x", "world_est_y", "world_est_z", "localization_valid", "localization_state",
        "gt_x", "gt_y", "gt_z", "error_3d", "ground_truth_used_for_perception",
        "ground_truth_used_for_localization", "ground_truth_used_for_controller",
        "ground_truth_used_for_evaluation",
    )


def _as_value(value: float | int | str | None) -> str | float | int:
    if value is None:
        return ""
    return f"{value:.9f}" if isinstance(value, float) else value


def _build_row(
    time_s: float,
    scene_label: str,
    scene_index: int,
    item: LocalizedTarget,
    camera_world_position: Sequence[float],
) -> dict[str, object]:
    """Create one target/frame row after evaluation-only GT comparison."""

    detection = item.detection
    localization = item.localization
    if detection.centroid is None:
        u = v = None
    else:
        u, v = detection.centroid
    row: dict[str, object] = {
        "time_s": _as_value(time_s), "scene": scene_label, "scene_index": scene_index,
        "target_id": item.target_id, "class_name": item.class_name,
        "detected": int(detection.detected), "detection_valid": int(detection.valid),
        "u": _as_value(u), "v": _as_value(v), "area_px": _as_value(detection.area),
        "confidence": _as_value(detection.confidence), "depth_m": _as_value(localization.depth.median_m),
        "depth_sample_count": localization.depth.sample_count,
        "camera_x": "", "camera_y": "", "camera_z": "",
        "camera_world_x": _as_value(float(camera_world_position[0])),
        "camera_world_y": _as_value(float(camera_world_position[1])),
        "camera_world_z": _as_value(float(camera_world_position[2])),
        "world_surface_x": "", "world_surface_y": "", "world_surface_z": "",
        "world_est_x": "", "world_est_y": "", "world_est_z": "",
        "localization_valid": int(localization.valid), "localization_state": localization.state,
        "gt_x": _as_value(item.ground_truth_world_m[0]), "gt_y": _as_value(item.ground_truth_world_m[1]),
        "gt_z": _as_value(item.ground_truth_world_m[2]), "error_3d": _as_value(item.error_3d_m),
        "ground_truth_used_for_perception": False, "ground_truth_used_for_localization": False,
        "ground_truth_used_for_controller": False, "ground_truth_used_for_evaluation": True,
    }
    if localization.valid:
        assert localization.center_camera_m is not None
        assert localization.surface_world_m is not None
        assert localization.center_world_m is not None
        row.update(
            {
                "camera_x": _as_value(localization.center_camera_m[0]),
                "camera_y": _as_value(localization.center_camera_m[1]),
                "camera_z": _as_value(localization.center_camera_m[2]),
                "world_surface_x": _as_value(localization.surface_world_m[0]),
                "world_surface_y": _as_value(localization.surface_world_m[1]),
                "world_surface_z": _as_value(localization.surface_world_m[2]),
                "world_est_x": _as_value(localization.center_world_m[0]),
                "world_est_y": _as_value(localization.center_world_m[1]),
                "world_est_z": _as_value(localization.center_world_m[2]),
            }
        )
    return row


def _summary_fields() -> tuple[str, ...]:
    return (
        "target", "target_id", "frames", "detected_frames", "detection_rate_percent",
        "mean_3d_error_m", "rmse_3d_error_m", "max_3d_error_m", "all_targets_detection_rate_percent",
        "target_confusion_events", "result",
    )


def _target_summary(
    spec: TargetSpec,
    frames: int,
    detected_frames: int,
    errors: Sequence[float],
    confusion_events: int,
) -> dict[str, object]:
    rate = 100.0 * detected_frames / frames if frames else 0.0
    return {
        "target": spec.class_name, "target_id": spec.target_id, "frames": frames,
        "detected_frames": detected_frames, "detection_rate_percent": rate,
        "mean_3d_error_m": float(np.mean(errors)) if errors else None,
        "rmse_3d_error_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "max_3d_error_m": float(np.max(errors)) if errors else None,
        "all_targets_detection_rate_percent": "", "target_confusion_events": confusion_events,
        "result": "PASS" if frames and detected_frames == frames and errors else "FAIL",
    }


def _all_three_summary(total_frames: int, all_detected_frames: int) -> dict[str, object]:
    rate = 100.0 * all_detected_frames / total_frames if total_frames else 0.0
    return {
        "target": "ALL_THREE", "target_id": "", "frames": total_frames,
        "detected_frames": all_detected_frames, "detection_rate_percent": rate,
        "mean_3d_error_m": "", "rmse_3d_error_m": "", "max_3d_error_m": "",
        "all_targets_detection_rate_percent": rate, "target_confusion_events": 0,
        "result": "PASS" if total_frames and all_detected_frames == total_frames else "FAIL",
    }


def run_stage15_multitarget_perception() -> None:
    """Run five static RGB-only-detected, independent RGB-D target scenes."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 15.")
    display: EyeInHandRgbDisplay | None = None
    try:
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB-D - Stage 15 Multi-Target")
        detector = MultiTargetDetector()
        intrinsics = camera_intrinsics_from_fov(640, 480, CAMERA_FOV_Y_DEGREES)
        localizer = RGBDTargetLocalizer(
            intrinsics, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE, TARGET_RADIUS_M
        )
        print("Stage 15: Multi-Target RGB-D Perception")
        print("Detector: RGB HSV contours only. Segmentation/object IDs are not used for perception.")
        print("Robot: HOLD. No target selection, switching, visual-servo command, or IK command is issued.")
        print("Target IDs: RED=target_1, GREEN=target_2, BLUE=target_3.")
        print(f"Each sphere radius: {TARGET_RADIUS_M:.3f} m. Each scene requires {SAMPLES_PER_SCENE} valid RGB-D frames.")

        total_frames = all_three_detected_frames = 0
        target_frames = {spec.target_id: 0 for spec in TARGET_SPECS}
        target_detected = {spec.target_id: 0 for spec in TARGET_SPECS}
        target_errors: dict[str, list[float]] = {spec.target_id: [] for spec in TARGET_SPECS}
        target_confusion = {spec.target_id: 0 for spec in TARGET_SPECS}
        scene_results: list[bool] = []
        simulation_time_s = 0.0
        render_to_camera: np.ndarray | None = None
        max_locked_deviation = 0.0

        with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for scene_index, (scene_label, configuration) in enumerate(SCENE_CONFIGURATIONS, start=1):
                # Each scene begins from the same frozen Stage 14 HOLD pose.  This
                # prevents renderer/physics state from a previous scene becoming
                # an unrecorded experimental variable; no visual-servo command is
                # issued by this setup helper.
                context = stage14._create_static_localization_context(client_id)
                # Stage 14 creates the red sphere.  Stage 15 adds independent
                # static green and blue spheres for this scene only.
                target_body_ids: dict[str, int] = {"target_1": context.target_body_id}
                for spec in TARGET_SPECS[1:]:
                    target_body_ids[spec.target_id] = _create_coloured_sphere(spec, client_id)
                markers = {
                    spec.target_id: TargetDebugMarker(client_id, spec.class_name, spec.marker_rgb, [])
                    for spec in TARGET_SPECS
                }
                for spec in TARGET_SPECS:
                    _set_target_pose(target_body_ids[spec.target_id], configuration[spec.target_id], client_id)
                stage14._run_hold_steps(context, round(SCENE_SETTLE_SECONDS / sim.TIME_STEP), client_id)
                simulation_time_s += SCENE_SETTLE_SECONDS
                valid_all_frames = 0
                capture_steps = 0
                max_steps = round(MAX_CAPTURE_SECONDS / sim.TIME_STEP)
                context.debug_text_id = sim.update_motion_debug_text(
                    f"STAGE 15 - MULTI-TARGET RGB-D\nScene {scene_index}/5: {scene_label}\n"
                    "Robot: HOLD | RGB-D: ACTIVE | GT: EVALUATION ONLY",
                    context.debug_text_id, client_id,
                )
                while valid_all_frames < SAMPLES_PER_SCENE and capture_steps < max_steps:
                    stage10._step_physics(context, client_id)
                    simulation_time_s += sim.TIME_STEP
                    if capture_steps % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                        sim.update_camera_reference_axes(
                            context.robot_id, client_id, context.camera_axis_debug_item_ids
                        )
                        camera_position, camera_orientation = sim.get_camera_optical_center_pose(
                            context.robot_id, client_id
                        )
                        rgbd_frame = render_live_eye_in_hand_rgbd_frame(
                            camera_position, camera_orientation, client_id
                        )
                        frame = rgbd_frame.live_rgb_frame
                        detections = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
                        detection_by_id = {detection.target_id: detection for detection in detections}
                        camera_to_world, _, _ = stage14._validate_camera_pose_and_render_frame(
                            context.robot_id, frame, client_id
                        )
                        current_transform = stage14._render_to_camera_axis_transform(camera_to_world, frame)
                        if render_to_camera is None:
                            render_to_camera = current_transform
                            print("C_render -> C axis transform:\n" + np.array2string(render_to_camera, precision=5, suppress_small=True))
                        elif not np.allclose(render_to_camera, current_transform, atol=1e-6):
                            raise RuntimeError("Stage 15 render-to-C transform unexpectedly changed.")

                        # All localizations are complete before any target world pose is read.
                        localizations: dict[str, RGBDLocalizationResult] = {}
                        for spec in TARGET_SPECS:
                            detection = detection_by_id[spec.target_id]
                            if detection.detected:
                                assert detection.centroid is not None
                                localizations[spec.target_id] = localizer.localize_from_mask(
                                    detection.centroid[0], detection.centroid[1], detection.mask,
                                    rgbd_frame.depth_buffer, camera_to_world, render_to_camera,
                                )
                            else:
                                localizations[spec.target_id] = _empty_localization()

                        observations: list[LocalizedTarget] = []
                        for spec in TARGET_SPECS:
                            # Evaluation-only read: no resulting value flows back to detection/localization/control.
                            gt_position, _ = p.getBasePositionAndOrientation(
                                target_body_ids[spec.target_id], physicsClientId=client_id
                            )
                            localization = localizations[spec.target_id]
                            error = None
                            if localization.valid:
                                assert localization.center_world_m is not None
                                error = float(
                                    np.linalg.norm(np.asarray(localization.center_world_m) - np.asarray(gt_position))
                                )
                            observations.append(
                                LocalizedTarget(
                                    spec.target_id, spec.class_name, detection_by_id[spec.target_id], localization,
                                    tuple(float(value) for value in gt_position), error,
                                )
                            )
                        all_valid = all(
                            observation.detection.valid and observation.localization.valid for observation in observations
                        )
                        total_frames += 1
                        all_three_detected_frames += int(all_valid)
                        for observation in observations:
                            target_frames[observation.target_id] += 1
                            target_detected[observation.target_id] += int(observation.detection.valid)
                            if observation.error_3d_m is not None:
                                target_errors[observation.target_id].append(observation.error_3d_m)
                            # Different HSV masks must yield exactly one stable ID per class. A mismatch is a confusion.
                            target_confusion[observation.target_id] += int(
                                observation.detection.target_id != observation.target_id
                                or observation.detection.class_name != observation.class_name
                            )
                            writer.writerow(
                                _build_row(
                                    simulation_time_s, scene_label, scene_index, observation,
                                    camera_position,
                                )
                            )
                            if observation.localization.valid:
                                markers[observation.target_id].update(observation.localization.center_world_m)
                        display.show(frame, _overlay(frame, detections, observations, scene_label, scene_index))
                        valid_all_frames += int(all_valid)
                    capture_steps += 1
                    time.sleep(sim.TIME_STEP)
                scene_ok = valid_all_frames == SAMPLES_PER_SCENE
                scene_results.append(scene_ok)
                max_locked_deviation = max(max_locked_deviation, context.max_locked_deviation)
                print(
                    f"Scene {scene_index} ({scene_label}): {valid_all_frames}/{SAMPLES_PER_SCENE} "
                    f"valid all-target RGB-D frames -> {'PASS' if scene_ok else 'FAIL'}"
                )

        summaries = [
            _target_summary(
                spec, target_frames[spec.target_id], target_detected[spec.target_id],
                target_errors[spec.target_id], target_confusion[spec.target_id],
            )
            for spec in TARGET_SPECS
        ]
        summaries.append(_all_three_summary(total_frames, all_three_detected_frames))
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=_summary_fields())
            writer.writeheader()
            for summary in summaries:
                writer.writerow({field: summary.get(field, "") for field in _summary_fields()})

        from plot_stage15 import generate_stage15_figures

        figure_paths = generate_stage15_figures()
        locked_pass = max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
        all_target_pass = all(summary["result"] == "PASS" for summary in summaries)
        stage15_pass = all(scene_results) and all_target_pass and locked_pass
        print("\n===== STAGE 15 MULTI-TARGET SUMMARY =====")
        for summary in summaries:
            if summary["target"] == "ALL_THREE":
                print(f"ALL THREE detection rate: {summary['all_targets_detection_rate_percent']:.1f}%")
            else:
                print(
                    f"{summary['target']}: detection={summary['detection_rate_percent']:.1f}%, "
                    f"mean/RMSE/max 3D error="
                    f"{float(summary['mean_3d_error_m'])*1000:.2f}/"
                    f"{float(summary['rmse_3d_error_m'])*1000:.2f}/"
                    f"{float(summary['max_3d_error_m'])*1000:.2f} mm, "
                    f"confusion={summary['target_confusion_events']}"
                )
        print(f"Locked joint max deviation: {max_locked_deviation:.6f} rad")
        print("GT audit: perception=False, localization=False, controller=False, evaluation=True")
        print("CSV:", LOG_PATH)
        print("Summary:", SUMMARY_PATH)
        print("Figures:", *figure_paths)
        print("Stage 15:", "PASS" if stage15_pass else "FAIL")
    finally:
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)
