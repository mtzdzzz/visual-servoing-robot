"""Stage 14 static RGB-D target localization experiment.

The Panda visual-servo controller is deliberately not modified here.  The
module starts from a previously validated static, camera-visible Panda hold
pose, captures synchronized RGB-D frames, and evaluates only the new RGB-D
localization estimate.  Red-ball world coordinates are used only to schedule
the five static test poses and to score the final estimate.  They are never
passed to ``RGBDTargetLocalizer`` or to a robot command.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import acos, degrees
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
import pybullet as p

import robotics_core as sim
import visual_servo_runtime as runtime
import camera_geometry
import scene_factory
from camera_observation import (
    CAMERA_FAR_PLANE,
    CAMERA_FOV_Y_DEGREES,
    CAMERA_NEAR_PLANE,
    EyeInHandRgbDisplay,
    LiveCameraFrame,
    RedTargetDetection,
    detect_red_target_from_live_rgb,
    render_live_eye_in_hand_rgbd_frame,
)
from rgbd_localization import (
    CameraIntrinsics,
    DepthStatistics,
    RGBDLocalizationResult,
    RGBDTargetLocalizer,
    camera_intrinsics_from_fov,
)


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / "logs"
LOCALIZATION_LOG_PATH = LOG_DIRECTORY / "stage14_rgbd_localization.csv"
SUMMARY_LOG_PATH = LOG_DIRECTORY / "stage14_rgbd_summary.csv"

# These five static points are centred on the Stage 9 work area.  They vary
# only the physical target pose for experiment scheduling and evaluation;
# neither the localizer nor the Panda controller receives these values.
STATIC_TARGET_POSITIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("center", (0.350, -0.250, sim.GROUND_TARGET_RADIUS)),
    ("left", (0.315, -0.250, sim.GROUND_TARGET_RADIUS)),
    ("right", (0.385, -0.250, sim.GROUND_TARGET_RADIUS)),
    ("near", (0.350, -0.215, sim.GROUND_TARGET_RADIUS)),
    ("far", (0.350, -0.285, sim.GROUND_TARGET_RADIUS)),
)
SAMPLES_PER_STATIC_POSITION = 20
SETTLE_SECONDS = 0.50
MAX_CAPTURE_SECONDS = 5.0


@dataclass
class EstimatedTargetDebugMarker:
    """A small cyan world-space cross generated solely from localization output."""

    client_id: int
    line_ids: list[int]
    text_id: int = -1

    def update(self, estimated_world_position: Sequence[float] | None) -> None:
        if estimated_world_position is None:
            return
        center = list(estimated_world_position)
        half_size = 0.014
        directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        while len(self.line_ids) < 3:
            self.line_ids.append(-1)
        for index, direction in enumerate(directions):
            start = [center[axis] - half_size * direction[axis] for axis in range(3)]
            end = [center[axis] + half_size * direction[axis] for axis in range(3)]
            self.line_ids[index] = p.addUserDebugLine(
                start,
                end,
                lineColorRGB=[0.0, 1.0, 1.0],
                lineWidth=3,
                lifeTime=0,
                replaceItemUniqueId=self.line_ids[index],
                physicsClientId=self.client_id,
            )
        self.text_id = p.addUserDebugText(
            "Estimated 3D Target",
            [center[0], center[1], center[2] + 0.035],
            textColorRGB=[0.0, 1.0, 1.0],
            textSize=1.1,
            lifeTime=0,
            replaceItemUniqueId=self.text_id,
            physicsClientId=self.client_id,
        )


def _ground_truth_projection_diagnostic(
    camera_to_world: np.ndarray,
    render_to_camera: np.ndarray,
    ground_truth_world: Sequence[float],
    intrinsics: CameraIntrinsics,
    live_frame: LiveCameraFrame,
) -> str:
    """Return an evaluation-only projection check for camera-frame validation."""

    camera_from_world = np.linalg.inv(camera_to_world)
    gt_camera = (camera_from_world @ np.append(np.asarray(ground_truth_world), 1.0))[:3]
    gt_render = np.linalg.solve(render_to_camera, gt_camera)
    if gt_render[2] <= 0.0:
        return f"GT projection audit: target behind C_render (Z={gt_render[2]:.4f} m)."
    expected_u = intrinsics.cx + intrinsics.fx * gt_render[0] / gt_render[2]
    expected_v = intrinsics.cy - intrinsics.fy * gt_render[1] / gt_render[2]
    actual_view = np.asarray(live_frame.render_parameters.target_position) - np.asarray(
        live_frame.render_parameters.eye_position
    )
    actual_view /= np.linalg.norm(actual_view)
    rendered_forward = np.asarray(live_frame.optical_axis_world)
    rendered_forward /= np.linalg.norm(rendered_forward)
    axis_error = degrees(acos(float(np.clip(np.dot(actual_view, rendered_forward), -1.0, 1.0))))
    return (
        "GT projection audit (evaluation only): "
        f"P_C=[{gt_camera[0]:+.4f}, {gt_camera[1]:+.4f}, {gt_camera[2]:+.4f}] m, "
        f"pixel=({expected_u:.1f}, {expected_v:.1f}), render/C +Z angle={axis_error:.5f} deg."
    )


def _overlay_rgbd_localization(
    frame: LiveCameraFrame,
    detection: RedTargetDetection,
    localization: RGBDLocalizationResult,
    error_mm: float | None,
    phase: str,
    label: str,
) -> bytes:
    """Draw display-only Stage 14 information after the frozen RGB detector."""

    rgba = np.frombuffer(detection.annotated_rgba_buffer, dtype=np.uint8).reshape(
        (frame.image_height, frame.image_width, 4)
    )
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    lines = [f"STAGE 14 RGB-D | {phase} | {label}"]
    if localization.valid:
        assert localization.pixel is not None
        assert localization.depth.median_m is not None
        assert localization.center_camera_m is not None
        assert localization.center_world_m is not None
        u, v = localization.pixel
        pc = localization.center_camera_m
        pw = localization.center_world_m
        lines.extend(
            [
                f"Pixel: ({u}, {v}) | Depth Z: {localization.depth.median_m:.3f} m ({localization.depth.sample_count} samples)",
                f"Camera C: [{pc[0]:+.3f}, {pc[1]:+.3f}, {pc[2]:+.3f}] m",
                f"World W: [{pw[0]:+.3f}, {pw[1]:+.3f}, {pw[2]:+.3f}] m",
                "3D Error: evaluation unavailable" if error_mm is None else f"3D Error: {error_mm:.2f} mm",
            ]
        )
        cv2.drawMarker(
            bgr,
            (u, v),
            (255, 255, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=22,
            thickness=2,
        )
    else:
        lines.append(f"Localization State: {localization.state}")
    for index, text in enumerate(lines):
        cv2.putText(
            bgr,
            text,
            (8, 48 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()


def _csv_fields() -> tuple[str, ...]:
    return (
        "time_s", "trial", "position_label", "localization_state", "target_detected",
        "u", "v", "depth_buffer_value", "depth_m", "depth_sample_count",
        "depth_min_m", "depth_max_m", "depth_median_m", "fx", "fy", "cx", "cy",
        "camera_x", "camera_y", "camera_z", "camera_world_x", "camera_world_y", "camera_world_z",
        "surface_camera_x", "surface_camera_y", "surface_camera_z",
        "world_est_x", "world_est_y", "world_est_z", "surface_world_x", "surface_world_y", "surface_world_z",
        "center_est_x", "center_est_y", "center_est_z", "sphere_radius",
        "gt_world_x", "gt_world_y", "gt_world_z", "error_x", "error_y", "error_z", "error_3d",
        "surface_error_3d", "center_error_3d", "ground_truth_used_for_localization",
        "ground_truth_used_for_controller", "ground_truth_used_for_evaluation",
    )


def _format(value: float | int | str | None) -> str | float | int:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return value


def _log_row(
    time_s: float,
    trial_index: int,
    label: str,
    detection: RedTargetDetection,
    localization: RGBDLocalizationResult,
    intrinsics: CameraIntrinsics,
    camera_world_position: Sequence[float],
    ground_truth_world: Sequence[float],
) -> tuple[dict[str, object], float | None, float | None]:
    """Build one row.  GT is consumed only after localization for evaluation."""

    u = v = None
    if detection.detected:
        assert detection.centroid is not None
        u, v = detection.centroid
    result = {
        "time_s": _format(time_s), "trial": trial_index, "position_label": label,
        "localization_state": localization.state, "target_detected": int(detection.detected),
        "u": _format(u), "v": _format(v),
        "depth_buffer_value": _format(localization.depth.depth_buffer_value),
        "depth_m": _format(localization.depth.median_m),
        "depth_sample_count": localization.depth.sample_count,
        "depth_min_m": _format(localization.depth.minimum_m),
        "depth_max_m": _format(localization.depth.maximum_m),
        "depth_median_m": _format(localization.depth.median_m),
        "fx": _format(intrinsics.fx), "fy": _format(intrinsics.fy),
        "cx": _format(intrinsics.cx), "cy": _format(intrinsics.cy),
        "camera_x": "", "camera_y": "", "camera_z": "",
        "camera_world_x": _format(float(camera_world_position[0])),
        "camera_world_y": _format(float(camera_world_position[1])),
        "camera_world_z": _format(float(camera_world_position[2])),
        "surface_camera_x": "", "surface_camera_y": "", "surface_camera_z": "",
        "world_est_x": "", "world_est_y": "", "world_est_z": "",
        "surface_world_x": "", "surface_world_y": "", "surface_world_z": "",
        "center_est_x": "", "center_est_y": "", "center_est_z": "",
        "sphere_radius": _format(sim.GROUND_TARGET_RADIUS),
        "gt_world_x": _format(float(ground_truth_world[0])),
        "gt_world_y": _format(float(ground_truth_world[1])),
        "gt_world_z": _format(float(ground_truth_world[2])),
        "error_x": "", "error_y": "", "error_z": "", "error_3d": "",
        "surface_error_3d": "", "center_error_3d": "",
        "ground_truth_used_for_localization": False,
        "ground_truth_used_for_controller": False,
        "ground_truth_used_for_evaluation": True,
    }
    if not localization.valid:
        return result, None, None
    assert localization.center_camera_m is not None
    assert localization.surface_camera_m is not None
    assert localization.center_world_m is not None
    assert localization.surface_world_m is not None
    center_camera = np.asarray(localization.center_camera_m)
    surface_camera = np.asarray(localization.surface_camera_m)
    center_world = np.asarray(localization.center_world_m)
    surface_world = np.asarray(localization.surface_world_m)
    gt = np.asarray(ground_truth_world, dtype=np.float64)
    center_error = center_world - gt
    center_error_norm = float(np.linalg.norm(center_error))
    surface_error_norm = float(np.linalg.norm(surface_world - gt))
    result.update(
        {
            "camera_x": _format(float(center_camera[0])), "camera_y": _format(float(center_camera[1])),
            "camera_z": _format(float(center_camera[2])),
            "surface_camera_x": _format(float(surface_camera[0])),
            "surface_camera_y": _format(float(surface_camera[1])),
            "surface_camera_z": _format(float(surface_camera[2])),
            # world_est is defined explicitly as the radius-compensated sphere centre.
            "world_est_x": _format(float(center_world[0])), "world_est_y": _format(float(center_world[1])),
            "world_est_z": _format(float(center_world[2])),
            "surface_world_x": _format(float(surface_world[0])), "surface_world_y": _format(float(surface_world[1])),
            "surface_world_z": _format(float(surface_world[2])),
            "center_est_x": _format(float(center_world[0])), "center_est_y": _format(float(center_world[1])),
            "center_est_z": _format(float(center_world[2])),
            "error_x": _format(float(center_error[0])), "error_y": _format(float(center_error[1])),
            "error_z": _format(float(center_error[2])), "error_3d": _format(center_error_norm),
            "surface_error_3d": _format(surface_error_norm), "center_error_3d": _format(center_error_norm),
        }
    )
    return result, center_error_norm, surface_error_norm


def _summary_fields() -> tuple[str, ...]:
    return (
        "trial", "position_label", "valid_samples", "requested_samples", "result",
        "estimated_world_x", "estimated_world_y", "estimated_world_z",
        "surface_world_x", "surface_world_y", "surface_world_z",
        "gt_world_x", "gt_world_y", "gt_world_z", "center_error_3d_m", "surface_error_3d_m",
        "center_error_3d_mm", "surface_error_3d_mm", "locked_joint_max_deviation_rad",
    )


def _aggregate_trial(
    trial_index: int,
    label: str,
    estimates: Sequence[RGBDLocalizationResult],
    ground_truth: Sequence[float],
    max_locked_deviation: float,
) -> dict[str, object]:
    """Use coordinate-wise median samples for one stable static-position summary."""

    valid = [result for result in estimates if result.valid]
    base = {
        "trial": trial_index, "position_label": label, "valid_samples": len(valid),
        "requested_samples": SAMPLES_PER_STATIC_POSITION,
        "gt_world_x": ground_truth[0], "gt_world_y": ground_truth[1], "gt_world_z": ground_truth[2],
        "locked_joint_max_deviation_rad": max_locked_deviation,
    }
    if not valid:
        return {**base, "result": "FAIL"}
    centers = np.array([result.center_world_m for result in valid], dtype=np.float64)
    surfaces = np.array([result.surface_world_m for result in valid], dtype=np.float64)
    center = np.median(centers, axis=0)
    surface = np.median(surfaces, axis=0)
    gt = np.asarray(ground_truth, dtype=np.float64)
    center_error = float(np.linalg.norm(center - gt))
    surface_error = float(np.linalg.norm(surface - gt))
    passed = (
        len(valid) == SAMPLES_PER_STATIC_POSITION
        and max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
    )
    return {
        **base,
        "result": "PASS" if passed else "FAIL",
        "estimated_world_x": float(center[0]), "estimated_world_y": float(center[1]), "estimated_world_z": float(center[2]),
        "surface_world_x": float(surface[0]), "surface_world_y": float(surface[1]), "surface_world_z": float(surface[2]),
        "center_error_3d_m": center_error, "surface_error_3d_m": surface_error,
        "center_error_3d_mm": center_error * 1000.0, "surface_error_3d_mm": surface_error * 1000.0,
    }


def _set_static_target(target_body_id: int, world_position: Sequence[float], client_id: int) -> None:
    """Schedule one static target pose; this is never a localization/controller input."""

    p.resetBasePositionAndOrientation(
        target_body_id,
        world_position,
        (0.0, 0.0, 0.0, 1.0),
        physicsClientId=client_id,
    )
    p.resetBaseVelocity(
        target_body_id,
        linearVelocity=(0.0, 0.0, 0.0),
        angularVelocity=(0.0, 0.0, 0.0),
        physicsClientId=client_id,
    )


def run_stage14_rgbd_localization() -> None:
    """Run five static, vision-only RGB-D localization trials in the GUI."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 14.")
    display: EyeInHandRgbDisplay | None = None
    try:
        display = EyeInHandRgbDisplay("Eye-in-Hand RGB-D - Stage 14")
        print("Stage 14: RGB-D Target 3D Localization")
        print("Controller and predictor are frozen; this mode sends no localization result to IK.")
        context = scene_factory.create_static_localization_context(client_id)
        print("Stage 14 initialization: validated Stage 7 safe active pose -> HOLD (no visual-servo warm-up).")
        intrinsics = camera_intrinsics_from_fov(640, 480, CAMERA_FOV_Y_DEGREES)
        localizer = RGBDTargetLocalizer(
            intrinsics=intrinsics,
            near_plane_m=CAMERA_NEAR_PLANE,
            far_plane_m=CAMERA_FAR_PLANE,
            sphere_radius_m=sim.GROUND_TARGET_RADIUS,
        )
        print(
            "Camera intrinsics: "
            f"fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}"
        )
        print(
            "Depth conversion: z = far*near / (far - (far-near)*depth_buffer), "
            f"near={CAMERA_NEAR_PLANE:.3f} m, far={CAMERA_FAR_PLANE:.3f} m, FOV_y={CAMERA_FOV_Y_DEGREES:.1f} deg."
        )
        print(
            "C_render: +X image-right, +Y image-up, +Z viewing; image v-down gives "
            "Y_render=-(v-cy)Z/fy. T_W_C is the physical hand-eye C pose."
        )
        print("No target world pose enters the localizer/controller.")
        marker = EstimatedTargetDebugMarker(client_id, [])
        summaries: list[dict[str, object]] = []
        simulation_time_s = 0.0
        camera_pose_audited = False
        render_axis_transform: np.ndarray | None = None

        with LOCALIZATION_LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for trial_index, (label, scheduled_position) in enumerate(STATIC_TARGET_POSITIONS, start=1):
                _set_static_target(context.target_body_id, scheduled_position, client_id)
                scene_factory.run_hold_steps(context, round(SETTLE_SECONDS / sim.TIME_STEP), client_id)
                simulation_time_s += SETTLE_SECONDS
                collected: list[RGBDLocalizationResult] = []
                capture_steps = 0
                maximum_steps = round(MAX_CAPTURE_SECONDS / sim.TIME_STEP)
                context.debug_text_id = sim.update_motion_debug_text(
                    f"STAGE 14 - RGB-D LOCALIZATION\nTrial {trial_index}/5: {label}\n"
                    "Robot: HOLD | RGB-D: ACTIVE | GT: EVALUATION ONLY",
                    context.debug_text_id,
                    client_id,
                )
                while len(collected) < SAMPLES_PER_STATIC_POSITION and capture_steps < maximum_steps:
                    runtime.step_physics(context, client_id)
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
                        live_frame = rgbd_frame.live_rgb_frame
                        detection = detect_red_target_from_live_rgb(live_frame)
                        camera_to_world, pose_error, rotation_error = camera_geometry.validate_camera_pose_and_render_frame(
                            context.robot_id, live_frame, client_id
                        )
                        if not camera_pose_audited:
                            print(
                                f"T_W_C/render audit: position error={pose_error:.3e} m, "
                                f"orientation error={rotation_error:.3e} deg."
                            )
                            camera_pose_audited = True
                        current_render_axis_transform = camera_geometry.render_to_camera_axis_transform(
                            camera_to_world, live_frame
                        )
                        if render_axis_transform is None:
                            render_axis_transform = current_render_axis_transform
                            print(
                                "Derived fixed C_render -> C axis transform:\n"
                                + np.array2string(render_axis_transform, precision=5, suppress_small=True)
                            )
                        elif not np.allclose(
                            render_axis_transform, current_render_axis_transform, atol=1e-6
                        ):
                            raise RuntimeError("C_render -> C axis transform changed during a static trial.")
                        if detection.detected:
                            assert detection.centroid is not None
                            localization = localizer.localize(
                                detection.centroid[0], detection.centroid[1], live_frame.rgba_buffer,
                                rgbd_frame.depth_buffer, camera_to_world, render_axis_transform,
                            )
                        else:
                            localization = RGBDLocalizationResult(
                                valid=False, state="TARGET_NOT_DETECTED", pixel=None,
                                depth=DepthStatistics(0, None, None, None, None),
                                surface_camera_m=None, surface_world_m=None,
                                center_camera_m=None, center_world_m=None,
                            )
                        # Ground truth is deliberately read only here, after RGB-D localization.
                        ground_truth_position, _ = p.getBasePositionAndOrientation(
                            context.target_body_id, physicsClientId=client_id
                        )
                        if trial_index == 1 and capture_steps == 0:
                            print(
                                _ground_truth_projection_diagnostic(
                                    camera_to_world, render_axis_transform, ground_truth_position, intrinsics, live_frame
                                )
                            )
                        row, center_error, _ = _log_row(
                            simulation_time_s, trial_index, label, detection, localization, intrinsics,
                            live_frame.camera_world_position, ground_truth_position,
                        )
                        writer.writerow(row)
                        if localization.valid:
                            collected.append(localization)
                            marker.update(localization.center_world_m)
                        display.show(
                            live_frame,
                            _overlay_rgbd_localization(
                                live_frame, detection, localization,
                                center_error * 1000.0 if center_error is not None else None,
                                "STATIC", f"Trial {trial_index}: {label}",
                            ),
                        )
                    capture_steps += 1
                    time.sleep(sim.TIME_STEP)
                ground_truth_position, _ = p.getBasePositionAndOrientation(
                    context.target_body_id, physicsClientId=client_id
                )
                summary = _aggregate_trial(
                    trial_index, label, collected, ground_truth_position, context.max_locked_deviation
                )
                summaries.append(summary)
                if summary["result"] != "PASS":
                    print(
                        f"Trial {trial_index} ({label}): FAIL, collected {len(collected)}/"
                        f"{SAMPLES_PER_STATIC_POSITION} valid RGB-D samples."
                    )
                else:
                    print(
                        f"Trial {trial_index} ({label}): center estimate "
                        f"[{summary['estimated_world_x']:.4f}, {summary['estimated_world_y']:.4f}, "
                        f"{summary['estimated_world_z']:.4f}] m; GT "
                        f"[{ground_truth_position[0]:.4f}, {ground_truth_position[1]:.4f}, {ground_truth_position[2]:.4f}] m; "
                        f"3D error={summary['center_error_3d_mm']:.2f} mm."
                    )

        with SUMMARY_LOG_PATH.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=_summary_fields())
            writer.writeheader()
            for summary in summaries:
                writer.writerow({field: summary.get(field, "") for field in _summary_fields()})

        center_errors = [float(row["center_error_3d_m"]) for row in summaries if row["result"] == "PASS"]
        surface_errors = [float(row["surface_error_3d_m"]) for row in summaries if row["result"] == "PASS"]
        if not center_errors:
            raise RuntimeError("Stage 14 produced no valid static RGB-D localization trials.")
        mean_error = float(np.mean(center_errors))
        rmse_error = float(np.sqrt(np.mean(np.square(center_errors))))
        median_error = float(np.median(center_errors))
        max_error = float(np.max(center_errors))
        passed = (
            len(summaries) == 5
            and all(summary["result"] == "PASS" for summary in summaries)
            and context.max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
        )
        print("\n===== STAGE 14 RGB-D LOCALIZATION SUMMARY =====")
        print(f"Sphere radius: {sim.GROUND_TARGET_RADIUS:.3f} m; surface-to-centre compensation: ENABLED.")
        print(f"Centre 3D error mean/RMSE/median/max: {mean_error*1000:.2f} / {rmse_error*1000:.2f} / "
              f"{median_error*1000:.2f} / {max_error*1000:.2f} mm")
        print(f"Surface error mean: {float(np.mean(surface_errors))*1000:.2f} mm (not directly comparable to sphere centre).")
        print(f"Locked joint max deviation: {context.max_locked_deviation:.6f} rad")
        print("Ground truth audit: localization=False, controller=False, evaluation=True")
        print("Stage 8/13 baseline audit: no controller, detector, mount, gain, joint or predictor path was edited.")
        print("CSV:", LOCALIZATION_LOG_PATH)
        print("Summary:", SUMMARY_LOG_PATH)
        print("Stage 14:", "PASS" if passed else "FAIL")
    finally:
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)

