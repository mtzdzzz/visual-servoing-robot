"""Reusable RGB-D obstacle occupancy capture and display integration."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import cv2
import numpy as np
import pybullet as p

import camera_geometry
import robotics_core as sim
import visual_servo_runtime as runtime
from camera_observation import (CAMERA_FAR_PLANE, CAMERA_FOV_Y_DEGREES, CAMERA_NEAR_PLANE,
                                EyeInHandRgbDisplay, LiveCameraFrame,
                                render_live_eye_in_hand_rgbd_frame)
from collision_checker import AxisAlignedBox
from multi_target_detector import MultiTargetDetector, annotate_multi_target_detections
from obstacle_detector import ObstacleDetection, YellowObstacleDetector
from obstacle_localization import ObstacleLocalizationResult, ObstacleLocalizer
from rgbd_localization import camera_intrinsics_from_fov


@dataclass
class OccupancyDebugBox:
    client_id: int
    line_ids: list[int]
    text_id: int = -1

    def update(self, minimum: Sequence[float] | None, maximum: Sequence[float] | None) -> None:
        if minimum is None or maximum is None:
            return
        lo, hi = list(minimum), list(maximum)
        corners = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]),
                   (lo[0], hi[1], lo[2]), (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
                   (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
        while len(self.line_ids) < len(edges):
            self.line_ids.append(-1)
        for index, (start, end) in enumerate(edges):
            self.line_ids[index] = p.addUserDebugLine(
                corners[start], corners[end], lineColorRGB=(1., .85, 0.), lineWidth=2,
                lifeTime=0, replaceItemUniqueId=self.line_ids[index], physicsClientId=self.client_id)
        label = ((lo[0] + hi[0]) / 2., (lo[1] + hi[1]) / 2., hi[2] + .02)
        self.text_id = p.addUserDebugText(
            "ESTIMATED OCCUPANCY", label, textColorRGB=(1., .85, 0.), textSize=1., lifeTime=0,
            replaceItemUniqueId=self.text_id, physicsClientId=self.client_id)


def annotated_obstacle_overlay(frame: LiveCameraFrame, target_detections: Sequence[object],
                               obstacle: ObstacleDetection, localization: ObstacleLocalizationResult,
                               scene: str, scene_index: int) -> bytes:
    annotated = annotate_multi_target_detections(frame.rgba_buffer, frame.image_width,
                                                  frame.image_height, target_detections)
    rgba = np.frombuffer(annotated, dtype=np.uint8).reshape((frame.image_height, frame.image_width, 4))
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cv2.rectangle(bgr, (4, 4), (635, 104), (25, 25, 25), thickness=-1)
    availability = " ".join(f"{d.class_name[0]}:{'Y' if d.valid else 'N'}" for d in target_detections)
    lines = [f"STAGE 17 RGB-D OBSTACLE | Scene {scene_index}/5: {scene}",
             f"Targets: {availability} | Robot: HOLD | RGB-D: ACTIVE"]
    if obstacle.valid and obstacle.bounding_box is not None and obstacle.centroid is not None:
        x, y, width, height = obstacle.bounding_box
        cv2.drawContours(bgr, [obstacle.contour], -1, (0, 255, 255), 2)
        cv2.rectangle(bgr, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.drawMarker(bgr, obstacle.centroid, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        lines.append(f"Obstacle: YELLOW {obstacle.centroid} | area={obstacle.area:.0f}px")
        lines.append(f"Depth median: {localization.depth_median_m:.3f} m | Point samples: {localization.valid_depth_samples}"
                     if localization.valid else f"Obstacle localization: {localization.state}")
    else:
        lines.append(f"Obstacle: NOT DETECTED ({obstacle.state})")
    for index, line in enumerate(lines):
        cv2.putText(bgr, line, (10, 25 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()


def capture_visual_occupancy(context: runtime.ReadyContext, display: EyeInHandRgbDisplay,
                             client_id: int, obstacle_dimensions: Sequence[float],
                             safety_margin: float, required_frames: int,
                             max_capture_seconds: float) -> tuple[AxisAlignedBox, OccupancyDebugBox, int]:
    """Capture a median vision-only occupancy; no GT pose/AABB/body ID is read."""
    intrinsics = camera_intrinsics_from_fov(640, 480, CAMERA_FOV_Y_DEGREES)
    localizer = ObstacleLocalizer(intrinsics, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE,
                                  obstacle_dimensions, safety_margin)
    obstacle_detector, target_detector = YellowObstacleDetector(), MultiTargetDetector()
    debug_box = OccupancyDebugBox(client_id, [])
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    render_to_camera = None
    steps = 0
    maximum_steps = round(max_capture_seconds / sim.TIME_STEP)
    while len(minima) < required_frames and steps < maximum_steps:
        runtime.step_physics(context, client_id)
        if steps % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
            sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
            position, orientation = sim.get_camera_optical_center_pose(context.robot_id, client_id)
            rgbd = render_live_eye_in_hand_rgbd_frame(position, orientation, client_id)
            frame = rgbd.live_rgb_frame
            targets = target_detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
            detection = obstacle_detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
            camera_to_world, _, _ = camera_geometry.validate_camera_pose_and_render_frame(
                context.robot_id, frame, client_id)
            axes = camera_geometry.render_to_camera_axis_transform(camera_to_world, frame)
            if render_to_camera is None:
                render_to_camera = axes
            elif not np.allclose(render_to_camera, axes, atol=1e-6):
                raise RuntimeError("Render-to-C axis transform changed unexpectedly.")
            localization = (localizer.localize(detection.mask, rgbd.depth_buffer, camera_to_world, axes)
                            if detection.valid else localizer._invalid("OBSTACLE_NOT_DETECTED"))
            if localization.valid:
                minima.append(np.asarray(localization.occupancy_aabb_min_m, dtype=np.float64))
                maxima.append(np.asarray(localization.occupancy_aabb_max_m, dtype=np.float64))
                debug_box.update(localization.occupancy_aabb_min_m, localization.occupancy_aabb_max_m)
            display.show(frame, annotated_obstacle_overlay(frame, targets, detection, localization, "center", 1))
        steps += 1
        time.sleep(sim.TIME_STEP)
    if len(minima) != required_frames:
        raise RuntimeError(f"Need {required_frames} valid visual occupancy frames; got {len(minima)}.")
    occupancy = AxisAlignedBox(tuple(float(v) for v in np.median(np.stack(minima), axis=0)),
                              tuple(float(v) for v in np.median(np.stack(maxima), axis=0)))
    debug_box.update(occupancy.minimum, occupancy.maximum)
    return occupancy, debug_box, steps
