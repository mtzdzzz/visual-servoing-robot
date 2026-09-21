"""RGB-D obstacle point-cloud and conservative occupancy estimation for Stage 17.

This module deliberately accepts only a yellow RGB-derived mask, synchronized
depth samples, Stage-14 camera intrinsics, and the current ``T_W_C``.  It does
not accept an obstacle body ID, pose, AABB, or any ground-truth information.
Ground truth is evaluated separately by the Stage 17 runner after this module
has completed its estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rgbd_localization import CameraIntrinsics, depth_buffer_to_metric_depth


@dataclass(frozen=True)
class ObstacleLocalizationResult:
    """Fresh visible-surface cloud and a known-shape conservative occupancy."""

    valid: bool
    state: str
    valid_depth_samples: int
    depth_min_m: float | None
    depth_median_m: float | None
    depth_max_m: float | None
    visible_points_camera_m: np.ndarray | None
    visible_points_world_m: np.ndarray | None
    visible_aabb_min_m: tuple[float, float, float] | None
    visible_aabb_max_m: tuple[float, float, float] | None
    estimated_center_world_m: tuple[float, float, float] | None
    occupancy_aabb_min_m: tuple[float, float, float] | None
    occupancy_aabb_max_m: tuple[float, float, float] | None


class ObstacleLocalizer:
    """Construct an RGB-D visible cloud and fixed-margin cuboid occupancy.

    The cuboid's dimensions are a documented known-shape prior.  Its spatial
    location is estimated from the visual point cloud and camera pose only.
    Thus the output is a conservative known-shape occupancy estimate rather
    than a claim that one RGB-D view reconstructs every hidden face.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        near_plane_m: float,
        far_plane_m: float,
        known_dimensions_m: Sequence[float],
        safety_margin_m: float,
        pixel_stride: int = 3,
        minimum_depth_samples: int = 30,
    ) -> None:
        dimensions = np.asarray(known_dimensions_m, dtype=np.float64)
        if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            raise ValueError("known_dimensions_m must contain three positive finite values.")
        if safety_margin_m < 0.0 or pixel_stride < 1 or minimum_depth_samples < 1:
            raise ValueError("Invalid safety margin, stride, or minimum depth sample count.")
        self.intrinsics = intrinsics
        self.near_plane_m = near_plane_m
        self.far_plane_m = far_plane_m
        self.known_dimensions_m = dimensions
        self.safety_margin_m = float(safety_margin_m)
        self.pixel_stride = int(pixel_stride)
        self.minimum_depth_samples = int(minimum_depth_samples)

    def localize(
        self,
        obstacle_mask: object,
        depth_buffer: object,
        camera_to_world: Sequence[Sequence[float]] | np.ndarray,
        render_to_camera: Sequence[Sequence[float]] | np.ndarray,
    ) -> ObstacleLocalizationResult:
        """Back-project RGB mask pixels with Stage 14's depth and axes path."""

        mask = np.asarray(obstacle_mask)
        if mask.shape != (self.intrinsics.height, self.intrinsics.width):
            return self._invalid("INVALID_OBSTACLE_MASK")
        depth = np.asarray(depth_buffer, dtype=np.float64)
        if depth.size != self.intrinsics.width * self.intrinsics.height:
            return self._invalid("INVALID_DEPTH_BUFFER")
        depth = depth.reshape((self.intrinsics.height, self.intrinsics.width))
        transform = np.asarray(camera_to_world, dtype=np.float64)
        axes = np.asarray(render_to_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return self._invalid("INVALID_CAMERA_TRANSFORM")
        if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
            return self._invalid("INVALID_RENDER_TO_CAMERA_TRANSFORM")

        rows, columns = np.nonzero(mask > 0)
        if rows.size == 0:
            return self._invalid("EMPTY_OBSTACLE_MASK")
        rows, columns = rows[:: self.pixel_stride], columns[:: self.pixel_stride]
        buffer_samples = depth[rows, columns]
        metric_samples = depth_buffer_to_metric_depth(
            buffer_samples, self.near_plane_m, self.far_plane_m
        )
        valid = (
            np.isfinite(metric_samples)
            & (metric_samples > self.near_plane_m + 1e-6)
            & (metric_samples < self.far_plane_m - 1e-6)
        )
        rows, columns, metric_samples = rows[valid], columns[valid], metric_samples[valid]
        if metric_samples.size < self.minimum_depth_samples:
            return self._invalid("INSUFFICIENT_OBSTACLE_DEPTH_SAMPLES", int(metric_samples.size))

        # This is exactly the Stage 14 pinhole convention in vector form:
        # C_render=[(u-cx)Z/fx, -(v-cy)Z/fy, Z].  Applying the validated
        # C_render->C matrix happens before homogeneous T_W_C.
        render_points = np.column_stack(
            (
                (columns - self.intrinsics.cx) * metric_samples / self.intrinsics.fx,
                -(rows - self.intrinsics.cy) * metric_samples / self.intrinsics.fy,
                metric_samples,
            )
        )
        camera_points = (axes @ render_points.T).T
        world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
        if not np.all(np.isfinite(camera_points)) or not np.all(np.isfinite(world_points)):
            return self._invalid("NONFINITE_POINT_CLOUD", int(metric_samples.size))

        visible_min = np.min(world_points, axis=0)
        visible_max = np.max(world_points, axis=0)
        estimated_center = self._estimate_known_shape_center(
            visible_min, visible_max, transform[:3, 3]
        )
        half_dimensions = self.known_dimensions_m / 2.0
        occupancy_min = estimated_center - half_dimensions - self.safety_margin_m
        occupancy_max = estimated_center + half_dimensions + self.safety_margin_m
        if np.any(occupancy_min >= occupancy_max) or not np.all(np.isfinite(estimated_center)):
            return self._invalid("INVALID_OCCUPANCY", int(metric_samples.size))
        return ObstacleLocalizationResult(
            True,
            "VALID",
            int(metric_samples.size),
            float(np.min(metric_samples)),
            float(np.median(metric_samples)),
            float(np.max(metric_samples)),
            camera_points,
            world_points,
            tuple(float(value) for value in visible_min),
            tuple(float(value) for value in visible_max),
            tuple(float(value) for value in estimated_center),
            tuple(float(value) for value in occupancy_min),
            tuple(float(value) for value in occupancy_max),
        )

    def _estimate_known_shape_center(
        self,
        visible_min: np.ndarray,
        visible_max: np.ndarray,
        camera_world_position: np.ndarray,
    ) -> np.ndarray:
        """Estimate a cuboid centre from its camera-facing visible faces.

        For every world axis, the camera side identifies whether the observed
        extreme is the negative or positive face.  The known half dimension
        shifts that RGB-D-visible face to the cuboid centre.  This uses neither
        a PyBullet pose nor an AABB and intentionally leaves a fixed safety
        margin around the resulting occupancy.
        """

        visible_mid = (visible_min + visible_max) / 2.0
        estimated = visible_mid.copy()
        for axis in range(3):
            if camera_world_position[axis] <= visible_mid[axis]:
                estimated[axis] = visible_min[axis] + self.known_dimensions_m[axis] / 2.0
            else:
                estimated[axis] = visible_max[axis] - self.known_dimensions_m[axis] / 2.0
        return estimated

    def _invalid(self, state: str, valid_depth_samples: int = 0) -> ObstacleLocalizationResult:
        return ObstacleLocalizationResult(
            False, state, valid_depth_samples, None, None, None, None, None,
            None, None, None, None, None,
        )
