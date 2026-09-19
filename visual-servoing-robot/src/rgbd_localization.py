"""RGB-D back-projection utilities for Stage 14.

This module is intentionally independent from the Panda controller and from
PyBullet target objects.  Its localization inputs are limited to the RGB
image, an already detected RGB centroid, the depth buffer from the *same*
camera render, camera intrinsics, and the supplied camera-to-world transform.
Ground-truth data is deliberately not accepted by :class:`RGBDTargetLocalizer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import Sequence

import cv2
import numpy as np

from camera_observation import (
    RED_HSV_LOWER_RANGE,
    RED_HSV_UPPER_RANGE,
    RED_MORPH_KERNEL_SIZE,
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics derived from the active PyBullet FOV and image size."""

    width: int
    height: int
    fov_y_degrees: float
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class DepthStatistics:
    """Depth statistics from red pixels in the small centroid neighbourhood."""

    sample_count: int
    depth_buffer_value: float | None
    minimum_m: float | None
    maximum_m: float | None
    median_m: float | None


@dataclass(frozen=True)
class RGBDLocalizationResult:
    """A fresh RGB-D localization result, or an explicit invalid state."""

    valid: bool
    state: str
    pixel: tuple[int, int] | None
    depth: DepthStatistics
    surface_camera_m: tuple[float, float, float] | None
    surface_world_m: tuple[float, float, float] | None
    center_camera_m: tuple[float, float, float] | None
    center_world_m: tuple[float, float, float] | None


def camera_intrinsics_from_fov(
    width: int,
    height: int,
    fov_y_degrees: float,
) -> CameraIntrinsics:
    """Calculate intrinsics for PyBullet's vertical-FOV perspective camera.

    ``cx`` and ``cy`` intentionally follow the project's existing pixel
    convention: for 640 x 480 they are exactly (320, 240), matching the
    OpenCV visual-servo image centre.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Camera image dimensions must be positive.")
    if not 0.0 < fov_y_degrees < 180.0:
        raise ValueError("Vertical FOV must lie strictly between 0 and 180 degrees.")
    fy = (height / 2.0) / tan(radians(fov_y_degrees) / 2.0)
    fx = fy * (width / height)
    return CameraIntrinsics(
        width=width,
        height=height,
        fov_y_degrees=fov_y_degrees,
        fx=fx,
        fy=fy,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def depth_buffer_to_metric_depth(
    depth_buffer: np.ndarray | float,
    near_plane_m: float,
    far_plane_m: float,
) -> np.ndarray:
    """Convert PyBullet/OpenGL depth-buffer values to positive metric depth.

    The perspective projection used by ``computeProjectionMatrixFOV`` maps a
    positive optical range ``Z`` to the returned buffer ``d`` as

    ``Z = far * near / (far - (far - near) * d)``.

    The returned ``Z`` is distance along the camera's positive optical axis,
    not Euclidean ray length.
    """

    if not 0.0 < near_plane_m < far_plane_m:
        raise ValueError("Expected 0 < near_plane_m < far_plane_m.")
    buffer_array = np.asarray(depth_buffer, dtype=np.float64)
    denominator = far_plane_m - (far_plane_m - near_plane_m) * buffer_array
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = (far_plane_m * near_plane_m) / denominator
    return depth_m


def _rgba_to_red_mask(rgba_buffer: object, width: int, height: int) -> np.ndarray:
    """Recreate the frozen HSV red mask only for depth-sample selection.

    This does not select a centroid and does not feed a controller.  The
    project detector remains the sole source of ``(u, v)``; this small mask is
    used only to reject neighbouring ground/robot depth values around it.
    """

    rgba_bytes = rgba_buffer.tobytes() if hasattr(rgba_buffer, "tobytes") else bytes(rgba_buffer)
    expected_size = width * height * 4
    if len(rgba_bytes) != expected_size:
        raise ValueError(
            f"Unexpected RGBA length: expected {expected_size}, got {len(rgba_bytes)}."
        )
    rgba_image = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((height, width, 4))
    hsv_image = cv2.cvtColor(cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2HSV)
    lower = cv2.inRange(
        hsv_image,
        np.array(RED_HSV_LOWER_RANGE[0], dtype=np.uint8),
        np.array(RED_HSV_LOWER_RANGE[1], dtype=np.uint8),
    )
    upper = cv2.inRange(
        hsv_image,
        np.array(RED_HSV_UPPER_RANGE[0], dtype=np.uint8),
        np.array(RED_HSV_UPPER_RANGE[1], dtype=np.uint8),
    )
    mask = cv2.bitwise_or(lower, upper)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (RED_MORPH_KERNEL_SIZE, RED_MORPH_KERNEL_SIZE),
    )
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


class RGBDTargetLocalizer:
    """Back-project a red-target RGB centroid from a synchronized RGB-D frame.

    Pixel back-projection first produces the frame used by PyBullet's
    ``computeViewMatrix`` (``C_render``): ``+X`` image-right, ``+Y`` image-up
    and ``+Z`` forward.  The camera transform supplied to the localizer is
    the physical hand-eye frame ``C``.  ``render_to_camera`` is therefore an
    explicit, validated fixed axis conversion from ``C_render`` into that
    physical frame before ``T_W_C`` is applied.

    Images index ``v`` downward, therefore the pinhole conversion always has
    ``Y_render = -(v-cy) * Z / fy``.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        near_plane_m: float,
        far_plane_m: float,
        sphere_radius_m: float,
        sample_half_window_pixels: int = 3,
        minimum_depth_samples: int = 5,
    ) -> None:
        if sphere_radius_m <= 0.0:
            raise ValueError("sphere_radius_m must be positive.")
        if sample_half_window_pixels < 1:
            raise ValueError("sample_half_window_pixels must be at least one.")
        if minimum_depth_samples < 1:
            raise ValueError("minimum_depth_samples must be at least one.")
        self.intrinsics = intrinsics
        self.near_plane_m = near_plane_m
        self.far_plane_m = far_plane_m
        self.sphere_radius_m = sphere_radius_m
        self.sample_half_window_pixels = sample_half_window_pixels
        self.minimum_depth_samples = minimum_depth_samples

    def localize(
        self,
        u: int,
        v: int,
        rgba_buffer: object,
        depth_buffer: object,
        camera_to_world: Sequence[Sequence[float]] | np.ndarray,
        render_to_camera: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> RGBDLocalizationResult:
        """Stage 14 red-target entry point; its geometry remains unchanged."""
        try:
            red_mask = _rgba_to_red_mask(
                rgba_buffer, self.intrinsics.width, self.intrinsics.height
            )
        except (ValueError, cv2.error):
            return self._invalid("INVALID_RGB_BUFFER", (u, v))
        return self.localize_from_mask(
            u, v, red_mask, depth_buffer, camera_to_world, render_to_camera
        )

    def localize_from_mask(
        self,
        u: int,
        v: int,
        target_mask: object,
        depth_buffer: object,
        camera_to_world: Sequence[Sequence[float]] | np.ndarray,
        render_to_camera: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> RGBDLocalizationResult:
        """Apply frozen Stage 14 geometry to one target's RGB-derived mask.

        This interface extension retains exactly the Stage 14 depth conversion,
        local median sampling, render-to-C conversion and sphere compensation.
        No object ID or world-coordinate data is accepted.
        """

        if not (0 <= u < self.intrinsics.width and 0 <= v < self.intrinsics.height):
            return self._invalid("PIXEL_OUT_OF_BOUNDS", (u, v))
        matrix = np.asarray(camera_to_world, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return self._invalid("INVALID_CAMERA_TRANSFORM", (u, v))
        axis_transform = (
            np.eye(3, dtype=np.float64)
            if render_to_camera is None
            else np.asarray(render_to_camera, dtype=np.float64)
        )
        if axis_transform.shape != (3, 3) or not np.all(np.isfinite(axis_transform)):
            return self._invalid("INVALID_RENDER_TO_CAMERA_TRANSFORM", (u, v))
        mask_array = np.asarray(target_mask)
        if mask_array.shape != (self.intrinsics.height, self.intrinsics.width):
            return self._invalid("INVALID_TARGET_MASK", (u, v))
        depth_array = np.asarray(depth_buffer, dtype=np.float64)
        if depth_array.size != self.intrinsics.width * self.intrinsics.height:
            return self._invalid("INVALID_DEPTH_BUFFER", (u, v))
        depth_array = depth_array.reshape((self.intrinsics.height, self.intrinsics.width))
        if not np.all(np.isfinite(depth_array)):
            return self._invalid("NONFINITE_DEPTH_BUFFER", (u, v))

        row_start = max(0, v - self.sample_half_window_pixels)
        row_end = min(self.intrinsics.height, v + self.sample_half_window_pixels + 1)
        column_start = max(0, u - self.sample_half_window_pixels)
        column_end = min(self.intrinsics.width, u + self.sample_half_window_pixels + 1)
        local_mask = mask_array[row_start:row_end, column_start:column_end] > 0
        local_buffer = depth_array[row_start:row_end, column_start:column_end]
        local_metric = depth_buffer_to_metric_depth(
            local_buffer, self.near_plane_m, self.far_plane_m
        )
        valid_mask = (
            local_mask
            & np.isfinite(local_metric)
            & (local_metric >= self.near_plane_m)
            & (local_metric <= self.far_plane_m)
        )
        metric_samples = local_metric[valid_mask]
        buffer_samples = local_buffer[valid_mask]
        if metric_samples.size < self.minimum_depth_samples:
            return self._invalid(
                "INSUFFICIENT_TARGET_DEPTH_SAMPLES",
                (u, v),
                sample_count=int(metric_samples.size),
            )

        median_depth = float(np.median(metric_samples))
        statistics = DepthStatistics(
            sample_count=int(metric_samples.size),
            depth_buffer_value=float(np.median(buffer_samples)),
            minimum_m=float(np.min(metric_samples)),
            maximum_m=float(np.max(metric_samples)),
            median_m=median_depth,
        )
        surface_render = self.pixel_depth_to_render_camera(u, v, median_depth)
        surface_camera = axis_transform @ surface_render
        surface_world = self.transform_camera_point_to_world(surface_camera, matrix)

        # The RGB-D measurement lands on the visible sphere surface.  The
        # known centre is one radius farther from the optical centre along the
        # same camera ray.  Both estimates are returned to avoid mixing their
        # distinct geometric meanings in later evaluation.
        ray_direction = surface_camera / np.linalg.norm(surface_camera)
        center_camera = surface_camera + self.sphere_radius_m * ray_direction
        center_world = self.transform_camera_point_to_world(center_camera, matrix)
        if not np.all(np.isfinite(surface_world)) or not np.all(np.isfinite(center_world)):
            return self._invalid("NONFINITE_ESTIMATE", (u, v), statistics.sample_count)
        return RGBDLocalizationResult(
            valid=True,
            state="VALID",
            pixel=(u, v),
            depth=statistics,
            surface_camera_m=tuple(float(value) for value in surface_camera),
            surface_world_m=tuple(float(value) for value in surface_world),
            center_camera_m=tuple(float(value) for value in center_camera),
            center_world_m=tuple(float(value) for value in center_world),
        )

    def pixel_depth_to_render_camera(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """Back-project one image pixel into PyBullet's active render frame."""

        if not np.isfinite(depth_m) or not self.near_plane_m <= depth_m <= self.far_plane_m:
            raise ValueError("Depth is not finite or lies outside the configured clip range.")
        return np.array(
            [
                (u - self.intrinsics.cx) * depth_m / self.intrinsics.fx,
                -(v - self.intrinsics.cy) * depth_m / self.intrinsics.fy,
                depth_m,
            ],
            dtype=np.float64,
        )

    def pixel_depth_to_camera(
        self,
        u: int,
        v: int,
        depth_m: float,
        render_to_camera: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Back-project a pixel then apply the explicit render-to-C conversion."""

        axis_transform = (
            np.eye(3, dtype=np.float64)
            if render_to_camera is None
            else np.asarray(render_to_camera, dtype=np.float64)
        )
        if axis_transform.shape != (3, 3) or not np.all(np.isfinite(axis_transform)):
            raise ValueError("render_to_camera must be a finite 3x3 matrix.")
        return axis_transform @ self.pixel_depth_to_render_camera(u, v, depth_m)

    @staticmethod
    def transform_camera_point_to_world(
        point_camera: Sequence[float] | np.ndarray,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
        """Apply an auditable homogeneous ``T_W_C`` to a camera-frame point."""

        homogeneous = np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
        point_world_h = camera_to_world @ homogeneous
        if abs(point_world_h[3]) < 1e-12:
            raise ValueError("Camera-to-world transform returned an invalid homogeneous point.")
        return point_world_h[:3] / point_world_h[3]

    def _invalid(
        self,
        state: str,
        pixel: tuple[int, int] | None,
        sample_count: int = 0,
    ) -> RGBDLocalizationResult:
        return RGBDLocalizationResult(
            valid=False,
            state=state,
            pixel=pixel,
            depth=DepthStatistics(sample_count, None, None, None, None),
            surface_camera_m=None,
            surface_world_m=None,
            center_camera_m=None,
            center_world_m=None,
        )
