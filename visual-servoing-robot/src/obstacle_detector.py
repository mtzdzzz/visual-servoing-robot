"""RGB-only yellow obstacle detector for Stage 17.

The detector receives only pixels from the Eye-in-Hand RGB camera.  It has no
PyBullet body ID, segmentation buffer, depth buffer, or world-pose input, so
the resulting mask remains an auditable perception input for RGB-D geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Kept separate from the frozen RED/GREEN/BLUE detector thresholds.  The
# Stage 17 box uses a saturated yellow material, safely away from all target
# colour ranges.
YELLOW_HSV_LOWER = (20, 100, 100)
YELLOW_HSV_UPPER = (40, 255, 255)
OBSTACLE_MIN_CONTOUR_AREA_PIXELS = 100.0
OBSTACLE_MORPH_KERNEL_SIZE = 5


@dataclass(frozen=True)
class ObstacleDetection:
    """One obstacle observation derived exclusively from a rendered RGB frame."""

    detected: bool
    valid: bool
    centroid: tuple[int, int] | None
    contour: np.ndarray | None
    mask: np.ndarray
    bounding_box: tuple[int, int, int, int] | None
    area: float
    state: str


class YellowObstacleDetector:
    """Largest-contour HSV detector for the Stage 17 yellow cuboid."""

    def __init__(
        self,
        min_contour_area_pixels: float = OBSTACLE_MIN_CONTOUR_AREA_PIXELS,
        morphology_kernel_size: int = OBSTACLE_MORPH_KERNEL_SIZE,
    ) -> None:
        if min_contour_area_pixels <= 0.0:
            raise ValueError("min_contour_area_pixels must be positive.")
        if morphology_kernel_size < 1 or morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be a positive odd integer.")
        self.min_contour_area_pixels = min_contour_area_pixels
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size)
        )

    def detect(self, rgba_buffer: object, width: int, height: int) -> ObstacleDetection:
        """Return a yellow RGB mask, contour, 2D box, and centroid.

        ``rgba_buffer`` is the RGB image returned by the same PyBullet camera
        call whose depth is later sampled by ``ObstacleLocalizer``.
        """

        rgba_bytes = rgba_buffer.tobytes() if hasattr(rgba_buffer, "tobytes") else bytes(rgba_buffer)
        expected_size = width * height * 4
        if len(rgba_bytes) != expected_size:
            return self._empty(height, width, "INVALID_RGB_BUFFER")
        try:
            rgba = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((height, width, 4))
            hsv = cv2.cvtColor(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                np.asarray(YELLOW_HSV_LOWER, dtype=np.uint8),
                np.asarray(YELLOW_HSV_UPPER, dtype=np.uint8),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        except cv2.error:
            return self._empty(height, width, "INVALID_RGB_BUFFER")

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contours = [item for item in contours if cv2.contourArea(item) >= self.min_contour_area_pixels]
        if not contours:
            return ObstacleDetection(False, False, None, None, mask, None, 0.0, "NOT_DETECTED")
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-9:
            return ObstacleDetection(False, False, None, contour, mask, None, area, "INVALID_CONTOUR")
        centroid = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )
        return ObstacleDetection(
            True,
            True,
            centroid,
            contour,
            mask,
            tuple(int(value) for value in cv2.boundingRect(contour)),
            area,
            "VALID",
        )

    @staticmethod
    def _empty(height: int, width: int, state: str) -> ObstacleDetection:
        return ObstacleDetection(
            False, False, None, None, np.zeros((height, width), dtype=np.uint8), None, 0.0, state
        )
