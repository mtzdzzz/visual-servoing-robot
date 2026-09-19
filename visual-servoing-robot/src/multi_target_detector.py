"""Classical RGB-only RED/GREEN/BLUE target detector for Stage 15.

This module receives only the Eye-in-Hand RGBA image.  It never receives a
PyBullet object identifier, segmentation buffer, target pose, or depth data.
Each colour has an independent mask and largest-contour decision so nearby
objects of different colours cannot be merged into one target detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from camera_observation import (
    RED_HSV_LOWER_RANGE,
    RED_HSV_UPPER_RANGE,
    RED_MIN_CONTOUR_AREA_PIXELS,
    RED_MORPH_KERNEL_SIZE,
)


@dataclass(frozen=True)
class TargetClassConfig:
    """Fixed RGB-class definition; IDs are stable across all Stage 15 scenes."""

    target_id: str
    class_name: str
    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    overlay_bgr: tuple[int, int, int]


TARGET_CLASSES: tuple[TargetClassConfig, ...] = (
    TargetClassConfig(
        "target_1", "RED", (RED_HSV_LOWER_RANGE, RED_HSV_UPPER_RANGE), (0, 0, 255)
    ),
    TargetClassConfig(
        "target_2", "GREEN", (((45, 80, 50), (85, 255, 255)),), (0, 255, 0)
    ),
    TargetClassConfig(
        "target_3", "BLUE", (((100, 80, 50), (140, 255, 255)),), (255, 0, 0)
    ),
)


@dataclass(frozen=True)
class DetectedTarget:
    """One RGB-derived target observation and its own binary pixel mask."""

    target_id: str
    class_name: str
    detected: bool
    valid: bool
    confidence: float
    centroid: tuple[int, int] | None
    contour: np.ndarray | None
    mask: np.ndarray
    bounding_box: tuple[int, int, int, int] | None
    area: float


class MultiTargetDetector:
    """Detect one largest valid contour for each frozen colour class."""

    def __init__(
        self,
        class_configs: Sequence[TargetClassConfig] = TARGET_CLASSES,
        min_contour_area_pixels: float = RED_MIN_CONTOUR_AREA_PIXELS,
        morphology_kernel_size: int = RED_MORPH_KERNEL_SIZE,
    ) -> None:
        if not class_configs:
            raise ValueError("At least one target colour class is required.")
        if min_contour_area_pixels <= 0.0:
            raise ValueError("min_contour_area_pixels must be positive.")
        if morphology_kernel_size < 1 or morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be a positive odd integer.")
        self.class_configs = tuple(class_configs)
        self.min_contour_area_pixels = min_contour_area_pixels
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_kernel_size, morphology_kernel_size),
        )

    def detect(self, rgba_buffer: object, width: int, height: int) -> list[DetectedTarget]:
        """Return RED, GREEN and BLUE detections in their stable ID order."""

        rgba_bytes = rgba_buffer.tobytes() if hasattr(rgba_buffer, "tobytes") else bytes(rgba_buffer)
        expected_size = width * height * 4
        if len(rgba_bytes) != expected_size:
            raise ValueError(
                f"Unexpected RGBA length: expected {expected_size}, got {len(rgba_bytes)}."
            )
        rgba_image = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((height, width, 4))
        bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return [self._detect_class(hsv_image, config) for config in self.class_configs]

    def _detect_class(
        self,
        hsv_image: np.ndarray,
        config: TargetClassConfig,
    ) -> DetectedTarget:
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in config.hsv_ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv_image,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                ),
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        valid_contours = [
            contour for contour in contours if cv2.contourArea(contour) >= self.min_contour_area_pixels
        ]
        if not valid_contours:
            return DetectedTarget(
                target_id=config.target_id,
                class_name=config.class_name,
                detected=False,
                valid=False,
                confidence=0.0,
                centroid=None,
                contour=None,
                mask=mask,
                bounding_box=None,
                area=0.0,
            )
        contour = max(valid_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-9:
            return DetectedTarget(
                target_id=config.target_id,
                class_name=config.class_name,
                detected=False,
                valid=False,
                confidence=0.0,
                centroid=None,
                contour=None,
                mask=mask,
                bounding_box=None,
                area=area,
            )
        centroid = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )
        # Area-normalised confidence is metadata for logs/GUI only.  It does
        # not select, rank, or control any robot behaviour.
        confidence = min(1.0, area / (4.0 * self.min_contour_area_pixels))
        return DetectedTarget(
            target_id=config.target_id,
            class_name=config.class_name,
            detected=True,
            valid=True,
            confidence=confidence,
            centroid=centroid,
            contour=contour,
            mask=mask,
            bounding_box=tuple(int(value) for value in cv2.boundingRect(contour)),
            area=area,
        )


def annotate_multi_target_detections(
    rgba_buffer: object,
    width: int,
    height: int,
    detections: Sequence[DetectedTarget],
) -> bytes:
    """Draw RGB-only detections for display after all masks are computed."""

    rgba_bytes = rgba_buffer.tobytes() if hasattr(rgba_buffer, "tobytes") else bytes(rgba_buffer)
    rgba_image = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    config_by_name = {config.class_name: config for config in TARGET_CLASSES}
    for detection in detections:
        config = config_by_name[detection.class_name]
        if not detection.detected:
            continue
        assert detection.contour is not None and detection.centroid is not None
        assert detection.bounding_box is not None
        cv2.drawContours(bgr, [detection.contour], -1, config.overlay_bgr, 2)
        x, y, width_box, height_box = detection.bounding_box
        cv2.rectangle(bgr, (x, y), (x + width_box, y + height_box), config.overlay_bgr, 2)
        cv2.drawMarker(
            bgr, detection.centroid, config.overlay_bgr,
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
        )
        cv2.putText(
            bgr,
            f"{detection.class_name[0]} {detection.centroid}",
            (max(6, x), max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            config.overlay_bgr,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()
