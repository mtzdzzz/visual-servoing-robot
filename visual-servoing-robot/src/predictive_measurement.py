"""Vision-only predicted-centroid adapter shared by Stage 13+ and demos."""

from __future__ import annotations

from camera_observation import RedTargetDetection
from target_motion_estimator import MotionEstimate

PREDICTION_ALPHA = 0.20
PREDICTION_OFFSET_CLAMP_PIXELS = 12.0


def predicted_measurement(
    raw_detection: RedTargetDetection,
    estimate: MotionEstimate | None,
) -> tuple[RedTargetDetection, str, bool]:
    if not raw_detection.detected:
        return raw_detection, "TARGET_LOST", False
    if estimate is None or not estimate.estimator_valid:
        return raw_detection, "CURRENT_FALLBACK", False
    assert raw_detection.centroid is not None
    u_raw, v_raw = raw_detection.centroid
    offset_u = estimate.u_pred - u_raw
    offset_v = estimate.v_pred - v_raw
    limited_u = max(-PREDICTION_OFFSET_CLAMP_PIXELS, min(PREDICTION_OFFSET_CLAMP_PIXELS, offset_u))
    limited_v = max(-PREDICTION_OFFSET_CLAMP_PIXELS, min(PREDICTION_OFFSET_CLAMP_PIXELS, offset_v))
    clamped = abs(limited_u - offset_u) > 1e-12 or abs(limited_v - offset_v) > 1e-12
    centroid = (u_raw + limited_u, v_raw + limited_v)
    cx, cy = raw_detection.image_center
    return RedTargetDetection(
        detected=True, bounding_box=raw_detection.bounding_box,
        contour_area=raw_detection.contour_area, centroid=centroid,
        image_center=raw_detection.image_center,
        pixel_error=(centroid[0] - cx, centroid[1] - cy),
        annotated_rgba_buffer=raw_detection.annotated_rgba_buffer,
    ), "PREDICTED_CLAMPED" if clamped else "PREDICTED", clamped

