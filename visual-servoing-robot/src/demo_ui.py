"""Small display helper shared by Stage 21 demos; no control data is changed."""

from __future__ import annotations

import cv2
import numpy as np


def status_panel(rgba_buffer: object, width: int, height: int, lines: list[str]) -> bytes:
    """Draw readable status text over a copy of an RGBA camera frame."""
    if isinstance(rgba_buffer, (bytes, bytearray, memoryview)):
        pixels = np.frombuffer(rgba_buffer, dtype=np.uint8)
    else:
        pixels = np.asarray(rgba_buffer, dtype=np.uint8)
    rgba = pixels.reshape(height, width, 4).copy()
    panel_height = min(height, 16 + 22 * len(lines))
    overlay = rgba.copy()
    overlay[:panel_height, :] = (12, 12, 12, 255)
    rgba[:panel_height] = cv2.addWeighted(
        overlay[:panel_height], 0.80, rgba[:panel_height], 0.20, 0
    )
    for index, line in enumerate(lines):
        cv2.putText(
            rgba, str(line), (10, 20 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255, 255), 1, cv2.LINE_AA,
        )
    return rgba.tobytes()
