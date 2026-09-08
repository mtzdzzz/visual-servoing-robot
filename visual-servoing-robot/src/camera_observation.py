"""PyBullet-only ground-target and RGB camera observation helpers.

This module is deliberately independent of Panda joint control and inverse
kinematics.  It receives an already-computed Camera Reference world pose,
then creates a ground target and renders an RGB image from a rigidly attached
virtual camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan, degrees, radians, sqrt, tan
from pathlib import Path
from struct import pack
from typing import Sequence

import cv2
import numpy as np
import pybullet as p


GROUND_TARGET_RADIUS = 0.04  # metres
GROUND_TARGET_WORLD_POSITION = (0.35, -0.25, GROUND_TARGET_RADIUS)
GROUND_TARGET_RGBA = (1.0, 0.0, 0.0, 1.0)
CAMERA_IMAGE_WIDTH = 640
CAMERA_IMAGE_HEIGHT = 480
CAMERA_FOV_Y_DEGREES = 60.0
CAMERA_NEAR_PLANE = 0.01
CAMERA_FAR_PLANE = 3.0
# HSV red wraps around OpenCV's hue range (0--180), therefore detection uses
# the two edge intervals below. These parameters apply only to RGB-image
# analysis and never feed a robot controller.
RED_HSV_LOWER_RANGE = ((0, 100, 70), (10, 255, 255))
RED_HSV_UPPER_RANGE = ((170, 100, 70), (180, 255, 255))
RED_MORPH_KERNEL_SIZE = 5
RED_MIN_CONTOUR_AREA_PIXELS = 80.0
CAMERA_RGB_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "eye_in_hand_rgb_latest.bmp"
)
STAGE4_FINAL_RGB_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "stage4_final_rgb.bmp"
)
STAGE4_SECOND_POSE_RGB_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "stage4_second_pose_rgb.bmp"
)
CAMERA_FORCED_LOOK_AT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "eye_in_hand_rgb_forced_look_at.bmp"
)
CAMERA_OFFSET_DEBUG_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "eye_in_hand_rgb_offset_debug.bmp"
)
CAMERA_STATIC_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "eye_in_hand_rgb_static_debug.bmp"
)

# The supplied pose is already the physical optical-center camera frame C,
# composed in simulation.py as T_W_C = T_W_E @ T_E_C.  The old extra virtual
# camera rotation has been removed: T_C_K is identity, so K and C are the
# same frame.  C is right-handed: +X_C is image-right, +Y_C is image-up, and
# +Z_C is the optical/viewing axis.  PyBullet's computeViewMatrix receives
# +Z_C as its world target direction and +Y_C as its camera-up direction.
T_C_CAMERA_POSITION = (0.0, 0.0, 0.0)
CAMERA_VIEW_DEBUG_LINE_LENGTH = 0.5  # metres
T_C_CAMERA_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class CameraRenderParameters:
    """Exact values passed to PyBullet's ``getCameraImage`` call."""

    eye_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    up_vector: tuple[float, float, float]
    view_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]


@dataclass(frozen=True)
class CameraObservation:
    """One RGB render from camera K rigidly attached to Camera Reference C."""

    camera_reference_world_position: tuple[float, float, float]
    camera_reference_world_orientation: tuple[float, float, float, float]
    camera_world_position: tuple[float, float, float]
    camera_world_orientation: tuple[float, float, float, float]
    camera_x_axis_world: tuple[float, float, float]
    camera_y_axis_world: tuple[float, float, float]
    viewing_direction: tuple[float, float, float]
    actual_view_direction: tuple[float, float, float]
    vector_to_target: tuple[float, float, float]
    target_camera_coordinates: tuple[float, float, float]
    target_render_camera_coordinates: tuple[float, float, float]
    render_parameters: CameraRenderParameters
    image_width: int
    image_height: int
    target_visible_pixel_count: int
    image_path: Path


@dataclass(frozen=True)
class LiveCameraFrame:
    """One unsaved RGB frame rendered from the current fixed camera pose."""

    camera_world_position: tuple[float, float, float]
    camera_world_orientation: tuple[float, float, float, float]
    optical_axis_world: tuple[float, float, float]
    render_parameters: CameraRenderParameters
    image_width: int
    image_height: int
    rgba_buffer: object


@dataclass(frozen=True)
class RedTargetDetection:
    """Result obtained strictly from one Eye-in-Hand RGB frame.

    This data type deliberately contains no PyBullet body identifier,
    segmentation mask, or target world-coordinate input. It is suitable for
    Stage 5 image-space observation only; no control command is derived from
    it.
    """

    detected: bool
    bounding_box: tuple[int, int, int, int] | None
    contour_area: float | None
    centroid: tuple[int, int] | None
    image_center: tuple[int, int]
    pixel_error: tuple[int, int] | None
    annotated_rgba_buffer: bytes


def create_red_ground_target(client_id: int) -> tuple[int, list[float]]:
    """Create a static red sphere resting on the z=0 ground plane."""
    collision_shape_id = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=GROUND_TARGET_RADIUS,
        physicsClientId=client_id,
    )
    visual_shape_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=GROUND_TARGET_RADIUS,
        rgbaColor=GROUND_TARGET_RGBA,
        physicsClientId=client_id,
    )
    target_body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=GROUND_TARGET_WORLD_POSITION,
        physicsClientId=client_id,
    )
    target_position, _ = p.getBasePositionAndOrientation(
        target_body_id,
        physicsClientId=client_id,
    )
    marker_z = target_position[2] + GROUND_TARGET_RADIUS
    p.addUserDebugLine(
        (target_position[0] - GROUND_TARGET_RADIUS, target_position[1], marker_z),
        (target_position[0] + GROUND_TARGET_RADIUS, target_position[1], marker_z),
        lineColorRGB=[1.0, 1.0, 1.0],
        lineWidth=2,
        lifeTime=0,
        physicsClientId=client_id,
    )
    p.addUserDebugLine(
        (target_position[0], target_position[1] - GROUND_TARGET_RADIUS, marker_z),
        (target_position[0], target_position[1] + GROUND_TARGET_RADIUS, marker_z),
        lineColorRGB=[1.0, 1.0, 1.0],
        lineWidth=2,
        lifeTime=0,
        physicsClientId=client_id,
    )
    p.addUserDebugText(
        "Red ground target",
        (target_position[0], target_position[1], marker_z + 0.075),
        textColorRGB=[1.0, 0.2, 0.2],
        textSize=1.4,
        lifeTime=0,
        physicsClientId=client_id,
    )
    return target_body_id, list(target_position)


def _world_direction(
    camera_world_position: Sequence[float],
    camera_world_orientation: Sequence[float],
    local_direction: Sequence[float],
) -> list[float]:
    """Transform one local unit direction into world coordinates."""
    endpoint, _ = p.multiplyTransforms(
        camera_world_position,
        camera_world_orientation,
        local_direction,
        (0.0, 0.0, 0.0, 1.0),
    )
    return [endpoint[index] - camera_world_position[index] for index in range(3)]


def get_rigid_camera_pose(
    camera_reference_world_position: Sequence[float],
    camera_reference_world_orientation: Sequence[float],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Compose the rigid virtual-camera pose and its world-frame axes from C."""
    camera_position, camera_orientation = p.multiplyTransforms(
        camera_reference_world_position,
        camera_reference_world_orientation,
        T_C_CAMERA_POSITION,
        T_C_CAMERA_ORIENTATION,
    )
    camera_x_axis = _world_direction(
        camera_position,
        camera_orientation,
        (1.0, 0.0, 0.0),
    )
    camera_y_axis = _world_direction(
        camera_position,
        camera_orientation,
        (0.0, 1.0, 0.0),
    )
    optical_axis = _world_direction(
        camera_position,
        camera_orientation,
        (0.0, 0.0, 1.0),
    )
    return (
        list(camera_position),
        list(camera_orientation),
        camera_x_axis,
        camera_y_axis,
        optical_axis,
    )


def _normalize(vector: Sequence[float]) -> list[float]:
    """Normalize a non-zero direction vector."""
    length = sqrt(sum(component * component for component in vector))
    if length < 1e-9:
        raise ValueError("A camera direction vector must have non-zero length.")
    return [component / length for component in vector]


def _cross(first_vector: Sequence[float], second_vector: Sequence[float]) -> list[float]:
    """Return the three-dimensional cross product ``first_vector x second_vector``."""
    return [
        first_vector[1] * second_vector[2] - first_vector[2] * second_vector[1],
        first_vector[2] * second_vector[0] - first_vector[0] * second_vector[2],
        first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0],
    ]


def build_camera_render_parameters(
    eye_position: Sequence[float],
    target_position: Sequence[float],
    up_vector: Sequence[float],
) -> CameraRenderParameters:
    """Build and retain the exact view/projection arguments given to PyBullet."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=list(eye_position),
        cameraTargetPosition=list(target_position),
        cameraUpVector=list(up_vector),
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV_Y_DEGREES,
        aspect=CAMERA_IMAGE_WIDTH / CAMERA_IMAGE_HEIGHT,
        nearVal=CAMERA_NEAR_PLANE,
        farVal=CAMERA_FAR_PLANE,
    )
    return CameraRenderParameters(
        eye_position=tuple(eye_position),
        target_position=tuple(target_position),
        up_vector=tuple(up_vector),
        view_matrix=tuple(view_matrix),
        projection_matrix=tuple(projection_matrix),
    )


def _rgba_buffer_to_bytes(rgba_buffer: object) -> bytes:
    if hasattr(rgba_buffer, "tobytes"):
        return rgba_buffer.tobytes()  # type: ignore[union-attr]
    return bytes(rgba_buffer)  # type: ignore[arg-type]


def _save_rgb_as_bmp(
    image_width: int,
    image_height: int,
    rgba_buffer: object,
    image_path: Path,
) -> Path:
    """Save a dependency-free 24-bit BMP preview; no OpenCV/Pillow is used."""
    rgba_bytes = _rgba_buffer_to_bytes(rgba_buffer)
    expected_rgba_byte_count = image_width * image_height * 4
    if len(rgba_bytes) != expected_rgba_byte_count:
        raise RuntimeError(
            "PyBullet returned an unexpected RGBA buffer length: "
            f"expected {expected_rgba_byte_count}, got {len(rgba_bytes)}."
        )

    row_byte_count = image_width * 3
    padded_row_byte_count = (row_byte_count + 3) & ~3
    pixel_bytes = bytearray(padded_row_byte_count * image_height)
    for source_row in range(image_height):
        source_row_offset = source_row * image_width * 4
        destination_row_offset = (image_height - 1 - source_row) * padded_row_byte_count
        for column in range(image_width):
            source_offset = source_row_offset + column * 4
            destination_offset = destination_row_offset + column * 3
            pixel_bytes[destination_offset] = rgba_bytes[source_offset + 2]
            pixel_bytes[destination_offset + 1] = rgba_bytes[source_offset + 1]
            pixel_bytes[destination_offset + 2] = rgba_bytes[source_offset]

    bitmap_file_size = 14 + 40 + len(pixel_bytes)
    bitmap_file_header = pack("<2sIHHI", b"BM", bitmap_file_size, 0, 0, 54)
    bitmap_info_header = pack(
        "<IIIHHIIIIII",
        40,
        image_width,
        image_height,
        1,
        24,
        0,
        len(pixel_bytes),
        2835,
        2835,
        0,
        0,
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(bitmap_file_header + bitmap_info_header + pixel_bytes)
    return image_path


def _count_visible_object_pixels(segmentation_buffer: object, object_id: int) -> int:
    values = (
        segmentation_buffer.flat  # type: ignore[union-attr]
        if hasattr(segmentation_buffer, "flat")
        else segmentation_buffer
    )
    object_id_mask = (1 << 24) - 1
    return sum(1 for value in values if (int(value) & object_id_mask) == object_id)


def _render_with_parameters(
    render_parameters: CameraRenderParameters,
    client_id: int,
) -> tuple[int, int, object, object]:
    """Call ``getCameraImage`` with the retained, auditable render arguments."""
    (
        image_width,
        image_height,
        rgba_buffer,
        _,
        segmentation_buffer,
    ) = p.getCameraImage(
        CAMERA_IMAGE_WIDTH,
        CAMERA_IMAGE_HEIGHT,
        viewMatrix=render_parameters.view_matrix,
        projectionMatrix=render_parameters.projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=client_id,
    )
    return image_width, image_height, rgba_buffer, segmentation_buffer


def _render_rgb_with_parameters(
    render_parameters: CameraRenderParameters,
    client_id: int,
) -> tuple[int, int, object]:
    """Render RGB only for the real-time Stage 5 image-analysis path.

    No segmentation flag is requested and no ground-truth render output is
    returned to the caller.
    """
    connection_info = p.getConnectionInfo(physicsClientId=client_id)
    # The live Stage 5/6 path remains the same 640x480 getCameraImage RGB
    # camera with the same view/projection matrices.  In a GUI connection we
    # select PyBullet's hardware renderer so rendering does not become the
    # dominant visual-servo delay. DIRECT retains Tiny Renderer as a portable
    # fallback for headless tests.
    renderer = (
        p.ER_BULLET_HARDWARE_OPENGL
        if connection_info.get("connectionMethod") == p.GUI
        else p.ER_TINY_RENDERER
    )
    image_width, image_height, rgba_buffer, _, _ = p.getCameraImage(
        CAMERA_IMAGE_WIDTH,
        CAMERA_IMAGE_HEIGHT,
        viewMatrix=render_parameters.view_matrix,
        projectionMatrix=render_parameters.projection_matrix,
        renderer=renderer,
        physicsClientId=client_id,
    )
    return image_width, image_height, rgba_buffer


def render_live_eye_in_hand_rgb_frame(
    camera_reference_world_position: Sequence[float],
    camera_reference_world_orientation: Sequence[float],
    client_id: int,
) -> LiveCameraFrame:
    """Render one normal fixed-mount camera frame without any target look-at.

    The supplied pose is the current physical C pose.  The view target is
    constructed only from C's fixed optical axis, never from the red target:
    ``target = eye + R_W_C @ [0, 0, 1]``.
    """
    (
        camera_world_position,
        camera_world_orientation,
        _,
        camera_y_axis_world,
        optical_axis_world,
    ) = get_rigid_camera_pose(
        camera_reference_world_position,
        camera_reference_world_orientation,
    )
    render_parameters = build_camera_render_parameters(
        camera_world_position,
        [
            position + direction
            for position, direction in zip(camera_world_position, optical_axis_world)
        ],
        camera_y_axis_world,
    )
    image_width, image_height, rgba_buffer = _render_rgb_with_parameters(
        render_parameters,
        client_id,
    )
    return LiveCameraFrame(
        camera_world_position=tuple(camera_world_position),
        camera_world_orientation=tuple(camera_world_orientation),
        optical_axis_world=tuple(optical_axis_world),
        render_parameters=render_parameters,
        image_width=image_width,
        image_height=image_height,
        rgba_buffer=rgba_buffer,
    )


def detect_red_target_from_live_rgb(frame: LiveCameraFrame) -> RedTargetDetection:
    """Detect the largest plausible red contour from raw Eye-in-Hand RGB.

    The input is exclusively LiveCameraFrame.rgba_buffer produced by
    getCameraImage. In particular, this function accepts neither a target
    position nor a PyBullet segmentation/object-ID value.
    """
    rgba_bytes = _rgba_buffer_to_bytes(frame.rgba_buffer)
    expected_byte_count = frame.image_width * frame.image_height * 4
    if len(rgba_bytes) != expected_byte_count:
        raise RuntimeError(
            "PyBullet returned an unexpected live RGBA buffer length: "
            f"expected {expected_byte_count}, got {len(rgba_bytes)}."
        )

    rgba_image = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(
        (frame.image_height, frame.image_width, 4)
    )
    bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower_red_mask = cv2.inRange(
        hsv_image,
        np.array(RED_HSV_LOWER_RANGE[0], dtype=np.uint8),
        np.array(RED_HSV_LOWER_RANGE[1], dtype=np.uint8),
    )
    upper_red_mask = cv2.inRange(
        hsv_image,
        np.array(RED_HSV_UPPER_RANGE[0], dtype=np.uint8),
        np.array(RED_HSV_UPPER_RANGE[1], dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask)
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (RED_MORPH_KERNEL_SIZE, RED_MORPH_KERNEL_SIZE),
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, morph_kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morph_kernel)

    contour_result = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = contour_result[-2]
    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= RED_MIN_CONTOUR_AREA_PIXELS
    ]

    annotated_bgr = bgr_image.copy()
    image_center = (frame.image_width // 2, frame.image_height // 2)
    cv2.drawMarker(
        annotated_bgr,
        image_center,
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )
    # This is a display-only annotation. It does not participate in HSV
    # thresholding, contour selection, or pixel-error calculation.
    cv2.putText(
        annotated_bgr,
        f"Center: ({image_center[0]}, {image_center[1]})",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if not valid_contours:
        cv2.putText(
            annotated_bgr,
            "Target not detected",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        annotated_rgba = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGBA).tobytes()
        return RedTargetDetection(
            detected=False,
            bounding_box=None,
            contour_area=None,
            centroid=None,
            image_center=image_center,
            pixel_error=None,
            annotated_rgba_buffer=annotated_rgba,
        )

    target_contour = max(valid_contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(target_contour))
    bounding_box = tuple(int(value) for value in cv2.boundingRect(target_contour))
    moments = cv2.moments(target_contour)
    if moments["m00"] <= 1e-9:
        raise RuntimeError("Selected red contour has zero area moment.")
    centroid = (
        int(round(moments["m10"] / moments["m00"])),
        int(round(moments["m01"] / moments["m00"])),
    )
    pixel_error = (
        centroid[0] - image_center[0],
        centroid[1] - image_center[1],
    )
    x, y, width, height = bounding_box
    cv2.drawContours(annotated_bgr, [target_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(
        annotated_bgr,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2,
    )
    cv2.drawMarker(
        annotated_bgr,
        centroid,
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )
    cv2.putText(
        annotated_bgr,
        f"Target: ({centroid[0]}, {centroid[1]})",
        (max(8, x), max(24, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_bgr,
        f"ex: {pixel_error[0]}  ey: {pixel_error[1]}",
        (max(8, x), min(frame.image_height - 12, y + height + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    annotated_rgba = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGBA).tobytes()
    return RedTargetDetection(
        detected=True,
        bounding_box=bounding_box,
        contour_area=contour_area,
        centroid=centroid,
        image_center=image_center,
        pixel_error=pixel_error,
        annotated_rgba_buffer=annotated_rgba,
    )


def add_servo_state_overlay(
    frame: LiveCameraFrame,
    annotated_rgba_buffer: object,
    servo_state: str,
) -> bytes:
    """Overlay the current Stage 6A state without changing RGB detection."""
    rgba_bytes = _rgba_buffer_to_bytes(annotated_rgba_buffer)
    expected_byte_count = frame.image_width * frame.image_height * 4
    if len(rgba_bytes) != expected_byte_count:
        raise RuntimeError(
            "Annotated Eye-in-Hand RGBA buffer has unexpected length: "
            f"expected {expected_byte_count}, got {len(rgba_bytes)}."
        )
    rgba_image = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(
        (frame.image_height, frame.image_width, 4)
    )
    annotated_bgr = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    cv2.rectangle(
        annotated_bgr,
        (8, frame.image_height - 42),
        (min(frame.image_width - 8, 620), frame.image_height - 8),
        (30, 30, 30),
        thickness=-1,
    )
    cv2.putText(
        annotated_bgr,
        f"Servo: {servo_state}",
        (16, frame.image_height - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGBA).tobytes()


class EyeInHandRgbDisplay:
    """A lightweight, non-blocking Tkinter viewer for live PyBullet RGB frames."""

    def __init__(self, title: str = "Eye-in-Hand RGB") -> None:
        self._root: object | None = None
        self._label: object | None = None
        self._tk: object | None = None
        self._closed = False
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title(title)
            root.resizable(False, False)
            label = tk.Label(root)
            label.pack()
            root.protocol("WM_DELETE_WINDOW", self.close)
            self._root = root
            self._label = label
            self._tk = tk
            root.update_idletasks()
            root.update()
            print(f"Opened real-time {title!r} display (Tkinter).")
        except Exception as error:  # GUI availability is platform-dependent.
            self._closed = True
            print(f"Live Eye-in-Hand RGB display unavailable: {error}")

    @property
    def is_open(self) -> bool:
        return not self._closed and self._root is not None

    def close(self) -> None:
        """Close the viewer without altering the PyBullet simulation."""
        if self._closed:
            return
        self._closed = True
        if self._root is not None:
            try:
                self._root.destroy()  # type: ignore[union-attr]
            except Exception:
                pass
        self._root = None
        self._label = None

    def show(
        self,
        frame: LiveCameraFrame,
        rgba_buffer: object | None = None,
    ) -> None:
        """Display a raw or annotated RGBA Eye-in-Hand frame without saving it."""
        if not self.is_open or self._root is None or self._label is None or self._tk is None:
            return
        try:
            rgba_bytes = _rgba_buffer_to_bytes(
                frame.rgba_buffer if rgba_buffer is None else rgba_buffer
            )
            expected_byte_count = frame.image_width * frame.image_height * 4
            if len(rgba_bytes) != expected_byte_count:
                raise RuntimeError(
                    "PyBullet returned an unexpected live RGBA buffer length: "
                    f"expected {expected_byte_count}, got {len(rgba_bytes)}."
                )
            # Keep the exact RGB pixels but avoid a 307,200-iteration Python
            # loop on every live frame. This display-only vectorised alpha
            # drop does not alter getCameraImage, HSV detection, or overlays.
            rgba_image = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(
                (frame.image_height, frame.image_width, 4)
            )
            rgb_bytes = rgba_image[:, :, :3].tobytes()
            ppm_header = f"P6\n{frame.image_width} {frame.image_height}\n255\n".encode("ascii")
            image = self._tk.PhotoImage(data=ppm_header + rgb_bytes, format="PPM")
            self._label.configure(image=image)
            self._label.image = image
            self._root.update_idletasks()
            self._root.update()
        except Exception as error:
            print(f"Live Eye-in-Hand RGB display stopped: {error}")
            self.close()


def add_camera_viewing_direction_debug_line(
    camera_world_position: Sequence[float],
    optical_axis_world: Sequence[float],
    client_id: int,
    previous_debug_item_id: int = -1,
) -> int:
    """Draw a visible yellow 0.5 m line from K along its fixed optical axis."""
    line_end = [
        position + CAMERA_VIEW_DEBUG_LINE_LENGTH * direction
        for position, direction in zip(camera_world_position, optical_axis_world)
    ]
    return p.addUserDebugLine(
        camera_world_position,
        line_end,
        lineColorRGB=[1.0, 1.0, 0.0],
        lineWidth=5,
        lifeTime=0,
        replaceItemUniqueId=previous_debug_item_id,
        physicsClientId=client_id,
    )


def add_camera_diagnostic_debug_lines(
    observation: CameraObservation,
    target_world_position: Sequence[float],
    client_id: int,
    previous_debug_item_ids: Sequence[int] = (-1, -1, -1),
) -> tuple[int, int, int]:
    """Draw the theoretical, rendered, and target directions from one eye.

    These lines are a camera-only diagnostic.  Their shared origin is the
    exact eye position supplied to ``getCameraImage``; this makes any
    difference between the rigid-camera optical axis and PyBullet's actual
    eye/target direction immediately visible in the GUI.
    """
    if len(previous_debug_item_ids) != 3:
        raise ValueError("Exactly three prior camera diagnostic IDs are required.")

    camera_origin = observation.render_parameters.eye_position

    def draw_direction(
        direction: Sequence[float],
        colour: Sequence[float],
        previous_debug_item_id: int,
    ) -> int:
        line_end = [
            position + CAMERA_VIEW_DEBUG_LINE_LENGTH * component
            for position, component in zip(camera_origin, direction)
        ]
        return p.addUserDebugLine(
            camera_origin,
            line_end,
            lineColorRGB=colour,
            lineWidth=5,
            lifeTime=0,
            replaceItemUniqueId=previous_debug_item_id,
            physicsClientId=client_id,
        )

    theoretical_axis_id = draw_direction(
        observation.viewing_direction,
        [1.0, 1.0, 0.0],  # yellow: rigid camera +Z_K optical axis
        previous_debug_item_ids[0],
    )
    actual_view_id = draw_direction(
        observation.actual_view_direction,
        [0.0, 0.4, 1.0],  # blue: target - eye actually passed to PyBullet
        previous_debug_item_ids[1],
    )
    target_vector_id = p.addUserDebugLine(
        camera_origin,
        target_world_position,
        lineColorRGB=[0.0, 1.0, 0.0],  # green: camera-origin to red target
        lineWidth=5,
        lifeTime=0,
        replaceItemUniqueId=previous_debug_item_ids[2],
        physicsClientId=client_id,
    )
    return theoretical_axis_id, actual_view_id, target_vector_id


def get_target_alignment_angle_degrees(
    camera_world_position: Sequence[float],
    optical_axis_world: Sequence[float],
    target_world_position: Sequence[float],
) -> tuple[list[float], float]:
    """Return target vector and the angle between it and K's optical axis."""
    vector_to_target = [
        target - camera
        for target, camera in zip(target_world_position, camera_world_position)
    ]
    target_distance = sqrt(sum(value * value for value in vector_to_target))
    optical_axis_length = sqrt(sum(value * value for value in optical_axis_world))
    if target_distance < 1e-9 or optical_axis_length < 1e-9:
        raise ValueError("Camera axis and target vector must have non-zero length.")
    dot_product = sum(
        optical * target
        for optical, target in zip(optical_axis_world, vector_to_target)
    )
    cosine = max(-1.0, min(1.0, dot_product / (optical_axis_length * target_distance)))
    return vector_to_target, degrees(acos(cosine))


def _angle_between_degrees(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> float:
    """Return the bounded angle between two non-zero vectors in degrees."""
    first_length = sqrt(sum(component * component for component in first_vector))
    second_length = sqrt(sum(component * component for component in second_vector))
    if first_length < 1e-9 or second_length < 1e-9:
        raise ValueError("Cannot calculate an angle for a zero-length vector.")
    cosine = sum(
        first * second for first, second in zip(first_vector, second_vector)
    ) / (first_length * second_length)
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def print_camera_render_diagnostics(
    observation: CameraObservation,
    target_world_position: Sequence[float],
) -> None:
    """Print the exact render inputs and geometry used for one RGB frame.

    The camera-frame coordinates below use the rigid K frame (+X image-right,
    +Y image-up, +Z optical).  They are intentionally reported separately
    from the diagnostic forced-look-at view, which is not a physical change
    to K or to the Panda.
    """
    render = observation.render_parameters
    target_coordinates = observation.target_render_camera_coordinates
    vertical_half_fov_tangent = tan(radians(CAMERA_FOV_Y_DEGREES) / 2.0)
    horizontal_half_fov_degrees = degrees(
        atan((CAMERA_IMAGE_WIDTH / CAMERA_IMAGE_HEIGHT) * vertical_half_fov_tangent)
    )
    horizontal_half_fov_tangent = tan(radians(horizontal_half_fov_degrees))
    target_in_front = target_coordinates[2] > 0.0
    target_in_fov = (
        target_in_front
        and abs(target_coordinates[0]) <= target_coordinates[2] * horizontal_half_fov_tangent
        and abs(target_coordinates[1]) <= target_coordinates[2] * vertical_half_fov_tangent
    )
    target_in_clipping_range = (
        CAMERA_NEAR_PLANE <= target_coordinates[2] <= CAMERA_FAR_PLANE
    )

    print("\ngetCameraImage render diagnostics:")
    print("  camera eye position:", [round(value, 6) for value in render.eye_position])
    print(
        "  camera target position:",
        [round(value, 6) for value in render.target_position],
    )
    print("  camera up vector:", [round(value, 6) for value in render.up_vector])
    print("  view matrix:", [round(value, 6) for value in render.view_matrix])
    print("  projection matrix:", [round(value, 6) for value in render.projection_matrix])
    print(
        "  theoretical optical axis (+Z_K, world):",
        [round(value, 6) for value in observation.viewing_direction],
    )
    print(
        "  actual_view_direction (normalize(target - eye)):",
        [round(value, 6) for value in observation.actual_view_direction],
    )
    print(
        "  vector_to_red_target:",
        [round(value, 6) for value in observation.vector_to_target],
    )
    print(
        "  angle(theoretical_axis, actual_view_direction): "
        f"{_angle_between_degrees(observation.viewing_direction, observation.actual_view_direction):.4f} deg"
    )
    print(
        "  angle(actual_view_direction, target_vector): "
        f"{_angle_between_degrees(observation.actual_view_direction, observation.vector_to_target):.4f} deg"
    )
    print(
        "  angle(theoretical_axis, target_vector): "
        f"{_angle_between_degrees(observation.viewing_direction, observation.vector_to_target):.4f} deg"
    )
    print(
        "  red target in rigid camera frame K [x_c, y_c, z_c]:",
        [round(value, 6) for value in observation.target_camera_coordinates],
    )
    print(
        "  red target in actual getCameraImage frame [x_c, y_c, z_c]:",
        [round(value, 6) for value in target_coordinates],
    )
    print(f"  target in front of K: {target_in_front}")
    print(
        "  target in K FOV "
        f"(vertical half-FOV {CAMERA_FOV_Y_DEGREES / 2.0:.1f} deg, "
        f"horizontal half-FOV {horizontal_half_fov_degrees:.1f} deg): {target_in_fov}"
    )
    print(
        "  target in clipping range "
        f"[{CAMERA_NEAR_PLANE:.2f}, {CAMERA_FAR_PLANE:.1f}] m: {target_in_clipping_range}"
    )


def _capture_with_render_parameters(
    camera_reference_world_position: Sequence[float],
    camera_reference_world_orientation: Sequence[float],
    target_world_position: Sequence[float],
    target_body_id: int,
    client_id: int,
    render_parameters: CameraRenderParameters,
    image_path: Path,
) -> CameraObservation:
    """Render one auditable frame using already-defined camera parameters."""
    (
        camera_world_position,
        camera_world_orientation,
        camera_x_axis_world,
        camera_y_axis_world,
        optical_axis_world,
    ) = get_rigid_camera_pose(
        camera_reference_world_position,
        camera_reference_world_orientation,
    )
    (
        image_width,
        image_height,
        rgba_buffer,
        segmentation_buffer,
    ) = _render_with_parameters(
        render_parameters,
        client_id,
    )
    actual_view_direction = _normalize(
        [
            target - eye
            for target, eye in zip(
                render_parameters.target_position,
                render_parameters.eye_position,
            )
        ]
    )
    vector_to_target = [
        target - camera
        for target, camera in zip(target_world_position, camera_world_position)
    ]
    target_camera_coordinates = (
        sum(component * axis for component, axis in zip(vector_to_target, camera_x_axis_world)),
        sum(component * axis for component, axis in zip(vector_to_target, camera_y_axis_world)),
        sum(component * axis for component, axis in zip(vector_to_target, optical_axis_world)),
    )
    # ``computeViewMatrix`` defines an orthonormal image basis from its exact
    # target and up vectors.  Diagnose its basis as well as the rigid K frame.
    actual_right_axis = _normalize(
        _cross(actual_view_direction, render_parameters.up_vector)
    )
    actual_up_axis = _normalize(_cross(actual_right_axis, actual_view_direction))
    render_target_vector = [
        target - eye
        for target, eye in zip(target_world_position, render_parameters.eye_position)
    ]
    target_render_camera_coordinates = (
        sum(component * axis for component, axis in zip(render_target_vector, actual_right_axis)),
        sum(component * axis for component, axis in zip(render_target_vector, actual_up_axis)),
        sum(
            component * axis
            for component, axis in zip(render_target_vector, actual_view_direction)
        ),
    )
    return CameraObservation(
        camera_reference_world_position=tuple(camera_reference_world_position),
        camera_reference_world_orientation=tuple(camera_reference_world_orientation),
        camera_world_position=tuple(camera_world_position),
        camera_world_orientation=tuple(camera_world_orientation),
        camera_x_axis_world=tuple(camera_x_axis_world),
        camera_y_axis_world=tuple(camera_y_axis_world),
        viewing_direction=tuple(optical_axis_world),
        actual_view_direction=tuple(actual_view_direction),
        vector_to_target=tuple(vector_to_target),
        target_camera_coordinates=target_camera_coordinates,
        target_render_camera_coordinates=target_render_camera_coordinates,
        render_parameters=render_parameters,
        image_width=image_width,
        image_height=image_height,
        target_visible_pixel_count=_count_visible_object_pixels(
            segmentation_buffer,
            target_body_id,
        ),
        image_path=_save_rgb_as_bmp(
            image_width,
            image_height,
            rgba_buffer,
            image_path,
        ),
    )


def capture_eye_in_hand_rgb(
    camera_reference_world_position: Sequence[float],
    camera_reference_world_orientation: Sequence[float],
    target_world_position: Sequence[float],
    target_body_id: int,
    client_id: int,
    image_path: Path = CAMERA_RGB_OUTPUT_PATH,
) -> CameraObservation:
    """Capture the normal rigid-mount view with no target-facing rotation."""
    (
        camera_world_position,
        camera_world_orientation,
        _,
        camera_y_axis_world,
        optical_axis_world,
    ) = get_rigid_camera_pose(
        camera_reference_world_position,
        camera_reference_world_orientation,
    )
    del camera_world_orientation
    render_parameters = build_camera_render_parameters(
        camera_world_position,
        [
            position + direction
            for position, direction in zip(camera_world_position, optical_axis_world)
        ],
        camera_y_axis_world,
    )
    return _capture_with_render_parameters(
        camera_reference_world_position,
        camera_reference_world_orientation,
        target_world_position,
        target_body_id,
        client_id,
        render_parameters,
        image_path,
    )


def capture_forced_look_at_rgb(
    camera_reference_world_position: Sequence[float],
    camera_reference_world_orientation: Sequence[float],
    target_world_position: Sequence[float],
    target_body_id: int,
    client_id: int,
    image_path: Path = CAMERA_FORCED_LOOK_AT_OUTPUT_PATH,
) -> CameraObservation:
    """Render a diagnostic-only frame whose getCameraImage target is the red ball."""
    (
        camera_world_position,
        camera_world_orientation,
        _,
        camera_y_axis_world,
        _,
    ) = get_rigid_camera_pose(
        camera_reference_world_position,
        camera_reference_world_orientation,
    )
    del camera_world_orientation
    render_parameters = build_camera_render_parameters(
        camera_world_position,
        target_world_position,
        camera_y_axis_world,
    )
    return _capture_with_render_parameters(
        camera_reference_world_position,
        camera_reference_world_orientation,
        target_world_position,
        target_body_id,
        client_id,
        render_parameters,
        image_path,
    )


def print_camera_coordinate_definition() -> None:
    """Print the fixed C-to-camera mounting convention once per application run."""
    print("\nEye-in-Hand virtual camera frame K:")
    print("  origin: physical Camera optical center C (T_C_K = Identity)")
    print("  +X_C: image-right axis")
    print("  +Y_C: image-up axis")
    print("  +Z_C: optical/viewing axis")
    print("  PyBullet view: eye = origin(C), target = origin(C) + +Z_C, up = +Y_C")


def print_camera_observation(
    observation: CameraObservation,
    target_world_position: Sequence[float],
) -> None:
    """Print camera state and rendered-target visibility without image detection."""
    print("\nEye-in-Hand RGB observation:")
    print(
        "Camera Reference C world position:",
        [round(value, 4) for value in observation.camera_reference_world_position],
    )
    print(
        "Camera Reference C world orientation xyzw:",
        [round(value, 4) for value in observation.camera_reference_world_orientation],
    )
    print(
        "Rigid camera K world position:",
        [round(value, 4) for value in observation.camera_world_position],
    )
    print(
        "Rigid camera K world orientation xyzw:",
        [round(value, 4) for value in observation.camera_world_orientation],
    )
    print(
        "Camera +X_K world direction:",
        [round(value, 4) for value in observation.camera_x_axis_world],
    )
    print(
        "Camera +Y_K world direction:",
        [round(value, 4) for value in observation.camera_y_axis_world],
    )
    print(
        "Camera +Z_K optical/viewing direction:",
        [round(value, 4) for value in observation.viewing_direction],
    )
    print("Target world position:", [round(value, 4) for value in target_world_position])
    print(f"Camera image width/height: {observation.image_width} x {observation.image_height}")
    print(
        "Red target visible pixels (PyBullet segmentation, not HSV): "
        f"{observation.target_visible_pixel_count}"
    )
    print(f"Saved RGB image: {observation.image_path}")


def print_camera_target_alignment(
    observation: CameraObservation,
    target_world_position: Sequence[float],
) -> float:
    """Print the frame/target geometry used to diagnose the RGB field of view."""
    vector_to_target, angle_degrees = get_target_alignment_angle_degrees(
        observation.camera_world_position,
        observation.viewing_direction,
        target_world_position,
    )
    print("Camera world position:", [round(value, 4) for value in observation.camera_world_position])
    print(
        "Camera optical axis in world coordinates:",
        [round(value, 4) for value in observation.viewing_direction],
    )
    print("Red target world position:", [round(value, 4) for value in target_world_position])
    print("vector_to_target:", [round(value, 4) for value in vector_to_target])
    print(f"Optical-axis / target-vector angle: {angle_degrees:.2f} deg")
    return angle_degrees
