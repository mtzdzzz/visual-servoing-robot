"""Stage 21 Demo 2: frozen Stage 16 manual multi-target integration."""

from __future__ import annotations

import time
import pybullet as p

import robotics_core as sim
import visual_servo_runtime as runtime
import predictive_measurement as predictive
import target_selection_runtime as selection
from camera_observation import EyeInHandRgbDisplay
from demo_config import DEMO_MULTITARGET_TITLE, DEFAULT_SELECTED_TARGET_ID
from multi_target_detector import MultiTargetDetector
from target_manager import TargetManager


class _NullDisplay:
    is_open = True
    def show(self, *_args, **_kwargs):
        return None


def run_demo_multitarget(*, gui: bool = True, realtime: bool = True, max_frames: int | None = None) -> dict[str, object]:
    """Only 1/2/3 can switch the selected target; production has no timeout."""
    client_id = p.connect(p.GUI if gui else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Unable to connect to PyBullet for Demo 2.")
    display = EyeInHandRgbDisplay(DEMO_MULTITARGET_TITLE) if gui else _NullDisplay()
    manager = TargetManager(
        selection.PREDICTION_ALPHA, selection.PREDICTION_HORIZON_S,
        initial_target_id=DEFAULT_SELECTED_TARGET_ID,
    )
    detector = MultiTargetDetector()
    frames = step = 0
    source = "CURRENT_SWITCH_FALLBACK"
    raw_error = None
    available: list[str] = []
    context = None
    try:
        context, _ = selection.create_multitarget_ready_context(
            client_id, display, detector, manager, DEMO_MULTITARGET_TITLE,
            selection.DEFAULT_TARGET_SPECS, selection.DEFAULT_TARGET_WORLD_POSITIONS,
            realtime=realtime, perform_warmup=False,
        )
        print(f"{DEMO_MULTITARGET_TITLE}: ACTIVE")
        print("Selected: RED | [1] RED [2] GREEN [3] BLUE [Q/ESC] Quit")
        while p.isConnected(client_id) and display.is_open:
            timestamp = step * sim.TIME_STEP
            if gui:
                event, quit_requested = selection.manual_keyboard_input(manager, timestamp, client_id)
                if quit_requested:
                    break
                if event is not None:
                    print(f"Manual selection: {event.previous_class} -> {event.selected_class}")
            runtime.step_physics(context, client_id)
            if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
                camera_position, camera_orientation = sim.get_camera_optical_center_pose(context.robot_id, client_id)
                frame = sim.render_live_eye_in_hand_rgb_frame(camera_position, camera_orientation, client_id)
                detections = detector.detect(frame.rgba_buffer, frame.image_width, frame.image_height)
                available = sorted(detection.class_name for detection in detections if detection.valid)
                estimates = manager.update_detections(timestamp, detections)
                raw = selection.controller_detection_from_selected(
                    manager.selected_detection(), (frame.image_width // 2, frame.image_height // 2),
                    frame.rgba_buffer,
                )
                estimate = estimates[manager.selected_target_id]
                source = manager.controller_source_for_selected(estimate)
                measurement = raw
                if source == "PREDICTED":
                    measurement, source, _ = predictive.predicted_measurement(raw, estimate)
                runtime.control_latest_measurement(context, measurement, client_id)
                raw_error = raw.pixel_error if raw.detected else None
                display.show(frame, selection.annotated_overlay(
                    frame.rgba_buffer, frame.image_width, frame.image_height, detections,
                    manager, DEMO_MULTITARGET_TITLE, "MANUAL / TRACKING", source,
                    context.servo.state if raw.detected else "TARGET LOST / HOLD",
                ))
                context.debug_text_id = sim.update_motion_debug_text(
                    f"{DEMO_MULTITARGET_TITLE}\nSelected: {manager.selected_class}\n"
                    f"Raw error: {raw_error if raw.detected else 'N/A'} | Controller: {source}\n"
                    "[1] RED [2] GREEN [3] BLUE [Q/ESC] Quit",
                    context.debug_text_id, client_id,
                )
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
            step += 1
            if realtime:
                time.sleep(sim.TIME_STEP)
        return {
            "frames": frames, "selected": manager.selected_class,
            "selection_source": manager.last_selection_source,
            "controller_source": source, "raw_error": raw_error,
            "available_targets": available,
            "locked_joint_max_deviation": context.max_locked_deviation if context else None,
            "automatic_switching": False,
        }
    except KeyboardInterrupt:
        return {"frames": frames, "selected": manager.selected_class,
                "automatic_switching": False}
    except p.error:
        if p.isConnected(client_id):
            raise
        return {"frames": frames, "selected": manager.selected_class,
                "automatic_switching": False}
    finally:
        if isinstance(display, EyeInHandRgbDisplay):
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


