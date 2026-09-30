"""Stage 21 Demo 1: manual red-ball drag plus frozen predictive visual servo."""

from __future__ import annotations

import time
import pybullet as p
import pybullet_data

import robotics_core as sim
import visual_servo_runtime as runtime
import predictive_measurement as predictive
import config
from camera_observation import (
    EyeInHandRgbDisplay, ManualTargetDragController,
    create_manual_draggable_red_ground_target, detect_red_target_from_live_rgb,
)
from demo_config import DEMO_TRACKING_TITLE
from demo_ui import status_panel
from target_motion_estimator import TargetMotionEstimator


def _quit_requested(client_id: int) -> bool:
    events = p.getKeyboardEvents(physicsClientId=client_id)
    return any(events.get(key, 0) & p.KEY_WAS_TRIGGERED for key in (ord("q"), ord("Q"), 27))


def _create_context(client_id: int) -> tuple[runtime.ReadyContext, ManualTargetDragController]:
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, 0, physicsClientId=client_id)
    p.setTimeStep(sim.TIME_STEP, physicsClientId=client_id)
    sim.enable_pybullet_camera_debug_previews(client_id)
    if p.getConnectionInfo(client_id).get("connectionMethod") == p.GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=client_id)
    initial = sim.print_panda_arm_joint_configuration(robot_id, client_id)
    joints = sim.get_panda_arm_joint_indices(robot_id, client_id)
    names = sim.get_panda_arm_joint_names(robot_id, client_id)
    locked = [names[name] for name in sim.BASELINE_LOCKED_JOINT_NAMES]
    locked_positions = {joint: initial[joint] for joint in locked}
    active = [joint for joint in joints if joint not in locked]
    active_hold = sim.capture_active_joint_hold_targets(robot_id, active, client_id)
    target_id = create_manual_draggable_red_ground_target(client_id)
    axes: list[int] = []
    sim.update_camera_reference_axes(robot_id, client_id, axes)
    context = runtime.ReadyContext(
        robot_id=robot_id, target_body_id=target_id, arm_joint_indices=joints,
        active_joint_indices=active, locked_joint_indices=locked,
        locked_initial_positions=locked_positions,
        commanded_joint_targets=sim.build_hold_joint_targets(joints, locked_positions, active_hold),
        active_hold_targets=active_hold, servo=sim.TwoDimensionalVisualServo(),
        camera_axis_debug_item_ids=axes, debug_text_id=None, warmup_duration_s=0.0,
        warmup_initial_error=None, ready_ex=None, ready_ey=None, max_locked_deviation=0.0,
    )
    return context, ManualTargetDragController(target_id, client_id)


def run_demo_tracking(*, gui: bool = True, realtime: bool = True, max_frames: int | None = None) -> dict[str, object]:
    """Run indefinitely in production; ``max_frames`` is a smoke-test hook only."""
    client_id = p.connect(p.GUI if gui else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Unable to connect to PyBullet for Demo 1.")
    display = EyeInHandRgbDisplay(DEMO_TRACKING_TITLE) if gui else None
    drag: ManualTargetDragController | None = None
    frames = step = 0
    source = "CURRENT_FALLBACK"
    last_error = None
    initial_error = None
    try:
        context, drag = _create_context(client_id)
        estimator = TargetMotionEstimator(predictive.PREDICTION_ALPHA, config.PREDICTION_HORIZON_S)
        print(f"{DEMO_TRACKING_TITLE}: ACTIVE")
        print("Mouse: drag RED | Controller: RGB centroid/prediction only | Q/ESC: quit")
        while p.isConnected(client_id):
            if gui and _quit_requested(client_id):
                break
            if display is not None and not display.is_open:
                break
            if drag is not None and gui:
                drag.update()
            runtime.step_physics(context, client_id)
            if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                timestamp = step * sim.TIME_STEP
                sim.update_camera_reference_axes(context.robot_id, client_id, context.camera_axis_debug_item_ids)
                camera_position, camera_orientation = sim.get_camera_optical_center_pose(context.robot_id, client_id)
                frame = sim.render_live_eye_in_hand_rgb_frame(camera_position, camera_orientation, client_id)
                raw = detect_red_target_from_live_rgb(frame)
                estimate = None
                if raw.detected:
                    assert raw.centroid is not None
                    estimate = estimator.update(timestamp, *raw.centroid)
                else:
                    estimator.target_lost()
                measurement, source, _ = predictive.predicted_measurement(raw, estimate)
                runtime.control_latest_measurement(context, measurement, client_id)
                last_error = raw.pixel_error if raw.detected else None
                if initial_error is None and last_error is not None:
                    initial_error = last_error
                if raw.detected:
                    assert raw.centroid is not None
                    prediction = (
                        f"({estimate.u_pred:.1f}, {estimate.v_pred:.1f})"
                        if estimate is not None else "N/A"
                    )
                    target_text = str(raw.centroid)
                    error_text = f"ex={raw.pixel_error[0]} ey={raw.pixel_error[1]}"
                    tracking_state = context.servo.state
                else:
                    prediction = target_text = error_text = "N/A"
                    tracking_state = "TARGET LOST / HOLD"
                overlay = status_panel(raw.annotated_rgba_buffer, frame.image_width, frame.image_height, [
                    DEMO_TRACKING_TITLE,
                    f"Target: {target_text} | Raw error: {error_text}",
                    f"Prediction: {prediction} | Controller: {source}",
                    f"Tracking state: {tracking_state}",
                    f"Mouse: {drag.interaction_status if drag else 'DISABLED'} | [Q/ESC] Quit",
                ])
                if display is not None:
                    display.show(frame, overlay)
                context.debug_text_id = sim.update_motion_debug_text(
                    f"{DEMO_TRACKING_TITLE}\nController: {source}\nState: {tracking_state}",
                    context.debug_text_id, client_id,
                )
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
            step += 1
            if realtime:
                time.sleep(sim.TIME_STEP)
        return {
            "frames": frames, "controller_source": source,
            "initial_raw_error": initial_error, "raw_error": last_error,
            "locked_joint_max_deviation": context.max_locked_deviation,
            "world_coordinates_used_by_controller": False,
        }
    except KeyboardInterrupt:
        return {"frames": frames, "controller_source": source,
                "initial_raw_error": initial_error, "raw_error": last_error,
                "world_coordinates_used_by_controller": False}
    except p.error:
        if p.isConnected(client_id):
            raise
        return {"frames": frames, "controller_source": source,
                "initial_raw_error": initial_error, "raw_error": last_error,
                "world_coordinates_used_by_controller": False}
    finally:
        if drag is not None:
            drag.close()
        if display is not None:
            display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


