"""Stage 13 online A/B evaluation: current-centroid vs predicted-centroid servo.

The only controller-input difference between the two modes is the RGB
centroid supplied to the frozen Stage 9 latest-observation command function.
All score metrics below are calculated from the raw OpenCV centroid.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from math import sqrt
from pathlib import Path
import time
from typing import Literal, Sequence

import pybullet as p

import simulation as sim
import stage10_evaluation as stage10
from camera_observation import EyeInHandRgbDisplay, RedTargetDetection
from target_motion_estimator import MotionEstimate, TargetMotionEstimator


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / "logs"
STAGE13_DURATION_SECONDS = sim.STAGE9_DURATION_SECONDS
PREDICTION_ALPHA = 0.20
PREDICTION_HORIZONS_S = (0.15, 0.20, 0.30)
# Fixed before every A/B trial. It is a prediction safety limit, not a
# controller gain and is never adapted from a trial result.
PREDICTION_OFFSET_CLAMP_PIXELS = 12.0
TRAJECTORIES: tuple[tuple[str, float], ...] = (("MEDIUM", 0.10), ("FAST", 0.20))

ControllerMode = Literal["BASELINE", "PREDICTIVE"]


def _stage13_csv_fields() -> tuple[str, ...]:
    return (
        "time_s", "speed_level", "target_frequency_hz", "controller_mode",
        "prediction_alpha", "prediction_horizon_s", "target_detected",
        "u_raw", "v_raw", "center_x", "center_y", "raw_ex", "raw_ey",
        "error_norm_raw", "u_pred", "v_pred", "u_dot_raw", "v_dot_raw",
        "u_dot_filtered", "v_dot_filtered", "estimator_valid",
        "controller_measurement_source", "controller_u", "controller_v",
        "servo_ex", "servo_ey", "prediction_clamped", "servo_state",
        "servo_decision", "ik_command_issued", "nonzero_correction_issued",
        "delta_x_m", "delta_y_m",
    )


def _metric(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_tracking_error": None, "rmse": None, "p95_error": None, "max_error": None}
    ordered = sorted(values)
    percentile_index = (len(ordered) - 1) * 0.95
    lower = int(percentile_index)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (percentile_index - lower) * (ordered[upper] - ordered[lower])
    return {
        "mean_tracking_error": sum(values) / len(values),
        "rmse": sqrt(sum(value * value for value in values) / len(values)),
        "p95_error": p95,
        "max_error": max(values),
    }


def _capture_raw_detection(
    context: stage10.ReadyContext,
    client_id: int,
    camera_display: EyeInHandRgbDisplay,
    overlay_text: str,
) -> RedTargetDetection:
    """Obtain the current Eye-in-Hand RGB/OpenCV measurement only."""

    camera_position, camera_orientation = sim.get_camera_optical_center_pose(
        context.robot_id, client_id
    )
    frame = sim.render_live_eye_in_hand_rgb_frame(camera_position, camera_orientation, client_id)
    raw_detection = sim.detect_red_target_from_live_rgb(frame)
    camera_display.show(frame, raw_detection.annotated_rgba_buffer)
    return raw_detection


def _predicted_measurement(
    raw_detection: RedTargetDetection,
    estimate: MotionEstimate | None,
) -> tuple[RedTargetDetection, str, bool]:
    """Create the controller measurement from vision-only prediction.

    The returned detection is the same type used by the frozen controller.
    It contains no object id or world-coordinate information.  Before the
    estimator has two valid frames, the raw current detection is used.
    """

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
    controller_centroid = (u_raw + limited_u, v_raw + limited_v)
    center_x, center_y = raw_detection.image_center
    return (
        RedTargetDetection(
            detected=True,
            bounding_box=raw_detection.bounding_box,
            contour_area=raw_detection.contour_area,
            centroid=controller_centroid,
            image_center=raw_detection.image_center,
            pixel_error=(controller_centroid[0] - center_x, controller_centroid[1] - center_y),
            annotated_rgba_buffer=raw_detection.annotated_rgba_buffer,
        ),
        "PREDICTED_CLAMPED" if clamped else "PREDICTED",
        clamped,
    )


def _log_row(
    simulation_time_s: float,
    speed_level: str,
    frequency_hz: float,
    mode: ControllerMode,
    horizon_s: float | None,
    raw_detection: RedTargetDetection,
    estimate: MotionEstimate | None,
    controller_measurement: RedTargetDetection,
    measurement_source: str,
    prediction_clamped: bool,
    context: stage10.ReadyContext,
    ik_command_issued: bool,
    nonzero_correction_issued: bool,
) -> dict[str, object]:
    """Write raw and controller measurements separately for a fair audit."""

    estimate_values = asdict(estimate) if estimate is not None else {}
    if raw_detection.detected:
        assert raw_detection.centroid is not None and raw_detection.pixel_error is not None
        u_raw, v_raw = raw_detection.centroid
        center_x, center_y = raw_detection.image_center
        raw_ex, raw_ey = raw_detection.pixel_error
        raw_norm: float | str = sqrt(raw_ex * raw_ex + raw_ey * raw_ey)
    else:
        u_raw = v_raw = center_x = center_y = raw_ex = raw_ey = raw_norm = ""
    if controller_measurement.detected:
        assert controller_measurement.centroid is not None and controller_measurement.pixel_error is not None
        controller_u, controller_v = controller_measurement.centroid
        servo_ex, servo_ey = controller_measurement.pixel_error
    else:
        controller_u = controller_v = servo_ex = servo_ey = ""
    return {
        "time_s": f"{simulation_time_s:.6f}",
        "speed_level": speed_level,
        "target_frequency_hz": f"{frequency_hz:.6f}",
        "controller_mode": mode,
        "prediction_alpha": PREDICTION_ALPHA if mode == "PREDICTIVE" else "",
        "prediction_horizon_s": horizon_s if horizon_s is not None else "",
        "target_detected": int(raw_detection.detected),
        "u_raw": u_raw,
        "v_raw": v_raw,
        "center_x": center_x,
        "center_y": center_y,
        "raw_ex": raw_ex,
        "raw_ey": raw_ey,
        "error_norm_raw": raw_norm,
        "u_pred": estimate_values.get("u_pred"),
        "v_pred": estimate_values.get("v_pred"),
        "u_dot_raw": estimate_values.get("u_dot_raw"),
        "v_dot_raw": estimate_values.get("v_dot_raw"),
        "u_dot_filtered": estimate_values.get("u_dot_filtered"),
        "v_dot_filtered": estimate_values.get("v_dot_filtered"),
        "estimator_valid": int(estimate.estimator_valid) if estimate is not None else 0,
        "controller_measurement_source": measurement_source,
        "controller_u": controller_u,
        "controller_v": controller_v,
        "servo_ex": servo_ex,
        "servo_ey": servo_ey,
        "prediction_clamped": int(prediction_clamped),
        "servo_state": context.servo.state,
        "servo_decision": 1,
        "ik_command_issued": int(ik_command_issued),
        "nonzero_correction_issued": int(nonzero_correction_issued),
        "delta_x_m": f"{context.servo.last_delta_x:.9f}",
        "delta_y_m": f"{context.servo.last_delta_y:.9f}",
    }


def _trial_filename(mode: ControllerMode, horizon_s: float | None, speed_level: str) -> Path:
    suffix = speed_level.lower()
    if mode == "BASELINE":
        return LOG_DIRECTORY / f"stage13_baseline_{suffix}.csv"
    assert horizon_s is not None
    return LOG_DIRECTORY / f"stage13_predictive_tau{int(round(horizon_s * 1000)):03d}_{suffix}.csv"


def _run_trial(
    mode: ControllerMode,
    horizon_s: float | None,
    speed_level: str,
    frequency_hz: float,
    client_id: int,
    camera_display: EyeInHandRgbDisplay,
) -> dict[str, object]:
    """Run one READY-gated A/B episode using the frozen Stage 9 mechanics."""

    label = f"STAGE 13 {mode} {speed_level}" + (f" tau={horizon_s:.2f}" if horizon_s else "")
    # Warm-up always uses the existing raw-baseline controller, making the
    # dynamic t=0 robot state comparable for all A/B trials.
    context = stage10._create_ready_context(client_id, camera_display, label)
    log_path = _trial_filename(mode, horizon_s, speed_level)
    estimator = (
        TargetMotionEstimator(PREDICTION_ALPHA, float(horizon_s))
        if mode == "PREDICTIVE"
        else None
    )
    total_steps = max(1, round(STAGE13_DURATION_SECONDS / sim.TIME_STEP))
    target_orientation = (0.0, 0.0, 0.0, 1.0)
    raw_errors: list[float] = []
    total_frames = detected_frames = servo_decisions = ik_commands = nonzero_corrections = 0
    clamp_count = target_lost_count = 0
    was_detected = True

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=_stage13_csv_fields())
        writer.writeheader()
        for step in range(total_steps):
            simulation_time_s = step * sim.TIME_STEP
            # Schedule-only ground truth. This value never enters the
            # estimator, controller measurement, or IK command.
            target_world_position = sim._stage9_target_world_position(simulation_time_s, frequency_hz)
            p.resetBasePositionAndOrientation(
                context.target_body_id, target_world_position, target_orientation, physicsClientId=client_id
            )
            p.resetBaseVelocity(
                context.target_body_id,
                linearVelocity=sim._stage9_target_world_velocity(simulation_time_s, frequency_hz),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client_id,
            )
            stage10._step_physics(context, client_id)
            if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                sim.update_camera_reference_axes(
                    context.robot_id, client_id, context.camera_axis_debug_item_ids
                )
                raw_detection = _capture_raw_detection(
                    context,
                    client_id,
                    camera_display,
                    f"STAGE 13 {mode} | {speed_level} | t={simulation_time_s:.1f}s",
                )
                total_frames += 1
                estimate: MotionEstimate | None = None
                if raw_detection.detected:
                    detected_frames += 1
                    assert raw_detection.centroid is not None and raw_detection.pixel_error is not None
                    raw_ex, raw_ey = raw_detection.pixel_error
                    raw_errors.append(sqrt(raw_ex * raw_ex + raw_ey * raw_ey))
                    if estimator is not None:
                        estimate = estimator.update(simulation_time_s, *raw_detection.centroid)
                elif estimator is not None:
                    estimator.target_lost()

                if mode == "PREDICTIVE":
                    controller_measurement, source, prediction_clamped = _predicted_measurement(
                        raw_detection, estimate
                    )
                else:
                    controller_measurement, source, prediction_clamped = raw_detection, "CURRENT", False

                if prediction_clamped:
                    clamp_count += 1
                if was_detected and not raw_detection.detected:
                    target_lost_count += 1
                was_detected = raw_detection.detected
                servo_decisions += 1
                ik_command_issued, nonzero_correction_issued = stage10._control_latest_measurement(
                    context, controller_measurement, client_id
                )
                ik_commands += int(ik_command_issued)
                nonzero_corrections += int(nonzero_correction_issued)
                writer.writerow(
                    _log_row(
                        simulation_time_s, speed_level, frequency_hz, mode, horizon_s,
                        raw_detection, estimate, controller_measurement, source,
                        prediction_clamped, context, ik_command_issued, nonzero_correction_issued,
                    )
                )
                context.debug_text_id = sim.update_motion_debug_text(
                    "STAGE 13 - PREDICTIVE VISUAL SERVO A/B\n"
                    f"Mode: {mode} | Target: {speed_level} | t={simulation_time_s:.1f}/{STAGE13_DURATION_SECONDS:.1f}s\n"
                    f"Source: {source} | Clamp count: {clamp_count} | Servo: {context.servo.state}\n"
                    "Metrics use RAW OpenCV error only.",
                    context.debug_text_id,
                    client_id,
                )
            time.sleep(sim.TIME_STEP)

    metric = _metric(raw_errors)
    rate = lambda count: count / STAGE13_DURATION_SECONDS
    completed = (
        len(raw_errors) > 0
        and context.max_locked_deviation < sim.LOCKED_JOINT_DEVIATION_LIMIT
    )
    summary = {
        "speed_level": speed_level,
        "frequency_hz": frequency_hz,
        "controller_mode": mode,
        "prediction_alpha": PREDICTION_ALPHA if mode == "PREDICTIVE" else None,
        "prediction_horizon_s": horizon_s,
        "duration_s": STAGE13_DURATION_SECONDS,
        "warmup_duration_s": context.warmup_duration_s,
        "warmup_initial_error": context.warmup_initial_error,
        "ready_ex": context.ready_ex,
        "ready_ey": context.ready_ey,
        "frames": total_frames,
        "mean_tracking_error": metric["mean_tracking_error"],
        "rmse": metric["rmse"],
        "p95_error": metric["p95_error"],
        "max_error": metric["max_error"],
        "target_lost_count": target_lost_count,
        "detection_rate_percent": 100.0 * detected_frames / total_frames if total_frames else 0.0,
        "servo_decision_fps": rate(servo_decisions),
        "ik_command_fps": rate(ik_commands),
        "nonzero_correction_fps": rate(nonzero_corrections),
        "prediction_clamp_count": clamp_count,
        "locked_joint_max_deviation_rad": context.max_locked_deviation,
        "result": "COMPLETED" if completed else "FAIL",
        "log_path": str(log_path),
    }
    print(
        f"Stage 13 {mode} {speed_level}" + (f" tau={horizon_s:.2f}" if horizon_s else "")
        + f": raw RMSE={summary['rmse']:.3f}px, mean={summary['mean_tracking_error']:.3f}px, "
        + f"P95={summary['p95_error']:.3f}px, lost={target_lost_count}, clamps={clamp_count}, "
        + f"result={summary['result']}"
    )
    return summary


def _write_summary(summaries: Sequence[dict[str, object]]) -> Path:
    path = LOG_DIRECTORY / "stage13_summary.csv"
    fields = (
        "speed_level", "frequency_hz", "controller_mode", "prediction_alpha",
        "prediction_horizon_s", "duration_s", "warmup_duration_s", "warmup_initial_error",
        "ready_ex", "ready_ey", "frames", "mean_tracking_error", "rmse", "p95_error",
        "max_error", "target_lost_count", "detection_rate_percent", "servo_decision_fps",
        "ik_command_fps", "nonzero_correction_fps", "prediction_clamp_count",
        "locked_joint_max_deviation_rad", "result", "log_path",
    )
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
    return path


def run_stage13_predictive_visual_servo_evaluation() -> None:
    """Run baseline and three frozen-alpha predictive A/B trials at two speeds."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 13.")
    camera_display: EyeInHandRgbDisplay | None = None
    try:
        camera_display = EyeInHandRgbDisplay("Eye-in-Hand RGB - Stage 13 A/B")
        print("Stage 13: Predictive Visual Servoing A/B Evaluation")
        print("Frozen controller/camera/IK/joint baseline; only current vs predicted centroid differs.")
        print(
            f"Predictor: alpha={PREDICTION_ALPHA:.2f}; horizons={PREDICTION_HORIZONS_S}; "
            f"fixed per-axis prediction clamp=±{PREDICTION_OFFSET_CLAMP_PIXELS:.1f}px."
        )
        print("All final tracking metrics use raw OpenCV RGB centroid error.")
        summaries: list[dict[str, object]] = []
        for speed_level, frequency_hz in TRAJECTORIES:
            summaries.append(
                _run_trial("BASELINE", None, speed_level, frequency_hz, client_id, camera_display)
            )
            summaries.extend(
                _run_trial("PREDICTIVE", horizon_s, speed_level, frequency_hz, client_id, camera_display)
                for horizon_s in PREDICTION_HORIZONS_S
            )
        summary_path = _write_summary(summaries)
        completed = all(summary["result"] == "COMPLETED" for summary in summaries)
        print("\n===== Stage 13 Completion =====")
        print("Summary CSV:", summary_path)
        print("World-coordinate controller/predictor input: CONFIRMED ABSENT.")
        print("Stage 13:", "PASS" if completed else "FAIL")
    finally:
        if camera_display is not None:
            camera_display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)
