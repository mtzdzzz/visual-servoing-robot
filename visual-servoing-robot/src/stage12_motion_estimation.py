"""Stage 12 image-plane motion-estimation experiment.

Only the experiment scheduler moves the simulated red target.  The estimator
accepts timestamped OpenCV centroids only, while the frozen Stage 9 controller
continues to receive the current, raw ``RedTargetDetection`` on every RGB
frame. Predicted centroids are never supplied to IK or motor control.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from math import isfinite, sqrt
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import pybullet as p

import robotics_core as sim
import visual_servo_runtime as runtime
from camera_observation import (
    EyeInHandRgbDisplay,
    LiveCameraFrame,
    RedTargetDetection,
    add_servo_state_overlay,
)
from target_motion_estimator import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_PREDICTION_HORIZON_SECONDS,
    MotionEstimate,
    TargetMotionEstimator,
)


LOG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "logs" / "stage12_motion_estimation.csv"
STAGE12_DURATION_SECONDS = sim.STAGE9_DURATION_SECONDS
STAGE12_MEDIUM_FREQUENCY_HZ = 0.10
FUTURE_TIMESTAMP_TOLERANCE_SECONDS = 0.020


def _csv_fields() -> tuple[str, ...]:
    return (
        "time_s", "target_detected", "u_raw", "v_raw", "ex_raw", "ey_raw",
        "estimator_valid", "estimator_status", "u_dot_raw", "v_dot_raw",
        "u_dot_filtered", "v_dot_filtered", "u_pred", "v_pred",
        "prediction_horizon_s", "servo_ex", "servo_ey", "servo_state",
        "ik_command_issued", "nonzero_correction_issued", "prediction_used_for_control",
        # Filled only by the offline, measurement-only future comparison.
        "future_timestamp_s", "future_u_raw", "future_v_raw",
        "baseline_future_error", "motion_prediction_future_error",
    )


def _stage12_overlay(
    frame: LiveCameraFrame,
    detection: RedTargetDetection,
    estimate: MotionEstimate | None,
    servo_state: str,
) -> bytes:
    """Draw current/predicted image points without altering RGB detection."""
    base_rgba = add_servo_state_overlay(
        frame,
        detection.annotated_rgba_buffer,
        f"STAGE 12 | {servo_state}",
    )
    rgba_image = np.frombuffer(base_rgba, dtype=np.uint8).reshape(
        (frame.image_height, frame.image_width, 4)
    )
    annotated_bgr = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    cv2.rectangle(annotated_bgr, (8, 36), (455, 98), (25, 25, 25), thickness=-1)

    if detection.detected:
        assert detection.centroid is not None
        current = tuple(int(round(value)) for value in detection.centroid)
        cv2.circle(annotated_bgr, current, 11, (0, 255, 255), thickness=2)
        if estimate is not None:
            predicted = (
                int(round(max(0, min(frame.image_width - 1, estimate.u_pred)))),
                int(round(max(0, min(frame.image_height - 1, estimate.v_pred)))),
            )
            cv2.arrowedLine(annotated_bgr, current, predicted, (255, 255, 0), 2, tipLength=0.22)
            cv2.drawMarker(
                annotated_bgr, predicted, (255, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2,
            )
            label = "VALID" if estimate.estimator_valid else "INITIALIZING"
            cv2.putText(
                annotated_bgr,
                f"Velocity: du={estimate.u_dot_filtered:+.1f} dv={estimate.v_dot_filtered:+.1f} px/s [{label}]",
                (16, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated_bgr,
                f"Prediction: ({estimate.u_pred:.1f}, {estimate.v_pred:.1f}), tau={estimate.prediction_horizon_s:.2f}s",
                (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )
    else:
        cv2.putText(
            annotated_bgr, "Motion estimator: INVALID (TARGET LOST; reset)",
            (16, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 255), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGBA).tobytes()


def _capture_stage12_rgb(
    context: runtime.ReadyContext,
    simulation_time_s: float,
    estimator: TargetMotionEstimator,
    camera_display: EyeInHandRgbDisplay,
    client_id: int,
) -> tuple[RedTargetDetection, MotionEstimate | None]:
    """Capture RGB, detect red, and update estimator from only timestamp/u/v."""
    camera_position, camera_orientation = sim.get_camera_optical_center_pose(
        context.robot_id,
        client_id,
    )
    frame = sim.render_live_eye_in_hand_rgb_frame(
        camera_position, camera_orientation, client_id
    )
    raw_detection = sim.detect_red_target_from_live_rgb(frame)
    if raw_detection.detected:
        assert raw_detection.centroid is not None
        estimate = estimator.update(simulation_time_s, *raw_detection.centroid)
    else:
        estimator.target_lost()
        estimate = None
    camera_display.show(
        frame,
        _stage12_overlay(frame, raw_detection, estimate, context.servo.state),
    )
    return raw_detection, estimate


def _log_row(
    simulation_time_s: float,
    raw_detection: RedTargetDetection,
    estimate: MotionEstimate | None,
    context: runtime.ReadyContext,
    ik_command_issued: bool,
    nonzero_correction_issued: bool,
) -> dict[str, object]:
    """Create one current-observation record; no target world data is logged."""
    if raw_detection.detected:
        assert raw_detection.centroid is not None and raw_detection.pixel_error is not None
        u_raw, v_raw = raw_detection.centroid
        ex_raw, ey_raw = raw_detection.pixel_error
    else:
        u_raw = v_raw = ex_raw = ey_raw = None
    estimate_values = asdict(estimate) if estimate is not None else {}
    return {
        "time_s": f"{simulation_time_s:.6f}",
        "target_detected": bool(raw_detection.detected),
        "u_raw": u_raw,
        "v_raw": v_raw,
        "ex_raw": ex_raw,
        "ey_raw": ey_raw,
        "estimator_valid": bool(estimate.estimator_valid) if estimate is not None else False,
        "estimator_status": estimate.status if estimate is not None else "TARGET_LOST_RESET",
        "u_dot_raw": estimate_values.get("u_dot_raw"),
        "v_dot_raw": estimate_values.get("v_dot_raw"),
        "u_dot_filtered": estimate_values.get("u_dot_filtered"),
        "v_dot_filtered": estimate_values.get("v_dot_filtered"),
        "u_pred": estimate_values.get("u_pred"),
        "v_pred": estimate_values.get("v_pred"),
        "prediction_horizon_s": estimate_values.get(
            "prediction_horizon_s", DEFAULT_PREDICTION_HORIZON_SECONDS
        ),
        # Controller uses raw_detection below, so servo errors must match raw errors.
        "servo_ex": ex_raw,
        "servo_ey": ey_raw,
        "servo_state": context.servo.state,
        "ik_command_issued": bool(ik_command_issued),
        "nonzero_correction_issued": bool(nonzero_correction_issued),
        "prediction_used_for_control": False,
        "future_timestamp_s": None,
        "future_u_raw": None,
        "future_v_raw": None,
        "baseline_future_error": None,
        "motion_prediction_future_error": None,
    }


def _offline_future_evaluation(log_path: Path) -> dict[str, float | int | None]:
    """Compare current/predicted centroids to later *measured* RGB centroids.

    This intentionally finds the closest valid timestamp to ``t + tau``; it
    never assumes a fixed FPS and never evaluates against target world pose.
    """
    data = pd.read_csv(log_path)
    numeric_columns = (
        "time_s", "target_detected", "u_raw", "v_raw", "estimator_valid",
        "u_pred", "v_pred", "prediction_horizon_s",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid_measurements = data.loc[
        data["target_detected"].eq(1)
        & data["u_raw"].notna()
        & data["v_raw"].notna()
        & data["time_s"].notna()
    ].copy()
    future_time: list[float | None] = [None] * len(data)
    future_u: list[float | None] = [None] * len(data)
    future_v: list[float | None] = [None] * len(data)
    baseline_error: list[float | None] = [None] * len(data)
    prediction_error: list[float | None] = [None] * len(data)
    for index, row in data.iterrows():
        if not (
            row["target_detected"] == 1
            and row["estimator_valid"] == 1
            and isfinite(float(row["u_pred"]))
            and isfinite(float(row["v_pred"]))
        ):
            continue
        desired_time = float(row["time_s"] + row["prediction_horizon_s"])
        candidates = valid_measurements.loc[valid_measurements["time_s"] > float(row["time_s"])]
        if candidates.empty:
            continue
        nearest_index = (candidates["time_s"] - desired_time).abs().idxmin()
        candidate = candidates.loc[nearest_index]
        if abs(float(candidate["time_s"]) - desired_time) > FUTURE_TIMESTAMP_TOLERANCE_SECONDS:
            continue
        future_time[index] = float(candidate["time_s"])
        future_u[index] = float(candidate["u_raw"])
        future_v[index] = float(candidate["v_raw"])
        baseline_error[index] = sqrt(
            (float(row["u_raw"]) - float(candidate["u_raw"])) ** 2
            + (float(row["v_raw"]) - float(candidate["v_raw"])) ** 2
        )
        prediction_error[index] = sqrt(
            (float(row["u_pred"]) - float(candidate["u_raw"])) ** 2
            + (float(row["v_pred"]) - float(candidate["v_raw"])) ** 2
        )
    data["future_timestamp_s"] = future_time
    data["future_u_raw"] = future_u
    data["future_v_raw"] = future_v
    data["baseline_future_error"] = baseline_error
    data["motion_prediction_future_error"] = prediction_error
    data.to_csv(log_path, index=False)
    baseline = pd.Series(baseline_error, dtype=float).dropna().to_numpy()
    prediction = pd.Series(prediction_error, dtype=float).dropna().to_numpy()
    if not len(baseline) or not len(prediction):
        return {
            "evaluation_samples": 0,
            "baseline_mean_error": None, "baseline_rmse": None,
            "prediction_mean_error": None, "prediction_rmse": None,
            "improvement_percent": None,
        }
    baseline_mean = float(np.mean(baseline))
    prediction_mean = float(np.mean(prediction))
    return {
        "evaluation_samples": int(len(baseline)),
        "baseline_mean_error": baseline_mean,
        "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline)))),
        "prediction_mean_error": prediction_mean,
        "prediction_rmse": float(np.sqrt(np.mean(np.square(prediction)))),
        "improvement_percent": 100.0 * (baseline_mean - prediction_mean) / baseline_mean if baseline_mean else None,
    }


def _format(value: float | int | None, unit: str = "") -> str:
    return f"{float(value):.3f}{unit}" if value is not None else "N/A"


def run_stage12_motion_estimation() -> None:
    """Run Stage 12 without connecting prediction to visual-servo control."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Unable to open the PyBullet GUI for Stage 12.")
    camera_display: EyeInHandRgbDisplay | None = None
    try:
        camera_display = EyeInHandRgbDisplay("Eye-in-Hand RGB - Stage 12 Motion Estimation")
        print("Stage 12: Target Motion Estimation")
        print("Estimator input: timestamp + raw RGB/OpenCV centroid only.")
        print("Controller input: CURRENT raw RGB/OpenCV detection only; prediction is display/log-only.")
        context = runtime.create_ready_context(client_id, camera_display, "STAGE 12 WARM-UP")
        estimator = TargetMotionEstimator(DEFAULT_EMA_ALPHA, DEFAULT_PREDICTION_HORIZON_SECONDS)
        total_steps = max(1, round(STAGE12_DURATION_SECONDS / sim.TIME_STEP))
        target_orientation = (0.0, 0.0, 0.0, 1.0)
        valid_rgb_frames = estimator_valid_frames = 0
        detected_frames = 0
        velocity_u: list[float] = []
        velocity_v: list[float] = []
        last_detection: RedTargetDetection | None = None
        print(
            "Stage 12 dynamic measurement starts after READY; "
            f"trajectory=Stage 9 MEDIUM ({STAGE12_MEDIUM_FREQUENCY_HZ:.2f} Hz), "
            f"tau={DEFAULT_PREDICTION_HORIZON_SECONDS:.2f} s, alpha={DEFAULT_EMA_ALPHA:.2f}."
        )
        with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=_csv_fields())
            writer.writeheader()
            for step in range(total_steps):
                simulation_time_s = step * sim.TIME_STEP
                # Schedule only: world position/velocity never cross into
                # TargetMotionEstimator or _control_latest_measurement.
                target_world_position = sim._stage9_target_world_position(
                    simulation_time_s, STAGE12_MEDIUM_FREQUENCY_HZ
                )
                p.resetBasePositionAndOrientation(
                    context.target_body_id, target_world_position, target_orientation,
                    physicsClientId=client_id,
                )
                p.resetBaseVelocity(
                    context.target_body_id,
                    linearVelocity=sim._stage9_target_world_velocity(
                        simulation_time_s, STAGE12_MEDIUM_FREQUENCY_HZ
                    ),
                    angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client_id,
                )
                runtime.step_physics(context, client_id)
                if step % sim.CAMERA_UPDATE_INTERVAL_STEPS == 0:
                    sim.update_camera_reference_axes(
                        context.robot_id, client_id, context.camera_axis_debug_item_ids
                    )
                    raw_detection, estimate = _capture_stage12_rgb(
                        context, simulation_time_s, estimator, camera_display, client_id
                    )
                    last_detection = raw_detection
                    ik_command_issued, nonzero_correction_issued = runtime.control_latest_measurement(
                        context, raw_detection, client_id
                    )
                    if raw_detection.detected:
                        detected_frames += 1
                    if estimate is not None:
                        valid_rgb_frames += 1
                        if estimate.estimator_valid:
                            estimator_valid_frames += 1
                            velocity_u.append(abs(estimate.u_dot_filtered))
                            velocity_v.append(abs(estimate.v_dot_filtered))
                    writer.writerow(
                        _log_row(
                            simulation_time_s, raw_detection, estimate, context,
                            ik_command_issued, nonzero_correction_issued,
                        )
                    )
                    status = "DETECTED" if raw_detection.detected else "TARGET LOST"
                    context.debug_text_id = sim.update_motion_debug_text(
                        "STAGE 12 - TARGET MOTION ESTIMATION\n"
                        f"Time: {simulation_time_s:.1f}/{STAGE12_DURATION_SECONDS:.1f}s | {status}\n"
                        f"Estimator: {estimate.status if estimate is not None else 'RESET'} | "
                        "Prediction 鈫?display/log only; controller 鈫?current raw RGB",
                        context.debug_text_id, client_id,
                    )
                time.sleep(sim.TIME_STEP)

        metrics = _offline_future_evaluation(LOG_PATH)
        lock_pass = context.max_locked_deviation < 0.005
        prediction_used_for_control = False
        print("\n====================================")
        print("STAGE 12 MOTION ESTIMATION SUMMARY")
        print("====================================")
        print("Duration:", _format(STAGE12_DURATION_SECONDS, " s"))
        print("Valid RGB frames:", detected_frames)
        print("Estimator valid frames:", estimator_valid_frames)
        print("Prediction horizon:", _format(DEFAULT_PREDICTION_HORIZON_SECONDS, " s"))
        print("EMA alpha:", f"{DEFAULT_EMA_ALPHA:.2f}")
        print("Mean |u_dot|:", _format(float(np.mean(velocity_u)) if velocity_u else None, " px/s"))
        print("Mean |v_dot|:", _format(float(np.mean(velocity_v)) if velocity_v else None, " px/s"))
        print("Baseline future prediction: Mean Error:", _format(metrics["baseline_mean_error"], " px"))
        print("Baseline future prediction: RMSE:", _format(metrics["baseline_rmse"], " px"))
        print("Motion prediction: Mean Error:", _format(metrics["prediction_mean_error"], " px"))
        print("Motion prediction: RMSE:", _format(metrics["prediction_rmse"], " px"))
        print("Improvement:", _format(metrics["improvement_percent"], " %"))
        print("Prediction used for control:", prediction_used_for_control)
        print("World coordinates used by estimator:", False)
        print("World coordinates used by controller:", False)
        print("Locked-joint max deviation:", _format(context.max_locked_deviation, " rad"), "| PASS:" , lock_pass)
        print("CSV:", LOG_PATH)
        print("====================================")
    finally:
        if camera_display is not None:
            camera_display.close()
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)

