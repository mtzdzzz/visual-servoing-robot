"""Read-only plots and quality checks for Stage 12 motion estimation."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "stage12_motion_estimation.csv"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_figures"
REQUIRED_COLUMNS = {
    "time_s", "target_detected", "u_raw", "v_raw", "ex_raw", "ey_raw",
    "estimator_valid", "u_dot_raw", "v_dot_raw", "u_dot_filtered", "v_dot_filtered",
    "u_pred", "v_pred", "prediction_horizon_s", "servo_ex", "servo_ey",
    "prediction_used_for_control", "future_timestamp_s", "future_u_raw", "future_v_raw",
    "baseline_future_error", "motion_prediction_future_error",
}


def _load() -> pd.DataFrame:
    if not LOG_PATH.is_file():
        raise FileNotFoundError(f"Missing Stage 12 log: {LOG_PATH}")
    data = pd.read_csv(LOG_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"stage12_motion_estimation.csv missing columns: {', '.join(missing)}")
    numeric = REQUIRED_COLUMNS - {"prediction_used_for_control"}
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _quality_check(data: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if data.empty:
        issues.append("CSV has no RGB-frame records.")
        return issues
    times = data["time_s"]
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        issues.append("time_s must be finite and strictly increasing.")
    duration = float(times.iloc[-1] - times.iloc[0])
    if not 19.8 <= duration <= 20.1:
        issues.append(f"formal duration is {duration:.3f} s, expected approximately 20 s.")
    prediction_control = data["prediction_used_for_control"].astype(str).str.lower()
    if not prediction_control.eq("false").all():
        issues.append("prediction_used_for_control contains a value other than False.")
    detected = data["target_detected"].eq(1)
    if detected.any():
        mismatch = (
            data.loc[detected, "servo_ex"].sub(data.loc[detected, "ex_raw"]).abs().gt(1e-12)
            | data.loc[detected, "servo_ey"].sub(data.loc[detected, "ey_raw"]).abs().gt(1e-12)
        )
        if mismatch.any():
            issues.append("servo_ex/servo_ey differ from current raw ex_raw/ey_raw.")
    finite_columns = ("u_dot_raw", "v_dot_raw", "u_dot_filtered", "v_dot_filtered", "u_pred", "v_pred")
    estimated = data["estimator_valid"].eq(1)
    for column in finite_columns:
        values = data.loc[estimated, column].dropna().to_numpy(dtype=float)
        if len(values) and not np.isfinite(values).all():
            issues.append(f"{column} contains non-finite estimator values.")
    if any("world" in column.lower() for column in data.columns):
        issues.append("Stage 12 log unexpectedly contains world-coordinate data.")
    return issues


def _metrics(data: pd.DataFrame) -> dict[str, float | int | None]:
    baseline = data["baseline_future_error"].dropna().to_numpy(dtype=float)
    prediction = data["motion_prediction_future_error"].dropna().to_numpy(dtype=float)
    if not len(baseline) or not len(prediction):
        return {
            "samples": 0, "baseline_mean": None, "baseline_rmse": None,
            "prediction_mean": None, "prediction_rmse": None, "improvement": None,
        }
    baseline_mean = float(np.mean(baseline))
    prediction_mean = float(np.mean(prediction))
    return {
        "samples": int(len(baseline)),
        "baseline_mean": baseline_mean,
        "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline)))),
        "prediction_mean": prediction_mean,
        "prediction_rmse": float(np.sqrt(np.mean(np.square(prediction)))),
        "improvement": 100.0 * (baseline_mean - prediction_mean) / baseline_mean if baseline_mean else None,
    }


def _save(figure: plt.Figure, filename: str) -> Path:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIRECTORY / filename
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def _prediction_trace(data: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    valid_estimate = data["estimator_valid"].eq(1)
    evaluation = data["future_timestamp_s"].notna()
    for axis, raw, predicted, future, label in (
        (axes[0], "u_raw", "u_pred", "future_u_raw", "Horizontal centroid u (px)"),
        (axes[1], "v_raw", "v_pred", "future_v_raw", "Vertical centroid v (px)"),
    ):
        axis.plot(data["time_s"], data[raw], color="#2a6fbb", linewidth=1.4, label="Current RGB centroid")
        axis.plot(data.loc[valid_estimate, "time_s"], data.loc[valid_estimate, predicted], color="#c74440", linewidth=1.3, label="Motion prediction")
        axis.plot(data.loc[evaluation, "time_s"], data.loc[evaluation, future], color="#4daf4a", linestyle="--", linewidth=1.1, label="Measured centroid at t + tau")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.30)
        axis.legend(loc="upper right")
    axes[0].set_title("Stage 12 Image-Plane Motion Prediction")
    axes[1].set_xlabel("Time (s)")
    return _save(figure, "stage12_motion_prediction.png")


def _prediction_error(metrics: dict[str, float | int | None]) -> Path:
    if metrics["baseline_mean"] is None or metrics["prediction_mean"] is None:
        raise ValueError("No valid future-measurement pairs exist for Stage 12 evaluation.")
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    labels = ("Current centroid\n(baseline)", "Motion prediction")
    positions = np.arange(2)
    width = 0.34
    axis.bar(positions - width / 2, [metrics["baseline_mean"], metrics["prediction_mean"]], width, label="Mean Error", color="#4c78a8")
    axis.bar(positions + width / 2, [metrics["baseline_rmse"], metrics["prediction_rmse"]], width, label="RMSE", color="#f58518")
    axis.set(title="Stage 12 Future Centroid Prediction Error", xlabel="Prediction method", ylabel="Future Centroid Error (px)", xticks=positions, xticklabels=labels)
    axis.grid(axis="y", alpha=0.30)
    axis.legend()
    improvement = metrics["improvement"]
    axis.text(0.5, -0.18, f"Mean-error improvement versus current-centroid baseline: {float(improvement):+.2f}%", transform=axis.transAxes, ha="center", fontsize=10)
    return _save(figure, "stage12_prediction_error.png")


def _format(value: float | int | None, unit: str = "") -> str:
    return f"{float(value):.3f}{unit}" if value is not None else "N/A"


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    data = _load()
    issues = _quality_check(data)
    metrics = _metrics(data)
    figures = (_prediction_trace(data), _prediction_error(metrics))
    print("\n===== Stage 12 Offline Evaluation =====")
    print("Rows:", len(data))
    print("Evaluation pairs:", metrics["samples"])
    print("Baseline future mean / RMSE:", _format(metrics["baseline_mean"], " px"), "/", _format(metrics["baseline_rmse"], " px"))
    print("Motion prediction mean / RMSE:", _format(metrics["prediction_mean"], " px"), "/", _format(metrics["prediction_rmse"], " px"))
    print("Improvement:", _format(metrics["improvement"], " %"))
    print("prediction_used_for_control all False:", not any("prediction_used" in issue for issue in issues))
    if issues:
        for issue in issues:
            print("WARNING:", issue)
    else:
        print("Data quality: PASS")
    print("Figures:")
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
