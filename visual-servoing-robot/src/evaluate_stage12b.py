"""Offline grid evaluation for Stage 12 image-plane motion prediction.

This module is deliberately offline-only.  It reads the immutable Stage 12
RGB-centroid log and evaluates finite-difference plus EMA predictions over a
fixed alpha/tau grid.  It neither imports PyBullet nor sends robot commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from target_motion_estimator import TargetMotionEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "stage12_motion_estimation.csv"
GRID_LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "stage12b_prediction_grid.csv"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_figures"

ALPHAS = (0.20, 0.35, 0.50, 0.70, 1.00)
HORIZONS_S = (0.05, 0.10, 0.15, 0.20, 0.30)
FUTURE_TIMESTAMP_TOLERANCE_S = 0.020
REQUIRED_COLUMNS = {"time_s", "target_detected", "u_raw", "v_raw"}


@dataclass(frozen=True)
class FutureMatch:
    """One timestamp-based future RGB-centroid lookup result."""

    matched: bool
    future_time_s: float = float("nan")
    future_u: float = float("nan")
    future_v: float = float("nan")


def _load_stage12_log() -> pd.DataFrame:
    """Load and validate the original Stage 12 log without changing it."""

    if not INPUT_LOG_PATH.is_file():
        raise FileNotFoundError(f"Missing Stage 12 log: {INPUT_LOG_PATH}")
    data = pd.read_csv(INPUT_LOG_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Stage 12 log is missing columns: {', '.join(missing)}")
    if any("world" in column.lower() for column in data.columns):
        raise ValueError("Offline estimator must not receive world-coordinate log columns.")

    for column in ("time_s", "u_raw", "v_raw"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["target_detected"] = data["target_detected"].astype(bool)
    if data["time_s"].isna().any() or not data["time_s"].is_monotonic_increasing:
        raise ValueError("Stage 12 time_s must be finite and monotonic increasing.")
    detected = data["target_detected"]
    if data.loc[detected, ["u_raw", "v_raw"]].isna().any().any():
        raise ValueError("Detected Stage 12 frames require finite u_raw and v_raw.")
    return data


def _future_matches(data: pd.DataFrame, tau_s: float) -> list[FutureMatch]:
    """Find each closest *timestamped* valid centroid near t + tau.

    This lookup intentionally does not derive a frame interval or assume a
    fixed camera FPS.  Frames without a target detection are ineligible as
    future truth, exactly as they would be in a vision-only evaluation.
    """

    valid = data.loc[data["target_detected"], ["time_s", "u_raw", "v_raw"]]
    matches: list[FutureMatch] = []

    for time_s in data["time_s"].to_numpy(dtype=float):
        desired_time = time_s + tau_s
        # Use the same strictly-later measurement rule as Stage 12.  idxmin
        # deterministically resolves equal-distance frame candidates by their
        # original chronological order, keeping every alpha/tau comparison on
        # the exact existing evaluation convention.
        candidates = valid.loc[valid["time_s"] > time_s]
        if candidates.empty:
            matches.append(FutureMatch(False))
            continue
        nearest_index = (candidates["time_s"] - desired_time).abs().idxmin()
        closest = candidates.loc[nearest_index]
        if abs(float(closest["time_s"]) - desired_time) > FUTURE_TIMESTAMP_TOLERANCE_S:
            matches.append(FutureMatch(False))
            continue
        matches.append(
            FutureMatch(
                True,
                float(closest["time_s"]),
                float(closest["u_raw"]),
                float(closest["v_raw"]),
            )
        )
    return matches


def _predictions(data: pd.DataFrame, alpha: float, tau_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute image-plane prediction solely from timestamp and centroid."""

    estimator = TargetMotionEstimator(ema_alpha=alpha, prediction_horizon_s=tau_s)
    estimator_valid = np.zeros(len(data), dtype=bool)
    u_pred = np.full(len(data), np.nan, dtype=float)
    v_pred = np.full(len(data), np.nan, dtype=float)

    for row_index, row in enumerate(data.itertuples(index=False)):
        if not bool(row.target_detected):
            estimator.target_lost()
            continue
        estimate = estimator.update(float(row.time_s), float(row.u_raw), float(row.v_raw))
        estimator_valid[row_index] = estimate.estimator_valid
        if estimate.estimator_valid:
            u_pred[row_index] = estimate.u_pred
            v_pred[row_index] = estimate.v_pred
    return estimator_valid, u_pred, v_pred


def _evaluate_configuration(data: pd.DataFrame, alpha: float, tau_s: float) -> dict[str, float | int]:
    """Evaluate one alpha/tau configuration against future RGB observations."""

    estimator_valid, u_pred, v_pred = _predictions(data, alpha, tau_s)
    matches = _future_matches(data, tau_s)
    u = data["u_raw"].to_numpy(dtype=float)
    v = data["v_raw"].to_numpy(dtype=float)
    detected = data["target_detected"].to_numpy(dtype=bool)

    baseline_errors: list[float] = []
    prediction_errors: list[float] = []
    for index, match in enumerate(matches):
        if not (detected[index] and estimator_valid[index] and match.matched):
            continue
        baseline_errors.append(float(np.hypot(u[index] - match.future_u, v[index] - match.future_v)))
        prediction_errors.append(float(np.hypot(u_pred[index] - match.future_u, v_pred[index] - match.future_v)))

    if not baseline_errors:
        raise ValueError(f"No timestamp-matched RGB future samples for alpha={alpha}, tau={tau_s}.")
    baseline = np.asarray(baseline_errors, dtype=float)
    prediction = np.asarray(prediction_errors, dtype=float)
    baseline_mean = float(np.mean(baseline))
    prediction_mean = float(np.mean(prediction))
    return {
        "alpha": alpha,
        "tau": tau_s,
        "baseline_mean_error": baseline_mean,
        "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline)))),
        "prediction_mean_error": prediction_mean,
        "prediction_rmse": float(np.sqrt(np.mean(np.square(prediction)))),
        "improvement_percent": 100.0 * (baseline_mean - prediction_mean) / baseline_mean,
        "valid_samples": int(len(baseline)),
    }


def _evaluate_grid(data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _evaluate_configuration(data, alpha, tau_s)
        for alpha in ALPHAS
        for tau_s in HORIZONS_S
    ]
    return pd.DataFrame(rows)


def _save_figure(figure: plt.Figure, name: str) -> Path:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIRECTORY / name
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def _annotated_heatmap(grid: pd.DataFrame, column: str, title: str, colorbar_label: str, filename: str, fmt: str) -> Path:
    matrix = grid.pivot(index="alpha", columns="tau", values=column).reindex(index=ALPHAS, columns=HORIZONS_S)
    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set(
        title=title,
        xlabel="Prediction horizon tau (s)",
        ylabel="EMA alpha",
        xticks=np.arange(len(HORIZONS_S)),
        xticklabels=[f"{tau:.2f}" for tau in HORIZONS_S],
        yticks=np.arange(len(ALPHAS)),
        yticklabels=[f"{alpha:.2f}" for alpha in ALPHAS],
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    values = matrix.to_numpy(dtype=float)
    midpoint = (float(np.nanmin(values)) + float(np.nanmax(values))) / 2.0
    for row in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row, column_index]
            axis.text(
                column_index,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
                fontsize=9,
            )
    return _save_figure(figure, filename)


def _best_comparison(best: pd.Series) -> Path:
    labels = ("Current centroid\n(baseline)", "EMA motion\nprediction")
    mean_values = (float(best["baseline_mean_error"]), float(best["prediction_mean_error"]))
    rmse_values = (float(best["baseline_rmse"]), float(best["prediction_rmse"]))
    positions = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.bar(positions - width / 2, mean_values, width, color="#4c78a8", label="Mean Error")
    axis.bar(positions + width / 2, rmse_values, width, color="#f58518", label="RMSE")
    axis.set(
        title=f"Stage 12B Best Offline Prediction (alpha={best['alpha']:.2f}, tau={best['tau']:.2f} s)",
        xlabel="Prediction method",
        ylabel="Future Centroid Error (px)",
        xticks=positions,
        xticklabels=labels,
    )
    axis.grid(axis="y", alpha=0.30)
    axis.legend()
    axis.text(
        0.5,
        -0.16,
        f"Mean-error improvement: {best['improvement_percent']:+.2f}% | Valid samples: {int(best['valid_samples'])}",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
    )
    return _save_figure(figure, "stage12b_baseline_vs_prediction.png")


def _print_grid(grid: pd.DataFrame) -> None:
    display = grid.copy()
    for column in (
        "baseline_mean_error", "baseline_rmse", "prediction_mean_error",
        "prediction_rmse", "improvement_percent",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    print("\nAll 25 offline configurations:")
    print(display.to_string(index=False))


def main() -> None:
    """Run the complete read-only Stage 12B evaluation."""

    plt.style.use("seaborn-v0_8-whitegrid")
    data = _load_stage12_log()
    grid = _evaluate_grid(data)
    GRID_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_LOG_PATH, index=False)

    best = grid.loc[grid["prediction_rmse"].idxmin()]
    figures = (
        _annotated_heatmap(
            grid,
            "prediction_rmse",
            "Stage 12B EMA Motion Prediction RMSE",
            "Prediction RMSE (px)",
            "stage12b_prediction_rmse_heatmap.png",
            ".3f",
        ),
        _annotated_heatmap(
            grid,
            "improvement_percent",
            "Stage 12B Mean-Error Improvement versus Current Centroid",
            "Improvement (%)",
            "stage12b_prediction_improvement_heatmap.png",
            ".1f",
        ),
        _best_comparison(best),
    )

    print("\n===== STAGE 12B OFFLINE PREDICTION MODEL EVALUATION =====")
    print("Source log:", INPUT_LOG_PATH)
    print("Source rows:", len(data))
    print("Evaluation is RGB-centroid + timestamp only; no PyBullet/world/controller input.")
    _print_grid(grid)
    print("\nBest configuration by prediction RMSE:")
    print(f"Best alpha: {best['alpha']:.2f}")
    print(f"Best tau: {best['tau']:.2f} s")
    print(f"Best prediction RMSE: {best['prediction_rmse']:.4f} px")
    print(f"Corresponding baseline RMSE: {best['baseline_rmse']:.4f} px")
    print(f"Mean-error improvement: {best['improvement_percent']:+.2f}%")
    print("\nGrid CSV:", GRID_LOG_PATH)
    print("Figures:")
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
