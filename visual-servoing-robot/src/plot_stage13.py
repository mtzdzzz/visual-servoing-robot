"""Read-only Stage 13 A/B result plots using raw RGB tracking error."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_figures"
SUMMARY_PATH = LOG_DIRECTORY / "stage13_summary.csv"


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing Stage 13 summary: {SUMMARY_PATH}")
    data = pd.read_csv(SUMMARY_PATH)
    required = {"speed_level", "controller_mode", "prediction_horizon_s", "rmse", "mean_tracking_error", "p95_error"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Stage 13 summary missing: {', '.join(missing)}")
    for column in ("prediction_horizon_s", "rmse", "mean_tracking_error", "p95_error", "max_error"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _save(figure: plt.Figure, filename: str) -> Path:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIRECTORY / filename
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def _medium_rows(summary: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    medium = summary.loc[summary["speed_level"].eq("MEDIUM")].copy()
    baseline = medium.loc[medium["controller_mode"].eq("BASELINE")]
    predictive = medium.loc[medium["controller_mode"].eq("PREDICTIVE")].sort_values("prediction_horizon_s")
    if len(baseline) != 1 or len(predictive) != 3:
        raise ValueError("Expected exactly one MEDIUM baseline and three predictive horizons.")
    return baseline.iloc[0], predictive


def _rmse_comparison(summary: pd.DataFrame) -> Path:
    baseline, predictive = _medium_rows(summary)
    labels = ["Baseline"] + [f"tau={value:.2f}" for value in predictive["prediction_horizon_s"]]
    values = [float(baseline["rmse"])] + predictive["rmse"].astype(float).tolist()
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    bars = axis.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    axis.set(title="Stage 13 MEDIUM Raw Tracking RMSE", xlabel="Controller measurement", ylabel="Raw RGB Tracking RMSE (px)")
    axis.grid(axis="y", alpha=0.30)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    return _save(figure, "stage13_baseline_vs_predictive_rmse.png")


def _timeseries(summary: pd.DataFrame) -> Path:
    baseline, predictive = _medium_rows(summary)
    best = predictive.loc[predictive["rmse"].idxmin()]
    baseline_log = pd.read_csv(baseline["log_path"])
    best_log = pd.read_csv(best["log_path"])
    for data in (baseline_log, best_log):
        for column in ("time_s", "error_norm_raw"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.plot(baseline_log["time_s"], baseline_log["error_norm_raw"], color="#4c78a8", linewidth=1.35, label="Baseline")
    axis.plot(best_log["time_s"], best_log["error_norm_raw"], color="#f58518", linewidth=1.35, label=f"Best predictive tau={best['prediction_horizon_s']:.2f}")
    axis.set(title="Stage 13 MEDIUM Raw Tracking Error", xlabel="Time (s)", ylabel="Raw RGB Tracking Error (px)")
    axis.grid(True, alpha=0.30)
    axis.legend()
    return _save(figure, "stage13_tracking_error_timeseries.png")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = _load_summary()
    outputs = (_rmse_comparison(summary), _timeseries(summary))
    print("Stage 13 figures:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
