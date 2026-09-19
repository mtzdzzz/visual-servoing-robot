"""Generate Stage 15 summary figures from unmodified experiment CSV output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "logs" / "stage15_summary.csv"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_figures"
DETECTION_FIGURE_PATH = FIGURE_DIRECTORY / "stage15_detection_rate.png"
ERROR_FIGURE_PATH = FIGURE_DIRECTORY / "stage15_3d_localization_error.png"


def _read_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Stage 15 summary CSV does not exist: {SUMMARY_PATH}")
    data = pd.read_csv(SUMMARY_PATH)
    required = {
        "target", "detection_rate_percent", "mean_3d_error_m", "rmse_3d_error_m",
        "max_3d_error_m", "all_targets_detection_rate_percent",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Stage 15 summary misses required columns: {sorted(missing)}")
    if set(("RED", "GREEN", "BLUE", "ALL_THREE")).difference(set(data["target"])):
        raise ValueError("Stage 15 summary does not contain all required target rows.")
    return data


def generate_stage15_figures() -> tuple[Path, Path]:
    """Save the required detection-rate and independent RGB-D error charts."""

    data = _read_summary()
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    individual = data[data["target"].isin(["RED", "GREEN", "BLUE"])].copy()
    all_three = data.loc[data["target"] == "ALL_THREE"].iloc[0]
    labels = [*individual["target"].tolist(), "ALL THREE"]
    rates = [*individual["detection_rate_percent"].astype(float).tolist(), float(all_three["all_targets_detection_rate_percent"])]
    colours = ["#d62728", "#2ca02c", "#1f77b4", "#555555"]

    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    bars = axis.bar(labels, rates, color=colours)
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Detection rate (%)")
    axis.set_title("Stage 15 Multi-Target Detection Rate")
    axis.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        axis.text(bar.get_x() + bar.get_width() / 2.0, rate + 1.5, f"{rate:.1f}%", ha="center")
    figure.savefig(DETECTION_FIGURE_PATH, dpi=220)
    plt.close(figure)

    x = np.arange(len(individual))
    mean_mm = individual["mean_3d_error_m"].astype(float).to_numpy() * 1000.0
    rmse_mm = individual["rmse_3d_error_m"].astype(float).to_numpy() * 1000.0
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    axis.bar(x - 0.18, mean_mm, width=0.36, label="Mean 3D error")
    axis.bar(x + 0.18, rmse_mm, width=0.36, label="RMSE 3D error")
    axis.set_xticks(x, individual["target"].tolist())
    axis.set_ylabel("3D localization error (mm)")
    axis.set_title("Stage 15 Per-Target RGB-D Localization Error")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(ERROR_FIGURE_PATH, dpi=220)
    plt.close(figure)
    return DETECTION_FIGURE_PATH, ERROR_FIGURE_PATH


if __name__ == "__main__":
    paths = generate_stage15_figures()
    for path in paths:
        print("Generated:", path)
