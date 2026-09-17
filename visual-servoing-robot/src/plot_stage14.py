"""Create Stage 14 RGB-D localization figures from unmodified CSV logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "logs" / "stage14_rgbd_summary.csv"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_figures"
ERROR_FIGURE_PATH = FIGURE_DIRECTORY / "stage14_3d_localization_error.png"
COORDINATE_FIGURE_PATH = FIGURE_DIRECTORY / "stage14_estimated_vs_ground_truth.png"


def _read_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Stage 14 summary CSV does not exist: {SUMMARY_PATH}")
    data = pd.read_csv(SUMMARY_PATH)
    required = {
        "position_label", "result", "center_error_3d_mm", "surface_error_3d_mm",
        "estimated_world_x", "estimated_world_y", "estimated_world_z",
        "gt_world_x", "gt_world_y", "gt_world_z",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Stage 14 summary misses required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Stage 14 summary contains no trial rows.")
    if data["position_label"].duplicated().any():
        raise ValueError("Stage 14 summary contains duplicate position labels.")
    numeric = list(required - {"position_label", "result"})
    if data[numeric].isna().any().any():
        print("Data-quality warning: one or more localization summary values are NaN.")
    return data


def generate_stage14_figures() -> tuple[Path, Path]:
    """Generate two direct, non-smoothed figures from the five trial summaries."""

    data = _read_summary()
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    labels = data["position_label"].astype(str).str.title().to_list()
    x = np.arange(len(data))

    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.bar(x - 0.18, data["center_error_3d_mm"], width=0.36, label="Centre compensated")
    axis.bar(x + 0.18, data["surface_error_3d_mm"], width=0.36, label="Visible surface")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Static target position")
    axis.set_ylabel("3D localization error (mm)")
    axis.set_title("Stage 14 RGB-D Localization Error")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(ERROR_FIGURE_PATH, dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), sharex=True, constrained_layout=True)
    coordinate_names = ("X", "Y", "Z")
    for axis, coordinate in zip(axes, coordinate_names):
        estimated = data[f"estimated_world_{coordinate.lower()}"]
        ground_truth = data[f"gt_world_{coordinate.lower()}"]
        axis.plot(x, ground_truth, "o-", label="Ground truth", color="#1f77b4")
        axis.plot(x, estimated, "s--", label="RGB-D estimate", color="#d62728")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_xlabel("Static target position")
        axis.set_ylabel(f"{coordinate} coordinate (m)")
        axis.set_title(f"World {coordinate}")
        axis.grid(alpha=0.3)
    axes[0].legend(loc="best")
    figure.suptitle("Stage 14 Estimated vs Ground-Truth World Coordinates")
    figure.savefig(COORDINATE_FIGURE_PATH, dpi=220)
    plt.close(figure)
    return ERROR_FIGURE_PATH, COORDINATE_FIGURE_PATH


if __name__ == "__main__":
    first, second = generate_stage14_figures()
    print("Generated:", first)
    print("Generated:", second)
