"""Plots for Stage 17 obstacle perception evaluation only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "logs" / "stage17_summary.csv"
FIGURE_DIRECTORY = ROOT / "outputs" / "final_figures"


def generate_stage17_figures() -> list[Path]:
    """Create two PNGs from the immutable Stage 17 summary CSV."""

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Stage 17 summary not found: {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH)
    required = {
        "scene", "mean_center_error_m", "mean_aabb_iou", "mean_gt_coverage"
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Stage 17 summary is missing columns: {sorted(missing)}")
    scenes = summary[summary["scene"] != "OVERALL"].copy()
    if scenes.empty:
        raise ValueError("Stage 17 summary has no per-scene rows.")
    if scenes.isna().any().any():
        raise ValueError("Stage 17 summary has NaN values; raw logs were not altered.")
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})

    center_path = FIGURE_DIRECTORY / "stage17_obstacle_center_error.png"
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(scenes["scene"], scenes["mean_center_error_m"] * 1000.0, color="#e6a700")
    axis.set_title("Stage 17 Obstacle Center Localization Error")
    axis.set_xlabel("Static obstacle scene")
    axis.set_ylabel("3D center error (mm)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(center_path, dpi=220)
    plt.close(figure)

    occupancy_path = FIGURE_DIRECTORY / "stage17_obstacle_occupancy.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    axes[0].bar(scenes["scene"], scenes["mean_aabb_iou"] * 100.0, color="#4c78a8")
    axes[0].set_title("Estimated Occupancy IoU")
    axes[0].set_ylabel("Percent (%)")
    axes[1].bar(scenes["scene"], scenes["mean_gt_coverage"] * 100.0, color="#59a14f")
    axes[1].set_title("Ground-Truth AABB Coverage")
    for axis in axes:
        axis.set_xlabel("Static obstacle scene")
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(0.0, 105.0)
    figure.suptitle("Stage 17 Conservative Obstacle Occupancy")
    figure.tight_layout()
    figure.savefig(occupancy_path, dpi=220)
    plt.close(figure)
    return [center_path, occupancy_path]
