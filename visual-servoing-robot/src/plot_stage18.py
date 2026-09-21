"""CSV-only Stage 18 collision evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "outputs" / "logs" / "stage18_collision_checking.csv"
FIGURE_DIRECTORY = ROOT / "outputs" / "final_figures"


def generate_stage18_figures() -> list[Path]:
    """Generate confusion and path-type figures from true saved path results."""

    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Stage 18 log not found: {LOG_PATH}")
    data = pd.read_csv(LOG_PATH)
    required = {"path_id", "path_type", "estimated_collision", "gt_collision", "classification"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Stage 18 CSV is missing columns: {sorted(missing)}")
    if len(data) < 20 or data[sorted(required)].isna().any().any():
        raise ValueError("Stage 18 CSV has fewer than 20 paths or required missing values.")
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})

    labels = ("TN", "FP", "FN", "TP")
    counts = {label: int((data["classification"] == label).sum()) for label in labels}
    matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    confusion_path = FIGURE_DIRECTORY / "stage18_collision_confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Path count")
    axis.set_xticks((0, 1), ("Estimated safe", "Estimated collision"))
    axis.set_yticks((0, 1), ("GT safe", "GT collision"))
    axis.set_title("Stage 18 Collision Classification")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=16)
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=220)
    plt.close(figure)

    results_path = FIGURE_DIRECTORY / "stage18_path_results.png"
    path_types = [item for item in ("SAFE", "COLLISION", "NEAR_BOUNDARY") if item in set(data["path_type"])]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    bottoms = np.zeros(len(path_types), dtype=float)
    colors = {"TP": "#e15759", "TN": "#59a14f", "FP": "#f2cf5b", "FN": "#b279a2"}
    for label in ("TN", "TP", "FP", "FN"):
        values = np.array([
            int(((data["path_type"] == path_type) & (data["classification"] == label)).sum())
            for path_type in path_types
        ])
        axis.bar(path_types, values, bottom=bottoms, label=label, color=colors[label])
        bottoms += values
    axis.set_title("Stage 18 Path Results by Candidate Type")
    axis.set_xlabel("Candidate path type")
    axis.set_ylabel("Path count")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Classification")
    figure.tight_layout()
    figure.savefig(results_path, dpi=220)
    plt.close(figure)
    return [confusion_path, results_path]
