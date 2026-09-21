"""Generate Stage 19 RRT-Connect planning figures from saved CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "outputs" / "logs" / "stage19_rrt_connect.csv"
FIGURE_DIRECTORY = ROOT / "outputs" / "final_figures"


def _read_rows() -> list[dict[str, str]]:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Stage 19 log is missing: {LOG_PATH}")
    with LOG_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("Stage 19 log has no planning rows.")
    return rows


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


def generate_stage19_figures() -> tuple[Path, Path, Path]:
    """Create success-rate, planning-time, and path-length PNGs from raw CSV."""

    rows = _read_rows()
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rrt_rows = [row for row in rows if _is_true(row["rrt_required"]) and row["seed"]]
    if not rrt_rows:
        raise RuntimeError("Stage 19 CSV contains no RRT trials to plot.")

    direct_rows = [row for row in rows if not _is_true(row["rrt_required"])]
    direct_rate = 100.0 * sum(_is_true(row["final_path_estimated_safe"]) for row in direct_rows) / max(1, len(direct_rows))
    rrt_rate = 100.0 * sum(_is_true(row["planning_success"]) for row in rrt_rows) / len(rrt_rows)
    success_path = FIGURE_DIRECTORY / "stage19_planning_success_rate.png"
    fig, axis = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    bars = axis.bar(["Direct path", "RRT-Connect"], [direct_rate, rrt_rate], color=["#4c9f50", "#3178b6"])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Successful plans (%)")
    axis.set_title("Stage 19 Planning Success Rate")
    axis.grid(axis="y", alpha=0.28)
    for bar, value in zip(bars, (direct_rate, rrt_rate)):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(success_path)
    plt.close(fig)

    time_path = FIGURE_DIRECTORY / "stage19_planning_time.png"
    labels = [f"{row['scenario_id']}\nseed {row['seed']}" for row in rrt_rows]
    times = [float(row["planning_time_s"]) for row in rrt_rows]
    colours = ["#3178b6" if _is_true(row["planning_success"]) else "#c74343" for row in rrt_rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(labels) * 0.40), 4.4), dpi=220)
    axis.bar(np.arange(len(labels)), times, color=colours)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90)
    axis.set_ylabel("Planning time (s)")
    axis.set_title("Stage 19 RRT-Connect Planning Time")
    axis.grid(axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(time_path)
    plt.close(fig)

    length_path = FIGURE_DIRECTORY / "stage19_path_length.png"
    successful = [row for row in rrt_rows if _is_true(row["planning_success"]) and row["rrt_path_length"]]
    if not successful:
        raise RuntimeError("Stage 19 contains no successful RRT paths to plot.")
    labels = [f"{row['scenario_id']}:{row['seed']}" for row in successful]
    direct_lengths = [float(row["direct_joint_distance"]) for row in successful]
    rrt_lengths = [float(row["rrt_path_length"]) for row in successful]
    locations = np.arange(len(labels))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(8.0, len(labels) * 0.42), 4.5), dpi=220)
    axis.bar(locations - width / 2, direct_lengths, width, label="Direct joint distance", color="#9da9b5")
    axis.bar(locations + width / 2, rrt_lengths, width, label="RRT path length", color="#4c9f50")
    axis.set_xticks(locations, labels, rotation=90)
    axis.set_ylabel("Joint-space length (rad)")
    axis.set_title("Stage 19 Direct Distance vs RRT Path Length")
    axis.grid(axis="y", alpha=0.28)
    axis.legend()
    fig.tight_layout()
    fig.savefig(length_path)
    plt.close(fig)
    return success_path, time_path, length_path


if __name__ == "__main__":
    for figure in generate_stage19_figures():
        print(figure)
