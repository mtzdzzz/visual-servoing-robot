"""Visualise Stage 10 robustness logs without changing any experimental data.

Run after ``python src/simulation.py --stage10``.  The script reads the CSV
files as immutable experimental evidence and writes the required figures to
``outputs/figures``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "figures"
TARGET_LOST_LOG = LOG_DIRECTORY / "stage10_target_lost.csv"
SUMMARY_LOG = LOG_DIRECTORY / "stage10_summary.csv"


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 10 data: {path}")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return frame


def _quality_check(label: str, frame: pd.DataFrame, duration_seconds: float | None = None) -> None:
    time_values = pd.to_numeric(frame["time_s"], errors="coerce")
    print(f"{label}: {len(frame)} rows")
    print("  NaN time rows:", int(time_values.isna().sum()))
    print("  Time monotonic:", bool(time_values.dropna().is_monotonic_increasing))
    if not time_values.dropna().empty:
        actual_duration = float(time_values.max() - time_values.min())
        print(f"  Recorded time range: {actual_duration:.3f} s")
        if duration_seconds is not None and abs(actual_duration - duration_seconds) > 0.2:
            print(f"  WARNING: expected approximately {duration_seconds:.1f} s.")
    numeric_error = pd.to_numeric(frame.get("error_norm"), errors="coerce")
    print("  NaN / unavailable error rows:", int(numeric_error.isna().sum()))
    if "phase" in frame.columns and (frame["phase"] == "WARM-UP").any():
        print("  WARNING: WARM-UP rows found in formal Stage 10 data.")


def contiguous_false_intervals(time_values: np.ndarray, detected: np.ndarray) -> list[tuple[float, float]]:
    """Return spans where target_detected is false for readable shading."""
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    previous_time: float | None = None
    for current_time, is_detected in zip(time_values, detected):
        if not is_detected and start is None:
            start = float(current_time)
        if is_detected and start is not None:
            intervals.append((start, float(previous_time if previous_time is not None else current_time)))
            start = None
        previous_time = float(current_time)
    if start is not None and previous_time is not None:
        intervals.append((start, previous_time))
    return intervals


def plot_target_lost_recovery(frame: pd.DataFrame) -> Path:
    time_values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy()
    errors = pd.to_numeric(frame["error_norm"], errors="coerce").to_numpy()
    detected = pd.to_numeric(frame["target_detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(time_values, errors, color="#1f77b4", linewidth=1.6, label="RGB tracking error norm")
    for index, (start, end) in enumerate(contiguous_false_intervals(time_values, detected)):
        axis.axvspan(start, end, color="#d62728", alpha=0.20, label="Target lost" if index == 0 else None)
    axis.axhline(2.0 * np.sqrt(2.0), linestyle="--", color="#2ca02c", linewidth=1.1,
                 label="2 px/axis precision bound")
    axis.set_title("Stage 10A — Target Lost and Automatic Recovery")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Tracking error norm (px)")
    axis.grid(True, alpha=0.30)
    axis.legend(loc="upper right")
    path = FIGURE_DIRECTORY / "stage10_target_lost_recovery.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _noise_value(condition: object) -> float:
    try:
        return float(str(condition).split()[0])
    except (ValueError, IndexError):
        return float("inf")


def plot_noise_comparison(summary: pd.DataFrame) -> Path:
    noise = summary.loc[summary["experiment"] == "VISUAL_NOISE"].copy()
    if noise.empty:
        raise ValueError("stage10_summary.csv has no VISUAL_NOISE rows.")
    noise["noise_px"] = noise["condition"].map(_noise_value)
    noise = noise.sort_values("noise_px")
    labels = [f"{value:g} px" for value in noise["noise_px"]]
    mean_error = pd.to_numeric(noise["mean_error"], errors="coerce").to_numpy()
    rmse = pd.to_numeric(noise["rmse"], errors="coerce").to_numpy()
    positions = np.arange(len(noise))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    axis.bar(positions - width / 2, mean_error, width, color="#4c78a8", label="Mean error norm")
    axis.bar(positions + width / 2, rmse, width, color="#f58518", label="RMSE")
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Injected RGB measurement noise")
    axis.set_ylabel("Pixel error (px)")
    axis.set_title("Stage 10B — Visual Noise Robustness")
    axis.grid(axis="y", alpha=0.30)
    axis.legend()
    path = FIGURE_DIRECTORY / "stage10_noise_comparison.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _format(value: object, unit: str = "px") -> str:
    try:
        return f"{float(value):.2f} {unit}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> int:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    target_lost = _read_csv(TARGET_LOST_LOG, {"time_s", "error_norm", "target_detected", "phase"})
    summary = _read_csv(
        SUMMARY_LOG,
        {"experiment", "condition", "mean_error", "rmse", "p95_error", "max_error", "result"},
    )
    _quality_check("stage10_target_lost.csv", target_lost, duration_seconds=14.0)
    for noise_level in (0, 1, 3):
        noise_log = _read_csv(
            LOG_DIRECTORY / f"stage10_noise_{noise_level}.csv",
            {"time_s", "error_norm", "target_detected", "measurement_u", "measurement_v"},
        )
        _quality_check(f"stage10_noise_{noise_level}.csv", noise_log, duration_seconds=20.0)
    target_lost_figure = plot_target_lost_recovery(target_lost)
    noise_figure = plot_noise_comparison(summary)
    print("\nStage 10 summaries:")
    for _, row in summary.iterrows():
        print(
            f"  {row['experiment']} / {row['condition']}: "
            f"mean={_format(row['mean_error'])}; RMSE={_format(row['rmse'])}; "
            f"P95={_format(row['p95_error'])}; max={_format(row['max_error'])}; "
            f"result={row['result']}"
        )
    print("\nFigures:")
    print(target_lost_figure)
    print(noise_figure)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Stage 10 visualisation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
