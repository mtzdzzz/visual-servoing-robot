"""Create quality-checked Stage 9 pure-dynamic tracking figures.

This module is deliberately offline and read-only with respect to Stage 9
logs.  It never imports simulation.py and cannot alter the visual-servo
controller, robot, camera, target trajectory, or their parameters.

Run from the project root with:
    .\\.venv\\Scripts\\python.exe .\\src\\plot_stage9.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # Save portable PNGs; do not require an interactive GUI.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "figures"
SUMMARY_PATH = LOG_DIRECTORY / "stage9_summary.csv"
TRIAL_PATHS = {
    "SLOW": LOG_DIRECTORY / "stage9_slow.csv",
    "MEDIUM": LOG_DIRECTORY / "stage9_medium.csv",
    "FAST": LOG_DIRECTORY / "stage9_fast.csv",
}
TRIAL_COLORS = {"SLOW": "#2a6fbb", "MEDIUM": "#e58f26", "FAST": "#c74440"}

TRIAL_REQUIRED_COLUMNS = {
    "time_s",
    "ex",
    "ey",
    "error_norm",
    "target_detected",
    "servo_decision",
    "ik_command_issued",
    "nonzero_correction_issued",
}
SUMMARY_REQUIRED_COLUMNS = {
    "speed_level",
    "duration_s",
    "mean_error_norm",
    "rmse",
    "percentile_95_error_norm",
    "max_error_norm",
    "rgb_fps",
    "detection_fps",
    "servo_decision_fps",
    "ik_command_fps",
    "nonzero_correction_fps",
}
NUMERIC_TRIAL_COLUMNS = (
    "time_s",
    "ex",
    "ey",
    "error_norm",
    "target_detected",
    "servo_decision",
    "ik_command_issued",
    "nonzero_correction_issued",
)


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {', '.join(missing)}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> pd.DataFrame:
    checked = frame.copy()
    for column in columns:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    if checked[list(columns)].isna().any().any():
        invalid = checked[list(columns)].columns[checked[list(columns)].isna().any()].tolist()
        raise ValueError(f"{path.name}: NaN/non-numeric values in {invalid}")
    return checked


def _load_trial(level: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage 9 log is missing: {path}")
    frame = _numeric(pd.read_csv(path), NUMERIC_TRIAL_COLUMNS, path)
    _require_columns(frame, TRIAL_REQUIRED_COLUMNS, path)
    if frame.empty:
        raise ValueError(f"{path.name}: no formal dynamic rows")
    if not frame["time_s"].is_monotonic_increasing or frame["time_s"].duplicated().any():
        raise ValueError(f"{path.name}: time_s must be strictly increasing")
    duration = float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0])
    if not 19.8 <= duration <= 20.1:
        raise ValueError(
            f"{path.name}: formal duration is {duration:.3f} s, expected approximately 20 s"
        )
    if not np.isclose(float(frame["time_s"].iloc[0]), 0.0, atol=1e-6):
        raise ValueError(f"{path.name}: formal data does not start at t=0")
    # The fixed protocol only writes formal rows after READY. A large first
    # error signals accidental warm-up inclusion without modifying any data.
    first_error = float(frame["error_norm"].iloc[0])
    if first_error > 5.0:
        raise ValueError(
            f"{path.name}: first formal error={first_error:.3f}px; likely contains WARM-UP"
        )
    print(
        f"{level}: quality PASS | rows={len(frame)}, duration={duration:.3f}s, "
        f"first error={first_error:.3f}px, NaN=0, monotonic time=YES"
    )
    return frame


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Required Stage 9 summary is missing: {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH)
    _require_columns(summary, SUMMARY_REQUIRED_COLUMNS, SUMMARY_PATH)
    summary = summary.set_index("speed_level").reindex(TRIAL_PATHS)
    if summary.index.isna().any() or summary.isna().all(axis=1).any():
        raise ValueError("stage9_summary.csv: SLOW/MEDIUM/FAST rows are required")
    numeric_columns = SUMMARY_REQUIRED_COLUMNS - {"speed_level"}
    summary = _numeric(summary, numeric_columns, SUMMARY_PATH)
    return summary


def _save(figure: plt.Figure, filename: str) -> Path:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIRECTORY / filename
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_error_norm(trials: dict[str, pd.DataFrame]) -> Path:
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    for level, frame in trials.items():
        axis.plot(
            frame["time_s"], frame["error_norm"], label=level,
            color=TRIAL_COLORS[level], linewidth=1.8,
        )
    axis.set(
        title="Stage 9 Pure Dynamic Tracking Error",
        xlabel="Time (s)",
        ylabel="Tracking Error Norm (px)",
        xlim=(0.0, 20.0),
    )
    axis.grid(True, alpha=0.3)
    axis.legend(title="Target speed")
    return _save(figure, "stage9_error_norm_comparison.png")


def _plot_ex_ey(level: str, frame: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.plot(frame["time_s"], frame["ex"], label="ex", color="#1f77b4", linewidth=1.6)
    axis.plot(frame["time_s"], frame["ey"], label="ey", color="#d95f02", linewidth=1.6)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="zero error")
    axis.set(
        title=f"Stage 9 {level}: Pixel Errors",
        xlabel="Time (s)",
        ylabel="Pixel Error (px)",
        xlim=(0.0, 20.0),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save(figure, f"stage9_{level.lower()}_ex_ey.png")


def _grouped_bars(
    summary: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    figure, axis = plt.subplots(figsize=(10.0, 5.6))
    levels = list(summary.index)
    x = np.arange(len(levels))
    width = 0.8 / len(columns)
    for index, (column, label) in enumerate(zip(columns, labels)):
        positions = x - 0.4 + width / 2 + index * width
        axis.bar(positions, summary[column], width, label=label)
    axis.set(
        title=title,
        xlabel="Target speed level",
        ylabel=ylabel,
        xticks=x,
        xticklabels=levels,
    )
    axis.grid(True, axis="y", alpha=0.3)
    if filename == "stage9_fps_comparison.png":
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    else:
        axis.legend()
    return _save(figure, filename)


def _plot_max_error(summary: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(7.8, 5.1))
    levels = list(summary.index)
    bars = axis.bar(levels, summary["max_error_norm"], color=[TRIAL_COLORS[level] for level in levels])
    for bar, value in zip(bars, summary["max_error_norm"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    axis.set(
        title="Stage 9 Maximum Dynamic Tracking Error",
        xlabel="Target speed level",
        ylabel="Maximum Tracking Error (px)",
    )
    axis.grid(True, axis="y", alpha=0.3)
    return _save(figure, "stage9_max_error.png")


def _dominant_frequency(frame: pd.DataFrame) -> float | None:
    values = frame["error_norm"].to_numpy(dtype=float)
    time_values = frame["time_s"].to_numpy(dtype=float)
    if len(values) < 4:
        return None
    period = float(np.median(np.diff(time_values)))
    spectrum = np.abs(np.fft.rfft(values - np.mean(values)))
    frequencies = np.fft.rfftfreq(len(values), d=period)
    if len(frequencies) <= 1:
        return None
    peak_index = int(np.argmax(spectrum[1:]) + 1)
    return float(frequencies[peak_index])


def _report(summary: pd.DataFrame, trials: dict[str, pd.DataFrame]) -> None:
    print("\n===== Stage 9 Visualization Summary =====")
    for level in TRIAL_PATHS:
        row = summary.loc[level]
        frame = trials[level]
        mean_abs_ex = float(frame["ex"].abs().mean())
        mean_abs_ey = float(frame["ey"].abs().mean())
        dominant = _dominant_frequency(frame)
        print(
            f"{level}: Mean Error={row['mean_error_norm']:.3f}px | "
            f"RMSE={row['rmse']:.3f}px | P95={row['percentile_95_error_norm']:.3f}px | "
            f"Max={row['max_error_norm']:.3f}px | "
            f"mean|ex|={mean_abs_ex:.3f}px, mean|ey|={mean_abs_ey:.3f}px | "
            f"dominant error frequency={dominant:.3f}Hz"
        )
        print(
            f"  RGB={row['rgb_fps']:.3f}Hz, Detection={row['detection_fps']:.3f}Hz, "
            f"Decision={row['servo_decision_fps']:.3f}Hz, IK={row['ik_command_fps']:.3f}Hz, "
            f"Non-zero correction={row['nonzero_correction_fps']:.3f}Hz"
        )

    levels = list(TRIAL_PATHS)
    for metric, label in (("rmse", "RMSE"), ("mean_error_norm", "Mean Error"), ("max_error_norm", "Max Error")):
        values = summary.loc[levels, metric].to_numpy(dtype=float)
        direction = "increases" if np.all(np.diff(values) > 0.0) else "does not increase monotonically"
        print(f"{label} with speed: {direction} ({', '.join(f'{value:.3f}' for value in values)} px)")
    horizontal_dominant = all(
        trials[level]["ex"].abs().mean() > trials[level]["ey"].abs().mean()
        for level in levels
    )
    print("Dominant pixel axis:", "ex (horizontal)" if horizontal_dominant else "mixed")
    fps_spread = float(summary["servo_decision_fps"].max() - summary["servo_decision_fps"].min())
    ik_spread = float(summary["ik_command_fps"].max() - summary["ik_command_fps"].min())
    print(
        f"FPS spread across trials: Decision={fps_spread:.3f}Hz, IK={ik_spread:.3f}Hz. "
        "Inspect the FPS figure for run-to-run stability."
    )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    trials = {level: _load_trial(level, path) for level, path in TRIAL_PATHS.items()}
    summary = _load_summary()
    figure_paths = [_plot_error_norm(trials)]
    figure_paths.extend(_plot_ex_ey(level, frame) for level, frame in trials.items())
    figure_paths.append(
        _grouped_bars(
            summary,
            ["mean_error_norm", "rmse", "percentile_95_error_norm"],
            ["Mean Error Norm", "RMSE", "95th Percentile Error"],
            "Stage 9 Dynamic Error Metrics",
            "Tracking Error (px)",
            "stage9_error_metrics.png",
        )
    )
    figure_paths.append(
        _grouped_bars(
            summary,
            ["rgb_fps", "detection_fps", "servo_decision_fps", "ik_command_fps", "nonzero_correction_fps"],
            ["RGB FPS", "Detection FPS", "Servo Decision FPS", "IK Command FPS", "Non-zero Correction FPS"],
            "Stage 9 Processing Frequencies",
            "Frequency (Hz)",
            "stage9_fps_comparison.png",
        )
    )
    figure_paths.append(_plot_max_error(summary))
    _report(summary, trials)
    print("\nGenerated PNG files:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
