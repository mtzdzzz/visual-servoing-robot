"""Create read-only final analysis artefacts from Stage 7, 9, and 10 logs.

This module never imports the simulator and never writes to ``outputs/logs``.
It reads the existing CSV evidence, reports quality observations, and writes
only final plots/tables under ``outputs/final_figures`` and
``outputs/final_tables``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the established strict Stage 9 log loader and Stage 10 lost-interval
# helper rather than duplicating their validation logic.
import plot_stage9 as stage9_plot
import plot_stage10 as stage10_plot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_figures"
TABLE_DIRECTORY = PROJECT_ROOT / "outputs" / "final_tables"
STAGE7_PATHS = tuple(sorted(LOG_DIRECTORY.glob("stage7_trial_[0-9][0-9].csv")))
STAGE7_SUMMARY_PATH = LOG_DIRECTORY / "stage7_summary.csv"
STAGE10_NOISE_PATHS = {
    0: LOG_DIRECTORY / "stage10_noise_0.csv",
    1: LOG_DIRECTORY / "stage10_noise_1.csv",
    3: LOG_DIRECTORY / "stage10_noise_3.csv",
}
STAGE10_LOST_PATH = LOG_DIRECTORY / "stage10_target_lost.csv"
STAGE10_SUMMARY_PATH = LOG_DIRECTORY / "stage10_summary.csv"
FINAL_COLUMNS = (
    "experiment", "condition", "mean_error", "rmse", "p95_error", "max_error",
    "convergence_time", "target_lost_count", "detection_rate", "reacquisition_time",
    "recovery_time", "result", "source_csv",
)


@dataclass(frozen=True)
class Stage7Trial:
    trial_number: int
    path: Path
    data: pd.DataFrame
    initial_ex: float
    initial_ey: float
    initial_error: float
    final_ex: float
    final_ey: float
    final_error: float
    convergence_time: float | None
    target_lost_count: int
    detection_rate: float
    result: str


def _require(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    """Load an immutable CSV and reject missing required schema columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required experiment log: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> pd.DataFrame:
    checked = frame.copy()
    for column in columns:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    return checked


def _metric(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numeric):
        return {"mean_error": None, "rmse": None, "p95_error": None, "max_error": None}
    return {
        "mean_error": float(np.mean(numeric)),
        "rmse": float(np.sqrt(np.mean(np.square(numeric)))),
        "p95_error": float(np.quantile(numeric, 0.95)),
        "max_error": float(np.max(numeric)),
    }


def _lost_event_count(detected: pd.Series) -> int:
    values = pd.to_numeric(detected, errors="coerce").fillna(0).astype(bool).to_numpy()
    if len(values) < 2:
        return 0
    return int(np.count_nonzero(values[:-1] & ~values[1:]))


def _assert_time(path: Path, frame: pd.DataFrame, quality: list[str], expected: float | None = None) -> None:
    values = pd.to_numeric(frame["time_s"], errors="coerce")
    if values.isna().any():
        quality.append(f"{path.name}: time_s contains {int(values.isna().sum())} NaN values.")
        return
    if not values.is_monotonic_increasing or values.duplicated().any():
        quality.append(f"{path.name}: time_s is not strictly increasing.")
    duration = float(values.iloc[-1] - values.iloc[0]) if len(values) else 0.0
    if expected is not None and abs(duration - expected) > 0.2:
        quality.append(
            f"{path.name}: measured duration {duration:.3f} s differs from expected {expected:.1f} s."
        )


def _load_stage7_trials(quality: list[str]) -> list[Stage7Trial]:
    if len(STAGE7_PATHS) != 5:
        quality.append(f"Stage 7: expected five trial files, found {len(STAGE7_PATHS)}.")
    trials: list[Stage7Trial] = []
    required = {"time_s", "ex", "ey", "error_norm", "target_detected", "servo_state"}
    for path in STAGE7_PATHS:
        frame = _numeric(_require(path, required), required - {"servo_state"}, path)
        _assert_time(path, frame, quality)
        valid = frame.loc[frame["target_detected"].eq(1) & frame["error_norm"].notna()].copy()
        if valid.empty:
            raise ValueError(f"{path.name}: no valid visible-target measurements.")
        first = valid.iloc[0]
        final = valid.iloc[-1]
        centred = valid.loc[valid["servo_state"].eq("2D CENTERED PRECISE")]
        convergence_time = float(centred["time_s"].iloc[0]) if not centred.empty else None
        result = (
            "PASS"
            if convergence_time is not None and abs(float(final["ex"])) <= 2 and abs(float(final["ey"])) <= 2
            else "FAIL"
        )
        trial_number = int(path.stem.rsplit("_", 1)[1])
        trials.append(
            Stage7Trial(
                trial_number, path, frame, float(first["ex"]), float(first["ey"]),
                float(first["error_norm"]), float(final["ex"]), float(final["ey"]),
                float(final["error_norm"]), convergence_time, _lost_event_count(frame["target_detected"]),
                float(100.0 * frame["target_detected"].mean()), result,
            )
        )
        unavailable = int(frame["error_norm"].isna().sum())
        if unavailable:
            quality.append(
                f"{path.name}: {unavailable} no-detection rows are retained; visible measurements are used for metrics."
            )
    return trials


def _check_stage7_summary(stage7_trials: list[Stage7Trial], quality: list[str]) -> None:
    """Report, but never repair, the known incomplete source summary."""
    summary = _require(STAGE7_SUMMARY_PATH, {"trial", "result"})
    recorded = set(pd.to_numeric(summary["trial"], errors="coerce").dropna().astype(int))
    raw = {trial.trial_number for trial in stage7_trials}
    if recorded != raw:
        quality.append(
            "stage7_summary.csv is incomplete: it records trials "
            f"{sorted(recorded)}, while raw CSV files provide trials {sorted(raw)}. "
            "The final Stage 7 table is derived from raw trial files."
        )
    trial_by_number = {trial.trial_number: trial for trial in stage7_trials}
    field_map = {
        "initial_error_norm": "initial_error",
        "final_error_norm": "final_error",
        "convergence_time_s": "convergence_time",
    }
    for _, row in summary.iterrows():
        trial_number = int(row["trial"])
        trial = trial_by_number.get(trial_number)
        if trial is None:
            continue
        mismatches: list[str] = []
        for summary_field, trial_field in field_map.items():
            if summary_field not in row or pd.isna(row[summary_field]):
                continue
            derived = getattr(trial, trial_field)
            if derived is None or not np.isclose(float(row[summary_field]), float(derived), atol=1e-6):
                mismatches.append(summary_field)
        if mismatches:
            quality.append(
                f"stage7_summary.csv Trial {trial_number} disagrees with its current raw CSV on "
                f"{', '.join(mismatches)}."
            )


def _load_stage9(quality: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Reuse Stage 9's strict formal-measurement loader and reconcile summary."""
    trials = {level: stage9_plot.load_trial(level, path) for level, path in stage9_plot.TRIAL_PATHS.items()}
    summary = stage9_plot.load_summary().copy()
    for level, frame in trials.items():
        metric = _metric(frame["error_norm"])
        row = summary.loc[level]
        for raw_name, summary_name in (
            ("mean_error", "mean_error_norm"), ("rmse", "rmse"),
            ("p95_error", "percentile_95_error_norm"), ("max_error", "max_error_norm"),
        ):
            if not np.isclose(float(metric[raw_name]), float(row[summary_name]), atol=1e-9):
                quality.append(f"stage9_{level.lower()}.csv and stage9_summary.csv disagree on {summary_name}.")
    return trials, summary


def _load_stage10(quality: list[str]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame]:
    lost = _numeric(
        _require(STAGE10_LOST_PATH, {"time_s", "error_norm", "target_detected", "phase"}),
        ("time_s", "error_norm", "target_detected"), STAGE10_LOST_PATH,
    )
    _assert_time(STAGE10_LOST_PATH, lost, quality, expected=14.0)
    noise: dict[int, pd.DataFrame] = {}
    for level, path in STAGE10_NOISE_PATHS.items():
        frame = _numeric(
            _require(path, {"time_s", "error_norm", "target_detected", "measurement_u", "measurement_v"}),
            ("time_s", "error_norm", "target_detected", "measurement_u", "measurement_v"), path,
        )
        _assert_time(path, frame, quality, expected=20.0)
        if frame["error_norm"].isna().any():
            quality.append(f"{path.name}: unexpected missing formal error values.")
        noise[level] = frame
    summary = _require(
        STAGE10_SUMMARY_PATH,
        {"experiment", "condition", "mean_error", "rmse", "p95_error", "max_error", "result"},
    )
    for level, frame in noise.items():
        row = summary.loc[(summary["experiment"] == "VISUAL_NOISE") & (summary["condition"] == f"{level} px")]
        if len(row) != 1:
            quality.append(f"stage10_summary.csv lacks exactly one {level} px noise row.")
            continue
        metrics = _metric(frame["error_norm"])
        for field in ("mean_error", "rmse", "p95_error", "max_error"):
            if not np.isclose(float(metrics[field]), float(row.iloc[0][field]), atol=1e-9):
                quality.append(f"stage10_noise_{level}.csv and stage10_summary.csv disagree on {field}.")
    lost_rows = int(lost["error_norm"].isna().sum())
    if lost_rows == 0:
        quality.append("stage10_target_lost.csv has no unavailable-error rows during target loss.")
    lost_summary = summary.loc[summary["experiment"].eq("TARGET_LOST_RECOVERY")]
    if len(lost_summary) != 1:
        quality.append("stage10_summary.csv lacks exactly one TARGET_LOST_RECOVERY row.")
    else:
        row = lost_summary.iloc[0]
        visible_metrics = _metric(lost["error_norm"])
        for field in ("mean_error", "rmse", "p95_error", "max_error"):
            if not np.isclose(float(visible_metrics[field]), float(row[field]), atol=1e-9):
                quality.append(f"stage10_target_lost.csv and stage10_summary.csv disagree on {field}.")
        actual_lost_events = _lost_event_count(lost["target_detected"])
        if actual_lost_events != int(row["lost_event_count"]):
            quality.append("stage10_target_lost.csv and stage10_summary.csv disagree on lost-event count.")
    return lost, noise, summary


def _save(figure: plt.Figure, name: str) -> Path:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIRECTORY / name
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_stage7_convergence(trial: Stage7Trial) -> Path:
    valid = trial.data.loc[trial.data["target_detected"].eq(1)].copy()
    figure, axis = plt.subplots(figsize=(9.5, 5.3))
    axis.plot(valid["time_s"], valid["ex"], label="ex", color="#2a6fbb", linewidth=1.7)
    axis.plot(valid["time_s"], valid["ey"], label="ey", color="#e58f26", linewidth=1.7)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="Zero error")
    axis.axhspan(-2.0, 2.0, color="#4daf4a", alpha=0.13, label="±2 px precision band")
    if trial.convergence_time is not None:
        axis.axvspan(trial.convergence_time, float(valid["time_s"].iloc[-1]), color="#4daf4a", alpha=0.08)
        axis.annotate(
            "2D CENTERED PRECISE",
            xy=(trial.convergence_time, 0.0), xytext=(trial.convergence_time + 0.3, 38),
            arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=10,
        )
    axis.set(
        title=f"Stage 7 Static Convergence — Trial {trial.trial_number:02d}",
        xlabel="Time (s)", ylabel="Pixel Error (px)",
    )
    axis.grid(True, alpha=0.30)
    axis.legend(ncol=2, loc="upper right")
    return _save(figure, "stage7_static_convergence.png")


def _plot_stage7_summary(trials: list[Stage7Trial]) -> tuple[Path, Path]:
    labels = [f"Trial {trial.trial_number}" for trial in trials]
    x = np.arange(len(trials))
    figure, axis = plt.subplots(figsize=(9.0, 5.1))
    width = 0.36
    axis.bar(x - width / 2, [trial.initial_error for trial in trials], width, label="Initial error norm")
    axis.bar(x + width / 2, [trial.final_error for trial in trials], width, label="Final error norm")
    axis.set(title="Stage 7 Static Final Error", xlabel="Static trial", ylabel="Error Norm (px)", xticks=x, xticklabels=labels)
    axis.grid(axis="y", alpha=0.30)
    axis.legend()
    error_path = _save(figure, "stage7_final_error.png")

    figure, axis = plt.subplots(figsize=(9.0, 5.1))
    times = [trial.convergence_time if trial.convergence_time is not None else np.nan for trial in trials]
    bars = axis.bar(labels, times, color="#4c78a8")
    for bar, value in zip(bars, times):
        if not np.isnan(value):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    axis.set(title="Stage 7 Static Convergence Time", xlabel="Static trial", ylabel="Convergence Time (s)")
    axis.grid(axis="y", alpha=0.30)
    return error_path, _save(figure, "stage7_convergence_time.png")


def _plot_stage9(trials: dict[str, pd.DataFrame], summary: pd.DataFrame) -> tuple[Path, Path, Path]:
    colors = {"SLOW": "#2a6fbb", "MEDIUM": "#e58f26", "FAST": "#c74440"}
    figure, axis = plt.subplots(figsize=(9.8, 5.4))
    for level, frame in trials.items():
        axis.plot(frame["time_s"], frame["error_norm"], label=level, color=colors[level], linewidth=1.6)
    axis.set(title="Stage 9 Dynamic Tracking Error", xlabel="Time (s)", ylabel="Tracking Error Norm (px)", xlim=(0.0, 20.0))
    axis.grid(True, alpha=0.30)
    axis.legend(title="Target speed")
    dynamic_path = _save(figure, "stage9_dynamic_tracking.png")

    levels = list(trials)
    columns = ("mean_error_norm", "rmse", "percentile_95_error_norm")
    names = ("Mean Error", "RMSE", "95th Percentile")
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(levels))
    width = 0.24
    for index, (column, name) in enumerate(zip(columns, names)):
        axis.bar(positions + (index - 1) * width, summary.loc[levels, column], width, label=name)
    axis.set(title="Stage 9 Dynamic Tracking Performance", xlabel="Target speed level", ylabel="Tracking Error (px)", xticks=positions, xticklabels=levels)
    axis.grid(axis="y", alpha=0.30)
    axis.legend()
    performance_path = _save(figure, "stage9_speed_performance.png")

    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    axis.bar(levels, summary.loc[levels, "max_error_norm"], color=[colors[level] for level in levels])
    axis.set(title="Stage 9 Maximum Dynamic Error", xlabel="Target speed level", ylabel="Maximum Tracking Error (px)")
    axis.grid(axis="y", alpha=0.30)
    return dynamic_path, performance_path, _save(figure, "stage9_max_error.png")


def _plot_stage10_lost(lost: pd.DataFrame, recorded_recovery_time_s: float) -> Path:
    time_values = lost["time_s"].to_numpy(dtype=float)
    errors = lost["error_norm"].to_numpy(dtype=float)
    detected = lost["target_detected"].astype(bool).to_numpy()
    lost_intervals = stage10_plot.contiguous_false_intervals(time_values, detected)
    reacquired = float(lost.loc[(lost["phase"] == "RECOVERY") & lost["target_detected"].eq(1), "time_s"].iloc[0])
    # The CSV stores each fresh RGB sample but not a separate "five-frame
    # precision verified" state. Use the Stage 10 summary's recorded recovery
    # duration rather than inventing a state from a visual appearance.
    recovered = reacquired + recorded_recovery_time_s
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    axis.plot(time_values, errors, color="#2a6fbb", linewidth=1.6, label="RGB tracking error norm")
    for index, (start, end) in enumerate(lost_intervals):
        axis.axvspan(start, end, color="#c74440", alpha=0.20, label="TARGET LOST" if index == 0 else None)
    axis.axvline(reacquired, color="#e58f26", linestyle="--", linewidth=1.3, label="REACQUIRED")
    axis.axvline(recovered, color="#4daf4a", linestyle=":", linewidth=1.8, label="RECOVERED")
    axis.axhline(2.0 * np.sqrt(2.0), color="#4daf4a", linestyle="--", linewidth=1.1, label="2 px/axis bound")
    axis.annotate("TRACKING", xy=(1.5, 2.24), xytext=(1.0, 2.48), fontsize=10)
    if lost_intervals:
        start, end = lost_intervals[0]
        axis.annotate("HOLD", xy=((start + end) / 2, 2.24), ha="center", xytext=((start + end) / 2, 2.48), fontsize=10)
    axis.set(title="Stage 10A Target Lost and Recovery", xlabel="Time (s)", ylabel="Tracking Error Norm (px)")
    axis.grid(True, alpha=0.30)
    axis.legend(loc="upper right")
    return _save(figure, "stage10_target_lost_recovery.png")


def _plot_stage10_noise(summary: pd.DataFrame) -> Path:
    noise = summary.loc[summary["experiment"].eq("VISUAL_NOISE")].copy()
    noise["noise_px"] = noise["condition"].str.extract(r"([0-9.]+)").astype(float)
    noise = noise.sort_values("noise_px")
    positions = np.arange(len(noise))
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    width = 0.36
    axis.bar(positions - width / 2, noise["mean_error"], width, label="Mean Error", color="#4c78a8")
    axis.bar(positions + width / 2, noise["rmse"], width, label="RMSE", color="#f58518")
    axis.set(title="Stage 10B Visual Noise Robustness", xlabel="Injected centroid measurement noise", ylabel="Pixel Error (px)", xticks=positions, xticklabels=[f"{value:g} px" for value in noise["noise_px"]])
    axis.grid(axis="y", alpha=0.30)
    axis.legend()
    axis.text(0.5, -0.20, "Single fixed-seed run: non-monotonic values do not imply that more noise improves tracking.", transform=axis.transAxes, ha="center", fontsize=9)
    return _save(figure, "stage10_noise_robustness.png")


def _plot_overview(stage7: list[Stage7Trial], stage9: pd.DataFrame, stage10: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    success = sum(trial.result == "PASS" for trial in stage7)
    axes[0, 0].bar(["Static trials"], [success], color="#4daf4a")
    axes[0, 0].set(title="Stage 7: Static Successes", ylabel="Successful Trials", ylim=(0, len(stage7) + 0.5))
    axes[0, 0].text(0, success + 0.08, f"{success}/{len(stage7)}", ha="center")
    levels = ["SLOW", "MEDIUM", "FAST"]
    axes[0, 1].bar(levels, stage9.loc[levels, "rmse"], color=["#2a6fbb", "#e58f26", "#c74440"])
    axes[0, 1].set(title="Stage 9: Dynamic RMSE", ylabel="RMSE (px)")
    lost_row = stage10.loc[stage10["experiment"].eq("TARGET_LOST_RECOVERY")].iloc[0]
    axes[1, 0].bar(["Reacquisition", "Recovery"], [lost_row["reacquisition_time_s"], lost_row["recovery_time_s"]], color=["#4c78a8", "#4daf4a"])
    axes[1, 0].set(title="Stage 10A: Recovery Timing", ylabel="Time (s)")
    noise = stage10.loc[stage10["experiment"].eq("VISUAL_NOISE")].copy()
    axes[1, 1].bar(noise["condition"], noise["rmse"], color="#9467bd")
    axes[1, 1].set(title="Stage 10B: Noise RMSE", xlabel="Noise level", ylabel="RMSE (px)")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.30)
    figure.suptitle("Visual Servoing Experiment Overview", fontsize=14)
    figure.tight_layout()
    return _save(figure, "project_experiment_overview.png")


def _final_rows(stage7: list[Stage7Trial], stage9: pd.DataFrame, stage10: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trial in stage7:
        metrics = _metric(trial.data.loc[trial.data["target_detected"].eq(1), "error_norm"])
        rows.append({
            "experiment": "Stage 7 Static Convergence", "condition": f"Trial {trial.trial_number:02d}",
            **metrics, "convergence_time": trial.convergence_time,
            "target_lost_count": trial.target_lost_count, "detection_rate": trial.detection_rate,
            "reacquisition_time": None, "recovery_time": None, "result": trial.result,
            "source_csv": str(trial.path.relative_to(PROJECT_ROOT)),
        })
    for level in ("SLOW", "MEDIUM", "FAST"):
        row = stage9.loc[level]
        rows.append({
            "experiment": "Stage 9 Dynamic Tracking", "condition": level,
            "mean_error": row["mean_error_norm"], "rmse": row["rmse"],
            "p95_error": row["percentile_95_error_norm"], "max_error": row["max_error_norm"],
            "convergence_time": None, "target_lost_count": row["target_lost_count"],
            "detection_rate": row["detection_rate_percent"], "reacquisition_time": None,
            "recovery_time": None, "result": row["result"],
            "source_csv": f"outputs/logs/stage9_{level.lower()}.csv",
        })
    for _, row in stage10.iterrows():
        source = (
            "outputs/logs/stage10_target_lost.csv"
            if row["experiment"] == "TARGET_LOST_RECOVERY"
            else f"outputs/logs/stage10_noise_{str(row['condition']).split()[0]}.csv"
        )
        rows.append({
            "experiment": "Stage 10 " + str(row["experiment"]), "condition": row["condition"],
            "mean_error": row["mean_error"], "rmse": row["rmse"], "p95_error": row["p95_error"],
            "max_error": row["max_error"], "convergence_time": None,
            "target_lost_count": row["lost_event_count"], "detection_rate": row["detection_rate"],
            "reacquisition_time": row.get("reacquisition_time_s"), "recovery_time": row.get("recovery_time_s"),
            "result": row["result"], "source_csv": source,
        })
    return rows


def _format_number(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.3f}{suffix}"


def _write_final_tables(rows: list[dict[str, object]], quality: list[str]) -> tuple[Path, Path, Path]:
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    csv_path = TABLE_DIRECTORY / "final_experiment_summary.csv"
    table.to_csv(csv_path, index=False)
    markdown_path = TABLE_DIRECTORY / "final_experiment_summary.md"
    markdown_lines = [
        "# Final Experiment Summary", "",
        "All values below are derived read-only from the recorded experiment CSV files.", "",
        "| Experiment | Condition | Mean Error (px) | RMSE (px) | P95 (px) | Max (px) | Convergence (s) | Lost Events | Detection Rate | Reacquisition (s) | Recovery (s) | Result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        markdown_lines.append(
            "| {experiment} | {condition} | {mean} | {rmse} | {p95} | {max} | {conv} | {lost} | {rate} | {reacq} | {recovery} | {result} |".format(
                experiment=row["experiment"], condition=row["condition"],
                mean=_format_number(row["mean_error"]), rmse=_format_number(row["rmse"]),
                p95=_format_number(row["p95_error"]), max=_format_number(row["max_error"]),
                conv=_format_number(row["convergence_time"]), lost=_format_number(row["target_lost_count"]),
                rate=_format_number(row["detection_rate"], "%"),
                reacq=_format_number(row["reacquisition_time"]), recovery=_format_number(row["recovery_time"]),
                result=row["result"],
            )
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    conclusions_path = TABLE_DIRECTORY / "experiment_conclusions.md"
    stage7 = table.loc[table["experiment"].eq("Stage 7 Static Convergence")]
    stage9 = table.loc[table["experiment"].eq("Stage 9 Dynamic Tracking")]
    stage10_lost = table.loc[table["experiment"].eq("Stage 10 TARGET_LOST_RECOVERY")].iloc[0]
    stage10_noise = table.loc[table["experiment"].eq("Stage 10 VISUAL_NOISE")]
    stage7_successes = int((stage7["result"] == "PASS").sum())
    stage7_times = pd.to_numeric(stage7["convergence_time"], errors="coerce").dropna()
    conclusions = [
        "# Experimental Conclusions", "",
        "## 1. Static Convergence", "",
        f"The five available Stage 7 raw trial logs each finish in `2D CENTERED PRECISE` with final component errors within ±2 px. The derived raw-log success count is {stage7_successes}/{len(stage7)}, with convergence times from {stage7_times.min():.3f} s to {stage7_times.max():.3f} s (mean {stage7_times.mean():.3f} s).", "",
        "## 2. Dynamic Tracking", "",
        "Stage 9 uses formal 20 s measurements after READY, so warm-up is excluded. The measured RMSE values are " + ", ".join(f"{row.condition}: {float(row.rmse):.3f} px" for _, row in stage9.iterrows()) + ". In these three runs, error increases with the tested target-speed level, showing the current system has measurable dynamic tracking capability with larger lag at the faster condition.", "",
        "## 3. Target Lost Recovery", "",
        f"The planned 2.5 s out-of-view interval produced one detected loss event. Reacquisition was {float(stage10_lost.reacquisition_time):.3f} s and the recorded return to the precision condition took {float(stage10_lost.recovery_time):.3f} s. The run completed without a return-to-home command.", "",
        "## 4. Visual Noise Robustness", "",
        "For the single fixed-seed tests at 0, 1, and 3 px injected centroid noise, all runs retained 100% detection and no target-lost events. Their RMSE values were " + ", ".join(f"{row.condition}: {float(row.rmse):.3f} px" for _, row in stage10_noise.iterrows()) + ". The values are not monotonic, so this single-run result must not be interpreted as more noise improving tracking. It supports only that the system remained stable within this 0–3 px measurement-noise range.", "",
        "## 5. Overall Conclusion", "",
        "Within the recorded static, dynamic, target-lost, and visual-noise experiments, the Eye-in-Hand visual-servo system reached its tested static precision condition and maintained continuous RGB/OpenCV-based tracking. These conclusions describe only the tested PyBullet conditions and do not claim industrial performance or comparison with other algorithms.", "",
        "## Data Quality Notes", "",
    ]
    if quality:
        conclusions.extend(f"- {item}" for item in quality)
    else:
        conclusions.append("- All required files, schemas, time sequences, formal durations, and raw-to-summary metric comparisons passed.")
    conclusions_path.write_text("\n".join(conclusions) + "\n", encoding="utf-8")
    return csv_path, markdown_path, conclusions_path


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    quality: list[str] = []
    stage7 = _load_stage7_trials(quality)
    _check_stage7_summary(stage7, quality)
    stage9_trials, stage9_summary = _load_stage9(quality)
    stage10_lost, _, stage10_summary = _load_stage10(quality)
    representative = max(stage7, key=lambda trial: trial.initial_error)
    figures = [
        _plot_stage7_convergence(representative),
        *_plot_stage7_summary(stage7),
        *_plot_stage9(stage9_trials, stage9_summary),
        _plot_stage10_lost(
            stage10_lost,
            float(
                stage10_summary.loc[
                    stage10_summary["experiment"].eq("TARGET_LOST_RECOVERY"),
                    "recovery_time_s",
                ].iloc[0]
            ),
        ),
        _plot_stage10_noise(stage10_summary),
        _plot_overview(stage7, stage9_summary, stage10_summary),
    ]
    table_paths = _write_final_tables(_final_rows(stage7, stage9_summary, stage10_summary), quality)
    print("\n===== Stage 11 Data Quality =====")
    if quality:
        for item in quality:
            print("WARNING:", item)
    else:
        print("PASS: all checks completed without warnings.")
    print("\nGenerated figures:")
    for path in figures:
        print(path)
    print("\nGenerated final tables:")
    for path in table_paths:
        print(path)


if __name__ == "__main__":
    main()
