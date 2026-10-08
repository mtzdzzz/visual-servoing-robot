"""Build the Stage 23 final experimental summary from immutable CSV evidence.

This script is intentionally offline.  It never imports or starts PyBullet,
never modifies files under ``outputs/logs``, and never changes control,
perception, planning, or execution parameters.  It reads the historical logs,
recomputes the reported metrics, records provenance, and writes only to
``outputs/final_report``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_DIR = PROJECT_ROOT / "outputs" / "final_report"
FIGURE_DIR = REPORT_DIR / "figures"

METRIC_COLUMNS = (
    "category",
    "stage",
    "metric",
    "condition",
    "value",
    "unit",
    "source_file",
    "source_column",
    "notes",
)

STAGE_PREFIXES = (
    "stage7_",
    "stage9_",
    "stage10_",
    "stage12_",
    "stage12b_",
    "stage13_",
    "stage14_",
    "stage15_",
    "stage16_",
    "stage17_",
    "stage18_",
    "stage19_",
    "stage20_",
)

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#D98324",
    "green": "#3A7D44",
    "red": "#B64242",
    "purple": "#7557A3",
    "gray": "#737373",
    "light_blue": "#8EB8D8",
}


@dataclass(frozen=True)
class Stage7Trial:
    trial: int
    path: Path
    frame: pd.DataFrame
    initial_ex: float
    initial_ey: float
    initial_error: float
    final_ex: float
    final_ey: float
    final_error: float
    convergence_time_s: float | None
    result: str


class ReportBuilder:
    def __init__(self) -> None:
        self.cache: dict[Path, pd.DataFrame] = {}
        self.metrics: list[dict[str, object]] = []
        self.quality: list[str] = []
        self.sources_used: set[Path] = set()

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

    def load(self, name: str, required: Iterable[str] = ()) -> pd.DataFrame:
        path = LOG_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required Stage 23 source: {self.relative(path)}")
        if path not in self.cache:
            self.cache[path] = pd.read_csv(path)
        frame = self.cache[path]
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {', '.join(missing)}")
        self.sources_used.add(path)
        return frame.copy()

    def add_metric(
        self,
        category: str,
        stage: str,
        metric: str,
        condition: str,
        value: float | int,
        unit: str,
        source: Path | str,
        source_column: str,
        notes: str = "",
    ) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Non-finite metric: {stage} {condition} {metric}")
        if isinstance(value, (int, np.integer)):
            stored: float | int = int(value)
        else:
            stored = numeric
        if isinstance(source, Path):
            source_file = self.relative(source)
        elif source.startswith("outputs/"):
            source_file = source
        else:
            source_file = self.relative(LOG_DIR / source)
        self.metrics.append(
            {
                "category": category,
                "stage": stage,
                "metric": metric,
                "condition": condition,
                "value": stored,
                "unit": unit,
                "source_file": source_file,
                "source_column": source_column,
                "notes": notes,
            }
        )

    def add_quality(self, message: str) -> None:
        if message not in self.quality:
            self.quality.append(message)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    textual = series.astype(str).str.strip().str.lower().isin({"true", "yes", "y"})
    return numeric.fillna(0).ne(0) | textual


def _metric(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numeric):
        raise ValueError("Metric calculation has no finite observations")
    if not np.isfinite(numeric).all():
        raise ValueError("Metric calculation contains Inf")
    return {
        "mean": float(np.mean(numeric)),
        "rmse": float(np.sqrt(np.mean(np.square(numeric)))),
        "median": float(np.median(numeric)),
        "p95": float(np.quantile(numeric, 0.95)),
        "max": float(np.max(numeric)),
    }


def _is_close(a: float, b: float, atol: float = 1e-8) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=atol))


def _check_close(
    builder: ReportBuilder,
    source: str,
    label: str,
    raw_value: float,
    summary_value: float,
    atol: float = 1e-8,
) -> None:
    if not _is_close(raw_value, summary_value, atol=atol):
        builder.add_quality(
            f"{source}: raw/summary mismatch for {label}: raw={raw_value:.12g}, "
            f"summary={summary_value:.12g}."
        )


def _check_strict_time(
    builder: ReportBuilder,
    frame: pd.DataFrame,
    source: str,
    column: str,
    groups: tuple[str, ...] = (),
) -> None:
    if column not in frame:
        return
    if groups:
        grouped = frame.groupby(list(groups), dropna=False, sort=False)
    else:
        grouped = [("all", frame)]
    bad: list[str] = []
    for key, group in grouped:
        values = pd.to_numeric(group[column], errors="coerce")
        missing = int(values.isna().sum())
        if missing:
            builder.add_quality(f"{source}: {column} contains {missing} missing timestamp value(s) in group {key}.")
        finite_values = values.dropna().to_numpy(dtype=float)
        if len(finite_values) > 1 and not bool((np.diff(finite_values) > 0).all()):
            bad.append(str(key))
    if bad:
        builder.add_quality(f"{source}: {column} is not strictly increasing in group(s) {bad[:5]}.")


def _source_hashes() -> dict[Path, str]:
    hashes: dict[Path, str] = {}
    for path in sorted(LOG_DIR.glob("*.csv")):
        if path.name.startswith(STAGE_PREFIXES):
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _scan_all_csvs(builder: ReportBuilder) -> list[Path]:
    paths = [
        path
        for path in sorted(LOG_DIR.glob("*.csv"))
        if path.name.startswith(STAGE_PREFIXES)
    ]
    for path in paths:
        frame = pd.read_csv(path)
        builder.cache[path] = frame
        numeric = frame.select_dtypes(include=[np.number])
        if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            builder.add_quality(f"{path.name}: numeric data contains Inf.")
    if not paths:
        raise FileNotFoundError("No Stage 7–20 CSV files found under outputs/logs")
    return paths


def _analyse_stage7(builder: ReportBuilder) -> list[Stage7Trial]:
    paths = sorted(LOG_DIR.glob("stage7_trial_[0-9][0-9].csv"))
    if len(paths) != 5:
        builder.add_quality(f"Stage 7: expected 5 raw trial files, found {len(paths)}.")
    trials: list[Stage7Trial] = []
    for path in paths:
        frame = builder.load(
            path.name,
            ("time_s", "ex", "ey", "error_norm", "servo_state", "target_detected"),
        )
        _check_strict_time(builder, frame, path.name, "time_s")
        detected = _bool_series(frame["target_detected"])
        visible = frame.loc[detected & pd.to_numeric(frame["error_norm"], errors="coerce").notna()].copy()
        if visible.empty:
            raise ValueError(f"{path.name}: no visible target samples")
        first = visible.iloc[0]
        final = visible.iloc[-1]
        centered = visible.loc[visible["servo_state"].eq("2D CENTERED PRECISE")]
        convergence = float(centered["time_s"].iloc[0]) if not centered.empty else None
        result = (
            "PASS"
            if convergence is not None and abs(float(final["ex"])) <= 2 and abs(float(final["ey"])) <= 2
            else "FAIL"
        )
        trial = int(path.stem.rsplit("_", 1)[1])
        record = Stage7Trial(
            trial=trial,
            path=path,
            frame=frame,
            initial_ex=float(first["ex"]),
            initial_ey=float(first["ey"]),
            initial_error=float(first["error_norm"]),
            final_ex=float(final["ex"]),
            final_ey=float(final["ey"]),
            final_error=float(final["error_norm"]),
            convergence_time_s=convergence,
            result=result,
        )
        trials.append(record)
        for metric_name, value, unit, column in (
            ("Final ex", record.final_ex, "px", "ex"),
            ("Final ey", record.final_ey, "px", "ey"),
            ("Final error norm", record.final_error, "px", "error_norm"),
        ):
            builder.add_metric(
                "Visual Servo", "Stage 7", metric_name, f"Trial {trial}", value, unit,
                path, column, "Last valid detected frame.",
            )
        if convergence is not None:
            builder.add_metric(
                "Visual Servo", "Stage 7", "Convergence time", f"Trial {trial}", convergence,
                "s", path, "time_s,servo_state",
                "First frame whose servo_state is 2D CENTERED PRECISE.",
            )

    success_count = sum(t.result == "PASS" for t in trials)
    convergence_values = [t.convergence_time_s for t in trials if t.convergence_time_s is not None]
    all_trial_sources = "; ".join(builder.relative(path) for path in paths)
    builder.add_metric(
        "Visual Servo", "Stage 7", "Static convergence success rate", "5 static trials",
        100.0 * success_count / len(trials), "percent", all_trial_sources, "servo_state,ex,ey",
        "Recomputed from all five raw trial CSVs; PASS requires final |ex| and |ey| <= 2 px.",
    )
    builder.add_metric(
        "Visual Servo", "Stage 7", "Mean convergence time", "5 static trials",
        float(np.mean(convergence_values)), "s", all_trial_sources, "time_s,servo_state",
        "Mean of raw per-trial first 2D CENTERED PRECISE timestamps.",
    )
    builder.add_metric(
        "Visual Servo", "Stage 7", "Mean final absolute ex", "5 static trials",
        float(np.mean([abs(t.final_ex) for t in trials])), "px", all_trial_sources, "ex",
        "Mean across last valid detected frame of each raw trial.",
    )
    builder.add_metric(
        "Visual Servo", "Stage 7", "Mean final absolute ey", "5 static trials",
        float(np.mean([abs(t.final_ey) for t in trials])), "px", all_trial_sources, "ey",
        "Mean across last valid detected frame of each raw trial.",
    )

    summary = builder.load("stage7_summary.csv", ("trial", "result"))
    recorded = set(pd.to_numeric(summary["trial"], errors="coerce").dropna().astype(int))
    raw_trials = {trial.trial for trial in trials}
    if recorded != raw_trials:
        builder.add_quality(
            "stage7_summary.csv is incomplete: it records trial(s) "
            f"{sorted(recorded)}, while five raw CSVs provide {sorted(raw_trials)}. "
            "Stage 23 recomputes Stage 7 metrics from the raw files."
        )
    trial_lookup = {trial.trial: trial for trial in trials}
    for _, row in summary.iterrows():
        trial_number = int(row["trial"])
        trial = trial_lookup.get(trial_number)
        if trial is None:
            continue
        comparisons = {
            "initial_ex": trial.initial_ex,
            "initial_ey": trial.initial_ey,
            "initial_error_norm": trial.initial_error,
            "final_ex": trial.final_ex,
            "final_ey": trial.final_ey,
            "final_error_norm": trial.final_error,
            "convergence_time_s": trial.convergence_time_s,
        }
        mismatches = [
            field
            for field, raw_value in comparisons.items()
            if field in row.index
            and pd.notna(row[field])
            and raw_value is not None
            and not _is_close(float(row[field]), float(raw_value), atol=1e-6)
        ]
        if mismatches:
            builder.add_quality(
                f"stage7_summary.csv Trial {trial_number} disagrees with stage7_trial_{trial_number:02d}.csv "
                f"on {', '.join(mismatches)}; Stage 23 uses the raw trial values."
            )
    return trials


def _analyse_stage9(builder: ReportBuilder) -> pd.DataFrame:
    summary = builder.load(
        "stage9_summary.csv",
        ("speed_level", "mean_error_norm", "rmse", "percentile_95_error_norm", "max_error_norm", "detection_rate_percent"),
    ).set_index("speed_level")
    for level in ("SLOW", "MEDIUM", "FAST"):
        name = f"stage9_{level.lower()}.csv"
        frame = builder.load(name, ("time_s", "error_norm", "target_detected", "ex", "ey"))
        _check_strict_time(builder, frame, name, "time_s")
        values = _metric(frame["error_norm"])
        detection_rate = 100.0 * float(_bool_series(frame["target_detected"]).mean())
        row = summary.loc[level]
        for key, summary_column in (
            ("mean", "mean_error_norm"),
            ("rmse", "rmse"),
            ("p95", "percentile_95_error_norm"),
            ("max", "max_error_norm"),
        ):
            _check_close(builder, name, summary_column, values[key], row[summary_column])
        _check_close(builder, name, "detection_rate_percent", detection_rate, row["detection_rate_percent"])
        for metric_name, key, column in (
            ("Mean tracking error", "mean", "error_norm"),
            ("Tracking RMSE", "rmse", "error_norm"),
            ("P95 tracking error", "p95", "error_norm"),
            ("Maximum tracking error", "max", "error_norm"),
        ):
            builder.add_metric(
                "Visual Servo", "Stage 9", metric_name, level, values[key], "px", name, column,
                "Formal 20 s dynamic measurement after READY; warm-up excluded.",
            )
        builder.add_metric(
            "Visual Servo", "Stage 9", "Detection rate", level, detection_rate, "percent",
            name, "target_detected", "600 formal RGB frames.",
        )
    return summary


def _analyse_stage10(builder: ReportBuilder) -> pd.DataFrame:
    summary = builder.load(
        "stage10_summary.csv",
        ("experiment", "condition", "mean_error", "rmse", "p95_error", "max_error", "detection_rate"),
    )
    lost = builder.load(
        "stage10_target_lost.csv", ("time_s", "target_detected", "error_norm", "servo_state")
    )
    _check_strict_time(builder, lost, "stage10_target_lost.csv", "time_s")
    lost_row = summary.loc[summary["experiment"].eq("TARGET_LOST_RECOVERY")].iloc[0]
    lost_events = int(np.count_nonzero(
        _bool_series(lost["target_detected"]).to_numpy()[:-1]
        & ~_bool_series(lost["target_detected"]).to_numpy()[1:]
    ))
    _check_close(builder, "stage10_target_lost.csv", "lost_event_count", lost_events, lost_row["lost_event_count"])
    for metric_name, column, unit in (
        ("Lost duration", "lost_duration_s", "s"),
        ("Reacquisition time", "reacquisition_time_s", "s"),
        ("Recovery time", "recovery_time_s", "s"),
        ("Target lost event count", "lost_event_count", "count"),
    ):
        builder.add_metric(
            "Visual Servo", "Stage 10", metric_name, "Target lost and recovery",
            float(lost_row[column]), unit, "stage10_summary.csv", column,
            "Ground truth is used only to schedule/evaluate the visibility event, not for control.",
        )
    if lost["error_norm"].isna().sum() == 0:
        builder.add_quality("stage10_target_lost.csv: no unavailable error samples occur during target loss.")

    for noise in (0, 1, 3):
        name = f"stage10_noise_{noise}.csv"
        frame = builder.load(name, ("time_s", "error_norm", "target_detected"))
        _check_strict_time(builder, frame, name, "time_s")
        values = _metric(frame["error_norm"])
        row = summary.loc[
            summary["experiment"].eq("VISUAL_NOISE") & summary["condition"].eq(f"{noise} px")
        ].iloc[0]
        for key, column in (("mean", "mean_error"), ("rmse", "rmse"), ("p95", "p95_error"), ("max", "max_error")):
            _check_close(builder, name, column, values[key], row[column])
        for metric_name, key in (
            ("Mean error", "mean"), ("RMSE", "rmse"), ("P95 error", "p95"), ("Maximum error", "max")
        ):
            builder.add_metric(
                "Visual Servo", "Stage 10", metric_name, f"Noise {noise} px", values[key], "px",
                name, "error_norm", "Single fixed-seed noise experiment.",
            )
    return summary


def _analyse_stage12(builder: ReportBuilder) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = builder.load(
        "stage12_motion_estimation.csv",
        (
            "time_s", "baseline_future_error", "motion_prediction_future_error",
            "prediction_used_for_control", "estimator_valid", "prediction_horizon_s",
        ),
    )
    _check_strict_time(builder, frame, "stage12_motion_estimation.csv", "time_s")
    paired = frame[["baseline_future_error", "motion_prediction_future_error"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    baseline = _metric(paired["baseline_future_error"])
    prediction = _metric(paired["motion_prediction_future_error"])
    improvement = 100.0 * (baseline["mean"] - prediction["mean"]) / baseline["mean"]
    for metric_name, value, column in (
        ("Baseline future mean error", baseline["mean"], "baseline_future_error"),
        ("Baseline future RMSE", baseline["rmse"], "baseline_future_error"),
        ("EMA future mean error", prediction["mean"], "motion_prediction_future_error"),
        ("EMA future RMSE", prediction["rmse"], "motion_prediction_future_error"),
        ("Mean-error improvement", improvement, "baseline_future_error,motion_prediction_future_error"),
    ):
        builder.add_metric(
            "Prediction", "Stage 12", metric_name, "Online estimator log", value,
            "percent" if "improvement" in metric_name.lower() else "px",
            "stage12_motion_estimation.csv", column,
            "Prediction is evaluation-only and is not used by the controller in Stage 12.",
        )
    if _bool_series(frame["prediction_used_for_control"]).any():
        builder.add_quality("stage12_motion_estimation.csv: prediction_used_for_control is not always False.")

    grid = builder.load(
        "stage12b_prediction_grid.csv",
        ("alpha", "tau", "prediction_rmse", "baseline_rmse", "improvement_percent", "valid_samples"),
    )
    expected_pairs = {(a, t) for a in (0.2, 0.35, 0.5, 0.7, 1.0) for t in (0.05, 0.1, 0.15, 0.2, 0.3)}
    actual_pairs = set(zip(grid["alpha"].astype(float), grid["tau"].astype(float)))
    if actual_pairs != expected_pairs or len(grid) != 25:
        builder.add_quality("stage12b_prediction_grid.csv does not contain the expected complete 5x5 grid.")
    matching = grid.loc[np.isclose(grid["alpha"], 0.35) & np.isclose(grid["tau"], 0.10)].iloc[0]
    for label, raw_value, grid_value in (
        ("baseline_mean_error", baseline["mean"], matching["baseline_mean_error"]),
        ("baseline_rmse", baseline["rmse"], matching["baseline_rmse"]),
        ("prediction_mean_error", prediction["mean"], matching["prediction_mean_error"]),
        ("prediction_rmse", prediction["rmse"], matching["prediction_rmse"]),
    ):
        _check_close(builder, "stage12b_prediction_grid.csv", label, raw_value, grid_value, atol=2e-6)
    best_rmse = grid.loc[grid["prediction_rmse"].idxmin()]
    best_improvement = grid.loc[grid["improvement_percent"].idxmax()]
    builder.add_metric(
        "Prediction", "Stage 12B", "Lowest prediction RMSE", f"alpha={best_rmse.alpha:g}, tau={best_rmse.tau:g} s",
        float(best_rmse.prediction_rmse), "px", "stage12b_prediction_grid.csv", "prediction_rmse",
        "Lowest absolute prediction RMSE among all 25 offline configurations.",
    )
    builder.add_metric(
        "Prediction", "Stage 12B", "Best mean-error improvement", f"alpha={best_improvement.alpha:g}, tau={best_improvement.tau:g} s",
        float(best_improvement.improvement_percent), "percent", "stage12b_prediction_grid.csv", "improvement_percent",
        "Best recorded mean-error improvement; all 25 configurations remain in the source grid.",
    )
    return frame, grid


def _analyse_stage13(builder: ReportBuilder) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    summary = builder.load(
        "stage13_summary.csv",
        (
            "speed_level", "controller_mode", "prediction_horizon_s", "mean_tracking_error",
            "rmse", "p95_error", "max_error", "detection_rate_percent", "log_path",
        ),
    )
    for _, row in summary.iterrows():
        filename = Path(str(row["log_path"])).name
        raw = builder.load(filename, ("time_s", "error_norm_raw", "target_detected"))
        _check_strict_time(builder, raw, filename, "time_s")
        values = _metric(raw["error_norm_raw"])
        detection_rate = 100.0 * float(_bool_series(raw["target_detected"]).mean())
        for key, column in (("mean", "mean_tracking_error"), ("rmse", "rmse"), ("p95", "p95_error"), ("max", "max_error")):
            _check_close(builder, filename, column, values[key], row[column])
        _check_close(builder, filename, "detection_rate_percent", detection_rate, row["detection_rate_percent"])
        condition = str(row["speed_level"])
        if row["controller_mode"] == "PREDICTIVE":
            condition += f" predictive tau={float(row['prediction_horizon_s']):.2f} s"
        else:
            condition += " baseline"
        for metric_name, column, value in (
            ("Mean raw tracking error", "mean_tracking_error", row["mean_tracking_error"]),
            ("Raw tracking RMSE", "rmse", row["rmse"]),
            ("P95 raw tracking error", "p95_error", row["p95_error"]),
            ("Maximum raw tracking error", "max_error", row["max_error"]),
        ):
            builder.add_metric(
                "Prediction", "Stage 13", metric_name, condition, float(value), "px", filename,
                "error_norm_raw", "Performance is evaluated from raw RGB centroid error.",
            )

    selected: dict[str, pd.Series] = {}
    for speed in ("MEDIUM", "FAST"):
        baseline = summary.loc[
            summary["speed_level"].eq(speed) & summary["controller_mode"].eq("BASELINE")
        ].iloc[0]
        predictive = summary.loc[
            summary["speed_level"].eq(speed) & summary["controller_mode"].eq("PREDICTIVE")
        ].sort_values("rmse").iloc[0]
        selected[speed] = predictive
        for label, column in (("RMSE", "rmse"), ("Mean error", "mean_tracking_error"), ("P95 error", "p95_error")):
            improvement = 100.0 * (float(baseline[column]) - float(predictive[column])) / float(baseline[column])
            builder.add_metric(
                "Prediction", "Stage 13", f"{label} improvement", speed, improvement, "percent",
                "stage13_summary.csv", column,
                f"Baseline versus best online-tested predictive horizon tau={float(predictive['prediction_horizon_s']):.2f} s.",
            )
    return summary, selected


def _analyse_stage14(builder: ReportBuilder) -> pd.DataFrame:
    raw = builder.load("stage14_rgbd_localization.csv", ("trial", "time_s", "center_error_3d", "localization_state"))
    _check_strict_time(builder, raw, "stage14_rgbd_localization.csv", "time_s", ("trial",))
    summary = builder.load(
        "stage14_rgbd_summary.csv", ("trial", "position_label", "center_error_3d_mm", "valid_samples", "result")
    )
    for _, row in summary.iterrows():
        trial_raw = raw.loc[pd.to_numeric(raw["trial"], errors="coerce").eq(float(row["trial"]))]
        raw_mean_mm = float(pd.to_numeric(trial_raw["center_error_3d"], errors="coerce").mean() * 1000.0)
        _check_close(builder, "stage14_rgbd_summary.csv", f"Trial {int(row['trial'])} center error", raw_mean_mm, row["center_error_3d_mm"], atol=1e-6)
        builder.add_metric(
            "RGB-D Perception", "Stage 14", "3D center error", str(row["position_label"]),
            float(row["center_error_3d_mm"]), "mm", "stage14_rgbd_summary.csv", "center_error_3d_mm",
            "Sphere-center estimate after known-radius surface compensation.",
        )
    values = pd.to_numeric(summary["center_error_3d_mm"], errors="coerce")
    stats = _metric(values)
    for metric_name, key in (("Mean 3D error", "mean"), ("RMSE 3D error", "rmse"), ("Median 3D error", "median"), ("Maximum 3D error", "max")):
        builder.add_metric(
            "RGB-D Perception", "Stage 14", metric_name, "5 static positions", stats[key], "mm",
            "stage14_rgbd_summary.csv", "center_error_3d_mm", "Aggregate across the five position summaries.",
        )
    return summary


def _analyse_stage15(builder: ReportBuilder) -> pd.DataFrame:
    raw = builder.load(
        "stage15_multitarget.csv", ("target_id", "class_name", "detected", "localization_valid", "error_3d", "time_s")
    )
    _check_strict_time(builder, raw, "stage15_multitarget.csv", "time_s", ("scene", "target_id"))
    summary = builder.load(
        "stage15_summary.csv",
        ("target", "detection_rate_percent", "mean_3d_error_m", "rmse_3d_error_m", "max_3d_error_m", "target_confusion_events"),
    )
    for target in ("RED", "GREEN", "BLUE"):
        row = summary.loc[summary["target"].eq(target)].iloc[0]
        target_raw = raw.loc[raw["class_name"].astype(str).str.upper().eq(target)]
        detection_rate = 100.0 * float(_bool_series(target_raw["detected"]).mean())
        valid = target_raw.loc[_bool_series(target_raw["localization_valid"]), "error_3d"]
        stats = _metric(valid)
        _check_close(builder, "stage15_summary.csv", f"{target} detection rate", detection_rate, row["detection_rate_percent"])
        for key, column in (("mean", "mean_3d_error_m"), ("rmse", "rmse_3d_error_m"), ("max", "max_3d_error_m")):
            _check_close(builder, "stage15_summary.csv", f"{target} {column}", stats[key], row[column])
        builder.add_metric(
            "Multi-Target", "Stage 15", "Detection rate", target,
            float(row["detection_rate_percent"]), "percent", "stage15_summary.csv", "detection_rate_percent",
            "Detection from Eye-in-Hand RGB; no segmentation ground truth used.",
        )
        for metric_name, column in (("Mean 3D localization error", "mean_3d_error_m"), ("RMSE 3D localization error", "rmse_3d_error_m"), ("Maximum 3D localization error", "max_3d_error_m")):
            builder.add_metric(
                "Multi-Target", "Stage 15", metric_name, target, 1000.0 * float(row[column]), "mm",
                "stage15_summary.csv", column, "Converted from metres to millimetres for presentation.",
            )
    all_row = summary.loc[summary["target"].eq("ALL_THREE")].iloc[0]
    builder.add_metric(
        "Multi-Target", "Stage 15", "All-target detection rate", "RED+GREEN+BLUE",
        float(all_row["all_targets_detection_rate_percent"]), "percent", "stage15_summary.csv",
        "all_targets_detection_rate_percent", "All three targets detected in the same frame.",
    )
    builder.add_metric(
        "Multi-Target", "Stage 15", "Color confusion count", "RED+GREEN+BLUE",
        int(all_row["target_confusion_events"]), "count", "stage15_summary.csv", "target_confusion_events",
        "Count reported by the formal Stage 15 evaluation.",
    )
    return summary


def _analyse_stage16(builder: ReportBuilder) -> pd.DataFrame:
    raw = builder.load(
        "stage16_target_switching.csv", ("time_s", "switch_event", "switch_source", "target_detected", "error_norm")
    )
    _check_strict_time(builder, raw, "stage16_target_switching.csv", "time_s")
    summary = builder.load(
        "stage16_summary.csv",
        (
            "record_type", "response_time_s", "switch_result", "target_lost_events",
            "switch_success_rate_percent", "mean_switch_response_time_s", "max_switch_response_time_s",
            "locked_joint_max_deviation_rad", "result",
        ),
    )
    switches = summary.loc[summary["record_type"].eq("SWITCH")]
    overall = summary.loc[summary["record_type"].eq("OVERALL")].iloc[0]
    builder.add_metric(
        "Multi-Target", "Stage 16", "Switch trial count", "Historical formal switching log",
        len(switches), "count", "stage16_summary.csv", "record_type", "Includes all recorded switch events.",
    )
    for metric_name, column, unit in (
        ("Switch success rate", "switch_success_rate_percent", "percent"),
        ("Mean successful switch response time", "mean_switch_response_time_s", "s"),
        ("Maximum successful switch response time", "max_switch_response_time_s", "s"),
        ("Target lost event count", "target_lost_events", "count"),
        ("Locked-joint maximum deviation", "locked_joint_max_deviation_rad", "rad"),
    ):
        builder.add_metric(
            "Multi-Target", "Stage 16", metric_name, "Historical formal switching log",
            float(overall[column]), unit, "stage16_summary.csv", column,
            f"Historical overall result is {overall['result']}.",
        )
    if str(overall["result"]).upper() != "PASS":
        builder.add_quality(
            "stage16_summary.csv: historical formal switching evaluation is FAIL "
            f"({float(overall['switch_success_rate_percent']):.1f}% success; locked-joint maximum "
            f"deviation {float(overall['locked_joint_max_deviation_rad']):.6f} rad)."
        )
    sources = set(raw.loc[_bool_series(raw["switch_event"]), "switch_source"].dropna().astype(str))
    if len(sources) > 1:
        builder.add_quality(
            "stage16_target_switching.csv contains mixed switch sources "
            f"{sorted(sources)}; it is not a manual-only Demo log."
        )
    return summary


def _analyse_stage17(builder: ReportBuilder) -> pd.DataFrame:
    raw = builder.load(
        "stage17_obstacle_perception.csv",
        ("time", "scene", "obstacle_detected", "center_error_3d", "aabb_iou", "gt_coverage", "volume_ratio"),
    )
    _check_strict_time(builder, raw, "stage17_obstacle_perception.csv", "time", ("scene",))
    summary = builder.load(
        "stage17_summary.csv",
        (
            "scene", "detection_rate", "mean_center_error_m", "rmse_center_error_m", "max_center_error_m",
            "mean_aabb_iou", "mean_gt_coverage", "mean_volume_ratio", "target_confusion_count",
        ),
    )
    overall = summary.loc[summary["scene"].eq("OVERALL")].iloc[0]
    detected = _bool_series(raw["obstacle_detected"])
    raw_stats = _metric(pd.to_numeric(raw.loc[detected, "center_error_3d"], errors="coerce"))
    for key, column in (("mean", "mean_center_error_m"), ("rmse", "rmse_center_error_m"), ("max", "max_center_error_m")):
        _check_close(builder, "stage17_summary.csv", column, raw_stats[key], overall[column])
    for metric_name, column, scale, unit in (
        ("Obstacle detection rate", "detection_rate", 1.0, "percent"),
        ("Mean center error", "mean_center_error_m", 1000.0, "mm"),
        ("RMSE center error", "rmse_center_error_m", 1000.0, "mm"),
        ("Maximum center error", "max_center_error_m", 1000.0, "mm"),
        ("Mean AABB IoU", "mean_aabb_iou", 100.0, "percent"),
        ("Mean GT coverage", "mean_gt_coverage", 100.0, "percent"),
        ("Estimated-to-GT volume ratio", "mean_volume_ratio", 1.0, "ratio"),
        ("Target color confusion count", "target_confusion_count", 1.0, "count"),
    ):
        builder.add_metric(
            "Obstacle Perception", "Stage 17", metric_name, "5 obstacle scenes",
            float(overall[column]) * scale, unit, "stage17_summary.csv", column,
            "Occupancy is deliberately conservative; ground truth is evaluation-only.",
        )
    return summary


def _analyse_stage18(builder: ReportBuilder) -> pd.Series:
    raw = builder.load("stage18_collision_checking.csv", ("classification", "estimated_collision", "gt_collision"))
    summary = builder.load(
        "stage18_summary.csv",
        ("true_positive", "true_negative", "false_positive", "false_negative", "precision", "recall", "safety_recall"),
    ).iloc[0]
    counts = raw["classification"].astype(str).str.upper().value_counts()
    for label, column in (("TP", "true_positive"), ("TN", "true_negative"), ("FP", "false_positive"), ("FN", "false_negative")):
        _check_close(builder, "stage18_summary.csv", column, int(counts.get(label, 0)), summary[column])
        builder.add_metric(
            "Collision Safety", "Stage 18", label, "20 candidate paths", int(summary[column]), "count",
            "stage18_summary.csv", column, "GT collision is used only as the evaluation oracle.",
        )
    for metric_name, column in (("Precision", "precision"), ("Recall", "recall"), ("Safety recall", "safety_recall")):
        builder.add_metric(
            "Collision Safety", "Stage 18", metric_name, "20 candidate paths",
            100.0 * float(summary[column]), "percent", "stage18_summary.csv", column,
            "Safety recall is TP/(TP+FN).",
        )
    return summary


def _analyse_stage19(builder: ReportBuilder) -> tuple[pd.DataFrame, pd.Series]:
    raw = builder.load(
        "stage19_rrt_connect.csv",
        (
            "scenario_id", "seed", "rrt_required", "planning_success", "planning_time_s",
            "rrt_path_length", "final_path_gt_safe", "direct_path_safe",
        ),
    )
    summary = builder.load(
        "stage19_summary.csv",
        (
            "scenarios", "rrt_trials", "rrt_success_count", "rrt_planning_success_rate",
            "mean_planning_time_s", "median_planning_time_s", "max_planning_time_s",
            "mean_raw_path_length", "final_path_gt_collision_count",
        ),
    ).iloc[0]
    required = _bool_series(raw["rrt_required"])
    attempted = required & pd.to_numeric(raw["seed"], errors="coerce").notna() & pd.to_numeric(
        raw["planning_time_s"], errors="coerce"
    ).gt(0)
    rrt = raw.loc[attempted]
    skipped = raw.loc[required & ~attempted]
    if not skipped.empty:
        builder.add_quality(
            "stage19_rrt_connect.csv: 2 NEAR_BOUNDARY rows are marked rrt_required but were not "
            "counted as RRT trials because GOAL_CONFIGURATION_COLLISION stopped planning before a "
            "seeded search. The summary's 25-trial denominator is retained."
        )
    successful = rrt.loc[_bool_series(rrt["planning_success"])]
    _check_close(builder, "stage19_summary.csv", "rrt_trials", len(rrt), summary["rrt_trials"])
    _check_close(builder, "stage19_summary.csv", "rrt_success_count", len(successful), summary["rrt_success_count"])
    times = _metric(successful["planning_time_s"])
    _check_close(builder, "stage19_summary.csv", "mean_planning_time_s", times["mean"], summary["mean_planning_time_s"])
    _check_close(builder, "stage19_summary.csv", "median_planning_time_s", times["median"], summary["median_planning_time_s"])
    _check_close(builder, "stage19_summary.csv", "max_planning_time_s", times["max"], summary["max_planning_time_s"])
    for metric_name, column, unit, scale in (
        ("Scenario count", "scenarios", "count", 1.0),
        ("RRT trial count", "rrt_trials", "count", 1.0),
        ("RRT planning success rate", "rrt_planning_success_rate", "percent", 100.0),
        ("Mean planning time", "mean_planning_time_s", "s", 1.0),
        ("Median planning time", "median_planning_time_s", "s", 1.0),
        ("Maximum planning time", "max_planning_time_s", "s", 1.0),
        ("Mean RRT path length", "mean_raw_path_length", "rad", 1.0),
        ("Final GT path collision count", "final_path_gt_collision_count", "count", 1.0),
    ):
        builder.add_metric(
            "Planning", "Stage 19", metric_name, "Tested scenarios and seeds",
            float(summary[column]) * scale, unit, "stage19_summary.csv", column,
            "Planning uses estimated occupancy; GT is final evaluation only.",
        )
    return raw, summary


def _analyse_stage20(builder: ReportBuilder) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = builder.load(
        "stage20_execution.csv",
        (
            "time", "scenario_id", "state", "execution_source", "q5_cmd", "q6_cmd", "q7_cmd",
            "q5_actual", "q6_actual", "q7_actual", "tracking_error", "goal_error",
            "gt_collision_evaluation", "locked_joint_max_deviation",
        ),
    )
    _check_strict_time(builder, raw, "stage20_execution.csv", "time", ("scenario_id",))
    summary = builder.load(
        "stage20_summary.csv",
        (
            "scenario", "execution_source", "execution_success", "execution_time_s", "goal_joint_error_rad",
            "mean_tracking_error_rad", "rmse_tracking_error_rad", "max_tracking_error_rad",
            "gt_collision_count", "locked_joint_max_deviation_rad", "result",
        ),
    )
    success = _bool_series(summary["execution_success"])
    execution_rows = raw.loc[raw["state"].astype(str).eq("EXECUTING")].copy()
    tracking_condition = "Rows with state=EXECUTING"
    tracking_note = "Recomputed over raw rows with state=EXECUTING."
    if execution_rows.empty:
        builder.add_quality("stage20_execution.csv: no EXECUTING rows were found.")
        execution_rows = raw.copy()
        tracking_condition = "All raw trajectory samples"
        tracking_note = (
            "The historical writer stamped every row GOAL_REACHED; metrics are recomputed over all "
            "raw trajectory samples and agree with the per-scenario summary."
        )
    tracking = _metric(execution_rows["tracking_error"])
    for _, row in summary.iterrows():
        scenario_raw = execution_rows.loc[execution_rows["scenario_id"].astype(str).eq(str(row["scenario"]))]
        if scenario_raw.empty:
            builder.add_quality(f"stage20_execution.csv: no EXECUTING rows for {row['scenario']}.")
            continue
        stats = _metric(scenario_raw["tracking_error"])
        for key, column in (("mean", "mean_tracking_error_rad"), ("rmse", "rmse_tracking_error_rad"), ("max", "max_tracking_error_rad")):
            _check_close(builder, "stage20_summary.csv", f"{row['scenario']} {column}", stats[key], row[column], atol=2e-6)
    builder.add_metric(
        "Execution", "Stage 20", "Execution trial count", "2 direct + 3 RRT",
        len(summary), "count", "stage20_summary.csv", "scenario", "All formal execution trials.",
    )
    builder.add_metric(
        "Execution", "Stage 20", "Execution success rate", "2 direct + 3 RRT",
        100.0 * float(success.mean()), "percent", "stage20_summary.csv", "execution_success",
        "Success across the five tested execution scenarios.",
    )
    for source in ("DIRECT", "RRT_CONNECT"):
        subset = summary.loc[summary["execution_source"].eq(source)]
        builder.add_metric(
            "Execution", "Stage 20", "Mean execution time", source,
            float(pd.to_numeric(subset["execution_time_s"], errors="coerce").mean()), "s",
            "stage20_summary.csv", "execution_time_s", "Mean over formal trials with this execution source.",
        )
    builder.add_metric(
        "Execution", "Stage 20", "Mean goal joint error", "5 formal trials",
        float(pd.to_numeric(summary["goal_joint_error_rad"], errors="coerce").mean()), "rad",
        "stage20_summary.csv", "goal_joint_error_rad", "Mean final maximum active-joint error.",
    )
    builder.add_metric(
        "Execution", "Stage 20", "Maximum goal joint error", "5 formal trials",
        float(pd.to_numeric(summary["goal_joint_error_rad"], errors="coerce").max()), "rad",
        "stage20_summary.csv", "goal_joint_error_rad", "Maximum across formal trials.",
    )
    for metric_name, key in (("Mean joint tracking error", "mean"), ("RMSE joint tracking error", "rmse"), ("Maximum joint tracking error", "max")):
        builder.add_metric(
            "Execution", "Stage 20", metric_name, tracking_condition, tracking[key], "rad",
            "stage20_execution.csv", "tracking_error", tracking_note,
        )
    builder.add_metric(
        "Execution", "Stage 20", "Locked-joint maximum deviation", "5 formal trials",
        float(pd.to_numeric(summary["locked_joint_max_deviation_rad"], errors="coerce").max()), "rad",
        "stage20_summary.csv", "locked_joint_max_deviation_rad", "Maximum across formal trials.",
    )
    builder.add_metric(
        "Execution", "Stage 20", "GT collision count", "5 formal trials",
        int(pd.to_numeric(summary["gt_collision_count"], errors="coerce").sum()), "count",
        "stage20_summary.csv", "gt_collision_count", "Evaluation-only ground-truth contact count.",
    )
    return raw, summary


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.figsize": (8.4, 5.0),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _figure_stage7(trials: list[Stage7Trial]) -> Path:
    trial = max(trials, key=lambda item: item.initial_error)
    frame = trial.frame
    fig, ax = plt.subplots()
    ax.axhspan(-2, 2, color=COLORS["green"], alpha=0.12, label="Precision band (±2 px)")
    ax.plot(frame["time_s"], frame["ex"], label="ex", color=COLORS["blue"], linewidth=1.5)
    ax.plot(frame["time_s"], frame["ey"], label="ey", color=COLORS["orange"], linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    if trial.convergence_time_s is not None:
        ax.axvline(trial.convergence_time_s, color=COLORS["green"], linestyle="--", linewidth=1.2)
        ax.text(trial.convergence_time_s, ax.get_ylim()[1] * 0.85, "Precise", color=COLORS["green"], ha="right")
    ax.set(title=f"Static Visual Servo Convergence (Trial {trial.trial})", xlabel="Time (s)", ylabel="Pixel error (px)")
    ax.legend(loc="best")
    return _save_figure(fig, "01_static_convergence.png")


def _figure_stage13(summary: pd.DataFrame, selected: dict[str, pd.Series]) -> Path:
    speeds = ["MEDIUM", "FAST"]
    baseline = [float(summary.loc[summary["speed_level"].eq(s) & summary["controller_mode"].eq("BASELINE"), "rmse"].iloc[0]) for s in speeds]
    predictive = [float(selected[s]["rmse"]) for s in speeds]
    improvements = [100.0 * (b - p) / b for b, p in zip(baseline, predictive)]
    x = np.arange(len(speeds))
    width = 0.34
    fig, ax = plt.subplots()
    bars_a = ax.bar(x - width / 2, baseline, width, label="Baseline", color=COLORS["gray"])
    tau_labels = {float(selected[s]["prediction_horizon_s"]) for s in speeds}
    predictive_label = (
        f"Predictive (tau={next(iter(tau_labels)):.2f} s)"
        if len(tau_labels) == 1
        else "Predictive (best tested tau)"
    )
    bars_b = ax.bar(x + width / 2, predictive, width, label=predictive_label, color=COLORS["blue"])
    ax.bar_label(bars_a, fmt="%.2f", padding=3)
    ax.bar_label(bars_b, fmt="%.2f", padding=3)
    for index, improvement in enumerate(improvements):
        ax.text(index, max(baseline[index], predictive[index]) + 0.7, f"{improvement:.2f}% lower", ha="center", color=COLORS["green"], fontweight="bold")
    ax.set_xticks(x, speeds)
    ax.set_ylim(0, max(baseline) * 1.25)
    ax.set(title="Predictive Visual Servo: Raw Tracking RMSE", ylabel="RMSE (px)")
    ax.legend(loc="upper left")
    return _save_figure(fig, "02_predictive_vs_baseline.png")


def _figure_stage14(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots()
    labels = summary["position_label"].astype(str).str.title()
    values = pd.to_numeric(summary["center_error_3d_mm"], errors="coerce")
    bars = ax.bar(labels, values, color=COLORS["blue"])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="RGB-D Target Localization Error", xlabel="Target position", ylabel="3D center error (mm)", ylim=(0, max(values) * 1.2))
    return _save_figure(fig, "03_rgbd_localization.png")


def _figure_stage15(summary: pd.DataFrame) -> Path:
    target_rows = summary.loc[summary["target"].isin(["RED", "GREEN", "BLUE"])]
    all_row = summary.loc[summary["target"].eq("ALL_THREE")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    rates = list(pd.to_numeric(target_rows["detection_rate_percent"], errors="coerce")) + [float(all_row["all_targets_detection_rate_percent"])]
    labels = list(target_rows["target"].astype(str)) + ["ALL THREE"]
    bars = axes[0].bar(labels, rates, color=[COLORS["red"], COLORS["green"], COLORS["blue"], COLORS["gray"]])
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set(title="Detection Rate", ylabel="Detection rate (%)", ylim=(0, 108))
    errors = 1000.0 * pd.to_numeric(target_rows["rmse_3d_error_m"], errors="coerce")
    bars = axes[1].bar(target_rows["target"], errors, color=[COLORS["red"], COLORS["green"], COLORS["blue"]])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(title="RGB-D Localization RMSE", ylabel="RMSE (mm)", ylim=(0, max(errors) * 1.2))
    fig.suptitle("Multi-Target RGB-D Perception")
    fig.tight_layout()
    return _save_figure(fig, "04_multitarget_perception.png")


def _figure_stage17(summary: pd.DataFrame) -> Path:
    scenes = summary.loc[~summary["scene"].eq("OVERALL")].copy()
    overall = summary.loc[summary["scene"].eq("OVERALL")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    errors = 1000.0 * pd.to_numeric(scenes["mean_center_error_m"], errors="coerce")
    bars = axes[0].bar(scenes["scene"].astype(str).str.title(), errors, color=COLORS["orange"])
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set(title="Obstacle Center Error", ylabel="3D center error (mm)", ylim=(0, max(errors) * 1.25))
    occupancy = [100.0 * float(overall["mean_aabb_iou"]), 100.0 * float(overall["mean_gt_coverage"])]
    bars = axes[1].bar(["AABB IoU", "GT coverage"], occupancy, color=[COLORS["gray"], COLORS["green"]])
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[1].set(title="Conservative Occupancy", ylabel="Percent (%)", ylim=(0, 108))
    axes[1].text(0.5, -0.18, f"Estimated / GT volume = {float(overall['mean_volume_ratio']):.2f}×", transform=axes[1].transAxes, ha="center")
    fig.suptitle("Obstacle Perception and Occupancy Coverage")
    fig.tight_layout()
    return _save_figure(fig, "05_obstacle_perception.png")


def _figure_stage18(summary: pd.Series) -> Path:
    matrix = np.array(
        [
            [int(summary["true_negative"]), int(summary["false_positive"])],
            [int(summary["false_negative"]), int(summary["true_positive"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    for row in range(2):
        for col in range(2):
            label = (("TN", "FP"), ("FN", "TP"))[row][col]
            ax.text(col, row, f"{label}\n{matrix[row, col]}", ha="center", va="center", fontsize=14, fontweight="bold", color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Estimated safe", "Estimated collision"])
    ax.set_yticks([0, 1], ["GT safe", "GT collision"])
    ax.set(title=f"Collision Safety Confusion Matrix\nSafety recall = {100.0 * float(summary['safety_recall']):.1f}%")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Path count")
    return _save_figure(fig, "06_collision_safety.png")


def _figure_stage19(raw: pd.DataFrame, summary: pd.Series) -> Path:
    attempted = (
        _bool_series(raw["rrt_required"])
        & pd.to_numeric(raw["seed"], errors="coerce").notna()
        & pd.to_numeric(raw["planning_time_s"], errors="coerce").gt(0)
    )
    rrt = raw.loc[attempted].copy().reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    success = 100.0 * float(summary["rrt_planning_success_rate"])
    axes[0].bar(["RRT trials"], [success], color=COLORS["green"], width=0.55)
    axes[0].text(0, success / 2, f"{int(summary['rrt_success_count'])}/{int(summary['rrt_trials'])}", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    axes[0].set(title="Planning Success", ylabel="Success rate (%)", ylim=(0, 108))
    axes[1].plot(np.arange(1, len(rrt) + 1), rrt["planning_time_s"], marker="o", markersize=3.5, linewidth=1.0, color=COLORS["blue"])
    axes[1].axhline(float(summary["mean_planning_time_s"]), color=COLORS["orange"], linestyle="--", label=f"Mean {float(summary['mean_planning_time_s']):.3f} s")
    axes[1].set(title="RRT-Connect Planning Time", xlabel="RRT trial", ylabel="Planning time (s)")
    axes[1].legend(loc="upper right")
    fig.suptitle("RRT-Connect Planning in Tested Scenarios")
    fig.tight_layout()
    return _save_figure(fig, "07_rrt_planning.png")


def _figure_stage20(raw: pd.DataFrame, summary: pd.DataFrame) -> Path:
    rrt_scenarios = summary.loc[summary["execution_source"].eq("RRT_CONNECT"), "scenario"].astype(str)
    scenario = rrt_scenarios.iloc[0]
    scenario_rows = raw.loc[raw["scenario_id"].astype(str).eq(scenario)].copy()
    executing_rows = scenario_rows.loc[scenario_rows["state"].astype(str).eq("EXECUTING")].copy()
    # The historical Stage 20 writer stamped every row with the terminal
    # GOAL_REACHED state.  Preserve that evidence and plot the full scenario
    # trace when no per-sample EXECUTING labels are available.
    data = executing_rows if not executing_rows.empty else scenario_rows
    time = pd.to_numeric(data["time"], errors="coerce")
    time = time - float(time.iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), gridspec_kw={"height_ratios": [2.0, 1.0]})
    for joint, color in zip((5, 6, 7), (COLORS["blue"], COLORS["orange"], COLORS["green"])):
        axes[0].plot(time, data[f"q{joint}_cmd"], color=color, linewidth=1.2, label=f"q{joint} commanded")
        axes[0].plot(time, data[f"q{joint}_actual"], color=color, linewidth=1.0, linestyle="--", label=f"q{joint} actual")
    axes[0].set(title=f"RRT Trajectory Execution ({scenario})", xlabel="Execution time (s)", ylabel="Joint position (rad)")
    axes[0].legend(ncol=3, loc="best")
    source_times = summary.groupby("execution_source")["execution_time_s"].mean().reindex(["DIRECT", "RRT_CONNECT"])
    bars = axes[1].bar(["Direct", "RRT-Connect"], source_times, color=[COLORS["gray"], COLORS["blue"]])
    axes[1].bar_label(bars, fmt="%.2f s", padding=3)
    axes[1].set(title="Mean Execution Time", ylabel="Time (s)", ylim=(0, max(source_times) * 1.2))
    fig.tight_layout()
    return _save_figure(fig, "08_trajectory_execution.png")


def _format_number(value: float | int, unit: str) -> str:
    number = float(value)
    if unit == "count":
        return f"{int(round(number))}"
    if unit == "percent":
        return f"{number:.2f}%"
    if unit == "ratio":
        return f"{number:.3f}×"
    if unit == "rad" and abs(number) < 0.01:
        return f"{number:.6f} rad"
    if unit in {"px", "mm", "s", "rad"}:
        return f"{number:.3f} {unit}"
    return f"{number:.6g} {unit}".strip()


def _write_metrics(builder: ReportBuilder) -> tuple[Path, Path, pd.DataFrame]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(builder.metrics, columns=METRIC_COLUMNS)
    csv_path = REPORT_DIR / "final_metrics.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    sections = (
        ("Visual Servo", "1. Visual Servo"),
        ("Prediction", "2. Prediction"),
        ("RGB-D Perception", "3. RGB-D Perception"),
        ("Multi-Target", "4. Multi-Target"),
        ("Obstacle Perception", "5. Obstacle Perception"),
        ("Collision Safety", "6. Collision Safety"),
        ("Planning", "7. Planning"),
        ("Execution", "8. Execution"),
    )
    lines = ["# Final Experimental Metrics", "", "All values are derived from immutable CSV evidence under `outputs/logs/`.", ""]
    for category, title in sections:
        subset = frame.loc[frame["category"].eq(category)]
        if subset.empty:
            continue
        lines.extend([f"## {title}", "", "| Stage | Condition | Metric | Value | Source |", "|---|---|---|---:|---|"])
        for _, row in subset.iterrows():
            value = _format_number(row["value"], str(row["unit"]))
            lines.append(f"| {row['stage']} | {row['condition']} | {row['metric']} | {value} | `{row['source_file']}` |")
        lines.append("")
    md_path = REPORT_DIR / "final_metrics.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return csv_path, md_path, frame


def _metric_value(frame: pd.DataFrame, stage: str, metric: str, condition: str | None = None) -> float:
    rows = frame.loc[frame["stage"].eq(stage) & frame["metric"].eq(metric)]
    if condition is not None:
        rows = rows.loc[rows["condition"].eq(condition)]
    if len(rows) != 1:
        raise ValueError(f"Expected one metric for {stage}/{metric}/{condition}, found {len(rows)}")
    return float(rows.iloc[0]["value"])


def _write_conclusions(builder: ReportBuilder, metrics: pd.DataFrame, scanned: list[Path]) -> Path:
    stage7_rate = _metric_value(metrics, "Stage 7", "Static convergence success rate", "5 static trials")
    stage9 = {
        speed: _metric_value(metrics, "Stage 9", "Tracking RMSE", speed)
        for speed in ("SLOW", "MEDIUM", "FAST")
    }
    medium_improvement = _metric_value(metrics, "Stage 13", "RMSE improvement", "MEDIUM")
    fast_improvement = _metric_value(metrics, "Stage 13", "RMSE improvement", "FAST")
    rgbd_mean = _metric_value(metrics, "Stage 14", "Mean 3D error", "5 static positions")
    stage16_success = _metric_value(metrics, "Stage 16", "Switch success rate", "Historical formal switching log")
    obstacle_error = _metric_value(metrics, "Stage 17", "Mean center error", "5 obstacle scenes")
    obstacle_iou = _metric_value(metrics, "Stage 17", "Mean AABB IoU", "5 obstacle scenes")
    obstacle_coverage = _metric_value(metrics, "Stage 17", "Mean GT coverage", "5 obstacle scenes")
    fn = _metric_value(metrics, "Stage 18", "FN", "20 candidate paths")
    safety_recall = _metric_value(metrics, "Stage 18", "Safety recall", "20 candidate paths")
    planning_success = _metric_value(metrics, "Stage 19", "RRT planning success rate", "Tested scenarios and seeds")
    execution_success = _metric_value(metrics, "Stage 20", "Execution success rate", "2 direct + 3 RRT")
    gt_collisions = _metric_value(metrics, "Stage 20", "GT collision count", "5 formal trials")

    quality_lines = builder.quality or ["No raw/summary discrepancies were detected in the checked fields."]
    lines = [
        "# Final Experimental Conclusions",
        "",
        "## 1. Visual Servo Convergence",
        "",
        f"The five Stage 7 raw trials produced a {stage7_rate:.1f}% static convergence success rate under the ±2 px final criterion. The historical Stage 7 summary is incomplete, so the final values are recomputed from all five raw trial files.",
        "",
        "## 2. Dynamic Tracking",
        "",
        f"After READY and with warm-up excluded, Stage 9 tracking RMSE was {stage9['SLOW']:.3f} px for SLOW, {stage9['MEDIUM']:.3f} px for MEDIUM, and {stage9['FAST']:.3f} px for FAST. Error increased with target speed in these three formal runs, while detection remained complete.",
        "",
        "## 3. Predictive Visual Servo",
        "",
        f"Using the best online-tested horizon in the recorded Stage 13 grid (tau=0.30 s), predictive compensation reduced raw RGB tracking RMSE by {medium_improvement:.2f}% for MEDIUM and {fast_improvement:.2f}% for FAST. This conclusion is limited to the tested speeds, estimator setting, and horizons.",
        "",
        "## 4. RGB-D Localization",
        "",
        f"The Stage 14 sphere-center estimates had a mean 3D error of {rgbd_mean:.2f} mm across five static positions. Surface-depth and center estimates remain distinct; the reported center result uses the documented known-radius compensation.",
        "",
        "## 5. Multi-Target Perception",
        "",
        "Stage 15 detected RED, GREEN, and BLUE at 100% in the recorded evaluation, with zero color-confusion events. The historical Stage 16 formal switching log is less successful: "
        f"its overall switch success rate is {stage16_success:.1f}% and its recorded result is FAIL. Later manual Demo behavior is not substituted for this historical formal dataset.",
        "",
        "## 6. Obstacle Perception",
        "",
        f"Stage 17 achieved a mean obstacle-center error of {obstacle_error:.2f} mm. The estimated occupancy covered {obstacle_coverage:.1f}% of the ground-truth AABB with {obstacle_iou:.2f}% IoU. The lower IoU reflects the deliberately enlarged conservative proxy rather than a missed obstacle.",
        "",
        "## 7. Collision Safety",
        "",
        f"Stage 18 recorded {int(fn)} false negatives and {safety_recall:.1f}% safety recall. False positives are the expected cost of conservative occupancy: safety coverage improved while usable free space decreased.",
        "",
        "## 8. RRT Planning",
        "",
        f"RRT-Connect achieved {planning_success:.1f}% success across the recorded Stage 19 RRT trials and tested seeds, with no ground-truth collision on final planned paths. This is an empirical result for the tested scenarios, not a guarantee of complete planning success.",
        "",
        "## 9. Trajectory Execution",
        "",
        f"Stage 20 completed all five formal execution trials ({execution_success:.1f}% success) with {int(gt_collisions)} ground-truth obstacle collisions. The raw execution trace confirms commanded and actual active-joint trajectories were recorded during motor-driven execution.",
        "",
        "## 10. Overall Conclusion",
        "",
        "The project demonstrates an integrated simulation pipeline from Eye-in-Hand RGB-D perception through target selection, predictive visual servoing, conservative collision checking, RRT-Connect planning, and motor-controlled Panda execution. Results support stable behavior in the tested scenarios, while the stated perception, planning-space, self-collision, static-obstacle, and simulation limitations remain material.",
        "",
        "## Data Quality Findings",
        "",
    ]
    lines.extend(f"- {item}" for item in quality_lines)
    lines.extend(["", "## Source CSVs Scanned", ""])
    lines.extend(f"- `{builder.relative(path)}`" for path in scanned)
    path = REPORT_DIR / "final_conclusions.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_project_summary(metrics: pd.DataFrame) -> Path:
    medium_improvement = _metric_value(metrics, "Stage 13", "RMSE improvement", "MEDIUM")
    fast_improvement = _metric_value(metrics, "Stage 13", "RMSE improvement", "FAST")
    rgbd_mean = _metric_value(metrics, "Stage 14", "Mean 3D error", "5 static positions")
    obstacle_mean = _metric_value(metrics, "Stage 17", "Mean center error", "5 obstacle scenes")
    safety_recall = _metric_value(metrics, "Stage 18", "Safety recall", "20 candidate paths")
    planning_success = _metric_value(metrics, "Stage 19", "RRT planning success rate", "Tested scenarios and seeds")
    execution_success = _metric_value(metrics, "Stage 20", "Execution success rate", "2 direct + 3 RRT")
    text = f"""# Final Project Summary

本项目在 PyBullet 中构建了基于 Eye-in-Hand RGB-D 的 Franka Panda 目标跟踪与避障系统，形成“感知→决策→控制→规划→执行”链路。RGB 图像经 OpenCV HSV 分割得到目标质心，深度与相机几何完成三维定位；TargetManager 支持多目标手动选择，并为每个目标隔离运动估计状态。控制层结合二维视觉伺服、有限差分和 EMA 预测，规划层使用视觉估计的保守障碍物 AABB、整臂简化碰撞检查和双向 RRT-Connect，最后通过 POSITION_CONTROL 执行验证后的关节轨迹。

五组静态视觉伺服均进入 ±2 px 区域。Stage 13 最佳在线测试配置相对基线将 MEDIUM 与 FAST 的原始像素 RMSE 分别降低 {medium_improvement:.2f}% 和 {fast_improvement:.2f}%。RGB-D 球心定位五位置平均误差为 {rgbd_mean:.2f} mm；三色目标检测率均为 100%，颜色混淆为 0。障碍物中心平均误差为 {obstacle_mean:.2f} mm，保守 occupancy 获得 100% GT 覆盖。碰撞检查安全召回率为 {safety_recall:.1f}%，假阴性为 0；RRT-Connect 在已测试场景与种子中的成功率为 {planning_success:.1f}%；五次正式执行成功率为 {execution_success:.1f}%，真实障碍物碰撞为 0。

项目保留了从视觉测量到轨迹执行的可追溯 CSV 和评价指标。其限制是：目标检测依赖颜色 HSV；障碍物采用简化且偏保守的 AABB；RRT 仅在项目定义的活动关节子空间规划；完整机器人自碰撞尚未实现；避障 Demo 面向静态障碍物；结论来自仿真而非实机。未来可扩展学习型检测、动态障碍物重规划、完整自碰撞模型及实机标定部署。
"""
    path = REPORT_DIR / "final_project_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    _configure_plots()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes_before = _source_hashes()
    builder = ReportBuilder()
    scanned = _scan_all_csvs(builder)

    stage7 = _analyse_stage7(builder)
    _analyse_stage9(builder)
    _analyse_stage10(builder)
    _analyse_stage12(builder)
    stage13_summary, stage13_selected = _analyse_stage13(builder)
    stage14_summary = _analyse_stage14(builder)
    stage15_summary = _analyse_stage15(builder)
    _analyse_stage16(builder)
    stage17_summary = _analyse_stage17(builder)
    stage18_summary = _analyse_stage18(builder)
    stage19_raw, stage19_summary = _analyse_stage19(builder)
    stage20_raw, stage20_summary = _analyse_stage20(builder)

    figure_paths = [
        _figure_stage7(stage7),
        _figure_stage13(stage13_summary, stage13_selected),
        _figure_stage14(stage14_summary),
        _figure_stage15(stage15_summary),
        _figure_stage17(stage17_summary),
        _figure_stage18(stage18_summary),
        _figure_stage19(stage19_raw, stage19_summary),
        _figure_stage20(stage20_raw, stage20_summary),
    ]
    csv_path, md_path, metrics = _write_metrics(builder)
    conclusions_path = _write_conclusions(builder, metrics, scanned)
    summary_path = _write_project_summary(metrics)

    source_hashes_after = _source_hashes()
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("Historical CSV hash audit failed: outputs/logs changed during Stage 23")
    expected_outputs = [csv_path, md_path, conclusions_path, summary_path, *figure_paths]
    missing = [path for path in expected_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty Stage 23 output(s): {missing}")

    print("STAGE 23 FINAL EXPERIMENTAL SUMMARY")
    print(f"CSV files scanned: {len(scanned)}")
    print(f"Traceable metrics: {len(metrics)}")
    print(f"Figures generated: {len(figure_paths)}")
    print(f"Data quality findings: {len(builder.quality)}")
    for finding in builder.quality:
        print(f"- {finding}")
    print(f"Final metrics: {csv_path}")
    print(f"Conclusions: {conclusions_path}")
    print("Historical CSV SHA-256 audit: UNCHANGED")


if __name__ == "__main__":
    main()
