"""Plot Stage 16 target-switching logs without altering source CSV data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "outputs" / "logs" / "stage16_target_switching.csv"
SUMMARY_PATH = ROOT / "outputs" / "logs" / "stage16_summary.csv"
FIGURE_DIRECTORY = ROOT / "outputs" / "final_figures"


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def generate_stage16_figures() -> tuple[Path, Path]:
    """Save the required raw-error and per-switch response-time figures."""

    if not LOG_PATH.exists() or not SUMMARY_PATH.exists():
        raise FileNotFoundError("Stage 16 log and summary CSV files must exist before plotting.")
    raw = pd.read_csv(LOG_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    _require_columns(raw, {"time_s", "error_norm", "selected_class", "switch_event", "previous_target"}, "Stage 16 log")
    _require_columns(summary, {"record_type", "switch_index", "response_time_s", "switch_result"}, "Stage 16 summary")
    if raw.empty:
        raise ValueError("Stage 16 log contains no RGB controller frames.")
    if raw["time_s"].isna().any() or not raw["time_s"].is_monotonic_increasing:
        raise ValueError("Stage 16 time values must be present and monotonic.")
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    error_path = FIGURE_DIRECTORY / "stage16_target_switching_error.png"
    response_path = FIGURE_DIRECTORY / "stage16_switch_response_time.png"

    error = pd.to_numeric(raw["error_norm"], errors="coerce")
    figure, axis = plt.subplots(figsize=(11, 5.5), dpi=220)
    axis.plot(raw["time_s"], error, color="#253b70", linewidth=1.4, label="Selected target raw error")
    switches = raw.loc[raw["switch_event"].eq(1)]
    for _, switch in switches.iterrows():
        event_time = float(switch["time_s"])
        previous = str(switch["previous_target"]).replace("target_", "T")
        selected = str(switch["selected_target_id"]).replace("target_", "T")
        axis.axvline(event_time, color="#c0392b", linestyle="--", linewidth=1.0)
        axis.annotate(
            f"{previous}->{selected}", xy=(event_time, axis.get_ylim()[1]), xytext=(3, -14),
            textcoords="offset points", color="#c0392b", fontsize=8, rotation=90, va="top",
        )
    axis.axhline(2.0 * 2**0.5, color="#2e8b57", linestyle=":", linewidth=1.1, label="±2 px per-axis bound")
    axis.set_title("Stage 16 Selected-Target Raw Tracking Error")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Raw tracking error norm (px)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(error_path)
    plt.close(figure)

    switch_rows = summary.loc[summary["record_type"].eq("SWITCH")].copy()
    labels = [f"S{int(value)}" for value in switch_rows["switch_index"]]
    response = pd.to_numeric(switch_rows["response_time_s"], errors="coerce")
    colours = ["#2e8b57" if result == "PASS" else "#c0392b" for result in switch_rows["switch_result"]]
    figure, axis = plt.subplots(figsize=(8.5, 5), dpi=220)
    bars = axis.bar(labels, response.fillna(0.0), color=colours)
    for bar, result, value in zip(bars, switch_rows["switch_result"], response):
        text = f"{value:.2f}s" if pd.notna(value) else "FAIL"
        axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.08, text, ha="center", va="bottom", fontsize=9)
    axis.set_title("Stage 16 Target-Switch Response Time")
    axis.set_xlabel("Switch event")
    axis.set_ylabel("Response time to ±2 px for 5 frames (s)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(response_path)
    plt.close(figure)
    return error_path, response_path


if __name__ == "__main__":
    generated = generate_stage16_figures()
    print("Generated:", *generated)
