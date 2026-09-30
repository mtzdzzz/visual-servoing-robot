"""Small CSV/output helpers that preserve caller-provided historical schemas."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO


def ensure_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_dict_writer(path: Path, fieldnames: Sequence[str]) -> tuple[TextIO, csv.DictWriter]:
    """Open a UTF-8 CSV and write exactly the schema supplied by its Stage."""
    ensure_output_directory(path.parent)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    handle, writer = open_dict_writer(path, fieldnames)
    try:
        writer.writerows(rows)
    finally:
        handle.close()
