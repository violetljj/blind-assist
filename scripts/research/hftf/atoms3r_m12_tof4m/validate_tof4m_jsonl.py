#!/usr/bin/env python3
"""Validate AtomS3R-M12 + Unit ToF4M development JSONL captures."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SAMPLE_SCHEMA = "blindassist_atoms3r_tof4m_sample_r0"
EVENT_SCHEMA = "blindassist_atoms3r_tof4m_event_r0"
VALID_STATUSES = {
    "VALID",
    "INVALID_TIMEOUT",
    "INVALID_SENSOR_STATUS",
    "INVALID_RANGE",
}


def _require(row: dict[str, Any], keys: Iterable[str], line_number: int) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"line {line_number}: missing keys {missing}")


def validate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    previous_timestamp: dict[tuple[str, str], int] = {}
    previous_sample_index: dict[str, int] = {}
    status_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    samples = 0
    events = 0

    for line_number, row in enumerate(rows, start=1):
        schema = row.get("schema")
        if schema == EVENT_SCHEMA:
            _require(
                row,
                ("sequence_id", "timestamp_ns", "clock_domain", "event", "status"),
                line_number,
            )
            events += 1
            event_counts[str(row["event"])] += 1
            continue
        if schema != SAMPLE_SCHEMA:
            raise ValueError(f"line {line_number}: unsupported schema {schema!r}")

        _require(
            row,
            (
                "sequence_id",
                "sample_index",
                "timestamp_ns",
                "timestamp_semantics",
                "clock_domain",
                "sensor_id",
                "measurement_status",
                "timeout",
                "range_status_code",
                "range_mm",
                "range_m",
                "peak_signal_rate_mcps",
                "ambient_rate_mcps",
            ),
            line_number,
        )
        sequence_id = str(row["sequence_id"])
        clock_domain = str(row["clock_domain"])
        if not sequence_id or not clock_domain:
            raise ValueError(f"line {line_number}: empty sequence or clock domain")
        if row["timestamp_semantics"] != "sensor_read_complete":
            raise ValueError(f"line {line_number}: unsupported timestamp semantics")

        timestamp_ns = int(row["timestamp_ns"])
        clock_key = (sequence_id, clock_domain)
        if timestamp_ns <= previous_timestamp.get(clock_key, -1):
            raise ValueError(f"line {line_number}: timestamp is not strictly increasing")
        previous_timestamp[clock_key] = timestamp_ns

        sample_index = int(row["sample_index"])
        expected_index = previous_sample_index.get(sequence_id, -1) + 1
        if sample_index != expected_index:
            raise ValueError(
                f"line {line_number}: expected sample_index {expected_index}, got {sample_index}"
            )
        previous_sample_index[sequence_id] = sample_index

        measurement_status = str(row["measurement_status"])
        if measurement_status not in VALID_STATUSES:
            raise ValueError(f"line {line_number}: invalid measurement status")
        timed_out = row["timeout"]
        if not isinstance(timed_out, bool):
            raise TypeError(f"line {line_number}: timeout must be boolean")
        if (measurement_status == "INVALID_TIMEOUT") != timed_out:
            raise ValueError(f"line {line_number}: timeout/status disagreement")

        range_mm = int(row["range_mm"])
        range_m = row["range_m"]
        if measurement_status == "VALID":
            if timed_out or int(row["range_status_code"]) != 0:
                raise ValueError(f"line {line_number}: VALID row has invalid sensor status")
            if not 40 <= range_mm <= 4000:
                raise ValueError(f"line {line_number}: VALID range outside admitted bounds")
            if not isinstance(range_m, (int, float)) or not math.isclose(
                float(range_m), range_mm / 1000.0, abs_tol=0.0005
            ):
                raise ValueError(f"line {line_number}: range_m does not match range_mm")
        elif range_m is not None:
            raise ValueError(f"line {line_number}: invalid row must use null range_m")

        for field in ("peak_signal_rate_mcps", "ambient_rate_mcps"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"line {line_number}: {field} must be finite and nonnegative")

        samples += 1
        status_counts[measurement_status] += 1

    if samples == 0:
        raise ValueError("capture contains no ToF sample rows")
    return {
        "schema": "blindassist_atoms3r_tof4m_validation_r0",
        "status": "VALID",
        "sample_count": samples,
        "event_count": events,
        "measurement_status_counts": dict(sorted(status_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "sequence_count": len(previous_sample_index),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"line {line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_rows(load_jsonl(args.capture))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
