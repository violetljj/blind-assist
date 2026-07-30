from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0"
IMPLEMENTATION_ID = "DUAL_LOOP_CAUSAL_TRACK_TRISTATE_IMPL_R0"
SELECTION_SEED = "dual-loop-causal-track-tristate-confirmation-r0"
WINDOW_FRAMES = 360
SELECTED_SEQUENCES = 3
HISTORY_FRAMES = 7
SOURCE_SLOPE_THRESHOLD_PER_S = 0.2
TRUTH_RATE_DEADBAND_MPS = 0.1
MINIMUM_TOTAL_EVIDENCE = 100
MINIMUM_SESSION_EVIDENCE = 20
MINIMUM_SESSION_COVERAGE = 0.005
MINIMUM_SESSION_PRECISION = 0.90
MINIMUM_POOLED_DIRECTION_PRECISION = 0.90
MINIMUM_DISTINCT_TRACKS = 10

EXCLUDED_OUTCOME_OPEN_SEQUENCES = frozenset(
    {
        "bytes-cafe-2019-02-07_0",
        "clark-center-2019-02-28_0",
        "clark-center-2019-02-28_1",
        "clark-center-intersection-2019-02-28_0",
        "cubberly-auditorium-2019-04-22_0",
        "gates-ai-lab-2019-02-08_0",
        "gates-basement-elevators-2019-01-17_1",
        "hewlett-packard-intersection-2019-01-24_0",
        "meyer-green-2019-03-16_0",
        "packard-poster-session-2019-03-20_0",
        "stlc-111-2019-04-19_0",
        "svl-meeting-gates-2-2019-04-08_1",
        "tressider-2019-04-26_2",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def ols_slope(times_s: list[float], values: list[float]) -> float:
    if len(times_s) != len(values) or len(times_s) < 2:
        raise ValueError("OLS requires paired observations")
    if not all(finite_number(value) for value in (*times_s, *values)):
        raise ValueError("OLS input is non-finite")
    center_t = sum(times_s) / len(times_s)
    center_y = sum(values) / len(values)
    denominator = sum((value - center_t) ** 2 for value in times_s)
    if denominator <= 0:
        raise ValueError("OLS timestamps are degenerate")
    return sum(
        (time_s - center_t) * (value - center_y)
        for time_s, value in zip(times_s, values)
    ) / denominator


def source_decision(
    times_s: list[float],
    log_heights: list[float],
) -> tuple[str, float | None, str]:
    if len(times_s) != HISTORY_FRAMES or len(log_heights) != HISTORY_FRAMES:
        return "ABSTAIN", None, "INSUFFICIENT_CONTIGUOUS_HISTORY"
    if any(right <= left for left, right in zip(times_s, times_s[1:])):
        return "ABSTAIN", None, "NON_MONOTONIC_TIMESTAMP"
    slope = ols_slope(times_s, log_heights)
    deltas = [
        right - left for left, right in zip(log_heights, log_heights[1:])
    ]
    if (
        slope >= SOURCE_SLOPE_THRESHOLD_PER_S
        and all(value > 0 for value in deltas)
    ):
        return "CONFIRM_APPROACH", slope, "SEVEN_FRAME_UNANIMOUS_GROWTH"
    if (
        slope <= -SOURCE_SLOPE_THRESHOLD_PER_S
        and all(value < 0 for value in deltas)
    ):
        return "CONTRADICT_APPROACH", slope, "SEVEN_FRAME_UNANIMOUS_SHRINKAGE"
    return "ABSTAIN", slope, "TREND_NOT_SELECTIVE"


def source_parameters() -> dict[str, Any]:
    return {
        "history_frames": HISTORY_FRAMES,
        "feature": "log_bbox_height",
        "fit": "causal_ordinary_least_squares",
        "adjacent_sign_rule": "all_six_strictly_same_sign",
        "slope_threshold_per_s": SOURCE_SLOPE_THRESHOLD_PER_S,
        "outputs": [
            "CONFIRM_APPROACH",
            "CONTRADICT_APPROACH",
            "ABSTAIN",
        ],
    }


def source_parameter_sha256() -> str:
    return sha256_bytes(
        json.dumps(
            source_parameters(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def frame_detection_id(
    sequence: str,
    frame_stem: str,
    label_id: str,
) -> str:
    return sha256_bytes(f"{sequence}|{frame_stem}|{label_id}".encode("utf-8"))


def immutable_roi_id(
    detection_id: str,
    box_xywh: list[float],
) -> str:
    encoded = json.dumps(
        [float(value) for value in box_xywh],
        separators=(",", ":"),
    )
    return sha256_bytes(f"{detection_id}|{encoded}".encode("utf-8"))
