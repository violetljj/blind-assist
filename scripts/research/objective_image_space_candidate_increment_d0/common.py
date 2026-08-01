"""Shared identity and metric primitives for objective candidate increment D0."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    aggregate_confusion,
    component_metrics,
    pixel_metrics,
)

from . import ANALYSIS_HEIGHT, ANALYSIS_WIDTH, PROTOCOL_ID


class ContractError(ValueError):
    """Raised when an input violates the frozen D0 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ContractError(f"{path}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise ContractError(f"{path}: empty JSONL")
    return rows


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ContractError(f"{label} SHA drift: {actual} != {expected}")


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "FROZEN_PRE_OUTPUT"
    ):
        raise ContractError("protocol is not the frozen D0 contract")
    return value


def load_objective_view(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    required = {
        "schema_version",
        "source_session_id",
        "observation_index",
        "source_frame_index",
        "timestamp_ns",
        "image_path",
        "image_sha256",
        "oracle_mask_path",
        "oracle_mask_sha256",
        "image_width",
        "image_height",
        "mask_width",
        "mask_height",
    }
    seen: set[tuple[str, int, str]] = set()
    previous: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != "blindassist.objective_image_space_view.v1":
            raise ContractError(f"view row {index}: schema drift")
        missing = required - row.keys()
        if missing:
            raise ContractError(f"view row {index}: missing {sorted(missing)}")
        session = str(row["source_session_id"])
        frame = int(row["source_frame_index"])
        timestamp = int(row["timestamp_ns"])
        image_sha = str(row["image_sha256"])
        key = (session, frame, image_sha)
        if key in seen:
            raise ContractError(f"view row {index}: duplicate identity {key}")
        if session in previous and timestamp <= previous[session]:
            raise ContractError(f"view row {index}: non-increasing timestamp")
        if (int(row["image_width"]), int(row["image_height"])) != (512, 288):
            raise ContractError(f"view row {index}: image dimensions drift")
        if (int(row["mask_width"]), int(row["mask_height"])) != (
            ANALYSIS_WIDTH,
            ANALYSIS_HEIGHT,
        ):
            raise ContractError(f"view row {index}: mask dimensions drift")
        if set(row) - required:
            raise ContractError(
                f"view row {index}: non-objective fields present: {sorted(set(row) - required)}"
            )
        seen.add(key)
        previous[session] = timestamp
    return rows


def trapezoid_roi() -> np.ndarray:
    """Return the frozen mechanical ROI; it is not a route or action label."""

    ys = (np.arange(ANALYSIS_HEIGHT, dtype=np.float64) + 0.5) / ANALYSIS_HEIGHT
    xs = (np.arange(ANALYSIS_WIDTH, dtype=np.float64) + 0.5) / ANALYSIS_WIDTH
    roi = np.zeros((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), dtype=bool)
    for y_index, y_value in enumerate(ys):
        if y_value < 0.42:
            continue
        progress = (y_value - 0.42) / (1.0 - 0.42)
        half_width = 0.16 + progress * (0.42 - 0.16)
        roi[y_index] = np.abs(xs - 0.5) <= half_width
    return roi


def summarize_masks(
    truth_ids: np.ndarray,
    predicted_ids: np.ndarray,
    detector_masks: np.ndarray,
    *,
    roi: np.ndarray | None = None,
) -> dict[str, Any]:
    if (
        truth_ids.shape != predicted_ids.shape
        or truth_ids.shape != detector_masks.shape
        or truth_ids.ndim != 3
    ):
        raise ContractError("mask ledger arrays must be equal [N,H,W]")
    spatial = np.ones(truth_ids.shape[1:], dtype=bool) if roi is None else roi
    valid = truth_ids != 3
    truth = np.isin(truth_ids, (1, 2)) & valid & spatial
    predicted = np.isin(predicted_ids, (1, 2)) & valid & spatial
    detector = detector_masks.astype(bool) & valid & spatial
    combined = detector | predicted
    residual_truth = truth & ~detector
    candidate = predicted & ~detector

    arm_rows: dict[str, list[dict[str, Any]]] = {
        "A_YOLO_ONLY": [],
        "B_PIDNET_ONLY": [],
        "C_YOLO_PLUS_PIDNET": [],
        "RESIDUAL_CANDIDATE": [],
    }
    component_rows: list[dict[str, Any]] = []
    for index in range(truth.shape[0]):
        arm_rows["A_YOLO_ONLY"].append(pixel_metrics(detector[index], truth[index]))
        arm_rows["B_PIDNET_ONLY"].append(pixel_metrics(predicted[index], truth[index]))
        arm_rows["C_YOLO_PLUS_PIDNET"].append(pixel_metrics(combined[index], truth[index]))
        arm_rows["RESIDUAL_CANDIDATE"].append(
            pixel_metrics(candidate[index], residual_truth[index])
        )
        component_rows.append(component_metrics(candidate[index], residual_truth[index]))

    arms = {name: aggregate_confusion(rows) for name, rows in arm_rows.items()}
    predicted_components = sum(
        int(row["predicted_component_count"]) for row in component_rows
    )
    truth_components = sum(int(row["truth_component_count"]) for row in component_rows)
    hit_predicted = sum(
        int(row["hit_predicted_component_count"]) for row in component_rows
    )
    hit_truth = sum(int(row["hit_truth_component_count"]) for row in component_rows)
    false_components = sum(
        int(row["false_activation_component_count"]) for row in component_rows
    )
    component_summary = {
        "predicted_component_count": predicted_components,
        "truth_component_count": truth_components,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": (
            hit_predicted / predicted_components if predicted_components else None
        ),
        "component_recall": hit_truth / truth_components if truth_components else None,
        "false_activation_component_count": false_components,
        "false_activation_components_per_frame": false_components / truth.shape[0],
    }
    return {
        "arms": arms,
        "delta_recall_c_minus_a": (
            float(arms["C_YOLO_PLUS_PIDNET"]["recall"])
            - float(arms["A_YOLO_ONLY"]["recall"])
        ),
        "added_false_positive_area_fraction": (
            int(arms["C_YOLO_PLUS_PIDNET"]["fp"])
            - int(arms["A_YOLO_ONLY"]["fp"])
        )
        / int(arms["A_YOLO_ONLY"]["pixel_count"]),
        "components": component_summary,
    }


def event_summaries(
    rows: Sequence[dict[str, Any]],
    truth_ids: np.ndarray,
    predicted_ids: np.ndarray,
    detector_masks: np.ndarray,
) -> list[dict[str, Any]]:
    indices_by_session: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_session.setdefault(str(row["source_session_id"]), []).append(index)
    summaries: list[dict[str, Any]] = []
    for session, indices in sorted(indices_by_session.items()):
        result = summarize_masks(
            truth_ids[indices], predicted_ids[indices], detector_masks[indices]
        )
        summaries.append(
            {
                "source_session_id": session,
                "frame_count": len(indices),
                "delta_recall_c_minus_a": result["delta_recall_c_minus_a"],
                "added_false_positive_area_fraction": result[
                    "added_false_positive_area_fraction"
                ],
            }
        )
    return summaries


def percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "p50": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def runtime_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContractError("runtime samples missing or non-finite")
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(np.max(array)),
    }


def decision(
    protocol: dict[str, Any],
    full: dict[str, Any],
    corridor: dict[str, Any],
    by_class: dict[str, dict[str, Any]],
    events: Sequence[dict[str, Any]],
    operator_runtime: dict[str, Any],
) -> dict[str, Any]:
    gates = protocol["decision_gates"]
    event_gain = [float(row["delta_recall_c_minus_a"]) for row in events]
    event_fp = [
        float(row["added_false_positive_area_fraction"]) for row in events
    ]
    checks = {
        "whole_recall_gain": full["delta_recall_c_minus_a"]
        >= float(gates["min_delta_recall"]),
        "residual_component_recall": float(
            full["components"]["component_recall"] or 0.0
        )
        >= float(gates["min_candidate_component_recall"]),
        "added_fp_area": full["added_false_positive_area_fraction"]
        <= float(gates["max_added_false_positive_area_fraction"]),
        "false_activation_components": float(
            full["components"]["false_activation_components_per_frame"]
        )
        <= float(gates["max_false_activation_components_per_frame"]),
        "event_median_gain": float(np.median(event_gain))
        >= float(gates["min_event_median_recall_gain"]),
        "event_p90_added_fp": float(np.percentile(event_fp, 90))
        <= float(gates["max_event_p90_added_false_positive_area_fraction"]),
        "blocking_obstacle_recall_gain": by_class["blocking_obstacle"][
            "delta_recall_c_minus_a"
        ]
        >= float(gates["min_per_class_delta_recall"]),
        "boundary_level_change_recall_gain": by_class["boundary_level_change"][
            "delta_recall_c_minus_a"
        ]
        >= float(gates["min_per_class_delta_recall"]),
        "corridor_no_direction_reversal": corridor["delta_recall_c_minus_a"] >= 0.0,
        "host_operator_budget": float(operator_runtime["p95_ms"])
        <= float(gates["max_host_operator_p95_ms"]),
    }
    utility_pass = all(checks.values())
    device = protocol["fixed_budget_evidence"]
    device_checks = {
        "device_total_p95": float(device["total_p95_ms"])
        <= float(gates["max_device_total_p95_ms"]),
        "thermal_ratio": float(device["final_over_initial_p95_ratio"])
        <= float(gates["max_final_over_initial_ratio"]),
        "thermal_status": int(device["maximum_thermal_status"])
        < int(gates["severe_thermal_status"]),
        "device_failure_count": int(device["failure_count"]) == 0,
        "full_qnn_delegation": bool(device["full_qnn_delegation"]),
    }
    budget_pass = all(device_checks.values())
    if not utility_pass:
        terminal = "STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_NO_ROBUST_INCREMENT"
    elif not budget_pass:
        terminal = "STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_BUDGET_FAIL"
    else:
        terminal = (
            "OBJECTIVE_IMAGE_SPACE_INCREMENT_SUPPORTED_BUT_ONSET_COHORT_REQUIRED"
        )
    return {
        "terminal": terminal,
        "coverage_false_activation_pass": utility_pass,
        "fixed_budget_pass": budget_pass,
        "timing_status": "NOT_EVALUABLE_ONSET_INCOMPLETE",
        "checks": checks,
        "device_checks": device_checks,
    }
