#!/usr/bin/env python3
"""Evaluate aligned depth caches against the frozen DA V2 model-variant gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_metric3d_clearance_field_a0 import (
    BANDS,
    HORIZONS_M,
    clearance_field,
    intrinsics_matrix,
    summarize,
    tum_depth_metres,
)
from prepare_bonn_rgbd_metric_depth_manifest import (
    normalize_depth_image,
)

SCHEMA = "blindassist_dav2_model_variant_gate_r0_result"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def depth_metrics(candidate: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    truth_valid = np.isfinite(truth) & (truth >= 0.25) & (truth <= 6.0)
    paired = truth_valid & np.isfinite(candidate) & (candidate > 0.0)
    truth_count = int(np.sum(truth_valid))
    paired_count = int(np.sum(paired))
    if truth_count == 0 or paired_count == 0:
        return {
            "truth_valid_pixels": truth_count,
            "paired_valid_pixels": paired_count,
            "paired_coverage": 0.0,
            "metric_abs_rel_median": math.inf,
            "scale_aligned_abs_rel_median": math.inf,
            "median_scale": math.nan,
        }
    candidate_values = np.asarray(candidate[paired], dtype=np.float64)
    truth_values = np.asarray(truth[paired], dtype=np.float64)
    abs_rel = np.abs(candidate_values - truth_values) / truth_values
    scale = float(np.median(truth_values / candidate_values))
    scale_abs_rel = np.abs(scale * candidate_values - truth_values) / truth_values
    return {
        "truth_valid_pixels": truth_count,
        "paired_valid_pixels": paired_count,
        "paired_coverage": paired_count / truth_count,
        "metric_abs_rel_median": float(np.median(abs_rel)),
        "scale_aligned_abs_rel_median": float(np.median(scale_abs_rel)),
        "median_scale": scale,
    }


def _field_signature(field: dict[str, Any]) -> tuple[Any, ...]:
    status = str(field.get("status", "UNKNOWN_MISSING_STATUS"))
    if status != "VALID":
        return (status,)
    values: list[Any] = [status]
    for band in BANDS:
        band_field = field["bands"][band]
        for horizon in HORIZONS_M:
            values.append(band_field["occupied_by_horizon"][str(horizon)])
    return tuple(values)


def _false_clear_counts(
    rows: list[dict[str, Any]], candidate_key: str
) -> tuple[int, int, int]:
    false_clear = 0
    occupied_truth = 0
    known = 0
    for row in rows:
        truth = row["sensor"]
        candidate = row[candidate_key]
        if truth.get("status") != "VALID" or candidate.get("status") != "VALID":
            continue
        for band in BANDS:
            for horizon in HORIZONS_M:
                truth_value = truth["bands"][band]["occupied_by_horizon"][str(horizon)]
                candidate_value = candidate["bands"][band]["occupied_by_horizon"][
                    str(horizon)
                ]
                if truth_value is None or candidate_value is None:
                    continue
                known += 1
                occupied_truth += int(bool(truth_value))
                false_clear += int(bool(truth_value) and not bool(candidate_value))
    return false_clear, occupied_truth, known


def compare_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_equal = 0
    state_equal = 0
    clearance_differences: list[float] = []
    transition_equal = 0
    transition_count = 0
    previous: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    confusion: dict[str, int] = {}
    for row in rows:
        baseline = row["baseline"]
        candidate = row["candidate"]
        baseline_status = str(baseline.get("status"))
        candidate_status = str(candidate.get("status"))
        status_equal += int(baseline_status == candidate_status)
        confusion_key = f"{baseline_status}->{candidate_status}"
        confusion[confusion_key] = confusion.get(confusion_key, 0) + 1
        baseline_signature = _field_signature(baseline)
        candidate_signature = _field_signature(candidate)
        state_equal += int(baseline_signature == candidate_signature)
        if baseline_status == "VALID" and candidate_status == "VALID":
            for band in BANDS:
                baseline_clearance = baseline["bands"][band]["clearance_m"]
                candidate_clearance = candidate["bands"][band]["clearance_m"]
                if baseline_clearance is not None and candidate_clearance is not None:
                    clearance_differences.append(
                        abs(float(candidate_clearance) - float(baseline_clearance))
                    )
        sequence_id = str(row["sequence_id"])
        if sequence_id in previous:
            previous_baseline, previous_candidate = previous[sequence_id]
            baseline_changed = baseline_signature != previous_baseline
            candidate_changed = candidate_signature != previous_candidate
            transition_equal += int(baseline_changed == candidate_changed)
            transition_count += 1
        previous[sequence_id] = (baseline_signature, candidate_signature)
    count = len(rows)
    return {
        "status_exact_agreement": status_equal / count if count else 0.0,
        "geometry_state_exact_agreement": state_equal / count if count else 0.0,
        "geometry_state_change_frames": count - state_equal,
        "transition_change_agreement": (
            transition_equal / transition_count if transition_count else None
        ),
        "transition_pairs": transition_count,
        "clearance_difference_median_m": (
            statistics.median(clearance_differences)
            if clearance_differences
            else None
        ),
        "clearance_difference_p95_m": _quantile(clearance_differences, 0.95),
        "status_confusion": dict(sorted(confusion.items())),
    }


def summarize_depth(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": len(per_frame),
        "minimum_paired_pixel_coverage": min(
            float(row["paired_coverage"]) for row in per_frame
        ),
        "frame_median_metric_abs_rel_median": statistics.median(
            float(row["metric_abs_rel_median"]) for row in per_frame
        ),
        "frame_median_metric_abs_rel_p95": _quantile(
            [float(row["metric_abs_rel_median"]) for row in per_frame], 0.95
        ),
        "frame_median_scale_aligned_abs_rel_median": statistics.median(
            float(row["scale_aligned_abs_rel_median"]) for row in per_frame
        ),
        "frame_median_scale_aligned_abs_rel_p95": _quantile(
            [
                float(row["scale_aligned_abs_rel_median"])
                for row in per_frame
            ],
            0.95,
        ),
    }


def _task_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    remapped = [
        {
            "sequence_id": row["sequence_id"],
            "timestamp": row["timestamp"],
            "sensor": row["sensor"],
            "candidate": row[key],
            "latency_ms": 0.0,
        }
        for row in rows
    ]
    summary = summarize(remapped)
    false_clear, occupied_truth, known = _false_clear_counts(rows, key)
    sensor_valid = sum(row["sensor"].get("status") == "VALID" for row in rows)
    candidate_valid_on_sensor_valid = sum(
        row["sensor"].get("status") == "VALID"
        and row[key].get("status") == "VALID"
        for row in rows
    )
    truth_status_exact = sum(
        row["sensor"].get("status") == row[key].get("status") for row in rows
    )
    return {
        "paired_valid_frames": summary["paired_valid_frames"],
        "paired_valid_fraction": summary["paired_valid_fraction"],
        "ground_recovery_success_rate": (
            candidate_valid_on_sensor_valid / sensor_valid if sensor_valid else 0.0
        ),
        "truth_status_exact_agreement": (
            truth_status_exact / len(rows) if rows else 0.0
        ),
        "clearance_mae_m": summary["clearance_mae_m"],
        "collision_agreement": summary["collision_agreement"],
        "false_clear_rate_all_known_decisions": summary["false_clear_rate"],
        "false_clear_count": false_clear,
        "truth_occupied_decisions": occupied_truth,
        "false_clear_rate_given_truth_occupied": (
            false_clear / occupied_truth if occupied_truth else None
        ),
        "known_collision_decisions": known,
        "temporal_clearance_delta_mae_m": summary[
            "temporal_clearance_delta_mae_m"
        ],
        "camera_height_mae_m": summary["camera_height_mae_m"],
    }


def _resolve_frame(source_root: Path, row: dict[str, Any], key: str) -> Path:
    return source_root / str(row["sequence_root"]) / str(row[key])


def _verify_roster(
    protocol: dict[str, Any], roster: dict[str, Any], source_root: Path
) -> None:
    if roster.get("schema") != "blindassist_dav2_model_variant_gate_r0_roster":
        raise ValueError("roster schema mismatch")
    rows = roster.get("rows")
    expected = int(protocol["cohort"]["frames"])
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"roster must contain exactly {expected} rows")
    if sha256_file(Path(protocol["roster_path"])) != protocol["roster_sha256"]:
        raise ValueError("roster SHA-256 mismatch")
    for row in rows:
        rgb_path = _resolve_frame(source_root, row, "rgb_path")
        depth_path = _resolve_frame(source_root, row, "depth_path")
        if sha256_file(rgb_path) != row["rgb_sha256"]:
            raise ValueError(f"RGB hash mismatch: {row['frame_id']}")
        if sha256_file(depth_path) != row["depth_sha256"]:
            raise ValueError(f"depth hash mismatch: {row['frame_id']}")


def evaluate(
    protocol_path: Path,
    roster_path: Path,
    source_root: Path,
    baseline_depth_path: Path,
    candidate_depth_path: Path,
    candidate_id: str,
    baseline_latency_p95_ms: float | None = None,
    candidate_latency_p95_ms: float | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    protocol = dict(protocol)
    protocol["roster_path"] = str(roster_path.resolve())
    _verify_roster(protocol, roster, source_root.resolve())
    baseline_hash = sha256_file(baseline_depth_path)
    if baseline_hash != protocol["baseline"]["aligned_depth_sha256"]:
        raise ValueError("baseline aligned-depth SHA-256 mismatch")
    baseline_depth = np.load(baseline_depth_path, mmap_mode="r")
    candidate_depth = np.load(candidate_depth_path, mmap_mode="r")
    required_shape = tuple(protocol["cohort"]["aligned_depth_shape"])
    if tuple(baseline_depth.shape) != required_shape:
        raise ValueError("baseline depth shape mismatch")
    if tuple(candidate_depth.shape) != required_shape:
        raise ValueError("candidate depth shape mismatch")

    frame_rows: list[dict[str, Any]] = []
    baseline_depth_metrics: list[dict[str, Any]] = []
    candidate_depth_metrics: list[dict[str, Any]] = []
    for index, roster_row in enumerate(roster["rows"]):
        depth_path = _resolve_frame(source_root, roster_row, "depth_path")
        sensor_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        sensor_depth = tum_depth_metres(
            normalize_depth_image(sensor_raw, depth_path)
        )
        baseline = np.asarray(baseline_depth[index], dtype=np.float32)
        candidate = np.asarray(candidate_depth[index], dtype=np.float32)
        baseline_metrics = depth_metrics(baseline, sensor_depth)
        candidate_metrics = depth_metrics(candidate, sensor_depth)
        baseline_depth_metrics.append(baseline_metrics)
        candidate_depth_metrics.append(candidate_metrics)
        intrinsics = intrinsics_matrix(roster_row)
        frame_rows.append(
            {
                "index": index,
                "frame_id": roster_row["frame_id"],
                "sequence_id": roster_row["sequence_id"],
                "timestamp": roster_row["timestamp"],
                "sensor": clearance_field(sensor_depth, intrinsics),
                "baseline": clearance_field(baseline, intrinsics),
                "candidate": clearance_field(candidate, intrinsics),
                "baseline_depth": baseline_metrics,
                "candidate_depth": candidate_metrics,
            }
        )

    baseline_depth_summary = summarize_depth(baseline_depth_metrics)
    candidate_depth_summary = summarize_depth(candidate_depth_metrics)
    baseline_task = _task_summary(frame_rows, "baseline")
    candidate_task = _task_summary(frame_rows, "candidate")
    geometry = compare_geometry(frame_rows)
    tolerances = protocol["engineering_noninferiority_tolerances"]
    engineering_gates = {
        "minimum_pixel_coverage": candidate_depth_summary[
            "minimum_paired_pixel_coverage"
        ]
        >= baseline_depth_summary["minimum_paired_pixel_coverage"]
        - float(tolerances["minimum_pixel_coverage_drop"]),
        "metric_abs_rel": candidate_depth_summary[
            "frame_median_metric_abs_rel_median"
        ]
        <= baseline_depth_summary["frame_median_metric_abs_rel_median"]
        + float(tolerances["metric_abs_rel_absolute_increase"]),
        "scale_aligned_abs_rel": candidate_depth_summary[
            "frame_median_scale_aligned_abs_rel_median"
        ]
        <= baseline_depth_summary[
            "frame_median_scale_aligned_abs_rel_median"
        ]
        + float(tolerances["scale_aligned_abs_rel_absolute_increase"]),
        "ground_recovery": candidate_task["ground_recovery_success_rate"]
        >= baseline_task["ground_recovery_success_rate"]
        - float(tolerances["ground_recovery_rate_drop"]),
        "clearance_mae": candidate_task["clearance_mae_m"]
        <= baseline_task["clearance_mae_m"]
        + float(tolerances["clearance_mae_increase_m"]),
        "collision_agreement": candidate_task["collision_agreement"]
        >= baseline_task["collision_agreement"]
        - float(tolerances["collision_agreement_drop"]),
        "false_clear": candidate_task["false_clear_rate_all_known_decisions"]
        <= baseline_task["false_clear_rate_all_known_decisions"]
        + float(tolerances["false_clear_rate_increase"]),
        "temporal_clearance_delta": candidate_task[
            "temporal_clearance_delta_mae_m"
        ]
        <= baseline_task["temporal_clearance_delta_mae_m"]
        + float(tolerances["temporal_clearance_delta_mae_increase_m"]),
        "valid_unknown_consistency": geometry["status_exact_agreement"]
        >= float(tolerances["minimum_status_exact_agreement"]),
        "geometry_state_consistency": geometry[
            "geometry_state_exact_agreement"
        ]
        >= float(tolerances["minimum_geometry_state_exact_agreement"]),
        "geometry_transition_consistency": geometry[
            "transition_change_agreement"
        ]
        >= float(tolerances["minimum_transition_change_agreement"]),
    }
    absolute = protocol["absolute_consumed_task_gates"]
    absolute_gates = {
        "paired_valid_fraction": candidate_task["paired_valid_fraction"]
        >= float(absolute["minimum_paired_valid_fraction"]),
        "clearance_mae": candidate_task["clearance_mae_m"]
        <= float(absolute["maximum_clearance_mae_m"]),
        "collision_agreement": candidate_task["collision_agreement"]
        >= float(absolute["minimum_collision_agreement"]),
        "false_clear": candidate_task["false_clear_rate_all_known_decisions"]
        <= float(absolute["maximum_false_clear_rate"]),
        "temporal_clearance_delta": candidate_task[
            "temporal_clearance_delta_mae_m"
        ]
        <= float(absolute["maximum_temporal_clearance_delta_mae_m"]),
    }
    latency = None
    if baseline_latency_p95_ms is not None and candidate_latency_p95_ms is not None:
        speedup = baseline_latency_p95_ms / candidate_latency_p95_ms
        latency = {
            "baseline_p95_ms": baseline_latency_p95_ms,
            "candidate_p95_ms": candidate_latency_p95_ms,
            "speedup": speedup,
            "minimum_required_speedup": protocol["performance_gate"][
                "minimum_p95_speedup"
            ],
            "passed": speedup
            >= float(protocol["performance_gate"]["minimum_p95_speedup"]),
        }
    engineering_passed = all(engineering_gates.values())
    absolute_passed = all(absolute_gates.values())
    if not engineering_passed:
        terminal = "MODEL_VARIANT_ENGINEERING_NONINFERIORITY_FAIL"
    elif absolute_passed:
        terminal = "MODEL_VARIANT_CONSUMED_TASK_AND_ENGINEERING_GATES_PASS"
    else:
        terminal = "MODEL_VARIANT_ENGINEERING_PASS_STANDALONE_TASK_NOT_SUPPORTED"
    failure_indices = set(protocol["typical_failure_cases"]["frame_indices"])
    failure_rows = [row for row in frame_rows if row["index"] in failure_indices]
    return {
        "schema": SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "roster_sha256": sha256_file(roster_path),
        "candidate_id": candidate_id,
        "candidate_aligned_depth_sha256": sha256_file(candidate_depth_path),
        "baseline_aligned_depth_sha256": baseline_hash,
        "data_role": protocol["data_role"],
        "claim_ceiling": protocol["claim_ceiling"],
        "baseline": {
            "depth": baseline_depth_summary,
            "task": baseline_task,
        },
        "candidate": {
            "depth": candidate_depth_summary,
            "task": candidate_task,
        },
        "candidate_vs_baseline_geometry": geometry,
        "engineering_noninferiority_gates": engineering_gates,
        "engineering_noninferiority_passed": engineering_passed,
        "absolute_consumed_task_gates": absolute_gates,
        "absolute_consumed_task_passed": absolute_passed,
        "performance_gate": latency,
        "typical_failure_cases": failure_rows,
        "terminal": terminal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-depth", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--baseline-latency-p95-ms", type=float)
    parser.add_argument("--candidate-latency-p95-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.protocol.resolve(),
        args.roster.resolve(),
        args.source_root.resolve(),
        args.baseline_depth.resolve(),
        args.candidate_depth.resolve(),
        args.candidate_id,
        args.baseline_latency_p95_ms,
        args.candidate_latency_p95_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "typical_failure_cases"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
