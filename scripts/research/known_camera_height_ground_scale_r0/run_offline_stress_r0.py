#!/usr/bin/env python3
"""Run frozen cached-depth stress tests for the sealed scale student.

This runner deliberately has no fit, refit, update, or threshold-search entrypoint.
RGB blur and low-light are out of scope: they require a real frozen-DA rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sealed_student import MODEL_ID, SealedScaleStudent

import core as scale_core

REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_consumed_tartanground import strict_band_values
from evaluate_metric3d_clearance_field_a0 import clearance_field

DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf/CAMERA_CONDITIONED_SCALE_STUDENT_OFFLINE_STRESS_R0_PROTOCOL_2026-08-04.json"
)
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
FEATURE_NAMES = (
    "log_r0_known_height_scale",
    "log_known_camera_height_m",
    "r0_plane_normal_x",
    "r0_plane_normal_y",
    "r0_plane_normal_z",
    "r0_normalized_plane_residual",
    "log_da_depth_q10",
    "log_da_depth_q50",
    "log_da_depth_q90",
    "log_da_depth_q90_over_q10",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()


def sealed_runtime_features(
    relative_depth: np.ndarray, height_m: float, recovery: dict[str, Any]
) -> np.ndarray | None:
    """Rebuild only the ten frozen runtime features; this is not a fit path."""
    if recovery.get("status") != "VALID":
        return None
    values = np.asarray(relative_depth, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if len(finite) < 500:
        return None
    q10, q50, q90 = np.quantile(finite, (0.10, 0.50, 0.90))
    if q10 <= 0.0 or q50 <= 0.0 or q90 <= 0.0:
        return None
    plane = recovery["ground"]
    features = np.asarray(
        [
            np.log(float(recovery["scale"])),
            np.log(height_m),
            *np.asarray(plane.normal, dtype=np.float64).tolist(),
            float(plane.normalized_median_residual),
            np.log(q10),
            np.log(q50),
            np.log(q90),
            np.log(q90 / q10),
        ],
        dtype=np.float64,
    )
    return features if np.all(np.isfinite(features)) else None


def signed_id(value: float, scale: float = 1.0) -> str:
    integer = round(value * scale)
    return ("p" if integer >= 0 else "m") + str(abs(integer))


def build_scenarios(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    perturbations = protocol["cached_depth_perturbations"]
    scenarios: list[dict[str, Any]] = [
        {"id": "clean", "family": "clean", "truth_comparable": True}
    ]
    for delta_cm in perturbations["camera_height_error_cm"]:
        scenarios.append(
            {
                "id": f"height_cm_{signed_id(delta_cm)}",
                "family": "camera_height_error",
                "delta_m": float(delta_cm) / 100.0,
                "truth_comparable": True,
            }
        )
    for fraction in perturbations["focal_length_error_fraction"]:
        scenarios.append(
            {
                "id": f"focal_pct_{signed_id(fraction, 100)}",
                "family": "focal_length_error",
                "fraction": float(fraction),
                "truth_comparable": True,
            }
        )
    for fraction in perturbations["global_da_scale_error_fraction"]:
        scenarios.append(
            {
                "id": f"global_da_pct_{signed_id(fraction, 100)}",
                "family": "global_da_scale_error",
                "fraction": float(fraction),
                "truth_comparable": True,
            }
        )
    local = perturbations["local_shape"]
    for field in local["fields"]:
        for amplitude in local["amplitudes"]:
            for polarity in local["polarities"]:
                scenarios.append(
                    {
                        "id": f"local_{field}_a{round(amplitude * 100):02d}_{signed_id(polarity)}",
                        "family": "local_shape",
                        "field": field,
                        "amplitude": float(amplitude),
                        "polarity": int(polarity),
                        "truth_comparable": True,
                    }
                )
    roi = perturbations["ground_roi_occlusion"]
    for pattern in roi["patterns"]:
        for fraction in roi["occluded_roi_fractions"]:
            scenarios.append(
                {
                    "id": f"roi_{pattern}_{round(fraction * 100):02d}",
                    "family": "ground_roi_occlusion",
                    "pattern": pattern,
                    "fraction": float(fraction),
                    "truth_comparable": True,
                }
            )
    canaries = protocol["geometric_canaries"]
    for degrees in canaries["pitch_degrees"]:
        scenarios.append(
            {
                "id": f"canary_pitch_deg_{signed_id(degrees)}",
                "family": "geometric_pitch_canary",
                "degrees": float(degrees),
                "truth_comparable": False,
            }
        )
    for degrees in canaries["roll_degrees"]:
        scenarios.append(
            {
                "id": f"canary_roll_deg_{signed_id(degrees)}",
                "family": "geometric_roll_canary",
                "degrees": float(degrees),
                "truth_comparable": False,
            }
        )
    for retained in canaries["center_crop_retained_fraction"]:
        scenarios.append(
            {
                "id": f"canary_center_crop_{round(retained * 100):02d}",
                "family": "geometric_crop_canary",
                "retained_fraction": float(retained),
                "truth_comparable": False,
            }
        )
    ids = [row["id"] for row in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids are not unique")
    return scenarios


def local_multiplier(shape: tuple[int, int], scenario: dict[str, Any]) -> np.ndarray:
    height, width = shape
    amplitude = float(scenario["amplitude"])
    polarity = float(scenario["polarity"])
    field = scenario["field"]
    if field == "vertical_linear":
        base = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
        base = np.broadcast_to(base, shape)
    elif field == "horizontal_linear":
        base = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :]
        base = np.broadcast_to(base, shape)
    elif field == "bandwise_left_to_right":
        base = np.zeros(shape, dtype=np.float64)
        boundaries = np.linspace(0, width, 4, dtype=int)
        for index, value in enumerate((-1.0, 0.0, 1.0)):
            base[:, boundaries[index] : boundaries[index + 1]] = value
    else:
        raise ValueError(f"unsupported local field: {field}")
    multiplier = 1.0 + polarity * amplitude * base
    multiplier /= float(np.median(multiplier))
    if np.any(multiplier <= 0.0) or not np.all(np.isfinite(multiplier)):
        raise ValueError("invalid local multiplier")
    return multiplier


def mask_ground_roi(depth: np.ndarray, scenario: dict[str, Any], start: float) -> np.ndarray:
    output = depth.copy()
    height, width = output.shape
    y0 = int(np.ceil(start * height))
    roi_height = height - y0
    fraction = float(scenario["fraction"])
    if scenario["pattern"] == "full_width_bottom":
        rows = max(1, round(roi_height * fraction))
        output[height - rows :, :] = np.nan
    elif scenario["pattern"] == "center_block":
        side = np.sqrt(fraction)
        block_height = max(1, round(roi_height * side))
        block_width = max(1, round(width * side))
        top = y0 + (roi_height - block_height) // 2
        left = (width - block_width) // 2
        output[top : top + block_height, left : left + block_width] = np.nan
    else:
        raise ValueError(f"unsupported ROI pattern: {scenario['pattern']}")
    return output


def rotation_homography(
    intrinsics: np.ndarray, degrees: float, axis: str
) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    if axis == "pitch":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    elif axis == "roll":
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported rotation axis: {axis}")
    return intrinsics @ rotation @ np.linalg.inv(intrinsics)


def warp_depth_canary(
    depth: np.ndarray, intrinsics: np.ndarray, degrees: float, axis: str
) -> np.ndarray:
    height, width = depth.shape
    matrix = rotation_homography(intrinsics, degrees, axis)
    return cv2.warpPerspective(
        depth.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    ).astype(np.float64)


def center_crop_canary(depth: np.ndarray, retained_fraction: float) -> np.ndarray:
    height, width = depth.shape
    crop_height = max(2, round(height * retained_fraction))
    crop_width = max(2, round(width * retained_fraction))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    crop = depth[top : top + crop_height, left : left + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR).astype(
        np.float64
    )


def apply_scenario(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    height_m: float,
    scenario: dict[str, Any],
    lower_roi_start_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    output = np.asarray(depth, dtype=np.float64).copy()
    adjusted_intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    adjusted_height = float(height_m)
    family = scenario["family"]
    if family == "clean":
        pass
    elif family == "camera_height_error":
        adjusted_height += float(scenario["delta_m"])
    elif family == "focal_length_error":
        factor = 1.0 + float(scenario["fraction"])
        adjusted_intrinsics[0, 0] *= factor
        adjusted_intrinsics[1, 1] *= factor
    elif family == "global_da_scale_error":
        output *= 1.0 + float(scenario["fraction"])
    elif family == "local_shape":
        output *= local_multiplier(output.shape, scenario)
    elif family == "ground_roi_occlusion":
        output = mask_ground_roi(output, scenario, lower_roi_start_fraction)
    elif family == "geometric_pitch_canary":
        output = warp_depth_canary(output, adjusted_intrinsics, scenario["degrees"], "pitch")
    elif family == "geometric_roll_canary":
        output = warp_depth_canary(output, adjusted_intrinsics, scenario["degrees"], "roll")
    elif family == "geometric_crop_canary":
        output = center_crop_canary(output, scenario["retained_fraction"])
    else:
        raise ValueError(f"unsupported scenario family: {family}")
    return output, adjusted_intrinsics, adjusted_height


def is_accepted_bad(candidate: dict[str, float] | None, truth: Any) -> bool | None:
    if candidate is None or not isinstance(truth, dict):
        return None
    clearance_bad = any(abs(candidate[band] - float(truth[band])) > 0.25 for band in BANDS)
    false_clear = any(
        float(truth[band]) <= horizon < candidate[band]
        for band in BANDS
        for horizon in HORIZONS
    )
    return bool(clearance_bad or false_clear)


def evaluate_one(
    row: dict[str, Any],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    height_m: float,
    student: SealedScaleStudent,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    parent_id = str(row["parent_id"])
    receipt = scale_core.CameraHeightReceipt(parent_id, parent_id, height_m, 0.0)
    recovery = scale_core.recover_metric_scale(
        depth, intrinsics, receipt, parent_id, parent_id
    )
    candidate = None
    predicted_scale = None
    reason = None
    feature_values = sealed_runtime_features(depth, height_m, recovery)
    if feature_values is None:
        reason = str(recovery.get("reason", "INVALID_RUNTIME_FEATURES"))
    else:
        prediction = student.predict(feature_values)
        if prediction.get("status") != "VALID":
            reason = str(prediction.get("reason", "STUDENT_UNKNOWN"))
        else:
            predicted_scale = float(prediction["scale"])
            plane = recovery["ground"]
            candidate = strict_band_values(
                clearance_field(
                    depth * predicted_scale,
                    intrinsics,
                    plane_override=(
                        plane.normal,
                        height_m,
                        plane.normalized_median_residual * height_m,
                    ),
                )
            )
            if candidate is None:
                reason = "STRICT_CLEARANCE_BAND_UNKNOWN"
    plane = recovery.get("ground")
    truth_comparable = bool(scenario["truth_comparable"])
    truth = row.get("truth") if truth_comparable else None
    return {
        "scenario_id": scenario["id"],
        "scenario_family": scenario["family"],
        "truth_comparable": truth_comparable,
        "parent_id": parent_id,
        "anchor_frame_id": int(row["anchor_frame_id"]),
        "prediction_sha256": row.get("prediction_sha256"),
        "effective_height_m": float(height_m),
        "candidate": candidate,
        "truth": truth,
        "accepted_bad": is_accepted_bad(candidate, truth),
        "unknown_reason": reason,
        "student_predicted_scale": predicted_scale,
        "aligned_scale_diagnostic": row.get("aligned_scale_diagnostic"),
        "ground": (
            {
                "candidate_count": int(plane.candidate_count),
                "inlier_count": int(plane.inlier_count),
                "inlier_fraction": float(plane.inlier_fraction),
                "normalized_median_residual": float(plane.normalized_median_residual),
                "normal": np.asarray(plane.normal, dtype=float).tolist(),
                "relative_height": float(plane.relative_height),
            }
            if plane is not None
            else None
        ),
    }


def mean_or_none(values: Iterable[float]) -> float | None:
    rows = list(values)
    return float(np.mean(rows)) if rows else None


def summarize_parent(rows: list[dict[str, Any]], truth_comparable: bool) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["anchor_frame_id"])
    accepted = [row for row in ordered if row["candidate"] is not None]
    summary: dict[str, Any] = {
        "parent_id": ordered[0]["parent_id"],
        "frame_count": len(ordered),
        "accepted_frame_count": len(accepted),
        "known_coverage": len(accepted) / len(ordered) if ordered else None,
        "unknown_reason_counts": dict(Counter(row["unknown_reason"] for row in ordered if row["unknown_reason"])),
    }
    if not truth_comparable:
        summary.update(
            {
                "clearance_mae_m": None,
                "envelope_agreement": None,
                "false_clear_rate": None,
                "conditional_false_clear_rate": None,
                "temporal_delta_mae_m": None,
                "accepted_bad_rate_all_truth_frames": None,
                "accepted_bad_rate_among_accepted_frames": None,
            }
        )
        return summary
    errors: list[float] = []
    agreements: list[float] = []
    false_clears: list[float] = []
    conditional_false: list[float] = []
    temporal_errors: list[float] = []
    accepted_bad = [row for row in accepted if row["accepted_bad"]]
    previous: dict[str, Any] | None = None
    for row in ordered:
        candidate, truth = row["candidate"], row["truth"]
        if candidate is None or not isinstance(truth, dict):
            previous = None
            continue
        for band in BANDS:
            errors.append(abs(candidate[band] - float(truth[band])))
            for horizon in HORIZONS:
                occupied = float(truth[band]) <= horizon
                predicted_occupied = candidate[band] <= horizon
                agreements.append(float(occupied == predicted_occupied))
                false_clears.append(float(occupied and not predicted_occupied))
                if occupied:
                    conditional_false.append(float(not predicted_occupied))
        if previous is not None:
            for band in BANDS:
                truth_delta = float(truth[band]) - float(previous["truth"][band])
                candidate_delta = candidate[band] - previous["candidate"][band]
                temporal_errors.append(abs(candidate_delta - truth_delta))
        previous = row
    summary.update(
        {
            "clearance_mae_m": mean_or_none(errors),
            "envelope_agreement": mean_or_none(agreements),
            "false_clear_rate": mean_or_none(false_clears),
            "conditional_false_clear_rate": mean_or_none(conditional_false),
            "temporal_delta_mae_m": mean_or_none(temporal_errors),
            "accepted_bad_rate_all_truth_frames": len(accepted_bad) / len(ordered) if ordered else None,
            "accepted_bad_rate_among_accepted_frames": len(accepted_bad) / len(accepted) if accepted else None,
        }
    )
    return summary


def worst_parent(parent_rows: list[dict[str, Any]]) -> dict[str, Any]:
    directions = {
        "known_coverage": "minimum",
        "clearance_mae_m": "maximum",
        "envelope_agreement": "minimum",
        "false_clear_rate": "maximum",
        "conditional_false_clear_rate": "maximum",
        "temporal_delta_mae_m": "maximum",
        "accepted_bad_rate_all_truth_frames": "maximum",
        "accepted_bad_rate_among_accepted_frames": "maximum",
    }
    output = {}
    for metric, direction in directions.items():
        eligible = [row for row in parent_rows if row.get(metric) is not None]
        if not eligible:
            output[metric] = None
            continue
        selected = (min if direction == "minimum" else max)(eligible, key=lambda row: row[metric])
        output[metric] = {
            "parent_id": selected["parent_id"],
            "value": selected[metric],
            "direction": direction,
        }
    return output


def summarize_scenario(rows: list[dict[str, Any]], scenario: dict[str, Any]) -> dict[str, Any]:
    parents = sorted({row["parent_id"] for row in rows})
    parent_rows = [
        summarize_parent(
            [row for row in rows if row["parent_id"] == parent],
            bool(scenario["truth_comparable"]),
        )
        for parent in parents
    ]
    metrics = (
        "known_coverage",
        "clearance_mae_m",
        "envelope_agreement",
        "false_clear_rate",
        "conditional_false_clear_rate",
        "temporal_delta_mae_m",
        "accepted_bad_rate_all_truth_frames",
        "accepted_bad_rate_among_accepted_frames",
    )
    macro = {
        metric: mean_or_none(row[metric] for row in parent_rows if row[metric] is not None)
        for metric in metrics
    }
    return {
        "scenario": scenario,
        "record_count": len(rows),
        "parent_count": len(parents),
        "unknown_reason_counts": dict(Counter(row["unknown_reason"] for row in rows if row["unknown_reason"])),
        "parent_macro": macro,
        "worst_parent": worst_parent(parent_rows),
        "parents": parent_rows,
    }


def add_clean_delta(summary: dict[str, Any], clean: dict[str, Any]) -> None:
    delta = {}
    for metric, value in summary["parent_macro"].items():
        baseline = clean["parent_macro"].get(metric)
        delta[metric] = (
            float(value - baseline)
            if value is not None and baseline is not None
            else None
        )
    summary["delta_vs_clean"] = delta


def validate_inputs(
    protocol_path: Path,
    protocol: dict[str, Any],
    input_path: Path,
    model_path: Path,
    input_result: dict[str, Any],
    limit: int | None,
) -> None:
    if protocol.get("status") != "FROZEN_BEFORE_OFFLINE_STRESS_EXECUTION":
        raise ValueError("stress protocol is not frozen")
    if sha256(input_path) != protocol["input"]["result_sha256"]:
        raise ValueError("input result hash mismatch")
    if sha256(model_path) != protocol["sealed_student"]["receipt_sha256"]:
        raise ValueError("sealed student hash mismatch")
    if protocol["sealed_student"]["model_id"] != MODEL_ID:
        raise ValueError("sealed model identity mismatch")
    records = input_result.get("records", [])
    if limit is None:
        if len(records) != int(protocol["input"]["required_record_count"]):
            raise ValueError("unexpected formal record count")
        if len({row["parent_id"] for row in records}) != int(protocol["input"]["required_parent_count"]):
            raise ValueError("unexpected formal parent count")
    rgb_layer = protocol["rgb_second_layer"]
    if rgb_layer.get("cached_runner_implemented") is not False:
        raise ValueError("cached-depth runner cannot claim RGB stress")
    if rgb_layer.get("rgb_runner_implemented") is not True:
        raise ValueError("protocol must route RGB stress to its dedicated runner")
    if not protocol_path.is_file():
        raise ValueError("missing protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input-result", type=Path)
    parser.add_argument("--sealed-model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Development smoke only; produces a non-authoritative terminal.",
    )
    arguments = parser.parse_args()
    if arguments.output_root.exists():
        raise FileExistsError(arguments.output_root)
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    input_path = arguments.input_result or REPO_ROOT / protocol["input"]["result_path"]
    model_path = arguments.sealed_model or REPO_ROOT / protocol["sealed_student"]["receipt_path"]
    input_result = json.loads(input_path.read_text(encoding="utf-8"))
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("limit must be positive")
    validate_inputs(
        arguments.protocol,
        protocol,
        input_path,
        model_path,
        input_result,
        arguments.limit,
    )
    student = SealedScaleStudent.load(model_path)
    if tuple(student.feature_names) != FEATURE_NAMES:
        raise ValueError("sealed feature order mismatch")
    scenarios = build_scenarios(protocol)
    source_rows = input_result["records"]
    if arguments.limit is not None:
        source_rows = source_rows[: arguments.limit]
    intrinsics = np.asarray(protocol["input"]["intrinsics"], dtype=np.float64)
    expected_shape = tuple(int(value) for value in protocol["input"]["cached_shape"])
    lower_roi_start = float(
        protocol["cached_depth_perturbations"]["ground_roi_occlusion"]["lower_roi_start_fraction"]
    )

    arguments.output_root.mkdir(parents=True)
    records_path = arguments.output_root / protocol["execution"]["records_file"]
    progress_path = arguments.output_root / protocol["execution"]["progress_file"]
    summaries = []
    verified_prediction_paths: set[Path] = set()
    with records_path.open("x", encoding="utf-8", newline="\n") as records_handle, progress_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as progress_handle:
        for scenario_index, scenario in enumerate(scenarios, 1):
            scenario_rows = []
            for source_index, row in enumerate(source_rows, 1):
                prediction_path = Path(row["prediction_path"])
                if (
                    protocol["execution"]["prediction_hash_verification"]
                    and prediction_path not in verified_prediction_paths
                ):
                    if sha256(prediction_path) != str(row["prediction_sha256"]).upper():
                        raise ValueError(f"prediction hash mismatch: {prediction_path}")
                    verified_prediction_paths.add(prediction_path)
                with np.load(prediction_path) as payload:
                    depth = payload[protocol["input"]["cached_array_key"]].astype(np.float64)
                if depth.shape != expected_shape:
                    raise ValueError(f"unexpected cached depth shape: {depth.shape}")
                stressed_depth, stressed_intrinsics, stressed_height = apply_scenario(
                    depth,
                    intrinsics,
                    float(row["height_m"]),
                    scenario,
                    lower_roi_start,
                )
                record = evaluate_one(
                    row,
                    stressed_depth,
                    stressed_intrinsics,
                    stressed_height,
                    student,
                    scenario,
                )
                scenario_rows.append(record)
                append_jsonl(records_handle, record)
                if source_index % 50 == 0:
                    append_jsonl(
                        progress_handle,
                        {
                            "event": "scenario_progress",
                            "scenario_id": scenario["id"],
                            "processed": source_index,
                            "total": len(source_rows),
                        },
                    )
            summary = summarize_scenario(scenario_rows, scenario)
            summaries.append(summary)
            append_jsonl(
                progress_handle,
                {
                    "event": "scenario_complete",
                    "scenario_index": scenario_index,
                    "scenario_count": len(scenarios),
                    "scenario_id": scenario["id"],
                    "record_count": len(scenario_rows),
                    "unknown_reason_counts": summary["unknown_reason_counts"],
                    "parent_macro": summary["parent_macro"],
                },
            )
            print(
                json.dumps(
                    {
                        "scenario": scenario["id"],
                        "index": scenario_index,
                        "total": len(scenarios),
                    }
                ),
                flush=True,
            )

    clean = summaries[0]
    for summary in summaries:
        add_clean_delta(summary, clean)
    result = {
        "schema": "blindassist_camera_conditioned_scale_student_offline_stress_r0_result_v1",
        "protocol_sha256": sha256(arguments.protocol),
        "input_result_sha256": sha256(input_path),
        "sealed_model_sha256": sha256(model_path),
        "model_id": MODEL_ID,
        "data_role": protocol["data_role"],
        "claim_ceiling": protocol["claim_ceiling"],
        "source_record_count": len(source_rows),
        "source_parent_count": len({row["parent_id"] for row in source_rows}),
        "scenario_count": len(scenarios),
        "records_path": str(records_path.resolve()),
        "progress_path": str(progress_path.resolve()),
        "rgb_blur_low_light_evaluated": False,
        "rgb_second_layer_requirement": protocol["rgb_second_layer"],
        "scenarios": summaries,
        "terminal": (
            "CAMERA_CONDITIONED_SCALE_STUDENT_OFFLINE_STRESS_R0_COMPLETE_CONSUMED_SYNTHETIC_ONLY"
            if arguments.limit is None
            else "DEVELOPMENT_SMOKE_NOT_PROTOCOL_EXECUTION"
        ),
    }
    write_json_new(arguments.output_root / protocol["execution"]["result_file"], result)
    print(json.dumps({"terminal": result["terminal"], "scenario_count": len(scenarios)}, indent=2))


if __name__ == "__main__":
    main()
