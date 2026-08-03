#!/usr/bin/env python3
"""Train and evaluate the frozen DA-feature metric calibration head."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    REPO_ROOT,
    fit_dense_affine,
    frame_key,
    load_json,
    report_frames,
    resolve,
    sha256,
    write_json_new,
)

HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_metric3d_clearance_field_a0 import clearance_field, summarize
from produce_external_rgb_metric_depth_observations import intrinsics_matrix

SCHEMA = "blindassist_metric_depth_calibration_head_distillation_r0_result"
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf/METRIC_DEPTH_CALIBRATION_HEAD_DISTILLATION_R0_PROTOCOL_2026-08-03.json"
)
DEFAULT_DENSE_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf/DENSE_METRIC_DEPTH_PROPAGATION_R0_PROTOCOL_2026-08-03.json"
)


def fit_ridge_head(
    features: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
) -> dict[str, np.ndarray]:
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    standardized = (features - feature_mean) / feature_std
    target_mean = np.mean(targets, axis=0)
    centered_targets = targets - target_mean
    gram = standardized.T @ standardized
    kernel = np.linalg.solve(
        gram + ridge_lambda * np.eye(gram.shape[0]),
        standardized.T @ centered_targets,
    )
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "kernel": kernel,
    }


def predict_head(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["feature_mean"]) / model["feature_std"]
    return model["target_mean"] + standardized @ model["kernel"]


def valid_parameters(parameters: np.ndarray, protocol: dict[str, Any]) -> bool:
    slope, offset = (float(value) for value in parameters)
    lower_slope, upper_slope = protocol["student"]["output_validity"]["slope_bounds"]
    lower_offset, upper_offset = protocol["student"]["output_validity"][
        "offset_bounds_m"
    ]
    return (
        np.all(np.isfinite(parameters))
        and lower_slope <= slope <= upper_slope
        and lower_offset <= offset <= upper_offset
    )


def calibrated_field(
    depth: np.ndarray,
    parameters: np.ndarray,
    row: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if not valid_parameters(parameters, protocol):
        return {"status": "UNKNOWN_HEAD_OUTPUT_BOUNDS"}
    slope, offset = (float(value) for value in parameters)
    calibrated = np.clip(slope * np.asarray(depth, dtype=np.float32) + offset, 0, 300)
    return clearance_field(calibrated, intrinsics_matrix(row))


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "frames"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dense-protocol", type=Path, default=DEFAULT_DENSE_PROTOCOL)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    dense_protocol = load_json(args.dense_protocol)
    if protocol.get("status") != "FROZEN_BEFORE_TEACHER_LABEL_OR_HEAD_OUTCOME_READ":
        raise ValueError("head protocol not frozen")
    manifest = load_json(args.cache_root / "manifest.json")
    if manifest.get("protocol_sha256") != sha256(args.dense_protocol):
        raise ValueError("dense cache protocol mismatch")
    paths = {
        key: Path(manifest["outputs"][key]["path"])
        for key in ("dav2_depth", "metric3d_depth", "dav2_layer11_cls")
    }
    for key, path in paths.items():
        if sha256(path) != manifest["outputs"][key]["sha256"]:
            raise ValueError(f"cache hash mismatch: {key}")
    da_depth = np.load(paths["dav2_depth"], mmap_mode="r")
    metric_depth = np.load(paths["metric3d_depth"], mmap_mode="r")
    features = np.asarray(
        np.load(paths["dav2_layer11_cls"], mmap_mode="r"), dtype=np.float64
    )

    cache_rows = manifest["rows"]
    if len(cache_rows) != len(features):
        raise ValueError("cache row count differs from feature count")
    frames = [
        {
            "sequence_id": row["sequence_id"],
            "timestamp": row["timestamp"],
            "frame_path": row["frame_path"],
            "intrinsics_fx_fy_cx_cy": row["intrinsics_fx_fy_cx_cy"],
        }
        for row in cache_rows
    ]

    affine_config = dense_protocol["candidate"]["affine_fit"]
    teacher_targets = np.full((len(frames), 2), np.nan, dtype=np.float64)
    teacher_status = []
    for index in range(len(frames)):
        fit = fit_dense_affine(da_depth[index], metric_depth[index], affine_config)
        teacher_status.append(fit["status"])
        if fit["status"] == "VALID":
            teacher_targets[index] = [fit["slope"], fit["intercept_m"]]

    sequences = sorted({str(row["sequence_id"]) for row in frames})
    if len(sequences) != 4:
        raise ValueError("frozen head experiment requires four sequences")
    ridge_lambda = float(protocol["student"]["ridge_lambda"])
    arm_names = ("raw_da", "constant_affine", "feature_head", "metric3d_oracle")
    fold_candidates = []
    predictions = np.full_like(teacher_targets, np.nan)
    constant_predictions = np.full_like(teacher_targets, np.nan)

    for heldout in sequences:
        train_indices = np.asarray(
            [
                index
                for index, row in enumerate(frames)
                if str(row["sequence_id"]) != heldout
                and np.all(np.isfinite(teacher_targets[index]))
            ],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [
                index
                for index, row in enumerate(frames)
                if str(row["sequence_id"]) == heldout
            ],
            dtype=np.int64,
        )
        if len(train_indices) < 60 or len(test_indices) != 30:
            raise ValueError("unexpected LOSO opportunity")
        model = fit_ridge_head(
            features[train_indices], teacher_targets[train_indices], ridge_lambda
        )
        predictions[test_indices] = predict_head(model, features[test_indices])
        constant = np.median(teacher_targets[train_indices], axis=0)
        constant_predictions[test_indices] = constant

        truth_free_rows: dict[str, list[dict[str, Any]]] = {
            arm: [] for arm in arm_names
        }
        for index in test_indices:
            candidate_fields = {
                "raw_da": clearance_field(
                    da_depth[index], intrinsics_matrix(frames[index])
                ),
                "constant_affine": calibrated_field(
                    da_depth[index],
                    constant_predictions[index],
                    frames[index],
                    protocol,
                ),
                "feature_head": calibrated_field(
                    da_depth[index], predictions[index], frames[index], protocol
                ),
                "metric3d_oracle": clearance_field(
                    metric_depth[index], intrinsics_matrix(frames[index])
                ),
            }
            for arm, candidate in candidate_fields.items():
                truth_free_rows[arm].append(
                    {"index": int(index), "candidate": candidate}
                )
        fold_candidates.append(
            {
                "heldout_sequence": heldout,
                "train_frames": len(train_indices),
                "test_frames": len(test_indices),
                "truth_free_rows": truth_free_rows,
            }
        )

    valid_all = np.all(np.isfinite(teacher_targets), axis=1)
    final_model = fit_ridge_head(
        features[valid_all], teacher_targets[valid_all], ridge_lambda
    )

    # Do not open the hash-bound sensor report until all fold predictions exist.
    reports = []
    for key in ("metric_report", "fast_report"):
        receipt = dense_protocol["inputs"][key]
        path = resolve(receipt["path"])
        if sha256(path) != receipt["sha256"]:
            raise ValueError(f"report hash mismatch: {key}")
        reports.append(report_frames(load_json(path)))
    metric_frames, fast_frames = reports
    expected_keys = [frame_key(row) for row in frames]
    if [frame_key(row) for row in metric_frames] != expected_keys:
        raise ValueError("metric report frames differ from truth-free cache rows")
    if [frame_key(row) for row in fast_frames] != expected_keys:
        raise ValueError("fast report frames differ from truth-free cache rows")

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arm_names}
    fold_results = []
    for fold in fold_candidates:
        fold_arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arm_names}
        for arm, rows in fold["truth_free_rows"].items():
            for row in rows:
                index = int(row["index"])
                joined = {
                    "sequence_root": metric_frames[index].get("sequence_root"),
                    "sequence_id": metric_frames[index]["sequence_id"],
                    "timestamp": metric_frames[index]["timestamp"],
                    "frame_path": metric_frames[index]["frame_path"],
                    "latency_ms": float(fast_frames[index]["latency_ms"]),
                    "candidate": copy.deepcopy(row["candidate"]),
                    "sensor": copy.deepcopy(metric_frames[index]["sensor"]),
                }
                fold_arm_rows[arm].append(joined)
                arm_rows[arm].append(joined)
        fold_results.append(
            {
                "heldout_sequence": fold["heldout_sequence"],
                "train_frames": fold["train_frames"],
                "test_frames": fold["test_frames"],
                "arms": {
                    arm: compact(summarize(rows)) for arm, rows in fold_arm_rows.items()
                },
            }
        )

    aggregates = {arm: compact(summarize(rows)) for arm, rows in arm_rows.items()}
    head_vs_raw_positive = sum(
        fold["arms"]["feature_head"]["clearance_mae_m"]
        < fold["arms"]["raw_da"]["clearance_mae_m"]
        for fold in fold_results
    )
    head_vs_constant_positive = sum(
        fold["arms"]["feature_head"]["clearance_mae_m"]
        < fold["arms"]["constant_affine"]["clearance_mae_m"]
        for fold in fold_results
    )
    trainable_parameters = int(protocol["student"]["trainable_parameters"])
    increment_gates = {
        "head_mae_better_than_raw_positive_folds": head_vs_raw_positive
        >= int(
            protocol["evaluation"]["increment_gates"][
                "head_mae_better_than_raw_positive_folds_min"
            ]
        ),
        "head_mae_better_than_constant_positive_folds": head_vs_constant_positive
        >= int(
            protocol["evaluation"]["increment_gates"][
                "head_mae_better_than_constant_positive_folds_min"
            ]
        ),
        "source_macro_false_clear_not_worse_than_constant": float(
            np.mean(
                [
                    fold["arms"]["feature_head"]["false_clear_rate"]
                    for fold in fold_results
                ]
            )
        )
        <= float(
            np.mean(
                [
                    fold["arms"]["constant_affine"]["false_clear_rate"]
                    for fold in fold_results
                ]
            )
        ),
        "parameter_budget": trainable_parameters
        <= int(
            protocol["evaluation"]["increment_gates"]["maximum_trainable_parameters"]
        ),
    }
    teacher_admissible = all(aggregates["metric3d_oracle"]["gates"].values())
    supported = (
        teacher_admissible
        and all(aggregates["feature_head"]["gates"].values())
        and all(increment_gates.values())
    )

    if args.model_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.model_output}")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    partial_model = args.model_output.with_name(args.model_output.stem + ".partial.npz")
    if partial_model.exists():
        raise FileExistsError(f"partial model exists: {partial_model}")
    np.savez(
        partial_model,
        feature_mean=final_model["feature_mean"].astype(np.float32),
        feature_std=final_model["feature_std"].astype(np.float32),
        target_mean=final_model["target_mean"].astype(np.float32),
        kernel=final_model["kernel"].astype(np.float32),
    )
    os.replace(partial_model, args.model_output)
    result = {
        "schema": SCHEMA,
        "protocol_sha256": sha256(args.protocol),
        "dense_cache_manifest_sha256": sha256(args.cache_root / "manifest.json"),
        "data_role": protocol["authority"]["data_role"],
        "fresh_data_opened": False,
        "sensor_truth_used_for_training": False,
        "sensor_truth_opened_after_all_fold_predictions": True,
        "teacher_target_status_counts": {
            status: teacher_status.count(status)
            for status in sorted(set(teacher_status))
        },
        "folds": fold_results,
        "aggregates": aggregates,
        "increment": {
            "head_vs_raw_positive_folds": head_vs_raw_positive,
            "head_vs_constant_positive_folds": head_vs_constant_positive,
            "gates": increment_gates,
        },
        "student": {
            "feature_dimension": int(features.shape[1]),
            "trainable_parameters": trainable_parameters,
            "all_consumed_model_path": str(args.model_output.resolve()),
            "all_consumed_model_sha256": sha256(args.model_output),
        },
        "teacher_admissible_on_consumed_proxy": teacher_admissible,
        "terminal": (
            "CALIBRATION_HEAD_CONSUMED_DEVELOPMENT_SUPPORTED_FRESH_PENDING"
            if supported
            else "CALIBRATION_HEAD_CONSUMED_DEVELOPMENT_NOT_SUPPORTED"
        ),
        "claim_ceiling": "four consumed TUM sequences; student similarity to Metric3D is not metric truth or phone deployment evidence",
    }
    write_json_new(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "folds"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
