"""Evaluate a frozen parent-disjoint lightweight log-scale ridge student."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import core as scale_core

REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_consumed_tartanground import (  # noqa: E402
    INTRINSICS,
    sha256,
    strict_band_values,
    summarize_arm,
    write_json_new,
)
from evaluate_metric3d_clearance_field_a0 import clearance_field  # noqa: E402


RIDGE_ALPHA = 1.0
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


def runtime_features(
    relative_depth: np.ndarray, height_m: float, recovery: dict[str, Any]
) -> np.ndarray | None:
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


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float = RIDGE_ALPHA
) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) <= x.shape[1]:
        raise ValueError("insufficient or malformed ridge training data")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or alpha < 0.0:
        raise ValueError("ridge inputs must be finite and alpha non-negative")
    mean = np.mean(x, axis=0)
    standard_deviation = np.std(x, axis=0)
    standard_deviation[standard_deviation < 1e-9] = 1.0
    standardized = (x - mean) / standard_deviation
    design = np.column_stack((np.ones(len(x)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "standard_deviation": standard_deviation, "weights": weights}


def predict_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    standardized = (x - model["mean"]) / model["standard_deviation"]
    return float(model["weights"][0] + standardized @ model["weights"][1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    r0 = json.loads(arguments.r0_result.read_text(encoding="utf-8"))
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_STUDENT_EFFECT_EXECUTION":
        raise ValueError("student protocol is not frozen")
    if protocol["input_result_sha256"] != sha256(arguments.r0_result):
        raise ValueError("R0 result hash mismatch")
    if tuple(protocol["runtime_features"]) != FEATURE_NAMES:
        raise ValueError("protocol/implementation feature mismatch")
    if float(protocol["student"]["ridge_alpha"]) != RIDGE_ALPHA:
        raise ValueError("protocol/implementation ridge mismatch")

    prepared = []
    for row in r0["records"]:
        with np.load(row["prediction_path"]) as payload:
            relative_depth = payload["da_depth"].astype(np.float64)
        recovery = scale_core.recover_metric_scale(
            relative_depth,
            INTRINSICS,
            scale_core.CameraHeightReceipt(
                row["parent_id"], row["parent_id"], row["height_m"], 0.0
            ),
            row["parent_id"],
            row["parent_id"],
        )
        prepared.append(
            {
                "row": row,
                "relative_depth": relative_depth,
                "recovery": recovery,
                "features": runtime_features(relative_depth, row["height_m"], recovery),
            }
        )

    parents = sorted({item["row"]["parent_id"] for item in prepared})
    if len(parents) != int(protocol["fold_count"]):
        raise ValueError("unexpected parent/fold count")
    output_records = []
    fold_receipts = []
    for test_parent in parents:
        training_items = [
            item
            for item in prepared
            if item["row"]["parent_id"] != test_parent
            and item["features"] is not None
            and item["row"]["aligned_scale_diagnostic"] is not None
        ]
        training_parents = sorted({item["row"]["parent_id"] for item in training_items})
        if len(training_parents) != 4 or test_parent in training_parents:
            raise ValueError("parent leakage or missing training parent")
        x_train = np.stack([item["features"] for item in training_items])
        y_train = np.log(
            np.asarray(
                [item["row"]["aligned_scale_diagnostic"] for item in training_items],
                dtype=np.float64,
            )
        )
        model = fit_ridge(x_train, y_train)
        fold_receipts.append(
            {
                "test_parent_id": test_parent,
                "training_parent_ids": training_parents,
                "training_record_count": len(training_items),
                "feature_mean": model["mean"].tolist(),
                "feature_standard_deviation": model["standard_deviation"].tolist(),
                "weights_intercept_then_features": model["weights"].tolist(),
            }
        )
        for item in (value for value in prepared if value["row"]["parent_id"] == test_parent):
            row = item["row"]
            candidate = None
            reason = None
            predicted_scale = None
            if item["features"] is None:
                reason = str(item["recovery"].get("reason", "INVALID_RUNTIME_FEATURES"))
            else:
                predicted_scale = float(np.exp(predict_ridge(model, item["features"])))
                if not scale_core.SCALE_RANGE[0] <= predicted_scale <= scale_core.SCALE_RANGE[1]:
                    reason = "STUDENT_SCALE_OUT_OF_RANGE"
                else:
                    plane = item["recovery"]["ground"]
                    candidate = strict_band_values(
                        clearance_field(
                            item["relative_depth"] * predicted_scale,
                            INTRINSICS,
                            plane_override=(
                                plane.normal,
                                row["height_m"],
                                plane.normalized_median_residual * row["height_m"],
                            ),
                        )
                    )
                    if candidate is None:
                        reason = "STRICT_CLEARANCE_BAND_UNKNOWN"
            output_records.append(
                {
                    **row,
                    "student_candidate": candidate,
                    "student_unknown_reason": reason,
                    "student_predicted_scale": predicted_scale,
                    "student_test_parent_id": test_parent,
                    "student_training_parent_ids": training_parents,
                }
            )

    summary = summarize_arm(output_records, "student_candidate")
    raw_by_parent = {row["parent_id"]: row for row in r0["raw_da"]["parents"]}
    jointly_better = []
    for row in summary["parents"]:
        raw = raw_by_parent[row["parent_id"]]
        better = (
            row["clearance_mae_m"] is not None
            and raw["clearance_mae_m"] is not None
            and row["clearance_mae_m"] < raw["clearance_mae_m"]
            and row["false_clear_rate"] is not None
            and raw["false_clear_rate"] is not None
            and row["false_clear_rate"] <= raw["false_clear_rate"]
        )
        jointly_better.append({"parent_id": row["parent_id"], "jointly_better": better})
    macro = summary["parent_macro"]
    gates = {
        "known_coverage": macro["known_coverage"] is not None and macro["known_coverage"] >= 0.60,
        "clearance_mae": macro["clearance_mae_m"] is not None and macro["clearance_mae_m"] <= 0.25,
        "envelope_agreement": macro["envelope_agreement"] is not None and macro["envelope_agreement"] >= 0.90,
        "false_clear": macro["false_clear_rate"] is not None and macro["false_clear_rate"] <= 0.05,
        "temporal_delta_mae": macro["temporal_delta_mae_m"] is not None and macro["temporal_delta_mae_m"] <= 0.15,
        "jointly_better_parents": sum(row["jointly_better"] for row in jointly_better) >= 3,
    }
    unknown_reasons: dict[str, int] = {}
    for row in output_records:
        if row["student_candidate"] is None:
            reason = row["student_unknown_reason"] or "UNKNOWN"
            unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1
    result = {
        "schema": "blindassist_camera_conditioned_lightweight_scale_student_r0_result_v1",
        "data_role": protocol["data_role"],
        "claim_ceiling": protocol["claim_ceiling"],
        "r0_result_sha256": sha256(arguments.r0_result),
        "protocol_sha256": sha256(arguments.protocol),
        "feature_names": list(FEATURE_NAMES),
        "ridge_alpha": RIDGE_ALPHA,
        "fold_receipts": fold_receipts,
        "records": output_records,
        "raw_da": r0["raw_da"],
        "r0_known_height_candidate": r0["known_height_candidate"],
        "student_candidate": summary,
        "jointly_better_parents": jointly_better,
        "candidate_unknown_reason_counts": unknown_reasons,
        "gates": gates,
        "terminal": (
            "CAMERA_CONDITIONED_SCALE_STUDENT_R0_ALL_GATES_PASS_REQUIRES_FRESH_TEST"
            if all(gates.values())
            else protocol["failure_terminal"]
        ),
    }
    write_json_new(arguments.output, result)
    print(json.dumps({key: result[key] for key in ("student_candidate", "gates", "terminal")}, indent=2))


if __name__ == "__main__":
    main()
