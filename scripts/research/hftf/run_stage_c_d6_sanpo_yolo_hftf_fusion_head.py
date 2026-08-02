#!/usr/bin/env python3
"""Cross-validate a low-capacity fusion head over YOLO and HFTF signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_BASELINE,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    ManifestFrames,
    aggregate,
    infer_manifest_probabilities,
    load_model,
    reference_comparison,
    score_event,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    FOLD_COUNT,
    FOLD_SEED,
    L2_STRENGTH,
    PRIMARY_CONFIRMATION_STEPS,
    build_training_rows,
    event_balanced_weights,
    feature_matrix,
    fit_logistic,
    fold_assignments,
    predict_event,
    weighted_standardize,
)
from train_stage_c_d5_tartanground_development_student import sha256


DEFAULT_YOLO_TRACE = Path(
    "artifacts.local/evidence/riskseg-r0/event-eval/"
    "device-runs/seed-20260801/results/frame_traces.jsonl"
)
YOLO_ARM = "A_CURRENT_YOLO_ONLY"
RISK_ORDINAL = {
    "NONE": 0.0,
    "LOW": 1.0 / 3.0,
    "MEDIUM": 2.0 / 3.0,
    "HIGH": 1.0,
}
YOLO_FEATURE_NAMES = [
    "yolo/detection-count-max-200ms",
    "yolo/actual-alert-any-200ms",
    "yolo/raw-risk-max-200ms",
    "yolo/stable-risk-max-200ms",
    "yolo/direction-left-any-200ms",
    "yolo/direction-center-any-200ms",
    "yolo/direction-right-any-200ms",
]


def load_yolo_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        row = json.loads(line)
        if row["arm"] != YOLO_ARM:
            continue
        key = (row["parent_event_id"], int(row["frame_index"]))
        if key in output:
            raise ValueError(
                f"Duplicate YOLO trace row at line {line_number}: {key}"
            )
        output[key] = row
    return output


def yolo_feature_matrices(
    events: list[dict[str, Any]],
    rows: dict[tuple[str, int], dict[str, Any]],
) -> list[np.ndarray]:
    matrices = []
    expected_keys = set()
    for event in events:
        event_id = event["parent_event_id"]
        matrix = np.zeros(
            (len(event["frames"]), len(YOLO_FEATURE_NAMES)),
            dtype=np.float64,
        )
        for frame_index, frame in enumerate(event["frames"]):
            window_indices = (
                [frame_index]
                if frame_index == 0
                else [frame_index - 1, frame_index]
            )
            window = []
            for index in window_indices:
                key = (event_id, index)
                expected_keys.add(key)
                if key not in rows:
                    raise ValueError(f"Missing YOLO trace row: {key}")
                row = rows[key]
                expected_source = int(
                    event["frames"][index]["source_frame_index"]
                )
                if int(row["source_frame_index"]) != expected_source:
                    raise ValueError(
                        f"YOLO source-frame mismatch: {key}"
                    )
                window.append(row)
            matrix[frame_index] = [
                max(float(row["detection_count"]) for row in window),
                float(any(bool(row["actual_alert"]) for row in window)),
                max(RISK_ORDINAL[row["raw_risk_level"]] for row in window),
                max(
                    RISK_ORDINAL[row["stable_risk_level"]]
                    for row in window
                ),
                float(any(row["risk_direction"] == "LEFT" for row in window)),
                float(
                    any(
                        row["risk_direction"] == "CENTER"
                        for row in window
                    )
                ),
                float(
                    any(
                        row["risk_direction"] == "RIGHT"
                        for row in window
                    )
                ),
            ]
        matrices.append(matrix)
    if set(rows) != expected_keys:
        extras = sorted(set(rows) - expected_keys)
        raise ValueError(
            f"Unexpected YOLO trace coverage: {len(extras)} extra rows"
        )
    return matrices


def infer_hftf_feature_matrices(
    model: Any,
    dataset: ManifestFrames,
    manifest: dict[str, Any],
    batch_size: int,
) -> tuple[list[np.ndarray], list[str]]:
    risks, knowns = infer_manifest_probabilities(
        model,
        dataset,
        manifest,
        batch_size,
    )
    matrices = []
    feature_names = None
    for risk, known in zip(risks, knowns, strict=True):
        matrix, names = feature_matrix(risk, known)
        matrices.append(matrix)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("HFTF feature order drift")
    if feature_names is None:
        raise ValueError("No HFTF features")
    return matrices, feature_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--yolo-trace", type=Path, default=DEFAULT_YOLO_TRACE)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    events = manifest["events"]
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in events) != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    model, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    dataset = ManifestFrames(args.manifest, manifest)
    hftf_matrices, hftf_feature_names = infer_hftf_feature_matrices(
        model,
        dataset,
        manifest,
        args.batch_size,
    )

    yolo_rows = load_yolo_rows(args.yolo_trace)
    yolo_matrices = yolo_feature_matrices(events, yolo_rows)
    matrices = [
        np.concatenate([hftf, yolo], axis=1)
        for hftf, yolo in zip(
            hftf_matrices,
            yolo_matrices,
            strict=True,
        )
    ]
    feature_names = hftf_feature_names + YOLO_FEATURE_NAMES

    assignments = fold_assignments(events)
    event_outputs: dict[str, dict[str, Any]] = {}
    fold_outputs = []
    for fold in range(FOLD_COUNT):
        x_train, y_train, training_event_ids = build_training_rows(
            events,
            matrices,
            assignments,
            fold,
        )
        weights = event_balanced_weights(
            training_event_ids,
            y_train,
        )
        mean, scale = weighted_standardize(
            x_train,
            weights,
        )
        standardized = (x_train - mean) / scale
        coefficient, intercept, loss = fit_logistic(
            standardized,
            y_train,
            weights,
        )
        heldout = [
            event
            for event in events
            if assignments[event["parent_event_id"]] == fold
        ]
        for event in heldout:
            event_index = events.index(event)
            active, probabilities = predict_event(
                event,
                matrices[event_index],
                coefficient,
                intercept,
                mean,
                scale,
            )
            score = score_event(event, active)
            score["heldout_fold"] = fold
            score["probability_median"] = float(
                np.median(probabilities)
            )
            score["probability_max"] = float(
                np.max(probabilities)
            )
            event_outputs[event["parent_event_id"]] = score
        fold_outputs.append(
            {
                "fold": fold,
                "train_event_count": len(set(training_event_ids)),
                "train_frame_count": len(y_train),
                "train_positive_frame_count": int(y_train.sum()),
                "test_event_ids": sorted(
                    event["parent_event_id"] for event in heldout
                ),
                "weighted_regularized_train_loss": loss,
                "intercept": intercept,
                "coefficient_l2_norm": float(
                    np.linalg.norm(coefficient)
                ),
                "coefficients": {
                    name: float(value)
                    for name, value in zip(
                        feature_names,
                        coefficient,
                        strict=True,
                    )
                },
            }
        )
    if set(event_outputs) != {
        event["parent_event_id"] for event in events
    }:
        raise ValueError("Out-of-fold event coverage mismatch")
    scored_events = [
        event_outputs[event["parent_event_id"]] for event in events
    ]
    metrics = aggregate(scored_events)
    baseline_result = json.loads(
        args.baseline.read_text(encoding="utf-8")
    )
    baseline = baseline_result["event_evaluation"][
        "current_yolo_reference"
    ]
    result = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_"
            "yolo_hftf_fusion_head_cv_v0"
        ),
        "status": (
            "SANPO_YOLO_HFTF_FUSION_HEAD_"
            "CROSS_VALIDATION_COMPLETE"
        ),
        "policy": {
            "data_role": "consumed_development",
            "source_session_heldout": True,
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "fixed_hftf_backbone": True,
            "fixed_yolo_system": True,
            "hftf_feature_family": (
                "five_direction_profiles_x_six_directions"
            ),
            "hftf_feature_count": len(hftf_feature_names),
            "yolo_feature_family": (
                "causal_200ms_detection_alert_risk_direction"
            ),
            "yolo_feature_count": len(YOLO_FEATURE_NAMES),
            "feature_count": len(feature_names),
            "event_balanced_training": True,
            "class_balanced_training": True,
            "l2_strength": L2_STRENGTH,
            "probability_threshold": 0.5,
            "causal_confirmation_steps_at_5hz": (
                PRIMARY_CONFIRMATION_STEPS
            ),
            "test_sessions_used_for_standardization_or_fit": False,
            "human_safety_or_app_claim": False,
        },
        "model": {
            "name": args.name,
            "architecture": checkpoint.get("architecture", "pooled"),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "yolo_trace_path": str(args.yolo_trace.resolve()),
            "yolo_trace_sha256": sha256(args.yolo_trace),
            "yolo_arm": YOLO_ARM,
            "event_count": manifest["event_count"],
            "frame_count": len(dataset),
            "bucket_counts": manifest["bucket_counts"],
        },
        "current_yolo_reference": baseline,
        "metrics": metrics,
        "comparison_to_current_yolo": reference_comparison(
            metrics,
            baseline,
        ),
        "folds": fold_outputs,
        "events": scored_events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
