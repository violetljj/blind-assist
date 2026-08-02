#!/usr/bin/env python3
"""Evaluate the frozen D43.1 track-only metric residual student."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    DEFAULT_PACKETS,
    REPO_ROOT,
    sha256,
)
from evaluate_stage_c_d33_jrdb_detector_track_future_range import load_jsonl
from evaluate_stage_c_d42_jrdb_ego_object_metric_teacher import (
    EXPECTED_PRODUCER_RECEIPT_SHA256,
    EXPECTED_TRACKS_SHA256,
)
from evaluate_stage_c_d43_jrdb_track_imu_metric_residual_student import (
    MINIMUM_FOLD_OPPORTUNITIES,
    MINIMUM_IDENTITIES,
    MINIMUM_OPPORTUNITIES,
    RIDGE_ALPHA,
    TRACK_FEATURE_NAMES,
    build_sequence_rows,
    fit_predict,
    flatten_values,
    load_packet_with_imu,
    relative_reduction,
    vector_errors,
)
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d43_1_jrdb_"
    "track_only_metric_residual_student_v0"
)
SUPPORTED_STATUS = (
    "D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_"
    "SUPPORTED_DEVELOPMENT_ONLY"
)
NOT_SUPPORTED_STATUS = (
    "D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_NOT_SUPPORTED"
)
NOT_EVALUABLE_STATUS = (
    "D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_NOT_EVALUABLE"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts.local/evidence/hftf/"
    "stage-c-d43-1-jrdb-track-only-metric-residual-student-v0/report.json"
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher = np.asarray(
        [row["teacher_target"] for row in rows],
        dtype=np.float64,
    )
    actual = np.asarray(
        [row["actual_target"] for row in rows],
        dtype=np.float64,
    )
    zero = np.zeros((len(rows), 2), dtype=np.float64)
    candidate = np.asarray(
        [row["prediction"] for row in rows],
        dtype=np.float64,
    )
    zero_teacher = vector_errors(zero, teacher)
    candidate_teacher = vector_errors(candidate, teacher)
    zero_actual = vector_errors(zero, actual)
    candidate_actual = vector_errors(candidate, actual)
    zero_teacher_mean = float(np.mean(zero_teacher))
    candidate_teacher_mean = float(np.mean(candidate_teacher))
    zero_actual_mean = float(np.mean(zero_actual))
    candidate_actual_mean = float(np.mean(candidate_actual))
    return {
        "opportunities": len(rows),
        "distinct_native_identities": len(
            {
                (str(row["sequence"]), str(row["native_label_id"]))
                for row in rows
            }
        ),
        "zero": {
            "mean_teacher_vector_error_m": zero_teacher_mean,
            "mean_actual_future_vector_error_m": zero_actual_mean,
        },
        "track_only": {
            "mean_teacher_vector_error_m": candidate_teacher_mean,
            "mean_actual_future_vector_error_m": candidate_actual_mean,
        },
        "track_only_vs_zero": {
            "teacher_error_relative_reduction": relative_reduction(
                zero_teacher_mean,
                candidate_teacher_mean,
            ),
            "actual_error_relative_reduction": relative_reduction(
                zero_actual_mean,
                candidate_actual_mean,
            ),
            "actual_error_better_fraction": float(
                np.mean(candidate_actual < zero_actual)
            ),
        },
    }


def evaluate_folds(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequences = sorted({str(row["sequence"]) for row in rows})
    if len(sequences) != 4:
        raise ValueError("D43.1 sequence count drift")
    by_fold = []
    predictions = []
    for test_sequence in sequences:
        train = [
            row for row in rows if str(row["sequence"]) != test_sequence
        ]
        test = [
            row for row in rows if str(row["sequence"]) == test_sequence
        ]
        predicted, receipt = fit_predict(
            train,
            test,
            "track_features",
        )
        fold_rows = [
            {
                **row,
                "prediction": predicted[index],
            }
            for index, row in enumerate(test)
        ]
        predictions.extend(fold_rows)
        summary = summarize(fold_rows)
        summary["test_sequence"] = test_sequence
        summary["training_sequences"] = [
            sequence for sequence in sequences if sequence != test_sequence
        ]
        summary["model_receipt"] = receipt
        by_fold.append(summary)
    return by_fold, predictions


def determine_terminal(
    pooled: dict[str, Any],
    by_fold: list[dict[str, Any]],
    source_frames: int,
    maximum_transform_parity_error_m: float,
) -> tuple[dict[str, bool], dict[str, bool], str]:
    evaluability = {
        "source_binding": (
            source_frames == 480
            and maximum_transform_parity_error_m <= 1e-9
        ),
        "opportunity_count": (
            int(pooled["opportunities"]) >= MINIMUM_OPPORTUNITIES
        ),
        "identity_count": (
            int(pooled["distinct_native_identities"]) >= MINIMUM_IDENTITIES
        ),
        "four_isolated_folds": (
            len(by_fold) == 4
            and all(
                int(row["opportunities"]) >= MINIMUM_FOLD_OPPORTUNITIES
                and len(row["training_sequences"]) == 3
                for row in by_fold
            )
        ),
        "feature_width": all(
            int(row["model_receipt"]["feature_count"])
            == len(TRACK_FEATURE_NAMES)
            for row in by_fold
        ),
        "finite_outputs": all(
            value is None
            or not isinstance(value, float)
            or math.isfinite(value)
            for row in [pooled, *by_fold]
            for value in flatten_values(row)
        ),
    }
    effect = pooled["track_only_vs_zero"]
    teacher_fold_reductions = [
        float(row["track_only_vs_zero"]["teacher_error_relative_reduction"])
        for row in by_fold
    ]
    actual_fold_reductions = [
        float(row["track_only_vs_zero"]["actual_error_relative_reduction"])
        for row in by_fold
    ]
    support = {
        "pooled_teacher_error_reduction": (
            float(effect["teacher_error_relative_reduction"]) >= 0.20
        ),
        "pooled_actual_error_reduction": (
            float(effect["actual_error_relative_reduction"]) >= 0.10
        ),
        "actual_better_fraction": (
            float(effect["actual_error_better_fraction"]) >= 0.55
        ),
        "teacher_fold_breadth": (
            sum(value > 0 for value in teacher_fold_reductions) >= 3
        ),
        "actual_fold_breadth": (
            sum(value > 0 for value in actual_fold_reductions) >= 3
        ),
        "no_actual_fold_material_harm": all(
            value >= -0.05 for value in actual_fold_reductions
        ),
    }
    if not all(evaluability.values()):
        status = NOT_EVALUABLE_STATUS
    elif all(support.values()):
        status = SUPPORTED_STATUS
    else:
        status = NOT_SUPPORTED_STATUS
    return evaluability, support, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument(
        "--producer-receipt",
        type=Path,
        default=DEFAULT_PRODUCER_RECEIPT,
    )
    parser.add_argument(
        "--packets",
        type=Path,
        nargs=4,
        default=DEFAULT_PACKETS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tracks_hash = sha256(args.tracks)
    receipt_hash = sha256(args.producer_receipt)
    if tracks_hash != EXPECTED_TRACKS_SHA256:
        raise ValueError("D43.1 detector-track binding drift")
    if receipt_hash != EXPECTED_PRODUCER_RECEIPT_SHA256:
        raise ValueError("D43.1 producer-receipt binding drift")
    receipt = json.loads(
        args.producer_receipt.read_text(encoding="utf-8")
    )
    source_rows = load_jsonl(args.tracks)
    source_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_sequence[str(row["sequence"])].append(row)
    all_rows = []
    maximum_parity_error = 0.0
    packet_bindings = {}
    for packet_path in args.packets:
        sequence, frames, parity_error = load_packet_with_imu(packet_path)
        maximum_parity_error = max(maximum_parity_error, parity_error)
        packet_bindings[sequence] = sha256(packet_path)
        all_rows.extend(
            build_sequence_rows(
                sequence,
                frames,
                source_by_sequence.get(sequence, []),
                include_imu=False,
            )
        )
    by_fold, prediction_rows = evaluate_folds(all_rows)
    pooled = summarize(prediction_rows)
    evaluability, support, status = determine_terminal(
        pooled,
        by_fold,
        int(receipt["frame_count"]),
        maximum_parity_error,
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "evaluable": all(evaluability.values()),
        "supported": status == SUPPORTED_STATUS,
        "model": {
            "type": "population StandardScaler + multi-output Ridge",
            "alpha": RIDGE_ALPHA,
            "feature_names": list(TRACK_FEATURE_NAMES),
            "training_target": "D42_HISTORY_ONLY_METRIC_DISPLACEMENT_XY",
            "outer_split": "LEAVE_ONE_SEQUENCE_OUT_4_FOLDS",
        },
        "source": {
            "frames": int(receipt["frame_count"]),
            "track_occurrences": len(source_rows),
            "sequences": 4,
            "maximum_transform_parity_error_m": maximum_parity_error,
        },
        "pooled": pooled,
        "by_fold": by_fold,
        "evaluability_gates": evaluability,
        "support_gates": support,
        "bindings": {
            "tracks_sha256": tracks_hash,
            "producer_receipt_sha256": receipt_hash,
            "packet_sha256": packet_bindings,
        },
        "claims": {
            "sequence_held_out_track_only_learnability": True,
            "imu_evaluated": False,
            "inference_uses_native_geometry": False,
            "future_truth_used_for_training": False,
            "event_utility": False,
            "android_runtime": False,
            "mainline_promotion": False,
            "default_app_changed": False,
            "product_or_safety": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256(args.output)}  {args.output.name}\n",
        encoding="ascii",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
