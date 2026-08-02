#!/usr/bin/env python3
"""Evaluate a sequence-held-out track/IMU metric residual student."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    DEFAULT_PACKETS,
    FUTURE_FRAME_OFFSET,
    HISTORY_COUNT,
    REPO_ROOT,
    ols_slope,
    sha256,
)
from evaluate_stage_c_d33_jrdb_detector_track_future_range import (
    associate,
    load_jsonl,
)
from evaluate_stage_c_d41_jrdb_causal_future_box_field import box_state
from evaluate_stage_c_d42_jrdb_ego_object_metric_teacher import (
    ARM_CURRENT,
    ARM_FULL,
    EXPECTED_PRODUCER_RECEIPT_SHA256,
    EXPECTED_TRACKS_SHA256,
    finite_vector,
    load_packet,
    predict_arms,
    yaw_from_quaternion,
)
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d43_jrdb_"
    "track_imu_metric_residual_student_v0"
)
SUPPORTED_STATUS = (
    "D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_"
    "SUPPORTED_DEVELOPMENT_ONLY"
)
NOT_SUPPORTED_STATUS = (
    "D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_NOT_SUPPORTED"
)
NOT_EVALUABLE_STATUS = (
    "D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_NOT_EVALUABLE"
)
TRACK_ONLY = "TRACK_ONLY"
TRACK_IMU = "TRACK_IMU"
ZERO = "ZERO_RESIDUAL"
FRAME_WIDTH = 3_760.0
FRAME_HEIGHT = 480.0
RIDGE_ALPHA = 1.0
MINIMUM_OPPORTUNITIES = 400
MINIMUM_IDENTITIES = 15
MINIMUM_FOLD_OPPORTUNITIES = 50
TRACK_FEATURE_NAMES = (
    "current_center_x_norm",
    "current_center_y_norm",
    "current_log_width_norm",
    "current_log_height_norm",
    "slope_center_x_norm_per_s",
    "slope_center_y_norm_per_s",
    "slope_log_width_per_s",
    "slope_log_height_per_s",
    "current_confidence",
    "history_mean_confidence",
)
IMU_FEATURE_NAMES = (
    "history_mean_angular_velocity_x",
    "history_mean_angular_velocity_y",
    "history_mean_angular_velocity_z",
    "history_mean_linear_acceleration_x",
    "history_mean_linear_acceleration_y",
    "history_mean_linear_acceleration_z",
    "current_orientation_yaw_sin",
    "current_orientation_yaw_cos",
)
FULL_FEATURE_NAMES = TRACK_FEATURE_NAMES + IMU_FEATURE_NAMES
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts.local/evidence/hftf/"
    "stage-c-d43-jrdb-track-imu-metric-residual-student-v0/report.json"
)


def load_packet_with_imu(
    path: Path,
) -> tuple[str, dict[int, dict[str, Any]], float]:
    sequence, frames, parity_error = load_packet(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for raw_frame in raw["frames"]:
        frame_index = int(raw_frame["frame_index"])
        imu = raw_frame["imu"]
        angular = imu.get("angular_velocity")
        acceleration = imu.get("linear_acceleration")
        orientation_value = imu.get("orientation_xyzw")
        frames[frame_index]["imu_angular_velocity"] = (
            finite_vector(angular, "IMU angular velocity")
            if isinstance(angular, list)
            else None
        )
        frames[frame_index]["imu_linear_acceleration"] = (
            finite_vector(acceleration, "IMU linear acceleration")
            if isinstance(acceleration, list)
            else None
        )
        orientation = (
            np.asarray(orientation_value, dtype=np.float64)
            if isinstance(orientation_value, list)
            else None
        )
        if orientation is not None and (
            orientation.shape != (4,)
            or not np.all(np.isfinite(orientation))
        ):
            raise ValueError("D43 invalid finite IMU orientation")
        frames[frame_index]["imu_orientation_xyzw"] = orientation
    return sequence, frames, parity_error


def imu_frame_complete(frame: dict[str, Any]) -> bool:
    return bool(
        frame["imu_angular_velocity"] is not None
        and frame["imu_linear_acceleration"] is not None
        and frame["imu_orientation_xyzw"] is not None
    )


def imu_source_census(
    source_rows: list[dict[str, Any]],
    frames: dict[int, dict[str, Any]],
) -> dict[str, int]:
    tracks: dict[int, set[int]] = defaultdict(set)
    for row in source_rows:
        tracks[int(row["track_id"])].add(int(row["frame_index"]))
    histories = 0
    complete_histories = 0
    for frame_indices in tracks.values():
        for current_frame in sorted(frame_indices):
            history_frames = list(
                range(
                    current_frame - HISTORY_COUNT + 1,
                    current_frame + 1,
                )
            )
            if any(frame not in frame_indices for frame in history_frames):
                continue
            histories += 1
            if all(imu_frame_complete(frames[frame]) for frame in history_frames):
                complete_histories += 1
    return {
        "frames": len(frames),
        "imu_complete_frames": sum(
            imu_frame_complete(frame) for frame in frames.values()
        ),
        "contiguous_track_histories": histories,
        "imu_complete_track_histories": complete_histories,
    }


def track_features(
    detector_history: list[dict[str, Any]],
    frames: dict[int, dict[str, Any]],
) -> np.ndarray:
    if len(detector_history) != HISTORY_COUNT:
        raise ValueError("D43 detector history must contain seven rows")
    frame_indices = [int(row["frame_index"]) for row in detector_history]
    if any(
        right != left + 1
        for left, right in zip(frame_indices, frame_indices[1:])
    ):
        raise ValueError("D43 detector history is not contiguous")
    timestamps_s = [
        int(frames[frame]["timestamp_ns"]) / 1_000_000_000.0
        for frame in frame_indices
    ]
    states = []
    confidences = []
    for row in detector_history:
        center_x, center_y, log_width, log_height = box_state(
            row["bbox_xyxy"]
        )
        states.append(
            (
                center_x / FRAME_WIDTH,
                center_y / FRAME_HEIGHT,
                log_width - math.log(FRAME_WIDTH),
                log_height - math.log(FRAME_HEIGHT),
            )
        )
        confidence = float(row["confidence"])
        if not math.isfinite(confidence):
            raise ValueError("D43 non-finite detector confidence")
        confidences.append(confidence)
    current = states[-1]
    slopes = [
        ols_slope(
            timestamps_s,
            [state[index] for state in states],
        )
        for index in range(4)
    ]
    features = np.asarray(
        [
            *current,
            *slopes,
            confidences[-1],
            statistics.fmean(confidences),
        ],
        dtype=np.float64,
    )
    if features.shape != (len(TRACK_FEATURE_NAMES),):
        raise ValueError("D43 track feature width drift")
    return features


def imu_features(
    history_frames: list[int],
    frames: dict[int, dict[str, Any]],
) -> np.ndarray:
    angular = np.asarray(
        [frames[frame]["imu_angular_velocity"] for frame in history_frames],
        dtype=np.float64,
    )
    acceleration = np.asarray(
        [frames[frame]["imu_linear_acceleration"] for frame in history_frames],
        dtype=np.float64,
    )
    yaw = yaw_from_quaternion(
        frames[history_frames[-1]]["imu_orientation_xyzw"]
    )
    features = np.asarray(
        [
            *np.mean(angular, axis=0).tolist(),
            *np.mean(acceleration, axis=0).tolist(),
            math.sin(yaw),
            math.cos(yaw),
        ],
        dtype=np.float64,
    )
    if features.shape != (len(IMU_FEATURE_NAMES),):
        raise ValueError("D43 IMU feature width drift")
    return features


def build_sequence_rows(
    sequence: str,
    frames: dict[int, dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_frame[int(row["frame_index"])].append(row)
    matched_identity: dict[tuple[int, int], str] = {}
    for frame_index, frame in sorted(frames.items()):
        source = source_by_frame.get(frame_index, [])
        for source_index, truth_index, _ in associate(
            source,
            frame["truth"],
        ):
            source_row = source[source_index]
            truth_row = frame["truth"][truth_index]
            matched_identity[
                (frame_index, int(source_row["track_id"]))
            ] = str(truth_row["label_id"])
    tracks: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        tracks[int(row["track_id"])][int(row["frame_index"])] = row
    rows = []
    for track_id, by_frame in sorted(tracks.items()):
        for current_frame in sorted(by_frame):
            history_frames = list(
                range(
                    current_frame - HISTORY_COUNT + 1,
                    current_frame + 1,
                )
            )
            if any(frame not in by_frame for frame in history_frames):
                continue
            label_id = matched_identity.get((current_frame, track_id))
            if label_id is None:
                continue
            future_frame_index = current_frame + FUTURE_FRAME_OFFSET
            future_frame = frames.get(future_frame_index)
            if future_frame is None:
                continue
            future_truth = future_frame["truth_by_id"].get(label_id)
            if future_truth is None:
                continue
            if any(
                label_id not in frames[frame]["truth_by_id"]
                for frame in history_frames
            ):
                continue
            detector_history = [by_frame[frame] for frame in history_frames]
            teacher_history = []
            for frame_index in history_frames:
                frame = frames[frame_index]
                identity = frame["truth_by_id"][label_id]
                teacher_history.append(
                    {
                        "timestamp_ns": frame["timestamp_ns"],
                        "pose_translation": frame["pose_translation"],
                        "pose_quaternion": frame["pose_quaternion"],
                        "center_base_link_m": identity[
                            "center_base_link_m"
                        ],
                        "center_odom_m": identity["center_odom_m"],
                    }
                )
            teacher_predictions = predict_arms(
                teacher_history,
                int(future_frame["timestamp_ns"]),
            )
            current_center = teacher_predictions[ARM_CURRENT][:2]
            teacher_target = (
                teacher_predictions[ARM_FULL][:2] - current_center
            )
            actual_target = (
                finite_vector(
                    future_truth["center_base_link_m"],
                    "future center",
                )[:2]
                - current_center
            )
            track = track_features(detector_history, frames)
            full = np.concatenate(
                [track, imu_features(history_frames, frames)]
            )
            arrays = (track, full, teacher_target, actual_target)
            if not all(np.all(np.isfinite(value)) for value in arrays):
                raise ValueError("D43 non-finite dataset row")
            rows.append(
                {
                    "sequence": sequence,
                    "native_label_id": label_id,
                    "track_features": track,
                    "full_features": full,
                    "teacher_target": teacher_target,
                    "actual_target": actual_target,
                }
            )
    return rows


def vector_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("D43 invalid prediction/target shape")
    errors = np.linalg.norm(predictions - targets, axis=1)
    if not np.all(np.isfinite(errors)):
        raise ValueError("D43 non-finite vector error")
    return errors


def fit_predict(
    training_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = np.asarray(
        [row[feature_key] for row in training_rows],
        dtype=np.float64,
    )
    y_train = np.asarray(
        [row["teacher_target"] for row in training_rows],
        dtype=np.float64,
    )
    x_test = np.asarray(
        [row[feature_key] for row in test_rows],
        dtype=np.float64,
    )
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)
    x_train_scaled = (x_train - feature_mean) / feature_scale
    x_test_scaled = (x_test - feature_mean) / feature_scale
    target_mean = np.mean(y_train, axis=0)
    y_centered = y_train - target_mean
    gram = x_train_scaled.T @ x_train_scaled
    coefficient = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(gram.shape[0]),
        x_train_scaled.T @ y_centered,
    )
    predictions = x_test_scaled @ coefficient + target_mean
    receipt = {
        "training_rows": len(training_rows),
        "test_rows": len(test_rows),
        "feature_count": int(x_train.shape[1]),
        "coefficient_count": int(coefficient.size),
        "maximum_absolute_coefficient": float(
            np.max(np.abs(coefficient))
        ),
        "maximum_absolute_intercept": float(
            np.max(np.abs(target_mean))
        ),
        "minimum_feature_scale": float(np.min(feature_scale)),
        "maximum_feature_scale": float(np.max(feature_scale)),
    }
    if (
        not np.all(np.isfinite(predictions))
        or not all(
            math.isfinite(float(value))
            for value in receipt.values()
            if isinstance(value, float)
        )
    ):
        raise ValueError("D43 non-finite model output")
    return predictions, receipt


def evaluate_folds(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequences = sorted({str(row["sequence"]) for row in rows})
    if len(sequences) != 4:
        raise ValueError("D43 sequence count drift")
    fold_summaries = []
    prediction_rows = []
    for test_sequence in sequences:
        training = [
            row for row in rows if str(row["sequence"]) != test_sequence
        ]
        test = [
            row for row in rows if str(row["sequence"]) == test_sequence
        ]
        track_predictions, track_receipt = fit_predict(
            training,
            test,
            "track_features",
        )
        full_predictions, full_receipt = fit_predict(
            training,
            test,
            "full_features",
        )
        zero_predictions = np.zeros((len(test), 2), dtype=np.float64)
        for index, row in enumerate(test):
            prediction_rows.append(
                {
                    **row,
                    "fold": test_sequence,
                    "predictions": {
                        ZERO: zero_predictions[index],
                        TRACK_ONLY: track_predictions[index],
                        TRACK_IMU: full_predictions[index],
                    },
                }
            )
        fold_rows = prediction_rows[-len(test):]
        summary = summarize_predictions(fold_rows)
        summary["test_sequence"] = test_sequence
        summary["training_sequences"] = [
            sequence
            for sequence in sequences
            if sequence != test_sequence
        ]
        summary["model_receipts"] = {
            TRACK_ONLY: track_receipt,
            TRACK_IMU: full_receipt,
        }
        fold_summaries.append(summary)
    return fold_summaries, prediction_rows


def relative_reduction(baseline: float, candidate: float) -> float | None:
    return (
        (baseline - candidate) / baseline
        if baseline > 0
        else None
    )


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_targets = np.asarray(
        [row["teacher_target"] for row in rows],
        dtype=np.float64,
    )
    actual_targets = np.asarray(
        [row["actual_target"] for row in rows],
        dtype=np.float64,
    )
    arms = {}
    error_cache = {}
    for arm in (ZERO, TRACK_ONLY, TRACK_IMU):
        predictions = np.asarray(
            [row["predictions"][arm] for row in rows],
            dtype=np.float64,
        )
        teacher_errors = vector_errors(predictions, teacher_targets)
        actual_errors = vector_errors(predictions, actual_targets)
        error_cache[arm] = {
            "teacher": teacher_errors,
            "actual": actual_errors,
        }
        arms[arm] = {
            "mean_teacher_vector_error_m": float(
                np.mean(teacher_errors)
            ),
            "mean_actual_future_vector_error_m": float(
                np.mean(actual_errors)
            ),
        }
    zero = arms[ZERO]
    track = arms[TRACK_ONLY]
    full = arms[TRACK_IMU]
    return {
        "opportunities": len(rows),
        "distinct_native_identities": len(
            {
                (str(row["sequence"]), str(row["native_label_id"]))
                for row in rows
            }
        ),
        "arms": arms,
        "track_imu_vs_zero": {
            "teacher_error_relative_reduction": relative_reduction(
                float(zero["mean_teacher_vector_error_m"]),
                float(full["mean_teacher_vector_error_m"]),
            ),
            "actual_error_relative_reduction": relative_reduction(
                float(zero["mean_actual_future_vector_error_m"]),
                float(full["mean_actual_future_vector_error_m"]),
            ),
            "actual_error_better_fraction": float(
                np.mean(
                    error_cache[TRACK_IMU]["actual"]
                    < error_cache[ZERO]["actual"]
                )
            ),
        },
        "track_imu_vs_track_only": {
            "actual_error_relative_delta": (
                float(full["mean_actual_future_vector_error_m"])
                / float(track["mean_actual_future_vector_error_m"])
                - 1.0
                if float(track["mean_actual_future_vector_error_m"]) > 0
                else None
            ),
            "teacher_error_relative_delta": (
                float(full["mean_teacher_vector_error_m"])
                / float(track["mean_teacher_vector_error_m"])
                - 1.0
                if float(track["mean_teacher_vector_error_m"]) > 0
                else None
            ),
        },
    }


def flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [
            item
            for nested in value.values()
            for item in flatten_values(nested)
        ]
    if isinstance(value, list):
        return [
            item
            for nested in value
            for item in flatten_values(nested)
        ]
    return [value]


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
                len(row["training_sequences"]) == 3
                and int(row["opportunities"])
                >= MINIMUM_FOLD_OPPORTUNITIES
                for row in by_fold
            )
        ),
        "feature_width": all(
            int(row["model_receipts"][TRACK_ONLY]["feature_count"])
            == len(TRACK_FEATURE_NAMES)
            and int(row["model_receipts"][TRACK_IMU]["feature_count"])
            == len(FULL_FEATURE_NAMES)
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
    effect = pooled["track_imu_vs_zero"]
    actual_fold_reductions = [
        float(row["track_imu_vs_zero"]["actual_error_relative_reduction"])
        for row in by_fold
    ]
    teacher_fold_reductions = [
        float(row["track_imu_vs_zero"]["teacher_error_relative_reduction"])
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
        "track_imu_noninferior_to_track_only": (
            float(
                pooled["track_imu_vs_track_only"][
                    "actual_error_relative_delta"
                ]
            )
            <= 0.05
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
        raise ValueError("D43 detector-track binding drift")
    if receipt_hash != EXPECTED_PRODUCER_RECEIPT_SHA256:
        raise ValueError("D43 producer-receipt binding drift")
    receipt = json.loads(
        args.producer_receipt.read_text(encoding="utf-8")
    )
    source_rows = load_jsonl(args.tracks)
    source_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_sequence[str(row["sequence"])].append(row)
    all_rows = []
    packet_bindings = {}
    maximum_parity_error = 0.0
    packet_frames = {}
    imu_census = {}
    for packet_path in args.packets:
        sequence, frames, parity_error = load_packet_with_imu(packet_path)
        maximum_parity_error = max(maximum_parity_error, parity_error)
        packet_bindings[sequence] = sha256(packet_path)
        packet_frames[sequence] = frames
        imu_census[sequence] = imu_source_census(
            source_by_sequence.get(sequence, []),
            frames,
        )
    if any(
        int(row["imu_complete_track_histories"])
        < MINIMUM_FOLD_OPPORTUNITIES
        for row in imu_census.values()
    ):
        payload = {
            "schema": SCHEMA,
            "status": NOT_EVALUABLE_STATUS,
            "evaluable": False,
            "supported": False,
            "reason": "IMU_SEQUENCE_COVERAGE_INSUFFICIENT",
            "model_training_executed": False,
            "future_outcome_evaluated": False,
            "source": {
                "frames": int(receipt["frame_count"]),
                "track_occurrences": len(source_rows),
                "sequences": 4,
                "maximum_transform_parity_error_m": maximum_parity_error,
                "imu_census_by_sequence": imu_census,
            },
            "bindings": {
                "tracks_sha256": tracks_hash,
                "producer_receipt_sha256": receipt_hash,
                "packet_sha256": packet_bindings,
            },
            "claims": {
                "track_imu_student": False,
                "track_only_student": False,
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
    for sequence, frames in packet_frames.items():
        all_rows.extend(
            build_sequence_rows(
                sequence,
                frames,
                source_by_sequence.get(sequence, []),
            )
        )
    by_fold, prediction_rows = evaluate_folds(all_rows)
    pooled = summarize_predictions(prediction_rows)
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
            "type": "StandardScaler + multi-output Ridge",
            "alpha": RIDGE_ALPHA,
            "track_feature_names": list(TRACK_FEATURE_NAMES),
            "track_imu_feature_names": list(FULL_FEATURE_NAMES),
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
            "sequence_held_out_offline_learnability": True,
            "teacher_target_uses_native_history_geometry": True,
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
