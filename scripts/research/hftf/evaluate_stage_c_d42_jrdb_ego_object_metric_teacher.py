#!/usr/bin/env python3
"""Evaluate a causal ego/object metric-geometry teacher on JRDB."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    DEFAULT_PACKETS,
    FUTURE_FRAME_OFFSET,
    HISTORY_COUNT,
    REPO_ROOT,
    sha256,
)
from evaluate_stage_c_d33_jrdb_detector_track_future_range import (
    associate,
    load_jsonl,
)
from evaluate_stage_c_d41_jrdb_causal_future_box_field import ols_predict
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = "blindassist_hftf_stage_c_d42_jrdb_ego_object_metric_teacher_v0"
SUPPORTED_STATUS = (
    "D42_JRDB_EGO_OBJECT_METRIC_TEACHER_SUPPORTED_DEVELOPMENT_ONLY"
)
NOT_SUPPORTED_STATUS = (
    "D42_JRDB_EGO_OBJECT_METRIC_TEACHER_NOT_SUPPORTED"
)
NOT_EVALUABLE_STATUS = (
    "D42_JRDB_EGO_OBJECT_METRIC_TEACHER_NOT_EVALUABLE"
)
EXPECTED_TRACKS_SHA256 = (
    "efa249fdfe8114dfeb1da419ffdb359189e3d4e6b1f406fabad04a31a39a0fa1"
)
EXPECTED_PRODUCER_RECEIPT_SHA256 = (
    "fa91162274222b9fe2254ae675ccb95af3fcdd6dca50ab267d476d74764be318"
)
MINIMUM_OPPORTUNITIES = 400
MINIMUM_IDENTITIES = 15
MINIMUM_SEQUENCE_OPPORTUNITIES = 50
MINIMUM_EVALUABLE_SEQUENCES = 3
MAXIMUM_TRANSFORM_PARITY_ERROR_M = 1e-9
ARM_CURRENT = "CURRENT_RELATIVE_STATIC"
ARM_EGO = "EGO_KINEMATIC_OBJECT_STATIC"
ARM_FULL = "EGO_OBJECT_KINEMATIC"
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts.local/evidence/hftf/"
    "stage-c-d42-jrdb-ego-object-metric-teacher-v0/report.json"
)


def finite_vector(value: Any, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"D42 invalid {field}")
    return result


def quaternion(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError("D42 invalid pose quaternion")
    norm = float(np.linalg.norm(result))
    if norm <= 0:
        raise ValueError("D42 zero pose quaternion")
    return result / norm


def yaw_from_quaternion(value: Any) -> float:
    x, y, z, w = quaternion(value)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def load_packet(path: Path) -> tuple[str, dict[int, dict[str, Any]], float]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    sequence = str(packet["sequence"])
    frames: dict[int, dict[str, Any]] = {}
    maximum_parity_error = 0.0
    for raw_frame in packet["frames"]:
        frame_index = int(raw_frame["frame_index"])
        pose = raw_frame["pose"]
        translation = finite_vector(pose["translation"], "pose translation")
        pose_quaternion = quaternion(pose["quaternion_xyzw"])
        rotation = Rotation.from_quat(pose_quaternion).as_matrix()
        truth = []
        truth_by_id = {}
        for item in raw_frame["labels"]["joined"]:
            box = item["box_2d_xywh"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("D42 invalid native 2D box")
            x, y, width, height = (float(value) for value in box)
            if (
                not all(
                    math.isfinite(value)
                    for value in (x, y, width, height)
                )
                or width <= 0
                or height <= 0
            ):
                raise ValueError("D42 invalid native 2D box extent")
            center_base = finite_vector(
                item["center_base_link_m"],
                "center_base_link_m",
            )
            center_odom = finite_vector(
                item["center_odom_m"],
                "center_odom_m",
            )
            parity_error = float(
                np.max(
                    np.abs(
                        rotation @ center_base
                        + translation
                        - center_odom
                    )
                )
            )
            maximum_parity_error = max(
                maximum_parity_error,
                parity_error,
            )
            row = {
                "label_id": str(item["label_id"]),
                "bbox_xyxy": [x, y, x + width, y + height],
                "center_base_link_m": center_base,
                "center_odom_m": center_odom,
            }
            truth.append(row)
            truth_by_id[row["label_id"]] = row
        frames[frame_index] = {
            "timestamp_ns": int(raw_frame["time"]["image_timestamp_ns"]),
            "pose_translation": translation,
            "pose_quaternion": pose_quaternion,
            "truth": truth,
            "truth_by_id": truth_by_id,
        }
    if len(frames) != 120:
        raise ValueError(f"D42 packet frame count drift: {sequence}")
    return sequence, frames, maximum_parity_error


def unwrap_yaws(values: list[float]) -> list[float]:
    return np.unwrap(np.asarray(values, dtype=np.float64)).tolist()


def relative_from_world(
    object_odom: np.ndarray,
    ego_translation: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    delta = object_odom - ego_translation
    cosine = math.cos(ego_yaw)
    sine = math.sin(ego_yaw)
    return np.asarray(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            delta[2],
        ],
        dtype=np.float64,
    )


def predict_arms(
    history: list[dict[str, Any]],
    target_timestamp_ns: int,
) -> dict[str, np.ndarray]:
    if len(history) != HISTORY_COUNT:
        raise ValueError("D42 history must contain seven frames")
    timestamps_s = [
        int(row["timestamp_ns"]) / 1_000_000_000.0 for row in history
    ]
    target_s = target_timestamp_ns / 1_000_000_000.0
    if target_s <= timestamps_s[-1]:
        raise ValueError("D42 target is not future")
    ego_translations = [
        finite_vector(row["pose_translation"], "ego translation")
        for row in history
    ]
    object_centers = [
        finite_vector(row["center_odom_m"], "object odom center")
        for row in history
    ]
    yaws = unwrap_yaws(
        [yaw_from_quaternion(row["pose_quaternion"]) for row in history]
    )
    predicted_ego_translation = np.asarray(
        [
            ols_predict(
                timestamps_s,
                [value[axis] for value in ego_translations],
                target_s,
            )
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    predicted_ego_yaw = ols_predict(timestamps_s, yaws, target_s)
    predicted_object_center = np.asarray(
        [
            ols_predict(
                timestamps_s,
                [value[axis] for value in object_centers],
                target_s,
            )
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    current = history[-1]
    outputs = {
        ARM_CURRENT: finite_vector(
            current["center_base_link_m"],
            "current base center",
        ),
        ARM_EGO: relative_from_world(
            finite_vector(
                current["center_odom_m"],
                "current odom center",
            ),
            predicted_ego_translation,
            predicted_ego_yaw,
        ),
        ARM_FULL: relative_from_world(
            predicted_object_center,
            predicted_ego_translation,
            predicted_ego_yaw,
        ),
    }
    if not all(np.all(np.isfinite(value)) for value in outputs.values()):
        raise ValueError("D42 prediction is non-finite")
    return outputs


def bearing_error_degrees(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    predicted = math.atan2(float(prediction[1]), float(prediction[0]))
    actual = math.atan2(float(truth[1]), float(truth[0]))
    delta = math.atan2(
        math.sin(predicted - actual),
        math.cos(predicted - actual),
    )
    return abs(math.degrees(delta))


def arm_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    horizontal_error = float(np.linalg.norm(prediction[:2] - truth[:2]))
    range_error = abs(
        float(np.linalg.norm(prediction))
        - float(np.linalg.norm(truth))
    )
    bearing_error = bearing_error_degrees(prediction, truth)
    values = (horizontal_error, range_error, bearing_error)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("D42 metric is non-finite")
    return {
        "horizontal_error_m": horizontal_error,
        "absolute_range_error_m": range_error,
        "absolute_bearing_error_deg": bearing_error,
    }


def evaluate_sequence(
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
    opportunities = []
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
            history = []
            for frame_index in history_frames:
                frame = frames[frame_index]
                identity = frame["truth_by_id"][label_id]
                history.append(
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
            predictions = predict_arms(
                history,
                int(future_frame["timestamp_ns"]),
            )
            truth = finite_vector(
                future_truth["center_base_link_m"],
                "future base center",
            )
            opportunities.append(
                {
                    "sequence": sequence,
                    "native_label_id": label_id,
                    "metrics": {
                        arm: arm_metrics(prediction, truth)
                        for arm, prediction in predictions.items()
                    },
                }
            )
    return opportunities


def relative_reduction(baseline: float, candidate: float) -> float | None:
    return (
        (baseline - candidate) / baseline
        if baseline > 0
        else None
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "opportunities": 0,
            "distinct_native_identities": 0,
        }
    arms = {}
    for arm in (ARM_CURRENT, ARM_EGO, ARM_FULL):
        horizontal = [
            float(row["metrics"][arm]["horizontal_error_m"])
            for row in rows
        ]
        ranges = [
            float(row["metrics"][arm]["absolute_range_error_m"])
            for row in rows
        ]
        bearings = [
            float(row["metrics"][arm]["absolute_bearing_error_deg"])
            for row in rows
        ]
        arms[arm] = {
            "mean_horizontal_error_m": statistics.fmean(horizontal),
            "median_horizontal_error_m": statistics.median(horizontal),
            "mean_absolute_range_error_m": statistics.fmean(ranges),
            "mean_absolute_bearing_error_deg": statistics.fmean(bearings),
        }
    baseline = arms[ARM_CURRENT]
    ego = arms[ARM_EGO]
    full = arms[ARM_FULL]
    full_better = sum(
        float(row["metrics"][ARM_FULL]["horizontal_error_m"])
        < float(row["metrics"][ARM_CURRENT]["horizontal_error_m"])
        for row in rows
    ) / len(rows)
    return {
        "opportunities": len(rows),
        "distinct_native_identities": len(
            {
                (str(row["sequence"]), str(row["native_label_id"]))
                for row in rows
            }
        ),
        "arms": arms,
        "full_vs_current": {
            "mean_horizontal_error_relative_reduction": relative_reduction(
                float(baseline["mean_horizontal_error_m"]),
                float(full["mean_horizontal_error_m"]),
            ),
            "median_horizontal_error_relative_reduction": relative_reduction(
                float(baseline["median_horizontal_error_m"]),
                float(full["median_horizontal_error_m"]),
            ),
            "horizontal_error_better_fraction": full_better,
            "mean_range_error_relative_reduction": relative_reduction(
                float(baseline["mean_absolute_range_error_m"]),
                float(full["mean_absolute_range_error_m"]),
            ),
            "mean_bearing_error_relative_reduction": relative_reduction(
                float(baseline["mean_absolute_bearing_error_deg"]),
                float(full["mean_absolute_bearing_error_deg"]),
            ),
        },
        "ego_vs_current": {
            "mean_horizontal_error_relative_reduction": relative_reduction(
                float(baseline["mean_horizontal_error_m"]),
                float(ego["mean_horizontal_error_m"]),
            ),
        },
        "full_vs_ego": {
            "mean_horizontal_error_relative_reduction": relative_reduction(
                float(ego["mean_horizontal_error_m"]),
                float(full["mean_horizontal_error_m"]),
            ),
        },
    }


def determine_terminal(
    pooled: dict[str, Any],
    by_sequence: list[dict[str, Any]],
    source_frames: int,
    maximum_transform_parity_error_m: float,
) -> tuple[dict[str, bool], dict[str, bool], str]:
    evaluable_sequences = [
        row
        for row in by_sequence
        if int(row["opportunities"]) >= MINIMUM_SEQUENCE_OPPORTUNITIES
    ]
    evaluability = {
        "exact_source_frames": source_frames == 480,
        "opportunity_count": (
            int(pooled["opportunities"]) >= MINIMUM_OPPORTUNITIES
        ),
        "distinct_identity_count": (
            int(pooled["distinct_native_identities"]) >= MINIMUM_IDENTITIES
        ),
        "sequence_opportunity_count": (
            len(evaluable_sequences) >= MINIMUM_EVALUABLE_SEQUENCES
        ),
        "transform_parity": (
            math.isfinite(maximum_transform_parity_error_m)
            and maximum_transform_parity_error_m
            <= MAXIMUM_TRANSFORM_PARITY_ERROR_M
        ),
        "finite_metrics": all(
            value is None
            or not isinstance(value, float)
            or math.isfinite(value)
            for row in [pooled, *by_sequence]
            for value in flatten_values(row)
        ),
    }
    effect = pooled["full_vs_current"]
    sequence_reductions = [
        float(
            row["full_vs_current"][
                "mean_horizontal_error_relative_reduction"
            ]
        )
        for row in evaluable_sequences
    ]
    support = {
        "pooled_mean_horizontal_reduction": (
            float(effect["mean_horizontal_error_relative_reduction"])
            >= 0.20
        ),
        "pooled_median_horizontal_reduction": (
            float(effect["median_horizontal_error_relative_reduction"])
            >= 0.20
        ),
        "horizontal_better_fraction": (
            float(effect["horizontal_error_better_fraction"]) >= 0.60
        ),
        "pooled_range_reduction": (
            float(effect["mean_range_error_relative_reduction"]) >= 0.15
        ),
        "pooled_bearing_reduction": (
            float(effect["mean_bearing_error_relative_reduction"]) >= 0.10
        ),
        "sequence_breadth": (
            sum(value > 0 for value in sequence_reductions) >= 3
        ),
        "no_sequence_material_harm": all(
            value >= -0.05 for value in sequence_reductions
        ),
    }
    if not all(evaluability.values()):
        status = NOT_EVALUABLE_STATUS
    elif all(support.values()):
        status = SUPPORTED_STATUS
    else:
        status = NOT_SUPPORTED_STATUS
    return evaluability, support, status


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
        raise ValueError("D42 detector-track binding drift")
    if receipt_hash != EXPECTED_PRODUCER_RECEIPT_SHA256:
        raise ValueError("D42 producer-receipt binding drift")
    receipt = json.loads(
        args.producer_receipt.read_text(encoding="utf-8")
    )
    if (
        str(receipt["status"]) != "COMPLETE"
        or not bool(receipt["source_only"])
        or str(receipt["tracks_sha256"]) != tracks_hash
    ):
        raise ValueError("D42 producer receipt is not admissible")
    source_rows = load_jsonl(args.tracks)
    source_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_sequence[str(row["sequence"])].append(row)
    all_rows = []
    by_sequence = []
    packet_bindings = {}
    maximum_parity_error = 0.0
    for packet_path in args.packets:
        sequence, frames, parity_error = load_packet(packet_path)
        maximum_parity_error = max(maximum_parity_error, parity_error)
        packet_bindings[sequence] = sha256(packet_path)
        rows = evaluate_sequence(
            sequence,
            frames,
            source_by_sequence.get(sequence, []),
        )
        all_rows.extend(rows)
        summary = summarize(rows)
        summary["sequence"] = sequence
        by_sequence.append(summary)
    pooled = summarize(all_rows)
    evaluability, support, status = determine_terminal(
        pooled,
        by_sequence,
        int(receipt["frame_count"]),
        maximum_parity_error,
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "evaluable": all(evaluability.values()),
        "supported": status == SUPPORTED_STATUS,
        "source": {
            "frames": int(receipt["frame_count"]),
            "track_occurrences": len(source_rows),
            "sequences": len(by_sequence),
            "maximum_transform_parity_error_m": maximum_parity_error,
        },
        "pooled": pooled,
        "by_sequence": by_sequence,
        "evaluability_gates": evaluability,
        "support_gates": support,
        "bindings": {
            "tracks_sha256": tracks_hash,
            "producer_receipt_sha256": receipt_hash,
            "packet_sha256": packet_bindings,
        },
        "claims": {
            "source_native_metric_teacher": True,
            "ego_object_motion_decomposition": True,
            "rgb_or_imu_student_learnability": False,
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
