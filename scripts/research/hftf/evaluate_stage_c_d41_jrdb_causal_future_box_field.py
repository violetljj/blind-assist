#!/usr/bin/env python3
"""Evaluate a causal detector-track future-box field on JRDB."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    packet_truth,
)
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = "blindassist_hftf_stage_c_d41_jrdb_causal_future_box_field_v0"
SUPPORTED_STATUS = (
    "D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_SUPPORTED_DEVELOPMENT_ONLY"
)
NOT_SUPPORTED_STATUS = (
    "D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_NOT_SUPPORTED"
)
NOT_EVALUABLE_STATUS = (
    "D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_NOT_EVALUABLE"
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
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts.local/evidence/hftf/"
    "stage-c-d41-jrdb-causal-future-box-field-v0/report.json"
)


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"D41 non-finite {field}")
    return result


def ols_predict(
    timestamps_s: list[float],
    values: list[float],
    target_s: float,
) -> float:
    if len(timestamps_s) != len(values) or len(values) != HISTORY_COUNT:
        raise ValueError("D41 invalid OLS inputs")
    if any(
        right <= left
        for left, right in zip(timestamps_s, timestamps_s[1:])
    ):
        raise ValueError("D41 history timestamps are not increasing")
    mean_time = statistics.fmean(timestamps_s)
    mean_value = statistics.fmean(values)
    denominator = sum((value - mean_time) ** 2 for value in timestamps_s)
    if denominator <= 0:
        raise ValueError("D41 zero OLS time denominator")
    slope = sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(timestamps_s, values, strict=True)
    ) / denominator
    return mean_value + slope * (target_s - mean_time)


def box_state(box: list[float]) -> tuple[float, float, float, float]:
    left, top, right, bottom = (
        finite(value, "source box") for value in box
    )
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("D41 source box has non-positive extent")
    return (
        (left + right) / 2.0,
        (top + bottom) / 2.0,
        math.log(width),
        math.log(height),
    )


def project_box(
    history: list[dict[str, Any]],
    target_timestamp_ns: int,
) -> list[float]:
    if len(history) != HISTORY_COUNT:
        raise ValueError("D41 history must contain seven rows")
    frame_indices = [int(row["frame_index"]) for row in history]
    if any(
        right != left + 1
        for left, right in zip(frame_indices, frame_indices[1:])
    ):
        raise ValueError("D41 history frames are not contiguous")
    timestamps_s = [
        int(row["timestamp_ns"]) / 1_000_000_000.0 for row in history
    ]
    target_s = target_timestamp_ns / 1_000_000_000.0
    if target_s <= timestamps_s[-1]:
        raise ValueError("D41 target timestamp is not future")
    states = [box_state(row["bbox_xyxy"]) for row in history]
    predicted = [
        ols_predict(
            timestamps_s,
            [state[index] for state in states],
            target_s,
        )
        for index in range(4)
    ]
    center_x, center_y, log_width, log_height = predicted
    width = math.exp(log_width)
    height = math.exp(log_height)
    values = (center_x, center_y, width, height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("D41 forecast is non-finite")
    if width <= 0 or height <= 0:
        raise ValueError("D41 forecast has non-positive extent")
    left = center_x - width / 2.0
    top = center_y - height / 2.0
    right = center_x + width / 2.0
    bottom = center_y + height / 2.0
    return [left, top, right, bottom]


def box_iou(left_box: list[float], right_box: list[float]) -> float:
    left = max(left_box[0], right_box[0])
    top = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    bottom = min(left_box[3], right_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    left_area = max(0.0, left_box[2] - left_box[0]) * max(
        0.0,
        left_box[3] - left_box[1],
    )
    right_area = max(0.0, right_box[2] - right_box[0]) * max(
        0.0,
        right_box[3] - right_box[1],
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_error(
    predicted: list[float],
    truth: list[float],
) -> float:
    predicted_x = (predicted[0] + predicted[2]) / 2.0
    predicted_y = (predicted[1] + predicted[3]) / 2.0
    truth_x = (truth[0] + truth[2]) / 2.0
    truth_y = (truth[1] + truth[3]) / 2.0
    diagonal = math.hypot(truth[2] - truth[0], truth[3] - truth[1])
    if diagonal <= 0:
        raise ValueError("D41 truth box has non-positive diagonal")
    return math.hypot(predicted_x - truth_x, predicted_y - truth_y) / diagonal


def absolute_log_area_error(
    predicted: list[float],
    truth: list[float],
) -> float:
    predicted_area = (predicted[2] - predicted[0]) * (
        predicted[3] - predicted[1]
    )
    truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    if predicted_area <= 0 or truth_area <= 0:
        raise ValueError("D41 box area is non-positive")
    return abs(math.log(predicted_area / truth_area))


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
        frame_index = int(row["frame_index"])
        if frame_index not in frames:
            raise ValueError(f"D41 source frame absent: {sequence}")
        tracks[int(row["track_id"])][frame_index] = {
            **row,
            "timestamp_ns": int(frames[frame_index]["timestamp_ns"]),
        }

    opportunities: list[dict[str, Any]] = []
    for track_id, by_frame in sorted(tracks.items()):
        for current_frame, current in sorted(by_frame.items()):
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
            future_frame = current_frame + FUTURE_FRAME_OFFSET
            future = frames.get(future_frame)
            if future is None:
                continue
            future_truth = future["truth_by_id"].get(label_id)
            if future_truth is None:
                continue
            baseline_box = [
                finite(value, "baseline box")
                for value in current["bbox_xyxy"]
            ]
            candidate_box = project_box(
                [by_frame[frame] for frame in history_frames],
                int(future["timestamp_ns"]),
            )
            truth_box = [
                finite(value, "future truth box")
                for value in future_truth["bbox_xyxy"]
            ]
            baseline_iou = box_iou(baseline_box, truth_box)
            candidate_iou = box_iou(candidate_box, truth_box)
            baseline_center = normalized_center_error(
                baseline_box,
                truth_box,
            )
            candidate_center = normalized_center_error(
                candidate_box,
                truth_box,
            )
            baseline_area = absolute_log_area_error(
                baseline_box,
                truth_box,
            )
            candidate_area = absolute_log_area_error(
                candidate_box,
                truth_box,
            )
            values = (
                baseline_iou,
                candidate_iou,
                baseline_center,
                candidate_center,
                baseline_area,
                candidate_area,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("D41 metric is non-finite")
            opportunities.append(
                {
                    "sequence": sequence,
                    "native_label_id": label_id,
                    "baseline_iou": baseline_iou,
                    "candidate_iou": candidate_iou,
                    "baseline_center_error": baseline_center,
                    "candidate_center_error": candidate_center,
                    "baseline_log_area_error": baseline_area,
                    "candidate_log_area_error": candidate_area,
                }
            )
    return opportunities


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "opportunities": 0,
            "distinct_native_identities": 0,
        }
    baseline_ious = [float(row["baseline_iou"]) for row in rows]
    candidate_ious = [float(row["candidate_iou"]) for row in rows]
    iou_deltas = [
        candidate - baseline
        for baseline, candidate in zip(
            baseline_ious,
            candidate_ious,
            strict=True,
        )
    ]
    baseline_center = statistics.fmean(
        float(row["baseline_center_error"]) for row in rows
    )
    candidate_center = statistics.fmean(
        float(row["candidate_center_error"]) for row in rows
    )
    baseline_area = statistics.fmean(
        float(row["baseline_log_area_error"]) for row in rows
    )
    candidate_area = statistics.fmean(
        float(row["candidate_log_area_error"]) for row in rows
    )
    return {
        "opportunities": len(rows),
        "distinct_native_identities": len(
            {
                (str(row["sequence"]), str(row["native_label_id"]))
                for row in rows
            }
        ),
        "baseline_mean_iou": statistics.fmean(baseline_ious),
        "candidate_mean_iou": statistics.fmean(candidate_ious),
        "mean_iou_delta": statistics.fmean(iou_deltas),
        "median_iou_delta": statistics.median(iou_deltas),
        "candidate_iou_better_fraction": sum(
            delta > 0 for delta in iou_deltas
        )
        / len(iou_deltas),
        "baseline_mean_normalized_center_error": baseline_center,
        "candidate_mean_normalized_center_error": candidate_center,
        "center_error_relative_reduction": (
            (baseline_center - candidate_center) / baseline_center
            if baseline_center > 0
            else None
        ),
        "baseline_mean_absolute_log_area_error": baseline_area,
        "candidate_mean_absolute_log_area_error": candidate_area,
        "mean_absolute_log_area_error_delta": candidate_area - baseline_area,
    }


def determine_terminal(
    pooled: dict[str, Any],
    by_sequence: list[dict[str, Any]],
    source_frames: int,
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
        "finite_metrics": all(
            value is None
            or not isinstance(value, float)
            or math.isfinite(value)
            for row in [pooled, *by_sequence]
            for value in row.values()
        ),
    }
    support = {
        "pooled_mean_iou_gain": float(pooled["mean_iou_delta"]) >= 0.02,
        "pooled_median_iou_gain": (
            float(pooled["median_iou_delta"]) >= 0.02
        ),
        "candidate_better_fraction": (
            float(pooled["candidate_iou_better_fraction"]) >= 0.55
        ),
        "center_error_reduction": (
            float(pooled["center_error_relative_reduction"]) >= 0.10
        ),
        "log_area_noninferiority": (
            float(pooled["mean_absolute_log_area_error_delta"]) <= 0.0
        ),
        "sequence_breadth": (
            sum(float(row["mean_iou_delta"]) > 0 for row in evaluable_sequences)
            >= 3
        ),
        "no_sequence_material_harm": all(
            float(row["mean_iou_delta"]) >= -0.02
            for row in evaluable_sequences
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
    if tracks_hash != EXPECTED_TRACKS_SHA256:
        raise ValueError("D41 detector-track binding drift")
    producer_receipt_hash = sha256(args.producer_receipt)
    if producer_receipt_hash != EXPECTED_PRODUCER_RECEIPT_SHA256:
        raise ValueError("D41 producer-receipt binding drift")
    producer_receipt = json.loads(
        args.producer_receipt.read_text(encoding="utf-8")
    )
    if (
        str(producer_receipt["status"]) != "COMPLETE"
        or not bool(producer_receipt["source_only"])
        or str(producer_receipt["tracks_sha256"]) != tracks_hash
    ):
        raise ValueError("D41 producer receipt is not admissible")
    receipt_frame_count = int(producer_receipt["frame_count"])
    source_rows = load_jsonl(args.tracks)
    source_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_sequence[str(row["sequence"])].append(row)
    packet_bindings = {}
    all_rows: list[dict[str, Any]] = []
    by_sequence = []
    source_frame_keys = set()
    for packet_path in args.packets:
        sequence, frames = packet_truth(packet_path)
        packet_bindings[sequence] = sha256(packet_path)
        sequence_source = source_by_sequence.get(sequence, [])
        source_frame_keys.update(
            (sequence, int(row["frame_index"])) for row in sequence_source
        )
        rows = evaluate_sequence(sequence, frames, sequence_source)
        all_rows.extend(rows)
        summary = summarize(rows)
        summary["sequence"] = sequence
        by_sequence.append(summary)
    pooled = summarize(all_rows)
    evaluability, support, status = determine_terminal(
        pooled,
        by_sequence,
        receipt_frame_count,
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "evaluable": all(evaluability.values()),
        "supported": status == SUPPORTED_STATUS,
        "source": {
            "frames": receipt_frame_count,
            "frames_with_tracked_occurrences": len(source_frame_keys),
            "track_occurrences": len(source_rows),
            "sequences": len(by_sequence),
        },
        "pooled": pooled,
        "by_sequence": by_sequence,
        "evaluability_gates": evaluability,
        "support_gates": support,
        "bindings": {
            "tracks_sha256": tracks_hash,
            "producer_receipt_sha256": producer_receipt_hash,
            "packet_sha256": packet_bindings,
        },
        "claims": {
            "detector_bound_future_spatial_representation": True,
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
