#!/usr/bin/env python3
"""Evaluate a causal relative metric-track future geometry primitive."""

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
    sha256,
)
from evaluate_stage_c_d33_jrdb_detector_track_future_range import (
    associate,
    load_jsonl,
)
from evaluate_stage_c_d41_jrdb_causal_future_box_field import ols_predict
from evaluate_stage_c_d42_jrdb_ego_object_metric_teacher import (
    EXPECTED_PRODUCER_RECEIPT_SHA256,
    EXPECTED_TRACKS_SHA256,
    arm_metrics,
    finite_vector,
    load_packet,
)
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d44_jrdb_"
    "causal_relative_metric_track_v0"
)
SUPPORTED_STATUS = (
    "D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_"
    "SUPPORTED_DEVELOPMENT_ONLY"
)
NOT_SUPPORTED_STATUS = (
    "D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_NOT_SUPPORTED"
)
NOT_EVALUABLE_STATUS = (
    "D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_NOT_EVALUABLE"
)
CURRENT = "CURRENT_RELATIVE_STATIC"
CANDIDATE = "CAUSAL_RELATIVE_METRIC_TRACK"
MINIMUM_OPPORTUNITIES = 400
MINIMUM_IDENTITIES = 15
MINIMUM_SEQUENCE_OPPORTUNITIES = 50
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts.local/evidence/hftf/"
    "stage-c-d44-jrdb-causal-relative-metric-track-v0/report.json"
)


def predict_relative_metric_track(
    history: list[dict[str, Any]],
    target_timestamp_ns: int,
) -> np.ndarray:
    if len(history) != HISTORY_COUNT:
        raise ValueError("D44 history must contain seven frames")
    frame_indices = [int(row["frame_index"]) for row in history]
    if any(
        right != left + 1
        for left, right in zip(frame_indices, frame_indices[1:])
    ):
        raise ValueError("D44 history is not frame-contiguous")
    timestamps_s = [
        int(row["timestamp_ns"]) / 1_000_000_000.0 for row in history
    ]
    target_s = target_timestamp_ns / 1_000_000_000.0
    if target_s <= timestamps_s[-1]:
        raise ValueError("D44 target timestamp is not future")
    centers = [
        finite_vector(row["center_base_link_m"], "relative center")
        for row in history
    ]
    prediction = np.asarray(
        [
            ols_predict(
                timestamps_s,
                [center[axis] for center in centers],
                target_s,
            )
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("D44 prediction is non-finite")
    return prediction


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
    tracks: dict[int, set[int]] = defaultdict(set)
    for row in source_rows:
        tracks[int(row["track_id"])].add(int(row["frame_index"]))
    rows = []
    for track_id, frame_indices in sorted(tracks.items()):
        for current_frame in sorted(frame_indices):
            history_frames = list(
                range(
                    current_frame - HISTORY_COUNT + 1,
                    current_frame + 1,
                )
            )
            if any(frame not in frame_indices for frame in history_frames):
                continue
            label_id = matched_identity.get((current_frame, track_id))
            if label_id is None:
                continue
            future_index = current_frame + FUTURE_FRAME_OFFSET
            future = frames.get(future_index)
            if future is None:
                continue
            future_truth = future["truth_by_id"].get(label_id)
            if future_truth is None:
                continue
            if any(
                label_id not in frames[frame]["truth_by_id"]
                for frame in history_frames
            ):
                continue
            history = [
                {
                    "frame_index": frame,
                    "timestamp_ns": frames[frame]["timestamp_ns"],
                    "center_base_link_m": frames[frame][
                        "truth_by_id"
                    ][label_id]["center_base_link_m"],
                }
                for frame in history_frames
            ]
            baseline = finite_vector(
                history[-1]["center_base_link_m"],
                "current relative center",
            )
            candidate = predict_relative_metric_track(
                history,
                int(future["timestamp_ns"]),
            )
            truth = finite_vector(
                future_truth["center_base_link_m"],
                "future relative center",
            )
            rows.append(
                {
                    "sequence": sequence,
                    "native_label_id": label_id,
                    "metrics": {
                        CURRENT: arm_metrics(baseline, truth),
                        CANDIDATE: arm_metrics(candidate, truth),
                    },
                }
            )
    return rows


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
    horizontal_cache = {}
    for arm in (CURRENT, CANDIDATE):
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
        horizontal_cache[arm] = horizontal
        arms[arm] = {
            "mean_horizontal_error_m": statistics.fmean(horizontal),
            "median_horizontal_error_m": statistics.median(horizontal),
            "mean_absolute_range_error_m": statistics.fmean(ranges),
            "mean_absolute_bearing_error_deg": statistics.fmean(bearings),
        }
    baseline = arms[CURRENT]
    candidate = arms[CANDIDATE]
    return {
        "opportunities": len(rows),
        "distinct_native_identities": len(
            {
                (str(row["sequence"]), str(row["native_label_id"]))
                for row in rows
            }
        ),
        "arms": arms,
        "candidate_vs_current": {
            "mean_horizontal_error_relative_reduction": relative_reduction(
                float(baseline["mean_horizontal_error_m"]),
                float(candidate["mean_horizontal_error_m"]),
            ),
            "median_horizontal_error_relative_reduction": (
                relative_reduction(
                    float(baseline["median_horizontal_error_m"]),
                    float(candidate["median_horizontal_error_m"]),
                )
            ),
            "horizontal_error_better_fraction": sum(
                candidate_value < baseline_value
                for baseline_value, candidate_value in zip(
                    horizontal_cache[CURRENT],
                    horizontal_cache[CANDIDATE],
                    strict=True,
                )
            )
            / len(rows),
            "mean_range_error_relative_reduction": relative_reduction(
                float(baseline["mean_absolute_range_error_m"]),
                float(candidate["mean_absolute_range_error_m"]),
            ),
            "mean_bearing_error_relative_reduction": relative_reduction(
                float(baseline["mean_absolute_bearing_error_deg"]),
                float(candidate["mean_absolute_bearing_error_deg"]),
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
    by_sequence: list[dict[str, Any]],
    source_frames: int,
) -> tuple[dict[str, bool], dict[str, bool], str]:
    evaluable_sequences = [
        row
        for row in by_sequence
        if int(row["opportunities"]) >= MINIMUM_SEQUENCE_OPPORTUNITIES
    ]
    evaluability = {
        "source_binding": source_frames == 480,
        "opportunity_count": (
            int(pooled["opportunities"]) >= MINIMUM_OPPORTUNITIES
        ),
        "identity_count": (
            int(pooled["distinct_native_identities"]) >= MINIMUM_IDENTITIES
        ),
        "four_sequence_opportunity": len(evaluable_sequences) == 4,
        "finite_metrics": all(
            value is None
            or not isinstance(value, float)
            or math.isfinite(value)
            for row in [pooled, *by_sequence]
            for value in flatten_values(row)
        ),
    }
    effect = pooled["candidate_vs_current"]
    sequence_reductions = [
        float(
            row["candidate_vs_current"][
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
        raise ValueError("D44 detector-track binding drift")
    if receipt_hash != EXPECTED_PRODUCER_RECEIPT_SHA256:
        raise ValueError("D44 producer-receipt binding drift")
    receipt = json.loads(
        args.producer_receipt.read_text(encoding="utf-8")
    )
    source_rows = load_jsonl(args.tracks)
    source_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_sequence[str(row["sequence"])].append(row)
    all_rows = []
    by_sequence = []
    packet_bindings = {}
    for packet_path in args.packets:
        sequence, frames, _ = load_packet(packet_path)
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
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "evaluable": all(evaluability.values()),
        "supported": status == SUPPORTED_STATUS,
        "source": {
            "frames": int(receipt["frame_count"]),
            "track_occurrences": len(source_rows),
            "sequences": 4,
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
            "causal_relative_metric_track_ceiling": True,
            "runtime_metric_depth_available": False,
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
