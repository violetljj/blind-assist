#!/usr/bin/env python3
"""Evaluate D33 detector-track source decisions against short-future range."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    DEFAULT_PACKETS,
    FUTURE_FRAME_OFFSET,
    HISTORY_COUNT,
    REPO_ROOT,
    TRUTH_RATE_DEADBAND_MPS,
    fraction,
    range_m,
    sha256,
    source_decision,
)
from produce_stage_c_d33_jrdb_detector_tracks import (
    DEFAULT_RECEIPT as DEFAULT_PRODUCER_RECEIPT,
    DEFAULT_TRACKS,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d33_jrdb_"
    "detector_track_future_range_v0"
)
MINIMUM_MATCH_IOU = 0.30

MINIMUM_SOURCE_FRAMES = 480
MINIMUM_CURRENT_MATCHES = 400
MINIMUM_OPPORTUNITIES = 400
MINIMUM_TOTAL_EVIDENCE = 60
MINIMUM_DISTINCT_IDENTITIES = 15
MINIMUM_SEQUENCES_WITH_EVIDENCE = 3
MINIMUM_SEQUENCE_EVIDENCE = 10
MINIMUM_DIRECTION_EVIDENCE = 15

MINIMUM_OVERALL_PRECISION = 0.85
MINIMUM_DIRECTION_PRECISION = 0.80
MINIMUM_DIRECTION_LIFT = 0.10
MINIMUM_SEQUENCE_PRECISION = 0.75
MINIMUM_SUPPORTING_SEQUENCES = 3

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/evidence/hftf/"
    "stage-c-d33-jrdb-detector-track-future-range-v0/report.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"D33 invalid JSONL line {line_number}"
                ) from exc
    return rows


def iou_matrix(
    source_boxes: np.ndarray,
    truth_boxes: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_boxes, dtype=np.float64).reshape(-1, 4)
    truth = np.asarray(truth_boxes, dtype=np.float64).reshape(-1, 4)
    if not len(source) or not len(truth):
        return np.zeros((len(source), len(truth)), dtype=np.float64)
    left = np.maximum(source[:, None, 0], truth[None, :, 0])
    top = np.maximum(source[:, None, 1], truth[None, :, 1])
    right = np.minimum(source[:, None, 2], truth[None, :, 2])
    bottom = np.minimum(source[:, None, 3], truth[None, :, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(
        0.0,
        bottom - top,
    )
    source_area = np.maximum(0.0, source[:, 2] - source[:, 0]) * (
        np.maximum(0.0, source[:, 3] - source[:, 1])
    )
    truth_area = np.maximum(0.0, truth[:, 2] - truth[:, 0]) * (
        np.maximum(0.0, truth[:, 3] - truth[:, 1])
    )
    union = source_area[:, None] + truth_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def associate(
    source_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
) -> list[tuple[int, int, float]]:
    if not source_rows or not truth_rows:
        return []
    matrix = iou_matrix(
        np.asarray([row["bbox_xyxy"] for row in source_rows]),
        np.asarray([row["bbox_xyxy"] for row in truth_rows]),
    )
    source_indices, truth_indices = linear_sum_assignment(
        1.0 - matrix
    )
    matches = []
    for source_index, truth_index in zip(
        source_indices,
        truth_indices,
        strict=True,
    ):
        value = float(matrix[source_index, truth_index])
        if value >= MINIMUM_MATCH_IOU:
            matches.append(
                (int(source_index), int(truth_index), value)
            )
    return matches


def packet_truth(
    path: Path,
) -> tuple[str, dict[int, dict[str, Any]]]:
    packet = load_json(path)
    sequence = str(packet["sequence"])
    frames: dict[int, dict[str, Any]] = {}
    for frame in packet["frames"]:
        frame_index = int(frame["frame_index"])
        timestamp_ns = int(frame["time"]["image_timestamp_ns"])
        truth = []
        for item in frame["labels"]["joined"]:
            box = item["box_2d_xywh"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("D33 invalid native 2D bbox")
            x, y, width, height = (float(value) for value in box)
            if (
                not all(
                    math.isfinite(value)
                    for value in (x, y, width, height)
                )
                or width <= 0
                or height <= 0
            ):
                raise ValueError("D33 non-finite native 2D bbox")
            truth.append(
                {
                    "label_id": str(item["label_id"]),
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "range_m": range_m(item["center_base_link_m"]),
                }
            )
        frames[frame_index] = {
            "timestamp_ns": timestamp_ns,
            "truth": truth,
            "truth_by_id": {
                row["label_id"]: row for row in truth
            },
        }
    if len(frames) != 120:
        raise ValueError(f"D33 packet frame count drift: {sequence}")
    return sequence, frames


def evaluate_sequence(
    sequence: str,
    frames: dict[int, dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_frame[int(row["frame_index"])].append(row)
    matched_identity: dict[tuple[int, int], dict[str, Any]] = {}
    match_ious = []
    current_matches = 0
    native_truth_rows = 0
    for frame_index, frame in sorted(frames.items()):
        source = source_by_frame.get(frame_index, [])
        truth = frame["truth"]
        native_truth_rows += len(truth)
        for source_index, truth_index, match_iou in associate(
            source,
            truth,
        ):
            source_row = source[source_index]
            truth_row = truth[truth_index]
            key = (frame_index, int(source_row["track_id"]))
            if key in matched_identity:
                raise ValueError("D33 duplicate matched track occurrence")
            matched_identity[key] = {
                "label_id": truth_row["label_id"],
                "iou": match_iou,
            }
            match_ious.append(match_iou)
            current_matches += 1
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        frame_index = int(row["frame_index"])
        frame = frames.get(frame_index)
        if frame is None:
            raise ValueError(f"D33 source frame absent: {sequence}")
        box = row["bbox_xyxy"]
        height = float(box[3]) - float(box[1])
        if not math.isfinite(height) or height <= 0:
            raise ValueError("D33 source bbox height is invalid")
        tracks[int(row["track_id"])].append(
            {
                **row,
                "timestamp_ns": frame["timestamp_ns"],
                "height": height,
            }
        )
    opportunities: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for track_id, track_rows in sorted(tracks.items()):
        ordered = sorted(
            track_rows,
            key=lambda row: int(row["frame_index"]),
        )
        by_frame = {
            int(row["frame_index"]): row for row in ordered
        }
        if len(by_frame) != len(ordered):
            raise ValueError(f"D33 duplicate source track frame: {sequence}")
        for current in ordered:
            current_frame = int(current["frame_index"])
            history_frames = list(
                range(
                    current_frame - HISTORY_COUNT + 1,
                    current_frame + 1,
                )
            )
            if any(frame not in by_frame for frame in history_frames):
                continue
            current_match = matched_identity.get(
                (current_frame, track_id)
            )
            if current_match is None:
                continue
            future_frame = current_frame + FUTURE_FRAME_OFFSET
            future = frames.get(future_frame)
            if future is None:
                continue
            label_id = str(current_match["label_id"])
            future_truth = future["truth_by_id"].get(label_id)
            if future_truth is None:
                continue
            current_truth = frames[current_frame]["truth_by_id"].get(
                label_id
            )
            if current_truth is None:
                raise ValueError("D33 current matched truth disappeared")
            history = [by_frame[frame] for frame in history_frames]
            decision, slope = source_decision(history)
            elapsed_s = (
                int(future["timestamp_ns"])
                - int(frames[current_frame]["timestamp_ns"])
            ) / 1_000_000_000.0
            if elapsed_s <= 0:
                raise ValueError("D33 future timestamp is not later")
            future_rate = (
                float(current_truth["range_m"])
                - float(future_truth["range_m"])
            ) / elapsed_s
            truth_state = (
                "APPROACHING"
                if future_rate >= TRUTH_RATE_DEADBAND_MPS
                else "RECEDING"
                if future_rate <= -TRUTH_RATE_DEADBAND_MPS
                else "QUASI_STATIC"
            )
            history_matches = [
                matched_identity.get((frame, track_id))
                for frame in history_frames
            ]
            history_current_identity_count = sum(
                match is not None and match["label_id"] == label_id
                for match in history_matches
            )
            opportunity = {
                "sequence": sequence,
                "frame_index": current_frame,
                "track_id": track_id,
                "native_label_id": label_id,
                "current_match_iou": float(current_match["iou"]),
                "source_decision": decision,
                "source_log_height_slope_per_s": slope,
                "future_frame_index": future_frame,
                "future_elapsed_s": elapsed_s,
                "truth_future_approach_rate_mps": future_rate,
                "truth_state": truth_state,
                "history_current_identity_fraction": (
                    history_current_identity_count / HISTORY_COUNT
                ),
                "history_all_current_identity": (
                    history_current_identity_count == HISTORY_COUNT
                ),
            }
            opportunities.append(opportunity)
            if decision == "ABSTAIN":
                continue
            correct = (
                truth_state == "APPROACHING"
                if decision == "CONFIRM_APPROACH"
                else truth_state != "APPROACHING"
            )
            evidence.append({**opportunity, "correct": correct})
    return {
        "sequence": sequence,
        "source_occurrences": len(source_rows),
        "source_tracks": len(tracks),
        "native_truth_rows": native_truth_rows,
        "current_matches": current_matches,
        "match_ious": match_ious,
        "opportunities": opportunities,
        "evidence": evidence,
    }


def summarize_direction(
    rows: list[dict[str, Any]],
    decision: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["source_decision"] == decision]
    correct = sum(bool(row["correct"]) for row in selected)
    return {
        "rows": len(selected),
        "correct_rows": correct,
        "precision": fraction(correct, len(selected)),
        "distinct_native_identities": len(
            {
                (row["sequence"], row["native_label_id"])
                for row in selected
            }
        ),
    }


def determine_terminal(
    *,
    source_frames: int,
    current_matches: int,
    opportunities: int,
    evidence_rows: int,
    distinct_identities: int,
    sequences_with_evidence: int,
    confirm_rows: int,
    contradict_rows: int,
    effect_gates: Iterable[bool],
) -> tuple[str, bool, bool]:
    evaluable = bool(
        source_frames >= MINIMUM_SOURCE_FRAMES
        and current_matches >= MINIMUM_CURRENT_MATCHES
        and opportunities >= MINIMUM_OPPORTUNITIES
        and evidence_rows >= MINIMUM_TOTAL_EVIDENCE
        and distinct_identities >= MINIMUM_DISTINCT_IDENTITIES
        and sequences_with_evidence >= MINIMUM_SEQUENCES_WITH_EVIDENCE
        and confirm_rows >= MINIMUM_DIRECTION_EVIDENCE
        and contradict_rows >= MINIMUM_DIRECTION_EVIDENCE
    )
    supported = evaluable and all(effect_gates)
    terminal = (
        "D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_SUPPORTED"
        if supported
        else "D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_NOT_SUPPORTED"
        if evaluable
        else "D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_NOT_EVALUABLE"
    )
    return terminal, evaluable, supported


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values), value))


def build_report(
    packet_paths: tuple[Path, ...],
    tracks_path: Path,
    producer_receipt_path: Path,
) -> dict[str, Any]:
    producer = load_json(producer_receipt_path)
    if (
        producer.get("status") != "COMPLETE"
        or producer.get("tracks_sha256") != sha256(tracks_path)
    ):
        raise ValueError("D33 producer binding drift")
    source_rows = load_jsonl(tracks_path)
    source_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in source_rows:
        source_by_sequence[str(row["sequence"])].append(row)
    sequence_results = []
    input_records = []
    for packet_path in packet_paths:
        sequence, frames = packet_truth(packet_path)
        sequence_results.append(
            evaluate_sequence(
                sequence,
                frames,
                source_by_sequence.get(sequence, []),
            )
        )
        input_records.append(
            {
                "sequence": sequence,
                "path": str(packet_path.resolve()),
                "sha256": sha256(packet_path),
            }
        )
    opportunities = [
        row
        for result in sequence_results
        for row in result["opportunities"]
    ]
    evidence = [
        row
        for result in sequence_results
        for row in result["evidence"]
    ]
    match_ious = [
        value
        for result in sequence_results
        for value in result["match_ious"]
    ]
    confirm = summarize_direction(evidence, "CONFIRM_APPROACH")
    contradict = summarize_direction(
        evidence,
        "CONTRADICT_APPROACH",
    )
    approach_rows = sum(
        row["truth_state"] == "APPROACHING" for row in opportunities
    )
    approach_prevalence = fraction(approach_rows, len(opportunities))
    not_approach_prevalence = (
        None if approach_prevalence is None else 1.0 - approach_prevalence
    )
    confirm_lift = (
        None
        if confirm["precision"] is None or approach_prevalence is None
        else confirm["precision"] - approach_prevalence
    )
    contradict_lift = (
        None
        if contradict["precision"] is None
        or not_approach_prevalence is None
        else contradict["precision"] - not_approach_prevalence
    )
    correct_rows = sum(bool(row["correct"]) for row in evidence)
    overall_precision = fraction(correct_rows, len(evidence))
    distinct_identities = len(
        {
            (row["sequence"], row["native_label_id"])
            for row in evidence
        }
    )
    per_sequence: dict[str, Any] = {}
    sequences_with_evidence = 0
    supporting_sequences = 0
    for result in sequence_results:
        rows = result["evidence"]
        correct = sum(bool(row["correct"]) for row in rows)
        precision = fraction(correct, len(rows))
        enough = len(rows) >= MINIMUM_SEQUENCE_EVIDENCE
        supports = bool(
            enough
            and precision is not None
            and precision >= MINIMUM_SEQUENCE_PRECISION
        )
        sequences_with_evidence += int(enough)
        supporting_sequences += int(supports)
        per_sequence[result["sequence"]] = {
            "source_occurrences": result["source_occurrences"],
            "source_tracks": result["source_tracks"],
            "native_truth_rows": result["native_truth_rows"],
            "current_matches": result["current_matches"],
            "current_match_fraction": fraction(
                result["current_matches"],
                result["native_truth_rows"],
            ),
            "match_iou_median": percentile(result["match_ious"], 0.5),
            "opportunity_rows": len(result["opportunities"]),
            "evidence_rows": len(rows),
            "coverage": fraction(
                len(rows),
                len(result["opportunities"]),
            ),
            "correct_rows": correct,
            "precision": precision,
            "confirm": summarize_direction(
                rows,
                "CONFIRM_APPROACH",
            ),
            "contradict": summarize_direction(
                rows,
                "CONTRADICT_APPROACH",
            ),
            "history_all_current_identity_fraction": fraction(
                sum(
                    bool(row["history_all_current_identity"])
                    for row in rows
                ),
                len(rows),
            ),
            "minimum_evidence_met": enough,
            "sequence_support_gate_passed": supports,
        }
    effect_gate_values = {
        "overall_precision": bool(
            overall_precision is not None
            and overall_precision >= MINIMUM_OVERALL_PRECISION
        ),
        "confirm_precision": bool(
            confirm["precision"] is not None
            and confirm["precision"] >= MINIMUM_DIRECTION_PRECISION
        ),
        "contradict_precision": bool(
            contradict["precision"] is not None
            and contradict["precision"] >= MINIMUM_DIRECTION_PRECISION
        ),
        "confirm_lift": bool(
            confirm_lift is not None
            and confirm_lift >= MINIMUM_DIRECTION_LIFT
        ),
        "contradict_lift": bool(
            contradict_lift is not None
            and contradict_lift >= MINIMUM_DIRECTION_LIFT
        ),
        "supporting_sequences": (
            supporting_sequences >= MINIMUM_SUPPORTING_SEQUENCES
        ),
    }
    source_frames = int(producer["frame_count"])
    current_matches = sum(
        result["current_matches"] for result in sequence_results
    )
    terminal, evaluable, supported = determine_terminal(
        source_frames=source_frames,
        current_matches=current_matches,
        opportunities=len(opportunities),
        evidence_rows=len(evidence),
        distinct_identities=distinct_identities,
        sequences_with_evidence=sequences_with_evidence,
        confirm_rows=confirm["rows"],
        contradict_rows=contradict["rows"],
        effect_gates=effect_gate_values.values(),
    )
    return {
        "schema": SCHEMA,
        "status": terminal,
        "evaluable": evaluable,
        "supported": supported,
        "input_records": input_records,
        "producer": {
            "receipt_path": str(producer_receipt_path.resolve()),
            "receipt_sha256": sha256(producer_receipt_path),
            "tracks_path": str(tracks_path.resolve()),
            "tracks_sha256": sha256(tracks_path),
            "source_frames": source_frames,
        },
        "evaluable_gates": {
            "minimum_source_frames": MINIMUM_SOURCE_FRAMES,
            "minimum_current_matches": MINIMUM_CURRENT_MATCHES,
            "minimum_opportunities": MINIMUM_OPPORTUNITIES,
            "minimum_total_evidence": MINIMUM_TOTAL_EVIDENCE,
            "minimum_distinct_identities": (
                MINIMUM_DISTINCT_IDENTITIES
            ),
            "minimum_sequences_with_evidence": (
                MINIMUM_SEQUENCES_WITH_EVIDENCE
            ),
            "minimum_sequence_evidence": MINIMUM_SEQUENCE_EVIDENCE,
            "minimum_direction_evidence": MINIMUM_DIRECTION_EVIDENCE,
            "sequences_with_evidence": sequences_with_evidence,
            "passed": evaluable,
        },
        "effect_gates": {
            "thresholds": {
                "minimum_overall_precision": MINIMUM_OVERALL_PRECISION,
                "minimum_direction_precision": (
                    MINIMUM_DIRECTION_PRECISION
                ),
                "minimum_direction_lift": MINIMUM_DIRECTION_LIFT,
                "minimum_sequence_precision": MINIMUM_SEQUENCE_PRECISION,
                "minimum_supporting_sequences": (
                    MINIMUM_SUPPORTING_SEQUENCES
                ),
            },
            "values": effect_gate_values,
            "passed": all(effect_gate_values.values()),
        },
        "metrics": {
            "source_frames": source_frames,
            "source_occurrences": len(source_rows),
            "native_truth_rows": sum(
                result["native_truth_rows"]
                for result in sequence_results
            ),
            "current_matches": current_matches,
            "current_match_iou_median": percentile(match_ious, 0.5),
            "current_match_iou_p10": percentile(match_ious, 0.1),
            "opportunity_rows": len(opportunities),
            "evidence_rows": len(evidence),
            "coverage": fraction(len(evidence), len(opportunities)),
            "correct_rows": correct_rows,
            "overall_precision": overall_precision,
            "distinct_native_identities": distinct_identities,
            "approaching_opportunity_rows": approach_rows,
            "approaching_prevalence": approach_prevalence,
            "not_approaching_prevalence": not_approach_prevalence,
            "confirm": {
                **confirm,
                "lift_over_prevalence": confirm_lift,
            },
            "contradict": {
                **contradict,
                "lift_over_prevalence": contradict_lift,
            },
            "history_all_current_identity_fraction": fraction(
                sum(
                    bool(row["history_all_current_identity"])
                    for row in evidence
                ),
                len(evidence),
            ),
            "supporting_sequences": supporting_sequences,
        },
        "per_sequence": per_sequence,
        "evidence_rows": evidence,
        "claim": (
            "JRDB_DETECTOR_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED"
            if supported
            else "NO_POSITIVE_CLAIM"
        ),
        "claim_ceiling": (
            "OFFLINE_JRDB_DETECTOR_TRACK_SHORT_FUTURE_MECHANISM_ONLY"
        ),
        "limitations": [
            "future 3D identity and range truth remain JRDB annotation-derived",
            "evaluation association uses current-frame annotation IoU",
            "rows within tracks are repeated longitudinal observations",
            "offline tiled YOLO and ByteTrack are not Android runtime evidence",
            "no event utility, product, or human-safety claim",
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        action="append",
        type=Path,
        dest="packets",
    )
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument(
        "--producer-receipt",
        type=Path,
        default=DEFAULT_PRODUCER_RECEIPT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    packets = tuple(args.packets) if args.packets else DEFAULT_PACKETS
    report = build_report(
        packets,
        args.tracks,
        args.producer_receipt,
    )
    digest = write_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "evaluable": report["evaluable"],
                "supported": report["supported"],
                "evidence_rows": report["metrics"]["evidence_rows"],
                "overall_precision": report["metrics"][
                    "overall_precision"
                ],
                "sha256": digest,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
