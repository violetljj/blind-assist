#!/usr/bin/env python3
"""Evaluate whether a causal 2D track trend predicts one-second future range."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_hftf_stage_c_d32_jrdb_future_range_v0"
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SLOPE_THRESHOLD_PER_S = 0.2
TRUTH_RATE_DEADBAND_MPS = 0.1
HISTORY_COUNT = 7
FUTURE_FRAME_OFFSET = 15

MINIMUM_TOTAL_EVIDENCE = 80
MINIMUM_DISTINCT_TRACKS = 20
MINIMUM_SEQUENCES_WITH_EVIDENCE = 3
MINIMUM_SEQUENCE_EVIDENCE = 10
MINIMUM_DIRECTION_EVIDENCE = 20

MINIMUM_OVERALL_PRECISION = 0.85
MINIMUM_DIRECTION_PRECISION = 0.80
MINIMUM_DIRECTION_LIFT = 0.10
MINIMUM_SEQUENCE_PRECISION = 0.75
MINIMUM_SUPPORTING_SEQUENCES = 3

DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts.local/evidence/hftf/"
    "stage-c-d32-jrdb-causal-track-future-range-v0/report.json"
)
DEFAULT_PACKETS = (
    REPO_ROOT
    / (
        "artifacts.local/datasets/"
        "jrdb-person-3d-trajectory-sensor-support-and-bias-"
        "cross-sequence-replication-r0/"
        "clark-center-2019-02-28_0/observation-packet.json"
    ),
    REPO_ROOT
    / (
        "artifacts.local/datasets/"
        "jrdb-person-3d-trajectory-sensor-support-and-bias-"
        "cross-sequence-replication-r0/"
        "gates-basement-elevators-2019-01-17_1/"
        "observation-packet.json"
    ),
    REPO_ROOT
    / (
        "artifacts.local/evidence/"
        "jrdb-single-sequence-native-multisensor-person-"
        "geometry-canary-r0/observation-packet.json"
    ),
    REPO_ROOT
    / (
        "artifacts.local/datasets/"
        "jrdb-person-3d-trajectory-sensor-support-and-bias-"
        "cross-sequence-replication-r0/"
        "stlc-111-2019-04-19_0/observation-packet.json"
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"D32 non-finite {field}")
    return result


def range_m(center: Any) -> float:
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError("D32 center_base_link_m must contain three values")
    values = [finite_float(value, "center_base_link_m") for value in center]
    return math.sqrt(sum(value * value for value in values))


def ols_slope(times_s: list[float], values: list[float]) -> float:
    if len(times_s) != len(values) or len(times_s) < 2:
        raise ValueError("D32 invalid OLS inputs")
    mean_time = sum(times_s) / len(times_s)
    mean_value = sum(values) / len(values)
    denominator = sum((value - mean_time) ** 2 for value in times_s)
    if denominator <= 0:
        raise ValueError("D32 non-increasing history timestamps")
    return sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(times_s, values, strict=True)
    ) / denominator


def source_decision(
    history: list[dict[str, Any]],
) -> tuple[str, float]:
    if len(history) != HISTORY_COUNT:
        raise ValueError("D32 source history must contain seven rows")
    frame_indices = [int(row["frame_index"]) for row in history]
    if any(
        right != left + 1
        for left, right in zip(frame_indices, frame_indices[1:])
    ):
        raise ValueError("D32 source history is not frame-contiguous")
    times_s = [
        int(row["timestamp_ns"]) / 1_000_000_000.0 for row in history
    ]
    if any(
        right <= left
        for left, right in zip(times_s, times_s[1:])
    ):
        raise ValueError("D32 source history timestamps are not increasing")
    heights = [finite_float(row["height"], "box height") for row in history]
    if any(value <= 0 for value in heights):
        raise ValueError("D32 box height must be positive")
    log_heights = [math.log(value) for value in heights]
    slope = ols_slope(times_s, log_heights)
    changes = [
        right - left
        for left, right in zip(heights, heights[1:])
    ]
    if slope >= SOURCE_SLOPE_THRESHOLD_PER_S and all(
        value > 0 for value in changes
    ):
        return "CONFIRM_APPROACH", slope
    if slope <= -SOURCE_SLOPE_THRESHOLD_PER_S and all(
        value < 0 for value in changes
    ):
        return "CONTRADICT_APPROACH", slope
    return "ABSTAIN", slope


def load_packet(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        packet = json.load(handle)
    sequence = str(packet["sequence"])
    frames = packet["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"D32 packet has no frames: {sequence}")
    rows: list[dict[str, Any]] = []
    seen_frame_indices: set[int] = set()
    seen_identities: set[tuple[int, str]] = set()
    previous_frame_index: int | None = None
    previous_timestamp_ns: int | None = None
    for frame in frames:
        frame_index = int(frame["frame_index"])
        timestamp_ns = int(frame["time"]["image_timestamp_ns"])
        if frame_index in seen_frame_indices:
            raise ValueError(f"D32 duplicate frame index: {sequence}")
        if (
            previous_frame_index is not None
            and frame_index != previous_frame_index + 1
        ):
            raise ValueError(f"D32 packet frames are not contiguous: {sequence}")
        if (
            previous_timestamp_ns is not None
            and timestamp_ns <= previous_timestamp_ns
        ):
            raise ValueError(
                f"D32 packet timestamps are not increasing: {sequence}"
            )
        seen_frame_indices.add(frame_index)
        previous_frame_index = frame_index
        previous_timestamp_ns = timestamp_ns
        joined = frame["labels"]["joined"]
        if not isinstance(joined, list):
            raise ValueError(f"D32 joined labels are not a list: {sequence}")
        for item in joined:
            label_id = str(item["label_id"])
            identity = (frame_index, label_id)
            if identity in seen_identities:
                raise ValueError(
                    f"D32 duplicate frame/track identity: {sequence}"
                )
            seen_identities.add(identity)
            box = item["box_2d_xywh"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("D32 box_2d_xywh must contain four values")
            rows.append(
                {
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "frame_stem": str(frame["frame_stem"]),
                    "timestamp_ns": timestamp_ns,
                    "track_id": label_id,
                    "height": finite_float(box[3], "box height"),
                    "range_m": range_m(item["center_base_link_m"]),
                    "label_3d_interpolated": bool(
                        item["label_3d_interpolated"]
                    ),
                }
            )
    return sequence, rows


def evaluate_sequence(
    sequence: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tracks[str(row["track_id"])].append(row)
    opportunities: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for track_id, track_rows in sorted(tracks.items()):
        ordered = sorted(track_rows, key=lambda row: int(row["frame_index"]))
        by_frame = {int(row["frame_index"]): row for row in ordered}
        if len(by_frame) != len(ordered):
            raise ValueError(f"D32 duplicate track frame: {sequence}/{track_id}")
        for current in ordered:
            current_frame = int(current["frame_index"])
            history_frames = range(
                current_frame - HISTORY_COUNT + 1,
                current_frame + 1,
            )
            if any(frame not in by_frame for frame in history_frames):
                continue
            future = by_frame.get(current_frame + FUTURE_FRAME_OFFSET)
            if future is None:
                continue
            history = [by_frame[frame] for frame in history_frames]
            decision, slope = source_decision(history)
            elapsed_s = (
                int(future["timestamp_ns"]) - int(current["timestamp_ns"])
            ) / 1_000_000_000.0
            if elapsed_s <= 0:
                raise ValueError("D32 future timestamp is not later")
            future_rate = (
                finite_float(current["range_m"], "current range")
                - finite_float(future["range_m"], "future range")
            ) / elapsed_s
            truth_state = (
                "APPROACHING"
                if future_rate >= TRUTH_RATE_DEADBAND_MPS
                else "RECEDING"
                if future_rate <= -TRUTH_RATE_DEADBAND_MPS
                else "QUASI_STATIC"
            )
            opportunity = {
                "sequence": sequence,
                "frame_index": current_frame,
                "frame_stem": str(current["frame_stem"]),
                "track_id": track_id,
                "source_decision": decision,
                "source_log_height_slope_per_s": slope,
                "future_frame_index": int(future["frame_index"]),
                "future_elapsed_s": elapsed_s,
                "truth_future_approach_rate_mps": future_rate,
                "truth_state": truth_state,
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
        "joined_rows": len(rows),
        "track_count": len(tracks),
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
        "distinct_tracks": len(
            {(row["sequence"], row["track_id"]) for row in selected}
        ),
    }


def determine_terminal(
    *,
    evidence_rows: int,
    distinct_tracks: int,
    sequences_with_evidence: int,
    confirm_rows: int,
    contradict_rows: int,
    effect_gates: Iterable[bool],
) -> tuple[str, bool, bool]:
    evaluable = bool(
        evidence_rows >= MINIMUM_TOTAL_EVIDENCE
        and distinct_tracks >= MINIMUM_DISTINCT_TRACKS
        and sequences_with_evidence >= MINIMUM_SEQUENCES_WITH_EVIDENCE
        and confirm_rows >= MINIMUM_DIRECTION_EVIDENCE
        and contradict_rows >= MINIMUM_DIRECTION_EVIDENCE
    )
    supported = evaluable and all(effect_gates)
    terminal = (
        "D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_SUPPORTED"
        if supported
        else "D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_NOT_SUPPORTED"
        if evaluable
        else "D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_NOT_EVALUABLE"
    )
    return terminal, evaluable, supported


def build_report(packet_paths: tuple[Path, ...]) -> dict[str, Any]:
    per_sequence_raw: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    sequences: set[str] = set()
    for path in packet_paths:
        sequence, rows = load_packet(path)
        if sequence in sequences:
            raise ValueError(f"D32 duplicate packet sequence: {sequence}")
        sequences.add(sequence)
        input_records.append(
            {
                "sequence": sequence,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        per_sequence_raw.append(evaluate_sequence(sequence, rows))
    all_opportunities = [
        row
        for sequence_result in per_sequence_raw
        for row in sequence_result["opportunities"]
    ]
    all_evidence = [
        row
        for sequence_result in per_sequence_raw
        for row in sequence_result["evidence"]
    ]
    confirm = summarize_direction(all_evidence, "CONFIRM_APPROACH")
    contradict = summarize_direction(
        all_evidence,
        "CONTRADICT_APPROACH",
    )
    approach_opportunities = sum(
        row["truth_state"] == "APPROACHING" for row in all_opportunities
    )
    approach_prevalence = fraction(
        approach_opportunities,
        len(all_opportunities),
    )
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
    correct_rows = sum(bool(row["correct"]) for row in all_evidence)
    overall_precision = fraction(correct_rows, len(all_evidence))
    distinct_tracks = len(
        {(row["sequence"], row["track_id"]) for row in all_evidence}
    )
    per_sequence: dict[str, Any] = {}
    supporting_sequences = 0
    sequences_with_evidence = 0
    for sequence_result in per_sequence_raw:
        rows = sequence_result["evidence"]
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
        per_sequence[sequence_result["sequence"]] = {
            "joined_rows": sequence_result["joined_rows"],
            "track_count": sequence_result["track_count"],
            "opportunity_rows": len(sequence_result["opportunities"]),
            "evidence_rows": len(rows),
            "coverage": fraction(
                len(rows),
                len(sequence_result["opportunities"]),
            ),
            "correct_rows": correct,
            "precision": precision,
            "confirm": summarize_direction(rows, "CONFIRM_APPROACH"),
            "contradict": summarize_direction(
                rows,
                "CONTRADICT_APPROACH",
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
    terminal, evaluable, supported = determine_terminal(
        evidence_rows=len(all_evidence),
        distinct_tracks=distinct_tracks,
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
        "estimand": {
            "history_count": HISTORY_COUNT,
            "source_slope_threshold_per_s": (
                SOURCE_SLOPE_THRESHOLD_PER_S
            ),
            "future_frame_offset": FUTURE_FRAME_OFFSET,
            "truth_rate_deadband_mps": TRUTH_RATE_DEADBAND_MPS,
            "source_input": "JRDB stitched 2D annotation boxes and label_id",
            "truth": (
                "same-identity JRDB annotation center_base_link_m "
                "future range change"
            ),
        },
        "input_records": input_records,
        "evaluable_gates": {
            "minimum_total_evidence": MINIMUM_TOTAL_EVIDENCE,
            "minimum_distinct_tracks": MINIMUM_DISTINCT_TRACKS,
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
            "sequence_count": len(per_sequence_raw),
            "opportunity_rows": len(all_opportunities),
            "evidence_rows": len(all_evidence),
            "coverage": fraction(
                len(all_evidence),
                len(all_opportunities),
            ),
            "correct_rows": correct_rows,
            "overall_precision": overall_precision,
            "distinct_tracks": distinct_tracks,
            "approaching_opportunity_rows": approach_opportunities,
            "approaching_prevalence": approach_prevalence,
            "not_approaching_prevalence": not_approach_prevalence,
            "confirm": {**confirm, "lift_over_prevalence": confirm_lift},
            "contradict": {
                **contradict,
                "lift_over_prevalence": contradict_lift,
            },
            "supporting_sequences": supporting_sequences,
        },
        "per_sequence": per_sequence,
        "evidence_rows": all_evidence,
        "claim": (
            "JRDB_ANNOTATION_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED"
            if supported
            else "NO_POSITIVE_CLAIM"
        ),
        "claim_ceiling": (
            "JRDB_ANNOTATION_IDENTITY_BOUND_SHORT_FUTURE_MECHANISM_ONLY"
        ),
        "limitations": [
            "2D boxes and identity are annotations, not live detector-track output",
            "3D centers are annotation-derived and marked interpolated",
            "rows within a track are repeated longitudinal observations",
            "four short sequences are Development evidence, not deployment evidence",
            "no event utility, Android, product, or human-safety claim",
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    partial.write_text(payload, encoding="utf-8")
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
        help="Observation packet; repeat for each sequence",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    packet_paths = tuple(args.packets) if args.packets else DEFAULT_PACKETS
    report = build_report(packet_paths)
    digest = write_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "evaluable": report["evaluable"],
                "supported": report["supported"],
                "evidence_rows": report["metrics"]["evidence_rows"],
                "overall_precision": report["metrics"]["overall_precision"],
                "sha256": digest,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
