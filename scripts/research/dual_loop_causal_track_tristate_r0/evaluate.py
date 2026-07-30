from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    MINIMUM_DISTINCT_TRACKS,
    MINIMUM_POOLED_DIRECTION_PRECISION,
    MINIMUM_SESSION_COVERAGE,
    MINIMUM_SESSION_EVIDENCE,
    MINIMUM_SESSION_PRECISION,
    MINIMUM_TOTAL_EVIDENCE,
    PROTOCOL_ID,
    TRUTH_RATE_DEADBAND_MPS,
    read_json,
    read_jsonl,
    sha256_file,
    source_parameter_sha256,
    write_exclusive,
)


def _range_m(item: dict[str, Any]) -> float | None:
    try:
        box = item["box"]
        values = [float(box[key]) for key in ("cx", "cy", "cz")]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return math.sqrt(sum(value * value for value in values))


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def run(
    freeze_path: Path,
    truth_root: Path,
    truth_acquisition_receipt: Path,
    producer_output: Path,
    producer_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    freeze = read_json(freeze_path)
    producer_receipt = read_json(producer_receipt_path)
    truth_receipt = read_json(truth_acquisition_receipt)
    if (
        producer_receipt.get("freeze_sha256") != sha256_file(freeze_path)
        or producer_receipt.get("output_sha256") != sha256_file(producer_output)
        or producer_receipt.get("parameter_sha256") != source_parameter_sha256()
    ):
        raise ValueError("producer binding drift")
    if (
        truth_receipt.get("role") != "truth_3d"
        or truth_receipt.get("freeze_sha256") != sha256_file(freeze_path)
    ):
        raise ValueError("truth acquisition binding drift")
    source_rows = read_jsonl(producer_output)
    source_index = {
        (
            str(row["sequence"]),
            int(row["local_position"]),
            str(row["track_id"]),
        ): row
        for row in source_rows
    }
    if len(source_index) != len(source_rows):
        raise ValueError("duplicate producer identity")
    truth_bindings = {
        str(row["sequence"]): row for row in truth_receipt["records"]
    }
    evaluated: list[dict[str, Any]] = []
    opportunity: Counter[str] = Counter()
    for selected in freeze["selected"]:
        sequence = str(selected["sequence"])
        truth_path = truth_root / "truth_3d" / f"{sequence}.json"
        binding = truth_bindings.get(sequence)
        if binding is None or sha256_file(truth_path) != binding["sha256"]:
            raise ValueError(f"truth payload binding drift: {sequence}")
        labels = read_json(truth_path)["labels"]
        previous: dict[str, tuple[int, float, float]] = {}
        for local_position, frame in enumerate(selected["frames"]):
            stem = str(frame["frame_stem"])
            timestamp_s = int(frame["timestamp_ns"]) / 1_000_000_000.0
            items = labels.get(f"{stem}.pcd")
            if not isinstance(items, list):
                raise ValueError(f"truth frame absent: {sequence}/{stem}")
            current_ids = set()
            for item in items:
                label_id = str(item["label_id"])
                current_ids.add(label_id)
                range_m = _range_m(item)
                prior = previous.get(label_id)
                source = source_index.get((sequence, local_position, label_id))
                if (
                    range_m is not None
                    and prior is not None
                    and local_position == prior[0] + 1
                    and source is not None
                    and int(source["history_count"]) == 7
                ):
                    delta_s = timestamp_s - prior[1]
                    if delta_s <= 0:
                        raise ValueError("truth timestamp is not increasing")
                    truth_rate = (prior[2] - range_m) / delta_s
                    truth_state = (
                        "APPROACHING"
                        if truth_rate >= TRUTH_RATE_DEADBAND_MPS
                        else "RECEDING"
                        if truth_rate <= -TRUTH_RATE_DEADBAND_MPS
                        else "QUASI_STATIC"
                    )
                    opportunity[sequence] += 1
                    decision = str(source["decision"])
                    if decision != "ABSTAIN":
                        correct = (
                            truth_state == "APPROACHING"
                            if decision == "CONFIRM_APPROACH"
                            else truth_state != "APPROACHING"
                        )
                        evaluated.append(
                            {
                                "sequence": sequence,
                                "local_position": local_position,
                                "frame_stem": stem,
                                "track_id": label_id,
                                "frame_detection_id": source[
                                    "frame_detection_id"
                                ],
                                "immutable_roi_id": source["immutable_roi_id"],
                                "decision": decision,
                                "source_slope_per_s": source[
                                    "signed_log_height_slope_per_s"
                                ],
                                "truth_signed_approach_rate_mps": truth_rate,
                                "truth_state": truth_state,
                                "correct": correct,
                            }
                        )
                if range_m is not None:
                    previous[label_id] = (local_position, timestamp_s, range_m)
            for label_id in set(previous) - current_ids:
                previous.pop(label_id)
    per_session: dict[str, Any] = {}
    session_passes = []
    for selected in freeze["selected"]:
        sequence = str(selected["sequence"])
        rows = [row for row in evaluated if row["sequence"] == sequence]
        correct = sum(bool(row["correct"]) for row in rows)
        coverage = _fraction(len(rows), opportunity[sequence]) or 0.0
        precision = _fraction(correct, len(rows))
        session_pass = bool(
            len(rows) >= MINIMUM_SESSION_EVIDENCE
            and coverage >= MINIMUM_SESSION_COVERAGE
            and precision is not None
            and precision >= MINIMUM_SESSION_PRECISION
        )
        session_passes.append(session_pass)
        per_session[sequence] = {
            "opportunity_rows": opportunity[sequence],
            "evidence_rows": len(rows),
            "coverage": coverage,
            "correct_rows": correct,
            "precision": precision,
            "confirm_rows": sum(
                row["decision"] == "CONFIRM_APPROACH" for row in rows
            ),
            "contradict_rows": sum(
                row["decision"] == "CONTRADICT_APPROACH" for row in rows
            ),
            "distinct_tracks": len({row["track_id"] for row in rows}),
            "session_gate_passed": session_pass,
        }
    confirm = [
        row for row in evaluated if row["decision"] == "CONFIRM_APPROACH"
    ]
    contradict = [
        row for row in evaluated if row["decision"] == "CONTRADICT_APPROACH"
    ]
    confirm_precision = _fraction(
        sum(bool(row["correct"]) for row in confirm), len(confirm)
    )
    contradict_precision = _fraction(
        sum(bool(row["correct"]) for row in contradict), len(contradict)
    )
    distinct_tracks = len(
        {(row["sequence"], row["track_id"]) for row in evaluated}
    )
    pooled_gate = bool(
        len(evaluated) >= MINIMUM_TOTAL_EVIDENCE
        and distinct_tracks >= MINIMUM_DISTINCT_TRACKS
        and confirm_precision is not None
        and confirm_precision >= MINIMUM_POOLED_DIRECTION_PRECISION
        and contradict_precision is not None
        and contradict_precision >= MINIMUM_POOLED_DIRECTION_PRECISION
    )
    passed = pooled_gate and all(session_passes)
    terminal = (
        "ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS"
        if passed
        else "ANNOTATION_TRACK_SOURCE_CONFIRMATION_NOT_MET"
    )
    result = {
        "schema": "blindassist.dual_loop_causal_track_tristate_confirmation_result.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "terminal": terminal,
        "confirmation_passed": passed,
        "freeze_sha256": sha256_file(freeze_path),
        "producer_output_sha256": sha256_file(producer_output),
        "producer_receipt_sha256": sha256_file(producer_receipt_path),
        "truth_acquisition_receipt_sha256": sha256_file(
            truth_acquisition_receipt
        ),
        "truth_rate_deadband_mps": TRUTH_RATE_DEADBAND_MPS,
        "gates": {
            "minimum_total_evidence": MINIMUM_TOTAL_EVIDENCE,
            "minimum_session_evidence": MINIMUM_SESSION_EVIDENCE,
            "minimum_session_coverage": MINIMUM_SESSION_COVERAGE,
            "minimum_session_precision": MINIMUM_SESSION_PRECISION,
            "minimum_pooled_direction_precision": MINIMUM_POOLED_DIRECTION_PRECISION,
            "minimum_distinct_tracks": MINIMUM_DISTINCT_TRACKS,
        },
        "metrics": {
            "opportunity_rows": sum(opportunity.values()),
            "evidence_rows": len(evaluated),
            "coverage": _fraction(len(evaluated), sum(opportunity.values())),
            "correct_rows": sum(bool(row["correct"]) for row in evaluated),
            "precision": _fraction(
                sum(bool(row["correct"]) for row in evaluated), len(evaluated)
            ),
            "confirm_rows": len(confirm),
            "confirm_correct_rows": sum(
                bool(row["correct"]) for row in confirm
            ),
            "confirm_precision": confirm_precision,
            "contradict_rows": len(contradict),
            "contradict_correct_rows": sum(
                bool(row["correct"]) for row in contradict
            ),
            "contradict_precision": contradict_precision,
            "distinct_tracks": distinct_tracks,
        },
        "per_session": per_session,
        "pooled_gate_passed": pooled_gate,
        "evaluated_rows": evaluated,
        "claim_ceiling": "ANNOTATION_TRACK_MECHANISM_CONFIRMATION_ONLY",
        "limitations": [
            "source target identity and boxes are JRDB annotations, not live detector output",
            "truth is JRDB 3D annotation center range, not mobile-device sensing",
            "frame rows within tracks are repeated longitudinal observations",
            "no alert, Android, product, or human-safety effectiveness claim",
        ],
    }
    write_exclusive(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--truth-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--producer-output", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.truth_root,
                args.truth_acquisition_receipt,
                args.producer_output,
                args.producer_receipt,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
