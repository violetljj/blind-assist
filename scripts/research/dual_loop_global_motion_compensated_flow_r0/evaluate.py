"""Event-level evaluation of the frozen compensated-flow output."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from produce import (
    ARM_ID,
    EXPECTED_REPLAY_SHA256,
    EXPECTED_ROWS,
    IMPLEMENTATION_ID,
    PARAMETER_SHA256,
    PROTOCOL_ID,
    sha256_file,
)


EXPECTED_EVENTS = 469
EXPECTED_TRUTH_ROWS = 17_160
EXPECTED_EVENT_LEDGER_ROWS = 1_660
DEADBAND = 0.02


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def state(score: float) -> str:
    if score > DEADBAND:
        return "approaching"
    if score < -DEADBAND:
        return "receding"
    return "quasi_static"


def wrong_signed(truth: str, predicted: str | None) -> bool:
    return (truth, predicted) in {
        ("approaching", "receding"),
        ("receding", "approaching"),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "events": total,
        "correct_events": sum(bool(row["correct"]) for row in rows),
        "correct_fraction": sum(bool(row["correct"]) for row in rows) / total,
        "wrong_signed_events": sum(bool(row["wrong_signed"]) for row in rows),
        "wrong_signed_fraction": sum(bool(row["wrong_signed"]) for row in rows) / total,
        "evaluable_events": sum(bool(row["evaluable"]) for row in rows),
        "evaluable_fraction": sum(bool(row["evaluable"]) for row in rows) / total,
        "median_event_coverage": float(median(row["coverage"] for row in rows)),
    }


def evaluate(
    replay_path: Path,
    producer_output_path: Path,
    producer_receipt_path: Path,
    truth_path: Path,
    events_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    if sha256_file(replay_path) != EXPECTED_REPLAY_SHA256:
        raise ValueError("replay input hash drift")
    receipt = read_json(producer_receipt_path)
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("truth_opened") is not False
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("implementation_id") != IMPLEMENTATION_ID
        or receipt.get("parameter_sha256") != PARAMETER_SHA256
        or receipt.get("output_sha256") != sha256_file(producer_output_path)
    ):
        raise ValueError("producer receipt does not close")
    replay = read_jsonl(replay_path)
    outputs = read_jsonl(producer_output_path)
    truth = read_jsonl(truth_path)
    events = read_jsonl(events_path)
    if (
        len(replay) != EXPECTED_ROWS
        or len(outputs) != EXPECTED_ROWS
        or len(truth) != EXPECTED_TRUTH_ROWS
        or len(events) != EXPECTED_EVENT_LEDGER_ROWS
    ):
        raise ValueError("frozen denominator drift")
    primary_events = [
        event for event in events if event.get("primary_event_eligible") is True
    ]
    if len(primary_events) != EXPECTED_EVENTS:
        raise ValueError("primary-event denominator drift")
    output_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row, source in zip(outputs, replay, strict=True):
        for field in ("source_frame_id", "target_id", "track_epoch", "region", "captured_at_ns"):
            if row[field] != source[field]:
                raise ValueError(f"producer identity drift: {field}")
        if (
            row["protocol_id"] != PROTOCOL_ID
            or row["implementation_id"] != IMPLEMENTATION_ID
            or row["parameter_sha256"] != PARAMETER_SHA256
            or row["arm_id"] != ARM_ID
        ):
            raise ValueError("producer implementation identity drift")
        key = (str(row["source_frame_id"]), str(row["target_id"]))
        if key in output_by_key:
            raise ValueError("duplicate producer output key")
        if (
            row.get("available_at_ns") != row["captured_at_ns"]
            or row.get("ttl_ns") != 100_000_000
            or row.get("valid_until_ns") != row["captured_at_ns"] + 100_000_000
        ):
            raise ValueError("producer time contract drift")
        quality = row.get("quality")
        if not isinstance(quality, dict) or not isinstance(quality.get("components"), dict):
            raise ValueError("producer quality schema drift")
        quality_score = quality.get("score")
        if (
            isinstance(quality_score, bool)
            or not isinstance(quality_score, (int, float))
            or not math.isfinite(float(quality_score))
            or not 0.0 <= float(quality_score) <= 1.0
        ):
            raise ValueError("producer quality value drift")
        score = row.get("signed_approach_rate_per_s")
        if row.get("abstention_reason") is None:
            if (
                not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(quality_score) < 0.50
            ):
                raise ValueError("admitted producer row is invalid")
        elif score is not None:
            raise ValueError("abstained producer row carries a score")
        output_by_key[key] = row
    expected_keys = {
        (str(row["source_frame_id"]), str(row["target_id"])) for row in replay
    }
    if set(output_by_key) != expected_keys:
        raise ValueError("producer output keyset drift")
    truth_by_index_target = {
        (int(row["source_frame_index"]), str(row["target_id"])): row for row in truth
    }
    event_rows: list[dict[str, Any]] = []
    for event in primary_events:
        candidate_rows: list[dict[str, Any]] = []
        for frame_index in range(int(event["start_source_frame_index"]), int(event["end_source_frame_index"]) + 1):
            truth_row = truth_by_index_target.get((frame_index, str(event["target_id"])))
            if truth_row is None or truth_row.get("event_id") != event["event_id"]:
                continue
            output = output_by_key.get((str(truth_row["source_frame_id"]), str(event["target_id"])))
            if output is not None:
                candidate_rows.append(output)
        finite = [
            float(row["signed_approach_rate_per_s"])
            for row in candidate_rows
            if row.get("abstention_reason") is None
            and isinstance(row.get("signed_approach_rate_per_s"), (int, float))
            and math.isfinite(float(row["signed_approach_rate_per_s"]))
        ]
        denominator = int(event["eligible_frame_count"])
        if len(candidate_rows) != denominator:
            raise ValueError(f"event row denominator drift: {event['event_id']}")
        coverage = len(finite) / denominator if denominator else 0.0
        evaluable = len(finite) >= 3 and coverage >= 0.50
        score = float(median(finite)) if evaluable else None
        predicted = state(score) if score is not None else None
        truth_state = str(event["truth_state"])
        event_rows.append(
            {
                "event_id": event["event_id"],
                "target_id": event["target_id"],
                "anchor_region": event["anchor_region"],
                "truth_state": truth_state,
                "denominator_rows": denominator,
                "finite_rows": len(finite),
                "coverage": coverage,
                "evaluable": evaluable,
                "event_score_per_s": score,
                "predicted_state": predicted,
                "correct": bool(evaluable and predicted == truth_state),
                "wrong_signed": bool(evaluable and wrong_signed(truth_state, predicted)),
            }
        )
    overall = metrics(event_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["truth_state"])].append(row)
    by_truth = {key: metrics(value) for key, value in sorted(grouped.items())}
    ready = bool(
        overall["correct_fraction"] >= 0.60
        and overall["wrong_signed_fraction"] <= 0.20
        and overall["evaluable_fraction"] >= 0.80
        and all(by_truth.get(value, {}).get("correct_fraction", 0.0) >= 0.50 for value in ("approaching", "quasi_static", "receding"))
    )
    result = {
        "schema": "blindassist.dual_loop_gmc_target_flow_evaluation.v1",
        "status": "COMPLETE",
        "stage": "DISCOVERY",
        "claim_ceiling": "SINGLE_CAPTURE_ORACLE_ROI_DEVELOPMENT_ONLY",
        "protocol_id": PROTOCOL_ID,
        "parameter_sha256": PARAMETER_SHA256,
        "event_deadband_per_s": DEADBAND,
        "threshold_search_performed": False,
        "primary_event_denominator": len(event_rows),
        "metrics": overall,
        "by_truth_state": by_truth,
        "readiness_passed": ready,
        "terminal": (
            "SOURCE_READINESS_MET_FOR_FUTURE_INDEPENDENT_SCREEN / DEVELOPMENT_ONLY"
            if ready
            else "SOURCE_READINESS_NOT_MET / INDEPENDENT_INFORMATION_NOT_EVALUATED / DEVELOPMENT_ONLY"
        ),
        "event_rows": event_rows,
        "producer_output_sha256": sha256_file(producer_output_path),
        "producer_receipt_sha256": sha256_file(producer_receipt_path),
        "truth_sha256": sha256_file(truth_path),
        "events_sha256": sha256_file(events_path),
        "limitations": [
            "single burned REveL capture",
            "oracle source target ROI",
            "Vicon truth is offline and noncausal",
            "retrospective Development because the same REveL truth was opened by predecessor work",
            "469 events include 159 overlapping cross-target pairs in 310 overlap components",
            "no dependency-aware uncertainty interval or inferential significance test",
            "no alert or fusion outcome",
            "no Android runtime evidence",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--producer-output", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.replay_input, args.producer_output, args.producer_receipt, args.truth, args.events, args.output)
    print(json.dumps({key: result[key] for key in ("status", "terminal", "metrics", "by_truth_state")}, sort_keys=True))


if __name__ == "__main__":
    main()
