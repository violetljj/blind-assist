from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA"
SOURCE_ID = "wikimedia_commons_matoaka_west_virginia_walk_2019"
EXPECTED_FRAMES = 10_724
EXPECTED_NEGATIVE_IDS = {f"F1A-N-{index:03d}" for index in range(8, 20)}
EXPECTED_POSITIVE_IDS = {f"F1A-P-{index:03d}" for index in range(11, 18)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def score_interval(
    rows: list[dict[str, Any]], start_s: float, end_s: float, field: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if start_s
        <= row["source_capture_timestamp_ns"] / 1_000_000_000
        <= end_s
        and row[field]
    ]
    return {
        "trigger_count": len(selected),
        "first_trigger_frame_id": selected[0]["frame_id"] if selected else None,
        "first_trigger_seconds": (
            selected[0]["source_capture_timestamp_ns"] / 1_000_000_000
            if selected
            else None
        ),
    }


def evaluate(trace_path: Path, receipt_path: Path, ledger_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_id"] != PROTOCOL_ID or receipt["status"] != "COMPLETE":
        raise ValueError("cross-source producer receipt is not complete")
    if receipt["truth_read"] or receipt["risk_mutation_count"] != 0:
        raise ValueError("producer truth firewall or risk non-mutation failed")
    if receipt["trace_sha256"] != sha256_file(trace_path):
        raise ValueError("cross-source trace identity mismatch")
    rows = read_jsonl(trace_path)
    if len(rows) != EXPECTED_FRAMES:
        raise ValueError("cross-source trace frame count mismatch")
    for index, row in enumerate(rows):
        if (
            row["protocol_id"] != PROTOCOL_ID
            or row["source_id"] != SOURCE_ID
            or row["frame_id"] != index
            or row["source_capture_timestamp_ns"] != index * 100_000_000
        ):
            raise ValueError(f"cross-source frame identity mismatch at {index}")
        if row["dual_loop"]["event_mutation_allowed"]:
            raise ValueError(f"event mutation was allowed at {index}")

    ledger = [
        row
        for row in read_jsonl(ledger_path)
        if row.get("source_id") == SOURCE_ID
        and row.get("session_id") == "commons_matoaka_walk_2019_full"
    ]
    negative = {
        row["negative_window_id"]: row
        for row in ledger
        if row["item_kind"] == "negative_window"
    }
    positive = {
        row["event_id"]: row
        for row in ledger
        if row["item_kind"] == "positive_event"
    }
    if set(negative) != EXPECTED_NEGATIVE_IDS or set(positive) != EXPECTED_POSITIVE_IDS:
        raise ValueError("Matoaka truth item identity mismatch")
    if any(
        row["role"] != "DEVELOPMENT"
        or row["outcome_access_state_after_repair"]
        != "TUNED_ON_PRIOR_PUBLIC_VIDEO_DEVELOPMENT_ONLY"
        for row in ledger
    ):
        raise ValueError("Matoaka evidence role drift")

    per_item: dict[str, Any] = {}
    for item_id, item in sorted(negative.items()):
        interval = item["window_interval_seconds"]
        per_item[item_id] = {
            "kind": "negative_window",
            "interval_seconds": interval,
            "baseline": score_interval(
                rows,
                float(interval["start"]),
                float(interval["end"]),
                "baseline_feedback_triggered",
            ),
            "candidate": score_interval(
                rows,
                float(interval["start"]),
                float(interval["end"]),
                "candidate_feedback_triggered",
            ),
        }
    for item_id, item in sorted(positive.items()):
        interval = item["alertable_start_interval_seconds"]
        baseline = score_interval(
            rows,
            float(interval["start"]),
            float(interval["end"]),
            "baseline_feedback_triggered",
        )
        candidate = score_interval(
            rows,
            float(interval["start"]),
            float(interval["end"]),
            "candidate_feedback_triggered",
        )
        delay_frames = (
            None
            if baseline["first_trigger_frame_id"] is None
            or candidate["first_trigger_frame_id"] is None
            else candidate["first_trigger_frame_id"] - baseline["first_trigger_frame_id"]
        )
        per_item[item_id] = {
            "kind": "positive_event",
            "interval_seconds": interval,
            "baseline": baseline,
            "candidate": candidate,
            "candidate_delay_frames": delay_frames,
        }

    baseline_negative_rows = sum(
        per_item[item_id]["baseline"]["trigger_count"] for item_id in negative
    )
    candidate_negative_rows = sum(
        per_item[item_id]["candidate"]["trigger_count"] for item_id in negative
    )
    baseline_negative_windows = sum(
        per_item[item_id]["baseline"]["trigger_count"] > 0 for item_id in negative
    )
    candidate_negative_windows = sum(
        per_item[item_id]["candidate"]["trigger_count"] > 0 for item_id in negative
    )
    baseline_positive_recall = sum(
        per_item[item_id]["baseline"]["trigger_count"] > 0 for item_id in positive
    )
    candidate_positive_recall = sum(
        per_item[item_id]["candidate"]["trigger_count"] > 0 for item_id in positive
    )
    delays = [
        per_item[item_id]["candidate_delay_frames"]
        for item_id in positive
        if per_item[item_id]["candidate_delay_frames"] is not None
    ]
    if candidate_positive_recall < baseline_positive_recall:
        terminal = "CROSS_SOURCE_POSITIVE_GUARDRAIL_FAILED"
    elif (
        candidate_negative_windows < baseline_negative_windows
        or candidate_negative_rows < baseline_negative_rows
    ):
        terminal = "CROSS_SOURCE_DEVELOPMENT_SIGNAL_REPLICATED"
    else:
        terminal = "CROSS_SOURCE_NO_EVENT_EFFECT"

    return {
        "schema_version": "blindassist.dual_loop_scene_scale_cross_source_result.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "authority": "LOCKED_CROSS_SOURCE_DEVELOPMENT",
        "inputs": {
            "trace_sha256": sha256_file(trace_path),
            "producer_receipt_sha256": sha256_file(receipt_path),
            "truth_ledger_sha256": sha256_file(ledger_path),
            "frame_count": len(rows),
            "truth_read_by_device_producer": False,
            "risk_mutation_count": 0,
        },
        "full_video": {
            "baseline_feedback_trigger_count":
                sum(row["baseline_feedback_triggered"] for row in rows),
            "candidate_feedback_trigger_count":
                sum(row["candidate_feedback_triggered"] for row in rows),
        },
        "scored": {
            "negative_window_count": len(negative),
            "baseline_alerted_negative_windows": baseline_negative_windows,
            "candidate_alerted_negative_windows": candidate_negative_windows,
            "baseline_negative_trigger_rows": baseline_negative_rows,
            "candidate_negative_trigger_rows": candidate_negative_rows,
            "positive_event_count": len(positive),
            "baseline_positive_recall": baseline_positive_recall,
            "candidate_positive_recall": candidate_positive_recall,
            "maximum_candidate_delay_frames": max(delays) if delays else None,
            "per_item": per_item,
        },
        "scientific_terminal": terminal,
        "claim_ceiling": "LOCKED_CROSS_SOURCE_DEVELOPMENT_ONLY",
        "important_limit": (
            "Matoaka labels are prior tuned Development evidence, not independent "
            "Confirmation or product/safety evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_exclusive(
        args.output,
        evaluate(args.trace, args.producer_receipt, args.ledger),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
