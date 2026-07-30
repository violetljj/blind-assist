#!/usr/bin/env python3
"""Freeze truth-window to source-frame membership without opening candidate output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
TRUTH_LEDGER = (
    "artifacts.local/evidence/dual-loop/f1a-negative-category-supplement-r1/"
    "combined_event_window_ledger.jsonl"
)
TRUTH_LEDGER_SHA256 = "4f514b1277449d754d2ca45469d655610c55d4b56d2e2c1031efbf7c8b9d23c5"
SESSIONS = (
    "defaced_2021-03-27-11-51-18_filtered_lidar_odom",
    "defaced_2021-03-27-11-55-00_filtered_lidar_odom",
)
CANDIDATE_OUTPUT_DIRECTORY_NAMES = (
    "device-producer",
    "sealed-producer",
    "evaluation",
    "confirmation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_ns(value: Any) -> int:
    scaled = Decimal(str(value)) * Decimal(1_000_000_000)
    integral = scaled.to_integral_exact()
    if scaled != integral:
        raise ValueError(f"truth boundary cannot be represented as integer ns: {value}")
    return int(integral)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    truth_path = repo_root / TRUTH_LEDGER
    if sha256_file(truth_path) != TRUTH_LEDGER_SHA256:
        raise ValueError("truth ledger hash mismatch")

    evidence_root = (
        repo_root
        / "artifacts.local"
        / "evidence"
        / "dual-loop"
        / "production-temporal-geometry-factorial-ab-r0"
    )
    candidate_namespace_state = {
        name: (evidence_root / name).exists() for name in CANDIDATE_OUTPUT_DIRECTORY_NAMES
    }
    if any(candidate_namespace_state.values()):
        raise ValueError("candidate output namespace is not empty")

    truth_rows = [
        json.loads(line)
        for line in truth_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decision_rows = [row for row in truth_rows if row.get("role") == "DECISION"]
    if len(decision_rows) != 17:
        raise ValueError(f"expected 17 DECISION truth items, found {len(decision_rows)}")

    frames_by_session: dict[str, list[tuple[str, int]]] = {}
    for session_id in SESSIONS:
        ledger_path = (
            repo_root
            / "artifacts.local"
            / "camera-source-prescreen-r1"
            / "dataset"
            / "crowdbot_0327_shared_control"
            / "sequences"
            / session_id
            / "frames.jsonl"
        )
        rows = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        origin_ns = int(rows[0]["source_capture_timestamp_ns"])
        frames_by_session[session_id] = [
            (str(row["frame_id"]), int(row["source_capture_timestamp_ns"]) - origin_ns)
            for row in rows
        ]

    item_receipts: list[dict[str, Any]] = []
    memberships: dict[tuple[str, str], list[str]] = defaultdict(list)
    scoreable_positive_ids: list[str] = []
    unavailable_positive_ids: list[str] = []
    session_denominators: dict[str, Counter[str]] = {
        session_id: Counter() for session_id in SESSIONS
    }

    for row in decision_rows:
        session_id = str(row["session_id"])
        if session_id not in frames_by_session:
            raise ValueError(f"unexpected session in truth ledger: {session_id}")
        if row["positive_or_negative"] == "POSITIVE":
            item_id = str(row["event_id"])
            valid_start_ns = exact_ns(row["alertable_start_interval_seconds"]["end"])
            valid_end_ns = exact_ns(row["end_or_clear_interval_seconds"]["start"])
            premature_start_ns = exact_ns(row["onset_interval_seconds"]["start"])
            premature_end_ns = exact_ns(row["alertable_start_interval_seconds"]["start"])
            valid_frames = [
                frame_id
                for frame_id, relative_ns in frames_by_session[session_id]
                if valid_start_ns <= relative_ns <= valid_end_ns
            ]
            premature_frames = [
                frame_id
                for frame_id, relative_ns in frames_by_session[session_id]
                if premature_start_ns <= relative_ns < premature_end_ns
            ]
            for frame_id in valid_frames:
                memberships[(session_id, frame_id)].append(f"{item_id}:VALID_POSITIVE")
            for frame_id in premature_frames:
                memberships[(session_id, frame_id)].append(f"{item_id}:PREMATURE_POSITIVE")
            scoreable = bool(valid_frames)
            if scoreable:
                scoreable_positive_ids.append(item_id)
                session_denominators[session_id]["positive"] += 1
            else:
                unavailable_positive_ids.append(item_id)
            item_receipts.append(
                {
                    "item_id": item_id,
                    "session_id": session_id,
                    "item_kind": "positive_event",
                    "valid_interval_ns": [valid_start_ns, valid_end_ns],
                    "valid_frame_count": len(valid_frames),
                    "premature_interval_ns": [premature_start_ns, premature_end_ns],
                    "premature_interval_end_exclusive": True,
                    "premature_frame_count": len(premature_frames),
                    "temporal_scoring_status": "SCOREABLE" if scoreable else "TEMPORAL_SCORING_NOT_EVALUABLE",
                }
            )
        elif row["positive_or_negative"] == "NEGATIVE":
            item_id = str(row["negative_window_id"])
            start_ns = exact_ns(row["window_interval_seconds"]["start"])
            end_ns = exact_ns(row["window_interval_seconds"]["end"])
            negative_frames = [
                frame_id
                for frame_id, relative_ns in frames_by_session[session_id]
                if start_ns <= relative_ns <= end_ns
            ]
            if not negative_frames:
                raise ValueError(f"negative window has zero frames: {item_id}")
            for frame_id in negative_frames:
                memberships[(session_id, frame_id)].append(f"{item_id}:NEGATIVE")
            session_denominators[session_id]["negative"] += 1
            item_receipts.append(
                {
                    "item_id": item_id,
                    "session_id": session_id,
                    "item_kind": "negative_window",
                    "interval_ns": [start_ns, end_ns],
                    "frame_count": len(negative_frames),
                    "temporal_scoring_status": "SCOREABLE",
                }
            )
        else:
            raise ValueError(f"unexpected truth polarity: {row.get('positive_or_negative')}")

    overlaps = [
        {
            "session_id": session_id,
            "frame_id": frame_id,
            "memberships": labels,
        }
        for (session_id, frame_id), labels in memberships.items()
        if len(labels) > 1
    ]
    if overlaps:
        raise ValueError(f"cross-item or cross-class frame overlap: {overlaps[:3]}")
    if sorted(unavailable_positive_ids) != ["F1A-P-007", "F1A-P-009"]:
        raise ValueError(f"unexpected unavailable positives: {unavailable_positive_ids}")
    if len(scoreable_positive_ids) != 8:
        raise ValueError(f"expected 8 scoreable positives, found {len(scoreable_positive_ids)}")
    if sum(counter["negative"] for counter in session_denominators.values()) != 7:
        raise ValueError("expected 7 scoreable negative windows")

    receipt = {
        "schema_version": "blindassist.dual_loop_truth_membership_preflight.v1",
        "protocol_id": PROTOCOL_ID,
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "status": "VALID",
        "truth_ledger_path": TRUTH_LEDGER,
        "truth_ledger_sha256": TRUTH_LEDGER_SHA256,
        "truth_opened": True,
        "candidate_output_opened": False,
        "candidate_output_namespace_audit": {
            "root": str(evidence_root.relative_to(repo_root)).replace("\\", "/"),
            "required_absent_directories": candidate_namespace_state,
            "all_absent": True,
        },
        "raw_truth_item_count": len(decision_rows),
        "scoreable_positive_ids": sorted(scoreable_positive_ids),
        "temporal_scoring_not_evaluable_positive_ids": sorted(unavailable_positive_ids),
        "scoreable_positive_count": len(scoreable_positive_ids),
        "scoreable_negative_count": 7,
        "fixed_scored_item_denominator": 15,
        "session_scored_denominators": {
            session_id: {
                "positive": counter["positive"],
                "negative": counter["negative"],
                "total": counter["positive"] + counter["negative"],
            }
            for session_id, counter in session_denominators.items()
        },
        "cross_item_or_class_frame_overlap_count": 0,
        "item_membership": sorted(item_receipts, key=lambda item: item["item_id"]),
        "errors": [],
    }
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps({"status": "VALID", "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
