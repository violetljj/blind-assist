#!/usr/bin/env python3
"""Repair stale RGB hashes caused by duplicate CrowdBot bag timestamps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


POLICY = "same_bag_timestamp_last_message_wins_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def repair_sequence(sequence_dir: Path, tool_sha256: str) -> dict[str, Any]:
    bundle_path = sequence_dir / "bundle.json"
    frames_path = sequence_dir / "frames.jsonl"
    receipt_path = sequence_dir / "duplicate-rgb-timestamp-repair-receipt.json"
    if receipt_path.exists():
        raise RuntimeError(f"repair receipt already exists: {receipt_path}")
    bundle = load_json(bundle_path)
    if bundle.get("candidate_outputs_executed") is not False:
        raise RuntimeError(f"candidate output leak in {bundle_path}")
    before_bundle_sha = sha256_file(bundle_path)
    before_frames_sha = sha256_file(frames_path)
    if bundle.get("frames_sha256") != before_frames_sha:
        raise RuntimeError(f"pre-repair frames binding mismatch: {frames_path}")
    rows = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != bundle.get("rgb_frame_count"):
        raise RuntimeError(f"pre-repair RGB count mismatch: {sequence_dir}")
    groups: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        timestamp = int(row["source_capture_timestamp_ns"])
        groups.setdefault(timestamp, []).append(row)
    repaired: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_extra_rows = 0
    stale_hash_rows = 0
    identical_early_rows = 0
    for timestamp, group in groups.items():
        rgb_paths = {row["rgb_path"] for row in group}
        if len(rgb_paths) != 1:
            raise RuntimeError(f"duplicate timestamp maps to multiple RGB paths: {sequence_dir} {timestamp}")
        rgb_path = sequence_dir / group[-1]["rgb_path"]
        actual_sha = sha256_file(rgb_path)
        if group[-1].get("rgb_sha256") != actual_sha:
            raise RuntimeError(f"last duplicate row does not bind final RGB file: {rgb_path}")
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_extra_rows += len(group) - 1
            for earlier in group[:-1]:
                if earlier.get("rgb_sha256") == actual_sha:
                    identical_early_rows += 1
                else:
                    stale_hash_rows += 1
        kept = dict(group[-1])
        kept["frame_id"] = f"{len(repaired):06d}"
        if kept.get("exact_aligned_depth"):
            depth_path = sequence_dir / kept["aligned_depth_path"]
            if sha256_file(depth_path) != kept.get("aligned_depth_sha256"):
                raise RuntimeError(f"exact depth binding mismatch: {depth_path}")
        repaired.append(kept)
    if duplicate_extra_rows == 0:
        raise RuntimeError(f"no duplicate timestamp rows to repair: {sequence_dir}")
    expected_rgb_paths = {path.resolve() for path in (sequence_dir / "rgb").glob("*.png")}
    repaired_rgb_paths = {(sequence_dir / row["rgb_path"]).resolve() for row in repaired}
    if expected_rgb_paths != repaired_rgb_paths:
        raise RuntimeError(f"RGB inventory does not collapse to unique timestamps: {sequence_dir}")
    exact_count = sum(bool(row.get("exact_aligned_depth")) for row in repaired)
    write_jsonl_atomic(frames_path, repaired)
    after_frames_sha = sha256_file(frames_path)
    bundle.update(
        {
            "rgb_frame_count": len(repaired),
            "exact_rgb_depth_frame_count": exact_count,
            "frames_sha256": after_frames_sha,
            "rgb_message_count": len(rows),
            "rgb_duplicate_extra_message_count": duplicate_extra_rows,
            "rgb_duplicate_timestamp_policy": POLICY,
        }
    )
    write_json_atomic(bundle_path, bundle)
    after_bundle_sha = sha256_file(bundle_path)
    receipt = {
        "schema": "blindassist_crowdbot_duplicate_rgb_timestamp_repair_receipt_r1",
        "authority": "materialization_integrity_repair_not_truth_not_candidate_score",
        "candidate_outputs_executed": False,
        "source_id": bundle["source_id"],
        "sequence_id": bundle["sequence_id"],
        "policy": POLICY,
        "tool_path": Path(__file__).as_posix(),
        "tool_sha256": tool_sha256,
        "before_frames_sha256": before_frames_sha,
        "after_frames_sha256": after_frames_sha,
        "before_bundle_sha256": before_bundle_sha,
        "after_bundle_sha256": after_bundle_sha,
        "rgb_message_count": len(rows),
        "unique_rgb_timestamp_count": len(repaired),
        "duplicate_timestamp_group_count": duplicate_groups,
        "duplicate_extra_row_count": duplicate_extra_rows,
        "stale_hash_row_count": stale_hash_rows,
        "identical_early_row_count": identical_early_rows,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--source-id", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = load_json(args.state)
    if (
        state.get("status") != "complete"
        or state.get("sequence_completed") != state.get("sequence_total")
        or state.get("candidate_outputs_executed") is not False
    ):
        raise RuntimeError("materialization must be complete and candidate-blind before repair")
    tool_sha = sha256_file(Path(__file__))
    receipts: list[dict[str, Any]] = []
    for source_id in args.source_id:
        sequence_root = args.dataset_root / source_id / "sequences"
        for sequence_dir in sorted(path for path in sequence_root.iterdir() if path.is_dir()):
            receipts.append(repair_sequence(sequence_dir, tool_sha))
    payload = {
        "schema": "blindassist_crowdbot_duplicate_rgb_timestamp_repair_report_r1",
        "authority": "materialization_integrity_repair_not_truth_not_candidate_score",
        "candidate_outputs_executed": False,
        "policy": POLICY,
        "tool_sha256": tool_sha,
        "sequence_count": len(receipts),
        "rgb_message_count": sum(row["rgb_message_count"] for row in receipts),
        "unique_rgb_timestamp_count": sum(row["unique_rgb_timestamp_count"] for row in receipts),
        "duplicate_timestamp_group_count": sum(row["duplicate_timestamp_group_count"] for row in receipts),
        "duplicate_extra_row_count": sum(row["duplicate_extra_row_count"] for row in receipts),
        "stale_hash_row_count": sum(row["stale_hash_row_count"] for row in receipts),
        "identical_early_row_count": sum(row["identical_early_row_count"] for row in receipts),
        "receipts": receipts,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"status": "DUPLICATE_RGB_TIMESTAMPS_REPAIRED", **{key: payload[key] for key in (
        "sequence_count",
        "rgb_message_count",
        "unique_rgb_timestamp_count",
        "duplicate_timestamp_group_count",
        "duplicate_extra_row_count",
        "stale_hash_row_count",
        "identical_early_row_count",
    )}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
