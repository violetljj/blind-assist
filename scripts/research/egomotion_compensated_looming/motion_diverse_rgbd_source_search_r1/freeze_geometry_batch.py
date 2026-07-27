"""Freeze the next deterministic geometry batch from pose-ranked queues."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["sequence_id"] != right["sequence_id"]:
        return False
    return not (
        Decimal(left["end_timestamp_s"]) <= Decimal(right["start_timestamp_s"])
        or Decimal(right["end_timestamp_s"]) <= Decimal(left["start_timestamp_s"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pose-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-file", type=Path, required=True)
    parser.add_argument("--positive-count", type=int, default=2)
    parser.add_argument("--below-count", type=int, default=2)
    parser.add_argument("--positive-offset", type=int, default=0)
    parser.add_argument("--below-offset", type=int, default=0)
    parser.add_argument("--batch-id", default="ETH3D_R1_INITIAL_2P2B_PROXY")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    queue_path = args.pose_queue.resolve()
    contract = load(contract_path)
    queue = load(queue_path)
    if queue["contract_sha256"] != sha(contract_path):
        raise ValueError("POSE_QUEUE_CONTRACT_DRIFT")
    positive_pairs = list(
        enumerate(
            queue["positive_proxy_queue"][
                args.positive_offset : args.positive_offset + args.positive_count
            ],
            start=args.positive_offset,
        )
    )
    positive = [row for _, row in positive_pairs]
    below_pairs = []
    skipped_below = []
    if args.below_count:
        for queue_index, row in enumerate(
            queue["below_proxy_queue"][args.below_offset :],
            start=args.below_offset,
        ):
            if any(overlaps(row, positive_row) for positive_row in positive):
                skipped_below.append(
                    {"window_id": row["window_id"], "reason": "OVERLAPS_FROZEN_POSITIVE_PROXY"}
                )
                continue
            below_pairs.append((queue_index, row))
            if len(below_pairs) == args.below_count:
                break
    if len(positive_pairs) != args.positive_count or len(below_pairs) != args.below_count:
        raise ValueError("INSUFFICIENT_NONOVERLAPPING_QUEUE_ROWS")
    windows = []
    seen = set()
    for proxy_queue, selected in (("positive", positive_pairs), ("below", below_pairs)):
        for queue_index, row in selected:
            if row["window_id"] in seen:
                raise ValueError("WINDOW_DUPLICATED_BETWEEN_QUEUES")
            seen.add(row["window_id"])
            windows.append(
                {
                    **row,
                    "proxy_queue": proxy_queue,
                    "proxy_queue_index": queue_index,
                    "geometry_role": "UNKNOWN",
                }
            )
    members = sorted(
        {
            member
            for window in windows
            for member in window["depth_members"]
        }
    )
    manifest = {
        "schema": "rcle.motion_diverse_rgbd.source_search.geometry_batch.v1",
        "protocol_id": contract["protocol_id"],
        "contract_sha256": sha(contract_path),
        "pose_queue_sha256": sha(queue_path),
        "batch_id": args.batch_id,
        "frozen_before_depth_geometry": True,
        "trajectory_role_authority": False,
        "windows": windows,
        "depth_member_count": len(members),
        "skipped_below_proxy_rows": skipped_below,
        "rgb_access_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.include_file.parent.mkdir(parents=True, exist_ok=True)
    args.include_file.write_text("\n".join(members) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "batch_id": manifest["batch_id"],
                "windows": [row["window_id"] for row in windows],
                "depth_member_count": len(members),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
