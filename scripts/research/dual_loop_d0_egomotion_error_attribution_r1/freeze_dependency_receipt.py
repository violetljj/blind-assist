from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_ID = "D0_EGOMOTION_ERROR_ATTRIBUTION_R1"
EXPECTED_NATURAL_EVENTS_SHA256 = (
    "078881620709efe17f74b8b01a5a76f4e861bfb6363143b9a9e0a589a87a030a"
)
EXPECTED_ROW_COUNT = 1660
EXPECTED_PRIMARY_EVENT_COUNT = 469
EXPECTED_CROSS_TARGET_OVERLAP_PAIR_COUNT = 159
EXPECTED_SAME_TARGET_OVERLAP_PAIR_COUNT = 0
EXPECTED_COMPONENT_COUNT = 310
EXPECTED_COMPONENT_SIZE_COUNTS = {
    "1": 222,
    "2": 48,
    "3": 24,
    "4": 9,
    "5": 3,
    "6": 3,
    "10": 1,
}
TIME_BLOCK_WIDTH_NS = 60_000_000_000
EXPECTED_TIME_BLOCK_EVENT_COUNTS = [69, 38, 52, 101, 98, 111]
EXPECTED_TARGET_IDS = {"track-000", "track-001"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def load_events(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank natural-event row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"natural-event row {line_number} must be an object")
            rows.append(row)

    primary = [row for row in rows if row.get("primary_event_eligible") is True]
    event_ids: set[str] = set()
    for row in primary:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("primary event_id must be non-empty")
        if event_id in event_ids:
            raise ValueError(f"duplicate primary event_id: {event_id}")
        event_ids.add(event_id)
        start = require_int(row.get("start_timestamp_ns"), f"{event_id}.start_timestamp_ns")
        end = require_int(row.get("end_timestamp_ns"), f"{event_id}.end_timestamp_ns")
        eligible = require_int(row.get("eligible_frame_count"), f"{event_id}.eligible_frame_count")
        if start > end:
            raise ValueError(f"primary event interval is reversed: {event_id}")
        if eligible < 5:
            raise ValueError(f"primary event has fewer than five frames: {event_id}")
        if row.get("capture_id") != "REVEL_DYNAMIC_V1":
            raise ValueError(f"unexpected capture_id: {event_id}")
        if row.get("target_id") not in EXPECTED_TARGET_IDS:
            raise ValueError(f"unexpected target_id: {event_id}")
    return rows, primary


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return max(first["start_timestamp_ns"], second["start_timestamp_ns"]) <= min(
        first["end_timestamp_ns"],
        second["end_timestamp_ns"],
    )


def build_dependency_receipt(
    natural_events_path: Path,
    *,
    enforce_frozen_identity: bool = True,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(natural_events_path)
    rows, primary = load_events(natural_events_path)
    if enforce_frozen_identity:
        if actual_sha256 != EXPECTED_NATURAL_EVENTS_SHA256:
            raise ValueError("natural-events SHA-256 drift")
        if len(rows) != EXPECTED_ROW_COUNT:
            raise ValueError("natural-events row-count drift")
        if len(primary) != EXPECTED_PRIMARY_EVENT_COUNT:
            raise ValueError("primary-event count drift")

    ordered = sorted(
        primary,
        key=lambda row: (
            row["start_timestamp_ns"],
            row["end_timestamp_ns"],
            row["target_id"],
            row["event_id"],
        ),
    )
    disjoint = DisjointSet(len(ordered))
    overlap_pairs: list[list[str]] = []
    cross_target_count = 0
    same_target_count = 0
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            if second["start_timestamp_ns"] > first["end_timestamp_ns"]:
                break
            if not intervals_overlap(first, second):
                continue
            pair = sorted([first["event_id"], second["event_id"]])
            overlap_pairs.append(pair)
            if first["target_id"] == second["target_id"]:
                same_target_count += 1
            else:
                cross_target_count += 1
            disjoint.union(first_index, second_index)

    component_members: dict[int, list[dict[str, Any]]] = {}
    for index, event in enumerate(ordered):
        component_members.setdefault(disjoint.find(index), []).append(event)
    components = sorted(
        component_members.values(),
        key=lambda members: min(
            (row["start_timestamp_ns"], row["event_id"]) for row in members
        ),
    )
    event_to_component: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for component_index, members in enumerate(components):
        component_id = f"component-{component_index:04d}"
        event_ids = sorted(row["event_id"] for row in members)
        for event_id in event_ids:
            event_to_component[event_id] = component_id
        component_rows.append(
            {
                "component_id": component_id,
                "event_count": len(members),
                "event_ids": event_ids,
                "start_timestamp_ns": min(row["start_timestamp_ns"] for row in members),
                "end_timestamp_ns": max(row["end_timestamp_ns"] for row in members),
            }
        )

    component_size_counts = {
        str(size): count
        for size, count in sorted(
            Counter(len(members) for members in components).items()
        )
    }
    origin_ns = min(row["start_timestamp_ns"] for row in ordered)
    event_bindings: list[dict[str, Any]] = []
    block_counts: Counter[int] = Counter()
    for event in sorted(ordered, key=lambda row: row["event_id"]):
        midpoint_ns = (
            event["start_timestamp_ns"] + event["end_timestamp_ns"]
        ) // 2
        block_id = (midpoint_ns - origin_ns) // TIME_BLOCK_WIDTH_NS
        block_counts[block_id] += 1
        event_bindings.append(
            {
                "event_id": event["event_id"],
                "target_id": event["target_id"],
                "truth_state": event["truth_state"],
                "anchor_region": event["anchor_region"],
                "start_timestamp_ns": event["start_timestamp_ns"],
                "end_timestamp_ns": event["end_timestamp_ns"],
                "midpoint_timestamp_ns": midpoint_ns,
                "overlap_component_id": event_to_component[event["event_id"]],
                "time_block_id_60s": int(block_id),
            }
        )
    block_ids = sorted(block_counts)
    block_event_counts = [block_counts[block_id] for block_id in block_ids]

    if enforce_frozen_identity:
        checks = [
            (
                cross_target_count == EXPECTED_CROSS_TARGET_OVERLAP_PAIR_COUNT,
                "cross-target overlap-pair count drift",
            ),
            (
                same_target_count == EXPECTED_SAME_TARGET_OVERLAP_PAIR_COUNT,
                "same-target overlap-pair count drift",
            ),
            (
                len(components) == EXPECTED_COMPONENT_COUNT,
                "overlap-component count drift",
            ),
            (
                component_size_counts == EXPECTED_COMPONENT_SIZE_COUNTS,
                "overlap-component size distribution drift",
            ),
            (
                block_ids == list(range(len(EXPECTED_TIME_BLOCK_EVENT_COUNTS))),
                "60-second time-block identity drift",
            ),
            (
                block_event_counts == EXPECTED_TIME_BLOCK_EVENT_COUNTS,
                "60-second time-block event-count drift",
            ),
        ]
        for passed, message in checks:
            if not passed:
                raise ValueError(message)

    overlap_pairs.sort()
    receipt = {
        "schema_version": "blindassist.d0_dependency_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "natural_events": {
            "path": natural_events_path.as_posix(),
            "sha256": actual_sha256,
            "row_count": len(rows),
            "primary_event_count": len(primary),
        },
        "independence": {
            "highest_independence_unit": "CAPTURE_SESSION",
            "capture_session_count": 1,
            "analysis_unit": "PARENT_NATURAL_EVENT",
            "independent_event_claim": False,
        },
        "overlap_definition": (
            "Closed event intervals overlap when "
            "max(start_timestamp_ns) <= min(end_timestamp_ns)."
        ),
        "cross_target_overlap_pair_count": cross_target_count,
        "same_target_overlap_pair_count": same_target_count,
        "overlap_pairs_sha256": canonical_sha256(overlap_pairs),
        "exact_overlap_component_count": len(components),
        "component_size_counts": component_size_counts,
        "components": component_rows,
        "time_block": {
            "origin_timestamp_ns": origin_ns,
            "width_ns": TIME_BLOCK_WIDTH_NS,
            "assignment": (
                "floor((floor((start_timestamp_ns + end_timestamp_ns) / 2) "
                "- origin_timestamp_ns) / width_ns)"
            ),
            "block_ids": block_ids,
            "event_counts": block_event_counts,
        },
        "event_bindings_sha256": canonical_sha256(event_bindings),
        "event_bindings": event_bindings,
        "candidate_output_opened": False,
        "production_ab_trace_opened": False,
        "old_f1b_decision_opened": False,
        "confirmation_opened": False,
        "errors": [],
    }
    return receipt


def write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build_dependency_receipt(args.natural_events)
    write_exclusive_json(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "primary_event_count": receipt["natural_events"][
                    "primary_event_count"
                ],
                "exact_overlap_component_count": receipt[
                    "exact_overlap_component_count"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
