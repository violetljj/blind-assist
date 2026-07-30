from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.dual_loop_d0_egomotion_error_attribution_r1.freeze_dependency_receipt import (
    build_dependency_receipt,
    intervals_overlap,
    sha256_file,
    write_exclusive_json,
)


class DependencyReceiptTest(unittest.TestCase):
    def test_closed_interval_overlap_includes_shared_endpoint(self) -> None:
        first = {"start_timestamp_ns": 10, "end_timestamp_ns": 20}
        touching = {"start_timestamp_ns": 20, "end_timestamp_ns": 30}
        separated = {"start_timestamp_ns": 21, "end_timestamp_ns": 30}
        self.assertTrue(intervals_overlap(first, touching))
        self.assertFalse(intervals_overlap(first, separated))

    def test_synthetic_components_and_blocks_are_deterministic(self) -> None:
        rows = [
            self.event("track-000:event-0000", "track-000", 0, 10),
            self.event("track-001:event-0000", "track-001", 10, 20),
            self.event("track-000:event-0001", "track-000", 70_000_000_000, 70_000_000_010),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            first = build_dependency_receipt(path, enforce_frozen_identity=False)
            second = build_dependency_receipt(path, enforce_frozen_identity=False)
        self.assertEqual(first["event_bindings_sha256"], second["event_bindings_sha256"])
        self.assertEqual(first["cross_target_overlap_pair_count"], 1)
        self.assertEqual(first["same_target_overlap_pair_count"], 0)
        self.assertEqual(first["exact_overlap_component_count"], 2)
        self.assertEqual(first["component_size_counts"], {"1": 1, "2": 1})
        self.assertEqual(first["time_block"]["block_ids"], [0, 1])
        self.assertEqual(first["time_block"]["event_counts"], [2, 1])

    def test_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_exclusive_json(path, {"status": "VALID"})
            with self.assertRaises(FileExistsError):
                write_exclusive_json(path, {"status": "VALID"})

    def test_transitive_overlap_component_is_not_split(self) -> None:
        rows = [
            self.event("track-000:event-0000", "track-000", 0, 10),
            self.event("track-001:event-0000", "track-001", 10, 20),
            self.event("track-000:event-0001", "track-000", 20, 30),
        ]
        receipt = self.build_synthetic(rows)
        self.assertEqual(receipt["cross_target_overlap_pair_count"], 2)
        self.assertEqual(receipt["same_target_overlap_pair_count"], 0)
        self.assertEqual(receipt["exact_overlap_component_count"], 1)
        self.assertEqual(receipt["component_size_counts"], {"3": 1})

    def test_midpoint_on_exact_sixty_second_boundary_enters_next_block(self) -> None:
        rows = [
            self.event("track-000:event-0000", "track-000", 0, 0),
            self.event(
                "track-001:event-0000",
                "track-001",
                60_000_000_000,
                60_000_000_000,
            ),
        ]
        receipt = self.build_synthetic(rows)
        self.assertEqual(receipt["time_block"]["block_ids"], [0, 1])
        self.assertEqual(receipt["time_block"]["event_counts"], [1, 1])

    def test_frozen_dependency_receipt_matches_real_golden_when_available(self) -> None:
        natural_events = Path(
            "artifacts.local/evidence/dual-loop/"
            "target-track-causal-radial-geometry-lite-r0/input-freeze/"
            "natural_events.jsonl"
        )
        if not natural_events.exists():
            self.skipTest("frozen ignored natural-events input is unavailable")
        receipt = build_dependency_receipt(natural_events)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dependency_receipt.json"
            write_exclusive_json(output, receipt)
            digest = sha256_file(output)
        self.assertEqual(
            digest,
            "0377944df2abdeb6044d49182e1f4bc1908b4bf8ba40eb632a091b4d2d10dc7f",
        )
        self.assertEqual(receipt["cross_target_overlap_pair_count"], 159)
        self.assertEqual(receipt["exact_overlap_component_count"], 310)
        self.assertEqual(
            receipt["time_block"]["event_counts"],
            [69, 38, 52, 101, 98, 111],
        )

    @staticmethod
    def event(
        event_id: str,
        target_id: str,
        start_timestamp_ns: int,
        end_timestamp_ns: int,
    ) -> dict[str, object]:
        return {
            "anchor_region": "CENTER",
            "capture_id": "REVEL_DYNAMIC_V1",
            "eligible_frame_count": 5,
            "end_timestamp_ns": end_timestamp_ns,
            "event_id": event_id,
            "primary_event_eligible": True,
            "start_timestamp_ns": start_timestamp_ns,
            "target_id": target_id,
            "truth_state": "approaching",
        }

    @staticmethod
    def build_synthetic(rows: list[dict[str, object]]) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            return build_dependency_receipt(path, enforce_frozen_identity=False)


if __name__ == "__main__":
    unittest.main()
