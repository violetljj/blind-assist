from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro.audit_observation_pair_support import (
    PairSupportError,
    audit_cohorts,
    audit_pair_support,
)


POSE = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class ObservationPairSupportAuditTest(unittest.TestCase):
    def write_frame(
        self,
        root: Path,
        name: str,
        timestamp_ns: int,
        *,
        pose: object = POSE,
    ) -> None:
        payload = {
            "schema": "blindassist.taro.test_candidate_input.v1",
            "parent_id": "parent-a",
            "video_id": "video-a",
            "sensor_timestamp_ns": timestamp_ns,
            "camera_to_world_4x4": pose,
        }
        (root / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_counts_pose_valid_adjacent_pairs_in_each_window(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.write_frame(root, "a.json", 0)
            self.write_frame(root, "b.json", 500_000_000)
            self.write_frame(root, "c.json", 2_000_000_000)
            result = audit_pair_support("dense", root)
        self.assertEqual(3, result["frame_count"])
        self.assertEqual(2, result["adjacent_pair_count"])
        self.assertEqual(1, result["pairs_within_passive_window"])
        self.assertEqual(1, result["pose_valid_pairs_within_passive_window"])
        self.assertEqual(2, result["pairs_within_extended_window"])
        self.assertEqual("PASSIVE_PAIR_SUPPORT_AVAILABLE", result["decision"])

    def test_missing_pose_does_not_create_observability_support(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.write_frame(root, "a.json", 0, pose=None)
            self.write_frame(root, "b.json", 500_000_000)
            result = audit_cohorts([("missing-pose", root)])
        cohort = result["cohorts"][0]
        self.assertEqual(1, cohort["pairs_within_passive_window"])
        self.assertEqual(0, cohort["pose_valid_pairs_within_passive_window"])
        self.assertEqual(
            "CURRENT_COHORTS_NOT_EVALUABLE_FOR_PASSIVE_OBSERVABILITY",
            result["decision"],
        )

    def test_rejects_non_candidate_input_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "bad.json").write_text('{"schema":"other"}', encoding="utf-8")
            with self.assertRaises(PairSupportError):
                audit_pair_support("bad", root)


if __name__ == "__main__":
    unittest.main()
