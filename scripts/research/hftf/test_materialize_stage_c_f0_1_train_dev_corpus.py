from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_f0_1_train_dev_corpus import (
    _aggregate_records,
    _canonical_jsonl,
    _flatten_label,
    _nested_nullable_targets,
    _student_record_firewall,
)


class StageCF01CorpusMaterializationTest(unittest.TestCase):
    def test_unknown_risk_is_null_not_numeric_safe(self) -> None:
        known = np.zeros((6, 6, 2), dtype=bool)
        known[0, 0, 0] = True
        support = np.zeros((6, 6, 2), dtype=np.int64)
        label = _nested_nullable_targets(known, support)
        self.assertEqual(0, label["risk_target_nullable"][0][0][0])
        self.assertIsNone(label["risk_target_nullable"][1][0][0])
        restored_known, restored_risk = _flatten_label(label)
        self.assertTrue(restored_known[0, 0, 0])
        self.assertFalse(restored_known[1, 0, 0])
        self.assertEqual(0, restored_risk[0, 0, 0])

    def test_numeric_unknown_target_fails_closed(self) -> None:
        label = {
            "known_target": np.zeros((2, 6, 6), dtype=int).tolist(),
            "risk_target_nullable": np.zeros((2, 6, 6), dtype=int).tolist(),
        }
        with self.assertRaisesRegex(ValueError, "UNKNOWN risk"):
            _flatten_label(label)

    def test_student_record_rejects_teacher_only_keys(self) -> None:
        clean = {
            "history_rgb": [
                {
                    "relative_time_s": relative_time,
                    "image_path": str(Path("root") / "images" / f"{index}.png"),
                    "image_sha256": "hash",
                }
                for index, relative_time in enumerate(
                    (-0.8, -0.6, -0.4, -0.2, 0.0)
                )
            ],
            "labels": {"future": {"known_target": []}},
        }
        self.assertTrue(_student_record_firewall(clean))
        for key in (
            "future_rgb_path",
            "source_depth_path",
            "camera_pose",
            "semantic_class",
            "teacher_receipt",
        ):
            damaged = dict(clean)
            damaged[key] = "forbidden"
            self.assertFalse(_student_record_firewall(damaged), key)
        damaged_value = dict(clean)
        damaged_value["debug"] = "root/source_depth/train/0001.gz"
        self.assertFalse(_student_record_firewall(damaged_value))

    def test_aggregate_counts_nullable_targets(self) -> None:
        known = np.zeros((6, 6, 2), dtype=bool)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        known[0, 0, 0] = True
        support[0, 0, 0] = 2
        label = _nested_nullable_targets(known, support)
        record = {
            "session_id": "source",
            "role": "train",
            "teacher_view": "candidate",
            "labels": {"current": label, "future": label},
        }
        aggregate = _aggregate_records([record])["source"]
        self.assertEqual(1, aggregate["horizons"]["future"]["body"]["known"])
        self.assertEqual(
            1,
            aggregate["horizons"]["future"]["body"]["positive_known"],
        )
        self.assertEqual(
            35, aggregate["horizons"]["future"]["body"]["unknown"]
        )
        self.assertEqual(36, aggregate["denominator_per_height_per_horizon"])

    def test_canonical_jsonl_is_order_independent_for_keys(self) -> None:
        first = _canonical_jsonl([{"b": 2, "a": 1}])
        second = _canonical_jsonl([{"a": 1, "b": 2}])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
