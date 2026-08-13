from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as canary


def frame(parent: str, timestamp: float, x: float) -> canary.Frame:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return canary.Frame(parent, timestamp, Path("rgb.png"), Path("depth.png"), transform)


class PositiveOracleCanaryTest(unittest.TestCase):
    def test_reference_support_requires_legal_micro_pair(self) -> None:
        frames = [
            frame("p", 0.0, 0.00),
            frame("p", 0.2, 0.02),
            frame("p", 0.4, 0.06),
            frame("p", 1.6, 0.12),
        ]
        rows = canary.build_reference_support(frames)
        self.assertEqual(1, len(rows))
        self.assertEqual("p:0.40000", rows[0].reference.frame_id)
        self.assertGreaterEqual(len(rows[0].micro_candidates), 1)

    def test_reference_selection_is_pose_only_and_deterministically_thinned(self) -> None:
        frames = [frame("p", index * 0.2, index * 0.03) for index in range(8)]
        selected = canary.select_references(canary.build_reference_support(frames), limit=2)
        self.assertEqual(2, len(selected))
        self.assertGreaterEqual(selected[1].reference.timestamp_s - selected[0].reference.timestamp_s, 0.5)
        self.assertEqual("p:1.00000", selected[-1].reference.frame_id)

    def test_task_oracle_uses_one_frame_and_prefers_recovery_without_false_occupied(self) -> None:
        reference = frame("p", 1.0, 0.10)
        left = canary._pair(reference, frame("p", 0.5, 0.04))
        right = canary._pair(reference, frame("p", 0.7, 0.07))
        row = canary.ReferenceSupport(reference, (left, right), (left, right))
        pair_rows = [
            {"pair": left, "coverage": 0.9, "states": (True, True, False)},
            {"pair": right, "coverage": 0.8, "states": (True, False, False)},
        ]
        labels = ("OCCUPIED_OBSERVED", "CLEAR_OBSERVED", "UNKNOWN")
        selected = canary.select_arm_rows(row, pair_rows, (False, False, False), labels)
        self.assertEqual(right.neighbor.frame_id, selected["task_directed_oracle"]["pair"].neighbor.frame_id)
        self.assertEqual(left.neighbor.frame_id, selected["passive"]["pair"].neighbor.frame_id)

    def test_unknown_is_not_counted_as_clear_negative(self) -> None:
        counts = canary._empty_counts()
        canary._accumulate(
            counts,
            (True, True, False),
            (False, False, False),
            ("OCCUPIED_OBSERVED", "UNKNOWN", "CLEAR_OBSERVED"),
            None,
        )
        self.assertEqual(1, counts["truth_clear"])
        self.assertEqual(1, counts["truth_unknown"])
        self.assertEqual(0, counts["false_occupied"])

    def test_exclusive_output_rejects_collision(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            canary._write_exclusive(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                canary._write_exclusive(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
