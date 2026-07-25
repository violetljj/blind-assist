#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_hot3d_clips_continuity_r0 as subject  # noqa: E402


def clip(sequence_id: str, start_ns: int) -> dict:
    return {
        "sequence_id": sequence_id,
        "device": "Aria",
        "per_frame_timestamps_ns": [
            {"214-1": start_ns + index * 33_333_333} for index in range(150)
        ],
    }


class Hot3dContinuityAuditTest(unittest.TestCase):
    def test_stream_timestamps_require_150_strict_rows(self) -> None:
        values = subject.stream_timestamps_ns(clip("P0001_a", 0))
        self.assertEqual(len(values), 150)
        malformed = clip("P0001_a", 0)
        malformed["per_frame_timestamps_ns"][1]["214-1"] = 0
        with self.assertRaises(ValueError):
            subject.stream_timestamps_ns(malformed)

    def test_stable_hash_is_reproducible_and_order_sensitive(self) -> None:
        self.assertEqual(subject.stable_hash("a", "b"), subject.stable_hash("a", "b"))
        self.assertNotEqual(subject.stable_hash("a", "b"), subject.stable_hash("b", "a"))


if __name__ == "__main__":
    unittest.main()
