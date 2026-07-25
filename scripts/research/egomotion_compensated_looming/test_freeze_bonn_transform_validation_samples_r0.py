#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_bonn_transform_validation_samples_r0 as subject  # noqa: E402


class BonnTransformValidationSampleFreezeTest(unittest.TestCase):
    def test_nearest_uses_absolute_timestamp_distance(self) -> None:
        rows = [["1.0", "a"], ["2.0", "b"], ["4.0", "c"]]
        index, row = subject.nearest(rows, 2.8)
        self.assertEqual(index, 1)
        self.assertEqual(row[1], "b")

    def test_offsets_are_fixed_before_payload_decode(self) -> None:
        self.assertEqual(subject.SAMPLE_OFFSETS_SECONDS, (0.0, 5.0, 9.9))
        self.assertEqual(subject.MAX_DEPTH_JOIN_DELTA_SECONDS, 0.040)


if __name__ == "__main__":
    unittest.main()
