#!/usr/bin/env python3
"""Tests for TartanGround true future-onset materialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d16_tartanground_future_onset import (
    onset_target,
)


class D16TartanGroundOnsetTests(unittest.TestCase):
    def test_current_risk_is_not_counted_as_future_onset(self) -> None:
        known = [[[1] * 6 for _ in range(6)] for _ in range(3)]
        current = [[[0.0] * 6 for _ in range(6)] for _ in range(3)]
        near = [[[0.0] * 6 for _ in range(6)] for _ in range(3)]
        far = [[[0.0] * 6 for _ in range(6)] for _ in range(3)]
        current[1][2][2] = 1.0
        near[1][2][2] = 1.0
        near[1][2][3] = 1.0
        record = {
            "labels": {
                "current": {
                    "known_target": known,
                    "risk_score_target_nullable": current,
                },
                "near": {
                    "known_target": known,
                    "risk_score_target_nullable": near,
                },
                "far": {
                    "known_target": known,
                    "risk_score_target_nullable": far,
                },
            }
        }
        result = onset_target(record)
        body_near = result["cell_onset"][0][0]
        self.assertEqual(body_near[2][2], 0)
        self.assertEqual(body_near[2][3], 1)
        self.assertTrue(result["sample_onset"]["near_body"])


if __name__ == "__main__":
    unittest.main()
