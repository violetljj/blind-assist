#!/usr/bin/env python3
"""Tests for JRDB true-future transition materialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d15_jrdb_future_onset import current_state


class D15JrdbFutureOnsetTests(unittest.TestCase):
    def test_current_state_separates_corridor_and_proximity(self) -> None:
        state = current_state(
            [
                {
                    "box": {"cx": 0.1, "cy": 1.0},
                    "label_id": "near-lateral",
                },
                {
                    "box": {"cx": 3.0, "cy": 0.1},
                    "label_id": "far-corridor",
                },
            ]
        )
        self.assertTrue(state["proximity_le_1_25m"])
        self.assertTrue(state["corridor_intrusion"])
        self.assertEqual(state["valid_person_count"], 2)

    def test_empty_current_frame_is_safe(self) -> None:
        state = current_state([])
        self.assertFalse(state["proximity_le_1_25m"])
        self.assertFalse(state["corridor_intrusion"])
        self.assertIsNone(state["minimum_distance_m"])


if __name__ == "__main__":
    unittest.main()
