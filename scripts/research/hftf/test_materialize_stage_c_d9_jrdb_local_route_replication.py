#!/usr/bin/env python3
"""Tests for JRDB local-route replication materialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d9_jrdb_local_route_replication import (
    future_target,
)


class JrdbLocalRouteReplicationTests(unittest.TestCase):
    def test_future_corridor_and_proximity_are_independent(self) -> None:
        labels = {
            index: [
                {
                    "label_id": "pedestrian:1",
                    "box": {"cx": 3.0, "cy": 0.5},
                }
            ]
            for index in range(1, 31)
        }
        target = future_target(labels, 0)
        self.assertIsNotNone(target)
        self.assertTrue(target["future_corridor_intrusion"])
        self.assertFalse(target["future_proximity_le_1_25m"])

    def test_close_lateral_person_is_not_corridor_intrusion(self) -> None:
        labels = {
            1: [
                {
                    "label_id": "pedestrian:2",
                    "box": {"cx": 0.2, "cy": 1.0},
                }
            ]
        }
        target = future_target(labels, 0)
        self.assertIsNotNone(target)
        self.assertTrue(target["future_proximity_le_1_25m"])
        self.assertFalse(target["future_corridor_intrusion"])


if __name__ == "__main__":
    unittest.main()
