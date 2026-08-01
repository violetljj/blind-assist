from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qualify_r4_obstacle_reference_opportunity as module
from qualify_r4_obstacle_reference_opportunity import _decision, _firewall


class R4ObstacleReferenceOpportunityTest(unittest.TestCase):
    def test_obstacle_only_gates_can_qualify(self) -> None:
        known = {name: 0.2 for name in ("foot", "body", "head")}
        positive = {name: 5 for name in ("foot", "body", "head")}
        negative = {name: 20 for name in ("foot", "body", "head")}
        sensitivity = {
            str(value): {"positive": 1, "negative": 1}
            for value in (1, 2, 4, 8)
        }
        gates = {
            "minimum_known_coverage_each_height": 0.1,
            "minimum_positive_known_cells_each_height": 5,
            "minimum_negative_known_cells_each_height": 20,
        }
        self.assertTrue(
            all(
                _decision(
                    known, positive, negative, sensitivity, gates
                ).values()
            )
        )

    def test_ground_and_arm_firewalls_are_explicit(self) -> None:
        firewall = _firewall(reference_computed=True)
        self.assertTrue(firewall["obstacle_reference_grid_computed"])
        self.assertFalse(firewall["ground_reference_computed"])
        self.assertFalse(firewall["candidate_grid_computed"])
        self.assertFalse(firewall["angular_baseline_computed"])
        self.assertFalse(firewall["arm_metric_or_delta_computed"])
        source = inspect.getsource(module)
        self.assertNotIn("_ground_support", source)

    def test_missing_height_opportunity_rejects(self) -> None:
        known = {name: 0.2 for name in ("foot", "body", "head")}
        positive = {"foot": 5, "body": 4, "head": 5}
        negative = {name: 20 for name in ("foot", "body", "head")}
        sensitivity = {"1": {"positive": 1, "negative": 1}}
        gates = {
            "minimum_known_coverage_each_height": 0.1,
            "minimum_positive_known_cells_each_height": 5,
            "minimum_negative_known_cells_each_height": 20,
        }
        checks = _decision(
            known, positive, negative, sensitivity, gates
        )
        self.assertFalse(checks["obstacle_primary_positive_each_height"])


if __name__ == "__main__":
    unittest.main()
