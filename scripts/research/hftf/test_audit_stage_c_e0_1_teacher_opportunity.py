from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_stage_c_e0_1_teacher_opportunity as subject


class E01TeacherOpportunityTest(unittest.TestCase):
    def test_anchor_requires_only_plus_two(self) -> None:
        self.assertEqual(subject._anchors(8), [5])
        self.assertEqual(subject._anchors(13), [5, 10])

    def test_source_gate_accepts_exact_thresholds(self) -> None:
        source = {
            "role": "dev",
            "trajectory": "t",
            "formal_anchor_count": 80,
            "ground_plane_known_fraction": 0.95,
            "history_speed_eligible_fraction": 0.95,
            "candidate_known_direction_fraction_0_4_s": 0.7,
            "candidate_lost_known_direction_count": 0,
            "unknown_to_safe_violation_count": 0,
        }
        gate = {
            "minimum_formal_anchor_count": 80,
            "minimum_ground_plane_known_fraction": 0.95,
            "minimum_history_speed_eligible_fraction": 0.95,
            "minimum_candidate_known_direction_fraction_0_4_s": 0.7,
            "maximum_candidate_known_cells_lost_vs_baseline": 0,
            "maximum_unknown_to_safe_violations": 0,
        }
        self.assertTrue(subject._source_gate(source, gate)["passed"])

    def test_role_gate_fails_missing_risk(self) -> None:
        sources = [
            {
                "role": role,
                "candidate_risk_proxy_cell_count": 0,
                "distinct_risk_proxy_anchors": [],
                "distinct_risk_proxy_directions": [],
                "candidate_known_no_risk_count": 100,
            }
            for role in ("dev", "heldout")
        ]
        gate = {
            role: {
                "minimum_risk_proxy_cells": 2,
                "minimum_distinct_risk_proxy_anchors": 2,
                "minimum_distinct_sources_with_risk_proxy": 1,
                "minimum_distinct_risk_proxy_directions": 1,
            }
            for role in ("dev", "heldout")
        }
        gate["minimum_known_no_risk_cells_each_role"] = 100
        self.assertFalse(
            all(item["passed"] for item in subject._role_gate(sources, gate))
        )


if __name__ == "__main__":
    unittest.main()
