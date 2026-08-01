from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_stage_c_e0_teacher_opportunity as subject


class TeacherOpportunityTest(unittest.TestCase):
    def test_role_metrics_deduplicate_physical_anchor_across_horizons(self) -> None:
        source = {
            "role": "heldout",
            "trajectory": "t",
            "summary_by_horizon": {
                "0.4": {
                    "candidate_risk_proxy_cell_count": 2,
                    "candidate_known_no_risk_count": 100,
                    "distinct_risk_proxy_anchors": [5, 10],
                    "distinct_risk_proxy_directions": [0.0],
                },
                "0.8": {
                    "candidate_risk_proxy_cell_count": 1,
                    "candidate_known_no_risk_count": 100,
                    "distinct_risk_proxy_anchors": [5],
                    "distinct_risk_proxy_directions": [15.0],
                },
            },
        }
        metrics = subject._role_metrics([source])["heldout"]
        self.assertEqual(metrics["risk_proxy_cell_count"], 3)
        self.assertEqual(metrics["distinct_risk_proxy_anchor_count"], 2)
        self.assertEqual(metrics["distinct_risk_proxy_direction_count"], 2)

    def test_role_gate_passes_exact_thresholds(self) -> None:
        metrics = {
            role: {
                "risk_proxy_cell_count": 2,
                "distinct_risk_proxy_anchor_count": 2,
                "distinct_sources_with_risk_proxy": 1,
                "distinct_risk_proxy_direction_count": 1,
                "known_no_risk_cell_count": 100,
            }
            for role in ("train", "dev", "heldout")
        }
        gates = {
            role: {
                "minimum_risk_proxy_cells": 2,
                "minimum_distinct_risk_proxy_anchors": 2,
                "minimum_distinct_sources_with_risk_proxy": 1,
                "minimum_distinct_risk_proxy_directions": 1,
            }
            for role in ("train", "dev", "heldout")
        }
        gates["minimum_known_no_risk_cells_each_role"] = 100
        self.assertTrue(
            all(item["passed"] for item in subject._role_gate(metrics, gates))
        )

    def test_source_gate_fails_known_fraction(self) -> None:
        source = {
            "role": "train",
            "trajectory": "t",
            "formal_anchor_count": 100,
            "ground_plane_known_fraction": 1.0,
            "history_speed_eligible_fraction": 1.0,
            "summary_by_horizon": {
                "0.4": {
                    "candidate_known_direction_fraction": 0.69,
                    "candidate_lost_known_direction_count": 0,
                    "unknown_to_safe_violation_count": 0,
                }
            },
        }
        gates = {
            "minimum_formal_anchor_count": 80,
            "minimum_ground_plane_known_fraction": 0.95,
            "minimum_history_speed_eligible_fraction": 0.95,
            "minimum_candidate_known_direction_fraction_each_future_horizon": 0.7,
            "maximum_candidate_known_cells_lost_vs_baseline": 0,
            "maximum_unknown_to_safe_violations": 0,
        }
        result = subject._source_gate(source, gates)
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["failures"], ["0.4:candidate_known_fraction"]
        )


if __name__ == "__main__":
    unittest.main()
