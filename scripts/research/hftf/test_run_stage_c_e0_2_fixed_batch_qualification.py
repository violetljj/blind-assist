from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage_c_e0_2_fixed_batch_qualification as subject


class E02FixedBatchQualificationTest(unittest.TestCase):
    def test_role_metrics_require_distinct_sources_and_anchors(self) -> None:
        sources = [
            {
                "role": "dev",
                "trajectory": "a",
                "candidate_risk_proxy_cell_count": 2,
                "distinct_risk_proxy_anchors": [5, 10],
                "distinct_risk_proxy_directions": [0.0],
                "candidate_known_no_risk_count": 200,
            },
            {
                "role": "dev",
                "trajectory": "b",
                "candidate_risk_proxy_cell_count": 2,
                "distinct_risk_proxy_anchors": [5, 10],
                "distinct_risk_proxy_directions": [15.0],
                "candidate_known_no_risk_count": 200,
            },
        ]
        metrics = subject._role_metrics(sources)["dev"]
        self.assertEqual(metrics["risk_proxy_cell_count"], 4)
        self.assertEqual(metrics["distinct_risk_proxy_anchor_count"], 4)
        self.assertEqual(metrics["distinct_sources_with_risk_proxy"], 2)
        self.assertEqual(metrics["distinct_risk_proxy_direction_count"], 2)

    def test_role_gate_passes_exact_thresholds(self) -> None:
        metrics = {
            role: {
                "risk_proxy_cell_count": 4,
                "distinct_risk_proxy_anchor_count": 4,
                "distinct_sources_with_risk_proxy": 2,
                "distinct_risk_proxy_direction_count": 2,
                "known_no_risk_cell_count": 300,
            }
            for role in ("dev", "heldout")
        }
        gate = {
            role: {
                "minimum_risk_proxy_cells": 4,
                "minimum_distinct_risk_proxy_anchors": 4,
                "minimum_distinct_sources_with_risk_proxy": 2,
                "minimum_distinct_risk_proxy_directions": 2,
                "minimum_known_no_risk_cells": 300,
            }
            for role in ("dev", "heldout")
        }
        self.assertTrue(
            all(item["passed"] for item in subject._role_gate(metrics, gate))
        )

    def test_unlocked_batch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not source-locked"):
            subject._validate_lock(
                {},
                Path(__file__),
                {"schema": subject.LOCK_SCHEMA, "terminal": "NO"},
            )


if __name__ == "__main__":
    unittest.main()
