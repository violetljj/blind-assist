#!/usr/bin/env python3
"""Pure tests for the risk-profile exit-router experiment."""

from __future__ import annotations

import unittest

import run_public_silver_risk_profile_exit_router as subject


class RiskProfileExitRouterTests(unittest.TestCase):
    def test_selected_detection_count_unions_groups(self) -> None:
        summary = {
            "semantic_class_counts": {
                "sand box": 1,
                "construction site": 2,
                "sandwich": 8,
            }
        }
        self.assertEqual(1, subject.selected_detection_count(summary, ["surface_material"]))
        self.assertEqual(2, subject.selected_detection_count(summary, ["barrier_structure"]))
        self.assertEqual(
            3,
            subject.selected_detection_count(summary, ["surface_material", "barrier_structure"]),
        )

    def test_unknown_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown semantic group"):
            subject.selected_detection_count({"semantic_class_counts": {}}, ["unknown"])


if __name__ == "__main__":
    unittest.main()
