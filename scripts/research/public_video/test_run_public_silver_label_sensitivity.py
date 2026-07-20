#!/usr/bin/env python3
"""Pure tests for post-hoc label-influence ranking."""

from __future__ import annotations

import unittest

import run_public_silver_label_sensitivity as subject


class PublicSilverLabelSensitivityTest(unittest.TestCase):
    def test_rank_prefers_balanced_accuracy_then_minimum_recall(self) -> None:
        rows = [
            {"episode_id": "b", "balanced_accuracy_delta": 0.1, "minimum_class_recall_delta": 0.2},
            {"episode_id": "a", "balanced_accuracy_delta": 0.1, "minimum_class_recall_delta": 0.3},
            {"episode_id": "c", "balanced_accuracy_delta": -0.1, "minimum_class_recall_delta": 0.5},
        ]
        self.assertEqual(["a", "b", "c"], [row["episode_id"] for row in subject.rank_influence(rows)])


if __name__ == "__main__":
    unittest.main()
