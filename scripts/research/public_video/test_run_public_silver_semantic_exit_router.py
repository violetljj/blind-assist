#!/usr/bin/env python3
"""Pure tests for the causal semantic-exit router."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_silver_semantic_exit_router as subject


class PublicSilverSemanticExitRouterTest(unittest.TestCase):
    def test_surface_count_ignores_unregistered_classes(self) -> None:
        summary = {"semantic_class_counts": {"sand box": 1, "sandwich": 5, "barrier": 2}}
        self.assertEqual(1, subject.surface_detection_count(summary))

    def test_timestamp_gap_is_causal_and_bounded(self) -> None:
        previous = {"end_timestamp_ms": 1000, "end_manifest_index": 2}
        current = {"start_timestamp_ms": 3500, "start_manifest_index": 3}
        accepted, evidence = subject.bounded_gap(previous, current, max_gap_ms=5000, max_manifest_gap=3)
        self.assertTrue(accepted)
        self.assertEqual("source_timestamp_ms", evidence["gap_kind"])
        current["start_timestamp_ms"] = 7001
        rejected, _ = subject.bounded_gap(previous, current, max_gap_ms=5000, max_manifest_gap=3)
        self.assertFalse(rejected)

    def test_router_changes_only_precomputed_exit_candidates(self) -> None:
        labels = np.asarray([1, 0, 1])
        baseline = np.asarray([1, 1, 1])
        routed, metrics = subject.routed_metrics(
            labels,
            ["hazard", "post-event", "new-dynamic"],
            baseline,
            {"post-event"},
        )
        self.assertEqual([1, 0, 1], routed.tolist())
        self.assertEqual(1.0, metrics["balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
