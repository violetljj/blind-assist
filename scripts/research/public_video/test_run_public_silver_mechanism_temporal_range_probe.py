#!/usr/bin/env python3
"""Pure tests for mechanism-specific temporal-range evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_public_silver_mechanism_temporal_range_probe as probe


class PublicSilverMechanismTemporalRangeProbeTest(unittest.TestCase):
    def test_dynamic_range_tracks_peak_to_peak_occupancy(self) -> None:
        frames = [
            [{"area": 0.10, "corridor_overlap": 0.20}],
            [{"area": 0.40, "corridor_overlap": 0.50}],
            [],
        ]
        self.assertAlmostEqual(0.20, probe.dynamic_temporal_range(frames))

    def test_log_midpoint_is_between_training_classes(self) -> None:
        threshold = probe.log_midpoint_threshold(0.01, 0.04)
        self.assertGreater(threshold, 0.01)
        self.assertLess(threshold, 0.04)

    def test_leave_one_pair_out_accepts_clean_source_normalized_channels(self) -> None:
        rows = [
            {"counterfactual_pair_id": "a", "source_id": "s1", "no_alert_score": 0.001, "alert_score": 0.020},
            {"counterfactual_pair_id": "b", "source_id": "s2", "no_alert_score": 0.003, "alert_score": 0.030},
            {"counterfactual_pair_id": "c", "source_id": "s3", "no_alert_score": 0.002, "alert_score": 0.025},
        ]
        result = probe.leave_one_pair_out(rows)
        self.assertEqual(1.0, result["pair_ordering_rate"])
        self.assertEqual(1.0, result["held_out_endpoint_accuracy"])
        self.assertTrue(result["all_training_folds_separable"])

    def test_leave_one_pair_out_fails_closed_on_overlapping_training_scores(self) -> None:
        rows = [
            {"counterfactual_pair_id": "a", "source_id": "s1", "no_alert_score": 0.010, "alert_score": 0.020},
            {"counterfactual_pair_id": "b", "source_id": "s2", "no_alert_score": 0.030, "alert_score": 0.040},
            {"counterfactual_pair_id": "c", "source_id": "s3", "no_alert_score": 0.015, "alert_score": 0.025},
        ]
        result = probe.leave_one_pair_out(rows)
        self.assertFalse(result["all_training_folds_separable"])

    def test_independent_direction_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secondary-corridor-causal" / "range.json"
            with self.assertRaisesRegex(ValueError, "independent model direction"):
                probe.reject_independent_direction(path)


if __name__ == "__main__":
    unittest.main()
