import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_weak_relation_head import (
    event_balanced_weights,
    fit_logistic,
    fold_assignments,
    training_phase_labels,
    weighted_standardize,
)


class SanpoWeakRelationHeadTest(unittest.TestCase):
    def test_bucket_stratified_folds_are_deterministic(self):
        events = [
            {
                "parent_event_id": f"{bucket}-{index}",
                "bucket": bucket,
            }
            for bucket in ("positive", "negative")
            for index in range(7)
        ]
        first = fold_assignments(events)
        second = fold_assignments(events)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {row["parent_event_id"] for row in events})
        for bucket in ("positive", "negative"):
            counts = [
                sum(
                    first[row["parent_event_id"]] == fold
                    for row in events
                    if row["bucket"] == bucket
                )
                for fold in range(5)
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_training_labels_exclude_positive_transition_gap(self):
        event = {
            "parent_event_id": "positive",
            "bucket": "blocking_obstacle_positive",
            "alertable_interval_frames": [0, 2],
            "passed_interval_frames": [6, 8],
            "frames": [
                {"timestamp_ms": index * 100} for index in range(9)
            ],
        }
        self.assertEqual(
            {0: 1, 2: 1, 6: 0, 8: 0},
            training_phase_labels(event),
        )

    def test_event_and_class_balancing(self):
        event_ids = ["a", "a", "b", "c", "c", "c"]
        labels = np.asarray([1, 1, 1, 0, 0, 0])
        weights = event_balanced_weights(event_ids, labels)
        self.assertAlmostEqual(0.5, float(weights[labels == 1].sum()))
        self.assertAlmostEqual(0.5, float(weights[labels == 0].sum()))
        self.assertAlmostEqual(weights[0], weights[1])
        self.assertAlmostEqual(weights[3], weights[4])

    def test_weighted_standardization_and_logistic_fit(self):
        features = np.asarray(
            [[-2.0], [-1.0], [1.0], [2.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        weights = np.full(4, 0.25, dtype=np.float64)
        mean, scale = weighted_standardize(features, weights)
        standardized = (features - mean) / scale
        coefficient, intercept, loss = fit_logistic(
            standardized,
            labels,
            weights,
        )
        probabilities = 1.0 / (
            1.0 + np.exp(-(standardized @ coefficient + intercept))
        )
        self.assertLess(loss, 0.7)
        self.assertTrue(np.all(probabilities[:2] < 0.5))
        self.assertTrue(np.all(probabilities[2:] > 0.5))


if __name__ == "__main__":
    unittest.main()
