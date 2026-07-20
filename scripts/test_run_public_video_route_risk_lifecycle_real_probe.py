#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_route_risk_lifecycle_real_probe as probe


class RouteRiskLifecycleTest(unittest.TestCase):
    def test_state_labels_apply_transitions_at_timestamp(self) -> None:
        frames = [{"timestamp_ms": value} for value in (0, 1000, 2000, 3000, 4000)]
        transitions = [{"timestamp_ms": 1000, "state": "intervention_needed"},
                       {"timestamp_ms": 3000, "state": "route_clear"}]
        self.assertEqual(probe.state_labels(frames, transitions).tolist(), [0, 1, 1, 0, 0])

    def test_hierarchical_weights_equalize_classes(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        sources = np.asarray(["a", "a", "b", "a", "b"])
        events = np.asarray(["e1", "e1", "e2", "e3", "e4"])
        weights = probe.hierarchical_weights(labels, sources, events)
        self.assertAlmostEqual(float(weights[labels == 0].sum()), float(weights[labels == 1].sum()))

    def test_first_consecutive_respects_gaps_and_after(self) -> None:
        times = [0, 1000, 3000, 4000, 5000]
        labels = [1, 1, 0, 0, 0]
        self.assertEqual(probe.first_consecutive(times, labels, 1, 2), 1000)
        self.assertEqual(probe.first_consecutive(times, labels, 0, 2, after=1000), 4000)


if __name__ == "__main__":
    unittest.main()
