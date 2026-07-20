#!/usr/bin/env python3

import unittest

import numpy as np

import run_public_video_route_conditioned_real_transfer_probe as probe


class RealTransferProbeTest(unittest.TestCase):
    def test_global_field_features_are_finite_and_seven_dimensional(self) -> None:
        result = probe.global_field_features(np.arange(16, dtype=np.float64).reshape(4, 4))
        self.assertEqual(result.shape, (7,))
        self.assertTrue(np.isfinite(result).all())

    def test_build_event_matrices_uses_anchors_not_obstacle_hit(self) -> None:
        oracle = {"events": [{
            "item_id": "e1", "parent_source_id": "s1", "reference_intervention_required": True,
            "frames": [{"timestamp_ms": 1000, "valid_anchor_count": 3, "trace_intrusion_score": 1.0,
                        "anchors": [{"point_xy_norm": [0.2, 0.8], "obstacle_hit": True},
                                    {"point_xy_norm": [0.5, 0.6], "obstacle_hit": False},
                                    {"point_xy_norm": [0.8, 0.4], "obstacle_hit": True}]}],
        }]}
        maps = {("s1", 1000): np.arange(16, dtype=np.float64).reshape(4, 4)}
        global_x, route_x, labels, sources, ids, counts = probe.build_event_matrices(oracle, maps)
        self.assertEqual(global_x.shape, (1, 7))
        self.assertEqual(route_x.shape, (1, 13))
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(sources.tolist(), ["s1"])
        self.assertEqual(ids, ["e1"])
        self.assertEqual(counts, [1])

    def test_source_loso_is_deterministic_and_complete(self) -> None:
        features = np.asarray([[0.0], [0.1], [0.9], [1.0], [0.2], [0.8]])
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
        sources = np.asarray(["a", "b", "a", "b", "c", "c"])
        first, folds = probe.source_loso_predictions(features, labels, sources, 1.0)
        second, _ = probe.source_loso_predictions(features, labels, sources, 1.0)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(folds), 3)
        self.assertTrue(np.all(first >= 0))


if __name__ == "__main__":
    unittest.main()
