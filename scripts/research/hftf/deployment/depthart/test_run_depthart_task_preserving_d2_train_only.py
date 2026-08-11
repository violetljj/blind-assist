import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_train_only import (
    FEATURE_ORDER,
    TaskHead,
    candidate_features,
    chunk_schedule,
    train_head,
)


class D2TrainOnlyTest(unittest.TestCase):
    def test_schedule_is_four_by_six(self):
        protocol = {"execution": {"chunk_size_frames": 50}, "train_scope": [
            {"visit_id": str(i), "video_id": str(i + 10)} for i in range(4)
        ]}
        chunks = chunk_schedule(protocol)
        self.assertEqual(24, len(chunks))
        self.assertEqual((0, 50), (chunks[0]["frame_start"], chunks[0]["frame_stop"]))
        self.assertEqual((250, 300), (chunks[-1]["frame_start"], chunks[-1]["frame_stop"]))

    def test_feature_order_and_surrogate(self):
        geometry = {
            "valid_depth_fraction": 0.8,
            "ground_plane": {"support_fraction": 0.2, "median_residual_m": 0.03},
            "bands": {"center": {"clearance_m": None, "support_points": 30, "intrusion_points": 0,
                                   "observed_forward_m": 3.0,
                                   "occupied_by_horizon": {"1.0": False, "1.5": False, "2.0": False}}},
        }
        features, evidence = candidate_features(geometry, "center")
        self.assertEqual(11, len(FEATURE_ORDER))
        self.assertEqual([0.0, 1.0, 0.0], features[-3:])
        self.assertEqual(2.0, features[1])
        self.assertTrue(evidence["ground_plane_available"])

    def test_training_is_deterministic_and_277_parameters(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(12, 11))
        known = np.asarray([[1, 1, 1], [1, 1, 0], [0, 0, 0]] * 4, dtype=np.float64)
        occupied = np.asarray([[0, 1, 1], [1, 0, 0], [0, 0, 0]] * 4, dtype=np.float64)
        dataset = {
            "features": features, "known": known, "occupied": occupied,
            "raw_clearance": np.linspace(0.2, 1.8, 12),
            "truth_clearance": np.linspace(0.3, 1.7, 12),
            "clearance_paired": np.ones(12, dtype=bool),
        }
        left, _ = train_head(dataset, steps=2, seed=17)
        right, _ = train_head(dataset, steps=2, seed=17)
        self.assertEqual(277, sum(value.numel() for value in TaskHead().parameters()))
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
