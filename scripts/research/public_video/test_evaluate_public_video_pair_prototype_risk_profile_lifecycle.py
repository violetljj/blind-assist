import unittest

import numpy as np

import evaluate_public_video_pair_prototype_risk_profile_lifecycle as subject


class PairPrototypeRiskProfileLifecycleTest(unittest.TestCase):
    def test_causal_open_requires_consecutive_positive_samples(self) -> None:
        timestamp = subject.causal_open_timestamp([0, 1000, 2000, 3000], np.asarray([1.0, -1.0, 0.2, 0.3]), 2)
        self.assertEqual(3000, timestamp)

    def test_event_profile_uses_only_initial_baseline(self) -> None:
        models = [{"scale": np.ones(1), "weight": np.ones(1)}]
        vectors = np.asarray([[0.0], [0.0], [0.0], [1.0], [2.0]])
        profile = subject.event_profile(models, vectors, [0, 1000, 2000, 3000, 4000], 3, 2)
        self.assertEqual(4000, profile["open_timestamp_ms"])

    def test_safe_profile_does_not_open(self) -> None:
        models = [{"scale": np.ones(1), "weight": np.ones(1)}]
        vectors = np.asarray([[1.0], [1.0], [1.0], [0.9], [1.0]])
        profile = subject.event_profile(models, vectors, [0, 1000, 2000, 3000, 4000], 3, 2)
        self.assertIsNone(profile["open_timestamp_ms"])


if __name__ == "__main__":
    unittest.main()
