import unittest

import numpy as np

import evaluate_public_video_majority_pair_prototype_lifecycle as subject


class MajorityPairPrototypeLifecycleTest(unittest.TestCase):
    def test_bootstrap_prototype_orders_simple_deltas(self) -> None:
        deltas = np.asarray([[1.0, 0.0], [2.0, 0.1], [1.5, -0.1]])
        model, audit = subject.fit_bootstrap_prototype(deltas, np.asarray(["a", "b", "c"]), 4)
        projection = (deltas / model["scale"]) @ model["weight"]
        self.assertTrue(np.all(projection > 0.0))
        self.assertGreaterEqual(audit["sampled_unique_source_count"], 1)


if __name__ == "__main__":
    unittest.main()
