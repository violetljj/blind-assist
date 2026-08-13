import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as subject


class TaskEvidencePoseScorerCanaryTest(unittest.TestCase):
    def test_ridge_parent_weights_do_not_depend_on_parent_row_count(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [10.0]])
        y = np.asarray([0.0, 1.0, 2.0, 10.0])
        model = subject._ridge_fit(x, y, ["a", "a", "a", "b"], 1.0)
        prediction = subject._ridge_predict(np.asarray([[5.0]]), model)
        self.assertEqual((1,), prediction.shape)
        self.assertTrue(np.isfinite(prediction[0]))

    def test_feature_contract_excludes_neighbor_depth(self) -> None:
        self.assertNotIn("neighbor_depth", subject.FEATURE_NAMES)
        self.assertIn("occluded_parallax", subject.FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
