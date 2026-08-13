import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as subject
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer


class PairwiseRankerTest(unittest.TestCase):
    def test_reference_relative_features_remove_reference_constant(self) -> None:
        records = []
        for value in (1.0, 2.0):
            record = scorer.CandidateRecord("p", "FIT", "r", None, np.full(len(scorer.FEATURE_NAMES), value), {})  # type: ignore[arg-type]
            records.append(record)
        features = subject.reference_relative_features(records)
        self.assertEqual((2, len(scorer.FEATURE_NAMES) * 2), features.shape)
        self.assertAlmostEqual(-1.0, float(features[0, len(scorer.FEATURE_NAMES)]))
        self.assertAlmostEqual(1.0, float(features[1, len(scorer.FEATURE_NAMES)]))

    def test_pairwise_prediction_is_antisymmetric_under_feature_swap(self) -> None:
        model = (np.asarray([2.0]), np.asarray([0.0]), np.asarray([1.0]))
        scores = subject.predict_pairwise(np.asarray([[1.0], [-1.0]]), model)
        self.assertAlmostEqual(float(scores[0]), -float(scores[1]))


if __name__ == "__main__":
    unittest.main()
