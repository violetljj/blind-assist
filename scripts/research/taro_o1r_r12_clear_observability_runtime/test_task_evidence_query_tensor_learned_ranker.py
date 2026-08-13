import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_query_tensor_learned_ranker as subject


class QueryTensorLearnedRankerTest(unittest.TestCase):
    def test_feature_shape_contract(self) -> None:
        self.assertEqual((6, 3, 4), subject.CELL_SHAPE)
        self.assertEqual(9 * 6 * 4, subject.STATIC_TENSOR_FEATURE_COUNT)
        self.assertEqual(9 * 4, subject.QUERY_GEOMETRY_FEATURE_COUNT)
        self.assertEqual(6 * 4, subject.ALONG_GEOMETRY_FEATURE_COUNT)
        self.assertEqual(4 * 4, subject.HEIGHT_GEOMETRY_FEATURE_COUNT)
        self.assertEqual(325, subject.TOTAL_FEATURE_COUNT)

    def test_model_and_gates_unchanged_from_r21(self) -> None:
        self.assertEqual((32, 16), subject.r21.HIDDEN_WIDTHS)
        self.assertEqual((12013, 12031, 12047), subject.r21.SEEDS)
        self.assertEqual(0.75, subject.r21.RESIDUAL_SCALE)
        self.assertEqual(0.5, subject.r21.MIN_STRICT_WIN_FRACTION)


if __name__ == "__main__":
    unittest.main()
