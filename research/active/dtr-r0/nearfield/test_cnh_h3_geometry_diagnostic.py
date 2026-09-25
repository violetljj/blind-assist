import unittest
import numpy as np
from cnh_h3_geometry_diagnostic import query_weights, train_threshold, summarize


class H3GeometryTest(unittest.TestCase):
    def test_enclosing_box_and_outside_range(self):
        weights = query_weights(np.array([[[-10, -10, .001], [10, 10, 10]],
                                         [[-10, -10, 10], [10, 10, 11]]]))
        np.testing.assert_allclose(weights[0, :, 1:], 1, atol=1e-12)
        np.testing.assert_array_equal(weights[1], 0)

    def test_left_right_reflection_and_vertical_exclusion(self):
        weights = query_weights().reshape(6, 8, 8, 16)
        np.testing.assert_allclose(weights[0], weights[4, :, ::-1], atol=1e-12)
        np.testing.assert_allclose(weights[1], weights[5, :, ::-1], atol=1e-12)
        np.testing.assert_array_equal(weights[1, :4], 0)

    def test_train_threshold_and_unknown(self):
        labels = np.array([0, 0, 1, 1, -1])
        scores = np.array([0., 1., 2., 3., 999.])
        threshold = train_threshold(labels, scores)
        self.assertEqual(threshold, 2.)
        result = summarize(labels, scores, threshold)
        self.assertEqual((result['tp'], result['fp'], result['unknown']), (2, 0, 1))
        self.assertEqual(result['auprc'], 1.)


if __name__ == '__main__':
    unittest.main()
