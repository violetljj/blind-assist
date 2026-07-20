import unittest

import numpy as np

import run_public_video_marker_relation_linear_probe as subject


class MarkerRelationLinearProbeTest(unittest.TestCase):
    def test_relation_vector_has_contract_dimension_and_geometry(self) -> None:
        grid = np.zeros((43, 2, 2), dtype=np.float32)
        grid[:, 1, 1] = 2.0
        mask = np.asarray([[False, False], [False, True]])
        vector = subject.relation_vector(grid, mask)
        self.assertEqual(132, len(vector))
        np.testing.assert_allclose([0.25, 1.0, 1.0], vector[-3:])

    def test_marker_grid_mask_retains_sub_patch_detection(self) -> None:
        detection = {
            "features": {
                "center_x_norm": 0.24609375,
                "bottom_y_norm": 0.5694444444444444,
                "width_norm": 0.0109375,
                "height_norm": 0.03333333333333333,
            }
        }
        mask = subject.marker_grid_mask([detection], 16, 0.5)
        self.assertEqual(1, int(mask.sum()))
        self.assertTrue(mask[8, 3])

    def test_source_class_balancing_equalizes_source_mass(self) -> None:
        sources = np.asarray(["a", "a", "a", "b", "b"])
        active = np.asarray([True, False, False, False, False])
        weights = subject.source_class_balanced_weights(sources, active)
        self.assertAlmostEqual(0.5, float(weights[sources == "a"].sum()))
        self.assertAlmostEqual(0.5, float(weights[sources == "b"].sum()))
        self.assertAlmostEqual(0.25, float(weights[0]))
        self.assertAlmostEqual(0.25, float(weights[1:].take([0, 1]).sum()))

    def test_roc_auc_handles_ties(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.0, 1.0, 0.0, 0.5])
        self.assertEqual(1.0, subject.roc_auc(labels, scores))

    def test_weighted_ridge_learns_order(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = np.asarray([0.0, 0.0, 1.0, 1.0])
        weights = np.full(4, 0.25)
        model = subject.fit_weighted_ridge(x, y, weights, alpha=0.1)
        prediction = subject.predict_ridge(model, x)
        self.assertGreater(prediction[-1], prediction[0])


if __name__ == "__main__":
    unittest.main()
