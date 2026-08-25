import unittest

import numpy as np

from grail_visual_orientation_r1cv import (
    SIGN_THRESHOLD,
    arm_frames,
    estimate_sign,
    estimate_undirected_axis,
    ordinals_from_frames,
    predict_visual_frames,
)


class VisualOrientationR1CVTest(unittest.TestCase):
    def test_pca_selects_more_horizontal_component_without_sign(self) -> None:
        axis, source = estimate_undirected_axis([
            np.asarray([10.0, 10.0]), np.asarray([12.0, 30.0]), np.asarray([14.0, 50.0]),
        ])
        self.assertGreaterEqual(axis[0], 0.0)
        self.assertGreater(abs(axis[0]), abs(axis[1]))
        self.assertEqual(source, "CENTER_PCA_HORIZONTAL_COMPONENT")

    def test_sign_abstains_on_symmetric_gradient_and_directs_asymmetry(self) -> None:
        symmetric = np.zeros((20, 20, 3), dtype=np.uint8)
        symmetric[:, 4:6] = 255
        symmetric[:, 14:16] = 255
        self.assertFalse(estimate_sign(symmetric, (0, 0, 20, 20), np.asarray([1.0, 0.0]))["evaluable"])
        asymmetric = np.zeros((20, 20, 3), dtype=np.uint8)
        asymmetric[:, 15:18] = 255
        sign = estimate_sign(asymmetric, (0, 0, 20, 20), np.asarray([1.0, 0.0]))
        self.assertTrue(sign["evaluable"])
        self.assertGreater(sign["moment"], SIGN_THRESHOLD)
        self.assertGreater(sign["directed_axis"][0], 0.0)

    def test_unknown_sign_never_fabricates_multi_sibling_horizontal_slot(self) -> None:
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        candidates = [
            {"object_type": "Drawer", "bbox": [2, 5, 8, 12]},
            {"object_type": "Drawer", "bbox": [20, 5, 26, 12]},
        ]
        predicted = predict_visual_frames(image, candidates, [0, 0])
        oracle = {(0, "Drawer"): {"evaluable": True, "directed_axis": np.asarray([1.0, 0.0])}}
        final = arm_frames(image, predicted, oracle, "R1C_V_FINAL")
        self.assertEqual(
            ordinals_from_frames(candidates, [0, 0], final),
            [("NOT_EVALUABLE", "SINGLE"), ("NOT_EVALUABLE", "SINGLE")],
        )


if __name__ == "__main__":
    unittest.main()

