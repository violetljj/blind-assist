import unittest
from collections import Counter

import cv2
import numpy as np
from audit_arkitscenes_counterexample_r0 import (
    ambiguity_class,
    dense_reconstructions,
    official_orientation_index,
    rotate_by_orientation,
    route_implication,
    trajectory_pose,
)


class ARKitScenesCounterexampleAuditTest(unittest.TestCase):
    def test_official_orientation_and_rotation(self):
        # Official ARKitScenes rectify_im.py case 2 is documented as left/1.
        _, pose = trajectory_pose(
            "803.47236621 1.6851560664954446 -1.7402208764128138 "
            "-0.8469396625258023 0.0404551 0.0562208 -0.00155703"
        )
        self.assertEqual(official_orientation_index(pose), 1)
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        expected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        np.testing.assert_array_equal(rotate_by_orientation(image, 1), expected)

    def test_band_local_nearest_never_borrows_across_band(self):
        depth = np.full((20, 30), 1000, dtype=np.uint16)
        confidence = np.full((20, 30), 2, dtype=np.uint8)
        confidence[:, 10:20] = 0
        contract = {"confidence_value": 2, "valid_depth_m": [0.25, 6.0], "minimum_source_valid_fraction": 0.5}
        global_depth, local_depth, _valid, _fraction, cross = dense_reconstructions(depth, confidence, contract)
        self.assertIsNotNone(global_depth)
        self.assertTrue(np.all(np.isnan(local_depth[:, 10:20])))
        self.assertGreater(cross, 0.0)

    def test_ambiguity_proxies(self):
        ambiguous = {"status": "VALID", "label": "AMBIGUOUS"}
        left = {"status": "VALID", "label": "RELATIVELY_OPEN_LEFT"}
        self.assertEqual(ambiguity_class(ambiguous, ambiguous), "REASONABLE_PROXY")
        self.assertEqual(ambiguity_class(ambiguous, left), "WRONG_REFUSAL_PROXY")
        self.assertEqual(ambiguity_class(left, ambiguous), "OVER_ANSWER_PROXY")

    def test_route_implication_is_bounded_to_unrectified_output(self):
        self.assertEqual(
            route_implication(Counter({"left": 150}), 150),
            "DO_NOT_USE_UNRECTIFIED_OUTPUT_AS_AUXILIARY_OR_FALLBACK",
        )
        self.assertEqual(
            route_implication(Counter({"upright": 150}), 150),
            "NO_ROUTE_ROLE_DECISION_FROM_THIS_CONSUMED_VISIT_ALONE",
        )


if __name__ == "__main__":
    unittest.main()
