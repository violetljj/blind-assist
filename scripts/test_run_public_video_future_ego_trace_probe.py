import unittest

import numpy as np

import run_public_video_future_ego_trace_probe as subject


class FutureEgoTraceGeometryTest(unittest.TestCase):
    def test_identity_maps_future_anchor_to_same_normalized_point(self):
        point = subject.map_future_anchor(np.eye(3), 400, 200, [0.5, 0.9])
        self.assertAlmostEqual(0.5, point[0])
        self.assertAlmostEqual(0.9, point[1])

    def test_inverse_translation_maps_future_anchor_back(self):
        matrix = np.asarray([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        point = subject.map_future_anchor(matrix, 400, 200, [0.5, 0.9])
        self.assertAlmostEqual(0.45, point[0])
        self.assertAlmostEqual(0.9, point[1])

    def test_expanded_detection_separates_center_hit_from_lateral_clear(self):
        detections = [{"xyxy": [180.0, 120.0, 220.0, 180.0]}]
        self.assertTrue(subject.point_hits_expanded_detection((0.5, 0.75), detections, 400, 200, 0.5))
        self.assertFalse(subject.point_hits_expanded_detection((0.1, 0.75), detections, 400, 200, 0.5))


if __name__ == "__main__":
    unittest.main()
