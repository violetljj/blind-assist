import unittest

import numpy as np

import run_public_video_ego_route_distance_field_probe as subject


class EgoRouteDistanceFieldProbeTest(unittest.TestCase):
    def test_route_field_peaks_near_polyline(self):
        anchors = [{"horizon_ms": 1000, "point_xy_norm": [0.5, 0.5]}]
        field = subject.route_distance_field(anchors, 16, 1.0)
        self.assertGreater(field[8, 8], field[8, 0])

    def test_lateral_obstacle_mask_excludes_center(self):
        detections = [{"features": {"center_x_norm": 0.1, "bottom_y_norm": 0.9,
                                     "width_norm": 0.05, "height_norm": 0.2}}]
        mask = subject.obstacle_grid_mask(detections, 16, 0.5)
        self.assertFalse(mask[14, 8])
        self.assertTrue(mask[14, 1])

    def test_fixed_projection_is_deterministic(self):
        first = subject.fixed_projection(8, 4, 0)
        second = subject.fixed_projection(8, 4, 0)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
