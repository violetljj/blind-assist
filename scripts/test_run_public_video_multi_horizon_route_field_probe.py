import unittest

import numpy as np

import run_public_video_multi_horizon_route_field_probe as subject


class MultiHorizonRouteFieldProbeTest(unittest.TestCase):
    def test_horizon_channels_keep_distinct_points(self):
        anchors = [
            {"horizon_ms": 1000, "point_xy_norm": [0.25, 0.5]},
            {"horizon_ms": 2000, "point_xy_norm": [0.5, 0.5]},
            {"horizon_ms": 3000, "point_xy_norm": [0.75, 0.5]},
        ]
        fields = subject.horizon_fields(anchors, 16, 1.0, [1000, 2000, 3000])
        points = [subject.argmax_point(field) for field in fields]
        self.assertLess(points[0][0], points[1][0])
        self.assertLess(points[1][0], points[2][0])

    def test_argmax_returns_patch_center(self):
        field = np.zeros((4, 4))
        field[2, 1] = 1.0
        self.assertEqual((0.375, 0.625), subject.argmax_point(field))


if __name__ == "__main__":
    unittest.main()
