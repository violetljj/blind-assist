import unittest

import audit_explicit_route_geometry_conformance as subject


class ExplicitRouteGeometryConformanceTest(unittest.TestCase):
    def test_inside_box_hits_without_expansion(self):
        detections = [{"xyxy": [40, 60, 60, 80]}]
        self.assertTrue(subject.point_hits_normalized((0.5, 0.7), detections, 100, 100, 0.0))

    def test_half_height_expansion_is_inclusive(self):
        detections = [{"xyxy": [55, 80, 60, 90]}]
        self.assertTrue(subject.point_hits_normalized((0.5, 0.75), detections, 100, 100, 0.5))

    def test_point_outside_expanded_box_does_not_hit(self):
        detections = [{"xyxy": [55, 80, 60, 90]}]
        self.assertFalse(subject.point_hits_normalized((0.49, 0.75), detections, 100, 100, 0.5))

    def test_non_square_frame_uses_pixel_height_for_x_margin(self):
        detections = [{"xyxy": [550, 800, 600, 900]}]
        self.assertTrue(subject.point_hits_normalized((0.5, 0.4), detections, 1000, 2000, 0.5))


if __name__ == "__main__":
    unittest.main()
