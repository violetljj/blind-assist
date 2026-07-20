import unittest

import evaluate_public_video_explicit_choice_route_template as subject


class ExplicitChoiceRouteTemplateTest(unittest.TestCase):
    def test_direction_boundaries_are_fail_stable(self) -> None:
        self.assertEqual("LEFT", subject.direction_from_mean_x(0.469, 0.47, 0.53))
        self.assertEqual("STRAIGHT", subject.direction_from_mean_x(0.47, 0.47, 0.53))
        self.assertEqual("STRAIGHT", subject.direction_from_mean_x(0.53, 0.47, 0.53))
        self.assertEqual("RIGHT", subject.direction_from_mean_x(0.531, 0.47, 0.53))

    def test_template_points_preserve_horizon_order(self) -> None:
        spec = {"LEFT_x_norm": [0.47, 0.42, 0.36], "y_norm": [0.92, 0.86, 0.8]}
        self.assertEqual([(0.47, 0.92), (0.42, 0.86), (0.36, 0.8)],
                         subject.template_points(spec, "LEFT"))

    def test_source_dimensions_prefers_bound_metadata(self) -> None:
        self.assertEqual((426, 240), subject.source_dimensions({"video_width": 426, "video_height": 240}))


if __name__ == "__main__":
    unittest.main()
