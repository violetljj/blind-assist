import unittest

import numpy as np

from generate_scale_free_model_video_demo import ScaleFreeOperator, _bev_point


class ScaleFreeModelVideoDemoTest(unittest.TestCase):
    def test_relative_bev_preserves_image_left_and_right(self) -> None:
        origin = (200, 220)

        left = _bev_point(origin, 100, -30)
        center = _bev_point(origin, 100, 0)
        right = _bev_point(origin, 100, 30)

        self.assertLess(left[0], origin[0])
        self.assertEqual(center[0], origin[0])
        self.assertGreater(right[0], origin[0])
        self.assertLess(left[1], origin[1])
        self.assertLess(right[1], origin[1])

    def test_scores_are_invariant_to_global_positive_depth_scale(self) -> None:
        depth = np.full((120, 200), 4.0, dtype=np.float32)
        depth[36:108, 10:70] = 1.0
        depth[36:108, 130:190] = 1.0

        first = ScaleFreeOperator.raw_scores(depth)
        scaled = ScaleFreeOperator.raw_scores(depth * 3.5)

        self.assertEqual("VALID", first["status"])
        for name in ("left", "center", "right"):
            self.assertAlmostEqual(
                first["scores"][name], scaled["scores"][name], places=6
            )

    def test_requires_warmup_then_reports_only_relative_open_band(self) -> None:
        depth = np.full((120, 200), 4.0, dtype=np.float32)
        depth[36:108, 10:70] = 1.0
        depth[36:108, 130:190] = 1.0
        operator = ScaleFreeOperator()

        decisions = [operator.update(depth) for _ in range(5)]

        self.assertTrue(all(item["status"] == "UNKNOWN" for item in decisions[:4]))
        self.assertEqual("RELATIVELY_OPEN_CENTER", decisions[4]["status"])


if __name__ == "__main__":
    unittest.main()
