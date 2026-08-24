import unittest

import cv2
import numpy as np

from .dense_boundary_observation import FINAL_SUPPORT_MINIMUM_PX, fit_dense_vertical_lines
from .pose_accumulation_observation import _field_support
from .two_view_observation import ImageLine, _normalised_line


class DenseBoundaryObservationTest(unittest.TestCase):
    def test_fragment_fusion_recovers_one_supported_boundary(self) -> None:
        support = np.zeros((96, 128), dtype=np.uint8)
        cv2.line(support, (40, 5), (41, 27), 1, 1)
        cv2.line(support, (41, 38), (42, 61), 1, 1)
        cv2.line(support, (42, 72), (43, 90), 1, 1)

        lines, diagnostics = fit_dense_vertical_lines(support.astype(bool))

        self.assertGreaterEqual(len(lines), 1)
        boundary = min(lines, key=lambda line: abs(line.x_at(48.0) - 41.5))
        self.assertGreaterEqual(boundary.support_length_px, FINAL_SUPPORT_MINIMUM_PX)
        self.assertGreaterEqual(boundary.segment_count, 2)
        self.assertGreater(diagnostics["raw_fragment_count"], 1)

    def test_short_dense_fragment_does_not_become_boundary(self) -> None:
        support = np.zeros((96, 128), dtype=np.uint8)
        cv2.line(support, (52, 40), (52, 45), 1, 1)

        lines, _ = fit_dense_vertical_lines(support.astype(bool))

        self.assertEqual(lines, [])

    def test_field_support_rewards_matching_vertical_hypothesis(self) -> None:
        distance = np.full((48, 64), 5.0, dtype=np.float32)
        orientation = np.zeros((48, 64), dtype=np.float32)
        distance[:, 30] = 0.2
        orientation[:, 30] = np.pi / 2
        line = ImageLine(tuple(_normalised_line(30, 0, 30, 47)), 1.0, 1)

        support, score = _field_support({"distance": distance, "orientation": orientation}, line)

        self.assertEqual(support, 48.0)
        self.assertGreater(score, 0.8)


if __name__ == "__main__":
    unittest.main()
