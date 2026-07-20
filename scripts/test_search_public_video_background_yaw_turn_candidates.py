import unittest

import numpy as np

import search_public_video_background_yaw_turn_candidates as subject


SPEC = {
    "background_roi_xyxy_norm": [0.1, 0.1, 0.9, 0.7],
    "minimum_absolute_median_dx_norm": 0.015,
    "minimum_horizontal_to_vertical_median_ratio": 1.5,
    "minimum_same_sign_dx_fraction": 0.7,
}


class BackgroundYawTurnCandidateTest(unittest.TestCase):
    def test_positive_background_motion_means_camera_left(self) -> None:
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[..., 0] = 6.0
        self.assertEqual("LEFT", subject.classify_flow(flow, SPEC)["direction"])

    def test_negative_background_motion_means_camera_right(self) -> None:
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[..., 0] = -6.0
        self.assertEqual("RIGHT", subject.classify_flow(flow, SPEC)["direction"])

    def test_vertical_motion_is_not_a_turn(self) -> None:
        flow = np.zeros((100, 200, 2), dtype=np.float32)
        flow[..., 0] = 6.0
        flow[..., 1] = 8.0
        self.assertEqual("NONE", subject.classify_flow(flow, SPEC)["direction"])


if __name__ == "__main__":
    unittest.main()
